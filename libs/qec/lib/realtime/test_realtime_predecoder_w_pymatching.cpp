/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/*******************************************************************************
 * Hybrid Realtime Pipeline Test with Real ONNX Pre-Decoder
 *
 * Uses model1_d7_r7_unified_Z_batch1.onnx:
 *   Input:  all_measurements  [1, 72, 7]  INT32  (2016 bytes)
 *   Output: residual_detectors [1, 336]   INT32  (1344 bytes)
 *   Output: logical_frame      [1]        INT32  (4 bytes)
 *
 * Pipeline:
 * 1. Ring Buffer setup
 * 2. Dispatcher Kernel -> 4x AIPreDecoderService instances (GPU, TRT from ONNX)
 * 3. GPU -> CPU N-Deep Pinned Memory Queue handoff
 * 4. Dedicated Polling Thread -> 4-Worker PyMatching Thread Pool
 * 5. CPU Workers closing the transaction (Setting TX flags)
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <memory>
#include <cstring>
#include <unistd.h>
#include <random>
#include <mutex>

#include <cuda_runtime.h>

#ifndef CUDA_VERSION
#define CUDA_VERSION 13000
#endif
#include "cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include "cudaq/qec/realtime/ai_predecoder_service.h"
#include "cudaq/qec/utils/thread_pool.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/decoder.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

using namespace cudaq::qec;

// =============================================================================
// Configuration
// =============================================================================
constexpr size_t NUM_SLOTS = 64;
constexpr size_t SLOT_SIZE = 4096;          // Enough for RPC header + 2016-byte payload + response
constexpr int NUM_PREDECODERS = 4;
constexpr int QUEUE_DEPTH = 16;

// d=7, r=7 surface code Z-type model dimensions
constexpr int MEAS_QUBITS = 72;
constexpr int NUM_ROUNDS = 7;
constexpr int INPUT_ELEMENTS = MEAS_QUBITS * NUM_ROUNDS;   // 504 int32s = 2016 bytes
constexpr int RESIDUAL_DETECTORS = 336;                     // 336 int32s = 1344 bytes

constexpr std::uint32_t fnv1a_hash(std::string_view str) {
    std::uint32_t hash = 0x811c9dc5;
    for (char c : str) { hash ^= static_cast<std::uint32_t>(c); hash *= 0x01000193; }
    return hash;
}

struct SystemContext {
    volatile uint64_t* tx_flags_host = nullptr;
    uint8_t* rx_data_host = nullptr;
    size_t slot_size = SLOT_SIZE;
};
SystemContext g_sys_ctx;

// =============================================================================
// Thread Pool Worker (Real PyMatching MWPM Decoder)
// =============================================================================

// d=7 surface code: 24 Z stabilizers per spatial slice
constexpr int Z_STABILIZERS = 24;
constexpr int NUM_SPATIAL_SLICES = RESIDUAL_DETECTORS / Z_STABILIZERS; // 336/24 = 14

void pymatching_worker_task(PreDecoderJob job, AIPreDecoderService* predecoder,
                            cudaq::qec::decoder* pm_decoder, std::mutex* decode_mtx) {
    size_t num_detectors = predecoder->get_output_size() / sizeof(int32_t);
    const int32_t* residual = static_cast<const int32_t*>(job.inference_data);

    // Decode each spatial slice of Z-stabilizer detectors independently
    // using code-capacity PyMatching (H_z is [24 x 49])
    int total_corrections = 0;
    bool all_converged = true;

    for (int s = 0; s < NUM_SPATIAL_SLICES; ++s) {
        const int32_t* slice = residual + s * Z_STABILIZERS;
        std::vector<double> syndrome(Z_STABILIZERS);
        for (int i = 0; i < Z_STABILIZERS; ++i)
            syndrome[i] = static_cast<double>(slice[i]);

        cudaq::qec::decoder_result result;
        {
            std::lock_guard<std::mutex> lock(*decode_mtx);
            result = pm_decoder->decode(syndrome);
        }

        all_converged &= result.converged;
        for (auto v : result.result)
            if (v > 0.5f) total_corrections++;
    }

    // Write RPC Response
    struct __attribute__((packed)) DecodeResponse {
        int32_t total_corrections;
        int32_t converged;
    };
    DecodeResponse resp_data{total_corrections, all_converged ? 1 : 0};

    char* response_payload = (char*)job.ring_buffer_ptr + sizeof(cudaq::nvqlink::RPCResponse);
    std::memcpy(response_payload, &resp_data, sizeof(resp_data));

    auto* header = static_cast<cudaq::nvqlink::RPCResponse*>(job.ring_buffer_ptr);
    header->magic = cudaq::nvqlink::RPC_MAGIC_RESPONSE;
    header->status = 0;
    header->result_len = sizeof(resp_data);

    std::atomic_thread_fence(std::memory_order_release);

    size_t slot_idx = ((uint8_t*)job.ring_buffer_ptr - g_sys_ctx.rx_data_host) / g_sys_ctx.slot_size;
    predecoder->release_job(job.slot_idx);

    uint64_t rx_value = reinterpret_cast<uint64_t>(job.ring_buffer_ptr);
    g_sys_ctx.tx_flags_host[slot_idx] = rx_value;
}

// =============================================================================
// Incoming Polling Thread
// =============================================================================
void incoming_polling_loop(
    std::vector<std::unique_ptr<AIPreDecoderService>>& predecoders,
    cudaq::qec::utils::ThreadPool& thread_pool,
    cudaq::qec::decoder* pm_decoder,
    std::mutex& decode_mtx,
    std::atomic<bool>& stop_signal)
{
    PreDecoderJob job;
    while (!stop_signal.load(std::memory_order_relaxed)) {
        bool found_work = false;
        for (auto& predecoder : predecoders) {
            if (predecoder->poll_next_job(job)) {
                AIPreDecoderService* pd_ptr = predecoder.get();
                thread_pool.enqueue([job, pd_ptr, pm_decoder, &decode_mtx]() {
                    pymatching_worker_task(job, pd_ptr, pm_decoder, &decode_mtx);
                });
                found_work = true;
            }
        }
        if (!found_work) {
            QEC_CPU_RELAX();
        }
    }
}

// =============================================================================
// Generate Realistic Syndrome Data
// =============================================================================
void fill_measurement_payload(int32_t* payload, std::mt19937& rng,
                              double error_rate = 0.01) {
    std::bernoulli_distribution err_dist(error_rate);
    for (int i = 0; i < INPUT_ELEMENTS; ++i) {
        payload[i] = err_dist(rng) ? 1 : 0;
    }
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "--- Initializing Hybrid AI Realtime Pipeline (d=7 r=7 Z) ---\n";
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    std::string onnx_path = ONNX_MODEL_PATH;
    std::cout << "[Setup] Building TRT engines from: " << onnx_path << "\n";

    // Create PyMatching decoder from d=7 surface code Z parity check matrix
    std::cout << "[Setup] Creating PyMatching decoder (d=7 surface code, Z stabilizers)...\n";
    auto surface_code = cudaq::qec::get_code("surface_code", {{"distance", 7}});
    auto H_z = surface_code->get_parity_z();
    std::cout << "[Setup] H_z shape: [" << H_z.shape()[0] << " x " << H_z.shape()[1] << "]\n";

    cudaqx::heterogeneous_map pm_params;
    pm_params.insert("merge_strategy", std::string("smallest_weight"));
    auto pm_decoder = cudaq::qec::decoder::get("pymatching", H_z, pm_params);
    std::mutex decode_mtx;
    std::cout << "[Setup] PyMatching decoder ready.\n";

    // Allocate Ring Buffers
    void* tmp = nullptr;
    volatile uint64_t *rx_flags_host, *tx_flags_host;
    volatile uint64_t *rx_flags_dev, *tx_flags_dev;
    uint8_t *rx_data_host, *rx_data_dev;

    CUDA_CHECK(cudaHostAlloc(&tmp, NUM_SLOTS * sizeof(uint64_t), cudaHostAllocMapped));
    rx_flags_host = static_cast<volatile uint64_t*>(tmp);
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&rx_flags_dev, tmp, 0));

    CUDA_CHECK(cudaHostAlloc(&tmp, NUM_SLOTS * sizeof(uint64_t), cudaHostAllocMapped));
    tx_flags_host = static_cast<volatile uint64_t*>(tmp);
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&tx_flags_dev, tmp, 0));

    CUDA_CHECK(cudaHostAlloc(&rx_data_host, NUM_SLOTS * SLOT_SIZE, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&rx_data_dev, rx_data_host, 0));

    std::memset((void*)rx_flags_host, 0, NUM_SLOTS * sizeof(uint64_t));
    std::memset((void*)tx_flags_host, 0, NUM_SLOTS * sizeof(uint64_t));

    g_sys_ctx.tx_flags_host = tx_flags_host;
    g_sys_ctx.rx_data_host = rx_data_host;

    // Allocate Global Mailbox Bank & Control signals
    void** d_global_mailbox_bank;
    CUDA_CHECK(cudaMalloc(&d_global_mailbox_bank, NUM_PREDECODERS * sizeof(void*)));
    CUDA_CHECK(cudaMemset(d_global_mailbox_bank, 0, NUM_PREDECODERS * sizeof(void*)));

    int* shutdown_flag_host;
    CUDA_CHECK(cudaHostAlloc(&shutdown_flag_host, sizeof(int), cudaHostAllocMapped));
    *shutdown_flag_host = 0;
    int* d_shutdown_flag;
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_shutdown_flag, shutdown_flag_host, 0));

    uint64_t* d_stats;
    CUDA_CHECK(cudaMalloc(&d_stats, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(uint64_t)));

    // Initialize 4 AIPreDecoder Instances from ONNX
    std::cout << "[Setup] Capturing 4x AIPreDecoder Graphs (ONNX -> TRT)...\n";
    cudaStream_t capture_stream;
    CUDA_CHECK(cudaStreamCreate(&capture_stream));

    std::vector<std::unique_ptr<AIPreDecoderService>> predecoders;
    std::vector<cudaq_function_entry_t> function_entries(NUM_PREDECODERS);

    for (int i = 0; i < NUM_PREDECODERS; ++i) {
        void** my_mailbox = d_global_mailbox_bank + i;
        auto pd = std::make_unique<AIPreDecoderService>(onnx_path, my_mailbox, QUEUE_DEPTH);

        std::cout << "[Setup] Decoder " << i
                  << ": input_size=" << pd->get_input_size()
                  << " output_size=" << pd->get_output_size() << "\n";

        pd->capture_graph(capture_stream);

        cudaGraphExec_t gexec = pd->get_executable_graph();
        std::string func_name = "predecode_target_" + std::to_string(i);
        function_entries[i].function_id = fnv1a_hash(func_name);
        function_entries[i].dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
        function_entries[i].handler.graph_exec = gexec;
        function_entries[i].mailbox_idx = i;
        function_entries[i].d_queue_idx = pd->get_device_queue_idx();
        function_entries[i].d_ready_flags = pd->get_device_ready_flags();
        function_entries[i].d_inflight_flag = pd->get_device_inflight_flag();

        predecoders.push_back(std::move(pd));
    }
    int actual_func_count = NUM_PREDECODERS;

    cudaq_function_entry_t* d_function_entries;
    CUDA_CHECK(cudaMalloc(&d_function_entries, actual_func_count * sizeof(cudaq_function_entry_t)));
    CUDA_CHECK(cudaMemcpy(d_function_entries, function_entries.data(),
               actual_func_count * sizeof(cudaq_function_entry_t), cudaMemcpyHostToDevice));

    // Start GPU Dispatcher
    std::cout << "[Setup] Launching Dispatcher Kernel...\n";
    cudaq_dispatch_graph_context* dispatch_ctx = nullptr;
    CUDA_CHECK(cudaq_create_dispatch_graph_regular(
        rx_flags_dev, tx_flags_dev, d_function_entries, actual_func_count,
        d_global_mailbox_bank, d_shutdown_flag, d_stats, NUM_SLOTS, 1, 32, capture_stream, &dispatch_ctx
    ));
    CUDA_CHECK(cudaq_launch_dispatch_graph(dispatch_ctx, capture_stream));

    // Start CPU Infrastructure
    std::cout << "[Setup] Booting Thread Pool & Polling Loop...\n";
    cudaq::qec::utils::ThreadPool pymatching_pool(4);
    std::atomic<bool> system_stop{false};

    cudaq::qec::decoder* pm_raw = pm_decoder.get();
    std::thread incoming_thread([&]() {
        incoming_polling_loop(predecoders, pymatching_pool, pm_raw, decode_mtx, system_stop);
    });

    // =========================================================================
    // Test Stimulus: Fire requests in batches of NUM_PREDECODERS.
    // The dispatcher advances its slot pointer linearly and only retries
    // while rx_value != 0, so we must wait for each batch to complete
    // before firing the next to avoid stranding un-dispatched slots.
    // =========================================================================
    constexpr int TOTAL_REQUESTS = 20;
    constexpr int BATCH_SIZE = NUM_PREDECODERS;
    std::cout << "\n[Test] Firing " << TOTAL_REQUESTS
              << " syndromes in batches of " << BATCH_SIZE
              << " (d=7, r=7, error_rate=0.01)...\n";

    std::mt19937 rng(42);
    const size_t payload_bytes = INPUT_ELEMENTS * sizeof(int32_t);
    int requests_sent = 0;
    int responses_received = 0;

    for (int batch_start = 0; batch_start < TOTAL_REQUESTS; batch_start += BATCH_SIZE) {
        int batch_end = std::min(batch_start + BATCH_SIZE, TOTAL_REQUESTS);
        int batch_count = batch_end - batch_start;

        // Fire one batch
        for (int i = batch_start; i < batch_end; ++i) {
            int target_decoder = i % NUM_PREDECODERS;
            std::string target_func = "predecode_target_" + std::to_string(target_decoder);

            int slot = i % NUM_SLOTS;
            while (rx_flags_host[slot] != 0) usleep(10);

            uint8_t* slot_data = rx_data_host + (slot * SLOT_SIZE);
            auto* header = reinterpret_cast<cudaq::nvqlink::RPCHeader*>(slot_data);
            header->magic = cudaq::nvqlink::RPC_MAGIC_REQUEST;
            header->function_id = fnv1a_hash(target_func);
            header->arg_len = static_cast<uint32_t>(payload_bytes);

            int32_t* payload = reinterpret_cast<int32_t*>(slot_data + sizeof(cudaq::nvqlink::RPCHeader));
            fill_measurement_payload(payload, rng, 0.01);

            __sync_synchronize();
            rx_flags_host[slot] = reinterpret_cast<uint64_t>(slot_data);
            requests_sent++;
        }

        // Wait for this batch to complete
        for (int i = batch_start; i < batch_end; ++i) {
            int slot = i % NUM_SLOTS;

            int timeout = 10000;
            while (tx_flags_host[slot] == 0 && timeout-- > 0) usleep(1000);

            uint64_t tv = tx_flags_host[slot];
            if (tv != 0 && (tv >> 48) == 0xDEAD) {
                int cuda_err = (int)(tv & 0xFFFF);
                std::cerr << "  [FAIL] Slot " << slot << " cudaGraphLaunch error "
                          << cuda_err << " (" << cudaGetErrorString((cudaError_t)cuda_err) << ")\n";
            } else if (tv != 0) {
                responses_received++;
                uint8_t* slot_data = rx_data_host + (slot * SLOT_SIZE);
                int32_t corrections = 0, converged = 0;
                std::memcpy(&corrections,
                            slot_data + sizeof(cudaq::nvqlink::RPCResponse),
                            sizeof(int32_t));
                std::memcpy(&converged,
                            slot_data + sizeof(cudaq::nvqlink::RPCResponse) + sizeof(int32_t),
                            sizeof(int32_t));
                std::cout << "  -> Slot " << slot << ": OK, corrections=" << corrections
                          << " converged=" << (converged ? "yes" : "no") << "\n";
            } else {
                std::cerr << "  [FAIL] Timeout waiting for slot " << slot << "\n";
            }

            tx_flags_host[slot] = 0;
        }
    }

    std::cout << "\n[Result] Processed " << responses_received << "/" << requests_sent
              << " requests successfully.\n";

    // Teardown
    std::cout << "[Teardown] Shutting down...\n";
    *shutdown_flag_host = 1;
    __sync_synchronize();
    system_stop = true;

    incoming_thread.join();
    CUDA_CHECK(cudaStreamSynchronize(capture_stream));

    uint64_t dispatched_packets = 0;
    CUDA_CHECK(cudaMemcpy(&dispatched_packets, d_stats, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    std::cout << "[Stats] Dispatcher processed " << dispatched_packets << " packets.\n";

    CUDA_CHECK(cudaq_destroy_dispatch_graph(dispatch_ctx));

    cudaFreeHost((void*)rx_flags_host);
    cudaFreeHost((void*)tx_flags_host);
    cudaFreeHost(rx_data_host);
    cudaFreeHost(shutdown_flag_host);
    cudaFree(d_global_mailbox_bank);
    cudaFree(d_stats);
    cudaFree(d_function_entries);
    cudaStreamDestroy(capture_stream);

    std::cout << "Done.\n";
    return 0;
}
