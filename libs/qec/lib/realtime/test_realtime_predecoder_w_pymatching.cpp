/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/*******************************************************************************
 * Hybrid Realtime Pipeline Test with Real ONNX Pre-Decoder + PyMatching
 *
 * Supports multiple surface code configurations:
 *
 *   d=7  r=7  (model1_d7_r7_unified_Z_batch1.onnx)
 *     Input:  all_measurements  [1, 72, 7]   INT32  (2016 bytes)
 *     Output: residual_detectors [1, 336]    INT32  (1344 bytes)
 *     Output: logical_frame      [1]         INT32  (4 bytes)
 *
 *   d=13 r=13 (model1_d13_r13_unified_Z_batch1.onnx)
 *     Input:  all_measurements  [1, 252, 13]  INT32  (13104 bytes)
 *     Output: residual_detectors [1, 2184]   INT32  (8736 bytes)
 *     Output: logical_frame      [1]         INT32  (4 bytes)
 *
 *   d=21 r=21 (model1_d21_r21_unified_Z_batch1.onnx)
 *     Input:  all_measurements  [1, 660, 21]  INT32  (55440 bytes)
 *     Output: residual_detectors [1, 9240]   INT32  (36960 bytes)
 *     Output: logical_frame      [1]         INT32  (4 bytes)
 *
 *   d=31 r=31 (model1_d31_r31_unified_Z_batch1.onnx)
 *     Input:  all_measurements  [1, 1440, 31] INT32  (178560 bytes)
 *     Output: residual_detectors [1, 29760]  INT32  (119040 bytes)
 *     Output: logical_frame      [1]         INT32  (4 bytes)
 *
 * Pipeline:
 * 1. Ring Buffer setup
 * 2. Dispatcher Kernel -> Nx AIPreDecoderService instances (GPU, TRT from ONNX)
 * 3. GPU -> CPU N-Deep Pinned Memory Queue handoff
 * 4. Dedicated Polling Thread -> Worker PyMatching Thread Pool
 * 5. CPU Workers closing the transaction (Setting TX flags)
 *
 * Usage: test_realtime_predecoder_w_pymatching [d7|d13|d21|d31]
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
#include <string>
#include <iomanip>

#include <cuda_runtime.h>

#ifndef CUDA_VERSION
#define CUDA_VERSION 13000
#endif
#include "cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include "cudaq/qec/realtime/ai_predecoder_service.h"
#include "cudaq/qec/utils/thread_pool.h"
#include "cudaq/qec/utils/pipeline_benchmarks.h"
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
// Pipeline Configuration
// =============================================================================

constexpr size_t NUM_SLOTS = 64;

struct PipelineConfig {
    std::string label;
    int distance;
    int num_rounds;
    int meas_qubits;          // ONNX input shape[1]
    int residual_detectors;   // ONNX output dim
    std::string onnx_filename;
    size_t slot_size;         // must fit RPCHeader + input payload
    int total_requests;
    int num_predecoders;
    int queue_depth;
    int num_workers;

    int input_elements() const { return meas_qubits * num_rounds; }
    size_t input_bytes() const { return input_elements() * sizeof(int32_t); }

    std::string onnx_path() const {
        return std::string(ONNX_MODEL_DIR) + "/" + onnx_filename;
    }

    static PipelineConfig d7_r7() {
        return {
            "d7_r7_Z",
            /*distance=*/7,
            /*num_rounds=*/7,
            /*meas_qubits=*/72,
            /*residual_detectors=*/336,
            "model1_d7_r7_unified_Z_batch1.onnx",
            /*slot_size=*/4096,
            /*total_requests=*/100,
            /*num_predecoders=*/4,
            /*queue_depth=*/16,
            /*num_workers=*/4
        };
    }

    static PipelineConfig d13_r13() {
        return {
            "d13_r13_Z",
            /*distance=*/13,
            /*num_rounds=*/13,
            /*meas_qubits=*/252,
            /*residual_detectors=*/2184,
            "model1_d13_r13_unified_Z_batch1.onnx",
            /*slot_size=*/16384,
            /*total_requests=*/100,
            /*num_predecoders=*/4,
            /*queue_depth=*/16,
            /*num_workers=*/4
        };
    }

    static PipelineConfig d21_r21() {
        return {
            "d21_r21_Z",
            /*distance=*/21,
            /*num_rounds=*/21,
            /*meas_qubits=*/660,
            /*residual_detectors=*/9240,
            "model1_d21_r21_unified_X_batch1.onnx",
            /*slot_size=*/65536,
            /*total_requests=*/100,
            /*num_predecoders=*/4,
            /*queue_depth=*/16,
            /*num_workers=*/4
        };
    }

    static PipelineConfig d31_r31() {
        return {
            "d31_r31_Z",
            /*distance=*/31,
            /*num_rounds=*/31,
            /*meas_qubits=*/1440,
            /*residual_detectors=*/29760,
            "model1_d31_r31_unified_Z_batch1.onnx",
            /*slot_size=*/262144,
            /*total_requests=*/100,
            /*num_predecoders=*/4,
            /*queue_depth=*/16,
            /*num_workers=*/4
        };
    }
};

// Runtime decoder state populated during setup
struct DecoderContext {
    std::unique_ptr<cudaq::qec::decoder> pm_decoder;
    std::mutex decode_mtx;
    int z_stabilizers = 0;
    int spatial_slices = 0;

    // Per-worker timing accumulators (protected by decode_mtx)
    std::atomic<int64_t> total_decode_us{0};
    std::atomic<int64_t> total_worker_us{0};
    std::atomic<int> decode_count{0};
};

constexpr std::uint32_t fnv1a_hash(std::string_view str) {
    std::uint32_t hash = 0x811c9dc5;
    for (char c : str) { hash ^= static_cast<std::uint32_t>(c); hash *= 0x01000193; }
    return hash;
}

struct SystemContext {
    volatile uint64_t* tx_flags_host = nullptr;
    uint8_t* rx_data_host = nullptr;
    size_t slot_size = 0;
};
SystemContext g_sys_ctx;

// =============================================================================
// Thread Pool Worker (Real PyMatching MWPM Decoder)
// =============================================================================

struct __attribute__((packed)) DecodeResponse {
    int32_t total_corrections;
    int32_t converged;
};

void pymatching_worker_task(PreDecoderJob job, AIPreDecoderService* predecoder,
                            DecoderContext* ctx) {
    using hrclock = std::chrono::high_resolution_clock;
    auto worker_start = hrclock::now();

    const int32_t* residual = static_cast<const int32_t*>(job.inference_data);

    int total_corrections = 0;
    bool all_converged = true;

    auto decode_start = hrclock::now();
    for (int s = 0; s < ctx->spatial_slices; ++s) {
        const int32_t* slice = residual + s * ctx->z_stabilizers;
        std::vector<double> syndrome(ctx->z_stabilizers);
        for (int i = 0; i < ctx->z_stabilizers; ++i)
            syndrome[i] = static_cast<double>(slice[i]);

        cudaq::qec::decoder_result result;
        {
            std::lock_guard<std::mutex> lock(ctx->decode_mtx);
            result = ctx->pm_decoder->decode(syndrome);
        }

        all_converged &= result.converged;
        for (auto v : result.result)
            if (v > 0.5) total_corrections++;
    }
    auto decode_end = hrclock::now();

    DecodeResponse resp_data{total_corrections, all_converged ? 1 : 0};

    char* response_payload = (char*)job.ring_buffer_ptr + sizeof(cudaq::nvqlink::RPCResponse);
    std::memcpy(response_payload, &resp_data, sizeof(resp_data));

    auto* header = static_cast<cudaq::nvqlink::RPCResponse*>(job.ring_buffer_ptr);
    header->magic = cudaq::nvqlink::RPC_MAGIC_RESPONSE;
    header->status = 0;
    header->result_len = sizeof(resp_data);

    std::atomic_thread_fence(std::memory_order_release);

    auto worker_end = hrclock::now();
    auto decode_us = std::chrono::duration_cast<std::chrono::microseconds>(
        decode_end - decode_start).count();
    auto worker_us = std::chrono::duration_cast<std::chrono::microseconds>(
        worker_end - worker_start).count();
    ctx->total_decode_us.fetch_add(decode_us, std::memory_order_relaxed);
    ctx->total_worker_us.fetch_add(worker_us, std::memory_order_relaxed);
    ctx->decode_count.fetch_add(1, std::memory_order_relaxed);

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
    DecoderContext* ctx,
    std::atomic<bool>& stop_signal)
{
    PreDecoderJob job;
    while (!stop_signal.load(std::memory_order_relaxed)) {
        bool found_work = false;
        for (auto& predecoder : predecoders) {
            if (predecoder->poll_next_job(job)) {
                AIPreDecoderService* pd_ptr = predecoder.get();
                thread_pool.enqueue([job, pd_ptr, ctx]() {
                    pymatching_worker_task(job, pd_ptr, ctx);
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
void fill_measurement_payload(int32_t* payload, int input_elements,
                              std::mt19937& rng, double error_rate = 0.01) {
    std::bernoulli_distribution err_dist(error_rate);
    for (int i = 0; i < input_elements; ++i) {
        payload[i] = err_dist(rng) ? 1 : 0;
    }
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
    // Select configuration
    std::string config_name = "d7";
    if (argc > 1)
        config_name = argv[1];

    PipelineConfig config;
    if (config_name == "d7") {
        config = PipelineConfig::d7_r7();
    } else if (config_name == "d13") {
        config = PipelineConfig::d13_r13();
    } else if (config_name == "d21") {
        config = PipelineConfig::d21_r21();
    } else if (config_name == "d31") {
        config = PipelineConfig::d31_r31();
    } else {
        std::cerr << "Usage: " << argv[0] << " [d7|d13|d21|d31]\n"
                  << "  d7  - distance 7, 7 rounds (default)\n"
                  << "  d13 - distance 13, 13 rounds\n"
                  << "  d21 - distance 21, 21 rounds\n"
                  << "  d31 - distance 31, 31 rounds\n";
        return 1;
    }

    std::cout << "--- Initializing Hybrid AI Realtime Pipeline ("
              << config.label << ") ---\n";
    std::cout << "[Config] distance=" << config.distance
              << " rounds=" << config.num_rounds
              << " meas_qubits=" << config.meas_qubits
              << " residual_detectors=" << config.residual_detectors
              << " input_bytes=" << config.input_bytes()
              << " slot_size=" << config.slot_size << "\n";

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    std::string onnx_path = config.onnx_path();
    std::cout << "[Setup] Building TRT engines from: " << onnx_path << "\n";

    // Create PyMatching decoder from surface code Z parity check matrix
    std::cout << "[Setup] Creating PyMatching decoder (d=" << config.distance
              << " surface code, Z stabilizers)...\n";
    auto surface_code = cudaq::qec::get_code("surface_code",
                                              {{"distance", config.distance}});
    auto H_z = surface_code->get_parity_z();

    DecoderContext decoder_ctx;
    decoder_ctx.z_stabilizers = static_cast<int>(H_z.shape()[0]);
    decoder_ctx.spatial_slices = config.residual_detectors / decoder_ctx.z_stabilizers;
    std::cout << "[Setup] H_z shape: [" << H_z.shape()[0] << " x "
              << H_z.shape()[1] << "]"
              << "  z_stabilizers=" << decoder_ctx.z_stabilizers
              << "  spatial_slices=" << decoder_ctx.spatial_slices << "\n";

    cudaqx::heterogeneous_map pm_params;
    pm_params.insert("merge_strategy", std::string("smallest_weight"));
    decoder_ctx.pm_decoder = cudaq::qec::decoder::get("pymatching", H_z, pm_params);
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

    CUDA_CHECK(cudaHostAlloc(&rx_data_host, NUM_SLOTS * config.slot_size, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&rx_data_dev, rx_data_host, 0));

    std::memset((void*)rx_flags_host, 0, NUM_SLOTS * sizeof(uint64_t));
    std::memset((void*)tx_flags_host, 0, NUM_SLOTS * sizeof(uint64_t));

    g_sys_ctx.tx_flags_host = tx_flags_host;
    g_sys_ctx.rx_data_host = rx_data_host;
    g_sys_ctx.slot_size = config.slot_size;

    // Allocate Global Mailbox Bank & Control signals
    void** d_global_mailbox_bank;
    CUDA_CHECK(cudaMalloc(&d_global_mailbox_bank, config.num_predecoders * sizeof(void*)));
    CUDA_CHECK(cudaMemset(d_global_mailbox_bank, 0, config.num_predecoders * sizeof(void*)));

    int* shutdown_flag_host;
    CUDA_CHECK(cudaHostAlloc(&shutdown_flag_host, sizeof(int), cudaHostAllocMapped));
    *shutdown_flag_host = 0;
    int* d_shutdown_flag;
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_shutdown_flag, shutdown_flag_host, 0));

    uint64_t* d_stats;
    CUDA_CHECK(cudaMalloc(&d_stats, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(uint64_t)));

    // Initialize AIPreDecoder Instances from ONNX
    std::cout << "[Setup] Capturing " << config.num_predecoders
              << "x AIPreDecoder Graphs (ONNX -> TRT)...\n";
    cudaStream_t capture_stream;
    CUDA_CHECK(cudaStreamCreate(&capture_stream));

    std::vector<std::unique_ptr<AIPreDecoderService>> predecoders;
    std::vector<cudaq_function_entry_t> function_entries(config.num_predecoders);

    for (int i = 0; i < config.num_predecoders; ++i) {
        void** my_mailbox = d_global_mailbox_bank + i;
        auto pd = std::make_unique<AIPreDecoderService>(onnx_path, my_mailbox,
                                                         config.queue_depth);

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

    cudaq_function_entry_t* d_function_entries;
    CUDA_CHECK(cudaMalloc(&d_function_entries,
               config.num_predecoders * sizeof(cudaq_function_entry_t)));
    CUDA_CHECK(cudaMemcpy(d_function_entries, function_entries.data(),
               config.num_predecoders * sizeof(cudaq_function_entry_t),
               cudaMemcpyHostToDevice));

    // Start GPU Dispatcher
    std::cout << "[Setup] Launching Dispatcher Kernel...\n";
    cudaq_dispatch_graph_context* dispatch_ctx = nullptr;
    CUDA_CHECK(cudaq_create_dispatch_graph_regular(
        rx_flags_dev, tx_flags_dev, d_function_entries, config.num_predecoders,
        d_global_mailbox_bank, d_shutdown_flag, d_stats, NUM_SLOTS, 1, 32,
        capture_stream, &dispatch_ctx
    ));
    CUDA_CHECK(cudaq_launch_dispatch_graph(dispatch_ctx, capture_stream));

    // Start CPU Infrastructure
    std::cout << "[Setup] Booting Thread Pool (" << config.num_workers
              << " workers) & Polling Loop...\n";
    cudaq::qec::utils::ThreadPool pymatching_pool(config.num_workers);
    std::atomic<bool> system_stop{false};

    std::thread incoming_thread([&]() {
        incoming_polling_loop(predecoders, pymatching_pool, &decoder_ctx,
                              system_stop);
    });

    // =========================================================================
    // Test Stimulus: Fire requests in batches of num_predecoders.
    // The dispatcher advances its slot pointer linearly and only retries
    // while rx_value != 0, so we must wait for each batch to complete
    // before firing the next to avoid stranding un-dispatched slots.
    // =========================================================================
    const int batch_size = config.num_predecoders;
    std::cout << "\n[Test] Firing " << config.total_requests
              << " syndromes in batches of " << batch_size
              << " (" << config.label << ", error_rate=0.01)...\n";

    cudaq::qec::utils::PipelineBenchmark bench(config.label,
                                                config.total_requests);

    std::mt19937 rng(42);
    const size_t payload_bytes = config.input_bytes();
    int requests_sent = 0;
    int responses_received = 0;

    bench.start();

    for (int batch_start = 0; batch_start < config.total_requests;
         batch_start += batch_size) {
        int batch_end = std::min(batch_start + batch_size, config.total_requests);

        // Fire one batch
        for (int i = batch_start; i < batch_end; ++i) {
            int target_decoder = i % config.num_predecoders;
            std::string target_func = "predecode_target_" + std::to_string(target_decoder);

            int slot = i % (int)NUM_SLOTS;
            while (rx_flags_host[slot] != 0) usleep(10);

            uint8_t* slot_data = rx_data_host + (slot * config.slot_size);
            auto* header = reinterpret_cast<cudaq::nvqlink::RPCHeader*>(slot_data);
            header->magic = cudaq::nvqlink::RPC_MAGIC_REQUEST;
            header->function_id = fnv1a_hash(target_func);
            header->arg_len = static_cast<uint32_t>(payload_bytes);

            int32_t* payload = reinterpret_cast<int32_t*>(
                slot_data + sizeof(cudaq::nvqlink::RPCHeader));
            fill_measurement_payload(payload, config.input_elements(), rng, 0.01);

            __sync_synchronize();
            bench.mark_submit(i);
            rx_flags_host[slot] = reinterpret_cast<uint64_t>(slot_data);
            requests_sent++;
        }

        // Wait for this batch to complete (spin-wait for accurate latency)
        for (int i = batch_start; i < batch_end; ++i) {
            int slot = i % (int)NUM_SLOTS;

            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (tx_flags_host[slot] == 0) {
                if (std::chrono::steady_clock::now() > deadline) break;
                QEC_CPU_RELAX();
            }

            uint64_t tv = tx_flags_host[slot];
            if (tv != 0 && (tv >> 48) == 0xDEAD) {
                int cuda_err = (int)(tv & 0xFFFF);
                std::cerr << "  [FAIL] Slot " << slot << " cudaGraphLaunch error "
                          << cuda_err << " (" << cudaGetErrorString((cudaError_t)cuda_err) << ")\n";
            } else if (tv != 0) {
                bench.mark_complete(i);
                responses_received++;
                uint8_t* slot_data = rx_data_host + (slot * config.slot_size);
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

    bench.stop();

    std::cout << "\n[Result] Processed " << responses_received << "/" << requests_sent
              << " requests successfully.\n";

    bench.report();

    // Worker timing breakdown
    int n_decoded = decoder_ctx.decode_count.load();
    if (n_decoded > 0) {
        double avg_decode = (double)decoder_ctx.total_decode_us.load() / n_decoded;
        double avg_worker = (double)decoder_ctx.total_worker_us.load() / n_decoded;
        double avg_overhead = avg_worker - avg_decode;
        auto stats = bench.compute_stats();
        double avg_pipeline_overhead = stats.mean_us - avg_worker;

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "\n  Worker Timing Breakdown (avg over " << n_decoded << " requests):\n";
        std::cout << "    PyMatching decode:   " << std::setw(8) << avg_decode
                  << " us  (" << std::setw(4) << (100.0 * avg_decode / stats.mean_us) << "%)\n";
        std::cout << "    Worker overhead:      " << std::setw(8) << avg_overhead
                  << " us  (" << std::setw(4) << (100.0 * avg_overhead / stats.mean_us) << "%)\n";
        std::cout << "    GPU+dispatch+poll:    " << std::setw(8) << avg_pipeline_overhead
                  << " us  (" << std::setw(4) << (100.0 * avg_pipeline_overhead / stats.mean_us) << "%)\n";
        std::cout << "    Total end-to-end:     " << std::setw(8) << stats.mean_us << " us\n";
        std::cout << "    Per-round (/" << config.num_rounds << "):     "
                  << std::setw(8) << (stats.mean_us / config.num_rounds) << " us/round\n";
    }

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
