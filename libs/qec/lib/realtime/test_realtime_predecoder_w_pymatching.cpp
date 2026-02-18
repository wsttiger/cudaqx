/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/*******************************************************************************
 * Standalone Hybrid Realtime Pipeline Test
 * Demonstrates:
 * 1. Ring Buffer setup
 * 2. Dispatcher Kernel -> 4x AIPreDecoderService instances (GPU)
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
#include <fstream>

#include <cuda_runtime.h>
#include <NvInfer.h>

// Ensure graph-based dispatch API is visible (guarded by CUDA_VERSION in cudaq_realtime.h)
#ifndef CUDA_VERSION
#define CUDA_VERSION 13000
#endif
#include "cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include "cudaq/qec/realtime/ai_predecoder_service.h"
#include "cudaq/qec/utils/thread_pool.h"

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
// Configuration & Globals
// =============================================================================
constexpr size_t NUM_SLOTS = 64;
constexpr size_t SLOT_SIZE = 256;
constexpr int NUM_PREDECODERS = 4;
constexpr int QUEUE_DEPTH = 16;
constexpr int SYNDROME_FLOATS = 16; // 64 bytes

// Helper to generate Function IDs
constexpr std::uint32_t fnv1a_hash(std::string_view str) {
    std::uint32_t hash = 0x811c9dc5;
    for (char c : str) { hash ^= static_cast<std::uint32_t>(c); hash *= 0x01000193; }
    return hash;
}

// Global context to pass to workers without massive argument lists
struct SystemContext {
    volatile uint64_t* tx_flags_host = nullptr;
    uint8_t* rx_data_host = nullptr;
    size_t slot_size = SLOT_SIZE;
};
SystemContext g_sys_ctx;

// =============================================================================
// 1. Thread Pool Worker (PyMatching Simulation)
// =============================================================================
void pymatching_worker_task(PreDecoderJob job, AIPreDecoderService* predecoder) {
    // A. "PyMatching" CPU Algorithm
    // Convert 16 floats (logits) back to 16 bits
    size_t num_elements = predecoder->get_output_size() / sizeof(float);
    std::vector<uint8_t> final_corrections(num_elements);

    // Simulation placeholder: in production this would run the PyMatching decoder.
    for (size_t i = 0; i < num_elements; ++i) {
        final_corrections[i] = (job.inference_data[i] > 0.5f) ? 1 : 0;
    }

    // B. Write RPC Response
    char* response_payload = (char*)job.ring_buffer_ptr + sizeof(cudaq::nvqlink::RPCResponse);
    std::memcpy(response_payload, final_corrections.data(), final_corrections.size());

    auto* header = static_cast<cudaq::nvqlink::RPCResponse*>(job.ring_buffer_ptr);
    header->magic = cudaq::nvqlink::RPC_MAGIC_RESPONSE;
    header->status = 0;
    header->result_len = static_cast<uint32_t>(final_corrections.size());

    std::atomic_thread_fence(std::memory_order_release);

    // C. Calculate the original Ring Buffer Slot Index
    size_t slot_idx = ((uint8_t*)job.ring_buffer_ptr - g_sys_ctx.rx_data_host) / g_sys_ctx.slot_size;

    // D. Release GPU Queue Slot
    predecoder->release_job(job.slot_idx);

    // E. Acknowledge to FPGA
    // Reconstruct the original rx_value (which is just the pointer cast to uint64_t)
    uint64_t rx_value = reinterpret_cast<uint64_t>(job.ring_buffer_ptr);
    g_sys_ctx.tx_flags_host[slot_idx] = rx_value;
}

// =============================================================================
// 2. Incoming Polling Thread
// =============================================================================
void incoming_polling_loop(
    std::vector<std::unique_ptr<AIPreDecoderService>>& predecoders,
    cudaq::qec::utils::ThreadPool& thread_pool,
    std::atomic<bool>& stop_signal)
{
    PreDecoderJob job;
    while (!stop_signal.load(std::memory_order_relaxed)) {
        bool found_work = false;

        // Round-robin poll across all 4 PreDecoder instances
        for (auto& predecoder : predecoders) {
            if (predecoder->poll_next_job(job)) {
                // Enqueue the job. Capture raw pointer to specific predecoder instance.
                AIPreDecoderService* pd_ptr = predecoder.get();
                thread_pool.enqueue([job, pd_ptr]() {
                    pymatching_worker_task(job, pd_ptr);
                });
                found_work = true;
            }
        }

        // If all 4 queues were empty, yield the pipeline
        if (!found_work) {
            QEC_CPU_RELAX();
        }
    }
}

// =============================================================================
// 3. Helper: Dummy TRT Engine Generator
// =============================================================================
void create_dummy_engine(const std::string& filepath) {
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {}
    } logger;

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

    // Identity network: 16 floats in, 16 floats out
    auto input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {SYNDROME_FLOATS}});
    auto identity = network->addIdentity(*input);
    identity->getOutput(0)->setName("output");
    network->markOutput(*identity->getOutput(0));

    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));

    std::ofstream file(filepath, std::ios::binary);
    file.write(static_cast<const char*>(plan->data()), plan->size());
}

// =============================================================================
// 4. Main Application
// =============================================================================
int main() {
    std::cout << "--- Initializing Hybrid AI Realtime Pipeline ---\n";
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    // A. Generate Dummy Model
    std::string engine_path = "predecoder_dummy.engine";
    create_dummy_engine(engine_path);

    // B. Allocate Ring Buffers
    void* tmp = nullptr;

    volatile uint64_t *rx_flags_host, *tx_flags_host;
    volatile uint64_t *rx_flags_dev, *tx_flags_dev;
    uint8_t *rx_data_host;
    uint8_t *rx_data_dev;

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

    // C. Allocate Global Mailbox Bank & Control signals
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

    // D. Initialize the 4 AIPreDecoder Instances
    std::cout << "[Setup] Capturing 4x AIPreDecoder Graphs...\n";
    cudaStream_t capture_stream;
    CUDA_CHECK(cudaStreamCreate(&capture_stream));

    std::vector<std::unique_ptr<AIPreDecoderService>> predecoders;
    std::vector<cudaq_function_entry_t> function_entries(NUM_PREDECODERS);

    for (int i = 0; i < NUM_PREDECODERS; ++i) {
        void** my_mailbox = d_global_mailbox_bank + i;
        auto pd = std::make_unique<AIPreDecoderService>(engine_path, my_mailbox, QUEUE_DEPTH);
        pd->capture_graph(capture_stream);

        cudaGraphExec_t gexec = pd->get_executable_graph();
        std::cout << "[Setup] Decoder " << i << ": graph_exec=" << gexec << "\n";

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

    // Print struct layout for host/device verification
    std::cout << "[Debug] sizeof(cudaq_function_entry_t) = " << sizeof(cudaq_function_entry_t) << "\n";
    std::cout << "[Debug] offsetof handler       = " << offsetof(cudaq_function_entry_t, handler) << "\n";
    std::cout << "[Debug] offsetof function_id    = " << offsetof(cudaq_function_entry_t, function_id) << "\n";
    std::cout << "[Debug] offsetof dispatch_mode  = " << offsetof(cudaq_function_entry_t, dispatch_mode) << "\n";
    std::cout << "[Debug] offsetof schema         = " << offsetof(cudaq_function_entry_t, schema) << "\n";
    std::cout << "[Debug] offsetof mailbox_idx    = " << offsetof(cudaq_function_entry_t, mailbox_idx) << "\n";
    std::cout << "[Debug] offsetof d_queue_idx    = " << offsetof(cudaq_function_entry_t, d_queue_idx) << "\n";
    std::cout << "[Debug] offsetof d_ready_flags  = " << offsetof(cudaq_function_entry_t, d_ready_flags) << "\n";
    std::cout << "[Debug] offsetof d_inflight_flag= " << offsetof(cudaq_function_entry_t, d_inflight_flag) << "\n";
    std::cout << "[Debug] sizeof(cudaq_handler_schema_t) = " << sizeof(cudaq_handler_schema_t) << "\n";

    cudaq_function_entry_t* d_function_entries;
    CUDA_CHECK(cudaMalloc(&d_function_entries, actual_func_count * sizeof(cudaq_function_entry_t)));
    CUDA_CHECK(cudaMemcpy(d_function_entries, function_entries.data(),
               actual_func_count * sizeof(cudaq_function_entry_t), cudaMemcpyHostToDevice));

    // E. Start GPU Dispatcher
    std::cout << "[Setup] Launching Dispatcher Kernel...\n";
    cudaq_dispatch_graph_context* dispatch_ctx = nullptr;
    CUDA_CHECK(cudaq_create_dispatch_graph_regular(
        rx_flags_dev, tx_flags_dev, d_function_entries, actual_func_count,
        d_global_mailbox_bank, d_shutdown_flag, d_stats, NUM_SLOTS, 1, 32, capture_stream, &dispatch_ctx
    ));
    CUDA_CHECK(cudaq_launch_dispatch_graph(dispatch_ctx, capture_stream));

    // F. Start CPU Infrastructure
    std::cout << "[Setup] Booting Thread Pool & Polling Loop...\n";
    cudaq::qec::utils::ThreadPool pymatching_pool(4);
    std::atomic<bool> system_stop{false};

    std::thread incoming_thread([&]() {
        incoming_polling_loop(predecoders, pymatching_pool, system_stop);
    });

    // =========================================================================
    // 5. The Test Stimulus (Acting as the FPGA)
    //
    // Original pattern: fire 8 requests (2 per decoder) all at once,
    // then wait for all responses.
    // =========================================================================
    std::cout << "\n[Test] Firing Syndromes...\n";

    int requests_sent = 0;
    for (int i = 0; i < 8; ++i) {
        int target_decoder = i % NUM_PREDECODERS;
        std::string target_func = "predecode_target_" + std::to_string(target_decoder);

        int slot = i % NUM_SLOTS;
        while (rx_flags_host[slot] != 0) usleep(10);

        uint8_t* slot_data = rx_data_host + (slot * SLOT_SIZE);
        auto* header = reinterpret_cast<cudaq::nvqlink::RPCHeader*>(slot_data);
        header->magic = cudaq::nvqlink::RPC_MAGIC_REQUEST;
        header->function_id = fnv1a_hash(target_func);
        header->arg_len = SYNDROME_FLOATS * sizeof(float);

        float* payload = reinterpret_cast<float*>(slot_data + sizeof(cudaq::nvqlink::RPCHeader));
        for (int j = 0; j < SYNDROME_FLOATS; ++j) payload[j] = 1.0f;

        __sync_synchronize();
        rx_flags_host[slot] = reinterpret_cast<uint64_t>(slot_data);
        requests_sent++;
    }

    // Wait for all 8 responses
    int responses_received = 0;
    for (int i = 0; i < requests_sent; ++i) {
        int slot = i % NUM_SLOTS;

        int timeout = 3000;
        while (tx_flags_host[slot] == 0 && timeout-- > 0) usleep(1000);

        uint64_t tv = tx_flags_host[slot];
        if (tv != 0 && (tv >> 48) == 0xDEAD) {
            int cuda_err = (int)(tv & 0xFFFF);
            std::cerr << "  [FAIL] Slot " << slot << " cudaGraphLaunch error "
                      << cuda_err << " (" << cudaGetErrorString((cudaError_t)cuda_err) << ")\n";
        } else if (tv != 0) {
            responses_received++;
            std::cout << "  -> Success: Slot " << slot << " completed the full trip!\n";
        } else {
            std::cerr << "  [FAIL] Timeout waiting for slot " << slot << "\n";
        }

        tx_flags_host[slot] = 0;
    }

    std::cout << "\n[Result] Processed " << responses_received << "/" << requests_sent
              << " requests successfully.\n";

    // =========================================================================
    // 6. Teardown
    // =========================================================================
    std::cout << "[Teardown] Shutting down...\n";
    *shutdown_flag_host = 1;
    __sync_synchronize();
    system_stop = true;

    incoming_thread.join();
    CUDA_CHECK(cudaStreamSynchronize(capture_stream));

    // Read back dispatcher stats for sanity check
    uint64_t dispatched_packets = 0;
    CUDA_CHECK(cudaMemcpy(&dispatched_packets, d_stats, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    std::cout << "[Stats] Dispatcher processed " << dispatched_packets << " packets.\n";

    CUDA_CHECK(cudaq_destroy_dispatch_graph(dispatch_ctx));

    // Cleanup memory
    cudaFreeHost((void*)rx_flags_host);
    cudaFreeHost((void*)tx_flags_host);
    cudaFreeHost(rx_data_host);
    cudaFreeHost(shutdown_flag_host);
    cudaFree(d_global_mailbox_bank);
    cudaFree(d_stats);
    cudaFree(d_function_entries);
    cudaStreamDestroy(capture_stream);

    remove(engine_path.c_str());

    std::cout << "Done.\n";
    return 0;
}
