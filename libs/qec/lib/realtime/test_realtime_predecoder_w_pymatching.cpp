/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/*******************************************************************************
 * Hybrid Realtime Pipeline Test with Real ONNX Pre-Decoder + PyMatching
 *
 * Supports multiple surface code configurations:
 *
 * d=7  r=7  (model1_d7_r7_unified_Z_batch1.onnx)
 * Input:  all_measurements  [1, 72, 7]    INT32  (2016 bytes)
 * Output: residual_detectors [1, 336]     INT32  (1344 bytes)
 * Output: logical_frame      [1]          INT32  (4 bytes)
 *
 * d=13 r=13 (model1_d13_r13_unified_Z_batch1.onnx)
 * Input:  all_measurements  [1, 252, 13]  INT32  (13104 bytes)
 * Output: residual_detectors [1, 2184]    INT32  (8736 bytes)
 * Output: logical_frame      [1]          INT32  (4 bytes)
 *
 * d=21 r=21 (model1_d21_r21_unified_Z_batch1.onnx)
 * Input:  all_measurements  [1, 660, 21]  INT32  (55440 bytes)
 * Output: residual_detectors [1, 9240]    INT32  (36960 bytes)
 * Output: logical_frame      [1]          INT32  (4 bytes)
 *
 * d=31 r=31 (model1_d31_r31_unified_Z_batch1.onnx)
 * Input:  all_measurements  [1, 1440, 31] INT32  (178560 bytes)
 * Output: residual_detectors [1, 29760]   INT32  (119040 bytes)
 * Output: logical_frame      [1]          INT32  (4 bytes)
 *
 * Pipeline:
 * 1. Ring Buffer setup
 * 2. Dispatcher Kernel -> Nx AIPreDecoderService instances (GPU, TRT from ONNX)
 * 3. GPU -> CPU N-Deep Pinned Memory Queue handoff
 * 4. Dedicated Polling Thread -> Worker PyMatching Thread Pool
 * 5. CPU Workers closing the transaction (Setting TX flags)
 *
 * Usage: test_realtime_predecoder_w_pymatching [d7|d13|d21|d31] [rate_us] [duration_s]
 ******************************************************************************/

 // Run the test:
 // ./build/unittests/test_realtime_predecoder_w_pymatching d13 30 10
 // distance 13, 30 us between requests, 10 seconds

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
 #include <fstream>
 #include <pthread.h>
#include <sched.h>
#include <nvtx3/nvToolsExt.h>

 #include <cuda_runtime.h>
 
 #ifndef CUDA_VERSION
 #define CUDA_VERSION 13000
 #endif
 #include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
 #include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
 #include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"
 
 #include "cudaq/qec/realtime/ai_decoder_service.h"
 #include "cudaq/qec/realtime/ai_predecoder_service.h"
 #include <cuda/std/atomic>
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

// Pin a thread to a specific CPU core (Cores 2-5 = spinning infra, 10+ = workers; 0-1 = OS).
static void pin_thread_to_core(std::thread& t, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Warning: Failed to pin thread to core " << core_id << " (Error: " << rc << ")\n";
    }
}

static void pin_current_thread_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Warning: Failed to pin current thread to core " << core_id << " (Error: " << rc << ")\n";
    }
}

using namespace cudaq::qec;
namespace realtime_ns = cudaq::realtime;
 
 // =============================================================================
 // Pipeline Configuration
 // =============================================================================
 
 constexpr size_t NUM_SLOTS = 32;
 
 struct PipelineConfig {
     std::string label;
     int distance;
     int num_rounds;
     int meas_qubits;          // ONNX input shape[1]
     int residual_detectors;   // ONNX output dim
     std::string onnx_filename;
     size_t slot_size;         // must fit RPC header (CUDAQ_RPC_HEADER_SIZE) + input payload
     int num_predecoders;
     int num_workers;
 
     int input_elements() const { return meas_qubits * num_rounds; }
     size_t input_bytes() const { return input_elements() * sizeof(int32_t); }
 
     std::string onnx_path() const {
         return std::string(ONNX_MODEL_DIR) + "/" + onnx_filename;
     }
 
     std::string engine_path() const {
         std::string name = onnx_filename;
         auto dot = name.rfind('.');
         if (dot != std::string::npos)
             name = name.substr(0, dot);
         return std::string(ONNX_MODEL_DIR) + "/" + name + ".engine";
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
             /*num_predecoders=*/16,
             /*num_workers=*/16
         };
     }

     static PipelineConfig d13_r13() {
         return {
             "d13_r13_Z",
             /*distance=*/13,
             /*num_rounds=*/13,
             /*meas_qubits=*/252,
             /*residual_detectors=*/2184,
             "predecoder_memory_d13_T13_X.onnx",
             /*slot_size=*/16384,
             /*num_predecoders=*/16,
             /*num_workers=*/16
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
             /*num_predecoders=*/16,
             /*num_workers=*/16
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
             /*num_predecoders=*/16,
             /*num_workers=*/16
         };
     }
 };
 
 // Runtime decoder state populated during setup
 struct DecoderContext {
     std::vector<std::unique_ptr<cudaq::qec::decoder>> decoders;
     std::atomic<int> next_decoder_idx{0};
     int z_stabilizers = 0;
     int spatial_slices = 0;
 
     cudaq::qec::decoder* acquire_decoder() {
         thread_local int my_idx = next_decoder_idx.fetch_add(1, std::memory_order_relaxed);
         return decoders[my_idx % decoders.size()].get();
     }
 
     // Per-worker timing accumulators (lock-free)
     std::atomic<int64_t> total_decode_us{0};
     std::atomic<int64_t> total_worker_us{0};
     std::atomic<int> decode_count{0};
 };
 
 struct SystemContext {
     realtime_ns::atomic_uint64_sys* tx_flags_host = nullptr;
     uint8_t* rx_data_host = nullptr;
     size_t slot_size = 0;
 };
 SystemContext g_sys_ctx;

 /// Context for dynamic worker pool: worker task writes tx_flags[origin_slot] and frees idle_mask.
 struct WorkerPoolContext {
     realtime_ns::atomic_uint64_sys* tx_flags = nullptr;
     realtime_ns::atomic_uint64_sys* idle_mask = nullptr;
     int* inflight_slot_tags = nullptr;
     uint64_t* debug_poll_ts = nullptr;      // when worker poll_next_job succeeded (ns epoch)
     uint64_t* debug_worker_done_ts = nullptr; // when worker set tx_flags (ns epoch)
 };
 
 // =============================================================================
 // Thread Pool Worker (Real PyMatching MWPM Decoder)
 // =============================================================================
 
 struct __attribute__((packed)) DecodeResponse {
     int32_t total_corrections;
     int32_t converged;
 };
 
 void pymatching_worker_task(PreDecoderJob job, int worker_id,
                             AIPreDecoderService* predecoder,
                             DecoderContext* ctx,
    WorkerPoolContext* pool_ctx) {
    nvtxRangePushA("Worker Task");
    using hrclock = std::chrono::high_resolution_clock;
    auto worker_start = hrclock::now();

    if (pool_ctx && pool_ctx->debug_poll_ts) {
        pool_ctx->debug_poll_ts[job.origin_slot] = std::chrono::duration_cast<std::chrono::nanoseconds>(
            worker_start.time_since_epoch()).count();
    }

    int total_corrections = 0;
    bool all_converged = true;

    auto decode_start = hrclock::now();
#if !defined(DISABLE_PYMATCHING)
    const int32_t* residual = static_cast<const int32_t*>(job.inference_data);
    auto* my_decoder = ctx->acquire_decoder();

    nvtxRangePushA("PyMatching Decode");
    
    cudaqx::tensor<uint8_t> syndrome_tensor({(size_t)ctx->z_stabilizers});
    uint8_t* syn_data = syndrome_tensor.data();

    for (int s = 0; s < ctx->spatial_slices; ++s) {
        const int32_t* slice = residual + s * ctx->z_stabilizers;
        for (int i = 0; i < ctx->z_stabilizers; ++i) {
            syn_data[i] = static_cast<uint8_t>(slice[i]);
        }

        auto result = my_decoder->decode(syndrome_tensor);

        all_converged &= result.converged;
        for (auto v : result.result)
            if (v > 0.5) total_corrections++;
    }
    nvtxRangePop(); // PyMatching Decode
#endif
    auto decode_end = hrclock::now();

    DecodeResponse resp_data{total_corrections, all_converged ? 1 : 0};

    char* response_payload = (char*)job.ring_buffer_ptr + sizeof(realtime_ns::RPCResponse);
    std::memcpy(response_payload, &resp_data, sizeof(resp_data));

    auto* header = static_cast<realtime_ns::RPCResponse*>(job.ring_buffer_ptr);
    header->magic = realtime_ns::RPC_MAGIC_RESPONSE;
    header->status = 0;
    header->result_len = sizeof(resp_data);

    uint64_t rx_value = reinterpret_cast<uint64_t>(job.ring_buffer_ptr);
    int origin_slot = job.origin_slot;

    if (pool_ctx && pool_ctx->tx_flags) {
        pool_ctx->tx_flags[origin_slot].store(rx_value, cuda::std::memory_order_release);
    } else {
        size_t slot_idx = ((uint8_t*)job.ring_buffer_ptr - g_sys_ctx.rx_data_host) / g_sys_ctx.slot_size;
        g_sys_ctx.tx_flags_host[slot_idx].store(rx_value, cuda::std::memory_order_release);
    }

    if (pool_ctx && pool_ctx->debug_worker_done_ts) {
        pool_ctx->debug_worker_done_ts[origin_slot] = std::chrono::duration_cast<std::chrono::nanoseconds>(
            hrclock::now().time_since_epoch()).count();
    }

    predecoder->release_job(job.slot_idx);

    if (pool_ctx && pool_ctx->idle_mask) {
        pool_ctx->idle_mask->fetch_or(1ULL << worker_id, cuda::std::memory_order_release);
    }

    auto worker_end = hrclock::now();
    auto decode_us = std::chrono::duration_cast<std::chrono::microseconds>(
        decode_end - decode_start).count();
    auto worker_us = std::chrono::duration_cast<std::chrono::microseconds>(
        worker_end - worker_start).count();
    ctx->total_decode_us.fetch_add(decode_us, std::memory_order_relaxed);
    ctx->total_worker_us.fetch_add(worker_us, std::memory_order_relaxed);
    ctx->decode_count.fetch_add(1, std::memory_order_relaxed);
    nvtxRangePop(); // Worker Task
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
 // Streaming Test Mode (simulates FPGA continuous syndrome arrival)
 // =============================================================================
 
 struct StreamingConfig {
     int rate_us = 0;       // inter-arrival time in us (0 = open-loop)
     int duration_s = 5;    // how long to run
     int warmup_count = 20; // discard first N from latency stats
 };
 
 void run_streaming_test(
     const PipelineConfig& config,
     const StreamingConfig& scfg,
     uint8_t* rx_data_host,
     uint8_t* rx_data_dev,
     realtime_ns::atomic_uint64_sys* rx_flags,
     realtime_ns::atomic_uint64_sys* tx_flags,
     DecoderContext& decoder_ctx,
     std::vector<std::unique_ptr<AIPreDecoderService>>& predecoders,
     std::atomic<bool>& system_stop,
     void** h_mailbox_bank,
     std::vector<cudaStream_t>& predecoder_streams,
     WorkerPoolContext* pool_ctx,
     std::atomic<uint64_t>* total_claimed = nullptr)
 {
     using hrclock = std::chrono::high_resolution_clock;
     using atomic_uint64_sys = realtime_ns::atomic_uint64_sys;
     using atomic_int_sys = realtime_ns::atomic_int_sys;
 
     const int num_workers = config.num_predecoders;
     const int max_requests = 500000;
     const size_t payload_bytes = config.input_bytes();
 
     std::vector<hrclock::time_point> submit_ts(max_requests);
     std::vector<hrclock::time_point> complete_ts(max_requests);
     std::vector<bool> completed(max_requests, false);
     std::vector<uint64_t> dispatch_ts(max_requests, 0);
     std::vector<uint64_t> poll_ts(max_requests, 0);
     std::vector<uint64_t> worker_done_ts(max_requests, 0);
 
     std::vector<int> slot_request(NUM_SLOTS, -1);
     std::vector<uint64_t> debug_dispatch_ts_arr(NUM_SLOTS, 0);
 
     std::atomic<int> total_submitted{0};
     std::atomic<int> total_completed{0};
     std::atomic<int64_t> backpressure_stalls{0};
     std::atomic<bool> producer_done{false};
     std::atomic<bool> consumer_stop{false};

    atomic_int_sys shutdown_flag(0);
    uint64_t dispatcher_stats = 0;
    atomic_uint64_sys live_dispatched(0);

    // Build function table for realtime host dispatcher (lookup by function_id).
    std::vector<cudaq_function_entry_t> function_table(num_workers);
    for (int i = 0; i < num_workers; ++i) {
        std::string func_name = "predecode_target_" + std::to_string(i);
        function_table[i].function_id = realtime_ns::fnv1a_hash(func_name.c_str());
        function_table[i].dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
        function_table[i].handler.graph_exec = predecoders[i]->get_executable_graph();
        std::memset(&function_table[i].schema, 0, sizeof(function_table[i].schema));
    }

    realtime_ns::HostDispatcherConfig disp_cfg;
    disp_cfg.rx_flags = rx_flags;
    disp_cfg.tx_flags = tx_flags;
    disp_cfg.rx_data_host = rx_data_host;
    disp_cfg.rx_data_dev = rx_data_dev;
    disp_cfg.tx_data_host = nullptr;
     disp_cfg.tx_data_dev = nullptr;
     disp_cfg.tx_stride_sz = config.slot_size;
    disp_cfg.h_mailbox_bank = h_mailbox_bank;
    disp_cfg.num_slots = NUM_SLOTS;
    disp_cfg.slot_size = config.slot_size;
    disp_cfg.function_table = function_table.data();
    disp_cfg.function_table_count = num_workers;
    disp_cfg.shutdown_flag = &shutdown_flag;
    disp_cfg.stats_counter = &dispatcher_stats;
    disp_cfg.live_dispatched = &live_dispatched;
    disp_cfg.idle_mask = pool_ctx->idle_mask;
    disp_cfg.inflight_slot_tags = pool_ctx->inflight_slot_tags;
    disp_cfg.workers.resize(num_workers);
    for (int i = 0; i < num_workers; ++i) {
        disp_cfg.workers[i].graph_exec = predecoders[i]->get_executable_graph();
        disp_cfg.workers[i].stream = predecoder_streams[i];
        disp_cfg.workers[i].function_id = function_table[i].function_id;
    }

    std::thread dispatcher_thread([&disp_cfg]() {
        realtime_ns::host_dispatcher_loop(disp_cfg);
    });
    pin_thread_to_core(dispatcher_thread, 2);

    // Ring buffer view for producer/consumer helpers (realtime C API).
    cudaq_ringbuffer_t rb{};
    rb.rx_flags = reinterpret_cast<volatile uint64_t*>(rx_flags);
    rb.tx_flags = reinterpret_cast<volatile uint64_t*>(tx_flags);
    rb.rx_data = rx_data_dev;
    rb.tx_data = rx_data_dev;
    rb.rx_stride_sz = config.slot_size;
    rb.tx_stride_sz = config.slot_size;
    rb.rx_flags_host = reinterpret_cast<volatile uint64_t*>(rx_flags);
    rb.tx_flags_host = reinterpret_cast<volatile uint64_t*>(tx_flags);
    rb.rx_data_host = rx_data_host;
    rb.tx_data_host = rx_data_host;

     auto run_deadline = std::chrono::steady_clock::now()
                       + std::chrono::seconds(scfg.duration_s);

     std::string rate_label = (scfg.rate_us > 0)
         ? std::to_string(scfg.rate_us) + " us"
         : "open-loop";

    std::cout << "\n[Stream] Starting streaming test (" << config.label
              << ", HOST dispatcher)\n"
              << "  Rate:       " << rate_label << "\n"
              << "  Duration:   " << scfg.duration_s << " s\n"
              << "  Warmup:     " << scfg.warmup_count << " requests\n"
              << "  Predecoders:" << config.num_predecoders << " (dedicated streams)\n"
              << "  Max reqs:   " << max_requests << "\n\n"
              << std::flush;

    // Progress reporter (debug only; set to true to print submitted/completed every second)
    constexpr bool kEnableProgressReporter = true;
    std::atomic<bool> progress_done{false};
    std::thread progress_reporter;
    if (kEnableProgressReporter) {
        progress_reporter = std::thread([&]() {
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (progress_done.load(std::memory_order_acquire)) break;
                bool pdone = producer_done.load(std::memory_order_acquire);
                int nsub = total_submitted.load(std::memory_order_acquire);
                int ncomp = total_completed.load(std::memory_order_acquire);
                uint64_t disp = live_dispatched.load(cuda::std::memory_order_relaxed);
                uint64_t claimed = total_claimed ? total_claimed->load(std::memory_order_relaxed) : 0;
                uint64_t mask = pool_ctx->idle_mask ? pool_ctx->idle_mask->load(cuda::std::memory_order_relaxed) : 0;
                std::cout << "  [progress] submitted=" << nsub << " completed=" << ncomp
                          << " dispatched=" << disp << " claimed=" << claimed
                          << " idle_mask=0x" << std::hex << mask << std::dec << std::endl;
                if (pdone && ncomp >= nsub) break;
            }
        });
    }

     // --- Producer thread (simulates FPGA) ---
     std::thread producer([&]() {
         std::mt19937 rng(42);
         int next_slot = 0;
         int req_id = 0;

         while (std::chrono::steady_clock::now() < run_deadline
                && req_id < max_requests) {

            int slot = next_slot % (int)NUM_SLOTS;

            while (!cudaq_host_ringbuffer_slot_available(&rb, static_cast<uint32_t>(slot))) {
                 backpressure_stalls.fetch_add(1, std::memory_order_relaxed);
                 QEC_CPU_RELAX();
                 if (std::chrono::steady_clock::now() >= run_deadline) return;
             }

             int target = req_id % config.num_predecoders;
             std::string func = "predecode_target_" + std::to_string(target);
             uint32_t function_id = realtime_ns::fnv1a_hash(func.c_str());

             uint8_t* slot_data = rx_data_host + (slot * config.slot_size);
             int32_t* payload = reinterpret_cast<int32_t*>(
                 slot_data + CUDAQ_RPC_HEADER_SIZE);
             fill_measurement_payload(payload, config.input_elements(), rng, 0.01);

             cudaq_host_ringbuffer_write_rpc_request(&rb, static_cast<uint32_t>(slot),
                 function_id, payload, static_cast<uint32_t>(payload_bytes));

             slot_request[slot] = req_id;
             submit_ts[req_id] = hrclock::now();
             cudaq_host_ringbuffer_signal_slot(&rb, static_cast<uint32_t>(slot));
             total_submitted.fetch_add(1, std::memory_order_release);

             next_slot++;
             req_id++;

             if (scfg.rate_us > 0) {
                 auto target_time = submit_ts[req_id - 1]
                                  + std::chrono::microseconds(scfg.rate_us);
                 while (hrclock::now() < target_time)
                     QEC_CPU_RELAX();
             }
         }

         producer_done.store(true, std::memory_order_seq_cst);
     });
    pin_thread_to_core(producer, 3);

     // --- Consumer thread (harvests completions out-of-order) ---
     std::thread consumer([&]() {
         while (true) {
             if (consumer_stop.load(std::memory_order_acquire))
                 break;
             bool pdone = producer_done.load(std::memory_order_acquire);
             int nsub = total_submitted.load(std::memory_order_acquire);
             int ncomp = total_completed.load(std::memory_order_relaxed);

             if (pdone && ncomp >= nsub)
                 break;

             bool found_any = false;
             for (uint32_t s = 0; s < NUM_SLOTS; ++s) {
                 if (slot_request[s] < 0) continue;

                 int cuda_error = 0;
                 cudaq_tx_status_t status = cudaq_host_ringbuffer_poll_tx_flag(
                     &rb, s, &cuda_error);

                 if (status == CUDAQ_TX_READY) {
                     int rid = slot_request[s];
                     if (rid >= 0) {
                         complete_ts[rid] = hrclock::now();
                         poll_ts[rid] = pool_ctx->debug_poll_ts ? pool_ctx->debug_poll_ts[s] : 0;
                         worker_done_ts[rid] = pool_ctx->debug_worker_done_ts ? pool_ctx->debug_worker_done_ts[s] : 0;
                         completed[rid] = true;
                         total_completed.fetch_add(1, std::memory_order_relaxed);
                     }
                     slot_request[s] = -1;
                     __sync_synchronize();
                     cudaq_host_ringbuffer_clear_slot(&rb, s);
                     found_any = true;
                 } else if (status == CUDAQ_TX_ERROR) {
                     std::cerr << "  [FAIL] Slot " << s
                               << " cudaGraphLaunch error " << cuda_error
                               << " (" << cudaGetErrorString(static_cast<cudaError_t>(cuda_error))
                               << ")\n";
                     total_completed.fetch_add(1, std::memory_order_relaxed);
                     slot_request[s] = -1;
                     __sync_synchronize();
                     cudaq_host_ringbuffer_clear_slot(&rb, s);
                     found_any = true;
                 }
             }
             if (!found_any) QEC_CPU_RELAX();
         }
     });
    pin_thread_to_core(consumer, 4);

     std::cout << "  [shutdown] joining producer...\n" << std::flush;
     producer.join();

     // Grace period for in-flight requests
     auto grace_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
     while (total_completed.load() < total_submitted.load()
            && std::chrono::steady_clock::now() < grace_deadline) {
         usleep(1000);
     }

     if (total_completed.load() < total_submitted.load()) {
         int nsub_dbg = total_submitted.load();
         int ncomp_dbg = total_completed.load();
         std::cerr << "  [DEBUG] Stuck: submitted=" << nsub_dbg << " completed=" << ncomp_dbg
                   << " diff=" << (nsub_dbg - ncomp_dbg) << "\n";
         for (uint32_t s = 0; s < NUM_SLOTS; ++s) {
             uint64_t rx_val = reinterpret_cast<volatile uint64_t*>(rx_flags)[s];
             uint64_t tx_val = reinterpret_cast<volatile uint64_t*>(tx_flags)[s];
             int rid = slot_request[s];
             if (rx_val != 0 || tx_val != 0 || rid >= 0) {
                 std::cerr << "    slot[" << s << "] rx=0x" << std::hex << rx_val
                           << " tx=0x" << tx_val << std::dec
                           << " slot_request=" << rid
                           << " (completed=" << (rid >= 0 ? (completed[rid] ? "YES" : "NO") : "n/a")
                           << ")\n";
             }
         }
         for (int w = 0; w < config.num_predecoders; ++w) {
             auto* pd = predecoders[w].get();
             std::cerr << "    worker[" << w << "] inflight_slot_tag="
                       << pool_ctx->inflight_slot_tags[w]
                       << " idle=" << ((pool_ctx->idle_mask->load(cuda::std::memory_order_relaxed) >> w) & 1)
                       << "\n";
         }
     }

     consumer_stop.store(true, std::memory_order_release);

     shutdown_flag.store(1, cuda::std::memory_order_release);
     std::cout << "  [shutdown] joining dispatcher...\n" << std::flush;
     dispatcher_thread.join();
     std::cout << "  [shutdown] joining consumer...\n" << std::flush;
     consumer.join();

     if (kEnableProgressReporter) {
         progress_done.store(true, std::memory_order_release);
         progress_reporter.join();
     }

     // ===== Report =====
     auto run_end = std::chrono::steady_clock::now();
     int nsub = total_submitted.load();
     int ncomp = total_completed.load();
     if (ncomp < nsub)
         std::cerr << "  [WARN] " << (nsub - ncomp) << " in-flight requests did not complete before grace period.\n";

     // Build PipelineBenchmark from timestamps (skip warmup)
     int warmup = std::min(scfg.warmup_count, nsub);
     int bench_count = nsub - warmup;
 
     cudaq::qec::utils::PipelineBenchmark bench(
         config.label + " (stream)", bench_count);
     bench.start();
 
     for (int i = warmup; i < nsub; ++i) {
         int bench_id = i - warmup;
         bench.mark_submit(bench_id);
     }
 
     std::vector<double> latencies;
     latencies.reserve(bench_count);
     for (int i = warmup; i < nsub; ++i) {
         if (!completed[i]) continue;
         auto dt = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
             complete_ts[i] - submit_ts[i]);
         latencies.push_back(dt.count());
     }
 
     bench.stop();
 
     std::sort(latencies.begin(), latencies.end());
 
     auto pct = [&](double p) -> double {
         if (latencies.empty()) return 0;
         double idx = (p / 100.0) * (latencies.size() - 1);
         size_t lo = (size_t)idx;
         size_t hi = std::min(lo + 1, latencies.size() - 1);
         double frac = idx - lo;
         return latencies[lo] * (1.0 - frac) + latencies[hi] * frac;
     };
 
     double mean = 0;
     for (auto v : latencies) mean += v;
     mean = latencies.empty() ? 0 : mean / latencies.size();
 
     double stddev = 0;
     for (auto v : latencies) stddev += (v - mean) * (v - mean);
     stddev = latencies.empty() ? 0 : std::sqrt(stddev / latencies.size());
 
     auto wall_us = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
         run_end - (run_deadline - std::chrono::seconds(scfg.duration_s))).count();
     double throughput = (wall_us > 0) ? (ncomp * 1e6 / wall_us) : 0;
 
     double actual_rate = (nsub > 1)
         ? std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
               submit_ts[nsub - 1] - submit_ts[0]).count() / (nsub - 1)
         : 0;
 
     std::cout << std::fixed;
     std::cout << "\n================================================================\n";
     std::cout << "  Streaming Benchmark: " << config.label << "\n";
     std::cout << "================================================================\n";
     std::cout << "  Submitted:          " << nsub << "\n";
     std::cout << "  Completed:          " << ncomp << "\n";
     if (nsub > ncomp)
         std::cout << "  Dropped/timeout:    " << (nsub - ncomp) << "\n";
     std::cout << std::setprecision(1);
     std::cout << "  Wall time:          " << wall_us / 1000.0 << " ms\n";
     std::cout << "  Throughput:         " << throughput << " req/s\n";
     std::cout << "  Actual arrival rate:" << std::setw(8) << actual_rate << " us/req\n";
     std::cout << "  Backpressure stalls:" << std::setw(8)
               << backpressure_stalls.load() << "\n";
     std::cout << "  ---------------------------------------------------------------\n";
     std::cout << "  Latency (us)  [steady-state, " << latencies.size()
               << " requests after " << warmup << " warmup]\n";
     std::cout << std::setprecision(1);
     if (!latencies.empty()) {
         std::cout << "    min    = " << std::setw(10) << latencies.front() << "\n";
         std::cout << "    p50    = " << std::setw(10) << pct(50) << "\n";
         std::cout << "    mean   = " << std::setw(10) << mean << "\n";
         std::cout << "    p90    = " << std::setw(10) << pct(90) << "\n";
         std::cout << "    p95    = " << std::setw(10) << pct(95) << "\n";
         std::cout << "    p99    = " << std::setw(10) << pct(99) << "\n";
         std::cout << "    max    = " << std::setw(10) << latencies.back() << "\n";
         std::cout << "    stddev = " << std::setw(10) << stddev << "\n";
     }
     std::cout << "  ---------------------------------------------------------------\n";
 
     // Worker timing breakdown
     int n_decoded = decoder_ctx.decode_count.load();
     if (n_decoded > 0) {
         double avg_decode = (double)decoder_ctx.total_decode_us.load() / n_decoded;
         double avg_worker = (double)decoder_ctx.total_worker_us.load() / n_decoded;
         double avg_overhead = avg_worker - avg_decode;

         // Per-request breakdown using submit, poll (worker start), worker_done, complete timestamps.
         // Stage A: submit → poll_ts  = dispatch + graph launch + GPU execution + poll CAS
         // Stage B: poll_ts → worker_done_ts = worker task (decode + response write + tx_flags set)
         // Stage C: worker_done_ts → complete_ts = consumer polling delay
         double sum_stage_a = 0, sum_stage_b = 0, sum_stage_c = 0;
         int count_valid = 0;
         std::vector<double> stage_a_samples, stage_b_samples, stage_c_samples;
         for (int i = warmup; i < nsub; ++i) {
             if (!completed[i] || poll_ts[i] == 0 || worker_done_ts[i] == 0) continue;
             uint64_t submit_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                 submit_ts[i].time_since_epoch()).count();
             uint64_t complete_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                 complete_ts[i].time_since_epoch()).count();
             if (poll_ts[i] <= submit_ns || worker_done_ts[i] < poll_ts[i] || complete_ns < worker_done_ts[i])
                 continue;
             double a = (poll_ts[i] - submit_ns) / 1000.0;
             double b = (worker_done_ts[i] - poll_ts[i]) / 1000.0;
             double c = (complete_ns - worker_done_ts[i]) / 1000.0;
             sum_stage_a += a; sum_stage_b += b; sum_stage_c += c;
             stage_a_samples.push_back(a);
             stage_b_samples.push_back(b);
             stage_c_samples.push_back(c);
             count_valid++;
         }

         auto percentile = [](std::vector<double>& v, double pct) -> double {
             if (v.empty()) return 0;
             std::sort(v.begin(), v.end());
             size_t idx = std::min((size_t)(pct / 100.0 * v.size()), v.size() - 1);
             return v[idx];
         };

         double avg_a = count_valid > 0 ? sum_stage_a / count_valid : 0;
         double avg_b = count_valid > 0 ? sum_stage_b / count_valid : 0;
         double avg_c = count_valid > 0 ? sum_stage_c / count_valid : 0;

         std::cout << std::setprecision(1);
         std::cout << "  Pipeline Timing Breakdown (" << count_valid << " valid samples):\n";
         std::cout << "    [A] Submit→Worker poll:" << std::setw(9) << avg_a
                   << " us  (p50=" << percentile(stage_a_samples, 50)
                   << " p99=" << percentile(stage_a_samples, 99) << ")\n";
         std::cout << "        (dispatch + graph launch + GPU exec + CAS)\n";
         std::cout << "    [B] Worker task:       " << std::setw(9) << avg_b
                   << " us  (p50=" << percentile(stage_b_samples, 50)
                   << " p99=" << percentile(stage_b_samples, 99) << ")\n";
         std::cout << "        (decode + response write + tx_flags set)\n";
         std::cout << "    [C] Consumer poll lag: " << std::setw(9) << avg_c
                   << " us  (p50=" << percentile(stage_c_samples, 50)
                   << " p99=" << percentile(stage_c_samples, 99) << ")\n";
         std::cout << "        (tx_flags set → consumer sees it)\n";
         std::cout << "    [A+B+C] Sum:           " << std::setw(9) << (avg_a + avg_b + avg_c) << " us\n";
         std::cout << "    End-to-end mean:       " << std::setw(9) << mean << " us\n";
         std::cout << "    Per-round (/" << config.num_rounds << "):      "
                   << std::setw(9) << (mean / config.num_rounds) << " us/round\n";
         std::cout << "  ---------------------------------------------------------------\n";
         std::cout << "  Worker-level averages (" << n_decoded << " completed):\n";
         std::cout << "    PyMatching decode:    " << std::setw(9) << avg_decode << " us\n";
         std::cout << "    Total worker:         " << std::setw(9) << avg_worker << " us\n";
         std::cout << "    Worker overhead:      " << std::setw(9) << avg_overhead << " us\n";
     }
     std::cout << "  ---------------------------------------------------------------\n";
     std::cout << "  Host dispatcher processed " << dispatcher_stats << " packets.\n";
     std::cout << "================================================================\n";
 }
 
 // =============================================================================
 // Main
 // =============================================================================
 int main(int argc, char* argv[]) {
     // Parse arguments: <config> [rate_us] [duration_s]
     std::string config_name = "d7";
     StreamingConfig stream_cfg;

     if (argc > 1)
         config_name = argv[1];
     if (argc > 2 && std::isdigit(argv[2][0]))
         stream_cfg.rate_us = std::stoi(argv[2]);
     if (argc > 3 && std::isdigit(argv[3][0]))
         stream_cfg.duration_s = std::stoi(argv[3]);

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
         std::cerr << "Usage: " << argv[0] << " [d7|d13|d21|d31] [rate_us] [duration_s]\n"
                   << "  d7     - distance 7, 7 rounds (default)\n"
                   << "  d13    - distance 13, 13 rounds\n"
                   << "  d21    - distance 21, 21 rounds\n"
                   << "  d31    - distance 31, 31 rounds\n"
                   << "  rate_us    - inter-arrival time in us (0 = open-loop, default)\n"
                   << "  duration_s - test duration in seconds (default: 5)\n"
                   << "\nExamples:\n"
                   << "  " << argv[0] << " d13           # open-loop, 5s\n"
                   << "  " << argv[0] << " d13 50        # 50 us between requests, 5s\n"
                   << "  " << argv[0] << " d13 50 10     # 50 us rate, 10s duration\n";
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
 
     std::string engine_file = config.engine_path();
     std::string onnx_file = config.onnx_path();
     std::string model_path;
 
     std::ifstream engine_probe(engine_file, std::ios::binary);
     if (engine_probe.good()) {
         engine_probe.close();
         model_path = engine_file;
         std::cout << "[Setup] Loading cached TRT engine: " << engine_file << "\n";
     } else {
         model_path = onnx_file;
         std::cout << "[Setup] Building TRT engines from ONNX: " << onnx_file << "\n";
         std::cout << "[Setup] Engine will be cached to: " << engine_file << "\n";
     }
 
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
     std::cout << "[Setup] Pre-allocating " << config.num_workers
               << " PyMatching decoders (one per worker)...\n";
     for (int i = 0; i < config.num_workers; ++i)
         decoder_ctx.decoders.push_back(
             cudaq::qec::decoder::get("pymatching", H_z, pm_params));
     std::cout << "[Setup] PyMatching decoder pool ready.\n";
 
    // =========================================================================
    // System-Scope Atomics & Ring Buffer Allocation (Replaces volatile setup)
    // =========================================================================
    using atomic_uint64_sys = realtime_ns::atomic_uint64_sys;
    using atomic_int_sys = realtime_ns::atomic_int_sys;

    void* buf_rx = nullptr;
     CUDA_CHECK(cudaHostAlloc(&buf_rx, NUM_SLOTS * sizeof(atomic_uint64_sys), cudaHostAllocMapped));
     atomic_uint64_sys* rx_flags_host = static_cast<atomic_uint64_sys*>(buf_rx);
     for (size_t i = 0; i < NUM_SLOTS; ++i) new (rx_flags_host + i) atomic_uint64_sys(0);
     
     void* buf_tx = nullptr;
     CUDA_CHECK(cudaHostAlloc(&buf_tx, NUM_SLOTS * sizeof(atomic_uint64_sys), cudaHostAllocMapped));
     atomic_uint64_sys* tx_flags_host = static_cast<atomic_uint64_sys*>(buf_tx);
     for (size_t i = 0; i < NUM_SLOTS; ++i) new (tx_flags_host + i) atomic_uint64_sys(0);
 
     uint64_t* rx_flags_dev = nullptr;
     uint64_t* tx_flags_dev = nullptr;
     CUDA_CHECK(cudaHostGetDevicePointer((void**)&rx_flags_dev, buf_rx, 0));
     CUDA_CHECK(cudaHostGetDevicePointer((void**)&tx_flags_dev, buf_tx, 0));
 
     uint8_t *rx_data_host, *rx_data_dev;
     CUDA_CHECK(cudaHostAlloc(&rx_data_host, NUM_SLOTS * config.slot_size, cudaHostAllocMapped));
     CUDA_CHECK(cudaHostGetDevicePointer((void**)&rx_data_dev, rx_data_host, 0));
 
     g_sys_ctx.tx_flags_host = tx_flags_host;
     g_sys_ctx.rx_data_host = rx_data_host;
     g_sys_ctx.slot_size = config.slot_size;
 
     // Define the dynamic pool variables HERE so they live until the program exits
     // Avoid 1ULL<<64 (UB); for 64 workers use all-ones mask.
     uint64_t initial_idle = (config.num_predecoders >= 64)
         ? ~0ULL
         : ((1ULL << config.num_predecoders) - 1);
     atomic_uint64_sys idle_mask(initial_idle);  
     std::vector<int> inflight_slot_tags(config.num_predecoders, 0);
     std::vector<uint64_t> debug_poll_ts_arr(NUM_SLOTS, 0);
    std::vector<uint64_t> debug_worker_done_ts_arr(NUM_SLOTS, 0);
     
     WorkerPoolContext pool_ctx;
     pool_ctx.tx_flags = tx_flags_host;
     pool_ctx.idle_mask = &idle_mask;
     pool_ctx.inflight_slot_tags = inflight_slot_tags.data();
     pool_ctx.debug_poll_ts = debug_poll_ts_arr.data();
     pool_ctx.debug_worker_done_ts = debug_worker_done_ts_arr.data();
 
     // =========================================================================
     // Mailbox & Dispatcher Setup (mode-dependent)
     // =========================================================================
 
     void** h_mailbox_bank = nullptr;
     void** d_mailbox_bank = nullptr;
     CUDA_CHECK(cudaHostAlloc(&h_mailbox_bank, config.num_predecoders * sizeof(void*), cudaHostAllocMapped));
     std::memset(h_mailbox_bank, 0, config.num_predecoders * sizeof(void*));
     CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_mailbox_bank, h_mailbox_bank, 0));

     std::vector<cudaStream_t> predecoder_streams;
     for (int i = 0; i < config.num_predecoders; ++i) {
         cudaStream_t s;
         CUDA_CHECK(cudaStreamCreate(&s));
         predecoder_streams.push_back(s);
     }

     std::cout << "[Setup] Capturing " << config.num_predecoders
               << "x AIPreDecoder Graphs (host-launch)...\n";
     cudaStream_t capture_stream;
     CUDA_CHECK(cudaStreamCreate(&capture_stream));

     std::vector<std::unique_ptr<AIPreDecoderService>> predecoders;
     bool need_save = (model_path == onnx_file);
     const int predecoder_queue_depth = 1;
     for (int i = 0; i < config.num_predecoders; ++i) {
         std::string save_path = (need_save && i == 0) ? engine_file : "";
         auto pd = std::make_unique<AIPreDecoderService>(model_path, d_mailbox_bank + i,
                                                         predecoder_queue_depth,
                                                         save_path);

         std::cout << "[Setup] Decoder " << i
                   << ": input_size=" << pd->get_input_size()
                   << " output_size=" << pd->get_output_size() << "\n";

         pd->capture_graph(capture_stream, false /* host-launch */);

         predecoders.push_back(std::move(pd));
     }

     std::cout << "[Setup] Host-side dispatcher will be launched in streaming test.\n";
 
    std::atomic<bool> system_stop{false};
    std::atomic<uint64_t> total_claimed{0};

    std::cout << "[Setup] Booting " << config.num_workers << " Dedicated Polling/Worker Threads...\n";
    std::vector<std::thread> worker_threads;
    for (int i = 0; i < config.num_workers; ++i) {
        worker_threads.emplace_back([i, &predecoders, &decoder_ctx, &system_stop, &pool_ctx, &total_claimed]() {
            int target_core = 10 + i;
            pin_current_thread_to_core(target_core);

            AIPreDecoderService* pd_ptr = predecoders[i].get();

            nvtxRangePushA("Worker Loop");
            PreDecoderJob job;
            while (!system_stop.load(std::memory_order_relaxed)) {
                // Wait for GPU to set ready flag to 1
                if (pd_ptr->poll_next_job(job)) {
                    nvtxRangePushA("Process Job");
                    
                    total_claimed.fetch_add(1, std::memory_order_relaxed);

                    if (pool_ctx.inflight_slot_tags) {
                        job.origin_slot = pool_ctx.inflight_slot_tags[i];
                    } else {
                        job.origin_slot = static_cast<int>(((uint8_t*)job.ring_buffer_ptr - g_sys_ctx.rx_data_host) / g_sys_ctx.slot_size);
                    }

                    pymatching_worker_task(job, i, pd_ptr, &decoder_ctx, &pool_ctx);
                    nvtxRangePop(); // Process Job
                } else {
                    QEC_CPU_RELAX();
                }
            }
            nvtxRangePop(); // Worker Loop
        });
    }
 
     // =========================================================================
     // Streaming test
     // =========================================================================
     run_streaming_test(config, stream_cfg,
                        rx_data_host, rx_data_dev, rx_flags_host, tx_flags_host,
                        decoder_ctx, predecoders, system_stop,
                        h_mailbox_bank, predecoder_streams, &pool_ctx, &total_claimed);

     // Teardown
     std::cout << "[Teardown] Shutting down...\n";
     system_stop = true;

     for (auto& t : worker_threads) {
         if (t.joinable()) t.join();
     }
     CUDA_CHECK(cudaStreamSynchronize(capture_stream));

     for (auto& s : predecoder_streams) {
         cudaStreamSynchronize(s);
         cudaStreamDestroy(s);
     }

     // Explicitly call destructors for libcu++ atomics before freeing memory
     for (size_t i = 0; i < NUM_SLOTS; ++i) {
         rx_flags_host[i].~atomic_uint64_sys();
         tx_flags_host[i].~atomic_uint64_sys();
     }

     cudaFreeHost(buf_rx);
     cudaFreeHost(buf_tx);
     cudaFreeHost(rx_data_host);
     cudaFreeHost(h_mailbox_bank);
     cudaStreamDestroy(capture_stream);
 
     std::cout << "Done.\n";
     return 0;
 }