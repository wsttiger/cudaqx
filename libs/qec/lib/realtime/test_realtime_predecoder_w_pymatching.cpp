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
 * Usage: test_realtime_predecoder_w_pymatching [d7|d13|d21|d31] [stream [rate_us] [duration_s]]
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
#include <fstream>

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
    auto* my_decoder = ctx->acquire_decoder();

    int total_corrections = 0;
    bool all_converged = true;

    auto decode_start = hrclock::now();
    for (int s = 0; s < ctx->spatial_slices; ++s) {
        const int32_t* slice = residual + s * ctx->z_stabilizers;
        std::vector<double> syndrome(ctx->z_stabilizers);
        for (int i = 0; i < ctx->z_stabilizers; ++i)
            syndrome[i] = static_cast<double>(slice[i]);

        auto result = my_decoder->decode(syndrome);

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
    volatile uint64_t* rx_flags_host,
    volatile uint64_t* tx_flags_host,
    uint8_t* rx_data_host,
    DecoderContext& decoder_ctx,
    std::vector<std::unique_ptr<AIPreDecoderService>>& predecoders,
    cudaq::qec::utils::ThreadPool& pymatching_pool,
    std::atomic<bool>& system_stop)
{
    using hrclock = std::chrono::high_resolution_clock;

    const int max_requests = 500000;
    const size_t payload_bytes = config.input_bytes();

    std::vector<hrclock::time_point> submit_ts(max_requests);
    std::vector<hrclock::time_point> complete_ts(max_requests);
    std::vector<bool> completed(max_requests, false);

    // slot -> request_id mapping so consumer can correlate completions
    std::vector<int> slot_request(NUM_SLOTS, -1);

    std::atomic<int> total_submitted{0};
    std::atomic<int> total_completed{0};
    std::atomic<int> in_flight{0};
    std::atomic<int64_t> backpressure_stalls{0};
    std::atomic<bool> producer_done{false};

    // Cap in-flight to num_predecoders. The dispatcher scans slots
    // sequentially and only advances on non-empty slots. With the inflight
    // flag limiting one graph launch per predecoder, only num_predecoders
    // slots can be consumed per scan. Any excess slots get backpressured,
    // then the dispatcher parks on an empty slot and never revisits them.
    const int max_in_flight = config.num_predecoders;

    auto run_deadline = std::chrono::steady_clock::now()
                      + std::chrono::seconds(scfg.duration_s);

    std::string rate_label = (scfg.rate_us > 0)
        ? std::to_string(scfg.rate_us) + " us"
        : "open-loop";

    std::cout << "\n[Stream] Starting streaming test (" << config.label << ")\n"
              << "  Rate:       " << rate_label << "\n"
              << "  Duration:   " << scfg.duration_s << " s\n"
              << "  Warmup:     " << scfg.warmup_count << " requests\n"
              << "  Max flight: " << max_in_flight << "\n"
              << "  Max reqs:   " << max_requests << "\n\n";

    // --- Producer thread (simulates FPGA) ---
    std::thread producer([&]() {
        std::mt19937 rng(42);
        int next_slot = 0;
        int req_id = 0;

        while (std::chrono::steady_clock::now() < run_deadline
               && req_id < max_requests) {

            // Throttle: don't exceed max_in_flight to prevent ring buffer flooding
            while (in_flight.load(std::memory_order_acquire) >= max_in_flight) {
                QEC_CPU_RELAX();
                if (std::chrono::steady_clock::now() >= run_deadline) return;
            }

            int slot = next_slot % (int)NUM_SLOTS;

            // Wait for slot to be fully free (dispatcher consumed + response harvested)
            while (rx_flags_host[slot] != 0 || tx_flags_host[slot] != 0) {
                backpressure_stalls.fetch_add(1, std::memory_order_relaxed);
                QEC_CPU_RELAX();
                if (std::chrono::steady_clock::now() >= run_deadline) return;
            }

            int target = req_id % config.num_predecoders;
            std::string func = "predecode_target_" + std::to_string(target);

            uint8_t* slot_data = rx_data_host + (slot * config.slot_size);
            auto* hdr = reinterpret_cast<cudaq::nvqlink::RPCHeader*>(slot_data);
            hdr->magic = cudaq::nvqlink::RPC_MAGIC_REQUEST;
            hdr->function_id = fnv1a_hash(func);
            hdr->arg_len = static_cast<uint32_t>(payload_bytes);

            int32_t* payload = reinterpret_cast<int32_t*>(
                slot_data + sizeof(cudaq::nvqlink::RPCHeader));
            fill_measurement_payload(payload, config.input_elements(), rng, 0.01);

            slot_request[slot] = req_id;

            __sync_synchronize();
            submit_ts[req_id] = hrclock::now();
            rx_flags_host[slot] = reinterpret_cast<uint64_t>(slot_data);
            in_flight.fetch_add(1, std::memory_order_release);
            total_submitted.fetch_add(1, std::memory_order_release);

            next_slot++;
            req_id++;

            // Rate limiting (busy-wait for precision)
            if (scfg.rate_us > 0) {
                auto target_time = submit_ts[req_id - 1]
                                 + std::chrono::microseconds(scfg.rate_us);
                while (hrclock::now() < target_time)
                    QEC_CPU_RELAX();
            }
        }

        producer_done.store(true, std::memory_order_release);
    });

    // --- Consumer thread (harvests completions sequentially) ---
    std::thread consumer([&]() {
        int next_harvest = 0;

        while (true) {
            bool pdone = producer_done.load(std::memory_order_acquire);
            int nsub = total_submitted.load(std::memory_order_acquire);
            int ncomp = total_completed.load(std::memory_order_relaxed);

            if (pdone && ncomp >= nsub)
                break;

            // Nothing to harvest yet
            if (next_harvest >= nsub) {
                QEC_CPU_RELAX();
                continue;
            }

            int slot = next_harvest % (int)NUM_SLOTS;
            uint64_t tv = tx_flags_host[slot];

            if (tv != 0) {
                int rid = slot_request[slot];
                if (rid >= 0 && (tv >> 48) != 0xDEAD) {
                    complete_ts[rid] = hrclock::now();
                    completed[rid] = true;
                    total_completed.fetch_add(1, std::memory_order_relaxed);
                } else if ((tv >> 48) == 0xDEAD) {
                    int cuda_err = (int)(tv & 0xFFFF);
                    std::cerr << "  [FAIL] Slot " << slot
                              << " cudaGraphLaunch error " << cuda_err
                              << " (" << cudaGetErrorString((cudaError_t)cuda_err)
                              << ")\n";
                    total_completed.fetch_add(1, std::memory_order_relaxed);
                }

                tx_flags_host[slot] = 0;
                slot_request[slot] = -1;
                in_flight.fetch_sub(1, std::memory_order_release);
                next_harvest++;
            } else {
                QEC_CPU_RELAX();
            }
        }
    });

    producer.join();

    // Grace period for in-flight requests
    auto grace_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (total_completed.load() < total_submitted.load()
           && std::chrono::steady_clock::now() < grace_deadline) {
        usleep(1000);
    }

    consumer.join();

    // ===== Report =====
    auto run_end = std::chrono::steady_clock::now();
    int nsub = total_submitted.load();
    int ncomp = total_completed.load();

    // Build PipelineBenchmark from timestamps (skip warmup)
    int warmup = std::min(scfg.warmup_count, nsub);
    int bench_count = nsub - warmup;

    cudaq::qec::utils::PipelineBenchmark bench(
        config.label + " (stream)", bench_count);
    bench.start();

    for (int i = warmup; i < nsub; ++i) {
        int bench_id = i - warmup;
        bench.mark_submit(bench_id);
        // Override the internal submit timestamp with the real one
    }

    // We can't override PipelineBenchmark's internal timestamps, so compute
    // stats manually for the steady-state window.
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
        double avg_pipeline = mean - avg_worker;

        std::cout << std::setprecision(1);
        std::cout << "  Worker Timing Breakdown (avg over " << n_decoded << " requests):\n";
        std::cout << "    PyMatching decode:" << std::setw(10) << avg_decode
                  << " us  (" << std::setw(4) << (mean > 0 ? 100.0 * avg_decode / mean : 0)
                  << "%)\n";
        std::cout << "    Worker overhead:  " << std::setw(10) << avg_overhead
                  << " us  (" << std::setw(4) << (mean > 0 ? 100.0 * avg_overhead / mean : 0)
                  << "%)\n";
        std::cout << "    GPU+dispatch+poll:" << std::setw(10) << avg_pipeline
                  << " us  (" << std::setw(4) << (mean > 0 ? 100.0 * avg_pipeline / mean : 0)
                  << "%)\n";
        std::cout << "    Total end-to-end: " << std::setw(10) << mean << " us\n";
        std::cout << "    Per-round (/" << config.num_rounds << "): "
                  << std::setw(10) << (mean / config.num_rounds) << " us/round\n";
    }
    std::cout << "================================================================\n";
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
    // Parse arguments: <config> [stream [rate_us] [duration_s]]
    std::string config_name = "d7";
    bool streaming_mode = false;
    StreamingConfig stream_cfg;

    if (argc > 1)
        config_name = argv[1];

    int stream_positional = 0; // tracks positional args after "stream"
    for (int a = 2; a < argc; ++a) {
        std::string arg = argv[a];
        if (arg == "stream") {
            streaming_mode = true;
        } else if (streaming_mode && stream_positional == 0 && std::isdigit(arg[0])) {
            stream_cfg.rate_us = std::stoi(arg);
            stream_positional++;
        } else if (streaming_mode && stream_positional == 1 && std::isdigit(arg[0])) {
            stream_cfg.duration_s = std::stoi(arg);
            stream_positional++;
        }
    }

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
        std::cerr << "Usage: " << argv[0] << " [d7|d13|d21|d31] [stream [rate_us] [duration_s]]\n"
                  << "  d7     - distance 7, 7 rounds (default)\n"
                  << "  d13    - distance 13, 13 rounds\n"
                  << "  d21    - distance 21, 21 rounds\n"
                  << "  d31    - distance 31, 31 rounds\n"
                  << "\n"
                  << "  stream - continuous FPGA-like submission (default: batch mode)\n"
                  << "  rate_us  - inter-arrival time in us (0 = open-loop, default)\n"
                  << "  duration_s - test duration in seconds (default: 5)\n"
                  << "\n"
                  << "Examples:\n"
                  << "  " << argv[0] << " d13              # batch mode\n"
                  << "  " << argv[0] << " d13 stream       # streaming, open-loop\n"
                  << "  " << argv[0] << " d13 stream 50    # streaming, 50 us between requests\n"
                  << "  " << argv[0] << " d13 stream 50 10 # streaming, 50 us rate, 10s duration\n";
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

    // Prefer cached .engine file; fall back to ONNX build + save
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
    std::cout << "[Setup] Pre-allocating " << config.num_workers
              << " PyMatching decoders (one per worker)...\n";
    for (int i = 0; i < config.num_workers; ++i)
        decoder_ctx.decoders.push_back(
            cudaq::qec::decoder::get("pymatching", H_z, pm_params));
    std::cout << "[Setup] PyMatching decoder pool ready.\n";

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
              << "x AIPreDecoder Graphs...\n";
    cudaStream_t capture_stream;
    CUDA_CHECK(cudaStreamCreate(&capture_stream));

    std::vector<std::unique_ptr<AIPreDecoderService>> predecoders;
    std::vector<cudaq_function_entry_t> function_entries(config.num_predecoders);

    bool need_save = (model_path == onnx_file);
    for (int i = 0; i < config.num_predecoders; ++i) {
        void** my_mailbox = d_global_mailbox_bank + i;
        std::string save_path = (need_save && i == 0) ? engine_file : "";
        auto pd = std::make_unique<AIPreDecoderService>(model_path, my_mailbox,
                                                         config.queue_depth,
                                                         save_path);

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
    // Test Stimulus
    // =========================================================================
    if (streaming_mode) {
        run_streaming_test(config, stream_cfg, rx_flags_host, tx_flags_host,
                           rx_data_host, decoder_ctx, predecoders,
                           pymatching_pool, system_stop);
    } else {
        // Batch mode: fire requests in batches of num_predecoders, wait for
        // each batch to complete before firing the next.
        const int batch_size = config.num_predecoders;
        std::cout << "\n[Batch] Firing " << config.total_requests
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

            for (int i = batch_start; i < batch_end; ++i) {
                int target_decoder = i % config.num_predecoders;
                std::string target_func = "predecode_target_"
                                        + std::to_string(target_decoder);

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

            for (int i = batch_start; i < batch_end; ++i) {
                int slot = i % (int)NUM_SLOTS;

                auto deadline = std::chrono::steady_clock::now()
                              + std::chrono::seconds(10);
                while (tx_flags_host[slot] == 0) {
                    if (std::chrono::steady_clock::now() > deadline) break;
                    QEC_CPU_RELAX();
                }

                uint64_t tv = tx_flags_host[slot];
                if (tv != 0 && (tv >> 48) == 0xDEAD) {
                    int cuda_err = (int)(tv & 0xFFFF);
                    std::cerr << "  [FAIL] Slot " << slot
                              << " cudaGraphLaunch error " << cuda_err
                              << " (" << cudaGetErrorString((cudaError_t)cuda_err)
                              << ")\n";
                } else if (tv != 0) {
                    bench.mark_complete(i);
                    responses_received++;
                    uint8_t* slot_data = rx_data_host + (slot * config.slot_size);
                    int32_t corrections = 0, converged = 0;
                    std::memcpy(&corrections,
                                slot_data + sizeof(cudaq::nvqlink::RPCResponse),
                                sizeof(int32_t));
                    std::memcpy(&converged,
                                slot_data + sizeof(cudaq::nvqlink::RPCResponse)
                                    + sizeof(int32_t),
                                sizeof(int32_t));
                    std::cout << "  -> Slot " << slot
                              << ": OK, corrections=" << corrections
                              << " converged=" << (converged ? "yes" : "no") << "\n";
                } else {
                    std::cerr << "  [FAIL] Timeout waiting for slot " << slot << "\n";
                }

                tx_flags_host[slot] = 0;
            }
        }

        bench.stop();

        std::cout << "\n[Result] Processed " << responses_received << "/"
                  << requests_sent << " requests successfully.\n";

        bench.report();

        int n_decoded = decoder_ctx.decode_count.load();
        if (n_decoded > 0) {
            double avg_decode = (double)decoder_ctx.total_decode_us.load() / n_decoded;
            double avg_worker = (double)decoder_ctx.total_worker_us.load() / n_decoded;
            double avg_overhead = avg_worker - avg_decode;
            auto stats = bench.compute_stats();
            double avg_pipeline_overhead = stats.mean_us - avg_worker;

            std::cout << std::fixed << std::setprecision(1);
            std::cout << "\n  Worker Timing Breakdown (avg over "
                      << n_decoded << " requests):\n";
            std::cout << "    PyMatching decode:   " << std::setw(8) << avg_decode
                      << " us  (" << std::setw(4)
                      << (100.0 * avg_decode / stats.mean_us) << "%)\n";
            std::cout << "    Worker overhead:      " << std::setw(8) << avg_overhead
                      << " us  (" << std::setw(4)
                      << (100.0 * avg_overhead / stats.mean_us) << "%)\n";
            std::cout << "    GPU+dispatch+poll:    " << std::setw(8)
                      << avg_pipeline_overhead << " us  (" << std::setw(4)
                      << (100.0 * avg_pipeline_overhead / stats.mean_us) << "%)\n";
            std::cout << "    Total end-to-end:     " << std::setw(8)
                      << stats.mean_us << " us\n";
            std::cout << "    Per-round (/" << config.num_rounds << "):     "
                      << std::setw(8) << (stats.mean_us / config.num_rounds)
                      << " us/round\n";
        }
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
