/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/*******************************************************************************
 * Hybrid Realtime Pipeline Benchmark with AI Pre-Decoder + PyMatching
 *
 * Uses the RealtimePipeline scaffolding to hide all ring buffer, atomics,
 * and thread management. Application code only provides:
 *   1. GPU stage factory (AIPreDecoderService instances)
 *   2. CPU stage callback (PyMatching decode)
 *   3. Completion callback (timestamp recording)
 *
 * Usage: test_realtime_predecoder_w_pymatching [d7|d13|d13_r104|d21|d31] [rate_us] [duration_s]
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <atomic>
#include <memory>
#include <cstring>
#include <unistd.h>
#include <random>
#include <string>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>

#ifndef CUDA_VERSION
#define CUDA_VERSION 13000
#endif

#include "cudaq/realtime/pipeline.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include "cudaq/qec/realtime/ai_predecoder_service.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/decoder.h"

using namespace cudaq::qec;
namespace realtime_ns = cudaq::realtime;

// Portable CPU Yield
#ifndef QEC_CPU_RELAX
#if defined(__x86_64__)
#include <immintrin.h>
#define QEC_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define QEC_CPU_RELAX() __asm__ volatile("yield" ::: "memory")
#else
#define QEC_CPU_RELAX() do { } while(0)
#endif
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// =============================================================================
// Pipeline Configuration (application-level, no atomics)
// =============================================================================

constexpr size_t NUM_SLOTS = 32;

struct PipelineConfig {
    std::string label;
    int distance;
    int num_rounds;
    int meas_qubits;
    int residual_detectors;
    std::string onnx_filename;
    size_t slot_size;
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
            "d7_r7_Z", 7, 7, 72, 336,
            "model1_d7_r7_unified_Z_batch1.onnx",
            4096, 16, 16
        };
    }

    static PipelineConfig d13_r13() {
        return {
            "d13_r13_Z", 13, 13, 252, 2184,
            "predecoder_memory_d13_T13_X.onnx",
            16384, 16, 16
        };
    }

    static PipelineConfig d13_r104() {
        return {
            "d13_r104_Z", 13, 104, 252, 2184,
            "predecoder_memory_d13_T104_X.onnx",
            131072, 16, 16
        };
    }

    static PipelineConfig d21_r21() {
        return {
            "d21_r21_Z", 21, 21, 660, 9240,
            "model1_d21_r21_unified_X_batch1.onnx",
            65536, 16, 16
        };
    }

    static PipelineConfig d31_r31() {
        return {
            "d31_r31_Z", 31, 31, 1440, 29760,
            "model1_d31_r31_unified_Z_batch1.onnx",
            262144, 16, 16
        };
    }
};

// =============================================================================
// Decoder Context (application-level)
// =============================================================================

struct DecoderContext {
    std::vector<std::unique_ptr<cudaq::qec::decoder>> decoders;
    std::atomic<int> next_decoder_idx{0};
    int z_stabilizers = 0;
    int spatial_slices = 0;

    cudaq::qec::decoder* acquire_decoder() {
        thread_local int my_idx = next_decoder_idx.fetch_add(1, std::memory_order_relaxed);
        return decoders[my_idx % decoders.size()].get();
    }

    std::atomic<int64_t> total_decode_us{0};
    std::atomic<int64_t> total_worker_us{0};
    std::atomic<int> decode_count{0};
};

// =============================================================================
// Pre-launch DMA copy callback
// =============================================================================

struct PreLaunchCopyCtx {
    void*  d_trt_input;
    size_t input_size;
    void** h_ring_ptrs;
};

static void pre_launch_input_copy(void* user_data, void* slot_dev, cudaStream_t stream) {
    auto* ctx = static_cast<PreLaunchCopyCtx*>(user_data);
    ctx->h_ring_ptrs[0] = slot_dev;
    cudaMemcpyAsync(ctx->d_trt_input,
                    static_cast<uint8_t*>(slot_dev) + CUDAQ_RPC_HEADER_SIZE,
                    ctx->input_size, cudaMemcpyDeviceToDevice, stream);
}

// =============================================================================
// Worker context (passed through user_context)
// =============================================================================

struct WorkerCtx {
    AIPreDecoderService* predecoder;
    DecoderContext* decoder_ctx;
};

struct __attribute__((packed)) DecodeResponse {
    int32_t total_corrections;
    int32_t converged;
};

// =============================================================================
// Data generation
// =============================================================================

void fill_measurement_payload(int32_t* payload, int input_elements,
                              std::mt19937& rng, double error_rate = 0.01) {
    std::bernoulli_distribution err_dist(error_rate);
    for (int i = 0; i < input_elements; ++i) {
        payload[i] = err_dist(rng) ? 1 : 0;
    }
}

// =============================================================================
// Streaming Config
// =============================================================================

struct StreamingConfig {
    int rate_us = 0;
    int duration_s = 5;
    int warmup_count = 20;
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    using hrclock = std::chrono::high_resolution_clock;

    // --- Parse arguments ---
    std::string config_name = "d7";
    StreamingConfig scfg;

    if (argc > 1)
        config_name = argv[1];
    if (argc > 2 && std::isdigit(argv[2][0]))
        scfg.rate_us = std::stoi(argv[2]);
    if (argc > 3 && std::isdigit(argv[3][0]))
        scfg.duration_s = std::stoi(argv[3]);

    PipelineConfig config;
    if (config_name == "d7") {
        config = PipelineConfig::d7_r7();
    } else if (config_name == "d13") {
        config = PipelineConfig::d13_r13();
    } else if (config_name == "d13_r104") {
        config = PipelineConfig::d13_r104();
    } else if (config_name == "d21") {
        config = PipelineConfig::d21_r21();
    } else if (config_name == "d31") {
        config = PipelineConfig::d31_r31();
    } else {
        std::cerr << "Usage: " << argv[0] << " [d7|d13|d13_r104|d21|d31] [rate_us] [duration_s]\n"
                  << "  d7       - distance 7, 7 rounds (default)\n"
                  << "  d13      - distance 13, 13 rounds\n"
                  << "  d13_r104 - distance 13, 104 rounds\n"
                  << "  d21      - distance 21, 21 rounds\n"
                  << "  d31      - distance 31, 31 rounds\n"
                  << "  rate_us    - inter-arrival time in us (0 = open-loop)\n"
                  << "  duration_s - test duration in seconds (default: 5)\n";
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

    // --- Model path ---
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
    }

    // --- Create PyMatching decoders ---
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
              << " PyMatching decoders...\n";
    for (int i = 0; i < config.num_workers; ++i)
        decoder_ctx.decoders.push_back(
            cudaq::qec::decoder::get("pymatching", H_z, pm_params));
    std::cout << "[Setup] PyMatching decoder pool ready.\n";

    // --- Create GPU resources (predecoders, streams, mailbox) ---
    void** h_mailbox_bank = nullptr;
    void** d_mailbox_bank = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_mailbox_bank,
        config.num_predecoders * sizeof(void*), cudaHostAllocMapped));
    std::memset(h_mailbox_bank, 0, config.num_predecoders * sizeof(void*));
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&d_mailbox_bank), h_mailbox_bank, 0));

    std::vector<cudaStream_t> predecoder_streams;
    for (int i = 0; i < config.num_predecoders; ++i) {
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));
        predecoder_streams.push_back(s);
    }

    std::cout << "[Setup] Capturing " << config.num_predecoders
              << "x AIPreDecoder Graphs...\n";
    cudaStream_t capture_stream;
    CUDA_CHECK(cudaStreamCreate(&capture_stream));

    std::vector<std::unique_ptr<AIPreDecoderService>> predecoders;
    bool need_save = (model_path == onnx_file);
    for (int i = 0; i < config.num_predecoders; ++i) {
        std::string save_path = (need_save && i == 0) ? engine_file : "";
        auto pd = std::make_unique<AIPreDecoderService>(
            model_path, d_mailbox_bank + i, 1, save_path);
        std::cout << "[Setup] Decoder " << i
                  << ": input_size=" << pd->get_input_size()
                  << " output_size=" << pd->get_output_size() << "\n";
        pd->capture_graph(capture_stream, false);
        predecoders.push_back(std::move(pd));
    }

    // Pre-launch DMA contexts
    std::vector<PreLaunchCopyCtx> pre_launch_ctxs(config.num_predecoders);
    for (int i = 0; i < config.num_predecoders; ++i) {
        pre_launch_ctxs[i].d_trt_input = predecoders[i]->get_trt_input_ptr();
        pre_launch_ctxs[i].input_size  = predecoders[i]->get_input_size();
        pre_launch_ctxs[i].h_ring_ptrs = predecoders[i]->get_host_ring_ptrs();
    }

    // Worker contexts (per-worker, application-specific)
    std::vector<WorkerCtx> worker_ctxs(config.num_workers);
    for (int i = 0; i < config.num_workers; ++i) {
        worker_ctxs[i].predecoder   = predecoders[i].get();
        worker_ctxs[i].decoder_ctx  = &decoder_ctx;
    }

    // Build function table for RPC dispatch
    std::vector<uint32_t> function_ids(config.num_workers);
    for (int i = 0; i < config.num_workers; ++i) {
        std::string func = "predecode_target_" + std::to_string(i);
        function_ids[i] = realtime_ns::fnv1a_hash(func.c_str());
    }

    // =========================================================================
    // Create pipeline (all atomics hidden inside)
    // =========================================================================

    realtime_ns::PipelineStageConfig stage_cfg;
    stage_cfg.num_workers = config.num_workers;
    stage_cfg.num_slots   = NUM_SLOTS;
    stage_cfg.slot_size   = config.slot_size;
    stage_cfg.cores       = {.dispatcher = 2, .consumer = 4, .worker_base = 10};

    realtime_ns::RealtimePipeline pipeline(stage_cfg);

    // --- GPU stage factory ---
    pipeline.set_gpu_stage([&](int w) -> realtime_ns::GpuWorkerResources {
        return {
            .graph_exec     = predecoders[w]->get_executable_graph(),
            .stream          = predecoder_streams[w],
            .pre_launch_fn   = pre_launch_input_copy,
            .pre_launch_data = &pre_launch_ctxs[w],
            .function_id     = function_ids[w],
            .user_context    = &worker_ctxs[w]
        };
    });

    // --- CPU stage callback (poll + PyMatching decode) ---
    // Called repeatedly by the pipeline's worker thread.
    // Returns 0 if GPU isn't ready, >0 when a job was processed.
    pipeline.set_cpu_stage([](const realtime_ns::CpuStageContext& ctx) -> size_t {
        auto* wctx = static_cast<WorkerCtx*>(ctx.user_context);
        auto* pd = wctx->predecoder;
        auto* dctx = wctx->decoder_ctx;

        PreDecoderJob job;
        if (!pd->poll_next_job(job))
            return 0;  // GPU not done yet

        using hrclock = std::chrono::high_resolution_clock;
        auto worker_start = hrclock::now();

        int total_corrections = 0;
        bool all_converged = true;

        auto decode_start = hrclock::now();
#if !defined(DISABLE_PYMATCHING)
        const int32_t* residual = static_cast<const int32_t*>(job.inference_data);
        auto* my_decoder = dctx->acquire_decoder();

        cudaqx::tensor<uint8_t> syndrome_tensor({(size_t)dctx->z_stabilizers});
        uint8_t* syn_data = syndrome_tensor.data();

        for (int s = 0; s < dctx->spatial_slices; ++s) {
            const int32_t* slice = residual + s * dctx->z_stabilizers;
            for (int i = 0; i < dctx->z_stabilizers; ++i)
                syn_data[i] = static_cast<uint8_t>(slice[i]);

            auto result = my_decoder->decode(syndrome_tensor);
            all_converged &= result.converged;
            for (auto v : result.result)
                if (v > 0.5) total_corrections++;
        }
#endif
        auto decode_end = hrclock::now();

        // Write RPC response into ring buffer slot
        DecodeResponse resp{total_corrections, all_converged ? 1 : 0};
        char* response_payload = (char*)job.ring_buffer_ptr + sizeof(realtime_ns::RPCResponse);
        std::memcpy(response_payload, &resp, sizeof(resp));

        auto* header = static_cast<realtime_ns::RPCResponse*>(job.ring_buffer_ptr);
        header->magic = realtime_ns::RPC_MAGIC_RESPONSE;
        header->status = 0;
        header->result_len = sizeof(resp);

        pd->release_job(job.slot_idx);

        auto worker_end = hrclock::now();
        auto decode_us = std::chrono::duration_cast<std::chrono::microseconds>(
            decode_end - decode_start).count();
        auto worker_us = std::chrono::duration_cast<std::chrono::microseconds>(
            worker_end - worker_start).count();
        dctx->total_decode_us.fetch_add(decode_us, std::memory_order_relaxed);
        dctx->total_worker_us.fetch_add(worker_us, std::memory_order_relaxed);
        dctx->decode_count.fetch_add(1, std::memory_order_relaxed);

        return 1;
    });

    // --- Completion callback (record timestamps) ---
    const int max_requests = 500000;
    std::vector<hrclock::time_point> submit_ts(max_requests);
    std::vector<hrclock::time_point> complete_ts(max_requests);
    std::vector<bool> completed(max_requests, false);

    pipeline.set_completion_handler([&](const realtime_ns::Completion& c) {
        if (c.request_id < static_cast<uint64_t>(max_requests)) {
            complete_ts[c.request_id] = hrclock::now();
            completed[c.request_id] = c.success;
        }
    });

    // =========================================================================
    // Start pipeline and run producer
    // =========================================================================

    std::cout << "[Setup] Starting pipeline...\n";
    pipeline.start();

    auto run_deadline = std::chrono::steady_clock::now()
                      + std::chrono::seconds(scfg.duration_s);

    std::string rate_label = (scfg.rate_us > 0)
        ? std::to_string(scfg.rate_us) + " us" : "open-loop";

    std::cout << "\n[Stream] Starting streaming test (" << config.label << ")\n"
              << "  Rate:       " << rate_label << "\n"
              << "  Duration:   " << scfg.duration_s << " s\n"
              << "  Warmup:     " << scfg.warmup_count << " requests\n"
              << "  Predecoders:" << config.num_predecoders << " (dedicated streams)\n"
              << "  Max reqs:   " << max_requests << "\n\n" << std::flush;

    // --- Producer loop (runs on main thread) ---
    std::mt19937 rng(42);
    const size_t payload_bytes = std::min(
        config.input_bytes(),
        config.slot_size - static_cast<size_t>(CUDAQ_RPC_HEADER_SIZE));
    std::vector<uint8_t> payload_buf(CUDAQ_RPC_HEADER_SIZE + payload_bytes);
    int req_id = 0;
    int target = 0;

    while (std::chrono::steady_clock::now() < run_deadline
           && req_id < max_requests) {

        int32_t* payload = reinterpret_cast<int32_t*>(
            payload_buf.data() + CUDAQ_RPC_HEADER_SIZE);
        int fill_elems = static_cast<int>(payload_bytes / sizeof(int32_t));
        fill_measurement_payload(payload, fill_elems, rng, 0.01);

        std::string func = "predecode_target_" + std::to_string(target);
        uint32_t fid = realtime_ns::fnv1a_hash(func.c_str());

        submit_ts[req_id] = hrclock::now();
        pipeline.submit(fid, payload, static_cast<uint32_t>(payload_bytes),
                        static_cast<uint64_t>(req_id));

        target = (target + 1) % config.num_predecoders;
        req_id++;

        if (scfg.rate_us > 0) {
            auto target_time = submit_ts[req_id - 1]
                             + std::chrono::microseconds(scfg.rate_us);
            while (hrclock::now() < target_time)
                QEC_CPU_RELAX();
        }
    }

    // --- Shutdown ---
    pipeline.stop();

    // =========================================================================
    // Report
    // =========================================================================

    auto final_stats = pipeline.stats();
    uint64_t nsub = final_stats.submitted;
    uint64_t ncomp = final_stats.completed;

    if (ncomp < nsub)
        std::cerr << "  [WARN] " << (nsub - ncomp)
                  << " requests did not complete.\n";

    int warmup = std::min(scfg.warmup_count, static_cast<int>(nsub));
    std::vector<double> latencies;
    latencies.reserve(nsub - warmup);

    for (uint64_t i = warmup; i < nsub; ++i) {
        if (!completed[i]) continue;
        auto dt = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
            complete_ts[i] - submit_ts[i]);
        latencies.push_back(dt.count());
    }

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
        std::chrono::steady_clock::now() -
        (run_deadline - std::chrono::seconds(scfg.duration_s))).count();
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
    std::cout << std::setprecision(1);
    std::cout << "  Wall time:          " << wall_us / 1000.0 << " ms\n";
    std::cout << "  Throughput:         " << throughput << " req/s\n";
    std::cout << "  Actual arrival rate:" << std::setw(8) << actual_rate << " us/req\n";
    std::cout << "  Backpressure stalls:" << std::setw(8)
              << final_stats.backpressure_stalls << "\n";
    std::cout << "  ---------------------------------------------------------------\n";
    std::cout << "  Latency (us)  [steady-state, " << latencies.size()
              << " requests after " << warmup << " warmup]\n";
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

    int n_decoded = decoder_ctx.decode_count.load();
    if (n_decoded > 0) {
        double avg_decode = (double)decoder_ctx.total_decode_us.load() / n_decoded;
        double avg_worker = (double)decoder_ctx.total_worker_us.load() / n_decoded;
        double avg_overhead = avg_worker - avg_decode;
        std::cout << "  ---------------------------------------------------------------\n";
        std::cout << "  Worker-level averages (" << n_decoded << " completed):\n";
        std::cout << "    PyMatching decode:    " << std::setw(9) << avg_decode << " us\n";
        std::cout << "    Total worker:         " << std::setw(9) << avg_worker << " us\n";
        std::cout << "    Worker overhead:      " << std::setw(9) << avg_overhead << " us\n";
    }

    std::cout << "  ---------------------------------------------------------------\n";
    std::cout << "  Host dispatcher processed " << final_stats.dispatched << " packets.\n";
    std::cout << "================================================================\n";

    // --- Cleanup ---
    std::cout << "[Teardown] Shutting down...\n";
    CUDA_CHECK(cudaStreamSynchronize(capture_stream));
    for (auto& s : predecoder_streams) {
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
    }
    cudaFreeHost(h_mailbox_bank);
    cudaStreamDestroy(capture_stream);

    std::cout << "Done.\n";
    return 0;
}
