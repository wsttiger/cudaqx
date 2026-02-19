/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace cudaq::qec::utils {

/// Reusable latency / throughput tracker for realtime decoding pipelines.
///
/// Usage:
///   PipelineBenchmark bench("my test", num_requests);
///   bench.start();
///   for (int i = 0; i < n; ++i) {
///       bench.mark_submit(i);
///       // ... submit request ...
///       // ... wait for response ...
///       bench.mark_complete(i);
///   }
///   bench.stop();
///   bench.report();
///
class PipelineBenchmark {
public:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;
    using duration_us = std::chrono::duration<double, std::micro>;

    explicit PipelineBenchmark(const std::string &label = "Pipeline",
                               size_t expected_requests = 0)
        : label_(label) {
        if (expected_requests > 0) {
            submit_times_.resize(expected_requests);
            complete_times_.resize(expected_requests);
        }
    }

    void start() { run_start_ = clock::now(); }
    void stop() { run_end_ = clock::now(); }

    void mark_submit(int request_id) {
        ensure_capacity(request_id);
        submit_times_[request_id] = clock::now();
    }

    void mark_complete(int request_id) {
        ensure_capacity(request_id);
        complete_times_[request_id] = clock::now();
    }

    struct Stats {
        size_t count = 0;
        double min_us = 0, max_us = 0, mean_us = 0;
        double p50_us = 0, p90_us = 0, p95_us = 0, p99_us = 0;
        double stddev_us = 0;
        double total_wall_us = 0;
        double throughput_rps = 0;
    };

    /// Return per-request latencies in microseconds.
    std::vector<double> latencies_us() const {
        size_t n = std::min(submit_times_.size(), complete_times_.size());
        std::vector<double> lats;
        lats.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            auto dt = std::chrono::duration_cast<duration_us>(
                complete_times_[i] - submit_times_[i]);
            lats.push_back(dt.count());
        }
        return lats;
    }

    Stats compute_stats() const {
        auto lats = latencies_us();
        Stats s;
        s.count = lats.size();
        if (s.count == 0)
            return s;

        std::sort(lats.begin(), lats.end());

        s.min_us = lats.front();
        s.max_us = lats.back();
        s.mean_us =
            std::accumulate(lats.begin(), lats.end(), 0.0) / s.count;
        s.p50_us = percentile(lats, 50.0);
        s.p90_us = percentile(lats, 90.0);
        s.p95_us = percentile(lats, 95.0);
        s.p99_us = percentile(lats, 99.0);

        double sum_sq = 0;
        for (auto v : lats)
            sum_sq += (v - s.mean_us) * (v - s.mean_us);
        s.stddev_us = std::sqrt(sum_sq / s.count);

        auto wall =
            std::chrono::duration_cast<duration_us>(run_end_ - run_start_);
        s.total_wall_us = wall.count();
        s.throughput_rps =
            (s.total_wall_us > 0) ? (s.count * 1e6 / s.total_wall_us) : 0;

        return s;
    }

    void report(std::ostream &os = std::cout) const {
        auto s = compute_stats();
        auto lats = latencies_us();

        os << "\n";
        os << "================================================================\n";
        os << "  Benchmark: " << label_ << "\n";
        os << "================================================================\n";
        os << std::fixed;
        os << "  Requests:       " << s.count << "\n";
        os << std::setprecision(1);
        os << "  Wall time:      " << s.total_wall_us / 1000.0 << " ms\n";
        os << "  Throughput:     " << s.throughput_rps << " req/s\n";
        os << "  ---------------------------------------------------------------\n";
        os << "  Latency (us)\n";
        os << std::setprecision(1);
        os << "    min    = " << std::setw(10) << s.min_us << "\n";
        os << "    p50    = " << std::setw(10) << s.p50_us << "\n";
        os << "    mean   = " << std::setw(10) << s.mean_us << "\n";
        os << "    p90    = " << std::setw(10) << s.p90_us << "\n";
        os << "    p95    = " << std::setw(10) << s.p95_us << "\n";
        os << "    p99    = " << std::setw(10) << s.p99_us << "\n";
        os << "    max    = " << std::setw(10) << s.max_us << "\n";
        os << "    stddev = " << std::setw(10) << s.stddev_us << "\n";
        os << "  ---------------------------------------------------------------\n";

        // Per-request breakdown (compact, one line per request)
        if (!lats.empty()) {
            os << "  Per-request latencies (us):\n";
            for (size_t i = 0; i < lats.size(); ++i) {
                os << "    [" << std::setw(4) << i << "] "
                   << std::setprecision(1) << std::setw(10) << lats[i]
                   << "\n";
            }
        }
        os << "================================================================\n";
    }

private:
    std::string label_;
    time_point run_start_{}, run_end_{};
    std::vector<time_point> submit_times_;
    std::vector<time_point> complete_times_;

    void ensure_capacity(int id) {
        size_t needed = static_cast<size_t>(id) + 1;
        if (submit_times_.size() < needed)
            submit_times_.resize(needed);
        if (complete_times_.size() < needed)
            complete_times_.resize(needed);
    }

    static double percentile(const std::vector<double> &sorted, double p) {
        if (sorted.empty())
            return 0;
        double idx = (p / 100.0) * (sorted.size() - 1);
        size_t lo = static_cast<size_t>(idx);
        size_t hi = std::min(lo + 1, sorted.size() - 1);
        double frac = idx - lo;
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    }
};

} // namespace cudaq::qec::utils
