/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include <atomic>
#include <cuda/atomic>
#include <iosfwd>
#include <string>
#include <vector>

// Portable CPU Yield Macro for busy-polling (skip if already defined by
// realtime API)
#ifndef CUDAQ_REALTIME_CPU_RELAX
#if defined(__x86_64__)
#include <immintrin.h>
#define CUDAQ_REALTIME_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define CUDAQ_REALTIME_CPU_RELAX() __asm__ volatile("yield" ::: "memory")
#else
#define CUDAQ_REALTIME_CPU_RELAX()                                             \
  std::atomic_thread_fence(std::memory_order_seq_cst)
#endif
#endif

namespace cudaq::qec::realtime::experimental {

/// @brief Per-kernel resource usage captured from the predecoder CUDA graph.
struct kernel_resource_info {
  std::string name;          ///< Kernel symbol name (demangled if available).
  dim3 grid_dim;             ///< Grid dimensions from the graph node.
  dim3 block_dim;            ///< Block dimensions from the graph node.
  size_t static_shmem;       ///< Static shared memory per block (bytes).
  size_t dynamic_shmem;      ///< Dynamic shared memory per block (bytes).
  size_t local_mem;          ///< Local memory per thread (bytes).
  size_t const_mem;          ///< Constant memory used by the kernel (bytes).
  int num_regs;              ///< Registers per thread.
  int max_threads_per_block; ///< Hardware max threads for this kernel.
};

/// @brief Aggregate resource usage for the predecoder CUDA graph.
struct graph_resource_info {
  size_t total_nodes = 0;
  size_t kernel_nodes = 0;
  size_t memcpy_nodes = 0;
  size_t host_nodes = 0;
  size_t other_nodes = 0;
  std::vector<kernel_resource_info> kernels;
};

struct pre_decoder_job {
  int slot_idx;    ///< Worker/slot index (for release_job; always 0)
  int origin_slot; ///< FPGA ring slot for tx_flags routing (dynamic pool)
  void *ring_buffer_ptr;
  void *inference_data; ///< Points into the pinned output (single slot)

  // Performance Tracking
  uint64_t submit_ts_ns;
  uint64_t dispatch_ts_ns;
  uint64_t poll_ts_ns;
};

class ai_predecoder_service : public ai_decoder_service {
public:
  ai_predecoder_service(const std::string &engine_path,
                        void **device_mailbox_slot, int queue_depth = 1,
                        const std::string &engine_save_path = "");

  /// Create a passthrough (identity copy) instance for testing without TRT.
  static std::unique_ptr<ai_predecoder_service>
  create_passthrough(void **device_mailbox_slot, int queue_depth = 1,
                     size_t input_bytes = 1600 * sizeof(float),
                     size_t output_bytes = 1600 * sizeof(float));

  virtual ~ai_predecoder_service();

  /// @param stream CUDA stream to use for capture and warm-up inference.
  /// @param device_launch If true, instantiate the graph for device launch.
  /// @param collect_resources If true, walk the captured graph and populate
  ///        graph_resources_ for later inspection via
  ///        get_graph_resources() / print_graph_resources().  Off by default
  ///        because graph introspection can perturb CUDA context state in
  ///        ways that interfere with DOCA/Hololink GPU-RoCE setup on the
  ///        FPGA bridge path.
  void capture_graph(cudaStream_t stream, bool device_launch,
                     bool collect_resources = false);
  void capture_graph(cudaStream_t stream) override {
    capture_graph(stream, true, false);
  }

  bool poll_next_job(pre_decoder_job &out_job);
  void release_job(int slot_idx);

  /// Stub for device-dispatcher batch path (returns nullptr; streaming uses
  /// host dispatcher)
  int *get_device_queue_idx() const { return nullptr; }
  cuda::atomic<int, cuda::thread_scope_system> *get_device_ready_flags() const {
    return d_ready_flags_;
  }
  int *get_device_inflight_flag() const { return nullptr; }

  cuda::atomic<int, cuda::thread_scope_system> *get_host_ready_flags() const {
    return h_ready_flags_;
  }
  volatile int *get_host_queue_idx() const { return nullptr; }
  int get_queue_depth() const { return queue_depth_; }

  void **get_host_ring_ptrs() const { return h_ring_ptrs_; }

  /// @brief Return the resource usage snapshot collected during capture_graph.
  /// @details Fields are zero until capture_graph() has been called.
  const graph_resource_info &get_graph_resources() const {
    return graph_resources_;
  }

  /// @brief Print resource usage of the captured predecoder CUDA graph.
  /// @param os Output stream to write to (e.g. std::cout).
  /// @details Prints a per-kernel breakdown followed by aggregate totals
  /// (node counts, total registers across launches, total shared memory,
  /// total threads).  Safe to call only after capture_graph() has run.
  void print_graph_resources(std::ostream &os) const;

private:
  /// Passthrough constructor (delegates to base passthrough constructor).
  ai_predecoder_service(void **device_mailbox_slot, int queue_depth,
                        size_t input_bytes, size_t output_bytes);

  /// Walk the captured CUDA graph and populate graph_resources_.
  void collect_graph_resources(cudaGraph_t graph);

  int queue_depth_; // Always 1

  cuda::atomic<int, cuda::thread_scope_system> *h_ready_flags_ = nullptr;
  void **h_ring_ptrs_ = nullptr;
  void *h_predecoder_outputs_ = nullptr;

  cuda::atomic<int, cuda::thread_scope_system> *d_ready_flags_ = nullptr;
  void **d_ring_ptrs_ = nullptr;
  void *d_predecoder_outputs_ = nullptr;

  graph_resource_info graph_resources_;
};

} // namespace cudaq::qec::realtime::experimental
