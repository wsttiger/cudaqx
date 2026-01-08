/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <atomic>
#include <memory>
#include <vector>

namespace cudaq::qec {

/// @brief Persistent AI Decoder - True GPU persistent kernel for syndrome
/// decoding
///
/// This class implements a true GPU persistent kernel pattern where the kernel
/// launches ONCE and stays resident on the GPU, continuously processing
/// syndromes as they become available.
///
/// Key Innovation: One CUDA graph per work slot
/// ===============================================
/// Each work slot has its own captured CUDA graph. The graph was captured with
/// that slot's specific input/output buffers, so we can just launch it without
/// needing to update parameters or call TensorRT APIs.
///
/// Architecture:
/// - GPU kernel launches once and runs continuously
/// - Work scheduling happens on the GPU (device-side)
/// - Each work slot has a pre-captured CUDA graph for TensorRT inference
/// - Minimal host involvement (only for enqueue/dequeue)
/// - Uses device-side work queue with atomic operations
///
/// Key characteristics:
/// - True GPU persistent kernel (kernel runs continuously)
/// - Work scheduling on device (minimal host overhead)
/// - Ultra-low latency (~10-100 μs)
/// - Requires compute capability 7.0+ for device-side CUDA graph launch
///
/// @note This implementation captures N CUDA graphs (one per work slot).
/// Memory overhead: ~1-10 KB per graph depending on model complexity.
class persistent_ai_decoder {
public:
  /// @brief Configuration for the persistent decoder
  struct Config {
    size_t syndrome_size = 0;       ///< Size of each syndrome (elements)
    size_t output_size = 0;         ///< Size of each output (elements)
    size_t num_work_slots = 16;     ///< Number of concurrent work slots
    size_t num_blocks = 8;          ///< Number of GPU blocks for persistent kernel
    size_t threads_per_block = 256; ///< Threads per block
  };

  /// @brief Work slot for device-side queue
  /// Each slot has its own captured CUDA graph with the correct buffers
  struct WorkSlot {
    float *input_buffer_device;     ///< GPU input buffer
    float *output_buffer_device;    ///< GPU output buffer
    cudaGraphExec_t graph_exec;     ///< Captured graph for this slot
    cudaStream_t stream;            ///< Stream for this slot's graph
    int status;                     ///< 0=free, 1=input_ready, 2=processing, 3=output_ready
    int padding[3];                 ///< Align to 64 bytes for cache efficiency
  };

  /// @brief Constructor
  /// @param engine TensorRT engine (must have static shapes)
  /// @param context TensorRT execution context
  /// @param input_index Index of input tensor
  /// @param output_index Index of output tensor
  /// @param config Configuration parameters
  ///
  /// Note: This will capture one CUDA graph per work slot during initialization
  persistent_ai_decoder(nvinfer1::ICudaEngine *engine,
                        nvinfer1::IExecutionContext *context,
                        int input_index,
                        int output_index,
                        const Config &config);

  /// @brief Destructor - stops the persistent kernel and cleans up resources
  ~persistent_ai_decoder();

  // Disable copy and move
  persistent_ai_decoder(const persistent_ai_decoder &) = delete;
  persistent_ai_decoder &operator=(const persistent_ai_decoder &) = delete;
  persistent_ai_decoder(persistent_ai_decoder &&) = delete;
  persistent_ai_decoder &operator=(persistent_ai_decoder &&) = delete;

  /// @brief Start the persistent GPU kernel
  /// @return true if started successfully, false otherwise
  bool start();

  /// @brief Stop the persistent GPU kernel
  void stop();

  /// @brief Check if the persistent kernel is running
  /// @return true if running, false otherwise
  bool is_running() const;

  /// @brief Enqueue a syndrome for decoding
  /// @param syndrome Input syndrome data
  /// @return true if enqueued successfully, false if queue is full
  ///
  /// This copies data to a free work slot and marks it as ready.
  /// The GPU kernel will pick it up and process it.
  bool enqueue_syndrome(const std::vector<float> &syndrome);

  /// @brief Try to dequeue a decoded result
  /// @param result Output buffer to store the result
  /// @return true if a result was available, false if queue is empty
  ///
  /// This checks for completed work slots and retrieves the result.
  bool try_dequeue_result(std::vector<float> &result);

  /// @brief Get the number of syndromes currently queued for decoding
  size_t get_queue_depth() const;

  /// @brief Get the number of results available for retrieval
  size_t get_available_results() const;

private:
  // TensorRT resources (non-owning pointers, used only during initialization)
  nvinfer1::ICudaEngine *engine_;
  nvinfer1::IExecutionContext *context_;
  int input_index_;
  int output_index_;

  // Configuration
  Config config_;

  // Device-side work queue
  WorkSlot *work_slots_device_; ///< Work slots on GPU
  WorkSlot *work_slots_host_;   ///< Mirror on host for management

  // Pinned host buffers for data transfer
  float *staging_input_host_;
  float *staging_output_host_;

  // Control flag for stopping the kernel
  int *stop_flag_device_; ///< Device-side stop flag
  int *stop_flag_host_;   ///< Host-side mirror

  // Running state
  std::atomic<bool> running_{false};

  // Private methods
  void cleanup_resources();
  bool allocate_and_capture_graphs();
  void free_resources();
  int find_free_slot();
  int find_ready_slot();
};

} // namespace cudaq::qec
