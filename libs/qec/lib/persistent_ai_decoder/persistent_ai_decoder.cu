/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/persistent_ai_decoder.h"
#include "cudaq/qec/cuda_graph_utils.h"
#include "common/Logger.h"
#include <cooperative_groups.h>
#include <cuda_device_runtime_api.h>
#include <cstring>

namespace cg = cooperative_groups;

// Remove the leading path elements from a fully qualified filename
static inline void trim_filename(std::string &filename) {
  std::size_t lastSlashPos = filename.find_last_of('/');
  if (lastSlashPos != std::string::npos)
    filename = filename.substr(lastSlashPos + 1);
}

#ifndef HANDLE_CUDA_ERROR
#define HANDLE_CUDA_ERROR(x)                                                   \
  do {                                                                         \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      std::string filename = __FILE__;                                         \
      trim_filename(filename);                                                 \
      printf("CUDA ERROR %s:%d: '%s'\n", filename.c_str(), __LINE__,           \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

#ifndef HANDLE_CUDA_ERROR_NO_THROW
#define HANDLE_CUDA_ERROR_NO_THROW(x)                                          \
  do {                                                                         \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      std::string filename = __FILE__;                                         \
      trim_filename(filename);                                                 \
      printf("CUDA ERROR %s:%d: '%s'\n", filename.c_str(), __LINE__,           \
             cudaGetErrorString(err));                                         \
    }                                                                          \
  } while (0)
#endif

namespace cudaq::qec {

// =============================================================================
// GPU Persistent Kernel (Simplified with Pre-Captured Graphs!)
// =============================================================================

/// @brief GPU persistent kernel for continuous syndrome decoding
/// 
/// This kernel launches once and stays resident on the GPU. Each thread block
/// continuously polls for work, and when found, launches the pre-captured
/// CUDA graph for that work slot.
///
/// Key Insight: Each work slot has its own CUDA graph that was captured with
/// that slot's specific input/output buffers. We just launch the graph - no
/// need to update parameters or call TensorRT APIs!
///
/// @param work_slots Array of work slots (device memory)
/// @param num_slots Number of work slots
/// @param stop_flag Pointer to stop flag (device memory)
__global__ void __launch_bounds__(256, 2)
    persistent_decoder_kernel(cudaq::qec::persistent_ai_decoder::WorkSlot *work_slots,
                              int num_slots,
                              volatile int *stop_flag) {

  // Get cooperative group for this thread block
  cg::thread_block block = cg::this_thread_block();

  // Persistent loop - kernel runs until stop_flag is set
  while (!(*stop_flag)) {
    bool work_found = false;
    int my_slot = -1;

    // Thread 0 in each block searches for work
    if (block.thread_rank() == 0) {
      // Each block checks different slots (stride by gridDim.x)
      for (int slot = blockIdx.x; slot < num_slots; slot += gridDim.x) {
        // Try to claim this work slot
        // Change status from 1 (input_ready) to 2 (processing)
        int old_status = atomicCAS(&work_slots[slot].status, 1, 2);
        if (old_status == 1) {
          // We won! This block will process this work
          my_slot = slot;
          work_found = true;
          break;
        }
      }
    }

    // Broadcast the result to all threads in the block using shared memory
    __shared__ bool shared_work;
    __shared__ int shared_slot;
    if (block.thread_rank() == 0) {
      shared_work = work_found;
      shared_slot = my_slot;
    }
    block.sync();
    work_found = shared_work;
    my_slot = shared_slot;

    if (work_found) {
      // All threads in block know we have work to do
      auto &slot = work_slots[my_slot];

      // =================================================================
      // LAUNCH THE PRE-CAPTURED CUDA GRAPH!
      // =================================================================
      // The graph was captured with this slot's specific buffers, so we
      // just launch it - no parameter updates needed!
      //
      // This is the key insight: once captured into a graph, we don't
      // need TensorRT's enqueueV3() anymore. The graph contains all the
      // TensorRT operations already.
      // =================================================================

      if (block.thread_rank() == 0) {
        // Launch this slot's graph on its stream
        cudaError_t err = cudaGraphLaunch(slot.graph_exec, slot.stream);
        if (err != cudaSuccess) {
          // Error handling - mark as free so it can be retried
          atomicExch(&slot.status, 0);
        } else {
          // Mark as output_ready (asynchronous)
          // Host will synchronize the stream before reading results
          atomicExch(&slot.status, 3);
        }
      }
      block.sync();
    } else {
      // No work available, brief sleep to avoid busy-waiting
      if (block.thread_rank() == 0) {
        __nanosleep(100); // 100 nanoseconds
      }
      block.sync();
    }
  }
}

// =============================================================================
// Host-side Implementation
// =============================================================================

persistent_ai_decoder::persistent_ai_decoder(
    nvinfer1::ICudaEngine *engine, nvinfer1::IExecutionContext *context,
    int input_index, int output_index, const Config &config)
    : engine_(engine), context_(context), input_index_(input_index),
      output_index_(output_index), config_(config), work_slots_device_(nullptr),
      work_slots_host_(nullptr), staging_input_host_(nullptr),
      staging_output_host_(nullptr), stop_flag_device_(nullptr),
      stop_flag_host_(nullptr) {

  CUDAQ_INFO("Initializing Persistent AI Decoder with {} work slots, {} "
             "blocks, syndrome size={}, output size={}",
             config_.num_work_slots, config_.num_blocks, config_.syndrome_size,
             config_.output_size);

  // Allocate resources and capture one graph per work slot
  if (!allocate_and_capture_graphs()) {
    throw std::runtime_error(
        "Failed to allocate resources and capture graphs for persistent decoder");
  }

  CUDAQ_INFO("Persistent AI Decoder initialized successfully");
  CUDAQ_INFO("Captured {} CUDA graphs (one per work slot)", config_.num_work_slots);
}

persistent_ai_decoder::~persistent_ai_decoder() {
  CUDAQ_INFO("Destroying Persistent AI Decoder");
  stop();
  cleanup_resources();
}

bool persistent_ai_decoder::allocate_and_capture_graphs() {
  // Allocate work slots on device
  size_t slots_size = config_.num_work_slots * sizeof(WorkSlot);
  HANDLE_CUDA_ERROR(cudaMalloc(&work_slots_device_, slots_size));

  // Allocate work slots on host for management
  HANDLE_CUDA_ERROR(cudaMallocHost(&work_slots_host_, slots_size));
  std::memset(work_slots_host_, 0, slots_size);

  // Initialize each work slot and capture its graph
  CUDAQ_INFO("Capturing {} CUDA graphs (this may take a moment)...", 
             config_.num_work_slots);

  for (size_t i = 0; i < config_.num_work_slots; ++i) {
    auto &slot = work_slots_host_[i];

    // Allocate device buffers for this slot
    HANDLE_CUDA_ERROR(cudaMalloc(&slot.input_buffer_device,
                                 config_.syndrome_size * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&slot.output_buffer_device,
                                 config_.output_size * sizeof(float)));

    // Create a stream for this slot
    HANDLE_CUDA_ERROR(cudaStreamCreate(&slot.stream));

    // =========================================================================
    // CAPTURE CUDA GRAPH FOR THIS SLOT
    // =========================================================================
    // Key: Set TensorRT to use THIS slot's buffers, then capture
    // =========================================================================

    // Set TensorRT to use this slot's buffers
    context_->setTensorAddress(engine_->getIOTensorName(input_index_),
                              slot.input_buffer_device);
    context_->setTensorAddress(engine_->getIOTensorName(output_index_),
                              slot.output_buffer_device);

    // Use the shared utility to capture the graph
    // This is the same code that streaming decoder uses, now refactored!
    try {
      slot.graph_exec = cuda_graph_utils::capture_graph_with_buffers(
          context_, slot.stream);
    } catch (const std::exception &e) {
      CUDAQ_WARN("Failed to capture graph for slot {}: {}", i, e.what());
      free_resources();
      return false;
    }

    // Initialize status
    slot.status = 0; // free

    if ((i + 1) % 4 == 0 || i == config_.num_work_slots - 1) {
      CUDAQ_INFO("Captured {}/{} graphs", i + 1, config_.num_work_slots);
    }
  }

  // Copy initialized slots to device
  HANDLE_CUDA_ERROR(cudaMemcpy(work_slots_device_, work_slots_host_,
                                slots_size, cudaMemcpyHostToDevice));

  // Allocate staging buffers for host-device transfers
  HANDLE_CUDA_ERROR(cudaMallocHost(&staging_input_host_,
                                   config_.syndrome_size * sizeof(float)));
  HANDLE_CUDA_ERROR(cudaMallocHost(&staging_output_host_,
                                   config_.output_size * sizeof(float)));

  // Allocate stop flag
  HANDLE_CUDA_ERROR(cudaMalloc(&stop_flag_device_, sizeof(int)));
  HANDLE_CUDA_ERROR(cudaMallocHost(&stop_flag_host_, sizeof(int)));
  *stop_flag_host_ = 0;
  HANDLE_CUDA_ERROR(cudaMemcpy(stop_flag_device_, stop_flag_host_, sizeof(int),
                                cudaMemcpyHostToDevice));

  CUDAQ_INFO("Successfully captured and allocated {} work slots", config_.num_work_slots);
  return true;
}

void persistent_ai_decoder::free_resources() {
  // Free work slot resources
  if (work_slots_host_) {
    for (size_t i = 0; i < config_.num_work_slots; ++i) {
      auto &slot = work_slots_host_[i];
      
      if (slot.graph_exec) {
        HANDLE_CUDA_ERROR_NO_THROW(cudaGraphExecDestroy(slot.graph_exec));
      }
      if (slot.stream) {
        HANDLE_CUDA_ERROR_NO_THROW(cudaStreamDestroy(slot.stream));
      }
      if (slot.input_buffer_device) {
        HANDLE_CUDA_ERROR_NO_THROW(cudaFree(slot.input_buffer_device));
      }
      if (slot.output_buffer_device) {
        HANDLE_CUDA_ERROR_NO_THROW(cudaFree(slot.output_buffer_device));
      }
    }
    HANDLE_CUDA_ERROR_NO_THROW(cudaFreeHost(work_slots_host_));
    work_slots_host_ = nullptr;
  }

  if (work_slots_device_) {
    HANDLE_CUDA_ERROR_NO_THROW(cudaFree(work_slots_device_));
    work_slots_device_ = nullptr;
  }

  if (staging_input_host_) {
    HANDLE_CUDA_ERROR_NO_THROW(cudaFreeHost(staging_input_host_));
    staging_input_host_ = nullptr;
  }

  if (staging_output_host_) {
    HANDLE_CUDA_ERROR_NO_THROW(cudaFreeHost(staging_output_host_));
    staging_output_host_ = nullptr;
  }

  if (stop_flag_device_) {
    HANDLE_CUDA_ERROR_NO_THROW(cudaFree(stop_flag_device_));
    stop_flag_device_ = nullptr;
  }

  if (stop_flag_host_) {
    HANDLE_CUDA_ERROR_NO_THROW(cudaFreeHost(stop_flag_host_));
    stop_flag_host_ = nullptr;
  }
}

void persistent_ai_decoder::cleanup_resources() {
  free_resources();
}

bool persistent_ai_decoder::start() {
  if (running_.load()) {
    CUDAQ_WARN("Persistent decoder is already running");
    return false;
  }

  CUDAQ_INFO("Starting Persistent AI Decoder (launching GPU kernel)");

  // Reset stop flag
  *stop_flag_host_ = 0;
  HANDLE_CUDA_ERROR(cudaMemcpy(stop_flag_device_, stop_flag_host_, sizeof(int),
                                cudaMemcpyHostToDevice));

  // Launch persistent kernel
  // Note: Using fewer blocks than SM count for true persistent kernel pattern
  dim3 grid(config_.num_blocks);
  dim3 block(config_.threads_per_block);

  CUDAQ_INFO("Launching persistent kernel with {} blocks, {} threads/block",
             config_.num_blocks, config_.threads_per_block);

  persistent_decoder_kernel<<<grid, block>>>(
      work_slots_device_, config_.num_work_slots, stop_flag_device_);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    CUDAQ_WARN("Kernel launch failed: {} ({})", 
               cudaGetErrorString(launch_err), static_cast<int>(launch_err));
    return false;
  }

  running_.store(true);
  CUDAQ_INFO("Persistent kernel launched successfully - now running on GPU!");

  return true;
}

void persistent_ai_decoder::stop() {
  if (!running_.load()) {
    return;
  }

  CUDAQ_INFO("Stopping Persistent AI Decoder");

  // Set stop flag
  *stop_flag_host_ = 1;
  HANDLE_CUDA_ERROR(cudaMemcpy(stop_flag_device_, stop_flag_host_, sizeof(int),
                                cudaMemcpyHostToDevice));

  // Wait for kernel to finish (synchronize all streams)
  for (size_t i = 0; i < config_.num_work_slots; ++i) {
    if (work_slots_host_[i].stream) {
      HANDLE_CUDA_ERROR(cudaStreamSynchronize(work_slots_host_[i].stream));
    }
  }
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  running_.store(false);
  CUDAQ_INFO("Persistent AI Decoder stopped");
}

bool persistent_ai_decoder::is_running() const { return running_.load(); }

int persistent_ai_decoder::find_free_slot() {
  // Copy work slot statuses from device
  size_t slots_size = config_.num_work_slots * sizeof(WorkSlot);
  HANDLE_CUDA_ERROR(cudaMemcpy(work_slots_host_, work_slots_device_,
                                slots_size, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < config_.num_work_slots; ++i) {
    if (work_slots_host_[i].status == 0) {
      return i;
    }
  }
  return -1; // No free slot
}

int persistent_ai_decoder::find_ready_slot() {
  // Copy work slot statuses from device
  size_t slots_size = config_.num_work_slots * sizeof(WorkSlot);
  HANDLE_CUDA_ERROR(cudaMemcpy(work_slots_host_, work_slots_device_,
                                slots_size, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < config_.num_work_slots; ++i) {
    if (work_slots_host_[i].status == 3) {  // output_ready
      return i;
    }
  }
  return -1; // No ready slot
}

bool persistent_ai_decoder::enqueue_syndrome(
    const std::vector<float> &syndrome) {
  if (syndrome.size() != config_.syndrome_size) {
    CUDAQ_WARN("Syndrome size mismatch: expected {}, got {}",
               config_.syndrome_size, syndrome.size());
    return false;
  }

  if (!running_.load()) {
    CUDAQ_WARN("Persistent decoder is not running");
    return false;
  }

  // Find a free work slot
  int slot_idx = find_free_slot();
  if (slot_idx < 0) {
    return false; // Queue full
  }

  auto &slot = work_slots_host_[slot_idx];

  // Copy syndrome to this slot's device buffer
  HANDLE_CUDA_ERROR(cudaMemcpy(slot.input_buffer_device, syndrome.data(),
                                config_.syndrome_size * sizeof(float),
                                cudaMemcpyHostToDevice));

  // Mark slot as input_ready (status = 1)
  slot.status = 1;
  HANDLE_CUDA_ERROR(cudaMemcpy(&work_slots_device_[slot_idx].status,
                                &slot.status, sizeof(int),
                                cudaMemcpyHostToDevice));

  return true;
}

bool persistent_ai_decoder::try_dequeue_result(std::vector<float> &result) {
  if (!running_.load()) {
    return false;
  }

  // Find a ready work slot (status == 3)
  int slot_idx = find_ready_slot();
  if (slot_idx < 0) {
    return false; // No results available
  }

  auto &slot = work_slots_host_[slot_idx];

  // Synchronize the stream to ensure inference is complete
  HANDLE_CUDA_ERROR(cudaStreamSynchronize(slot.stream));

  // Copy result from this slot's device buffer
  result.resize(config_.output_size);
  HANDLE_CUDA_ERROR(cudaMemcpy(result.data(), slot.output_buffer_device,
                                config_.output_size * sizeof(float),
                                cudaMemcpyDeviceToHost));

  // Mark slot as free (status = 0)
  slot.status = 0;
  HANDLE_CUDA_ERROR(cudaMemcpy(&work_slots_device_[slot_idx].status,
                                &slot.status, sizeof(int),
                                cudaMemcpyHostToDevice));

  return true;
}

size_t persistent_ai_decoder::get_queue_depth() const {
  // Copy work slots from device
  size_t slots_size = config_.num_work_slots * sizeof(WorkSlot);
  HANDLE_CUDA_ERROR(cudaMemcpy(work_slots_host_, work_slots_device_,
                                slots_size, cudaMemcpyDeviceToHost));

  size_t count = 0;
  for (size_t i = 0; i < config_.num_work_slots; ++i) {
    if (work_slots_host_[i].status == 1 || work_slots_host_[i].status == 2) {
      count++;
    }
  }
  return count;
}

size_t persistent_ai_decoder::get_available_results() const {
  // Copy work slots from device
  size_t slots_size = config_.num_work_slots * sizeof(WorkSlot);
  HANDLE_CUDA_ERROR(cudaMemcpy(work_slots_host_, work_slots_device_,
                                slots_size, cudaMemcpyDeviceToHost));

  size_t count = 0;
  for (size_t i = 0; i < config_.num_work_slots; ++i) {
    if (work_slots_host_[i].status == 3) {  // output_ready
      count++;
    }
  }
  return count;
}

} // namespace cudaq::qec
