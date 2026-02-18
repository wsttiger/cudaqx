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

// Portable CPU Yield Macro for busy-polling (Fix #5)
#if defined(__x86_64__)
    #include <immintrin.h>
    #define QEC_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
    #define QEC_CPU_RELAX() asm volatile("yield" ::: "memory")
#else
    #define QEC_CPU_RELAX() std::atomic_thread_fence(std::memory_order_seq_cst)
#endif

namespace cudaq::qec {

// Represents a single job handed off from GPU to CPU
struct PreDecoderJob {
    int slot_idx;            // The queue index (needed for release)
    void* ring_buffer_ptr;   // The FPGA mapped memory address
    float* inference_data;   // Pointer to the TensorRT output
};

class AIPreDecoderService : public AIDecoderService {
public:
    AIPreDecoderService(const std::string& engine_path, void** device_mailbox_slot, int queue_depth = 16);
    virtual ~AIPreDecoderService();

    // Overrides the standard graph with the CPU-Handoff graph
    void capture_graph(cudaStream_t stream) override;

    // --- CPU Thread Interfaces ---

    /// @brief Polls the circular buffer for a new job. Non-blocking.
    bool poll_next_job(PreDecoderJob& out_job);

    /// @brief Releases the slot back to the GPU once the Outgoing Thread finishes.
    void release_job(int slot_idx);

    /// @brief Returns the device pointer to the queue tail index (for dispatcher backpressure).
    int* get_device_queue_idx() const { return d_queue_idx_; }

    /// @brief Returns the device-mapped pointer to the ready flags (for dispatcher backpressure).
    volatile int* get_device_ready_flags() const { return d_ready_flags_; }

    /// @brief Returns the device pointer to the in-flight flag (for single-launch guarantee).
    /// Dispatcher sets to 1 before launching; output kernel clears to 0 when done.
    int* get_device_inflight_flag() const { return d_inflight_flag_; }

private:
    int queue_depth_;
    int cpu_poll_idx_ = 0;

    // --- Pinned Host Memory (The Queue) ---
    volatile int* h_ready_flags_ = nullptr; 
    void** h_ring_ptrs_ = nullptr;          
    float* h_outputs_ = nullptr;            

    // --- Device Mapped Pointers (For the Graph to write to) ---
    volatile int* d_ready_flags_ = nullptr;
    void** d_ring_ptrs_ = nullptr;
    float* d_outputs_ = nullptr;

    // --- Device State ---
    int* d_queue_idx_ = nullptr;      // Tracks the current slot tail on the GPU
    int* d_claimed_slot_ = nullptr;   // Passes claimed slot from input to output kernel
    int* d_inflight_flag_ = nullptr;  // 0 = idle, 1 = graph in flight (set by dispatcher, cleared by output kernel)
};

} // namespace cudaq::qec
