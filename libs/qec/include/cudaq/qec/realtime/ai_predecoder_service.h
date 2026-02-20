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

// Portable CPU Yield Macro for busy-polling
#if defined(__x86_64__)
    #include <immintrin.h>
    #define QEC_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
    #define QEC_CPU_RELAX() asm volatile("yield" ::: "memory")
#else
    #define QEC_CPU_RELAX() std::atomic_thread_fence(std::memory_order_seq_cst)
#endif

namespace cudaq::qec {

struct PreDecoderJob {
    int slot_idx;
    void* ring_buffer_ptr;
    void* inference_data;       // Points into the pinned output queue (type-agnostic)
};

class AIPreDecoderService : public AIDecoderService {
public:
    AIPreDecoderService(const std::string& engine_path, void** device_mailbox_slot,
                        int queue_depth = 16, const std::string& engine_save_path = "");
    virtual ~AIPreDecoderService();

    /// @param device_launch If true, instantiate graph with DeviceLaunch flag
    ///        (for device-side dispatcher). If false, use standard instantiation
    ///        (for host-side dispatcher).
    void capture_graph(cudaStream_t stream, bool device_launch);
    void capture_graph(cudaStream_t stream) override { capture_graph(stream, true); }

    bool poll_next_job(PreDecoderJob& out_job);
    void release_job(int slot_idx);

    int* get_device_queue_idx() const { return d_queue_idx_; }
    volatile int* get_device_ready_flags() const { return d_ready_flags_; }
    int* get_device_inflight_flag() const { return d_inflight_flag_; }

    // Host-side accessors (for host dispatcher backpressure checks)
    volatile int* get_host_ready_flags() const { return h_ready_flags_; }
    volatile int* get_host_queue_idx() const { return h_queue_idx_; }
    int get_queue_depth() const { return queue_depth_; }

private:
    int queue_depth_;
    int cpu_poll_idx_ = 0;

    // Pinned Host Memory (The Queue)
    volatile int* h_ready_flags_ = nullptr; 
    void** h_ring_ptrs_ = nullptr;          
    void* h_outputs_ = nullptr;             // Type-agnostic pinned output queue

    // Device Mapped Pointers (For the Graph to write to)
    volatile int* d_ready_flags_ = nullptr;
    void** d_ring_ptrs_ = nullptr;
    void* d_outputs_ = nullptr;

    // Queue index: mapped pinned so both GPU and host can access
    volatile int* h_queue_idx_ = nullptr;   // Host pointer
    int* d_queue_idx_ = nullptr;            // Device pointer (same physical memory)
    int* d_claimed_slot_ = nullptr;
    int* d_inflight_flag_ = nullptr;
};

} // namespace cudaq::qec
