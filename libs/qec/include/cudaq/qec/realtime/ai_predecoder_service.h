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
    AIPreDecoderService(const std::string& engine_path, void** device_mailbox_slot, int queue_depth = 16);
    virtual ~AIPreDecoderService();

    void capture_graph(cudaStream_t stream) override;

    bool poll_next_job(PreDecoderJob& out_job);
    void release_job(int slot_idx);

    int* get_device_queue_idx() const { return d_queue_idx_; }
    volatile int* get_device_ready_flags() const { return d_ready_flags_; }
    int* get_device_inflight_flag() const { return d_inflight_flag_; }

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

    // Device State
    int* d_queue_idx_ = nullptr;
    int* d_claimed_slot_ = nullptr;
    int* d_inflight_flag_ = nullptr;
};

} // namespace cudaq::qec
