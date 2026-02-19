/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/realtime/ai_predecoder_service.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h"
#include <cstdlib>
#include <stdexcept>
#include <string>

#define SERVICE_CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error in AIPreDecoderService: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

namespace cudaq::qec {

// =============================================================================
// Kernels
// =============================================================================

__global__ void predecoder_input_kernel(
    void** mailbox_slot_ptr, int* d_queue_idx, volatile int* d_ready_flags, 
    void** d_ring_ptrs, void* trt_input, size_t input_size_bytes,
    int* d_claimed_slot) 
{
    __shared__ int slot_idx;
    __shared__ void* ring_ptr;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ring_ptr = *mailbox_slot_ptr;
        slot_idx = *d_queue_idx;
        *d_claimed_slot = slot_idx;

        if (d_ready_flags[slot_idx] == 1) {
            ring_ptr = nullptr;
        } else {
            d_ring_ptrs[slot_idx] = ring_ptr; 
        }
    }
    __syncthreads();

    if (!ring_ptr) return;

    const char* src = (const char*)ring_ptr + sizeof(cudaq::nvqlink::RPCHeader);
    char* dst = (char*)trt_input;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < input_size_bytes; i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

__global__ void predecoder_output_kernel(
    int* d_claimed_slot, int* d_queue_idx, int queue_depth,
    volatile int* d_ready_flags, void* d_outputs, const void* trt_output,
    size_t output_size_bytes, volatile int* d_inflight_flag)
{
    int slot_idx = *d_claimed_slot;

    char* dst = (char*)d_outputs + (slot_idx * output_size_bytes);
    const char* src = (const char*)trt_output;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < output_size_bytes; i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }

    __syncthreads();
    __threadfence_system();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_ready_flags[slot_idx] = 1; 
        *d_queue_idx = (slot_idx + 1) % queue_depth;
        __threadfence_system();
        *d_inflight_flag = 0;
    }
}

__global__ void passthrough_copy_kernel(void* dst, const void* src, size_t num_bytes) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_bytes; i += blockDim.x * gridDim.x) {
        ((char*)dst)[i] = ((const char*)src)[i];
    }
}

// =============================================================================
// Class Implementation
// =============================================================================

AIPreDecoderService::AIPreDecoderService(const std::string& path, void** mailbox,
                                         int queue_depth, const std::string& engine_save_path)
    : AIDecoderService(path, mailbox, engine_save_path), queue_depth_(queue_depth)
{
    SERVICE_CUDA_CHECK(cudaHostAlloc(&h_ready_flags_, queue_depth_ * sizeof(int), cudaHostAllocMapped));
    SERVICE_CUDA_CHECK(cudaHostAlloc(&h_ring_ptrs_, queue_depth_ * sizeof(void*), cudaHostAllocMapped));
    SERVICE_CUDA_CHECK(cudaHostAlloc(&h_outputs_, queue_depth_ * get_output_size(), cudaHostAllocMapped));

    memset((void*)h_ready_flags_, 0, queue_depth_ * sizeof(int));

    SERVICE_CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_ready_flags_, (void*)h_ready_flags_, 0));
    SERVICE_CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_ring_ptrs_, (void*)h_ring_ptrs_, 0));
    SERVICE_CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_outputs_, (void*)h_outputs_, 0));

    SERVICE_CUDA_CHECK(cudaMalloc(&d_queue_idx_, sizeof(int)));
    SERVICE_CUDA_CHECK(cudaMemset(d_queue_idx_, 0, sizeof(int)));

    SERVICE_CUDA_CHECK(cudaMalloc(&d_claimed_slot_, sizeof(int)));
    SERVICE_CUDA_CHECK(cudaMemset(d_claimed_slot_, 0, sizeof(int)));

    SERVICE_CUDA_CHECK(cudaMalloc(&d_inflight_flag_, sizeof(int)));
    SERVICE_CUDA_CHECK(cudaMemset(d_inflight_flag_, 0, sizeof(int)));
}

AIPreDecoderService::~AIPreDecoderService() {
    if (h_ready_flags_) cudaFreeHost((void*)h_ready_flags_);
    if (h_ring_ptrs_) cudaFreeHost(h_ring_ptrs_);
    if (h_outputs_) cudaFreeHost(h_outputs_);
    if (d_queue_idx_) cudaFree(d_queue_idx_);
    if (d_claimed_slot_) cudaFree(d_claimed_slot_);
    if (d_inflight_flag_) cudaFree(d_inflight_flag_);
}

void AIPreDecoderService::capture_graph(cudaStream_t stream) {
    bool skip_trt = (std::getenv("SKIP_TRT") != nullptr);

    if (!skip_trt) {
        for (auto& b : all_bindings_) {
            context_->setTensorAddress(b.name.c_str(), b.d_buffer);
        }
        context_->enqueueV3(stream);
    }
    SERVICE_CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph;
    SERVICE_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    predecoder_input_kernel<<<1, 128, 0, stream>>>(
        device_mailbox_slot_, d_queue_idx_, d_ready_flags_, 
        d_ring_ptrs_, d_trt_input_, get_input_size(),
        d_claimed_slot_);

    if (skip_trt) {
        passthrough_copy_kernel<<<1, 128, 0, stream>>>(
            d_trt_output_, d_trt_input_, get_input_size());
    } else {
        context_->enqueueV3(stream);
    }

    predecoder_output_kernel<<<1, 128, 0, stream>>>(
        d_claimed_slot_, d_queue_idx_, queue_depth_, d_ready_flags_, 
        d_outputs_, d_trt_output_, get_output_size(),
        d_inflight_flag_);

    SERVICE_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    
    cudaError_t inst_err = cudaGraphInstantiateWithFlags(&graph_exec_, graph, cudaGraphInstantiateFlagDeviceLaunch);
    if (inst_err != cudaSuccess) {
        cudaGraphDestroy(graph);
        throw std::runtime_error(
            std::string("cudaGraphInstantiateWithFlags FAILED: ") + cudaGetErrorString(inst_err));
    }
    
    SERVICE_CUDA_CHECK(cudaGraphUpload(graph_exec_, stream));
    cudaGraphDestroy(graph);
    SERVICE_CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool AIPreDecoderService::poll_next_job(PreDecoderJob& out_job) {
    if (h_ready_flags_[cpu_poll_idx_] == 1) {
        std::atomic_thread_fence(std::memory_order_acquire);
        
        out_job.slot_idx = cpu_poll_idx_;
        out_job.ring_buffer_ptr = h_ring_ptrs_[cpu_poll_idx_];
        out_job.inference_data = static_cast<char*>(h_outputs_) + (cpu_poll_idx_ * get_output_size());

        cpu_poll_idx_ = (cpu_poll_idx_ + 1) % queue_depth_;
        return true;
    }
    return false;
}

void AIPreDecoderService::release_job(int slot_idx) {
    __atomic_store_n(&h_ready_flags_[slot_idx], 0, __ATOMIC_RELEASE);
}

} // namespace cudaq::qec
