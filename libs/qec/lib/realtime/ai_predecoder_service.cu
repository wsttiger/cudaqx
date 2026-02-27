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
#include <cuda/atomic>
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

// System scope for NVLink/PCIe visibility to host (design: no __threadfence_system)
using atomic_int_sys = cuda::atomic<int, cuda::thread_scope_system>;

// =============================================================================
// Kernels (single slot 0 only; queue removed for host-side dynamic pool)
// =============================================================================

__global__ void predecoder_input_kernel(
    void** mailbox_slot_ptr,
    atomic_int_sys* d_ready_flags,
    void** d_ring_ptrs,
    void* trt_input,
    size_t input_size_bytes)
{
    __shared__ void* ring_ptr;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ring_ptr = *mailbox_slot_ptr;
        d_ring_ptrs[0] = ring_ptr;
    }
    __syncthreads();

    if (!ring_ptr) return;

    // RPCHeader is 12 bytes (3 x uint32_t), so src is 4-byte aligned.
    const uint32_t* src4 = (const uint32_t*)((const char*)ring_ptr + sizeof(cudaq::nvqlink::RPCHeader));
    uint32_t* dst4 = (uint32_t*)trt_input;
    size_t n4 = input_size_bytes / sizeof(uint32_t);
    for (size_t i = threadIdx.x; i < n4; i += blockDim.x)
        dst4[i] = src4[i];

    size_t done = n4 * sizeof(uint32_t);
    const char* src_tail = (const char*)src4 + done;
    char* dst_tail = (char*)trt_input + done;
    for (size_t i = done + threadIdx.x; i < input_size_bytes; i += blockDim.x)
        dst_tail[i - done] = src_tail[i - done];
}

__global__ void predecoder_signal_ready_kernel(atomic_int_sys* d_ready_flags)
{
    if (threadIdx.x == 0)
        d_ready_flags[0].store(1, cuda::std::memory_order_release);
}

__global__ void passthrough_copy_kernel(void* dst, const void* src, size_t num_bytes) {
    const uint4* src4 = (const uint4*)src;
    uint4* dst4 = (uint4*)dst;
    size_t n4 = num_bytes / sizeof(uint4);
    for (size_t i = threadIdx.x; i < n4; i += blockDim.x)
        dst4[i] = src4[i];

    size_t done = n4 * sizeof(uint4);
    for (size_t i = done + threadIdx.x; i < num_bytes; i += blockDim.x)
        ((char*)dst)[i] = ((const char*)src)[i];
}

// =============================================================================
// Class Implementation
// =============================================================================

AIPreDecoderService::AIPreDecoderService(const std::string& path, void** mailbox,
                                         int /* queue_depth (ignored; always 1) */,
                                         const std::string& engine_save_path)
    : AIDecoderService(path, mailbox, engine_save_path), queue_depth_(1)
{
    void* buf = nullptr;

    SERVICE_CUDA_CHECK(cudaHostAlloc(&buf, sizeof(atomic_int_sys), cudaHostAllocMapped));
    h_ready_flags_ = static_cast<atomic_int_sys*>(buf);
    new (h_ready_flags_) atomic_int_sys(0);

    SERVICE_CUDA_CHECK(cudaHostAlloc(&h_ring_ptrs_, sizeof(void*), cudaHostAllocMapped));
    SERVICE_CUDA_CHECK(cudaHostAlloc(&h_outputs_, get_output_size(), cudaHostAllocMapped));

    SERVICE_CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_ready_flags_, (void*)h_ready_flags_, 0));
    SERVICE_CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_ring_ptrs_, (void*)h_ring_ptrs_, 0));
    SERVICE_CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_outputs_, (void*)h_outputs_, 0));
}

AIPreDecoderService::~AIPreDecoderService() {
    if (h_ready_flags_) {
        h_ready_flags_[0].~atomic_int_sys();
        cudaFreeHost((void*)h_ready_flags_);
        h_ready_flags_ = nullptr;
        d_ready_flags_ = nullptr;
    }
    if (h_ring_ptrs_) {
        cudaFreeHost(h_ring_ptrs_);
        h_ring_ptrs_ = nullptr;
    }
    if (h_outputs_) {
        cudaFreeHost(h_outputs_);
        h_outputs_ = nullptr;
    }
}

void AIPreDecoderService::capture_graph(cudaStream_t stream, bool device_launch) {
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

    if (skip_trt) {
        passthrough_copy_kernel<<<1, 256, 0, stream>>>(
            d_trt_output_, d_trt_input_, get_input_size());
    } else {
        context_->enqueueV3(stream);
    }

    SERVICE_CUDA_CHECK(cudaMemcpyAsync(
        d_outputs_, d_trt_output_, get_output_size(),
        cudaMemcpyDeviceToDevice, stream));

    predecoder_signal_ready_kernel<<<1, 1, 0, stream>>>(
        static_cast<atomic_int_sys*>(d_ready_flags_));

    SERVICE_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    if (device_launch) {
        cudaError_t inst_err = cudaGraphInstantiateWithFlags(
            &graph_exec_, graph, cudaGraphInstantiateFlagDeviceLaunch);
        if (inst_err != cudaSuccess) {
            cudaGraphDestroy(graph);
            throw std::runtime_error(
                std::string("cudaGraphInstantiateWithFlags (DeviceLaunch) FAILED: ")
                + cudaGetErrorString(inst_err));
        }
        SERVICE_CUDA_CHECK(cudaGraphUpload(graph_exec_, stream));
    } else {
        cudaError_t inst_err = cudaGraphInstantiate(&graph_exec_, graph, 0);
        if (inst_err != cudaSuccess) {
            cudaGraphDestroy(graph);
            throw std::runtime_error(
                std::string("cudaGraphInstantiate FAILED: ")
                + cudaGetErrorString(inst_err));
        }
    }

    cudaGraphDestroy(graph);
    SERVICE_CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool AIPreDecoderService::poll_next_job(PreDecoderJob& out_job) {
    auto* sys_flags = static_cast<atomic_int_sys*>(h_ready_flags_);
    int expected = 1;
    // Atomically claim: 1 (Ready) -> 2 (Processing) so we enqueue the job exactly once.
    // Use relaxed on failure so spinning doesn't add barriers that delay seeing GPU's store(1).
    if (sys_flags[0].compare_exchange_strong(expected, 2,
            cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed)) {
        out_job.slot_idx = 0;
        out_job.ring_buffer_ptr = h_ring_ptrs_[0];
        out_job.inference_data = h_outputs_;
        return true;
    }
    return false;
}

void AIPreDecoderService::release_job(int /* slot_idx */) {
    auto* sys_flags = static_cast<atomic_int_sys*>(h_ready_flags_);
    // PyMatching done: 2 (Processing) -> 0 (Idle)
    sys_flags[0].store(0, cuda::std::memory_order_release);
}

} // namespace cudaq::qec
