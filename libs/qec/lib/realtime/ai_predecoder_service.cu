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

    const char* src = (const char*)ring_ptr + sizeof(cudaq::nvqlink::RPCHeader);
    char* dst = (char*)trt_input;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < input_size_bytes; i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

__global__ void predecoder_output_kernel(
    atomic_int_sys* d_ready_flags,
    void* d_outputs,
    const void* trt_output,
    size_t output_size_bytes)
{
    char* dst = (char*)d_outputs;
    const char* src = (const char*)trt_output;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < output_size_bytes; i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }

    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_ready_flags[0].store(1, cuda::std::memory_order_release);
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

    predecoder_input_kernel<<<1, 128, 0, stream>>>(
        device_mailbox_slot_,
        static_cast<atomic_int_sys*>(d_ready_flags_),
        d_ring_ptrs_, d_trt_input_, get_input_size());

    if (skip_trt) {
        passthrough_copy_kernel<<<1, 128, 0, stream>>>(
            d_trt_output_, d_trt_input_, get_input_size());
    } else {
        context_->enqueueV3(stream);
    }

    predecoder_output_kernel<<<1, 128, 0, stream>>>(
        static_cast<atomic_int_sys*>(d_ready_flags_),
        d_outputs_, d_trt_output_, get_output_size());

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
