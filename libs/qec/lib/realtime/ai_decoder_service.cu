/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h" // For RPCHeader, RPCResponse
#include <cstdlib>
#include <fstream>
#include <iostream>

namespace cudaq::qec {

// =============================================================================
// Gateway Kernels (The Bridge)
// =============================================================================

/// @brief Reads the dynamic buffer address from the mailbox and copies to fixed buffer
__global__ void gateway_input_kernel(
    void** mailbox_slot_ptr,    // The specific slot in the Global Bank
    float* trt_fixed_input,     // The persistent TRT input buffer
    size_t copy_size_bytes) 
{
    // 1. Read the pointer provided by the Dispatcher
    void* ring_buffer_data = *mailbox_slot_ptr;

    if (ring_buffer_data == nullptr) return;

    // 2. Skip RPC Header to find payload
    const char* src = (const char*)ring_buffer_data + sizeof(cudaq::nvqlink::RPCHeader);
    char* dst = (char*)trt_fixed_input;

    // 3. Grid-Stride Copy
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < copy_size_bytes; i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

/// @brief Copies result back to Ring Buffer and writes RPC Response
__global__ void gateway_output_kernel(
    void** mailbox_slot_ptr,
    const float* trt_fixed_output,
    size_t result_size_bytes)
{
    void* ring_buffer_data = *mailbox_slot_ptr;
    if (ring_buffer_data == nullptr) return;

    // 1. Write Result Payload (Overwriting input args in this design, or append after)
    // Assuming Input/Output fit in the same slot allocation.
    char* dst = (char*)ring_buffer_data + sizeof(cudaq::nvqlink::RPCHeader);
    const char* src = (const char*)trt_fixed_output;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < result_size_bytes; i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }

    // 2. Write RPC Response Header (Thread 0 only)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        auto* response = (cudaq::nvqlink::RPCResponse*)ring_buffer_data;
        response->magic = cudaq::nvqlink::RPC_MAGIC_RESPONSE;
        response->status = 0; // Success
        response->result_len = static_cast<uint32_t>(result_size_bytes);
        
        // Ensure memory visibility
        __threadfence_system();
    }
}

// =============================================================================
// Class Implementation
// =============================================================================

AIDecoderService::Logger AIDecoderService::gLogger;

void AIDecoderService::Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::printf("[TensorRT] %s\n", msg);
    }
}

AIDecoderService::AIDecoderService(const std::string& engine_path, void** device_mailbox_slot)
    : device_mailbox_slot_(device_mailbox_slot) {
    
    if (std::getenv("SKIP_TRT")) {
        // Skip TRT entirely; use fixed sizes for testing
        input_size_ = 16 * sizeof(float);
        output_size_ = 16 * sizeof(float);
        input_idx_ = 0;
        output_idx_ = 1;
        allocate_resources();
    } else {
        load_engine(engine_path);
        allocate_resources();
    }
}

AIDecoderService::~AIDecoderService() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (d_trt_input_) cudaFree(d_trt_input_);
    if (d_trt_output_) cudaFree(d_trt_output_);
    // Note: We do not free device_mailbox_slot_ as it is a view into the global bank
}

void AIDecoderService::load_engine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) throw std::runtime_error("Error opening engine file: " + path);
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    
    runtime_.reset(nvinfer1::createInferRuntime(gLogger));
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    context_.reset(engine_->createExecutionContext());

    // Auto-detect bindings
    input_idx_ = 0; // Simplified assumption, use engine_->getBindingName() in prod
    output_idx_ = 1;
    
    // Inspect shapes (assuming static shapes for realtime)
    auto input_dims = engine_->getTensorShape(engine_->getIOTensorName(input_idx_));
    auto output_dims = engine_->getTensorShape(engine_->getIOTensorName(output_idx_));

    // Calculate sizes (Assuming float)
    auto volume = [](const nvinfer1::Dims& d) {
        size_t v = 1;
        for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
        return v;
    };
    
    input_size_ = volume(input_dims) * sizeof(float);
    output_size_ = volume(output_dims) * sizeof(float);
}

void AIDecoderService::allocate_resources() {
    if (cudaMalloc(&d_trt_input_, input_size_) != cudaSuccess) 
        throw std::runtime_error("Failed to allocate TRT Input");
    if (cudaMalloc(&d_trt_output_, output_size_) != cudaSuccess) 
        throw std::runtime_error("Failed to allocate TRT Output");
}

void AIDecoderService::capture_graph(cudaStream_t stream) {
    // 1. Bind TensorRT to our fixed buffers
    context_->setTensorAddress(engine_->getIOTensorName(input_idx_), d_trt_input_);
    context_->setTensorAddress(engine_->getIOTensorName(output_idx_), d_trt_output_);

    // 2. Warmup
    context_->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    // 3. Capture
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // --- Node A: Gateway Input ---
    // Reads from *device_mailbox_slot_ -> Writes to d_trt_input_
    gateway_input_kernel<<<1, 128, 0, stream>>>(device_mailbox_slot_, d_trt_input_, input_size_);

    // --- Node B: TensorRT ---
    context_->enqueueV3(stream);

    // --- Node C: Gateway Output ---
    // Reads from d_trt_output_ -> Writes back to *device_mailbox_slot_
    gateway_output_kernel<<<1, 128, 0, stream>>>(device_mailbox_slot_, d_trt_output_, output_size_);

    cudaStreamEndCapture(stream, &graph);

    // 4. Instantiate for Device Launch
    cudaGraphInstantiateWithFlags(&graph_exec_, graph, cudaGraphInstantiateFlagDeviceLaunch);
    
    // 5. Upload & Cleanup
    cudaGraphUpload(graph_exec_, stream);
    cudaGraphDestroy(graph);
    
    cudaStreamSynchronize(stream);
}

} // namespace cudaq::qec
