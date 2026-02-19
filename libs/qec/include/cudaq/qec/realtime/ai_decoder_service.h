/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

namespace cudaq::qec {

class AIDecoderService {
public:
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    } static gLogger;

    /// @brief Constructor. Accepts a serialized TRT engine (.engine/.plan) or
    ///        an ONNX model (.onnx) which will be compiled to a TRT engine.
    /// @param model_path Path to the model file
    /// @param device_mailbox_slot Pointer to the specific slot in the global mailbox bank
    AIDecoderService(const std::string& model_path, void** device_mailbox_slot);

    virtual ~AIDecoderService();

    virtual void capture_graph(cudaStream_t stream);

    cudaGraphExec_t get_executable_graph() const { return graph_exec_; }

    /// @brief Size of the primary input tensor in bytes (payload from RPC)
    size_t get_input_size() const { return input_size_; }

    /// @brief Size of the primary output tensor in bytes (forwarded to CPU)
    size_t get_output_size() const { return output_size_; }

protected:
    void load_engine(const std::string& path);
    void build_engine_from_onnx(const std::string& onnx_path);
    void setup_bindings();
    void allocate_resources();

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaGraphExec_t graph_exec_ = nullptr;

    void** device_mailbox_slot_;
    void* d_trt_input_ = nullptr;            // Primary input buffer
    void* d_trt_output_ = nullptr;           // Primary output buffer (residual_detectors)
    std::vector<void*> d_aux_buffers_;       // Additional I/O buffers TRT needs

    struct TensorBinding {
        std::string name;
        void* d_buffer = nullptr;
        size_t size_bytes = 0;
        bool is_input = false;
    };
    std::vector<TensorBinding> all_bindings_;

    size_t input_size_ = 0;
    size_t output_size_ = 0;
};

} // namespace cudaq::qec
