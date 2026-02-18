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
    // Logger interface for NvInfer
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    } static gLogger;

    /// @brief Constructor
    /// @param engine_path Path to the serialized TensorRT engine file
    /// @param device_mailbox_slot Pointer to the specific slot in the global mailbox bank
    ///                            that this decoder will listen to.
    AIDecoderService(const std::string& engine_path, void** device_mailbox_slot);

    virtual ~AIDecoderService();

    /// @brief Captures the CUDA Graph (Gateway In -> TRT -> Gateway Out)
    /// @param stream The stream to use for capture
    virtual void capture_graph(cudaStream_t stream);

    /// @brief Returns the executable graph for the Dispatcher table
    cudaGraphExec_t get_executable_graph() const { return graph_exec_; }

    /// @brief Returns the required input/output sizes for verification
    size_t get_input_size() const { return input_size_; }
    size_t get_output_size() const { return output_size_; }

protected:
    void load_engine(const std::string& path);
    void allocate_resources();

    // NvInfer resources
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // Graph resources
    cudaGraphExec_t graph_exec_ = nullptr;
    
    // Memory resources (Resident on Device)
    void** device_mailbox_slot_; // Address where Dispatcher writes the data pointer
    float* d_trt_input_ = nullptr;
    float* d_trt_output_ = nullptr;

    // Metadata
    size_t input_size_ = 0;
    size_t output_size_ = 0;
    int input_idx_ = -1;
    int output_idx_ = -1;
};

} // namespace cudaq::qec
