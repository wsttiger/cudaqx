/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <iostream>
#include <string>

namespace cudaq::qec::cuda_graph_utils {

/// @brief Simple TensorRT logger
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << msg << std::endl;
  }
};

/// @brief Result structure for CUDA graph capture attempts
struct CaptureResult {
  bool success = false;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  std::string error_message;
};

/// @brief Attempt to capture a CUDA graph for TensorRT inference
/// @param context TensorRT execution context
/// @param stream CUDA stream to use for capture
/// @param input_buffer GPU buffer for input data
/// @param output_buffer GPU buffer for output data
/// @param input_index Index of input tensor in the engine
/// @param output_index Index of output tensor in the engine
/// @param engine TensorRT engine
/// @param input_size Size of input buffer (number of elements)
/// @return CaptureResult containing success status and graph handles
///
/// This function performs a "dry run" of TensorRT inference using dummy input
/// data to capture all CUDA operations into a graph. The captured graph can
/// then be launched repeatedly with different input data for optimized
/// performance.
CaptureResult capture_cuda_graph(nvinfer1::IExecutionContext *context,
                                  cudaStream_t stream, void *input_buffer,
                                  void *output_buffer, int input_index,
                                  int output_index,
                                  nvinfer1::ICudaEngine *engine,
                                  size_t input_size);

/// @brief Capture CUDA graph with pre-configured buffers (lightweight version)
/// @param context TensorRT execution context (with buffers already set via setTensorAddress)
/// @param stream CUDA stream to use for capture
/// @return Executable CUDA graph
///
/// This is a lighter-weight version that assumes:
/// - context->setTensorAddress() has already been called for input/output
/// - Buffers are already allocated and ready
/// - No dummy data allocation needed
///
/// Throws std::runtime_error on failure.
cudaGraphExec_t capture_graph_with_buffers(nvinfer1::IExecutionContext *context,
                                            cudaStream_t stream);

/// @brief Check if CUDA graphs are supported for this engine
/// @param engine TensorRT engine to check
/// @return true if CUDA graphs are supported, false otherwise
///
/// CUDA graphs are not supported for engines with:
/// - Dynamic shapes (dimensions with -1)
/// - Multiple optimization profiles
bool supports_cuda_graphs(const nvinfer1::ICudaEngine *engine);

} // namespace cudaq::qec::cuda_graph_utils
