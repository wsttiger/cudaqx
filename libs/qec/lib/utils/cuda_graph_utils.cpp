/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/cuda_graph_utils.h"
#include "common/Logger.h"
#include <exception>
#include <vector>

// Remove the leading path elements from a fully qualified filename
static inline void trim_filename(std::string &filename) {
  std::size_t lastSlashPos = filename.find_last_of('/');
  if (lastSlashPos != std::string::npos)
    filename = filename.substr(lastSlashPos + 1);
}

#ifndef HANDLE_CUDA_ERROR
#define HANDLE_CUDA_ERROR(x)                                                   \
  do {                                                                         \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      std::string filename = __FILE__;                                         \
      trim_filename(filename);                                                 \
      printf("CUDA ERROR %s:%d: '%s'\n", filename.c_str(), __LINE__,           \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

namespace cudaq::qec::cuda_graph_utils {

CaptureResult capture_cuda_graph(nvinfer1::IExecutionContext *context,
                                  cudaStream_t stream, void *input_buffer,
                                  void *output_buffer, int input_index,
                                  int output_index,
                                  nvinfer1::ICudaEngine *engine,
                                  size_t input_size) {
  CaptureResult result;

  try {
    // Generate dummy input data (values don't matter for capture, just shape)
    std::vector<float> dummy_input(input_size, 0.0f);

    // Copy dummy data to GPU
    cudaError_t err =
        cudaMemcpy(input_buffer, dummy_input.data(), input_size * sizeof(float),
                   cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      result.error_message =
          "Failed to copy dummy data: " + std::string(cudaGetErrorString(err));
      return result;
    }

    // Attempt to capture the graph
    CUDAQ_INFO("Attempting to capture CUDA graph during initialization...");

    err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
      result.error_message = "cudaStreamBeginCapture failed: " +
                             std::string(cudaGetErrorString(err));
      return result;
    }

    // Record TensorRT operations
    context->setTensorAddress(engine->getIOTensorName(input_index),
                              input_buffer);
    context->setTensorAddress(engine->getIOTensorName(output_index),
                              output_buffer);
    context->enqueueV3(stream);

    err = cudaStreamEndCapture(stream, &result.graph);
    if (err != cudaSuccess) {
      result.error_message = "cudaStreamEndCapture failed: " +
                             std::string(cudaGetErrorString(err));
      return result;
    }

    // Instantiate the graph
    err = cudaGraphInstantiate(&result.graph_exec, result.graph, 0);
    if (err != cudaSuccess) {
      result.error_message = "cudaGraphInstantiate failed: " +
                             std::string(cudaGetErrorString(err));
      if (result.graph) {
        cudaGraphDestroy(result.graph);
        result.graph = nullptr;
      }
      return result;
    }

    CUDAQ_INFO("CUDA graph captured successfully during initialization");
    result.success = true;

  } catch (const std::exception &e) {
    result.error_message = "Exception during capture: " + std::string(e.what());
    // Clean up on failure
    if (result.graph_exec) {
      cudaGraphExecDestroy(result.graph_exec);
      result.graph_exec = nullptr;
    }
    if (result.graph) {
      cudaGraphDestroy(result.graph);
      result.graph = nullptr;
    }
  }

  return result;
}

cudaGraphExec_t capture_graph_with_buffers(nvinfer1::IExecutionContext *context,
                                            cudaStream_t stream) {
  // This lightweight version assumes setTensorAddress() has already been called
  // No dummy data allocation - buffers should already be set up
  
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  
  // Begin capture
  cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaStreamBeginCapture failed: " + 
                             std::string(cudaGetErrorString(err)));
  }
  
  // Record TensorRT operations (buffers already configured)
  context->enqueueV3(stream);
  
  // End capture
  err = cudaStreamEndCapture(stream, &graph);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaStreamEndCapture failed: " + 
                             std::string(cudaGetErrorString(err)));
  }
  
  // Instantiate the graph for device-side launch
  // Required flags: DeviceLaunch must be combined with AutoFreeOnLaunch
  unsigned long long flags = cudaGraphInstantiateFlagDeviceLaunch | 
                             cudaGraphInstantiateFlagAutoFreeOnLaunch;
  err = cudaGraphInstantiateWithFlags(&graph_exec, graph, flags);
  if (err != cudaSuccess) {
    if (graph) {
      cudaGraphDestroy(graph);
    }
    throw std::runtime_error("cudaGraphInstantiateWithFlags failed: " + 
                             std::string(cudaGetErrorString(err)));
  }
  
  // Destroy the graph template (we only need graph_exec)
  HANDLE_CUDA_ERROR(cudaGraphDestroy(graph));
  
  CUDAQ_INFO("CUDA graph captured successfully");
  
  return graph_exec;
}

bool supports_cuda_graphs(const nvinfer1::ICudaEngine *engine) {
  // Check for dynamic shapes
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char *name = engine->getIOTensorName(i);
    auto dims = engine->getTensorShape(name);
    for (int j = 0; j < dims.nbDims; ++j) {
      if (dims.d[j] == -1) {
        CUDAQ_INFO(
            "Dynamic shape detected in tensor '{}', CUDA graphs not supported",
            name);
        return false;
      }
    }
  }

  // Check for multiple optimization profiles (often used with dynamic shapes)
  if (engine->getNbOptimizationProfiles() > 1) {
    CUDAQ_INFO(
        "Multiple optimization profiles detected, CUDA graphs not supported");
    return false;
  }

  return true;
}

} // namespace cudaq::qec::cuda_graph_utils
