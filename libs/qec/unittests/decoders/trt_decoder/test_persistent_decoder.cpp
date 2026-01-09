/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/persistent_ai_decoder.h"
#include "cudaq/qec/cuda_graph_utils.h"
#include "trt_test_data.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/trt_decoder_internal.h"
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace cudaq::qec;

// Test fixture for persistent decoder tests
class PersistentDecoderTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Check if CUDA is available
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA not available or no GPU found";
    }

    // Check compute capability (need 7.0+ for device-side launch)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int compute_capability = prop.major * 10 + prop.minor;
    if (compute_capability < 70) {
      GTEST_SKIP() << "Persistent decoder requires compute capability 7.0+ (Volta or newer), found "
                   << prop.major << "." << prop.minor;
    }
  }
};

// Test basic initialization
TEST_F(PersistentDecoderTest, BasicInitialization) {
  // Check if test ONNX model exists
  std::string onnx_path = "../assets/tests/surface_code_decoder.onnx";
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model not found: " << onnx_path;
  }

  // Load TensorRT engine
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;

  try {
    // Build from ONNX
    cuda_graph_utils::Logger logger;
    cudaqx::heterogeneous_map params;
    params.insert("onnx_load_path", onnx_path);
    engine = trt_decoder_internal::build_engine_from_onnx(onnx_path, params, logger);
    context.reset(engine->createExecutionContext());
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TensorRT engine: " << e.what();
  }

  // Get tensor indices
  int input_index = 0;
  int output_index = 1;

  // Get tensor sizes
  auto input_dims = engine->getTensorShape(engine->getIOTensorName(input_index));
  auto output_dims = engine->getTensorShape(engine->getIOTensorName(output_index));

  size_t syndrome_size = 1;
  for (int i = 0; i < input_dims.nbDims; ++i) {
    syndrome_size *= input_dims.d[i];
  }

  size_t output_size = 1;
  for (int i = 0; i < output_dims.nbDims; ++i) {
    output_size *= output_dims.d[i];
  }

  // Create persistent decoder config
  persistent_ai_decoder::Config config;
  config.syndrome_size = syndrome_size;
  config.output_size = output_size;
  config.num_work_slots = 4;  // Small number for testing
  config.num_blocks = 2;
  config.threads_per_block = 256;

  // Create persistent decoder (this captures graphs - may take a few seconds)
  std::cout << "Creating persistent decoder (capturing " << config.num_work_slots 
            << " graphs)..." << std::endl;

  auto start_init = std::chrono::high_resolution_clock::now();
  
  EXPECT_NO_THROW({
    persistent_ai_decoder decoder(engine.get(), context.get(), 
                                  input_index, output_index, config);
  });

  auto end_init = std::chrono::high_resolution_clock::now();
  auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     end_init - start_init).count();

  std::cout << "Initialization took " << init_ms << " ms" << std::endl;

  // Initialization should complete in reasonable time (< 30 seconds for 4 graphs)
  EXPECT_LT(init_ms, 30000) << "Initialization took too long: " << init_ms << " ms";
}

// Test start/stop lifecycle
TEST_F(PersistentDecoderTest, StartStopLifecycle) {
  std::string onnx_path = "../assets/tests/surface_code_decoder.onnx";
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model not found";
  }

  // Setup (similar to above)
  cuda_graph_utils::Logger logger;
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  auto engine = trt_decoder_internal::build_engine_from_onnx(onnx_path, params, logger);
  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

  int input_index = 0;
  int output_index = 1;

  auto input_dims = engine->getTensorShape(engine->getIOTensorName(input_index));
  size_t syndrome_size = 1;
  for (int i = 0; i < input_dims.nbDims; ++i) {
    syndrome_size *= input_dims.d[i];
  }

  auto output_dims = engine->getTensorShape(engine->getIOTensorName(output_index));
  size_t output_size = 1;
  for (int i = 0; i < output_dims.nbDims; ++i) {
    output_size *= output_dims.d[i];
  }

  persistent_ai_decoder::Config config;
  config.syndrome_size = syndrome_size;
  config.output_size = output_size;
  config.num_work_slots = 2;
  config.num_blocks = 1;

  persistent_ai_decoder decoder(engine.get(), context.get(), 
                                input_index, output_index, config);

  // Test initial state
  EXPECT_FALSE(decoder.is_running());

  // Test start
  EXPECT_TRUE(decoder.start());
  EXPECT_TRUE(decoder.is_running());

  // Give it a moment to actually launch
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(decoder.is_running());

  // Test double start (should fail)
  EXPECT_FALSE(decoder.start());

  // Test stop
  decoder.stop();
  EXPECT_FALSE(decoder.is_running());

  // Test double stop (should be safe)
  decoder.stop();
  EXPECT_FALSE(decoder.is_running());
}

// Test enqueue/dequeue operations
TEST_F(PersistentDecoderTest, EnqueueDequeueOperations) {
  std::string onnx_path = "../assets/tests/surface_code_decoder.onnx";
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model not found";
  }

  // Setup
  cuda_graph_utils::Logger logger;
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  auto engine = trt_decoder_internal::build_engine_from_onnx(onnx_path, params, logger);
  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

  int input_index = 0;
  int output_index = 1;

  auto input_dims = engine->getTensorShape(engine->getIOTensorName(input_index));
  size_t syndrome_size = 1;
  for (int i = 0; i < input_dims.nbDims; ++i) {
    syndrome_size *= input_dims.d[i];
  }

  auto output_dims = engine->getTensorShape(engine->getIOTensorName(output_index));
  size_t output_size = 1;
  for (int i = 0; i < output_dims.nbDims; ++i) {
    output_size *= output_dims.d[i];
  }

  persistent_ai_decoder::Config config;
  config.syndrome_size = syndrome_size;
  config.output_size = output_size;
  config.num_work_slots = 4;
  config.num_blocks = 2;

  persistent_ai_decoder decoder(engine.get(), context.get(), 
                                input_index, output_index, config);

  // Start decoder
  ASSERT_TRUE(decoder.start());

  // Test enqueue
  std::vector<float> test_syndrome(syndrome_size, 0.5f);
  EXPECT_TRUE(decoder.enqueue_syndrome(test_syndrome));

  // Give it time to process
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Test dequeue
  std::vector<float> result;
  EXPECT_TRUE(decoder.try_dequeue_result(result));
  EXPECT_EQ(result.size(), output_size);

  // Stop decoder
  decoder.stop();
}

// Test multiple syndromes
TEST_F(PersistentDecoderTest, MultipleSyndromes) {
  std::string onnx_path = "../assets/tests/surface_code_decoder.onnx";
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model not found";
  }

  // Setup
  cuda_graph_utils::Logger logger;
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  auto engine = trt_decoder_internal::build_engine_from_onnx(onnx_path, params, logger);
  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

  int input_index = 0;
  int output_index = 1;

  auto input_dims = engine->getTensorShape(engine->getIOTensorName(input_index));
  size_t syndrome_size = 1;
  for (int i = 0; i < input_dims.nbDims; ++i) {
    syndrome_size *= input_dims.d[i];
  }

  auto output_dims = engine->getTensorShape(engine->getIOTensorName(output_index));
  size_t output_size = 1;
  for (int i = 0; i < output_dims.nbDims; ++i) {
    output_size *= output_dims.d[i];
  }

  persistent_ai_decoder::Config config;
  config.syndrome_size = syndrome_size;
  config.output_size = output_size;
  config.num_work_slots = 8;
  config.num_blocks = 4;

  persistent_ai_decoder decoder(engine.get(), context.get(), 
                                input_index, output_index, config);

  ASSERT_TRUE(decoder.start());

  // Enqueue multiple syndromes
  const int num_syndromes = 10;
  std::cout << "Processing " << num_syndromes << " syndromes..." << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  int enqueued = 0;
  int dequeued = 0;

  // Producer-consumer pattern
  while (dequeued < num_syndromes) {
    // Enqueue more syndromes
    while (enqueued < num_syndromes) {
      std::vector<float> syndrome(syndrome_size, static_cast<float>(enqueued) / num_syndromes);
      if (decoder.enqueue_syndrome(syndrome)) {
        enqueued++;
      } else {
        break; // Queue full
      }
    }

    // Dequeue results
    std::vector<float> result;
    if (decoder.try_dequeue_result(result)) {
      EXPECT_EQ(result.size(), output_size);
      dequeued++;
    }

    // Brief sleep if blocked
    if (enqueued == dequeued) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_time - start_time).count();

  std::cout << "Processed " << num_syndromes << " syndromes in " << duration_us << " μs" << std::endl;
  std::cout << "Average latency: " << (duration_us / num_syndromes) << " μs/syndrome" << std::endl;

  EXPECT_EQ(dequeued, num_syndromes);

  decoder.stop();
}

// Note: More advanced tests (comparing with streaming decoder, stress testing, etc.)
// can be added here
