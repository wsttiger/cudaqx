/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Example demonstrating the UPDATED persistent_ai_decoder
// NOTE: This is not compiled by default - it's for reference only

#include "cudaq/qec/persistent_ai_decoder.h"
#include "cudaq/qec/cuda_graph_utils.h"
#include "cudaq/qec/trt_decoder_internal.h"
#include <chrono>
#include <iostream>
#include <vector>

// Example usage of the Persistent AI Decoder (Updated with graph-per-slot)
// This demonstrates the complete workflow

void example_persistent_decoder_usage() {
  // Step 1: Load or build a TensorRT engine
  std::string engine_path = "path/to/your/decoder_engine.trt";

  // Load the engine (pseudo-code)
  nvinfer1::IRuntime *runtime = nullptr; // Create runtime
  nvinfer1::ICudaEngine *engine = nullptr; // Load engine
  nvinfer1::IExecutionContext *context = nullptr; // Create context

  // Step 2: Get tensor information
  int input_index = 0;
  int output_index = 1;

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

  std::cout << "Model info: syndrome_size=" << syndrome_size
            << ", output_size=" << output_size << std::endl;

  // Step 3: Check if CUDA graphs are supported
  if (!cudaq::qec::cuda_graph_utils::supports_cuda_graphs(engine)) {
    std::cerr << "Error: CUDA graphs not supported for this model" << std::endl;
    return;
  }

  // Step 4: Configure the persistent decoder
  cudaq::qec::persistent_ai_decoder::Config config;
  config.syndrome_size = syndrome_size;
  config.output_size = output_size;
  config.num_work_slots = 16;     // Will capture 16 graphs
  config.num_blocks = 8;          // Fewer blocks for persistent pattern
  config.threads_per_block = 256;

  // Step 5: Create the persistent decoder
  // NOTE: This will capture one CUDA graph per work slot
  // Initialization may take a few seconds
  std::cout << "Creating persistent decoder (capturing " << config.num_work_slots 
            << " graphs, this may take a moment)..." << std::endl;
  
  auto start_init = std::chrono::high_resolution_clock::now();
  
  cudaq::qec::persistent_ai_decoder decoder(
      engine, context, input_index, output_index, config);
  
  auto end_init = std::chrono::high_resolution_clock::now();
  auto init_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_init - start_init).count();
  
  std::cout << "Decoder initialized in " << init_time << " ms" << std::endl;

  // Step 6: Start the persistent kernel
  std::cout << "Starting persistent decoder (launching GPU kernel)..." << std::endl;
  if (!decoder.start()) {
    std::cerr << "Failed to start persistent decoder" << std::endl;
    return;
  }

  std::cout << "Persistent kernel is now running on GPU!" << std::endl;

  // Step 7: Process syndromes
  const int num_syndromes = 100;
  std::cout << "Processing " << num_syndromes << " syndromes..." << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  int enqueued = 0;
  int dequeued = 0;

  // Producer-consumer pattern
  while (dequeued < num_syndromes) {
    // Enqueue more syndromes if we haven't sent them all
    while (enqueued < num_syndromes) {
      // Create test syndrome
      std::vector<float> syndrome(syndrome_size, 
                                   static_cast<float>(enqueued) / num_syndromes);
      
      if (decoder.enqueue_syndrome(syndrome)) {
        enqueued++;
        if (enqueued % 20 == 0) {
          std::cout << "Enqueued " << enqueued << " syndromes" << std::endl;
        }
      } else {
        // Queue full, try again later
        break;
      }
    }

    // Dequeue results
    std::vector<float> result;
    if (decoder.try_dequeue_result(result)) {
      dequeued++;
      if (dequeued % 20 == 0) {
        std::cout << "Dequeued " << dequeued << " results" << std::endl;
      }
    }

    // Brief sleep if both operations blocked
    if (enqueued == dequeued) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - start_time).count();

  std::cout << "\nProcessed " << num_syndromes << " syndromes in " << duration
            << " μs" << std::endl;
  std::cout << "Average latency: " << (duration / num_syndromes) << " μs"
            << std::endl;
  std::cout << "Throughput: " 
            << (num_syndromes * 1000000.0 / duration) << " syndromes/sec"
            << std::endl;

  // Step 8: Stop the persistent decoder
  std::cout << "\nStopping persistent decoder..." << std::endl;
  decoder.stop();

  std::cout << "Done!" << std::endl;
}

// Comparison benchmark: Streaming vs Persistent
void benchmark_comparison() {
  std::cout << "Benchmark: Persistent Decoder" << std::endl;
  std::cout << "=============================" << std::endl;
  
  // This would:
  // 1. Run workload with persistent_ai_decoder
  // 2. Measure:
  //    - Initialization time
  //    - Average latency per syndrome
  //    - Throughput
  //    - Memory usage
  //    - Latency distribution (min/max/p50/p99)
  
  std::cout << "\nExpected results:" << std::endl;
  std::cout << "- Init time:    ~1000-5000 ms (captures N graphs)" << std::endl;
  std::cout << "- Latency:      ~10-100 μs" << std::endl;
  std::cout << "- Jitter:       Very low (GPU scheduling)" << std::endl;
}

// Understanding the graph-per-slot approach
void explain_graph_per_slot() {
  std::cout << "\n==================================================" << std::endl;
  std::cout << "How Graph-Per-Slot Works" << std::endl;
  std::cout << "==================================================" << std::endl;
  
  std::cout << "\n1. INITIALIZATION (Host-side, one-time):" << std::endl;
  std::cout << "   for each work slot:" << std::endl;
  std::cout << "     - Allocate input_buffer[i], output_buffer[i]" << std::endl;
  std::cout << "     - Set TensorRT to use these buffers" << std::endl;
  std::cout << "     - context->setTensorAddress('input', input_buffer[i])" << std::endl;
  std::cout << "     - context->setTensorAddress('output', output_buffer[i])" << std::endl;
  std::cout << "     - Capture: cudaStreamBeginCapture()" << std::endl;
  std::cout << "     - context->enqueueV3()  // Records into graph" << std::endl;
  std::cout << "     - cudaStreamEndCapture(&graph[i])" << std::endl;
  std::cout << "     - cudaGraphInstantiate(&graph_exec[i], graph[i])" << std::endl;
  std::cout << "   Result: Each slot has its own graph tied to its buffers\n" << std::endl;
  
  std::cout << "2. EXECUTION (GPU-side, persistent):" << std::endl;
  std::cout << "   GPU Persistent Kernel:" << std::endl;
  std::cout << "     while (!stop):" << std::endl;
  std::cout << "       slot = find_work_slot()  // Uses atomics" << std::endl;
  std::cout << "       if (slot >= 0):" << std::endl;
  std::cout << "         // Just launch this slot's graph!" << std::endl;
  std::cout << "         cudaGraphLaunch(graph_exec[slot], stream[slot])" << std::endl;
  std::cout << "         cudaStreamSynchronize(stream[slot])" << std::endl;
  std::cout << "         mark_complete(slot)" << std::endl;
  std::cout << "   Result: No TensorRT API calls, no parameter updates!\n" << std::endl;
  
  std::cout << "3. KEY INSIGHT:" << std::endl;
  std::cout << "   - Graph was captured WITH specific buffers" << std::endl;
  std::cout << "   - Graph REMEMBERS those buffers" << std::endl;
  std::cout << "   - Just launch the graph, it uses the right buffers!" << std::endl;
  std::cout << "   - No need for enqueueV3() after capture!" << std::endl;
  
  std::cout << "\n==================================================" << std::endl;
}

int main() {
  std::cout << "Persistent AI Decoder Example (Updated!)" << std::endl;
  std::cout << "=========================================" << std::endl;
  std::cout << std::endl;

  std::cout << "This example demonstrates the UPDATED persistent decoder" << std::endl;
  std::cout << "that uses the 'graph-per-slot' approach." << std::endl;
  std::cout << std::endl;

  std::cout << "Key insight: Once captured, CUDA graphs don't need TensorRT APIs!" << std::endl;
  std::cout << "Each work slot has its own graph tied to its buffers." << std::endl;
  std::cout << std::endl;

  // Explain the approach
  explain_graph_per_slot();

  std::cout << "\nNOTE: To run the actual example, you need to:" << std::endl;
  std::cout << "1. Have a trained TensorRT model for decoding" << std::endl;
  std::cout << "2. Update the paths and configuration" << std::endl;
  std::cout << "3. Compile with the TRT decoder plugin" << std::endl;
  std::cout << "4. Have a GPU with compute capability 7.0+ (Volta or newer)" << std::endl;
  std::cout << std::endl;

  // Uncomment to run (after setup):
  // example_persistent_decoder_usage();
  // benchmark_comparison();

  return 0;
}
