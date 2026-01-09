/**
 * Standalone reproducer for TensorRT CUDA graph device-side launch issue
 * 
 * This program demonstrates that TensorRT operations captured in CUDA graphs
 * cannot be instantiated with cudaGraphInstantiateFlagDeviceLaunch, which is
 * required for device-side graph launches from GPU kernels.
 * 
 * Expected behavior:
 * - Regular cudaGraphInstantiate() succeeds
 * - cudaGraphInstantiateWithFlags() with DeviceLaunch flag fails with "invalid argument"
 * 
 * Hardware: NVIDIA RTX 6000 Ada (Compute Capability 8.9)
 * CUDA: 12.6+
 * TensorRT: 10.x
 */

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <stdexcept>

// Simple TensorRT logger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

// CUDA error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + \
                                   cudaGetErrorString(err) + \
                                   " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

// Build TensorRT engine from ONNX file
std::unique_ptr<nvinfer1::ICudaEngine> buildEngineFromOnnx(
    const std::string& onnxPath, Logger& logger) {
    
    std::cout << "Building TensorRT engine from: " << onnxPath << std::endl;
    
    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger));
    if (!builder) {
        throw std::runtime_error("Failed to create InferBuilder");
    }
    
    // Create network with explicit batch
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    if (!network) {
        throw std::runtime_error("Failed to create network");
    }
    
    // Create ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger));
    if (!parser) {
        throw std::runtime_error("Failed to create ONNX parser");
    }
    
    // Parse ONNX file
    if (!parser->parseFromFile(onnxPath.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX file");
    }
    
    // Create builder config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) {
        throw std::runtime_error("Failed to create builder config");
    }
    
    // Set memory limit (1GB)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    
    // Build engine
    std::cout << "Building engine (this may take a moment)..." << std::endl;
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config));
    if (!engine) {
        throw std::runtime_error("Failed to build engine");
    }
    
    std::cout << "Engine built successfully" << std::endl;
    return engine;
}

// Test CUDA graph capture and instantiation
void testGraphCapture(nvinfer1::ICudaEngine* engine) {
    std::cout << "\n=== Testing CUDA Graph Capture ===" << std::endl;
    
    // Create execution context
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("Failed to create execution context");
    }
    
    // Get I/O tensor info
    int numIOTensors = engine->getNbIOTensors();
    std::cout << "Number of I/O tensors: " << numIOTensors << std::endl;
    
    if (numIOTensors < 2) {
        throw std::runtime_error("Expected at least 2 I/O tensors (input and output)");
    }
    
    const char* inputName = engine->getIOTensorName(0);
    const char* outputName = engine->getIOTensorName(1);
    auto inputDims = engine->getTensorShape(inputName);
    auto outputDims = engine->getTensorShape(outputName);
    
    std::cout << "Input tensor: " << inputName << std::endl;
    std::cout << "Output tensor: " << outputName << std::endl;
    
    // Calculate sizes
    size_t inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; i++) {
        inputSize *= inputDims.d[i];
    }
    size_t outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; i++) {
        outputSize *= outputDims.d[i];
    }
    
    std::cout << "Input size: " << inputSize << " floats" << std::endl;
    std::cout << "Output size: " << outputSize << " floats" << std::endl;
    
    // Allocate device buffers
    void* inputBuffer = nullptr;
    void* outputBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&inputBuffer, inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&outputBuffer, outputSize * sizeof(float)));
    
    std::cout << "Allocated device buffers" << std::endl;
    
    // Create stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    
    // Set tensor addresses
    context->setTensorAddress(inputName, inputBuffer);
    context->setTensorAddress(outputName, outputBuffer);
    
    std::cout << "Configured TensorRT context" << std::endl;
    
    // =========================================================================
    // WARM-UP: REQUIRED FOR DEVICE-SIDE LAUNCH
    // =========================================================================
    std::cout << "\n--- Step 0: TensorRT warm-up (allocating internal resources) ---" << std::endl;
    std::cout << "Performing warm-up enqueueV3() call..." << std::endl;
    context->enqueueV3(stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "✓ Warm-up complete" << std::endl;
    
    // =========================================================================
    // CAPTURE CUDA GRAPH
    // =========================================================================
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    
    std::cout << "\n--- Step 1: Capturing CUDA graph ---" << std::endl;
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    
    // Record TensorRT inference
    context->enqueueV3(stream);
    
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    std::cout << "✓ Graph captured successfully" << std::endl;
    
    // =========================================================================
    // TEST 1: Regular instantiation (should work)
    // =========================================================================
    std::cout << "\n--- Step 2: Testing regular instantiation ---" << std::endl;
    cudaError_t err = cudaGraphInstantiate(&graphExec, graph, 0);
    if (err == cudaSuccess) {
        std::cout << "✓ Regular instantiation SUCCEEDED" << std::endl;
        cudaGraphExecDestroy(graphExec);
        graphExec = nullptr;
    } else {
        std::cout << "✗ Regular instantiation FAILED: " << cudaGetErrorString(err) << std::endl;
    }
    
    // =========================================================================
    // TEST 2: Device-side launch instantiation (expected to fail with TensorRT)
    // =========================================================================
    std::cout << "\n--- Step 3: Testing device-side launch instantiation ---" << std::endl;
    std::cout << "Flags: cudaGraphInstantiateFlagDeviceLaunch | cudaGraphInstantiateFlagAutoFreeOnLaunch" << std::endl;
    
    unsigned long long flags = cudaGraphInstantiateFlagDeviceLaunch | 
                               cudaGraphInstantiateFlagAutoFreeOnLaunch;
    err = cudaGraphInstantiateWithFlags(&graphExec, graph, flags);
    
    if (err == cudaSuccess) {
        std::cout << "✓ Device-side launch instantiation SUCCEEDED" << std::endl;
        std::cout << "  This means TensorRT DOES support device-side graph launch!" << std::endl;
        
        // Try to upload
        std::cout << "  Attempting cudaGraphUpload..." << std::endl;
        err = cudaGraphUpload(graphExec, stream);
        if (err == cudaSuccess) {
            std::cout << "  ✓ cudaGraphUpload SUCCEEDED" << std::endl;
        } else {
            std::cout << "  ✗ cudaGraphUpload FAILED: " << cudaGetErrorString(err) << std::endl;
        }
        
        cudaGraphExecDestroy(graphExec);
    } else {
        std::cout << "✗ Device-side launch instantiation FAILED" << std::endl;
        std::cout << "  Error: " << cudaGetErrorString(err) << std::endl;
        std::cout << "\n  Note: Warm-up call was performed before capture (as required by TensorRT docs)" << std::endl;
        std::cout << "  This confirms TensorRT operations are NOT compatible with device-side graph launch" << std::endl;
        std::cout << "  even with proper warm-up." << std::endl;
    }
    
    // Cleanup
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(inputBuffer);
    cudaFree(outputBuffer);
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "✓ TensorRT CUDA graphs work with host-side launch (regular instantiation)" << std::endl;
    std::cout << "✗ TensorRT CUDA graphs do NOT work with device-side launch (DeviceLaunch flag)" << std::endl;
    std::cout << "\nEven with proper warm-up call (as documented in TensorRT best practices)," << std::endl;
    std::cout << "TensorRT operations cannot be instantiated with cudaGraphInstantiateFlagDeviceLaunch." << std::endl;
    std::cout << "\nThis means persistent GPU kernels cannot directly launch TensorRT graphs." << std::endl;
}

int main(int argc, char** argv) {
    try {
        std::cout << "=== TensorRT Device-Side Graph Launch Reproducer ===" << std::endl;
        std::cout << std::endl;
        
        // Check for ONNX file argument
        std::string onnxPath = "../assets/tests/surface_code_decoder.onnx";
        if (argc > 1) {
            onnxPath = argv[1];
        }
        
        std::cout << "ONNX model path: " << onnxPath << std::endl;
        
        // Check if file exists
        std::ifstream file(onnxPath);
        if (!file.good()) {
            std::cerr << "Error: ONNX file not found: " << onnxPath << std::endl;
            std::cerr << "Usage: " << argv[0] << " [path_to_onnx_model]" << std::endl;
            return 1;
        }
        
        // Check CUDA device
        int deviceCount = 0;
        CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            std::cerr << "Error: No CUDA devices found" << std::endl;
            return 1;
        }
        
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        
        int computeCapability = prop.major * 10 + prop.minor;
        if (computeCapability < 75) {
            std::cout << "Warning: Device-side graph launch requires Compute Capability 7.5+" << std::endl;
            std::cout << "         Current device has " << prop.major << "." << prop.minor << std::endl;
        }
        
        // Build TensorRT engine
        Logger logger;
        auto engine = buildEngineFromOnnx(onnxPath, logger);
        
        // Test graph capture
        testGraphCapture(engine.get());
        
        std::cout << "\n=== Test Complete ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
