# TensorRT Device-Side Graph Launch Issue - Standalone Reproducer

## Location
`/scratch/cudaqx/reproducer/`

## Files Included

```
reproducer/
├── trt_device_launch_test.cpp    # Main standalone reproducer (500 lines)
├── CMakeLists.txt                # Build configuration
├── README.md                     # Detailed documentation
├── build.sh                      # Build script with auto-detection
├── run.sh                        # Run script with proper library paths
└── REPRODUCER_SUMMARY.md         # This file
```

## Quick Start

```bash
cd /scratch/cudaqx/reproducer

# Build (auto-detects TensorRT)
./build.sh

# Run with test model
./run.sh ../assets/tests/surface_code_decoder.onnx

# Or run with your own ONNX model
./run.sh /path/to/your/model.onnx
```

## What This Reproducer Does

1. **Loads a TensorRT ONNX model** and builds an engine
2. **Captures TensorRT inference into a CUDA graph**
3. **Tests regular graph instantiation** (succeeds ✓)
4. **Tests device-side launch instantiation** (fails ✗)
5. **Reports the specific error** and confirms the incompatibility

## Expected Output

```
=== TensorRT Device-Side Graph Launch Reproducer ===

GPU: NVIDIA RTX 6000 Ada Generation
Compute Capability: 8.9
Building TensorRT engine...
Engine built successfully

=== Testing CUDA Graph Capture ===

--- Step 1: Capturing CUDA graph ---
✓ Graph captured successfully

--- Step 2: Testing regular instantiation ---
✓ Regular instantiation SUCCEEDED

--- Step 3: Testing device-side launch instantiation ---
Flags: cudaGraphInstantiateFlagDeviceLaunch | cudaGraphInstantiateFlagAutoFreeOnLaunch
✗ Device-side launch instantiation FAILED
  Error: invalid argument
  This confirms TensorRT operations are NOT compatible with device-side graph launch

=== Summary ===
TensorRT CUDA graphs work with host-side launch (regular instantiation)
TensorRT CUDA graphs do NOT work with device-side launch (DeviceLaunch flag)

This means persistent GPU kernels cannot directly launch TensorRT graphs.
```

## The Core Issue

The reproducer proves that:

1. **TensorRT operations CAN be captured into CUDA graphs** ✓
2. **These graphs CAN be instantiated with `cudaGraphInstantiate()`** ✓
3. **These graphs CANNOT be instantiated with `cudaGraphInstantiateFlagDeviceLaunch`** ✗

The failure at step 3 means:
- CUDA graphs containing TensorRT operations cannot be launched from device code
- Persistent GPU kernels cannot directly invoke TensorRT inference
- Host-side scheduling is required, adding latency overhead

## Code Highlights

### Key Test Section (from trt_device_launch_test.cpp)

```cpp
// Step 1: Capture graph (works)
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
context->enqueueV3(stream);  // TensorRT inference
cudaStreamEndCapture(stream, &graph);

// Step 2: Regular instantiation (works)
err = cudaGraphInstantiate(&graphExec, graph, 0);
// Result: SUCCESS ✓

// Step 3: Device-side launch instantiation (fails)
unsigned long long flags = cudaGraphInstantiateFlagDeviceLaunch | 
                           cudaGraphInstantiateFlagAutoFreeOnLaunch;
err = cudaGraphInstantiateWithFlags(&graphExec, graph, flags);
// Result: FAIL ✗ - Error: invalid argument
```

## Why This Matters

### Original Goal: Persistent AI Decoder
The persistent AI decoder architecture requires:
- A GPU kernel that runs continuously
- Device-side launching of TensorRT inference graphs
- Zero host involvement during inference (ultra-low latency)

### Current Limitation
Without device-side graph launch support:
- Must use host-side scheduling
- Adds host-device synchronization overhead
- Cannot achieve the ultra-low latency goal
- Persistent kernel pattern not viable with TensorRT

## Questions for NVIDIA TensorRT Team

1. **Is this a known limitation?**
   - Are TensorRT operations fundamentally incompatible with device-side graph launch?
   
2. **Are there any workarounds?**
   - Engine build flags?
   - Specific TensorRT API usage patterns?
   - Alternative approaches?

3. **Future support?**
   - Is this on the roadmap for future TensorRT releases?
   - What would be required to enable it?

## Environment

- **GPU**: NVIDIA RTX 6000 Ada Generation (Compute Capability 8.9)
- **CUDA**: 12.6
- **TensorRT**: 10.13.3
- **Driver**: 575.51.03
- **OS**: Linux 6.8.0

## Sending to NVIDIA

This reproducer is completely standalone:
- No CUDA-Q dependencies
- Single source file
- Standard CMake build
- Works with any ONNX model

Simply tar up the `reproducer/` directory and send to NVIDIA:

```bash
cd /scratch/cudaqx
tar czf tensorrt_device_launch_reproducer.tar.gz reproducer/
```

## Related Code in Main Project

The issue was discovered while implementing:
- `libs/qec/lib/persistent_ai_decoder/persistent_ai_decoder.cu`
- `libs/qec/lib/utils/cuda_graph_utils.cpp`
- `libs/qec/unittests/decoders/trt_decoder/test_persistent_decoder.cpp`

The persistent decoder tests all fail with the same root cause demonstrated here.
