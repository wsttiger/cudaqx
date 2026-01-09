# TensorRT Device-Side Graph Launch Issue Reproducer

## Problem Statement

This reproducer demonstrates that **TensorRT operations captured in CUDA graphs cannot be instantiated with the `cudaGraphInstantiateFlagDeviceLaunch` flag**, which is required for device-side graph launches from GPU kernels.

### Impact

This limitation prevents implementation of truly persistent GPU kernels that can directly launch TensorRT inference graphs from device code, which would be beneficial for:
- Ultra-low latency inference (eliminating host-device scheduling overhead)
- Persistent AI decoder architectures
- GPU-resident inference pipelines

## Technical Details

### What Works
- TensorRT operations can be captured into CUDA graphs using `cudaStreamBeginCapture` / `cudaStreamEndCapture`
- These graphs can be instantiated with regular `cudaGraphInstantiate()` 
- Host-side launch with `cudaGraphLaunch()` works fine

### What Fails
- When attempting to instantiate the same graph with `cudaGraphInstantiateWithFlags()` using `cudaGraphInstantiateFlagDeviceLaunch` flag
- Error returned: `cudaError_t::cudaErrorInvalidValue` ("invalid argument")
- This means the graph cannot be launched from device-side (GPU kernel) code

### Root Cause (Hypothesis)
TensorRT's internal CUDA operations may include:
- Host callbacks
- Operations incompatible with device-side launch restrictions
- Memory operations that aren't supported in device-launchable graphs

## System Requirements

- CUDA 12.0+
- TensorRT 8.0+ (tested with 10.x)
- GPU with Compute Capability 7.5+ (for device-side launch support)
- CMake 3.24+
- C++17 compiler

## Build Instructions

```bash
# Set TensorRT path
export TENSORRT_ROOT=/path/to/TensorRT-10.x

# Create build directory
mkdir build
cd build

# Configure
cmake .. -DTENSORRT_ROOT=$TENSORRT_ROOT

# Build
cmake --build .
```

## Running the Reproducer

```bash
# From build directory
./trt_device_launch_test ../assets/tests/surface_code_decoder.onnx

# Or with your own ONNX model
./trt_device_launch_test /path/to/your/model.onnx
```

## Expected Output

```
=== TensorRT Device-Side Graph Launch Reproducer ===

ONNX model path: ../assets/tests/surface_code_decoder.onnx
GPU: NVIDIA RTX 6000 Ada Generation
Compute Capability: 8.9
Building TensorRT engine from: ../assets/tests/surface_code_decoder.onnx
...

=== Testing CUDA Graph Capture ===
...

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

## Files Included

- `trt_device_launch_test.cpp` - Standalone reproducer with detailed comments
- `CMakeLists.txt` - Build configuration
- `README.md` - This file

## Questions for NVIDIA TensorRT Team

1. **Is device-side CUDA graph launch intended to be supported for TensorRT?**
   - If yes: What conditions must be met? (specific TensorRT version, engine configuration, etc.)
   - If no: Are there any plans to support this in the future?

2. **What specific TensorRT operations are incompatible with device-side launch?**
   - Can these be identified and potentially avoided?
   - Are there engine build flags that could enable compatibility?

3. **Alternative approaches for low-latency GPU-resident inference?**
   - Is there a recommended way to achieve similar functionality?
   - Can CUDA MPS or other technologies help bridge this gap?

## Workarounds Considered

### Option 1: Host-side launch from persistent kernel
- Launch graphs from host in response to device signals
- Higher latency due to host-device synchronization

### Option 2: Polling-based host scheduler
- Host thread continuously polls device memory
- Better latency but still involves host CPU

### Option 3: CUDA Graph sequences with conditional nodes (CUDA 12.3+)
- Use graph conditional nodes for flow control
- Limited flexibility, may not suit all use cases

None of these achieve the ultra-low latency of true device-side launch.

## Environment Information

- Hardware: NVIDIA RTX 6000 Ada Generation (Compute Capability 8.9)
- CUDA Version: 12.6
- TensorRT Version: 10.x
- Driver Version: 575.51.03

## Contact

For questions about this reproducer or the underlying issue, please contact the CUDA-Q development team.
