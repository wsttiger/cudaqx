#!/bin/bash
# Build script for TensorRT device-side launch reproducer

# Set TensorRT path (adjust as needed)
if [ -z "$TENSORRT_ROOT" ]; then
    # Try to find TensorRT in common locations
    if [ -d "/scratch/installed/TensorRT-10.13.3.9" ]; then
        export TENSORRT_ROOT=/scratch/installed/TensorRT-10.13.3.9
    else
        echo "Error: TENSORRT_ROOT not set and couldn't find TensorRT"
        echo "Please set: export TENSORRT_ROOT=/path/to/TensorRT"
        exit 1
    fi
fi

echo "Using TensorRT from: $TENSORRT_ROOT"

# Create build directory
mkdir -p build
cd build

# Configure
cmake .. -DTENSORRT_ROOT=$TENSORRT_ROOT

# Build
cmake --build . -j$(nproc)

echo ""
echo "Build complete!"
echo ""
echo "To run:"
echo "  cd build"
echo "  LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:\$LD_LIBRARY_PATH ./trt_device_launch_test ../../assets/tests/surface_code_decoder.onnx"
echo ""
echo "Or use the provided run script: ./run.sh"
