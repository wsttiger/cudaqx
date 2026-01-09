#!/bin/bash
# Run script for TensorRT device-side launch reproducer

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

# Set library path
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH

# Default ONNX model path (relative to build directory)
ONNX_PATH="../../assets/tests/surface_code_decoder.onnx"

# Override with command line argument if provided
if [ $# -gt 0 ]; then
    ONNX_PATH="$1"
fi

# Check if executable exists
if [ ! -f "build/trt_device_launch_test" ]; then
    echo "Error: Executable not found. Please build first:"
    echo "  ./build.sh"
    exit 1
fi

echo "Running reproducer with ONNX model: $ONNX_PATH"
echo ""

cd build
./trt_device_launch_test "$ONNX_PATH"
