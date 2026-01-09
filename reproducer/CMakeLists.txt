cmake_minimum_required(VERSION 3.24)
project(TensorRT_DeviceLaunch_Reproducer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find TensorRT
# Set TENSORRT_ROOT environment variable or pass -DTENSORRT_ROOT=...
if(NOT DEFINED TENSORRT_ROOT)
    set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
endif()

if(NOT TENSORRT_ROOT)
    message(FATAL_ERROR "TENSORRT_ROOT not set. Please set it to your TensorRT installation directory.")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_ONNX_PARSER nvonnxparser
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64 lib/x64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_ONNX_PARSER)
    message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_ROOT correctly.")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
message(STATUS "TensorRT ONNX parser: ${TENSORRT_ONNX_PARSER}")

# Create executable
add_executable(trt_device_launch_test 
    trt_device_launch_test.cpp)

# Set include directories
target_include_directories(trt_device_launch_test PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})

# Link libraries
target_link_libraries(trt_device_launch_test PRIVATE
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNX_PARSER}
    CUDA::cudart)

# Set CUDA architectures (adjust as needed for your GPU)
set_target_properties(trt_device_launch_test PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90")
