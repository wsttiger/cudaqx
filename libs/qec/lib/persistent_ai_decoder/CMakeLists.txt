# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Find TensorRT (required for persistent_ai_decoder)
find_package(CUDAToolkit REQUIRED)

# Check for TensorRT availability
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY nvinfer
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
  message(STATUS "TensorRT found, building persistent_ai_decoder")
  
  # Enable CUDA language for .cu files
  enable_language(CUDA)
  
  # Find CUDA device runtime library for device-side API
  find_library(CUDA_DEVRT_LIBRARY cudadevrt
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    REQUIRED)
  
  # Create the persistent_ai_decoder library
  add_library(cudaq-qec-persistent-ai-decoder SHARED
    persistent_ai_decoder.cu
  )
  
  # Set CUDA architectures and compilation properties
  set_target_properties(cudaq-qec-persistent-ai-decoder PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"  # Volta, Turing, Ampere, Ada, Hopper
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  
  # Enable device-side runtime API (needed for cudaGraphLaunch from device)
  target_compile_options(cudaq-qec-persistent-ai-decoder PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--cudart shared>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
  )
  
  # Set the include directories
  target_include_directories(cudaq-qec-persistent-ai-decoder
    PUBLIC
      ${CMAKE_SOURCE_DIR}/libs/qec/include
      ${CMAKE_SOURCE_DIR}/libs/core/include
    PRIVATE
      ${TENSORRT_INCLUDE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS}
  )
  
  # Link against required libraries
  target_link_libraries(cudaq-qec-persistent-ai-decoder
    PUBLIC
      CUDA::cudart
    PRIVATE
      cudaq::cudaq-common
      ${TENSORRT_LIBRARY}
      cudaq-qec-utils  # For cuda_graph_utils
      ${CUDA_DEVRT_LIBRARY}  # Device runtime for device-side cudaGraphLaunch
  )
  
  # Set compile definitions
  target_compile_definitions(cudaq-qec-persistent-ai-decoder PRIVATE TENSORRT_AVAILABLE)
  
  # Install the library
  install(TARGETS cudaq-qec-persistent-ai-decoder 
    DESTINATION lib
    EXPORT cudaq-qec-persistent-ai-decoder-targets)
  
else()
  message(STATUS "TensorRT not found, skipping persistent_ai_decoder")
endif()
