# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Find TensorRT (required for cuda_graph_utils)
find_package(CUDAToolkit REQUIRED)

# Check for TensorRT availability
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY nvinfer
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
  message(STATUS "TensorRT found, building cuda_graph_utils")
  
  # Create the utils library
  add_library(cudaq-qec-utils SHARED
    cuda_graph_utils.cpp
  )
  
  # Set the include directories
  target_include_directories(cudaq-qec-utils
    PUBLIC
      ${CMAKE_SOURCE_DIR}/libs/qec/include
      ${CMAKE_SOURCE_DIR}/libs/core/include
    PRIVATE
      ${TENSORRT_INCLUDE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS}
  )
  
  # Link against required libraries
  target_link_libraries(cudaq-qec-utils
    PUBLIC
      CUDA::cudart
    PRIVATE
      cudaq::cudaq-common
      ${TENSORRT_LIBRARY}
  )
  
  # Set compile definitions
  target_compile_definitions(cudaq-qec-utils PRIVATE TENSORRT_AVAILABLE)
  
  # Install the library
  install(TARGETS cudaq-qec-utils 
    DESTINATION lib
    EXPORT cudaq-qec-utils-targets)
  
else()
  message(STATUS "TensorRT not found, skipping cuda_graph_utils")
endif()
