# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-solvers-skqd)

# Check if CUDA is available
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  
  # Find CUDA libraries first
  find_package(CUDAToolkit REQUIRED)
  
  # CUDA kernel library - use STATIC to avoid linking issues
  add_library(${LIBRARY_NAME}-device STATIC
    device/skqd_kernels.cu
  )
  
  set_target_properties(${LIBRARY_NAME}-device PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  
  target_include_directories(${LIBRARY_NAME}-device
    PRIVATE
      ${CUDAQX_SOLVERS_INCLUDE_DIR}
      ${CUDA_INCLUDE_DIRS}
  )
  
  # Link to cudaq to get proper include paths
  target_link_libraries(${LIBRARY_NAME}-device
    PRIVATE
      cudaq::cudaq
      CUDA::cudart
  )
  
  # Main library with CUDA support
  target_sources(cudaq-solvers PRIVATE
    skqd.cpp
  )
  
  # Link CUDA libraries and device code
  target_link_libraries(cudaq-solvers
    PUBLIC
      CUDA::cudart
      ${LIBRARY_NAME}-device
  )
  
  # Link cuSOLVER for eigenvalue computation
  # Try to find cusolver library
  find_library(CUSOLVER_LIB 
    NAMES cusolver libcusolver.so.11 libcusolver.so
    PATHS ${CUDAToolkit_LIBRARY_DIR}
    NO_DEFAULT_PATH
  )
  
  if(CUSOLVER_LIB)
    target_link_libraries(cudaq-solvers PUBLIC ${CUSOLVER_LIB})
    message(STATUS "SKQD: cuSOLVER found at ${CUSOLVER_LIB}")
  else()
    message(WARNING "SKQD: cuSOLVER not found. Eigenvalue computation will be limited.")
  endif()
  
  # Link other CUDA libraries
  if(TARGET CUDA::cublas)
    target_link_libraries(cudaq-solvers PUBLIC CUDA::cublas)
  endif()
  
  if(TARGET CUDA::cusparse)
    target_link_libraries(cudaq-solvers PUBLIC CUDA::cusparse)
  endif()
  
else()
  message(WARNING "CUDA not found. SKQD will be compiled without GPU acceleration.")
  
  # Fallback: CPU-only version (if needed)
  target_sources(cudaq-solvers PRIVATE
    skqd.cpp
  )
endif()
