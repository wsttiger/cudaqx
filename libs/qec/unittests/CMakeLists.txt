# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# External Dependencies 
# ==============================================================================

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(googletest)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Bug in GCC 12 leads to spurious warnings (-Wrestrict)
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
if (CMAKE_COMPILER_IS_GNUCXX 
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0 
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(gtest PUBLIC --param=evrp-mode=legacy)
endif()
include(GoogleTest)

find_package(CUDAToolkit REQUIRED)

# ==============================================================================
add_compile_options(-Wno-attributes) 

add_executable(test_decoders test_decoders.cpp decoders/sample_decoder.cpp)
target_link_libraries(test_decoders PRIVATE GTest::gtest_main cudaq-qec cudaq-qec-realtime-decoding cudaq::cudaq)
add_dependencies(CUDAQXQECUnitTests test_decoders)
gtest_discover_tests(test_decoders)

add_executable(test_decoders_yaml test_decoders_yaml.cpp decoders/sample_decoder.cpp)
target_link_libraries(test_decoders_yaml PRIVATE GTest::gtest_main cudaq-qec cudaq-qec-realtime-decoding cudaq::cudaq)
add_dependencies(CUDAQXQECUnitTests test_decoders_yaml)
gtest_discover_tests(test_decoders_yaml)

add_executable(test_qec test_qec.cpp)
target_link_libraries(test_qec PRIVATE GTest::gtest_main cudaq-qec cudaq::cudaq-stim-target)
add_dependencies(CUDAQXQECUnitTests test_qec)
gtest_discover_tests(test_qec)


# TensorRT decoder test is only built for x86 architectures
if(CUDAQ_QEC_BUILD_TRT_DECODER AND CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
  add_executable(test_trt_decoder ./decoders/trt_decoder/test_trt_decoder.cpp)

  # Find TensorRT for the test
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
      PATHS
          /usr/include/x86_64-linux-gnu
          /usr/local/cuda/include
          /usr/local/tensorrt/include
          /opt/tensorrt/include
      NO_DEFAULT_PATH
  )

  target_include_directories(test_trt_decoder PRIVATE
      ${CUDAToolkit_INCLUDE_DIRS}
  )

  target_link_libraries(test_trt_decoder PRIVATE GTest::gtest_main cudaq-qec cudaq-qec-trt-decoder cudaq::cudaq)
  add_dependencies(CUDAQXQECUnitTests test_trt_decoder)
  gtest_discover_tests(test_trt_decoder)
endif()

# ==============================================================================

# GPU kernel tests for realtime decoder infrastructure
if(CUDAQ_REALTIME_ROOT AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  
  add_executable(test_gpu_kernels 
    ${CMAKE_CURRENT_SOURCE_DIR}/decoders/realtime/test_gpu_kernels.cu
  )
  
  set_target_properties(test_gpu_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
  )
  
  target_include_directories(test_gpu_kernels PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/libs/core/include
  )
  
  # Define the test data directory path
  target_compile_definitions(test_gpu_kernels PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/decoders/realtime/data"
  )
  
  target_link_libraries(test_gpu_kernels PRIVATE 
    GTest::gtest_main 
    CUDA::cudart
    cudaq-qec-realtime-cudevice
  )
  
  add_dependencies(CUDAQXQECUnitTests test_gpu_kernels)
  gtest_discover_tests(test_gpu_kernels)
  
  # Realtime decoding end-to-end test with mock decoder
  # This test integrates with cuda-quantum's host API (libcudaq-realtime.so)

  # Optional hint for cuda-quantum install prefix
  set(CUDAQ_REALTIME_ROOT "" CACHE PATH "Prefix to cuda-quantum install")

  set(_cudaq_realtime_prefixes "")
  if(CUDAQ_REALTIME_ROOT)
    list(APPEND _cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}")
    list(APPEND _cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}/build")
  endif()
  if(CUDAQ_INSTALL_PREFIX)
    list(APPEND _cudaq_realtime_prefixes "${CUDAQ_INSTALL_PREFIX}")
  endif()

  # Realtime API lives under install prefix (CUDAQ_REALTIME_ROOT = install directory).
  # Header layout: include/cudaq/realtime/daemon/dispatcher/cudaq_realtime.h
  find_path(CUDAQ_REALTIME_INCLUDE_DIR
    NAMES cudaq/realtime/daemon/dispatcher/cudaq_realtime.h
    PATHS ${_cudaq_realtime_prefixes}
    PATH_SUFFIXES include
  )
  if(NOT CUDAQ_REALTIME_INCLUDE_DIR)
    find_path(CUDAQ_REALTIME_INCLUDE_DIR
      NAMES cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h
      PATHS ${_cudaq_realtime_prefixes}
      PATH_SUFFIXES include ../include
    )
  endif()

  find_library(CUDAQ_REALTIME_LIBRARY
    NAMES cudaq-realtime
    PATHS ${_cudaq_realtime_prefixes}
    PATH_SUFFIXES lib
  )

  find_library(CUDAQ_REALTIME_DISPATCH_LIBRARY
    NAMES cudaq-realtime-dispatch
    PATHS ${_cudaq_realtime_prefixes}
    PATH_SUFFIXES lib
  )

  # In-tree realtime (built from top-level add_subdirectory(realtime)) provides new API
  set(_predecoder_use_in_tree_realtime FALSE)
  if(TARGET cudaq-realtime)
    set(_predecoder_use_in_tree_realtime TRUE)
    message(STATUS "Using in-tree realtime (cudaq-realtime) for predecoder test")
  endif()

  set(_have_realtime_for_tests FALSE)
  if(CUDAQ_REALTIME_INCLUDE_DIR AND CUDAQ_REALTIME_LIBRARY AND CUDAQ_REALTIME_DISPATCH_LIBRARY)
    set(_have_realtime_for_tests TRUE)
    message(STATUS "Found cuda-quantum realtime headers at ${CUDAQ_REALTIME_INCLUDE_DIR}")
    message(STATUS "Found cuda-quantum realtime library at ${CUDAQ_REALTIME_LIBRARY}")
    message(STATUS "Found cuda-quantum realtime dispatch library at ${CUDAQ_REALTIME_DISPATCH_LIBRARY}")
  endif()
  if(TARGET cudaq-realtime)
    set(_have_realtime_for_tests TRUE)
  endif()

  if(_have_realtime_for_tests)

    add_executable(test_realtime_decoding 
      ${CMAKE_CURRENT_SOURCE_DIR}/decoders/realtime/test_realtime_decoding.cu
    )

    set_target_properties(test_realtime_decoding PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_STANDARD 17
    )

    target_include_directories(test_realtime_decoding PRIVATE
      ${CUDAToolkit_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/../include
      ${CMAKE_SOURCE_DIR}/libs/core/include
      ${CUDAQ_REALTIME_INCLUDE_DIR}
    )

    target_compile_definitions(test_realtime_decoding PRIVATE
      TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/decoders/realtime/data"
    )

    target_link_libraries(test_realtime_decoding PRIVATE 
      GTest::gtest_main 
      CUDA::cudart
      cudaq-qec-realtime-cudevice
      ${CUDAQ_REALTIME_LIBRARY}
      ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
    )

    # Ensure runtime can locate libcudaq-realtime.so
    get_filename_component(CUDAQ_REALTIME_LIB_DIR "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
    set_target_properties(test_realtime_decoding PROPERTIES
      BUILD_RPATH "${CUDAQ_REALTIME_LIB_DIR}"
      INSTALL_RPATH "${CUDAQ_REALTIME_LIB_DIR}"
    )

    add_dependencies(CUDAQXQECUnitTests test_realtime_decoding)
    gtest_discover_tests(test_realtime_decoding
      TEST_PREFIX "test_realtime_decoding."
    )
    # Hybrid AI predecoder + PyMatching pipeline test
    # Requires TensorRT + ONNX parser for building engines from ONNX models
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
      PATHS
        ${TENSORRT_ROOT}/include
        /usr/include/x86_64-linux-gnu
        /usr/local/cuda/include
        /usr/local/tensorrt/include
        /opt/tensorrt/include
      NO_DEFAULT_PATH
    )
    find_library(TENSORRT_LIBRARY nvinfer
      PATHS
        ${TENSORRT_ROOT}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/tensorrt/lib
        /opt/tensorrt/lib
    )
    find_library(TENSORRT_ONNX_PARSER_LIBRARY nvonnxparser
      PATHS
        ${TENSORRT_ROOT}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/tensorrt/lib
        /opt/tensorrt/lib
    )

    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND TENSORRT_ONNX_PARSER_LIBRARY)
      add_executable(test_realtime_predecoder_w_pymatching
        ${CMAKE_SOURCE_DIR}/libs/qec/lib/realtime/test_realtime_predecoder_w_pymatching.cpp
        ${CMAKE_SOURCE_DIR}/libs/qec/lib/realtime/ai_decoder_service.cu
        ${CMAKE_SOURCE_DIR}/libs/qec/lib/realtime/ai_predecoder_service.cu
      )

      set_target_properties(test_realtime_predecoder_w_pymatching PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD 17
        LINKER_LANGUAGE CUDA
      )

      target_compile_definitions(test_realtime_predecoder_w_pymatching PRIVATE
        ONNX_MODEL_DIR="${CMAKE_SOURCE_DIR}/libs/qec/lib/realtime"
      )

      # libcu++ (cuda/std/atomic) lives in CUDA toolkit under cccl/
      get_filename_component(_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
      get_filename_component(_cuda_root "${_cuda_bin}" DIRECTORY)
      set(_cuda_cccl_include "${_cuda_root}/include/cccl")

      # Includes: in-tree realtime target brings include; else in-repo or install dir
      set(_realtime_predecoder_includes "")
      if(NOT _predecoder_use_in_tree_realtime)
        set(_realtime_include "${CMAKE_SOURCE_DIR}/realtime/include")
        if(EXISTS "${_realtime_include}/cudaq/realtime/daemon/dispatcher/cudaq_realtime.h")
          list(APPEND _realtime_predecoder_includes "${_realtime_include}")
        endif()
      endif()
      target_include_directories(test_realtime_predecoder_w_pymatching PRIVATE
        ${_cuda_cccl_include}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/libs/core/include
        ${_realtime_predecoder_includes}
        ${CUDAQ_REALTIME_INCLUDE_DIR}
      )

      if(_predecoder_use_in_tree_realtime)
        target_link_libraries(test_realtime_predecoder_w_pymatching PRIVATE
          CUDA::cudart
          ${TENSORRT_LIBRARY}
          ${TENSORRT_ONNX_PARSER_LIBRARY}
          cudaq-realtime
          cudaq-realtime-host-dispatch
          cudaq-realtime-dispatch
          cudaq-qec
          cudaq::cudaq
        )
        set_target_properties(test_realtime_predecoder_w_pymatching PROPERTIES
          BUILD_RPATH "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/realtime/lib"
          INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/realtime/lib"
        )
      else()
        target_link_libraries(test_realtime_predecoder_w_pymatching PRIVATE
          CUDA::cudart
          ${TENSORRT_LIBRARY}
          ${TENSORRT_ONNX_PARSER_LIBRARY}
          ${CUDAQ_REALTIME_LIBRARY}
          ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
          cudaq-qec
          cudaq::cudaq
        )
        target_link_directories(test_realtime_predecoder_w_pymatching PRIVATE
          ${CMAKE_BINARY_DIR}/lib
        )
        set_target_properties(test_realtime_predecoder_w_pymatching PROPERTIES
          BUILD_RPATH "${CUDAQ_REALTIME_LIB_DIR};${CMAKE_BINARY_DIR}/lib"
          INSTALL_RPATH "${CUDAQ_REALTIME_LIB_DIR};${CMAKE_BINARY_DIR}/lib"
        )
      endif()

      add_dependencies(CUDAQXQECUnitTests test_realtime_predecoder_w_pymatching)
    else()
      message(WARNING "TensorRT or ONNX parser not found. Skipping test_realtime_predecoder_w_pymatching.")
    endif()

  else()
    message(WARNING "cuda-quantum realtime dependency not found. "
                    "Set CUDAQ_REALTIME_ROOT or build with in-tree realtime to enable "
                    "test_realtime_decoding and test_realtime_predecoder_w_pymatching.")
  endif()
endif()

# ==============================================================================

add_subdirectory(backend-specific)
add_subdirectory(realtime)
add_subdirectory(decoders/pymatching)

