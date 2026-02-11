# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(MODULE_NAME "cudaq-qec-composite-decoder")

project(${MODULE_NAME})

# Composite decoder chains trt_decoder (pre-decoder) with a global decoder
# (e.g. pymatching). Build only when TRT decoder is enabled.
if(NOT CUDAQ_QEC_BUILD_TRT_DECODER)
  message(WARNING "composite_decoder requires CUDAQ_QEC_BUILD_TRT_DECODER; skipping ${MODULE_NAME} build.")
  return()
endif()

set(PLUGIN_SRC
  composite_decoder.cpp
)

add_library(${MODULE_NAME} SHARED ${PLUGIN_SRC})

target_include_directories(${MODULE_NAME}
  PUBLIC
    ${CMAKE_SOURCE_DIR}/libs/qec/include
    ${CMAKE_SOURCE_DIR}/libs/core/include
)

target_link_libraries(${MODULE_NAME}
  PUBLIC
    cudaqx-core
    cudaq::cudaq-operator
  PRIVATE
    cudaq::cudaq-common
    cudaq-qec
)

set_target_properties(${MODULE_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/decoder-plugins
)

if (NOT SKBUILD)
  set_target_properties(${MODULE_NAME} PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
  )
  set_target_properties(${MODULE_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
  set_target_properties(${MODULE_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../lib"
  )
endif()

install(TARGETS ${MODULE_NAME}
  COMPONENT qec-lib-plugins
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/decoder-plugins
)
