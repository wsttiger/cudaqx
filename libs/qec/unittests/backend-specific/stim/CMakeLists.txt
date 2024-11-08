# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# FIXME This needs to be in CUDA-Q.
add_library(cudaq::cudaq-stim-target SHARED IMPORTED)
set_target_properties(cudaq::cudaq-stim-target PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libnvqir-stim${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libnvqir-stim${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-default;cudaq::cudaq-em-default")

cudaqx_set_target(stim) 

add_executable(test_qec_stim test_qec_stim.cpp)
target_link_libraries(test_qec_stim
  PRIVATE
    GTest::gtest_main
    cudaq::cudaq_stim
    cudaq-qec
)
add_dependencies(CUDAQXQECUnitTests test_qec_stim)
gtest_discover_tests(test_qec_stim)
