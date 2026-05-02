# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# External Dependencies
# ==============================================================================

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Bug in GCC 12 leads to spurious warnings (-Wrestrict)
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
if (CMAKE_COMPILER_IS_GNUCXX
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(gtest PUBLIC --param=evrp-mode=legacy)
endif()
include(GoogleTest)

# ==============================================================================
add_compile_options(-Wno-attributes)

add_executable(test_chromobius test_chromobius.cpp)
target_link_libraries(test_chromobius PRIVATE GTest::gtest_main cudaq-qec cudaq::cudaq)
add_dependencies(test_chromobius cudaq-qec-chromobius)
add_dependencies(CUDAQXQECUnitTests test_chromobius)
gtest_discover_tests(test_chromobius)
