# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

FetchContent_Declare(
  prima
  GIT_REPOSITORY http://github.com/libprima/prima
  GIT_TAG v0.7.2
  GIT_SUBMODULES ""
)
if(NOT prima_POPULATED)
  FetchContent_Populate(prima)
  # Prima doesn't honor its PRIMA_ENABLE_TESTING flag. Regardless of its setting
  # test will be added. The workarround here is to define `add_test` to an empty
  # macro.
  macro(add_test)
  endmacro()
  set(BUILD_SHARED_LIBS OFF)
  add_subdirectory(${prima_SOURCE_DIR} ${prima_BINARY_DIR} EXCLUDE_FROM_ALL)
  macro(add_test)
    _add_test(${ARGN})
  endmacro()
endif()

# ==============================================================================

target_sources(cudaq-solvers PRIVATE cobyla.cpp)

target_include_directories(cudaq-solvers
  PUBLIC
    $<BUILD_INTERFACE:${prima_SOURCE_DIR}/prima/c/include>
)

target_link_libraries(cudaq-solvers PUBLIC primac gfortran) 
