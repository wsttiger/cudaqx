# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(CUDA_FOUND)
  add_library(cudaq-realtime-pipeline SHARED
    realtime_pipeline.cu
  )

  target_include_directories(cudaq-realtime-pipeline
    PUBLIC
      $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(cudaq-realtime-pipeline
    PUBLIC
      CUDA::cudart_static
    PRIVATE
      cudaq-realtime
      cudaq-realtime-host-dispatch
  )

  set_target_properties(cudaq-realtime-pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )

  install(TARGETS cudaq-realtime-pipeline
    COMPONENT realtime-lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
