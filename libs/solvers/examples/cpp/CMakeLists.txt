# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(qsvt_bring_your_own_phases EXCLUDE_FROM_ALL
  qsvt_bring_your_own_phases.cpp
)

target_link_libraries(qsvt_bring_your_own_phases
  PRIVATE
    cudaq-solvers
    cudaq::cudaq
)

add_executable(qsvt_hamiltonian_simulation EXCLUDE_FROM_ALL
  qsvt_hamiltonian_simulation.cpp
)

target_link_libraries(qsvt_hamiltonian_simulation
  PRIVATE
    cudaq-solvers
    cudaq::cudaq
)
