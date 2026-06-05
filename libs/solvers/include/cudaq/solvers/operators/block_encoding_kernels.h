/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under     *
 * the terms of the Apache License 2.0 which accompanies this distribution.     *
 ******************************************************************************/
#pragma once

#include "cudaq.h"

#include <vector>

namespace cudaq::solvers::block_encoding {

/// \pure_device_kernel
///
/// @brief Apply a PauliLCU PREPARE circuit from flattened kernel data.
__qpu__ void prepare(cudaq::qview<> ancilla,
                     const std::vector<double> &state_prep_angles);

/// \pure_device_kernel
///
/// @brief Apply the adjoint of a PauliLCU PREPARE circuit.
__qpu__ void unprepare(cudaq::qview<> ancilla,
                       const std::vector<double> &state_prep_angles);

/// \pure_device_kernel
///
/// @brief Apply a PauliLCU SELECT circuit from flattened kernel data.
__qpu__ void select(cudaq::qview<> ancilla, cudaq::qview<> system,
                    const std::vector<int> &term_controls,
                    const std::vector<int> &term_ops,
                    const std::vector<int> &term_lengths,
                    const std::vector<int> &term_signs);

/// \pure_device_kernel
///
/// @brief Apply a full PauliLCU block encoding from flattened kernel data.
__qpu__ void apply(cudaq::qview<> ancilla, cudaq::qview<> system,
                   const std::vector<double> &state_prep_angles,
                   const std::vector<int> &term_controls,
                   const std::vector<int> &term_ops,
                   const std::vector<int> &term_lengths,
                   const std::vector<int> &term_signs);

} // namespace cudaq::solvers::block_encoding
