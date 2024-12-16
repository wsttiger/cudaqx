/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *                                                                             *
 * This file was translated and modified from bravyi_kitaev.py                 *
 * which was adapted from https://doi.org/10.1063/1.4768229                    *
 * Original work Copyright OpenFermion                                         *
 * Licensed under the Apache License, Version 2.0                              *
 *                                                                             *
 * Modifications:                                                              *
 * - Translated from Python to C++                                             *
 *                                                                             *
 * Licensed under the Apache License, Version 2.0 (the "License");             *
 * you may not use this file except in compliance with the License.            *
 * You may obtain a copy of the License at                                     *
 *                                                                             *
 *     http://www.apache.org/licenses/LICENSE-2.0                              *
 *                                                                             *
 * Unless required by applicable law or agreed to in writing, software         *
 * distributed under the License is distributed on an "AS IS" BASIS,           *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    *
 * See the License for the specific language governing permissions and         *
 * limitations under the License.                                              *
 ******************************************************************************/
#pragma once

#include "cudaq/solvers/operators/molecule/fermion_compiler.h"

namespace cudaq::solvers {

/// @brief Helper function used by the Bravyi-Kitaev transformation.
cudaq::spin_op seeley_richard_love(std::size_t i, std::size_t j,
                                   std::complex<double> coef, int n_qubits);

/// @brief Map fermionic operators to spin operators via the
/// Bravyi-Kitaev transformation.
class bravyi_kitaev : public fermion_compiler {
public:
  cudaq::spin_op generate(const double constant, const cudaqx::tensor<> &hpq,
                          const cudaqx::tensor<> &hpqrs,
                          const cudaqx::heterogeneous_map &options) override;

  CUDAQ_EXTENSION_CREATOR_FUNCTION(fermion_compiler, bravyi_kitaev)
};
CUDAQ_REGISTER_TYPE(bravyi_kitaev)
} // namespace cudaq::solvers
