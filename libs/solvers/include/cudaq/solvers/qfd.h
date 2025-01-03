/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cuda-qx/core/tensor.h"

using namespace cudaqx;

namespace cudaq::solvers::qfd {

cudaq::spin_op identity(std::size_t num_qubits);

cudaqx::tensor<> create_krylov_subspace_matrix(const cudaq::spin_op& op, 
                                               const cudaq::spin_op& h_op, 
                                               const std::size_t num_qubits,
                                               const std::size_t krylov_dim,
                                               const double dt,
                                               const std::vector<double>& vec);

} // namespace cudaq::solvers

