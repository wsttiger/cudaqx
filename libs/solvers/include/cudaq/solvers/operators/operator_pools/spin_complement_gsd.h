/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "../operator_pool.h"

namespace cudaq::solvers {

// adapted from
// https://github.com/mayhallgroup/adapt-vqe/blob/master/src/operator_pools.py

/// @brief Test
class spin_complement_gsd : public operator_pool {

public:
  std::vector<cudaq::spin_op>
  generate(const heterogeneous_map &config) const override;
  CUDAQ_EXTENSION_CREATOR_FUNCTION(operator_pool, spin_complement_gsd)
};
CUDAQ_REGISTER_TYPE(spin_complement_gsd)
} // namespace cudaq::solvers