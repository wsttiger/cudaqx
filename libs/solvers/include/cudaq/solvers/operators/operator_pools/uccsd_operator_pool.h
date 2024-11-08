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

/// @brief Test
class uccsd : public operator_pool {

public:
  std::vector<cudaq::spin_op>
  generate(const heterogeneous_map &config) const override;
  CUDAQ_EXTENSION_CREATOR_FUNCTION(operator_pool, uccsd)
};
CUDAQ_REGISTER_TYPE(uccsd)

} // namespace cudaq::solvers
