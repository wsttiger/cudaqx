/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "../operator_pool.h"

namespace cudaq::solvers {

/// @brief Test
class qaoa_pool : public operator_pool {
public:
  std::vector<cudaq::spin_op>
  generate(const heterogeneous_map &config) const override;
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      qaoa_pool, "qaoa", static std::unique_ptr<operator_pool> create() {
        return std::make_unique<qaoa_pool>();
      })
};
CUDAQ_REGISTER_TYPE(qaoa_pool)

} // namespace cudaq::solvers
