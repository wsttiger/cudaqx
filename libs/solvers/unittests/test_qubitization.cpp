/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq.h"
#include "cudaq/solvers/operators/qubitization.h"

TEST(QubitizationTester, checkReflectionKernelsCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);

  auto zero_reflection_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    reflect_about_zero(anc);
  };
  EXPECT_NO_THROW(zero_reflection_test());

  auto prepared_reflection_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    reflect_about_prepare(anc, encoding);
  };
  EXPECT_NO_THROW(prepared_reflection_test());
}

TEST(QubitizationTester, checkObservableBuilders) {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);

  EXPECT_NO_THROW(build_ancilla_zero_projector(encoding.num_ancilla()));
  EXPECT_NO_THROW(
      build_qubitization_reflection_observable(encoding.num_ancilla()));
  EXPECT_NO_THROW(build_lcu_select_observable(encoding));
}
