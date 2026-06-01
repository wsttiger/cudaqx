/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cudaq/solvers/operators/qsvt.h"
#include "cudaq/solvers/operators/qubitization.h"

TEST(QSVTTester, checkSignalPhaseKernelCompile) {
  using namespace cudaq::solvers;

  auto one_signal_test = []() __qpu__ {
    cudaq::qvector<> signal(1);
    apply_qsvt_signal_phase(signal, 0.25);
  };
  EXPECT_NO_THROW(one_signal_test());

  auto three_signal_test = []() __qpu__ {
    cudaq::qvector<> signal(3);
    qsvt_signal_phase{}(signal, -0.5);
  };
  EXPECT_NO_THROW(three_signal_test());
}

TEST(QSVTTester, checkSequenceKernelCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  auto phases = make_qsvt_phase_sequence({0.1, -0.2, 0.3});

  auto sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, phases.data());
  };
  EXPECT_NO_THROW(sequence_test());

  auto sequence_functor_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    qsvt_sequence{}(signal, system, encoding, phases.data());
  };
  EXPECT_NO_THROW(sequence_functor_test());
}

TEST(QSVTTester, checkPhaseSequenceMetadata) {
  using namespace cudaq::solvers;

  qsvt_phase_sequence phases({0.1, -0.2, 0.3, 0.4});

  EXPECT_FALSE(phases.empty());
  EXPECT_EQ(phases.size(), 4);
  EXPECT_EQ(phases.degree(), 3);
  EXPECT_DOUBLE_EQ(phases[0], 0.1);
  EXPECT_DOUBLE_EQ(phases.data()[2], 0.3);
}

TEST(QSVTTester, checkPhaseSequenceFactory) {
  using namespace cudaq::solvers;

  auto phases = make_qsvt_phase_sequence({0.0, 1.0});

  EXPECT_EQ(phases.size(), 2);
  EXPECT_EQ(phases.degree(), 1);
}

TEST(QSVTTester, checkPhaseSequenceValidation) {
  using namespace cudaq::solvers;

  EXPECT_TRUE(is_valid_qsvt_phase_sequence({0.0}));
  EXPECT_TRUE(is_valid_qsvt_phase_sequence({0.0, 1.0, -1.0}));
  EXPECT_FALSE(is_valid_qsvt_phase_sequence({}));
  EXPECT_FALSE(is_valid_qsvt_phase_sequence(
      {0.0, std::numeric_limits<double>::quiet_NaN()}));
  EXPECT_FALSE(is_valid_qsvt_phase_sequence(
      {0.0, std::numeric_limits<double>::infinity()}));

  EXPECT_NO_THROW(validate_qsvt_phase_sequence({0.0, 1.0}));
  EXPECT_THROW(validate_qsvt_phase_sequence({}), std::invalid_argument);
  EXPECT_THROW(
      validate_qsvt_phase_sequence({std::numeric_limits<double>::quiet_NaN()}),
      std::invalid_argument);
}

TEST(QSVTTester, checkPolynomialDegreeConvention) {
  using namespace cudaq::solvers;

  EXPECT_EQ(qsvt_polynomial_degree(1), 0);
  EXPECT_EQ(qsvt_polynomial_degree(4), 3);
  EXPECT_THROW(qsvt_polynomial_degree(0), std::invalid_argument);
}
