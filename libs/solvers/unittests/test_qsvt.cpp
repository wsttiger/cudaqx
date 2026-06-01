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
