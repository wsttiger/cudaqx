/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/qsvt.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace cudaq::solvers {

qsvt_phase_sequence::qsvt_phase_sequence(std::vector<double> input_phases)
    : phases(std::move(input_phases)) {
  validate_qsvt_phase_sequence(phases);
}

std::size_t qsvt_phase_sequence::degree() const {
  return qsvt_polynomial_degree(phases.size());
}

bool is_valid_qsvt_phase_sequence(const std::vector<double> &phases) {
  if (phases.empty())
    return false;

  for (double phase : phases) {
    if (!std::isfinite(phase))
      return false;
  }

  return true;
}

void validate_qsvt_phase_sequence(const std::vector<double> &phases) {
  if (phases.empty())
    throw std::invalid_argument("QSVT phase sequence must not be empty.");

  for (std::size_t i = 0; i < phases.size(); ++i) {
    if (!std::isfinite(phases[i]))
      throw std::invalid_argument(
          "QSVT phase sequence contains a non-finite phase.");
  }
}

std::size_t qsvt_polynomial_degree(std::size_t num_phases) {
  if (num_phases == 0)
    throw std::invalid_argument(
        "QSVT polynomial degree requires at least one phase.");

  return num_phases - 1;
}

qsvt_phase_sequence make_qsvt_phase_sequence(std::vector<double> phases) {
  return qsvt_phase_sequence(std::move(phases));
}

} // namespace cudaq::solvers
