/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <vector>

namespace cudaq::solvers {

/// @brief Host-side QSVT phase sequence.
///
/// QSVT and QSP routines are parameterized by a sequence of phase angles. A
/// degree-d polynomial conventionally uses d + 1 phases. This type keeps that
/// convention explicit and validates the sequence before it is consumed by
/// device-side QSVT kernels.
struct qsvt_phase_sequence {
  std::vector<double> phases;

  qsvt_phase_sequence() = default;
  explicit qsvt_phase_sequence(std::vector<double> input_phases);

  bool empty() const { return phases.empty(); }
  std::size_t size() const { return phases.size(); }
  std::size_t degree() const;
  const std::vector<double> &data() const { return phases; }
  double operator[](std::size_t index) const { return phases[index]; }
};

/// @brief Return true if a QSVT phase sequence is non-empty and finite.
bool is_valid_qsvt_phase_sequence(const std::vector<double> &phases);

/// @brief Validate a QSVT phase sequence.
/// @throws std::invalid_argument if the sequence is empty or contains a
/// non-finite phase.
void validate_qsvt_phase_sequence(const std::vector<double> &phases);

/// @brief Return the polynomial degree represented by num_phases phases.
/// @details A degree-d QSVT polynomial is represented by d + 1 phases.
/// @throws std::invalid_argument if num_phases is zero.
std::size_t qsvt_polynomial_degree(std::size_t num_phases);

/// @brief Construct and validate a QSVT phase sequence.
qsvt_phase_sequence make_qsvt_phase_sequence(std::vector<double> phases);

} // namespace cudaq::solvers
