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

qsvt_sequence_policy::qsvt_sequence_policy(
    std::vector<int> input_walk_directions)
    : walk_directions(std::move(input_walk_directions)) {
  validate_qsvt_sequence_policy(walk_directions.size(), *this);
}

qsvt_plan::qsvt_plan(qsvt_phase_sequence input_phases)
    : phase_sequence(std::move(input_phases)),
      sequence_policy(make_qsvt_sequence_policy(phase_sequence.degree())) {
  validate_qsvt_phase_sequence(phase_sequence.data());
  validate_qsvt_sequence_policy(phase_sequence.degree(), sequence_policy);
}

qsvt_plan::qsvt_plan(qsvt_phase_sequence input_phases,
                     qsvt_sequence_policy input_policy)
    : phase_sequence(std::move(input_phases)),
      sequence_policy(std::move(input_policy)) {
  validate_qsvt_phase_sequence(phase_sequence.data());
  validate_qsvt_sequence_policy(phase_sequence.degree(), sequence_policy);
}

qsvt_plan::qsvt_plan(std::vector<double> input_phases)
    : qsvt_plan(qsvt_phase_sequence(std::move(input_phases))) {}

qsvt_plan::qsvt_plan(std::vector<double> input_phases,
                     qsvt_sequence_policy input_policy)
    : qsvt_plan(qsvt_phase_sequence(std::move(input_phases)),
                std::move(input_policy)) {}

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

int qsvt_walk_direction_code(qsvt_walk_direction direction) {
  return direction == qsvt_walk_direction::adjoint ? qsvt_adjoint_walk
                                                   : qsvt_forward_walk;
}

bool is_valid_qsvt_sequence_policy(std::size_t degree,
                                   const qsvt_sequence_policy &policy) {
  if (policy.size() != degree)
    return false;

  for (int direction : policy.walk_direction_data()) {
    if (direction != qsvt_forward_walk && direction != qsvt_adjoint_walk)
      return false;
  }

  return true;
}

void validate_qsvt_sequence_policy(std::size_t degree,
                                   const qsvt_sequence_policy &policy) {
  if (policy.size() != degree)
    throw std::invalid_argument(
        "QSVT sequence policy length must match the polynomial degree.");

  for (int direction : policy.walk_direction_data()) {
    if (direction != qsvt_forward_walk && direction != qsvt_adjoint_walk)
      throw std::invalid_argument(
          "QSVT sequence policy contains an unknown walk direction.");
  }
}

qsvt_phase_sequence make_qsvt_phase_sequence(std::vector<double> phases) {
  return qsvt_phase_sequence(std::move(phases));
}

qsvt_sequence_policy make_qsvt_sequence_policy(std::size_t degree,
                                               qsvt_walk_direction direction) {
  std::vector<int> walk_directions(degree, qsvt_walk_direction_code(direction));
  return qsvt_sequence_policy(std::move(walk_directions));
}

qsvt_sequence_policy
make_qsvt_sequence_policy(std::vector<qsvt_walk_direction> directions) {
  std::vector<int> walk_directions;
  walk_directions.reserve(directions.size());
  for (auto direction : directions)
    walk_directions.push_back(qsvt_walk_direction_code(direction));

  return qsvt_sequence_policy(std::move(walk_directions));
}

qsvt_sequence_policy
make_alternating_qsvt_sequence_policy(std::size_t degree,
                                      qsvt_walk_direction first_direction) {
  std::vector<int> walk_directions;
  walk_directions.reserve(degree);

  auto direction = first_direction;
  for (std::size_t i = 0; i < degree; ++i) {
    walk_directions.push_back(qsvt_walk_direction_code(direction));
    direction = direction == qsvt_walk_direction::forward
                    ? qsvt_walk_direction::adjoint
                    : qsvt_walk_direction::forward;
  }

  return qsvt_sequence_policy(std::move(walk_directions));
}

qsvt_plan make_qsvt_plan(std::vector<double> phases) {
  return qsvt_plan(std::move(phases));
}

qsvt_plan make_qsvt_plan(std::vector<double> phases,
                         qsvt_sequence_policy policy) {
  return qsvt_plan(std::move(phases), std::move(policy));
}

} // namespace cudaq::solvers
