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

qsvt_transform_plan::qsvt_transform_plan(
    qsvt_transform_descriptor input_descriptor, qsvt_plan input_plan)
    : transform_descriptor(std::move(input_descriptor)),
      sequence_plan(std::move(input_plan)) {
  validate_qsvt_transform_phase_sequence(transform_descriptor,
                                         sequence_plan.phase_data());
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

bool is_valid_qsvt_transform_descriptor(
    const qsvt_transform_descriptor &descriptor) {
  if (!std::isfinite(descriptor.evolution_time) ||
      !std::isfinite(descriptor.condition_number) ||
      !std::isfinite(descriptor.target_error) ||
      !std::isfinite(descriptor.normalization))
    return false;

  if (descriptor.target_error < 0.0 || descriptor.normalization <= 0.0)
    return false;

  switch (descriptor.kind) {
  case qsvt_transform_kind::linear_solve:
    return descriptor.condition_number >= 1.0;
  case qsvt_transform_kind::real_time_hamiltonian_simulation:
  case qsvt_transform_kind::imaginary_time_hamiltonian_simulation:
    return descriptor.evolution_time >= 0.0;
  case qsvt_transform_kind::custom:
    return true;
  }

  return false;
}

void validate_qsvt_transform_descriptor(
    const qsvt_transform_descriptor &descriptor) {
  if (!is_valid_qsvt_transform_descriptor(descriptor))
    throw std::invalid_argument("Invalid QSVT transform descriptor.");
}

void validate_qsvt_transform_phase_sequence(
    const qsvt_transform_descriptor &descriptor,
    const std::vector<double> &phases) {
  validate_qsvt_transform_descriptor(descriptor);
  validate_qsvt_phase_sequence(phases);

  if (descriptor.degree_hint != 0 &&
      qsvt_polynomial_degree(phases.size()) != descriptor.degree_hint)
    throw std::invalid_argument(
        "QSVT transform phase sequence degree does not match the descriptor "
        "degree hint.");
}

qsvt_transform_plan
make_qsvt_transform_plan(const qsvt_transform_descriptor &descriptor,
                         std::vector<double> phases) {
  validate_qsvt_transform_phase_sequence(descriptor, phases);
  return qsvt_transform_plan(descriptor, qsvt_plan(std::move(phases)));
}

qsvt_transform_plan
make_qsvt_transform_plan(const qsvt_transform_descriptor &descriptor,
                         std::vector<double> phases,
                         qsvt_sequence_policy policy) {
  validate_qsvt_transform_phase_sequence(descriptor, phases);
  return qsvt_transform_plan(descriptor,
                             qsvt_plan(std::move(phases), std::move(policy)));
}

qsvt_transform_descriptor
make_linear_solve_qsvt_transform(double condition_number, double target_error,
                                 std::size_t degree_hint,
                                 double normalization) {
  qsvt_transform_descriptor descriptor;
  descriptor.kind = qsvt_transform_kind::linear_solve;
  descriptor.condition_number = condition_number;
  descriptor.target_error = target_error;
  descriptor.normalization = normalization;
  descriptor.degree_hint = degree_hint;
  validate_qsvt_transform_descriptor(descriptor);
  return descriptor;
}

qsvt_transform_descriptor make_real_time_hamiltonian_simulation_qsvt_transform(
    double evolution_time, double target_error, std::size_t degree_hint,
    double normalization) {
  qsvt_transform_descriptor descriptor;
  descriptor.kind = qsvt_transform_kind::real_time_hamiltonian_simulation;
  descriptor.evolution_time = evolution_time;
  descriptor.target_error = target_error;
  descriptor.normalization = normalization;
  descriptor.degree_hint = degree_hint;
  validate_qsvt_transform_descriptor(descriptor);
  return descriptor;
}

qsvt_transform_descriptor
make_imaginary_time_hamiltonian_simulation_qsvt_transform(
    double evolution_time, double target_error, std::size_t degree_hint,
    double normalization) {
  qsvt_transform_descriptor descriptor;
  descriptor.kind = qsvt_transform_kind::imaginary_time_hamiltonian_simulation;
  descriptor.evolution_time = evolution_time;
  descriptor.target_error = target_error;
  descriptor.normalization = normalization;
  descriptor.degree_hint = degree_hint;
  validate_qsvt_transform_descriptor(descriptor);
  return descriptor;
}

} // namespace cudaq::solvers
