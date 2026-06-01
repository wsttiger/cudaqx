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
  qsvt_plan plan(phases);
  auto phase_data = plan.phase_data();

  auto sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, phase_data);
  };
  EXPECT_NO_THROW(sequence_test());

  auto sequence_functor_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    qsvt_sequence{}(signal, system, encoding, phase_data);
  };
  EXPECT_NO_THROW(sequence_functor_test());
}

TEST(QSVTTester, checkSequenceWalkDirectionKernelCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  auto plan = make_qsvt_plan({0.1, -0.2, 0.3});
  auto phase_data = plan.phase_data();

  auto adjoint_sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, phase_data,
                        qsvt_walk_direction::adjoint);
  };
  EXPECT_NO_THROW(adjoint_sequence_test());

  auto adjoint_sequence_functor_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    qsvt_sequence{}(signal, system, encoding, phase_data,
                    qsvt_walk_direction::adjoint);
  };
  EXPECT_NO_THROW(adjoint_sequence_functor_test());
}

TEST(QSVTTester, checkQubitizationAndQSVTExecution) {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  cudaq::spin_op h = x(0);
  pauli_lcu encoding(h, 1);

  auto walk_once = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qubitization_walk(signal, system, encoding);
  };
  auto walk_counts = cudaq::sample(100, walk_once);
  EXPECT_FLOAT_EQ(1.0, walk_counts.probability("1"));

  auto one_walk_plan = make_qsvt_plan({0.0, 0.0});
  auto one_walk_kernel_data = one_walk_plan.kernel_data();
  auto one_walk_phases = one_walk_kernel_data.phases;
  auto one_walk_directions = one_walk_kernel_data.walk_directions;

  auto qsvt_one_walk = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, one_walk_phases,
                        one_walk_directions);
  };
  auto one_walk_counts = cudaq::sample(100, qsvt_one_walk);
  EXPECT_FLOAT_EQ(1.0, one_walk_counts.probability("1"));

  auto two_walk_plan =
      make_qsvt_plan({0.0, 0.0, 0.0}, make_alternating_qsvt_sequence_policy(2));
  auto two_walk_kernel_data = two_walk_plan.kernel_data();
  auto two_walk_phases = two_walk_kernel_data.phases;
  auto two_walk_directions = two_walk_kernel_data.walk_directions;

  auto qsvt_two_walks = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, two_walk_phases,
                        two_walk_directions);
  };
  auto two_walk_counts = cudaq::sample(100, qsvt_two_walks);
  EXPECT_FLOAT_EQ(1.0, two_walk_counts.probability("0"));
}

TEST(QSVTTester, checkSequencePolicyKernelCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  auto policy =
      make_alternating_qsvt_sequence_policy(3, qsvt_walk_direction::adjoint);
  auto plan = make_qsvt_plan({0.1, -0.2, 0.3, -0.4}, policy);
  auto phase_data = plan.phase_data();
  auto walk_direction_data = plan.walk_direction_data();

  auto sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, phase_data,
                        walk_direction_data);
  };
  EXPECT_NO_THROW(sequence_test());

  auto sequence_functor_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    qsvt_sequence{}(signal, system, encoding, phase_data, walk_direction_data);
  };
  EXPECT_NO_THROW(sequence_functor_test());
}

TEST(QSVTTester, checkPlanMetadata) {
  using namespace cudaq::solvers;

  qsvt_plan plan(std::vector<double>{0.1, -0.2, 0.3});

  EXPECT_EQ(plan.num_phases(), 3);
  EXPECT_EQ(plan.degree(), 2);
  EXPECT_EQ(plan.phases().size(), 3);
  EXPECT_DOUBLE_EQ(plan.phase_data()[1], -0.2);
  EXPECT_EQ(plan.policy().size(), 2);
  EXPECT_EQ(plan.walk_direction_data()[0], qsvt_forward_walk);
  EXPECT_EQ(plan.walk_direction_data()[1], qsvt_forward_walk);

  auto kernel_data = plan.kernel_data();
  EXPECT_EQ(kernel_data.phases.size(), 3);
  EXPECT_EQ(kernel_data.walk_directions.size(), 2);
  EXPECT_DOUBLE_EQ(kernel_data.phases[1], -0.2);
  EXPECT_EQ(kernel_data.walk_directions[0], qsvt_forward_walk);
}

TEST(QSVTTester, checkPlanPolicyMetadata) {
  using namespace cudaq::solvers;

  auto policy = make_alternating_qsvt_sequence_policy(3);
  auto plan = make_qsvt_plan({0.1, -0.2, 0.3, -0.4}, policy);

  EXPECT_EQ(plan.degree(), 3);
  EXPECT_EQ(plan.policy().degree(), 3);
  EXPECT_EQ(plan.walk_direction_data()[0], qsvt_forward_walk);
  EXPECT_EQ(plan.walk_direction_data()[1], qsvt_adjoint_walk);
  EXPECT_EQ(plan.walk_direction_data()[2], qsvt_forward_walk);
}

TEST(QSVTTester, checkPlanFactoryAndValidation) {
  using namespace cudaq::solvers;

  auto plan = make_qsvt_plan({0.25});
  EXPECT_EQ(plan.num_phases(), 1);
  EXPECT_EQ(plan.degree(), 0);

  EXPECT_THROW(qsvt_plan(qsvt_phase_sequence{}), std::invalid_argument);
  EXPECT_THROW(make_qsvt_plan({0.1, 0.2}, make_qsvt_sequence_policy(2)),
               std::invalid_argument);
  EXPECT_NO_THROW(make_qsvt_plan(
      {0.1, 0.2, 0.3},
      make_qsvt_sequence_policy(2, qsvt_walk_direction::adjoint)));
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

TEST(QSVTTester, checkSequencePolicyFactoryAndValidation) {
  using namespace cudaq::solvers;

  auto adjoint_policy =
      make_qsvt_sequence_policy(3, qsvt_walk_direction::adjoint);
  EXPECT_EQ(adjoint_policy.size(), 3);
  EXPECT_EQ(adjoint_policy.walk_direction_data()[0], qsvt_adjoint_walk);
  EXPECT_EQ(adjoint_policy.walk_direction_data()[2], qsvt_adjoint_walk);

  auto custom_policy = make_qsvt_sequence_policy(
      {qsvt_walk_direction::forward, qsvt_walk_direction::adjoint});
  EXPECT_EQ(custom_policy.size(), 2);
  EXPECT_EQ(custom_policy.walk_direction_data()[0], qsvt_forward_walk);
  EXPECT_EQ(custom_policy.walk_direction_data()[1], qsvt_adjoint_walk);

  auto alternating_policy =
      make_alternating_qsvt_sequence_policy(4, qsvt_walk_direction::adjoint);
  EXPECT_EQ(alternating_policy.walk_direction_data()[0], qsvt_adjoint_walk);
  EXPECT_EQ(alternating_policy.walk_direction_data()[1], qsvt_forward_walk);
  EXPECT_EQ(alternating_policy.walk_direction_data()[2], qsvt_adjoint_walk);
  EXPECT_EQ(alternating_policy.walk_direction_data()[3], qsvt_forward_walk);

  EXPECT_TRUE(is_valid_qsvt_sequence_policy(2, custom_policy));
  EXPECT_FALSE(is_valid_qsvt_sequence_policy(3, custom_policy));
  EXPECT_THROW(validate_qsvt_sequence_policy(
                   1, qsvt_sequence_policy(std::vector<int>{2})),
               std::invalid_argument);
}

TEST(QSVTTester, checkTransformDescriptorFactories) {
  using namespace cudaq::solvers;

  auto linear_solve = make_linear_solve_qsvt_transform(12.0, 1e-3, 27, 2.0);
  EXPECT_EQ(linear_solve.kind, qsvt_transform_kind::linear_solve);
  EXPECT_EQ(linear_solve.phase_convention, qsvt_phase_convention::qsvt);
  EXPECT_DOUBLE_EQ(linear_solve.condition_number, 12.0);
  EXPECT_DOUBLE_EQ(linear_solve.target_error, 1e-3);
  EXPECT_DOUBLE_EQ(linear_solve.normalization, 2.0);
  EXPECT_EQ(linear_solve.degree_hint, 27);
  EXPECT_TRUE(is_valid_qsvt_transform_descriptor(linear_solve));

  auto real_time =
      make_real_time_hamiltonian_simulation_qsvt_transform(0.75, 1e-4, 32, 1.5);
  EXPECT_EQ(real_time.kind,
            qsvt_transform_kind::real_time_hamiltonian_simulation);
  EXPECT_DOUBLE_EQ(real_time.evolution_time, 0.75);

  auto imaginary_time =
      make_imaginary_time_hamiltonian_simulation_qsvt_transform(1.25, 1e-5, 48);
  EXPECT_EQ(imaginary_time.kind,
            qsvt_transform_kind::imaginary_time_hamiltonian_simulation);
  EXPECT_DOUBLE_EQ(imaginary_time.evolution_time, 1.25);

  qsvt_transform_descriptor invalid_normalization;
  invalid_normalization.normalization = 0.0;
  EXPECT_FALSE(is_valid_qsvt_transform_descriptor(invalid_normalization));

  EXPECT_THROW(make_linear_solve_qsvt_transform(0.5, 1e-3),
               std::invalid_argument);
  EXPECT_THROW(make_real_time_hamiltonian_simulation_qsvt_transform(-0.1, 1e-3),
               std::invalid_argument);
  EXPECT_THROW(
      make_imaginary_time_hamiltonian_simulation_qsvt_transform(1.0, -1e-3),
      std::invalid_argument);
}

TEST(QSVTTester, checkPolynomialDegreeConvention) {
  using namespace cudaq::solvers;

  EXPECT_EQ(qsvt_polynomial_degree(1), 0);
  EXPECT_EQ(qsvt_polynomial_degree(4), 3);
  EXPECT_THROW(qsvt_polynomial_degree(0), std::invalid_argument);
}
