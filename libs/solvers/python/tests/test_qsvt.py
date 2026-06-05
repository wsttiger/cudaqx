#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for QSVT host-side Python bindings."""

import os
import sys

if 'cudaq_solvers' not in sys.modules:
    build_path = os.path.join(os.path.dirname(__file__),
                              '../../../../build/python')
    if os.path.exists(build_path):
        sys.path.insert(0, build_path)

import pytest
import cudaq
from cudaq import spin
import cudaq_solvers as solvers


def test_pauli_lcu_metadata_binding():
    h = 2.0 + 0.5 * spin.z(0) - 0.25 * spin.x(0)
    encoding = solvers.PauliLCU(h, num_qubits=1)

    assert encoding.constant_term == pytest.approx(2.0)
    assert encoding.term_count == 3
    assert encoding.padded_term_count == 4
    assert hasattr(encoding, 'controlled_select')

    metadata = encoding.metadata()
    assert isinstance(metadata, solvers.PauliLCUMetadata)
    assert metadata.num_system_qubits == 1
    assert metadata.num_ancilla_qubits == 2
    assert metadata.num_terms == 3
    assert metadata.padded_num_terms == 4
    assert metadata.normalization == pytest.approx(2.75)
    assert metadata.constant_term == pytest.approx(2.0)
    assert metadata.coefficient_threshold == pytest.approx(1e-12)


def test_pauli_lcu_block_encoding_device_interop():
    h = 0.6 * spin.x(0) + 0.8 * spin.z(0)
    encoding = solvers.PauliLCU(h, num_qubits=1)

    angles = list(encoding.get_angles())
    term_controls = list(encoding.get_term_controls())
    term_ops = list(encoding.get_term_ops())
    term_lengths = list(encoding.get_term_lengths())
    term_signs = list(encoding.get_term_signs())
    num_ancilla = encoding.num_ancilla
    num_system = encoding.num_system

    @cudaq.kernel
    def kernel():
        ancilla = cudaq.qvector(num_ancilla)
        system = cudaq.qvector(num_system)
        solvers.block_encoding.prepare(ancilla, angles)
        solvers.block_encoding.select(ancilla, system, term_controls, term_ops,
                                      term_lengths, term_signs)
        solvers.block_encoding.unprepare(ancilla, angles)
        solvers.block_encoding.apply(ancilla, system, angles, term_controls,
                                     term_ops, term_lengths, term_signs)

    counts = cudaq.sample(kernel, shots_count=16)
    assert len(counts) > 0


def test_qsvt_phase_sequence_and_walk_policy():
    phases = solvers.make_qsvt_phase_sequence([0.1, -0.2, 0.3])
    assert isinstance(phases, solvers.QSVTPhaseSequence)
    assert len(phases) == 3
    assert phases.size == 3
    assert phases.degree == 2
    assert phases.phases == pytest.approx([0.1, -0.2, 0.3])
    assert phases.data() == pytest.approx([0.1, -0.2, 0.3])
    assert phases[1] == pytest.approx(-0.2)

    assert solvers.qsvt_polynomial_degree(3) == 2
    assert solvers.qsvt_walk_direction_code(
        solvers.QSVTWalkDirection.forward) == 0
    assert solvers.qsvt_walk_direction_code(
        solvers.QSVTWalkDirection.adjoint) == 1

    policy = solvers.make_alternating_qsvt_sequence_policy(
        3, solvers.QSVTWalkDirection.adjoint)
    assert isinstance(policy, solvers.QSVTSequencePolicy)
    assert len(policy) == 3
    assert policy.degree == 3
    assert policy.walk_directions == [1, 0, 1]
    assert policy.walk_direction_data() == [1, 0, 1]
    assert policy[0] == 1
    assert solvers.is_valid_qsvt_sequence_policy(3, policy)

    custom = solvers.make_custom_qsvt_sequence_policy([
        solvers.QSVTWalkDirection.forward,
        solvers.QSVTWalkDirection.adjoint,
    ])
    assert custom.walk_directions == [0, 1]


def test_qsvt_plan_and_transform_descriptor():
    policy = solvers.make_qsvt_sequence_policy(
        2, solvers.QSVTWalkDirection.adjoint)
    plan = solvers.make_qsvt_plan([0.1, -0.2, 0.3], policy)

    assert isinstance(plan, solvers.QSVTPlan)
    assert plan.num_phases == 3
    assert plan.degree == 2
    assert plan.phase_data == pytest.approx([0.1, -0.2, 0.3])
    assert plan.walk_direction_data == [1, 1]
    assert plan.kernel_data()["phases"] == pytest.approx([0.1, -0.2, 0.3])
    assert plan.kernel_data()["walk_directions"] == [1, 1]

    descriptor = solvers.make_real_time_hamiltonian_simulation_qsvt_transform(
        evolution_time=0.75,
        target_error=1e-4,
        degree_hint=2,
        normalization=1.5)
    assert (descriptor.kind ==
            solvers.QSVTTransformKind.real_time_hamiltonian_simulation)
    assert descriptor.phase_convention == solvers.QSVTPhaseConvention.qsvt
    assert descriptor.evolution_time == pytest.approx(0.75)
    assert descriptor.target_error == pytest.approx(1e-4)
    assert descriptor.degree_hint == 2
    assert descriptor.normalization == pytest.approx(1.5)
    assert solvers.is_valid_qsvt_transform_descriptor(descriptor)

    transform_plan = solvers.make_qsvt_transform_plan(
        descriptor, [0.1, -0.2, 0.3], policy)
    assert isinstance(transform_plan, solvers.QSVTTransformPlan)
    assert transform_plan.num_phases == 3
    assert transform_plan.degree == 2
    assert transform_plan.phase_data == pytest.approx([0.1, -0.2, 0.3])
    assert transform_plan.walk_direction_data == [1, 1]
    assert transform_plan.descriptor.evolution_time == pytest.approx(0.75)
    assert transform_plan.plan.degree == 2


def test_qsvt_response_evaluation_and_error_estimation():
    response = solvers.evaluate_qsvt_response([0.0, 0.0], 0.5)
    assert isinstance(response, solvers.QSVTResponse)
    assert response.value.real == pytest.approx(0.5)
    assert response.value.imag == pytest.approx(0.0)
    assert response.magnitude == pytest.approx(0.5)
    assert response.probability == pytest.approx(0.25)

    qsp_response = solvers.evaluate_qsvt_response(
        [0.1, -0.2], 0.5, solvers.QSVTPhaseConvention.qsp)
    assert isinstance(qsp_response.value, complex)

    sample_points = solvers.make_uniform_qsvt_sample_points(-1.0, 1.0, 5)
    assert sample_points == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])
    chebyshev_points = solvers.make_chebyshev_qsvt_sample_points(-1.0, 1.0, 3)
    assert chebyshev_points == pytest.approx([-1.0, 0.0, 1.0])

    error = solvers.estimate_qsvt_response_error(
        [0.0, 0.0], lambda x: complex(x, 0.0), sample_points)
    assert isinstance(error, solvers.QSVTResponseError)
    assert error.max_abs_error == pytest.approx(0.0, abs=1e-12)
    assert error.rms_error == pytest.approx(0.0, abs=1e-12)
    assert error.num_samples == 5


def test_qsvt_validation_errors():
    assert not solvers.is_valid_qsvt_phase_sequence([])
    with pytest.raises(Exception):
        solvers.validate_qsvt_phase_sequence([])
    with pytest.raises(Exception):
        solvers.qsvt_polynomial_degree(0)
    with pytest.raises(Exception):
        solvers.make_qsvt_plan([0.1, 0.2], solvers.QSVTSequencePolicy([0, 1]))
    with pytest.raises(Exception):
        solvers.make_linear_solve_qsvt_transform(
            condition_number=0.5, target_error=1e-3)
