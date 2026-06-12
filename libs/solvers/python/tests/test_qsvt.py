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
import numpy as np
import cudaq
from cudaq import spin
import cudaq_solvers as solvers

_FOUR_QUBIT_TERMS = [
    (0.31, ((0, "X"),)),
    (-0.22, ((1, "Z"),)),
    (0.17, ((2, "Y"),)),
    (0.13, ((3, "X"),)),
    (0.11, ((0, "Z"), (2, "Z"))),
    (-0.19, ((1, "X"), (3, "Y"))),
    (0.07, ((0, "Y"), (2, "X"), (3, "Z"))),
    (0.05, ((0, "Z"), (1, "Z"), (2, "Z"), (3, "Z"))),
]


def _spin_term(paulis):
    term = None
    for qubit, op_name in paulis:
        if op_name == "X":
            op = spin.x(qubit)
        elif op_name == "Y":
            op = spin.y(qubit)
        elif op_name == "Z":
            op = spin.z(qubit)
        else:
            raise ValueError(f"Unsupported Pauli operator: {op_name}")
        term = op if term is None else term * op
    return term


def _four_qubit_hamiltonian():
    hamiltonian = None
    for coefficient, paulis in _FOUR_QUBIT_TERMS:
        term = coefficient * _spin_term(paulis)
        hamiltonian = term if hamiltonian is None else hamiltonian + term
    return hamiltonian


def _pauli_sum_matrix(terms, num_qubits):
    dimension = 1 << num_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)

    for coefficient, paulis in terms:
        for column in range(dimension):
            row = column
            phase = 1.0 + 0.0j
            for qubit, op_name in paulis:
                bit = (column >> qubit) & 1
                if op_name == "X":
                    row ^= (1 << qubit)
                elif op_name == "Y":
                    row ^= (1 << qubit)
                    phase *= 1.0j if bit == 0 else -1.0j
                elif op_name == "Z":
                    phase *= 1.0 if bit == 0 else -1.0
                else:
                    raise ValueError(f"Unsupported Pauli operator: {op_name}")
            matrix[row, column] += coefficient * phase

    return matrix


def _random_normalized_ket(num_qubits, seed):
    rng = np.random.default_rng(seed)
    dimension = 1 << num_qubits
    ket = rng.normal(size=dimension) + 1.0j * rng.normal(size=dimension)
    ket /= np.linalg.norm(ket)
    return ket.astype(np.complex128)


def _kernel_data(encoding):
    return (
        [float(value) for value in encoding.get_angles()],
        [int(value) for value in encoding.get_term_controls()],
        [int(value) for value in encoding.get_term_ops()],
        [int(value) for value in encoding.get_term_lengths()],
        [int(value) for value in encoding.get_term_signs()],
    )


def _zero_ancilla_component(full_state, num_system, num_ancilla):
    state_vector = np.asarray(full_state, dtype=np.complex128)
    system_dimension = 1 << num_system
    expected_dimension = 1 << (num_system + num_ancilla)
    assert state_vector.shape == (expected_dimension,)
    # The tests allocate the system state first and the LCU ancillas second.
    # CUDA-Q stores q[0] as the least-significant statevector bit, so the
    # all-zero ancilla subspace is the first contiguous system block.
    return state_vector[:system_dimension].copy()


@pytest.fixture
def qpp_cpu_target():
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def _assert_good_component_matches(good_component, expected_component):
    observed_probability = np.vdot(good_component, good_component).real
    expected_probability = np.vdot(expected_component, expected_component).real

    assert observed_probability > 1e-14
    assert observed_probability == pytest.approx(expected_probability,
                                                 abs=1e-10)
    assert np.allclose(good_component, expected_component, atol=1e-10)


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
        solvers.qubitization.reflect_about_zero(ancilla)
        solvers.qubitization.reflect_about_prepare(ancilla, angles)
        solvers.qubitization.apply_walk(ancilla, system, angles, term_controls,
                                        term_ops, term_lengths, term_signs)
        solvers.qubitization.apply_adjoint_walk_power(ancilla, system, angles,
                                                      term_controls, term_ops,
                                                      term_lengths, term_signs,
                                                      1)

    counts = cudaq.sample(kernel, shots_count=16)
    assert len(counts) > 0


def test_pauli_lcu_block_encoding_matches_numpy_good_subspace(qpp_cpu_target):
    num_system = 4
    hamiltonian = _four_qubit_hamiltonian()
    hamiltonian_matrix = _pauli_sum_matrix(_FOUR_QUBIT_TERMS, num_system)
    initial_ket = _random_normalized_ket(num_system, seed=2026)
    initial_state = cudaq.State.from_data(initial_ket)

    encoding = solvers.PauliLCU(hamiltonian, num_qubits=num_system)
    assert encoding.num_ancilla == 3
    assert encoding.term_count == len(_FOUR_QUBIT_TERMS)

    angles, term_controls, term_ops, term_lengths, term_signs = _kernel_data(
        encoding)
    num_ancilla = encoding.num_ancilla

    @cudaq.kernel
    def block_encode(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(num_ancilla)
        solvers.block_encoding.apply(ancilla, system, angles, term_controls,
                                     term_ops, term_lengths, term_signs)

    full_state = cudaq.get_state(block_encode, initial_state)
    good_component = _zero_ancilla_component(full_state, num_system,
                                             num_ancilla)
    expected_component = (
        hamiltonian_matrix @ initial_ket) / encoding.normalization

    _assert_good_component_matches(good_component, expected_component)


def test_pauli_lcu_qubitization_walk_matches_numpy_good_subspace(
        qpp_cpu_target):
    num_system = 4
    hamiltonian = _four_qubit_hamiltonian()
    hamiltonian_matrix = _pauli_sum_matrix(_FOUR_QUBIT_TERMS, num_system)
    initial_ket = _random_normalized_ket(num_system, seed=2027)
    initial_state = cudaq.State.from_data(initial_ket)

    encoding = solvers.PauliLCU(hamiltonian, num_qubits=num_system)
    assert encoding.num_ancilla == 3

    angles, term_controls, term_ops, term_lengths, term_signs = _kernel_data(
        encoding)
    num_ancilla = encoding.num_ancilla

    @cudaq.kernel
    def walk_once(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(num_ancilla)
        solvers.block_encoding.prepare(ancilla, angles)
        solvers.qubitization.apply_walk(ancilla, system, angles, term_controls,
                                        term_ops, term_lengths, term_signs)

        # Move the PREPARE signal state back to |0...0> so the statevector
        # test can postselect the good subspace and ignore the orthogonal junk
        # space.
        solvers.block_encoding.unprepare(ancilla, angles)

    full_state = cudaq.get_state(walk_once, initial_state)
    good_component = _zero_ancilla_component(full_state, num_system,
                                             num_ancilla)

    # The current zero-state reflection applies a -1 phase to |0...0>, so the
    # prepared-signal block of one walk is -H / alpha for this convention.
    expected_component = -(
        hamiltonian_matrix @ initial_ket) / encoding.normalization

    _assert_good_component_matches(good_component, expected_component)


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

    transform_plan = solvers.make_qsvt_transform_plan(descriptor,
                                                      [0.1, -0.2, 0.3], policy)
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

    error = solvers.estimate_qsvt_response_error([0.0, 0.0],
                                                 lambda x: complex(x, 0.0),
                                                 sample_points)
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
        solvers.make_linear_solve_qsvt_transform(condition_number=0.5,
                                                 target_error=1e-3)
