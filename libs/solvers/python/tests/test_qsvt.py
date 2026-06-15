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

_QSVT_HAMILTONIAN_SIMULATION_TERMS = [
    (0.70, "ZIII"),
    (-0.43, "IZII"),
    (0.31, "IIZI"),
    (-0.22, "IIIZ"),
    (0.19, "XXII"),
    (-0.17, "IYYI"),
    (0.13, "IZZX"),
    (0.11, "XYYX"),
]

_QSVT_COS_PHASES = [0.15, -0.42, 0.88, -0.42, 0.15]
_QSVT_SIN_PHASES = [0.23, 0.54, -0.12, -0.54, -0.23]


def _pauli_string_terms_to_indexed_terms(terms):
    return [(coefficient,
             tuple((qubit, op)
                   for qubit, op in enumerate(word)
                   if op != "I"))
            for coefficient, word in terms]


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


def _hamiltonian_from_indexed_terms(terms):
    hamiltonian = None
    for coefficient, paulis in terms:
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


def _signal_projector_phase_matrix(phase,
                                   signal_dimension,
                                   system_dimension,
                                   phase_scale=1.0):
    signal_phase = np.ones(signal_dimension, dtype=np.complex128)
    signal_phase[0] = np.exp(1.0j * phase_scale * phase)
    return np.kron(np.diag(signal_phase),
                   np.eye(system_dimension, dtype=np.complex128))


def _numpy_pauli_lcu_qsvt_good_component(terms,
                                         num_qubits,
                                         initial_ket,
                                         phases,
                                         walk_directions,
                                         normalization,
                                         phase_scale=1.0):
    system_dimension = 1 << num_qubits
    signal_dimension = len(terms)
    assert signal_dimension == 1 << int(np.log2(signal_dimension))
    assert len(phases) == len(walk_directions) + 1

    coefficients = np.array([coefficient for coefficient, _ in terms],
                            dtype=np.float64)
    beta = np.sqrt(np.abs(coefficients) / normalization).astype(np.complex128)
    signal_reflection = (np.eye(signal_dimension, dtype=np.complex128) -
                         2.0 * np.outer(beta, beta.conj()))
    reflection = np.kron(signal_reflection,
                         np.eye(system_dimension, dtype=np.complex128))

    select = np.zeros((signal_dimension * system_dimension,
                       signal_dimension * system_dimension),
                      dtype=np.complex128)
    for term_index, (coefficient, paulis) in enumerate(terms):
        block_start = term_index * system_dimension
        block_end = block_start + system_dimension
        sign = 1.0 if coefficient >= 0.0 else -1.0
        pauli_matrix = _pauli_sum_matrix([(1.0, paulis)], num_qubits)
        select[block_start:block_end,
               block_start:block_end] = sign * pauli_matrix

    forward_walk = reflection @ select
    adjoint_walk = select @ reflection

    state = np.kron(beta, initial_ket)
    state = _signal_projector_phase_matrix(
        phases[0], signal_dimension, system_dimension, phase_scale) @ state

    for direction, phase in zip(walk_directions, phases[1:]):
        state = (adjoint_walk if direction == 1 else forward_walk) @ state
        state = _signal_projector_phase_matrix(
            phase, signal_dimension, system_dimension, phase_scale) @ state

    return beta.conj() @ state.reshape(signal_dimension, system_dimension)


def _run_qsvt_good_component(initial_state, num_system, num_ancilla, phases,
                             walk_directions, kernel_data):
    angles, term_controls, term_ops, term_lengths, term_signs = kernel_data

    @cudaq.kernel
    def qsvt_kernel(state: cudaq.State):
        system = cudaq.qvector(state)
        signal = cudaq.qvector(num_ancilla)
        solvers.qsvt.apply_phase_sequence(signal, system, phases,
                                          walk_directions, angles,
                                          term_controls, term_ops, term_lengths,
                                          term_signs)

    full_state = cudaq.get_state(qsvt_kernel, initial_state)
    return _zero_ancilla_component(full_state, num_system, num_ancilla)


def _run_explicit_qsvt_good_component(initial_state, num_system, num_ancilla,
                                      phases, walk_directions, kernel_data):
    assert all(direction == 0 for direction in walk_directions)
    angles, term_controls, term_ops, term_lengths, term_signs = kernel_data

    @cudaq.kernel
    def qsvt_kernel(state: cudaq.State):
        system = cudaq.qvector(state)
        signal = cudaq.qvector(num_ancilla)
        solvers.qsvt.apply_signal_phase(signal, phases[0])
        for i in range(1, len(phases)):
            solvers.block_encoding.apply(signal, system, angles, term_controls,
                                         term_ops, term_lengths, term_signs)
            solvers.qubitization.reflect_about_zero(signal)
            solvers.qsvt.apply_signal_phase(signal, phases[i])

    full_state = cudaq.get_state(qsvt_kernel, initial_state)
    return _zero_ancilla_component(full_state, num_system, num_ancilla)


def _qsp_to_projector_phases(phases):
    return [2.0 * float(phase) for phase in phases]


def _qsppack_hamiltonian_simulation_phases(tau, degree=16):
    qsppack = pytest.importorskip("qsppack")
    scipy_special = pytest.importorskip("scipy.special")
    jv = scipy_special.jv

    cos_coefficients = np.array(
        [0.5 * jv(0, tau)] +
        [((-1)**k) * jv(2 * k, tau) for k in range(1, degree // 2 + 1)],
        dtype=np.float64)
    sin_coefficients = np.array(
        [((-1)**k) * jv(2 * k + 1, tau) for k in range(degree // 2)],
        dtype=np.float64)
    common_options = {
        "criteria": 1e-12,
        "method": "Newton",
        "typePhi": "full",
        "useReal": True,
    }

    cos_phases, cos_info = qsppack.solve(cos_coefficients, 0, {
        **common_options, "targetPre": True
    })
    sin_phases, sin_info = qsppack.solve(sin_coefficients, 1, {
        **common_options, "targetPre": False
    })
    return ([float(phase) for phase in cos_phases],
            [float(phase) for phase in sin_phases], cos_info, sin_info)


def _exact_time_evolved_state(hamiltonian_matrix, initial_ket, evolution_time):
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
    amplitudes = eigenvectors.conj().T @ initial_ket
    phases = np.exp(-1.0j * evolution_time * eigenvalues)
    return eigenvectors @ (phases * amplitudes), eigenvalues, eigenvectors


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


def test_qsvt_hamiltonian_simulation_sequence_matches_explicit_loop(
        qpp_cpu_target):
    num_system = 4
    terms = _pauli_string_terms_to_indexed_terms(
        _QSVT_HAMILTONIAN_SIMULATION_TERMS)
    hamiltonian = _hamiltonian_from_indexed_terms(terms)
    initial_ket = np.zeros(1 << num_system, dtype=np.complex128)
    initial_ket[0] = 1.0
    initial_state = cudaq.State.from_data(initial_ket)

    encoding = solvers.PauliLCU(hamiltonian, num_qubits=num_system)
    assert encoding.num_ancilla == 3
    assert encoding.term_count == len(terms)

    kernel_data = _kernel_data(encoding)
    cos_plan = solvers.make_qsvt_plan(_QSVT_COS_PHASES)
    sin_plan = solvers.make_qsvt_plan(_QSVT_SIN_PHASES)
    cos_phases = list(cos_plan.phase_data)
    sin_phases = list(sin_plan.phase_data)
    cos_walk_directions = list(cos_plan.walk_direction_data)
    sin_walk_directions = list(sin_plan.walk_direction_data)

    quantum_cos_state = _run_qsvt_good_component(initial_state, num_system,
                                                 encoding.num_ancilla,
                                                 cos_phases,
                                                 cos_walk_directions,
                                                 kernel_data)
    quantum_sin_state = _run_qsvt_good_component(initial_state, num_system,
                                                 encoding.num_ancilla,
                                                 sin_phases,
                                                 sin_walk_directions,
                                                 kernel_data)
    quantum_time_evolved_state = quantum_cos_state - 1.0j * quantum_sin_state

    explicit_cos_state = _run_explicit_qsvt_good_component(
        initial_state, num_system, encoding.num_ancilla, cos_phases,
        cos_walk_directions, kernel_data)
    explicit_sin_state = _run_explicit_qsvt_good_component(
        initial_state, num_system, encoding.num_ancilla, sin_phases,
        sin_walk_directions, kernel_data)
    explicit_time_evolved_state = explicit_cos_state - 1.0j * explicit_sin_state

    # These fixed phase sequences are deterministic stand-ins for externally
    # generated Hamiltonian-simulation phases. This test validates that the
    # public QSVT sequence helper matches the equivalent explicit composition of
    # signal phases, PauliLCU block encodings, and zero-signal reflections.
    assert np.linalg.norm(quantum_cos_state - explicit_cos_state) < 1e-10
    assert np.linalg.norm(quantum_sin_state - explicit_sin_state) < 1e-10
    assert (np.linalg.norm(quantum_time_evolved_state -
                           explicit_time_evolved_state) < 1e-10)
    assert np.linalg.norm(quantum_time_evolved_state) > 1e-8


def test_qsppack_generated_phases_validate_device_sequence_and_exact_response(
        qpp_cpu_target):
    num_system = 4
    evolution_time = 0.8
    phase_generation_degree = 16
    terms = _pauli_string_terms_to_indexed_terms(
        _QSVT_HAMILTONIAN_SIMULATION_TERMS)
    hamiltonian = _hamiltonian_from_indexed_terms(terms)
    hamiltonian_matrix = _pauli_sum_matrix(terms, num_system)
    rng = np.random.default_rng(13)
    initial_ket = rng.normal(size=1 << num_system)
    initial_ket = (initial_ket / np.linalg.norm(initial_ket)).astype(
        np.complex128)

    encoding = solvers.PauliLCU(hamiltonian, num_qubits=num_system)
    alpha = float(encoding.normalization)
    tau = alpha * evolution_time
    cos_phases, sin_phases, cos_info, sin_info = (
        _qsppack_hamiltonian_simulation_phases(tau, phase_generation_degree))

    assert len(cos_phases) == phase_generation_degree + 1
    assert len(sin_phases) == phase_generation_degree
    assert cos_info["value"] < 1e-12
    assert sin_info["value"] < 1e-12

    kernel_data = _kernel_data(encoding)
    initial_state = cudaq.State.from_data(initial_ket)
    cos_walk_directions = [0] * (len(cos_phases) - 1)
    sin_walk_directions = [0] * (len(sin_phases) - 1)
    quantum_cos_state = _run_qsvt_good_component(
        initial_state, num_system, encoding.num_ancilla,
        _qsp_to_projector_phases(cos_phases), cos_walk_directions, kernel_data)
    quantum_sin_state = _run_qsvt_good_component(
        initial_state, num_system, encoding.num_ancilla,
        _qsp_to_projector_phases(sin_phases), sin_walk_directions, kernel_data)

    quantum_cos_state *= np.exp(-1.0j * np.sum(cos_phases))
    quantum_sin_state *= np.exp(-1.0j * np.sum(sin_phases))

    exact_state, eigenvalues, eigenvectors = _exact_time_evolved_state(
        hamiltonian_matrix, initial_ket, evolution_time)
    cos_poly = solvers.qsvt.phases_to_poly(cos_phases,
                                           solvers.QSVTPhaseConvention.qsp)
    sin_poly = solvers.qsvt.phases_to_poly(sin_phases,
                                           solvers.QSVTPhaseConvention.qsp)
    sample_errors = []
    for scaled_eigenvalue in eigenvalues / alpha:
        walk_eigenvalue = -scaled_eigenvalue
        cos_response = cos_poly(float(walk_eigenvalue))
        sin_response = sin_poly(float(walk_eigenvalue))
        target = np.exp(-1.0j * tau * scaled_eigenvalue)
        response = 2.0 * (cos_response.real + 1.0j * sin_response.imag)
        sample_errors.append(abs(response - target))

    # With W = R_zero U, the QSP response is evaluated at -H / alpha. QSPPACK's
    # cosine target is in the real part and its sine target is in the imaginary
    # part. For this real Hamiltonian and real input state, those components can
    # be recovered from statevector simulation and combined into exp(-i H t)|psi>.
    quantum_time_evolved_state = 2.0 * (quantum_cos_state.real +
                                        1.0j * quantum_sin_state.imag)
    qsp_l2_error = np.linalg.norm(quantum_time_evolved_state - exact_state)
    qsp_max_error = np.max(np.abs(quantum_time_evolved_state - exact_state))
    qsp_fidelity = abs(np.vdot(exact_state, quantum_time_evolved_state))**2

    assert max(sample_errors) < 1e-10
    assert qsp_l2_error < 1e-10
    assert qsp_max_error < 1e-10
    assert qsp_fidelity == pytest.approx(1.0, abs=1e-10)


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
    poly = solvers.qsvt.phases_to_poly([0.0, 0.0])
    response = poly(0.5)
    assert isinstance(response, complex)
    assert response.real == pytest.approx(0.5)
    assert response.imag == pytest.approx(0.0)

    qsp_poly = solvers.qsvt.phases_to_poly([0.1, -0.2],
                                           solvers.QSVTPhaseConvention.qsp)
    qsp_response = qsp_poly(0.5)
    assert isinstance(qsp_response, complex)

    sample_points = solvers.make_uniform_qsvt_sample_points(-1.0, 1.0, 5)
    assert sample_points == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])
    chebyshev_points = solvers.make_chebyshev_qsvt_sample_points(-1.0, 1.0, 3)
    assert chebyshev_points == pytest.approx([-1.0, 0.0, 1.0])

    error = solvers.qsvt.estimate_poly_error(poly,
                                             lambda x: complex(x, 0.0),
                                             domain=(-1.0, 1.0),
                                             num_points=5)
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
