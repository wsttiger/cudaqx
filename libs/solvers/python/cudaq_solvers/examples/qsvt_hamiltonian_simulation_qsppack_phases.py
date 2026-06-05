#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
QSPPACK phase validation for 4-qubit Hamiltonian simulation.

This example uses the same Hamiltonian, evolution time, and seed-13 initial ket
as qsvt_hamiltonian_simulation_reference.py. It hard-codes QSPPACK-generated
QSP phases for the parity components of

    exp(-i H t) = cos(H t) - i sin(H t).

The phases approximate the scaled scalar functions

    0.5 * cos(alpha * t * x)
    0.5 * sin(alpha * t * x)

where x is an eigenvalue of H / alpha and alpha is the Pauli-LCU 1-norm. The
factor of 0.5 keeps each polynomial safely inside the QSP unit disk; the final
complex response multiplies the combined components by 2.

QSPPACK was used only to generate the constants below. It is not imported at
runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy import linalg as la
from cudaq import spin
import cudaq_solvers as solvers


EVOLUTION_TIME = 0.8
INITIAL_STATE_SEED = 13
PHASE_GENERATION_DEGREE = 16

# Generated with qsppack==0.3.12 using Newton solve, criteria=1e-12,
# targetPre=True, typePhi="full", and Chebyshev coefficients for
# 0.5 * cos(1.808 * x). These phases use the QSP Z-rotation convention.
COS_QSP_PHASES = [
    0.78539816339744339,
    1.3785910485838523e-12,
    -3.0449692126760338e-10,
    4.8625486806333666e-08,
    -5.2674428156834571e-06,
    0.00035167761396427813,
    -0.012313332829073025,
    0.15998916183828821,
    -0.1782797982872823,
    0.15998916183828821,
    -0.012313332829073025,
    0.00035167761396427813,
    -5.2674428156834571e-06,
    4.8625486806333666e-08,
    -3.0449692126760338e-10,
    1.3785910485838523e-12,
    0.78539816339744339,
]

# Generated with qsppack==0.3.12 using Newton solve, criteria=1e-12,
# targetPre=False, typePhi="full", and Chebyshev coefficients for
# 0.5 * sin(1.808 * x). The target is carried by the imaginary part of the
# QSP response in our evaluator.
SIN_QSP_PHASES = [
    -8.8938765361745635e-14,
    2.2494080921322152e-11,
    -4.2518366000152621e-09,
    5.6441496573790198e-07,
    -4.8694958643126732e-05,
    0.0024156290072621586,
    -0.054918458956657874,
    0.30630415304777148,
    0.30630415304777148,
    -0.054918458956657874,
    0.0024156290072621586,
    -4.8694958643126732e-05,
    5.6441496573790198e-07,
    -4.2518366000152621e-09,
    2.2494080921322152e-11,
    -8.8938765361745635e-14,
]


PAULI = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


@dataclass(frozen=True)
class PauliTerm:
    coefficient: float
    word: str


REFERENCE_TERMS = [
    PauliTerm(0.70, "ZIII"),
    PauliTerm(-0.43, "IZII"),
    PauliTerm(0.31, "IIZI"),
    PauliTerm(-0.22, "IIIZ"),
    PauliTerm(0.19, "XXII"),
    PauliTerm(-0.17, "IYYI"),
    PauliTerm(0.13, "IZZX"),
    PauliTerm(0.11, "XYYX"),
]


def kron_all(operators: list[np.ndarray]) -> np.ndarray:
    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def pauli_word_matrix(word: str) -> np.ndarray:
    return kron_all([PAULI[label] for label in word])


def hamiltonian_matrix(terms: list[PauliTerm]) -> np.ndarray:
    num_qubits = len(terms[0].word)
    dimension = 2**num_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for term in terms:
        if len(term.word) != num_qubits:
            raise ValueError("All Pauli words must have the same length.")
        matrix += term.coefficient * pauli_word_matrix(term.word)
    return matrix


def spin_word(word: str):
    op = None
    for qubit, label in enumerate(word):
        if label == "I":
            continue
        factor = {
            "X": spin.x,
            "Y": spin.y,
            "Z": spin.z,
        }[label](qubit)
        op = factor if op is None else op * factor
    return 1.0 if op is None else op


def spin_hamiltonian(terms: list[PauliTerm]):
    hamiltonian = 0.0
    for term in terms:
        hamiltonian = hamiltonian + term.coefficient * spin_word(term.word)
    return hamiltonian


def random_normalized_ket(num_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dimension = 2**num_qubits
    state = rng.normal(size=dimension) + 1.0j * rng.normal(size=dimension)
    return state / np.linalg.norm(state)


def qsp_response(phases: list[float], x: float) -> complex:
    return solvers.evaluate_qsvt_response(
        phases, x, solvers.QSVTPhaseConvention.qsp
    ).value


def component_errors(tau: float, sample_points: np.ndarray) -> tuple[float, float]:
    cos_errors = []
    sin_errors = []
    for x in sample_points:
        cos_value = qsp_response(COS_QSP_PHASES, float(x)).real
        sin_value = qsp_response(SIN_QSP_PHASES, float(x)).imag
        cos_errors.append(abs(cos_value - 0.5 * np.cos(tau * x)))
        sin_errors.append(abs(sin_value - 0.5 * np.sin(tau * x)))
    return max(cos_errors), max(sin_errors)


def qsp_phase_factor(x: float) -> complex:
    cos_part = qsp_response(COS_QSP_PHASES, x).real
    sin_part = qsp_response(SIN_QSP_PHASES, x).imag
    return 2.0 * (cos_part - 1.0j * sin_part)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--num-samples", type=int, default=401)
    args = parser.parse_args()

    num_qubits = len(REFERENCE_TERMS[0].word)
    alpha = sum(abs(term.coefficient) for term in REFERENCE_TERMS)
    tau = alpha * EVOLUTION_TIME

    encoding = solvers.PauliLCU(spin_hamiltonian(REFERENCE_TERMS), num_qubits)
    metadata = encoding.metadata()
    if abs(metadata.normalization - alpha) > 1e-12:
        raise RuntimeError("PauliLCU normalization does not match reference alpha")

    hamiltonian = hamiltonian_matrix(REFERENCE_TERMS)
    initial_state = random_normalized_ket(num_qubits, INITIAL_STATE_SEED)
    dense_evolved = la.expm(-1.0j * EVOLUTION_TIME * hamiltonian) @ initial_state

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    amplitudes = eigenvectors.conj().T @ initial_state
    exact_phases = np.exp(-1.0j * EVOLUTION_TIME * eigenvalues)
    diagonal_evolved = eigenvectors @ (exact_phases * amplitudes)

    scaled_eigenvalues = eigenvalues / alpha
    qsp_factors = np.array([qsp_phase_factor(float(x)) for x in scaled_eigenvalues])
    qsp_evolved = eigenvectors @ (qsp_factors * amplitudes)

    sample_points = np.linspace(-1.0, 1.0, args.num_samples)
    max_cos_error, max_sin_error = component_errors(tau, sample_points)
    scalar_error = max(
        abs(qsp_phase_factor(float(x)) - np.exp(-1.0j * tau * x))
        for x in sample_points
    )

    dense_diagonal_l2 = np.linalg.norm(dense_evolved - diagonal_evolved)
    qsp_l2 = np.linalg.norm(qsp_evolved - diagonal_evolved)
    qsp_max_amplitude_error = np.max(np.abs(qsp_evolved - diagonal_evolved))
    qsp_fidelity = abs(np.vdot(diagonal_evolved, qsp_evolved)) ** 2

    print("QSPPACK phase validation for 4-qubit Hamiltonian simulation")
    print("=" * 64)
    print(f"Number of Pauli terms: {len(REFERENCE_TERMS)}")
    print(f"LCU normalization alpha: {alpha:.12f}")
    print(f"Evolution time: {EVOLUTION_TIME:.12f}")
    print(f"Scaled tau = alpha * time: {tau:.12f}")
    print(f"Phase-generation degree: {PHASE_GENERATION_DEGREE}")
    print(f"Cosine phases: {len(COS_QSP_PHASES)}")
    print(f"Sine phases: {len(SIN_QSP_PHASES)}")
    print()
    print("Scalar QSP response errors on [-1, 1]")
    print(f"  Max cosine component error: {max_cos_error:.6e}")
    print(f"  Max sine component error:   {max_sin_error:.6e}")
    print(f"  Max exp(-i tau x) error:    {scalar_error:.6e}")
    print()
    print("State comparison in Hamiltonian eigenbasis")
    print(f"  Dense expm vs diagonal L2:  {dense_diagonal_l2:.6e}")
    print(f"  QSP vs diagonal L2:         {qsp_l2:.6e}")
    print(f"  QSP max amplitude error:    {qsp_max_amplitude_error:.6e}")
    print(f"  QSP fidelity:               {qsp_fidelity:.12f}")

    if max_cos_error > args.tolerance:
        return 1
    if max_sin_error > args.tolerance:
        return 1
    if scalar_error > args.tolerance:
        return 1
    if qsp_l2 > args.tolerance:
        return 1
    if abs(1.0 - qsp_fidelity) > args.tolerance:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
