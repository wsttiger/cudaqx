#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
Reference: Hamiltonian time evolution for QSVT/QSP validation.

This example builds a 4-qubit Pauli Hamiltonian, prepares a random normalized
state vector, evolves it exactly using a dense matrix exponential, then evolves
the same state by diagonalizing the Hamiltonian with NumPy. The two evolved
states are compared by norm error and fidelity.

This is a classical reference path for future QSVT/QSP Hamiltonian simulation
examples. The current QSVT primitive layer can validate and execute externally
provided phases; generic phase generation is intentionally not performed here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy import linalg as la


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


def random_normalized_ket(num_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dimension = 2**num_qubits
    state = rng.normal(size=dimension) + 1.0j * rng.normal(size=dimension)
    return state / np.linalg.norm(state)


def evolve_by_dense_exponential(
    hamiltonian: np.ndarray, state: np.ndarray, time: float
) -> np.ndarray:
    return la.expm(-1.0j * time * hamiltonian) @ state


def evolve_by_diagonalization(
    hamiltonian: np.ndarray, state: np.ndarray, time: float
) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    amplitudes = eigenvectors.conj().T @ state
    phases = np.exp(-1.0j * time * eigenvalues)
    evolved = eigenvectors @ (phases * amplitudes)
    return evolved, eigenvalues


def expectation_value(state: np.ndarray, operator: np.ndarray) -> complex:
    return np.vdot(state, operator @ state)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    terms = [
        PauliTerm(0.70, "ZIII"),
        PauliTerm(-0.43, "IZII"),
        PauliTerm(0.31, "IIZI"),
        PauliTerm(-0.22, "IIIZ"),
        PauliTerm(0.19, "XXII"),
        PauliTerm(-0.17, "IYYI"),
        PauliTerm(0.13, "IZZX"),
        PauliTerm(0.11, "XYYX"),
    ]
    num_qubits = len(terms[0].word)
    hamiltonian = hamiltonian_matrix(terms)
    initial_state = random_normalized_ket(num_qubits, args.seed)

    dense_evolved = evolve_by_dense_exponential(
        hamiltonian, initial_state, args.time
    )
    diagonal_evolved, eigenvalues = evolve_by_diagonalization(
        hamiltonian, initial_state, args.time
    )

    l2_error = np.linalg.norm(dense_evolved - diagonal_evolved)
    max_amplitude_error = np.max(np.abs(dense_evolved - diagonal_evolved))
    fidelity = np.abs(np.vdot(dense_evolved, diagonal_evolved)) ** 2

    initial_energy = expectation_value(initial_state, hamiltonian).real
    final_energy = expectation_value(diagonal_evolved, hamiltonian).real

    alpha = sum(abs(term.coefficient) for term in terms)
    scaled_eigenvalues = eigenvalues / alpha

    print("4-qubit Hamiltonian simulation reference")
    print("=" * 48)
    print(f"Number of Pauli terms: {len(terms)}")
    print(f"LCU normalization alpha: {alpha:.12f}")
    print(
        "Scaled eigenvalue range: "
        f"[{scaled_eigenvalues.min():.12f}, {scaled_eigenvalues.max():.12f}]"
    )
    print(f"Evolution time: {args.time:.12f}")
    print(f"Initial state seed: {args.seed}")
    print()
    print("Comparison: dense expm vs NumPy diagonalization")
    print(f"  L2 state error:          {l2_error:.6e}")
    print(f"  Max amplitude error:     {max_amplitude_error:.6e}")
    print(f"  Fidelity:                {fidelity:.12f}")
    print(f"  Initial <H>:             {initial_energy:.12f}")
    print(f"  Final <H>:               {final_energy:.12f}")
    print(f"  Energy drift:            {abs(final_energy - initial_energy):.6e}")

    if l2_error > args.tolerance:
        return 1
    if abs(1.0 - fidelity) > args.tolerance:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
