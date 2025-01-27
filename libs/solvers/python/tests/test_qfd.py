# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest
import numpy as np
import scipy

import cudaq
from cudaq import spin

import cudaq_solvers as solvers


def test_time_evolution_fixed_hamiltonian():
    np.random.seed(12345)
    num_qubits = 3
    # Create fixed hamiltonian
    H = -5.0*spin.x(0)*spin.y(1)*spin.z(2) + 2.4*spin.z(0)*spin.z(1)*spin.y(2) + 0.342*spin.y(0)*spin.x(1)*spin.x(2)
    # Set relatively small timestep (this will depend on the deepest eigenvalue of H)
    dt = 0.1 

    # Create random initial vector
    v0 = np.random.randn(2**num_qubits)
    v0 = v0 / np.linalg.norm(v0)

    # Create "exact" time-evolution operator (matrix)
    Hm = H.to_matrix()
    U = scipy.linalg.expm(1.0j * dt * Hm)

    # Apply the "exact" TE operator
    result_gold = np.dot(U, v0)

    # Higher orders of Trotter splittiing should yield closer results to the "exact" result
    threshs = [0.9, 0.45, 0.25, 0.12]
    orders = [1, 2, 4, 8]
    for thresh,order in zip(threshs, orders):
        result = np.array(solvers.time_evolve_state(H, num_qubits, dt, np.real(v0), order=order))
        assert(np.max(np.abs(result_gold - result) < thresh))

def test_time_evolved_krylov_subspace_creation():
    np.random.seed(12345)
    num_qubits = 8
    krylov_dim = 10
    # Create random hamiltonian
    H = cudaq.SpinOperator.random(qubit_count=num_qubits, term_count=42)
    # Set to a small timestep
    dt = 0.01 

    # Create random intial vector
    v0 = np.random.randn(2**num_qubits)
    v0 = v0 / np.linalg.norm(v0)

    # Build up Krylov subspace
    u = [np.array(solvers.time_evolve_state(H, num_qubits, m*dt, np.real(v0), order=1)) for m in range(krylov_dim)]
    uvecs = np.zeros((2**num_qubits, krylov_dim)).astype(np.complex128)
    for i in range(krylov_dim): uvecs[:,i] = u[i]

    # Create a set of orthonormal vectors in the Krylov subspace
    B = scipy.linalg.orth(uvecs)

    # Verify the shape
    assert(B.shape == (256,10))
