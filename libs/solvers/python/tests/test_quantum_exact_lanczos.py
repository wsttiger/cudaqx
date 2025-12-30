#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Tests for Quantum Exact Lanczos (QEL) algorithm Python bindings."""

import sys
import os
# Add build directory to path if running tests directly
if 'cudaq_solvers' not in sys.modules:
    build_path = os.path.join(os.path.dirname(__file__), '../../../../build/python')
    if os.path.exists(build_path):
        sys.path.insert(0, build_path)

import pytest
import numpy as np
from cudaq import spin
import cudaq_solvers as solvers


def test_pauli_lcu_simple():
    """Test PauliLCU with a simple Hamiltonian."""
    # H = 0.5*X + 0.3*Z
    h = 0.5 * spin.x(0) + 0.3 * spin.z(0)
    
    encoding = solvers.PauliLCU(h, num_qubits=1)
    
    # Check basic properties
    assert encoding.num_ancilla == 1, "Expected 1 ancilla for 2 terms"
    assert encoding.num_system == 1, "Expected 1 system qubit"
    assert abs(encoding.normalization - 0.8) < 1e-10, "Expected norm of 0.8"
    assert isinstance(encoding, solvers.BlockEncoding)
    assert isinstance(encoding, solvers.PauliLCU)
    
    # Check debug methods return numpy arrays
    angles = encoding.get_angles()
    assert isinstance(angles, np.ndarray)
    assert len(angles) == 1
    
    controls = encoding.get_term_controls()
    assert isinstance(controls, np.ndarray)
    
    ops = encoding.get_term_ops()
    assert isinstance(ops, np.ndarray)
    
    lengths = encoding.get_term_lengths()
    assert isinstance(lengths, np.ndarray)
    
    signs = encoding.get_term_signs()
    assert isinstance(signs, np.ndarray)


def test_pauli_lcu_larger():
    """Test PauliLCU with a larger Hamiltonian."""
    # H2 Hamiltonian (simplified)
    h2 = (-1.0523732 + 
          0.39793742 * spin.z(0) - 
          0.39793742 * spin.z(1) - 
          0.01128010 * spin.z(2) + 
          0.01128010 * spin.z(3) + 
          0.18093120 * spin.x(0) * spin.x(1) * spin.y(2) * spin.y(3))
    
    encoding = solvers.PauliLCU(h2, num_qubits=4)
    
    # Should need log2(6 terms) = 3 ancilla
    assert encoding.num_ancilla == 3
    assert encoding.num_system == 4
    assert encoding.normalization > 0.0


def test_qel_result_structure():
    """Test QELResult structure and methods."""
    result = solvers.QELResult()
    
    # Set test values
    result.krylov_dimension = 5
    result.num_system = 4
    result.num_ancilla = 3
    result.normalization = 2.0
    result.constant_term = -1.05
    result.moments = [0.1, 0.2, 0.3, 0.4, 0.5]
    result.hamiltonian_matrix = list(range(25))  # 5x5 flattened
    result.overlap_matrix = list(range(25, 50))
    
    # Check properties
    assert result.krylov_dimension == 5
    assert result.num_system == 4
    assert result.num_ancilla == 3
    assert abs(result.normalization - 2.0) < 1e-10
    assert abs(result.constant_term - (-1.05)) < 1e-10
    
    # Check helper methods return properly shaped arrays
    H = result.get_hamiltonian_matrix()
    S = result.get_overlap_matrix()
    moments = result.get_moments()
    
    assert isinstance(H, np.ndarray)
    assert isinstance(S, np.ndarray)
    assert isinstance(moments, np.ndarray)
    
    assert H.shape == (5, 5)
    assert S.shape == (5, 5)
    assert len(moments) == 5
    
    # Check __repr__
    repr_str = repr(result)
    assert "QELResult" in repr_str
    assert "krylov_dimension=5" in repr_str


def test_quantum_exact_lanczos_function_exists():
    """Test that quantum_exact_lanczos function is properly exposed."""
    assert hasattr(solvers, 'quantum_exact_lanczos')
    assert callable(solvers.quantum_exact_lanczos)
    
    # Check docstring exists
    doc = solvers.quantum_exact_lanczos.__doc__
    assert doc is not None
    assert len(doc) > 100
    assert "Quantum Exact Lanczos" in doc


def test_quantum_exact_lanczos_simple_hamiltonian():
    """Test QEL with a simple Hamiltonian."""
    # Simple 2-qubit Hamiltonian
    h = 0.5 * spin.z(0) + 0.3 * spin.z(1) + 0.2 * spin.x(0) * spin.x(1)
    
    # Run with small Krylov dimension
    result = solvers.quantum_exact_lanczos(
        h,
        num_qubits=2,
        n_electrons=1,
        krylov_dim=3,
        shots=-1,
        verbose=False
    )
    
    # Check result structure
    assert result.krylov_dimension == 3
    assert result.num_system == 2
    assert result.num_ancilla >= 1
    
    # Check matrices are the right size
    H = result.get_hamiltonian_matrix()
    S = result.get_overlap_matrix()
    
    assert H.shape == (3, 3)
    assert S.shape == (3, 3)
    
    # Moments should have 2*krylov_dim entries
    moments = result.get_moments()
    assert len(moments) == 6


def test_quantum_exact_lanczos_default_parameters():
    """Test QEL with default parameters."""
    h = 0.7 * spin.z(0)
    
    # Should work with only required parameters
    result = solvers.quantum_exact_lanczos(h, num_qubits=1, n_electrons=0)
    
    # Default krylov_dim is 10
    assert result.krylov_dimension == 10
    assert result.num_system == 1


def test_inheritance_hierarchy():
    """Test that PauliLCU properly inherits from BlockEncoding."""
    h = 0.7 * spin.z(0) + 0.3 * spin.x(0)
    encoding = solvers.PauliLCU(h, num_qubits=1)
    
    # Check inheritance
    assert isinstance(encoding, solvers.BlockEncoding)
    assert isinstance(encoding, solvers.PauliLCU)
    
    # Check all BlockEncoding methods exist
    assert hasattr(encoding, 'prepare')
    assert hasattr(encoding, 'unprepare')
    assert hasattr(encoding, 'select')
    assert hasattr(encoding, 'apply')
    assert hasattr(encoding, 'num_ancilla')
    assert hasattr(encoding, 'num_system')
    assert hasattr(encoding, 'normalization')
    
    # Check PauliLCU-specific methods
    assert hasattr(encoding, 'get_angles')
    assert hasattr(encoding, 'get_term_controls')
    assert hasattr(encoding, 'get_term_ops')
    assert hasattr(encoding, 'get_term_lengths')
    assert hasattr(encoding, 'get_term_signs')


if __name__ == "__main__":
    # Allow running tests individually for debugging
    test_pauli_lcu_simple()
    test_pauli_lcu_larger()
    test_qel_result_structure()
    test_quantum_exact_lanczos_function_exists()
    test_quantum_exact_lanczos_simple_hamiltonian()
    test_quantum_exact_lanczos_default_parameters()
    test_inheritance_hierarchy()
    
    print("All tests passed!")

