# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np

import cudaq
import cudaq_solvers as solvers


def test_skqd_basic():
    """Test basic SKQD functionality with H2"""
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
    
    # Create SKQD solver
    solver = solvers.SampleBasedKrylov(molecule.hamiltonian)
    
    # Configure for small test
    config = solvers.SKQDConfig()
    config.krylov_dim = 5
    config.dt = 0.1
    config.shots = 1000
    config.trotter_order = 1
    config.verbose = 0
    
    # Solve
    result = solver.solve(config)
    
    # Check result structure
    assert result.basis_size > 0
    assert result.nnz > 0
    assert result.ground_state_energy < 0
    
    # Should be reasonably close to true ground state
    # H2 ground state is approximately -1.137 Hartree
    assert result.ground_state_energy < -0.5  # At least qualitatively correct
    
    print(f"H2 ground state energy (SKQD): {result.ground_state_energy:.6f}")
    print(f"Basis size: {result.basis_size}")


def test_skqd_lih():
    """Test SKQD with LiH molecule"""
    geometry = [('Li', (0.3925, 0., 0.)), ('H', (-1.1774, 0., .0))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=4,
                                       norb_cas=4,
                                       casci=True)
    
    # Create SKQD solver
    solver = solvers.SampleBasedKrylov(molecule.hamiltonian)
    
    # Configure parameters
    config = solvers.SKQDConfig()
    config.krylov_dim = 8
    config.dt = 0.1
    config.shots = 3000
    config.trotter_order = 2  # Use second-order Trotter for better accuracy
    config.verbose = 1
    config.max_basis_size = 100  # Limit for test speed
    
    # Solve
    result = solver.solve(config)
    
    # Check basic validity
    assert result.basis_size > 0
    assert result.nnz > 0
    assert result.ground_state_energy < 0
    
    # LiH CASCI(4,4) ground state is approximately -7.8638 Hartree
    # SKQD should get within reasonable range
    print(f"LiH ground state energy (SKQD): {result.ground_state_energy:.4f}")
    print(f"Expected (VQE reference): -7.8638")
    print(f"Basis size: {result.basis_size}")
    print(f"Sampling time: {result.sampling_time:.3f}s")
    print(f"Matrix construction time: {result.matrix_construction_time:.3f}s")
    
    # Loose tolerance for MVP - should be in the right ballpark
    assert result.ground_state_energy < -6.0  # Should be strongly bound
    assert result.ground_state_energy > -10.0  # Should be realistic


def test_skqd_n2():
    """Test SKQD with N2 molecule"""
    geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=4,
                                       norb_cas=4,
                                       casci=True)
    
    # Create SKQD solver
    solver = solvers.SampleBasedKrylov(molecule.hamiltonian)
    
    # Configure parameters
    config = solvers.SKQDConfig()
    config.krylov_dim = 8
    config.dt = 0.1
    config.shots = 3000
    config.trotter_order = 2  # Use second-order Trotter
    config.verbose = 1
    config.max_basis_size = 100  # Limit for test speed
    
    # Solve
    result = solver.solve(config)
    
    # Check basic validity
    assert result.basis_size > 0
    assert result.nnz > 0
    assert result.ground_state_energy < 0
    
    # N2 CASCI(4,4) ground state is approximately -107.5421 Hartree
    print(f"N2 ground state energy (SKQD): {result.ground_state_energy:.4f}")
    print(f"Expected (VQE reference): -107.5421")
    print(f"Basis size: {result.basis_size}")
    print(f"Sampling time: {result.sampling_time:.3f}s")
    print(f"Matrix construction time: {result.matrix_construction_time:.3f}s")
    
    # Loose tolerance for MVP
    assert result.ground_state_energy < -100.0  # Should be strongly bound
    assert result.ground_state_energy > -120.0  # Should be realistic


def test_skqd_h2o():
    """Test SKQD with H2O molecule"""
    # H2O geometry at equilibrium (bond length ~0.958 Å, angle ~104.5°)
    geometry = [
        ('O', (0.0000, 0.0000, 0.1173)),
        ('H', (0.0000, 0.7572, -0.4692)),
        ('H', (0.0000, -0.7572, -0.4692))
    ]
    
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=4,
                                       norb_cas=4,
                                       casci=True)
    
    # Create SKQD solver
    solver = solvers.SampleBasedKrylov(molecule.hamiltonian)
    
    # Configure parameters
    config = solvers.SKQDConfig()
    config.krylov_dim = 8
    config.dt = 0.1
    config.shots = 3000
    config.trotter_order = 2  # Use second-order Trotter
    config.verbose = 1
    config.max_basis_size = 100  # Limit for test speed
    
    # Solve
    result = solver.solve(config)
    
    # Check basic validity
    assert result.basis_size > 0
    assert result.nnz > 0
    assert result.ground_state_energy < 0
    
    # H2O CASCI(4,4) ground state is approximately -75.7 Hartree
    # SKQD should get within reasonable range
    print(f"H2O ground state energy (SKQD): {result.ground_state_energy:.4f}")
    print(f"Expected range: approximately -75 to -76 Hartree")
    print(f"Basis size: {result.basis_size}")
    print(f"Sampling time: {result.sampling_time:.3f}s")
    print(f"Matrix construction time: {result.matrix_construction_time:.3f}s")
    
    # Loose tolerance for MVP - should be in the right ballpark
    assert result.ground_state_energy < -70.0  # Should be strongly bound
    assert result.ground_state_energy > -80.0  # Should be realistic


def test_skqd_benzene():
    """Test SKQD with benzene (C6H6) molecule"""
    # Benzene geometry - regular hexagon with C-C bond length 1.39 Å, C-H bond length 1.09 Å
    # Carbons form a hexagon in xy-plane
    import math
    r_cc = 1.39  # C-C bond length in Angstroms
    r_ch = 1.09  # C-H bond length in Angstroms
    
    geometry = []
    # Carbon atoms in a hexagon
    for i in range(6):
        angle = i * math.pi / 3.0  # 60 degrees between carbons
        x = r_cc * math.cos(angle)
        y = r_cc * math.sin(angle)
        geometry.append(('C', (x, y, 0.0)))
    
    # Hydrogen atoms bonded to each carbon (pointing outward)
    for i in range(6):
        angle = i * math.pi / 3.0
        r_total = r_cc + r_ch  # Distance from center to H
        x = r_total * math.cos(angle)
        y = r_total * math.sin(angle)
        geometry.append(('H', (x, y, 0.0)))
    
    # Use 6 electrons in 6 orbitals - the π-system (12 qubits)
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=6,
                                       norb_cas=6,
                                       casci=True)
    
    # Create SKQD solver
    solver = solvers.SampleBasedKrylov(molecule.hamiltonian)
    
    # Configure parameters - benzene is larger, may need more resources
    config = solvers.SKQDConfig()
    config.krylov_dim = 6  # Smaller for larger system
    config.dt = 0.1
    config.shots = 2000  # Fewer shots for speed
    config.trotter_order = 2
    config.verbose = 1
    config.max_basis_size = 100  # Limit for test speed
    
    # Solve
    result = solver.solve(config)
    
    # Check basic validity
    assert result.basis_size > 0
    assert result.nnz > 0
    assert result.ground_state_energy < 0
    
    # Benzene CASCI(6,6) ground state is approximately -230 Hartree
    print(f"Benzene ground state energy (SKQD): {result.ground_state_energy:.4f}")
    print(f"Expected range: approximately -225 to -235 Hartree")
    print(f"Basis size: {result.basis_size}")
    print(f"Sampling time: {result.sampling_time:.3f}s")
    print(f"Matrix construction time: {result.matrix_construction_time:.3f}s")
    
    # Loose tolerance for MVP
    assert result.ground_state_energy < -200.0  # Should be strongly bound
    assert result.ground_state_energy > -250.0  # Should be realistic


def test_skqd_config():
    """Test SKQD configuration options"""
    # Simple test Hamiltonian
    from cudaq import spin
    hamiltonian = -1.0 * spin.z(0) - 0.5 * spin.x(0)
    
    # Test configuration defaults
    config = solvers.SKQDConfig()
    assert config.krylov_dim == 15
    assert config.dt == 0.1
    assert config.shots == 10000
    assert config.num_eigenvalues == 1
    assert config.trotter_order == 1
    assert config.max_basis_size == 0
    assert config.verbose == 0
    
    # Test configuration modification
    config.krylov_dim = 10
    config.dt = 0.05
    config.shots = 2000
    config.trotter_order = 2
    config.verbose = 1
    
    assert config.krylov_dim == 10
    assert config.dt == 0.05
    assert config.shots == 2000
    assert config.trotter_order == 2
    assert config.verbose == 1


def test_skqd_trotter_orders():
    """Test both first and second order Trotter"""
    from cudaq import spin
    hamiltonian = -1.0 * spin.z(0) - 0.5 * spin.x(0)
    
    # Test first-order Trotter
    solver1 = solvers.SampleBasedKrylov(hamiltonian)
    config1 = solvers.SKQDConfig()
    config1.krylov_dim = 5
    config1.dt = 0.1
    config1.shots = 1000
    config1.trotter_order = 1
    config1.verbose = 0
    
    result1 = solver1.solve(config1)
    assert result1.basis_size > 0
    assert result1.ground_state_energy < 0
    
    # Test second-order Trotter (Suzuki)
    solver2 = solvers.SampleBasedKrylov(hamiltonian)
    config2 = solvers.SKQDConfig()
    config2.krylov_dim = 5
    config2.dt = 0.1
    config2.shots = 1000
    config2.trotter_order = 2
    config2.verbose = 0
    
    result2 = solver2.solve(config2)
    assert result2.basis_size > 0
    assert result2.ground_state_energy < 0
    
    print(f"First-order Trotter energy: {result1.ground_state_energy:.6f}")
    print(f"Second-order Trotter energy: {result2.ground_state_energy:.6f}")


def test_skqd_result_structure():
    """Test SKQD result structure and attributes"""
    from cudaq import spin
    hamiltonian = -1.0 * spin.z(0)
    
    solver = solvers.SampleBasedKrylov(hamiltonian)
    config = solvers.SKQDConfig()
    config.krylov_dim = 3
    config.shots = 500
    config.verbose = 0
    
    result = solver.solve(config)
    
    # Check all attributes exist
    assert hasattr(result, 'ground_state_energy')
    assert hasattr(result, 'eigenvalues')
    assert hasattr(result, 'basis_size')
    assert hasattr(result, 'nnz')
    assert hasattr(result, 'sampling_time')
    assert hasattr(result, 'matrix_construction_time')
    assert hasattr(result, 'diagonalization_time')
    
    # Check types
    assert isinstance(result.ground_state_energy, float)
    assert isinstance(result.eigenvalues, list)
    assert isinstance(result.basis_size, int)
    assert isinstance(result.nnz, int)
    assert isinstance(result.sampling_time, float)
    assert isinstance(result.matrix_construction_time, float)
    assert isinstance(result.diagonalization_time, float)
    
    # Check values are reasonable
    assert result.basis_size > 0
    assert result.nnz > 0
    assert result.sampling_time >= 0
    assert result.matrix_construction_time >= 0
    assert result.diagonalization_time >= 0
    assert len(result.eigenvalues) >= 1
    
    # Test float conversion
    energy_float = float(result)
    assert energy_float == result.ground_state_energy


def test_skqd_get_eigenvalues():
    """Test getting multiple eigenvalues"""
    from cudaq import spin
    hamiltonian = spin.z(0) + spin.z(1)  # Two qubits for more eigenvalues
    
    solver = solvers.SampleBasedKrylov(hamiltonian)
    config = solvers.SKQDConfig()
    config.krylov_dim = 5
    config.shots = 1000
    config.num_eigenvalues = 3  # Request multiple eigenvalues
    config.verbose = 0
    
    result = solver.solve(config)
    
    # Get eigenvalues
    eigenvalues = solver.get_eigenvalues(3)
    
    assert len(eigenvalues) > 0
    assert isinstance(eigenvalues, list)
    
    # Eigenvalues should be in ascending order (lowest first)
    for i in range(len(eigenvalues) - 1):
        assert eigenvalues[i] <= eigenvalues[i + 1]
    
    print(f"Lowest 3 eigenvalues: {eigenvalues}")


def test_skqd_max_basis_size():
    """Test max basis size limiting"""
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
    
    solver = solvers.SampleBasedKrylov(molecule.hamiltonian)
    config = solvers.SKQDConfig()
    config.krylov_dim = 10
    config.dt = 0.1
    config.shots = 2000
    config.max_basis_size = 20  # Limit basis size
    config.verbose = 0
    
    result = solver.solve(config)
    
    # Basis size should not exceed the limit
    assert result.basis_size <= config.max_basis_size
    
    print(f"Basis size with limit {config.max_basis_size}: {result.basis_size}")


if __name__ == "__main__":
    # Run tests individually for debugging
    print("Testing basic SKQD...")
    test_skqd_basic()
    
    print("\nTesting LiH...")
    test_skqd_lih()
    
    print("\nTesting N2...")
    test_skqd_n2()
    
    print("\nAll tests passed!")
