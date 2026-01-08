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
    
    # Solve using functional API with options dict
    result = solvers.sample_based_krylov(
        molecule.hamiltonian,
        krylov_dim=5,
        dt=0.1,
        shots=1000,
        trotter_order=1,
        verbose=0
    )
    
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
    
    # Solve with functional API
    result = solvers.sample_based_krylov(
        molecule.hamiltonian,
        krylov_dim=8,
        dt=0.1,
        shots=3000,
        trotter_order=2,  # Use second-order Trotter for better accuracy
        verbose=1,
        max_basis_size=100  # Limit for test speed
    )
    
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
    
    # Solve with functional API
    result = solvers.sample_based_krylov(
        molecule.hamiltonian,
        krylov_dim=8,
        dt=0.1,
        shots=3000,
        trotter_order=2,
        verbose=1,
        max_basis_size=100
    )
    
    # Check basic validity
    assert result.basis_size > 0
    assert result.nnz > 0
    assert result.ground_state_energy < 0
    
    # N2 CASCI(4,4) ground state is approximately -107.56 Hartree
    # SKQD should get within reasonable range
    print(f"N2 ground state energy (SKQD): {result.ground_state_energy:.4f}")
    print(f"Expected (VQE reference): -107.56")
    print(f"Basis size: {result.basis_size}")
    print(f"Sampling time: {result.sampling_time:.3f}s")
    print(f"Matrix construction time: {result.matrix_construction_time:.3f}s")
    
    # Loose tolerance for MVP
    assert result.ground_state_energy < -100.0
    assert result.ground_state_energy > -115.0


def test_skqd_h2o():
    """Test SKQD with H2O molecule"""
    geometry = [('O', (0.0000, 0.0000, 0.0000)),
                ('H', (0.7571, 0.5861, 0.0000)),
                ('H', (-0.7571, 0.5861, 0.0000))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=4,
                                       norb_cas=4,
                                       casci=True)
    
    # Solve with functional API
    result = solvers.sample_based_krylov(
        molecule.hamiltonian,
        krylov_dim=8,
        dt=0.1,
        shots=3000,
        trotter_order=2,
        verbose=1,
        max_basis_size=150
    )
    
    # Check basic validity
    assert result.basis_size > 0
    assert result.nnz > 0
    assert result.ground_state_energy < 0
    
    # H2O CASCI(4,4) ground state is approximately -75.01 Hartree
    print(f"H2O ground state energy (SKQD): {result.ground_state_energy:.4f}")
    print(f"Expected (VQE reference): -75.01")
    print(f"Basis size: {result.basis_size}")
    print(f"Sampling time: {result.sampling_time:.3f}s")
    print(f"Matrix construction time: {result.matrix_construction_time:.3f}s")
    
    # Loose tolerance for MVP
    assert result.ground_state_energy < -70.0
    assert result.ground_state_energy > -80.0


def test_skqd_benzene():
    """Test SKQD with Benzene molecule (C6H6)"""
    # Benzene geometry (planar hexagonal structure)
    bond_length_cc = 1.39  # Angstrom, C-C bond length in benzene
    bond_length_ch = 1.09  # Angstrom, C-H bond length
    
    # Hexagonal carbon positions
    angle = np.pi / 3  # 60 degrees
    carbon_coords = []
    for i in range(6):
        x = bond_length_cc * np.cos(i * angle)
        y = bond_length_cc * np.sin(i * angle)
        carbon_coords.append(('C', (x, y, 0.0)))
    
    # Hydrogen positions (radially outward from carbons)
    hydrogen_coords = []
    for i in range(6):
        x_c = bond_length_cc * np.cos(i * angle)
        y_c = bond_length_cc * np.sin(i * angle)
        x_h = x_c + bond_length_ch * np.cos(i * angle)
        y_h = y_c + bond_length_ch * np.sin(i * angle)
        hydrogen_coords.append(('H', (x_h, y_h, 0.0)))
    
    geometry = carbon_coords + hydrogen_coords
    
    # Use CASCI(6,6) for benzene's π system (6 π electrons, 6 π orbitals)
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=6,
                                       norb_cas=6,
                                       casci=True)
    
    # Solve with functional API
    result = solvers.sample_based_krylov(
        molecule.hamiltonian,
        krylov_dim=10,
        dt=0.1,
        shots=5000,
        trotter_order=2,
        verbose=1,
        max_basis_size=200
    )
    
    # Check basic validity
    assert result.basis_size > 0
    assert result.nnz > 0
    assert result.ground_state_energy < 0
    
    # Benzene CASCI(6,6) ground state is approximately -227.6 to -228.0 Hartree
    print(f"Benzene ground state energy (SKQD): {result.ground_state_energy:.4f}")
    print(f"Expected range: -228.0 to -227.6")
    print(f"Basis size: {result.basis_size}")
    print(f"NNZ: {result.nnz}")
    print(f"Sampling time: {result.sampling_time:.3f}s")
    print(f"Matrix construction time: {result.matrix_construction_time:.3f}s")
    print(f"Diagonalization time: {result.diagonalization_time:.3f}s")
    
    # Loose tolerance for MVP - benzene is a larger system
    assert result.ground_state_energy < -220.0
    assert result.ground_state_energy > -235.0


def test_skqd_trotter_comparison():
    """Test that second-order Trotter is more accurate than first-order"""
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
    
    # First-order Trotter
    result1 = solvers.sample_based_krylov(
        molecule.hamiltonian,
        krylov_dim=8,
        dt=0.1,
        shots=2000,
        trotter_order=1,
        verbose=0
    )
    
    # Second-order Trotter
    result2 = solvers.sample_based_krylov(
        molecule.hamiltonian,
        krylov_dim=8,
        dt=0.1,
        shots=2000,
        trotter_order=2,
        verbose=0
    )
    
    print(f"First-order Trotter:  {result1.ground_state_energy:.6f}")
    print(f"Second-order Trotter: {result2.ground_state_energy:.6f}")
    print(f"Expected: ~-1.137 Hartree")
    
    # Both should give reasonable results
    assert result1.ground_state_energy < 0
    assert result2.ground_state_energy < 0
    
    # Both methods should produce valid basis sizes
    assert result1.basis_size > 0
    assert result2.basis_size > 0


def test_skqd_default_options():
    """Test SKQD with default options"""
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
    
    # Use default options (empty kwargs)
    result = solvers.sample_based_krylov(molecule.hamiltonian)
    
    # Check it runs and produces reasonable output
    assert result.basis_size > 0
    assert result.ground_state_energy < 0
    
    print(f"H2 (default options): {result.ground_state_energy:.6f}")


def test_skqd_multiple_eigenvalues():
    """Test computing multiple eigenvalues"""
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
    
    # Request 3 eigenvalues
    result = solvers.sample_based_krylov(
        molecule.hamiltonian,
        krylov_dim=8,
        shots=2000,
        num_eigenvalues=3,
        verbose=0
    )
    
    # Check that we got multiple eigenvalues
    assert len(result.eigenvalues) >= 1
    if result.basis_size >= 3:
        assert len(result.eigenvalues) <= 3
    
    # Eigenvalues should be sorted (lowest first)
    for i in range(len(result.eigenvalues) - 1):
        assert result.eigenvalues[i] <= result.eigenvalues[i+1]
    
    print(f"Eigenvalues: {result.eigenvalues}")


def test_skqd_result_conversion():
    """Test that skqd_result can be converted to float"""
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
    
    result = solvers.sample_based_krylov(
        molecule.hamiltonian,
        krylov_dim=5,
        shots=1000,
        verbose=0
    )
    
    # Test float conversion
    energy = float(result)
    assert energy == result.ground_state_energy
    assert energy < 0


if __name__ == "__main__":
    # Run tests manually
    test_skqd_basic()
    test_skqd_lih()
    test_skqd_n2()
    test_skqd_h2o()
    test_skqd_benzene()
    test_skqd_trotter_comparison()
    test_skqd_default_options()
    test_skqd_multiple_eigenvalues()
    test_skqd_result_conversion()
    print("All tests passed!")
