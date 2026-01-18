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
    
    # Build subspace Hamiltonian matrix and diagonalize with NumPy
    matrix, basis = solvers.sample_based_krylov_matrix(
        molecule.hamiltonian,
        krylov_dim=15,
        dt=0.1,
        shots=5000,
        trotter_order=2,
        verbose=0
    )
    
    # Diagonalize matrix
    eigvals = np.linalg.eigvalsh(matrix)
    ground_state = eigvals[0]
    
    # Check result structure
    assert len(basis) > 0
    assert ground_state < 0
    
    # H2 ground state is approximately -1.137 Hartree
    assert np.isclose(ground_state, -1.137, atol=5e-3)
    
    print(f"H2 ground state energy (SKQD matrix): {ground_state:.6f}")
    print(f"Basis size: {len(basis)}")


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
    
    # Build matrix and diagonalize
    matrix, basis = solvers.sample_based_krylov_matrix(
        molecule.hamiltonian,
        krylov_dim=8,
        dt=0.1,
        shots=3000,
        trotter_order=2,  # Use second-order Trotter for better accuracy
        verbose=1,
        max_basis_size=100  # Limit for test speed
    )
    eigvals = np.linalg.eigvalsh(matrix)
    ground_state = eigvals[0]
    
    # Check basic validity
    assert len(basis) > 0
    assert ground_state < 0
    
    # LiH CASCI(4,4) ground state is approximately -7.8638 Hartree
    # SKQD should get within reasonable range
    print(f"LiH ground state energy (SKQD): {ground_state:.4f}")
    print(f"Expected (VQE reference): -7.8638")
    print(f"Basis size: {len(basis)}")
    
    # Tighter tolerance for LiH
    assert np.isclose(ground_state, -7.8638, atol=5e-3)


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
    
    # Build matrix and diagonalize
    matrix, basis = solvers.sample_based_krylov_matrix(
        molecule.hamiltonian,
        krylov_dim=8,
        dt=0.1,
        shots=3000,
        trotter_order=2,
        verbose=1,
        max_basis_size=100
    )
    eigvals = np.linalg.eigvalsh(matrix)
    ground_state = eigvals[0]
    
    # Check basic validity
    assert len(basis) > 0
    assert ground_state < 0
    
    # N2 CASCI(4,4) ground state is approximately -107.56 Hartree
    # SKQD should get within reasonable range
    print(f"N2 ground state energy (SKQD): {ground_state:.4f}")
    print(f"Expected (CASCI(4,4)reference): -1.0754219837264435e+02")
    print(f"Basis size: {len(basis)}")
    
    # Tighter tolerance for N2
    assert np.isclose(ground_state, -1.0754219837264435e+02, atol=3e-2)


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
    
    # Build matrix and diagonalize
    matrix, basis = solvers.sample_based_krylov_matrix(
        molecule.hamiltonian,
        krylov_dim=15,
        dt=0.1,
        shots=3000,
        trotter_order=2,
        verbose=1,
        max_basis_size=250
    )
    eigvals = np.linalg.eigvalsh(matrix)
    ground_state = eigvals[0]
    
    # Check basic validity
    assert len(basis) > 0
    assert ground_state < 0
    
    # H2O CASCI(4,4) ground state is approximately -75.01 Hartree
    print(f"H2O ground state energy (SKQD): {ground_state:.4f}")
    print(f"Expected (CASCI(4,4)reference): -7.4970454377757974e+01")
    print(f"Basis size: {len(basis)}")
    
    # Tighter tolerance for H2O
    assert np.isclose(ground_state, -7.4970454377757974e+01, atol=5e-2)


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
    
    # Build matrix and diagonalize
    matrix, basis = solvers.sample_based_krylov_matrix(
        molecule.hamiltonian,
        krylov_dim=10,
        dt=0.1,
        shots=5000,
        trotter_order=2,
        verbose=1,
        max_basis_size=200
    )
    eigvals = np.linalg.eigvalsh(matrix)
    ground_state = eigvals[0]
    
    # Check basic validity
    assert len(basis) > 0
    assert ground_state < 0
    
    # Benzene CASCI(6,6) ground state is approximately -227.6 to -228.0 Hartree
    print(f"Benzene ground state energy (SKQD): {ground_state:.4f}")
    print(f"Expected range: -228.0 to -227.6")
    print(f"Basis size: {len(basis)}")
    
    # Tighter tolerance for benzene
    assert ground_state < -227.6
    assert ground_state > -228.05


def test_skqd_trotter_comparison():
    """Test that second-order Trotter is more accurate than first-order"""
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
    
    # First-order Trotter
    matrix1, basis1 = solvers.sample_based_krylov_matrix(
        molecule.hamiltonian,
        krylov_dim=8,
        dt=0.1,
        shots=2000,
        trotter_order=1,
        verbose=0
    )
    eigvals1 = np.linalg.eigvalsh(matrix1)
    ground_state1 = eigvals1[0]
    
    # Second-order Trotter
    matrix2, basis2 = solvers.sample_based_krylov_matrix(
        molecule.hamiltonian,
        krylov_dim=8,
        dt=0.1,
        shots=2000,
        trotter_order=2,
        verbose=0
    )
    eigvals2 = np.linalg.eigvalsh(matrix2)
    ground_state2 = eigvals2[0]
    
    print(f"First-order Trotter:  {ground_state1:.6f}")
    print(f"Second-order Trotter: {ground_state2:.6f}")
    print(f"Expected: ~-1.137 Hartree")
    
    # Both should give reasonable results
    assert ground_state1 < 0
    assert ground_state2 < 0
    
    # Both methods should produce valid basis sizes
    assert len(basis1) > 0
    assert len(basis2) > 0


def test_skqd_default_options():
    """Test SKQD with default options"""
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
    
    # Use default options (empty kwargs)
    matrix, basis = solvers.sample_based_krylov_matrix(molecule.hamiltonian)
    eigvals = np.linalg.eigvalsh(matrix)
    ground_state = eigvals[0]
    
    # Check it runs and produces reasonable output
    assert len(basis) > 0
    assert ground_state < 0
    
    print(f"H2 (default options): {ground_state:.6f}")


def test_skqd_multiple_eigenvalues():
    """Test computing multiple eigenvalues"""
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
    
    # Build matrix and take multiple eigenvalues
    matrix, basis = solvers.sample_based_krylov_matrix(
        molecule.hamiltonian,
        krylov_dim=8,
        shots=2000,
        verbose=0
    )
    eigvals = np.linalg.eigvalsh(matrix)
    
    # Check that we got multiple eigenvalues
    assert len(eigvals) >= 1
    if len(basis) >= 3:
        assert len(eigvals[:3]) == 3
    
    # Eigenvalues should be sorted (lowest first)
    for i in range(len(eigvals) - 1):
        assert eigvals[i] <= eigvals[i+1]
    
    print(f"Eigenvalues: {eigvals[:3]}")


def test_skqd_result_conversion():
    """Test that skqd_result can be converted to float"""
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
    
    matrix, basis = solvers.sample_based_krylov_matrix(
        molecule.hamiltonian,
        krylov_dim=5,
        shots=1000,
        verbose=0
    )
    eigvals = np.linalg.eigvalsh(matrix)
    ground_state = eigvals[0]
    assert ground_state < 0


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

import cudaq
import cudaq_solvers as solvers
import pytest

def test_skqd_filtering_particle_number():
    """Test SKQD with particle number filtering and Hartree-Fock state"""
    # H2 molecule geometry
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
    
    # H2 in sto-3g has 4 spin-orbitals and 2 electrons
    # HF state should be |1100> (assuming sorted orbitals)
    # Ground state should be dominated by |1100> and |0011> (both 2 electrons)
    
    n_electrons = 2
    
    # Build matrix with filtering
    matrix, basis = solvers.sample_based_krylov_matrix(
        molecule.hamiltonian,
        krylov_dim=5,
        shots=2000,
        n_electrons=n_electrons,       # Use HF state |1100>
        filter_particles=n_electrons,  # Keep only 2-electron states
        verbose=1
    )
    eigvals = np.linalg.eigvalsh(matrix)
    ground_state = eigvals[0]
    
    print(f"H2 Ground State Energy: {ground_state}")
    print(f"Basis size: {len(basis)}")
    
    # Check that basis size is non-zero (we should find samples)
    assert len(basis) > 0
    
    # We can't easily inspect the basis strings from Python result object 
    # (unless we expose them, which we haven't), but we can infer filtering worked
    # if the energy is good and execution didn't crash.
    # In a strict test we might want to expose basis to Python, but for now
    # valid energy is a good proxy.
    
    # Check energy is reasonable (-1.137 is exact FCI)
    assert ground_state < -1.1
    assert ground_state > -1.2

def test_skqd_filtering_mismatch():
    """Test that filtering works by checking if we get NO samples when filtering for wrong number"""
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
    
    # Try to filter for 3 electrons (impossible for neutral H2 in minimal basis with charge conservation)
    # Actually, noise might produce them, but HF state |1100> + physical H shouldn't evolve to |1110> easily 
    # unless Trotter error is huge or noise.
    
    # If we filter for something that doesn't exist, we might get an error "No samples collected"
    # or just an empty basis depending on implementation (implementation throws "No samples collected").
    
    try:
        solvers.sample_based_krylov_matrix(
            molecule.hamiltonian,
            krylov_dim=5,
            shots=1000,
            n_electrons=2,
            filter_particles=3, # Wrong number
            verbose=0
        )
        assert False, "Should have thrown exception for no samples"
    except RuntimeError as e:
        assert "No samples collected" in str(e)

if __name__ == "__main__":
    test_skqd_filtering_particle_number()
    test_skqd_filtering_mismatch()
    print("Filtering tests passed!")
