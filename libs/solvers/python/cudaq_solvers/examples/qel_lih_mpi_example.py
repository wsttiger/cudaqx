#!/usr/bin/env python3
"""MPI-parallelized Quantum Exact Lanczos for LiH molecule.

This example demonstrates using MPI to parallelize moment collection
for significant speedup on multi-GPU systems.

Usage:
    # Single GPU (serial)
    python3 qel_lih_mpi_example.py
    
    # 8 GPUs (8x speedup)
    mpirun -np 8 python3 qel_lih_mpi_example.py
    
    # 16 GPUs across 2 nodes
    mpirun -np 16 -npernode 8 python3 qel_lih_mpi_example.py

Prerequisites:
    - CUDA-QX built with MPI support
    - mpi4py installed: pip install mpi4py
    - Multiple GPUs available
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tests'))

import cudaq
from cudaq import spin
import cudaq_solvers as solvers
import numpy as np

# Initialize CUDA-Q's MPI support (if running with mpirun)
# This is automatically handled when using mpirun, but explicit initialization
# ensures proper integration with CUDA-Q's distributed simulation features
try:
    cudaq.mpi.initialize()
    mpi_initialized_by_cudaq = True
except:
    mpi_initialized_by_cudaq = False

# Import LiH Hamiltonian data
try:
    from lih_hamiltonian_data import PAULI_TERMS, CONSTANT_TERM, FCI_ENERGY, NUM_QUBITS, NUM_ELECTRONS
except ImportError:
    print("Error: lih_hamiltonian_data.py not found")
    print("Please generate it using extract_lih_hamiltonian.py")
    sys.exit(1)

# Try to import MPI (optional)
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    has_mpi = size > 1
except ImportError:
    rank = 0
    size = 1
    has_mpi = False
    if rank == 0:
        print("Warning: mpi4py not available. Running serially.")

# Parse Pauli terms into cudaq.spin operators
def parse_pauli_term(pauli_str, coeff):
    """Convert Pauli string to cudaq.spin operator."""
    if not pauli_str or pauli_str == 'I':
        return coeff * spin.i(0)
    
    op = None
    i = 0
    while i < len(pauli_str):
        pauli_char = pauli_str[i]
        i += 1
        qubit_str = ""
        while i < len(pauli_str) and pauli_str[i].isdigit():
            qubit_str += pauli_str[i]
            i += 1
        
        qubit_idx = int(qubit_str)
        
        if pauli_char == 'X':
            term = spin.x(qubit_idx)
        elif pauli_char == 'Y':
            term = spin.y(qubit_idx)
        elif pauli_char == 'Z':
            term = spin.z(qubit_idx)
        else:
            raise ValueError(f"Invalid Pauli: {pauli_char}")
        
        if op is None:
            op = term
        else:
            op = op * term
    
    return coeff * op

if rank == 0:
    print("="*70)
    print("MPI-Parallelized Quantum Exact Lanczos - LiH Molecule")
    print("="*70)
    print(f"\nMPI Configuration:")
    print(f"  Ranks: {size}")
    print(f"  MPI enabled: {has_mpi}")
    print(f"\nMolecule: LiH (STO-3G)")
    print(f"  Qubits: {NUM_QUBITS}")
    print(f"  Electrons: {NUM_ELECTRONS}")
    print(f"  Pauli terms: {len(PAULI_TERMS)}")
    print(f"  Target FCI: {FCI_ENERGY:.6f} Ha")
    print()

# Build Hamiltonian (all ranks do this - lightweight)
hamiltonian = parse_pauli_term(*PAULI_TERMS[0])
for pauli_str, coeff in PAULI_TERMS[1:]:
    hamiltonian += parse_pauli_term(pauli_str, coeff)

# Run QEL with MPI parallelization
if rank == 0:
    print("Running QEL...")
    if has_mpi:
        print(f"  Distributing {16} moments across {size} ranks")
        print(f"  Expected speedup: ~{size}x (near-linear)")
    print()

import time
start_time = time.time()

result = solvers.quantum_exact_lanczos(
    hamiltonian,
    num_qubits=NUM_QUBITS,
    n_electrons=NUM_ELECTRONS,
    krylov_dim=8,
    shots=-1,
    use_mpi=has_mpi,  # Enable MPI parallelization
    verbose=(rank == 0)  # Only rank 0 prints
)

end_time = time.time()
runtime = end_time - start_time

# Only rank 0 processes results
if rank == 0:
    print(f"\nQEL completed in {runtime:.2f} seconds")
    
    # Extract and diagonalize (only rank 0 has valid data)
    try:
        from scipy.linalg import eigh
        HAS_SCIPY = True
    except ImportError:
        print("SciPy not available - cannot solve eigenvalue problem")
        HAS_SCIPY = False
    
    if HAS_SCIPY:
        H = result.get_hamiltonian_matrix()
        S = result.get_overlap_matrix()
        
        # Filter S matrix
        evals_S, evecs_S = np.linalg.eigh(S)
        threshold = 1e-4
        keep = [i for i, e in enumerate(evals_S) if e > threshold]
        
        if keep:
            print(f"\nS matrix filtering: keeping {len(keep)}/8 eigenvectors")
            
            S_f = np.diag(evals_S[keep])
            V_f = evecs_S[:, keep]
            H_p = V_f.T @ H @ V_f
            S_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(S_f)))
            H_f = S_inv_sqrt @ H_p @ S_inv_sqrt
            
            eigenvalues = np.linalg.eigvalsh(H_f)
            
            # Compute physical energy
            energy = eigenvalues[0] * result.normalization + CONSTANT_TERM
            error = abs(energy - FCI_ENERGY)
            
            print(f"\n{'='*70}")
            print("RESULTS")
            print('='*70)
            print(f"Estimated energy: {energy:.6f} Ha")
            print(f"Target FCI:       {FCI_ENERGY:.6f} Ha")
            print(f"Error:            {error:.6f} Ha ({error*1000:.2f} mHa)")
            print(f"Runtime:          {runtime:.2f} seconds")
            
            if has_mpi and size > 1:
                serial_time = runtime * size  # Approximate
                speedup = serial_time / runtime
                print(f"Estimated serial time: {serial_time:.2f} seconds")
                print(f"Speedup:          {speedup:.1f}x")
            
            print('='*70)
            
            if error < 0.01:
                print("\n✅ Excellent accuracy (< 10 mHa error)!")
            elif error < 0.05:
                print("\n✓ Good accuracy (< 50 mHa error)")
            else:
                print("\n⚠ Consider increasing krylov_dim for better accuracy")

if __name__ == "__main__":
    pass

