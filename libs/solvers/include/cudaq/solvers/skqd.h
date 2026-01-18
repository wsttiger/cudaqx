/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/spin_op.h"
#include "cudaq/platform/quantum_platform.h"
#include "cuda-qx/core/heterogeneous_map.h"
#include <vector>
#include <cstdint>
#include <string>
#include <utility>

using namespace cudaqx;

// Define CUDA host/device attributes for cross-platform compatibility
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

namespace cudaq::solvers {

/// @brief Result structure for SKQD computation
struct skqd_result {
  /// @brief Ground state energy (lowest eigenvalue)
  double ground_state_energy = 0.0;
  
  /// @brief All computed eigenvalues
  std::vector<double> eigenvalues;
  
  /// @brief Size of the constructed subspace basis
  std::size_t basis_size = 0;
  
  /// @brief Number of non-zero matrix elements
  std::size_t nnz = 0;
  
  /// @brief Total sampling time (seconds)
  double sampling_time = 0.0;
  
  /// @brief Matrix construction time (seconds)
  double matrix_construction_time = 0.0;
  
  /// @brief Diagonalization time (seconds)
  double diagonalization_time = 0.0;
  
  /// @brief Conversion operator to double for convenience
  operator double() const { return ground_state_energy; }
};

/// @brief 128-bit bitstring representation for quantum states
/// Supports up to 128 qubits
struct alignas(16) bit_string_128 {
  uint64_t low;   ///< Lower 64 bits
  uint64_t high;  ///< Upper 64 bits
  
  /// @brief Default constructor
  __host__ __device__ bit_string_128() : low(0), high(0) {}
  
  /// @brief Constructor from two 64-bit integers
  __host__ __device__ bit_string_128(uint64_t l, uint64_t h) : low(l), high(h) {}
  
  /// @brief Comparison operator for sorting
  __host__ __device__ bool operator<(const bit_string_128& other) const {
    if (high != other.high) return high < other.high;
    return low < other.low;
  }
  
  /// @brief Equality operator
  __host__ __device__ bool operator==(const bit_string_128& other) const {
    return high == other.high && low == other.low;
  }
  
  /// @brief Inequality operator
  __host__ __device__ bool operator!=(const bit_string_128& other) const {
    return !(*this == other);
  }
  
  /// @brief Get bit at position i
  __host__ __device__ bool get_bit(int i) const {
    if (i < 64) return (low >> i) & 1;
    else return (high >> (i - 64)) & 1;
  }
  
  /// @brief Set bit at position i
  __host__ __device__ void set_bit(int i, bool val) {
    if (i < 64) {
      if (val) low |= (1ULL << i);
      else low &= ~(1ULL << i);
    } else {
      if (val) high |= (1ULL << (i - 64));
      else high &= ~(1ULL << (i - 64));
    }
  }
  
  /// @brief Flip bit at position i
  __host__ __device__ void flip_bit(int i) {
    if (i < 64) low ^= (1ULL << i);
    else high ^= (1ULL << (i - 64));
  }
};

/// @brief GPU-friendly representation of a Pauli Hamiltonian
struct gpu_pauli_hamiltonian {
  /// @brief Number of terms in the Hamiltonian
  int num_terms;
  
  /// @brief Number of qubits
  int num_qubits;
  
  /// @brief Coefficients for each term (size: num_terms)
  double* coeffs;
  
  /// @brief Flattened Pauli operations [code, index, code, index, ...]
  /// Code: 0=I, 1=X, 2=Y, 3=Z
  int* flattened_ops;
  
  /// @brief Number of operations per term (size: num_terms)
  int* ops_per_term;
  
  /// @brief Offset to the start of operations for each term (size: num_terms)
  int* ops_offsets;
};

/// @brief Sample-based Krylov Quantum Diagonalization algorithm
/// 
/// This function implements a hybrid quantum-classical algorithm for computing
/// ground state energies of large quantum systems (80+ qubits) by combining
/// quantum subspace sampling with GPU-accelerated classical post-processing.
///
/// The algorithm generates a Krylov subspace through time evolution, samples
/// quantum states, and constructs a reduced Hamiltonian matrix in the sampled
/// subspace for efficient diagonalization on GPU.
///
/// @param hamiltonian The Hamiltonian to diagonalize
/// @param options Configuration options. Supported keys:
///  - "krylov_dim" (int): Number of Krylov subspace time steps [default: 15]
///  - "dt" (double): Time step size for time evolution [default: 0.1]
///  - "shots" (int): Number of measurement samples per time step [default: 10000]
///  - "num_eigenvalues" (int): Number of lowest eigenvalues to compute [default: 1]
///  - "trotter_order" (int): Trotter order for time evolution (1 or 2) [default: 1]
///     Order 1: exp(-iHt) ≈ ∏ exp(-i c_k P_k t) - O(t²) error
///     Order 2: Symmetric Suzuki formula - O(t³) error, more accurate
///  - "max_basis_size" (int): Maximum dimension of subspace basis (0 = unlimited) [default: 0]
///  - "verbose" (int): Verbosity level (0 = quiet, 1 = normal, 2 = debug) [default: 0]
/// @return Result structure containing ground state energy and additional information
///
/// @throws std::runtime_error if Hamiltonian has zero qubits or exceeds 128 qubits
///
/// Example:
/// @code
///   cudaq::spin_op h = /* construct Hamiltonian */;
///   heterogeneous_map options;
///   options.insert("krylov_dim", 20);
///   options.insert("shots", 5000);
///   options.insert("trotter_order", 2);
///   options.insert("verbose", 1);
///   auto result = sample_based_krylov(h, options);
///   printf("Ground state energy: %.12f\n", result.ground_state_energy);
/// @endcode
skqd_result sample_based_krylov(const cudaq::spin_op& hamiltonian,
                                heterogeneous_map options = heterogeneous_map());

/// @brief Build the SKQD subspace Hamiltonian matrix and return it to the caller
///
/// This function runs the SKQD sampling phase and constructs the projected
/// Hamiltonian matrix in the sampled subspace. The matrix is returned in dense
/// row-major format along with the basis ordering used for rows/columns.
///
/// @param hamiltonian The Hamiltonian to project
/// @param options Configuration options. Supported keys:
///  - "krylov_dim" (int): Number of Krylov subspace time steps [default: 15]
///  - "dt" (double): Time step size for time evolution [default: 0.1]
///  - "shots" (int): Number of measurement samples per time step [default: 10000]
///  - "trotter_order" (int): Trotter order for time evolution (1 or 2) [default: 1]
///  - "max_basis_size" (int): Maximum dimension of subspace basis (0 = unlimited) [default: 0]
///  - "verbose" (int): Verbosity level (0 = quiet, 1 = normal, 2 = debug) [default: 0]
///  - "n_electrons" (int): Number of electrons for Hartree-Fock initialization [default: 0]
///  - "filter_particles" (int): Keep only bitstrings with this particle count [-1 = off]
/// @return Pair of (dense matrix in row-major, basis strings)
std::pair<std::vector<double>, std::vector<std::string>>
sample_based_krylov_matrix(const cudaq::spin_op& hamiltonian,
                           heterogeneous_map options = heterogeneous_map());

} // namespace cudaq::solvers
