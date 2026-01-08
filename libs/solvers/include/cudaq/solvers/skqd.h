/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/spin_op.h"
#include <vector>
#include <cstdint>

// Define CUDA host/device attributes for cross-platform compatibility
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

namespace cudaq::solvers {

/// @brief Configuration parameters for Sample-based Krylov Quantum Diagonalization
struct skqd_config {
  /// @brief Number of Krylov subspace time steps
  int krylov_dim = 15;
  
  /// @brief Time step size for time evolution
  double dt = 0.1;
  
  /// @brief Number of measurement samples per time step
  int shots = 10000;
  
  /// @brief Number of lowest eigenvalues to compute
  int num_eigenvalues = 1;
  
  /// @brief Trotter order for time evolution approximation (1 or 2)
  /// Order 1: exp(-iHt) ≈ ∏ exp(-i c_k P_k t) - O(t²) error
  /// Order 2: Symmetric Suzuki formula - O(t³) error, more accurate
  int trotter_order = 1;
  
  /// @brief Maximum dimension of the subspace basis (0 = unlimited)
  int max_basis_size = 0;
  
  /// @brief Verbosity level (0 = quiet, 1 = normal, 2 = debug)
  int verbose = 0;
};

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
struct alignas(16) Bitstring128 {
  uint64_t low;   ///< Lower 64 bits
  uint64_t high;  ///< Upper 64 bits
  
  /// @brief Default constructor
  __host__ __device__ Bitstring128() : low(0), high(0) {}
  
  /// @brief Constructor from two 64-bit integers
  __host__ __device__ Bitstring128(uint64_t l, uint64_t h) : low(l), high(h) {}
  
  /// @brief Comparison operator for sorting
  __host__ __device__ bool operator<(const Bitstring128& other) const {
    if (high != other.high) return high < other.high;
    return low < other.low;
  }
  
  /// @brief Equality operator
  __host__ __device__ bool operator==(const Bitstring128& other) const {
    return high == other.high && low == other.low;
  }
  
  /// @brief Inequality operator
  __host__ __device__ bool operator!=(const Bitstring128& other) const {
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
struct GpuPauliHamiltonian {
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

/// @brief Sample-based Krylov Quantum Diagonalization solver
/// 
/// This class implements a hybrid quantum-classical algorithm for computing
/// ground state energies of large quantum systems (80+ qubits) by combining
/// quantum subspace sampling with GPU-accelerated classical post-processing.
class SampleBasedKrylov {
public:
  /// @brief Construct a SampleBasedKrylov solver
  /// @param hamiltonian The Hamiltonian to diagonalize
  SampleBasedKrylov(const cudaq::spin_op& hamiltonian);
  
  /// @brief Destructor
  ~SampleBasedKrylov();
  
  /// @brief Solve for the ground state energy
  /// @param config Configuration parameters for the algorithm
  /// @return Result structure containing the ground state energy and additional information
  skqd_result solve(const skqd_config& config = skqd_config());
  
  /// @brief Get multiple eigenvalues
  /// @param k Number of lowest eigenvalues to return
  /// @return Vector of eigenvalues
  std::vector<double> get_eigenvalues(int k);
  
  /// @brief Get the last computed result
  /// @return The result from the most recent solve() call
  const skqd_result& get_last_result() const { return last_result_; }

private:
  /// @brief The Hamiltonian to diagonalize
  cudaq::spin_op hamiltonian_;
  
  /// @brief Number of qubits in the system
  std::size_t num_qubits_;
  
  /// @brief Last computed result
  skqd_result last_result_;
  
  /// @brief Build the subspace Hamiltonian matrix on the GPU
  /// @param basis Unique basis bitstrings
  /// @param num_basis Size of the basis
  /// @param rows Output: row indices (COO format)
  /// @param cols Output: column indices (COO format) 
  /// @param vals Output: matrix values (COO format)
  /// @return Number of non-zero elements
  std::size_t build_subspace_matrix_gpu(
      const std::vector<Bitstring128>& basis,
      std::vector<int>& rows,
      std::vector<int>& cols,
      std::vector<double>& vals);
  
  /// @brief Convert COO format to CSR format
  /// @param num_basis Size of the basis
  /// @param rows Row indices (COO format)
  /// @param cols Column indices (COO format)
  /// @param vals Matrix values (COO format)
  /// @param csr_row_ptr Output: CSR row pointers
  /// @param csr_col_ind Output: CSR column indices
  /// @param csr_vals Output: CSR values
  void convert_coo_to_csr(
      std::size_t num_basis,
      const std::vector<int>& rows,
      const std::vector<int>& cols,
      const std::vector<double>& vals,
      std::vector<int>& csr_row_ptr,
      std::vector<int>& csr_col_ind,
      std::vector<double>& csr_vals);
  
  /// @brief Compute eigenvalues using cuSOLVER
  /// @param num_basis Size of the matrix
  /// @param csr_row_ptr CSR row pointers
  /// @param csr_col_ind CSR column indices
  /// @param csr_vals CSR values
  /// @param num_eigenvalues Number of eigenvalues to compute
  /// @return Vector of eigenvalues
  std::vector<double> compute_eigenvalues_gpu(
      std::size_t num_basis,
      const std::vector<int>& csr_row_ptr,
      const std::vector<int>& csr_col_ind,
      const std::vector<double>& csr_vals,
      int num_eigenvalues);
  
  /// @brief Compute eigenvalues using dense cuSOLVER routine
  /// @param num_basis Size of the matrix
  /// @param csr_row_ptr CSR row pointers
  /// @param csr_col_ind CSR column indices
  /// @param csr_vals CSR values
  /// @param num_eigenvalues Number of eigenvalues to compute
  /// @return Vector of eigenvalues
  std::vector<double> compute_eigenvalues_dense(
      std::size_t num_basis,
      const std::vector<int>& csr_row_ptr,
      const std::vector<int>& csr_col_ind,
      const std::vector<double>& csr_vals,
      int num_eigenvalues);
  
  /// @brief Convert cudaq::spin_op to GpuPauliHamiltonian
  /// @return GPU-friendly Hamiltonian representation
  GpuPauliHamiltonian convert_to_gpu_hamiltonian();
  
  /// @brief Free GPU memory for Hamiltonian
  /// @param gpu_ham GPU Hamiltonian to free
  void free_gpu_hamiltonian(GpuPauliHamiltonian& gpu_ham);
};

} // namespace cudaq::solvers
