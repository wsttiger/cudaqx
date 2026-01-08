/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/skqd.h"
#include "cudaq.h"
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <map>
#include <cuda_runtime.h>

// Try to include cuSOLVER if available
#ifdef __has_include
#  if __has_include(<cusolverDn.h>)
#    include <cusolverDn.h>
#    define HAVE_CUSOLVER 1
#  endif
#  if __has_include(<cusparse.h>)
#    include <cusparse.h>
#    define HAVE_CUSPARSE 1
#  endif
#endif

// Forward declarations for CUDA kernel launchers
extern "C" {
void launch_construct_matrix_kernel(
    const cudaq::solvers::Bitstring128* d_basis,
    int num_basis,
    const cudaq::solvers::GpuPauliHamiltonian& d_gpu_ham,
    int* d_rows,
    int* d_cols,
    double* d_vals,
    int* d_nnz_count,
    cudaStream_t stream);

void launch_count_nnz_per_row_kernel(
    const cudaq::solvers::Bitstring128* d_basis,
    int num_basis,
    const cudaq::solvers::GpuPauliHamiltonian& d_gpu_ham,
    int* d_nnz_per_row,
    cudaStream_t stream);
}

namespace cudaq::solvers {

namespace {

// Helper function to convert bitstring to Bitstring128
Bitstring128 string_to_bitstring128(const std::string& bitstring) {
  Bitstring128 result;
  for (size_t i = 0; i < bitstring.length() && i < 128; i++) {
    if (bitstring[bitstring.length() - 1 - i] == '1') {
      result.set_bit(i, true);
    }
  }
  return result;
}

// Helper struct to allow Bitstring128 as hash key
struct Bitstring128Hash {
  std::size_t operator()(const Bitstring128& bs) const {
    return std::hash<uint64_t>()(bs.low) ^ (std::hash<uint64_t>()(bs.high) << 1);
  }
};

} // anonymous namespace

SampleBasedKrylov::SampleBasedKrylov(const cudaq::spin_op& hamiltonian)
    : hamiltonian_(hamiltonian) {
  // Extract number of qubits from the Hamiltonian
  num_qubits_ = hamiltonian_.num_qubits();
  
  if (num_qubits_ == 0) {
    throw std::runtime_error("[SKQD] Hamiltonian has zero qubits");
  }
  
  if (num_qubits_ > 128) {
    throw std::runtime_error("[SKQD] Currently supports up to 128 qubits");
  }
}

SampleBasedKrylov::~SampleBasedKrylov() {
  // Cleanup if needed
}

skqd_result SampleBasedKrylov::solve(const skqd_config& config) {
  auto solve_start = std::chrono::high_resolution_clock::now();
  
  skqd_result result;
  
  // Phase 1: Quantum Sampling
  if (config.verbose > 0) {
    std::cout << "[SKQD] Starting quantum sampling phase..." << std::endl;
    std::cout << "[SKQD] System size: " << num_qubits_ << " qubits" << std::endl;
    std::cout << "[SKQD] Krylov dimension: " << config.krylov_dim << std::endl;
    std::cout << "[SKQD] Shots per step: " << config.shots << std::endl;
  }
  
  auto sampling_start = std::chrono::high_resolution_clock::now();
  
  // Collect all samples across all time steps
  std::unordered_set<Bitstring128, Bitstring128Hash> unique_samples_set;
  
  // Create initial state |+...+> (equal superposition)
  auto initial_state_kernel = [this]() __qpu__ {
    cudaq::qvector q(num_qubits_);
    // Apply Hadamard to all qubits
    for (size_t i = 0; i < num_qubits_; i++) {
      h(q[i]);
    }
    mz(q);
  };
  
  // Sample at t=0 (initial state)
  if (config.verbose > 1) {
    std::cout << "[SKQD] Sampling at t = 0.0" << std::endl;
  }
  
  auto counts_0 = cudaq::sample(config.shots, initial_state_kernel);
  for (auto& [bits, count] : counts_0) {
    unique_samples_set.insert(string_to_bitstring128(bits));
  }
  
  // Sample at each Krylov time step
  for (int k = 1; k < config.krylov_dim; k++) {
    double t = k * config.dt;
    
    if (config.verbose > 1) {
      std::cout << "[SKQD] Sampling at t = " << t << std::endl;
    }
    
    // Apply proper Trotterization for time evolution
    // exp(-iHt) ≈ ∏ exp(-i c_k P_k t) for each Hamiltonian term c_k P_k
    auto evolved_kernel = [this, t, &config]() __qpu__ {
      cudaq::qvector q(num_qubits_);
      
      // Prepare initial state |+...+> (equal superposition)
      for (size_t i = 0; i < num_qubits_; i++) {
        h(q[i]);
      }
      
      // Apply Trotterized time evolution
      if (config.trotter_order == 1) {
        // First-order Trotter: exp(-iHt) ≈ ∏_k exp(-i c_k P_k t)
        for (const auto& term : hamiltonian_) {
          double coeff = term.evaluate_coefficient().real();
          auto pauli_word = term.get_pauli_word(num_qubits_);
          
          // Apply exp(-i * coeff * t * P) where P is the Pauli string
          // exp_pauli applies exp(i * theta * P), so we use theta = -coeff * t
          exp_pauli(-coeff * t, q, pauli_word.c_str());
        }
      } else if (config.trotter_order == 2) {
        // Second-order Trotter (Suzuki formula):
        // exp(-iHt) ≈ ∏_k exp(-i c_k P_k t/2) * ∏_k exp(-i c_k P_k t/2) (reverse)
        // This is symmetric and has O(t³) error vs O(t²) for first-order
        
        // Store terms for reverse iteration
        std::vector<std::pair<double, std::string>> terms;
        for (const auto& term : hamiltonian_) {
          double coeff = term.evaluate_coefficient().real();
          auto pauli_word = term.get_pauli_word(num_qubits_);
          terms.emplace_back(coeff, pauli_word);
        }
        
        // Forward sweep with t/2
        for (const auto& [coeff, pauli_word] : terms) {
          exp_pauli(-coeff * t * 0.5, q, pauli_word.c_str());
        }
        
        // Backward sweep with t/2 (reverse order)
        for (auto it = terms.rbegin(); it != terms.rend(); ++it) {
          exp_pauli(-it->first * t * 0.5, q, it->second.c_str());
        }
      } else {
        // Fallback to first order for unsupported orders
        for (const auto& term : hamiltonian_) {
          double coeff = term.evaluate_coefficient().real();
          auto pauli_word = term.get_pauli_word(num_qubits_);
          exp_pauli(-coeff * t, q, pauli_word.c_str());
        }
      }
      
      mz(q);
    };
    
    try {
      auto counts = cudaq::sample(config.shots, evolved_kernel);
      
      // Add unique bitstrings to the set
      for (auto& [bits, count] : counts) {
        unique_samples_set.insert(string_to_bitstring128(bits));
        
        // Limit basis size if configured
        if (config.max_basis_size > 0 && 
            unique_samples_set.size() >= static_cast<size_t>(config.max_basis_size)) {
          break;
        }
      }
    } catch (const std::exception& e) {
      if (config.verbose > 0) {
        std::cerr << "[SKQD] Warning: Error during sampling at t=" << t 
                  << ": " << e.what() << std::endl;
      }
    }
    
    if (config.max_basis_size > 0 && 
        unique_samples_set.size() >= static_cast<size_t>(config.max_basis_size)) {
      if (config.verbose > 0) {
        std::cout << "[SKQD] Reached maximum basis size limit" << std::endl;
      }
      break;
    }
  }
  
  // Convert set to sorted vector for binary search
  std::vector<Bitstring128> basis(unique_samples_set.begin(), unique_samples_set.end());
  std::sort(basis.begin(), basis.end());
  
  auto sampling_end = std::chrono::high_resolution_clock::now();
  result.sampling_time = std::chrono::duration<double>(sampling_end - sampling_start).count();
  result.basis_size = basis.size();
  
  if (config.verbose > 0) {
    std::cout << "[SKQD] Sampling complete. Basis size: " << result.basis_size << std::endl;
    std::cout << "[SKQD] Sampling time: " << result.sampling_time << " seconds" << std::endl;
  }
  
  if (basis.size() == 0) {
    throw std::runtime_error("[SKQD] No samples collected");
  }
  
  // Phase 2: Matrix Construction on GPU
  if (config.verbose > 0) {
    std::cout << "[SKQD] Building subspace Hamiltonian matrix..." << std::endl;
  }
  
  auto matrix_start = std::chrono::high_resolution_clock::now();
  
  std::vector<int> rows, cols;
  std::vector<double> vals;
  
  result.nnz = build_subspace_matrix_gpu(basis, rows, cols, vals);
  
  auto matrix_end = std::chrono::high_resolution_clock::now();
  result.matrix_construction_time = std::chrono::duration<double>(matrix_end - matrix_start).count();
  
  if (config.verbose > 0) {
    std::cout << "[SKQD] Matrix construction complete. NNZ: " << result.nnz << std::endl;
    std::cout << "[SKQD] Matrix construction time: " << result.matrix_construction_time 
              << " seconds" << std::endl;
  }
  
  if (result.nnz == 0) {
    throw std::runtime_error("[SKQD] Matrix has no non-zero elements");
  }
  
  // Phase 3: Convert to CSR format
  std::vector<int> csr_row_ptr, csr_col_ind;
  std::vector<double> csr_vals;
  convert_coo_to_csr(basis.size(), rows, cols, vals, csr_row_ptr, csr_col_ind, csr_vals);
  
  // Phase 4: Diagonalization
  if (config.verbose > 0) {
    std::cout << "[SKQD] Computing eigenvalues..." << std::endl;
  }
  
  auto diag_start = std::chrono::high_resolution_clock::now();
  
  result.eigenvalues = compute_eigenvalues_gpu(
      basis.size(), csr_row_ptr, csr_col_ind, csr_vals, config.num_eigenvalues);
  
  auto diag_end = std::chrono::high_resolution_clock::now();
  result.diagonalization_time = std::chrono::duration<double>(diag_end - diag_start).count();
  
  if (result.eigenvalues.size() > 0) {
    result.ground_state_energy = result.eigenvalues[0];
  }
  
  if (config.verbose > 0) {
    std::cout << "[SKQD] Diagonalization complete." << std::endl;
    std::cout << "[SKQD] Ground state energy: " << result.ground_state_energy << std::endl;
    std::cout << "[SKQD] Diagonalization time: " << result.diagonalization_time 
              << " seconds" << std::endl;
    
    auto solve_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(solve_end - solve_start).count();
    std::cout << "[SKQD] Total time: " << total_time << " seconds" << std::endl;
  }
  
  last_result_ = result;
  return result;
}

std::vector<double> SampleBasedKrylov::get_eigenvalues(int k) {
  if (last_result_.eigenvalues.size() == 0) {
    throw std::runtime_error("[SKQD] No eigenvalues computed yet. Call solve() first.");
  }
  
  int num_to_return = std::min(k, static_cast<int>(last_result_.eigenvalues.size()));
  return std::vector<double>(
      last_result_.eigenvalues.begin(), 
      last_result_.eigenvalues.begin() + num_to_return);
}

GpuPauliHamiltonian SampleBasedKrylov::convert_to_gpu_hamiltonian() {
  GpuPauliHamiltonian gpu_ham;
  
  // Count terms in the Hamiltonian
  std::vector<double> coeffs;
  std::vector<int> flattened_ops;
  std::vector<int> ops_per_term;
  std::vector<int> ops_offsets;
  
  int current_offset = 0;
  
  // Iterate through Hamiltonian terms
  for (const auto& term : hamiltonian_) {
    double coeff = term.evaluate_coefficient().real();
    coeffs.push_back(coeff);
    
    int num_ops = 0;
    
    // Get the Pauli word for this term (returns a string like "IXYZ")
    auto pauli_word = term.get_pauli_word(num_qubits_);
    
    // Extract Pauli operations from the word string
    for (std::size_t qubit_idx = 0; qubit_idx < pauli_word.size() && qubit_idx < num_qubits_; qubit_idx++) {
      char p_char = pauli_word[qubit_idx];
      int pauli_code = 0;
      switch (p_char) {
        case 'I': pauli_code = 0; break;
        case 'X': pauli_code = 1; break;
        case 'Y': pauli_code = 2; break;
        case 'Z': pauli_code = 3; break;
        default: pauli_code = 0; break;
      }
      
      if (pauli_code != 0) {  // Skip identity
        flattened_ops.push_back(pauli_code);
        flattened_ops.push_back(static_cast<int>(qubit_idx));
        num_ops++;
      }
    }
    
    ops_per_term.push_back(num_ops);
    ops_offsets.push_back(current_offset);
    current_offset += num_ops * 2;  // Each op has 2 integers
  }
  
  gpu_ham.num_terms = coeffs.size();
  gpu_ham.num_qubits = num_qubits_;
  
  // Allocate GPU memory
  cudaMalloc(&gpu_ham.coeffs, coeffs.size() * sizeof(double));
  cudaMalloc(&gpu_ham.flattened_ops, flattened_ops.size() * sizeof(int));
  cudaMalloc(&gpu_ham.ops_per_term, ops_per_term.size() * sizeof(int));
  cudaMalloc(&gpu_ham.ops_offsets, ops_offsets.size() * sizeof(int));
  
  // Copy data to GPU
  cudaMemcpy(gpu_ham.coeffs, coeffs.data(), 
             coeffs.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_ham.flattened_ops, flattened_ops.data(), 
             flattened_ops.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_ham.ops_per_term, ops_per_term.data(), 
             ops_per_term.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_ham.ops_offsets, ops_offsets.data(), 
             ops_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
  
  return gpu_ham;
}

void SampleBasedKrylov::free_gpu_hamiltonian(GpuPauliHamiltonian& gpu_ham) {
  cudaFree(gpu_ham.coeffs);
  cudaFree(gpu_ham.flattened_ops);
  cudaFree(gpu_ham.ops_per_term);
  cudaFree(gpu_ham.ops_offsets);
}

std::size_t SampleBasedKrylov::build_subspace_matrix_gpu(
    const std::vector<Bitstring128>& basis,
    std::vector<int>& rows,
    std::vector<int>& cols,
    std::vector<double>& vals) {
  
  int num_basis = basis.size();
  
  // Allocate device memory for basis
  Bitstring128* d_basis;
  cudaMalloc(&d_basis, num_basis * sizeof(Bitstring128));
  cudaMemcpy(d_basis, basis.data(), num_basis * sizeof(Bitstring128), 
             cudaMemcpyHostToDevice);
  
  // Convert Hamiltonian to GPU format
  GpuPauliHamiltonian gpu_ham = convert_to_gpu_hamiltonian();
  
  // Estimate maximum NNZ (very conservative: num_basis * num_terms)
  size_t max_nnz = static_cast<size_t>(num_basis) * gpu_ham.num_terms;
  
  // Allocate device memory for output
  int* d_rows;
  int* d_cols;
  double* d_vals;
  int* d_nnz_count;
  
  cudaMalloc(&d_rows, max_nnz * sizeof(int));
  cudaMalloc(&d_cols, max_nnz * sizeof(int));
  cudaMalloc(&d_vals, max_nnz * sizeof(double));
  cudaMalloc(&d_nnz_count, sizeof(int));
  
  // Initialize NNZ counter
  int zero = 0;
  cudaMemcpy(d_nnz_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
  
  // Launch kernel
  launch_construct_matrix_kernel(d_basis, num_basis, gpu_ham, 
                                 d_rows, d_cols, d_vals, d_nnz_count, 0);
  
  // Wait for completion
  cudaDeviceSynchronize();
  
  // Get NNZ count
  int nnz_count;
  cudaMemcpy(&nnz_count, d_nnz_count, sizeof(int), cudaMemcpyDeviceToHost);
  
  // Copy results back
  rows.resize(nnz_count);
  cols.resize(nnz_count);
  vals.resize(nnz_count);
  
  cudaMemcpy(rows.data(), d_rows, nnz_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(cols.data(), d_cols, nnz_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(vals.data(), d_vals, nnz_count * sizeof(double), cudaMemcpyDeviceToHost);
  
  // Free device memory
  cudaFree(d_basis);
  cudaFree(d_rows);
  cudaFree(d_cols);
  cudaFree(d_vals);
  cudaFree(d_nnz_count);
  free_gpu_hamiltonian(gpu_ham);
  
  return nnz_count;
}

void SampleBasedKrylov::convert_coo_to_csr(
    std::size_t num_basis,
    const std::vector<int>& rows,
    const std::vector<int>& cols,
    const std::vector<double>& vals,
    std::vector<int>& csr_row_ptr,
    std::vector<int>& csr_col_ind,
    std::vector<double>& csr_vals) {
  
  // Create a map to collect entries by row
  std::map<std::pair<int,int>, double> matrix_map;
  
  for (size_t i = 0; i < rows.size(); i++) {
    auto key = std::make_pair(rows[i], cols[i]);
    matrix_map[key] += vals[i];  // Accumulate if duplicate
  }
  
  // Build CSR format
  csr_row_ptr.resize(num_basis + 1, 0);
  csr_col_ind.clear();
  csr_vals.clear();
  
  int current_row = -1;
  int nnz_so_far = 0;
  
  for (const auto& [key, val] : matrix_map) {
    int row = key.first;
    int col = key.second;
    
    // Fill row pointers for any empty rows
    while (current_row < row) {
      current_row++;
      csr_row_ptr[current_row] = nnz_so_far;
    }
    
    csr_col_ind.push_back(col);
    csr_vals.push_back(val);
    nnz_so_far++;
  }
  
  // Fill remaining row pointers
  while (current_row < static_cast<int>(num_basis)) {
    current_row++;
    csr_row_ptr[current_row] = nnz_so_far;
  }
}

std::vector<double> SampleBasedKrylov::compute_eigenvalues_gpu(
    std::size_t num_basis,
    const std::vector<int>& csr_row_ptr,
    const std::vector<int>& csr_col_ind,
    const std::vector<double>& csr_vals,
    int num_eigenvalues) {
  
  // Use dense eigensolver for small problems (< 1000), sparse for larger
  const std::size_t DENSE_THRESHOLD = 1000;
  
  if (num_basis <= DENSE_THRESHOLD) {
    // Dense eigensolver using cuSOLVER
    return compute_eigenvalues_dense(num_basis, csr_row_ptr, csr_col_ind, 
                                      csr_vals, num_eigenvalues);
  } else {
    // For larger matrices, we'd use sparse eigensolvers
    // For MVP, fall back to dense (will be slow for large matrices)
    return compute_eigenvalues_dense(num_basis, csr_row_ptr, csr_col_ind, 
                                      csr_vals, num_eigenvalues);
  }
}

std::vector<double> SampleBasedKrylov::compute_eigenvalues_dense(
    std::size_t num_basis,
    const std::vector<int>& csr_row_ptr,
    const std::vector<int>& csr_col_ind,
    const std::vector<double>& csr_vals,
    int num_eigenvalues) {
  
#ifdef HAVE_CUSOLVER
  // Use cuSOLVER for accurate eigenvalue computation
  
  // Convert CSR to dense matrix (column-major for LAPACK/cuSOLVER)
  std::vector<double> dense_matrix(num_basis * num_basis, 0.0);
  
  for (size_t i = 0; i < num_basis; i++) {
    int row_start = csr_row_ptr[i];
    int row_end = csr_row_ptr[i + 1];
    
    for (int j = row_start; j < row_end; j++) {
      int col = csr_col_ind[j];
      double val = csr_vals[j];
      // Column-major: A[i,j] = A[j * num_basis + i]
      dense_matrix[col * num_basis + i] = val;
    }
  }
  
  // Allocate device memory
  double* d_A;
  double* d_W;  // Eigenvalues
  int* d_info;
  
  cudaMalloc(&d_A, num_basis * num_basis * sizeof(double));
  cudaMalloc(&d_W, num_basis * sizeof(double));
  cudaMalloc(&d_info, sizeof(int));
  
  // Copy matrix to device
  cudaMemcpy(d_A, dense_matrix.data(), num_basis * num_basis * sizeof(double),
             cudaMemcpyHostToDevice);
  
  // Create cuSOLVER handle
  cusolverDnHandle_t cusolverH = nullptr;
  cusolverDnCreate(&cusolverH);
  
  // Query working space
  int lwork = 0;
  cusolverDnDsyevd_bufferSize(
      cusolverH,
      CUSOLVER_EIG_MODE_NOVECTOR,  // Only compute eigenvalues
      CUBLAS_FILL_MODE_LOWER,
      num_basis,
      d_A,
      num_basis,
      d_W,
      &lwork);
  
  // Allocate workspace
  double* d_work;
  cudaMalloc(&d_work, lwork * sizeof(double));
  
  // Compute eigenvalues
  cusolverDnDsyevd(
      cusolverH,
      CUSOLVER_EIG_MODE_NOVECTOR,
      CUBLAS_FILL_MODE_LOWER,
      num_basis,
      d_A,
      num_basis,
      d_W,
      d_work,
      lwork,
      d_info);
  
  // Wait for completion
  cudaDeviceSynchronize();
  
  // Check for errors
  int info_host = 0;
  cudaMemcpy(&info_host, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  
  if (info_host != 0) {
    // Cleanup before throwing
    cudaFree(d_A);
    cudaFree(d_W);
    cudaFree(d_work);
    cudaFree(d_info);
    cusolverDnDestroy(cusolverH);
    
    throw std::runtime_error("[SKQD] cuSOLVER eigenvalue computation failed with error code " 
                             + std::to_string(info_host));
  }
  
  // Copy eigenvalues back to host
  std::vector<double> all_eigenvalues(num_basis);
  cudaMemcpy(all_eigenvalues.data(), d_W, num_basis * sizeof(double),
             cudaMemcpyDeviceToHost);
  
  // Cleanup
  cudaFree(d_A);
  cudaFree(d_W);
  cudaFree(d_work);
  cudaFree(d_info);
  cusolverDnDestroy(cusolverH);
  
  // cuSOLVER returns eigenvalues in ascending order
  // Return only the requested number of lowest eigenvalues
  int num_to_return = std::min(num_eigenvalues, static_cast<int>(num_basis));
  std::vector<double> eigenvalues(all_eigenvalues.begin(), 
                                  all_eigenvalues.begin() + num_to_return);
  
  return eigenvalues;
  
#else
  // Fallback: Simple diagonal extraction when cuSOLVER is not available
  // This is a placeholder that gives rough estimates
  std::vector<double> eigenvalues(num_eigenvalues, 0.0);
  
  if (num_basis > 0 && csr_vals.size() > 0) {
    // Extract diagonal elements as rough eigenvalue estimates
    double min_diag = 0.0;
    for (size_t i = 0; i < num_basis; i++) {
      int row_start = csr_row_ptr[i];
      int row_end = csr_row_ptr[i + 1];
      
      for (int j = row_start; j < row_end; j++) {
        int col = csr_col_ind[j];
        if (col == static_cast<int>(i)) {
          if (i == 0 || csr_vals[j] < min_diag) {
            min_diag = csr_vals[j];
          }
        }
      }
    }
    eigenvalues[0] = min_diag;
  }
  
  return eigenvalues;
#endif
}

} // namespace cudaq::solvers
