/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/skqd.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

namespace cudaq::solvers {

// Binary search in sorted basis array
__device__ int binary_search_bitstring(const bit_string_128* basis, 
                                       int num_basis, 
                                       const bit_string_128& target) {
  int left = 0;
  int right = num_basis - 1;
  
  while (left <= right) {
    int mid = left + (right - left) / 2;
    
    if (basis[mid] == target) {
      return mid;
    }
    
    if (basis[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  
  return -1;  // Not found
}

// Apply a Pauli operation to a bitstring
__device__ bit_string_128 apply_pauli_term(const bit_string_128& state,
                                         const int* pauli_ops,
                                         int num_ops,
                                         double& phase) {
  bit_string_128 result = state;
  phase = 1.0;
  
  for (int i = 0; i < num_ops; i++) {
    int pauli_code = pauli_ops[2*i];      // 0=I, 1=X, 2=Y, 3=Z
    int qubit_idx = pauli_ops[2*i + 1];   // Qubit index
    
    switch (pauli_code) {
      case 0:  // Identity - no change
        break;
        
      case 1:  // Pauli X - flip bit
        result.flip_bit(qubit_idx);
        break;
        
      case 2:  // Pauli Y - flip bit and add phase
        {
          bool bit_val = result.get_bit(qubit_idx);
          result.flip_bit(qubit_idx);
          // Y = iXZ, so phase depends on initial bit value
          // Y|0> = i|1>, Y|1> = -i|0>
          phase *= bit_val ? -1.0 : 1.0;  // Simplified: track only real part
        }
        break;
        
      case 3:  // Pauli Z - add phase based on bit value
        {
          bool bit_val = result.get_bit(qubit_idx);
          // Z|0> = |0>, Z|1> = -|1>
          if (bit_val) phase *= -1.0;
        }
        break;
    }
  }
  
  return result;
}

// Main kernel for constructing the Hamiltonian matrix in the subspace
__global__ void construct_matrix_kernel(
    const bit_string_128* basis,
    int num_basis,
    const gpu_pauli_hamiltonian gpu_ham,
    int* rows,
    int* cols,
    double* vals,
    int* nnz_count) {
  
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_basis) return;
  
  bit_string_128 my_state = basis[idx];
  
  // Iterate over all Hamiltonian terms
  for (int t = 0; t < gpu_ham.num_terms; t++) {
    double coeff = gpu_ham.coeffs[t];
    int ops_offset = gpu_ham.ops_offsets[t];
    int num_ops = gpu_ham.ops_per_term[t];
    
    // Apply this Pauli term to get the connected state
    double phase = 1.0;
    bit_string_128 connected_state = apply_pauli_term(
        my_state, 
        &gpu_ham.flattened_ops[ops_offset], 
        num_ops,
        phase);
    
    // Search for this state in the basis
    int target_idx = binary_search_bitstring(basis, num_basis, connected_state);
    
    if (target_idx != -1) {
      // Found a matrix element H[idx, target_idx] = coeff * phase
      int pos = atomicAdd(nnz_count, 1);
      rows[pos] = idx;
      cols[pos] = target_idx;
      vals[pos] = coeff * phase;
    }
  }
}

// Kernel for initial count of non-zero elements per row
__global__ void count_nnz_per_row_kernel(
    const bit_string_128* basis,
    int num_basis,
    const gpu_pauli_hamiltonian gpu_ham,
    int* nnz_per_row) {
  
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_basis) return;
  
  bit_string_128 my_state = basis[idx];
  int count = 0;
  
  // Iterate over all Hamiltonian terms
  for (int t = 0; t < gpu_ham.num_terms; t++) {
    int ops_offset = gpu_ham.ops_offsets[t];
    int num_ops = gpu_ham.ops_per_term[t];
    
    double phase = 1.0;
    bit_string_128 connected_state = apply_pauli_term(
        my_state, 
        &gpu_ham.flattened_ops[ops_offset], 
        num_ops,
        phase);
    
    int target_idx = binary_search_bitstring(basis, num_basis, connected_state);
    if (target_idx != -1) {
      count++;
    }
  }
  
  nnz_per_row[idx] = count;
}

// Helper function to launch the kernel from C++
extern "C" {

void launch_construct_matrix_kernel(
    const bit_string_128* d_basis,
    int num_basis,
    const gpu_pauli_hamiltonian& d_gpu_ham,
    int* d_rows,
    int* d_cols,
    double* d_vals,
    int* d_nnz_count,
    cudaStream_t stream) {
  
  int threads_per_block = 256;
  int num_blocks = (num_basis + threads_per_block - 1) / threads_per_block;
  
  construct_matrix_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      d_basis, num_basis, d_gpu_ham, d_rows, d_cols, d_vals, d_nnz_count);
}

void launch_count_nnz_per_row_kernel(
    const bit_string_128* d_basis,
    int num_basis,
    const gpu_pauli_hamiltonian& d_gpu_ham,
    int* d_nnz_per_row,
    cudaStream_t stream) {
  
  int threads_per_block = 256;
  int num_blocks = (num_basis + threads_per_block - 1) / threads_per_block;
  
  count_nnz_per_row_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      d_basis, num_basis, d_gpu_ham, d_nnz_per_row);
}

} // extern "C"

} // namespace cudaq::solvers
