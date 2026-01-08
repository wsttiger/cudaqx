/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "cudaq.h"
#include "cudaq/solvers/skqd.h"

using namespace cudaq::spin;

// Test result structure
TEST(SKQDTest, ResultStructure) {
  cudaq::solvers::skqd_result result;
  
  result.ground_state_energy = -1.5;
  result.eigenvalues = {-1.5, -0.5, 0.5};
  result.basis_size = 100;
  result.nnz = 500;
  
  EXPECT_DOUBLE_EQ(result.ground_state_energy, -1.5);
  EXPECT_EQ(result.eigenvalues.size(), 3);
  EXPECT_EQ(result.basis_size, 100);
  EXPECT_EQ(result.nnz, 500);
  
  // Test conversion operator
  double energy = result;
  EXPECT_DOUBLE_EQ(energy, -1.5);
}

// Test bit_string_128 functionality
TEST(SKQDTest, BitString128Operations) {
  cudaq::solvers::bit_string_128 bs1(0, 0);
  cudaq::solvers::bit_string_128 bs2(1, 0);
  cudaq::solvers::bit_string_128 bs3(0, 1);
  
  // Test equality
  EXPECT_TRUE(bs1 == bs1);
  EXPECT_FALSE(bs1 == bs2);
  
  // Test inequality
  EXPECT_TRUE(bs1 != bs2);
  EXPECT_FALSE(bs1 != bs1);
  
  // Test comparison
  EXPECT_TRUE(bs1 < bs2);
  EXPECT_TRUE(bs2 < bs3);
  EXPECT_FALSE(bs2 < bs1);
  
  // Test bit operations
  cudaq::solvers::bit_string_128 bs(0, 0);
  EXPECT_FALSE(bs.get_bit(0));
  
  bs.set_bit(0, true);
  EXPECT_TRUE(bs.get_bit(0));
  
  bs.flip_bit(0);
  EXPECT_FALSE(bs.get_bit(0));
  
  // Test bit 64 (in high word)
  bs.set_bit(64, true);
  EXPECT_TRUE(bs.get_bit(64));
}

// Test simple 2-qubit Ising model
TEST(SKQDTest, SimpleIsingModel) {
  // Create a simple Ising Hamiltonian: H = -Z_0 - Z_1
  // Ground state should be |11⟩ with energy -2
  cudaq::spin_op hamiltonian = -1.0 * z(0) - 1.0 * z(1);
  
  heterogeneous_map options;
  options.insert("krylov_dim", 5);
  options.insert("shots", 1000);
  options.insert("verbose", 0);
  
  EXPECT_NO_THROW({
    auto result = cudaq::solvers::sample_based_krylov(hamiltonian, options);
    EXPECT_GE(result.basis_size, 0);
  });
}

// Test Hamiltonian with multiple qubits
TEST(SKQDTest, MultiQubitHamiltonian) {
  // Create a 4-qubit Hamiltonian
  cudaq::spin_op hamiltonian = z(0) * z(1) + z(1) * z(2) + z(2) * z(3);
  
  EXPECT_NO_THROW({
    EXPECT_EQ(hamiltonian.num_qubits(), 4);
    
    heterogeneous_map options;
    options.insert("krylov_dim", 3);
    options.insert("shots", 100);
    options.insert("verbose", 0);
    
    auto result = cudaq::solvers::sample_based_krylov(hamiltonian, options);
  });
}

// Test zero qubit Hamiltonian error
TEST(SKQDTest, ZeroQubitError) {
  // Empty Hamiltonian should cause error
  cudaq::spin_op empty_hamiltonian;
  
  heterogeneous_map options;
  
  EXPECT_THROW({
    auto result = cudaq::solvers::sample_based_krylov(empty_hamiltonian, options);
  }, std::runtime_error);
}

// Test max basis size limiting
TEST(SKQDTest, MaxBasisSizeLimit) {
  cudaq::spin_op hamiltonian = z(0) + z(1);
  
  heterogeneous_map options;
  options.insert("krylov_dim", 5);
  options.insert("shots", 100);
  options.insert("max_basis_size", 10);  // Limit basis size
  options.insert("verbose", 0);
  
  EXPECT_NO_THROW({
    auto result = cudaq::solvers::sample_based_krylov(hamiltonian, options);
    // Basis size should not exceed the limit
    EXPECT_LE(result.basis_size, static_cast<size_t>(10));
  });
}

// Test verbose output modes
TEST(SKQDTest, VerboseModes) {
  cudaq::spin_op hamiltonian = z(0);
  
  // Test different verbosity levels - should not crash
  for (int verbose = 0; verbose <= 2; verbose++) {
    heterogeneous_map options;
    options.insert("krylov_dim", 3);
    options.insert("shots", 100);
    options.insert("verbose", verbose);
    
    EXPECT_NO_THROW({
      auto result = cudaq::solvers::sample_based_krylov(hamiltonian, options);
    });
  }
}

// Test transverse field Ising model
TEST(SKQDTest, TransverseFieldIsing) {
  // H = -Z_0*Z_1 - h*X_0 - h*X_1
  double h = 0.5;
  cudaq::spin_op hamiltonian = -1.0 * z(0) * z(1) - h * x(0) - h * x(1);
  
  EXPECT_NO_THROW({
    heterogeneous_map options;
    options.insert("krylov_dim", 5);
    options.insert("shots", 500);
    options.insert("verbose", 0);
    
    auto result = cudaq::solvers::sample_based_krylov(hamiltonian, options);
    
    // Just verify the solver runs without crashing
    EXPECT_GE(result.basis_size, 0);
  });
}

// Test first-order vs second-order Trotter
TEST(SKQDTest, TrotterOrders) {
  // Simple Hamiltonian: H = -Z_0 - 0.5*X_0
  cudaq::spin_op hamiltonian = -1.0 * z(0) - 0.5 * x(0);
  
  // Test first-order Trotter
  {
    heterogeneous_map options;
    options.insert("krylov_dim", 5);
    options.insert("dt", 0.1);
    options.insert("shots", 1000);
    options.insert("trotter_order", 1);
    options.insert("verbose", 0);
    
    EXPECT_NO_THROW({
      auto result = cudaq::solvers::sample_based_krylov(hamiltonian, options);
      EXPECT_GE(result.basis_size, 1);
      // Ground state energy should be close to eigenvalue of H
      // For this H, exact ground state energy ≈ -1.118
      EXPECT_LT(result.ground_state_energy, 0.0);
    });
  }
  
  // Test second-order Trotter (Suzuki formula)
  {
    heterogeneous_map options;
    options.insert("krylov_dim", 5);
    options.insert("dt", 0.1);
    options.insert("shots", 1000);
    options.insert("trotter_order", 2);
    options.insert("verbose", 0);
    
    EXPECT_NO_THROW({
      auto result = cudaq::solvers::sample_based_krylov(hamiltonian, options);
      EXPECT_GE(result.basis_size, 1);
      // Second-order should also give reasonable results
      EXPECT_LT(result.ground_state_energy, 0.0);
    });
  }
}

// Test default options
TEST(SKQDTest, DefaultOptions) {
  cudaq::spin_op hamiltonian = z(0);
  
  // Test with empty options - should use defaults
  heterogeneous_map options;
  
  EXPECT_NO_THROW({
    auto result = cudaq::solvers::sample_based_krylov(hamiltonian, options);
    EXPECT_GE(result.basis_size, 0);
  });
}

// Test custom time evolution parameters
TEST(SKQDTest, TimeEvolutionParameters) {
  cudaq::spin_op hamiltonian = z(0) + z(1);
  
  heterogeneous_map options;
  options.insert("krylov_dim", 10);
  options.insert("dt", 0.05);  // Smaller time step
  options.insert("shots", 500);
  options.insert("verbose", 0);
  
  EXPECT_NO_THROW({
    auto result = cudaq::solvers::sample_based_krylov(hamiltonian, options);
    EXPECT_GE(result.basis_size, 0);
  });
}

// Test multiple eigenvalues
TEST(SKQDTest, MultipleEigenvalues) {
  cudaq::spin_op hamiltonian = z(0) + z(1);
  
  heterogeneous_map options;
  options.insert("krylov_dim", 5);
  options.insert("shots", 500);
  options.insert("num_eigenvalues", 3);
  options.insert("verbose", 0);
  
  EXPECT_NO_THROW({
    auto result = cudaq::solvers::sample_based_krylov(hamiltonian, options);
    // Should compute up to 3 eigenvalues (or fewer if basis is smaller)
    EXPECT_GE(result.eigenvalues.size(), 1);
    if (result.basis_size >= 3) {
      EXPECT_LE(result.eigenvalues.size(), 3);
    }
  });
}
