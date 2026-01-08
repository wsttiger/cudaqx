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

// Test basic SKQD solver construction
TEST(SKQDTest, SolverConstruction) {
  // Create a simple 2-qubit Hamiltonian
  cudaq::spin_op hamiltonian = z(0) + z(1);
  
  // Construct solver - should not throw
  EXPECT_NO_THROW({
    cudaq::solvers::SampleBasedKrylov solver(hamiltonian);
  });
}

// Test SKQD configuration
TEST(SKQDTest, ConfigurationDefaults) {
  cudaq::solvers::skqd_config config;
  
  // Check default values
  EXPECT_EQ(config.krylov_dim, 15);
  EXPECT_DOUBLE_EQ(config.dt, 0.1);
  EXPECT_EQ(config.shots, 10000);
  EXPECT_EQ(config.num_eigenvalues, 1);
  EXPECT_EQ(config.trotter_order, 1);
  EXPECT_EQ(config.max_basis_size, 0);
  EXPECT_EQ(config.verbose, 0);
}

// Test SKQD configuration modification
TEST(SKQDTest, ConfigurationModification) {
  cudaq::solvers::skqd_config config;
  
  config.krylov_dim = 10;
  config.dt = 0.05;
  config.shots = 5000;
  config.verbose = 1;
  
  EXPECT_EQ(config.krylov_dim, 10);
  EXPECT_DOUBLE_EQ(config.dt, 0.05);
  EXPECT_EQ(config.shots, 5000);
  EXPECT_EQ(config.verbose, 1);
}

// Test simple 2-qubit Ising model
TEST(SKQDTest, SimpleIsingModel) {
  // Create a simple Ising Hamiltonian: H = -Z_0 - Z_1
  // Ground state should be |11⟩ with energy -2
  cudaq::spin_op hamiltonian = -1.0 * z(0) - 1.0 * z(1);
  
  cudaq::solvers::SampleBasedKrylov solver(hamiltonian);
  
  cudaq::solvers::skqd_config config;
  config.krylov_dim = 5;
  config.shots = 1000;
  config.verbose = 0;
  
  // Note: This is a placeholder test
  // Full test would require proper time evolution implementation
  // For now, just check that solve() doesn't crash
  EXPECT_NO_THROW({
    auto result = solver.solve(config);
    EXPECT_GE(result.basis_size, 0);
  });
}

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

// Test Bitstring128 functionality
TEST(SKQDTest, Bitstring128Operations) {
  cudaq::solvers::Bitstring128 bs1(0, 0);
  cudaq::solvers::Bitstring128 bs2(1, 0);
  cudaq::solvers::Bitstring128 bs3(0, 1);
  
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
  cudaq::solvers::Bitstring128 bs(0, 0);
  EXPECT_FALSE(bs.get_bit(0));
  
  bs.set_bit(0, true);
  EXPECT_TRUE(bs.get_bit(0));
  
  bs.flip_bit(0);
  EXPECT_FALSE(bs.get_bit(0));
  
  // Test bit 64 (in high word)
  bs.set_bit(64, true);
  EXPECT_TRUE(bs.get_bit(64));
}

// Test get_eigenvalues function
TEST(SKQDTest, GetEigenvalues) {
  cudaq::spin_op hamiltonian = z(0);
  cudaq::solvers::SampleBasedKrylov solver(hamiltonian);
  
  // Should throw before solve() is called
  EXPECT_THROW({
    solver.get_eigenvalues(1);
  }, std::runtime_error);
}

// Test Hamiltonian with multiple qubits
TEST(SKQDTest, MultiQubitHamiltonian) {
  // Create a 4-qubit Hamiltonian
  cudaq::spin_op hamiltonian = z(0) * z(1) + z(1) * z(2) + z(2) * z(3);
  
  EXPECT_NO_THROW({
    cudaq::solvers::SampleBasedKrylov solver(hamiltonian);
    EXPECT_EQ(hamiltonian.num_qubits(), 4);
  });
}

// Test zero qubit Hamiltonian error
TEST(SKQDTest, ZeroQubitError) {
  // Empty Hamiltonian should cause error
  cudaq::spin_op empty_hamiltonian;
  
  EXPECT_THROW({
    cudaq::solvers::SampleBasedKrylov solver(empty_hamiltonian);
  }, std::runtime_error);
}

// Test max basis size limiting
TEST(SKQDTest, MaxBasisSizeLimit) {
  cudaq::spin_op hamiltonian = z(0) + z(1);
  cudaq::solvers::SampleBasedKrylov solver(hamiltonian);
  
  cudaq::solvers::skqd_config config;
  config.krylov_dim = 5;
  config.shots = 100;
  config.max_basis_size = 10;  // Limit basis size
  config.verbose = 0;
  
  EXPECT_NO_THROW({
    auto result = solver.solve(config);
    // Basis size should not exceed the limit
    EXPECT_LE(result.basis_size, static_cast<size_t>(config.max_basis_size));
  });
}

// Test verbose output modes
TEST(SKQDTest, VerboseModes) {
  cudaq::spin_op hamiltonian = z(0);
  cudaq::solvers::SampleBasedKrylov solver(hamiltonian);
  
  // Test different verbosity levels - should not crash
  for (int verbose = 0; verbose <= 2; verbose++) {
    cudaq::solvers::skqd_config config;
    config.krylov_dim = 3;
    config.shots = 100;
    config.verbose = verbose;
    
    EXPECT_NO_THROW({
      auto result = solver.solve(config);
    });
  }
}

// Test transverse field Ising model
TEST(SKQDTest, TransverseFieldIsing) {
  // H = -Z_0*Z_1 - h*X_0 - h*X_1
  double h = 0.5;
  cudaq::spin_op hamiltonian = -1.0 * z(0) * z(1) - h * x(0) - h * x(1);
  
  EXPECT_NO_THROW({
    cudaq::solvers::SampleBasedKrylov solver(hamiltonian);
    
    cudaq::solvers::skqd_config config;
    config.krylov_dim = 5;
    config.shots = 500;
    config.verbose = 0;
    
    auto result = solver.solve(config);
    
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
    cudaq::solvers::SampleBasedKrylov solver(hamiltonian);
    cudaq::solvers::skqd_config config;
    config.krylov_dim = 5;
    config.dt = 0.1;
    config.shots = 1000;
    config.trotter_order = 1;
    config.verbose = 0;
    
    EXPECT_NO_THROW({
      auto result = solver.solve(config);
      EXPECT_GE(result.basis_size, 1);
      // Ground state energy should be close to eigenvalue of H
      // For this H, exact ground state energy ≈ -1.118
      EXPECT_LT(result.ground_state_energy, 0.0);
    });
  }
  
  // Test second-order Trotter (Suzuki formula)
  {
    cudaq::solvers::SampleBasedKrylov solver(hamiltonian);
    cudaq::solvers::skqd_config config;
    config.krylov_dim = 5;
    config.dt = 0.1;
    config.shots = 1000;
    config.trotter_order = 2;
    config.verbose = 0;
    
    EXPECT_NO_THROW({
      auto result = solver.solve(config);
      EXPECT_GE(result.basis_size, 1);
      // Second-order should also give reasonable results
      EXPECT_LT(result.ground_state_energy, 0.0);
    });
  }
}
