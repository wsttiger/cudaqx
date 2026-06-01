/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/quantum_exact_lanczos.h"
#include "cudaq/solvers/operators/qubitization.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace cudaq::solvers {

// ============================================================================
// QUANTUM KERNELS FOR QEL
// ============================================================================

/// @brief State preparation for odd moment measurement
struct qel_state_prep_kernel {
  void operator()(cudaq::qview<> anc, cudaq::qview<> sys,
                  const pauli_lcu &encoding, int k_half) const __qpu__ {
    // Apply k iterations of SELECT + REFLECT
    encoding.prepare(anc);
    for (int i = 0; i < k_half; ++i) {
      qubitization_walk{}(anc, sys, encoding);
    }
  }
};

/// @brief State preparation for even moment measurement
struct qel_measure_even_kernel {
  void operator()(cudaq::qview<> anc, cudaq::qview<> sys,
                  const pauli_lcu &encoding, int k_half) const __qpu__ {
    // Apply k iterations of SELECT + REFLECT
    encoding.prepare(anc);
    for (int i = 0; i < k_half; ++i) {
      qubitization_walk{}(anc, sys, encoding);
    }
    // Unprepare for measurement in |0⟩ basis
    encoding.unprepare(anc);
  }
};

// ============================================================================
// KRYLOV MATRIX CONSTRUCTION
// ============================================================================

/// @brief Build Krylov Hamiltonian and overlap matrices from moments
/// @details Constructs H and S matrices that can be diagonalized by the user
/// to extract eigenvalues: H|v⟩ = E·S|v⟩
std::pair<std::vector<double>, std::vector<double>>
build_krylov_matrices(const std::vector<double> &moments, int krylov_dim) {

  std::vector<double> H_mat(krylov_dim * krylov_dim);
  std::vector<double> S_mat(krylov_dim * krylov_dim);

  for (int i = 0; i < krylov_dim; ++i) {
    for (int j = 0; j < krylov_dim; ++j) {
      int idx = i * krylov_dim + j; // Row-major indexing

      S_mat[idx] = 0.5 * (moments[i + j] + moments[std::abs(i - j)]);
      H_mat[idx] =
          0.25 * (moments[i + j + 1] + moments[std::abs(i + j - 1)] +
                  moments[std::abs(i - j + 1)] + moments[std::abs(i - j - 1)]);
    }
  }

  return {H_mat, S_mat};
}

// ============================================================================
// MAIN QEL ALGORITHM
// ============================================================================

qel_result quantum_exact_lanczos(const pauli_lcu &encoding,
                                 std::size_t n_electrons,
                                 heterogeneous_map options) {

  // Extract options
  int krylov_dim = options.get("krylov_dim", 10);
  int shots = options.get("shots", -1);
  bool verbose = options.get("verbose", false);

  std::size_t n_anc = encoding.num_ancilla();
  std::size_t n_sys = encoding.num_system();
  double one_norm = encoding.normalization();
  double constant_term = encoding.constant_term();

  if (verbose) {
    std::cout << "\n=== Quantum Exact Lanczos ===" << std::endl;
    std::cout << "System qubits: " << n_sys << std::endl;
    std::cout << "Ancilla qubits: " << n_anc << std::endl;
    std::cout << "Electrons: " << n_electrons << std::endl;
    std::cout << "1-Norm (α): " << one_norm << std::endl;
    std::cout << "Constant term: " << constant_term << " Ha" << std::endl;
    std::cout << "Krylov dimension: " << krylov_dim << std::endl;
  }

  // Build observables
  auto R_op = build_qubitization_reflection_observable(n_anc);
  auto U_op = build_lcu_select_observable(encoding);

  // Collect moments
  std::vector<double> moments;
  moments.reserve(2 * krylov_dim);

  if (verbose) {
    std::cout << "\nCollecting moments..." << std::endl;
  }

  for (int k = 0; k < 2 * krylov_dim; ++k) {
    int m = k / 2;

    // Create quantum kernel for this moment
    auto measure_kernel = [&, m, n_anc, n_sys, n_electrons]() __qpu__ {
      cudaq::qvector<> anc(n_anc);
      cudaq::qvector<> sys(n_sys);

      // Initialize Hartree-Fock state
      for (std::size_t i = 0; i < n_electrons; ++i) {
        x(sys[i]);
      }

      if (k % 2 == 0) {
        // Even moment
        qel_measure_even_kernel{}(anc, sys, encoding, m);
      } else {
        // Odd moment
        qel_state_prep_kernel{}(anc, sys, encoding, m);
      }
    };

    // Measure
    cudaq::spin_op obs = (k % 2 == 0) ? R_op : U_op;
    auto result = cudaq::observe(shots, measure_kernel, obs);
    double moment = result.expectation();
    moments.push_back(moment);

    if (verbose) {
      std::cout << "  k=" << k << ": " << moment << std::endl;
    }
  }

  // Build Krylov matrices
  if (verbose) {
    std::cout << "\nBuilding Krylov matrices..." << std::endl;
  }

  auto [H_mat, S_mat] = build_krylov_matrices(moments, krylov_dim);

  if (verbose) {
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Hamiltonian matrix: " << krylov_dim << "×" << krylov_dim
              << std::endl;
    std::cout << "Overlap matrix: " << krylov_dim << "×" << krylov_dim
              << std::endl;
    std::cout << "Total moments collected: " << moments.size() << std::endl;
    std::cout << "\nTo extract eigenvalues, solve: H|v⟩ = E·S|v⟩" << std::endl;
    std::cout << "Then convert: E_physical = E_scaled * α + constant"
              << std::endl;
  }

  // Return result
  return qel_result{H_mat,         S_mat,    moments, krylov_dim,
                    constant_term, one_norm, n_anc,   n_sys};
}

qel_result quantum_exact_lanczos(const cudaq::spin_op &hamiltonian,
                                 std::size_t num_qubits,
                                 std::size_t n_electrons,
                                 heterogeneous_map options) {
  return quantum_exact_lanczos(pauli_lcu(hamiltonian, num_qubits), n_electrons,
                               options);
}

} // namespace cudaq::solvers
