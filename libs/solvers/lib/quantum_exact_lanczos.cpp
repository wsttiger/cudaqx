/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/quantum_exact_lanczos.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

// MPI support via CUDA-Q's MPI interface
#ifdef CUDAQ_HAS_MPI
#include <cudaq/platform.h>  // CUDA-Q MPI interface
#include <mpi.h>              // Raw MPI for advanced collectives (Gatherv)
#define MPI_ENABLED 1
#else
#define MPI_ENABLED 0
#endif

namespace cudaq::solvers {

// ============================================================================
// QUANTUM KERNELS FOR QEL
// ============================================================================

/// @brief Reflection operator for amplitude amplification
/// @details Implements: PREPARE† → X → Multi-controlled-Z → X → PREPARE
struct qel_reflection_kernel {
  void operator()(cudaq::qview<> anc, const pauli_lcu &encoding) const __qpu__ {
    // PREPARE†
    encoding.unprepare(anc);
    
    // X on all ancilla
    for (std::size_t i = 0; i < anc.size(); ++i) {
      x(anc[i]);
    }
    
    // Multi-controlled Z (controlled by all but last, target is last)
    std::size_t n_anc = anc.size();
    if (n_anc == 1) {
      z(anc[0]);
    } else if (n_anc == 2) {
      z<cudaq::ctrl>(anc[0], anc[1]);
    } else if (n_anc == 3) {
      z<cudaq::ctrl>(anc[0], anc[1], anc[2]);
    } else if (n_anc == 4) {
      z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3]);
    } else if (n_anc == 5) {
      z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3], anc[4]);
    } else if (n_anc == 6) {
      z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3], anc[4], anc[5]);
    } else if (n_anc == 7) {
      z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3], anc[4], anc[5], anc[6]);
    } else if (n_anc == 8) {
      z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3], anc[4], anc[5], anc[6], anc[7]);
    } else if (n_anc == 9) {
      z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3], anc[4], anc[5], anc[6], anc[7], anc[8]);
    } else if (n_anc == 10) {
      z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3], anc[4], anc[5], anc[6], anc[7], anc[8], anc[9]);
    }
    
    // X on all ancilla (uncompute)
    for (std::size_t i = 0; i < anc.size(); ++i) {
      x(anc[i]);
    }
    
    // PREPARE
    encoding.prepare(anc);
  }
};

/// @brief State preparation for odd moment measurement
struct qel_state_prep_kernel {
  void operator()(cudaq::qview<> anc, cudaq::qview<> sys,
                  const pauli_lcu &encoding, int k_half) const __qpu__ {
    // Apply k iterations of SELECT + REFLECT
    encoding.prepare(anc);
    for (int i = 0; i < k_half; ++i) {
      encoding.select(anc, sys);
      qel_reflection_kernel{}(anc, encoding);
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
      encoding.select(anc, sys);
      qel_reflection_kernel{}(anc, encoding);
    }
    // Unprepare for measurement in |0⟩ basis
    encoding.unprepare(anc);
  }
};

// ============================================================================
// OBSERVABLE CONSTRUCTION
// ============================================================================

/// @brief Build projector |0⟩⟨0| on ancilla qubits
cudaq::spin_op build_ancilla_projector(std::size_t n_anc) {
  // P_0 = (I + Z) / 2 for each qubit
  cudaq::spin_op projector = 0.5 * (cudaq::spin::i(0) + cudaq::spin::z(0));
  for (std::size_t q = 1; q < n_anc; ++q) {
    projector *= 0.5 * (cudaq::spin::i(q) + cudaq::spin::z(q));
  }
  
  return projector;
}

/// @brief Build R observable for even moments: R = 2*P_0 - I
cudaq::spin_op build_R_observable(std::size_t n_anc) {
  auto P_zero = build_ancilla_projector(n_anc);
  return 2.0 * P_zero - cudaq::spin::i(0);
}

/// @brief Build U observable for odd moments
/// @details U = Σᵢ sign(cᵢ) * P_i ⊗ Pᵢ where P_i projects onto ancilla state |i⟩
cudaq::spin_op build_U_observable(const pauli_lcu &encoding) {
  std::size_t n_anc = encoding.num_ancilla();
  std::size_t n_sys = encoding.num_system();
  
  const auto &controls = encoding.get_term_controls();
  const auto &ops = encoding.get_term_ops();
  const auto &lengths = encoding.get_term_lengths();
  const auto &signs = encoding.get_term_signs();
  
  cudaq::spin_op U_op;
  bool first_term = true;
  
  int ctrl_ptr = 0;
  int ops_ptr = 0;
  
  for (std::size_t term_idx = 0; term_idx < lengths.size(); ++term_idx) {
    // Build ancilla projector for this term's index
    cudaq::spin_op anc_proj;
    bool first_anc = true;
    
    for (std::size_t b = 0; b < n_anc; ++b) {
      int bit_val = controls[ctrl_ptr++];
      cudaq::spin_op proj_bit = (bit_val == 0) 
        ? 0.5 * (cudaq::spin::i(b) + cudaq::spin::z(b))   // |0⟩⟨0|
        : 0.5 * (cudaq::spin::i(b) - cudaq::spin::z(b));  // |1⟩⟨1|
      
      if (first_anc) {
        anc_proj = proj_bit;
        first_anc = false;
      } else {
        anc_proj = anc_proj * proj_bit;
      }
    }
    
    // Build system Pauli operator
    cudaq::spin_op sys_pauli;
    bool first_pauli = true;
    
    int n_ops = lengths[term_idx];
    for (int k = 0; k < n_ops; ++k) {
      int code = ops[ops_ptr++];
      int qubit = ops[ops_ptr++] + n_anc; // Offset by ancilla qubits
      
      cudaq::spin_op pauli_op;
      if (code == 1) pauli_op = cudaq::spin::x(qubit);
      else if (code == 2) pauli_op = cudaq::spin::y(qubit);
      else if (code == 3) pauli_op = cudaq::spin::z(qubit);
      
      if (first_pauli) {
        sys_pauli = pauli_op;
        first_pauli = false;
      } else {
        sys_pauli = sys_pauli * pauli_op;
      }
    }
    
    // If no Pauli operators, use identity
    if (first_pauli) {
      sys_pauli = cudaq::spin::i(n_anc);
    }
    
    // Combine with sign
    double sign = signs[term_idx];
    cudaq::spin_op term = sign * anc_proj * sys_pauli;
    
    if (first_term) {
      U_op = term;
      first_term = false;
    } else {
      U_op = U_op + term;
    }
  }
  
  return U_op;
}

// ============================================================================
// KRYLOV MATRIX CONSTRUCTION
// ============================================================================

/// @brief Build Krylov Hamiltonian and overlap matrices from moments
/// @details Constructs H and S matrices that can be diagonalized by the user
/// to extract eigenvalues: H|v⟩ = E·S|v⟩
std::pair<std::vector<double>, std::vector<double>> build_krylov_matrices(
    const std::vector<double> &moments,
    int krylov_dim) {
  
  std::vector<double> H_mat(krylov_dim * krylov_dim);
  std::vector<double> S_mat(krylov_dim * krylov_dim);
  
  for (int i = 0; i < krylov_dim; ++i) {
    for (int j = 0; j < krylov_dim; ++j) {
      int idx = i * krylov_dim + j;  // Row-major indexing
      
      S_mat[idx] = 0.5 * (moments[i + j] + moments[std::abs(i - j)]);
      H_mat[idx] = 0.25 * (moments[i + j + 1] + moments[std::abs(i + j - 1)] +
                           moments[std::abs(i - j + 1)] + moments[std::abs(i - j - 1)]);
    }
  }
  
  return {H_mat, S_mat};
}

// ============================================================================
// MAIN QEL ALGORITHM
// ============================================================================

qel_result quantum_exact_lanczos(
    const cudaq::spin_op &hamiltonian,
    std::size_t num_qubits,
    std::size_t n_electrons,
    heterogeneous_map options) {
  
  // Extract options
  int krylov_dim = options.get("krylov_dim", 10);
  int shots = options.get("shots", -1);
  bool verbose = options.get("verbose", false);
  bool use_mpi = options.get("use_mpi", false);
  
  // MPI setup using CUDA-Q's MPI interface
  int mpi_rank = 0;
  int mpi_size = 1;
  
#if MPI_ENABLED
  if (use_mpi) {
    // Try to use CUDA-Q's MPI if available, otherwise fall back to raw MPI
    // CUDA-Q's is_initialized() may return false even with mpirun, so we
    // query raw MPI directly which works reliably
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // If only 1 rank, disable MPI (not actually parallel)
    if (mpi_size == 1) {
      if (verbose) {
        std::cerr << "Warning: use_mpi=True but only 1 MPI rank detected. Running serially." << std::endl;
      }
      use_mpi = false;
    }
  }
#else
  if (use_mpi && mpi_rank == 0) {
    std::cerr << "Warning: MPI requested but not available at compile time. Running serially." << std::endl;
    use_mpi = false;
  }
#endif
  
  // Create block encoding
  pauli_lcu encoding(hamiltonian, num_qubits);
  
  std::size_t n_anc = encoding.num_ancilla();
  std::size_t n_sys = encoding.num_system();
  double one_norm = encoding.normalization();
  
  // Extract constant term from Hamiltonian
  double constant_term = 0.0;
  for (const auto &term : hamiltonian) {
    auto word = term.get_pauli_word(num_qubits);
    // Check if this is an identity term (all characters are 'I')
    bool is_identity = (word.find_first_not_of('I') == std::string::npos);
    if (is_identity) {
      constant_term = term.evaluate_coefficient().real();
      break;
    }
  }
  
  if (verbose && mpi_rank == 0) {
    std::cout << "\n=== Quantum Exact Lanczos ===" << std::endl;
    std::cout << "System qubits: " << n_sys << std::endl;
    std::cout << "Ancilla qubits: " << n_anc << std::endl;
    std::cout << "Electrons: " << n_electrons << std::endl;
    std::cout << "1-Norm (α): " << one_norm << std::endl;
    std::cout << "Constant term: " << constant_term << " Ha" << std::endl;
    std::cout << "Krylov dimension: " << krylov_dim << std::endl;
    if (use_mpi) {
      std::cout << "MPI parallelization: " << mpi_size << " ranks" << std::endl;
    }
  }
  
  // Build observables
  auto R_op = build_R_observable(n_anc);
  auto U_op = build_U_observable(encoding);
  
  if (verbose && mpi_rank == 0) {
    std::cout << "Observable complexity:" << std::endl;
    std::cout << "  R_op (even moments): " << R_op.num_terms() << " terms" << std::endl;
    std::cout << "  U_op (odd moments):  " << U_op.num_terms() << " terms" << std::endl;
  }
  
  // Collect moments (MPI-parallelized if requested)
  int total_moments = 2 * krylov_dim;
  std::vector<double> moments(total_moments, 0.0);  // Pre-allocate on all ranks
  
  if (verbose && mpi_rank == 0) {
    std::cout << "\nCollecting moments..." << std::endl;
    if (use_mpi) {
      std::cout << "  (parallelized across " << mpi_size << " MPI ranks)" << std::endl;
    }
  }
  
  // Determine which moments this rank computes using cost-aware load balancing
  // Cost model: circuit_depth × observable_terms
  // This balances both circuit depth variation (higher k = deeper circuits)
  // and observable complexity (even vs odd moments)
  std::vector<int> my_moment_indices;
  
  if (use_mpi && mpi_size > 1) {
    // Compute cost for each moment
    std::vector<double> moment_costs(total_moments);
    for (int k = 0; k < total_moments; ++k) {
      // Circuit depth: k/2 SELECT-REFLECT iterations
      double circuit_depth = (k == 0) ? 0.5 : (k / 2.0);
      
      // Observable complexity: number of Pauli terms
      int observable_terms = (k % 2 == 0) ? R_op.num_terms() : U_op.num_terms();
      
      // Total cost (arbitrary units, used for relative comparison)
      moment_costs[k] = circuit_depth * observable_terms;
    }
    
    // Greedy bin packing: assign moments to minimize maximum load
    std::vector<std::vector<int>> rank_assignments(mpi_size);
    std::vector<double> rank_loads(mpi_size, 0.0);
    
    // Sort moments by cost (descending) for better packing
    std::vector<int> sorted_moments(total_moments);
    std::iota(sorted_moments.begin(), sorted_moments.end(), 0);
    std::sort(sorted_moments.begin(), sorted_moments.end(),
              [&](int a, int b) { return moment_costs[a] > moment_costs[b]; });
    
    // Assign each moment to least-loaded rank
    for (int k : sorted_moments) {
      auto min_it = std::min_element(rank_loads.begin(), rank_loads.end());
      int min_rank = std::distance(rank_loads.begin(), min_it);
      rank_assignments[min_rank].push_back(k);
      rank_loads[min_rank] += moment_costs[k];
    }
    
    // Get this rank's assignment
    my_moment_indices = rank_assignments[mpi_rank];
    
    // Sort by k for sequential processing (better cache behavior)
    std::sort(my_moment_indices.begin(), my_moment_indices.end());
    
    // Print load balance analysis
    if (verbose && mpi_rank == 0) {
      std::cout << "\nLoad balance (cost-aware distribution):" << std::endl;
      double total_cost = std::accumulate(moment_costs.begin(), moment_costs.end(), 0.0);
      double ideal_per_rank = total_cost / mpi_size;
      double max_load = *std::max_element(rank_loads.begin(), rank_loads.end());
      double min_load = *std::min_element(rank_loads.begin(), rank_loads.end());
      
      for (int r = 0; r < mpi_size; ++r) {
        double imbalance = 100.0 * (rank_loads[r] - ideal_per_rank) / ideal_per_rank;
        std::cout << "  Rank " << r << ": " << std::fixed << std::setprecision(0) 
                  << rank_loads[r] << " cost units";
        std::cout << " (" << std::showpos << std::setprecision(1) << imbalance 
                  << "% from ideal)" << std::noshowpos << std::endl;
        std::cout << "    Moments: ";
        for (int k : rank_assignments[r]) {
          std::cout << k << " ";
        }
        std::cout << std::endl;
      }
      
      double balance_factor = min_load / max_load;
      std::cout << "  Balance efficiency: " << std::setprecision(1) 
                << (balance_factor * 100.0) << "%" << std::endl;
    }
  } else {
    // Serial execution: compute all moments
    my_moment_indices.resize(total_moments);
    std::iota(my_moment_indices.begin(), my_moment_indices.end(), 0);
  }
  
  // Each rank computes its assigned moments
  std::vector<double> my_moments;
  my_moments.reserve(my_moment_indices.size());
  
  for (int k : my_moment_indices) {
    int m = k / 2;
    
    // Create quantum kernel for this moment
    auto measure_kernel = [&, m, k, n_anc, n_sys, n_electrons]() __qpu__ {
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
    my_moments.push_back(moment);
    
    if (verbose) {
      std::cout << "  Rank " << mpi_rank << ": k=" << k << ": " << moment << std::endl;
    }
  }
  
#if MPI_ENABLED
  if (use_mpi) {
    // Gather moments to rank 0 using raw MPI_Gatherv
    // Note: We use raw MPI here because CUDA-Q's MPI interface doesn't provide
    // variable-length gather (Gatherv). CUDA-Q MPI is used for rank/size queries
    // above for consistency with CUDA-Q's distributed state vector support.
    
    // First, gather the count from each rank
    std::vector<int> recv_counts(mpi_size);
    int my_count = my_moment_indices.size();
    MPI_Gather(&my_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<int> all_indices;
    std::vector<double> all_moments;
    
    if (mpi_rank == 0) {
      // Prepare to receive all data
      int total_recv = 0;
      std::vector<int> displacements(mpi_size);
      for (int i = 0; i < mpi_size; ++i) {
        displacements[i] = total_recv;
        total_recv += recv_counts[i];
      }
      
      all_indices.resize(total_recv);
      all_moments.resize(total_recv);
      
      MPI_Gatherv(my_moment_indices.data(), my_count, MPI_INT,
                  all_indices.data(), recv_counts.data(), displacements.data(),
                  MPI_INT, 0, MPI_COMM_WORLD);
      
      MPI_Gatherv(my_moments.data(), my_count, MPI_DOUBLE,
                  all_moments.data(), recv_counts.data(), displacements.data(),
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
      
      // Reconstruct full moment array in order
      for (size_t i = 0; i < all_indices.size(); ++i) {
        moments[all_indices[i]] = all_moments[i];
      }
    } else {
      // Non-root ranks just send their data
      MPI_Gatherv(my_moment_indices.data(), my_count, MPI_INT,
                  nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
      
      MPI_Gatherv(my_moments.data(), my_count, MPI_DOUBLE,
                  nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  } else
#endif
  {
    // Serial execution - just copy the moments in order
    for (size_t i = 0; i < my_moment_indices.size(); ++i) {
      moments[my_moment_indices[i]] = my_moments[i];
    }
  }
  
  // Verbose output on rank 0 (after gathering)
  if (verbose && mpi_rank == 0 && !use_mpi) {
    // Already printed during computation for MPI case
    for (int k = 0; k < total_moments; ++k) {
      std::cout << "  k=" << k << ": " << moments[k] << std::endl;
    }
  }
  
  // Build Krylov matrices (only on rank 0)
  std::vector<double> H_mat, S_mat;
  
  if (mpi_rank == 0) {
    if (verbose) {
      std::cout << "\nBuilding Krylov matrices..." << std::endl;
    }
    
    auto [H, S] = build_krylov_matrices(moments, krylov_dim);
    H_mat = std::move(H);
    S_mat = std::move(S);
    
    if (verbose) {
      std::cout << "\n=== Results ===" << std::endl;
      std::cout << "Hamiltonian matrix: " << krylov_dim << "×" << krylov_dim << std::endl;
      std::cout << "Overlap matrix: " << krylov_dim << "×" << krylov_dim << std::endl;
      std::cout << "Total moments collected: " << moments.size() << std::endl;
      std::cout << "\nTo extract eigenvalues, solve: H|v⟩ = E·S|v⟩" << std::endl;
      std::cout << "Then convert: E_physical = E_scaled * α + constant" << std::endl;
    }
  } else {
    // Non-root ranks return empty matrices
    // (User should only use result from rank 0)
    H_mat.resize(krylov_dim * krylov_dim, 0.0);
    S_mat.resize(krylov_dim * krylov_dim, 0.0);
    moments.clear(); // Save memory
  }
  
  // Return result
  return qel_result{
    H_mat,
    S_mat,
    moments,
    krylov_dim,
    constant_term,
    one_norm,
    n_anc,
    n_sys
  };
}

} // namespace cudaq::solvers

