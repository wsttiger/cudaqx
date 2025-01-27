/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cuda-qx/core/tensor.h"
#include "device/qfd.h"

#include <iostream>

namespace cudaq::solvers::qfd {	

std::vector<std::complex<double>> term_coefficients(cudaq::spin_op op) {
  std::vector<std::complex<double>> result{};
  op.for_each_term([&](cudaq::spin_op &term) {
    const auto coeff = term.get_coefficient();
    result.push_back(coeff);
  });
  return result;
}

std::vector<cudaq::pauli_word> term_words(cudaq::spin_op op) {
  std::vector<cudaq::pauli_word> result{};
  op.for_each_term(
      [&](cudaq::spin_op &term) { result.push_back(term.to_string(false)); });
  return result;
}

cudaq::spin_op identity(std::size_t num_qubits) {
  cudaq::spin_op identity;
  for (std::size_t i = 0; i < num_qubits; i++) {
    identity *= cudaq::spin::i(i);
  }
  return identity;
}

auto unzip_op(const cudaq::spin_op& op, std::size_t num_qubits) {
  return std::make_tuple(term_coefficients(op), term_words(op));
}

std::vector<std::vector<int>> translate_pauli_words_to_ints(const std::vector<cudaq::pauli_word> &words, std::size_t num_qubits) {
  // Convert the op_words into int's (not sure if this is necessary)
  std::map<char, int> char_to_int = {{'I', 0}, {'X', 1}, {'Y', 2}, {'Z', 3}};
  std::vector<std::vector<int>> op_words_int(words.size());
  std::transform(words.begin(), 
                 words.end(), 
                 op_words_int.begin(), 
                 [&] (const auto& word) {
                   std::vector<int> r(num_qubits, 0);
                   for (std::size_t i = 0; i < num_qubits; i++) {
                     r[i] = char_to_int[word.data()[i]];
                   }
                   return r;
                 });
  return op_words_int;
}

cudaq::state time_evolve_state(const cudaq::spin_op &h_op,
                               const std::size_t num_qubits,
                               const int order,
                               const double dt,
                               const std::vector<double>& vec) {
  // Pull apart the operator into coef's and words
  auto [h_coefs, h_words] = unzip_op(h_op, num_qubits);

  auto state = cudaq::get_state(cudaq::U_t, order, dt, h_coefs, h_words, vec);  
  return state;
}

std::complex<double> compute_time_evolved_amplitude(double dt_m, 
                                                    double dt_n, 
                                                    const std::vector<std::complex<double>>& h_coefs, 
                                                    const std::vector<cudaq::pauli_word>& h_words, 
                                                    const std::vector<std::vector<int>>& op_words_int,
                                                    const std::vector<double>& vec) {
  cudaq::spin_op x_0 = cudaq::spin::x(0);
  cudaq::spin_op y_0 = cudaq::spin::y(0);

  double mat_real = 0.0;
  double mat_imag = 0.0;
  for (std::size_t iword = 0; iword < op_words_int.size(); iword++) {
    auto results = cudaq::observe(cudaq::qfd_kernel,
                                  std::vector<cudaq::spin_op>({x_0, y_0}),
                                  dt_m,
                                  dt_n,
                                  h_coefs,
                                  h_words,
                                  op_words_int[iword],
                                  vec);
  }
  return std::complex(mat_real, mat_imag);
}

std::complex<double>  compute_time_evolved_amplitude(const cudaq::spin_op& op,
                                                     const cudaq::spin_op& h_op,
                                                     const std::size_t num_qubits,
                                                     const double dt_m,
                                                     const double dt_n,
                                                     const std::vector<double>& vec) {
  // Pull apart the operator into coef's and words
  auto [h_coefs, h_words]   = unzip_op(h_op, num_qubits);

  // Set trotter order to 1
  int order = 20;
  //
  // Create vectors for <bra| and |ket>
  std::size_t vec_size = 1ULL << num_qubits;
  std::vector<std::complex<double>> ket_vector(vec_size);
  std::vector<std::complex<double>> bra_vector(vec_size);
  //
  // Time evolve |vec> using dt_n and dt_m
  auto ket_state = cudaq::get_state(cudaq::U_t, order, dt_n, h_coefs, h_words, vec);
  auto bra_state = cudaq::get_state(cudaq::U_t, order, dt_m, h_coefs, h_words, vec);
  //
  // Copy to vectors (this is ugly)
  for (std::size_t i = 0; i < vec_size; i++) {
    ket_vector[i] = ket_state[i];
    bra_vector[i] = bra_state[i];
  }
  // Apply op as a matrix (for now)
  auto op_matrix = op.to_matrix();
  auto tmp_vector = op_matrix * ket_vector;

  std::complex<double> result(0.0, 0.0);
  for (int i = 0; i < vec_size; i++) {
    result += std::conj(bra_vector[i])*tmp_vector.data()[i];
  }
  return result;
}

// std::complex<double>  compute_time_evolved_amplitude(const cudaq::spin_op& op,
//                                                      const cudaq::spin_op& h_op,
//                                                      const std::size_t num_qubits,
//                                                      const double dt_m,
//                                                      const double dt_n,
//                                                      const std::vector<double>& vec) {
//   // Pull apart the operator into coef's and words
//   auto [op_coefs, op_words] = unzip_op(op, num_qubits);
//   auto [h_coefs, h_words]   = unzip_op(h_op, num_qubits);
// 
//   auto op_words_int = translate_pauli_words_to_ints(op_words, num_qubits);
//   return compute_time_evolved_amplitude(dt_m, dt_n, h_coefs, h_words, op_words_int, vec);
// }

cudaqx::tensor<> create_krylov_subspace_matrix(const cudaq::spin_op& op, 
                                               const cudaq::spin_op& h_op, 
                                               const std::size_t num_qubits,
                                               const std::size_t krylov_dim,
                                               const double dt,
                                               const std::vector<double>& vec) {

  // Pull apart the operator into coef's and words
  auto [op_coefs, op_words] = unzip_op(op, num_qubits);
  auto [h_coefs, h_words]   = unzip_op(h_op, num_qubits);

  auto op_words_int = translate_pauli_words_to_ints(op_words, num_qubits);

  cudaqx::tensor<> result({krylov_dim, krylov_dim});
  for (size_t m = 0; m < krylov_dim; m++) {
    double dt_m = m * dt;
    for (size_t n = 0; n < krylov_dim; n++) {
      double dt_n = n * dt;
      result.at({m, n}) = compute_time_evolved_amplitude(op, h_op, num_qubits, dt_m, dt_n, vec);
      if (n != m) {
        result.at({n, m}) = std::conj(result.at({m,n}));
      }
    }
  }
  return result;
}


} // namespace cudaq::solvers:qfd:
