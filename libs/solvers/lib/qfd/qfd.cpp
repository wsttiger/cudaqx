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

auto identity(std::size_t num_qubits) {
  cudaq::spin_op identity;
  for (std::size_t i = 0; i < num_qubits; i++) {
    identity *= cudaq::spin::i(i);
  }
  return identity;
}

auto unzip_op(const cudaq::spin_op& op, std::size_t num_qubits) {
  return std::make_tuple(term_coefficients(op), term_words(op));
}

auto create_krylov_subspace_matrix(const cudaq::spin_op& op, 
                                   const cudaq::spin_op& h_op, 
                                   const std::size_t num_qubits,
                                   const std::size_t krylov_dim,
                                   const double dt,
                                   const std::vector<double>& vec) {

  cudaq::spin_op x_0 = cudaq::spin::x(0);
  cudaq::spin_op y_0 = cudaq::spin::y(0);

  // Pull apart the operator into coef's and words
  auto [op_coefs, op_words] = unzip_op(op, num_qubits);
  auto [h_coefs, h_words]   = unzip_op(h_op, num_qubits);

  // Convert the op_words into int's (not sure if this is necessary)
  std::map<char, int> char_to_int = {{'I', 0}, {'X', 1}, {'Y', 2}, {'Z', 3}};
  std::vector<std::vector<int>> op_words_int(op_words.size());
  std::transform(op_words.begin(), 
                 op_words.end(), 
                 op_words_int.begin(), 
                 [&] (const auto& word) {
                   std::vector<int> r(num_qubits, 0);
                   for (std::size_t i = 0; i < num_qubits; i++) {
                     r[i] = char_to_int[word.data()[i]];
                   }
                   return r;
                 });

  cudaqx::tensor<> result({krylov_dim, krylov_dim});
  for (size_t m = 0; m < krylov_dim; m++) {
    double dt_m = m * dt;
    for (size_t n = 0; n <= m; n++) {
      double dt_n = n * dt;
      double mat_real = 0.0;
      double mat_imag = 0.0;
      for (std::size_t iword = 0; iword < op_words.size(); iword++) {

        auto results = cudaq::observe(cudaq::qfd_kernel,
                                      std::vector<cudaq::spin_op>({x_0, y_0}),
                                      dt_m,
                                      dt_n,
                                      h_coefs,
                                      h_words,
                                      op_words_int[iword],
                                      vec);
        mat_real += results[0].expectation(); 
        mat_imag += results[1].expectation();
      }
      result.at({m, n}) = std::complex(mat_real, mat_imag);
      result.at({n, m}) = std::conj(result.at({m,n}));
    }
  }
  return result;
}


} // namespace cudaq::solvers:qfd:
