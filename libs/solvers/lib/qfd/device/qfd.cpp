/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"

namespace cudaq {

__qpu__ void U_n(qview<> qubits, double dt,
                 const std::vector<std::complex<double>> &coefficients,
                 const std::vector<pauli_word> &words) {
  for (std::size_t i = 0; i < coefficients.size(); i++) {
    exp_pauli(dt * std::real(coefficients[i]), qubits, words[i]);
  }
}

__qpu__ void U_t(int order,
                 double dt,
                 const std::vector<std::complex<double>> &coefficients,
                 const std::vector<pauli_word> &words,
                 const std::vector<double> &vec) {
  cudaq::qvector qreg(vec);
  double dt2 = dt / order;
  for (std::size_t o = 0; o < order; o++) {
    for (std::size_t i = 0; i < coefficients.size(); i++) {
      exp_pauli(dt2 * std::real(coefficients[i]), qreg, words[i]);
    }
  }
}

__qpu__ void apply_pauli(qview<> qubits, const std::vector<int> &word) {
  for (std::size_t i = 0; i < word.size(); i++) {
    if (word[i] == 1) {
      x(qubits[i]);
    }
    if (word[i] == 2) {
      y(qubits[i]);
    }
    if (word[i] == 3) {
      z(qubits[i]);
    }
  }
}

__qpu__ void apply_pauli(state* initial_state, const std::vector<int> &word) {
  cudaq::qvector qubits(initial_state);
  for (std::size_t i = 0; i < word.size(); i++) {
    if (word[i] == 1) {
      x(qubits[i]);
    }
    if (word[i] == 2) {
      y(qubits[i]);
    }
    if (word[i] == 3) {
      z(qubits[i]);
    }
  }
}
__qpu__ void qfd_kernel(double dt_alpha, double dt_beta,
                        const std::vector<std::complex<double>> &coefficients,
                        const std::vector<pauli_word> &words,
                        const std::vector<int> &word_list,
                        const std::vector<double> &vec) {
  cudaq::qubit ancilla;
  cudaq::qvector qreg(vec);
  h(ancilla);
  x(ancilla);
  control(U_n, ancilla, qreg, dt_alpha, coefficients, words);
  x(ancilla);
  // When the exp_pauli bug is fixed need to delete the apply_pauli(initial_state) version
  // as it appears that nvq++ doesn't like overloading kernel names
  // control(apply_pauli, ancilla, qreg, word_list);
  control(U_n, ancilla, qreg, dt_beta, coefficients, words);
  x(ancilla);
}

} // namespace cudaq
