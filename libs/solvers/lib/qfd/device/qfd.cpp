/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"

namespace cudaq {

__qpu__ void U_m(qview<> qubits, 
		 double dt,
		 std::vector<double> &coefficients,
	         const std::vector<pauli_word> &words) {
}

__qpu__ void foo(qview<> qubits) {
}

__qpu__ void
qfd_kernel(double dt_alpha,
	   double dt_beta,
           const std::vector<double> &coefficients,
	   const std::vector<pauli_word> &words,
	   const std::vector<pauli_word> &word_list,
           const std::vector<double> &vec) {
  cudaq::qubit ancilla;
  cudaq::qvector qreg(vec);
  h(ancilla);

  x(ancilla);
  //control(U_m, ancilla, qreg, dt_alpha, coefficients, words);
  control(U_m, ancilla, qreg);
  x(ancilla);

  // control(apply_pauli, ancilla, qreg, word_list);
//  control(U_m, ancilla, qreg, dt_beta, coefficients, words);
}
} // namespace cudaq
