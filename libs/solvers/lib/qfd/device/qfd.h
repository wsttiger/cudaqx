/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/qis/pauli_word.h"
#include "cudaq/qis/qkernel.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/qis/qview.h"

namespace cudaq {

void U_m(qview<> qubits, double dt,
         const std::vector<std::complex<double>> &coefficients,
         const std::vector<pauli_word> &words); 

void U_n(qview<> qubits, double dt,
         const std::vector<std::complex<double>> &coefficients,
         const std::vector<pauli_word> &words); 

__qpu__ void U_t(int order,
                 double dt,
                 const std::vector<std::complex<double>> &coefficients,
                 const std::vector<pauli_word> &words,
                 const std::vector<double> &vec);

void apply_pauli(qview<> qubits, const std::vector<int> &word);

void qfd_kernel(double dt_alpha, double dt_beta,
                const std::vector<std::complex<double>> &coefficients,
                const std::vector<pauli_word> &words,
                const std::vector<int> &word_list,
                const std::vector<double> &vec);

} // namespace cudaq
