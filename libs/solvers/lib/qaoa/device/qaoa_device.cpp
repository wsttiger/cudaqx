/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/qaoa/qaoa_device.h"

namespace cudaq::solvers {
__qpu__ void qaoa_kernel(std::size_t numQubits, std::size_t numLayers,
                         const std::vector<double> &gamma_beta,
                         const std::vector<double> &problemHCoeffs,
                         const std::vector<cudaq::pauli_word> &problemH,
                         const std::vector<double> &referenceHCoeffs,
                         const std::vector<cudaq::pauli_word> &referenceH,
                         bool full_param, bool counterdiabatic) {
  cudaq::qvector q(numQubits);
  h(q);
  for (std::size_t angleCounter = 0, layer = 0; layer < numLayers; layer++) {
    for (std::size_t i = 0; i < problemHCoeffs.size(); i++) {
      exp_pauli(gamma_beta[angleCounter] * problemHCoeffs[i], q, problemH[i]);
      if (full_param)
        angleCounter++;
    }

    if (!full_param)
      angleCounter++;

    for (std::size_t i = 0; i < referenceHCoeffs.size(); i++) {
      exp_pauli(gamma_beta[angleCounter] * referenceHCoeffs[i], q,
                referenceH[i]);
      if (full_param)
        angleCounter++;
    }
    if (!full_param)
      angleCounter++;

    if (counterdiabatic)
      for (std::size_t i = 0; i < numQubits; i++)
        ry(gamma_beta[angleCounter++], q[i]);
  }
}
} // namespace cudaq::solvers
