/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/solvers/operators/qsvt.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

int main() {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  // Bring-your-own phases: this degree-1 sequence has scalar response f(x) = x
  // in the abstract QSVT response model used by the host-side validators.
  std::vector<double> phases{0.0, 0.0};
  auto plan = make_qsvt_plan(phases);

  auto sample_points = make_uniform_qsvt_sample_points(-1.0, 1.0, 9);
  auto identity_target = [](double x) { return std::complex<double>(x, 0.0); };
  auto response_error =
      estimate_qsvt_response_error(plan, identity_target, sample_points);

  std::cout << "QSVT response max error: " << response_error.max_abs_error
            << "\n";
  if (response_error.max_abs_error > 1e-12)
    return 1;

  cudaq::spin_op h = x(0);
  pauli_lcu encoding(h, 1);
  auto kernel_data = plan.kernel_data();
  auto phase_data = kernel_data.phases;
  auto walk_direction_data = kernel_data.walk_directions;

  auto qsvt_kernel = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());

    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, phase_data,
                        walk_direction_data);
  };

  auto counts = cudaq::sample(100, qsvt_kernel);
  const double one_probability = counts.probability("1");
  std::cout << "Measured probability for |1>: " << one_probability << "\n";

  return std::abs(one_probability - 1.0) < 1e-12 ? 0 : 1;
}
