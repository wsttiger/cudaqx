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

  constexpr double pi = 3.141592653589793238462643383279502884;
  constexpr double evolution_time = pi / 2.0;
  constexpr double normalization = 1.0;
  constexpr double target_error = 1e-12;

  // Toy real-time Hamiltonian simulation example:
  //
  //   H = X, alpha = 1, t = pi / 2
  //   exp(-i H t) = -i X
  //
  // The phase sequence below gives scalar response f(x) = -i x in the current
  // QSVT projector-phase convention. That response is exact on the spectrum
  // {-1, 1} of X for exp(-i t x). Generic Hamiltonians require phases generated
  // by an external tool such as QSPPACK or a future phase-generation API.
  std::vector<double> phases{-pi / 4.0, -pi / 4.0};

  auto descriptor = make_real_time_hamiltonian_simulation_qsvt_transform(
      evolution_time, target_error, phases.size() - 1, normalization);
  auto transform_plan = make_qsvt_transform_plan(descriptor, phases);

  std::vector<double> spectral_points{-1.0, 1.0};
  auto target_response = [](double x) {
    constexpr double local_pi = 3.141592653589793238462643383279502884;
    return std::exp(std::complex<double>(0.0, -local_pi * x / 2.0));
  };
  auto response_error = estimate_qsvt_response_error(
      transform_plan, target_response, spectral_points);

  std::cout << "Spectral response max error: " << response_error.max_abs_error
            << "\n";
  if (response_error.max_abs_error > target_error)
    return 1;

  cudaq::spin_op h = x(0);
  pauli_lcu encoding(h, 1);
  auto kernel_data = transform_plan.kernel_data();
  auto phase_data = kernel_data.phases;
  auto walk_direction_data = kernel_data.walk_directions;

  auto simulation_kernel = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());

    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, phase_data,
                        walk_direction_data);
  };

  auto counts = cudaq::sample(100, simulation_kernel);
  const double one_probability = counts.probability("1");
  std::cout << "Measured probability for |1>: " << one_probability << "\n";

  return std::abs(one_probability - 1.0) < target_error ? 0 : 1;
}
