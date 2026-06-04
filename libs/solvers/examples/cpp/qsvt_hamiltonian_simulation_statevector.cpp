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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double reference_time = 0.8;
constexpr double comparison_tolerance = 1e-6;

using complex = std::complex<double>;

std::vector<complex> reference_initial_state() {
  return {{0.2758743754707127, -0.055531643454665276},
          {-0.46488564039274177, -0.27285911065527407},
          {0.14468556273063377, 0.25359170539392289},
          {0.010516522623530502, -0.033872113754366183},
          {0.19908038657525357, 0.20195388047633422},
          {0.058237222648333864, 0.063045105459893377},
          {0.27595019706432961, 0.29357470401007457},
          {0.0047939007983036266, 0.2321323289535869},
          {-0.077960292422475486, 0.048069012092028039},
          {0.087664070109315997, 0.22362294606066374},
          {0.065256210444886525, -0.14348640584615982},
          {-0.053889410896279646, 0.19007485822921849},
          {-0.037347498258936518, -0.22357163107907563},
          {0.1086489859312339, 0.051835105245546631},
          {0.10636487597736641, 0.16081624521642909},
          {-0.074593298459491267, 0.033772632196799574}};
}

std::vector<complex> reference_evolved_state() {
  return {{0.2246830682116136, -0.11889767539822951},
          {-0.53499503066394272, 0.0048778817868362322},
          {0.10645532541256297, 0.28988174777489412},
          {-0.037435119731278235, 0.0028605952537975043},
          {0.27595135875867144, -0.003623116849318252},
          {0.096340803146168913, -0.04550870260636429},
          {0.34321868668129324, 0.072147116470046735},
          {0.21343423465456984, 0.18724857939199868},
          {-0.030060941145948483, -0.043648937184147987},
          {-0.043402781461925666, 0.26327802487868857},
          {0.22534918495423156, 0.0051304092850225982},
          {-0.15083410216211418, 0.077914779189460326},
          {-0.014142708635609517, -0.22967981301415055},
          {0.12994922397583875, 0.078338563573905631},
          {0.055797676257343443, 0.1638113408838778},
          {-0.026323542276385038, 0.031193711473036626}};
}

cudaq::spin_op reference_hamiltonian() {
  using namespace cudaq::spin;
  return 0.70 * z(0) - 0.43 * z(1) + 0.31 * z(2) - 0.22 * z(3) +
         0.19 * x(0) * x(1) - 0.17 * y(1) * y(2) + 0.13 * z(1) * z(2) * x(3) +
         0.11 * x(0) * y(1) * y(2) * x(3);
}

std::vector<double> load_phases(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("Unable to open phase file: " + path);

  std::vector<double> phases;
  double phase = 0.0;
  while (input >> phase)
    phases.push_back(phase);

  if (phases.empty())
    throw std::runtime_error("Phase file is empty: " + path);
  return phases;
}

std::vector<int> basis_bits(std::size_t index, std::size_t num_qubits) {
  std::vector<int> bits(num_qubits, 0);
  for (std::size_t q = 0; q < num_qubits; ++q) {
    const auto shift = num_qubits - q - 1;
    bits[q] = static_cast<int>((index >> shift) & 1u);
  }
  return bits;
}

std::vector<complex> postselect_zero_signal(cudaq::state &state,
                                            std::size_t num_signal_qubits,
                                            std::size_t num_system_qubits) {
  const std::size_t dimension = 1u << num_system_qubits;
  std::vector<std::vector<int>> basis_states;
  basis_states.reserve(dimension);

  for (std::size_t i = 0; i < dimension; ++i) {
    std::vector<int> basis(num_signal_qubits + num_system_qubits, 0);
    auto system_bits = basis_bits(i, num_system_qubits);
    for (std::size_t q = 0; q < num_system_qubits; ++q)
      basis[num_signal_qubits + q] = system_bits[q];
    basis_states.push_back(std::move(basis));
  }

  auto amplitudes = state.amplitudes(basis_states);
  double success_probability = 0.0;
  for (const auto &amplitude : amplitudes)
    success_probability += std::norm(amplitude);

  if (success_probability == 0.0)
    throw std::runtime_error(
        "QSVT signal-zero postselection probability is 0.");

  const double normalization = std::sqrt(success_probability);
  for (auto &amplitude : amplitudes)
    amplitude /= normalization;

  std::cout << "Signal-zero postselection probability: " << success_probability
            << "\n";
  return amplitudes;
}

complex inner_product(const std::vector<complex> &left,
                      const std::vector<complex> &right) {
  complex result = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i)
    result += std::conj(left[i]) * right[i];
  return result;
}

double l2_error_up_to_global_phase(const std::vector<complex> &actual,
                                   const std::vector<complex> &expected) {
  auto overlap = inner_product(expected, actual);
  complex phase = 1.0;
  if (std::abs(overlap) > 0.0)
    phase = std::conj(overlap / std::abs(overlap));

  double error = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i)
    error += std::norm(phase * actual[i] - expected[i]);
  return std::sqrt(error);
}

double fidelity(const std::vector<complex> &left,
                const std::vector<complex> &right) {
  return std::norm(inner_product(left, right));
}

std::vector<complex> normalize(std::vector<complex> state) {
  double norm = 0.0;
  for (const auto &amplitude : state)
    norm += std::norm(amplitude);
  norm = std::sqrt(norm);
  for (auto &amplitude : state)
    amplitude /= norm;
  return state;
}

std::vector<complex> run_qsvt_statevector(cudaq::spin_op hamiltonian,
                                          std::size_t num_system_qubits,
                                          const std::vector<complex> &initial,
                                          const std::vector<double> &phases,
                                          double evolution_time) {
  using namespace cudaq::solvers;

  pauli_lcu encoding(hamiltonian, num_system_qubits);
  auto metadata = encoding.metadata();
  auto descriptor = make_real_time_hamiltonian_simulation_qsvt_transform(
      evolution_time, comparison_tolerance, phases.size() - 1,
      metadata.normalization);
  auto transform_plan = make_qsvt_transform_plan(descriptor, phases);
  auto kernel_data = transform_plan.kernel_data();
  auto phase_data = kernel_data.phases;
  auto walk_direction_data = kernel_data.walk_directions;

  std::vector<std::complex<float>> initial_state_data;
  initial_state_data.reserve(initial.size());
  for (const auto &amplitude : initial)
    initial_state_data.emplace_back(static_cast<float>(amplitude.real()),
                                    static_cast<float>(amplitude.imag()));
  cudaq::state initial_state = cudaq::state::from_data(initial_state_data);

  auto qsvt_kernel = [&](cudaq::state &input_state) __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(input_state);

    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, phase_data,
                        walk_direction_data);
    encoding.unprepare(signal);
  };

  auto output_state = cudaq::get_state(qsvt_kernel, initial_state);
  return postselect_zero_signal(output_state, encoding.num_ancilla(),
                                encoding.num_system());
}

int run_toy_x_self_test() {
  using namespace cudaq::spin;

  constexpr double pi = 3.141592653589793238462643383279502884;
  auto initial = normalize({{0.6, 0.2}, {-0.3, 0.7}});
  std::vector<double> phases{-pi / 4.0, -pi / 4.0};

  auto actual = run_qsvt_statevector(x(0), 1, initial, phases, pi / 2.0);
  std::vector<complex> expected{complex{0.0, -1.0} * initial[1],
                                complex{0.0, -1.0} * initial[0]};

  const auto l2_error = l2_error_up_to_global_phase(actual, expected);
  const auto state_fidelity = fidelity(actual, expected);

  std::cout << std::setprecision(12);
  std::cout << "Toy H = X QSVT statevector self-test\n";
  std::cout << "L2 error up to global phase: " << l2_error << "\n";
  std::cout << "Fidelity: " << state_fidelity << "\n";

  if (l2_error > comparison_tolerance)
    return 1;
  if (std::abs(1.0 - state_fidelity) > comparison_tolerance)
    return 1;
  return 0;
}

void print_usage(const char *program) {
  std::cout << "QSVT Hamiltonian simulation statevector validation\n"
            << "\n"
            << "This simulator-only example uses the 4-qubit Hamiltonian and "
               "seed-13\n"
            << "initial ket from qsvt_hamiltonian_simulation_reference.py.\n"
            << "\n"
            << "Usage:\n  " << program << " --phases <phase-file>\n"
            << "  " << program << " --self-test-x\n\n"
            << "The phase file should contain whitespace-separated QSVT "
               "phases for\n"
            << "exp(-i H t) with t = " << reference_time
            << ". Phase generation is intentionally external.\n";
}

} // namespace

int main(int argc, char **argv) {
  std::string phase_file;
  bool run_self_test = false;
  for (int i = 1; i < argc; ++i) {
    std::string argument = argv[i];
    if (argument == "--phases" && i + 1 < argc) {
      phase_file = argv[++i];
    } else if (argument == "--self-test-x") {
      run_self_test = true;
    } else if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argument << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  if (run_self_test)
    return run_toy_x_self_test();

  if (phase_file.empty()) {
    print_usage(argv[0]);
    return 0;
  }

  using namespace cudaq::solvers;

  const auto phases = load_phases(phase_file);
  auto hamiltonian = reference_hamiltonian();
  pauli_lcu encoding(hamiltonian, 4);
  auto metadata = encoding.metadata();

  auto actual = run_qsvt_statevector(hamiltonian, 4, reference_initial_state(),
                                     phases, reference_time);
  auto expected = reference_evolved_state();

  const auto l2_error = l2_error_up_to_global_phase(actual, expected);
  const auto state_fidelity = fidelity(actual, expected);

  std::cout << std::setprecision(12);
  std::cout << "Reference Hamiltonian QSVT statevector validation\n";
  std::cout << "Number of phases: " << phases.size() << "\n";
  std::cout << "LCU normalization alpha: " << metadata.normalization << "\n";
  std::cout << "L2 error up to global phase: " << l2_error << "\n";
  std::cout << "Fidelity: " << state_fidelity << "\n";

  if (l2_error > comparison_tolerance)
    return 1;
  if (std::abs(1.0 - state_fidelity) > comparison_tolerance)
    return 1;
  return 0;
}
