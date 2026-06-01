/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq.h"
#include "cudaq/solvers/operators/qubitization.h"

#include <cstddef>
#include <vector>

namespace cudaq::solvers {

/// @brief Direction of the qubitization walk used between QSVT phases.
enum class qsvt_walk_direction { forward, adjoint };

/// @brief QSVT host/device API boundary.
///
/// qsvt_phase_sequence and qsvt_plan are host-side validation and metadata
/// types. QPU-facing helpers consume qview objects, pauli_lcu encodings,
/// primitive numeric values, and plain std::vector<double> phase data extracted
/// from a validated host-side sequence or plan. Do not pass qsvt_plan directly
/// into __qpu__ kernels.

/// @brief Apply a phase to the all-zero signal/ancilla state.
/// @details Implements exp(i phase |0><0|) on the signal register by mapping
/// |0...0> to |1...1>, applying a multi-controlled r1 phase, and unmapping.
/// This helper intentionally uses explicit arities to match CUDA-Q controlled
/// gate lowering constraints used by the Pauli LCU primitives.
__qpu__ inline void apply_qsvt_signal_phase(cudaq::qview<> signal,
                                            double phase) {
  for (std::size_t i = 0; i < signal.size(); ++i)
    x(signal[i]);

  std::size_t num_signal = signal.size();
  if (num_signal == 0) {
    return;
  } else if (num_signal == 1) {
    r1(phase, signal[0]);
  } else if (num_signal == 2) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1]);
  } else if (num_signal == 3) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2]);
  } else if (num_signal == 4) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2], signal[3]);
  } else if (num_signal == 5) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2], signal[3],
                    signal[4]);
  } else if (num_signal == 6) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5]);
  } else if (num_signal == 7) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5], signal[6]);
  } else if (num_signal == 8) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5], signal[6], signal[7]);
  } else if (num_signal == 9) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5], signal[6], signal[7], signal[8]);
  } else if (num_signal == 10) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5], signal[6], signal[7], signal[8],
                    signal[9]);
  }

  for (std::size_t i = 0; i < signal.size(); ++i)
    x(signal[i]);
}

/// @brief Kernel functor wrapper for a QSVT signal phase.
struct qsvt_signal_phase {
  void operator()(cudaq::qview<> signal, double phase) const __qpu__ {
    apply_qsvt_signal_phase(signal, phase);
  }
};

/// @brief Apply a QSVT-style phase/walk sequence for a Pauli LCU encoding.
/// @details Uses the convention phase[0], then repeats a qubitization walk
/// followed by phase[j] for j = 1..degree. This helper does not validate phases
/// in device code; callers should construct or verify a qsvt_phase_sequence on
/// the host before invoking it.
__qpu__ inline void apply_qsvt_sequence(cudaq::qview<> signal,
                                        cudaq::qview<> system,
                                        const pauli_lcu &encoding,
                                        const std::vector<double> &phases,
                                        qsvt_walk_direction direction) {
  if (phases.empty())
    return;

  apply_qsvt_signal_phase(signal, phases[0]);
  for (std::size_t i = 1; i < phases.size(); ++i) {
    if (direction == qsvt_walk_direction::forward)
      apply_qubitization_walk(signal, system, encoding);
    else
      apply_adjoint_qubitization_walk(signal, system, encoding);
    apply_qsvt_signal_phase(signal, phases[i]);
  }
}

/// @brief Apply a QSVT-style phase/walk sequence with forward walks.
__qpu__ inline void apply_qsvt_sequence(cudaq::qview<> signal,
                                        cudaq::qview<> system,
                                        const pauli_lcu &encoding,
                                        const std::vector<double> &phases) {
  apply_qsvt_sequence(signal, system, encoding, phases,
                      qsvt_walk_direction::forward);
}

/// @brief Kernel functor wrapper for applying a QSVT-style phase/walk sequence.
struct qsvt_sequence {
  void operator()(cudaq::qview<> signal, cudaq::qview<> system,
                  const pauli_lcu &encoding,
                  const std::vector<double> &phases) const __qpu__ {
    apply_qsvt_sequence(signal, system, encoding, phases);
  }

  void operator()(cudaq::qview<> signal, cudaq::qview<> system,
                  const pauli_lcu &encoding, const std::vector<double> &phases,
                  qsvt_walk_direction direction) const __qpu__ {
    apply_qsvt_sequence(signal, system, encoding, phases, direction);
  }
};

/// @brief Host-side QSVT phase sequence.
///
/// QSVT and QSP routines are parameterized by a sequence of phase angles. A
/// degree-d polynomial conventionally uses d + 1 phases. This type keeps that
/// convention explicit and validates the sequence before it is consumed by
/// device-side QSVT kernels.
struct qsvt_phase_sequence {
  std::vector<double> phases;

  qsvt_phase_sequence() = default;
  explicit qsvt_phase_sequence(std::vector<double> input_phases);

  bool empty() const { return phases.empty(); }
  std::size_t size() const { return phases.size(); }
  std::size_t degree() const;
  const std::vector<double> &data() const { return phases; }
  double operator[](std::size_t index) const { return phases[index]; }
};

/// @brief Host-side plan for applying a QSVT phase/walk sequence.
///
/// The plan is intentionally a host-side object. CUDA-Q kernels should consume
/// plain data extracted from the plan, such as phase_data(), rather than taking
/// qsvt_plan directly as a kernel argument.
struct qsvt_plan {
  qsvt_phase_sequence phase_sequence;

  explicit qsvt_plan(qsvt_phase_sequence input_phases);
  explicit qsvt_plan(std::vector<double> input_phases);

  std::size_t num_phases() const { return phase_sequence.size(); }
  std::size_t degree() const { return phase_sequence.degree(); }
  const qsvt_phase_sequence &phases() const { return phase_sequence; }
  const std::vector<double> &phase_data() const {
    return phase_sequence.data();
  }
};

/// @brief Return true if a QSVT phase sequence is non-empty and finite.
bool is_valid_qsvt_phase_sequence(const std::vector<double> &phases);

/// @brief Validate a QSVT phase sequence.
/// @throws std::invalid_argument if the sequence is empty or contains a
/// non-finite phase.
void validate_qsvt_phase_sequence(const std::vector<double> &phases);

/// @brief Return the polynomial degree represented by num_phases phases.
/// @details A degree-d QSVT polynomial is represented by d + 1 phases.
/// @throws std::invalid_argument if num_phases is zero.
std::size_t qsvt_polynomial_degree(std::size_t num_phases);

/// @brief Construct and validate a QSVT phase sequence.
qsvt_phase_sequence make_qsvt_phase_sequence(std::vector<double> phases);

/// @brief Construct a host-side QSVT plan from a phase sequence.
qsvt_plan make_qsvt_plan(std::vector<double> phases);

} // namespace cudaq::solvers
