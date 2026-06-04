/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sstream>

#include "bindings/utils/type_casters.h"
#include "cuda-qx/core/kwargs_utils.h"
#include "cudaq/python/PythonCppInterop.h"
#include "cudaq/solvers/operators/block_encoding.h"
#include "cudaq/solvers/operators/qsvt.h"
#include "cudaq/solvers/quantum_exact_lanczos.h"

namespace nb = nanobind;
using namespace cudaqx;

namespace {
template <typename T>
nb::object numpy_array(const std::vector<T> &values) {
  auto np = nb::module_::import_("numpy");
  return np.attr("array")(values);
}

nb::object numpy_matrix(const std::vector<double> &values, int dim) {
  auto np = nb::module_::import_("numpy");
  return np.attr("array")(values).attr("reshape")(nb::make_tuple(dim, dim));
}
} // namespace

namespace cudaq::solvers {

void bindBlockEncoding(nb::module_ &mod) {

  // ============================================================================
  // PAULI LCU BLOCK ENCODING
  // ============================================================================

  nb::class_<pauli_lcu_metadata>(
      mod, "PauliLCUMetadata",
      R"(Scalar metadata for a Pauli LCU block encoding.)")
      .def(nb::init<>())
      .def_rw("num_system_qubits", &pauli_lcu_metadata::num_system_qubits)
      .def_rw("num_ancilla_qubits", &pauli_lcu_metadata::num_ancilla_qubits)
      .def_rw("num_terms", &pauli_lcu_metadata::num_terms)
      .def_rw("padded_num_terms", &pauli_lcu_metadata::padded_num_terms)
      .def_rw("normalization", &pauli_lcu_metadata::normalization)
      .def_rw("constant_term", &pauli_lcu_metadata::constant_term)
      .def_rw("coefficient_threshold",
              &pauli_lcu_metadata::coefficient_threshold)
      .def("__repr__", [](const pauli_lcu_metadata &self) {
        std::ostringstream oss;
        oss << "PauliLCUMetadata(num_system_qubits=" << self.num_system_qubits
            << ", num_ancilla_qubits=" << self.num_ancilla_qubits
            << ", num_terms=" << self.num_terms
            << ", padded_num_terms=" << self.padded_num_terms
            << ", normalization=" << self.normalization
            << ", constant_term=" << self.constant_term << ")";
        return oss.str();
      });

  nb::class_<pauli_lcu>(
      mod, "PauliLCU",
      R"(Block encoding using Pauli Linear Combination of Unitaries.

This implementation is optimized for Hamiltonians expressed as sums of Pauli
strings (e.g., molecular Hamiltonians from quantum chemistry). It uses:
  - PREPARE: State preparation tree with controlled rotations
  - SELECT: Controlled Pauli operations indexed by ancilla state

The encoding uses log₂(# terms) ancilla qubits and achieves α = ||H||₁.

Example:
    >>> from cudaq import spin
    >>> import cudaq_solvers as solvers
    >>> 
    >>> # Define Hamiltonian
    >>> h = 0.5 * spin.x(0) + 0.3 * spin.z(0)
    >>> 
    >>> # Create block encoding
    >>> encoding = solvers.PauliLCU(h, num_qubits=1)
    >>> 
    >>> print(f"Ancilla qubits needed: {encoding.num_ancilla}")
    >>> print(f"Normalization: {encoding.normalization}")
    >>> 
    >>> # Use in quantum kernel
    >>> @cudaq.kernel
    >>> def my_circuit():
    >>>     anc = cudaq.qvector(encoding.num_ancilla)
    >>>     sys = cudaq.qvector(encoding.num_system)
    >>>     encoding.apply(anc, sys))")
      .def(nb::init<const cudaq::spin_op &, std::size_t>(),
           nb::arg("hamiltonian"), nb::arg("num_qubits"),
           R"(Initialize Pauli LCU block encoding.

Args:
    hamiltonian: Target Hamiltonian as a SpinOperator
    num_qubits: Number of system qubits

Raises:
    RuntimeError: If Hamiltonian contains complex coefficients
    RuntimeError: If Hamiltonian has no terms)")
      .def_prop_ro("num_ancilla", &pauli_lcu::num_ancilla,
                   "Number of ancilla qubits: ⌈log₂(# terms)⌉")
      .def_prop_ro("num_system", &pauli_lcu::num_system,
                   "Number of system qubits")
      .def_prop_ro("normalization", &pauli_lcu::normalization,
                   "Normalization constant: α = ||H||₁ (1-norm)")
      .def_prop_ro("constant_term", &pauli_lcu::constant_term,
                   "Constant identity component retained in the encoding")
      .def_prop_ro("term_count", &pauli_lcu::term_count,
                   "Number of retained LCU terms before padding")
      .def_prop_ro("padded_term_count", &pauli_lcu::padded_term_count,
                   "Number of LCU leaves after power-of-two padding")
      .def("metadata", &pauli_lcu::metadata,
           "Return scalar metadata for transform setup")
      .def("prepare", &pauli_lcu::prepare, nb::arg("ancilla"),
           R"(Apply the PREPARE operation to ancilla qubits.
          
Prepares a superposition state on the ancilla qubits that
encodes the coefficients of the Hamiltonian terms.

Args:
    ancilla: View of ancilla qubits)")
      .def("unprepare", &pauli_lcu::unprepare, nb::arg("ancilla"),
           R"(Apply the PREPARE† (adjoint/uncomputation) operation.

Args:
    ancilla: View of ancilla qubits)")
      .def("select", &pauli_lcu::select, nb::arg("ancilla"), nb::arg("system"),
           R"(Apply the SELECT operation.
          
Applies the appropriate Hamiltonian term conditioned on the
ancilla register state.

Args:
    ancilla: View of ancilla qubits (control register)
    system: View of system qubits (target register))")
      .def("controlled_select", &pauli_lcu::controlled_select,
           nb::arg("control"), nb::arg("ancilla"), nb::arg("system"),
           R"(Apply SELECT controlled by an additional qubit.)")
      .def("apply", &pauli_lcu::apply, nb::arg("ancilla"), nb::arg("system"),
           R"(Apply the full block encoding: PREPARE → SELECT → PREPARE†.

Args:
    ancilla: View of ancilla qubits
    system: View of system qubits)")
      .def(
          "get_angles",
          [](const pauli_lcu &self) { return numpy_array(self.get_angles()); },
          "Get state preparation angles as NumPy array (for debugging)")
      .def(
          "get_term_controls",
          [](const pauli_lcu &self) {
            return numpy_array(self.get_term_controls());
          },
          "Get binary control patterns as NumPy array (for debugging)")
      .def(
          "get_term_ops",
          [](const pauli_lcu &self) {
            return numpy_array(self.get_term_ops());
          },
          "Get flattened Pauli operations as NumPy array (for debugging)")
      .def(
          "get_term_lengths",
          [](const pauli_lcu &self) {
            return numpy_array(self.get_term_lengths());
          },
          "Get number of operators per term as NumPy array (for debugging)")
      .def(
          "get_term_signs",
          [](const pauli_lcu &self) {
            return numpy_array(self.get_term_signs());
          },
          "Get sign of each coefficient as NumPy array (for debugging)");

  // ============================================================================
  // QSVT HOST-SIDE PRIMITIVES
  // ============================================================================

  nb::enum_<qsvt_walk_direction>(mod, "QSVTWalkDirection",
                                 R"(Direction of a qubitization walk.)")
      .value("forward", qsvt_walk_direction::forward)
      .value("adjoint", qsvt_walk_direction::adjoint)
      .export_values();

  nb::enum_<qsvt_phase_convention>(
      mod, "QSVTPhaseConvention",
      R"(Convention used to interpret QSP/QSVT phases.)")
      .value("qsvt", qsvt_phase_convention::qsvt)
      .value("qsp", qsvt_phase_convention::qsp)
      .export_values();

  nb::enum_<qsvt_transform_kind>(
      mod, "QSVTTransformKind",
      R"(Host-side QSVT matrix-function transform kind.)")
      .value("custom", qsvt_transform_kind::custom)
      .value("linear_solve", qsvt_transform_kind::linear_solve)
      .value("real_time_hamiltonian_simulation",
             qsvt_transform_kind::real_time_hamiltonian_simulation)
      .value("imaginary_time_hamiltonian_simulation",
             qsvt_transform_kind::imaginary_time_hamiltonian_simulation)
      .export_values();

  nb::class_<qsvt_response>(mod, "QSVTResponse",
                            R"(Scalar response of a QSP/QSVT phase sequence.)")
      .def(nb::init<>())
      .def_rw("value", &qsvt_response::value)
      .def_rw("magnitude", &qsvt_response::magnitude)
      .def_rw("probability", &qsvt_response::probability);

  nb::class_<qsvt_response_error>(
      mod, "QSVTResponseError", R"(Sampled QSVT response approximation error.)")
      .def(nb::init<>())
      .def_rw("max_abs_error", &qsvt_response_error::max_abs_error)
      .def_rw("rms_error", &qsvt_response_error::rms_error)
      .def_rw("max_error_x", &qsvt_response_error::max_error_x)
      .def_rw("num_samples", &qsvt_response_error::num_samples);

  nb::class_<qsvt_phase_sequence>(mod, "QSVTPhaseSequence",
                                  R"(Validated host-side QSVT phase sequence.)")
      .def(nb::init<std::vector<double>>(), nb::arg("phases"))
      .def("__len__", &qsvt_phase_sequence::size)
      .def("__getitem__", [](const qsvt_phase_sequence &self,
                             std::size_t index) { return self[index]; })
      .def_prop_ro("size", &qsvt_phase_sequence::size)
      .def_prop_ro("degree", &qsvt_phase_sequence::degree)
      .def_prop_ro("phases",
                   [](const qsvt_phase_sequence &self) {
                     return std::vector<double>(self.data().begin(),
                                                self.data().end());
                   })
      .def("data", [](const qsvt_phase_sequence &self) {
        return std::vector<double>(self.data().begin(), self.data().end());
      });

  nb::class_<qsvt_sequence_policy>(mod, "QSVTSequencePolicy",
                                   R"(Validated host-side QSVT walk policy.)")
      .def(nb::init<std::vector<int>>(), nb::arg("walk_directions"))
      .def("__len__", &qsvt_sequence_policy::size)
      .def("__getitem__", [](const qsvt_sequence_policy &self,
                             std::size_t index) { return self[index]; })
      .def_prop_ro("size", &qsvt_sequence_policy::size)
      .def_prop_ro("degree", &qsvt_sequence_policy::degree)
      .def_prop_ro("walk_directions",
                   [](const qsvt_sequence_policy &self) {
                     const auto &directions = self.walk_direction_data();
                     return std::vector<int>(directions.begin(),
                                             directions.end());
                   })
      .def("walk_direction_data", [](const qsvt_sequence_policy &self) {
        const auto &directions = self.walk_direction_data();
        return std::vector<int>(directions.begin(), directions.end());
      });

  nb::class_<qsvt_transform_descriptor>(
      mod, "QSVTTransformDescriptor",
      R"(Host-side metadata describing a target QSVT transform.

This object does not generate phases. It validates and records the transform
that user-supplied or externally generated phases are intended to implement.)")
      .def(nb::init<>())
      .def_rw("kind", &qsvt_transform_descriptor::kind)
      .def_rw("phase_convention", &qsvt_transform_descriptor::phase_convention)
      .def_rw("evolution_time", &qsvt_transform_descriptor::evolution_time)
      .def_rw("condition_number", &qsvt_transform_descriptor::condition_number)
      .def_rw("target_error", &qsvt_transform_descriptor::target_error)
      .def_rw("normalization", &qsvt_transform_descriptor::normalization)
      .def_rw("degree_hint", &qsvt_transform_descriptor::degree_hint);

  nb::class_<qsvt_plan>(mod, "QSVTPlan",
                        R"(Validated host-side QSVT phase/walk sequence plan.)")
      .def(nb::init<std::vector<double>>(), nb::arg("phases"))
      .def(nb::init<std::vector<double>, qsvt_sequence_policy>(),
           nb::arg("phases"), nb::arg("policy"))
      .def_prop_ro("num_phases", &qsvt_plan::num_phases)
      .def_prop_ro("degree", &qsvt_plan::degree)
      .def_prop_ro("phase_data",
                   [](const qsvt_plan &self) {
                     const auto &phases = self.phase_data();
                     return std::vector<double>(phases.begin(), phases.end());
                   })
      .def_prop_ro("walk_direction_data",
                   [](const qsvt_plan &self) {
                     const auto &directions = self.walk_direction_data();
                     return std::vector<int>(directions.begin(),
                                             directions.end());
                   })
      .def("kernel_data", [](const qsvt_plan &self) {
        nb::dict data;
        const auto &phases = self.phase_data();
        const auto &directions = self.walk_direction_data();
        data["phases"] =
            nb::cast(std::vector<double>(phases.begin(), phases.end()));
        data["walk_directions"] =
            nb::cast(std::vector<int>(directions.begin(), directions.end()));
        return data;
      });

  nb::class_<qsvt_transform_plan>(
      mod, "QSVTTransformPlan",
      R"(Validated QSVT plan associated with target transform metadata.)")
      .def_prop_ro("descriptor", &qsvt_transform_plan::descriptor)
      .def_prop_ro("plan", &qsvt_transform_plan::plan)
      .def_prop_ro("num_phases", &qsvt_transform_plan::num_phases)
      .def_prop_ro("degree", &qsvt_transform_plan::degree)
      .def_prop_ro("phase_data",
                   [](const qsvt_transform_plan &self) {
                     const auto &phases = self.phase_data();
                     return std::vector<double>(phases.begin(), phases.end());
                   })
      .def_prop_ro("walk_direction_data",
                   [](const qsvt_transform_plan &self) {
                     const auto &directions = self.walk_direction_data();
                     return std::vector<int>(directions.begin(),
                                             directions.end());
                   })
      .def("kernel_data", [](const qsvt_transform_plan &self) {
        nb::dict data;
        const auto &phases = self.phase_data();
        const auto &directions = self.walk_direction_data();
        data["phases"] =
            nb::cast(std::vector<double>(phases.begin(), phases.end()));
        data["walk_directions"] =
            nb::cast(std::vector<int>(directions.begin(), directions.end()));
        return data;
      });

  mod.def("is_valid_qsvt_phase_sequence", &is_valid_qsvt_phase_sequence,
          nb::arg("phases"));
  mod.def("validate_qsvt_phase_sequence", &validate_qsvt_phase_sequence,
          nb::arg("phases"));
  mod.def("qsvt_polynomial_degree", &qsvt_polynomial_degree,
          nb::arg("num_phases"));
  mod.def("qsvt_walk_direction_code", &qsvt_walk_direction_code,
          nb::arg("direction"));
  mod.def("is_valid_qsvt_sequence_policy", &is_valid_qsvt_sequence_policy,
          nb::arg("degree"), nb::arg("policy"));
  mod.def("validate_qsvt_sequence_policy", &validate_qsvt_sequence_policy,
          nb::arg("degree"), nb::arg("policy"));
  mod.def("make_qsvt_phase_sequence", &make_qsvt_phase_sequence,
          nb::arg("phases"));
  mod.def(
      "make_qsvt_sequence_policy",
      [](std::size_t degree, qsvt_walk_direction direction) {
        return make_qsvt_sequence_policy(degree, direction);
      },
      nb::arg("degree"), nb::arg("direction") = qsvt_walk_direction::forward);
  mod.def(
      "make_custom_qsvt_sequence_policy",
      [](std::vector<qsvt_walk_direction> directions) {
        return make_qsvt_sequence_policy(std::move(directions));
      },
      nb::arg("directions"));
  mod.def("make_alternating_qsvt_sequence_policy",
          &make_alternating_qsvt_sequence_policy, nb::arg("degree"),
          nb::arg("first_direction") = qsvt_walk_direction::forward);
  mod.def(
      "make_qsvt_plan",
      [](std::vector<double> phases) {
        return make_qsvt_plan(std::move(phases));
      },
      nb::arg("phases"));
  mod.def(
      "make_qsvt_plan",
      [](std::vector<double> phases, qsvt_sequence_policy policy) {
        return make_qsvt_plan(std::move(phases), std::move(policy));
      },
      nb::arg("phases"), nb::arg("policy"));

  mod.def("is_valid_qsvt_transform_descriptor",
          &is_valid_qsvt_transform_descriptor, nb::arg("descriptor"));
  mod.def("validate_qsvt_transform_descriptor",
          &validate_qsvt_transform_descriptor, nb::arg("descriptor"));
  mod.def("validate_qsvt_transform_phase_sequence",
          &validate_qsvt_transform_phase_sequence, nb::arg("descriptor"),
          nb::arg("phases"));
  mod.def(
      "make_qsvt_transform_plan",
      [](const qsvt_transform_descriptor &descriptor,
         std::vector<double> phases) {
        return make_qsvt_transform_plan(descriptor, std::move(phases));
      },
      nb::arg("descriptor"), nb::arg("phases"));
  mod.def(
      "make_qsvt_transform_plan",
      [](const qsvt_transform_descriptor &descriptor,
         std::vector<double> phases, qsvt_sequence_policy policy) {
        return make_qsvt_transform_plan(descriptor, std::move(phases),
                                        std::move(policy));
      },
      nb::arg("descriptor"), nb::arg("phases"), nb::arg("policy"));
  mod.def("make_linear_solve_qsvt_transform", &make_linear_solve_qsvt_transform,
          nb::arg("condition_number"), nb::arg("target_error"),
          nb::arg("degree_hint") = 0, nb::arg("normalization") = 1.0);
  mod.def("make_real_time_hamiltonian_simulation_qsvt_transform",
          &make_real_time_hamiltonian_simulation_qsvt_transform,
          nb::arg("evolution_time"), nb::arg("target_error"),
          nb::arg("degree_hint") = 0, nb::arg("normalization") = 1.0);
  mod.def("make_imaginary_time_hamiltonian_simulation_qsvt_transform",
          &make_imaginary_time_hamiltonian_simulation_qsvt_transform,
          nb::arg("evolution_time"), nb::arg("target_error"),
          nb::arg("degree_hint") = 0, nb::arg("normalization") = 1.0);

  mod.def(
      "evaluate_qsvt_response",
      [](std::vector<double> phases, double x,
         qsvt_phase_convention convention) {
        return evaluate_qsvt_response(phases, x, convention);
      },
      nb::arg("phases"), nb::arg("x"),
      nb::arg("convention") = qsvt_phase_convention::qsvt);
  mod.def(
      "evaluate_qsvt_response",
      [](const qsvt_phase_sequence &phases, double x,
         qsvt_phase_convention convention) {
        return evaluate_qsvt_response(phases, x, convention);
      },
      nb::arg("phases"), nb::arg("x"),
      nb::arg("convention") = qsvt_phase_convention::qsvt);
  mod.def(
      "evaluate_qsvt_response",
      [](const qsvt_plan &plan, double x, qsvt_phase_convention convention) {
        return evaluate_qsvt_response(plan, x, convention);
      },
      nb::arg("plan"), nb::arg("x"),
      nb::arg("convention") = qsvt_phase_convention::qsvt);
  mod.def(
      "evaluate_qsvt_response",
      [](const qsvt_transform_plan &plan, double x) {
        return evaluate_qsvt_response(plan, x);
      },
      nb::arg("plan"), nb::arg("x"));

  mod.def(
      "estimate_qsvt_response_error",
      [](std::vector<double> phases,
         const std::function<std::complex<double>(double)> &target,
         std::vector<double> sample_points, qsvt_phase_convention convention) {
        return estimate_qsvt_response_error(phases, target, sample_points,
                                            convention);
      },
      nb::arg("phases"), nb::arg("target"), nb::arg("sample_points"),
      nb::arg("convention") = qsvt_phase_convention::qsvt);
  mod.def(
      "estimate_qsvt_response_error",
      [](const qsvt_phase_sequence &phases,
         const std::function<std::complex<double>(double)> &target,
         std::vector<double> sample_points, qsvt_phase_convention convention) {
        return estimate_qsvt_response_error(phases, target, sample_points,
                                            convention);
      },
      nb::arg("phases"), nb::arg("target"), nb::arg("sample_points"),
      nb::arg("convention") = qsvt_phase_convention::qsvt);
  mod.def(
      "estimate_qsvt_response_error",
      [](const qsvt_plan &plan,
         const std::function<std::complex<double>(double)> &target,
         std::vector<double> sample_points, qsvt_phase_convention convention) {
        return estimate_qsvt_response_error(plan, target, sample_points,
                                            convention);
      },
      nb::arg("plan"), nb::arg("target"), nb::arg("sample_points"),
      nb::arg("convention") = qsvt_phase_convention::qsvt);
  mod.def(
      "estimate_qsvt_response_error",
      [](const qsvt_transform_plan &plan,
         const std::function<std::complex<double>(double)> &target,
         std::vector<double> sample_points) {
        return estimate_qsvt_response_error(plan, target, sample_points);
      },
      nb::arg("plan"), nb::arg("target"), nb::arg("sample_points"));

  mod.def("make_uniform_qsvt_sample_points", &make_uniform_qsvt_sample_points,
          nb::arg("min_x"), nb::arg("max_x"), nb::arg("num_points"));
  mod.def("make_chebyshev_qsvt_sample_points",
          &make_chebyshev_qsvt_sample_points, nb::arg("min_x"),
          nb::arg("max_x"), nb::arg("num_points"));

  // ============================================================================
  // QUANTUM EXACT LANCZOS RESULT
  // ============================================================================

  nb::class_<qel_result>(mod, "QELResult",
                         R"(Result from Quantum Exact Lanczos algorithm.

Contains Krylov matrices that can be diagonalized to extract eigenvalues.
Solve the generalized eigenvalue problem H|v⟩ = E·S|v⟩ to get scaled
eigenvalues, then convert to physical energies:
    E_physical = E_scaled * normalization + constant_term

Example:
    >>> import numpy as np
    >>> from scipy import linalg as la
    >>> 
    >>> result = quantum_exact_lanczos(hamiltonian, ...)
    >>> 
    >>> # Reshape flat matrices
    >>> H = result.hamiltonian_matrix.reshape(
    >>>     result.krylov_dimension, result.krylov_dimension
    >>> )
    >>> S = result.overlap_matrix.reshape(
    >>>     result.krylov_dimension, result.krylov_dimension
    >>> )
    >>> 
    >>> # Solve generalized eigenvalue problem
    >>> eigenvalues = la.eigh(H, S, eigvals_only=True)
    >>> 
    >>> # Convert to physical energies
    >>> energies = eigenvalues * result.normalization + result.constant_term
    >>> ground_state = energies.min())")
      .def(nb::init<>())
      .def_rw("hamiltonian_matrix", &qel_result::hamiltonian_matrix,
              "Krylov H matrix (flattened, row-major, krylov_dim × krylov_dim)")
      .def_rw("overlap_matrix", &qel_result::overlap_matrix,
              "Krylov S matrix (flattened, row-major, krylov_dim × krylov_dim)")
      .def_rw("moments", &qel_result::moments,
              "Collected moments: μₖ = ⟨ψ|Tₖ(H)|ψ⟩")
      .def_rw("krylov_dimension", &qel_result::krylov_dimension,
              "Dimension of the Krylov subspace")
      .def_rw("constant_term", &qel_result::constant_term,
              "Constant term from Hamiltonian (add to eigenvalues)")
      .def_rw("normalization", &qel_result::normalization,
              "Normalization constant α = ||H||₁")
      .def_rw("num_ancilla", &qel_result::num_ancilla,
              "Number of ancilla qubits used")
      .def_rw("num_system", &qel_result::num_system, "Number of system qubits")
      .def(
          "get_hamiltonian_matrix",
          [](const qel_result &self) {
            return numpy_matrix(self.hamiltonian_matrix, self.krylov_dimension);
          },
          "Get Hamiltonian matrix as 2D NumPy array")
      .def(
          "get_overlap_matrix",
          [](const qel_result &self) {
            return numpy_matrix(self.overlap_matrix, self.krylov_dimension);
          },
          "Get overlap matrix as 2D NumPy array")
      .def(
          "get_moments",
          [](const qel_result &self) { return numpy_array(self.moments); },
          "Get moments as NumPy array")
      .def("__repr__", [](const qel_result &self) {
        std::ostringstream oss;
        oss << "QELResult(krylov_dimension=" << self.krylov_dimension
            << ", num_system=" << self.num_system
            << ", num_ancilla=" << self.num_ancilla
            << ", normalization=" << self.normalization
            << ", constant_term=" << self.constant_term << ")";
        return oss.str();
      });

  // ============================================================================
  // QUANTUM EXACT LANCZOS FUNCTION
  // ============================================================================

  mod.def(
      "quantum_exact_lanczos",
      [](const cudaq::spin_op &hamiltonian, std::size_t num_qubits,
         std::size_t n_electrons, nb::kwargs options) {
        heterogeneous_map opts;
        opts.insert("krylov_dim", getValueOr<int>(options, "krylov_dim", 10));
        opts.insert("shots", getValueOr<int>(options, "shots", -1));
        opts.insert("verbose", getValueOr<bool>(options, "verbose", false));

        return quantum_exact_lanczos(hamiltonian, num_qubits, n_electrons,
                                     opts);
      },
      nb::arg("hamiltonian"), nb::arg("num_qubits"), nb::arg("n_electrons"),
      nb::arg("**kwargs"),
      R"(Run Quantum Exact Lanczos algorithm to compute ground state energy.

Uses block encoding and amplitude amplification to build a Krylov subspace
via quantum moment collection. Returns matrices for classical eigenvalue
extraction.

Args:
    hamiltonian: Target Hamiltonian as SpinOperator
    num_qubits: Number of system qubits
    n_electrons: Number of electrons (for Hartree-Fock initialization)

Keyword Args:
    krylov_dim (int): Dimension of Krylov subspace (default: 10)
    shots (int): Number of measurement shots, -1 for exact (default: -1)
    verbose (bool): Enable detailed output (default: False)

Returns:
    QELResult: Contains Krylov matrices and metadata

Example (basic usage):
    >>> from cudaq import spin
    >>> import cudaq_solvers as solvers
    >>> import numpy as np
    >>> from scipy import linalg as la
    >>> 
    >>> # H2 Hamiltonian
    >>> h2 = -1.05 + 0.4*spin.z(0) - 0.4*spin.z(1) + 0.18*spin.x(0)*spin.x(1)
    >>> 
    >>> # Run QEL
    >>> result = solvers.quantum_exact_lanczos(
    >>>     h2, num_qubits=2, n_electrons=2, krylov_dim=5
    >>> )
    >>> 
    >>> # Extract ground state energy
    >>> H = result.get_hamiltonian_matrix()
    >>> S = result.get_overlap_matrix()
    >>> eigenvalues = la.eigh(H, S, eigvals_only=True)
    >>> energies = eigenvalues * result.normalization + result.constant_term
    >>> ground_state = energies.min()
    >>> print(f"Ground state: {ground_state:.6f} Ha")

Example (with filtering):
    >>> result = solvers.quantum_exact_lanczos(h2, 2, 2, krylov_dim=8)
    >>> 
    >>> H = result.get_hamiltonian_matrix()
    >>> S = result.get_overlap_matrix() + 1e-12 * np.eye(8)  # Regularization
    >>> 
    >>> eigenvalues = la.eigh(H, S, eigvals_only=True)
    >>> 
    >>> # Filter to Chebyshev range
    >>> mask = np.abs(eigenvalues) <= 1.0
    >>> physical_evals = eigenvalues[mask] * result.normalization + result.constant_term
    >>> ground_state = physical_evals.min()

Notes:
    - Requires ⌈log₂(# terms)⌉ ancilla qubits
    - Supports Hamiltonians with up to 1024 terms (10 ancilla)
    - Returns matrices for user to diagonalize with preferred library
    - Initial state is Hartree-Fock (first n_electrons qubits in |1⟩))");
}

} // namespace cudaq::solvers
