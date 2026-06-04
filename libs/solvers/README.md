# CUDA-Q Solvers Library

CUDA-Q Solvers provides GPU-accelerated implementations of common
quantum-classical hybrid algorithms and numerical routines frequently
used in quantum computing applications. The library is designed to
work seamlessly with CUDA-Q quantum programs.

**Note**: CUDA-Q Solvers is currently only supported on Linux operating systems
using `x86_64` processors or `aarch64`/`arm64` processors. CUDA-Q Solvers does
not require a GPU to use, but some components are GPU-accelerated.

**Note**: CUDA-Q Solvers will require the presence of `libgfortran`, which is not distributed with the Python wheel, for provided classical optimizers. If `libgfortran` is not installed, you will need to install it via your distribution's package manager. On debian based systems, you can install this with `apt-get install gfortran`.

## Features

- Variational quantum eigensolvers (VQE)
- ADAPT-VQE
- Quantum approximate optimization algorithm (QAOA)
- Hamiltonian simulation routines

## FTQC Primitives

The solvers library also contains early fault-tolerant quantum computing
primitives built around Pauli LCU block encodings. These are intended to be
composable building blocks rather than complete applications.

| Layer | Host-side pieces | QPU-facing primitives |
| --- | --- | --- |
| Pauli LCU block encoding | `lcu_decomposition`, `pauli_lcu_kernel_data`, `pauli_lcu` metadata | `prepare`, `unprepare`, `select`, `controlled_select`, `apply` |
| Qubitization | PREPARE-state conventions and observable builders | zero/PREPARE reflections, forward and adjoint walks, controlled walks, walk powers, controlled walk powers |
| QSVT/QSP sequencing | `qsvt_phase_sequence`, `qsvt_sequence_policy`, `qsvt_plan`, `qsvt_transform_descriptor`, `qsvt_transform_plan` | signal phases, controlled signal phases, QSVT phase/walk sequences, controlled QSVT sequences |
| QSVT/QSP validation | `qsvt_response`, `qsvt_response_error`, response evaluators, response error estimators, uniform and Chebyshev sample grids | N/A; these are host-side diagnostics |

Host-side objects validate metadata, define conventions, and own data-layout
decisions. CUDA-Q kernels should consume only QPU-facing data extracted from
those host objects: registers, `pauli_lcu` encodings, primitive scalar values,
`std::vector<double>` phase data, and `std::vector<int>` walk-direction data.
`pauli_lcu::metadata()` exposes the encoded operator scale `H / alpha`, term
counts, constant term, and register sizes for transform setup.
`qsvt_plan::kernel_data()` and `qsvt_transform_plan::kernel_data()` are
convenience views for extracting the phase and walk-direction vectors before a
kernel invocation.

A typical QSVT call pattern is:

```c++
cudaq::spin_op h = /* Pauli operator */;
cudaq::solvers::pauli_lcu encoding(h, num_qubits);
auto plan = cudaq::solvers::make_qsvt_plan(
    phases, cudaq::solvers::make_alternating_qsvt_sequence_policy(degree));
auto kernel_data = plan.kernel_data();
auto phase_data = kernel_data.phases;
auto walk_direction_data = kernel_data.walk_directions;

auto kernel = [&]() __qpu__ {
  cudaq::qubit control;
  cudaq::qvector<> signal(encoding.num_ancilla());
  cudaq::qvector<> system(encoding.num_system());

  encoding.prepare(signal);
  cudaq::solvers::apply_controlled_qsvt_sequence(
      control, signal, system, encoding, phase_data, walk_direction_data);
};
```

The QSVT layer includes host-side transform descriptors for the primitive
matrix functions needed by algorithms such as linear solve, real-time
Hamiltonian simulation (`exp(-i H t)`), and imaginary-time evolution
(`exp(-H t)`). These descriptors capture validated metadata only; they do not
synthesize QSP/QSVT phases. A `qsvt_transform_plan` retains that descriptor
metadata while exposing the same kernel-ready phase and walk-direction data as a
plain `qsvt_plan`.

Users currently bring their own phase sequences from a paper, external
toolchain, or future phase-generation API. The host-side validation utilities
let users evaluate the scalar QSVT/QSP response for those phases, estimate the
maximum and RMS error against a target function on explicit sample points, and
build uniform or Chebyshev sample grids over the QSVT domain. These utilities
are deliberately host-only so phase validation can happen before a CUDA-Q
kernel invocation.

The C++ example `examples/cpp/qsvt_bring_your_own_phases.cpp` demonstrates the
current intended workflow:

```text
external phases
 -> qsvt_plan
 -> response evaluation and sampled error estimation
 -> kernel_data()
 -> apply_qsvt_sequence in a CUDA-Q kernel
```

Current limitations:

- Block encodings are currently Pauli LCU encodings for `cudaq::spin_op` inputs.
- Pauli LCU decomposition currently supports real coefficients.
- Controlled gate lowering uses explicit arities in the primitive kernels.
- QSVT phase synthesis is not implemented yet; callers provide phase sequences.
- Response evaluation and error estimation are classical diagnostics for the
  abstract scalar signal model, not circuit simulation replacements.
- Transform descriptors are metadata for future phase-generation APIs.
- Domain-level workflows such as linear solve or Hamiltonian simulation should
  live in examples until the underlying primitive APIs stabilize.
- A future phase-generation implementation should be developed and validated in
  a separate slice from these execution and validation primitives.

Note: if you would like to use our Generative Quantum Eigensolver API, you will need
additional dependencies installed. You can install them with
`pip install cudaq-solvers[gqe]`.

## Getting Started

For detailed documentation, tutorials, and API reference,
visit the [CUDA-Q Solvers Documentation](https://nvidia.github.io/cudaqx/components/solvers/introduction.html).

## License

CUDA-Q Solvers is an open source project. The source code is available on
[GitHub][github_link] and licensed under [Apache License
2.0](https://github.com/NVIDIA/cudaqx/blob/main/LICENSE).

[github_link]: https://github.com/NVIDIA/cudaqx/tree/main/libs/solvers
