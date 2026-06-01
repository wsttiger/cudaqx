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
composable building blocks rather than complete applications:

- Pauli LCU decomposition and block-encoding helpers
- PREPARE, SELECT, and unprepare primitives for `cudaq::spin_op` inputs
- Qubitization reflections, forward walks, adjoint walks, and walk powers
- QSVT signal phases, validated phase sequences, and phase/walk sequence plans

QSVT support is currently a primitive API, not a full polynomial compiler. Host
code owns validation and policy construction through `qsvt_phase_sequence`,
`qsvt_sequence_policy`, and `qsvt_plan`. CUDA-Q kernels should consume only the
primitive data extracted from those objects, such as phase vectors and integer
walk-direction vectors. `qsvt_plan::kernel_data()` is a host-side convenience
view for extracting those vectors before a kernel invocation. The default QSVT
plan uses forward qubitization walks; callers can also request adjoint or
alternating forward/adjoint walk policies.

The QSVT layer also includes host-side transform descriptors for the primitive
matrix functions needed by algorithms such as linear solve, real-time
Hamiltonian simulation (`exp(-i H t)`), and imaginary-time evolution
(`exp(-H t)`). These descriptors capture validated metadata only; future work
should add phase-generation APIs for target polynomial transforms,
convention-specific sequence builders, and higher-level examples that compose
these primitives with block encodings.

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
