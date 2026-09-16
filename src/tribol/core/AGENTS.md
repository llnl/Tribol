# Core Instructions

This directory is the dependency-free foundation of the rewritten library.

- Include only the C++ standard library and other headers under `tribol/core` or `tribol/method`.
- Do not include MFEM, MPI, Axom, RAJA, Umpire, redecomp, CUDA runtime, or HIP runtime headers.
- Views passed to kernels must be trivially copyable, non-owning, and explicit about layout and lifetime.
- Keep ownership, allocation, communication, and host-library conversion outside kernel-facing views.
- Host/device functions must not allocate, throw, perform I/O, or call virtual functions.
- Add compile-time and runtime contract tests under `src/tests/spec` or the relevant component test directory.
