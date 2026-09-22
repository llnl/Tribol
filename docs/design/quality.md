---
title: Rewrite Quality Gates
status: accepted
normative: true
---

# Rewrite Quality Gates

## Fast Checks

```bash
python3 -m unittest discover -s scripts/spec/tests -v
python3 -m unittest discover -s scripts/quality/tests -v
python3 scripts/quality/check.py custom spec format headers
git diff --check
```

`custom` checks dependency direction, forbidden named-method declarations, file-size limits, install/export manifests,
requirement evidence, documentation links, and device-backend isolation. Shared device files may use RAJA but may not
contain CUDA/HIP runtime APIs, CUB/hipCUB calls, launch syntax, or runtime headers. `spec` compiles generated assertions
for every advertised method variant and execution policy. `format` checks rewrite sources, benchmark adapters, and
tests. `headers` compiles each dependency-free public header in isolation.

## Strict Static Tools

```bash
python3 scripts/quality/check.py all --strict-tools --compiler clang++-19
```

Strict mode requires clang-format, clang-tidy, include-what-you-use, cppcheck, codespell, cmake-format, cmake-lint,
shellcheck, shfmt, Ruff, and mypy. C++ analyzers are scoped to the rewritten architecture; the external legacy adapter
is format-checked but requires a separately installed reference package to compile.
The CI workflow runs the Python unit tests, manifest validator, and strict quality driver.

## Runtime Gates

- Build and run the focused interface, capability-matrix, component, adapter, execution, and allocation tests first.
- Run `tribol_install_consumer` after building installable targets.
- Run `tribol_mfem_adapter_smoke`, `tribol_mfem_adapter_2rank`, and `tribol_mfem_adapter_4rank` for MFEM/MPI changes.
- Run `tribol_execution_cuda` on a physical CUDA device for CUDA changes.
- Run `tribol_execution_hip` on a physical AMD GPU with the matching ROCm toolchain for HIP changes.
- Run `python3 scripts/benchmarks/compare.py --suite smoke --reference-dir <build-or-install>` for cross-version changes.

Unavailable hardware is an unverified requirement, not a passing test.
