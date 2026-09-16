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
requirement evidence, and documentation links. `spec` compiles generated assertions for every advertised method variant
and execution policy. `format` checks only rewrite sources and tests. `headers` compiles each dependency-free public
header in isolation.

## Strict Static Tools

```bash
python3 scripts/quality/check.py all --strict-tools --compiler clang++-19
```

Strict mode requires clang-format, clang-tidy, include-what-you-use, cppcheck, codespell, cmake-format, cmake-lint,
shellcheck, shfmt, Ruff, and mypy. C++ analyzers are scoped to the rewritten architecture rather than legacy sources.
The CI workflow runs the Python unit tests, manifest validator, and strict quality driver.

## Runtime Gates

- Build and run the focused interface, capability-matrix, component, adapter, execution, and allocation tests first.
- Run `tribol_install_consumer` after building installable targets.
- Run `tribol_mfem_adapter_smoke`, `tribol_mfem_adapter_2rank`, and `tribol_mfem_adapter_4rank` for MFEM/MPI changes.
- Run `tribol_execution_cuda` on a physical CUDA device for CUDA changes.

Unavailable hardware is an unverified requirement, not a passing test.
