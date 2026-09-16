---
name: tribol-verify-rewrite
description: Run or maintain the Tribol rewrite verification matrix across interface specifications, mechanics conformance, static quality checks, MFEM/raw adapters, CUDA, and MPI.
---

# Verify the Tribol Rewrite

Use the narrowest relevant configured build first. Read `docs/design/testing-contract.md` and the manifests under
`src/tests/spec`.

1. Validate manifests and generated compile-time tuples.
2. Build and run specification and changed component tests.
3. Run analytic conformance and finite-difference derivative tests with frozen interactions.
4. Run MFEM/raw-adapter equivalence, installed-package, and header-isolation checks.
5. Run static quality checks in strict mode.
6. Run MPI rank variants and real CUDA execution when those requirements are affected. Do not treat device compilation
   or a one-rank run as runtime evidence.

Report exact commands, configuration, pass/fail counts, skipped optional tools, and any unverified requirement IDs.
Use `docs/design/support-matrix.md` to avoid broadening a result: current CUDA evidence is default-method only, and
current MPI evidence is the default MFEM path backed by replicated all-gather surfaces.
