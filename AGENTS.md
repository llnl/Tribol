# Tribol Contributor Contract

## Mission

Tribol is being rebuilt as a policy-composed contact mechanics library. Tests are the normative interface and capability
specification. The legacy implementation and `src/tribol/future` are temporary behavioral references, not architectural
templates to extend.

## Read First

- `docs/design/architecture.md`
- `docs/design/policy-model.md`
- `docs/design/testing-contract.md`
- `docs/design/support-matrix.md`
- `docs/design/public-api.md`
- `src/tests/spec/README.md` when present

## Non-Negotiable Boundaries

- Do not add production classes representing named methods such as CommonPlane, SingleMortar, AlignedMortar, or
  EnergyMortar. Express behavior through orthogonal policies.
- Keep `src/tribol/core` and `src/tribol/method` independent of MFEM, Axom, RAJA, Umpire, redecomp, and MPI headers.
- Put third-party integration behind adapters. Dependencies point from adapters to core, never from core to adapters.
- Keep kernel-facing views trivially copyable and usable in host and device code.
- Do not introduce global registries or integer-ID lifecycle APIs.
- Treat invalid policy combinations as compile-time errors when their incompatibility is structural.
- Add or update specification tests before changing the public interface.
- Do not advertise `Analytic`, `Enzyme`, HIP, scalable distributed search, native high order, or non-default CUDA/MPI
  support until matching executable evidence exists.

## Validation Order

Run the narrowest applicable checks first, then expand:

```bash
cmake --build build-clang19-enzyme-raja-umpire-caliper-debug --target tribol_interface_spec_test -j 4
ctest --test-dir build-clang19-enzyme-raja-umpire-caliper-debug --output-on-failure -R '^tribol_interface_spec$'
clang-format-19 --dry-run --Werror <changed-cpp-files>
git diff --check
```

Use `scripts/quality/check.py` and `scripts/spec/validate.py` after those tools exist. GPU and MPI behavior must be
validated by dedicated tests rather than inferred from a successful CPU build.

## Change Discipline

- Preserve unrelated working-tree changes.
- Prefer focused files with one mechanical responsibility.
- Document new extension points and invariants in the same change.
- Reference stable requirement IDs from `src/tests/spec/requirements.yaml` in conformance tests.
- Keep legacy parity tests until equivalent policy-composed tests provide stronger coverage.
