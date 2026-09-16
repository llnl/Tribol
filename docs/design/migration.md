---
title: Rewrite Migration Strategy
status: active
normative: false
---

# Rewrite Migration Strategy

## Coexistence

The dependency-free core and policy model are the default build. Legacy Tribol remains in the repository as an opt-in
numerical oracle under `TRIBOL_ENABLE_LEGACY=ON`; it is not part of the supported rewrite interface. New public
contracts do not preserve global registration, integer coupling-scheme IDs, or method enums.

## Sequence

1. Freeze public syntax and capability tuples in specification tests.
2. Build independent views, policies, search, geometry, integration, and evaluation components.
3. Add the borrowed-array adapter and then the MFEM adapter.
4. Port each behavior into policy tuples and compare independent analytic expectations and legacy results where useful.
5. Establish CPU, CUDA, MPI, derivative, install, and quality gates.
6. Redirect examples and exported targets to the new interface.
7. Keep `src/tribol/future`, global managers, and named-method implementations disabled by default; remove them after
   any remaining numerical-oracle value is exhausted.

## Completion Condition

The supported rewrite is complete when all manifest capabilities are verified, no core dependency boundary is violated,
GPU and MPI suites pass, and installed consumers work. Physical removal of disabled legacy sources is separate cleanup;
`src/tribol/future` and the original implementation are references, not extension points.
