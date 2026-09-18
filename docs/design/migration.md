---
title: Rewrite Migration Strategy
status: completed
normative: false
---

# Rewrite Migration Strategy

## Current Boundary

The policy-composed C++20 implementation is the only Tribol implementation in this repository. Global registries,
integer coupling-scheme IDs, named-method dispatch, the experimental `future` tree, redecomp, and their fixture data
were removed after their behavior was captured by specification, component, parity, MPI, and CUDA tests.

The tests under `src/tests/legacy_parity` contain no legacy implementation. Their names record the origin of the
analytic behavior they preserve. They remain fast in-tree regression tests for the supported policy tuples.

## External Comparison

Cross-version evidence is produced by `scripts/benchmarks/compare.py` and `benchmarks/suites.json`. The current driver
links `tribol::core`; the benchmark-only legacy adapter links a separately installed named-method Tribol. A caller can
select an existing build/install directory or let the script clone, build, and install upstream `develop`.

This executable boundary avoids ABI collisions and keeps legacy headers, dependencies, global state, and method enums
out of production targets. The common JSON protocol records complete comparable vectors, scalar invariants, and raw
timing samples. Numerical mismatch is a failure; timing ratios are informational unless downstream automation applies
its own threshold.
