---
title: Test-Defined Product Contract
status: accepted
normative: true
---

# Test-Defined Product Contract

## Authority

The supported-combination manifest and executable tests define what users may rely on. Documentation explains those
contracts but does not enlarge them. A feature is supported only when its interface, component behavior, integration,
and required execution environments are tested.

## Test Layers

1. **Specification:** public includes, construction syntax, defaults, result types, supported and rejected policy tuples.
2. **Component:** geometry, quadrature, basis, restriction, search, enforcement, response, and differentiation contracts.
3. **Conformance:** analytic patch problems, conservation, symmetry, adjointness, objectivity, and derivative checks.
4. **Adapter:** MFEM and borrowed-array workflows produce equivalent core inputs and mapped outputs.
5. **Parallel:** rank partitioning, complete CPU/device parity, and deterministic mode for advertised combinations.
6. **Quality:** architecture boundaries, header isolation, install consumption, formatting, analysis, and documentation.

## Required Evidence

- `linearization::Exact` propagates exact directional tangents on frozen interactions; tests compare those derivatives
  with independent centered finite differences away from activation and clipping kinks.
- Restriction operators verify the primal/dual transpose identity globally across ranks.
- Conservative formulations verify force balance and energy-gradient consistency.
- Assembled and matrix-free operators agree on the same state.
- Device tests execute kernels on a real GPU; device compilation alone is insufficient.
- MPI tests exercise at least one and two ranks, with four-rank coverage for partition-sensitive cases.
- The installed `tribol::core` target compiles and runs a minimal independent consumer.

The current MPI evidence covers the default MFEM adapter path at one, two, and four ranks. The current CUDA evidence
covers complete `Contact<DefaultMethod, ..., execution::Cuda>` evaluation on a physical CUDA device. No HIP method,
GPU-aware MPI transport, or scalable distributed search is advertised by the rewrite.

## Requirement IDs

Stable IDs in `src/tests/spec/requirements.yaml` connect tests to capabilities. Removing the last test for a required ID
is a contract change and requires an explicit design decision.
