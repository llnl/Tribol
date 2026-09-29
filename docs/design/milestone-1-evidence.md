---
title: Milestone 1 Evidence Ledger
status: implementation-ready-validation-pending
milestone: 1
normative: true
---

# Milestone 1 Evidence Ledger

## Purpose

This ledger maps the [Milestone 1 delivery plan](milestone-1-delivery-plan.md) to stable, executable evidence. It does
not expand the supported feature set. A compile-only result is not runtime evidence, and an unavailable environment is
not a pass.

## In-Tree Evidence

| Deliverable | Primary executable evidence | Current implementation state |
| --- | --- | --- |
| M1-01 native contract | `tribol_interface_spec`, `tribol_parent_trace` | Implemented |
| M1-02 overlap provenance | `tribol_overlap_cells`, `tribol_projected_overlap`, `tribol_native_refinement_convergence` | Implemented |
| M1-03 automatic rows | `tribol_explicit_rows`, `tribol_linearization`, `tribol_state_linearization` | Implemented |
| M1-04 physical-mass stability | `tribol_explicit_stability`, `tribol_parent_trace_kinematic_vote`, `tribol_coupled_rows_*` | Implemented |
| M1-05 CUDA/HIP native pipeline | `tribol_execution_cuda`, `tribol_execution_hip`, `tribol_execution_native_orders_cuda`, `tribol_execution_native_orders_hip` | Implemented; real-device runs required |
| M1-06 adapters and ownership | `tribol_array_adapter`, `tribol_mfem_adapter_*`, `tribol_mfem_native_contact_*`, `tribol_mfem_native_stability_*` | Implemented; multi-rank runs required |
| M1-07 diagnostics and attempts | `tribol_contact_production_diagnostics`, `tribol_mfem_production_diagnostics_*`, `tribol_mfem_attempt_lifecycle_*` | Implemented; multi-rank runs required |
| M1-08 conformance and invariance | Q1--Q4 convergence, row/candidate/collection-order, allocation, install, quality, and benchmark tests | Implemented; platform matrix required |
| M1-09 application integration | Marbl/Blast/MAPP acceptance jobs | External repositories and inputs required |
| M1-10 release gate | Sharp-notch P20 CPU/MPI/CUDA/HIP reports | External application and physical GPU runs required |

The requirement IDs attached to each test remain authoritative. This table is navigation, not a replacement for
`src/tests/spec/requirements.yaml` or `src/tests/spec/capabilities.yaml`.

## Required Validation Matrix

Run these gates from a clean checkout at the candidate revision.

### Static And Serial

```bash
python3 -m unittest discover -s scripts/spec/tests -v
python3 -m unittest discover -s scripts/quality/tests -v
python3 -m unittest discover -s scripts/benchmarks/tests -v
python3 scripts/quality/check.py all --strict-tools --compiler clang++-19
cmake --build <cpu-build> -j
ctest --test-dir <cpu-build> --output-on-failure
```

The strict quality command requires every tool listed in [quality.md](quality.md). The install consumer and benchmark
orchestration tests are part of the CTest run and must not be replaced by an in-tree include or link check.

### MPI

The following tests must pass with one, two, and four ranks on a host where MPI process launch and local sockets are
allowed:

- `tribol_coupled_rows_*`
- `tribol_mfem_adapter_*`
- `tribol_mfem_native_contact_*`
- `tribol_mfem_native_convergence_*`
- `tribol_mfem_native_stability_*`
- `tribol_mfem_production_diagnostics_*`
- `tribol_mfem_attempt_lifecycle_*`

Use CTest rather than invoking the multi-rank executables directly so the configured launcher, rank count, and test
environment remain part of the evidence.

### Accelerators

Run `tribol_execution_cuda` and `tribol_execution_native_orders_cuda` on a physical NVIDIA GPU. Run
`tribol_execution_hip` and `tribol_execution_native_orders_hip` on a physical AMD GPU. Each run must exercise pointer
residency, device BVH search, overlap generation, native rows, diagnostics, exact derivatives, deterministic scatter,
allocation stability, and retry/restart behavior. Successful compilation or emulation does not satisfy this gate.

### Cross-Version Physics

```bash
python3 scripts/benchmarks/compare.py \
  --current-build <release-build> \
  --suite all \
  --output <cpu-report.json>
```

Run the corresponding `gpu-scaling` suite with `--execution cuda` and `--execution hip` on the matching hardware.
Archive reports only when their provenance names the clean candidate revision, the exact upstream reference revision,
and a non-dirty source tree.

## External Release Gates

M1-09 requires pinned Marbl, Blast, MAPP, and Tribol revisions plus the production input deck. M1-10 additionally
requires the accepted 0--5 microsecond sharp-notch P20 reference data and CPU, MPI, NVIDIA, and AMD execution access.
Until those artifacts are available and every required cell passes:

- keep `DefaultMethod` unchanged;
- do not advertise native curved-face clipping, GPU-aware MPI, or distributed device search;
- do not treat cross-compilation as CUDA or HIP runtime evidence; and
- do not mark Milestone 1 complete.

## Evidence Recording

For every external run, record the repository revision, dirty state, host configuration, compiler, MPI implementation,
accelerator model, driver/runtime version, command, rank count, and output artifact path. Preserve failures as evidence;
do not overwrite them with later passing output.
