---
title: Tested Support Matrix
status: accepted
normative: true
---

# Tested Support Matrix

The authoritative tuple-by-tuple data is `src/tests/spec/capabilities.yaml`. This page records cross-cutting limits that
are easy to misstate in prose.

Named-method sources are absent from this repository. `scripts/benchmarks/compare.py` builds a separate reference
adapter against a supplied Tribol installation or a fresh checkout of upstream `develop`.

| Area | Supported now | Not claimed |
| --- | --- | --- |
| Dimensions | 2D and 3D as listed per tuple | Other dimensions |
| Surface topology | Linear segments, triangles, and quadrilaterals | Volume contact elements |
| High order | MFEM boundary tessellation into linear subelements | Native high-order kernels or exact curved overlap |
| Search | `CartesianProduct`, `Grid`, `Bvh`; absolute/relative inflation; pointwise self-contact filtering and penetration rejection; rank-box distributed MFEM broad phase | Fully device-side search |
| Host execution | `Sequential`, reproducible `Deterministic`, and two-pass, thread-local `OpenMP` for all manifest tuples | Other host runtimes |
| Device execution | Full default-method `Contact` evaluation with `execution::Cuda` | Other policy tuples, HIP, device-side search |
| Differentiation | In-tree nested-forward `Exact` energy gradients, directional Hessian actions, state actions, and assembled Jacobians on frozen candidates | Enzyme and finite differences as production backends |
| MPI | Distributed MFEM ownership/ghost exchange, tested at 1/2/4 ranks | Raw-array MPI |
| Host adapters | Borrowed arrays and MFEM `ParMesh`/true-DOF residual mapping | Other host frameworks |

Variational nodal penalty first assembles weighted gaps and tributary areas globally, then evaluates the nodal pressure
law, and finally differentiates pair energy. OpenMP preserves that ordering with a parallel kinematics reduction and a
parallel force-gradient pass; it does not substitute pair-local nodal gaps.

`SmoothedSegment<Points>` is supported for one to three Gauss points, 2D segment surfaces, and variational nodal or
quadrature penalty only. Its endpoint width is in `[0, 0.5)`, and its geometry-dependent smoothed bounds participate in
exact derivatives.

External-pressure variational contact accepts pressure alone for a residual with pressure held fixed and zero reported
potential. Supplying potential density, pressure, and pressure tangent together yields conservative energy and a
coordinate Jacobian consistent with the staged host law.

## MPI Implementation Note

`MfemSurface` keeps mortar elements on their owning rank. It all-gathers only one expanded bounding box per rank, sends
nonmortar element packets to overlapping mortar-rank boxes with `MPI_Alltoallv`, and reverses ghost residual packets to
their owners before MFEM true-DOF assembly. A supplied-pair search intentionally requests replicated geometry because
its indices refer to the global concatenated surface; broad-phase search policies use the scalable path. MFEM contact
evaluation and summary production are collective operations. `updateGeometry()` preserves the current ghost topology
for frozen-neighbor steps, while collective `rebuildGeometry()` recomputes rank neighbors after larger motion and
invalidates candidate pairs until `updateInteractions()` is called.

## CUDA Implementation Note

CUDA dispatch is accepted only for `DefaultMethod`. `updateInteractions()` sizes reusable host/device workspaces;
subsequent evaluation does not allocate. Search still runs on the host. The CUDA conformance test compares the complete
result with sequential execution on a physical device.
