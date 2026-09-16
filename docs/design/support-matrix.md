---
title: Tested Support Matrix
status: accepted
normative: true
---

# Tested Support Matrix

The authoritative tuple-by-tuple data is `src/tests/spec/capabilities.yaml`. This page records cross-cutting limits that
are easy to misstate in prose.

Fresh builds set `TRIBOL_ENABLE_LEGACY=OFF`; named-method sources and examples are available only as an explicit
migration oracle.

| Area | Supported now | Not claimed |
| --- | --- | --- |
| Dimensions | 2D and 3D as listed per tuple | Other dimensions |
| Surface topology | Linear segments, triangles, and quadrilaterals | Volume contact elements |
| High order | MFEM boundary tessellation into linear subelements | Native high-order kernels or exact curved overlap |
| Search | `CartesianProduct`, `Grid`, `Bvh`; pointwise self-contact filtering | Distributed/scalable broad phase |
| Host execution | `Sequential` and reproducible `Deterministic` for all manifest tuples | OpenMP policy |
| Device execution | Full default-method `Contact` evaluation with `execution::Cuda` | Other policy tuples, HIP, device-side search |
| Differentiation | In-tree `Exact` directional actions and assembled Jacobians on frozen candidates | Enzyme and finite differences as production backends |
| MPI | Default MFEM path, tested at 1/2/4 ranks | Non-default MFEM state mapping, raw-array MPI, scalable decomposition |
| Host adapters | Borrowed arrays and MFEM `ParMesh`/true-DOF residual mapping | Other host frameworks |

## MPI Implementation Note

`MfemSurface` gathers every selected boundary subelement and coordinate sample to every rank with `MPI_Allgatherv`.
Each rank evaluates the replicated contact problem; dual transpose accumulation returns only locally owned
contributions before MFEM true-DOF assembly. This design prioritizes a simple partition-invariant contract. It must be
replaced by distributed ownership, ghosting, and redecomposition before large-scale scalability is claimed.

## CUDA Implementation Note

CUDA dispatch is accepted only for `DefaultMethod`. `updateInteractions()` sizes reusable host/device workspaces;
subsequent evaluation does not allocate. Search still runs on the host. The CUDA conformance test compares the complete
result with sequential execution on a physical device.
