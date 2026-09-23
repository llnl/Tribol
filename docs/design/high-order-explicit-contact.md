---
title: Native Parent-Trace Explicit Contact
status: accepted-design
normative: true
milestone: 1
---

# Native Parent-Trace Explicit Contact

## Purpose

Milestone 1 adds an accurate higher-order explicit penalty path without creating a named contact method. The path
retains piecewise-linear projected-overlap geometry while evaluating physical fields and scattering forces directly in
the original parent finite-element trace space.

The implementation has one data pipeline:

```text
ParentTraceView
  -> OverlapCellBatch
  -> ConstraintRowBatch
  -> explicit force, energy, work, diagnostics, B, B^T, contact tangent, and stability vote
```

No consumer may reconstruct geometry, quadrature, parent basis values, normals, weights, or penalty stiffness from a
different representation.

## Geometry And Physical Spaces

The geometry space and physical field space are intentionally different:

- Linear subfacets provide broad-phase bounds, projected-overlap construction, clipping, and the integration domain.
- Every subfacet vertex records its reference coordinate in one original parent face.
- An overlap point maps through the subfacet parameterization to each parent reference space.
- Coordinates, velocity, and basis functions are evaluated in the parent trace space at that mapped point.
- Force and impulse are scattered directly to parent trace DOFs.

The subfacet mesh is not a displacement space. Interpolating a physical field only at subfacet vertices and then
scattering through those vertices is not the native parent-trace path.

## Units And Signs

The signed gap is positive in separation. A row normal consistently maps nonmortar motion relative to mortar motion.
The exact sign convention is tested through the `B`/`B^T` adjoint identity and equal-and-opposite force checks.

Each constraint row stores its physical integration weight exactly once. Penalty-law values use these units:

| Value | Meaning |
| --- | --- |
| `pressure_stiffness` | Normal traction per unit gap before quadrature integration. |
| `pressure` | Normal traction returned by the penalty law. |
| `integration_weight` | Quadrature weight including overlap measure. |
| `row_stiffness` | `integration_weight * pressure_stiffness`. |
| `row_force` | Integrated row force or impulse contribution. |

No layer may multiply by overlap measure or quadrature weight a second time.

## Parent Trace Contract

A kernel-facing parent trace view contains only borrowed arrays and scalar metadata. It records:

- spatial dimension;
- parent face topology and polynomial order;
- parent face ownership and stable identity;
- parent local DOF offsets and indices;
- geometry-subfacet topology and connectivity;
- the parent face for each subfacet; and
- parent reference coordinates for every subfacet vertex.

The built-in parent basis supports the explicitly tested segment, triangle, and quadrilateral trace families. An
adapter must reject an unsupported finite-element family rather than silently falling back to subfacet interpolation.

## Overlap Cell Contract

An overlap cell is an interval in 2D or polygon in 3D. Each cell records:

- the two parent faces and geometry subfacets;
- consistently oriented common-plane normal;
- projected cell vertices and their reference coordinates in both parent faces;
- physical measure;
- owner rank, contact-pair identity, and stable attempt-local identity; and
- an explicit geometry status.

Geometry status distinguishes no overlap from invalid mapping, inconsistent orientation, excessive mapping error, and
degenerate overlap. A failed parent map never switches to a lower-order field path.

Duplicate physical cells created by neighboring subfacets or ghosts have one deterministic owner.

## Constraint Row Contract

The row batch uses segmented structure-of-arrays storage. A row records:

- contact-pair, overlap-cell, and row identity;
- owner rank;
- normal and integration weight;
- reference coordinate in each parent face;
- parent DOF indices and basis values on each side;
- signed gap and optional normal rate; and
- shared penalty-law inputs and outputs.

Variable row and basis counts use count, prefix-scan, and fill stages. Production storage is not tied to a fixed parent
order or a fixed number of quadrature points. The row view is trivially copyable and identical for host and device
mechanics.

Constraint and diagnostic quadrature are independent policies built from the same overlap cells. Diagnostic points do
not participate in force or stability assembly.

The built-in automatic rule targets polynomial exactness equal to the maximum parent-field order plus the traction
order. When no separate traction order is supplied, the maximum parent-field order is used for both terms. Segments
use Gauss-Legendre quadrature and polygons use a fan triangulation with a tensor-product Duffy rule. The built-in
tables support exactness through degree eleven; requests above that limit are rejected rather than silently clipped.

## Operators

The shared row batch defines:

```text
B v                    signed normal relative velocity per row
B^T lambda             equal-and-opposite parent-space dual impulse
K_contact v = B^T D B v
```

`D` is assembled from the derivative of the shared penalty law and the row integration weight. The contact tangent is
an operator action; dense matrices are test oracles only.

Explicit force, contact work, contact energy, the contact tangent, and the stability vote use the same active rows and
penalty state.

## Explicit Stability Contract

The production explicit penalty path returns a finite contact-stiffness vote whenever explicit contact stiffness is
active. The vote uses:

- the shared contact tangent;
- application-supplied physical mass or inertia information;
- the application integrator's stability factor; and
- coupled accumulation for all contact pairs sharing parent velocity DOFs.

A conservative row-sum bound is permitted only for mass assumptions under which it is proven safe. Other supported
configurations use the validated matrix-free spectral path. Missing mass or integrator information is an unsupported
configuration, not a reason to return a dimensional `1 / sqrt(stiffness)` estimate.

Infinity means that no explicit contact stiffness is active. It does not replace hydro, mesh, search, or other
application timestep restrictions.

## Lifecycle

The contact object owns overlap cells, rows, intermediates, and results.

1. Construction validates parent trace and method parameters.
2. `updateInteractions()` searches, canonicalizes candidates, prepares overlap cells and row capacity, and advances
   the interaction version.
3. `updateGeometry()` preserves frozen candidates, refreshes overlap cells and rows, and advances the geometry version.
4. `rebuildGeometry()` replaces topology and provenance, invalidates candidates, and requires another
   `updateInteractions()`.
5. Explicit evaluation and stability actions perform no allocation.

Mass, geometry, interaction, penalty, and integrator generations invalidate cached stability information.

## Parallel And Device Contract

Each physical constraint row has one owner. Other ranks retain only parent and field ghosts required by owned rows.
Forces and `B^T` contributions are summed through the parent true-DOF map.

Host, OpenMP, CUDA, and HIP execute the same mechanics operations. Execution backends may provide allocation,
prefix-scan, sort, and reduction adapters, but may not fork geometry, basis, penalty, or operator formulas. Device
execution keeps parent traces, cells, rows, state, and results resident and uses deterministic parent-DOF scatter.

## Public Product Gate

This design does not itself advertise native high-order support. The capability manifest is updated only after
component, conformance, MFEM, MPI, CUDA, HIP, and integrated Marbl evidence passes. The existing default method remains
unchanged until the complete Milestone 1 acceptance gate succeeds.
