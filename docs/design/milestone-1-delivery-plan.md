---
title: Milestone 1 Delivery Plan
status: active
milestone: 1
scope: rewritten-tribol
---

# Milestone 1 Delivery Plan

## Outcome

Milestone 1 delivers production-quality explicit penalty contact on native parent traces. Linear subfacets remain the
search and overlap geometry, while coordinates, velocities, basis values, force scatter, diagnostics, and stability
use the original parent finite-element space.

The production method remains an unnamed policy composition:

```cpp
using NativeExplicitContact = tribol::Method<
    tribol::geometry::ProjectedOverlap<tribol::normal::MeanPlane>,
    tribol::integration::Automatic,
    tribol::constraint::QuadraturePoint,
    tribol::enforcement::Penalty<tribol::stiffness::Constant, tribol::rate::None>,
    tribol::response::Frictionless,
    tribol::formulation::PointwiseTraction,
    tribol::linearization::Exact>;
```

No PR may introduce a production `CommonPlane`, `Mortar`, or application-specific method class. New behavior belongs
in an orthogonal policy, adapter, row consumer, or execution backend.

## Shared Invariants

Every PR in this plan preserves these invariants:

1. One parent-trace/overlap/row pipeline supplies force, work, energy, tangent actions, and stability.
2. Diagnostic quadrature may use a different rule, but it samples the same accepted overlap cells and parent fields.
3. Geometry subfacets never become a displacement or velocity space.
4. Host and device paths use the same geometry, basis, penalty, and row formulas.
5. CUDA and HIP differ only in allocation, launch, scan, sort, and reduction adapters.
6. Every physical row has one deterministic owner and one stable identity.
7. Evaluation allocates no interaction-sized storage after `updateInteractions()`.
8. Capability claims are added only after executable CPU, MPI, CUDA, and HIP evidence exists.

## Dependency Graph

```text
M1-01 contract
  -> M1-02 overlap provenance
  -> M1-03 native row mechanics
     -> M1-04 explicit stability
     -> M1-05 device execution
     -> M1-06 adapters and MPI ownership
        -> M1-07 diagnostics and lifecycle
        -> M1-08 conformance and invariance
           -> M1-09 external application integration
              -> M1-10 release gate and default switch
```

M1-04, M1-05, and the adapter-independent portion of M1-06 may proceed in parallel after M1-03. Each PR must remain
reviewable on its own and must not depend on uncommitted changes from another repository.

## PR M1-01: Native Contact Contract

**Purpose:** Freeze the data model and public syntax before implementation details spread through the codebase.

**Changes:**

- Define trivially copyable parent-trace, parent-field, overlap-cell, constraint-row, and result views.
- Define stable owner, pair, cell, and row identities and document units and signs.
- Validate supported segment, triangle, and quadrilateral trace families and orders.
- Reject unsupported policy combinations and finite-element families during initialization.
- Document geometry-versus-physics spaces and the no-LOR-field-transfer invariant.

**Tests:** Header isolation, aggregate construction, malformed view rejection, compile-time policy validation, and
Q1--Q4 basis partition/reproduction tests.

**Exit:** Downstream PRs can consume the views without changing their public meaning.

## PR M1-02: Overlap Geometry And Provenance

**Purpose:** Produce deterministic LOR overlap cells that retain enough information for native parent evaluation.

**Changes:**

- Generate zero or more interval/polygon cells per candidate subfacet pair.
- Map every overlap vertex to both parent reference spaces.
- Preserve parent face, subfacet, owner, pair, and attempt-local cell identities.
- Report no-overlap, degenerate, orientation, provenance, and mapping outcomes explicitly.
- Deduplicate ghost and neighboring-subfacet copies deterministically.

**Tests:** Analytic 2D/3D intersections, curved parent traces represented by LOR facets, reversed orientation,
near-tangency, duplicate ownership, mapping failure, and geometry-refinement convergence.

**Exit:** Accepted cells are independent of force, quadrature, and adapter code.

## PR M1-03: Automatic Quadrature And Explicit Rows

**Purpose:** Make the native row batch the sole production mechanics representation.

**Changes:**

- Select quadrature exactness from parent field order and traction order, with an explicit validation override.
- Generate variable row and basis counts with count/scan/fill stages.
- Evaluate parent position, velocity, signed basis values, gap, normal rate, and physical weight per row.
- Evaluate the shared penalty law and scatter equal-and-opposite force directly to parent DOFs.
- Provide allocation-free `B`, `B^T`, and `B^T D B` actions from the same rows.

**Tests:** Polynomial integration, parent-field reproduction, resultant/moment conservation, work and energy,
finite-difference tangent checks, variable points per cell, and legacy affine one-point parity.

**Exit:** No native explicit force, derivative, or work calculation reconstructs a centroid contact state.

## PR M1-04: Physical-Mass Stability

**Purpose:** Return a finite conservative contact timestep whenever an explicit spring or dashpot is active.

**Changes:**

- Consume row stiffness and rate derivatives from the shared penalty evaluation.
- Accept physical inverse mass and integrator stability limits from the host.
- Couple all contact pairs sharing parent DOFs before forming the bound.
- Provide a matrix-free spectral estimator as a validation oracle.
- Key cached data by coordinates, topology, interactions, mass, BCs, penalty, and integrator generations.

**Tests:** Scalar and dense eigenvalue oracles, constrained DOFs, signed high-order basis values, stiffness/rate
combinations, stable/unstable timestep sweeps, shared-DOF pairs, empty ranks, and MPI global minimum reduction.

**Exit:** Infinity means only that no explicit contact stiffness or damping is active.

## PR M1-05: CUDA And HIP Native Pipeline

**Purpose:** Keep native explicit contact resident on the selected accelerator for the full contact step.

**Changes:**

- Keep meshes, BVH nodes, candidates, mapped patches, rows, penalty outputs, and force arrays on device.
- Perform BVH search, overlap generation, parent mapping, row count/scan/fill, interpolation, physics, and stability on
  device.
- Scatter through deterministic emit, sort, and segmented-reduce stages.
- Share all mechanics kernels between CUDA and HIP; isolate runtime-specific primitives in backend adapters.
- Reserve all interaction-sized allocations in `updateInteractions()`.

**Tests:** Real-device CPU parity, pointer residency, deterministic replay, candidate/patch/row permutations,
allocation stability, and backend compilation plus runtime execution on NVIDIA and AMD hardware.

**Exit:** Device policy tuples contain no hidden host fallback or per-cycle host row assembly.

## PR M1-06: MFEM And Borrowed-Array Adapters

**Purpose:** Connect host applications to native rows without leaking third-party types into core mechanics.

**Changes:**

- Build identity parent traces for borrowed linear arrays.
- Build Q1--Q4 parent traces and LOR provenance from `mfem::ParMesh` and its coordinate space.
- Exchange native parent coordinates and velocities for owned rows; never transfer physical fields through LOR nodes.
- Return ghost parent residual contributions to owners and assemble into MFEM true DOFs.
- Register physical inverse mass and integrator limits with explicit generation changes.

**Tests:** Linear and curved Q1--Q4 surfaces, nonmatching refinement, multiple pairs, shared true DOFs, one/two/four
ranks, owner-only row assembly, reverse residual exchange, and array/MFEM agreement.

**Exit:** Adapter output is a parent true-DOF residual and a communicator-consistent stability vote.

## PR M1-07: Diagnostics And Attempt Lifecycle

**Purpose:** Make correctness, retries, and production failures observable without a second physics path.

**Changes:**

- Derive force, moment, impulse, traction, spring energy, dashpot dissipation, and contact work from evaluated rows.
- Add independently configurable diagnostic quadrature, automatic by default, over the same overlap cells.
- Report physical maximum penetration and closing rate from diagnostic points.
- Distinguish accepted steps from rejected attempts and classify search, geometry, stability, and host failures.
- Invalidate caches on coordinate, topology, mass, essential-BC, penalty, and integrator generation changes.
- Emit deterministic JSON with explicit local, owned, sum, maximum, and global semantics.

**Tests:** Analytic diagnostic totals, enforcement/diagnostic rule independence, ownership reduction, rejected-attempt
isolation, retry, restart, generation invalidation, and no-contact output.

**Exit:** Diagnostics never reconstruct force or pressure from a different contact discretization.

## PR M1-08: Conformance, Scaling, And Invariance

**Purpose:** Close coverage gaps before application integration.

**Changes:**

- Add analytic and overintegrated Q1--Q4 references in 2D and 3D.
- Add curved/nonmatching and LOR-refinement convergence suites.
- Add row-order, candidate-order, rank-partition, and multiple-pair invariance suites.
- Preserve relevant legacy physics tests through the external compatibility benchmark harness.
- Record tolerances, units, ownership rules, and expected convergence in test names or requirement annotations.

**Tests:** Full CPU suite, MPI at one/two/four ranks, CUDA runtime, HIP runtime, install consumer, header isolation,
formatting, static checks, and cross-version exactness benchmarks.

**Exit:** Every Milestone 1 claim maps to a stable requirement ID and executable evidence.

## PR M1-09: Marbl/Blast/MAPP Integration

**Purpose:** Exercise rewritten Tribol through the production application stack.

**Changes:**

- Add `auto` and explicit constraint/diagnostic quadrature inputs.
- Register native parent fields, physical mass, essential BCs, and integrator limits.
- Propagate contact, hydro, mesh, and search votes independently.
- Implement accepted/rejected attempt ordering, rollback, restart, and adaptation invalidation.
- Pin compatible Tribol/Blast/Marbl/MAPP revisions during review.

**Tests:** Production 2D/3D, CPU/GPU, MPI, multiple-pair, restart, retry, and moving-topology jobs.

**Exit:** The application has no legacy contact data transfer or alternate centroid force path.

## PR M1-10: Release Gate And Default Switch

**Purpose:** Establish the explicit P20 production baseline required before Milestone 2.

**Changes:**

- Run the 0--5 microsecond sharp-notch case with fixed production inputs.
- Archive exactness, force, moment, work, energy, gap, timestep, and timing reports.
- Compare against the accepted legacy and overintegrated references.
- Update `capabilities.yaml`, support documentation, and release notes only after all gates pass.
- Switch `DefaultMethod` to the native automatic-quadrature tuple only in this PR.

**Tests:** All M1-08 gates plus the integrated Marbl acceptance matrix on supported CPU, NVIDIA, and AMD systems.

**Exit:** A stable, reproducible, contact-voted explicit P20 reference is available for Milestone 2 comparisons.

## Review Rules

- A PR changes one architectural layer or one vertical capability, not both unless the public test requires it.
- Tests land with the implementation and identify requirements from `src/tests/spec/requirements.yaml`.
- New helpers replace duplicated formulas before adding another caller.
- Backend-specific code contains no mechanics formulas.
- Temporary compatibility shims are isolated, benchmark-only, and deleted before the default switch.
- External acceptance cannot be inferred from unit tests; record the exact repository revisions and commands used.

## Final Evidence Matrix

| Evidence | CPU | MPI | CUDA | HIP | External App |
| --- | --- | --- | --- | --- | --- |
| Parent mapping and basis reproduction | required | required | required | required | sampled |
| Force, moment, work, and energy | required | required | required | required | required |
| Contact-stiffness and kinematic votes | required | required | required | required | required |
| Determinism and ownership | required | required | required | required | required |
| Retry, restart, and invalidation | required | required | required | required | required |
| Allocation stability | required | required | required | required | profiled |
| Sharp-notch P20 baseline | reference | required | required | required | release gate |

Milestone 1 is not complete while an evidence cell marked `required` is missing. A cross-compiled HIP build is useful,
but only a runtime test on AMD hardware satisfies the HIP evidence column.
