---
title: Policy-Composed Architecture
status: accepted
normative: true
---

# Policy-Composed Architecture

## Goals

Tribol provides reusable contact mechanics components, an ergonomic MFEM path, and a borrowed-view path for other
hosts. A host constructs an object, updates interaction topology explicitly, evaluates results, and applies or assembles
coordinate derivatives. The capability manifest states which method, execution, and parallel combinations are tested.

## Dependency Layers

Dependencies flow downward only:

1. **Public API** owns lifecycle, validation, defaults, results, and stable error vocabulary.
2. **Adapters** translate MFEM or host arrays into core views and map primal/dual results back to host spaces.
3. **Adapter orchestration** handles restrictions, transpose restrictions, ownership, and MPI.
4. **Contact engine** sequences search, geometry, integration, constraint, enforcement, response, and linearization.
5. **Policies and components** implement one operation each and expose constrained, device-callable contracts.
6. **Core** provides scalar/index types, layouts, views, buffers, versions, and small math primitives.

`core` and `method` may not include third-party headers. Adapters may depend on core; core never depends on adapters.

## Data Ownership

- Mesh and field input views are borrowed and document the lifetime they require.
- A contact object owns candidate pairs, evaluation workspaces, and result storage.
- Kernel views contain pointers and scalar metadata only.
- Evaluation performs no allocation after a topology or geometry rebuild has prepared storage.
- Returned result views borrow storage owned by the contact object and remain valid until the next evaluation, geometry
  rebuild, or destruction of that object.

## Lifecycle

1. Construct from a host adapter or `SurfacePairView` and validated options.
2. Call `updateInteractions()` after construction and whenever search must be repeated.
3. Call `updateGeometry()` for coordinate-only changes that intentionally preserve the frozen candidate pairs.
4. Call `rebuildGeometry()` when topology changes; this invalidates candidates and requires `updateInteractions()`.
5. Call `evaluate()` or `addResidual()` with the fields required by `MethodTraits`.
6. Apply a matrix-free coordinate derivative or assemble the coordinate Jacobian on the same frozen interactions.

Every `ContactResultView` records geometry and interaction versions. Version comparison is available to hosts, but the
view itself does not perform stale-use checks.

## Parallel Model

- Execution policies select sequential, deterministic, OpenMP, or CUDA dispatch without changing mechanics policies.
- Sequential, deterministic, and OpenMP execution support all advertised host methods. Nodal variational OpenMP uses a
  thread-local kinematics pass, global nodal reduction, and thread-local energy-gradient pass. CUDA currently supports only
  `DefaultMethod` and is rejected at compile time for other tuples.
- Communication is host-side orchestration around device-capable local kernels.
- MPI ownership and halo behavior live in adapters/assembly, not geometric kernels.
- The MFEM adapter assigns evaluation to mortar-owning ranks, exchanges nonmortar elements only between ranks whose
  expanded spatial bounds overlap, and returns ghost residuals to their owning ranks.
- MPI partition invariance is currently demonstrated for the default MFEM path. CUDA parity is demonstrated for the
  complete default core `Contact` evaluation on a physical device.

## Extension Rule

Add a primitive policy when one stage changes. Add a new engine stage only when existing stages cannot express the
operation without coupling unrelated responsibilities. Never add a class whose purpose is to recreate a named legacy
method.
