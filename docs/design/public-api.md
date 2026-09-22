---
title: Public Contact API
status: accepted
normative: true
---

# Public Contact API

## Entry Points

Include `tribol/Tribol.hpp` for the dependency-free interface. Use one of three construction paths:

| Host data | Entry point | Ownership |
| --- | --- | --- |
| Existing Tribol views | `tribol::Contact<Method, Search, Execution>` | Borrows both `SurfaceMeshView` objects. |
| Plain contiguous arrays | `tribol::array::UniformSurface` and `tribol::array::makeContact` | Owns derived topology; borrows coordinate and connectivity arrays. |
| MFEM | `tribol::mfem::makeContact` from `MfemContact.hpp` | Owns extracted surfaces; borrows the `ParMesh` and coordinate space. |

`Contact<>` selects `DefaultMethod`, `search::CartesianProduct`, and `execution::Sequential`. Choose template arguments
only when changing mechanics, search, or execution behavior.

## Lifecycle

```cpp
tribol::Contact<> contact({mortar_surface, nonmortar_surface});
contact.updateInteractions();
const tribol::ContactResultView result = contact.evaluate();
```

- `updateInteractions()` searches and prepares all interaction-sized storage.
- `updateGeometry(new_surfaces)` accepts coordinate-only changes with identical topology and keeps candidate pairs
  frozen. This is the required state for consistent coordinate derivatives.
- `rebuildGeometry(new_surfaces)` accepts topology changes, clears candidate pairs, and requires a new search.
- `evaluate(state)` clears and fills contact-owned result storage.
- `evaluateNodalKinematics()` stages gap, weighted gap, and tributary area before a host-defined pressure law is known;
  it is available for nodal variational methods.
- `addResidual(state, output)` accumulates into caller-owned residual arrays.
- `applyCoordinateDerivative(state, direction, output)` adds an exact directional Jacobian action computed by in-tree
  tangent propagation; it does not perturb the input geometry.
- `applyMultiplierDerivative()` and `applyExternalPressureDerivative()` apply the supported state-coupling blocks.
- `assembleCoordinateJacobian(state)` returns a row-major dense matrix view.
- `assembleSystemJacobian(state)` and its CSR counterpart return the primal/multiplier block system when applicable.

Exact derivatives hold the candidate set fixed and differentiate the active primal geometry/clipping branch. Contact
activation and clipping transitions are nonsmooth; callers should rebuild interactions after crossing such a boundary.

Evaluation and linearization throw if interactions have not been prepared. The contact object is intentionally
non-copyable and non-movable because its views and workspaces have stable ownership relationships.

## Device Workflow

The supported CUDA and HIP specializations are intentionally exact:

```cpp
using DeviceExecution = tribol::execution::Cuda;  // or tribol::execution::Hip
using DeviceContact =
    tribol::Contact<tribol::DefaultMethod, tribol::search::Bvh, DeviceExecution>;

DeviceContact contact({mortar_surface, nonmortar_surface}, options);
contact.updateInteractions();
const tribol::ContactResultView device_result = contact.evaluateDevice(state);
```

Construction copies mesh coordinates and topology to device-owned storage. `updateInteractions()` builds and queries a
Morton-ordered device BVH, canonicalizes device candidate pairs, and prepares all patch, physics, reduction, diagnostic,
and result buffers. `evaluateDevice()` performs projected-overlap generation, pointwise penalty physics, timestep
voting, and deterministic two-stage scatter on the device. Its array and field members are device pointers; only the
small `EvaluationSummary` metadata is synchronized to the host.

`devicePipelineView()` exposes trivially copyable device views of the resident surfaces, candidates, generated patches,
and current results for downstream kernels. `cudaPipelineView()` and `hipPipelineView()` preserve explicit backend
spelling when useful. These pointers are owned by the contact object. Candidate and
interaction-sized pointers remain valid until `updateInteractions()`, `setInteractions()`, `rebuildGeometry()`, or
destruction; result contents are overwritten by the next evaluation. `updateGeometry()` copies coordinates into the
existing device allocations and preserves the frozen candidate set. A pipeline result retains the geometry and
interaction versions of the evaluation that produced it, so comparing those values with the contact object's current
versions identifies stale payloads after geometry or interaction updates.

Use `evaluate(state)` when host-readable result arrays are required; it runs the device calculation and explicitly
downloads contact-owned mirrors. Calling `interactions()` similarly downloads a host candidate mirror. Exact
matrix-free coordinate derivative physics runs on the device. Dense Jacobian assembly currently launches one device
directional action per column from host orchestration and returns host-owned matrix storage.

`DeviceExecution` is `tribol::execution::Cuda` in a CUDA build or `tribol::execution::Hip` in a ROCm build. The public
lifecycle and pointer-validity rules are identical. Device mechanics are shared RAJA kernels; backend adapters provide
the RAJA resource plus CUB or hipCUB sort, scan, run-length, and reduction primitives.

## Search And Self-Contact

Cartesian-product, grid, and BVH search parameters provide both absolute `expansion` and element-relative
`proximity_scale`. Each element box is inflated by `proximity_scale * longest_element_extent + expansion`.
`search::legacyProximityScale` is `4.0`, matching the legacy default. `Contact` folds residual gap into the absolute
expansion so search cannot omit a pair solely because residual-gap activation makes it active.

Constructing `Contact` from one surface enables self-contact. It removes duplicate and adjacent pairs and, by default,
rejects a projected pair when its effective gap is more negative than `0.95` times the smaller element thickness. Set
`options.reject_excessive_self_penetration = false` to disable this thin-structure safeguard. Evaluation requires
positive element-thickness state on both sides when the safeguard is enabled.

## Inputs

`ContactStateView` is sparse by design: leave unused fields empty. `MethodTraits<Method>::capabilities` tells a host
which fields are required:

- velocity fields for rate penalties and viscous tangential response;
- reference coordinates for tied response;
- element thickness and material modulus for material-scaled stiffness;
- mortar multipliers for multiplier enforcement;
- mortar nodal pressure for external-pressure enforcement;
- optionally, matching external potential-density and pressure-tangent arrays for conservative energy and a coordinate
  Jacobian consistent with a host-defined law.

For an external pressure law, first call `evaluateNodalKinematics()`, evaluate the law at the returned gap, and pass its
potential density, pressure, and `dp/dg` fields in `ContactStateView`. Passing pressure alone is a supported residual-only
mode: pressure is held fixed during coordinate differentiation and `summary.energy` is zero. Potential and tangent must
be provided together.

Fields are borrowed for the duration of the call and use the entity/component layout recorded by `FieldView`.

## Results

`ContactResultView` contains borrowed views into the contact object:

| Member | Meaning |
| --- | --- |
| `mortar_force`, `nonmortar_force` | Nodal force residuals on the two extracted surfaces. |
| `constraint_residual` | Mortar nodal constraint equations for multiplier methods. |
| `gap`, `weighted_gap`, `tributary_area` | Mortar nodal diagnostics. |
| `mortar_weights` | Dense mortar-to-nonmortar diagnostic weight table. |
| `mortar_mass_weights` | Dense mortar-to-mortar mass weight table used with `mortar_weights` by mortar projections. |
| `quadrature_gap`, `quadrature_pressure` | Active quadrature-point diagnostics. |
| `summary.energy` | Conservative potential contribution when the method produces one. |
| `summary.timestep_vote` | Stable-step recommendation; infinity means no finite vote. |
| `summary.active_interactions`, `summary.quadrature_points` | Work completed by the evaluation. |
| `geometry_version`, `interaction_version` | Host-visible lifecycle versions. |

Only outputs listed for a method in `src/tests/spec/capabilities.yaml` are contractual. Result views are invalidated by
the next evaluation, a geometry rebuild, or destruction of the contact object.

## Diagnostics

`diagnostics::writeJson()` and `diagnostics::writeVtk()` write individual snapshots. For time-dependent runs,
`diagnostics::writeScheduled<Method>()` applies a cycle interval, creates the output directory, adds stable cycle and
optional rank suffixes, and can emit JSON state, surface VTK, and projected-overlap VTK files. Overlap files carry the
signed contact gap and both source element indices as cell data.

## MFEM Workflow

`MfemContact` takes a `ParMesh`, its nodal coordinate `ParGridFunction`, and two boundary-attribute lists. It maps every
host tuple in the capability manifest into coordinate true-DOF residuals, accepts velocity/reference/material,
multiplier, and external-pressure-law state, and provides matrix-free and assembled distributed Jacobians.
`evaluateNodalKinematics(gap, weighted_gap, tributary_area)` stages external-pressure inputs directly in a scalar MFEM
space. `restrictPrimal()` and `addDualTranspose()` expose the exact interpolation restriction and its weighted transpose
for custom host integration.

High-order boundary elements are sampled into linear segments, triangles, or quadrilaterals. This preserves curved
coordinate interpolation at sample nodes, but it is tessellation rather than native high-order contact integration.
Pass `mfem::SurfaceDiscretization{.subdivision_factor = n}` to `mfem::makeContact` to select an explicit subdivision;
zero uses the coordinate finite-element order.
