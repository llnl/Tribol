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
- `addResidual(state, output)` accumulates into caller-owned residual arrays.
- `applyCoordinateDerivative(state, direction, output)` adds an exact directional Jacobian action computed by in-tree
  tangent propagation; it does not perturb the input geometry.
- `assembleCoordinateJacobian(state)` returns a row-major dense matrix view.

Exact derivatives hold the candidate set fixed and differentiate the active primal geometry/clipping branch. Contact
activation and clipping transitions are nonsmooth; callers should rebuild interactions after crossing such a boundary.

Evaluation and linearization throw if interactions have not been prepared. The contact object is intentionally
non-copyable and non-movable because its views and workspaces have stable ownership relationships.

## Inputs

`ContactStateView` is sparse by design: leave unused fields empty. `MethodTraits<Method>::capabilities` tells a host
which fields are required:

- velocity fields for rate penalties and viscous tangential response;
- reference coordinates for tied response;
- element thickness and material modulus for material-scaled stiffness;
- mortar multipliers for multiplier enforcement;
- mortar nodal pressure for external-pressure enforcement.

Fields are borrowed for the duration of the call and use the entity/component layout recorded by `FieldView`.

## Results

`ContactResultView` contains borrowed views into the contact object:

| Member | Meaning |
| --- | --- |
| `mortar_force`, `nonmortar_force` | Nodal force residuals on the two extracted surfaces. |
| `constraint_residual` | Mortar nodal constraint equations for multiplier methods. |
| `gap`, `weighted_gap`, `tributary_area` | Mortar nodal diagnostics. |
| `mortar_weights` | Dense mortar-to-nonmortar diagnostic weight table. |
| `quadrature_gap`, `quadrature_pressure` | Active quadrature-point diagnostics. |
| `summary.energy` | Conservative potential contribution when the method produces one. |
| `summary.timestep_vote` | Stable-step recommendation; infinity means no finite vote. |
| `summary.active_interactions`, `summary.quadrature_points` | Work completed by the evaluation. |
| `geometry_version`, `interaction_version` | Host-visible lifecycle versions. |

Only outputs listed for a method in `src/tests/spec/capabilities.yaml` are contractual. Result views are invalidated by
the next evaluation, a geometry rebuild, or destruction of the contact object.

## MFEM Workflow

`MfemContact` takes a `ParMesh`, its nodal coordinate `ParGridFunction`, and two boundary-attribute lists. The current
adapter maps only the default method residual directly into the coordinate true-DOF space. `restrictPrimal()` and
`addDualTranspose()` expose the exact interpolation restriction and its weighted transpose for host integration.

High-order boundary elements are sampled into linear segments, triangles, or quadrilaterals. This preserves curved
coordinate interpolation at sample nodes, but it is tessellation rather than native high-order contact integration.
