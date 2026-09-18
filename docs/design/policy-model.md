---
title: Contact Policy Model
status: accepted
normative: true
---

# Contact Policy Model

## Composition Axes

`tribol::Method` composes these independent choices:

| Axis | Built-in vocabulary | Responsibility |
| --- | --- | --- |
| Geometry | `ProjectedOverlap<Normal>`, `ConformingOverlap` | Build the interaction manifold. |
| Normal | `MeanPlane`, `MortarSurface` | Define orientation and signed gap direction. |
| Integration | `Centroid`, `Polygon<Order>`, `Face<Order>`, `SmoothedSegment<Points>` | Place and weight evaluation points. |
| Constraint | `Pointwise`, `Nodal<Basis>`, `QuadraturePoint` | Represent the discrete gap. |
| Basis | `Primal`, `Dual` | Interpolate/test nodal constraints. |
| Enforcement | `Penalty<Stiffness,Rate>`, `LagrangeMultiplier`, `ExternalPressure`, `None` | Convert constraints into equations or diagnostics. |
| Response | `Frictionless`, `ViscousTangential`, `TiedNormal`, `TiedFull` | Select normal/tangential kinematics and traction behavior. |
| Formulation | `PointwiseTraction`, `WeightedWeakForm`, `Variational`, `DiagnosticWeights` | Define residual/constraint assembly semantics. |
| Linearization | `Exact` | Propagate exact directional tangents through reusable contact stages and assemble dense Jacobians from exact actions. |

Search, surface representation, execution, memory, accumulation, and host adaptation configure the contact object but do
not define the mechanics method.

## Compatibility

`SupportedMethod<T>` is the public compile-time contract. Built-in compatibility rules accept the tuples listed by the
capability manifest. `MethodCompatibility<T>` is the customization point for a deliberately supported external method
definition; specializing it does not exempt component policies from their concepts.

Structural incompatibilities are compile-time errors. Runtime validation is reserved for values such as negative
stiffness, invalid mesh dimensions, missing required fields, and unsupported input topology. The legacy-named `Analytic`
and `Enzyme` tags remain reserved and unsupported; production differentiation is provided by `Exact` without Enzyme.

`Exact` follows the active geometry, clipping, and contact branches selected by the primal state. At activation and
clipping boundaries the contact map is nonsmooth, so the reported tangent is the derivative of the selected branch.

`Polygon<Order>` supports orders 1, 2, and 4; order 4 exactly integrates products of bilinear face bases on planar
overlap polygons. `Face<Order>` supports orders 1 and 2. `SmoothedSegment<Points>` uses one to three
Gauss points and reproduces the legacy C1 endpoint-bound map for 2D segment variational penalties. It is deliberately
rejected for multiplier and external-pressure enforcement. A policy type existing in a header does not by itself make
every composition containing it supported; the capability manifest and `SupportedMethod` must agree.

## Legacy Parity Without Legacy Types

Legacy names appear only in migration documentation and test metadata:

- mean-plane projected overlap + centroid + pointwise penalty reproduces common-plane behavior;
- projected overlap + polygon integration + nodal primal constraint + multiplier reproduces single mortar;
- conforming overlap + face integration + nodal primal constraint + multiplier reproduces aligned mortar;
- projected overlap + diagnostic formulation emits mortar weights;
- projected overlap + polygon integration + nodal or quadrature constraints + variational enforcement reproduces energy
  mortar behavior;
- projected overlap + smoothed-segment integration + nodal or quadrature penalty reproduces legacy endpoint-smoothed
  Energy Mortar behavior.

The architecture contains no named method implementation or switch over a method enum. Cross-version comparisons build
the benchmark-only legacy adapter against an external Tribol package.
