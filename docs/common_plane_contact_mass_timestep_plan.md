# Plan: Common Plane Contact Mass-Based Timestep Limit

## Summary

Refactor the `COMMON_PLANE + PENALTY` timestep stability limit from a host-`dt` scaling into an absolute contact timestep estimate based on host-provided **lumped nodal masses**. Implement the core registration path in `tribol.hpp` and add an MFEM convenience wrapper in `mfem_tribol.hpp`.

This work should be implemented **after** higher-order Common Plane quadrature is added, since both changes touch overlap integration, contact weighting, and how overlap contributions are accumulated.

## Key Changes

- Replace the current CFL-like stability vote with an absolute contact timestep for active Common Plane penalty contact:
  \[
  \Delta t_{\text{contact}} = \alpha \, 2 \sqrt{\frac{m_{\text{pair}}}{k_{\text{pair}}}}
  \]
  where `alpha = timestep_scale`, `k_pair = k_c * overlap_area`, and `k_c` is the existing interface stiffness per area.
- Compute side effective mass from **lumped nodal masses** using the same contact weights already used to distribute force:
  \[
  m_{i,\text{eff}} = \left(\sum_a \frac{\phi_{i,a}^2}{m_{i,a}}\right)^{-1}
  \]
  using the overlap-based weak-form weights for each side.
- Combine the two sides into a pair effective mass with harmonic reduction:
  \[
  m_{\text{pair}} = \left(\frac{1}{m_{1,\text{eff}}} + \frac{1}{m_{2,\text{eff}}}\right)^{-1}
  \]
- Do not add extra overlap-area fraction scaling for mass in v1. The overlap localization is handled through the contact weights.
- Keep the existing gap-based and velocity-projection timestep checks unchanged. Only the stability-limit bucket changes from a `dt_in` scaler to an absolute contact vote.

## Public API / Interface Changes

- Add a core registration function in `tribol.hpp` / `tribol.cpp`:
  - `registerLumpedNodalMass( IndexT mesh_id, const RealT* lumped_nodal_mass )`
- Add storage and validation for lumped nodal mass in mesh nodal data.
  - Scalar field, length = number of contact surface nodes on the registered mesh.
  - All values must be strictly positive.
- Add an MFEM convenience wrapper in `mfem_tribol.hpp` / `mfem_tribol.cpp`:
  - `registerMfemLumpedNodalMass( IndexT cs_id, const mfem::ParGridFunction& lumped_mass )`
- MFEM helper behavior:
  - Accept a scalar nodal `ParGridFunction` on the parent mesh.
  - Transfer/project it onto the Tribol contact surface nodes using the existing parent-to-submesh machinery.
- Validation rule:
  - If `enableTimestepStabilityLimits()` is on for a Common Plane penalty scheme, lumped nodal mass must be registered and valid on both meshes.
  - Missing or nonpositive nodal mass makes the coupling scheme invalid during `init()`, with a warning, matching existing invalid-enforcement-data behavior.

## Implementation Notes

- Reuse the existing overlap weights for the effective-mass calculation so the inertial reduction follows the same contact discretization as the force calculation.
- Use the existing stiffness path:
  - `KINEMATIC_CONSTANT`: `k_i = pen_scale * const_penalty`
  - `KINEMATIC_ELEMENT`: `k_i = pen_scale * mat_mod / thickness`
  - `k_c = (k1 * k2) / (k1 + k2)`
  - `k_pair = k_c * plane.m_area`
- Guard all effective-mass calculations against zero or invalid mass before division.
- Keep the existing invalid-thickness early return in Common Plane penalty.
- Do not support consistent mass in v1 and do not implement internal mass lumping. The host is responsible for providing already-lumped nodal masses.
- When higher-order quadrature lands, generalize the single-point weight usage to a quadrature-point accumulation rather than reintroducing a single-point assumption.

## Test Plan

- Add/update core tests for Common Plane penalty:
  - Constant-penalty contact with registered nodal masses: verify absolute contact timestep matches `2*sqrt(m_pair/k_pair)` times `timestep_scale`.
  - Element-penalty contact with registered nodal masses: same verification using element-based stiffness.
  - Stationary contact with stability limits enabled: verify the absolute contact limit is computed from mass/stiffness and is not a function of incoming `dt`.
  - Missing nodal mass with stability limits enabled: verify the coupling scheme is skipped as invalid.
  - Nonpositive nodal mass with stability limits enabled: verify invalid scheme and no contact response.
- Add MFEM-path coverage:
  - Register MFEM lumped nodal mass through the new helper and verify the resulting timestep vote matches the core formula on a simple contact case.
- Re-run existing `tribol_timestep_vote` and `tribol_common_plane_penalty` suites to ensure the penetration-based votes still behave the same.

## Assumptions

- V1 supports only **lumped nodal masses** as inertial input.
- V1 computes contact mass from contact weights and does not apply extra overlap-fraction scaling.
- V1 adds both the core Tribol API support and an MFEM convenience registration helper.
- `timestep_scale` remains the user-facing safety factor and multiplies the absolute contact critical timestep.
- The stability constant is `2`, corresponding to the lumped-mass single-DOF spring stability bound.
