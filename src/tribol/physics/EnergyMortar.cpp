#include "tribol/physics/EnergyMortar.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"
#include "tribol/physics/PairEvaluation.hpp"
#include "tribol/common/BasicTypes.hpp"

#include <stdexcept>

#include "axom/slic.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/integ/BasisFunction.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iterator>
#include <vector>

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

namespace {

using LinearEdgeBasis = BasisFunction<EDGE, LINEAR_LAGRANGE>;
// View each endpoint as one [node][physical component] row so the owning coordinates retain the basis layout.
using LinearEdgeCoords = MdArray<RealT, LinearEdgeBasis::node_count, LinearEdgeBasis::spatial_dimension>;

// Keep only inactive configuration in the Enzyme parameter block; coordinates arrive through the active x argument.
template <typename QuadratureRule>
struct KernelParams {
  QuadratureRule quadrature_rule;
  RealT k;
  RealT residual_gap;
};

TRIBOL_ENZYME_INLINE RealT effective_gap( RealT gap_normal, RealT normal_cosine, RealT residual_gap )
{
  return gap_normal * normal_cosine - residual_gap;
}

// Compute a unit normal vector for the line segment from coord1 to coord2
TRIBOL_ENZYME_INLINE void find_normal( const RealT* coord1, const RealT* coord2, RealT* normal )
{
  RealT dx = coord2[0] - coord1[0];
  RealT dy = coord2[1] - coord1[1];
  RealT len = std::sqrt( dy * dy + dx * dx );
  dx /= len;
  dy /= len;
  normal[0] = dy;
  normal[1] = -dx;
}

// Integrate the nodal smoothed gap and tributary area contributions over edge A.
// The generated weights already contain the physical integration measure.
TRIBOL_ENZYME_INLINE void gtilde_kernel( const RealT* x, PairQuadratureView<1, 1, true> quadrature, RealT residual_gap,
                                         RealT* g_tilde_out, RealT* A_out )
{
  // x stores the two endpoints of edge A followed by the two endpoints of edge B.
  const RealT A0[2] = { x[0], x[1] };
  const RealT A1[2] = { x[2], x[3] };
  const RealT B0[2] = { x[4], x[5] };
  const RealT B1[2] = { x[6], x[7] };

  RealT nB[2];
  find_normal( B0, B1, nB );

  RealT nA[2];
  find_normal( A0, A1, nA );

  // Only keep the contribution when the edge normals oppose each other.
  // Refined search rejects co-oriented normals; direct calculator calls also need this clamp.
  RealT dot = nB[0] * nA[0] + nB[1] * nA[1];
  RealT eta = ( dot < 0 ) ? dot : 0.0;

  RealT g1 = 0.0, g2 = 0.0;
  RealT AI_1 = 0.0, AI_2 = 0.0;

  for ( int i = 0; i < quadrature.weights.size(); ++i ) {
    const RealT xiA = quadrature.geometry1_coords( i, 0 );
    const RealT w = quadrature.weights[i];

    RealT basis_values[LinearEdgeBasis::node_count];
    LinearEdgeBasis::eval(
        MdSpan<const RealT, LinearEdgeBasis::parametric_dimension>{ &xiA, LinearEdgeBasis::parametric_dimension },
        basis_values );

    const auto x1 = LinearEdgeGeometry::position( { x, LinearEdgeCoords::size() }, xiA );
    const auto x2 = LinearEdgeGeometry::position( { x + LinearEdgeCoords::size(), LinearEdgeCoords::size() },
                                                  quadrature.geometry2_coords( i, 0 ) );
    const RealT gn = ( x2[0] - x1[0] ) * nB[0] + ( x2[1] - x1[1] ) * nB[1];
    const RealT g = effective_gap( gn, eta, residual_gap );

    g1 += w * basis_values[0] * g;
    g2 += w * basis_values[1] * g;

    AI_1 += w * basis_values[0];
    AI_2 += w * basis_values[1];
  }

  g_tilde_out[0] = g1;
  g_tilde_out[1] = g2;

  A_out[0] = AI_1;
  A_out[1] = AI_2;
}

// Select which scalar quantity is extracted from the gap/area kernel for Enzyme differentiation.
enum class KernelOutput
{
  GTILDE1,
  GTILDE2,
  A1,
  A2
};

// Wrap the fixed-quadrature kernel as a scalar-valued function for Enzyme.
template <typename QuadratureRule, KernelOutput Output>
static void kernel_out( const RealT* x, const void* integration_surface_rule_void, const void* kp_void, RealT* out )
{
  const auto* kp = static_cast<const KernelParams<QuadratureRule>*>( kp_void );
  const auto* integration_surface_rule =
      static_cast<const typename QuadratureRule::IntegrationSurfaceRule*>( integration_surface_rule_void );
  if ( integration_surface_rule->result.status != QuadratureStatus::Success )
    reportQuadratureFailure( integration_surface_rule->result.status );
  const auto quadrature = kp->quadrature_rule.mapIntegrationSurfaceRuleToElements(
      { x, LinearEdgeCoords::size() }, { x + LinearEdgeCoords::size(), LinearEdgeCoords::size() },
      integration_surface_rule->getView() );
  if ( quadrature.result.status != QuadratureStatus::Success ) reportQuadratureFailure( quadrature.result.status );
  RealT gt[2];
  RealT A_out[2];
  gtilde_kernel( x, quadrature.getView(), kp->residual_gap, gt, A_out );

  // Extract the requested scalar output for differentiation.
  if constexpr ( Output == KernelOutput::GTILDE1 )
    *out = gt[0];
  else if constexpr ( Output == KernelOutput::GTILDE2 )
    *out = gt[1];
  else if constexpr ( Output == KernelOutput::A1 )
    *out = A_out[0];
  else if constexpr ( Output == KernelOutput::A2 )
    *out = A_out[1];
}

// Differentiate the selected fixed-quadrature scalar kernel with respect to the 8 endpoint coordinates.
template <typename QuadratureRule, KernelOutput Output>
void grad_kernel( const RealT* x, const typename QuadratureRule::IntegrationSurfaceRule* integration_surface_rule,
                  const KernelParams<QuadratureRule>* kp, RealT* dout_du )
{
  RealT dx[8] = { 0.0 };
  RealT out = 0.0;
  RealT dout = 1.0;

  // Seed the scalar output with 1.0 so Enzyme accumulates dOutput/dx into dx.
  __enzyme_autodiff<void>( (void*)kernel_out<QuadratureRule, Output>, enzyme_dup, x, dx, enzyme_const,
                           (const void*)integration_surface_rule, enzyme_const, (const void*)kp, enzyme_dup, &out,
                           &dout );

  for ( int i = 0; i < 8; ++i ) dout_du[i] = dx[i];
}

//**************************************** */
// Enzyme functions for varying quadrature:

// Wrap the varying-quadrature kernel as a scalar-valued function for Enzyme.
template <typename QuadratureRule, KernelOutput Output>
static void kernel_out_enzyme( const RealT* x, const void* kp_void, RealT* out )
{
  const KernelParams<QuadratureRule>* kp = static_cast<const KernelParams<QuadratureRule>*>( kp_void );
  const auto quadrature = kp->quadrature_rule.generate( { x, LinearEdgeCoords::size() },
                                                        { x + LinearEdgeCoords::size(), LinearEdgeCoords::size() } );

  if ( quadrature.result.status != QuadratureStatus::Success ) reportQuadratureFailure( quadrature.result.status );
  RealT gt[2];
  RealT A_out[2];
  gtilde_kernel( x, quadrature.getView(), kp->residual_gap, gt, A_out );

  // Extract the requested scalar output for differentiation.
  if constexpr ( Output == KernelOutput::GTILDE1 )
    *out = gt[0];
  else if constexpr ( Output == KernelOutput::GTILDE2 )
    *out = gt[1];
  else if constexpr ( Output == KernelOutput::A1 )
    *out = A_out[0];
  else if constexpr ( Output == KernelOutput::A2 )
    *out = A_out[1];
}

// Differentiate the selected varying-quadrature scalar kernel with respect to the 8 endpoint coordinates.
template <typename QuadratureRule, KernelOutput Output>
void grad_kernel_enzyme( const RealT* x, const KernelParams<QuadratureRule>* kp, RealT* dout_du )
{
  RealT dx[8] = { 0.0 };
  RealT out = 0.0;
  RealT dout = 1.0;

  // Seed the scalar output with 1.0 so Enzyme accumulates dOutput/dx into dx.
  __enzyme_autodiff<void>( (void*)kernel_out_enzyme<QuadratureRule, Output>, enzyme_dup, x, dx, enzyme_const,
                           (const void*)kp, enzyme_dup, &out, &dout );

  for ( int i = 0; i < 8; ++i ) {
    dout_du[i] = dx[i];
  }
}

// Compute the Hessian of the selected varying-quadrature scalar kernel.
template <typename QuadratureRule, KernelOutput Output>
void d2_kernel( const RealT* x, const KernelParams<QuadratureRule>* kp, RealT* H )
{
  for ( int col = 0; col < 8; ++col ) {
    RealT dx[8] = { 0.0 };
    dx[col] = 1.0;

    RealT grad[8] = { 0.0 };
    RealT dgrad[8] = { 0.0 };

    // Differentiate the gradient in coordinate direction col to form one Hessian column.
    __enzyme_fwddiff<void>( (void*)grad_kernel_enzyme<QuadratureRule, Output>, enzyme_dup, x, dx, enzyme_const,
                            (const void*)kp, enzyme_dup, grad, dgrad );

    for ( int row = 0; row < 8; ++row ) H[row * 8 + col] = dgrad[row];
  }
}

// Compute the Hessian of the selected fixed-quadrature scalar kernel.
template <typename QuadratureRule, KernelOutput Output>
void d2_kernel_quad( const RealT* x, const typename QuadratureRule::IntegrationSurfaceRule* integration_surface_rule,
                     const KernelParams<QuadratureRule>* kp, RealT* H )
{
  for ( int col = 0; col < 8; ++col ) {
    RealT dx[8] = { 0.0 };
    dx[col] = 1.0;
    RealT grad[8] = { 0.0 };
    RealT dgrad[8] = { 0.0 };

    // Differentiate the gradient in coordinate direction col to form one Hessian column
    __enzyme_fwddiff<void>( (void*)grad_kernel<QuadratureRule, Output>, enzyme_dup, x, dx, enzyme_const,
                            (const void*)integration_surface_rule, enzyme_const, (const void*)kp, enzyme_dup, grad,
                            dgrad );
    for ( int row = 0; row < 8; ++row ) H[row * 8 + col] = dgrad[row];
  }
}

}  // namespace

// Integration-surface points and reference weights are fixed under frozen-quadrature differentiation.
template <typename QuadratureRule>
typename EnergyMortarCalculator<QuadratureRule>::IntegrationSurfaceRule
EnergyMortarCalculator<QuadratureRule>::generateIntegrationSurfaceRule( CoordinateView coordinates ) const
{
  const auto integration_surface_rule = quadrature_rule_.generateIntegrationSurfaceRule(
      { coordinates.data(), element_dof_count }, { coordinates.data() + element_dof_count, element_dof_count } );
  if ( integration_surface_rule.result.status != QuadratureStatus::Success )
    reportQuadratureFailure( integration_surface_rule.result.status );
  return integration_surface_rule;
}

// Evaluate the weighted normal gap at local coordinate xiA on the first edge.
template <typename QuadratureRule>
RealT EnergyMortarCalculator<QuadratureRule>::compute_weighted_normal_gap( CoordinateView coordinates, RealT xiA ) const
{
  constexpr int dimension = LinearEdgeGeometry::spatial_dimension;
  const auto* A0 = coordinates.data();
  const auto* A1 = coordinates.data() + dimension;
  const auto* B0 = coordinates.data() + element_dof_count;
  const auto* B1 = B0 + dimension;
  RealT nA[dimension] = { 0.0 };
  RealT nB[dimension] = { 0.0 };
  find_normal( A0, A1, nA );
  find_normal( B0, B1, nB );
  const LinearEdgeGeometry::Coords edge_coords{ coordinates.data(), element_dof_count };
  RealT basis_values[LinearEdgeBasis::node_count];
  LinearEdgeBasis::eval(
      MdSpan<const RealT, LinearEdgeBasis::parametric_dimension>{ &xiA, LinearEdgeBasis::parametric_dimension },
      basis_values );
  RealT x1[dimension] = { 0.0 };
  mapToPhysical<LinearEdgeBasis>( basis_values, edge_coords, x1 );
  const RealT gn = ( B0[0] - x1[0] ) * nB[0] + ( B0[1] - x1[1] ) * nB[1];
  const RealT dot = nB[0] * nA[0] + nB[1] * nA[1];
  const RealT eta = dot < 0.0 ? dot : 0.0;
  return effective_gap( gn, eta, p_.residual_gap );
}

template <typename QuadratureRule>
NodalContactData EnergyMortarCalculator<QuadratureRule>::compute_nodal_contact_data( CoordinateView coordinates ) const
{
  const auto quadrature = quadrature_rule_.generate( { coordinates.data(), element_dof_count },
                                                     { coordinates.data() + element_dof_count, element_dof_count } );
  if ( quadrature.result.status != QuadratureStatus::Success ) reportQuadratureFailure( quadrature.result.status );
  NodalContactData result{};
  gtilde_kernel( coordinates.data(), quadrature.getView(), p_.residual_gap, result.g_tilde.data(), result.AI.data() );
  return result;
}

template <typename QuadratureRule>
void EnergyMortarCalculator<QuadratureRule>::compute_gtilde_and_area( CoordinateView coordinates, RealT gtilde[2],
                                                                      RealT area[2] ) const
{
  const auto ncd = compute_nodal_contact_data( coordinates );
  gtilde[0] = ncd.g_tilde[0];
  gtilde[1] = ncd.g_tilde[1];
  area[0] = ncd.AI[0];
  area[1] = ncd.AI[1];
}

template <typename QuadratureRule>
void EnergyMortarCalculator<QuadratureRule>::grad_gtilde( CoordinateView coordinates, RealT dgt1_dx[8],
                                                          RealT dgt2_dx[8] ) const
{
  RealT dg1_du[dof_count] = { 0.0 };
  RealT dg2_du[dof_count] = { 0.0 };
  const KernelParams<QuadratureRule> kp{ quadrature_rule_, p_.k, p_.residual_gap };
  if ( !p_.enzyme_quadrature ) {
    // Freeze integration-surface coordinates and weights, while differentiating the current geometry.
    const auto integration_surface_rule = generateIntegrationSurfaceRule( coordinates );
    grad_kernel<QuadratureRule, KernelOutput::GTILDE1>( coordinates.data(), &integration_surface_rule, &kp, dg1_du );
    grad_kernel<QuadratureRule, KernelOutput::GTILDE2>( coordinates.data(), &integration_surface_rule, &kp, dg2_du );
  } else {
    grad_kernel_enzyme<QuadratureRule, KernelOutput::GTILDE1>( coordinates.data(), &kp, dg1_du );
    grad_kernel_enzyme<QuadratureRule, KernelOutput::GTILDE2>( coordinates.data(), &kp, dg2_du );
  }
  for ( IndexT i = 0; i < dof_count; ++i ) {
    dgt1_dx[i] = dg1_du[i];
    dgt2_dx[i] = dg2_du[i];
  }
}

template <typename QuadratureRule>
void EnergyMortarCalculator<QuadratureRule>::grad_trib_area( CoordinateView coordinates, RealT dA1_dx[8],
                                                             RealT dA2_dx[8] ) const
{
  const KernelParams<QuadratureRule> kp{ quadrature_rule_, p_.k, p_.residual_gap };
  if ( !p_.enzyme_quadrature ) {
    const auto integration_surface_rule = generateIntegrationSurfaceRule( coordinates );
    grad_kernel<QuadratureRule, KernelOutput::A1>( coordinates.data(), &integration_surface_rule, &kp, dA1_dx );
    grad_kernel<QuadratureRule, KernelOutput::A2>( coordinates.data(), &integration_surface_rule, &kp, dA2_dx );
  } else {
    grad_kernel_enzyme<QuadratureRule, KernelOutput::A1>( coordinates.data(), &kp, dA1_dx );
    grad_kernel_enzyme<QuadratureRule, KernelOutput::A2>( coordinates.data(), &kp, dA2_dx );
  }
}

template <typename QuadratureRule>
void EnergyMortarCalculator<QuadratureRule>::d2_g2tilde( CoordinateView coordinates, RealT H1[64], RealT H2[64] ) const
{
  RealT d2g1_d2u[dof_count * dof_count] = { 0.0 };
  RealT d2g2_d2u[dof_count * dof_count] = { 0.0 };
  const KernelParams<QuadratureRule> kp{ quadrature_rule_, p_.k, p_.residual_gap };
  if ( !p_.enzyme_quadrature ) {
    const auto integration_surface_rule = generateIntegrationSurfaceRule( coordinates );
    d2_kernel_quad<QuadratureRule, KernelOutput::GTILDE1>( coordinates.data(), &integration_surface_rule, &kp,
                                                           d2g1_d2u );
    d2_kernel_quad<QuadratureRule, KernelOutput::GTILDE2>( coordinates.data(), &integration_surface_rule, &kp,
                                                           d2g2_d2u );
  } else {
    d2_kernel<QuadratureRule, KernelOutput::GTILDE1>( coordinates.data(), &kp, d2g1_d2u );
    d2_kernel<QuadratureRule, KernelOutput::GTILDE2>( coordinates.data(), &kp, d2g2_d2u );
  }
  for ( IndexT i = 0; i < dof_count * dof_count; ++i ) {
    H1[i] = d2g1_d2u[i];
    H2[i] = d2g2_d2u[i];
  }
}

template <typename QuadratureRule>
void EnergyMortarCalculator<QuadratureRule>::compute_d2A_d2u( CoordinateView coordinates, RealT d2A1[64],
                                                              RealT d2A2[64] ) const
{
  RealT d2A1_d2u[dof_count * dof_count] = { 0.0 };
  RealT d2A2_d2u[dof_count * dof_count] = { 0.0 };
  const KernelParams<QuadratureRule> kp{ quadrature_rule_, p_.k, p_.residual_gap };
  if ( !p_.enzyme_quadrature ) {
    const auto integration_surface_rule = generateIntegrationSurfaceRule( coordinates );
    d2_kernel_quad<QuadratureRule, KernelOutput::A1>( coordinates.data(), &integration_surface_rule, &kp, d2A1_d2u );
    d2_kernel_quad<QuadratureRule, KernelOutput::A2>( coordinates.data(), &integration_surface_rule, &kp, d2A2_d2u );
  } else {
    d2_kernel<QuadratureRule, KernelOutput::A1>( coordinates.data(), &kp, d2A1_d2u );
    d2_kernel<QuadratureRule, KernelOutput::A2>( coordinates.data(), &kp, d2A2_d2u );
  }
  for ( IndexT i = 0; i < dof_count * dof_count; ++i ) {
    d2A1[i] = d2A1_d2u[i];
    d2A2[i] = d2A2_d2u[i];
  }
}

template <typename QuadratureRule>
RealT EnergyMortarCalculator<QuadratureRule>::compute_quadrature_point_penalty_energy(
    CoordinateView coordinates ) const
{
  const EnergyMortarEnergy<QuadratureRule> definition{ quadrature_rule_, p_.k, p_.residual_gap };
  return definition.energy( coordinates );
}

template <typename QuadratureRule>
QuadraturePointPenaltyData EnergyMortarCalculator<QuadratureRule>::compute_quadrature_point_penalty_data(
    CoordinateView coordinates ) const
{
  const EnergyMortarEnergy<QuadratureRule> definition{ quadrature_rule_, p_.k, p_.residual_gap };
  const auto primal = definition.evaluate( coordinates );
  if ( primal.quadrature_result.status != QuadratureStatus::Success )
    reportQuadratureFailure( primal.quadrature_result.status );
  QuadraturePointPenaltyData result{};
  result.energy = primal.energy;
  result.has_active_qp = primal.has_active_qp;
  if ( !primal.has_active_qp ) return result;
  const auto derivatives =
      EnergyPairEvaluator<EnergyMortarEnergy<QuadratureRule>>::evaluate( coordinates, definition, primal.energy );
  for ( IndexT i = 0; i < dof_count; ++i ) result.force[i] = derivatives.residual[i];
  for ( IndexT i = 0; i < dof_count * dof_count; ++i ) result.stiffness[i] = derivatives.jacobian[i];
  return result;
}

template class EnergyMortarCalculator<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>>;
template class EnergyMortarCalculator<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>>;
template class EnergyMortarCalculator<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>>;

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
