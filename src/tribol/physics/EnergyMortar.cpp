#include "tribol/physics/EnergyMortar.hpp"

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
using LinearEdgePointView = MdSpan<const double, LinearEdgeBasis::spatial_dimension>;
using LinearEdgeCoordinates = MdArray<double, LinearEdgeBasis::node_count, LinearEdgeBasis::spatial_dimension>;

// Keep only inactive configuration in the Enzyme parameter block; coordinates arrive through the active x argument.
template <typename QuadratureRule>
struct KernelParams {
  QuadratureRule quadrature_rule;
  double k;
  double residual_gap;
};

TRIBOL_ENZYME_INLINE double effective_gap( double gap_normal, double normal_cosine, double residual_gap )
{
  return gap_normal * normal_cosine - residual_gap;
}

// Compute a unit normal vector for the line segment from coord1 to coord2
TRIBOL_ENZYME_INLINE void find_normal( const double* coord1, const double* coord2, double* normal )
{
  double dx = coord2[0] - coord1[0];
  double dy = coord2[1] - coord1[1];
  double len = std::sqrt( dy * dy + dx * dx );
  dx /= len;
  dy /= len;
  normal[0] = dy;
  normal[1] = -dx;
}

// returns P0 and P1 which are the edge vertex coordinates associated with the edge with elem_id.
// coordinates it returns
TRIBOL_ENZYME_INLINE void endpoints( const MeshData::Viewer& mesh, int elem_id, double P0[2], double P1[2] )
{
  double P0_P1[4];
  mesh.getFaceCoords( elem_id, P0_P1 );
  P0[0] = P0_P1[0];
  P0[1] = P0_P1[1];
  P1[0] = P0_P1[2];
  P1[1] = P0_P1[3];
}

// Integrate the nodal smoothed gap and tributary area contributions over edge A.
// The generated weights already contain the physical integration measure.
TRIBOL_ENZYME_INLINE void gtilde_kernel( const double* x, PairQuadratureView<1, 1, true> quadrature,
                                         double residual_gap, double* g_tilde_out, double* A_out )
{
  // x stores the two endpoints of edge A followed by the two endpoints of edge B.
  const double A0[2] = { x[0], x[1] };
  const double A1[2] = { x[2], x[3] };
  const double B0[2] = { x[4], x[5] };
  const double B1[2] = { x[6], x[7] };

  double nB[2];
  find_normal( B0, B1, nB );

  double nA[2];
  find_normal( A0, A1, nA );

  // Only keep the contribution when the edge normals oppose each other.
  // Refined search rejects co-oriented normals; direct calculator calls also need this clamp.
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  double eta = ( dot < 0 ) ? dot : 0.0;

  double g1 = 0.0, g2 = 0.0;
  double AI_1 = 0.0, AI_2 = 0.0;

  for ( int i = 0; i < quadrature.physical_weights.size(); ++i ) {
    const double xiA = quadrature.mesh1_parent_coordinates( i, 0 );
    const double w = quadrature.physical_weights[i];

    double basis_values[LinearEdgeBasis::node_count];
    LinearEdgeBasis::Eval(
        MdSpan<const double, LinearEdgeBasis::parent_dimension>{ &xiA, LinearEdgeBasis::parent_dimension },
        basis_values );

    const auto x1 = LinearEdgeGeometry::position( { x, LinearEdgeCoordinates::size() }, xiA );
    const auto x2 = LinearEdgeGeometry::position( { x + LinearEdgeCoordinates::size(), LinearEdgeCoordinates::size() },
                                                  quadrature.mesh2_parent_coordinates( i, 0 ) );
    const double gn = ( x2[0] - x1[0] ) * nB[0] + ( x2[1] - x1[1] ) * nB[1];
    const double g = effective_gap( gn, eta, residual_gap );

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
static void kernel_out( const double* x, const void* reference_void, const void* kp_void, double* out )
{
  const auto* kp = static_cast<const KernelParams<QuadratureRule>*>( kp_void );
  const auto* reference = static_cast<const typename QuadratureRule::ReferenceQuadrature*>( reference_void );
  const auto quadrature = kp->quadrature_rule.mapReferenceQuadrature(
      { x, LinearEdgeCoordinates::size() }, { x + LinearEdgeCoordinates::size(), LinearEdgeCoordinates::size() },
      reference->getView() );
  double gt[2];
  double A_out[2];
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
void grad_kernel( const double* x, const typename QuadratureRule::ReferenceQuadrature* reference,
                  const KernelParams<QuadratureRule>* kp, double* dout_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  // Seed the scalar output with 1.0 so Enzyme accumulates dOutput/dx into dx.
  __enzyme_autodiff<void>( (void*)kernel_out<QuadratureRule, Output>, enzyme_dup, x, dx, enzyme_const,
                           (const void*)reference, enzyme_const, (const void*)kp, enzyme_dup, &out, &dout );

  for ( int i = 0; i < 8; ++i ) dout_du[i] = dx[i];
}

//**************************************** */
// Enzyme functions for varying quadrature:

// Wrap the varying-quadrature kernel as a scalar-valued function for Enzyme.
template <typename QuadratureRule, KernelOutput Output>
static void kernel_out_enzyme( const double* x, const void* kp_void, double* out )
{
  const KernelParams<QuadratureRule>* kp = static_cast<const KernelParams<QuadratureRule>*>( kp_void );
  const auto quadrature = kp->quadrature_rule.generate(
      { x, LinearEdgeCoordinates::size() }, { x + LinearEdgeCoordinates::size(), LinearEdgeCoordinates::size() } );

  double gt[2];
  double A_out[2];
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
void grad_kernel_enzyme( const double* x, const KernelParams<QuadratureRule>* kp, double* dout_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  // Seed the scalar output with 1.0 so Enzyme accumulates dOutput/dx into dx.
  __enzyme_autodiff<void>( (void*)kernel_out_enzyme<QuadratureRule, Output>, enzyme_dup, x, dx, enzyme_const,
                           (const void*)kp, enzyme_dup, &out, &dout );

  for ( int i = 0; i < 8; ++i ) {
    dout_du[i] = dx[i];
  }
}

// Compute the Hessian of the selected varying-quadrature scalar kernel.
template <typename QuadratureRule, KernelOutput Output>
void d2_kernel( const double* x, const KernelParams<QuadratureRule>* kp, double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;

    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    // Differentiate the gradient in coordinate direction col to form one Hessian column.
    __enzyme_fwddiff<void>( (void*)grad_kernel_enzyme<QuadratureRule, Output>, enzyme_dup, x, dx, enzyme_const,
                            (const void*)kp, enzyme_dup, grad, dgrad );

    for ( int row = 0; row < 8; ++row ) H[row * 8 + col] = dgrad[row];
  }
}

TRIBOL_ENZYME_INLINE double qp_penalty_kernel_qp_energy( double w, const double* x1, const double* x2, const double* nB,
                                                         double eta, double residual_gap, double penalty,
                                                         bool* pair_has_active_qp )
{
  const double gn = ( x2[0] - x1[0] ) * nB[0] + ( x2[1] - x1[1] ) * nB[1];
  const double gap = effective_gap( gn, eta, residual_gap );
  const bool is_active = gap <= 0.0;

  *pair_has_active_qp = *pair_has_active_qp || is_active;

  return is_active ? 0.5 * penalty * gap * gap * w : 0.0;
}

template <typename QuadratureRule>
TRIBOL_ENZYME_INLINE void qp_penalty_kernel( const double* x, const KernelParams<QuadratureRule>* kp, double* energy,
                                             bool* pair_has_active_qp )
{
  *pair_has_active_qp = false;

  double A0[2] = { x[0], x[1] };
  double A1[2] = { x[2], x[3] };
  double B0[2] = { x[4], x[5] };
  double B1[2] = { x[6], x[7] };

  const auto generated = kp->quadrature_rule.generate(
      { x, LinearEdgeCoordinates::size() }, { x + LinearEdgeCoordinates::size(), LinearEdgeCoordinates::size() } );
  const auto quadrature = generated.getView();

  double nB[2];
  find_normal( B0, B1, nB );
  double nA[2];
  find_normal( A0, A1, nA );
  // Only keep the contribution when the edge normals oppose each other.
  // Refined search rejects co-oriented normals; direct calculator calls also need this clamp.
  const double dot = nA[0] * nB[0] + nA[1] * nB[1];
  const double eta = ( dot < 0 ) ? dot : 0.0;
  double value = 0.0;
  for ( int i = 0; i < quadrature.physical_weights.size(); ++i ) {
    const auto x1 = LinearEdgeGeometry::position( { x, LinearEdgeCoordinates::size() },
                                                  quadrature.mesh1_parent_coordinates( i, 0 ) );
    const auto x2 = LinearEdgeGeometry::position( { x + LinearEdgeCoordinates::size(), LinearEdgeCoordinates::size() },
                                                  quadrature.mesh2_parent_coordinates( i, 0 ) );
    value += qp_penalty_kernel_qp_energy( quadrature.physical_weights[i], x1.data(), x2.data(), nB, eta,
                                          kp->residual_gap, kp->k, pair_has_active_qp );
  }

  *energy = value;
}

template <typename QuadratureRule>
void grad_qp_penalty_kernel( const double* x, const KernelParams<QuadratureRule>* kp, double* dout_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;
  bool pair_has_active_qp = false;
  __enzyme_autodiff<void>( (void*)qp_penalty_kernel<QuadratureRule>, enzyme_dup, x, dx, enzyme_const, (const void*)kp,
                           enzyme_dup, &out, &dout, enzyme_const, &pair_has_active_qp );

  for ( int i = 0; i < 8; ++i ) {
    dout_du[i] = dx[i];
  }
}

template <typename QuadratureRule>
void d2_qp_penalty_kernel( const double* x, const KernelParams<QuadratureRule>* kp, double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;
    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };
    __enzyme_fwddiff<void>( (void*)grad_qp_penalty_kernel<QuadratureRule>, enzyme_dup, x, dx, enzyme_const,
                            (const void*)kp, enzyme_dup, grad, dgrad );
    for ( int row = 0; row < 8; ++row ) {
      H[row * 8 + col] = dgrad[row];
    }
  }
}

// Compute the Hessian of the selected fixed-quadrature scalar kernel.
template <typename QuadratureRule, KernelOutput Output>
void d2_kernel_quad( const double* x, const typename QuadratureRule::ReferenceQuadrature* reference,
                     const KernelParams<QuadratureRule>* kp, double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;
    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    // Differentiate the gradient in coordinate direction col to form one Hessian column
    __enzyme_fwddiff<void>( (void*)grad_kernel<QuadratureRule, Output>, enzyme_dup, x, dx, enzyme_const,
                            (const void*)reference, enzyme_const, (const void*)kp, enzyme_dup, grad, dgrad );
    for ( int row = 0; row < 8; ++row ) H[row * 8 + col] = dgrad[row];
  }
}

}  // namespace

// Reference points and weights are kept separate for the frozen-quadrature derivative path.
template <typename QuadratureRule>
typename EnergyMortarCalculator<QuadratureRule>::ReferenceQuadrature
EnergyMortarCalculator<QuadratureRule>::constructReferenceQuadrature( const InterfacePair& pair,
                                                                      const MeshData::Viewer& mesh1,
                                                                      const MeshData::Viewer& mesh2 ) const
{
  LinearEdgeCoordinates coords1, coords2;
  mesh1.getFaceCoords( pair.element_id1_, coords1.data() );
  mesh2.getFaceCoords( pair.element_id2_, coords2.data() );
  const auto reference = quadrature_rule_.generateReferenceQuadrature( coords1.view(), coords2.view() );
  if ( reference.result.status == QuadratureStatus::InvalidGeometry ) {
    throw std::invalid_argument( "Cannot generate a quadrature rule for invalid pair geometry." );
  }
  return reference;
}

// Evaluate the weighted normal gap at local coordinate xiA on edge A.
template <typename QuadratureRule>
double EnergyMortarCalculator<QuadratureRule>::compute_weighted_normal_gap( const InterfacePair& pair,
                                                                            const MeshData::Viewer& mesh1,
                                                                            const MeshData::Viewer& mesh2,
                                                                            double xiA ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.element_id1_, A0, A1 );
  endpoints( mesh2, pair.element_id2_, B0, B1 );

  double nA[2] = { 0.0 };
  double nB[2] = { 0.0 };
  find_normal( A0, A1, nA );
  find_normal( B0, B1, nB );

  const LinearEdgeCoordinates edge_coordinates{ LinearEdgePointView{ A0 }, LinearEdgePointView{ A1 } };
  double basis_values[LinearEdgeBasis::node_count];
  LinearEdgeBasis::Eval(
      MdSpan<const double, LinearEdgeBasis::parent_dimension>{ &xiA, LinearEdgeBasis::parent_dimension },
      basis_values );
  double x1[2] = { 0.0 };
  MapToPhysical<LinearEdgeBasis>( basis_values, edge_coordinates.view(), x1 );

  const double gn = ( B0[0] - x1[0] ) * nB[0] + ( B0[1] - x1[1] ) * nB[1];
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  double eta = ( dot < 0 ) ? dot : 0.0;

  return effective_gap( gn, eta, p_.residual_gap );
}

// Assemble nodal gap and tributary area data for the current interface pair.
template <typename QuadratureRule>
NodalContactData EnergyMortarCalculator<QuadratureRule>::compute_nodal_contact_data(
    const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) const
{
  double x[8];
  mesh1.getFaceCoords( pair.element_id1_, x );
  mesh2.getFaceCoords( pair.element_id2_, x + 4 );
  const auto quadrature = quadrature_rule_.generate(
      { x, LinearEdgeCoordinates::size() }, { x + LinearEdgeCoordinates::size(), LinearEdgeCoordinates::size() } );
  NodalContactData result{};
  gtilde_kernel( x, quadrature.getView(), p_.residual_gap, result.g_tilde.data(), result.AI.data() );
  return result;
}

// Return the nodal smoothed gaps and tributary areas for the interface pair.
template <typename QuadratureRule>
void EnergyMortarCalculator<QuadratureRule>::compute_gtilde_and_area( const InterfacePair& pair,
                                                                      const MeshData::Viewer& mesh1,
                                                                      const MeshData::Viewer& mesh2, double gtilde[2],
                                                                      double area[2] ) const
{
  auto ncd = compute_nodal_contact_data( pair, mesh1, mesh2 );
  gtilde[0] = ncd.g_tilde[0];
  gtilde[1] = ncd.g_tilde[1];
  area[0] = ncd.AI[0];
  area[1] = ncd.AI[1];
}

// Compute derivatives of the two nodal smoothed gaps with respect to the endpoint coordinates.
template <typename QuadratureRule>
void EnergyMortarCalculator<QuadratureRule>::grad_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                          const MeshData::Viewer& mesh2, double dgt1_dx[8],
                                                          double dgt2_dx[8] ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.element_id1_, A0, A1 );
  endpoints( mesh2, pair.element_id2_, B0, B1 );

  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  double nB[2], nA[2];
  find_normal( B0, B1, nB );
  find_normal( A0, A1, nA );

  double dg1_du[8] = { 0.0 };
  double dg2_du[8] = { 0.0 };

  const KernelParams<QuadratureRule> kp{ quadrature_rule_, p_.k, p_.residual_gap };
  if ( !p_.enzyme_quadrature ) {
    // Hold the quadrature rule fixed while differentiating the gap kernel.
    const auto reference = constructReferenceQuadrature( pair, mesh1, mesh2 );
    grad_kernel<QuadratureRule, KernelOutput::GTILDE1>( x, &reference, &kp, dg1_du );
    grad_kernel<QuadratureRule, KernelOutput::GTILDE2>( x, &reference, &kp, dg2_du );

  } else {
    // Differentiate through the geometry-dependent quadrature construction.
    grad_kernel_enzyme<QuadratureRule, KernelOutput::GTILDE1>( x, &kp, dg1_du );
    grad_kernel_enzyme<QuadratureRule, KernelOutput::GTILDE2>( x, &kp, dg2_du );
  }

  for ( int i = 0; i < 8; ++i ) {
    dgt1_dx[i] = dg1_du[i];
    dgt2_dx[i] = dg2_du[i];
  }
}

// Compute derivatives of the two nodal tributary areas with respect to the endpoint coordinates
template <typename QuadratureRule>
void EnergyMortarCalculator<QuadratureRule>::grad_trib_area( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                             const MeshData::Viewer& mesh2, double dA1_dx[8],
                                                             double dA2_dx[8] ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.element_id1_, A0, A1 );
  endpoints( mesh2, pair.element_id2_, B0, B1 );

  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  double nB[2], nA[2];
  find_normal( B0, B1, nB );
  find_normal( A0, A1, nA );

  const KernelParams<QuadratureRule> kp{ quadrature_rule_, p_.k, p_.residual_gap };
  if ( !p_.enzyme_quadrature ) {
    // Hold the quadrature rule fixed while differentiating the area kernel.
    const auto reference = constructReferenceQuadrature( pair, mesh1, mesh2 );
    grad_kernel<QuadratureRule, KernelOutput::A1>( x, &reference, &kp, dA1_dx );
    grad_kernel<QuadratureRule, KernelOutput::A2>( x, &reference, &kp, dA2_dx );
  } else {
    // Differentiate through the geometry-dependent quadrature construction.
    grad_kernel_enzyme<QuadratureRule, KernelOutput::A1>( x, &kp, dA1_dx );
    grad_kernel_enzyme<QuadratureRule, KernelOutput::A2>( x, &kp, dA2_dx );
  }
}

// Compute the Hessians of the two nodal smoothed gaps with respect to the endpoint coordinates.
template <typename QuadratureRule>
void EnergyMortarCalculator<QuadratureRule>::d2_g2tilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                         const MeshData::Viewer& mesh2, double H1[64],
                                                         double H2[64] ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.element_id1_, A0, A1 );
  endpoints( mesh2, pair.element_id2_, B0, B1 );

  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  double nB[2], nA[2];
  find_normal( B0, B1, nB );
  find_normal( A0, A1, nA );

  double d2g1_d2u[64] = { 0.0 };
  double d2g2_d2u[64] = { 0.0 };

  const KernelParams<QuadratureRule> kp{ quadrature_rule_, p_.k, p_.residual_gap };
  if ( !p_.enzyme_quadrature ) {
    // Hold the quadrature rule fixed while differentiating the gap gradients.
    const auto reference = constructReferenceQuadrature( pair, mesh1, mesh2 );
    d2_kernel_quad<QuadratureRule, KernelOutput::GTILDE1>( x, &reference, &kp, d2g1_d2u );
    d2_kernel_quad<QuadratureRule, KernelOutput::GTILDE2>( x, &reference, &kp, d2g2_d2u );

  } else {
    // Differentiate through the geometry-dependent quadrature construction.
    d2_kernel<QuadratureRule, KernelOutput::GTILDE1>( x, &kp, d2g1_d2u );
    d2_kernel<QuadratureRule, KernelOutput::GTILDE2>( x, &kp, d2g2_d2u );
  }

  for ( int i = 0; i < 64; ++i ) {
    H1[i] = d2g1_d2u[i];
    H2[i] = d2g2_d2u[i];
  }
}

// Compute the Hessians of the two nodal tributary areas with respect to the endpoint coordinates.
template <typename QuadratureRule>
void EnergyMortarCalculator<QuadratureRule>::compute_d2A_d2u( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                              const MeshData::Viewer& mesh2, double d2A1[64],
                                                              double d2A2[64] ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.element_id1_, A0, A1 );
  endpoints( mesh2, pair.element_id2_, B0, B1 );

  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  double nB[2], nA[2];

  find_normal( B0, B1, nB );
  find_normal( A0, A1, nA );

  double d2A1_d2u[64] = { 0.0 };
  double d2A2_d2u[64] = { 0.0 };

  const KernelParams<QuadratureRule> kp{ quadrature_rule_, p_.k, p_.residual_gap };
  if ( !p_.enzyme_quadrature ) {
    // Hold the quadrature rule fixed while differentiating the area gradients.
    const auto reference = constructReferenceQuadrature( pair, mesh1, mesh2 );
    d2_kernel_quad<QuadratureRule, KernelOutput::A1>( x, &reference, &kp, d2A1_d2u );
    d2_kernel_quad<QuadratureRule, KernelOutput::A2>( x, &reference, &kp, d2A2_d2u );
  } else {
    // Differentiate through the geometry-dependent quadrature construction.
    d2_kernel<QuadratureRule, KernelOutput::A1>( x, &kp, d2A1_d2u );
    d2_kernel<QuadratureRule, KernelOutput::A2>( x, &kp, d2A2_d2u );
  }

  for ( int i = 0; i < 64; ++i ) {
    d2A1[i] = d2A1_d2u[i];
    d2A2[i] = d2A2_d2u[i];
  }
}

template <typename QuadratureRule>
double EnergyMortarCalculator<QuadratureRule>::compute_quadrature_point_penalty_energy(
    const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.element_id1_, A0, A1 );
  endpoints( mesh2, pair.element_id2_, B0, B1 );

  const double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };
  const KernelParams<QuadratureRule> kp{ quadrature_rule_, p_.k, p_.residual_gap };
  double energy = 0.0;
  bool pair_has_active_qp = false;
  qp_penalty_kernel<QuadratureRule>( x, &kp, &energy, &pair_has_active_qp );
  return energy;
}

template <typename QuadratureRule>
QuadraturePointPenaltyData EnergyMortarCalculator<QuadratureRule>::compute_quadrature_point_penalty_data(
    const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.element_id1_, A0, A1 );
  endpoints( mesh2, pair.element_id2_, B0, B1 );

  const double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };
  const KernelParams<QuadratureRule> kp{ quadrature_rule_, p_.k, p_.residual_gap };

  QuadraturePointPenaltyData result;
  qp_penalty_kernel<QuadratureRule>( x, &kp, &result.energy, &result.has_active_qp );
  grad_qp_penalty_kernel<QuadratureRule>( x, &kp, result.force.data() );
  d2_qp_penalty_kernel<QuadratureRule>( x, &kp, result.stiffness.data() );
  return result;
}

template class EnergyMortarCalculator<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<1>>>;
template class EnergyMortarCalculator<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<2>>>;
template class EnergyMortarCalculator<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<3>>>;

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
