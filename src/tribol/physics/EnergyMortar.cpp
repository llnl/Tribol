#include "EnergyMortar.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/mesh/MeshData.hpp"
#include <set>
#include <map>

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

namespace {

using namespace energy_mortar_kernel_detail;

// Subset of EnergyMortarOptions needed by the Enzyme-differentiated kernel wrappers.
// Passed as enzyme_const so Enzyme doesn't try to differentiate through the params.
struct KernelParams {
  int N = 3;
  double del = 0.1;
  SmoothingType smoothing_type = SmoothingType::Hermite;
};

KernelParams make_kernel_params( const EnergyMortarOptions& opts )
{
  return { opts.N, opts.del, opts.smoothing_type };
}

inline void endpoints( const MeshData::Viewer& mesh, int elem_id, double P0[2], double P1[2] )
{
  double P0_P1[4];
  mesh.getFaceCoords( elem_id, P0_P1 );
  P0[0] = P0_P1[0];
  P0[1] = P0_P1[1];
  P1[0] = P0_P1[2];
  P1[1] = P0_P1[3];
}

// Select which scalar quantity is extracted from the gap/area kernel for Enzyme differentiation.
enum class KernelOutput
{
  GTILDE1,
  GTILDE2,
  A1,
  A2
};

// Fixed-quadrature scalar wrapper for Enzyme
template <KernelOutput Output>
static void kernel_out( const double* x, const void* gp_void, double* out )
{
  const Gparams* gp = static_cast<const Gparams*>( gp_void );
  double gt[2];
  double A_out[2];
  gap_area_kernel( x, *gp, gt, A_out );

  if constexpr ( Output == KernelOutput::GTILDE1 )
    *out = gt[0];
  else if constexpr ( Output == KernelOutput::GTILDE2 )
    *out = gt[1];
  else if constexpr ( Output == KernelOutput::A1 )
    *out = A_out[0];
  else if constexpr ( Output == KernelOutput::A2 )
    *out = A_out[1];
}

// Gradient of fixed-quadrature scalar kernel
template <KernelOutput Output>
void grad_kernel( const double* x, const Gparams* gp, double* dout_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  __enzyme_autodiff<void>( (void*)kernel_out<Output>, enzyme_dup, x, dx, enzyme_const, (const void*)gp, enzyme_dup,
                           &out, &dout );

  for ( int i = 0; i < 8; ++i ) dout_du[i] = dx[i];
}

// Varying-quadrature scalar wrapper for Enzyme
template <KernelOutput Output>
static void kernel_out_enzyme( const double* x, const void* kp_void, double* out )
{
  const auto* kp = static_cast<const KernelParams*>( kp_void );

  double gt[2];
  double A_out[2];
  energy_mortar_varying_quadrature_kernel( x, kp->del, kp->N, kp->smoothing_type, gt, A_out );

  if constexpr ( Output == KernelOutput::GTILDE1 )
    *out = gt[0];
  else if constexpr ( Output == KernelOutput::GTILDE2 )
    *out = gt[1];
  else if constexpr ( Output == KernelOutput::A1 )
    *out = A_out[0];
  else if constexpr ( Output == KernelOutput::A2 )
    *out = A_out[1];
}

// Gradient of varying-quadrature scalar kernel
template <KernelOutput Output>
void grad_kernel_enzyme( const double* x, const KernelParams* kp, double* dout_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  __enzyme_autodiff<void>( (void*)kernel_out_enzyme<Output>, enzyme_dup, x, dx, enzyme_const, (const void*)kp,
                           enzyme_dup, &out, &dout );

  for ( int i = 0; i < 8; ++i ) {
    dout_du[i] = dx[i];
  }
}

// Hessian of varying-quadrature scalar kernel
template <KernelOutput Output>
void d2_kernel( const double* x, const KernelParams* kp, double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;

    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    __enzyme_fwddiff<void>( (void*)grad_kernel_enzyme<Output>, enzyme_dup, x, dx, enzyme_const, (const void*)kp,
                            enzyme_dup, grad, dgrad );

    for ( int row = 0; row < 8; ++row ) H[row * 8 + col] = dgrad[row];
  }
}

// Hessian of fixed-quadrature scalar kernel
template <KernelOutput Output>
void d2_kernel_quad( const double* x, const Gparams* gp, double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;
    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    __enzyme_fwddiff<void>( (void*)grad_kernel<Output>, enzyme_dup, x, dx, enzyme_const, (const void*)gp, enzyme_dup,
                            grad, dgrad );
    for ( int row = 0; row < 8; ++row ) H[row * 8 + col] = dgrad[row];
  }
}

}  // namespace

// ============================================================================
// EnergyMortarCalculator implementations
// ============================================================================

Gparams EnergyMortarCalculator::construct_gparams( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                   const MeshData::Viewer& mesh2 ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  auto projs = compute_projection_bounds( pair, mesh1, mesh2 );
  auto bounds = smoother_.bounds_from_projections( projs, opts_.del );
  auto sb = smoother_.smooth_bounds( bounds, opts_.del, opts_.smoothing_type );
  auto qp = compute_quadrature( sb, opts_.N );

  Gparams gp;
  for ( std::size_t i = 0; i < qp.qp.size(); ++i ) {
    gp.qp[i] = qp.qp[i];
    gp.w[i] = qp.w[i];
  }

  return gp;
}

std::array<double, 2> EnergyMortarCalculator::projections( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                           const MeshData::Viewer& mesh2 ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  double projs[2];
  get_projections( A0, A1, B0, B1, projs );

  return { projs[0], projs[1] };
}

double EnergyMortarCalculator::compute_weighted_normal_gap( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                            const MeshData::Viewer& mesh2, double xiA ) const
{
  // Kept for diagnostic/FD-test callers only. The residual path now goes through
  // gap_area_kernel so eta matches the Enzyme Jacobian path (eta = -dot*dot).
  // This helper uses the same smoothed eta for consistency.
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  double nA[2], nB[2];
  find_normal( A0, A1, nA );
  find_normal( B0, B1, nB );

  double x1[2];
  iso_map( A0, A1, xiA, x1 );

  double x2[2];
  find_intersection( B0, B1, x1, nB, x2 );

  double dx = x1[0] - x2[0];
  double dy = x1[1] - x2[1];

  double gn = -( dx * nB[0] + dy * nB[1] );
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  double eta = ( dot < 0.0 ) ? -dot * dot : 0.0;

  return gn * eta;
}

NodalContactData EnergyMortarCalculator::compute_nodal_contact_data( const InterfacePair& pair,
                                                                     const MeshData::Viewer& mesh1,
                                                                     const MeshData::Viewer& mesh2 ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  const double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  Gparams gp = construct_gparams( pair, mesh1, mesh2 );

  double g_tilde_out[2];
  double area_out[2];
  gap_area_kernel( x, gp, g_tilde_out, area_out );

  if constexpr ( energy_mortar_debug_level >= 1 ) {
    assert( area_out[0] >= 0.0 && "compute_nodal_contact_data: negative tributary area node 1" );
    assert( area_out[1] >= 0.0 && "compute_nodal_contact_data: negative tributary area node 2" );
  }

  NodalContactData contact_data;
  contact_data.AI = { area_out[0], area_out[1] };
  contact_data.g_tilde = { g_tilde_out[0], g_tilde_out[1] };

  return contact_data;
}

void EnergyMortarCalculator::compute_gtilde_and_area( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                      const MeshData::Viewer& mesh2, double gtilde[2],
                                                      double area[2] ) const
{
  auto ncd = compute_nodal_contact_data( pair, mesh1, mesh2 );
  gtilde[0] = ncd.g_tilde[0];
  gtilde[1] = ncd.g_tilde[1];
  area[0] = ncd.AI[0];
  area[1] = ncd.AI[1];
}

void EnergyMortarCalculator::grad_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                          const MeshData::Viewer& mesh2, double dgt1_dx[8], double dgt2_dx[8] ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  if ( !opts_.enzyme_quadrature ) {
    Gparams gp = construct_gparams( pair, mesh1, mesh2 );
    grad_kernel<KernelOutput::GTILDE1>( x, &gp, dgt1_dx );
    grad_kernel<KernelOutput::GTILDE2>( x, &gp, dgt2_dx );
  } else {
    const auto kp = make_kernel_params( opts_ );
    grad_kernel_enzyme<KernelOutput::GTILDE1>( x, &kp, dgt1_dx );
    grad_kernel_enzyme<KernelOutput::GTILDE2>( x, &kp, dgt2_dx );
  }
}

void EnergyMortarCalculator::grad_trib_area( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                             const MeshData::Viewer& mesh2, double dA1_dx[8], double dA2_dx[8] ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  if ( !opts_.enzyme_quadrature ) {
    Gparams gp = construct_gparams( pair, mesh1, mesh2 );
    grad_kernel<KernelOutput::A1>( x, &gp, dA1_dx );
    grad_kernel<KernelOutput::A2>( x, &gp, dA2_dx );
  } else {
    const auto kp = make_kernel_params( opts_ );
    grad_kernel_enzyme<KernelOutput::A1>( x, &kp, dA1_dx );
    grad_kernel_enzyme<KernelOutput::A2>( x, &kp, dA2_dx );
  }
}

void EnergyMortarCalculator::d2_g2tilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                         const MeshData::Viewer& mesh2, double H1[64], double H2[64] ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  if ( !opts_.enzyme_quadrature ) {
    Gparams gp = construct_gparams( pair, mesh1, mesh2 );
    d2_kernel_quad<KernelOutput::GTILDE1>( x, &gp, H1 );
    d2_kernel_quad<KernelOutput::GTILDE2>( x, &gp, H2 );
  } else {
    const auto kp = make_kernel_params( opts_ );
    d2_kernel<KernelOutput::GTILDE1>( x, &kp, H1 );
    d2_kernel<KernelOutput::GTILDE2>( x, &kp, H2 );
  }
}

void EnergyMortarCalculator::compute_d2A_d2u( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                              const MeshData::Viewer& mesh2, double d2A1[64], double d2A2[64] ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  if ( !opts_.enzyme_quadrature ) {
    Gparams gp = construct_gparams( pair, mesh1, mesh2 );
    d2_kernel_quad<KernelOutput::A1>( x, &gp, d2A1 );
    d2_kernel_quad<KernelOutput::A2>( x, &gp, d2A2 );
  } else {
    const auto kp = make_kernel_params( opts_ );
    d2_kernel<KernelOutput::A1>( x, &kp, d2A1 );
    d2_kernel<KernelOutput::A2>( x, &kp, d2A2 );
  }
}

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
