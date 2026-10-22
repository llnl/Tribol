// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortar.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

namespace {

// Compute unit normal vector pointing to the left of the directed segment coord1 -> coord2
void findNormal( const double* coord1, const double* coord2, double* normal )
{
  double dx = coord2[0] - coord1[0];
  double dy = coord2[1] - coord1[1];
  double len = std::sqrt( dy * dy + dx * dx );
  dx /= len;
  dy /= len;
  normal[0] = dy;
  normal[1] = -dx;
}

// Compute Gauss-Legendre quadrature nodes on reference interval [-1, 1]
void determineLegendreNodes( int N, std::array<double, 5>& x )
{
  if ( N == 1 ) {
    x[0] = 0.0;
  } else if ( N == 2 ) {
    const double a = 1.0 / std::sqrt( 3.0 );
    x[0] = -a;
    x[1] = a;
  } else if ( N == 3 ) {
    const double a = std::sqrt( 3.0 / 5.0 );
    x[0] = -a;
    x[1] = 0.0;
    x[2] = a;
  } else if ( N == 4 ) {
    const double a = std::sqrt( ( 3.0 - 2.0 * std::sqrt( 6.0 / 5.0 ) ) / 7.0 );
    const double b = std::sqrt( ( 3.0 + 2.0 * std::sqrt( 6.0 / 5.0 ) ) / 7.0 );
    x[0] = -b;
    x[1] = -a;
    x[2] = a;
    x[3] = b;
  } else if ( N == 5 ) {
    const double a = std::sqrt( 5.0 - 2.0 * std::sqrt( 10.0 / 7.0 ) ) / 3.0;
    const double b = std::sqrt( 5.0 + 2.0 * std::sqrt( 10.0 / 7.0 ) ) / 3.0;
    x[0] = -b;
    x[1] = -a;
    x[2] = 0.0;
    x[3] = a;
    x[4] = b;
  } else {
    assert( false && "Unsupported quadrature order (expected 1 <= N <= 5)" );
  }
}

// Compute Gauss-Legendre quadrature weights on reference interval [-1, 1]
void determineLegendreWeights( int N, std::array<double, 5>& W )
{
  if ( N == 1 ) {
    W[0] = 2.0;
  } else if ( N == 2 ) {
    W[0] = 1.0;
    W[1] = 1.0;
  } else if ( N == 3 ) {
    W[0] = 5.0 / 9.0;
    W[1] = 8.0 / 9.0;
    W[2] = 5.0 / 9.0;
  } else if ( N == 4 ) {
    W[0] = ( 18.0 - std::sqrt( 30.0 ) ) / 36.0;
    W[1] = ( 18.0 + std::sqrt( 30.0 ) ) / 36.0;
    W[2] = ( 18.0 + std::sqrt( 30.0 ) ) / 36.0;
    W[3] = ( 18.0 - std::sqrt( 30.0 ) ) / 36.0;
  } else if ( N == 5 ) {
    W[0] = ( 322.0 - 13.0 * std::sqrt( 70.0 ) ) / 900.0;
    W[1] = ( 322.0 + 13.0 * std::sqrt( 70.0 ) ) / 900.0;
    W[2] = 128.0 / 225.0;
    W[3] = ( 322.0 + 13.0 * std::sqrt( 70.0 ) ) / 900.0;
    W[4] = ( 322.0 - 13.0 * std::sqrt( 70.0 ) ) / 900.0;
  } else {
    assert( false && "Unsupported quadrature order (expected 1 <= N <= 5)" );
  }
}

// Map a 1D parent coordinate xi in [-0.5, 0.5] to physical 2D segment coordinates
void isoMap( const double* coord1, const double* coord2, double xi, double* mapped_coord )
{
  double N1 = 0.5 - xi;
  double N2 = 0.5 + xi;
  mapped_coord[0] = N1 * coord1[0] + N2 * coord2[0];
  mapped_coord[1] = N1 * coord1[1] + N2 * coord2[1];
}

// Extract physical endpoint coordinates of an element face
inline void endpoints( const MeshData::Viewer& mesh, int elem_id, double P0[2], double P1[2] )
{
  double P0_P1[4];
  mesh.getFaceCoords( elem_id, P0_P1 );
  P0[0] = P0_P1[0];
  P0[1] = P0_P1[1];
  P1[0] = P0_P1[2];
  P1[1] = P0_P1[3];
}

// Find intersection of the ray from point p along direction nB with the line through A0 and A1
void findIntersection( const double* A0, const double* A1, const double* p, const double* nB, double* intersection )
{
  const double tA[2] = { A1[0] - A0[0], A1[1] - A0[1] };
  const double d[2] = { p[0] - A0[0], p[1] - A0[1] };

  const double nlen = std::sqrt( nB[0] * nB[0] + nB[1] * nB[1] );
  if ( nlen < 1e-14 ) {
    intersection[0] = p[0];
    intersection[1] = p[1];
    return;
  }
  const double n[2] = { nB[0] / nlen, nB[1] / nlen };

  const double det = tA[0] * n[1] - tA[1] * n[0];

  if ( std::abs( det ) < 1e-12 ) {
    intersection[0] = p[0];
    intersection[1] = p[1];
    return;
  }

  const double inv_det = 1.0 / det;
  double alpha = ( d[0] * n[1] - d[1] * n[0] ) * inv_det;

  intersection[0] = A0[0] + alpha * tA[0];
  intersection[1] = A0[1] + alpha * tA[1];
}

// Project endpoints of edge B onto edge A along normal nB and return parameter interval [xi_min, xi_max] on A
void getProjections( const double* A0, const double* A1, const double* B0, const double* B1, double* projections )
{
  double nB[2] = { 0.0, 0.0 };
  findNormal( B0, B1, nB );

  const double dxA = A1[0] - A0[0];
  const double dyA = A1[1] - A0[1];
  const double len2A = dxA * dxA + dyA * dyA;

  const double* B_endpoints[2] = { B0, B1 };

  double xi0 = 0.0, xi1 = 0.0;
  for ( int i = 0; i < 2; ++i ) {
    double q[2] = { 0.0, 0.0 };
    findIntersection( A0, A1, B_endpoints[i], nB, q );
    const double alphaA = ( ( q[0] - A0[0] ) * dxA + ( q[1] - A0[1] ) * dyA ) / len2A;
    const double xiA = alphaA - 0.5;

    if ( i == 0 )
      xi0 = xiA;
    else
      xi1 = xiA;
  }

  double xi_min = std::min( xi0, xi1 );
  double xi_max = std::max( xi0, xi1 );

  projections[0] = xi_min;
  projections[1] = xi_max;
}

// Compute smoothed nodal gaps g_tilde and tributary areas A given fixed quadrature points/weights
void energyMortarKernelQuad( const double* x, const void* gp_void, double* g_tilde_out, double* A_out )
{
  const Gparams* gp = static_cast<const Gparams*>( gp_void );
  const double A0[2] = { x[0], x[1] };
  const double A1[2] = { x[2], x[3] };
  const double B0[2] = { x[4], x[5] };
  const double B1[2] = { x[6], x[7] };

  const double J = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );
  const double J_ref = J;

  double nB[2];
  findNormal( B0, B1, nB );

  double nA[2];
  findNormal( A0, A1, nA );
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  double eta = dot;

  double g1 = 0.0, g2 = 0.0;
  double AI_1 = 0.0, AI_2 = 0.0;

  for ( int i = 0; i < gp->N; ++i ) {
    const double xiA = gp->qp[i];
    const double w = gp->w[i];

    const double N1 = 0.5 - xiA;
    const double N2 = 0.5 + xiA;

    double x1[2];
    isoMap( A0, A1, xiA, x1 );

    double x2[2];
    findIntersection( B0, B1, x1, nB, x2 );

    const double dx = x1[0] - x2[0];
    const double dy = x1[1] - x2[1];

    const double gn = -( dx * nB[0] + dy * nB[1] );
    const double g = gn * eta;

    g1 += w * N1 * g * J;
    g2 += w * N2 * g * J;

    AI_1 += w * N1 * J_ref;
    AI_2 += w * N2 * J_ref;
  }

  g_tilde_out[0] = g1;
  g_tilde_out[1] = g2;
  A_out[0] = AI_1;
  A_out[1] = AI_2;
}

// Compute smoothed nodal gaps and tributary areas while differentiating through geometry-dependent quadrature
void energyMortarKernelEnzyme( const double* x, const ContactParams* params, double* g_tilde_out, double* A_out )
{
  double A0[2] = { x[0], x[1] };
  double A1[2] = { x[2], x[3] };
  double B0[2] = { x[4], x[5] };
  double B1[2] = { x[6], x[7] };

  double projs[2] = { 0 };
  getProjections( A0, A1, B0, B1, projs );
  std::array<double, 2> projections = { projs[0], projs[1] };

  auto bounds = ContactSmoothing::boundsFromProjections( projections, params->del );
  auto xi_bounds = ContactSmoothing::smoothBounds( bounds, params->del );

  auto qp = EnergyMortarCalculator::computeQuadrature( xi_bounds, params->N );

  energyMortarKernelQuad( x, &qp, g_tilde_out, A_out );
}

// Scalar wrappers for second-derivative autodiff passes
template <int OutIdx>
void scalarKernelQuad( const double* x, const void* gp_void, double* out )
{
  double gt[2];
  double A[2];
  energyMortarKernelQuad( x, gp_void, gt, A );
  if constexpr ( OutIdx == 0 )
    *out = gt[0];
  else if constexpr ( OutIdx == 1 )
    *out = gt[1];
  else if constexpr ( OutIdx == 2 )
    *out = A[0];
  else if constexpr ( OutIdx == 3 )
    *out = A[1];
}

template <int OutIdx>
void scalarKernelEnzyme( const double* x, const ContactParams* params, double* out )
{
  double gt[2];
  double A[2];
  energyMortarKernelEnzyme( x, params, gt, A );
  if constexpr ( OutIdx == 0 )
    *out = gt[0];
  else if constexpr ( OutIdx == 1 )
    *out = gt[1];
  else if constexpr ( OutIdx == 2 )
    *out = A[0];
  else if constexpr ( OutIdx == 3 )
    *out = A[1];
}

template <int OutIdx>
void gradScalarQuad( const double* x, const Gparams* gp, double* grad )
{
  double dx[8] = { 0.0 };
  double out = 0.0, dout = 1.0;
  __enzyme_autodiff<void>( (void*)scalarKernelQuad<OutIdx>, enzyme_dup, x, dx, enzyme_const, (const void*)gp,
                           enzyme_dup, &out, &dout );
  for ( int i = 0; i < 8; ++i ) {
    grad[i] = dx[i];
  }
}

template <int OutIdx>
void gradScalarEnzyme( const double* x, const ContactParams* params, double* grad )
{
  double dx[8] = { 0.0 };
  double out = 0.0, dout = 1.0;
  __enzyme_autodiff<void>( (void*)scalarKernelEnzyme<OutIdx>, enzyme_dup, x, dx, enzyme_const, params, enzyme_dup,
                           &out, &dout );
  for ( int i = 0; i < 8; ++i ) {
    grad[i] = dx[i];
  }
}

template <int OutIdx>
void hessianQuad( const double* x, const Gparams* gp, double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;
    double grad[8] = { 0.0 }, dgrad[8] = { 0.0 };
    __enzyme_fwddiff<void>( (void*)gradScalarQuad<OutIdx>, enzyme_dup, x, dx, enzyme_const, (const void*)gp,
                            enzyme_dup, grad, dgrad );
    for ( int row = 0; row < 8; ++row ) {
      H[row * 8 + col] = dgrad[row];
    }
  }
}

template <int OutIdx>
void hessianEnzyme( const double* x, const ContactParams* params, double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;
    double grad[8] = { 0.0 }, dgrad[8] = { 0.0 };
    __enzyme_fwddiff<void>( (void*)gradScalarEnzyme<OutIdx>, enzyme_dup, x, dx, enzyme_const, params, enzyme_dup,
                            grad, dgrad );
    for ( int row = 0; row < 8; ++row ) {
      H[row * 8 + col] = dgrad[row];
    }
  }
}

}  // namespace

Gparams EnergyMortarCalculator::constructGparams( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                   const MeshData::Viewer& mesh2 ) const
{
  auto projs = projections( pair, mesh1, mesh2 );
  auto bounds = smoother_.boundsFromProjections( projs, p_.del );
  auto smooth_bounds = smoother_.smoothBounds( bounds, p_.del );

  return EnergyMortarCalculator::computeQuadrature( smooth_bounds, p_.N );
}

std::array<double, 2> EnergyMortarCalculator::projections( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                            const MeshData::Viewer& mesh2 ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  double projs[2];
  getProjections( A0, A1, B0, B1, projs );

  return { projs[0], projs[1] };
}

std::array<double, 2> ContactSmoothing::boundsFromProjections( const std::array<double, 2>& proj, double del )
{
  double xi_min = std::min( proj[0], proj[1] );
  double xi_max = std::max( proj[0], proj[1] );

  if ( xi_max < -0.5 - del ) xi_max = -0.5 - del;
  if ( xi_min > 0.5 + del ) xi_min = 0.5 + del;
  if ( xi_min < -0.5 - del ) xi_min = -0.5 - del;
  if ( xi_max > 0.5 + del ) xi_max = 0.5 + del;

  return { xi_min, xi_max };
}

std::array<double, 2> ContactSmoothing::smoothBounds( const std::array<double, 2>& bounds, double del )
{
  std::array<double, 2> smooth_bounds;
  for ( int i = 0; i < 2; ++i ) {
    double xi = bounds[i] + 0.5;
    double xi_hat = 0.0;

    if ( del == 0.0 ) {
      xi_hat = xi;
    } else {
      if ( 0.0 - del <= xi && xi <= del ) {
        xi_hat = ( 1.0 / ( 4 * del ) ) * ( xi * xi ) + 0.5 * xi + del / 4.0;
      } else if ( ( 1.0 - del ) <= xi && xi <= 1.0 + del ) {
        double b = -1.0 / ( 4.0 * del );
        double c = 0.5 + 1.0 / ( 2.0 * del );
        double d = 1.0 - del + ( 1.0 / ( 4.0 * del ) ) * pow( 1.0 - del, 2 ) - 0.5 * ( 1.0 - del ) -
                   ( 1.0 - del ) / ( 2.0 * del );
        xi_hat = b * xi * xi + c * xi + d;
      } else if ( del <= xi && xi <= ( 1.0 - del ) ) {
        xi_hat = xi;
      }
    }
    smooth_bounds[i] = xi_hat - 0.5;
  }
  return smooth_bounds;
}

QuadPoints EnergyMortarCalculator::computeQuadrature( const std::array<double, 2>& xi_bounds, int N )
{
  QuadPoints out;
  out.N = N;
  std::array<double, 5> qpoints{};
  std::array<double, 5> weights{};

  determineLegendreNodes( N, qpoints );
  determineLegendreWeights( N, weights );

  const double xi_min = xi_bounds[0];
  const double xi_max = xi_bounds[1];
  const double J = 0.5 * ( xi_max - xi_min );

  for ( int i = 0; i < N; ++i ) {
    out.qp[i] = 0.5 * ( xi_max - xi_min ) * qpoints[i] + 0.5 * ( xi_max + xi_min );
    out.w[i] = weights[i] * J;
  }
  return out;
}

void EnergyMortarCalculator::computeContactQuantities( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                       const MeshData::Viewer& mesh2, double gtilde[2],
                                                       double area[2], double dgt1_dx[8], double dgt2_dx[8],
                                                       double dA1_dx[8], double dA2_dx[8] ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );
  
  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  if ( !p_.enzyme_quadrature ) {
    Gparams gp = constructGparams( pair, mesh1, mesh2 );
    energyMortarKernelQuad( x, &gp, gtilde, area );

    for ( int col = 0; col < 8; ++col ) {
      double dx[8] = { 0.0 }; dx[col] = 1.0;
      double gt[2] = { 0.0 }, dgt[2] = { 0.0 };
      double A[2] = { 0.0 }, dA[2] = { 0.0 };

      __enzyme_fwddiff<void>( (void*)energyMortarKernelQuad, enzyme_dup, x, dx, enzyme_const, (const void*)&gp,
                              enzyme_dup, gt, dgt, enzyme_dup, A, dA );

      dgt1_dx[col] = dgt[0];
      dgt2_dx[col] = dgt[1];
      dA1_dx[col] = dA[0];
      dA2_dx[col] = dA[1];
    }
  } else {
    energyMortarKernelEnzyme( x, &p_, gtilde, area );

    for ( int col = 0; col < 8; ++col ) {
      double dx[8] = { 0.0 }; dx[col] = 1.0;
      double gt[2] = { 0.0 }, dgt[2] = { 0.0 };
      double A[2] = { 0.0 }, dA[2] = { 0.0 };

      __enzyme_fwddiff<void>( (void*)energyMortarKernelEnzyme, enzyme_dup, x, dx, enzyme_const, &p_,
                              enzyme_dup, gt, dgt, enzyme_dup, A, dA );

      dgt1_dx[col] = dgt[0];
      dgt2_dx[col] = dgt[1];
      dA1_dx[col] = dA[0];
      dA2_dx[col] = dA[1];
    }
  }
}

void EnergyMortarCalculator::computeSecondDerivatives( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                       const MeshData::Viewer& mesh2, double d2gt1_dx2[64],
                                                       double d2gt2_dx2[64], double d2A1_dx2[64], double d2A2_dx2[64] ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );
  
  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  if ( !p_.enzyme_quadrature ) {
    Gparams gp = constructGparams( pair, mesh1, mesh2 );
    if ( d2gt1_dx2 ) hessianQuad<0>( x, &gp, d2gt1_dx2 );
    if ( d2gt2_dx2 ) hessianQuad<1>( x, &gp, d2gt2_dx2 );
    if ( d2A1_dx2 )  hessianQuad<2>( x, &gp, d2A1_dx2 );
    if ( d2A2_dx2 )  hessianQuad<3>( x, &gp, d2A2_dx2 );
  } else {
    if ( d2gt1_dx2 ) hessianEnzyme<0>( x, &p_, d2gt1_dx2 );
    if ( d2gt2_dx2 ) hessianEnzyme<1>( x, &p_, d2gt2_dx2 );
    if ( d2A1_dx2 )  hessianEnzyme<2>( x, &p_, d2A1_dx2 );
    if ( d2A2_dx2 )  hessianEnzyme<3>( x, &p_, d2A2_dx2 );
  }
}

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol

