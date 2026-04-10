#include "new_method.hpp"
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

static ContactSmoothing smoother( ContactParams{} );

struct Gparams {
  int N;
  const double* qp; 
  const double* w;
  const double* x2;
};

void find_normal( const double* coord1, const double* coord2, double* normal )
{
  double dx = coord2[0] - coord1[0];
  double dy = coord2[1] - coord1[1];
  double len = std::sqrt( dy * dy + dx * dx );
  dx /= len;
  dy /= len;
  normal[0] = dy;
  normal[1] = -dx;
}

void determine_legendre_nodes( int N, std::array<double, 3>& x )
{
  // x.resize( N );
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
    assert( false && "Unsupported quadrature order" );
  }
}

void determine_legendre_weights( int N, std::array<double, 3>& W )
{
  // W.resize( N );
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
    W[0] = ( 18 - std::sqrt( 30 ) ) / 36.0;
    W[1] = ( 18 + std::sqrt( 30 ) ) / 36.0;
    W[2] = ( 18 + std::sqrt( 30 ) ) / 36.0;
    W[3] = ( 18 - std::sqrt( 30 ) ) / 36.0;
  } else if ( N == 5 ) {
    W[0] = ( 322.0 - 13.0 * std::sqrt( 70.0 ) ) / 900.0;
    W[1] = ( 322.0 + 13.0 * std::sqrt( 70.0 ) ) / 900.0;
    W[2] = 128.0 / 225.0;
    W[3] = ( 322.0 + 13.0 * std::sqrt( 70.0 ) ) / 900.0;
    W[4] = ( 322.0 - 13.0 * std::sqrt( 70.0 ) ) / 900.0;
  } else {
    assert( false && "Unsupported quadrature order" );
  }
} 

void iso_map( const double* coord1, const double* coord2, double xi, double* mapped_coord )
{
  double N1 = 0.5 - xi;
  double N2 = 0.5 + xi;
  mapped_coord[0] = N1 * coord1[0] + N2 * coord2[0];
  mapped_coord[1] = N1 * coord1[1] + N2 * coord2[1];
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

void find_intersection( const double* A0, const double* A1, const double* p, const double* nB, double* intersection )
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

void get_projections( const double* A0, const double* A1, const double* B0, const double* B1, double* projections )
{
  double nB[2] = { 0.0, 0.0 };
  find_normal( B0, B1, nB );

  const double dxA = A1[0] - A0[0];
  const double dyA = A1[1] - A0[1];
  const double len2A = dxA * dxA + dyA * dyA;






  const double* B_endpoints[2] = { B0, B1 };

  double xi0 = 0.0, xi1 = 0.0;
  for ( int i = 0; i < 2; ++i ) {
    double q[2] = { 0.0, 0.0 };
    find_intersection( A0, A1, B_endpoints[i], nB, q );

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

void gtilde_kernel( const double* x, Gparams* gp, double* g_tilde_out, double* A_out )
{
  const double A0[2] = { x[0], x[1] };
  const double A1[2] = { x[2], x[3] };
  const double B0[2] = { x[4], x[5] };
  const double B1[2] = { x[6], x[7] };

  const double J = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  const double J_ref = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  double nB[2];
  find_normal( B0, B1, nB );

  double nA[2];
  find_normal( A0, A1, nA );
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  double eta = ( dot < 0 ) ? dot : 0.0;

  double g1 = 0.0, g2 = 0.0;
  double AI_1 = 0.0, AI_2 = 0.0;

  for ( int i = 0; i < gp->N; ++i ) {
    const double xiA = gp->qp[i];
    const double w = gp->w[i];

    const double N1 = 0.5 - xiA;
    const double N2 = 0.5 + xiA;

    // x1 on segment A
    double x1[2];
    iso_map( A0, A1, xiA, x1 );

    double x2[2];
    find_intersection( B0, B1, x1, nB, x2 );

    const double dx = x1[0] - x2[0];
    const double dy = x1[1] - x2[1];

    // lagged normal on B
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

//**************************************** */
// Enzyme functions for constant quadrature:

void gtilde_kernel_quad( const double* x, const Gparams* gp, double* g_tilde_out, double* A_out )
{
  const double A0[2] = { x[0], x[1] };
  const double A1[2] = { x[2], x[3] };
  const double B0[2] = { x[4], x[5] };
  const double B1[2] = { x[6], x[7] };

  const double J = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  const double J_ref = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  double nB[2];
  find_normal( B0, B1, nB );

  double nA[2];
  find_normal( A0, A1, nA );
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  double eta = ( dot < 0 ) ? dot : 0.0;

  double g1 = 0.0, g2 = 0.0;
  double AI_1 = 0.0, AI_2 = 0.0;

  for ( int i = 0; i < gp->N; ++i ) {
    const double xiA = gp->qp[i];
    const double w = gp->w[i];

    const double N1 = 0.5 - xiA;
    const double N2 = 0.5 + xiA;

    // x1 on segment A
    double x1[2];
    iso_map( A0, A1, xiA, x1 );

    double x2[2];
    find_intersection( B0, B1, x1, nB, x2 );

    const double dx = x1[0] - x2[0];
    const double dy = x1[1] - x2[1];

    // lagged normal on B
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



enum class KernelOutput { GTILDE1, GTILDE2, A1, A2 };

template <KernelOutput Output>
static void kernel_out( const double* x, const void* gp_void, double* out) {
  const Gparams* gp = static_cast<const Gparams*>( gp_void );
  double gt[2];
  double A_out[2];
  gtilde_kernel_quad( x, gp, gt, A_out);
  if constexpr ( Output == KernelOutput::GTILDE1) *out = gt[0];
  else if constexpr ( Output == KernelOutput::GTILDE2 ) *out = gt[1];
  else if constexpr ( Output == KernelOutput::A1 ) *out = A_out[0];
  else if constexpr ( Output == KernelOutput::A2 ) *out = A_out[1];
}

template <KernelOutput Output>
void grad_kernel( const double* x, const Gparams* gp, double* dout_du) {
  double dx[8] = {0.0};
  double out = 0.0;
  double dout = 1.0;

  __enzyme_autodiff<void>( (void*)kernel_out<Output>, enzyme_dup, x, dx, enzyme_const, (const void*)gp, enzyme_dup, &out, &dout );

  for ( int i = 0; i < 8; ++i ) dout_du[i] = dx[i];
}



//**************************************** */
// Enzyme functions for varying quadrature:

template <KernelOutput Output> 
static void kernel_out_enzyme( const double* x, double* out) {
  double A0[2], A1[2], B0[2], B1[2];
  A0[0] = x[0];
  A0[1] = x[1];
  A1[0] = x[2];
  A1[1] = x[3];
  B0[0] = x[4];
  B0[1] = x[5];
  B1[0] = x[6];
  B1[1] = x[7];

  double projs[2] = { 0 };
  get_projections( A0, A1, B0, B1, projs );
  std::array<double, 2> projections = { projs[0], projs[1] };

  auto bounds = ContactSmoothing::bounds_from_projections( projections, smoother.get_del() );
  auto xi_bounds = ContactSmoothing::smooth_bounds( bounds, smoother.get_del() );

  auto qp = ContactEvaluator::compute_quadrature( xi_bounds );

  const int N = static_cast<int>( qp.qp.size() );

  Gparams gp;
  gp.N = N;
  gp.qp = qp.qp.data();
  gp.w = qp.w.data();
  gp.x2 = nullptr;

  double gt[2];
  double A_out[2];
  gtilde_kernel( x, &gp, gt, A_out );

  if constexpr ( Output == KernelOutput::GTILDE1 ) *out = gt[0];
  else if constexpr ( Output == KernelOutput::GTILDE2 ) *out = gt[1];
  else if constexpr ( Output == KernelOutput::A1 )     *out = A_out[0];
  else if constexpr ( Output == KernelOutput::A2 )     *out = A_out[1];

}

template <KernelOutput Output>
void grad_kernel_enzyme( const double* x, double* dout_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  __enzyme_autodiff<void>( (void*)kernel_out_enzyme<Output>, enzyme_dup, x, dx, enzyme_dup, &out, &dout );

  for ( int i = 0; i < 8; ++i ) {
    dout_du[i] = dx[i];
  } 
}


template <KernelOutput Output>
void d2_kernel( const double* x, double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;

    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    __enzyme_fwddiff<void>( (void*)grad_kernel_enzyme<Output>, enzyme_dup, x, dx, enzyme_dup, grad, dgrad );

    for ( int row = 0; row < 8; ++row ) H[row * 8 + col] = dgrad[row];
  }
}



template <KernelOutput Output> 
void d2_kernel_quad( const double* x, const Gparams* gp, double* H ) 
{
  for ( int col = 0; col < 8; ++col) {
    double dx[8] = {0.0};
    dx[col] = 1.0;
    double grad[8] = {0.0};
    double dgrad[8] = {0.0};

    __enzyme_fwddiff<void>( (void*)grad_kernel<Output>, enzyme_dup, x, dx, enzyme_const, (const void*)gp, enzyme_dup, grad, dgrad  );
    for ( int row = 0; row < 8; ++row ) H[row * 8 + col] = dgrad[row];
  }
}

}  // namespace

std::array<double, 2> ContactEvaluator::projections( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                     const MeshData::Viewer& mesh2 ) const
{
  double A0[2];
  double A1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  double B0[2];
  double B1[2];
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  double projs[2];
  get_projections( A0, A1, B0, B1, projs );

  return { projs[0], projs[1] };
}

std::array<double, 2> ContactSmoothing::bounds_from_projections( const std::array<double, 2>& proj, double del )
{
  double xi_min = std::min( proj[0], proj[1] );
  double xi_max = std::max( proj[0], proj[1] );

  if ( xi_max < -0.5 - del ) {
    xi_max = -0.5 - del;
  }
  if ( xi_min > 0.5 + del ) {
    xi_min = 0.5 + del;
  }
  if ( xi_min < -0.5 - del ) {
    xi_min = -0.5 - del;
  }
  if ( xi_max > 0.5 + del ) {
    xi_max = 0.5 + del;
  }

  return { xi_min, xi_max };
}

std::array<double, 2> ContactSmoothing::smooth_bounds( const std::array<double, 2>& bounds, double del )
{
  std::array<double, 2> smooth_bounds;
  for ( int i = 0; i < 2; ++i ) {
    double xi = 0.0;
    double xi_hat = 0.0;
    xi = bounds[i] + 0.5;
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

QuadPoints ContactEvaluator::compute_quadrature( const std::array<double, 2>& xi_bounds )
{
  const int N = 3;
  QuadPoints out;

  std::array<double, 3> qpoints;
  std::array<double, 3> weights;

  determine_legendre_nodes( N, qpoints );
  determine_legendre_weights( N, weights );

  const double xi_min = xi_bounds[0];
  const double xi_max = xi_bounds[1];
  const double J = 0.5 * ( xi_max - xi_min );

  for ( int i = 0; i < N; ++i ) {
    out.qp[i] = 0.5 * ( xi_max - xi_min ) * qpoints[i] + 0.5 * ( xi_max + xi_min );
    out.w[i] = weights[i] * J;
  }

  return out;
}

double ContactEvaluator::gap( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                              double xiA ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  double nA[2] = { 0.0 };
  double nB[2] = { 0.0 };
  find_normal( A0, A1, nA );
  find_normal( B0, B1, nB );

  double x1[2] = { 0.0 };
  iso_map( A0, A1, xiA, x1 );

  double x2[2] = { 0.0 };
  find_intersection( B0, B1, x1, nB, x2 );

  double dx = x1[0] - x2[0];
  double dy = x1[1] - x2[1];

  double gn = -( dx * nB[0] + dy * nB[1] );  // signed normal gap
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  double eta = ( dot < 0 ) ? dot : 0.0;

  return gn * eta;
}

NodalContactData ContactEvaluator::compute_nodal_contact_data( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                               const MeshData::Viewer& mesh2 ) const
{
  double A0[2];
  double A1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );

  double J = std::sqrt( ( std::pow( ( A1[0] - A0[0] ), 2 ) + std::pow( ( A1[1] - A0[1] ), 2 ) ) );
  double J_ref = std::sqrt( std::pow( A1[0] - A0[0], 2 ) + std::pow( A1[1] - A0[1], 2 ) );

  auto projs = projections( pair, mesh1, mesh2 );

  auto bounds = smoother_.bounds_from_projections( projs, smoother.get_del() );
  auto smooth_bounds = smoother_.smooth_bounds( bounds, smoother.get_del() );

  auto qp = compute_quadrature( smooth_bounds );


  double g_tilde1 = 0.0;
  double g_tilde2 = 0.0;
  double AI_1 = 0.0;
  double AI_2 = 0.0;

  for ( size_t i = 0; i < qp.qp.size(); ++i ) {
    double xiA = qp.qp[i];
    double w = qp.w[i];
    double N1 = 0.5 - xiA;
    double N2 = 0.5 + xiA;
    double gn = gap( pair, mesh1, mesh2, xiA );
    double gn_active = gn;


    g_tilde1 += w * N1 * gn_active * J;
    g_tilde2 += w * N2 * gn_active * J;

    AI_1 += w * N1 * J_ref;
    AI_2 += w * N2 * J_ref;
  }

  NodalContactData contact_data;

  contact_data.AI = { AI_1, AI_2 };
  contact_data.g_tilde = { g_tilde1, g_tilde2 };

  return contact_data;
}

std::array<double, 2> ContactEvaluator::compute_pressures( const NodalContactData& ncd ) const
{
  double gt1 = ncd.g_tilde[0];
  double gt2 = ncd.g_tilde[1];

  double A1 = ncd.AI[0];
  double A2 = ncd.AI[1];

  double g1 = gt1 / A1;
  double g2 = gt2 / A2;

  // //KKT Conditons
  double p1 = ( g1 < 0.0 ) ? p_.k * g1 : 0.0;
  double p2 = ( g2 < 0.0 ) ? p_.k * g2 : 0.0;
  std::array<double, 2> pressures;

  pressures = { p1, p2 };

  for ( int i = 0; i < 2; ++i ) {
    if ( ncd.AI[i] < 1e-12 ) {
      pressures[i] = 0.0;
    }
  }

  return pressures;
}

double ContactEvaluator::compute_contact_energy( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                 const MeshData::Viewer& mesh2 ) const
{
  NodalContactData contact_data;
  contact_data = compute_nodal_contact_data( pair, mesh1, mesh2 );

  std::array<double, 2> pressures;
  pressures = compute_pressures( contact_data );

  double contact_energy = pressures[0] * contact_data.g_tilde[0] + pressures[1] * contact_data.g_tilde[1];
  return contact_energy;
}

void ContactEvaluator::gtilde_and_area( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                        const MeshData::Viewer& mesh2, double gtilde[2], double area[2] ) const
{
  auto ncd = compute_nodal_contact_data( pair, mesh1, mesh2 );
  gtilde[0] = ncd.g_tilde[0];
  gtilde[1] = ncd.g_tilde[1];
  area[0] = ncd.AI[0];
  area[1] = ncd.AI[1];
}

void ContactEvaluator::grad_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                    const MeshData::Viewer& mesh2, double dgt1_dx[8], double dgt2_dx[8] ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };


  double nB[2], nA[2];
  find_normal( B0, B1, nB );
  find_normal( A0, A1, nA );


  double dg1_du[8] = { 0.0 };
  double dg2_du[8] = { 0.0 };

  if ( !p_.enzyme_quadrature ) {
    auto projs = projections( pair, mesh1, mesh2 );

    auto bounds = smoother_.bounds_from_projections( projs, smoother.get_del() );
    auto smooth_bounds = smoother_.smooth_bounds( bounds, smoother.get_del() );

    auto qp = compute_quadrature( smooth_bounds );

    const int N = static_cast<int>( qp.qp.size() );

    std::vector<double> x2( 2 * N );

    for ( int i = 0; i < N; ++i ) {
      double x1[2] = { 0.0 };
      iso_map( A0, A1, qp.qp[i], x1 );
      double x2_i[2] = { 0.0 };
      find_intersection( B0, B1, x1, nB, x2_i );
      x2[2 * i] = x2_i[0];
      x2[2 * i + 1] = x2_i[1];
    }

    Gparams gp;
    gp.N = N;
    gp.qp = qp.qp.data();
    gp.w = qp.w.data();
    gp.x2 = x2.data();

    grad_kernel<KernelOutput::GTILDE1>( x, &gp, dg1_du );
    grad_kernel<KernelOutput::GTILDE2>(  x, &gp, dg2_du);
    // grad_gtilde1_quad( x, &gp, dg1_du );
    // grad_gtilde2_quad( x, &gp, dg2_du );

  } else {
    // grad_gtilde1( x, dg1_du );
    // grad_gtilde2( x, dg2_du );
    grad_kernel_enzyme<KernelOutput::GTILDE1>(x, dg1_du);
    grad_kernel_enzyme<KernelOutput::GTILDE2>(x, dg2_du);
  }

  for ( int i = 0; i < 8; ++i ) {
    dgt1_dx[i] = dg1_du[i];
    dgt2_dx[i] = dg2_du[i];
  }
}

void ContactEvaluator::grad_trib_area( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                       const MeshData::Viewer& mesh2, double dA1_dx[8], double dA2_dx[8] ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };


  double nB[2], nA[2];
  find_normal( B0, B1, nB );
  find_normal( A0, A1, nA );



  if ( !p_.enzyme_quadrature ) {
    auto projs = projections( pair, mesh1, mesh2 );
    auto bounds = smoother_.bounds_from_projections( projs, smoother.get_del() );
    auto smooth_bounds = smoother_.smooth_bounds( bounds, smoother.get_del() );

    auto qp = compute_quadrature( smooth_bounds );

    const int N = static_cast<int>( qp.qp.size() );

    std::vector<double> x2( 2 * N );

    for ( int i = 0; i < N; ++i ) {
      double x1[2] = { 0.0 };
      iso_map( A0, A1, qp.qp[i], x1 );
      double x2_i[2] = { 0.0 };
      find_intersection( B0, B1, x1, nB, x2_i );
      x2[2 * i] = x2_i[0];
      x2[2 * i + 1] = x2_i[1];
    }

    Gparams gp;
    gp.N = N;
    gp.qp = qp.qp.data();
    gp.w = qp.w.data();
    gp.x2 = x2.data();


    grad_kernel<KernelOutput::A1>( x, &gp, dA1_dx);
    grad_kernel<KernelOutput::A2>( x, &gp, dA2_dx);
    // grad_A1_quad( x, &gp, dA1_dx );
    // grad_A2_quad( x, &gp, dA2_dx );
  } else {

    grad_kernel_enzyme<KernelOutput::A1>(x, dA1_dx);
    grad_kernel_enzyme<KernelOutput::A2>(x, dA2_dx);
    // grad_A1( x, dA1_dx );
    // grad_A2( x, dA2_dx );
  }
}

std::array<double, 8> ContactEvaluator::compute_contact_forces( const InterfacePair& pair,
                                                                const MeshData::Viewer& mesh1,
                                                                const MeshData::Viewer& mesh2 ) const
{
  double dg_tilde1[8] = { 0.0 };
  double dg_tilde2[8] = { 0.0 };
  double dA1[8] = { 0.0 };
  double dA2[8] = { 0.0 };
  std::array<double*, 2> dg_t;
  std::array<double*, 2> dA_I;
  dg_t = { dg_tilde1, dg_tilde2 };
  dA_I = { dA1, dA2 };

  grad_gtilde( pair, mesh1, mesh2, dg_tilde1, dg_tilde2 );
  grad_trib_area( pair, mesh1, mesh2, dA1, dA2 );

  NodalContactData ncd;
  ncd = compute_nodal_contact_data( pair, mesh1, mesh2 );

  std::array<double, 2> pressures;
  pressures = compute_pressures( ncd );

  std::array<double, 8> f = { 0.0 };

  for ( int i = 0; i < 8; ++i ) {
    for ( int j = 0; j < 2; ++j ) {
      double g = 0.0;
      g = ncd.g_tilde[j] / ncd.AI[j];
      if ( ncd.AI[j] < 1e-12 ) {
        g = 0.0;
      }
      f[i] += ( 2 * pressures[j] * dg_t[j][i] - pressures[j] * g * dA_I[j][i] );
    }
  }
  return f;
}

void ContactEvaluator::d2_g2tilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                   const MeshData::Viewer& mesh2, double H1[64], double H2[64] ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };


  double nB[2], nA[2];
  find_normal( B0, B1, nB );
  find_normal( A0, A1, nA );

  double d2g1_d2u[64] = { 0.0 };
  double d2g2_d2u[64] = { 0.0 };

  if ( !p_.enzyme_quadrature ) {
    auto projs = projections( pair, mesh1, mesh2 );
    auto bounds = smoother_.bounds_from_projections( projs, smoother.get_del() );
    auto smooth_bounds = smoother_.smooth_bounds( bounds, smoother.get_del() );

    auto qp = compute_quadrature( smooth_bounds );

    const int N = static_cast<int>( qp.qp.size() );

    std::vector<double> x2( 2 * N );

    for ( int i = 0; i < N; ++i ) {
      double x1[2] = { 0.0 };
      iso_map( A0, A1, qp.qp[i], x1 );
      double x2_i[2] = { 0.0 };
      find_intersection( B0, B1, x1, nB, x2_i );
      x2[2 * i] = x2_i[0];
      x2[2 * i + 1] = x2_i[1];
    }

    Gparams gp;
    gp.N = N;
    gp.qp = qp.qp.data();
    gp.w = qp.w.data();
    gp.x2 = x2.data();

    d2_kernel_quad<KernelOutput::GTILDE1>(x,&gp,d2g1_d2u);
    d2_kernel_quad<KernelOutput::GTILDE2>(x,&gp,d2g2_d2u);



    // grad_A1_quad( x, &gp, dA1_dx );
    // grad_A2_quad( x, &gp, dA2_dx );
  } 
  else{
    d2_kernel<KernelOutput::GTILDE1>( x, d2g1_d2u);
    d2_kernel<KernelOutput::GTILDE2>( x, d2g2_d2u);
  }

  // d2gtilde1( x, d2g1_d2u );
  // d2gtilde2( x, d2g2_d2u );

  for ( int i = 0; i < 64; ++i ) {
    H1[i] = d2g1_d2u[i];
    H2[i] = d2g2_d2u[i];
  }
}

void ContactEvaluator::compute_d2A_d2u( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                        const MeshData::Viewer& mesh2, double d2A1[64], double d2A2[64] ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };


  double nB[2], nA[2];

  find_normal( B0, B1, nB );
  find_normal( A0, A1, nA );

  double d2A1_d2u[64] = { 0.0 };
  double d2A2_d2u[64] = { 0.0 };

    if ( !p_.enzyme_quadrature ) {
    auto projs = projections( pair, mesh1, mesh2 );
    auto bounds = smoother_.bounds_from_projections( projs, smoother.get_del() );
    auto smooth_bounds = smoother_.smooth_bounds( bounds, smoother.get_del() );

    auto qp = compute_quadrature( smooth_bounds );

    const int N = static_cast<int>( qp.qp.size() );

    std::vector<double> x2( 2 * N );

    for ( int i = 0; i < N; ++i ) {
      double x1[2] = { 0.0 };
      iso_map( A0, A1, qp.qp[i], x1 );
      double x2_i[2] = { 0.0 };
      find_intersection( B0, B1, x1, nB, x2_i );
      x2[2 * i] = x2_i[0];
      x2[2 * i + 1] = x2_i[1];
    }

    Gparams gp;
    gp.N = N;
    gp.qp = qp.qp.data();
    gp.w = qp.w.data();
    gp.x2 = x2.data();

    d2_kernel_quad<KernelOutput::A1>(x,&gp,d2A1_d2u);
    d2_kernel_quad<KernelOutput::A2>(x,&gp,d2A2_d2u);



    // grad_A1_quad( x, &gp, dA1_dx );
    // grad_A2_quad( x, &gp, dA2_dx );
  } 
    else {
  d2_kernel<KernelOutput::A1>( x, d2A1_d2u);
  d2_kernel<KernelOutput::A2>( x, d2A2_d2u);
  }
  

  // get_d2A1( x, d2A1_d2u );
  // get_d2A2( x, d2A2_d2u );

  for ( int i = 0; i < 64; ++i ) {
    d2A1[i] = d2A1_d2u[i];
    d2A2[i] = d2A2_d2u[i];
  }
}

std::array<std::array<double, 8>, 8> ContactEvaluator::compute_stiffness_matrix( const InterfacePair& pair,
                                                                                 const MeshData::Viewer& mesh1,
                                                                                 const MeshData::Viewer& mesh2 ) const
{
  NodalContactData ncd;
  ncd = compute_nodal_contact_data( pair, mesh1, mesh2 );

  std::array<double, 2> gI;
  for ( int i = 0; i < 2; ++i ) {
    gI[i] = ncd.g_tilde[i] / ncd.AI[i];
  }

  double dg_tilde1[8], dg_tilde2[8], dAI1[8], dAI2[8];

  grad_gtilde( pair, mesh1, mesh2, dg_tilde1, dg_tilde2 );
  grad_trib_area( pair, mesh1, mesh2, dAI1, dAI2 );

  double d2_gtilde1[64], d2_gtilde2[64], d2_dA1[64], d2_dA2[64];

  d2_g2tilde( pair, mesh1, mesh2, d2_gtilde1, d2_gtilde2 );
  compute_d2A_d2u( pair, mesh1, mesh2, d2_dA1, d2_dA2 );

  std::array<double*, 2> dg_t = { dg_tilde1, dg_tilde2 };
  std::array<double*, 2> dA = { dAI1, dAI2 };

  std::array<double*, 2> ddg_t = { d2_gtilde1, d2_gtilde2 };
  std::array<double*, 2> ddA = { d2_dA1, d2_dA2 };

  std::array<std::array<double, 8>, 8> K_mat = { { { 0.0 } } };

  for ( int i = 0; i < 2; ++i ) {
    for ( int k = 0; k < 8; ++k ) {
      for ( int j = 0; j < 8; ++j ) {
        // term 1:
        K_mat[k][j] += p_.k * ( 2 / ncd.AI[i] ) * dg_t[i][k] * dg_t[i][j];

        // term2:
        K_mat[k][j] += -p_.k * ( 2 * gI[i] / ncd.AI[i] ) * dg_t[i][k] * dA[i][j];

        // term3:
        K_mat[k][j] += -p_.k * ( 2 * gI[i] / ncd.AI[i] ) * dA[i][k] * dg_t[i][j];

        // term 4:
        K_mat[k][j] += p_.k * ( 2 * gI[i] * gI[i] / ncd.AI[i] ) * dA[i][k] * dA[i][j];

        // term 5;
        K_mat[k][j] += p_.k * 2.0 * gI[i] * ddg_t[i][k * 8 + j];

        // term 6:
        K_mat[k][j] += -p_.k * gI[i] * gI[i] * ddA[i][k * 8 + j];

        if ( ncd.AI[i] < 1e-12 ) {
          K_mat[k][j] = 0.0;
        }
      }
    }
  }
  return K_mat;
}

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
