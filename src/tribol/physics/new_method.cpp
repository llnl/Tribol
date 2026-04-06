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
  } else if (N ==4) {
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
  const double lenA = std::sqrt(dxA * dxA + dyA * dyA);
  // std::cout << "Length A: " << lenA << std::endl;

  const double dxB = B1[0] - B0[0];
  const double dyB = B1[1] - B0[1];
  const double lenB = std::sqrt(dxB * dxB + dyB * dyB);
  // std::cout << "Length B: " << lenB << std::endl;


  const double* B_endpoints[2] = { B0, B1 };

  double xi0 = 0.0, xi1 = 0.0;
  for ( int i = 0; i < 2; ++i ) {
    double q[2] = { 0.0, 0.0 };
    find_intersection( A0, A1, B_endpoints[i], nB, q );

    // std::cout << "Intersection on A: " << q[0] << ", " << q[1] << std::endl;

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

void gtilde_kernel( const double* x,  Gparams* gp, double* g_tilde_out, const double* nB, double length_A, double* A_out)
{
  const double A0[2] = { x[0], x[1] };
  const double A1[2] = { x[2], x[3] };
  const double B0[2] = { x[4], x[5] };
  const double B1[2] = { x[6], x[7] };

  const double J = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  const double J_ref = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  // double nB[2];
  // find_normal( B0, B1, nB );

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
  // std::cout << "G tilde: " << g1 << ", " << g2 << std::endl;
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
  // std::cout << "G tilde: " << g1 << ", " << g2 << std::endl;
}

static void gtilde1_out_quad( const double* x, const void* gp_void, double* out )
{
  const Gparams* gp = static_cast<const Gparams*>( gp_void );
  double gt[2];
  double A_out[2];
  gtilde_kernel_quad( x, gp, gt, A_out );
  *out = gt[0];
}

void grad_gtilde1_quad( const double* x, const Gparams* gp, double* dgt1_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  __enzyme_autodiff<void>( (void*)gtilde1_out_quad, enzyme_dup, x, dx, enzyme_const, (const void*)gp, enzyme_dup, &out,
                           &dout );

  for ( int i = 0; i < 8; ++i ) {
    dgt1_du[i] = dx[i];
  }
}

static void gtilde2_out_quad( const double* x, const void* gp_void, double* out )
{
  const Gparams* gp = static_cast<const Gparams*>( gp_void );
  double gt[2];
  double A_out[2];
  gtilde_kernel_quad( x, gp, gt, A_out );
  *out = gt[1];
}

static void A1_out_quad( const double* x, const void* gp_void, double* out )
{
  const Gparams* gp = static_cast<const Gparams*>( gp_void );
  double gt[2];
  double A_out[2];
  gtilde_kernel_quad( x, gp, gt, A_out );
  *out = A_out[0];
}

static void A2_out_quad( const double* x, const void* gp_void, double* out )
{
  const Gparams* gp = static_cast<const Gparams*>( gp_void );
  double gt[2];
  double A_out[2];
  gtilde_kernel_quad( x, gp, gt, A_out );
  *out = A_out[1];
}

void grad_gtilde2_quad( const double* x, const Gparams* gp, double* dgt2_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  __enzyme_autodiff<void>( (void*)gtilde2_out_quad, enzyme_dup, x, dx, enzyme_const, (const void*)gp, enzyme_dup, &out,
                           &dout );

  for ( int i = 0; i < 8; ++i ) {
    dgt2_du[i] = dx[i];
  }
}

void grad_A1_quad( const double* x, const Gparams* gp, double* dA1_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  __enzyme_autodiff<void>( (void*)A1_out_quad, enzyme_dup, x, dx, enzyme_const, (const void*)gp, enzyme_dup, &out, &dout );

  for ( int i = 0; i < 8; ++i ) {
    dA1_du[i] = dx[i];
  }
}

void grad_A2_quad( const double* x, const Gparams* gp, double* dA2_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  __enzyme_autodiff<void>( (void*)A2_out_quad, enzyme_dup, x, dx, enzyme_const, (const void*)gp, enzyme_dup, &out, &dout );

  for ( int i = 0; i < 8; ++i ) {
    dA2_du[i] = dx[i];
  }
}

//**************************************** */
// Enzyme functions for varying quadrature:

static void gtilde1_out( const double* x, const double* nB, double length_A, double* out)
{
  double A0[2], A1[2], B0[2], B1[2];

  A0[0] = x[0];
  A0[1] = x[1];
  A1[0] = x[2];
  A1[1] = x[3];
  B0[0] = x[4];
  B0[1] = x[5];
  B1[0] = x[6];
  B1[1] = x[7];

  // double length_A = std::sqrt((x[2]-x[0])*(x[2]-x[0]) + (x[3]-x[1])*(x[3]-x[1]));
  double projs[2] = {0};
  get_projections(A0, A1, B1, B0, projs);
  std::array<double, 2> projections = {projs[0], projs[1]};


  auto bounds = ContactSmoothing::bounds_from_projections(projections);
  auto xi_bounds = ContactSmoothing::smooth_bounds(bounds);


  auto qp = ContactEvaluator::compute_quadrature(xi_bounds);

  const int N = static_cast<int>(qp.qp.size());

  Gparams gp;
  gp.N = N;
  gp.qp = qp.qp.data();
  gp.w = qp.w.data();
  gp.x2 =nullptr;


  double gt[2];
  double A_out[2];
  gtilde_kernel( x, &gp, gt, nB, length_A, A_out);
  *out = gt[0];
}

static void gtilde2_out( const double* x, const double* nB, double length_A, double* out)
{
  double A0[2], A1[2], B0[2], B1[2];

  A0[0] = x[0];
  A0[1] = x[1];
  A1[0] = x[2];
  A1[1] = x[3];
  B0[0] = x[4];
  B0[1] = x[5];
  B1[0] = x[6];
  B1[1] = x[7];

  double projs[2] = {0};
  get_projections(A0, A1, B1, B0, projs);
  std::array<double, 2> projections = {projs[0], projs[1]};


  auto bounds = ContactSmoothing::bounds_from_projections(projections);
  auto xi_bounds = ContactSmoothing::smooth_bounds(bounds);


  auto qp = ContactEvaluator::compute_quadrature(xi_bounds);

  const int N = static_cast<int>(qp.qp.size());
  Gparams gp;
  gp.N = N;
  gp.qp = qp.qp.data();
  gp.w = qp.w.data();
  gp.x2 =nullptr;

//   const Gparams* gp = static_cast<const Gparams*>(gp_void);
  double gt[2];
  double A_out[2];
  gtilde_kernel(x, &gp, gt, nB, length_A, A_out);
  *out = gt[1];
}


static void A1_out( const double* x, double* out )
{
  double A0[2] = { x[0], x[1] };
  double A1[2] = { x[2], x[3] };
  double B0[2] = { x[4], x[5] };
  double B1[2] = { x[6], x[7] };

  double projs[2] = { 0 };
  get_projections( A0, A1, B0, B1, projs );

  const double del = 0.1;
  double xi_min = std::min( projs[0], projs[1] );
  double xi_max = std::max( projs[0], projs[1] );
  if ( xi_max < -0.5 - del ) xi_max = -0.5 - del;
  if ( xi_min >  0.5 + del ) xi_min =  0.5 + del;
  if ( xi_min < -0.5 - del ) xi_min = -0.5 - del;
  if ( xi_max >  0.5 + del ) xi_max =  0.5 + del;

  double sb[2];
  double raw[2] = { xi_min, xi_max };
  for ( int i = 0; i < 2; ++i ) {
    double xi = raw[i] + 0.5;
    double xi_hat = 0.0;
    if ( -del <= xi && xi <= del ) {
      xi_hat = ( 1.0/(4.0*del) )*xi*xi + 0.5*xi + del/4.0;
    } else if ( (1.0-del) <= xi && xi <= (1.0+del) ) {
      double b = -1.0/(4.0*del);
      double c = 0.5 + 1.0/(2.0*del);
      double d = 1.0-del + (1.0/(4.0*del))*(1.0-del)*(1.0-del)
                 - 0.5*(1.0-del) - (1.0-del)/(2.0*del);
      xi_hat = b*xi*xi + c*xi + d;
    } else {
      xi_hat = xi;
    }
    sb[i] = xi_hat - 0.5;
  }

  const double J_ref = std::sqrt( (A1[0]-A0[0])*(A1[0]-A0[0]) + (A1[1]-A0[1])*(A1[1]-A0[1]) );
  double AI_1 = J_ref * ( 0.5*(sb[1] - sb[0]) - 0.5*(sb[1]*sb[1] - sb[0]*sb[0]) );
  *out = AI_1;
}

static void A2_out( const double* x, double* out )
{
  double A0[2] = { x[0], x[1] };
  double A1[2] = { x[2], x[3] };
  double B0[2] = { x[4], x[5] };
  double B1[2] = { x[6], x[7] };

  double projs[2] = { 0 };
  get_projections( A0, A1, B0, B1, projs );

  const double del = 0.1;
  double xi_min = std::min( projs[0], projs[1] );
  double xi_max = std::max( projs[0], projs[1] );
  if ( xi_max < -0.5 - del ) xi_max = -0.5 - del;
  if ( xi_min >  0.5 + del ) xi_min =  0.5 + del;
  if ( xi_min < -0.5 - del ) xi_min = -0.5 - del;
  if ( xi_max >  0.5 + del ) xi_max =  0.5 + del;

  double sb[2];
  double raw[2] = { xi_min, xi_max };
  for ( int i = 0; i < 2; ++i ) {
    double xi = raw[i] + 0.5;
    double xi_hat = 0.0;
    if ( -del <= xi && xi <= del ) {
      xi_hat = ( 1.0/(4.0*del) )*xi*xi + 0.5*xi + del/4.0;
    } else if ( (1.0-del) <= xi && xi <= (1.0+del) ) {
      double b = -1.0/(4.0*del);
      double c = 0.5 + 1.0/(2.0*del);
      double d = 1.0-del + (1.0/(4.0*del))*(1.0-del)*(1.0-del)
                 - 0.5*(1.0-del) - (1.0-del)/(2.0*del);
      xi_hat = b*xi*xi + c*xi + d;
    } else {
      xi_hat = xi;
    }
    sb[i] = xi_hat - 0.5;
  }

  const double J_ref = std::sqrt( (A1[0]-A0[0])*(A1[0]-A0[0]) + (A1[1]-A0[1])*(A1[1]-A0[1]) );
  double AI_2 = J_ref * ( 0.5*(sb[1] - sb[0]) + 0.5*(sb[1]*sb[1] - sb[0]*sb[0]) );
  *out = AI_2;
}






// static void A1_out( const double* x, const double* nB, double length_A, double* out)
// {
//   double A0[2], A1[2], B0[2], B1[2];

//   A0[0] = x[0];
//   A0[1] = x[1];
//   A1[0] = x[2];
//   A1[1] = x[3];
//   B0[0] = x[4];
//   B0[1] = x[5];
//   B1[0] = x[6];
//   B1[1] = x[7];

//   // double length_A = std::sqrt((x[2]-x[0])*(x[2]-x[0]) + (x[3]-x[1])*(x[3]-x[1]));
//   double projs[2] = {0};
//   get_projections(A0, A1, B1, B0, projs);
//   std::array<double, 2> projections = {projs[0], projs[1]};


//   auto bounds = ContactSmoothing::bounds_from_projections(projections);
//   auto xi_bounds = ContactSmoothing::smooth_bounds(bounds);


//   auto qp = ContactEvaluator::compute_quadrature(xi_bounds);

//   const int N = static_cast<int>(qp.qp.size());

//   Gparams gp;
//   gp.N = N;
//   gp.qp = qp.qp.data();
//   gp.w = qp.w.data();
//   gp.x2 =nullptr;

//   double gt[2];
//   double A_out[2];
//   gtilde_kernel(x, &gp, gt, nB, length_A, A_out);
//   *out = A_out[0];
// }

// static void A2_out(const double* x, const double* nB, double length_A, double* out)
// {
//     double A0[2], A1[2], B0[2], B1[2];

//     A0[0] = x[0];
//     A0[1] = x[1];
//     A1[0] = x[2];
//     A1[1] = x[3];
//     B0[0] = x[4];
//     B0[1] = x[5];
//     B1[0] = x[6];
//     B1[1] = x[7];

//     // double length_A = std::sqrt((x[2]-x[0])*(x[2]-x[0]) + (x[3]-x[1])*(x[3]-x[1]));
//     double projs[2] = {0};
//     get_projections(A0, A1, B1, B0, projs);
//     std::array<double, 2> projections = {projs[0], projs[1]};


//     auto bounds = ContactSmoothing::bounds_from_projections(projections);
//     auto xi_bounds = ContactSmoothing::smooth_bounds(bounds);


//     auto qp = ContactEvaluator::compute_quadrature(xi_bounds);

//     const int N = static_cast<int>(qp.qp.size());

//     Gparams gp;
//     gp.N = N;
//     gp.qp = qp.qp.data();
//     gp.w = qp.w.data();
//     gp.x2 =nullptr;

//   double gt[2];
//   double A_out[2];
//   gtilde_kernel(x, &gp, gt, nB, length_A, A_out);
//   *out = A_out[1];
// }

void grad_gtilde1( const double* x, const double* nB, double length_A, double* dgt1_du)
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  __enzyme_autodiff<void>( (void*)gtilde1_out, enzyme_dup, x, dx, enzyme_const, nB, enzyme_const, length_A, enzyme_dup, &out, &dout);

  for ( int i = 0; i < 8; ++i ) {
    dgt1_du[i] = dx[i];
  }
}

void grad_gtilde2( const double* x, const double* nB, double length_A, double* dgt2_du)
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  __enzyme_autodiff<void>( (void*)gtilde2_out, enzyme_dup, x, dx, enzyme_const, nB, enzyme_const, length_A, enzyme_dup, &out, &dout);

  for ( int i = 0; i < 8; ++i ) {
    dgt2_du[i] = dx[i];
  }
}


void grad_A1( const double* x, const double* nB, double length_A, double* dA1_du )
{
  double dx[8] = { 0.0 };
  double out   = 0.0;
  double dout  = 1.0;

  __enzyme_autodiff<void>( (void*)A1_out, enzyme_dup, x, dx, enzyme_dup, &out, &dout );

  for ( int i = 0; i < 8; ++i ) {
    dA1_du[i] = dx[i];
  }
}

void grad_A2( const double* x, const double* nB, double length_A, double* dA2_du )
{
  double dx[8] = { 0.0 };
  double out   = 0.0;
  double dout  = 1.0;

  __enzyme_autodiff<void>( (void*)A2_out, enzyme_dup, x, dx, enzyme_dup, &out, &dout );

  for ( int i = 0; i < 8; ++i ) {
    dA2_du[i] = dx[i];
  }
}



// void grad_A1(const double* x, const double* nB, double length_A, double* dA1_du)
// {
//   double dx[8] = { 0.0 };
//   double out = 0.0;
//   double dout = 1.0;

//   __enzyme_autodiff<void>( (void*)A1_out, enzyme_dup, x, dx, enzyme_const, nB, enzyme_const, length_A, enzyme_dup, &out, &dout);

//   for ( int i = 0; i < 8; ++i ) {
//     dA1_du[i] = dx[i];
//   }
// }

// void grad_A2( const double* x, const double* nB, double length_A, double* dA2_du)
// {
//   double dx[8] = { 0.0 };
//   double out = 0.0;
//   double dout = 1.0;

//   __enzyme_autodiff<void>( (void*)A2_out, enzyme_dup, x, dx, enzyme_const, nB, enzyme_const, length_A, enzyme_dup, &out, &dout);

//   for ( int i = 0; i < 8; ++i ) {
//     dA2_du[i] = dx[i];
//   }
// }

void d2gtilde1( const double* x, const double* nB, double length_A, double* H1 )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;

    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    __enzyme_fwddiff<void>( (void*)grad_gtilde1, x, dx, enzyme_const, nB, enzyme_const, length_A, enzyme_dup, grad, dgrad );

    for ( int row = 0; row < 8; ++row ) {
      H1[row * 8 + col] = dgrad[row];
    }
  }
}

void d2gtilde2( const double* x, const double* nB, double length_A, double* H2 )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;

    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    __enzyme_fwddiff<void>( (void*)grad_gtilde2, x, dx, enzyme_const, nB, enzyme_const, length_A, enzyme_dup, grad, dgrad );

    for ( int row = 0; row < 8; ++row ) {
      H2[row * 8 + col] = dgrad[row];
    }
  }
}

void get_d2A1( const double* x, const double* nB, double length_A, double* H1 )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;

    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    __enzyme_fwddiff<void>( (void*)grad_A1, x, dx, enzyme_const, nB, enzyme_const, length_A, enzyme_dup, grad, dgrad );

    for ( int row = 0; row < 8; ++row ) {
      H1[row * 8 + col] = dgrad[row];
    }
  }
}

void get_d2A2( const double* x, const double* nB, double length_A, double* H1 )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;

    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    __enzyme_fwddiff<void>( (void*)grad_A2, x, dx, enzyme_const, nB, enzyme_const, length_A, enzyme_dup, grad, dgrad );

    for ( int row = 0; row < 8; ++row ) {
      H1[row * 8 + col] = dgrad[row];
    }
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

  // std::cout << "Projections: " << projs[0] << ", " << projs[1] << std::endl;
  return { projs[0], projs[1] };
}

std::array<double, 2> ContactSmoothing::bounds_from_projections( const std::array<double, 2>& proj )
{
  double xi_min = std::min( proj[0], proj[1] );
  double xi_max = std::max( proj[0], proj[1] );

  const double del = 0.1;

  //   if ( xi_max < -0.5) {
  //   xi_max = -0.5;
  // }
  // if ( xi_min > 0.5 ) {
  //   xi_min = 0.5;
  // }
  // if ( xi_min < -0.5 ) {
  //   xi_min = -0.5;
  // }
  // if ( xi_max > 0.5 ) {
  //   xi_max = 0.5;
  // }

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


std::array<double, 2> ContactSmoothing::smooth_bounds( const std::array<double, 2>& bounds ) 
{
  std::array<double, 2> smooth_bounds;
  const double del = 0.1;
  for ( int i = 0; i < 2; ++i ) {
    double xi = 0.0;
    double xi_hat = 0.0;
    xi = bounds[i] + 0.5;
    if (del == 0.0) {
      xi_hat = xi;
    }
    else{
    if ( 0.0 - del <= xi && xi <= del ) {
      xi_hat = ( 1.0 / ( 4 * del ) ) * ( xi * xi ) + 0.5 * xi + del / 4.0;
      // std::cout << "zone1" << std::endl;
    } else if ( ( 1.0 - del ) <= xi && xi <= 1.0 + del ) {
      // std::cout << "Zone 2: " << std::endl;
      double b = -1.0 / ( 4.0 * del );
      double c = 0.5 + 1.0 / ( 2.0 * del );
      double d = 1.0 - del + ( 1.0 / ( 4.0 * del ) ) * pow( 1.0 - del, 2 ) - 0.5 * ( 1.0 - del ) -
                 ( 1.0 - del ) / ( 2.0 * del );

      xi_hat = b * xi * xi + c * xi + d;
    } else if ( del <= xi && xi <= ( 1.0 - del ) ) {
      xi_hat = xi;
      // std::cout << "zone3" << std::endl;
    }
    }
    smooth_bounds[i] = xi_hat - 0.5;
      // std::cout << "Smooth Bounds: " << smooth_bounds[i] << std::endl;
  }

  return smooth_bounds;
}



// std::array<double, 2> ContactSmoothing::smooth_bounds( const std::array<double, 2>& bounds ) const
// {
//   std::array<double, 2> smooth_bounds;
//   const double del = p_.del;
//   for ( int i = 0; i < 2; ++i ) {
//     double xi = 0.0;
//     double xi_hat = 0.0;
//     xi = bounds[i] + 0.5;
//     if (del == 0.0) {
//       xi_hat = xi;
//     }
//     else{
//     if ( 0.0 <= xi && xi <= del ) {
//       xi_hat = (xi*xi*(2 * del -xi)) / (del * del);
//       // std::cout << "zone1" << std::endl;
//     } else if ( ( 1.0 - del ) <= xi && xi <= 1.0 ) {
//       // std::cout << "Zone 2: " << std::endl;
//       xi_hat = 1 - ((1 - xi)*(1 - xi) * (2 * del - ( 1 - xi )) / (del * del));
    
//     } else if ( del <= xi && xi <= ( 1.0 - del ) ) {
//       xi_hat = xi;
//       // std::cout << "zone3" << std::endl;
//     }
//     }
//     smooth_bounds[i] = xi_hat - 0.5;
//       // std::cout << "Smooth Bounds: " << smooth_bounds[i] << std::endl;
//   }

//   return smooth_bounds;
// }









QuadPoints ContactEvaluator::compute_quadrature( const std::array<double, 2>& xi_bounds )
{
  const int N = 3;
  QuadPoints out;
  // out.qp.resize( N );
  // out.w.resize( N );

  // std::vector<double> qpoints( N );
  // std::vector<double> weights( N );
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

  // Print quadrature points
  // std::cout << "Quad points: ";
  // for (int i = 0; i < N; ++i) {
  //     // std::cout << out.qp[i] << " ";
  // }
  // // std::cout << std::endl;

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

  // std::cout << "x1: " << x1[0] << ", " << x1[1] << std::endl;

  double x2[2] = { 0.0 };
  find_intersection( B0, B1, x1, nB, x2 );

  double dx = x1[0] - x2[0];
  double dy = x1[1] - x2[1];

  double gn = -( dx * nB[0] + dy * nB[1] );  // signed normal gap
  // std::cout << "gap: " << gn << std::endl;
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  double eta = ( dot < 0 ) ? dot : 0.0;

  // std::cout << "GAP: " << gn << "  eta = " << eta << " smooth gap = " << gn * eta << std::endl;

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
  // double J_ref = std::sqrt((std::pow((1.0 - 0.0), 2) + std::pow((-0.5 + 0.5), 2)));

  auto projs = projections( pair, mesh1, mesh2 );

  auto bounds = smoother_.bounds_from_projections( projs );
  auto smooth_bounds = smoother_.smooth_bounds( bounds );

  auto qp = compute_quadrature( smooth_bounds );
  auto qp_full = compute_quadrature( { -0.5, 0.5 } );  // for Ai

  double g_tilde1 = 0.0;
  double g_tilde2 = 0.0;
  double AI_1 = 0.0;
  double AI_2 = 0.0;

  // for (size_t i = 0; i < qp_full.qp.size(); ++i) {
  //     double xiA_full = qp_full.qp[i];
  //     double w_full = qp_full.w[i];
  //     double N1_full = 0.5 - xiA_full;
  //     double N2_full = 0.5 + xiA_full;

  //     AI_1 += w_full * N1_full * J_ref;
  //     AI_2 += w_full * N2_full * J_ref;
  // }

  for ( size_t i = 0; i < qp.qp.size(); ++i ) {
    double xiA = qp.qp[i];
    double w = qp.w[i];
    // double w_full = qp_full.w[i];
    // double xiA_full = qp_full.qp[i];

    // std::cout << "xiA: " << xiA << std::endl;

    double N1 = 0.5 - xiA;
    double N2 = 0.5 + xiA;

    // double N1_full = 0.5 - xiA_full;
    // double N2_full = 0.5 + xiA_full;

    double gn = gap( pair, mesh1, mesh2, xiA );
    // double gn_active = (gn < 0.0) ? gn : 0.0;
    double gn_active = gn;
    // std::cout << "gap: " << gn << std::endl;

    g_tilde1 += w * N1 * gn_active * J;
    g_tilde2 += w * N2 * gn_active * J;
    // double G = g_tilde1 + g_tilde2;
    // std::cout << "G: " << G << std::endl;

    // std::cout << "G~1: " << g_tilde1 << ", G~2:" << g_tilde2 << std::endl;

    AI_1 += w * N1 * J_ref;
    AI_2 += w * N2 * J_ref;
    // std::cout <<  AI_1 << ","<<  AI_2 << std::endl;
  }
  // std::cout <<  AI_1 << ","<<  AI_2 << std::endl;
  // std::cout << "A: " << AI_1 << ", " << AI_2 << std::endl;
  // std::cout <<  g_tilde1 << ","<<  g_tilde2 << std::endl;

  NodalContactData contact_data;

  contact_data.AI = { AI_1, AI_2 };
  contact_data.g_tilde = { g_tilde1, g_tilde2 };
  // double g1 = g_tilde1 / AI_1;
  // double g2 = g_tilde2 / AI_2;
  // // std::cout <<  g1 << ","<<  g2 << std::endl;

  // //KKT Conditons
  // double p1 = (g1 < 0.0) ? p_.k * g1 : 0.0;
  // double p2 = (g2 < 0.0) ? p_.k * g2 : 0.0;

  // NodalContactData contact_data;

  // contact_data.pressures = {p1, p2};
  // contact_data.g_tilde = {g_tilde1, g_tilde2};

  return contact_data;
}

std::array<double, 2> ContactEvaluator::compute_pressures( const NodalContactData& ncd ) const
{
  double gt1 = ncd.g_tilde[0];
  double gt2 = ncd.g_tilde[1];

  // std::cout << "gt: " << gt1 << ", " << gt2 << std::endl;

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
  // std::cout << "*************************pressures: " << pressures[0] << ", " << pressures[1] << std::endl;

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
  double length_A = std::sqrt((x[2]-x[0])*(x[2]-x[0]) + (x[3]-x[1])*(x[3]-x[1]));

  double nB[2], nA[2];
  find_normal( B0, B1, nB );
  find_normal( A0, A1, nA );

  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  // double eta = (dot < 0) ? dot:0.0;
  double dg1_du[8] = { 0.0 };
  double dg2_du[8] = { 0.0 };



  if(!p_.enzyme_quadrature) {
    auto projs = projections( pair, mesh1, mesh2 );

    auto bounds = smoother_.bounds_from_projections( projs );
    auto smooth_bounds = smoother_.smooth_bounds( bounds );

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
    // gp.nB[0] = nB[0];
    // gp.nB[1] = nB[1];
    // gp.eta = eta;
    // gp.del = p_.del;

    grad_gtilde1_quad(x, &gp, dg1_du);
    grad_gtilde2_quad(x, &gp, dg2_du);

  }
  else{
    grad_gtilde1( x, nB, length_A, dg1_du);
    grad_gtilde2( x, nB, length_A, dg2_du);
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
  double length_A = std::sqrt((x[2]-x[0])*(x[2]-x[0]) + (x[3]-x[1])*(x[3]-x[1]));

  double nB[2], nA[2];
  find_normal( B0, B1, nB );
  find_normal( A0, A1, nA );

  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  // double eta = (dot < 0) ? dot:0.0;

  if(!p_.enzyme_quadrature) {
    auto projs = projections( pair, mesh1, mesh2 );
    auto bounds = smoother_.bounds_from_projections( projs );
    auto smooth_bounds = smoother_.smooth_bounds( bounds );

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

    grad_A1_quad(x, &gp, dA1_dx);
    grad_A2_quad(x, &gp, dA2_dx);
  }
  else{  
    // grad_A1( x, nB, length_A, dA1_dx);
    // grad_A2( x, nB, length_A, dA2_dx);
    grad_A1( x, nB, length_A, dA1_dx);
    grad_A2( x, nB, length_A, dA2_dx);
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
  // std::cout << "A: " << ncd.AI[0] << ", " << ncd.AI[1] << std::endl;
  // std::cout << "g: " << ncd.g_tilde[0] << ", " << ncd.g_tilde[1] << std::endl;

  std::array<double, 2> pressures;
  pressures = compute_pressures( ncd );
  // std::cout << "Pressures: " << pressures[0] << ", " << pressures[1] << std::endl;

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
  double length_A = std::sqrt((x[2]-x[0])*(x[2]-x[0]) + (x[3]-x[1])*(x[3]-x[1]));

  double nB[2], nA[2];
  find_normal( B0, B1, nB );
  find_normal( A0, A1, nA );

  // double dot = nB[0] * nA[0] + nB[1] * nA[1];
  // double eta = (dot < 0) ? dot:0.0;

  auto projs = projections( pair, mesh1, mesh2 );
  auto bounds = smoother_.bounds_from_projections( projs );
  // auto smooth_bounds = smoother_.smooth_bounds( bounds );

  // auto qp = compute_quadrature( smooth_bounds );

  // const int N = static_cast<int>( qp.qp.size() );
  // std::vector<double> x2( 2 * N );

  // for ( int i = 0; i < N; ++i ) {
  //   double x1[2] = { 0.0 };
  //   iso_map( A0, A1, qp.qp[i], x1 );
  //   double x2_i[2] = { 0.0 };
  //   find_intersection( B0, B1, x1, nB, x2_i );
  //   x2[2 * i] = x2_i[0];
  //   x2[2 * i + 1] = x2_i[1];
  // }

  // Gparams gp;
  // gp.N = N;
  // gp.qp = qp.qp.data();
  // gp.w = qp.w.data();
  // gp.x2 = x2.data();

  double d2g1_d2u[64] = { 0.0 };
  double d2g2_d2u[64] = { 0.0 };

  d2gtilde1( x, nB, length_A, d2g1_d2u);
  d2gtilde2( x, nB, length_A, d2g2_d2u);

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
  double length_A = std::sqrt((x[2]-x[0])*(x[2]-x[0]) + (x[3]-x[1])*(x[3]-x[1]));

  double nB[2], nA[2];

  find_normal( B0, B1, nB );
  find_normal( A0, A1, nA );

  // double dot = nB[0] * nA[0] + nB[1] * nA[1];

  auto projs = projections( pair, mesh1, mesh2 );
  auto bounds = smoother_.bounds_from_projections( projs );
  // auto smooth_bounds = smoother_.smooth_bounds( bounds );

  // auto qp = compute_quadrature( smooth_bounds );

  // const int N = static_cast<int>( qp.qp.size() );
  // std::vector<double> x2( 2 * N );

  // for ( int i = 0; i < N; ++i ) {
  //   double x1[2] = { 0.0 };
  //   iso_map( A0, A1, qp.qp[i], x1 );
  //   double x2_i[2] = { 0.0 };
  //   find_intersection( B0, B1, x1, nB, x2_i );
  //   x2[2 * i] = x2_i[0];
  //   x2[2 * i + 1] = x2_i[1];
  // }

  // Gparams gp;
  // gp.N = N;
  // gp.qp = qp.qp.data();
  // gp.w = qp.w.data();
  // gp.x2 = x2.data();

  double d2A1_d2u[64] = { 0.0 };
  double d2A2_d2u[64] = { 0.0 };

  get_d2A1( x, nB, length_A, d2A1_d2u );
  get_d2A2( x, nB, length_A, d2A2_d2u );

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

std::pair<double, double> ContactEvaluator::eval_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                         const MeshData::Viewer& mesh2 ) const
{
  NodalContactData ncd = compute_nodal_contact_data( pair, mesh1, mesh2 );
  // double gt1 = ncd.g_tilde[0];
  // double gt2 = ncd.g_tilde[1];
  double A1 = ncd.g_tilde[0];
  double A2 = ncd.g_tilde[1];

  return { A1, A2 };
}



FiniteDiffResult ContactEvaluator::validate_g_tilde( const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
                                                     double epsilon ) const
{
  FiniteDiffResult result;

  auto viewer1 = mesh1.getView();
  auto viewer2 = mesh2.getView();

  auto projs0         = projections( pair, viewer1, viewer2 );
  auto bounds0        = smoother_.bounds_from_projections( projs0 );
  auto smooth_bounds0 = smoother_.smooth_bounds( bounds0 );
  QuadPoints qp0;
  if ( !p_.enzyme_quadrature ) {
    qp0 = compute_quadrature( smooth_bounds0 );
  }

  auto [g1_base, g2_base] = eval_gtilde( pair, viewer1, viewer2 );
  std::cout << "eval_gtilde:           " << g1_base << ", " << g2_base << "\n";
  if ( !p_.enzyme_quadrature ) {
    auto [g1_check, g2_check] = eval_gtilde_fixed_qp( pair, viewer1, viewer2, qp0 );
    std::cout << "eval_gtilde_fixed_qp:  " << g1_check << ", " << g2_check << "\n";
  }

  result.g_tilde1_baseline = g1_base;
  result.g_tilde2_baseline = g2_base;

  auto A_conn = viewer1.getConnectivity()( pair.m_element_id1 );
  auto B_conn = viewer2.getConnectivity()( pair.m_element_id2 );

  result.node_ids = { (int)A_conn[0], (int)A_conn[1], (int)B_conn[0], (int)B_conn[1] };

  const int num_dofs = 8;
  result.fd_gradient_g1.resize( num_dofs );
  result.fd_gradient_g2.resize( num_dofs );
  result.analytical_gradient_g1.resize( num_dofs );
  result.analytical_gradient_g2.resize( num_dofs );

  // ===== ANALYTICAL GRADIENTS =====
  double dgt1_dx[8] = { 0.0 };
  double dgt2_dx[8] = { 0.0 };
  grad_trib_area( pair, viewer1, viewer2, dgt1_dx, dgt2_dx );
  for ( int i = 0; i < 8; ++i ) {
    result.analytical_gradient_g1[i] = dgt1_dx[i];
    result.analytical_gradient_g2[i] = dgt2_dx[i];
  }

  // ===== SNAPSHOT ORIGINAL COORDS (long-lived so setPosition never dangles) =====
  int num_nodes1 = mesh1.numberOfNodes();
  std::vector<RealT> x1_orig( num_nodes1 ), y1_orig( num_nodes1 );
  {
    auto pos = mesh1.getView().getPosition();
    for ( int i = 0; i < num_nodes1; ++i ) {
      x1_orig[i] = pos[0][i];
      y1_orig[i] = pos[1][i];
    }
  }

  int num_nodes2 = mesh2.numberOfNodes();
  std::vector<RealT> x2_orig( num_nodes2 ), y2_orig( num_nodes2 );
  {
    auto pos = mesh2.getView().getPosition();
    for ( int i = 0; i < num_nodes2; ++i ) {
      x2_orig[i] = pos[0][i];
      y2_orig[i] = pos[1][i];
    }
  }

  // Helper lambda to evaluate using the correct path
  auto eval = [&]( const MeshData::Viewer& v1, const MeshData::Viewer& v2 ) -> std::pair<double, double> {
    return p_.enzyme_quadrature
               ? eval_gtilde( pair, v1, v2 )
               : eval_gtilde_fixed_qp( pair, v1, v2, qp0 );
  };

  // ===== FINITE DIFFERENCE GRADIENTS =====
  int dof_idx = 0;

  // A nodes → perturb mesh1
  for ( int k = 0; k < 2; ++k ) {
    int local_node = A_conn[k];

    // x perturbation
    {
      auto x_pert = x1_orig;
      x_pert[local_node] += epsilon;
      mesh1.setPosition( x_pert.data(), y1_orig.data(), nullptr );
      auto [g1p, g2p] = eval( mesh1.getView(), mesh2.getView() );

      x_pert[local_node] = x1_orig[local_node] - epsilon;
      mesh1.setPosition( x_pert.data(), y1_orig.data(), nullptr );
      auto [g1m, g2m] = eval( mesh1.getView(), mesh2.getView() );

      mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
      result.fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
      result.fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
      dof_idx++;
    }

    // y perturbation
    {
      auto y_pert = y1_orig;
      y_pert[local_node] += epsilon;
      mesh1.setPosition( x1_orig.data(), y_pert.data(), nullptr );
      auto [g1p, g2p] = eval( mesh1.getView(), mesh2.getView() );

      y_pert[local_node] = y1_orig[local_node] - epsilon;
      mesh1.setPosition( x1_orig.data(), y_pert.data(), nullptr );
      auto [g1m, g2m] = eval( mesh1.getView(), mesh2.getView() );

      mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
      result.fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
      result.fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
      dof_idx++;
    }
  }

  // B nodes → perturb mesh2
  for ( int k = 0; k < 2; ++k ) {
    int local_node = B_conn[k];

    // x perturbation
    {
      auto x_pert = x2_orig;
      x_pert[local_node] += epsilon;
      mesh2.setPosition( x_pert.data(), y2_orig.data(), nullptr );
      auto [g1p, g2p] = eval( mesh1.getView(), mesh2.getView() );

      x_pert[local_node] = x2_orig[local_node] - epsilon;
      mesh2.setPosition( x_pert.data(), y2_orig.data(), nullptr );
      auto [g1m, g2m] = eval( mesh1.getView(), mesh2.getView() );

      mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );
      result.fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
      result.fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
      dof_idx++;
    }

    // y perturbation
    {
      auto y_pert = y2_orig;
      y_pert[local_node] += epsilon;
      mesh2.setPosition( x2_orig.data(), y_pert.data(), nullptr );
      auto [g1p, g2p] = eval( mesh1.getView(), mesh2.getView() );

      y_pert[local_node] = y2_orig[local_node] - epsilon;
      mesh2.setPosition( x2_orig.data(), y_pert.data(), nullptr );
      auto [g1m, g2m] = eval( mesh1.getView(), mesh2.getView() );

      mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );
      result.fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
      result.fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
      dof_idx++;
    }
  }

  return result;
}






//GOOD TEST BELOW********

// FiniteDiffResult ContactEvaluator::validate_g_tilde( const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
//                                                      double epsilon ) const
// {
//   FiniteDiffResult result;

//   auto viewer1 = mesh1.getView();
//   auto viewer2 = mesh2.getView();

//   auto projs0      = projections( pair, viewer1, viewer2 );
//   auto bounds0     = smoother_.bounds_from_projections( projs0 );
//   auto smooth_bounds0 = smoother_.smooth_bounds( bounds0 );
//   QuadPoints qp0;
//   if ( !p_.enzyme_quadrature ) {
//     qp0 = compute_quadrature( smooth_bounds0 );
//   }

//   auto [g1_base, g2_base]   = eval_gtilde( pair, viewer1, viewer2 );
//   auto [g1_check, g2_check] = eval_gtilde_fixed_qp( pair, viewer1, viewer2, qp0 );
//   std::cout << "eval_gtilde:           " << g1_base  << ", " << g2_base  << "\n";
//   std::cout << "eval_gtilde_fixed_qp:  " << g1_check << ", " << g2_check << "\n";

//   result.g_tilde1_baseline = g1_base;
//   result.g_tilde2_baseline = g2_base;

//   auto A_conn = viewer1.getConnectivity()( pair.m_element_id1 );
//   auto B_conn = viewer2.getConnectivity()( pair.m_element_id2 );

//   result.node_ids = { (int)A_conn[0], (int)A_conn[1], (int)B_conn[0], (int)B_conn[1] };

//   const int num_dofs = 8;
//   result.fd_gradient_g1.resize( num_dofs );
//   result.fd_gradient_g2.resize( num_dofs );
//   result.analytical_gradient_g1.resize( num_dofs );
//   result.analytical_gradient_g2.resize( num_dofs );

//   // ===== ANALYTICAL GRADIENTS =====
//   double dgt1_dx[8] = { 0.0 };
//   double dgt2_dx[8] = { 0.0 };
//   grad_trib_area( pair, viewer1, viewer2, dgt1_dx, dgt2_dx );
//   for ( int i = 0; i < 8; ++i ) {
//     result.analytical_gradient_g1[i] = dgt1_dx[i];
//     result.analytical_gradient_g2[i] = dgt2_dx[i];
//   }

//   // ===== SNAPSHOT ORIGINAL COORDS (long-lived so setPosition never dangles) =====
//   int num_nodes1 = mesh1.numberOfNodes();
//   int dim1       = mesh1.spatialDimension();
//   std::vector<RealT> x1_orig( num_nodes1 ), y1_orig( num_nodes1 );
//   {
//     auto pos = mesh1.getView().getPosition();
//     for ( int i = 0; i < num_nodes1; ++i ) {
//       x1_orig[i] = pos[0][i];
//       y1_orig[i] = pos[1][i];
//     }
//   }

//   int num_nodes2 = mesh2.numberOfNodes();
//   int dim2       = mesh2.spatialDimension();
//   std::vector<RealT> x2_orig( num_nodes2 ), y2_orig( num_nodes2 );
//   {
//     auto pos = mesh2.getView().getPosition();
//     for ( int i = 0; i < num_nodes2; ++i ) {
//       x2_orig[i] = pos[0][i];
//       y2_orig[i] = pos[1][i];
//     }
//   }

//   // ===== FINITE DIFFERENCE GRADIENTS =====
//   int dof_idx = 0;

//   // A nodes → perturb mesh1
//   for ( int k = 0; k < 2; ++k ) {
//     int local_node = A_conn[k];

//     // x perturbation
//     {
//       auto x_pert = x1_orig;
//       x_pert[local_node] += epsilon;
//       mesh1.setPosition( x_pert.data(), y1_orig.data(), nullptr );

//       // DEBUG
//     auto dbg = mesh1.getView().getPosition();
//     auto [g1p, g2p] = eval_gtilde_fixed_qp( pair, mesh1.getView(), mesh2.getView(), qp0 );

//       x_pert[local_node] = x1_orig[local_node] - epsilon;
//       mesh1.setPosition( x_pert.data(), y1_orig.data(), nullptr );
//       auto [g1m, g2m] = eval_gtilde_fixed_qp( pair, mesh1.getView(), mesh2.getView(), qp0 );

//       mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
//       result.fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
//       result.fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
//       dof_idx++;
//     }

//     // y perturbation
//     {
//       auto y_pert = y1_orig;
//       y_pert[local_node] += epsilon;
//       mesh1.setPosition( x1_orig.data(), y_pert.data(), nullptr );
//       auto [g1p, g2p] = eval_gtilde_fixed_qp( pair, mesh1.getView(), mesh2.getView(), qp0 );

//       y_pert[local_node] = y1_orig[local_node] - epsilon;
//       mesh1.setPosition( x1_orig.data(), y_pert.data(), nullptr );
//       auto [g1m, g2m] = eval_gtilde_fixed_qp( pair, mesh1.getView(), mesh2.getView(), qp0 );

//       mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
//       result.fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
//       result.fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
//       dof_idx++;
//     }
//   }

//   // B nodes → perturb mesh2
//   for ( int k = 0; k < 2; ++k ) {
//     int local_node = B_conn[k];

//     // x perturbation
//     {
//       auto x_pert = x2_orig;
//       x_pert[local_node] += epsilon;
//       mesh2.setPosition( x_pert.data(), y2_orig.data(), nullptr );
//       auto [g1p, g2p] = eval_gtilde_fixed_qp( pair, mesh1.getView(), mesh2.getView(), qp0 );

//       x_pert[local_node] = x2_orig[local_node] - epsilon;
//       mesh2.setPosition( x_pert.data(), y2_orig.data(), nullptr );
//       auto [g1m, g2m] = eval_gtilde_fixed_qp( pair, mesh1.getView(), mesh2.getView(), qp0 );

//       mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );
//       result.fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
//       result.fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
//       dof_idx++;
//     }

//     // y perturbation
//     {
//       auto y_pert = y2_orig;
//       y_pert[local_node] += epsilon;
//       mesh2.setPosition( x2_orig.data(), y_pert.data(), nullptr );
//       auto [g1p, g2p] = eval_gtilde_fixed_qp( pair, mesh1.getView(), mesh2.getView(), qp0 );

//       y_pert[local_node] = y2_orig[local_node] - epsilon;
//       mesh2.setPosition( x2_orig.data(), y_pert.data(), nullptr );
//       auto [g1m, g2m] = eval_gtilde_fixed_qp( pair, mesh1.getView(), mesh2.getView(), qp0 );

//       mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );
//       result.fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
//       result.fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
//       dof_idx++;
//     }
//   }

//   return result;
// }

// GOOOD TEST END ********


// FiniteDiffResult ContactEvaluator::validate_g_tilde( const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
//                                                      double epsilon ) const
// {
//   FiniteDiffResult result;

//   auto viewer1 = mesh1.getView();
//   auto viewer2 = mesh2.getView();

//   auto projs0 = projections( pair, viewer1, viewer2 );
//   auto bounds0 = smoother_.bounds_from_projections( projs0 );
//   auto smooth_bounds0 = smoother_.smooth_bounds( bounds0 );
//   QuadPoints qp0;
//   if ( !p_.enzyme_quadrature ) {
//     qp0 = compute_quadrature( smooth_bounds0 );
//   }

//   // auto [g1_base, g2_base] = eval_gtilde_fixed_qp(mesh, A, B, qp0);

//   auto [g1_base, g2_base] = eval_gtilde( pair, viewer1, viewer2 );
//   auto [g1_check, g2_check] = eval_gtilde_fixed_qp( pair, viewer1, viewer2, qp0 );
//   std::cout << "eval_gtilde:           " << g1_base  << ", " << g2_base  << "\n";
//   std::cout << "eval_gtilde_fixed_qp:  " << g1_check << ", " << g2_check << "\n";
//   result.g_tilde1_baseline = g1_base;
//   result.g_tilde2_baseline = g2_base;

//   // Collect nodes in sorted order
//   std::set<int> node_set;
//   auto A_conn = viewer1.getConnectivity()( pair.m_element_id1 );
//   node_set.insert( A_conn[0] );
//   node_set.insert( A_conn[1] );
//   auto B_conn = viewer2.getConnectivity()( pair.m_element_id2 );
//   node_set.insert( B_conn[0] );
//   node_set.insert( B_conn[1] );

//   result.node_ids = std::vector<int>( node_set.begin(), node_set.end() );
//   // std::sort(result.node_ids.begin(), result.node_ids.end()); //Redundant??

//   int num_dofs = result.node_ids.size() * 2;
//   result.fd_gradient_g1.resize( num_dofs );
//   result.fd_gradient_g2.resize( num_dofs );

//   // ===== GET AND REORDER ENZYME GRADIENTS =====
//   double dgt1_dx[8] = { 0.0 };
//   double dgt2_dx[8] = { 0.0 };
//   grad_trib_area( pair, viewer1, viewer2, dgt1_dx, dgt2_dx );

//   // Map from node_id to position in x[8]
//   std::map<int, int> node_to_x_idx;
//   node_to_x_idx[A_conn[0]] = 0;  // A0 → x[0,1]
//   node_to_x_idx[A_conn[1]] = 1;  // A1 → x[2,3]
//   node_to_x_idx[B_conn[0]] = 2;  // B0 → x[4,5]
//   node_to_x_idx[B_conn[1]] = 3;  // B1 → x[6,7]

//   // Reorder Enzyme gradients to match sorted node order
//   result.analytical_gradient_g1.resize( num_dofs );
//   result.analytical_gradient_g2.resize( num_dofs );

//   for ( size_t i = 0; i < result.node_ids.size(); ++i ) {
//     int node_id = result.node_ids[i];
//     int x_idx = node_to_x_idx[node_id];

//     result.analytical_gradient_g1[2 * i + 0] = dgt1_dx[2 * x_idx + 0];  // x component
//     result.analytical_gradient_g1[2 * i + 1] = dgt1_dx[2 * x_idx + 1];  // y component
//     result.analytical_gradient_g2[2 * i + 0] = dgt2_dx[2 * x_idx + 0];
//     result.analytical_gradient_g2[2 * i + 1] = dgt2_dx[2 * x_idx + 1];
//   }
//   // =

//   int dof_idx = 0;
//   // X-direction

//   std::set<IndexT> mesh1_nodes = { A_conn[0], A_conn[1] };
//   std::set<IndexT> mesh2_nodes = { B_conn[0], B_conn[1] };

//   for ( int node_id : result.node_ids ) {
//     {
//       bool is_in_mesh1 = ( mesh1_nodes.count( node_id ) > 0 );
//       MeshData& mesh_to_perturb = is_in_mesh1 ? mesh1 : mesh2;

//       // Store Original Mesh coords:
//       auto pos = mesh_to_perturb.getView().getPosition();
//       int num_nodes = mesh_to_perturb.numberOfNodes();
//       int dim = mesh_to_perturb.spatialDimension();

//       std::vector<RealT> x_original( num_nodes );
//       std::vector<RealT> y_original( num_nodes );
//       std::vector<RealT> z_original( num_nodes );

//       for ( int i = 0; i < num_nodes; ++i ) {
//         x_original[i] = pos[0][i];
//         y_original[i] = pos[1][i];
//         if ( dim == 3 ) z_original[i] = pos[2][i];
//       }

//       std::vector<RealT> x_pert = x_original;
//       x_pert[node_id] += epsilon;
//       mesh_to_perturb.setPosition( x_pert.data(), y_original.data(), dim == 3 ? z_original.data() : nullptr );

//       // Evalaute with x_plus
//       auto viewer1_plus = mesh1.getView();
//       auto viewer2_plus = mesh2.getView();

//       auto [g1_plus, g2_plus] = p_.enzyme_quadrature
//       ? eval_gtilde( pair, viewer1_plus, viewer2_plus )
//       : eval_gtilde_fixed_qp( pair, viewer1_plus, viewer2_plus, qp0 );

//       x_pert[node_id] = x_original[node_id] - epsilon;

//       mesh_to_perturb.setPosition( x_pert.data(), y_original.data(), dim == 3 ? z_original.data() : nullptr );

//       auto viewer1_minus = mesh1.getView();
//       auto viewer2_minus = mesh2.getView();

//       auto [g1_minus, g2_minus] = p_.enzyme_quadrature
//       ? eval_gtilde( pair, viewer1_minus, viewer2_minus )
//       : eval_gtilde_fixed_qp( pair, viewer1_minus, viewer2_minus, qp0 );

//       // Restore orginal
//       mesh_to_perturb.setPosition( x_original.data(), y_original.data(), dim == 3 ? z_original.data() : nullptr );

//       // Compute gradient
//       result.fd_gradient_g1[dof_idx] = ( g1_plus - g1_minus ) / ( 2.0 * epsilon );
//       result.fd_gradient_g2[dof_idx] = ( g2_plus - g2_minus ) / ( 2.0 * epsilon );

//       dof_idx++;
//     }
//     {
//       bool is_in_mesh1 = ( mesh1_nodes.count( node_id ) > 0 );
//       MeshData& mesh_to_perturb = is_in_mesh1 ? mesh1 : mesh2;

//       // Store Original Mesh coords:
//       auto pos = mesh_to_perturb.getView().getPosition();
//       int num_nodes = mesh_to_perturb.numberOfNodes();
//       int dim = mesh_to_perturb.spatialDimension();

//       std::vector<RealT> x_original( num_nodes );
//       std::vector<RealT> y_original( num_nodes );
//       std::vector<RealT> z_original( num_nodes );

//       for ( int i = 0; i < num_nodes; ++i ) {
//         x_original[i] = pos[0][i];
//         y_original[i] = pos[1][i];
//         if ( dim == 3 ) z_original[i] = pos[2][i];
//       }
//       std::vector<RealT> y_pert = y_original;

//       y_pert[node_id] += epsilon;

//       mesh_to_perturb.setPosition( x_original.data(), y_pert.data(), dim == 3 ? z_original.data() : nullptr );

//       auto viewer1_plus2 = mesh1.getView();
//       auto viewer2_plus2 = mesh2.getView();

//       // auto [g1_plus, g2_plus] = eval_gtilde_fixed_qp( pair, viewer1_plus2, viewer2_plus2, qp0 );

//       auto [g1_plus, g2_plus] = p_.enzyme_quadrature
//       ? eval_gtilde( pair, viewer1_plus2, viewer2_plus2 )
//       : eval_gtilde_fixed_qp( pair, viewer1_plus2, viewer2_plus2, qp0 );

//       y_pert[node_id] = y_original[node_id] - epsilon;

//       mesh_to_perturb.setPosition( x_original.data(), y_pert.data(), dim == 3 ? z_original.data() : nullptr );

//       auto viewer1_minus2 = mesh1.getView();
//       auto viewer2_minus2 = mesh2.getView();
//       // auto [g1_minus, g2_minus] = eval_gtilde_fixed_qp( pair, viewer1_minus2, viewer2_minus2, qp0 );

//       auto [g1_minus, g2_minus] = p_.enzyme_quadrature
//       ? eval_gtilde( pair, viewer1_minus2, viewer2_minus2 )
//       : eval_gtilde_fixed_qp( pair, viewer1_minus2, viewer2_minus2, qp0 );

//       mesh_to_perturb.setPosition( x_original.data(), y_original.data(), dim == 3 ? z_original.data() : nullptr );

//       result.fd_gradient_g1[dof_idx] = ( g1_plus - g1_minus ) / ( 2.0 * epsilon );
//       result.fd_gradient_g2[dof_idx] = ( g2_plus - g2_minus ) / ( 2.0 * epsilon );

//       dof_idx++;
//     }

//     //         double original = mesh.node(node_id).x;

//     //         double x_plus =

//     //         mesh.node(node_id).x = original + epsilon;
//     //         auto [g1_plus, g2_plus] = eval_gtilde_fixed_qp(mesh, A, B, qp0);

//     //         mesh.node(node_id).x = original - epsilon;
//     //         auto [g1_minus, g2_minus] = eval_gtilde_fixed_qp(mesh, A, B, qp0);

//     //         //Restorre orginal
//     //         mesh.node(node_id).x = original;

//     //         result.fd_gradient_g1[dof_idx] = (g1_plus - g1_minus) / (2.0 * epsilon);
//     //         result.fd_gradient_g2[dof_idx] = (g2_plus - g2_minus) / (2.0 * epsilon);

//     //         dof_idx++;
//     //     }

//     // //y - direction
//     //     {
//     //         double original = mesh.node(node_id).y;

//     //         // +epsilon
//     //         mesh.node(node_id).y = original + epsilon;
//     //         auto [g1_plus, g2_plus] = eval_gtilde_fixed_qp(mesh, A, B, qp0);

//     //         // -epsilon
//     //         mesh.node(node_id).y = original - epsilon;
//     //         auto [g1_minus, g2_minus] = eval_gtilde_fixed_qp(mesh, A, B, qp0);

//     //         // Restore
//     //         mesh.node(node_id).y = original;

//     //         // Central difference
//     //         result.fd_gradient_g1[dof_idx] = (g1_plus - g1_minus) / (2.0 * epsilon);
//     //         result.fd_gradient_g2[dof_idx] = (g2_plus - g2_minus) / (2.0 * epsilon);

//     //         dof_idx++;
//     //     }
//   }
//   return result;
// }


// void ContactEvaluator::grad_gtilde_with_qp( const InterfacePair& pair, const MeshData::Viewer& mesh1,
//                                             const MeshData::Viewer& mesh2, const QuadPoints& qp_fixed,
//                                             double dgt1_dx[8], double dgt2_dx[8] ) const
// {
//   double A0[2], A1[2], B0[2], B1[2];
//   endpoints( mesh1, pair.m_element_id1, A0, A1 );
//   endpoints( mesh2, pair.m_element_id2, B0, B1 );

//   double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

//   const int N = static_cast<int>( qp_fixed.qp.size() );

//   Gparams gp;
//   gp.N = N;
//   gp.qp = qp_fixed.qp.data();  // Use FIXED quadrature
//   gp.w = qp_fixed.w.data();

//   grad_A1( x, &gp, dgt1_dx );
//   grad_A2( x, &gp, dgt2_dx );
// }



std::pair<double, double> ContactEvaluator::eval_gtilde_fixed_qp( const InterfacePair& pair,
                                                                  const MeshData::Viewer& mesh1,
                                                                  const MeshData::Viewer& /*mesh2*/,
                                                                  const QuadPoints& qp_fixed ) const
{
  double A0[2], A1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );

  const double J = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  double gt1 = 0.0, gt2 = 0.0;

  for ( size_t i = 0; i < qp_fixed.qp.size(); ++i ) {
    const double xiA = qp_fixed.qp[i];
    const double w = qp_fixed.w[i];

    const double N1 = 0.5 - xiA;
    const double N2 = 0.5 + xiA;

    // const double gn = gap(pair, mesh1, mesh2, xiA);   // still depends on geometry
    // const double gn_active = gn;              // or your (gn<0?gn:0) logic

    gt1 += w * N1 * J;
    gt2 += w * N2 * J;
  }

  return { gt1, gt2 };
}





FiniteDiffResult ContactEvaluator::validate_hessian( const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
                                                      double epsilon ) const
{
  FiniteDiffResult result;

  auto viewer1 = mesh1.getView();
  auto viewer2 = mesh2.getView();

  double hess1[64] = { 0.0 };
  double hess2[64] = { 0.0 };
  compute_d2A_d2u( pair, viewer1, viewer2, hess1, hess2 );

  const int ndof = 8;
  result.fd_gradient_g1.assign( ndof * ndof, 0.0 );
  result.fd_gradient_g2.assign( ndof * ndof, 0.0 );
  result.analytical_gradient_g1.assign( hess1, hess1 + 64 );
  result.analytical_gradient_g2.assign( hess2, hess2 + 64 );

  auto A_conn = viewer1.getConnectivity()( pair.m_element_id1 );
  auto B_conn = viewer2.getConnectivity()( pair.m_element_id2 );

  result.node_ids = { (int)A_conn[0], (int)A_conn[1], (int)B_conn[0], (int)B_conn[1] };

  // Use varying-qp evaluator to match what compute_d2A_d2u differentiates through
  ContactParams params_varying = p_;
  params_varying.enzyme_quadrature = true;
  ContactEvaluator eval_varying( params_varying );

  // ===== SNAPSHOT ORIGINAL COORDS =====
  int num_nodes1 = mesh1.numberOfNodes();
  std::vector<RealT> x1_orig( num_nodes1 ), y1_orig( num_nodes1 );
  {
    auto pos = mesh1.getView().getPosition();
    for ( int i = 0; i < num_nodes1; ++i ) {
      x1_orig[i] = pos[0][i];
      y1_orig[i] = pos[1][i];
    }
  }

  int num_nodes2 = mesh2.numberOfNodes();
  std::vector<RealT> x2_orig( num_nodes2 ), y2_orig( num_nodes2 );
  {
    auto pos = mesh2.getView().getPosition();
    for ( int i = 0; i < num_nodes2; ++i ) {
      x2_orig[i] = pos[0][i];
      y2_orig[i] = pos[1][i];
    }
  }

  // ===== FINITE DIFFERENCE HESSIAN =====
  int col = 0;

  // A nodes → perturb mesh1
  for ( int k = 0; k < 2; ++k ) {
    int local_node = A_conn[k];

    // x perturbation
    {
      double g1p[8] = { 0.0 }, g1m[8] = { 0.0 };
      double g2p[8] = { 0.0 }, g2m[8] = { 0.0 };

      auto x_pert = x1_orig;
      x_pert[local_node] += epsilon;
      mesh1.setPosition( x_pert.data(), y1_orig.data(), nullptr );
      auto v1p = mesh1.getView(); auto v2p = mesh2.getView();
      eval_varying.grad_trib_area( pair, v1p, v2p, g1p, g2p );

      x_pert[local_node] = x1_orig[local_node] - epsilon;
      mesh1.setPosition( x_pert.data(), y1_orig.data(), nullptr );
      auto v1m = mesh1.getView(); auto v2m = mesh2.getView();
      eval_varying.grad_trib_area( pair, v1m, v2m, g1m, g2m );

      mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
      for ( int i = 0; i < 8; ++i ) {
        result.fd_gradient_g1[i * 8 + col] = ( g1p[i] - g1m[i] ) / ( 2.0 * epsilon );
        result.fd_gradient_g2[i * 8 + col] = ( g2p[i] - g2m[i] ) / ( 2.0 * epsilon );
      }
      ++col;
    }

    // y perturbation
    {
      double g1p[8] = { 0.0 }, g1m[8] = { 0.0 };
      double g2p[8] = { 0.0 }, g2m[8] = { 0.0 };

      auto y_pert = y1_orig;
      y_pert[local_node] += epsilon;
      mesh1.setPosition( x1_orig.data(), y_pert.data(), nullptr );
      auto v1p = mesh1.getView(); auto v2p = mesh2.getView();
      eval_varying.grad_trib_area( pair, v1p, v2p, g1p, g2p );

      y_pert[local_node] = y1_orig[local_node] - epsilon;
      mesh1.setPosition( x1_orig.data(), y_pert.data(), nullptr );
      auto v1m = mesh1.getView(); auto v2m = mesh2.getView();
      eval_varying.grad_trib_area( pair, v1m, v2m, g1m, g2m );

      mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
      for ( int i = 0; i < 8; ++i ) {
        result.fd_gradient_g1[i * 8 + col] = ( g1p[i] - g1m[i] ) / ( 2.0 * epsilon );
        result.fd_gradient_g2[i * 8 + col] = ( g2p[i] - g2m[i] ) / ( 2.0 * epsilon );
      }
      ++col;
    }
  }

  // B nodes → perturb mesh2
  for ( int k = 0; k < 2; ++k ) {
    int local_node = B_conn[k];

    // x perturbation
    {
      double g1p[8] = { 0.0 }, g1m[8] = { 0.0 };
      double g2p[8] = { 0.0 }, g2m[8] = { 0.0 };

      auto x_pert = x2_orig;
      x_pert[local_node] += epsilon;
      mesh2.setPosition( x_pert.data(), y2_orig.data(), nullptr );
      auto v1p = mesh1.getView(); auto v2p = mesh2.getView();
      eval_varying.grad_trib_area( pair, v1p, v2p, g1p, g2p );

      x_pert[local_node] = x2_orig[local_node] - epsilon;
      mesh2.setPosition( x_pert.data(), y2_orig.data(), nullptr );
      auto v1m = mesh1.getView(); auto v2m = mesh2.getView();
      eval_varying.grad_trib_area( pair, v1m, v2m, g1m, g2m );

      mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );
      for ( int i = 0; i < 8; ++i ) {
        result.fd_gradient_g1[i * 8 + col] = ( g1p[i] - g1m[i] ) / ( 2.0 * epsilon );
        result.fd_gradient_g2[i * 8 + col] = ( g2p[i] - g2m[i] ) / ( 2.0 * epsilon );
      }
      ++col;
    }

    // y perturbation
    {
      double g1p[8] = { 0.0 }, g1m[8] = { 0.0 };
      double g2p[8] = { 0.0 }, g2m[8] = { 0.0 };

      auto y_pert = y2_orig;
      y_pert[local_node] += epsilon;
      mesh2.setPosition( x2_orig.data(), y_pert.data(), nullptr );
      auto v1p = mesh1.getView(); auto v2p = mesh2.getView();
      eval_varying.grad_trib_area( pair, v1p, v2p, g1p, g2p );

      y_pert[local_node] = y2_orig[local_node] - epsilon;
      mesh2.setPosition( x2_orig.data(), y_pert.data(), nullptr );
      auto v1m = mesh1.getView(); auto v2m = mesh2.getView();
      eval_varying.grad_trib_area( pair, v1m, v2m, g1m, g2m );

      mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );
      for ( int i = 0; i < 8; ++i ) {
        result.fd_gradient_g1[i * 8 + col] = ( g1p[i] - g1m[i] ) / ( 2.0 * epsilon );
        result.fd_gradient_g2[i * 8 + col] = ( g2p[i] - g2m[i] ) / ( 2.0 * epsilon );
      }
      ++col;
    }
  }

  return result;
}







// FiniteDiffResult ContactEvaluator::validate_hessian(const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
//                                                      double epsilon ) const{
//     FiniteDiffResult result;

//     auto viewer1 = mesh1.getView();
//     auto viewer2 = mesh2.getView();

//     auto projs0 =  projections( pair, viewer1, viewer2 );
//     auto bounds0 = smoother_.bounds_from_projections(projs0);
//     auto smooth_bounds0 = smoother_.smooth_bounds(bounds0);
//     QuadPoints qp0 = compute_quadrature(smooth_bounds0);
//     double hess1[64] = {0.0};
//     double hess2[64] = {0.0};
//     compute_d2A_d2u(pair, viewer1, viewer2, hess1, hess2);

//     const int ndof = 8;
//     result.fd_gradient_g1.assign(ndof*ndof, 0.0);
//     result.fd_gradient_g2.assign(ndof*ndof, 0.0);
//     result.analytical_gradient_g1.resize(ndof * ndof);
//     result.analytical_gradient_g2.resize(ndof * ndof);

//     result.analytical_gradient_g1.assign(hess1, hess1 + 64);
//     result.analytical_gradient_g2.assign(hess2, hess2 + 64);

//       // Collect nodes in sorted order
//     std::set<int> node_set;
//     auto A_conn = viewer1.getConnectivity()( pair.m_element_id1 );
//     node_set.insert( A_conn[0] );
//     node_set.insert( A_conn[1] );
//     auto B_conn = viewer2.getConnectivity()( pair.m_element_id2 );
//     node_set.insert( B_conn[0] );
//     node_set.insert( B_conn[1] );

//     result.node_ids = std::vector<int>( node_set.begin(), node_set.end() );

//     // Map from node_id to position in x[8]
//     std::map<int, int> node_to_x_idx;
//     node_to_x_idx[A_conn[0]] = 0;  // A0 → x[0,1]
//     node_to_x_idx[A_conn[1]] = 1;  // A1 → x[2,3]
//     node_to_x_idx[B_conn[0]] = 2;  // B0 → x[4,5]
//     node_to_x_idx[B_conn[1]] = 3;  // B1 → x[6,7]

//     // nodes in x[8] order to match analytical Hessian layout
//     IndexT nodes[4] = { A_conn[0], A_conn[1], B_conn[0], B_conn[1] };

//     std::set<IndexT> mesh1_nodes = { A_conn[0], A_conn[1] };
//     std::set<IndexT> mesh2_nodes = { B_conn[0], B_conn[1] };

//     int col = 0;
//     for ( int k = 0; k < 4; ++k ) {
//         int node_id = nodes[k];
//         bool is_in_mesh1 = ( mesh1_nodes.count( node_id ) > 0 );
//         MeshData& mesh_to_perturb = is_in_mesh1 ? mesh1 : mesh2;

//         for ( int comp = 0; comp < 2; ++comp ) {
//             auto pos = mesh_to_perturb.getView().getPosition();
//             int num_nodes = mesh_to_perturb.numberOfNodes();
//             int dim = mesh_to_perturb.spatialDimension();

//             std::vector<RealT> x_original( num_nodes );
//             std::vector<RealT> y_original( num_nodes );
//             std::vector<RealT> z_original( num_nodes );
//             for ( int i = 0; i < num_nodes; ++i ) {
//                 x_original[i] = pos[0][i];
//                 y_original[i] = pos[1][i];
//                 if ( dim == 3 ) z_original[i] = pos[2][i];
//             }

//             double g1p[8] = { 0.0 }, g1m[8] = { 0.0 };
//             double g2p[8] = { 0.0 }, g2m[8] = { 0.0 };

//             if ( comp == 0 ) {
//                 std::vector<RealT> x_pert = x_original;

//                 x_pert[node_id] += epsilon;
//                 mesh_to_perturb.setPosition( x_pert.data(), y_original.data(), dim == 3 ? z_original.data() : nullptr );
//                 auto v1p = mesh1.getView(); auto v2p = mesh2.getView();
//                 grad_gtilde( pair, v1p, v2p, g1p, g2p );

//                 x_pert[node_id] = x_original[node_id] - epsilon;
//                 mesh_to_perturb.setPosition( x_pert.data(), y_original.data(), dim == 3 ? z_original.data() : nullptr );
//                 auto v1m = mesh1.getView(); auto v2m = mesh2.getView();
//                 grad_gtilde( pair, v1m, v2m, g1m, g2m );
//             } else {
//                 std::vector<RealT> y_pert = y_original;

//                 y_pert[node_id] += epsilon;
//                 mesh_to_perturb.setPosition( x_original.data(), y_pert.data(), dim == 3 ? z_original.data() : nullptr );
//                 auto v1p = mesh1.getView(); auto v2p = mesh2.getView();
//                 grad_gtilde( pair, v1p, v2p, g1p, g2p );

//                 y_pert[node_id] = y_original[node_id] - epsilon;
//                 mesh_to_perturb.setPosition( x_original.data(), y_pert.data(), dim == 3 ? z_original.data() : nullptr );
//                 auto v1m = mesh1.getView(); auto v2m = mesh2.getView();
//                 grad_gtilde( pair, v1m, v2m, g1m, g2m );
//             }

//             mesh_to_perturb.setPosition( x_original.data(), y_original.data(), dim == 3 ? z_original.data() : nullptr );

//             for ( int i = 0; i < 8; ++i ) {
//                 result.fd_gradient_g1[i * 8 + col] = ( g1p[i] - g1m[i] ) / ( 2.0 * epsilon );
//                 result.fd_gradient_g2[i * 8 + col] = ( g2p[i] - g2m[i] ) / ( 2.0 * epsilon );
//             }
//             ++col;
//         }
//     }
//     return result;
// }
    







// int nodes[4] = { A.node_ids[0], A.node_ids[1], B.node_ids[0], B.node_ids[1] };

// int col = 0;
// for (int k = 0; k < 4; ++k) {
//   for (int comp = 0; comp < 2; ++comp) { // 0=x, 1=y
//     Node& n = mesh.node(nodes[k]);
//     double& coord = (comp == 0) ? n.x : n.y;
//     double orig = coord;

//     double g1p[8]={0}, g1m[8]={0}, g2p[8]={0}, g2m[8]={0};

//     coord = orig + epsilon; grad_gtilde_with_qp(pair, viewer1, viewer2, qp0, g1p, g2p);
//     coord = orig - epsilon; grad_gtilde_with_qp(pair, viewer1, viewer2, qp0, g1m, g2m);
//     coord = orig;

//     for (int i = 0; i < 8; ++i) {
//       result.fd_gradient_g1[i*8 + col] = (g1p[i] - g1m[i]) / (2*epsilon);
//       result.fd_gradient_g2[i*8 + col] = (g2p[i] - g2m[i]) / (2*epsilon);
//     }
//     ++col;
//   }
// }
// return result;
// }


#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
