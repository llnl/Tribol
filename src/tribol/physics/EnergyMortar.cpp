#include "tribol/physics/EnergyMortar.hpp"

#include "axom/slic.hpp"
#include "tribol/common/Enzyme.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iterator>
#include <vector>

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

namespace {

// This MUST match what the ContactParams struct has in EnergyMortarAdapter
// These had to be saved locally in order for enzyme to work correctly
struct KernelParams {
  int N{ 3 };         // No. of quadrature points
  double del{ 0.1 };  // Smoothing parameter
  double k{ 1.0 };    // Penalty stiffness
};

// Return the line-element mapping Jacobian. Local edge coordinates span [-0.5, 0.5], so the Jacobian is the physical
// length of the segment from A0 to A1.
TRIBOL_ENZYME_INLINE double line_jacobian( const double* A0, const double* A1 )
{
  return std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );
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

// Gets the respective gauss-legendre nodes dependant on quadrature order
TRIBOL_ENZYME_INLINE void determine_legendre_nodes( int N, double* x )
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
    assert( false && "Unsupported quadrature order" );
  }
}

// Gets the respective gauss-legendre weights dependant on quadrature order
TRIBOL_ENZYME_INLINE void determine_legendre_weights( int N, double* W )
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

// Map a point from the 1D parent segment coordinate to physical coordinates.
// Parametric space: [-0.5, 0.5]
TRIBOL_ENZYME_INLINE void iso_map( const double* coord1, const double* coord2, double xi, double* mapped_coord )
{
  double N1 = 0.5 - xi;
  double N2 = 0.5 + xi;
  mapped_coord[0] = N1 * coord1[0] + N2 * coord2[0];
  mapped_coord[1] = N1 * coord1[1] + N2 * coord2[1];
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

// Projects the point p onto the infinite line defined by edge A, using nB as
// the projection direction.
//
// Edge A defines the target line:
//
//     x_A(xiA) = A0 + xiA * (A1 - A0)
//
// The projection line is the line passing through p in the direction nB:
// Point p is assumed to be on edge B
//
//     x_proj(beta) = p + beta * nB
//
// This function computes the point where the projection line intersects the
// infinite line containing edge A.
// The returned intersection is not restricted to the finite segment A0--A1.
// If xiA is outside [0, 1], the intersection lies on the infinite extension of
// edge A. If nB is degenerate, or if the projection direction is nearly parallel
// to edge A, the function falls back to returning p.
TRIBOL_ENZYME_INLINE void find_intersection( const double* A0, const double* A1, const double* p, const double* nB,
                                             double* intersection )
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

  // If the projection direction is nearly parallel to edge A, use p as a safe fallback.
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

// Project the verticies of edge B onto edge A and return their local coordinates on A.
// The variable projections is retuned with the coordinates in the parametric space where
// the projections of edge B intersect edge A
// If the projection lies outside of Edge A, the bounds (in the parametric space) are returned
TRIBOL_ENZYME_INLINE void get_projections( const double* A0, const double* A1, const double* B0, const double* B1,
                                           double* projections )
{
  double nB[2] = { 0.0, 0.0 };
  find_normal( B0, B1, nB );

  const double dxA = A1[0] - A0[0];
  const double dyA = A1[1] - A0[1];
  const double len2A = dxA * dxA + dyA * dyA;

  double q0[2] = { 0.0, 0.0 };
  find_intersection( A0, A1, B0, nB, q0 );
  // Convert the physical projection point on A to the local coordinate xi in [-0.5, 0.5].
  const double alphaA0 = ( ( q0[0] - A0[0] ) * dxA + ( q0[1] - A0[1] ) * dyA ) / len2A;
  const double xi0 = alphaA0 - 0.5;

  double q1[2] = { 0.0, 0.0 };
  find_intersection( A0, A1, B1, nB, q1 );
  const double alphaA1 = ( ( q1[0] - A0[0] ) * dxA + ( q1[1] - A0[1] ) * dyA ) / len2A;
  const double xi1 = alphaA1 - 0.5;

  double xi_min = std::min( xi0, xi1 );
  double xi_max = std::max( xi0, xi1 );

  projections[0] = xi_min;
  projections[1] = xi_max;
}

// Isolate each endpoint to avoid incorrect loop-local tape reuse in Enzyme reverse mode.
TRIBOL_ENZYME_INLINE double smooth_bound( double bound, double del )
{
  double xi = 0.0;
  double xi_hat = 0.0;

  // Shift from the local coordinate interval [-0.5, 0.5] to [0, 1].
  xi = bound + 0.5;
  if ( del == 0.0 ) {
    xi_hat = xi;
  } else {
    // Apply quadratic ramps near the endpoints and leave the interior unchanged.
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
  // Shift the smoothed coordinate back to [-0.5, 0.5].
  return xi_hat - 0.5;
}

// Integrate the nodal smoothed gap and tributary area contributions over edge A.
// The quadrature rule is supplied through gp, allowing this kernel to be reused
// for both fixed-quadrature and geometry-dependent quadrature paths.
TRIBOL_ENZYME_INLINE void gtilde_kernel( const double* x, Gparams* gp, double* g_tilde_out, double* A_out )
{
  // x stores the two endpoints of edge A followed by the two endpoints of edge B.
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

  // Only keep the contribution when the edge normals oppose each other.
  // NOTE: geomFilter already rejects pairs with co-oriented normals (dot > 0),
  // but the clamp is retained for defensive correctness in tests and direct calls.
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  double eta = ( dot < 0 ) ? dot : 0.0;

  double g1 = 0.0, g2 = 0.0;
  double AI_1 = 0.0, AI_2 = 0.0;

  for ( int i = 0; i < 3; ++i ) {
    const double xiA = gp->qp[i];
    const double w = gp->w[i];

    const double N1 = 0.5 - xiA;
    const double N2 = 0.5 + xiA;

    // x1 on segment A
    double x1[2];
    iso_map( A0, A1, xiA, x1 );

    // Project the quadrature point on edge A (nonmortar) onto edge B (mortar) along B's normal.
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

// Integrate the nodal smoothed gap and tributary area contributions using fixed quadrature.
// The quadrature data in gp is treated as constant for Enzyme derivative calculations.
TRIBOL_ENZYME_INLINE void gtilde_kernel_quad( const double* x, const Gparams* gp, double* g_tilde_out, double* A_out )
{
  // x stores the two endpoints of edge A followed by the two endpoints of edge B.
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
  // Only keep the contribution when the edge normals oppose each other.
  // NOTE: geomFilter already rejects pairs with co-oriented normals (dot > 0),
  // but the clamp is retained for defensive correctness in tests and direct calls.
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  double eta = ( dot < 0 ) ? dot : 0.0;

  double g1 = 0.0, g2 = 0.0;
  double AI_1 = 0.0, AI_2 = 0.0;

  for ( int i = 0; i < 3; ++i ) {
    const double xiA = gp->qp[i];
    const double w = gp->w[i];

    const double N1 = 0.5 - xiA;
    const double N2 = 0.5 + xiA;

    // x1 on segment A
    double x1[2];
    iso_map( A0, A1, xiA, x1 );

    // Project the quadrature point on edge A (non mortar) onto edge B (mortar) along B's normal.
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

// Select which scalar quantity is extracted from the gap/area kernel for Enzyme differentiation.
enum class KernelOutput
{
  GTILDE1,
  GTILDE2,
  A1,
  A2
};

// Wrap the fixed-quadrature kernel as a scalar-valued function for Enzyme.
template <KernelOutput Output>
static void kernel_out( const double* x, const void* gp_void, double* out )
{
  const Gparams* gp = static_cast<const Gparams*>( gp_void );
  double gt[2];
  double A_out[2];
  gtilde_kernel_quad( x, gp, gt, A_out );

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
template <KernelOutput Output>
void grad_kernel( const double* x, const Gparams* gp, double* dout_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  // Seed the scalar output with 1.0 so Enzyme accumulates dOutput/dx into dx.
  __enzyme_autodiff<void>( (void*)kernel_out<Output>, enzyme_dup, x, dx, enzyme_const, (const void*)gp, enzyme_dup,
                           &out, &dout );

  for ( int i = 0; i < 8; ++i ) dout_du[i] = dx[i];
}

//**************************************** */
// Enzyme functions for varying quadrature:

// Wrap the varying-quadrature kernel as a scalar-valued function for Enzyme.
template <KernelOutput Output>
static void kernel_out_enzyme( const double* x, const void* kp_void, double* out )
{
  const KernelParams* kp = static_cast<const KernelParams*>( kp_void );
  // x stores the two endpoints of edge A followed by the two endpoints of edge B.
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

  // Recompute the integration bounds and quadrature from the current geometry.
  double bounds[2];
  ContactSmoothing::bounds_from_projections( projs, kp->del, bounds );
  double xi_bounds[2];
  ContactSmoothing::smooth_bounds( bounds, kp->del, xi_bounds );
  QuadPoints qp;
  EnergyMortarCalculator::compute_quadrature( xi_bounds, kp->N, &qp );

  Gparams gp;
  for ( std::size_t i = 0; i < qp.qp.size(); ++i ) {
    gp.qp[i] = qp.qp[i];
    gp.w[i] = qp.w[i];
  }

  double gt[2];
  double A_out[2];
  gtilde_kernel( x, &gp, gt, A_out );

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
template <KernelOutput Output>
void grad_kernel_enzyme( const double* x, const KernelParams* kp, double* dout_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  // Seed the scalar output with 1.0 so Enzyme accumulates dOutput/dx into dx.
  __enzyme_autodiff<void>( (void*)kernel_out_enzyme<Output>, enzyme_dup, x, dx, enzyme_const, (const void*)kp,
                           enzyme_dup, &out, &dout );

  for ( int i = 0; i < 8; ++i ) {
    dout_du[i] = dx[i];
  }
}

// Compute the Hessian of the selected varying-quadrature scalar kernel.
template <KernelOutput Output>
void d2_kernel( const double* x, const KernelParams* kp, double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;

    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    // Differentiate the gradient in coordinate direction col to form one Hessian column.
    __enzyme_fwddiff<void>( (void*)grad_kernel_enzyme<Output>, enzyme_dup, x, dx, enzyme_const, (const void*)kp,
                            enzyme_dup, grad, dgrad );

    for ( int row = 0; row < 8; ++row ) H[row * 8 + col] = dgrad[row];
  }
}

// Isolate loop-local arrays to avoid a leak in Enzyme's reverse-mode tape.
TRIBOL_ENZYME_INLINE double qp_penalty_kernel_qp_energy( double xiA, double w, const double* A0, const double* A1,
                                                         const double* B0, const double* B1, const double* nB,
                                                         double eta, double penalty, double J, bool* has_active_qp )
{
  double x1[2];
  iso_map( A0, A1, xiA, x1 );

  double x2[2];
  find_intersection( B0, B1, x1, nB, x2 );

  const double dx = x1[0] - x2[0];
  const double dy = x1[1] - x2[1];
  const double gn = -( dx * nB[0] + dy * nB[1] );
  const double gap = gn * eta;
  const bool is_active = gap <= 0.0;

  *has_active_qp = *has_active_qp || is_active;

  return is_active ? 0.5 * penalty * gap * gap * w * J : 0.0;
}

TRIBOL_ENZYME_INLINE void qp_penalty_kernel( const double* x, const KernelParams* kp, double* energy,
                                             bool* has_active_qp )
{
  *has_active_qp = false;

  double A0[2] = { x[0], x[1] };
  double A1[2] = { x[2], x[3] };
  double B0[2] = { x[4], x[5] };
  double B1[2] = { x[6], x[7] };

  double projs[2] = { 0.0, 0.0 };
  get_projections( A0, A1, B0, B1, projs );
  double bounds[2];
  ContactSmoothing::bounds_from_projections( projs, kp->del, bounds );
  double xi_bounds[2];
  ContactSmoothing::smooth_bounds( bounds, kp->del, xi_bounds );
  QuadPoints qp;
  EnergyMortarCalculator::compute_quadrature( xi_bounds, kp->N, &qp );

  double nB[2];
  find_normal( B0, B1, nB );
  double nA[2];
  find_normal( A0, A1, nA );
  // Only keep the contribution when the edge normals oppose each other.
  // NOTE: geomFilter already rejects pairs with co-oriented normals (dot > 0),
  // but the clamp is retained for defensive correctness in tests and direct calls.
  const double dot = nA[0] * nB[0] + nA[1] * nB[1];
  const double eta = ( dot < 0 ) ? dot : 0.0;
  const double J = line_jacobian( A0, A1 );

  double value = 0.0;
  for ( int i = 0; i < kp->N; ++i ) {
    value += qp_penalty_kernel_qp_energy( qp.qp[i], qp.w[i], A0, A1, B0, B1, nB, eta, kp->k, J, has_active_qp );
  }

  *energy = value;
}

void grad_qp_penalty_kernel( const double* x, const KernelParams* kp, double* dout_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;
  bool has_active_qp = false;
  __enzyme_autodiff<void>( (void*)qp_penalty_kernel, enzyme_dup, x, dx, enzyme_const, (const void*)kp, enzyme_dup, &out,
                           &dout, enzyme_const, &has_active_qp );

  for ( int i = 0; i < 8; ++i ) {
    dout_du[i] = dx[i];
  }
}

void d2_qp_penalty_kernel( const double* x, const KernelParams* kp, double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;
    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };
    __enzyme_fwddiff<void>( (void*)grad_qp_penalty_kernel, enzyme_dup, x, dx, enzyme_const, (const void*)kp, enzyme_dup,
                            grad, dgrad );
    for ( int row = 0; row < 8; ++row ) {
      H[row * 8 + col] = dgrad[row];
    }
  }
}

// Compute the Hessian of the selected fixed-quadrature scalar kernel.
template <KernelOutput Output>
void d2_kernel_quad( const double* x, const Gparams* gp, double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;
    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    // Differentiate the gradient in coordinate direction col to form one Hessian column
    __enzyme_fwddiff<void>( (void*)grad_kernel<Output>, enzyme_dup, x, dx, enzyme_const, (const void*)gp, enzyme_dup,
                            grad, dgrad );
    for ( int row = 0; row < 8; ++row ) H[row * 8 + col] = dgrad[row];
  }
}

}  // namespace

// Construct the quadrature data needed to evaluate the smoothed gap kernel.
Gparams EnergyMortarCalculator::construct_gparams( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                   const MeshData::Viewer& mesh2 ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );
  double nB[2] = { 0.0 };
  find_normal( B0, B1, nB );

  // Build the smoothed integration bounds from the projection of edge B onto edge A.
  auto projs = EnergyMortarCalculator::compute_projection_bounds( pair, mesh1, mesh2 );
  double bounds[2];
  smoother_.bounds_from_projections( projs.data(), p_.del, bounds );
  double smooth_bounds[2];
  smoother_.smooth_bounds( bounds, p_.del, smooth_bounds );

  QuadPoints qp;
  EnergyMortarCalculator::compute_quadrature( smooth_bounds, p_.N, &qp );

  const int N = static_cast<int>( qp.qp.size() );

  std::vector<double> x2( 2 * N );

  for ( int i = 0; i < N; ++i ) {
    double x1[2] = { 0.0 };
    iso_map( A0, A1, qp.qp[i], x1 );

    // Cache the projection of each quadrature point on edge A onto edge B.
    double x2_i[2] = { 0.0 };
    find_intersection( B0, B1, x1, nB, x2_i );
    x2[2 * i] = x2_i[0];
    x2[2 * i + 1] = x2_i[1];
  }

  Gparams gp;
  // int N = eval.get_N();

  for ( std::size_t i = 0; i < qp.qp.size(); ++i ) {
    gp.qp[i] = qp.qp[i];
    gp.w[i] = qp.w[i];
  }

  return gp;
}

// Return the local projection bounds of edge B onto edge A for this interface pair.
std::array<double, 2> EnergyMortarCalculator::projections( const InterfacePair& pair, const MeshData::Viewer& mesh1,
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

// Clamp the projection interval to the local smoothing support around edge A.
TRIBOL_ENZYME_INLINE void ContactSmoothing::bounds_from_projections( const double* projections, double del,
                                                                     double* bounds )
{
  double xi_min = std::min( projections[0], projections[1] );
  double xi_max = std::max( projections[0], projections[1] );

  // Limit the integration interval to the extended range [-0.5 - del, 0.5 + del].
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

  bounds[0] = xi_min;
  bounds[1] = xi_max;
}

// Smooth the integration bounds using a C1 ramp near the ends of edge A.
// Specific too the smoothing techniques in EnergyMortar. This smooths the
// Bounds of intergration by applying a quadratic ramping function near the ends of the paramteric
// space. The smooth region/length is defined by the input del. The returned 'bounds' is the new bounds
// of intergation that result after the quadratic ramping has been applied.
TRIBOL_ENZYME_INLINE void ContactSmoothing::smooth_bounds( const double* bounds, double del, double* smooth_bounds )
{
  smooth_bounds[0] = smooth_bound( bounds[0], del );
  smooth_bounds[1] = smooth_bound( bounds[1], del );
}

// Build a three-point Gauss-Legendre quadrature rule over the local integration bounds.
TRIBOL_ENZYME_INLINE void EnergyMortarCalculator::compute_quadrature( const double* xi_bounds, int N,
                                                                      QuadPoints* quadrature )
{
  double qpoints[3] = { 0.0, 0.0, 0.0 };
  double weights[3] = { 0.0, 0.0, 0.0 };

  determine_legendre_nodes( N, qpoints );
  determine_legendre_weights( N, weights );

  const double xi_min = xi_bounds[0];
  const double xi_max = xi_bounds[1];
  // Map the reference quadrature rule to [xi_min, xi_max].
  const double J = 0.5 * ( xi_max - xi_min );

  for ( int i = 0; i < N; ++i ) {
    quadrature->qp[i] = 0.5 * ( xi_max - xi_min ) * qpoints[i] + 0.5 * ( xi_max + xi_min );
    quadrature->w[i] = weights[i] * J;
  }
}

// Evaluate the weighted normal gap at local coordinate xiA on edge A.
double EnergyMortarCalculator::compute_weighted_normal_gap( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                            const MeshData::Viewer& mesh2, double xiA ) const
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

  // Project the point on edge A onto edge B along B's normal.
  double x2[2] = { 0.0 };
  find_intersection( B0, B1, x1, nB, x2 );

  double dx = x1[0] - x2[0];
  double dy = x1[1] - x2[1];

  double gn = -( dx * nB[0] + dy * nB[1] );  // signed normal gap
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  double eta = ( dot < 0 ) ? dot : 0.0;

  return gn * eta;
}

// Assemble nodal gap and tributary area data for the current interface pair.
NodalContactData EnergyMortarCalculator::compute_nodal_contact_data( const InterfacePair& pair,
                                                                     const MeshData::Viewer& mesh1,
                                                                     const MeshData::Viewer& mesh2 ) const
{
  double A0[2];
  double A1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );

  double J = std::sqrt( ( std::pow( ( A1[0] - A0[0] ), 2 ) + std::pow( ( A1[1] - A0[1] ), 2 ) ) );
  double J_ref = std::sqrt( std::pow( A1[0] - A0[0], 2 ) + std::pow( A1[1] - A0[1], 2 ) );

  auto projs = projections( pair, mesh1, mesh2 );

  // Build the smoothed integration interval from the projection bounds.
  double bounds[2];
  smoother_.bounds_from_projections( projs.data(), p_.del, bounds );
  double smooth_bounds[2];
  smoother_.smooth_bounds( bounds, p_.del, smooth_bounds );

  QuadPoints qp;
  compute_quadrature( smooth_bounds, p_.N, &qp );

  double g_tilde1 = 0.0;
  double g_tilde2 = 0.0;
  double AI_1 = 0.0;
  double AI_2 = 0.0;

  for ( size_t i = 0; i < qp.qp.size(); ++i ) {
    double xiA = qp.qp[i];
    double w = qp.w[i];
    double N1 = 0.5 - xiA;
    double N2 = 0.5 + xiA;

    // Evaluate the weighted gap at the current quadrature point on edge A.
    double gn = compute_weighted_normal_gap( pair, mesh1, mesh2, xiA );
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

// Return the nodal smoothed gaps and tributary areas for the interface pair.
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

// Compute derivatives of the two nodal smoothed gaps with respect to the endpoint coordinates.
void EnergyMortarCalculator::grad_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
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
    // Hold the quadrature rule fixed while differentiating the gap kernel.
    Gparams gp = construct_gparams( pair, mesh1, mesh2 );
    grad_kernel<KernelOutput::GTILDE1>( x, &gp, dg1_du );
    grad_kernel<KernelOutput::GTILDE2>( x, &gp, dg2_du );

  } else {
    // Differentiate through the geometry-dependent quadrature construction.
    const KernelParams kp{ p_.N, p_.del, p_.k };
    grad_kernel_enzyme<KernelOutput::GTILDE1>( x, &kp, dg1_du );
    grad_kernel_enzyme<KernelOutput::GTILDE2>( x, &kp, dg2_du );
  }

  for ( int i = 0; i < 8; ++i ) {
    dgt1_dx[i] = dg1_du[i];
    dgt2_dx[i] = dg2_du[i];
  }
}

// Compute derivatives of the two nodal tributary areas with respect to the endpoint coordinates
void EnergyMortarCalculator::grad_trib_area( const InterfacePair& pair, const MeshData::Viewer& mesh1,
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
    // Hold the quadrature rule fixed while differentiating the area kernel.
    Gparams gp = construct_gparams( pair, mesh1, mesh2 );
    grad_kernel<KernelOutput::A1>( x, &gp, dA1_dx );
    grad_kernel<KernelOutput::A2>( x, &gp, dA2_dx );
  } else {
    // Differentiate through the geometry-dependent quadrature construction.
    const KernelParams kp{ p_.N, p_.del, p_.k };
    grad_kernel_enzyme<KernelOutput::A1>( x, &kp, dA1_dx );
    grad_kernel_enzyme<KernelOutput::A2>( x, &kp, dA2_dx );
  }
}

// Compute the Hessians of the two nodal smoothed gaps with respect to the endpoint coordinates.
void EnergyMortarCalculator::d2_g2tilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
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
    // Hold the quadrature rule fixed while differentiating the gap gradients.
    Gparams gp = construct_gparams( pair, mesh1, mesh2 );
    d2_kernel_quad<KernelOutput::GTILDE1>( x, &gp, d2g1_d2u );
    d2_kernel_quad<KernelOutput::GTILDE2>( x, &gp, d2g2_d2u );

  } else {
    // Differentiate through the geometry-dependent quadrature construction.
    const KernelParams kp{ p_.N, p_.del, p_.k };
    d2_kernel<KernelOutput::GTILDE1>( x, &kp, d2g1_d2u );
    d2_kernel<KernelOutput::GTILDE2>( x, &kp, d2g2_d2u );
  }

  for ( int i = 0; i < 64; ++i ) {
    H1[i] = d2g1_d2u[i];
    H2[i] = d2g2_d2u[i];
  }
}

// Compute the Hessians of the two nodal tributary areas with respect to the endpoint coordinates.
void EnergyMortarCalculator::compute_d2A_d2u( const InterfacePair& pair, const MeshData::Viewer& mesh1,
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
    // Hold the quadrature rule fixed while differentiating the area gradients.
    Gparams gp = construct_gparams( pair, mesh1, mesh2 );
    d2_kernel_quad<KernelOutput::A1>( x, &gp, d2A1_d2u );
    d2_kernel_quad<KernelOutput::A2>( x, &gp, d2A2_d2u );
  } else {
    // Differentiate through the geometry-dependent quadrature construction.
    const KernelParams kp{ p_.N, p_.del, p_.k };
    d2_kernel<KernelOutput::A1>( x, &kp, d2A1_d2u );
    d2_kernel<KernelOutput::A2>( x, &kp, d2A2_d2u );
  }

  for ( int i = 0; i < 64; ++i ) {
    d2A1[i] = d2A1_d2u[i];
    d2A2[i] = d2A2_d2u[i];
  }
}

double EnergyMortarCalculator::compute_quadrature_point_penalty_energy( const InterfacePair& pair,
                                                                        const MeshData::Viewer& mesh1,
                                                                        const MeshData::Viewer& mesh2 ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  const double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };
  const KernelParams kp{ p_.N, p_.del, p_.k };
  double energy = 0.0;
  bool has_active_qp = false;
  qp_penalty_kernel( x, &kp, &energy, &has_active_qp );
  return energy;
}

QuadraturePointPenaltyData EnergyMortarCalculator::compute_quadrature_point_penalty_data(
    const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) const
{
  double A0[2], A1[2], B0[2], B1[2];

  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  const double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };
  const KernelParams kp{ p_.N, p_.del, p_.k };

  QuadraturePointPenaltyData result;
  qp_penalty_kernel( x, &kp, &result.energy, &result.has_active_qp );
  grad_qp_penalty_kernel( x, &kp, result.force.data() );
  d2_qp_penalty_kernel( x, &kp, result.stiffness.data() );
  return result;
}

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
