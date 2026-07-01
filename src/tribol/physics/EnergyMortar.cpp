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
// Theese had to be saved locally in order for enzyme to work correctly
struct KernelParams {
  int N = 3;                  // No. of quadrature points
  double del = 0.1;           // Smoothing parameter
  double residual_gap = 0.0;  // User-defined gap offset
};

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

TRIBOL_ENZYME_INLINE double normalize2( double* v )
{
  const double mag = std::sqrt( v[0] * v[0] + v[1] * v[1] );
  if ( mag > 1.0e-14 ) {
    v[0] /= mag;
    v[1] /= mag;
  }
  return mag;
}

TRIBOL_ENZYME_INLINE void interp_normal( const double* n0, const double* n1, double xi, double* n )
{
  const double N1 = 0.5 - xi;
  const double N2 = 0.5 + xi;
  n[0] = N1 * n0[0] + N2 * n1[0];
  n[1] = N1 * n0[1] + N2 * n1[1];
  normalize2( n );
}

TRIBOL_ENZYME_INLINE double nodal_energy_basis_value( EnergyMortarNodalEnergyBasis basis, int node, double xi )
{
  const double t = xi + 0.5;
  if ( basis == EnergyMortarNodalEnergyBasis::CUBIC_SPLINE ) {
    const double h = 3.0 * t * t - 2.0 * t * t * t;
    return node == 0 ? 1.0 - h : h;
  }
  return node == 0 ? 1.0 - t : t;
}

TRIBOL_ENZYME_INLINE void interp_normal_basis( const double* n0, const double* n1, double xi,
                                               EnergyMortarNodalEnergyBasis basis, double* n )
{
  const double N1 = nodal_energy_basis_value( basis, 0, xi );
  const double N2 = nodal_energy_basis_value( basis, 1, xi );
  n[0] = N1 * n0[0] + N2 * n1[0];
  n[1] = N1 * n0[1] + N2 * n1[1];
  normalize2( n );
}

TRIBOL_ENZYME_INLINE double local_coord_on_segment( const double* A0, const double* A1, const double* p )
{
  const double dx = A1[0] - A0[0];
  const double dy = A1[1] - A0[1];
  const double len2 = dx * dx + dy * dy;
  if ( len2 < 1.0e-28 ) {
    return 0.0;
  }
  return ( ( p[0] - A0[0] ) * dx + ( p[1] - A0[1] ) * dy ) / len2 - 0.5;
}

// Gets the respective gauss-legendre nodes dependant on quadrature order
TRIBOL_ENZYME_INLINE void determine_legendre_nodes( int N, std::array<double, 3>& x )
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

// Gets the respective gauss-legendre weights dependant on quadrature order
TRIBOL_ENZYME_INLINE void determine_legendre_weights( int N, std::array<double, 3>& W )
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

TRIBOL_ENZYME_INLINE void endpoint_normals( const MeshData::Viewer& mesh, int elem_id, double N0[2], double N1[2] )
{
  SLIC_ERROR_IF( !mesh.hasNodalNormals(), "ENERGY_MORTAR H1 normal mode requires nodal normals." );
  const auto conn = mesh.getConnectivity()( elem_id );
  N0[0] = mesh.getNodalNormals()( 0, conn[0] );
  N0[1] = mesh.getNodalNormals()( 1, conn[0] );
  N1[0] = mesh.getNodalNormals()( 0, conn[1] );
  N1[1] = mesh.getNodalNormals()( 1, conn[1] );
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

TRIBOL_ENZYME_INLINE bool project_to_edge_along_direction( const double* B0, const double* B1, const double* p,
                                                           const double* direction, double* intersection,
                                                           double* xiB )
{
  const double tB[2] = { B1[0] - B0[0], B1[1] - B0[1] };
  const double d[2] = { p[0] - B0[0], p[1] - B0[1] };

  const double dir_len = std::sqrt( direction[0] * direction[0] + direction[1] * direction[1] );
  if ( dir_len < 1.0e-14 ) {
    return false;
  }
  const double n[2] = { direction[0] / dir_len, direction[1] / dir_len };

  const double det = tB[0] * n[1] - tB[1] * n[0];
  if ( std::abs( det ) < 1.0e-10 ) {
    return false;
  }

  const double alpha = ( d[0] * n[1] - d[1] * n[0] ) / det;
  *xiB = alpha - 0.5;
  if ( *xiB < -0.5 || *xiB > 0.5 ) {
    return false;
  }

  intersection[0] = B0[0] + alpha * tB[0];
  intersection[1] = B0[1] + alpha * tB[1];
  return true;
}

TRIBOL_ENZYME_INLINE double nodal_energy_angle_weight( const double* nA, const double* nB, bool enabled )
{
  if ( !enabled ) {
    return 1.0;
  }

  double c = -( nA[0] * nB[0] + nA[1] * nB[1] );
  if ( c > 1.0 ) {
    c = 1.0;
  }
  if ( c < -1.0 ) {
    c = -1.0;
  }

  constexpr double pi = 3.14159265358979323846264338327950288;
  constexpr double theta0 = 80.0 * pi / 180.0;
  constexpr double theta1 = 0.5 * pi;
  constexpr double cos_theta1 = 0.0;
  const double cos_theta0 = std::cos( theta0 );

  if ( c >= cos_theta0 ) {
    return 1.0;
  }
  if ( c <= cos_theta1 ) {
    return 0.0;
  }

  const double theta = std::acos( c );
  const double t = ( theta - theta0 ) / ( theta1 - theta0 );
  const double smooth = 3.0 * t * t - 2.0 * t * t * t;
  return 1.0 - smooth;
}

TRIBOL_ENZYME_INLINE double active_set_smoothing_weight( double gap, double transition_gap )
{
  if ( transition_gap <= 0.0 || gap <= -transition_gap ) {
    return gap < 0.0 ? 1.0 : 0.0;
  }
  if ( gap >= 0.0 ) {
    return 0.0;
  }

  const double t = ( gap + transition_gap ) / transition_gap;
  const double smooth = 3.0 * t * t - 2.0 * t * t * t;
  return 1.0 - smooth;
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

TRIBOL_ENZYME_INLINE void get_projections_h1( const double* A0, const double* A1, const double* B0, const double* B1,
                                              const double* nB0, const double* nB1, double* projections )
{
  double nB0_unit[2] = { nB0[0], nB0[1] };
  normalize2( nB0_unit );
  double q0[2] = { 0.0, 0.0 };
  find_intersection( A0, A1, B0, nB0_unit, q0 );
  const double xi0 = local_coord_on_segment( A0, A1, q0 );

  double nB1_unit[2] = { nB1[0], nB1[1] };
  normalize2( nB1_unit );
  double q1[2] = { 0.0, 0.0 };
  find_intersection( A0, A1, B1, nB1_unit, q1 );
  const double xi1 = local_coord_on_segment( A0, A1, q1 );

  projections[0] = ( xi0 < xi1 ) ? xi0 : xi1;
  projections[1] = ( xi0 > xi1 ) ? xi0 : xi1;
}

TRIBOL_ENZYME_INLINE bool get_projections_along_direction( const double* A0, const double* A1, const double* B0,
                                                           const double* B1, const double* direction,
                                                           double* projections )
{
  const double dir_len = std::sqrt( direction[0] * direction[0] + direction[1] * direction[1] );
  if ( dir_len < 1.0e-14 ) {
    projections[0] = 0.0;
    projections[1] = 0.0;
    return false;
  }
  const double n[2] = { direction[0] / dir_len, direction[1] / dir_len };
  const double tA[2] = { A1[0] - A0[0], A1[1] - A0[1] };
  const double det = tA[0] * n[1] - tA[1] * n[0];
  if ( std::abs( det ) < 1.0e-10 ) {
    projections[0] = 0.0;
    projections[1] = 0.0;
    return false;
  }

  double q0[2] = { 0.0, 0.0 };
  find_intersection( A0, A1, B0, n, q0 );
  const double xi0 = local_coord_on_segment( A0, A1, q0 );

  double q1[2] = { 0.0, 0.0 };
  find_intersection( A0, A1, B1, n, q1 );
  const double xi1 = local_coord_on_segment( A0, A1, q1 );

  projections[0] = std::min( xi0, xi1 );
  projections[1] = std::max( xi0, xi1 );
  return true;
}

TRIBOL_ENZYME_INLINE void project_to_edge_h1( const double* B0, const double* B1, const double* nB0,
                                              const double* nB1, const double* x1, double* x2, double* nB )
{
  double xiB = local_coord_on_segment( B0, B1, x1 );
  for ( int iter = 0; iter < 3; ++iter ) {
    interp_normal( nB0, nB1, xiB, nB );
    find_intersection( B0, B1, x1, nB, x2 );
    xiB = local_coord_on_segment( B0, B1, x2 );
  }
  interp_normal( nB0, nB1, xiB, nB );
}

TRIBOL_ENZYME_INLINE void bounds_from_projections_raw( const double* proj, double del, double* bounds )
{
  double xi_min = ( proj[0] < proj[1] ) ? proj[0] : proj[1];
  double xi_max = ( proj[0] > proj[1] ) ? proj[0] : proj[1];

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

TRIBOL_ENZYME_INLINE double smooth_bound_value_raw( double bound, double del )
{
  const double xi = bound + 0.5;
  double xi_hat = 0.0;

  if ( del == 0.0 ) {
    xi_hat = xi;
  } else if ( 0.0 - del <= xi && xi <= del ) {
    xi_hat = ( 1.0 / ( 4.0 * del ) ) * ( xi * xi ) + 0.5 * xi + del / 4.0;
  } else if ( ( 1.0 - del ) <= xi && xi <= 1.0 + del ) {
    const double b = -1.0 / ( 4.0 * del );
    const double c = 0.5 + 1.0 / ( 2.0 * del );
    const double one_minus_del = 1.0 - del;
    const double d = 1.0 - del + ( 1.0 / ( 4.0 * del ) ) * one_minus_del * one_minus_del -
                     0.5 * one_minus_del - one_minus_del / ( 2.0 * del );
    xi_hat = b * xi * xi + c * xi + d;
  } else if ( del <= xi && xi <= ( 1.0 - del ) ) {
    xi_hat = xi;
  }

  return xi_hat - 0.5;
}

TRIBOL_ENZYME_INLINE void smooth_bounds_raw( const double* bounds, double del, double* smooth_bounds )
{
  smooth_bounds[0] = smooth_bound_value_raw( bounds[0], del );
  smooth_bounds[1] = smooth_bound_value_raw( bounds[1], del );
}

TRIBOL_ENZYME_INLINE void recover_h1_normal_at_node( const double* x, int num_nodes, int num_elems,
                                                     const int elem_nodes[h1_max_stencil_elems_per_mesh][2],
                                                     const double* xref, int query_node, double* n )
{
  double n_x = 0.0;
  double n_y = 0.0;
  double nref_x = 0.0;
  double nref_y = 0.0;

  for ( int e = 0; e < num_elems; ++e ) {
    const int node0 = elem_nodes[e][0];
    const int node1 = elem_nodes[e][1];
    if ( query_node != node0 && query_node != node1 ) {
      continue;
    }

    const double x0 = x[node0];
    const double y0 = x[num_nodes + node0];
    const double x1 = x[node1];
    const double y1 = x[num_nodes + node1];
    const double xr0 = xref[node0];
    const double yr0 = xref[num_nodes + node0];
    const double xr1 = xref[node1];
    const double yr1 = xref[num_nodes + node1];
    const double dx_ref = xr1 - xr0;
    const double dy_ref = yr1 - yr0;
    const double len_ref = std::sqrt( dx_ref * dx_ref + dy_ref * dy_ref );
    if ( len_ref < 1.0e-14 ) {
      continue;
    }

    const double nc_x = ( y1 - y0 ) / len_ref;
    const double nc_y = -( x1 - x0 ) / len_ref;
    n_x += nc_x;
    n_y += nc_y;
    nref_x += dy_ref / len_ref;
    nref_y += -dx_ref / len_ref;
  }

  const double mag_ref = std::sqrt( nref_x * nref_x + nref_y * nref_y );
  if ( mag_ref > 1.0e-14 ) {
    n_x /= mag_ref;
    n_y /= mag_ref;
  }
  n[0] = n_x;
  n[1] = n_y;
}

TRIBOL_ENZYME_INLINE void h1_kernel_eval( const double* x, const H1KernelData* data, double* g_tilde_out,
                                          double* A_out )
{
  const int n1 = data->num_nodes1;
  const int n2 = data->num_nodes2;
  const double* x1_all = x;
  const double* x2_all = x + 2 * n1;

  const int A_node0 = data->contact_nodes1[0];
  const int A_node1 = data->contact_nodes1[1];
  const int B_node0 = data->contact_nodes2[0];
  const int B_node1 = data->contact_nodes2[1];

  const double A0[2] = { x1_all[A_node0], x1_all[n1 + A_node0] };
  const double A1[2] = { x1_all[A_node1], x1_all[n1 + A_node1] };
  const double B0[2] = { x2_all[B_node0], x2_all[n2 + B_node0] };
  const double B1[2] = { x2_all[B_node1], x2_all[n2 + B_node1] };
  double nA0[2];
  double nA1[2];
  double nB0[2];
  double nB1[2];
  recover_h1_normal_at_node( x1_all, n1, data->num_elems1, data->elem_nodes1, data->xref1, A_node0, nA0 );
  recover_h1_normal_at_node( x1_all, n1, data->num_elems1, data->elem_nodes1, data->xref1, A_node1, nA1 );
  recover_h1_normal_at_node( x2_all, n2, data->num_elems2, data->elem_nodes2, data->xref2, B_node0, nB0 );
  recover_h1_normal_at_node( x2_all, n2, data->num_elems2, data->elem_nodes2, data->xref2, B_node1, nB1 );

  double projs_raw[2];
  get_projections_h1( A0, A1, B0, B1, nB0, nB1, projs_raw );
  double bounds_raw[2];
  bounds_from_projections_raw( projs_raw, data->del, bounds_raw );
  double xi_bounds_raw[2];
  if ( data->projection_smoothing ) {
    smooth_bounds_raw( bounds_raw, data->del, xi_bounds_raw );
  } else {
    xi_bounds_raw[0] = bounds_raw[0];
    xi_bounds_raw[1] = bounds_raw[1];
  }
  const std::array<double, 2> xi_bounds{ xi_bounds_raw[0], xi_bounds_raw[1] };
  auto qp = EnergyMortarCalculator::compute_quadrature( xi_bounds, data->N );

  const double J = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  double g1 = 0.0;
  double g2 = 0.0;
  double Atrib1 = 0.0;
  double Atrib2 = 0.0;

  for ( int i = 0; i < data->N; ++i ) {
    const double xiA = qp.qp[i];
    const double w = qp.w[i];
    const double N1 = 0.5 - xiA;
    const double N2 = 0.5 + xiA;

    double xA[2];
    iso_map( A0, A1, xiA, xA );

    double nA[2], nB[2], xB[2];
    interp_normal( nA0, nA1, xiA, nA );
    project_to_edge_h1( B0, B1, nB0, nB1, xA, xB, nB );

    const double dx = xA[0] - xB[0];
    const double dy = xA[1] - xB[1];
    const double gn = -( dx * nB[0] + dy * nB[1] );
    const double dot = nB[0] * nA[0] + nB[1] * nA[1];
    const double eta = ( dot < 0.0 ) ? dot : 0.0;
    const double g = gn * eta - data->residual_gap;

    g1 += w * N1 * g * J;
    g2 += w * N2 * g * J;
    Atrib1 += w * N1 * J;
    Atrib2 += w * N2 * J;
  }

  g_tilde_out[0] = g1;
  g_tilde_out[1] = g2;
  A_out[0] = Atrib1;
  A_out[1] = Atrib2;
}

TRIBOL_ENZYME_INLINE void h1_qp_penalty_kernel_eval( const double* x, const H1KernelData* data, double* energy_out )
{
  const int n1 = data->num_nodes1;
  const int n2 = data->num_nodes2;
  const double* x1_all = x;
  const double* x2_all = x + 2 * n1;

  const int A_node0 = data->contact_nodes1[0];
  const int A_node1 = data->contact_nodes1[1];
  const int B_node0 = data->contact_nodes2[0];
  const int B_node1 = data->contact_nodes2[1];

  const double A0[2] = { x1_all[A_node0], x1_all[n1 + A_node0] };
  const double A1[2] = { x1_all[A_node1], x1_all[n1 + A_node1] };
  const double B0[2] = { x2_all[B_node0], x2_all[n2 + B_node0] };
  const double B1[2] = { x2_all[B_node1], x2_all[n2 + B_node1] };
  double nA0[2];
  double nA1[2];
  double nB0[2];
  double nB1[2];
  recover_h1_normal_at_node( x1_all, n1, data->num_elems1, data->elem_nodes1, data->xref1, A_node0, nA0 );
  recover_h1_normal_at_node( x1_all, n1, data->num_elems1, data->elem_nodes1, data->xref1, A_node1, nA1 );
  recover_h1_normal_at_node( x2_all, n2, data->num_elems2, data->elem_nodes2, data->xref2, B_node0, nB0 );
  recover_h1_normal_at_node( x2_all, n2, data->num_elems2, data->elem_nodes2, data->xref2, B_node1, nB1 );

  double projs_raw[2];
  get_projections_h1( A0, A1, B0, B1, nB0, nB1, projs_raw );
  double bounds_raw[2];
  bounds_from_projections_raw( projs_raw, data->del, bounds_raw );
  double xi_bounds_raw[2];
  if ( data->projection_smoothing ) {
    smooth_bounds_raw( bounds_raw, data->del, xi_bounds_raw );
  } else {
    xi_bounds_raw[0] = bounds_raw[0];
    xi_bounds_raw[1] = bounds_raw[1];
  }
  const std::array<double, 2> xi_bounds{ xi_bounds_raw[0], xi_bounds_raw[1] };
  auto qp = EnergyMortarCalculator::compute_quadrature( xi_bounds, data->N );

  const double J = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  double energy = 0.0;
  for ( int i = 0; i < data->N; ++i ) {
    double xA[2];
    iso_map( A0, A1, qp.qp[i], xA );

    double nA[2], nB[2], xB[2];
    interp_normal( nA0, nA1, qp.qp[i], nA );
    project_to_edge_h1( B0, B1, nB0, nB1, xA, xB, nB );

    const double dx = xA[0] - xB[0];
    const double dy = xA[1] - xB[1];
    const double gn = -( dx * nB[0] + dy * nB[1] );
    const double dot = nB[0] * nA[0] + nB[1] * nA[1];
    const double eta = ( dot < 0.0 ) ? dot : 0.0;
    const double gap = gn * eta - data->residual_gap;

    const double active_weight = active_set_smoothing_weight( gap, data->active_set_smoothing_gap );
    if ( active_weight > 0.0 ) {
      energy += data->k * qp.w[i] * J * active_weight * gap * gap;
    }
  }

  *energy_out = energy;
}

TRIBOL_ENZYME_INLINE void h1_nodal_energy_kernel_eval( const double* x, const H1KernelData* data, double* energy_out )
{
  const int n1 = data->num_nodes1;
  const int n2 = data->num_nodes2;
  const double* x1_all = x;
  const double* x2_all = x + 2 * n1;

  const int A_node0 = data->contact_nodes1[0];
  const int A_node1 = data->contact_nodes1[1];
  const int B_node0 = data->contact_nodes2[0];
  const int B_node1 = data->contact_nodes2[1];

  const double A0[2] = { x1_all[A_node0], x1_all[n1 + A_node0] };
  const double A1[2] = { x1_all[A_node1], x1_all[n1 + A_node1] };
  const double B0[2] = { x2_all[B_node0], x2_all[n2 + B_node0] };
  const double B1[2] = { x2_all[B_node1], x2_all[n2 + B_node1] };
  double nA_nodes[2][2];
  double nB0[2];
  double nB1[2];
  recover_h1_normal_at_node( x1_all, n1, data->num_elems1, data->elem_nodes1, data->xref1, A_node0, nA_nodes[0] );
  recover_h1_normal_at_node( x1_all, n1, data->num_elems1, data->elem_nodes1, data->xref1, A_node1, nA_nodes[1] );
  recover_h1_normal_at_node( x2_all, n2, data->num_elems2, data->elem_nodes2, data->xref2, B_node0, nB0 );
  recover_h1_normal_at_node( x2_all, n2, data->num_elems2, data->elem_nodes2, data->xref2, B_node1, nB1 );
  normalize2( nA_nodes[0] );
  normalize2( nA_nodes[1] );

  const double J = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  double energy = 0.0;
  for ( int node = 0; node < 2; ++node ) {
    double projs_raw[2];
    const bool valid_bounds = get_projections_along_direction( A0, A1, B0, B1, nA_nodes[node], projs_raw );
    if ( !valid_bounds ) {
      continue;
    }
    double bounds_raw[2];
    bounds_from_projections_raw( projs_raw, data->del, bounds_raw );
    double xi_bounds_raw[2];
    if ( data->projection_smoothing ) {
      smooth_bounds_raw( bounds_raw, data->del, xi_bounds_raw );
    } else {
      xi_bounds_raw[0] = bounds_raw[0];
      xi_bounds_raw[1] = bounds_raw[1];
    }
    const std::array<double, 2> xi_bounds{ xi_bounds_raw[0], xi_bounds_raw[1] };
    auto qp = EnergyMortarCalculator::compute_quadrature( xi_bounds, data->N );

    for ( int i = 0; i < data->N; ++i ) {
      const double xiA = qp.qp[i];
      const double phi = nodal_energy_basis_value( data->nodal_energy_basis, node, xiA );
      double xA[2];
      iso_map( A0, A1, xiA, xA );

      double xB[2] = { 0.0, 0.0 };
      double xiB = 0.0;
      const bool valid_projection = project_to_edge_along_direction( B0, B1, xA, nA_nodes[node], xB, &xiB );
      if ( !valid_projection ) {
        continue;
      }

      double nB[2];
      interp_normal_basis( nB0, nB1, xiB, data->nodal_energy_basis, nB );
      const double angle_weight =
          nodal_energy_angle_weight( nA_nodes[node], nB, data->nodal_energy_angle_smoothing );
      if ( angle_weight <= 0.0 ) {
        continue;
      }

      const double dx = xA[0] - xB[0];
      const double dy = xA[1] - xB[1];
      const double gap = -( dx * nA_nodes[node][0] + dy * nA_nodes[node][1] ) - data->residual_gap;
      const double active_weight = active_set_smoothing_weight( gap, data->active_set_smoothing_gap );
      if ( active_weight > 0.0 ) {
        energy += data->k * qp.w[i] * J * phi * angle_weight * active_weight * gap * gap;
      }
    }
  }

  *energy_out = energy;
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
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  // double eta = ( dot < 0 ) ? dot : 0.0; //Normal smoothing
  double eta = dot;

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
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  // double eta = ( dot < 0 ) ? dot : 0.0;
  double eta = dot;

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

struct QPPenaltyKernelData {
  int N{ 3 };
  double del{ 0.1 };
  double k{ 1.0 };
  double residual_gap{ 0.0 };
  double active_set_smoothing_gap{ 0.0 };
  bool projection_smoothing{ true };
  bool fixed_quadrature{ false };
  Gparams qp{};
};

TRIBOL_ENZYME_INLINE void qp_penalty_kernel_eval( const double* x, const QPPenaltyKernelData* data,
                                                  double* energy_out )
{
  const double A0[2] = { x[0], x[1] };
  const double A1[2] = { x[2], x[3] };
  const double B0[2] = { x[4], x[5] };
  const double B1[2] = { x[6], x[7] };

  QuadPoints qp;
  if ( data->fixed_quadrature ) {
    qp.qp = data->qp.qp;
    qp.w = data->qp.w;
  } else {
    double projs_raw[2];
    get_projections( A0, A1, B0, B1, projs_raw );
    double bounds_raw[2];
    bounds_from_projections_raw( projs_raw, data->del, bounds_raw );
    double xi_bounds_raw[2];
    if ( data->projection_smoothing ) {
      smooth_bounds_raw( bounds_raw, data->del, xi_bounds_raw );
    } else {
      xi_bounds_raw[0] = bounds_raw[0];
      xi_bounds_raw[1] = bounds_raw[1];
    }
    const std::array<double, 2> xi_bounds{ xi_bounds_raw[0], xi_bounds_raw[1] };
    qp = EnergyMortarCalculator::compute_quadrature( xi_bounds, data->N );
  }

  const double J = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  double nB[2];
  find_normal( B0, B1, nB );

  double nA[2];
  find_normal( A0, A1, nA );

  const double dot = nB[0] * nA[0] + nB[1] * nA[1];
  const double eta = ( dot < 0.0 ) ? dot : 0.0;

  double energy = 0.0;
  for ( int i = 0; i < data->N; ++i ) {
    double x1[2];
    iso_map( A0, A1, qp.qp[i], x1 );

    double x2[2];
    find_intersection( B0, B1, x1, nB, x2 );

    const double dx = x1[0] - x2[0];
    const double dy = x1[1] - x2[1];
    const double gn = -( dx * nB[0] + dy * nB[1] );
    const double gap = gn * eta - data->residual_gap;

    const double active_weight = active_set_smoothing_weight( gap, data->active_set_smoothing_gap );
    if ( active_weight > 0.0 ) {
      energy += data->k * qp.w[i] * J * active_weight * gap * gap;
    }
  }

  *energy_out = energy;
}

TRIBOL_ENZYME_INLINE void qp_penalty_kernel_out( const double* x, const void* data_void, double* out )
{
  const QPPenaltyKernelData* data = static_cast<const QPPenaltyKernelData*>( data_void );
  qp_penalty_kernel_eval( x, data, out );
}

TRIBOL_ENZYME_INLINE void grad_qp_penalty_kernel_void( const double* x, const void* data_void, double* dout_du )
{
  const QPPenaltyKernelData* data = static_cast<const QPPenaltyKernelData*>( data_void );
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  __enzyme_autodiff<void>( (void*)qp_penalty_kernel_out, enzyme_dup, x, dx, enzyme_const, (const void*)data, enzyme_dup,
                           &out, &dout );

  for ( int i = 0; i < 8; ++i ) {
    dout_du[i] = dx[i];
  }
}

void grad_qp_penalty_kernel( const double* x, const QPPenaltyKernelData* data, double* dout_du )
{
  grad_qp_penalty_kernel_void( x, static_cast<const void*>( data ), dout_du );
}

void d2_qp_penalty_kernel( const double* x, const QPPenaltyKernelData* data, double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;

    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    __enzyme_fwddiff<void>( (void*)grad_qp_penalty_kernel_void, enzyme_dup, x, dx, enzyme_const, (const void*)data,
                            enzyme_dup, grad, dgrad );

    for ( int row = 0; row < 8; ++row ) {
      H[row * 8 + col] = dgrad[row];
    }
  }
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
TRIBOL_ENZYME_INLINE void kernel_out( const double* x, const void* gp_void, double* out )
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
TRIBOL_ENZYME_INLINE void grad_kernel( const double* x, const Gparams* gp, double* dout_du )
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
TRIBOL_ENZYME_INLINE void kernel_out_enzyme( const double* x, double* out )
{
  KernelParams kp;
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
  double bounds_raw[2];
  bounds_from_projections_raw( projs, kp.del, bounds_raw );
  double xi_bounds_raw[2];
  smooth_bounds_raw( bounds_raw, kp.del, xi_bounds_raw );
  const std::array<double, 2> xi_bounds{ xi_bounds_raw[0], xi_bounds_raw[1] };

  auto qp = EnergyMortarCalculator::compute_quadrature( xi_bounds, kp.N );

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
TRIBOL_ENZYME_INLINE void grad_kernel_enzyme( const double* x, double* dout_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  // Seed the scalar output with 1.0 so Enzyme accumulates dOutput/dx into dx.
  __enzyme_autodiff<void>( (void*)kernel_out_enzyme<Output>, enzyme_dup, x, dx, enzyme_dup, &out, &dout );

  for ( int i = 0; i < 8; ++i ) {
    dout_du[i] = dx[i];
  }
}

// Compute the Hessian of the selected varying-quadrature scalar kernel.
template <KernelOutput Output>
void d2_kernel( const double* x, double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;

    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    // Differentiate the gradient in coordinate direction col to form one Hessian column.
    __enzyme_fwddiff<void>( (void*)grad_kernel_enzyme<Output>, enzyme_dup, x, dx, enzyme_dup, grad, dgrad );

    for ( int row = 0; row < 8; ++row ) H[row * 8 + col] = dgrad[row];
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

template <KernelOutput Output>
TRIBOL_ENZYME_INLINE void h1_kernel_out( const double* x, const void* data_void, double* out )
{
  const H1KernelData* data = static_cast<const H1KernelData*>( data_void );
  double gt[2];
  double A_out[2];
  h1_kernel_eval( x, data, gt, A_out );

  if constexpr ( Output == KernelOutput::GTILDE1 )
    *out = gt[0];
  else if constexpr ( Output == KernelOutput::GTILDE2 )
    *out = gt[1];
  else if constexpr ( Output == KernelOutput::A1 )
    *out = A_out[0];
  else if constexpr ( Output == KernelOutput::A2 )
    *out = A_out[1];
}

template <KernelOutput Output>
TRIBOL_ENZYME_INLINE void grad_h1_kernel( const double* x, const H1KernelData* data, double* dout_du )
{
  const int ndof = 2 * ( data->num_nodes1 + data->num_nodes2 );
  double dx[2 * 2 * h1_max_stencil_nodes_per_mesh] = { 0.0 };
  for ( int i = 0; i < ndof; ++i ) {
    double out = 0.0;
    double dout = 0.0;
    dx[i] = 1.0;
    __enzyme_fwddiff<void>( (void*)h1_kernel_out<Output>, enzyme_dup, x, dx, enzyme_const, (const void*)data,
                            enzyme_dup, &out, &dout );
    dout_du[i] = dout;
    dx[i] = 0.0;
  }
}

template <KernelOutput Output>
void d2_h1_kernel( const double* x, const H1KernelData* data, double* H )
{
  const int ndof = 2 * ( data->num_nodes1 + data->num_nodes2 );
  for ( int col = 0; col < ndof; ++col ) {
    double dx[2 * 2 * h1_max_stencil_nodes_per_mesh] = { 0.0 };
    dx[col] = 1.0;

    double grad[2 * 2 * h1_max_stencil_nodes_per_mesh] = { 0.0 };
    double dgrad[2 * 2 * h1_max_stencil_nodes_per_mesh] = { 0.0 };

    __enzyme_fwddiff<void>( (void*)grad_h1_kernel<Output>, enzyme_dup, x, dx, enzyme_const, data, enzyme_dup, grad,
                            dgrad );

    for ( int row = 0; row < ndof; ++row ) {
      H[row * ndof + col] = dgrad[row];
    }
  }
}

TRIBOL_ENZYME_INLINE void h1_qp_penalty_kernel_out( const double* x, const void* data_void, double* out )
{
  const H1KernelData* data = static_cast<const H1KernelData*>( data_void );
  h1_qp_penalty_kernel_eval( x, data, out );
}

TRIBOL_ENZYME_INLINE void grad_h1_qp_penalty_kernel_void( const double* x, const void* data_void, double* dout_du )
{
  const H1KernelData* data = static_cast<const H1KernelData*>( data_void );
  const int ndof = 2 * ( data->num_nodes1 + data->num_nodes2 );
  double dx[2 * 2 * h1_max_stencil_nodes_per_mesh] = { 0.0 };
  for ( int i = 0; i < ndof; ++i ) {
    double out = 0.0;
    double dout = 0.0;
    dx[i] = 1.0;
    __enzyme_fwddiff<void>( (void*)h1_qp_penalty_kernel_out, enzyme_dup, x, dx, enzyme_const, (const void*)data,
                            enzyme_dup, &out, &dout );
    dout_du[i] = dout;
    dx[i] = 0.0;
  }
}

void grad_h1_qp_penalty_kernel( const double* x, const H1KernelData* data, double* dout_du )
{
  grad_h1_qp_penalty_kernel_void( x, static_cast<const void*>( data ), dout_du );
}

void d2_h1_qp_penalty_kernel( const double* x, const H1KernelData* data, double* H )
{
  const int ndof = 2 * ( data->num_nodes1 + data->num_nodes2 );
  for ( int col = 0; col < ndof; ++col ) {
    double dx[2 * 2 * h1_max_stencil_nodes_per_mesh] = { 0.0 };
    dx[col] = 1.0;

    double grad[2 * 2 * h1_max_stencil_nodes_per_mesh] = { 0.0 };
    double dgrad[2 * 2 * h1_max_stencil_nodes_per_mesh] = { 0.0 };

    __enzyme_fwddiff<void>( (void*)grad_h1_qp_penalty_kernel_void, enzyme_dup, x, dx, enzyme_const,
                            (const void*)data, enzyme_dup, grad, dgrad );

    for ( int row = 0; row < ndof; ++row ) {
      H[row * ndof + col] = dgrad[row];
    }
  }
}

TRIBOL_ENZYME_INLINE void h1_nodal_energy_kernel_out( const double* x, const void* data_void, double* out )
{
  const H1KernelData* data = static_cast<const H1KernelData*>( data_void );
  h1_nodal_energy_kernel_eval( x, data, out );
}

TRIBOL_ENZYME_INLINE void grad_h1_nodal_energy_kernel_void( const double* x, const void* data_void, double* dout_du )
{
  const H1KernelData* data = static_cast<const H1KernelData*>( data_void );
  const int ndof = 2 * ( data->num_nodes1 + data->num_nodes2 );
  double dx[2 * 2 * h1_max_stencil_nodes_per_mesh] = { 0.0 };
  for ( int i = 0; i < ndof; ++i ) {
    double out = 0.0;
    double dout = 0.0;
    dx[i] = 1.0;
    __enzyme_fwddiff<void>( (void*)h1_nodal_energy_kernel_out, enzyme_dup, x, dx, enzyme_const, (const void*)data,
                            enzyme_dup, &out, &dout );
    dout_du[i] = dout;
    dx[i] = 0.0;
  }
}

void grad_h1_nodal_energy_kernel( const double* x, const H1KernelData* data, double* dout_du )
{
  grad_h1_nodal_energy_kernel_void( x, static_cast<const void*>( data ), dout_du );
}

void d2_h1_nodal_energy_kernel( const double* x, const H1KernelData* data, double* H )
{
  const int ndof = 2 * ( data->num_nodes1 + data->num_nodes2 );
  for ( int col = 0; col < ndof; ++col ) {
    double dx[2 * 2 * h1_max_stencil_nodes_per_mesh] = { 0.0 };
    dx[col] = 1.0;

    double grad[2 * 2 * h1_max_stencil_nodes_per_mesh] = { 0.0 };
    double dgrad[2 * 2 * h1_max_stencil_nodes_per_mesh] = { 0.0 };

    __enzyme_fwddiff<void>( (void*)grad_h1_nodal_energy_kernel_void, enzyme_dup, x, dx, enzyme_const,
                            (const void*)data, enzyme_dup, grad, dgrad );

    for ( int row = 0; row < ndof; ++row ) {
      H[row * ndof + col] = dgrad[row];
    }
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

  // Build the smoothed integration bounds from the projection of edge B onto edge A.
  auto projs = EnergyMortarCalculator::compute_projection_bounds( pair, mesh1, mesh2 );
  auto bounds = smoother_.bounds_from_projections( projs, p_.del );
  auto smooth_bounds = p_.projection_smoothing ? smoother_.smooth_bounds( bounds, p_.del ) : bounds;

  auto qp = EnergyMortarCalculator::compute_quadrature( smooth_bounds, p_.N );

  const int N = static_cast<int>( qp.qp.size() );

  std::vector<double> x2( 2 * N );

  for ( int i = 0; i < N; ++i ) {
    double x1[2] = { 0.0 };
    iso_map( A0, A1, qp.qp[i], x1 );

    if ( p_.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL ) {
      double nB0[2], nB1[2], nB[2], x2_i[2];
      endpoint_normals( mesh2, pair.m_element_id2, nB0, nB1 );
      project_to_edge_h1( B0, B1, nB0, nB1, x1, x2_i, nB );
      x2[2 * i] = x2_i[0];
      x2[2 * i + 1] = x2_i[1];
    } else {
      double nB[2] = { 0.0 };
      find_normal( B0, B1, nB );
      double x2_i[2] = { 0.0 };
      find_intersection( B0, B1, x1, nB, x2_i );
      x2[2 * i] = x2_i[0];
      x2[2 * i + 1] = x2_i[1];
    }
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
  if ( p_.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL ) {
    double nB0[2], nB1[2];
    endpoint_normals( mesh2, pair.m_element_id2, nB0, nB1 );
    get_projections_h1( A0, A1, B0, B1, nB0, nB1, projs );
  } else {
    get_projections( A0, A1, B0, B1, projs );
  }

  return { projs[0], projs[1] };
}

// Clamp the projection interval to the local smoothing support around edge A.
TRIBOL_ENZYME_INLINE std::array<double, 2> ContactSmoothing::bounds_from_projections(
    const std::array<double, 2>& proj, double del )
{
  double xi_min = std::min( proj[0], proj[1] );
  double xi_max = std::max( proj[0], proj[1] );

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

  return { xi_min, xi_max };
}

// Smooth the integration bounds using a C1 ramp near the ends of edge A.
// Specific too the smoothing techniques in EnergyMortar. This smooths the
// Bounds of intergration by applying a quadratic ramping function near the ends of the paramteric
// space. The smooth region/length is defined by the input del. The returned 'bounds' is the new bounds
// of intergation that result after the quadratic ramping has been applied.
TRIBOL_ENZYME_INLINE std::array<double, 2> ContactSmoothing::smooth_bounds( const std::array<double, 2>& bounds,
                                                                            double del )
{
  std::array<double, 2> smooth_bounds;
  for ( int i = 0; i < 2; ++i ) {
    double xi = 0.0;
    double xi_hat = 0.0;

    // Shift from the local coordinate interval [-0.5, 0.5] to [0, 1].
    xi = bounds[i] + 0.5;
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
    smooth_bounds[i] = xi_hat - 0.5;
  }

  return smooth_bounds;
}

// Build a three-point Gauss-Legendre quadrature rule over the local integration bounds.
TRIBOL_ENZYME_INLINE QuadPoints EnergyMortarCalculator::compute_quadrature( const std::array<double, 2>& xi_bounds,
                                                                            int N )
{
  QuadPoints out;

  std::array<double, 3> qpoints;
  std::array<double, 3> weights;

  determine_legendre_nodes( N, qpoints );
  determine_legendre_weights( N, weights );

  const double xi_min = xi_bounds[0];
  const double xi_max = xi_bounds[1];
  // Map the reference quadrature rule to [xi_min, xi_max].
  const double J = 0.5 * ( xi_max - xi_min );

  for ( int i = 0; i < N; ++i ) {
    out.qp[i] = 0.5 * ( xi_max - xi_min ) * qpoints[i] + 0.5 * ( xi_max + xi_min );
    out.w[i] = weights[i] * J;
  }

  return out;
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

  double x1[2] = { 0.0 };
  iso_map( A0, A1, xiA, x1 );

  double x2[2] = { 0.0 };
  if ( p_.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL ) {
    double nA0[2], nA1[2], nB0[2], nB1[2];
    endpoint_normals( mesh1, pair.m_element_id1, nA0, nA1 );
    endpoint_normals( mesh2, pair.m_element_id2, nB0, nB1 );
    interp_normal( nA0, nA1, xiA, nA );
    project_to_edge_h1( B0, B1, nB0, nB1, x1, x2, nB );
  } else {
    find_normal( A0, A1, nA );
    find_normal( B0, B1, nB );
    // Project the point on edge A onto edge B along B's normal.
    find_intersection( B0, B1, x1, nB, x2 );
  }

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
  auto bounds = smoother_.bounds_from_projections( projs, p_.del );
  auto smooth_bounds = p_.projection_smoothing ? smoother_.smooth_bounds( bounds, p_.del ) : bounds;

  auto qp = compute_quadrature( smooth_bounds, p_.N );

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
  contact_data.g_tilde = { g_tilde1 - p_.residual_gap * AI_1, g_tilde2 - p_.residual_gap * AI_2 };

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
  SLIC_ERROR_ROOT_IF( p_.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL,
                      "Use compute_h1_total_derivatives() for H1_NODAL_NORMAL derivatives." );
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
    grad_kernel_enzyme<KernelOutput::GTILDE1>( x, dg1_du );
    grad_kernel_enzyme<KernelOutput::GTILDE2>( x, dg2_du );
  }

  for ( int i = 0; i < 8; ++i ) {
    dgt1_dx[i] = dg1_du[i];
    dgt2_dx[i] = dg2_du[i];
  }
  if ( p_.residual_gap != 0.0 ) {
    double dA1_dx[8] = { 0.0 };
    double dA2_dx[8] = { 0.0 };
    grad_trib_area( pair, mesh1, mesh2, dA1_dx, dA2_dx );
    for ( int i = 0; i < 8; ++i ) {
      dgt1_dx[i] -= p_.residual_gap * dA1_dx[i];
      dgt2_dx[i] -= p_.residual_gap * dA2_dx[i];
    }
  }
}

// Compute derivatives of the two nodal tributary areas with respect to the endpoint coordinates
void EnergyMortarCalculator::grad_trib_area( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                             const MeshData::Viewer& mesh2, double dA1_dx[8], double dA2_dx[8] ) const
{
  SLIC_ERROR_ROOT_IF( p_.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL,
                      "Use compute_h1_total_derivatives() for H1_NODAL_NORMAL derivatives." );
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
    grad_kernel_enzyme<KernelOutput::A1>( x, dA1_dx );
    grad_kernel_enzyme<KernelOutput::A2>( x, dA2_dx );
  }
}

// Compute the Hessians of the two nodal smoothed gaps with respect to the endpoint coordinates.
void EnergyMortarCalculator::d2_g2tilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                         const MeshData::Viewer& mesh2, double H1[64], double H2[64] ) const
{
  SLIC_ERROR_ROOT_IF( p_.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL,
                      "Use compute_h1_total_derivatives() for H1_NODAL_NORMAL derivatives." );
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
    d2_kernel<KernelOutput::GTILDE1>( x, d2g1_d2u );
    d2_kernel<KernelOutput::GTILDE2>( x, d2g2_d2u );
  }

  for ( int i = 0; i < 64; ++i ) {
    H1[i] = d2g1_d2u[i];
    H2[i] = d2g2_d2u[i];
  }
  if ( p_.residual_gap != 0.0 ) {
    double d2A1[64] = { 0.0 };
    double d2A2[64] = { 0.0 };
    compute_d2A_d2u( pair, mesh1, mesh2, d2A1, d2A2 );
    for ( int i = 0; i < 64; ++i ) {
      H1[i] -= p_.residual_gap * d2A1[i];
      H2[i] -= p_.residual_gap * d2A2[i];
    }
  }
}

// Compute the Hessians of the two nodal tributary areas with respect to the endpoint coordinates.
void EnergyMortarCalculator::compute_d2A_d2u( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                              const MeshData::Viewer& mesh2, double d2A1[64], double d2A2[64] ) const
{
  SLIC_ERROR_ROOT_IF( p_.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL,
                      "Use compute_h1_total_derivatives() for H1_NODAL_NORMAL derivatives." );
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
    d2_kernel<KernelOutput::A1>( x, d2A1_d2u );
    d2_kernel<KernelOutput::A2>( x, d2A2_d2u );
  }

  for ( int i = 0; i < 64; ++i ) {
    d2A1[i] = d2A1_d2u[i];
    d2A2[i] = d2A2_d2u[i];
  }
}

QuadraturePointPenaltyData EnergyMortarCalculator::compute_quadrature_point_penalty_data(
    const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) const
{
  SLIC_ERROR_ROOT_IF( p_.penalty_mode == EnergyMortarPenaltyMode::NODAL_ENERGY &&
                          p_.normal_mode != EnergyMortarNormalMode::H1_NODAL_NORMAL,
                      "ENERGY_MORTAR NODAL_ENERGY penalty mode requires H1_NODAL_NORMAL." );

  if ( p_.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL ) {
    QuadraturePointPenaltyData result;
    H1KernelData data;
    data.N = p_.N;
    data.del = p_.del;
    data.k = p_.k;
    data.residual_gap = p_.residual_gap;
    data.active_set_smoothing_gap = p_.h1_active_set_smoothing_gap;
    data.projection_smoothing = p_.projection_smoothing;
    data.nodal_energy_basis = p_.nodal_energy_basis;
    data.nodal_energy_angle_smoothing = p_.nodal_energy_angle_smoothing;

    auto build_side = []( const MeshData::Viewer& mesh, int contact_elem, int& num_nodes, int& num_elems,
                          std::array<int, h1_max_stencil_nodes_per_mesh>& node_ids,
                          std::array<int, h1_max_stencil_nodes_per_mesh>& owner_elems, int contact_nodes[2],
                          int elem_nodes[h1_max_stencil_elems_per_mesh][2], double* xref ) {
      std::vector<int> nodes;
      std::vector<int> owners;

      auto add_node = [&]( int node_id ) {
        auto it = std::find( nodes.begin(), nodes.end(), node_id );
        if ( it != nodes.end() ) {
          return static_cast<int>( std::distance( nodes.begin(), it ) );
        }
        SLIC_ERROR_ROOT_IF( static_cast<int>( nodes.size() ) >= h1_max_stencil_nodes_per_mesh,
                            "ENERGY_MORTAR H1 normal stencil exceeded supported node count." );
        nodes.push_back( node_id );
        owners.push_back( -1 );
        return static_cast<int>( nodes.size() - 1 );
      };

      const int contact_node0 = mesh.getGlobalNodeId( contact_elem, 0 );
      const int contact_node1 = mesh.getGlobalNodeId( contact_elem, 1 );
      contact_nodes[0] = add_node( contact_node0 );
      contact_nodes[1] = add_node( contact_node1 );

      num_elems = 0;
      for ( int e = 0; e < mesh.numberOfElements(); ++e ) {
        const int elem_node0 = mesh.getGlobalNodeId( e, 0 );
        const int elem_node1 = mesh.getGlobalNodeId( e, 1 );
        if ( elem_node0 != contact_node0 && elem_node0 != contact_node1 && elem_node1 != contact_node0 &&
             elem_node1 != contact_node1 ) {
          continue;
        }
        SLIC_ERROR_ROOT_IF( num_elems >= h1_max_stencil_elems_per_mesh,
                            "ENERGY_MORTAR H1 normal stencil exceeded supported element count." );
        const int local0 = add_node( elem_node0 );
        const int local1 = add_node( elem_node1 );
        elem_nodes[num_elems][0] = local0;
        elem_nodes[num_elems][1] = local1;
        if ( owners[local0] < 0 ) {
          owners[local0] = e;
        }
        if ( owners[local1] < 0 ) {
          owners[local1] = e;
        }
        ++num_elems;
      }

      num_nodes = static_cast<int>( nodes.size() );
      for ( int i = 0; i < num_nodes; ++i ) {
        node_ids[i] = nodes[i];
        owner_elems[i] = owners[i] >= 0 ? owners[i] : contact_elem;
        xref[i] =
            mesh.hasReferencePosition() ? mesh.getReferencePosition()[0][nodes[i]] : mesh.getPosition()[0][nodes[i]];
        xref[num_nodes + i] =
            mesh.hasReferencePosition() ? mesh.getReferencePosition()[1][nodes[i]] : mesh.getPosition()[1][nodes[i]];
      }
    };

    build_side( mesh1, pair.m_element_id1, result.num_mesh1_nodes, data.num_elems1, result.mesh1_nodes,
                result.mesh1_owner_elems, data.contact_nodes1, data.elem_nodes1, data.xref1 );
    data.num_nodes1 = result.num_mesh1_nodes;
    build_side( mesh2, pair.m_element_id2, result.num_mesh2_nodes, data.num_elems2, result.mesh2_nodes,
                result.mesh2_owner_elems, data.contact_nodes2, data.elem_nodes2, data.xref2 );
    data.num_nodes2 = result.num_mesh2_nodes;

    const int ndof = 2 * ( data.num_nodes1 + data.num_nodes2 );
    double x[2 * 2 * h1_max_stencil_nodes_per_mesh] = { 0.0 };
    for ( int i = 0; i < data.num_nodes1; ++i ) {
      const int node_id = result.mesh1_nodes[i];
      x[i] = mesh1.getPosition()[0][node_id];
      x[data.num_nodes1 + i] = mesh1.getPosition()[1][node_id];
    }
    const int side2_offset = 2 * data.num_nodes1;
    for ( int i = 0; i < data.num_nodes2; ++i ) {
      const int node_id = result.mesh2_nodes[i];
      x[side2_offset + i] = mesh2.getPosition()[0][node_id];
      x[side2_offset + data.num_nodes2 + i] = mesh2.getPosition()[1][node_id];
    }

    result.h1_force.assign( ndof, 0.0 );
    result.h1_stiffness.assign( ndof * ndof, 0.0 );
    if ( p_.penalty_mode == EnergyMortarPenaltyMode::NODAL_ENERGY ) {
      h1_nodal_energy_kernel_eval( x, &data, &result.energy );
      grad_h1_nodal_energy_kernel( x, &data, result.h1_force.data() );
      d2_h1_nodal_energy_kernel( x, &data, result.h1_stiffness.data() );
    } else {
      h1_qp_penalty_kernel_eval( x, &data, &result.energy );
      grad_h1_qp_penalty_kernel( x, &data, result.h1_force.data() );
      d2_h1_qp_penalty_kernel( x, &data, result.h1_stiffness.data() );
    }
    return result;
  }

  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );
  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  QPPenaltyKernelData data;
  data.N = p_.N;
  data.del = p_.del;
  data.k = p_.k;
  data.residual_gap = p_.residual_gap;
  data.active_set_smoothing_gap = p_.h1_active_set_smoothing_gap;
  data.projection_smoothing = p_.projection_smoothing;
  data.fixed_quadrature = !p_.enzyme_quadrature;
  if ( data.fixed_quadrature ) {
    data.qp = construct_gparams( pair, mesh1, mesh2 );
  }

  QuadraturePointPenaltyData result;
  qp_penalty_kernel_eval( x, &data, &result.energy );
  grad_qp_penalty_kernel( x, &data, result.force.data() );
  d2_qp_penalty_kernel( x, &data, result.stiffness.data() );
  return result;
}

H1TotalDerivatives EnergyMortarCalculator::compute_h1_total_derivatives( const InterfacePair& pair,
                                                                         const MeshData::Viewer& mesh1,
                                                                         const MeshData::Viewer& mesh2,
                                                                         bool compute_second_derivatives ) const
{
  H1TotalDerivatives result;
  H1KernelData data;
  data.N = p_.N;
  data.del = p_.del;
  data.residual_gap = p_.residual_gap;
  data.projection_smoothing = p_.projection_smoothing;

  auto build_side = []( const MeshData::Viewer& mesh, int contact_elem, int& num_nodes, int& num_elems,
                        std::array<int, h1_max_stencil_nodes_per_mesh>& node_ids,
                        std::array<int, h1_max_stencil_nodes_per_mesh>& owner_elems, int contact_nodes[2],
                        int elem_nodes[h1_max_stencil_elems_per_mesh][2], double* xref ) {
    std::vector<int> nodes;
    std::vector<int> owners;

    auto add_node = [&]( int node_id ) {
      auto it = std::find( nodes.begin(), nodes.end(), node_id );
      if ( it != nodes.end() ) {
        return static_cast<int>( std::distance( nodes.begin(), it ) );
      }
      SLIC_ERROR_ROOT_IF( static_cast<int>( nodes.size() ) >= h1_max_stencil_nodes_per_mesh,
                          "ENERGY_MORTAR H1 normal stencil exceeded supported node count." );
      nodes.push_back( node_id );
      owners.push_back( -1 );
      return static_cast<int>( nodes.size() - 1 );
    };

    const int contact_node0 = mesh.getGlobalNodeId( contact_elem, 0 );
    const int contact_node1 = mesh.getGlobalNodeId( contact_elem, 1 );
    contact_nodes[0] = add_node( contact_node0 );
    contact_nodes[1] = add_node( contact_node1 );

    num_elems = 0;
    for ( int e = 0; e < mesh.numberOfElements(); ++e ) {
      const int elem_node0 = mesh.getGlobalNodeId( e, 0 );
      const int elem_node1 = mesh.getGlobalNodeId( e, 1 );
      if ( elem_node0 != contact_node0 && elem_node0 != contact_node1 && elem_node1 != contact_node0 &&
           elem_node1 != contact_node1 ) {
        continue;
      }
      SLIC_ERROR_ROOT_IF( num_elems >= h1_max_stencil_elems_per_mesh,
                          "ENERGY_MORTAR H1 normal stencil exceeded supported element count." );
      const int local0 = add_node( elem_node0 );
      const int local1 = add_node( elem_node1 );
      elem_nodes[num_elems][0] = local0;
      elem_nodes[num_elems][1] = local1;
      if ( owners[local0] < 0 ) {
        owners[local0] = e;
      }
      if ( owners[local1] < 0 ) {
        owners[local1] = e;
      }
      ++num_elems;
    }

    num_nodes = static_cast<int>( nodes.size() );
    for ( int i = 0; i < num_nodes; ++i ) {
      node_ids[i] = nodes[i];
      owner_elems[i] = owners[i] >= 0 ? owners[i] : contact_elem;
      xref[i] =
          mesh.hasReferencePosition() ? mesh.getReferencePosition()[0][nodes[i]] : mesh.getPosition()[0][nodes[i]];
      xref[num_nodes + i] =
          mesh.hasReferencePosition() ? mesh.getReferencePosition()[1][nodes[i]] : mesh.getPosition()[1][nodes[i]];
    }
  };

  build_side( mesh1, pair.m_element_id1, result.num_mesh1_nodes, data.num_elems1, result.mesh1_nodes,
              result.mesh1_owner_elems, data.contact_nodes1, data.elem_nodes1, data.xref1 );
  data.num_nodes1 = result.num_mesh1_nodes;
  build_side( mesh2, pair.m_element_id2, result.num_mesh2_nodes, data.num_elems2, result.mesh2_nodes,
              result.mesh2_owner_elems, data.contact_nodes2, data.elem_nodes2, data.xref2 );
  data.num_nodes2 = result.num_mesh2_nodes;

  const int ndof = 2 * ( data.num_nodes1 + data.num_nodes2 );
  double x[2 * 2 * h1_max_stencil_nodes_per_mesh] = { 0.0 };
  for ( int i = 0; i < data.num_nodes1; ++i ) {
    const int node_id = result.mesh1_nodes[i];
    x[i] = mesh1.getPosition()[0][node_id];
    x[data.num_nodes1 + i] = mesh1.getPosition()[1][node_id];
  }
  const int side2_offset = 2 * data.num_nodes1;
  for ( int i = 0; i < data.num_nodes2; ++i ) {
    const int node_id = result.mesh2_nodes[i];
    x[side2_offset + i] = mesh2.getPosition()[0][node_id];
    x[side2_offset + data.num_nodes2 + i] = mesh2.getPosition()[1][node_id];
  }

  double gt[2];
  double A[2];
  h1_kernel_eval( x, &data, gt, A );
  result.g_tilde = { gt[0], gt[1] };
  result.area = { A[0], A[1] };

  result.dg1_dx.assign( ndof, 0.0 );
  result.dg2_dx.assign( ndof, 0.0 );
  result.dA1_dx.assign( ndof, 0.0 );
  result.dA2_dx.assign( ndof, 0.0 );
  grad_h1_kernel<KernelOutput::GTILDE1>( x, &data, result.dg1_dx.data() );
  grad_h1_kernel<KernelOutput::GTILDE2>( x, &data, result.dg2_dx.data() );
  grad_h1_kernel<KernelOutput::A1>( x, &data, result.dA1_dx.data() );
  grad_h1_kernel<KernelOutput::A2>( x, &data, result.dA2_dx.data() );

  if ( compute_second_derivatives ) {
    result.d2g1_dx2.assign( ndof * ndof, 0.0 );
    result.d2g2_dx2.assign( ndof * ndof, 0.0 );
    result.d2A1_dx2.assign( ndof * ndof, 0.0 );
    result.d2A2_dx2.assign( ndof * ndof, 0.0 );
    d2_h1_kernel<KernelOutput::GTILDE1>( x, &data, result.d2g1_dx2.data() );
    d2_h1_kernel<KernelOutput::GTILDE2>( x, &data, result.d2g2_dx2.data() );
    d2_h1_kernel<KernelOutput::A1>( x, &data, result.d2A1_dx2.data() );
    d2_h1_kernel<KernelOutput::A2>( x, &data, result.d2A2_dx2.data() );
  }

  return result;
}

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
