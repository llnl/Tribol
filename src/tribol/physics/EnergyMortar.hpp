#pragma once
#include <vector>
#include <array>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "tribol/config.hpp"
#include "tribol/physics/EnergyMortarTypes.hpp"

#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

// 0 = no asserts, 1 = cheap asserts, 2 = expensive asserts (FD checks, etc.)
constexpr int energy_mortar_debug_level = 1;

struct QuadPoints {
  std::array<double, 3> qp;
  std::array<double, 3> w;
};

/// Unified options for the Energy Mortar contact formulation.
struct EnergyMortarOptions {
  double k = 1000.0;                // Penalty stiffness
  double del = 0.1;                 // Smoothing parameter for integration bounds
  int N = 3;                        // Quadrature order
  bool enzyme_quadrature = true;    // Differentiate through quadrature construction
  SmoothingType smoothing_type = SmoothingType::Quadratic;
  PenaltySmoothing penalty_smoothing = PenaltySmoothing::Smooth;
  double penalty_smoothing_del = 1.0e-3;
};

// Legacy alias — will be removed once all callers migrate
using ContactParams = EnergyMortarOptions;

struct PenaltyRamp {
  double value;
  double first_derivative;
  double second_derivative;
};

// Weighted gap and trib area
struct NodalContactData {
  std::array<double, 2> AI;       // Trib area
  std::array<double, 2> g_tilde;  // Weighted gap
};

/// Stores finite-difference and analytical derivative data for validation tests.
struct FiniteDiffResult {
  std::vector<double> fd_gradient_g1;
  std::vector<double> fd_gradient_g2;
  std::vector<double> analytical_gradient_g1;
  std::vector<double> analytical_gradient_g2;
  std::vector<int> node_ids;
  double g_tilde1_baseline{ 0.0 };
  double g_tilde2_baseline{ 0.0 };
};

struct Gparams {
  std::array<double, 3> qp;
  std::array<double, 3> w;
};

// ============================================================================
// Inline kernel implementations — single source of truth for both the element
// evaluation path and the Enzyme AD path.
// ============================================================================

namespace energy_mortar_kernel_detail {

inline void find_normal( const double* coord1, const double* coord2, double* normal )
{
  double dx = coord2[0] - coord1[0];
  double dy = coord2[1] - coord1[1];
  const double len = std::sqrt( dy * dy + dx * dx );
  dx /= len;
  dy /= len;
  normal[0] = dy;
  normal[1] = -dx;
}

inline void iso_map( const double* coord1, const double* coord2, double xi, double* mapped_coord )
{
  const double N1 = 0.5 - xi;
  const double N2 = 0.5 + xi;
  mapped_coord[0] = N1 * coord1[0] + N2 * coord2[0];
  mapped_coord[1] = N1 * coord1[1] + N2 * coord2[1];
}

inline void find_intersection( const double* A0, const double* A1, const double* p, const double* nB,
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
  if ( std::abs( det ) < 1e-12 ) {
    intersection[0] = p[0];
    intersection[1] = p[1];
    return;
  }
  const double alpha = ( d[0] * n[1] - d[1] * n[0] ) / det;
  intersection[0] = A0[0] + alpha * tA[0];
  intersection[1] = A0[1] + alpha * tA[1];
}

inline void get_projections( const double* A0, const double* A1, const double* B0, const double* B1,
                             double* projections )
{
  double nB[2];
  find_normal( B0, B1, nB );

  const double dxA = A1[0] - A0[0];
  const double dyA = A1[1] - A0[1];
  const double len2A = dxA * dxA + dyA * dyA;
  const double* B_endpoints[2] = { B0, B1 };

  double xis[2];
  for ( int i = 0; i < 2; ++i ) {
    double q[2];
    find_intersection( A0, A1, B_endpoints[i], nB, q );
    const double alphaA = ( ( q[0] - A0[0] ) * dxA + ( q[1] - A0[1] ) * dyA ) / len2A;
    xis[i] = alphaA - 0.5;
  }

  projections[0] = std::min( xis[0], xis[1] );
  projections[1] = std::max( xis[0], xis[1] );
}

inline std::array<double, 2> bounds_from_projections( const std::array<double, 2>& proj, double /*del*/ )
{
  double xi_min = std::min( proj[0], proj[1] );
  double xi_max = std::max( proj[0], proj[1] );
  constexpr double xi_lo = -0.5;
  constexpr double xi_hi = 0.5;

  xi_min = std::max( xi_lo, std::min( xi_hi, xi_min ) );
  xi_max = std::max( xi_lo, std::min( xi_hi, xi_max ) );

  return { xi_min, xi_max };
}

inline std::array<double, 2> smooth_bounds( const std::array<double, 2>& bounds, double del, SmoothingType type )
{
  std::array<double, 2> result;
  for ( int i = 0; i < 2; ++i ) {
    double xi = std::max( 0.0, std::min( 1.0, bounds[i] + 0.5 ) );

    double xi_hat = xi;
    if ( del > 0.0 && del < 0.5 ) {
      if ( type == SmoothingType::Hermite ) {
        if ( xi <= del ) {
          const double t = xi / del;
          xi_hat = del * t * t * ( 2.0 - t );
        } else if ( xi >= 1.0 - del ) {
          const double s = ( 1.0 - xi ) / del;
          xi_hat = 1.0 - del * s * s * ( 2.0 - s );
        }
      } else {
        const double a = 1.0 / ( 2.0 * del * ( 1.0 - del ) );
        const double m = 1.0 / ( 1.0 - del );
        if ( xi <= del ) {
          xi_hat = a * xi * xi;
        } else if ( xi >= 1.0 - del ) {
          xi_hat = 1.0 - a * ( 1.0 - xi ) * ( 1.0 - xi );
        } else {
          xi_hat = m * xi - del * 0.5 * m;
        }
      }
    }

    if constexpr ( energy_mortar_debug_level >= 1 ) {
      assert( xi_hat >= 0.0 && "smooth_bounds: xi_hat below 0" );
      assert( xi_hat <= 1.0 && "smooth_bounds: xi_hat above 1" );
    }

    result[i] = xi_hat - 0.5;
  }

  if constexpr ( energy_mortar_debug_level >= 1 ) {
    assert( result[0] >= -0.5 && result[0] <= 0.5 && "smooth_bounds: output[0] out of range" );
    assert( result[1] >= -0.5 && result[1] <= 0.5 && "smooth_bounds: output[1] out of range" );
  }

  return result;
}

inline PenaltyRamp penalty_ramp( double g, double del, PenaltySmoothing type )
{
  if ( type == PenaltySmoothing::Hard ) {
    return g < 0.0 ? PenaltyRamp{ g, 1.0, 0.0 } : PenaltyRamp{ 0.0, 0.0, 0.0 };
  }

  const double d2 = 0.5 * del;

  if ( g >= d2 ) {
    return { 0.0, 0.0, 0.0 };
  }

  if ( del <= 0.0 || g <= -d2 ) {
    return { g, 1.0, 0.0 };
  }

  const double shifted = g - d2;
  return { -shifted * shifted / ( 2.0 * del ), -shifted / del, -1.0 / del };
}

inline QuadPoints compute_quadrature( const std::array<double, 2>& xi_bounds, int N )
{
  QuadPoints out;
  std::array<double, 3> qpoints = { 0.0, 0.0, 0.0 };
  std::array<double, 3> weights = { 0.0, 0.0, 0.0 };

  if ( N == 1 ) {
    qpoints[0] = 0.0;
    weights[0] = 2.0;
  } else if ( N == 2 ) {
    const double a = 1.0 / std::sqrt( 3.0 );
    qpoints[0] = -a;
    qpoints[1] = a;
    weights[0] = 1.0;
    weights[1] = 1.0;
  } else {
    const double a = std::sqrt( 3.0 / 5.0 );
    qpoints[0] = -a;
    qpoints[1] = 0.0;
    qpoints[2] = a;
    weights[0] = 5.0 / 9.0;
    weights[1] = 8.0 / 9.0;
    weights[2] = 5.0 / 9.0;
  }

  const double xi_min = xi_bounds[0];
  const double xi_max = xi_bounds[1];
  const double J = 0.5 * ( xi_max - xi_min );
  for ( int i = 0; i < N; ++i ) {
    out.qp[i] = 0.5 * ( xi_max - xi_min ) * qpoints[i] + 0.5 * ( xi_max + xi_min );
    out.w[i] = weights[i] * J;
  }
  return out;
}

inline void gap_area_kernel( const double* x, const Gparams& gp, double* g_tilde_out, double* A_out )
{
  const double A0[2] = { x[0], x[1] };
  const double A1[2] = { x[2], x[3] };
  const double B0[2] = { x[4], x[5] };
  const double B1[2] = { x[6], x[7] };

  const double J = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) +
                              ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  double nB[2];
  find_normal( B0, B1, nB );
  double nA[2];
  find_normal( A0, A1, nA );
  const double dot = nB[0] * nA[0] + nB[1] * nA[1];
  const double eta = ( dot < 0.0 ) ? dot : 0.0;

  double g1 = 0.0;
  double g2 = 0.0;
  double AI_1 = 0.0;
  double AI_2 = 0.0;

  for ( int i = 0; i < 3; ++i ) {
    const double xiA = gp.qp[i];
    const double w = gp.w[i];
    const double N1 = 0.5 - xiA;
    const double N2 = 0.5 + xiA;

    double x1[2];
    iso_map( A0, A1, xiA, x1 );
    double x2[2];
    find_intersection( B0, B1, x1, nB, x2 );

    const double dx = x1[0] - x2[0];
    const double dy = x1[1] - x2[1];
    const double gn = -( dx * nB[0] + dy * nB[1] );
    const double g = gn * eta;

    g1 += w * N1 * g * J;
    g2 += w * N2 * g * J;
    AI_1 += w * N1 * J;
    AI_2 += w * N2 * J;
  }

  g_tilde_out[0] = g1;
  g_tilde_out[1] = g2;
  A_out[0] = AI_1;
  A_out[1] = AI_2;
}

}  // namespace energy_mortar_kernel_detail

// ============================================================================
// Public kernel entry points
// ============================================================================

inline void energy_mortar_fixed_quadrature_kernel( const double x[8], const Gparams& gp, double g_tilde[2],
                                                   double area[2] )
{
  energy_mortar_kernel_detail::gap_area_kernel( x, gp, g_tilde, area );
}

inline void energy_mortar_varying_quadrature_kernel( const double x[8], double del, int N,
                                                     SmoothingType smoothing_type, double g_tilde[2],
                                                     double area[2] )
{
  const double A0[2] = { x[0], x[1] };
  const double A1[2] = { x[2], x[3] };
  const double B0[2] = { x[4], x[5] };
  const double B1[2] = { x[6], x[7] };

  double projs[2] = { 0.0, 0.0 };
  energy_mortar_kernel_detail::get_projections( A0, A1, B0, B1, projs );
  const std::array<double, 2> projections = { projs[0], projs[1] };
  const auto bounds = energy_mortar_kernel_detail::bounds_from_projections( projections, del );
  const auto xi_bounds = energy_mortar_kernel_detail::smooth_bounds( bounds, del, smoothing_type );
  const auto qp = energy_mortar_kernel_detail::compute_quadrature( xi_bounds, N );

  Gparams gp;
  for ( std::size_t i = 0; i < qp.qp.size(); ++i ) {
    gp.qp[i] = qp.qp[i];
    gp.w[i] = qp.w[i];
  }

  energy_mortar_kernel_detail::gap_area_kernel( x, gp, g_tilde, area );
}

// ============================================================================
// ContactSmoothing — static interface forwarding to the inline implementations
// ============================================================================

class ContactSmoothing {
 public:
  static std::array<double, 2> bounds_from_projections( const std::array<double, 2>& proj, double del )
  {
    return energy_mortar_kernel_detail::bounds_from_projections( proj, del );
  }

  static std::array<double, 2> smooth_bounds( const std::array<double, 2>& bounds, double del,
                                               SmoothingType type = SmoothingType::Hermite )
  {
    return energy_mortar_kernel_detail::smooth_bounds( bounds, del, type );
  }

  static PenaltyRamp penalty_ramp( double g, double del, PenaltySmoothing type = PenaltySmoothing::Hard )
  {
    return energy_mortar_kernel_detail::penalty_ramp( g, del, type );
  }
};

// ============================================================================
// EnergyMortarCalculator — element-level evaluator
// ============================================================================

class EnergyMortarCalculator {
 public:
  explicit EnergyMortarCalculator( const EnergyMortarOptions& opts ) : opts_( opts ) {}

  int get_N() const { return opts_.N; }

  std::array<double, 2> compute_projection_bounds( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                   const MeshData::Viewer& mesh2 ) const
  {
    return projections( pair, mesh1, mesh2 );
  }

  static QuadPoints compute_quadrature( const std::array<double, 2>& xi_bounds, int N )
  {
    return energy_mortar_kernel_detail::compute_quadrature( xi_bounds, N );
  }

  void compute_gtilde_and_area( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                double gtilde[2], double area[2] ) const;

  void grad_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                    double dgt1_dx[8], double dgt2_dx[8] ) const;

  void grad_trib_area( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                       double dA1_dx[8], double dA2_dx[8] ) const;

  void d2_g2tilde( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                   double dgt1_dx[64], double dgt2_dx[64] ) const;

  void compute_d2A_d2u( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                        double dgt1_dx[64], double dgt2_dx[64] ) const;

  std::pair<double, double> eval_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                         const MeshData::Viewer& mesh2 ) const;

  FiniteDiffResult validate_g_tilde( const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
                                     double epsilon = 1e-7 ) const;

  std::pair<double, double> eval_gtilde_fixed_qp( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                  const MeshData::Viewer& mesh2, const QuadPoints& qp_fixed ) const;

  FiniteDiffResult validate_hessian( const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
                                     double epsilon = 1e-7 ) const;

 private:
  EnergyMortarOptions opts_;
  ContactSmoothing smoother_;

  Gparams construct_gparams( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                             const MeshData::Viewer& mesh2 ) const;

  std::array<double, 2> projections( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                     const MeshData::Viewer& mesh2 ) const;

  void grad_gtilde_with_qp( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                            const QuadPoints& qp_fixed, double dgt1_dx[8], double dgt2_dx[8] ) const;

  double compute_weighted_normal_gap( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                      const MeshData::Viewer& mesh2, double xiA ) const;

  NodalContactData compute_nodal_contact_data( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                               const MeshData::Viewer& mesh2 ) const;
};

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
