#pragma once
#include <vector>
#include <array>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "tribol/config.hpp"

#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

// 0 = no asserts, 1 = cheap asserts, 2 = expensive asserts (FD checks, etc.)
constexpr int energy_mortar_debug_level = 1;

// EnergyMortar uses a 3-point Gauss-Legendre quad rule
struct QuadPoints {
  std::array<double, 3> qp;  // qp locations
  std::array<double, 3> w;   // weights
};

enum class SmoothingType
{
  Hermite,   // C1 cubic Hermite ramp. Slope varies within ramp (peaks at 4/3).
  Quadratic  // C1 quadratic ramp. Linear section has slope 1/(1-del) != 1.
};

enum class PenaltySmoothing
{
  Hard,  // Hard clamp: p = k * max(-g, 0). C0 kink in Jacobian at g = 0.
  Smooth // C1 quadratic ramp over [-del/2, +del/2]: penalty decays smoothly to zero.
};

struct ContactParams {
  double del;              // Smoothing Parameter
  double k;                // Penalty
  int N;                   // Quadrature Points
  bool enzyme_quadrature;  // Determines how enzyming is performed (default = True)
  SmoothingType smoothing_type = SmoothingType::Quadratic;
  PenaltySmoothing penalty_smoothing = PenaltySmoothing::Smooth;
  double penalty_smoothing_del = 1.0e-3;
};

// Weighted gap and trib area
struct NodalContactData {
  std::array<double, 2> AI;       // Trib area
  std::array<double, 2> g_tilde;  // Weighted gap
};

/// Stores finite-difference and analytical derivative data for validation tests.
struct FiniteDiffResult {
  /// Finite-difference approximation of the gradient/Hessian for the first
  /// nodal smoothed gap contribution.
  std::vector<double> fd_gradient_g1;

  /// Finite-difference approximation of the gradient/Hessian for the second
  /// nodal smoothed gap contribution.
  std::vector<double> fd_gradient_g2;

  /// Analytical/Enzyme-computed gradient/Hessian for the first nodal smoothed
  /// gap contribution.
  std::vector<double> analytical_gradient_g1;

  /// Analytical/Enzyme-computed gradient/Hessian for the second nodal smoothed
  /// gap contribution.
  std::vector<double> analytical_gradient_g2;

  /// Global node ids associated with the differentiated degrees of freedom.
  /// The expected ordering is the two nodes of edge A followed by the two nodes
  /// of edge B.
  std::vector<int> node_ids;

  /// Baseline value of the first nodal smoothed gap contribution before applying
  /// any finite-difference perturbations.
  double g_tilde1_baseline{ 0.0 };

  /// Baseline value of the second nodal smoothed gap contribution before applying
  /// any finite-difference perturbations.
  double g_tilde2_baseline{ 0.0 };
};

struct Gparams {
  std::array<double, 3> qp;
  std::array<double, 3> w;
};

/// Evaluate the fixed-quadrature gap/area kernel for a local edge-pair coordinate vector.
///
/// `x` stores `[A0x, A0y, A1x, A1y, B0x, B0y, B1x, B1y]`.
void energy_mortar_fixed_quadrature_kernel( const double x[8], const Gparams& gp, double g_tilde[2], double area[2] );

/// Evaluate the geometry-dependent quadrature gap/area kernel for a local edge-pair coordinate vector.
///
/// `x` stores `[A0x, A0y, A1x, A1y, B0x, B0y, B1x, B1y]`. This function recomputes projection bounds, applies contact
/// smoothing, constructs quadrature, then evaluates the same gap/area kernel as the element path.
void energy_mortar_varying_quadrature_kernel( const double x[8], double del, int N, SmoothingType smoothing_type,
                                              double g_tilde[2], double area[2] );

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

inline std::array<double, 2> bounds_from_projections( const std::array<double, 2>& proj, double del )
{
  double xi_min = std::min( proj[0], proj[1] );
  double xi_max = std::max( proj[0], proj[1] );
  const double xi_lo = -0.5;
  const double xi_hi = 0.5;

  if ( xi_max < xi_lo ) {
    xi_max = xi_lo;
  }
  if ( xi_min > xi_hi ) {
    xi_min = xi_hi;
  }
  if ( xi_min < xi_lo ) {
    xi_min = xi_lo;
  }
  if ( xi_max > xi_hi ) {
    xi_max = xi_hi;
  }

  return { xi_min, xi_max };
}

inline std::array<double, 2> smooth_bounds( const std::array<double, 2>& bounds, double del, SmoothingType type )
{
  std::array<double, 2> smooth_bounds;
  for ( int i = 0; i < 2; ++i ) {
    double xi = bounds[i] + 0.5;
    xi = std::max( 0.0, std::min( 1.0, xi ) );

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
    smooth_bounds[i] = xi_hat - 0.5;
  }
  return smooth_bounds;
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
  const double J_ref = J;

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
    AI_1 += w * N1 * J_ref;
    AI_2 += w * N2 * J_ref;
  }

  g_tilde_out[0] = g1;
  g_tilde_out[1] = g2;
  A_out[0] = AI_1;
  A_out[1] = AI_2;
}

}  // namespace energy_mortar_kernel_detail

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

struct PenaltyRamp {
  double value;
  double first_derivative;
  double second_derivative;
};

/// Provides smoothing operations for the Energy Mortar contact formulation.
///
/// This class stores the contact parameters and provides helper routines for
/// constructing smoothed integration bounds from projected overlap intervals.
class ContactSmoothing {
 public:
  /// Clamp the projected overlap interval to the selected smoothing support.
  ///
  /// The input `proj` contains the local projection bounds of edge B onto edge A.
  /// Inside restricts to `[-0.5, 0.5]`; Outside restricts to `[-0.5-del, 0.5+del]`.
  static std::array<double, 2> bounds_from_projections(
      const std::array<double, 2>& proj, double del );

  /// Smooth the integration bounds using the smoothing length `del`.
  ///
  /// The returned bounds are obtained by applying the endpoint smoothing map to
  /// the clamped integration interval. When `del = 0`, the bounds are returned
  /// without smoothing.
  static std::array<double, 2> smooth_bounds( const std::array<double, 2>& bounds, double del,
                                               SmoothingType type = SmoothingType::Hermite );

  /// Evaluate the penalty ramp H(g), H'(g), and H''(g).
  static PenaltyRamp penalty_ramp( double g, double del, PenaltySmoothing type = PenaltySmoothing::Hard );
};

/// Evaluates Energy Mortar contact quantities for a single interface pair.
///
/// This class computes the smoothed mortar gap, tributary areas, contact energy,
/// contact forces, stiffness contributions, and derivative checks used by the
/// Energy Mortar contact formulation. The interface pair is assumed to contain
/// one face from mesh1, treated as edge A/non-mortar/integration side, and one
/// face from mesh2, treated as edge B/mortar/projection side.
class EnergyMortarCalculator {
 public:
  /// Construct a contact evaluator with the supplied contact parameters.
  ///
  /// The parameters define the penalty stiffness, smoothing length, and
  /// derivative path used by the evaluator.
  explicit EnergyMortarCalculator( const ContactParams& p )
      : p_( p ), smoother_() {}  // constructor - copies params into the object

  int get_N() const { return p_.N; }

  std::array<double, 2> compute_projection_bounds( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                   const MeshData::Viewer& mesh2 ) const
  {
    return projections( pair, mesh1, mesh2 );
  }

  /// Construct a three-point Gauss-Legendre quadrature rule over local bounds.
  ///
  /// The input bounds are local coordinates on edge A. The returned quadrature
  /// points and weights are mapped from the reference interval to
  /// `[xi_bounds[0], xi_bounds[1]]`
  static QuadPoints compute_quadrature( const std::array<double, 2>& xi_bounds, int N );

  /// Compute the nodal smoothed gap integrals and tributary areas.
  ///
  /// The output arrays each have length 2, with entries corresponding to the
  /// two nodes of edge A. `gtilde` stores the integrated smoothed gap
  /// contributions, while `area` stores the corresponding tributary area
  /// contributions.
  void compute_gtilde_and_area( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                double gtilde[2], double area[2] ) const;

  /// Compute first derivatives of the nodal smoothed gap integrals.
  ///
  /// `dgt1_dx` and `dgt2_dx` each have length 8 and store derivatives of the
  /// two nodal gap integrals with respect to the endpoint coordinate vector
  /// described in the class documentation. If `enzyme_quadrature` is false,
  /// the quadrature rule is held fixed during differentiation. If true, the
  /// derivative includes the geometry-dependent quadrature construction.
  void grad_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                    double dgt1_dx[8], double dgt2_dx[8] ) const;

  /// Compute first derivatives of the nodal tributary areas.
  ///
  /// `dA1_dx` and `dA2_dx` each have length 8 and store derivatives of the
  /// two nodal tributary area contributions with respect to the endpoint
  /// coordinate vector described in the class documentation. If
  /// `enzyme_quadrature` is false, the quadrature rule is held fixed during
  /// differentiation. If true, the derivative includes the geometry-dependent
  /// quadrature construction.
  void grad_trib_area( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                       double dA1_dx[8], double dA2_dx[8] ) const;

  /// Compute second derivatives of the nodal smoothed gap integrals.
  ///
  /// `H1` and `H2` each have length 64 and store flattened 8 by 8 Hessian
  /// matrices for the two nodal gap integrals. Entries use row-major indexing:
  /// `H[row * 8 + col]`. If `enzyme_quadrature` is false, the quadrature rule
  /// is held fixed during differentiation. If true, the derivative includes the
  /// geometry-dependent quadrature construction.
  void d2_g2tilde( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                   double dgt1_dx[64], double dgt2_dx[64] ) const;

  /// Compute second derivatives of the nodal tributary areas.
  ///
  /// `d2A1` and `d2A2` each have length 64 and store flattened 8 by 8 Hessian
  /// matrices for the two nodal tributary area contributions. Entries use
  /// row-major indexing: `H[row * 8 + col]`. If `enzyme_quadrature` is false,
  /// the quadrature rule is held fixed during differentiation. If true, the
  /// derivative includes the geometry-dependent quadrature construction.
  void compute_d2A_d2u( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                        double dgt1_dx[64], double dgt2_dx[64] ) const;

  /// Evaluate and return the two nodal smoothed gap integrals.
  ///
  /// This is a convenience wrapper for obtaining only the gap integral
  /// quantities without also returning the tributary areas for the Finite
  /// difference test.
  std::pair<double, double> eval_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                         const MeshData::Viewer& mesh2 ) const;

  /// Validate first derivatives of the smoothed gap integrals using finite differences.
  ///
  /// This routine perturbs the endpoint coordinates of the interface pair and
  /// compares finite-difference approximations against the Enzyme-computed
  /// gradients. The perturbation size is controlled by `epsilon`.
  FiniteDiffResult validate_g_tilde( const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
                                     double epsilon = 1e-7 ) const;

  /// Evaluate the two nodal smoothed gap integrals using a fixed quadrature rule.
  ///
  /// This is used for derivative verification when the quadrature points and
  /// weights should remain fixed under coordinate perturbations. Holding the
  /// quadrature fixed isolates derivatives of the gap kernel from derivatives
  /// of the geometry-dependent quadrature construction.
  std::pair<double, double> eval_gtilde_fixed_qp( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                  const MeshData::Viewer& mesh2, const QuadPoints& qp_fixed ) const;

  /// Validate second derivatives of the smoothed gap integrals using finite differences.
  ///
  /// This routine compares Enzyme-computed Hessians against finite-difference
  /// approximations of the first derivatives. The perturbation size is
  /// controlled by `epsilon`.
  FiniteDiffResult validate_hessian( const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
                                     double epsilon = 1e-7 ) const;

 private:
  /// Contact parameters controlling penalty stiffness, smoothing, and derivative behavior.
  ContactParams p_;
  /// Helper used to construct smoothed integration bounds
  ContactSmoothing smoother_;

  /// Construct the gap-kernel parameter bundle for the current interface pair.
  ///
  /// This builds the smoothed integration bounds, quadrature points, quadrature
  /// weights, and projected quadrature-point coordinates needed by the lower-level
  /// gap kernel.
  Gparams construct_gparams( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                             const MeshData::Viewer& mesh2 ) const;

  /// Compute the local projection bounds of edge B onto edge A.
  ///
  /// The returned values are local coordinates on edge A and define the interval
  /// used to construct the smoothed integration bounds.
  std::array<double, 2> projections( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                     const MeshData::Viewer& mesh2 ) const;

  /// Compute smoothed gap gradients while holding the quadrature rule fixed.
  ///
  /// This is used by finite-difference verification routines to isolate
  /// derivatives of the gap kernel from derivatives of the quadrature rule.
  void grad_gtilde_with_qp( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                            const QuadPoints& qp_fixed, double dgt1_dx[8], double dgt2_dx[8] ) const;

  /// Evaluate the signed normal gap at a local coordinate on edge A.
  ///
  /// The point on edge A is projected onto edge B, and the gap is evaluated
  /// using the normal from edge B together with the normal-alignment factor.
  double compute_weighted_normal_gap( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                      const MeshData::Viewer& mesh2, double xiA ) const;

  /// Assemble nodal gap and tributary area data for one interface pair.
  ///
  /// This computes the two smoothed nodal gap integrals and the two corresponding
  /// tributary area contributions used to evaluate pressures, forces, energy,
  /// and stiffness terms.
  NodalContactData compute_nodal_contact_data( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                               const MeshData::Viewer& mesh2 ) const;
};

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
