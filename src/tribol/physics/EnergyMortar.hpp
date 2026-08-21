#pragma once
#include <vector>
#include <array>

#include "tribol/config.hpp"

#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

/// Stores quadrature-point locations and weights for the supported Gauss-Legendre rule.
struct QuadPoints {
  std::array<double, 3> qp;  ///< Quadrature-point locations in the local coordinate of the integration edge.
  std::array<double, 3> w;   ///< Quadrature weights mapped to the local integration interval.
};

/// Parameters controlling ENERGY_MORTAR contact evaluation.
struct ContactParams {
  double del;              ///< Smoothing length used for integration bounds.
  double k;                ///< Penalty stiffness.
  int N;                   ///< Number of quadrature points.
  bool enzyme_quadrature;  ///< Whether Enzyme differentiates the quadrature construction.
};

/// Stores quadrature-point penalty energy derivatives for one interface pair.
struct QuadraturePointPenaltyData {
  static constexpr int dim = 2;                 ///< Spatial dimension.
  static constexpr int max_nodes_per_elem = 2;  ///< Maximum nodes on each line element.
  static constexpr int pair_size = 2;           ///< Number of elements in an interface pair.
  static constexpr int num_force_dofs = dim * max_nodes_per_elem * pair_size;  ///< Pair coordinate degrees of freedom.
  static constexpr int num_stiffness_entries = num_force_dofs * num_force_dofs;  ///< Flattened stiffness size.

  double energy{ 0.0 };                                   ///< Penalty energy for the interface pair.
  std::array<double, num_force_dofs> force{};             ///< Derivative with respect to pair coordinates.
  std::array<double, num_stiffness_entries> stiffness{};  ///< Flattened force derivative matrix.
};

/// Stores weighted nodal gaps and tributary areas for one interface pair.
struct NodalContactData {
  std::array<double, 2> AI;       ///< Tributary areas for the two integration-edge nodes.
  std::array<double, 2> g_tilde;  ///< Weighted gaps for the two integration-edge nodes.
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

/// Stores fixed quadrature data passed to differentiated kernels.
struct Gparams {
  std::array<double, 3> qp;  ///< Quadrature-point locations in the integration-edge local coordinate.
  std::array<double, 3> w;   ///< Quadrature weights mapped to the local integration interval.
};

/// Provides smoothing operations for the Energy Mortar contact formulation.
///
/// This class stores the contact parameters and provides helper routines for
/// constructing smoothed integration bounds from projected overlap intervals.
class ContactSmoothing {
 public:
  /// Clamp the projected overlap interval to the extended smoothing support.
  ///
  /// The input `projections` contains the local projection bounds of edge B onto edge A.
  /// The output `bounds` is restricted to the extended local range `[-0.5 - del, 0.5 + del]`.
  static void bounds_from_projections( const double* projections, double del, double* bounds );

  /// Smooth the integration bounds using the smoothing length `del`.
  ///
  /// The output `smooth_bounds` is obtained by applying the endpoint smoothing map to
  /// the clamped integration interval. When `del = 0`, the bounds are unchanged.
  static void smooth_bounds( const double* bounds, double del, double* smooth_bounds );
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
  /// `[xi_bounds[0], xi_bounds[1]]` and written to `quadrature`.
  static void compute_quadrature( const double* xi_bounds, int N, QuadPoints* quadrature );

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

  /// Compute local energy, force, and stiffness for quadrature-point penalty enforcement.
  QuadraturePointPenaltyData compute_quadrature_point_penalty_data( const InterfacePair& pair,
                                                                    const MeshData::Viewer& mesh1,
                                                                    const MeshData::Viewer& mesh2 ) const;

  /// Evaluate only the local quadrature-point penalty energy.
  double compute_quadrature_point_penalty_energy( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                  const MeshData::Viewer& mesh2 ) const;

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
