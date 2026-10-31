#pragma once
#include <vector>
#include <array>

#include "tribol/config.hpp"
#include "tribol/common/BasicTypes.hpp"

#include "tribol/common/Parameters.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/integ/PairQuadratureRule.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

/// Parameters controlling ENERGY_MORTAR contact evaluation.
struct ContactParams {
  RealT k;                    ///< Penalty stiffness.
  bool enzyme_quadrature;     ///< Whether Enzyme differentiates the quadrature construction.
  RealT residual_gap{ 0.0 };  ///< User-defined gap offset subtracted from the kinematic gap.
};

/// Stores quadrature-point penalty energy derivatives for one interface pair.
struct QuadraturePointPenaltyData {
  static constexpr int dim = 2;                 ///< Spatial dimension.
  static constexpr int max_nodes_per_elem = 2;  ///< Maximum nodes on each line element.
  static constexpr int pair_size = 2;           ///< Number of elements in an interface pair.
  static constexpr int num_force_dofs = dim * max_nodes_per_elem * pair_size;  ///< Pair coordinate degrees of freedom.
  static constexpr int num_stiffness_entries = num_force_dofs * num_force_dofs;  ///< Flattened stiffness size.

  bool has_active_qp{ false };                           ///< True when any quadrature-point gap is nonpositive.
  RealT energy{ 0.0 };                                   ///< Penalty energy for the interface pair.
  std::array<RealT, num_force_dofs> force{};             ///< Derivative with respect to pair coordinates.
  std::array<RealT, num_stiffness_entries> stiffness{};  ///< Flattened force derivative matrix.
};

/// Stores weighted nodal gaps and tributary areas for one interface pair.
struct NodalContactData {
  std::array<RealT, 2> AI;       ///< Tributary areas for the two integration-edge nodes.
  std::array<RealT, 2> g_tilde;  ///< Weighted gaps for the two integration-edge nodes.
};

/// Stores finite-difference and analytical derivative data for validation tests.
struct FiniteDiffResult {
  /// Finite-difference approximation of the gradient/Hessian for the first
  /// nodal smoothed gap contribution.
  std::vector<RealT> fd_gradient_g1;

  /// Finite-difference approximation of the gradient/Hessian for the second
  /// nodal smoothed gap contribution.
  std::vector<RealT> fd_gradient_g2;

  /// Analytical/Enzyme-computed gradient/Hessian for the first nodal smoothed
  /// gap contribution.
  std::vector<RealT> analytical_gradient_g1;

  /// Analytical/Enzyme-computed gradient/Hessian for the second nodal smoothed
  /// gap contribution.
  std::vector<RealT> analytical_gradient_g2;

  /// Baseline value of the first nodal smoothed gap contribution before applying
  /// any finite-difference perturbations.
  RealT g_tilde1_baseline{ 0.0 };

  /// Baseline value of the second nodal smoothed gap contribution before applying
  /// any finite-difference perturbations.
  RealT g_tilde2_baseline{ 0.0 };
};

/// Evaluates Energy Mortar quantities from coordinates of one element pair.
///
/// Coordinates contain the first (non-mortar) element followed by the second (mortar) element, with each element's
/// nodes in finite-element order and each node's spatial components adjacent. The calculator retains no coordinate
/// views. Current normals and geometry remain active in differentiation; frozen quadrature fixes integration-surface
/// coordinates and reference weights only. Local edge coordinates use the normalized [-1, 1] interval.
/// Calculations throw std::invalid_argument for invalid geometry and std::length_error for insufficient quadrature
/// storage. Failed quadrature data is never consumed; valid zero-point rules retain their zero contribution.
/// @tparam QuadratureRule Linear two-dimensional element-pair quadrature generator.
template <typename QuadratureRule>
class EnergyMortarCalculator {
  static_assert( QuadratureRule::FirstGeometry::parametric_dimension == 1 &&
                     QuadratureRule::FirstGeometry::vertex_count == 2 &&
                     QuadratureRule::FirstGeometry::spatial_dimension == 2,
                 "Energy Mortar requires a one-dimensional first geometry with two vertices in 2D." );
  static_assert( QuadratureRule::SecondGeometry::parametric_dimension == 1 &&
                     QuadratureRule::SecondGeometry::vertex_count == 2 &&
                     QuadratureRule::SecondGeometry::spatial_dimension == 2,
                 "Energy Mortar requires a one-dimensional second geometry with two vertices in 2D." );

 public:
  static constexpr IndexT element_dof_count = QuadratureRule::FirstGeometry::vertex_count *
                                              QuadratureRule::FirstGeometry::spatial_dimension;  ///< DOFs per edge.
  static constexpr IndexT dof_count = 2 * element_dof_count;  ///< Coordinate DOFs for both elements.
  using Coordinates = MdArray<RealT, dof_count>;          ///< Owning pair coordinates in element/node/component order.
  using CoordinateView = MdSpan<const RealT, dof_count>;  ///< Read-only pair coordinates, used only during a call.
  using IntegrationSurfaceRule = typename QuadratureRule::IntegrationSurfaceRule;  ///< Frozen surface points/weights.

  /// Store a reference to the generator, which must outlive this calculator. Geometry is supplied for each calculation.
  /// @param p Penalty, residual-gap, and derivative settings copied into this calculator.
  /// @param quadrature_rule Quadrature generator that must outlive the calculator.
  EnergyMortarCalculator( const ContactParams& p, const QuadratureRule& quadrature_rule )
      : p_( p ), quadrature_rule_( quadrature_rule )
  {
  }

  EnergyMortarCalculator( const ContactParams&, const QuadratureRule&& ) = delete;

  /// Compute the nodal smoothed gap integrals and tributary areas.
  ///
  /// The output arrays each have length 2, with entries corresponding to the
  /// two nodes of edge A. `gtilde` stores the integrated smoothed gap
  /// contributions, while `area` stores the corresponding tributary area
  /// contributions.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @param[out] gtilde Weighted gaps at the first element's two nodes.
  /// @param[out] area Corresponding nodal tributary areas.
  void compute_gtilde_and_area( CoordinateView coordinates, RealT gtilde[2], RealT area[2] ) const;

  /// Compute first derivatives of the nodal smoothed gap integrals.
  ///
  /// `dgt1_dx` and `dgt2_dx` each have length 8 and store derivatives of the
  /// two nodal gap integrals with respect to the endpoint coordinate vector
  /// described in the class documentation. If `enzyme_quadrature` is false,
  /// the quadrature rule is held fixed during differentiation. If true, the
  /// derivative includes the geometry-dependent quadrature construction.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @param[out] dgt1_dx Gradient of the first weighted gap in pair DOF order.
  /// @param[out] dgt2_dx Gradient of the second weighted gap in pair DOF order.
  void grad_gtilde( CoordinateView coordinates, RealT dgt1_dx[8], RealT dgt2_dx[8] ) const;

  /// Compute first derivatives of the nodal tributary areas.
  ///
  /// `dA1_dx` and `dA2_dx` each have length 8 and store derivatives of the
  /// two nodal tributary area contributions with respect to the endpoint
  /// coordinate vector described in the class documentation. If
  /// `enzyme_quadrature` is false, the quadrature rule is held fixed during
  /// differentiation. If true, the derivative includes the geometry-dependent
  /// quadrature construction.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @param[out] dA1_dx Gradient of the first tributary area in pair DOF order.
  /// @param[out] dA2_dx Gradient of the second tributary area in pair DOF order.
  void grad_trib_area( CoordinateView coordinates, RealT dA1_dx[8], RealT dA2_dx[8] ) const;

  /// Compute second derivatives of the nodal smoothed gap integrals.
  ///
  /// `H1` and `H2` each have length 64 and store flattened 8 by 8 Hessian
  /// matrices for the two nodal gap integrals. Entries use row-major indexing:
  /// `H[row * 8 + col]`. If `enzyme_quadrature` is false, the quadrature rule
  /// is held fixed during differentiation. If true, the derivative includes the
  /// geometry-dependent quadrature construction.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @param[out] dgt1_dx Row-major Hessian of the first weighted gap.
  /// @param[out] dgt2_dx Row-major Hessian of the second weighted gap.
  void d2_g2tilde( CoordinateView coordinates, RealT dgt1_dx[64], RealT dgt2_dx[64] ) const;

  /// Compute second derivatives of the nodal tributary areas.
  ///
  /// `d2A1` and `d2A2` each have length 64 and store flattened 8 by 8 Hessian
  /// matrices for the two nodal tributary area contributions. Entries use
  /// row-major indexing: `H[row * 8 + col]`. If `enzyme_quadrature` is false,
  /// the quadrature rule is held fixed during differentiation. If true, the
  /// derivative includes the geometry-dependent quadrature construction.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @param[out] dgt1_dx Row-major Hessian of the first tributary area.
  /// @param[out] dgt2_dx Row-major Hessian of the second tributary area.
  void compute_d2A_d2u( CoordinateView coordinates, RealT dgt1_dx[64], RealT dgt2_dx[64] ) const;

  /// Compute local energy, force, and stiffness for quadrature-point penalty enforcement.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @return Owned energy, activation, residual, and Jacobian; inactive or zero-measure pairs give zeros.
  QuadraturePointPenaltyData compute_quadrature_point_penalty_data( CoordinateView coordinates ) const;

  /// Evaluate only the local quadrature-point penalty energy.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @return Penalty energy, including zero for inactive or zero-measure pairs.
  RealT compute_quadrature_point_penalty_energy( CoordinateView coordinates ) const;

  /// Evaluate and return the two nodal smoothed gap integrals.
  ///
  /// This is a convenience wrapper for obtaining only the gap integral
  /// quantities without also returning the tributary areas for the Finite
  /// difference test.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @return First and second weighted nodal gap integrals.
  std::pair<RealT, RealT> eval_gtilde( CoordinateView coordinates ) const;

  /// Validate first derivatives of the smoothed gap integrals using finite differences.
  ///
  /// This routine perturbs the endpoint coordinates of the interface pair and
  /// compares finite-difference approximations against the Enzyme-computed
  /// gradients. The perturbation size is controlled by `epsilon`.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @param epsilon Positive central-difference perturbation of each pair coordinate.
  /// @return Finite-difference and analytical gradients in pair coordinate order.
  FiniteDiffResult validate_g_tilde( CoordinateView coordinates, RealT epsilon = 1e-7 ) const;

  /// Evaluate the two nodal smoothed gap integrals using a fixed quadrature rule.
  ///
  /// This is used for derivative verification when the quadrature points and
  /// weights should remain fixed under coordinate perturbations. Holding the
  /// quadrature fixed isolates derivatives of the gap kernel from derivatives
  /// of the geometry-dependent quadrature construction.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @param qp_fixed Integration-surface coordinates and weights held fixed under perturbations.
  /// @return Weighted nodal gaps with fixed integration-surface coordinates and reference weights.
  std::pair<RealT, RealT> eval_gtilde_fixed_qp( CoordinateView coordinates,
                                                const IntegrationSurfaceRule& qp_fixed ) const;

  /// Validate second derivatives of the smoothed gap integrals using finite differences.
  ///
  /// This routine compares Enzyme-computed Hessians against finite-difference
  /// approximations of the first derivatives. The perturbation size is
  /// controlled by `epsilon`.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @param epsilon Positive central-difference perturbation of each pair coordinate.
  /// @return Row-major finite-difference and analytical Hessians in pair coordinate order.
  FiniteDiffResult validate_hessian( CoordinateView coordinates, RealT epsilon = 1e-7 ) const;

 private:
  /// Contact parameters controlling penalty stiffness and derivative behavior.
  ContactParams p_;
  /// Generator owned by the adapter or caller; no mesh views or generated quadrature are retained here.
  const QuadratureRule& quadrature_rule_;

  /// Generate an integration-surface rule for frozen-quadrature derivatives from the supplied coordinates.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @return Frozen integration-surface points and reference weights, including an empty rule for zero measure.
  IntegrationSurfaceRule generateIntegrationSurfaceRule( CoordinateView coordinates ) const;

  /// Compute smoothed gap gradients while holding the quadrature rule fixed.
  ///
  /// This is used by finite-difference verification routines to isolate
  /// derivatives of the gap kernel from derivatives of the quadrature rule.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @param qp_fixed Integration-surface coordinates and weights held fixed.
  /// @param[out] dgt1_dx First weighted-gap gradient.
  /// @param[out] dgt2_dx Second weighted-gap gradient.
  void grad_gtilde_with_qp( CoordinateView coordinates, const IntegrationSurfaceRule& qp_fixed, RealT dgt1_dx[8],
                            RealT dgt2_dx[8] ) const;

  /// Evaluate the signed normal gap at a local coordinate on edge A.
  ///
  /// The point on edge A is projected onto edge B, and the gap is evaluated
  /// using the normal from edge B together with the normal-alignment factor.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @param xiA Parametric coordinate on the first element.
  /// @return Alignment-corrected signed normal distance minus the residual gap.
  RealT compute_weighted_normal_gap( CoordinateView coordinates, RealT xiA ) const;

  /// Assemble nodal gap and tributary area data for one interface pair.
  ///
  /// This computes the two smoothed nodal gap integrals and the two corresponding
  /// tributary area contributions used to evaluate pressures, forces, energy,
  /// and stiffness terms.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @return Weighted nodal gaps and tributary areas; valid zero-measure rules give zeros.
  NodalContactData compute_nodal_contact_data( CoordinateView coordinates ) const;
};

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
