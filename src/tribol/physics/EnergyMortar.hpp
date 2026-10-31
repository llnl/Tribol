// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTAR_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTAR_HPP_

/**
 * @file
 * @brief Coordinate-based nodal Energy Mortar integrals and derivatives.
 */

#include "tribol/config.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/MdArray.hpp"
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

/// Weighted nodal gaps and tributary areas for one linear-edge pair.
struct NodalContactData {
  static constexpr IndexT node_count = LinearEdgeGeometry::vertex_count;  ///< Nodes on the linear integration edge.
  MdArray<RealT, node_count> g_tilde{};  ///< Weighted gaps in first-element node order.
  MdArray<RealT, node_count> AI{};       ///< Corresponding nodal tributary areas.
};

/// Evaluates nodal Energy Mortar integrals and derivatives from coordinates of one element pair.
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

  /// Nodal integrals and coordinate gradients for one element pair.
  struct NodalContactQuantities : NodalContactData {
    MdArray<RealT, node_count, dof_count> gap_gradients{};   ///< Weighted-gap gradients [node][pair DOF].
    MdArray<RealT, node_count, dof_count> area_gradients{};  ///< Tributary-area gradients [node][pair DOF].
  };

  /// Nodal integral Hessians with coordinate rows and columns in pair DOF order.
  struct NodalContactHessians {
    MdArray<RealT, NodalContactData::node_count, dof_count, dof_count> gap_hessians{};   ///< Weighted-gap Hessians.
    MdArray<RealT, NodalContactData::node_count, dof_count, dof_count> area_hessians{};  ///< Tributary-area Hessians.
  };

  /// Calculate nodal integrals without calculating derivatives.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @return Owned weighted gaps and tributary areas; valid zero-measure rules give zeros.
  NodalContactData computeNodalValues( CoordinateView coordinates ) const;

  /// Calculate nodal integrals and their gradients using the configured quadrature derivative mode.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @return Owned integrals and gradients; valid zero-measure rules give zeros.
  NodalContactQuantities computeContactQuantities( CoordinateView coordinates ) const;

  /// Calculate nodal integral Hessians using the configured quadrature derivative mode.
  /// @param coordinates Current pair coordinates in element/node/component order; not retained.
  /// @param include_area Whether to calculate tributary-area Hessians; LM enforcement does not need them.
  /// @return Owned Hessians; area_hessians remain zero when include_area is false. Zero-measure rules give zeros.
  NodalContactHessians computeSecondDerivatives( CoordinateView coordinates, bool include_area = true ) const;

 private:
  /// Contact parameters supplying the residual gap and quadrature derivative mode.
  ContactParams p_;
  /// Generator owned by the adapter or caller; no mesh views or generated quadrature are retained here.
  const QuadratureRule& quadrature_rule_;
};

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_ENERGYMORTAR_HPP_
