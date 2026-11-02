// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Quadrature-point and nodal Energy Mortar gap and energy definitions.
 */
#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTARENERGY_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTARENERGY_HPP_

#include "tribol/geom/OpposingNormalGap.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/integ/PairIntegral.hpp"
#include "tribol/physics/PairDefinition.hpp"

namespace tribol {

/**
 * @brief Define quadrature-point Energy Mortar penalty energy using an owned gap definition.
 * @tparam GapDefinition Geometric or offset gap callable evaluated at each pair quadrature point.
 */
template <typename GapDefinition = OpposingNormalGap>
struct EnergyMortarPenalty {
  GapDefinition gap{};  ///< Owned gap configuration; no current geometry is retained.
  /**
   * @brief Evaluate quadrature-point penalty energy over the supplied integration rule.
   * @tparam IntegrationRule Pair integration rule context.
   * @param integration_rule Pair integration rule.
   * @param penalty Penalty stiffness.
   * @return Integrated quadrature-point penalty energy.
   */
  template <typename IntegrationRule>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT operator()( const IntegrationRule& integration_rule,
                                                            RealT penalty ) const
  {
    const auto compressed_gap = min( gap, 0.0 );
    return integral( 0.5 * penalty * compressed_gap * compressed_gap, integration_rule );
  }
};

/**
 * @brief Define nodal Energy Mortar integrals and energies using an owned gap definition.
 * @tparam GapDefinition Geometric or offset gap callable evaluated at each pair quadrature point.
 */
template <typename GapDefinition = OpposingNormalGap>
struct EnergyMortarNodal {
  GapDefinition gap{};                               ///< Owned gap configuration, used only by the primal gap integral.
  using Basis = NodalBasis1<EDGE, LINEAR_LAGRANGE>;  ///< Dual basis on the first linear edge.
  static constexpr IndexT node_count = Basis::node_count;  ///< Dual finite-element nodes on each element.
  /**
   * @brief Calculate gap and area directly from the same pair integration context.
   * @tparam IntegrationRule Current pair rule supporting integral().
   * @param integration_rule Current geometry and generated quadrature.
   * @return Named, unclamped weighted-gap and area integrals.
   */
  template <typename IntegrationRule>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE NodalPairIntegrals<node_count> integrals(
      const IntegrationRule& integration_rule ) const
  {
    return { integral( nodal_basis * gap, integration_rule ), integral( nodal_basis, integration_rule ) };
  }

  /**
   * @brief Define nodal penalty energy using either numerical arrays or named assembled integral expressions.
   * @tparam WeightedGap Nodal weighted-gap array or expression.
   * @tparam Area Nodal tributary-area array or expression.
   * @param weighted_gap Gap integrals before nodal activation.
   * @param area Corresponding tributary-area integrals.
   * @param penalty Fixed penalty stiffness.
   * @return Numerical energy for arrays, or a deferred energy for expressions.
   */
  template <typename WeightedGap, typename Area>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto penaltyEnergy( const WeightedGap& weighted_gap, const Area& area,
                                                              RealT penalty ) const
  {
    const auto g_tilde = min( weighted_gap, 0.0 );
    return 0.5 * penalty * sum( g_tilde, g_tilde, 1.0 / area );
  }

  /**
   * @brief Contract fixed pressure with either a numerical weighted gap or its assembled integral expression.
   * @tparam WeightedGap Nodal weighted-gap array or expression.
   * @tparam Pressure Independent nodal pressure array or field expression.
   * @param weighted_gap Assembled gap integrals.
   * @param pressure Fixed Lagrange multiplier values in matching DOF order.
   * @return Numerical or deferred nodal energy.
   */
  template <typename WeightedGap, typename Pressure>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto lagrangeMultiplierEnergy( const WeightedGap& weighted_gap,
                                                                         const Pressure& pressure ) const
  {
    return dot( pressure, weighted_gap );
  }
};

/** @brief Primal pair energy, activation, and quadrature status. */
using EnergyMortarEnergyValue = PairEnergyValue;

/**
 * @brief Quadrature-point Energy Mortar energy with a selected gap definition.
 * @tparam QuadratureRule Pair quadrature generator.
 * @tparam GapDefinition Geometric or offset gap type; defaults to the unshifted geometric gap.
 */
template <typename QuadratureRule, typename GapDefinition = OpposingNormalGap>
using EnergyMortarEnergy = PairEnergyDefinition<QuadratureRule, EnergyMortarPenalty<GapDefinition>>;

/**
 * @brief Nodal Energy Mortar integrals with a selected gap definition.
 * @tparam QuadratureRule Pair quadrature generator.
 * @tparam GapDefinition Geometric or offset gap type; defaults to the unshifted geometric gap.
 */
template <typename QuadratureRule, typename GapDefinition = OpposingNormalGap>
using EnergyMortarNodalEnergy = NodalPairDefinition<QuadratureRule, EnergyMortarNodal<GapDefinition>>;

}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_ENERGYMORTARENERGY_HPP_
