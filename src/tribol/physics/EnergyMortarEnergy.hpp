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
#include "tribol/mesh/GlobalAD.hpp"

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

#ifdef BUILD_REDECOMP
  /**
   * @brief Define nodal penalty energy directly on distributed AD vectors.
   * @param weighted_gap Unclamped assembled gap with supplied coordinate derivatives.
   * @param area Assembled tributary area with supplied coordinate derivatives.
   * @param penalty Fixed penalty stiffness.
   * @param area_tolerance Unsupported area magnitude; reciprocal values and derivatives are zero there.
   * @return Global energy whose coordinate gradient and Hessian follow distributed operator rules.
   */
  ADScalar penaltyEnergy( const ADVector& weighted_gap, const ADVector& area, RealT penalty,
                          RealT area_tolerance = 1.0e-14 ) const
  {
    const auto g = min( weighted_gap, 0.0 );
    return 0.5 * penalty * sum( g, g, reciprocal( area, area_tolerance ) );
  }
  /**
   * @brief Contract distributed multipliers and weighted gaps on the host.
   * @param weighted_gap Assembled gap AD vector.
   * @param pressure Pressure AD vector; LM multipliers are held fixed during coordinate differentiation.
   * @return Global energy and its distributed derivative calculation.
   */
  ADScalar lagrangeMultiplierEnergy( const ADVector& weighted_gap, const ADVector& pressure ) const
  {
    return dot( pressure, weighted_gap );
  }
#endif

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
