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

#include "tribol/geom/LinearEdgeGeometry.hpp"
#include "tribol/integ/PairIntegral.hpp"
#include "tribol/physics/PairMethod.hpp"

namespace tribol {

/**
 * @brief Calculate the opposing-normal-weighted gap at one pair quadrature point.
 *
 * Mesh1 is the non-mortar integration side. The gap projects the separation from mesh1 to mesh2 onto mesh2's current
 * directed normal, scales by the opposing-normal alignment factor min(normal1 . normal2, 0), and subtracts the fixed
 * residual gap offset.
 */
struct OpposingNormalGap {
  RealT residual_gap{ 0.0 };  ///< Gap offset held constant during differentiation.

  /**
   * @brief Evaluate the normal gap at a pair quadrature point.
   * @tparam Point PairQuadraturePoint type with 1D linear-edge geometries.
   * @param point Current pair quadrature point.
   * @return Normal gap at the quadrature point.
   */
  template <typename Point>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT operator()( const Point& point ) const
  {
    const auto normal1 = point.geometry1.normal(), normal2 = point.geometry2.normal();
    const RealT cosine = normal1.dot( normal2 );
    const RealT alignment = cosine < 0.0 ? cosine : 0.0;
    return LinearEdgeGeometry::Vector( point.position1(), point.position2() ).dot( normal2 ) * alignment - residual_gap;
  }
};

/** @brief Define quadrature-point Energy Mortar penalty energy from OpposingNormalGap and the integration rule. */
struct EnergyMortarPenalty {
  /**
   * @brief Evaluate quadrature-point penalty energy over the supplied integration rule.
   * @tparam IntegrationRule Pair integration rule context.
   * @param integration_rule Pair integration rule.
   * @param penalty Penalty stiffness.
   * @param residual_gap Gap offset held fixed during differentiation.
   * @return Integrated quadrature-point penalty energy.
   */
  template <typename IntegrationRule>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT operator()( const IntegrationRule& integration_rule, RealT penalty,
                                                            RealT residual_gap ) const
  {
    const auto gap = min( OpposingNormalGap{ residual_gap }, 0.0 );
    return integral( 0.5 * penalty * gap * gap, integration_rule );
  }
};

/** @brief Define nodal Energy Mortar penalty and Lagrange multiplier energies from nodal basis and gap integrals. */
struct EnergyMortarNodal {
  /**
   * @brief Evaluate nodal penalty energy from nodal weighted-gap and tributary-area integrals.
   * @tparam IntegrationRule Pair or nodal integration rule context.
   * @param integration_rule Integration rule context.
   * @param penalty Penalty stiffness.
   * @param residual_gap Gap offset held fixed during differentiation.
   * @return Nodal penalty contact energy.
   */
  template <typename IntegrationRule>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT penaltyEnergy( const IntegrationRule& integration_rule, RealT penalty,
                                                               RealT residual_gap ) const
  {
    const auto gap = OpposingNormalGap{ residual_gap };
    const auto g_tilde = min( integral( nodal_basis * gap, integration_rule ), 0.0 );
    return 0.5 * penalty * sum( g_tilde, g_tilde, 1.0 / integral( nodal_basis, integration_rule ) );
  }

  /**
   * @brief Evaluate nodal Lagrange multiplier energy from nodal pressures and weighted-gap integrals.
   * @tparam IntegrationRule Pair or nodal integration rule context.
   * @tparam Pressure Nodal pressure array or span type.
   * @param integration_rule Integration rule context.
   * @param pressure Nodal Lagrange multipliers on mesh1's nodes.
   * @param residual_gap Gap offset held fixed during differentiation.
   * @return Nodal Lagrange multiplier contact energy.
   */
  template <typename IntegrationRule, typename Pressure>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT lagrangeMultiplierEnergy( const IntegrationRule& integration_rule,
                                                                          const Pressure& pressure,
                                                                          RealT residual_gap ) const
  {
    const auto gap = OpposingNormalGap{ residual_gap };
    const auto g_tilde = integral( nodal_basis * gap, integration_rule );
    return dot( pressure, g_tilde );
  }
};

/** @brief Primal pair energy, activation, and quadrature status. */
using EnergyMortarEnergyValue = PairEnergyValue;

/** @brief Quadrature-point Energy Mortar pair energy definition for a given pair quadrature rule. */
template <typename QuadratureRule>
using EnergyMortarEnergy = PairEnergyDefinition<QuadratureRule, EnergyMortarPenalty>;

/** @brief Nodal Energy Mortar pair definition for a given pair quadrature rule. */
template <typename QuadratureRule>
using EnergyMortarNodalEnergy = NodalPairDefinition<QuadratureRule, EnergyMortarNodal>;

/** @brief Nodal Energy Mortar penalty state energy definition. */
using NodalPenaltyEnergy = NodalPenaltyStateEnergy<EnergyMortarNodal>;

/** @brief Nodal Energy Mortar Lagrange multiplier state energy definition. */
using NodalLagrangeMultiplierEnergy = NodalLagrangeMultiplierStateEnergy<EnergyMortarNodal>;

}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_ENERGYMORTARENERGY_HPP_
