// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Adapt local integral expressions to scalar-energy, residual, and nodal-integral derivative interfaces.
 */
#ifndef SRC_TRIBOL_PHYSICS_PAIRDEFINITION_HPP_
#define SRC_TRIBOL_PHYSICS_PAIRDEFINITION_HPP_

#include "tribol/integ/PairIntegral.hpp"

namespace tribol {
/** @brief Primal pair energy, activation, and quadrature status. */
struct PairEnergyValue {
  RealT energy{ 0.0 };                   ///< Integrated pair energy; discard when quadrature fails.
  bool has_active_points{ false };       ///< Whether any quadrature point uses an active branch.
  QuadratureResult quadrature_result{};  ///< Generation status and point count; valid empty rules give zero energy.
};

/**
 * @brief Wrap a concise quadrature-point energy callable into a pair definition consumed by EnergyPairEvaluator.
 * @tparam QuadratureRule Element-pair quadrature generator.
 * @tparam EnergyFunctor Callable with operator()(integration_rule, penalty).
 */
template <typename QuadratureRule, typename EnergyFunctor>
class PairEnergyDefinition {
 public:
  using Physics = EnergyFunctor;         ///< Owned energy expression configuration.
  using Quadrature = QuadratureRule;     ///< Pair quadrature configuration accepted by this definition.
  using PrimalResult = PairEnergyValue;  ///< Primal energy, activation, and quadrature status.
  static constexpr IndexT dof_count = PairIntegrationRule<QuadratureRule>::dof_count;  ///< Pair coordinate DOFs.

  /**
   * @brief Store the integration rule and fixed penalty inputs.
   * @param quadrature_rule Pair quadrature generator.
   * @param penalty Penalty stiffness.
   * @param physics Owned energy expression, including any gap configuration.
   */
  TRIBOL_HOST_DEVICE PairEnergyDefinition( QuadratureRule quadrature_rule, RealT penalty, Physics physics = {} )
      : quadrature_rule_( quadrature_rule ), penalty_( penalty ), physics_( physics )
  {
  }

  /**
   * @brief Calculate energy and activation without consuming failed quadrature.
   * @param coordinates Endpoints ordered by element, node, and spatial component.
   * @return Energy, activation, and quadrature status.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairEnergyValue evaluate( MdSpan<const RealT, dof_count> coordinates ) const
  {
    const PairIntegrationRule<QuadratureRule> integration_rule( coordinates, quadrature_rule_ );
    if ( integration_rule.quadrature_result.status != QuadratureStatus::Success )
      return { 0.0, false, integration_rule.quadrature_result };
    const RealT value = physics_( integration_rule, penalty_ );
    return { value, integration_rule.has_active_points, integration_rule.quadrature_result };
  }

  /**
   * @brief Return the scalar energy consumed by EnergyPairEvaluator.
   * @param coordinates Current pair coordinates.
   * @return Integrated pair energy, including zero for valid empty or inactive rules.
   * @throws std::invalid_argument Pair geometry is invalid.
   * @throws std::length_error Quadrature capacity is insufficient.
   */
  TRIBOL_ENZYME_INLINE RealT energy( MdSpan<const RealT, dof_count> coordinates ) const
  {
    const auto value = evaluate( coordinates );
    if ( value.quadrature_result.status != QuadratureStatus::Success )
      reportQuadratureFailure( value.quadrature_result.status );
    return value.energy;
  }

 private:
  QuadratureRule quadrature_rule_;  ///< Owned pair quadrature configuration.
  RealT penalty_;                   ///< Inactive penalty stiffness.
  Physics physics_;                 ///< Owned mathematical expression and gap configuration.
};

/**
 * @brief Own an integrated pair residual with its quadrature status and branch activation.
 * @tparam DofCount Number of pair coordinate and residual entries.
 */
template <IndexT DofCount>
struct PairResidualValue {
  static constexpr IndexT coordinate_count = DofCount;  ///< Coordinate and residual entries.
  MdArray<RealT, DofCount> residual{};                  ///< Residual in element/node/component order.
  QuadratureResult quadrature_result{};                 ///< Generation status; discard residual on failure.
  bool has_active_points{ false };                      ///< Whether any point uses an active branch, including at zero.
};

/**
 * @brief Wrap a residual expression using the same pair integration context as scalar energy definitions.
 * @tparam QuadratureRule Element-pair quadrature generator.
 * @tparam ResidualFunctor Callable taking (integration_rule, penalty) and returning the pair residual.
 */
template <typename QuadratureRule, typename ResidualFunctor>
class PairResidualDefinition {
 public:
  using Physics = ResidualFunctor;    ///< Owned residual expression configuration.
  using Quadrature = QuadratureRule;  ///< Pair quadrature configuration.
  static constexpr IndexT dof_count = PairIntegrationRule<QuadratureRule>::dof_count;  ///< Pair coordinate count.
  using PrimalResult = PairResidualValue<dof_count>;  ///< Residual, quadrature status, and activation.
  /**
   * @brief Own the integration configuration and inactive scalar inputs.
   * @param quadrature_rule Pair generator.
   * @param penalty Fixed effective stiffness.
   * @param physics Owned residual expression, including any gap configuration.
   */
  TRIBOL_HOST_DEVICE PairResidualDefinition( QuadratureRule quadrature_rule, RealT penalty, Physics physics = {} )
      : quadrature_rule_( quadrature_rule ), penalty_( penalty ), physics_( physics )
  {
  }
  /**
   * @brief Evaluate the residual expression using quadrature generated from current coordinates.
   * @param coordinates Current pair coordinates in element/node/component order.
   * @return Residual, activation, and status. Failed quadrature returns zero residual; valid empty rules also return
   * zero.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PrimalResult evaluate( MdSpan<const RealT, dof_count> coordinates ) const
  {
    const PairIntegrationRule<QuadratureRule> rule( coordinates, quadrature_rule_ );
    if ( rule.quadrature_result.status != QuadratureStatus::Success ) return { {}, rule.quadrature_result, false };
    const auto residual = physics_( rule, penalty_ );
    return { residual, rule.quadrature_result, rule.has_active_points };
  }
  /**
   * @brief Expose the residual to ForcePairEvaluator.
   * @param coordinates Current pair coordinates.
   * @return Residual, including zero for a valid empty or inactive rule.
   * @throws std::invalid_argument Geometry is invalid.
   * @throws std::length_error Quadrature capacity is insufficient.
   */
  TRIBOL_ENZYME_INLINE MdArray<RealT, dof_count> residual( MdSpan<const RealT, dof_count> coordinates ) const
  {
    const auto value = evaluate( coordinates );
    if ( value.quadrature_result.status != QuadratureStatus::Success )
      reportQuadratureFailure( value.quadrature_result.status );
    return value.residual;
  }

 private:
  QuadratureRule quadrature_rule_;  ///< Owned pair mapping and rule configuration.
  RealT penalty_;                   ///< Inactive stiffness.
  Physics physics_;                 ///< Owned mathematical expression and gap configuration.
};

/**
 * @brief Name the nodal integrals calculated by a local expression before any energy contraction or activation.
 * @tparam NodeCount Finite-element nodes on mesh1's dual element.
 */
template <IndexT NodeCount>
struct NodalPairIntegrals {
  MdArray<RealT, NodeCount> weighted_gap{};  ///< Basis-weighted gaps in element-node order.
  MdArray<RealT, NodeCount> area{};          ///< Basis-weighted integration measures in the same order.
};

/**
 * @brief Weight named nodal integrals for differentiation of their coordinate Hessians.
 * @tparam Definition Nodal pair definition providing evaluate(), values(), node_count, and dof_count.
 */
template <typename Definition>
class WeightedNodalPairDefinition {
 public:
  static constexpr IndexT dof_count = Definition::dof_count;  ///< Active coordinate count.
  /**
   * @brief Own an integral definition and fixed nodal weights.
   * @param definition Primal integral expression and quadrature configuration.
   * @param gap_weights Outer energy partials with respect to each weighted gap.
   * @param area_weights Outer energy partials with respect to each area.
   */
  TRIBOL_HOST_DEVICE WeightedNodalPairDefinition( Definition definition,
                                                  MdArray<RealT, Definition::node_count> gap_weights,
                                                  MdArray<RealT, Definition::node_count> area_weights )
      : definition_( definition ), gap_weights_( gap_weights ), area_weights_( area_weights )
  {
  }
  /**
   * @brief Contract freshly generated integrals with fixed weights, preserving geometry and quadrature derivatives.
   * @param coordinates Active pair coordinates.
   * @return Weighted sum of gap and area integrals; valid empty rules return zero.
   * @throws std::invalid_argument Geometry is invalid.
   * @throws std::length_error Quadrature capacity is insufficient.
   */
  TRIBOL_ENZYME_INLINE RealT energy( MdSpan<const RealT, dof_count> coordinates ) const
  {
    constexpr IndexT count = Definition::node_count;
    const auto values = definition_.values( coordinates );
    return dot( gap_weights_, MdSpan<const RealT, count>{ values.data(), count } ) +
           dot( area_weights_, MdSpan<const RealT, count>{ values.data() + count, count } );
  }

 private:
  Definition definition_;                                ///< Owned generator and primal integral expression.
  MdArray<RealT, Definition::node_count> gap_weights_;   ///< Inactive gap weights.
  MdArray<RealT, Definition::node_count> area_weights_;  ///< Inactive area weights.
};

/**
 * @brief Evaluate named nodal integral expressions for their coordinate derivatives and outer energy contractions.
 * @tparam QuadratureRule Element-pair quadrature generator.
 * @tparam NodalFunctor Provides node_count and integrals(integration_rule), returning NodalPairIntegrals.
 */
template <typename QuadratureRule, typename NodalFunctor>
class NodalPairDefinition {
 public:
  using Quadrature = QuadratureRule;                              ///< Pair generator configuration.
  using Physics = NodalFunctor;                                   ///< Integral and assembled nodal energy expressions.
  static constexpr IndexT node_count = NodalFunctor::node_count;  ///< Dual finite-element nodes per element.
  static constexpr IndexT value_count = nodal_integral_count * node_count;  ///< Packed gap and area integrals.
  static constexpr IndexT element_dof_count = QuadratureRule::FirstGeometry::Coords::size();  ///< First-element DOFs.
  static constexpr IndexT dof_count = PairIntegrationRule<QuadratureRule>::dof_count;  ///< Pair coordinate count.
  using Coordinates = MdArray<RealT, dof_count>;                        ///< Owning packed pair coordinates.
  using CoordinateView = MdSpan<const RealT, dof_count>;                ///< Borrowed packed coordinates.
  using PrimalResult = PairIntegralValue<MdArray<RealT, value_count>>;  ///< Packed values, activation, and status.
  /**
   * @brief Own the pair generator and nodal integral expression.
   * @param quadrature_rule Pair rule and mapping.
   * @param physics Owned integral and energy expression configuration.
   */
  TRIBOL_HOST_DEVICE NodalPairDefinition( QuadratureRule quadrature_rule, Physics physics = {} )
      : quadrature_rule_( quadrature_rule ), physics_( physics )
  {
  }
  /**
   * @brief Integrate gap and area with shared local quadrature and pack their named outputs for VectorPairEvaluator.
   * @param coordinates Current pair coordinates in element/node/component order.
   * @return Weighted gaps followed by areas, positive-area activation, and status. Failed/empty rules return zero
   * values.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PrimalResult evaluate( CoordinateView coordinates ) const
  {
    const PairIntegrationRule<QuadratureRule> rule( coordinates, quadrature_rule_ );
    if ( rule.quadrature_result.status != QuadratureStatus::Success ) return { {}, false, rule.quadrature_result };
    const auto integrals = physics_.integrals( rule );
    MdArray<RealT, value_count> values;
    bool active = false;
    for ( IndexT node = 0; node < node_count; ++node ) {
      values[node] = integrals.weighted_gap[node];
      values[node_count + node] = integrals.area[node];
      active = active || integrals.area[node] > 0.0;
    }
    return { values, active, rule.quadrature_result };
  }
  /**
   * @brief Expose the packed integrals for their coordinate gradients and Hessians.
   * @param coordinates Active pair coordinates.
   * @return Owned gap integrals followed by areas; valid empty rules return zero values.
   * @throws std::invalid_argument Geometry is invalid.
   * @throws std::length_error Quadrature capacity is insufficient.
   */
  TRIBOL_ENZYME_INLINE MdArray<RealT, value_count> values( CoordinateView coordinates ) const
  {
    const auto result = evaluate( coordinates );
    if ( result.quadrature_result.status != QuadratureStatus::Success )
      reportQuadratureFailure( result.quadrature_result.status );
    return result.value;
  }
  /** @brief Read the configured nodal expression. @return Borrowed reference valid while this definition is alive. */
  TRIBOL_HOST_DEVICE const Physics& physics() const& { return physics_; }
  /** @brief Prevent borrowing expression configuration from a temporary definition. */
  const Physics& physics() const&& = delete;
  /**
   * @brief Bind outer nodal energy partials for the inner coordinate Hessian calculation.
   * @param gap_weights Fixed weights on gap integrals.
   * @param area_weights Fixed weights on area integrals; zero by default for LM energy.
   * @return Scalar definition consumed by EnergyPairEvaluator.
   */
  TRIBOL_HOST_DEVICE auto withWeights( MdArray<RealT, node_count> gap_weights,
                                       MdArray<RealT, node_count> area_weights = {} ) const
  {
    return WeightedNodalPairDefinition<NodalPairDefinition>{ *this, gap_weights, area_weights };
  }

 private:
  QuadratureRule quadrature_rule_;  ///< Owned pair quadrature configuration.
  Physics physics_;                 ///< Owned mathematical expression and gap configuration.
};
}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_PAIRDEFINITION_HPP_
