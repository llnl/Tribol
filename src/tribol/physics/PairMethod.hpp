// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Prepare pair definitions and select energy or force derivatives independently of contact physics.
 */
#ifndef SRC_TRIBOL_PHYSICS_PAIRMETHOD_HPP_
#define SRC_TRIBOL_PHYSICS_PAIRMETHOD_HPP_

#include <stdexcept>
#include <type_traits>
#include <utility>
#include "tribol/integ/PairIntegral.hpp"
#include "tribol/integ/QuadratureCommon.hpp"
#include "tribol/physics/PairEvaluation.hpp"
#include "tribol/physics/PairMethodInputs.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MfemData.hpp"

namespace tribol {

/** @brief Primal pair energy, activation, and quadrature status. */
struct PairEnergyValue {
  RealT energy{ 0.0 };                   ///< Integrated pair energy; discard when quadrature fails.
  bool has_active_points{ false };       ///< Whether any quadrature point uses an active branch.
  QuadratureResult quadrature_result{};  ///< Generation status and point count; valid empty rules give zero energy.
};

namespace detail {

/**
 * @brief Pair integration rule context that records each nodal integral evaluated by a nodal energy expression.
 * @tparam QuadratureRule Element-pair quadrature generator.
 * @tparam NodeCount Nodes on the first (non-mortar) element.
 */
template <typename QuadratureRule, IndexT NodeCount>
struct NodalQuantityRule {
  using FirstGeometry = typename QuadratureRule::FirstGeometry;          ///< First element geometry type.
  using SecondGeometry = typename QuadratureRule::SecondGeometry;        ///< Second element geometry type.
  static constexpr IndexT first_dofs = FirstGeometry::Coords::size();    ///< First-element coordinate DOFs.
  static constexpr IndexT second_dofs = SecondGeometry::Coords::size();  ///< Second-element coordinate DOFs.
  static constexpr IndexT dof_count = first_dofs + second_dofs;          ///< Pair coordinate DOFs.
  static constexpr IndexT value_count = 2 * NodeCount;                   ///< Recorded nodal integral entries.

  FirstGeometry geometry1;                         ///< First current element geometry.
  SecondGeometry geometry2;                        ///< Second current element geometry.
  typename QuadratureRule::QuadratureData points;  ///< Generated pair quadrature points and weights.
  mutable MdArray<RealT, value_count> values{};    ///< Recorded nodal integrals in evaluation order.
  mutable IndexT slot{ 0 };                        ///< Next nodal integral slot to populate.
  QuadratureResult quadrature_result{};            ///< Generation status and point count.

  /**
   * @brief Construct both element geometries and generate pair quadrature once for the current pair coordinates.
   * @param coordinates Packed pair coordinates with element1 followed by element2.
   * @param quadrature_rule Configured pair quadrature generator.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE NodalQuantityRule( MdSpan<const RealT, dof_count> coordinates,
                                                             const QuadratureRule& quadrature_rule )
      : geometry1( { coordinates.data(), first_dofs } ),
        geometry2( { coordinates.data() + first_dofs, second_dofs } ),
        points( quadrature_rule.generate( geometry1, geometry2 ) ),
        quadrature_result{ points.result.status, points.result.required_point_count }
  {
  }

  /**
   * @brief Integrate a nodal-basis integrand on the pair and record its nodal values.
   * @tparam Integrand Point integrand returning MdArray<RealT, NodeCount>.
   * @param integrand Point integrand expression.
   * @return Integrated nodal array on the pair.
   */
  template <typename Integrand>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE MdArray<RealT, NodeCount> integrate( Integrand&& integrand ) const
  {
    const auto result = integratePair( geometry1, geometry2, points, std::forward<Integrand>( integrand ) );
    for ( IndexT i = 0; i < NodeCount; ++i ) values[slot * NodeCount + i] = result.value[i];
    ++slot;
    return result.value;
  }
};

/**
 * @brief Pair integration rule context that accumulates a nodal-weighted sum of pair integrals for Hessian evaluation.
 * @tparam QuadratureRule Element-pair quadrature generator.
 * @tparam NodeCount Nodes on the first (non-mortar) element.
 */
template <typename QuadratureRule, IndexT NodeCount>
struct WeightedPairRule {
  using FirstGeometry = typename QuadratureRule::FirstGeometry;          ///< First element geometry type.
  using SecondGeometry = typename QuadratureRule::SecondGeometry;        ///< Second element geometry type.
  static constexpr IndexT first_dofs = FirstGeometry::Coords::size();    ///< First-element coordinate DOFs.
  static constexpr IndexT second_dofs = SecondGeometry::Coords::size();  ///< Second-element coordinate DOFs.
  static constexpr IndexT dof_count = first_dofs + second_dofs;          ///< Pair coordinate DOFs.

  FirstGeometry geometry1;                         ///< First current element geometry.
  SecondGeometry geometry2;                        ///< Second current element geometry.
  typename QuadratureRule::QuadratureData points;  ///< Generated pair quadrature points and weights.
  MdArray<RealT, NodeCount> gap_weight;            ///< Nodal weights for the first (weighted-gap) integral.
  MdArray<RealT, NodeCount> area_weight;           ///< Nodal weights for the second (tributary-area) integral.
  mutable IndexT slot{ 0 };                        ///< Current integral index.
  mutable RealT weighted_energy{ 0.0 };            ///< Accumulated nodal-weighted pair energy.
  QuadratureResult quadrature_result{};            ///< Generation status and point count.

  /**
   * @brief Construct both element geometries, generate pair quadrature, and store nodal integral weights.
   * @param coordinates Packed pair coordinates with element1 followed by element2.
   * @param quadrature_rule Configured pair quadrature generator.
   * @param pressure Nodal weights for the weighted-gap integral.
   * @param area_w Nodal weights for the tributary-area integral.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE WeightedPairRule( MdSpan<const RealT, dof_count> coordinates,
                                                            const QuadratureRule& quadrature_rule,
                                                            MdArray<RealT, NodeCount> pressure,
                                                            MdArray<RealT, NodeCount> area_w )
      : geometry1( { coordinates.data(), first_dofs } ),
        geometry2( { coordinates.data() + first_dofs, second_dofs } ),
        points( quadrature_rule.generate( geometry1, geometry2 ) ),
        gap_weight( pressure ),
        area_weight( area_w ),
        quadrature_result{ points.result.status, points.result.required_point_count }
  {
  }

  /**
   * @brief Integrate a nodal-basis integrand on the pair and add its weighted dot product to weighted_energy.
   * @tparam Integrand Point integrand returning MdArray<RealT, NodeCount>.
   * @param integrand Point integrand expression.
   * @return Integrated nodal array on the pair.
   */
  template <typename Integrand>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE MdArray<RealT, NodeCount> integrate( Integrand&& integrand ) const
  {
    const auto result = integratePair( geometry1, geometry2, points, std::forward<Integrand>( integrand ) );
    weighted_energy += dot( slot == 0 ? gap_weight : area_weight, result.value );
    ++slot;
    return result.value;
  }
};

/**
 * @brief Single-node integration context that supplies assembled nodal integrals {weighted_gap, area} to a nodal
 * energy expression for Enzyme differentiation with respect to those assembled integrals.
 */
struct NodalStateRule {
  MdSpan<const RealT, 2> nodal_state;  ///< Assembled nodal state {weighted_gap, area}.
  mutable IndexT slot{ 0 };            ///< Next nodal integral component to read.

  /**
   * @brief Return the assembled nodal value corresponding to the current integral in the nodal energy expression.
   * @tparam Integrand Point integrand type (unused during nodal state evaluation).
   * @return Single-entry array holding the assembled nodal integral value.
   */
  template <typename Integrand>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE MdArray<RealT, 1> integrate( Integrand&& ) const
  {
    return MdArray<RealT, 1>{ nodal_state[slot++] };
  }
};

}  // namespace detail

/**
 * @brief Wrap a concise quadrature-point energy callable into a pair definition consumed by EnergyPairEvaluator.
 * @tparam QuadratureRule Element-pair quadrature generator.
 * @tparam EnergyFunctor Callable with operator()(integration_rule, penalty, residual_gap).
 */
template <typename QuadratureRule, typename EnergyFunctor>
class PairEnergyDefinition {
 public:
  using Quadrature = QuadratureRule;     ///< Pair quadrature configuration accepted by this definition.
  using PrimalResult = PairEnergyValue;  ///< Primal energy, activation, and quadrature status.
  static constexpr IndexT dof_count = PairIntegrationRule<QuadratureRule>::dof_count;  ///< Pair coordinate DOFs.

  /**
   * @brief Store the integration rule and fixed penalty inputs.
   * @param quadrature_rule Pair quadrature generator.
   * @param penalty Penalty stiffness.
   * @param residual_gap Gap offset held fixed during differentiation.
   */
  TRIBOL_HOST_DEVICE PairEnergyDefinition( QuadratureRule quadrature_rule, RealT penalty, RealT residual_gap )
      : quadrature_rule_( quadrature_rule ), penalty_( penalty ), residual_gap_( residual_gap )
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
    const RealT value = EnergyFunctor{}( integration_rule, penalty_, residual_gap_ );
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
  RealT residual_gap_;              ///< Inactive gap offset.
};

/**
 * @brief Wrap a concise nodal energy functor into a pair definition consumed by VectorPairEvaluator and
 * EnergyPairEvaluator.
 * @tparam QuadratureRule Element-pair quadrature generator.
 * @tparam NodalEnergyFunctor Functor providing penaltyEnergy(rule, penalty, residual_gap) and
 * lagrangeMultiplierEnergy(rule, pressure, residual_gap).
 */
template <typename QuadratureRule, typename NodalEnergyFunctor>
class NodalPairDefinition {
 public:
  using Quadrature = QuadratureRule;   ///< Pair quadrature configuration.
  using Physics = NodalEnergyFunctor;  ///< Underlying nodal energy functor.
  static constexpr IndexT node_count =
      QuadratureRule::FirstGeometry::vertex_count;       ///< Nodes on the first (non-mortar) element.
  static constexpr IndexT value_count = 2 * node_count;  ///< Weighted gaps followed by tributary areas.
  static constexpr IndexT element_dof_count =
      QuadratureRule::FirstGeometry::Coords::size();                                   ///< Coordinate DOFs per element.
  static constexpr IndexT dof_count = PairIntegrationRule<QuadratureRule>::dof_count;  ///< Pair coordinate DOFs.
  using Coordinates = MdArray<RealT, dof_count>;                        ///< Owning pair coordinates in DOF order.
  using CoordinateView = MdSpan<const RealT, dof_count>;                ///< Read-only pair coordinates.
  using PrimalResult = PairIntegralValue<MdArray<RealT, value_count>>;  ///< Nodal integrals, activation, and status.

  /**
   * @brief Store the integration rule, fixed gap offset, and optional nodal gap/area weights.
   * @param quadrature_rule Pair quadrature generator.
   * @param residual_gap Gap offset held fixed during differentiation.
   * @param pressure Nodal Lagrange multipliers or gap weights on mesh1's element nodes.
   * @param area_weight Optional nodal tributary-area weights on mesh1's element nodes.
   */
  TRIBOL_HOST_DEVICE NodalPairDefinition( QuadratureRule quadrature_rule, RealT residual_gap = 0.0,
                                          MdArray<RealT, node_count> pressure = {},
                                          MdArray<RealT, node_count> area_weight = {} )
      : quadrature_rule_( quadrature_rule ),
        residual_gap_( residual_gap ),
        pressure_( pressure ),
        area_weight_( area_weight )
  {
  }

  /**
   * @brief Return a copy of this definition bound to the given nodal gap and area weights.
   * @param pressure Nodal Lagrange multipliers or gap weights on mesh1's element nodes.
   * @param area_weight Optional nodal tributary-area weights on mesh1's element nodes.
   * @return Nodal definition with fixed nodal weights.
   */
  TRIBOL_HOST_DEVICE NodalPairDefinition withWeights( MdArray<RealT, node_count> pressure,
                                                      MdArray<RealT, node_count> area_weight = {} ) const
  {
    return { quadrature_rule_, residual_gap_, pressure, area_weight };
  }

  /**
   * @brief Calculate nodal weighted gaps and tributary areas without consuming failed quadrature.
   * @param coordinates Endpoints ordered by element, node, and spatial component.
   * @return Nodal integrals {g_tilde_0, g_tilde_1, A_0, A_1}, positive-area activation, and quadrature status.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PrimalResult evaluate( MdSpan<const RealT, dof_count> coordinates ) const
  {
    const detail::NodalQuantityRule<QuadratureRule, node_count> rule( coordinates, quadrature_rule_ );
    (void)NodalEnergyFunctor{}.penaltyEnergy( rule, 1.0, residual_gap_ );
    const bool active = rule.values[node_count] > 0.0 || rule.values[node_count + 1] > 0.0;
    return { rule.values, active, rule.quadrature_result };
  }

  /**
   * @brief Return nodal weighted gaps and tributary areas consumed by VectorPairEvaluator.
   * @param coordinates Current pair coordinates.
   * @return Owned array {g_tilde_0, g_tilde_1, A_0, A_1}.
   * @throws std::invalid_argument Pair geometry is invalid.
   * @throws std::length_error Quadrature capacity is insufficient.
   */
  TRIBOL_ENZYME_INLINE MdArray<RealT, value_count> values( MdSpan<const RealT, dof_count> coordinates ) const
  {
    const auto result = evaluate( coordinates );
    if ( result.quadrature_result.status != QuadratureStatus::Success )
      reportQuadratureFailure( result.quadrature_result.status );
    return result.value;
  }

  /**
   * @brief Return the pair nodal Lagrange multiplier energy for the supplied nodal pressures.
   * @param coordinates Current pair coordinates.
   * @param pressure Nodal Lagrange multipliers or gap weights on mesh1's element nodes.
   * @return Pair nodal contact energy.
   * @throws std::invalid_argument Pair geometry is invalid.
   * @throws std::length_error Quadrature capacity is insufficient.
   */
  TRIBOL_ENZYME_INLINE RealT energy( MdSpan<const RealT, dof_count> coordinates,
                                     MdSpan<const RealT, node_count> pressure ) const
  {
    const PairIntegrationRule<QuadratureRule> rule( coordinates, quadrature_rule_ );
    const RealT value = NodalEnergyFunctor{}.lagrangeMultiplierEnergy( rule, pressure, residual_gap_ );
    if ( rule.quadrature_result.status != QuadratureStatus::Success )
      reportQuadratureFailure( rule.quadrature_result.status );
    return value;
  }

  /**
   * @brief Return the nodal-weighted pair energy consumed by EnergyPairEvaluator.
   * @param coordinates Current pair coordinates.
   * @return Pair nodal energy weighted by pressure_ and area_weight_.
   * @throws std::invalid_argument Pair geometry is invalid.
   * @throws std::length_error Quadrature capacity is insufficient.
   */
  TRIBOL_ENZYME_INLINE RealT energy( MdSpan<const RealT, dof_count> coordinates ) const
  {
    if ( area_weight_[0] == 0.0 && area_weight_[1] == 0.0 ) {
      return energy( coordinates, pressure_.view() );
    }
    const detail::WeightedPairRule<QuadratureRule, node_count> rule( coordinates, quadrature_rule_, pressure_,
                                                                     area_weight_ );
    (void)NodalEnergyFunctor{}.penaltyEnergy( rule, 1.0, residual_gap_ );
    if ( rule.quadrature_result.status != QuadratureStatus::Success )
      reportQuadratureFailure( rule.quadrature_result.status );
    return rule.weighted_energy;
  }

 private:
  QuadratureRule quadrature_rule_;          ///< Owned pair quadrature configuration.
  RealT residual_gap_;                      ///< Inactive gap offset.
  MdArray<RealT, node_count> pressure_;     ///< Inactive nodal Lagrange multipliers or gap weights.
  MdArray<RealT, node_count> area_weight_;  ///< Inactive nodal tributary-area weights.
};

/**
 * @brief Evaluate a nodal energy functor's penalty energy on assembled nodal state {weighted_gap, area} for Enzyme
 * differentiation with respect to those assembled integrals.
 * @tparam NodalEnergyFunctor Nodal physics functor providing penaltyEnergy(rule, penalty, residual_gap).
 */
template <typename NodalEnergyFunctor>
struct NodalPenaltyStateEnergy {
  static constexpr IndexT dof_count = 2;  ///< Nodal state components: {weighted_gap, area}.
  RealT penalty{};                        ///< Inactive nodal penalty stiffness.
  RealT residual_gap{};                   ///< Inactive gap offset.

  /**
   * @brief Return nodal penalty energy evaluated at the assembled nodal state {weighted_gap, area}.
   * @param nodal_state Two-component span {weighted_gap, area}.
   * @return Nodal penalty energy at one node.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT energy( MdSpan<const RealT, dof_count> nodal_state ) const
  {
    return NodalEnergyFunctor{}.penaltyEnergy( detail::NodalStateRule{ nodal_state }, penalty, residual_gap );
  }
};

/**
 * @brief Evaluate a nodal energy functor's Lagrange multiplier energy on assembled nodal state {weighted_gap, area}
 * for Enzyme differentiation with respect to those assembled integrals.
 * @tparam NodalEnergyFunctor Nodal physics functor providing lagrangeMultiplierEnergy(rule, pressure, residual_gap).
 */
template <typename NodalEnergyFunctor>
struct NodalLagrangeMultiplierStateEnergy {
  static constexpr IndexT dof_count = 2;  ///< Nodal state components: {weighted_gap, area}.
  RealT multiplier{};                     ///< Inactive nodal Lagrange multiplier.
  RealT residual_gap{};                   ///< Inactive gap offset.

  /**
   * @brief Return nodal Lagrange multiplier energy evaluated at the assembled nodal state {weighted_gap, area}.
   * @param nodal_state Two-component span {weighted_gap, area}.
   * @return Nodal Lagrange multiplier energy at one node.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT energy( MdSpan<const RealT, dof_count> nodal_state ) const
  {
    return NodalEnergyFunctor{}.lagrangeMultiplierEnergy( detail::NodalStateRule{ nodal_state },
                                                          MdArray<RealT, 1>{ multiplier }, residual_gap );
  }
};

#ifdef BUILD_REDECOMP
namespace detail {
/**
 * @brief Require the mesh to provide one coordinate row per node expected by its geometry.
 * @tparam Geometry Element geometry with fixed-size Coords and a spatial_dimension.
 * @param mesh Current mesh; no view is retained.
 */
template <typename Geometry>
void validatePairMesh( MeshData::Viewer mesh )
{
  if ( mesh.spatialDimension() != Geometry::spatial_dimension ||
       mesh.numberOfNodesPerElement() != Geometry::Coords::template extent<0>() ) {
    throw std::invalid_argument( "Mesh dimensions and node counts must match the pair definition's geometries." );
  }
}

/**
 * @brief Shared preparation and derivative calls behind the public energy and force pair adapters.
 * @tparam PairDefinition Physics with Quadrature, PrimalResult, dof_count, and an evaluate() method.
 * @tparam InputProvider Reads current fixed scalar parameters through prepare().
 * @tparam HasEnergy Select scalar energy differentiation or direct residual differentiation.
 */
template <typename PairDefinition, typename InputProvider, bool HasEnergy>
class PairMethod {
 public:
  using Definition = PairDefinition;                           ///< Local physics definition.
  using Quadrature = typename Definition::Quadrature;          ///< Pair quadrature configuration.
  using FirstGeometry = typename Quadrature::FirstGeometry;    ///< First-element coordinate layout.
  using SecondGeometry = typename Quadrature::SecondGeometry;  ///< Second-element coordinate layout.
  using PrimalResult = typename Definition::PrimalResult;      ///< Primal value, quadrature status, and activation.
  using Derivatives = std::conditional_t<HasEnergy, PairEnergyDerivatives<Definition::dof_count>,
                                         PairResidualJacobian<Definition::dof_count>>;  ///< Owned derivatives.
  static constexpr bool is_nodal = false;        ///< Uses parent-only pair assembly.
  static constexpr bool has_energy = HasEnergy;  ///< Whether assembled output includes energy.

  /**
   * @brief Store rule, gap offset, and input-provider configuration; retain no current geometry or scalar fields.
   * @param quadrature_rule Owned pair quadrature configuration, supplied explicitly for either adapter.
   * @param residual_gap Fixed gap offset, validated according to the selected input provider.
   * @param inputs Owned provider, which reads current values for every prepare() call.
   */
  PairMethod( Quadrature quadrature_rule, RealT residual_gap, InputProvider inputs )
      : quadrature_rule_( std::move( quadrature_rule ) ), residual_gap_( residual_gap ), inputs_( std::move( inputs ) )
  {
  }

  /**
   * @brief Check coordinate layouts and construct the definition using current scalar parameters.
   * @param mesh_data Current registered MFEM data.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Definition owning the rule configuration and current scalar inputs; no mesh views are retained.
   * @throws std::invalid_argument Mesh dimensions/node counts or the provider's input requirements are invalid.
   */
  Definition prepare( const MfemMeshData& mesh_data, MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
  {
    validatePairMesh<FirstGeometry>( mesh1 );
    validatePairMesh<SecondGeometry>( mesh2 );
    const auto parameters = inputs_.prepare( mesh_data, mesh1, mesh2, residual_gap_ );
    return Definition{ quadrature_rule_, parameters.penalty, parameters.residual_gap };
  }

  /**
   * @brief Reuse the successful primal value while calculating coordinate derivatives.
   * @param coordinates Current pair coordinates in element/node/component order.
   * @param definition Prepared fixed configuration; current geometry must come from coordinates.
   * @param primal Successful, active evaluation at these same coordinates and inputs.
   * @return Owned residual and Jacobian, with energy only for the energy adapter. Evaluator specializations are
   * honored.
   */
  static Derivatives differentiate( MdSpan<const RealT, Definition::dof_count> coordinates,
                                    const Definition& definition, const PrimalResult& primal )
  {
    if constexpr ( HasEnergy ) {
      return EnergyPairEvaluator<Definition>::evaluate( coordinates, definition, primal.energy );
    } else {
      return ForcePairEvaluator<Definition>::evaluate( coordinates, definition, primal.residual.view() );
    }
  }

 private:
  Quadrature quadrature_rule_;  ///< Owned mapping and integration-rule configuration.
  RealT residual_gap_;          ///< Fixed gap offset supplied to the input provider.
  InputProvider inputs_;        ///< Owned reader of current fixed parameters; retains no mesh views.
};
}  // namespace detail

/**
 * @brief Prepare a nodal mortar definition and select nodal penalty or Lagrange multiplier derivatives.
 * @tparam NodalDefinition Nodal physics with Quadrature, PrimalResult, node_count, value_count, dof_count,
 * evaluate(), values(), and energy().
 * @tparam InputProvider Reads current fixed penalty parameters when penalty enforcement is enabled.
 */
template <typename NodalDefinition, typename InputProvider>
class NodalPairMethod {
 public:
  using Definition = NodalDefinition;                          ///< Local nodal physics definition.
  using Quadrature = typename Definition::Quadrature;          ///< Pair quadrature configuration.
  using FirstGeometry = typename Quadrature::FirstGeometry;    ///< First-element coordinate layout.
  using SecondGeometry = typename Quadrature::SecondGeometry;  ///< Second-element coordinate layout.
  using PrimalResult = typename Definition::PrimalResult;      ///< Nodal integrals, activation, and quadrature status.
  static constexpr bool is_nodal = true;                       ///< Uses nodal submesh assembly.
  static constexpr bool has_energy = true;                     ///< Assembled output includes contact energy.
  static constexpr RealT area_tolerance = 1.0e-14;             ///< Minimum tributary area for gap normalization.

  /**
   * @brief Store rule, gap offset, and input-provider configuration; retain no current geometry or scalar fields.
   * @param quadrature_rule Owned pair quadrature configuration.
   * @param residual_gap Fixed gap offset.
   * @param inputs Owned provider, which reads current penalty values on each penalty force request.
   */
  NodalPairMethod( Quadrature quadrature_rule, RealT residual_gap, InputProvider inputs )
      : quadrature_rule_( std::move( quadrature_rule ) ), residual_gap_( residual_gap ), inputs_( std::move( inputs ) )
  {
  }

  /**
   * @brief Report whether nodal penalty enforcement is selected instead of Lagrange multipliers.
   * @return True for nodal penalty enforcement; false for nodal Lagrange multiplier enforcement.
   */
  bool usesPenalty() const noexcept { return inputs_.isEnabled(); }

  /**
   * @brief Validate mesh coordinate layouts and construct the nodal integral definition.
   * @param mesh1 Current first (non-mortar) mesh.
   * @param mesh2 Current second (mortar) mesh.
   * @return Definition owning the rule configuration and residual gap; no mesh views are retained.
   * @throws std::invalid_argument Mesh dimensions or node counts do not match the pair geometries.
   */
  Definition prepare( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
  {
    detail::validatePairMesh<FirstGeometry>( mesh1 );
    detail::validatePairMesh<SecondGeometry>( mesh2 );
    return Definition{ quadrature_rule_, residual_gap_ };
  }

  /**
   * @brief Read current penalty parameters when nodal penalty enforcement is enabled.
   * @param mesh_data Current registered MFEM data.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Combined penalty stiffness and residual gap.
   */
  PairDefinitionParameters preparePenalty( const MfemMeshData& mesh_data, MeshData::Viewer mesh1,
                                           MeshData::Viewer mesh2 ) const
  {
    return inputs_.prepare( mesh_data, mesh1, mesh2, residual_gap_ );
  }

  /**
   * @brief Reuse successful primal nodal integrals while calculating their coordinate gradients.
   * @param coordinates Current pair coordinates in element/node/component order.
   * @param definition Prepared nodal definition.
   * @param primal Successful, active nodal integral evaluation at these same coordinates.
   * @return Owned nodal values and coordinate gradients.
   */
  static auto differentiateQuantities( MdSpan<const RealT, Definition::dof_count> coordinates,
                                       const Definition& definition, const PrimalResult& primal )
  {
    return VectorPairEvaluator<Definition>::evaluateGradients( coordinates, definition, primal.value.view() );
  }

  /**
   * @brief Calculate the coordinate Hessian of the nodal-weighted pair energy.
   * @param coordinates Current pair coordinates in element/node/component order.
   * @param definition Nodal definition with fixed nodal gap and area weights.
   * @return Owned pair energy, gradient, and Hessian.
   */
  static auto differentiateWeightedEnergy( MdSpan<const RealT, Definition::dof_count> coordinates,
                                           const Definition& definition )
  {
    return EnergyPairEvaluator<Definition>::evaluate( coordinates, definition );
  }

 private:
  Quadrature quadrature_rule_;  ///< Owned mapping and integration-rule configuration.
  RealT residual_gap_;          ///< Fixed gap offset.
  InputProvider inputs_;        ///< Owned reader of current fixed penalty parameters.
};

/**
 * @brief Prepare an energy definition and obtain its residual/Jacobian through EnergyPairEvaluator.
 * @tparam Definition Energy physics with Quadrature, PrimalResult, dof_count, evaluate(), and energy().
 * @tparam InputProvider Reads fixed parameters from current MFEM/mesh data through prepare().
 */
template <typename Definition, typename InputProvider>
using EnergyPairMethod = detail::PairMethod<Definition, InputProvider, true>;

/**
 * @brief Prepare a force definition and obtain its Jacobian through ForcePairEvaluator.
 * @tparam Definition Force physics with Quadrature, PrimalResult, dof_count, evaluate(), and residual().
 * @tparam InputProvider Reads fixed parameters from current MFEM/mesh data through prepare().
 */
template <typename Definition, typename InputProvider>
using ForcePairMethod = detail::PairMethod<Definition, InputProvider, false>;
#endif  // BUILD_REDECOMP
}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_PAIRMETHOD_HPP_
