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
#include "tribol/integ/QuadratureCommon.hpp"
#include "tribol/physics/PairEvaluation.hpp"
#include "tribol/physics/PairMethodInputs.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MfemData.hpp"

namespace tribol {

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
 * @tparam PairDefinition Local definition with Physics, Quadrature, PrimalResult, dof_count, and evaluate().
 * @tparam InputProvider Reads current fixed scalar parameters through prepare().
 * @tparam HasEnergy Select scalar energy differentiation or direct residual differentiation.
 */
template <typename PairDefinition, typename InputProvider, bool HasEnergy>
class PairMethod {
 public:
  using Definition = PairDefinition;                         ///< Local physics definition.
  using Physics = typename Definition::Physics;              ///< Mathematical expression owning its gap configuration.
  using Quadrature = typename Definition::Quadrature;        ///< Pair quadrature configuration.
  using FirstGeometry = typename Quadrature::FirstGeometry;  ///< First-element coordinate layout.
  using SecondGeometry = typename Quadrature::SecondGeometry;  ///< Second-element coordinate layout.
  using PrimalResult = typename Definition::PrimalResult;      ///< Primal value, quadrature status, and activation.
  using Derivatives = std::conditional_t<HasEnergy, PairEnergyDerivatives<Definition::dof_count>,
                                         PairResidualJacobian<Definition::dof_count>>;  ///< Owned derivatives.
  static constexpr bool is_nodal = false;        ///< Uses parent-only pair assembly.
  static constexpr bool has_energy = HasEnergy;  ///< Whether assembled output includes energy.

  /**
   * @brief Own the rule, mathematical expression, and current-penalty reader; retain no geometry or calculated fields.
   * @param quadrature_rule Owned pair quadrature configuration, supplied explicitly for either adapter.
   * @param physics Owned mathematical expression and gap definition.
   * @param inputs Owned provider, which reads current values for every prepare() call.
   */
  PairMethod( Quadrature quadrature_rule, Physics physics, InputProvider inputs )
      : quadrature_rule_( std::move( quadrature_rule ) ),
        physics_( std::move( physics ) ),
        inputs_( std::move( inputs ) )
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
    const RealT penalty = inputs_.prepare( mesh_data, mesh1, mesh2 );
    return Definition{ quadrature_rule_, penalty, physics_ };
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
  Physics physics_;             ///< Owned gap configuration and method inputs.
  InputProvider inputs_;        ///< Owned reader of current fixed parameters; retains no mesh views.
};
}  // namespace detail

/**
 * @brief Configure energy evaluated from assembled element-pair integrals.
 *
 * Element-pair quadrature produces the integrals before assembly. This does not imply node-to-segment contact.
 * The current implementation uses nodal weighted-gap and area fields for Energy Mortar, with penalty or LM enforcement.
 * @tparam NodalDefinition Local definition with Physics, Quadrature, PrimalResult, node_count, value_count, dof_count,
 * evaluate(), values(), and withWeights().
 * @tparam InputProvider Reads current fixed penalty parameters when penalty enforcement is enabled.
 */
template <typename NodalDefinition, typename InputProvider>
class AssembledEnergyMethod {
 public:
  using Definition = NodalDefinition;                        ///< Local nodal physics definition.
  using Physics = typename Definition::Physics;              ///< Mathematical expression owning its gap configuration.
  using Quadrature = typename Definition::Quadrature;        ///< Pair quadrature configuration.
  using FirstGeometry = typename Quadrature::FirstGeometry;  ///< First-element coordinate layout.
  using SecondGeometry = typename Quadrature::SecondGeometry;  ///< Second-element coordinate layout.
  using PrimalResult = typename Definition::PrimalResult;      ///< Nodal integrals, activation, and quadrature status.
  static constexpr bool is_nodal = true;                       ///< Assembles integral values on the dual submesh.
  static constexpr bool has_energy = true;                     ///< Assembled output includes contact energy.
  static constexpr RealT area_tolerance = 1.0e-14;             ///< Minimum tributary area for gap normalization.

  /**
   * @brief Own the rule, mathematical expression, and current-penalty reader; retain no geometry or calculated fields.
   * @param quadrature_rule Owned pair quadrature configuration.
   * @param physics Owned nodal gap configuration.
   * @param inputs Owned provider, which reads current penalty values on each penalty force request.
   */
  AssembledEnergyMethod( Quadrature quadrature_rule, Physics physics, InputProvider inputs )
      : quadrature_rule_( std::move( quadrature_rule ) ),
        physics_( std::move( physics ) ),
        inputs_( std::move( inputs ) )
  {
  }

  /**
   * @brief Report whether nodal penalty enforcement is selected instead of Lagrange multipliers.
   * @return True for nodal penalty enforcement; false for nodal Lagrange multiplier enforcement.
   */
  bool usesPenalty() const noexcept { return inputs_.isEnabled(); }

  /**
   * @brief Validate mesh coordinate layouts and construct the nodal integral definition.
   * @param mesh_data Registered MFEM data, unused when preparing geometry-only integrals.
   * @param mesh1 Current first (non-mortar) mesh.
   * @param mesh2 Current second (mortar) mesh.
   * @return Definition owning the rule and mathematical expression; no mesh views are retained.
   * @throws std::invalid_argument Mesh dimensions or node counts do not match the pair geometries.
   */
  Definition prepare( [[maybe_unused]] const MfemMeshData& mesh_data, MeshData::Viewer mesh1,
                      MeshData::Viewer mesh2 ) const
  {
    detail::validatePairMesh<FirstGeometry>( mesh1 );
    detail::validatePairMesh<SecondGeometry>( mesh2 );
    return Definition{ quadrature_rule_, physics_ };
  }

  /**
   * @brief Read current penalty parameters when nodal penalty enforcement is enabled.
   * @param mesh_data Current registered MFEM data.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Combined penalty stiffness.
   */
  RealT preparePenalty( const MfemMeshData& mesh_data, MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
  {
    return inputs_.prepare( mesh_data, mesh1, mesh2 );
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
   * @tparam WeightedDefinition Scalar contraction returned by withWeights().
   * @param coordinates Current pair coordinates in element/node/component order.
   * @param definition Nodal definition with fixed nodal gap and area weights.
   * @return Owned pair energy, gradient, and Hessian.
   */
  template <typename WeightedDefinition>
  static auto differentiateWeightedEnergy( MdSpan<const RealT, Definition::dof_count> coordinates,
                                           const WeightedDefinition& definition )
  {
    return EnergyPairEvaluator<WeightedDefinition>::evaluate( coordinates, definition );
  }

 private:
  Quadrature quadrature_rule_;  ///< Owned mapping and integration-rule configuration.
  Physics physics_;             ///< Owned gap configuration and method inputs.
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
