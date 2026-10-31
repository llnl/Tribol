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
#include "tribol/physics/PairEvaluation.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MfemData.hpp"

namespace tribol {
#ifdef BUILD_REDECOMP
namespace detail {
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
    validateMesh<FirstGeometry>( mesh1 );
    validateMesh<SecondGeometry>( mesh2 );
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
  /**
   * @brief Require the mesh to provide one coordinate row per node expected by its geometry.
   * @tparam Geometry Element geometry with fixed-size Coords and a spatial_dimension.
   * @param mesh Current mesh; no view is retained.
   */
  template <typename Geometry>
  static void validateMesh( MeshData::Viewer mesh )
  {
    if ( mesh.spatialDimension() != Geometry::spatial_dimension ||
         mesh.numberOfNodesPerElement() != Geometry::Coords::template extent<0>() ) {
      throw std::invalid_argument( "Mesh dimensions and node counts must match the pair definition's geometries." );
    }
  }
  Quadrature quadrature_rule_;  ///< Owned mapping and integration-rule configuration.
  RealT residual_gap_;          ///< Fixed gap offset supplied to the input provider.
  InputProvider inputs_;        ///< Owned reader of current fixed parameters; retains no mesh views.
};
}  // namespace detail

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
