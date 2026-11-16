// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Quadrature-point Energy Mortar pair energy and contact formulation. */
#ifndef SRC_TRIBOL_PHYSICS_QUADRATUREPOINTENERGYMORTAR_HPP_
#define SRC_TRIBOL_PHYSICS_QUADRATUREPOINTENERGYMORTAR_HPP_

#include "tribol/config.hpp"
#include "tribol/enforcement/PenaltyEnergy.hpp"
#include "tribol/geom/OpposingNormalGap.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"
#include "tribol/physics/PairIntegrals.hpp"

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
#include "tribol/assembly/PairEvaluationAssembly.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/search/PairListBuilder.hpp"
#include <utility>
#endif

namespace tribol {

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )

/**
 * @brief Quadrature-point Energy Mortar with energy, force, and force-Jacobian assembly.
 *
 * Uses PairEnergyIntegral with PenaltyEnergyDensity; forces and their coordinate Jacobian are energy derivatives.
 * @tparam Exec Concrete execution mode for search, evaluation, and assembly kernels.
 * @tparam PairRuleGenerator Generates quadrature points and physical weights for the current edge pair.
 * @tparam GapDefinition Geometric or offset alignment-corrected gap.
 */
template <ExecutionMode Exec, typename PairRuleGenerator, typename GapDefinition = OpposingNormalGap>
class QuadraturePointEnergyMortar final : public ContactFormulation {
 public:
  /**
   * @brief Construct the penalty energy integral, own the search, and reference vector and matrix transfer data.
   * @param mesh_data Displacement transfer data; must outlive this formulation.
   * @param jacobian_data Coordinate-matrix transfer data; must outlive this formulation.
   * @param rule_generator Configured pair quadrature generator, retained by value.
   * @param penalty Fixed effective penalty stiffness.
   * @param gap Configured alignment-corrected gap, retained by value.
   * @param pair_builder Owned pair search.
   */
  QuadraturePointEnergyMortar( const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data,
                               PairRuleGenerator rule_generator, RealT penalty, GapDefinition gap,
                               PairListBuilder pair_builder )
      : mesh_data_( mesh_data ),
        jacobian_data_( jacobian_data ),
        pair_integral_( std::move( rule_generator ), { penalty, std::move( gap ) } ),
        pair_builder_( std::move( pair_builder ) )
  {
  }

  /**
   * @brief Evaluate pair energies and assemble energy, force, and coordinate Jacobian.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned energy, force, and df_dx; other outputs are absent.
   */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override
  {
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    return evaluateAndAssemblePairEnergyJacobian<Exec>( pairs, mesh1, mesh2, mesh_data_, jacobian_data_,
                                                        pair_integral_ );
  }

 private:
  const MfemMeshData& mesh_data_;          ///< Displacement transfer data referenced for this formulation's lifetime.
  const MfemJacobianData& jacobian_data_;  ///< Matrix transfer data referenced for this formulation's lifetime.
  const PairEnergyIntegral<PairRuleGenerator, PenaltyEnergyDensity<GapDefinition>>
      pair_integral_;             ///< Integrates the fixed compressive penalty energy density.
  PairListBuilder pair_builder_;  ///< Owned pair search.
};

#endif

}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_QUADRATUREPOINTENERGYMORTAR_HPP_
