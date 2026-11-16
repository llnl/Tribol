// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Opt-in composed Common Plane contact formulation, separate from the legacy response path. */
#ifndef SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_

#include "tribol/config.hpp"
#include "tribol/enforcement/CommonPlanePenaltyTraction.hpp"
#include "tribol/geom/CommonPlaneEdgeMapping.hpp"
#include "tribol/geom/CommonPlaneGap.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"
#include "tribol/assembly/PairEvaluationAssembly.hpp"
#include "tribol/physics/PairIntegrals.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/search/PairListBuilder.hpp"
#include <utility>

namespace tribol {

#ifdef BUILD_REDECOMP
/**
 * @brief Common Plane pair search followed by residual-only evaluation and force assembly.
 *
 * Uses PairTractionIntegral with CommonPlanePenaltyTraction and linear Lagrange residual bases on both edges.
 * @tparam Exec Concrete execution mode for search, evaluation, and assembly kernels.
 * @tparam PairRuleGenerator Generates quadrature points and physical weights for the current edge pair.
 * @tparam GapDefinition Geometric or offset common-plane gap.
 */
template <ExecutionMode Exec, typename PairRuleGenerator, typename GapDefinition = CommonPlaneGap>
class CommonPlaneResidualFormulation final : public ContactFormulation {
 public:
  /**
   * @brief Construct the penalty traction integral, own the pair search, and reference displacement transfer data.
   * @param mesh_data Displacement transfer data; must outlive this formulation.
   * @param rule_generator Configured pair quadrature generator, retained by value.
   * @param penalty Fixed effective penalty stiffness.
   * @param gap Configured common-plane gap, retained by value.
   * @param pair_builder Owned pair search composition.
   */
  CommonPlaneResidualFormulation( const MfemMeshData& mesh_data, PairRuleGenerator rule_generator, RealT penalty,
                                  GapDefinition gap, PairListBuilder pair_builder )
      : mesh_data_( mesh_data ),
        pair_integral_( std::move( rule_generator ), { penalty, std::move( gap ) } ),
        pair_builder_( std::move( pair_builder ) )
  {
  }

  /**
   * @brief Compute the Common Plane force without calculating a Jacobian.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned force; df_dx and other outputs are absent.
   */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override
  {
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    return evaluateAndAssemblePairResidual<Exec>( pairs, mesh1, mesh2, mesh_data_, pair_integral_ );
  }

 private:
  const MfemMeshData& mesh_data_;  ///< Displacement transfer data referenced for this formulation's lifetime.
  const PairTractionIntegral<PairRuleGenerator, CommonPlanePenaltyTraction<GapDefinition>,
                             BasisFunction<EDGE, LINEAR_LAGRANGE>>
      pair_integral_;             ///< Integrates the fixed Common Plane penalty traction with linear edge bases.
  PairListBuilder pair_builder_;  ///< Owned search composition.
};

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Common Plane pair search followed by residual and coordinate-Jacobian evaluation and assembly.
 *
 * Uses PairTractionIntegral with CommonPlanePenaltyTraction and linear Lagrange residual bases on both edges.
 * @tparam Exec Concrete execution mode for search, evaluation, and assembly kernels.
 * @tparam PairRuleGenerator Generates quadrature points and physical weights for the current edge pair.
 * @tparam GapDefinition Geometric or offset common-plane gap.
 */
template <ExecutionMode Exec, typename PairRuleGenerator, typename GapDefinition = CommonPlaneGap>
class CommonPlaneResidualJacobianFormulation final : public ContactFormulation {
 public:
  /**
   * @brief Construct the penalty traction integral, own the pair search, and reference vector and matrix transfer data.
   * @param mesh_data Displacement transfer data; must outlive this formulation.
   * @param jacobian_data Coordinate-matrix transfer data; must outlive this formulation.
   * @param rule_generator Configured pair quadrature generator, retained by value.
   * @param penalty Fixed effective penalty stiffness.
   * @param gap Configured common-plane gap, retained by value.
   * @param pair_builder Owned pair search composition.
   */
  CommonPlaneResidualJacobianFormulation( const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data,
                                          PairRuleGenerator rule_generator, RealT penalty, GapDefinition gap,
                                          PairListBuilder pair_builder )
      : mesh_data_( mesh_data ),
        jacobian_data_( jacobian_data ),
        pair_integral_( std::move( rule_generator ), { penalty, std::move( gap ) } ),
        pair_builder_( std::move( pair_builder ) )
  {
  }

  /**
   * @brief Compute the Common Plane force and its coordinate Jacobian.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned force and df_dx; other outputs are absent.
   */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override
  {
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    return evaluateAndAssemblePairResidualJacobian<Exec>( pairs, mesh1, mesh2, mesh_data_, jacobian_data_,
                                                          pair_integral_ );
  }

 private:
  const MfemMeshData& mesh_data_;          ///< Displacement transfer data referenced for this formulation's lifetime.
  const MfemJacobianData& jacobian_data_;  ///< Matrix transfer data referenced for this formulation's lifetime.
  const PairTractionIntegral<PairRuleGenerator, CommonPlanePenaltyTraction<GapDefinition>,
                             BasisFunction<EDGE, LINEAR_LAGRANGE>>
      pair_integral_;             ///< Integrates the fixed Common Plane penalty traction with linear edge bases.
  PairListBuilder pair_builder_;  ///< Owned search composition.
};
#endif  // TRIBOL_USE_ENZYME
#endif

}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
