// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/physics/NodalEnergyMortar.hpp"
#include "tribol/physics/QuadraturePointEnergyMortar.hpp"
#include "tribol/physics/ContactFormulationFactory.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/search/CoarseSearchFactory.hpp"
#include "tribol/search/ProjectedEdgeSearch.hpp"
#include "tribol/integ/QuadratureRuleGeneratorFactory.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"

#include <utility>
#include <cmath>
#include "axom/slic/interface/slic.hpp"
#include "axom/slic/interface/slic_macros.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/physics/CommonPlaneFormulation.hpp"
#include "tribol/enforcement/PenaltyStiffness.hpp"

namespace tribol {

#ifdef BUILD_REDECOMP
namespace {
template <ExecutionMode Exec>
std::unique_ptr<ContactFormulation> createCommonPlane( CouplingScheme* cs );
#ifdef TRIBOL_USE_ENZYME
template <ExecutionMode Exec>
std::unique_ptr<ContactFormulation> createEnergyMortar( CouplingScheme* cs );
#endif
}  // namespace
#endif

std::unique_ptr<ContactFormulation> createContactFormulation( CouplingScheme* cs )
{
  if ( !cs ) {
    SLIC_ERROR_ROOT( "User must register coupling scheme prior to calling createContactFormulation" );
    return nullptr;
  }

  if ( cs->getContactMethod() == COMMON_PLANE && cs->getParameters().use_common_plane_formulation ) {
    if ( !detail::validateCommonPlaneFormulation( *cs, cs->getParameters().common_plane_compute_jacobian ) ) {
      return nullptr;
    }
#ifdef BUILD_REDECOMP
    return dispatchExecutionMode( cs->getExecutionMode(),
                                  [&]<ExecutionMode Exec>() { return createCommonPlane<Exec>( cs ); } );
#endif
  }

  if ( cs->getContactMethod() == ENERGY_MORTAR ) {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
    return dispatchExecutionMode( cs->getExecutionMode(),
                                  [&]<ExecutionMode Exec>() { return createEnergyMortar<Exec>( cs ); } );
#else
    SLIC_ERROR_ROOT( "ENERGY_MORTAR requires Enzyme and redecomp to be built." );
    return nullptr;
#endif
  }

  return nullptr;
}

namespace detail {
bool validateCommonPlaneFormulation( CouplingScheme& cs, bool compute_jacobian )
{
#ifndef TRIBOL_USE_ENZYME
  if ( compute_jacobian ) {
    SLIC_ERROR_ROOT( "Common Plane Jacobian calculation requires Enzyme." );
    return false;
  }
#else
  (void)compute_jacobian;
#endif
#ifdef BUILD_REDECOMP
  const auto& options = cs.getEnforcementOptions().penalty_options;
  if ( cs.getContactMethod() != COMMON_PLANE || !cs.hasMfemData() || cs.getContactModel() != FRICTIONLESS ||
       cs.getEnforcementMethod() != PENALTY ||
       cs.getParameters().enforcement_location != EnforcementLocation::QuadraturePoint ||
       cs.getContactCase() == TIED_NORMAL || cs.getContactCase() == TIED_FULL || !options.kinematic_calc_set ||
       options.kinematic_calculation != KINEMATIC_CONSTANT || !options.constraint_type_set ||
       options.constraint_type != KINEMATIC ||
       ( options.rate_calc_set && options.rate_calculation != NO_RATE_PENALTY ) ) {
    SLIC_ERROR_ROOT(
        "Common Plane requires MFEM frictionless common-plane constant-penalty quadrature-point contact." );
    return false;
  }
  const auto& data = *cs.getMfemMeshData();
  const int order = data.GetParentCoords().ParFESpace()->FEColl()->GetOrder();
  if ( data.GetParentCoords().ParFESpace()->GetParMesh()->SpaceDimension() != 2 ||
       data.GetParentCoords().ParFESpace()->GetVDim() != 2 || order < 1 ||
       ( order > 1 && data.GetLORMesh() == nullptr ) ) {
    SLIC_ERROR_ROOT( "Common Plane supports only 2D linear or LOR-subdivided coordinates." );
    return false;
  }
  const auto* penalty1 = data.GetMesh1KinematicConstantPenalty();
  const auto* penalty2 = data.GetMesh2KinematicConstantPenalty();
  if ( !penalty1 || !penalty2 || !std::isfinite( *penalty1 ) || !std::isfinite( *penalty2 ) || *penalty1 <= 0.0 ||
       *penalty2 <= 0.0 || cs.getParameters().residual_gap != 0.0 || cs.getParameters().quadrature_order < 0 ||
       cs.getParameters().quadrature_order > 5 ) {
    SLIC_ERROR_ROOT(
        "Common Plane requires positive finite mesh penalties, zero residual gap, and quadrature order 0-5." );
    return false;
  }
  return true;
#else
  (void)cs;
  SLIC_ERROR_ROOT( "Common Plane requires Redecomp." );
  return false;
#endif
}
}  // namespace detail

#ifdef BUILD_REDECOMP
namespace {
/** @brief Construct the coarse and refined searches from the coupling scheme's current settings.
 * @tparam Exec Execution mode for both searches.
 * @param cs Coupling scheme supplying spatial dimension, binning settings, gap offset, and contact case.
 * @return Pair-list builder owning both searches. */
template <ExecutionMode Exec>
PairListBuilder makePairListBuilder( CouplingScheme& cs )
{
  return { createCoarseSearch( cs.spatialDim(), cs.getBinningMethod(), Exec, cs.getEffectiveBinningProximityScale(),
                               cs.getParameters().residual_gap ),
           std::make_unique<ProjectedEdgeSearch>( Exec, cs.getContactCase() == AUTO ) };
}

/**
 * @brief Build Common Plane with the requested residual evaluation and current search settings.
 * @tparam Exec Concrete execution mode for the searches and evaluator kernels.
 * @param cs Validated scheme owning displacement and Jacobian-transfer data.
 * @return Formulation owning its searches and reference-rule configuration.
 */
template <ExecutionMode Exec>
std::unique_ptr<ContactFormulation> createCommonPlane( CouplingScheme* cs )
{
  if ( cs->getParameters().common_plane_compute_jacobian && !cs->hasMfemJacobianData() ) {
    SLIC_ERROR_ROOT( "Common Plane requires parent Jacobian transfer data." );
    return nullptr;
  }
  const auto& data = *cs->getMfemMeshData();
  const auto* scale1 = data.GetMesh1KinematicPenaltyScale();
  const auto* scale2 = data.GetMesh2KinematicPenaltyScale();
  const RealT penalty1 =
      detail::checkedPenaltyStiffness( *data.GetMesh1KinematicConstantPenalty() * ( scale1 ? *scale1 : 1.0 ) );
  const RealT penalty2 =
      detail::checkedPenaltyStiffness( *data.GetMesh2KinematicConstantPenalty() * ( scale2 ? *scale2 : 1.0 ) );
  const RealT penalty = detail::checkedPenaltyStiffness( SeriesPenaltyStiffness{}( penalty1, penalty2 ) );
  return createWithQuadratureRuleGenerator(
      cs->getParameters().quadrature_order, CommonPlaneEdgeMapping{},
      [&]( auto rule_generator ) -> std::unique_ptr<ContactFormulation> {
        auto pair_builder = makePairListBuilder<Exec>( *cs );
        if ( cs->getParameters().common_plane_compute_jacobian ) {
#ifdef TRIBOL_USE_ENZYME
          return std::make_unique<
              CommonPlaneFormulation<Exec, decltype( rule_generator ), CommonPlaneGap, ResidualAndJacobian>>(
              std::move( rule_generator ), penalty, CommonPlaneGap{},
              ResidualAndJacobian{ *cs->getMfemMeshData(), *cs->getMfemJacobianData() }, std::move( pair_builder ) );
#else
          SLIC_ERROR_ROOT( "Common Plane Jacobian calculation requires Enzyme." );
          return nullptr;
#endif
        }
        return std::make_unique<CommonPlaneFormulation<Exec, decltype( rule_generator ), CommonPlaneGap, ResidualOnly>>(
            std::move( rule_generator ), penalty, CommonPlaneGap{}, ResidualOnly{ *cs->getMfemMeshData() },
            std::move( pair_builder ) );
      } );
}

#ifdef TRIBOL_USE_ENZYME
template <ExecutionMode Exec, typename PairRuleGenerator>
std::unique_ptr<ContactFormulation> createEnergyMortarForRuleGenerator( CouplingScheme* cs,
                                                                        PairRuleGenerator rule_generator )
{
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(), "ENERGY_MORTAR requires MFEM mesh data." );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemJacobianData(), "ENERGY_MORTAR requires MFEM Jacobian data." );

  const auto residual_gap = cs->getParameters().residual_gap;
  const bool use_penalty = cs->getEnforcementMethod() == PENALTY;
  const auto& data = *cs->getMfemMeshData();
  const RealT penalty = use_penalty
                            ? detail::checkedPenaltyStiffness( AveragePenaltyStiffness{}(
                                  detail::checkedPenaltyStiffness( *data.GetMesh1KinematicConstantPenalty() ),
                                  detail::checkedPenaltyStiffness( *data.GetMesh2KinematicConstantPenalty() ) ) )
                            : 0.0;
  auto pair_builder = makePairListBuilder<Exec>( *cs );
  if ( cs->getParameters().enforcement_location == EnforcementLocation::QuadraturePoint ) {
    return std::make_unique<QuadraturePointEnergyMortar<Exec, PairRuleGenerator, OffsetGap<OpposingNormalGap>>>(
        data, *cs->getMfemJacobianData(), std::move( rule_generator ), penalty,
        OffsetGap{ OpposingNormalGap{}, residual_gap }, std::move( pair_builder ) );
  }
  SLIC_ERROR_ROOT_IF( !cs->hasMfemSubmeshData(), "Nodal Energy Mortar requires pressure/gap data." );
  if ( use_penalty ) {
    return std::make_unique<
        NodalEnergyMortar<Exec, PairRuleGenerator, OffsetGap<OpposingNormalGap>, NodalPenaltyEnforcement>>(
        data, *cs->getMfemSubmeshData(), *cs->getMfemJacobianData(), std::move( rule_generator ),
        OffsetGap{ OpposingNormalGap{}, residual_gap }, NodalPenaltyEnforcement{ penalty }, std::move( pair_builder ) );
  }
  return std::make_unique<
      NodalEnergyMortar<Exec, PairRuleGenerator, OffsetGap<OpposingNormalGap>, NodalLagrangeMultiplierEnforcement>>(
      data, *cs->getMfemSubmeshData(), *cs->getMfemJacobianData(), std::move( rule_generator ),
      OffsetGap{ OpposingNormalGap{}, residual_gap }, NodalLagrangeMultiplierEnforcement{}, std::move( pair_builder ) );
}

template <ExecutionMode Exec>
std::unique_ptr<ContactFormulation> createEnergyMortar( CouplingScheme* cs )
{
  if ( !cs->hasMfemData() ) {
    SLIC_ERROR_ROOT( "ENERGY_MORTAR requires MFEM mesh data." );
    return nullptr;
  }
  if ( cs->getEnforcementMethod() == PENALTY ) {
    const auto& options = cs->getEnforcementOptions().penalty_options;
    const auto& data = *cs->getMfemMeshData();
    if ( !options.kinematic_calc_set || !data.GetMesh1KinematicConstantPenalty() ||
         !data.GetMesh2KinematicConstantPenalty() ) {
      return nullptr;
    }
    if ( options.kinematic_calculation != KINEMATIC_CONSTANT ) {
      SLIC_ERROR_ROOT( "Energy Mortar requires constant penalty stiffness." );
      return nullptr;
    }
  } else if ( cs->getEnforcementMethod() != LAGRANGE_MULTIPLIER ||
              cs->getParameters().enforcement_location != EnforcementLocation::Nodal ) {
    SLIC_ERROR_ROOT( "Energy Mortar requires penalty enforcement or nodal Lagrange multipliers." );
    return nullptr;
  }
  const SmoothedProjectedEdgeMapping mapping{};
  return createWithQuadratureRuleGenerator(
      cs->getParameters().quadrature_order, mapping, [&]( auto rule_generator ) -> std::unique_ptr<ContactFormulation> {
        return createEnergyMortarForRuleGenerator<Exec>( cs, std::move( rule_generator ) );
      } );
}

#endif  // TRIBOL_USE_ENZYME
}  // namespace
#endif  // BUILD_REDECOMP

}  // namespace tribol
