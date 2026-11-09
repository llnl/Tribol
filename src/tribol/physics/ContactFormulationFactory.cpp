// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/common/BasicTypes.hpp"
#include "tribol/physics/ContactFormulationFactory.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/search/CoarseSearchFactory.hpp"
#include "tribol/search/ProjectedEdgeSearch.hpp"
#include "tribol/integ/QuadratureRuleFactory.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"

#include <utility>
#include <cmath>
#include <stdexcept>
#include "tribol/physics/PairContactFormulation.hpp"

namespace tribol {

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
namespace {
std::unique_ptr<ContactFormulation> createEnergyMortar( CouplingScheme* cs );
std::unique_ptr<ContactFormulation> createCommonPlane( CouplingScheme* cs );
}  // namespace
#endif

std::unique_ptr<ContactFormulation> createContactFormulation( CouplingScheme* cs )
{
  if ( !cs ) {
    SLIC_ERROR_ROOT( "User must register coupling scheme prior to calling createContactFormulation" );
    return nullptr;
  }

  if ( cs->getContactMethod() == COMMON_PLANE && cs->getParameters().use_common_plane_formulation ) {
    detail::validateCommonPlaneFormulation( *cs );
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
    return createCommonPlane( cs );
#endif
  }

  if ( cs->getContactMethod() == ENERGY_MORTAR ) {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
    return createEnergyMortar( cs );
#else
    SLIC_ERROR_ROOT( "ENERGY_MORTAR requires Enzyme and redecomp to be built." );
    return nullptr;
#endif
  }

  return nullptr;
}

namespace detail {
void validateCommonPlaneFormulation( CouplingScheme& cs )
{
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
  const auto& options = cs.getEnforcementOptions().penalty_options;
  if ( cs.getContactMethod() != COMMON_PLANE || !cs.hasMfemData() || cs.getContactModel() != FRICTIONLESS ||
       cs.getEnforcementMethod() != PENALTY ||
       cs.getParameters().enforcement_location != EnforcementLocation::QuadraturePoint ||
       cs.getContactCase() == TIED_NORMAL || cs.getContactCase() == TIED_FULL || isOnDevice( cs.getExecutionMode() ) ||
       !options.kinematic_calc_set || options.kinematic_calculation != KINEMATIC_CONSTANT ||
       !options.constraint_type_set || options.constraint_type != KINEMATIC ||
       ( options.rate_calc_set && options.rate_calculation != NO_RATE_PENALTY ) ) {
    throw std::invalid_argument(
        "Common Plane requires host MFEM frictionless common-plane constant-penalty quadrature-point "
        "contact." );
  }
  const auto& data = *cs.getMfemMeshData();
  if ( data.GetParentCoords().ParFESpace()->GetParMesh()->SpaceDimension() != 2 ||
       data.GetParentCoords().ParFESpace()->FEColl()->GetOrder() != 1 ||
       data.GetParentCoords().ParFESpace()->GetVDim() != 2 || data.GetLORMesh() != nullptr ) {
    throw std::invalid_argument( "Common Plane supports only linear 2D coordinates without LOR subdivision." );
  }
  const auto* penalty1 = data.GetMesh1KinematicConstantPenalty();
  const auto* penalty2 = data.GetMesh2KinematicConstantPenalty();
  if ( !penalty1 || !penalty2 || !std::isfinite( *penalty1 ) || !std::isfinite( *penalty2 ) || *penalty1 <= 0.0 ||
       *penalty2 <= 0.0 || !std::isfinite( cs.getParameters().residual_gap ) || cs.getParameters().residual_gap < 0.0 ||
       cs.getParameters().quadrature_order < 0 || cs.getParameters().quadrature_order > 5 ) {
    throw std::invalid_argument(
        "Common Plane requires positive finite mesh penalties, a nonnegative finite residual gap, and "
        "quadrature order 0-5." );
  }
#else
  (void)cs;
  throw std::runtime_error( "Common Plane requires Enzyme and Redecomp." );
#endif
}
}  // namespace detail

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
namespace {
/**
 * @brief Build the requested common-plane specialization using current search settings.
 * @param cs Validated scheme owning displacement and Jacobian-transfer data.
 * @return Formulation owning its searches and reference-rule configuration.
 */
std::unique_ptr<ContactFormulation> createCommonPlane( CouplingScheme* cs )
{
  if ( !cs->hasMfemJacobianData() )
    throw std::invalid_argument( "Common Plane requires parent Jacobian transfer data." );
  return createWithQuadratureRule(
      cs->getParameters().quadrature_order, [&]<typename BaseRule>() -> std::unique_ptr<ContactFormulation> {
        auto coarse = createCoarseSearch( cs->spatialDimension(), cs->getBinningMethod(), cs->getExecutionMode(),
                                          cs->getEffectiveBinningProximityScale(), cs->getParameters().residual_gap );
        auto refined = std::make_unique<ProjectedEdgeSearch>( cs->getExecutionMode(), cs->getContactCase() == AUTO );
        return std::make_unique<PairContactFormulation<CommonPlanePairMethod<BaseRule>>>(
            *cs->getMfemMeshData(), *cs->getMfemJacobianData(), std::move( coarse ), std::move( refined ),
            CommonPlanePairMethod<BaseRule>{
                CommonPlaneQuadratureRule<BaseRule>{},
                CommonPlaneResidual{ OffsetGap{ CommonPlaneGap{}, cs->getParameters().residual_gap } },
                ScaledMeshPenaltyInputs<>{} } );
      } );
}

template <typename QuadratureRule>
std::unique_ptr<ContactFormulation> createEnergyMortarForRule( CouplingScheme* cs, QuadratureRule quadrature_rule )
{
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(), "ENERGY_MORTAR requires MFEM mesh data." );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemJacobianData(), "ENERGY_MORTAR requires MFEM Jacobian data." );

  const auto residual_gap = cs->getParameters().residual_gap;
  const auto execution_mode = cs->getExecutionMode();
  const auto binning_method = cs->getBinningMethod();
  const auto binning_proximity_scale = cs->getEffectiveBinningProximityScale();
  auto coarse_search = createCoarseSearch( cs->spatialDimension(), binning_method, execution_mode,
                                           binning_proximity_scale, residual_gap );
  auto refined_search = std::make_unique<ProjectedEdgeSearch>( execution_mode, cs->getContactCase() == AUTO );
  const bool use_penalty = cs->getEnforcementOptions().penalty_options.kinematic_calc_set;
  if ( cs->getParameters().enforcement_location == EnforcementLocation::QuadraturePoint ) {
    using Method = EnergyMortarQuadraturePointMethod<QuadratureRule>;
    return std::make_unique<PairContactFormulation<Method>>(
        *cs->getMfemMeshData(), *cs->getMfemJacobianData(), std::move( coarse_search ), std::move( refined_search ),
        Method{ std::move( quadrature_rule ), EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, residual_gap } },
                RegisteredPenaltyInputs<>{ use_penalty } } );
  }
  SLIC_ERROR_ROOT_IF( !cs->hasMfemSubmeshData(), "Nodal Energy Mortar requires pressure/gap data." );
  using Method = EnergyMortarAssembledMethod<QuadratureRule>;
  return std::make_unique<PairContactFormulation<Method>>(
      *cs->getMfemMeshData(), *cs->getMfemSubmeshData(), *cs->getMfemJacobianData(), std::move( coarse_search ),
      std::move( refined_search ),
      Method{ std::move( quadrature_rule ), EnergyMortarNodal{ OffsetGap{ OpposingNormalGap{}, residual_gap } },
              RegisteredPenaltyInputs<>{ use_penalty } } );
}

std::unique_ptr<ContactFormulation> createEnergyMortar( CouplingScheme* cs )
{
  const SmoothedProjectedEdgeMapping mapping{};
  return createWithQuadratureRule( cs->getParameters().quadrature_order, mapping,
                                   [&]( auto quadrature_rule ) -> std::unique_ptr<ContactFormulation> {
                                     return createEnergyMortarForRule( cs, std::move( quadrature_rule ) );
                                   } );
}

}  // namespace
#endif

}  // namespace tribol
