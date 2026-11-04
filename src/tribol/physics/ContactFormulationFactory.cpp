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
}  // namespace
#endif

std::unique_ptr<ContactFormulation> createContactFormulation( CouplingScheme* cs )
{
  if ( !cs ) {
    SLIC_ERROR_ROOT( "User must register coupling scheme prior to calling createContactFormulation" );
    return nullptr;
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

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
namespace {
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
    using Method = EnergyMortarPairMethod<QuadratureRule>;
    return std::make_unique<PairContactFormulation<Method>>(
        *cs->getMfemMeshData(), *cs->getMfemJacobianData(), std::move( coarse_search ), std::move( refined_search ),
        Method{ std::move( quadrature_rule ), EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, residual_gap } },
                RegisteredPenaltyInputs<>{ use_penalty } } );
  }
  SLIC_ERROR_ROOT_IF( !cs->hasMfemSubmeshData(), "Nodal Energy Mortar requires pressure/gap data." );
  using Method = EnergyMortarNodalPairMethod<QuadratureRule>;
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
