// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/ContactFormulationFactory.hpp"
#include "tribol/physics/EnergyMortarAdapter.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/search/CoarseSearchFactory.hpp"
#include "tribol/search/ProjectedEdgeSearch.hpp"
#include "tribol/integ/QuadratureRuleFactory.hpp"

#include <utility>

namespace tribol {

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
namespace {
template <template <typename> class EnforcementLocation, typename QuadratureRule>
std::unique_ptr<ContactFormulation> createEnergyMortarAdapter( CouplingScheme* cs )
{
  // Use the same smoothing fraction for pair acceptance and integration bounds.
  double k = 1000.0;
  const SmoothedEdgeProjection projection{};
  const double delta = projection.smoothing_fraction;
  const bool enzyme_quadrature = true;

  // ENERGY_MORTAR supports a penalty-style mode driven by the kinematic penalty parameters, even if the coupling
  // scheme is registered with LM enforcement (which is often done to enable submesh/pressure infrastructure).
  const auto& penalty_opts = cs->getEnforcementOptions().penalty_options;
  bool use_penalty = penalty_opts.kinematic_calc_set;

  if ( cs->hasMfemData() ) {
    // Attempt to get penalty from MfemMeshData if available
    auto* k1_ptr = cs->getMfemMeshData()->GetMesh1KinematicConstantPenalty();
    auto* k2_ptr = cs->getMfemMeshData()->GetMesh2KinematicConstantPenalty();
    if ( k1_ptr && k2_ptr ) {
      k = 0.5 * ( *k1_ptr + *k2_ptr );
    }
  }

  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(), "ENERGY_MORTAR requires MFEM mesh data." );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemSubmeshData(), "ENERGY_MORTAR requires MFEM submesh data." );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemJacobianData(), "ENERGY_MORTAR requires MFEM Jacobian data." );

  const auto residual_gap = cs->getParameters().residual_gap;
  const auto execution_mode = cs->getExecutionMode();
  const auto binning_method = cs->getBinningMethod();
  const auto binning_proximity_scale = cs->getEffectiveBinningProximityScale();
  auto coarse_search = createCoarseSearch( cs->spatialDimension(), binning_method, execution_mode,
                                           binning_proximity_scale, residual_gap );
  auto refined_search = std::make_unique<ProjectedEdgeSearch<SmoothedEdgeProjection>>( execution_mode, projection,
                                                                                       cs->getContactCase() == AUTO );
  return std::make_unique<EnergyMortarAdapter<EnforcementLocation, QuadratureRule>>(
      *cs->getMfemMeshData(), *cs->getMfemSubmeshData(), *cs->getMfemJacobianData(), std::move( coarse_search ),
      std::move( refined_search ), k, QuadratureRule{}, SmoothProjectedEdgeInterval{ delta }, enzyme_quadrature,
      use_penalty, residual_gap );
}

template <template <typename> class EnforcementLocation>
std::unique_ptr<ContactFormulation> createEnergyMortarForLocation( CouplingScheme* cs )
{
  return createWithQuadratureRule( cs->getParameters().quadrature_order,
                                   [&]<typename QuadratureRule>() -> std::unique_ptr<ContactFormulation> {
                                     return createEnergyMortarAdapter<EnforcementLocation, QuadratureRule>( cs );
                                   } );
}

std::unique_ptr<ContactFormulation> createEnergyMortar( CouplingScheme* cs )
{
  if ( cs->getParameters().enforcement_location == EnforcementLocation::QuadraturePoint ) {
    return createEnergyMortarForLocation<QuadraturePoint>( cs );
  }
  return createEnergyMortarForLocation<Nodal>( cs );
}
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

}  // namespace tribol
