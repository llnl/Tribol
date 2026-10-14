// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/ContactFormulationFactory.hpp"
#include "tribol/physics/EnergyMortarAdapter.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/common/Parameters.hpp"

namespace tribol {

std::unique_ptr<ContactFormulation> createContactFormulation( CouplingScheme* cs )
{
  if ( !cs ) {
    SLIC_ERROR_ROOT( "User must register coupling scheme prior to calling createContactFormulation" );
    return nullptr;
  }

  if ( cs->getContactMethod() == ENERGY_MORTAR ) {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
    SLIC_ERROR_ROOT_IF( !cs->hasMfemData(), "ENERGY_MORTAR requires MFEM mesh data." );
    SLIC_ERROR_ROOT_IF( !cs->hasMfemSubmeshData(), "ENERGY_MORTAR requires MFEM submesh data." );
    SLIC_ERROR_ROOT_IF( !cs->hasMfemJacobianData(), "ENERGY_MORTAR requires MFEM Jacobian data." );

    ContactParams contact_params;
    contact_params.k = 1000.0;
    contact_params.del = cs->getParameters().energy_mortar_smoothing_length;
    contact_params.normal_smoothing_start_angle = cs->getParameters().energy_mortar_normal_smoothing_start_angle;
    contact_params.residual_gap = cs->getParameters().residual_gap;
    contact_params.is_auto_contact = cs->getParameters().auto_contact_check;
    contact_params.auto_contact_penetration_fraction = cs->getParameters().auto_contact_pen_frac;
    const double residual_gap_ramp_angle = cs->getParameters().energy_mortar_residual_gap_ramp_angle;
    const bool updates_residual_gap_ramp = cs->getParameters().energy_mortar_residual_gap_ramp_updates;

    // ENERGY_MORTAR supports a penalty-style mode driven by the kinematic penalty parameters, even if the coupling
    // scheme is registered with LM enforcement (which is often done to enable submesh/pressure infrastructure).
    const auto& penalty_opts = cs->getEnforcementOptions().penalty_options;
    const bool use_penalty = penalty_opts.kinematic_calc_set;

    // When both surfaces provide a constant penalty, EnergyMortar uses their arithmetic mean.
    auto* k1_ptr = cs->getMfemMeshData()->GetMesh1KinematicConstantPenalty();
    auto* k2_ptr = cs->getMfemMeshData()->GetMesh2KinematicConstantPenalty();
    if ( k1_ptr && k2_ptr ) {
      contact_params.k = 0.5 * ( *k1_ptr + *k2_ptr );
    }

    const auto enforcement_location = cs->getParameters().enforcement_location;
    if ( enforcement_location == EnforcementLocation::QuadraturePoint ) {
      return std::make_unique<EnergyMortarAdapter<QuadraturePoint>>(
          *cs->getMfemMeshData(), *cs->getMfemSubmeshData(), *cs->getMfemJacobianData(), contact_params,
          residual_gap_ramp_angle, updates_residual_gap_ramp, use_penalty );
    } else {
      return std::make_unique<EnergyMortarAdapter<Nodal>>(
          *cs->getMfemMeshData(), *cs->getMfemSubmeshData(), *cs->getMfemJacobianData(), contact_params,
          residual_gap_ramp_angle, updates_residual_gap_ramp, use_penalty );
    }
#else
    SLIC_ERROR_ROOT( "ENERGY_MORTAR requires Enzyme and redecomp to be built." );
    return nullptr;
#endif
  }

  return nullptr;
}

}  // namespace tribol
