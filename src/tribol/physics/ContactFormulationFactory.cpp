// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/ContactFormulationFactory.hpp"
#include "tribol/physics/EnergyMortarFormulation.hpp"
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
#if defined( TRIBOL_USE_ENZYME )
    double k = 1000.0;
    double delta = 0.1;
    int N = 3;
    bool enzyme_quadrature = true;

    // ENERGY_MORTAR supports a penalty-style mode driven by the kinematic penalty parameters, even if the coupling
    // scheme is registered with LM enforcement (which is often done to enable submesh/pressure infrastructure).
    const auto& penalty_opts = cs->getEnforcementOptions().penalty_options;
    bool use_penalty = penalty_opts.kinematic_calc_set;

#ifdef BUILD_REDECOMP
    if ( cs->hasMfemData() ) {
      auto* k1_ptr = cs->getMfemMeshData()->GetMesh1KinematicConstantPenalty();
      auto* k2_ptr = cs->getMfemMeshData()->GetMesh2KinematicConstantPenalty();
      if ( k1_ptr && k2_ptr ) {
        k = 0.5 * ( *k1_ptr + *k2_ptr );
      }
    }
#endif

    const auto enforcement_location = cs->getParameters().enforcement_location;
    if ( auto* field_data = dynamic_cast<TribolFieldData*>( cs->getFieldData() ) ) {
      const auto& mesh1_data = field_data->mortarMesh().getElementData();
      const auto& mesh2_data = field_data->nonmortarMesh().getElementData();
      if ( mesh1_data.m_is_kinematic_constant_penalty_set && mesh2_data.m_is_kinematic_constant_penalty_set ) {
        k = 0.5 * ( mesh1_data.m_penalty_stiffness + mesh2_data.m_penalty_stiffness );
      }
      return std::make_unique<TribolEnergyMortarFormulation>( *field_data, k, delta, N, enzyme_quadrature,
                                                               use_penalty, enforcement_location );
    }
#ifdef BUILD_REDECOMP
    if ( auto* field_data = dynamic_cast<MfemFieldData*>( cs->getFieldData() ) ) {
      return std::make_unique<MfemEnergyMortarFormulation>( *field_data, k, delta, N, enzyme_quadrature,
                                                            use_penalty, enforcement_location );
    }
#endif
    SLIC_ERROR_ROOT( "ENERGY_MORTAR requires a TribolFieldData or MfemFieldData object." );
    return nullptr;
#else
    SLIC_ERROR_ROOT( "ENERGY_MORTAR requires Enzyme support." );
    return nullptr;
#endif
  }

  return nullptr;
}

}  // namespace tribol
