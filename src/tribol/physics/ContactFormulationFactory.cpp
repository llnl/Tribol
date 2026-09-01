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
    // Default parameters for now, or extract from CouplingScheme if available
    double k = 1000.0;
    double delta = cs->getParameters().energy_mortar_smoothing_length;
    double normal_smoothing_start_angle = cs->getParameters().energy_mortar_normal_smoothing_start_angle;
    int N = 3;
    bool enzyme_quadrature = true;

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

    const auto enforcement_location = cs->getParameters().enforcement_location;
    if ( enforcement_location == EnforcementLocation::QuadraturePoint ) {
      return std::make_unique<EnergyMortarAdapter<QuadraturePoint>>( *cs->getMfemMeshData(), *cs->getMfemSubmeshData(),
                                                                     *cs->getMfemJacobianData(), k, delta,
                                                                     normal_smoothing_start_angle, N, enzyme_quadrature,
                                                                     use_penalty );
    } else {
      return std::make_unique<EnergyMortarAdapter<Nodal>>( *cs->getMfemMeshData(), *cs->getMfemSubmeshData(),
                                                           *cs->getMfemJacobianData(), k, delta,
                                                           normal_smoothing_start_angle, N, enzyme_quadrature,
                                                           use_penalty );
    }
#else
    SLIC_ERROR_ROOT( "ENERGY_MORTAR requires Enzyme and redecomp to be built." );
    return nullptr;
#endif
  }

  return nullptr;
}

}  // namespace tribol
