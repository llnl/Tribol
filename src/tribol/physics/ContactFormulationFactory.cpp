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
    EnergyMortarOptions opts;
    // ENERGY_MORTAR supports a penalty-style mode driven by the kinematic penalty parameters, even if the coupling
    // scheme is registered with LM enforcement (which is often done to enable submesh/pressure infrastructure).
    const auto& penalty_opts = cs->getEnforcementOptions().penalty_options;
    // TODO: Figure out how contact formulations interact with coupling scheme duplication (SRW)
    bool use_penalty = penalty_opts.kinematic_calc_set;

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
    if ( cs->hasMfemData() ) {
      auto* k_ptr = cs->getMfemMeshData()->GetMesh1KinematicConstantPenalty();
      if ( k_ptr ) {
        opts.k = *k_ptr;
      }
      if ( auto* em_opts = cs->getMfemMeshData()->GetEnergyMortarOptions() ) {
        opts.smoothing_type = em_opts->smoothing_type;
        opts.penalty_smoothing = em_opts->penalty_smoothing;
        opts.penalty_smoothing_del = em_opts->penalty_smoothing_del;
      }
    }

    SLIC_ERROR_ROOT_IF( !cs->hasMfemData(), "ENERGY_MORTAR requires MFEM mesh data." );
    SLIC_ERROR_ROOT_IF( !cs->hasMfemSubmeshData(), "ENERGY_MORTAR requires MFEM submesh data." );
    SLIC_ERROR_ROOT_IF( !cs->hasMfemJacobianData(), "ENERGY_MORTAR requires MFEM Jacobian data." );

    return std::make_unique<EnergyMortarAdapter>(
        *cs->getMfemMeshData(), *cs->getMfemSubmeshData(), *cs->getMfemJacobianData(), cs->getMesh1(), cs->getMesh2(),
        opts, use_penalty );
#else
    SLIC_ERROR_ROOT( "ENERGY_MORTAR requires Enzyme and redecomp to be built." );
    return nullptr;
#endif
  }

  return nullptr;
}

}  // namespace tribol
