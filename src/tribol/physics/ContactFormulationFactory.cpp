// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/ContactFormulationFactory.hpp"
#include "tribol/physics/NewMethodAdapter.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/common/Parameters.hpp"

namespace tribol {

std::unique_ptr<ContactFormulation> createContactFormulation( CouplingScheme* cs )
{
  if ( !cs ) {
    return nullptr;
  }

  if ( cs->getContactMethod() == ENERGY_MORTAR ) {
    // Default parameters for now, or extract from CouplingScheme if available
    double k = 1000.0;
    double delta = 0.2;
    int N = 3;
    bool use_penalty_ = (cs->getEnforcementMethod() == PENALTY);

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
    if ( cs->hasMfemData() ) {
      // Attempt to get penalty from MfemMeshData if available
      auto* k_ptr = cs->getMfemMeshData()->GetMesh1KinematicConstantPenalty();
      if ( k_ptr ) {
        k = *k_ptr;
      }
    }

    SLIC_ERROR_ROOT_IF( !cs->hasMfemSubmeshData(), "ENERGY_MORTAR requires MFEM submesh data." );
    SLIC_ERROR_ROOT_IF( !cs->hasMfemJacobianData(), "ENERGY_MORTAR requires MFEM Jacobian data." );

    return std::make_unique<NewMethodAdapter>( *cs->getMfemSubmeshData(), *cs->getMfemJacobianData(), cs->getMesh1(),
                                               cs->getMesh2(), k, delta, N, use_penalty_ );
#else
    SLIC_ERROR_ROOT( "ENERGY_MORTAR requires Enzyme and redecomp to be built." );
    return nullptr;
#endif
  }

  return nullptr;
}

}  // namespace tribol
