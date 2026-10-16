// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/ContactFormulationFactory.hpp"
#include "tribol/physics/OperatorEnergyMortar.hpp"
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
    double delta = 0.1;
    int N = 3;
    bool enzyme_quadrature = true;

    const auto residual_gap = cs->getParameters().residual_gap;
    SLIC_ERROR_ROOT_IF( !cs->hasMfemContactData(), "ENERGY_MORTAR requires operator-based MFEM contact registration." );
    SLIC_ERROR_ROOT_IF( cs->getParameters().enforcement_location != EnforcementLocation::QuadraturePoint,
                        "Operator-based ENERGY_MORTAR currently requires quadrature-point enforcement." );
    if ( cs->getMfemContactData()->KinematicConstantPenalty() ) {
      const auto& penalties = *cs->getMfemContactData()->KinematicConstantPenalty();
      k = 0.5 * ( penalties[0] + penalties[1] );
    }
    return std::make_unique<OperatorEnergyMortar>( *cs, k, delta, N, enzyme_quadrature, residual_gap );
#else
    SLIC_ERROR_ROOT( "ENERGY_MORTAR requires Enzyme and redecomp to be built." );
    return nullptr;
#endif
  }

  return nullptr;
}

}  // namespace tribol
