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
    // Default parameters for now, or extract from CouplingScheme if available
    double k = 1000.0;
    double delta = 0.1;
    int N = 3;
    // ENERGY_MORTAR supports a penalty-style mode driven by the kinematic penalty parameters, even if the coupling
    // scheme is registered with LM enforcement (which is often done to enable submesh/pressure infrastructure).
    const auto& penalty_opts = cs->getEnforcementOptions().penalty_options;
    // TODO: Figure out how contact formulations interact with coupling scheme duplication (SRW)
    bool use_penalty_ = penalty_opts.kinematic_calc_set;

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
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

    return std::make_unique<EnergyMortarAdapter>(
        *cs->getMfemMeshData(), *cs->getMfemSubmeshData(), *cs->getMfemJacobianData(), k, delta, N,
        cs->getParameters().energy_mortar_enzyme_quadrature,
        cs->getParameters().energy_mortar_fixed_integration_jacobian, use_penalty_,
        cs->getParameters().energy_mortar_normal_mode, cs->getParameters().energy_mortar_projection_smoothing,
        cs->getParameters().energy_mortar_projection_smoothing_curve,
        cs->getParameters().energy_mortar_h1_active_set_smoothing_gap,
        cs->getParameters().energy_mortar_qp_derivative_blend_min_gap,
        cs->getParameters().energy_mortar_qp_derivative_blend_max_gap,
        cs->getParameters().energy_mortar_qp_derivative_blend_weight,
        cs->getParameters().energy_mortar_qp_derivative_blend_enzyme_gap_weight,
        cs->getParameters().energy_mortar_qp_frozen_integration,
        cs->getParameters().energy_mortar_penalty_mode,
        cs->getParameters().energy_mortar_nodal_energy_basis,
        cs->getParameters().energy_mortar_eta_gap_scaling,
        cs->getParameters().energy_mortar_eta_angle_smoothing,
        cs->getParameters().energy_mortar_eta_angle_smoothing_start,
        cs->getParameters().energy_mortar_nodal_energy_angle_smoothing, cs->getParameters().residual_gap );
#else
    SLIC_ERROR_ROOT( "ENERGY_MORTAR requires Enzyme and redecomp to be built." );
    return nullptr;
#endif
  }

  return nullptr;
}

}  // namespace tribol
