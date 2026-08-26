// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortarFormulation.hpp"

#include "axom/slic.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

namespace {

void legacyComputeError()
{
  SLIC_ERROR_ROOT( "ENERGY_MORTAR is evaluated through the owning compute APIs, not tribol::update()." );
}

}  // namespace

TribolEnergyMortarFormulation::TribolEnergyMortarFormulation( TribolFieldData& field_data, RealT penalty,
                                                               RealT smoothing_length, int quadrature_points,
                                                               bool differentiated_quadrature, bool use_penalty,
                                                               EnforcementLocation enforcement_location )
    : adapter_( field_data, penalty, smoothing_length, quadrature_points, differentiated_quadrature, use_penalty,
                enforcement_location )
{ }

void TribolEnergyMortarFormulation::setInterfacePairs( const ArrayT<InterfacePair>& pairs, int check_level )
{
  adapter_.setInterfacePairs( pairs, check_level );
}

void TribolEnergyMortarFormulation::updateIntegrationRule() { legacyComputeError(); }
void TribolEnergyMortarFormulation::updateNodalGaps() { legacyComputeError(); }
void TribolEnergyMortarFormulation::updateNodalForces() { legacyComputeError(); }

void TribolEnergyMortarFormulation::updateMeshes( MeshData& mesh1, MeshData& mesh2 )
{
  adapter_.fieldData().update( mesh1, mesh2 );
}

void TribolEnergyMortarFormulation::updateConstantPenaltyStiffness( double mesh1_penalty, double mesh2_penalty )
{
  adapter_.setPenalty( mesh1_penalty, mesh2_penalty );
}

void TribolEnergyMortarFormulation::updateEnforcementLocation( EnforcementLocation location )
{
  adapter_.setEnforcementLocation( location );
}

IntegrationRule TribolEnergyMortarFormulation::computeTribolIntegrationRule() const
{
  return adapter_.computeIntegrationRule();
}

TribolGapData TribolEnergyMortarFormulation::computeTribolNodalGaps( const IntegrationRule& rule ) const
{
  return adapter_.computeNodalGaps( rule );
}

TribolForceData TribolEnergyMortarFormulation::computeTribolNodalForces( const IntegrationRule& rule ) const
{
  return adapter_.computeNodalForces( rule );
}

TribolForceData TribolEnergyMortarFormulation::computeTribolNodalForces( const IntegrationRule& rule,
                                                                         const TribolGapData& gaps ) const
{
  return adapter_.computeNodalForces( rule, gaps );
}

TribolContactData TribolEnergyMortarFormulation::computeTribolContactData( const IntegrationRule& rule ) const
{
  return adapter_.computeContactData( rule );
}

#ifdef BUILD_REDECOMP

MfemEnergyMortarFormulation::MfemEnergyMortarFormulation( MfemFieldData& field_data, RealT penalty,
                                                           RealT smoothing_length, int quadrature_points,
                                                           bool differentiated_quadrature, bool use_penalty,
                                                           EnforcementLocation enforcement_location )
    : adapter_( field_data, penalty, smoothing_length, quadrature_points, differentiated_quadrature, use_penalty,
                enforcement_location )
{ }

void MfemEnergyMortarFormulation::setInterfacePairs( const ArrayT<InterfacePair>& pairs, int check_level )
{
  adapter_.setInterfacePairs( pairs, check_level );
}

void MfemEnergyMortarFormulation::updateIntegrationRule() { legacyComputeError(); }
void MfemEnergyMortarFormulation::updateNodalGaps() { legacyComputeError(); }
void MfemEnergyMortarFormulation::updateNodalForces() { legacyComputeError(); }

void MfemEnergyMortarFormulation::updateConstantPenaltyStiffness( double mesh1_penalty, double mesh2_penalty )
{
  adapter_.setPenalty( mesh1_penalty, mesh2_penalty );
}

void MfemEnergyMortarFormulation::updateEnforcementLocation( EnforcementLocation location )
{
  adapter_.setEnforcementLocation( location );
}

IntegrationRule MfemEnergyMortarFormulation::computeMfemIntegrationRule() const
{
  return adapter_.computeIntegrationRule();
}

MfemGapData MfemEnergyMortarFormulation::computeMfemNodalGaps( const IntegrationRule& rule ) const
{
  return adapter_.computeNodalGaps( rule );
}

MfemForceData MfemEnergyMortarFormulation::computeMfemNodalForces( const IntegrationRule& rule ) const
{
  return adapter_.computeNodalForces( rule );
}

MfemForceData MfemEnergyMortarFormulation::computeMfemNodalForces( const IntegrationRule& rule,
                                                                   const MfemGapData& gaps ) const
{
  return adapter_.computeNodalForces( rule, gaps );
}

MfemContactData MfemEnergyMortarFormulation::computeMfemContactData( const IntegrationRule& rule ) const
{
  return adapter_.computeContactData( rule );
}

#endif
#endif

}  // namespace tribol
