// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTARFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTARFORMULATION_HPP_

#include "tribol/config.hpp"

#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/EnergyMortarAdapter.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

class TribolEnergyMortarFormulation : public ContactFormulation {
 public:
  TribolEnergyMortarFormulation( TribolFieldData& field_data, RealT penalty, RealT smoothing_length,
                                 int quadrature_points, bool differentiated_quadrature, bool use_penalty,
                                 EnforcementLocation enforcement_location );

  void setInterfacePairs( const ArrayT<InterfacePair>& pairs, int check_level ) override;
  void updateIntegrationRule() override;
  void updateNodalGaps() override;
  void updateNodalForces() override;
  bool hasTimeStepCalculation() override { return false; }
  RealT getEnergy() const override { return 0.0; }
  void updateMeshes( MeshData& mesh1, MeshData& mesh2 ) override;
  void updateConstantPenaltyStiffness( double mesh1_penalty, double mesh2_penalty ) override;
  void updateEnforcementLocation( EnforcementLocation location ) override;

  IntegrationRule computeTribolIntegrationRule() const override;
  TribolGapData computeTribolNodalGaps( const IntegrationRule& rule ) const override;
  TribolForceData computeTribolNodalForces( const IntegrationRule& rule ) const override;
  TribolForceData computeTribolNodalForces( const IntegrationRule& rule, const TribolGapData& gaps ) const override;
  TribolContactData computeTribolContactData( const IntegrationRule& rule ) const override;

 private:
  EnergyMortarAdapter<TribolFieldData> adapter_;
};

#ifdef BUILD_REDECOMP

class MfemEnergyMortarFormulation : public ContactFormulation {
 public:
  MfemEnergyMortarFormulation( MfemFieldData& field_data, RealT penalty, RealT smoothing_length,
                               int quadrature_points, bool differentiated_quadrature, bool use_penalty,
                               EnforcementLocation enforcement_location );

  void setInterfacePairs( const ArrayT<InterfacePair>& pairs, int check_level ) override;
  void updateIntegrationRule() override;
  void updateNodalGaps() override;
  void updateNodalForces() override;
  bool hasTimeStepCalculation() override { return false; }
  RealT getEnergy() const override { return 0.0; }
  void updateConstantPenaltyStiffness( double mesh1_penalty, double mesh2_penalty ) override;
  void updateEnforcementLocation( EnforcementLocation location ) override;

  IntegrationRule computeMfemIntegrationRule() const override;
  MfemGapData computeMfemNodalGaps( const IntegrationRule& rule ) const override;
  MfemForceData computeMfemNodalForces( const IntegrationRule& rule ) const override;
  MfemForceData computeMfemNodalForces( const IntegrationRule& rule, const MfemGapData& gaps ) const override;
  MfemContactData computeMfemContactData( const IntegrationRule& rule ) const override;

 private:
  EnergyMortarAdapter<MfemFieldData> adapter_;
};

#endif
#endif

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_ENERGYMORTARFORMULATION_HPP_ */
