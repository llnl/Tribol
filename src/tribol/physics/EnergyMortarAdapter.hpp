// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTARADAPTER_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTARADAPTER_HPP_

#include "tribol/config.hpp"

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/physics/EnergyMortar.hpp"
#include "tribol/physics/EnergyMortarData.hpp"
#include "tribol/physics/EnergyMortarFieldData.hpp"

#include <atomic>
#include <cstdint>
#include <vector>

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

template <typename FieldDataT>
class EnergyMortarAdapter {
 public:
  using GapData = typename FieldDataT::GapData;
  using ForceData = typename FieldDataT::ForceData;
  using ContactData = typename FieldDataT::ContactData;

  EnergyMortarAdapter( FieldDataT& field_data, RealT penalty, RealT smoothing_length, int quadrature_points,
                       bool differentiated_quadrature, bool use_penalty,
                       EnforcementLocation enforcement_location );

  EnergyMortarAdapter( const EnergyMortarAdapter& ) = delete;
  EnergyMortarAdapter& operator=( const EnergyMortarAdapter& ) = delete;

  void setInterfacePairs( const ArrayT<InterfacePair>& pairs, int check_level = 0 );
  void setInterfacePairs( const std::vector<InterfacePair>& pairs );

  void setEnforcementLocation( EnforcementLocation location );
  void setPenalty( RealT mesh1_penalty, RealT mesh2_penalty );
  void setUsePenalty( bool use_penalty ) { use_penalty_ = use_penalty; }
  void setTiedContact( bool tied_contact ) { tied_contact_ = tied_contact; }

  IntegrationRule computeIntegrationRule() const;
  GapData computeNodalGaps( const IntegrationRule& rule ) const;
  ForceData computeNodalForces( const IntegrationRule& rule ) const;
  ForceData computeNodalForces( const IntegrationRule& rule, const GapData& gaps ) const;
  ContactData computeContactData( const IntegrationRule& rule ) const;

  FieldDataT& fieldData() const { return field_data_; }
  EnforcementLocation enforcementLocation() const { return enforcement_location_; }
  bool usesPenalty() const { return use_penalty_; }

 private:
  struct NativeGapEvaluation {
    TribolGapData data;
    PrimitivePairContributions primitive;
  };

  struct NativeForceEvaluation {
    TribolForceData data;
    PrimitivePairContributions primitive;
  };

  void validateRule( const IntegrationRule& rule ) const;
  void validateGaps( const IntegrationRule& rule, const TribolGapData& gaps ) const;
#ifdef BUILD_REDECOMP
  void validateGaps( const IntegrationRule& rule, const MfemGapData& gaps ) const;
#endif
  NativeGapEvaluation computeNativeNodalGaps( const IntegrationRule& rule, bool filter_opening ) const;
  NativeForceEvaluation computeNativeNodalForces( const IntegrationRule& rule, const TribolGapData* gaps,
                                                   const std::vector<RealT>* pressure = nullptr ) const;

#ifdef BUILD_REDECOMP
  MfemGapData makeMfemGapData( const IntegrationRule& rule, const NativeGapEvaluation& gaps ) const;
  MfemForceData makeMfemForceData( const IntegrationRule& rule, const NativeGapEvaluation* native_gaps,
                                   const MfemGapData* mfem_gaps ) const;
#endif

  FieldDataT& field_data_;
  ContactParams params_;
  bool use_penalty_;
  bool tied_contact_{ false };
  EnforcementLocation enforcement_location_;
  std::vector<InterfacePair> pairs_;
  static std::atomic<std::uint64_t> next_rule_id_;
};

#endif

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_ENERGYMORTARADAPTER_HPP_ */
