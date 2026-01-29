// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/MortarFormulation.hpp"

namespace tribol {

template <typename IntegrationRule, typename PointwiseGapAndNormal, typename ForceAndGapMethod>
void MortarFormulation<IntegrationRule, PointwiseGapAndNormal, ForceAndGapMethod>::setInterfacePairs(
    ArrayT<InterfacePair>&& pairs, int check_level )
{
  integration_rule_.template findPairsInContact<PointwiseGapAndNormal>( std::move( pairs ), check_level );
}

template <typename IntegrationRule, typename PointwiseGapAndNormal, typename ForceAndGapMethod>
void MortarFormulation<IntegrationRule, PointwiseGapAndNormal, ForceAndGapMethod>::updateIntegrationRule()
{
  integration_rule_.template updateRule<PointwiseGapAndNormal>();
}

template <typename IntegrationRule, typename PointwiseGapAndNormal, typename ForceAndGapMethod>
void MortarFormulation<IntegrationRule, PointwiseGapAndNormal, ForceAndGapMethod>::updateNodalGaps()
{
  force_and_gap_method_.template updateNodalGaps<PointwiseGapAndNormal>( integration_rule_.getRule() );
}

template <typename IntegrationRule, typename PointwiseGapAndNormal, typename ForceAndGapMethod>
void MortarFormulation<IntegrationRule, PointwiseGapAndNormal, ForceAndGapMethod>::updateNodalForces()
{
  force_and_gap_method_.template updateNodalForces<PointwiseGapAndNormal>( integration_rule_.getRule() );
}

template <typename IntegrationRule, typename PointwiseGapAndNormal, typename ForceAndGapMethod>
RealT MortarFormulation<IntegrationRule, PointwiseGapAndNormal, ForceAndGapMethod>::computeTimeStep()
{
  force_and_gap_method_.computeTimeStep( integration_rule_.getRule() );
}

template <typename IntegrationRule, typename PointwiseGapAndNormal, typename ForceAndGapMethod>
void MortarFormulation<IntegrationRule, PointwiseGapAndNormal, ForceAndGapMethod>::getMfemForce(
    mfem::Vector& forces ) const
{
  force_and_gap_method_.getMfemForce( forces );
}

template <typename IntegrationRule, typename PointwiseGapAndNormal, typename ForceAndGapMethod>
void MortarFormulation<IntegrationRule, PointwiseGapAndNormal, ForceAndGapMethod>::getMfemGap(
    mfem::Vector& gaps ) const
{
  force_and_gap_method_.getMfemGap( gaps );
}

template <typename IntegrationRule, typename PointwiseGapAndNormal, typename ForceAndGapMethod>
mfem::ParGridFunction& MortarFormulation<IntegrationRule, PointwiseGapAndNormal, ForceAndGapMethod>::getMfemPressure()
{
  return force_and_gap_method_.getMfemPressure();
}

template <typename IntegrationRule, typename PointwiseGapAndNormal, typename ForceAndGapMethod>
std::unique_ptr<mfem::HypreParMatrix>
MortarFormulation<IntegrationRule, PointwiseGapAndNormal, ForceAndGapMethod>::getMfemDfDx() const
{
  return force_and_gap_method_.getMfemDfDx();
}

template <typename IntegrationRule, typename PointwiseGapAndNormal, typename ForceAndGapMethod>
std::unique_ptr<mfem::HypreParMatrix>
MortarFormulation<IntegrationRule, PointwiseGapAndNormal, ForceAndGapMethod>::getMfemDgDx() const
{
  return force_and_gap_method_.getMfemDgDx();
}

template <typename IntegrationRule, typename PointwiseGapAndNormal, typename ForceAndGapMethod>
std::unique_ptr<mfem::HypreParMatrix>
MortarFormulation<IntegrationRule, PointwiseGapAndNormal, ForceAndGapMethod>::getMfemDfDp() const
{
  return force_and_gap_method_.getMfemDfDp();
}

}  // namespace tribol