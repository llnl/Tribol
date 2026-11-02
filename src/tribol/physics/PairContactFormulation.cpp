// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/PairContactFormulation.hpp"
#include "tribol/mesh/MfemContactAssembly.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include <stdexcept>
#include <utility>

namespace tribol {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
template <typename PairMethod>
PairContactFormulation<PairMethod>::PairContactFormulation( MfemMeshData& mesh_data, MfemJacobianData& jacobian_data,
                                                            std::unique_ptr<CoarseSearch> coarse_search,
                                                            std::unique_ptr<RefinedSearch> refined_search,
                                                            PairMethod method )
    : mesh_data_( mesh_data ),
      jacobian_data_( jacobian_data ),
      coarse_search_( std::move( coarse_search ) ),
      refined_search_( std::move( refined_search ) ),
      method_( std::move( method ) )
{
  if ( PairMethod::is_nodal ) throw std::invalid_argument( "Nodal pair contact requires dual submesh data." );
  if ( !coarse_search_ || !refined_search_ )
    throw std::invalid_argument( "Pair contact requires both search objects." );
}

template <typename PairMethod>
PairContactFormulation<PairMethod>::PairContactFormulation( MfemMeshData& mesh_data, MfemSubmeshData& submesh_data,
                                                            MfemJacobianData& jacobian_data,
                                                            std::unique_ptr<CoarseSearch> coarse_search,
                                                            std::unique_ptr<RefinedSearch> refined_search,
                                                            PairMethod method )
    : mesh_data_( mesh_data ),
      submesh_data_( &submesh_data ),
      jacobian_data_( jacobian_data ),
      coarse_search_( std::move( coarse_search ) ),
      refined_search_( std::move( refined_search ) ),
      method_( std::move( method ) )
{
  if ( !PairMethod::is_nodal )
    throw std::invalid_argument( "Parent-only pair contact does not use dual submesh data." );
  if ( !coarse_search_ || !refined_search_ )
    throw std::invalid_argument( "Pair contact requires both search objects." );
}

template <typename PairMethod>
ContactFormulationResult PairContactFormulation<PairMethod>::computeNodalGaps( MeshData::Viewer mesh1,
                                                                               MeshData::Viewer mesh2 ) const
{
  if constexpr ( !PairMethod::is_nodal ) {
    return ContactFormulation::computeNodalGaps( mesh1, mesh2 );
  } else {
    const auto definition = method_.prepare( mesh1, mesh2 );
    const auto candidates = coarse_search_->findCandidatePairs( mesh1, mesh2 );
    const auto pairs = refined_search_->filterCandidatePairs( candidates, mesh1, mesh2 );
    MfemNodalAssembler assembler( mesh_data_, *submesh_data_, jacobian_data_, mesh1, mesh2, pairs.size() );
    const Domain domain{ mesh1, mesh2, pairs.view() };
    return assembleNodalContactGaps<PairMethod>( domain, definition, method_.usesPenalty(), assembler );
  }
}

template <typename PairMethod>
ContactFormulationResult PairContactFormulation<PairMethod>::computeNodalForces( MeshData::Viewer mesh1,
                                                                                 MeshData::Viewer mesh2 ) const
{
  if constexpr ( !PairMethod::is_nodal ) {
    const auto definition = method_.prepare( mesh_data_, mesh1, mesh2 );
    const auto candidates = coarse_search_->findCandidatePairs( mesh1, mesh2 );
    const auto pairs = refined_search_->filterCandidatePairs( candidates, mesh1, mesh2 );
    const Domain domain{ mesh1, mesh2, pairs.view() };
    MfemPairAssembler assembler( mesh_data_, jacobian_data_, mesh1, mesh2, pairs.size() );
    return assemblePairContact<PairMethod>( domain, definition, assembler );
  } else {
    const auto definition = method_.prepare( mesh1, mesh2 );
    const auto candidates = coarse_search_->findCandidatePairs( mesh1, mesh2 );
    const auto pairs = refined_search_->filterCandidatePairs( candidates, mesh1, mesh2 );
    const Domain domain{ mesh1, mesh2, pairs.view() };
    MfemNodalAssembler assembler( mesh_data_, *submesh_data_, jacobian_data_, mesh1, mesh2, pairs.size() );
    return assembleNodalContactForces<PairMethod>( domain, definition, method_, mesh_data_, assembler );
  }
}

template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarNodalPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<1>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarNodalPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<1>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarNodalPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<2>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarNodalPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<2>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarNodalPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<3>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarNodalPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<3>, SeriesPenaltyStiffness>>;
#endif
}  // namespace tribol
