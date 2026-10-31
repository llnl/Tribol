// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/PairContactFormulation.hpp"
#include "tribol/mesh/MfemPairAssembler.hpp"
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
  if ( !coarse_search_ || !refined_search_ )
    throw std::invalid_argument( "Pair contact requires both search objects." );
}

template <typename PairMethod>
ContactFormulationResult PairContactFormulation<PairMethod>::computeNodalForces( MeshData::Viewer mesh1,
                                                                                 MeshData::Viewer mesh2 ) const
{
  const auto definition = method_.prepare( mesh_data_, mesh1, mesh2 );
  using Definition = typename PairMethod::Definition;
  constexpr IndexT pair_dofs = Definition::dof_count;
  constexpr IndexT first_dofs = PairMethod::FirstGeometry::Coords::size();
  constexpr IndexT second_dofs = PairMethod::SecondGeometry::Coords::size();
  static_assert( pair_dofs == first_dofs + second_dofs );
  const auto candidates = coarse_search_->findCandidatePairs( mesh1, mesh2 );
  const auto pairs = refined_search_->filterCandidatePairs( candidates, mesh1, mesh2 );
  MfemPairAssembler assembler( mesh_data_, jacobian_data_, mesh1, mesh2, pairs.size() );
  RealT energy = 0.0;
  for ( const auto& pair : pairs ) {
    MdArray<RealT, pair_dofs> coordinates;
    mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
    mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + first_dofs );
    const auto primal = definition.evaluate( coordinates.view() );
    if ( primal.quadrature_result.status != QuadratureStatus::Success )
      reportQuadratureFailure( primal.quadrature_result.status );
    if ( !primal.has_active_points ) continue;
    const auto derivatives = PairMethod::differentiate( coordinates.view(), definition, primal );
    if constexpr ( PairMethod::has_energy ) energy += derivatives.energy;
    assembler.addPair( pair, { derivatives.residual.data(), pair_dofs },
                       { derivatives.jacobian.data(), { pair_dofs, pair_dofs } } );
  }
  auto result = assembler.assemble();
  if constexpr ( PairMethod::has_energy ) result.energy = energy;
  return result;
}
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<1>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<1>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<2>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<2>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<3>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<3>, SeriesPenaltyStiffness>>;
#endif
}  // namespace tribol
