// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/PairContactFormulation.hpp"
#include "tribol/mesh/MfemPairAssembler.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace tribol {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
namespace {
/// Check the element types supported by both current pair methods before reading their coordinate arrays.
/// @param mesh1 First current mesh.
/// @param mesh2 Second current mesh.
void validateLinearEdges( MeshData::Viewer mesh1, MeshData::Viewer mesh2 )
{
  for ( const auto& mesh : { mesh1, mesh2 } ) {
    if ( mesh.spatialDimension() != LinearEdgeGeometry::spatial_dimension || mesh.getElementType() != EDGE ||
         mesh.getFaceOrder() != LINEAR_LAGRANGE ||
         mesh.numberOfNodesPerElement() != BasisFunction<EDGE, LINEAR_LAGRANGE>::node_count ) {
      throw std::invalid_argument( "Pair contact methods require linear two-dimensional edges." );
    }
  }
}
}  // namespace

template <typename QuadratureRule, typename PenaltyStiffness>
typename EnergyMortarPairMethod<QuadratureRule, PenaltyStiffness>::Definition
EnergyMortarPairMethod<QuadratureRule, PenaltyStiffness>::prepare( const MfemMeshData& mesh_data,
                                                                   MeshData::Viewer mesh1,
                                                                   MeshData::Viewer mesh2 ) const
{
  if ( !penalty_enabled_ )
    throw std::invalid_argument( "Quadrature-point Energy Mortar requires penalty enforcement." );
  validateLinearEdges( mesh1, mesh2 );
  // Use Energy Mortar's default stiffness when no constant mesh penalties are supplied.
  constexpr RealT default_penalty = 1000.0;
  const auto* penalty1 = mesh_data.GetMesh1KinematicConstantPenalty();
  const auto* penalty2 = mesh_data.GetMesh2KinematicConstantPenalty();
  const RealT penalty = penalty1 && penalty2 ? penalty_stiffness_( *penalty1, *penalty2 ) : default_penalty;
  return Definition{ quadrature_rule_, penalty, residual_gap_ };
}

template <typename BaseQuadratureRule, typename PenaltyStiffness>
typename CommonPlanePairMethod<BaseQuadratureRule, PenaltyStiffness>::Definition
CommonPlanePairMethod<BaseQuadratureRule, PenaltyStiffness>::prepare( const MfemMeshData&, MeshData::Viewer mesh1,
                                                                      MeshData::Viewer mesh2 ) const
{
  validateLinearEdges( mesh1, mesh2 );
  const RealT penalty1 = mesh1.getElementData().m_penalty_stiffness * mesh1.getElementData().m_penalty_scale;
  const RealT penalty2 = mesh2.getElementData().m_penalty_stiffness * mesh2.getElementData().m_penalty_scale;
  if ( !std::isfinite( penalty1 ) || !std::isfinite( penalty2 ) || penalty1 <= 0.0 || penalty2 <= 0.0 ||
       !std::isfinite( residual_gap_ ) || residual_gap_ < 0.0 ) {
    throw std::invalid_argument(
        "Common Plane requires positive finite penalties and a nonnegative finite gap offset." );
  }
  return Definition{ CommonPlaneQuadratureRule<BaseQuadratureRule>{}, penalty_stiffness_( penalty1, penalty2 ),
                     residual_gap_ };
}

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
    if ( !PairMethod::isActive( primal ) ) continue;
    const auto derivatives = PairMethod::differentiate( coordinates.view(), definition, primal );
    if constexpr ( PairMethod::has_energy ) energy += derivatives.energy;
    assembler.addPair( pair, { derivatives.residual.data(), pair_dofs },
                       { derivatives.jacobian.data(), { pair_dofs, pair_dofs } } );
  }
  auto result = assembler.assemble();
  if constexpr ( PairMethod::has_energy ) result.energy = energy;
  return result;
}
template class EnergyMortarPairMethod<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>,
                                      AveragePenaltyStiffness>;
template class CommonPlanePairMethod<EdgeQuadratureRuleGenerator<1>, AveragePenaltyStiffness>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<1>, AveragePenaltyStiffness>>;
template class EnergyMortarPairMethod<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>,
                                      SeriesPenaltyStiffness>;
template class CommonPlanePairMethod<EdgeQuadratureRuleGenerator<1>, SeriesPenaltyStiffness>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<1>, SeriesPenaltyStiffness>>;
template class EnergyMortarPairMethod<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>,
                                      AveragePenaltyStiffness>;
template class CommonPlanePairMethod<EdgeQuadratureRuleGenerator<2>, AveragePenaltyStiffness>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<2>, AveragePenaltyStiffness>>;
template class EnergyMortarPairMethod<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>,
                                      SeriesPenaltyStiffness>;
template class CommonPlanePairMethod<EdgeQuadratureRuleGenerator<2>, SeriesPenaltyStiffness>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<2>, SeriesPenaltyStiffness>>;
template class EnergyMortarPairMethod<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>,
                                      AveragePenaltyStiffness>;
template class CommonPlanePairMethod<EdgeQuadratureRuleGenerator<3>, AveragePenaltyStiffness>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<3>, AveragePenaltyStiffness>>;
template class EnergyMortarPairMethod<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>,
                                      SeriesPenaltyStiffness>;
template class CommonPlanePairMethod<EdgeQuadratureRuleGenerator<3>, SeriesPenaltyStiffness>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<3>, SeriesPenaltyStiffness>>;
#endif
}  // namespace tribol
