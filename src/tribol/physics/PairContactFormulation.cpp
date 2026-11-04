// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/PairContactFormulation.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"

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
  if ( !coarse_search_ || !refined_search_ ) throw std::invalid_argument( "Pair contact requires both searches." );
}

template <typename PairMethod>
PairContactFormulation<PairMethod>::PairContactFormulation( MfemMeshData& mesh_data, MfemSubmeshData& submesh_data,
                                                            MfemJacobianData& jacobian_data,
                                                            std::unique_ptr<CoarseSearch> coarse_search,
                                                            std::unique_ptr<RefinedSearch> refined_search,
                                                            PairMethod method )
    : mesh_data_( mesh_data ),
      jacobian_data_( jacobian_data ),
      submesh_data_( &submesh_data ),
      coarse_search_( std::move( coarse_search ) ),
      refined_search_( std::move( refined_search ) ),
      method_( std::move( method ) )
{
  if ( !PairMethod::is_nodal ) throw std::invalid_argument( "Quadrature-point contact does not use dual data." );
  if ( !coarse_search_ || !refined_search_ ) throw std::invalid_argument( "Pair contact requires both searches." );
}

template <typename PairMethod>
ArrayT<InterfacePair> PairContactFormulation<PairMethod>::findPairs( MeshData::Viewer mesh1,
                                                                     MeshData::Viewer mesh2 ) const
{
  const auto candidates = coarse_search_->findCandidatePairs( mesh1, mesh2 );
  return refined_search_->filterCandidatePairs( candidates, mesh1, mesh2 );
}

template <typename PairMethod>
auto PairContactFormulation<PairMethod>::getPairCoordinates( InterfacePair pair, MeshData::Viewer mesh1,
                                                             MeshData::Viewer mesh2 ) -> Coordinates
{
  Coordinates coordinates;
  mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
  mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + Definition::first_dofs );
  return coordinates;
}

template <typename PairMethod>
MfemNodalQuantityData PairContactFormulation<PairMethod>::assembleNodalQuantities(
    const Definition& definition, MeshData::Viewer mesh1, MeshData::Viewer mesh2, ArrayViewT<const InterfacePair> pairs,
    MfemNodalAssembler& assembler ) const
{
  if constexpr ( PairMethod::is_nodal ) {
    constexpr IndexT nodes = Definition::node_count, dofs = Definition::dof_count;
    for ( const auto& pair : pairs ) {
      const auto coordinates = getPairCoordinates( pair, mesh1, mesh2 );
      const auto primal = definition.evaluate( coordinates.view() );
      if ( primal.quadrature_result.status != QuadratureStatus::Success )
        reportQuadratureFailure( primal.quadrature_result.status );
      if ( !primal.has_active_points ) continue;
      const auto derivatives = PairMethod::differentiateQuantities( coordinates.view(), definition, primal );
      assembler.addPairQuantities( pair, { derivatives.values.data(), nodes },
                                   { derivatives.values.data() + nodes, nodes },
                                   { derivatives.gradients.data(), { nodes, dofs } },
                                   { derivatives.gradients.data() + nodes * dofs, { nodes, dofs } } );
    }
    return assembler.assembleQuantities( false, PairMethod::area_tolerance );
  } else {
    throw std::logic_error( "Quadrature-point contact does not assemble nodal gaps." );
  }
}

template <typename PairMethod>
ContactFormulationResult PairContactFormulation<PairMethod>::takeGapResult( MfemNodalQuantityData& quantities ) const
{
  if constexpr ( PairMethod::is_nodal ) {
    const bool penalty = method_.usesPenalty();
    if ( penalty ) {
      mfem::Array<int> open_nodes;
      for ( int node = 0; node < quantities.weighted_gap.size(); ++node )
        if ( quantities.weighted_gap[node] > 0.0 ) {
          quantities.normalized_gap[node] = 0.0;
          open_nodes.Append( node );
        }
      quantities.weighted_gap_jacobian.eliminateRows( open_nodes );
    }
    ContactFormulationResult result;
    result.gap.reset( penalty ? quantities.normalized_gap.release() : quantities.weighted_gap.release() );
    result.dg_dx.reset( quantities.weighted_gap_jacobian.release() );
    return result;
  } else {
    throw std::logic_error( "Quadrature-point contact has no nodal gap field." );
  }
}

template <typename PairMethod>
ContactFormulationResult PairContactFormulation<PairMethod>::computeNodalGaps( MeshData::Viewer mesh1,
                                                                               MeshData::Viewer mesh2 ) const
{
  if constexpr ( PairMethod::is_nodal ) {
    const auto definition = method_.prepare( mesh_data_, mesh1, mesh2 );
    const auto pairs = findPairs( mesh1, mesh2 );
    MfemNodalAssembler assembler( mesh_data_, *submesh_data_, jacobian_data_, mesh1, mesh2, pairs.size() );
    auto quantities = assembleNodalQuantities( definition, mesh1, mesh2, pairs.view(), assembler );
    return takeGapResult( quantities );
  } else {
    return ContactFormulation::computeNodalGaps( mesh1, mesh2 );
  }
}

template <typename PairMethod>
ContactFormulationResult PairContactFormulation<PairMethod>::computeNodalForces( MeshData::Viewer mesh1,
                                                                                 MeshData::Viewer mesh2 ) const
{
  const auto definition = method_.prepare( mesh_data_, mesh1, mesh2 );
  const auto pairs = findPairs( mesh1, mesh2 );
  if constexpr ( PairMethod::is_nodal ) {
    return assembleNodalForces( definition, mesh1, mesh2, pairs.view() );
  } else {
    MfemPairAssembler assembler( mesh_data_, jacobian_data_, mesh1, mesh2, pairs.size() );
    RealT local_energy = 0.0;
    for ( const auto& pair : pairs ) {
      const auto coordinates = getPairCoordinates( pair, mesh1, mesh2 );
      const auto primal = definition.evaluate( coordinates.view() );
      if ( primal.quadrature_result.status != QuadratureStatus::Success )
        reportQuadratureFailure( primal.quadrature_result.status );
      if ( !primal.has_active_points ) continue;
      const auto derivatives = PairMethod::differentiate( coordinates.view(), definition, primal );
      if constexpr ( PairMethod::has_energy ) local_energy += derivatives.energy;
      assembler.addPair( pair, { derivatives.residual.data(), Definition::dof_count },
                         { derivatives.jacobian.data(), { Definition::dof_count, Definition::dof_count } } );
    }
    auto result = assembler.assemble();
    if constexpr ( PairMethod::has_energy ) {
      RealT energy = 0.0;
      MPI_Allreduce( &local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, result.force->GetComm() );
      result.energy = energy;
    }
    return result;
  }
}

template <typename PairMethod>
ContactFormulationResult PairContactFormulation<PairMethod>::assembleNodalForces(
    const Definition& definition, MeshData::Viewer mesh1, MeshData::Viewer mesh2,
    ArrayViewT<const InterfacePair> pairs ) const
{
  if constexpr ( PairMethod::is_nodal ) {
    constexpr IndexT nodes = Definition::node_count, dofs = Definition::dof_count;
    const bool use_penalty = method_.usesPenalty();
    MfemNodalAssembler assembler( mesh_data_, *submesh_data_, jacobian_data_, mesh1, mesh2, pairs.size() );
    auto quantities = assembleNodalQuantities( definition, mesh1, mesh2, pairs, assembler );
    const auto make_vector = [&] {
      shared::ParVector vector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_->GetSubmeshFESpace() ) );
      vector.fill( 0.0 );
      return vector;
    };
    auto gap_weights = make_vector(), area_weights = make_vector();
    shared::ParVector gap_gap, gap_area, area_area;
    RealT energy = 0.0;
    if ( use_penalty ) {
      const RealT stiffness = method_.preparePenalty( mesh_data_, mesh1, mesh2 );
      gap_gap = make_vector();
      gap_area = make_vector();
      area_area = make_vector();
      RealT local_energy = 0.0;
      for ( int node = 0; node < gap_weights.size(); ++node ) {
        const auto partials = Definition::Physics::penaltyPartials(
            quantities.weighted_gap[node], quantities.area[node], stiffness, PairMethod::area_tolerance );
        local_energy += partials.energy;
        gap_weights[node] = partials.gap;
        area_weights[node] = partials.area;
        gap_gap[node] = partials.gap_gap;
        gap_area[node] = partials.gap_area;
        area_area[node] = partials.area_area;
      }
      MPI_Allreduce( &local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, gap_weights.get().GetComm() );
    } else {
      gap_weights = assembler.getSubmeshPressureTrueDofs();
      if ( gap_weights.size() != quantities.weighted_gap.size() )
        throw std::invalid_argument( "LM values must match dual true DOFs." );
      energy = gap_weights.dot( quantities.weighted_gap );
    }
    auto force = gap_weights * quantities.weighted_gap_jacobian;
    if ( use_penalty ) force += area_weights * quantities.area_jacobian;

    const auto redecomp_gap_weights = assembler.transferToRedecomp( gap_weights.get() );
    const auto redecomp_area_weights =
        use_penalty ? assembler.transferToRedecomp( area_weights.get() ) : mfem::GridFunction{};
    for ( const auto& pair : pairs ) {
      const auto connectivity = mesh1.getConnectivity()( pair.element_id1_ );
      MdArray<RealT, nodes> gap{}, area{};
      bool has_weights = false;
      for ( IndexT node = 0; node < nodes; ++node ) {
        gap[node] = redecomp_gap_weights( connectivity[node] );
        if ( use_penalty ) area[node] = redecomp_area_weights( connectivity[node] );
        has_weights = has_weights || gap[node] != 0.0 || area[node] != 0.0;
      }
      if ( !has_weights ) continue;
      const auto coordinates = getPairCoordinates( pair, mesh1, mesh2 );
      const auto weighted = definition.withWeights( gap, area );
      const auto derivatives = PairMethod::differentiateWeightedEnergy( coordinates.view(), weighted );
      assembler.addPairJacobian( pair, { derivatives.jacobian.data(), { dofs, dofs } } );
    }
    auto jacobian = assembler.assembleHessians();
    if ( use_penalty ) {
      const auto gg = assembler.makeSubmeshDiagonalMatrix( gap_gap.get() );
      const auto ga = assembler.makeSubmeshDiagonalMatrix( gap_area.get() );
      const auto aa = assembler.makeSubmeshDiagonalMatrix( area_area.get() );
      jacobian += shared::ParSparseMat::rap( gg, quantities.weighted_gap_jacobian );
      const auto mixed = shared::ParSparseMat::rap( quantities.weighted_gap_jacobian, ga, quantities.area_jacobian );
      jacobian += mixed;
      jacobian += mixed.transpose();
      jacobian += shared::ParSparseMat::rap( aa, quantities.area_jacobian );
    }
    ContactFormulationResult result;
    if ( !use_penalty ) {
      auto df_dp = quantities.weighted_gap_jacobian.transpose();
      result.df_dp.reset( df_dp.release() );
    }
    auto gaps = takeGapResult( quantities );
    result.gap = std::move( gaps.gap );
    result.dg_dx = std::move( gaps.dg_dx );
    result.energy = energy;
    result.force.reset( force.release() );
    result.df_dx.reset( jacobian.release() );
    result.pressure.reset( gap_weights.release() );
    return result;
  } else {
    throw std::logic_error( "Quadrature-point contact does not use nodal assembly." );
  }
}

template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarNodalPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarNodalPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarNodalPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarNodalPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarNodalPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>, AveragePenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>, SeriesPenaltyStiffness>>;
template class PairContactFormulation<EnergyMortarNodalPairMethod<
    PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>, SeriesPenaltyStiffness>>;

#endif
}  // namespace tribol
