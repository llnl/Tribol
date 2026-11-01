// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/PairContactFormulation.hpp"
#include "tribol/mesh/MfemNodalAssembler.hpp"
#include "tribol/mesh/MfemPairAssembler.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tribol {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
namespace {
/**
 * @brief Evaluate nodal energy, pressure, and Enzyme partial derivatives with respect to (weighted_gap, area) at each
 * dual submesh true degree of freedom.
 * @tparam NodeDefinitionFactory Callable taking a dual true-DOF index and returning a two-DOF nodal energy definition.
 * @param assembler Nodal assembler providing dual true-DOF vector allocation.
 * @param quantities Assembled nodal weighted gaps and tributary areas on dual submesh true DOFs.
 * @param has_area_derivatives Whether the nodal definition depends on tributary area.
 * @param has_outer_hessian Whether second partial derivatives of nodal energy are nonzero.
 * @param make_node_definition Factory returning the nodal energy definition for a given dual true DOF.
 * @return Nodal energy, pressure, and first/second partial derivatives on dual submesh true DOFs.
 */
template <typename NodeDefinitionFactory>
MfemNodalDerivativeData evaluateNodalDerivatives( const MfemNodalAssembler& assembler,
                                                  const MfemNodalQuantityData& quantities, bool has_area_derivatives,
                                                  bool has_outer_hessian, NodeDefinitionFactory&& make_node_definition )
{
  auto data = assembler.makeDerivativeData( has_area_derivatives, has_outer_hessian );
  const int num_dofs = quantities.weighted_gap.size();
  RealT local_energy = 0.0;
  for ( int i = 0; i < num_dofs; ++i ) {
    const MdArray<RealT, 2> nodal_state{ quantities.weighted_gap[i], quantities.area[i] };
    const auto definition = make_node_definition( i );
    using Definition = std::decay_t<decltype( definition )>;
    const auto evaluated = EnergyPairEvaluator<Definition>::evaluate( nodal_state.view(), definition );
    local_energy += evaluated.energy;
    data.pressure[i] = evaluated.residual[0];
    data.d_energy_d_gap[i] = evaluated.residual[0];
    if ( has_area_derivatives ) {
      data.d_energy_d_area[i] = evaluated.residual[1];
    }
    if ( has_outer_hessian ) {
      data.d2_energy_dg2[i] = evaluated.jacobian( 0, 0 );
      if ( has_area_derivatives ) {
        data.d2_energy_dg_da[i] = evaluated.jacobian( 0, 1 );
        data.d2_energy_da2[i] = evaluated.jacobian( 1, 1 );
      }
    }
  }
  MPI_Allreduce( &local_energy, &data.energy, 1, MPI_DOUBLE, MPI_SUM, quantities.weighted_gap.get().GetComm() );
  return data;
}

/**
 * @brief Evaluate nodal integrals and gradients on each accepted pair and assemble them onto the dual submesh.
 * @tparam PairMethod Nodal pair method type.
 * @param definition Prepared nodal integral definition.
 * @param mesh1 Current first (non-mortar) mesh.
 * @param mesh2 Current second (mortar) mesh.
 * @param pairs Accepted interface element pairs.
 * @param use_penalty Whether nodal penalty enforcement is active (clamping positive weighted gaps).
 * @param[in,out] assembler Nodal assembler accumulating quantities and gradients.
 * @return Assembled nodal weighted gaps, areas, normalized gaps, and coordinate Jacobians.
 */
template <typename PairMethod>
MfemNodalQuantityData assembleNodalQuantities( const typename PairMethod::Definition& definition,
                                               MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                               ArrayViewT<const InterfacePair> pairs, bool use_penalty,
                                               MfemNodalAssembler& assembler )
{
  using Definition = typename PairMethod::Definition;
  constexpr IndexT node_count = Definition::node_count;
  constexpr IndexT pair_dofs = Definition::dof_count;
  constexpr IndexT first_dofs = PairMethod::FirstGeometry::Coords::size();
  for ( const auto& pair : pairs ) {
    MdArray<RealT, pair_dofs> coordinates;
    mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
    mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + first_dofs );
    const auto primal = definition.evaluate( coordinates.view() );
    if ( primal.quadrature_result.status != QuadratureStatus::Success )
      reportQuadratureFailure( primal.quadrature_result.status );
    if ( !primal.has_active_points ) continue;
    const auto derivatives = PairMethod::differentiateQuantities( coordinates.view(), definition, primal );
    assembler.addPairQuantities( pair, { derivatives.values.data(), node_count },
                                 { derivatives.values.data() + node_count, node_count },
                                 { derivatives.gradients.data(), { node_count, pair_dofs } },
                                 { derivatives.gradients.data() + node_count * pair_dofs, { node_count, pair_dofs } } );
  }
  return assembler.assembleQuantities( use_penalty, PairMethod::area_tolerance );
}

/**
 * @brief Move the selected nodal gap vector and weighted-gap Jacobian into a ContactFormulationResult.
 * @param[in,out] quantities Assembled nodal quantities whose gap and Jacobian pointers are transferred.
 * @param use_penalty If true, return the area-normalized nodal gap; otherwise return the weighted nodal gap.
 * @return Result owning gap and dg_dx.
 */
ContactFormulationResult takeGapResult( MfemNodalQuantityData& quantities, bool use_penalty )
{
  ContactFormulationResult result;
  result.gap.reset( use_penalty ? quantities.normalized_gap.release() : quantities.weighted_gap.release() );
  result.dg_dx.reset( quantities.weighted_gap_jacobian.release() );
  return result;
}
}  // namespace

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
    auto quantities =
        assembleNodalQuantities<PairMethod>( definition, mesh1, mesh2, pairs.view(), method_.usesPenalty(), assembler );
    return takeGapResult( quantities, method_.usesPenalty() );
  }
}

template <typename PairMethod>
ContactFormulationResult PairContactFormulation<PairMethod>::computeNodalForces( MeshData::Viewer mesh1,
                                                                                 MeshData::Viewer mesh2 ) const
{
  using Definition = typename PairMethod::Definition;
  constexpr IndexT pair_dofs = Definition::dof_count;
  constexpr IndexT first_dofs = PairMethod::FirstGeometry::Coords::size();
  constexpr IndexT second_dofs = PairMethod::SecondGeometry::Coords::size();
  static_assert( pair_dofs == first_dofs + second_dofs );
  if constexpr ( !PairMethod::is_nodal ) {
    const auto definition = method_.prepare( mesh_data_, mesh1, mesh2 );
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
  } else {
    constexpr IndexT node_count = Definition::node_count;
    const auto definition = method_.prepare( mesh1, mesh2 );
    const bool use_penalty = method_.usesPenalty();
    const auto candidates = coarse_search_->findCandidatePairs( mesh1, mesh2 );
    const auto pairs = refined_search_->filterCandidatePairs( candidates, mesh1, mesh2 );
    MfemNodalAssembler assembler( mesh_data_, *submesh_data_, jacobian_data_, mesh1, mesh2, pairs.size() );
    auto quantities =
        assembleNodalQuantities<PairMethod>( definition, mesh1, mesh2, pairs.view(), use_penalty, assembler );

    using NodalPhysics = typename Definition::Physics;
    MfemNodalDerivativeData derivatives;
    if ( use_penalty ) {
      const auto parameters = method_.preparePenalty( mesh_data_, mesh1, mesh2 );
      const NodalPenaltyStateEnergy<NodalPhysics> penalty_energy{ parameters.penalty, parameters.residual_gap };
      derivatives =
          evaluateNodalDerivatives( assembler, quantities, true, true, [&]( int ) { return penalty_energy; } );
    } else {
      const auto lm_vec = assembler.getSubmeshPressureTrueDofs();
      if ( lm_vec.size() != quantities.weighted_gap.size() ) {
        throw std::invalid_argument( "Lagrange multiplier vector size must match the dual submesh true-DOF count." );
      }
      derivatives = evaluateNodalDerivatives( assembler, quantities, false, false, [&]( int node ) {
        return NodalLagrangeMultiplierStateEnergy<NodalPhysics>{ lm_vec[node], 0.0 };
      } );
    }

    const auto redecomp_d_gap = assembler.transferToRedecomp( derivatives.d_energy_d_gap.get() );
    const auto redecomp_d_area = derivatives.has_area_derivatives
                                     ? assembler.transferToRedecomp( derivatives.d_energy_d_area.get() )
                                     : mfem::GridFunction{};
    for ( const auto& pair : pairs ) {
      const auto conn1 = mesh1.getConnectivity()( pair.element_id1_ );
      const MdArray<RealT, node_count> gap_weights{ redecomp_d_gap( conn1[0] ), redecomp_d_gap( conn1[1] ) };
      if ( use_penalty && gap_weights[0] == 0.0 && gap_weights[1] == 0.0 ) {
        continue;
      }
      MdArray<RealT, pair_dofs> coordinates;
      mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
      mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + first_dofs );
      const MdArray<RealT, node_count> area_weights =
          derivatives.has_area_derivatives
              ? MdArray<RealT, node_count>{ redecomp_d_area( conn1[0] ), redecomp_d_area( conn1[1] ) }
              : MdArray<RealT, node_count>{};
      const auto weighted_definition = definition.withWeights( gap_weights, area_weights );
      const auto pair_derivatives = PairMethod::differentiateWeightedEnergy( coordinates.view(), weighted_definition );
      assembler.addPairHessian( pair, { pair_derivatives.jacobian.data(), { pair_dofs, pair_dofs } } );
    }

    auto force_vec = assembler.assembleForce( quantities, derivatives );
    auto df_dx = assembler.assembleJacobian( quantities, derivatives );

    ContactFormulationResult result;
    if ( !use_penalty ) {
      auto df_dp = quantities.weighted_gap_jacobian.transpose();
      result.df_dp.reset( df_dp.release() );
    }
    auto gaps = takeGapResult( quantities, use_penalty );
    result.gap = std::move( gaps.gap );
    result.dg_dx = std::move( gaps.dg_dx );
    result.energy = derivatives.energy;
    result.pressure.reset( derivatives.pressure.release() );
    result.force.reset( force_vec.release() );
    result.df_dx.reset( df_dx.release() );
    return result;
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
