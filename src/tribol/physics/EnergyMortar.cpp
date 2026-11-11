// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortar.hpp"

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
#include <mpi.h>
#include <stdexcept>
#include "tribol/common/LoopExec.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/mesh/MfemNodalAssembler.hpp"
#include "tribol/mesh/MfemPairAssembler.hpp"
#include "tribol/physics/GlobalScalarAssembler.hpp"
#include "tribol/physics/PairContribution.hpp"
#include "tribol/physics/PairKernel.hpp"

namespace tribol {
namespace {

template <typename Function>
MfemNodalQuantityData assembleNodalQuantities(
    ExecutionMode execution_mode, const ArrayT<InterfacePair>& pairs,
    PairValueJacobianBuffer<Function::value_count, Function::dof_count>& buffer, MfemNodalAssembler& assembler )
{
  constexpr IndexT node_count = Function::node_count;
  constexpr IndexT dof_count = Function::dof_count;
  const auto view = buffer.view();
  assembler.addPairQuantities<node_count, dof_count>(
      execution_mode, pairs,
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT value ) {
        return view.has_active_points[pair] ? view.value( pair, value ) : 0.0;
      },
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT value, IndexT coordinate ) {
        return view.has_active_points[pair] ? view.jacobian( pair, value, coordinate ) : 0.0;
      } );
  return assembler.assembleQuantities( false, NodalEnergyMortar<typename Function::Quadrature>::area_tolerance );
}

template <typename Inputs>
ContactFormulationResult takeGapResult( MfemNodalQuantityData& quantities,
                                        [[maybe_unused]] const PenaltyEnforcement<Inputs>& enforcement )
{
  mfem::Array<int> open_nodes;
  for ( int node = 0; node < quantities.weighted_gap.size(); ++node ) {
    if ( quantities.weighted_gap[node] > 0.0 ) {
      quantities.normalized_gap[node] = 0.0;
      open_nodes.Append( node );
    }
  }
  quantities.weighted_gap_jacobian.eliminateRows( open_nodes );
  ContactFormulationResult result;
  result.gap.reset( quantities.normalized_gap.release() );
  result.dg_dx.reset( quantities.weighted_gap_jacobian.release() );
  return result;
}

ContactFormulationResult takeGapResult( MfemNodalQuantityData& quantities,
                                        [[maybe_unused]] const LagrangeMultiplierEnforcement& enforcement )
{
  ContactFormulationResult result;
  result.gap.reset( quantities.weighted_gap.release() );
  result.dg_dx.reset( quantities.weighted_gap_jacobian.release() );
  return result;
}

}  // namespace

template <typename QuadratureRule, typename PenaltyStiffness>
ContactFormulationResult QuadraturePointEnergyMortar<QuadratureRule, PenaltyStiffness>::computeNodalForces(
    MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  const RealT penalty = std::visit(
      [&]( const auto& enforcement ) { return enforcement.prepare( mesh_data_, mesh1, mesh2 ); }, enforcement_ );
  const PairFunction function{ quadrature_, penalty, EnergyMortarPenalty{ gap_ } };
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  const int allocator_id = pairs.getAllocatorID();
  PairEnergyHessianBuffer<PairFunction::dof_count> contributions( pairs.size(), allocator_id );
  evaluatePairEnergiesAndHessians( execution_mode_, pairs.view(), mesh1, mesh2, function, contributions.view() );
  checkPairQuadratureResults( execution_mode_, contributions.quadratureResults() );

  MfemPairAssembler assembler( mesh_data_, jacobian_data_, mesh1, mesh2, pairs.size() );
  const auto contribution_view = contributions.view();
  assembler.addPairs<PairFunction::dof_count>(
      execution_mode_, pairs,
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) { return contribution_view.derivatives.gradient( pair, row ); },
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
        return contribution_view.derivatives.hessian( pair, row, column );
      } );
  auto result = assembler.assemble();
  result.energy = GlobalScalarAssembler::sum(
      execution_mode_, pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) { return contribution_view.energies[pair]; },
      result.force->GetComm() );
  return result;
}

template <typename QuadratureRule, typename PenaltyStiffness>
ContactFormulationResult NodalEnergyMortar<QuadratureRule, PenaltyStiffness>::computeNodalGaps(
    MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  const PairFunction function{ quadrature_, EnergyMortarNodal{ gap_ } };
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  PairValueJacobianBuffer<PairFunction::value_count, PairFunction::dof_count> values( pairs.size(),
                                                                                      pairs.getAllocatorID() );
  evaluatePairValuesAndJacobians( execution_mode_, pairs.view(), mesh1, mesh2, function, values.view() );
  checkPairQuadratureResults( execution_mode_, values.quadratureResults() );
  MfemNodalAssembler assembler( mesh_data_, submesh_data_, jacobian_data_, mesh1, mesh2, pairs.size() );
  auto quantities = assembleNodalQuantities<PairFunction>( execution_mode_, pairs, values, assembler );
  return std::visit( [&]( const auto& enforcement ) { return takeGapResult( quantities, enforcement ); },
                     enforcement_ );
}

template <typename QuadratureRule, typename PenaltyStiffness>
ContactFormulationResult NodalEnergyMortar<QuadratureRule, PenaltyStiffness>::computeNodalForces(
    MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  constexpr IndexT node_count = PairFunction::node_count;
  const PairFunction function{ quadrature_, EnergyMortarNodal{ gap_ } };
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  const int allocator_id = pairs.getAllocatorID();
  PairValueJacobianBuffer<PairFunction::value_count, PairFunction::dof_count> values( pairs.size(), allocator_id );
  evaluatePairValuesAndJacobians( execution_mode_, pairs.view(), mesh1, mesh2, function, values.view() );
  checkPairQuadratureResults( execution_mode_, values.quadratureResults() );
  MfemNodalAssembler assembler( mesh_data_, submesh_data_, jacobian_data_, mesh1, mesh2, pairs.size() );
  auto quantities = assembleNodalQuantities<PairFunction>( execution_mode_, pairs, values, assembler );

  const auto make_vector = [&] {
    shared::ParVector vector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
    vector.fill( 0.0 );
    return vector;
  };
  const bool penalty_enforcement_selected =
      std::holds_alternative<RegisteredPenaltyEnforcement<PenaltyStiffness>>( enforcement_ );
  auto gap_weights = make_vector();
  auto area_weights = make_vector();
  shared::ParVector gap_gap, gap_area, area_area;
  RealT energy = 0.0;
  if ( penalty_enforcement_selected ) {
    const RealT stiffness =
        std::get<RegisteredPenaltyEnforcement<PenaltyStiffness>>( enforcement_ ).prepare( mesh_data_, mesh1, mesh2 );
    gap_gap = make_vector();
    gap_area = make_vector();
    area_area = make_vector();
    Array1D<RealT> nodal_energy( gap_weights.size(), gap_weights.size(), allocator_id );
    const bool use_device = isOnDevice( execution_mode_ );
    const auto* weighted_gap = quantities.weighted_gap.get().Read( use_device );
    const auto* area = quantities.area.get().Read( use_device );
    auto* gap_weight = gap_weights.get().Write( use_device );
    auto* area_weight = area_weights.get().Write( use_device );
    auto* gap_gap_data = gap_gap.get().Write( use_device );
    auto* gap_area_data = gap_area.get().Write( use_device );
    auto* area_area_data = area_area.get().Write( use_device );
    auto* nodal_energy_data = nodal_energy.data();
    forAllExec( execution_mode_, gap_weights.size(), [=] TRIBOL_HOST_DEVICE( IndexT node ) {
      const auto partials = EnergyMortarNodal<OffsetGap<OpposingNormalGap>>::penaltyPartials(
          weighted_gap[node], area[node], stiffness, area_tolerance );
      nodal_energy_data[node] = partials.energy;
      gap_weight[node] = partials.gap;
      area_weight[node] = partials.area;
      gap_gap_data[node] = partials.gap_gap;
      gap_area_data[node] = partials.gap_area;
      area_area_data[node] = partials.area_area;
    } );
    energy = GlobalScalarAssembler::sum(
        execution_mode_, gap_weights.size(), [=] TRIBOL_HOST_DEVICE( IndexT node ) { return nodal_energy_data[node]; },
        gap_weights.get().GetComm() );
  } else {
    gap_weights = assembler.getSubmeshPressureTrueDofs();
    if ( gap_weights.size() != quantities.weighted_gap.size() ) {
      throw std::invalid_argument( "LM values must match dual true DOFs." );
    }
    energy = gap_weights.dot( quantities.weighted_gap );
  }

  auto force = gap_weights * quantities.weighted_gap_jacobian;
  if ( penalty_enforcement_selected ) {
    force += area_weights * quantities.area_jacobian;
  }
  const auto redecomp_gap_weights = assembler.transferToRedecomp( gap_weights.get() );
  const auto redecomp_area_weights =
      penalty_enforcement_selected ? assembler.transferToRedecomp( area_weights.get() ) : mfem::GridFunction{};
  Array1D<RealT> pair_weights( PairFunction::value_count * pairs.size(), PairFunction::value_count * pairs.size(),
                               allocator_id );
  const auto pair_view = pairs.view();
  const auto connectivity = mesh1.getConnectivity();
  const auto* gap_data = redecomp_gap_weights.Read( isOnDevice( execution_mode_ ) );
  const auto* area_data =
      penalty_enforcement_selected ? redecomp_area_weights.Read( isOnDevice( execution_mode_ ) ) : nullptr;
  auto* weight_data = pair_weights.data();
  const IndexT pair_count = pairs.size();
  forAllExec( execution_mode_, pair_count, [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto nodes = connectivity( pair_view[pair_index].element_id1_ );
    for ( IndexT node = 0; node < node_count; ++node ) {
      weight_data[node * pair_count + pair_index] = gap_data[nodes[node]];
      weight_data[( node_count + node ) * pair_count + pair_index] = area_data ? area_data[nodes[node]] : 0.0;
    }
  } );

  PairHessianBuffer<PairFunction::dof_count> hessians( pair_count, allocator_id );
  evaluatePairWeightedHessians(
      execution_mode_, pairs.view(), mesh1, mesh2, function,
      [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
        MdArray<RealT, PairFunction::value_count> weights{};
        for ( IndexT value = 0; value < PairFunction::value_count; ++value ) {
          weights[value] = weight_data[value * pair_count + pair_index];
        }
        return weights;
      },
      hessians.view() );
  const auto hessian_view = hessians.view();
  assembler.addPairJacobians<PairFunction::dof_count>(
      execution_mode_, pairs, [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
        return hessian_view.hessian( pair, row, column );
      } );
  auto jacobian = assembler.assembleHessians();
  if ( penalty_enforcement_selected ) {
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
  if ( !penalty_enforcement_selected ) {
    auto df_dp = quantities.weighted_gap_jacobian.transpose();
    result.df_dp.reset( df_dp.release() );
  }
  auto gaps =
      std::visit( [&]( const auto& enforcement ) { return takeGapResult( quantities, enforcement ); }, enforcement_ );
  result.gap = std::move( gaps.gap );
  result.dg_dx = std::move( gaps.dg_dx );
  result.energy = energy;
  result.force.reset( force.release() );
  result.df_dx.reset( jacobian.release() );
  result.pressure.reset( gap_weights.release() );
  return result;
}

#define TRIBOL_INSTANTIATE_ENERGY_MORTAR( Rule, Penalty )    \
  template class QuadraturePointEnergyMortar<Rule, Penalty>; \
  template class NodalEnergyMortar<Rule, Penalty>

using EnergyRule1 = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>;
using EnergyRule2 = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>;
using EnergyRule3 = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>;
TRIBOL_INSTANTIATE_ENERGY_MORTAR( EnergyRule1, AveragePenaltyStiffness );
TRIBOL_INSTANTIATE_ENERGY_MORTAR( EnergyRule1, SeriesPenaltyStiffness );
TRIBOL_INSTANTIATE_ENERGY_MORTAR( EnergyRule2, AveragePenaltyStiffness );
TRIBOL_INSTANTIATE_ENERGY_MORTAR( EnergyRule2, SeriesPenaltyStiffness );
TRIBOL_INSTANTIATE_ENERGY_MORTAR( EnergyRule3, AveragePenaltyStiffness );
TRIBOL_INSTANTIATE_ENERGY_MORTAR( EnergyRule3, SeriesPenaltyStiffness );

#undef TRIBOL_INSTANTIATE_ENERGY_MORTAR

}  // namespace tribol
#endif
