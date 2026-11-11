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
#include "tribol/mesh/MfemFieldAssembler.hpp"
#include "tribol/mesh/MfemPairAssembler.hpp"
#include "tribol/physics/GlobalScalarAssembler.hpp"
#include "tribol/physics/PairContribution.hpp"
#include "tribol/physics/PairKernel.hpp"

namespace tribol {
namespace {

/** @brief Energy Mortar's assembled weighted gaps and tributary areas, before activation. */
struct EnergyMortarNodalQuantities {
  AssembledField weighted_gap;  ///< Weighted gaps and their coordinate Jacobian.
  AssembledField area;          ///< Tributary areas and their coordinate Jacobian.
};

/**
 * @brief Assemble Energy Mortar's two scalar fields in one batch.
 * @tparam Function Pair function returning weighted gaps followed by areas.
 * @param execution_mode Host or device execution policy.
 * @param pairs Current pair list.
 * @param buffer Pair values and coordinate gradients.
 * @param assembler Scalar field accumulator configured for two fields.
 * @return Unmodified weighted gaps, areas, and their coordinate Jacobians.
 */
template <typename Function>
EnergyMortarNodalQuantities assembleNodalQuantities(
    ExecutionMode execution_mode, const ArrayT<InterfacePair>& pairs,
    PairValueJacobianBuffer<Function::value_count, Function::dof_count>& buffer, MfemFieldAssembler& assembler )
{
  constexpr IndexT node_count = Function::node_count;
  constexpr IndexT dof_count = Function::dof_count;
  static_assert( Function::value_count == 2 * node_count );
  const auto view = buffer.view();
  assembler.addPairs<2, node_count, dof_count>(
      execution_mode, pairs,
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT field, IndexT node ) {
        return view.has_active_points[pair] ? view.value( pair, field * node_count + node ) : 0.0;
      },
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT field, IndexT node, IndexT coordinate ) {
        return view.has_active_points[pair] ? view.jacobian( pair, field * node_count + node, coordinate ) : 0.0;
      } );
  auto fields = assembler.assemble();
  return { std::move( fields[0] ), std::move( fields[1] ) };
}

/**
 * @brief Normalize penalty gaps and remove positive-gap derivative rows.
 * @tparam Inputs Penalty input reader.
 * @param quantities Assembled unmodified gap and area fields.
 * @param enforcement Selected penalty equation.
 * @param area_tolerance Smallest supported area magnitude for normalization.
 * @return Normalized, nonpositive gaps and the weighted-gap coordinate Jacobian.
 */
template <typename Inputs>
ContactFormulationResult takeGapResult( EnergyMortarNodalQuantities& quantities,
                                        [[maybe_unused]] const PenaltyEnforcement<Inputs>& enforcement,
                                        RealT area_tolerance )
{
  auto normalized_gap = quantities.weighted_gap.values.divide( quantities.area.values, area_tolerance );
  mfem::Array<int> open_nodes;
  for ( int node = 0; node < quantities.weighted_gap.values.size(); ++node ) {
    if ( quantities.weighted_gap.values[node] > 0.0 ) {
      normalized_gap[node] = 0.0;
      open_nodes.Append( node );
    }
  }
  quantities.weighted_gap.dvalues_dx.eliminateRows( open_nodes );
  ContactFormulationResult result;
  result.gap.reset( normalized_gap.release() );
  result.dg_dx.reset( quantities.weighted_gap.dvalues_dx.release() );
  return result;
}

/**
 * @brief Return the weighted LM constraint without normalization or gap activation.
 * @param quantities Assembled gap and area fields.
 * @param enforcement Selected LM equation.
 * @param area_tolerance Unused normalization tolerance.
 * @return Weighted gaps and their coordinate Jacobian.
 */
ContactFormulationResult takeGapResult( EnergyMortarNodalQuantities& quantities,
                                        [[maybe_unused]] const LagrangeMultiplierEnforcement& enforcement,
                                        [[maybe_unused]] RealT area_tolerance )
{
  ContactFormulationResult result;
  result.gap.reset( quantities.weighted_gap.values.release() );
  result.dg_dx.reset( quantities.weighted_gap.dvalues_dx.release() );
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
  MfemFieldAssembler assembler( mesh_data_, submesh_data_, jacobian_data_, mesh1, mesh2, 2, pairs.size() );
  auto quantities = assembleNodalQuantities<PairFunction>( execution_mode_, pairs, values, assembler );
  return std::visit(
      [&]( const auto& enforcement ) { return takeGapResult( quantities, enforcement, area_tolerance ); },
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
  MfemFieldAssembler assembler( mesh_data_, submesh_data_, jacobian_data_, mesh1, mesh2, 2, pairs.size() );
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
    const auto* weighted_gap = quantities.weighted_gap.values.get().Read( use_device );
    const auto* area = quantities.area.values.get().Read( use_device );
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
    submesh_data_.GetSubmeshPressure().GetTrueDofs( gap_weights.get() );
    if ( gap_weights.size() != quantities.weighted_gap.values.size() ) {
      throw std::invalid_argument( "LM values must match dual true DOFs." );
    }
    energy = gap_weights.dot( quantities.weighted_gap.values );
  }

  auto force = gap_weights * quantities.weighted_gap.dvalues_dx;
  if ( penalty_enforcement_selected ) {
    force += area_weights * quantities.area.dvalues_dx;
  }
  const auto redecomp_gap_weights = submesh_data_.TransferTrueDofsToRedecomp( gap_weights.get() );
  const auto redecomp_area_weights = penalty_enforcement_selected
                                         ? submesh_data_.TransferTrueDofsToRedecomp( area_weights.get() )
                                         : mfem::GridFunction{};
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
  MfemPairAssembler hessian_assembler( mesh_data_, jacobian_data_, mesh1, mesh2, pair_count, false );
  hessian_assembler.addPairJacobians<PairFunction::dof_count>(
      execution_mode_, pairs, [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
        return hessian_view.hessian( pair, row, column );
      } );
  auto jacobian = hessian_assembler.assembleJacobian();
  if ( penalty_enforcement_selected ) {
    const auto make_diagonal = [&]( const shared::ParVector& values ) {
      const auto& fes = submesh_data_.GetSubmeshFESpace();
      return shared::ParSparseMat::diagonalMatrix( fes.GetComm(), fes.GlobalTrueVSize(), fes.GetTrueDofOffsets(),
                                                   values.get() );
    };
    const auto gg = make_diagonal( gap_gap );
    const auto ga = make_diagonal( gap_area );
    const auto aa = make_diagonal( area_area );
    jacobian += shared::ParSparseMat::rap( gg, quantities.weighted_gap.dvalues_dx );
    const auto mixed = shared::ParSparseMat::rap( quantities.weighted_gap.dvalues_dx, ga, quantities.area.dvalues_dx );
    jacobian += mixed;
    jacobian += mixed.transpose();
    jacobian += shared::ParSparseMat::rap( aa, quantities.area.dvalues_dx );
  }

  ContactFormulationResult result;
  if ( !penalty_enforcement_selected ) {
    auto df_dp = quantities.weighted_gap.dvalues_dx.transpose();
    result.df_dp.reset( df_dp.release() );
  }
  auto gaps =
      std::visit( [&]( const auto& enforcement ) { return takeGapResult( quantities, enforcement, area_tolerance ); },
                  enforcement_ );
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
