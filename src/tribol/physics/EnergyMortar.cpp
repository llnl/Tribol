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
#include "tribol/assembly/MfemFieldAssembler.hpp"
#include "tribol/assembly/MfemPairAssembler.hpp"
#include "tribol/common/GlobalScalarAssembler.hpp"
#include "tribol/assembly/PairContribution.hpp"
#include "tribol/assembly/PairEvaluationAssembly.hpp"
#include "tribol/assembly/PairKernel.hpp"

namespace tribol {
namespace {

/** @brief Energy Mortar's assembled weighted gaps and tributary areas, before activation. */
struct EnergyMortarNodalQuantities {
  AssembledField weighted_gap;  ///< Weighted gaps and their coordinate Jacobian.
  AssembledField area;          ///< Tributary areas and their coordinate Jacobian.
};

/**
 * @brief Evaluate and assemble weighted gaps, tributary areas, and their coordinate Jacobians.
 * @tparam Function Pair function returning weighted gaps followed by areas.
 * @param execution_mode Host or device execution mode.
 * @param pairs Current accepted pair list.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param function Fixed pair-function configuration.
 * @param mesh_data Current displacement transfer data.
 * @param submesh_data Scalar field transfer data.
 * @param jacobian_data Parent and scalar-field Jacobian transfer data.
 * @return Unmodified weighted gaps, areas, and their coordinate Jacobians, including allocated zeros for empty pairs.
 * @throws std::invalid_argument Geometry, execution mode, or assembly dimensions are invalid; no result is returned.
 * @throws std::length_error Quadrature capacity or the assembly index range is exceeded; no result is returned.
 */
template <typename Function>
EnergyMortarNodalQuantities evaluateAndAssembleNodalQuantities( ExecutionMode execution_mode,
                                                                const ArrayT<InterfacePair>& pairs,
                                                                MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                                const Function& function, const MfemMeshData& mesh_data,
                                                                const MfemSubmeshData& submesh_data,
                                                                const MfemJacobianData& jacobian_data )
{
  constexpr IndexT node_count = Function::node_count;
  constexpr IndexT dof_count = Function::dof_count;
  static_assert( Function::value_count == 2 * node_count );
  PairValueJacobianBuffer<Function::value_count, dof_count> buffer( pairs.size(), pairs.getAllocatorID() );
  const auto view = buffer.view();
  evaluatePairValuesAndJacobians( execution_mode, pairs.view(), mesh1, mesh2, function, view );
  checkPairQuadratureResults( execution_mode, buffer.quadratureResults() );
  MfemFieldAssembler assembler( mesh_data, submesh_data, jacobian_data, mesh1, mesh2, 2, pairs.size() );
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
 * @brief Assemble the coordinate Hessian of fixed nodal weights contracted with pair gap and area integrals.
 * @tparam Function Pair function returning weighted gaps followed by areas.
 * @tparam WeightReader Device-callable node reader returning gap and area weights in that order.
 * @param execution_mode Host or device execution mode.
 * @param pairs Current accepted pair list.
 * @param mesh1 Current first mesh, whose scalar finite-element nodes carry the weights.
 * @param mesh2 Current second mesh.
 * @param function Fixed pair-function configuration.
 * @param mesh_data Current displacement transfer data.
 * @param jacobian_data Current Jacobian transfer data.
 * @param read_weights Callable with signature MdArray<RealT, 2>(IndexT) reading redecomposed scalar node weights.
 * @return Owned parent coordinate Hessian; an empty pair list returns an allocated zero matrix.
 * @throws std::invalid_argument Execution mode or assembly dimensions are invalid; no matrix is returned.
 * @throws std::length_error Assembly storage exceeds the supported index range; no matrix is returned.
 */
template <typename Function, typename WeightReader>
shared::ParSparseMat assembleWeightedNodalHessian( ExecutionMode execution_mode, const ArrayT<InterfacePair>& pairs,
                                                   MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                   const Function& function, const MfemMeshData& mesh_data,
                                                   const MfemJacobianData& jacobian_data, WeightReader read_weights )
{
  constexpr IndexT node_count = Function::node_count;
  static_assert( Function::value_count == 2 * node_count );
  const IndexT pair_count = pairs.size();
  const int allocator_id = pairs.getAllocatorID();
  Array1D<RealT> pair_weights( Function::value_count * pair_count, Function::value_count * pair_count, allocator_id );
  const auto pair_view = pairs.view();
  const auto connectivity = mesh1.getConnectivity();
  auto* weight_data = pair_weights.data();
  forAllExec( execution_mode, pair_count, [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto nodes = connectivity( pair_view[pair_index].element_id1_ );
    for ( IndexT node = 0; node < node_count; ++node ) {
      const auto weights = read_weights( nodes[node] );
      weight_data[node * pair_count + pair_index] = weights[0];
      weight_data[( node_count + node ) * pair_count + pair_index] = weights[1];
    }
  } );

  PairHessianBuffer<Function::dof_count> hessians( pair_count, allocator_id );
  evaluatePairWeightedJacobiansAndHessians(
      execution_mode, pairs.view(), mesh1, mesh2, function,
      [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
        MdArray<RealT, Function::value_count> weights{};
        for ( IndexT value = 0; value < Function::value_count; ++value ) {
          weights[value] = weight_data[value * pair_count + pair_index];
        }
        return weights;
      },
      hessians.view() );
  const auto hessian_view = hessians.view();
  MfemPairAssembler assembler( mesh_data, jacobian_data, mesh1, mesh2, pair_count, false );
  assembler.addPairJacobians<Function::dof_count>( execution_mode, pairs,
                                                   [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
                                                     return hessian_view.hessian( pair, row, column );
                                                   } );
  return assembler.assembleJacobian();
}

/**
 * @brief Normalize penalty gaps and remove positive-gap derivative rows.
 * @param quantities Assembled unmodified gap and area fields.
 * @param area_tolerance Smallest supported area magnitude for normalization.
 * @return Normalized, nonpositive gaps and the weighted-gap coordinate Jacobian.
 */
ContactFormulationResult takePenaltyGapResult( EnergyMortarNodalQuantities& quantities, RealT area_tolerance )
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
 * @return Weighted gaps and their coordinate Jacobian.
 */
ContactFormulationResult takeLagrangeMultiplierGapResult( EnergyMortarNodalQuantities& quantities )
{
  ContactFormulationResult result;
  result.gap.reset( quantities.weighted_gap.values.release() );
  result.dg_dx.reset( quantities.weighted_gap.dvalues_dx.release() );
  return result;
}

}  // namespace

template <typename QuadratureRule>
ContactFormulationResult QuadraturePointEnergyMortar<QuadratureRule>::computeNodalForces( MeshData::Viewer mesh1,
                                                                                          MeshData::Viewer mesh2 ) const
{
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  return detail::evaluateAndAssembleEnergyPairs( execution_mode_, pairs, mesh1, mesh2, function_, mesh_data_,
                                                 jacobian_data_ );
}

template <typename QuadratureRule>
ContactFormulationResult NodalEnergyMortar<QuadratureRule>::computeNodalGaps( MeshData::Viewer mesh1,
                                                                              MeshData::Viewer mesh2 ) const
{
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  auto quantities = evaluateAndAssembleNodalQuantities( execution_mode_, pairs, mesh1, mesh2, function_, mesh_data_,
                                                        submesh_data_, jacobian_data_ );
  if ( std::holds_alternative<PenaltyEnforcement>( enforcement_ ) ) {
    return takePenaltyGapResult( quantities, area_tolerance );
  }
  return takeLagrangeMultiplierGapResult( quantities );
}

template <typename QuadratureRule>
ContactFormulationResult NodalEnergyMortar<QuadratureRule>::computeNodalForces( MeshData::Viewer mesh1,
                                                                                MeshData::Viewer mesh2 ) const
{
  if ( const auto* penalty = std::get_if<PenaltyEnforcement>( &enforcement_ ) ) {
    return computePenaltyForces( mesh1, mesh2, penalty->stiffness() );
  }
  return computeLagrangeMultiplierForces( mesh1, mesh2 );
}

template <typename QuadratureRule>
ContactFormulationResult NodalEnergyMortar<QuadratureRule>::computePenaltyForces( MeshData::Viewer mesh1,
                                                                                  MeshData::Viewer mesh2,
                                                                                  RealT stiffness ) const
{
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  auto quantities = evaluateAndAssembleNodalQuantities( execution_mode_, pairs, mesh1, mesh2, function_, mesh_data_,
                                                        submesh_data_, jacobian_data_ );
  const auto make_vector = [&] {
    shared::ParVector vector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
    vector.fill( 0.0 );
    return vector;
  };
  auto gap_weights = make_vector();
  auto area_weights = make_vector();
  auto gap_gap = make_vector();
  auto gap_area = make_vector();
  auto area_area = make_vector();
  const int allocator_id = pairs.getAllocatorID();
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
    const auto partials =
        EnergyMortarNodalPartials::evaluate( weighted_gap[node], area[node], stiffness, area_tolerance );
    nodal_energy_data[node] = partials.energy;
    gap_weight[node] = partials.gap;
    area_weight[node] = partials.area;
    gap_gap_data[node] = partials.gap_gap;
    gap_area_data[node] = partials.gap_area;
    area_area_data[node] = partials.area_area;
  } );
  const RealT energy = GlobalScalarAssembler::sum(
      execution_mode_, gap_weights.size(), [=] TRIBOL_HOST_DEVICE( IndexT node ) { return nodal_energy_data[node]; },
      gap_weights.get().GetComm() );
  auto force = gap_weights * quantities.weighted_gap.dvalues_dx;
  force += area_weights * quantities.area.dvalues_dx;

  const auto redecomp_gap_weights = submesh_data_.TransferTrueDofsToRedecomp( gap_weights.get() );
  const auto redecomp_area_weights = submesh_data_.TransferTrueDofsToRedecomp( area_weights.get() );
  const auto* gap_data = redecomp_gap_weights.Read( use_device );
  const auto* area_data = redecomp_area_weights.Read( use_device );
  auto jacobian = assembleWeightedNodalHessian(
      execution_mode_, pairs, mesh1, mesh2, function_, mesh_data_, jacobian_data_,
      [=] TRIBOL_HOST_DEVICE( IndexT node ) { return MdArray<RealT, 2>{ gap_data[node], area_data[node] }; } );
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

  auto result = takePenaltyGapResult( quantities, area_tolerance );
  result.energy = energy;
  result.force.reset( force.release() );
  result.df_dx.reset( jacobian.release() );
  result.pressure.reset( gap_weights.release() );
  return result;
}

template <typename QuadratureRule>
ContactFormulationResult NodalEnergyMortar<QuadratureRule>::computeLagrangeMultiplierForces(
    MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  auto quantities = evaluateAndAssembleNodalQuantities( execution_mode_, pairs, mesh1, mesh2, function_, mesh_data_,
                                                        submesh_data_, jacobian_data_ );
  shared::ParVector pressure( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_data_.GetSubmeshPressure().GetTrueDofs( pressure.get() );
  if ( pressure.size() != quantities.weighted_gap.values.size() ) {
    throw std::invalid_argument( "LM values must match dual true DOFs." );
  }
  const RealT energy = pressure.dot( quantities.weighted_gap.values );
  auto force = pressure * quantities.weighted_gap.dvalues_dx;
  const auto redecomp_pressure = submesh_data_.TransferTrueDofsToRedecomp( pressure.get() );
  const auto* pressure_data = redecomp_pressure.Read( isOnDevice( execution_mode_ ) );
  auto jacobian = assembleWeightedNodalHessian(
      execution_mode_, pairs, mesh1, mesh2, function_, mesh_data_, jacobian_data_,
      [=] TRIBOL_HOST_DEVICE( IndexT node ) { return MdArray<RealT, 2>{ pressure_data[node], 0.0 }; } );
  auto df_dp = quantities.weighted_gap.dvalues_dx.transpose();

  auto result = takeLagrangeMultiplierGapResult( quantities );
  result.energy = energy;
  result.force.reset( force.release() );
  result.df_dx.reset( jacobian.release() );
  result.df_dp.reset( df_dp.release() );
  result.pressure.reset( pressure.release() );
  return result;
}

#define TRIBOL_INSTANTIATE_ENERGY_MORTAR( Rule )    \
  template class QuadraturePointEnergyMortar<Rule>; \
  template class NodalEnergyMortar<Rule>

using EnergyRule1 = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>;
using EnergyRule2 = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>;
using EnergyRule3 = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>;
TRIBOL_INSTANTIATE_ENERGY_MORTAR( EnergyRule1 );
TRIBOL_INSTANTIATE_ENERGY_MORTAR( EnergyRule2 );
TRIBOL_INSTANTIATE_ENERGY_MORTAR( EnergyRule3 );

#undef TRIBOL_INSTANTIATE_ENERGY_MORTAR

}  // namespace tribol
#endif
