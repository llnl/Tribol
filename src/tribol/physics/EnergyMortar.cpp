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
#include "tribol/assembly/MfemVectorAssembler.hpp"
#include "tribol/assembly/MfemMatrixAssembler.hpp"
#include "tribol/common/GlobalScalarAssembler.hpp"
#include "tribol/assembly/PairContribution.hpp"
#include "tribol/assembly/PairEvaluationAssembly.hpp"
#include "tribol/assembly/PairKernel.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {
namespace {

constexpr RealT nodal_area_tolerance = 1.0e-14;  ///< Minimum supported tributary-area magnitude for penalty contact.

/** @brief Energy Mortar's assembled weighted gaps and tributary areas, before activation. */
struct EnergyMortarNodalQuantities {
  AssembledField weighted_gap;  ///< Weighted gaps and their coordinate Jacobian.
  AssembledField area;          ///< Tributary areas and their coordinate Jacobian.
};

/**
 * @brief Evaluate and assemble weighted gaps, tributary areas, and their coordinate Jacobians.
 * @tparam Function Pair function returning weighted gaps followed by areas.
 * @tparam Exec Concrete execution mode for evaluation and numerical assembly.
 * @param pairs Current accepted pair list.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param function Fixed pair-function configuration.
 * @param mesh_data Current displacement transfer data.
 * @param submesh_data Scalar field transfer data.
 * @param jacobian_data Parent and scalar-field Jacobian transfer data.
 * @return Unmodified weighted gaps, areas, and their coordinate Jacobians, including allocated zeros for empty pairs.
 * @throws std::invalid_argument Geometry or assembly dimensions are invalid; no result is returned.
 * @throws std::length_error Quadrature capacity or the assembly index range is exceeded; no result is returned.
 */
template <ExecutionMode Exec, typename Function>
EnergyMortarNodalQuantities evaluateAndAssembleNodalQuantities( const ArrayT<InterfacePair>& pairs,
                                                                MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                                const Function& function, const MfemMeshData& mesh_data,
                                                                const MfemSubmeshData& submesh_data,
                                                                const MfemJacobianData& jacobian_data )
{
  constexpr IndexT num_nodes = Function::num_nodes;
  constexpr IndexT num_coords = Function::num_coords;
  static_assert( Function::num_values == 2 * num_nodes );
  auto buffer = evaluatePairValuesAndJacobians<Exec>( pairs, mesh1, mesh2, function );
  const auto contributions_view = buffer.view();
  checkPairQuadratureResults<Exec>( buffer.quadratureResults() );
  constexpr IndexT num_fields = 2;
  std::vector<MfemVectorAssembler> vectors;
  std::vector<MfemMatrixAssembler> matrices;
  vectors.reserve( num_fields );
  matrices.reserve( num_fields );
  for ( IndexT field = 0; field < num_fields; ++field ) {
    vectors.push_back( detail::makeNodalVectorAssembler( submesh_data, mesh1 ) );
    matrices.push_back( detail::makeNodalCoordinateMatrixAssembler( mesh_data, submesh_data, jacobian_data, mesh1,
                                                                    mesh2, pairs.size() ) );
  }
  detail::addNodalValuesAndJacobians<Exec, num_fields, num_nodes, num_coords>(
      pairs, vectors, matrices,
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT field, IndexT node ) {
        return contributions_view.has_active_points[pair] ? contributions_view.value( pair, field * num_nodes + node )
                                                          : 0.0;
      },
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT field, IndexT node, IndexT coordinate ) {
        return contributions_view.has_active_points[pair]
                   ? contributions_view.jacobian( pair, field * num_nodes + node, coordinate )
                   : 0.0;
      } );
  return { { vectors[0].assemble(), matrices[0].assemble() }, { vectors[1].assemble(), matrices[1].assemble() } };
}

/**
 * @brief Assemble the coordinate Hessian of fixed nodal weights contracted with pair gap and area integrals.
 * @tparam Function Pair function returning weighted gaps followed by areas.
 * @tparam WeightReader Device-callable node reader returning gap and area weights in that order.
 * @tparam Exec Concrete execution mode for evaluation and numerical assembly.
 * @param pairs Current accepted pair list.
 * @param mesh1 Current first mesh, whose scalar finite-element nodes carry the weights.
 * @param mesh2 Current second mesh.
 * @param function Fixed pair-function configuration.
 * @param mesh_data Current displacement transfer data.
 * @param jacobian_data Current Jacobian transfer data.
 * @param read_weights Callable with signature MdArray<RealT, 2>(IndexT) reading redecomposed scalar node weights.
 * @return Owned parent coordinate Hessian; an empty pair list returns an allocated zero matrix.
 * @throws std::invalid_argument Assembly dimensions are invalid; no matrix is returned.
 * @throws std::length_error Assembly storage exceeds the supported index range; no matrix is returned.
 */
template <ExecutionMode Exec, typename Function, typename WeightReader>
shared::ParSparseMat assembleWeightedNodalHessian( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                   MeshData::Viewer mesh2, const Function& function,
                                                   const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data,
                                                   WeightReader read_weights )
{
  constexpr IndexT num_nodes = Function::num_nodes;
  static_assert( Function::num_values == 2 * num_nodes );
  const IndexT num_pairs = pairs.size();
  const auto pairs_view = pairs.view();
  const auto connectivity = mesh1.getConnectivity();
  auto hessians = evaluatePairWeightedJacobiansAndHessians<Exec>(
      pairs, mesh1, mesh2, function, [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
        MdArray<RealT, Function::num_values> weights{};
        const auto nodes = connectivity( pairs_view[pair_index].element_id1_ );
        for ( IndexT node = 0; node < num_nodes; ++node ) {
          const auto nodal_weights = read_weights( nodes[node] );
          weights[node] = nodal_weights[0];
          weights[num_nodes + node] = nodal_weights[1];
        }
        return weights;
      } );
  const auto hessian_view = hessians.view();
  auto assembler = detail::makeCoordinateMatrixAssembler( mesh_data, jacobian_data, mesh1, mesh2, num_pairs );
  assembler.addPairs<Exec, Function::num_coords / 2, Function::num_coords / 2, 4>(
      pairs, [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
        return hessian_view.hessian( pair, row, column );
      } );
  return assembler.assemble();
}

/**
 * @brief Normalize penalty gaps and remove positive-gap derivative rows.
 * @param quantities Assembled unmodified gap and area fields.
 * @param enforcement Select normalized compressive penalty gaps.
 * @return Normalized, nonpositive gaps and the weighted-gap coordinate Jacobian.
 */
ContactFormulationResult takeGapResult( EnergyMortarNodalQuantities& quantities,
                                        [[maybe_unused]] const PenaltyEnforcement& enforcement )
{
  auto normalized_gap = quantities.weighted_gap.values.divide( quantities.area.values, nodal_area_tolerance );
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
 * @param enforcement Select signed weighted LM constraints.
 * @return Weighted gaps and their coordinate Jacobian.
 */
ContactFormulationResult takeGapResult( EnergyMortarNodalQuantities& quantities,
                                        [[maybe_unused]] LagrangeMultiplierEnforcement enforcement )
{
  ContactFormulationResult result;
  result.gap.reset( quantities.weighted_gap.values.release() );
  result.dg_dx.reset( quantities.weighted_gap.dvalues_dx.release() );
  return result;
}

}  // namespace

template <typename PairRuleGenerator, ExecutionMode Exec>
ContactFormulationResult QuadraturePointEnergyMortar<PairRuleGenerator, Exec>::computeNodalForces(
    MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  return detail::evaluateAndAssembleEnergyPairs<Exec>( pairs, mesh1, mesh2, function_, mesh_data_, jacobian_data_ );
}

template <typename PairRuleGenerator, ExecutionMode Exec, typename Enforcement>
ContactFormulationResult NodalEnergyMortar<PairRuleGenerator, Exec, Enforcement>::computeNodalGaps(
    MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  auto quantities = evaluateAndAssembleNodalQuantities<Exec>( pairs, mesh1, mesh2, function_, mesh_data_, submesh_data_,
                                                              jacobian_data_ );
  return takeGapResult( quantities, enforcement_ );
}

template <typename PairRuleGenerator, ExecutionMode Exec, typename Enforcement>
ContactFormulationResult NodalEnergyMortar<PairRuleGenerator, Exec, Enforcement>::computeNodalForces(
    MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  return computeForces( mesh1, mesh2, enforcement_ );
}

template <typename PairRuleGenerator, ExecutionMode Exec, typename Enforcement>
ContactFormulationResult NodalEnergyMortar<PairRuleGenerator, Exec, Enforcement>::computeForces(
    MeshData::Viewer mesh1, MeshData::Viewer mesh2, const PenaltyEnforcement& penalty ) const
{
  const RealT stiffness = penalty.stiffness();
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  auto quantities = evaluateAndAssembleNodalQuantities<Exec>( pairs, mesh1, mesh2, function_, mesh_data_, submesh_data_,
                                                              jacobian_data_ );
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
  const bool use_device = isOnDevice( Exec );
  const auto* weighted_gap = quantities.weighted_gap.values.get().Read( use_device );
  const auto* area = quantities.area.values.get().Read( use_device );
  auto* gap_weight = gap_weights.get().Write( use_device );
  auto* area_weight = area_weights.get().Write( use_device );
  auto* gap_gap_data = gap_gap.get().Write( use_device );
  auto* gap_area_data = gap_area.get().Write( use_device );
  auto* area_area_data = area_area.get().Write( use_device );
  auto* nodal_energy_data = nodal_energy.data();
  forAllExec<Exec>( gap_weights.size(), [=] TRIBOL_HOST_DEVICE( IndexT node ) {
    const auto partials =
        EnergyMortarNodalPartials::evaluate( weighted_gap[node], area[node], stiffness, nodal_area_tolerance );
    nodal_energy_data[node] = partials.energy;
    gap_weight[node] = partials.gap;
    area_weight[node] = partials.area;
    gap_gap_data[node] = partials.gap_gap;
    gap_area_data[node] = partials.gap_area;
    area_area_data[node] = partials.area_area;
  } );
  const RealT energy = GlobalScalarAssembler::sum<Exec>(
      gap_weights.size(), [=] TRIBOL_HOST_DEVICE( IndexT node ) { return nodal_energy_data[node]; },
      gap_weights.get().GetComm() );
  auto force = gap_weights * quantities.weighted_gap.dvalues_dx;
  force += area_weights * quantities.area.dvalues_dx;

  const auto redecomp_gap_weights = submesh_data_.TransferTrueDofsToRedecomp( gap_weights.get() );
  const auto redecomp_area_weights = submesh_data_.TransferTrueDofsToRedecomp( area_weights.get() );
  const auto* gap_data = redecomp_gap_weights.Read( use_device );
  const auto* area_data = redecomp_area_weights.Read( use_device );
  auto jacobian = assembleWeightedNodalHessian<Exec>(
      pairs, mesh1, mesh2, function_, mesh_data_, jacobian_data_,
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

  auto result = takeGapResult( quantities, penalty );
  result.energy = energy;
  result.force.reset( force.release() );
  result.df_dx.reset( jacobian.release() );
  result.pressure.reset( gap_weights.release() );
  return result;
}

template <typename PairRuleGenerator, ExecutionMode Exec, typename Enforcement>
ContactFormulationResult NodalEnergyMortar<PairRuleGenerator, Exec, Enforcement>::computeForces(
    MeshData::Viewer mesh1, MeshData::Viewer mesh2, LagrangeMultiplierEnforcement enforcement ) const
{
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  auto quantities = evaluateAndAssembleNodalQuantities<Exec>( pairs, mesh1, mesh2, function_, mesh_data_, submesh_data_,
                                                              jacobian_data_ );
  shared::ParVector pressure( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_data_.GetSubmeshPressure().GetTrueDofs( pressure.get() );
  if ( pressure.size() != quantities.weighted_gap.values.size() ) {
    throw std::invalid_argument( "LM values must match dual true DOFs." );
  }
  const RealT energy = pressure.dot( quantities.weighted_gap.values );
  auto force = pressure * quantities.weighted_gap.dvalues_dx;
  const auto redecomp_pressure = submesh_data_.TransferTrueDofsToRedecomp( pressure.get() );
  const auto* pressure_data = redecomp_pressure.Read( isOnDevice( Exec ) );
  auto jacobian = assembleWeightedNodalHessian<Exec>(
      pairs, mesh1, mesh2, function_, mesh_data_, jacobian_data_,
      [=] TRIBOL_HOST_DEVICE( IndexT node ) { return MdArray<RealT, 2>{ pressure_data[node], 0.0 }; } );
  auto df_dp = quantities.weighted_gap.dvalues_dx.transpose();

  auto result = takeGapResult( quantities, enforcement );
  result.energy = energy;
  result.force.reset( force.release() );
  result.df_dx.reset( jacobian.release() );
  result.df_dp.reset( df_dp.release() );
  result.pressure.reset( pressure.release() );
  return result;
}

using EnergyRuleGenerator1 = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>;
using EnergyRuleGenerator2 = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>;
using EnergyRuleGenerator3 = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>;

#define TRIBOL_INSTANTIATE_ENERGY_MORTAR_RULE_GENERATOR( RuleGenerator, Exec ) \
  template class QuadraturePointEnergyMortar<RuleGenerator, Exec>;             \
  template class NodalEnergyMortar<RuleGenerator, Exec, PenaltyEnforcement>;   \
  template class NodalEnergyMortar<RuleGenerator, Exec, LagrangeMultiplierEnforcement>
#define TRIBOL_INSTANTIATE_ENERGY_MORTAR( Exec )                                 \
  TRIBOL_INSTANTIATE_ENERGY_MORTAR_RULE_GENERATOR( EnergyRuleGenerator1, Exec ); \
  TRIBOL_INSTANTIATE_ENERGY_MORTAR_RULE_GENERATOR( EnergyRuleGenerator2, Exec ); \
  TRIBOL_INSTANTIATE_ENERGY_MORTAR_RULE_GENERATOR( EnergyRuleGenerator3, Exec )

TRIBOL_INSTANTIATE_ENERGY_MORTAR( ExecutionMode::Sequential );
#ifdef TRIBOL_USE_OPENMP
TRIBOL_INSTANTIATE_ENERGY_MORTAR( ExecutionMode::OpenMP );
#endif
#ifdef TRIBOL_USE_CUDA
TRIBOL_INSTANTIATE_ENERGY_MORTAR( ExecutionMode::Cuda );
#endif
#ifdef TRIBOL_USE_HIP
TRIBOL_INSTANTIATE_ENERGY_MORTAR( ExecutionMode::Hip );
#endif
#undef TRIBOL_INSTANTIATE_ENERGY_MORTAR
#undef TRIBOL_INSTANTIATE_ENERGY_MORTAR_RULE_GENERATOR

}  // namespace tribol
#endif
