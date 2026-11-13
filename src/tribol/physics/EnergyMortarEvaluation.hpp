// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Nodal Energy Mortar gap, penalty, and Lagrange multiplier evaluation and assembly. */
#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTAREVALUATION_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTAREVALUATION_HPP_
#include "tribol/config.hpp"
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
#include "tribol/assembly/PairEvaluationAssembly.hpp"
#include "tribol/physics/ContactEnforcement.hpp"
#include "tribol/physics/EnergyMortarPairFunctions.hpp"

namespace tribol {
namespace detail {
inline constexpr RealT nodal_area_tolerance =
    1.0e-14;  ///< Minimum supported tributary-area magnitude for penalty contact.

/** @brief Energy Mortar's assembled weighted gaps and tributary areas, before activation. */
struct EnergyMortarNodalQuantities {
  AssembledField weighted_gap;    ///< Weighted gaps and their coordinate Jacobian.
  AssembledField tributary_area;  ///< Tributary areas and their coordinate Jacobian.
};

/**
 * @brief Evaluate and assemble weighted gaps, tributary areas, and their coordinate Jacobians.
 * @tparam PairFunction Pair function returning weighted gaps followed by areas.
 * @tparam Exec Concrete execution mode for evaluation and numerical assembly.
 * @param pairs Current accepted pair list.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param pair_function Fixed pair-function configuration.
 * @param mesh_data Current displacement transfer data.
 * @param submesh_data Scalar field transfer data.
 * @param jacobian_data Parent and scalar-field Jacobian transfer data.
 * @return Unmodified weighted gaps, areas, and their coordinate Jacobians, including allocated zeros for empty pairs.
 * @throws std::invalid_argument Geometry or assembly dimensions are invalid; no result is returned.
 * @throws std::length_error Quadrature capacity or the assembly index range is exceeded; no result is returned.
 */
template <ExecutionMode Exec, typename PairFunction>
EnergyMortarNodalQuantities evaluateAndAssembleNodalQuantities( const ArrayT<InterfacePair>& pairs,
                                                                MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                                const PairFunction& pair_function,
                                                                const MfemMeshData& mesh_data,
                                                                const MfemSubmeshData& submesh_data,
                                                                const MfemJacobianData& jacobian_data )
{
  using PairResult = typename PairFunction::Result;
  constexpr IndexT num_nodes = PairFunction::num_nodes;
  constexpr IndexT size_coords = PairFunction::size_coords;
  static_assert( PairFunction::size_values == 2 * num_nodes );
  auto buffer = evaluatePairValuesAndJacobians<Exec>( pairs, mesh1, mesh2, pair_function );
  const auto contributions_view = buffer.view();
  checkPairQuadratureResults<Exec>( buffer.quadratureResults() );
  constexpr IndexT num_fields = PairResult::num_fields;
  std::vector<MfemVectorAssembler> vectors;
  std::vector<MfemMatrixAssembler> matrices;
  vectors.reserve( num_fields );
  matrices.reserve( num_fields );
  for ( IndexT field = 0; field < num_fields; ++field ) {
    vectors.push_back( detail::makeNodalVectorAssembler( submesh_data, mesh1 ) );
    matrices.push_back( detail::makeNodalCoordinateMatrixAssembler( mesh_data, submesh_data, jacobian_data, mesh1,
                                                                    mesh2, pairs.size(), pairs.getAllocatorID() ) );
  }
  detail::addNodalValuesAndJacobians<Exec, num_fields, num_nodes, size_coords>(
      pairs, vectors, matrices,
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT field, IndexT node ) {
        const auto integral_field = static_cast<typename PairResult::Field>( field );
        return contributions_view.has_active_points[pair]
                   ? contributions_view.value( pair, PairResult::valueIndex( integral_field, node ) )
                   : 0.0;
      },
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT field, IndexT node, IndexT coordinate ) {
        const auto integral_field = static_cast<typename PairResult::Field>( field );
        return contributions_view.has_active_points[pair]
                   ? contributions_view.jacobian( pair, PairResult::valueIndex( integral_field, node ), coordinate )
                   : 0.0;
      } );
  return { { vectors[0].assemble(), matrices[0].assemble() }, { vectors[1].assemble(), matrices[1].assemble() } };
}

/**
 * @brief Assemble the coordinate Hessian of fixed nodal weights contracted with pair gap and area integrals.
 * @tparam PairFunction Pair function returning weighted gaps followed by areas.
 * @tparam WeightReader Device-callable node reader returning gap and area weights in that order.
 * @tparam Exec Concrete execution mode for evaluation and numerical assembly.
 * @param pairs Current accepted pair list.
 * @param mesh1 Current first mesh, whose scalar finite-element nodes carry the weights.
 * @param mesh2 Current second mesh.
 * @param pair_function Fixed pair-function configuration.
 * @param mesh_data Current displacement transfer data.
 * @param jacobian_data Current Jacobian transfer data.
 * @param read_weights Reader indexed by scalar submesh node.
 * @return Coordinate Hessian assembled on parent true DOFs.
 * @throws std::invalid_argument Geometry or assembly dimensions are invalid.
 * @throws std::length_error Pair buffers or matrix storage exceed the supported index range.
 */
template <ExecutionMode Exec, typename PairFunction, typename WeightReader>
shared::ParSparseMat assembleWeightedNodalHessian( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                   MeshData::Viewer mesh2, const PairFunction& pair_function,
                                                   const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data,
                                                   WeightReader read_weights )
{
  using PairResult = typename PairFunction::Result;
  constexpr IndexT num_nodes = PairFunction::num_nodes;
  static_assert( PairFunction::size_values == 2 * num_nodes );
  const IndexT num_pairs = pairs.size();
  const auto pairs_view = pairs.view();
  const auto connectivity = mesh1.getConnectivity();
  auto hessians = evaluatePairWeightedJacobiansAndHessians<Exec>(
      pairs, mesh1, mesh2, pair_function, [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
        MdArray<RealT, PairFunction::size_values> weights{};
        const auto nodes = connectivity( pairs_view[pair_index].element_id1_ );
        for ( IndexT node = 0; node < num_nodes; ++node ) {
          const auto nodal_weights = read_weights( nodes[node] );
          weights[PairResult::weightedGapIndex( node )] = nodal_weights[0];
          weights[PairResult::tributaryAreaIndex( node )] = nodal_weights[1];
        }
        return weights;
      } );
  const auto hessian_view = hessians.view();
  auto assembler = detail::makeCoordinateMatrixAssembler( mesh_data, jacobian_data, mesh1, mesh2, num_pairs,
                                                          pairs.getAllocatorID() );
  assembler.addPairs<Exec, PairFunction::size_coords / 2, PairFunction::size_coords / 2, 4>(
      pairs, [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
        return hessian_view.hessian( pair, row, column );
      } );
  return assembler.assemble();
}

/**
 * @brief Convert assembled nodal quantities to normalized compressive penalty gaps.
 * @param quantities Assembled weighted gaps and areas, consumed for the result.
 * @return Owned gap and coordinate-Jacobian fields.
 */
ContactFormulationResult makeNodalPenaltyGapResult( EnergyMortarNodalQuantities& quantities );

/**
 * @brief Convert assembled nodal quantities to signed weighted Lagrange multiplier gaps.
 * @param quantities Assembled weighted gaps and areas, consumed for the result.
 * @return Owned gap and coordinate-Jacobian fields.
 */
ContactFormulationResult makeNodalLagrangeMultiplierGapResult( EnergyMortarNodalQuantities& quantities );

}  // namespace detail

/**
 * @brief Assemble nodal integrals and calculate normalized compressive penalty gaps.
 * @tparam Exec Concrete execution mode for evaluation and assembly kernels.
 * @tparam PairFunction Pair function returning weighted-gap integrals followed by tributary areas.
 * @param pairs Accepted element pairs; not retained.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param mesh_data Displacement transfer data used for assembly.
 * @param submesh_data Pressure and gap transfer data used for assembly.
 * @param jacobian_data Parent and scalar-field Jacobian transfer data used for assembly.
 * @param pair_function Configured nodal-integral pair_function.
 * @return Owned gaps and their coordinate Jacobian; other outputs are absent. Empty pairs give zero gaps.
 * @throws std::invalid_argument Pair geometry or assembly dimensions are invalid.
 * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
ContactFormulationResult evaluateAndAssembleNodalPenaltyGaps(
    const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1, MeshData::Viewer mesh2, const MfemMeshData& mesh_data,
    const MfemSubmeshData& submesh_data, const MfemJacobianData& jacobian_data, const PairFunction& pair_function )
{
  auto quantities = detail::evaluateAndAssembleNodalQuantities<Exec>( pairs, mesh1, mesh2, pair_function, mesh_data,
                                                                      submesh_data, jacobian_data );
  return detail::makeNodalPenaltyGapResult( quantities );
}

/**
 * @brief Assemble nodal integrals, apply penalty enforcement, and calculate forces and derivatives.
 * @tparam Exec Concrete execution mode for evaluation and assembly kernels.
 * @tparam PairFunction Pair function returning weighted-gap integrals followed by tributary areas.
 * @param pairs Accepted element pairs; not retained.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param mesh_data Displacement transfer data used for assembly.
 * @param submesh_data Pressure and gap transfer data used for assembly.
 * @param jacobian_data Parent and scalar-field Jacobian transfer data used for assembly.
 * @param pair_function Configured nodal-integral pair_function.
 * @param penalty Fixed positive, finite penalty enforcement.
 * @return Owned gaps, pressure, energy, force, weighted_gap_jacobian, and df_dx; df_dp is absent. Empty pairs
 * contribute zero force and energy; unsupported nodal areas give zero penalty pressure.
 * @throws std::invalid_argument Geometry, assembly dimensions, or nodal field dimensions are invalid.
 * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
ContactFormulationResult evaluateAndAssembleNodalPenalty( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                          MeshData::Viewer mesh2, const MfemMeshData& mesh_data,
                                                          const MfemSubmeshData& submesh_data,
                                                          const MfemJacobianData& jacobian_data,
                                                          const PairFunction& pair_function,
                                                          const PenaltyEnforcement& penalty )
{
  const RealT stiffness = penalty.stiffness();
  auto quantities = detail::evaluateAndAssembleNodalQuantities<Exec>( pairs, mesh1, mesh2, pair_function, mesh_data,
                                                                      submesh_data, jacobian_data );
  const auto make_vector = [&] {
    shared::ParVector vector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data.GetSubmeshFESpace() ) );
    vector.fill( 0.0 );
    return vector;
  };
  auto gap_weights = make_vector();
  auto area_weights = make_vector();
  auto d2energy_dweighted_gap2 = make_vector();
  auto d2energy_dweighted_gap_dtributary_area = make_vector();
  auto d2energy_dtributary_area2 = make_vector();
  const int allocator_id = pairs.getAllocatorID();
  Array1D<RealT> nodal_energy( gap_weights.size(), gap_weights.size(), allocator_id );
  const bool use_device = isOnDevice( Exec );
  const auto* weighted_gap = quantities.weighted_gap.values.get().Read( use_device );
  const auto* area = quantities.tributary_area.values.get().Read( use_device );
  auto* gap_weight = gap_weights.get().Write( use_device );
  auto* area_weight = area_weights.get().Write( use_device );
  auto* gap_gap_data = d2energy_dweighted_gap2.get().Write( use_device );
  auto* gap_area_data = d2energy_dweighted_gap_dtributary_area.get().Write( use_device );
  auto* area_area_data = d2energy_dtributary_area2.get().Write( use_device );
  auto* nodal_energy_data = nodal_energy.data();
  forAllExec<Exec>( gap_weights.size(), [=] TRIBOL_HOST_DEVICE( IndexT node ) {
    const auto partials = EnergyMortarNodalPenaltyPartials::evaluate( weighted_gap[node], area[node], stiffness,
                                                                      detail::nodal_area_tolerance );
    nodal_energy_data[node] = partials.energy;
    gap_weight[node] = partials.denergy_dweighted_gap;
    area_weight[node] = partials.denergy_dtributary_area;
    gap_gap_data[node] = partials.d2energy_dweighted_gap2;
    gap_area_data[node] = partials.d2energy_dweighted_gap_dtributary_area;
    area_area_data[node] = partials.d2energy_dtributary_area2;
  } );
  const RealT energy = GlobalScalarAssembler::sum<Exec>(
      gap_weights.size(), [=] TRIBOL_HOST_DEVICE( IndexT node ) { return nodal_energy_data[node]; },
      gap_weights.get().GetComm() );
  auto force = gap_weights * quantities.weighted_gap.jacobian;
  force += area_weights * quantities.tributary_area.jacobian;

  const auto redecomp_gap_weights = submesh_data.TransferTrueDofsToRedecomp( gap_weights.get() );
  const auto redecomp_area_weights = submesh_data.TransferTrueDofsToRedecomp( area_weights.get() );
  const auto* gap_data = redecomp_gap_weights.Read( use_device );
  const auto* area_data = redecomp_area_weights.Read( use_device );
  auto jacobian = detail::assembleWeightedNodalHessian<Exec>(
      pairs, mesh1, mesh2, pair_function, mesh_data, jacobian_data,
      [=] TRIBOL_HOST_DEVICE( IndexT node ) { return MdArray<RealT, 2>{ gap_data[node], area_data[node] }; } );
  const auto make_diagonal = [&]( const shared::ParVector& values ) {
    const auto& fes = submesh_data.GetSubmeshFESpace();
    return shared::ParSparseMat::diagonalMatrix( fes.GetComm(), fes.GlobalTrueVSize(), fes.GetTrueDofOffsets(),
                                                 values.get() );
  };
  const auto gg = make_diagonal( d2energy_dweighted_gap2 );
  const auto ga = make_diagonal( d2energy_dweighted_gap_dtributary_area );
  const auto aa = make_diagonal( d2energy_dtributary_area2 );
  jacobian += shared::ParSparseMat::rap( gg, quantities.weighted_gap.jacobian );
  const auto mixed =
      shared::ParSparseMat::rap( quantities.weighted_gap.jacobian, ga, quantities.tributary_area.jacobian );
  jacobian += mixed;
  jacobian += mixed.transpose();
  jacobian += shared::ParSparseMat::rap( aa, quantities.tributary_area.jacobian );

  auto result = detail::makeNodalPenaltyGapResult( quantities );
  result.energy = energy;
  result.force.reset( force.release() );
  result.df_dx.reset( jacobian.release() );
  result.pressure.reset( gap_weights.release() );
  return result;
}

/**
 * @brief Assemble nodal integrals and calculate signed weighted Lagrange multiplier gaps.
 * @tparam Exec Concrete execution mode for evaluation and assembly kernels.
 * @tparam PairFunction Pair function returning weighted-gap integrals followed by tributary areas.
 * @param pairs Accepted element pairs; not retained.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param mesh_data Displacement transfer data used for assembly.
 * @param submesh_data Pressure and gap transfer data used for assembly.
 * @param jacobian_data Parent and scalar-field Jacobian transfer data used for assembly.
 * @param pair_function Configured nodal-integral pair_function.
 * @return Owned gaps and their coordinate Jacobian; other outputs are absent. Empty pairs give zero gaps.
 * @throws std::invalid_argument Pair geometry or assembly dimensions are invalid.
 * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
ContactFormulationResult evaluateAndAssembleNodalLagrangeMultiplierGaps(
    const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1, MeshData::Viewer mesh2, const MfemMeshData& mesh_data,
    const MfemSubmeshData& submesh_data, const MfemJacobianData& jacobian_data, const PairFunction& pair_function )
{
  auto quantities = detail::evaluateAndAssembleNodalQuantities<Exec>( pairs, mesh1, mesh2, pair_function, mesh_data,
                                                                      submesh_data, jacobian_data );
  return detail::makeNodalLagrangeMultiplierGapResult( quantities );
}

/**
 * @brief Assemble nodal integrals, apply Lagrange multiplier enforcement, and calculate forces and derivatives.
 * @tparam Exec Concrete execution mode for evaluation and assembly kernels.
 * @tparam PairFunction Pair function returning weighted-gap integrals followed by tributary areas.
 * @param pairs Accepted element pairs; not retained.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param mesh_data Displacement transfer data used for assembly.
 * @param submesh_data Pressure and gap transfer data used for assembly.
 * @param jacobian_data Parent and scalar-field Jacobian transfer data used for assembly.
 * @param pair_function Configured nodal-integral pair_function.
 * @return Owned gaps, pressure, energy, force, weighted_gap_jacobian, df_dx, and df_dp. Empty pairs contribute zero
 * force and energy; pressure contains the supplied LM values.
 * @throws std::invalid_argument Geometry, assembly dimensions, or nodal field dimensions are invalid.
 * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
ContactFormulationResult evaluateAndAssembleNodalLagrangeMultiplier(
    const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1, MeshData::Viewer mesh2, const MfemMeshData& mesh_data,
    const MfemSubmeshData& submesh_data, const MfemJacobianData& jacobian_data, const PairFunction& pair_function )
{
  auto quantities = detail::evaluateAndAssembleNodalQuantities<Exec>( pairs, mesh1, mesh2, pair_function, mesh_data,
                                                                      submesh_data, jacobian_data );
  shared::ParVector pressure( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data.GetSubmeshFESpace() ) );
  submesh_data.GetSubmeshPressure().GetTrueDofs( pressure.get() );
  if ( pressure.size() != quantities.weighted_gap.values.size() ) {
    throw std::invalid_argument( "LM values must match dual true DOFs." );
  }
  const RealT energy = pressure.dot( quantities.weighted_gap.values );
  auto force = pressure * quantities.weighted_gap.jacobian;
  const auto redecomp_pressure = submesh_data.TransferTrueDofsToRedecomp( pressure.get() );
  const auto* pressure_data = redecomp_pressure.Read( isOnDevice( Exec ) );
  auto jacobian = detail::assembleWeightedNodalHessian<Exec>(
      pairs, mesh1, mesh2, pair_function, mesh_data, jacobian_data,
      [=] TRIBOL_HOST_DEVICE( IndexT node ) { return MdArray<RealT, 2>{ pressure_data[node], 0.0 }; } );
  auto df_dp = quantities.weighted_gap.jacobian.transpose();

  auto result = detail::makeNodalLagrangeMultiplierGapResult( quantities );
  result.energy = energy;
  result.force.reset( force.release() );
  result.df_dx.reset( jacobian.release() );
  result.df_dp.reset( df_dp.release() );
  result.pressure.reset( pressure.release() );
  return result;
}

}  // namespace tribol
#endif
#endif  // SRC_TRIBOL_PHYSICS_ENERGYMORTAREVALUATION_HPP_
