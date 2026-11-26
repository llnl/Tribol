// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Evaluate single mortar with frozen-geometry or exact Jacobian strategies. */
#ifndef SRC_TRIBOL_PHYSICS_SINGLEMORTARFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_SINGLEMORTARFORMULATION_HPP_

#include "tribol/config.hpp"
#ifdef BUILD_REDECOMP
#include <cstdint>
#include <utility>
#include <vector>

#include "tribol/assembly/EdgeAveragedNodalNormalAssembler.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/integrals/SingleMortarPairEvaluator.hpp"
#include "tribol/search/PairListBuilder.hpp"

namespace tribol {

namespace detail {

/**
 * @brief Gather non-mortar nodal pressures for one surface element from the Redecomp pressure array.
 * @tparam NumNodes Number of nodes on the surface element.
 * @param mesh1 Current non-mortar mesh supplying global node IDs.
 * @param element_id Local element ID on mesh1.
 * @param pressures Redecomp nodal pressure array indexed by global node ID.
 * @return Nodal pressures on the element.
 */
template <IndexT NumNodes>
TRIBOL_HOST_DEVICE MdArray<RealT, NumNodes> gatherElementPressures( MeshData::Viewer mesh1, IndexT element_id,
                                                                    const RealT* pressures )
{
  MdArray<RealT, NumNodes> element_pressures{};
  for ( IndexT node = 0; node < NumNodes; ++node ) {
    element_pressures[node] = pressures[mesh1.getGlobalNodeId( element_id, node )];
  }
  return element_pressures;
}

/**
 * @brief Add one pair's nodal weighted gaps to the non-mortar element vector.
 * @tparam NumNodes Number of nodes on geometry1.
 * @param gap_view Element-vector view for weighted gaps.
 * @param redecomp_element1 Redecomp element index for geometry1.
 * @param gaps Nodal weighted gaps for this pair.
 */
template <IndexT NumNodes>
TRIBOL_HOST_DEVICE void assembleElementGapVector( ArrayViewT<RealT, 2> gap_view, int redecomp_element1,
                                                  const MdArray<RealT, NumNodes>& gaps )
{
  detail::addElementVector( gap_view, redecomp_element1, NumNodes, 1,
                            [&]( IndexT node, IndexT ) { return gaps[node]; } );
}

/**
 * @brief Add one pair's nodal residuals to both geometry1 and geometry2 element vectors.
 * @tparam NumNodes Number of nodes per surface element.
 * @tparam SpatialDim Spatial dimension of the residual vector.
 * @param residual_view Element-vector view for nodal residuals.
 * @param redecomp_element1 Redecomp element index for geometry1.
 * @param redecomp_element2 Redecomp element index for geometry2.
 * @param residuals Pair residuals ordered as geometry1 followed by geometry2.
 */
template <IndexT NumNodes, IndexT SpatialDim>
TRIBOL_HOST_DEVICE void assembleElementResidualVectors( ArrayViewT<RealT, 2> residual_view, int redecomp_element1,
                                                        int redecomp_element2,
                                                        const MdArray<RealT, 2 * NumNodes * SpatialDim>& residuals )
{
  constexpr IndexT size_coords_per_element = NumNodes * SpatialDim;
  detail::addElementVector( residual_view, redecomp_element1, NumNodes, SpatialDim,
                            [&]( IndexT node, IndexT d ) { return residuals[node * SpatialDim + d]; } );
  detail::addElementVector( residual_view, redecomp_element2, NumNodes, SpatialDim, [&]( IndexT node, IndexT d ) {
    return residuals[size_coords_per_element + node * SpatialDim + d];
  } );
}

/**
 * @brief Gather pair coordinates and nodal normals, invoke a per-pair kernel, and check quadrature diagnostics.
 * @tparam Exec Concrete execution mode for the pair loop and reductions.
 * @tparam Integral Single mortar pair integral supplying coordinate dimensions and views.
 * @tparam EvaluatePair Device-callable `(index, pair, coords1, coords2, normals1) -> QuadratureMetadata`.
 * @param pairs Refined candidate pairs in local element numbering.
 * @param mesh1 Current non-mortar mesh.
 * @param mesh2 Current mortar mesh.
 * @param redecomp_normals Normalized nodal normal field on Redecomp.
 * @param comm MPI communicator for reducing quadrature diagnostics.
 * @param evaluate_pair Per-pair evaluation and packing callable.
 * @return True if no rank encountered insufficient quadrature capacity; false after reporting an overflow error.
 */
template <ExecutionMode Exec, typename Integral, typename EvaluatePair>
bool runSingleMortarPairLoop( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                              const mfem::GridFunction& redecomp_normals, MPI_Comm comm, EvaluatePair evaluate_pair )
{
  constexpr IndexT num_nodes = Integral::num_nodes1;
  constexpr IndexT spatial_dim = Integral::spatial_dim;
  const auto pair_view = pairs.view();
  const auto* normal_values = redecomp_normals.Read( isOnDevice( Exec ) );
  const IndexT num_normal_nodes = redecomp_normals.FESpace()->GetNDofs();
  ReduceMax<Exec, int> insufficient_capacity( 0 );
  ReduceSum<Exec, std::uint64_t> num_invalid_projections( 0 );
  forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT index ) {
    const auto pair = pair_view[index];
    const auto coords1 = mesh1.getFaceCoords<typename Integral::Coords1>( pair.element_id1_ );
    const auto coords2 = mesh2.getFaceCoords<typename Integral::Coords2>( pair.element_id2_ );
    MdArray<RealT, num_nodes, spatial_dim> normals1{};
    for ( IndexT node = 0; node < num_nodes; ++node ) {
      const auto node_id = mesh1.getGlobalNodeId( pair.element_id1_, node );
      for ( IndexT d = 0; d < spatial_dim; ++d ) {
        normals1( node, d ) = normal_values[d * num_normal_nodes + node_id];
      }
    }
    const QuadratureMetadata metadata = evaluate_pair( index, pair, coords1.view(), coords2.view(), normals1.view() );
    if ( metadata.status == QuadratureStatus::InsufficientCapacity ) {
      insufficient_capacity.max( 1 );
    }
    if ( metadata.status == QuadratureStatus::InvalidGeometry ) {
      num_invalid_projections += 1;
    }
  } );

  // Report diagnostics on the coupling scheme's root after all ranks have finished evaluating pairs.
  const auto overflow = static_cast<std::uint64_t>( insufficient_capacity.get() );
  MdArray<std::uint64_t, 2> quadrature_counts{ overflow, num_invalid_projections.get() };
  MPI_Allreduce( MPI_IN_PLACE, quadrature_counts.data(), quadrature_counts.size(), MPI_UINT64_T, MPI_SUM, comm );
  if ( quadrature_counts[0] > 0 ) {
    reportQuadratureFailure( QuadratureStatus::InsufficientCapacity );
    return false;
  }
  if ( quadrature_counts[1] > 0 ) {
    int rank = 0;
    MPI_Comm_rank( comm, &rank );
    if ( rank == 0 ) {
      SLIC_WARNING( "Single mortar skipped " << quadrature_counts[1]
                                             << " pair evaluations with invalid projected geometry." );
    }
  }
  return true;
}

}  // namespace detail

/**
 * @brief Assemble single mortar gaps, residuals, and frozen-geometry Jacobians (holding weights and normals fixed).
 *
 * In residual mode, df_dp is assembled as the transpose of the frozen weighted-gap Jacobian and df_dx is left null.
 */
struct SingleMortarFrozenJacobian {
  const MfemMeshData& mesh_data;          ///< Transfer data for coordinates and element numbering.
  const MfemSubmeshData& submesh_data;    ///< Transfer data for scalar LM pressure and gaps.
  const MfemJacobianData& jacobian_data;  ///< Transfer paths for assembled true DOFs.

  /**
   * @brief Assemble weighted gaps and the frozen weighted-gap Jacobian.
   * @tparam Exec Concrete execution mode for normal assembly and pair evaluation.
   * @tparam PairEvaluator Configured single mortar pair evaluator.
   * @param pairs Refined candidate pairs in local element numbering.
   * @param mesh1 Current non-mortar mesh with reference coordinates.
   * @param mesh2 Current mortar mesh.
   * @param evaluator Configured pair evaluator.
   * @return Owned gap and frozen weighted-gap Jacobian, or an empty result on quadrature overflow.
   */
  template <ExecutionMode Exec, typename PairEvaluator>
  ContactFormulationResult computeGaps( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                        MeshData::Viewer mesh2, const PairEvaluator& evaluator ) const
  {
    constexpr IndexT num_nodes = PairEvaluator::num_nodes1;
    constexpr IndexT size_geom1_coords = PairEvaluator::size_geom1_coords;
    const auto redecomp_normals =
        detail::assembleEdgeAveragedNodalNormals<Exec, num_nodes>( mesh1, mesh_data, jacobian_data );
    const auto allocator_id = pairs.getAllocatorID();
    const auto& field_fes = submesh_data.GetRedecompFESpace();
    const auto map1 = mesh_data.GetElemMap1().view();

    detail::MfemElementVectorAssembler gap( jacobian_data.SubmeshPath(), field_fes, allocator_id );
    auto gap_dx = detail::makeScalarFieldJacobianAssembler( mesh_data, submesh_data, jacobian_data, mesh1, mesh2,
                                                            pairs.size(), allocator_id );
    const auto gap_view = gap.view();
    const auto gap_dx_view = gap_dx.template view<num_nodes, size_geom1_coords, 2>();

    const bool ok = detail::runSingleMortarPairLoop<Exec, PairEvaluator>(
        pairs, mesh1, mesh2, redecomp_normals, submesh_data.GetSubmeshFESpace().GetComm(),
        [=] TRIBOL_HOST_DEVICE( IndexT index, const InterfacePair& pair, typename PairEvaluator::Coords1 coords1,
                                typename PairEvaluator::Coords2 coords2, typename PairEvaluator::Coords1 normals1 ) {
          const auto values = evaluator.evaluateGaps( coords1, coords2, normals1 );
          detail::assembleElementGapVector<num_nodes>( gap_view, map1[pair.element_id1_], values.gaps );
          gap_dx_view.packPair( index, pair, [&]( IndexT, IndexT row, IndexT column ) {
            return values.frozen_gap_jacobian( row, column );
          } );
          return values.metadata;
        } );
    if ( !ok ) {
      return {};
    }

    ContactFormulationResult result;
    result.gap.reset( gap.assemble().release() );
    result.weighted_gap_jacobian.reset( gap_dx.assemble().release() );
    return result;
  }

  /**
   * @brief Assemble weighted gaps, residuals, pressure, frozen gap Jacobian, and df_dp; leave df_dx null.
   * @tparam Exec Concrete execution mode for normal assembly and pair evaluation.
   * @tparam PairEvaluator Configured single mortar pair evaluator.
   * @param pairs Refined candidate pairs in local element numbering.
   * @param mesh1 Current non-mortar mesh with reference coordinates.
   * @param mesh2 Current mortar mesh.
   * @param evaluator Configured pair evaluator.
   * @param multiplier Caller-supplied pressure on submesh true DOFs.
   * @return Owned gap, residual, pressure, weighted-gap Jacobian, and df_dp, or an empty result on overflow.
   */
  template <ExecutionMode Exec, typename PairEvaluator>
  ContactFormulationResult computeResiduals( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                             MeshData::Viewer mesh2, const PairEvaluator& evaluator,
                                             const mfem::HypreParVector& multiplier ) const
  {
    constexpr IndexT num_nodes = PairEvaluator::num_nodes1;
    constexpr IndexT spatial_dim = PairEvaluator::spatial_dim;
    constexpr IndexT size_geom1_coords = PairEvaluator::size_geom1_coords;
    const auto redecomp_normals =
        detail::assembleEdgeAveragedNodalNormals<Exec, num_nodes>( mesh1, mesh_data, jacobian_data );
    const auto redecomp_pressure = submesh_data.TransferTrueDofsToRedecomp( multiplier );
    const auto* pressures = redecomp_pressure.Read( isOnDevice( Exec ) );
    const auto allocator_id = pairs.getAllocatorID();
    const auto& coords_fes = *mesh_data.GetRedecompResponse().FESpace();
    const auto& field_fes = submesh_data.GetRedecompFESpace();
    const auto map1 = mesh_data.GetElemMap1().view();
    const auto map2 = mesh_data.GetElemMap2().view();

    detail::MfemElementVectorAssembler gap( jacobian_data.SubmeshPath(), field_fes, allocator_id );
    detail::MfemElementVectorAssembler residual( jacobian_data.ParentPath(), coords_fes, allocator_id );
    auto gap_dx = detail::makeScalarFieldJacobianAssembler( mesh_data, submesh_data, jacobian_data, mesh1, mesh2,
                                                            pairs.size(), allocator_id );
    const auto gap_view = gap.view();
    const auto residual_view = residual.view();
    const auto gap_dx_view = gap_dx.template view<num_nodes, size_geom1_coords, 2>();

    const bool ok = detail::runSingleMortarPairLoop<Exec, PairEvaluator>(
        pairs, mesh1, mesh2, redecomp_normals, submesh_data.GetSubmeshFESpace().GetComm(),
        [=] TRIBOL_HOST_DEVICE( IndexT index, const InterfacePair& pair, typename PairEvaluator::Coords1 coords1,
                                typename PairEvaluator::Coords2 coords2, typename PairEvaluator::Coords1 normals1 ) {
          const auto pressures1 = detail::gatherElementPressures<num_nodes>( mesh1, pair.element_id1_, pressures );
          const auto values = evaluator.evaluateResiduals( coords1, coords2, normals1, pressures1.view() );
          detail::assembleElementGapVector<num_nodes>( gap_view, map1[pair.element_id1_], values.gap.gaps );
          detail::assembleElementResidualVectors<num_nodes, spatial_dim>( residual_view, map1[pair.element_id1_],
                                                                          map2[pair.element_id2_], values.residuals );
          gap_dx_view.packPair( index, pair, [&]( IndexT, IndexT row, IndexT column ) {
            return values.gap.frozen_gap_jacobian( row, column );
          } );
          return values.gap.metadata;
        } );
    if ( !ok ) {
      return {};
    }

    auto gap_jacobian = gap_dx.assemble();
    ContactFormulationResult result;
    result.gap.reset( gap.assemble().release() );
    result.residual.reset( residual.assemble().release() );
    result.pressure.reset( shared::ParVector( multiplier ).release() );
    result.df_dp.reset( gap_jacobian.transpose().release() );
    result.weighted_gap_jacobian.reset( gap_jacobian.release() );
    return result;
  }
};

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Assemble single mortar gaps, residuals, and exact coordinate Jacobians including weight and normal
 * derivatives.
 */
struct SingleMortarExactJacobian {
  const MfemMeshData& mesh_data;          ///< Transfer data for coordinates and element numbering.
  const MfemSubmeshData& submesh_data;    ///< Transfer data for scalar LM pressure and gaps.
  const MfemJacobianData& jacobian_data;  ///< Transfer paths for assembled true DOFs.

  /**
   * @brief Assemble weighted gaps and the exact weighted-gap coordinate Jacobian.
   * @tparam Exec Concrete execution mode for normal assembly and pair evaluation.
   * @tparam PairEvaluator Configured single mortar pair evaluator.
   * @param pairs Refined candidate pairs in local element numbering.
   * @param mesh1 Current non-mortar mesh with reference coordinates.
   * @param mesh2 Current mortar mesh.
   * @param evaluator Configured pair evaluator.
   * @return Owned gap and exact weighted-gap Jacobian, or an empty result on quadrature overflow.
   */
  template <ExecutionMode Exec, typename PairEvaluator>
  ContactFormulationResult computeGaps( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                        MeshData::Viewer mesh2, const PairEvaluator& evaluator ) const
  {
    constexpr IndexT num_nodes = PairEvaluator::num_nodes1;
    constexpr IndexT size_geom1_coords = PairEvaluator::size_geom1_coords;
    const auto normals =
        detail::assembleEdgeAveragedNodalNormalsAndJacobian<Exec, num_nodes>( mesh1, mesh_data, jacobian_data );
    const auto allocator_id = pairs.getAllocatorID();
    const auto& coords_fes = *mesh_data.GetRedecompResponse().FESpace();
    const auto& field_fes = submesh_data.GetRedecompFESpace();
    const auto& pressure_path = jacobian_data.SubmeshPath();
    const auto& elem_map1 = mesh_data.GetElemMap1();
    const auto map1 = elem_map1.view();

    detail::MfemElementVectorAssembler gap( pressure_path, field_fes, allocator_id );
    auto gap_dx = detail::makeScalarFieldJacobianAssembler( mesh_data, submesh_data, jacobian_data, mesh1, mesh2,
                                                            pairs.size(), allocator_id );
    const MfemMatrixSpace normal_space{ jacobian_data.SurfacePath(), coords_fes };
    MfemMatrixAssembler gap_dn(
        MfemMatrixSpace{ pressure_path, field_fes }, normal_space,
        std::vector<MfemMatrixBlock>{ { 0, 0, num_nodes, num_nodes, 0, 0, elem_map1, elem_map1 } }, pairs.size(),
        allocator_id );
    const auto gap_view = gap.view();
    const auto gap_dx_view = gap_dx.template view<num_nodes, size_geom1_coords, 2>();
    const auto gap_dn_view = gap_dn.template view<num_nodes, size_geom1_coords, 1>();

    const bool ok = detail::runSingleMortarPairLoop<Exec, PairEvaluator>(
        pairs, mesh1, mesh2, normals.redecomp_normals, submesh_data.GetSubmeshFESpace().GetComm(),
        [=] TRIBOL_HOST_DEVICE( IndexT index, const InterfacePair& pair, typename PairEvaluator::Coords1 coords1,
                                typename PairEvaluator::Coords2 coords2, typename PairEvaluator::Coords1 normals1 ) {
          const auto differentiated = evaluator.evaluateAndDifferentiateGaps( coords1, coords2, normals1 );
          detail::assembleElementGapVector<num_nodes>( gap_view, map1[pair.element_id1_],
                                                       differentiated.evaluation.gaps );
          gap_dx_view.packPair(
              index, pair, [&]( IndexT, IndexT row, IndexT column ) { return differentiated.gap_dx( row, column ); } );
          gap_dn_view.packPair(
              index, pair, [&]( IndexT, IndexT row, IndexT column ) { return differentiated.gap_dn( row, column ); } );
          return differentiated.evaluation.metadata;
        } );
    if ( !ok ) {
      return {};
    }

    auto gap_jacobian = gap_dx.assemble();
    const shared::ParSparseMatView dn_dx( normals.dn_dx.get() );
    gap_jacobian += gap_dn.assemble() * dn_dx;

    ContactFormulationResult result;
    result.gap.reset( gap.assemble().release() );
    result.weighted_gap_jacobian.reset( gap_jacobian.release() );
    return result;
  }

  /**
   * @brief Assemble weighted gaps, residuals, pressure, exact weighted-gap Jacobian, df_dp, and exact df_dx.
   * @tparam Exec Concrete execution mode for normal assembly and pair evaluation.
   * @tparam PairEvaluator Configured single mortar pair evaluator.
   * @param pairs Refined candidate pairs in local element numbering.
   * @param mesh1 Current non-mortar mesh with reference coordinates.
   * @param mesh2 Current mortar mesh.
   * @param evaluator Configured pair evaluator.
   * @param multiplier Caller-supplied pressure on submesh true DOFs.
   * @return Owned gap, residual, pressure, exact weighted-gap Jacobian, df_dp, and exact df_dx, or empty on overflow.
   */
  template <ExecutionMode Exec, typename PairEvaluator>
  ContactFormulationResult computeResiduals( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                             MeshData::Viewer mesh2, const PairEvaluator& evaluator,
                                             const mfem::HypreParVector& multiplier ) const
  {
    constexpr IndexT num_nodes = PairEvaluator::num_nodes1;
    constexpr IndexT spatial_dim = PairEvaluator::spatial_dim;
    constexpr IndexT size_geom1_coords = PairEvaluator::size_geom1_coords;
    const auto normals =
        detail::assembleEdgeAveragedNodalNormalsAndJacobian<Exec, num_nodes>( mesh1, mesh_data, jacobian_data );
    const auto redecomp_pressure = submesh_data.TransferTrueDofsToRedecomp( multiplier );
    const auto* pressures = redecomp_pressure.Read( isOnDevice( Exec ) );
    const auto allocator_id = pairs.getAllocatorID();
    const auto& coords_fes = *mesh_data.GetRedecompResponse().FESpace();
    const auto& field_fes = submesh_data.GetRedecompFESpace();
    const auto& coordinate_path = jacobian_data.ParentPath();
    const auto& pressure_path = jacobian_data.SubmeshPath();
    const auto& elem_map1 = mesh_data.GetElemMap1();
    const auto& elem_map2 = mesh_data.GetElemMap2();
    const auto map1 = elem_map1.view();
    const auto map2 = elem_map2.view();

    detail::MfemElementVectorAssembler gap( pressure_path, field_fes, allocator_id );
    detail::MfemElementVectorAssembler residual( coordinate_path, coords_fes, allocator_id );
    auto gap_dx = detail::makeScalarFieldJacobianAssembler( mesh_data, submesh_data, jacobian_data, mesh1, mesh2,
                                                            pairs.size(), allocator_id );
    auto frozen_gap_dx = detail::makeScalarFieldJacobianAssembler( mesh_data, submesh_data, jacobian_data, mesh1, mesh2,
                                                                   pairs.size(), allocator_id );
    auto residual_dx =
        detail::makeResidualJacobianAssembler( mesh_data, jacobian_data, mesh1, mesh2, pairs.size(), allocator_id );
    const MfemMatrixSpace normal_space{ jacobian_data.SurfacePath(), coords_fes };
    MfemMatrixAssembler gap_dn(
        MfemMatrixSpace{ pressure_path, field_fes }, normal_space,
        std::vector<MfemMatrixBlock>{ { 0, 0, num_nodes, num_nodes, 0, 0, elem_map1, elem_map1 } }, pairs.size(),
        allocator_id );
    MfemMatrixAssembler residual_dn(
        MfemMatrixSpace{ coordinate_path, coords_fes }, normal_space,
        std::vector<MfemMatrixBlock>{ { 0, 0, num_nodes, num_nodes, 0, 0, elem_map1, elem_map1 },
                                      { 1, 0, num_nodes, num_nodes, size_geom1_coords, 0, elem_map2, elem_map1 } },
        pairs.size(), allocator_id );

    const auto gap_view = gap.view();
    const auto residual_view = residual.view();
    const auto gap_dx_view = gap_dx.template view<num_nodes, size_geom1_coords, 2>();
    const auto frozen_gap_dx_view = frozen_gap_dx.template view<num_nodes, size_geom1_coords, 2>();
    const auto residual_dx_view = residual_dx.template view<size_geom1_coords, size_geom1_coords, 4>();
    const auto gap_dn_view = gap_dn.template view<num_nodes, size_geom1_coords, 1>();
    const auto residual_dn_view = residual_dn.template view<size_geom1_coords, size_geom1_coords, 2>();

    const bool ok = detail::runSingleMortarPairLoop<Exec, PairEvaluator>(
        pairs, mesh1, mesh2, normals.redecomp_normals, submesh_data.GetSubmeshFESpace().GetComm(),
        [=] TRIBOL_HOST_DEVICE( IndexT index, const InterfacePair& pair, typename PairEvaluator::Coords1 coords1,
                                typename PairEvaluator::Coords2 coords2, typename PairEvaluator::Coords1 normals1 ) {
          const auto pressures1 = detail::gatherElementPressures<num_nodes>( mesh1, pair.element_id1_, pressures );
          const auto differentiated =
              evaluator.evaluateAndDifferentiateResiduals( coords1, coords2, normals1, pressures1.view() );
          detail::assembleElementGapVector<num_nodes>( gap_view, map1[pair.element_id1_],
                                                       differentiated.evaluation.gap.gaps );
          detail::assembleElementResidualVectors<num_nodes, spatial_dim>(
              residual_view, map1[pair.element_id1_], map2[pair.element_id2_], differentiated.evaluation.residuals );
          gap_dx_view.packPair(
              index, pair, [&]( IndexT, IndexT row, IndexT column ) { return differentiated.gap_dx( row, column ); } );
          gap_dn_view.packPair(
              index, pair, [&]( IndexT, IndexT row, IndexT column ) { return differentiated.gap_dn( row, column ); } );
          frozen_gap_dx_view.packPair( index, pair, [&]( IndexT, IndexT row, IndexT column ) {
            return differentiated.evaluation.gap.frozen_gap_jacobian( row, column );
          } );
          residual_dx_view.packPair( index, pair, [&]( IndexT, IndexT row, IndexT column ) {
            return differentiated.residual_dx( row, column );
          } );
          residual_dn_view.packPair( index, pair, [&]( IndexT, IndexT row, IndexT column ) {
            return differentiated.residual_dn( row, column );
          } );
          return differentiated.evaluation.gap.metadata;
        } );
    if ( !ok ) {
      return {};
    }

    const shared::ParSparseMatView dn_dx( normals.dn_dx.get() );
    auto gap_jacobian = gap_dx.assemble();
    gap_jacobian += gap_dn.assemble() * dn_dx;
    auto residual_jacobian = residual_dx.assemble();
    residual_jacobian += residual_dn.assemble() * dn_dx;

    ContactFormulationResult result;
    result.gap.reset( gap.assemble().release() );
    result.residual.reset( residual.assemble().release() );
    result.pressure.reset( shared::ParVector( multiplier ).release() );
    result.df_dp.reset( frozen_gap_dx.assemble().transpose().release() );
    result.df_dx.reset( residual_jacobian.release() );
    result.weighted_gap_jacobian.reset( gap_jacobian.release() );
    return result;
  }
};
#endif  // TRIBOL_USE_ENZYME

/**
 * @brief Assemble nodal single mortar on parent displacement and submesh pressure true DOFs.
 * @tparam Exec Execution mode for search and evaluation kernels.
 * @tparam RuleGenerator Generates quadrature on the full projected overlap.
 * @tparam JacobianMode Selects frozen-geometry or exact Jacobian assembly
 * (SingleMortarFrozenJacobian or SingleMortarExactJacobian).
 */
template <ExecutionMode Exec, typename RuleGenerator, typename JacobianMode>
class SingleMortarFormulation final : public ContactFormulation {
  using PairEvaluator = detail::SingleMortarPairEvaluator<RuleGenerator>;
  static constexpr IndexT num_nodes = PairEvaluator::num_nodes1;
  static constexpr IndexT spatial_dim = PairEvaluator::spatial_dim;
  static_assert( PairEvaluator::num_nodes2 == num_nodes );

 public:
  /**
   * @brief Keep the search, integration, and Jacobian strategy settings.
   * @param rule_generator Generates the pair's quadrature.
   * @param jacobian_mode Selects frozen-geometry or exact Jacobian evaluation and owns the transfer references.
   * @param pair_builder Owns the coarse and refined searches.
   */
  SingleMortarFormulation( RuleGenerator rule_generator, JacobianMode jacobian_mode, PairListBuilder pair_builder )
      : evaluator_( std::move( rule_generator ) ),
        jacobian_mode_( std::move( jacobian_mode ) ),
        pair_builder_( std::move( pair_builder ) )
  {
  }

  /**
   * @brief Calculate assembled weighted gaps without pressure input.
   * @param mesh1 Current non-mortar mesh.
   * @param mesh2 Current mortar mesh.
   * @return Owned weighted gap and its coordinate Jacobian, or an empty result on input error.
   */
  ContactFormulationResult computeNodalGaps( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override
  {
    if ( !validateMeshes( mesh1, mesh2 ) ) {
      return {};
    }
    const auto refined_candidates = pair_builder_.build( mesh1, mesh2 );
    return jacobian_mode_.template computeGaps<Exec>( refined_candidates, mesh1, mesh2, evaluator_ );
  }

  /**
   * @brief Calculate the residual, weighted gap, pressure, and requested Jacobians.
   * @param mesh1 Current non-mortar mesh.
   * @param mesh2 Current mortar mesh.
   * @param inputs Includes pressure on the scalar submesh true DOFs.
   * @return Owned outputs, or an empty result on input error. Frozen-geometry mode leaves df_dx null.
   */
  ContactFormulationResult computeNodalResiduals( MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                  const ContactFormulationInputs& inputs ) const override
  {
    if ( !validateResidualInputs( mesh1, mesh2, inputs.multiplier ) ) {
      return {};
    }
    const auto refined_candidates = pair_builder_.build( mesh1, mesh2 );
    return jacobian_mode_.template computeResiduals<Exec>( refined_candidates, mesh1, mesh2, evaluator_,
                                                           *inputs.multiplier );
  }

 private:
  /**
   * @brief Check mesh and decomposition validity collectively across MPI ranks.
   * @param mesh1 Current non-mortar mesh, including reference coordinates.
   * @param mesh2 Current mortar mesh.
   * @return True when all ranks have valid meshes; false after reporting a root SLIC error.
   */
  bool validateMeshes( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
  {
    return validateCollective( hasValidMeshes( mesh1, mesh2 ) );
  }

  /**
   * @brief Check mesh, decomposition, and multiplier validity collectively across MPI ranks.
   * @param mesh1 Current non-mortar mesh, including reference coordinates.
   * @param mesh2 Current mortar mesh.
   * @param multiplier Caller-supplied multiplier on submesh true DOFs.
   * @return True when all ranks have valid meshes and multiplier; false after reporting a root SLIC error.
   */
  bool validateResidualInputs( MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                               const mfem::HypreParVector* multiplier ) const
  {
    const auto& dual_fes = jacobian_mode_.submesh_data.GetSubmeshFESpace();
    const bool valid_multiplier = multiplier && multiplier->Size() == dual_fes.GetTrueVSize() &&
                                  multiplier->GlobalSize() == dual_fes.GlobalTrueVSize();
    return validateCollective( hasValidMeshes( mesh1, mesh2 ) && valid_multiplier );
  }

  /**
   * @brief Check that the local mesh views and parent Jacobian transfer path satisfy single mortar requirements.
   * @param mesh1 Current non-mortar mesh, including reference coordinates.
   * @param mesh2 Current mortar mesh.
   * @return True if local mesh requirements hold.
   */
  bool hasValidMeshes( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
  {
    return jacobian_mode_.jacobian_data.HasParentPath() && mesh1.spatialDim() == spatial_dim &&
           mesh2.spatialDim() == spatial_dim && mesh1.getFaceOrder() == LINEAR_LAGRANGE &&
           mesh2.getFaceOrder() == LINEAR_LAGRANGE && mesh1.numberOfNodesPerElement() == num_nodes &&
           mesh2.numberOfNodesPerElement() == num_nodes &&
           ( mesh1.numberOfElements() == 0 || mesh1.hasReferencePosition() );
  }

  /**
   * @brief Reduce a local input-validity flag across all MPI ranks and report a root error on failure.
   * @param local_valid Whether this rank's inputs are valid.
   * @return True when every rank is valid; false after reporting a root SLIC error.
   */
  bool validateCollective( bool local_valid ) const
  {
    const auto comm = jacobian_mode_.submesh_data.GetSubmeshFESpace().GetComm();
    int invalid_input = !local_valid;
    MPI_Allreduce( MPI_IN_PLACE, &invalid_input, 1, MPI_INT, MPI_MAX, comm );
    if ( invalid_input ) {
      SLIC_ERROR_ROOT(
          "Single mortar requires current MFEM decomposition, reference coordinates, and an LM input "
          "on submesh true DOFs when calculating the residual." );
      return false;
    }
    return true;
  }

  PairEvaluator evaluator_;       ///< Shared pair evaluator for gap and residual calculations.
  JacobianMode jacobian_mode_;    ///< Selected frozen-geometry or exact Jacobian assembly strategy.
  PairListBuilder pair_builder_;  ///< Current search settings.
};

}  // namespace tribol

#endif  // BUILD_REDECOMP

#endif  // SRC_TRIBOL_PHYSICS_SINGLEMORTARFORMULATION_HPP_
