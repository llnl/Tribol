// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Evaluate 3D vector-LM frictional single mortar with frozen-geometry or exact Jacobian strategies. */
#ifndef SRC_TRIBOL_PHYSICS_FRICTIONALSINGLEMORTARFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_FRICTIONALSINGLEMORTARFORMULATION_HPP_

#include "tribol/config.hpp"
#ifdef BUILD_REDECOMP
#include <utility>
#include <vector>

#include "tribol/assembly/AssemblePairs.hpp"
#include "tribol/assembly/EdgeAveragedNodalNormalAssembler.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/integrals/FrictionalSingleMortarPairEvaluator.hpp"
#include "tribol/search/PairListBuilder.hpp"

namespace tribol {

namespace detail {

/**
 * @brief Gather an element's previous-step coordinates in [node][spatial_dim] order.
 * @tparam NumNodes Number of nodes on the surface element.
 * @tparam SpatialDim Spatial dimension.
 * @param mesh Surface mesh with previous-step coordinates registered.
 * @param element_id Local element ID within mesh.
 * @return Previous-step nodal coordinates on the element.
 */
template <IndexT NumNodes, IndexT SpatialDim>
TRIBOL_HOST_DEVICE MdArray<RealT, NumNodes, SpatialDim> gatherPreviousCoords( MeshData::Viewer mesh, IndexT element_id )
{
  MdArray<RealT, NumNodes, SpatialDim> prev_coords{};
  for ( IndexT node = 0; node < NumNodes; ++node ) {
    const IndexT node_id = mesh.getGlobalNodeId( element_id, node );
    for ( IndexT d = 0; d < SpatialDim; ++d ) {
      prev_coords( node, d ) = mesh.getPreviousPosition()[d][node_id];
    }
  }
  return prev_coords;
}

}  // namespace detail

/**
 * @brief Assemble frictional single mortar 3D kinematic gaps, residuals, nodal normals, and frozen-geometry Jacobians.
 *
 * In residual mode, df_dp is assembled as the transpose of the frozen kinematic-gap Jacobian and df_dx is left null.
 */
struct FrictionalSingleMortarFrozenJacobian {
  const MfemMeshData& mesh_data;          ///< Transfer data for coordinates and element numbering.
  const MfemSubmeshData& submesh_data;    ///< Transfer data for 3D vector LM and kinematic gaps.
  const MfemJacobianData& jacobian_data;  ///< Transfer paths for assembled true DOFs.

  /**
   * @brief Assemble 3D kinematic gaps, dual-space unit nodal normals, and the frozen kinematic-gap Jacobian.
   * @tparam Exec Concrete execution mode for normal assembly and pair evaluation.
   * @tparam PairEvaluator Configured frictional single mortar pair evaluator.
   * @param pairs Refined candidate pairs in local element numbering.
   * @param mesh1 Current non-mortar mesh with reference and previous-step coordinates.
   * @param mesh2 Current mortar mesh with previous-step coordinates.
   * @param evaluator Configured pair evaluator.
   * @return Owned 3D gap, nodal_normals, and frozen weighted_gap_jacobian, or empty on overflow.
   */
  template <ExecutionMode Exec, typename PairEvaluator>
  ContactFormulationResult computeGaps( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                        MeshData::Viewer mesh2, const PairEvaluator& evaluator ) const
  {
    constexpr IndexT num_nodes = PairEvaluator::num_nodes1;
    constexpr IndexT spatial_dim = PairEvaluator::spatial_dim;
    constexpr IndexT size_geom1_coords = PairEvaluator::size_geom1_coords;
    const auto normals =
        detail::assembleEdgeAveragedNodalNormalsWithSurface<Exec, num_nodes>( mesh1, mesh_data, jacobian_data );
    const auto* normal_values = normals.redecomp_normals.Read( isOnDevice( Exec ) );
    const IndexT num_normal_nodes = normals.redecomp_normals.FESpace()->GetNDofs();
    const auto allocator_id = pairs.getAllocatorID();
    const auto& field_fes = submesh_data.GetRedecompFESpace();
    const auto map1 = mesh_data.GetElemMap1().view();

    detail::MfemElementVectorAssembler gap( jacobian_data.SubmeshPath(), field_fes, allocator_id );
    auto gap_dx = detail::makeSubmeshFieldJacobianAssembler( mesh_data, submesh_data, jacobian_data, mesh1, mesh2,
                                                             pairs.size(), allocator_id );
    const auto gap_view = gap.view();
    const auto gap_dx_view = gap_dx.template view<size_geom1_coords, size_geom1_coords, 2>();

    const bool ok = detail::forEachPairChecked<Exec, typename PairEvaluator::Coords1, typename PairEvaluator::Coords2>(
        pairs, mesh1, mesh2, submesh_data.GetSubmeshFESpace().GetComm(),
        [=] TRIBOL_HOST_DEVICE( IndexT index, const InterfacePair& pair, typename PairEvaluator::Coords1 coords1,
                                typename PairEvaluator::Coords2 coords2 ) {
          const auto normals1 = detail::gatherRedecompNodalVectors<num_nodes, spatial_dim>(
              mesh1, pair.element_id1_, normal_values, num_normal_nodes );
          const auto prev_coords1 = detail::gatherPreviousCoords<num_nodes, spatial_dim>( mesh1, pair.element_id1_ );
          const auto prev_coords2 = detail::gatherPreviousCoords<num_nodes, spatial_dim>( mesh2, pair.element_id2_ );
          const auto values =
              evaluator.evaluateGaps( coords1, coords2, prev_coords1.view(), prev_coords2.view(), normals1.view() );
          detail::assembleElementFieldVector<num_nodes, spatial_dim>( gap_view, map1[pair.element_id1_],
                                                                      values.kinematic_gap );
          gap_dx_view.packPair( index, pair, [&]( IndexT, IndexT row, IndexT column ) {
            return values.frozen_gap_jacobian( row, column );
          } );
          return values.metadata;
        } );
    if ( !ok ) {
      return {};
    }

    auto dual_normals =
        detail::transferEdgeAveragedNodalNormalsToDualSpace( normals.surface_normals, nullptr, jacobian_data );
    ContactFormulationResult result;
    result.gap.reset( gap.assemble().release() );
    result.weighted_gap_jacobian.reset( gap_dx.assemble().release() );
    result.nodal_normals = std::move( dual_normals.normals );
    return result;
  }

  /**
   * @brief Assemble 3D kinematic gaps, residuals, multipliers, nodal normals, frozen gap Jacobian, and df_dp.
   * @tparam Exec Concrete execution mode for normal assembly and pair evaluation.
   * @tparam PairEvaluator Configured frictional single mortar pair evaluator.
   * @param pairs Refined candidate pairs in local element numbering.
   * @param mesh1 Current non-mortar mesh with reference and previous-step coordinates.
   * @param mesh2 Current mortar mesh with previous-step coordinates.
   * @param evaluator Configured pair evaluator.
   * @param multiplier Caller-supplied 3D global Cartesian multipliers on submesh true DOFs.
   * @return Owned gap, residual, pressure, nodal_normals, weighted_gap_jacobian, and df_dp; df_dx is null.
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
        detail::assembleEdgeAveragedNodalNormalsWithSurface<Exec, num_nodes>( mesh1, mesh_data, jacobian_data );
    const auto* normal_values = normals.redecomp_normals.Read( isOnDevice( Exec ) );
    const IndexT num_normal_nodes = normals.redecomp_normals.FESpace()->GetNDofs();
    const auto redecomp_multiplier = submesh_data.TransferTrueDofsToRedecomp( multiplier );
    const auto* multipliers = redecomp_multiplier.Read( isOnDevice( Exec ) );
    const IndexT num_multiplier_nodes = redecomp_multiplier.FESpace()->GetNDofs();
    const auto allocator_id = pairs.getAllocatorID();
    const auto& coords_fes = *mesh_data.GetRedecompResponse().FESpace();
    const auto& field_fes = submesh_data.GetRedecompFESpace();
    const auto map1 = mesh_data.GetElemMap1().view();
    const auto map2 = mesh_data.GetElemMap2().view();

    detail::MfemElementVectorAssembler gap( jacobian_data.SubmeshPath(), field_fes, allocator_id );
    detail::MfemElementVectorAssembler residual( jacobian_data.ParentPath(), coords_fes, allocator_id );
    auto gap_dx = detail::makeSubmeshFieldJacobianAssembler( mesh_data, submesh_data, jacobian_data, mesh1, mesh2,
                                                             pairs.size(), allocator_id );
    const auto gap_view = gap.view();
    const auto residual_view = residual.view();
    const auto gap_dx_view = gap_dx.template view<size_geom1_coords, size_geom1_coords, 2>();

    const bool ok = detail::forEachPairChecked<Exec, typename PairEvaluator::Coords1, typename PairEvaluator::Coords2>(
        pairs, mesh1, mesh2, submesh_data.GetSubmeshFESpace().GetComm(),
        [=] TRIBOL_HOST_DEVICE( IndexT index, const InterfacePair& pair, typename PairEvaluator::Coords1 coords1,
                                typename PairEvaluator::Coords2 coords2 ) {
          const auto normals1 = detail::gatherRedecompNodalVectors<num_nodes, spatial_dim>(
              mesh1, pair.element_id1_, normal_values, num_normal_nodes );
          const auto prev_coords1 = detail::gatherPreviousCoords<num_nodes, spatial_dim>( mesh1, pair.element_id1_ );
          const auto prev_coords2 = detail::gatherPreviousCoords<num_nodes, spatial_dim>( mesh2, pair.element_id2_ );
          const auto multipliers1 = detail::gatherRedecompNodalVectors<num_nodes, spatial_dim>(
              mesh1, pair.element_id1_, multipliers, num_multiplier_nodes );
          const auto values = evaluator.evaluateResiduals( coords1, coords2, prev_coords1.view(), prev_coords2.view(),
                                                           normals1.view(), multipliers1.view() );
          detail::assembleElementFieldVector<num_nodes, spatial_dim>( gap_view, map1[pair.element_id1_],
                                                                      values.gap.kinematic_gap );
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
    auto dual_normals =
        detail::transferEdgeAveragedNodalNormalsToDualSpace( normals.surface_normals, nullptr, jacobian_data );
    ContactFormulationResult result;
    result.gap.reset( gap.assemble().release() );
    result.residual.reset( residual.assemble().release() );
    result.pressure.reset( shared::ParVector( multiplier ).release() );
    result.df_dp.reset( gap_jacobian.transpose().release() );
    result.weighted_gap_jacobian.reset( gap_jacobian.release() );
    result.nodal_normals = std::move( dual_normals.normals );
    return result;
  }
};

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Assemble frictional single mortar 3D kinematic gaps, residuals, nodal normals, and exact Jacobians.
 *
 * Includes mortar weight derivatives and normal derivatives (gap_dn * dn_dx) for weighted_gap_jacobian, mortar weight
 * derivatives for df_dx, and transfers both nodal_normals and dn_dx onto dual submesh true DOFs.
 */
struct FrictionalSingleMortarExactJacobian {
  const MfemMeshData& mesh_data;          ///< Transfer data for coordinates and element numbering.
  const MfemSubmeshData& submesh_data;    ///< Transfer data for 3D vector LM and kinematic gaps.
  const MfemJacobianData& jacobian_data;  ///< Transfer paths for assembled true DOFs.

  /**
   * @brief Assemble 3D kinematic gaps, dual-space unit nodal normals, dn_dx, and exact weighted_gap_jacobian.
   * @tparam Exec Concrete execution mode for normal assembly and pair evaluation.
   * @tparam PairEvaluator Configured frictional single mortar pair evaluator.
   * @param pairs Refined candidate pairs in local element numbering.
   * @param mesh1 Current non-mortar mesh with reference and previous-step coordinates.
   * @param mesh2 Current mortar mesh with previous-step coordinates.
   * @param evaluator Configured pair evaluator.
   * @return Owned 3D gap, nodal_normals, dn_dx, and exact weighted_gap_jacobian, or empty on overflow.
   */
  template <ExecutionMode Exec, typename PairEvaluator>
  ContactFormulationResult computeGaps( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                        MeshData::Viewer mesh2, const PairEvaluator& evaluator ) const
  {
    constexpr IndexT num_nodes = PairEvaluator::num_nodes1;
    constexpr IndexT spatial_dim = PairEvaluator::spatial_dim;
    constexpr IndexT size_geom1_coords = PairEvaluator::size_geom1_coords;
    const auto normals =
        detail::assembleEdgeAveragedNodalNormalsAndJacobian<Exec, num_nodes>( mesh1, mesh_data, jacobian_data );
    const auto* normal_values = normals.redecomp_normals.Read( isOnDevice( Exec ) );
    const IndexT num_normal_nodes = normals.redecomp_normals.FESpace()->GetNDofs();
    const auto allocator_id = pairs.getAllocatorID();
    const auto& coords_fes = *mesh_data.GetRedecompResponse().FESpace();
    const auto& field_fes = submesh_data.GetRedecompFESpace();
    const auto& pressure_path = jacobian_data.SubmeshPath();
    const auto& elem_map1 = mesh_data.GetElemMap1();
    const auto map1 = elem_map1.view();

    detail::MfemElementVectorAssembler gap( pressure_path, field_fes, allocator_id );
    auto gap_dx = detail::makeSubmeshFieldJacobianAssembler( mesh_data, submesh_data, jacobian_data, mesh1, mesh2,
                                                             pairs.size(), allocator_id );
    const MfemMatrixSpace normal_space{ jacobian_data.SurfacePath(), coords_fes };
    MfemMatrixAssembler gap_dn(
        MfemMatrixSpace{ pressure_path, field_fes }, normal_space,
        std::vector<MfemMatrixBlock>{ { 0, 0, num_nodes, num_nodes, 0, 0, elem_map1, elem_map1 } }, pairs.size(),
        allocator_id );
    const auto gap_view = gap.view();
    const auto gap_dx_view = gap_dx.template view<size_geom1_coords, size_geom1_coords, 2>();
    const auto gap_dn_view = gap_dn.template view<size_geom1_coords, size_geom1_coords, 1>();

    const bool ok = detail::forEachPairChecked<Exec, typename PairEvaluator::Coords1, typename PairEvaluator::Coords2>(
        pairs, mesh1, mesh2, submesh_data.GetSubmeshFESpace().GetComm(),
        [=] TRIBOL_HOST_DEVICE( IndexT index, const InterfacePair& pair, typename PairEvaluator::Coords1 coords1,
                                typename PairEvaluator::Coords2 coords2 ) {
          const auto normals1 = detail::gatherRedecompNodalVectors<num_nodes, spatial_dim>(
              mesh1, pair.element_id1_, normal_values, num_normal_nodes );
          const auto prev_coords1 = detail::gatherPreviousCoords<num_nodes, spatial_dim>( mesh1, pair.element_id1_ );
          const auto prev_coords2 = detail::gatherPreviousCoords<num_nodes, spatial_dim>( mesh2, pair.element_id2_ );
          const auto differentiated = evaluator.evaluateAndDifferentiateGaps( coords1, coords2, prev_coords1.view(),
                                                                              prev_coords2.view(), normals1.view() );
          detail::assembleElementFieldVector<num_nodes, spatial_dim>( gap_view, map1[pair.element_id1_],
                                                                      differentiated.evaluation.kinematic_gap );
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
    auto dual_normals = detail::transferEdgeAveragedNodalNormalsToDualSpace( normals.surface_normals,
                                                                             normals.dn_dx.get(), jacobian_data );

    ContactFormulationResult result;
    result.gap.reset( gap.assemble().release() );
    result.weighted_gap_jacobian.reset( gap_jacobian.release() );
    result.nodal_normals = std::move( dual_normals.normals );
    result.dn_dx = std::move( dual_normals.dn_dx );
    return result;
  }

  /**
   * @brief Assemble 3D kinematic gaps, residuals, multipliers, nodal normals, dn_dx, df_dp, df_dx, and exact dg_dx.
   * @tparam Exec Concrete execution mode for normal assembly and pair evaluation.
   * @tparam PairEvaluator Configured frictional single mortar pair evaluator.
   * @param pairs Refined candidate pairs in local element numbering.
   * @param mesh1 Current non-mortar mesh with reference and previous-step coordinates.
   * @param mesh2 Current mortar mesh with previous-step coordinates.
   * @param evaluator Configured pair evaluator.
   * @param multiplier Caller-supplied 3D global Cartesian multipliers on submesh true DOFs.
   * @return Owned gap, residual, pressure, nodal_normals, dn_dx, df_dp, df_dx, and weighted_gap_jacobian.
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
    const auto* normal_values = normals.redecomp_normals.Read( isOnDevice( Exec ) );
    const IndexT num_normal_nodes = normals.redecomp_normals.FESpace()->GetNDofs();
    const auto redecomp_multiplier = submesh_data.TransferTrueDofsToRedecomp( multiplier );
    const auto* multipliers = redecomp_multiplier.Read( isOnDevice( Exec ) );
    const IndexT num_multiplier_nodes = redecomp_multiplier.FESpace()->GetNDofs();
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
    auto gap_dx = detail::makeSubmeshFieldJacobianAssembler( mesh_data, submesh_data, jacobian_data, mesh1, mesh2,
                                                             pairs.size(), allocator_id );
    auto frozen_gap_dx = detail::makeSubmeshFieldJacobianAssembler( mesh_data, submesh_data, jacobian_data, mesh1,
                                                                    mesh2, pairs.size(), allocator_id );
    auto residual_dx =
        detail::makeResidualJacobianAssembler( mesh_data, jacobian_data, mesh1, mesh2, pairs.size(), allocator_id );
    const MfemMatrixSpace normal_space{ jacobian_data.SurfacePath(), coords_fes };
    MfemMatrixAssembler gap_dn(
        MfemMatrixSpace{ pressure_path, field_fes }, normal_space,
        std::vector<MfemMatrixBlock>{ { 0, 0, num_nodes, num_nodes, 0, 0, elem_map1, elem_map1 } }, pairs.size(),
        allocator_id );

    const auto gap_view = gap.view();
    const auto residual_view = residual.view();
    const auto gap_dx_view = gap_dx.template view<size_geom1_coords, size_geom1_coords, 2>();
    const auto frozen_gap_dx_view = frozen_gap_dx.template view<size_geom1_coords, size_geom1_coords, 2>();
    const auto residual_dx_view = residual_dx.template view<size_geom1_coords, size_geom1_coords, 4>();
    const auto gap_dn_view = gap_dn.template view<size_geom1_coords, size_geom1_coords, 1>();

    const bool ok = detail::forEachPairChecked<Exec, typename PairEvaluator::Coords1, typename PairEvaluator::Coords2>(
        pairs, mesh1, mesh2, submesh_data.GetSubmeshFESpace().GetComm(),
        [=] TRIBOL_HOST_DEVICE( IndexT index, const InterfacePair& pair, typename PairEvaluator::Coords1 coords1,
                                typename PairEvaluator::Coords2 coords2 ) {
          const auto normals1 = detail::gatherRedecompNodalVectors<num_nodes, spatial_dim>(
              mesh1, pair.element_id1_, normal_values, num_normal_nodes );
          const auto prev_coords1 = detail::gatherPreviousCoords<num_nodes, spatial_dim>( mesh1, pair.element_id1_ );
          const auto prev_coords2 = detail::gatherPreviousCoords<num_nodes, spatial_dim>( mesh2, pair.element_id2_ );
          const auto multipliers1 = detail::gatherRedecompNodalVectors<num_nodes, spatial_dim>(
              mesh1, pair.element_id1_, multipliers, num_multiplier_nodes );
          const auto differentiated = evaluator.evaluateAndDifferentiateResiduals(
              coords1, coords2, prev_coords1.view(), prev_coords2.view(), normals1.view(), multipliers1.view() );
          detail::assembleElementFieldVector<num_nodes, spatial_dim>( gap_view, map1[pair.element_id1_],
                                                                      differentiated.evaluation.gap.kinematic_gap );
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
          return differentiated.evaluation.gap.metadata;
        } );
    if ( !ok ) {
      return {};
    }

    const shared::ParSparseMatView dn_dx( normals.dn_dx.get() );
    auto gap_jacobian = gap_dx.assemble();
    gap_jacobian += gap_dn.assemble() * dn_dx;
    auto dual_normals = detail::transferEdgeAveragedNodalNormalsToDualSpace( normals.surface_normals,
                                                                             normals.dn_dx.get(), jacobian_data );

    ContactFormulationResult result;
    result.gap.reset( gap.assemble().release() );
    result.residual.reset( residual.assemble().release() );
    result.pressure.reset( shared::ParVector( multiplier ).release() );
    result.df_dp.reset( frozen_gap_dx.assemble().transpose().release() );
    result.df_dx.reset( residual_dx.assemble().release() );
    result.weighted_gap_jacobian.reset( gap_jacobian.release() );
    result.nodal_normals = std::move( dual_normals.normals );
    result.dn_dx = std::move( dual_normals.dn_dx );
    return result;
  }
};
#endif  // TRIBOL_USE_ENZYME

/**
 * @brief Assemble 3D vector-LM frictional single mortar on parent displacement and 3D submesh multiplier true DOFs.
 * @tparam Exec Execution mode for search and evaluation kernels.
 * @tparam RuleGenerator Generates quadrature on the full projected overlap.
 * @tparam JacobianMode Selects frozen-geometry or exact Jacobian assembly
 * (FrictionalSingleMortarFrozenJacobian or FrictionalSingleMortarExactJacobian).
 */
template <ExecutionMode Exec, typename RuleGenerator, typename JacobianMode>
class FrictionalSingleMortarFormulation final : public ContactFormulation {
  using PairEvaluator = detail::FrictionalSingleMortarPairEvaluator<RuleGenerator>;
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
  FrictionalSingleMortarFormulation( RuleGenerator rule_generator, JacobianMode jacobian_mode,
                                     PairListBuilder pair_builder )
      : evaluator_( std::move( rule_generator ) ),
        jacobian_mode_( std::move( jacobian_mode ) ),
        pair_builder_( std::move( pair_builder ) )
  {
  }

  /**
   * @brief Calculate assembled 3D kinematic gaps and dual-space unit nodal normals without multiplier input.
   * @param mesh1 Current non-mortar mesh.
   * @param mesh2 Current mortar mesh.
   * @return Owned 3D gap, nodal_normals, weighted_gap_jacobian, and optional dn_dx, or empty on input error.
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
   * @brief Calculate the residual, 3D kinematic gap, multipliers, nodal normals, and requested Jacobians.
   * @param mesh1 Current non-mortar mesh.
   * @param mesh2 Current mortar mesh.
   * @param inputs Includes 3D global Cartesian multipliers on the vector submesh true DOFs.
   * @return Owned outputs, or an empty result on input error. Frozen-geometry mode leaves df_dx and dn_dx null.
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
   * @brief Check mesh, decomposition, and previous-coordinate validity collectively across MPI ranks.
   * @param mesh1 Current non-mortar mesh, including reference and previous-step coordinates.
   * @param mesh2 Current mortar mesh, including previous-step coordinates.
   * @return True when all ranks have valid meshes; false after reporting a root SLIC error.
   */
  bool validateMeshes( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
  {
    return validateCollective( hasValidMeshes( mesh1, mesh2 ) );
  }

  /**
   * @brief Check mesh, decomposition, previous-coordinate, and 3D multiplier validity collectively across MPI ranks.
   * @param mesh1 Current non-mortar mesh, including reference and previous-step coordinates.
   * @param mesh2 Current mortar mesh, including previous-step coordinates.
   * @param multiplier Caller-supplied 3D vector multiplier on submesh true DOFs.
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
   * @brief Check that local mesh views, reference/previous coordinates, and vector dual space are valid.
   * @param mesh1 Current non-mortar mesh.
   * @param mesh2 Current mortar mesh.
   * @return True if local mesh requirements hold.
   */
  bool hasValidMeshes( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
  {
    const auto& dual_fes = jacobian_mode_.submesh_data.GetSubmeshFESpace();
    return jacobian_mode_.jacobian_data.HasParentPath() && dual_fes.GetVDim() == spatial_dim &&
           mesh1.spatialDim() == spatial_dim && mesh2.spatialDim() == spatial_dim &&
           mesh1.getFaceOrder() == LINEAR_LAGRANGE && mesh2.getFaceOrder() == LINEAR_LAGRANGE &&
           mesh1.numberOfNodesPerElement() == num_nodes && mesh2.numberOfNodesPerElement() == num_nodes &&
           ( mesh1.numberOfElements() == 0 || ( mesh1.hasReferencePosition() && mesh1.hasPreviousPosition() ) ) &&
           ( mesh2.numberOfElements() == 0 || mesh2.hasPreviousPosition() );
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
          "Frictional single mortar requires current MFEM decomposition, reference coordinates, previous-step "
          "coordinates, a 3D vector submesh space, and a 3D LM input on submesh true DOFs when calculating the "
          "residual." );
      return false;
    }
    return true;
  }

  PairEvaluator evaluator_;       ///< Shared pair evaluator for 3D gap and residual calculations.
  JacobianMode jacobian_mode_;    ///< Selected frozen-geometry or exact Jacobian assembly strategy.
  PairListBuilder pair_builder_;  ///< Current search settings.
};

}  // namespace tribol

#endif  // BUILD_REDECOMP

#endif  // SRC_TRIBOL_PHYSICS_FRICTIONALSINGLEMORTARFORMULATION_HPP_
