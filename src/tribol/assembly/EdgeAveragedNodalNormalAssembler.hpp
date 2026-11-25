// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Assemble the single mortar edge-averaged nodal normal field on the linear contact surface. */
#ifndef SRC_TRIBOL_ASSEMBLY_EDGEAVERAGEDNODALNORMALASSEMBLER_HPP_
#define SRC_TRIBOL_ASSEMBLY_EDGEAVERAGEDNODALNORMALASSEMBLER_HPP_

#include "tribol/config.hpp"
#ifdef BUILD_REDECOMP
#include <cmath>
#include <memory>

#include "tribol/assembly/MfemMatrixAssembler.hpp"
#include "tribol/assembly/MfemVectorAssembler.hpp"
#include "tribol/geom/EdgeAveragedNodalNormalKernel.hpp"
#include "redecomp/transfer/TransferByElements.hpp"

namespace tribol::detail {

/** @brief Owned normal field on Redecomp and on linear surface true DOFs (without dn_dx). */
struct AssembledEdgeAveragedNodalNormalField {
  mfem::GridFunction redecomp_normals;  ///< Field read by pair kernels. Its FE space belongs to MfemMeshData.
  shared::ParVector surface_normals;    ///< Normalized nodal normals on linear surface true DOFs.
};

/** @brief Owned normal field and its coordinate derivative on linear surface true DOFs. */
struct AssembledEdgeAveragedNodalNormals {
  mfem::GridFunction redecomp_normals;          ///< Field read by pair kernels. Its FE space belongs to MfemMeshData.
  shared::ParVector surface_normals;            ///< Normalized nodal normals on linear surface true DOFs.
  std::unique_ptr<mfem::HypreParMatrix> dn_dx;  ///< Normal rows on linear surface true DOFs and parent coord columns.
};

/** @brief Host-facing unit nodal normals and optional coordinate derivative on dual submesh true DOFs. */
struct DualEdgeAveragedNodalNormals {
  std::unique_ptr<mfem::HypreParVector> normals;  ///< Unit nodal normals on dual submesh true DOFs.
  std::unique_ptr<mfem::HypreParMatrix> dn_dx;  ///< Derivative of normals with respect to parent coordinates, or null.
};

/** @brief Intermediate primal normal data reused when assembling dn_dx. */
struct OwnedEdgeAveragedNodalNormalState {
  mfem::GridFunction redecomp_normals;     ///< Normalized nodal normals transferred to Redecomp.
  shared::ParVector surface_normals;       ///< Normalized nodal normals on linear surface true DOFs.
  Array1D<InterfacePair> owned_elements;   ///< Owned non-mortar elements on this rank.
  mfem::Vector inverse_reference_lengths;  ///< Per-true-DOF inverse reference sum magnitudes.
};

/**
 * @brief Gather an element's reference coordinates in the same order as its current coordinates.
 * @tparam NormalKernel Kernel that supplies the coordinate layout.
 * @param mesh First mesh, with reference coordinates registered.
 * @param element Element ID within that mesh.
 * @return Reference coordinates in node, then spatial-coordinate order.
 */
template <typename NormalKernel>
TRIBOL_HOST_DEVICE typename NormalKernel::NodalVectors gatherReferenceCoords( MeshData::Viewer mesh, IndexT element )
{
  typename NormalKernel::NodalVectors coords{};
  for ( IndexT node = 0; node < coords.template extent<0>(); ++node ) {
    const IndexT node_id = mesh.getGlobalNodeId( element, node );
    for ( IndexT d = 0; d < NormalKernel::spatial_dim; ++d ) {
      coords( node, d ) = mesh.getReferencePosition()[d][node_id];
    }
  }
  return coords;
}

/**
 * @brief Sum owned-element normal contributions, normalize on linear surface true DOFs, and transfer to Redecomp.
 * @tparam Exec Execution mode of the element kernels.
 * @tparam NumNodes Number of nodes per linear surface element.
 * @param mesh1 Current non-mortar mesh with registered reference coordinates.
 * @param mesh_data Current Redecomp FE space and element map.
 * @param jacobian_data Initialized surface and parent transfer paths.
 * @return Normalized Redecomp normals, surface true-DOF normals, owned elements, and true-DOF normalization factors.
 */
template <ExecutionMode Exec, IndexT NumNodes>
OwnedEdgeAveragedNodalNormalState assembleOwnedEdgeAveragedNodalNormals( MeshData::Viewer mesh1,
                                                                         const MfemMeshData& mesh_data,
                                                                         const MfemJacobianData& jacobian_data )
{
  using NormalKernel = EdgeAveragedNodalNormalKernel<NumNodes>;
  using Coords = typename NormalKernel::Coords;
  const auto& redecomp_fes = *mesh_data.GetRedecompResponse().FESpace();
  const auto& redecomp_mesh = dynamic_cast<const redecomp::RedecompMesh&>( *redecomp_fes.GetMesh() );
  const auto& surface_path = jacobian_data.SurfacePath();
  const int allocator_id = mesh1.getAllocatorId();
  const auto& surface_fes = *surface_path.final_fes;

  Array1D<bool, MemorySpace::Host> owns_element( redecomp_mesh.GetNE() );
  owns_element.fill( true );
  for ( const auto& ghosts : redecomp_mesh.getRedecompToParentGhostElems() ) {
    for ( const auto element : ghosts ) {
      owns_element[element] = false;
    }
  }
  const Array1D<int, MemorySpace::Host> host_map( mesh_data.GetElemMap1() );
  Array1D<InterfacePair, MemorySpace::Host> host_elements( 0, host_map.size() );
  for ( IndexT element = 0; element < host_map.size(); ++element ) {
    if ( owns_element[host_map[element]] ) {
      host_elements.push_back( { element, element } );
    }
  }
  Array1D<InterfacePair> elements( host_elements, allocator_id );
  const auto element_view = elements.view();
  const auto element_map = mesh_data.GetElemMap1().view();

  MfemElementVectorAssembler current( surface_path, redecomp_fes, allocator_id );
  MfemElementVectorAssembler reference( surface_path, redecomp_fes, allocator_id );
  const auto current_values = current.view();
  const auto reference_values = reference.view();
  forAllExec<Exec>( elements.size(), [=] TRIBOL_HOST_DEVICE( IndexT index ) {
    const auto element = element_view[index].element_id1_;
    const auto coords = mesh1.getFaceCoords<Coords>( element );
    const auto ref_coords = gatherReferenceCoords<NormalKernel>( mesh1, element );
    const auto normals = NormalKernel::evaluate( coords.view(), ref_coords.view() );
    const auto ref_normals = NormalKernel::evaluate( ref_coords.view(), ref_coords.view() );
    for ( IndexT d = 0; d < NormalKernel::spatial_dim; ++d ) {
      for ( IndexT node = 0; node < NumNodes; ++node ) {
        current_values( element_map[element], d * NumNodes + node ) = normals( node, d );
        reference_values( element_map[element], d * NumNodes + node ) = ref_normals( node, d );
      }
    }
  } );

  auto normals = current.assemble();
  const auto reference_normals = reference.assemble();
  mfem::Vector scale( surface_fes.GetTrueVSize() );
  scale = 0.0;
  // The reference sum is dimensionless because each reference corner contributes a unit vector.
  constexpr RealT minimum_reference_sum_magnitude = 1.0e-15;
  for ( int node = 0; node < surface_fes.GetNDofs(); ++node ) {
    MdArray<int, NormalKernel::spatial_dim> true_dofs{};
    for ( IndexT d = 0; d < NormalKernel::spatial_dim; ++d ) {
      true_dofs[d] = surface_fes.GetLocalTDofNumber( surface_fes.DofToVDof( node, d ) );
    }
    if ( true_dofs[0] < 0 ) {
      continue;
    }
    RealT magnitude_squared = 0.0;
    for ( IndexT d = 0; d < NormalKernel::spatial_dim; ++d ) {
      const RealT value = reference_normals.get()( true_dofs[d] );
      magnitude_squared += value * value;
    }
    const RealT magnitude = std::sqrt( magnitude_squared );
    const RealT inverse_magnitude = magnitude >= minimum_reference_sum_magnitude ? 1.0 / magnitude : 0.0;
    for ( IndexT d = 0; d < NormalKernel::spatial_dim; ++d ) {
      scale[true_dofs[d]] = inverse_magnitude;
      normals[true_dofs[d]] *= inverse_magnitude;
    }
  }

  mfem::GridFunction redecomp_normals( const_cast<mfem::FiniteElementSpace*>( &redecomp_fes ) );
  mfem::ParGridFunction surface_normals( const_cast<mfem::ParFiniteElementSpace*>( &surface_fes ) );
  {
    shared::detail::HypreMemoryLocationGuard guard( shared::ParVectorMemory::Host );
    surface_path.ops.front()->Mult( normals.get(), surface_normals );
  }
  surface_normals.UseDevice( isOnDevice( Exec ) );
  redecomp_normals.UseDevice( isOnDevice( Exec ) );
  redecomp::TransferByElements{}.TransferToSerial( surface_normals, redecomp_normals );
  return { std::move( redecomp_normals ), std::move( normals ), std::move( elements ), std::move( scale ) };
}

/**
 * @brief Assemble edge-averaged nodal normals on Redecomp without computing their coordinate derivative.
 * @tparam Exec Execution mode of the element kernels.
 * @tparam NumNodes Number of nodes per linear surface element.
 * @param mesh1 Current non-mortar mesh with registered reference coordinates.
 * @param mesh_data Current Redecomp FE space and element map.
 * @param jacobian_data Initialized surface and parent transfer paths.
 * @return Normalized nodal normal field on Redecomp.
 */
template <ExecutionMode Exec, IndexT NumNodes>
mfem::GridFunction assembleEdgeAveragedNodalNormals( MeshData::Viewer mesh1, const MfemMeshData& mesh_data,
                                                     const MfemJacobianData& jacobian_data )
{
  return assembleOwnedEdgeAveragedNodalNormals<Exec, NumNodes>( mesh1, mesh_data, jacobian_data ).redecomp_normals;
}

/**
 * @brief Assemble edge-averaged nodal normals on both Redecomp and linear surface true DOFs.
 * @tparam Exec Execution mode of the element kernels.
 * @tparam NumNodes Number of nodes per linear surface element.
 * @param mesh1 Current non-mortar mesh with registered reference coordinates.
 * @param mesh_data Current Redecomp FE space and element map.
 * @param jacobian_data Initialized surface and parent transfer paths.
 * @return Normalized nodal normals on Redecomp and on linear surface true DOFs.
 */
template <ExecutionMode Exec, IndexT NumNodes>
AssembledEdgeAveragedNodalNormalField assembleEdgeAveragedNodalNormalsWithSurface(
    MeshData::Viewer mesh1, const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data )
{
  auto state = assembleOwnedEdgeAveragedNodalNormals<Exec, NumNodes>( mesh1, mesh_data, jacobian_data );
  return { std::move( state.redecomp_normals ), std::move( state.surface_normals ) };
}

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Assemble edge-averaged nodal normals on Redecomp together with their coordinate Jacobian dn_dx.
 * @tparam Exec Execution mode of the element kernels.
 * @tparam NumNodes Number of nodes per linear surface element.
 * @param mesh1 Current non-mortar mesh with registered reference coordinates.
 * @param mesh_data Current Redecomp FE space and element map.
 * @param jacobian_data Initialized surface and parent transfer paths.
 * @return Normalized nodal normal field on Redecomp, on surface true DOFs, and its coordinate Jacobian dn_dx.
 */
template <ExecutionMode Exec, IndexT NumNodes>
AssembledEdgeAveragedNodalNormals assembleEdgeAveragedNodalNormalsAndJacobian( MeshData::Viewer mesh1,
                                                                               const MfemMeshData& mesh_data,
                                                                               const MfemJacobianData& jacobian_data )
{
  using NormalKernel = EdgeAveragedNodalNormalKernel<NumNodes>;
  using Coords = typename NormalKernel::Coords;
  constexpr IndexT size_coords = NormalKernel::size_coords;
  auto state = assembleOwnedEdgeAveragedNodalNormals<Exec, NumNodes>( mesh1, mesh_data, jacobian_data );
  const auto& redecomp_fes = *mesh_data.GetRedecompResponse().FESpace();
  const MfemMatrixSpace normal_space{ jacobian_data.SurfacePath(), redecomp_fes };
  const MfemMatrixSpace coordinate_space{ jacobian_data.ParentPath(), redecomp_fes };
  const auto& map1 = mesh_data.GetElemMap1();
  const auto element_view = state.owned_elements.view();
  MfemMatrixAssembler assembler( normal_space, coordinate_space, { { 0, 0, NumNodes, NumNodes, 0, 0, map1, map1 } },
                                 state.owned_elements.size(), mesh1.getAllocatorId() );
  const auto matrix = assembler.template view<size_coords, size_coords, 1>();
  forAllExec<Exec>( state.owned_elements.size(), [=] TRIBOL_HOST_DEVICE( IndexT index ) {
    const auto element = element_view[index].element_id1_;
    const auto coords = mesh1.getFaceCoords<Coords>( element );
    const auto ref_coords = gatherReferenceCoords<NormalKernel>( mesh1, element );
    const auto jacobian = NormalKernel::differentiate( coords.view(), ref_coords.view() );
    matrix.packPair( index, element_view[index],
                     [&]( IndexT, IndexT row, IndexT column ) { return jacobian( row, column ); } );
  } );
  auto dn_dx = assembler.assemble();
  shared::detail::HypreMemoryLocationGuard guard( shared::ParVectorMemory::Host );
  dn_dx->ScaleRows( state.inverse_reference_lengths );
  return { std::move( state.redecomp_normals ), std::move( state.surface_normals ),
           std::unique_ptr<mfem::HypreParMatrix>( dn_dx.release() ) };
}
#endif

/**
 * @brief Transfer linear-surface nodal normals (and optional dn_dx) onto the dual submesh true-DOF space.
 *
 * For linear meshes without LOR, SurfacePath and SubmeshPath share the same FE space, so the vectors and matrices are
 * returned directly. For higher-order LOR meshes, the linear-surface normals are mapped onto the higher-order dual
 * submesh via (T_ho_to_lor * P_sub)^T * P_lor, renormalized to unit length at each owned dual node, and the optional
 * dn_dx is multiplied by the exact block-diagonal normalization Jacobian.
 *
 * @param surface_normals Normalized nodal normals on linear surface true DOFs.
 * @param surface_dn_dx Optional derivative of surface_normals with respect to parent coordinates (may be null).
 * @param jacobian_data Initialized surface and vector dual submesh transfer paths.
 * @return Unit nodal normals and optional coordinate derivative on dual submesh true DOFs.
 */
inline DualEdgeAveragedNodalNormals transferEdgeAveragedNodalNormalsToDualSpace(
    const shared::ParVector& surface_normals, const mfem::HypreParMatrix* surface_dn_dx,
    const MfemJacobianData& jacobian_data )
{
  const auto& submesh_path = jacobian_data.SubmeshPath();
  const auto& surface_path = jacobian_data.SurfacePath();
  const auto& dual_fes = *submesh_path.final_fes;
  constexpr int spatial_dim = 3;
  const bool uses_lor = submesh_path.ops.size() > 1;

  std::unique_ptr<shared::ParSparseMat> lor_to_dual;
  if ( uses_lor ) {
    const auto submesh_to_lor = submesh_path.ops[1] * submesh_path.ops[0];
    lor_to_dual = std::make_unique<shared::ParSparseMat>( submesh_to_lor.transpose() * surface_path.ops[0] );
  }
  auto dual_normals = uses_lor ? ( *lor_to_dual * surface_normals ) : shared::ParVector( surface_normals.get() );

  const int true_vsize = dual_fes.GetTrueVSize();
  mfem::SparseMatrix local_norm_jac( true_vsize, true_vsize );
  constexpr RealT minimum_normal_magnitude = 1.0e-15;
  for ( int node = 0; node < dual_fes.GetNDofs(); ++node ) {
    MdArray<int, spatial_dim> true_dofs{};
    for ( int d = 0; d < spatial_dim; ++d ) {
      true_dofs[d] = dual_fes.GetLocalTDofNumber( dual_fes.DofToVDof( node, d ) );
    }
    if ( true_dofs[0] < 0 ) {
      continue;
    }
    RealT magnitude_squared = 0.0;
    for ( int d = 0; d < spatial_dim; ++d ) {
      const RealT value = dual_normals[true_dofs[d]];
      magnitude_squared += value * value;
    }
    const RealT magnitude = std::sqrt( magnitude_squared );
    if ( magnitude < minimum_normal_magnitude ) {
      for ( int d = 0; d < spatial_dim; ++d ) {
        dual_normals[true_dofs[d]] = 0.0;
      }
      continue;
    }
    const RealT inverse_magnitude = 1.0 / magnitude;
    MdArray<RealT, spatial_dim> unit_normal{};
    for ( int d = 0; d < spatial_dim; ++d ) {
      unit_normal[d] = dual_normals[true_dofs[d]] * inverse_magnitude;
      dual_normals[true_dofs[d]] = unit_normal[d];
    }
    if ( surface_dn_dx ) {
      for ( int r = 0; r < spatial_dim; ++r ) {
        for ( int c = 0; c < spatial_dim; ++c ) {
          const RealT entry = ( ( r == c ? 1.0 : 0.0 ) - unit_normal[r] * unit_normal[c] ) * inverse_magnitude;
          local_norm_jac.Set( true_dofs[r], true_dofs[c], entry );
        }
      }
    }
  }

  std::unique_ptr<mfem::HypreParMatrix> dual_dn_dx;
  if ( surface_dn_dx ) {
    local_norm_jac.Finalize();
    shared::detail::HypreMemoryLocationGuard guard( shared::ParVectorMemory::Host );
    mfem::HypreParMatrix norm_jac_par( dual_fes.GetComm(), dual_fes.GlobalTrueVSize(),
                                       const_cast<HYPRE_BigInt*>( dual_fes.GetTrueDofOffsets() ), &local_norm_jac );
    if ( uses_lor ) {
      const auto raw_dual_dn_dx =
          *lor_to_dual * shared::ParSparseMatView( const_cast<mfem::HypreParMatrix*>( surface_dn_dx ) );
      auto normalized_dual_dn_dx = shared::ParSparseMatView( &norm_jac_par ) * raw_dual_dn_dx;
      dual_dn_dx.reset( normalized_dual_dn_dx.release() );
    } else {
      auto normalized_dual_dn_dx = shared::ParSparseMatView( &norm_jac_par ) *
                                   shared::ParSparseMatView( const_cast<mfem::HypreParMatrix*>( surface_dn_dx ) );
      dual_dn_dx.reset( normalized_dual_dn_dx.release() );
    }
  }
  return { std::unique_ptr<mfem::HypreParVector>( dual_normals.release() ), std::move( dual_dn_dx ) };
}

}  // namespace tribol::detail

#endif  // BUILD_REDECOMP

#endif  // SRC_TRIBOL_ASSEMBLY_EDGEAVERAGEDNODALNORMALASSEMBLER_HPP_
