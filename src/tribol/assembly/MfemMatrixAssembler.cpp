// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/assembly/MfemMatrixAssembler.hpp"
#ifdef BUILD_REDECOMP
#include <array>
#include <utility>

namespace tribol {

MfemMatrixAssembler::MfemMatrixAssembler( MfemMatrixSpace rows, MfemMatrixSpace columns,
                                          std::vector<MfemMatrixBlock> blocks, IndexT num_pairs, int allocator_id )
    : transfer_( rows.path, columns.path ), num_pairs_( num_pairs )
{
  const int num_row_values_per_node = rows.redecomp_fes.GetVDim();
  const int num_column_values_per_node = columns.redecomp_fes.GetVDim();
  if ( num_pairs < 0 || blocks.empty() || num_row_values_per_node <= 0 || num_column_values_per_node <= 0 ) {
    throw std::invalid_argument(
        "Matrix assembly requires blocks, positive numbers of values per node, and a nonnegative pair count." );
  }
  layouts_.reserve( blocks.size() );
  block_storage_.reserve( blocks.size() );
  contribution_metadata_.reserve( blocks.size() );
  for ( const auto& block : blocks ) {
    if ( block.row_geometry < 0 || block.row_geometry > 1 || block.column_geometry < 0 || block.column_geometry > 1 ||
         block.num_row_nodes <= 0 || block.num_column_nodes <= 0 || block.row_offset < 0 || block.column_offset < 0 ) {
      throw std::invalid_argument( "Matrix block geometry selectors, node counts, and offsets are invalid." );
    }
    const IndexT size_rows = detail::checkedMatrixSize( block.num_row_nodes, num_row_values_per_node );
    const IndexT size_columns = detail::checkedMatrixSize( block.num_column_nodes, num_column_values_per_node );
    if ( block.row_offset > std::numeric_limits<IndexT>::max() - size_rows ||
         block.column_offset > std::numeric_limits<IndexT>::max() - size_columns ) {
      throw std::length_error( "Matrix input extents exceed the supported index range." );
    }
    const IndexT block_size = detail::checkedMatrixSize( size_rows, size_columns );
    detail::checkedMatrixSize( num_pairs, block_size );
    layouts_.push_back( { block.row_geometry, block.column_geometry, block.num_row_nodes, block.num_column_nodes,
                          block.row_offset, block.column_offset, num_row_values_per_node,
                          num_column_values_per_node } );
    block_storage_.emplace_back( num_pairs, block_size, allocator_id );
    contribution_metadata_.emplace_back( *rows.path.surface_fes, *columns.path.surface_fes, rows.redecomp_fes,
                                         columns.redecomp_fes, block.row_element_map, block.column_element_map );
  }
}

shared::ParSparseMat MfemMatrixAssembler::assemble() const
{
  auto contributions = contribution_metadata_;
  for ( std::size_t block = 0; block < block_storage_.size(); ++block ) {
    contributions[block].row_elem_ids = Array1D<IndexT, MemorySpace::Host>( block_storage_[block].rowElementIds() );
    contributions[block].col_elem_ids = Array1D<IndexT, MemorySpace::Host>( block_storage_[block].columnElementIds() );
    contributions[block].value_offsets = Array1D<IndexT, MemorySpace::Host>( block_storage_[block].valueOffsets() );
    contributions[block].jacobian_data = Array1D<RealT, MemorySpace::Host>( block_storage_[block].values() );
  }
  return transfer_.Assemble( contributions );
}

namespace detail {
MfemMatrixAssembler makeCoordinateMatrixAssembler( const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data,
                                                   MeshData::Viewer mesh1, MeshData::Viewer mesh2, IndexT num_pairs,
                                                   int allocator_id )
{
  const int dim = mesh1.spatialDim();
  const MdArray<IndexT, 2> nodes{ mesh1.numberOfNodesPerElement(), mesh2.numberOfNodesPerElement() };
  const auto& coordinate_space = *mesh_data.GetRedecompResponse().FESpace();
  if ( dim <= 0 || mesh2.spatialDim() != dim || coordinate_space.GetVDim() != dim || nodes[0] <= 0 || nodes[1] <= 0 ) {
    throw std::invalid_argument( "Coordinate matrix assembly requires matching dimensions and positive node counts." );
  }
  const MdArray<IndexT, 2> offsets{ 0, checkedMatrixSize( nodes[0], dim ) };
  const std::array<const Array1D<int>*, 2> maps{ &mesh_data.GetElemMap1(), &mesh_data.GetElemMap2() };
  std::vector<MfemMatrixBlock> blocks;
  blocks.reserve( 4 );
  for ( IndexT row_geometry = 0; row_geometry < 2; ++row_geometry ) {
    for ( IndexT column_geometry = 0; column_geometry < 2; ++column_geometry ) {
      blocks.push_back( { row_geometry, column_geometry, nodes[row_geometry], nodes[column_geometry],
                          offsets[row_geometry], offsets[column_geometry], *maps[row_geometry],
                          *maps[column_geometry] } );
    }
  }
  const MfemMatrixSpace space{ jacobian_data.ParentPath(), coordinate_space };
  return { space, space, std::move( blocks ), num_pairs, allocator_id };
}
MfemMatrixAssembler makeNodalCoordinateMatrixAssembler( const MfemMeshData& mesh_data,
                                                        const MfemSubmeshData& submesh_data,
                                                        const MfemJacobianData& jacobian_data, MeshData::Viewer mesh1,
                                                        MeshData::Viewer mesh2, IndexT num_pairs, int allocator_id )
{
  const auto& coordinates = *mesh_data.GetRedecompResponse().FESpace();
  const auto& field = submesh_data.GetRedecompFESpace();
  if ( field.GetVDim() != 1 || submesh_data.GetSubmeshFESpace().GetVDim() != 1 || mesh1.spatialDim() <= 0 ||
       mesh1.spatialDim() != coordinates.GetVDim() || mesh2.spatialDim() != coordinates.GetVDim() ) {
    throw std::invalid_argument( "Nodal coordinate matrices require scalar rows and matching coordinate dimensions." );
  }
  const IndexT nodes1 = mesh1.numberOfNodesPerElement();
  const auto& map1 = mesh_data.GetElemMap1();
  const auto& map2 = mesh_data.GetElemMap2();
  return { { jacobian_data.SubmeshPath(), field },
           { jacobian_data.ParentPath(), coordinates },
           { { 0, 0, nodes1, nodes1, 0, 0, map1, map1 },
             { 0, 1, nodes1, mesh2.numberOfNodesPerElement(), 0, checkedMatrixSize( nodes1, coordinates.GetVDim() ),
               map1, map2 } },
           num_pairs,
           allocator_id };
}
}  // namespace detail

}  // namespace tribol
#endif  // BUILD_REDECOMP
