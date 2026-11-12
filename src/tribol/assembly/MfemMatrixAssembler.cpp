// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/assembly/MfemMatrixAssembler.hpp"
#ifdef BUILD_REDECOMP
#include <algorithm>
#include <array>
#include <utility>

namespace tribol {

MfemMatrixAssembler::MfemMatrixAssembler( MfemMatrixSpace rows, MfemMatrixSpace columns,
                                          std::vector<MfemMatrixBlock> blocks, IndexT pair_capacity )
    : transfer_( rows.path, columns.path )
{
  const int row_components = rows.redecomp_fes.GetVDim();
  const int column_components = columns.redecomp_fes.GetVDim();
  if ( pair_capacity < 0 || blocks.empty() || row_components <= 0 || column_components <= 0 ) {
    throw std::invalid_argument(
        "Matrix assembly requires blocks, positive component counts, and nonnegative capacity." );
  }
  IndexT largest_block = 0;
  layouts_.reserve( blocks.size() );
  contributions_.reserve( blocks.size() );
  for ( const auto& block : blocks ) {
    if ( block.row_geometry < 0 || block.row_geometry > 1 || block.column_geometry < 0 || block.column_geometry > 1 ||
         block.row_nodes <= 0 || block.column_nodes <= 0 || block.row_offset < 0 || block.column_offset < 0 ) {
      throw std::invalid_argument( "Matrix block geometry selectors, node counts, and offsets are invalid." );
    }
    const IndexT row_dofs = detail::checkedMatrixSize( block.row_nodes, row_components );
    const IndexT column_dofs = detail::checkedMatrixSize( block.column_nodes, column_components );
    if ( block.row_offset > std::numeric_limits<IndexT>::max() - row_dofs ||
         block.column_offset > std::numeric_limits<IndexT>::max() - column_dofs ) {
      throw std::length_error( "Matrix input extents exceed the supported index range." );
    }
    row_count_ = std::max( row_count_, block.row_offset + row_dofs );
    column_count_ = std::max( column_count_, block.column_offset + column_dofs );
    const IndexT block_size = detail::checkedMatrixSize( row_dofs, column_dofs );
    largest_block = std::max( largest_block, block_size );
    layouts_.push_back( { block.row_geometry, block.column_geometry, block.row_nodes, block.column_nodes,
                          block.row_offset, block.column_offset, row_components, column_components } );
    contributions_.emplace_back( *rows.path.surface_fes, *columns.path.surface_fes, rows.redecomp_fes,
                                 columns.redecomp_fes, block.row_element_map, block.column_element_map );
    contributions_.back().reserve( pair_capacity, detail::checkedMatrixSize( pair_capacity, block_size ) );
  }
  block_.resize( largest_block );
}

void MfemMatrixAssembler::addPair( InterfacePair pair, ArrayViewT<const RealT, 2> matrix )
{
  if ( matrix.shape()[0] != row_count_ || matrix.shape()[1] != column_count_ ) {
    throw std::invalid_argument( "Matrix input dimensions do not match the configured block extents." );
  }
  addPairFromReader( pair, [=]( IndexT row, IndexT column ) { return matrix( row, column ); } );
}

shared::ParSparseMat MfemMatrixAssembler::assemble() const { return transfer_.Assemble( contributions_ ); }

namespace detail {
MfemMatrixAssembler makeCoordinateMatrixAssembler( const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data,
                                                   MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                   IndexT pair_capacity )
{
  const int dimension = mesh1.spatialDimension();
  const MdArray<IndexT, 2> nodes{ mesh1.numberOfNodesPerElement(), mesh2.numberOfNodesPerElement() };
  const auto& coordinate_space = *mesh_data.GetRedecompResponse().FESpace();
  if ( dimension <= 0 || mesh2.spatialDimension() != dimension || coordinate_space.GetVDim() != dimension ||
       nodes[0] <= 0 || nodes[1] <= 0 ) {
    throw std::invalid_argument( "Coordinate matrix assembly requires matching dimensions and positive node counts." );
  }
  const MdArray<IndexT, 2> offsets{ 0, checkedMatrixSize( nodes[0], dimension ) };
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
  return { space, space, std::move( blocks ), pair_capacity };
}
MfemMatrixAssembler makeNodalCoordinateMatrixAssembler( const MfemMeshData& mesh_data,
                                                        const MfemSubmeshData& submesh_data,
                                                        const MfemJacobianData& jacobian_data, MeshData::Viewer mesh1,
                                                        MeshData::Viewer mesh2, IndexT pair_capacity )
{
  const auto& coordinates = *mesh_data.GetRedecompResponse().FESpace();
  const auto& field = submesh_data.GetRedecompFESpace();
  if ( field.GetVDim() != 1 || submesh_data.GetSubmeshFESpace().GetVDim() != 1 || mesh1.spatialDimension() <= 0 ||
       mesh1.spatialDimension() != coordinates.GetVDim() || mesh2.spatialDimension() != coordinates.GetVDim() ) {
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
           pair_capacity };
}
}  // namespace detail

}  // namespace tribol
#endif  // BUILD_REDECOMP
