// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/assembly/MfemMatrixAssembler.hpp"
#ifdef BUILD_REDECOMP
#include <algorithm>

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

}  // namespace tribol
#endif  // BUILD_REDECOMP
