// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Pack rectangular element-pair matrices and transfer independent row and column spaces to true DOFs. */
#ifndef SRC_TRIBOL_ASSEMBLY_MFEMMATRIXASSEMBLER_HPP_
#define SRC_TRIBOL_ASSEMBLY_MFEMMATRIXASSEMBLER_HPP_

#include "tribol/config.hpp"
#ifdef BUILD_REDECOMP
#include <limits>
#include <stdexcept>
#include <vector>
#include "tribol/common/LoopExec.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/assembly/PairContribution.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

/** @brief Reference a nodal FE space on Redecomp and its transfer to solver true DOFs. */
struct MfemMatrixSpace {
  const MfemJacobianPath& path;                  ///< Surface and final FE spaces with their transfer operators.
  const mfem::FiniteElementSpace& redecomp_fes;  ///< Element DOFs before transfer.
};

/**
 * @brief Configure one rectangular block within each pair's input matrix.
 *
 * Geometry indices are 0 for geometry1 and 1 for geometry2. Offsets count input rows and columns, whose DOFs use
 * node/component order. Element maps translate the selected geometry's Tribol element IDs to
 * Redecomp element IDs independently for rows and columns. Maps are referenced directly and must outlive assembly.
 */
struct MfemMatrixBlock {
  IndexT row_geometry;                     ///< Geometry supplying the row element ID.
  IndexT column_geometry;                  ///< Geometry supplying the column element ID.
  IndexT row_nodes;                        ///< Row FE nodes per element.
  IndexT column_nodes;                     ///< Column FE nodes per element.
  IndexT row_offset;                       ///< First input row of this block.
  IndexT column_offset;                    ///< First input column of this block.
  const Array1D<int>& row_element_map;     ///< Row element map referenced during assembly.
  const Array1D<int>& column_element_map;  ///< Column element map referenced during assembly.
};

namespace detail {
/** @brief Device-copyable block indexing, without MFEM objects or element-map storage. */
struct MfemMatrixBlockLayout {
  IndexT row_geometry;     ///< Pair geometry supplying rows.
  IndexT column_geometry;  ///< Pair geometry supplying columns.
  IndexT row_nodes;        ///< Row FE nodes per element.
  IndexT column_nodes;     ///< Column FE nodes per element.
  IndexT row_offset;       ///< Input row offset.
  IndexT column_offset;    ///< Input column offset.
  int row_components;      ///< Row-space components per node.
  int column_components;   ///< Column-space components per node.

  /** @brief Convert an MFEM element row to the input row. @param row Component/node row. @return Node/component row. */
  TRIBOL_HOST_DEVICE IndexT rowDof( IndexT row ) const
  {
    return row_offset + ( row % row_nodes ) * row_components + row / row_nodes;
  }
  /** @brief Convert an MFEM element column to the input column.
   * @param column Component/node column. @return Node/component column. */
  TRIBOL_HOST_DEVICE IndexT columnDof( IndexT column ) const
  {
    return column_offset + ( column % column_nodes ) * column_components + column / column_nodes;
  }
};

/** @brief Check a storage-size product before allocating. @param num_entries Nonnegative entries.
 * @param size Nonnegative values per entry. @return Product representable by IndexT.
 * @throws std::invalid_argument A size is negative. @throws std::length_error The product exceeds IndexT. */
inline IndexT checkedMatrixSize( IndexT num_entries, IndexT size )
{
  if ( num_entries < 0 || size < 0 ) {
    throw std::invalid_argument( "Matrix storage sizes must be nonnegative." );
  }
  if ( size != 0 && num_entries > std::numeric_limits<IndexT>::max() / size ) {
    throw std::length_error( "Matrix storage exceeds the supported index range." );
  }
  return num_entries * size;
}
}  // namespace detail

/**
 * @brief Write rectangular element blocks from a host or device kernel.
 * @tparam RowDofs Rows per block in MFEM element order.
 * @tparam ColumnDofs Columns per block in MFEM element order.
 * @tparam NumBlocks Configured blocks per pair.
 */
template <IndexT RowDofs, IndexT ColumnDofs, IndexT NumBlocks>
struct MfemMatrixAssemblyView {
  MdArray<ElementBlockView<RowDofs, ColumnDofs>, NumBlocks>
      blocks;                                                 ///< Writable views into the assembler's block storage.
  MdArray<detail::MfemMatrixBlockLayout, NumBlocks> layouts;  ///< Independent row and column indexing.

  /**
   * @brief Pack one pair's blocks, converting node/component inputs to MFEM component/node element order.
   * @tparam MatrixReader Callable returning an entry from `(pair, indices..., row, column)`.
   * @tparam Indices Optional field indices preceding the row and column arguments.
   * @param pair_index Writable pair slot.
   * @param pair Pair supplying the two geometry element IDs.
   * @param read_matrix Input matrix reader.
   * @param indices Optional field indices, allowing multiple fields to share one reader.
   */
  template <typename MatrixReader, typename... Indices>
  TRIBOL_HOST_DEVICE void packPair( IndexT pair_index, InterfacePair pair, MatrixReader read_matrix,
                                    Indices... indices ) const
  {
    const MdArray<IndexT, 2> elements{ pair.element_id1_, pair.element_id2_ };
    for ( IndexT block = 0; block < NumBlocks; ++block ) {
      const auto layout = layouts[block];
      const auto block_view = blocks[block];
      block_view.setElements( pair_index, elements[layout.row_geometry], elements[layout.column_geometry] );
      for ( IndexT column = 0; column < ColumnDofs; ++column ) {
        for ( IndexT row = 0; row < RowDofs; ++row ) {
          block_view.setValue(
              pair_index, row, column,
              read_matrix( pair_index, indices..., layout.rowDof( row ), layout.columnDof( column ) ) );
        }
      }
    }
  }
};

/**
 * @brief Assemble rectangular pair matrices with independent row and column nodal spaces and transfers.
 *
 * Supports the displacement and scalar nodal spaces used by contact assembly in either orientation. Input matrices
 * use node/component DOF order on both axes. Packed MFEM element blocks use component/node DOF order and column-major
 * storage. No symmetry is assumed. Geometry selectors and element maps are independent for each block's two axes.
 * FE spaces, transfer paths, and element maps must remain valid for the assembler's lifetime. Each assembler owns
 * storage for one supplied pair list and must be recreated after decomposition, mesh re-registration, or a pair-list
 * change. This class assembles matrices only. Callers can use its view alongside vector accumulation in a shared
 * kernel.
 */
class MfemMatrixAssembler {
 public:
  /**
   * @brief Configure blocks and allocate storage for one pair list.
   * @param rows Row space and initialized transfer path; both must outlive the assembler.
   * @param columns Column space and initialized transfer path; both must outlive the assembler.
   * @param blocks Nonempty list of rectangular blocks; no storage is allocated for unlisted geometry combinations.
   * @param num_pairs Number of pair contributions.
   * @param allocator_id Execution-accessible allocator for element blocks.
   * @throws std::invalid_argument Pair count, node counts, geometry selectors, or offsets are invalid.
   * @throws std::length_error Required storage exceeds the supported index range.
   */
  MfemMatrixAssembler( MfemMatrixSpace rows, MfemMatrixSpace columns, std::vector<MfemMatrixBlock> blocks,
                       IndexT num_pairs, int allocator_id );

  /**
   * @brief Return a writable view of the owned element blocks for an execution-space kernel.
   * @tparam RowDofs Rows per block. @tparam ColumnDofs Columns per block. @tparam NumBlocks Blocks per pair.
   * @return Non-owning view that can pack matrices alongside other per-pair operations.
   * Keep the assembler alive and in place while kernels use the view.
   * @throws std::invalid_argument Template dimensions or count do not match the configuration.
   */
  template <IndexT RowDofs, IndexT ColumnDofs, IndexT NumBlocks>
  MfemMatrixAssemblyView<RowDofs, ColumnDofs, NumBlocks> view()
  {
    static_assert( RowDofs > 0 && ColumnDofs > 0 && NumBlocks > 0 );
    if ( layouts_.size() != static_cast<std::size_t>( NumBlocks ) ) {
      throw std::invalid_argument( "Matrix view block count does not match its configuration." );
    }
    MfemMatrixAssemblyView<RowDofs, ColumnDofs, NumBlocks> result{};
    for ( IndexT block = 0; block < NumBlocks; ++block ) {
      result.layouts[block] = layouts_[block];
      if ( result.layouts[block].row_nodes * result.layouts[block].row_components != RowDofs ||
           result.layouts[block].column_nodes * result.layouts[block].column_components != ColumnDofs ) {
        throw std::invalid_argument( "Matrix view dimensions do not match its configured blocks." );
      }
      result.blocks[block] = block_storage_[block].view<RowDofs, ColumnDofs>();
    }
    return result;
  }

  /**
   * @brief Pack the configured pair matrices in a dedicated kernel.
   * @tparam RowDofs Rows per block. @tparam ColumnDofs Columns per block. @tparam NumBlocks Blocks per pair.
   * @tparam MatrixReader Device-callable `(pair, row, column) -> RealT` reader.
   * @tparam Exec Concrete execution mode.
   * @param pairs Supplied execution-accessible pairs.
   * @param read_matrix Matrix reader in node/component order.
   * @throws std::invalid_argument Pair count or matrix dimensions do not match the configuration.
   */
  template <ExecutionMode Exec, IndexT RowDofs, IndexT ColumnDofs, IndexT NumBlocks, typename MatrixReader>
  void addPairs( const ArrayT<InterfacePair>& pairs, MatrixReader read_matrix )
  {
    if ( pairs.size() != num_pairs_ ) {
      throw std::invalid_argument( "Matrix pair count does not match its allocated storage." );
    }
    const auto matrix_view = view<RowDofs, ColumnDofs, NumBlocks>();
    const auto pairs_view = pairs.view();
    forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
      matrix_view.packPair( pair, pairs_view[pair], read_matrix );
    } );
  }

  /** @brief Transfer the packed blocks to solver true DOFs.
   * @return Owned rectangular matrix; no contributions produce an allocated zero matrix of the configured spaces.
   * Results remain valid after this assembler is destroyed while the FE spaces and communicator remain alive. */
  shared::ParSparseMat assemble() const;

 private:
  MfemJacobianTransfer transfer_;                       ///< References to independent row and column transfer paths.
  IndexT num_pairs_;                                    ///< Pair slots allocated in every block.
  std::vector<detail::MfemMatrixBlockLayout> layouts_;  ///< Input indexing for each configured block.
  std::vector<ElementBlockStorage> block_storage_;      ///< Owned execution-space blocks for one pair list.
  std::vector<PackedPairJacobianContribs>
      contribution_metadata_;  ///< FE spaces and element maps corresponding to each block.
};

namespace detail {
/**
 * @brief Configure all four coordinate blocks for a pair of displacement elements.
 * @param mesh_data Displacement spaces and element maps for the current decomposition; must outlive the assembler.
 * @param jacobian_data Initialized parent-coordinate transfer path; must outlive the returned assembler.
 * @param mesh1 First geometry's nodal layout.
 * @param mesh2 Second geometry's nodal layout.
 * @param num_pairs Number of contributing pairs.
 * @param allocator_id Execution-accessible allocator for element blocks.
 * @return Matrix assembler referencing the supplied displacement FE spaces, element maps, and transfer path.
 * @throws std::invalid_argument Geometry dimensions or node counts do not match the displacement layout.
 * @throws std::length_error Required storage exceeds the supported index range.
 */
MfemMatrixAssembler makeCoordinateMatrixAssembler( const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data,
                                                   MeshData::Viewer mesh1, MeshData::Viewer mesh2, IndexT num_pairs,
                                                   int allocator_id );
/** @brief Configure a scalar nodal field's two coordinate-Jacobian blocks.
 * @param mesh_data Displacement spaces and element maps; must outlive the returned assembler.
 * @param submesh_data Scalar field spaces for the current decomposition; must outlive the returned assembler.
 * @param jacobian_data Initialized scalar and parent-coordinate transfer paths.
 * @param mesh1 Geometry carrying scalar rows. @param mesh2 Second geometry contributing coordinate columns.
 * @param num_pairs Number of contributing pairs.
 * @param allocator_id Execution-accessible allocator for element blocks.
 * @return Matrix assembler with scalar submesh rows and parent displacement columns.
 * @throws std::invalid_argument Field spaces are not scalar or geometry dimensions do not match the coordinate space.
 * @throws std::length_error Required storage exceeds the supported index range. */
MfemMatrixAssembler makeNodalCoordinateMatrixAssembler( const MfemMeshData& mesh_data,
                                                        const MfemSubmeshData& submesh_data,
                                                        const MfemJacobianData& jacobian_data, MeshData::Viewer mesh1,
                                                        MeshData::Viewer mesh2, IndexT num_pairs, int allocator_id );
}  // namespace detail

}  // namespace tribol
#endif  // BUILD_REDECOMP
#endif  // SRC_TRIBOL_ASSEMBLY_MFEMMATRIXASSEMBLER_HPP_
