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
 * @brief Write rectangular blocks for one batch from a host or device kernel.
 * @tparam RowDofs Rows per block in MFEM element order.
 * @tparam ColumnDofs Columns per block in MFEM element order.
 * @tparam NumBlocks Configured blocks per pair.
 */
template <IndexT RowDofs, IndexT ColumnDofs, IndexT NumBlocks>
struct MfemMatrixBatchView {
  MdArray<ElementBlockView<RowDofs, ColumnDofs>, NumBlocks>
      blocks;                                                 ///< Writable views into the matrix batch's buffers.
  MdArray<detail::MfemMatrixBlockLayout, NumBlocks> layouts;  ///< Independent row and column indexing.

  /**
   * @brief Pack one pair's blocks, converting node/component inputs to MFEM component/node element order.
   * @tparam MatrixReader Callable returning an entry from `(pair, indices..., row, column)`.
   * @tparam Indices Optional field indices preceding the row and column arguments.
   * @param pair_index Writable batch slot.
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

class MfemMatrixAssembler;

/**
 * @brief Own scratch blocks until packing and transfer to an assembler finish.
 * @tparam RowDofs Rows per block.
 * @tparam ColumnDofs Columns per block.
 * @tparam NumBlocks Configured blocks per pair.
 */
template <IndexT RowDofs, IndexT ColumnDofs, IndexT NumBlocks>
class MfemMatrixBatch {
 public:
  /** @brief Return a writable kernel view. @return View into this batch's buffers; the batch must remain alive while
   * the view is used. */
  MfemMatrixBatchView<RowDofs, ColumnDofs, NumBlocks> view()
  {
    MfemMatrixBatchView<RowDofs, ColumnDofs, NumBlocks> result{ {}, layouts_ };
    for ( IndexT block = 0; block < NumBlocks; ++block ) {
      result.blocks[block] = blocks_[block].view();
    }
    return result;
  }

 private:
  friend class MfemMatrixAssembler;
  /** @brief Allocate a validated batch. @param owner Assembler that accepts these blocks. @param num_pairs Pairs.
   * @param allocator_id Execution allocator. @param layouts Validated block layouts. */
  MfemMatrixBatch( const MfemMatrixAssembler* owner, IndexT num_pairs, int allocator_id,
                   MdArray<detail::MfemMatrixBlockLayout, NumBlocks> layouts )
      : owner_( owner ), layouts_( layouts )
  {
    blocks_.reserve( NumBlocks );
    for ( IndexT block = 0; block < NumBlocks; ++block ) {
      blocks_.emplace_back( num_pairs, allocator_id );
    }
  }
  const MfemMatrixAssembler* owner_;  ///< Assembler that created the batch; it must remain in place until addBatch().
  MdArray<detail::MfemMatrixBlockLayout, NumBlocks> layouts_;    ///< Indexing copied into the kernel view.
  std::vector<ElementBlockBuffer<RowDofs, ColumnDofs>> blocks_;  ///< Scratch proportional to this batch's pairs.
};

/**
 * @brief Accumulate rectangular pair matrices with independent row and column nodal spaces and transfers.
 *
 * Supports the displacement and scalar nodal spaces used by contact assembly in either orientation. Input matrices
 * use node/component DOF order on both axes. Packed MFEM element blocks use component/node DOF order and column-major
 * storage. No symmetry is assumed. Geometry selectors and element maps are independent for each block's two axes.
 * FE spaces, transfer paths, and element maps must remain valid for the assembler's lifetime. Recreate after
 * decomposition or mesh re-registration.
 * This class assembles matrices only. Callers can use a batch view alongside vector accumulation in a shared kernel.
 */
class MfemMatrixAssembler {
 public:
  /**
   * @brief Configure blocks and reserve storage for the expected contributing pairs.
   * @param rows Row space and initialized transfer path; both must outlive the assembler.
   * @param columns Column space and initialized transfer path; both must outlive the assembler.
   * @param blocks Nonempty list of rectangular blocks; no storage is allocated for unlisted geometry combinations.
   * @param pair_capacity Expected pair contributions, independent of all possible mesh pairs.
   * @throws std::invalid_argument Capacity, node counts, geometry selectors, or offsets are invalid.
   * @throws std::length_error Required storage exceeds the supported index range.
   */
  MfemMatrixAssembler( MfemMatrixSpace rows, MfemMatrixSpace columns, std::vector<MfemMatrixBlock> blocks,
                       IndexT pair_capacity );

  /**
   * @brief Append one host matrix, without retaining its input storage.
   * @param pair Valid element IDs in the configured geometry domains.
   * @param matrix Matrix in node/component row and column order, including all configured offsets.
   * @throws std::invalid_argument Dimensions do not match the configured input extents; nothing is appended.
   */
  void addPair( InterfacePair pair, ArrayViewT<const RealT, 2> matrix );

  /**
   * @brief Read and append one host matrix through a caller-supplied entry reader.
   * @tparam MatrixReader Callable `(row, column) -> RealT` using the configured input extents and ordering.
   * @param pair Valid geometry element IDs.
   * @param read_matrix Host matrix reader; not retained.
   */
  template <typename MatrixReader>
  void addPairFromReader( InterfacePair pair, MatrixReader read_matrix )
  {
    const MdArray<IndexT, 2> elements{ pair.element_id1_, pair.element_id2_ };
    for ( IndexT block = 0; block < static_cast<IndexT>( layouts_.size() ); ++block ) {
      const auto layout = layouts_[block];
      const IndexT rows = layout.row_nodes * layout.row_components;
      const IndexT columns = layout.column_nodes * layout.column_components;
      for ( IndexT column = 0; column < columns; ++column ) {
        for ( IndexT row = 0; row < rows; ++row ) {
          block_[row + column * rows] = read_matrix( layout.rowDof( row ), layout.columnDof( column ) );
        }
      }
      contributions_[block].append( elements[layout.row_geometry], elements[layout.column_geometry], block_.data(),
                                    rows * columns );
    }
  }

  /**
   * @brief Allocate fixed-size block scratch for an execution-space kernel.
   * @tparam RowDofs Rows per block. @tparam ColumnDofs Columns per block. @tparam NumBlocks Blocks per pair.
   * @param num_pairs Supplied pair count. @param allocator_id Execution-accessible allocator.
   * @return Owning batch whose view can pack matrices alongside other per-pair operations.
   * Keep the assembler alive and in place until addBatch() finishes. Scratch must remain alive while kernels use its
   * view.
   * @throws std::invalid_argument Template dimensions or count do not match the configuration, or num_pairs is
   * negative.
   * @throws std::length_error Scratch size exceeds the supported index range; no batch is returned.
   */
  template <IndexT RowDofs, IndexT ColumnDofs, IndexT NumBlocks>
  MfemMatrixBatch<RowDofs, ColumnDofs, NumBlocks> makeBatch( IndexT num_pairs, int allocator_id ) const
  {
    static_assert( RowDofs > 0 && ColumnDofs > 0 && NumBlocks > 0 );
    if ( layouts_.size() != static_cast<std::size_t>( NumBlocks ) ) {
      throw std::invalid_argument( "Matrix batch block count does not match its configuration." );
    }
    MdArray<detail::MfemMatrixBlockLayout, NumBlocks> layouts{};
    for ( IndexT block = 0; block < NumBlocks; ++block ) {
      layouts[block] = layouts_[block];
      if ( layouts[block].row_nodes * layouts[block].row_components != RowDofs ||
           layouts[block].column_nodes * layouts[block].column_components != ColumnDofs ) {
        throw std::invalid_argument( "Matrix batch dimensions do not match its configured blocks." );
      }
    }
    detail::checkedMatrixSize( num_pairs, detail::checkedMatrixSize( RowDofs, ColumnDofs ) );
    return { this, num_pairs, allocator_id, layouts };
  }

  /**
   * @brief Copy completed execution-space blocks to host accumulation storage.
   * @tparam RowDofs Rows per block. @tparam ColumnDofs Columns per block. @tparam NumBlocks Blocks per pair.
   * @param batch Batch created by this assembler; its buffers must remain valid until the call completes.
   * @throws std::invalid_argument Batch belongs to another assembler; nothing is appended.
   * @throws std::length_error Accumulated blocks exceed the supported index range.
   */
  template <IndexT RowDofs, IndexT ColumnDofs, IndexT NumBlocks>
  void addBatch( const MfemMatrixBatch<RowDofs, ColumnDofs, NumBlocks>& batch )
  {
    if ( batch.owner_ != this ) {
      throw std::invalid_argument( "Matrix batch must be appended to the assembler that created it." );
    }
    for ( IndexT block = 0; block < NumBlocks; ++block ) {
      appendElementBlocks( contributions_[block], batch.blocks_[block] );
    }
  }

  /**
   * @brief Pack and append matrices in a dedicated kernel; earlier contributions accumulate.
   * @tparam RowDofs Rows per block. @tparam ColumnDofs Columns per block. @tparam NumBlocks Blocks per pair.
   * @tparam MatrixReader Device-callable `(pair, row, column) -> RealT` reader.
   * @tparam Exec Concrete execution mode.
   * @param pairs Supplied execution-accessible pairs.
   * @param read_matrix Matrix reader in node/component order.
   * @throws std::invalid_argument Batch dimensions do not match the configuration; nothing is appended.
   * @throws std::length_error Required storage exceeds the supported index range.
   */
  template <ExecutionMode Exec, IndexT RowDofs, IndexT ColumnDofs, IndexT NumBlocks, typename MatrixReader>
  void addPairs( const ArrayT<InterfacePair>& pairs, MatrixReader read_matrix )
  {
    auto batch = makeBatch<RowDofs, ColumnDofs, NumBlocks>( pairs.size(), pairs.getAllocatorID() );
    const auto batch_view = batch.view();
    const auto pairs_view = pairs.view();
    forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
      batch_view.packPair( pair, pairs_view[pair], read_matrix );
    } );
    addBatch( batch );
  }

  /** @brief Transfer accumulated blocks to solver true DOFs.
   * @return Owned rectangular matrix; no contributions produce an allocated zero matrix of the configured spaces.
   * Results remain valid after this assembler is destroyed while the FE spaces and communicator remain alive. */
  shared::ParSparseMat assemble() const;

 private:
  /**
   * @brief Copy packed element blocks to the host and append them after any earlier contributions.
   * @tparam RowDofs Dense block row count.
   * @tparam ColumnDofs Dense block column count.
   * @param destination Accumulated host blocks; existing entries and value offsets are preserved.
   * @param blocks Execution-space blocks to copy; not retained.
   * @throws std::length_error Accumulated entries or values exceed the supported index range; nothing is appended.
   */
  template <IndexT RowDofs, IndexT ColumnDofs>
  static void appendElementBlocks( PackedPairJacobianContribs& destination,
                                   const ElementBlockBuffer<RowDofs, ColumnDofs>& blocks )
  {
    const auto max_size = std::numeric_limits<IndexT>::max();
    if ( blocks.rowElementIds().size() > max_size - destination.row_elem_ids.size() ||
         blocks.values().size() > max_size - destination.jacobian_data.size() ) {
      throw std::length_error( "Accumulated element blocks exceed the supported index range." );
    }
    if ( blocks.rowElementIds().size() == 0 ) {
      return;
    }
    const Array1D<IndexT, MemorySpace::Host> rows( blocks.rowElementIds() );
    const Array1D<IndexT, MemorySpace::Host> columns( blocks.columnElementIds() );
    Array1D<IndexT, MemorySpace::Host> offsets( blocks.valueOffsets() );
    const Array1D<RealT, MemorySpace::Host> values( blocks.values() );
    const IndexT value_offset = destination.jacobian_data.size();
    for ( IndexT& offset : offsets ) {
      offset += value_offset;
    }
    destination.row_elem_ids.append( rows.view() );
    destination.col_elem_ids.append( columns.view() );
    destination.value_offsets.append( offsets.view() );
    destination.jacobian_data.append( values.view() );
  }

  MfemJacobianTransfer transfer_;                       ///< References to independent row and column transfer paths.
  std::vector<detail::MfemMatrixBlockLayout> layouts_;  ///< Input block indexing, independent of pair count.
  std::vector<PackedPairJacobianContribs>
      contributions_;                        ///< Accumulated blocks referencing FE spaces and element maps.
  IndexT num_rows_ = 0;                      ///< Input rows including configured offsets.
  IndexT num_columns_ = 0;                   ///< Input columns including configured offsets.
  Array1D<RealT, MemorySpace::Host> block_;  ///< Reused host scratch for the largest configured block.
};

namespace detail {
/**
 * @brief Configure all four coordinate blocks for a pair of displacement elements.
 * @param mesh_data Displacement spaces and element maps for the current decomposition; must outlive the assembler.
 * @param jacobian_data Initialized parent-coordinate transfer path; must outlive the returned assembler.
 * @param mesh1 First geometry's nodal layout.
 * @param mesh2 Second geometry's nodal layout.
 * @param pair_capacity Expected contributing pairs, independent of all possible mesh pairs.
 * @return Matrix assembler referencing the supplied displacement FE spaces, element maps, and transfer path.
 * @throws std::invalid_argument Geometry dimensions or node counts do not match the displacement layout.
 * @throws std::length_error Required storage exceeds the supported index range.
 */
MfemMatrixAssembler makeCoordinateMatrixAssembler( const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data,
                                                   MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                   IndexT pair_capacity );
/** @brief Configure a scalar nodal field's two coordinate-Jacobian blocks.
 * @param mesh_data Displacement spaces and element maps; must outlive the returned assembler.
 * @param submesh_data Scalar field spaces for the current decomposition; must outlive the returned assembler.
 * @param jacobian_data Initialized scalar and parent-coordinate transfer paths.
 * @param mesh1 Geometry carrying scalar rows. @param mesh2 Second geometry contributing coordinate columns.
 * @param pair_capacity Expected contributing pairs.
 * @return Matrix assembler with scalar submesh rows and parent displacement columns.
 * @throws std::invalid_argument Field spaces are not scalar or geometry dimensions do not match the coordinate space.
 * @throws std::length_error Required storage exceeds the supported index range. */
MfemMatrixAssembler makeNodalCoordinateMatrixAssembler( const MfemMeshData& mesh_data,
                                                        const MfemSubmeshData& submesh_data,
                                                        const MfemJacobianData& jacobian_data, MeshData::Viewer mesh1,
                                                        MeshData::Viewer mesh2, IndexT pair_capacity );
}  // namespace detail

}  // namespace tribol
#endif  // BUILD_REDECOMP
#endif  // SRC_TRIBOL_ASSEMBLY_MFEMMATRIXASSEMBLER_HPP_
