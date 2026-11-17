// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Preallocated dense element-pair matrix block storage and device-copyable views.
 */
#ifndef SRC_TRIBOL_ASSEMBLY_ELEMENTBLOCKSTORAGE_HPP_
#define SRC_TRIBOL_ASSEMBLY_ELEMENTBLOCKSTORAGE_HPP_

#include <limits>
#include <stdexcept>

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/BasicTypes.hpp"

namespace tribol {

/**
 * @brief Device-copyable view of one fixed-size element block per pair.
 * @tparam SizeRows Dense block row count.
 * @tparam SizeColumns Dense block column count.
 */
template <IndexT SizeRows, IndexT SizeColumns>
struct ElementBlockView {
  IndexT* row_element_ids{ nullptr };     ///< Row element ID for each block.
  IndexT* column_element_ids{ nullptr };  ///< Column element ID for each block.
  IndexT* value_offsets{ nullptr };       ///< Offset of each block in values.
  RealT* values{ nullptr };               ///< Column-major dense block values.
  IndexT num_pairs{ 0 };                  ///< Number of writable blocks.

  /** @brief Set element IDs and offset. @param pair Pair slot. @param row Row element. @param column Column element. */
  TRIBOL_HOST_DEVICE void setElements( IndexT pair, IndexT row, IndexT column ) const
  {
    row_element_ids[pair] = row;
    column_element_ids[pair] = column;
    value_offsets[pair] = pair * SizeRows * SizeColumns;
  }

  /** @brief Store a column-major block entry. @param pair Pair slot. @param row Row. @param column Column.
   * @param value Entry value. */
  TRIBOL_HOST_DEVICE void setValue( IndexT pair, IndexT row, IndexT column, RealT value ) const
  {
    values[pair * SizeRows * SizeColumns + row + column * SizeRows] = value;
  }
};

namespace detail {
/** @brief Check the number of values required by fixed-size element blocks.
 * @param num_pairs Number of blocks.
 * @param values_per_block Scalar values in each block.
 * @return Total scalar values representable by IndexT.
 * @throws std::invalid_argument A size is negative.
 * @throws std::length_error The total exceeds the supported index range. */
inline IndexT checkedMatrixSize( IndexT num_pairs, IndexT values_per_block )
{
  if ( num_pairs < 0 || values_per_block < 0 ) {
    throw std::invalid_argument( "Element block storage sizes must be nonnegative." );
  }
  if ( values_per_block != 0 && num_pairs > std::numeric_limits<IndexT>::max() / values_per_block ) {
    throw std::length_error( "Element block storage exceeds the supported index range." );
  }
  return num_pairs * values_per_block;
}

}  // namespace detail

/** @brief Own runtime-sized dense element blocks in execution-space memory. */
class ElementBlockStorage {
 public:
  /** @brief Allocate all blocks before packing.
   * @param num_pairs Pair count.
   * @param values_per_block Scalar values in each dense block.
   * @param allocator_id Execution allocator. */
  ElementBlockStorage( IndexT num_pairs, IndexT values_per_block, int allocator_id )
      : num_pairs_( num_pairs ),
        values_per_block_( values_per_block ),
        row_element_ids_( detail::checkedMatrixSize( num_pairs, 1 ), detail::checkedMatrixSize( num_pairs, 1 ),
                          allocator_id ),
        column_element_ids_( detail::checkedMatrixSize( num_pairs, 1 ), detail::checkedMatrixSize( num_pairs, 1 ),
                             allocator_id ),
        value_offsets_( detail::checkedMatrixSize( num_pairs, 1 ), detail::checkedMatrixSize( num_pairs, 1 ),
                        allocator_id ),
        values_( detail::checkedMatrixSize( num_pairs, values_per_block ),
                 detail::checkedMatrixSize( num_pairs, values_per_block ), allocator_id )
  {
  }

  /** @brief Return a device-copyable writable view.
   * @tparam SizeRows Dense block row count.
   * @tparam SizeColumns Dense block column count.
   * @return View retaining no ownership.
   * @throws std::invalid_argument The requested dimensions do not match the allocated block size. */
  template <IndexT SizeRows, IndexT SizeColumns>
  ElementBlockView<SizeRows, SizeColumns> view()
  {
    static_assert( SizeRows > 0 && SizeColumns > 0 );
    if ( SizeRows * SizeColumns != values_per_block_ ) {
      throw std::invalid_argument( "Element block view dimensions do not match its storage." );
    }
    return { row_element_ids_.data(), column_element_ids_.data(), value_offsets_.data(), values_.data(), num_pairs_ };
  }

  /** @brief Return row element IDs. @return Owning array. */
  const Array1D<IndexT>& rowElementIds() const { return row_element_ids_; }
  /** @brief Return column element IDs. @return Owning array. */
  const Array1D<IndexT>& columnElementIds() const { return column_element_ids_; }
  /** @brief Return block value offsets. @return Owning array. */
  const Array1D<IndexT>& valueOffsets() const { return value_offsets_; }
  /** @brief Return block values. @return Owning array. */
  const Array1D<RealT>& values() const { return values_; }

 private:
  IndexT num_pairs_;                    ///< Number of blocks in each metadata array.
  IndexT values_per_block_;             ///< Scalar values in each dense block.
  Array1D<IndexT> row_element_ids_;     ///< Row element IDs.
  Array1D<IndexT> column_element_ids_;  ///< Column element IDs.
  Array1D<IndexT> value_offsets_;       ///< Block value offsets.
  Array1D<RealT> values_;               ///< Column-major dense blocks.
};

}  // namespace tribol

#endif  // SRC_TRIBOL_ASSEMBLY_ELEMENTBLOCKSTORAGE_HPP_
