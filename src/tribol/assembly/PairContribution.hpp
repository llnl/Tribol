// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Preallocated pair-local values and coordinate derivatives for bulk contact assembly.
 *
 * Activation flags retain pair-function values or derivatives after successful quadrature. Nodal integration may
 * remain active for opening gaps, before enforcement selects the contact active set.
 */
#ifndef SRC_TRIBOL_ASSEMBLY_PAIRCONTRIBUTION_HPP_
#define SRC_TRIBOL_ASSEMBLY_PAIRCONTRIBUTION_HPP_

#include <limits>
#include <stdexcept>

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/** @brief Device-copyable pair values and quadrature state without derivatives. @tparam SizeValues Values per pair. */
template <IndexT SizeValues>
struct PairValueView {
  RealT* values{ nullptr };                 ///< Values in `[value][pair]` order.
  QuadratureResult* quadrature{ nullptr };  ///< Pair quadrature status and point count.
  IndexT* has_active_points{ nullptr };     ///< Nonzero when a successfully evaluated pair has active points.
  IndexT num_pairs{ 0 };                    ///< Number of writable pair slots.

  /** @brief Store one value. @param pair Pair slot. @param value Function output index. @param result Numerical value.
   */
  TRIBOL_HOST_DEVICE void setValue( IndexT pair, IndexT value, RealT result ) const
  {
    values[value * num_pairs + pair] = result;
  }
  /** @brief Read one value. @param pair Pair slot. @param value Function output index. @return Numerical value. */
  TRIBOL_HOST_DEVICE RealT value( IndexT pair, IndexT value ) const { return values[value * num_pairs + pair]; }
};

/** @brief Own pair values and quadrature state without derivative storage. @tparam SizeValues Values per pair. */
template <IndexT SizeValues>
class PairValueBuffer {
 public:
  /** @brief Allocate the supplied pair slots. @param num_pairs Pair count. @param allocator_id Execution allocator. */
  PairValueBuffer( IndexT num_pairs, int allocator_id )
      : num_pairs_( num_pairs ),
        values_( SizeValues * num_pairs, SizeValues * num_pairs, allocator_id ),
        quadrature_( num_pairs, num_pairs, allocator_id ),
        has_active_points_( num_pairs, num_pairs, allocator_id )
  {
  }
  /** @brief Return a writable view. @return View into this buffer's arrays; the buffer must remain alive while the view
   * is used. */
  PairValueView<SizeValues> view()
  {
    return { values_.data(), quadrature_.data(), has_active_points_.data(), num_pairs_ };
  }
  /** @brief Return pair slots. @return Number of pairs allocated. */
  IndexT numberOfPairs() const { return num_pairs_; }
  /** @brief Return the owning values. @return Values in `[value][pair]` order. */
  const Array1D<RealT>& values() const { return values_; }
  /** @brief Return quadrature results. @return One result per pair. */
  const Array1D<QuadratureResult>& quadratureResults() const { return quadrature_; }
  /** @brief Return pair-function activation flags. @return One activation flag per pair after successful quadrature. */
  const Array1D<IndexT>& activePairs() const { return has_active_points_; }

 private:
  IndexT num_pairs_;                      ///< Allocated pair slots.
  Array1D<RealT> values_;                 ///< Numerical values in structure-of-arrays order.
  Array1D<QuadratureResult> quadrature_;  ///< Quadrature status for each pair.
  Array1D<IndexT> has_active_points_;     ///< Activation flag for each pair after successful quadrature.
};

/**
 * @brief Device-copyable view of pair values and their first coordinate derivatives.
 *
 * Pair index is the contiguous dimension. When adjacent GPU threads write the same function output or Jacobian entry,
 * their writes are coalesced.
 *
 * @tparam SizeValues Number of scalar values produced by one pair.
 * @tparam SizeCoords Number of active pair coordinates.
 */
template <IndexT SizeValues, IndexT SizeCoords>
struct PairValueJacobianView {
  RealT* values{ nullptr };                 ///< Values in `[value][pair]` order.
  RealT* jacobians{ nullptr };              ///< Derivatives in `[value][coordinate][pair]` order.
  QuadratureResult* quadrature{ nullptr };  ///< Pair quadrature status and point count.
  IndexT* has_active_points{ nullptr };     ///< Nonzero when a successfully evaluated pair has active points.
  IndexT num_pairs{ 0 };                    ///< Number of writable pair slots.

  /** @brief Store one pair value. @param pair Pair slot. @param value Function output index. @param result Scalar
   * value. */
  TRIBOL_HOST_DEVICE void setValue( IndexT pair, IndexT value, RealT result ) const
  {
    values[value * num_pairs + pair] = result;
  }

  /** @brief Read one pair value. @param pair Pair slot. @param value Function output index. @return Scalar value. */
  TRIBOL_HOST_DEVICE RealT value( IndexT pair, IndexT value ) const { return values[value * num_pairs + pair]; }

  /**
   * @brief Store one coordinate derivative.
   * @param pair Pair slot.
   * @param value Function output index.
   * @param coordinate Coordinate index.
   * @param result Derivative value.
   */
  TRIBOL_HOST_DEVICE void setJacobian( IndexT pair, IndexT value, IndexT coordinate, RealT result ) const
  {
    jacobians[( value * SizeCoords + coordinate ) * num_pairs + pair] = result;
  }

  /**
   * @brief Read one coordinate derivative.
   * @param pair Pair slot.
   * @param value Function output index.
   * @param coordinate Coordinate index.
   * @return Derivative value.
   */
  TRIBOL_HOST_DEVICE RealT jacobian( IndexT pair, IndexT value, IndexT coordinate ) const
  {
    return jacobians[( value * SizeCoords + coordinate ) * num_pairs + pair];
  }
};

/**
 * @brief Own pair values and Jacobians in an execution-space allocation.
 * @tparam SizeValues Number of scalar values produced by one pair.
 * @tparam SizeCoords Number of active pair coordinates.
 */
template <IndexT SizeValues, IndexT SizeCoords>
class PairValueJacobianBuffer {
 public:
  /**
   * @brief Allocate every pair slot before evaluation begins.
   * @param num_pairs Number of pair contributions.
   * @param allocator_id Axom/Umpire allocator accessible from the selected execution mode.
   */
  PairValueJacobianBuffer( IndexT num_pairs, int allocator_id )
      : num_pairs_( num_pairs ),
        values_( SizeValues * num_pairs, SizeValues * num_pairs, allocator_id ),
        jacobians_( SizeValues * SizeCoords * num_pairs, SizeValues * SizeCoords * num_pairs, allocator_id ),
        quadrature_( num_pairs, num_pairs, allocator_id ),
        has_active_points_( num_pairs, num_pairs, allocator_id )
  {
  }

  /** @brief Return a device-copyable writable view. @return View retaining no ownership. */
  PairValueJacobianView<SizeValues, SizeCoords> view()
  {
    return { values_.data(), jacobians_.data(), quadrature_.data(), has_active_points_.data(), num_pairs_ };
  }

  /** @brief Return the number of pair slots. @return Pair count. */
  IndexT numberOfPairs() const { return num_pairs_; }

  /** @brief Return the owning value array. @return Values in structure-of-arrays order. */
  const Array1D<RealT>& values() const { return values_; }

  /** @brief Return the owning Jacobian array. @return Jacobians in structure-of-arrays order. */
  const Array1D<RealT>& jacobians() const { return jacobians_; }

  /** @brief Return quadrature results. @return One result per pair. */
  const Array1D<QuadratureResult>& quadratureResults() const { return quadrature_; }

  /** @brief Return pair-function activation flags. @return One activation flag per pair after successful quadrature. */
  const Array1D<IndexT>& activePairs() const { return has_active_points_; }

 private:
  IndexT num_pairs_;                      ///< Pair slots allocated in every array.
  Array1D<RealT> values_;                 ///< Values in `[value][pair]` order.
  Array1D<RealT> jacobians_;              ///< Derivatives in `[value][coordinate][pair]` order.
  Array1D<QuadratureResult> quadrature_;  ///< Quadrature status for each pair.
  Array1D<IndexT> has_active_points_;     ///< Activation flag for each pair after successful quadrature.
};

/**
 * @brief Device-copyable coordinate Jacobians and Hessians of scalar functions or fixed weighted sums.
 * @tparam SizeCoords Number of active pair coordinates.
 */
template <IndexT SizeCoords>
struct PairJacobianHessianView {
  RealT* jacobians{ nullptr };  ///< Coordinate Jacobians in `[coordinate][pair]` order.
  RealT* hessians{ nullptr };   ///< Coordinate Hessians in `[row][column][pair]` order.
  IndexT num_pairs{ 0 };        ///< Number of writable pair slots.

  /** @brief Store a Jacobian entry. @param pair Pair slot. @param coordinate Coordinate. @param value Entry value. */
  TRIBOL_HOST_DEVICE void setJacobian( IndexT pair, IndexT coordinate, RealT value ) const
  {
    jacobians[coordinate * num_pairs + pair] = value;
  }

  /** @brief Read a Jacobian entry. @param pair Pair slot. @param coordinate Coordinate. @return Entry value. */
  TRIBOL_HOST_DEVICE RealT jacobian( IndexT pair, IndexT coordinate ) const
  {
    return jacobians[coordinate * num_pairs + pair];
  }

  /**
   * @brief Store a Hessian entry.
   * @param pair Pair slot.
   * @param row Hessian row.
   * @param column Hessian column.
   * @param value Entry value.
   */
  TRIBOL_HOST_DEVICE void setHessian( IndexT pair, IndexT row, IndexT column, RealT value ) const
  {
    hessians[( row * SizeCoords + column ) * num_pairs + pair] = value;
  }

  /** @brief Read a Hessian entry. @param pair Pair slot. @param row Row. @param column Column. @return Entry value. */
  TRIBOL_HOST_DEVICE RealT hessian( IndexT pair, IndexT row, IndexT column ) const
  {
    return hessians[( row * SizeCoords + column ) * num_pairs + pair];
  }
};

/** @brief Own pair Jacobians and Hessians in execution-space memory. @tparam SizeCoords Pair coordinate count.
 */
template <IndexT SizeCoords>
class PairJacobianHessianBuffer {
 public:
  /** @brief Allocate all derivative slots. @param num_pairs Pair count. @param allocator_id Execution allocator. */
  PairJacobianHessianBuffer( IndexT num_pairs, int allocator_id )
      : num_pairs_( num_pairs ),
        jacobians_( SizeCoords * num_pairs, SizeCoords * num_pairs, allocator_id ),
        hessians_( SizeCoords * SizeCoords * num_pairs, SizeCoords * SizeCoords * num_pairs, allocator_id )
  {
  }

  /** @brief Return a device-copyable writable view. @return View retaining no ownership. */
  PairJacobianHessianView<SizeCoords> view() { return { jacobians_.data(), hessians_.data(), num_pairs_ }; }

  /** @brief Return the number of pair slots. @return Pair count. */
  IndexT numberOfPairs() const { return num_pairs_; }

  /** @brief Return the owning Jacobian array. @return Jacobians in structure-of-arrays order. */
  const Array1D<RealT>& jacobians() const { return jacobians_; }

  /** @brief Return the owning Hessian array. @return Hessians in structure-of-arrays order. */
  const Array1D<RealT>& hessians() const { return hessians_; }

 private:
  IndexT num_pairs_;          ///< Pair slots allocated in both arrays.
  Array1D<RealT> jacobians_;  ///< Coordinate Jacobians in structure-of-arrays order.
  Array1D<RealT> hessians_;   ///< Coordinate Hessians in structure-of-arrays order.
};

/** @brief Device-copyable pair values, quadrature results, Jacobians, and Hessians. @tparam SizeCoords Coordinates.
 */
template <IndexT SizeCoords>
struct PairValueJacobianHessianView {
  RealT* values{ nullptr };                           ///< One scalar value per pair.
  QuadratureResult* quadrature{ nullptr };            ///< One quadrature result per pair.
  PairJacobianHessianView<SizeCoords> derivatives{};  ///< Coordinate Jacobians and Hessians.
};

/** @brief Own pair values and their coordinate derivatives in execution-space memory. @tparam SizeCoords Coordinates.
 */
template <IndexT SizeCoords>
class PairValueJacobianHessianBuffer {
 public:
  /** @brief Allocate all pair outputs. @param num_pairs Pair count. @param allocator_id Execution allocator. */
  PairValueJacobianHessianBuffer( IndexT num_pairs, int allocator_id )
      : values_( num_pairs, num_pairs, allocator_id ),
        quadrature_( num_pairs, num_pairs, allocator_id ),
        derivatives_( num_pairs, allocator_id )
  {
  }

  /** @brief Return a device-copyable writable view. @return View retaining no ownership. */
  PairValueJacobianHessianView<SizeCoords> view()
  {
    return { values_.data(), quadrature_.data(), derivatives_.view() };
  }

  /** @brief Return quadrature results. @return One result per pair. */
  const Array1D<QuadratureResult>& quadratureResults() const { return quadrature_; }

 private:
  Array1D<RealT> values_;                              ///< Scalar values, zero for empty or inactive pairs.
  Array1D<QuadratureResult> quadrature_;               ///< Quadrature status for each pair.
  PairJacobianHessianBuffer<SizeCoords> derivatives_;  ///< Coordinate Jacobians and Hessians.
};

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

/** @brief Check the number of values required by fixed-size element blocks.
 * @param num_pairs Number of blocks.
 * @param values_per_block Scalar values in each block.
 * @return Total scalar values representable by IndexT.
 * @throws std::invalid_argument A size is negative.
 * @throws std::length_error The total exceeds the supported index range. */
inline IndexT checkedElementBlockValues( IndexT num_pairs, IndexT values_per_block )
{
  if ( num_pairs < 0 || values_per_block < 0 ) {
    throw std::invalid_argument( "Element block storage sizes must be nonnegative." );
  }
  if ( values_per_block != 0 && num_pairs > std::numeric_limits<IndexT>::max() / values_per_block ) {
    throw std::length_error( "Element block storage exceeds the supported index range." );
  }
  return num_pairs * values_per_block;
}

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
        row_element_ids_( checkedElementBlockValues( num_pairs, 1 ), checkedElementBlockValues( num_pairs, 1 ),
                          allocator_id ),
        column_element_ids_( checkedElementBlockValues( num_pairs, 1 ), checkedElementBlockValues( num_pairs, 1 ),
                             allocator_id ),
        value_offsets_( checkedElementBlockValues( num_pairs, 1 ), checkedElementBlockValues( num_pairs, 1 ),
                        allocator_id ),
        values_( checkedElementBlockValues( num_pairs, values_per_block ),
                 checkedElementBlockValues( num_pairs, values_per_block ), allocator_id )
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

/**
 * @brief Own one fixed-size dense element block per pair in execution-space memory.
 * @tparam SizeRows Dense block row count.
 * @tparam SizeColumns Dense block column count.
 */
template <IndexT SizeRows, IndexT SizeColumns>
class ElementBlockBuffer : public ElementBlockStorage {
 public:
  /** @brief Allocate all blocks before packing. @param num_pairs Pair count. @param allocator_id Execution allocator.
   */
  ElementBlockBuffer( IndexT num_pairs, int allocator_id )
      : ElementBlockStorage( num_pairs, SizeRows * SizeColumns, allocator_id )
  {
  }

  /** @brief Return a device-copyable writable view. @return View retaining no ownership. */
  ElementBlockView<SizeRows, SizeColumns> view() { return ElementBlockStorage::view<SizeRows, SizeColumns>(); }
};

}  // namespace tribol

#endif  // SRC_TRIBOL_ASSEMBLY_PAIRCONTRIBUTION_HPP_
