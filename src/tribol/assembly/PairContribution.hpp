// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Preallocated pair-local values and coordinate derivatives for bulk contact assembly.
 */
#ifndef SRC_TRIBOL_ASSEMBLY_PAIRCONTRIBUTION_HPP_
#define SRC_TRIBOL_ASSEMBLY_PAIRCONTRIBUTION_HPP_

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/** @brief Device-copyable pair values and quadrature state without derivatives. @tparam ValueCt Values per pair. */
template <IndexT ValueCt>
struct PairValueView {
  RealT* values{ nullptr };                 ///< Values in `[value][pair]` order.
  QuadratureResult* quadrature{ nullptr };  ///< Pair quadrature status and point count.
  IndexT* has_active_points{ nullptr };     ///< Nonzero when a pair contains an active contact point.
  IndexT pair_ct{ 0 };                      ///< Number of writable pair slots.

  /** @brief Store one value. @param pair Pair slot. @param value Value component. @param result Numerical value. */
  TRIBOL_HOST_DEVICE void setValue( IndexT pair, IndexT value, RealT result ) const
  {
    values[value * pair_ct + pair] = result;
  }
  /** @brief Read one value. @param pair Pair slot. @param value Value component. @return Numerical value. */
  TRIBOL_HOST_DEVICE RealT value( IndexT pair, IndexT value ) const { return values[value * pair_ct + pair]; }
};

/** @brief Own pair values and quadrature state without derivative storage. @tparam ValueCt Values per pair. */
template <IndexT ValueCt>
class PairValueBuffer {
 public:
  /** @brief Allocate the supplied pair slots. @param pair_ct Pair count. @param allocator_id Execution allocator. */
  PairValueBuffer( IndexT pair_ct, int allocator_id )
      : pair_ct_( pair_ct ),
        values_( ValueCt * pair_ct, ValueCt * pair_ct, allocator_id ),
        quadrature_( pair_ct, pair_ct, allocator_id ),
        has_active_points_( pair_ct, pair_ct, allocator_id )
  {
  }
  /** @brief Return a writable view. @return View into this buffer's arrays; the buffer must remain alive while the view
   * is used. */
  PairValueView<ValueCt> view() { return { values_.data(), quadrature_.data(), has_active_points_.data(), pair_ct_ }; }
  /** @brief Return pair slots. @return Number of pairs allocated. */
  IndexT pairCt() const { return pair_ct_; }
  /** @brief Return the owning values. @return Values in `[value][pair]` order. */
  const Array1D<RealT>& values() const { return values_; }
  /** @brief Return quadrature results. @return One result per pair. */
  const Array1D<QuadratureResult>& quadratureResults() const { return quadrature_; }
  /** @brief Return activation flags. @return One flag per pair. */
  const Array1D<IndexT>& activePoints() const { return has_active_points_; }

 private:
  IndexT pair_ct_;                        ///< Allocated pair slots.
  Array1D<RealT> values_;                 ///< Numerical values in structure-of-arrays order.
  Array1D<QuadratureResult> quadrature_;  ///< Quadrature status for each pair.
  Array1D<IndexT> has_active_points_;     ///< Contact activation for each pair.
};

/**
 * @brief Device-copyable view of pair values and their first coordinate derivatives.
 *
 * Pair index is the contiguous dimension. When adjacent GPU threads write the same value or derivative component,
 * their writes are coalesced.
 *
 * @tparam ValueCt Number of scalar values produced by one pair.
 * @tparam DofCt Number of active pair coordinates.
 */
template <IndexT ValueCt, IndexT DofCt>
struct PairValueJacobianView {
  RealT* values{ nullptr };                 ///< Values in `[value][pair]` order.
  RealT* jacobians{ nullptr };              ///< Derivatives in `[value][coordinate][pair]` order.
  QuadratureResult* quadrature{ nullptr };  ///< Pair quadrature status and point count.
  IndexT* has_active_points{ nullptr };     ///< Nonzero when a pair contains an active contact point.
  IndexT pair_ct{ 0 };                      ///< Number of writable pair slots.

  /** @brief Store one pair value. @param pair Pair slot. @param value Value component. @param result Scalar value. */
  TRIBOL_HOST_DEVICE void setValue( IndexT pair, IndexT value, RealT result ) const
  {
    values[value * pair_ct + pair] = result;
  }

  /** @brief Read one pair value. @param pair Pair slot. @param value Value component. @return Scalar value. */
  TRIBOL_HOST_DEVICE RealT value( IndexT pair, IndexT value ) const { return values[value * pair_ct + pair]; }

  /**
   * @brief Store one coordinate derivative.
   * @param pair Pair slot.
   * @param value Value component.
   * @param coordinate Coordinate component.
   * @param result Derivative value.
   */
  TRIBOL_HOST_DEVICE void setJacobian( IndexT pair, IndexT value, IndexT coordinate, RealT result ) const
  {
    jacobians[( value * DofCt + coordinate ) * pair_ct + pair] = result;
  }

  /**
   * @brief Read one coordinate derivative.
   * @param pair Pair slot.
   * @param value Value component.
   * @param coordinate Coordinate component.
   * @return Derivative value.
   */
  TRIBOL_HOST_DEVICE RealT jacobian( IndexT pair, IndexT value, IndexT coordinate ) const
  {
    return jacobians[( value * DofCt + coordinate ) * pair_ct + pair];
  }
};

/**
 * @brief Own pair values and Jacobians in an execution-space allocation.
 * @tparam ValueCt Number of scalar values produced by one pair.
 * @tparam DofCt Number of active pair coordinates.
 */
template <IndexT ValueCt, IndexT DofCt>
class PairValueJacobianBuffer {
 public:
  /**
   * @brief Allocate every pair slot before evaluation begins.
   * @param pair_ct Number of pair contributions.
   * @param allocator_id Axom/Umpire allocator accessible from the selected execution mode.
   */
  PairValueJacobianBuffer( IndexT pair_ct, int allocator_id )
      : pair_ct_( pair_ct ),
        values_( ValueCt * pair_ct, ValueCt * pair_ct, allocator_id ),
        jacobians_( ValueCt * DofCt * pair_ct, ValueCt * DofCt * pair_ct, allocator_id ),
        quadrature_( pair_ct, pair_ct, allocator_id ),
        has_active_points_( pair_ct, pair_ct, allocator_id )
  {
  }

  /** @brief Return a device-copyable writable view. @return View retaining no ownership. */
  PairValueJacobianView<ValueCt, DofCt> view()
  {
    return { values_.data(), jacobians_.data(), quadrature_.data(), has_active_points_.data(), pair_ct_ };
  }

  /** @brief Return the number of pair slots. @return Pair count. */
  IndexT pairCt() const { return pair_ct_; }

  /** @brief Return the owning value array. @return Values in structure-of-arrays order. */
  const Array1D<RealT>& values() const { return values_; }

  /** @brief Return the owning Jacobian array. @return Jacobians in structure-of-arrays order. */
  const Array1D<RealT>& jacobians() const { return jacobians_; }

  /** @brief Return quadrature results. @return One result per pair. */
  const Array1D<QuadratureResult>& quadratureResults() const { return quadrature_; }

  /** @brief Return activation flags. @return One flag per pair. */
  const Array1D<IndexT>& activePoints() const { return has_active_points_; }

 private:
  IndexT pair_ct_;                        ///< Pair slots allocated in every array.
  Array1D<RealT> values_;                 ///< Values in `[value][pair]` order.
  Array1D<RealT> jacobians_;              ///< Derivatives in `[value][coordinate][pair]` order.
  Array1D<QuadratureResult> quadrature_;  ///< Quadrature status for each pair.
  Array1D<IndexT> has_active_points_;     ///< Contact activation for each pair.
};

/**
 * @brief Device-copyable coordinate Jacobians and Hessians of scalar functions or fixed weighted sums.
 * @tparam DofCt Number of active pair coordinates.
 */
template <IndexT DofCt>
struct PairJacobianHessianView {
  RealT* jacobians{ nullptr };  ///< Coordinate Jacobians in `[coordinate][pair]` order.
  RealT* hessians{ nullptr };   ///< Coordinate Hessians in `[row][column][pair]` order.
  IndexT pair_ct{ 0 };          ///< Number of writable pair slots.

  /** @brief Store a Jacobian entry. @param pair Pair slot. @param coordinate Coordinate. @param value Entry value. */
  TRIBOL_HOST_DEVICE void setJacobian( IndexT pair, IndexT coordinate, RealT value ) const
  {
    jacobians[coordinate * pair_ct + pair] = value;
  }

  /** @brief Read a Jacobian entry. @param pair Pair slot. @param coordinate Coordinate. @return Entry value. */
  TRIBOL_HOST_DEVICE RealT jacobian( IndexT pair, IndexT coordinate ) const
  {
    return jacobians[coordinate * pair_ct + pair];
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
    hessians[( row * DofCt + column ) * pair_ct + pair] = value;
  }

  /** @brief Read a Hessian entry. @param pair Pair slot. @param row Row. @param column Column. @return Entry value. */
  TRIBOL_HOST_DEVICE RealT hessian( IndexT pair, IndexT row, IndexT column ) const
  {
    return hessians[( row * DofCt + column ) * pair_ct + pair];
  }
};

/** @brief Own pair Jacobians and Hessians in execution-space memory. @tparam DofCt Pair coordinate count.
 */
template <IndexT DofCt>
class PairJacobianHessianBuffer {
 public:
  /** @brief Allocate all derivative slots. @param pair_ct Pair count. @param allocator_id Execution allocator. */
  PairJacobianHessianBuffer( IndexT pair_ct, int allocator_id )
      : pair_ct_( pair_ct ),
        jacobians_( DofCt * pair_ct, DofCt * pair_ct, allocator_id ),
        hessians_( DofCt * DofCt * pair_ct, DofCt * DofCt * pair_ct, allocator_id )
  {
  }

  /** @brief Return a device-copyable writable view. @return View retaining no ownership. */
  PairJacobianHessianView<DofCt> view() { return { jacobians_.data(), hessians_.data(), pair_ct_ }; }

  /** @brief Return the number of pair slots. @return Pair count. */
  IndexT pairCt() const { return pair_ct_; }

  /** @brief Return the owning Jacobian array. @return Jacobians in structure-of-arrays order. */
  const Array1D<RealT>& jacobians() const { return jacobians_; }

  /** @brief Return the owning Hessian array. @return Hessians in structure-of-arrays order. */
  const Array1D<RealT>& hessians() const { return hessians_; }

 private:
  IndexT pair_ct_;            ///< Pair slots allocated in both arrays.
  Array1D<RealT> jacobians_;  ///< Coordinate Jacobians in structure-of-arrays order.
  Array1D<RealT> hessians_;   ///< Coordinate Hessians in structure-of-arrays order.
};

/** @brief Device-copyable pair values, quadrature results, Jacobians, and Hessians. @tparam DofCt Coordinates.
 */
template <IndexT DofCt>
struct PairValueJacobianHessianView {
  RealT* values{ nullptr };                      ///< One scalar value per pair.
  QuadratureResult* quadrature{ nullptr };       ///< One quadrature result per pair.
  PairJacobianHessianView<DofCt> derivatives{};  ///< Coordinate Jacobians and Hessians.
};

/** @brief Own pair values and their coordinate derivatives in execution-space memory. @tparam DofCt Coordinates.
 */
template <IndexT DofCt>
class PairValueJacobianHessianBuffer {
 public:
  /** @brief Allocate all pair outputs. @param pair_ct Pair count. @param allocator_id Execution allocator. */
  PairValueJacobianHessianBuffer( IndexT pair_ct, int allocator_id )
      : values_( pair_ct, pair_ct, allocator_id ),
        quadrature_( pair_ct, pair_ct, allocator_id ),
        derivatives_( pair_ct, allocator_id )
  {
  }

  /** @brief Return a device-copyable writable view. @return View retaining no ownership. */
  PairValueJacobianHessianView<DofCt> view() { return { values_.data(), quadrature_.data(), derivatives_.view() }; }

  /** @brief Return quadrature results. @return One result per pair. */
  const Array1D<QuadratureResult>& quadratureResults() const { return quadrature_; }

 private:
  Array1D<RealT> values_;                         ///< Scalar values, zero for empty or inactive pairs.
  Array1D<QuadratureResult> quadrature_;          ///< Quadrature status for each pair.
  PairJacobianHessianBuffer<DofCt> derivatives_;  ///< Coordinate Jacobians and Hessians.
};

/**
 * @brief Device-copyable view of one fixed-size element block per pair.
 * @tparam RowDofs Dense block row count.
 * @tparam ColumnDofs Dense block column count.
 */
template <IndexT RowDofs, IndexT ColumnDofs>
struct ElementBlockView {
  IndexT* row_element_ids{ nullptr };     ///< Row element ID for each block.
  IndexT* column_element_ids{ nullptr };  ///< Column element ID for each block.
  IndexT* value_offsets{ nullptr };       ///< Offset of each block in values.
  RealT* values{ nullptr };               ///< Column-major dense block values.
  IndexT pair_ct{ 0 };                    ///< Number of writable blocks.

  /** @brief Set element IDs and offset. @param pair Pair slot. @param row Row element. @param column Column element. */
  TRIBOL_HOST_DEVICE void setElements( IndexT pair, IndexT row, IndexT column ) const
  {
    row_element_ids[pair] = row;
    column_element_ids[pair] = column;
    value_offsets[pair] = pair * RowDofs * ColumnDofs;
  }

  /** @brief Store a column-major block entry. @param pair Pair slot. @param row Row. @param column Column.
   * @param value Entry value. */
  TRIBOL_HOST_DEVICE void setValue( IndexT pair, IndexT row, IndexT column, RealT value ) const
  {
    values[pair * RowDofs * ColumnDofs + row + column * RowDofs] = value;
  }
};

/**
 * @brief Own one fixed-size dense element block per pair in execution-space memory.
 * @tparam RowDofs Dense block row count.
 * @tparam ColumnDofs Dense block column count.
 */
template <IndexT RowDofs, IndexT ColumnDofs>
class ElementBlockBuffer {
 public:
  /** @brief Allocate all blocks before packing. @param pair_ct Pair count. @param allocator_id Execution allocator.
   */
  ElementBlockBuffer( IndexT pair_ct, int allocator_id )
      : pair_ct_( pair_ct ),
        row_element_ids_( pair_ct, pair_ct, allocator_id ),
        column_element_ids_( pair_ct, pair_ct, allocator_id ),
        value_offsets_( pair_ct, pair_ct, allocator_id ),
        values_( RowDofs * ColumnDofs * pair_ct, RowDofs * ColumnDofs * pair_ct, allocator_id )
  {
  }

  /** @brief Return a device-copyable writable view. @return View retaining no ownership. */
  ElementBlockView<RowDofs, ColumnDofs> view()
  {
    return { row_element_ids_.data(), column_element_ids_.data(), value_offsets_.data(), values_.data(), pair_ct_ };
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
  IndexT pair_ct_;                      ///< Number of blocks in each metadata array.
  Array1D<IndexT> row_element_ids_;     ///< Row element IDs.
  Array1D<IndexT> column_element_ids_;  ///< Column element IDs.
  Array1D<IndexT> value_offsets_;       ///< Block value offsets.
  Array1D<RealT> values_;               ///< Column-major dense blocks.
};

}  // namespace tribol

#endif  // SRC_TRIBOL_ASSEMBLY_PAIRCONTRIBUTION_HPP_
