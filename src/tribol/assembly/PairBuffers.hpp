// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Preallocated pair-local values and coordinate derivatives in structure-of-arrays buffers.
 *
 * Pair functions returning quadrature metadata store zero numerical outputs for failed or inactive evaluations.
 * That metadata must be checked before using the numerical outputs. Weighted derivative functions require prior
 * quadrature validation and return only derivatives.
 */
#ifndef SRC_TRIBOL_ASSEMBLY_PAIRBUFFERS_HPP_
#define SRC_TRIBOL_ASSEMBLY_PAIRBUFFERS_HPP_

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/** @brief Device-copyable pair values and quadrature state without derivatives. @tparam SizeValues Values per pair. */
template <IndexT SizeValues>
struct PairValueView {
  RealT* values{ nullptr };                            ///< Values in `[value][pair]` order.
  QuadratureMetadata* quadrature_metadata{ nullptr };  ///< Pair quadrature status and point count.
  IndexT num_pairs{ 0 };                               ///< Number of writable pair slots.

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
        quadrature_metadata_( num_pairs, num_pairs, allocator_id )
  {
  }
  /** @brief Return a writable view. @return View into this buffer's arrays; the buffer must remain alive while the view
   * is used. */
  PairValueView<SizeValues> view() { return { values_.data(), quadrature_metadata_.data(), num_pairs_ }; }
  /** @brief Return pair slots. @return Number of pairs allocated. */
  IndexT numberOfPairs() const { return num_pairs_; }
  /** @brief Return the owning values. @return Values in `[value][pair]` order. */
  const Array1D<RealT>& values() const { return values_; }
  /** @brief Return quadrature metadata. @return One entry per pair. */
  const Array1D<QuadratureMetadata>& quadratureMetadata() const { return quadrature_metadata_; }

 private:
  IndexT num_pairs_;                                 ///< Allocated pair slots.
  Array1D<RealT> values_;                            ///< Numerical values in structure-of-arrays order.
  Array1D<QuadratureMetadata> quadrature_metadata_;  ///< Quadrature metadata for each pair.
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
  RealT* values{ nullptr };                            ///< Values in `[value][pair]` order.
  RealT* jacobians{ nullptr };                         ///< Derivatives in `[value][coordinate][pair]` order.
  QuadratureMetadata* quadrature_metadata{ nullptr };  ///< Pair quadrature status and point count.
  IndexT num_pairs{ 0 };                               ///< Number of writable pair slots.

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
        quadrature_metadata_( num_pairs, num_pairs, allocator_id )
  {
  }

  /** @brief Return a device-copyable writable view. @return View retaining no ownership. */
  PairValueJacobianView<SizeValues, SizeCoords> view()
  {
    return { values_.data(), jacobians_.data(), quadrature_metadata_.data(), num_pairs_ };
  }

  /** @brief Return the number of pair slots. @return Pair count. */
  IndexT numberOfPairs() const { return num_pairs_; }

  /** @brief Return the owning value array. @return Values in structure-of-arrays order. */
  const Array1D<RealT>& values() const { return values_; }

  /** @brief Return the owning Jacobian array. @return Jacobians in structure-of-arrays order. */
  const Array1D<RealT>& jacobians() const { return jacobians_; }

  /** @brief Return quadrature metadata. @return One entry per pair. */
  const Array1D<QuadratureMetadata>& quadratureMetadata() const { return quadrature_metadata_; }

 private:
  IndexT num_pairs_;                                 ///< Pair slots allocated in every array.
  Array1D<RealT> values_;                            ///< Values in `[value][pair]` order.
  Array1D<RealT> jacobians_;                         ///< Derivatives in `[value][coordinate][pair]` order.
  Array1D<QuadratureMetadata> quadrature_metadata_;  ///< Quadrature metadata for each pair.
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

/** @brief Device-copyable pair values, quadrature metadata, Jacobians, and Hessians. @tparam SizeCoords Coordinates.
 */
template <IndexT SizeCoords>
struct PairValueJacobianHessianView {
  RealT* values{ nullptr };                            ///< One scalar value per pair.
  QuadratureMetadata* quadrature_metadata{ nullptr };  ///< One quadrature metadata entry per pair.
  PairJacobianHessianView<SizeCoords> derivatives{};   ///< Coordinate Jacobians and Hessians.

  /** @brief Read the scalar value. @param pair Pair slot. @return Function value. */
  TRIBOL_HOST_DEVICE RealT value( IndexT pair ) const { return values[pair]; }

  /** @brief Read a Jacobian entry. @param pair Pair slot. @param coordinate Coordinate index. @return Entry value. */
  TRIBOL_HOST_DEVICE RealT jacobian( IndexT pair, IndexT coordinate ) const
  {
    return derivatives.jacobian( pair, coordinate );
  }

  /** @brief Read a Hessian entry. @param pair Pair slot. @param row Row index. @param column Column index.
   * @return Entry value. */
  TRIBOL_HOST_DEVICE RealT hessian( IndexT pair, IndexT row, IndexT column ) const
  {
    return derivatives.hessian( pair, row, column );
  }
};

/** @brief Own pair values and their coordinate derivatives in execution-space memory. @tparam SizeCoords Coordinates.
 */
template <IndexT SizeCoords>
class PairValueJacobianHessianBuffer {
 public:
  /** @brief Allocate all pair outputs. @param num_pairs Pair count. @param allocator_id Execution allocator. */
  PairValueJacobianHessianBuffer( IndexT num_pairs, int allocator_id )
      : values_( num_pairs, num_pairs, allocator_id ),
        quadrature_metadata_( num_pairs, num_pairs, allocator_id ),
        derivatives_( num_pairs, allocator_id )
  {
  }

  /** @brief Return a device-copyable writable view. @return View retaining no ownership. */
  PairValueJacobianHessianView<SizeCoords> view()
  {
    return { values_.data(), quadrature_metadata_.data(), derivatives_.view() };
  }

  /** @brief Return quadrature metadata. @return One entry per pair. */
  const Array1D<QuadratureMetadata>& quadratureMetadata() const { return quadrature_metadata_; }

 private:
  Array1D<RealT> values_;                              ///< Scalar values, zero for empty or inactive pairs.
  Array1D<QuadratureMetadata> quadrature_metadata_;    ///< Quadrature metadata for each pair.
  PairJacobianHessianBuffer<SizeCoords> derivatives_;  ///< Coordinate Jacobians and Hessians.
};

}  // namespace tribol

#endif  // SRC_TRIBOL_ASSEMBLY_PAIRBUFFERS_HPP_
