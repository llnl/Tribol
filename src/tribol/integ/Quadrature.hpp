// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Owning arrays, borrowed views, and completion results for source-surface quadrature.
 */

#ifndef SRC_TRIBOL_INTEG_QUADRATURE_HPP_
#define SRC_TRIBOL_INTEG_QUADRATURE_HPP_

#include <cassert>

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"

namespace tribol {

/** @brief Whether a complete rule was generated, needs more storage, or has invalid geometry */
enum class QuadratureStatus
{
  Success,               ///< Every reported point is initialized; zero points means zero measure.
  InsufficientCapacity,  ///< Reported count exceeds capacity; discard partial output.
  InvalidGeometry        ///< Geometry calculation failed; discard output and report zero points.
};

/**
 * @brief Reports the complete point count, even when the supplied storage is too small
 *
 * Success initializes every point below required_point_count; success with zero points means zero integration
 * measure. InsufficientCapacity may leave partial output, which must be discarded. InvalidGeometry
 * returns zero points and its output must also be discarded. Unused capacity is never part of a completed rule.
 */
struct QuadratureResult {
  QuadratureStatus status{ QuadratureStatus::Success };  ///< Whether the complete rule can be consumed.
  IndexT required_point_count{ 0 };  ///< Complete count on success or overflow; zero for invalid geometry.
};

/**
 * @brief Borrow source parent points and weights before applying a physical Jacobian.
 *
 * Arrays are indexed by point, then parent component. The caller retains ownership.
 *
 * @tparam Value Coordinate and weight scalar type.
 * @tparam ParentDimension Number of parent-coordinate components per point.
 */
template <typename Value, int ParentDimension>
struct ReferenceQuadratureView {
  static_assert( ParentDimension == 1 || ParentDimension == 2 );
  ArrayViewT<Value, 2> parent_coordinates;  ///< Borrowed source parent coordinates [point][component].
  ArrayViewT<Value> reference_weights;      ///< Borrowed weights before physical Jacobian scaling.

  /**
   * @brief Return the number of borrowed reference points.
   *
   * @return Length of reference_weights.
   */
  TRIBOL_HOST_DEVICE IndexT numberOfPoints() const { return reference_weights.size(); }

  /**
   * @brief Borrow matching coordinate and weight arrays.
   *
   * Both arrays must have the same point count and remain alive while the view is used.
   *
   * @param coordinates Parent-coordinate array [point][ParentDimension].
   * @param weights Reference weights, one per point.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ReferenceQuadratureView( ArrayViewT<Value, 2> coordinates,
                                                                   ArrayViewT<Value> weights )
      : parent_coordinates( coordinates ), reference_weights( weights )
  {
    assert( coordinates.shape()[0] == weights.size() && coordinates.shape()[1] == ParentDimension );
  }
};

/**
 * @brief Own fixed-capacity parent points and weights that can remain frozen while geometry changes.
 *
 * Views borrow this object's arrays and must not outlive it.
 *
 * @tparam Value Coordinate and weight scalar type.
 * @tparam ParentDimension Number of parent-coordinate components per point.
 * @tparam Capacity Number of points available in the fixed-size backing arrays.
 */
template <typename Value, int ParentDimension, IndexT Capacity>
struct ReferenceQuadratureData {
  MdArray<Value, Capacity, ParentDimension>
      parent_coordinates{};                      ///< Owned source parent coordinates [point][component].
  MdArray<Value, Capacity> reference_weights{};  ///< Owned weights before physical Jacobian scaling.
  QuadratureResult result{};                     ///< Completion status and number of populated points.

  /**
   * @brief Borrow only completed reference points.
   *
   * @pre result.status is Success and required_point_count does not exceed Capacity.
   *
   * @return Read-only points and weights valid while this object remains alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ReferenceQuadratureView<const Value, ParentDimension> getView() const&
  {
    assert( result.status == QuadratureStatus::Success );
    assert( result.required_point_count >= 0 && result.required_point_count <= Capacity );
    const axom::StackArray<axom::IndexType, 2> shape{ result.required_point_count, ParentDimension };
    return { ArrayViewT<const Value, 2>{ parent_coordinates.data(), shape },
             ArrayViewT<const Value>{ reference_weights.data(), result.required_point_count } };
  }

  /**
   * @brief Prevent borrowing reference points from a temporary owner.
   */
  ReferenceQuadratureView<const Value, ParentDimension> getView() const&& = delete;
};

/**
 * @brief Borrow source-surface parent coordinates, physical coordinates, and physical weights.
 *
 * Coordinate arrays are indexed by point, then component. Weights include only the source surface measure; target
 * projection must not apply another Jacobian.
 *
 * @tparam Value Coordinate and weight scalar type.
 * @tparam ParentDimension Number of parent-coordinate components per point.
 * @tparam SpatialDimension Number of physical-coordinate components per point.
 */
template <typename Value, int ParentDimension, int SpatialDimension>
struct SurfaceQuadratureView {
  ArrayViewT<Value, 2> parent_coordinates;    ///< Borrowed source parent coordinates [point][component].
  ArrayViewT<Value, 2> physical_coordinates;  ///< Borrowed physical source points [point][component].
  ArrayViewT<Value> physical_weights;         ///< Borrowed weights including the complete source measure.

  /**
   * @brief Return the number of source points that fit in the borrowed arrays.
   *
   * @return Length of physical_weights.
   */
  TRIBOL_HOST_DEVICE IndexT capacity() const { return physical_weights.size(); }

  /**
   * @brief Borrow matching source-point and weight arrays.
   *
   * All arrays must have the same point count and remain alive while the view is used.
   *
   * @param parent Parent coordinates [point][ParentDimension].
   * @param physical Physical points [point][SpatialDimension].
   * @param weights Physical source weights, one per point.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE SurfaceQuadratureView( ArrayViewT<Value, 2> parent,
                                                                 ArrayViewT<Value, 2> physical,
                                                                 ArrayViewT<Value> weights )
      : parent_coordinates( parent ), physical_coordinates( physical ), physical_weights( weights )
  {
    assert( parent.shape()[0] == weights.size() && physical.shape()[0] == weights.size() );
    assert( parent.shape()[1] == ParentDimension && physical.shape()[1] == SpatialDimension );
  }
};

/**
 * @brief Own fixed-capacity source quadrature for allocation-free element calculations.
 *
 * Copies own their values. Views borrow this object's arrays and must not outlive it.
 *
 * @tparam Value Coordinate and weight scalar type.
 * @tparam ParentDimension Number of parent-coordinate components per point.
 * @tparam SpatialDimension Number of physical-coordinate components per point.
 * @tparam Capacity Number of points available in the fixed-size backing arrays.
 */
template <typename Value, int ParentDimension, int SpatialDimension, IndexT Capacity>
struct SurfaceQuadratureData {
  MdArray<Value, Capacity, ParentDimension>
      parent_coordinates{};  ///< Owned source parent coordinates [point][component].
  MdArray<Value, Capacity, SpatialDimension>
      physical_coordinates{};                   ///< Owned physical source points [point][component].
  MdArray<Value, Capacity> physical_weights{};  ///< Owned weights including the complete source measure.
  QuadratureResult result{};                    ///< Completion status and number of populated points.

  /**
   * @brief Borrow all available source storage for generation.
   *
   * @return Writable arrays with Capacity entries, valid while this object remains alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE SurfaceQuadratureView<Value, ParentDimension, SpatialDimension>
  getOutputView() &
  {
    return { ArrayViewT<Value, 2>{ parent_coordinates.data(),
                                   axom::StackArray<axom::IndexType, 2>{ Capacity, ParentDimension } },
             ArrayViewT<Value, 2>{ physical_coordinates.data(),
                                   axom::StackArray<axom::IndexType, 2>{ Capacity, SpatialDimension } },
             ArrayViewT<Value>{ physical_weights.data(), Capacity } };
  }

  /**
   * @brief Borrow only completed source points.
   *
   * @pre result.status is Success and required_point_count does not exceed Capacity.
   *
   * @return Read-only arrays sized to required_point_count, valid while this object remains alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE SurfaceQuadratureView<const Value, ParentDimension, SpatialDimension>
  getView() const&
  {
    assert( result.status == QuadratureStatus::Success );
    assert( result.required_point_count >= 0 && result.required_point_count <= Capacity );
    const IndexT count = result.required_point_count;
    return { ArrayViewT<const Value, 2>{ parent_coordinates.data(),
                                         axom::StackArray<axom::IndexType, 2>{ count, ParentDimension } },
             ArrayViewT<const Value, 2>{ physical_coordinates.data(),
                                         axom::StackArray<axom::IndexType, 2>{ count, SpatialDimension } },
             ArrayViewT<const Value>{ physical_weights.data(), count } };
  }

  /**
   * @brief Prevent borrowing writable storage from a temporary owner.
   */
  SurfaceQuadratureView<Value, ParentDimension, SpatialDimension> getOutputView() && = delete;
  /**
   * @brief Prevent borrowing source points from a temporary owner.
   */
  SurfaceQuadratureView<const Value, ParentDimension, SpatialDimension> getView() const&& = delete;
};

}  // namespace tribol

#endif
