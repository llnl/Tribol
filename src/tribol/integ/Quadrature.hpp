// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Local reference and element-pair quadrature arrays, views, and generation results.
 */

#ifndef SRC_TRIBOL_INTEG_QUADRATURE_HPP_
#define SRC_TRIBOL_INTEG_QUADRATURE_HPP_

#include <cassert>
#include <type_traits>

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"

namespace tribol {

/**
 * @brief Bounds on a one-dimensional integration surface in normalized parent coordinates.
 */
struct EdgeQuadratureInterval {
  MdArray<RealT, 2> bounds{ -1.0, 1.0 };  ///< Lower and upper bounds; default to the whole reference edge.
  bool valid{ true };                     ///< False for invalid geometry; collapsed valid bounds have zero measure.
};

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
 * @brief Borrow read-only integration-surface points and weights before applying the mapping's physical measure.
 *
 * Arrays are indexed by point, then parent component. The caller retains ownership.
 *
 * @tparam ParentDimension Number of parent-coordinate components per point.
 */
template <int ParentDimension>
struct ReferenceQuadratureView {
  static_assert( ParentDimension == 1 || ParentDimension == 2 );
  ArrayViewT<const RealT, 2> parent_coordinates;  ///< Borrowed integration-surface coordinates [point][component].
  ArrayViewT<const RealT> reference_weights;      ///< Borrowed weights before physical Jacobian scaling.

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
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ReferenceQuadratureView( ArrayViewT<const RealT, 2> coordinates,
                                                                   ArrayViewT<const RealT> weights )
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
 * @tparam ParentDimension Number of parent-coordinate components per point.
 * @tparam Capacity Number of points available in the fixed-size backing arrays.
 */
template <int ParentDimension, IndexT Capacity>
struct ReferenceQuadratureData {
  MdArray<RealT, Capacity, ParentDimension>
      parent_coordinates{};                      ///< Owned integration-surface coordinates [point][component].
  MdArray<RealT, Capacity> reference_weights{};  ///< Owned weights before physical Jacobian scaling.
  QuadratureResult result{};                     ///< Completion status and number of populated points.

  /**
   * @brief Borrow only completed reference points.
   *
   * @pre result.status is Success and required_point_count does not exceed Capacity.
   *
   * @return Read-only points and weights valid while this object remains alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ReferenceQuadratureView<ParentDimension> getView() const&
  {
    assert( result.status == QuadratureStatus::Success );
    assert( result.required_point_count >= 0 && result.required_point_count <= Capacity );
    const axom::StackArray<axom::IndexType, 2> shape{ result.required_point_count, ParentDimension };
    return { ArrayViewT<const RealT, 2>{ parent_coordinates.data(), shape },
             ArrayViewT<const RealT>{ reference_weights.data(), result.required_point_count } };
  }

  /**
   * @brief Prevent borrowing reference points from a temporary owner.
   */
  ReferenceQuadratureView<ParentDimension> getView() const&& = delete;
};

/**
 * @brief Borrow parent points on both elements and weights for one physical integration measure.
 *
 * Arrays are indexed by point, then parent component. Mapping to the second element does not add another Jacobian.
 * @tparam ParentDimension1 Parent components on geometry1.
 * @tparam ParentDimension2 Parent components on geometry2.
 * @tparam ReadOnly Whether the borrowed coordinates and weights are read-only.
 */
template <int ParentDimension1, int ParentDimension2, bool ReadOnly = false>
struct PairQuadratureView {
  using Value = std::conditional_t<ReadOnly, const RealT, RealT>;  ///< RealT with the requested write access.
  ArrayViewT<Value, 2> geometry1_parent_coordinates;               ///< Points on geometry1 [point][parent component].
  ArrayViewT<Value, 2> geometry2_parent_coordinates;  ///< Corresponding points on geometry2 [point][parent component].
  ArrayViewT<Value> physical_weights;                 ///< Weights including the mapping's physical integration measure.

  /** @brief Return the available point capacity. @return Number of weight entries. */
  TRIBOL_HOST_DEVICE IndexT capacity() const { return physical_weights.size(); }

  /**
   * @brief Borrow matching coordinate and weight arrays.
   * @param geometry1_points Parent-coordinate array for the first geometry.
   * @param geometry2_points Parent-coordinate array for the second geometry.
   * @param weights Physical integration weights.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureView( ArrayViewT<Value, 2> geometry1_points,
                                                              ArrayViewT<Value, 2> geometry2_points,
                                                              ArrayViewT<Value> weights )
      : geometry1_parent_coordinates( geometry1_points ),
        geometry2_parent_coordinates( geometry2_points ),
        physical_weights( weights )
  {
    assert( geometry1_points.shape()[0] == weights.size() && geometry1_points.shape()[1] == ParentDimension1 );
    assert( geometry2_points.shape()[0] == weights.size() && geometry2_points.shape()[1] == ParentDimension2 );
  }
};

/**
 * @brief Fixed-capacity quadrature consumed within one element-pair calculation.
 * @tparam ParentDimension1 Parent components on geometry1.
 * @tparam ParentDimension2 Parent components on geometry2.
 * @tparam Capacity Available points in the local backing arrays.
 */
template <int ParentDimension1, int ParentDimension2, IndexT Capacity>
struct PairQuadratureData {
  MdArray<RealT, Capacity, ParentDimension1> geometry1_parent_coordinates{};  ///< Owned geometry1 parent points.
  MdArray<RealT, Capacity, ParentDimension2> geometry2_parent_coordinates{};  ///< Owned geometry2 parent points.
  MdArray<RealT, Capacity> physical_weights{};                                ///< Owned physical integration weights.
  QuadratureResult result{};                                                  ///< Status and initialized point count.

  /** @brief Borrow all available output storage. @return Writable arrays valid while this owner is alive. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureView<ParentDimension1, ParentDimension2> getOutputView() &
  {
    return { { geometry1_parent_coordinates.data(), { Capacity, ParentDimension1 } },
             { geometry2_parent_coordinates.data(), { Capacity, ParentDimension2 } },
             { physical_weights.data(), Capacity } };
  }

  /**
   * @brief Borrow only initialized points of a completed rule.
   * @pre result.status is Success and required_point_count does not exceed Capacity.
   * @return Read-only arrays valid while this owner is alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureView<ParentDimension1, ParentDimension2, true> getView() const&
  {
    assert( result.status == QuadratureStatus::Success );
    assert( result.required_point_count >= 0 && result.required_point_count <= Capacity );
    const IndexT count = result.required_point_count;
    return { { geometry1_parent_coordinates.data(), { count, ParentDimension1 } },
             { geometry2_parent_coordinates.data(), { count, ParentDimension2 } },
             { physical_weights.data(), count } };
  }

  /** @brief Prevent borrowing output arrays from a temporary. */
  PairQuadratureView<ParentDimension1, ParentDimension2> getOutputView() && = delete;
  /** @brief Prevent borrowing initialized arrays from a temporary. */
  PairQuadratureView<ParentDimension1, ParentDimension2, true> getView() const&& = delete;
};

}  // namespace tribol

#endif
