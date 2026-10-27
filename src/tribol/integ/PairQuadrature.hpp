// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Owning arrays and borrowed views for source/target parent coordinates sharing source weights.
 */

#ifndef SRC_TRIBOL_INTEG_PAIRQUADRATURE_HPP_
#define SRC_TRIBOL_INTEG_PAIRQUADRATURE_HPP_

#include <cassert>

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/integ/Quadrature.hpp"
#include "tribol/common/MdArray.hpp"

namespace tribol {

/**
 * @brief Borrow runtime-capacity source and target parent coordinates with source physical weights.
 *
 * Coordinates are [point][parent component]. The caller owns the arrays and keeps them alive while the view is used.
 * Capacity describes storage, independently of the number of points generated.
 *
 * @tparam Value Coordinate and weight scalar type.
 * @tparam ParentDimension Number of parent-coordinate components per point.
 */
template <typename Value, int ParentDimension>
struct PairQuadratureView {
  static_assert( ParentDimension == 1 || ParentDimension == 2 );

  ArrayViewT<Value, 2> mesh1_parent_coordinates;  ///< Borrowed source parent coordinates [point][component].
  ArrayViewT<Value, 2> mesh2_parent_coordinates;  ///< Borrowed target parent coordinates [point][component].
  ArrayViewT<Value> physical_weights;             ///< Borrowed weights containing the full source measure.

  /**
   * @brief Return the number of paired points that fit in the borrowed arrays.
   *
   * @return Length of physical_weights.
   */
  TRIBOL_HOST_DEVICE IndexT capacity() const { return physical_weights.size(); }

  /**
   * @brief Borrow paired coordinate arrays and source physical weights.
   *
   * All arrays must have matching point counts and remain alive while the view is used.
   *
   * @param mesh1 Source parent coordinates [point][ParentDimension].
   * @param mesh2 Target parent coordinates [point][ParentDimension].
   * @param weights Source physical weights, one per point.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureView( ArrayViewT<Value, 2> mesh1, ArrayViewT<Value, 2> mesh2,
                                                              ArrayViewT<Value> weights )
      : mesh1_parent_coordinates( mesh1 ), mesh2_parent_coordinates( mesh2 ), physical_weights( weights )
  {
    assert( mesh1.shape()[0] == weights.size() && mesh2.shape()[0] == weights.size() );
    assert( mesh1.shape()[1] == ParentDimension && mesh2.shape()[1] == ParentDimension );
  }
};

/**
 * @brief Own fixed-capacity paired quadrature for allocation-free element kernels.
 *
 * Copies retain their values. Views borrow this object's arrays and must not outlive it. Other generators may use
 * larger runtime-sized backing arrays.
 *
 * @tparam Value Coordinate and weight scalar type.
 * @tparam ParentDimension Number of parent-coordinate components per point.
 * @tparam Capacity Number of points available in the fixed-size backing arrays.
 */
template <typename Value, int ParentDimension, IndexT Capacity>
struct PairQuadratureData {
  MdArray<Value, Capacity, ParentDimension>
      mesh1_parent_coordinates{};  ///< Owned source parent coordinates [point][component].
  MdArray<Value, Capacity, ParentDimension>
      mesh2_parent_coordinates{};               ///< Owned target parent coordinates [point][component].
  MdArray<Value, Capacity> physical_weights{};  ///< Owned weights containing the full source measure.
  QuadratureResult result{};                    ///< Completion status and number of populated points.

  /**
   * @brief Borrow all available paired storage for generation.
   *
   * @return Writable arrays with Capacity entries, valid while this object remains alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureView<Value, ParentDimension> getOutputView() &
  {
    const axom::StackArray<axom::IndexType, 2> shape{ Capacity, ParentDimension };
    return { ArrayViewT<Value, 2>{ mesh1_parent_coordinates.data(), shape },
             ArrayViewT<Value, 2>{ mesh2_parent_coordinates.data(), shape },
             ArrayViewT<Value>{ physical_weights.data(), Capacity } };
  }

  /**
   * @brief Borrow only completed paired points.
   *
   * @pre result.status is Success and required_point_count does not exceed Capacity.
   *
   * @return Read-only arrays sized to required_point_count, valid while this object remains alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureView<const Value, ParentDimension> getView() const&
  {
    assert( result.status == QuadratureStatus::Success );
    assert( result.required_point_count >= 0 && result.required_point_count <= Capacity );
    const axom::StackArray<axom::IndexType, 2> shape{ result.required_point_count, ParentDimension };
    return { ArrayViewT<const Value, 2>{ mesh1_parent_coordinates.data(), shape },
             ArrayViewT<const Value, 2>{ mesh2_parent_coordinates.data(), shape },
             ArrayViewT<const Value>{ physical_weights.data(), result.required_point_count } };
  }

  /**
   * @brief Prevent borrowing writable paired storage from a temporary owner.
   */
  PairQuadratureView<Value, ParentDimension> getOutputView() && = delete;
  /**
   * @brief Prevent borrowing paired points from a temporary owner.
   */
  PairQuadratureView<const Value, ParentDimension> getView() const&& = delete;
};

}  // namespace tribol

#endif
