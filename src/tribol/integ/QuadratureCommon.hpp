// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Integration-surface and element-pair quadrature arrays, views, and generation results.
 *
 * Integration-surface representations shared by mappings and rule generators belong in this header. Keep
 * representations for additional element topologies or geometries here alongside EdgeQuadratureInterval.
 */

#ifndef SRC_TRIBOL_INTEG_QUADRATURECOMMON_HPP_
#define SRC_TRIBOL_INTEG_QUADRATURECOMMON_HPP_

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
 * @brief Borrow read-only quadrature points and weights expressed in the integration surface's coordinates.
 *
 * The mapping defines the coordinates and any conversion of weights to physical integration weights. Arrays are
 * indexed by point, then coordinate component. The caller retains ownership.
 *
 * @tparam CoordinateDimension Number of coordinate components stored for each integration-surface quadrature point.
 */
template <int CoordinateDimension>
struct IntegrationSurfaceRuleView {
  static_assert( CoordinateDimension == 1 || CoordinateDimension == 2 );
  ArrayViewT<const RealT, 2> coords;  ///< Borrowed integration-surface coordinates [point][component].
  ArrayViewT<const RealT> weights;    ///< Borrowed weights in integration-surface coordinates.

  /**
   * @brief Return the number of borrowed integration-surface quadrature points.
   *
   * @return Length of weights.
   */
  TRIBOL_HOST_DEVICE IndexT numberOfPoints() const { return weights.size(); }

  /**
   * @brief Borrow matching coordinate and weight arrays.
   *
   * Both arrays must have the same point count and remain alive while the view is used.
   *
   * @param coords Integration-surface coordinate array [point][CoordinateDimension].
   * @param weights Weights in the integration surface's coordinates, one per point.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurfaceRuleView( ArrayViewT<const RealT, 2> coords,
                                                                      ArrayViewT<const RealT> weights )
      : coords( coords ), weights( weights )
  {
    assert( coords.shape()[0] == weights.size() && coords.shape()[1] == CoordinateDimension );
  }
};

/**
 * @brief Own fixed-capacity integration-surface quadrature points and weights that can remain frozen as geometry
 * changes.
 *
 * Views borrow this object's arrays and must not outlive it.
 *
 * @tparam CoordinateDimension Number of coordinate components stored for each integration-surface quadrature point.
 * @tparam Capacity Number of points available in the fixed-size backing arrays.
 */
template <int CoordinateDimension, IndexT Capacity>
struct IntegrationSurfaceRule {
  MdArray<RealT, Capacity, CoordinateDimension> coords{};  ///< Integration-surface coordinates [point][component].
  MdArray<RealT, Capacity> weights{};                      ///< Owned weights in integration-surface coordinates.
  QuadratureResult result{};                               ///< Completion status and number of populated points.

  /**
   * @brief Borrow only completed integration-surface quadrature points and weights.
   *
   * @pre result.status is Success and required_point_count does not exceed Capacity.
   *
   * @return Read-only points and weights valid while this object remains alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurfaceRuleView<CoordinateDimension> getView() const&
  {
    assert( result.status == QuadratureStatus::Success );
    assert( result.required_point_count >= 0 && result.required_point_count <= Capacity );
    const axom::StackArray<axom::IndexType, 2> shape{ result.required_point_count, CoordinateDimension };
    return { ArrayViewT<const RealT, 2>{ coords.data(), shape },
             ArrayViewT<const RealT>{ weights.data(), result.required_point_count } };
  }

  /**
   * @brief Prevent borrowing integration-surface quadrature from a temporary owner.
   */
  IntegrationSurfaceRuleView<CoordinateDimension> getView() const&& = delete;
};

/**
 * @brief Borrow quadrature points in each element's parametric coordinates and physical integration weights.
 *
 * Parametric coordinates are the local element variables used for geometry mapping and basis evaluation.
 * LinearEdgeGeometry uses one coordinate, xi, with vertices at -1 and 1. Arrays are indexed by point, then coordinate
 * component. Mapping to the second element does not add another Jacobian.
 * @tparam ParametricDimension1 Number of coordinates used to parameterize geometry1: one for an edge, two for a face.
 * @tparam ParametricDimension2 Number of coordinates used to parameterize geometry2: one for an edge, two for a face.
 * @tparam ReadOnly Whether the borrowed coordinates and weights are read-only.
 */
template <int ParametricDimension1, int ParametricDimension2, bool ReadOnly = false>
struct PairQuadratureView {
  using Value = std::conditional_t<ReadOnly, const RealT, RealT>;  ///< RealT with the requested write access.
  ArrayViewT<Value, 2> geometry1_coords;  ///< Quadrature-point coordinates in geometry1's element parameterization.
  ArrayViewT<Value, 2> geometry2_coords;  ///< Quadrature-point coordinates in geometry2's element parameterization.
  ArrayViewT<Value> weights;              ///< Weights including the mapping's physical integration measure.

  /** @brief Return the available point capacity. @return Number of weight entries. */
  TRIBOL_HOST_DEVICE IndexT capacity() const { return weights.size(); }

  /**
   * @brief Borrow matching coordinate and weight arrays.
   * @param geometry1_points Quadrature-point coordinates in geometry1's element parameterization.
   * @param geometry2_points Quadrature-point coordinates in geometry2's element parameterization.
   * @param weights Physical integration weights.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureView( ArrayViewT<Value, 2> geometry1_points,
                                                              ArrayViewT<Value, 2> geometry2_points,
                                                              ArrayViewT<Value> weights )
      : geometry1_coords( geometry1_points ), geometry2_coords( geometry2_points ), weights( weights )
  {
    assert( geometry1_points.shape()[0] == weights.size() && geometry1_points.shape()[1] == ParametricDimension1 );
    assert( geometry2_points.shape()[0] == weights.size() && geometry2_points.shape()[1] == ParametricDimension2 );
  }
};

/**
 * @brief Own fixed-capacity quadrature points in each element's parametric coordinates and physical integration
 * weights.
 *
 * Parametric coordinates are the local element variables used for geometry mapping and basis evaluation.
 * LinearEdgeGeometry uses one coordinate, xi, with vertices at -1 and 1.
 * @tparam ParametricDimension1 Number of coordinates used to parameterize geometry1: one for an edge, two for a face.
 * @tparam ParametricDimension2 Number of coordinates used to parameterize geometry2: one for an edge, two for a face.
 * @tparam Capacity Available points in the local backing arrays.
 */
template <int ParametricDimension1, int ParametricDimension2, IndexT Capacity>
struct PairQuadratureData {
  MdArray<RealT, Capacity, ParametricDimension1>
      geometry1_coords{};  ///< Quadrature-point coordinates in geometry1's element parameterization.
  MdArray<RealT, Capacity, ParametricDimension2>
      geometry2_coords{};              ///< Quadrature-point coordinates in geometry2's element parameterization.
  MdArray<RealT, Capacity> weights{};  ///< Owned physical integration weights.
  QuadratureResult result{};           ///< Status and initialized point count.

  /** @brief Borrow all available output storage. @return Writable arrays valid while this owner is alive. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureView<ParametricDimension1, ParametricDimension2>
  getOutputView() &
  {
    return { { geometry1_coords.data(), { Capacity, ParametricDimension1 } },
             { geometry2_coords.data(), { Capacity, ParametricDimension2 } },
             { weights.data(), Capacity } };
  }

  /**
   * @brief Borrow only initialized points of a completed rule.
   * @pre result.status is Success and required_point_count does not exceed Capacity.
   * @return Read-only arrays valid while this owner is alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureView<ParametricDimension1, ParametricDimension2, true> getView()
      const&
  {
    assert( result.status == QuadratureStatus::Success );
    assert( result.required_point_count >= 0 && result.required_point_count <= Capacity );
    const IndexT count = result.required_point_count;
    return { { geometry1_coords.data(), { count, ParametricDimension1 } },
             { geometry2_coords.data(), { count, ParametricDimension2 } },
             { weights.data(), count } };
  }

  /** @brief Prevent borrowing output arrays from a temporary. */
  PairQuadratureView<ParametricDimension1, ParametricDimension2> getOutputView() && = delete;
  /** @brief Prevent borrowing initialized arrays from a temporary. */
  PairQuadratureView<ParametricDimension1, ParametricDimension2, true> getView() const&& = delete;
};

}  // namespace tribol

#endif
