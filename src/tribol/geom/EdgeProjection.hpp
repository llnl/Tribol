// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Straight-edge projection, support checks, and smoothed integration bounds without FE basis dependencies.
 */

#ifndef SRC_TRIBOL_GEOM_EDGEPROJECTION_HPP_
#define SRC_TRIBOL_GEOM_EDGEPROJECTION_HPP_

#include <cmath>
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"

namespace tribol {

/**
 * @brief Integration bounds in the source edge's normalized [-1, 1] coordinates.
 *
 * Invalid geometry is distinct from a valid collapsed interval, which contributes zero points.
 *
 * @tparam Value Coordinate and weight scalar type.
 */
template <typename Value>
struct EdgeQuadratureInterval {
  MdArray<Value, 2> bounds{ -1.0, 1.0 };  ///< Lower and upper bounds; default to the whole source edge.
  bool valid{ true };                     ///< Whether the geometry calculation succeeded.
};

/**
 * @brief Normalized projection bounds; invalid edges return zero bounds and valid=false.
 *
 * @tparam Value Coordinate and weight scalar type.
 */
template <typename Value>
struct EdgeProjection {
  /** @brief Increasing bounds in the integration edge's normalized coordinate; zero when invalid */
  MdArray<Value, 2> bounds{ Value{}, Value{} };
  /** @brief Whether both edges are nondegenerate; overlap with the integration edge is checked separately */
  bool valid{ false };
};

/**
 * @brief Physical and parent coordinates of a point on a target edge.
 *
 * Invalid targets return zero coordinates and valid=false.
 *
 * @tparam Value Coordinate and weight scalar type.
 */
template <typename Value>
struct ProjectedEdgePoint {
  MdArray<Value, 2> physical_coordinates{};  ///< Projected physical point.
  Value parent_coordinate{};                 ///< Target parent coordinate on [-1, 1], possibly extrapolated.
  bool valid{ false };                       ///< Whether the target edge defines a valid projection.
};

namespace detail {
inline constexpr double minimum_edge_length = 1.0e-14;  ///< Physical edge-length degeneracy threshold.
inline constexpr double minimum_projection_determinant =
    1.0e-12;  ///< Determinant threshold for the line-intersection fallback.

/**
 * @brief Calculate the unit normal (dy, -dx) from ordered edge endpoints.
 *
 * @tparam Value Coordinate and weight scalar type.
 * @param [in] point0 First physical endpoint, with two components.
 * @param [in] point1 Second physical endpoint, with two components.
 * @param [out] normal Unit normal, or zero for a degenerate edge.
 * @return True for an edge meeting the minimum length; false otherwise.
 */
template <typename Value>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE bool linearEdgeNormal( const Value* point0, const Value* point1, Value* normal )
{
  const Value edge_x = point1[0] - point0[0];
  const Value edge_y = point1[1] - point0[1];
  const Value edge_length = std::sqrt( edge_x * edge_x + edge_y * edge_y );
  if ( edge_length < static_cast<Value>( minimum_edge_length ) ) {
    normal[0] = Value{};
    normal[1] = Value{};
    return false;
  }
  normal[0] = edge_y / edge_length;
  normal[1] = -edge_x / edge_length;
  return true;
}

/**
 * @brief Intersect a line with the line through a point in a supplied direction.
 *
 * @tparam Value Coordinate and weight scalar type.
 * @param [in] line_point0 First physical endpoint of the target line.
 * @param [in] line_point1 Second physical endpoint of the target line.
 * @param [in] projected_point Physical point to project.
 * @param [in] projection_direction Unit direction of the projection line.
 * @param [out] intersection Physical intersection, or projected_point when the determinant is below its threshold.
 */
template <typename Value>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE void lineIntersection( const Value* line_point0, const Value* line_point1,
                                                               const Value* projected_point,
                                                               const Value* projection_direction, Value* intersection )
{
  const Value line_x = line_point1[0] - line_point0[0];
  const Value line_y = line_point1[1] - line_point0[1];
  const Value point_x = projected_point[0] - line_point0[0];
  const Value point_y = projected_point[1] - line_point0[1];
  const Value determinant = line_x * projection_direction[1] - line_y * projection_direction[0];
  if ( std::abs( determinant ) < static_cast<Value>( minimum_projection_determinant ) ) {
    intersection[0] = projected_point[0];
    intersection[1] = projected_point[1];
    return;
  }

  const Value line_fraction = ( point_x * projection_direction[1] - point_y * projection_direction[0] ) / determinant;
  intersection[0] = line_point0[0] + line_fraction * line_x;
  intersection[1] = line_point0[1] + line_fraction * line_y;
}

/**
 * @brief Project target endpoints along the target normal onto the source line.
 *
 * An intersection determinant below its threshold uses the unprojected endpoint to calculate its parent coordinate.
 *
 * @tparam Value Coordinate and weight scalar type.
 * @param integration_point0 First source endpoint, with two physical components.
 * @param integration_point1 Second source endpoint, with two physical components.
 * @param projected_point0 First target endpoint, with two physical components.
 * @param projected_point1 Second target endpoint, with two physical components.
 * @return Sorted source parent bounds; degenerate edges return zero bounds and valid=false.
 */
template <typename Value>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE EdgeProjection<Value> linearEdgeProjectionBounds(
    const Value* integration_point0, const Value* integration_point1, const Value* projected_point0,
    const Value* projected_point1 )
{
  Value projected_normal[2];
  if ( !linearEdgeNormal( projected_point0, projected_point1, projected_normal ) ) {
    return {};
  }

  const Value integration_x = integration_point1[0] - integration_point0[0];
  const Value integration_y = integration_point1[1] - integration_point0[1];
  const Value integration_length_squared = integration_x * integration_x + integration_y * integration_y;
  const Value minimum_length = static_cast<Value>( minimum_edge_length );
  if ( integration_length_squared <= minimum_length * minimum_length ) {
    return {};
  }

  Value projected_intersection0[2];
  lineIntersection( integration_point0, integration_point1, projected_point0, projected_normal,
                    projected_intersection0 );
  const Value integration_fraction0 = ( ( projected_intersection0[0] - integration_point0[0] ) * integration_x +
                                        ( projected_intersection0[1] - integration_point0[1] ) * integration_y ) /
                                      integration_length_squared;

  Value projected_intersection1[2];
  lineIntersection( integration_point0, integration_point1, projected_point1, projected_normal,
                    projected_intersection1 );
  const Value integration_fraction1 = ( ( projected_intersection1[0] - integration_point0[0] ) * integration_x +
                                        ( projected_intersection1[1] - integration_point0[1] ) * integration_y ) /
                                      integration_length_squared;

  const Value coordinate0 = 2.0 * integration_fraction0 - 1.0;
  const Value coordinate1 = 2.0 * integration_fraction1 - 1.0;
  return { MdArray<Value, 2>{ coordinate0 < coordinate1 ? coordinate0 : coordinate1,
                              coordinate0 < coordinate1 ? coordinate1 : coordinate0 },
           true };
}
/**
 * @brief Check whether projected bounds reach into the expanded source support.
 *
 * @tparam Value Coordinate and weight scalar type.
 * @param bounds Increasing bounds in normalized source coordinates.
 * @param smoothing_fraction Support extension at each end as a fraction of physical source length.
 * @return Whether the upper bound passes the left support limit and the lower bound precedes the right limit.
 */
template <typename Value>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE bool edgeProjectionOverlapsSupport( MdSpan<const Value, 2> bounds,
                                                                            Value smoothing_fraction )
{
  const Value extension = 2.0 * smoothing_fraction;
  return bounds[1] > -1.0 - extension && bounds[0] < 1.0 + extension;
}

}  // namespace detail

/** @brief Integrate the whole source segment without projecting an interval or generating target coordinates */
struct NoEdgeProjection {
  static constexpr bool uses_target = false;  ///< Source integration needs no target edge.

  /**
   * @brief Apply no projected-overlap restriction beyond the search's other pair checks.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param source Source endpoints; unused by this projection.
   * @param target Target endpoints; unused by this projection.
   * @return Always true.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE bool mayContribute( [[maybe_unused]] MdSpan<const Value, 2, 2> source,
                                         [[maybe_unused]] MdSpan<const Value, 2, 2> target ) const
  {
    return true;
  }

  /**
   * @brief Select the whole source interval without consulting another surface.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param source Source endpoints; unused for selecting the reference interval.
   * @return The valid normalized interval [-1, 1].
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE EdgeQuadratureInterval<Value> integrationInterval(
      [[maybe_unused]] MdSpan<const Value, 2, 2> source ) const
  {
    return {};
  }
};

/**
 * @brief Project along the target edge's normal and smooth the source integration interval
 *
 * Both edges are straight segments in two dimensions. Source bounds use [-1, 1] parent coordinates. Search and
 * quadrature use the same smoothing fraction, expressed as a fraction of source length. Only configuration is retained.
 */
struct SmoothedEdgeProjection {
  static constexpr bool uses_target = true;  ///< Interval selection requires a target edge.
  double smoothing_fraction{ 0.1 };          ///< Smoothing length as a fraction of source length.

  /**
   * @brief Check projected bounds against the source support expanded by the smoothing fraction.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param source Source endpoints indexed by endpoint, then physical component.
   * @param target Target endpoints indexed by endpoint, then physical component.
   * @return False for degenerate edges or bounds outside the expanded support; true otherwise.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE bool mayContribute( MdSpan<const Value, 2, 2> source,
                                                              MdSpan<const Value, 2, 2> target ) const
  {
    const auto bounds =
        detail::linearEdgeProjectionBounds( source.data(), source.data() + 2, target.data(), target.data() + 2 );
    return bounds.valid && detail::edgeProjectionOverlapsSupport<Value>( bounds.bounds.view(),
                                                                         static_cast<Value>( smoothing_fraction ) );
  }

  /**
   * @brief Calculate projected source bounds and smooth them near its reference endpoints.
   *
   * A valid collapsed interval has zero integration measure.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param source Source endpoints indexed by endpoint, then physical component.
   * @param target Target endpoints projected along their edge normal.
   * @return Smoothed source bounds; degenerate edges return zero bounds and valid=false.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE EdgeQuadratureInterval<Value> integrationInterval(
      MdSpan<const Value, 2, 2> source, MdSpan<const Value, 2, 2> target ) const
  {
    const auto interval =
        detail::linearEdgeProjectionBounds( source.data(), source.data() + 2, target.data(), target.data() + 2 );
    if ( !interval.valid ) return { interval.bounds, false };
    const Value delta = static_cast<Value>( smoothing_fraction );
    const Value support_min = -1.0 - 2.0 * delta;
    const Value support_max = 1.0 + 2.0 * delta;
    const Value lower = interval.bounds[0] < support_min   ? support_min
                        : interval.bounds[0] > support_max ? support_max
                                                           : interval.bounds[0];
    const Value upper = interval.bounds[1] < support_min   ? support_min
                        : interval.bounds[1] > support_max ? support_max
                                                           : interval.bounds[1];
    const Value smooth_lower = smoothBound( lower, delta );
    const Value smooth_upper = smoothBound( upper, delta );
    return { MdArray<Value, 2>{ smooth_lower, smooth_upper }, true };
  }

  /**
   * @brief Check whether the target edge is long enough to define a projection.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param target Target endpoints indexed by endpoint, then physical component.
   * @return True when the target meets the minimum physical edge length.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE bool targetIsValid( MdSpan<const Value, 2, 2> target ) const
  {
    const Value dx = target( 1, 0 ) - target( 0, 0 );
    const Value dy = target( 1, 1 ) - target( 0, 1 );
    return dx * dx + dy * dy >= detail::minimum_edge_length * detail::minimum_edge_length;
  }

  /**
   * @brief Project a physical point along the target normal onto the infinite target line.
   *
   * Parent coordinates may lie outside [-1, 1].
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param point Source point with two physical components.
   * @param target Target endpoints indexed by endpoint, then physical component.
   * @return Physical and normalized target coordinates; invalid targets return zero coordinates and valid=false.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ProjectedEdgePoint<Value> projectPoint(
      MdSpan<const Value, 2> point, MdSpan<const Value, 2, 2> target ) const
  {
    if ( !targetIsValid( target ) ) return {};
    const Value dx = target( 1, 0 ) - target( 0, 0 );
    const Value dy = target( 1, 1 ) - target( 0, 1 );
    const Value fraction =
        ( ( point[0] - target( 0, 0 ) ) * dx + ( point[1] - target( 0, 1 ) ) * dy ) / ( dx * dx + dy * dy );
    return { MdArray<Value, 2>{ target( 0, 0 ) + fraction * dx, target( 0, 1 ) + fraction * dy }, 2.0 * fraction - 1.0,
             true };
  }

 private:
  /**
   * @brief Apply quadratic smoothing near the source reference endpoints.
   *
   * Keep endpoint calls separate so Enzyme does not reuse loop-local reverse-mode tape entries.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param bound Source parent coordinate after clipping to the expanded support.
   * @param delta Smoothing length as a fraction of physical source length; zero leaves bound unchanged.
   * @return Smoothed parent coordinate.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static Value smoothBound( Value bound, Value delta )
  {
    const Value unit = 0.5 * ( bound + 1.0 );
    Value smoothed = unit;
    if ( delta != 0.0 ) {
      if ( -delta <= unit && unit <= delta ) {
        smoothed = unit * unit / ( 4.0 * delta ) + 0.5 * unit + delta / 4.0;
      } else if ( 1.0 - delta <= unit && unit <= 1.0 + delta ) {
        const Value b = -1.0 / ( 4.0 * delta );
        const Value c = 0.5 + 1.0 / ( 2.0 * delta );
        const Value d = 1.0 - delta + ( 1.0 - delta ) * ( 1.0 - delta ) / ( 4.0 * delta ) - 0.5 * ( 1.0 - delta ) -
                        ( 1.0 - delta ) / ( 2.0 * delta );
        smoothed = b * unit * unit + c * unit + d;
      }
    }
    return 2.0 * smoothed - 1.0;
  }
};

}  // namespace tribol
#endif
