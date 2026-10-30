// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Forward and reverse mappings between a projected integration interval and two linear edges.
 */

#ifndef SRC_TRIBOL_GEOM_SMOOTHEDPROJECTEDEDGEMAPPING_HPP_
#define SRC_TRIBOL_GEOM_SMOOTHEDPROJECTEDEDGEMAPPING_HPP_

#include <cmath>

#include "tribol/geom/LinearEdgeGeometry.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/**
 * @brief Integrate on edge1's smoothed projected interval and map its points to both straight edges.
 *
 * The forward mapping projects edge2's endpoints along its normal onto edge1. The reverse mapping keeps each
 * point's edge1 parent coordinate and projects its physical position along edge2's normal onto edge2.
 * Parent points may lie outside [-1, 1]. Weights use edge1's physical measure exactly once; edge2 adds no
 * Jacobian factor. Co-oriented normals contribute no points; perpendicular normals remain allowed.
 * Only configuration is retained; current endpoint coordinates are supplied to each operation.
 */
struct SmoothedProjectedEdgeMapping {
  using FirstGeometry = LinearEdgeGeometry;   ///< Non-mortar integration edge type required by PairQuadratureRule.
  using SecondGeometry = LinearEdgeGeometry;  ///< Mortar projection edge type required by PairQuadratureRule.
  using Point = LinearEdgeGeometry::Point;    ///< Physical point used for forward and reverse projection.
  using Vector = LinearEdgeGeometry::Vector;  ///< Directed edge, normal, or point displacement.
  /** @brief Native integration surface, expressed in edge1 parent coordinates. */
  using IntegrationSurface = EdgeQuadratureInterval;
  RealT smoothing_fraction{ 0.1 };  ///< Smoothing length as a fraction of the first edge's length.

  /**
   * @brief Calculate projected edge1 bounds and smooth them near its reference endpoints.
   *
   * Co-oriented geometry returns a valid collapsed interval with zero integration measure.
   *
   * @param edge1 Current non-mortar integration edge.
   * @param edge2 Current mortar edge projected along its normal.
   * @return Smoothed edge1 bounds; degenerate edges return zero bounds and valid=false.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurface makeIntegrationSurface( const FirstGeometry& edge1,
                                                                                     const SecondGeometry& edge2 ) const
  {
    const auto interval = projectedBounds( edge1, edge2 );
    if ( !interval.valid ) return { interval.bounds, false };
    const RealT delta = smoothing_fraction;
    const RealT support_min = -1.0 - 2.0 * delta;
    const RealT support_max = 1.0 + 2.0 * delta;
    const RealT lower = interval.bounds[0] < support_min   ? support_min
                        : interval.bounds[0] > support_max ? support_max
                                                           : interval.bounds[0];
    const RealT upper = interval.bounds[1] < support_min   ? support_min
                        : interval.bounds[1] > support_max ? support_max
                                                           : interval.bounds[1];
    const RealT smooth_lower = smoothBound( lower, delta );
    const RealT smooth_upper = smoothBound( upper, delta );
    return { MdArray<RealT, 2>{ smooth_lower, smooth_upper }, true };
  }

  /**
   * @brief Map integration-surface quadrature points onto both current elements and scale their weights.
   *
   * No overlap search is repeated. This also maps frozen edge1 parent points while geometry changes. Co-oriented
   * current geometry contributes zero points even when the supplied integration-surface quadrature is nonempty.
   * @param edge1 Current non-mortar integration edge.
   * @param edge2 Current mortar projection edge.
   * @param integration_surface_rule Quadrature points and weights expressed in edge1's parent coordinate.
   * @param [out] output Parent points on both elements and physical integration weights.
   * @return Success with the initialized count, InsufficientCapacity with the full count, or InvalidGeometry with zero
   * points. Invalid geometry and insufficient capacity leave output untouched. Valid zero measure returns zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult
  mapToElements( const FirstGeometry& edge1, const SecondGeometry& edge2,
                 IntegrationSurfaceRuleView<1> integration_surface_rule, PairQuadratureView<1, 1> output ) const
  {
    if ( edge1.squaredLength() < minimum_edge_length_squared_ ||
         edge2.squaredLength() < minimum_edge_length_squared_ ) {
      return { QuadratureStatus::InvalidGeometry, 0 };
    }
    if ( normalsAreCooriented( edge1, edge2 ) ) return { QuadratureStatus::Success, 0 };
    const IndexT count = integration_surface_rule.numberOfPoints();
    if ( output.capacity() < count ) return { QuadratureStatus::InsufficientCapacity, count };
    if ( count == 0 ) return { QuadratureStatus::Success, 0 };
    const RealT jacobian = edge1.jacobian();
    for ( IndexT q = 0; q < count; ++q ) {
      const RealT parent1 = integration_surface_rule.coords( q, 0 );
      const auto physical = edge1.position( parent1 );
      output.geometry1_coords( q, 0 ) = parent1;
      output.geometry2_coords( q, 0 ) = edge2.parentCoordinate( physical );
      output.weights[q] = integration_surface_rule.weights[q] * jacobian;
    }
    return { QuadratureStatus::Success, count };
  }

 private:
  inline static constexpr RealT minimum_edge_length_ = 1.0e-14;  ///< Shortest accepted physical edge length.
  inline static constexpr RealT minimum_edge_length_squared_ =
      minimum_edge_length_ * minimum_edge_length_;  ///< Squared threshold; equality is accepted on either element.
  inline static constexpr RealT minimum_projection_determinant_ = 1.0e-12;  ///< Line-intersection fallback threshold.

  /**
   * @brief Check whether the two edge normals have a positive dot product without normalizing them.
   * @param edge1 Current integration edge.
   * @param edge2 Current projection edge.
   * @pre Both edges are nondegenerate.
   * @return True for co-oriented normals; perpendicular normals remain allowed.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static bool normalsAreCooriented( const FirstGeometry& edge1,
                                                                            const SecondGeometry& edge2 )
  {
    // Rotating both directed edges to their normals preserves the dot product's sign.
    return edge1.direction().dot( edge2.direction() ) > 0.0;
  }

  /**
   * @brief Project both edge2 endpoints along its normal onto the infinite edge1 line.
   * @param edge1 Non-mortar integration edge.
   * @param edge2 Mortar projection edge.
   * @return Increasing edge1 parent bounds, valid zero bounds for co-oriented normals, or invalid zero bounds
   * when either edge is degenerate.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static IntegrationSurface projectedBounds( const FirstGeometry& edge1,
                                                                                     const SecondGeometry& edge2 )
  {
    if ( edge1.squaredLength() < minimum_edge_length_squared_ ||
         edge2.squaredLength() < minimum_edge_length_squared_ ) {
      return { MdArray<RealT, 2>{ 0.0, 0.0 }, false };
    }
    if ( normalsAreCooriented( edge1, edge2 ) ) return { MdArray<RealT, 2>{ 0.0, 0.0 }, true };
    const auto normal = edge2.normal();
    const RealT determinant = Vector::cross_product( edge1.direction(), normal )[2];
    const RealT coordinate0 = projectedParentCoordinate( edge1, edge2.origin(), normal, determinant );
    const RealT coordinate1 = projectedParentCoordinate( edge1, edge2.position( 1.0 ), normal, determinant );
    return { MdArray<RealT, 2>{ coordinate0 < coordinate1 ? coordinate0 : coordinate1,
                                coordinate0 < coordinate1 ? coordinate1 : coordinate0 },
             true };
  }

  /**
   * @brief Calculate the edge1 parent coordinate of a edge2 point projected along its normal.
   * @param edge1 Non-mortar integration edge.
   * @param point Physical point on edge2.
   * @param normal Projection direction from edge2 to the edge1 line.
   * @param determinant Cross product of the edge1 direction and projection normal, shared by both endpoints.
   * @return Parent coordinate of the intersection; nearly parallel lines use the point's orthogonal parent coordinate.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static RealT projectedParentCoordinate( const FirstGeometry& edge1,
                                                                                  const Point& point,
                                                                                  const Vector& normal,
                                                                                  RealT determinant )
  {
    if ( std::abs( determinant ) < minimum_projection_determinant_ ) return edge1.parentCoordinate( point );
    const RealT fraction = Vector::cross_product( Vector( edge1.origin(), point ), normal )[2] / determinant;
    return 2.0 * fraction - 1.0;
  }

  /**
   * @brief Apply quadratic smoothing near the edge1 reference endpoints.
   *
   * Keep endpoint calls separate so Enzyme does not reuse loop-local reverse-mode tape entries.
   *
   * @param bound Edge1 parent coordinate after clipping to the expanded support.
   * @param delta Smoothing length as a fraction of physical edge1 length; zero leaves bound unchanged.
   * @return Smoothed parent coordinate.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static RealT smoothBound( RealT bound, RealT delta )
  {
    const RealT unit = 0.5 * ( bound + 1.0 );
    RealT smoothed = unit;
    if ( delta != 0.0 ) {
      if ( -delta <= unit && unit <= delta ) {
        const RealT distance = unit + delta;
        smoothed = distance * distance / ( 4.0 * delta );
      } else if ( 1.0 - delta <= unit && unit <= 1.0 + delta ) {
        const RealT distance = 1.0 + delta - unit;
        smoothed = 1.0 - distance * distance / ( 4.0 * delta );
      }
    }
    return 2.0 * smoothed - 1.0;
  }
};

}  // namespace tribol

#endif
