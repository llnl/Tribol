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
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/Quadrature.hpp"

namespace tribol {

/**
 * @brief Geometry of a straight edge with two endpoints in two physical dimensions, independent of an FE basis.
 */
struct LinearEdgeGeometry {
  static constexpr int endpoint_count = 2;     ///< Endpoints defining the straight edge.
  static constexpr int parent_dimension = 1;   ///< Components in normalized [-1, 1] parent coordinates.
  static constexpr int spatial_dimension = 2;  ///< Physical components per endpoint.

  /** @brief Borrow endpoints by endpoint, then physical component. */
  using Coordinates = MdSpan<const RealT, endpoint_count, spatial_dimension>;

  /**
   * @brief Evaluate a physical point on the straight edge, allowing extrapolation.
   * @param coordinates Edge endpoint coordinates.
   * @param parent Normalized parent coordinate.
   * @return Physical point at parent.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static MdArray<RealT, spatial_dimension> position( Coordinates coordinates,
                                                                                             RealT parent )
  {
    MdArray<RealT, spatial_dimension> result;
    for ( int d = 0; d < spatial_dimension; ++d ) {
      result[d] = 0.5 * ( ( 1.0 - parent ) * coordinates( 0, d ) + ( 1.0 + parent ) * coordinates( 1, d ) );
    }
    return result;
  }
};

/**
 * @brief Integrate on mesh1's smoothed projected interval and map its points to both straight edges.
 *
 * The forward mapping projects mesh2's endpoints along its normal onto mesh1. The reverse mapping keeps each point's
 * mesh1 parent coordinate and projects its physical position along mesh2's normal onto mesh2. Parent points may lie
 * outside [-1, 1]. Weights use mesh1's physical measure exactly once; mesh2 adds no Jacobian factor.
 * Only configuration is retained; current endpoint coordinates are supplied to each operation.
 */
struct SmoothedProjectedEdgeMapping {
  using Mesh1Geometry = LinearEdgeGeometry;  ///< Non-mortar integration geometry.
  using Mesh2Geometry = LinearEdgeGeometry;  ///< Mortar projection geometry.
  /** @brief Native integration surface, expressed in mesh1 parent coordinates. */
  using IntegrationSurface = EdgeQuadratureInterval;
  RealT smoothing_fraction{ 0.1 };  ///< Smoothing length as a fraction of mesh1 edge length.

  /**
   * @brief Calculate projected source bounds and smooth them near its reference endpoints.
   *
   * A valid collapsed interval has zero integration measure.
   *
   * @param source Source endpoints indexed by endpoint, then physical component.
   * @param target Target endpoints projected along their edge normal.
   * @return Smoothed source bounds; degenerate edges return zero bounds and valid=false.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurface
  makeIntegrationSurface( MdSpan<const RealT, 2, 2> source, MdSpan<const RealT, 2, 2> target ) const
  {
    const auto interval = projectTargetOntoSource( source, target );
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
   * @brief Map integration-interval points onto both current elements and scale their reference weights.
   *
   * No overlap search is repeated. This also maps frozen mesh1 parent points while geometry changes.
   * @param mesh1 Non-mortar endpoint coordinates.
   * @param mesh2 Mortar endpoint coordinates.
   * @param reference Points on the source parent interval and their reference weights.
   * @param [out] output Parent points on both elements and physical integration weights.
   * @return Success with the initialized count, InsufficientCapacity with the full count, or InvalidGeometry with zero
   * points. Invalid geometry and insufficient capacity leave output untouched. Valid zero measure returns zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult mapToElements( Mesh1Geometry::Coordinates mesh1,
                                                                          Mesh2Geometry::Coordinates mesh2,
                                                                          ReferenceQuadratureView<1> reference,
                                                                          PairQuadratureView<1, 1> output ) const
  {
    const RealT dx1 = mesh1( 1, 0 ) - mesh1( 0, 0 ), dy1 = mesh1( 1, 1 ) - mesh1( 0, 1 );
    const RealT dx2 = mesh2( 1, 0 ) - mesh2( 0, 0 ), dy2 = mesh2( 1, 1 ) - mesh2( 0, 1 );
    const RealT length1 = std::sqrt( dx1 * dx1 + dy1 * dy1 );
    const RealT length2 = std::sqrt( dx2 * dx2 + dy2 * dy2 );
    if ( length1 < minimum_edge_length_ || length2 < minimum_edge_length_ ) {
      return { QuadratureStatus::InvalidGeometry, 0 };
    }
    const IndexT count = reference.numberOfPoints();
    if ( output.capacity() < count ) return { QuadratureStatus::InsufficientCapacity, count };
    for ( IndexT q = 0; q < count; ++q ) {
      const RealT parent1 = reference.parent_coordinates( q, 0 );
      const auto physical = Mesh1Geometry::position( mesh1, parent1 );
      const RealT fraction2 =
          ( ( physical[0] - mesh2( 0, 0 ) ) * dx2 + ( physical[1] - mesh2( 0, 1 ) ) * dy2 ) / ( dx2 * dx2 + dy2 * dy2 );
      output.mesh1_parent_coordinates( q, 0 ) = parent1;
      output.mesh2_parent_coordinates( q, 0 ) = 2.0 * fraction2 - 1.0;
      output.physical_weights[q] = reference.reference_weights[q] * ( 0.5 * length1 );
    }
    return { QuadratureStatus::Success, count };
  }

 private:
  inline static constexpr RealT minimum_edge_length_ = 1.0e-14;
  inline static constexpr RealT minimum_projection_determinant_ = 1.0e-12;

  /**
   * @brief Project both target endpoints along the target normal onto the infinite source line.
   * @param source Source endpoints indexed by endpoint, then physical component.
   * @param target Target endpoints indexed by endpoint, then physical component.
   * @return Increasing source parent-coordinate bounds, or invalid zero bounds when either edge is degenerate.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static IntegrationSurface projectTargetOntoSource(
      MdSpan<const RealT, 2, 2> source, MdSpan<const RealT, 2, 2> target )
  {
    const RealT target_x = target( 1, 0 ) - target( 0, 0 );
    const RealT target_y = target( 1, 1 ) - target( 0, 1 );
    const RealT target_length = std::sqrt( target_x * target_x + target_y * target_y );
    const RealT source_x = source( 1, 0 ) - source( 0, 0 );
    const RealT source_y = source( 1, 1 ) - source( 0, 1 );
    const RealT source_length_squared = source_x * source_x + source_y * source_y;
    if ( target_length < minimum_edge_length_ ||
         source_length_squared <= minimum_edge_length_ * minimum_edge_length_ ) {
      return { MdArray<RealT, 2>{ 0.0, 0.0 }, false };
    }

    const MdArray<RealT, 2> target_normal{ target_y / target_length, -target_x / target_length };
    const auto projected_point0 = projectTargetPointOntoSourceLine( source, target, 0, target_normal );
    const auto projected_point1 = projectTargetPointOntoSourceLine( source, target, 1, target_normal );
    const RealT source_fraction0 =
        ( ( projected_point0[0] - source( 0, 0 ) ) * source_x + ( projected_point0[1] - source( 0, 1 ) ) * source_y ) /
        source_length_squared;
    const RealT source_fraction1 =
        ( ( projected_point1[0] - source( 0, 0 ) ) * source_x + ( projected_point1[1] - source( 0, 1 ) ) * source_y ) /
        source_length_squared;
    const RealT coordinate0 = 2.0 * source_fraction0 - 1.0;
    const RealT coordinate1 = 2.0 * source_fraction1 - 1.0;
    return { MdArray<RealT, 2>{ coordinate0 < coordinate1 ? coordinate0 : coordinate1,
                                coordinate0 < coordinate1 ? coordinate1 : coordinate0 },
             true };
  }

  /**
   * @brief Project one target endpoint along the target normal onto the infinite source line.
   * @param source Source endpoints defining the destination line.
   * @param target Target endpoints containing the point to project.
   * @param target_endpoint Target endpoint index.
   * @param target_normal Direction from the target point to the source line.
   * @return Intersection point, or the target point when the two lines are nearly parallel.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static MdArray<RealT, 2> projectTargetPointOntoSourceLine(
      MdSpan<const RealT, 2, 2> source, MdSpan<const RealT, 2, 2> target, int target_endpoint,
      const MdArray<RealT, 2>& target_normal )
  {
    const RealT source_x = source( 1, 0 ) - source( 0, 0 );
    const RealT source_y = source( 1, 1 ) - source( 0, 1 );
    const RealT point_x = target( target_endpoint, 0 ) - source( 0, 0 );
    const RealT point_y = target( target_endpoint, 1 ) - source( 0, 1 );
    const RealT determinant = source_x * target_normal[1] - source_y * target_normal[0];
    if ( std::abs( determinant ) < minimum_projection_determinant_ ) {
      return MdArray<RealT, 2>{ target( target_endpoint, 0 ), target( target_endpoint, 1 ) };
    }

    const RealT source_fraction = ( point_x * target_normal[1] - point_y * target_normal[0] ) / determinant;
    return MdArray<RealT, 2>{ source( 0, 0 ) + source_fraction * source_x,
                              source( 0, 1 ) + source_fraction * source_y };
  }

  /**
   * @brief Apply quadratic smoothing near the source reference endpoints.
   *
   * Keep endpoint calls separate so Enzyme does not reuse loop-local reverse-mode tape entries.
   *
   * @param bound Source parent coordinate after clipping to the expanded support.
   * @param delta Smoothing length as a fraction of physical source length; zero leaves bound unchanged.
   * @return Smoothed parent coordinate.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static RealT smoothBound( RealT bound, RealT delta )
  {
    const RealT unit = 0.5 * ( bound + 1.0 );
    RealT smoothed = unit;
    if ( delta != 0.0 ) {
      if ( -delta <= unit && unit <= delta ) {
        smoothed = unit * unit / ( 4.0 * delta ) + 0.5 * unit + delta / 4.0;
      } else if ( 1.0 - delta <= unit && unit <= 1.0 + delta ) {
        const RealT b = -1.0 / ( 4.0 * delta );
        const RealT c = 0.5 + 1.0 / ( 2.0 * delta );
        const RealT d = 1.0 - delta + ( 1.0 - delta ) * ( 1.0 - delta ) / ( 4.0 * delta ) - 0.5 * ( 1.0 - delta ) -
                        ( 1.0 - delta ) / ( 2.0 * delta );
        smoothed = b * unit * unit + c * unit + d;
      }
    }
    return 2.0 * smoothed - 1.0;
  }
};

}  // namespace tribol

#endif
