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
#include "tribol/search/InterfacePairChecks.hpp"
#include "tribol/integ/Quadrature.hpp"

namespace tribol {

/**
 * @brief Geometry of a straight edge with two endpoints in two physical dimensions, independent of an FE basis.
 */
struct LinearEdgeGeometry {
  static constexpr int endpoint_count = 2;     ///< Endpoints defining the straight edge.
  static constexpr int parent_dimension = 1;   ///< Components in normalized [-1, 1] parent coordinates.
  static constexpr int spatial_dimension = 2;  ///< Physical components per endpoint.

  /** @brief Borrow endpoints by endpoint, then physical component. @tparam Value Coordinate scalar type. */
  template <typename Value>
  using Coordinates = MdSpan<const Value, endpoint_count, spatial_dimension>;

  /**
   * @brief Evaluate a physical point on the straight edge, allowing extrapolation.
   * @tparam Value Coordinate scalar type.
   * @param coordinates Edge endpoint coordinates.
   * @param parent Normalized parent coordinate.
   * @return Physical point at parent.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static MdArray<Value, spatial_dimension> position(
      Coordinates<Value> coordinates, Value parent )
  {
    MdArray<Value, spatial_dimension> result;
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
  /** @brief Native integration surface, expressed in mesh1 parent coordinates. @tparam Value Coordinate scalar type. */
  template <typename Value>
  using IntegrationSurface = EdgeQuadratureInterval<Value>;
  double smoothing_fraction{ 0.1 };  ///< Smoothing length as a fraction of mesh1 edge length.

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
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurface<Value> makeIntegrationSurface(
      MdSpan<const Value, 2, 2> source, MdSpan<const Value, 2, 2> target ) const
  {
    const auto interval = InterfacePairChecks::computeLinearEdgeProjectionBounds( source.data(), source.data() + 2,
                                                                                  target.data(), target.data() + 2 );
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
   * @brief Map integration-interval points onto both current elements and scale their reference weights.
   *
   * No overlap search is repeated. This also maps frozen mesh1 parent points while geometry changes.
   * @tparam Value Coordinate and weight scalar type.
   * @param mesh1 Non-mortar endpoint coordinates.
   * @param mesh2 Mortar endpoint coordinates.
   * @param reference Points on the source parent interval and their reference weights.
   * @param [out] output Parent points on both elements and physical integration weights.
   * @return Success with the initialized count, InsufficientCapacity with the full count, or InvalidGeometry with zero
   * points. Invalid geometry and insufficient capacity leave output untouched. Valid zero measure returns zero points.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult
  mapToElements( Mesh1Geometry::Coordinates<Value> mesh1, Mesh2Geometry::Coordinates<Value> mesh2,
                 ReferenceQuadratureView<const Value, 1> reference, PairQuadratureView<Value, 1, 1> output ) const
  {
    const Value dx1 = mesh1( 1, 0 ) - mesh1( 0, 0 ), dy1 = mesh1( 1, 1 ) - mesh1( 0, 1 );
    const Value dx2 = mesh2( 1, 0 ) - mesh2( 0, 0 ), dy2 = mesh2( 1, 1 ) - mesh2( 0, 1 );
    const Value length1 = std::sqrt( dx1 * dx1 + dy1 * dy1 );
    const Value length2 = std::sqrt( dx2 * dx2 + dy2 * dy2 );
    constexpr double minimum_edge_length = 1.0e-14;
    if ( length1 < minimum_edge_length || length2 < minimum_edge_length ) {
      return { QuadratureStatus::InvalidGeometry, 0 };
    }
    const IndexT count = reference.numberOfPoints();
    if ( output.capacity() < count ) return { QuadratureStatus::InsufficientCapacity, count };
    for ( IndexT q = 0; q < count; ++q ) {
      const Value parent1 = reference.parent_coordinates( q, 0 );
      const auto physical = Mesh1Geometry::position<Value>( mesh1, parent1 );
      const Value fraction2 =
          ( ( physical[0] - mesh2( 0, 0 ) ) * dx2 + ( physical[1] - mesh2( 0, 1 ) ) * dy2 ) / ( dx2 * dx2 + dy2 * dy2 );
      output.mesh1_parent_coordinates( q, 0 ) = parent1;
      output.mesh2_parent_coordinates( q, 0 ) = 2.0 * fraction2 - 1.0;
      output.physical_weights[q] = reference.reference_weights[q] * ( 0.5 * length1 );
    }
    return { QuadratureStatus::Success, count };
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
