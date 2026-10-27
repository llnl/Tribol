// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Gauss-Legendre rules on source segments with configurable interval projection.
 */

#ifndef SRC_TRIBOL_INTEG_EDGEQUADRATURERULE_HPP_
#define SRC_TRIBOL_INTEG_EDGEQUADRATURERULE_HPP_

#include <cmath>
#include "tribol/integ/Quadrature.hpp"
#include "tribol/geom/EdgeProjection.hpp"

namespace tribol {

/**
 * @brief Generates Gauss-Legendre quadrature on a straight segment defined by two endpoints in two dimensions
 *
 * The supplied interval determines where to place the Gauss points; the default is the whole reference edge.
 * A configured Projection can select bounds from a source/target pair. The default NoEdgeProjection requires only the
 * source segment and uses its whole interval. Generation returns source-surface data without target coordinates.
 * NumPoints counts points per base rule, not output capacity. This rule produces either NumPoints points or none.
 *
 * @tparam NumPoints Points per base rule: one, two, or three.
 * @tparam Projection Value type that selects source bounds from the element pair.
 */
template <int NumPoints, typename Projection = NoEdgeProjection>
struct EdgeQuadratureRule {
  static_assert( NumPoints >= 1 && NumPoints <= 3, "Supported Gauss-Legendre point counts are 1, 2, and 3." );
  static constexpr int endpoint_count = 2;                    ///< Endpoints defining one straight segment.
  static constexpr int parent_dimension = 1;                  ///< Parent components per point.
  static constexpr int spatial_dimension = 2;                 ///< Physical components per point.
  static constexpr int points_per_base_rule = NumPoints;      ///< Points in one nonempty interval.
  static constexpr int polynomial_order = 2 * NumPoints - 1;  ///< Exactly integrated reference polynomial degree.

  /**
   * @brief Use the projection type's default configuration.
   */
  TRIBOL_DEFAULT_HOST_DEVICE EdgeQuadratureRule() = default;
  /**
   * @brief Retain a value copy of the selected projection.
   *
   * @param projection Interval-selection configuration.
   */
  TRIBOL_HOST_DEVICE explicit EdgeQuadratureRule( Projection projection ) : projection_( projection ) {}

  /**
   * @brief Borrow the rule's projection configuration.
   *
   * @return Projection reference valid while this rule remains alive.
   */
  TRIBOL_HOST_DEVICE const Projection& projection() const& { return projection_; }
  /**
   * @brief Prevent borrowing projection configuration from a temporary rule.
   */
  const Projection& projection() const&& = delete;

  /**
   * @brief Read-only endpoint coordinates indexed by endpoint, then physical component.
   *
   * @tparam Value Coordinate and weight scalar type.
   */
  template <typename Value>
  using Coordinates = MdSpan<const Value, endpoint_count, spatial_dimension>;
  /**
   * @brief Fixed-capacity source parent coordinates and reference weights.
   *
   * @tparam Value Coordinate and weight scalar type.
   */
  template <typename Value>
  using ReferenceQuadrature = ReferenceQuadratureData<Value, parent_dimension, NumPoints>;
  /**
   * @brief Fixed-capacity source parent points, physical points, and physical weights.
   *
   * @tparam Value Coordinate and weight scalar type.
   */
  template <typename Value>
  using QuadratureData = SurfaceQuadratureData<Value, parent_dimension, spatial_dimension, NumPoints>;

  /**
   * @brief Return the output capacity required for one nonempty interval.
   *
   * @return NumPoints.
   */
  TRIBOL_HOST_DEVICE constexpr IndexT pointCapacity() const { return NumPoints; }

  /**
   * @brief Generate source quadrature into caller-owned storage.
   *
   * No target coordinates or mesh data are retained.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param surface Two source endpoints, each with two physical components.
   * @param interval Chosen source parent bounds and geometry validity.
   * @param [out] output Writable source-point and physical-weight arrays.
   * @return Completion status and full required count; see QuadratureResult.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult
  generate( Coordinates<Value> surface, EdgeQuadratureInterval<Value> interval,
            SurfaceQuadratureView<Value, parent_dimension, spatial_dimension> output ) const
  {
    const auto reference = generateReferenceQuadrature( interval );
    if ( reference.result.status != QuadratureStatus::Success ) return { QuadratureStatus::InvalidGeometry, 0 };
    return mapReferenceQuadrature( surface, reference.getView(), output );
  }

  /**
   * @brief Generate source quadrature in fixed-capacity local storage.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param surface Source endpoint coordinates.
   * @param interval Chosen source parent bounds and geometry validity.
   * @return Owned points and physical weights, with completion status.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureData<Value> generate( Coordinates<Value> surface,
                                                                          EdgeQuadratureInterval<Value> interval ) const
  {
    QuadratureData<Value> output{};
    output.result = generate( surface, interval, output.getOutputView() );
    return output;
  }

  /**
   * @brief Use the retained projection to select bounds and generate source quadrature.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param source Source endpoint coordinates.
   * @param target Target endpoint coordinates used to select source bounds.
   * @param [out] output Writable source-point and physical-weight arrays.
   * @return Completion status and full required count; see QuadratureResult.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult
  generate( Coordinates<Value> source, Coordinates<Value> target,
            SurfaceQuadratureView<Value, parent_dimension, spatial_dimension> output ) const
  {
    return generate( source, projection_.template integrationInterval<Value>( source, target ), output );
  }

  /**
   * @brief Generate an owning source rule on the selected projected interval.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param source Source endpoint coordinates.
   * @param target Target endpoint coordinates used to select source bounds.
   * @return Owned source quadrature, with completion status.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureData<Value> generate( Coordinates<Value> source,
                                                                          Coordinates<Value> target ) const
  {
    QuadratureData<Value> output{};
    output.result = generate( source, target, output.getOutputView() );
    return output;
  }

  /**
   * @brief Generate an unprojected rule using only the source segment.
   *
   * Available only when Projection::uses_target is false.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param source Source endpoint coordinates.
   * @return Owned whole-segment quadrature, with completion status.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureData<Value> generate( Coordinates<Value> source ) const
    requires( !Projection::uses_target )
  {
    return generate( source, projection_.template integrationInterval<Value>( source ) );
  }

  /**
   * @brief Generate parent points and reference weights on the selected projected interval.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param source Source endpoint coordinates.
   * @param target Target endpoint coordinates used to select source bounds.
   * @return Owned reference quadrature; invalid geometry reports zero points and InvalidGeometry.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ReferenceQuadrature<Value> generateReferenceQuadrature(
      Coordinates<Value> source, Coordinates<Value> target ) const
  {
    return generateReferenceQuadrature( projection_.template integrationInterval<Value>( source, target ) );
  }

  /**
   * @brief Map reference points onto the current source and apply its Jacobian once.
   *
   * A degenerate source returns zero points. Failures write no output. Frozen reference data still acquire the current
   * physical Jacobian.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param surface Current source endpoint coordinates.
   * @param reference Borrowed source parent points and reference weights.
   * @param [out] output Writable source-point and physical-weight arrays.
   * @return Full point count and Success, InsufficientCapacity, or InvalidGeometry.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult
  mapReferenceQuadrature( Coordinates<Value> surface, ReferenceQuadratureView<const Value, parent_dimension> reference,
                          SurfaceQuadratureView<Value, parent_dimension, spatial_dimension> output ) const
  {
    const Value jac = jacobian( surface );
    constexpr double minimum_edge_length = 1.0e-14;
    if ( 2.0 * jac < minimum_edge_length ) return { QuadratureStatus::InvalidGeometry, 0 };
    const IndexT count = reference.numberOfPoints();
    if ( output.capacity() < count ) return { QuadratureStatus::InsufficientCapacity, count };
    for ( IndexT q = 0; q < count; ++q ) {
      const Value xi = reference.parent_coordinates( q, 0 );
      output.parent_coordinates( q, 0 ) = xi;
      for ( int d = 0; d < spatial_dimension; ++d ) {
        output.physical_coordinates( q, d ) = 0.5 * ( ( 1.0 - xi ) * surface( 0, d ) + ( 1.0 + xi ) * surface( 1, d ) );
      }
      output.physical_weights[q] = reference.reference_weights[q] * jac;
    }
    return { QuadratureStatus::Success, count };
  }

  /**
   * @brief Map borrowed reference quadrature into owning source storage.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param surface Current source endpoint coordinates.
   * @param reference Borrowed source parent points and reference weights.
   * @return Owned source quadrature, with completion status.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureData<Value> mapReferenceQuadrature(
      Coordinates<Value> surface, ReferenceQuadratureView<const Value, parent_dimension> reference ) const
  {
    QuadratureData<Value> output{};
    output.result = mapReferenceQuadrature( surface, reference, output.getOutputView() );
    return output;
  }

  /**
   * @brief Generate parent points and reference weights without physical mapping.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param interval Source parent bounds; the default is the valid whole interval [-1, 1].
   * @return Owned reference quadrature; invalid geometry reports zero points and InvalidGeometry.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ReferenceQuadrature<Value> generateReferenceQuadrature(
      EdgeQuadratureInterval<Value> interval = {} ) const
  {
    if ( !interval.valid ) {
      ReferenceQuadrature<Value> invalid{};
      invalid.result.status = QuadratureStatus::InvalidGeometry;
      return invalid;
    }
    return generateReferenceInterval<Value>( interval.bounds.view() );
  }

  /**
   * @brief Map the fixed Gauss rule onto supplied parent bounds.
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param bounds Lower and upper normalized source coordinates.
   * @return NumPoints points for increasing bounds; Success with zero points for collapsed or reversed bounds.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static ReferenceQuadrature<Value> generateReferenceInterval(
      MdSpan<const Value, 2> bounds )
  {
    ReferenceQuadrature<Value> result{};
    if ( bounds[1] <= bounds[0] ) return result;
    const Value scale = 0.5 * ( bounds[1] - bounds[0] );
    const Value center = 0.5 * ( bounds[1] + bounds[0] );
    for ( int q = 0; q < NumPoints; ++q ) {
      result.parent_coordinates( q, 0 ) = scale * gaussPoint<Value>( q ) + center;
      result.reference_weights[q] = scale * gaussWeight<Value>( q );
    }
    result.result.required_point_count = NumPoints;
    return result;
  }

  /**
   * @brief Calculate the physical Jacobian for the normalized source interval [-1, 1].
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param surface Source endpoint coordinates.
   * @return Half the physical segment length.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static Value jacobian( Coordinates<Value> surface )
  {
    const Value dx = surface( 1, 0 ) - surface( 0, 0 );
    const Value dy = surface( 1, 1 ) - surface( 0, 1 );
    return 0.5 * std::sqrt( dx * dx + dy * dy );
  }

 private:
  Projection projection_{};  ///< Owned interval-selection configuration.

  /**
   * @brief Select a Gauss-Legendre parent coordinate on [-1, 1].
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param q Point index in [0, NumPoints).
   * @return Parent coordinate for point q.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static Value gaussPoint( int q )
  {
    if constexpr ( NumPoints == 1 ) return Value{ 0.0 };
    if constexpr ( NumPoints == 2 ) return ( q == 0 ? -1.0 : 1.0 ) / std::sqrt( Value{ 3.0 } );
    if constexpr ( NumPoints == 3 ) return ( q - 1 ) * std::sqrt( Value{ 3.0 } / Value{ 5.0 } );
  }

  /**
   * @brief Select a Gauss-Legendre weight on [-1, 1].
   *
   * @tparam Value Coordinate and weight scalar type.
   * @param q Point index in [0, NumPoints).
   * @return Reference weight; the complete base rule has weight two.
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static Value gaussWeight( int q )
  {
    if constexpr ( NumPoints == 1 ) return Value{ 2.0 };
    if constexpr ( NumPoints == 2 ) return Value{ 1.0 };
    if constexpr ( NumPoints == 3 ) return q == 1 ? Value{ 8.0 } / Value{ 9.0 } : Value{ 5.0 } / Value{ 9.0 };
  }
};

}  // namespace tribol

#endif
