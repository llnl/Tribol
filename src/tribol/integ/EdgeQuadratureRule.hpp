// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

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
 * Selecting or smoothing an interval and projecting generated points are separate operations. NumPoints counts points
 * per base rule, not output capacity. This rule produces either NumPoints points or none.
 */
template <int NumPoints>
struct EdgeQuadratureRule {
  static_assert( NumPoints >= 1 && NumPoints <= 3, "Supported Gauss-Legendre point counts are 1, 2, and 3." );
  static constexpr int endpoint_count = 2;
  static constexpr int parent_dimension = 1;
  static constexpr int spatial_dimension = 2;
  static constexpr int points_per_base_rule = NumPoints;
  static constexpr int polynomial_order = 2 * NumPoints - 1;

  TRIBOL_DEFAULT_HOST_DEVICE EdgeQuadratureRule() = default;

  template <typename Value>
  using Coordinates = MdSpan<const Value, endpoint_count, spatial_dimension>;
  template <typename Value>
  using ReferenceQuadrature = ReferenceQuadratureData<Value, parent_dimension, NumPoints>;
  template <typename Value>
  using QuadratureData = SurfaceQuadratureData<Value, parent_dimension, spatial_dimension, NumPoints>;

  TRIBOL_HOST_DEVICE constexpr IndexT pointCapacity() const { return NumPoints; }

  /** @brief Generate source quadrature without allocating memory or computing any target-surface coordinates */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult
  generate( Coordinates<Value> surface, EdgeQuadratureInterval<Value> interval,
            SurfaceQuadratureView<Value, parent_dimension, spatial_dimension> output ) const
  {
    const auto reference = generateReferenceQuadrature( interval );
    if ( reference.result.status != QuadratureStatus::Success ) return { QuadratureStatus::InvalidGeometry, 0 };
    return mapReferenceQuadrature( surface, reference.getView(), output );
  }

  /** @brief Generate into owning local storage, defaulting to the whole reference edge */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureData<Value> generate(
      Coordinates<Value> surface, EdgeQuadratureInterval<Value> interval = {} ) const
  {
    QuadratureData<Value> output{};
    output.result = generate( surface, interval, output.getOutputView() );
    return output;
  }

  /**
   * @brief Map reference quadrature to the current source edge, applying its physical Jacobian exactly once
   *
   * A degenerate source returns InvalidGeometry with zero points. Insufficient capacity reports the complete point
   * count without writing output. Reference coordinates and weights can remain frozen while the source edge changes.
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

  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureData<Value> mapReferenceQuadrature(
      Coordinates<Value> surface, ReferenceQuadratureView<const Value, parent_dimension> reference ) const
  {
    QuadratureData<Value> output{};
    output.result = mapReferenceQuadrature( surface, reference, output.getOutputView() );
    return output;
  }

  /** @brief Generate parent points and reference weights on an interval, without retaining surface geometry */
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

  /** @brief A collapsed or reversed interval is valid and contributes no points */
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

  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static Value jacobian( Coordinates<Value> surface )
  {
    const Value dx = surface( 1, 0 ) - surface( 0, 0 );
    const Value dy = surface( 1, 1 ) - surface( 0, 1 );
    return 0.5 * std::sqrt( dx * dx + dy * dy );
  }

 private:
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static Value gaussPoint( int q )
  {
    if constexpr ( NumPoints == 1 ) return Value{ 0.0 };
    if constexpr ( NumPoints == 2 ) return ( q == 0 ? -1.0 : 1.0 ) / std::sqrt( Value{ 3.0 } );
    if constexpr ( NumPoints == 3 ) return ( q - 1 ) * std::sqrt( Value{ 3.0 } / Value{ 5.0 } );
  }

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
