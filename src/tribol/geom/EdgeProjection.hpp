// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Source integration intervals selected from straight-edge projections.
 */

#ifndef SRC_TRIBOL_GEOM_EDGEPROJECTION_HPP_
#define SRC_TRIBOL_GEOM_EDGEPROJECTION_HPP_

#include <cmath>
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/search/InterfacePairChecks.hpp"

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

/** @brief Integrate the whole source segment without projecting an interval or generating target coordinates */
struct NoEdgeProjection {
  static constexpr bool uses_target = false;  ///< Source integration needs no target edge.

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
 * Both edges are straight segments in two dimensions. Source bounds use [-1, 1] parent coordinates.
 * The smoothing fraction is expressed as a fraction of source length. Only configuration is retained.
 */
struct SmoothedEdgeProjection {
  static constexpr bool uses_target = true;  ///< Interval selection requires a target edge.
  double smoothing_fraction{ 0.1 };          ///< Smoothing length as a fraction of source length.

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
