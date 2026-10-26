// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_SMOOTHPROJECTEDEDGEINTERVAL_HPP_
#define SRC_TRIBOL_INTEG_SMOOTHPROJECTEDEDGEINTERVAL_HPP_

#include "tribol/integ/EdgeQuadratureProjection.hpp"

namespace tribol {

/**
 * @brief Find a projected source-edge interval and smooth its bounds near the reference endpoints
 *
 * Both edges are straight segments in two dimensions. Bounds use normalized [-1, 1] source coordinates. Invalid
 * geometry returns valid=false and zero bounds. Valid intervals may collapse to zero measure. Quadrature generation
 * and projection of quadrature points onto target surfaces are separate operations.
 */
struct SmoothProjectedEdgeInterval {
  double smoothing_fraction{ 0.1 };  ///< Smoothing length as a fraction of the source edge length.

  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE EdgeQuadratureInterval<Value> operator()(
      MdSpan<const Value, 2, 2> source, MdSpan<const Value, 2, 2> other ) const
  {
    const auto interval = projectEdgeInterval<Value>( source, other );
    if ( !interval.valid ) return interval;
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
  // Keep each endpoint calculation separate so Enzyme does not reuse loop-local reverse-mode tape entries.
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
