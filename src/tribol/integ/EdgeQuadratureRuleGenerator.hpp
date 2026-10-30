// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Gauss-Legendre quadrature points and weights on a supplied interval.
 *
 * Maps the standard rule on [-1,1] onto the supplied interval, scaling both points and weights.
 * The default interval is [-1,1].
 */

#ifndef SRC_TRIBOL_INTEG_EDGEQUADRATURERULEGENERATOR_HPP_
#define SRC_TRIBOL_INTEG_EDGEQUADRATURERULEGENERATOR_HPP_

#include <cmath>
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/**
 * @brief Generate Gauss-Legendre rules on a one-dimensional interval without element geometry.
 * @tparam NumPoints Points per nonempty generated rule: one, two, or three.
 */
template <int NumPoints>
struct EdgeQuadratureRuleGenerator {
  static_assert( NumPoints >= 1 && NumPoints <= 3 );
  static constexpr int parent_dimension = 1;    ///< Components on the integration interval.
  static constexpr int max_points = NumPoints;  ///< Maximum number of points in a successfully generated rule.
  /// Maximum polynomial degree integrated exactly in integration-surface coordinates.
  static constexpr int polynomial_order = 2 * NumPoints - 1;

  /** @brief Quadrature points and weights expressed in the supplied interval's coordinates. */
  using IntegrationSurfaceRule = tribol::IntegrationSurfaceRule<parent_dimension, NumPoints>;

  /**
   * @brief Generate quadrature points and weights on the supplied interval.
   * @param interval Integration bounds; invalid geometry returns InvalidGeometry with zero points.
   * @return Integration-surface rule; collapsed or reversed valid bounds give Success with zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurfaceRule generate( EdgeQuadratureInterval interval = {} ) const
  {
    if ( !interval.valid ) {
      IntegrationSurfaceRule rule{};
      rule.result.status = QuadratureStatus::InvalidGeometry;
      return rule;
    }
    return generate( interval.bounds.view() );
  }

  /**
   * @brief Map the standard Gauss-Legendre rule on [-1,1] onto the supplied interval.
   *
   * @param bounds Lower and upper integration-interval coordinates, stored at indices 0 and 1.
   * @return Success with NumPoints points if bounds[1] > bounds[0]; otherwise Success with zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static IntegrationSurfaceRule generate( MdSpan<const RealT, 2> bounds )
  {
    IntegrationSurfaceRule rule{};
    if ( bounds[1] <= bounds[0] ) return rule;
    const RealT scale = 0.5 * ( bounds[1] - bounds[0] );
    const RealT center = 0.5 * ( bounds[1] + bounds[0] );
    for ( int q = 0; q < NumPoints; ++q ) {
      rule.coords( q, 0 ) = scale * gaussPoint( q ) + center;
      rule.weights[q] = scale * gaussWeight( q );
    }
    rule.result.required_point_count = NumPoints;
    return rule;
  }

 private:
  /**
   * @brief Select a Gauss-Legendre parent coordinate on [-1, 1].
   *
   * @param q Point index in [0, NumPoints).
   * @return Parent coordinate for point q.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static RealT gaussPoint( int q )
  {
    if constexpr ( NumPoints == 1 ) return RealT{ 0.0 };
    if constexpr ( NumPoints == 2 ) return ( q == 0 ? -1.0 : 1.0 ) / std::sqrt( RealT{ 3.0 } );
    if constexpr ( NumPoints == 3 ) return ( q - 1 ) * std::sqrt( RealT{ 3.0 } / RealT{ 5.0 } );
  }

  /**
   * @brief Select a Gauss-Legendre weight on [-1, 1].
   *
   * @param q Point index in [0, NumPoints).
   * @return Reference weight; the complete base rule has weight two.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static RealT gaussWeight( int q )
  {
    if constexpr ( NumPoints == 1 ) return RealT{ 2.0 };
    if constexpr ( NumPoints == 2 ) return RealT{ 1.0 };
    if constexpr ( NumPoints == 3 ) return q == 1 ? RealT{ 8.0 } / RealT{ 9.0 } : RealT{ 5.0 } / RealT{ 9.0 };
  }
};

}  // namespace tribol

#endif
