// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Gauss-Legendre points and reference weights on a supplied interval.
 */

#ifndef SRC_TRIBOL_INTEG_EDGEQUADRATURERULE_HPP_
#define SRC_TRIBOL_INTEG_EDGEQUADRATURERULE_HPP_

#include <cmath>
#include "tribol/integ/Quadrature.hpp"

namespace tribol {

/**
 * @brief Place Gauss points and reference weights on a one-dimensional interval without element geometry.
 * @tparam NumPoints Points per nonempty base rule: one, two, or three.
 */
template <int NumPoints>
struct EdgeQuadratureRule {
  static_assert( NumPoints >= 1 && NumPoints <= 3 );
  static constexpr int parent_dimension = 1;                  ///< Components on the integration interval.
  static constexpr int points_per_base_rule = NumPoints;      ///< Points per nonempty interval.
  static constexpr int polynomial_order = 2 * NumPoints - 1;  ///< Exactly integrated reference degree.

  /** @brief Local reference points and weights. */
  using ReferenceQuadrature = ReferenceQuadratureData<parent_dimension, NumPoints>;

  /**
   * @brief Generate reference points on the supplied interval.
   * @param interval Integration bounds; invalid geometry returns InvalidGeometry with zero points.
   * @return Local reference quadrature; collapsed or reversed valid bounds give Success with zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ReferenceQuadrature generate( EdgeQuadratureInterval interval = {} ) const
  {
    if ( !interval.valid ) {
      ReferenceQuadrature result{};
      result.result.status = QuadratureStatus::InvalidGeometry;
      return result;
    }
    return generate( interval.bounds.view() );
  }

  /**
   * @brief Map the fixed Gauss rule onto supplied parent bounds.
   *
   * @param bounds Lower and upper normalized source coordinates.
   * @return NumPoints points for increasing bounds; Success with zero points for collapsed or reversed bounds.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static ReferenceQuadrature generate( MdSpan<const RealT, 2> bounds )
  {
    ReferenceQuadrature result{};
    if ( bounds[1] <= bounds[0] ) return result;
    const RealT scale = 0.5 * ( bounds[1] - bounds[0] );
    const RealT center = 0.5 * ( bounds[1] + bounds[0] );
    for ( int q = 0; q < NumPoints; ++q ) {
      result.parent_coordinates( q, 0 ) = scale * gaussPoint( q ) + center;
      result.reference_weights[q] = scale * gaussWeight( q );
    }
    result.result.required_point_count = NumPoints;
    return result;
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
