// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Area-centroid and fan-triangulated symmetric triangle Gauss quadrature generators on 2D convex polygons.
 */
#ifndef SRC_TRIBOL_INTEG_POLYGONQUADRATURERULEGENERATOR_HPP_
#define SRC_TRIBOL_INTEG_POLYGONQUADRATURERULEGENERATOR_HPP_

#include <cmath>

#include "tribol/common/Enzyme.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

namespace detail {

/**
 * @brief Compute the signed area and area-weighted centroid of a 2D polygon using the shoelace formula.
 * @param polygon Convex polygon in integration-surface coordinates.
 * @param[out] area Positive polygon area when valid; zero otherwise.
 * @param[out] centroid_u First coordinate of the area centroid.
 * @param[out] centroid_v Second coordinate of the area centroid.
 * @return True if polygon has at least 3 vertices and strictly positive finite area.
 */
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline bool computePolygonAreaAndCentroid(
    const PolygonQuadratureSurface& polygon, RealT& area, RealT& centroid_u, RealT& centroid_v )
{
  area = 0.0;
  centroid_u = 0.0;
  centroid_v = 0.0;
  if ( polygon.num_vertices < 3 ) {
    return false;
  }
  RealT twice_area = 0.0;
  RealT six_cx = 0.0;
  RealT six_cy = 0.0;
  for ( IndexT v = 0; v < polygon.num_vertices; ++v ) {
    const IndexT next_v = ( v + 1 < polygon.num_vertices ) ? ( v + 1 ) : 0;
    const RealT x0 = polygon.vertices( v, 0 );
    const RealT y0 = polygon.vertices( v, 1 );
    const RealT x1 = polygon.vertices( next_v, 0 );
    const RealT y1 = polygon.vertices( next_v, 1 );
    const RealT cross = x0 * y1 - x1 * y0;
    twice_area += cross;
    six_cx += ( x0 + x1 ) * cross;
    six_cy += ( y0 + y1 ) * cross;
  }
  if ( !std::isfinite( twice_area ) || twice_area <= 0.0 ) {
    return false;
  }
  area = RealT{ 0.5 } * twice_area;
  const RealT inv_three_twice_area = RealT{ 1.0 } / ( RealT{ 3.0 } * twice_area );
  centroid_u = six_cx * inv_three_twice_area;
  centroid_v = six_cy * inv_three_twice_area;
  return std::isfinite( centroid_u ) && std::isfinite( centroid_v );
}

}  // namespace detail

/**
 * @brief Generate a single-point quadrature rule at the area-weighted centroid of a 2D polygon.
 *
 * Exact for linear polynomials over the polygon and matches legacy 3D Common Plane single-point integration on flat
 * faces. On warped faces, a single point at the overlap centroid can miss localized penetration near corners.
 */
struct PolygonCentroidQuadratureRuleGenerator {
  static constexpr int coordinate_dim = 2;    ///< Number of coordinates per integration point.
  static constexpr int max_num_points = 1;    ///< One centroid point per nonempty polygon.
  static constexpr int polynomial_order = 1;  ///< Exact through degree 1.

  /** @brief Quadrature point and area weight in integration-surface coordinates. */
  using IntegrationSurfaceRule = tribol::IntegrationSurfaceRule<coordinate_dim, max_num_points>;

  /**
   * @brief Generate the single area-centroid quadrature point on polygon.
   * @param polygon Clipped overlap polygon in integration-surface coordinates.
   * @return Integration-surface rule; invalid geometry returns InvalidGeometry, empty polygons return zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurfaceRule
  generate( const PolygonQuadratureSurface& polygon = {} ) const
  {
    IntegrationSurfaceRule rule{};
    if ( !polygon.valid ) {
      rule.metadata.status = QuadratureStatus::InvalidGeometry;
      return rule;
    }
    RealT area = 0.0;
    RealT centroid_u = 0.0;
    RealT centroid_v = 0.0;
    if ( !detail::computePolygonAreaAndCentroid( polygon, area, centroid_u, centroid_v ) ) {
      return rule;
    }
    rule.coords( 0, 0 ) = centroid_u;
    rule.coords( 0, 1 ) = centroid_v;
    rule.weights[0] = area;
    rule.metadata.num_required_points = 1;
    return rule;
  }
};

/**
 * @brief Generate positive-weight symmetric triangle Gauss rules over a centroid fan triangulation of a 2D polygon.
 *
 * Supported point counts per fan triangle:
 * - 3 points: degree 2 exact,
 * - 6 points: degree 4 exact,
 * - 7 points: degree 5 exact.
 * All supported rules have strictly positive weights.
 *
 * @tparam NumPointsPerTriangle Symmetric triangle quadrature points per fan triangle (3, 6, or 7).
 */
template <int NumPointsPerTriangle>
struct PolygonQuadratureRuleGenerator {
  static_assert( NumPointsPerTriangle == 3 || NumPointsPerTriangle == 6 || NumPointsPerTriangle == 7 );
  static constexpr int coordinate_dim = 2;  ///< Number of coordinates per integration point.
  /// Maximum number of points across all fan triangles of a clipped polygon.
  static constexpr int max_num_points = PolygonQuadratureSurface::max_num_vertices * NumPointsPerTriangle;
  /// Maximum polynomial degree integrated exactly in integration-surface coordinates.
  static constexpr int polynomial_order = NumPointsPerTriangle == 3 ? 2 : ( NumPointsPerTriangle == 6 ? 4 : 5 );

  /** @brief Quadrature points and physical area weights in integration-surface coordinates. */
  using IntegrationSurfaceRule = tribol::IntegrationSurfaceRule<coordinate_dim, max_num_points>;

  /**
   * @brief Fan-triangulate polygon from its area centroid and populate symmetric triangle Gauss points.
   *
   * Any degenerate fan triangle with nonpositive signed area is skipped so all emitted weights remain strictly
   * positive.
   *
   * @param polygon Clipped overlap polygon in integration-surface coordinates.
   * @return Integration-surface rule; invalid geometry returns InvalidGeometry, empty polygons return zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurfaceRule
  generate( const PolygonQuadratureSurface& polygon = {} ) const
  {
    IntegrationSurfaceRule rule{};
    if ( !polygon.valid ) {
      rule.metadata.status = QuadratureStatus::InvalidGeometry;
      return rule;
    }
    RealT total_area = 0.0;
    RealT cx = 0.0;
    RealT cy = 0.0;
    if ( !detail::computePolygonAreaAndCentroid( polygon, total_area, cx, cy ) ) {
      return rule;
    }

    IndexT out_point = 0;
    for ( IndexT v = 0; v < polygon.num_vertices; ++v ) {
      const IndexT next_v = ( v + 1 < polygon.num_vertices ) ? ( v + 1 ) : 0;
      const RealT x0 = polygon.vertices( v, 0 );
      const RealT y0 = polygon.vertices( v, 1 );
      const RealT x1 = polygon.vertices( next_v, 0 );
      const RealT y1 = polygon.vertices( next_v, 1 );

      const RealT tri_area = RealT{ 0.5 } * ( ( x0 - cx ) * ( y1 - cy ) - ( x1 - cx ) * ( y0 - cy ) );
      if ( !std::isfinite( tri_area ) || tri_area <= 0.0 ) {
        continue;
      }

      for ( int q = 0; q < NumPointsPerTriangle; ++q ) {
        const RealT b1 = barycentricCoord1( q );
        const RealT b2 = barycentricCoord2( q );
        const RealT b0 = RealT{ 1.0 } - b1 - b2;
        rule.coords( out_point, 0 ) = b0 * cx + b1 * x0 + b2 * x1;
        rule.coords( out_point, 1 ) = b0 * cy + b1 * y0 + b2 * y1;
        rule.weights[out_point] = tri_area * referenceWeight( q );
        ++out_point;
      }
    }
    rule.metadata.num_required_points = out_point;
    return rule;
  }

 private:
  /** @brief Return the first barycentric coordinate for triangle quadrature point q. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static RealT barycentricCoord1( int q )
  {
    if constexpr ( NumPointsPerTriangle == 3 ) {
      constexpr RealT a = RealT{ 1.0 } / RealT{ 6.0 };
      constexpr RealT b = RealT{ 2.0 } / RealT{ 3.0 };
      constexpr RealT coords[3] = { a, b, a };
      return coords[q];
    } else if constexpr ( NumPointsPerTriangle == 6 ) {
      constexpr RealT a1 = RealT{ 0.09157621350977074 };
      constexpr RealT b1 = RealT{ 0.8168475729804585 };
      constexpr RealT a2 = RealT{ 0.4459484909159649 };
      constexpr RealT b2 = RealT{ 0.10810301816807022 };
      constexpr RealT coords[6] = { a1, b1, a1, a2, b2, a2 };
      return coords[q];
    } else {
      constexpr RealT c = RealT{ 1.0 } / RealT{ 3.0 };
      constexpr RealT a1 = RealT{ 0.7974269853530873 };   // (9 + 2 * sqrt(15)) / 21
      constexpr RealT b1 = RealT{ 0.10128650732345633 };  // (6 - sqrt(15)) / 21
      constexpr RealT a2 = RealT{ 0.05971587178976982 };  // (9 - 2 * sqrt(15)) / 21
      constexpr RealT b2 = RealT{ 0.4701420641051151 };   // (6 + sqrt(15)) / 21
      constexpr RealT coords[7] = { c, a1, b1, b1, a2, b2, b2 };
      return coords[q];
    }
  }

  /** @brief Return the second barycentric coordinate for triangle quadrature point q. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static RealT barycentricCoord2( int q )
  {
    if constexpr ( NumPointsPerTriangle == 3 ) {
      constexpr RealT a = RealT{ 1.0 } / RealT{ 6.0 };
      constexpr RealT b = RealT{ 2.0 } / RealT{ 3.0 };
      constexpr RealT coords[3] = { a, a, b };
      return coords[q];
    } else if constexpr ( NumPointsPerTriangle == 6 ) {
      constexpr RealT a1 = RealT{ 0.09157621350977074 };
      constexpr RealT b1 = RealT{ 0.8168475729804585 };
      constexpr RealT a2 = RealT{ 0.4459484909159649 };
      constexpr RealT b2 = RealT{ 0.10810301816807022 };
      constexpr RealT coords[6] = { a1, a1, b1, a2, a2, b2 };
      return coords[q];
    } else {
      constexpr RealT c = RealT{ 1.0 } / RealT{ 3.0 };
      constexpr RealT a1 = RealT{ 0.7974269853530873 };
      constexpr RealT b1 = RealT{ 0.10128650732345633 };
      constexpr RealT a2 = RealT{ 0.05971587178976982 };
      constexpr RealT b2 = RealT{ 0.4701420641051151 };
      constexpr RealT coords[7] = { c, b1, a1, b1, b2, a2, b2 };
      return coords[q];
    }
  }

  /** @brief Return the normalized reference weight (summing to 1.0) for triangle quadrature point q. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static RealT referenceWeight( int q )
  {
    if constexpr ( NumPointsPerTriangle == 3 ) {
      return RealT{ 1.0 } / RealT{ 3.0 };
    } else if constexpr ( NumPointsPerTriangle == 6 ) {
      constexpr RealT w1 = RealT{ 0.10995174365532187 };
      constexpr RealT w2 = RealT{ 0.22338158967801146 };
      return q < 3 ? w1 : w2;
    } else {
      constexpr RealT w0 = RealT{ 0.225 };                // 9 / 40
      constexpr RealT w1 = RealT{ 0.12593918054482717 };  // (155 - sqrt(15)) / 1200
      constexpr RealT w2 = RealT{ 0.13239415278850616 };  // (155 + sqrt(15)) / 1200
      return q == 0 ? w0 : ( q < 4 ? w1 : w2 );
    }
  }
};

}  // namespace tribol

#endif  // SRC_TRIBOL_INTEG_POLYGONQUADRATURERULEGENERATOR_HPP_
