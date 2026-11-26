// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Operations on convex polygons in integration-surface coordinates. */
#ifndef SRC_TRIBOL_GEOM_PROJECTEDPOLYGON_HPP_
#define SRC_TRIBOL_GEOM_PROJECTEDPOLYGON_HPP_

#include <cmath>

#include "axom/core/utilities/Utilities.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol::detail {

/// Relative distance from a clipping line treated as lying on the line.
inline constexpr RealT projected_polygon_on_line_tolerance = 1.0e-13;
/// Relative distance below which two consecutive vertices are treated as coincident.
inline constexpr RealT projected_polygon_duplicate_vertex_tolerance = 1.0e-12;
/// Relative area below which a polygon has no usable integration area.
inline constexpr RealT projected_polygon_minimum_area_fraction = 1.0e-14;

/**
 * @brief Complete a right-handed orthonormal frame around a unit normal.
 * @tparam Vector Three-dimensional vector type.
 * @param normal Unit plane normal.
 * @return Two in-plane vectors whose cross product is normal.
 */
template <typename Vector>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto makePlaneBasis( const Vector& normal )
{
  struct Basis {
    Vector e1;  ///< First in-plane direction.
    Vector e2;  ///< Second in-plane direction.
  };
  // Cross normal with the coordinate axis least aligned with it to form a smooth, nondegenerate in-plane basis.
  const RealT ax = std::abs( normal[0] );
  const RealT ay = std::abs( normal[1] );
  const RealT az = std::abs( normal[2] );
  Vector ref_axis{ 1.0, 0.0, 0.0 };
  if ( ay <= ax && ay <= az ) {
    ref_axis = Vector{ 0.0, 1.0, 0.0 };
  } else if ( az <= ax && az <= ay ) {
    ref_axis = Vector{ 0.0, 0.0, 1.0 };
  }
  Vector e1 = Vector::cross_product( ref_axis, normal );
  e1 /= std::sqrt( e1.dot( e1 ) );
  const Vector e2 = Vector::cross_product( normal, e1 );
  return Basis{ e1, e2 };
}

/** @brief Fixed-capacity 2D polygon scratch buffer for Sutherland-Hodgman clipping. */
struct Polygon2D {
  MdArray<RealT, PolygonQuadratureSurface::max_num_vertices, 2> vertices{};  ///< 2D vertices in (e1, e2) coordinates.
  IndexT num_vertices{ 0 };                                                  ///< Active vertex count.
};

/**
 * @brief Compute the signed area of a 2D polygon using the shoelace formula.
 * @param poly 2D polygon.
 * @return Positive area for CCW orientation, negative for CW.
 */
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline RealT signedPolygonArea( const Polygon2D& poly )
{
  if ( poly.num_vertices < 3 ) {
    return 0.0;
  }
  RealT twice_area = 0.0;
  for ( IndexT v = 0; v < poly.num_vertices; ++v ) {
    const IndexT next_v = ( v + 1 < poly.num_vertices ) ? ( v + 1 ) : 0;
    twice_area +=
        poly.vertices( v, 0 ) * poly.vertices( next_v, 1 ) - poly.vertices( next_v, 0 ) * poly.vertices( v, 1 );
  }
  return RealT{ 0.5 } * twice_area;
}

/**
 * @brief Reverse vertex order in place if the 2D polygon has negative signed area.
 * @param poly 2D polygon to orient counter-clockwise.
 */
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void ensureCounterClockwise( Polygon2D& poly )
{
  if ( signedPolygonArea( poly ) < 0.0 ) {
    for ( IndexT i = 0; i < poly.num_vertices / 2; ++i ) {
      const IndexT j = poly.num_vertices - 1 - i;
      const RealT tx = poly.vertices( i, 0 );
      const RealT ty = poly.vertices( i, 1 );
      poly.vertices( i, 0 ) = poly.vertices( j, 0 );
      poly.vertices( i, 1 ) = poly.vertices( j, 1 );
      poly.vertices( j, 0 ) = tx;
      poly.vertices( j, 1 ) = ty;
    }
  }
}

/**
 * @brief Append a 2D vertex to poly unless it duplicates the previous vertex or exceeds capacity.
 * @param poly Output polygon.
 * @param x First coordinate.
 * @param y Second coordinate.
 * @param duplicate_vertex_tolerance_squared Squared distance threshold for deduplication.
 */
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void appendVertex( Polygon2D& poly, RealT x, RealT y,
                                                                  RealT duplicate_vertex_tolerance_squared )
{
  if ( poly.num_vertices > 0 ) {
    const RealT dx = x - poly.vertices( poly.num_vertices - 1, 0 );
    const RealT dy = y - poly.vertices( poly.num_vertices - 1, 1 );
    if ( dx * dx + dy * dy <= duplicate_vertex_tolerance_squared ) {
      return;
    }
  }
  if ( poly.num_vertices < PolygonQuadratureSurface::max_num_vertices ) {
    poly.vertices( poly.num_vertices, 0 ) = x;
    poly.vertices( poly.num_vertices, 1 ) = y;
    ++poly.num_vertices;
  }
}

/**
 * @brief Clip a 2D polygon against the half-plane nx * u + ny * v + offset <= 0.
 *
 * Vertices within [-line_tolerance, +line_tolerance] are classified as on-the-line (inside) so nearly-parallel edges
 * resting on the clipping line do not trigger 0/0 intersections, and intersections are only emitted on strict sign
 * changes.
 *
 * @param input Input 2D polygon.
 * @param nx First component of the unit outward normal of the half-plane.
 * @param ny Second component of the unit outward normal of the half-plane.
 * @param offset Half-plane constant offset.
 * @param line_tolerance Scale-relative on-line band tolerance.
 * @param duplicate_vertex_tolerance_squared Squared distance threshold for deduplicating consecutive vertices.
 * @return Clipped 2D polygon.
 */
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline Polygon2D clipAgainstHalfPlane(
    const Polygon2D& input, RealT nx, RealT ny, RealT offset, RealT line_tolerance,
    RealT duplicate_vertex_tolerance_squared )
{
  Polygon2D output{};
  if ( input.num_vertices < 3 ) {
    return output;
  }

  for ( IndexT i = 0; i < input.num_vertices; ++i ) {
    const IndexT j = ( i + 1 < input.num_vertices ) ? ( i + 1 ) : 0;
    const RealT ax = input.vertices( i, 0 );
    const RealT ay = input.vertices( i, 1 );
    const RealT bx = input.vertices( j, 0 );
    const RealT by = input.vertices( j, 1 );

    const RealT fa = nx * ax + ny * ay + offset;
    const RealT fb = nx * bx + ny * by + offset;

    const bool a_strictly_out = fa > line_tolerance;
    const bool b_strictly_out = fb > line_tolerance;
    const bool a_strictly_in = fa < -line_tolerance;
    const bool b_strictly_in = fb < -line_tolerance;

    if ( !a_strictly_out && !b_strictly_out ) {
      // Both inside or on the line: emit B.
      appendVertex( output, bx, by, duplicate_vertex_tolerance_squared );
    } else if ( a_strictly_in && b_strictly_out ) {
      // Strictly leaving: emit intersection only.
      const RealT t = axom::utilities::clampVal( fa / ( fa - fb ), RealT{ 0.0 }, RealT{ 1.0 } );
      appendVertex( output, ax + t * ( bx - ax ), ay + t * ( by - ay ), duplicate_vertex_tolerance_squared );
    } else if ( a_strictly_out && b_strictly_in ) {
      // Strictly entering: emit intersection then B.
      const RealT t = axom::utilities::clampVal( fa / ( fa - fb ), RealT{ 0.0 }, RealT{ 1.0 } );
      appendVertex( output, ax + t * ( bx - ax ), ay + t * ( by - ay ), duplicate_vertex_tolerance_squared );
      appendVertex( output, bx, by, duplicate_vertex_tolerance_squared );
    } else if ( a_strictly_out && !b_strictly_out ) {
      // Entering onto the line (|fb| <= line_tolerance): emit B only.
      appendVertex( output, bx, by, duplicate_vertex_tolerance_squared );
    }
    // If |fa| <= line_tolerance and b_strictly_out, A was already emitted on the previous edge; emit nothing.
  }

  // Wrap-around deduplication between last and first vertex.
  while ( output.num_vertices > 1 ) {
    const RealT dx = output.vertices( output.num_vertices - 1, 0 ) - output.vertices( 0, 0 );
    const RealT dy = output.vertices( output.num_vertices - 1, 1 ) - output.vertices( 0, 1 );
    if ( dx * dx + dy * dy <= duplicate_vertex_tolerance_squared ) {
      --output.num_vertices;
    } else {
      break;
    }
  }
  return output;
}

/**
 * @brief Intersect two convex, counterclockwise polygons by clipping against each edge of the second polygon.
 * @param geometry1_polygon First projected polygon.
 * @param geometry2_polygon Second projected polygon.
 * @param line_tolerance Distance from a clipping line treated as lying on the line.
 * @param duplicate_vertex_tolerance_squared Squared distance below which two vertices are treated as coincident.
 * @return Intersection in the same coordinates, or an empty polygon for a degenerate edge or overlap.
 */
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline Polygon2D intersectProjectedPolygons(
    const Polygon2D& geometry1_polygon, const Polygon2D& geometry2_polygon, RealT line_tolerance,
    RealT duplicate_vertex_tolerance_squared )
{
  // Clip poly1 against each CCW edge of geometry2_polygon using Sutherland-Hodgman.
  Polygon2D overlap = geometry1_polygon;
  for ( IndexT edge = 0; edge < geometry2_polygon.num_vertices; ++edge ) {
    const IndexT next_edge = ( edge + 1 < geometry2_polygon.num_vertices ) ? ( edge + 1 ) : 0;
    const RealT ax = geometry2_polygon.vertices( edge, 0 );
    const RealT ay = geometry2_polygon.vertices( edge, 1 );
    const RealT bx = geometry2_polygon.vertices( next_edge, 0 );
    const RealT by = geometry2_polygon.vertices( next_edge, 1 );
    const RealT ex = bx - ax;
    const RealT ey = by - ay;
    const RealT edge_len_sq = ex * ex + ey * ey;
    if ( edge_len_sq <= duplicate_vertex_tolerance_squared ) {
      return {};
    }
    const RealT inv_edge_len = RealT{ 1.0 } / std::sqrt( edge_len_sq );
    // Outward normal of CCW edge A->B is (ey, -ex) / |AB|; inside is f(u, v) <= 0.
    const RealT nx = ey * inv_edge_len;
    const RealT ny = -ex * inv_edge_len;
    const RealT offset = -( nx * ax + ny * ay );

    overlap = clipAgainstHalfPlane( overlap, nx, ny, offset, line_tolerance, duplicate_vertex_tolerance_squared );
    if ( overlap.num_vertices < 3 ) {
      return {};
    }
  }

  return overlap;
}

}  // namespace tribol::detail

#endif  // SRC_TRIBOL_GEOM_PROJECTEDPOLYGON_HPP_
