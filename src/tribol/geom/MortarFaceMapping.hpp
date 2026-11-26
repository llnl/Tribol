// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Integrate over overlap projected onto the non-mortar element's tangent plane. */
#ifndef SRC_TRIBOL_GEOM_MORTARFACEMAPPING_HPP_
#define SRC_TRIBOL_GEOM_MORTARFACEMAPPING_HPP_

#include "tribol/geom/ProjectedPolygon.hpp"

namespace tribol::detail {

/**
 * @brief Map quadrature on the full projected overlap onto both surface elements.
 * @tparam Geometry1Type Non-mortar element geometry.
 * @tparam Geometry2Type Mortar element geometry.
 */
template <typename Geometry1Type, typename Geometry2Type = Geometry1Type>
struct MortarFaceMapping {
  using Geometry1 = Geometry1Type;            ///< Non-mortar geometry.
  using Geometry2 = Geometry2Type;            ///< Mortar geometry.
  using Point = typename Geometry1::Point;    ///< Point in spatial coordinates.
  using Vector = typename Geometry1::Vector;  ///< Vector in spatial coordinates.

  RealT overlap_area_frac = 0.0;  ///< Minimum overlap relative to the smaller projected element area.

  /**
   * @brief Intersect both elements' projections in orthonormal integration-surface coordinates.
   * @param geometry1 Current non-mortar geometry.
   * @param geometry2 Current mortar geometry.
   * @return Overlap polygon, with zero vertices for an empty overlap or an invalid projection.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PolygonQuadratureSurface
  makeIntegrationSurface( const Geometry1& geometry1, const Geometry2& geometry2 ) const
  {
    const auto normal1 = geometry1.unitNormal();
    const auto normal2 = geometry2.unitNormal();
    if ( !normal1.valid || !normal2.valid ) {
      return { {}, 0, false };
    }
    if ( normal1.normal.dot( normal2.normal ) >= 0.0 ) {
      return { {}, 0, true };
    }
    const auto plane = makePlane( geometry1, normal1.normal );
    const RealT size_squared =
        axom::utilities::max( geometry1.maxSquaredEdgeLength(), geometry2.maxSquaredEdgeLength() );
    const RealT length = std::sqrt( size_squared );
    const RealT line_tolerance = projected_polygon_on_line_tolerance * length;
    const RealT vertex_tolerance = projected_polygon_duplicate_vertex_tolerance * length;
    const RealT minimum_area = projected_polygon_minimum_area_fraction * size_squared;

    detail::Polygon2D poly1{};
    poly1.num_vertices = Geometry1::num_vertices;
    for ( IndexT v = 0; v < Geometry1::num_vertices; ++v ) {
      const Vector rel( plane.origin, geometry1.vertex( v ) );
      poly1.vertices( v, 0 ) = rel.dot( plane.e1 );
      poly1.vertices( v, 1 ) = rel.dot( plane.e2 );
    }
    detail::ensureCounterClockwise( poly1 );

    detail::Polygon2D poly2{};
    poly2.num_vertices = Geometry2::num_vertices;
    for ( IndexT v = 0; v < Geometry2::num_vertices; ++v ) {
      const Vector rel( plane.origin, geometry2.vertex( v ) );
      poly2.vertices( v, 0 ) = rel.dot( plane.e1 );
      poly2.vertices( v, 1 ) = rel.dot( plane.e2 );
    }
    detail::ensureCounterClockwise( poly2 );

    const RealT area1 = signedPolygonArea( poly1 );
    const RealT area2 = signedPolygonArea( poly2 );
    if ( !isConvexProjection( poly1, minimum_area ) || !isConvexProjection( poly2, minimum_area ) ) {
      return { {}, 0, false };
    }

    const auto overlap =
        intersectProjectedPolygons( poly1, poly2, line_tolerance, vertex_tolerance * vertex_tolerance );
    const RealT overlap_area = signedPolygonArea( overlap );
    if ( overlap_area <= minimum_area || overlap_area < overlap_area_frac * axom::utilities::min( area1, area2 ) ) {
      return { {}, 0, true };
    }
    return { overlap.vertices, overlap.num_vertices, true };
  }

  /**
   * @brief Project integration-plane quadrature points onto both 3D surface elements along the integration-plane
   * normal.
   *
   * @tparam Capacity Point capacity of the returned rule.
   * @param geometry1 First current surface element.
   * @param geometry2 Second current surface element.
   * @param surface_rule Integration-plane (e1, e2) coordinates and physical area weights.
   * @return Parametric coordinates on both faces, physical area weights, and the integration-plane normal at every
   * point.
   */
  template <IndexT Capacity>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureRule<2, 2, Capacity> mapToElements(
      const Geometry1& geometry1, const Geometry2& geometry2, IntegrationSurfaceRuleView<2> surface_rule ) const
  {
    PairQuadratureRule<2, 2, Capacity> output{};
    const auto n1 = geometry1.unitNormal();
    const auto n2 = geometry2.unitNormal();
    if ( !n1.valid || !n2.valid ) {
      output.metadata = { QuadratureStatus::InvalidGeometry, 0 };
      return output;
    }
    if ( n1.normal.dot( n2.normal ) >= 0.0 ) {
      output.metadata = { QuadratureStatus::Success, 0 };
      return output;
    }
    const IndexT num_points = surface_rule.numberOfPoints();
    if ( Capacity < num_points ) {
      output.metadata = { QuadratureStatus::InsufficientCapacity, num_points };
      return output;
    }
    if ( num_points == 0 ) {
      output.metadata = { QuadratureStatus::Success, 0 };
      return output;
    }

    const auto plane = makePlane( geometry1, n1.normal );
    for ( IndexT q = 0; q < num_points; ++q ) {
      const RealT u = surface_rule.coords( q, 0 );
      const RealT v = surface_rule.coords( q, 1 );
      Point plane_point;
      for ( IndexT d = 0; d < 3; ++d ) {
        plane_point[d] = plane.origin[d] + u * plane.e1[d] + v * plane.e2[d];
      }

      const auto proj1 = geometry1.parametricCoordinatesAlongDirection( plane_point, plane.e1, plane.e2 );
      const auto proj2 = geometry2.parametricCoordinatesAlongDirection( plane_point, plane.e1, plane.e2 );
      if ( !proj1.valid || !proj2.valid ) {
        output.metadata = { QuadratureStatus::InvalidGeometry, 0 };
        return output;
      }

      output.geom1_parametric_coords( q, 0 ) = proj1.parametric_coords[0];
      output.geom1_parametric_coords( q, 1 ) = proj1.parametric_coords[1];
      output.geom2_parametric_coords( q, 0 ) = proj2.parametric_coords[0];
      output.geom2_parametric_coords( q, 1 ) = proj2.parametric_coords[1];
      output.weights[q] = surface_rule.weights[q];
      for ( IndexT d = 0; d < 3; ++d ) {
        output.integration_surface_normals( q, d ) = plane.normal[d];
      }
    }
    output.metadata = { QuadratureStatus::Success, num_points };
    return output;
  }

 private:
  /** @brief Orthonormal frame for the current integration surface. */
  struct Plane {
    Point origin;   ///< Non-mortar element centroid.
    Vector normal;  ///< Non-mortar element unit normal.
    Vector e1;      ///< First in-plane direction.
    Vector e2;      ///< Second in-plane direction.
  };

  /**
   * @brief Form the tangent plane at the non-mortar centroid.
   * @param geometry1 Current non-mortar geometry.
   * @param normal Non-mortar unit normal.
   * @return Plane for projecting both elements.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static Plane makePlane( const Geometry1& geometry1, const Vector& normal )
  {
    const auto basis = makePlaneBasis( normal );
    return { geometry1.centroid(), normal, basis.e1, basis.e2 };
  }

  /**
   * @brief Check that each corner of a projected element makes a counterclockwise turn.
   * @param polygon Projected element with counterclockwise vertex order.
   * @param minimum_area Smallest usable area in integration-surface coordinates.
   * @return True if all turns are finite and nondegenerate.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static bool isConvexProjection( const Polygon2D& polygon, RealT minimum_area )
  {
    for ( IndexT vertex = 0; vertex < polygon.num_vertices; ++vertex ) {
      const IndexT next = ( vertex + 1 ) % polygon.num_vertices;
      const IndexT previous = ( vertex + polygon.num_vertices - 1 ) % polygon.num_vertices;
      const RealT ax = polygon.vertices( vertex, 0 ) - polygon.vertices( previous, 0 );
      const RealT ay = polygon.vertices( vertex, 1 ) - polygon.vertices( previous, 1 );
      const RealT bx = polygon.vertices( next, 0 ) - polygon.vertices( vertex, 0 );
      const RealT by = polygon.vertices( next, 1 ) - polygon.vertices( vertex, 1 );
      const RealT twice_area = ax * by - ay * bx;
      if ( !std::isfinite( twice_area ) || twice_area <= 2.0 * minimum_area ) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace tribol::detail

#endif  // SRC_TRIBOL_GEOM_MORTARFACEMAPPING_HPP_
