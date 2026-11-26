// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Map quadrature on the common plane between two directed 3D surface elements.
 */
#ifndef SRC_TRIBOL_GEOM_COMMONPLANEFACEMAPPING_HPP_
#define SRC_TRIBOL_GEOM_COMMONPLANEFACEMAPPING_HPP_

#include <cmath>

#include "axom/core/utilities/Utilities.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/**
 * @brief Project two 3D surface elements onto a common plane, clip their overlap, and map quadrature points back along
 * the common-plane normal onto the true 3D element surfaces.
 *
 * Integration coordinates measure physical distance along orthonormal in-plane basis vectors (e1, e2). Reference
 * weights therefore already include the physical area measure on the common plane; no element Jacobian is applied
 * during reverse mapping.
 *
 * @tparam Geometry1Type First 3D surface element geometry (e.g., LinearQuadGeometry or LinearTriangleGeometry).
 * @tparam Geometry2Type Second 3D surface element geometry; defaults to Geometry1Type.
 */
template <typename Geometry1Type, typename Geometry2Type = Geometry1Type>
struct CommonPlaneFaceMapping {
  using Geometry1 = Geometry1Type;                      ///< First surface element geometry.
  using Geometry2 = Geometry2Type;                      ///< Second surface element geometry.
  using Point = typename Geometry1::Point;              ///< Physical point in 3D.
  using Vector = typename Geometry1::Vector;            ///< Physical vector in 3D.
  using IntegrationSurface = PolygonQuadratureSurface;  ///< Clipped overlap polygon in (e1, e2) coordinates.

  static_assert( Geometry1::spatial_dim == 3 && Geometry2::spatial_dim == 3 );
  static_assert( Geometry1::parametric_dim == 2 && Geometry2::parametric_dim == 2 );

  /** @brief Orthonormal frame of the common plane, local to the current element pair. */
  struct CommonPlane {
    Point origin{};   ///< Midpoint of the two element centroids.
    Vector normal{};  ///< Unit direction of normal2 minus normal1.
    Vector e1{};      ///< First unit in-plane basis vector orthogonal to normal.
    Vector e2{};      ///< Second unit in-plane basis vector completing right-handed frame (e1 x e2 = normal).
  };

  /**
   * @brief Compute the common-plane origin, unit normal, and orthonormal in-plane basis.
   * @param geometry1 First current surface element.
   * @param geometry2 Second current surface element.
   * @param normal1 Valid unit normal of geometry1.
   * @param normal2 Valid unit normal of geometry2 (opposing normal1).
   * @return Owning common-plane frame; no geometry is retained.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static CommonPlane computeCommonPlane( const Geometry1& geometry1,
                                                                                 const Geometry2& geometry2,
                                                                                 const Vector& normal1,
                                                                                 const Vector& normal2 )
  {
    Vector normal = normal2 - normal1;
    normal /= std::sqrt( normal.dot( normal ) );

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
    const Point origin = Point::lerp( geometry1.centroid(), geometry2.centroid(), 0.5 );
    return { origin, normal, e1, e2 };
  }

  /**
   * @brief Project both elements onto the common plane, intersect their convex projections, and clip crossing faces to
   * the interpenetrating half-plane.
   *
   * When the average-plane gap changes sign strictly across the projected overlap polygon, the polygon is clipped to
   * the interpenetrating half-plane g_avg(u, v) <= 0. When all overlap vertices share the same gap sign (all
   * penetrating or all separated), the full projected overlap polygon is retained so opening pairs can still be
   * evaluated by velocity-projection timestep voting.
   *
   * @param geometry1 First current surface element.
   * @param geometry2 Second current surface element.
   * @return Clipped polygon in (e1, e2) coordinates; invalid faces return valid=false and zero vertices. Non-opposing
   * or disjoint/touching faces return valid=true with zero vertices.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurface makeIntegrationSurface( const Geometry1& geometry1,
                                                                                     const Geometry2& geometry2 ) const
  {
    const auto n1 = geometry1.unitNormal();
    const auto n2 = geometry2.unitNormal();
    if ( !n1.valid || !n2.valid ) {
      return { {}, 0, false };
    }
    if ( n1.normal.dot( n2.normal ) >= 0.0 ) {
      return { {}, 0, true };
    }

    const CommonPlane plane = computeCommonPlane( geometry1, geometry2, n1.normal, n2.normal );
    const RealT char_len_sq =
        axom::utilities::max( geometry1.maxSquaredEdgeLength(), geometry2.maxSquaredEdgeLength() );
    if ( !std::isfinite( char_len_sq ) || char_len_sq <= 0.0 ) {
      return { {}, 0, false };
    }
    const RealT char_len = std::sqrt( char_len_sq );
    const RealT plane_tol = on_line_tolerance_ * char_len;
    const RealT dup_tol_sq = ( duplicate_vertex_tolerance_ * char_len ) * ( duplicate_vertex_tolerance_ * char_len );

    // Project both faces into common-plane (e1, e2) coordinates and ensure both polygons are CCW.
    Polygon2D poly1{};
    poly1.num_vertices = Geometry1::num_vertices;
    for ( IndexT v = 0; v < Geometry1::num_vertices; ++v ) {
      const Vector rel( plane.origin, geometry1.vertex( v ) );
      poly1.vertices( v, 0 ) = rel.dot( plane.e1 );
      poly1.vertices( v, 1 ) = rel.dot( plane.e2 );
    }
    ensureCounterClockwise( poly1 );

    Polygon2D poly2{};
    poly2.num_vertices = Geometry2::num_vertices;
    for ( IndexT v = 0; v < Geometry2::num_vertices; ++v ) {
      const Vector rel( plane.origin, geometry2.vertex( v ) );
      poly2.vertices( v, 0 ) = rel.dot( plane.e1 );
      poly2.vertices( v, 1 ) = rel.dot( plane.e2 );
    }
    ensureCounterClockwise( poly2 );

    // Clip poly1 against each CCW edge of poly2 using Sutherland-Hodgman.
    Polygon2D overlap = poly1;
    for ( IndexT edge = 0; edge < poly2.num_vertices; ++edge ) {
      const IndexT next_edge = ( edge + 1 < poly2.num_vertices ) ? ( edge + 1 ) : 0;
      const RealT ax = poly2.vertices( edge, 0 );
      const RealT ay = poly2.vertices( edge, 1 );
      const RealT bx = poly2.vertices( next_edge, 0 );
      const RealT by = poly2.vertices( next_edge, 1 );
      const RealT ex = bx - ax;
      const RealT ey = by - ay;
      const RealT edge_len_sq = ex * ex + ey * ey;
      if ( edge_len_sq <= dup_tol_sq ) {
        return { {}, 0, true };
      }
      const RealT inv_edge_len = RealT{ 1.0 } / std::sqrt( edge_len_sq );
      // Outward normal of CCW edge A->B is (ey, -ex) / |AB|; inside is f(u, v) <= 0.
      const RealT nx = ey * inv_edge_len;
      const RealT ny = -ex * inv_edge_len;
      const RealT offset = -( nx * ax + ny * ay );

      overlap = clipAgainstHalfPlane( overlap, nx, ny, offset, plane_tol, dup_tol_sq );
      if ( overlap.num_vertices < 3 ) {
        return { {}, 0, true };
      }
    }

    // Check affine average-plane gap g_avg(u, v) = s1(u, v) - s2(u, v) across the overlap vertices.
    const RealT denom1 = plane.normal.dot( n1.normal );
    const RealT denom2 = plane.normal.dot( n2.normal );
    if ( std::abs( denom1 ) <= minimum_normal_alignment_ || std::abs( denom2 ) <= minimum_normal_alignment_ ) {
      return { {}, 0, false };
    }

    const Vector c1_rel( plane.origin, geometry1.centroid() );
    const Vector c2_rel( plane.origin, geometry2.centroid() );
    const RealT g0 = c1_rel.dot( n1.normal ) / denom1 - c2_rel.dot( n2.normal ) / denom2;
    const RealT gu = -( plane.e1.dot( n1.normal ) / denom1 - plane.e1.dot( n2.normal ) / denom2 );
    const RealT gv = -( plane.e2.dot( n1.normal ) / denom1 - plane.e2.dot( n2.normal ) / denom2 );

    RealT min_gap = g0 + gu * overlap.vertices( 0, 0 ) + gv * overlap.vertices( 0, 1 );
    RealT max_gap = min_gap;
    for ( IndexT v = 1; v < overlap.num_vertices; ++v ) {
      const RealT g = g0 + gu * overlap.vertices( v, 0 ) + gv * overlap.vertices( v, 1 );
      min_gap = axom::utilities::min( min_gap, g );
      max_gap = axom::utilities::max( max_gap, g );
    }

    // Clip to g_avg(u, v) <= 0 only when the average planes strictly cross inside the overlap polygon.
    if ( min_gap < -plane_tol && max_gap > plane_tol ) {
      const RealT grad_norm = std::sqrt( gu * gu + gv * gv );
      if ( grad_norm > 0.0 ) {
        const RealT inv_grad = RealT{ 1.0 } / grad_norm;
        overlap = clipAgainstHalfPlane( overlap, gu * inv_grad, gv * inv_grad, g0 * inv_grad, plane_tol, dup_tol_sq );
        if ( overlap.num_vertices < 3 ) {
          return { {}, 0, true };
        }
      }
    }

    const RealT overlap_area = signedPolygonArea( overlap );
    const RealT min_area = minimum_area_fraction_ * char_len_sq;
    if ( !std::isfinite( overlap_area ) || overlap_area <= min_area ) {
      return { {}, 0, true };
    }

    IntegrationSurface surface{};
    surface.valid = true;
    surface.num_vertices = overlap.num_vertices;
    for ( IndexT v = 0; v < overlap.num_vertices; ++v ) {
      surface.vertices( v, 0 ) = overlap.vertices( v, 0 );
      surface.vertices( v, 1 ) = overlap.vertices( v, 1 );
    }
    return surface;
  }

  /**
   * @brief Project common-plane quadrature points onto both 3D surface elements along the common-plane normal.
   *
   * @tparam Capacity Point capacity of the returned rule.
   * @param geometry1 First current surface element.
   * @param geometry2 Second current surface element.
   * @param reference Common-plane (e1, e2) coordinates and physical area weights.
   * @return Parametric coordinates on both faces, physical area weights, and the common-plane normal at every point.
   */
  template <IndexT Capacity>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureRule<2, 2, Capacity> mapToElements(
      const Geometry1& geometry1, const Geometry2& geometry2, IntegrationSurfaceRuleView<2> reference ) const
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
    const IndexT num_points = reference.numberOfPoints();
    if ( Capacity < num_points ) {
      output.metadata = { QuadratureStatus::InsufficientCapacity, num_points };
      return output;
    }
    if ( num_points == 0 ) {
      output.metadata = { QuadratureStatus::Success, 0 };
      return output;
    }

    const CommonPlane plane = computeCommonPlane( geometry1, geometry2, n1.normal, n2.normal );
    for ( IndexT q = 0; q < num_points; ++q ) {
      const RealT u = reference.coords( q, 0 );
      const RealT v = reference.coords( q, 1 );
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
      output.weights[q] = reference.weights[q];
      for ( IndexT d = 0; d < 3; ++d ) {
        output.integration_surface_normals( q, d ) = plane.normal[d];
      }
    }
    output.metadata = { QuadratureStatus::Success, num_points };
    return output;
  }

 private:
  static constexpr IndexT max_vertices_ = PolygonQuadratureSurface::max_num_vertices;
  static constexpr RealT on_line_tolerance_ = 1.0e-13;  ///< Relative on-line band for half-plane clipping.
  static constexpr RealT duplicate_vertex_tolerance_ =
      1.0e-12;                                                 ///< Relative tolerance for merging coincident vertices.
  static constexpr RealT minimum_area_fraction_ = 1.0e-14;     ///< Relative minimum overlap polygon area.
  static constexpr RealT minimum_normal_alignment_ = 1.0e-12;  ///< Minimum |n . n_k| for average-plane gap solve.

  /** @brief Fixed-capacity 2D polygon scratch buffer for Sutherland-Hodgman clipping. */
  struct Polygon2D {
    MdArray<RealT, max_vertices_, 2> vertices{};  ///< 2D vertices in (e1, e2) coordinates.
    IndexT num_vertices{ 0 };                     ///< Active vertex count.
  };

  /**
   * @brief Compute the signed area of a 2D polygon using the shoelace formula.
   * @param poly 2D polygon.
   * @return Positive area for CCW orientation, negative for CW.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static RealT signedPolygonArea( const Polygon2D& poly )
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
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void ensureCounterClockwise( Polygon2D& poly )
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
   * @param dup_tol_sq Squared distance threshold for deduplication.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void appendVertex( Polygon2D& poly, RealT x, RealT y,
                                                                    RealT dup_tol_sq )
  {
    if ( poly.num_vertices > 0 ) {
      const RealT dx = x - poly.vertices( poly.num_vertices - 1, 0 );
      const RealT dy = y - poly.vertices( poly.num_vertices - 1, 1 );
      if ( dx * dx + dy * dy <= dup_tol_sq ) {
        return;
      }
    }
    if ( poly.num_vertices < max_vertices_ ) {
      poly.vertices( poly.num_vertices, 0 ) = x;
      poly.vertices( poly.num_vertices, 1 ) = y;
      ++poly.num_vertices;
    }
  }

  /**
   * @brief Clip a 2D polygon against the half-plane nx * u + ny * v + offset <= 0.
   *
   * Vertices within [-plane_tol, +plane_tol] are classified as on-the-line (inside) so nearly-parallel edges resting
   * on the clipping line do not trigger 0/0 intersections, and intersections are only emitted on strict sign changes.
   *
   * @param input Input 2D polygon.
   * @param nx First component of the unit outward normal of the half-plane.
   * @param ny Second component of the unit outward normal of the half-plane.
   * @param offset Half-plane constant offset.
   * @param plane_tol Scale-relative on-line band tolerance.
   * @param dup_tol_sq Squared distance threshold for deduplicating consecutive vertices.
   * @return Clipped 2D polygon.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static Polygon2D clipAgainstHalfPlane( const Polygon2D& input, RealT nx,
                                                                                 RealT ny, RealT offset,
                                                                                 RealT plane_tol, RealT dup_tol_sq )
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

      const bool a_strictly_out = fa > plane_tol;
      const bool b_strictly_out = fb > plane_tol;
      const bool a_strictly_in = fa < -plane_tol;
      const bool b_strictly_in = fb < -plane_tol;

      if ( !a_strictly_out && !b_strictly_out ) {
        // Both inside or on the line: emit B.
        appendVertex( output, bx, by, dup_tol_sq );
      } else if ( a_strictly_in && b_strictly_out ) {
        // Strictly leaving: emit intersection only.
        const RealT t = axom::utilities::clampVal( fa / ( fa - fb ), RealT{ 0.0 }, RealT{ 1.0 } );
        appendVertex( output, ax + t * ( bx - ax ), ay + t * ( by - ay ), dup_tol_sq );
      } else if ( a_strictly_out && b_strictly_in ) {
        // Strictly entering: emit intersection then B.
        const RealT t = axom::utilities::clampVal( fa / ( fa - fb ), RealT{ 0.0 }, RealT{ 1.0 } );
        appendVertex( output, ax + t * ( bx - ax ), ay + t * ( by - ay ), dup_tol_sq );
        appendVertex( output, bx, by, dup_tol_sq );
      } else if ( a_strictly_out && !b_strictly_out ) {
        // Entering onto the line (|fb| <= plane_tol): emit B only.
        appendVertex( output, bx, by, dup_tol_sq );
      }
      // If |fa| <= plane_tol and b_strictly_out, A was already emitted on the previous edge; emit nothing.
    }

    // Wrap-around deduplication between last and first vertex.
    while ( output.num_vertices > 1 ) {
      const RealT dx = output.vertices( output.num_vertices - 1, 0 ) - output.vertices( 0, 0 );
      const RealT dy = output.vertices( output.num_vertices - 1, 1 ) - output.vertices( 0, 1 );
      if ( dx * dx + dy * dy <= dup_tol_sq ) {
        --output.num_vertices;
      } else {
        break;
      }
    }
    return output;
  }
};

}  // namespace tribol

#endif  // SRC_TRIBOL_GEOM_COMMONPLANEFACEMAPPING_HPP_
