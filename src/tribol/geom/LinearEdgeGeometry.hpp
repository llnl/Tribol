// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Local straight-edge geometry for interpolation, projection coordinates, normals, and integration measure.
 */
#ifndef SRC_TRIBOL_GEOM_LINEAREDGEGEOMETRY_HPP_
#define SRC_TRIBOL_GEOM_LINEAREDGEGEOMETRY_HPP_

#include <cmath>

#include "axom/primal/geometry/Point.hpp"
#include "axom/primal/geometry/Vector.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdSpan.hpp"

namespace tribol {

/**
 * @brief Own current straight-edge geometry for reuse within one element-pair calculation, independently of an FE
 * basis.
 *
 * Zero-length edges can be constructed and inspected. Normal and projection calculations require a nondegenerate edge;
 * callers choose their degeneracy tolerance before using those operations. No input coordinate views are retained.
 */
class LinearEdgeGeometry {
 public:
  static constexpr int vertex_count = 2;       ///< Number of vertices defining the straight edge.
  static constexpr int parent_dimension = 1;   ///< Components in normalized [-1, 1] parent coordinates.
  static constexpr int spatial_dimension = 2;  ///< Physical components per endpoint.
  using Point = axom::primal::Point<RealT, spatial_dimension>;    ///< Physical point on the edge.
  using Vector = axom::primal::Vector<RealT, spatial_dimension>;  ///< Directed displacement or normal.
  using Coords =
      MdSpan<const RealT, vertex_count, spatial_dimension>;  ///< Input coordinates [vertex][spatial component].

  /**
   * @brief Copy the edge vertices and calculate the edge vector and squared length once.
   *
   * The edge vector is (x1 - x0, y1 - y0), directed from the first input vertex to the second.
   * Its magnitude is the physical edge length; no normalization is applied.
   * @param coords Current vertices indexed by vertex, then physical component.
   * @pre Coordinates are finite.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE explicit LinearEdgeGeometry( Coords coords )
      : origin_( coords.data() ),
        end_( coords.data() + spatial_dimension ),
        edge_( origin_, end_ ),
        length_squared_( edge_.dot( edge_ ) )
  {
  }

  /**
   * @brief Access the first edge vertex.
   * @return Read-only reference valid while this geometry remains alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE const Point& origin() const& { return origin_; }
  /** @brief Prevent returning a vertex reference from a temporary geometry object. */
  const Point& origin() const&& = delete;

  /**
   * @brief Access the edge vector from the first vertex to the second.
   * @return Read-only reference to the unnormalized edge vector, valid while this geometry remains alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE const Vector& direction() const& { return edge_; }
  /** @brief Prevent returning an edge vector reference from a temporary geometry object. */
  const Vector& direction() const&& = delete;

  /** @brief Read the cached squared length. @return Squared physical edge length, including zero. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT squaredLength() const { return length_squared_; }
  /** @brief Calculate the physical length. @return Edge length, including zero. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT length() const { return std::sqrt( length_squared_ ); }
  /** @brief Calculate the [-1,1] reference Jacobian. @return Half the physical edge length. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT jacobian() const { return 0.5 * length(); }

  /**
   * @brief Map a parametric edge coordinate to a point in the current physical configuration.
   * @param xi Parametric coordinate; vertices are at -1 and 1. Values outside [-1,1] allow extrapolation.
   * @return Physical point, including for degenerate edge geometry.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Point position( RealT xi ) const
  {
    return Point::lerp( origin_, end_, 0.5 * ( xi + 1.0 ) );
  }

  /**
   * @brief Return the parametric coordinate of a physical point's orthogonal projection onto the edge's line.
   *
   * Projection is along the edge normal. The first vertex has xi = -1 and the second has xi = 1.
   * The projection lies on the edge segment for xi in [-1,1] and beyond a vertex otherwise.
   * @param point Physical point to project.
   * @pre The edge is nondegenerate.
   * @return Parametric coordinate xi of the projection.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT parentCoordinate( const Point& point ) const
  {
    return 2.0 * Vector( origin_, point ).dot( edge_ ) / length_squared_ - 1.0;
  }

  /**
   * @brief Construct the directed unit normal without imposing a length tolerance.
   * @pre The edge is nondegenerate.
   * @return Unit normal oriented (dy,-dx). No small-vector fallback is applied.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Vector normal() const
  {
    Vector result{ edge_[1], -edge_[0] };
    result /= length();
    return result;
  }

  /**
   * @brief Map a parametric edge coordinate using the supplied physical vertex coordinates.
   * @param coords Current edge vertex coordinates.
   * @param xi Parametric coordinate; vertices are at -1 and 1. Values outside [-1,1] allow extrapolation.
   * @return Physical point; no input views are retained.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static Point position( Coords coords, RealT xi )
  {
    return LinearEdgeGeometry( coords ).position( xi );
  }

 private:
  Point origin_;          ///< Owned first endpoint of the current edge.
  Point end_;             ///< Owned second endpoint, preserving interpolation at the input coordinates.
  Vector edge_;           ///< Owned displacement from the first endpoint to the second.
  RealT length_squared_;  ///< Cached squared physical edge length.
};

}  // namespace tribol
#endif
