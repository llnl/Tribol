// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Current three-node linear Lagrange triangle geometry in three spatial dimensions.
 */
#ifndef SRC_TRIBOL_GEOM_LINEARTRIANGLEGEOMETRY_HPP_
#define SRC_TRIBOL_GEOM_LINEARTRIANGLEGEOMETRY_HPP_

#include <cmath>

#include "axom/core/utilities/Utilities.hpp"
#include "axom/primal/geometry/Point.hpp"
#include "axom/primal/geometry/Vector.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/BasisFunction.hpp"

namespace tribol {

/**
 * @brief Own a current linear triangle in three spatial dimensions for pair-local geometry and quadrature mapping.
 *
 * Node ordering matches BasisFunction<TRIANGLE, LINEAR_LAGRANGE> on the standard simplex with vertices (0, 0), (1, 0),
 * and (0, 1).
 */
class LinearTriangleGeometry {
 public:
  using Basis = BasisFunction<TRIANGLE, LINEAR_LAGRANGE>;       ///< Basis defining interpolation and node ordering.
  static constexpr int num_vertices = Basis::size_basis;        ///< Number of geometric vertices.
  static constexpr int num_nodes = Basis::size_basis;           ///< Number of finite-element nodes.
  static constexpr int parametric_dim = Basis::parametric_dim;  ///< Number of parametric coordinates.
  static constexpr int spatial_dim = Basis::spatial_dim;        ///< Number of physical coordinates.
  using Point = axom::primal::Point<RealT, spatial_dim>;        ///< Physical point on the triangle.
  using Vector = axom::primal::Vector<RealT, spatial_dim>;      ///< Physical vector in three dimensions.
  using Coords = MdSpan<const RealT, num_nodes, spatial_dim>;   ///< Input coordinates [node][spatial_dim].

  /** @brief Unit normal and area together with a validity flag. */
  struct UnitNormalResult {
    Vector normal{};     ///< Outward unit normal when valid; zero otherwise.
    RealT area{ 0.0 };   ///< Triangle physical area when valid; zero otherwise.
    bool valid = false;  ///< True when coordinates are finite and area exceeds the degeneracy threshold.
  };

  /** @brief Element-parametric coordinates recovered by projecting a point along a direction. */
  struct ParametricProjectionResult {
    MdArray<RealT, parametric_dim> parametric_coords{};  ///< Parametric coordinates (xi, eta) on the simplex.
    bool valid = false;                                  ///< True when the 2x2 projection system is invertible.
  };

  /**
   * @brief Copy the three current vertex coordinates.
   * @param coords Current vertex coordinates indexed [node][spatial_dim].
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE explicit LinearTriangleGeometry( Coords coords )
  {
    for ( IndexT node = 0; node < num_nodes; ++node ) {
      for ( IndexT d = 0; d < spatial_dim; ++d ) {
        coords_( node, d ) = coords( node, d );
      }
    }
  }

  /**
   * @brief Return one vertex as a physical Point.
   * @param vertex_index Vertex index in [0, num_vertices).
   * @return Physical vertex point.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Point vertex( IndexT vertex_index ) const
  {
    return Point{ coords_( vertex_index, 0 ), coords_( vertex_index, 1 ), coords_( vertex_index, 2 ) };
  }

  /**
   * @brief Return the geometric centroid (average of the three vertices).
   * @return Physical centroid point.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Point centroid() const
  {
    constexpr RealT third = RealT{ 1.0 } / RealT{ 3.0 };
    Point center;
    for ( IndexT d = 0; d < spatial_dim; ++d ) {
      center[d] = third * ( coords_( 0, d ) + coords_( 1, d ) + coords_( 2, d ) );
    }
    return center;
  }

  /**
   * @brief Evaluate the linear map at element-parametric coordinates on the standard simplex.
   * @param parametric_coords Parametric coordinates (xi, eta).
   * @return Current physical point.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Point position( MdSpan<const RealT, parametric_dim> parametric_coords ) const
  {
    MdArray<RealT, num_nodes> values{};
    Basis::eval( parametric_coords, values.view() );
    Point point;
    for ( IndexT d = 0; d < spatial_dim; ++d ) {
      point[d] = 0.0;
      for ( IndexT node = 0; node < num_nodes; ++node ) {
        point[d] += values[node] * coords_( node, d );
      }
    }
    return point;
  }

  /**
   * @brief Compute the maximum squared edge length of the triangle.
   * @return Largest squared edge length, or a nonfinite value if coordinates are nonfinite.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT maxSquaredEdgeLength() const
  {
    const Vector e01( vertex( 0 ), vertex( 1 ) );
    const Vector e12( vertex( 1 ), vertex( 2 ) );
    const Vector e20( vertex( 2 ), vertex( 0 ) );
    return axom::utilities::max( e01.squared_norm(), axom::utilities::max( e12.squared_norm(), e20.squared_norm() ) );
  }

  /**
   * @brief Compute the triangle's outward unit normal and physical area.
   * @return Unit normal, area, and validity flag.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE UnitNormalResult unitNormal() const
  {
    for ( IndexT node = 0; node < num_nodes; ++node ) {
      for ( IndexT d = 0; d < spatial_dim; ++d ) {
        if ( !std::isfinite( coords_( node, d ) ) ) {
          return {};
        }
      }
    }
    const Vector e01( vertex( 0 ), vertex( 1 ) );
    const Vector e02( vertex( 0 ), vertex( 2 ) );
    const Vector cross = Vector::cross_product( e01, e02 );
    const RealT norm_sq = cross.squared_norm();
    const RealT min_area_sq = minimum_twice_area_ * minimum_twice_area_;
    if ( !std::isfinite( norm_sq ) || norm_sq < min_area_sq ) {
      return {};
    }
    const RealT twice_area = std::sqrt( norm_sq );
    return { cross / twice_area, RealT{ 0.5 } * twice_area, true };
  }

  /**
   * @brief Recover element-parametric coordinates whose projection onto the (e1, e2) plane matches plane_point.
   *
   * Solves the exact 2x2 affine system (x(xi, eta) - plane_point) . e1 = 0 and (x(xi, eta) - plane_point) . e2 = 0,
   * pairing plane_point with the triangle point along the normal e1 x e2.
   *
   * @param plane_point Physical point on the common plane.
   * @param e1 First unit in-plane basis vector orthogonal to the projection direction.
   * @param e2 Second unit in-plane basis vector orthogonal to the projection direction and e1.
   * @return Parametric coordinates and validity flag.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ParametricProjectionResult
  parametricCoordinatesAlongDirection( const Point& plane_point, const Vector& e1, const Vector& e2 ) const
  {
    const Point p0 = vertex( 0 );
    const Vector d1( p0, vertex( 1 ) );
    const Vector d2( p0, vertex( 2 ) );
    const Vector rhs( p0, plane_point );

    const RealT a00 = d1.dot( e1 );
    const RealT a01 = d2.dot( e1 );
    const RealT a10 = d1.dot( e2 );
    const RealT a11 = d2.dot( e2 );
    const RealT det = a00 * a11 - a01 * a10;

    const RealT scale_sq = axom::utilities::max( maxSquaredEdgeLength(), RealT{ 1.0 } );
    const RealT min_det = minimum_determinant_scale_ * scale_sq;
    if ( !std::isfinite( det ) || std::abs( det ) <= min_det ) {
      return {};
    }

    const RealT r0 = rhs.dot( e1 );
    const RealT r1 = rhs.dot( e2 );
    const RealT inv_det = RealT{ 1.0 } / det;
    const RealT xi = ( a11 * r0 - a01 * r1 ) * inv_det;
    const RealT eta = ( -a10 * r0 + a00 * r1 ) * inv_det;
    if ( !std::isfinite( xi ) || !std::isfinite( eta ) ) {
      return {};
    }
    return { MdArray<RealT, parametric_dim>{ xi, eta }, true };
  }

 private:
  static constexpr RealT minimum_twice_area_ = 1.0e-28;         ///< Degenerate twice-area threshold.
  static constexpr RealT minimum_determinant_scale_ = 1.0e-12;  ///< Relative 2x2 projection invertibility threshold.

  MdArray<RealT, num_nodes, spatial_dim> coords_{};  ///< Owned current nodal coordinates.
};

}  // namespace tribol

#endif  // SRC_TRIBOL_GEOM_LINEARTRIANGLEGEOMETRY_HPP_
