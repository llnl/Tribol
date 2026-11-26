// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Current four-node bilinear Lagrange quadrilateral geometry in three spatial dimensions.
 */
#ifndef SRC_TRIBOL_GEOM_LINEARQUADGEOMETRY_HPP_
#define SRC_TRIBOL_GEOM_LINEARQUADGEOMETRY_HPP_

#include <cmath>

#include "axom/core/utilities/Utilities.hpp"
#include "axom/primal/geometry/Point.hpp"
#include "axom/primal/geometry/Vector.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/BasisFunction.hpp"

namespace tribol {

/**
 * @brief Own a current bilinear quadrilateral in three spatial dimensions for pair-local geometry and quadrature
 * mapping.
 *
 * Node ordering matches BasisFunction<QUAD, LINEAR_LAGRANGE> on [-1, 1]^2: (-1, -1), (1, -1), (1, 1), (-1, 1).
 * Warped quadrilaterals retain their true bilinear surface when evaluating position() and
 * parametricCoordinatesAlongDirection().
 */
class LinearQuadGeometry {
 public:
  using Basis = BasisFunction<QUAD, LINEAR_LAGRANGE>;           ///< Basis defining interpolation and node ordering.
  static constexpr int num_vertices = Basis::size_basis;        ///< Number of geometric vertices.
  static constexpr int num_nodes = Basis::size_basis;           ///< Number of finite-element nodes.
  static constexpr int parametric_dim = Basis::parametric_dim;  ///< Number of parametric coordinates.
  static constexpr int spatial_dim = Basis::spatial_dim;        ///< Number of physical coordinates.
  using Point = axom::primal::Point<RealT, spatial_dim>;        ///< Physical point on the quadrilateral.
  using Vector = axom::primal::Vector<RealT, spatial_dim>;      ///< Physical vector in three dimensions.
  using Coords = MdSpan<const RealT, num_nodes, spatial_dim>;   ///< Input coordinates [node][spatial_dim].

  /** @brief Average unit normal and area estimate together with a validity flag. */
  struct UnitNormalResult {
    Vector normal{};     ///< Outward unit normal at parametric (0, 0) when valid; zero otherwise.
    RealT area{ 0.0 };   ///< Projected mean area 0.5 * |(p2 - p0) x (p3 - p1)| when valid; zero otherwise.
    bool valid = false;  ///< True when coordinates are finite and the normal is nondegenerate.
  };

  /** @brief Element-parametric coordinates recovered by projecting a point along a direction. */
  struct ParametricProjectionResult {
    MdArray<RealT, parametric_dim> parametric_coords{};  ///< Parametric coordinates (xi, eta) on [-1, 1]^2.
    bool valid = false;                                  ///< True when the 2x2 bilinear inverse map converged.
  };

  /**
   * @brief Copy the four current vertex coordinates.
   * @param coords Current vertex coordinates indexed [node][spatial_dim].
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE explicit LinearQuadGeometry( Coords coords )
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
   * @brief Return the quadrilateral center at parametric coordinates (0, 0) (average of the four vertices).
   * @return Physical center point.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Point centroid() const
  {
    Point center;
    for ( IndexT d = 0; d < spatial_dim; ++d ) {
      center[d] = RealT{ 0.25 } * ( coords_( 0, d ) + coords_( 1, d ) + coords_( 2, d ) + coords_( 3, d ) );
    }
    return center;
  }

  /**
   * @brief Evaluate the bilinear surface map on [-1, 1]^2.
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
   * @brief Compute the maximum squared edge length of the quadrilateral.
   * @return Largest squared edge length, or a nonfinite value if coordinates are nonfinite.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT maxSquaredEdgeLength() const
  {
    const Vector e01( vertex( 0 ), vertex( 1 ) );
    const Vector e12( vertex( 1 ), vertex( 2 ) );
    const Vector e23( vertex( 2 ), vertex( 3 ) );
    const Vector e30( vertex( 3 ), vertex( 0 ) );
    return axom::utilities::max( axom::utilities::max( e01.squared_norm(), e12.squared_norm() ),
                                 axom::utilities::max( e23.squared_norm(), e30.squared_norm() ) );
  }

  /**
   * @brief Compute the outward unit normal at parametric (0, 0) using the diagonal cross product (p2 - p0) x (p3 - p1).
   *
   * For a bilinear quad, 0.5 * (p2 - p0) x (p3 - p1) equals 4 * (dx/dxi x dx/deta) at (0, 0) and is symmetric across
   * all four vertices.
   *
   * @return Unit normal, mean area estimate, and validity flag.
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
    const Vector diag02( vertex( 0 ), vertex( 2 ) );
    const Vector diag13( vertex( 1 ), vertex( 3 ) );
    const Vector cross = Vector::cross_product( diag02, diag13 );
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
   * Solves (x(xi, eta) - plane_point) . e1 = 0 and (x(xi, eta) - plane_point) . e2 = 0 by Newton iteration on the true
   * bilinear surface, pairing plane_point with the surface point along e1 x e2.
   *
   * @param plane_point Physical point on the common plane.
   * @param e1 First unit in-plane basis vector orthogonal to the projection direction.
   * @param e2 Second unit in-plane basis vector orthogonal to the projection direction and e1.
   * @return Parametric coordinates and validity flag.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ParametricProjectionResult
  parametricCoordinatesAlongDirection( const Point& plane_point, const Vector& e1, const Vector& e2 ) const
  {
    const RealT scale_sq = axom::utilities::max( maxSquaredEdgeLength(), RealT{ 1.0 } );
    const RealT min_det = minimum_determinant_scale_ * scale_sq;
    const RealT residual_tol_sq = newton_tolerance_ * newton_tolerance_ * scale_sq;

    MdArray<RealT, parametric_dim> parametric_coords{ 0.0, 0.0 };
    bool converged = false;

    for ( int iter = 0; iter < max_newton_iterations_; ++iter ) {
      MdArray<RealT, num_nodes> values{};
      MdArray<RealT, num_nodes, parametric_dim> derivatives{};
      Basis::eval( parametric_coords.view(), values.view() );
      Basis::evalParametricDerivatives( parametric_coords.view(), derivatives.view() );

      Vector diff{};
      Vector dx_dxi{};
      Vector dx_deta{};
      for ( IndexT d = 0; d < spatial_dim; ++d ) {
        RealT pos_d = 0.0;
        RealT dxi_d = 0.0;
        RealT deta_d = 0.0;
        for ( IndexT node = 0; node < num_nodes; ++node ) {
          const RealT c = coords_( node, d );
          pos_d += values[node] * c;
          dxi_d += derivatives( node, 0 ) * c;
          deta_d += derivatives( node, 1 ) * c;
        }
        diff[d] = pos_d - plane_point[d];
        dx_dxi[d] = dxi_d;
        dx_deta[d] = deta_d;
      }

      const RealT f0 = diff.dot( e1 );
      const RealT f1 = diff.dot( e2 );

      const RealT j00 = dx_dxi.dot( e1 );
      const RealT j01 = dx_deta.dot( e1 );
      const RealT j10 = dx_dxi.dot( e2 );
      const RealT j11 = dx_deta.dot( e2 );
      const RealT det = j00 * j11 - j01 * j10;
      if ( !std::isfinite( det ) || std::abs( det ) <= min_det ) {
        return {};
      }

      const RealT inv_det = RealT{ 1.0 } / det;
      const RealT delta_xi = -( j11 * f0 - j01 * f1 ) * inv_det;
      const RealT delta_eta = -( -j10 * f0 + j00 * f1 ) * inv_det;
      parametric_coords[0] += delta_xi;
      parametric_coords[1] += delta_eta;

      if ( !std::isfinite( parametric_coords[0] ) || !std::isfinite( parametric_coords[1] ) ) {
        return {};
      }
      // Break only after applying the Newton step so the final iteration evaluates F and J at the converged root,
      // yielding the exact implicit derivative when differentiated with Enzyme (even for planar parallelograms).
      if ( f0 * f0 + f1 * f1 <= residual_tol_sq &&
           delta_xi * delta_xi + delta_eta * delta_eta <= newton_tolerance_ * newton_tolerance_ ) {
        converged = true;
        break;
      }
    }

    if ( !converged ) {
      // Check residual after final iteration in case the last step landed on the solution.
      const Point final_point = position( parametric_coords.view() );
      const Vector final_diff( plane_point, final_point );
      const RealT f0 = final_diff.dot( e1 );
      const RealT f1 = final_diff.dot( e2 );
      if ( !std::isfinite( f0 ) || !std::isfinite( f1 ) || f0 * f0 + f1 * f1 > residual_tol_sq ) {
        return {};
      }
    }

    return { parametric_coords, true };
  }

 private:
  static constexpr int max_newton_iterations_ = 12;             ///< Maximum Newton iterations for bilinear inversion.
  static constexpr RealT newton_tolerance_ = 1.0e-12;           ///< Relative residual and step convergence tolerance.
  static constexpr RealT minimum_twice_area_ = 1.0e-28;         ///< Degenerate twice-area threshold.
  static constexpr RealT minimum_determinant_scale_ = 1.0e-12;  ///< Relative 2x2 Jacobian invertibility threshold.

  MdArray<RealT, num_nodes, spatial_dim> coords_{};  ///< Owned current nodal coordinates.
};

}  // namespace tribol

#endif  // SRC_TRIBOL_GEOM_LINEARQUADGEOMETRY_HPP_
