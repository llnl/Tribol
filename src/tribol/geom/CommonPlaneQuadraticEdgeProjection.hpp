// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Closed-form projection of common-plane subelement points onto quadratic edges. */
#ifndef SRC_TRIBOL_GEOM_COMMONPLANEQUADRATICEDGEPROJECTION_HPP_
#define SRC_TRIBOL_GEOM_COMMONPLANEQUADRATICEDGEPROJECTION_HPP_

#include <cmath>

#include "tribol/geom/LinearEdgeGeometry.hpp"
#include "tribol/geom/QuadraticEdgeGeometry.hpp"
#include "tribol/integ/SubdividedPairQuadratureRuleGenerator.hpp"

namespace tribol {

/**
 * @brief Project each paired subelement point onto both quadratic edges along the subelement pair's integration-surface
 * normal.
 *
 * A Common Plane subelement rule pairs two subelement points on a line parallel to the subelement pair's common-plane
 * normal and stores that normal with the point. This projection intersects the same line with each quadratic edge, so
 * the evaluated element points are separated exactly along the stored normal. A traction along that normal then
 * produces no net moment. The intersection solves a scalar quadratic equation in closed form.
 */
struct CommonPlaneQuadraticEdgeProjection {
  using Vector = QuadraticEdgeGeometry::Vector;  ///< Physical vector.
  using Point = QuadraticEdgeGeometry::Point;    ///< Physical point.

  /**
   * @brief Intersect the subelement pairing line with both quadratic edges.
   * @param point1 First subelement point (provides geometry1, subelement1, interval1, and coordinate on subelement1).
   * @param point2 Second subelement point (provides geometry2 and interval2; both subelement points lie on the same
   * pairing line).
   * @param normal Unit integration-surface normal along which the subelement rule paired the point.
   * @return Element coordinates of both intersections; invalid when either line misses its edge inside [-1, 1].
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ProjectedElementCoords
  mapToElements( const SubelementPoint<QuadraticEdgeGeometry>& point1,
                 const SubelementPoint<QuadraticEdgeGeometry>& point2, const Vector& normal ) const
  {
    const MdArray<RealT, 1> subelement_coords1{ point1.subelement_coord };
    const Point line_point = point1.subelement.position( subelement_coords1.view() );
    const auto intersection1 = intersect( point1.element, line_point, normal, point1.interval );
    const auto intersection2 = intersect( point2.element, line_point, normal, point2.interval );
    return { intersection1.parametric_coord, intersection2.parametric_coord,
             intersection1.valid && intersection2.valid };
  }

 private:
  /** @brief Element parametric coordinate of a line-edge intersection, with a validity flag. */
  struct Intersection {
    RealT parametric_coord = 0.0;  ///< Element parametric coordinate of the selected intersection.
    bool valid = false;            ///< Whether the intersection is finite and inside the element domain [-1, 1].
  };

  /**
   * @brief Solve cross(x(xi) - line_point, direction) = 0 for the root nearest the subelement interval.
   *
   * The equation is a xi^2 + b xi + c = 0. The roots are formed as c / q and q / a with q = -(b + sign(b) sqrt(d)) / 2,
   * which avoids cancellation and keeps the near root finite as the edge straightens and a approaches zero.
   * @param geometry Quadratic edge.
   * @param line_point Point on the pairing line.
   * @param direction Unit direction of the pairing line.
   * @param interval Element parametric interval of the subelement that produced the point.
   * @return Selected intersection; invalid unless a finite intersection exists inside the element domain [-1, 1].
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static Intersection intersect( const QuadraticEdgeGeometry& geometry,
                                                                         const Point& line_point,
                                                                         const Vector& direction,
                                                                         SubelementInterval interval )
  {
    const auto coefficients = geometry.monomialCoefficients();
    // The z component of the cross product of two in-plane vectors is the scalar two-dimensional cross product.
    constexpr int normal_component = 2;
    const RealT a = Vector::cross_product( coefficients.quadratic, direction )[normal_component];
    const RealT b = Vector::cross_product( coefficients.linear, direction )[normal_component];
    const RealT c = Vector::cross_product( coefficients.constant - Vector( line_point ), direction )[normal_component];
    const RealT discriminant = b * b - 4.0 * a * c;
    if ( !std::isfinite( discriminant ) || discriminant < 0.0 ) {
      return {};
    }
    if ( b == 0.0 && c == 0.0 ) {
      return { 0.0, a != 0.0 };
    }
    const RealT root_discriminant = std::sqrt( discriminant );
    const RealT q = -0.5 * ( b >= 0.0 ? b + root_discriminant : b - root_discriminant );
    if ( q == 0.0 ) {
      return {};
    }
    const RealT interval_center = 0.5 * ( interval.lower + interval.upper );
    RealT root = c / q;
    if ( a != 0.0 ) {
      const RealT far_root = q / a;
      if ( std::abs( far_root - interval_center ) < std::abs( root - interval_center ) ) {
        root = far_root;
      }
    }
    return { root, std::isfinite( root ) && root >= -1.0 && root <= 1.0 };
  }
};

}  // namespace tribol

#endif  // SRC_TRIBOL_GEOM_COMMONPLANEQUADRATICEDGEPROJECTION_HPP_
