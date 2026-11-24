// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Closed-form projection of common-plane subelement points onto quadratic edges. */
#ifndef SRC_TRIBOL_GEOM_COMMONPLANEQUADRATICEDGEPROJECTION_HPP_
#define SRC_TRIBOL_GEOM_COMMONPLANEQUADRATICEDGEPROJECTION_HPP_

#include <cmath>

#include "tribol/geom/CommonPlaneEdgeMapping.hpp"
#include "tribol/geom/LinearEdgeGeometry.hpp"
#include "tribol/geom/QuadraticEdgeGeometry.hpp"
#include "tribol/integ/SubdividedPairQuadratureRuleGenerator.hpp"

namespace tribol {

/**
 * @brief Project each paired subelement point onto both quadratic edges along the subelement pair's common normal.
 *
 * CommonPlaneEdgeMapping pairs two subelement points on a line parallel to the subelement pair's common-plane normal.
 * This policy intersects the same line with each quadratic edge, so the evaluated element points are separated exactly
 * along that normal. A traction parallel to their separation then produces no net moment. The intersection solves a
 * scalar quadratic equation in closed form.
 */
struct CommonPlaneQuadraticEdgeProjection {
  using Vector = QuadraticEdgeGeometry::Vector;  ///< Physical vector.
  using Point = QuadraticEdgeGeometry::Point;    ///< Physical point.

  /**
   * @brief Intersect the subelement pairing line with both quadratic edges.
   * @param geometry1 First quadratic edge.
   * @param geometry2 Second quadratic edge.
   * @param subelement1 Linear subelement of geometry1 that produced the point.
   * @param subelement2 Linear subelement of geometry2 that produced the point.
   * @param interval1 Element parametric interval of subelement1, used to select the intersection.
   * @param interval2 Element parametric interval of subelement2, used to select the intersection.
   * @param subelement_coord1 Point's parametric coordinate on subelement1.
   * @param subelement_coord2 Point's parametric coordinate on subelement2, unused because both subelement points lie on
   * the same pairing line.
   * @return Element coordinates of both intersections; invalid when either line misses its edge inside [-1, 1].
   * @pre The subelements are nondegenerate and have opposing normals, as required for the linear rule to emit points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ProjectedElementCoords mapToElements(
      const QuadraticEdgeGeometry& geometry1, const QuadraticEdgeGeometry& geometry2,
      const LinearEdgeGeometry& subelement1, const LinearEdgeGeometry& subelement2, SubelementInterval interval1,
      SubelementInterval interval2, RealT subelement_coord1, RealT TRIBOL_UNUSED_PARAM( subelement_coord2 ) ) const
  {
    const auto normal = CommonPlaneEdgeMapping::computeCommonPlaneLine( subelement1, subelement2 ).normal;
    const MdArray<RealT, 1> subelement_coords1{ subelement_coord1 };
    const Point line_point = subelement1.position( subelement_coords1.view() );
    ProjectedElementCoords result;
    const bool valid1 = intersect( geometry1, line_point, normal, interval1, result.geom1_parametric_coord );
    const bool valid2 = intersect( geometry2, line_point, normal, interval2, result.geom2_parametric_coord );
    result.valid = valid1 && valid2;
    return result;
  }

 private:
  /**
   * @brief Solve cross(x(xi) - line_point, direction) = 0 for the root nearest the subelement interval.
   *
   * The equation is a xi^2 + b xi + c = 0. The roots are formed as c / q and q / a with q = -(b + sign(b) sqrt(d)) / 2,
   * which avoids cancellation and keeps the near root finite as the edge straightens and a approaches zero.
   * @param geometry Quadratic edge.
   * @param line_point Point on the pairing line.
   * @param direction Unit direction of the pairing line.
   * @param interval Element parametric interval of the subelement that produced the point.
   * @param[out] parametric_coord Element parametric coordinate of the selected intersection.
   * @return Whether a finite intersection exists inside the element domain [-1, 1].
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static bool intersect( const QuadraticEdgeGeometry& geometry,
                                                                 const Point& line_point, const Vector& direction,
                                                                 SubelementInterval interval, RealT& parametric_coord )
  {
    const auto coefficients = geometry.monomialCoefficients();
    // The z component of the cross product of two in-plane vectors is the scalar two-dimensional cross product.
    constexpr int normal_component = 2;
    const RealT a = Vector::cross_product( coefficients.quadratic, direction )[normal_component];
    const RealT b = Vector::cross_product( coefficients.linear, direction )[normal_component];
    const RealT c = Vector::cross_product( coefficients.constant - Vector( line_point ), direction )[normal_component];
    const RealT discriminant = b * b - 4.0 * a * c;
    if ( !std::isfinite( discriminant ) || discriminant < 0.0 ) {
      return false;
    }
    const RealT root_discriminant = std::sqrt( discriminant );
    const RealT q = -0.5 * ( b >= 0.0 ? b + root_discriminant : b - root_discriminant );
    if ( q == 0.0 ) {
      return false;
    }
    const RealT interval_center = 0.5 * ( interval.lower + interval.upper );
    RealT root = c / q;
    if ( a != 0.0 ) {
      const RealT far_root = q / a;
      if ( std::abs( far_root - interval_center ) < std::abs( root - interval_center ) ) {
        root = far_root;
      }
    }
    parametric_coord = root;
    return std::isfinite( root ) && root >= -1.0 && root <= 1.0;
  }
};

}  // namespace tribol

#endif  // SRC_TRIBOL_GEOM_COMMONPLANEQUADRATICEDGEPROJECTION_HPP_
