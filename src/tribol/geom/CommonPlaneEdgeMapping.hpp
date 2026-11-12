// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Map quadrature on the common line between two directed straight edges.
 */
#ifndef SRC_TRIBOL_GEOM_COMMONPLANEEDGEMAPPING_HPP_
#define SRC_TRIBOL_GEOM_COMMONPLANEEDGEMAPPING_HPP_

#include "tribol/geom/LinearEdgeGeometry.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/**
 * @brief Project two edges' contact interval onto a common line and map points back along its normal.
 *
 * Integration coordinates measure physical distance along a unit tangent. Reference weights therefore already
 * include the physical line measure. No element Jacobian is applied during reverse mapping.
 */
struct CommonPlaneEdgeMapping {
  using FirstGeometry = LinearEdgeGeometry;           ///< First directed edge.
  using SecondGeometry = LinearEdgeGeometry;          ///< Second directed edge.
  using Point = LinearEdgeGeometry::Point;            ///< Physical point.
  using Vector = LinearEdgeGeometry::Vector;          ///< Physical displacement or unit direction.
  using IntegrationSurface = EdgeQuadratureInterval;  ///< Physical tangent-coordinate interval.

  /** @brief Geometry of the common line, local to the current element pair. */
  struct CommonPlaneLine {
    Point origin;    ///< Average of the two edge midpoints.
    Vector normal;   ///< Unit direction of normal2 minus normal1.
    Vector tangent;  ///< Unit direction along the integration line.
  };

  /**
   * @brief Compute the common line used for projection and contact tractions.
   * @param edge1 First current edge.
   * @param edge2 Second current edge.
   * @pre Both edges are nondegenerate and have opposing normals.
   * @return Owning origin and unit directions; no geometry is retained.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static CommonPlaneLine computeCommonPlaneLine( const FirstGeometry& edge1,
                                                                                         const SecondGeometry& edge2 )
  {
    auto normal = edge2.normal() - edge1.normal();
    normal /= std::sqrt( normal.dot( normal ) );
    return { Point::lerp( edge1.position( 0.0 ), edge2.position( 0.0 ), 0.5 ), normal,
             Vector{ normal[1], -normal[0] } };
  }

  /**
   * @brief Intersect the two edges' projections and keep the interpenetrating interval for crossing edges.
   * @param edge1 First current edge.
   * @param edge2 Second current edge.
   * @return Physical tangent bounds; invalid edges return valid=false and zero bounds. Nonopposing normals and
   * disjoint or touching intervals return valid collapsed bounds and contribute no points. Noncrossing edges retain
   * their full projected overlap, allowing the pressure law to apply a prescribed residual gap.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurface makeIntegrationSurface( const FirstGeometry& edge1,
                                                                                     const SecondGeometry& edge2 ) const
  {
    if ( !validEdges( edge1, edge2 ) ) {
      return { MdArray<RealT, 2>{ 0.0, 0.0 }, false };
    }
    if ( !opposingNormals( edge1, edge2 ) ) {
      return { MdArray<RealT, 2>{ 0.0, 0.0 }, true };
    }
    const auto line = computeCommonPlaneLine( edge1, edge2 );
    const RealT a0 = Vector( line.origin, edge1.origin() ).dot( line.tangent );
    const RealT a1 = Vector( line.origin, edge1.position( 1.0 ) ).dot( line.tangent );
    const RealT b0 = Vector( line.origin, edge2.origin() ).dot( line.tangent );
    const RealT b1 = Vector( line.origin, edge2.position( 1.0 ) ).dot( line.tangent );
    // With opposing normals, the tangent follows edge1 and opposes edge2, so a0 < a1 and b1 < b0.
    RealT lower = a0 > b1 ? a0 : b1;
    RealT upper = a1 < b0 ? a1 : b0;
    if ( upper <= lower ) {
      return { MdArray<RealT, 2>{ lower, lower }, true };
    }

    // The gap is affine along straight edges. Opposite endpoint signs locate an intersection inside both edges.
    // Clip at the geometric zero gap before applying any residual-gap offset in the pressure calculation.
    const auto point1_lower = edge1.position( 2.0 * ( lower - a0 ) / ( a1 - a0 ) - 1.0 );
    const auto point2_lower = edge2.position( 2.0 * ( lower - b0 ) / ( b1 - b0 ) - 1.0 );
    const auto point1_upper = edge1.position( 2.0 * ( upper - a0 ) / ( a1 - a0 ) - 1.0 );
    const auto point2_upper = edge2.position( 2.0 * ( upper - b0 ) / ( b1 - b0 ) - 1.0 );
    const RealT lower_gap = Vector( point2_lower, point1_lower ).dot( line.normal );
    const RealT upper_gap = Vector( point2_upper, point1_upper ).dot( line.normal );
    if ( lower_gap < 0.0 && upper_gap > 0.0 ) {
      upper = lower + ( upper - lower ) * lower_gap / ( lower_gap - upper_gap );
    } else if ( lower_gap > 0.0 && upper_gap < 0.0 ) {
      lower = lower + ( upper - lower ) * lower_gap / ( lower_gap - upper_gap );
    }
    return { MdArray<RealT, 2>{ lower, upper }, true };
  }

  /**
   * @brief Project common-line points onto both edges without another physical-measure factor.
   * @param edge1 First current edge.
   * @param edge2 Second current edge.
   * @param reference Physical tangent coordinates and line weights.
   * @param[out] output Parent coordinates and physical line weights.
   * @return Success with the initialized count; nonopposing normals give zero points. InvalidGeometry returns zero
   * points, and InsufficientCapacity reports the full required count. Either failure leaves output untouched.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult mapToElements( const FirstGeometry& edge1,
                                                                          const SecondGeometry& edge2,
                                                                          IntegrationSurfaceRuleView<1> reference,
                                                                          PairQuadratureRuleView<1, 1> output ) const
  {
    if ( !validEdges( edge1, edge2 ) ) {
      return { QuadratureStatus::InvalidGeometry, 0 };
    }
    if ( !opposingNormals( edge1, edge2 ) ) {
      return { QuadratureStatus::Success, 0 };
    }
    const auto ct = reference.numberOfPoints();
    if ( output.capacity() < ct ) {
      return { QuadratureStatus::InsufficientCapacity, ct };
    }
    if ( ct == 0 ) {
      return { QuadratureStatus::Success, 0 };
    }
    const auto line = computeCommonPlaneLine( edge1, edge2 );
    const RealT determinant1 = Vector::cross_product( edge1.direction(), line.normal )[2];
    const RealT determinant2 = Vector::cross_product( edge2.direction(), line.normal )[2];
    for ( IndexT q = 0; q < ct; ++q ) {
      const auto point = line.origin + reference.coords( q, 0 ) * line.tangent;
      output.geometry1_coords( q, 0 ) =
          2.0 * Vector::cross_product( Vector( edge1.origin(), point ), line.normal )[2] / determinant1 - 1.0;
      output.geometry2_coords( q, 0 ) =
          2.0 * Vector::cross_product( Vector( edge2.origin(), point ), line.normal )[2] / determinant2 - 1.0;
      output.weights[q] = reference.weights[q];
    }
    return { QuadratureStatus::Success, ct };
  }

 private:
  static constexpr RealT minimum_edge_length_ =
      1.0e-14;  ///< Shortest accepted edge, matching projected-edge quadrature.

  /**
   * @brief Check lengths before any unit normal is calculated.
   * @param edge1 First current edge.
   * @param edge2 Second current edge.
   * @return Whether both squared lengths are finite and at least the squared minimum length.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static bool validEdges( const FirstGeometry& edge1,
                                                                  const SecondGeometry& edge2 )
  {
    const RealT minimum_squared = minimum_edge_length_ * minimum_edge_length_;
    return std::isfinite( edge1.squaredLength() ) && std::isfinite( edge2.squaredLength() ) &&
           edge1.squaredLength() >= minimum_squared && edge2.squaredLength() >= minimum_squared;
  }

  /**
   * @brief Check opposing orientation without normalizing either edge.
   * @param edge1 First current edge.
   * @param edge2 Second current edge.
   * @return True when the edge normals have a negative dot product.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static bool opposingNormals( const FirstGeometry& edge1,
                                                                       const SecondGeometry& edge2 )
  {
    return edge1.direction().dot( edge2.direction() ) < 0.0;
  }
};

}  // namespace tribol
#endif
