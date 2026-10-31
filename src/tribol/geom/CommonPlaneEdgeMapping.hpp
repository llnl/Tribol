// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
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
 * @brief Project two edges onto a common line and map points back along its normal.
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
  struct Frame {
    Point origin;    ///< Average of the two edge midpoints.
    Vector normal;   ///< Unit direction of normal2 minus normal1.
    Vector tangent;  ///< Unit direction along the integration line.
  };

  /**
   * @brief Construct the same common-line frame used for projection and contact tractions.
   * @param edge1 First current edge.
   * @param edge2 Second current edge.
   * @pre Both edges are nondegenerate and have opposing normals.
   * @return Owning origin and unit directions; no geometry is retained.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static Frame makeFrame( const FirstGeometry& edge1,
                                                                  const SecondGeometry& edge2 )
  {
    auto normal = edge2.normal() - edge1.normal();
    normal /= std::sqrt( normal.dot( normal ) );
    return { Point::lerp( edge1.position( 0.0 ), edge2.position( 0.0 ), 0.5 ), normal,
             Vector{ normal[1], -normal[0] } };
  }

  /**
   * @brief Intersect the two edges' projections onto the common line.
   * @param edge1 First current edge.
   * @param edge2 Second current edge.
   * @return Physical tangent bounds; invalid edges return valid=false and zero bounds. Nonopposing normals and
   * disjoint or touching intervals return valid collapsed bounds and contribute no points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurface makeIntegrationSurface( const FirstGeometry& edge1,
                                                                                     const SecondGeometry& edge2 ) const
  {
    if ( !validEdges( edge1, edge2 ) ) return { MdArray<RealT, 2>{ 0.0, 0.0 }, false };
    if ( !opposingNormals( edge1, edge2 ) ) return { MdArray<RealT, 2>{ 0.0, 0.0 }, true };
    const auto frame = makeFrame( edge1, edge2 );
    const RealT a0 = Vector( frame.origin, edge1.origin() ).dot( frame.tangent );
    const RealT a1 = Vector( frame.origin, edge1.position( 1.0 ) ).dot( frame.tangent );
    const RealT b0 = Vector( frame.origin, edge2.origin() ).dot( frame.tangent );
    const RealT b1 = Vector( frame.origin, edge2.position( 1.0 ) ).dot( frame.tangent );
    const RealT lower1 = a0 < a1 ? a0 : a1;
    const RealT upper1 = a0 < a1 ? a1 : a0;
    const RealT lower2 = b0 < b1 ? b0 : b1;
    const RealT upper2 = b0 < b1 ? b1 : b0;
    const RealT lower = lower1 > lower2 ? lower1 : lower2;
    const RealT upper = upper1 < upper2 ? upper1 : upper2;
    return { MdArray<RealT, 2>{ lower, upper > lower ? upper : lower }, true };
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
                                                                          PairQuadratureView<1, 1> output ) const
  {
    if ( !validEdges( edge1, edge2 ) ) return { QuadratureStatus::InvalidGeometry, 0 };
    if ( !opposingNormals( edge1, edge2 ) ) return { QuadratureStatus::Success, 0 };
    const auto count = reference.numberOfPoints();
    if ( output.capacity() < count ) return { QuadratureStatus::InsufficientCapacity, count };
    if ( count == 0 ) return { QuadratureStatus::Success, 0 };
    const auto frame = makeFrame( edge1, edge2 );
    const RealT determinant1 = Vector::cross_product( edge1.direction(), frame.normal )[2];
    const RealT determinant2 = Vector::cross_product( edge2.direction(), frame.normal )[2];
    for ( IndexT q = 0; q < count; ++q ) {
      const auto point = frame.origin + reference.coords( q, 0 ) * frame.tangent;
      output.geometry1_coords( q, 0 ) =
          2.0 * Vector::cross_product( Vector( edge1.origin(), point ), frame.normal )[2] / determinant1 - 1.0;
      output.geometry2_coords( q, 0 ) =
          2.0 * Vector::cross_product( Vector( edge2.origin(), point ), frame.normal )[2] / determinant2 - 1.0;
      output.weights[q] = reference.weights[q];
    }
    return { QuadratureStatus::Success, count };
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
