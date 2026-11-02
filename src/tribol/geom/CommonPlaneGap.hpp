// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Evaluate geometric separation along the current common-plane normal.
 */
#ifndef SRC_TRIBOL_GEOM_COMMONPLANEGAP_HPP_
#define SRC_TRIBOL_GEOM_COMMONPLANEGAP_HPP_

#include "tribol/geom/CommonPlaneEdgeMapping.hpp"

namespace tribol {
/** @brief Calculate the unshifted common-plane gap on nondegenerate opposing linear edges. */
struct CommonPlaneGap {
  /**
   * @brief Form a common frame from current geometry and evaluate the normal separation.
   * @tparam Point Pair quadrature point type.
   * @param point Current points and element geometries.
   * @return Signed separation (point1 - point2) dotted with the common-plane unit normal.
   * @pre Geometry admits a valid common-plane frame.
   */
  template <typename Point>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT operator()( const Point& point ) const
  {
    return ( *this )( point, CommonPlaneEdgeMapping::makeFrame( point.geometry1, point.geometry2 ) );
  }
  /**
   * @brief Reuse the frame also used to calculate traction direction.
   * @tparam Point Pair quadrature point type.
   * @param point Current points on both elements.
   * @param frame Valid common frame computed from those same current element coordinates.
   * @return Signed normal separation; no frame is copied or retained.
   */
  template <typename Point>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT operator()( const Point& point,
                                                            const CommonPlaneEdgeMapping::Frame& frame ) const
  {
    return LinearEdgeGeometry::Vector( point.position2(), point.position1() ).dot( frame.normal );
  }
};
}  // namespace tribol
#endif  // SRC_TRIBOL_GEOM_COMMONPLANEGAP_HPP_
