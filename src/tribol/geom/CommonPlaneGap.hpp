// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Evaluate geometric separation along the current common-line normal. */
#ifndef SRC_TRIBOL_GEOM_COMMONPLANEGAP_HPP_
#define SRC_TRIBOL_GEOM_COMMONPLANEGAP_HPP_

#include "tribol/geom/CommonPlaneEdgeMapping.hpp"

namespace tribol {
/** @brief Calculate the unshifted gap along a valid current common-plane normal. */
struct CommonPlaneGap {
  /**
   * @brief Project the separation between physical quadrature points onto the common normal.
   * @param point1 Point on the first edge.
   * @param point2 Point on the second edge.
   * @param frame Common-line frame calculated from those same current edges.
   * @return (point1-point2).normal, negative in compression under the Common Plane orientation convention.
   * @pre Both edges are nondegenerate and their directed normals oppose each other.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT operator()( const LinearEdgeGeometry::Point& point1,
                                                            const LinearEdgeGeometry::Point& point2,
                                                            const CommonPlaneEdgeMapping::Frame& frame ) const
  {
    return LinearEdgeGeometry::Vector( point2, point1 ).dot( frame.normal );
  }
};
}  // namespace tribol
#endif
