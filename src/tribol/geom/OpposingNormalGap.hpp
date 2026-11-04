// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Evaluate the alignment-corrected geometric gap used by Energy Mortar. */
#ifndef SRC_TRIBOL_GEOM_OPPOSINGNORMALGAP_HPP_
#define SRC_TRIBOL_GEOM_OPPOSINGNORMALGAP_HPP_

#include "tribol/geom/LinearEdgeGeometry.hpp"

namespace tribol {
/** @brief Evaluate separation projected onto mesh2's normal with opposing-normal alignment. */
struct OpposingNormalGap {
  /**
   * @brief Calculate geometric separation without retaining points or normals.
   * @param point1 Current point on mesh1, the non-mortar side.
   * @param point2 Current point on mesh2, the mortar side.
   * @param normal1 Current unit normal of mesh1's edge.
   * @param normal2 Current unit normal of mesh2's edge.
   * @return (point2-point1).normal2 times min(normal1.normal2,0).
   * @pre Normals come from nondegenerate current geometry.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT operator()( const LinearEdgeGeometry::Point& point1,
                                                            const LinearEdgeGeometry::Point& point2,
                                                            const LinearEdgeGeometry::Vector& normal1,
                                                            const LinearEdgeGeometry::Vector& normal2 ) const
  {
    const RealT cosine = normal1.dot( normal2 );
    return LinearEdgeGeometry::Vector( point1, point2 ).dot( normal2 ) * ( cosine < 0.0 ? cosine : 0.0 );
  }
};
}  // namespace tribol
#endif
