// Copyright (c) Lawrence Livermore National Security, LLC and
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
  /** @brief Current unit normals used by all quadrature points of a pair. */
  struct Normals {
    LinearEdgeGeometry::Vector normal1;  ///< Geometry1 unit normal.
    LinearEdgeGeometry::Vector normal2;  ///< Geometry2 unit normal.
  };

  /** @brief Calculate both current normals once before evaluating quadrature-point gaps.
   * @param geometry1 First nondegenerate edge.
   * @param geometry2 Second nondegenerate edge.
   * @return Owning unit normals, with no references to the geometries.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Normals computePairGeometryData( const LinearEdgeGeometry& geometry1,
                                                                           const LinearEdgeGeometry& geometry2 ) const
  {
    return { geometry1.normal(), geometry2.normal() };
  }

  /**
   * @brief Calculate geometric separation using the current pair's unit normals.
   * @param point1 Current point on geometry1, the non-mortar side.
   * @param point2 Current point on geometry2, the mortar side.
   * @param normals Unit normals of those geometries.
   * @return (point2-point1).normal2 times min(normal1.normal2,0).
   * @pre Normals come from nondegenerate current geometry.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT operator()( const LinearEdgeGeometry::Point& point1,
                                                            const LinearEdgeGeometry::Point& point2,
                                                            const Normals& normals ) const
  {
    const RealT cosine = normals.normal1.dot( normals.normal2 );
    return LinearEdgeGeometry::Vector( point1, point2 ).dot( normals.normal2 ) * ( cosine < 0.0 ? cosine : 0.0 );
  }
};
}  // namespace tribol
#endif
