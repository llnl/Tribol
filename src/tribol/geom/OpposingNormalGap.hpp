// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Evaluate the geometric opposing-normal gap used by Energy Mortar.
 */
#ifndef SRC_TRIBOL_GEOM_OPPOSINGNORMALGAP_HPP_
#define SRC_TRIBOL_GEOM_OPPOSINGNORMALGAP_HPP_

#include "tribol/geom/LinearEdgeGeometry.hpp"

namespace tribol {
/**
 * @brief Calculate the opposing-normal-weighted gap at one pair quadrature point.
 *
 * Mesh1 is the non-mortar integration side. The gap projects the separation from mesh1 to mesh2 onto mesh2's current
 * directed normal and scales by the opposing-normal alignment factor min(normal1 . normal2, 0). Nondegenerate linear
 * edges are required before evaluating their unit normals.
 */
struct OpposingNormalGap {
  /**
   * @brief Evaluate the normal gap at a pair quadrature point.
   * @tparam Point PairQuadraturePoint type with 1D linear-edge geometries.
   * @param point Current pair quadrature point.
   * @return Normal gap at the quadrature point.
   */
  template <typename Point>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT operator()( const Point& point ) const
  {
    const auto normal1 = point.geometry1.normal(), normal2 = point.geometry2.normal();
    const RealT cosine = normal1.dot( normal2 );
    const RealT alignment = cosine < 0.0 ? cosine : 0.0;
    return LinearEdgeGeometry::Vector( point.position1(), point.position2() ).dot( normal2 ) * alignment;
  }
};

}  // namespace tribol
#endif  // SRC_TRIBOL_GEOM_OPPOSINGNORMALGAP_HPP_
