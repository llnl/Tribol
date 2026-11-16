// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Constant-penalty traction and quadrature-point activation using common-plane gap and normal geometry.
 */
#ifndef SRC_TRIBOL_ENFORCEMENT_COMMONPLANEPENALTYTRACTION_HPP_
#define SRC_TRIBOL_ENFORCEMENT_COMMONPLANEPENALTYTRACTION_HPP_

#include "tribol/assembly/PairEvaluation.hpp"
#include "tribol/geom/CommonPlaneEdgeMapping.hpp"
#include "tribol/geom/CommonPlaneGap.hpp"

namespace tribol {

/** @brief Constant-penalty Common Plane traction, with activation determined by the configured gap.
 * @tparam GapDefinition Geometric or offset common-plane gap.
 */
template <typename GapDefinition = CommonPlaneGap>
struct CommonPlanePenaltyTraction {
  RealT penalty;        ///< Fixed effective penalty stiffness.
  GapDefinition gap{};  ///< Owned gap configuration.

  /** @brief Compute the common line once for the current pair.
   * @param geometry1 First current edge.
   * @param geometry2 Second current edge.
   * @return Common line used for both gap evaluation and traction direction.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto computePairGeometryData( const LinearEdgeGeometry& geometry1,
                                                                        const LinearEdgeGeometry& geometry2 ) const
  {
    return gap.computePairGeometryData( geometry1, geometry2 );
  }

  /** @brief Compute penalty traction at one quadrature point; zero gap retains its compressive tangent.
   * @param point1 Physical point on geometry1.
   * @param point2 Physical point on geometry2.
   * @param parametric_coords1 Geometry1 parametric point, unused by this penalty law.
   * @param parametric_coords2 Geometry2 parametric point, unused by this penalty law.
   * @param line Common line prepared for this pair.
   * @return Penalty traction and active=true for nonpositive gap; zero traction and inactive state for opening.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadraturePointValue<MdArray<RealT, LinearEdgeGeometry::spatial_dim>>
  operator()( const LinearEdgeGeometry::Point& point1, const LinearEdgeGeometry::Point& point2,
              MdSpan<const RealT, LinearEdgeGeometry::parametric_dim> TRIBOL_UNUSED_PARAM( parametric_coords1 ),
              MdSpan<const RealT, LinearEdgeGeometry::parametric_dim> TRIBOL_UNUSED_PARAM( parametric_coords2 ),
              const CommonPlaneEdgeMapping::CommonPlaneLine& line ) const
  {
    const RealT separation = gap( point1, point2, line );
    if ( separation > 0.0 ) {
      return {};
    }
    QuadraturePointValue<MdArray<RealT, LinearEdgeGeometry::spatial_dim>> result;
    result.active = true;
    const RealT pressure = penalty * separation;
    for ( IndexT spatial_dim = 0; spatial_dim < LinearEdgeGeometry::spatial_dim; ++spatial_dim ) {
      result.value[spatial_dim] = pressure * line.normal[spatial_dim];
    }
    return result;
  }
};

}  // namespace tribol
#endif  // SRC_TRIBOL_ENFORCEMENT_COMMONPLANEPENALTYTRACTION_HPP_
