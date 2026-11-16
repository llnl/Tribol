// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Compressive quadratic penalty energy density and partials with respect to nodal gap and area integrals.
 */
#ifndef SRC_TRIBOL_ENFORCEMENT_PENALTYENERGY_HPP_
#define SRC_TRIBOL_ENFORCEMENT_PENALTYENERGY_HPP_

#include <cmath>
#include "tribol/assembly/PairEvaluation.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Enzyme.hpp"

namespace tribol {

/** @brief Quadratic compressive penalty energy density using the configured gap.
 * @tparam GapDefinition Geometric or offset gap supporting per-pair geometry preparation.
 */
template <typename GapDefinition>
struct PenaltyEnergyDensity {
  RealT penalty;        ///< Fixed effective penalty stiffness.
  GapDefinition gap{};  ///< Owned gap configuration.

  /** @brief Compute the gap's geometry data once for the current pair.
   * @tparam Geometry1 First geometry type.
   * @tparam Geometry2 Second geometry type.
   * @param geometry1 First current geometry.
   * @param geometry2 Second current geometry.
   * @return Geometry data used by the configured gap.
   */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto computePairGeometryData( const Geometry1& geometry1,
                                                                        const Geometry2& geometry2 ) const
  {
    return gap.computePairGeometryData( geometry1, geometry2 );
  }

  /** @brief Compute 0.5*penalty*gap^2; zero gap retains the compressive tangent.
   * @tparam Point1 Geometry1 physical point type.
   * @tparam Point2 Geometry2 physical point type.
   * @tparam ParametricCoords1 Geometry1 parametric coordinate view type.
   * @tparam ParametricCoords2 Geometry2 parametric coordinate view type.
   * @tparam GeometryData Data prepared by the gap for this pair.
   * @param point1 Physical point on geometry1.
   * @param point2 Physical point on geometry2.
   * @param parametric_coords1 Geometry1 parametric point, unused by this energy law.
   * @param parametric_coords2 Geometry2 parametric point, unused by this energy law.
   * @param pair_geometry_data Current gap geometry data.
   * @return Energy density and active=true for nonpositive gap; zero density and inactive state for opening.
   */
  template <typename Point1, typename Point2, typename ParametricCoords1, typename ParametricCoords2,
            typename GeometryData>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadraturePointValue<RealT> operator()(
      const Point1& point1, const Point2& point2, ParametricCoords1 TRIBOL_UNUSED_PARAM( parametric_coords1 ),
      ParametricCoords2 TRIBOL_UNUSED_PARAM( parametric_coords2 ), const GeometryData& pair_geometry_data ) const
  {
    const RealT separation = gap( point1, point2, pair_geometry_data );
    if ( separation > 0.0 ) {
      return {};
    }
    return { 0.5 * penalty * separation * separation, true };
  }
};

/** @brief Energy and partials with respect to one assembled weighted gap and tributary area. */
struct EnergyMortarNodalPenaltyPartials {
  RealT energy = 0.0;                                  ///< Nodal penalty energy.
  RealT denergy_dweighted_gap = 0.0;                   ///< First weighted-gap partial, also the nodal pressure.
  RealT denergy_dtributary_area = 0.0;                 ///< First tributary-area partial.
  RealT d2energy_dweighted_gap2 = 0.0;                 ///< Second weighted-gap partial.
  RealT d2energy_dweighted_gap_dtributary_area = 0.0;  ///< Mixed weighted-gap/area partial.
  RealT d2energy_dtributary_area2 = 0.0;               ///< Second tributary-area partial.
  /**
   * @brief Calculate penalty energy and its field partials before coordinate assembly.
   * @param weighted_gap Assembled gap integral before activation.
   * @param tributary_area Assembled tributary area.
   * @param penalty Fixed effective stiffness.
   * @param area_tolerance Maximum unsupported absolute area.
   * @return Zero values and derivatives for unsupported area or open contact. At zero gap the compressive tangent
   * remains active. Division is performed only after checking area support.
   */
  TRIBOL_HOST_DEVICE static EnergyMortarNodalPenaltyPartials evaluate( RealT weighted_gap, RealT tributary_area,
                                                                       RealT penalty, RealT area_tolerance )
  {
    if ( std::abs( tributary_area ) <= area_tolerance || weighted_gap > 0.0 ) {
      return {};
    }
    const RealT inverse_area = 1.0 / tributary_area;
    const RealT pressure = penalty * weighted_gap * inverse_area;
    return { 0.5 * pressure * weighted_gap,
             pressure,
             -0.5 * pressure * weighted_gap * inverse_area,
             penalty * inverse_area,
             -pressure * inverse_area,
             pressure * weighted_gap * inverse_area * inverse_area };
  }
};

}  // namespace tribol
#endif  // SRC_TRIBOL_ENFORCEMENT_PENALTYENERGY_HPP_
