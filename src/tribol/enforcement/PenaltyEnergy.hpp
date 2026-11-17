// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Nodal penalty energy and partials with respect to weighted gap and tributary area.
 */
#ifndef SRC_TRIBOL_ENFORCEMENT_PENALTYENERGY_HPP_
#define SRC_TRIBOL_ENFORCEMENT_PENALTYENERGY_HPP_

#include <cmath>
#include "tribol/common/BasicTypes.hpp"

namespace tribol {

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
