#ifndef TRIBOL_CONSTRAINT_PENETRATIONLIMIT_HPP_
#define TRIBOL_CONSTRAINT_PENETRATIONLIMIT_HPP_

#include "tribol/core/ExactTangent.hpp"
#include "tribol/evaluation/State.hpp"
#include "tribol/method/Method.hpp"
#include "tribol/search/Search.hpp"

#include <stdexcept>

namespace tribol::constraint {

template <typename Scalar>
[[nodiscard]] inline bool exceedsPenetrationLimit( const GapActivationParameters& parameters,
                                                   const ContactStateViewT<Scalar>& state, ElementPair pair,
                                                   const Scalar& effective_gap )
{
  if ( !parameters.reject_excessive_penetration ) {
    return false;
  }
  if ( state.mortar_element_thickness.size() <= pair.mortar_element ||
       state.nonmortar_element_thickness.size() <= pair.nonmortar_element ) {
    throw std::invalid_argument( "Thickness-limited activation requires element thicknesses on both surfaces." );
  }
  const Scalar mortar_thickness = state.mortar_element_thickness[pair.mortar_element];
  const Scalar nonmortar_thickness = state.nonmortar_element_thickness[pair.nonmortar_element];
  if ( linearization_detail::primal( mortar_thickness ) <= 0.0 ||
       linearization_detail::primal( nonmortar_thickness ) <= 0.0 ) {
    throw std::invalid_argument( "Thickness-limited activation requires positive element thicknesses." );
  }
  const Scalar minimum_thickness =
      linearization_detail::primal( mortar_thickness ) <= linearization_detail::primal( nonmortar_thickness )
          ? mortar_thickness
          : nonmortar_thickness;
  return linearization_detail::primal( effective_gap ) <
         -parameters.maximum_penetration_fraction * linearization_detail::primal( minimum_thickness );
}

}  // namespace tribol::constraint

#endif
