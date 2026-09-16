#ifndef TRIBOL_METHOD_VALIDATION_HPP_
#define TRIBOL_METHOD_VALIDATION_HPP_

#include "tribol/method/Method.hpp"

#include <stdexcept>
#include <type_traits>

namespace tribol {

template <SupportedMethod MethodType>
void validateMethodParameters( const typename MethodType::Parameters& parameters )
{
  using Geometry = typename MethodType::geometry_policy;
  using Enforcement = typename MethodType::enforcement_policy;
  using Response = typename MethodType::response_policy;
  if constexpr ( detail::is_projected_overlap_v<Geometry> ) {
    if ( parameters.geometry.minimum_measure <= 0.0 ) {
      throw std::invalid_argument( "Projected-overlap minimum measure must be positive." );
    }
  } else if constexpr ( std::same_as<Geometry, geometry::ConformingOverlap> ) {
    if ( parameters.geometry.alignment_tolerance < 0.0 ) {
      throw std::invalid_argument( "Conforming-overlap alignment tolerance cannot be negative." );
    }
  }
  if constexpr ( detail::is_penalty_v<Enforcement> ) {
    using Stiffness = typename Enforcement::stiffness_policy;
    using Rate = typename Enforcement::rate_policy;
    if constexpr ( std::same_as<Stiffness, stiffness::Constant> ) {
      if ( parameters.enforcement.stiffness.value < 0.0 ) {
        throw std::invalid_argument( "Constant penalty stiffness cannot be negative." );
      }
    } else if ( parameters.enforcement.stiffness.scale < 0.0 ) {
      throw std::invalid_argument( "Material penalty scale cannot be negative." );
    }
    if constexpr ( std::same_as<Rate, rate::Constant> ) {
      if ( parameters.enforcement.rate.value < 0.0 ) {
        throw std::invalid_argument( "Constant rate penalty cannot be negative." );
      }
    } else if constexpr ( std::same_as<Rate, rate::Percentage> ) {
      if ( parameters.enforcement.rate.ratio < 0.0 ) {
        throw std::invalid_argument( "Percentage rate penalty cannot be negative." );
      }
    }
  }
  if constexpr ( std::same_as<Response, response::ViscousTangential> ) {
    if ( parameters.response.damping < 0.0 ) {
      throw std::invalid_argument( "Tangential damping cannot be negative." );
    }
  }
}

}  // namespace tribol

#endif
