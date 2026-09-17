#ifndef TRIBOL_METHOD_VALIDATION_HPP_
#define TRIBOL_METHOD_VALIDATION_HPP_

#include "tribol/method/Method.hpp"
#include "tribol/core/MeshView.hpp"

#include <stdexcept>
#include <type_traits>

namespace tribol {

template <SupportedMethod MethodType>
void validateMethodParameters( const typename MethodType::Parameters& parameters )
{
  using Geometry = typename MethodType::geometry_policy;
  using Integration = typename MethodType::integration_policy;
  using Enforcement = typename MethodType::enforcement_policy;
  using Response = typename MethodType::response_policy;
  if constexpr ( detail::is_projected_overlap_v<Geometry> ) {
    if ( parameters.geometry.minimum_measure <= 0.0 ) {
      throw std::invalid_argument( "Projected-overlap minimum measure must be positive." );
    }
    if ( parameters.geometry.minimum_overlap_fraction < 0.0 || parameters.geometry.minimum_overlap_fraction > 1.0 ) {
      throw std::invalid_argument( "Projected-overlap minimum fraction must be in [0, 1]." );
    }
  } else if constexpr ( std::same_as<Geometry, geometry::ConformingOverlap> ) {
    if ( parameters.geometry.alignment_tolerance < 0.0 ) {
      throw std::invalid_argument( "Conforming-overlap alignment tolerance cannot be negative." );
    }
  }
  if constexpr ( detail::is_smoothed_segment_integration_v<Integration> ) {
    if ( parameters.integration.endpoint_width < 0.0 || parameters.integration.endpoint_width >= 0.5 ) {
      throw std::invalid_argument( "Smoothed-segment endpoint width must be in [0, 0.5)." );
    }
  }
  if ( parameters.constraint.activation.residual_gap < 0.0 ) {
    throw std::invalid_argument( "Residual gap cannot be negative." );
  }
  if ( parameters.constraint.activation.gap_tolerance < 0.0 ) {
    throw std::invalid_argument( "Gap tolerance cannot be negative." );
  }
  if ( parameters.constraint.activation.reject_excessive_penetration &&
       parameters.constraint.activation.maximum_penetration_fraction <= 0.0 ) {
    throw std::invalid_argument( "Maximum penetration fraction must be positive." );
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
    if ( parameters.enforcement.stiffness.mortar_scale < 0.0 ||
         parameters.enforcement.stiffness.nonmortar_scale < 0.0 ) {
      throw std::invalid_argument( "Penalty side scales cannot be negative." );
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
    if constexpr ( !std::same_as<Rate, rate::None> ) {
      if ( parameters.enforcement.rate.mortar_scale < 0.0 || parameters.enforcement.rate.nonmortar_scale < 0.0 ) {
        throw std::invalid_argument( "Rate-penalty side scales cannot be negative." );
      }
    }
  }
  if constexpr ( std::same_as<Response, response::ViscousTangential> ) {
    if ( parameters.response.damping < 0.0 ) {
      throw std::invalid_argument( "Tangential damping cannot be negative." );
    }
  }
}

template <SupportedMethod MethodType, typename Scalar>
void validateMethodDomain( const SurfacePairViewT<Scalar>& surfaces )
{
  using Integration = typename MethodType::integration_policy;
  if constexpr ( detail::is_smoothed_segment_integration_v<Integration> ) {
    if ( surfaces.mortar.dimension != 2 || surfaces.nonmortar.dimension != 2 ) {
      throw std::invalid_argument( "Smoothed-segment integration requires two-dimensional surfaces." );
    }
    for ( ElementTopology topology : surfaces.mortar.topologies ) {
      if ( topology != ElementTopology::Segment ) {
        throw std::invalid_argument( "Smoothed-segment integration requires segment elements." );
      }
    }
    for ( ElementTopology topology : surfaces.nonmortar.topologies ) {
      if ( topology != ElementTopology::Segment ) {
        throw std::invalid_argument( "Smoothed-segment integration requires segment elements." );
      }
    }
  }
}

}  // namespace tribol

#endif
