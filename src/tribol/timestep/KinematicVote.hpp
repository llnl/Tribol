#ifndef TRIBOL_TIMESTEP_KINEMATICVOTE_HPP_
#define TRIBOL_TIMESTEP_KINEMATICVOTE_HPP_

#include "tribol/constraint/NormalConstraint.hpp"
#include "tribol/geom/ProjectedOverlap.hpp"
#include "tribol/method/Method.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tribol::timestep {

struct Kinematic {
  struct Parameters {
    bool enabled{};
    Real current_step{ std::numeric_limits<Real>::infinity() };
    Real penetration_fraction{ 0.3 };
    Real scale{ 1.0 };
    Real minimum_current_step{ 1.0e-8 };
    Real velocity_tolerance{ 1.0e-12 };
  };
};

inline void validate( const Kinematic::Parameters& parameters )
{
  if ( parameters.penetration_fraction < 0.0 ) {
    throw std::invalid_argument( "Timestep penetration fraction cannot be negative." );
  }
  if ( parameters.scale <= 0.0 ) {
    throw std::invalid_argument( "Timestep scale must be positive." );
  }
  if ( parameters.minimum_current_step < 0.0 || parameters.velocity_tolerance < 0.0 ) {
    throw std::invalid_argument( "Timestep tolerances cannot be negative." );
  }
  if ( parameters.enabled && ( !std::isfinite( parameters.current_step ) || parameters.current_step <= 0.0 ) ) {
    throw std::invalid_argument( "An enabled kinematic timestep vote requires a finite positive current step." );
  }
}

namespace detail {

inline void requireVectorField( const FieldView<const Real>& field, const SurfaceMeshView& surface,
                                const char* message )
{
  if ( !field.isStructurallyValid() || field.entities != surface.numberOfNodes() ||
       field.components != surface.dimension ) {
    throw std::invalid_argument( message );
  }
}

inline void requireElementField( ArrayView<const Real> field, const SurfaceMeshView& surface, const char* message )
{
  if ( !field.isStructurallyValid() || field.size() != surface.numberOfElements() ) {
    throw std::invalid_argument( message );
  }
}

inline std::array<Real, 3> interpolate( const SurfaceMeshView& surface, Index element,
                                        const FieldView<const Real>& field, const basis::ShapeValues& shape )
{
  std::array<Real, 3> value{};
  for ( int local_node = 0; local_node < shape.size; ++local_node ) {
    const Index node = surface.connectivity[surface.element_offsets[element] + local_node];
    for ( int component = 0; component < surface.dimension; ++component ) {
      value[component] += shape[local_node] * field( node, component );
    }
  }
  return value;
}

inline Real dot( const std::array<Real, 3>& left, const std::array<Real, 3>& right, int dimension )
{
  Real value{};
  for ( int component = 0; component < dimension; ++component ) {
    value += left[component] * right[component];
  }
  return value;
}

inline Real nonzeroProjection( Real value, Real tolerance )
{
  return value + ( value >= 0.0 ? tolerance : -tolerance );
}

inline std::array<Real, 3> elementNormal( const SurfaceMeshView& surface, Index element )
{
  return surface.dimension == 2 ? projected_overlap_detail::segmentNormal( surface, element )
                                : projected_overlap_detail::faceNormal( surface, element );
}

inline void includePositiveVote( Real candidate, Real& vote )
{
  if ( candidate > 0.0 ) {
    vote = std::min( vote, candidate );
  }
}

}  // namespace detail

template <SupportedMethod MethodType>
[[nodiscard]] Real kinematicVote( const SurfacePairView& surfaces, ArrayView<const ElementPair> interactions,
                                  const typename MethodType::Parameters& method_parameters,
                                  const ContactStateView& state, const Kinematic::Parameters& parameters )
{
  validate( parameters );
  if ( !parameters.enabled ) {
    return std::numeric_limits<Real>::infinity();
  }
  if ( parameters.current_step < parameters.minimum_current_step ) {
    return parameters.current_step;
  }
  if constexpr ( !tribol::detail::is_mean_plane_overlap_v<typename MethodType::geometry_policy> ) {
    throw std::invalid_argument( "Kinematic timestep voting requires projected mean-plane geometry." );
  } else {
    detail::requireVectorField( state.mortar_velocity, surfaces.mortar,
                                "Kinematic timestep voting requires mortar nodal velocity." );
    detail::requireVectorField( state.nonmortar_velocity, surfaces.nonmortar,
                                "Kinematic timestep voting requires nonmortar nodal velocity." );
    detail::requireElementField( state.mortar_element_thickness, surfaces.mortar,
                                 "Kinematic timestep voting requires mortar element thickness." );
    detail::requireElementField( state.nonmortar_element_thickness, surfaces.nonmortar,
                                 "Kinematic timestep voting requires nonmortar element thickness." );

    Real vote = parameters.current_step;
    for ( const ElementPair pair : interactions ) {
      const auto patch = projectedOverlapPatch<normal::MeanPlane>( surfaces, pair, method_parameters.geometry );
      if ( !patch.valid ) {
        continue;
      }
      const auto sample = constraint::stageNormalConstraint<constraint::Pointwise>(
          surfaces, pair, patch.mortar_centroid, patch.nonmortar_centroid, patch.normal, patch.measure );
      const auto mortar_velocity =
          detail::interpolate( surfaces.mortar, pair.mortar_element, state.mortar_velocity, sample.mortar_trial );
      const auto nonmortar_velocity = detail::interpolate( surfaces.nonmortar, pair.nonmortar_element,
                                                           state.nonmortar_velocity, sample.nonmortar_trial );
      const auto mortar_normal = detail::elementNormal( surfaces.mortar, pair.mortar_element );
      const auto nonmortar_normal = detail::elementNormal( surfaces.nonmortar, pair.nonmortar_element );
      const int dimension = surfaces.mortar.dimension;
      const Real mortar_overlap_velocity = detail::nonzeroProjection(
          detail::dot( mortar_velocity, patch.normal, dimension ), parameters.velocity_tolerance );
      const Real nonmortar_overlap_velocity = detail::nonzeroProjection(
          detail::dot( nonmortar_velocity, patch.normal, dimension ), parameters.velocity_tolerance );
      const Real mortar_normal_velocity = detail::nonzeroProjection(
          detail::dot( mortar_velocity, mortar_normal, dimension ), parameters.velocity_tolerance );
      const Real nonmortar_normal_velocity = detail::nonzeroProjection(
          detail::dot( nonmortar_velocity, nonmortar_normal, dimension ), parameters.velocity_tolerance );
      const Real mortar_limit = parameters.penetration_fraction * state.mortar_element_thickness[pair.mortar_element];
      const Real nonmortar_limit =
          parameters.penetration_fraction * state.nonmortar_element_thickness[pair.nonmortar_element];
      if ( mortar_limit < 0.0 || nonmortar_limit < 0.0 ) {
        throw std::invalid_argument( "Kinematic timestep voting requires nonnegative element thickness." );
      }

      std::array<Real, 3> gap_vector{};
      std::array<Real, 3> projected_gap{};
      for ( int component = 0; component < dimension; ++component ) {
        gap_vector[component] = patch.mortar_centroid[component] - patch.nonmortar_centroid[component];
        projected_gap[component] = gap_vector[component] + parameters.current_step * ( mortar_velocity[component] -
                                                                                       nonmortar_velocity[component] );
      }
      const Real mortar_gap = detail::dot( gap_vector, mortar_normal, dimension );
      const Real nonmortar_gap = detail::dot( gap_vector, nonmortar_normal, dimension );
      const Real projected_mortar_penetration = detail::dot( projected_gap, nonmortar_normal, dimension );
      const Real projected_nonmortar_penetration = -detail::dot( projected_gap, mortar_normal, dimension );
      const bool mortar_closing = mortar_overlap_velocity < 0.0;
      const bool nonmortar_closing = nonmortar_overlap_velocity > 0.0;
      const bool active = sample.gap - method_parameters.constraint.activation.residual_gap <=
                          method_parameters.constraint.activation.gap_tolerance;
      const bool mortar_gap_exceeded = active && mortar_closing && mortar_limit - mortar_gap < 0.0;
      const bool nonmortar_gap_exceeded = active && nonmortar_closing && nonmortar_limit + nonmortar_gap < 0.0;

      if ( mortar_gap_exceeded ) {
        detail::includePositiveVote( parameters.scale * mortar_limit / mortar_normal_velocity, vote );
      }
      if ( nonmortar_gap_exceeded ) {
        detail::includePositiveVote( parameters.scale * nonmortar_limit / nonmortar_normal_velocity, vote );
      }
      if ( mortar_closing && !mortar_gap_exceeded && projected_mortar_penetration < -mortar_limit ) {
        detail::includePositiveVote( parameters.scale * mortar_limit / mortar_normal_velocity, vote );
      }
      if ( nonmortar_closing && !nonmortar_gap_exceeded && projected_nonmortar_penetration < -nonmortar_limit ) {
        detail::includePositiveVote( parameters.scale * nonmortar_limit / nonmortar_normal_velocity, vote );
      }
    }
    return vote;
  }
}

}  // namespace tribol::timestep

#endif
