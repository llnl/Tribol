#ifndef TRIBOL_EVALUATION_POINTWISEPENALTY_HPP_
#define TRIBOL_EVALUATION_POINTWISEPENALTY_HPP_

#include "tribol/constraint/NormalConstraint.hpp"
#include "tribol/constraint/PenetrationLimit.hpp"
#include "tribol/evaluation/State.hpp"
#include "tribol/geom/ProjectedOverlap.hpp"
#include "tribol/method/Traits.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <type_traits>

namespace tribol {

template <typename MethodType>
concept PointwisePenaltyMethod =
    SupportedMethod<MethodType> && detail::is_penalty_v<typename MethodType::enforcement_policy> &&
    std::same_as<typename MethodType::formulation_policy, formulation::PointwiseTraction>;

namespace pointwise_penalty_detail {

template <typename Scalar>
TRIBOL_HOST_DEVICE inline Scalar seriesStiffness( Scalar mortar, Scalar nonmortar )
{
  const Scalar sum = mortar + nonmortar;
  return linearization_detail::absolute( linearization_detail::primal( sum ) ) > 1.0e-28 ? mortar * nonmortar / sum
                                                                                         : Scalar{};
}

template <typename FieldScalar, typename MeshScalar>
inline void requireVectorField( const FieldView<FieldScalar>& field, const SurfaceMeshViewT<MeshScalar>& mesh,
                                const char* name )
{
  if ( !field.isStructurallyValid() || field.entities != mesh.numberOfNodes() || field.components != mesh.dimension ) {
    throw std::invalid_argument( name );
  }
}

template <typename MeshScalar, typename FieldScalar, typename ShapeScalar>
inline auto interpolateField( const SurfaceMeshViewT<MeshScalar>& mesh, Index element,
                              const FieldView<FieldScalar>& field, const basis::ShapeValuesT<ShapeScalar>& shape )
{
  using Scalar = decltype( field( Index{}, 0 ) * shape[0] );
  std::array<Scalar, 3> average{};
  const Index begin = mesh.element_offsets[element];
  for ( int local_node = 0; local_node < shape.size; ++local_node ) {
    const Index node = mesh.connectivity[begin + local_node];
    for ( int component = 0; component < mesh.dimension; ++component ) {
      average[component] += field( node, component ) * shape[local_node];
    }
  }
  return average;
}

template <typename Enforcement, typename Scalar>
inline Scalar penaltyStiffness( const typename Enforcement::Parameters& parameters,
                                const ContactStateViewT<Scalar>& state, ElementPair pair )
{
  using Stiffness = typename Enforcement::stiffness_policy;
  if constexpr ( std::same_as<Stiffness, stiffness::Constant> ) {
    return parameters.stiffness.value;
  } else {
    if ( state.mortar_element_thickness.size() <= pair.mortar_element ||
         state.nonmortar_element_thickness.size() <= pair.nonmortar_element ||
         state.mortar_material_modulus.size() <= pair.mortar_element ||
         state.nonmortar_material_modulus.size() <= pair.nonmortar_element ) {
      throw std::invalid_argument( "Material penalty requires thickness and modulus fields on both surfaces." );
    }
    if ( state.mortar_element_thickness[pair.mortar_element] <= 0.0 ||
         state.nonmortar_element_thickness[pair.nonmortar_element] <= 0.0 ) {
      throw std::invalid_argument( "Material penalty requires positive element thicknesses." );
    }
    const Scalar mortar_stiffness =
        state.mortar_material_modulus[pair.mortar_element] / state.mortar_element_thickness[pair.mortar_element];
    const Scalar nonmortar_stiffness = state.nonmortar_material_modulus[pair.nonmortar_element] /
                                       state.nonmortar_element_thickness[pair.nonmortar_element];
    return parameters.stiffness.scale * seriesStiffness( parameters.stiffness.mortar_scale * mortar_stiffness,
                                                         parameters.stiffness.nonmortar_scale * nonmortar_stiffness );
  }
}

template <typename Enforcement, typename Scalar>
inline Scalar pointwisePenaltyStiffness( const typename Enforcement::Parameters& parameters,
                                         const ContactStateViewT<Scalar>& state, ElementPair pair )
{
  using Stiffness = typename Enforcement::stiffness_policy;
  if constexpr ( std::same_as<Stiffness, stiffness::Constant> ) {
    const Scalar mortar = parameters.stiffness.mortar_scale * parameters.stiffness.value;
    const Scalar nonmortar = parameters.stiffness.nonmortar_scale * parameters.stiffness.value;
    return seriesStiffness( mortar, nonmortar );
  } else {
    return penaltyStiffness<Enforcement>( parameters, state, pair );
  }
}

template <typename Enforcement, typename Scalar>
inline Scalar rateCoefficient( const typename Enforcement::Parameters& parameters, Scalar stiffness )
{
  using Rate = typename Enforcement::rate_policy;
  if constexpr ( std::same_as<Rate, rate::None> ) {
    return 0.0;
  } else if constexpr ( std::same_as<Rate, rate::Constant> ) {
    return 0.5 * parameters.rate.value * ( parameters.rate.mortar_scale + parameters.rate.nonmortar_scale );
  } else {
    return 0.5 * parameters.rate.ratio * ( parameters.rate.mortar_scale + parameters.rate.nonmortar_scale ) * stiffness;
  }
}

template <typename LeftScalar, typename RightScalar>
inline auto difference( const std::array<LeftScalar, 3>& left, const std::array<RightScalar, 3>& right )
{
  using Scalar = decltype( left[0] - right[0] );
  return std::array<Scalar, 3>{ left[0] - right[0], left[1] - right[1], left[2] - right[2] };
}

template <typename LeftScalar, typename RightScalar>
inline auto dot( const std::array<LeftScalar, 3>& left, const std::array<RightScalar, 3>& right, int dimension )
{
  using Scalar = decltype( left[0] * right[0] );
  Scalar result{};
  for ( int component = 0; component < dimension; ++component ) {
    result += left[component] * right[component];
  }
  return result;
}

template <typename Scalar>
inline void scatterPairForce( const SurfacePairViewT<Scalar>& surfaces, ElementPair pair,
                              const std::array<Scalar, 3>& force, ContactResidualViewT<Scalar> residual )
{
  const SurfaceMeshViewT<Scalar> meshes[2] = { surfaces.mortar, surfaces.nonmortar };
  const Index elements[2] = { pair.mortar_element, pair.nonmortar_element };
  FieldView<Scalar> fields[2] = { residual.mortar, residual.nonmortar };
  for ( int side = 0; side < 2; ++side ) {
    const Index begin = meshes[side].element_offsets[elements[side]];
    const Index end = meshes[side].element_offsets[elements[side] + 1];
    const Real scale = ( side == 0 ? 1.0 : -1.0 ) / static_cast<Real>( end - begin );
    for ( Index local_node = begin; local_node < end; ++local_node ) {
      const Index node = meshes[side].connectivity[local_node];
      for ( int component = 0; component < meshes[side].dimension; ++component ) {
        fields[side]( node, component ) += scale * force[component];
      }
    }
  }
}

template <typename Scalar>
inline void scatterPairForce( const SurfacePairViewT<Scalar>& surfaces, ElementPair pair,
                              const std::array<Scalar, 3>& force,
                              const constraint::NormalConstraintSampleT<Scalar>& sample,
                              ContactResidualViewT<Scalar> residual )
{
  const SurfaceMeshViewT<Scalar> meshes[2] = { surfaces.mortar, surfaces.nonmortar };
  const Index elements[2] = { pair.mortar_element, pair.nonmortar_element };
  const basis::ShapeValuesT<Scalar> shapes[2] = { sample.mortar_trial, sample.nonmortar_trial };
  FieldView<Scalar> fields[2] = { residual.mortar, residual.nonmortar };
  for ( int side = 0; side < 2; ++side ) {
    const Index begin = meshes[side].element_offsets[elements[side]];
    const Scalar sign = side == 0 ? Scalar{ 1.0 } : Scalar{ -1.0 };
    for ( int local_node = 0; local_node < shapes[side].size; ++local_node ) {
      const Index node = meshes[side].connectivity[begin + local_node];
      for ( int component = 0; component < meshes[side].dimension; ++component ) {
        fields[side]( node, component ) += sign * shapes[side][local_node] * force[component];
      }
    }
  }
}

}  // namespace pointwise_penalty_detail

template <PointwisePenaltyMethod MethodType, typename Scalar>
inline EvaluationSummary addPointwisePenaltyResidual( const SurfacePairViewT<Scalar>& surfaces,
                                                      ArrayView<const ElementPair> interactions,
                                                      const typename MethodType::Parameters& parameters,
                                                      const ContactStateViewT<Scalar>& state,
                                                      ContactOutputViewT<Scalar> output )
{
  const auto residual = output.residual;
  pointwise_penalty_detail::requireVectorField( residual.mortar, surfaces.mortar, "Invalid mortar residual field." );
  pointwise_penalty_detail::requireVectorField( residual.nonmortar, surfaces.nonmortar,
                                                "Invalid nonmortar residual field." );
  if ( !output.pressure.empty() && output.tributary_area.empty() ) {
    throw std::invalid_argument( "Pressure output requires a tributary-area work array." );
  }
  using Geometry = typename MethodType::geometry_policy;
  using Normal = typename Geometry::normal_policy;
  using Enforcement = typename MethodType::enforcement_policy;
  using Rate = typename Enforcement::rate_policy;
  using Response = typename MethodType::response_policy;

  if constexpr ( !std::same_as<Rate, rate::None> || std::same_as<Response, response::ViscousTangential> ) {
    pointwise_penalty_detail::requireVectorField( state.mortar_velocity, surfaces.mortar,
                                                  "Pointwise rate response requires velocity." );
    pointwise_penalty_detail::requireVectorField( state.nonmortar_velocity, surfaces.nonmortar,
                                                  "Pointwise rate response requires velocity." );
  }
  if constexpr ( std::same_as<Response, response::TiedNormal> || std::same_as<Response, response::TiedFull> ) {
    pointwise_penalty_detail::requireVectorField( state.mortar_reference_coordinates, surfaces.mortar,
                                                  "Tied response requires reference coordinates." );
    pointwise_penalty_detail::requireVectorField( state.nonmortar_reference_coordinates, surfaces.nonmortar,
                                                  "Tied response requires reference coordinates." );
  }

  EvaluationSummary summary;
  Index quadrature_offset{};
  for ( const ElementPair pair : interactions ) {
    const auto geometry = projectedOverlap<Normal>( surfaces, pair, parameters.geometry );
    if ( !geometry.valid ) {
      continue;
    }
    const Scalar effective_gap = geometry.gap - parameters.constraint.activation.residual_gap;
    if ( constraint::exceedsPenetrationLimit( parameters.constraint.activation, state, pair, effective_gap ) ) {
      continue;
    }
    const auto sample = constraint::stageNormalConstraint<constraint::Pointwise>(
        surfaces, pair, geometry.mortar_point, geometry.nonmortar_point, geometry.normal, geometry.measure );

    const Scalar stiffness =
        pointwise_penalty_detail::pointwisePenaltyStiffness<Enforcement>( parameters.enforcement, state, pair );
    const Real primal_stiffness = linearization_detail::primal( stiffness );
    if ( primal_stiffness < 0.0 ) {
      throw std::invalid_argument( "Penalty stiffness cannot be negative." );
    }
    std::array<Scalar, 3> traction{};
    Scalar potential_density{};

    if constexpr ( std::same_as<Response, response::TiedNormal> || std::same_as<Response, response::TiedFull> ) {
      const auto mortar_coordinates = pointwise_penalty_detail::interpolateField(
          surfaces.mortar, pair.mortar_element, surfaces.mortar.coordinates, sample.mortar_trial );
      const auto nonmortar_coordinates = pointwise_penalty_detail::interpolateField(
          surfaces.nonmortar, pair.nonmortar_element, surfaces.nonmortar.coordinates, sample.nonmortar_trial );
      const auto mortar_reference = pointwise_penalty_detail::interpolateField(
          surfaces.mortar, pair.mortar_element, state.mortar_reference_coordinates, sample.mortar_trial );
      const auto nonmortar_reference = pointwise_penalty_detail::interpolateField(
          surfaces.nonmortar, pair.nonmortar_element, state.nonmortar_reference_coordinates, sample.nonmortar_trial );
      const auto displacement_jump = pointwise_penalty_detail::difference(
          pointwise_penalty_detail::difference( mortar_coordinates, mortar_reference ),
          pointwise_penalty_detail::difference( nonmortar_coordinates, nonmortar_reference ) );
      if constexpr ( std::same_as<Response, response::TiedFull> ) {
        for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
          traction[component] = stiffness * displacement_jump[component];
          potential_density += 0.5 * stiffness * displacement_jump[component] * displacement_jump[component];
        }
      } else {
        const Scalar normal_displacement =
            pointwise_penalty_detail::dot( displacement_jump, geometry.normal, surfaces.mortar.dimension );
        potential_density = 0.5 * stiffness * normal_displacement * normal_displacement;
        for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
          traction[component] = stiffness * normal_displacement * geometry.normal[component];
        }
      }
    } else {
      const Scalar active_gap =
          linearization_detail::primal( effective_gap ) <= parameters.constraint.activation.gap_tolerance
              ? effective_gap
              : Scalar{};
      potential_density = 0.5 * stiffness * active_gap * active_gap;
      for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
        traction[component] = stiffness * active_gap * geometry.normal[component];
      }
    }

    if constexpr ( !std::same_as<Rate, rate::None> || std::same_as<Response, response::ViscousTangential> ) {
      const auto mortar_velocity = pointwise_penalty_detail::interpolateField(
          surfaces.mortar, pair.mortar_element, state.mortar_velocity, sample.mortar_trial );
      const auto nonmortar_velocity = pointwise_penalty_detail::interpolateField(
          surfaces.nonmortar, pair.nonmortar_element, state.nonmortar_velocity, sample.nonmortar_trial );
      const auto relative_velocity = pointwise_penalty_detail::difference( mortar_velocity, nonmortar_velocity );
      const Scalar normal_rate =
          pointwise_penalty_detail::dot( relative_velocity, geometry.normal, surfaces.mortar.dimension );
      const Scalar active_rate = linearization_detail::primal( normal_rate ) < 0.0 ? normal_rate : Scalar{};
      const Scalar rate_traction =
          pointwise_penalty_detail::rateCoefficient<Enforcement>( parameters.enforcement, stiffness ) * active_rate;
      for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
        traction[component] += rate_traction * geometry.normal[component];
      }
      if constexpr ( std::same_as<Response, response::ViscousTangential> ) {
        for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
          traction[component] +=
              parameters.response.damping * ( relative_velocity[component] - normal_rate * geometry.normal[component] );
        }
      }
    }

    std::array<Scalar, 3> force{};
    Scalar normal_pressure{};
    for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
      force[component] = geometry.measure * traction[component];
      normal_pressure += traction[component] * geometry.normal[component];
    }
    pointwise_penalty_detail::scatterPairForce( surfaces, pair, force, sample, residual );
    for ( int local_node = 0; local_node < sample.mortar_test.size; ++local_node ) {
      const Index node =
          surfaces.mortar.connectivity[surfaces.mortar.element_offsets[pair.mortar_element] + local_node];
      const Scalar weight = geometry.measure * sample.mortar_test[local_node];
      if ( !output.weighted_gap.empty() ) {
        output.weighted_gap[node] += weight * effective_gap;
      }
      if ( !output.tributary_area.empty() ) {
        output.tributary_area[node] += weight;
      }
      if ( !output.pressure.empty() ) {
        output.pressure[node] += weight * normal_pressure;
      }
      if ( !output.mortar_weights.empty() ) {
        for ( int nonmortar_node = 0; nonmortar_node < sample.nonmortar_trial.size; ++nonmortar_node ) {
          const Index trial =
              surfaces.nonmortar
                  .connectivity[surfaces.nonmortar.element_offsets[pair.nonmortar_element] + nonmortar_node];
          output.mortar_weights[node * surfaces.nonmortar.numberOfNodes() + trial] +=
              weight * sample.nonmortar_trial[nonmortar_node];
        }
      }
      if ( !output.mortar_mass_weights.empty() ) {
        for ( int trial_node = 0; trial_node < sample.mortar_trial.size; ++trial_node ) {
          const Index trial =
              surfaces.mortar.connectivity[surfaces.mortar.element_offsets[pair.mortar_element] + trial_node];
          output.mortar_mass_weights[node * surfaces.mortar.numberOfNodes() + trial] +=
              weight * sample.mortar_trial[trial_node];
        }
      }
    }
    if ( !output.quadrature_gap.empty() ) {
      output.quadrature_gap[quadrature_offset] = effective_gap;
    }
    if ( !output.quadrature_pressure.empty() ) {
      output.quadrature_pressure[quadrature_offset] = normal_pressure;
    }
    summary.energy += linearization_detail::primal( geometry.measure * potential_density );
    if ( primal_stiffness > 0.0 ) {
      summary.timestep_vote = std::min( summary.timestep_vote, 1.0 / std::sqrt( primal_stiffness ) );
    }
    ++summary.active_interactions;
    ++summary.quadrature_points;
    ++quadrature_offset;
  }
  if ( !output.gap.empty() || !output.pressure.empty() ) {
    for ( Index node = 0; node < surfaces.mortar.numberOfNodes(); ++node ) {
      const Scalar area = output.tributary_area[node];
      if ( linearization_detail::absolute( linearization_detail::primal( area ) ) > 1.0e-28 ) {
        if ( !output.gap.empty() ) {
          output.gap[node] = output.weighted_gap[node] / area;
        }
        if ( !output.pressure.empty() ) {
          output.pressure[node] /= area;
        }
      }
    }
  }
  return summary;
}

}  // namespace tribol

#endif
