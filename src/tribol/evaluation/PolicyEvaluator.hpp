#ifndef TRIBOL_EVALUATION_POLICYEVALUATOR_HPP_
#define TRIBOL_EVALUATION_POLICYEVALUATOR_HPP_

#include "tribol/core/StaticGradient.hpp"
#include "tribol/constraint/NormalConstraint.hpp"
#include "tribol/evaluation/PointwisePenalty.hpp"
#include "tribol/geom/ConformingOverlap.hpp"
#include "tribol/geom/SmoothedSegmentOverlap.hpp"
#include "tribol/integration/InteractionQuadrature.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace tribol {

namespace policy_evaluator_detail {

template <typename Scalar>
inline bool hasField( const FieldView<Scalar>& field )
{
  return !field.values.empty();
}

template <typename ResidualScalar, typename CoordinateScalar>
inline void requireResidualField( const FieldView<ResidualScalar>& field,
                                  const SurfaceMeshViewT<CoordinateScalar>& mesh, const char* message )
{
  if ( hasField( field ) && ( !field.isStructurallyValid() || field.entities != mesh.numberOfNodes() ||
                              field.components != mesh.dimension ) ) {
    throw std::invalid_argument( message );
  }
}

template <typename Scalar>
inline void requireArraySize( ArrayView<Scalar> values, Index size, const char* message )
{
  if ( !values.empty() && ( !values.isStructurallyValid() || values.size() < size ) ) {
    throw std::invalid_argument( message );
  }
}

template <typename Scalar>
inline void requireStateArray( ArrayView<const Scalar> values, Index size, const char* message )
{
  if ( !values.isStructurallyValid() || values.size() != size ) {
    throw std::invalid_argument( message );
  }
}

template <typename Geometry, typename Integration, typename Scalar>
inline InteractionPatchT<Scalar> makePatch( const SurfacePairViewT<Scalar>& surfaces, ElementPair pair,
                                            const typename Geometry::Parameters& geometry_parameters,
                                            const typename Integration::Parameters& integration_parameters )
{
  if constexpr ( detail::is_projected_overlap_v<Geometry> ) {
    if constexpr ( detail::is_smoothed_segment_integration_v<Integration> ) {
      return smoothedProjectedSegmentPatch<typename Geometry::normal_policy, Integration>(
          surfaces, pair, geometry_parameters, integration_parameters );
    } else {
      return projectedOverlapPatch<typename Geometry::normal_policy>( surfaces, pair, geometry_parameters );
    }
  } else {
    return conformingOverlapPatch( surfaces, pair, geometry_parameters );
  }
}

template <typename Integration, typename Scalar>
inline integration::InteractionQuadratureRuleT<Scalar> makeQuadrature( const InteractionPatchT<Scalar>& patch )
{
  return integration::interactionQuadrature( patch, Integration{} );
}

template <typename Scalar>
inline Index globalNode( const SurfaceMeshViewT<Scalar>& mesh, Index element, int local_node )
{
  return mesh.connectivity[mesh.element_offsets[element] + local_node];
}

template <typename Scalar>
inline void addForce( const SurfacePairViewT<Scalar>& surfaces, ElementPair pair,
                      const constraint::NormalConstraintSampleT<Scalar>& sample, Scalar pressure,
                      ContactResidualViewT<Scalar> residual )
{
  if ( hasField( residual.mortar ) ) {
    for ( int local_node = 0; local_node < sample.mortar_trial.size; ++local_node ) {
      const Index node = globalNode( surfaces.mortar, pair.mortar_element, local_node );
      for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
        residual.mortar( node, component ) +=
            sample.weight * pressure * sample.mortar_trial[local_node] * sample.normal[component];
      }
    }
  }
  if ( hasField( residual.nonmortar ) ) {
    for ( int local_node = 0; local_node < sample.nonmortar_trial.size; ++local_node ) {
      const Index node = globalNode( surfaces.nonmortar, pair.nonmortar_element, local_node );
      for ( int component = 0; component < surfaces.nonmortar.dimension; ++component ) {
        residual.nonmortar( node, component ) -=
            sample.weight * pressure * sample.nonmortar_trial[local_node] * sample.normal[component];
      }
    }
  }
}

template <typename Scalar>
inline void addDiagnostics( const SurfacePairViewT<Scalar>& surfaces, ElementPair pair,
                            const constraint::NormalConstraintSampleT<Scalar>& sample, Scalar effective_gap,
                            ContactOutputViewT<Scalar> output )
{
  for ( int local_node = 0; local_node < sample.mortar_test.size; ++local_node ) {
    const Index node = globalNode( surfaces.mortar, pair.mortar_element, local_node );
    if ( !output.weighted_gap.empty() ) {
      output.weighted_gap[node] += sample.weight * sample.mortar_test[local_node] * effective_gap;
    }
    if ( !output.tributary_area.empty() ) {
      output.tributary_area[node] += sample.weight * sample.mortar_test[local_node];
    }
    if ( !output.mortar_weights.empty() ) {
      for ( int nonmortar_node = 0; nonmortar_node < sample.nonmortar_trial.size; ++nonmortar_node ) {
        const Index trial = globalNode( surfaces.nonmortar, pair.nonmortar_element, nonmortar_node );
        output.mortar_weights[node * surfaces.nonmortar.numberOfNodes() + trial] +=
            sample.weight * sample.mortar_test[local_node] * sample.nonmortar_trial[nonmortar_node];
      }
    }
    if ( !output.mortar_mass_weights.empty() ) {
      for ( int trial_node = 0; trial_node < sample.mortar_trial.size; ++trial_node ) {
        const Index trial = globalNode( surfaces.mortar, pair.mortar_element, trial_node );
        output.mortar_mass_weights[node * surfaces.mortar.numberOfNodes() + trial] +=
            sample.weight * sample.mortar_trial[local_node] * sample.mortar_trial[trial_node];
      }
    }
  }
}

template <typename Constraint, typename Scalar>
inline Scalar nodalPressure( const SurfaceMeshViewT<Scalar>& mesh, Index element,
                             const constraint::NormalConstraintSampleT<Scalar>& sample,
                             const std::array<Scalar, basis::maximumLinearNodes>& nodal_values )
{
  Scalar pressure{};
  for ( int local_node = 0; local_node < sample.mortar_trial.size; ++local_node ) {
    (void)mesh;
    (void)element;
    pressure += sample.mortar_trial[local_node] * nodal_values[local_node];
  }
  return pressure;
}

template <typename Scalar>
inline void finalizeGap( ContactOutputViewT<Scalar> output )
{
  if ( output.gap.empty() ) {
    return;
  }
  for ( Index node = 0; node < output.gap.size(); ++node ) {
    output.gap[node] =
        linearization_detail::absolute( linearization_detail::primal( output.tributary_area[node] ) ) > 1.0e-28
            ? output.weighted_gap[node] / output.tributary_area[node]
            : Scalar{};
  }
}

template <SupportedMethod MethodType, typename Scalar>
inline void validateEvaluation( const SurfacePairViewT<Scalar>& surfaces, const ContactStateViewT<Scalar>& state,
                                ContactOutputViewT<Scalar> output )
{
  using Enforcement = typename MethodType::enforcement_policy;
  requireResidualField( output.residual.mortar, surfaces.mortar,
                        "Mortar residual must match the mortar coordinate field." );
  requireResidualField( output.residual.nonmortar, surfaces.nonmortar,
                        "Nonmortar residual must match the nonmortar coordinate field." );
  requireArraySize( output.residual.constraint, surfaces.mortar.numberOfNodes(),
                    "Constraint residual must have one entry per mortar node." );
  requireArraySize( output.gap, surfaces.mortar.numberOfNodes(), "Gap output must have one entry per mortar node." );
  requireArraySize( output.weighted_gap, surfaces.mortar.numberOfNodes(),
                    "Weighted-gap output must have one entry per mortar node." );
  requireArraySize( output.tributary_area, surfaces.mortar.numberOfNodes(),
                    "Tributary-area output must have one entry per mortar node." );
  requireArraySize( output.mortar_weights, surfaces.mortar.numberOfNodes() * surfaces.nonmortar.numberOfNodes(),
                    "Mortar-weight output must hold the dense mortar-by-nonmortar matrix." );
  requireArraySize( output.mortar_mass_weights, surfaces.mortar.numberOfNodes() * surfaces.mortar.numberOfNodes(),
                    "Mortar-mass output must hold the dense mortar-by-mortar matrix." );
  if ( !output.gap.empty() && ( output.weighted_gap.empty() || output.tributary_area.empty() ) ) {
    throw std::invalid_argument( "Gap output requires weighted-gap and tributary-area work arrays." );
  }
  if ( !output.pressure.empty() && output.tributary_area.empty() ) {
    throw std::invalid_argument( "Pressure output requires a tributary-area work array." );
  }
  if constexpr ( std::same_as<Enforcement, enforcement::LagrangeMultiplier> ) {
    requireStateArray( state.multiplier, surfaces.mortar.numberOfNodes(),
                       "Multiplier enforcement requires one value per mortar node." );
  }
  if constexpr ( std::same_as<Enforcement, enforcement::ExternalPressure> ) {
    requireStateArray( state.external_pressure, surfaces.mortar.numberOfNodes(),
                       "External pressure requires one value per mortar node." );
    const bool has_potential = !state.external_potential_density.empty();
    const bool has_tangent = !state.external_pressure_tangent.empty();
    if ( has_potential != has_tangent ) {
      throw std::invalid_argument(
          "External-pressure potential density and pressure tangent must either both be present or both be absent." );
    }
    if ( has_potential ) {
      requireStateArray( state.external_potential_density, surfaces.mortar.numberOfNodes(),
                         "External potential density requires one value per mortar node." );
      requireStateArray( state.external_pressure_tangent, surfaces.mortar.numberOfNodes(),
                         "External pressure tangent requires one value per mortar node." );
    }
  }
}

}  // namespace policy_evaluator_detail

#include "tribol/evaluation/VariationalEvaluator.inl"

template <SupportedMethod MethodType, typename Scalar>
EvaluationSummary addPolicyResidual( const SurfacePairViewT<Scalar>& surfaces,
                                     ArrayView<const ElementPair> interactions,
                                     const typename MethodType::Parameters& parameters,
                                     const ContactStateViewT<Scalar>& state, ContactOutputViewT<Scalar> output )
{
  using Geometry = typename MethodType::geometry_policy;
  using Integration = typename MethodType::integration_policy;
  using Constraint = typename MethodType::constraint_policy;
  using Enforcement = typename MethodType::enforcement_policy;

  policy_evaluator_detail::validateEvaluation<MethodType>( surfaces, state, output );

  EvaluationSummary summary;
  Index quadrature_offset{};
  for ( const ElementPair pair : interactions ) {
    const auto patch = policy_evaluator_detail::makePatch<Geometry, Integration>( surfaces, pair, parameters.geometry,
                                                                                  parameters.integration );
    if ( !patch.valid ) {
      continue;
    }
    const auto quadrature = policy_evaluator_detail::makeQuadrature<Integration>( patch );
    std::array<constraint::NormalConstraintSampleT<Scalar>, integration::maximumQuadraturePoints> samples{};
    std::array<Scalar, basis::maximumLinearNodes> nodal_weighted_gap{};
    std::array<Scalar, basis::maximumLinearNodes> nodal_area{};
    std::array<Scalar, basis::maximumLinearNodes> nodal_pressure{};
    for ( int point = 0; point < quadrature.size; ++point ) {
      const auto& quadrature_point = quadrature[point];
      samples[point] = constraint::stageNormalConstraint<Constraint>(
          surfaces, pair, quadrature_point.mortar_position, quadrature_point.nonmortar_position,
          quadrature_point.normal, quadrature_point.weight );
      const auto& sample = samples[point];
      const Scalar effective_gap = sample.gap - parameters.constraint.activation.residual_gap;
      for ( int local_node = 0; local_node < sample.mortar_test.size; ++local_node ) {
        nodal_weighted_gap[local_node] += sample.weight * sample.mortar_test[local_node] * effective_gap;
        nodal_area[local_node] += sample.weight * sample.mortar_test[local_node];
      }
      policy_evaluator_detail::addDiagnostics( surfaces, pair, sample, effective_gap, output );
    }
    if constexpr ( detail::is_penalty_v<Enforcement> && detail::is_nodal_constraint_v<Constraint> ) {
      const Scalar stiffness =
          pointwise_penalty_detail::penaltyStiffness<Enforcement>( parameters.enforcement, state, pair );
      for ( int local_node = 0; local_node < samples[0].mortar_test.size; ++local_node ) {
        const Scalar gap =
            linearization_detail::absolute( linearization_detail::primal( nodal_area[local_node] ) ) > 1.0e-28
                ? nodal_weighted_gap[local_node] / nodal_area[local_node]
                : Scalar{};
        nodal_pressure[local_node] =
            stiffness *
            ( linearization_detail::primal( gap ) <= parameters.constraint.activation.gap_tolerance ? gap : Scalar{} );
      }
    }

    for ( int point = 0; point < quadrature.size; ++point ) {
      const auto& sample = samples[point];
      const Scalar effective_gap = sample.gap - parameters.constraint.activation.residual_gap;
      Scalar pressure{};
      if constexpr ( detail::is_penalty_v<Enforcement> ) {
        const Scalar stiffness =
            pointwise_penalty_detail::penaltyStiffness<Enforcement>( parameters.enforcement, state, pair );
        if constexpr ( detail::is_nodal_constraint_v<Constraint> ) {
          pressure = policy_evaluator_detail::nodalPressure<Constraint>( surfaces.mortar, pair.mortar_element, sample,
                                                                         nodal_pressure );
        } else {
          pressure = stiffness *
                     ( linearization_detail::primal( effective_gap ) <= parameters.constraint.activation.gap_tolerance
                           ? effective_gap
                           : Scalar{} );
        }
        const Real primal_stiffness = linearization_detail::primal( stiffness );
        if ( primal_stiffness > 0.0 ) {
          summary.energy += linearization_detail::primal( 0.5 * sample.weight * pressure * pressure / stiffness );
          summary.timestep_vote = std::min( summary.timestep_vote, 1.0 / std::sqrt( primal_stiffness ) );
        }
      } else if constexpr ( std::same_as<Enforcement, enforcement::LagrangeMultiplier> ) {
        for ( int local_node = 0; local_node < sample.mortar_test.size; ++local_node ) {
          const Index node = policy_evaluator_detail::globalNode( surfaces.mortar, pair.mortar_element, local_node );
          pressure += sample.mortar_test[local_node] * state.multiplier[node];
          if ( !output.residual.constraint.empty() ) {
            output.residual.constraint[node] += sample.weight * sample.mortar_test[local_node] * effective_gap;
          }
        }
      } else if constexpr ( std::same_as<Enforcement, enforcement::ExternalPressure> ) {
        for ( int local_node = 0; local_node < sample.mortar_trial.size; ++local_node ) {
          const Index node = policy_evaluator_detail::globalNode( surfaces.mortar, pair.mortar_element, local_node );
          pressure += sample.mortar_trial[local_node] * state.external_pressure[node];
        }
        summary.energy += linearization_detail::primal( sample.weight * pressure * effective_gap );
      }
      if constexpr ( !std::same_as<Enforcement, enforcement::None> ) {
        policy_evaluator_detail::addForce( surfaces, pair, sample, pressure, output.residual );
      }
      if ( !output.quadrature_gap.empty() ) {
        output.quadrature_gap[quadrature_offset] = effective_gap;
      }
      if ( !output.quadrature_pressure.empty() ) {
        output.quadrature_pressure[quadrature_offset] = pressure;
      }
      if ( !output.pressure.empty() ) {
        for ( int local_node = 0; local_node < sample.mortar_test.size; ++local_node ) {
          const Index node = policy_evaluator_detail::globalNode( surfaces.mortar, pair.mortar_element, local_node );
          output.pressure[node] += sample.weight * sample.mortar_test[local_node] * pressure;
        }
      }
      ++quadrature_offset;
    }
    ++summary.active_interactions;
  }
  summary.quadrature_points = quadrature_offset;
  policy_evaluator_detail::finalizeGap( output );
  if ( !output.pressure.empty() ) {
    for ( Index node = 0; node < output.pressure.size(); ++node ) {
      output.pressure[node] =
          linearization_detail::absolute( linearization_detail::primal( output.tributary_area[node] ) ) > 1.0e-28
              ? output.pressure[node] / output.tributary_area[node]
              : Scalar{};
    }
  }
  return summary;
}

template <SupportedMethod MethodType, typename Scalar>
EvaluationSummary evaluateMethod( const SurfacePairViewT<Scalar>& surfaces, ArrayView<const ElementPair> interactions,
                                  const typename MethodType::Parameters& parameters,
                                  const ContactStateViewT<Scalar>& state, ContactOutputViewT<Scalar> output )
{
  if constexpr ( PointwisePenaltyMethod<MethodType> ) {
    return addPointwisePenaltyResidual<MethodType>( surfaces, interactions, parameters, state, output );
  } else if constexpr ( std::same_as<typename MethodType::formulation_policy, formulation::Variational> ) {
    return addVariationalResidual<MethodType>( surfaces, interactions, parameters, state, output );
  } else {
    return addPolicyResidual<MethodType>( surfaces, interactions, parameters, state, output );
  }
}

}  // namespace tribol

#endif
