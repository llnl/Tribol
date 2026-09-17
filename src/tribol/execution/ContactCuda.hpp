#ifndef TRIBOL_EXECUTION_CONTACTCUDA_HPP_
#define TRIBOL_EXECUTION_CONTACTCUDA_HPP_

#include "tribol/evaluation/PointwisePenalty.hpp"
#include "tribol/execution/CudaPenalty.hpp"

#include <cmath>
#include <vector>

namespace tribol::execution {

template <typename MethodType, typename Execution>
concept SupportedContactExecution =
    SupportedMethod<MethodType> && std::same_as<typename MethodType::linearization_policy, linearization::Exact> &&
    Policy<Execution> && ( !std::same_as<Execution, Cuda> || std::same_as<MethodType, DefaultMethod> );

inline EvaluationSummary evaluateDefaultContact( const SurfacePairView& surfaces,
                                                 ArrayView<const ElementPair> interactions,
                                                 const DefaultMethod::Parameters& parameters, ContactOutputView output,
                                                 const ContactStateView& state, CudaPenaltyWorkspace& workspace,
                                                 std::vector<InteractionPatch>& patches,
                                                 std::vector<PenaltyContribution>& contributions )
{
  pointwise_penalty_detail::requireVectorField( output.residual.mortar, surfaces.mortar,
                                                "Mortar residual does not match the mortar surface." );
  pointwise_penalty_detail::requireVectorField( output.residual.nonmortar, surfaces.nonmortar,
                                                "Nonmortar residual does not match the nonmortar surface." );
  for ( Index interaction = 0; interaction < interactions.size(); ++interaction ) {
    auto& patch = patches[static_cast<std::size_t>( interaction )];
    patch = projectedOverlapPatch<normal::MeanPlane>( surfaces, interactions[interaction], parameters.geometry );
    if ( patch.valid ) {
      Real effective_gap = -parameters.constraint.activation.residual_gap;
      for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
        effective_gap +=
            ( patch.mortar_centroid[component] - patch.nonmortar_centroid[component] ) * patch.normal[component];
      }
      patch.valid = !constraint::exceedsPenetrationLimit( parameters.constraint.activation, state,
                                                          interactions[interaction], effective_gap );
    }
  }
  const Real mortar_stiffness = parameters.enforcement.stiffness.mortar_scale * parameters.enforcement.stiffness.value;
  const Real nonmortar_stiffness =
      parameters.enforcement.stiffness.nonmortar_scale * parameters.enforcement.stiffness.value;
  const Real stiffness = pointwise_penalty_detail::seriesStiffness( mortar_stiffness, nonmortar_stiffness );
  workspace.evaluatePointwise( { patches.data(), static_cast<Index>( patches.size() ) }, stiffness,
                               parameters.constraint.activation,
                               { contributions.data(), static_cast<Index>( contributions.size() ) } );
  EvaluationSummary summary;
  Index quadrature_offset{};
  for ( Index interaction = 0; interaction < interactions.size(); ++interaction ) {
    const auto& contribution = contributions[static_cast<std::size_t>( interaction )];
    if ( !contribution.active ) {
      continue;
    }
    const auto& patch = patches[static_cast<std::size_t>( interaction )];
    const auto sample = constraint::stageNormalConstraint<constraint::Pointwise>(
        surfaces, interactions[interaction], patch.mortar_centroid, patch.nonmortar_centroid, patch.normal,
        patch.measure );
    pointwise_penalty_detail::scatterPairForce( surfaces, interactions[interaction], contribution.mortar_force, sample,
                                                output.residual );
    for ( int local_node = 0; local_node < sample.mortar_test.size; ++local_node ) {
      const Index node =
          surfaces.mortar
              .connectivity[surfaces.mortar.element_offsets[interactions[interaction].mortar_element] + local_node];
      const Real weight = patch.measure * sample.mortar_test[local_node];
      if ( !output.weighted_gap.empty() ) {
        output.weighted_gap[node] += weight * contribution.effective_gap;
      }
      if ( !output.tributary_area.empty() ) {
        output.tributary_area[node] += weight;
      }
      if ( !output.pressure.empty() ) {
        output.pressure[node] += weight * contribution.pressure;
      }
      if ( !output.mortar_weights.empty() ) {
        for ( int nonmortar_node = 0; nonmortar_node < sample.nonmortar_trial.size; ++nonmortar_node ) {
          const Index trial =
              surfaces.nonmortar
                  .connectivity[surfaces.nonmortar.element_offsets[interactions[interaction].nonmortar_element] +
                                nonmortar_node];
          output.mortar_weights[node * surfaces.nonmortar.numberOfNodes() + trial] +=
              weight * sample.nonmortar_trial[nonmortar_node];
        }
      }
      if ( !output.mortar_mass_weights.empty() ) {
        for ( int trial_node = 0; trial_node < sample.mortar_trial.size; ++trial_node ) {
          const Index trial =
              surfaces.mortar
                  .connectivity[surfaces.mortar.element_offsets[interactions[interaction].mortar_element] + trial_node];
          output.mortar_mass_weights[node * surfaces.mortar.numberOfNodes() + trial] +=
              weight * sample.mortar_trial[trial_node];
        }
      }
    }
    if ( !output.quadrature_gap.empty() ) {
      output.quadrature_gap[quadrature_offset] = contribution.effective_gap;
    }
    if ( !output.quadrature_pressure.empty() ) {
      output.quadrature_pressure[quadrature_offset] = contribution.pressure;
    }
    summary.energy += contribution.energy;
    ++summary.active_interactions;
    ++summary.quadrature_points;
    ++quadrature_offset;
  }
  if ( !output.gap.empty() || !output.pressure.empty() ) {
    for ( Index node = 0; node < surfaces.mortar.numberOfNodes(); ++node ) {
      const Real area = output.tributary_area[node];
      if ( std::abs( area ) > 1.0e-28 ) {
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

}  // namespace tribol::execution

#endif
