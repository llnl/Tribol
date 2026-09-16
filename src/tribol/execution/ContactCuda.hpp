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
                                                 CudaPenaltyWorkspace& workspace,
                                                 std::vector<InteractionPatch>& patches,
                                                 std::vector<PenaltyContribution>& contributions )
{
  pointwise_penalty_detail::requireVectorField( output.residual.mortar, surfaces.mortar,
                                                "Mortar residual does not match the mortar surface." );
  pointwise_penalty_detail::requireVectorField( output.residual.nonmortar, surfaces.nonmortar,
                                                "Nonmortar residual does not match the nonmortar surface." );
  for ( Index interaction = 0; interaction < interactions.size(); ++interaction ) {
    patches[static_cast<std::size_t>( interaction )] =
        projectedOverlapPatch<normal::MeanPlane>( surfaces, interactions[interaction], parameters.geometry );
  }
  workspace.evaluatePointwise( { patches.data(), static_cast<Index>( patches.size() ) },
                               parameters.enforcement.stiffness.value,
                               { contributions.data(), static_cast<Index>( contributions.size() ) } );
  EvaluationSummary summary;
  for ( Index interaction = 0; interaction < interactions.size(); ++interaction ) {
    const auto& contribution = contributions[static_cast<std::size_t>( interaction )];
    if ( !contribution.active ) {
      continue;
    }
    pointwise_penalty_detail::scatterPairForce( surfaces, interactions[interaction], contribution.mortar_force,
                                                output.residual );
    summary.energy += contribution.energy;
    ++summary.active_interactions;
    ++summary.quadrature_points;
  }
  if ( summary.active_interactions > 0 && parameters.enforcement.stiffness.value > 0.0 ) {
    summary.timestep_vote = 1.0 / std::sqrt( parameters.enforcement.stiffness.value );
  }
  return summary;
}

}  // namespace tribol::execution

#endif
