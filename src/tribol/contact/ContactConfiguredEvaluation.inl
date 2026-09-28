EvaluationSummary evaluateConfigured( const SurfacePairView& surfaces, const ContactStateView& state,
                                      ContactOutputView output, timestep::DiagonalInverseMassView inverse_mass = {},
                                      timestep::IntegratorStabilityLimits integrator = {} ) const
{
  if constexpr ( detail::is_native_explicit_method_v<MethodType> ) {
    native_evaluation_current_ = false;
    native_diagnostic_quadrature_current_ = false;
  }
  EvaluationSummary summary;
  if constexpr ( detail::is_native_explicit_method_v<MethodType> ) {
    summary = evaluateNativeRows( state, output, inverse_mass, integrator );
  } else if constexpr ( execution::DevicePolicy<Execution> ) {
    summary = execution_workspace_.evaluateDefault( options_.method, state, options_.timestep );
    execution_workspace_.downloadResult( output );
    last_device_summary_ = summary;
    device_result_geometry_version_ = geometry_version_;
    device_result_interaction_version_ = interaction_version_;
  } else if constexpr ( std::same_as<Execution, execution::OpenMP> ) {
    summary = execution::evaluateOpenMPContact<MethodType>( surfaces, candidates_.view(), options_.method, state,
                                                            output, execution_workspace_ );
  } else {
    summary = evaluateMethod<MethodType>( surfaces, candidates_.view(), options_.method, state, output );
  }
  if constexpr ( !execution::DevicePolicy<Execution> ) {
    if constexpr ( detail::is_native_explicit_method_v<MethodType> ) {
      const Real kinematic_vote = timestep::kinematicVote<MethodType>( parent_traces_, candidates_.view(),
                                                                       options_.method, state, options_.timestep );
      summary.timestep_vote = std::min( summary.timestep_vote, kinematic_vote );
    } else {
      summary.timestep_vote = timestep::kinematicVote<MethodType>( surfaces, candidates_.view(), options_.method, state,
                                                                   options_.timestep );
    }
  }
  if constexpr ( detail::is_native_explicit_method_v<MethodType> ) {
    native_evaluation_generations_ = generations();
    native_evaluation_current_ = true;
  }
  return summary;
}
