EvaluationSummary addResidual( const ContactStateView& state, ContactResidualView residual ) const
{
  if ( !has_interactions_ ) {
    throw std::logic_error( "updateInteractions() must be called before contact evaluation." );
  }
  clearEvaluationWorkspace();
  if constexpr ( std::same_as<Execution, execution::Cuda> ) {
    const auto summary = evaluateConfigured( surfaces_, state, resultOutput() );
    addResultField( result_mortar_force_, surfaces_.mortar, residual.mortar );
    addResultField( result_nonmortar_force_, surfaces_.nonmortar, residual.nonmortar );
    return summary;
  } else {
    return evaluateConfigured( surfaces_, state,
                               ContactOutputView{ .residual = residual,
                                                  .gap = mutableView( result_gap_ ),
                                                  .weighted_gap = mutableView( result_weighted_gap_ ),
                                                  .tributary_area = mutableView( result_tributary_area_ ),
                                                  .mortar_weights = mutableView( result_mortar_weights_ ),
                                                  .mortar_mass_weights = mutableView( result_mortar_mass_weights_ ),
                                                  .quadrature_gap = mutableView( result_quadrature_gap_ ),
                                                  .quadrature_pressure = mutableView( result_quadrature_pressure_ ),
                                                  .pressure = mutableView( result_pressure_ ) } );
  }
}

[[nodiscard]] ContactResultView evaluate( const ContactStateView& state = {} ) const
{
  if ( !has_interactions_ ) {
    throw std::logic_error( "updateInteractions() must be called before contact evaluation." );
  }
  clearEvaluationWorkspace();
  const auto summary = evaluateConfigured( surfaces_, state, resultOutput() );
  return {
      .mortar_force = constField( result_mortar_force_, surfaces_.mortar ),
      .nonmortar_force = constField( result_nonmortar_force_, surfaces_.nonmortar ),
      .constraint_residual = constView( result_constraint_ ),
      .gap = constView( result_gap_ ),
      .weighted_gap = constView( result_weighted_gap_ ),
      .tributary_area = constView( result_tributary_area_ ),
      .mortar_weights = constView( result_mortar_weights_ ),
      .mortar_mass_weights = constView( result_mortar_mass_weights_ ),
      .quadrature_gap = { result_quadrature_gap_.data(), summary.quadrature_points },
      .quadrature_pressure = { result_quadrature_pressure_.data(), summary.quadrature_points },
      .pressure = constView( result_pressure_ ),
      .summary = summary,
      .geometry_version = geometry_version_,
      .interaction_version = interaction_version_,
  };
}

[[nodiscard]] ContactResultView evaluateDevice( const ContactStateView& state = {} ) const
  requires std::same_as<Execution, execution::Cuda>
{
  if ( !has_interactions_ ) {
    throw std::logic_error( "updateInteractions() must be called before contact evaluation." );
  }
  last_cuda_summary_ = execution_workspace_.evaluateDefault( options_.method, state, options_.timestep );
  cuda_result_geometry_version_ = geometry_version_;
  cuda_result_interaction_version_ = interaction_version_;
  return execution_workspace_.deviceResult( last_cuda_summary_, cuda_result_geometry_version_,
                                            cuda_result_interaction_version_ );
}

[[nodiscard]] execution::CudaPipelineView cudaPipelineView() const
  requires std::same_as<Execution, execution::Cuda>
{
  return execution_workspace_.pipelineView( last_cuda_summary_, cuda_result_geometry_version_,
                                            cuda_result_interaction_version_ );
}
