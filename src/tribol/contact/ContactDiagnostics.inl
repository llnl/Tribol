[[nodiscard]] diagnostics::RankProductionDiagnostics productionDiagnostics(
    const ContactStateView& state, Real time_step, int local_rank = 0, int ranks = 1,
    diagnostics::QuadratureSamplingOptions sampling = {} ) const
  requires( detail::is_native_explicit_method_v<MethodType> && !execution::DevicePolicy<Execution> )
{
  if ( !native_evaluation_current_ || native_evaluation_generations_ != generations() || !native_rows_ ||
       !native_cells_ ) {
    throw std::logic_error( "Production diagnostics require a current native contact evaluation." );
  }
  ParentTraceFieldPairView fields{ .mortar_coordinates = parent_fields_.mortar_coordinates,
                                   .nonmortar_coordinates = parent_fields_.nonmortar_coordinates };
  if ( sampling.include_normal_rates ) {
    fields.mortar_velocity = state.mortar_velocity;
    fields.nonmortar_velocity = state.nonmortar_velocity;
  }
  diagnostics::sampleQuadrature( parent_traces_, native_cells_->view(), fields, native_diagnostic_quadrature_,
                                 sampling );
  native_diagnostic_quadrature_current_ = true;
  const auto rows = native_rows_->view();
  return diagnostics::summarizeProductionDiagnostics(
      rows,
      { .effective_gap = { result_quadrature_gap_.data(), rows.numberOfRows() },
        .pressure = { result_quadrature_pressure_.data(), rows.numberOfRows() },
        .row_force = { native_row_force_.data(), rows.numberOfRows() },
        .row_stiffness = { native_row_stiffness_.data(), rows.numberOfRows() },
        .row_rate_coefficient = { native_row_rate_.data(), rows.numberOfRows() } },
      native_diagnostic_quadrature_.view(), time_step, local_rank, ranks );
}

[[nodiscard]] diagnostics::QuadratureSampleView diagnosticQuadrature() const
  requires( detail::is_native_explicit_method_v<MethodType> && !execution::DevicePolicy<Execution> )
{
  if ( !native_evaluation_current_ || native_evaluation_generations_ != generations() ||
       !native_diagnostic_quadrature_current_ ) {
    throw std::logic_error( "Diagnostic quadrature requires current production diagnostics." );
  }
  return native_diagnostic_quadrature_.view();
}

[[nodiscard]] diagnostics::RankProductionDiagnostics productionDiagnostics(
    const ContactStateView& state, Real time_step, int local_rank = 0, int ranks = 1,
    diagnostics::QuadratureSamplingOptions sampling = {} ) const
  requires( detail::is_native_explicit_method_v<MethodType> && execution::DevicePolicy<Execution> )
{
  if ( !native_evaluation_current_ || native_evaluation_generations_ != generations() ) {
    throw std::logic_error( "Production diagnostics require a current native contact evaluation." );
  }
  ParentTraceFieldPairView fields{ .mortar_coordinates = parent_fields_.mortar_coordinates,
                                   .nonmortar_coordinates = parent_fields_.nonmortar_coordinates };
  if ( sampling.include_normal_rates ) {
    fields.mortar_velocity = state.mortar_velocity;
    fields.nonmortar_velocity = state.nonmortar_velocity;
  }
  return native_device_workspace_.productionDiagnostics( fields, time_step, local_rank, ranks, sampling );
}

[[nodiscard]] diagnostics::QuadratureSampleView diagnosticQuadrature() const
  requires( detail::is_native_explicit_method_v<MethodType> && execution::DevicePolicy<Execution> )
{
  if ( !native_evaluation_current_ || native_evaluation_generations_ != generations() ) {
    throw std::logic_error( "Diagnostic quadrature requires current production diagnostics." );
  }
  return native_device_workspace_.deviceDiagnosticQuadrature();
}
