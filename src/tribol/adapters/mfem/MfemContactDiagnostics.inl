[[nodiscard]] diagnostics::ProductionDiagnosticsPayload productionDiagnostics(
    const ContactState& state = {}, Real time_step = 0.0, diagnostics::QuadratureSamplingOptions sampling = {} )
  requires tribol::detail::is_native_explicit_method_v<MethodType>
{
  const auto communicator = mesh_.GetComm();
  const auto communicator_info = detail::communicatorInfo( communicator );
  const auto restricted_state = restrictState( state );
  auto rank_diagnostics = core_.productionDiagnostics( restricted_state, time_step, communicator_info.rank,
                                                       communicator_info.size, sampling );
  auto global_diagnostics = detail::reduceOwnedProductionDiagnostics( communicator, rank_diagnostics.owned );
  return diagnostics::makeProductionDiagnosticsPayload( rank_diagnostics, global_diagnostics,
                                                        diagnostics::summarizeAttempts( core_.attemptLifecycle() ) );
}
