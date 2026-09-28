template <>
DeviceRowWorkspace<ActiveExecution>::DeviceRowWorkspace() : impl_( new Impl )
{
}

template <>
DeviceRowWorkspace<ActiveExecution>::~DeviceRowWorkspace()
{
  delete impl_;
}

template <>
void DeviceRowWorkspace<ActiveExecution>::prepare( ConstraintRowBatchView rows, DeviceRowDofLayout layout )
{
  impl_->prepare( rows, layout );
}

template <>
void DeviceRowWorkspace<ActiveExecution>::prepareNative( ParentTracePairView traces, ParentTraceFieldPairView fields,
                                                         ArrayView<const ElementPair> device_candidates,
                                                         DeviceRowDofLayout layout, DeviceNativeRowOptions options )
{
  impl_->prepareNative( traces, fields, device_candidates, layout, options );
}

template <>
void DeviceRowWorkspace<ActiveExecution>::updateNativeFields( ParentTraceFieldPairView fields )
{
  impl_->updateNativeFields( fields );
}

template <>
DeviceRowEvaluationSummary DeviceRowWorkspace<ActiveExecution>::evaluate(
    PenaltyRowInputView penalty, timestep::DiagonalInverseMassView inverse_mass,
    timestep::IntegratorStabilityLimits integrator, GeometryVersion geometry_version,
    InteractionVersion interaction_version, PenaltyVersion penalty_version, DeviceNativeKinematicInputView kinematic )
{
  return impl_->evaluate( penalty, inverse_mass, integrator, geometry_version, interaction_version, penalty_version,
                          kinematic );
}

template <>
DeviceRowResultView DeviceRowWorkspace<ActiveExecution>::deviceResult( DeviceRowEvaluationSummary summary ) const
{
  return impl_->deviceResult( summary );
}

template <>
ArrayView<const InteractionPatch> DeviceRowWorkspace<ActiveExecution>::devicePatches() const
{
  return impl_->devicePatches();
}

template <>
diagnostics::RankProductionDiagnostics DeviceRowWorkspace<ActiveExecution>::productionDiagnostics(
    ParentTraceFieldPairView fields, Real time_step, int local_rank, int ranks,
    diagnostics::QuadratureSamplingOptions sampling )
{
  return impl_->productionDiagnostics( fields, time_step, local_rank, ranks, sampling );
}

template <>
diagnostics::QuadratureSampleView DeviceRowWorkspace<ActiveExecution>::deviceDiagnosticQuadrature() const
{
  return impl_->deviceDiagnosticQuadrature();
}

template <>
void DeviceRowWorkspace<ActiveExecution>::download( ArrayView<Real> force, PenaltyRowOutputView row_output ) const
{
  impl_->download( force, row_output );
}
