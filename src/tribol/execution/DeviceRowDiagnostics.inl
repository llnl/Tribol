void configureDiagnosticWorkspace( Index candidates )
{
  const Index axis_points = integration::automaticAxisPointCount( integration::maximumAutomaticExactness );
  const Index samples_per_candidate =
      layout_.dimension == 2 ? axis_points : ( maximumOverlapVertices - 2 ) * axis_points * axis_points;
  const Index sample_capacity = candidates * samples_per_candidate;
  diagnostic_sample_counts_.resize( static_cast<std::size_t>( candidates + 1 ) );
  diagnostic_samples_.reserve( layout_.dimension, candidates, sample_capacity );
  row_diagnostic_entries_.resize( static_cast<std::size_t>( rows_.rows ) );
  row_diagnostic_result_.resize( 1 );
  quadrature_diagnostic_entries_.resize( static_cast<std::size_t>( sample_capacity ) );
  quadrature_diagnostic_result_.resize( 1 );

  std::size_t bytes = temporary_storage_.size();
  bytes = std::max(
      bytes, ActiveDeviceBackend::exclusiveSumTemporaryBytes(
                 diagnostic_sample_counts_.data(), diagnostic_samples_.cell_sample_offsets.data(), candidates + 1 ) );
  if ( rows_.rows > 0 ) {
    bytes = std::max( bytes, ActiveDeviceBackend::reduceTemporaryBytes(
                                 row_diagnostic_entries_.data(), row_diagnostic_result_.data(), rows_.rows,
                                 DeviceRowDiagnosticReduction{}, DeviceRowDiagnosticPair{} ) );
  }
  if ( sample_capacity > 0 ) {
    bytes = std::max( bytes,
                      ActiveDeviceBackend::reduceTemporaryBytes(
                          quadrature_diagnostic_entries_.data(), quadrature_diagnostic_result_.data(), sample_capacity,
                          DeviceQuadratureDiagnosticReduction{}, DeviceQuadratureDiagnosticPair{} ) );
  }
  temporary_storage_.resize( bytes );
}

[[nodiscard]] diagnostics::QuadratureSampleView prepareDiagnosticSamples(
    ParentTraceFieldPairView fields, diagnostics::QuadratureSamplingOptions options )
{
  if ( options.exactness < -1 || options.exactness > integration::maximumAutomaticExactness ) {
    throw std::invalid_argument( "Device diagnostic quadrature exactness exceeds the built-in range." );
  }
  if ( options.include_normal_rates ) {
    validateNativeFields( fields, layout_, true );
    native_fields_.uploadVelocity( fields, true );
  }
  const Index candidates = static_cast<Index>( native_candidates_.size() );
  const ParentTracePairView traces = native_traces_.view();
  const InteractionPatch* patches = native_patches_.data();
  const Index* mortar_faces = native_mortar_parent_faces_.data();
  const Index* nonmortar_faces = native_nonmortar_parent_faces_.data();
  Index* counts = diagnostic_sample_counts_.data();
  launchRowKernel( candidates + 1, [=] RAJA_DEVICE( Index candidate ) {
    countDeviceDiagnosticSamples( candidate, traces, candidates, patches, mortar_faces, nonmortar_faces, options,
                                  counts );
  } );
  exclusiveScan( diagnostic_sample_counts_, diagnostic_samples_.cell_sample_offsets );
  const Index samples = readLast( diagnostic_samples_.cell_sample_offsets );
  diagnostic_samples_.resize( samples, options.include_normal_rates );

  DeviceDiagnosticSampleArrays output{
      .contact_pair_ids = diagnostic_samples_.contact_pair_ids.data(),
      .cell_ids = diagnostic_samples_.cell_ids.data(),
      .owner_ranks = diagnostic_samples_.owner_ranks.data(),
      .overlap_cells = diagnostic_samples_.overlap_cells.data(),
      .normals = diagnostic_samples_.normals.data(),
      .integration_weights = diagnostic_samples_.integration_weights.data(),
      .mortar_parent_reference_coordinates = diagnostic_samples_.mortar_parent_reference_coordinates.data(),
      .nonmortar_parent_reference_coordinates = diagnostic_samples_.nonmortar_parent_reference_coordinates.data(),
      .physical_gaps = diagnostic_samples_.physical_gaps.data(),
      .normal_rates = options.include_normal_rates ? diagnostic_samples_.normal_rates.data() : nullptr,
  };
  const ParentTraceFieldPairView device_fields = native_fields_.view();
  const int* owners = native_owner_ranks_.data();
  const Real* mortar_reference = native_mortar_patch_reference_.data();
  const Real* nonmortar_reference = native_nonmortar_patch_reference_.data();
  const Index* offsets = diagnostic_samples_.cell_sample_offsets.data();
  const auto overlap = native_options_.overlap;
  launchRowKernel( candidates, [=] RAJA_DEVICE( Index candidate ) {
    fillDeviceDiagnosticSamples( candidate, traces, device_fields, candidates, patches, mortar_faces, nonmortar_faces,
                                 owners, mortar_reference, nonmortar_reference, offsets, options, overlap, output );
  } );
  return diagnostic_samples_.view();
}

[[nodiscard]] diagnostics::RankProductionDiagnostics productionDiagnostics(
    ParentTraceFieldPairView fields, Real time_step, int local_rank, int ranks,
    diagnostics::QuadratureSamplingOptions sampling )
{
  requirePrepared();
  if ( !native_prepared_ || !evaluated_ ) {
    throw std::logic_error( "Device production diagnostics require a current native row evaluation." );
  }
  if ( !std::isfinite( time_step ) || time_step < 0.0 || local_rank < 0 || ranks <= 0 || local_rank >= ranks ) {
    throw std::invalid_argument( "Device production diagnostics require a valid timestep and rank." );
  }

  const ConstraintRowBatchView rows = rows_.view();
  const DevicePenaltyRowOutputView row_output{
      .effective_gap = constDeviceView( effective_gap_ ),
      .pressure = constDeviceView( pressure_ ),
      .row_force = constDeviceView( row_force_ ),
      .row_stiffness = constDeviceView( row_stiffness_ ),
      .row_rate_coefficient = constDeviceView( row_rate_ ),
  };
  DeviceRowDiagnosticPair row_diagnostics;
  if ( rows.numberOfRows() > 0 ) {
    DeviceRowDiagnosticPair* entries = row_diagnostic_entries_.data();
    launchRowKernel( rows.numberOfRows(), [=] RAJA_DEVICE( Index row ) {
      entries[row] = makeDeviceRowDiagnostic( rows, row_output, row, time_step, local_rank );
    } );
    const std::size_t bytes =
        ActiveDeviceBackend::reduceTemporaryBytes( entries, row_diagnostic_result_.data(), rows.numberOfRows(),
                                                   DeviceRowDiagnosticReduction{}, DeviceRowDiagnosticPair{} );
    ActiveDeviceBackend::reduce( temporary_storage_.data(), bytes, entries, row_diagnostic_result_.data(),
                                 rows.numberOfRows(), DeviceRowDiagnosticReduction{}, DeviceRowDiagnosticPair{} );
    ActiveDeviceBackend::copy( &row_diagnostics, row_diagnostic_result_.data(), sizeof( row_diagnostics ) );
  } else {
    row_diagnostics.local.minimum_gap = 0.0;
    row_diagnostics.owned.minimum_gap = 0.0;
  }
  if ( row_diagnostics.owned.rows == 0 ) {
    row_diagnostics.owned.minimum_gap = 0.0;
  }

  const auto samples = prepareDiagnosticSamples( fields, sampling );
  DeviceQuadratureDiagnosticPair quadrature_diagnostics;
  if ( samples.numberOfSamples() > 0 ) {
    DeviceQuadratureDiagnosticPair* entries = quadrature_diagnostic_entries_.data();
    launchRowKernel( samples.numberOfSamples(), [=] RAJA_DEVICE( Index sample ) {
      entries[sample] = makeDeviceQuadratureDiagnostic( samples, sample, local_rank );
    } );
    const std::size_t bytes = ActiveDeviceBackend::reduceTemporaryBytes(
        entries, quadrature_diagnostic_result_.data(), samples.numberOfSamples(), DeviceQuadratureDiagnosticReduction{},
        DeviceQuadratureDiagnosticPair{} );
    ActiveDeviceBackend::reduce( temporary_storage_.data(), bytes, entries, quadrature_diagnostic_result_.data(),
                                 samples.numberOfSamples(), DeviceQuadratureDiagnosticReduction{},
                                 DeviceQuadratureDiagnosticPair{} );
    ActiveDeviceBackend::copy( &quadrature_diagnostics, quadrature_diagnostic_result_.data(),
                               sizeof( quadrature_diagnostics ) );
  }
  quadrature_diagnostics.local.owned_samples = quadrature_diagnostics.owned.owned_samples;
  diagnostics_current_ = true;
  return { .rank = local_rank,
           .ranks = ranks,
           .local = { .rows = row_diagnostics.local, .quadrature = quadrature_diagnostics.local },
           .owned = { .rows = row_diagnostics.owned, .quadrature = quadrature_diagnostics.owned } };
}

[[nodiscard]] diagnostics::QuadratureSampleView deviceDiagnosticQuadrature() const
{
  if ( !diagnostics_current_ ) {
    throw std::logic_error( "Device diagnostic quadrature requires current production diagnostics." );
  }
  return diagnostic_samples_.view();
}
