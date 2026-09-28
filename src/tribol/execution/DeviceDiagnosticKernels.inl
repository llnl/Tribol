struct DeviceDiagnosticSampleArrays {
  std::uint64_t* contact_pair_ids{};
  std::uint64_t* cell_ids{};
  int* owner_ranks{};
  Index* overlap_cells{};
  Real* normals{};
  Real* integration_weights{};
  Real* mortar_parent_reference_coordinates{};
  Real* nonmortar_parent_reference_coordinates{};
  Real* physical_gaps{};
  Real* normal_rates{};
};

struct DeviceRowDiagnosticPair {
  diagnostics::ExplicitRowDiagnostics local{};
  diagnostics::ExplicitRowDiagnostics owned{};
};

struct DeviceQuadratureDiagnosticPair {
  diagnostics::QuadratureSampleSummary local{};
  diagnostics::QuadratureSampleSummary owned{};
};

struct DeviceRowDiagnosticReduction {
  [[nodiscard]] TRIBOL_HOST_DEVICE DeviceRowDiagnosticPair operator()( DeviceRowDiagnosticPair left,
                                                                       const DeviceRowDiagnosticPair& right ) const
  {
    diagnostics::accumulate( left.local, right.local );
    diagnostics::accumulate( left.owned, right.owned );
    return left;
  }
};

struct DeviceQuadratureDiagnosticReduction {
  [[nodiscard]] TRIBOL_HOST_DEVICE DeviceQuadratureDiagnosticPair
  operator()( DeviceQuadratureDiagnosticPair left, const DeviceQuadratureDiagnosticPair& right ) const
  {
    left.local.samples += right.local.samples;
    left.local.owned_samples += right.local.owned_samples;
    left.local.maximum_penetration = right.local.maximum_penetration > left.local.maximum_penetration
                                         ? right.local.maximum_penetration
                                         : left.local.maximum_penetration;
    left.local.maximum_closing_rate = right.local.maximum_closing_rate > left.local.maximum_closing_rate
                                          ? right.local.maximum_closing_rate
                                          : left.local.maximum_closing_rate;
    left.owned.samples += right.owned.samples;
    left.owned.owned_samples += right.owned.owned_samples;
    left.owned.maximum_penetration = right.owned.maximum_penetration > left.owned.maximum_penetration
                                         ? right.owned.maximum_penetration
                                         : left.owned.maximum_penetration;
    left.owned.maximum_closing_rate = right.owned.maximum_closing_rate > left.owned.maximum_closing_rate
                                          ? right.owned.maximum_closing_rate
                                          : left.owned.maximum_closing_rate;
    return left;
  }
};

[[nodiscard]] TRIBOL_HOST_DEVICE int deviceDiagnosticExactness( ParentTracePairView traces, Index mortar_face,
                                                                Index nonmortar_face,
                                                                diagnostics::QuadratureSamplingOptions options )
{
  return options.exactness >= 0
             ? options.exactness
             : integration::automaticQuadratureExactness( traces.mortar.parent_face_orders[mortar_face],
                                                          traces.nonmortar.parent_face_orders[nonmortar_face],
                                                          integration::Automatic::Parameters{} );
}

TRIBOL_HOST_DEVICE void countDeviceDiagnosticSamples( Index candidate, ParentTracePairView traces,
                                                      Index candidate_count, const InteractionPatch* patches,
                                                      const Index* mortar_parent_faces,
                                                      const Index* nonmortar_parent_faces,
                                                      diagnostics::QuadratureSamplingOptions options, Index* counts )
{
  if ( candidate > candidate_count ) {
    return;
  }
  if ( candidate == candidate_count || !patches[candidate].valid ) {
    counts[candidate] = 0;
    return;
  }
  const int exactness =
      deviceDiagnosticExactness( traces, mortar_parent_faces[candidate], nonmortar_parent_faces[candidate], options );
  const Index axis_points = integration::automaticAxisPointCount( exactness );
  counts[candidate] = traces.mortar.geometry.dimension == 2
                          ? axis_points
                          : ( patches[candidate].vertex_count - 2 ) * axis_points * axis_points;
}

TRIBOL_HOST_DEVICE void fillDeviceDiagnosticSamples(
    Index candidate, ParentTracePairView traces, ParentTraceFieldPairView fields, Index candidate_count,
    const InteractionPatch* patches, const Index* mortar_parent_faces, const Index* nonmortar_parent_faces,
    const int* owner_ranks, const Real* mortar_patch_reference, const Real* nonmortar_patch_reference,
    const Index* sample_offsets, diagnostics::QuadratureSamplingOptions options, OverlapCellBuildOptions overlap,
    DeviceDiagnosticSampleArrays output )
{
  if ( candidate >= candidate_count || !patches[candidate].valid ) {
    return;
  }
  const Index first_sample = sample_offsets[candidate];
  const Index sample_count = sample_offsets[candidate + 1] - first_sample;
  const Index mortar_face = mortar_parent_faces[candidate];
  const Index nonmortar_face = nonmortar_parent_faces[candidate];
  const int exactness = deviceDiagnosticExactness( traces, mortar_face, nonmortar_face, options );
  const Index reference_offset = nativeReferenceComponents * candidate * maximumOverlapVertices;
  const DeviceMappedPatchAccessor cell{ &patches[candidate], mortar_patch_reference + reference_offset,
                                        nonmortar_patch_reference + reference_offset };
  for ( Index local_sample = 0; local_sample < sample_count; ++local_sample ) {
    const Index sample = first_sample + local_sample;
    const auto point = integration::automaticQuadraturePointFromCell( cell, exactness, local_sample );
    output.contact_pair_ids[sample] = overlap.contact_pair_id;
    output.cell_ids[sample] = overlap.first_cell_id + static_cast<std::uint64_t>( candidate );
    output.owner_ranks[sample] = owner_ranks[candidate];
    output.overlap_cells[sample] = candidate;
    output.integration_weights[sample] = point.weight;
    for ( int component = 0; component < 3; ++component ) {
      output.normals[3 * sample + component] = point.normal[component];
    }
    for ( int component = 0; component < traces.mortar.geometry.dimension - 1; ++component ) {
      output.mortar_parent_reference_coordinates[( traces.mortar.geometry.dimension - 1 ) * sample + component] =
          component == 0 ? point.mortar_reference.first : point.mortar_reference.second;
      output.nonmortar_parent_reference_coordinates[( traces.mortar.geometry.dimension - 1 ) * sample + component] =
          component == 0 ? point.nonmortar_reference.first : point.nonmortar_reference.second;
    }
    const auto mortar_position =
        basis::interpolateParentField( traces.mortar, mortar_face, fields.mortar_coordinates, point.mortar_reference );
    const auto nonmortar_position = basis::interpolateParentField(
        traces.nonmortar, nonmortar_face, fields.nonmortar_coordinates, point.nonmortar_reference );
    Real gap{};
    for ( int component = 0; component < traces.mortar.geometry.dimension; ++component ) {
      gap += ( mortar_position[component] - nonmortar_position[component] ) * point.normal[component];
    }
    output.physical_gaps[sample] = gap;
    if ( options.include_normal_rates ) {
      const auto mortar_velocity =
          basis::interpolateParentField( traces.mortar, mortar_face, fields.mortar_velocity, point.mortar_reference );
      const auto nonmortar_velocity = basis::interpolateParentField(
          traces.nonmortar, nonmortar_face, fields.nonmortar_velocity, point.nonmortar_reference );
      Real rate{};
      for ( int component = 0; component < traces.mortar.geometry.dimension; ++component ) {
        rate += ( mortar_velocity[component] - nonmortar_velocity[component] ) * point.normal[component];
      }
      output.normal_rates[sample] = rate;
    }
  }
}

[[nodiscard]] TRIBOL_HOST_DEVICE DeviceRowDiagnosticPair makeDeviceRowDiagnostic( ConstraintRowBatchView rows,
                                                                                  DevicePenaltyRowOutputView output,
                                                                                  Index row, Real time_step,
                                                                                  int local_rank )
{
  DeviceRowDiagnosticPair result;
  result.local = diagnostics::explicitRowDiagnostic( rows, output, row, time_step, local_rank );
  if ( rows.owner_ranks[row] == local_rank ) {
    result.owned = result.local;
  }
  return result;
}

[[nodiscard]] TRIBOL_HOST_DEVICE DeviceQuadratureDiagnosticPair
makeDeviceQuadratureDiagnostic( diagnostics::QuadratureSampleView samples, Index sample, int local_rank )
{
  DeviceQuadratureDiagnosticPair result;
  const Real gap = samples.physical_gaps[sample];
  const Real rate = samples.hasNormalRates() ? samples.normal_rates[sample] : 0.0;
  result.local.samples = 1;
  result.local.owned_samples = samples.owner_ranks[sample] == local_rank ? 1 : 0;
  result.local.maximum_penetration = gap < 0.0 ? -gap : 0.0;
  result.local.maximum_closing_rate = rate < 0.0 ? -rate : 0.0;
  if ( samples.owner_ranks[sample] == local_rank ) {
    result.owned = result.local;
    result.owned.samples = 1;
    result.owned.owned_samples = 1;
  }
  return result;
}
