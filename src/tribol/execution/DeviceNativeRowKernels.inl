inline constexpr int nativeReferenceComponents = 2;

struct DeviceMappedPatchAccessor {
  const InteractionPatch* patch{};
  const Real* mortar_reference{};
  const Real* nonmortar_reference{};

  [[nodiscard]] TRIBOL_HOST_DEVICE int dimension() const { return patch->manifold_dimension + 1; }
  [[nodiscard]] TRIBOL_HOST_DEVICE Index vertexCount() const { return patch->vertex_count; }
  [[nodiscard]] TRIBOL_HOST_DEVICE Real normal( int component ) const { return patch->normal[component]; }
  [[nodiscard]] TRIBOL_HOST_DEVICE Real measure() const { return patch->measure; }
  [[nodiscard]] TRIBOL_HOST_DEVICE Real integrationVertex( Index vertex, int component ) const
  {
    return patch->integration_vertices[vertex][component];
  }
  [[nodiscard]] TRIBOL_HOST_DEVICE Real parentReference( bool mortar, Index vertex, int component ) const
  {
    const Real* values = mortar ? mortar_reference : nonmortar_reference;
    return values[nativeReferenceComponents * vertex + component];
  }
};

struct DeviceNativeRowArrays {
  std::uint64_t* contact_pair_ids{};
  std::uint64_t* cell_ids{};
  std::uint64_t* row_ids{};
  int* owner_ranks{};
  Index* overlap_cells{};
  Index* mortar_parent_faces{};
  Index* nonmortar_parent_faces{};
  Real* normals{};
  Real* integration_weights{};
  Real* mortar_parent_reference_coordinates{};
  Real* nonmortar_parent_reference_coordinates{};
  Real* mortar_positions{};
  Real* nonmortar_positions{};
  Real* gaps{};
  Real* normal_rates{};
  Index* mortar_dof_offsets{};
  Index* mortar_dofs{};
  Real* mortar_basis_values{};
  Index* nonmortar_dof_offsets{};
  Index* nonmortar_dofs{};
  Real* nonmortar_basis_values{};
};

TRIBOL_HOST_DEVICE void rejectDevicePatch( InteractionPatch& patch, GeometryStatus status )
{
  patch = {};
  patch.status = status;
  patch.valid = false;
}

TRIBOL_HOST_DEVICE void buildDeviceMappedPatch(
    Index candidate, ParentTracePairView traces, const ElementPair* candidates, Index candidate_count,
    geometry::ProjectedOverlap<normal::MeanPlane>::Parameters geometry_parameters, OverlapCellBuildOptions options,
    InteractionPatch* patches, Index* mortar_parent_faces, Index* nonmortar_parent_faces, int* owner_ranks,
    Real* mortar_reference, Real* nonmortar_reference )
{
  if ( candidate >= candidate_count ) {
    return;
  }
  InteractionPatch& patch = patches[candidate];
  rejectDevicePatch( patch, GeometryStatus::InvalidProvenance );
  const ElementPair pair = candidates[candidate];
  if ( pair.mortar_element < 0 || pair.mortar_element >= traces.mortar.numberOfSubfacets() ||
       pair.nonmortar_element < 0 || pair.nonmortar_element >= traces.nonmortar.numberOfSubfacets() ) {
    return;
  }
  const Index mortar_face = traces.mortar.subfacet_parent_faces[pair.mortar_element];
  const Index nonmortar_face = traces.nonmortar.subfacet_parent_faces[pair.nonmortar_element];
  mortar_parent_faces[candidate] = mortar_face;
  nonmortar_parent_faces[candidate] = nonmortar_face;
  owner_ranks[candidate] = traces.mortar.parent_face_owners[mortar_face];
  const auto mortar_normal =
      traces.mortar.geometry.dimension == 2
          ? projected_overlap_detail::segmentNormal( traces.mortar.geometry, pair.mortar_element )
          : projected_overlap_detail::faceNormal( traces.mortar.geometry, pair.mortar_element );
  const auto nonmortar_normal =
      traces.nonmortar.geometry.dimension == 2
          ? projected_overlap_detail::segmentNormal( traces.nonmortar.geometry, pair.nonmortar_element )
          : projected_overlap_detail::faceNormal( traces.nonmortar.geometry, pair.nonmortar_element );
  if ( projected_overlap_detail::norm( mortar_normal ) <= 1.0e-28 ||
       projected_overlap_detail::norm( nonmortar_normal ) <= 1.0e-28 ) {
    rejectDevicePatch( patch, GeometryStatus::DegenerateOverlap );
    return;
  }
  if ( projected_overlap_detail::dot( mortar_normal, nonmortar_normal ) >= -options.orientation_tolerance ) {
    rejectDevicePatch( patch, GeometryStatus::InconsistentOrientation );
    return;
  }
  const SurfacePairView surfaces{ traces.mortar.geometry, traces.nonmortar.geometry };
  patch = projectedOverlapPatch<normal::MeanPlane>( surfaces, pair, geometry_parameters );
  for ( int vertex = 0; patch.valid && vertex < patch.vertex_count; ++vertex ) {
    const auto mortar = overlap_cell_detail::mapToParent( traces.mortar, pair.mortar_element,
                                                          patch.mortar_vertices[vertex], options.mapping_tolerance );
    const auto nonmortar = overlap_cell_detail::mapToParent(
        traces.nonmortar, pair.nonmortar_element, patch.nonmortar_vertices[vertex], options.mapping_tolerance );
    const GeometryStatus status = mortar.status != GeometryStatus::Accepted ? mortar.status : nonmortar.status;
    if ( status != GeometryStatus::Accepted ) {
      rejectDevicePatch( patch, status );
      return;
    }
    const Index offset = nativeReferenceComponents * ( candidate * maximumOverlapVertices + vertex );
    mortar_reference[offset] = mortar.parent_reference.first;
    mortar_reference[offset + 1] = mortar.parent_reference.second;
    nonmortar_reference[offset] = nonmortar.parent_reference.first;
    nonmortar_reference[offset + 1] = nonmortar.parent_reference.second;
  }
}

[[nodiscard]] TRIBOL_HOST_DEVICE bool sameDeviceMappedPatch( Index current, Index previous, ParentTracePairView traces,
                                                             const InteractionPatch* patches,
                                                             const Index* mortar_parent_faces,
                                                             const Index* nonmortar_parent_faces, Real tolerance )
{
  const InteractionPatch& patch = patches[current];
  const InteractionPatch& stored = patches[previous];
  if ( !stored.valid ||
       traces.mortar.parent_face_ids[mortar_parent_faces[previous]] !=
           traces.mortar.parent_face_ids[mortar_parent_faces[current]] ||
       traces.nonmortar.parent_face_ids[nonmortar_parent_faces[previous]] !=
           traces.nonmortar.parent_face_ids[nonmortar_parent_faces[current]] ||
       overlap_cell_detail::distance( { stored.measure, 0.0, 0.0 }, { patch.measure, 0.0, 0.0 }, 1 ) > tolerance ||
       stored.vertex_count != patch.vertex_count ) {
    return false;
  }
  for ( int vertex = 0; vertex < patch.vertex_count; ++vertex ) {
    bool found{};
    for ( int candidate = 0; candidate < stored.vertex_count; ++candidate ) {
      found = found ||
              overlap_cell_detail::distance( stored.integration_vertices[candidate], patch.integration_vertices[vertex],
                                             patch.manifold_dimension + 1 ) <= tolerance;
    }
    if ( !found ) {
      return false;
    }
  }
  return true;
}

TRIBOL_HOST_DEVICE void countDeviceNativeRows( Index candidate, ParentTracePairView traces, Index candidate_count,
                                               integration::Automatic::Parameters quadrature,
                                               OverlapCellBuildOptions overlap, InteractionPatch* patches,
                                               const Index* mortar_parent_faces, const Index* nonmortar_parent_faces,
                                               Index* row_counts, Index* mortar_entry_counts,
                                               Index* nonmortar_entry_counts )
{
  if ( candidate > candidate_count ) {
    return;
  }
  if ( candidate == candidate_count ) {
    row_counts[candidate] = 0;
    mortar_entry_counts[candidate] = 0;
    nonmortar_entry_counts[candidate] = 0;
    return;
  }
  InteractionPatch& patch = patches[candidate];
  for ( Index previous = 0; patch.valid && previous < candidate; ++previous ) {
    if ( sameDeviceMappedPatch( candidate, previous, traces, patches, mortar_parent_faces, nonmortar_parent_faces,
                                overlap.mapping_tolerance ) ) {
      patch.status = GeometryStatus::Duplicate;
      patch.valid = false;
    }
  }
  if ( !patch.valid ) {
    row_counts[candidate] = 0;
    mortar_entry_counts[candidate] = 0;
    nonmortar_entry_counts[candidate] = 0;
    return;
  }
  const Index mortar_face = mortar_parent_faces[candidate];
  const Index nonmortar_face = nonmortar_parent_faces[candidate];
  const int exactness = integration::automaticQuadratureExactness(
      traces.mortar.parent_face_orders[mortar_face], traces.nonmortar.parent_face_orders[nonmortar_face], quadrature );
  const Index axis_points = integration::automaticAxisPointCount( exactness );
  const Index rows =
      traces.mortar.geometry.dimension == 2 ? axis_points : ( patch.vertex_count - 2 ) * axis_points * axis_points;
  row_counts[candidate] = rows;
  mortar_entry_counts[candidate] = rows * parentFaceDofCount( traces.mortar.parent_face_topologies[mortar_face],
                                                              traces.mortar.parent_face_orders[mortar_face] );
  nonmortar_entry_counts[candidate] =
      rows * parentFaceDofCount( traces.nonmortar.parent_face_topologies[nonmortar_face],
                                 traces.nonmortar.parent_face_orders[nonmortar_face] );
}

TRIBOL_HOST_DEVICE void fillDeviceNativeRows(
    Index candidate, ParentTracePairView traces, ParentTraceFieldPairView fields, Index candidate_count,
    integration::Automatic::Parameters quadrature, OverlapCellBuildOptions overlap, const InteractionPatch* patches,
    const Index* mortar_parent_faces, const Index* nonmortar_parent_faces, const int* owner_ranks,
    const Real* mortar_patch_reference, const Real* nonmortar_patch_reference, const Index* row_offsets,
    const Index* mortar_entry_offsets, const Index* nonmortar_entry_offsets, std::uint64_t first_row_id,
    bool include_normal_rates, DeviceNativeRowArrays output )
{
  if ( candidate >= candidate_count || !patches[candidate].valid ) {
    return;
  }
  const Index first_row = row_offsets[candidate];
  const Index row_count = row_offsets[candidate + 1] - first_row;
  const Index mortar_face = mortar_parent_faces[candidate];
  const Index nonmortar_face = nonmortar_parent_faces[candidate];
  const int exactness = integration::automaticQuadratureExactness(
      traces.mortar.parent_face_orders[mortar_face], traces.nonmortar.parent_face_orders[nonmortar_face], quadrature );
  const Index reference_offset = nativeReferenceComponents * candidate * maximumOverlapVertices;
  const DeviceMappedPatchAccessor cell{ &patches[candidate], mortar_patch_reference + reference_offset,
                                        nonmortar_patch_reference + reference_offset };
  const Index mortar_dofs =
      traces.mortar.parent_dof_offsets[mortar_face + 1] - traces.mortar.parent_dof_offsets[mortar_face];
  const Index nonmortar_dofs =
      traces.nonmortar.parent_dof_offsets[nonmortar_face + 1] - traces.nonmortar.parent_dof_offsets[nonmortar_face];
  for ( Index local_row = 0; local_row < row_count; ++local_row ) {
    const Index row = first_row + local_row;
    const auto point = integration::automaticQuadraturePointFromCell( cell, exactness, local_row );
    output.contact_pair_ids[row] = overlap.contact_pair_id;
    output.cell_ids[row] = overlap.first_cell_id + static_cast<std::uint64_t>( candidate );
    output.row_ids[row] = first_row_id + static_cast<std::uint64_t>( row );
    output.owner_ranks[row] = owner_ranks[candidate];
    output.overlap_cells[row] = candidate;
    output.mortar_parent_faces[row] = mortar_face;
    output.nonmortar_parent_faces[row] = nonmortar_face;
    output.integration_weights[row] = point.weight;
    for ( int component = 0; component < 3; ++component ) {
      output.normals[3 * row + component] = point.normal[component];
    }
    for ( int component = 0; component < traces.mortar.geometry.dimension - 1; ++component ) {
      output.mortar_parent_reference_coordinates[( traces.mortar.geometry.dimension - 1 ) * row + component] =
          component == 0 ? point.mortar_reference.first : point.mortar_reference.second;
      output.nonmortar_parent_reference_coordinates[( traces.mortar.geometry.dimension - 1 ) * row + component] =
          component == 0 ? point.nonmortar_reference.first : point.nonmortar_reference.second;
    }
    const auto mortar_position =
        basis::interpolateParentField( traces.mortar, mortar_face, fields.mortar_coordinates, point.mortar_reference );
    const auto nonmortar_position = basis::interpolateParentField(
        traces.nonmortar, nonmortar_face, fields.nonmortar_coordinates, point.nonmortar_reference );
    Real gap{};
    for ( int component = 0; component < 3; ++component ) {
      output.mortar_positions[3 * row + component] = mortar_position[component];
      output.nonmortar_positions[3 * row + component] = nonmortar_position[component];
      gap += ( mortar_position[component] - nonmortar_position[component] ) * point.normal[component];
    }
    output.gaps[row] = gap;
    if ( include_normal_rates ) {
      const auto mortar_velocity =
          basis::interpolateParentField( traces.mortar, mortar_face, fields.mortar_velocity, point.mortar_reference );
      const auto nonmortar_velocity = basis::interpolateParentField(
          traces.nonmortar, nonmortar_face, fields.nonmortar_velocity, point.nonmortar_reference );
      Real rate{};
      for ( int component = 0; component < traces.mortar.geometry.dimension; ++component ) {
        rate += ( mortar_velocity[component] - nonmortar_velocity[component] ) * point.normal[component];
      }
      output.normal_rates[row] = rate;
    }
    const auto mortar_shape = basis::parentShape( traces.mortar, mortar_face, point.mortar_reference );
    const Index mortar_entry = mortar_entry_offsets[candidate] + local_row * mortar_dofs;
    output.mortar_dof_offsets[row] = mortar_entry;
    for ( Index local_dof = 0; local_dof < mortar_dofs; ++local_dof ) {
      output.mortar_dofs[mortar_entry + local_dof] =
          traces.mortar.parent_dofs[traces.mortar.parent_dof_offsets[mortar_face] + local_dof];
      output.mortar_basis_values[mortar_entry + local_dof] =
          traces.mortar.parent_dof_signs[traces.mortar.parent_dof_offsets[mortar_face] + local_dof] *
          mortar_shape[local_dof];
    }
    const auto nonmortar_shape = basis::parentShape( traces.nonmortar, nonmortar_face, point.nonmortar_reference );
    const Index nonmortar_entry = nonmortar_entry_offsets[candidate] + local_row * nonmortar_dofs;
    output.nonmortar_dof_offsets[row] = nonmortar_entry;
    for ( Index local_dof = 0; local_dof < nonmortar_dofs; ++local_dof ) {
      output.nonmortar_dofs[nonmortar_entry + local_dof] =
          traces.nonmortar.parent_dofs[traces.nonmortar.parent_dof_offsets[nonmortar_face] + local_dof];
      output.nonmortar_basis_values[nonmortar_entry + local_dof] =
          traces.nonmortar.parent_dof_signs[traces.nonmortar.parent_dof_offsets[nonmortar_face] + local_dof] *
          nonmortar_shape[local_dof];
    }
  }
}

TRIBOL_HOST_DEVICE void updateDeviceNativeRowFields( Index row, ConstraintRowBatchView rows,
                                                     ParentTraceFieldPairView fields, Real* mortar_positions,
                                                     Real* nonmortar_positions, Real* gaps, Real* normal_rates )
{
  if ( row >= rows.numberOfRows() ) {
    return;
  }
  const auto mortar_position =
      interpolateConstraintRowField( row, rows.mortar_dof_offsets, rows.mortar_dofs, rows.mortar_basis_values,
                                     fields.mortar_coordinates, rows.dimension );
  const auto nonmortar_position =
      interpolateConstraintRowField( row, rows.nonmortar_dof_offsets, rows.nonmortar_dofs, rows.nonmortar_basis_values,
                                     fields.nonmortar_coordinates, rows.dimension );
  Real gap{};
  for ( int component = 0; component < 3; ++component ) {
    mortar_positions[3 * row + component] = mortar_position[component];
    nonmortar_positions[3 * row + component] = nonmortar_position[component];
    gap += ( mortar_position[component] - nonmortar_position[component] ) * rows.normals( row, component );
  }
  gaps[row] = gap;
  if ( normal_rates != nullptr ) {
    const auto mortar_velocity =
        interpolateConstraintRowField( row, rows.mortar_dof_offsets, rows.mortar_dofs, rows.mortar_basis_values,
                                       fields.mortar_velocity, rows.dimension );
    const auto nonmortar_velocity =
        interpolateConstraintRowField( row, rows.nonmortar_dof_offsets, rows.nonmortar_dofs,
                                       rows.nonmortar_basis_values, fields.nonmortar_velocity, rows.dimension );
    Real rate{};
    for ( int component = 0; component < rows.dimension; ++component ) {
      rate += ( mortar_velocity[component] - nonmortar_velocity[component] ) * rows.normals( row, component );
    }
    normal_rates[row] = rate;
  }
}
