  void prepareNative( ParentTracePairView traces, ParentTraceFieldPairView fields,
                      ArrayView<const ElementPair> device_candidates, DeviceRowDofLayout layout,
                      DeviceNativeRowOptions options )
  {
    validateNativeInput( traces, fields, device_candidates, layout, options );
    layout_ = layout;
    native_options_ = options;
    native_traces_.upload( traces );
    native_fields_.uploadCoordinates( fields );
    const bool include_kinematic_vote =
        options.kinematic.enabled && options.kinematic.current_step >= options.kinematic.minimum_current_step;
    native_fields_.uploadVelocity( fields, options.include_normal_rates || include_kinematic_vote );
    copyToDevice( native_candidates_, device_candidates, "copy native candidates to device" );
    const Index candidates = device_candidates.size();
    native_patches_.resize( static_cast<std::size_t>( candidates ) );
    native_mortar_parent_faces_.resize( static_cast<std::size_t>( candidates ) );
    native_nonmortar_parent_faces_.resize( static_cast<std::size_t>( candidates ) );
    native_owner_ranks_.resize( static_cast<std::size_t>( candidates ) );
    const std::size_t references = static_cast<std::size_t>( nativeReferenceComponents * maximumOverlapVertices ) *
                                   static_cast<std::size_t>( candidates );
    native_mortar_patch_reference_.resize( references );
    native_nonmortar_patch_reference_.resize( references );
    native_row_counts_.resize( static_cast<std::size_t>( candidates + 1 ) );
    native_row_offsets_.resize( static_cast<std::size_t>( candidates + 1 ) );
    native_mortar_entry_counts_.resize( static_cast<std::size_t>( candidates + 1 ) );
    native_mortar_entry_offsets_.resize( static_cast<std::size_t>( candidates + 1 ) );
    native_nonmortar_entry_counts_.resize( static_cast<std::size_t>( candidates + 1 ) );
    native_nonmortar_entry_offsets_.resize( static_cast<std::size_t>( candidates + 1 ) );
    native_mortar_element_thickness_.resize(
        include_kinematic_vote ? static_cast<std::size_t>( traces.mortar.numberOfSubfacets() ) : 0 );
    native_nonmortar_element_thickness_.resize(
        include_kinematic_vote ? static_cast<std::size_t>( traces.nonmortar.numberOfSubfacets() ) : 0 );
    native_kinematic_votes_.resize( include_kinematic_vote ? static_cast<std::size_t>( candidates ) : 0 );
    native_kinematic_vote_.resize( include_kinematic_vote ? 1 : 0 );

    const ParentTracePairView device_traces = native_traces_.view();
    InteractionPatch* patches = native_patches_.data();
    Index* mortar_faces = native_mortar_parent_faces_.data();
    Index* nonmortar_faces = native_nonmortar_parent_faces_.data();
    int* owners = native_owner_ranks_.data();
    Real* mortar_reference = native_mortar_patch_reference_.data();
    Real* nonmortar_reference = native_nonmortar_patch_reference_.data();
    const ElementPair* candidate_data = native_candidates_.data();
    const auto geometry = options.geometry;
    const auto overlap = options.overlap;
    launchRowKernel( candidates, [=] RAJA_DEVICE( Index candidate ) {
      buildDeviceMappedPatch( candidate, device_traces, candidate_data, candidates, geometry, overlap, patches,
                              mortar_faces, nonmortar_faces, owners, mortar_reference, nonmortar_reference );
    } );

    Index* row_counts = native_row_counts_.data();
    Index* mortar_counts = native_mortar_entry_counts_.data();
    Index* nonmortar_counts = native_nonmortar_entry_counts_.data();
    const auto quadrature = options.quadrature;
    launchRowKernel( candidates + 1, [=] RAJA_DEVICE( Index candidate ) {
      countDeviceNativeRows( candidate, device_traces, candidates, quadrature, overlap, patches, mortar_faces,
                             nonmortar_faces, row_counts, mortar_counts, nonmortar_counts );
    } );
    exclusiveScan( native_row_counts_, native_row_offsets_ );
    exclusiveScan( native_mortar_entry_counts_, native_mortar_entry_offsets_ );
    exclusiveScan( native_nonmortar_entry_counts_, native_nonmortar_entry_offsets_ );
    const Index rows = readLast( native_row_offsets_ );
    const Index mortar_entries = readLast( native_mortar_entry_offsets_ );
    const Index nonmortar_entries = readLast( native_nonmortar_entry_offsets_ );
    rows_.resize( traces.mortar.geometry.dimension, candidates, rows, mortar_entries, nonmortar_entries,
                  options.include_normal_rates );
    ActiveDeviceBackend::copy( rows_.cell_row_offsets.data(), native_row_offsets_.data(),
                               native_row_offsets_.size() * sizeof( Index ) );
    fillNativeRows( device_traces, native_fields_.view(), candidates );
    ActiveDeviceBackend::copy( rows_.mortar_dof_offsets.data() + rows, &mortar_entries, sizeof( Index ) );
    ActiveDeviceBackend::copy( rows_.nonmortar_dof_offsets.data() + rows, &nonmortar_entries, sizeof( Index ) );
    configureEvaluation();
    native_prepared_ = true;
    prepared_ = true;
  }
  void updateNativeFields( ParentTraceFieldPairView fields )
  {
    requirePrepared();
    if ( !native_prepared_ ) {
      throw std::logic_error( "Device native fields require prepareNative() first." );
    }
    const auto& kinematic = native_options_.kinematic;
    const bool require_velocity =
        native_options_.include_normal_rates || ( kinematic.enabled && kinematic.current_step >= kinematic.minimum_current_step );
    validateNativeFields( fields, layout_, require_velocity );
    native_fields_.uploadCoordinates( fields );
    native_fields_.uploadVelocity( fields, require_velocity );
    const ConstraintRowBatchView rows = rows_.view();
    const ParentTraceFieldPairView device_fields = native_fields_.view();
    Real* mortar_positions = rows_.mortar_positions.data();
    Real* nonmortar_positions = rows_.nonmortar_positions.data();
    Real* gaps = rows_.gaps.data();
    Real* normal_rates = native_options_.include_normal_rates ? rows_.normal_rates.data() : nullptr;
    launchRowKernel( rows.numberOfRows(), [=] RAJA_DEVICE( Index row ) {
      updateDeviceNativeRowFields( row, rows, device_fields, mortar_positions, nonmortar_positions, gaps,
                                   normal_rates );
    } );
  }

  [[nodiscard]] ArrayView<const InteractionPatch> devicePatches() const { return constDeviceView( native_patches_ ); }

  void configureEvaluation()
  {
    const std::size_t row_count = static_cast<std::size_t>( rows_.rows );
    const std::size_t force_values = static_cast<std::size_t>( layout_.forceValues() );
    const std::size_t force_entries =
        static_cast<std::size_t>( rows_.dimension ) * ( rows_.mortar_dofs.size() + rows_.nonmortar_dofs.size() );
    pressure_stiffness_.resize( row_count );
    rate_coefficients_.resize( row_count );
    effective_gap_.resize( row_count );
    pressure_.resize( row_count );
    row_force_.resize( row_count );
    row_stiffness_.resize( row_count );
    row_rate_.resize( row_count );
    summary_entries_.resize( row_count );
    summary_.resize( 1 );
    inverse_mass_.resize( force_values );
    force_.resize( force_values );
    stability_row_sums_.resize( 2 * force_values );
    stability_bounds_.resize( 2 );
    const std::size_t scatter_capacity = 2 * force_entries;
    scatter_keys_.resize( scatter_capacity );
    scatter_keys_alt_.resize( scatter_capacity );
    scatter_values_.resize( scatter_capacity );
    scatter_values_alt_.resize( scatter_capacity );
    unique_keys_.resize( scatter_capacity );
    run_counts_.resize( scatter_capacity );
    run_offsets_.resize( scatter_capacity );
    run_count_.resize( 1 );
    sizeTemporaryStorage( static_cast<Index>( scatter_capacity ), static_cast<Index>( row_count ),
                          static_cast<Index>( force_values ) );
  }

  static void validateNativeFields( ParentTraceFieldPairView fields, DeviceRowDofLayout layout, bool require_velocity )
  {
    const auto valid = [layout]( FieldView<const Real> field, Index entities ) {
      return field.isStructurallyValid() && field.entities == entities && field.components == layout.dimension;
    };
    if ( !valid( fields.mortar_coordinates, layout.mortar_dofs ) ||
         !valid( fields.nonmortar_coordinates, layout.nonmortar_dofs ) ||
         ( require_velocity && ( !valid( fields.mortar_velocity, layout.mortar_dofs ) ||
                                 !valid( fields.nonmortar_velocity, layout.nonmortar_dofs ) ) ) ) {
      throw std::invalid_argument( "Device native parent fields do not match the DOF layout." );
    }
  }

  static void validateNativeInput( ParentTracePairView traces, ParentTraceFieldPairView fields,
                                   ArrayView<const ElementPair> device_candidates, DeviceRowDofLayout layout,
                                   const DeviceNativeRowOptions& options )
  {
    timestep::validate( options.kinematic );
    if ( !traces.isStructurallyValid() || !device_candidates.isStructurallyValid() ||
         traces.mortar.geometry.dimension != layout.dimension || options.overlap.mapping_tolerance <= 0.0 ||
         options.overlap.orientation_tolerance < 0.0 || options.quadrature.minimum_exactness < 0 ||
         options.quadrature.traction_order < -1 ) {
      throw std::invalid_argument( "Invalid device native row input." );
    }
    validateNativeFields( fields, layout, fields.hasVelocity() );
    const auto validate_dofs = []( ParentTraceView trace, Index entities ) {
      for ( Index dof : trace.parent_dofs ) {
        if ( dof < 0 || dof >= entities ) {
          throw std::out_of_range( "A native parent trace references a DOF outside the device layout." );
        }
      }
    };
    validate_dofs( traces.mortar, layout.mortar_dofs );
    validate_dofs( traces.nonmortar, layout.nonmortar_dofs );
    int maximum_order{};
    for ( int order : traces.mortar.parent_face_orders ) {
      maximum_order = order > maximum_order ? order : maximum_order;
    }
    for ( int order : traces.nonmortar.parent_face_orders ) {
      maximum_order = order > maximum_order ? order : maximum_order;
    }
    if ( integration::automaticQuadratureExactness( maximum_order, maximum_order, options.quadrature ) >
         integration::maximumAutomaticExactness ) {
      throw std::invalid_argument( "Device native quadrature exceeds the built-in exactness." );
    }
  }

  void exclusiveScan( DeviceBuffer<Index>& input, DeviceBuffer<Index>& output )
  {
    const Index count = static_cast<Index>( input.size() );
    const std::size_t bytes = ActiveDeviceBackend::exclusiveSumTemporaryBytes( input.data(), output.data(), count );
    temporary_storage_.resize( bytes );
    ActiveDeviceBackend::exclusiveSum( temporary_storage_.data(), bytes, input.data(), output.data(), count );
  }

  static Index readLast( const DeviceBuffer<Index>& values )
  {
    Index result{};
    if ( values.size() > 0 ) {
      ActiveDeviceBackend::copy( &result, values.data() + values.size() - 1, sizeof( Index ) );
    }
    return result;
  }

  void fillNativeRows( ParentTracePairView traces, ParentTraceFieldPairView fields, Index candidates )
  {
    DeviceNativeRowArrays output{
        .contact_pair_ids = rows_.contact_pair_ids.data(),
        .cell_ids = rows_.cell_ids.data(),
        .row_ids = rows_.row_ids.data(),
        .owner_ranks = rows_.owner_ranks.data(),
        .overlap_cells = rows_.overlap_cells.data(),
        .mortar_parent_faces = rows_.mortar_parent_faces.data(),
        .nonmortar_parent_faces = rows_.nonmortar_parent_faces.data(),
        .normals = rows_.normals.data(),
        .integration_weights = rows_.integration_weights.data(),
        .mortar_parent_reference_coordinates = rows_.mortar_parent_reference_coordinates.data(),
        .nonmortar_parent_reference_coordinates = rows_.nonmortar_parent_reference_coordinates.data(),
        .mortar_positions = rows_.mortar_positions.data(),
        .nonmortar_positions = rows_.nonmortar_positions.data(),
        .gaps = rows_.gaps.data(),
        .normal_rates = native_options_.include_normal_rates ? rows_.normal_rates.data() : nullptr,
        .mortar_dof_offsets = rows_.mortar_dof_offsets.data(),
        .mortar_dofs = rows_.mortar_dofs.data(),
        .mortar_basis_values = rows_.mortar_basis_values.data(),
        .nonmortar_dof_offsets = rows_.nonmortar_dof_offsets.data(),
        .nonmortar_dofs = rows_.nonmortar_dofs.data(),
        .nonmortar_basis_values = rows_.nonmortar_basis_values.data(),
    };
    const auto quadrature = native_options_.quadrature;
    const auto overlap = native_options_.overlap;
    const bool include_normal_rates = native_options_.include_normal_rates;
    const std::uint64_t first_row_id = native_options_.first_row_id;
    const InteractionPatch* patches = native_patches_.data();
    const Index* mortar_faces = native_mortar_parent_faces_.data();
    const Index* nonmortar_faces = native_nonmortar_parent_faces_.data();
    const int* owners = native_owner_ranks_.data();
    const Real* mortar_reference = native_mortar_patch_reference_.data();
    const Real* nonmortar_reference = native_nonmortar_patch_reference_.data();
    const Index* row_offsets = native_row_offsets_.data();
    const Index* mortar_entry_offsets = native_mortar_entry_offsets_.data();
    const Index* nonmortar_entry_offsets = native_nonmortar_entry_offsets_.data();
    launchRowKernel( candidates, [=] RAJA_DEVICE( Index candidate ) {
      fillDeviceNativeRows( candidate, traces, fields, candidates, quadrature, overlap, patches, mortar_faces,
                            nonmortar_faces, owners, mortar_reference, nonmortar_reference, row_offsets,
                            mortar_entry_offsets, nonmortar_entry_offsets, first_row_id, include_normal_rates, output );
    } );
  }
