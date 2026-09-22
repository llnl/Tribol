void uploadSurfaces( SurfacePairView surfaces )
{
  host_surfaces_ = surfaces;
  mortar_.upload( surfaces.mortar );
  nonmortar_.upload( surfaces.nonmortar );
  device_surfaces_ = { mortar_.view(), nonmortar_.view() };
  state_.reserve( surfaces );
  mortar_direction_.resize( static_cast<std::size_t>( surfaces.mortar.coordinates.values.size() ) );
  nonmortar_direction_.resize( static_cast<std::size_t>( surfaces.nonmortar.coordinates.values.size() ) );
  exact_mortar_coordinates_.resize( mortar_direction_.size() );
  exact_nonmortar_coordinates_.resize( nonmortar_direction_.size() );
  host_derivative_.resize( std::max( mortar_direction_.size(), nonmortar_direction_.size() ) );
  result_layout_ = makeResultLayout( surfaces );
  results_.resize( static_cast<std::size_t>( result_layout_.total ) );
  gap_.resize( static_cast<std::size_t>( surfaces.mortar.numberOfNodes() ) );
  derivative_.resize( static_cast<std::size_t>( result_layout_.weighted_gap ) );
  surfaces_ready_ = true;
}

void updateGeometry( SurfacePairView surfaces )
{
  if ( !surfaces_ready_ ) {
    uploadSurfaces( surfaces );
    return;
  }
  host_surfaces_ = surfaces;
  mortar_.updateCoordinates( surfaces.mortar );
  nonmortar_.updateCoordinates( surfaces.nonmortar );
  device_surfaces_ = { mortar_.view(), nonmortar_.view() };
}

void invalidate()
{
  candidates_.clear();
  candidate_count_ = 0;
  surfaces_ready_ = false;
}

Index findCandidates( const search::Bvh::Parameters& parameters, bool self_contact, bool exclude_adjacent )
{
  requireSurfaces();
  const Index mortar_count = device_surfaces_.mortar.numberOfElements();
  const Index nonmortar_count = device_surfaces_.nonmortar.numberOfElements();
  if ( mortar_count == 0 || nonmortar_count == 0 ) {
    candidate_count_ = 0;
    candidates_.clear();
    prepareEvaluation();
    return 0;
  }
  mortar_bounds_.resize( static_cast<std::size_t>( mortar_count ) );
  global_bounds_.resize( 1 );
  morton_keys_.resize( static_cast<std::size_t>( mortar_count ) );
  morton_keys_alt_.resize( static_cast<std::size_t>( mortar_count ) );
  mortar_order_.resize( static_cast<std::size_t>( mortar_count ) );
  mortar_order_alt_.resize( static_cast<std::size_t>( mortar_count ) );
  const SurfaceMeshView mortar_surface = device_surfaces_.mortar;
  const Real expansion = parameters.expansion;
  const Real proximity_scale = parameters.proximity_scale;
  DeviceBounds* mortar_bounds = mortar_bounds_.data();
  launch1d( mortar_count, [=] RAJA_DEVICE( Index element ) {
    buildBounds( element, mortar_surface, expansion, proximity_scale, mortar_bounds );
  } );
  reduceBounds( mortar_count );
  const DeviceBounds* global_bounds = global_bounds_.data();
  std::uint32_t* morton_keys = morton_keys_.data();
  Index* mortar_order = mortar_order_.data();
  launch1d( mortar_count, [=] RAJA_DEVICE( Index element ) {
    assignMortonKey( element, mortar_bounds, mortar_count, global_bounds, morton_keys, mortar_order );
  } );
  sortMorton( mortar_count );
  buildBvh( mortar_count );

  candidate_counts_.resize( static_cast<std::size_t>( nonmortar_count + 1 ) );
  candidate_offsets_.resize( static_cast<std::size_t>( nonmortar_count + 1 ) );
  const SurfacePairView surfaces = device_surfaces_;
  const DeviceBvhNode* bvh_nodes = bvh_nodes_.data();
  const Index bvh_root = bvh_root_;
  Index* candidate_counts = candidate_counts_.data();
  launch1d( nonmortar_count, [=] RAJA_DEVICE( Index nonmortar ) {
    countCandidates( nonmortar, surfaces, bvh_nodes, bvh_root, expansion, proximity_scale, self_contact,
                     exclude_adjacent, candidate_counts );
  } );
  Index* candidate_scan_end = candidate_counts_.data() + nonmortar_count;
  launch1d( 1, [=] RAJA_DEVICE( Index ) { setZero( candidate_scan_end ); } );
  exclusiveScan( candidate_counts_.data(), candidate_offsets_.data(), nonmortar_count + 1 );
  ActiveDeviceBackend::copy( &candidate_count_, candidate_offsets_.data() + nonmortar_count, sizeof( Index ) );
  candidates_.resize( static_cast<std::size_t>( candidate_count_ ) );
  candidate_scratch_.resize( static_cast<std::size_t>( candidate_count_ ) );
  candidate_keys_.resize( static_cast<std::size_t>( candidate_count_ ) );
  candidate_keys_alt_.resize( static_cast<std::size_t>( candidate_count_ ) );
  if ( candidate_count_ > 0 ) {
    const Index* candidate_offsets = candidate_offsets_.data();
    ElementPair* candidates = candidates_.data();
    launch1d( nonmortar_count, [=] RAJA_DEVICE( Index nonmortar ) {
      fillCandidates( nonmortar, surfaces, bvh_nodes, bvh_root, expansion, proximity_scale, self_contact,
                      exclude_adjacent, candidate_offsets, candidates );
    } );
    canonicalizeCandidates();
  }
  prepareEvaluation();
  ActiveDeviceBackend::synchronize();
  return candidate_count_;
}

void setCandidates( ArrayView<const ElementPair> candidates )
{
  requireSurfaces();
  copyToDevice( candidates_, candidates, "copy supplied candidates to device" );
  candidate_count_ = candidates.size();
  prepareEvaluation();
}

void downloadCandidates( std::vector<ElementPair>& candidates ) const
{
  candidates.resize( static_cast<std::size_t>( candidate_count_ ) );
  if ( candidate_count_ > 0 ) {
    ActiveDeviceBackend::copy( candidates.data(), candidates_.data(), candidates.size() * sizeof( ElementPair ) );
  }
}

EvaluationSummary evaluate( const DefaultMethod::Parameters& parameters, const ContactStateView& state,
                            const timestep::Kinematic::Parameters& timestep_parameters )
{
  requireSurfaces();
  if ( parameters.enforcement.stiffness.value < 0.0 ) {
    throw std::invalid_argument( "Penalty stiffness cannot be negative." );
  }
  const bool evaluate_timestep =
      timestep_parameters.enabled && timestep_parameters.current_step >= timestep_parameters.minimum_current_step;
  validateThickness( state, parameters.constraint.activation.reject_excessive_penetration, evaluate_timestep );
  const bool need_thickness = parameters.constraint.activation.reject_excessive_penetration || evaluate_timestep;
  const ContactStateView device_state = state_.upload( state, host_surfaces_, evaluate_timestep, need_thickness );
  ActiveDeviceBackend::clear( results_.data(), results_.size() * sizeof( Real ) );
  if ( !gap_.size() ) {
    return {};
  }
  ActiveDeviceBackend::clear( gap_.data(), gap_.size() * sizeof( Real ) );
  if ( candidate_count_ > 0 ) {
    const SurfacePairView surfaces = device_surfaces_;
    const ElementPair* candidates = candidates_.data();
    const Index candidate_count = candidate_count_;
    const ResultLayout result_layout = result_layout_;
    InteractionPatch* patches = patches_.data();
    PenaltyContribution* contributions = contributions_.data();
    std::uint64_t* scatter_keys = scatter_keys_.data();
    Real* scatter_values = scatter_values_.data();
    Index* active_flags = active_flags_.data();
    launch1d( candidate_count, [=] RAJA_DEVICE( Index interaction ) {
      evaluateCommonPlane( interaction, surfaces, candidates, candidate_count, parameters, device_state, result_layout,
                           patches, contributions, scatter_keys, scatter_values, active_flags );
    } );
    Index* active_scan_end = active_flags_.data() + candidate_count_;
    launch1d( 1, [=] RAJA_DEVICE( Index ) { setZero( active_scan_end ); } );
    exclusiveScan( active_flags_.data(), active_offsets_.data(), candidate_count_ + 1 );
    const Index* active_offsets = active_offsets_.data();
    Real* quadrature_gap = quadrature_gap_.data();
    Real* quadrature_pressure = quadrature_pressure_.data();
    launch1d( candidate_count, [=] RAJA_DEVICE( Index interaction ) {
      compactQuadrature( interaction, contributions, active_offsets, candidate_count, quadrature_gap,
                         quadrature_pressure );
    } );
    deterministicScatter( static_cast<Index>( scatter_keys_.size() ), results_.data() );
    const Index mortar_nodes = host_surfaces_.mortar.numberOfNodes();
    Real* results = results_.data();
    Real* gap = gap_.data();
    launch1d( mortar_nodes,
              [=] RAJA_DEVICE( Index node ) { computeGap( node, result_layout, mortar_nodes, results, gap ); } );
  }

  const Real* votes = nullptr;
  Real initial_vote = std::numeric_limits<Real>::infinity();
  if ( timestep_parameters.enabled ) {
    initial_vote = timestep_parameters.current_step;
    if ( evaluate_timestep && candidate_count_ > 0 ) {
      const SurfacePairView surfaces = device_surfaces_;
      const ElementPair* candidates = candidates_.data();
      const InteractionPatch* patches = patches_.data();
      const Index candidate_count = candidate_count_;
      Real* timestep_votes = timestep_votes_.data();
      launch1d( candidate_count, [=] RAJA_DEVICE( Index interaction ) {
        timestepVote( interaction, surfaces, candidates, candidate_count, patches, parameters, device_state,
                      timestep_parameters, timestep_votes );
      } );
      votes = timestep_votes_.data();
    }
  }
  EvaluationSummary summary{ .timestep_vote = initial_vote };
  if ( candidate_count_ > 0 ) {
    const PenaltyContribution* contributions = contributions_.data();
    const Index candidate_count = candidate_count_;
    EvaluationSummary* summary_entries = summary_entries_.data();
    launch1d( candidate_count, [=] RAJA_DEVICE( Index interaction ) {
      makeSummaryEntry( interaction, contributions, votes, candidate_count, initial_vote, summary_entries );
    } );
    reduceSummary( candidate_count, initial_vote );
    ActiveDeviceBackend::copy( &summary, summary_.data(), sizeof( summary ) );
  }
  last_summary_ = summary;
  return summary;
}

void downloadResult( ContactOutputView output ) const
{
  requireInterleavedField( output.residual.mortar, host_surfaces_.mortar, "download mortar force" );
  requireInterleavedField( output.residual.nonmortar, host_surfaces_.nonmortar, "download nonmortar force" );
  copyResultSegment( output.residual.mortar.values, result_layout_.mortar_force,
                     result_layout_.nonmortar_force - result_layout_.mortar_force, "download mortar force" );
  copyResultSegment( output.residual.nonmortar.values, result_layout_.nonmortar_force,
                     result_layout_.weighted_gap - result_layout_.nonmortar_force, "download nonmortar force" );
  copyResultSegment( output.weighted_gap, result_layout_.weighted_gap, host_surfaces_.mortar.numberOfNodes(),
                     "download weighted gap" );
  copyResultSegment( output.tributary_area, result_layout_.tributary_area, host_surfaces_.mortar.numberOfNodes(),
                     "download tributary area" );
  copyResultSegment( output.pressure, result_layout_.pressure, host_surfaces_.mortar.numberOfNodes(),
                     "download pressure" );
  copyToHost( output.gap, gap_, "download gap" );
  copyPrefix( output.quadrature_gap, quadrature_gap_, last_summary_.quadrature_points, "download quadrature gap" );
  copyPrefix( output.quadrature_pressure, quadrature_pressure_, last_summary_.quadrature_points,
              "download quadrature pressure" );
}

ContactResultView deviceResult( EvaluationSummary summary, GeometryVersion geometry_version,
                                InteractionVersion interaction_version ) const
{
  return {
      .mortar_force = { { results_.data() + result_layout_.mortar_force,
                          result_layout_.nonmortar_force - result_layout_.mortar_force },
                        host_surfaces_.mortar.numberOfNodes(),
                        host_surfaces_.mortar.dimension,
                        FieldLayout::Interleaved },
      .nonmortar_force = { { results_.data() + result_layout_.nonmortar_force,
                             result_layout_.weighted_gap - result_layout_.nonmortar_force },
                           host_surfaces_.nonmortar.numberOfNodes(),
                           host_surfaces_.nonmortar.dimension,
                           FieldLayout::Interleaved },
      .gap = constDeviceView( gap_ ),
      .weighted_gap = { results_.data() + result_layout_.weighted_gap, host_surfaces_.mortar.numberOfNodes() },
      .tributary_area = { results_.data() + result_layout_.tributary_area, host_surfaces_.mortar.numberOfNodes() },
      .quadrature_gap = { quadrature_gap_.data(), summary.quadrature_points },
      .quadrature_pressure = { quadrature_pressure_.data(), summary.quadrature_points },
      .pressure = { results_.data() + result_layout_.pressure, host_surfaces_.mortar.numberOfNodes() },
      .summary = summary,
      .geometry_version = geometry_version,
      .interaction_version = interaction_version,
  };
}

void applyDerivative( const DefaultMethod::Parameters& parameters, const ContactStateView& state,
                      ContactDirectionView direction, ContactResidualView derivative )
{
  validateThickness( state, parameters.constraint.activation.reject_excessive_penetration, false );
  const ContactStateView device_state =
      state_.upload( state, host_surfaces_, false, parameters.constraint.activation.reject_excessive_penetration );
  copyDirection( direction.mortar, host_surfaces_.mortar, mortar_direction_, "copy mortar direction to device" );
  copyDirection( direction.nonmortar, host_surfaces_.nonmortar, nonmortar_direction_,
                 "copy nonmortar direction to device" );
  const Index mortar_values = host_surfaces_.mortar.coordinates.values.size();
  const Index nonmortar_values = host_surfaces_.nonmortar.coordinates.values.size();
  const FieldView<const Real> mortar_coordinates = device_surfaces_.mortar.coordinates;
  const FieldView<const Real> mortar_direction = { { mortar_direction_.data(), mortar_values },
                                                   host_surfaces_.mortar.numberOfNodes(),
                                                   host_surfaces_.mortar.dimension,
                                                   direction.mortar.layout };
  linearization_detail::ExactTangent* exact_mortar_coordinates = exact_mortar_coordinates_.data();
  launch1d( mortar_values, [=] RAJA_DEVICE( Index value ) {
    seedCoordinate( value, mortar_coordinates, mortar_direction, exact_mortar_coordinates );
  } );
  const FieldView<const Real> nonmortar_coordinates = device_surfaces_.nonmortar.coordinates;
  const FieldView<const Real> nonmortar_direction = { { nonmortar_direction_.data(), nonmortar_values },
                                                      host_surfaces_.nonmortar.numberOfNodes(),
                                                      host_surfaces_.nonmortar.dimension,
                                                      direction.nonmortar.layout };
  linearization_detail::ExactTangent* exact_nonmortar_coordinates = exact_nonmortar_coordinates_.data();
  launch1d( nonmortar_values, [=] RAJA_DEVICE( Index value ) {
    seedCoordinate( value, nonmortar_coordinates, nonmortar_direction, exact_nonmortar_coordinates );
  } );
  const SurfacePairViewT<linearization_detail::ExactTangent> exact_surfaces{
      exactSurface( device_surfaces_.mortar, exact_mortar_coordinates_ ),
      exactSurface( device_surfaces_.nonmortar, exact_nonmortar_coordinates_ ),
  };
  ActiveDeviceBackend::clear( derivative_.data(), derivative_.size() * sizeof( Real ) );
  if ( candidate_count_ > 0 ) {
    const ElementPair* candidates = candidates_.data();
    const Index candidate_count = candidate_count_;
    const ResultLayout result_layout = result_layout_;
    std::uint64_t* scatter_keys = scatter_keys_.data();
    Real* scatter_values = scatter_values_.data();
    launch1d( candidate_count, [=] RAJA_DEVICE( Index interaction ) {
      evaluateDerivative( interaction, exact_surfaces, candidates, candidate_count, parameters, device_state,
                          result_layout, scatter_keys, scatter_values );
    } );
    deterministicScatter( static_cast<Index>( scatter_keys_.size() ), derivative_.data() );
  }
  addDerivative( derivative.mortar, result_layout_.mortar_force,
                 result_layout_.nonmortar_force - result_layout_.mortar_force, "download mortar derivative" );
  addDerivative( derivative.nonmortar, result_layout_.nonmortar_force,
                 result_layout_.weighted_gap - result_layout_.nonmortar_force, "download nonmortar derivative" );
}

[[nodiscard]] SurfacePairView deviceSurfaces() const { return device_surfaces_; }
[[nodiscard]] ArrayView<const ElementPair> deviceCandidates() const { return constDeviceView( candidates_ ); }
[[nodiscard]] ArrayView<const InteractionPatch> devicePatches() const { return constDeviceView( patches_ ); }
[[nodiscard]] Index interactionCount() const { return candidate_count_; }
