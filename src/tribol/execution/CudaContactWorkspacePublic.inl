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
  launch1d( mortar_count, buildBoundsKernel, device_surfaces_.mortar, parameters.expansion, parameters.proximity_scale,
            mortar_bounds_.data() );
  reduceBounds( mortar_count );
  launch1d( mortar_count, mortonKernel, mortar_bounds_.data(), mortar_count, global_bounds_.data(), morton_keys_.data(),
            mortar_order_.data() );
  sortMorton( mortar_count );
  buildBvh( mortar_count );

  candidate_counts_.resize( static_cast<std::size_t>( nonmortar_count + 1 ) );
  candidate_offsets_.resize( static_cast<std::size_t>( nonmortar_count + 1 ) );
  launch1d( nonmortar_count, countCandidatesKernel, device_surfaces_, bvh_nodes_.data(), bvh_root_,
            parameters.expansion, parameters.proximity_scale, self_contact, exclude_adjacent,
            candidate_counts_.data() );
  setZeroKernel<<<1, 1>>>( candidate_counts_.data() + nonmortar_count );
  requireKernel( "initialize CUDA candidate scan" );
  exclusiveScan( candidate_counts_.data(), candidate_offsets_.data(), nonmortar_count + 1 );
  requireCuda( cudaMemcpy( &candidate_count_, candidate_offsets_.data() + nonmortar_count, sizeof( Index ),
                           cudaMemcpyDeviceToHost ),
               "copy CUDA candidate count" );
  candidates_.resize( static_cast<std::size_t>( candidate_count_ ) );
  candidate_scratch_.resize( static_cast<std::size_t>( candidate_count_ ) );
  candidate_keys_.resize( static_cast<std::size_t>( candidate_count_ ) );
  candidate_keys_alt_.resize( static_cast<std::size_t>( candidate_count_ ) );
  if ( candidate_count_ > 0 ) {
    launch1d( nonmortar_count, fillCandidatesKernel, device_surfaces_, bvh_nodes_.data(), bvh_root_,
              parameters.expansion, parameters.proximity_scale, self_contact, exclude_adjacent,
              candidate_offsets_.data(), candidates_.data() );
    canonicalizeCandidates();
  }
  prepareEvaluation();
  requireCuda( cudaDeviceSynchronize(), "complete CUDA BVH search" );
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
    requireCuda( cudaMemcpy( candidates.data(), candidates_.data(), candidates.size() * sizeof( ElementPair ),
                             cudaMemcpyDeviceToHost ),
                 "copy CUDA candidates to host" );
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
  requireCuda( cudaMemset( results_.data(), 0, results_.size() * sizeof( Real ) ), "clear CUDA contact results" );
  if ( !gap_.size() ) {
    return {};
  }
  requireCuda( cudaMemset( gap_.data(), 0, gap_.size() * sizeof( Real ) ), "clear CUDA gap results" );
  if ( candidate_count_ > 0 ) {
    launch1d( candidate_count_, evaluateCommonPlaneKernel, device_surfaces_, candidates_.data(), candidate_count_,
              parameters, device_state, result_layout_, patches_.data(), contributions_.data(), scatter_keys_.data(),
              scatter_values_.data(), active_flags_.data() );
    setZeroKernel<<<1, 1>>>( active_flags_.data() + candidate_count_ );
    requireKernel( "initialize CUDA active scan" );
    exclusiveScan( active_flags_.data(), active_offsets_.data(), candidate_count_ + 1 );
    launch1d( candidate_count_, compactQuadratureKernel, contributions_.data(), active_offsets_.data(),
              candidate_count_, quadrature_gap_.data(), quadrature_pressure_.data() );
    deterministicScatter( static_cast<Index>( scatter_keys_.size() ), results_.data() );
    launch1d( host_surfaces_.mortar.numberOfNodes(), computeGapKernel, result_layout_,
              host_surfaces_.mortar.numberOfNodes(), results_.data(), gap_.data() );
  }

  const Real* votes = nullptr;
  Real initial_vote = std::numeric_limits<Real>::infinity();
  if ( timestep_parameters.enabled ) {
    initial_vote = timestep_parameters.current_step;
    if ( evaluate_timestep && candidate_count_ > 0 ) {
      launch1d( candidate_count_, timestepVoteKernel, device_surfaces_, candidates_.data(), candidate_count_,
                patches_.data(), parameters, device_state, timestep_parameters, timestep_votes_.data() );
      votes = timestep_votes_.data();
    }
  }
  summarizeKernel<<<1, 256>>>( contributions_.data(), votes, candidate_count_, initial_vote, summary_.data() );
  requireKernel( "summarize CUDA contact" );
  EvaluationSummary summary;
  requireCuda( cudaMemcpy( &summary, summary_.data(), sizeof( summary ), cudaMemcpyDeviceToHost ),
               "copy CUDA contact summary" );
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
  constexpr int block_size = 128;
  const Index mortar_values = host_surfaces_.mortar.coordinates.values.size();
  const Index nonmortar_values = host_surfaces_.nonmortar.coordinates.values.size();
  seedCoordinatesKernel<<<blocks( mortar_values, block_size ), block_size>>>(
      device_surfaces_.mortar.coordinates,
      { { mortar_direction_.data(), mortar_values },
        host_surfaces_.mortar.numberOfNodes(),
        host_surfaces_.mortar.dimension,
        direction.mortar.layout },
      exact_mortar_coordinates_.data() );
  seedCoordinatesKernel<<<blocks( nonmortar_values, block_size ), block_size>>>(
      device_surfaces_.nonmortar.coordinates,
      { { nonmortar_direction_.data(), nonmortar_values },
        host_surfaces_.nonmortar.numberOfNodes(),
        host_surfaces_.nonmortar.dimension,
        direction.nonmortar.layout },
      exact_nonmortar_coordinates_.data() );
  requireKernel( "seed CUDA coordinate derivatives" );
  const SurfacePairViewT<linearization_detail::ExactTangent> exact_surfaces{
      exactSurface( device_surfaces_.mortar, exact_mortar_coordinates_ ),
      exactSurface( device_surfaces_.nonmortar, exact_nonmortar_coordinates_ ),
  };
  requireCuda( cudaMemset( derivative_.data(), 0, derivative_.size() * sizeof( Real ) ),
               "clear CUDA derivative results" );
  if ( candidate_count_ > 0 ) {
    launch1d( candidate_count_, evaluateDerivativeKernel, exact_surfaces, candidates_.data(), candidate_count_,
              parameters, device_state, result_layout_, scatter_keys_.data(), scatter_values_.data() );
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
