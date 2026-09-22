template <typename Body>
static void launch1d( Index count, Body&& body )
{
  if ( count <= 0 ) {
    return;
  }
  RAJA::forall<typename ActiveDeviceBackend::ForallPolicy>(
      ActiveDeviceBackend::resource(), RAJA::TypedRangeSegment<Index>( 0, count ), std::forward<Body>( body ) );
  ActiveDeviceBackend::checkLaunch( "launch device contact kernel" );
}

void requireSurfaces() const
{
  if ( !surfaces_ready_ ) {
    throw std::logic_error( "Surfaces must be uploaded before device contact operations." );
  }
}

void ensureTemporaryStorage( std::size_t bytes )
{
  temporary_storage_.resize( std::max( temporary_storage_.size(), bytes ) );
}

void reduceBounds( Index count )
{
  std::size_t bytes{};
  const DeviceBounds initial = emptyBounds();
  bytes = ActiveDeviceBackend::reduceTemporaryBytes( mortar_bounds_.data(), global_bounds_.data(), count, BoundsUnion{},
                                                     initial );
  ensureTemporaryStorage( bytes );
  ActiveDeviceBackend::reduce( temporary_storage_.data(), bytes, mortar_bounds_.data(), global_bounds_.data(), count,
                               BoundsUnion{}, initial );
}

void sortMorton( Index count )
{
  std::size_t bytes = ActiveDeviceBackend::sortPairsTemporaryBytes(
      morton_keys_.data(), morton_keys_alt_.data(), mortar_order_.data(), mortar_order_alt_.data(), count );
  ensureTemporaryStorage( bytes );
  ActiveDeviceBackend::sortPairs( temporary_storage_.data(), bytes, morton_keys_.data(), morton_keys_alt_.data(),
                                  mortar_order_.data(), mortar_order_alt_.data(), count );
  morton_keys_.swap( morton_keys_alt_ );
  mortar_order_.swap( mortar_order_alt_ );
}

void buildBvh( Index leaf_count )
{
  std::vector<Index> offsets;
  std::vector<Index> counts;
  offsets.push_back( 0 );
  counts.push_back( leaf_count );
  Index total = leaf_count;
  while ( counts.back() > 1 ) {
    offsets.push_back( total );
    counts.push_back( ( counts.back() + 1 ) / 2 );
    total += counts.back();
  }
  bvh_nodes_.resize( static_cast<std::size_t>( total ) );
  const DeviceBounds* element_bounds = mortar_bounds_.data();
  const Index* order = mortar_order_.data();
  DeviceBvhNode* nodes = bvh_nodes_.data();
  launch1d( leaf_count,
            [=] RAJA_DEVICE( Index leaf ) { initializeLeaf( leaf, element_bounds, order, leaf_count, nodes ); } );
  for ( std::size_t level = 1; level < counts.size(); ++level ) {
    const Index child_offset = offsets[level - 1];
    const Index child_count = counts[level - 1];
    const Index parent_offset = offsets[level];
    launch1d( counts[level], [=] RAJA_DEVICE( Index parent ) {
      buildBvhNode( parent, nodes, child_offset, child_count, parent_offset );
    } );
  }
  bvh_root_ = offsets.back();
}

void exclusiveScan( const Index* input, Index* output, Index count )
{
  std::size_t bytes = ActiveDeviceBackend::exclusiveSumTemporaryBytes( input, output, count );
  ensureTemporaryStorage( bytes );
  ActiveDeviceBackend::exclusiveSum( temporary_storage_.data(), bytes, input, output, count );
}

void canonicalizeCandidates()
{
  stableSortCandidates<false>( candidates_, candidate_scratch_ );
  stableSortCandidates<true>( candidate_scratch_, candidates_ );
}

template <bool MortarKey>
void stableSortCandidates( DeviceBuffer<ElementPair>& input, DeviceBuffer<ElementPair>& output )
{
  const ElementPair* pairs = input.data();
  Index* keys = candidate_keys_.data();
  const Index candidate_count = candidate_count_;
  launch1d( candidate_count_,
            [=] RAJA_DEVICE( Index index ) { assignPairKey<MortarKey>( index, pairs, candidate_count, keys ); } );
  std::size_t bytes = ActiveDeviceBackend::sortPairsTemporaryBytes( candidate_keys_.data(), candidate_keys_alt_.data(),
                                                                    input.data(), output.data(), candidate_count_ );
  ensureTemporaryStorage( bytes );
  ActiveDeviceBackend::sortPairs( temporary_storage_.data(), bytes, candidate_keys_.data(), candidate_keys_alt_.data(),
                                  input.data(), output.data(), candidate_count_ );
}

void prepareEvaluation()
{
  const std::size_t candidates = static_cast<std::size_t>( candidate_count_ );
  patches_.resize( candidates );
  contributions_.resize( candidates );
  active_flags_.resize( candidates + 1 );
  active_offsets_.resize( candidates + 1 );
  quadrature_gap_.resize( candidates );
  quadrature_pressure_.resize( candidates );
  timestep_votes_.resize( candidates );
  summary_entries_.resize( candidates );
  summary_.resize( 1 );
  const std::size_t entries = candidates * maximumScatterEntries;
  scatter_keys_.resize( entries );
  scatter_keys_alt_.resize( entries );
  scatter_values_.resize( entries );
  scatter_values_alt_.resize( entries );
  unique_keys_.resize( entries );
  run_counts_.resize( entries );
  run_offsets_.resize( entries );
  run_count_.resize( 1 );
  if ( candidates > static_cast<std::size_t>( std::numeric_limits<int>::max() ) / maximumScatterEntries ) {
    throw std::overflow_error( "Device contact scatter exceeds the supported backend item count." );
  }
  if ( entries > 0 ) {
    sizeScatterTemporary( static_cast<Index>( entries ) );
  }
  if ( candidates > 0 ) {
    sizeSummaryTemporary( candidate_count_ );
  }
}

void sizeSummaryTemporary( Index count )
{
  const EvaluationSummary initial{};
  const std::size_t bytes = ActiveDeviceBackend::reduceTemporaryBytes( summary_entries_.data(), summary_.data(), count,
                                                                       SummaryReduction{}, initial );
  ensureTemporaryStorage( bytes );
}

void reduceSummary( Index count, Real initial_vote )
{
  const EvaluationSummary initial{ .timestep_vote = initial_vote };
  const std::size_t bytes = ActiveDeviceBackend::reduceTemporaryBytes( summary_entries_.data(), summary_.data(), count,
                                                                       SummaryReduction{}, initial );
  ActiveDeviceBackend::reduce( temporary_storage_.data(), bytes, summary_entries_.data(), summary_.data(), count,
                               SummaryReduction{}, initial );
}

void sizeScatterTemporary( Index entries )
{
  std::size_t bytes = ActiveDeviceBackend::sortPairsTemporaryBytes(
      scatter_keys_.data(), scatter_keys_alt_.data(), scatter_values_.data(), scatter_values_alt_.data(), entries );
  ensureTemporaryStorage( bytes );
  bytes = ActiveDeviceBackend::runLengthEncodeTemporaryBytes( scatter_keys_alt_.data(), unique_keys_.data(),
                                                              run_counts_.data(), run_count_.data(), entries );
  ensureTemporaryStorage( bytes );
  bytes = ActiveDeviceBackend::exclusiveSumTemporaryBytes( run_counts_.data(), run_offsets_.data(), entries );
  ensureTemporaryStorage( bytes );
}

void deterministicScatter( Index entries, Real* output )
{
  if ( entries <= 0 ) {
    return;
  }
  std::size_t bytes = ActiveDeviceBackend::sortPairsTemporaryBytes(
      scatter_keys_.data(), scatter_keys_alt_.data(), scatter_values_.data(), scatter_values_alt_.data(), entries );
  ActiveDeviceBackend::sortPairs( temporary_storage_.data(), bytes, scatter_keys_.data(), scatter_keys_alt_.data(),
                                  scatter_values_.data(), scatter_values_alt_.data(), entries );
  bytes = ActiveDeviceBackend::runLengthEncodeTemporaryBytes( scatter_keys_alt_.data(), unique_keys_.data(),
                                                              run_counts_.data(), run_count_.data(), entries );
  ActiveDeviceBackend::clear( run_counts_.data(), run_counts_.size() * sizeof( Index ) );
  ActiveDeviceBackend::runLengthEncode( temporary_storage_.data(), bytes, scatter_keys_alt_.data(), unique_keys_.data(),
                                        run_counts_.data(), run_count_.data(), entries );
  bytes = ActiveDeviceBackend::exclusiveSumTemporaryBytes( run_counts_.data(), run_offsets_.data(), entries );
  ActiveDeviceBackend::exclusiveSum( temporary_storage_.data(), bytes, run_counts_.data(), run_offsets_.data(),
                                     entries );
  const std::uint64_t* unique_keys = unique_keys_.data();
  const Index* run_counts = run_counts_.data();
  const Index* run_offsets = run_offsets_.data();
  const Index* run_count = run_count_.data();
  const Real* sorted_values = scatter_values_alt_.data();
  launch1d( entries, [=] RAJA_DEVICE( Index run ) {
    reduceScatterRun( run, unique_keys, run_counts, run_offsets, run_count, sorted_values, output );
  } );
}

void validateThickness( const ContactStateView& state, bool penetration, bool timestep_enabled ) const
{
  if ( !penetration && !timestep_enabled ) {
    return;
  }
  if ( state.mortar_element_thickness.size() != host_surfaces_.mortar.numberOfElements() ||
       state.nonmortar_element_thickness.size() != host_surfaces_.nonmortar.numberOfElements() ) {
    throw std::invalid_argument( "Device contact thickness fields must contain one value per surface element." );
  }
  for ( Index element = 0; element < state.mortar_element_thickness.size(); ++element ) {
    if ( state.mortar_element_thickness[element] < 0.0 ||
         ( penetration && state.mortar_element_thickness[element] == 0.0 ) ) {
      throw std::invalid_argument( "Device contact requires valid positive element thickness." );
    }
  }
  for ( Index element = 0; element < state.nonmortar_element_thickness.size(); ++element ) {
    if ( state.nonmortar_element_thickness[element] < 0.0 ||
         ( penetration && state.nonmortar_element_thickness[element] == 0.0 ) ) {
      throw std::invalid_argument( "Device contact requires valid positive element thickness." );
    }
  }
}

template <typename Scalar>
static SurfaceMeshViewT<Scalar> exactSurface( const SurfaceMeshView& source, const DeviceBuffer<Scalar>& coordinates )
{
  return {
      .dimension = source.dimension,
      .coordinates = { { coordinates.data(), static_cast<Index>( coordinates.size() ) },
                       source.numberOfNodes(),
                       source.dimension,
                       FieldLayout::Interleaved },
      .element_offsets = source.element_offsets,
      .connectivity = source.connectivity,
      .topologies = source.topologies,
      .attributes = source.attributes,
  };
}

static void copyDirection( const FieldView<const Real>& direction, const SurfaceMeshView& surface,
                           DeviceBuffer<Real>& destination, const char* operation )
{
  if ( !direction.isStructurallyValid() || direction.entities != surface.numberOfNodes() ||
       direction.components != surface.dimension ) {
    throw std::invalid_argument( "Device coordinate direction must match its surface." );
  }
  static_cast<void>( operation );
  ActiveDeviceBackend::copy( destination.data(), direction.values.data(), destination.size() * sizeof( Real ) );
}

static void requireInterleavedField( const FieldView<Real>& field, const SurfaceMeshView& surface,
                                     const char* operation )
{
  if ( !field.isStructurallyValid() || field.entities != surface.numberOfNodes() ||
       field.components != surface.dimension || field.layout != FieldLayout::Interleaved ) {
    throw std::invalid_argument( std::string( operation ) + " requires an interleaved host field." );
  }
}

void copyResultSegment( ArrayView<Real> destination, Index offset, Index count, const char* operation ) const
{
  if ( destination.size() != count ) {
    throw std::invalid_argument( std::string( operation ) + " has a mismatched destination size." );
  }
  if ( count > 0 ) {
    ActiveDeviceBackend::copy( destination.data(), results_.data() + offset,
                               static_cast<std::size_t>( count ) * sizeof( Real ) );
  }
}

static void copyPrefix( ArrayView<Real> destination, const DeviceBuffer<Real>& source, Index count,
                        const char* operation )
{
  if ( destination.size() < count ) {
    throw std::invalid_argument( std::string( operation ) + " has an undersized destination." );
  }
  if ( count > 0 ) {
    ActiveDeviceBackend::copy( destination.data(), source.data(), static_cast<std::size_t>( count ) * sizeof( Real ) );
  }
}

void addDerivative( FieldView<Real> output, Index offset, Index count, const char* operation )
{
  const auto& surface = offset == result_layout_.mortar_force ? host_surfaces_.mortar : host_surfaces_.nonmortar;
  if ( !output.isStructurallyValid() || output.entities != surface.numberOfNodes() ||
       output.components != surface.dimension ) {
    throw std::invalid_argument( std::string( operation ) + " has a mismatched destination field." );
  }
  if ( count > 0 ) {
    ActiveDeviceBackend::copy( host_derivative_.data(), derivative_.data() + offset,
                               static_cast<std::size_t>( count ) * sizeof( Real ) );
    for ( Index node = 0; node < surface.numberOfNodes(); ++node ) {
      for ( int component = 0; component < surface.dimension; ++component ) {
        output( node, component ) += host_derivative_[static_cast<std::size_t>( node * surface.dimension + component )];
      }
    }
  }
}

SurfacePairView host_surfaces_{};
SurfacePairView device_surfaces_{};
DeviceSurfaceStorage mortar_;
DeviceSurfaceStorage nonmortar_;
DeviceStateStorage state_;
bool surfaces_ready_{};

DeviceBuffer<DeviceBounds> mortar_bounds_;
DeviceBuffer<DeviceBounds> global_bounds_;
DeviceBuffer<std::uint32_t> morton_keys_;
DeviceBuffer<std::uint32_t> morton_keys_alt_;
DeviceBuffer<Index> mortar_order_;
DeviceBuffer<Index> mortar_order_alt_;
DeviceBuffer<DeviceBvhNode> bvh_nodes_;
Index bvh_root_{};
DeviceBuffer<Index> candidate_counts_;
DeviceBuffer<Index> candidate_offsets_;
DeviceBuffer<ElementPair> candidates_;
DeviceBuffer<ElementPair> candidate_scratch_;
DeviceBuffer<Index> candidate_keys_;
DeviceBuffer<Index> candidate_keys_alt_;
Index candidate_count_{};

DeviceBuffer<InteractionPatch> patches_;
DeviceBuffer<PenaltyContribution> contributions_;
DeviceBuffer<Index> active_flags_;
DeviceBuffer<Index> active_offsets_;
DeviceBuffer<Real> quadrature_gap_;
DeviceBuffer<Real> quadrature_pressure_;
DeviceBuffer<Real> timestep_votes_;
DeviceBuffer<EvaluationSummary> summary_entries_;
DeviceBuffer<EvaluationSummary> summary_;
EvaluationSummary last_summary_{};

ResultLayout result_layout_{};
DeviceBuffer<Real> results_;
DeviceBuffer<Real> gap_;
DeviceBuffer<Real> derivative_;
DeviceBuffer<Real> mortar_direction_;
DeviceBuffer<Real> nonmortar_direction_;
DeviceBuffer<linearization_detail::ExactTangent> exact_mortar_coordinates_;
DeviceBuffer<linearization_detail::ExactTangent> exact_nonmortar_coordinates_;
std::vector<Real> host_derivative_;
DeviceBuffer<std::uint64_t> scatter_keys_;
DeviceBuffer<std::uint64_t> scatter_keys_alt_;
DeviceBuffer<Real> scatter_values_;
DeviceBuffer<Real> scatter_values_alt_;
DeviceBuffer<std::uint64_t> unique_keys_;
DeviceBuffer<Index> run_counts_;
DeviceBuffer<Index> run_offsets_;
DeviceBuffer<Index> run_count_;
DeviceBuffer<unsigned char> temporary_storage_;
