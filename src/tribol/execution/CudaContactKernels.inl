struct ResultLayout {
  Index mortar_force{};
  Index nonmortar_force{};
  Index weighted_gap{};
  Index tributary_area{};
  Index pressure{};
  Index total{};
};

ResultLayout makeResultLayout( const SurfacePairView& surfaces )
{
  ResultLayout layout;
  layout.nonmortar_force = surfaces.mortar.numberOfNodes() * surfaces.mortar.dimension;
  layout.weighted_gap = layout.nonmortar_force + surfaces.nonmortar.numberOfNodes() * surfaces.nonmortar.dimension;
  layout.tributary_area = layout.weighted_gap + surfaces.mortar.numberOfNodes();
  layout.pressure = layout.tributary_area + surfaces.mortar.numberOfNodes();
  layout.total = layout.pressure + surfaces.mortar.numberOfNodes();
  return layout;
}

inline constexpr int maximumScatterEntries = 2 * basis::maximumLinearNodes * 3 + 3 * basis::maximumLinearNodes;
inline constexpr std::uint64_t invalidScatterKey = std::numeric_limits<std::uint64_t>::max();

TRIBOL_HOST_DEVICE void emitEntry( std::uint64_t* keys, Real* values, int& entry, std::uint64_t key, Real value )
{
  keys[entry] = key;
  values[entry] = value;
  ++entry;
}

TRIBOL_HOST_DEVICE bool excessivePenetration( const constraint::GapActivationParameters& activation,
                                              const ContactStateView& state, ElementPair pair, Real effective_gap )
{
  if ( !activation.reject_excessive_penetration ) {
    return false;
  }
  const Real mortar_thickness = state.mortar_element_thickness[pair.mortar_element];
  const Real nonmortar_thickness = state.nonmortar_element_thickness[pair.nonmortar_element];
  const Real minimum_thickness = mortar_thickness < nonmortar_thickness ? mortar_thickness : nonmortar_thickness;
  return effective_gap < -activation.maximum_penetration_fraction * minimum_thickness;
}

TRIBOL_HOST_DEVICE void evaluateCommonPlane( Index interaction, SurfacePairView surfaces, const ElementPair* candidates,
                                             Index count, DefaultMethod::Parameters parameters, ContactStateView state,
                                             ResultLayout layout, InteractionPatch* patches,
                                             PenaltyContribution* contributions, std::uint64_t* scatter_keys,
                                             Real* scatter_values, Index* active_flags )
{
  if ( interaction >= count ) {
    return;
  }
  std::uint64_t* keys = scatter_keys + static_cast<std::size_t>( interaction ) * maximumScatterEntries;
  Real* values = scatter_values + static_cast<std::size_t>( interaction ) * maximumScatterEntries;
  for ( int entry = 0; entry < maximumScatterEntries; ++entry ) {
    keys[entry] = invalidScatterKey;
    values[entry] = 0.0;
  }

  const ElementPair pair = candidates[interaction];
  InteractionPatch patch = projectedOverlapPatch<normal::MeanPlane>( surfaces, pair, parameters.geometry );
  patches[interaction] = patch;
  PenaltyContribution contribution;
  if ( patch.valid ) {
    contribution = evaluatePointwisePenaltyPatch(
        patch,
        ::tribol::pointwise_penalty_detail::seriesStiffness(
            parameters.enforcement.stiffness.mortar_scale * parameters.enforcement.stiffness.value,
            parameters.enforcement.stiffness.nonmortar_scale * parameters.enforcement.stiffness.value ),
        parameters.constraint.activation );
    if ( excessivePenetration( parameters.constraint.activation, state, pair, contribution.effective_gap ) ) {
      contribution = {};
    }
  }
  contributions[interaction] = contribution;
  active_flags[interaction] = contribution.active ? 1 : 0;
  if ( !contribution.active ) {
    return;
  }

  const auto sample = constraint::stageNormalConstraint<constraint::Pointwise>(
      surfaces, pair, patch.mortar_centroid, patch.nonmortar_centroid, patch.normal, patch.measure );
  int entry{};
  const Index mortar_begin = surfaces.mortar.element_offsets[pair.mortar_element];
  for ( int local_node = 0; local_node < sample.mortar_trial.size; ++local_node ) {
    const Index node = surfaces.mortar.connectivity[mortar_begin + local_node];
    for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
      emitEntry( keys, values, entry,
                 static_cast<std::uint64_t>( layout.mortar_force + node * surfaces.mortar.dimension + component ),
                 sample.mortar_trial[local_node] * contribution.mortar_force[component] );
    }
    const Real weight = patch.measure * sample.mortar_test[local_node];
    emitEntry( keys, values, entry, static_cast<std::uint64_t>( layout.weighted_gap + node ),
               weight * contribution.effective_gap );
    emitEntry( keys, values, entry, static_cast<std::uint64_t>( layout.tributary_area + node ), weight );
    emitEntry( keys, values, entry, static_cast<std::uint64_t>( layout.pressure + node ),
               weight * contribution.pressure );
  }
  const Index nonmortar_begin = surfaces.nonmortar.element_offsets[pair.nonmortar_element];
  for ( int local_node = 0; local_node < sample.nonmortar_trial.size; ++local_node ) {
    const Index node = surfaces.nonmortar.connectivity[nonmortar_begin + local_node];
    for ( int component = 0; component < surfaces.nonmortar.dimension; ++component ) {
      emitEntry( keys, values, entry,
                 static_cast<std::uint64_t>( layout.nonmortar_force + node * surfaces.nonmortar.dimension + component ),
                 -sample.nonmortar_trial[local_node] * contribution.mortar_force[component] );
    }
  }
}

TRIBOL_HOST_DEVICE void compactQuadrature( Index interaction, const PenaltyContribution* contributions,
                                           const Index* active_offsets, Index count, Real* gap, Real* pressure )
{
  if ( interaction < count && contributions[interaction].active ) {
    const Index output = active_offsets[interaction];
    gap[output] = contributions[interaction].effective_gap;
    pressure[output] = contributions[interaction].pressure;
  }
}

TRIBOL_HOST_DEVICE void reduceScatterRun( Index run, const std::uint64_t* unique_keys, const Index* run_counts,
                                          const Index* run_offsets, const Index* run_count, const Real* sorted_values,
                                          Real* output )
{
  if ( run >= *run_count || unique_keys[run] == invalidScatterKey ) {
    return;
  }
  Real sum{};
  const Index begin = run_offsets[run];
  const Index end = begin + run_counts[run];
  for ( Index entry = begin; entry < end; ++entry ) {
    sum += sorted_values[entry];
  }
  output[unique_keys[run]] = sum;
}

TRIBOL_HOST_DEVICE void computeGap( Index node, ResultLayout layout, Index mortar_nodes, Real* result, Real* gap )
{
  if ( node >= mortar_nodes ) {
    return;
  }
  const Real area = result[layout.tributary_area + node];
  gap[node] = ( area > 1.0e-28 || area < -1.0e-28 ) ? result[layout.weighted_gap + node] / area : 0.0;
  if ( area > 1.0e-28 || area < -1.0e-28 ) {
    result[layout.pressure + node] /= area;
  }
}

TRIBOL_HOST_DEVICE std::array<Real, 3> interpolateField( const SurfaceMeshView& mesh, Index element,
                                                         const FieldView<const Real>& field,
                                                         const basis::ShapeValues& shape )
{
  std::array<Real, 3> result{};
  const Index begin = mesh.element_offsets[element];
  for ( int local_node = 0; local_node < shape.size; ++local_node ) {
    const Index node = mesh.connectivity[begin + local_node];
    for ( int component = 0; component < mesh.dimension; ++component ) {
      result[component] += shape[local_node] * field( node, component );
    }
  }
  return result;
}

TRIBOL_HOST_DEVICE Real vectorDot( const std::array<Real, 3>& left, const std::array<Real, 3>& right, int dimension )
{
  Real result{};
  for ( int component = 0; component < dimension; ++component ) {
    result += left[component] * right[component];
  }
  return result;
}

TRIBOL_HOST_DEVICE void includePositiveVote( Real candidate, Real& vote )
{
  if ( candidate > 0.0 && candidate < vote ) {
    vote = candidate;
  }
}

TRIBOL_HOST_DEVICE void timestepVote( Index interaction, SurfacePairView surfaces, const ElementPair* candidates,
                                      Index count, const InteractionPatch* patches, DefaultMethod::Parameters method,
                                      ContactStateView state, timestep::Kinematic::Parameters parameters, Real* votes )
{
  if ( interaction >= count ) {
    return;
  }
  Real vote = std::numeric_limits<Real>::infinity();
  const InteractionPatch& patch = patches[interaction];
  if ( !patch.valid ) {
    votes[interaction] = vote;
    return;
  }
  const ElementPair pair = candidates[interaction];
  const auto sample = constraint::stageNormalConstraint<constraint::Pointwise>(
      surfaces, pair, patch.mortar_centroid, patch.nonmortar_centroid, patch.normal, patch.measure );
  const auto mortar_velocity =
      interpolateField( surfaces.mortar, pair.mortar_element, state.mortar_velocity, sample.mortar_trial );
  const auto nonmortar_velocity =
      interpolateField( surfaces.nonmortar, pair.nonmortar_element, state.nonmortar_velocity, sample.nonmortar_trial );
  const auto mortar_normal = surfaces.mortar.dimension == 2
                                 ? projected_overlap_detail::segmentNormal( surfaces.mortar, pair.mortar_element )
                                 : projected_overlap_detail::faceNormal( surfaces.mortar, pair.mortar_element );
  const auto nonmortar_normal =
      surfaces.nonmortar.dimension == 2
          ? projected_overlap_detail::segmentNormal( surfaces.nonmortar, pair.nonmortar_element )
          : projected_overlap_detail::faceNormal( surfaces.nonmortar, pair.nonmortar_element );
  const int dimension = surfaces.mortar.dimension;
  const auto nonzero_projection = [&]( Real value ) {
    return value + ( value >= 0.0 ? parameters.velocity_tolerance : -parameters.velocity_tolerance );
  };
  const Real mortar_overlap_velocity = nonzero_projection( vectorDot( mortar_velocity, patch.normal, dimension ) );
  const Real nonmortar_overlap_velocity =
      nonzero_projection( vectorDot( nonmortar_velocity, patch.normal, dimension ) );
  const Real mortar_normal_velocity = nonzero_projection( vectorDot( mortar_velocity, mortar_normal, dimension ) );
  const Real nonmortar_normal_velocity =
      nonzero_projection( vectorDot( nonmortar_velocity, nonmortar_normal, dimension ) );
  const Real mortar_limit = parameters.penetration_fraction * state.mortar_element_thickness[pair.mortar_element];
  const Real nonmortar_limit =
      parameters.penetration_fraction * state.nonmortar_element_thickness[pair.nonmortar_element];
  std::array<Real, 3> gap_vector{};
  std::array<Real, 3> projected_gap{};
  for ( int component = 0; component < dimension; ++component ) {
    gap_vector[component] = patch.mortar_centroid[component] - patch.nonmortar_centroid[component];
    projected_gap[component] = gap_vector[component] +
                               parameters.current_step * ( mortar_velocity[component] - nonmortar_velocity[component] );
  }
  const Real mortar_gap = vectorDot( gap_vector, mortar_normal, dimension );
  const Real nonmortar_gap = vectorDot( gap_vector, nonmortar_normal, dimension );
  const Real projected_mortar_penetration = vectorDot( projected_gap, nonmortar_normal, dimension );
  const Real projected_nonmortar_penetration = -vectorDot( projected_gap, mortar_normal, dimension );
  const bool mortar_closing = mortar_overlap_velocity < 0.0;
  const bool nonmortar_closing = nonmortar_overlap_velocity > 0.0;
  const bool active =
      sample.gap - method.constraint.activation.residual_gap <= method.constraint.activation.gap_tolerance;
  const bool mortar_gap_exceeded = active && mortar_closing && mortar_limit - mortar_gap < 0.0;
  const bool nonmortar_gap_exceeded = active && nonmortar_closing && nonmortar_limit + nonmortar_gap < 0.0;
  if ( mortar_gap_exceeded ) {
    includePositiveVote( parameters.scale * mortar_limit / mortar_normal_velocity, vote );
  }
  if ( nonmortar_gap_exceeded ) {
    includePositiveVote( parameters.scale * nonmortar_limit / nonmortar_normal_velocity, vote );
  }
  if ( mortar_closing && !mortar_gap_exceeded && projected_mortar_penetration < -mortar_limit ) {
    includePositiveVote( parameters.scale * mortar_limit / mortar_normal_velocity, vote );
  }
  if ( nonmortar_closing && !nonmortar_gap_exceeded && projected_nonmortar_penetration < -nonmortar_limit ) {
    includePositiveVote( parameters.scale * nonmortar_limit / nonmortar_normal_velocity, vote );
  }
  votes[interaction] = vote;
}

struct SummaryReduction {
  TRIBOL_HOST_DEVICE EvaluationSummary operator()( const EvaluationSummary& left, const EvaluationSummary& right ) const
  {
    return {
        .energy = left.energy + right.energy,
        .timestep_vote = left.timestep_vote < right.timestep_vote ? left.timestep_vote : right.timestep_vote,
        .active_interactions = left.active_interactions + right.active_interactions,
        .quadrature_points = left.quadrature_points + right.quadrature_points,
    };
  }
};

TRIBOL_HOST_DEVICE void makeSummaryEntry( Index interaction, const PenaltyContribution* contributions,
                                          const Real* votes, Index count, Real initial_vote,
                                          EvaluationSummary* entries )
{
  if ( interaction >= count ) {
    return;
  }
  const Index active = contributions[interaction].active ? 1 : 0;
  entries[interaction] = {
      .energy = contributions[interaction].energy,
      .timestep_vote = votes != nullptr && votes[interaction] < initial_vote ? votes[interaction] : initial_vote,
      .active_interactions = active,
      .quadrature_points = active,
  };
}

template <typename Scalar>
TRIBOL_HOST_DEVICE void seedCoordinate( Index value, FieldView<const Real> coordinates, FieldView<const Real> direction,
                                        Scalar* seeded )
{
  const Index total = coordinates.entities * coordinates.components;
  if ( value < total ) {
    const Index node = value / coordinates.components;
    const int component = static_cast<int>( value % coordinates.components );
    seeded[value] = Scalar{ coordinates( node, component ), direction( node, component ) };
  }
}

TRIBOL_HOST_DEVICE void evaluateDerivative( Index interaction,
                                            SurfacePairViewT<linearization_detail::ExactTangent> surfaces,
                                            const ElementPair* candidates, Index count,
                                            DefaultMethod::Parameters parameters, ContactStateView state,
                                            ResultLayout layout, std::uint64_t* scatter_keys, Real* scatter_values )
{
  using Tangent = linearization_detail::ExactTangent;
  if ( interaction >= count ) {
    return;
  }
  std::uint64_t* keys = scatter_keys + static_cast<std::size_t>( interaction ) * maximumScatterEntries;
  Real* values = scatter_values + static_cast<std::size_t>( interaction ) * maximumScatterEntries;
  for ( int entry = 0; entry < maximumScatterEntries; ++entry ) {
    keys[entry] = invalidScatterKey;
    values[entry] = 0.0;
  }
  const ElementPair pair = candidates[interaction];
  const auto patch = projectedOverlapPatch<normal::MeanPlane>( surfaces, pair, parameters.geometry );
  if ( !patch.valid ) {
    return;
  }
  Tangent gap{};
  for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
    gap += ( patch.mortar_centroid[component] - patch.nonmortar_centroid[component] ) * patch.normal[component];
  }
  const Tangent effective_gap = gap - Tangent{ parameters.constraint.activation.residual_gap };
  if ( excessivePenetration( parameters.constraint.activation, state, pair,
                             linearization_detail::primal( effective_gap ) ) ) {
    return;
  }
  const Tangent active_gap =
      linearization_detail::primal( effective_gap ) <= parameters.constraint.activation.gap_tolerance ? effective_gap
                                                                                                      : Tangent{};
  const Real stiffness = ::tribol::pointwise_penalty_detail::seriesStiffness(
      parameters.enforcement.stiffness.mortar_scale * parameters.enforcement.stiffness.value,
      parameters.enforcement.stiffness.nonmortar_scale * parameters.enforcement.stiffness.value );
  std::array<Tangent, 3> force{};
  for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
    force[component] = patch.measure * Tangent{ stiffness } * active_gap * patch.normal[component];
  }
  const auto sample = constraint::stageNormalConstraint<constraint::Pointwise>(
      surfaces, pair, patch.mortar_centroid, patch.nonmortar_centroid, patch.normal, patch.measure );
  int entry{};
  const Index mortar_begin = surfaces.mortar.element_offsets[pair.mortar_element];
  for ( int local_node = 0; local_node < sample.mortar_trial.size; ++local_node ) {
    const Index node = surfaces.mortar.connectivity[mortar_begin + local_node];
    for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
      emitEntry( keys, values, entry,
                 static_cast<std::uint64_t>( layout.mortar_force + node * surfaces.mortar.dimension + component ),
                 linearization_detail::tangent( sample.mortar_trial[local_node] * force[component] ) );
    }
  }
  const Index nonmortar_begin = surfaces.nonmortar.element_offsets[pair.nonmortar_element];
  for ( int local_node = 0; local_node < sample.nonmortar_trial.size; ++local_node ) {
    const Index node = surfaces.nonmortar.connectivity[nonmortar_begin + local_node];
    for ( int component = 0; component < surfaces.nonmortar.dimension; ++component ) {
      emitEntry( keys, values, entry,
                 static_cast<std::uint64_t>( layout.nonmortar_force + node * surfaces.nonmortar.dimension + component ),
                 -linearization_detail::tangent( sample.nonmortar_trial[local_node] * force[component] ) );
    }
  }
}

template <typename T>
ArrayView<const T> constDeviceView( const DeviceBuffer<T>& buffer )
{
  return { buffer.data(), static_cast<Index>( buffer.size() ) };
}

template <typename T>
ArrayView<T> deviceView( DeviceBuffer<T>& buffer )
{
  return { buffer.data(), static_cast<Index>( buffer.size() ) };
}
