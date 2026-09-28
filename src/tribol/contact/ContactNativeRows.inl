static void requireValidNativeInput( ParentTracePairView traces, ParentTraceFieldPairView fields )
{
  if ( !traces.isStructurallyValid() ) {
    throw std::invalid_argument( "Native contact requires structurally valid parent traces." );
  }
  constraint_row_detail::requireField( fields.mortar_coordinates, traces.mortar,
                                       "Native contact requires mortar parent coordinates." );
  constraint_row_detail::requireField( fields.nonmortar_coordinates, traces.nonmortar,
                                       "Native contact requires nonmortar parent coordinates." );
}

[[nodiscard]] Index resultEntities( bool mortar ) const
{
  if constexpr ( detail::is_native_explicit_method_v<MethodType> ) {
    return mortar ? parent_fields_.mortar_coordinates.entities : parent_fields_.nonmortar_coordinates.entities;
  } else {
    return mortar ? surfaces_.mortar.numberOfNodes() : surfaces_.nonmortar.numberOfNodes();
  }
}

[[nodiscard]] FieldView<const Real> resultField( const std::vector<Real>& values, bool mortar ) const
{
  return { constView( values ), resultEntities( mortar ), surfaces_.mortar.dimension, FieldLayout::Interleaved };
}

void prepareInteractionWorkspace()
{
  if constexpr ( detail::is_native_explicit_method_v<MethodType> ) {
    prepareNativeRows();
  } else {
    resizeInteractionWorkspace();
  }
}

void prepareNativeRows()
{
  using Geometry = typename MethodType::geometry_policy;
  using Enforcement = typename MethodType::enforcement_policy;
  constexpr bool include_normal_rates = !std::same_as<typename Enforcement::rate_policy, rate::None>;
  Index row_count{};
  if constexpr ( execution::DevicePolicy<Execution> ) {
    native_cells_.reset();
    native_rows_.reset();
    execution::DeviceNativeRowOptions row_options;
    row_options.geometry = options_.method.geometry;
    row_options.quadrature = options_.method.integration;
    row_options.kinematic = options_.timestep;
    row_options.include_normal_rates = include_normal_rates;
    native_device_workspace_.prepareNative(
        parent_traces_, parent_fields_, execution_workspace_.deviceCandidates(),
        execution::DeviceRowDofLayout::separate( surfaces_.mortar.dimension, resultEntities( true ),
                                                 resultEntities( false ) ),
        row_options );
    row_count = native_device_workspace_.deviceResult( {} ).rows.numberOfRows();
  } else {
    native_cells_.emplace( buildOverlapCellBatch<typename Geometry::normal_policy>( parent_traces_, candidates_.view(),
                                                                                    options_.method.geometry ) );
    ParentTraceFieldPairView fields;
    fields.mortar_coordinates = parent_fields_.mortar_coordinates;
    fields.nonmortar_coordinates = parent_fields_.nonmortar_coordinates;
    ConstraintRowBuildOptions row_options;
    row_options.quadrature = options_.method.integration;
    row_options.include_normal_rates = include_normal_rates;
    native_rows_.emplace( buildConstraintRowBatch( parent_traces_, native_cells_->view(), fields, row_options ) );
    native_diagnostic_quadrature_.reserve( native_cells_->view() );
    native_diagnostic_quadrature_current_ = false;
    row_count = native_rows_->numberOfRows();
  }
  const std::size_t rows = static_cast<std::size_t>( row_count );
  result_quadrature_gap_.resize( rows );
  result_quadrature_pressure_.resize( rows );
  native_row_force_.resize( rows );
  native_row_stiffness_.resize( rows );
  native_row_rate_.resize( rows );
  native_pressure_stiffness_.resize( rows );
  if constexpr ( std::same_as<typename Enforcement::rate_policy, rate::None> ) {
    native_rate_coefficient_.clear();
  } else {
    native_rate_coefficient_.resize( rows );
  }
  const auto& parameters = options_.method.enforcement;
  const Real stiffness =
      enforcement::seriesPenaltyStiffness( parameters.stiffness.mortar_scale * parameters.stiffness.value,
                                           parameters.stiffness.nonmortar_scale * parameters.stiffness.value );
  const Real rate_coefficient = enforcement::penaltyRateCoefficient<Enforcement>( parameters, stiffness );
  std::fill( native_pressure_stiffness_.begin(), native_pressure_stiffness_.end(), stiffness );
  std::fill( native_rate_coefficient_.begin(), native_rate_coefficient_.end(), rate_coefficient );
  native_mortar_row_sums_.resize( result_mortar_force_.size() );
  native_nonmortar_row_sums_.resize( result_nonmortar_force_.size() );
  native_mortar_rate_row_sums_.resize( result_mortar_force_.size() );
  native_nonmortar_rate_row_sums_.resize( result_nonmortar_force_.size() );
  native_stability_current_ = false;
  native_device_force_.resize( result_mortar_force_.size() + result_nonmortar_force_.size() );
}

[[nodiscard]] execution::DeviceRowEvaluationSummary evaluateNativeDeviceRows(
    const ContactStateView& state, timestep::DiagonalInverseMassView inverse_mass,
    timestep::IntegratorStabilityLimits integrator ) const
  requires execution::DevicePolicy<Execution>
{
  using Enforcement = typename MethodType::enforcement_policy;
  ParentTraceFieldPairView fields{ .mortar_coordinates = parent_fields_.mortar_coordinates,
                                   .nonmortar_coordinates = parent_fields_.nonmortar_coordinates };
  if constexpr ( !std::same_as<typename Enforcement::rate_policy, rate::None> ) {
    fields.mortar_velocity = state.mortar_velocity;
    fields.nonmortar_velocity = state.nonmortar_velocity;
  } else if ( options_.timestep.enabled && options_.timestep.current_step >= options_.timestep.minimum_current_step ) {
    fields.mortar_velocity = state.mortar_velocity;
    fields.nonmortar_velocity = state.nonmortar_velocity;
  }
  native_device_workspace_.updateNativeFields( fields );
  PenaltyRowInputView penalty;
  penalty.pressure_stiffness = { native_pressure_stiffness_.data(),
                                 static_cast<Index>( native_pressure_stiffness_.size() ) };
  penalty.rate_coefficients = { native_rate_coefficient_.data(),
                                static_cast<Index>( native_rate_coefficient_.size() ) };
  penalty.activation = options_.method.constraint.activation;
  last_native_device_summary_ = native_device_workspace_.evaluate(
      penalty, inverse_mass, integrator, geometry_version_, interaction_version_, penalty_version_,
      { state.mortar_element_thickness, state.nonmortar_element_thickness } );
  last_device_summary_ = { .energy = last_native_device_summary_.explicit_response.energy,
                           .timestep_vote = std::min( last_native_device_summary_.stability.timestep_vote,
                                                      last_native_device_summary_.kinematic_timestep_vote ),
                           .active_interactions = last_native_device_summary_.explicit_response.active_rows,
                           .quadrature_points = last_native_device_summary_.explicit_response.rows };
  device_result_geometry_version_ = geometry_version_;
  device_result_interaction_version_ = interaction_version_;
  return last_native_device_summary_;
}

[[nodiscard]] ContactResultView nativeDeviceResult() const
  requires execution::DevicePolicy<Execution>
{
  const auto result = native_device_workspace_.deviceResult( last_native_device_summary_ );
  return {
      .mortar_force = result.mortar_force,
      .nonmortar_force = result.nonmortar_force,
      .quadrature_gap = result.row_output.effective_gap,
      .quadrature_pressure = result.row_output.pressure,
      .summary = last_device_summary_,
      .geometry_version = device_result_geometry_version_,
      .interaction_version = device_result_interaction_version_,
  };
}

[[nodiscard]] EvaluationSummary evaluateNativeRows( const ContactStateView& state, ContactOutputView output,
                                                    timestep::DiagonalInverseMassView inverse_mass,
                                                    timestep::IntegratorStabilityLimits integrator ) const
{
  using Enforcement = typename MethodType::enforcement_policy;
  if constexpr ( execution::DevicePolicy<Execution> ) {
    static_cast<void>( evaluateNativeDeviceRows( state, inverse_mass, integrator ) );
    PenaltyRowOutputView row_output;
    row_output.effective_gap = output.quadrature_gap;
    row_output.pressure = output.quadrature_pressure;
    row_output.row_force = { native_row_force_.data(), static_cast<Index>( native_row_force_.size() ) };
    row_output.row_stiffness = { native_row_stiffness_.data(), static_cast<Index>( native_row_stiffness_.size() ) };
    row_output.row_rate_coefficient = { native_row_rate_.data(), static_cast<Index>( native_row_rate_.size() ) };
    native_device_workspace_.download( mutableView( native_device_force_ ), row_output );
    for ( Index entity = 0; entity < output.residual.mortar.entities; ++entity ) {
      for ( int component = 0; component < output.residual.mortar.components; ++component ) {
        output.residual.mortar( entity, component ) +=
            native_device_force_[static_cast<std::size_t>( entity * output.residual.mortar.components + component )];
      }
    }
    const Index nonmortar_offset = output.residual.mortar.entities * output.residual.mortar.components;
    for ( Index entity = 0; entity < output.residual.nonmortar.entities; ++entity ) {
      for ( int component = 0; component < output.residual.nonmortar.components; ++component ) {
        output.residual.nonmortar( entity, component ) += native_device_force_[static_cast<std::size_t>(
            nonmortar_offset + entity * output.residual.nonmortar.components + component )];
      }
    }
    return last_device_summary_;
  }
  if ( !native_rows_ ) {
    throw std::logic_error( "Native contact rows must be prepared before evaluation." );
  }
  ParentTraceFieldPairView fields{ .mortar_coordinates = parent_fields_.mortar_coordinates,
                                   .nonmortar_coordinates = parent_fields_.nonmortar_coordinates };
  if constexpr ( !std::same_as<typename Enforcement::rate_policy, rate::None> ) {
    fields.mortar_velocity = state.mortar_velocity;
    fields.nonmortar_velocity = state.nonmortar_velocity;
  }
  native_rows_->updateFields( fields );
  const auto rows = native_rows_->view();
  const PenaltyRowInputView penalty{
      .pressure_stiffness = { native_pressure_stiffness_.data(), rows.numberOfRows() },
      .rate_coefficients = { native_rate_coefficient_.data(), static_cast<Index>( native_rate_coefficient_.size() ) },
      .activation = options_.method.constraint.activation,
  };
  const auto row_summary =
      evaluateExplicitPenaltyRows( rows, penalty, output.residual,
                                   { .effective_gap = output.quadrature_gap,
                                     .pressure = output.quadrature_pressure,
                                     .row_force = { native_row_force_.data(), rows.numberOfRows() },
                                     .row_stiffness = { native_row_stiffness_.data(), rows.numberOfRows() },
                                     .row_rate_coefficient = { native_row_rate_.data(), rows.numberOfRows() } } );
  std::fill( native_mortar_row_sums_.begin(), native_mortar_row_sums_.end(), 0.0 );
  std::fill( native_nonmortar_row_sums_.begin(), native_nonmortar_row_sums_.end(), 0.0 );
  std::fill( native_mortar_rate_row_sums_.begin(), native_mortar_rate_row_sums_.end(), 0.0 );
  std::fill( native_nonmortar_rate_row_sums_.begin(), native_nonmortar_rate_row_sums_.end(), 0.0 );
  native_stability_current_ = false;
  Real stability_vote = std::numeric_limits<Real>::infinity();
  if ( row_summary.active_stiffness_rows > 0 || row_summary.active_rate_rows > 0 ) {
    const auto stability = timestep::explicitPenaltyStabilityVote(
        rows,
        { .row_stiffness = { native_row_stiffness_.data(), rows.numberOfRows() },
          .row_rate_coefficient = native_rate_coefficient_.empty()
                                      ? ArrayView<const Real>{}
                                      : ArrayView<const Real>{ native_row_rate_.data(), rows.numberOfRows() },
          .inverse_mass = inverse_mass,
          .integrator = integrator,
          .geometry_version = geometry_version_,
          .interaction_version = interaction_version_,
          .penalty_version = penalty_version_ },
        { .mortar_row_sums = { { native_mortar_row_sums_.data(), static_cast<Index>( native_mortar_row_sums_.size() ) },
                               resultEntities( true ),
                               surfaces_.mortar.dimension,
                               FieldLayout::Interleaved },
          .nonmortar_row_sums = {
              { native_nonmortar_row_sums_.data(), static_cast<Index>( native_nonmortar_row_sums_.size() ) },
              resultEntities( false ),
              surfaces_.nonmortar.dimension,
              FieldLayout::Interleaved } } );
    stability_vote = stability.timestep_vote;
    if ( !native_rate_coefficient_.empty() ) {
      std::copy( native_mortar_row_sums_.begin(), native_mortar_row_sums_.end(), native_mortar_rate_row_sums_.begin() );
      std::copy( native_nonmortar_row_sums_.begin(), native_nonmortar_row_sums_.end(),
                 native_nonmortar_rate_row_sums_.begin() );
      if ( row_summary.active_stiffness_rows > 0 ) {
        static_cast<void>( timestep::explicit_stability_detail::rowSumBound(
            rows, { native_row_stiffness_.data(), rows.numberOfRows() }, inverse_mass,
            { .mortar_row_sums = { { native_mortar_row_sums_.data(),
                                     static_cast<Index>( native_mortar_row_sums_.size() ) },
                                   resultEntities( true ),
                                   surfaces_.mortar.dimension,
                                   FieldLayout::Interleaved },
              .nonmortar_row_sums = {
                  { native_nonmortar_row_sums_.data(), static_cast<Index>( native_nonmortar_row_sums_.size() ) },
                  resultEntities( false ),
                  surfaces_.nonmortar.dimension,
                  FieldLayout::Interleaved } } ) );
      } else {
        std::fill( native_mortar_row_sums_.begin(), native_mortar_row_sums_.end(), 0.0 );
        std::fill( native_nonmortar_row_sums_.begin(), native_nonmortar_row_sums_.end(), 0.0 );
      }
    }
  }
  native_stability_current_ = true;
  return { .energy = row_summary.energy,
           .timestep_vote = stability_vote,
           .active_interactions = row_summary.active_rows,
           .quadrature_points = row_summary.rows };
}
