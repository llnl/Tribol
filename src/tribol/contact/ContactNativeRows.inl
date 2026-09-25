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
  native_cells_.emplace( buildOverlapCellBatch<typename Geometry::normal_policy>( parent_traces_, candidates_.view(),
                                                                                  options_.method.geometry ) );
  ParentTraceFieldPairView fields;
  fields.mortar_coordinates = parent_fields_.mortar_coordinates;
  fields.nonmortar_coordinates = parent_fields_.nonmortar_coordinates;
  ConstraintRowBuildOptions row_options;
  row_options.quadrature = options_.method.integration;
  row_options.include_normal_rates = !std::same_as<typename Enforcement::rate_policy, rate::None>;
  native_rows_.emplace( buildConstraintRowBatch( parent_traces_, native_cells_->view(), fields, row_options ) );
  const std::size_t rows = static_cast<std::size_t>( native_rows_->numberOfRows() );
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
}

[[nodiscard]] EvaluationSummary evaluateNativeRows( const ContactStateView& state, ContactOutputView output,
                                                    timestep::DiagonalInverseMassView inverse_mass,
                                                    timestep::IntegratorStabilityLimits integrator ) const
{
  using Enforcement = typename MethodType::enforcement_policy;
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
  Real stability_vote = std::numeric_limits<Real>::infinity();
  if ( row_summary.active_stiffness_rows > 0 || row_summary.active_rate_rows > 0 ) {
    const auto stability = timestep::explicitPenaltyStabilityVote(
        rows,
        { .row_stiffness = { native_row_stiffness_.data(), rows.numberOfRows() },
          .row_rate_coefficient = { native_row_rate_.data(), rows.numberOfRows() },
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
  }
  return { .energy = row_summary.energy,
           .timestep_vote = stability_vote,
           .active_interactions = row_summary.active_rows,
           .quadrature_points = row_summary.rows };
}
