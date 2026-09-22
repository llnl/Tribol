void applyCoordinateDerivative( const ContactStateView& state, ContactDirectionView direction,
                                ContactResidualView derivative ) const
{
  requireDirection( direction );
  applyDerivative( state, ContactLinearizationDirectionView{ .coordinates = direction }, derivative );
}

void applyMultiplierDerivative( const ContactStateView& state, ArrayView<const Real> direction,
                                ContactResidualView derivative ) const
{
  if constexpr ( !MethodTraits<MethodType>::capabilities.needs_multiplier ) {
    throw std::logic_error( "Multiplier derivatives require a multiplier-enforced method." );
  } else {
    applyDerivative( state, ContactLinearizationDirectionView{ .state = { .multiplier = direction } }, derivative );
  }
}

void applyExternalPressureDerivative( const ContactStateView& state, ArrayView<const Real> direction,
                                      ContactResidualView derivative ) const
{
  if constexpr ( !MethodTraits<MethodType>::capabilities.accepts_external_pressure ) {
    throw std::logic_error( "External-pressure derivatives require an external-pressure method." );
  } else {
    applyDerivative( state, ContactLinearizationDirectionView{ .state = { .external_pressure = direction } },
                     derivative );
  }
}

void applyDerivative( const ContactStateView& state, ContactLinearizationDirectionView direction,
                      ContactResidualView derivative ) const
{
  if ( !has_interactions_ ) {
    throw std::logic_error( "updateInteractions() must be called before contact linearization." );
  }
  if constexpr ( execution::DevicePolicy<Execution> ) {
    requireDirection( direction.coordinates );
    execution_workspace_.applyDefaultCoordinateDerivative( options_.method, state, direction.coordinates, derivative );
    return;
  }
  seedCoordinates( surfaces_.mortar, direction.coordinates.mortar, exact_mortar_coordinates_ );
  seedCoordinates( surfaces_.nonmortar, direction.coordinates.nonmortar, exact_nonmortar_coordinates_ );
  const auto exact_state = seededState( state, direction.state );
  std::fill( exact_mortar_residual_.begin(), exact_mortar_residual_.end(), linearization_detail::ExactTangent{} );
  std::fill( exact_nonmortar_residual_.begin(), exact_nonmortar_residual_.end(), linearization_detail::ExactTangent{} );
  std::fill( exact_constraint_residual_.begin(), exact_constraint_residual_.end(),
             linearization_detail::ExactTangent{} );
  std::fill( exact_gap_.begin(), exact_gap_.end(), linearization_detail::ExactTangent{} );
  std::fill( exact_weighted_gap_.begin(), exact_weighted_gap_.end(), linearization_detail::ExactTangent{} );
  std::fill( exact_tributary_area_.begin(), exact_tributary_area_.end(), linearization_detail::ExactTangent{} );
  std::fill( exact_pressure_.begin(), exact_pressure_.end(), linearization_detail::ExactTangent{} );

  const auto exact_surfaces = seededSurfaces();
  evaluateMethod<MethodType>( exact_surfaces, candidates_.view(), options_.method, exact_state,
                              ContactOutputViewT<linearization_detail::ExactTangent>{
                                  .residual = workspaceResidual( exact_mortar_residual_, exact_nonmortar_residual_,
                                                                 mutableView( exact_constraint_residual_ ) ),
                                  .gap = mutableView( exact_gap_ ),
                                  .weighted_gap = mutableView( exact_weighted_gap_ ),
                                  .tributary_area = mutableView( exact_tributary_area_ ),
                                  .pressure = mutableView( exact_pressure_ ) } );
  addTangents( exact_mortar_residual_, derivative.mortar );
  addTangents( exact_nonmortar_residual_, derivative.nonmortar );
  addTangents( exact_constraint_residual_, derivative.constraint );
}

[[nodiscard]] DenseMatrixView assembleCoordinateJacobian( const ContactStateView& state ) const
{
  if ( !has_interactions_ ) {
    throw std::logic_error( "updateInteractions() must be called before contact linearization." );
  }
  const Index mortar_values = surfaces_.mortar.numberOfNodes() * surfaces_.mortar.dimension;
  const Index nonmortar_values = surfaces_.nonmortar.numberOfNodes() * surfaces_.nonmortar.dimension;
  const Index constraint_values =
      MethodTraits<MethodType>::capabilities.produces_constraint_residual ? surfaces_.mortar.numberOfNodes() : 0;
  const Index columns = mortar_values + nonmortar_values;
  const Index rows = columns + constraint_values;
  std::fill( coordinate_jacobian_.begin(), coordinate_jacobian_.end(), 0.0 );
  for ( Index column = 0; column < columns; ++column ) {
    std::fill( direction_mortar_.begin(), direction_mortar_.end(), 0.0 );
    std::fill( direction_nonmortar_.begin(), direction_nonmortar_.end(), 0.0 );
    std::fill( derivative_mortar_.begin(), derivative_mortar_.end(), 0.0 );
    std::fill( derivative_nonmortar_.begin(), derivative_nonmortar_.end(), 0.0 );
    std::fill( derivative_constraint_.begin(), derivative_constraint_.end(), 0.0 );
    if ( column < mortar_values ) {
      direction_mortar_[static_cast<std::size_t>( column )] = 1.0;
    } else {
      direction_nonmortar_[static_cast<std::size_t>( column - mortar_values )] = 1.0;
    }
    applyCoordinateDerivative(
        state,
        ContactDirectionView{ .mortar = constField( direction_mortar_, surfaces_.mortar ),
                              .nonmortar = constField( direction_nonmortar_, surfaces_.nonmortar ) },
        workspaceResidual( derivative_mortar_, derivative_nonmortar_, mutableView( derivative_constraint_ ) ) );
    for ( Index row = 0; row < mortar_values; ++row ) {
      coordinate_jacobian_[static_cast<std::size_t>( row * columns + column )] =
          derivative_mortar_[static_cast<std::size_t>( row )];
    }
    for ( Index row = 0; row < nonmortar_values; ++row ) {
      coordinate_jacobian_[static_cast<std::size_t>( ( mortar_values + row ) * columns + column )] =
          derivative_nonmortar_[static_cast<std::size_t>( row )];
    }
    for ( Index row = 0; row < constraint_values; ++row ) {
      coordinate_jacobian_[static_cast<std::size_t>( ( columns + row ) * columns + column )] =
          derivative_constraint_[static_cast<std::size_t>( row )];
    }
  }
  return { { coordinate_jacobian_.data(), rows * columns }, rows, columns };
}

[[nodiscard]] DenseMatrixView assembleSystemJacobian( const ContactStateView& state ) const
{
  if ( !has_interactions_ ) {
    throw std::logic_error( "updateInteractions() must be called before contact linearization." );
  }
  const Index mortar_values = surfaces_.mortar.numberOfNodes() * surfaces_.mortar.dimension;
  const Index nonmortar_values = surfaces_.nonmortar.numberOfNodes() * surfaces_.nonmortar.dimension;
  const Index coordinate_columns = mortar_values + nonmortar_values;
  const Index multiplier_columns =
      MethodTraits<MethodType>::capabilities.needs_multiplier ? surfaces_.mortar.numberOfNodes() : 0;
  const Index constraint_values = multiplier_columns;
  const Index columns = coordinate_columns + multiplier_columns;
  const Index rows = coordinate_columns + constraint_values;
  std::fill( system_jacobian_.begin(), system_jacobian_.end(), 0.0 );
  for ( Index column = 0; column < columns; ++column ) {
    clearDerivativeWorkspace();
    if ( column < coordinate_columns ) {
      std::fill( direction_mortar_.begin(), direction_mortar_.end(), 0.0 );
      std::fill( direction_nonmortar_.begin(), direction_nonmortar_.end(), 0.0 );
      if ( column < mortar_values ) {
        direction_mortar_[static_cast<std::size_t>( column )] = 1.0;
      } else {
        direction_nonmortar_[static_cast<std::size_t>( column - mortar_values )] = 1.0;
      }
      applyCoordinateDerivative(
          state,
          ContactDirectionView{ .mortar = constField( direction_mortar_, surfaces_.mortar ),
                                .nonmortar = constField( direction_nonmortar_, surfaces_.nonmortar ) },
          workspaceResidual( derivative_mortar_, derivative_nonmortar_, mutableView( derivative_constraint_ ) ) );
    } else {
      std::fill( direction_multiplier_.begin(), direction_multiplier_.end(), 0.0 );
      direction_multiplier_[static_cast<std::size_t>( column - coordinate_columns )] = 1.0;
      applyMultiplierDerivative(
          state, constView( direction_multiplier_ ),
          workspaceResidual( derivative_mortar_, derivative_nonmortar_, mutableView( derivative_constraint_ ) ) );
    }
    writeJacobianColumn( system_jacobian_, rows, columns, column );
  }
  if constexpr ( MethodTraits<MethodType>::capabilities.needs_multiplier ) {
    for ( Index constraint = 0; constraint < constraint_values; ++constraint ) {
      const Index row = coordinate_columns + constraint;
      bool active{};
      for ( Index column = 0; column < coordinate_columns; ++column ) {
        active = active || std::abs( system_jacobian_[static_cast<std::size_t>( row * columns + column )] ) > 1.0e-28;
      }
      if ( !active ) {
        system_jacobian_[static_cast<std::size_t>( row * columns + coordinate_columns + constraint )] = 1.0;
      }
    }
  }
  return { { system_jacobian_.data(), rows * columns }, rows, columns };
}

[[nodiscard]] CsrMatrixView assembleCoordinateJacobianCsr( const ContactStateView& state,
                                                           Real drop_tolerance = 0.0 ) const
{
  return compress( assembleCoordinateJacobian( state ), drop_tolerance );
}

[[nodiscard]] CsrMatrixView assembleSystemJacobianCsr( const ContactStateView& state, Real drop_tolerance = 0.0 ) const
{
  return compress( assembleSystemJacobian( state ), drop_tolerance );
}

[[nodiscard]] const SurfacePairView& surfaces() const { return surfaces_; }
[[nodiscard]] const Options& options() const { return options_; }
[[nodiscard]] ArrayView<const ElementPair> interactions() const
{
  if constexpr ( execution::DevicePolicy<Execution> ) {
    execution_workspace_.downloadCandidates( device_candidate_mirror_ );
    return { device_candidate_mirror_.data(), static_cast<Index>( device_candidate_mirror_.size() ) };
  } else {
    return candidates_.view();
  }
}
[[nodiscard]] bool hasInteractions() const { return has_interactions_; }
