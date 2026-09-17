[[nodiscard]] std::unique_ptr<::mfem::HypreParMatrix> assembleCoordinateJacobian( const ContactStateView& state,
                                                                                  Real drop_tolerance = 0.0 )
{
  if ( drop_tolerance < 0.0 ) {
    throw std::invalid_argument( "MFEM Jacobian drop tolerance cannot be negative." );
  }
  const int local_size = coordinate_space_.GetTrueVSize();
  const HYPRE_BigInt global_size = coordinate_space_.GlobalTrueVSize();
  const HYPRE_BigInt local_begin = coordinate_space_.GetMyTDofOffset();
  std::vector<Real> dense( static_cast<std::size_t>( local_size ) * static_cast<std::size_t>( global_size ) );
  ::mfem::Vector direction( local_size );
  ::mfem::Vector action( local_size );
  for ( HYPRE_BigInt column = 0; column < global_size; ++column ) {
    direction = 0.0;
    action = 0.0;
    if ( column >= local_begin && column < local_begin + local_size ) {
      direction[static_cast<int>( column - local_begin )] = 1.0;
    }
    addCoordinateJacobianMult( state, direction, action );
    for ( int row = 0; row < local_size; ++row ) {
      dense[static_cast<std::size_t>( row ) * static_cast<std::size_t>( global_size ) +
            static_cast<std::size_t>( column )] = action[row];
    }
  }
  std::vector<int> row_offsets( static_cast<std::size_t>( local_size ) + 1 );
  std::vector<HYPRE_BigInt> columns;
  std::vector<::mfem::real_t> values;
  for ( int row = 0; row < local_size; ++row ) {
    for ( HYPRE_BigInt column = 0; column < global_size; ++column ) {
      const Real value = dense[static_cast<std::size_t>( row ) * static_cast<std::size_t>( global_size ) +
                               static_cast<std::size_t>( column )];
      if ( std::abs( value ) > drop_tolerance ) {
        columns.push_back( column );
        values.push_back( value );
      }
    }
    row_offsets[static_cast<std::size_t>( row + 1 )] = static_cast<int>( columns.size() );
  }
  return std::make_unique<::mfem::HypreParMatrix>(
      mesh_.GetComm(), local_size, global_size, global_size, row_offsets.data(), columns.data(), values.data(),
      coordinate_space_.GetTrueDofOffsets(), coordinate_space_.GetTrueDofOffsets() );
}

[[nodiscard]] std::unique_ptr<::mfem::HypreParMatrix> assembleCoordinateJacobian( const ContactState& state,
                                                                                  Real drop_tolerance = 0.0 )
{
  return assembleCoordinateJacobian( restrictState( state ), drop_tolerance );
}

[[nodiscard]] std::unique_ptr<JacobianBlocks> assembleSystemJacobian( const ContactState& state,
                                                                      Real drop_tolerance = 0.0 )
{
  if constexpr ( !MethodTraits<MethodType>::capabilities.needs_multiplier ) {
    throw std::logic_error( "A two-block MFEM Jacobian requires multiplier enforcement." );
  } else {
    if ( state.multiplier == nullptr ) {
      throw std::invalid_argument( "A two-block MFEM Jacobian requires an MFEM multiplier field." );
    }
    const auto& multiplier_space = requireScalarSpace( *state.multiplier );
    const ContactStateView core_state = restrictState( state );
    auto force_coordinates = assembleCoordinateJacobian( core_state, drop_tolerance );
    auto force_multiplier = assembleParallelOperator(
        coordinate_space_, multiplier_space,
        [&]( const ::mfem::Vector& direction, ::mfem::Vector& action ) {
          addMultiplierJacobianMult( state, direction, action );
        },
        drop_tolerance );
    auto constraint_coordinates = assembleParallelOperator(
        multiplier_space, coordinate_space_,
        [&]( const ::mfem::Vector& direction, ::mfem::Vector& action ) {
          addConstraintCoordinateJacobianMult( state, direction, action );
        },
        drop_tolerance );
    const auto active = activeMultiplierRows( state, multiplier_space );
    auto constraint_multiplier = inactiveIdentity( multiplier_space, active );
    return std::make_unique<JacobianBlocks>( coordinate_space_.GetTrueVSize(), multiplier_space.GetTrueVSize(),
                                             std::move( force_coordinates ), std::move( force_multiplier ),
                                             std::move( constraint_coordinates ), std::move( constraint_multiplier ) );
  }
}

[[nodiscard]] std::unique_ptr<::mfem::HypreParMatrix> assembleExternalPressureJacobian( const ContactState& state,
                                                                                        Real drop_tolerance = 0.0 )
{
  if constexpr ( !MethodTraits<MethodType>::capabilities.accepts_external_pressure ) {
    throw std::logic_error( "An external-pressure Jacobian requires external-pressure enforcement." );
  } else {
    if ( state.external_pressure == nullptr ) {
      throw std::invalid_argument( "An external-pressure Jacobian requires an MFEM pressure field." );
    }
    const auto& pressure_space = requireScalarSpace( *state.external_pressure );
    return assembleParallelOperator(
        coordinate_space_, pressure_space,
        [&]( const ::mfem::Vector& direction, ::mfem::Vector& action ) {
          addExternalPressureJacobianMult( state, direction, action );
        },
        drop_tolerance );
  }
}
