EvaluationSummary finalizedSummary( EvaluationSummary local )
{
  if constexpr ( tribol::detail::is_native_explicit_method_v<MethodType> && !execution::DevicePolicy<Execution> ) {
    prepareNativeStabilityContributions();
  }
  auto global = globalSummary( local );
  if constexpr ( tribol::detail::is_native_explicit_method_v<MethodType> && !execution::DevicePolicy<Execution> ) {
    global.timestep_vote = std::min( global.timestep_vote, nativeStabilityVote() );
  }
  return global;
}

void resizeNativeStabilityWorkspace()
  requires( tribol::detail::is_native_explicit_method_v<MethodType> && !execution::DevicePolicy<Execution> )
{
  true_stiffness_row_sums_.SetSize( coordinate_space_.GetTrueVSize() );
  true_rate_row_sums_.SetSize( coordinate_space_.GetTrueVSize() );
  native_stability_current_ = false;
}

void mapNativeRowSumsToTrueDofs( FieldView<const Real> mortar, FieldView<const Real> nonmortar,
                                 ::mfem::Vector& true_row_sums )
  requires( tribol::detail::is_native_explicit_method_v<MethodType> && !execution::DevicePolicy<Execution> )
{
  local_parent_residual_ = 0.0;
  const int dimension = coordinate_space_.GetVDim();
  const auto add_side = [&]( const detail::SurfaceStorage& surface, FieldView<const Real> distributed ) {
    const auto local = surface.reduceParentDofValues( distributed.values, dimension );
    for ( std::size_t parent_dof = 0; parent_dof < surface.localParentDofCount(); ++parent_dof ) {
      const int scalar_dof = surface.localParentDof( parent_dof );
      for ( int component = 0; component < dimension; ++component ) {
        const int vector_dof = coordinate_space_.DofToVDof( scalar_dof, component );
        local_parent_residual_[detail::decodeDof( vector_dof )] +=
            local[parent_dof * static_cast<std::size_t>( dimension ) + component];
      }
    }
  };
  add_side( surfaces_.mortar(), mortar );
  add_side( surfaces_.nonmortar(), nonmortar );
  true_row_sums = 0.0;
  const auto* prolongation = coordinate_space_.GetProlongationMatrix();
  if ( prolongation == nullptr ) {
    true_row_sums = local_parent_residual_;
  } else {
    prolongation->MultTranspose( local_parent_residual_, true_row_sums );
  }
}

void prepareNativeStabilityContributions()
  requires( tribol::detail::is_native_explicit_method_v<MethodType> && !execution::DevicePolicy<Execution> )
{
  native_stability_current_ = false;
  const auto contributions = core_.explicitStabilityContributions();
  mapNativeRowSumsToTrueDofs( contributions.mortar_stiffness_row_sums, contributions.nonmortar_stiffness_row_sums,
                              true_stiffness_row_sums_ );
  mapNativeRowSumsToTrueDofs( contributions.mortar_rate_row_sums, contributions.nonmortar_rate_row_sums,
                              true_rate_row_sums_ );
  native_stability_current_ = true;
}

void addNativeStabilityContributions( ::mfem::Vector& stiffness_row_sums, ::mfem::Vector& rate_row_sums ) const
  requires( tribol::detail::is_native_explicit_method_v<MethodType> && !execution::DevicePolicy<Execution> )
{
  if ( !native_stability_current_ ) {
    throw std::logic_error( "MFEM native stability contributions require a current contact evaluation." );
  }
  if ( stiffness_row_sums.Size() != coordinate_space_.GetTrueVSize() ||
       rate_row_sums.Size() != coordinate_space_.GetTrueVSize() ) {
    throw std::invalid_argument( "MFEM native stability contributions require coordinate true-DOF vectors." );
  }
  stiffness_row_sums += true_stiffness_row_sums_;
  rate_row_sums += true_rate_row_sums_;
}

[[nodiscard]] Real nativeStabilityVote() const
  requires( tribol::detail::is_native_explicit_method_v<MethodType> && !execution::DevicePolicy<Execution> )
{
  Real local_bounds[2]{};
  for ( int dof = 0; dof < true_stiffness_row_sums_.Size(); ++dof ) {
    local_bounds[0] = std::max( local_bounds[0], true_stiffness_row_sums_[dof] );
    local_bounds[1] = std::max( local_bounds[1], true_rate_row_sums_[dof] );
  }
  Real global_bounds[2]{};
  MPI_Allreduce( local_bounds, global_bounds, 2, detail::realMpiType(), MPI_MAX, mesh_.GetComm() );
  return timestep::explicit_stability_detail::stabilityVote( global_bounds[0], global_bounds[1], integrator_ );
}

[[nodiscard]] bool hasMatchingStabilityRegistration( const MfemContact& other ) const
  requires( tribol::detail::is_native_explicit_method_v<MethodType> && !execution::DevicePolicy<Execution> )
{
  const bool same_integrator =
      integrator_.maximum_undamped_frequency_step == other.integrator_.maximum_undamped_frequency_step &&
      integrator_.maximum_decay_step == other.integrator_.maximum_decay_step &&
      integrator_.safety_factor == other.integrator_.safety_factor;
  if ( !same_integrator || has_registered_inverse_mass_ != other.has_registered_inverse_mass_ ) {
    return false;
  }
  return !has_registered_inverse_mass_ ||
         ( registered_true_inverse_mass_.Size() == other.registered_true_inverse_mass_.Size() &&
           std::equal( registered_true_inverse_mass_.GetData(),
                       registered_true_inverse_mass_.GetData() + registered_true_inverse_mass_.Size(),
                       other.registered_true_inverse_mass_.GetData() ) );
}
