void registerInverseMass( const ::mfem::Vector& parent_true_inverse_mass )
  requires tribol::detail::is_native_explicit_method_v<MethodType>
{
  if ( parent_true_inverse_mass.Size() != coordinate_space_.GetTrueVSize() ) {
    throw std::invalid_argument( "MFEM inverse mass must use the coordinate true-DOF space." );
  }
  for ( int dof = 0; dof < parent_true_inverse_mass.Size(); ++dof ) {
    if ( !std::isfinite( parent_true_inverse_mass[dof] ) || parent_true_inverse_mass[dof] < 0.0 ) {
      throw std::invalid_argument( "MFEM inverse-mass entries must be finite and nonnegative." );
    }
  }
  physical_true_inverse_mass_.SetSize( parent_true_inverse_mass.Size() );
  physical_true_inverse_mass_ = parent_true_inverse_mass;
  has_registered_inverse_mass_ = true;
  applyEssentialBoundaryConditions();
  native_stability_current_ = false;
  inverse_mass_version_.advance();
  core_.setMassGeneration( inverse_mass_version_ );
  refreshInverseMass();
}

void registerInverseMass( const ::mfem::ParGridFunction& inverse_mass )
  requires tribol::detail::is_native_explicit_method_v<MethodType>
{
  requireSameSpace( inverse_mass );
  ::mfem::Vector true_inverse_mass;
  inverse_mass.GetTrueDofs( true_inverse_mass );
  registerInverseMass( true_inverse_mass );
}

void registerEssentialTrueDofs( const ::mfem::Array<int>& essential_true_dofs )
  requires tribol::detail::is_native_explicit_method_v<MethodType>
{
  std::vector<int> canonical;
  canonical.reserve( static_cast<std::size_t>( essential_true_dofs.Size() ) );
  for ( int index = 0; index < essential_true_dofs.Size(); ++index ) {
    const int true_dof = essential_true_dofs[index];
    if ( true_dof < 0 || true_dof >= coordinate_space_.GetTrueVSize() ) {
      throw std::out_of_range( "MFEM essential true DOF is outside the coordinate true-DOF space." );
    }
    canonical.push_back( true_dof );
  }
  std::sort( canonical.begin(), canonical.end() );
  canonical.erase( std::unique( canonical.begin(), canonical.end() ), canonical.end() );
  if ( canonical == essential_true_dofs_ ) {
    return;
  }
  essential_true_dofs_ = std::move( canonical );
  auto generation = core_.generations().boundary_conditions;
  generation.advance();
  core_.setBoundaryConditionGeneration( generation );
  applyEssentialBoundaryConditions();
  refreshInverseMass();
  native_stability_current_ = false;
}

void registerIntegrator( timestep::IntegratorStabilityLimits integrator )
  requires tribol::detail::is_native_explicit_method_v<MethodType>
{
  if ( !std::isfinite( integrator.maximum_undamped_frequency_step ) ||
       integrator.maximum_undamped_frequency_step <= 0.0 || !std::isfinite( integrator.maximum_decay_step ) ||
       integrator.maximum_decay_step <= 0.0 || !std::isfinite( integrator.safety_factor ) ||
       integrator.safety_factor <= 0.0 || integrator.safety_factor > 1.0 ) {
    throw std::invalid_argument( "MFEM explicit integrator limits must be positive and finite with safety in (0,1]." );
  }
  integrator_version_.advance();
  integrator.version = integrator_version_;
  integrator_ = integrator;
  native_stability_current_ = false;
  core_.setIntegratorGeneration( integrator_version_ );
}
