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
  registered_true_inverse_mass_.SetSize( parent_true_inverse_mass.Size() );
  registered_true_inverse_mass_ = parent_true_inverse_mass;
  has_registered_inverse_mass_ = true;
  native_stability_current_ = false;
  inverse_mass_version_.advance();
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
}
