[[nodiscard]] GeometryVersion geometryVersion() const { return geometry_version_; }

[[nodiscard]] GeometryVersion interactionGeometryVersion() const { return interaction_geometry_version_; }

[[nodiscard]] InteractionVersion interactionVersion() const { return interaction_version_; }

[[nodiscard]] TopologyVersion topologyVersion() const { return topology_version_; }

[[nodiscard]] ContactGenerations generations() const
{
  return {
      .coordinates = geometry_version_,
      .topology = topology_version_,
      .interactions = interaction_version_,
      .mass = mass_version_,
      .boundary_conditions = boundary_condition_version_,
      .penalty = penalty_version_,
      .integrator = integrator_version_,
  };
}

void setMassGeneration( MassVersion generation )
{
  if ( mass_version_ != generation ) {
    mass_version_ = generation;
    native_stability_current_ = false;
  }
}

void setBoundaryConditionGeneration( BoundaryConditionVersion generation )
{
  if ( boundary_condition_version_ != generation ) {
    boundary_condition_version_ = generation;
    native_stability_current_ = false;
  }
}

void setPenaltyGeneration( PenaltyVersion generation )
{
  if ( penalty_version_ != generation ) {
    penalty_version_ = generation;
    native_stability_current_ = false;
  }
}

void setIntegratorGeneration( IntegratorVersion generation )
{
  if ( integrator_version_ != generation ) {
    integrator_version_ = generation;
    native_stability_current_ = false;
  }
}

[[nodiscard]] ContactAttemptRecord beginAttempt() { return attempt_lifecycle_.begin( generations() ); }

void acceptAttempt() { attempt_lifecycle_.accept(); }

void rejectAttempt( FailureCategory failure ) { attempt_lifecycle_.reject( failure ); }

[[nodiscard]] ContactAttemptCheckpoint checkpoint() const { return attempt_lifecycle_.checkpoint(); }

void restore( ContactAttemptCheckpoint checkpoint ) { attempt_lifecycle_.restore( checkpoint ); }

[[nodiscard]] ContactInvalidation invalidationsSinceAcceptedAttempt() const
{
  return invalidationsSince( attempt_lifecycle_.lastAccepted().generations, generations() );
}

[[nodiscard]] ContactAttemptRecord currentAttempt() const { return attempt_lifecycle_.current(); }

[[nodiscard]] ContactAttemptRecord lastAcceptedAttempt() const { return attempt_lifecycle_.lastAccepted(); }

[[nodiscard]] ContactAttemptRecord lastRejectedAttempt() const { return attempt_lifecycle_.lastRejected(); }

[[nodiscard]] std::uint64_t acceptedAttempts() const { return attempt_lifecycle_.acceptedAttempts(); }

[[nodiscard]] std::uint64_t rejectedAttempts() const { return attempt_lifecycle_.rejectedAttempts(); }

[[nodiscard]] const ContactAttemptLifecycle& attemptLifecycle() const { return attempt_lifecycle_; }
