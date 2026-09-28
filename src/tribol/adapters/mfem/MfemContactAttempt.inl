[[nodiscard]] ContactGenerations generations() const { return core_.generations(); }

void setBoundaryConditionGeneration( BoundaryConditionVersion generation )
{
  core_.setBoundaryConditionGeneration( generation );
}

void setPenaltyGeneration( PenaltyVersion generation ) { core_.setPenaltyGeneration( generation ); }

[[nodiscard]] ContactAttemptRecord beginAttempt() { return core_.beginAttempt(); }

void acceptAttempt() { core_.acceptAttempt(); }

void rejectAttempt( FailureCategory failure ) { core_.rejectAttempt( failure ); }

[[nodiscard]] ContactAttemptCheckpoint checkpoint() const { return core_.checkpoint(); }

void restore( ContactAttemptCheckpoint checkpoint ) { core_.restore( checkpoint ); }

[[nodiscard]] ContactInvalidation invalidationsSinceAcceptedAttempt() const
{
  return core_.invalidationsSinceAcceptedAttempt();
}

[[nodiscard]] ContactAttemptRecord currentAttempt() const { return core_.currentAttempt(); }

[[nodiscard]] ContactAttemptRecord lastAcceptedAttempt() const { return core_.lastAcceptedAttempt(); }

[[nodiscard]] ContactAttemptRecord lastRejectedAttempt() const { return core_.lastRejectedAttempt(); }

[[nodiscard]] std::uint64_t acceptedAttempts() const { return core_.acceptedAttempts(); }

[[nodiscard]] std::uint64_t rejectedAttempts() const { return core_.rejectedAttempts(); }

[[nodiscard]] const ContactAttemptLifecycle& attemptLifecycle() const { return core_.attemptLifecycle(); }
