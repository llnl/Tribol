#include "tribol/core/AttemptLifecycle.hpp"

#include <stdexcept>

// Requirements: API-006

namespace {

using namespace tribol;

bool generationInvalidationContract()
{
  ContactGenerations accepted;
  ContactGenerations changed = accepted;
  changed.mass.advance();
  changed.boundary_conditions.advance();
  auto invalidation = invalidationsSince( accepted, changed );
  if ( invalidatesRows( invalidation ) || !invalidatesStability( invalidation ) ||
       !contains( invalidation, ContactInvalidation::Mass ) ||
       !contains( invalidation, ContactInvalidation::BoundaryConditions ) ) {
    return false;
  }
  changed.coordinates.advance();
  invalidation = invalidationsSince( accepted, changed );
  return invalidatesRows( invalidation ) && invalidatesStability( invalidation ) &&
         contains( invalidation, ContactInvalidation::Coordinates );
}

bool retryAndRestartContract()
{
  ContactAttemptLifecycle lifecycle;
  ContactGenerations first_generations;
  const auto first = lifecycle.begin( first_generations );
  lifecycle.reject( FailureCategory::Geometry );
  if ( first.id != 1 || first.retry != 0 || lifecycle.lastRejected().failure != FailureCategory::Geometry ) {
    return false;
  }

  first_generations.coordinates.advance();
  const auto retry = lifecycle.begin( first_generations );
  lifecycle.accept();
  if ( retry.id != 2 || retry.retry != 1 || lifecycle.lastAccepted().generations != first_generations ||
       lifecycle.acceptedAttempts() != 1 || lifecycle.rejectedAttempts() != 1 ) {
    return false;
  }
  const auto checkpoint = lifecycle.checkpoint();

  ContactGenerations rejected_generations = first_generations;
  rejected_generations.mass.advance();
  static_cast<void>( lifecycle.begin( rejected_generations ) );
  lifecycle.reject( FailureCategory::Stability );
  lifecycle.restore( checkpoint );
  const auto resumed = lifecycle.begin( first_generations );
  return resumed.id == 3 && resumed.retry == 0 && lifecycle.lastAccepted().id == 2 &&
         lifecycle.lastRejected().status == AttemptStatus::None && lifecycle.acceptedAttempts() == 1 &&
         lifecycle.rejectedAttempts() == 1;
}

bool invalidTransitionContract()
{
  ContactAttemptLifecycle lifecycle;
  try {
    lifecycle.accept();
    return false;
  } catch ( const std::logic_error& ) {
  }
  static_cast<void>( lifecycle.begin( {} ) );
  try {
    lifecycle.reject( FailureCategory::None );
    return false;
  } catch ( const std::invalid_argument& ) {
  }
  try {
    static_cast<void>( lifecycle.checkpoint() );
    return false;
  } catch ( const std::logic_error& ) {
  }
  lifecycle.reject( FailureCategory::HostApplication );
  return true;
}

}  // namespace

int main()
{
  return generationInvalidationContract() && retryAndRestartContract() && invalidTransitionContract() ? 0 : 1;
}
