#include "tribol/adapters/mfem/MfemContact.hpp"

#include "mfem.hpp"
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>

// Requirements: ADAPTER-002, API-003, API-004, API-006, PAR-002

namespace {

using NativeExplicitMethod =
    tribol::Method<tribol::geometry::ProjectedOverlap<tribol::normal::MeanPlane>, tribol::integration::Automatic,
                   tribol::constraint::QuadraturePoint, tribol::enforcement::Penalty<>, tribol::response::Frictionless,
                   tribol::formulation::PointwiseTraction, tribol::linearization::Exact>;
using NativeContact = tribol::mfem::MfemContact<NativeExplicitMethod>;

bool close( double actual, double expected, double tolerance = 1.0e-12 )
{
  return std::abs( actual - expected ) <= tolerance * std::max( { 1.0, std::abs( actual ), std::abs( expected ) } );
}

bool throwsStaleDiagnostics( NativeContact& contact )
{
  try {
    static_cast<void>( contact.productionDiagnostics() );
  } catch ( const std::logic_error& ) {
    return true;
  }
  return false;
}

bool hasTrialInvalidations( tribol::ContactInvalidation invalidation )
{
  return tribol::contains( invalidation, tribol::ContactInvalidation::Coordinates ) &&
         tribol::contains( invalidation, tribol::ContactInvalidation::Mass ) &&
         tribol::contains( invalidation, tribol::ContactInvalidation::BoundaryConditions ) &&
         tribol::contains( invalidation, tribol::ContactInvalidation::Penalty ) &&
         tribol::contains( invalidation, tribol::ContactInvalidation::Integrator ) &&
         !tribol::contains( invalidation, tribol::ContactInvalidation::Topology ) &&
         !tribol::contains( invalidation, tribol::ContactInvalidation::Interactions );
}

void dilateNormalCoordinates( const ::mfem::ParGridFunction& source, ::mfem::ParGridFunction& destination )
{
  destination = source;
  auto& space = *destination.ParFESpace();
  for ( int scalar_dof = 0; scalar_dof < space.GetNDofs(); ++scalar_dof ) {
    const int normal_dof = space.DofToVDof( scalar_dof, 1 );
    destination[normal_dof] *= 1.5;
  }
}

bool attemptRecordsMatchAcrossRanks( const tribol::diagnostics::AttemptDiagnostics& attempts )
{
  const std::uint64_t local[]{ attempts.current.id,
                               attempts.current.retry,
                               static_cast<std::uint64_t>( attempts.current.status ),
                               static_cast<std::uint64_t>( attempts.current.failure ),
                               attempts.last_accepted.id,
                               attempts.last_rejected.id,
                               attempts.accepted_attempts,
                               attempts.rejected_attempts };
  std::uint64_t minimum[8]{};
  std::uint64_t maximum[8]{};
  MPI_Allreduce( local, minimum, 8, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD );
  MPI_Allreduce( local, maximum, 8, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD );
  return std::equal( std::begin( minimum ), std::end( minimum ), std::begin( maximum ) );
}

int exerciseAttemptLifecycle( int rank )
{
  auto serial_mesh = ::mfem::Mesh::MakeCartesian2D( 4, 1, ::mfem::Element::QUADRILATERAL, true, 1.0, 0.1 );
  ::mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mesh.SetCurvature( 1 );
  auto* coordinates = dynamic_cast<::mfem::ParGridFunction*>( mesh.GetNodes() );
  if ( coordinates == nullptr ) {
    return 1;
  }

  ::mfem::ParGridFunction trial_coordinates( coordinates->ParFESpace() );
  dilateNormalCoordinates( *coordinates, trial_coordinates );

  NativeContact::Options options;
  options.search.expansion = 0.25;
  options.method.enforcement.stiffness.value = 10.0;
  NativeContact contact( mesh, *coordinates, { { 1 }, { 3 } }, options );
  ::mfem::Vector inverse_mass( coordinates->ParFESpace()->GetTrueVSize() );
  inverse_mass = 2.0;
  contact.registerInverseMass( inverse_mass );
  contact.registerIntegrator( tribol::timestep::IntegratorStabilityLimits::centralDifference() );
  contact.updateInteractions();

  const auto initial_attempt = contact.beginAttempt();
  const auto accepted_result = contact.evaluate();
  contact.acceptAttempt();
  const auto accepted_diagnostics = contact.productionDiagnostics();
  const auto accepted_checkpoint = contact.checkpoint();
  bool valid = initial_attempt.id == 1 && initial_attempt.retry == 0 &&
               accepted_result.summary.active_interactions > 0 && accepted_result.summary.energy > 0.0 &&
               accepted_diagnostics.attempts.current.status == tribol::AttemptStatus::Accepted &&
               accepted_diagnostics.attempts.accepted_attempts == 1 &&
               accepted_diagnostics.attempts.rejected_attempts == 0;

  contact.updateGeometry( trial_coordinates );
  tribol::BoundaryConditionVersion boundary_conditions;
  tribol::PenaltyVersion penalty;
  boundary_conditions.advance();
  penalty.advance();
  contact.setBoundaryConditionGeneration( boundary_conditions );
  contact.setPenaltyGeneration( penalty );
  contact.registerInverseMass( inverse_mass );
  contact.registerIntegrator( tribol::timestep::IntegratorStabilityLimits::centralDifference() );
  valid = valid && throwsStaleDiagnostics( contact );

  const auto rejected_attempt = contact.beginAttempt();
  valid = valid && rejected_attempt.id == 2 && rejected_attempt.retry == 0 &&
          hasTrialInvalidations( contact.invalidationsSinceAcceptedAttempt() );
  const auto rejected_result = contact.evaluate();
  contact.rejectAttempt( tribol::FailureCategory::Stability );
  const auto rejected_diagnostics = contact.productionDiagnostics();
  valid = valid && rejected_result.summary.energy > accepted_result.summary.energy &&
          rejected_diagnostics.attempts.current.status == tribol::AttemptStatus::Rejected &&
          rejected_diagnostics.attempts.last_accepted.id == initial_attempt.id &&
          rejected_diagnostics.attempts.last_rejected.id == rejected_attempt.id &&
          rejected_diagnostics.attempts.last_rejected.failure == tribol::FailureCategory::Stability &&
          rejected_diagnostics.attempts.accepted_attempts == 1 && rejected_diagnostics.attempts.rejected_attempts == 1;

  contact.updateGeometry( *coordinates );
  valid = valid && throwsStaleDiagnostics( contact );
  const auto retry_attempt = contact.beginAttempt();
  const auto retry_result = contact.evaluate();
  contact.acceptAttempt();
  const auto retry_diagnostics = contact.productionDiagnostics();
  valid = valid && retry_attempt.id == 3 && retry_attempt.retry == 1 &&
          close( retry_result.summary.energy, accepted_result.summary.energy ) &&
          close( retry_diagnostics.global.rows.spring_energy, accepted_diagnostics.global.rows.spring_energy ) &&
          retry_diagnostics.attempts.accepted_attempts == 2 && retry_diagnostics.attempts.rejected_attempts == 1;

  contact.restore( accepted_checkpoint );
  valid = valid && contact.currentAttempt().id == initial_attempt.id && contact.acceptedAttempts() == 1 &&
          contact.rejectedAttempts() == 0 && hasTrialInvalidations( contact.invalidationsSinceAcceptedAttempt() );
  const auto restarted_attempt = contact.beginAttempt();
  static_cast<void>( contact.evaluate() );
  contact.acceptAttempt();
  const auto restarted_diagnostics = contact.productionDiagnostics();
  valid = valid && restarted_attempt.id == 2 && restarted_attempt.retry == 0 &&
          contact.invalidationsSinceAcceptedAttempt() == tribol::ContactInvalidation::None &&
          restarted_diagnostics.attempts.accepted_attempts == 2 &&
          restarted_diagnostics.attempts.rejected_attempts == 0 &&
          attemptRecordsMatchAcrossRanks( restarted_diagnostics.attempts );

  contact.rebuildGeometry( *coordinates );
  valid = valid && throwsStaleDiagnostics( contact ) &&
          tribol::contains( contact.invalidationsSinceAcceptedAttempt(), tribol::ContactInvalidation::Coordinates ) &&
          tribol::contains( contact.invalidationsSinceAcceptedAttempt(), tribol::ContactInvalidation::Topology );
  try {
    static_cast<void>( contact.evaluate() );
    valid = false;
  } catch ( const std::logic_error& ) {
  }
  contact.updateInteractions();
  valid = valid &&
          tribol::contains( contact.invalidationsSinceAcceptedAttempt(), tribol::ContactInvalidation::Interactions );

  int local_valid = valid ? 1 : 0;
  int global_valid{};
  MPI_Allreduce( &local_valid, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD );
  if ( global_valid == 0 && rank == 0 ) {
    std::cerr << "MFEM attempt lifecycle, rollback, restart, or invalidation contract failed.\n";
  }
  return global_valid == 1 ? 0 : 1;
}

}  // namespace

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  int rank{};
  int size{};
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  const int result = size == 1 || size == 2 || size == 4 ? exerciseAttemptLifecycle( rank ) : 1;
  if ( result != 0 && rank == 0 && size != 1 && size != 2 && size != 4 ) {
    std::cerr << "MFEM attempt lifecycle test requires one, two, or four MPI ranks.\n";
  }
  MPI_Finalize();
  return result;
}
