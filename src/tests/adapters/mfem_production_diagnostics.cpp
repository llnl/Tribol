#include "tribol/adapters/mfem/MfemContactCollection.hpp"

#include "mfem.hpp"
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

// Requirements: API-006, OUTPUT-001, OUTPUT-003, PAR-002, PAR-005, PAR-007

namespace {

using NativeDampedPenalty =
    tribol::Method<tribol::geometry::ProjectedOverlap<tribol::normal::MeanPlane>, tribol::integration::Automatic,
                   tribol::constraint::QuadraturePoint,
                   tribol::enforcement::Penalty<tribol::stiffness::Constant, tribol::rate::Constant>,
                   tribol::response::Frictionless, tribol::formulation::PointwiseTraction,
                   tribol::linearization::Exact>;
using NativeContact = tribol::mfem::MfemContact<NativeDampedPenalty>;

bool close( double actual, double expected, double tolerance = 1.0e-11 )
{
  return std::abs( actual - expected ) <= tolerance * std::max( { 1.0, std::abs( actual ), std::abs( expected ) } );
}

void markContactSegments( ::mfem::Mesh& mesh )
{
  ::mfem::Array<int> vertices;
  for ( int boundary_element = 0; boundary_element < mesh.GetNBE(); ++boundary_element ) {
    const int attribute = mesh.GetBdrAttribute( boundary_element );
    if ( attribute != 1 && attribute != 3 ) {
      continue;
    }
    mesh.GetBdrElementVertices( boundary_element, vertices );
    double center{};
    for ( int vertex : vertices ) {
      center += mesh.GetVertex( vertex )[0];
    }
    center /= vertices.Size();
    if ( center > 0.25 && center < 0.5 ) {
      mesh.SetBdrAttribute( boundary_element, attribute == 1 ? 11 : 21 );
    } else if ( center > 0.5 && center < 0.75 ) {
      mesh.SetBdrAttribute( boundary_element, attribute == 1 ? 12 : 22 );
    }
  }
  mesh.SetAttributes();
}

NativeContact::Options options()
{
  NativeContact::Options result;
  result.search.expansion = 0.2;
  result.method.enforcement.stiffness.value = 10.0;
  result.method.enforcement.rate.value = 0.25;
  return result;
}

void initializeVelocity( const ::mfem::ParGridFunction& coordinates, ::mfem::ParGridFunction& velocity )
{
  auto& space = *coordinates.ParFESpace();
  velocity = 0.0;
  for ( int scalar_dof = 0; scalar_dof < space.GetNDofs(); ++scalar_dof ) {
    const int y_dof = space.DofToVDof( scalar_dof, 1 );
    velocity[y_dof] = coordinates[y_dof] < 0.05 ? -0.3 : 0.2;
  }
}

bool summariesMatch( const tribol::diagnostics::ProductionDiagnosticSummary& actual,
                     const tribol::diagnostics::ProductionDiagnosticSummary& expected )
{
  bool valid = actual.rows.rows == expected.rows.rows && actual.rows.owned_rows == expected.rows.owned_rows &&
               actual.rows.active_rows == expected.rows.active_rows &&
               actual.rows.nonfinite_rows == expected.rows.nonfinite_rows &&
               actual.quadrature.samples == expected.quadrature.samples &&
               actual.quadrature.owned_samples == expected.quadrature.owned_samples &&
               close( actual.rows.minimum_gap, expected.rows.minimum_gap ) &&
               close( actual.rows.maximum_penetration, expected.rows.maximum_penetration ) &&
               close( actual.rows.maximum_closing_rate, expected.rows.maximum_closing_rate ) &&
               close( actual.rows.maximum_traction, expected.rows.maximum_traction ) &&
               close( actual.rows.spring_energy, expected.rows.spring_energy ) &&
               close( actual.rows.dashpot_dissipation, expected.rows.dashpot_dissipation ) &&
               close( actual.rows.contact_work, expected.rows.contact_work ) &&
               close( actual.quadrature.maximum_penetration, expected.quadrature.maximum_penetration ) &&
               close( actual.quadrature.maximum_closing_rate, expected.quadrature.maximum_closing_rate );
  for ( int component = 0; component < 3; ++component ) {
    valid = valid && close( actual.rows.mortar_force[component], expected.rows.mortar_force[component] ) &&
            close( actual.rows.mortar_moment[component], expected.rows.mortar_moment[component] ) &&
            close( actual.rows.impulse[component], expected.rows.impulse[component] );
  }
  return valid;
}

tribol::diagnostics::ProductionDiagnosticSummary summed(
    const tribol::diagnostics::ProductionDiagnosticSummary& first,
    const tribol::diagnostics::ProductionDiagnosticSummary& second )
{
  tribol::diagnostics::ProductionDiagnosticSummary result;
  tribol::diagnostics::accumulate( result, first );
  tribol::diagnostics::accumulate( result, second );
  if ( result.rows.rows == 0 ) {
    result.rows.minimum_gap = 0.0;
  }
  return result;
}

bool allRanksAgree( double value )
{
  double minimum{};
  double maximum{};
  MPI_Allreduce( &value, &minimum, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
  MPI_Allreduce( &value, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  return close( minimum, maximum );
}

int exerciseDiagnostics( int rank )
{
  auto serial_mesh = ::mfem::Mesh::MakeCartesian2D( 4, 1, ::mfem::Element::QUADRILATERAL, true, 1.0, 0.1 );
  markContactSegments( serial_mesh );
  ::mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mesh.SetCurvature( 1 );
  auto* coordinates = dynamic_cast<::mfem::ParGridFunction*>( mesh.GetNodes() );
  if ( coordinates == nullptr ) {
    return 1;
  }

  ::mfem::ParGridFunction velocity( coordinates->ParFESpace() );
  initializeVelocity( *coordinates, velocity );
  const tribol::mfem::ContactState state{ .velocity = &velocity };
  NativeContact left( mesh, *coordinates, { { 11 }, { 21 } }, options() );
  NativeContact right( mesh, *coordinates, { { 12 }, { 22 } }, options() );
  tribol::mfem::MfemContactCollection collection( left );
  collection.add( right );
  tribol::mfem::MfemContactCollection reversed_collection( right );
  reversed_collection.add( left );

  ::mfem::Vector inverse_mass( coordinates->ParFESpace()->GetTrueVSize() );
  inverse_mass = 2.0;
  collection.registerInverseMass( inverse_mass );
  collection.registerIntegrator( tribol::timestep::IntegratorStabilityLimits::centralDifference() );
  collection.updateInteractions();
  static_cast<void>( left.beginAttempt() );
  static_cast<void>( right.beginAttempt() );
  const auto evaluation = collection.evaluate( state );
  left.acceptAttempt();
  right.acceptAttempt();

  constexpr double time_step = 0.25;
  const auto default_diagnostics = left.productionDiagnostics( state, time_step );
  const auto sampling = tribol::diagnostics::QuadratureSamplingOptions{ .exactness = 8, .include_normal_rates = true };
  const auto left_diagnostics = left.productionDiagnostics( state, time_step, sampling );
  const auto right_diagnostics = right.productionDiagnostics( state, time_step, sampling );
  const auto combined_diagnostics = collection.productionDiagnostics( state, time_step, sampling );
  const auto reversed_diagnostics = reversed_collection.productionDiagnostics( state, time_step, sampling );

  std::int64_t local_owned_counts[]{ static_cast<std::int64_t>( left_diagnostics.rank.owned.rows.rows ),
                                     static_cast<std::int64_t>( left_diagnostics.rank.owned.quadrature.samples ) };
  std::int64_t global_owned_counts[2]{};
  MPI_Allreduce( local_owned_counts, global_owned_counts, 2, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD );

  const auto expected_local = summed( left_diagnostics.rank.local, right_diagnostics.rank.local );
  const auto expected_owned = summed( left_diagnostics.rank.owned, right_diagnostics.rank.owned );
  const auto expected_global = summed( left_diagnostics.global, right_diagnostics.global );
  bool valid = evaluation.active_interactions > 0 && tribol::diagnostics::isStructurallyValid( left_diagnostics ) &&
               tribol::diagnostics::isStructurallyValid( combined_diagnostics ) &&
               left_diagnostics.global.rows.rows == 2 && left_diagnostics.global.quadrature.samples == 5 &&
               close( left_diagnostics.global.rows.spring_energy, 0.00625 ) &&
               close( left_diagnostics.global.quadrature.maximum_penetration, 0.1 ) &&
               close( left_diagnostics.global.quadrature.maximum_closing_rate, 0.5 ) &&
               combined_diagnostics.global.rows.rows == 4 && combined_diagnostics.global.quadrature.samples == 10 &&
               default_diagnostics.rank.local.quadrature.maximum_closing_rate == 0.0 &&
               left_diagnostics.global.quadrature.maximum_closing_rate > 0.0 &&
               left_diagnostics.rank.owned.rows.rows <= left_diagnostics.rank.local.rows.rows &&
               left_diagnostics.rank.owned.quadrature.samples <= left_diagnostics.rank.local.quadrature.samples &&
               left_diagnostics.global.rows.rows == global_owned_counts[0] &&
               left_diagnostics.global.quadrature.samples == global_owned_counts[1] &&
               summariesMatch( combined_diagnostics.rank.local, expected_local ) &&
               summariesMatch( combined_diagnostics.rank.owned, expected_owned ) &&
               summariesMatch( combined_diagnostics.global, expected_global ) &&
               summariesMatch( reversed_diagnostics.rank.local, combined_diagnostics.rank.local ) &&
               summariesMatch( reversed_diagnostics.rank.owned, combined_diagnostics.rank.owned ) &&
               summariesMatch( reversed_diagnostics.global, combined_diagnostics.global ) &&
               left_diagnostics.attempts.accepted_attempts == 1 &&
               left_diagnostics.attempts.last_accepted.status == tribol::AttemptStatus::Accepted &&
               combined_diagnostics.attempts.accepted_attempts == 0 &&
               allRanksAgree( combined_diagnostics.global.rows.spring_energy ) &&
               allRanksAgree( combined_diagnostics.global.quadrature.maximum_penetration );

  NativeContact empty( mesh, *coordinates, { { 11 }, { 22 } }, options() );
  empty.registerInverseMass( inverse_mass );
  empty.registerIntegrator( tribol::timestep::IntegratorStabilityLimits::centralDifference() );
  empty.updateInteractions();
  static_cast<void>( empty.evaluate( state ) );
  const auto empty_diagnostics = empty.productionDiagnostics( state, time_step, sampling );
  valid = valid && empty_diagnostics.global.rows.rows == 0 && empty_diagnostics.global.rows.minimum_gap == 0.0 &&
          empty_diagnostics.global.quadrature.samples == 0 &&
          empty_diagnostics.global.quadrature.maximum_penetration == 0.0 &&
          empty_diagnostics.global.quadrature.maximum_closing_rate == 0.0;

  if ( !valid && rank == 0 ) {
    std::cerr << "MFEM production diagnostics mismatch: left rows=" << left_diagnostics.global.rows.rows
              << " owned rows=" << global_owned_counts[0] << " samples=" << left_diagnostics.global.quadrature.samples
              << " owned samples=" << global_owned_counts[1]
              << " penetration=" << left_diagnostics.global.quadrature.maximum_penetration
              << " closing=" << left_diagnostics.global.quadrature.maximum_closing_rate
              << " energy=" << left_diagnostics.global.rows.spring_energy
              << " combined rows=" << combined_diagnostics.global.rows.rows
              << " empty rows=" << empty_diagnostics.global.rows.rows << '\n';
  }
  return valid ? 0 : 1;
}

}  // namespace

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  int size{};
  int rank{};
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  const int result = size == 1 || size == 2 || size == 4 ? exerciseDiagnostics( rank ) : 1;
  if ( result != 0 && rank == 0 && size != 1 && size != 2 && size != 4 ) {
    std::cerr << "MFEM production diagnostics test requires one, two, or four MPI ranks.\n";
  }
  MPI_Finalize();
  return result;
}
