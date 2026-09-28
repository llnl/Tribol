#include "tribol/adapters/mfem/MfemContactCollection.hpp"

#include "mfem.hpp"
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>

// Requirements: PAR-002, PAR-005, PAR-007, TIME-002

namespace {

using NativeExplicitMethod =
    tribol::Method<tribol::geometry::ProjectedOverlap<tribol::normal::MeanPlane>, tribol::integration::Automatic,
                   tribol::constraint::QuadraturePoint, tribol::enforcement::Penalty<>, tribol::response::Frictionless,
                   tribol::formulation::PointwiseTraction, tribol::linearization::Exact>;
using NativeContact = tribol::mfem::MfemContact<NativeExplicitMethod>;

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

int exerciseSharedTrueDofStability( int rank )
{
  auto serial_mesh = ::mfem::Mesh::MakeCartesian2D( 4, 1, ::mfem::Element::QUADRILATERAL, true, 1.0, 0.1 );
  markContactSegments( serial_mesh );
  ::mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mesh.SetCurvature( 1 );
  auto* coordinates = dynamic_cast<::mfem::ParGridFunction*>( mesh.GetNodes() );
  if ( coordinates == nullptr ) {
    return 1;
  }

  NativeContact::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.value = 10.0;
  NativeContact left( mesh, *coordinates, { { 11 }, { 21 } }, options );
  NativeContact right( mesh, *coordinates, { { 12 }, { 22 } }, options );
  tribol::mfem::MfemContactCollection collection( left );
  collection.add( right );

  ::mfem::Vector inverse_mass( coordinates->ParFESpace()->GetTrueVSize() );
  inverse_mass = 2.0;
  collection.registerInverseMass( inverse_mass );
  collection.registerIntegrator( tribol::timestep::IntegratorStabilityLimits::centralDifference() );
  collection.updateInteractions();

  ::mfem::Vector left_residual( inverse_mass.Size() );
  ::mfem::Vector right_residual( inverse_mass.Size() );
  ::mfem::Vector combined_residual( inverse_mass.Size() );
  left_residual = 0.0;
  right_residual = 0.0;
  combined_residual = 0.0;
  const auto left_summary = left.addResidual( left_residual );
  const auto right_summary = right.addResidual( right_residual );
  const auto combined_summary = collection.addResidual( combined_residual );

  combined_residual -= left_residual;
  combined_residual -= right_residual;
  double local_residual_error = combined_residual * combined_residual;
  double residual_error{};
  MPI_Allreduce( &local_residual_error, &residual_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  residual_error = std::sqrt( residual_error );

  const double expected_vote = left_summary.timestep_vote / std::sqrt( 2.0 );
  const auto evaluated_summary = collection.evaluate();
  const bool valid =
      collection.size() == 2 && left_summary.active_interactions > 0 &&
      right_summary.active_interactions == left_summary.active_interactions &&
      close( right_summary.timestep_vote, left_summary.timestep_vote ) &&
      close( combined_summary.timestep_vote, expected_vote ) &&
      combined_summary.timestep_vote < left_summary.timestep_vote &&
      close( combined_summary.energy, left_summary.energy + right_summary.energy ) &&
      combined_summary.active_interactions == left_summary.active_interactions + right_summary.active_interactions &&
      combined_summary.quadrature_points == left_summary.quadrature_points + right_summary.quadrature_points &&
      close( evaluated_summary.timestep_vote, combined_summary.timestep_vote ) && residual_error < 1.0e-12;
  if ( !valid && rank == 0 ) {
    std::cerr << "MFEM global native stability mismatch: left_dt=" << left_summary.timestep_vote
              << " right_dt=" << right_summary.timestep_vote << " combined_dt=" << combined_summary.timestep_vote
              << " expected_dt=" << expected_vote << " energy=" << combined_summary.energy
              << " residual_error=" << residual_error << '\n';
  }
  return valid ? 0 : 1;
}

}  // namespace

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  int rank{};
  int size{};
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  const int result = size == 1 || size == 2 || size == 4 ? exerciseSharedTrueDofStability( rank ) : 1;
  if ( result != 0 && rank == 0 && size != 1 && size != 2 && size != 4 ) {
    std::cerr << "MFEM native stability test requires one, two, or four MPI ranks.\n";
  }
  MPI_Finalize();
  return result;
}
