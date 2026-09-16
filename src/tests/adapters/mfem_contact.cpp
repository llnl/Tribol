#include "tribol/adapters/mfem/MfemContact.hpp"

#include "mfem.hpp"
#include <mpi.h>

#include <cmath>
#include <iostream>
#include <vector>

// Requirements: ADAPTER-002, ADAPTER-003, API-001, GEOM-001, PAR-002, PHYS-001

namespace {

void curveCoordinates( ::mfem::ParGridFunction& coordinates )
{
  auto& space = *coordinates.ParFESpace();
  for ( int scalar_dof = 0; scalar_dof < space.GetNDofs(); ++scalar_dof ) {
    const int x_dof = space.DofToVDof( scalar_dof, 0 );
    const int y_dof = space.DofToVDof( scalar_dof, 1 );
    const double x = coordinates[x_dof];
    coordinates[y_dof] += 0.02 * x * ( 1.0 - x );
  }
}

double globalSum( double local )
{
  double global{};
  MPI_Allreduce( &local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  return global;
}

int exerciseContact( int communicator_size, int rank, int order )
{
  ::mfem::Mesh serial_mesh =
      ::mfem::Mesh::MakeCartesian2D( 1, communicator_size, ::mfem::Element::QUADRILATERAL, true, 1.0, 0.1 );
  ::mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mesh.SetCurvature( order );
  auto* coordinates = dynamic_cast<::mfem::ParGridFunction*>( mesh.GetNodes() );
  if ( coordinates == nullptr ) {
    return 1;
  }
  if ( order > 1 ) {
    curveCoordinates( *coordinates );
  }

  tribol::mfem::MfemContact<>::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.value = 10.0;
  auto contact =
      tribol::mfem::makeContact( mesh, *coordinates, tribol::mfem::PairedBoundaryAttributes{ { 3 }, { 1 } }, options );
  contact.updateInteractions();

  ::mfem::Vector residual( coordinates->ParFESpace()->GetTrueVSize() );
  residual = 0.0;
  const auto summary = contact.addResidual( residual );
  double local_residual_sum{};
  for ( int dof = 0; dof < residual.Size(); ++dof ) {
    local_residual_sum += residual[dof];
  }
  const double residual_sum = globalSum( local_residual_sum );

  ::mfem::Vector primal( coordinates->ParFESpace()->GetTrueVSize() );
  for ( int dof = 0; dof < primal.Size(); ++dof ) {
    primal[dof] = 0.125 + rank + 0.25 * dof;
  }
  std::vector<tribol::Real> restricted_mortar;
  std::vector<tribol::Real> restricted_nonmortar;
  contact.restrictPrimal( primal, restricted_mortar, restricted_nonmortar );
  std::vector<tribol::Real> dual_mortar( restricted_mortar.size() );
  std::vector<tribol::Real> dual_nonmortar( restricted_nonmortar.size() );
  for ( std::size_t value = 0; value < dual_mortar.size(); ++value ) {
    dual_mortar[value] = 0.25 + static_cast<double>( value );
  }
  for ( std::size_t value = 0; value < dual_nonmortar.size(); ++value ) {
    dual_nonmortar[value] = -0.5 - static_cast<double>( value );
  }
  double restricted_dot{};
  for ( std::size_t value = 0; value < dual_mortar.size(); ++value ) {
    restricted_dot += restricted_mortar[value] * dual_mortar[value];
  }
  for ( std::size_t value = 0; value < dual_nonmortar.size(); ++value ) {
    restricted_dot += restricted_nonmortar[value] * dual_nonmortar[value];
  }
  ::mfem::Vector transpose( coordinates->ParFESpace()->GetTrueVSize() );
  transpose = 0.0;
  contact.addDualTranspose( { dual_mortar.data(), static_cast<tribol::Index>( dual_mortar.size() ) },
                            { dual_nonmortar.data(), static_cast<tribol::Index>( dual_nonmortar.size() ) }, transpose );
  double transpose_dot{};
  for ( int dof = 0; dof < transpose.Size(); ++dof ) {
    transpose_dot += primal[dof] * transpose[dof];
  }
  const double global_transpose_dot = globalSum( transpose_dot );
  const double global_restricted_dot = globalSum( rank == 0 ? restricted_dot : 0.0 );

  const auto surfaces = contact.core().surfaces();
  bool curved_sample_found = order == 1;
  for ( tribol::Index node = 0; node < surfaces.mortar.numberOfNodes(); ++node ) {
    curved_sample_found = curved_sample_found || surfaces.mortar.coordinates( node, 1 ) > 0.1001;
  }
  const bool expected_refinement =
      surfaces.mortar.numberOfElements() == order && surfaces.nonmortar.numberOfElements() == order &&
      surfaces.mortar.numberOfNodes() == 2 * order && surfaces.nonmortar.numberOfNodes() == 2 * order;
  const bool energy_is_correct = order == 1 ? std::abs( summary.energy - 0.05 ) <= 1.0e-12
                                            : std::isfinite( summary.energy ) && summary.energy > 0.0;
  const bool valid = !contact.core().interactions().empty() && expected_refinement && curved_sample_found &&
                     energy_is_correct && std::abs( residual_sum ) <= 1.0e-12 &&
                     std::abs( global_restricted_dot - global_transpose_dot ) <= 1.0e-11;
  if ( !valid && rank == 0 ) {
    std::cerr << "order=" << order << " interactions=" << contact.core().interactions().size()
              << " mortar_elements=" << surfaces.mortar.numberOfElements() << " energy=" << summary.energy
              << " residual_sum=" << residual_sum << " adjoint_error=" << global_restricted_dot - global_transpose_dot
              << " curved_sample=" << curved_sample_found << '\n';
  }
  return valid ? 0 : 1;
}

}  // namespace

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  int communicator_size{};
  int rank{};
  MPI_Comm_size( MPI_COMM_WORLD, &communicator_size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  const int result = exerciseContact( communicator_size, rank, 1 ) | exerciseContact( communicator_size, rank, 2 );
  MPI_Finalize();
  return result;
}
