#include "tribol/adapters/array/ArrayContact.hpp"
#include "tribol/adapters/mfem/MfemContact.hpp"

#include "mfem.hpp"
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <utility>

// Requirements: ADAPTER-001, ADAPTER-002, CONS-002, GEOM-004, PAR-002, PHYS-003, PHYS-004

namespace {

using NativePenalty =
    tribol::Method<tribol::geometry::ProjectedOverlap<tribol::normal::MeanPlane>, tribol::integration::Automatic,
                   tribol::constraint::QuadraturePoint, tribol::enforcement::Penalty<>, tribol::response::Frictionless,
                   tribol::formulation::PointwiseTraction, tribol::linearization::Exact>;
using NativeContact = tribol::mfem::MfemContact<NativePenalty>;

struct RunResult {
  tribol::EvaluationSummary evaluation{};
  tribol::diagnostics::ProductionDiagnosticsPayload diagnostics{};
  ::mfem::Vector residual{};
};

struct AdapterReference {
  tribol::EvaluationSummary evaluation{};
  tribol::diagnostics::ProductionDiagnosticSummary diagnostics{};
};

bool close( double actual, double expected, double tolerance = 2.0e-11 )
{
  return std::abs( actual - expected ) <= tolerance * std::max( { 1.0, std::abs( actual ), std::abs( expected ) } );
}

double globalSum( double local )
{
  double global{};
  MPI_Allreduce( &local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  return global;
}

void markHorizontalSurfaces( ::mfem::Mesh& mesh )
{
  ::mfem::Array<int> vertices;
  const int normal_component = mesh.Dimension() - 1;
  for ( int boundary_element = 0; boundary_element < mesh.GetNBE(); ++boundary_element ) {
    mesh.GetBdrElementVertices( boundary_element, vertices );
    double center{};
    for ( int vertex : vertices ) {
      center += mesh.GetVertex( vertex )[normal_component];
    }
    center /= vertices.Size();
    if ( std::abs( center ) < 1.0e-12 ) {
      mesh.SetBdrAttribute( boundary_element, 11 );
    } else if ( std::abs( center - 0.1 ) < 1.0e-12 ) {
      mesh.SetBdrAttribute( boundary_element, 21 );
    }
  }
  mesh.SetAttributes();
}

void curveCoordinates( ::mfem::ParGridFunction& coordinates )
{
  auto& space = *coordinates.ParFESpace();
  for ( int scalar_dof = 0; scalar_dof < space.GetNDofs(); ++scalar_dof ) {
    const int x_dof = space.DofToVDof( scalar_dof, 0 );
    const int y_dof = space.DofToVDof( scalar_dof, 1 );
    const int z_dof = space.DofToVDof( scalar_dof, 2 );
    const double x = coordinates[x_dof];
    const double y = coordinates[y_dof];
    coordinates[z_dof] += 0.2 * x * ( 1.0 - x ) * y * ( 1.0 - y );
  }
}

NativeContact::Options options( int minimum_exactness )
{
  NativeContact::Options result;
  result.search.expansion = 0.2;
  result.method.integration.minimum_exactness = minimum_exactness;
  result.method.enforcement.stiffness.value = 10.0;
  return result;
}

AdapterReference evaluateArrayReference( tribol::array::SingleTopologySurface mortar,
                                         tribol::array::SingleTopologySurface nonmortar )
{
  auto contact = tribol::array::makeContact<NativePenalty>( std::move( mortar ), std::move( nonmortar ), options( 1 ) );
  contact.updateInteractions();
  const auto surfaces = contact.core().surfaces();
  std::vector<tribol::Real> mortar_inverse_mass(
      static_cast<std::size_t>( surfaces.mortar.numberOfNodes() * surfaces.mortar.dimension ), 1.0 );
  std::vector<tribol::Real> nonmortar_inverse_mass(
      static_cast<std::size_t>( surfaces.nonmortar.numberOfNodes() * surfaces.nonmortar.dimension ), 1.0 );
  const tribol::timestep::DiagonalInverseMassView inverse_mass{
      .mortar = { { mortar_inverse_mass.data(), static_cast<tribol::Index>( mortar_inverse_mass.size() ) },
                  surfaces.mortar.numberOfNodes(),
                  surfaces.mortar.dimension,
                  tribol::FieldLayout::Interleaved },
      .nonmortar = { { nonmortar_inverse_mass.data(), static_cast<tribol::Index>( nonmortar_inverse_mass.size() ) },
                     surfaces.nonmortar.numberOfNodes(),
                     surfaces.nonmortar.dimension,
                     tribol::FieldLayout::Interleaved },
  };
  const auto evaluation =
      contact.evaluate( {}, inverse_mass, tribol::timestep::IntegratorStabilityLimits::centralDifference() ).summary;
  const auto diagnostics = contact.core().productionDiagnostics(
      {}, 0.25, 0, 1, tribol::diagnostics::QuadratureSamplingOptions{ .exactness = 8 } );
  return { evaluation, diagnostics.local };
}

AdapterReference arrayReference( int dimension )
{
  if ( dimension == 2 ) {
    const std::array<tribol::Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
    const std::array<tribol::Real, 4> nonmortar_coordinates{ 0.0, 0.1, 1.0, 0.1 };
    const std::array<tribol::Index, 2> mortar_connectivity{ 0, 1 };
    const std::array<tribol::Index, 2> nonmortar_connectivity{ 1, 0 };
    return evaluateArrayReference(
        tribol::array::SingleTopologySurface(
            2, tribol::ElementTopology::Segment,
            { mortar_coordinates.data(), static_cast<tribol::Index>( mortar_coordinates.size() ) },
            { mortar_connectivity.data(), static_cast<tribol::Index>( mortar_connectivity.size() ) } ),
        tribol::array::SingleTopologySurface(
            2, tribol::ElementTopology::Segment,
            { nonmortar_coordinates.data(), static_cast<tribol::Index>( nonmortar_coordinates.size() ) },
            { nonmortar_connectivity.data(), static_cast<tribol::Index>( nonmortar_connectivity.size() ) } ) );
  }
  const std::array<tribol::Real, 12> mortar_coordinates{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
  const std::array<tribol::Real, 12> nonmortar_coordinates{ 0.0, 0.0, 0.1, 1.0, 0.0, 0.1,
                                                            1.0, 1.0, 0.1, 0.0, 1.0, 0.1 };
  const std::array<tribol::Index, 4> mortar_connectivity{ 0, 3, 2, 1 };
  const std::array<tribol::Index, 4> nonmortar_connectivity{ 0, 1, 2, 3 };
  return evaluateArrayReference(
      tribol::array::SingleTopologySurface(
          3, tribol::ElementTopology::Quadrilateral,
          { mortar_coordinates.data(), static_cast<tribol::Index>( mortar_coordinates.size() ) },
          { mortar_connectivity.data(), static_cast<tribol::Index>( mortar_connectivity.size() ) } ),
      tribol::array::SingleTopologySurface(
          3, tribol::ElementTopology::Quadrilateral,
          { nonmortar_coordinates.data(), static_cast<tribol::Index>( nonmortar_coordinates.size() ) },
          { nonmortar_connectivity.data(), static_cast<tribol::Index>( nonmortar_connectivity.size() ) } ) );
}

RunResult runContact( const ::mfem::ParMesh& mesh, const ::mfem::ParGridFunction& coordinates, int minimum_exactness,
                      int subdivision_factor )
{
  NativeContact contact( mesh, coordinates, { { 11 }, { 21 } }, options( minimum_exactness ),
                         { .subdivision_factor = subdivision_factor } );
  ::mfem::Vector inverse_mass( coordinates.ParFESpace()->GetTrueVSize() );
  inverse_mass = 1.0;
  contact.registerInverseMass( inverse_mass );
  contact.registerIntegrator( tribol::timestep::IntegratorStabilityLimits::centralDifference() );
  contact.updateInteractions();
  ::mfem::Vector residual( inverse_mass.Size() );
  residual = 0.0;
  const auto evaluation = contact.addResidual( residual );
  const auto diagnostics =
      contact.productionDiagnostics( {}, 0.25, tribol::diagnostics::QuadratureSamplingOptions{ .exactness = 8 } );
  return { evaluation, diagnostics, std::move( residual ) };
}

double residualDifference( const ::mfem::Vector& left, const ::mfem::Vector& right )
{
  ::mfem::Vector difference( left );
  difference -= right;
  return std::sqrt( globalSum( difference * difference ) );
}

std::array<double, 3> globalResidualSum( const ::mfem::ParFiniteElementSpace& space, const ::mfem::Vector& residual )
{
  const int dimension = space.GetVDim();
  const int scalar_dofs = space.GetTrueVSize() / space.GetVDim();
  std::array<double, 3> result{};
  for ( int component = 0; component < dimension; ++component ) {
    double local{};
    for ( int scalar_dof = 0; scalar_dof < scalar_dofs; ++scalar_dof ) {
      const int vector_dof = space.GetOrdering() == ::mfem::Ordering::byNODES ? component * scalar_dofs + scalar_dof
                                                                              : dimension * scalar_dof + component;
      local += residual[vector_dof];
    }
    result[static_cast<std::size_t>( component )] = globalSum( local );
  }
  return result;
}

bool planarOrderContract( int dimension, int order, int rank )
{
  auto serial_mesh = dimension == 2
                         ? ::mfem::Mesh::MakeCartesian2D( 4, 1, ::mfem::Element::QUADRILATERAL, true, 1.0, 0.1 )
                         : ::mfem::Mesh::MakeCartesian3D( 4, 1, 1, ::mfem::Element::HEXAHEDRON, 1.0, 1.0, 0.1 );
  markHorizontalSurfaces( serial_mesh );
  ::mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mesh.SetCurvature( order );
  auto* coordinates = dynamic_cast<::mfem::ParGridFunction*>( mesh.GetNodes() );
  if ( coordinates == nullptr ) {
    return false;
  }

  const RunResult automatic = runContact( mesh, *coordinates, 1, order );
  const RunResult overintegrated = runContact( mesh, *coordinates, 8, order );
  const AdapterReference array = arrayReference( dimension );
  const auto force_balance = globalResidualSum( *coordinates->ParFESpace(), automatic.residual );
  const auto& summary = automatic.diagnostics.global.rows;
  const std::array<double, 3> expected_force =
      dimension == 2 ? std::array<double, 3>{ 0.0, -0.5, 0.0 } : std::array<double, 3>{ 0.0, 0.0, -0.5 };
  const std::array<double, 3> expected_moment =
      dimension == 2 ? std::array<double, 3>{ 0.0, 0.0, -0.25 } : std::array<double, 3>{ -0.25, 0.25, 0.0 };
  const int normal_component = dimension - 1;
  bool valid = automatic.evaluation.active_interactions > 0 && summary.rows > 0 &&
               summary.active_rows == summary.rows && summary.nonfinite_rows == 0 &&
               close( summary.minimum_gap, -0.1 ) && close( summary.maximum_penetration, 0.1 ) &&
               close( summary.maximum_traction, 0.5 ) && close( summary.impulse[normal_component], -0.125 ) &&
               close( summary.spring_energy, 0.025 ) && close( automatic.evaluation.energy, 0.025 ) &&
               automatic.diagnostics.global.quadrature.samples >= summary.rows &&
               close( automatic.diagnostics.global.quadrature.maximum_penetration, 0.1 ) &&
               residualDifference( automatic.residual, overintegrated.residual ) < 2.0e-11 &&
               close( automatic.evaluation.energy, overintegrated.evaluation.energy ) &&
               close( automatic.evaluation.energy, array.evaluation.energy ) &&
               close( summary.minimum_gap, array.diagnostics.rows.minimum_gap ) &&
               close( summary.maximum_penetration, array.diagnostics.rows.maximum_penetration ) &&
               close( summary.maximum_traction, array.diagnostics.rows.maximum_traction ) &&
               close( summary.spring_energy, array.diagnostics.rows.spring_energy ) &&
               close( automatic.diagnostics.global.quadrature.maximum_penetration,
                      array.diagnostics.quadrature.maximum_penetration );
  for ( int component = 0; component < 3; ++component ) {
    valid = valid && close( summary.mortar_force[component], expected_force[component] ) &&
            close( summary.mortar_moment[component], expected_moment[component] ) &&
            close( summary.mortar_force[component], array.diagnostics.rows.mortar_force[component] ) &&
            close( summary.mortar_moment[component], array.diagnostics.rows.mortar_moment[component] ) &&
            close( summary.impulse[component], array.diagnostics.rows.impulse[component] ) &&
            std::abs( force_balance[component] ) < 2.0e-11;
  }
  if ( !valid && rank == 0 ) {
    std::cerr << dimension << "D parent-order contract failed: order=" << order << " rows=" << summary.rows
              << " force=(" << summary.mortar_force[0] << ',' << summary.mortar_force[1] << ','
              << summary.mortar_force[2] << ") moment=(" << summary.mortar_moment[0] << ',' << summary.mortar_moment[1]
              << ',' << summary.mortar_moment[2] << ") gap=" << summary.minimum_gap
              << " traction=" << summary.maximum_traction << " energy=" << summary.spring_energy
              << " array_active=" << array.evaluation.active_interactions
              << " array_gap=" << array.diagnostics.rows.minimum_gap << " array_energy=" << array.evaluation.energy
              << " array_force=(" << array.diagnostics.rows.mortar_force[0] << ','
              << array.diagnostics.rows.mortar_force[1] << ',' << array.diagnostics.rows.mortar_force[2]
              << ") array_moment=(" << array.diagnostics.rows.mortar_moment[0] << ','
              << array.diagnostics.rows.mortar_moment[1] << ',' << array.diagnostics.rows.mortar_moment[2] << ')'
              << " over_error=" << residualDifference( automatic.residual, overintegrated.residual ) << '\n';
  }
  return valid;
}

double curvedEnergy( int subdivision_factor )
{
  auto serial_mesh = ::mfem::Mesh::MakeCartesian3D( 4, 1, 1, ::mfem::Element::HEXAHEDRON, 1.0, 1.0, 0.1 );
  markHorizontalSurfaces( serial_mesh );
  ::mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mesh.SetCurvature( tribol::maximumParentTraceOrder );
  auto* coordinates = dynamic_cast<::mfem::ParGridFunction*>( mesh.GetNodes() );
  if ( coordinates == nullptr ) {
    return -1.0;
  }
  curveCoordinates( *coordinates );
  return runContact( mesh, *coordinates, 8, subdivision_factor ).evaluation.energy;
}

bool curvedRefinementContract( int rank )
{
  const double coarse = curvedEnergy( 1 );
  const double medium = curvedEnergy( 2 );
  const double fine = curvedEnergy( 4 );
  const double reference = curvedEnergy( 8 );
  const double coarse_error = std::abs( coarse - reference );
  const double medium_error = std::abs( medium - reference );
  const double fine_error = std::abs( fine - reference );
  const bool valid =
      reference > 0.0 && coarse_error > medium_error && medium_error > fine_error && fine_error < 0.3 * coarse_error;
  if ( !valid && rank == 0 ) {
    std::cerr << "3D curved LOR convergence failed: energy=" << coarse << ',' << medium << ',' << fine << ','
              << reference << " error=" << coarse_error << ',' << medium_error << ',' << fine_error << '\n';
  }
  return valid;
}

}  // namespace

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  int size{};
  int rank{};
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  bool valid = size == 1 || size == 2 || size == 4;
  for ( int order = 1; order <= tribol::maximumParentTraceOrder; ++order ) {
    valid = planarOrderContract( 2, order, rank ) && valid;
    valid = planarOrderContract( 3, order, rank ) && valid;
  }
  valid = curvedRefinementContract( rank ) && valid;
  if ( !valid && rank == 0 && size != 1 && size != 2 && size != 4 ) {
    std::cerr << "MFEM native convergence test requires one, two, or four MPI ranks.\n";
  }
  MPI_Finalize();
  return valid ? 0 : 1;
}
