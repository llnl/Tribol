#include "tribol/adapters/mfem/MfemContact.hpp"

#include "mfem.hpp"
#include <mpi.h>

#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

// Requirements: ADAPTER-002, API-005, CONS-002, GEOM-001, PAR-002, PAR-005, PHYS-001, TIME-002

namespace {

using NativeExplicitMethod =
    tribol::Method<tribol::geometry::ProjectedOverlap<tribol::normal::MeanPlane>, tribol::integration::Automatic,
                   tribol::constraint::QuadraturePoint,
                   tribol::enforcement::Penalty<tribol::stiffness::Constant, tribol::rate::Constant>,
                   tribol::response::Frictionless, tribol::formulation::PointwiseTraction,
                   tribol::linearization::Exact>;

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

void curveCoordinates3D( ::mfem::ParGridFunction& coordinates )
{
  auto& space = *coordinates.ParFESpace();
  for ( int scalar_dof = 0; scalar_dof < space.GetNDofs(); ++scalar_dof ) {
    const int x_dof = space.DofToVDof( scalar_dof, 0 );
    const int y_dof = space.DofToVDof( scalar_dof, 1 );
    const int z_dof = space.DofToVDof( scalar_dof, 2 );
    const double x = coordinates[x_dof];
    const double y = coordinates[y_dof];
    coordinates[z_dof] += 0.02 * x * ( 1.0 - x ) * y * ( 1.0 - y );
  }
}

double globalSum( double local )
{
  double global{};
  MPI_Allreduce( &local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  return global;
}

double globalMax( double local )
{
  double global{};
  MPI_Allreduce( &local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  return global;
}

double parentTraceCoordinateError( const tribol::ParentTraceView& trace,
                                   tribol::FieldView<const tribol::Real> parent_coordinates )
{
  if ( !trace.isStructurallyValid() || !parent_coordinates.isStructurallyValid() ||
       parent_coordinates.components != trace.geometry.dimension ) {
    return 1.0;
  }
  double error{};
  for ( tribol::Index subfacet = 0; subfacet < trace.numberOfSubfacets(); ++subfacet ) {
    const tribol::Index parent_face = trace.subfacet_parent_faces[subfacet];
    for ( tribol::Index offset = trace.geometry.element_offsets[subfacet];
          offset < trace.geometry.element_offsets[subfacet + 1]; ++offset ) {
      const tribol::basis::ReferencePoint point{
          trace.subfacet_parent_reference_coordinates( offset, 0 ),
          trace.geometry.dimension == 3 ? trace.subfacet_parent_reference_coordinates( offset, 1 ) : 0.0 };
      const auto shape = tribol::basis::parentShape( trace, parent_face, point );
      const tribol::Index dof_begin = trace.parent_dof_offsets[parent_face];
      const tribol::Index geometry_node = trace.geometry.connectivity[offset];
      for ( int component = 0; component < trace.geometry.dimension; ++component ) {
        double expected{};
        for ( int local_dof = 0; local_dof < shape.size; ++local_dof ) {
          const tribol::Index dof = trace.parent_dofs[dof_begin + local_dof];
          expected +=
              shape[local_dof] * trace.parent_dof_signs[dof_begin + local_dof] * parent_coordinates( dof, component );
        }
        error = std::max( error, std::abs( expected - trace.geometry.coordinates( geometry_node, component ) ) );
      }
    }
  }
  return error;
}

bool parentDofIdentitiesAreCollisionFree( const tribol::ParentTraceView& trace,
                                          tribol::FieldView<const tribol::Real> parent_field )
{
  std::map<tribol::Index, int> owners;
  for ( tribol::Index face = 0; face < trace.numberOfParentFaces(); ++face ) {
    for ( tribol::Index offset = trace.parent_dof_offsets[face]; offset < trace.parent_dof_offsets[face + 1];
          ++offset ) {
      const tribol::Index dof = trace.parent_dofs[offset];
      if ( dof < 0 || dof >= parent_field.entities ) {
        return false;
      }
      const auto [iterator, inserted] = owners.emplace( dof, trace.parent_face_owners[face] );
      if ( !inserted && iterator->second != trace.parent_face_owners[face] ) {
        return false;
      }
    }
  }
  return true;
}

int exerciseParentTrace3D( int order, ::mfem::Element::Type element_type )
{
  ::mfem::Mesh serial_mesh = ::mfem::Mesh::MakeCartesian3D( 1, 1, 1, element_type, 1.0, 1.0, 1.0 );
  ::mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mesh.SetCurvature( order );
  auto* coordinates = dynamic_cast<::mfem::ParGridFunction*>( mesh.GetNodes() );
  if ( coordinates == nullptr ) {
    return 1;
  }
  curveCoordinates( *coordinates );
  const tribol::mfem::detail::SurfaceStorage surface( mesh, *coordinates, std::vector<int>{ 1 } );
  const auto trace = surface.parentTraceView();
  const double error = globalMax( parentTraceCoordinateError( trace, surface.parentCoordinateView() ) );
  const double global_faces = globalSum( trace.numberOfParentFaces() );
  return trace.isStructurallyValid() && global_faces > 0.0 && error <= 1.0e-10 ? 0 : 1;
}

int exerciseNativeParentContact( int communicator_size, int rank, int order )
{
  ::mfem::Mesh serial_mesh =
      ::mfem::Mesh::MakeCartesian2D( communicator_size, 1, ::mfem::Element::QUADRILATERAL, true, 1.0, 0.1 );
  ::mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mesh.SetCurvature( order );
  auto* coordinates = dynamic_cast<::mfem::ParGridFunction*>( mesh.GetNodes() );
  if ( coordinates == nullptr ) {
    return 1;
  }
  curveCoordinates( *coordinates );

  ::mfem::ParGridFunction velocity( coordinates->ParFESpace() );
  for ( int scalar_dof = 0; scalar_dof < coordinates->ParFESpace()->GetNDofs(); ++scalar_dof ) {
    const int x_dof = coordinates->ParFESpace()->DofToVDof( scalar_dof, 0 );
    const int y_dof = coordinates->ParFESpace()->DofToVDof( scalar_dof, 1 );
    velocity[x_dof] = 0.1 + 0.05 * ( *coordinates )[x_dof];
    velocity[y_dof] = -0.2 + 0.03 * ( *coordinates )[x_dof];
  }

  tribol::mfem::MfemContact<NativeExplicitMethod>::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.value = 10.0;
  options.method.enforcement.rate.value = 0.25;
  auto contact = tribol::mfem::makeContact<NativeExplicitMethod>(
      mesh, *coordinates, tribol::mfem::PairedBoundaryAttributes{ { 1 }, { 3 } }, options );
  ::mfem::Vector inverse_mass( coordinates->ParFESpace()->GetTrueVSize() );
  inverse_mass = 2.0;
  contact.registerInverseMass( inverse_mass );
  contact.registerIntegrator( tribol::timestep::IntegratorStabilityLimits::centralDifference() );
  contact.updateInteractions();

  const auto traces = contact.parentTraces();
  const auto fields = contact.parentFields();
  const double mortar_coordinate_error =
      globalMax( parentTraceCoordinateError( traces.mortar, fields.mortar_coordinates ) );
  const double nonmortar_coordinate_error =
      globalMax( parentTraceCoordinateError( traces.nonmortar, fields.nonmortar_coordinates ) );
  bool valid = traces.isStructurallyValid() &&
               parentDofIdentitiesAreCollisionFree( traces.mortar, fields.mortar_coordinates ) &&
               parentDofIdentitiesAreCollisionFree( traces.nonmortar, fields.nonmortar_coordinates ) &&
               mortar_coordinate_error < 1.0e-10 && nonmortar_coordinate_error < 1.0e-10;
  if ( order > 1 ) {
    valid = valid && fields.mortar_coordinates.entities < traces.mortar.geometry.numberOfNodes() &&
            fields.nonmortar_coordinates.entities < traces.nonmortar.geometry.numberOfNodes();
  }

  const auto state = contact.restrictState( tribol::mfem::ContactState{ .velocity = &velocity } );
  ::mfem::Vector true_velocity;
  velocity.GetTrueDofs( true_velocity );
  std::vector<tribol::Real> restricted_mortar;
  std::vector<tribol::Real> restricted_nonmortar;
  contact.restrictPrimal( true_velocity, restricted_mortar, restricted_nonmortar );
  if ( state.mortar_velocity.values.size() != static_cast<tribol::Index>( restricted_mortar.size() ) ||
       state.nonmortar_velocity.values.size() != static_cast<tribol::Index>( restricted_nonmortar.size() ) ) {
    valid = false;
  } else {
    for ( std::size_t value = 0; value < restricted_mortar.size(); ++value ) {
      valid = valid && std::abs( state.mortar_velocity.values[static_cast<tribol::Index>( value )] -
                                 restricted_mortar[value] ) < 1.0e-12;
    }
    for ( std::size_t value = 0; value < restricted_nonmortar.size(); ++value ) {
      valid = valid && std::abs( state.nonmortar_velocity.values[static_cast<tribol::Index>( value )] -
                                 restricted_nonmortar[value] ) < 1.0e-12;
    }
  }

  ::mfem::Vector primal( coordinates->ParFESpace()->GetTrueVSize() );
  const auto first_true_dof = coordinates->ParFESpace()->GetMyTDofOffset();
  for ( int dof = 0; dof < primal.Size(); ++dof ) {
    primal[dof] = 0.01 * static_cast<double>( first_true_dof + dof + 1 );
  }
  contact.restrictPrimal( primal, restricted_mortar, restricted_nonmortar );
  std::vector<tribol::Real> dual_mortar( restricted_mortar.size() );
  std::vector<tribol::Real> dual_nonmortar( restricted_nonmortar.size() );
  for ( std::size_t value = 0; value < dual_mortar.size(); ++value ) {
    dual_mortar[value] = 0.125 + 0.01 * static_cast<double>( value + rank );
  }
  for ( std::size_t value = 0; value < dual_nonmortar.size(); ++value ) {
    dual_nonmortar[value] = -0.25 - 0.02 * static_cast<double>( value + rank );
  }
  double restricted_dot{};
  for ( std::size_t value = 0; value < dual_mortar.size(); ++value ) {
    restricted_dot += restricted_mortar[value] * dual_mortar[value];
  }
  for ( std::size_t value = 0; value < dual_nonmortar.size(); ++value ) {
    restricted_dot += restricted_nonmortar[value] * dual_nonmortar[value];
  }
  ::mfem::Vector transpose( primal.Size() );
  transpose = 0.0;
  contact.addDualTranspose( { dual_mortar.data(), static_cast<tribol::Index>( dual_mortar.size() ) },
                            { dual_nonmortar.data(), static_cast<tribol::Index>( dual_nonmortar.size() ) }, transpose );
  const double transpose_error = std::abs( globalSum( restricted_dot ) - globalSum( primal * transpose ) );

  ::mfem::Vector residual( coordinates->ParFESpace()->GetTrueVSize() );
  residual = 0.0;
  const auto summary = contact.addResidual( tribol::mfem::ContactState{ .velocity = &velocity }, residual );
  const auto cells = contact.core().overlapCells();
  double local_accepted_cells{};
  for ( tribol::Index cell = 0; cell < cells.numberOfAttempts(); ++cell ) {
    local_accepted_cells += cells.isAccepted( cell ) ? 1.0 : 0.0;
  }
  const double accepted_cells = globalSum( local_accepted_cells );
  double local_force_sum{};
  for ( int dof = 0; dof < residual.Size(); ++dof ) {
    local_force_sum += residual[dof];
  }
  const double residual_norm = std::sqrt( globalSum( residual * residual ) );
  const double force_sum = globalSum( local_force_sum );
  valid = valid && summary.active_interactions > 0 && summary.quadrature_points > accepted_cells &&
          summary.energy > 0.0 && std::isfinite( summary.timestep_vote ) && summary.timestep_vote > 0.0 &&
          std::abs( force_sum ) < 1.0e-10 && transpose_error < 1.0e-10 && residual_norm > 0.0;
  if ( !valid && rank == 0 ) {
    std::cerr << "native parent rank=" << rank << " order=" << order << " active=" << summary.active_interactions
              << " rows=" << summary.quadrature_points << " energy=" << summary.energy
              << " dt=" << summary.timestep_vote << " residual_norm=" << residual_norm
              << " transpose_error=" << transpose_error << " interactions=" << contact.core().interactions().size()
              << " cells=" << cells.numberOfAttempts();
    for ( tribol::Index cell = 0; cell < cells.numberOfAttempts(); ++cell ) {
      std::cerr << " status[" << cell << "]=" << static_cast<int>( cells.statuses[cell] );
    }
    std::cerr << '\n';
  }
  return valid ? 0 : 1;
}

int exerciseNativeParentContact3D( int communicator_size, int rank, int order )
{
  ::mfem::Mesh serial_mesh =
      ::mfem::Mesh::MakeCartesian3D( communicator_size, 1, 1, ::mfem::Element::HEXAHEDRON, 1.0, 1.0, 0.1 );
  ::mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mesh.SetCurvature( order );
  auto* coordinates = dynamic_cast<::mfem::ParGridFunction*>( mesh.GetNodes() );
  if ( coordinates == nullptr ) {
    return 1;
  }
  curveCoordinates3D( *coordinates );
  ::mfem::ParGridFunction velocity( coordinates->ParFESpace() );
  velocity = 0.0;
  tribol::mfem::MfemContact<NativeExplicitMethod>::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.value = 10.0;
  options.method.enforcement.rate.value = 0.25;
  auto contact = tribol::mfem::makeContact<NativeExplicitMethod>(
      mesh, *coordinates, tribol::mfem::PairedBoundaryAttributes{ { 1 }, { 6 } }, options );
  ::mfem::Vector inverse_mass( coordinates->ParFESpace()->GetTrueVSize() );
  inverse_mass = 2.0;
  contact.registerInverseMass( inverse_mass );
  contact.registerIntegrator( tribol::timestep::IntegratorStabilityLimits::centralDifference() );
  contact.updateInteractions();

  ::mfem::Vector residual( coordinates->ParFESpace()->GetTrueVSize() );
  residual = 0.0;
  const auto summary = contact.addResidual( tribol::mfem::ContactState{ .velocity = &velocity }, residual );
  const auto cells = contact.core().overlapCells();
  double local_accepted_cells{};
  for ( tribol::Index cell = 0; cell < cells.numberOfAttempts(); ++cell ) {
    local_accepted_cells += cells.isAccepted( cell ) ? 1.0 : 0.0;
  }
  const double accepted_cells = globalSum( local_accepted_cells );
  std::array<double, 3> local_force{};
  auto& space = *coordinates->ParFESpace();
  const int true_scalar_dofs = space.GetTrueVSize() / space.GetVDim();
  for ( int scalar_dof = 0; scalar_dof < true_scalar_dofs; ++scalar_dof ) {
    for ( int component = 0; component < 3; ++component ) {
      const int vector_dof = space.GetOrdering() == ::mfem::Ordering::byNODES
                                 ? component * true_scalar_dofs + scalar_dof
                                 : 3 * scalar_dof + component;
      local_force[static_cast<std::size_t>( component )] += residual[vector_dof];
    }
  }
  bool valid = summary.active_interactions > 0 && summary.quadrature_points > accepted_cells && summary.energy > 0.0 &&
               std::isfinite( summary.timestep_vote ) && summary.timestep_vote > 0.0;
  std::array<double, 3> force{};
  for ( std::size_t component = 0; component < force.size(); ++component ) {
    force[component] = globalSum( local_force[component] );
    valid = valid && std::abs( force[component] ) < 1.0e-10;
  }
  if ( !valid && rank == 0 ) {
    const auto traces = contact.parentTraces();
    const auto mortar_normal = tribol::projected_overlap_detail::faceNormal( traces.mortar.geometry, 0 );
    const auto nonmortar_normal = tribol::projected_overlap_detail::faceNormal( traces.nonmortar.geometry, 0 );
    std::array<tribol::Index, 7> status_counts{};
    for ( tribol::Index cell = 0; cell < cells.numberOfAttempts(); ++cell ) {
      ++status_counts[static_cast<std::size_t>( cells.statuses[cell] )];
    }
    std::cerr << "native 3D parent order=" << order << " active=" << summary.active_interactions
              << " rows=" << summary.quadrature_points << " energy=" << summary.energy
              << " dt=" << summary.timestep_vote << " cells=" << cells.numberOfAttempts()
              << " accepted=" << accepted_cells << " force=(" << force[0] << ',' << force[1] << ',' << force[2]
              << ") normals=(" << mortar_normal[0] << ',' << mortar_normal[1] << ',' << mortar_normal[2] << ")/("
              << nonmortar_normal[0] << ',' << nonmortar_normal[1] << ',' << nonmortar_normal[2] << ") statuses=";
    for ( std::size_t status = 0; status < status_counts.size(); ++status ) {
      std::cerr << ( status == 0 ? '(' : ',' ) << status_counts[status];
    }
    std::cerr << ")\n";
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
  int native_parent{};
  for ( int order = 1; order <= tribol::maximumParentTraceOrder; ++order ) {
    native_parent |= exerciseNativeParentContact( communicator_size, rank, order );
    native_parent |= exerciseNativeParentContact3D( communicator_size, rank, order );
  }
  int parent_trace_3d{};
  for ( int order = 1; order <= tribol::maximumParentTraceOrder; ++order ) {
    parent_trace_3d |= exerciseParentTrace3D( order, ::mfem::Element::HEXAHEDRON );
  }
  parent_trace_3d |= exerciseParentTrace3D( 3, ::mfem::Element::TETRAHEDRON );
  const int result = native_parent | parent_trace_3d;
  if ( result != 0 && rank == 0 ) {
    std::cerr << "MFEM native subtests: native_parent=" << native_parent << " parent_trace_3d=" << parent_trace_3d
              << '\n';
  }
  MPI_Finalize();
  return result;
}
