#include "tribol/adapters/mfem/MfemContact.hpp"

#include "mfem.hpp"
#include <mpi.h>

#include <cmath>
#include <iostream>
#include <vector>

// Requirements: ADAPTER-002, ADAPTER-003, ADAPTER-004, ADAPTER-005, ADAPTER-006, API-001, DIFF-004, DIFF-006
// Requirements: GEOM-001, PAR-002, PAR-005, PHYS-001

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

int exerciseContact( int communicator_size, int rank, int order, int subdivision_factor = 0 )
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
  auto contact = tribol::mfem::makeContact( mesh, *coordinates, tribol::mfem::PairedBoundaryAttributes{ { 3 }, { 1 } },
                                            options, tribol::mfem::SurfaceDiscretization{ subdivision_factor } );
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
  ::mfem::ParGridFunction primal_field( coordinates->ParFESpace() );
  primal_field = 0.0;
  for ( int scalar_dof = 0; scalar_dof < coordinates->ParFESpace()->GetNDofs(); ++scalar_dof ) {
    const int y_dof = coordinates->ParFESpace()->DofToVDof( scalar_dof, 1 );
    primal_field[y_dof] = ( *coordinates )[y_dof];
  }
  primal_field.GetTrueDofs( primal );
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
  const double global_restricted_dot = globalSum( restricted_dot );

  ::mfem::Vector jacobian_action( coordinates->ParFESpace()->GetTrueVSize() );
  jacobian_action = 0.0;
  contact.addCoordinateJacobianMult( tribol::ContactStateView{}, primal, jacobian_action );
  auto assembled_jacobian = contact.assembleCoordinateJacobian( tribol::ContactStateView{} );
  ::mfem::Vector assembled_action( jacobian_action.Size() );
  assembled_jacobian->Mult( primal, assembled_action );
  assembled_action -= jacobian_action;
  const double assembled_error = std::sqrt( globalSum( assembled_action * assembled_action ) );
  ::mfem::Vector true_coordinates;
  coordinates->GetTrueDofs( true_coordinates );
  constexpr double finite_difference_step = 1.0e-6;
  ::mfem::Vector plus_true( true_coordinates );
  ::mfem::Vector minus_true( true_coordinates );
  plus_true.Add( finite_difference_step, primal );
  minus_true.Add( -finite_difference_step, primal );
  ::mfem::ParGridFunction plus_coordinates( coordinates->ParFESpace() );
  ::mfem::ParGridFunction minus_coordinates( coordinates->ParFESpace() );
  plus_coordinates.SetFromTrueDofs( plus_true );
  minus_coordinates.SetFromTrueDofs( minus_true );
  ::mfem::Vector plus_residual( residual.Size() );
  ::mfem::Vector minus_residual( residual.Size() );
  plus_residual = 0.0;
  minus_residual = 0.0;
  contact.updateGeometry( plus_coordinates );
  contact.addResidual( plus_residual );
  contact.updateGeometry( minus_coordinates );
  contact.addResidual( minus_residual );
  plus_residual -= minus_residual;
  plus_residual /= 2.0 * finite_difference_step;
  const double finite_difference_norm = std::sqrt( globalSum( plus_residual * plus_residual ) );
  const double jacobian_action_norm = std::sqrt( globalSum( jacobian_action * jacobian_action ) );
  plus_residual -= jacobian_action;
  const double jacobian_error = std::sqrt( globalSum( plus_residual * plus_residual ) );
  contact.updateGeometry( *coordinates );

  const auto surfaces = contact.core().surfaces();
  bool local_curved_sample_found = order == 1;
  for ( tribol::Index node = 0; node < surfaces.mortar.numberOfNodes(); ++node ) {
    local_curved_sample_found = local_curved_sample_found || surfaces.mortar.coordinates( node, 1 ) > 0.1001;
  }
  int curved_sample_found{};
  const int local_curved_sample = local_curved_sample_found ? 1 : 0;
  MPI_Allreduce( &local_curved_sample, &curved_sample_found, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );
  const int expected_factor = subdivision_factor > 0 ? subdivision_factor : order;
  const auto global_mortar_elements = globalSum( surfaces.mortar.numberOfElements() );
  const auto global_mortar_nodes = globalSum( surfaces.mortar.numberOfNodes() );
  const auto global_interactions = globalSum( contact.core().interactions().size() );
  const bool expected_refinement =
      global_mortar_elements == expected_factor && global_mortar_nodes == 2 * expected_factor;
  const bool energy_is_correct = order == 1 ? std::abs( summary.energy - 0.025 ) <= 1.0e-12
                                            : std::isfinite( summary.energy ) && summary.energy > 0.0;
  const bool valid = global_interactions > 0 && expected_refinement && curved_sample_found != 0 && energy_is_correct &&
                     std::abs( residual_sum ) <= 1.0e-12 &&
                     std::abs( global_restricted_dot - global_transpose_dot ) <= 1.0e-11 && jacobian_error <= 1.0e-7 &&
                     assembled_error <= 1.0e-12;
  if ( !valid && rank == 0 ) {
    std::cerr << "order=" << order << " interactions=" << contact.core().interactions().size()
              << " mortar_elements=" << global_mortar_elements << " energy=" << summary.energy
              << " residual_sum=" << residual_sum << " adjoint_error=" << global_restricted_dot - global_transpose_dot
              << " jacobian_error=" << jacobian_error << " fd_norm=" << finite_difference_norm
              << " action_norm=" << jacobian_action_norm << " assembled_error=" << assembled_error
              << " curved_sample=" << curved_sample_found << '\n';
    for ( int dof = 0; dof < plus_residual.Size(); ++dof ) {
      if ( std::abs( plus_residual[dof] ) > 1.0e-8 ) {
        std::cerr << "  derivative error dof=" << dof << " value=" << plus_residual[dof] << '\n';
      }
    }
  }
  return valid ? 0 : 1;
}

int exerciseStateMapping( int communicator_size, int order )
{
  ::mfem::Mesh serial_mesh =
      ::mfem::Mesh::MakeCartesian2D( 1, communicator_size, ::mfem::Element::QUADRILATERAL, true, 1.0, 0.1 );
  ::mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mesh.SetCurvature( order );
  auto* coordinates = dynamic_cast<::mfem::ParGridFunction*>( mesh.GetNodes() );
  if ( coordinates == nullptr ) {
    return 1;
  }

  using MethodType = tribol::Method<tribol::geometry::ProjectedOverlap<tribol::normal::MeanPlane>,
                                    tribol::integration::Centroid, tribol::constraint::Pointwise,
                                    tribol::enforcement::Penalty<tribol::stiffness::Material, tribol::rate::Percentage>,
                                    tribol::response::Frictionless, tribol::formulation::PointwiseTraction,
                                    tribol::linearization::Exact>;
  tribol::mfem::MfemContact<MethodType>::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.scale = 0.75;
  options.method.enforcement.rate.ratio = 0.1;
  options.timestep = { .enabled = true, .current_step = 1.0 };
  auto contact = tribol::mfem::makeContact<MethodType>(
      mesh, *coordinates, tribol::mfem::PairedBoundaryAttributes{ { 3 }, { 1 } }, options );
  contact.updateInteractions();

  ::mfem::ParGridFunction velocity( coordinates->ParFESpace() );
  velocity = 0.0;
  for ( int scalar_dof = 0; scalar_dof < coordinates->ParFESpace()->GetNDofs(); ++scalar_dof ) {
    velocity[coordinates->ParFESpace()->DofToVDof( scalar_dof, 0 )] = 2.0;
  }
  ::mfem::H1_FECollection scalar_collection( order, mesh.Dimension() );
  ::mfem::ParFiniteElementSpace scalar_space( &mesh, &scalar_collection );
  ::mfem::ParGridFunction pressure( &scalar_space );
  pressure = -2.0;
  ::mfem::ConstantCoefficient modulus( 5.0 );
  tribol::mfem::ContactState state{
      .velocity = &velocity,
      .reference_coordinates = coordinates,
      .multiplier = &pressure,
      .external_potential_density = &pressure,
      .external_pressure = &pressure,
      .external_pressure_tangent = &pressure,
      .material_modulus = &modulus,
  };
  const auto restricted = contact.restrictState( state );
  const auto surfaces = contact.core().surfaces();
  bool valid = restricted.mortar_velocity.entities == surfaces.mortar.numberOfNodes() &&
               restricted.nonmortar_velocity.entities == surfaces.nonmortar.numberOfNodes() &&
               restricted.mortar_element_thickness.size() == surfaces.mortar.numberOfElements() &&
               restricted.nonmortar_element_thickness.size() == surfaces.nonmortar.numberOfElements() &&
               restricted.multiplier.size() == surfaces.mortar.numberOfNodes() &&
               restricted.external_potential_density.size() == surfaces.mortar.numberOfNodes() &&
               restricted.external_pressure.size() == surfaces.mortar.numberOfNodes() &&
               restricted.external_pressure_tangent.size() == surfaces.mortar.numberOfNodes();
  for ( tribol::Real value : restricted.mortar_element_thickness ) {
    valid = valid && std::isfinite( value ) && value > 0.0;
  }
  for ( tribol::Real value : restricted.mortar_material_modulus ) {
    valid = valid && std::abs( value - 5.0 ) < 1.0e-12;
  }
  for ( tribol::Index node = 0; node < restricted.mortar_velocity.entities; ++node ) {
    valid = valid && std::abs( restricted.mortar_velocity( node, 0 ) - 2.0 ) < 1.0e-12 &&
            std::abs( restricted.multiplier[node] + 2.0 ) < 1.0e-12;
  }
  ::mfem::Vector residual( coordinates->ParFESpace()->GetTrueVSize() );
  residual = 0.0;
  const auto summary = contact.addResidual( state, residual );
  const auto result = contact.evaluate( state );
  ::mfem::ParGridFunction gap( &scalar_space );
  ::mfem::ParGridFunction contact_pressure( &scalar_space );
  contact.writeGap( result, gap );
  contact.writePressure( result, contact_pressure );
  ::mfem::Vector true_gap;
  ::mfem::Vector true_pressure;
  gap.GetTrueDofs( true_gap );
  contact_pressure.GetTrueDofs( true_pressure );
  const double gap_norm = std::sqrt( globalSum( true_gap * true_gap ) );
  const double pressure_norm = std::sqrt( globalSum( true_pressure * true_pressure ) );
  valid = valid && gap_norm > 0.0 && pressure_norm > 0.0;
  if ( !valid || summary.active_interactions == 0 || !std::isfinite( summary.timestep_vote ) ) {
    int rank{};
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    std::cerr << "state mapping rank=" << rank << " order=" << order << " valid=" << valid
              << " active=" << summary.active_interactions << " dt=" << summary.timestep_vote
              << " gap_norm=" << gap_norm << " pressure_norm=" << pressure_norm
              << " mortar_nodes=" << surfaces.mortar.numberOfNodes()
              << " thickness=" << restricted.mortar_element_thickness.size() << '\n';
  }
  return valid && summary.active_interactions > 0 && std::isfinite( summary.timestep_vote ) ? 0 : 1;
}

int exerciseExternalPressureJacobian( int communicator_size )
{
  ::mfem::Mesh serial_mesh =
      ::mfem::Mesh::MakeCartesian2D( 1, communicator_size, ::mfem::Element::QUADRILATERAL, true, 1.0, 0.1 );
  ::mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mesh.SetCurvature( 1 );
  auto* coordinates = dynamic_cast<::mfem::ParGridFunction*>( mesh.GetNodes() );
  if ( coordinates == nullptr ) {
    return 1;
  }
  ::mfem::H1_FECollection scalar_collection( 1, mesh.Dimension() );
  ::mfem::ParFiniteElementSpace scalar_space( &mesh, &scalar_collection );
  ::mfem::ParGridFunction pressure( &scalar_space );
  ::mfem::ParGridFunction potential( &scalar_space );
  ::mfem::ParGridFunction tangent( &scalar_space );
  ::mfem::ParGridFunction gap( &scalar_space );
  ::mfem::ParGridFunction weighted_gap( &scalar_space );
  ::mfem::ParGridFunction tributary_area( &scalar_space );
  using MethodType =
      tribol::Method<tribol::geometry::ProjectedOverlap<tribol::normal::MortarSurface>, tribol::integration::Polygon<2>,
                     tribol::constraint::Nodal<tribol::basis::Primal>, tribol::enforcement::ExternalPressure,
                     tribol::response::Frictionless, tribol::formulation::Variational, tribol::linearization::Exact>;
  tribol::mfem::MfemContact<MethodType>::Options options;
  options.search.expansion = 0.2;
  auto contact = tribol::mfem::makeContact<MethodType>(
      mesh, *coordinates, tribol::mfem::PairedBoundaryAttributes{ { 3 }, { 1 } }, options );
  contact.updateInteractions();
  contact.evaluateNodalKinematics( gap, weighted_gap, tributary_area );
  for ( int dof = 0; dof < pressure.Size(); ++dof ) {
    const auto active_gap = std::min( gap[dof], 0.0 );
    potential[dof] = 5.0 * active_gap * active_gap;
    pressure[dof] = 10.0 * active_gap;
    tangent[dof] = gap[dof] < 0.0 ? 10.0 : 0.0;
  }
  const tribol::mfem::ContactState state{
      .external_potential_density = &potential, .external_pressure = &pressure, .external_pressure_tangent = &tangent };
  ::mfem::Vector direction( scalar_space.GetTrueVSize() );
  direction = 1.0;
  ::mfem::Vector matrix_free( coordinates->ParFESpace()->GetTrueVSize() );
  matrix_free = 0.0;
  contact.addExternalPressureJacobianMult( state, direction, matrix_free );
  auto assembled = contact.assembleExternalPressureJacobian( state );
  ::mfem::Vector matrix_action( matrix_free.Size() );
  assembled->Mult( direction, matrix_action );
  matrix_action -= matrix_free;
  ::mfem::Vector true_area;
  tributary_area.GetTrueDofs( true_area );
  return std::sqrt( globalSum( matrix_action * matrix_action ) ) < 1.0e-12 && globalSum( true_area * true_area ) > 0.0
             ? 0
             : 1;
}

int exerciseDynamicRedistribution( int communicator_size )
{
  ::mfem::Mesh serial_mesh =
      ::mfem::Mesh::MakeCartesian2D( 1, communicator_size, ::mfem::Element::QUADRILATERAL, true, 1.0, 0.1 );
  ::mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mesh.SetCurvature( 1 );
  auto* coordinates = dynamic_cast<::mfem::ParGridFunction*>( mesh.GetNodes() );
  if ( coordinates == nullptr ) {
    return 1;
  }
  ::mfem::ParGridFunction separated_coordinates( coordinates->ParFESpace() );
  separated_coordinates = *coordinates;
  for ( int scalar_dof = 0; scalar_dof < coordinates->ParFESpace()->GetNDofs(); ++scalar_dof ) {
    const int y_dof = coordinates->ParFESpace()->DofToVDof( scalar_dof, 1 );
    if ( std::abs( ( *coordinates )[y_dof] ) < 1.0e-12 ) {
      separated_coordinates[y_dof] = -1.0;
    }
  }

  tribol::mfem::MfemContact<>::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.value = 10.0;
  auto contact = tribol::mfem::makeContact( mesh, separated_coordinates,
                                            tribol::mfem::PairedBoundaryAttributes{ { 3 }, { 1 } }, options );
  contact.updateInteractions();
  const double initial_interactions = globalSum( contact.core().interactions().size() );
  const double initial_nonmortar_elements = globalSum( contact.core().surfaces().nonmortar.numberOfElements() );

  contact.rebuildGeometry( *coordinates );
  contact.updateInteractions();
  const double rebuilt_interactions = globalSum( contact.core().interactions().size() );
  const double rebuilt_nonmortar_elements = globalSum( contact.core().surfaces().nonmortar.numberOfElements() );
  ::mfem::Vector residual( coordinates->ParFESpace()->GetTrueVSize() );
  residual = 0.0;
  const auto summary = contact.addResidual( residual );
  return initial_interactions == 0.0 && initial_nonmortar_elements == 0.0 && rebuilt_interactions > 0.0 &&
                 rebuilt_nonmortar_elements > 0.0 && summary.active_interactions > 0
             ? 0
             : 1;
}

int exerciseMultiplierSystem( int communicator_size )
{
  ::mfem::Mesh serial_mesh =
      ::mfem::Mesh::MakeCartesian2D( 1, communicator_size, ::mfem::Element::QUADRILATERAL, true, 1.0, 0.1 );
  ::mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mesh.SetCurvature( 1 );
  auto* coordinates = dynamic_cast<::mfem::ParGridFunction*>( mesh.GetNodes() );
  if ( coordinates == nullptr ) {
    return 1;
  }
  ::mfem::H1_FECollection scalar_collection( 1, mesh.Dimension() );
  ::mfem::ParFiniteElementSpace scalar_space( &mesh, &scalar_collection );
  ::mfem::ParGridFunction multiplier( &scalar_space );
  multiplier = -2.0;
  using MethodType = tribol::Method<tribol::geometry::ProjectedOverlap<tribol::normal::MortarSurface>,
                                    tribol::integration::Polygon<2>, tribol::constraint::Nodal<tribol::basis::Primal>,
                                    tribol::enforcement::LagrangeMultiplier, tribol::response::Frictionless,
                                    tribol::formulation::WeightedWeakForm, tribol::linearization::Exact>;
  tribol::mfem::MfemContact<MethodType>::Options options;
  options.search.expansion = 0.2;
  auto contact = tribol::mfem::makeContact<MethodType>(
      mesh, *coordinates, tribol::mfem::PairedBoundaryAttributes{ { 3 }, { 1 } }, options );
  contact.updateInteractions();
  const tribol::mfem::ContactState state{ .multiplier = &multiplier };
  const int coordinate_size = coordinates->ParFESpace()->GetTrueVSize();
  const int multiplier_size = scalar_space.GetTrueVSize();
  ::mfem::Vector force( coordinate_size );
  ::mfem::Vector constraint( multiplier_size );
  force = 0.0;
  constraint = 0.0;
  const auto summary = contact.addResidual( state, force, constraint );
  auto jacobian = contact.assembleSystemJacobian( state );

  ::mfem::Vector coordinate_direction( coordinate_size );
  ::mfem::Vector multiplier_direction( multiplier_size );
  for ( int entry = 0; entry < coordinate_size; ++entry ) {
    coordinate_direction[entry] = 0.01 * ( entry + 1 );
  }
  for ( int entry = 0; entry < multiplier_size; ++entry ) {
    multiplier_direction[entry] = 0.02 * ( entry + 1 );
  }
  ::mfem::Vector expected_force( coordinate_size );
  ::mfem::Vector expected_constraint( multiplier_size );
  ::mfem::Vector temporary_force( coordinate_size );
  ::mfem::Vector temporary_constraint( multiplier_size );
  jacobian->forceCoordinates().Mult( coordinate_direction, expected_force );
  jacobian->forceMultiplier().Mult( multiplier_direction, temporary_force );
  expected_force += temporary_force;
  jacobian->constraintCoordinates().Mult( coordinate_direction, expected_constraint );
  jacobian->constraintMultiplier().Mult( multiplier_direction, temporary_constraint );
  expected_constraint += temporary_constraint;

  ::mfem::Vector block_input( coordinate_size + multiplier_size );
  ::mfem::Vector block_output( coordinate_size + multiplier_size );
  for ( int entry = 0; entry < coordinate_size; ++entry ) {
    block_input[entry] = coordinate_direction[entry];
  }
  for ( int entry = 0; entry < multiplier_size; ++entry ) {
    block_input[coordinate_size + entry] = multiplier_direction[entry];
  }
  jacobian->blockOperator().Mult( block_input, block_output );
  tribol::Real local_error{};
  for ( int entry = 0; entry < coordinate_size; ++entry ) {
    const tribol::Real difference = block_output[entry] - expected_force[entry];
    local_error += difference * difference;
  }
  for ( int entry = 0; entry < multiplier_size; ++entry ) {
    const tribol::Real difference = block_output[coordinate_size + entry] - expected_constraint[entry];
    local_error += difference * difference;
  }
  return summary.active_interactions > 0 && std::sqrt( globalSum( local_error ) ) < 1.0e-12 ? 0 : 1;
}

}  // namespace

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  int communicator_size{};
  int rank{};
  MPI_Comm_size( MPI_COMM_WORLD, &communicator_size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  const int linear = exerciseContact( communicator_size, rank, 1 );
  const int curved = exerciseContact( communicator_size, rank, 2 );
  const int custom_subdivision = exerciseContact( communicator_size, rank, 2, 3 );
  const int linear_state = exerciseStateMapping( communicator_size, 1 );
  const int curved_state = exerciseStateMapping( communicator_size, 2 );
  const int multiplier = exerciseMultiplierSystem( communicator_size );
  const int external_pressure = exerciseExternalPressureJacobian( communicator_size );
  const int redistribution = exerciseDynamicRedistribution( communicator_size );
  const int result = linear | curved | custom_subdivision | linear_state | curved_state | multiplier |
                     external_pressure | redistribution;
  if ( result != 0 && rank == 0 ) {
    std::cerr << "MFEM adapter subtests: linear=" << linear << " curved=" << curved
              << " custom_subdivision=" << custom_subdivision << " linear_state=" << linear_state
              << " curved_state=" << curved_state << " multiplier=" << multiplier
              << " external_pressure=" << external_pressure << " redistribution=" << redistribution << '\n';
  }
  MPI_Finalize();
  return result;
}
