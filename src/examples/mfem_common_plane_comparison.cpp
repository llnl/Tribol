// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file mfem_common_plane_comparison.cpp
 *
 * @brief Compares legacy and ContactFormulation-based Common Plane in explicit dynamics.
 *
 * This example runs two 2D impact problems with both Common Plane implementations. Each implementation starts from an
 * independently rebuilt copy of the same mesh and uses the same material, penalty, time step, and initial conditions.
 * The example writes separate VisIt collections, reports displacement and velocity L2 differences over time, and
 * compares setup, time-integration, and contact runtimes. Each implementation performs one untimed warm-up step, and
 * device execution is synchronized at every timing boundary. VisIt writes and L2 diagnostics are excluded from the
 * timed regions so both implementations measure the same completed work.
 *
 * The available problems are a nonmatching block-on-block impact and a half-cylinder impacting a block. Both meshes are
 * assembled with shared::MeshBuilder. The ContactFormulation run uses one quadrature point because that integration
 * rule most closely matches the legacy 2D Common Plane force calculation.
 *
 * Example run from the repository root:
 * @code{.sh}
 * mpirun -np 2 <build-dir>/examples/mfem_common_plane_comparison_ex --problem all
 * @endcode
 *
 * Add @c -d @c cuda or @c -d @c hip to select device execution. For larger performance studies, @c --timing-only
 * suppresses VisIt and field-comparison output while retaining the runtime CSV.
 *
 * VisIt data and CSV comparisons are written under @c common_plane_comparison_output by default.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

// Tribol includes
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/ExecModel.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/config.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/utils/TestUtils.hpp"

// Shared includes
#include "shared/mesh/MeshBuilder.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// MFEM includes
#include "mfem.hpp"

// Axom includes
#include "axom/CLI11.hpp"
#include "axom/core/utilities/FileUtilities.hpp"
#include "axom/slic.hpp"

namespace {

using tribol::RealT;

enum class ProblemKind
{
  Blocks,
  HalfCylinder
};

enum class CommonPlaneImplementation
{
  Legacy,
  ContactFormulation
};

struct Options {
  std::string problem{ "all" };
  int refinement_levels{ 0 };
  int order{ 1 };
  RealT initial_velocity{ 0.2 };
  RealT time_step{ 1.0e-4 };
  RealT end_time{ 1.0e-2 };
  int output_cycles{ 10 };
  RealT density{ 1000.0 };
  RealT lambda{ 100000.0 };
  RealT mu{ 100000.0 };
  RealT penalty{ 500000.0 };
  std::string output_directory{ "common_plane_comparison_output" };
  std::string device{ "cpu" };
  bool timing_only{ false };
};

struct ProblemDefinition {
  std::string name;
  shared::MeshBuilder mesh;
  std::set<int> lower_contact_attributes;
  std::set<int> upper_contact_attributes;
  std::set<int> fixed_attributes;
  int moving_element_attribute;
};

struct FieldSnapshot {
  int cycle;
  RealT time;
  mfem::Vector displacement;
  mfem::Vector velocity;
  RealT displacement_l2;
  RealT velocity_l2;
};

struct FieldComparison {
  int cycle;
  RealT time;
  RealT legacy_displacement_l2;
  RealT formulation_displacement_l2;
  RealT displacement_difference_l2;
  RealT relative_displacement_difference;
  RealT legacy_velocity_l2;
  RealT formulation_velocity_l2;
  RealT velocity_difference_l2;
  RealT relative_velocity_difference;
};

struct RunTiming {
  double setup_seconds{ 0.0 };
  double integration_seconds{ 0.0 };
  double contact_seconds{ 0.0 };
  double contact_update_seconds{ 0.0 };
  double response_seconds{ 0.0 };
  int timed_steps{ 0 };
};

struct RunResult {
  std::vector<FieldSnapshot> snapshots;
  std::vector<FieldComparison> comparisons;
  RunTiming timing;
};

const char* implementationName( CommonPlaneImplementation implementation )
{
  return implementation == CommonPlaneImplementation::Legacy ? "legacy" : "formulation";
}

ProblemDefinition makeBlockProblem( int refinement_levels )
{
  constexpr int lower_horizontal_elements = 4;
  constexpr int upper_horizontal_elements = 5;
  constexpr int vertical_elements = 2;
  constexpr RealT block_width = 2.0;
  constexpr RealT overlap = 0.01;
  constexpr int upper_contact_attribute = 5;
  constexpr int upper_top_attribute = 6;
  constexpr int upper_right_attribute = 7;
  constexpr int upper_left_attribute = 8;
  constexpr int upper_element_attribute = 2;

  // The lower block occupies [0, 2] x [0, 1]. The upper block has a nonmatching five-edge contact surface and starts
  // at y=0.99, so the two horizontal contact surfaces initially overlap by 0.01.
  auto mesh =
      shared::MeshBuilder::Unify( { shared::MeshBuilder::SquareMesh( lower_horizontal_elements, vertical_elements )
                                        .scale( { block_width, 1.0 } ),
                                    shared::MeshBuilder::SquareMesh( upper_horizontal_elements, vertical_elements )
                                        .scale( { block_width, 1.0 } )
                                        .translate( { 0.0, 1.0 - overlap } )
                                        .updateAttrib( 1, upper_element_attribute )
                                        .updateBdrAttrib( 1, upper_contact_attribute )
                                        .updateBdrAttrib( 2, upper_right_attribute )
                                        .updateBdrAttrib( 3, upper_top_attribute )
                                        .updateBdrAttrib( 4, upper_left_attribute ) } )
          .refine( refinement_levels );

  constexpr int lower_bottom_attribute = 1;
  constexpr int lower_top_attribute = 3;
  return { "blocks",
           std::move( mesh ),
           { lower_top_attribute },
           { upper_contact_attribute },
           { lower_bottom_attribute },
           upper_element_attribute };
}

ProblemDefinition makeHalfCylinderProblem( int refinement_levels )
{
  constexpr int block_horizontal_elements = 8;
  constexpr int block_vertical_elements = 2;
  constexpr RealT block_width = 2.0;
  constexpr RealT block_height = 0.5;
  constexpr int radial_elements = 2;
  constexpr int hoop_elements = 8;
  constexpr RealT inner_radius = 0.25;
  constexpr RealT outer_radius = 0.5;
  constexpr RealT overlap = 0.01;
  constexpr int upper_contact_attribute = 5;
  constexpr int upper_left_cut_attribute = 6;
  constexpr int upper_right_cut_attribute = 7;
  constexpr int upper_inner_arc_attribute = 8;
  constexpr int upper_element_attribute = 2;

  // MeshBuilder::HalfCylinder2D creates the lower half of an annulus centered at the origin. Translating its center to
  // (1, 0.99) places the bottom of the outer arc at y=0.49, which overlaps the block top at y=0.5 by 0.01.
  auto mesh = shared::MeshBuilder::Unify(
                  { shared::MeshBuilder::SquareMesh( block_horizontal_elements, block_vertical_elements )
                        .scale( { block_width, block_height } ),
                    shared::MeshBuilder::HalfCylinder2D( radial_elements, hoop_elements, inner_radius, outer_radius )
                        .translate( { block_width / 2.0, block_height + outer_radius - overlap } )
                        .updateAttrib( 1, upper_element_attribute )
                        .updateBdrAttrib( 1, upper_left_cut_attribute )
                        .updateBdrAttrib( 2, upper_contact_attribute )
                        .updateBdrAttrib( 3, upper_right_cut_attribute )
                        .updateBdrAttrib( 4, upper_inner_arc_attribute ) } )
                  .refine( refinement_levels );

  constexpr int lower_bottom_attribute = 1;
  constexpr int lower_top_attribute = 3;
  return { "half_cylinder",
           std::move( mesh ),
           { lower_top_attribute },
           { upper_contact_attribute },
           { lower_bottom_attribute },
           upper_element_attribute };
}

ProblemDefinition makeProblem( ProblemKind problem, int refinement_levels )
{
  return problem == ProblemKind::Blocks ? makeBlockProblem( refinement_levels )
                                        : makeHalfCylinderProblem( refinement_levels );
}

tribol::ExecutionMode executionMode( const mfem::Device& device )
{
  (void)device;
#ifdef TRIBOL_USE_CUDA
  if ( device.Allows( mfem::Backend::CUDA_MASK ) ) {
    return tribol::ExecutionMode::Cuda;
  }
#endif
#ifdef TRIBOL_USE_HIP
  if ( device.Allows( mfem::Backend::HIP_MASK ) ) {
    return tribol::ExecutionMode::Hip;
  }
#endif
  return tribol::ExecutionMode::Sequential;
}

void synchronizeDevice( tribol::ExecutionMode execution_mode )
{
#if defined( TRIBOL_USE_CUDA ) || defined( TRIBOL_USE_HIP )
  if ( tribol::isOnDevice( execution_mode ) ) {
    MFEM_DEVICE_SYNC;
  }
#else
  (void)execution_mode;
#endif
}

void warmUpDevice( tribol::ExecutionMode execution_mode )
{
  if ( !tribol::isOnDevice( execution_mode ) ) {
    return;
  }
  mfem::Vector values;
  values.UseDevice( true );
  values.SetSize( 1 );
  auto* values_data = values.Write();
  mfem::forall( 1, [=] MFEM_HOST_DEVICE( int ) { values_data[0] = 0.0; } );
  synchronizeDevice( execution_mode );
}

mfem::Array<int> essentialVelocityDofs( mfem::ParFiniteElementSpace& field_space,
                                        const std::set<int>& fixed_attributes )
{
  mfem::Array<int> boundary_marker( field_space.GetMesh()->bdr_attributes.Max() );
  boundary_marker = 0;
  for ( int attribute : fixed_attributes ) {
    boundary_marker[attribute - 1] = 1;
  }

  mfem::Array<int> essential_dof_marker;
  field_space.GetEssentialVDofs( boundary_marker, essential_dof_marker );
  mfem::Array<int> essential_dofs;
  mfem::FiniteElementSpace::MarkerToList( essential_dof_marker, essential_dofs );
  return essential_dofs;
}

void setInitialVelocity( mfem::ParMesh& mesh, int moving_element_attribute, RealT speed,
                         mfem::ParGridFunction& velocity )
{
  mfem::Vector moving_velocity( { 0.0, -std::abs( speed ) } );
  mfem::VectorConstantCoefficient moving_velocity_coefficient( moving_velocity );
  mfem::Array<int> moving_attributes;
  moving_attributes.Append( moving_element_attribute );
  mfem::Array<mfem::VectorCoefficient*> velocity_coefficients;
  velocity_coefficients.Append( &moving_velocity_coefficient );
  mfem::PWVectorCoefficient initial_velocity( mesh.SpaceDimension(), moving_attributes, velocity_coefficients );
  velocity.ProjectCoefficient( initial_velocity );
}

RealT fieldL2Norm( const mfem::ParGridFunction& field, mfem::VectorCoefficient& zero_coefficient )
{
  return field.ComputeL2Error( zero_coefficient );
}

double globalMax( double local_value )
{
  double global_value = 0.0;
  MPI_Allreduce( &local_value, &global_value, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  return global_value;
}

RealT relativeDifference( RealT difference, RealT first_norm, RealT second_norm )
{
  const RealT scale = std::max( first_norm, second_norm );
  return scale > std::numeric_limits<RealT>::epsilon() ? difference / scale : 0.0;
}

void saveVisitState( mfem::VisItDataCollection& data_collection, int cycle, RealT time, RealT time_step )
{
  data_collection.SetCycle( cycle );
  data_collection.SetTime( time );
  data_collection.SetTimeStep( time_step );
  data_collection.Save();
}

void appendSnapshot( int cycle, RealT time, const mfem::ParGridFunction& displacement,
                     const mfem::ParGridFunction& velocity, mfem::VectorCoefficient& zero_coefficient,
                     std::vector<FieldSnapshot>& snapshots )
{
  snapshots.push_back( { cycle, time, mfem::Vector( displacement ), mfem::Vector( velocity ),
                         fieldL2Norm( displacement, zero_coefficient ), fieldL2Norm( velocity, zero_coefficient ) } );
}

void appendComparison( int cycle, RealT time, const mfem::ParGridFunction& displacement,
                       const mfem::ParGridFunction& velocity, const FieldSnapshot& legacy,
                       mfem::VectorCoefficient& zero_coefficient, std::vector<FieldComparison>& comparisons )
{
  constexpr RealT snapshot_time_tolerance = 1.0e-12;
  SLIC_ERROR_ROOT_IF( cycle != legacy.cycle || std::abs( time - legacy.time ) > snapshot_time_tolerance,
                      "Legacy and ContactFormulation snapshots do not represent the same simulation time." );
  SLIC_ERROR_ROOT_IF( displacement.Size() != legacy.displacement.Size() || velocity.Size() != legacy.velocity.Size(),
                      "Legacy and ContactFormulation fields have different local layouts." );

  const RealT formulation_displacement_l2 = fieldL2Norm( displacement, zero_coefficient );
  mfem::ParGridFunction displacement_difference( displacement.ParFESpace() );
  displacement_difference = displacement;
  displacement_difference -= legacy.displacement;
  const RealT displacement_difference_l2 = fieldL2Norm( displacement_difference, zero_coefficient );

  const RealT formulation_velocity_l2 = fieldL2Norm( velocity, zero_coefficient );
  mfem::ParGridFunction velocity_difference( velocity.ParFESpace() );
  velocity_difference = velocity;
  velocity_difference -= legacy.velocity;
  const RealT velocity_difference_l2 = fieldL2Norm( velocity_difference, zero_coefficient );

  comparisons.push_back(
      { cycle, time, legacy.displacement_l2, formulation_displacement_l2, displacement_difference_l2,
        relativeDifference( displacement_difference_l2, legacy.displacement_l2, formulation_displacement_l2 ),
        legacy.velocity_l2, formulation_velocity_l2, velocity_difference_l2,
        relativeDifference( velocity_difference_l2, legacy.velocity_l2, formulation_velocity_l2 ) } );
}

RunResult runProblem( ProblemKind problem_kind, CommonPlaneImplementation implementation, const Options& options,
                      tribol::ExecutionMode execution_mode, const std::vector<FieldSnapshot>* legacy_snapshots )
{
  synchronizeDevice( execution_mode );
  MPI_Barrier( MPI_COMM_WORLD );
  const double setup_start = MPI_Wtime();

  auto problem = makeProblem( problem_kind, options.refinement_levels );
  mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, std::move( problem.mesh ) );
  const int spatial_dimension = mesh.SpaceDimension();
  mfem::H1_FECollection field_collection( options.order, spatial_dimension );
  mfem::ParFiniteElementSpace field_space( &mesh, &field_collection, spatial_dimension, mfem::Ordering::byVDIM );

  mfem::ParGridFunction coordinates( &field_space );
  mesh.SetNodalGridFunction( &coordinates );
  mfem::ParGridFunction reference_coordinates( coordinates );
  mfem::ParGridFunction displacement( &field_space );
  displacement = 0.0;
  mfem::ParGridFunction velocity( &field_space );
  velocity = 0.0;
  setInitialVelocity( mesh, problem.moving_element_attribute, options.initial_velocity, velocity );

  const auto essential_velocity_dofs = essentialVelocityDofs( field_space, problem.fixed_attributes );
  mfem::ConstantCoefficient density_coefficient( options.density );
  mfem::ConstantCoefficient lambda_coefficient( options.lambda );
  mfem::ConstantCoefficient mu_coefficient( options.mu );
  mfem_ext::ExplicitMechanics mechanics( field_space, density_coefficient, lambda_coefficient, mu_coefficient );
  mfem_ext::CentralDiffSolver solver( essential_velocity_dofs );
  solver.Init( mechanics );

  constexpr tribol::IndexT coupling_scheme_id = 0;
  constexpr tribol::IndexT lower_mesh_id = 0;
  constexpr tribol::IndexT upper_mesh_id = 1;
  tribol::registerMfemCouplingScheme( coupling_scheme_id, lower_mesh_id, upper_mesh_id, mesh, coordinates,
                                      problem.lower_contact_attributes, problem.upper_contact_attributes,
                                      tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                      tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_BVH, execution_mode );
  tribol::setMPIComm( coupling_scheme_id, MPI_COMM_WORLD );
  tribol::registerMfemVelocity( coupling_scheme_id, velocity );
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, options.penalty, options.penalty );
  if ( implementation == CommonPlaneImplementation::ContactFormulation ) {
    constexpr bool enable_formulation = true;
    constexpr bool compute_jacobian = false;
    tribol::setMfemUseCommonPlaneFormulation( coupling_scheme_id, enable_formulation, compute_jacobian );
    constexpr int one_point_quadrature_order = 1;
    tribol::setQuadratureOrder( coupling_scheme_id, one_point_quadrature_order );
  }

  synchronizeDevice( execution_mode );
  RunResult result;
  result.timing.setup_seconds = globalMax( MPI_Wtime() - setup_start );

  mfem::Vector zero_vector( spatial_dimension );
  zero_vector = 0.0;
  mfem::VectorConstantCoefficient zero_coefficient( zero_vector );

  std::unique_ptr<mfem::VisItDataCollection> visit_data;
  if ( !options.timing_only ) {
    const std::string collection_name = problem.name + "_" + implementationName( implementation );
    visit_data = std::make_unique<mfem::VisItDataCollection>( collection_name, &mesh );
    visit_data->SetPrefixPath( options.output_directory );
    visit_data->RegisterField( "position", &coordinates );
    visit_data->RegisterField( "displacement", &displacement );
    visit_data->RegisterField( "velocity", &velocity );
  }

  std::size_t snapshot_index = 0;
  const auto record_state = [&]( int cycle, RealT time ) {
    if ( options.timing_only ) {
      return;
    }
    saveVisitState( *visit_data, cycle, time, options.time_step );
    if ( legacy_snapshots == nullptr ) {
      appendSnapshot( cycle, time, displacement, velocity, zero_coefficient, result.snapshots );
    } else {
      SLIC_ERROR_ROOT_IF( snapshot_index >= legacy_snapshots->size(),
                          "The ContactFormulation run produced more output states than the legacy run." );
      appendComparison( cycle, time, displacement, velocity, ( *legacy_snapshots )[snapshot_index], zero_coefficient,
                        result.comparisons );
      ++snapshot_index;
    }
  };

  int cycle = 0;
  RealT time = 0.0;
  record_state( cycle, time );

  const auto update_contact = [&]( int current_cycle, RealT current_time, RealT step_size ) {
    tribol::updateMfemParallelDecomposition();
    const int update_error = tribol::update( current_cycle, current_time, step_size );
    SLIC_ERROR_ROOT_IF( update_error != 0, "Tribol failed to calculate the Common Plane response." );
  };
  const auto load_contact_force = [&] {
    mechanics.f_ext = 0.0;
    if ( implementation == CommonPlaneImplementation::Legacy ) {
      tribol::getMfemResponse( coupling_scheme_id, mechanics.f_ext );
    } else {
      auto contact_residual = tribol::getMfemContactResidual( coupling_scheme_id );
      field_space.GetProlongationMatrix()->Mult( contact_residual, mechanics.f_ext );
      // ContactFormulation returns the residual contribution, while ExplicitMechanics expects the physical force.
      mechanics.f_ext *= -1.0;
    }
  };

  // The first update creates formulation objects and initializes backend-specific kernels and transfer data. Warm one
  // complete step, then restore the initial fields so both timed runs start from the same state after initialization.
  update_contact( cycle, time, options.time_step );
  load_contact_force();
  mechanics.SetTime( time );
  RealT warmup_time = time;
  RealT warmup_step_size = options.time_step;
  solver.Step( displacement, velocity, warmup_time, warmup_step_size );
  coordinates.Set( 1.0, reference_coordinates );
  coordinates += displacement;
  synchronizeDevice( execution_mode );

  displacement = 0.0;
  velocity = 0.0;
  setInitialVelocity( mesh, problem.moving_element_attribute, options.initial_velocity, velocity );
  coordinates.Set( 1.0, reference_coordinates );
  solver.Init( mechanics );
  mechanics.f_ext = 0.0;
  synchronizeDevice( execution_mode );
  MPI_Barrier( MPI_COMM_WORLD );

  constexpr RealT time_tolerance_multiplier = 100.0;
  const RealT time_tolerance = time_tolerance_multiplier * std::numeric_limits<RealT>::epsilon() *
                               std::max( { 1.0, std::abs( options.end_time ), std::abs( options.time_step ) } );
  while ( time < options.end_time - time_tolerance ) {
    RealT step_size = std::min( options.time_step, options.end_time - time );

    synchronizeDevice( execution_mode );
    const double integration_start = MPI_Wtime();
    const double contact_start = MPI_Wtime();
    const double update_start = MPI_Wtime();
    update_contact( cycle, time, step_size );
    synchronizeDevice( execution_mode );
    result.timing.contact_update_seconds += MPI_Wtime() - update_start;

    const double response_start = MPI_Wtime();
    load_contact_force();
    synchronizeDevice( execution_mode );
    result.timing.response_seconds += MPI_Wtime() - response_start;
    result.timing.contact_seconds += MPI_Wtime() - contact_start;

    mechanics.SetTime( time );
    solver.Step( displacement, velocity, time, step_size );
    coordinates.Set( 1.0, reference_coordinates );
    coordinates += displacement;
    synchronizeDevice( execution_mode );
    result.timing.integration_seconds += MPI_Wtime() - integration_start;
    ++result.timing.timed_steps;

    time += step_size;
    if ( std::abs( options.end_time - time ) <= time_tolerance ) {
      time = options.end_time;
    }
    ++cycle;
    if ( cycle % options.output_cycles == 0 || time >= options.end_time ) {
      record_state( cycle, time );
    }
  }

  if ( legacy_snapshots != nullptr ) {
    SLIC_ERROR_ROOT_IF( snapshot_index != legacy_snapshots->size(),
                        "The ContactFormulation run produced fewer output states than the legacy run." );
  }

  result.timing.integration_seconds = globalMax( result.timing.integration_seconds );
  result.timing.contact_seconds = globalMax( result.timing.contact_seconds );
  result.timing.contact_update_seconds = globalMax( result.timing.contact_update_seconds );
  result.timing.response_seconds = globalMax( result.timing.response_seconds );
  tribol::finalize();
  return result;
}

void writeFieldComparisons( const std::string& output_directory, const std::string& problem_name,
                            const std::vector<FieldComparison>& comparisons, int rank )
{
  if ( rank != 0 ) {
    return;
  }
  const std::string path =
      axom::utilities::filesystem::joinPath( output_directory, problem_name + "_field_comparison.csv" );
  std::ofstream output( path );
  output << "cycle,time,legacy_displacement_l2,formulation_displacement_l2,displacement_difference_l2,"
            "relative_displacement_difference,legacy_velocity_l2,formulation_velocity_l2,velocity_difference_l2,"
            "relative_velocity_difference\n";
  output << std::setprecision( 16 );
  for ( const auto& comparison : comparisons ) {
    output << comparison.cycle << ',' << comparison.time << ',' << comparison.legacy_displacement_l2 << ','
           << comparison.formulation_displacement_l2 << ',' << comparison.displacement_difference_l2 << ','
           << comparison.relative_displacement_difference << ',' << comparison.legacy_velocity_l2 << ','
           << comparison.formulation_velocity_l2 << ',' << comparison.velocity_difference_l2 << ','
           << comparison.relative_velocity_difference << '\n';
  }
}

void writeTimingHeader( std::ofstream& output )
{
  output << "problem,timed_steps,legacy_setup_seconds,formulation_setup_seconds,legacy_integration_seconds,"
            "formulation_integration_seconds,legacy_contact_seconds,formulation_contact_seconds,"
            "legacy_contact_update_seconds,formulation_contact_update_seconds,legacy_response_seconds,"
            "formulation_response_seconds,"
            "legacy_contact_seconds_per_step,formulation_contact_seconds_per_step,"
            "legacy_over_formulation_integration_ratio,legacy_over_formulation_contact_ratio\n";
  output << std::setprecision( 16 );
}

double speedRatio( double legacy_seconds, double formulation_seconds )
{
  return formulation_seconds > 0.0 ? legacy_seconds / formulation_seconds : std::numeric_limits<double>::infinity();
}

void writeTimingRow( std::ostream& output, const std::string& problem_name, const RunTiming& legacy,
                     const RunTiming& formulation )
{
  SLIC_ERROR_ROOT_IF( legacy.timed_steps != formulation.timed_steps,
                      "Legacy and ContactFormulation runs used different numbers of timed steps." );
  const double legacy_contact_per_step = legacy.contact_seconds / legacy.timed_steps;
  const double formulation_contact_per_step = formulation.contact_seconds / formulation.timed_steps;
  output << problem_name << ',' << legacy.timed_steps << ',' << legacy.setup_seconds << ',' << formulation.setup_seconds
         << ',' << legacy.integration_seconds << ',' << formulation.integration_seconds << ',' << legacy.contact_seconds
         << ',' << formulation.contact_seconds << ',' << legacy.contact_update_seconds << ','
         << formulation.contact_update_seconds << ',' << legacy.response_seconds << ',' << formulation.response_seconds
         << ',' << legacy_contact_per_step << ',' << formulation_contact_per_step << ','
         << speedRatio( legacy.integration_seconds, formulation.integration_seconds ) << ','
         << speedRatio( legacy.contact_seconds, formulation.contact_seconds ) << '\n';
}

void printTiming( const std::string& problem_name, const RunTiming& legacy, const RunTiming& formulation, int rank )
{
  if ( rank != 0 ) {
    return;
  }
  SLIC_ERROR_ROOT_IF( legacy.timed_steps != formulation.timed_steps,
                      "Legacy and ContactFormulation runs used different numbers of timed steps." );
  const double legacy_contact_per_step = legacy.contact_seconds / legacy.timed_steps;
  const double formulation_contact_per_step = formulation.contact_seconds / formulation.timed_steps;
  SLIC_INFO_ROOT( axom::fmt::format(
      "{} steady-state runtime comparison over {} steps (maximum across ranks):\n"
      "  legacy setup:                 {:.6f} s\n"
      "  formulation setup:            {:.6f} s\n"
      "  legacy time integration:      {:.6f} s\n"
      "  formulation time integration: {:.6f} s\n"
      "  legacy contact work:          {:.6f} s\n"
      "  formulation contact work:     {:.6f} s\n"
      "  legacy contact update:        {:.6f} s\n"
      "  formulation contact update:   {:.6f} s\n"
      "  legacy response extraction:   {:.6f} s\n"
      "  formulation response extract: {:.6f} s\n"
      "  legacy contact per step:      {:.6f} s\n"
      "  formulation contact per step: {:.6f} s\n"
      "  legacy/formulation integrate: {:.3f}x\n"
      "  legacy/formulation contact:   {:.3f}x",
      problem_name, legacy.timed_steps, legacy.setup_seconds, formulation.setup_seconds, legacy.integration_seconds,
      formulation.integration_seconds, legacy.contact_seconds, formulation.contact_seconds,
      legacy.contact_update_seconds, formulation.contact_update_seconds, legacy.response_seconds,
      formulation.response_seconds, legacy_contact_per_step, formulation_contact_per_step,
      speedRatio( legacy.integration_seconds, formulation.integration_seconds ),
      speedRatio( legacy.contact_seconds, formulation.contact_seconds ) ) );
}

Options parseOptions( int argc, char** argv )
{
  Options options;
  axom::CLI::App app{ "Compare legacy and ContactFormulation-based Common Plane in explicit MFEM problems." };
  app.add_option( "--problem", options.problem, "Problem to run: all, blocks, or cylinder." )
      ->capture_default_str()
      ->check( axom::CLI::IsMember( { "all", "blocks", "cylinder" } ) );
  app.add_option( "-r,--refine", options.refinement_levels, "Number of uniform mesh refinements." )
      ->capture_default_str();
  app.add_option( "-o,--order", options.order, "Finite element polynomial order." )
      ->capture_default_str()
      ->check( axom::CLI::PositiveNumber );
  app.add_option( "-v,--initial-velocity", options.initial_velocity, "Downward velocity of the upper body." )
      ->capture_default_str()
      ->check( axom::CLI::PositiveNumber );
  app.add_option( "-D,--dt", options.time_step, "Fixed explicit time step." )
      ->capture_default_str()
      ->check( axom::CLI::PositiveNumber );
  app.add_option( "-e,--end-time", options.end_time, "Simulation end time." )
      ->capture_default_str()
      ->check( axom::CLI::PositiveNumber );
  app.add_option( "-c,--output-cycles", options.output_cycles, "Cycles between VisIt and L2 output." )
      ->capture_default_str()
      ->check( axom::CLI::PositiveNumber );
  app.add_option( "-R,--density", options.density, "Material density." )
      ->capture_default_str()
      ->check( axom::CLI::PositiveNumber );
  app.add_option( "-l,--lambda", options.lambda, "First Lame parameter." )
      ->capture_default_str()
      ->check( axom::CLI::PositiveNumber );
  app.add_option( "-m,--mu", options.mu, "Shear modulus." )->capture_default_str()->check( axom::CLI::PositiveNumber );
  app.add_option( "-p,--penalty", options.penalty, "Constant kinematic penalty for both contact meshes." )
      ->capture_default_str()
      ->check( axom::CLI::PositiveNumber );
  app.add_option( "--output-directory", options.output_directory, "Directory for VisIt and CSV output." )
      ->capture_default_str();
  app.add_option( "-d,--device", options.device, "MFEM device configuration string." )->capture_default_str();
  app.add_flag( "--timing-only", options.timing_only, "Write runtime CSV data without VisIt or field comparisons." );
  try {
    app.parse( argc, argv );
  } catch ( const axom::CLI::ParseError& error ) {
    std::exit( app.exit( error ) );
  }
  SLIC_ERROR_ROOT_IF( options.refinement_levels < 0, "The number of refinement levels cannot be negative." );
  return options;
}

}  // namespace

int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  int rank = 0;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();
#endif

  axom::slic::SimpleLogger logger;
  axom::slic::setIsRoot( rank == 0 );
  const Options options = parseOptions( argc, argv );

  if ( rank == 0 && !axom::utilities::filesystem::pathExists( options.output_directory ) ) {
    axom::utilities::filesystem::makeDirsForPath( options.output_directory );
  }
  MPI_Barrier( MPI_COMM_WORLD );

  mfem::Device device( options.device );
  if ( rank == 0 ) {
    device.Print();
  }
  const auto execution_mode = executionMode( device );
  warmUpDevice( execution_mode );

  std::vector<ProblemKind> problems;
  if ( options.problem == "all" || options.problem == "blocks" ) {
    problems.push_back( ProblemKind::Blocks );
  }
  if ( options.problem == "all" || options.problem == "cylinder" ) {
    problems.push_back( ProblemKind::HalfCylinder );
  }

  std::ofstream timing_output;
  if ( rank == 0 ) {
    const std::string timing_path =
        axom::utilities::filesystem::joinPath( options.output_directory, "runtime_comparison.csv" );
    timing_output.open( timing_path );
    writeTimingHeader( timing_output );
  }

  for ( ProblemKind problem_kind : problems ) {
    const auto legacy = runProblem( problem_kind, CommonPlaneImplementation::Legacy, options, execution_mode, nullptr );
    const auto formulation = runProblem( problem_kind, CommonPlaneImplementation::ContactFormulation, options,
                                         execution_mode, &legacy.snapshots );
    const std::string problem_name = problem_kind == ProblemKind::Blocks ? "blocks" : "half_cylinder";
    if ( !options.timing_only ) {
      writeFieldComparisons( options.output_directory, problem_name, formulation.comparisons, rank );
    }
    printTiming( problem_name, legacy.timing, formulation.timing, rank );
    if ( rank == 0 ) {
      writeTimingRow( timing_output, problem_name, legacy.timing, formulation.timing );
    }
  }

  MPI_Finalize();
  return 0;
}
