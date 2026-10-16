// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <string>

#include "axom/CLI11.hpp"
#include "axom/slic.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/interface/tribol.hpp"

#if __has_include("tribol/interface/MfemContact.hpp")
#define TRIBOL_BENCHMARK_OPERATOR_REDECOMP 1
#else
#define TRIBOL_BENCHMARK_OPERATOR_REDECOMP 0
#endif

namespace {

struct BenchmarkOptions {
  int elements_x{ 64 };
  int elements_y{ 4 };
  int order{ 2 };
  int refinement_factor{ 2 };
  int iterations{ 20 };
  int rebuild_iterations{ 3 };
  double motion_skin{ 0.2 };
};

double MaximumRankSeconds( double local_seconds )
{
  double maximum_seconds = 0.0;
  MPI_Allreduce( &local_seconds, &maximum_seconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  return maximum_seconds;
}

template <typename Operation>
double TimeCollectiveOperation( Operation&& operation )
{
  MPI_Barrier( MPI_COMM_WORLD );
  const double start = MPI_Wtime();
  operation();
  return MaximumRankSeconds( MPI_Wtime() - start );
}

mfem::ParMesh BuildMesh( const BenchmarkOptions& options )
{
  return shared::ParMeshBuilder(
      MPI_COMM_WORLD,
      shared::MeshBuilder::Unify(
          { shared::MeshBuilder::CubeMesh( options.elements_x, 1, options.elements_y ),
            shared::MeshBuilder::CubeMesh( options.elements_x, 1, options.elements_y )
                .translate( { 0.0, 0.0, 0.95 } )
                .updateBdrAttrib( 1, 7 )
                .updateBdrAttrib( 6, 8 ) } ) );
}

void ProjectCoordinates( mfem::ParGridFunction& coordinates )
{
  mfem::VectorFunctionCoefficient identity(
      3, []( const mfem::Vector& point, mfem::Vector& value ) { value = point; } );
  coordinates.ProjectCoefficient( identity );
}

#if TRIBOL_BENCHMARK_OPERATOR_REDECOMP

constexpr const char* implementation_name = "operator";

void RegisterContact( mfem::ParMesh& mesh, mfem::ParGridFunction& coordinates, const BenchmarkOptions& options )
{
  tribol::MfemContactOptions contact_options;
  contact_options.execution_mode = tribol::ExecutionMode::Sequential;
  contact_options.refinement_factor = options.refinement_factor;
  contact_options.motion_skin = options.motion_skin;
  tribol::registerMfemCouplingScheme(
      0, tribol::MfemContactSurface( 0, mesh, coordinates, { 7 } ),
      tribol::MfemContactSurface( 1, mesh, coordinates, { 6 } ), tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
      tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, contact_options );
  tribol::setMfemKinematicConstantPenalty( 0, 100.0, 100.0 );
}

double UpdateContact( int cycle, bool force_rebuild )
{
  if ( force_rebuild ) tribol::forceMfemContactPatchRebuildOnNextUpdate( 0 );
  tribol::RealT time_step = 1.0;
  const int error = tribol::update( cycle, 0.0, time_step );
  SLIC_ERROR_ROOT_IF( error != 0, "The operator redecomp benchmark update failed." );
  double local_checksum = 0.0;
  for ( const auto& contact_residual : tribol::getMfemContactResiduals( 0 ) ) {
    local_checksum += contact_residual.true_dof_values * contact_residual.true_dof_values;
  }
  double global_checksum = 0.0;
  MPI_Allreduce( &local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  return std::sqrt( global_checksum );
}

#else

constexpr const char* implementation_name = "legacy";
mfem::Vector legacy_response;

void RegisterContact( mfem::ParMesh& mesh, mfem::ParGridFunction& coordinates, const BenchmarkOptions& options )
{
  tribol::registerMfemCouplingScheme( 0, 0, 1, mesh, coordinates, std::set<int>{ 7 }, std::set<int>{ 6 },
                                      tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                      tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_BVH,
                                      tribol::ExecutionMode::Sequential );
  tribol::setMPIComm( 0, MPI_COMM_WORLD );
  if ( options.refinement_factor > 1 ) tribol::setMfemLORFactor( 0, options.refinement_factor );
  tribol::setMfemRedecompTriggerDisplacement( 0, options.motion_skin );
  tribol::setMfemKinematicConstantPenalty( 0, 100.0, 100.0 );
  legacy_response.SetSize( coordinates.Size() );
  legacy_response = 0.0;
}

double UpdateContact( int cycle, bool force_rebuild )
{
  tribol::updateMfemParallelDecomposition( 0, force_rebuild );
  tribol::RealT time_step = 1.0;
  const int error = tribol::update( cycle, 0.0, time_step );
  SLIC_ERROR_ROOT_IF( error != 0, "The legacy redecomp benchmark update failed." );
  legacy_response = 0.0;
  tribol::getMfemResponse( 0, legacy_response );
  double local_checksum = legacy_response * legacy_response;
  double global_checksum = 0.0;
  MPI_Allreduce( &local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  return std::sqrt( global_checksum );
}

#endif

}  // namespace

int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  int rank = 0;
  int num_ranks = 1;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &num_ranks );
  axom::slic::SimpleLogger logger;
  axom::slic::setIsRoot( rank == 0 );

  BenchmarkOptions options;
  axom::CLI::App app{ "Compare legacy and operator-based redecomp contact updates" };
  app.add_option( "--nx", options.elements_x, "Elements per body in the tangential direction" );
  app.add_option( "--ny", options.elements_y, "Elements per body in the normal direction" );
  app.add_option( "--order", options.order, "Parent coordinate-space order" );
  app.add_option( "--refinement-factor", options.refinement_factor, "Linear child facets per parent direction" );
  app.add_option( "--iterations", options.iterations, "Cached update iterations" );
  app.add_option( "--rebuild-iterations", options.rebuild_iterations, "Forced topology rebuild iterations" );
  app.add_option( "--motion-skin", options.motion_skin, "Physical motion threshold for topology reuse" );
  CLI11_PARSE( app, argc, argv );

  SLIC_ERROR_ROOT_IF( options.elements_x <= 0 || options.elements_y <= 0, "Mesh dimensions must be positive." );
  SLIC_ERROR_ROOT_IF( options.order <= 0 || options.refinement_factor <= 0,
                      "Order and refinement factor must be positive." );
  SLIC_ERROR_ROOT_IF( options.iterations <= 0 || options.rebuild_iterations <= 0,
                      "Benchmark iteration counts must be positive." );

  auto mesh = BuildMesh( options );
  mfem::H1_FECollection coordinate_collection( options.order, 3 );
  mfem::ParFiniteElementSpace coordinate_space( &mesh, &coordinate_collection, 3 );
  mfem::ParGridFunction coordinates( &coordinate_space );
  ProjectCoordinates( coordinates );

  const double registration_seconds = TimeCollectiveOperation( [&] { RegisterContact( mesh, coordinates, options ); } );
  double checksum = 0.0;
  const double first_update_seconds = TimeCollectiveOperation( [&] { checksum = UpdateContact( 0, false ); } );

  double steady_seconds = 0.0;
  for ( int iteration = 0; iteration < options.iterations; ++iteration ) {
    steady_seconds +=
        TimeCollectiveOperation( [&] { checksum += UpdateContact( iteration + 1, false ); } );
  }
  steady_seconds /= options.iterations;

  double rebuild_seconds = 0.0;
  for ( int iteration = 0; iteration < options.rebuild_iterations; ++iteration ) {
    rebuild_seconds += TimeCollectiveOperation(
        [&] { checksum += UpdateContact( options.iterations + iteration + 1, true ); } );
  }
  rebuild_seconds /= options.rebuild_iterations;

  SLIC_ERROR_ROOT_IF( !std::isfinite( checksum ) || checksum == 0.0,
                      "The benchmark produced an empty or invalid contact response." );
  if ( rank == 0 ) {
    std::cout << "implementation,ranks,elements_x,elements_y,order,refinement_factor,registration_seconds,"
                 "first_update_seconds,steady_seconds,rebuild_seconds,checksum\n";
    std::cout << implementation_name << ',' << num_ranks << ',' << options.elements_x << ',' << options.elements_y
              << ',' << options.order << ',' << options.refinement_factor << ',' << registration_seconds << ','
              << first_update_seconds << ',' << steady_seconds << ',' << rebuild_seconds << ',' << checksum << '\n';
  }

  tribol::finalize();
  MPI_Finalize();
  return 0;
}
