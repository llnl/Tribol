// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file mfem_common_plane.cpp
 *
 * @brief Demonstrates explicit contact using Tribol's common plane algorithm
 *
 * Demonstrates contact in an explicit finite element code through a simple two block impact problem in three
 * dimensions. The first block occupies [0, 1]^3 and the second block occupies [0, 1]x[0, 1]x[1.01, 2.01]. An initial
 * velocity is applied to the second block in the negative z-direction and fixed velocity boundary conditions are
 * applied to the first block at z = 0. Contact constraints are enforced using the common plane algorithm.
 *
 * The example uses the Tribol MFEM interface, which supports decomposed (MPI) meshes and has support for higher order
 * meshes using low-order refinement of higher-order geometry representations. Comments in the main function below give
 * details on each step of the example code. The example code can run on HIP/CUDA, where available.
 *
 * Example runs (from repo root directory):
 *   - mpirun -np 4 {build_dir}/examples/mfem_common_plane_ex
 *
 * Example output can be viewed in VisIt or ParaView.
 */

#include <set>

// Tribol includes
#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/utils/TestUtils.hpp"

// Shared includes
#include "shared/mesh/MeshBuilder.hpp"

// Redecomp includes
#include "redecomp/redecomp.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// MFEM includes
#include "mfem.hpp"

// Axom includes
#include "axom/CLI11.hpp"
#include "axom/slic.hpp"

int main( int argc, char** argv )
{
  // initialize MPI
  MPI_Init( &argc, &argv );
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

  // initialize logger
  axom::slic::SimpleLogger logger;
  axom::slic::setIsRoot( rank == 0 );

  // define command line options
  // number of times to uniformly refine the serial mesh before constructing the parallel mesh
  int ref_levels = 2;
  // polynomial order of the finite element discretization
  int order = 1;
  // initial separation between the two blocks
  double initial_sep = 0.01;
  // initial velocity to apply in the negative z-direction to the top block
  double init_velocity = 0.02;
  // timestep size (fixed)
  double dt = 0.001;
  // end time of the simulation
  double t_end = 2.0;
  // use const penalty (true) or element penalty (false)
  bool use_const_penalty = false;
  // number of cycles to skip before next output
  int output_cycles = 20;
  // material density
  double rho = 1000.0;
  // Lame parameter lambda
  double lambda = 100000.0;
  // Lame parameter mu (shear modulus)
  double mu = 100000.0;
  // kinematic penalty parameter (only for constant penalty)
  // the kinematic constant penalty is chosen here to match the kinematic element penalty
  double p_kine = ( lambda + 2.0 / 3.0 * mu ) / ( 1.0 / std::pow( 2.0, ref_levels ) );
  // device configuration string (see mfem::Device::Configure() for valid options). This example has been tested on
  // "cpu" and "cuda"
  std::string device_config = "cpu";

  // parse command line options
  axom::CLI::App app{ "mfem_common_plane" };
  app.add_option( "-r,--refine", ref_levels, "Number of times to refine the mesh uniformly." )->capture_default_str();
  app.add_option( "-o,--order", order, "Finite element order (polynomial degree)." )->capture_default_str();
  app.add_option( "-s,--initials", initial_sep, "Initial separation between the two blocks." )->capture_default_str();
  app.add_option( "-v,--initialv", init_velocity, "Initial velocity of the top block." )->capture_default_str();
  app.add_option( "-D,--dt", dt, "Timestep size (fixed)." )->capture_default_str();
  app.add_option( "-e,--endtime", t_end, "End time of the simulation." )->capture_default_str();
  app.add_flag( "-C,--use-constant-penalty", use_const_penalty,
                "Use a constant penalty parameter (element thickness-based penalty otherwise)" )
      ->capture_default_str();
  app.add_option( "-p,--kinematicpenalty", p_kine, "Kinematic penalty parameter (if --use-element-penalty is false)." )
      ->capture_default_str()
      ->needs( "-C" );
  app.add_option( "-c,--outputcycles", output_cycles, "Number of cycles to skip before next output." )
      ->capture_default_str()
      ->check( axom::CLI::PositiveNumber );
  app.add_option( "-R,--rho", rho, "Material density." )->capture_default_str();
  app.add_option( "-l,--lambda", lambda, "Lame parameter lambda." )->capture_default_str();
  app.add_option( "-m,--mu", mu, "Lame parameter mu (shear modulus)." )->capture_default_str();
  app.add_option( "-d,--device", device_config,
                  "Device configuration string, see mfem::Device::Configure() for valid options." )
      ->capture_default_str();
  CLI11_PARSE( app, argc, argv );

  SLIC_INFO_ROOT( "Running mfem_common_plane with the following options:" );
  SLIC_INFO_ROOT( axom::fmt::format( "refinement levels: {0}", ref_levels ) );
  SLIC_INFO_ROOT( axom::fmt::format( "polynomial order:  {0}", order ) );
  SLIC_INFO_ROOT( axom::fmt::format( "initial sep:       {0}", initial_sep ) );
  SLIC_INFO_ROOT( axom::fmt::format( "initial velocity:  {0}", init_velocity ) );
  SLIC_INFO_ROOT( axom::fmt::format( "timestep size:     {0}", dt ) );
  SLIC_INFO_ROOT( axom::fmt::format( "end time:          {0}", t_end ) );
  SLIC_INFO_ROOT( axom::fmt::format( "kinematic penalty: {0}", p_kine ) );
  SLIC_INFO_ROOT( axom::fmt::format( "output interval:   {0}", output_cycles ) );
  SLIC_INFO_ROOT( axom::fmt::format( "density:           {0}", rho ) );
  SLIC_INFO_ROOT( axom::fmt::format( "lambda:            {0}", lambda ) );
  SLIC_INFO_ROOT( axom::fmt::format( "mu:                {0}", mu ) );
  SLIC_INFO_ROOT( axom::fmt::format( "device:            {0}\n", device_config ) );

  // configure the devices available for MFEM kernel launches
  mfem::Device device( device_config );
  if ( rank == 0 ) {
    device.Print();
  }

  tribol::ExecutionMode exec_mode = tribol::ExecutionMode::Sequential;
#ifdef TRIBOL_USE_CUDA
  if ( device.Allows( mfem::Backend::CUDA_MASK ) ) {
    exec_mode = tribol::ExecutionMode::Cuda;
  }
#endif
#ifdef TRIBOL_USE_HIP
  if ( device.Allows( mfem::Backend::HIP_MASK ) ) {
    exec_mode = tribol::ExecutionMode::Hip;
  }
#endif

  // fixed options
  // boundary element attributes of contact surface 1, the z = 1 plane of the first block
  auto contact_surf_1 = std::set<int>( { 6 } );
  // boundary element attributes of contact surface 2, the z = 1.01 plane of the second block
  auto contact_surf_2 = std::set<int>( { 7 } );
  // boundary element attributes of fixed surface (z = 0 plane of the first block, all t)
  auto fixed_attrs = std::set<int>( { 1 } );
  // element attribute corresponding to volume elements where an initial velocity will be applied in the z-direction
  auto moving_attrs = std::set<int>( { 2 } );

  // create an axom timer to give wall times for each step
  axom::utilities::Timer timer{ false };

  // This block of code will create two mfem::Meshes of cubes, unify them into one mfem::Mesh, refine the mesh, then
  // create an mfem::ParMesh.
  timer.start();
  // clang-format off
  mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::Unify( {
    shared::MeshBuilder::CubeMesh( 1, 1, 1 ),
    shared::MeshBuilder::CubeMesh( 1, 1, 1 )
      .translate( { 0.0, 0.0, 1.0 + initial_sep } )
      .updateAttrib( 1, 2 )    // changes attribute in all volume elements from 1 to 2 so it doesn't clash with the 
                               // first (bottom) mesh
      .updateBdrAttrib( 1, 7 ) // set the bottom surface to boundary attribute 7
      .updateBdrAttrib( 6, 8 ) // set the top surface to boundary attribute 8
  } ).refine( ref_levels ) );
  // clang-format on
  timer.stop();
  SLIC_INFO_ROOT( axom::fmt::format( "Time to create parallel mesh: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  // Set up an MFEM data collection for output. We output data in Paraview and VisIt formats.
  mfem::ParaViewDataCollection paraview_datacoll( "common_plane_pv", &mesh );
  mfem::VisItDataCollection visit_datacoll( "common_plane_vi", &mesh );

  // This block of code creates position, displacement, and velocity grid functions (and associated finite element
  // collections and finite element spaces) on the mesh. The displacement and velocity grid functions are initialized to
  // zero. The position grid function is initialized with the nodal coordinates. These grid functions are registered
  // with the data collections for output.
  timer.start();
  // Finite element collection (shared between all grid functions).
  mfem::H1_FECollection fec( order, mesh.SpaceDimension() );
  // Finite element space (shared between all grid functions).
  mfem::ParFiniteElementSpace fespace( &mesh, &fec, mesh.SpaceDimension() );
  // Create coordinate grid function
  mfem::ParGridFunction coords( &fespace );
  // Set coordinate grid function based on nodal locations. In MFEM, nodal locations of higher order meshes are stored
  // in a grid function. For linear MFEM meshes, nodal locations can be stored in a grid function or through the vertex
  // coordinates. Since we track coordinates in a grid function even for linear meshes, it makes sense to create a nodes
  // grid function so we don't have to manually update the vertices.
  mesh.SetNodalGridFunction( &coords );
  paraview_datacoll.RegisterField( "position", &coords );
  visit_datacoll.RegisterField( "position", &coords );

  // Save reference coordinates
  mfem::ParGridFunction ref_coords( coords );

  // Create a grid function for displacement
  mfem::ParGridFunction displacement( &fespace );
  paraview_datacoll.RegisterField( "displacement", &displacement );
  visit_datacoll.RegisterField( "displacement", &displacement );
  displacement = 0.0;

  // Create a grid function for velocity
  mfem::ParGridFunction velocity( &fespace );
  paraview_datacoll.RegisterField( "velocity", &velocity );
  visit_datacoll.RegisterField( "velocity", &velocity );
  velocity = 0.0;
  timer.stop();
  SLIC_INFO_ROOT( axom::fmt::format( "Time to create grid functions: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  timer.start();
  {
    // This block of code sets initial conditions on the velocity grid function. We create a piecewise constant vector
    // coefficient with an initial velocity in the z-component of the second block and zero elsewhere. First, we create
    // a constant vector coefficient, then create arrays of the vector coefficient and the element attributes which
    // belong to the second block to build the piecewise constant vector coefficient. In the PWVectorCoefficient class,
    // zeros are assigned to missing attribute numbers. Finally, the velocity coefficient is projected onto the velocity
    // grid function.
    mfem::Vector init_velocity_vector( { 0.0, 0.0, -std::abs( init_velocity ) } );
    mfem::VectorConstantCoefficient init_velocity_coeff( init_velocity_vector );
    mfem::Array<int> moving_attrs_array;
    mfem::Array<mfem::VectorCoefficient*> init_velocity_coeff_array;
    moving_attrs_array.Reserve( moving_attrs.size() );
    init_velocity_coeff_array.Reserve( moving_attrs.size() );
    for ( auto moving_attr : moving_attrs ) {
      moving_attrs_array.Append( moving_attr );
      init_velocity_coeff_array.Append( &init_velocity_coeff );
    }
    mfem::PWVectorCoefficient initial_v_coeff( mesh.SpaceDimension(), moving_attrs_array, init_velocity_coeff_array );
    velocity.ProjectCoefficient( initial_v_coeff );
  }

  // This block of code builds a list of degrees of freedom in the x, y, and z directions in the first block along the z
  // = 0 plane. A homogeneous velocity boundary condition is applied to these degrees of freedom.
  mfem::Array<int> ess_vdof_list;
  {
    // First, build an array of "markers" (i.e. booleans) to denote which vdofs are in the list.
    mfem::Array<int> ess_vdof_marker;
    // Also, convert active boundary attributes into markers.
    mfem::Array<int> ess_bdr( mesh.bdr_attributes.Max() );
    ess_bdr = 0;
    for ( auto fixed_attr : fixed_attrs ) {
      ess_bdr[fixed_attr - 1] = 1;
    }
    // Note no component is given as an argument here, so all components (x, y, and z) are returned.
    fespace.GetEssentialVDofs( ess_bdr, ess_vdof_marker );
    // Convert the marker array to a list.
    mfem::FiniteElementSpace::MarkerToList( ess_vdof_marker, ess_vdof_list );
  }
  timer.stop();
  SLIC_INFO_ROOT( axom::fmt::format( "Time to set up boundary conditions: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  // This block of code builds a small-deformation elasticity explicit, lumped mass update operator. Lumping is
  // performed using row summation.
  timer.start();
  mfem::ConstantCoefficient rho_coeff( rho );
  mfem::ConstantCoefficient lambda_coeff( lambda );
  mfem::ConstantCoefficient mu_coeff( mu );
  mfem_ext::ExplicitMechanics op( fespace, rho_coeff, lambda_coeff, mu_coeff );
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to set up elasticity bilinear form: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  // This block of code sets up a central difference time integration scheme. The constructor takes a list of degrees of
  // freedom in the velocity grid function that will have homogeneous boundary conditions applied to them.
  mfem_ext::CentralDiffSolver solver( ess_vdof_list );
  solver.Init( op );

  // This block of code does initial setup of Tribol.
  timer.start();
  // First, we create a Tribol coupling scheme between the contact surfaces on the MFEM mesh. To create the coupling
  // scheme requires several steps: 1) building a boundary submesh, 2) building a LOR mesh (if required), 3)
  // re-decomposing the domain to move spatially close surface element pairs on to the same rank, and 4) creating a
  // Tribol mesh. Steps 1 and 2 are performed when this method is called. Steps 3 and 4 are accomplished via a call to
  // updateMfemParallelDecomposition(), which is typically called before calling update().
  int coupling_scheme_id = 0;
  int mesh1_id = 0;
  int mesh2_id = 1;
  tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, mesh, coords, contact_surf_1,
                                      contact_surf_2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                      tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_BVH, exec_mode );
  tribol::setMPIComm( coupling_scheme_id, MPI_COMM_WORLD );
  // This API call adds a velocity field to the coupling scheme. This is used for computing the maximum allowable common
  // plane timestep and, if activated, gap rate penalty.
  tribol::registerMfemVelocity( coupling_scheme_id, velocity );
  // The type of penalty enforcement and the penalty parameters are set here (i.e. rate vs. kinematic and the method of
  // setting the penalty value).
  if ( use_const_penalty ) {
    tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, p_kine, p_kine );
  } else {
    // Any MFEM scalar coefficient can be used to set material moduli, but this example uses a constant bulk modulus for
    // each boundary attribute of the volume mesh.
    mfem::Vector bulk_moduli_by_bdry_attrib( mesh.bdr_attributes.Max() );
    bulk_moduli_by_bdry_attrib = lambda + 2.0 / 3.0 * mu;
    mfem::PWConstCoefficient mat_coeff( bulk_moduli_by_bdry_attrib );
    tribol::setMfemKinematicElementPenalty( coupling_scheme_id, mat_coeff );
  }
  timer.stop();
  SLIC_INFO_ROOT( axom::fmt::format( "Time to set up Tribol: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  // This is the main timestepping loop.
  int cycle{ 0 };
  for ( double t{ 0.0 }; t < t_end; t += dt ) {
    // Update the cycle information for the data collections.
    paraview_datacoll.SetCycle( cycle );
    paraview_datacoll.SetTime( t );
    paraview_datacoll.SetTimeStep( dt );
    visit_datacoll.SetCycle( cycle );
    visit_datacoll.SetTime( t );
    visit_datacoll.SetTimeStep( dt );

    // Write output if we are at the right cycle.
    if ( cycle % output_cycles == 0 ) {
      paraview_datacoll.Save();
      visit_datacoll.Save();
    }

    // This updates the parallel adjacency-based mesh redecomposition based on the latest coordinates. It also
    // constructs new Tribol meshes as subdomains of the redecomposed mesh.
    tribol::updateMfemParallelDecomposition();
    // This API call computes the contact response given the current mesh configuration.
    tribol::update( cycle, t, dt );
    // Nodal forces are returned to an MFEM grid function using this API function. We set the contact nodal forces equal
    // to the external forces in the explicit mechanics update operator. Tribol returns the nodal forces on host.
    op.f_ext = 0.0;
    tribol::getMfemResponse( coupling_scheme_id, op.f_ext );

    // Update the time in the explicit mechanics update.
    op.SetTime( t );
    // Take a step in time in the time integration scheme. This update is done on device (note contact forces are copied
    // from host to device in this method).
    solver.Step( displacement, velocity, t, dt );

    // Update the coordinates based on the new displacement.
    coords.Set( 1.0, ref_coords );
    coords += displacement;

    // Increment the cycle count.
    ++cycle;
  }

  // save output after last timestep
  paraview_datacoll.Save();
  visit_datacoll.Save();

  // Tribol cleanup: deletes the coupling schemes and clears associated memory.
  tribol::finalize();
  MPI_Finalize();

  return 0;
}
