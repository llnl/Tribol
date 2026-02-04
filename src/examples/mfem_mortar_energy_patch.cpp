// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file mfem_mortar_energy_patch.cpp
 *
 * @brief Demonstrates contact patch test using the energy mortar method
 *
 * Demonstrates a three dimensional contact patch test using the energy mortar method in Tribol. Contact is enforced
 * between two blocks which are initially in contact. The blocks occupy [0, 1]^3 and [0, 1]x[0, 1]x[0.99, 1.99]. To
 * enforce symmetry and prevent rigid body modes, Dirichlet boundary conditions are applied in the x-direction along the
 * x = 0 plane, in the y-direction along y = 0 plane, and in the z-direction along the z = 0 and z = 1.99 planes.
 * Enforcement is through Penalty. Small deformation contact is assumed and, consequently, the system is linear and the
 * solution is determined through a single linear solve (no timestepping).
 *
 * The linear system solved is
 *  (K + K_contact) u = f_contact
 *
 * where K is the system matrix for elasticity, K_contact is the stiffness matrix from contact penalty,
 * u is the vector of nodal displacements, and f_contact is the vector of nodal contact forces.
 *
 * The example uses the Tribol MFEM interface, which supports decomposed (MPI) meshes.
 *
 * Example runs (from repo root directory):
 *   - mpirun -np 4 {build_dir}/examples/mfem_mortar_energy_patch_ex
 *
 * Example output can be viewed in VisIt or ParaView.
 */

#include <set>

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// MFEM includes
#include "mfem.hpp"

// Axom includes
#include "axom/CLI11.hpp"
#include "axom/core.hpp"
#include "axom/slic.hpp"

// Shared includes
#include "shared/mesh/MeshBuilder.hpp"

// Tribol includes
#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"

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
  // Lame parameter lambda
  double lambda = 50.0;
  // Lame parameter mu (shear modulus)
  double mu = 50.0;
  // Penalty parameter
  double penalty = 50.0;
  // device configuration string (see mfem::Device::Configure() for valid options)
  std::string device_config = "cpu";

  // parse command line options
  axom::CLI::App app{ "mfem_mortar_energy_patch" };
  app.add_option( "-r,--refine", ref_levels, "Number of times to refine the mesh uniformly." )->capture_default_str();
  app.add_option( "-l,--lambda", lambda, "Lame parameter lambda." )->capture_default_str();
  app.add_option( "-m,--mu", mu, "Lame parameter mu (shear modulus)." )->capture_default_str();
  app.add_option( "-p,--penalty", penalty, "Contact penalty parameter." )->capture_default_str();
  // app.add_option( "-d,--device", device_config, "Device configuration string." )->capture_default_str();

  CLI11_PARSE( app, argc, argv );

  SLIC_INFO_ROOT( "Running mfem_mortar_energy_patch with the following options:" );
  SLIC_INFO_ROOT( axom::fmt::format( "refine:   {0}", ref_levels ) );
  SLIC_INFO_ROOT( axom::fmt::format( "lambda:   {0}", lambda ) );
  SLIC_INFO_ROOT( axom::fmt::format( "mu:       {0}", mu ) );
  SLIC_INFO_ROOT( axom::fmt::format( "penalty:  {0}\n", penalty ) );

  // configure the devices available for MFEM kernel launches
  mfem::Device device( device_config );
  if ( rank == 0 ) {
    device.Print();
  }

  // fixed options
  // boundary element attributes of mortar surface, the z = 1 plane of the first block
  std::set<int> mortar_attrs( { 4 } );
  // boundary element attributes of nonmortar surface, the z = 0.99 plane of the second block
  std::set<int> nonmortar_attrs( { 5 } );
  // boundary element attributes of x-fixed surfaces (at x = 0)
  std::vector<std::set<int>> fixed_attrs( 3 );
  fixed_attrs[0] = { 1 };
  // boundary element attributes of y-fixed surfaces (at y = 0)
  fixed_attrs[1] = { 2 };
  // boundary element attributes of z-fixed surfaces (3: surface at z = 0, 6: surface at z = 1.99)
  fixed_attrs[2] = { 3, 6 };

  // create an axom timer to give wall times for each step
  axom::utilities::Timer timer{ false };

  timer.start();
  // build mesh of 2 cubes
  int nel_per_dir = std::pow( 2, ref_levels );
  auto elem_type = mfem::Element::HEXAHEDRON;
  // clang-format off
  mfem::ParMesh mesh = shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
    shared::MeshBuilder::CubeMesh(nel_per_dir, nel_per_dir, nel_per_dir, elem_type)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(1, 3)
      .updateBdrAttrib(4, 7)
      .updateBdrAttrib(5, 1)
      .updateBdrAttrib(6, 4),
    shared::MeshBuilder::CubeMesh(nel_per_dir, nel_per_dir, nel_per_dir, elem_type)
      .translate({0.0, 0.0, 0.99})
      .updateBdrAttrib(1, 8)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(4, 7)
      .updateBdrAttrib(5, 1)
      .updateBdrAttrib(8, 5)
  }));
  // clang-format on
  timer.stop();
  SLIC_INFO_ROOT( axom::fmt::format( "Time to create parallel mesh: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  // Set up an MFEM data collection for output. We output data in Paraview and
  // VisIt formats.
  mfem::ParaViewDataCollection paraview_datacoll( "mortar_energy_patch_pv", &mesh );
  mfem::VisItDataCollection visit_datacoll( "mortar_energy_patch_vi", &mesh );

  timer.start();
  // Finite element collection (shared between all grid functions).
  mfem::H1_FECollection fec( order, mesh.SpaceDimension() );
  // Finite element space (shared between all grid functions).
  mfem::ParFiniteElementSpace fespace( &mesh, &fec, mesh.SpaceDimension() );
  // Create coordinate grid function
  mfem::ParGridFunction coords( &fespace );
  mesh.SetNodalGridFunction( &coords );
  paraview_datacoll.RegisterField( "position", &coords );
  visit_datacoll.RegisterField( "position", &coords );

  // Create a grid function for displacement
  mfem::ParGridFunction displacement( &fespace );
  paraview_datacoll.RegisterField( "displacement", &displacement );
  visit_datacoll.RegisterField( "displacement", &displacement );
  displacement = 0.0;
  timer.stop();
  SLIC_INFO_ROOT( axom::fmt::format( "Time to create grid functions: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  // save initial configuration
  paraview_datacoll.Save();
  visit_datacoll.Save();

  timer.start();
  mfem::Array<int> ess_tdof_list;
  {
    // First, build an array of "markers" (i.e. booleans) to denote which vdofs are in the list.
    mfem::Array<int> ess_vdof_marker( fespace.GetVSize() );
    ess_vdof_marker = 0;
    for ( int d = 0; d < 3; ++d ) {
      // convert boundary attributes into markers for active attributes on the dimension d
      mfem::Array<int> ess_bdr( mesh.bdr_attributes.Max() );
      ess_bdr = 0;
      for ( auto xfixed_attr : fixed_attrs[d] ) {
        ess_bdr[xfixed_attr - 1] = 1;
      }
      mfem::Array<int> new_ess_vdof_marker;
      // Find all vdofs with the given boundary marker
      fespace.GetEssentialVDofs( ess_bdr, new_ess_vdof_marker, d );
      // Compute union of existing marked vdofs with vdofs marked on dimension d
      for ( int j = 0; j < new_ess_vdof_marker.Size(); ++j ) {
        ess_vdof_marker[j] = ess_vdof_marker[j] || new_ess_vdof_marker[j];
      }
    }
    // Convert the vdofs to tdofs to remove duplicate values over ranks
    mfem::Array<int> ess_tdof_marker;
    fespace.GetRestrictionMatrix()->BooleanMult( ess_vdof_marker, ess_tdof_marker );
    // Convert the tdof marker array to a tdof list
    mfem::FiniteElementSpace::MarkerToList( ess_tdof_marker, ess_tdof_list );
  }
  timer.stop();
  SLIC_INFO_ROOT( axom::fmt::format( "Time to set up boundary conditions: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  // This block of code constructs a small-deformation linear elastic bilinear form.
  timer.start();
  mfem::ParBilinearForm a( &fespace );
  mfem::ConstantCoefficient lambda_coeff( lambda );
  mfem::ConstantCoefficient mu_coeff( mu );
  a.AddDomainIntegrator( new mfem::ElasticityIntegrator( lambda_coeff, mu_coeff ) );

  // Assemble the on-rank bilinear form stiffness matrix.
  a.Assemble();
  // Reduce to tdofs and form a hypre parallel matrix for parallel solution of the linear system.
  auto A_elasticity = std::make_unique<mfem::HypreParMatrix>();
  a.FormSystemMatrix( ess_tdof_list, *A_elasticity );
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to create and assemble internal stiffness: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  // This block of code does initial setup of Tribol.
  timer.start();

  int coupling_scheme_id = 0;
  int mesh1_id = 0;
  int mesh2_id = 1;
  tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, mesh, coords, mortar_attrs,
                                      nonmortar_attrs, tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING,
                                      tribol::ENERGY_MORTAR, tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER,
                                      tribol::BINNING_GRID, tribol::ExecutionMode::Sequential );
  tribol::setMPIComm( coupling_scheme_id, MPI_COMM_WORLD );
  tribol::setLagrangeMultiplierOptions( coupling_scheme_id, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN );
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, penalty, penalty );

  // Update the cycle information for the data collections. Also update time with a pseudotime for the solution.
  int cycle = 1;
  double time = 1.0;  // time is arbitrary here (no timesteps)
  double dt = 1.0;
  paraview_datacoll.SetCycle( cycle );
  paraview_datacoll.SetTime( time );
  paraview_datacoll.SetTimeStep( dt );
  visit_datacoll.SetCycle( cycle );
  visit_datacoll.SetTime( time );
  visit_datacoll.SetTimeStep( dt );

  // This creates the parallel adjacency-based mesh redecomposition. It also constructs new Tribol meshes as subsets of
  // the redecomposed mesh.
  tribol::updateMfemParallelDecomposition();
  // This API call computes the contact response and Jacobian given the current mesh configuration.
  tribol::update( cycle, time, dt );

  // Get Contact Stiffness
  auto A_contact = tribol::getMfemDfDx( coupling_scheme_id );

  // Add contact stiffness to elasticity stiffness
  auto A_total = std::unique_ptr<mfem::HypreParMatrix>( mfem::Add( 1.0, *A_elasticity, 1.0, *A_contact ) );

  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to setup Tribol and compute Jacobian: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  int n_disp_dofs = fespace.GetTrueVSize();
  SLIC_INFO_ROOT( axom::fmt::format( "  Number of displacement DOFs:        {0}", n_disp_dofs ) );

  timer.start();

  // Retrieve contact force (response)
  mfem::Vector f_contact( fespace.GetTrueVSize() );
  f_contact = 0.0;
  tribol::getMfemResponse( coupling_scheme_id, f_contact );

  // Create a solution vector storing displacement
  mfem::Vector X( fespace.GetTrueVSize() );
  X.UseDevice( true );
  X = 0.0;

  // Use a linear solver to find the block displacement/pressure vector.
  mfem::MINRESSolver solver( MPI_COMM_WORLD );
  solver.SetRelTol( 1.0e-8 );
  solver.SetAbsTol( 1.0e-12 );
  solver.SetMaxIter( 5000 );
  solver.SetPrintLevel( 3 );
  solver.SetOperator( *A_total );
  solver.Mult( f_contact, X );

  // Move the block displacements to the displacement grid function.
  fespace.GetProlongationMatrix()->Mult( X, displacement );

  // Update mesh coordinates given the displacement.
  coords += displacement;

  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to solve for updated displacements: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  // Save the deformed configuration
  paraview_datacoll.Save();
  visit_datacoll.Save();

  // Tribol cleanup: deletes the coupling schemes and clears associated memory.
  tribol::finalize();
  MPI_Finalize();

  return 0;
}
