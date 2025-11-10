// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file mfem_mortar_lm_patch.cpp
 *
 * @brief Demonstrates contact patch test using the mortar method
 *
 * Demonstrates a three dimensional contact patch test using the mortar method in Tribol. Contact is enforced between
 * two blocks which are initially in contact. The blocks occupy [0, 1]^3 and [0, 1]x[0, 1]x[0.99, 1.99]. To enforce
 * symmetry and prevent rigid body modes, Dirichlet boundary conditions are applied in the x-direction along the x = 0
 * plane, in the y-direction along y = 0 plane, and in the z-direction along the z = 0 and z = 1.99 planes. Enforcement
 * is through Lagrange multipliers. Small deformation contact is assumed and, consequently, the system is linear and the
 * solution is determined through a single linear solve (no timestepping). The elasticity solution for this problem
 * predicts a constant pressure field on the contact surface and linearly varying pressures. Since these fields can be
 * exactly represented by the finite element space, we expect the solution to be exact to machine precision.
 *
 * The linear system solved is
 *  | A B^T | | d | = | f |
 *  | B 0   | | p |   | g | ,
 *
 * where A is the system matrix for elasticity, B is the constraint matrix for mortar contact, d is the vector of nodal
 * displacements, p is the vector of contact pressures, f is the vector of nodal forces, and g is the vector of gaps on
 * the contact surface.  MFEM block operators and vectors are used to store the linear system.
 *
 * The example uses the Tribol MFEM interface, which supports decomposed (MPI) meshes and will support higher order
 * meshes (through LOR) in a future update (pending implementation of transfer of Jacobian from LOR to HO). Comments in
 * the main function below give details on each step of the example code.
 *
 * Example runs (from repo root directory):
 *   - mpirun -np 4 {build_dir}/examples/mfem_mortar_lm_patch_ex
 *   - mpirun -np 4 {build_dir}/examples/mfem_mortar_lm_patch_ex -r 4 -d cuda
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
  // device configuration string (see mfem::Device::Configure() for valid options)
  std::string device_config = "cpu";
  // Should we mesh with tet elements?
  bool use_tets = false;

  // parse command line options
  axom::CLI::App app{ "mfem_mortar_lm_patch" };
  app.add_option( "-r,--refine", ref_levels, "Number of times to refine the mesh uniformly." )->capture_default_str();
  // TODO: LOR support for implicit contact
  // app.add_option("-o,--order", order,
  //   "Finite element order (polynomial degree).")
  //   ->capture_default_str();
  app.add_option( "-l,--lambda", lambda, "Lame parameter lambda." )->capture_default_str();
  app.add_option( "-m,--mu", mu, "Lame parameter mu (shear modulus)." )->capture_default_str();
  // TODO: GPU support for implicit mortar
  // app.add_option( "-d,--device", device_config,
  //                 "Device configuration string, see mfem::Device::Configure() for valid options." )
  //     ->capture_default_str();
  app.add_option( "-t,--use-tets", use_tets, "Should we use tetrahedral elements?" )->capture_default_str();
  CLI11_PARSE( app, argc, argv );

  SLIC_INFO_ROOT( "Running mfem_mortar_lm_patch with the following options:" );
  SLIC_INFO_ROOT( axom::fmt::format( "refine:   {0}", ref_levels ) );
  SLIC_INFO_ROOT( axom::fmt::format( "lambda:   {0}", lambda ) );
  SLIC_INFO_ROOT( axom::fmt::format( "mu:       {0}", mu ) );
  SLIC_INFO_ROOT( axom::fmt::format( "use_tets: {0}\n", use_tets ) );

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

  // This block of code will read the mesh data given in two_hex_overlap.mesh, create an mfem::Mesh, refine the mesh,
  // then create an mfem::ParMesh. Optionally, the mfem::ParMesh can be refined further on each rank by setting
  // par_ref_levels >= 1, though this is disabled below.
  timer.start();
  // build mesh of 2 cubes
  int nel_per_dir = std::pow( 2, ref_levels );
  auto elem_type = use_tets ? mfem::Element::TETRAHEDRON : mfem::Element::HEXAHEDRON;
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
  mfem::ParaViewDataCollection paraview_datacoll( "mortar_patch_pv", &mesh );
  mfem::VisItDataCollection visit_datacoll( "mortar_patch_vi", &mesh );

  // This block of code creates position and displacement grid functions (and associated finite element collections and
  // finite element spaces) on the mesh. The displacement grid function is initialized to zero. The position grid
  // function is initialized with the nodal coordinates. These grid functions are registered with the data collections
  // for output.
  timer.start();
  // Finite element collection (shared between all grid functions).
  mfem::H1_FECollection fec( order, mesh.SpaceDimension() );
  // Finite element space (shared between all grid functions).
  mfem::ParFiniteElementSpace fespace( &mesh, &fec, mesh.SpaceDimension() );
  // Create coordinate grid function
  mfem::ParGridFunction coords( &fespace );
  // Set coordinate grid function based on nodal locations. In MFEM, nodal locations of higher order meshes are stored
  // in a grid function. For linear MFEM meshes, nodal locations can be stored in a grid function or through the vertex
  // coordinates. For consistency, we will create a nodal grid function even for linear meshes.
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

  // This block of code builds a list of degrees of freedom to which homogeneous displacement boundary conditions will
  // be applied. These boundary conditions enforce problem symmetry and prevent rigid body deformation modes. The
  // boundary attribute sets for each direction are identified in the fixed options above.
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
  // Next, we create a Tribol coupling scheme between the contact surfaces on the MFEM mesh. To create the coupling
  // scheme requires several steps: 1) building a boundary submesh, 2) building a LOR mesh (if required), 3)
  // re-decomposing the domain to move spatially close surface element pairs on to the same rank, 4) creating Tribol
  // meshes of each surface, and 5) registering the meshes and coupling scheme with Tribol. These 5 steps are performed
  // by calling two methods: 1) registerMfemCouplingScheme() (steps 1 and 2) and 2) updateMfemParallelDecomposition()
  // (steps 3, 4, and 5). registerMfemCouplingScheme() is called here and updateMfemParallelDecomposition() is typically
  // called before calling update(). Note, the mortar method only supports computation on host, so the ExecutionMode is
  // set to Sequential.
  int coupling_scheme_id = 0;
  int mesh1_id = 0;
  int mesh2_id = 1;
  tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, mesh, coords, mortar_attrs,
                                      nonmortar_attrs, tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING,
                                      tribol::SINGLE_MORTAR, tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER,
                                      tribol::BINNING_GRID, tribol::ExecutionMode::Sequential );
  tribol::setMPIComm( coupling_scheme_id, MPI_COMM_WORLD );
  // The basic Lagrange multiplier options are set here. For this problem, we ask Tribol to compute a contact residual
  // and a Jacobian (though we only use the Jacobian).
  tribol::setLagrangeMultiplierOptions( coupling_scheme_id, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN );

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

  // HypreParMatrices holding the Jacobian from contact and stored in an MFEM block operator are returned from this API
  // call.
  //
  // NOTE: The submesh contains both the mortar and nonmortar surfaces, but pressure DOFs are only present on the
  // nonmortar surface. The pressure DOFs on the mortar surface are eliminated in the returned matrix with ones on the
  // diagonal.
  auto A = tribol::getMfemBlockJacobian( coupling_scheme_id );
  // Add the Jacobian from the elasticity bilinear form to the top left block
  A->SetBlock( 0, 0, A_elasticity.release() );
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to setup Tribol and compute Jacobian: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  auto& pressure = tribol::getMfemPressure( coupling_scheme_id );

  int n_global_disp_dofs = fespace.GlobalTrueVSize();
  int n_global_lm_dofs = pressure.ParFESpace()->GlobalTrueVSize();
  int n_disp_dofs = fespace.GetTrueVSize();
  int n_lm_dofs = pressure.ParFESpace()->GetTrueVSize();

  SLIC_INFO_ROOT( axom::fmt::format( "  Number of displacement DOFs:        {0}", n_global_disp_dofs ) );
  SLIC_INFO_ROOT( axom::fmt::format( "  Number of Lagrange multiplier DOFs: {0}", n_global_lm_dofs ) );

  timer.start();
  // Create a RHS vector storing forces and gaps. Note no external forces are present in this problem.
  mfem::Vector B( A->Height() );
  // This tells MFEM to use the device for the vector operations (if a device is configured, defaults to CPU).
  B.UseDevice( true );
  B = 0.0;
  // This API call returns the mortar nodal gap vector to an (uninitialized) vector. The function sizes and initializes
  // the vector.
  mfem::Vector gap;
  gap.UseDevice( true );
  tribol::getMfemGap( coupling_scheme_id, gap );
  mfem::Vector gap_true( B, n_disp_dofs, n_lm_dofs );
  // gap is a dual vector, so (gap tdof vector) = P^T * (gap ldof vector)
  auto& P_submesh = *pressure.ParFESpace()->GetProlongationMatrix();
  P_submesh.MultTranspose( gap, gap_true );
  B.SyncMemory( gap_true );

  // Create a solution vector storing displacement and pressures.
  mfem::Vector X( A->Width() );
  X.UseDevice( true );
  X = 0.0;

  // Create a single HypreParMatrix from blocks for the solver. This process requires two steps: (1) store pointer to
  // the BlockOperator HypreParMatrixs in a 2D array and (2) call mfem::HypreParMatrixFromBlocks() to create the merged,
  // single HypreParMatrix (without blocks).
  mfem::Array2D<const mfem::HypreParMatrix*> hypre_blocks( 2, 2 );
  for ( int i{ 0 }; i < 2; ++i ) {
    for ( int j{ 0 }; j < 2; ++j ) {
      if ( A->GetBlock( i, j ).Height() != 0 && A->GetBlock( i, j ).Width() != 0 ) {
        hypre_blocks( i, j ) =
            const_cast<mfem::HypreParMatrix*>( dynamic_cast<const mfem::HypreParMatrix*>( &A->GetBlock( i, j ) ) );
      } else {
        hypre_blocks( i, j ) = nullptr;
      }
    }
  }
  auto A_merged = std::unique_ptr<mfem::HypreParMatrix>( mfem::HypreParMatrixFromBlocks( hypre_blocks ) );
  // Use a linear solver to find the block displacement/pressure vector.
  mfem::MINRESSolver solver( MPI_COMM_WORLD );
  solver.SetRelTol( 1.0e-8 );
  solver.SetAbsTol( 1.0e-12 );
  solver.SetMaxIter( 5000 );
  solver.SetPrintLevel( 3 );
  solver.SetOperator( *A_merged );
  // TODO: find a preconditioner
  // solver.SetPreconditioner( prec );
  solver.Mult( B, X );

  // Move the block displacements to the displacement grid function.
  mfem::Vector displacement_true( X, 0, n_disp_dofs );
  fespace.GetProlongationMatrix()->Mult( displacement_true, displacement );
  // Fix the sign on the displacements.
  displacement.Neg();
  // Update mesh coordinates given the displacement.
  coords += displacement;

  // Update the pressure degrees of freedom
  mfem::Vector pressure_true( X, n_disp_dofs, n_lm_dofs );
  P_submesh.Mult( pressure_true, pressure );

  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to solve for updated displacements: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  // Verify the forces are in equilibrium, i.e. f_int = A*u = f_contact = B^T*p.
  // This should be true if the solver converges.
  mfem::Vector f_int_true( fespace.GetTrueVSize() );
  f_int_true = 0.0;
  mfem::Vector f_contact_true( f_int_true );
  A->GetBlock( 0, 0 ).Mult( displacement_true, f_int_true );
  A->GetBlock( 0, 1 ).Mult( pressure_true, f_contact_true );
  mfem::Vector resid_true( f_int_true );
  resid_true += f_contact_true;
  for ( int i{ 0 }; i < ess_tdof_list.Size(); ++i ) {
    resid_true[ess_tdof_list[i]] = 0.0;
  }
  auto resid_linf = resid_true.Normlinf();
  if ( rank == 0 ) {
    MPI_Reduce( MPI_IN_PLACE, &resid_linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
    SLIC_INFO( axom::fmt::format( "|| force residual ||_(infty) = {0:e}", resid_linf ) );
  } else {
    MPI_Reduce( &resid_linf, &resid_linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
  }

  // Verify the gap is closed by the displacements, i.e. B*u = gap.
  // This should be true if the solver converges.
  mfem::Vector gap_resid_true( gap_true.Size() );
  gap_resid_true = 0.0;
  A->GetBlock( 1, 0 ).Mult( displacement_true, gap_resid_true );
  gap_resid_true -= gap_true;
  auto gap_resid_linf = gap_resid_true.Normlinf();
  if ( rank == 0 ) {
    MPI_Reduce( MPI_IN_PLACE, &gap_resid_linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
    SLIC_INFO( axom::fmt::format( "|| gap residual ||_(infty) = {0:e}", gap_resid_linf ) );
  } else {
    MPI_Reduce( &gap_resid_linf, &gap_resid_linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
  }

  // Save the deformed configuration
  paraview_datacoll.Save();
  visit_datacoll.Save();

  // Update the Tribol mesh based on deformed configuration
  tribol::updateMfemParallelDecomposition();

  // Create and save surface data
  mfem::ParaViewDataCollection paraview_surf_datacoll( "mortar_patch_surface_pv", pressure.ParFESpace()->GetMesh() );
  mfem::VisItDataCollection visit_surf_datacoll( "mortar_patch_surface_vi", pressure.ParFESpace()->GetMesh() );
  paraview_surf_datacoll.RegisterField( "pressure", &pressure );
  visit_surf_datacoll.RegisterField( "pressure", &pressure );
  paraview_surf_datacoll.Save();
  visit_surf_datacoll.Save();

  // Tribol cleanup: deletes the coupling schemes and clears associated memory.
  tribol::finalize();
  MPI_Finalize();

  return 0;
}
