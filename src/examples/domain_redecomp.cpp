// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file domain_redecomp.cpp
 *
 * @brief Demonstrates redecomp::RedecompMesh and redecomp transfer capability
 *
 * Demonstrates use of redecomp::RedecompMesh to build a RCB-based redecomposition of an mfem::ParMesh, then transfers
 * an mfem::ParGridFunction and an mfem::QuadratureFunction to an mfem::GridFunction and an mfem::QuadratureFunction on
 * the redecomp::RedecompMesh and vice versa.
 *
 * Though redecomp::RedecompMesh contains pieces of the mesh spread across ranks, it is a serial mesh which derives from
 * mfem::Mesh. All coordination with other ranks must be done by transferring data to the parent mfem::ParMesh level.
 * Beyond the decomposition method (RCB vs. typically metis k-way), a redecomp::RedecompMesh differs from an
 * mfem::ParMesh because it includes a layer of ghost elements at the on-rank domain boundaries. Quantities on ghost
 * elements are typically transferred to fields on the redecomp::RedecompMesh, but are not transferred back to the
 * mfem::ParMesh.
 *
 * This example illustrates the envisioned typical workflow for using redecomp::RedecompMesh:
 *   1. Create a RedecompMesh by providing a parent ParMesh (and optionally a method of repartitioning the domain)
 *   2. Create fields on the RedecompMesh (mfem::GridFunction and/or mfem::QuadratureFunction)
 *   3. Creating a RedecompTransfer object which provides methods to transfer an mfem::ParGridFunction and an
 *      mfem::QuadratureFunction from the parent ParMesh to the RedecompMesh and vice versa
 *   4. Transferring the mfem::ParGridfunction and/or mfem::QuadratureFunction on the parent ParMesh to the
 *      mfem::GridFunction and/or mfem::QuadratureFunction on the RedecompMesh
 *   5. Performing an update on the mfem::GridFunction and/or mfem::QuadratureFunction at the RedecompMesh level
 *   6. Transferring the mfem::GridFunction and/or mfem::QuadratureFunction on the RedecompMesh back to the
 *      corresponding functions on the ParMesh
 *
 * Example runs (from repo root directory):
 *   - mpirun -np 4 {build_dir}/examples/domain_redecomp_ex -r 1 -m data/two_hex.mesh
 *   - mpirun -np 4 {build_dir}/examples/domain_redecomp_ex -r 1 -m data/star.mesh
 *   - mpirun -np 4 {build_dir}/examples/domain_redecomp_ex -r 1 -o 2 -m data/two_hex.mesh
 *   - mpirun -np 4 {build_dir}/examples/domain_redecomp_ex -r 1 -o 2 -m data/star.mesh
 *
 * Example output on both the RedecompMesh and the ParMesh can be viewed in VisIt.
 */

#include <string>
#include <vector>

#include <mpi.h>

#include "axom/CLI11.hpp"
#include "axom/slic.hpp"

#include "mfem.hpp"

#include "redecomp/redecomp.hpp"

#include "tribol/config.hpp"

template <int NDIMS>
void RedecompExample( mfem::ParMesh& pmesh, int order, double max_out_of_balance );

int main( int argc, char** argv )
{
  /////////////////////////////////////////////////////////////////////////////
  // STEP 0: Problem setup
  /////////////////////////////////////////////////////////////////////////////

  // initialize MPI
  MPI_Init( &argc, &argv );
  int np, rank;
  MPI_Comm_size( MPI_COMM_WORLD, &np );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  // initialize logger
  axom::slic::SimpleLogger logger;
  axom::slic::setIsRoot( rank == 0 );

  // command line options
  // location of mesh file. TRIBOL_REPO_DIR is defined in tribol/config.hpp
  std::string mesh_file = TRIBOL_REPO_DIR "/data/star.mesh";
  // number of times to uniformly refine the serial mesh before constructing the parallel mesh
  int ref_levels = 0;
  // polynomial order of the finite element discretization
  int order = 1;
  // 0 < max_out_of_balance <= 1; determines the maximum proportion the number of elements in each partition is allowed
  // to deviate from an exactly balanced decomposition. lower values are better, but this may need to be increased for
  // coarse meshes. note, the RCB method prevents max_out_of_balance from being so small that decomposition is
  // impossible on very coarse meshes, though the decomposition can still fail with the simple checks implemented.
  double max_out_of_balance = 0.05;
  // device configuration string (see mfem::Device::Configure() for valid options)
  std::string device_config = "cpu";

  axom::CLI::App app{ "domain_redecomp" };
  app.add_option( "-m,--mesh", mesh_file, "Mesh file to use." )
      ->check( axom::CLI::ExistingFile )
      ->capture_default_str();
  app.add_option( "-r,--refine", ref_levels, "Number of times to refine the mesh uniformly." )->capture_default_str();
  app.add_option( "-o,--order", order, "Finite element order (polynomial degree)." )->capture_default_str();
  app.add_option( "-t,--tol", max_out_of_balance, "Max proportion of out-of-balance elements in RCB decomposition." )
      ->capture_default_str();
  app.add_option( "-d,--device", device_config,
                  "Device configuration string, see mfem::Device::Configure() for valid options." )
      ->capture_default_str();
  CLI11_PARSE( app, argc, argv );

  SLIC_INFO_ROOT( "Running domain_redecomp with the following options:" );
  SLIC_INFO_ROOT( axom::fmt::format( "mesh:   {0}", mesh_file ) );
  SLIC_INFO_ROOT( axom::fmt::format( "refine: {0}", ref_levels ) );
  SLIC_INFO_ROOT( axom::fmt::format( "order:  {0}", order ) );
  SLIC_INFO_ROOT( axom::fmt::format( "tol:    {0}", max_out_of_balance ) );
  SLIC_INFO_ROOT( axom::fmt::format( "device: {0}\n", device_config ) );

  // configure the devices available for MFEM kernel launches
  mfem::Device device( device_config );
  if ( rank == 0 ) {
    device.Print();
  }

  // read serial mesh
  auto mesh = std::make_unique<mfem::Mesh>( mesh_file.c_str(), 1, 1 );

  // refine serial mesh
  if ( ref_levels > 0 ) {
    for ( int i{ 0 }; i < ref_levels; ++i ) {
      mesh->UniformRefinement();
    }
  }

  // create parallel mesh from serial
  axom::utilities::Timer timer{ false };
  timer.start();
  auto pmesh = std::make_unique<mfem::ParMesh>( MPI_COMM_WORLD, *mesh );
  mesh.reset( nullptr );
  timer.stop();
  SLIC_INFO_ROOT( axom::fmt::format( "Time to create parallel mesh: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  // further refinement of parallel mesh
  {
    int par_ref_levels = 2;
    for ( int i{ 0 }; i < par_ref_levels; ++i ) {
      pmesh->UniformRefinement();
    }
  }

  // call dimension specific version of RedecompExample
  switch ( pmesh->SpaceDimension() ) {
    case 2:
      RedecompExample<2>( *pmesh, order, max_out_of_balance );
      break;
    case 3:
      RedecompExample<3>( *pmesh, order, max_out_of_balance );
      break;
    default:
      SLIC_ERROR( "Space dimension of the mesh must be 2 or 3." );
  }

  // cleanup
  MPI_Finalize();

  return 0;
}

template <int NDIMS>
void RedecompExample( mfem::ParMesh& pmesh, int order, double max_out_of_balance )
{
  auto rank = pmesh.GetMyRank();
  auto np = pmesh.GetNRanks();

  // grid function for higher-order nodes
  auto fe_coll = mfem::H1_FECollection( order, NDIMS );
  auto par_fe_space = mfem::ParFiniteElementSpace( &pmesh, &fe_coll, NDIMS );
  auto par_x_ref_elem = mfem::ParGridFunction( &par_fe_space );
  if ( order > 1 ) {
    pmesh.SetNodalGridFunction( &par_x_ref_elem, false );
  } else {
    pmesh.GetNodes( par_x_ref_elem );
  }
  // This is the grid function we are transferring. The "node" and "element" suffixes refer to the transfer methods that
  // will be used later (node-by-node and element-by-element, respectively). See the note after STEP 3A for more
  // information.
  auto par_x_ref_node = par_x_ref_elem;

  // create visit output data collection for pmesh
  auto pmesh_dc = mfem::VisItDataCollection( "pmesh", &pmesh );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 1: Create RedecompMesh
  /////////////////////////////////////////////////////////////////////////////
  // Note: if you want to accept the default options for RedecompMesh (transfer by elements and RCB partitioning), you
  // can call the constructor with just the mfem::ParMesh, i.e. auto redecomp_mesh = redecomp::RedecompMesh(pmesh); See
  // RedecompMesh.hpp for all available constructors

  axom::utilities::Timer timer{ false };
  timer.start();
  auto redecomp_mesh = redecomp::RedecompMesh(
      pmesh, std::make_unique<const redecomp::PartitionerByDim<NDIMS>>(
                 std::make_unique<const redecomp::PartitionElements<NDIMS>>(),
                 std::make_unique<const redecomp::RCB<NDIMS>>( MPI_COMM_WORLD, max_out_of_balance ) ) );
  timer.stop();
  SLIC_INFO_ROOT( axom::fmt::format( "Time to create Redecomp mesh: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  // create visit output data collection for redecomp
  auto redecomp_dc = mfem::VisItDataCollection( "redecomp" + std::to_string( rank ), &redecomp_mesh );

  /////////////////////////////////////////////////////////////////////////////
  // GridFunction transfer
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  // STEP 2: Create GridFunctions on RedecompMesh
  /////////////////////////////////////////////////////////////////////////////

  auto redecomp_fe_space = mfem::FiniteElementSpace( &redecomp_mesh, &fe_coll, NDIMS );
  auto redecomp_x_ref_elem = mfem::GridFunction( &redecomp_fe_space );
  auto redecomp_x_ref_node = mfem::GridFunction( &redecomp_fe_space );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 3A: Create transfer object (element-by-element)
  /////////////////////////////////////////////////////////////////////////////
  // Note: GridFunctions can be transferred element-by-element or node-by-node. Element-by-element transfer is trivially
  // constructed and can be applied to any GridFunction. Node-by-node transfer is specific to FiniteElementSpaces on
  // RedecompMesh and on the ParMesh. Since the node-by-node transfer map is not trivially constructed, it is intended
  // for transferring multiple H1 fields.  See TransferByElements.hpp and TransferByNodes.hpp for more information.
  //
  // Note: Only element-by-element transfer is needed for QuadratureFunctions. This transfer object is used later there.

  timer.start();
  auto elem_transfer = redecomp::RedecompTransfer();
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to create element transfer object: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 4A: Transfer to GridFunction on RedecompMesh (element-by-element)
  /////////////////////////////////////////////////////////////////////////////

  timer.start();
  // Redecomp does transfers on host, so if par_x_ref_elem DOF data is on device, it will be copied to host and the
  // device DOF data of redecomp_x_ref_elem will be marked as invalid.
  elem_transfer.TransferToSerial( par_x_ref_elem, redecomp_x_ref_elem );
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to transfer vector field by element: {0:f}ms", timer.elapsedTimeInMilliSec() ) );
  redecomp_dc.RegisterField( "pos_elem", &redecomp_x_ref_elem );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 3B: Create transfer object (node-by-node)
  /////////////////////////////////////////////////////////////////////////////

  timer.start();
  auto node_transfer = redecomp::RedecompTransfer( par_fe_space, redecomp_fe_space );
  timer.stop();
  SLIC_INFO_ROOT( axom::fmt::format( "Time to create nodal transfer object: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 4B: Transfer to GridFunction on RedecompMesh (node-by-node)
  /////////////////////////////////////////////////////////////////////////////

  timer.start();
  // Redecomp does transfers on host, so if par_x_ref_node DOF data is on device, it will be copied to host and the
  // device DOF data of redecomp_x_ref_node will be marked as invalid.
  node_transfer.TransferToSerial( par_x_ref_node, redecomp_x_ref_node );
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to transfer vector field by node: {0:f}ms", timer.elapsedTimeInMilliSec() ) );
  redecomp_dc.RegisterField( "pos_node", &redecomp_x_ref_node );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 5: Operate on data (no-op here)
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  // STEP 6A: Transfer back to ParGridFunction on ParMesh (element-by-element)
  /////////////////////////////////////////////////////////////////////////////

  timer.start();
  // Redecomp does transfers on host, so if redecomp_x_ref_elem DOF data is on device, it will be copied to host and the
  // device DOF data of par_x_ref_elem will be marked as invalid.
  elem_transfer.TransferToParallel( redecomp_x_ref_elem, par_x_ref_elem );
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to transfer back vector field by element: {0:f}ms", timer.elapsedTimeInMilliSec() ) );
  pmesh_dc.RegisterField( "pos_elem", &par_x_ref_elem );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 6B: Transfer back to ParGridFunction on ParMesh (node-by-node)
  /////////////////////////////////////////////////////////////////////////////

  timer.start();
  // Redecomp does transfers on host, so if redecomp_x_ref_node DOF data is on device, it will be copied to host and the
  // device DOF data of par_x_ref_node will be marked as invalid.
  node_transfer.TransferToParallel( redecomp_x_ref_node, par_x_ref_node );
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to transfer back vector field by node: {0:f}ms", timer.elapsedTimeInMilliSec() ) );
  pmesh_dc.RegisterField( "pos_node", &par_x_ref_node );

  /////////////////////////////////////////////////////////////////////////////
  // QuadratureFunction transfer
  /////////////////////////////////////////////////////////////////////////////

  // store global element number as a QuadratureFunction
  auto quad_space = mfem::QuadratureSpace( &pmesh, 0 );
  auto quad_fn = mfem::QuadratureFunction( &quad_space );
  for ( int e{ 0 }; e < pmesh.GetNE(); ++e ) {
    auto quad_val = mfem::Vector();
    quad_fn.GetValues( e, quad_val );
    for ( int i{ 0 }; i < quad_val.Size(); ++i ) {
      quad_val[i] = static_cast<double>( pmesh.GetGlobalElementNum( e ) );
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // STEP 1: Create RedecompMesh
  /////////////////////////////////////////////////////////////////////////////
  // Reuse redecomp. See STEP 1 above.

  /////////////////////////////////////////////////////////////////////////////
  // STEP 2: Create QuadratureFunctions on RedecompMesh
  /////////////////////////////////////////////////////////////////////////////

  auto redecomp_quad_space = mfem::QuadratureSpace( &redecomp_mesh, 0 );
  auto redecomp_quad_fn = mfem::QuadratureFunction( &redecomp_quad_space );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 3: Create transfer object (element-by-element)
  /////////////////////////////////////////////////////////////////////////////
  // Reuse elem_transfer. See STEP 3A above.

  /////////////////////////////////////////////////////////////////////////////
  // STEP 4: Transfer to QuadratureFunction on RedecompMesh
  /////////////////////////////////////////////////////////////////////////////

  timer.start();
  // Redecomp does transfers on host, so if quad_fn quadrature point data is on device, it will be copied to host and
  // the device quadrature point data in redecomp_quad_fn will be marked as invalid.
  elem_transfer.TransferToSerial( quad_fn, redecomp_quad_fn );
  timer.stop();
  SLIC_INFO_ROOT( axom::fmt::format( "Time to transfer quadrature function: {0:f}ms", timer.elapsedTimeInMilliSec() ) );
  redecomp_dc.RegisterQField( "elem_id", &redecomp_quad_fn );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 5: Operate on data (no-op here)
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  // STEP 6: Transfer back to QuadratureFunction on ParMesh
  /////////////////////////////////////////////////////////////////////////////

  timer.start();
  // Redecomp does transfers on host, so if redecomp_quad_fn quadrature point
  // data is on device, it will be copied to host and the device quadrature
  // point data in quad_fn will be marked as invalid.
  elem_transfer.TransferToParallel( redecomp_quad_fn, quad_fn );
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to transfer back quadrature function: {0:f}ms\n", timer.elapsedTimeInMilliSec() ) );
  pmesh_dc.RegisterQField( "elem_id", &quad_fn );

  pmesh_dc.Save();
  redecomp_dc.Save();

  // print mesh stats to screen
  if ( rank == 0 ) {
    auto n_els = std::vector<int>( np );
    n_els[0] = pmesh.GetNE();
    for ( int i{ 1 }; i < np; ++i ) {
      MPI_Status status;
      MPI_Recv( &n_els[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status );
    }
    SLIC_INFO( "Original ParMesh stats" );
    SLIC_INFO( "----------------------" );
    for ( int i{ 0 }; i < np; ++i ) {
      SLIC_INFO( axom::fmt::format( "Rank {0}: {1} elements", i, n_els[i] ) );
    }
    n_els[0] = redecomp_mesh.GetNE();
    for ( int i{ 1 }; i < np; ++i ) {
      MPI_Status status;
      MPI_Recv( &n_els[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status );
    }
    SLIC_INFO( "Redecomposed Mesh stats" );
    SLIC_INFO( "-----------------------" );
    for ( int i{ 0 }; i < np; ++i ) {
      SLIC_INFO( axom::fmt::format( "Rank {0}: {1} elements", i, n_els[i] ) );
    }
  } else {
    int n_els = pmesh.GetNE();
    MPI_Send( &n_els, 1, MPI_INT, 0, 0, MPI_COMM_WORLD );
    n_els = redecomp_mesh.GetNE();
    MPI_Send( &n_els, 1, MPI_INT, 0, 0, MPI_COMM_WORLD );
  }
}
