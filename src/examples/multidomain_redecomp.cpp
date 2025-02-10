// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file multidomain_redecomp.cpp
 *
 * @brief Demonstrates redecomp::MultiRedecomp transfer capability over multiple
 * meshes
 *
 * Demonstrates use of redecomp::MultiRedecomp to build a single RCB-based
 * redecomposition of two mfem::ParMeshes, then transfers mfem::ParGridFunctions
 * and mfem::QuadratureFunctions to mfem::GridFunctions and
 * mfem::QuadratureFunctions on the redecomp::RedecompMeshes and vice versa.
 *
 * This example builds on domain_redecomp.cpp, now building a single RCB
 * redecomposition for multiple input mfem::ParMeshes. The single
 * redecomposition is built over all input meshes, meaning the boundaries of the
 * RCB decomposition are the same for all input meshes and each partition
 * overlaps those of the other input meshes. Ordering of the input meshes and
 * output RedecompMeshes in the std::vector is consistent, that is, the output
 * RedecompMesh 1 corresponds to the ParMesh in position 1 of the std::vector
 * input. Each resulting RedecompMesh is independent, and each should have its
 * own GridFunctions and QuadratureFunctions as needed.
 *
 * This example illustrates the envisioned typical workflow for using
 * redecomp::MultiRedecomp:
 *   1. Create a MultiRedecomp object by providing a vector of parent ParMesh
 *      pointers (and optionally a method of repartitioning the domain)
 *   2. Create fields on the RedecompMeshes stored in the MultiRedecomp object
 *      (mfem::GridFunction and/or mfem::QuadratureFunction)
 *   3. Creating a RedecompTransfer object which provides methods to transfer an
 *      mfem::ParGridFunction and an mfem::QuadratureFunction from the selected
 *      parent ParMesh to the corresponding RedecompMesh and vice versa
 *   4. Transferring the mfem::ParGridfunction and/or mfem::QuadratureFunction
 *      on the parent ParMesh to the mfem::GridFunction and/or
 *      mfem::QuadratureFunction on the appropriate RedecompMesh
 *   5. Performing an update on the mfem::GridFunction and/or
 *      mfem::QuadratureFunction for functions on each RedecompMesh
 *   6. Transferring the mfem::GridFunction and/or mfem::QuadratureFunction on
 *      the RedecompMesh back to the corresponding functions on the appropriate
 *      ParMesh
 */

#include <string>
#include <vector>

#include <mpi.h>

#include "axom/CLI11.hpp"
#include "axom/core.hpp"
#include "axom/slic.hpp"
#include "mfem.hpp"

#include "redecomp/redecomp.hpp"
#include "tribol/config.hpp"

template <int NDIMS>
void RedecompExample( mfem::ParMesh& pmesh1, mfem::ParMesh& pmesh2, int order, double max_out_of_balance );

std::unique_ptr<mfem::ParMesh> MakePMesh( MPI_Comm comm, const std::string& mesh_file, double theta, int ref_levels );

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
  std::string mesh_file1 = TRIBOL_REPO_DIR "/data/star.mesh";
  std::string mesh_file2 = TRIBOL_REPO_DIR "/data/star.mesh";
  // angle of rotation of the mesh file (in the x-y plane)
  double rot1 = 0.0;
  double rot2 = 30.0;
  // number of times to uniformly refine the serial mesh before constructing the
  // parallel mesh
  int ref_levels = 0;
  // polynomial order of the finite element discretization
  int order = 1;
  // 0 < max_out_of_balance <= 1; determines the maximum proportion the number
  // of elements in each partition is allowed to deviate from an exactly
  // balanced decomposition. lower values are better, but this may need to be
  // increased for coarse meshes. note, the RCB method prevents
  // max_out_of_balance from being so small that decomposition is impossible on
  // very coarse meshes, though the decomposition can still fail with the simple
  // checks implemented.
  double max_out_of_balance = 0.05;

  axom::CLI::App app{ "multidomain_redecomp" };
  app.add_option( "--m1,--mesh1", mesh_file1, "First mesh file to use." )
      ->check( axom::CLI::ExistingFile )
      ->capture_default_str();
  app.add_option( "--m2,--mesh2", mesh_file2, "Second mesh file to use." )
      ->check( axom::CLI::ExistingFile )
      ->capture_default_str();
  app.add_option( "--th1,--theta1", rot1, "First mesh angle of rotation in x-y plane (degrees)." )
      ->capture_default_str();
  app.add_option( "--th2,--theta2", rot2, "Second mesh angle of rotation in x-y plane (degrees)." )
      ->capture_default_str();
  app.add_option( "-r,--refine", ref_levels, "Number of times to refine the mesh uniformly." )->capture_default_str();
  app.add_option( "-o,--order", order, "Finite element order (polynomial degree)." )->capture_default_str();
  app.add_option( "-t,--tol", max_out_of_balance, "Max proportion of out-of-balance elements in RCB decomposition." )
      ->capture_default_str();
  CLI11_PARSE( app, argc, argv );

  SLIC_INFO_ROOT( "Running multidomain_redecomp with the following options:" );
  SLIC_INFO_ROOT( axom::fmt::format( "mesh1:  {0}", mesh_file1 ) );
  SLIC_INFO_ROOT( axom::fmt::format( "mesh2:  {0}", mesh_file2 ) );
  SLIC_INFO_ROOT( axom::fmt::format( "theta1: {0}", rot1 ) );
  SLIC_INFO_ROOT( axom::fmt::format( "theta2: {0}", rot2 ) );
  SLIC_INFO_ROOT( axom::fmt::format( "refine: {0}", ref_levels ) );
  SLIC_INFO_ROOT( axom::fmt::format( "order:  {0}", order ) );
  SLIC_INFO_ROOT( axom::fmt::format( "tol:    {0}\n", max_out_of_balance ) );

  // create parallel meshes
  auto pmesh1 = MakePMesh( MPI_COMM_WORLD, mesh_file1, rot1, ref_levels );
  auto pmesh2 = MakePMesh( MPI_COMM_WORLD, mesh_file2, rot2, ref_levels );

  // call dimension specific version of RedecompExample
  SLIC_ERROR_ROOT_IF( pmesh1->Dimension() != pmesh2->Dimension(), "Dimension of meshes must match." );
  switch ( pmesh1->Dimension() ) {
    case 2:
      RedecompExample<2>( *pmesh1, *pmesh2, order, max_out_of_balance );
      break;
    case 3:
      RedecompExample<3>( *pmesh1, *pmesh2, order, max_out_of_balance );
      break;
    default:
      SLIC_ERROR( "Space dimension of the mesh must be 2 or 3." );
  }

  // cleanup
  MPI_Finalize();

  return 0;
}

template <int NDIMS>
void RedecompExample( mfem::ParMesh& pmesh1, mfem::ParMesh& pmesh2, int order, double max_out_of_balance )
{
  auto rank = pmesh1.GetMyRank();
  auto np = pmesh1.GetNRanks();

  // grid function for higher-order nodes
  auto fe_coll1 = mfem::H1_FECollection( order, NDIMS );
  auto fe_coll2 = mfem::H1_FECollection( order, NDIMS );
  auto par_fe_space1 = mfem::ParFiniteElementSpace( &pmesh1, &fe_coll1, NDIMS );
  auto par_fe_space2 = mfem::ParFiniteElementSpace( &pmesh2, &fe_coll2, NDIMS );
  auto par_x_ref_elem1 = mfem::ParGridFunction( &par_fe_space1 );
  auto par_x_ref_elem2 = mfem::ParGridFunction( &par_fe_space2 );
  if ( order > 1 ) {
    pmesh1.SetNodalGridFunction( &par_x_ref_elem1, false );
    pmesh2.SetNodalGridFunction( &par_x_ref_elem2, false );
  } else {
    pmesh1.GetNodes( par_x_ref_elem1 );
    pmesh2.GetNodes( par_x_ref_elem2 );
  }
  auto par_x_ref_node1 = par_x_ref_elem1;
  auto par_x_ref_node2 = par_x_ref_elem2;

  // create visit output data collection for pmesh
  auto pmesh_dc1 = mfem::VisItDataCollection( "pmesh1", &pmesh1 );
  auto pmesh_dc2 = mfem::VisItDataCollection( "pmesh2", &pmesh2 );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 1: Create RedecompMeshes using MultiRedecomp
  /////////////////////////////////////////////////////////////////////////////
  // Note: see MultiRedecomp.hpp for a simpler constructor

  axom::utilities::Timer timer{ false };
  timer.start();
  auto multiredecomp = redecomp::MultiRedecomp(
      // redecomp::Redecomp::RCB
      std::make_unique<const redecomp::PartitionerByDim<NDIMS>>(
          std::make_unique<const redecomp::PartitionElements<NDIMS>>(),
          std::make_unique<const redecomp::RCB<NDIMS>>( MPI_COMM_WORLD, max_out_of_balance ) ) );
  auto redecomp_meshes = multiredecomp.createRedecompMeshes( { &pmesh1, &pmesh2 } );
  timer.stop();
  SLIC_INFO_ROOT( axom::fmt::format( "Time to create Redecomp meshes: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  // create visit output data collection for redecomp
  auto redecomp_dc1 = mfem::VisItDataCollection( "redecomp1_" + std::to_string( rank ), redecomp_meshes[0].get() );
  auto redecomp_dc2 = mfem::VisItDataCollection( "redecomp2_" + std::to_string( rank ), redecomp_meshes[1].get() );

  /////////////////////////////////////////////////////////////////////////////
  // GridFunction transfer
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  // STEP 2: Create GridFunctions on each RedecompMesh
  /////////////////////////////////////////////////////////////////////////////

  auto redecomp_fe_space1 = mfem::FiniteElementSpace( redecomp_meshes[0].get(), &fe_coll1, NDIMS );
  auto redecomp_fe_space2 = mfem::FiniteElementSpace( redecomp_meshes[1].get(), &fe_coll2, NDIMS );
  auto redecomp_x_ref_elem1 = mfem::GridFunction( &redecomp_fe_space1 );
  auto redecomp_x_ref_elem2 = mfem::GridFunction( &redecomp_fe_space2 );
  auto redecomp_x_ref_node1 = mfem::GridFunction( &redecomp_fe_space1 );
  auto redecomp_x_ref_node2 = mfem::GridFunction( &redecomp_fe_space2 );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 3A: Create transfer object (element-by-element)
  /////////////////////////////////////////////////////////////////////////////
  // Note: GridFunctions can be transferred element-by-element or node-by-node.
  // Element-by-element transfer is trivially constructed and can be applied to
  // any GridFunction. Node-by-node transfer is specific to FiniteElementSpaces
  // on each RedecompMesh and on each ParMesh. Since the node-by-node transfer
  // map is not trivially constructed, it is intended for transferring multiple
  // H1 fields.  See TransferByElements.hpp and TransferByNodes.hpp for more
  // information.
  //
  // Note: Only element-by-element transfer is needed for QuadratureFunctions.
  // This transfer object is used later there.

  timer.start();
  auto elem_transfer = redecomp::RedecompTransfer();
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to create element transfer object: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 4A: Transfer to GridFunctions on RedecompMeshes (element-by-element)
  /////////////////////////////////////////////////////////////////////////////

  timer.start();
  elem_transfer.TransferToSerial( par_x_ref_elem1, redecomp_x_ref_elem1 );
  elem_transfer.TransferToSerial( par_x_ref_elem2, redecomp_x_ref_elem2 );
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to transfer vector field by element: {0:f}ms", timer.elapsedTimeInMilliSec() ) );
  redecomp_dc1.RegisterField( "pos_elem", &redecomp_x_ref_elem1 );
  redecomp_dc2.RegisterField( "pos_elem", &redecomp_x_ref_elem2 );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 3B: Create transfer objects (node-by-node)
  /////////////////////////////////////////////////////////////////////////////

  timer.start();
  auto node_transfer1 = redecomp::RedecompTransfer( par_fe_space1, redecomp_fe_space1 );
  auto node_transfer2 = redecomp::RedecompTransfer( par_fe_space2, redecomp_fe_space2 );
  timer.stop();
  SLIC_INFO_ROOT( axom::fmt::format( "Time to create nodal transfer object: {0:f}ms", timer.elapsedTimeInMilliSec() ) );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 4B: Transfer to GridFunctions on RedecompMeshes (node-by-node)
  /////////////////////////////////////////////////////////////////////////////

  timer.start();
  node_transfer1.TransferToSerial( par_x_ref_node1, redecomp_x_ref_node1 );
  node_transfer2.TransferToSerial( par_x_ref_node2, redecomp_x_ref_node2 );
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to transfer vector field by node: {0:f}ms", timer.elapsedTimeInMilliSec() ) );
  redecomp_dc1.RegisterField( "pos_node", &redecomp_x_ref_node1 );
  redecomp_dc2.RegisterField( "pos_node", &redecomp_x_ref_node2 );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 5: Operate on data (no-op here)
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  // STEP 6A: Transfer back to ParGridFunctions on ParMeshes (element-by-element)
  /////////////////////////////////////////////////////////////////////////////

  timer.start();
  elem_transfer.TransferToParallel( redecomp_x_ref_elem1, par_x_ref_elem1 );
  elem_transfer.TransferToParallel( redecomp_x_ref_elem2, par_x_ref_elem2 );
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to transfer back vector field by element: {0:f}ms", timer.elapsedTimeInMilliSec() ) );
  pmesh_dc1.RegisterField( "pos_elem", &par_x_ref_elem1 );
  pmesh_dc2.RegisterField( "pos_elem", &par_x_ref_elem2 );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 6B: Transfer back to ParGridFunctions on ParMeshes (node-by-node)
  /////////////////////////////////////////////////////////////////////////////

  timer.start();
  node_transfer1.TransferToParallel( redecomp_x_ref_node1, par_x_ref_node1 );
  node_transfer2.TransferToParallel( redecomp_x_ref_node2, par_x_ref_node2 );
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to transfer back vector field by node: {0:f}ms", timer.elapsedTimeInMilliSec() ) );
  pmesh_dc1.RegisterField( "pos_node", &par_x_ref_node1 );
  pmesh_dc2.RegisterField( "pos_node", &par_x_ref_node2 );

  /////////////////////////////////////////////////////////////////////////////
  // QuadratureFunction transfer
  /////////////////////////////////////////////////////////////////////////////

  // store global element number as a QuadratureFunction
  auto quad_space1 = mfem::QuadratureSpace( &pmesh1, 0 );
  auto quad_space2 = mfem::QuadratureSpace( &pmesh2, 0 );
  auto quad_fn1 = mfem::QuadratureFunction( &quad_space1 );
  auto quad_fn2 = mfem::QuadratureFunction( &quad_space2 );
  for ( int e{ 0 }; e < pmesh1.GetNE(); ++e ) {
    auto quad_val = mfem::Vector();
    quad_fn1.GetValues( e, quad_val );
    for ( int i{ 0 }; i < quad_val.Size(); ++i ) {
      quad_val[i] = static_cast<double>( pmesh1.GetGlobalElementNum( e ) );
    }
  }
  for ( int e{ 0 }; e < pmesh2.GetNE(); ++e ) {
    auto quad_val = mfem::Vector();
    quad_fn2.GetValues( e, quad_val );
    for ( int i{ 0 }; i < quad_val.Size(); ++i ) {
      quad_val[i] = static_cast<double>( pmesh2.GetGlobalElementNum( e ) );
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // STEP 1: Create RedecompMeshes using MultiRedecomp
  /////////////////////////////////////////////////////////////////////////////
  // Reuse multiredecomp. See STEP 1 above.

  /////////////////////////////////////////////////////////////////////////////
  // STEP 2: Create QuadratureFunctions on RedecompMeshes
  /////////////////////////////////////////////////////////////////////////////

  auto redecomp_quad_space1 = mfem::QuadratureSpace( redecomp_meshes[0].get(), 0 );
  auto redecomp_quad_space2 = mfem::QuadratureSpace( redecomp_meshes[1].get(), 0 );
  auto redecomp_quad_fn1 = mfem::QuadratureFunction( &redecomp_quad_space1 );
  auto redecomp_quad_fn2 = mfem::QuadratureFunction( &redecomp_quad_space2 );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 3: Create transfer object (element-by-element)
  /////////////////////////////////////////////////////////////////////////////
  // Reuse elem_transfer. See STEP 3A above.

  /////////////////////////////////////////////////////////////////////////////
  // STEP 4: Transfer to QuadratureFunctions on RedecompMeshes
  /////////////////////////////////////////////////////////////////////////////

  timer.start();
  elem_transfer.TransferToSerial( quad_fn1, redecomp_quad_fn1 );
  elem_transfer.TransferToSerial( quad_fn2, redecomp_quad_fn2 );
  timer.stop();
  SLIC_INFO_ROOT( axom::fmt::format( "Time to transfer quadrature function: {0:f}ms", timer.elapsedTimeInMilliSec() ) );
  redecomp_dc1.RegisterQField( "elem_id", &redecomp_quad_fn1 );
  redecomp_dc2.RegisterQField( "elem_id", &redecomp_quad_fn2 );

  /////////////////////////////////////////////////////////////////////////////
  // STEP 5: Operate on data (no-op here)
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  // STEP 6: Transfer back to QuadratureFunction on ParMesh
  /////////////////////////////////////////////////////////////////////////////

  timer.start();
  elem_transfer.TransferToParallel( redecomp_quad_fn1, quad_fn1 );
  elem_transfer.TransferToParallel( redecomp_quad_fn2, quad_fn2 );
  timer.stop();
  SLIC_INFO_ROOT(
      axom::fmt::format( "Time to transfer back quadrature function: {0:f}ms\n", timer.elapsedTimeInMilliSec() ) );
  pmesh_dc1.RegisterQField( "elem_id", &quad_fn1 );
  pmesh_dc2.RegisterQField( "elem_id", &quad_fn2 );

  pmesh_dc1.Save();
  pmesh_dc2.Save();
  redecomp_dc1.Save();
  redecomp_dc2.Save();

  // print mesh stats to screen
  if ( rank == 0 ) {
    auto n_els = std::vector<int>( np );
    n_els[0] = pmesh1.GetNE();
    for ( int i{ 1 }; i < np; ++i ) {
      MPI_Status status;
      MPI_Recv( &n_els[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status );
    }
    SLIC_INFO( "First original ParMesh stats" );
    SLIC_INFO( "----------------------------" );
    for ( int i{ 0 }; i < np; ++i ) {
      SLIC_INFO( axom::fmt::format( "Rank {0}: {1} elements", i, n_els[i] ) );
    }
    n_els[0] = redecomp_meshes[0]->GetNE();
    for ( int i{ 1 }; i < np; ++i ) {
      MPI_Status status;
      MPI_Recv( &n_els[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status );
    }
    SLIC_INFO( "First redecomposed Mesh stats" );
    SLIC_INFO( "-----------------------------" );
    for ( int i{ 0 }; i < np; ++i ) {
      SLIC_INFO( axom::fmt::format( "Rank {0}: {1} elements", i, n_els[i] ) );
    }
  } else {
    int n_els = pmesh1.GetNE();
    MPI_Send( &n_els, 1, MPI_INT, 0, 0, MPI_COMM_WORLD );
    n_els = redecomp_meshes[0]->GetNE();
    MPI_Send( &n_els, 1, MPI_INT, 0, 0, MPI_COMM_WORLD );
  }

  if ( rank == 0 ) {
    auto n_els = std::vector<int>( np );
    n_els[0] = pmesh2.GetNE();
    for ( int i{ 1 }; i < np; ++i ) {
      MPI_Status status;
      MPI_Recv( &n_els[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status );
    }
    SLIC_INFO( "Second original ParMesh stats" );
    SLIC_INFO( "-----------------------------" );
    for ( int i{ 0 }; i < np; ++i ) {
      SLIC_INFO( axom::fmt::format( "Rank {0}: {1} elements", i, n_els[i] ) );
    }
    n_els[0] = redecomp_meshes[1]->GetNE();
    for ( int i{ 1 }; i < np; ++i ) {
      MPI_Status status;
      MPI_Recv( &n_els[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status );
    }
    SLIC_INFO( "Second redecomposed Mesh stats" );
    SLIC_INFO( "------------------------------" );
    for ( int i{ 0 }; i < np; ++i ) {
      SLIC_INFO( axom::fmt::format( "Rank {0}: {1} elements", i, n_els[i] ) );
    }
  } else {
    int n_els = pmesh2.GetNE();
    MPI_Send( &n_els, 1, MPI_INT, 0, 0, MPI_COMM_WORLD );
    n_els = redecomp_meshes[1]->GetNE();
    MPI_Send( &n_els, 1, MPI_INT, 0, 0, MPI_COMM_WORLD );
  }
}

std::unique_ptr<mfem::ParMesh> MakePMesh( MPI_Comm comm, const std::string& mesh_file, double theta, int ref_levels )
{
  auto mesh = std::make_unique<mfem::Mesh>( mesh_file.c_str(), 1, 1 );

  // rotate unrefined serial mesh
  if ( theta != 0.0 ) {
    theta = theta * redecomp::pi / 180.0;
    auto R = axom::numerics::Matrix<double>::zeros( 3, 3 );
    R( 0, 0 ) = cos( theta );
    R( 0, 1 ) = -sin( theta );
    R( 0, 2 ) = 0.0;
    R( 1, 0 ) = sin( theta );
    R( 1, 1 ) = cos( theta );
    R( 1, 2 ) = 0.0;
    R( 2, 0 ) = 0.0;
    R( 2, 1 ) = 0.0;
    R( 2, 2 ) = 1.0;
    auto dim = mesh->Dimension();
    for ( int v{ 0 }; v < mesh->GetNV(); ++v ) {
      auto tmp_vert = axom::Array<double>( dim, dim );
      auto vert = mesh->GetVertex( v );
      axom::numerics::matrix_vector_multiply( R, vert, tmp_vert.data() );
      for ( int i{ 0 }; i < dim; ++i ) {
        vert[i] = tmp_vert[i];
      }
    }
    auto node_gf = mesh->GetNodes();
    if ( node_gf ) {
      auto node_fes = node_gf->FESpace();
      auto vdim = node_fes->GetVDim();
      for ( int n{ 0 }; n < node_fes->GetNDofs(); ++n ) {
        auto tmp_node = axom::Array<double>( vdim, vdim );
        for ( int i{ 0 }; i < vdim; ++i ) {
          for ( int j{ 0 }; j < vdim; ++j ) {
            tmp_node[i] = tmp_node[i] + R( i, j ) * ( *node_gf )( node_fes->DofToVDof( n, j ) );
          }
        }
        for ( int i{ 0 }; i < vdim; ++i ) {
          ( *node_gf )( node_fes->DofToVDof( n, i ) ) = tmp_node[i];
        }
      }
    }
  }

  // refine serial mesh
  if ( ref_levels > 0 ) {
    for ( int i{ 0 }; i < ref_levels; ++i ) {
      mesh->UniformRefinement();
    }
  }

  // create parallel mesh from serial
  auto pmesh = std::make_unique<mfem::ParMesh>( comm, *mesh );

  // further refinement of parallel mesh
  {
    int par_ref_levels = 2;
    for ( int i{ 0 }; i < par_ref_levels; ++i ) {
      pmesh->UniformRefinement();
    }
  }

  return pmesh;
}
