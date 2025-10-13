// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file element_matrix_redecomp.cpp
 *
 * @brief Demonstrates moving element matrices using redecomp::MatrixTransfer
 *
 * Demonstrates use of redecomp::MatrixTransfer to move element-level matrices
 * from finite element spaces on a redecomp::RedecompMesh to an assembled,
 * global matrix on finite element spaces on a mfem::ParMesh. Specifically,
 * element contributions of the mass matrix are computed on the
 * redecomp::RedecompMesh, then transferred to the parent mfem::ParMesh where
 * they are assembled.  These mass matrix values are compared to mass matrix
 * values computed directly on the mfem::ParMesh.
 *
 * For this example, the test space and trial space are the same, but
 * redecomp::MatrixTransfer supports rectangular matrices, with different trial
 * and test spaces.
 *
 * Example runs (from repo root directory):
 *   - mpirun -np 4 {build_dir}/examples/element_matrix_redecomp_ex -r 1 -m
 *     data/two_hex.mesh
 *   - mpirun -np 4 {build_dir}/examples/element_matrix_redecomp_ex -r 1 -m
 *     data/star.mesh
 *   - mpirun -np 4 {build_dir}/examples/element_matrix_redecomp_ex -r 1 -o 2 -m
 *     data/two_hex.mesh
 *   - mpirun -np 4 {build_dir}/examples/element_matrix_redecomp_ex -r 1 -o 2 -m
 *     data/star.mesh
 */

#include <string>

#include <mpi.h>

#include "axom/CLI11.hpp"
#include "axom/slic.hpp"

#include "mfem.hpp"

#include "redecomp/redecomp.hpp"
#include "redecomp/utils/ArrayUtility.hpp"

#include "tribol/config.hpp"

int main( int argc, char** argv )
{
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
  // number of times to uniformly refine the serial mesh before constructing the
  // parallel mesh
  int ref_levels = 0;
  // polynomial order of the finite element discretization
  int order = 1;
  // device configuration string (see mfem::Device::Configure() for valid options)
  std::string device_config = "cpu";

  axom::CLI::App app{ "element_matrix_redecomp" };
  app.add_option( "-m,--mesh", mesh_file, "Mesh file to use." )
      ->check( axom::CLI::ExistingFile )
      ->capture_default_str();
  app.add_option( "-r,--refine", ref_levels, "Number of times to refine the mesh uniformly." )->capture_default_str();
  app.add_option( "-o,--order", order, "Finite element order (polynomial degree)." )->capture_default_str();
  app.add_option( "-d,--device", device_config,
                  "Device configuration string, see mfem::Device::Configure() for valid options." )
      ->capture_default_str();
  CLI11_PARSE( app, argc, argv );

  SLIC_INFO_ROOT( "Running element_matrix_redecomp with the following options:" );
  SLIC_INFO_ROOT( axom::fmt::format( "mesh:   {0}", mesh_file ) );
  SLIC_INFO_ROOT( axom::fmt::format( "refine: {0}", ref_levels ) );
  SLIC_INFO_ROOT( axom::fmt::format( "order:  {0}", order ) );
  SLIC_INFO_ROOT( axom::fmt::format( "device: {0}\n", device_config ) );

  // configure the devices available for MFEM kernel launches
  mfem::Device device( device_config );
  if ( rank == 0 ) {
    device.Print();
  }

  SLIC_INFO_ROOT( "Creating mfem::ParMesh..." );
  // read serial mesh
  auto mesh = std::make_unique<mfem::Mesh>( mesh_file.c_str(), 1, 1 );

  // refine serial mesh
  for ( int i{ 0 }; i < ref_levels; ++i ) {
    mesh->UniformRefinement();
  }

  // create parallel mesh from serial
  auto pmesh = std::make_unique<mfem::ParMesh>( MPI_COMM_WORLD, *mesh );
  mesh.reset( nullptr );

  // further refinement of parallel mesh
  {
    int par_ref_levels = 0;
    for ( int i{ 0 }; i < par_ref_levels; ++i ) {
      pmesh->UniformRefinement();
    }
  }

  auto pmesh_dc = mfem::VisItDataCollection( "pmesh", pmesh.get() );

  SLIC_INFO_ROOT( "Computing mass matrix on mfem::ParMesh..." );
  // compute mass matrix on parallel mesh
  mfem::H1_FECollection fe_coll{ order, pmesh->SpaceDimension() };
  mfem::ParFiniteElementSpace par_fe_space{ pmesh.get(), &fe_coll, pmesh->SpaceDimension() };
  mfem::ParBilinearForm M_par{ &par_fe_space };
  mfem::ConstantCoefficient rho0{ 1.0 };
  M_par.AddDomainIntegrator( new mfem::VectorMassIntegrator( rho0 ) );
  M_par.Assemble();
  M_par.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> Mmat_par( M_par.ParallelAssemble() );

  // grid function for higher-order nodes
  auto par_x_ref_elem = mfem::ParGridFunction( &par_fe_space );
  if ( order > 1 ) {
    pmesh->SetNodalGridFunction( &par_x_ref_elem, false );
  } else {
    pmesh->GetNodes( par_x_ref_elem );
  }
  pmesh_dc.RegisterField( "ref_coord", &par_x_ref_elem );
  pmesh_dc.Save();

  SLIC_INFO_ROOT( "Creating redecomp::RedecompMesh..." );
  // create redecomp mesh
  redecomp::RedecompMesh redecomp_mesh{ *pmesh };

  SLIC_INFO_ROOT( "Computing mass matrix on redecomp::RedecompMesh..." );
  // compute mass matrix on redecomp mesh
  mfem::FiniteElementSpace redecomp_fe_space{ &redecomp_mesh, &fe_coll, redecomp_mesh.SpaceDimension() };
  mfem::BilinearForm M_redecomp{ &redecomp_fe_space };
  M_redecomp.AddDomainIntegrator( new mfem::VectorMassIntegrator( rho0 ) );
  int n_els = redecomp_fe_space.GetNE();
  auto elem_idx = redecomp::ArrayUtility::IndexArray<int>( n_els );
  axom::Array<mfem::DenseMatrix> elem_mats{ n_els, n_els };
  for ( int i{ 0 }; i < n_els; ++i ) {
    M_redecomp.ComputeElementMatrix( i, elem_mats[i] );
  }

  SLIC_INFO_ROOT( "Transferring mass matrix from RedecompMesh to ParMesh..." );
  // transfer redecomp mass matrix to parallel mesh
  redecomp::MatrixTransfer matrix_xfer{ par_fe_space, par_fe_space, redecomp_fe_space, redecomp_fe_space };
  auto Mmat_xfer = matrix_xfer.TransferToParallel( elem_idx, elem_idx, elem_mats );

  SLIC_INFO_ROOT( "Computing max norm of difference..." );
  // compare Mmat_par and Mmat_xfer
  mfem::SparseMatrix Msm_par;
  Mmat_par->MergeDiagAndOffd( Msm_par );
  mfem::DenseMatrix Mf_par;
  Msm_par.ToDenseMatrix( Mf_par );
  mfem::SparseMatrix Msm_xfer2;
  Mmat_xfer->MergeDiagAndOffd( Msm_xfer2 );
  mfem::DenseMatrix Mf_xfer2;
  Msm_xfer2.ToDenseMatrix( Mf_xfer2 );
  mfem::DenseMatrix Mf_diff( Mf_par );
  Mf_diff.Add( -1.0, Mf_xfer2 );
  auto diff = Mf_diff.MaxMaxNorm();
  redecomp::MPIUtility mpi_util( MPI_COMM_WORLD );
  diff = mpi_util.AllreduceValue( diff, MPI_MAX );
  SLIC_INFO_ROOT( axom::fmt::format( "max |M_diff| = {0:e}", diff ) );

  // cleanup
  MPI_Finalize();

  return 0;
}
