// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file sparse_matrix_redecomp.cpp
 *
 * @brief Demonstrates moving a sparse matrix using redecomp::SparseMatrixTransfer
 *
 * Demonstrates use of redecomp::SparseMatrixTransfer to move a symmetric mfem::SparseMatrix whose rows and columns are
 * associated with a mfem::FiniteElementSpace on a redecomp::RedecompMesh to its representation as an
 * mfem::HypreParMatrix on the associated mfem::ParMesh and mfem::ParFiniteElementSpace. The row and column finite
 * element spaces are the same in this example, but redecomp::SparseMatrixTransfer supports rectangular matrices with
 * different trial and test spaces. The current implementation requires order-0 L2 finite element spaces with vdim = 1
 * (one value per element).
 *
 * The transferred matrix operator is then applied to a field whose result is verified via comparison to an analytic
 * solution for the transformed field.
 *
 * Example runs (from repo root directory):
 *   - mpirun -np 4 {build_dir}/examples/sparse_matrix_redecomp_ex -d 2 -r 0 -R 0.1
 *   - mpirun -np 4 {build_dir}/examples/sparse_matrix_redecomp_ex -d 2 -r 3 -R 0.4
 *   - mpirun -np 4 {build_dir}/examples/sparse_matrix_redecomp_ex -d 3 -r 0 -R 0.4
 *   - mpirun -np 4 {build_dir}/examples/sparse_matrix_redecomp_ex -d 3 -r 2 -R 0.1
 *
 * @details This example relies on using a Cartesian mesh with uniform element size to obtain an analytical definition
 * of the filtered field. Do not change the mesh to anything other than a square/cube without adjusting the analytical
 * filtered definition accordingly.
 */

#include <string>

#include <mpi.h>

#include "axom/CLI11.hpp"
#include "axom/slic.hpp"

#include "mfem.hpp"

#include "tribol/config.hpp"

#include "redecomp/redecomp.hpp"

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
  // spatial dimension of the mesh
  int dim{ 2 };
  // number of times to uniformly refine the serial mesh before constructing the parallel mesh
  int ref_levels{ 0 };
  // filter radius of the explicit density filter
  double filter_radius{ 1.0 };
  // set and parse options
  axom::CLI::App app{ "sparse_matrix_redecomp" };
  app.add_option( "-d,--dim", dim, "Spatial dimension of the mesh" )->capture_default_str();
  app.add_option( "-r,--refine", ref_levels, "Number of times to refine the mesh" )->capture_default_str();
  app.add_option( "-R,--filter-radius", filter_radius, "Filter radius of the explicit density filter" )
      ->capture_default_str();
  CLI11_PARSE( app, argc, argv );
  // print options
  SLIC_INFO_ROOT( "Running sparse_matrix_redecomp with the following options:" );
  SLIC_INFO_ROOT( axom::fmt::format( "dim:           {0}", dim ) );
  SLIC_INFO_ROOT( axom::fmt::format( "refine:        {0}", ref_levels ) );
  SLIC_INFO_ROOT( axom::fmt::format( "filter-radius: {0}\n", filter_radius ) );

  SLIC_INFO_ROOT( "Creating mfem::ParMesh..." );
  double side_length{ 1.0 };  // side length of square domain
  int n_elem_per_dim{ 4 };    // number of elements per dimension
  mfem::Mesh serial_mesh;
  if ( dim == 2 ) {
    serial_mesh = mfem::Mesh::MakeCartesian2D( n_elem_per_dim, n_elem_per_dim, mfem::Element::Type::QUADRILATERAL,
                                               false, side_length, side_length );
  } else {
    serial_mesh = mfem::Mesh::MakeCartesian3D( n_elem_per_dim, n_elem_per_dim, n_elem_per_dim,
                                               mfem::Element::Type::HEXAHEDRON, side_length, side_length, side_length );
  }
  // refine serial mesh
  for ( int i{ 0 }; i < ref_levels; ++i ) {
    serial_mesh.UniformRefinement();
  }
  // update the number of elements per dimension to reflect the refined mesh
  n_elem_per_dim *= std::pow( 2, ref_levels );
  // compute the element side length
  double dx{ side_length / n_elem_per_dim };
  // create parallel mesh from serial
  mfem::ParMesh par_mesh{ MPI_COMM_WORLD, serial_mesh };

  SLIC_INFO_ROOT( "Creating redecomp::RedecompMesh..." );
  // the last argument sets the ghost radius == filter_radius
  redecomp::RedecompMesh redecomp_mesh{ par_mesh, filter_radius, redecomp::RedecompMesh::RCB };

  SLIC_INFO_ROOT( "Creating finite element spaces..." );
  // create 0-order L2 finite element space on parmesh (1 point per element)
  mfem::L2_FECollection l2_elems{ 0, par_mesh.SpaceDimension() };
  mfem::ParFiniteElementSpace par_fes{ &par_mesh, &l2_elems };
  // create 0-order L2 finite element space on redecomp mesh (1 point per element)
  mfem::FiniteElementSpace redecomp_fes{ &redecomp_mesh, &l2_elems };

  SLIC_INFO_ROOT( "Creating sparse matrix transfer operator..." );
  redecomp::SparseMatrixTransfer matrix_xfer{ par_fes, par_fes, redecomp_fes, redecomp_fes };

  SLIC_INFO_ROOT( "Create sparse matrix on redecomp::RedecompMesh finite element space..." );
  mfem::SparseMatrix W_redecomp{ redecomp_fes.GetVSize(), redecomp_fes.GetVSize() };

  // define kernel of density filter (using isotropic "cone" filter here)
  auto filter_kernel = [&filter_radius]( const mfem::Vector& xi, const mfem::Vector& xj ) {
    return std::max( 0.0, filter_radius - xi.DistanceTo( xj ) );
  };

  // intialize containers needed for filter evaluation
  int n_local_elem = redecomp_mesh.GetNE();
  mfem::Vector xi( dim ), xj( dim );
  std::vector<int> n_row_entries( n_local_elem );

  // loop over each element to form a row of the matrix
  for ( int i = 0; i < n_local_elem; i++ ) {
    redecomp_mesh.GetElementCenter( i, xi );

    // loop over each element to fill in each column (j) of row (i)
    n_row_entries[i] = 0;
    for ( int j = 0; j < n_local_elem; j++ ) {
      redecomp_mesh.GetElementCenter( j, xj );
      // evaluate kernel which takes in both element centroids
      double Wij = filter_kernel( xi, xj );
      // if kernel returns a non-zero value, create entry in the sparse redecomp matrix
      if ( Wij > 0.0 ) {
        W_redecomp.Add( i, j, Wij );
        n_row_entries[i]++;
      }
    }
  }

  // convert matrix from linked list to CSR.  redecomp::SparseMatrixTransfer::TransferToParallel() requires a
  // mfem::SparseMatrix in CSR format.
  W_redecomp.Finalize();

  // normalize each row to conserve mass
  for ( int i = 0; i < n_local_elem; i++ ) {
    mfem::Vector row_data( W_redecomp.GetRowEntries( i ), n_row_entries[i] );
    row_data /= row_data.Norml1();
  }

  SLIC_INFO_ROOT( "Transferring matrix from RedecompMesh to ParMesh..." );
  auto W = matrix_xfer.TransferToParallel( W_redecomp );

  SLIC_INFO_ROOT( "Comparing transferred operator to analytic solution for a field..." );
  // define analytical field for testing
  auto x_function = [&]( const mfem::Vector& x ) {
    double pi = 3.1415;
    double f = 0.0;
    for ( int i = 0; i < dim; i++ ) {
      f += std::sin( 2. * pi * x[i] / side_length );
    }
    return 0.5 + 0.5 * f / dim;
  };

  // leverage uniform mesh to write analytical definition of the filtered field
  auto xf_function = [&]( const mfem::Vector& x ) {
    mfem::Vector xj( dim );
    double value = 0.0;
    double denom = 0.0;
    if ( dim == 2 ) {
      for ( int i = 0; i < n_elem_per_dim; i++ ) {
        xj( 0 ) = dx / 2. + i * dx;
        for ( int j = 0; j < n_elem_per_dim; j++ ) {
          xj( 1 ) = dx / 2. + j * dx;
          value += filter_kernel( x, xj ) * x_function( xj );
          denom += filter_kernel( x, xj );
        }
      }
    } else {
      for ( int i = 0; i < n_elem_per_dim; i++ ) {
        xj( 0 ) = dx / 2. + i * dx;
        for ( int j = 0; j < n_elem_per_dim; j++ ) {
          xj( 1 ) = dx / 2. + j * dx;
          for ( int k = 0; k < n_elem_per_dim; k++ ) {
            xj( 2 ) = dx / 2. + k * dx;
            value += filter_kernel( x, xj ) * x_function( xj );
            denom += filter_kernel( x, xj );
          }
        }
      }
    }
    return value / denom;
  };

  // compare filtered filed with analytical solution
  mfem::FunctionCoefficient xCoef( x_function );
  mfem::FunctionCoefficient xfCoef( xf_function );
  mfem::ParGridFunction x( &par_fes );
  mfem::ParGridFunction xf_filt( &par_fes );
  mfem::ParGridFunction xf_func( &par_fes );

  // evaulate analytical field
  x.ProjectCoefficient( xCoef );

  // filter field
  W->Mult( x, xf_filt );

  // compute analytical filtered field
  xf_func.ProjectCoefficient( xfCoef );

  // compute error
  mfem::ParGridFunction error = xf_filt;
  error -= xf_func;
  auto l2_error = std::sqrt( mfem::InnerProduct( par_mesh.GetComm(), error, error ) );

  // write output
  SLIC_INFO_ROOT( axom::fmt::format( "L2 error between operator and exact solution: {0}\n", l2_error ) );
  SLIC_INFO_ROOT( "Writing output to disk..." );
  // write redecomp mesh
  mfem::VisItDataCollection redecomp_dc{ "redecomp_rank" + std::to_string( rank ), &redecomp_mesh };
  redecomp_dc.Save();
  // write parmesh and fields
  mfem::VisItDataCollection par_dc{ "parmesh", &par_mesh };
  par_dc.RegisterField( "x", &x );
  par_dc.RegisterField( "xf_operator", &xf_filt );
  par_dc.RegisterField( "xf_analytic", &xf_func );
  par_dc.RegisterField( "xf_error", &error );
  par_dc.Save();

  // cleanup
  MPI_Finalize();

  return 0;
}
