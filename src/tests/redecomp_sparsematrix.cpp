// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <gtest/gtest.h>

#include <mpi.h>

#include "mfem.hpp"

#include "tribol/config.hpp"

#include "redecomp/redecomp.hpp"

namespace redecomp {

/**
 * @brief This test transfers a symmetric mfem::SparseMatrix whose rows and colums are associated with an order-0 L2
 * field on a RedecompMesh to an mfem::HypreParMatrix on the associated parent mfem::ParMesh.  The transferred matrix is
 * verified by applying it to a field and comparing the result to an analytic solution for the operator.
 *
 * @details This test relies on using a cartesian mesh with uniform element size to obtain an analytical definition of
 * the filtered field. Do not change the mesh to anything other than a square without adjusting the analytical filtered
 * definition accordingly.
 *
 */
class SparseMatrixTest : public testing::TestWithParam<std::tuple<int, int, double>> {
 protected:
  double max_error_;
  void SetUp() override
  {
    auto dim = std::get<0>( GetParam() );            // spatial dimension of the mesh
    auto ref_levels = std::get<1>( GetParam() );     // number of times to refine the mesh
    auto filter_radius = std::get<2>( GetParam() );  // filter radius of the explicit density filter

    double side_length = 1.0;  // side length of square domain
    int n_elem_per_dim = 4;    // number of elements per dimension
    mfem::Mesh serial_mesh;
    if ( dim == 2 ) {
      serial_mesh = mfem::Mesh::MakeCartesian2D( n_elem_per_dim, n_elem_per_dim, mfem::Element::Type::QUADRILATERAL,
                                                 false, side_length, side_length );
    } else {
      serial_mesh =
          mfem::Mesh::MakeCartesian3D( n_elem_per_dim, n_elem_per_dim, n_elem_per_dim, mfem::Element::Type::HEXAHEDRON,
                                       side_length, side_length, side_length );
    }
    // refine mesh "ref_levels" times
    for ( int i{ 0 }; i < ref_levels; ++i ) {
      serial_mesh.UniformRefinement();
    }
    // update the number of elements per dimension to reflect the refined mesh
    n_elem_per_dim *= std::pow( 2, ref_levels );
    // compute the element side length
    double dx = side_length / n_elem_per_dim;

    // build a parallel mesh with associated 0-order, L2 finite element space
    mfem::ParMesh par_mesh{ MPI_COMM_WORLD, serial_mesh };
    mfem::L2_FECollection l2_elems{ 0, dim };
    mfem::ParFiniteElementSpace par_fes{ &par_mesh, &l2_elems };

    // create RedecompMesh and fe space and sparse matrix transfer object
    redecomp::RedecompMesh redecomp_mesh{ par_mesh, filter_radius, redecomp::RedecompMesh::RCB };
    mfem::FiniteElementSpace redecomp_fes{ &redecomp_mesh, &l2_elems };
    redecomp::SparseMatrixTransfer matrix_xfer{ par_fes, par_fes, redecomp_fes, redecomp_fes };

    // compute sparse matrix on redecomp_fes (rows and col)
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

    // convert linked list format to CSR format (required to use sparse matrix transfer)
    W_redecomp.Finalize();

    // normalize each row of filter matrix to conserve mass
    for ( int i = 0; i < n_local_elem; i++ ) {
      mfem::Vector row_data( W_redecomp.GetRowEntries( i ), n_row_entries[i] );
      row_data /= row_data.Norml1();
    }

    // transfer matrix to mfem::ParMesh
    auto W = matrix_xfer.TransferToParallel( W_redecomp );

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
    max_error_ = mfem::InnerProduct( par_mesh.GetComm(), error, error );
  }
};

TEST_P( SparseMatrixTest, mass_matrix_transfer )
{
  EXPECT_LT( max_error_, 1.0e-13 );

  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P( redecomp, SparseMatrixTest,
                          testing::Values( std::make_tuple( 2, 0, 0.1 ), std::make_tuple( 2, 0, 0.4 ),
                                           std::make_tuple( 2, 1, 0.1 ), std::make_tuple( 2, 1, 0.4 ),
                                           std::make_tuple( 2, 2, 0.1 ), std::make_tuple( 2, 2, 0.4 ),
                                           std::make_tuple( 2, 3, 0.1 ), std::make_tuple( 2, 3, 0.4 ),
                                           std::make_tuple( 3, 0, 0.1 ), std::make_tuple( 3, 0, 0.4 ),
                                           std::make_tuple( 3, 1, 0.1 ), std::make_tuple( 3, 1, 0.4 ),
                                           std::make_tuple( 3, 2, 0.1 ), std::make_tuple( 3, 2, 0.4 ) ) );

}  // namespace redecomp

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main( int argc, char* argv[] )
{
  int result = 0;

  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
