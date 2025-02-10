// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <gtest/gtest.h>

#include "mfem.hpp"

#include "tribol/config.hpp"
#include "redecomp/redecomp.hpp"
#include "redecomp/utils/ArrayUtility.hpp"

namespace redecomp {

/**
 * @brief This tests consistency of non-square matrix computation performed 1)
 * directly on an mfem::ParMesh and 2) on a redecomp::RedecompMesh, then
 * transferred to the mfem::ParMesh.  The matrix entries are products of finite
 * element basis functions and with different trial and test spaces.
 *
 */
class RectMatrixTest : public testing::TestWithParam<std::pair<std::string, int>> {
 protected:
  double max_error_;
  void SetUp() override
  {
    auto mesh_file = GetParam().first;
    auto fe_order = GetParam().second;

    std::string mesh_filename = std::string( TRIBOL_REPO_DIR ) + mesh_file;
    mfem::Mesh serial_mesh{ mesh_filename.c_str(), 1, 1, true };
    auto dim = serial_mesh.SpaceDimension();
    serial_mesh.UniformRefinement();
    serial_mesh.UniformRefinement();
    mfem::H1_FECollection h1_elems1{ fe_order, dim };
    mfem::H1_FECollection h1_elems2{ 2, dim };
    mfem::ParMesh par_mesh{ MPI_COMM_WORLD, serial_mesh };
    mfem::ParFiniteElementSpace par_fes1{ &par_mesh, &h1_elems1, 1 };
    mfem::ParFiniteElementSpace par_fes2{ &par_mesh, &h1_elems2, 1 };

    // compute rectangular matrix directly on ParMesh
    mfem::ParMixedBilinearForm par_bf{ &par_fes1, &par_fes2 };
    mfem::ConstantCoefficient rho0{ 1.0 };
    par_bf.AddDomainIntegrator( new mfem::MixedScalarMassIntegrator( rho0 ) );
    par_bf.Assemble();
    par_bf.Finalize();
    std::unique_ptr<mfem::HypreParMatrix> par_hpm{ par_bf.ParallelAssemble() };
    mfem::SparseMatrix par_sm;
    par_hpm->MergeDiagAndOffd( par_sm );
    mfem::DenseMatrix rect_direct;
    par_sm.ToDenseMatrix( rect_direct );

    // compute rectangular matrix on Redecomp and transfer to ParMesh
    redecomp::RedecompMesh redecomp_mesh{ par_mesh };
    mfem::FiniteElementSpace redecomp_fes1{ &redecomp_mesh, &h1_elems1, 1 };
    mfem::FiniteElementSpace redecomp_fes2{ &redecomp_mesh, &h1_elems2, 1 };
    mfem::MixedBilinearForm redecomp_bf{ &redecomp_fes1, &redecomp_fes2 };
    redecomp_bf.AddDomainIntegrator( new mfem::MixedScalarMassIntegrator( rho0 ) );
    int n_els = redecomp_fes1.GetNE();
    auto elem_idx = redecomp::ArrayUtility::IndexArray<int>( n_els );
    axom::Array<mfem::DenseMatrix> elem_mats{ n_els, n_els };
    for ( int i{ 0 }; i < n_els; ++i ) {
      redecomp_bf.ComputeElementMatrix( i, elem_mats[i] );
    }
    redecomp::MatrixTransfer matrix_xfer{ par_fes2, par_fes1, redecomp_fes2, redecomp_fes1 };
    auto redecomp_hpm = matrix_xfer.TransferToParallel( elem_idx, elem_idx, elem_mats );
    mfem::SparseMatrix redecomp_sm;
    redecomp_hpm->MergeDiagAndOffd( redecomp_sm );
    mfem::DenseMatrix rect_diff;
    redecomp_sm.ToDenseMatrix( rect_diff );

    rect_diff -= rect_direct;
    max_error_ = rect_diff.MaxMaxNorm();
    max_error_ = redecomp_mesh.getMPIUtility().AllreduceValue( max_error_, MPI_MAX );
  }
};

TEST_P( RectMatrixTest, rect_matrix_transfer )
{
  EXPECT_LT( max_error_, 1.0e-13 );

  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P( redecomp, RectMatrixTest,
                          testing::Values( std::make_pair( "/data/star.mesh", 1 ),
                                           std::make_pair( "/data/star.mesh", 3 ),
                                           std::make_pair( "/data/two_hex.mesh", 1 ),
                                           std::make_pair( "/data/two_hex.mesh", 3 ) ) );

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
