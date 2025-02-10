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
 * @brief This tests consistency of mass matrix computation performed 1)
 * directly on an mfem::ParMesh and 2) on a redecomp::RedecompMesh, then
 * transferred to the mfem::ParMesh.
 *
 */
class MassMatrixTest : public testing::TestWithParam<std::pair<std::string, int>> {
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
    mfem::H1_FECollection h1_elems{ fe_order, dim };
    mfem::ParMesh par_mesh{ MPI_COMM_WORLD, serial_mesh };
    mfem::ParFiniteElementSpace par_fes{ &par_mesh, &h1_elems, dim };

    // compute mass matrix directly on ParMesh
    mfem::ParBilinearForm par_bf{ &par_fes };
    mfem::ConstantCoefficient rho0{ 1.0 };
    par_bf.AddDomainIntegrator( new mfem::VectorMassIntegrator( rho0 ) );
    par_bf.Assemble();
    par_bf.Finalize();
    std::unique_ptr<mfem::HypreParMatrix> par_hpm{ par_bf.ParallelAssemble() };
    mfem::SparseMatrix par_sm;
    par_hpm->MergeDiagAndOffd( par_sm );
    mfem::DenseMatrix mass_direct;
    par_sm.ToDenseMatrix( mass_direct );

    // compute mass matrix on Redecomp and transfer to ParMesh
    redecomp::RedecompMesh redecomp_mesh{ par_mesh };
    mfem::FiniteElementSpace redecomp_fes{ &redecomp_mesh, &h1_elems, dim };
    mfem::BilinearForm redecomp_bf{ &redecomp_fes };
    redecomp_bf.AddDomainIntegrator( new mfem::VectorMassIntegrator( rho0 ) );
    int n_els = redecomp_fes.GetNE();
    auto elem_idx = redecomp::ArrayUtility::IndexArray<int>( n_els );
    axom::Array<mfem::DenseMatrix> elem_mats{ n_els, n_els };
    for ( int i{ 0 }; i < n_els; ++i ) {
      redecomp_bf.ComputeElementMatrix( i, elem_mats[i] );
    }
    redecomp::MatrixTransfer matrix_xfer{ par_fes, par_fes, redecomp_fes, redecomp_fes };
    auto redecomp_hpm = matrix_xfer.TransferToParallel( elem_idx, elem_idx, elem_mats );
    mfem::SparseMatrix redecomp_sm;
    redecomp_hpm->MergeDiagAndOffd( redecomp_sm );
    mfem::DenseMatrix mass_diff;
    redecomp_sm.ToDenseMatrix( mass_diff );

    mass_diff -= mass_direct;
    max_error_ = mass_diff.MaxMaxNorm();
    max_error_ = redecomp_mesh.getMPIUtility().AllreduceValue( max_error_, MPI_MAX );
  }
};

TEST_P( MassMatrixTest, mass_matrix_transfer )
{
  EXPECT_LT( max_error_, 1.0e-13 );

  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P( redecomp, MassMatrixTest,
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
