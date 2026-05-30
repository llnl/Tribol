// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <gtest/gtest.h>

#include "mfem.hpp"

#include "shared/math/ParSparseMat.hpp"

#include "redecomp/redecomp.hpp"
#include "redecomp/utils/ArrayUtility.hpp"

namespace redecomp {

namespace {

double MaxAbsEntry( mfem::HypreParMatrix& mat )
{
  // Hypre stores diagonal and off-diagonal CSR blocks separately, so inspect both.
  double local_max = 0.0;
  HYPRE_ParCSRMatrix csr = mat;
  auto* parcsr = (hypre_ParCSRMatrix*)csr;

  auto* diag = hypre_ParCSRMatrixDiag( parcsr );
  if ( auto* data = hypre_CSRMatrixData( diag ) ) {
    const int nnz = hypre_CSRMatrixNumNonzeros( diag );
    for ( int i = 0; i < nnz; ++i ) {
      local_max = std::max( local_max, std::abs( data[i] ) );
    }
  }

  auto* offd = hypre_ParCSRMatrixOffd( parcsr );
  if ( auto* data = hypre_CSRMatrixData( offd ) ) {
    const int nnz = hypre_CSRMatrixNumNonzeros( offd );
    for ( int i = 0; i < nnz; ++i ) {
      local_max = std::max( local_max, std::abs( data[i] ) );
    }
  }

  double global_max = 0.0;
  MPI_Allreduce( &local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  return global_max;
}

double MaxAbsDiff( const shared::ParSparseMat& A, const shared::ParSparseMat& B )
{
  auto* A_nc = const_cast<mfem::HypreParMatrix*>( &A.get() );
  auto* B_nc = const_cast<mfem::HypreParMatrix*>( &B.get() );
  auto diff = shared::ParSparseMatView( A_nc ) - shared::ParSparseMatView( B_nc );
  return MaxAbsEntry( diff.get() );
}

}  // namespace

// NOTE: This test exists only to keep the two MatrixTransfer overloads equivalent
// while the dense-matrix overload is still supported. Remove it when the
// DenseMatrix overload is deprecated/removed.
TEST( MatrixTransfer, dense_and_flat_overloads_match )
{
  // Small mesh so the test is cheap; MPI correctness is what we care about.
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D( 2, 2, mfem::Element::QUADRILATERAL, false, 1.0, 1.0 );
  mfem::ParMesh par_mesh( MPI_COMM_WORLD, serial_mesh );

  const int dim = par_mesh.SpaceDimension();
  mfem::H1_FECollection h1_trial( 1, dim );
  mfem::H1_FECollection h1_test( 2, dim );

  // Mixed bilinear form gives a rectangular matrix and exercises the test/trial
  // FE-space sizing logic in MatrixTransfer.
  mfem::ParFiniteElementSpace par_trial_fes( &par_mesh, &h1_trial, 1 );
  mfem::ParFiniteElementSpace par_test_fes( &par_mesh, &h1_test, 1 );

  redecomp::RedecompMesh redecomp_mesh( par_mesh );
  mfem::FiniteElementSpace redecomp_trial_fes( &redecomp_mesh, &h1_trial, 1 );
  mfem::FiniteElementSpace redecomp_test_fes( &redecomp_mesh, &h1_test, 1 );

  mfem::ConstantCoefficient rho0{ 1.0 };
  mfem::MixedBilinearForm redecomp_bf( &redecomp_trial_fes, &redecomp_test_fes );
  redecomp_bf.AddDomainIntegrator( new mfem::MixedScalarMassIntegrator( rho0 ) );

  const int n_els = redecomp_trial_fes.GetNE();
  auto elem_idx = redecomp::ArrayUtility::IndexArray<int>( n_els );

  axom::Array<mfem::DenseMatrix> elem_mats( n_els, n_els );
  axom::Array<double> flat_data;
  axom::Array<int> flat_offsets( n_els, n_els );

  int total_values = 0;
  for ( int e = 0; e < n_els; ++e ) {
    redecomp_bf.ComputeElementMatrix( e, elem_mats[e] );
    total_values += elem_mats[e].Height() * elem_mats[e].Width();
  }
  flat_data.reserve( total_values );

  for ( int e = 0; e < n_els; ++e ) {
    flat_offsets[e] = flat_data.size();
    const int size = elem_mats[e].Height() * elem_mats[e].Width();
    flat_data.append( axom::ArrayView<const double>( elem_mats[e].GetData(), size ) );
  }

  redecomp::MatrixTransfer xfer( par_test_fes, par_trial_fes, redecomp_test_fes, redecomp_trial_fes );

  for ( bool parallel_assemble : { false, true } ) {
    auto A = xfer.TransferToParallel( elem_idx, elem_idx, elem_mats, parallel_assemble );
    auto B = xfer.TransferToParallel( elem_idx, elem_idx, flat_data, flat_offsets, parallel_assemble );
    EXPECT_LT( MaxAbsDiff( A, B ), 1.0e-13 );
  }
}

}  // namespace redecomp

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main( int argc, char* argv[] )
{
  int result = 0;

  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
