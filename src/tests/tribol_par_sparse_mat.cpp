// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/config.hpp"

#include <gtest/gtest.h>

#ifdef TRIBOL_USE_MPI
#include <mpi.h>
#endif

#include "mfem.hpp"

#include "tribol/utils/ParSparseMat.hpp"

class ParSparseMatTest : public ::testing::Test {
 protected:
  mfem::Array<HYPRE_BigInt> GetRowStarts( MPI_Comm comm, HYPRE_BigInt global_size )
  {
    int rank, num_procs;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &num_procs );

    int local_size = global_size / num_procs;
    int remainder = global_size % num_procs;
    if ( rank < remainder ) {
      local_size++;
    }

    mfem::Array<HYPRE_BigInt> row_starts( num_procs + 1 );
    std::vector<int> local_sizes( num_procs );
    MPI_Allgather( &local_size, 1, MPI_INT, local_sizes.data(), 1, MPI_INT, comm );
    row_starts[0] = 0;
    for ( int i = 0; i < num_procs; ++i ) {
      row_starts[i + 1] = row_starts[i] + local_sizes[i];
    }
    if ( HYPRE_AssumedPartitionCheck() ) {
      auto total_dofs = row_starts[num_procs];
      row_starts.SetSize( 3 );
      row_starts[0] = row_starts[rank];
      row_starts[1] = row_starts[rank + 1];
      row_starts[2] = total_dofs;
    }

    return row_starts;
  }
};

// Test Construction
TEST_F( ParSparseMatTest, Construction )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  int num_procs;
  MPI_Comm_size( MPI_COMM_WORLD, &num_procs );
  constexpr int size = 10;
  int local_size = size / num_procs + ( rank < ( size % num_procs ) ? 1 : 0 );
  if ( rank == 0 ) std::cout << "Testing Construction..." << std::endl;

  auto row_starts_array = GetRowStarts( MPI_COMM_WORLD, size );

  // 1. From mfem::HypreParMatrix*
  mfem::HypreParMatrix* m1 =
      tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, size, row_starts_array, 1.0 ).release();
  tribol::ParSparseMat psm1( m1 );
  EXPECT_EQ( psm1.get().Height(), local_size );

  // 2. From unique_ptr
  auto m2 = std::unique_ptr<mfem::HypreParMatrix>(
      tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, size, row_starts_array, 2.0 ).release() );
  tribol::ParSparseMat psm2( std::move( m2 ) );
  EXPECT_EQ( psm2.get().Height(), local_size );

  // 3. From SparseMatrix rvalue
  mfem::SparseMatrix diag( local_size );
  for ( int i = 0; i < local_size; ++i ) diag.Set( i, i, 3.0 );
  diag.Finalize();

  tribol::ParSparseMat psm3( MPI_COMM_WORLD, (HYPRE_BigInt)size, row_starts_array.GetData(), std::move( diag ) );
  EXPECT_EQ( psm3.get().Height(), local_size );

  mfem::Vector x( local_size ), y( local_size );
  x = 1.0;
  psm3.get().Mult( x, y );
  EXPECT_NEAR( y.Max(), 3.0, 1e-12 );
}

// Test View
TEST_F( ParSparseMatTest, View )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing View..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  tribol::ParSparseMat A = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );

  // Construct View
  tribol::ParSparseMatView view( &A.get() );

  EXPECT_EQ( view.get().Height(), A.get().Height() );

  // Operate on View
  tribol::ParSparseMat B = view * 2.0;
  mfem::Vector x( A.get().Width() ), y( A.get().Height() );
  x = 1.0;
  B.get().Mult( x, y );
  EXPECT_NEAR( y.Max(), 4.0, 1e-12 );
}

// Test Addition
TEST_F( ParSparseMatTest, Addition )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Addition..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  tribol::ParSparseMat A = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );
  tribol::ParSparseMat B = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 3.0 );

  // A + B
  tribol::ParSparseMat C = A + B;
  mfem::Vector x( A.get().Width() ), y( A.get().Height() );
  x = 1.0;
  C.get().Mult( x, y );
  // Result should be (2+3)*1 = 5
  EXPECT_NEAR( y.Max(), 5.0, 1e-12 );
  EXPECT_NEAR( y.Min(), 5.0, 1e-12 );

  // A += B
  A += B;
  A.get().Mult( x, y );
  EXPECT_NEAR( y.Max(), 5.0, 1e-12 );
}

// Test Subtraction
TEST_F( ParSparseMatTest, Subtraction )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Subtraction..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  tribol::ParSparseMat A = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 5.0 );
  tribol::ParSparseMat B = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );

  // A - B
  tribol::ParSparseMat C = A - B;
  mfem::Vector x( A.get().Width() ), y( A.get().Height() );
  x = 1.0;
  C.get().Mult( x, y );
  // Result should be (5-2)*1 = 3
  EXPECT_NEAR( y.Max(), 3.0, 1e-12 );

  // A -= B
  A -= B;
  A.get().Mult( x, y );
  EXPECT_NEAR( y.Max(), 3.0, 1e-12 );
}

// Test Scalar Multiplication
TEST_F( ParSparseMatTest, ScalarMult )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Scalar Multiplication..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  tribol::ParSparseMat A = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );

  // A * s
  tribol::ParSparseMat B = A * 3.0;
  mfem::Vector x( A.get().Width() ), y( A.get().Height() );
  x = 1.0;
  B.get().Mult( x, y );
  EXPECT_NEAR( y.Max(), 6.0, 1e-12 );

  // s * A
  tribol::ParSparseMat C = 4.0 * A;
  C.get().Mult( x, y );
  EXPECT_NEAR( y.Max(), 8.0, 1e-12 );
}

// Test Matrix Multiplication
TEST_F( ParSparseMatTest, MatrixMult )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Matrix Multiplication..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  tribol::ParSparseMat A = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );
  tribol::ParSparseMat B = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 3.0 );

  // A * B
  tribol::ParSparseMat C = A * B;
  mfem::Vector x( A.get().Width() ), y( A.get().Height() );
  x = 1.0;
  C.get().Mult( x, y );
  // Result should be (2*3)*1 = 6
  EXPECT_NEAR( y.Max(), 6.0, 1e-12 );

  // A *= B
  A *= B;
  A.get().Mult( x, y );
  EXPECT_NEAR( y.Max(), 6.0, 1e-12 );
}

// Test Matrix-Vector Multiplication
TEST_F( ParSparseMatTest, MatVecMult )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Matrix-Vector Multiplication..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  tribol::ParSparseMat A = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );
  mfem::Vector x( A.get().Width() );
  x = 1.0;

  // y = A * x
  mfem::Vector y = A * x;
  EXPECT_NEAR( y.Max(), 2.0, 1e-12 );
}

// Test Vector-Matrix Multiplication
TEST_F( ParSparseMatTest, VecMatMult )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Vector-Matrix Multiplication..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  tribol::ParSparseMat A = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 3.0 );
  mfem::Vector x( A.get().Height() );
  x = 1.0;

  // y = x^T * A
  mfem::Vector y = x * A;
  EXPECT_NEAR( y.Max(), 3.0, 1e-12 );
  EXPECT_NEAR( y.Min(), 3.0, 1e-12 );
}

// Test Elimination
TEST_F( ParSparseMatTest, Elimination )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Elimination..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  tribol::ParSparseMat A = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 3.0 );

  // Eliminate row 0 (globally)
  // Determine if I own row 0
  mfem::Array<int> rows_to_elim;
  int row_starts_idx = HYPRE_AssumedPartitionCheck() ? 0 : rank;
  if ( row_starts[row_starts_idx] == 0 ) {
    rows_to_elim.Append( 0 );
  }
  A.EliminateRows( rows_to_elim );

  // Check if row 0 is identity (or zero with diagonal 1)
  // Diagonal matrix means we can just check multiplication
  mfem::Vector x( A.get().Height() ), y( A.get().Height() );
  x = 1.0;
  y = A * x;  // y = A * x

  // if rank owns row 0, the result for that row should be 0.0 * x[0] = 0.0 (since diag is 0.0)
  // other rows should be 3.0

  // local row 0 on rank 0 is global row 0
  if ( rank == 0 ) {
    EXPECT_NEAR( y[0], 0.0, 1e-12 );
    for ( int i = 1; i < y.Size(); ++i ) {
      EXPECT_NEAR( y[i], 3.0, 1e-12 );
    }
  } else {
    for ( int i = 0; i < y.Size(); ++i ) {
      EXPECT_NEAR( y[i], 3.0, 1e-12 );
    }
  }

  A = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 3.0 );
  int num_procs;
  MPI_Comm_size( MPI_COMM_WORLD, &num_procs );

  // Eliminate last local col
  auto last_local_col = A.get().Width() - 1;
  mfem::Array<int> cols_to_elim( { last_local_col } );

  tribol::ParSparseMat Ae = A.EliminateCols( cols_to_elim );

  // Now check A * e_last = 0
  // Create vector with 1 at last_local_col, 0 elsewhere
  x = 0.0;
  x[last_local_col] = 1.0;
  y = A * x;
  EXPECT_NEAR( y[last_local_col], 0.0, 1e-12 );

  // Check Ae * e_last = original value
  mfem::Vector ye = Ae * x;
  double expected_val = 3.0;

  EXPECT_NEAR( ye[last_local_col], expected_val, 1e-12 );
}

// Test Transpose and Square
TEST_F( ParSparseMatTest, TransposeSquare )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Transpose and Square..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  tribol::ParSparseMat A = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );

  // Transpose (Diagonal matrix is symmetric)
  tribol::ParSparseMat At = A.transpose();
  mfem::Vector x( A.get().Width() ), y( A.get().Height() );
  x = 1.0;
  At.get().Mult( x, y );
  EXPECT_NEAR( y.Max(), 2.0, 1e-12 );

  // Square
  tribol::ParSparseMat A2 = A.square();
  A2.get().Mult( x, y );
  EXPECT_NEAR( y.Max(), 4.0, 1e-12 );
}

// Test RAP
TEST_F( ParSparseMatTest, RAP )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing RAP..." << std::endl;

  // Use Identity for P to simplify testing: P^T * A * P = I * A * I = A
  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  tribol::ParSparseMat A = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 5.0 );
  tribol::ParSparseMat P = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 1.0 );
  tribol::ParSparseMat R = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 1.0 );

  // RAP(P)
  tribol::ParSparseMat Res1 = A.RAP( P );
  mfem::Vector x( A.get().Width() ), y( A.get().Height() );
  x = 1.0;
  Res1.get().Mult( x, y );
  EXPECT_NEAR( y.Max(), 5.0, 1e-12 );

  // RAP(R, A, P)
  tribol::ParSparseMat Res2 = tribol::ParSparseMat::RAP( R, A, P );
  Res2.get().Mult( x, y );
  EXPECT_NEAR( y.Max(), 5.0, 1e-12 );
}

// Test Accessors
TEST_F( ParSparseMatTest, Accessors )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  int num_procs;
  MPI_Comm_size( MPI_COMM_WORLD, &num_procs );
  constexpr int size = 10;
  int local_size = size / num_procs + ( rank < ( size % num_procs ) ? 1 : 0 );
  if ( rank == 0 ) std::cout << "Testing Accessors..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, size );
  tribol::ParSparseMat A = tribol::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, size, row_starts, 1.0 );

  // get()
  EXPECT_EQ( A.get().Height(), local_size );

  // operator->
  EXPECT_EQ( A->Height(), local_size );
}

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
