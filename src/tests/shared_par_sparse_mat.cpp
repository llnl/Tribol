// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "shared/config.hpp"

#include <gtest/gtest.h>

#ifdef TRIBOL_USE_MPI
#include <mpi.h>
#endif

#include "mfem.hpp"

#include "shared/math/ParVector.hpp"
#include "shared/math/ParSparseMat.hpp"

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
      // if this is true, row starts should be {starting on-rank dof, last on-rank dof + 1, total dofs}
      auto total_dofs = row_starts[num_procs];
      row_starts.SetSize( 3 );
      row_starts[0] = row_starts[rank];
      row_starts[1] = row_starts[rank + 1];
      row_starts[2] = total_dofs;
    }

    return row_starts;
  }

  std::unique_ptr<mfem::HypreParMatrix> ParMultWithMemoryLocation( const shared::ParSparseMatView& lhs,
                                                                   const shared::ParSparseMatView& rhs,
                                                                   HYPRE_MemoryLocation memory_location )
  {
    HYPRE_MemoryLocation old_hypre_mem_location;
    HYPRE_GetMemoryLocation( &old_hypre_mem_location );
    HYPRE_SetMemoryLocation( memory_location );
    auto* result = mfem::ParMult( const_cast<mfem::HypreParMatrix*>( &lhs.get() ),
                                  const_cast<mfem::HypreParMatrix*>( &rhs.get() ), true );
    HYPRE_SetMemoryLocation( old_hypre_mem_location );
    return std::unique_ptr<mfem::HypreParMatrix>( result );
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
      shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, size, row_starts_array, 1.0 ).release();
  shared::ParSparseMat psm1( m1 );
  EXPECT_EQ( psm1.height(), local_size );

  // 2. From unique_ptr
  auto m2 = std::unique_ptr<mfem::HypreParMatrix>(
      shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, size, row_starts_array, 2.0 ).release() );
  shared::ParSparseMat psm2( std::move( m2 ) );
  EXPECT_EQ( psm2.height(), local_size );

  // 3. From SparseMatrix rvalue
  mfem::SparseMatrix diag( local_size );
  for ( int i = 0; i < local_size; ++i ) diag.Set( i, i, 3.0 );
  diag.Finalize();

  shared::ParSparseMat psm3( MPI_COMM_WORLD, (HYPRE_BigInt)size, row_starts_array.GetData(), std::move( diag ) );
  EXPECT_EQ( psm3.height(), local_size );

  // basic, non-exhaustive check to make sure matrix multiplication is working
  mfem::Vector x( local_size ), y( local_size );
  x = 1.0;
  psm3->Mult( x, y );
  EXPECT_NEAR( y.Max(), 3.0, 1e-12 );
}

TEST_F( ParSparseMatTest, ForwardingConstructor )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  int num_procs;
  MPI_Comm_size( MPI_COMM_WORLD, &num_procs );
  constexpr int global_size = 10;
  const int local_size = global_size / num_procs + ( rank < ( global_size % num_procs ) ? 1 : 0 );
  if ( rank == 0 ) std::cout << "Testing Forwarding Constructor..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, global_size );
  const int row_starts_idx = HYPRE_AssumedPartitionCheck() ? 0 : rank;
  const HYPRE_BigInt first_global_row = row_starts[row_starts_idx];

  auto* I = new int[local_size + 1];
  auto* J = new HYPRE_BigInt[local_size];
  auto* data = new mfem::real_t[local_size];
  for ( int i = 0; i < local_size; ++i ) {
    I[i] = i;
    J[i] = first_global_row + i;
    data[i] = 7.0;
  }
  I[local_size] = local_size;

  shared::ParSparseMat A( MPI_COMM_WORLD, local_size, static_cast<HYPRE_BigInt>( global_size ),
                          static_cast<HYPRE_BigInt>( global_size ), I, J, data, row_starts.GetData(),
                          row_starts.GetData() );

  mfem::Vector x( A.width() ), y( A.height() );
  x = 1.0;
  A->Mult( x, y );
  EXPECT_NEAR( y.Max(), 7.0, 1e-12 );
  EXPECT_NEAR( y.Min(), 7.0, 1e-12 );
}

// Test View
TEST_F( ParSparseMatTest, View )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing View..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParSparseMat A = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );

  // Construct View
  shared::ParSparseMatView view( &A.get() );

  EXPECT_EQ( view.height(), A.height() );

  // Operate on View
  shared::ParSparseMat B = view * 2.0;
  // creates a shared::ParVector compatible with the dimensions of B and using the same MPI_Comm
  shared::ParVector x( B.get() );
  x.fill( 1.0 );
  auto y = B * x;
  EXPECT_NEAR( y.max(), 4.0, 1e-12 );
}

// Test Addition
TEST_F( ParSparseMatTest, Addition )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Addition..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParSparseMat A = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );
  shared::ParSparseMat B = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 3.0 );

  // A + B
  shared::ParSparseMat C = A + B;
  mfem::Vector x( A.width() ), y( A.height() );
  x = 1.0;
  C->Mult( x, y );
  // Result should be C*x = (A+B)*x = (2+3)*1 = 5
  EXPECT_NEAR( y.Max(), 5.0, 1e-12 );
  EXPECT_NEAR( y.Min(), 5.0, 1e-12 );

  // A += B
  A += B;
  A->Mult( x, y );
  EXPECT_NEAR( y.Max(), 5.0, 1e-12 );
}

// Test Subtraction
TEST_F( ParSparseMatTest, Subtraction )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Subtraction..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParSparseMat A = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 5.0 );
  shared::ParSparseMat B = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );

  // A - B
  shared::ParSparseMat C = A - B;
  mfem::Vector x( A.width() ), y( A.height() );
  x = 1.0;
  C->Mult( x, y );
  // Result should be (5-2)*1 = 3
  EXPECT_NEAR( y.Max(), 3.0, 1e-12 );

  // A -= B
  A -= B;
  A->Mult( x, y );
  EXPECT_NEAR( y.Max(), 3.0, 1e-12 );
}

// Test Scalar Multiplication
TEST_F( ParSparseMatTest, ScalarMult )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Scalar Multiplication..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParSparseMat A = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );

  // A * s
  shared::ParSparseMat B = A * 3.0;
  mfem::Vector x( A.width() ), y( A.height() );
  x = 1.0;
  B->Mult( x, y );
  EXPECT_NEAR( y.Max(), 6.0, 1e-12 );

  // s * A
  shared::ParSparseMat C = 4.0 * A;
  C->Mult( x, y );
  EXPECT_NEAR( y.Max(), 8.0, 1e-12 );
}

// Test Matrix Multiplication
TEST_F( ParSparseMatTest, MatrixMult )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Matrix Multiplication..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParSparseMat A = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );
  shared::ParSparseMat B = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 3.0 );

  // A * B
  shared::ParSparseMat C = A * B;
  mfem::Vector x( A.width() ), y( A.height() );
  x = 1.0;
  C->Mult( x, y );
  // Result should be (2*3)*1 = 6
  EXPECT_NEAR( y.Max(), 6.0, 1e-12 );

  // A *= B
  A *= B;
  A->Mult( x, y );
  EXPECT_NEAR( y.Max(), 6.0, 1e-12 );
}

#if defined( TRIBOL_USE_CUDA ) || defined( TRIBOL_USE_HIP )
TEST_F( ParSparseMatTest, ExternalMatrixResultIsNormalizedToHost )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing External Matrix Host Normalization..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParSparseMat A = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );
  shared::ParSparseMat B = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 3.0 );
  shared::ParSparseMat I = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 1.0 );

  auto raw_mult = ParMultWithMemoryLocation( A, B, HYPRE_MEMORY_DEVICE );
  ASSERT_NE( raw_mult, nullptr );

  shared::ParSparseMat external_result( std::move( raw_mult ) );

  mfem::Vector x( A.width() ), y( A.height() );
  x = 1.0;
  external_result->Mult( x, y );
  EXPECT_NEAR( y.Max(), 6.0, 1e-12 );

  auto sum = external_result + A;
  sum->Mult( x, y );
  EXPECT_NEAR( y.Max(), 8.0, 1e-12 );

  auto transposed = external_result.transpose();
  transposed->Mult( x, y );
  EXPECT_NEAR( y.Max(), 6.0, 1e-12 );

  auto rap_result = external_result.rap( I );
  rap_result->Mult( x, y );
  EXPECT_NEAR( y.Max(), 6.0, 1e-12 );
}
#endif

// Test Matrix-Vector Multiplication
TEST_F( ParSparseMatTest, MatVecMult )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Matrix-Vector Multiplication..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParSparseMat A = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );
  HYPRE_MemoryLocation old_hypre_mem_location;
  HYPRE_GetMemoryLocation( &old_hypre_mem_location );
  HYPRE_SetMemoryLocation( HYPRE_MEMORY_HOST );
  mfem::HypreParVector x_hypre( A.get(), 1 );
  HYPRE_SetMemoryLocation( old_hypre_mem_location );
  x_hypre = 1.0;
  shared::ParVectorView x( &x_hypre );

  // y = A * x
  shared::ParVector y = A * x;
  EXPECT_NEAR( y.max(), 2.0, 1e-12 );
}

// Test Vector-Matrix Multiplication
TEST_F( ParSparseMatTest, VecMatMult )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Vector-Matrix Multiplication..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParSparseMat A = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 3.0 );
  shared::ParVector x( A.get(), 0 );
  x.fill( 1.0 );

  // y = x^T * A
  shared::ParVector y = x * A;
  EXPECT_NEAR( y.max(), 3.0, 1e-12 );
  EXPECT_NEAR( y.min(), 3.0, 1e-12 );
}

// Test Elimination
TEST_F( ParSparseMatTest, Elimination )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Elimination..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParSparseMat A = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 3.0 );

  // Eliminate row 0 (globally)
  // Determine if I own row 0
  mfem::Array<int> indices_to_elim;
  int row_starts_idx = HYPRE_AssumedPartitionCheck() ? 0 : rank;
  if ( row_starts[row_starts_idx] == 0 ) {
    indices_to_elim.Append( 0 );
  }
  A.eliminateRows( indices_to_elim );

  // Check if row 0 is identity (or zero with diagonal 1)
  // Diagonal matrix means we can just check multiplication
  shared::ParVector x( A.get(), 1 );
  x.fill( 1.0 );
  shared::ParVector y = A * x;  // y = A * x

  // if rank owns row 0, the result for that row should be 0.0 * x[0] = 0.0 (since diag is 0.0)
  // other rows should be 3.0

  // local row 0 on rank 0 is global row 0
  if ( rank == 0 ) {
    EXPECT_NEAR( y[0], 0.0, 1e-12 );
    for ( int i = 1; i < y.size(); ++i ) {
      EXPECT_NEAR( y[i], 3.0, 1e-12 );
    }
  } else {
    for ( int i = 0; i < y.size(); ++i ) {
      EXPECT_NEAR( y[i], 3.0, 1e-12 );
    }
  }

  A = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 3.0 );
  shared::ParSparseMat Ae = A.eliminateRowsCols( indices_to_elim );
  x.fill( 1.0 );
  y = A * x;
  shared::ParVector ye = Ae * x;
  for ( int i = 0; i < y.size(); ++i ) {
    EXPECT_NEAR( y[i] + ye[i], 3.0, 1e-12 );
  }

  A = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 3.0 );
  int num_procs;
  MPI_Comm_size( MPI_COMM_WORLD, &num_procs );

  // Eliminate last local col
  auto last_local_col = A.width() - 1;
  mfem::Array<int> cols_to_elim( { last_local_col } );

  Ae = A.eliminateCols( cols_to_elim );

  // Now check A * e_last = 0
  // Create vector with 1 at last_local_col, 0 elsewhere
  shared::ParVector x_last( A.get(), 1 );
  x_last.fill( 0.0 );
  x_last[last_local_col] = 1.0;
  shared::ParVector y_last = A * x_last;
  EXPECT_NEAR( y_last[last_local_col], 0.0, 1e-12 );

  // Check Ae * e_last = original value
  ye = Ae * x_last;
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
  shared::ParSparseMat A = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 2.0 );

  // Transpose (Diagonal matrix is symmetric)
  shared::ParSparseMat At = A.transpose();
  shared::ParVector x( At.get(), 0 );
  x.fill( 1.0 );
  auto y = At * x;
  EXPECT_NEAR( y.max(), 2.0, 1e-12 );

  // Square
  shared::ParSparseMat A2 = A.square();
  y = A2 * x;
  EXPECT_NEAR( y.max(), 4.0, 1e-12 );
}

// Test RAP
TEST_F( ParSparseMatTest, RAP )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing RAP..." << std::endl;

  // Use Identity for P to simplify testing: P^T * A * P = I * A * I = A
  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParSparseMat A = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 5.0 );
  shared::ParSparseMat P = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 1.0 );
  shared::ParSparseMat R = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, 10, row_starts, 1.0 );

  // rap(P)
  shared::ParSparseMat Res1 = A.rap( P );
  shared::ParVector x( A.get(), 0 );
  x.fill( 1.0 );
  auto y = Res1 * x;
  EXPECT_NEAR( y.max(), 5.0, 1e-12 );

  // rap(A, P)
  shared::ParSparseMat ResMid = shared::ParSparseMat::rap( A, P );
  y = ResMid * x;
  EXPECT_NEAR( y.max(), 5.0, 1e-12 );

  // rap(R, A, P)
  shared::ParSparseMat Res2 = shared::ParSparseMat::rap( R, A, P );
  y = Res2 * x;
  EXPECT_NEAR( y.max(), 5.0, 1e-12 );
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
  shared::ParSparseMat A = shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, size, row_starts, 1.0 );

  // get()
  EXPECT_EQ( A.height(), local_size );

  // operator->
  EXPECT_EQ( A->Height(), local_size );
}

// Test Construction from Vector
TEST_F( ParSparseMatTest, DiagonalFromVector )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Construction from Vector..." << std::endl;

  int num_procs;
  MPI_Comm_size( MPI_COMM_WORLD, &num_procs );
  constexpr int size = 10;
  int local_size = size / num_procs + ( rank < ( size % num_procs ) ? 1 : 0 );

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, size );

  mfem::Vector diag_vals( local_size );
  for ( int i = 0; i < local_size; ++i ) {
    diag_vals[i] = static_cast<double>( rank * 100 + i );
  }

  shared::ParSparseMat A =
      shared::ParSparseMat::diagonalMatrix( MPI_COMM_WORLD, size, row_starts.GetData(), diag_vals );

  shared::ParVector x( A.get(), 0 );
  x.fill( 1.0 );
  auto y = A * x;

  for ( int i = 0; i < local_size; ++i ) {
    EXPECT_NEAR( y[i], static_cast<double>( rank * 100 + i ), 1e-12 );
  }
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
