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

class ParVectorTest : public ::testing::Test {
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
};

// Test Construction
TEST_F( ParVectorTest, Construction )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  int num_procs;
  MPI_Comm_size( MPI_COMM_WORLD, &num_procs );
  constexpr int size = 10;
  int local_size = size / num_procs + ( rank < ( size % num_procs ) ? 1 : 0 );
  if ( rank == 0 ) std::cout << "Testing Construction..." << std::endl;

  auto row_starts_array = GetRowStarts( MPI_COMM_WORLD, size );

  // 1. From mfem::HypreParVector*
  HYPRE_MemoryLocation old_hypre_mem_location;
  HYPRE_GetMemoryLocation( &old_hypre_mem_location );
  HYPRE_SetMemoryLocation( HYPRE_MEMORY_HOST );
  mfem::HypreParVector* v1 = new mfem::HypreParVector( MPI_COMM_WORLD, size, row_starts_array.GetData() );
  HYPRE_SetMemoryLocation( old_hypre_mem_location );
  shared::ParVector pv1( v1 );
  EXPECT_EQ( pv1.size(), local_size );

  // 2. From unique_ptr
  HYPRE_GetMemoryLocation( &old_hypre_mem_location );
  HYPRE_SetMemoryLocation( HYPRE_MEMORY_HOST );
  auto v2 = std::make_unique<mfem::HypreParVector>( MPI_COMM_WORLD, size, row_starts_array.GetData() );
  HYPRE_SetMemoryLocation( old_hypre_mem_location );
  shared::ParVector pv2( std::move( v2 ) );
  EXPECT_EQ( pv2.size(), local_size );

  // 3. Template constructor
  shared::ParVector pv3( MPI_COMM_WORLD, size, row_starts_array.GetData() );
  EXPECT_EQ( pv3.size(), local_size );
}

// Test View
TEST_F( ParVectorTest, View )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing View..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParVector v( MPI_COMM_WORLD, 10, row_starts.GetData() );
  v.fill( 1.0 );

  // Construct View
  shared::ParVectorView view( &v.get() );

  EXPECT_EQ( view.size(), v.size() );
  EXPECT_NEAR( view.max(), 1.0, 1e-12 );

  // Operate on View
  shared::ParVector v2 = view * 2.0;
  EXPECT_NEAR( v2.max(), 2.0, 1e-12 );
}

// Test Accessors
TEST_F( ParVectorTest, Accessors )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Accessors..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParVector v( MPI_COMM_WORLD, 10, row_starts.GetData() );
  v.fill( 0.0 );
  if ( v.size() > 0 ) {
    v[0] = 5.0;
    EXPECT_NEAR( v[0], 5.0, 1e-12 );
    EXPECT_NEAR( ( *v.operator->() )[0], 5.0, 1e-12 );
  }

  v.fill( 3.0 );
  EXPECT_NEAR( v.max(), 3.0, 1e-12 );
  EXPECT_NEAR( v.min(), 3.0, 1e-12 );
}

// Test Addition and Subtraction
TEST_F( ParVectorTest, AddSub )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Addition and Subtraction..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParVector v1( MPI_COMM_WORLD, 10, row_starts.GetData() );
  shared::ParVector v2( MPI_COMM_WORLD, 10, row_starts.GetData() );
  v1.fill( 2.0 );
  v2.fill( 3.0 );

  // v1 + v2
  shared::ParVector v3 = v1 + v2;
  EXPECT_NEAR( v3.max(), 5.0, 1e-12 );

  // v1 += v2
  v1 += v2;
  EXPECT_NEAR( v1.max(), 5.0, 1e-12 );

  // v1 - v2
  shared::ParVector v4 = v1 - v2;
  EXPECT_NEAR( v4.max(), 2.0, 1e-12 );

  // v1 -= v2
  v1 -= v2;
  EXPECT_NEAR( v1.max(), 2.0, 1e-12 );
}

// Test Scalar Multiplication
TEST_F( ParVectorTest, ScalarMult )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Scalar Multiplication..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParVector v( MPI_COMM_WORLD, 10, row_starts.GetData() );
  v.fill( 2.0 );

  // v * s
  shared::ParVector v2 = v * 3.0;
  EXPECT_NEAR( v2.max(), 6.0, 1e-12 );

  // s * v
  shared::ParVector v3 = 4.0 * v;
  EXPECT_NEAR( v3.max(), 8.0, 1e-12 );

  // v *= s
  v *= 5.0;
  EXPECT_NEAR( v.max(), 10.0, 1e-12 );
}

// Test Component-wise Multiplication and Division
TEST_F( ParVectorTest, ComponentWise )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Component-wise operations..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParVector v1( MPI_COMM_WORLD, 10, row_starts.GetData() );
  shared::ParVector v2( MPI_COMM_WORLD, 10, row_starts.GetData() );
  v1.fill( 2.0 );
  v2.fill( 4.0 );

  // multiply
  shared::ParVector v3 = v1.multiply( v2 );
  EXPECT_NEAR( v3.max(), 8.0, 1e-12 );

  // multiply in-place
  v1.multiplyInPlace( v2 );
  EXPECT_NEAR( v1.max(), 8.0, 1e-12 );

  // divide
  shared::ParVector v4 = v1.divide( v2 );
  EXPECT_NEAR( v4.max(), 2.0, 1e-12 );

  // divide in-place
  v1.divideInPlace( v2 );
  EXPECT_NEAR( v1.max(), 2.0, 1e-12 );
}

// Test Inverse
TEST_F( ParVectorTest, Inverse )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Inverse..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParVector v1( MPI_COMM_WORLD, 10, row_starts.GetData() );
  v1.fill( 2.0 );

  // inverse
  shared::ParVector v2 = v1.inverse();
  EXPECT_NEAR( v2.max(), 0.5, 1e-12 );
  EXPECT_NEAR( v2.min(), 0.5, 1e-12 );

  // inverse in-place
  v1.inverseInPlace();
  EXPECT_NEAR( v1.max(), 0.5, 1e-12 );
  EXPECT_NEAR( v1.min(), 0.5, 1e-12 );

  // Test with zero and tolerance
  shared::ParVector v3( MPI_COMM_WORLD, 10, row_starts.GetData() );
  v3.fill( 0.0 );
  v3.inverseInPlace( 1e-14 );
  EXPECT_NEAR( v3.max(), 0.0, 1e-12 );
}

// Test Move and Release
TEST_F( ParVectorTest, MoveAndRelease )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Move and Release..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParVector v1( MPI_COMM_WORLD, 10, row_starts.GetData() );
  v1.fill( 7.0 );

  // Move constructor
  shared::ParVector v2( std::move( v1 ) );
  EXPECT_NEAR( v2.max(), 7.0, 1e-12 );
  EXPECT_EQ( v1.operator->(), nullptr );

  // Move assignment
  shared::ParVector v3( MPI_COMM_WORLD, 10, row_starts.GetData() );
  v3 = std::move( v2 );
  EXPECT_NEAR( v3.max(), 7.0, 1e-12 );
  EXPECT_EQ( v2.operator->(), nullptr );

  // Release
  mfem::HypreParVector* raw = v3.release();
  EXPECT_NE( raw, nullptr );
  EXPECT_NEAR( raw->Max(), 7.0, 1e-12 );
  delete raw;
}

// Test Fill
TEST_F( ParVectorTest, Fill )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Fill..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParVector v( MPI_COMM_WORLD, 10, row_starts.GetData() );

  v.fill( 1.0 );
  EXPECT_NEAR( v.max(), 1.0, 1e-12 );
  EXPECT_NEAR( v.min(), 1.0, 1e-12 );

  v.fill( 2.5 );
  EXPECT_NEAR( v.max(), 2.5, 1e-12 );
  EXPECT_NEAR( v.min(), 2.5, 1e-12 );
}

// Test Copy
TEST_F( ParVectorTest, Copy )
{
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) std::cout << "Testing Copy..." << std::endl;

  auto row_starts = GetRowStarts( MPI_COMM_WORLD, 10 );
  shared::ParVector v1( MPI_COMM_WORLD, 10, row_starts.GetData() );
  v1.fill( 3.0 );

  // Copy constructor
  shared::ParVector v2( v1 );
  EXPECT_NEAR( v2.max(), 3.0, 1e-12 );

  // Verify it's a deep copy
  v1.fill( 4.0 );
  EXPECT_NEAR( v1.max(), 4.0, 1e-12 );
  EXPECT_NEAR( v2.max(), 3.0, 1e-12 );

  // Copy assignment
  shared::ParVector v3( MPI_COMM_WORLD, 10, row_starts.GetData() );
  v3.fill( 5.0 );
  v3 = v2;
  EXPECT_NEAR( v3.max(), 3.0, 1e-12 );

  // Verify deep copy for assignment
  v2.fill( 6.0 );
  EXPECT_NEAR( v2.max(), 6.0, 1e-12 );
  EXPECT_NEAR( v3.max(), 3.0, 1e-12 );
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
