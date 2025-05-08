// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// gtest includes
#include "gtest/gtest.h"

// Tribol includes
#include "tribol/common/Containers.hpp"
#include "tribol/common/LoopExec.hpp"

/*!
 *  Test fixture class to test memory functions
 */
class MemoryTest : public ::testing::Test {
 public:
 protected:
};

TEST_F( MemoryTest, sort_and_search )
{
  auto ten_ints = tribol::AllocatedMemory<int, tribol::MemorySpace::Host>( 10 );
  tribol::Memory<int> ten_ints_mem( ten_ints );
  tribol::forAllExec<tribol::ExecutionMode::Sequential>( ten_ints.size(), [=]( int i ) {
    auto int_array =
        tribol::BoundedArray<int, 1, tribol::PoolAllocation>( tribol::PoolAllocation<int, 1>( ten_ints_mem, i ) );
    int_array.push_back( i );
  } );
#ifdef TRIBOL_USE_CUDA
  auto ten_ints_cuda = tribol::AllocatedMemory<int, tribol::MemorySpace::Device>( 10 );
  tribol::Memory<int> ten_ints_cuda_mem( ten_ints_device );
  tribol::forAllExec<tribol::ExecutionMode::Cuda>( ten_ints_cuda.size(), [=]( int i ) {
    auto int_array =
        tribol::BoundedArray<int, 1, tribol::PoolAllocation>( tribol::PoolAllocation<int, 1>( ten_ints_cuda_mem, i ) );
    int_array.push_back( i );
  } );
#endif
#ifdef TRIBOL_USE_HIP
  auto ten_ints_hip = tribol::AllocatedMemory<int, tribol::MemorySpace::Device>( 10 );
  tribol::Memory<int> ten_ints_hip_mem( ten_ints_hip );
  tribol::forAllExec<tribol::ExecutionMode::Hip>( ten_ints_hip.size(), [=]( int i ) {
    auto int_array =
        tribol::BoundedArray<int, 1, tribol::PoolAllocation>( tribol::PoolAllocation<int, 1>( ten_ints_hip_mem, i ) );
    int_array.push_back( i );
  } );
#endif
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
