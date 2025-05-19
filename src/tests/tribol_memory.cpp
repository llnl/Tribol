// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <iostream>

// gtest includes
#include "gtest/gtest.h"

// Tribol includes
#include "tribol/common/ExecModel.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/common/Memory.hpp"
#include "tribol/common/Arrays.hpp"

/*!
 *  Test fixture class to test memory functions
 */
class MemoryTest : public ::testing::Test {
 public:
  constexpr static int N = 100000;

  using SizeVsCapacityT = tribol::SizeLECapacity<tribol::RuntimeCapacity>;

  template <tribol::MemorySpace MSPACE, tribol::ExecutionMode EXMODE>
  void RunFakeInitTest()
  {
    tribol::forAllExec<EXMODE>( 1, [=] TRIBOL_HOST_DEVICE( int i ) {
      auto int_array = tribol::BoundedArray<int, tribol::StackMemory<int, 1, tribol::SizeLECapacity>>( 0, 1 );
      int_array.push_back( i );
      tribol::FixedArray<int, 1, tribol::StackMemory<int, 1>> fixed_int_array;
      fixed_int_array[0] = 1;
    } );
  }

  template <tribol::MemorySpace MSPACE, tribol::ExecutionMode EXMODE>
  void RunStackTest()
  {
    tribol::forAllExec<EXMODE>( N, [=] TRIBOL_HOST_DEVICE( int i ) {
      auto int_array = tribol::BoundedArray<int, tribol::StackMemory<int, 1, tribol::SizeLECapacity>>( 0, 1 );
      int_array.push_back( i );
      tribol::FixedArray<int, 1, tribol::StackMemory<int, 1>> fixed_int_array;
      fixed_int_array[0] = 1;
    } );
  }

  template <tribol::MemorySpace MSPACE, tribol::ExecutionMode EXMODE>
  void RunHeapTest()
  {
    tribol::forAllExec<EXMODE>( N, [=] TRIBOL_HOST_DEVICE( int i ) {
      auto int_array =
          tribol::BoundedArray<int, tribol::AllocatedMemory<int, tribol::Allocator<int>, SizeVsCapacityT>>( 0, 1 );
      int_array.push_back( i );
      tribol::FixedArray<int, 1, tribol::AllocatedMemory<int, tribol::Allocator<int>, SizeVsCapacityT>> fixed_int_array;
      fixed_int_array[0] = 1;
    } );
  }

  template <tribol::MemorySpace MSPACE, tribol::ExecutionMode EXMODE>
  void RunPoolTest()
  {
    using MemT = typename tribol::AllocatedMemory<int,
#ifdef TRIBOL_USE_UMPIRE
                                                  tribol::UmpireAllocator<int, MSPACE>>
#else
                                                  tribol::Allocator<int>>
#endif
        ::view_type;

    // typical memory allocated for an array
    auto int_pool = tribol::AllocatedMemory<int,
#ifdef TRIBOL_USE_UMPIRE
                                            tribol::UmpireAllocator<int, MSPACE>>
#else
                                            tribol::Allocator<int>>
#endif
        ( 2 * N );
    MemT int_pool_view = int_pool.view();
    tribol::forAllExec<EXMODE>( N, [=] TRIBOL_HOST_DEVICE( int i ) {
      auto int_array = tribol::BoundedArray<int, MemT>( MemT( int_pool_view + i, 0, 1, 1 ) );
      int_array.push_back( i );
      tribol::FixedArray<int, 1, MemT> fixed_int_array( MemT( int_pool_view + i, 1 ) );
      fixed_int_array[0] = 1;
    } );
  }

 protected:
};

TEST_F( MemoryTest, fake_init )
{
#ifdef TRIBOL_USE_CUDA
  RunFakeInitTest<tribol::MemorySpace::Device, tribol::ExecutionMode::Cuda>();
#endif
#ifdef TRIBOL_USE_HIP
  RunFakeInitTest<tribol::MemorySpace::Device, tribol::ExecutionMode::Hip>();
#endif
}

TEST_F( MemoryTest, stack_test_host )
{
  std::cout << "Running stack test on host..." << std::endl;
  RunStackTest<tribol::MemorySpace::Dynamic, tribol::ExecutionMode::Sequential>();
}

TEST_F( MemoryTest, heap_test_host )
{
  std::cout << "Running heap test on host..." << std::endl;
  RunHeapTest<tribol::MemorySpace::Dynamic, tribol::ExecutionMode::Sequential>();
}

TEST_F( MemoryTest, pool_test )
{
  std::cout << "Running pool test on host..." << std::endl;
  RunPoolTest<tribol::MemorySpace::Dynamic, tribol::ExecutionMode::Sequential>();
}

#ifdef TRIBOL_USE_CUDA
TEST_F( MemoryTest, stack_test_cuda )
{
  std::cout << "Running stack test on CUDA..." << std::endl;
  RunStackTest<tribol::MemorySpace::Device, tribol::ExecutionMode::Cuda>();
}

TEST_F( MemoryTest, heap_test_cuda )
{
  std::cout << "Running heap test on CUDA..." << std::endl;
  RunHeapTest<tribol::MemorySpace::Device, tribol::ExecutionMode::Cuda>();
}

TEST_F( MemoryTest, pool_test_cuda )
{
  std::cout << "Running pool test on CUDA..." << std::endl;
  RunPoolTest<tribol::MemorySpace::Device, tribol::ExecutionMode::Cuda>();
}
#endif

#ifdef TRIBOL_USE_HIP
TEST_F( MemoryTest, stack_test_hip )
{
  std::cout << "Running stack test on HIP..." << std::endl;
  RunStackTest<tribol::MemorySpace::Device, tribol::ExecutionMode::Hip>();
}

TEST_F( MemoryTest, heap_test_hip )
{
  std::cout << "Running heap test on HIP..." << std::endl;
  RunHeapTest<tribol::MemorySpace::Device, tribol::ExecutionMode::Hip>();
}

TEST_F( MemoryTest, pool_test_hip )
{
  std::cout << "Running pool test on HIP..." << std::endl;
  RunPoolTest<tribol::MemorySpace::Device, tribol::ExecutionMode::Hip>();
}
#endif

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
