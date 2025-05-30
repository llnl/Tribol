// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <iostream>

// gtest includes
#include "gtest/gtest.h"

// Tribol includes
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/ExecModel.hpp"
#include "tribol/common/Memory.hpp"
#include "tribol/common/Arrays.hpp"
#include "tribol/common/LoopExec.hpp"

/*!
 *  Test fixture class to test container classes
 */
class ContainerTest : public ::testing::Test {
 public:
  template <tribol::ExecutionMode EXMODE>
  void BoundedArray2DTest() const
  {
    tribol::forAllExec<EXMODE>( 1, [] TRIBOL_HOST_DEVICE( int ) {
      auto bounded_array_2d = tribol::BoundedArray2D<int>( 0, 3, 20 );

      int count = 0;
      for ( int i{ 0 }; i < 20; ++i ) {
        bounded_array_2d.push_back( { count, count + 1, count + 2 } );
        count += 3;
#ifndef TRIBOL_DEVICE_CODE
        EXPECT_EQ( bounded_array_2d.size(), count );
#else
        assert( bounded_array_2d.size() == count );
#endif
        // printf( "size = %d\n", static_cast<int>( bounded_array_2d.size() ) );
      }

      // test base class accessor
      count = 0;
      for ( auto& array_val : bounded_array_2d ) {
#ifndef TRIBOL_DEVICE_CODE
        EXPECT_EQ( array_val, count++ );
#else
        assert( array_val == count++ );
#endif
      }

      // verify row view values
      for ( int i{ 0 }; i < 10; ++i ) {
        auto row_view = bounded_array_2d.rowView( i );
        for ( int j{ 0 }; j < 3; ++j ) {
#ifndef TRIBOL_DEVICE_CODE
          EXPECT_EQ( row_view[j], i * 3 + j );
#else
          assert( row_view[j] == i * 3 + j );
#endif
        }
      }

      // verify column view values
      for ( int j{ 0 }; j < 3; ++j ) {
        auto col_view = bounded_array_2d.colView( j );
        for ( int i{ 0 }; i < 10; ++i ) {
#ifndef TRIBOL_DEVICE_CODE
          EXPECT_EQ( col_view[i], i * 3 + j );
#else
          assert( col_view[i] == i * 3 + j );
#endif
        }
      }
    } );
#ifdef TRIBOL_USE_CUDA
    if constexpr ( EXMODE == tribol::ExecutionMode::Cuda ) {
      RAJA::synchronize<RAJA::cuda_synchronize>();
    }
#endif
  }

  template <tribol::ExecutionMode EXMODE>
  void ManagedArrayTest( int allocator_id )
  {
    auto base_array = tribol::Array<int>( 10, 10 );
    int count = 0;
    for ( auto& val : base_array ) {
      val = count++;
    }

    auto managed_array = tribol::ManagedArray<int>( std::move( base_array ), allocator_id );

    auto& host_read = managed_array.hostRead();
    for ( size_t i = 0; i < host_read.size(); ++i ) {
      EXPECT_EQ( host_read[i], i );
    }

    EXPECT_EQ( managed_array.sameArray(), tribol::getDefaultAllocatorID() == allocator_id );

    auto managed_write = managed_array.managedWrite();
    if ( managed_array.sameArray() ) {
      EXPECT_EQ( managed_write.memory().data(), host_read.memory().data() );
    }
    tribol::forAllExec<EXMODE>( 10, [=] TRIBOL_HOST_DEVICE( int i ) mutable { managed_write[i] = 2 * i; } );

    // changes haven't been synced back if the arrays are different
    for ( size_t i = 0; i < host_read.size(); ++i ) {
      if ( managed_array.sameArray() ) {
        EXPECT_EQ( host_read[i], 2 * i );
      } else {
        EXPECT_EQ( host_read[i], i );
      }
    }

    // call hostRead() again to synchronize changes to host_read
    managed_array.hostRead();
    for ( size_t i = 0; i < host_read.size(); ++i ) {
      EXPECT_EQ( host_read[i], 2 * i );
    }

    auto& host_write = managed_array.hostWrite( true );
    host_write.resize( 0 );
    for ( size_t i = 0; i < 20; ++i ) {
      host_write.push_back( 3 * i );
    }

    auto managed_readwrite = managed_array.managedReadWrite();
    tribol::forAllExec<EXMODE>( 20, [=] TRIBOL_HOST_DEVICE( int i ) mutable {
#ifdef TRIBOL_DEVICE_CODE
      assert( managed_readwrite[i] == 3 * i );
      printf( "array[%d] = %d vs. %d expected\n", static_cast<int>( i ), managed_readwrite[i],
              static_cast<int>( 3 * i ) );
#else
      EXPECT_EQ( managed_readwrite[i], 3 * i );
#endif
    } );

#ifdef TRIBOL_USE_CUDA
    if constexpr ( EXMODE == tribol::ExecutionMode::Cuda ) {
      RAJA::synchronize<RAJA::cuda_synchronize>();
    }
#endif

    auto base_array_2d = tribol::Array2D<int>( 10, 3, 10 );
    count = 0;
    for ( auto& val : base_array_2d ) {
      val = count++;
    }

    auto managed_array_2d = tribol::ManagedArray<int, tribol::Array2D>( std::move( base_array_2d ), allocator_id );

    auto& host_read_2d = managed_array_2d.hostRead();
    for ( size_t i = 0; i < host_read_2d.height(); ++i ) {
      for ( size_t j = 0; j < host_read_2d.width(); ++j ) {
        EXPECT_EQ( host_read_2d( i, j ), i * host_read_2d.width() + j );
      }
    }

    EXPECT_EQ( managed_array_2d.sameArray(), tribol::getDefaultAllocatorID() == allocator_id );

    auto managed_write_2d = managed_array_2d.managedWrite();
    if ( managed_array_2d.sameArray() ) {
      EXPECT_EQ( managed_write_2d.memory().data(), host_read_2d.memory().data() );
    }
    tribol::forAllExec<EXMODE>( 30, [=] TRIBOL_HOST_DEVICE( int i ) mutable { managed_write_2d[i] = 2 * i; } );

    // changes haven't been synced back if the arrays are different
    for ( size_t i = 0; i < host_read_2d.size(); ++i ) {
      if ( managed_array_2d.sameArray() ) {
        EXPECT_EQ( host_read_2d[i], 2 * i );
      } else {
        EXPECT_EQ( host_read_2d[i], i );
      }
    }

    // call hostRead() again to synchronize changes to host_read
    managed_array_2d.hostRead();
    for ( size_t i = 0; i < host_read_2d.size(); ++i ) {
      EXPECT_EQ( host_read_2d[i], 2 * i );
    }

    auto& host_write_2d = managed_array_2d.hostWrite( true );
    for ( int i = 10; i < 20; ++i ) {
      host_write_2d.push_back( { 2 * 3 * i, 2 * 3 * i + 2, 2 * 3 * i + 4 } );
    }

    auto managed_readwrite_2d = managed_array_2d.managedReadWrite();
    tribol::forAllExec<EXMODE>( 60, [=] TRIBOL_HOST_DEVICE( int i ) mutable {
#ifdef TRIBOL_DEVICE_CODE
      assert( managed_readwrite_2d[i] == 2 * i );
      printf( "array[%d] = %d vs. %d expected\n", static_cast<int>( i ), managed_readwrite_2d[i],
              static_cast<int>( 2 * i ) );
#else
      EXPECT_EQ( managed_readwrite_2d[i], 2 * i );
#endif
    } );

#ifdef TRIBOL_USE_CUDA
    if constexpr ( EXMODE == tribol::ExecutionMode::Cuda ) {
      RAJA::synchronize<RAJA::cuda_synchronize>();
    }
#endif
  }

 protected:
};

TEST_F( ContainerTest, array_base_stackmemory )
{
  std::cout << "Running ArrayBase test with StackMemory..." << std::endl;

  // create ArrayBase object with StackMemory
  // StackMemory is a fixed size array on the stack
  auto array_base = tribol::ArrayBase<tribol::StackMemory<int, 10>>( tribol::StackMemory<int, 10>() );
  for ( int i = 0; i < 10; ++i ) {
    array_base[i] = i;
  }
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base[i], i );
  }

  // copy ArrayBase object
  // this will create a new StackMemory object and copy the data
  auto array_base_copy = tribol::ArrayBase<tribol::StackMemory<int, 10>>( array_base );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_copy[i], i );
  }
  // add 1 to the copy
  for ( int i = 0; i < 10; ++i ) {
    array_base_copy[i] += 1;
  }
  // check that the copy is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_copy[i], i + 1 );
  }
  // make sure the copy is at a new address
  EXPECT_NE( array_base.memory().data(), array_base_copy.memory().data() );
  // check that the original is unchanged
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base[i], i );
  }

  // move ArrayBase object
  // this is the same as a copy, since the StackMemory is tied to the object
  auto array_base_move = tribol::ArrayBase<tribol::StackMemory<int, 10>>( std::move( array_base ) );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_move[i], i );
  }
  // add 1 to the moved object
  for ( int i = 0; i < 10; ++i ) {
    array_base_move[i] += 1;
  }
  // check that the moved object is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_move[i], i + 1 );
  }
  // verify the moved data is at a different address
  EXPECT_NE( array_base.memory().data(), array_base_move.memory().data() );
  // verify the capacity of the original object
  EXPECT_EQ( array_base.memory().capacity(), 10 );
}

TEST_F( ContainerTest, array_base_allocatedmemory_heapallocator_fixedsize )
{
  std::cout << "Running ArrayBase test with AllocatedMemory and an Allocator of fixed size..." << std::endl;

  using MemT = tribol::AllocatedMemory<int, tribol::Allocator<int>, tribol::SizeEqCapacity<tribol::FixedCapacity<10>>>;

  // create ArrayBase object with AllocatedMemory and an Allocator of fixed size
  // AllocatedMemory with an Allocator is a fixed size array on the heap
  auto array_base = tribol::ArrayBase<MemT>( MemT( 10 ) );
  for ( int i = 0; i < 10; ++i ) {
    array_base[i] = i;
  }
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base[i], i );
  }

  // copy ArrayBase object
  // this will create a new Allocator object and copy the data
  auto array_base_copy = tribol::ArrayBase<MemT>( array_base );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_copy[i], i );
  }
  // add 1 to the copy
  for ( int i = 0; i < 10; ++i ) {
    array_base_copy[i] += 1;
  }
  // check that the copy is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_copy[i], i + 1 );
  }
  // make sure the copy is at a new address
  EXPECT_NE( array_base.memory().data(), array_base_copy.memory().data() );
  // check that the original is unchanged
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base[i], i );
  }

  // move ArrayBase object
  // this should move the data to the new object (and allocate new memory for the moved object since the size is fixed)
  auto array_base_move = tribol::ArrayBase<MemT>( std::move( array_base ) );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_move[i], i );
  }
  // add 1 to the moved object
  for ( int i = 0; i < 10; ++i ) {
    array_base_move[i] += 1;
  }
  // check that the moved object is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_move[i], i + 1 );
  }
  // verify the moved data is at a different address
  EXPECT_NE( array_base.memory().data(), array_base_move.memory().data() );
  // verify the capacity of the original object
  EXPECT_EQ( array_base.memory().capacity(), 10 );
}

TEST_F( ContainerTest, array_base_allocatedmemory_heapallocator_dynamicsize )
{
  std::cout << "Running ArrayBase test with AllocatedMemory and an Allocator sized at runtime..." << std::endl;

  using MemT = tribol::AllocatedMemory<int, tribol::Allocator<int>, tribol::SizeEqCapacity<tribol::RuntimeCapacity>>;

  // create ArrayBase object with AllocatedMemory and an Allocator of dynamic size
  // AllocatedMemory with an Allocator is a dynamic size array on the heap
  auto array_base = tribol::ArrayBase<MemT>( MemT( 10 ) );
  for ( int i = 0; i < 10; ++i ) {
    array_base[i] = i;
  }
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base[i], i );
  }

  // copy ArrayBase object
  // this will create a new Allocator object and copy the data
  auto array_base_copy = tribol::ArrayBase<MemT>( array_base );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_copy[i], i );
  }
  // add 1 to the copy
  for ( int i = 0; i < 10; ++i ) {
    array_base_copy[i] += 1;
  }
  // check that the copy is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_copy[i], i + 1 );
  }
  // make sure the copy is at a new address
  EXPECT_NE( array_base.memory().data(), array_base_copy.memory().data() );
  // check that the original is unchanged
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base[i], i );
  }

  // move ArrayBase object
  // this should move the data to the new object (and make the original object empty)
  auto array_base_move = tribol::ArrayBase<MemT>( std::move( array_base ) );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_move[i], i );
  }
  // add 1 to the moved object
  for ( int i = 0; i < 10; ++i ) {
    array_base_move[i] += 1;
  }
  // check that the moved object is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_move[i], i + 1 );
  }
  // verify the moved data is at a different address
  EXPECT_NE( array_base.memory().data(), array_base_move.memory().data() );
  // verify the capacity of the original object
  EXPECT_EQ( array_base.memory().capacity(), 0 );
}

#ifdef TRIBOL_USE_UMPIRE

TEST_F( ContainerTest, array_base_allocatedmemory_umpireallocator_fixedsize )
{
  std::cout << "Running ArrayBase test with AllocatedMemory and a UmpireAllocator of fixed size..." << std::endl;

  using MemT = tribol::AllocatedMemory<int, tribol::UmpireAllocator<int, tribol::MemorySpace::Host>,
                                       tribol::SizeEqCapacity<tribol::FixedCapacity<10>>>;

  // create ArrayBase object with AllocatedMemory and a UmpireAllocator of fixed size
  // AllocatedMemory with a UmpireAllocator is a fixed size array
  auto array_base = tribol::ArrayBase<MemT>( MemT( 10 ) );
  for ( int i = 0; i < 10; ++i ) {
    array_base[i] = i;
  }
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base[i], i );
  }

  // copy ArrayBase object
  // this will create a new UmpireAllocator object and copy the data
  auto array_base_copy = tribol::ArrayBase<MemT>( array_base );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_copy[i], i );
  }
  // add 1 to the copy
  for ( int i = 0; i < 10; ++i ) {
    array_base_copy[i] += 1;
  }
  // check that the copy is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_copy[i], i + 1 );
  }
  // make sure the copy is at a new address
  EXPECT_NE( array_base.memory().data(), array_base_copy.memory().data() );
  // check that the original is unchanged
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base[i], i );
  }

  // move ArrayBase object
  // this should move the data to the new object (and allocate new memory for the moved object since the size is fixed)
  auto array_base_move = tribol::ArrayBase<MemT>( std::move( array_base ) );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_move[i], i );
  }
  // add 1 to the moved object
  for ( int i = 0; i < 10; ++i ) {
    array_base_move[i] += 1;
  }
  // check that the moved object is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_move[i], i + 1 );
  }
  // verify the moved data is at a different address
  EXPECT_NE( array_base.memory().data(), array_base_move.memory().data() );
  // verify the capacity of the original object
  EXPECT_EQ( array_base.memory().capacity(), 10 );
}

TEST_F( ContainerTest, array_base_allocatedmemory_umpireallocator_dynamicsize )
{
  std::cout << "Running ArrayBase test with AllocatedMemory and a UmpireAllocator sized at runtime..." << std::endl;

  using MemT = tribol::AllocatedMemory<int, tribol::UmpireAllocator<int, tribol::MemorySpace::Host>,
                                       tribol::SizeEqCapacity<tribol::RuntimeCapacity>>;

  // create ArrayBase object with AllocatedMemory and a UmpireAllocator of dynamic size
  // AllocatedMemory with a UmpireAllocator is a dynamic size array
  auto array_base = tribol::ArrayBase<MemT>( MemT( 10 ) );
  for ( int i = 0; i < 10; ++i ) {
    array_base[i] = i;
  }
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base[i], i );
  }

  // copy ArrayBase object
  // this will create a new UmpireAllocator object and copy the data
  auto array_base_copy = tribol::ArrayBase<MemT>( array_base );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_copy[i], i );
  }
  // add 1 to the copy
  for ( int i = 0; i < 10; ++i ) {
    array_base_copy[i] += 1;
  }
  // check that the copy is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_copy[i], i + 1 );
  }
  // make sure the copy is at a new address
  EXPECT_NE( array_base.memory().data(), array_base_copy.memory().data() );
  // check that the original is unchanged
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base[i], i );
  }

  // move ArrayBase object
  // this should move the data to the new object (and make the original object empty)
  auto array_base_move = tribol::ArrayBase<MemT>( std::move( array_base ) );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_move[i], i );
  }
  // add 1 to the moved object
  for ( int i = 0; i < 10; ++i ) {
    array_base_move[i] += 1;
  }
  // check that the moved object is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_move[i], i + 1 );
  }
  // verify the moved data is at a different address
  EXPECT_NE( array_base.memory().data(), array_base_move.memory().data() );
  // verify the capacity of the original object
  EXPECT_EQ( array_base.memory().capacity(), 0 );
}

TEST_F( ContainerTest, array_base_memoryview )
{
  std::cout << "Running ArrayBase test with a Memory view from AllocatedMemory..." << std::endl;

  // create ArrayBase object with AllocatedMemory and a UmpireAllocator of fixed size
  // AllocatedMemory with a UmpireAllocator is a fixed size array
  auto array_memory = tribol::AllocatedMemory<int, tribol::UmpireAllocator<int, tribol::MemorySpace::Host>>( 10 );
  // create a Memory view from the AllocatedMemory
  auto memory_view = array_memory.view();
  // create ArrayBase object with the Memory view
  auto array_base = tribol::ArrayBase<decltype( memory_view )>( std::move( memory_view ) );
  for ( int i = 0; i < 10; ++i ) {
    array_base[i] = i;
  }
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base[i], i );
  }

  // copy ArrayBase object
  // this will copy the memory view's pointer and size (shallow copy)
  auto array_base_copy = tribol::ArrayBase<decltype( memory_view )>( array_base );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_copy[i], i );
  }
  // verify the copy is at the same address
  EXPECT_EQ( array_base.memory().data(), array_base_copy.memory().data() );
  // add 1 to the copy
  for ( int i = 0; i < 10; ++i ) {
    array_base_copy[i] += 1;
  }
  // check that the copy is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_copy[i], i + 1 );
  }
  // check that the original also changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base[i], i + 1 );
  }
  // subtract 1 from the original
  for ( int i = 0; i < 10; ++i ) {
    array_base[i] -= 1;
  }

  // move ArrayBase object
  // this should also just shallow copy the memory view's pointer and size
  auto array_base_move = tribol::ArrayBase<decltype( memory_view )>( std::move( array_base ) );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_move[i], i );
  }
  // verify the moved data is at the same address
  EXPECT_EQ( array_base.memory().data(), array_base_move.memory().data() );
  // add 1 to the moved object
  for ( int i = 0; i < 10; ++i ) {
    array_base_move[i] += 1;
  }
  // check that the moved object is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base_move[i], i + 1 );
  }
  // verify the capacity of the original object
  EXPECT_EQ( array_base.memory().capacity(), 10 );
}

#endif

TEST_F( ContainerTest, fixedarray )
{
  std::cout << "Running FixedArray test with default memory (StackMemory)..." << std::endl;

  // create FixedArray object with default memory (StackMemory)
  // StackMemory is a fixed size array on the stack
  auto fixed_array = tribol::FixedArray<int, 10>();
  for ( int i = 0; i < 10; ++i ) {
    fixed_array[i] = i;
  }
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( fixed_array[i], i );
  }

  // test calling memory methods
  EXPECT_EQ( fixed_array.memory().capacity(), 10 );
  EXPECT_NE( fixed_array.memory().data(), nullptr );

  // test iterators
  for ( auto it = fixed_array.begin(); it != fixed_array.end(); ++it ) {
    EXPECT_EQ( *it, it - fixed_array.begin() );
  }
  for ( auto& array_val : fixed_array ) {
    EXPECT_EQ( array_val, &array_val - fixed_array.begin() );
  }

  // test copy constructor
  auto fixed_array_copy = tribol::FixedArray<int, 10>( fixed_array );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( fixed_array_copy[i], i );
  }
  // add 1 to the copy
  for ( int i = 0; i < 10; ++i ) {
    fixed_array_copy[i] += 1;
  }
  // check that the copy is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( fixed_array_copy[i], i + 1 );
  }
  // make sure the copy is at a new address
  EXPECT_NE( fixed_array.memory().data(), fixed_array_copy.memory().data() );
  // check that the original is unchanged
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( fixed_array[i], i );
  }

  // test move constructor (should be the same as copy constructor since we are using StackMemory)
  auto fixed_array_move = tribol::FixedArray<int, 10>( std::move( fixed_array ) );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( fixed_array_move[i], i );
  }
  // add 1 to the moved object
  for ( int i = 0; i < 10; ++i ) {
    fixed_array_move[i] += 1;
  }
  // check that the moved object is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( fixed_array_move[i], i + 1 );
  }
  // verify the moved data is at a different address
  EXPECT_NE( fixed_array.memory().data(), fixed_array_move.memory().data() );
  // verify the capacity of the original object
  EXPECT_EQ( fixed_array.memory().capacity(), 10 );

  // initialize a new FixedArray
  auto fixed_array2 = tribol::FixedArray<int, 10>();
  // verify all values are 0
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( fixed_array2[i], 0 );
  }
}

TEST_F( ContainerTest, boundedarray )
{
  std::cout << "Running BoundedArray test with default memory (heap AllocatedMemory)..." << std::endl;

  // create BoundedArray object with default memory (heap AllocatedMemory)
  // AllocatedMemory with an Allocator is a runtime sized array on the heap
  auto bounded_array = tribol::BoundedArray<int>( 0, 10 );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( bounded_array.size(), i );
    bounded_array.push_back( i );
  }
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( bounded_array[i], i );
  }
  // reset the size to 0
  bounded_array.resize( 0 );
  // fill using emplace_back
  for ( int i = 0; i < 10; ++i ) {
    bounded_array.emplace_back( i );
  }
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( bounded_array[i], i );
  }

  // test some methods
  EXPECT_EQ( bounded_array.capacity(), 10 );
  EXPECT_NE( bounded_array.memory().data(), nullptr );

  // test iterators
  for ( auto it = bounded_array.begin(); it != bounded_array.end(); ++it ) {
    EXPECT_EQ( *it, it - bounded_array.begin() );
  }
  for ( auto& array_val : bounded_array ) {
    EXPECT_EQ( array_val, &array_val - bounded_array.begin() );
  }

  // test copy constructor
  auto bounded_array_copy = tribol::BoundedArray<int>( bounded_array );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( bounded_array_copy[i], i );
  }
  // add 1 to the copy
  for ( int i = 0; i < 10; ++i ) {
    bounded_array_copy[i] += 1;
  }
  // check that the copy is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( bounded_array_copy[i], i + 1 );
  }
  // make sure the copy is at a new address
  EXPECT_NE( bounded_array.memory().data(), bounded_array_copy.memory().data() );
  // check that the original is unchanged
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( bounded_array[i], i );
  }

  // test move constructor
  auto bounded_array_move = tribol::BoundedArray<int>( std::move( bounded_array ) );
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( bounded_array_move[i], i );
  }
  // add 1 to the moved object
  for ( int i = 0; i < 10; ++i ) {
    bounded_array_move[i] += 1;
  }
  // check that the moved object is changed
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( bounded_array_move[i], i + 1 );
  }
  // verify the moved data is at a different address
  EXPECT_NE( bounded_array.memory().data(), bounded_array_move.memory().data() );
  // verify the capacity of the original object
  EXPECT_EQ( bounded_array.capacity(), 0 );
  // verify the address of the original object
  EXPECT_EQ( bounded_array.memory().data(), nullptr );

  // initialize a new BoundedArray
  auto bounded_array2 = tribol::BoundedArray<int>( 10, 10 );
  // verify all values are 0
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( bounded_array2[i], 0 );
  }
}

// strided bounded array on host test
TEST_F( ContainerTest, strided_boundedarray )
{
  std::cout << "Running strided BoundedArray test with default memory (heap AllocatedMemory)..." << std::endl;

  auto bounded_array_2d = tribol::BoundedArray2D<int>( 10, 3 );

  int count = 0;
  for ( int i{ 0 }; i < 10; ++i ) {
    for ( int j{ 0 }; j < 3; ++j ) {
      bounded_array_2d( i, j ) = count++;
    }
  }

  // test base class accessor
  count = 0;
  for ( auto& array_val : bounded_array_2d ) {
    EXPECT_EQ( array_val, count++ );
  }

  // verify row view values
  for ( int i{ 0 }; i < 10; ++i ) {
    auto row_view = bounded_array_2d.rowView( i );
    for ( int j{ 0 }; j < 3; ++j ) {
      EXPECT_EQ( row_view[j], i * 3 + j );
    }
  }

  // verify column view values
  for ( int j{ 0 }; j < 3; ++j ) {
    auto col_view = bounded_array_2d.colView( j );
    for ( int i{ 0 }; i < 10; ++i ) {
      EXPECT_EQ( col_view[i], i * 3 + j );
    }
  }

  // use row views to set all values to 10
  for ( int i{ 0 }; i < 10; ++i ) {
    auto row_view = bounded_array_2d.rowView( i );
    for ( auto& row_view_val : row_view ) {
      row_view_val = 10;
    }
  }

  // use column views to verify all values are set to 10
  for ( int j{ 0 }; j < 3; ++j ) {
    auto col_view = bounded_array_2d.colView( j );
    for ( auto col_view_val : col_view ) {
      EXPECT_EQ( col_view_val, 10 );
    }
  }
}

// tribol array test
TEST_F( ContainerTest, array )
{
  std::cout << "Running Array test with default memory (heap AllocatedMemory)..." << std::endl;

  // create Array object with default memory (heap AllocatedMemory)
  // AllocatedMemory with an Allocator is a runtime sized array on the heap
  auto tribol_array = tribol::Array<int>( 0, 10 );
  auto orig_capacity = tribol_array.capacity();
  auto orig_address = tribol_array.memory().data();
  // for loop exceeds capacity so test memory reallocation
  for ( int i = 0; i < 20; ++i ) {
    EXPECT_EQ( tribol_array.size(), i );
    tribol_array.push_back( i );
  }
  // verify the values are correct
  for ( int i = 0; i < 20; ++i ) {
    EXPECT_EQ( tribol_array[i], i );
  }
  // make sure the capacity is larger than the original
  EXPECT_GT( tribol_array.capacity(), orig_capacity );
  // make sure the address is different
  EXPECT_NE( tribol_array.memory().data(), orig_address );
}

// bounded array 2d test
TEST_F( ContainerTest, boundedarray2d )
{
  std::cout << "Running 2D BoundedArray test with default memory (heap AllocatedMemory) on host..." << std::endl;
  BoundedArray2DTest<tribol::ExecutionMode::Sequential>();

// repeat on device (where available)
#ifdef TRIBOL_USE_CUDA
  std::cout << "Running 2D BoundedArray test with default memory (heap AllocatedMemory) on device..." << std::endl;
  BoundedArray2DTest<tribol::ExecutionMode::Cuda>();
#endif
#ifdef TRIBOL_USE_HIP
  std::cout << "Running 2D BoundedArray test with default memory (heap AllocatedMemory) on device..." << std::endl;
  BoundedArray2DTest<tribol::ExecutionMode::Hip>();
#endif
}

// tribol array 2d test

// host array test
TEST_F( ContainerTest, hostarray )
{
  std::cout << "Running HostArray test with default memory..." << std::endl;

  auto tribol_host = tribol::HostArray<int, tribol::MemorySpace::Host>( 0, 10 );
  for ( int i = 0; i < 10; ++i ) {
    tribol_host.push_back( i );
  }
}

#ifdef TRIBOL_USE_UMPIRE
// managed array test
TEST_F( ContainerTest, managedarray )
{
  std::cout << "Running ManagedArray test on host..." << std::endl;
  ManagedArrayTest<tribol::ExecutionMode::Sequential>( tribol::getResourceAllocatorID( tribol::MemorySpace::Host ) );

#ifdef TRIBOL_USE_CUDA
  std::cout << "Running ManagedArray test on device..." << std::endl;
  ManagedArrayTest<tribol::ExecutionMode::Cuda>( tribol::getResourceAllocatorID( tribol::MemorySpace::Device ) );
#endif

#ifdef TRIBOL_USE_HIP
  std::cout << "Running ManagedArray test on device..." << std::endl;
  ManagedArrayTest<tribol::ExecutionMode::Hip>( tribol::getResourceAllocatorID( tribol::MemorySpace::Device ) );
#endif
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
