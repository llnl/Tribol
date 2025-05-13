// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <iostream>

// gtest includes
#include "gtest/gtest.h"

// Tribol includes
#include "tribol/common/Memory.hpp"
#include "tribol/common/Containers.hpp"

/*!
 *  Test fixture class to test container classes
 */
class ContainerTest : public ::testing::Test {
 public:
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
  std::cout << "Running ArrayBase test with AllocatedMemory and a HeapAllocator of fixed size..." << std::endl;

  using MemT = tribol::AllocatedMemory<int, tribol::HeapAllocator<int>, tribol::IndexT,
                                       tribol::SizeEqCapacity<tribol::FixedCapacity<tribol::IndexT, 10>>>;

  // create ArrayBase object with AllocatedMemory and a HeapAllocator of fixed size
  // AllocatedMemory with a HeapAllocator is a fixed size array on the heap
  auto array_base = tribol::ArrayBase<MemT>( MemT( 10 ) );
  for ( int i = 0; i < 10; ++i ) {
    array_base[i] = i;
  }
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base[i], i );
  }

  // copy ArrayBase object
  // this will create a new HeapAllocator object and copy the data
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
  std::cout << "Running ArrayBase test with AllocatedMemory and a HeapAllocator sized at runtime..." << std::endl;

  using MemT = tribol::AllocatedMemory<int, tribol::HeapAllocator<int>, tribol::IndexT,
                                       tribol::SizeEqCapacity<tribol::RuntimeCapacity<tribol::IndexT>>>;

  // create ArrayBase object with AllocatedMemory and a HeapAllocator of dynamic size
  // AllocatedMemory with a HeapAllocator is a dynamic size array on the heap
  auto array_base = tribol::ArrayBase<MemT>( MemT( 10 ) );
  for ( int i = 0; i < 10; ++i ) {
    array_base[i] = i;
  }
  for ( int i = 0; i < 10; ++i ) {
    EXPECT_EQ( array_base[i], i );
  }

  // copy ArrayBase object
  // this will create a new HeapAllocator object and copy the data
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

  using MemT = tribol::AllocatedMemory<int, tribol::UmpireAllocator<int, tribol::MemorySpace::Host>, tribol::IndexT,
                                       tribol::SizeEqCapacity<tribol::FixedCapacity<tribol::IndexT, 10>>>;

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

  using MemT = tribol::AllocatedMemory<int, tribol::UmpireAllocator<int, tribol::MemorySpace::Host>, tribol::IndexT,
                                       tribol::SizeEqCapacity<tribol::RuntimeCapacity<tribol::IndexT>>>;

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
  // AllocatedMemory with a HeapAllocator is a runtime sized array on the heap
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

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
