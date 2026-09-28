#ifndef TRIBOL_TESTS_ALLOCATIONCOUNTER_HPP_
#define TRIBOL_TESTS_ALLOCATIONCOUNTER_HPP_

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <new>

namespace tribol::test {

inline std::atomic<std::size_t> allocation_count{};
inline std::atomic<bool> count_allocations{};

inline void resetAllocationCount() { allocation_count.store( 0, std::memory_order_relaxed ); }

inline void startAllocationCounting() { count_allocations.store( true, std::memory_order_relaxed ); }

inline void stopAllocationCounting() { count_allocations.store( false, std::memory_order_relaxed ); }

inline std::size_t allocationCount() { return allocation_count.load( std::memory_order_relaxed ); }

inline void recordAllocation()
{
  if ( count_allocations.load( std::memory_order_relaxed ) ) {
    allocation_count.fetch_add( 1, std::memory_order_relaxed );
  }
}

}  // namespace tribol::test

void* operator new( std::size_t size )
{
  tribol::test::recordAllocation();
  if ( void* storage = std::malloc( std::max<std::size_t>( size, 1 ) ) ) {
    return storage;
  }
  throw std::bad_alloc{};
}

void* operator new[]( std::size_t size )
{
  tribol::test::recordAllocation();
  if ( void* storage = std::malloc( std::max<std::size_t>( size, 1 ) ) ) {
    return storage;
  }
  throw std::bad_alloc{};
}

void* operator new( std::size_t size, std::align_val_t alignment )
{
  tribol::test::recordAllocation();
  void* storage{};
  if ( ::posix_memalign( &storage, static_cast<std::size_t>( alignment ), std::max<std::size_t>( size, 1 ) ) == 0 ) {
    return storage;
  }
  throw std::bad_alloc{};
}

void* operator new[]( std::size_t size, std::align_val_t alignment ) { return ::operator new( size, alignment ); }

void operator delete( void* storage ) noexcept { std::free( storage ); }
void operator delete[]( void* storage ) noexcept { std::free( storage ); }
void operator delete( void* storage, std::size_t ) noexcept { std::free( storage ); }
void operator delete[]( void* storage, std::size_t ) noexcept { std::free( storage ); }
void operator delete( void* storage, std::align_val_t ) noexcept { std::free( storage ); }
void operator delete[]( void* storage, std::align_val_t ) noexcept { std::free( storage ); }
void operator delete( void* storage, std::size_t, std::align_val_t ) noexcept { std::free( storage ); }
void operator delete[]( void* storage, std::size_t, std::align_val_t ) noexcept { std::free( storage ); }

#endif
