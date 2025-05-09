// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef TRIBOL_COMMON_MEMORY_HPP_
#define TRIBOL_COMMON_MEMORY_HPP_

#include <memory>

#include "tribol/config.hpp"
#include "tribol/common/ExecModel.hpp"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/TypedAllocator.hpp"
#endif

namespace tribol {

// template <typename T>
// class Memory {
//  public:
//   TRIBOL_HOST_DEVICE Memory( const Memory& other ) : data_{ other.data_ }, size_{ other.size_ } {}

//   TRIBOL_HOST_DEVICE T* data() const { return data_; }

//   TRIBOL_HOST_DEVICE IndexT size() const { return size_; }

//  protected:
//   TRIBOL_HOST_DEVICE Memory( T* data, IndexT size ) : data_{ data }, size_{ size } {}

//   T* data_;
//   IndexT size_;
// };

// template <typename T>
// class StdAllocator : public std::allocator<T> {
//  public:
//   static_assert( MSPACE == MemorySpace::Dynamic || MSPACE == MemorySpace::Host,
//                  "StdAllocator only supports Host memory space" );
//   void copy( T* dst, T* src, size_t n ) { memcpy( dst, src, n * sizeof( std::allocator<T>::value_type ) ); }
// };

template <typename T>
class HeapAllocator {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;

  TRIBOL_HOST_DEVICE pointer allocate( size_t n ) { return static_cast<pointer>( malloc( n * sizeof( value_type ) ) ); }

  TRIBOL_HOST_DEVICE void deallocate( pointer p, size_t ) { free( static_cast<void*>( p ) ); }

  TRIBOL_HOST_DEVICE void copy( pointer dst, pointer src, size_t n ) { memcpy( dst, src, n * sizeof( value_type ) ); }
};

#ifdef TRIBOL_USE_UMPIRE
template <typename T, MemorySpace MSPACE>
class UmpireAllocator {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;

  UmpireAllocator( umpire::Allocator allocator ) : allocator_{ std::move( allocator ) } {}
  UmpireAllocator()
      : UmpireAllocator( umpire::ResourceManager::getInstance().getAllocator( getResourceAllocatorID( MSPACE ) ) )
  {
  }

  pointer allocate( size_t n ) { return static_cast<T*>( allocator_.allocate( n ) ); }

  void deallocate( pointer p, size_t n ) { allocator_.deallocate( p, n ); }

  void copy( pointer dst, pointer src, size_t n )
  {
    auto& rm = umpire::ResourceManager::getInstance();
    rm.copy( dst, src, n * sizeof( value_type ) );
  }

 private:
  umpire::TypedAllocator<value_type> allocator_;
};
#endif

// template <typename T, MemorySpace MSPACE,
//           template <typename, MemorySpace> class Allocator =
// #ifdef TRIBOL_USE_UMPIRE
//               UmpireAllocator
// #else
//               StdAllocator
// #endif
//           >
// class AllocatedMemory : public Memory<T> {
//  public:
//   AllocatedMemory( IndexT size, Allocator<T, MSPACE> allocator = Allocator<T, MSPACE>() )
//       : Memory<T>( allocator.allocate( size ), size ), allocator_{ std::move( allocator ) }
//   {
//   }

//   // Constructor from another memory space
//   template <MemorySpace MSPACE2>
//   AllocatedMemory( const AllocatedMemory<T, MSPACE2, Allocator>& other ) : AllocatedMemory( other.size() )
//   {
//     allocator_.copy( Memory<T>::data_, other.data(), other.size() );
//   }

//   ~AllocatedMemory() { allocator_.deallocate( Memory<T>::data_, Memory<T>::size_ ); }

//   // Copy constructor
//   AllocatedMemory( const AllocatedMemory& other ) : AllocatedMemory( other.size_ )
//   {
//     allocator_.copy( Memory<T>::data_, other.data_, other.size_ );
//   }

//   // Move constructor
//   AllocatedMemory( AllocatedMemory&& other ) : Memory<T>( other.data_, other.size_ ), allocator_{ other.allocator_ }
//   {
//     other.data_ = nullptr;
//     other.size_ = 0;
//   }

//   // Copy assignment operator
//   template <MemorySpace MSPACE2>
//   AllocatedMemory& operator=( const AllocatedMemory<T, MSPACE2, Allocator>& other )
//   {
//     if ( this != &other ) {
//       if ( Memory<T>::data_ != nullptr ) {
//         allocator_.deallocate( Memory<T>::data_, Memory<T>::size_ );
//       }
//       Memory<T>::data_ = allocator_.allocate( other.size() );
//       Memory<T>::size_ = other.size();
//       allocator_ = other.getAllocator();
//     }
//     allocator_.copy( Memory<T>::data_, other.data(), other.size() );
//     return *this;
//   }

//   // Move assignment operator
//   AllocatedMemory& operator=( AllocatedMemory&& other )
//   {
//     if ( this != &other ) {
//       if ( Memory<T>::data_ != nullptr ) {
//         allocator_.deallocate( Memory<T>::data_, Memory<T>::size_ );
//       }
//       Memory<T>::data_ = other.data_;
//       Memory<T>::size_ = other.size_;
//       allocator_ = other.allocator_;

//       other.data_ = nullptr;
//       other.size_ = 0;
//     }
//     return *this;
//   }

//   const Allocator<T, MSPACE>& getAllocator() const { return allocator_; }

//  private:
//   Allocator<T, MSPACE> allocator_;
// };

// #ifdef TRIBOL_USE_UMPIRE
// template <typename T>
// class UmpireMemory : public Memory<T> {
//  public:
//   UmpireMemory( umpire::Allocator& allocator, IndexT size )
//       : Memory<T>( allocator.allocate( size * sizeof( T ) ), size ), allocator_{ allocator }
//   {
//   }
//   ~UmpireMemory() { allocator_.deallocate( Memory<T>::data_ ); }
//   // Copy constructor
//   UmpireMemory( const UmpireMemory& other ) = delete;
//   // Move constructor
//   UmpireMemory( UmpireMemory&& other ) : Memory<T>( other.data_, other.size_ ), allocator_{ other.allocator_ }
//   {
//     other.size_ = 0;
//     other.data_ = nullptr;
//   }
//   // Copy assignment operator
//   UmpireMemory& operator=( const UmpireMemory& other ) = delete;
//   // Move assignment operator
//   UmpireMemory& operator=( UmpireMemory&& other )
//   {
//     if ( this != &other ) {
//       Memory<T>::size_ = other.size_;
//       Memory<T>::data_ = other.data_;
//       allocator_ = other.allocator_;
//       other.size_ = 0;
//       other.data_ = nullptr;
//     }
//     return *this;
//   }

//  private:
//   umpire::Allocator& allocator_;
// };
// #endif

// template <typename T>
// struct Deleter {
//   void operator()( T* ptr ) const
//   {
// #ifdef TRIBOL_USE_UMPIRE

// #else
//     delete ptr;
// #endif
//   }
// };

// template <typename T>
// using UniquePtr = std::unique_ptr<T, Deleter<T>>;

}  // namespace tribol

#endif /* TRIBOL_COMMON_MEMORY_HPP_ */
