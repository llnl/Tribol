// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef TRIBOL_COMMON_MEMORY_HPP_
#define TRIBOL_COMMON_MEMORY_HPP_

#include "tribol/config.hpp"
#include "tribol/common/ExecModel.hpp"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

namespace tribol {

template <typename T>
class Memory {
 public:
  TRIBOL_HOST_DEVICE Memory( const Memory& other ) : data_{ other.data_ }, size_{ other.size_ } {}

  TRIBOL_HOST_DEVICE T* data() const { return data_; }

  TRIBOL_HOST_DEVICE IndexT size() const { return size_; }

 protected:
  TRIBOL_HOST_DEVICE Memory( T* data, IndexT size ) : data_{ data }, size_{ size } {}

  T* data_;
  IndexT size_;
};

template <typename T, MemorySpace MSPACE>
class AllocatedMemory : public Memory<T> {
 public:
  AllocatedMemory( IndexT size ) : Memory<T>( makePool( size ), size )
  {
#ifdef TRIBOL_USE_UMPIRE
    auto& rm = umpire::ResourceManager::getInstance();
    allocator_ = rm.getAllocator( getResourceAllocatorID( MSPACE ) );
#endif
  }

  ~AllocatedMemory()
  {
#ifdef TRIBOL_USE_UMPIRE
    allocator_.deallocate( Memory<T>::data_ );
#else
    free( Memory<T>::data_ );
#endif
  }

  // Copy constructor
  AllocatedMemory( const AllocatedMemory& other ) = delete;

  // Move constructor
  AllocatedMemory( AllocatedMemory&& other )
      : Memory<T>( other.data_, other.size_ )
#ifdef TRIBOL_USE_UMPIRE
        ,
        allocator_{ other.allocator_ }
#endif
  {
    other.size_ = 0;
    other.pool_ = nullptr;
  }

  // Copy assignment operator
  AllocatedMemory& operator=( const AllocatedMemory& other ) = delete;

  // Move assignment operator
  AllocatedMemory& operator=( AllocatedMemory&& other )
  {
    if ( this != &other ) {
      Memory<T>::size_ = other.size_;
      Memory<T>::data_ = other.data_;
#ifdef TRIBOL_USE_UMPIRE
      Memory<T>::allocator_ = other.allocator_;
#endif

      other.size_ = 0;
      other.data_ = nullptr;
    }
    return *this;
  }

 private:
  T* makePool( IndexT size )
  {
#ifdef TRIBOL_USE_UMPIRE
    auto& rm = umpire::ResourceManager::getInstance();
    return rm.getAllocator( getResourceAllocatorID( MSPACE ) ).allocate( size * sizeof( T ) );
#else
    return malloc( size_ * sizeof( T ) );
#endif
  }

#ifdef TRIBOL_USE_UMPIRE
  umpire::Allocator& allocator_;
#endif
};

#ifdef TRIBOL_USE_UMPIRE
template <typename T>
class UmpireMemory : public Memory<T> {
 public:
  UmpireMemory( umpire::Allocator& allocator, IndexT size )
      : Memory<T>( allocator.allocate( size * sizeof( T ) ), size ), allocator_{ allocator }
  {
  }
  ~UmpireMemory() { allocator_.deallocate( Memory<T>::data_ ); }
  // Copy constructor
  UmpireMemory( const UmpireMemory& other ) = delete;
  // Move constructor
  UmpireMemory( UmpireMemory&& other ) : Memory<T>( other.data_, other.size_ ), allocator_{ other.allocator_ }
  {
    other.size_ = 0;
    other.data_ = nullptr;
  }
  // Copy assignment operator
  UmpireMemory& operator=( const UmpireMemory& other ) = delete;
  // Move assignment operator
  UmpireMemory& operator=( UmpireMemory&& other )
  {
    if ( this != &other ) {
      Memory<T>::size_ = other.size_;
      Memory<T>::data_ = other.data_;
      allocator_ = other.allocator_;
      other.size_ = 0;
      other.data_ = nullptr;
    }
    return *this;
  }

 private:
  umpire::Allocator& allocator_;
};
#endif

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
