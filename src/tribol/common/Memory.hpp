// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef TRIBOL_COMMON_MEMORY_HPP_
#define TRIBOL_COMMON_MEMORY_HPP_

#include <cassert>

#include "tribol/config.hpp"
#include "tribol/common/ExecModel.hpp"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/TypedAllocator.hpp"
#endif

namespace tribol {

template <typename SizeT, SizeT N>
class FixedCapacity {
 public:
  using size_type = SizeT;

  TRIBOL_HOST_DEVICE FixedCapacity( [[maybe_unused]] size_type capacity ) { assert( capacity == N ); }
  TRIBOL_HOST_DEVICE constexpr size_type capacity() const { return N; }

  TRIBOL_HOST_DEVICE constexpr size_type setCapacity( size_type ) const { return N; }

  constexpr static bool capacity_at_runtime_ = false;
};

template <typename SizeT>
class RuntimeCapacity {
 public:
  using size_type = SizeT;

  TRIBOL_HOST_DEVICE RuntimeCapacity( size_type capacity ) : capacity_( capacity >= 0 ? capacity : 0 )
  {
    assert( capacity >= 0 );
  }
  TRIBOL_HOST_DEVICE size_type capacity() const { return capacity_; }

  TRIBOL_HOST_DEVICE size_type setCapacity( size_type capacity )
  {
    assert( capacity >= 0 );
    capacity_ = capacity >= 0 ? capacity : 0;
    return capacity;
  }

  constexpr static bool capacity_at_runtime_ = true;

 private:
  size_type capacity_;
};

template <typename Capacity>
class SizeEqCapacity : public Capacity {
 public:
  using typename Capacity::size_type;

  TRIBOL_HOST_DEVICE SizeEqCapacity( size_type size, [[maybe_unused]] size_type capacity ) : Capacity( size )
  {
    assert( size == capacity );
  }
  TRIBOL_HOST_DEVICE SizeEqCapacity( size_type size ) : Capacity( size ) {}

  TRIBOL_HOST_DEVICE size_type size() const { return capacity(); }

  using Capacity::capacity;

  TRIBOL_HOST_DEVICE size_type setSize( size_type size ) { return Capacity::setCapacity( size ); }

  using Capacity::setCapacity;

  TRIBOL_HOST_DEVICE constexpr bool sizeAtCapacity() const { return true; }

  using Capacity::capacity_at_runtime_;

  constexpr static bool fixed_size_ = true;
};

template <typename Capacity>
class SizeLECapacity : public Capacity {
 public:
  using typename Capacity::size_type;

  TRIBOL_HOST_DEVICE SizeLECapacity( size_type size, size_type capacity )
      : Capacity( capacity >= size ? capacity : size ), size_( size >= 0 ? size : 0 )
  {
    assert( size <= capacity );
    assert( size >= 0 );
  }

  TRIBOL_HOST_DEVICE size_type size() const { return size_; }

  using Capacity::capacity;

  TRIBOL_HOST_DEVICE size_type setSize( size_type size )
  {
    assert( size <= capacity() );
    assert( size >= 0 );
    size_ = size <= capacity() ? size : capacity();
    if ( size_ < 0 ) {
      size_ = 0;
    }
    return size_;
  }

  using Capacity::setCapacity;

  TRIBOL_HOST_DEVICE bool sizeAtCapacity() const { return size() >= capacity(); }

  using Capacity::capacity_at_runtime_;

  constexpr static bool fixed_size_ = false;

 private:
  size_type size_;
};

template <typename T, class SizeAndCapacity>
class ContiguousMemory : public SizeAndCapacity {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using typename SizeAndCapacity::size_type;

  TRIBOL_HOST_DEVICE ContiguousMemory( pointer data, size_type size, size_type capacity )
      : SizeAndCapacity( size, capacity ), data_( data )
  {
  }
  TRIBOL_HOST_DEVICE ContiguousMemory( pointer data, size_type size, size_type capacity,
                                       [[maybe_unused]] size_type stride )
      : SizeAndCapacity( size, capacity ), data_( data )
  {
    assert( stride == 1 );
  }

  TRIBOL_HOST_DEVICE value_type& at( size_type i ) { return *( data_ + i ); }
  TRIBOL_HOST_DEVICE const value_type& at( size_type i ) const { return *( data_ + i ); }

  TRIBOL_HOST_DEVICE value_type& operator[]( size_type i ) { return at( i ); }
  TRIBOL_HOST_DEVICE const value_type& operator[]( size_type i ) const { return at( i ); }

  using iterator_type = pointer;
  using const_iterator_type = const_pointer;

  TRIBOL_HOST_DEVICE pointer begin() { return data_; }
  TRIBOL_HOST_DEVICE pointer end() { return data_ + size(); }
  TRIBOL_HOST_DEVICE const_pointer begin() const { return data_; }
  TRIBOL_HOST_DEVICE const_pointer end() const { return data_ + size(); }

  using SizeAndCapacity::capacity;
  using SizeAndCapacity::size;

  using SizeAndCapacity::setCapacity;
  using SizeAndCapacity::setSize;

  TRIBOL_HOST_DEVICE constexpr size_type stride() const { return 1; }

  TRIBOL_HOST_DEVICE pointer data() const { return data_; }
  TRIBOL_HOST_DEVICE operator pointer() const { return data_; }

  using SizeAndCapacity::capacity_at_runtime_;
  using SizeAndCapacity::fixed_size_;

 protected:
  pointer data_;
};

template <typename T, class SizeAndCapacity>
class FixedStride : public SizeAndCapacity {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using typename SizeAndCapacity::size_type;

  TRIBOL_HOST_DEVICE FixedStride( pointer data, size_type size, size_type capacity, size_type stride )
      : SizeAndCapacity( size, capacity ), data_( data ), stride_( stride )
  {
    assert( stride > 0 );
  }

  template <typename Ptr>
  struct IteratorBase {
    IteratorBase( Ptr ptr, size_type stride ) : ptr_( ptr ), stride_( stride ) {}
    IteratorBase& operator++()
    {
      ptr_ += stride_;
      return *this;
    }
    IteratorBase& operator--()
    {
      ptr_ -= stride_;
      return *this;
    }
    IteratorBase operator++( int )
    {
      IteratorBase tmp = *this;
      ++( *this );
      return tmp;
    }
    IteratorBase operator--( int )
    {
      IteratorBase tmp = *this;
      --( *this );
      return tmp;
    }
    IteratorBase operator+( size_type n ) const
    {
      IteratorBase tmp = *this;
      tmp.ptr_ += n * stride_;
      return tmp;
    }
    IteratorBase operator-( size_type n ) const
    {
      IteratorBase tmp = *this;
      tmp.ptr_ -= n * stride_;
      return tmp;
    }
    bool operator==( const IteratorBase& other ) const { return ptr_ == other.ptr_; }
    bool operator!=( const IteratorBase& other ) const { return !( *this == other ); }
    value_type& operator*() { return *ptr_; }

   private:
    Ptr ptr_;
    size_type stride_;
  };
  using Iterator = IteratorBase<pointer>;
  using ConstIterator = IteratorBase<const_pointer>;

  TRIBOL_HOST_DEVICE value_type& at( size_type i ) { return *( data_ + i * stride_ ); }
  TRIBOL_HOST_DEVICE const value_type& at( size_type i ) const { return *( data_ + i * stride_ ); }

  TRIBOL_HOST_DEVICE value_type& operator[]( size_type i ) { return at( i ); }
  TRIBOL_HOST_DEVICE const value_type& operator[]( size_type i ) const { return at( i ); }

  using iterator_type = Iterator;
  using const_iterator_type = ConstIterator;

  TRIBOL_HOST_DEVICE Iterator begin() { return Iterator( data_, stride_ ); }
  TRIBOL_HOST_DEVICE Iterator end() { return Iterator( data_ + size() * stride_, stride_ ); }
  TRIBOL_HOST_DEVICE ConstIterator begin() const { return ConstIterator( data_, stride_ ); }
  TRIBOL_HOST_DEVICE ConstIterator end() const { return ConstIterator( data_ + size() * stride_, stride_ ); }

  using SizeAndCapacity::capacity;
  using SizeAndCapacity::size;

  using SizeAndCapacity::setCapacity;
  using SizeAndCapacity::setSize;

  TRIBOL_HOST_DEVICE size_type stride() const { return stride_; }

  TRIBOL_HOST_DEVICE pointer data() const { return data_; }
  TRIBOL_HOST_DEVICE operator pointer() const { return data_; }

  using SizeAndCapacity::capacity_at_runtime_;
  using SizeAndCapacity::fixed_size_;

 protected:
  pointer data_;
  size_type stride_;
};

template <class Accessor>
class Memory : public Accessor {
 public:
  using typename Accessor::const_pointer;
  using typename Accessor::pointer;
  using typename Accessor::size_type;
  using typename Accessor::value_type;
  using view_type = Memory<Accessor>;

  TRIBOL_HOST_DEVICE Memory( pointer data, size_type size, size_type capacity, size_type stride )
      : Accessor( data, size, capacity, stride )
  {
  }
  TRIBOL_HOST_DEVICE Memory( pointer data, size_type size, size_type stride = 1 ) : Memory( data, size, size, stride )
  {
  }
  TRIBOL_HOST_DEVICE Memory( const Memory& other ) : Accessor( other ) {}
  TRIBOL_HOST_DEVICE Memory& operator=( const Memory& other )
  {
    if ( this != &other ) {
      Accessor::operator=( other );
    }
    return *this;
  }

  TRIBOL_HOST_DEVICE value_type& at( size_type i )
  {
    assert( i < size() );
    return Accessor::at( i );
  }
  TRIBOL_HOST_DEVICE const value_type& at( size_type i ) const
  {
    assert( i < size() );
    return Accessor::at( i );
  }

  TRIBOL_HOST_DEVICE value_type& operator[]( size_type i )
  {
    assert( i < size() );
    return Accessor::operator[]( i );
  }
  TRIBOL_HOST_DEVICE const value_type& operator[]( size_type i ) const
  {
    assert( i < size() );
    return Accessor::operator[]( i );
  }

  using typename Accessor::const_iterator_type;
  using typename Accessor::iterator_type;

  using Accessor::begin;
  using Accessor::end;

  template <typename NewAccessor>
  TRIBOL_HOST_DEVICE Memory<NewAccessor> view( size_type offset, size_type size, size_type capacity,
                                               size_type stride = 1 ) const
  {
    assert( offset + size * stride <= this->size() );
    return Memory<NewAccessor>( data() + offset, size, capacity, stride );
  }

  TRIBOL_HOST_DEVICE Memory<Accessor> view() const { return Memory<Accessor>( *this ); }

  using Accessor::capacity;
  using Accessor::size;

  using Accessor::setCapacity;
  using Accessor::setSize;

  using Accessor::stride;

  using Accessor::data;
  // not working with clang 16
  // using Accessor::operator pointer;
  TRIBOL_HOST_DEVICE operator pointer() const { return Accessor::operator pointer(); }

  using Accessor::capacity_at_runtime_;
  using Accessor::fixed_size_;

 protected:
  using Accessor::data_;
};

template <typename T, IndexT N, template <typename> class SizeVsCapacity = SizeEqCapacity>
class StackMemory : public Memory<ContiguousMemory<T, SizeVsCapacity<FixedCapacity<IndexT, N>>>> {
 public:
  using BaseClass = Memory<ContiguousMemory<T, SizeVsCapacity<FixedCapacity<IndexT, N>>>>;
  using typename BaseClass::const_pointer;
  using typename BaseClass::pointer;
  using typename BaseClass::value_type;
  using typename BaseClass::view_type;
  using size_type = IndexT;

  TRIBOL_HOST_DEVICE StackMemory( size_type size = N ) : BaseClass( nullptr, size, N, 1 ) { data_ = stack_data_; }
  TRIBOL_HOST_DEVICE StackMemory( size_type size, [[maybe_unused]] size_type capacity ) : StackMemory( size )
  {
    assert( capacity == N );
  }
  // copy constructor (deep copy)
  TRIBOL_HOST_DEVICE StackMemory( const StackMemory& other ) : StackMemory( other.size() )
  {
    assert( other.capacity() == N );
    if ( this != &other ) {
      for ( size_type i = 0; i < other.size(); ++i ) {
        stack_data_[i] = other.stack_data_[i];
      }
    }
  }
  // no move constructor (should be the same as copy constructor, deep copy)
  // copy assignment operator (deep copy)
  TRIBOL_HOST_DEVICE StackMemory& operator=( const StackMemory& other )
  {
    assert( other.capacity() == N );
    if ( this != &other ) {
      BaseClass::operator=( other );
      for ( size_type i = 0; i < other.size(); ++i ) {
        stack_data_[i] = other.stack_data_[i];
      }
    }
    return *this;
  }
  // no move assignment operator (should be the same as copy assignment operator, deep copy)
  // destructor
  TRIBOL_DEFAULT_HOST_DEVICE ~StackMemory() = default;

  using BaseClass::at;
  using BaseClass::operator[];

  using typename BaseClass::const_iterator_type;
  using typename BaseClass::iterator_type;

  using BaseClass::begin;
  using BaseClass::end;

  using BaseClass::view;

  using BaseClass::capacity;
  using BaseClass::size;

  using BaseClass::setCapacity;
  using BaseClass::setSize;

  using BaseClass::stride;

  using BaseClass::data;
  // not working with clang 16
  // using BaseClass::operator pointer;
  TRIBOL_HOST_DEVICE operator pointer() const { return BaseClass::operator pointer(); }

  using BaseClass::capacity_at_runtime_;
  using BaseClass::fixed_size_;

 private:
  using BaseClass::data_;
  T stack_data_[N];
};

template <typename T>
class HeapAllocator {
 public:
  using value_type = T;
  using pointer = T*;

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

template <typename T, class Allocator = HeapAllocator<T>, typename SizeT = IndexT,
          class SizeVsCapacity = SizeLECapacity<RuntimeCapacity<SizeT>>>
class AllocatedMemory : public Memory<ContiguousMemory<T, SizeVsCapacity>> {
 public:
  using BaseClass = Memory<ContiguousMemory<T, SizeVsCapacity>>;
  using typename BaseClass::const_pointer;
  using typename BaseClass::pointer;
  using typename BaseClass::size_type;
  using typename BaseClass::value_type;
  using typename BaseClass::view_type;

  static_assert( std::is_same<typename Allocator::value_type, value_type>::value,
                 "AllocatedMemory must be used with same type as allocator" );

#pragma nv_exec_check_disable
  TRIBOL_HOST_DEVICE AllocatedMemory( size_type size, size_type capacity, Allocator allocator = Allocator() )
      : BaseClass( allocator.allocate( capacity ), size, capacity, 1 ), allocator_( std::move( allocator ) )
  {
  }
  TRIBOL_HOST_DEVICE AllocatedMemory( size_type size, Allocator allocator = Allocator() )
      : AllocatedMemory( size, size, std::move( allocator ) )
  {
  }

#pragma nv_exec_check_disable
  TRIBOL_HOST_DEVICE ~AllocatedMemory() { allocator_.deallocate( data_, size() ); }

  // Copy constructor
#pragma nv_exec_check_disable
  TRIBOL_HOST_DEVICE AllocatedMemory( const AllocatedMemory& other )
      : AllocatedMemory( other.size(), Allocator( other.allocator_ ) )
  {
    // deep copy the data
    allocator_.copy( data_, other.data_, other.size() );
  }

  // Move constructor
#pragma nv_exec_check_disable
  TRIBOL_HOST_DEVICE AllocatedMemory( AllocatedMemory&& other )
      : BaseClass( other.data_, other.size(), other.capacity(), other.stride() ), allocator_{ other.allocator_ }
  {
    if constexpr ( !fixed_size_ ) {
      other.setSize( 0 );
    }
    if constexpr ( capacity_at_runtime_ ) {
      other.data_ = nullptr;
      other.setCapacity( 0 );
    } else {
      // allocate new memory for the moved object so the size is the same
      other.data_ = allocator_.allocate( other.size() );
    }
  }

  // Copy assignment operator
#pragma nv_exec_check_disable
  TRIBOL_HOST_DEVICE AllocatedMemory& operator=( const AllocatedMemory& other )
  {
    if ( this != &other ) {
      BaseClass::operator=( other );
      allocator_ = other.allocator();
      // deep copy the data
      data_ = allocator_.allocate( other.size() );
      allocator_.copy( data_, other.data(), other.size() );
    }
    return *this;
  }

  // Move assignment operator
#pragma nv_exec_check_disable
  TRIBOL_HOST_DEVICE AllocatedMemory& operator=( AllocatedMemory&& other )
  {
    if ( this != &other ) {
      BaseClass::operator=( std::move( other ) );
      allocator_ = other.allocator_;
      if ( capacity_at_runtime_ ) {
        other.data_ = nullptr;
        other.sizer.setSize( 0 );
      } else {
        // allocate new memory for the moved object so the size is the same
        other.data_ = allocator_.allocate( other.size_ );
      }
      other.stride_ = 1;
    }
    return *this;
  }

  using BaseClass::at;
  using BaseClass::operator[];

  using typename BaseClass::const_iterator_type;
  using typename BaseClass::iterator_type;

  using BaseClass::begin;
  using BaseClass::end;

  TRIBOL_HOST_DEVICE const Allocator& allocator() const { return allocator_; }

  using BaseClass::view;

  using BaseClass::capacity;
  using BaseClass::size;

  using BaseClass::setCapacity;
  using BaseClass::setSize;

  using BaseClass::stride;

  using BaseClass::data;
  // not working with clang 16
  // using BaseClass::operator pointer;
  TRIBOL_HOST_DEVICE operator pointer() const { return BaseClass::operator pointer(); }

  using BaseClass::capacity_at_runtime_;
  using BaseClass::fixed_size_;

 private:
  using BaseClass::data_;
  Allocator allocator_;
};

}  // namespace tribol

#endif /* TRIBOL_COMMON_MEMORY_HPP_ */
