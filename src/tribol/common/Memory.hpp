// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_MEMORY_HPP_
#define SRC_TRIBOL_COMMON_MEMORY_HPP_

#include <cassert>
#include <cstddef>

#include "tribol/common/ExecModel.hpp"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/TypedAllocator.hpp"
#endif

namespace tribol {

template <size_t N>
class FixedCapacity {
 public:
  using size_type = size_t;

  TRIBOL_HOST_DEVICE FixedCapacity( [[maybe_unused]] size_type capacity ) { assert( capacity == N ); }
  TRIBOL_HOST_DEVICE constexpr size_type capacity() const { return N; }

  TRIBOL_HOST_DEVICE constexpr size_type setCapacity( size_type ) const { return N; }

  using capacity_at_runtime_ = std::false_type;
};

class RuntimeCapacity {
 public:
  using size_type = size_t;

  TRIBOL_HOST_DEVICE RuntimeCapacity( size_type capacity ) : capacity_( capacity ) {}
  TRIBOL_HOST_DEVICE size_type capacity() const { return capacity_; }

  TRIBOL_HOST_DEVICE size_type setCapacity( size_type capacity )
  {
    capacity_ = capacity;
    return capacity;
  }

  using capacity_at_runtime_ = std::true_type;

 private:
  size_type capacity_;
};

template <typename Capacity>
class SizeEqCapacity : public Capacity {
 public:
  using typename Capacity::size_type;

  using capacity_type = Capacity;

  TRIBOL_HOST_DEVICE SizeEqCapacity( size_type size, [[maybe_unused]] size_type capacity ) : Capacity( size )
  {
    assert( size == capacity );
  }
  TRIBOL_HOST_DEVICE SizeEqCapacity( size_type size ) : Capacity( size ) {}

  TRIBOL_HOST_DEVICE constexpr size_type size() const { return capacity(); }

  using Capacity::capacity;

  TRIBOL_HOST_DEVICE size_type setSize( size_type size ) { return setCapacity( size ); }

  using Capacity::setCapacity;

  TRIBOL_HOST_DEVICE constexpr bool sizeAtCapacity() const { return true; }

  constexpr static bool fixed_size_ = true;
};

template <typename Capacity>
class SizeLECapacity : public Capacity {
 public:
  using typename Capacity::size_type;

  using capacity_type = Capacity;

  TRIBOL_HOST_DEVICE SizeLECapacity( size_type size, size_type capacity )
      : Capacity( capacity >= size ? capacity : size ), size_( size )
  {
    assert( size <= capacity );
  }

  TRIBOL_HOST_DEVICE size_type size() const { return size_; }

  using Capacity::capacity;

  TRIBOL_HOST_DEVICE size_type setSize( size_type size )
  {
    assert( size <= capacity() );
    size_ = size <= capacity() ? size : capacity();
    return size_;
  }

  TRIBOL_HOST_DEVICE bool sizeAtCapacity() const { return size() >= capacity(); }

  constexpr static bool fixed_size_ = false;

 private:
  size_type size_;
};

template <typename T, class SizeAndCapacity>
class ContiguousMemory : public SizeAndCapacity {
 public:
  using typename SizeAndCapacity::size_type;

  using size_and_capacity_type = SizeAndCapacity;

  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;

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

  using SizeAndCapacity::size;

  TRIBOL_HOST_DEVICE constexpr size_type stride() const { return 1; }

  TRIBOL_HOST_DEVICE pointer data() const { return data_; }
  TRIBOL_HOST_DEVICE operator pointer() const { return data_; }

 protected:
  pointer data_;
};

template <typename T, class SizeAndCapacity>
class FixedStride : public SizeAndCapacity {
 public:
  using typename SizeAndCapacity::size_type;

  using typename SizeAndCapacity::capacity_type;

  using size_and_capacity_type = SizeAndCapacity;

  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;

  TRIBOL_HOST_DEVICE FixedStride( pointer data, size_type size, size_type capacity, size_type stride )
      : SizeAndCapacity( size, capacity ), data_( data ), stride_( stride )
  {
    assert( stride > 0 );
  }

  template <typename Ptr>
  struct IteratorBase {
    TRIBOL_HOST_DEVICE IteratorBase( Ptr ptr, size_type stride ) : ptr_( ptr ), stride_( stride ) {}
    TRIBOL_HOST_DEVICE IteratorBase& operator++()
    {
      ptr_ += stride_;
      return *this;
    }
    TRIBOL_HOST_DEVICE IteratorBase& operator--()
    {
      ptr_ -= stride_;
      return *this;
    }
    TRIBOL_HOST_DEVICE IteratorBase operator++( int )
    {
      IteratorBase tmp = *this;
      ++( *this );
      return tmp;
    }
    TRIBOL_HOST_DEVICE IteratorBase operator--( int )
    {
      IteratorBase tmp = *this;
      --( *this );
      return tmp;
    }
    TRIBOL_HOST_DEVICE IteratorBase operator+( size_type n ) const
    {
      IteratorBase tmp = *this;
      tmp.ptr_ += n * stride_;
      return tmp;
    }
    TRIBOL_HOST_DEVICE IteratorBase operator-( size_type n ) const
    {
      IteratorBase tmp = *this;
      tmp.ptr_ -= n * stride_;
      return tmp;
    }
    TRIBOL_HOST_DEVICE bool operator==( const IteratorBase& other ) const { return ptr_ == other.ptr_; }
    TRIBOL_HOST_DEVICE bool operator!=( const IteratorBase& other ) const { return !( *this == other ); }
    TRIBOL_HOST_DEVICE value_type& operator*() { return *ptr_; }

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

  using SizeAndCapacity::size;

  TRIBOL_HOST_DEVICE size_type stride() const { return stride_; }

  TRIBOL_HOST_DEVICE pointer data() const { return data_; }
  TRIBOL_HOST_DEVICE operator pointer() const { return data_; }

 protected:
  pointer data_;
  size_type stride_;
};

template <class Accessor>
class Memory : public Accessor {
 public:
  using typename Accessor::size_type;

  using typename Accessor::pointer;
  using typename Accessor::value_type;

  using accessor_type = Accessor;
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

  template <typename NewAccessor>
  TRIBOL_HOST_DEVICE Memory<NewAccessor> view( size_type offset, size_type size, size_type capacity,
                                               size_type stride = 1 ) const
  {
    assert( offset + size * stride <= this->size() );
    return Memory<NewAccessor>( data() + offset, size, capacity, stride );
  }

  TRIBOL_HOST_DEVICE Memory<Accessor> view() { return Memory<Accessor>( *this ); }
  TRIBOL_HOST_DEVICE Memory<Accessor> view() const { return Memory<Accessor>( *this ); }

  using Accessor::size;

  using Accessor::data;

  // assume this is a view of memory and already initialized
  constexpr static bool initialized_ = true;
};

template <typename T, size_t N, template <typename> class SizeVsCapacity = SizeEqCapacity>
class StackMemory : public Memory<ContiguousMemory<T, SizeVsCapacity<FixedCapacity<N>>>> {
 public:
  using BaseClass = Memory<ContiguousMemory<T, SizeVsCapacity<FixedCapacity<N>>>>;
  using typename BaseClass::size_type;

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

  constexpr static bool initialized_ = false;

 private:
  using BaseClass::data_;
  T stack_data_[N];
};

template <typename T>
class Allocator {
  static_assert( !std::is_const<T>::value, "Allocator does not support const types" );
  static_assert( !std::is_volatile<T>::value, "Allocator does not support volatile types" );

 public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  TRIBOL_HOST_DEVICE T* allocate( size_type n ) const
  {
    return static_cast<T*>( ::operator new( n * sizeof( value_type ) ) );
  }

  TRIBOL_HOST_DEVICE void deallocate( T* p, size_type ) const { ::operator delete( p ); }

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE void uninitialized_copy( T* first, T* last, T* d_first ) const
  {
    std::uninitialized_copy( first, last, d_first );
  }
};

template <typename T, typename U>
TRIBOL_HOST_DEVICE inline constexpr bool operator==( const Allocator<T>&, const Allocator<U>& )
{
  return true;
}

#ifdef TRIBOL_USE_UMPIRE
template <typename T, MemorySpace MSPACE>
class UmpireAllocator {
 public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  UmpireAllocator( umpire::Allocator allocator ) : allocator_{ std::move( allocator ) } {}
  UmpireAllocator()
      : UmpireAllocator( umpire::ResourceManager::getInstance().getAllocator( getResourceAllocatorID( MSPACE ) ) )
  {
  }

  T* allocate( size_type n ) const { return static_cast<T*>( allocator_.allocate( n ) ); }

  void deallocate( T* p, size_type n ) const { allocator_.deallocate( p, n ); }

  void uninitialized_copy( T* first, T*, T* d_first ) const
  {
    auto& rm = umpire::ResourceManager::getInstance();
    rm.copy( d_first, first );
  }

 private:
  // NOTE: TypedAllocator has non-const member functions, so make this mutable
  mutable umpire::TypedAllocator<value_type> allocator_;
};
#endif

template <typename T>
class DynamicAllocator {
 public:
  using value_type = T;
  using size_type = size_t;
  using pointer = T*;

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE DynamicAllocator() : allocator_id_( getDefaultAllocatorID() ) {}
  TRIBOL_HOST_DEVICE DynamicAllocator( int allocator_id ) : allocator_id_( allocator_id ) {}

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE pointer allocate( size_type n ) const
  {
    return static_cast<pointer>(
#ifdef TRIBOL_USE_UMPIRE
        umpire::ResourceManager::getInstance().getAllocator( allocator_id_ ).allocate( n * sizeof( value_type ) )
#else
        Allocator<T>().allocate( n )
#endif
    );
  }

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE void deallocate( pointer p, [[maybe_unused]] size_type n ) const
  {
#ifdef TRIBOL_USE_UMPIRE
    umpire::ResourceManager::getInstance().getAllocator( allocator_id_ ).deallocate( p );
#else
    Allocator<T>().deallocate( p, n );
#endif
  }

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE void uninitialized_copy( T* first, [[maybe_unused]] T* last, T* d_first ) const
  {
#ifdef TRIBOL_USE_UMPIRE
    auto& rm = umpire::ResourceManager::getInstance();
    rm.copy( d_first, first );
#else
    Allocator<T>().uninitialized_copy( first, last, d_first );
#endif
  }

  TRIBOL_HOST_DEVICE int id() const { return allocator_id_; }

 private:
  int allocator_id_;
};

template <typename T, class Allocator = Allocator<T>, class SizeVsCapacity = SizeLECapacity<RuntimeCapacity>>
class AllocatedMemory : public Memory<ContiguousMemory<T, SizeVsCapacity>> {
 public:
  using BaseClass = Memory<ContiguousMemory<T, SizeVsCapacity>>;

  using typename BaseClass::size_type;

  using typename BaseClass::pointer;
  using typename BaseClass::value_type;

  static_assert( std::is_same<typename Allocator::value_type, value_type>::value,
                 "AllocatedMemory must be used with same type as allocator" );

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory( size_type size, size_type capacity, Allocator allocator = Allocator() )
      : BaseClass( allocator.allocate( capacity ), size, capacity, 1 ), allocator_( std::move( allocator ) )
  {
  }

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory( size_type size, Allocator allocator = Allocator() )
      : AllocatedMemory( size, size, std::move( allocator ) )
  {
  }

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory( BaseClass&& memory, Allocator&& allocator = Allocator() )
      : BaseClass( std::move( memory ) ), allocator_( std::move( allocator ) )
  {
  }

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory( const AllocatedMemory& src, AllocatedMemory&& dst )
      : BaseClass( dst.allocator_.allocate( 0 ), 0, 0, 1 )
  {
    assert( src.size() == dst.size() );
    ( *this ) = std::move( dst );
    allocator_.uninitialized_copy( src.data_, src.data_ + src.size(), data_ );
  }

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE ~AllocatedMemory() { allocator_.deallocate( data_, capacity() ); }

  // Copy constructor
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory( const AllocatedMemory& other )
      : AllocatedMemory( other.size(), Allocator( other.allocator_ ) )
  {
    // deep copy the data
    allocator_.uninitialized_copy( other.data_, other.data_ + other.size(), data_ );
  }

  // Move constructor
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory( AllocatedMemory&& other )
      : BaseClass( other.data_, other.size(), other.capacity(), other.stride() ), allocator_{ other.allocator_ }
  {
    if constexpr ( !fixed_size_ ) {
      other.setSize( 0 );
    }
    if constexpr ( capacity_at_runtime_::value ) {
      other.data_ = nullptr;
      other.setCapacity( 0 );
    } else {
      // allocate new memory for the moved object so the size is the same
      other.data_ = allocator_.allocate( other.size() );
    }
  }

  // Copy assignment operator
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
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
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory& operator=( AllocatedMemory&& other )
  {
    if ( this != &other ) {
      BaseClass::operator=( std::move( other ) );
      allocator_ = other.allocator_;
      if constexpr ( !fixed_size_ ) {
        other.setSize( 0 );
      }
      if constexpr ( capacity_at_runtime_::value ) {
        other.data_ = nullptr;
        other.setCapacity( 0 );
      } else {
        // allocate new memory for the moved object so the size is the same
        other.data_ = allocator_.allocate( other.size() );
      }
    }
    return *this;
  }

  TRIBOL_HOST_DEVICE const Allocator& allocator() const { return allocator_; }

  using BaseClass::capacity;
  using BaseClass::size;

  using BaseClass::fixed_size_;
  using typename BaseClass::capacity_at_runtime_;

  constexpr static bool initialized_ = false;

 private:
  using BaseClass::data_;
  Allocator allocator_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_COMMON_MEMORY_HPP_ */
