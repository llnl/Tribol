// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_ARRAYS_HPP_
#define SRC_TRIBOL_COMMON_ARRAYS_HPP_

#include <cassert>

// Tribol includes
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Memory.hpp"

namespace tribol {

template <typename MemoryT>
class ArrayBase {
 public:
  using value_type = typename MemoryT::value_type;
  using pointer = typename MemoryT::pointer;
  using const_pointer = typename MemoryT::const_pointer;
  using size_type = typename MemoryT::size_type;
  using memory_type = MemoryT;

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE ArrayBase( MemoryT&& memory ) : memory_( std::move( memory ) )
  {
    // initialize memory if needed
    if constexpr ( MemoryT::initialized_ == false ) {
      for ( auto& value : memory_ ) {
        value = value_type();
      }
    }
  }
  template <typename MemoryT2>
  TRIBOL_HOST_DEVICE ArrayBase( const ArrayBase<MemoryT2>& other, MemoryT&& memory )
      : memory_( other.memory(), std::move( memory ) )
  {
  }
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE ~ArrayBase()
  {
    // call destructor on all elements if needed
    if constexpr ( MemoryT::initialized_ == false ) {
      for ( auto& value : memory_ ) {
        value.~value_type();
      }
    }
  }
  TRIBOL_DEFAULT_HOST_DEVICE ArrayBase( const ArrayBase& other ) = default;
  TRIBOL_DEFAULT_HOST_DEVICE ArrayBase( ArrayBase&& other ) = default;
  TRIBOL_DEFAULT_HOST_DEVICE ArrayBase& operator=( const ArrayBase& other ) = default;
  TRIBOL_DEFAULT_HOST_DEVICE ArrayBase& operator=( ArrayBase&& other ) = default;

  TRIBOL_HOST_DEVICE value_type& at( size_type i ) { return memory_.at( i ); }
  TRIBOL_HOST_DEVICE const value_type& at( size_type i ) const { return memory_.at( i ); }

  TRIBOL_HOST_DEVICE value_type& operator[]( size_type i ) { return memory_.at( i ); }
  TRIBOL_HOST_DEVICE const value_type& operator[]( size_type i ) const { return memory_.at( i ); }

  using iterator_type = typename MemoryT::iterator_type;
  using const_iterator_type = typename MemoryT::const_iterator_type;

  TRIBOL_HOST_DEVICE iterator_type begin() { return memory_.begin(); }
  TRIBOL_HOST_DEVICE iterator_type end() { return memory_.end(); }

  TRIBOL_HOST_DEVICE const_iterator_type begin() const { return memory_.begin(); }
  TRIBOL_HOST_DEVICE const_iterator_type end() const { return memory_.end(); }

  TRIBOL_HOST_DEVICE MemoryT& memory() { return memory_; }
  TRIBOL_HOST_DEVICE const MemoryT& memory() const { return memory_; }

  TRIBOL_HOST_DEVICE operator typename MemoryT::view_type() const { return memory_; }

 protected:
  MemoryT memory_;
};

template <typename T, size_t N, class MemoryT = StackMemory<T, N>>
class FixedArray : public ArrayBase<MemoryT> {
 public:
  using value_type = T;
  using BaseClass = ArrayBase<MemoryT>;
  using typename BaseClass::size_type;

  static_assert( std::is_same<typename MemoryT::value_type, value_type>::value,
                 "BoundedArray must be used with same type as memory" );

  TRIBOL_HOST_DEVICE FixedArray( MemoryT&& memory = MemoryT( N ) ) : BaseClass( std::move( memory ) ) {}

  TRIBOL_HOST_DEVICE constexpr size_type size() const { return N; }

 private:
  using BaseClass::memory_;
};

template <typename T, class MemoryT = AllocatedMemory<T>>
class BoundedArray : public ArrayBase<MemoryT> {
 public:
  using value_type = T;
  using BaseClass = ArrayBase<MemoryT>;
  using typename BaseClass::size_type;

  static_assert( MemoryT::fixed_size_ == false, "BoundedArray must be used with non-fixed size memory" );
  static_assert( std::is_same<typename MemoryT::value_type, value_type>::value,
                 "BoundedArray must be used with same type as memory" );

  TRIBOL_HOST_DEVICE BoundedArray( size_type size, size_type capacity ) : BaseClass( MemoryT( size, capacity ) ) {}
  TRIBOL_HOST_DEVICE BoundedArray( MemoryT&& memory ) : BaseClass( std::move( memory ) ) {}

  using BaseClass::at;

  TRIBOL_HOST_DEVICE constexpr size_type size() const { return memory_.size(); }
  TRIBOL_HOST_DEVICE constexpr size_type capacity() const { return memory_.capacity(); }

  template <typename... Args>
  TRIBOL_HOST_DEVICE inline void emplace( size_type i, Args&&... args )
  {
    if ( i >= size() ) {
      i = memory_.setSize( i + 1 ) - 1;
    }
    ::new ( static_cast<void*>( memory_.data() + i ) ) T( std::forward<Args>( args )... );
  };
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE void push_back( T value ) { emplace_back( std::move( value ) ); }
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  template <typename... Args>
  TRIBOL_HOST_DEVICE void emplace_back( Args&&... args )
  {
    emplace( size(), std::forward<Args>( args )... );
  }
  TRIBOL_HOST_DEVICE void pop_back() { memory_.setSize( size() - 1 ); }
  TRIBOL_HOST_DEVICE void resize( size_type new_size )
  {
    assert( new_size <= capacity() );
    // destruct elements no longer in range
    for ( size_type i{ new_size }; i < size(); ++i ) {
      at( i ).~T();
    }
    // create empty new elements
    for ( size_type i{ size() }; i < new_size; ++i ) {
      at( i ) = T{};
    }
    memory_.setSize( new_size );
  }

 protected:
  template <typename MemoryT2>
  TRIBOL_HOST_DEVICE BoundedArray( const BoundedArray<T, MemoryT2>& other, MemoryT&& memory )
      : BaseClass( other, std::move( memory ) )
  {
  }

  using BaseClass::memory_;
};

template <typename T, class MemoryT = AllocatedMemory<T>>
class BoundedArray2D : public BoundedArray<T, MemoryT> {
 public:
  using value_type = T;
  using BaseClass = BoundedArray<T, MemoryT>;
  using typename BaseClass::size_type;

  TRIBOL_HOST_DEVICE BoundedArray2D( size_type height, size_type width, size_type max_height )
      : BaseClass( height * width, max_height * width ), height_( height ), width_( width ), max_height_( max_height )
  {
    assert( height >= 0 && width >= 0 );
    assert( size() == height * width );
    assert( capacity() == max_height * width );
  }
  TRIBOL_HOST_DEVICE BoundedArray2D( size_type height, size_type width ) : BoundedArray2D( height, width, height ) {}
  TRIBOL_HOST_DEVICE BoundedArray2D() : BoundedArray2D( 0, 0, 0 )  // default constructor initializes to empty array
  {
  }

  // constructor with forwarded arguments for memory
  template <typename... Args>
  TRIBOL_HOST_DEVICE BoundedArray2D( size_type height, size_type width, size_type max_height, Args&&... args )
      : BaseClass( MemoryT( height * width, max_height * width, std::forward<Args>( args )... ) ),
        height_( height ),
        width_( width ),
        max_height_( height )
  {
    assert( height >= 0 && width >= 0 );
    assert( size() == height * width );
    assert( capacity() == max_height_ * width );
  }

  using BaseClass::at;

  TRIBOL_HOST_DEVICE value_type& at( size_type i, size_type j )
  {
    assert( i < height_ && j < width_ );
    return at( i * width_ + j );
  }
  TRIBOL_HOST_DEVICE const value_type& at( size_type i, size_type j ) const
  {
    assert( i < height_ && j < width_ );
    return at( i * width_ + j );
  }
  TRIBOL_HOST_DEVICE value_type& operator()( size_type i, size_type j ) { return at( i, j ); }
  TRIBOL_HOST_DEVICE const value_type& operator()( size_type i, size_type j ) const { return at( i, j ); }

  using BaseClass::capacity;
  using BaseClass::size;

  TRIBOL_HOST_DEVICE size_type height() const { return height_; }
  TRIBOL_HOST_DEVICE size_type width() const { return width_; }
  TRIBOL_HOST_DEVICE size_type max_height() const { return max_height_; }

  TRIBOL_HOST_DEVICE void push_back( std::initializer_list<T> values )
  {
    assert( values.size() == width() );
    assert( height() < max_height_ );
    ++height_;
    memory_.setSize( height() * width() );
    size_type j = 0;
    for ( auto& value : values ) {
      at( height() - 1, j++ ) = value;
    }
  }

  TRIBOL_HOST_DEVICE BoundedArray<T, Memory<typename MemoryT::accessor_type>> rowView( size_type i )
  {
    assert( i < height_ );
    return BoundedArray<T, Memory<typename MemoryT::accessor_type>>(
        Memory<typename MemoryT::accessor_type>( &at( i, 0 ), width_, width_, memory_.stride() ) );
  }

  TRIBOL_HOST_DEVICE BoundedArray<T, Memory<FixedStride<T, typename MemoryT::size_and_capacity_type>>> colView(
      size_type j )
  {
    assert( j < width_ );
    return BoundedArray<T, Memory<FixedStride<T, typename MemoryT::size_and_capacity_type>>>(
        Memory<FixedStride<T, typename MemoryT::size_and_capacity_type>>( &at( 0, j ), height_, height_,
                                                                          memory_.stride() * width_ ) );
  }

 protected:
  template <typename MemoryT2>
  TRIBOL_HOST_DEVICE BoundedArray2D( const BoundedArray2D<T, MemoryT2>& other, MemoryT&& memory )
      : BaseClass( other, std::move( memory ) ),
        height_( other.height() ),
        width_( other.width() ),
        max_height_( other.max_height() )
  {
    assert( height_ >= 0 && width_ >= 0 );
    assert( size() == height_ * width_ );
    assert( capacity() == max_height_ * width_ );
  }

  using BaseClass::emplace_back;
  using BaseClass::pop_back;
  using BaseClass::push_back;
  using BaseClass::resize;

  using BaseClass::memory_;
  size_type height_;
  size_type width_;
  size_type max_height_;
};

template <typename T, class Allocator = Allocator<T>>
class ArrayResizer {
 public:
  using size_type = size_t;

  constexpr static RealT default_resize_ratio_ = 2.0;

  TRIBOL_HOST_DEVICE ArrayResizer( size_type min_delta_capacity = 1 ) : min_delta_capacity_( min_delta_capacity ) {}

  TRIBOL_HOST_DEVICE bool resizeNeeded( size_type new_size, const AllocatedMemory<T, Allocator>& memory )
  {
    return new_size > memory.capacity();
  }

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory<T, Allocator> resize( const AllocatedMemory<T, Allocator>& old_memory )
  {
    size_type new_capacity = static_cast<size_type>( old_memory.capacity() * default_resize_ratio_ );
    new_capacity = new_capacity > ( old_memory.capacity() + min_delta_capacity_ )
                       ? new_capacity
                       : ( old_memory.capacity() + min_delta_capacity_ );
    return resize( old_memory, new_capacity );
  }

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory<T, Allocator> resize( const AllocatedMemory<T, Allocator>& old_memory,
                                                           size_type new_capacity )
  {
    assert( new_capacity > old_memory.capacity() );
    AllocatedMemory<T, Allocator> new_memory( old_memory.size(), new_capacity );
    old_memory.allocator().uninitialized_copy( old_memory.data(), old_memory.data() + old_memory.size(),
                                               new_memory.data() );
    return std::move( new_memory );
  }

 private:
  size_type min_delta_capacity_;
};

template <typename T, class Allocator = DynamicAllocator<T>>
class Array : public BoundedArray<T, AllocatedMemory<T, Allocator>> {
 public:
  using value_type = T;
  using BaseClass = BoundedArray<T, AllocatedMemory<T, Allocator>>;
  using typename BaseClass::memory_type;
  using typename BaseClass::pointer;
  using typename BaseClass::size_type;

  constexpr static size_type default_capacity_ = 0;

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE Array( size_type size = 0, size_type capacity = default_capacity_ )
      : BaseClass( memory_type( size, capacity >= size ? capacity : size ) ), resizer_( 1 )
  {
  }

  // copy constructor with a custom allocator
  template <typename Allocator2>
  TRIBOL_HOST_DEVICE Array( const Array<T, Allocator2>& other, Allocator&& allocator )
      : BaseClass( other, AllocatedMemory<T, Allocator>( other.size(), other.capacity(), std::move( allocator ) ) )
  {
  }

  using BaseClass::capacity;
  using BaseClass::size;

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE void push_back( T value ) { emplace_back( std::move( value ) ); }
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  template <typename... Args>
  TRIBOL_HOST_DEVICE void emplace_back( Args&&... args )
  {
    if ( size() >= capacity() ) {
      memory_ = resizer_.resize( memory_ );
      addOneToEnd( std::forward<Args>( args )... );
    } else {
      addOneToEnd( std::forward<Args>( args )... );
    }
  }
  using BaseClass::pop_back;
  TRIBOL_HOST_DEVICE void resize( size_type new_size )
  {
    if ( resizer_.resizeNeeded( new_size, memory_ ) ) {
      memory_ = resizer_.resize( memory_ );
    }
    BaseClass::resize( new_size );
  }

  TRIBOL_HOST_DEVICE void reserve( size_type new_capacity )
  {
    if ( new_capacity > memory_.capacity() ) {
      memory_ = resizer_.resize( memory_, new_capacity );
    }
  }

 private:
  template <typename... Args>
  TRIBOL_HOST_DEVICE void addOneToEnd( Args&&... args )
  {
    ::new ( memory_.data() + size() ) T( std::forward<Args>( args )... );
    memory_.setSize( size() + 1 );
  }

  using BaseClass::memory_;
  ArrayResizer<T, Allocator> resizer_;
};

template <typename T, class Allocator = DynamicAllocator<T>>
class Array2D : public BoundedArray2D<T, AllocatedMemory<T, Allocator>> {
 public:
  using value_type = T;
  using BaseClass = BoundedArray2D<T, AllocatedMemory<T, Allocator>>;
  using typename BaseClass::size_type;

  static_assert( std::is_same<typename Allocator::value_type, value_type>::value,
                 "Allocator must be used with same type as Array" );

  constexpr static size_type default_height_capacity_ = 0;

  TRIBOL_HOST_DEVICE Array2D( size_type height, size_type width, size_type height_capacity = default_height_capacity_ )
      : BaseClass( height, width, height_capacity > height ? height_capacity : height ), resizer_( width )
  {
  }
  TRIBOL_HOST_DEVICE Array2D() : Array2D( 0, 1, 0 ) {}

  // constructor with argument forwarding for memory
  template <typename... Args>
  TRIBOL_HOST_DEVICE Array2D( size_type height, size_type width, size_type height_capacity, Args&&... args )
      : BaseClass( height, width, height_capacity > height ? height_capacity : height, std::forward<Args>( args )... ),
        resizer_( width )
  {
  }

  // copy constructor with a custom allocator
  template <typename Allocator2>
  TRIBOL_HOST_DEVICE Array2D( const Array2D<T, Allocator2>& other, Allocator&& allocator )
      : BaseClass( other, AllocatedMemory<T, Allocator>( other.size(), other.capacity(), std::move( allocator ) ) )
  {
  }

  using BaseClass::capacity;
  using BaseClass::size;

  using BaseClass::width;

  TRIBOL_HOST_DEVICE void push_back( std::initializer_list<T> values )
  {
    if ( resizer_.resizeNeeded( size() + width(), memory_ ) ) {
      memory_ = resizer_.resize( memory_ );
      max_height_ = capacity() / width();
    }
    BaseClass::push_back( values );
  }

 private:
  using BaseClass::max_height_;
  using BaseClass::memory_;
  ArrayResizer<T, Allocator> resizer_;
};

template <typename T, MemorySpace MSPACE, typename SizeT = IndexT>
using HostArray = Array<T,
#ifdef TRIBOL_USE_UMPIRE
                        UmpireAllocator<T, MSPACE>
#else
                        Allocator<T>
#endif
                        >;

template <typename T, template <typename, typename> class ArrayT = Array>
class ManagedArray {
 public:
  using array_type = ArrayT<T, DynamicAllocator<T>>;
  using memory_view_type = Memory<typename array_type::memory_type::accessor_type>;
  using view_type = ArrayBase<memory_view_type>;

  ManagedArray( array_type&& host_array, int managed_allocator_id )
      : host_array_( std::move( host_array ) ),
        same_array_( host_array_.memory().allocator().id() == managed_allocator_id ),
        local_managed_array_( same_array_ ? array_type()
                                          : array_type( host_array_, DynamicAllocator<T>( managed_allocator_id ) ) ),
        managed_array_( same_array_ ? host_array_ : local_managed_array_ ),
        host_array_synced_( true ),
        managed_array_synced_( true )
  {
  }

  const array_type& hostRead() const
  {
    if ( !same_array_ && !host_array_synced_ ) {
      host_array_ = array_type( local_managed_array_, DynamicAllocator<T>( host_array_.memory().allocator().id() ) );
      host_array_synced_ = true;
      managed_array_synced_ = true;
    }
    return host_array_;
  }

  array_type& hostWrite( bool skip_sync = false )
  {
    if ( !same_array_ ) {
      // even if we're just writing, this should be synced in case we don't write to all elements
      if ( !skip_sync && !host_array_synced_ ) {
        host_array_ = array_type( local_managed_array_, DynamicAllocator<T>( host_array_.memory().allocator().id() ) );
      }
      host_array_synced_ = true;
      managed_array_synced_ = false;
    }
    return host_array_;
  }

  array_type& hostReadWrite()
  {
    if ( !same_array_ ) {
      if ( !host_array_synced_ ) {
        host_array_ = array_type( local_managed_array_, DynamicAllocator<T>( host_array_.memory().allocator().id() ) );
      }
      host_array_synced_ = true;
      managed_array_synced_ = false;
    }
    return host_array_;
  }

  const view_type managedRead() const
  {
    if ( !same_array_ && !managed_array_synced_ ) {
      local_managed_array_ =
          array_type( host_array_, DynamicAllocator<T>( local_managed_array_.memory().allocator().id() ) );
      managed_array_synced_ = true;
      host_array_synced_ = true;
    }
    return view_type( managed_array_.memory().view() );
  }

  view_type managedWrite( bool skip_sync = false )
  {
    if ( !same_array_ ) {
      // even if we're just writing, this should be synced in case we don't write to all elements
      if ( !skip_sync && !managed_array_synced_ ) {
        local_managed_array_ =
            array_type( host_array_, DynamicAllocator<T>( local_managed_array_.memory().allocator().id() ) );
      }
      managed_array_synced_ = true;
      host_array_synced_ = false;
    }
    return view_type( managed_array_.memory().view() );
  }

  view_type managedReadWrite()
  {
    if ( !same_array_ ) {
      if ( !managed_array_synced_ ) {
        local_managed_array_ =
            array_type( host_array_, DynamicAllocator<T>( local_managed_array_.memory().allocator().id() ) );
        ;
      }
      managed_array_synced_ = true;
      host_array_synced_ = false;
    }
    return view_type( managed_array_.memory().view() );
  }

  bool sameArray() const { return same_array_; }

 private:
  mutable array_type host_array_;
  bool same_array_;
  mutable array_type local_managed_array_;
  array_type& managed_array_;
  mutable bool host_array_synced_;
  mutable bool managed_array_synced_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_COMMON_ARRAYS_HPP_ */
