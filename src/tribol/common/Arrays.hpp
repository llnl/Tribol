// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef TRIBOL_COMMON_ARRAYS_HPP_
#define TRIBOL_COMMON_ARRAYS_HPP_

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

  TRIBOL_HOST_DEVICE ArrayBase( MemoryT&& memory ) : memory_( std::move( memory ) )
  {
    // initialize memory if needed
    if constexpr ( MemoryT::initialized_ == false ) {
      for ( auto& value : memory_ ) {
        value = value_type{};
      }
    }
  }
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

template <typename T, IndexT N, class MemoryT = StackMemory<T, N>>
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

  TRIBOL_HOST_DEVICE void push_back( T value ) { at( memory_.setSize( size() + 1 ) - 1 ) = value; }
  template <typename... Args>
  TRIBOL_HOST_DEVICE void emplace_back( Args&&... args )
  {
    at( memory_.setSize( size() + 1 ) - 1 ) = T( std::forward<Args>( args )... );
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

  TRIBOL_HOST_DEVICE void push_back( std::initializer_list<T> values )
  {
    assert( values.size() == width() );
    assert( height() < max_height_ );
    ++height_;
    memory_.setSize( height() * width() );
    for ( size_type j( 0 ); j < width(); ++j ) {
      at( height() - 1, j ) = values[j];
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

 private:
  using BaseClass::emplace_back;
  using BaseClass::pop_back;
  using BaseClass::push_back;
  using BaseClass::resize;

  using BaseClass::memory_;
  size_type height_;
  size_type width_;
  size_type max_height_;
};

template <typename T, class Allocator = HeapAllocator<T>, typename SizeT = IndexT>
class ArrayResizer {
 public:
  constexpr static RealT default_resize_ratio_ = 2.0;

  TRIBOL_HOST_DEVICE bool resizeNeeded( SizeT new_size, const AllocatedMemory<T, Allocator, SizeT>& memory )
  {
    return new_size > memory.capacity();
  }

  TRIBOL_HOST_DEVICE AllocatedMemory<T, Allocator, SizeT> resize(
      const AllocatedMemory<T, Allocator, SizeT>& old_memory )
  {
    SizeT new_capacity = static_cast<SizeT>( old_memory.capacity() * default_resize_ratio_ );
    new_capacity = new_capacity > 0 ? new_capacity : 1;
    AllocatedMemory<T, Allocator, SizeT> new_memory( old_memory.size(), new_capacity );
    old_memory.allocator().copy( new_memory.data(), old_memory.data(), old_memory.size() );
    return std::move( new_memory );
  }
};

template <typename T, class Allocator = HeapAllocator<T>, typename SizeT = IndexT>
class Array : public BoundedArray<T, AllocatedMemory<T, Allocator, SizeT>> {
 public:
  using value_type = T;
  using BaseClass = BoundedArray<T, AllocatedMemory<T, Allocator>>;
  using typename BaseClass::size_type;

  static_assert( std::is_same<typename Allocator::value_type, value_type>::value,
                 "Allocator must be used with same type as Array" );

  constexpr static size_type default_capacity_ = 32;

#pragma nv_exec_check_disable
  TRIBOL_HOST_DEVICE Array( size_type size = 0, size_type capacity = default_capacity_ )
      : BaseClass( AllocatedMemory<T, Allocator>( size, capacity >= size ? capacity : size ) )
  {
  }

  using BaseClass::size;

  TRIBOL_HOST_DEVICE void push_back( T value )
  {
    if ( resizer_.resizeNeeded( size() + 1, memory_ ) ) {
      memory_ = resizer_.resize( memory_ );
    }
    BaseClass::push_back( value );
  }
  template <typename... Args>
  TRIBOL_HOST_DEVICE void emplace_back( Args&&... args )
  {
    if ( resizer_.resizeNeeded( size() + 1, memory_ ) ) {
      memory_ = resizer_.resize( memory_ );
    }
    BaseClass::emplace_back( std::forward<Args>( args )... );
  }
  using BaseClass::pop_back;
  TRIBOL_HOST_DEVICE void resize( size_type new_size )
  {
    if ( resizer_.resizeNeeded( new_size, memory_ ) ) {
      memory_ = resizer_.resize( memory_ );
    }
    BaseClass::resize( new_size );
  }

 private:
  using BaseClass::memory_;
  ArrayResizer<T, Allocator> resizer_;
};

}  // namespace tribol

#endif /* TRIBOL_COMMON_ARRAYS_HPP_ */
