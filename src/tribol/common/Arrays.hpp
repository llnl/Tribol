// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_ARRAYS_HPP_
#define SRC_TRIBOL_COMMON_ARRAYS_HPP_

// Tribol config include
#include "tribol/config.hpp"

// C includes
#include <cassert>

// Tribol includes
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Memory.hpp"

namespace tribol {

/**
 * @brief Base class template for array-like containers that use a memory management policy
 * @tparam _MemoryT The memory management policy type
 */
template <typename _MemoryT>
class ArrayBase {
 public:
  /** @brief Type of values stored in the array */
  using ValueT_ = typename _MemoryT::ValueT_;

  /** @brief Pointer type for array elements */
  using PointerT_ = typename _MemoryT::PointerT_;

  /** @brief Const pointer type for array elements */
  using ConstPointerT_ = typename _MemoryT::ConstPointerT_;

  /** @brief Memory management policy type */
  using MemoryT_ = _MemoryT;

  /**
   * @brief Constructs array with given memory policy
   * @param memory Memory policy to use
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE ArrayBase( MemoryT_&& memory ) : memory_( std::move( memory ) )
  {
    // initialize memory if needed
    if constexpr ( !MemoryT_::IsInitializedT_::value ) {
      for ( auto& value : memory_ ) {
        value = ValueT_();
      }
    }
  }

  /**
   * @brief Copy constructs array with different memory policy
   * @param other Source array to copy from
   * @param memory Memory policy for new array
   */
  template <typename _Memory2T>
  TRIBOL_HOST_DEVICE ArrayBase( const ArrayBase<_Memory2T>& other, MemoryT_&& memory )
      : memory_( other.memory(), std::move( memory ) )
  {
  }

  /**
   * @brief Destructor that handles cleanup of uninitialized memory
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE ~ArrayBase()
  {
    // call destructor on all elements if needed
    if constexpr ( !MemoryT_::IsInitializedT_::value ) {
      for ( auto& value : memory_ ) {
        value.~ValueT_();
      }
    }
  }

  /** @brief Copy constructor */
  TRIBOL_DEFAULT_HOST_DEVICE ArrayBase( const ArrayBase& other ) = default;

  /** @brief Move constructor */
  TRIBOL_DEFAULT_HOST_DEVICE ArrayBase( ArrayBase&& other ) = default;

  /** @brief Copy assignment operator */
  TRIBOL_DEFAULT_HOST_DEVICE ArrayBase& operator=( const ArrayBase& other ) = default;

  /** @brief Move assignment operator */
  TRIBOL_DEFAULT_HOST_DEVICE ArrayBase& operator=( ArrayBase&& other ) = default;

  /**
   * @brief Access element with bounds checking
   * @param i Index of element
   * @return Reference to element
   */
  TRIBOL_HOST_DEVICE ValueT_& at( SizeT i ) { return memory_.at( i ); }

  /**
   * @brief Access const element with bounds checking
   * @param i Index of element
   * @return Const reference to element
   */
  TRIBOL_HOST_DEVICE const ValueT_& at( SizeT i ) const { return memory_.at( i ); }

  /**
   * @brief Array subscript operator
   * @param i Index of element
   * @return Reference to element
   */
  TRIBOL_HOST_DEVICE ValueT_& operator[]( SizeT i ) { return memory_.at( i ); }

  /**
   * @brief Const array subscript operator
   * @param i Index of element
   * @return Const reference to element
   */
  TRIBOL_HOST_DEVICE const ValueT_& operator[]( SizeT i ) const { return memory_.at( i ); }

  /** @brief Iterator type */
  using IteratorT_ = typename MemoryT_::IteratorT_;

  /** @brief Const iterator type */
  using ConstIteratorT_ = typename MemoryT_::ConstIteratorT_;

  /** @brief Get iterator to beginning */
  TRIBOL_HOST_DEVICE IteratorT_ begin() { return memory_.begin(); }

  /** @brief Get iterator to end */
  TRIBOL_HOST_DEVICE IteratorT_ end() { return memory_.end(); }

  /** @brief Get const iterator to beginning */
  TRIBOL_HOST_DEVICE ConstIteratorT_ begin() const { return memory_.begin(); }

  /** @brief Get const iterator to end */
  TRIBOL_HOST_DEVICE ConstIteratorT_ end() const { return memory_.end(); }

  /** @brief Get reference to memory policy */
  TRIBOL_HOST_DEVICE MemoryT_& memory() { return memory_; }

  /** @brief Get const reference to memory policy */
  TRIBOL_HOST_DEVICE const MemoryT_& memory() const { return memory_; }

  /** @brief Convert to memory view type */
  TRIBOL_HOST_DEVICE operator typename MemoryT_::ViewT_() const { return memory_; }

 protected:
  /** @brief Memory policy instance */
  MemoryT_ memory_;
};

template <typename _T, SizeT _N, class _MemoryT = StackMemory<_T, _N>>
class FixedArray : public ArrayBase<_MemoryT> {
 public:
  using BaseClassT_ = ArrayBase<_MemoryT>;
  using typename BaseClassT_::MemoryT_;
  using typename BaseClassT_::ValueT_;

  static_assert( std::is_same<ValueT_, _T>::value, "BoundedArray must be used with same type as memory" );

  TRIBOL_HOST_DEVICE FixedArray( MemoryT_&& memory = MemoryT_( _N ) ) : BaseClassT_( std::move( memory ) ) {}

  TRIBOL_HOST_DEVICE constexpr SizeT size() const { return _N; }

 private:
  using BaseClassT_::memory_;
};

template <typename _T, class _MemoryT = AllocatedMemory<_T>>
class BoundedArray : public ArrayBase<_MemoryT> {
 public:
  using BaseClassT_ = ArrayBase<_MemoryT>;
  using typename BaseClassT_::MemoryT_;
  using typename BaseClassT_::ValueT_;

  static_assert( !_MemoryT::IsSizeEqCapacityT_::value, "BoundedArray must be used with non-fixed size memory" );
  static_assert( std::is_same<ValueT_, _T>::value, "BoundedArray must be used with same type as memory" );

  TRIBOL_HOST_DEVICE BoundedArray( SizeT size, SizeT capacity ) : BaseClassT_( _MemoryT( size, capacity ) ) {}
  TRIBOL_HOST_DEVICE BoundedArray( _MemoryT&& memory ) : BaseClassT_( std::move( memory ) ) {}

  using BaseClassT_::at;

  TRIBOL_HOST_DEVICE constexpr SizeT size() const { return memory_.size(); }
  TRIBOL_HOST_DEVICE constexpr SizeT capacity() const { return memory_.capacity(); }

  template <typename... _ArgsT>
  TRIBOL_HOST_DEVICE inline void emplace( SizeT i, _ArgsT&&... args )
  {
    if ( i >= size() ) {
      i = memory_.setSize( i + 1 ) - 1;
    }
    ::new ( static_cast<void*>( memory_.data() + i ) ) ValueT_( std::forward<_ArgsT>( args )... );
  };
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE void push_back( ValueT_ value ) { emplace_back( std::move( value ) ); }
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  template <typename... _ArgsT>
  TRIBOL_HOST_DEVICE void emplace_back( _ArgsT&&... args )
  {
    emplace( size(), std::forward<_ArgsT>( args )... );
  }
  TRIBOL_HOST_DEVICE void pop_back() { memory_.setSize( size() - 1 ); }
  TRIBOL_HOST_DEVICE void resize( SizeT new_size )
  {
    assert( new_size <= capacity() );
    // destruct elements no longer in range
    for ( SizeT i{ new_size }; i < size(); ++i ) {
      at( i ).~ValueT_();
    }
    // create empty new elements
    for ( SizeT i{ size() }; i < new_size; ++i ) {
      at( i ) = ValueT_();
    }
    memory_.setSize( new_size );
  }

 protected:
  template <typename _Memory2T>
  TRIBOL_HOST_DEVICE BoundedArray( const BoundedArray<ValueT_, _Memory2T>& other, _Memory2T&& memory )
      : BaseClassT_( other, std::move( memory ) )
  {
  }

  using BaseClassT_::memory_;
};

template <typename _T, class _MemoryT = AllocatedMemory<_T>>
class BoundedArray2D : public BoundedArray<_T, _MemoryT> {
 public:
  using BaseClassT_ = BoundedArray<_T, _MemoryT>;
  using typename BaseClassT_::MemoryT_;
  using typename BaseClassT_::ValueT_;

  TRIBOL_HOST_DEVICE BoundedArray2D( SizeT height, SizeT width, SizeT max_height )
      : BaseClassT_( height * width, max_height * width ), height_( height ), width_( width ), max_height_( max_height )
  {
    assert( size() == height * width );
    assert( capacity() == max_height * width );
  }
  TRIBOL_HOST_DEVICE BoundedArray2D( SizeT height, SizeT width ) : BoundedArray2D( height, width, height ) {}
  TRIBOL_HOST_DEVICE BoundedArray2D() : BoundedArray2D( 0, 0, 0 )  // default constructor initializes to empty array
  {
  }

  // constructor with forwarded arguments for memory
  template <typename... _ArgsT>
  TRIBOL_HOST_DEVICE BoundedArray2D( SizeT height, SizeT width, SizeT max_height, _ArgsT&&... args )
      : BaseClassT_( MemoryT_( height * width, max_height * width, std::forward<_ArgsT>( args )... ) ),
        height_( height ),
        width_( width ),
        max_height_( height )
  {
    assert( height >= 0 && width >= 0 );
    assert( size() == height * width );
    assert( capacity() == max_height_ * width );
  }

  using BaseClassT_::at;

  TRIBOL_HOST_DEVICE ValueT_& at( SizeT i, SizeT j )
  {
    assert( i < height_ && j < width_ );
    return at( i * width_ + j );
  }
  TRIBOL_HOST_DEVICE const ValueT_& at( SizeT i, SizeT j ) const
  {
    assert( i < height_ && j < width_ );
    return at( i * width_ + j );
  }
  TRIBOL_HOST_DEVICE ValueT_& operator()( SizeT i, SizeT j ) { return at( i, j ); }
  TRIBOL_HOST_DEVICE const ValueT_& operator()( SizeT i, SizeT j ) const { return at( i, j ); }

  using BaseClassT_::capacity;
  using BaseClassT_::size;

  TRIBOL_HOST_DEVICE SizeT height() const { return height_; }
  TRIBOL_HOST_DEVICE SizeT width() const { return width_; }
  TRIBOL_HOST_DEVICE SizeT max_height() const { return max_height_; }

  TRIBOL_HOST_DEVICE void push_back( std::initializer_list<ValueT_> values )
  {
    assert( values.size() == width() );
    assert( height() < max_height_ );
    ++height_;
    memory_.setSize( height() * width() );
    SizeT j = 0;
    for ( auto& value : values ) {
      at( height() - 1, j++ ) = value;
    }
  }

  TRIBOL_HOST_DEVICE BoundedArray<ValueT_, Memory<typename MemoryT_::AccessorT_>> rowView( SizeT i )
  {
    assert( i < height_ );
    return BoundedArray<ValueT_, Memory<typename MemoryT_::AccessorT_>>(
        Memory<typename MemoryT_::AccessorT_>( &at( i, 0 ), width_, width_, memory_.stride() ) );
  }

  TRIBOL_HOST_DEVICE BoundedArray<ValueT_, Memory<FixedStride<ValueT_, typename MemoryT_::SizeAndCapacityT_>>> colView(
      SizeT j )
  {
    assert( j < width_ );
    return BoundedArray<ValueT_, Memory<FixedStride<ValueT_, typename MemoryT_::SizeAndCapacityT_>>>(
        Memory<FixedStride<ValueT_, typename MemoryT_::SizeAndCapacityT_>>( &at( 0, j ), height_, height_,
                                                                            memory_.stride() * width_ ) );
  }

 protected:
  template <typename _Memory2T>
  TRIBOL_HOST_DEVICE BoundedArray2D( const BoundedArray2D<ValueT_, _Memory2T>& other, MemoryT_&& memory )
      : BaseClassT_( other, std::move( memory ) ),
        height_( other.height() ),
        width_( other.width() ),
        max_height_( other.max_height() )
  {
    assert( size() == height_ * width_ );
    assert( capacity() == max_height_ * width_ );
  }

  using BaseClassT_::emplace_back;
  using BaseClassT_::pop_back;
  using BaseClassT_::push_back;
  using BaseClassT_::resize;

  using BaseClassT_::memory_;
  SizeT height_;
  SizeT width_;
  SizeT max_height_;
};

template <typename _T, class _AllocatorT = Allocator<_T>>
class ArrayResizer {
 public:
  using ValueT_ = _T;
  using AllocatorT_ = _AllocatorT;
  constexpr static RealT default_resize_ratio_ = 2.0;

  TRIBOL_HOST_DEVICE ArrayResizer( SizeT min_delta_capacity = 1 ) : min_delta_capacity_( min_delta_capacity ) {}

  TRIBOL_HOST_DEVICE bool resizeNeeded( SizeT new_size, const AllocatedMemory<ValueT_, AllocatorT_>& memory )
  {
    return new_size > memory.capacity();
  }

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory<ValueT_, AllocatorT_> resize(
      const AllocatedMemory<ValueT_, AllocatorT_>& old_memory )
  {
    SizeT new_capacity = static_cast<SizeT>( old_memory.capacity() * default_resize_ratio_ );
    new_capacity = new_capacity > ( old_memory.capacity() + min_delta_capacity_ )
                       ? new_capacity
                       : ( old_memory.capacity() + min_delta_capacity_ );
    return resize( old_memory, new_capacity );
  }

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory<ValueT_, AllocatorT_> resize(
      const AllocatedMemory<ValueT_, AllocatorT_>& old_memory, SizeT new_capacity )
  {
    assert( new_capacity > old_memory.capacity() );
    AllocatedMemory<ValueT_, AllocatorT_> new_memory( old_memory.size(), new_capacity );
    old_memory.allocator().uninitialized_copy( old_memory.data(), old_memory.data() + old_memory.size(),
                                               new_memory.data() );
    return std::move( new_memory );
  }

 private:
  SizeT min_delta_capacity_;
};

template <typename _T, class _AllocatorT = DynamicAllocator<_T>>
class Array : public BoundedArray<_T, AllocatedMemory<_T, _AllocatorT>> {
 public:
  using ValueT_ = _T;
  using BaseClassT_ = BoundedArray<_T, AllocatedMemory<_T, _AllocatorT>>;
  using typename BaseClassT_::MemoryT_;
  using typename BaseClassT_::PointerT_;
  using AllocatorT_ = _AllocatorT;

  constexpr static SizeT default_capacity_ = 0;

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE Array( SizeT size = 0, SizeT capacity = default_capacity_ )
      : BaseClassT_( MemoryT_( size, capacity >= size ? capacity : size ) ), resizer_( 1 )
  {
  }

  // copy constructor with a custom allocator
  template <typename _Allocator2T>
  TRIBOL_HOST_DEVICE Array( const Array<ValueT_, _Allocator2T>& other, AllocatorT_&& allocator )
      : BaseClassT_( other,
                     AllocatedMemory<ValueT_, AllocatorT_>( other.size(), other.capacity(), std::move( allocator ) ) )
  {
  }

  using BaseClassT_::capacity;
  using BaseClassT_::size;

  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE void push_back( ValueT_ value ) { emplace_back( std::move( value ) ); }
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  template <typename... _ArgsT>
  TRIBOL_HOST_DEVICE void emplace_back( _ArgsT&&... args )
  {
    if ( size() >= capacity() ) {
      memory_ = resizer_.resize( memory_ );
      addOneToEnd( std::forward<_ArgsT>( args )... );
    } else {
      addOneToEnd( std::forward<_ArgsT>( args )... );
    }
  }
  using BaseClassT_::pop_back;
  TRIBOL_HOST_DEVICE void resize( SizeT new_size )
  {
    if ( resizer_.resizeNeeded( new_size, memory_ ) ) {
      memory_ = resizer_.resize( memory_ );
    }
    BaseClassT_::resize( new_size );
  }

  TRIBOL_HOST_DEVICE void reserve( SizeT new_capacity )
  {
    if ( new_capacity > memory_.capacity() ) {
      memory_ = resizer_.resize( memory_, new_capacity );
    }
  }

 private:
  template <typename... _ArgsT>
  TRIBOL_HOST_DEVICE void addOneToEnd( _ArgsT&&... args )
  {
    ::new ( memory_.data() + size() ) ValueT_( std::forward<_ArgsT>( args )... );
    memory_.setSize( size() + 1 );
  }

  using BaseClassT_::memory_;
  ArrayResizer<ValueT_, AllocatorT_> resizer_;
};

template <typename _T, class _AllocatorT = DynamicAllocator<_T>>
class Array2D : public BoundedArray2D<_T, AllocatedMemory<_T, _AllocatorT>> {
 public:
  using BaseClassT_ = BoundedArray2D<_T, AllocatedMemory<_T, _AllocatorT>>;
  using typename BaseClassT_::ValueT_;
  using AllocatorT_ = _AllocatorT;

  static_assert( std::is_same<typename AllocatorT_::ValueT_, ValueT_>::value,
                 "Allocator must be used with same type as Array" );

  constexpr static SizeT default_height_capacity_ = 0;

  TRIBOL_HOST_DEVICE Array2D( SizeT height, SizeT width, SizeT height_capacity = default_height_capacity_ )
      : BaseClassT_( height, width, height_capacity > height ? height_capacity : height ), resizer_( width )
  {
  }
  TRIBOL_HOST_DEVICE Array2D() : Array2D( 0, 1, 0 ) {}

  // constructor with argument forwarding for memory
  template <typename... _ArgsT>
  TRIBOL_HOST_DEVICE Array2D( SizeT height, SizeT width, SizeT height_capacity, _ArgsT&&... args )
      : BaseClassT_( height, width, height_capacity > height ? height_capacity : height,
                     std::forward<_ArgsT>( args )... ),
        resizer_( width )
  {
  }

  // copy constructor with a custom allocator
  template <typename _Allocator2T>
  TRIBOL_HOST_DEVICE Array2D( const Array2D<ValueT_, _Allocator2T>& other, _Allocator2T&& allocator )
      : BaseClassT_( other,
                     AllocatedMemory<ValueT_, _Allocator2T>( other.size(), other.capacity(), std::move( allocator ) ) )
  {
  }

  using BaseClassT_::capacity;
  using BaseClassT_::size;

  using BaseClassT_::width;

  TRIBOL_HOST_DEVICE void push_back( std::initializer_list<ValueT_> values )
  {
    if ( resizer_.resizeNeeded( size() + width(), memory_ ) ) {
      memory_ = resizer_.resize( memory_ );
      max_height_ = capacity() / width();
    }
    BaseClassT_::push_back( values );
  }

 private:
  using BaseClassT_::max_height_;
  using BaseClassT_::memory_;
  ArrayResizer<ValueT_, AllocatorT_> resizer_;
};

template <typename _T, MemorySpace _Mem>
using HostArray = Array<_T,
#ifdef TRIBOL_USE_UMPIRE
                        UmpireAllocator<_T, _Mem>
#else
                        Allocator<_T>
#endif
                        >;

template <typename _T, template <typename, typename> class _ArrayT = Array>
class ManagedArray {
 public:
  using ArrayT_ = _ArrayT<_T, DynamicAllocator<_T>>;
  using MemoryViewT_ = Memory<typename ArrayT_::MemoryT_::AccessorT_>;
  using ViewT_ = ArrayBase<MemoryViewT_>;
  using ValueT_ = _T;

  ManagedArray( ArrayT_&& host_array, int managed_allocator_id )
      : host_array_( std::move( host_array ) ),
        same_array_( host_array_.memory().allocator().id() == managed_allocator_id ),
        local_managed_array_( host_array_, DynamicAllocator<_T>( managed_allocator_id ) ),
        managed_array_( same_array_ ? host_array_ : local_managed_array_ ),
        host_array_synced_( true ),
        managed_array_synced_( true )
  {
  }

  const ArrayT_& hostRead() const
  {
    if ( !same_array_ && !host_array_synced_ ) {
      host_array_ = ArrayT_( local_managed_array_, DynamicAllocator<_T>( host_array_.memory().allocator().id() ) );
      host_array_synced_ = true;
      managed_array_synced_ = true;
    }
    return host_array_;
  }

  ArrayT_& hostWrite( bool skip_sync = false )
  {
    if ( !same_array_ ) {
      // even if we're just writing, this should be synced in case we don't write to all elements
      if ( !skip_sync && !host_array_synced_ ) {
        host_array_ = ArrayT_( local_managed_array_, DynamicAllocator<_T>( host_array_.memory().allocator().id() ) );
      }
      host_array_synced_ = true;
      managed_array_synced_ = false;
    }
    return host_array_;
  }

  ArrayT_& hostReadWrite()
  {
    if ( !same_array_ ) {
      if ( !host_array_synced_ ) {
        host_array_ = ArrayT_( local_managed_array_, DynamicAllocator<_T>( host_array_.memory().allocator().id() ) );
      }
      host_array_synced_ = true;
      managed_array_synced_ = false;
    }
    return host_array_;
  }

  const ViewT_ managedRead() const
  {
    if ( !same_array_ && !managed_array_synced_ ) {
      local_managed_array_ =
          ArrayT_( host_array_, DynamicAllocator<ValueT_>( local_managed_array_.memory().allocator().id() ) );
      managed_array_synced_ = true;
      host_array_synced_ = true;
    }
    return ViewT_( managed_array_.memory().view() );
  }

  ViewT_ managedWrite( bool skip_sync = false )
  {
    if ( !same_array_ ) {
      // even if we're just writing, this should be synced in case we don't write to all elements
      if ( !skip_sync && !managed_array_synced_ ) {
        local_managed_array_ =
            ArrayT_( host_array_, DynamicAllocator<ValueT_>( local_managed_array_.memory().allocator().id() ) );
      }
      managed_array_synced_ = true;
      host_array_synced_ = false;
    }
    return ViewT_( managed_array_.memory().view() );
  }

  ViewT_ managedReadWrite()
  {
    if ( !same_array_ ) {
      if ( !managed_array_synced_ ) {
        local_managed_array_ =
            ArrayT_( host_array_, DynamicAllocator<ValueT_>( local_managed_array_.memory().allocator().id() ) );
        ;
      }
      managed_array_synced_ = true;
      host_array_synced_ = false;
    }
    return ViewT_( managed_array_.memory().view() );
  }

  bool sameArray() const { return same_array_; }

 private:
  mutable ArrayT_ host_array_;
  bool same_array_;
  mutable ArrayT_ local_managed_array_;
  ArrayT_& managed_array_;
  mutable bool host_array_synced_;
  mutable bool managed_array_synced_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_COMMON_ARRAYS_HPP_ */
