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
 * @brief Base class template for array-like containers backed by a memory object.
 *
 * ArrayBase provides a thin container facade over a memory object type (`_MemoryT`). The memory type is expected to
 * provide element access, iteration, size/capacity semantics and a view type. This class centralizes common operations
 * for the array family (1D/2D, fixed/dynamic) such as element access, iterators and lifetime management of contained
 * values.
 *
 * @tparam _MemoryT Underlying memory type that provides storage and accessor semantics. Must expose `ValueT_`,
 * `PointerT_`, `ConstPointerT_`, `IteratorT_`, `ConstIteratorT_`, `ViewT_`, and `IsInitializedT_`.
 */
template <typename _MemoryT>
class ArrayBase {
 public:
  /// @brief Type of values stored in the array
  using ValueT_ = typename _MemoryT::ValueT_;

  /// @brief Type alias for the value type (for STL compatibility)
  using value_type = ValueT_;

  /// @brief Pointer type for array elements
  using PointerT_ = typename _MemoryT::PointerT_;

  /// @brief Const pointer type for array elements
  using ConstPointerT_ = typename _MemoryT::ConstPointerT_;

  /// @brief Type of memory holding underlying data
  using MemoryT_ = _MemoryT;

  /**
   * @brief Construct an ArrayBase that adopts the provided memory view.
   *
   * The constructor takes ownership (or a view, depending on `MemoryT_`) of the provided memory object. If the
   * underlying memory type indicates the storage is not initialized (`IsInitializedT_::value == false`), this
   * constructor will default initialize the contained elements. The constructor is annotated for host/device usage
   * where applicable.
   *
   * @param memory Memory object to wrap. The memory's element type must match `ValueT_`.
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
   * @brief Copy-construct from an ArrayBase with a different memory policy.
   *
   * This constructor creates a new ArrayBase by copying the view from `other` into the supplied `memory`. It is useful
   * for constructing array adapters that share the same logical contents but differ in how the underlying memory is
   * managed.
   *
   * @tparam _Memory2T Memory type of the source array.
   * @param other Source array to copy from.
   * @param memory Memory object (policy) to use for the newly-constructed array.
   */
  template <typename _Memory2T>
  TRIBOL_HOST_DEVICE ArrayBase( const ArrayBase<_Memory2T>& other, MemoryT_&& memory )
      : memory_( other.memory(), std::move( memory ) )
  {
  }

  /**
   * @brief Destructor.
   *
   * If the underlying memory type is not considered initialized (`IsInitializedT_` is false) the destructor will
   * explicitly call the element destructors to ensure proper cleanup. This preserves correct lifetime semantics for
   * types that require explicit destruction.
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

  /// @brief Copy constructor
  TRIBOL_DEFAULT_HOST_DEVICE ArrayBase( const ArrayBase& other ) = default;

  /// @brief Move constructor
  TRIBOL_DEFAULT_HOST_DEVICE ArrayBase( ArrayBase&& other ) = default;

  /// @brief Copy assignment operator
  TRIBOL_DEFAULT_HOST_DEVICE ArrayBase& operator=( const ArrayBase& other ) = default;

  /// @brief Move assignment operator
  TRIBOL_DEFAULT_HOST_DEVICE ArrayBase& operator=( ArrayBase&& other ) = default;

  /**
   * @brief Access element with bounds checking.
   *
   * This forwards to the underlying memory's `at()` which performs any bounds assertions appropriate for the build
   * configuration.
   *
   * @param i Index of the element to access (0-based).
   * @return Reference to the element at index `i`.
   */
  TRIBOL_HOST_DEVICE ValueT_& at( SizeT i ) { return memory_.at( i ); }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueT_& at( SizeT i ) const { return memory_.at( i ); }

  /**
   * @brief Subscript operator with bounds checking (delegates to `at`).
   *
   * @param i Index of the element to access.
   * @return Reference to the element at index `i`.
   */
  TRIBOL_HOST_DEVICE ValueT_& operator[]( SizeT i ) { return memory_.at( i ); }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueT_& operator[]( SizeT i ) const { return memory_.at( i ); }

  /// @brief Iterator type
  using IteratorT_ = typename MemoryT_::IteratorT_;

  /// @brief Const iterator type
  using ConstIteratorT_ = typename MemoryT_::ConstIteratorT_;

  /// @brief Get iterator to beginning
  TRIBOL_HOST_DEVICE IteratorT_ begin() { return memory_.begin(); }
  /// @overload
  TRIBOL_HOST_DEVICE ConstIteratorT_ begin() const { return memory_.begin(); }

  /// @brief Get iterator to end
  TRIBOL_HOST_DEVICE IteratorT_ end() { return memory_.end(); }
  /// @overload
  TRIBOL_HOST_DEVICE ConstIteratorT_ end() const { return memory_.end(); }

  /// @brief Get reference to underlying memory
  TRIBOL_HOST_DEVICE MemoryT_& memory() { return memory_; }
  /// @overload
  TRIBOL_HOST_DEVICE const MemoryT_& memory() const { return memory_; }

  /// @brief Convert to memory view type
  TRIBOL_HOST_DEVICE operator typename MemoryT_::ViewT_() const { return memory_; }

 protected:
  /// @brief Memory instance
  MemoryT_ memory_;
};

/**
 * @brief Fixed-size array container with compile-time size.
 *
 * FixedArray is a lightweight container for a statically-sized sequence of elements. The storage is provided by a
 * memory policy (defaulting to `StackMemory`) that allocates the required storage at compile-time. This class offers a
 * simple container interface with random access and sized iteration.
 *
 * @tparam _T Value type stored in the array.
 * @tparam _N Compile-time size (number of elements).
 * @tparam _MemoryT Memory policy type used to provide storage; defaults to
 *                   `StackMemory<_T, _N>`.
 *
 * Notes:
 * - The container size is equal to `_N` and cannot change at runtime.
 * - Elements are default-initialized in the constructor when the underlying memory
 *   indicates it is not pre-initialized.
 */
template <typename _T, SizeT _N, class _MemoryT = StackMemory<_T, _N>>
class FixedArray : public ArrayBase<_MemoryT> {
 public:
  /// @brief Base class type
  using BaseClassT_ = ArrayBase<_MemoryT>;

  /// @brief Memory type
  using typename BaseClassT_::MemoryT_;

  /// @brief Value type stored in the array
  using typename BaseClassT_::ValueT_;

  static_assert( std::is_same<ValueT_, _T>::value, "BoundedArray must be used with same type as memory" );

  /**
   * @brief Construct a FixedArray with default-initialized elements.
   *
   * The constructor creates the underlying memory object with size `_N`. If the memory type indicates it is not
   * pre-initialized, the constructor will default-initialize all elements.
   */
  TRIBOL_HOST_DEVICE FixedArray( MemoryT_&& memory = MemoryT_( _N ) ) : BaseClassT_( std::move( memory ) ) {}

  /**
   * @brief Get the size of the array.
   *
   * @return The compile-time size `_N`.
   */
  TRIBOL_HOST_DEVICE constexpr SizeT size() const { return _N; }

 private:
  /// @brief Using declaration to access the protected memory_ member from the base class.
  using BaseClassT_::memory_;
};

/**
 * @brief Dynamically-sized array with an upper bound (capacity).
 *
 * BoundedArray models a contiguous sequence of elements whose logical size may be less than or equal to a fixed
 * capacity. The capacity is provided by the underlying memory type. This container manages element construction and
 * destruction when the size changes and provides emplacement APIs to construct elements in-place.
 *
 * @tparam _T Value type stored in the array.
 * @tparam _MemoryT Memory policy type that must provide size/capacity management (e.g., `AllocatedMemory`). The memory
 * policy is expected to expose `size()`, `capacity()`, `data()`, `setSize()`, and `IsSizeEqCapacityT_`.
 */
template <typename _T, class _MemoryT = AllocatedMemory<_T>>
class BoundedArray : public ArrayBase<_MemoryT> {
 public:
  /// @brief Base class type
  using BaseClassT_ = ArrayBase<_MemoryT>;

  /// @brief Memory type
  using typename BaseClassT_::MemoryT_;

  /// @brief Value type stored in the array
  using typename BaseClassT_::ValueT_;

  static_assert( !_MemoryT::IsSizeEqCapacityT_::value, "BoundedArray must be used with non-fixed size memory" );
  static_assert( std::is_same<ValueT_, _T>::value, "BoundedArray must be used with same type as memory" );

  /**
   * @brief Construct a BoundedArray with explicit size and capacity.
   *
   * @param size Initial logical size (number of constructed elements).
   * @param capacity Maximum capacity (must be >= size).
   */
  TRIBOL_HOST_DEVICE BoundedArray( SizeT size, SizeT capacity ) : BaseClassT_( _MemoryT( size, capacity ) ) {}

  /**
   * @brief Construct from an existing memory object.
   *
   * This forwards the provided memory object to the base `ArrayBase` constructor.
   */
  TRIBOL_HOST_DEVICE BoundedArray( _MemoryT&& memory ) : BaseClassT_( std::move( memory ) ) {}

  /// @brief Using declaration to access the base class element accessor with bounds checking.
  using BaseClassT_::at;

  /// @brief Get the size of the array.
  TRIBOL_HOST_DEVICE constexpr SizeT size() const { return memory_.size(); }

  /// @brief Get the capacity of the array.
  TRIBOL_HOST_DEVICE constexpr SizeT capacity() const { return memory_.capacity(); }

  /**
   * @brief Emplace-construct an element at index `i`.
   *
   * If `i` is beyond the current size the logical size is increased to include the new element. The element is
   * constructed in-place using placement-new.
   *
   * @tparam _ArgsT Parameter pack forwarded to `ValueT_` constructor.
   * @param i Index at which to construct the element.
   * @param args Arguments forwarded to the element constructor.
   */
  template <typename... _ArgsT>
  TRIBOL_HOST_DEVICE inline void emplace( SizeT i, _ArgsT&&... args )
  {
    if ( i >= size() ) {
      i = memory_.setSize( i + 1 ) - 1;
    }
    ::new ( static_cast<void*>( memory_.data() + i ) ) ValueT_( std::forward<_ArgsT>( args )... );
  };

  /**
   * @brief Append a value by move/copy constructing it at the end of the array.
   *
   * @param value Value to append.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE void push_back( ValueT_ value ) { emplace_back( std::move( value ) ); }

  /**
   * @brief Emplace an element at the end of the array.
   *
   * The element is constructed in-place using the forwarded arguments.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  template <typename... _ArgsT>
  TRIBOL_HOST_DEVICE void emplace_back( _ArgsT&&... args )
  {
    emplace( size(), std::forward<_ArgsT>( args )... );
  }

  /**
   * @brief Remove the last element (reduce logical size by one).
   */
  TRIBOL_HOST_DEVICE void pop_back() { memory_.setSize( size() - 1 ); }

  /**
   * @brief Resize the logical size of the array.
   *
   * When shrinking, elements beyond `new_size` are destroyed. When growing, new elements are default-initialized. The
   * requested size must not exceed capacity.
   *
   * @param new_size Desired new logical size (must be <= capacity()).
   */
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
  /**
   * @brief Copy-construct from a BoundedArray with a different memory policy.
   *
   * This constructor creates a new BoundedArray by copying the view from `other` into the supplied `memory`. It is
   * useful for constructing array adapters that share the same logical contents but differ in how the underlying memory
   * is managed.
   *
   * @tparam _Memory2T Memory type of the source array.
   * @param other Source array to copy from.
   * @param memory Memory object (policy) to use for the newly-constructed array.
   */
  template <typename _Memory2T>
  TRIBOL_HOST_DEVICE BoundedArray( const BoundedArray<ValueT_, _Memory2T>& other, _Memory2T&& memory )
      : BaseClassT_( other, std::move( memory ) )
  {
  }

  /// @brief Using declaration to access the protected memory_ member from the base class.
  using BaseClassT_::memory_;
};

/**
 * @brief Two-dimensional bounded array with row/column views.
 *
 * BoundedArray2D stores elements in row-major order with a logical height and width and an optional maximum height (for
 * push_back growth of rows). It provides convenient indexing via `at(i,j)` and `operator()` and can return row/column
 * views as `BoundedArray` instances that reference the underlying storage.
 *
 * @tparam _T Value type stored in the 2D array.
 * @tparam _MemoryT Memory policy providing contiguous backing storage with stride support.
 */
template <typename _T, class _MemoryT = AllocatedMemory<_T>>
class BoundedArray2D : public BoundedArray<_T, _MemoryT> {
 public:
  /// @brief Base class type
  using BaseClassT_ = BoundedArray<_T, _MemoryT>;

  /// @brief Memory type
  using typename BaseClassT_::MemoryT_;

  /// @brief Value type stored in the array
  using typename BaseClassT_::ValueT_;

  /**
   * @brief Construct a BoundedArray2D with explicit sizes and capacity.
   *
   * @param height Logical height (number of rows).
   * @param width Number of columns per row.
   * @param max_height Maximum allowed height (capacity in rows).
   */
  TRIBOL_HOST_DEVICE BoundedArray2D( SizeT height, SizeT width, SizeT max_height )
      : BaseClassT_( height * width, max_height * width ), height_( height ), width_( width ), max_height_( max_height )
  {
    assert( size() == height * width );
    assert( capacity() == max_height * width );
  }

  /**
   * @brief Construct with height == max_height (capacity equals height).
   */
  TRIBOL_HOST_DEVICE BoundedArray2D( SizeT height, SizeT width ) : BoundedArray2D( height, width, height ) {}

  /**
   * @brief Default-construct an empty 2D array.
   */
  TRIBOL_HOST_DEVICE BoundedArray2D() : BoundedArray2D( 0, 0, 0 )  // default constructor initializes to empty array
  {
  }

  /**
   * @brief Constructor that forwards allocator or memory arguments to the underlying memory type.
   *
   * This allows constructing the backing `MemoryT_` with arguments such as an allocator ID or allocator object.
   */
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

  /// @brief Using declaration to access the base class element accessor with bounds checking.
  using BaseClassT_::at;

  /**
   * @brief Access element at row `i` and column `j` with bounds checking.
   *
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @return Reference to the element at (i, j).
   */
  TRIBOL_HOST_DEVICE ValueT_& at( SizeT i, SizeT j )
  {
    assert( i < height_ && j < width_ );
    return at( i * width_ + j );
  }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueT_& at( SizeT i, SizeT j ) const
  {
    assert( i < height_ && j < width_ );
    return at( i * width_ + j );
  }

  /**
   * @brief Convenience call operator for element access.
   *
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @return Reference to the element at (i, j).
   */
  TRIBOL_HOST_DEVICE ValueT_& operator()( SizeT i, SizeT j ) { return at( i, j ); }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueT_& operator()( SizeT i, SizeT j ) const { return at( i, j ); }

  /// @brief Using declaration to access base class capacity.
  using BaseClassT_::capacity;

  /// @brief Using declaration to access base class size.
  using BaseClassT_::size;

  /// @brief Return the logical height (number of rows).
  TRIBOL_HOST_DEVICE SizeT height() const { return height_; }

  /// @brief Return the number of columns (width).
  TRIBOL_HOST_DEVICE SizeT width() const { return width_; }

  /// @brief Return the maximum height (capacity in rows).
  TRIBOL_HOST_DEVICE SizeT max_height() const { return max_height_; }

  /**
   * @brief Append a row given an initializer list of width() values.
   *
   * The list must contain exactly `width()` elements. The height must be less than `max_height()` before calling this
   * method.
   *
   * @param values Values for the new row.
   */
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

  /**
   * @brief Return a view of row `i` as a BoundedArray (non-owning view).
   *
   * The returned BoundedArray references the existing storage; it does not own it.
   *
   * @param i Row index to view.
   */
  TRIBOL_HOST_DEVICE BoundedArray<ValueT_, Memory<typename MemoryT_::AccessorT_>> rowView( SizeT i )
  {
    assert( i < height_ );
    return BoundedArray<ValueT_, Memory<typename MemoryT_::AccessorT_>>(
        Memory<typename MemoryT_::AccessorT_>( &at( i, 0 ), width_, width_, memory_.stride() ) );
  }

  /**
   * @brief Return a view of column `j` as a BoundedArray (non-owning view).
   *
   * The column view uses a `FixedStride` accessor so elements are accessed with a stride equal to `memory_.stride() *
   * width()`.
   *
   * @param j Column index to view.
   */
  TRIBOL_HOST_DEVICE BoundedArray<ValueT_, Memory<FixedStride<ValueT_, typename MemoryT_::SizeAndCapacityT_>>> colView(
      SizeT j )
  {
    assert( j < width_ );
    return BoundedArray<ValueT_, Memory<FixedStride<ValueT_, typename MemoryT_::SizeAndCapacityT_>>>(
        Memory<FixedStride<ValueT_, typename MemoryT_::SizeAndCapacityT_>>( &at( 0, j ), height_, height_,
                                                                            memory_.stride() * width_ ) );
  }

 protected:
  /**
   * @brief Copy-construct from a BoundedArray2D with a different memory policy.
   *
   * This constructor creates a new BoundedArray2D by copying the view from `other` into the supplied `memory`. It is
   * useful for constructing array adapters that share the same logical contents but differ in how the underlying memory
   * is managed.
   *
   * @tparam _Memory2T Memory type of the source array.
   * @param other Source array to copy from.
   * @param memory Memory object (policy) to use for the newly-constructed array.
   */
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

  /// @brief Using declaration to access base class element emplacement.
  using BaseClassT_::emplace_back;

  /// @brief Using declaration to access base class pop_back.
  using BaseClassT_::pop_back;

  /// @brief Using declaration to access base class push_back.
  using BaseClassT_::push_back;

  /// @brief Using declaration to access base class resize.
  using BaseClassT_::resize;

  /// @brief Using declaration to access the protected memory_ member from the base class.
  using BaseClassT_::memory_;

  /// @brief Logical height (number of rows).
  SizeT height_;

  /// @brief Number of columns (width).
  SizeT width_;

  /// @brief Maximum height (capacity in rows).
  SizeT max_height_;
};

/**
 * @brief Policy class responsible for growing an AllocatedMemory when capacity is insufficient.
 *
 * ArrayResizer implements a resize growth strategy used by `Array` and other dynamic containers. By default it grows
 * capacity by a ratio (default_resize_ratio_) and ensures a minimal delta is applied when requested.
 *
 * @tparam _T Element type.
 * @tparam _AllocatorT Allocator type used by the AllocatedMemory instances.
 */
template <typename _T, class _AllocatorT = Allocator<_T>>
class ArrayResizer {
 public:
  using ValueT_ = _T;
  using AllocatorT_ = _AllocatorT;
  constexpr static RealT default_resize_ratio_ = 2.0;

  /**
   * @brief Construct an ArrayResizer.
   *
   * @param min_delta_capacity Minimum additional capacity to add when resizing.
   */
  TRIBOL_HOST_DEVICE ArrayResizer( SizeT min_delta_capacity = 1 ) : min_delta_capacity_( min_delta_capacity ) {}

  /**
   * @brief Query whether resizing is needed to accommodate `new_size`.
   *
   * @param new_size Desired new logical size.
   * @param memory Existing allocated memory to compare against.
   * @return true if `new_size` exceeds current capacity.
   */
  TRIBOL_HOST_DEVICE bool resizeNeeded( SizeT new_size, const AllocatedMemory<ValueT_, AllocatorT_>& memory )
  {
    return new_size > memory.capacity();
  }

  /**
   * @brief Resize using the default growth ratio.
   *
   * @param old_memory Existing memory to copy from.
   *
   * This computes a new capacity based on `default_resize_ratio_` and ensures that the capacity increases by at least
   * `min_delta_capacity_` before delegating to the overload that performs the allocation and copy.
   */
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

  /**
   * @brief Resize to an explicit new capacity and copy existing elements.
   *
   * Allocates a new `AllocatedMemory`, copies the existing constructed elements via the allocator's
   * `uninitialized_copy`, and returns the new memory object.
   *
   * @param old_memory Existing memory to copy from.
   * @param new_capacity Desired capacity (must be larger than the current capacity).
   * @return Newly allocated memory containing the copied elements.
   */
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
  /// @brief Minimum additional capacity to add when resizing.
  SizeT min_delta_capacity_;
};

/**
 * @brief Dynamically-resizable 1D array container.
 *
 * Array provides a vector-like interface backed by an `AllocatedMemory` instance and an `ArrayResizer` policy that
 * expands capacity when needed. Elements are constructed in-place; resizing and reserve operations are provided.
 *
 * @tparam _T Value type stored in the array.
 * @tparam _AllocatorT Allocator type to use for dynamic allocations (defaults to `DynamicAllocator<_T>`).
 */
template <typename _T, class _AllocatorT = DynamicAllocator<_T>>
class Array : public BoundedArray<_T, AllocatedMemory<_T, _AllocatorT>> {
 public:
  /// @brief Value type stored in the array
  using ValueT_ = _T;

  /// @brief Base class type
  using BaseClassT_ = BoundedArray<_T, AllocatedMemory<_T, _AllocatorT>>;

  /// @brief Memory type used by the array
  using typename BaseClassT_::MemoryT_;

  /// @brief Pointer type for array elements
  using typename BaseClassT_::PointerT_;

  /// @brief Allocator type used for dynamic allocations
  using AllocatorT_ = _AllocatorT;

  /// @brief Default initial capacity
  constexpr static SizeT default_capacity_ = 0;

  /**
   * @brief Construct an Array with optional initial size and capacity.
   *
   * @param size Initial logical size (default 0).
   * @param capacity Initial capacity; if < size then capacity is set to size.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE Array( SizeT size = 0, SizeT capacity = default_capacity_ )
      : BaseClassT_( MemoryT_( size, capacity >= size ? capacity : size ) ), resizer_( 1 )
  {
  }

  /**
   * @brief Copy-construct using a custom allocator.
   *
   * This creates a copy of `other` but allocates storage with the provided allocator instance.
   */
  template <typename _Allocator2T>
  TRIBOL_HOST_DEVICE Array( const Array<ValueT_, _Allocator2T>& other, AllocatorT_&& allocator )
      : BaseClassT_( other,
                     AllocatedMemory<ValueT_, AllocatorT_>( other.size(), other.capacity(), std::move( allocator ) ) )
  {
  }

  /// @brief Using declaration to access base class capacity.
  using BaseClassT_::capacity;

  /// @brief Using declaration to access base class size.
  using BaseClassT_::size;

  /**
   * @brief Append a value to the end of the array.
   *
   * If there is insufficient capacity, the resizer policy is invoked to grow the backing memory before the value is
   * constructed in-place.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE void push_back( ValueT_ value ) { emplace_back( std::move( value ) ); }

  /**
   * @brief Emplace-construct an element at the end of the array.
   *
   * If needed, this method will grow the underlying storage before constructing the element in-place.
   */
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

  /// @brief Using declaration to access base class pop_back.
  using BaseClassT_::pop_back;

  /**
   * @brief Resize the array to `new_size`, growing capacity if necessary.
   *
   * @param new_size Desired logical size.
   */
  TRIBOL_HOST_DEVICE void resize( SizeT new_size )
  {
    if ( resizer_.resizeNeeded( new_size, memory_ ) ) {
      memory_ = resizer_.resize( memory_ );
    }
    BaseClassT_::resize( new_size );
  }

  /**
   * @brief Ensure capacity is at least `new_capacity`.
   *
   * If `new_capacity` is larger than current capacity this triggers an allocation and copy of existing elements.
   */
  TRIBOL_HOST_DEVICE void reserve( SizeT new_capacity )
  {
    if ( new_capacity > memory_.capacity() ) {
      memory_ = resizer_.resize( memory_, new_capacity );
    }
  }

 private:
  /// @brief Helper to construct an element at the current end position and update size.
  template <typename... _ArgsT>
  TRIBOL_HOST_DEVICE void addOneToEnd( _ArgsT&&... args )
  {
    ::new ( memory_.data() + size() ) ValueT_( std::forward<_ArgsT>( args )... );
    memory_.setSize( size() + 1 );
  }

  /// @brief Using declaration to access the protected memory_ member from the base class.
  using BaseClassT_::memory_;

  /// @brief Array resizer used to manage dynamic resizing.
  ArrayResizer<ValueT_, AllocatorT_> resizer_;
};

/**
 * @brief Dynamically-resizable 2D array container.
 *
 * Array2D stores a 2D grid of elements in row-major order and supports growing the number of rows (height) using the
 * configured `ArrayResizer`. The container exposes row/column size queries and methods similar to the 1D `Array`.
 *
 * @tparam _T Element type stored in the 2D array.
 * @tparam _AllocatorT Allocator type used by the underlying `AllocatedMemory`.
 */
template <typename _T, class _AllocatorT = DynamicAllocator<_T>>
class Array2D : public BoundedArray2D<_T, AllocatedMemory<_T, _AllocatorT>> {
 public:
  /// @brief Base class type
  using BaseClassT_ = BoundedArray2D<_T, AllocatedMemory<_T, _AllocatorT>>;

  /// @brief Value type stored in the array
  using typename BaseClassT_::ValueT_;

  /// @brief Allocator type used for dynamic allocations
  using AllocatorT_ = _AllocatorT;

  static_assert( std::is_same<typename AllocatorT_::ValueT_, ValueT_>::value,
                 "Allocator must be used with same type as Array" );

  /// @brief Default initial height capacity
  constexpr static SizeT default_height_capacity_ = 0;

  /**
   * @brief Construct an Array2D with initial dimensions and optional height capacity.
   *
   * @param height Number of initial rows.
   * @param width Number of columns per row.
   * @param height_capacity Maximum number of rows (capacity); if less than `height`,
   *                        capacity will be set to `height`.
   */
  TRIBOL_HOST_DEVICE Array2D( SizeT height, SizeT width, SizeT height_capacity = default_height_capacity_ )
      : BaseClassT_( height, width, height_capacity > height ? height_capacity : height ), resizer_( width )
  {
  }

  /**
   * @brief Default-constructs an empty Array2D with a width of 1.
   */
  TRIBOL_HOST_DEVICE Array2D() : Array2D( 0, 1, 0 ) {}

  /**
   * @brief Constructor that forwards additional arguments to the underlying memory implementation (e.g., allocator
   * objects or IDs).
   *
   * @param height Number of initial rows.
   * @param width Number of columns per row.
   * @param height_capacity Maximum number of rows (capacity); if less than `height`,
   *                        capacity will be set to `height`.
   */
  template <typename... _ArgsT>
  TRIBOL_HOST_DEVICE Array2D( SizeT height, SizeT width, SizeT height_capacity, _ArgsT&&... args )
      : BaseClassT_( height, width, height_capacity > height ? height_capacity : height,
                     std::forward<_ArgsT>( args )... ),
        resizer_( width )
  {
  }

  /**
   * @brief Copy-construct using a custom allocator.
   *
   * This creates a copy of `other` but allocates storage with the provided allocator instance.
   *
   * @tparam _Allocator2T Allocator type of the source array.
   * @param other Source array to copy from.
   * @param allocator Allocator instance to use for the new array.
   */
  template <typename _Allocator2T>
  TRIBOL_HOST_DEVICE Array2D( const Array2D<ValueT_, _Allocator2T>& other, _Allocator2T&& allocator )
      : BaseClassT_( other,
                     AllocatedMemory<ValueT_, _Allocator2T>( other.size(), other.capacity(), std::move( allocator ) ) )
  {
  }

  /// @brief Using declaration to access base class capacity.
  using BaseClassT_::capacity;

  /// @brief Using declaration to access base class size.
  using BaseClassT_::size;

  /// @brief Using declaration to access base class width.
  using BaseClassT_::width;

  /**
   * @brief Append a row to the 2D array. The initializer list must have exactly `width()` elements.
   *
   * If capacity in rows is insufficient, the resizer is used to grow the backing storage before appending the row.
   *
   * @param values Values for the new row.
   */
  TRIBOL_HOST_DEVICE void push_back( std::initializer_list<ValueT_> values )
  {
    if ( resizer_.resizeNeeded( size() + width(), memory_ ) ) {
      memory_ = resizer_.resize( memory_ );
      max_height_ = capacity() / width();
    }
    BaseClassT_::push_back( values );
  }

 private:
  /// @brief Using declaration to access maximum height (capacity in rows) from the base class.
  using BaseClassT_::max_height_;

  /// @brief Using declaration to access the protected memory_ member from the base class.
  using BaseClassT_::memory_;

  /// @brief Array resizer used to manage dynamic resizing in the row dimension.
  ArrayResizer<ValueT_, AllocatorT_> resizer_;
};

/**
 * @brief Convenience alias for an Array allocated in a particular memory space.
 *
 * When Umpire is enabled this maps to an `Array` using the `UmpireAllocator`, otherwise it resolves to an `Array` using
 * the standard `Allocator`.
 *
 * @tparam _T Element type.
 * @tparam _Mem MemorySpace enumerator specifying the target memory space.
 */
template <typename _T, MemorySpace _Mem>
using HostArray = Array<_T,
#ifdef TRIBOL_USE_UMPIRE
                        UmpireAllocator<_T, _Mem>
#else
                        Allocator<_T>
#endif
                        >;

/**
 * @brief Helper type that manages synchronized host and (optionally) managed/backing arrays (e.g., device-accessible
 * memory).
 *
 * ManagedArray holds two Array instances: a host-visible array and a managed array that may live in a different
 * allocator/memory space. It provides convenient accessors for reading and writing on either side and performs lazy
 * synchronization between the two when necessary. This is useful when working with heterogeneous memory (host/device)
 * and a managed allocator ID.
 *
 * @tparam _T Element type.
 * @tparam _ArrayT Template template parameter selecting the concrete Array type to use (defaults to `Array`).
 */
template <typename _T, template <typename, typename> class _ArrayT = Array>
class ManagedArray {
 public:
  /// @brief Type alias for the host and managed array types.
  using ArrayT_ = _ArrayT<_T, DynamicAllocator<_T>>;

  /// @brief Type alias for the managed array memory view type.
  using MemoryViewT_ = Memory<typename ArrayT_::MemoryT_::AccessorT_>;

  /// @brief Type alias for the managed array view type.
  using ViewT_ = ArrayBase<MemoryViewT_>;

  /// @brief Value type stored in the arrays.
  using ValueT_ = _T;

  /**
   * @brief Construct a ManagedArray from a host array and a managed allocator id.
   *
   * If the host array already uses the requested managed allocator id then the managed view aliases the host array;
   * otherwise a local managed copy is maintained and synchronized on demand.
   *
   * @param host_array Host-side array (moved into the ManagedArray).
   * @param managed_allocator_id Allocator id to use for the managed/backing array.
   */
  ManagedArray( ArrayT_&& host_array, int managed_allocator_id )
      : host_array_( std::move( host_array ) ),
        same_array_( host_array_.memory().allocator().id() == managed_allocator_id ),
        local_managed_array_( host_array_, DynamicAllocator<_T>( managed_allocator_id ) ),
        managed_array_( same_array_ ? host_array_ : local_managed_array_ ),
        host_array_synced_( true ),
        managed_array_synced_( true )
  {
  }

  /**
   * @brief Ensure the host-side array is up-to-date and return a const reference.
   *
   * If the managed array has more recent modifications this will copy data back to
   * the host array before returning.
   *
   * @return Const reference to the host array.
   */
  const ArrayT_& hostRead() const
  {
    if ( !same_array_ && !host_array_synced_ ) {
      host_array_ = ArrayT_( local_managed_array_, DynamicAllocator<_T>( host_array_.memory().allocator().id() ) );
      host_array_synced_ = true;
      managed_array_synced_ = true;
    }
    return host_array_;
  }

  /**
   * @brief Obtain a non-const host array reference for writing.
   *
   * If requested, this will first synchronize from the managed array unless `skip_sync` is true. After calling this,
   * the host array is considered the authoritative source until a subsequent managed write occurs. Set `skip_sync` to
   * true if you intend to overwrite all elements in the host array.
   *
   * @param skip_sync If true, skip synchronization from the managed array.
   * @return Reference to the host array for writing.
   */
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

  /**
   * @brief Obtain a host array reference for read/write access.
   *
   * Ensures the host array is synchronized from the managed array if needed and marks the managed array as out-of-date.
   *
   * @return Reference to the host array for read/write access.
   */
  ArrayT_& hostReadWrite() { return hostWrite( false ); }

  /**
   * @brief Return a const view to the managed/backing array, synchronizing from the host if necessary.
   *
   * @return Const view to the managed array.
   */
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

  /**
   * @brief Return a writable view to the managed array.
   *
   * If `skip_sync` is false and the managed array is out-of-date, this will copy from the host array into the managed
   * storage before returning the view. Set `skip_sync` to true if you intend to overwrite all elements in the managed
   * array.
   *
   * @return Writable view to the managed array.
   */
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

  /**
   * @brief Obtain a managed array view for read/write access and ensure it is synchronized from the host if necessary.
   *
   * @return Writable view to the managed array.
   */
  ViewT_ managedReadWrite() { return managedWrite( false ); }

  /**
   * @brief Query whether the host and managed arrays are actually the same object.
   *
   * @return true if both arrays share the same underlying allocator id and no copies are maintained.
   */
  bool sameArray() const { return same_array_; }

 private:
  /// @brief Host-visible array.
  mutable ArrayT_ host_array_;

  /// @brief Flag indicating whether host and managed arrays are the same object.
  bool same_array_;

  /// @brief Local managed array used for synchronization when host and managed arrays differ.
  mutable ArrayT_ local_managed_array_;

  /// @brief Reference to the managed array (either aliases host_array_ or local_managed_array_).
  ArrayT_& managed_array_;

  /// @brief Flag indicating whether the host array is synchronized with the managed array.
  mutable bool host_array_synced_;

  /// @brief Flag indicating whether the managed array is synchronized with the host array.
  mutable bool managed_array_synced_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_COMMON_ARRAYS_HPP_ */
