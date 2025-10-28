// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_MEMORY_HPP_
#define SRC_TRIBOL_COMMON_MEMORY_HPP_

// Tribol config include
#include "tribol/config.hpp"

// C includes
#include <cassert>
#include <cstddef>

// Umpire includes
#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/TypedAllocator.hpp"
#endif

// Tribol includes
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/ExecModel.hpp"

namespace tribol {

/**
 * @brief Fixed-capacity container policy parameterized by a compile-time size.
 *
 * This class encodes a capacity that is fixed at compile time, sized via the template parameter _N. It is intended to
 * be used as a policy or helper type to represent containers or buffers whose capacity is known statically.
 *
 * @tparam _N The compile-time capacity (number of elements).
 */
template <SizeT _N>
class FixedCapacity {
 public:
  /**
   * @brief Construct a FixedCapacity instance.
   *
   * The constructor accepts a runtime capacity parameter for API uniformity but asserts that the provided value matches
   * the compile-time capacity _N.
   *
   * @param capacity The runtime capacity value that must equal _N (asserted).
   */
  TRIBOL_HOST_DEVICE FixedCapacity( [[maybe_unused]] SizeT capacity ) { assert( capacity == _N ); }

  /**
   * @brief Return the compile-time capacity.
   *
   * This function returns the capacity encoded by the template parameter _N.
   *
   * @return The compile-time capacity (_N).
   */
  TRIBOL_HOST_DEVICE constexpr SizeT capacity() const { return _N; }

  /**
   * @brief Set capacity at runtime (no-op for fixed-capacity type).
   *
   * For this fixed-capacity policy the capacity cannot be changed at runtime. This function signature exists for
   * interface compatibility and always returns the compile-time capacity _N while ignoring the provided argument.
   *
   * @param new_capacity Runtime capacity value (ignored).
   * @return The compile-time capacity (_N).
   */
  TRIBOL_HOST_DEVICE constexpr SizeT setCapacity( SizeT ) const { return _N; }

  /**
   * @brief Type indicating the capacity is fixed.
   *
   * For this type, capacity is known at compile time and cannot be changed at runtime, so this alias is
   * std::false_type. This can be used in template metaprogramming to select different code paths when capacity is fixed
   * vs. dynamic.
   */
  using IsCapacityFixedT_ = std::true_type;
};

/**
 * @brief Encapsulates a capacity value that is determined and modifiable at runtime.
 *
 * This class provides a small, type-safe wrapper around a size-type capacity so that capacity semantics are explicit in
 * APIs. It is intended for use where the capacity cannot be known at compile time but must be stored, queried, and
 * updated during program execution. The type is lightweight and intended to be usable in host/device code paths
 * (indicated by the TRIBOL_HOST_DEVICE annotation on member functions).
 */
class RuntimeCapacity {
 public:
  /**
   * @brief Construct a RuntimeCapacity with the given initial capacity.
   *
   * @param capacity The initial capacity value to store.
   *
   * @note This constructor is annotated for host/device usage (TRIBOL_HOST_DEVICE) so it can be called from both host
   * and device code when compiled for CUDA/HIP.
   */
  TRIBOL_HOST_DEVICE RuntimeCapacity( SizeT capacity ) : capacity_( capacity ) {}

  /**
   * @brief Return the currently stored capacity.
   *
   * This accessor does not modify the object and returns the raw capacity value.
   *
   * @return The current capacity.
   *
   * @note Annotated as TRIBOL_HOST_DEVICE to allow calls from host and device code.
   */
  TRIBOL_HOST_DEVICE SizeT capacity() const { return capacity_; }

  /**
   * @brief Set the capacity to a new value and return that value.
   *
   * This mutator updates the stored capacity to the provided value and returns the value that was set. It is useful for
   * fluent-style updates or when the caller needs confirmation of the stored value.
   *
   * @param capacity The new capacity value to store.
   * @return The capacity value that was stored (same as the parameter).
   *
   * @note Annotated as TRIBOL_HOST_DEVICE so it can be invoked from both host and device code. No thread-safety or
   * synchronization is provided by this type; callers must ensure correct concurrent access semantics if used in
   * parallel code.
   */
  TRIBOL_HOST_DEVICE SizeT setCapacity( SizeT capacity )
  {
    capacity_ = capacity;
    return capacity;
  }

  /**
   * @brief Type indicating that capacity can be changed at runtime.
   *
   * Presence of this alias (std::true_type) can be used in metaprogramming to detect that the capacity for an object is
   * not a compile-time constant but instead is provided/modified at runtime.
   */
  using IsCapacityFixedT_ = std::false_type;

 private:
  /**
   * @brief Underlying storage for the capacity value.
   *
   * Holds the numeric capacity. This member is private and should be accessed via the public accessor and mutator
   * methods.
   */
  SizeT capacity_;
};

/**
 * @brief An array size policy class where the size is always equal to the capacity.
 *
 * @tparam _Capacity The base class that provides the capacity management. It is expected to have a constructor taking a
 * `SizeT`, `capacity()` and `setCapacity()` methods, and a IsCapacityFixedT_ type.
 *
 * This class inherits from a given Capacity class and enforces the invariant that the logical size of the object is
 * always the same as its allocated capacity. It is useful for representing data structures that are always full or have
 * a fixed size determined at construction.
 */
template <typename _CapacityT>
class SizeEqCapacity : public _CapacityT {
 public:
  /// @brief Alias for the underlying Capacity type.
  using CapacityT_ = _CapacityT;

  /**
   * @brief Construct a SizeEqCapacity with an explicit size and capacity.
   *
   * @param size The logical size to use (also passed to the base Capacity constructor).
   * @param capacity The capacity value to verify. An assertion checks size == capacity.
   */
  TRIBOL_HOST_DEVICE SizeEqCapacity( SizeT size, [[maybe_unused]] SizeT capacity ) : CapacityT_( size )
  {
    assert( size == capacity );
  }

  /**
   * @brief Construct a SizeEqCapacity with a single size value.
   *
   * @param size The logical size (and implicit capacity) to use.
   */
  TRIBOL_HOST_DEVICE SizeEqCapacity( SizeT size ) : CapacityT_( size ) {}

  /**
   * @brief Return the current logical size.
   *
   * This returns the underlying capacity value; the two are guaranteed equal.
   *
   * @return The current size (equal to capacity()).
   */
  TRIBOL_HOST_DEVICE constexpr SizeT size() const { return capacity(); }

  /**
   * @brief Return the current capacity.
   *
   * Inherited from Capacity; exposed via a using-declaration.
   */
  using CapacityT_::capacity;

  /**
   * @brief Set the logical size.
   *
   * Implemented by forwarding to CapacityT_::setCapacity(size). Because this policy enforces size == capacity, setting
   * the size is equivalent to setting capacity.
   *
   * @param size The new size (and capacity) to set.
   * @return The resulting capacity after the operation (value returned by setCapacity).
   */
  TRIBOL_HOST_DEVICE SizeT setSize( SizeT size ) { return setCapacity( size ); }

  /**
   * @brief Set the capacity.
   *
   * Inherited from Capacity; exposed via a using-declaration.
   */
  using CapacityT_::setCapacity;

  /**
   * @brief Query whether size is at capacity.
   *
   * Always returns true for this policy since size() == capacity() by design.
   *
   * @return true
   */
  TRIBOL_HOST_DEVICE constexpr bool sizeAtCapacity() const { return true; }

  /// @brief Type indicating that the policy represents a container size that matches container capacity.
  using IsSizeEqCapacityT_ = std::true_type;
};

/**
 * @brief An array size policy class where the size can be less than or equal to the capacity.
 *
 * @tparam _Capacity The base class that provides the capacity management. It is expected to have a constructor taking a
 * `SizeT`, `capacity()` and `setCapacity()` methods, and a IsCapacityFixedT_ type.
 *
 * This class inherits from a given Capacity class and allows the logical size of the object to be less than or equal to
 * its allocated capacity. It is useful for representing data structures that may not always be full.
 */
template <typename _CapacityT>
class SizeLECapacity : public _CapacityT {
 public:
  /// @brief Alias for the underlying Capacity type.
  using CapacityT_ = _CapacityT;

  /**
   * @brief Construct a SizeLECapacity with an explicit size and capacity.
   *
   * @param size The logical size to use (also passed to the base Capacity constructor).
   * @param capacity The capacity value to verify. An assertion checks size <= capacity.
   */
  TRIBOL_HOST_DEVICE SizeLECapacity( SizeT size, SizeT capacity )
      : CapacityT_( capacity >= size ? capacity : size ), size_( size )
  {
    assert( size <= capacity );
  }

  /**
   * @brief Return the current logical size.
   *
   * This returns the underlying size value.
   *
   * @return The current size.
   */
  TRIBOL_HOST_DEVICE SizeT size() const { return size_; }

  /**
   * @brief Return the current capacity.
   *
   * Inherited from Capacity; exposed via a using-declaration.
   */
  using CapacityT_::capacity;

  /**
   * @brief Set the logical size.
   *
   * @param size The new size to set (must be <= capacity).
   * @return The resulting size after the operation.
   */
  TRIBOL_HOST_DEVICE SizeT setSize( SizeT size )
  {
    assert( size <= capacity() );
    size_ = size <= capacity() ? size : capacity();
    return size_;
  }

  /**
   * @brief Query whether size is at capacity.
   *
   * @return true if size() == capacity(), false otherwise.
   */
  TRIBOL_HOST_DEVICE bool sizeAtCapacity() const { return size() >= capacity(); }

  /// @brief Type indicating that the policy represents independent size and capacity values.
  using IsSizeEqCapacityT_ = std::false_type;

 private:
  /**
   * @brief Underlying storage for the logical size.
   *
   * Holds the numeric size. This member is private and should be accessed via the public accessor and mutator methods.
   */
  SizeT size_;
};

/**
 * @brief Contiguous memory accessor with unit stride.
 *
 * @tparam _T The type of elements stored in the memory.
 * @tparam _SizeAndCapacityT The policy class that provides size and capacity management. The policy class is expected
 * to have `CapacityT_`, `IsSizeEqCapacityT_`, and `IsCapacityFixedT_` type definitions; `size()`, `capacity()`,
 * `setSize()`, `setCapacity()`, and `sizeAtCapacity()` member functions; and a constructor taking size and capacity
 * parameters.
 *
 * This class provides random access to a contiguous block of memory with unit stride.
 * It inherits from a SizeAndCapacity policy to manage size and capacity semantics.
 */
template <typename _T, class _SizeAndCapacityT>
class ContiguousMemory : public _SizeAndCapacityT {
 public:
  /// @brief Alias for the underlying SizeAndCapacity type.
  using SizeAndCapacityT_ = _SizeAndCapacityT;

  /// @brief The type of elements stored in the memory.
  using ValueT_ = _T;

  /// @brief Pointer type for accessing elements.
  using PointerT_ = _T*;

  /// @brief Const pointer type for accessing elements.
  using ConstPointerT_ = const _T*;

  /**
   * @brief Construct a ContiguousMemory accessor.
   *
   * @param data Pointer to the beginning of the memory block.
   * @param size The number of elements in the memory block.
   * @param capacity The total capacity of the memory block.
   */
  TRIBOL_HOST_DEVICE ContiguousMemory( PointerT_ data, SizeT size, SizeT capacity )
      : SizeAndCapacityT_( size, capacity ), data_( data )
  {
  }

  /**
   * @brief Construct a ContiguousMemory accessor with specified stride (must be 1).
   *
   * @param data Pointer to the beginning of the memory block.
   * @param size The number of elements in the memory block.
   * @param capacity The total capacity of the memory block.
   * @param stride The stride between elements (must be 1).
   */
  TRIBOL_HOST_DEVICE ContiguousMemory( PointerT_ data, SizeT size, SizeT capacity, [[maybe_unused]] SizeT stride )
      : SizeAndCapacityT_( size, capacity ), data_( data )
  {
    assert( stride == 1 );
  }

  /**
   * @brief Access an element at a specific index.
   *
   * @param i The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  TRIBOL_HOST_DEVICE ValueT_& at( SizeT i ) { return *( data_ + i ); }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueT_& at( SizeT i ) const { return *( data_ + i ); }

  /**
   * @brief Access an element at a specific index using the subscript operator.
   *
   * @param i The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  TRIBOL_HOST_DEVICE ValueT_& operator[]( SizeT i ) { return at( i ); }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueT_& operator[]( SizeT i ) const { return at( i ); }

  /// @brief Iterator type for the memory block.
  using IteratorT_ = PointerT_;

  /// @brief Const iterator type for the memory block.
  using ConstIteratorT_ = ConstPointerT_;

  /**
   * @brief Get an iterator to the beginning of the memory block.
   *
   * @return An iterator to the first element.
   */
  TRIBOL_HOST_DEVICE IteratorT_ begin() { return data_; }
  /// @overload
  TRIBOL_HOST_DEVICE ConstIteratorT_ begin() const { return data_; }

  /**
   * @brief Get an iterator to the end of the memory block.
   *
   * @return An iterator to one past the last element.
   */
  TRIBOL_HOST_DEVICE IteratorT_ end() { return data_ + size(); }
  /// @overload
  TRIBOL_HOST_DEVICE ConstIteratorT_ end() const { return data_ + size(); }

  /// @brief Get the size of the memory block.
  using SizeAndCapacityT_::size;

  /**
   * @brief Get the stride between elements.
   *
   * @return The stride (always 1 for this class).
   */
  TRIBOL_HOST_DEVICE constexpr SizeT stride() const { return 1; }

  /**
   * @brief Get a pointer to the underlying data.
   *
   * @return A pointer to the beginning of the memory block.
   */
  TRIBOL_HOST_DEVICE PointerT_ data() const { return data_; }

  /**
   * @brief Implicit conversion to a pointer.
   *
   * @return A pointer to the beginning of the memory block.
   */
  TRIBOL_HOST_DEVICE operator PointerT_() const { return data_; }

 protected:
  /// @brief Pointer to the beginning of the memory block.
  PointerT_ data_;
};

/**
 * @brief Memory accessor with a fixed stride.
 *
 * @tparam _T The type of elements stored in the memory.
 * @tparam _SizeAndCapacityT The policy class that provides size and capacity management. The policy class is expected
 * to have `CapacityT_`, `IsSizeEqCapacityT_`, and `IsCapacityFixedT_` type definitions; `size()`, `capacity()`,
 * `setSize()`, `setCapacity()`, and `sizeAtCapacity()` member functions; and a constructor taking size and capacity
 * parameters.
 *
 * This class provides random access to a block of memory with a fixed stride. It inherits from a SizeAndCapacity policy
 * to manage size and capacity semantics.
 */
template <typename _T, class _SizeAndCapacityT>
class FixedStride : public _SizeAndCapacityT {
 public:
  /// @brief Alias for the underlying SizeAndCapacity type.
  using SizeAndCapacityT_ = _SizeAndCapacityT;

  /// @brief Alias for the underlying Capacity type.
  using typename SizeAndCapacityT_::CapacityT_;

  /// @brief The type of elements stored in the memory.
  using ValueT_ = _T;

  /// @brief Pointer type for accessing elements.
  using PointerT_ = _T*;

  /// @brief Const pointer type for accessing elements.
  using ConstPointerT_ = const _T*;

  /**
   * @brief Construct a FixedStride accessor.
   *
   * @param data Pointer to the beginning of the memory block.
   * @param size The number of elements in the memory block.
   * @param capacity The total capacity of the memory block.
   * @param stride The stride between elements.
   */
  TRIBOL_HOST_DEVICE FixedStride( PointerT_ data, SizeT size, SizeT capacity, SizeT stride )
      : SizeAndCapacityT_( size, capacity ), data_( data ), stride_( stride )
  {
    assert( stride > 0 );
  }

  /**
   * @brief Base class for iterators.
   *
   * @tparam _PointerT The pointer type for the iterator.
   */
  template <typename _PointerT>
  struct IteratorBase {
    /// @brief Pointer type for the iterator.
    using PointerT_ = _PointerT;

    /**
     * @brief Construct an IteratorBase.
     *
     * @param ptr Pointer to the current element.
     * @param stride The stride between elements.
     */
    TRIBOL_HOST_DEVICE IteratorBase( PointerT_ ptr, SizeT stride ) : ptr_( ptr ), stride_( stride ) {}

    /**
     * @brief Pre-increment operator.
     *
     * @return A reference to the incremented iterator.
     */
    TRIBOL_HOST_DEVICE IteratorBase& operator++()
    {
      ptr_ += stride_;
      return *this;
    }

    /**
     * @brief Pre-decrement operator.
     *
     * @return A reference to the decremented iterator.
     */
    TRIBOL_HOST_DEVICE IteratorBase& operator--()
    {
      ptr_ -= stride_;
      return *this;
    }

    /**
     * @brief Post-increment operator.
     *
     * @return A copy of the iterator before incrementing.
     */
    TRIBOL_HOST_DEVICE IteratorBase operator++( int )
    {
      IteratorBase tmp = *this;
      ++( *this );
      return tmp;
    }

    /**
     * @brief Post-decrement operator.
     *
     * @return A copy of the iterator before decrementing.
     */
    TRIBOL_HOST_DEVICE IteratorBase operator--( int )
    {
      IteratorBase tmp = *this;
      --( *this );
      return tmp;
    }

    /**
     * @brief Addition operator.
     *
     * @param n The number of elements to advance.
     * @return A new iterator advanced by n elements.
     */
    TRIBOL_HOST_DEVICE IteratorBase operator+( SizeT n ) const
    {
      IteratorBase tmp = *this;
      tmp.ptr_ += n * stride_;
      return tmp;
    }

    /**
     * @brief Subtraction operator.
     *
     * @param n The number of elements to move back.
     * @return A new iterator moved back by n elements.
     */
    TRIBOL_HOST_DEVICE IteratorBase operator-( SizeT n ) const
    {
      IteratorBase tmp = *this;
      tmp.ptr_ -= n * stride_;
      return tmp;
    }

    /**
     * @brief Equality operator.
     *
     * @param other The iterator to compare against.
     * @return true if the iterators point to the same element, false otherwise.
     */
    TRIBOL_HOST_DEVICE bool operator==( const IteratorBase& other ) const { return ptr_ == other.ptr_; }

    /**
     * @brief Inequality operator.
     *
     * @param other The iterator to compare against.
     * @return true if the iterators point to different elements, false otherwise.
     */
    TRIBOL_HOST_DEVICE bool operator!=( const IteratorBase& other ) const { return !( *this == other ); }

    /**
     * @brief Dereference operator.
     *
     * @return A reference to the element pointed to by the iterator.
     */
    TRIBOL_HOST_DEVICE ValueT_& operator*() { return *ptr_; }

   private:
    /// @brief Pointer to the beginning of the memory block.
    PointerT_ ptr_;

    /// @brief The stride between elements.
    SizeT stride_;
  };

  /// @brief Iterator type for the memory block.
  using IteratorT_ = IteratorBase<PointerT_>;

  /// @brief Const iterator type for the memory block.
  using ConstIteratorT_ = IteratorBase<ConstPointerT_>;

  /**
   * @brief Access an element at a specific index.
   *
   * @param i The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  TRIBOL_HOST_DEVICE ValueT_& at( SizeT i ) { return *( data_ + i * stride_ ); }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueT_& at( SizeT i ) const { return *( data_ + i * stride_ ); }

  /**
   * @brief Access an element at a specific index using the subscript operator.
   *
   * @param i The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  TRIBOL_HOST_DEVICE ValueT_& operator[]( SizeT i ) { return at( i ); }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueT_& operator[]( SizeT i ) const { return at( i ); }

  /**
   * @brief Get an iterator to the beginning of the memory block.
   *
   * @return An iterator to the first element.
   */
  TRIBOL_HOST_DEVICE IteratorT_ begin() { return IteratorT_( data_, stride_ ); }
  /// @overload
  TRIBOL_HOST_DEVICE ConstIteratorT_ begin() const { return ConstIteratorT_( data_, stride_ ); }

  /**
   * @brief Get an iterator to the end of the memory block.
   *
   * @return An iterator to one past the last element.
   */
  TRIBOL_HOST_DEVICE IteratorT_ end() { return IteratorT_( data_ + size() * stride_, stride_ ); }
  /// @overload
  TRIBOL_HOST_DEVICE ConstIteratorT_ end() const { return ConstIteratorT_( data_ + size() * stride_, stride_ ); }

  /// @brief Get the size of the memory block.
  using SizeAndCapacityT_::size;

  /**
   * @brief Get the stride between elements.
   *
   * @return The stride.
   */
  TRIBOL_HOST_DEVICE SizeT stride() const { return stride_; }

  /**
   * @brief Get a pointer to the underlying data.
   *
   * @return A pointer to the beginning of the memory block.
   */
  TRIBOL_HOST_DEVICE PointerT_ data() const { return data_; }

  /**
   * @brief Implicit conversion to a pointer.
   *
   * @return A pointer to the beginning of the memory block.
   */
  TRIBOL_HOST_DEVICE operator PointerT_() const { return data_; }

 protected:
  /// @brief Pointer to the beginning of the memory block.
  PointerT_ data_;

  /// @brief The stride between elements.
  SizeT stride_;
};

/**
 * @brief Base class for a memory block with arbitrary stride and size/capacity management. No data ownership so it can
 * be used as a view.
 *
 * @tparam _AccessorT The accessor policy that provides data access and size/capacity management. The accessor is
 * expected to have `PointerT_`, and `ValueT_` type definitions, as well as appropriate constructors and
 * data access methods.
 */
template <class _AccessorT>
class Memory : public _AccessorT {
 public:
  /// @brief Alias for the underlying _AccessorT template type.
  using AccessorT_ = _AccessorT;

  /// @brief Pointer type for the memory block.
  using typename AccessorT_::PointerT_;

  /// @brief The type of elements stored in the memory.
  using typename AccessorT_::ValueT_;

  /// @brief Alias for the view type.
  using ViewT_ = Memory<AccessorT_>;

  /**
   * @brief Construct a Memory view.
   *
   * @param data Pointer to the beginning of the memory block.
   * @param size The number of elements in the memory block.
   * @param capacity The total capacity of the memory block.
   * @param stride The stride between elements.
   */
  TRIBOL_HOST_DEVICE Memory( PointerT_ data, SizeT size, SizeT capacity, SizeT stride )
      : AccessorT_( data, size, capacity, stride )
  {
  }

  /**
   * @brief Construct a Memory view with size equal to capacity.
   *
   * @param data Pointer to the beginning of the memory block.
   * @param size The number of elements in the memory block.
   * @param stride The stride between elements.
   */
  TRIBOL_HOST_DEVICE Memory( PointerT_ data, SizeT size, SizeT stride = 1 ) : Memory( data, size, size, stride ) {}

  /**
   * @brief Copy constructor.
   *
   * @param other The Memory view to copy.
   *
   * The copy constructor is explicitly defined so the move constructor is not implicitly generated. Moving should
   * behave the same as copying for this view class.
   */
  TRIBOL_HOST_DEVICE Memory( const Memory& other ) = default;

  /**
   * @brief Copy assignment operator.
   *
   * @param other The Memory view to copy.
   * @return A reference to this Memory view.
   */
  TRIBOL_HOST_DEVICE Memory& operator=( const Memory& other ) = default;

  /**
   * @brief Access an element at a specific index with bounds checking.
   *
   * @param i The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  TRIBOL_HOST_DEVICE ValueT_& at( SizeT i )
  {
    assert( i < size() );
    return AccessorT_::at( i );
  }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueT_& at( SizeT i ) const
  {
    assert( i < size() );
    return AccessorT_::at( i );
  }

  /**
   * @brief Access an element at a specific index using the subscript operator with bounds checking.
   *
   * @param i The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  TRIBOL_HOST_DEVICE ValueT_& operator[]( SizeT i )
  {
    assert( i < size() );
    return AccessorT_::operator[]( i );
  }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueT_& operator[]( SizeT i ) const
  {
    assert( i < size() );
    return AccessorT_::operator[]( i );
  }

  /**
   * @brief Create a new view of a sub-region of the memory.
   *
   * @tparam _OtherAccessorT The accessor policy for the new view.
   * @param offset The starting offset of the sub-region.
   * @param size The number of elements in the sub-region.
   * @param capacity The total capacity of the sub-region.
   * @param stride The stride between elements in the sub-region.
   * @return A new Memory view of the sub-region.
   */
  template <typename _OtherAccessorT>
  TRIBOL_HOST_DEVICE Memory<_OtherAccessorT> view( SizeT offset, SizeT size, SizeT capacity, SizeT stride = 1 ) const
  {
    assert( offset + size * stride <= this->size() );
    return Memory<_OtherAccessorT>( data() + offset, size, capacity, stride );
  }

  /**
   * @brief Create a new view of the entire memory block.
   *
   * @return A new Memory view of the entire memory block.
   */
  TRIBOL_HOST_DEVICE Memory<AccessorT_> view() { return Memory<AccessorT_>( *this ); }
  /// @overload
  TRIBOL_HOST_DEVICE Memory<AccessorT_> view() const { return Memory<AccessorT_>( *this ); }

  /// @brief Get the size of the memory block.
  using AccessorT_::size;

  /// @brief Get a pointer to the underlying data.
  using AccessorT_::data;

  /**
   * @brief Type indicating that the memory is initialized.
   *
   * The memory is assumed to be a view that is already initialized.
   */
  using IsInitializedT_ = std::true_type;
};

/**
 * @brief A fixed-size memory block allocated on the stack.
 *
 * @tparam _T The type of elements stored in the memory.
 * @tparam _N The compile-time size of the memory block.
 * @tparam _SizeVsCapacityT The policy class that manages size vs. capacity. The policy class is expected to have
 * `CapacityT_`, `IsSizeEqCapacityT_`, and `IsCapacityFixedT_` type definitions; `size()`, `capacity()`, `setSize()`,
 * `setCapacity()`, and `sizeAtCapacity()` member functions; and a constructor taking size and capacity parameters.
 *
 * This class provides a fixed-size memory block allocated on the stack, with an interface similar to a standard
 * container.
 */
template <typename _T, SizeT _N, template <typename> class _SizeVsCapacityT = SizeEqCapacity>
class StackMemory : public Memory<ContiguousMemory<_T, _SizeVsCapacityT<FixedCapacity<_N>>>> {
 public:
  /// @brief Base class for the StackMemory.
  using BaseClassT_ = Memory<ContiguousMemory<_T, _SizeVsCapacityT<FixedCapacity<_N>>>>;

  /// @brief The type of elements stored in the memory.
  using typename BaseClassT_::ValueT_;

  /**
   * @brief Construct a StackMemory object.
   *
   * @param size The initial size of the memory block.
   */
  TRIBOL_HOST_DEVICE StackMemory( SizeT size = _N ) : BaseClassT_( nullptr, size, _N, 1 ) { data_ = stack_data_; }

  /**
   * @brief Construct a StackMemory object with explicit size and capacity.
   *
   * @param size The initial size of the memory block.
   * @param capacity The capacity of the memory block (must be N).
   */
  TRIBOL_HOST_DEVICE StackMemory( SizeT size, [[maybe_unused]] SizeT capacity ) : StackMemory( size )
  {
    assert( capacity == _N );
  }

  /**
   * @brief Copy constructor (deep copy).
   *
   * @param other The StackMemory object to copy.
   *
   * The move constructor is not implicitly generated since the copy constructor is explicitly defined. Moving should
   * behave the same as copying for this class.
   */
  TRIBOL_HOST_DEVICE StackMemory( const StackMemory& other ) : StackMemory( other.size() )
  {
    assert( other.capacity() == _N );
    if ( this != &other ) {
      for ( SizeT i = 0; i < other.size(); ++i ) {
        stack_data_[i] = other.stack_data_[i];
      }
    }
  }

  /**
   * @brief Copy assignment operator (deep copy).
   *
   * @param other The StackMemory object to copy.
   * @return A reference to this StackMemory object.
   *
   * The move assignment operator is not implicitly generated since the copy constructor is explicitly defined. Moving
   * should behave the same as copying for this class.
   */
  TRIBOL_HOST_DEVICE StackMemory& operator=( const StackMemory& other )
  {
    assert( other.capacity() == _N );
    if ( this != &other ) {
      BaseClassT_::operator=( other );
      for ( SizeT i = 0; i < other.size(); ++i ) {
        stack_data_[i] = other.stack_data_[i];
      }
    }
    return *this;
  }

  /// @brief Destructor.
  TRIBOL_DEFAULT_HOST_DEVICE ~StackMemory() = default;

  /// @brief Type indicator that the memory is not initialized.
  using IsInitializedT_ = std::false_type;

 private:
  /// @brief Using declaration to access the protected data_ member from the base class.
  using BaseClassT_::data_;

  /// @brief Underlying stack-allocated data storage.
  ValueT_ stack_data_[_N];
};

/**
 * @brief A simple allocator for standard memory management.
 *
 * @tparam _T The type of elements to allocate.
 */
template <typename _T>
class Allocator {
  static_assert( !std::is_const<_T>::value, "Allocator does not support const types" );
  static_assert( !std::is_volatile<_T>::value, "Allocator does not support volatile types" );

 public:
  /// @brief The type of elements to allocate.
  using ValueT_ = _T;

  /// @brief Pointer type for the allocated memory.
  using PointerT_ = _T*;

  /// @brief The signed/integral type used to represent differences between pointers.
  using DifferenceT_ = ptrdiff_t;

  /**
   * @brief Allocate a block of memory.
   *
   * @param n The number of elements to allocate.
   * @return A pointer to the allocated memory.
   */
  TRIBOL_HOST_DEVICE PointerT_ allocate( SizeT n ) const
  {
    return static_cast<PointerT_>( ::operator new( n * sizeof( ValueT_ ) ) );
  }

  /**
   * @brief Deallocate a block of memory.
   *
   * @param p A pointer to the memory to deallocate.
   * @param n The number of elements that were allocated.
   */
  TRIBOL_HOST_DEVICE void deallocate( PointerT_ p, SizeT ) const { ::operator delete( p ); }

  /**
   * @brief Copy uninitialized memory.
   *
   * @param first A pointer to the beginning of the source range.
   * @param last A pointer to the end of the source range.
   * @param d_first A pointer to the beginning of the destination range.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE void uninitialized_copy( PointerT_ first, PointerT_ last, PointerT_ d_first ) const
  {
    std::uninitialized_copy( first, last, d_first );
  }
};

/**
 * @brief Equality operator for Allocator.
 *
 * @tparam _T The type of the first allocator.
 * @tparam _U The type of the second allocator.
 * @return true, since all instances of Allocator are equivalent.
 */
template <typename _T, typename _U>
TRIBOL_HOST_DEVICE inline constexpr bool operator==( const Allocator<_T>&, const Allocator<_U>& )
{
  return true;
}

#ifdef TRIBOL_USE_UMPIRE
/**
 * @brief An allocator that uses a typed Umpire allocator for memory management.
 *
 * @tparam _T The type of elements to allocate.
 * @tparam _Mem The memory space to allocate in.
 */
template <typename _T, MemorySpace _Mem>
class UmpireAllocator {
 public:
  /// @brief The type of elements to allocate.
  using ValueT_ = _T;

  /// @brief Pointer type for the allocated memory.
  using PointerT_ = ValueT_*;

  /// @brief The signed/integral type used to represent differences between pointers.
  using DifferenceT_ = ptrdiff_t;

  /**
   * @brief Construct an UmpireAllocator.
   *
   * @param allocator The Umpire allocator to use.
   */
  UmpireAllocator( umpire::Allocator allocator ) : allocator_{ std::move( allocator ) } {}

  /// @brief Construct an UmpireAllocator for a specific memory space.
  UmpireAllocator()
      : UmpireAllocator( umpire::ResourceManager::getInstance().getAllocator( getResourceAllocatorID( _Mem ) ) )
  {
  }

  /**
   * @brief Allocate a block of memory.
   *
   * @param n The number of elements to allocate.
   * @return A pointer to the allocated memory.
   */
  PointerT_ allocate( SizeT n ) const { return static_cast<PointerT_>( allocator_.allocate( n ) ); }

  /**
   * @brief Deallocate a block of memory.
   *
   * @param p A pointer to the memory to deallocate.
   * @param n The number of elements that were allocated.
   */
  void deallocate( PointerT_ p, SizeT n ) const { allocator_.deallocate( p, n ); }

  /**
   * @brief Copy uninitialized memory.
   *
   * @param first A pointer to the beginning of the source range.
   * @param last A pointer to the end of the source range.
   * @param d_first A pointer to the beginning of the destination range.
   */
  void uninitialized_copy( PointerT_ first, PointerT_, PointerT_ d_first ) const
  {
    auto& rm = umpire::ResourceManager::getInstance();
    rm.copy( d_first, first );
  }

 private:
  /**
   * @brief The Umpire typed allocator used for memory management.
   *
   * @note Object is mutable to deal with umpire::TypedAllocator non-const member functions
   */
  mutable umpire::TypedAllocator<ValueT_> allocator_;
};
#endif

/**
 * @brief A dynamic allocator that can uses a standard allocator or an Umpire allocator, if available.
 *
 * @tparam _T The type of elements to allocate.
 */
template <typename _T>
class DynamicAllocator {
 public:
  /// @brief The type of elements to allocate.
  using ValueT_ = _T;

  /// @brief Pointer type for the allocated memory.
  using PointerT_ = ValueT_*;

  /// @brief Construct a DynamicAllocator with the default allocator ID.
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE DynamicAllocator() : allocator_id_( getDefaultAllocatorID() ) {}

  /**
   * @brief Construct a DynamicAllocator with a specific allocator ID.
   *
   * @param allocator_id The ID of the allocator to use.
   */
  TRIBOL_HOST_DEVICE DynamicAllocator( int allocator_id ) : allocator_id_( allocator_id ) {}

  /**
   * @brief Allocate a block of memory.
   *
   * @param n The number of elements to allocate.
   * @return A pointer to the allocated memory.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE PointerT_ allocate( SizeT n ) const
  {
    return static_cast<PointerT_>(
#ifdef TRIBOL_USE_UMPIRE
        umpire::ResourceManager::getInstance().getAllocator( allocator_id_ ).allocate( n * sizeof( ValueT_ ) )
#else
        Allocator<ValueT_>().allocate( n )
#endif
    );
  }

  /**
   * @brief Deallocate a block of memory.
   *
   * @param p A pointer to the memory to deallocate.
   * @param n The number of elements that were allocated.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE void deallocate( PointerT_ p, [[maybe_unused]] SizeT n ) const
  {
#ifdef TRIBOL_USE_UMPIRE
    umpire::ResourceManager::getInstance().getAllocator( allocator_id_ ).deallocate( p );
#else
    Allocator<ValueT_>().deallocate( p, n );
#endif
  }

  /**
   * @brief Copy uninitialized memory.
   *
   * @param first A pointer to the beginning of the source range.
   * @param last A pointer to the end of the source range.
   * @param d_first A pointer to the beginning of the destination range.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE void uninitialized_copy( PointerT_ first, [[maybe_unused]] PointerT_ last,
                                              PointerT_ d_first ) const
  {
#ifdef TRIBOL_USE_UMPIRE
    auto& rm = umpire::ResourceManager::getInstance();
    rm.copy( d_first, first );
#else
    Allocator<ValueT_>().uninitialized_copy( first, last, d_first );
#endif
  }

  /**
   * @brief Get the ID of the allocator.
   *
   * @return The allocator ID.
   */
  TRIBOL_HOST_DEVICE int id() const { return allocator_id_; }

 private:
  /// @brief The ID of the allocator to use.
  int allocator_id_;
};

/**
 * @brief A class that manages allocated memory.
 *
 * @tparam _T The type of elements stored in the memory.
 * @tparam _AllocatorT The allocator to use for memory management. The allocator must provide `allocate()`,
 * `deallocate()`, and `uninitialized_copy()` methods.
 * @tparam _SizeVsCapacityT The policy class that manages size vs. capacity. The policy class is expected to have
 * `CapacityT_`, `IsSizeEqCapacityT_`, and `IsCapacityFixedT_` type definitions; `size()`, `capacity()`, `setSize()`,
 * `setCapacity()`, and `sizeAtCapacity()` member functions; and a constructor taking size and capacity parameters.
 *
 * This class provides a container-like interface for a block of memory that is dynamically allocated and managed by an
 * allocator.
 */
template <typename _T, class _AllocatorT = Allocator<_T>, class _SizeVsCapacityT = SizeLECapacity<RuntimeCapacity>>
class AllocatedMemory : public Memory<ContiguousMemory<_T, _SizeVsCapacityT>> {
 public:
  /// @brief Base class for the AllocatedMemory.
  using BaseClassT_ = Memory<ContiguousMemory<_T, _SizeVsCapacityT>>;

  /// @brief Alias for the underlying Allocator type.
  using AllocatorT_ = _AllocatorT;

  /// @brief Pointer type for the memory block.
  using typename BaseClassT_::PointerT_;

  /// @brief Value type for the memory block.
  using typename BaseClassT_::ValueT_;

  static_assert( std::is_same<typename _AllocatorT::ValueT_, ValueT_>::value,
                 "AllocatedMemory must be used with same type as allocator" );

  /**
   * @brief Construct an AllocatedMemory object.
   *
   * @param size The initial size of the memory block.
   * @param capacity The capacity of the memory block.
   * @param allocator The allocator to use.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory( SizeT size, SizeT capacity, AllocatorT_ allocator = AllocatorT_() )
      : BaseClassT_( allocator.allocate( capacity ), size, capacity, 1 ), allocator_( std::move( allocator ) )
  {
  }

  /**
   * @brief Construct an AllocatedMemory object with size equal to capacity.
   *
   * @param size The initial size and capacity of the memory block.
   * @param allocator The allocator to use.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory( SizeT size, AllocatorT_ allocator = AllocatorT_() )
      : AllocatedMemory( size, size, std::move( allocator ) )
  {
  }

  /**
   * @brief Construct an AllocatedMemory object from a moved memory view.
   *
   * @param memory The memory view to move.
   * @param allocator The allocator to use.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory( BaseClassT_&& memory, AllocatorT_&& allocator = AllocatorT_() )
      : BaseClassT_( std::move( memory ) ), allocator_( std::move( allocator ) )
  {
  }

  /**
   * @brief Construct an AllocatedMemory object by copying from a source and moving from a destination.
   *
   * @param src The source object to copy from.
   * @param dst The destination object to move from.
   *
   * This constructor creates a new AllocatedMemory object by copying data from the source object using the allocator
   in
   * the destination object. Conceptually, this is similar to a copy constructor, but it allows the allocator
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory( const AllocatedMemory& src, AllocatedMemory&& dst )
      : BaseClassT_( dst.allocator_.allocate( 0 ), 0, 0, 1 )
  {
    assert( src.size() == dst.size() );
    ( *this ) = std::move( dst );
    allocator_.uninitialized_copy( src.data_, src.data_ + src.size(), data_ );
  }

  /// @brief Destructor.
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE ~AllocatedMemory() { allocator_.deallocate( data_, capacity() ); }

  /**
   * @brief Copy constructor.
   *
   * @param other The AllocatedMemory object to copy.
   *
   * This constructor performs a deep copy of the data from the other AllocatedMemory object.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory( const AllocatedMemory& other )
      : AllocatedMemory( other.size(), AllocatorT_( other.allocator_ ) )
  {
    // deep copy the data
    allocator_.uninitialized_copy( other.data_, other.data_ + other.size(), data_ );
  }

  /**
   * @brief Move constructor.
   *
   * @param other The AllocatedMemory object to move.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory( AllocatedMemory&& other )
      : BaseClassT_( other.data_, other.size(), other.capacity(), other.stride() ), allocator_{ other.allocator_ }
  {
    if constexpr ( !IsSizeEqCapacityT_::value ) {
      other.setSize( 0 );
    }
    if constexpr ( !IsCapacityFixedT_::value ) {
      other.data_ = nullptr;
      other.setCapacity( 0 );
    } else {
      // allocate new memory for the moved object so the size is the same
      other.data_ = allocator_.allocate( other.size() );
    }
  }

  /**
   * @brief Copy assignment operator.
   *
   * @param other The AllocatedMemory object to copy.
   * @return A reference to this AllocatedMemory object.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory& operator=( const AllocatedMemory& other )
  {
    if ( this != &other ) {
      BaseClassT_::operator=( other );
      allocator_ = other.allocator();
      // deep copy the data
      data_ = allocator_.allocate( other.size() );
      allocator_.copy( data_, other.data(), other.size() );
    }
    return *this;
  }

  /**
   * @brief Move assignment operator.
   *
   * @param other The AllocatedMemory object to move.
   * @return A reference to this AllocatedMemory object.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory& operator=( AllocatedMemory&& other )
  {
    if ( this != &other ) {
      BaseClassT_::operator=( std::move( other ) );
      allocator_ = other.allocator_;
      if constexpr ( !IsSizeEqCapacityT_::value ) {
        other.setSize( 0 );
      }
      if constexpr ( !IsCapacityFixedT_::value ) {
        other.data_ = nullptr;
        other.setCapacity( 0 );
      } else {
        // allocate new memory for the moved object so the size is the same
        other.data_ = allocator_.allocate( other.size() );
      }
    }
    return *this;
  }

  /**
   * @brief Get the allocator.
   *
   * @return A const reference to the allocator.
   */
  TRIBOL_HOST_DEVICE const AllocatorT_& allocator() const { return allocator_; }

  /// @brief Get the capacity of the memory block.
  using BaseClassT_::capacity;

  /// @brief Get the size of the memory block.
  using BaseClassT_::size;

  /// @brief Type indicating if capacity can be changed at runtime.
  using typename BaseClassT_::IsCapacityFixedT_;

  /// @brief Type indicating if size is fixed to match capacity.
  using typename BaseClassT_::IsSizeEqCapacityT_;

  /// @brief Type indicating that the memory is not initialized.
  using IsInitializedT_ = std::false_type;

 private:
  /// @brief Using declaration to access the protected data_ member from the base class.
  using BaseClassT_::data_;

  /// @brief Allocator used for memory management.
  AllocatorT_ allocator_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_COMMON_MEMORY_HPP_ */
