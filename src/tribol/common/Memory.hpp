// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_MEMORY_HPP_
#define SRC_TRIBOL_COMMON_MEMORY_HPP_

#include <cassert>
#include <cstddef>

#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/ExecModel.hpp"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/TypedAllocator.hpp"
#endif

namespace tribol {

/**
 * @brief Fixed-capacity container policy parameterized by a compile-time size.
 *
 * This class encodes a capacity that is fixed at compile time, sized via the template parameter _N. It is intended to
 * be used as a policy or helper type to represent containers or buffers whose capacity is known statically.
 *
 * @tparam _N The compile-time capacity (number of elements).
 */
template <size_t _N>
class FixedCapacity {
 public:
  /**
   * @brief Alias for the integral type used to represent sizes and capacities.
   *
   * Defined to improve readability and allow easy changes to the underlying size type if required.
   */
  using SizeType_ = size_t;

  /**
   * @brief Construct a FixedCapacity instance.
   *
   * The constructor accepts a runtime capacity parameter for API uniformity but asserts that the provided value matches
   * the compile-time capacity _N.
   *
   * @param capacity The runtime capacity value that must equal _N (asserted).
   */
  TRIBOL_HOST_DEVICE FixedCapacity( [[maybe_unused]] SizeType_ capacity ) { assert( capacity == _N ); }

  /**
   * @brief Return the compile-time capacity.
   *
   * This function returns the capacity encoded by the template parameter _N.
   *
   * @return The compile-time capacity (_N).
   */
  TRIBOL_HOST_DEVICE constexpr SizeType_ capacity() const { return _N; }

  /**
   * @brief Set capacity at runtime (no-op for fixed-capacity type).
   *
   * For this fixed-capacity policy the capacity cannot be changed at runtime. This function signature exists for
   * interface compatibility and always returns the compile-time capacity _N while ignoring the provided argument.
   *
   * @param new_capacity Runtime capacity value (ignored).
   * @return The compile-time capacity (_N).
   */
  TRIBOL_HOST_DEVICE constexpr SizeType_ setCapacity( SizeType_ ) const { return _N; }

  /**
   * @brief Type trait indicating whether capacity can be changed at runtime.
   *
   * For this type, capacity is known at compile time, so this alias is std::false_type. This can be used in template
   * metaprogramming to select different code paths when capacity is fixed vs. dynamic.
   */
  using CapacityAtRuntime_ = std::false_type;
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
   * @brief Alias for the integer type used to represent capacity values.
   *
   * This alias improves code readability and allows the implementation to change the
   * underlying integer type in one place if needed.
   */
  using SizeType_ = size_t;

  /**
   * @brief Construct a RuntimeCapacity with the given initial capacity.
   *
   * @param capacity The initial capacity value to store.
   *
   * @note This constructor is annotated for host/device usage (TRIBOL_HOST_DEVICE)
   *       so it can be called from both host and device code when compiled for CUDA/HIP.
   */
  TRIBOL_HOST_DEVICE RuntimeCapacity( SizeType_ capacity ) : capacity_( capacity ) {}

  /**
   * @brief Return the currently stored capacity.
   *
   * This accessor does not modify the object and returns the raw capacity value.
   *
   * @return The current capacity.
   *
   * @note Annotated as TRIBOL_HOST_DEVICE to allow calls from host and device code.
   */
  TRIBOL_HOST_DEVICE SizeType_ capacity() const { return capacity_; }

  /**
   * @brief Set the capacity to a new value and return that value.
   *
   * This mutator updates the stored capacity to the provided value and returns the value that was set. It is useful for
   * fluent-style updates or when the caller needs confirmation of the stored value.
   *
   * @param capacity The new capacity value to store.
   * @return The capacity value that was stored (same as the parameter).
   *
   * @note Annotated as TRIBOL_HOST_DEVICE so it can be invoked from both host and
   *       device code. No thread-safety or synchronization is provided by this type;
   *       callers must ensure correct concurrent access semantics if used in parallel code.
   */
  TRIBOL_HOST_DEVICE SizeType_ setCapacity( SizeType_ capacity )
  {
    capacity_ = capacity;
    return capacity;
  }

  /**
   * @brief Tag type indicating that capacity is determined at runtime.
   *
   * Presence of this alias (std::true_type) can be used in metaprogramming to detect that the capacity for an object is
   * not a compile-time constant but instead is provi ded/modified at runtime.
   */
  using CapacityAtRuntime_ = std::true_type;

 private:
  /**
   * @brief Underlying storage for the capacity value.
   *
   * Holds the numeric capacity. This member is private and should be accessed via the public accessor and mutator
   * methods.
   */
  SizeType_ capacity_;
};

/**
 * @brief An array size policy class where the size is always equal to the capacity.
 *
 * @tparam _Capacity The base class that provides the capacity management. It is expected to have a `SizeType_`
 * definition, a constructor taking a `SizeType_`, `capacity()` and `setCapacity()` methods.
 *
 * This class inherits from a given Capacity class and enforces the invariant that the logical size of the object is
 * always the same as its allocated capacity. It is useful for representing data structures that are always full or have
 * a fixed size determined at construction.
 */
template <typename _Capacity>
class SizeEqCapacity : public _Capacity {
 public:
  /**
   * @brief The unsigned/integral type used to represent sizes and capacities.
   */
  using typename _Capacity::SizeType_;

  /**
   * @brief Alias for the underlying Capacity type.
   */
  using CapacityType_ = _Capacity;

  /**
   * @brief Construct a SizeEqCapacity with an explicit size and capacity.
   *
   * @param size The logical size to use (also passed to the base Capacity constructor).
   * @param capacity The capacity value to verify. An assertion checks size == capacity.
   */
  TRIBOL_HOST_DEVICE SizeEqCapacity( SizeType_ size, [[maybe_unused]] SizeType_ capacity ) : _Capacity( size )
  {
    assert( size == capacity );
  }

  /**
   * @brief Construct a SizeEqCapacity with a single size value.
   *
   * @param size The logical size (and implicit capacity) to use.
   */
  TRIBOL_HOST_DEVICE SizeEqCapacity( SizeType_ size ) : _Capacity( size ) {}

  /**
   * @brief Return the current logical size.
   *
   * This returns the underlying capacity value; the two are guaranteed equal.
   *
   * @return The current size (equal to capacity()).
   */
  TRIBOL_HOST_DEVICE constexpr SizeType_ size() const { return capacity(); }

  /**
   * @brief Return the current capacity.
   *
   * Inherited from Capacity; exposed via a using-declaration.
   */
  using _Capacity::capacity;

  /**
   * @brief Set the logical size.
   *
   * Implemented by forwarding to Capacity::setCapacity(size). Because this policy enforces size == capacity, setting
   * the size is equivalent to setting capacity.
   *
   * @param size The new size (and capacity) to set.
   * @return The resulting capacity after the operation (value returned by setCapacity).
   */
  TRIBOL_HOST_DEVICE SizeType_ setSize( SizeType_ size ) { return setCapacity( size ); }

  /**
   * @brief Set the capacity.
   *
   * Inherited from Capacity; exposed via a using-declaration.
   */
  using _Capacity::setCapacity;

  /**
   * @brief Query whether size is at capacity.
   *
   * Always returns true for this policy since size() == capacity() by design.
   *
   * @return true
   */
  TRIBOL_HOST_DEVICE constexpr bool sizeAtCapacity() const { return true; }

  /**
   * @brief Tag type indicating that the policy represents a fixed-size container.
   */
  using FixedSize_ = std::true_type;
};

/**
 * @brief An array size policy class where the size can be less than or equal to the capacity.
 *
 * @tparam _Capacity The base class that provides the capacity management. It is expected to have a `SizeType_`
 * definition, a constructor taking a `SizeType_`, `capacity()` and `setCapacity()` methods.
 *
 * This class inherits from a given Capacity class and allows the logical size of the object to be less than or equal to
 * its allocated capacity. It is useful for representing data structures that may not always be full.
 */
template <typename _Capacity>
class SizeLECapacity : public _Capacity {
 public:
  /**
   * @brief The unsigned/integral type used to represent sizes and capacities.
   */
  using typename _Capacity::SizeType_;

  /**
   * @brief Alias for the underlying Capacity type.
   */
  using CapacityType_ = _Capacity;

  /**
   * @brief Construct a SizeLECapacity with an explicit size and capacity.
   *
   * @param size The logical size to use (also passed to the base Capacity constructor).
   * @param capacity The capacity value to verify. An assertion checks size <= capacity.
   */
  TRIBOL_HOST_DEVICE SizeLECapacity( SizeType_ size, SizeType_ capacity )
      : _Capacity( capacity >= size ? capacity : size ), size_( size )
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
  TRIBOL_HOST_DEVICE SizeType_ size() const { return size_; }

  /**
   * @brief Return the current capacity.
   *
   * Inherited from Capacity; exposed via a using-declaration.
   */
  using _Capacity::capacity;

  /**
   * @brief Set the logical size.
   *
   * @param size The new size to set (must be <= capacity).
   * @return The resulting size after the operation.
   */
  TRIBOL_HOST_DEVICE SizeType_ setSize( SizeType_ size )
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

  /**
   * @brief Tag type indicating that the policy represents a dynamic-size container.
   */
  using FixedSize_ = std::false_type;

 private:
  /**
   * @brief Underlying storage for the logical size.
   *
   * Holds the numeric size. This member is private and should be accessed via the public accessor and mutator methods.
   */
  SizeType_ size_;
};

/**
 * @brief Contiguous memory accessor with unit stride.
 *
 * @tparam _T The type of elements stored in the memory.
 * @tparam _SizeAndCapacity The policy class that provides size and capacity management. The policy class is expected
 * to have `SizeType_`, `CapacityType_`, and `FixedSize_` type definitions; `size()`, `capacity()`, `setSize()`,
 * `setCapacity()`, and `sizeAtCapacity()` member functions; and a constructor taking size and capacity parameters.
 *
 * This class provides random access to a contiguous block of memory with unit stride.
 * It inherits from a SizeAndCapacity policy to manage size and capacity semantics.
 */
template <typename _T, class _SizeAndCapacity>
class ContiguousMemory : public _SizeAndCapacity {
 public:
  /**
   * @brief The unsigned/integral type used to represent sizes and capacities.
   */
  using typename _SizeAndCapacity::SizeType_;

  /**
   * @brief Alias for the underlying SizeAndCapacity type.
   */
  using SizeAndCapacityType_ = _SizeAndCapacity;

  /**
   * @brief The type of elements stored in the memory.
   */
  using ValueType_ = _T;

  /**
   * @brief Pointer type for accessing elements.
   */
  using Pointer_ = _T*;

  /**
   * @brief Const pointer type for accessing elements.
   */
  using ConstPointer_ = const _T*;

  /**
   * @brief Construct a ContiguousMemory accessor.
   *
   * @param data Pointer to the beginning of the memory block.
   * @param size The number of elements in the memory block.
   * @param capacity The total capacity of the memory block.
   */
  TRIBOL_HOST_DEVICE ContiguousMemory( Pointer_ data, SizeType_ size, SizeType_ capacity )
      : _SizeAndCapacity( size, capacity ), data_( data )
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
  TRIBOL_HOST_DEVICE ContiguousMemory( Pointer_ data, SizeType_ size, SizeType_ capacity,
                                       [[maybe_unused]] SizeType_ stride )
      : _SizeAndCapacity( size, capacity ), data_( data )
  {
    assert( stride == 1 );
  }

  /**
   * @brief Access an element at a specific index.
   *
   * @param i The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  TRIBOL_HOST_DEVICE ValueType_& at( SizeType_ i ) { return *( data_ + i ); }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueType_& at( SizeType_ i ) const { return *( data_ + i ); }

  /**
   * @brief Access an element at a specific index using the subscript operator.
   *
   * @param i The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  TRIBOL_HOST_DEVICE ValueType_& operator[]( SizeType_ i ) { return at( i ); }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueType_& operator[]( SizeType_ i ) const { return at( i ); }

  /**
   * @brief Iterator type for the memory block.
   */
  using IteratorType_ = Pointer_;

  /**
   * @brief Const iterator type for the memory block.
   */
  using ConstIteratorType_ = ConstPointer_;

  /**
   * @brief Get an iterator to the beginning of the memory block.
   *
   * @return An iterator to the first element.
   */
  TRIBOL_HOST_DEVICE Pointer_ begin() { return data_; }
  /// @overload
  TRIBOL_HOST_DEVICE ConstPointer_ begin() const { return data_; }

  /**
   * @brief Get an iterator to the end of the memory block.
   *
   * @return An iterator to one past the last element.
   */
  TRIBOL_HOST_DEVICE Pointer_ end() { return data_ + size(); }
  /// @overload
  TRIBOL_HOST_DEVICE ConstPointer_ end() const { return data_ + size(); }

  /**
   * @brief Get the size of the memory block.
   */
  using _SizeAndCapacity::size;

  /**
   * @brief Get the stride between elements.
   *
   * @return The stride (always 1 for this class).
   */
  TRIBOL_HOST_DEVICE constexpr SizeType_ stride() const { return 1; }

  /**
   * @brief Get a pointer to the underlying data.
   *
   * @return A pointer to the beginning of the memory block.
   */
  TRIBOL_HOST_DEVICE Pointer_ data() const { return data_; }

  /**
   * @brief Implicit conversion to a pointer.
   *
   * @return A pointer to the beginning of the memory block.
   */
  TRIBOL_HOST_DEVICE operator Pointer_() const { return data_; }

 protected:
  /**
   * @brief Pointer to the beginning of the memory block.
   */
  Pointer_ data_;
};

/**
 * @brief Memory accessor with a fixed stride.
 *
 * @tparam _T The type of elements stored in the memory.
 * @tparam _SizeAndCapacity The policy class that provides size and capacity management.  The policy class is expected
 * to have `SizeType_`, `CapacityType_`, and `FixedSize_` type definitions; `size()`, `capacity()`, `setSize()`,
 * `setCapacity()`, and `sizeAtCapacity()` member functions; and a constructor taking size and capacity parameters.
 *
 * This class provides random access to a block of memory with a fixed stride. It inherits from a SizeAndCapacity policy
 * to manage size and capacity semantics.
 */
template <typename _T, class _SizeAndCapacity>
class FixedStride : public _SizeAndCapacity {
 public:
  /**
   * @brief The unsigned/integral type used to represent sizes and capacities.
   */
  using typename _SizeAndCapacity::SizeType_;

  /**
   * @brief Alias for the underlying Capacity type.
   */
  using typename _SizeAndCapacity::CapacityType_;

  /**
   * @brief Alias for the underlying SizeAndCapacity type.
   */
  using SizeAndCapacityType_ = _SizeAndCapacity;

  /**
   * @brief The type of elements stored in the memory.
   */
  using ValueType_ = _T;

  /**
   * @brief Pointer type for accessing elements.
   */
  using Pointer_ = _T*;

  /**
   * @brief Const pointer type for accessing elements.
   */
  using ConstPointer_ = const _T*;

  /**
   * @brief Construct a FixedStride accessor.
   *
   * @param data Pointer to the beginning of the memory block.
   * @param size The number of elements in the memory block.
   * @param capacity The total capacity of the memory block.
   * @param stride The stride between elements.
   */
  TRIBOL_HOST_DEVICE FixedStride( Pointer_ data, SizeType_ size, SizeType_ capacity, SizeType_ stride )
      : _SizeAndCapacity( size, capacity ), data_( data ), stride_( stride )
  {
    assert( stride > 0 );
  }

  /**
   * @brief Base class for iterators.
   *
   * @tparam _Ptr The pointer type for the iterator.
   */
  template <typename _Ptr>
  struct IteratorBase {
    /**
     * @brief Construct an IteratorBase.
     *
     * @param ptr Pointer to the current element.
     * @param stride The stride between elements.
     */
    TRIBOL_HOST_DEVICE IteratorBase( _Ptr ptr, SizeType_ stride ) : ptr_( ptr ), stride_( stride ) {}

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
    TRIBOL_HOST_DEVICE IteratorBase operator+( SizeType_ n ) const
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
    TRIBOL_HOST_DEVICE IteratorBase operator-( SizeType_ n ) const
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
    TRIBOL_HOST_DEVICE ValueType_& operator*() { return *ptr_; }

   private:
    /**
     * @brief Pointer to the beginning of the memory block.
     */
    _Ptr ptr_;

    /**
     * @brief The stride between elements.
     */
    SizeType_ stride_;
  };

  /**
   * @brief Iterator type for the memory block.
   */
  using IteratorType_ = IteratorBase<Pointer_>;

  /**
   * @brief Const iterator type for the memory block.
   */
  using ConstIteratorType_ = IteratorBase<ConstPointer_>;

  /**
   * @brief Access an element at a specific index.
   *
   * @param i The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  TRIBOL_HOST_DEVICE ValueType_& at( SizeType_ i ) { return *( data_ + i * stride_ ); }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueType_& at( SizeType_ i ) const { return *( data_ + i * stride_ ); }

  /**
   * @brief Access an element at a specific index using the subscript operator.
   *
   * @param i The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  TRIBOL_HOST_DEVICE ValueType_& operator[]( SizeType_ i ) { return at( i ); }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueType_& operator[]( SizeType_ i ) const { return at( i ); }

  /**
   * @brief Get an iterator to the beginning of the memory block.
   *
   * @return An iterator to the first element.
   */
  TRIBOL_HOST_DEVICE IteratorType_ begin() { return IteratorType_( data_, stride_ ); }
  /// @overload
  TRIBOL_HOST_DEVICE ConstIteratorType_ begin() const { return ConstIteratorType_( data_, stride_ ); }

  /**
   * @brief Get an iterator to the end of the memory block.
   *
   * @return An iterator to one past the last element.
   */
  TRIBOL_HOST_DEVICE IteratorType_ end() { return IteratorType_( data_ + size() * stride_, stride_ ); }
  /// @overload
  TRIBOL_HOST_DEVICE ConstIteratorType_ end() const { return ConstIteratorType_( data_ + size() * stride_, stride_ ); }

  /**
   * @brief Get the size of the memory block.
   */
  using _SizeAndCapacity::size;

  /**
   * @brief Get the stride between elements.
   *
   * @return The stride.
   */
  TRIBOL_HOST_DEVICE SizeType_ stride() const { return stride_; }

  /**
   * @brief Get a pointer to the underlying data.
   *
   * @return A pointer to the beginning of the memory block.
   */
  TRIBOL_HOST_DEVICE Pointer_ data() const { return data_; }

  /**
   * @brief Implicit conversion to a pointer.
   *
   * @return A pointer to the beginning of the memory block.
   */
  TRIBOL_HOST_DEVICE operator Pointer_() const { return data_; }

 protected:
  /**
   * @brief Pointer to the beginning of the memory block.
   */
  Pointer_ data_;

  /**
   * @brief The stride between elements.
   */
  SizeType_ stride_;
};

/**
 * @brief Base class for a memory block with arbitrary stride and size/capacity management. No data ownership so it can
 * be used as a view.
 *
 * @tparam _Accessor The accessor policy that provides data access and size/capacity management. The accessor is
 * expected to have `SizeType_`, `Pointer_`, and `ValueType_` type definitions, as well as appropriate constructors and
 * data access methods.
 */
template <class _Accessor>
class Memory : public _Accessor {
 public:
  /**
   * @brief The unsigned/integral type used to represent sizes and capacities.
   */
  using typename _Accessor::SizeType_;

  /**
   * @brief Pointer type for the memory block.
   */
  using typename _Accessor::Pointer_;

  /**
   * @brief The type of elements stored in the memory.
   */
  using typename _Accessor::ValueType_;

  /**
   * @brief Alias for the underlying Accessor type.
   */
  using AccessorType_ = _Accessor;

  /**
   * @brief Alias for the view type.
   */
  using ViewType_ = Memory<_Accessor>;

  /**
   * @brief Construct a Memory view.
   *
   * @param data Pointer to the beginning of the memory block.
   * @param size The number of elements in the memory block.
   * @param capacity The total capacity of the memory block.
   * @param stride The stride between elements.
   */
  TRIBOL_HOST_DEVICE Memory( Pointer_ data, SizeType_ size, SizeType_ capacity, SizeType_ stride )
      : _Accessor( data, size, capacity, stride )
  {
  }

  /**
   * @brief Construct a Memory view with size equal to capacity.
   *
   * @param data Pointer to the beginning of the memory block.
   * @param size The number of elements in the memory block.
   * @param stride The stride between elements.
   */
  TRIBOL_HOST_DEVICE Memory( Pointer_ data, SizeType_ size, SizeType_ stride = 1 ) : Memory( data, size, size, stride )
  {
  }

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
  TRIBOL_HOST_DEVICE ValueType_& at( SizeType_ i )
  {
    assert( i < size() );
    return _Accessor::at( i );
  }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueType_& at( SizeType_ i ) const
  {
    assert( i < size() );
    return _Accessor::at( i );
  }

  /**
   * @brief Access an element at a specific index using the subscript operator with bounds checking.
   *
   * @param i The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  TRIBOL_HOST_DEVICE ValueType_& operator[]( SizeType_ i )
  {
    assert( i < size() );
    return _Accessor::operator[]( i );
  }
  /// @overload
  TRIBOL_HOST_DEVICE const ValueType_& operator[]( SizeType_ i ) const
  {
    assert( i < size() );
    return _Accessor::operator[]( i );
  }

  /**
   * @brief Create a new view of a sub-region of the memory.
   *
   * @tparam NewAccessor The accessor policy for the new view.
   * @param offset The starting offset of the sub-region.
   * @param size The number of elements in the sub-region.
   * @param capacity The total capacity of the sub-region.
   * @param stride The stride between elements in the sub-region.
   * @return A new Memory view of the sub-region.
   */
  template <typename NewAccessor>
  TRIBOL_HOST_DEVICE Memory<NewAccessor> view( SizeType_ offset, SizeType_ size, SizeType_ capacity,
                                               SizeType_ stride = 1 ) const
  {
    assert( offset + size * stride <= this->size() );
    return Memory<NewAccessor>( data() + offset, size, capacity, stride );
  }

  /**
   * @brief Create a new view of the entire memory block.
   *
   * @return A new Memory view of the entire memory block.
   */
  TRIBOL_HOST_DEVICE Memory<_Accessor> view() { return Memory<_Accessor>( *this ); }
  /// @overload
  TRIBOL_HOST_DEVICE Memory<_Accessor> view() const { return Memory<_Accessor>( *this ); }

  /**
   * @brief Get the size of the memory block.
   */
  using _Accessor::size;

  /**
   * @brief Get a pointer to the underlying data.
   */
  using _Accessor::data;

  /**
   * @brief Tag type indicating that the memory is initialized.
   *
   * The memory is assumed to be a view that is already initialized.
   */
  using Initialized_ = std::true_type;
};

/**
 * @brief A fixed-size memory block allocated on the stack.
 *
 * @tparam _T The type of elements stored in the memory.
 * @tparam _N The compile-time size of the memory block.
 * @tparam _SizeVsCapacity The policy class that manages size vs. capacity. The policy class is expected
 * to have `SizeType_`, `CapacityType_`, and `FixedSize_` definitions, `size()`, `capacity()`, `setSize()`,
 * `setCapacity()`, and `sizeAtCapacity()` member functions, and a constructor taking size and capacity parameters.
 *
 * This class provides a fixed-size memory block allocated on the stack, with an interface similar to a standard
 * container.
 */
template <typename _T, size_t _N, template <typename> class _SizeVsCapacity = SizeEqCapacity>
class StackMemory : public Memory<ContiguousMemory<_T, _SizeVsCapacity<FixedCapacity<_N>>>> {
 public:
  /**
   * @brief Base class for the StackMemory.
   */
  using BaseClass_ = Memory<ContiguousMemory<_T, _SizeVsCapacity<FixedCapacity<_N>>>>;

  /**
   * @brief The unsigned/integral type used to represent sizes and capacities.
   */
  using typename BaseClass_::SizeType_;

  /**
   * @brief Construct a StackMemory object.
   *
   * @param size The initial size of the memory block.
   */
  TRIBOL_HOST_DEVICE StackMemory( SizeType_ size = _N ) : BaseClass_( nullptr, size, _N, 1 ) { data_ = stack_data_; }

  /**
   * @brief Construct a StackMemory object with explicit size and capacity.
   *
   * @param size The initial size of the memory block.
   * @param capacity The capacity of the memory block (must be N).
   */
  TRIBOL_HOST_DEVICE StackMemory( SizeType_ size, [[maybe_unused]] SizeType_ capacity ) : StackMemory( size )
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
      for ( SizeType_ i = 0; i < other.size(); ++i ) {
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
      BaseClass_::operator=( other );
      for ( SizeType_ i = 0; i < other.size(); ++i ) {
        stack_data_[i] = other.stack_data_[i];
      }
    }
    return *this;
  }

  /**
   * @brief Destructor.
   */
  TRIBOL_DEFAULT_HOST_DEVICE ~StackMemory() = default;

  /**
   * @brief Tag type indicator that the memory is not initialized.
   */
  using Initialized_ = std::false_type;

 private:
  /**
   * @brief Using declaration to access the protected data_ member from the base class.
   */
  using BaseClass_::data_;

  /**
   * @brief Underlying stack-allocated data storage.
   */
  _T stack_data_[_N];
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
  /**
   * @brief The type of elements to allocate.
   */
  using ValueType_ = _T;

  /**
   * @brief The unsigned/integral type used to represent sizes.
   */
  using SizeType_ = size_t;

  /**
   * @brief Pointer type for the allocated memory.
   */
  using Pointer_ = _T*;

  /**
   * @brief The signed/integral type used to represent differences between pointers.
   */
  using DIfferenceType_ = ptrdiff_t;

  /**
   * @brief Allocate a block of memory.
   *
   * @param n The number of elements to allocate.
   * @return A pointer to the allocated memory.
   */
  TRIBOL_HOST_DEVICE Pointer_ allocate( SizeType_ n ) const
  {
    return static_cast<Pointer_>( ::operator new( n * sizeof( ValueType_ ) ) );
  }

  /**
   * @brief Deallocate a block of memory.
   *
   * @param p A pointer to the memory to deallocate.
   * @param n The number of elements that were allocated.
   */
  TRIBOL_HOST_DEVICE void deallocate( Pointer_ p, SizeType_ ) const { ::operator delete( p ); }

  /**
   * @brief Copy uninitialized memory.
   *
   * @param first A pointer to the beginning of the source range.
   * @param last A pointer to the end of the source range.
   * @param d_first A pointer to the beginning of the destination range.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE void uninitialized_copy( Pointer_ first, Pointer_ last, Pointer_ d_first ) const
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
  /**
   * @brief The type of elements to allocate.
   */
  using ValueType_ = _T;

  /**
   * @brief The unsigned/integral type used to represent sizes.
   */
  using SizeType_ = size_t;

  /**
   * @brief Pointer type for the allocated memory.
   */
  using Pointer_ = _T*;

  /**
   * @brief The signed/integral type used to represent differences between pointers.
   */
  using DIfferenceType_ = ptrdiff_t;

  /**
   * @brief Construct an UmpireAllocator.
   *
   * @param allocator The Umpire allocator to use.
   */
  UmpireAllocator( umpire::Allocator allocator ) : allocator_{ std::move( allocator ) } {}

  /**
   * @brief Construct an UmpireAllocator for a specific memory space.
   */
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
  Pointer_ allocate( SizeType_ n ) const { return static_cast<Pointer_>( allocator_.allocate( n ) ); }

  /**
   * @brief Deallocate a block of memory.
   *
   * @param p A pointer to the memory to deallocate.
   * @param n The number of elements that were allocated.
   */
  void deallocate( Pointer_ p, SizeType_ n ) const { allocator_.deallocate( p, n ); }

  /**
   * @brief Copy uninitialized memory.
   *
   * @param first A pointer to the beginning of the source range.
   * @param last A pointer to the end of the source range.
   * @param d_first A pointer to the beginning of the destination range.
   */
  void uninitialized_copy( Pointer_ first, Pointer_, Pointer_ d_first ) const
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
  mutable umpire::TypedAllocator<ValueType_> allocator_;
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
  /**
   * @brief The type of elements to allocate.
   */
  using ValueType_ = _T;

  /**
   * @brief The unsigned/integral type used to represent sizes.
   */
  using SizeType_ = size_t;

  /**
   * @brief Pointer type for the allocated memory.
   */
  using Pointer_ = _T*;

  /**
   * @brief Construct a DynamicAllocator with the default allocator ID.
   */
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
  TRIBOL_HOST_DEVICE Pointer_ allocate( SizeType_ n ) const
  {
    return static_cast<Pointer_>(
#ifdef TRIBOL_USE_UMPIRE
        umpire::ResourceManager::getInstance().getAllocator( allocator_id_ ).allocate( n * sizeof( ValueType_ ) )
#else
        Allocator<ValueType_>().allocate( n )
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
  TRIBOL_HOST_DEVICE void deallocate( Pointer_ p, [[maybe_unused]] SizeType_ n ) const
  {
#ifdef TRIBOL_USE_UMPIRE
    umpire::ResourceManager::getInstance().getAllocator( allocator_id_ ).deallocate( p );
#else
    Allocator<ValueType_>().deallocate( p, n );
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
  TRIBOL_HOST_DEVICE void uninitialized_copy( Pointer_ first, [[maybe_unused]] Pointer_ last, Pointer_ d_first ) const
  {
#ifdef TRIBOL_USE_UMPIRE
    auto& rm = umpire::ResourceManager::getInstance();
    rm.copy( d_first, first );
#else
    Allocator<ValueType_>().uninitialized_copy( first, last, d_first );
#endif
  }

  /**
   * @brief Get the ID of the allocator.
   *
   * @return The allocator ID.
   */
  TRIBOL_HOST_DEVICE int id() const { return allocator_id_; }

 private:
  /**
   * @brief The ID of the allocator to use.
   */
  int allocator_id_;
};

/**
 * @brief A class that manages allocated memory.
 *
 * @tparam _T The type of elements stored in the memory.
 * @tparam _Allocator The allocator to use for memory management. The allocator must provide `allocate()`,
 * `deallocate()`, and `uninitialized_copy()` methods.
 * @tparam _SizeVsCapacity The policy class that manages size vs. capacity. The policy class is expected
 * to have `SizeType_`, `CapacityType_`, and `FixedSize_` definitions, `size()`, `capacity()`, `setSize()`,
 * `setCapacity()`, and `sizeAtCapacity()` member functions, and a constructor taking size and capacity parameters.
 *
 * This class provides a container-like interface for a block of memory that is dynamically allocated and managed by an
 * Allocator.
 */
template <typename _T, class _Allocator = Allocator<_T>, class _SizeVsCapacity = SizeLECapacity<RuntimeCapacity>>
class AllocatedMemory : public Memory<ContiguousMemory<_T, _SizeVsCapacity>> {
 public:
  /**
   * @brief Base class for the AllocatedMemory.
   */
  using BaseClass_ = Memory<ContiguousMemory<_T, _SizeVsCapacity>>;

  /**
   * @brief The unsigned/integral type used to represent sizes and capacities.
   */
  using typename BaseClass_::SizeType_;

  /**
   * @brief Pointer type for the memory block.
   */
  using typename BaseClass_::pointer;

  /**
   * @brief Value type for the memory block.
   */
  using typename BaseClass_::ValueType_;

  static_assert( std::is_same<typename _Allocator::ValueType_, ValueType_>::value,
                 "AllocatedMemory must be used with same type as allocator" );

  /**
   * @brief Construct an AllocatedMemory object.
   *
   * @param size The initial size of the memory block.
   * @param capacity The capacity of the memory block.
   * @param allocator The allocator to use.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory( SizeType_ size, SizeType_ capacity, _Allocator allocator = _Allocator() )
      : BaseClass_( allocator.allocate( capacity ), size, capacity, 1 ), allocator_( std::move( allocator ) )
  {
  }

  /**
   * @brief Construct an AllocatedMemory object with size equal to capacity.
   *
   * @param size The initial size and capacity of the memory block.
   * @param allocator The allocator to use.
   */
  TRIBOL_NVCC_EXEC_CHECK_DISABLE
  TRIBOL_HOST_DEVICE AllocatedMemory( SizeType_ size, _Allocator allocator = _Allocator() )
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
  TRIBOL_HOST_DEVICE AllocatedMemory( BaseClass_&& memory, _Allocator&& allocator = _Allocator() )
      : BaseClass_( std::move( memory ) ), allocator_( std::move( allocator ) )
  {
  }

  // /**
  //  * @brief Construct an AllocatedMemory object by copying from a source and moving from a destination.
  //  *
  //  * @param src The source object to copy from.
  //  * @param dst The destination object to move from.
  //  *
  //  * This constructor creates a new AllocatedMemory object by copying data from the source object using the allocator
  //  in
  //  * the destination object. Conceptually, this is similar to a copy constructor, but it allows the allocator
  //  */
  // TRIBOL_NVCC_EXEC_CHECK_DISABLE
  // TRIBOL_HOST_DEVICE AllocatedMemory( const AllocatedMemory& src, AllocatedMemory&& dst )
  //     : BaseClass( dst.allocator_.allocate( 0 ), 0, 0, 1 )
  // {
  //   assert( src.size() == dst.size() );
  //   ( *this ) = std::move( dst );
  //   allocator_.uninitialized_copy( src.data_, src.data_ + src.size(), data_ );
  // }

  /**
   * @brief Destructor.
   */
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
      : AllocatedMemory( other.size(), Allocator( other.allocator_ ) )
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
      : BaseClass_( other.data_, other.size(), other.capacity(), other.stride() ), allocator_{ other.allocator_ }
  {
    if constexpr ( !FixedSize_::value ) {
      other.setSize( 0 );
    }
    if constexpr ( CapacityAtRuntime_::value ) {
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
      BaseClass_::operator=( other );
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
      BaseClass_::operator=( std::move( other ) );
      allocator_ = other.allocator_;
      if constexpr ( !FixedSize_::value ) {
        other.setSize( 0 );
      }
      if constexpr ( CapacityAtRuntime_::value ) {
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
  TRIBOL_HOST_DEVICE const _Allocator& allocator() const { return allocator_; }

  /**
   * @brief Get the capacity of the memory block.
   */
  using BaseClass_::capacity;
  /**
   * @brief Get the size of the memory block.
   */
  using BaseClass_::size;

  /**
   * @brief Tag type indicating if capacity can be changed at runtime.
   */
  using typename BaseClass_::CapacityAtRuntime_;

  /**
   * @brief Tag type indicating if size is fixed at compile time.
   */
  using typename BaseClass_::FixedSize_;

  /**
   * @brief Tag type indicating that the memory is not initialized.
   */
  using Initialized_ = std::false_type;

 private:
  using BaseClass_::data_;
  _Allocator allocator_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_COMMON_MEMORY_HPP_ */
