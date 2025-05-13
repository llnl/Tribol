// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef TRIBOL_COMMON_CONTAINERS_HPP_
#define TRIBOL_COMMON_CONTAINERS_HPP_

#include <cassert>

// Tribol includes
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Memory.hpp"

namespace tribol {

template <typename Memory>
class ArrayBase {
 public:
  using value_type = typename Memory::value_type;
  using pointer = typename Memory::pointer;
  using const_pointer = typename Memory::const_pointer;
  using size_type = typename Memory::size_type;

  TRIBOL_HOST_DEVICE ArrayBase( Memory&& memory ) : memory_( std::move( memory ) )
  {
    // initialize memory
    for ( auto& value : memory_ ) {
      value = value_type{};
    }
  }
  TRIBOL_HOST_DEVICE ~ArrayBase()
  {
    // call destructor on all elements
    for ( auto& value : memory_ ) {
      value.~value_type();
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

  using iterator_type = typename Memory::iterator_type;
  using const_iterator_type = typename Memory::const_iterator_type;

  TRIBOL_HOST_DEVICE iterator_type begin() { return memory_.begin(); }
  TRIBOL_HOST_DEVICE iterator_type end() { return memory_.end(); }

  TRIBOL_HOST_DEVICE const_iterator_type begin() const { return memory_.begin(); }
  TRIBOL_HOST_DEVICE const_iterator_type end() const { return memory_.end(); }

  TRIBOL_HOST_DEVICE Memory& memory() { return memory_; }
  TRIBOL_HOST_DEVICE const Memory& memory() const { return memory_; }

 protected:
  Memory memory_;
};

template <typename T, IndexT N, class Memory = StackMemory<T, N>>
class FixedArray : public ArrayBase<Memory> {
 public:
  using value_type = T;
  using BaseClass = ArrayBase<Memory>;
  using typename BaseClass::const_pointer;
  using typename BaseClass::pointer;
  using typename BaseClass::size_type;

  static_assert( std::is_same<typename Memory::value_type, value_type>::value,
                 "BoundedArray must be used with same type as memory" );

  TRIBOL_HOST_DEVICE FixedArray( Memory&& memory = Memory( N ) ) : BaseClass( std::move( memory ) ) {}

  using BaseClass::at;
  using BaseClass::operator[];

  using typename BaseClass::const_iterator_type;
  using typename BaseClass::iterator_type;

  using BaseClass::begin;
  using BaseClass::end;

  TRIBOL_HOST_DEVICE constexpr size_type size() const { return N; }

 private:
  using BaseClass::memory_;
};

template <typename T, class Memory = AllocatedMemory<T>>
class BoundedArray : public ArrayBase<Memory> {
 public:
  using value_type = T;
  using BaseClass = ArrayBase<Memory>;
  using typename BaseClass::const_pointer;
  using typename BaseClass::pointer;
  using typename BaseClass::size_type;

  static_assert( Memory::fixed_size_ == false, "BoundedArray must be used with non-fixed size memory" );
  static_assert( std::is_same<typename Memory::value_type, value_type>::value,
                 "BoundedArray must be used with same type as memory" );

  TRIBOL_HOST_DEVICE BoundedArray( size_type size, size_type capacity ) : BaseClass( Memory( size, capacity ) ) {}
  TRIBOL_HOST_DEVICE BoundedArray( Memory&& memory ) : BaseClass( std::move( memory ) ) {}

  using BaseClass::at;
  using BaseClass::operator[];

  using typename BaseClass::const_iterator_type;
  using typename BaseClass::iterator_type;

  using BaseClass::begin;
  using BaseClass::end;

  TRIBOL_HOST_DEVICE constexpr size_type size() const { return memory_.size(); }
  TRIBOL_HOST_DEVICE constexpr size_type capacity() const { return memory_.capacity(); }

  TRIBOL_HOST_DEVICE void push_back( T value ) { operator[]( memory_.setSize( size() + 1 ) - 1 ) = value; }
  template <typename... Args>
  TRIBOL_HOST_DEVICE void emplace_back( Args&&... args )
  {
    operator[]( memory_.setSize( size() + 1 ) - 1 ) = T( std::forward<Args>( args )... );
  }
  TRIBOL_HOST_DEVICE void pop_back() { memory_.setSize( size() - 1 ); }
  TRIBOL_HOST_DEVICE void resize( size_type new_size )
  {
    assert( new_size <= capacity() );
    // destruct elements no longer in range
    for ( size_type i{ new_size }; i < size(); ++i ) {
      operator[]( i ).~T();
    }
    // create empty new elements
    for ( size_type i{ size() }; i < new_size; ++i ) {
      operator[]( i ) = T{};
    }
    memory_.setSize( new_size );
  }

 private:
  using BaseClass::memory_;
};

template <typename T, class Allocator = HeapAllocator<T>>
class Array : public BoundedArray<AllocatedMemory<T, Allocator>> {
 public:
  using value_type = T;
  using BaseClass = BoundedArray<AllocatedMemory<T, Allocator>>;
  using memory_type = typename BaseClass::memory_type;
  using pointer = typename BaseClass::pointer;
  using const_pointer = typename BaseClass::const_pointer;
  using size_type = typename BaseClass::size_type;

  constexpr static size_type default_capacity_ = 32;
  constexpr static RealT default_resize_ratio_ = 2.0;

#pragma nv_exec_check_disable
  TRIBOL_HOST_DEVICE Array( size_type size = 0, size_type capacity = default_capacity_ )
      : BaseClass( AllocatedMemory<T, Allocator>( size, capacity >= size ? capacity : size ) )
  {
  }

  using BaseClass::at;
  using BaseClass::operator[];

  using typename BaseClass::const_iterator_type;
  using typename BaseClass::iterator_type;

  using BaseClass::begin;
  using BaseClass::end;

  using BaseClass::capacity;
  using BaseClass::size;

  TRIBOL_HOST_DEVICE void push_back( T value )
  {
    if ( size() >= capacity() ) {
      setCapacity( static_cast<size_type>( capacity() * resize_ratio_ ) );
    }
    BaseClass::push_back( value );
  }
  template <typename... Args>
  TRIBOL_HOST_DEVICE void emplace_back( Args&&... args )
  {
    if ( size() >= capacity() ) {
      setCapacity( static_cast<size_type>( capacity() * resize_ratio_ ) );
    }
    BaseClass::emplace_back( std::forward<Args>( args )... );
  }
  using BaseClass::pop_back;
  TRIBOL_HOST_DEVICE void resize( size_type new_size )
  {
    if ( new_size > capacity() ) {
      setCapacity( static_cast<size_type>( new_size ) );
    }
    BaseClass::resize( new_size );
  }

 private:
  TRIBOL_HOST_DEVICE void setCapacity( size_type new_capacity )
  {
    assert( new_capacity > size() );
    AllocatedMemory<T, Allocator> new_memory( size(), new_capacity );
    memory_.allocator().copy( new_memory.data(), memory_.data(), size() );
    memory_ = std::move( new_memory );
  }

  using BaseClass::memory_;
  RealT resize_ratio_ = default_resize_ratio_;
};

// template <typename T>

/**
 * @brief Storage base class for device-compatible free store (heap) allocated
 * arrays
 *
 * @tparam T Datatype stored in array
 */
template <typename T>
class DeviceArrayData {
 protected:
  TRIBOL_HOST_DEVICE DeviceArrayData() : size_{ 0 }, data_{ nullptr } {}
  TRIBOL_HOST_DEVICE DeviceArrayData( IndexT size ) : size_{ size }, data_{ new T[size] } {}
  TRIBOL_HOST_DEVICE virtual ~DeviceArrayData() { deleteData(); }

  TRIBOL_HOST_DEVICE DeviceArrayData( const DeviceArrayData& other ) : DeviceArrayData( other.size_ )
  {
    // deep copy data
    for ( IndexT i{ 0 }; i < size_; ++i ) {
      data_[i] = other.data_[i];
    }
  }
  TRIBOL_HOST_DEVICE DeviceArrayData( DeviceArrayData&& other ) : size_{ other.size_ }, data_{ other.data_ }
  {
    // reset other
    other.size_ = 0;
    other.data_ = nullptr;
  }

  TRIBOL_HOST_DEVICE DeviceArrayData& operator=( const DeviceArrayData& other )
  {
    deleteData();
    size_ = other.size_;
    data_ = new T[size_];
    // deep copy data
    for ( IndexT i{ 0 }; i < size_; ++i ) {
      data_[i] = other.data_[i];
    }
    return *this;
  }

  TRIBOL_HOST_DEVICE DeviceArrayData& operator=( DeviceArrayData&& other )
  {
    deleteData();
    size_ = other.size_;
    data_ = other.data_;
    other.size_ = 0;
    other.data_ = nullptr;
    return *this;
  }

  IndexT size_;
  T* data_;

 private:
  TRIBOL_HOST_DEVICE void deleteData()
  {
    if ( data_ != nullptr ) {
      delete[] data_;
      size_ = 0;
      data_ = nullptr;
    }
  }
};

/**
 * @brief Simple free store (heap) allocated array that can be created on device
 *
 * @tparam T Datatype stored in array
 */
template <typename T>
class DeviceArray : public DeviceArrayData<T> {
 public:
  TRIBOL_HOST_DEVICE DeviceArray() : DeviceArrayData<T>() {}
  TRIBOL_HOST_DEVICE DeviceArray( IndexT size ) : DeviceArrayData<T>( size ) {}
  TRIBOL_DEFAULT_HOST_DEVICE ~DeviceArray() = default;

  TRIBOL_HOST_DEVICE DeviceArray( const DeviceArray& other ) : DeviceArrayData<T>( other ) {}
  TRIBOL_HOST_DEVICE DeviceArray( DeviceArray&& other ) : DeviceArrayData<T>( std::move( other ) ) {}

  TRIBOL_HOST_DEVICE DeviceArray& operator=( const DeviceArray& other )
  {
    DeviceArrayData<T>::operator=( other );
    return *this;
  }

  TRIBOL_HOST_DEVICE DeviceArray& operator=( DeviceArray&& other )
  {
    DeviceArrayData<T>::operator=( std::move( other ) );
    return *this;
  }

  TRIBOL_HOST_DEVICE T& operator[]( IndexT i ) { return DeviceArrayData<T>::data_[i]; }

  TRIBOL_HOST_DEVICE const T& operator[]( IndexT i ) const { return DeviceArrayData<T>::data_[i]; }

  TRIBOL_HOST_DEVICE IndexT size() const { return DeviceArrayData<T>::size_; }

  TRIBOL_HOST_DEVICE T* data() const { return DeviceArrayData<T>::data_; }
};

/**
 * @brief Simple free store (heap) allocated two-dimensional array that can be
 * created on device
 *
 * @tparam T Datatype stored in array
 */
template <typename T>
class DeviceArray2D : public DeviceArrayData<T> {
 public:
  TRIBOL_HOST_DEVICE DeviceArray2D() : DeviceArrayData<T>(), height_{ 0 }, width_{ 0 } {}
  TRIBOL_HOST_DEVICE DeviceArray2D( IndexT height, IndexT width )
      : DeviceArrayData<T>( width * height ), height_{ height }, width_{ width }
  {
  }
  TRIBOL_DEFAULT_HOST_DEVICE ~DeviceArray2D() = default;

  TRIBOL_HOST_DEVICE DeviceArray2D( const DeviceArray2D& other )
      : DeviceArrayData<T>( other ), height_{ other.height_ }, width_{ other.width_ }
  {
  }
  TRIBOL_HOST_DEVICE DeviceArray2D( DeviceArray2D&& other )
      : DeviceArrayData<T>( std::move( other ) ), height_{ other.height_ }, width_{ other.width_ }
  {
  }

  TRIBOL_HOST_DEVICE DeviceArray2D& operator=( const DeviceArray2D& other )
  {
    DeviceArrayData<T>::operator=( other );
    height_ = other.height_;
    width_ = other.width_;
    return *this;
  }

  TRIBOL_HOST_DEVICE DeviceArray2D& operator=( DeviceArray2D&& other )
  {
    DeviceArrayData<T>::operator=( std::move( other ) );
    height_ = other.height_;
    width_ = other.width_;
    other.width_ = 0;
    other.height_ = 0;
    return *this;
  }

  TRIBOL_HOST_DEVICE T& operator[]( IndexT i ) { return DeviceArrayData<T>::data_[i]; }

  TRIBOL_HOST_DEVICE const T& operator[]( IndexT i ) const { return DeviceArrayData<T>::data_[i]; }

  TRIBOL_HOST_DEVICE T& operator()( IndexT i, IndexT j ) { return DeviceArrayData<T>::data_[i + j * height_]; }

  TRIBOL_HOST_DEVICE const T& operator()( IndexT i, IndexT j ) const
  {
    return DeviceArrayData<T>::data_[i + j * height_];
  }

  TRIBOL_HOST_DEVICE IndexT size() const { return DeviceArrayData<T>::size_; }

  TRIBOL_HOST_DEVICE T* data() const { return DeviceArrayData<T>::data_; }

  TRIBOL_HOST_DEVICE IndexT height() const { return height_; }

  TRIBOL_HOST_DEVICE IndexT width() const { return width_; }

  TRIBOL_HOST_DEVICE void fill( T value )
  {
    for ( int i{ 0 }; i < size(); ++i ) {
      data()[i] = value;
    }
  }

 private:
  IndexT height_;
  IndexT width_;
};

/**
 * @brief Simple automatic storage (stack) allocated array that can be
 * created on device
 */
template <typename T, IndexT N>
class StackArray {
 public:
  TRIBOL_DEFAULT_HOST_DEVICE StackArray() = default;
  TRIBOL_HOST_DEVICE StackArray( IndexT width ) : width_{ width } {}
  TRIBOL_DEFAULT_HOST_DEVICE ~StackArray() = default;

  TRIBOL_DEFAULT_HOST_DEVICE StackArray( const StackArray& other ) = default;
  TRIBOL_DEFAULT_HOST_DEVICE StackArray( StackArray&& other ) = default;

  TRIBOL_DEFAULT_HOST_DEVICE StackArray& operator=( const StackArray& other ) = default;
  TRIBOL_DEFAULT_HOST_DEVICE StackArray& operator=( StackArray&& other ) = default;

  TRIBOL_HOST_DEVICE operator T*() noexcept { return &data_[0]; }
  TRIBOL_HOST_DEVICE operator const T*() const noexcept { return &data_[0]; }

  TRIBOL_HOST_DEVICE T& operator[]( IndexT i ) { return data_[i]; }

  TRIBOL_HOST_DEVICE const T& operator[]( IndexT i ) const { return data_[i]; }

  TRIBOL_HOST_DEVICE T& operator()( IndexT i, IndexT j ) { return data_[i * width_ + j]; }

  TRIBOL_HOST_DEVICE const T& operator()( IndexT i, IndexT j ) const { return data_[i * width_ + j]; }

 private:
  T data_[N];
  IndexT width_;
};

}  // namespace tribol
#endif /* TRIBOL_COMMON_CONTAINERS_HPP_ */
