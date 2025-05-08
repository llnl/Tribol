// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef TRIBOL_COMMON_CONTAINERS_HPP_
#define TRIBOL_COMMON_CONTAINERS_HPP_

#include <utility>

// Tribol includes
#include "tribol/config.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Memory.hpp"

namespace tribol {

template <typename T, IndexT N>
class StackAllocation {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using size_type = IndexT;

  TRIBOL_HOST_DEVICE pointer data() { return data_; }
  TRIBOL_HOST_DEVICE const_pointer data() const { return data_; }
  TRIBOL_HOST_DEVICE size_type size() const { return N; }

 private:
  T data_[N];
};

template <typename T, IndexT N>
class HeapAllocation {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using size_type = IndexT;

  TRIBOL_HOST_DEVICE HeapAllocation() : data_( new T[N] ) {}
  TRIBOL_HOST_DEVICE ~HeapAllocation() { delete[] data_; }

  TRIBOL_HOST_DEVICE HeapAllocation( const HeapAllocation& other ) : data_( new T[N] )
  {
    // deep copy data
    for ( size_type i{ 0 }; i < N; ++i ) {
      data_[i] = other.data_[i];
    }
  }
  TRIBOL_HOST_DEVICE HeapAllocation( HeapAllocation&& other ) : data_( other.data_ )
  {
    // reset other
    other.data_ = nullptr;
  }
  TRIBOL_HOST_DEVICE HeapAllocation& operator=( const HeapAllocation& other )
  {
    if ( this != &other ) {
      delete[] data_;
      data_ = new T[N];
      // deep copy data
      for ( size_type i{ 0 }; i < N; ++i ) {
        data_[i] = other.data_[i];
      }
    }
    return *this;
  }
  TRIBOL_HOST_DEVICE HeapAllocation& operator=( HeapAllocation&& other )
  {
    if ( this != &other ) {
      delete[] data_;
      data_ = other.data_;
      // reset other
      other.data_ = nullptr;
    }
    return *this;
  }

  TRIBOL_HOST_DEVICE pointer data() { return data_; }
  TRIBOL_HOST_DEVICE const_pointer data() const { return data_; }
  TRIBOL_HOST_DEVICE size_type size() const { return N; }

 private:
  T* data_;
};

template <typename T, IndexT N>
class PoolAllocation {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using size_type = IndexT;

  TRIBOL_HOST_DEVICE PoolAllocation( const Memory<T>& pool, size_type offset ) : pool_( pool.data() ), offset_( offset )
  {
    // TODO: check if offset + N < pool.size()
    // all memory in an allocation should be initialized to the default value
    for ( size_type i{ 0 }; i < N; ++i ) {
      pool_[offset_ + i] = T{};
    }
  }
  TRIBOL_HOST_DEVICE ~PoolAllocation()
  {
    // everything in the pool should be destroyed when the pool is destroyed
    for ( size_type i{ 0 }; i < N; ++i ) {
      pool_[offset_ + i].~T();
    }
  }
  // Copying not allowed with pool allocation
  TRIBOL_HOST_DEVICE PoolAllocation( const PoolAllocation& other ) = delete;
  TRIBOL_HOST_DEVICE PoolAllocation( PoolAllocation&& other ) : pool_( other.pool_ ), offset_( other.offset_ )
  {
    // reset other
    other.pool_ = nullptr;
    other.offset_ = 0;
  }
  // Copy assignment not allowed with pool allocation
  TRIBOL_HOST_DEVICE PoolAllocation& operator=( const PoolAllocation& other ) = delete;
  TRIBOL_HOST_DEVICE PoolAllocation& operator=( PoolAllocation&& other )
  {
    pool_ = other.pool_;
    offset_ = other.offset_;
    // reset other
    other.pool_ = nullptr;
    other.offset_ = 0;
    return *this;
  }

  TRIBOL_HOST_DEVICE pointer data() { return pool_ + offset_; }
  TRIBOL_HOST_DEVICE const_pointer data() const { return pool_ + offset_; }
  TRIBOL_HOST_DEVICE size_type size() const { return N; }

 private:
  T* pool_;
  size_type offset_;
};

template <typename T, IndexT N, typename Allocation>
class ArrayBase {
 public:
  using value_type = T;
  using allocation_type = Allocation;
  using pointer = typename Allocation::pointer;
  using const_pointer = typename Allocation::const_pointer;
  using size_type = typename Allocation::size_type;

  TRIBOL_HOST_DEVICE ArrayBase() = default;
  TRIBOL_HOST_DEVICE ArrayBase( Allocation&& allocation ) : allocation_( std::move( allocation ) ) {}

  TRIBOL_HOST_DEVICE pointer data() { return allocation_.data(); }
  TRIBOL_HOST_DEVICE const_pointer data() const { return allocation_.data(); }

 protected:
  Allocation allocation_;
};

template <typename T, IndexT N, typename Allocation = StackAllocation<T, N>>
class FixedArray : public ArrayBase<T, N, Allocation> {
 public:
  using value_type = T;
  using Base = ArrayBase<T, N, Allocation>;
  using allocation_type = typename Base::allocation_type;
  using pointer = typename Base::pointer;
  using const_pointer = typename Base::const_pointer;
  using size_type = typename Base::size_type;
  using Base::allocation_;

  TRIBOL_HOST_DEVICE FixedArray() = default;
  TRIBOL_HOST_DEVICE FixedArray( Allocation&& allocation ) : Base( std::move( allocation ) ) {}

  TRIBOL_HOST_DEVICE size_type size() const { return allocation_.size(); }

  TRIBOL_HOST_DEVICE pointer operator[]( size_type i ) { return allocation_.data() + i; }
  TRIBOL_HOST_DEVICE const_pointer operator[]( size_type i ) const { return allocation_.data() + i; }

  TRIBOL_HOST_DEVICE pointer begin() { return allocation_.data(); }
  TRIBOL_HOST_DEVICE const_pointer begin() const { return allocation_.data(); }

  TRIBOL_HOST_DEVICE pointer end() { return allocation_.data() + allocation_.size(); }
  TRIBOL_HOST_DEVICE const_pointer end() const { return allocation_.data() + allocation_.size(); }
};

template <typename T, IndexT N, template <typename, IndexT> class Allocation = StackAllocation>
class BoundedArray : public ArrayBase<T, N, Allocation<T, N>> {
 public:
  using value_type = T;
  using Base = ArrayBase<T, N, Allocation<T, N>>;
  using allocation_type = typename Base::allocation_type;
  using pointer = typename Base::pointer;
  using const_pointer = typename Base::const_pointer;
  using size_type = typename Base::size_type;
  using Base::allocation_;

  TRIBOL_HOST_DEVICE BoundedArray() = default;
  TRIBOL_HOST_DEVICE BoundedArray( size_type size ) : size_( size )
  {
    // TODO: check if size <= N
  }
  TRIBOL_HOST_DEVICE BoundedArray( Allocation<T, N>&& allocation ) : Base( std::move( allocation ) ) {}
  TRIBOL_HOST_DEVICE BoundedArray( size_type size, Allocation<T, N>&& allocation )
      : BoundedArray( std::move( allocation ) ), size_( size )
  {
  }

  TRIBOL_HOST_DEVICE size_type size() const { return size_; }
  TRIBOL_HOST_DEVICE size_type capacity() const { return allocation_.size(); }

  TRIBOL_HOST_DEVICE void push_back( T value )
  {
    // TODO: check if size_ < N
    allocation_.data()[size_] = value;
    ++size_;
  }
  template <typename... Args>
  TRIBOL_HOST_DEVICE void emplace_back( Args&&... args )
  {
    // TODO: check if size_ < N
    allocation_.data()[size_] = T( std::forward<Args>( args )... );
    ++size_;
  }
  TRIBOL_HOST_DEVICE void pop_back() { --size_; }
  TRIBOL_HOST_DEVICE void resize( size_type size )
  {
    // TODO: check if size <= N
    for ( size_type i{ size_ }; i < size; ++i ) {
      allocation_.data()[i] = T{};
    }
    size_ = size;
  }

  TRIBOL_HOST_DEVICE pointer operator[]( size_type i ) { return allocation_.data() + i; }
  TRIBOL_HOST_DEVICE const_pointer operator[]( size_type i ) const { return allocation_.data() + i; }

  TRIBOL_HOST_DEVICE pointer begin() { return allocation_.data(); }
  TRIBOL_HOST_DEVICE const_pointer begin() const { return allocation_.data(); }

  TRIBOL_HOST_DEVICE pointer end() { return allocation_.data() + size_; }
  TRIBOL_HOST_DEVICE const_pointer end() const { return allocation_.data() + size_; }

 private:
  size_type size_;
};

template <typename T>
class HeapAllocator {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using size_type = IndexT;

  TRIBOL_HOST_DEVICE pointer allocate( size_type n ) { return new T[n]; }
  TRIBOL_HOST_DEVICE void deallocate( pointer p ) { delete[] p; }
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
  TRIBOL_HOST_DEVICE ~DeviceArray() = default;

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
  TRIBOL_HOST_DEVICE ~DeviceArray2D() = default;

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
  TRIBOL_HOST_DEVICE StackArray() = default;
  TRIBOL_HOST_DEVICE StackArray( IndexT width ) : width_{ width } {}
  TRIBOL_HOST_DEVICE ~StackArray() = default;

  TRIBOL_HOST_DEVICE StackArray( const StackArray& other ) = default;
  TRIBOL_HOST_DEVICE StackArray( StackArray&& other ) = default;

  TRIBOL_HOST_DEVICE StackArray& operator=( const StackArray& other ) = default;
  TRIBOL_HOST_DEVICE StackArray& operator=( StackArray&& other ) = default;

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
