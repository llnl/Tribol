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

template <IndexT N>
class FixedSizer {
 public:
  TRIBOL_HOST_DEVICE FixedSizer( [[maybe_unused]] IndexT size ) { assert( size == N ); }
  TRIBOL_HOST_DEVICE constexpr IndexT size() const { return N; }

  using size_type = IndexT;
  TRIBOL_HOST_DEVICE constexpr operator IndexT() const { return size(); }
};

class DynamicSizer {
 public:
  TRIBOL_HOST_DEVICE DynamicSizer( IndexT size ) : size_( size ) {}
  TRIBOL_HOST_DEVICE IndexT size() const { return size_; }

  using size_type = IndexT;
  TRIBOL_HOST_DEVICE operator IndexT() const { return size(); }

 private:
  IndexT size_;
};

template <typename T, class Sizer = DynamicSizer>
class Memory {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using size_type = typename Sizer::size_type;

  TRIBOL_HOST_DEVICE Memory( pointer data, size_type size, size_type stride = 1 )
      : data_( data ), size_( size ), stride_( stride )
  {
  }
  TRIBOL_HOST_DEVICE Memory( const Memory& other )
      : data_( other.data_ ), size_( other.size_ ), stride_( other.stride_ )
  {
  }

  TRIBOL_HOST_DEVICE pointer data() const { return data_; }
  TRIBOL_HOST_DEVICE size_type size() const { return size_; }
  TRIBOL_HOST_DEVICE size_type stride() const { return stride_; }

  TRIBOL_HOST_DEVICE Memory<T, Sizer> view() const { return Memory<T, Sizer>( data_, size_, stride_ ); }

  operator pointer() const { return data_; }

 protected:
  pointer data_;
  Sizer size_;
  size_type stride_;
};

template <typename T, IndexT N>
class StackMemory : public Memory<T, FixedSizer<N>> {
 public:
  using value_type = typename Memory<T, FixedSizer<N>>::value_type;
  using pointer = typename Memory<T, FixedSizer<N>>::pointer;
  using const_pointer = typename Memory<T, FixedSizer<N>>::const_pointer;
  using size_type = typename Memory<T, FixedSizer<N>>::size_type;

  TRIBOL_HOST_DEVICE StackMemory() : Memory<T, FixedSizer<N>>( nullptr, N ) { data_ = stack_data_; }
  TRIBOL_HOST_DEVICE StackMemory( [[maybe_unused]] size_type size ) : StackMemory() { assert( size == N ); }

  using Memory<T, FixedSizer<N>>::data;
  using Memory<T, FixedSizer<N>>::size;
  using Memory<T, FixedSizer<N>>::stride;

  using Memory<T, FixedSizer<N>>::view;

 private:
  using Memory<T, FixedSizer<N>>::data_;
  T stack_data_[N];
};

template <typename T, class Allocator = HeapAllocator<T>, class Sizer = DynamicSizer>
class AllocatedMemory : public Memory<T, Sizer> {
 public:
  using value_type = typename Memory<T, Sizer>::value_type;
  using pointer = typename Memory<T, Sizer>::pointer;
  using const_pointer = typename Memory<T, Sizer>::const_pointer;
  using size_type = typename Memory<T, Sizer>::size_type;

  TRIBOL_HOST_DEVICE AllocatedMemory( size_type size, Allocator allocator = Allocator() )
      : Memory<T, Sizer>( allocator.allocate( size ), size ), allocator_( std::move( allocator ) )
  {
  }

  // // Constructor from another memory space
  // template <MemorySpace MSPACE2>
  // AllocatedMemory( const AllocatedMemory<T, Allocator>& other ) : AllocatedMemory( other.size() )
  // {
  //   allocator_.copy( data_, other.data(), other.size() );
  // }

  ~AllocatedMemory() { allocator_.deallocate( data_, size_ ); }

  // Copy constructor
  AllocatedMemory( const AllocatedMemory& other ) : AllocatedMemory( other.size_ )
  {
    allocator_.copy( data_, other.data_, other.size_ );
  }

  // Move constructor
  AllocatedMemory( AllocatedMemory&& other )
      : Memory<T, Sizer>( other.data_, other.size_ ), allocator_{ other.allocator_ }
  {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  // Copy assignment operator
  AllocatedMemory& operator=( const AllocatedMemory& other )
  {
    if ( this != &other ) {
      if ( data_ != nullptr ) {
        allocator_.deallocate( data_, size_ );
      }
      data_ = allocator_.allocate( other.size() );
      size_ = other.size();
      allocator_ = other.allocator();
    }
    allocator_.copy( data_, other.data(), other.size() );
    return *this;
  }

  // template <MemorySpace MSPACE2>
  // AllocatedMemory& operator=( const AllocatedMemory<T, MSPACE2, Allocator>& other )
  // {
  //   if ( this != &other ) {
  //     if ( data_ != nullptr ) {
  //       allocator_.deallocate( data_, size_ );
  //     }
  //     data_ = allocator_.allocate( other.size() );
  //     size_ = other.size();
  //     allocator_ = other.allocator();
  //   }
  //   allocator_.copy( data_, other.data(), other.size() );
  //   return *this;
  // }

  // Move assignment operator
  AllocatedMemory& operator=( AllocatedMemory&& other )
  {
    if ( this != &other ) {
      if ( data_ != nullptr ) {
        allocator_.deallocate( data_, size_ );
      }
      data_ = other.data_;
      size_ = other.size_;
      allocator_ = other.allocator_;

      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  using Memory<T, Sizer>::data;
  using Memory<T, Sizer>::size;
  using Memory<T, Sizer>::stride;
  const Allocator& allocator() const { return allocator_; }

  using Memory<T, Sizer>::view;

 private:
  using Memory<T, Sizer>::data_;
  using Memory<T, Sizer>::size_;
  using Memory<T, Sizer>::stride_;
  Allocator allocator_;
};

// template <typename T>
// class HeapAllocation {
//  public:
//   using value_type = T;
//   using pointer = T*;
//   using const_pointer = const T*;
//   using size_type = IndexT;

//   TRIBOL_HOST_DEVICE HeapAllocation( size_type size ) : data_( new T[size] ), size_( size ) {}
//   TRIBOL_HOST_DEVICE ~HeapAllocation() { delete[] data_; }

//   TRIBOL_HOST_DEVICE HeapAllocation( const HeapAllocation& other ) : data_( new T[other.size()] ), size_( other.size
//   )
//   {
//     // deep copy data
//     for ( size_type i{ 0 }; i < size_; ++i ) {
//       data_[i] = other.data_[i];
//     }
//   }
//   TRIBOL_HOST_DEVICE HeapAllocation( HeapAllocation&& other ) : data_( other.data_ )
//   {
//     // reset other
//     other.data_ = nullptr;
//   }
//   TRIBOL_HOST_DEVICE HeapAllocation& operator=( const HeapAllocation& other )
//   {
//     if ( this != &other ) {
//       delete[] data_;
//       data_ = new T[other.size()];
//       size_ = other.size();
//       // deep copy data
//       for ( size_type i{ 0 }; i < size_; ++i ) {
//         data_[i] = other.data_[i];
//       }
//     }
//     return *this;
//   }
//   TRIBOL_HOST_DEVICE HeapAllocation& operator=( HeapAllocation&& other )
//   {
//     if ( this != &other ) {
//       delete[] data_;
//       data_ = other.data_;
//       // reset other
//       other.data_ = nullptr;
//     }
//     return *this;
//   }

//   TRIBOL_HOST_DEVICE pointer data() { return data_; }
//   TRIBOL_HOST_DEVICE const_pointer data() const { return data_; }
//   TRIBOL_HOST_DEVICE size_type size() const { return N; }
//   TRIBOL_HOST_DEVICE size_type stride() const { return 1; }

//  private:
//   T* data_;
//   size_type size_;
// };

// template <typename T, IndexT N>
// class FixedHeapAllocation : public HeapAllocation<T> {
//  public:
//   using value_type = T;
//   using pointer = T*;
//   using const_pointer = const T*;
//   using size_type = IndexT;

//   TRIBOL_HOST_DEVICE HeapAllocation() : data_( new T[N] ) {}
//   TRIBOL_HOST_DEVICE ~HeapAllocation() { delete[] data_; }

//   TRIBOL_HOST_DEVICE HeapAllocation( const HeapAllocation& other ) : data_( new T[N] )
//   {
//     // deep copy data
//     for ( size_type i{ 0 }; i < N; ++i ) {
//       data_[i] = other.data_[i];
//     }
//   }
//   TRIBOL_HOST_DEVICE HeapAllocation( HeapAllocation&& other ) : data_( other.data_ )
//   {
//     // reset other
//     other.data_ = nullptr;
//   }
//   TRIBOL_HOST_DEVICE HeapAllocation& operator=( const HeapAllocation& other )
//   {
//     if ( this != &other ) {
//       delete[] data_;
//       data_ = new T[N];
//       // deep copy data
//       for ( size_type i{ 0 }; i < N; ++i ) {
//         data_[i] = other.data_[i];
//       }
//     }
//     return *this;
//   }
//   TRIBOL_HOST_DEVICE HeapAllocation& operator=( HeapAllocation&& other )
//   {
//     if ( this != &other ) {
//       delete[] data_;
//       data_ = other.data_;
//       // reset other
//       other.data_ = nullptr;
//     }
//     return *this;
//   }

//   TRIBOL_HOST_DEVICE pointer data() { return data_; }
//   TRIBOL_HOST_DEVICE const_pointer data() const { return data_; }
//   TRIBOL_HOST_DEVICE size_type size() const { return N; }
//   TRIBOL_HOST_DEVICE size_type stride() const { return 1; }

//  private:
//   T* data_;
// };

// template <typename T, IndexT N>
// class SubAllocation {
//  public:
//   using value_type = T;
//   using pointer = T*;
//   using const_pointer = const T*;
//   using size_type = IndexT;

//   TRIBOL_HOST_DEVICE SubAllocation( const Memory<T>& parent, size_type offset, size_type stride = 1 )
//       : parent_( parent.data() ), offset_( offset ), stride_( stride )
//   {
//     // TODO: check if offset + N * stride < parent.size()
//     // all memory in an allocation should be initialized to the default value
//     for ( size_type i{ 0 }; i < N; ++i ) {
//       parent_[offset_ + i * stride_] = T{};
//     }
//   }
//   TRIBOL_HOST_DEVICE ~SubAllocation()
//   {
//     // everything in the parent should be destroyed when the parent is destroyed
//     for ( size_type i{ 0 }; i < N; ++i ) {
//       parent_[offset_ + i * stride_].~T();
//     }
//   }
//   // Copying not allowed with sub-allocation
//   TRIBOL_HOST_DEVICE SubAllocation( const SubAllocation& other ) = delete;
//   TRIBOL_HOST_DEVICE SubAllocation( SubAllocation&& other ) : parent_( other.parent_ ), offset_( other.offset_ )
//   {
//     // reset other
//     other.parent_ = nullptr;
//     other.offset_ = 0;
//   }
//   // Copy assignment not allowed with sub-allocation
//   TRIBOL_HOST_DEVICE SubAllocation& operator=( const SubAllocation& other ) = delete;
//   TRIBOL_HOST_DEVICE SubAllocation& operator=( SubAllocation&& other )
//   {
//     parent_ = other.parent_;
//     offset_ = other.offset_;
//     // reset other
//     other.parent_ = nullptr;
//     other.offset_ = 0;
//     return *this;
//   }

//   TRIBOL_HOST_DEVICE pointer data() { return parent_ + offset_; }
//   TRIBOL_HOST_DEVICE const_pointer data() const { return parent_ + offset_; }
//   TRIBOL_HOST_DEVICE size_type size() const { return N; }
//   TRIBOL_HOST_DEVICE size_type stride() const { return stride_; }

//  private:
//   T* parent_;
//   size_type offset_;
//   size_type stride_;
// };

template <typename T, typename Memory>
class ArrayBase {
 public:
  using value_type = T;
  using memory_type = Memory;
  using pointer = typename Memory::pointer;
  using const_pointer = typename Memory::const_pointer;
  using size_type = typename Memory::size_type;

  TRIBOL_HOST_DEVICE ArrayBase( Memory&& memory ) : memory_( std::move( memory ) )
  {
    // initialize memory
    for ( size_type i{ 0 }; i < memory_.size(); ++i ) {
      memory_.data()[i] = T{};
    }
  }
  TRIBOL_HOST_DEVICE ~ArrayBase()
  {
    // call destructor on all elements
    for ( size_type i{ 0 }; i < memory_.size(); ++i ) {
      memory_.data()[i].~T();
    }
  }
  TRIBOL_HOST_DEVICE ArrayBase( const ArrayBase& other ) : memory_( other.memory_ ) {}
  TRIBOL_HOST_DEVICE ArrayBase( ArrayBase&& other ) : memory_( std::move( other.memory_ ) ) {}
  TRIBOL_HOST_DEVICE ArrayBase& operator=( const ArrayBase& other )
  {
    if ( this != &other ) {
      memory_ = other.memory_;
    }
    return *this;
  }
  TRIBOL_HOST_DEVICE ArrayBase& operator=( ArrayBase&& other )
  {
    if ( this != &other ) {
      memory_ = std::move( other.memory_ );
    }
    return *this;
  }

  TRIBOL_HOST_DEVICE pointer data() { return memory_.data(); }
  TRIBOL_HOST_DEVICE const_pointer data() const { return memory_.data(); }

 protected:
  Memory memory_;
};

template <typename T, IndexT N, class Memory = StackMemory<T, N>>
class FixedArray : public ArrayBase<T, Memory> {
 public:
  using value_type = T;
  using BaseClass = ArrayBase<T, Memory>;
  using memory_type = typename BaseClass::memory_type;
  using pointer = typename BaseClass::pointer;
  using const_pointer = typename BaseClass::const_pointer;
  using size_type = typename BaseClass::size_type;

  TRIBOL_HOST_DEVICE FixedArray( Memory&& memory = Memory( N ) ) : BaseClass( std::move( memory ) ) {}

  TRIBOL_HOST_DEVICE size_type size() const { return memory_.size(); }

  TRIBOL_HOST_DEVICE T& operator[]( size_type i ) { return *( memory_.data() + i * memory_.stride() ); }
  TRIBOL_HOST_DEVICE const T& operator[]( size_type i ) const { return *( memory_.data() + i * memory_.stride() ); }

  TRIBOL_HOST_DEVICE pointer begin() { return memory_.data(); }
  TRIBOL_HOST_DEVICE const_pointer begin() const { return memory_.data(); }

  TRIBOL_HOST_DEVICE pointer end() { return memory_.data() + memory_.size(); }
  TRIBOL_HOST_DEVICE const_pointer end() const { return memory_.data() + memory_.size(); }

 private:
  using BaseClass::memory_;
};

template <typename T, class Memory = AllocatedMemory<T, HeapAllocator<T>, DynamicSizer>>
class BoundedArray : public ArrayBase<T, Memory> {
 public:
  using value_type = T;
  using BaseClass = ArrayBase<T, Memory>;
  using memory_type = typename BaseClass::memory_type;
  using pointer = typename BaseClass::pointer;
  using const_pointer = typename BaseClass::const_pointer;
  using size_type = typename BaseClass::size_type;

  TRIBOL_HOST_DEVICE BoundedArray( size_type size, size_type capacity ) : BaseClass( Memory( capacity ) ), size_( size )
  {
    assert( capacity >= size );
  }
  TRIBOL_HOST_DEVICE BoundedArray( size_type size, Memory&& memory ) : BaseClass( std::move( memory ) ), size_( size )
  {
    assert( memory_.size() >= size );
  }

  TRIBOL_HOST_DEVICE size_type size() const { return size_; }
  TRIBOL_HOST_DEVICE size_type capacity() const { return memory_.size(); }

  TRIBOL_HOST_DEVICE void push_back( T value )
  {
    assert( size_ < memory_.size() );
    memory_.data()[size_] = value;
    ++size_;
  }
  template <typename... Args>
  TRIBOL_HOST_DEVICE void emplace_back( Args&&... args )
  {
    assert( size_ < memory_.size() );
    memory_.data()[size_] = T( std::forward<Args>( args )... );
    ++size_;
  }
  TRIBOL_HOST_DEVICE void pop_back() { --size_; }
  TRIBOL_HOST_DEVICE void resize( size_type size )
  {
    assert( size <= memory_.size() );
    for ( size_type i{ size_ }; i < size; ++i ) {
      memory_.data()[i] = T{};
    }
    size_ = size;
  }

  TRIBOL_HOST_DEVICE T& operator[]( size_type i ) { return *( memory_.data() + i ); }
  TRIBOL_HOST_DEVICE const T& operator[]( size_type i ) const { return *( memory_.data() + i ); }

  TRIBOL_HOST_DEVICE pointer begin() { return memory_.data(); }
  TRIBOL_HOST_DEVICE const_pointer begin() const { return memory_.data(); }

  TRIBOL_HOST_DEVICE pointer end() { return memory_.data() + size_; }
  TRIBOL_HOST_DEVICE const_pointer end() const { return memory_.data() + size_; }

 private:
  using BaseClass::memory_;
  size_type size_;
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
