// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef TRIBOL_COMMON_CONTAINERS_HPP_
#define TRIBOL_COMMON_CONTAINERS_HPP_

// Tribol includes
#include "tribol/common/BasicTypes.hpp"

namespace tribol
{

/**
 * @brief Storage base class for device-compatible free store (heap) allocated
 * arrays
 * 
 * @tparam T Datatype stored in array
 */
template <typename T>
class DeviceArrayData
{
protected:
  TRIBOL_HOST_DEVICE DeviceArrayData() 
  : size_ {0},
    data_ {nullptr} 
  {}
  TRIBOL_HOST_DEVICE DeviceArrayData(IndexT size)
  : size_ {size},
    data_ {new T[size]}
  {}
  TRIBOL_HOST_DEVICE virtual ~DeviceArrayData()
  {
    deleteData();
  }

  TRIBOL_HOST_DEVICE DeviceArrayData(const DeviceArrayData& other)
  : DeviceArrayData(other.size_)
  {
    // deep copy data
    for (IndexT i{0}; i < size_; ++i)
    {
      data_[i] = other.data_[i];
    }
  }
  TRIBOL_HOST_DEVICE DeviceArrayData(DeviceArrayData&& other)
  : size_ {other.size_},
    data_ {other.data_}
  {
    // reset other
    other.size_ = 0;
    other.data_ = nullptr;
  }

  TRIBOL_HOST_DEVICE DeviceArrayData& operator=(const DeviceArrayData& other)
  {
    deleteData();
    size_ = other.size_;
    data_ = new T[size_];
    // deep copy data
    for (IndexT i{0}; i < size_; ++i)
    {
      data_[i] = other.data_[i];
    }
    return *this;
  }

  TRIBOL_HOST_DEVICE DeviceArrayData& operator=(DeviceArrayData&& other)
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
    if (data_ != nullptr)
    {
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
class DeviceArray : public DeviceArrayData<T>
{
public:
  TRIBOL_HOST_DEVICE DeviceArray()
  : DeviceArrayData<T>()
  {}
  TRIBOL_HOST_DEVICE DeviceArray(IndexT size)
  : DeviceArrayData<T>(size)
  {}
  TRIBOL_HOST_DEVICE ~DeviceArray() = default;

  TRIBOL_HOST_DEVICE DeviceArray(const DeviceArray& other)
  : DeviceArrayData<T>(other)
  {}
  TRIBOL_HOST_DEVICE DeviceArray(DeviceArray&& other)
  : DeviceArrayData<T>(std::move(other))
  {}

  TRIBOL_HOST_DEVICE DeviceArray& operator=(const DeviceArray& other)
  {
    DeviceArrayData<T>::operator=(other);
    return *this;
  }

  TRIBOL_HOST_DEVICE DeviceArray& operator=(DeviceArray&& other)
  {
    DeviceArrayData<T>::operator=(std::move(other));
    return *this;
  }

  TRIBOL_HOST_DEVICE T& operator[](IndexT i)
  {
    return DeviceArrayData<T>::data_[i];
  }

  TRIBOL_HOST_DEVICE const T& operator[](IndexT i) const
  {
    return DeviceArrayData<T>::data_[i];
  }

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
class DeviceArray2D : public DeviceArrayData<T>
{
public:
  TRIBOL_HOST_DEVICE DeviceArray2D()
  : DeviceArrayData<T>(),
    height_ {0},
    width_ {0}
  {}
  TRIBOL_HOST_DEVICE DeviceArray2D(IndexT height, IndexT width)
  : DeviceArrayData<T>(width * height),
    height_ {height},
    width_ {width}
  {}
  TRIBOL_HOST_DEVICE ~DeviceArray2D() = default;

  TRIBOL_HOST_DEVICE DeviceArray2D(const DeviceArray2D& other)
  : DeviceArrayData<T>(other),
    height_ {other.height_},
    width_ {other.width_}
  {}
  TRIBOL_HOST_DEVICE DeviceArray2D(DeviceArray2D&& other)
  : DeviceArrayData<T>(std::move(other)),
    height_ {other.height_},
    width_ {other.width_}
  {}

  TRIBOL_HOST_DEVICE DeviceArray2D& operator=(const DeviceArray2D& other)
  {
    DeviceArrayData<T>::operator=(other);
    height_ = other.height_;
    width_ = other.width_;
    return *this;
  }

  TRIBOL_HOST_DEVICE DeviceArray2D& operator=(DeviceArray2D&& other)
  {
    DeviceArrayData<T>::operator=(std::move(other));
    height_ = other.height_;
    width_ = other.width_;
    other.width_ = 0;
    other.height_ = 0;
    return *this;
  }

  TRIBOL_HOST_DEVICE T& operator[](IndexT i)
  {
    return DeviceArrayData<T>::data_[i];
  }

  TRIBOL_HOST_DEVICE const T& operator[](IndexT i) const
  {
    return DeviceArrayData<T>::data_[i];
  }

  TRIBOL_HOST_DEVICE T& operator()(IndexT i, IndexT j)
  {
    return DeviceArrayData<T>::data_[i + j * height_];
  }

  TRIBOL_HOST_DEVICE const T& operator()(IndexT i, IndexT j) const
  {
    return DeviceArrayData<T>::data_[i + j * height_];
  }

  TRIBOL_HOST_DEVICE IndexT size() const { return DeviceArrayData<T>::size_; }

  TRIBOL_HOST_DEVICE T* data() const { return DeviceArrayData<T>::data_; }
  
  TRIBOL_HOST_DEVICE IndexT height() const { return height_; }
  
  TRIBOL_HOST_DEVICE IndexT width() const { return width_; }

  TRIBOL_HOST_DEVICE void fill(T value)
  {
    for (int i{0}; i < size(); ++i)
    {
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
class StackArray
{
public:
  TRIBOL_HOST_DEVICE StackArray() = default;
  TRIBOL_HOST_DEVICE StackArray(IndexT width)
  : width_ {width}
  {}
  TRIBOL_HOST_DEVICE ~StackArray() = default;

  TRIBOL_HOST_DEVICE StackArray(const StackArray& other) = default;
  TRIBOL_HOST_DEVICE StackArray(StackArray&& other) = default;

  TRIBOL_HOST_DEVICE StackArray& operator=(const StackArray& other) = default;
  TRIBOL_HOST_DEVICE StackArray& operator=(StackArray&& other) = default;

  TRIBOL_HOST_DEVICE operator T*() noexcept { return &data_[0]; }
  TRIBOL_HOST_DEVICE operator const T*() const noexcept { return &data_[0]; }

  TRIBOL_HOST_DEVICE T& operator[](IndexT i)
  {
    return data_[i];
  }

  TRIBOL_HOST_DEVICE const T& operator[](IndexT i) const
  {
    return data_[i];
  }

  TRIBOL_HOST_DEVICE T& operator()(IndexT i, IndexT j)
  {
    return data_[i * width_ + j];
  }

  TRIBOL_HOST_DEVICE const T& operator()(IndexT i, IndexT j) const
  {
    return data_[i * width_ + j];
  }
private:
  T data_[N];
  IndexT width_;
};

}
#endif /* TRIBOL_COMMON_CONTAINERS_HPP_ */
