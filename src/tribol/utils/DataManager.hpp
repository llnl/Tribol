// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_UTILS_DATAMANAGER_HPP_
#define SRC_UTILS_DATAMANAGER_HPP_

// C/C++ includes
#include <unordered_map>

// Tribol includes
#include "tribol/common/BasicTypes.hpp"

// Axom includes
#include "axom/fmt.hpp"
#include "axom/slic.hpp"

namespace tribol
{

template <typename T>
class DataManager
{
public:
  DataManager(const DataManager& other) = delete;
  DataManager(DataManager&& other) = delete;
  void operator=(const DataManager& other) = delete;
  void operator=(DataManager&& other) = delete;

  /**
   * @brief Return the instance of the DataManager singleton
   * 
   * @return Reference to the DataManager
   */
  static DataManager& getInstance()
  {
    static DataManager instance_;
    return instance_;
  }

  /**
   * @brief Returns the element at id
   * 
   * @note Throws std::out_of_range if the element doesn't exist
   * 
   * @param id Integer identifier
   * @return Reference to the element
   */
  T& at(IndexT id)
  {
    return data_map_.at(id);
  }

  /**
   * @brief Returns an iterator to the first element
   * 
   * @return std::unordered_map<IndexT, T>::iterator 
   */
  typename std::unordered_map<IndexT, T>::iterator begin() noexcept
  {
    return data_map_.begin();
  }

  /**
   * @brief Returns an iterator to the element following the last element
   * 
   * @return std::unordered_map<IndexT, T>::iterator 
   */
  typename std::unordered_map<IndexT, T>::iterator end() noexcept
  {
    return data_map_.end();
  }

  /**
   * @brief Removes the specified element from the container
   * 
   * @param id Integer identifier of the element to remove
   * @return Number of elements removed (0 or 1)
   */
  size_t erase(IndexT id)
  {
    return data_map_.erase(id);
  }

  /**
   * @brief Erases all elements from the container
   */
  void clear() noexcept
  {
    data_map_.clear();
  }

  /**
   * @brief Returns the number of elements in the container
   * 
   * @return Number of elements in the container
   */
  size_t size() noexcept
  {
    return data_map_.size();
  }

  /**
   * @brief Adds the key/value pair given by id and data
   * 
   * @param id Integer identifier for element
   * @param data Element to add
   * @return Reference to the element
   */
  T& addData(IndexT id, T&& data)
  {
    data_map_.erase(id);
    auto data_it = data_map_.emplace(id, std::move(data));
    return data_it.first->second;
  }

  /**
   * @brief Returns a pointer to the element 
   * 
   * @param id Integer identifier for element
   * @return Pointer to element if found, nullptr otherwise
   */
  T* findData(IndexT id)
  {
    auto data_it = data_map_.find(id);
    if (data_it == data_map_.end())
    {
      return nullptr;
    } 
    else
    {
      return &data_it->second;
    }
  }

  /**
   * @brief Returns a reference to the element
   * 
   * @note Calls SLIC_ERROR_IF macro if element doesn't exist
   * 
   * @param id Integer identifier for element
   * @return Reference to element
   */
  T& getData(IndexT id)
  {
    auto data_it = data_map_.find(id);
    SLIC_ERROR_IF(data_it == data_map_.end(),
      axom::fmt::format("No data exists for id = {}.", id));
    return data_it->second;
  }

private:
  DataManager() = default;
  ~DataManager() = default;

  /**
   * @brief Map holding elements with an integer key
   */
  std::unordered_map<IndexT, T> data_map_;
};

} // end namespace tribol

#endif /* SRC_UTILS_DATAMANAGER_HPP_ */
