// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_UTILS_ARRAYUTILITY_HPP_
#define SRC_REDECOMP_UTILS_ARRAYUTILITY_HPP_

#include <numeric>

#include "axom/core.hpp"

namespace redecomp {

/**
 * @brief Convenience class for generating common axom::Arrays
 */
class ArrayUtility {
 public:
  /**
   * @brief Create an index array at compile time
   *
   * @tparam T Array element type
   * @tparam N Size of array
   * @return axom::Array<T> holding index array [0, ..., N-1]
   */
  template <typename T, size_t N>
  static constexpr axom::Array<T> IndexArray()
  {
    return IndexArrayImpl<T>( std::make_index_sequence<N>{} );
  }

  /**
   * @brief Create an index array at run time.
   *
   * @tparam T Array element type
   * @param n Size of array
   * @return axom::Array<T> holding index array [0, ..., n-1]
   */
  template <typename T>
  static axom::Array<T> IndexArray( size_t n )
  {
    auto index = axom::Array<T>( n, n );
    std::iota( index.begin(), index.end(), 0 );
    return index;
  }

 private:
  /**
   * @brief IndexArray implementation
   *
   * @tparam T Array element type
   * @tparam I Indices
   * @return axom::Array<T> holding the index array I
   */
  template <typename T, size_t... I>
  static constexpr axom::Array<T> IndexArrayImpl( std::index_sequence<I...> )
  {
    return axom::Array<T>( { I... } );
  }
};

}  // end namespace redecomp

#endif /* SRC_REDECOMP_UTILS_ARRAYUTILITY_HPP_ */
