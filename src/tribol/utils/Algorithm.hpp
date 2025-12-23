// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_UTILS_ALGORITHM_HPP_
#define SRC_TRIBOL_UTILS_ALGORITHM_HPP_

#include "tribol/common/ArrayTypes.hpp"

#include <utility>
#include <limits>

namespace tribol {

namespace algorithm {

/**
 * @brief Implements a generic binary search algorithm
 *
 * @tparam LCOMP Function taking an IndexT input argument
 * @tparam HCOMP Function taking an IndexT input argument
 * @param size Number of elements to search through
 * @param lo_comparison Test if an element is too small
 * @param hi_comparison Test if an element is too large
 * @return Integer index of matching element
 */
template <typename LCOMP, typename HCOMP>
TRIBOL_HOST_DEVICE IndexT binarySearch( IndexT size, LCOMP&& lo_comparison, HCOMP&& hi_comparison )
{
  if ( size == 0 ) {
#ifdef TRIBOL_USE_HOST
    SLIC_DEBUG( "binarySearch: empty array given" );
#endif
    return -1;
  }

  IndexT l = 0;
  IndexT r = size - 1;
  while ( l <= r ) {
    IndexT m = ( l + r ) / 2;
    if ( lo_comparison( m ) ) {
      l = m + 1;
    } else if ( hi_comparison( m ) ) {
      r = m - 1;
    } else {
      return m;
    }
  }

#ifdef TRIBOL_USE_HOST
  SLIC_DEBUG( "binary_search: could not locate value in provided array." );
#endif
  return -1;
}

/**
 * @brief Binary search for value within ranges of values
 *
 * @tparam T Data type to compare; must have operator+ and comparators
 * @param array C-style array with each element giving the min of a range
 * @param range C-style array with each element giving the size of each range
 * @param size Number of elements in the array and range arrays
 * @param value Value to search for
 * @return Integer index of matching element range
 */
template <typename T>
TRIBOL_HOST_DEVICE IndexT binarySearch( const T* array, const T* range, IndexT size, T value )
{
  return binarySearch(
      size, [=] TRIBOL_HOST_DEVICE( IndexT i ) { return array[i] + range[i] <= value; },
      [=] TRIBOL_HOST_DEVICE( IndexT i ) { return array[i] > value; } );
}

/**
 * @brief Binary search for value within ranges of values
 *
 * @tparam ARRAY Array type holding elements of type T; must have data() and
 * size() implemented
 * @tparam T Data type to compare
 * @param array Array with each element giving the min of a range
 * @param range Array wtih each element giving the size of each range
 * @param value Value to search for
 * @return Integer index of matching element range
 */
template <typename ARRAY, typename T>
TRIBOL_HOST_DEVICE IndexT binarySearch( const ARRAY& array, const ARRAY& range, T value )
{
  return binarySearch( array.data(), range.data(), array.size(), value );
}

/**
 * @brief Given entries in the upper triangular portion of a symmetric matrix
 * stored row-major, find the row given an entry id
 *
 * @param value Entry in the 1D vector
 * @param matrix_width Number of columns in the matrix
 * @return Row of the given entry
 */
TRIBOL_HOST_DEVICE inline IndexT symmMatrixRow( IndexT value, IndexT matrix_width )
{
  return binarySearch(
      matrix_width, [=] TRIBOL_HOST_DEVICE( IndexT i ) { return ( i + 1 ) * ( i + 2 ) / 2 <= value; },
      [=] TRIBOL_HOST_DEVICE( IndexT i ) { return i * ( i + 1 ) / 2 > value; } );
}

/**
 * @brief Transposes a matrix stored as a 2D array
 *
 * @tparam MSPACE Memory space of the array
 * @tparam T Type of the array data
 * @param in Matrix to transpose
 * @param out Transposed matrix
 */
template <MemorySpace MSPACE, typename T>
TRIBOL_HOST_DEVICE void transpose( const ArrayT<T, 2, MSPACE>& in, ArrayT<T, 2, MSPACE>& out )
{
  auto h_in = in.shape()[0];
  auto w_in = in.shape()[1];

#ifdef TRIBOL_USE_HOST
  SLIC_ERROR_IF( h_in != out.shape()[1], "Input number of rows does not equal output number of columns." );
  SLIC_ERROR_IF( w_in != out.shape()[0], "Input number of columns does not equal output number of rows." );
#endif

  for ( IndexT i{ 0 }; i < h_in; ++i ) {
    for ( IndexT j{ 0 }; j < w_in; ++j ) {
      out( j, i ) = in( i, j );
    }
  }
}

/**
 * @brief Implements a generic bubble sort algorithm
 *
 * @tparam Container Type of container to sort
 * @tparam Compare Comparator function object
 * @param c Container to sort
 * @param comp Comparator, returns true if first arg is less than second
 */
template <typename Container, typename Compare>
TRIBOL_HOST_DEVICE void bubbleSort( Container& c, Compare comp )
{
  using std::swap;

  const auto size = c.size();
  if ( size < 2 ) {
    return;
  }

  for ( decltype( size ) i = 0; i < size - 1; ++i ) {
    for ( decltype( size ) j = 0; j < size - i - 1; ++j ) {
      if ( comp( c[j + 1], c[j] ) ) {
        swap( c[j], c[j + 1] );
      }
    }
  }
}

/**
 * @brief Computes the product of all elements in a container.
 *
 * @tparam Container Type of container
 * @param c Container to compute product of
 * @return The product of all elements in the container
 */
template <typename Container>
TRIBOL_HOST_DEVICE typename Container::value_type product( const Container& c )
{
  typename Container::value_type result = 1;
  for ( auto val : c ) {
    result *= val;
  }
  return result;
}

/**
 * @brief Finds the minimum element in a container.
 *
 * @tparam Container Type of container
 * @param c Container to find the minimum element of
 * @return The minimum element in the container
 */
template <typename Container>
TRIBOL_HOST_DEVICE typename Container::value_type min( const Container& c )
{
  typename Container::value_type result = std::numeric_limits<typename Container::value_type>::max();
  for ( auto val : c ) {
    if ( val < result ) {
      result = val;
    }
  }
  return result;
}

}  // namespace algorithm

}  // namespace tribol

#endif /* SRC_TRIBOL_UTILS_ALGORITHM_HPP_ */
