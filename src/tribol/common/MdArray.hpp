// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_MDARRAY_HPP_
#define SRC_TRIBOL_COMMON_MDARRAY_HPP_

#include <cmath>
#include <concepts>
#include <type_traits>

#include "tribol/common/MdSpan.hpp"

namespace tribol {

/**
 * @brief Owns a fixed-size, row-major multidimensional array without allocating memory
 *
 * MdArray stores only its elements. Its extents, size, and row-major indexing come from MdSpan so owning arrays and
 * non-owning views use the same layout. An MdArray creates a view when requested instead of storing a view that could
 * point to another object's elements after a copy.
 *
 * Views are available only from lvalues. Requiring the owning array to outlive its view prevents a temporary MdArray
 * from creating a dangling MdSpan.
 *
 * @tparam T Stored value type
 * @tparam Extents Size of each dimension
 */
template <typename T, IndexT... Extents>
class MdArray {
  static_assert( !std::is_const<T>::value, "MdArray cannot own const-qualified values." );

 public:
  using value_type = T;
  using span_type = MdSpan<T, Extents...>;
  using const_span_type = MdSpan<const T, Extents...>;

  /** @brief Number of dimensions in the array */
  TRIBOL_HOST_DEVICE static constexpr IndexT rank() { return span_type::rank(); }

  /** @brief Total number of values in the array */
  TRIBOL_HOST_DEVICE static constexpr IndexT size() { return span_type::size(); }

  /**
   * @brief Return the number of values along one dimension
   *
   * @tparam Dimension Zero-based dimension
   */
  template <IndexT Dimension>
  TRIBOL_HOST_DEVICE static constexpr IndexT extent()
  {
    return span_type::template extent<Dimension>();
  }

  TRIBOL_DEFAULT_HOST_DEVICE MdArray() = default;
  TRIBOL_DEFAULT_HOST_DEVICE ~MdArray() = default;

  TRIBOL_DEFAULT_HOST_DEVICE MdArray( const MdArray& other ) = default;
  TRIBOL_DEFAULT_HOST_DEVICE MdArray( MdArray&& other ) = default;

  /**
   * @brief Construct a rank-one array from exactly one value per entry
   *
   * The number of constructor arguments must match the array extent. Requiring every value prevents brace
   * initialization from silently filling omitted entries with zero.
   */
  template <typename... Values>
    requires( rank() == 1 && sizeof...( Values ) == size() && ( std::convertible_to<Values, T> && ... ) )
  TRIBOL_HOST_DEVICE constexpr explicit MdArray( Values... values ) : data_{ static_cast<T>( values )... }
  {
  }

  /**
   * @brief Construct a rank-two array from exactly one fixed-size owning or non-owning row
   *
   * Each MdArray or MdSpan row must contain the number of columns in this array. The constructor copies rows into
   * contiguous row-major storage so nested initialization retains the same layout used by MdSpan. A view row avoids an
   * intermediate owning row when the values already exist in caller storage.
   */
  template <typename... Rows>
    requires( rank() == 2 && sizeof...( Rows ) == extent<0>() &&
              ( ( std::same_as<std::remove_cvref_t<Rows>, MdArray<T, extent<1>()>> ||
                  std::same_as<std::remove_cvref_t<Rows>, MdSpan<T, extent<1>()>> ||
                  std::same_as<std::remove_cvref_t<Rows>, MdSpan<const T, extent<1>()>> ) &&
                ... ) )
  TRIBOL_HOST_DEVICE constexpr explicit MdArray( const Rows&... rows )
  {
    IndexT row_index = 0;
    ( CopyRow( row_index++, rows ), ... );
  }

  TRIBOL_DEFAULT_HOST_DEVICE MdArray& operator=( const MdArray& other ) = default;
  TRIBOL_DEFAULT_HOST_DEVICE MdArray& operator=( MdArray&& other ) = default;

  /** @brief Return a mutable view of this array */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE span_type view() & noexcept { return { data_, size() }; }

  /** @brief Return a read-only view of this array */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE const_span_type view() const& noexcept { return { data_, size() }; }

  // A view into a temporary would outlive its elements at the end of the full expression.
  span_type view() && = delete;
  const_span_type view() const&& = delete;

  /** @brief Return the owned contiguous storage */
  TRIBOL_HOST_DEVICE constexpr T* data() noexcept { return data_; }

  /** @brief Return the owned contiguous storage as read-only data */
  TRIBOL_HOST_DEVICE constexpr const T* data() const noexcept { return data_; }

  /** @brief Access the contiguous storage with one flat index */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE T& operator[]( IndexT index ) noexcept { return view()[index]; }

  /** @brief Access the contiguous storage with one read-only flat index */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE const T& operator[]( IndexT index ) const noexcept { return view()[index]; }

  /** @brief Access one value using a row-major index for every dimension */
  template <typename... Indices>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE T& operator()( Indices... indices ) noexcept
  {
    return view()( indices... );
  }

  /** @brief Access one read-only value using a row-major index for every dimension */
  template <typename... Indices>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE const T& operator()( Indices... indices ) const noexcept
  {
    return view()( indices... );
  }

 private:
  // Copying one complete row keeps the owning array contiguous for the MdSpan view returned later.
  template <typename Row>
  TRIBOL_HOST_DEVICE constexpr void CopyRow( IndexT row_index, const Row& row )
  {
    for ( IndexT column = 0; column < extent<1>(); ++column ) {
      data_[row_index * extent<1>() + column] = row[column];
    }
  }

  T data_[size()];
};

/**
 * @brief Scale each entry of a rank-one array by a scalar.
 * @tparam T Element value type.
 * @tparam Extent Number of entries in the vector.
 * @param array Input rank-one array.
 * @param scalar Scalar multiplier.
 * @return Owned rank-one array with each entry multiplied by scalar.
 */
template <typename T, IndexT Extent>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE constexpr MdArray<T, Extent> operator*( const MdArray<T, Extent>& array,
                                                                                T scalar )
{
  MdArray<T, Extent> result{};
  for ( IndexT i = 0; i < Extent; ++i ) result[i] = array[i] * scalar;
  return result;
}

/**
 * @brief Scale each entry of a rank-one array by a scalar on the left.
 * @tparam T Element value type.
 * @tparam Extent Number of entries in the vector.
 * @param scalar Scalar multiplier.
 * @param array Input rank-one array.
 * @return Owned rank-one array with each entry multiplied by scalar.
 */
template <typename T, IndexT Extent>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE constexpr MdArray<T, Extent> operator*( T scalar,
                                                                                const MdArray<T, Extent>& array )
{
  return array * scalar;
}

/**
 * @brief Divide a scalar by each entry of a rank-one array, returning zero for entries within tolerance of zero.
 * @tparam T Element value type.
 * @tparam Extent Number of entries in the vector.
 * @param scalar Numerator value.
 * @param array Denominator rank-one array.
 * @return Owned rank-one array with each nonzero entry dividing scalar.
 */
template <typename T, IndexT Extent>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE constexpr MdArray<T, Extent> operator/( T scalar,
                                                                                const MdArray<T, Extent>& array )
{
  constexpr T zero_tolerance = static_cast<T>( 1.0e-14 );
  MdArray<T, Extent> result{};
  for ( IndexT i = 0; i < Extent; ++i ) {
    result[i] = std::abs( array[i] ) > zero_tolerance ? scalar / array[i] : T{};
  }
  return result;
}

/**
 * @brief Clamp each entry of a rank-one array to at most an upper bound.
 * @tparam T Element value type.
 * @tparam Extent Number of entries in the vector.
 * @param array Input rank-one array.
 * @param upper_bound Maximum value retained for each entry.
 * @return Owned rank-one array with each entry equal to min(array[i], upper_bound).
 */
template <typename T, IndexT Extent>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE constexpr MdArray<T, Extent> min( const MdArray<T, Extent>& array,
                                                                          T upper_bound )
{
  MdArray<T, Extent> result{};
  for ( IndexT i = 0; i < Extent; ++i ) result[i] = array[i] < upper_bound ? array[i] : upper_bound;
  return result;
}

/**
 * @brief Sum the elementwise product of one or more rank-one owning arrays or views with matching extent.
 * @tparam First First rank-one MdArray or MdSpan.
 * @tparam Rest Additional rank-one MdArray or MdSpan types with the same extent as First.
 * @param first First vector.
 * @param rest Additional vectors multiplied entry-by-entry with first.
 * @return Sum over all entries of the elementwise product.
 */
template <typename First, typename... Rest>
  requires( First::rank() == 1 && ( ( Rest::rank() == 1 && Rest::size() == First::size() ) && ... ) )
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE constexpr auto sum( const First& first, const Rest&... rest )
{
  using ResultType = std::remove_cv_t<typename First::value_type>;
  ResultType total{};
  for ( IndexT i = 0; i < First::size(); ++i ) total += ( first[i] * ... * rest[i] );
  return total;
}

/**
 * @brief Compute the inner product of two rank-one owning arrays or views with matching extent.
 * @tparam Left Rank-one MdArray or MdSpan.
 * @tparam Right Rank-one MdArray or MdSpan with the same extent as Left.
 * @param left First vector.
 * @param right Second vector.
 * @return Sum of elementwise products.
 */
template <typename Left, typename Right>
  requires( Left::rank() == 1 && Right::rank() == 1 && Left::size() == Right::size() )
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE constexpr auto dot( const Left& left, const Right& right )
{
  return sum( left, right );
}

}  // namespace tribol

#endif /* SRC_TRIBOL_COMMON_MDARRAY_HPP_ */
