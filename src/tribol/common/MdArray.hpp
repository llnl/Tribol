// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_MDARRAY_HPP_
#define SRC_TRIBOL_COMMON_MDARRAY_HPP_

#include <concepts>
#include <type_traits>

#include "tribol/common/MdSpan.hpp"

namespace tribol {

template <typename T, IndexT... Extents>
class MdArray;

namespace detail {

template <typename Row, typename T, IndexT ColumnCount>
inline constexpr bool is_exact_mdarray_row = std::same_as<std::remove_cvref_t<Row>, MdArray<T, ColumnCount>> ||
                                             std::same_as<std::remove_cvref_t<Row>, MdSpan<T, ColumnCount>> ||
                                             std::same_as<std::remove_cvref_t<Row>, MdSpan<const T, ColumnCount>>;

// Only rank-two arrays accept rows, and every supplied row must match the array's column extent.
template <IndexT... Extents>
struct MdArrayExtents {
  template <typename T, typename... Rows>
  static constexpr bool have_exact_rows = false;
};

template <IndexT RowCount, IndexT ColumnCount>
struct MdArrayExtents<RowCount, ColumnCount> {
  template <typename T, typename... Rows>
  static constexpr bool have_exact_rows =
      sizeof...( Rows ) == RowCount && ( is_exact_mdarray_row<Rows, T, ColumnCount> && ... );
};

}  // namespace detail

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
   * @brief Return one compile-time extent
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
    requires( detail::MdArrayExtents<Extents...>::template have_exact_rows<T, Rows...> )
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

}  // namespace tribol

#endif /* SRC_TRIBOL_COMMON_MDARRAY_HPP_ */
