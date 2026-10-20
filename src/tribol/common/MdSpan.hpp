// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_MDSPAN_HPP_
#define SRC_TRIBOL_COMMON_MDSPAN_HPP_

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Enzyme.hpp"

namespace tribol {

/**
 * @brief Provides a non-owning, fixed-extent view of contiguous row-major data
 *
 * MdSpan stores only a pointer. Extents and row-major strides remain compile-time constants, so passing an MdSpan by
 * value has the same storage cost as passing a pointer. Exact-size C arrays are checked at compile time. Pointer-backed
 * storage must also provide its element count, which is checked when assertions are enabled but is not stored.
 *
 * @tparam T Viewed value type; const-qualified T creates a read-only view
 * @tparam Extents Size of each dimension
 */
template <typename T, IndexT... Extents>
class MdSpan {
  static_assert( sizeof...( Extents ) > 0, "MdSpan requires at least one extent." );
  static_assert( ( ( Extents > 0 ) && ... ), "MdSpan extents must be positive." );

 public:
  using value_type = T;

  /** @brief Number of dimensions in the view */
  TRIBOL_HOST_DEVICE static constexpr IndexT rank() { return sizeof...( Extents ); }

  /** @brief Total number of values in the view */
  TRIBOL_HOST_DEVICE static constexpr IndexT size() { return ( Extents * ... ); }

  /**
   * @brief Return one compile-time extent
   *
   * @tparam Dimension Zero-based dimension
   */
  template <IndexT Dimension>
  TRIBOL_HOST_DEVICE static constexpr IndexT extent()
  {
    static_assert( Dimension >= 0 && Dimension < rank(), "MdSpan dimension is out of range." );
    return extents_[Dimension];
  }

  /**
   * @brief View an exact-size C array
   *
   * Array extent mismatches prevent construction, so local arrays cannot silently provide too little storage.
   */
  template <typename U, std::size_t ElementCount,
            typename = std::enable_if_t<ElementCount == static_cast<std::size_t>( size() ) &&
                                        std::is_convertible<U*, T*>::value>>
  TRIBOL_HOST_DEVICE constexpr MdSpan( U ( &values )[ElementCount] ) noexcept : data_( values )
  {
  }

  /**
   * @brief View pointer-backed storage with a checked element count
   *
   * The count validates the fixed extent when assertions are enabled. MdSpan does not retain the count, so the view
   * remains pointer-sized in optimized host and device kernels.
   */
  TRIBOL_HOST_DEVICE constexpr MdSpan( T* data, IndexT element_count ) noexcept : data_( data )
  {
    assert( data != nullptr );
    assert( element_count == size() );
    static_cast<void>( element_count );
  }

  /** @brief Convert a mutable view to a read-only view with the same extents */
  template <typename U, typename = std::enable_if_t<std::is_convertible<U*, T*>::value>>
  TRIBOL_HOST_DEVICE constexpr MdSpan( MdSpan<U, Extents...> other ) noexcept : data_( other.data() )
  {
  }

  /** @brief Return the viewed data pointer */
  TRIBOL_HOST_DEVICE constexpr T* data() const noexcept { return data_; }

  /** @brief Access the contiguous storage with one flat index */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE T& operator[]( IndexT index ) const noexcept
  {
    assert( index >= 0 && index < size() );
    return data_[index];
  }

  /**
   * @brief Access one value using a row-major index for every dimension
   *
   * The number of indices is checked at compile time. Each index is checked against its corresponding static extent
   * when assertions are enabled.
   */
  template <typename... Indices>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE T& operator()( Indices... indices ) const noexcept
  {
    static_assert( sizeof...( Indices ) == sizeof...( Extents ), "MdSpan requires one index per dimension." );
    static_assert( ( std::is_integral<Indices>::value && ... ), "MdSpan indices must be integral." );
    return data_[ComputeOffset<0>( static_cast<IndexT>( indices )... )];
  }

 private:
  template <IndexT Dimension, typename FirstIndex, typename... RemainingIndices>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static IndexT ComputeOffset( FirstIndex first_index,
                                                                       RemainingIndices... remaining_indices ) noexcept
  {
    assert( first_index >= 0 && first_index < extent<Dimension>() );
    if constexpr ( sizeof...( RemainingIndices ) == 0 ) {
      return first_index;
    } else {
      return first_index * RowMajorStride<Dimension>() + ComputeOffset<Dimension + 1>( remaining_indices... );
    }
  }

  template <IndexT Dimension>
  TRIBOL_HOST_DEVICE static constexpr IndexT RowMajorStride() noexcept
  {
    IndexT stride = 1;
    for ( IndexT trailing_dimension = Dimension + 1; trailing_dimension < rank(); ++trailing_dimension ) {
      stride *= extents_[trailing_dimension];
    }
    return stride;
  }

  inline static constexpr IndexT extents_[rank()] = { Extents... };
  T* data_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_COMMON_MDSPAN_HPP_ */
