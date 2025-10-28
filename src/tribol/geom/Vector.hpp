// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_GEOM_VECTOR_HPP_
#define SRC_TRIBOL_GEOM_VECTOR_HPP_

// Tribol config include
#include "tribol/config.hpp"

// Tribol includes
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Arrays.hpp"

namespace tribol {

/**
 * @brief Generic N-dimensional vector container backed by a memory-backed Array.
 *
 * The `Vector` template provides a small vector-like wrapper around an underlying array memory type. It exposes
 * typical vector operations (arithmetic operators, dot/cross products, norms) while delegating storage and
 * capacity/size semantics to the `_MemoryT` type via `ArrayBase`.
 *
 * @tparam _T The element value type stored in the vector.
 * @tparam _MemoryT The underlying memory/container type used to store elements. Defaults to an
 *                  `AllocatedMemory` with `SizeEqCapacity<RuntimeCapacity>` policy.
 */
template <typename _T, typename _MemoryT = AllocatedMemory<_T, Allocator<_T>, SizeEqCapacity<RuntimeCapacity>>>
class Vector : public ArrayBase<_MemoryT> {
 public:
  /// @brief Base class type (Array facade over the memory type).
  using BaseClassT_ = ArrayBase<_MemoryT>;

  /// @brief Underlying memory type used by this vector.
  using typename BaseClassT_::MemoryT_;

  /// @brief Value type stored in the vector.
  using typename BaseClassT_::ValueT_;

  static_assert( std::is_same<_T, ValueT_>::value, "Vector must be used with the same type as the memory type" );

  /**
   * @brief Construct a vector with explicit dimension and capacity.
   *
   * @param dim Logical dimension (number of elements).
   * @param capacity Allocated capacity for the underlying storage.
   */
  TRIBOL_HOST_DEVICE Vector( SizeT dim, SizeT capacity ) : BaseClassT_( MemoryT_( dim, capacity ) ) {}

  /**
   * @brief Construct a vector with dimension equal to capacity.
   *
   * Convenience overload where capacity == dim.
   * @param dim The vector dimension and capacity.
   */
  TRIBOL_HOST_DEVICE Vector( SizeT dim ) : Vector( dim, dim ) {}

  /**
   * @brief Construct a vector from an existing memory object (move).
   *
   * @param memory Memory object to take ownership of (moved).
   */
  TRIBOL_HOST_DEVICE Vector( MemoryT_&& memory ) : BaseClassT_( std::move( memory ) ) {}

  /// @brief Access the underlying memory object.
  using BaseClassT_::memory;

  /// @brief Return the logical dimension (number of elements) of the vector.
  TRIBOL_HOST_DEVICE constexpr SizeT dim() const { return memory().size(); }

  /// @brief Bring base-class element accessor (at) into scope.
  using BaseClassT_::at;

  /**
   * @brief Element-wise addition.
   *
   * Returns a new Vector containing the element-wise sum of this vector and `other`.
   *
   * @param other Vector to add to this vector (must have same dimension).
   * @return A new Vector containing the element-wise sum.
   */
  TRIBOL_HOST_DEVICE Vector operator+( const Vector& other ) const
  {
    assert( other.dim() == dim() );

    Vector result( dim() );
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result.at( i ) = at( i ) + other.at( i );
    }
    return result;
  }
  /**
   * @brief Element-wise subtraction.
   *
   * Returns a new Vector containing the element-wise difference of this vector and `other`.
   *
   * @param other Vector to subtract from this vector (must have same dimension).
   * @return A new Vector containing the element-wise difference.
   */
  TRIBOL_HOST_DEVICE Vector operator-( const Vector& other ) const
  {
    assert( other.dim() == dim() );

    Vector result( dim() );
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result.at( i ) = at( i ) - other.at( i );
    }
    return result;
  }
  /**
   * @brief Scalar multiplication.
   *
   * Multiplies each element of the vector by `scalar` and returns the result.
   *
   * @param scalar Scalar value to multiply each element by.
   * @return A new Vector containing the scaled elements.
   */
  TRIBOL_HOST_DEVICE Vector operator*( const ValueT_& scalar ) const
  {
    Vector result( dim() );
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result.at( i ) = at( i ) * scalar;
    }
    return result;
  }
  /**
   * @brief Scalar division.
   *
   * Divides each element by `scalar` and returns the resulting vector. Asserts on division by zero.
   *
   * @param scalar Scalar value to divide each element by (must be non-zero).
   * @return A new Vector containing the scaled elements.
   */
  TRIBOL_HOST_DEVICE Vector operator/( const ValueT_& scalar ) const
  {
    assert( scalar != ValueT_( 0 ) );

    Vector result( dim() );
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result.at( i ) = at( i ) / scalar;
    }
    return result;
  }
  /**
   * @brief In-place element-wise addition.
   *
   * @param other Vector to add into this vector (must have same dimension).
   * @return Reference to this vector after modification.
   */
  TRIBOL_HOST_DEVICE Vector& operator+=( const Vector& other )
  {
    assert( other.dim() == dim() );

    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      at( i ) += other.at( i );
    }
    return *this;
  }
  /**
   * @brief In-place element-wise subtraction.
   *
   * @param other Vector to subtract from this vector (must have same dimension).
   * @return Reference to this vector after modification.
   */
  TRIBOL_HOST_DEVICE Vector& operator-=( const Vector& other )
  {
    assert( other.dim() == dim() );

    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      at( i ) -= other.at( i );
    }
    return *this;
  }
  /**
   * @brief In-place scalar multiplication.
   *
   * @param scalar Scalar value to multiply each element by.
   * @return Reference to this vector after modification.
   */
  TRIBOL_HOST_DEVICE Vector& operator*=( const ValueT_& scalar )
  {
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      at( i ) *= scalar;
    }
    return *this;
  }
  /**
   * @brief In-place scalar division.
   *
   * @param scalar Scalar value to divide each element by (must be non-zero).
   * @return Reference to this vector after modification.
   */
  TRIBOL_HOST_DEVICE Vector& operator/=( const ValueT_& scalar )
  {
    assert( scalar != ValueT_( 0 ) );

    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      at( i ) /= scalar;
    }
    return *this;
  }
  /**
   * @brief Equality comparison.
   *
   * Two vectors are equal if they have the same dimension and all corresponding elements compare equal.
   *
   * @param other Vector to compare with.
   * @return true if the vectors are equal, false otherwise.
   */
  TRIBOL_HOST_DEVICE bool operator==( const Vector& other ) const
  {
    if ( dim() != other.dim() ) {
      return false;
    }
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      if ( at( i ) != other.at( i ) ) {
        return false;
      }
    }
    return true;
  }
  /**
   * @brief Inequality comparison (negation of operator==).
   *
   * @param other Vector to compare with.
   * @return true if the vectors are not equal, false otherwise.
   */
  TRIBOL_HOST_DEVICE bool operator!=( const Vector& other ) const { return !operator==( other ); }

  /**
   * @brief Dot product with another vector.
   *
   * @tparam _Memory2T Memory type of the other vector.
   * @param other Vector to compute the dot product with. Must have same dimension.
   * @return The scalar dot product result.
   */
  template <typename _Memory2T>
  TRIBOL_HOST_DEVICE ValueT_ dot( const Vector<ValueT_, _Memory2T>& other )
  {
    assert( other.dim() == dim() );

    ValueT_ result = ValueT_();
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result += at( i ) * other.at( i );
    }
    return result;
  }
  /**
   * @brief Cross product (3D only).
   *
   * Computes the cross product of two 3-dimensional vectors and returns the resulting 3-vector.
   * @tparam _Memory2T Memory type of the other vector.
   * @param other Right-hand-side vector for the cross product (3D).
   * @return A 3-dimensional Vector representing the cross product.
   */
  template <typename _Memory2T>
  TRIBOL_HOST_DEVICE Vector cross( const Vector<ValueT_, _Memory2T>& other ) const
  {
    assert( dim() == 3 );
    assert( other.dim() == 3 );

    Vector<ValueT_, MemoryT_> result( 3 );
    result.at( 0 ) = at( 1 ) * other.at( 2 ) - at( 2 ) * other.at( 1 );
    result.at( 1 ) = at( 2 ) * other.at( 0 ) - at( 0 ) * other.at( 2 );
    result.at( 2 ) = at( 0 ) * other.at( 1 ) - at( 1 ) * other.at( 0 );
    return result;
  }
  /**
   * @brief Scalar triple product (this · (other1 × other2)).
   *
   * Computes the scalar triple product for three 3D vectors.
   * @param other1 First vector in the triple product (3D).
   * @param other2 Second vector in the triple product (3D).
   * @return Scalar value of the triple product.
   */
  template <typename _MemoryT2, typename _MemoryT3>
  TRIBOL_HOST_DEVICE ValueT_ tripleProduct( const Vector<ValueT_, _MemoryT2>& other1,
                                            const Vector<ValueT_, _MemoryT3>& other2 ) const
  {
    assert( dim() == 3 );
    assert( other1.dim() == 3 );
    assert( other2.dim() == 3 );

    ValueT_ result = ValueT_();
    result += at( 0 ) * ( other1.at( 1 ) * other2.at( 2 ) - other1.at( 2 ) * other2.at( 1 ) );
    result += at( 1 ) * ( other1.at( 2 ) * other2.at( 0 ) - other1.at( 0 ) * other2.at( 2 ) );
    result += at( 2 ) * ( other1.at( 0 ) * other2.at( 1 ) - other1.at( 1 ) * other2.at( 0 ) );
    return result;
  }

  /**
   * @brief Compute the squared Euclidean norm of the vector.
   *
   * Useful when relative magnitude is needed without taking a costly square root.
   *
   * @return The squared Euclidean norm (sum of squares of components).
   */
  TRIBOL_HOST_DEVICE ValueT_ normSquared() const
  {
    ValueT_ result = ValueT_();
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result += at( i ) * at( i );
    }
    return result;
  }

  /**
   * @brief Compute the Euclidean norm (magnitude) of the vector.
   * @return The Euclidean norm (sqrt of the sum of squares).
   */
  TRIBOL_HOST_DEVICE ValueT_ norm() const { return std::sqrt( normSquared() ); }

  /**
   * @brief Normalize the vector in-place.
   *
   * If the vector norm is zero no change is made.
   *
   * @return void
   */
  TRIBOL_HOST_DEVICE void normalize()
  {
    ValueT_ norm_value = norm();
    if ( norm_value > ValueT_( 0 ) ) {
      for ( SizeT i{ 0 }; i < dim(); ++i ) {
        at( i ) /= norm_value;
      }
    }
  }
};

/**
 * @brief Fixed-size vector with compile-time dimension.
 *
 * `FixedVector` is a convenience alias for a `Vector` whose storage is a `StackMemory` of size `_N`.
 * @tparam _T Element value type.
 * @tparam _N Compile-time fixed dimension.
 */
template <typename _T, SizeT _N>
class FixedVector : public Vector<_T, StackMemory<_T, _N>> {
};

/**
 * @brief Array-of-vectors stored as a 2D bounded array.
 *
 * `VectorArray` represents a collection of vectors all with the same dimension. It is implemented as a
 * `BoundedArray2D` where each row corresponds to a vector.
 *
 * @tparam _T Element value type for each vector component.
 * @tparam _MemoryT Underlying memory type for the 2D array.
 */
template <typename _T, typename _MemoryT = AllocatedMemory<_T, Allocator<_T>, SizeLECapacity<RuntimeCapacity>>>
class VectorArray : public BoundedArray2D<_T, _MemoryT> {
 public:
  using BaseClassT_ = BoundedArray2D<_T, _MemoryT>;
  using typename BaseClassT_::MemoryT_;
  using typename BaseClassT_::ValueT_;

  static_assert( std::is_same<_T, ValueT_>::value, "VectorArray must be used with the same type as the memory type" );

  /**
   * @brief Construct a VectorArray with explicit dimension and counts.
   *
   * @param dim Dimension of each vector (number of columns).
   * @param num_vectors Number of vectors (rows).
   * @param capacity Capacity (in rows) for the underlying storage.
   */
  TRIBOL_HOST_DEVICE VectorArray( SizeT dim, SizeT num_vectors, SizeT capacity )
      : BaseClassT_( num_vectors, dim, capacity )
  {
  }

  /**
   * @brief Construct a VectorArray where capacity == num_vectors.
   */
  TRIBOL_HOST_DEVICE VectorArray( SizeT dim, SizeT num_vectors ) : VectorArray( dim, num_vectors, num_vectors ) {}

  /// @brief Bring rowView from base class into scope.
  using BaseClassT_::rowView;

  /**
   * @brief Return the i-th row as a Vector view (non-owning).
   *
   * The returned Vector references the underlying memory; no copy is performed.
   * @param i Row index to retrieve.
   */
  TRIBOL_HOST_DEVICE Vector<_T, Memory<typename _MemoryT::AccessorT_>> getVector( SizeT i ) const
  {
    assert( i < this->height() );
    return Vector<_T, Memory<typename _MemoryT::AccessorT_>>( rowView( i ).memory() );
  }
};

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_VECTOR_HPP_ */
