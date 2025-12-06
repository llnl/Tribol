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
#include "tribol/common/ArrayTypes.hpp"

namespace tribol {

// Dynamic extent constant
inline constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);

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
template <typename _T, SizeT _Dim = dynamic_extent, typename _ContainerT = Array1D<_T>>
class Vector : public _ContainerT {
 public:
  /// @brief Base class type (Array facade over the memory type).
  using BaseClassT_ = _ContainerT;

  /// @brief Value type stored in the vector.
  using ValueT_ = _T;

  static_assert( std::is_same<_T, ValueT_>::value, "Vector must be used with the same type as the memory type" );

  /**
   * @brief Construct a vector with explicit dimension and capacity.
   *
   * @param dim Logical dimension (number of elements).
   * @param capacity Allocated capacity for the underlying storage.
   */
  TRIBOL_HOST_DEVICE Vector( SizeT dim, SizeT capacity ) : BaseClassT_( dim, capacity ) {
    if constexpr ( _Dim != dynamic_extent) {
      assert( dim == _Dim );
    }
  }

  /**
   * @brief Construct a vector with dimension equal to capacity.
   *
   * Convenience overload where capacity == dim.
   * @param dim The vector dimension and capacity.
   */
  TRIBOL_HOST_DEVICE Vector( SizeT dim ) : Vector( dim, dim ) {}

  /**
   * @brief Construct a vector from an existing array object (move).
   *
   * @param array Memory object to take ownership of (moved).
   */
  TRIBOL_HOST_DEVICE Vector( BaseClassT_&& array ) : BaseClassT_( std::move( array ) ) {
    if constexpr ( _Dim != dynamic_extent) {
      assert( dim() == _Dim );
    }
  }

  /// @brief Access the underlying memory.
  using BaseClassT_::data;

  /// @brief Return the logical dimension (number of elements) of the vector.
  TRIBOL_HOST_DEVICE constexpr SizeT dim() const { return data().size(); }

  /// @brief Bring base-class element accessor (operator[]) into scope.
  using BaseClassT_::operator[];

  /**
   * @brief Element-wise addition.
   *
   * Returns a new Vector containing the element-wise sum of this vector and `other`.
   *
   * @param other Vector to add to this vector (must have same dimension).
   * @return A new Vector containing the element-wise sum.
   */
  template <typename _T2, SizeT _Dim2, typename _ContainerT2>
  TRIBOL_HOST_DEVICE Vector operator+( const Vector<_T2, _Dim2, _ContainerT2>& other ) const
  {
    if constexpr ( _Dim == dynamic_extent || _Dim2 == dynamic_extent ) {
      assert( other.dim() == dim() );
    } else {
      static_assert( _Dim == _Dim2 );
    }

    Vector result( dim() );
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result[i] = (*this)[i] + other[i];
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
  template <typename _T2, SizeT _Dim2, typename _ContainerT2>
  TRIBOL_HOST_DEVICE Vector operator-( const Vector<_T2, _Dim2, _ContainerT2>& other ) const
  {
    if constexpr ( _Dim == dynamic_extent || _Dim2 == dynamic_extent ) {
      assert( other.dim() == dim() );
    } else {
      static_assert( _Dim == _Dim2 );
    }

    Vector result( dim() );
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result[i] = (*this)[i] - other[i];
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
      result[i] = (*this)[i] * scalar;
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
      result[i] = (*this)[i] / scalar;
    }
    return result;
  }
  /**
   * @brief In-place element-wise addition.
   *
   * @param other Vector to add into this vector (must have same dimension).
   * @return Reference to this vector after modification.
   */
  template <typename _T2, SizeT _Dim2, typename _ContainerT2>
  TRIBOL_HOST_DEVICE Vector& operator+=( const Vector<_T2, _Dim2, _ContainerT2>& other )
  {
    if constexpr ( _Dim == dynamic_extent || _Dim2 == dynamic_extent ) {
      assert( other.dim() == dim() );
    } else {
      static_assert( _Dim == _Dim2 );
    }

    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      (*this)[i] += other[i];
    }
    return *this;
  }
  /**
   * @brief In-place element-wise subtraction.
   *
   * @param other Vector to subtract from this vector (must have same dimension).
   * @return Reference to this vector after modification.
   */
  template <typename _T2, SizeT _Dim2, typename _ContainerT2>
  TRIBOL_HOST_DEVICE Vector& operator-=( const Vector<_T2, _Dim2, _ContainerT2>& other )
  {
    if constexpr ( _Dim == dynamic_extent || _Dim2 == dynamic_extent ) {
      assert( other.dim() == dim() );
    } else {
      static_assert( _Dim == _Dim2 );
    }

    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      (*this)[i] -= other[i];
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
      (*this)[i] *= scalar;
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
      (*this)[i] /= scalar;
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
  template <typename _T2, SizeT _Dim2, typename _ContainerT2>
  TRIBOL_HOST_DEVICE bool operator==( const Vector<_T2, _Dim2, _ContainerT2>& other ) const
  {
    if constexpr ( _Dim == dynamic_extent || _Dim2 == dynamic_extent ) {
      if ( dim() != other.dim() ) {
        return false;
      }
    } else {
      if constexpr ( _Dim != _Dim2 ) {
        return false;
      }
    }

    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      if ( (*this)[i] != other[i] ) {
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
  template <typename _T2, SizeT _Dim2, typename _ContainerT2>
  TRIBOL_HOST_DEVICE bool operator!=( const Vector<_T2, _Dim2, _ContainerT2>& other ) const { return !operator==( other ); }

  /**
   * @brief Dot product with another vector.
   *
   * @param other Vector to compute the dot product with. Must have same dimension.
   * @return The scalar dot product result.
   */
  template <typename _T2, SizeT _Dim2, typename _ContainerT2>
  TRIBOL_HOST_DEVICE ValueT_ dot( const Vector<_T2, _Dim2, _ContainerT2>& other )
  {
    if constexpr ( _Dim == dynamic_extent || _Dim2 == dynamic_extent ) {
      assert( other.dim() == dim() );
    } else {
      static_assert( _Dim == _Dim2 );
    }

    ValueT_ result = ValueT_();
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result += (*this)[i] * other[i];
    }
    return result;
  }
  /**
   * @brief Cross product (3D only).
   *
   * Computes the cross product of two 3-dimensional vectors and returns the resulting 3-vector.
   * @param other Right-hand-side vector for the cross product (3D).
   * @return A 3-dimensional Vector representing the cross product.
   */
  template <typename _T2, SizeT _Dim2, typename _ContainerT2>
  TRIBOL_HOST_DEVICE Vector cross( const Vector<_T2, _Dim2, _ContainerT2>& other ) const
  {
    if constexpr ( _Dim == dynamic_extent ) {
        assert( dim() == 3 );
    } else {
        static_assert( _Dim == 3 );
    }

    if constexpr ( _Dim2 == dynamic_extent ) {
        assert( other.dim() == 3 );
    } else {
        static_assert( _Dim2 == 3 );
    }

    Vector result( 3 );
    result[0] = (*this)[1] * other[2] - (*this)[2] * other[1];
    result[1] = (*this)[2] * other[0] - (*this)[0] * other[2];
    result[2] = (*this)[0] * other[1] - (*this)[1] * other[0];
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
  template <typename _T2, SizeT _Dim2, typename _ContainerT2, typename _T3, SizeT _Dim3, typename _ContainerT3>
  TRIBOL_HOST_DEVICE ValueT_ tripleProduct( const Vector<_T2, _Dim2, _ContainerT2>& other1,
                                            const Vector<_T3, _Dim3, _ContainerT3>& other2 ) const
  {
    if constexpr ( _Dim == dynamic_extent ) {
        assert( dim() == 3 );
    } else {
        static_assert( _Dim == 3 );
    }

    if constexpr ( _Dim2 == dynamic_extent ) {
        assert( other1.dim() == 3 );
    } else {
        static_assert( _Dim2 == 3 );
    }

    if constexpr ( _Dim3 == dynamic_extent ) {
        assert( other2.dim() == 3 );
    } else {
        static_assert( _Dim3 == 3 );
    }

    ValueT_ result = ValueT_();
    result += (*this)[0] * ( other1[1] * other2[2] - other1[2] * other2[1] );
    result += (*this)[1] * ( other1[2] * other2[0] - other1[0] * other2[2] );
    result += (*this)[2] * ( other1[0] * other2[1] - other1[1] * other2[0] );
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
      result += (*this)[i] * (*this)[i];
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
        (*this)[i] /= norm_value;
      }
    }
  }
};

/**
 * @brief Fixed-size vector with compile-time dimension.
 *
 * `FixedVector` is a convenience alias for a `Vector` whose storage is a `StackArrayT` of size `_N`.
 * @tparam _T Element value type.
 * @tparam _N Compile-time fixed dimension.
 */
template <typename _T, SizeT _N>
class FixedVector : public Vector<_T, _N, StackArrayT<_T, _N>> {
 public:
  using BaseClassT_ = Vector<_T, _N, StackArrayT<_T, _N>>;
  using ContainerT_ = StackArrayT<_T, _N>;

  /**
   * @brief Default constructor.
   */
  TRIBOL_HOST_DEVICE FixedVector() : BaseClassT_( ContainerT_() ) {}

  /**
   * @brief Construct from existing container.
   */
  TRIBOL_HOST_DEVICE FixedVector( ContainerT_&& array ) : BaseClassT_( std::move( array ) ) {}
};

/**
 * @brief Array-of-vectors stored as a 2D array.
 *
 * `VectorArray` represents a collection of vectors all with the same dimension. It is implemented as a
 * `Array2D` where each row corresponds to a vector.
 *
 * @tparam _T Element value type for each vector component.
 * @tparam _ContainerT Underlying memory type for the 2D array.
 */
template <typename _T, SizeT _Dim = dynamic_extent, typename _ContainerT = Array2D<_T>>
class VectorArray : public _ContainerT {
 public:
  using BaseClassT_ = _ContainerT;
  using ValueT_ = _T;

  static_assert( std::is_same<_T, typename BaseClassT_::value_type>::value, "VectorArray must be used with the same type as the memory type" );

  /**
   * @brief Construct a VectorArray with explicit dimension and counts.
   *
   * @param dim Dimension of each vector (number of columns).
   * @param num_vectors Number of vectors (rows).
   */
  TRIBOL_HOST_DEVICE VectorArray( SizeT dim, SizeT num_vectors )
      : BaseClassT_( num_vectors, dim )
  {
  }

  TRIBOL_HOST_DEVICE void setVector( SizeT i, const Vector<_T, _Dim, Array1D<_T>>& vector)
  {
    assert( i < this->shape()[0] );
    const SizeT num_cols = this->shape()[1];
    for ( SizeT j{ 0 }; j < num_cols; ++j ) {
      (*this)(i, j) = vector[j];
    }
  }

  /**
   * @brief Return the i-th row as a Vector view (non-owning).
   *
   * The returned Vector references the underlying memory; no copy is performed.
   * @param i Row index to retrieve.
   */
  TRIBOL_HOST_DEVICE Vector<_T, _Dim, Array1DView<_T>> getVector( SizeT i ) const
  {
    assert( i < this->shape()[0] );
    const SizeT num_cols = this->shape()[1];
    return Vector<_T, _Dim, Array1DView<_T>>( Array1DView<_T>( &(*this)(i, 0), num_cols ) );
  }

  /**
   * @brief Return the i-th row as a Vector view (non-owning) - Mutable.
   *
   * @param i Row index to retrieve.
   */
  TRIBOL_HOST_DEVICE Vector<_T, _Dim, Array1DView<_T>> getVector( SizeT i )
  {
    assert( i < this->shape()[0] );
    const SizeT num_cols = this->shape()[1];
    return Vector<_T, _Dim, Array1DView<_T>>( Array1DView<_T>( &(*this)(i, 0), num_cols ) );
  }
};

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_VECTOR_HPP_ */
