// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Propagate a scalar value, gradient, and Hessian through arithmetic without allocating memory.
 */
#ifndef SRC_TRIBOL_COMMON_SECONDORDERSCALAR_HPP_
#define SRC_TRIBOL_COMMON_SECONDORDERSCALAR_HPP_

#include <cassert>
#include "tribol/common/MdArray.hpp"

namespace tribol {
/**
 * @brief Second-order forward automatic differentiation in a fixed number of local variables.
 *
 * RealT constants convert with zero derivatives. Arithmetic propagates both derivative orders; comparisons use
 * values only. Storage is local and fixed-size, suitable for kernel stack data. NumVariables describes the local
 * inputs to an expression, not global mesh coordinates. Division requires a nonzero denominator; zero division
 * follows floating-point arithmetic and does not produce valid derivatives.
 * @tparam NumVariables Positive number of local differentiation variables.
 */
template <IndexT NumVariables>
struct SecondOrderScalar {
  static_assert( NumVariables > 0 );
  static constexpr IndexT variable_count = NumVariables;  ///< Number of gradient entries and Hessian rows/columns.
  RealT value;                                            ///< Primal scalar value.
  MdArray<RealT, NumVariables> gradient{};                ///< First derivatives in local variable order.
  MdArray<RealT, NumVariables, NumVariables> hessian{};   ///< Second derivatives in row-major variable order.

  /** @brief Construct a constant with zero derivatives. @param constant Scalar value, zero by default. */
  TRIBOL_HOST_DEVICE SecondOrderScalar( RealT constant = 0.0 ) : value( constant ) {}
  /** @brief Destroy stack-owned scalar and derivative storage. */
  TRIBOL_DEFAULT_HOST_DEVICE ~SecondOrderScalar() = default;
  /** @brief Copy a scalar and its derivatives. @param other Scalar to copy. */
  TRIBOL_DEFAULT_HOST_DEVICE SecondOrderScalar( const SecondOrderScalar& other ) = default;
  /** @brief Move a scalar and its derivatives. @param other Scalar to move. */
  TRIBOL_DEFAULT_HOST_DEVICE SecondOrderScalar( SecondOrderScalar&& other ) = default;
  /** @brief Copy a scalar and its derivatives. @param other Scalar to copy. @return This scalar. */
  TRIBOL_DEFAULT_HOST_DEVICE SecondOrderScalar& operator=( const SecondOrderScalar& other ) = default;
  /** @brief Move a scalar and its derivatives. @param other Scalar to move. @return This scalar. */
  TRIBOL_DEFAULT_HOST_DEVICE SecondOrderScalar& operator=( SecondOrderScalar&& other ) = default;

  /**
   * @brief Seed one independent variable with unit gradient and zero Hessian.
   * @param value Initial value.
   * @param index Variable index in [0, NumVariables).
   * @return Seeded variable. An invalid index violates the precondition and asserts in debug builds.
   */
  TRIBOL_HOST_DEVICE static SecondOrderScalar variable( RealT value, IndexT index )
  {
    assert( index >= 0 && index < NumVariables );
    SecondOrderScalar result( value );
    result.gradient[index] = 1.0;
    return result;
  }

  /** @brief Add values and derivatives. @param left First operand. @param right Second operand. @return Sum. */
  TRIBOL_HOST_DEVICE friend SecondOrderScalar operator+( const SecondOrderScalar& left, const SecondOrderScalar& right )
  {
    SecondOrderScalar result( left.value + right.value );
    for ( IndexT i = 0; i < NumVariables; ++i ) {
      result.gradient[i] = left.gradient[i] + right.gradient[i];
      for ( IndexT j = 0; j < NumVariables; ++j ) result.hessian( i, j ) = left.hessian( i, j ) + right.hessian( i, j );
    }
    return result;
  }
  /** @brief Negate values and derivatives. @return Negated scalar. */
  TRIBOL_HOST_DEVICE SecondOrderScalar operator-() const
  {
    SecondOrderScalar result( -value );
    for ( IndexT i = 0; i < NumVariables; ++i ) {
      result.gradient[i] = -gradient[i];
      for ( IndexT j = 0; j < NumVariables; ++j ) result.hessian( i, j ) = -hessian( i, j );
    }
    return result;
  }
  /** @brief Preserve values and derivatives. @return Copy of this scalar. */
  TRIBOL_HOST_DEVICE SecondOrderScalar operator+() const { return *this; }
  /** @brief Subtract values and derivatives. @param left First operand. @param right Second operand. @return
   * Difference. */
  TRIBOL_HOST_DEVICE friend SecondOrderScalar operator-( const SecondOrderScalar& left, const SecondOrderScalar& right )
  {
    return left + ( -right );
  }
  /** @brief Apply the product rule at both orders. @param left First operand. @param right Second operand.
   * @return Product with both mixed-gradient terms in its Hessian. */
  TRIBOL_HOST_DEVICE friend SecondOrderScalar operator*( const SecondOrderScalar& left, const SecondOrderScalar& right )
  {
    SecondOrderScalar result( left.value * right.value );
    for ( IndexT i = 0; i < NumVariables; ++i ) {
      result.gradient[i] = left.gradient[i] * right.value + left.value * right.gradient[i];
      for ( IndexT j = 0; j < NumVariables; ++j )
        result.hessian( i, j ) = left.hessian( i, j ) * right.value + left.gradient[i] * right.gradient[j] +
                                 left.gradient[j] * right.gradient[i] + left.value * right.hessian( i, j );
    }
    return result;
  }
  /** @brief Apply reciprocal and product rules. @param left Numerator. @param right Nonzero denominator.
   * @return Quotient and derivatives; zero division produces nonfinite, invalid results. */
  TRIBOL_HOST_DEVICE friend SecondOrderScalar operator/( const SecondOrderScalar& left, const SecondOrderScalar& right )
  {
    SecondOrderScalar inverse( 1.0 / right.value );
    const RealT first = -inverse.value * inverse.value;
    const RealT second = -2.0 * first * inverse.value;
    for ( IndexT i = 0; i < NumVariables; ++i ) {
      inverse.gradient[i] = first * right.gradient[i];
      for ( IndexT j = 0; j < NumVariables; ++j )
        inverse.hessian( i, j ) = first * right.hessian( i, j ) + second * right.gradient[i] * right.gradient[j];
    }
    return left * inverse;
  }
  /** @brief Assign the sum, allowing self-aliasing. @param right Second operand. @return This scalar. */
  TRIBOL_HOST_DEVICE SecondOrderScalar& operator+=( const SecondOrderScalar& right )
  {
    *this = *this + right;
    return *this;
  }
  /** @brief Assign the difference, allowing self-aliasing. @param right Second operand. @return This scalar. */
  TRIBOL_HOST_DEVICE SecondOrderScalar& operator-=( const SecondOrderScalar& right )
  {
    *this = *this - right;
    return *this;
  }
  /** @brief Assign the product, allowing self-aliasing. @param right Second operand. @return This scalar. */
  TRIBOL_HOST_DEVICE SecondOrderScalar& operator*=( const SecondOrderScalar& right )
  {
    *this = *this * right;
    return *this;
  }
  /** @brief Assign the quotient, allowing self-aliasing. @param right Second operand. @return This scalar. */
  TRIBOL_HOST_DEVICE SecondOrderScalar& operator/=( const SecondOrderScalar& right )
  {
    *this = *this / right;
    return *this;
  }
  /** @brief Compare primal values only. @param left First operand. @param right Second operand. @return Comparison. */
  TRIBOL_HOST_DEVICE friend bool operator<( const SecondOrderScalar& left, const SecondOrderScalar& right )
  {
    return left.value < right.value;
  }
  /** @brief Compare primal values only. @param left First operand. @param right Second operand. @return Comparison. */
  TRIBOL_HOST_DEVICE friend bool operator<=( const SecondOrderScalar& left, const SecondOrderScalar& right )
  {
    return left.value <= right.value;
  }
  /** @brief Compare primal values only. @param left First operand. @param right Second operand. @return Comparison. */
  TRIBOL_HOST_DEVICE friend bool operator>( const SecondOrderScalar& left, const SecondOrderScalar& right )
  {
    return left.value > right.value;
  }
  /** @brief Compare primal values only. @param left First operand. @param right Second operand. @return Comparison. */
  TRIBOL_HOST_DEVICE friend bool operator>=( const SecondOrderScalar& left, const SecondOrderScalar& right )
  {
    return left.value >= right.value;
  }
  /** @brief Compare primal values only. @param left First operand. @param right Second operand. @return Comparison. */
  TRIBOL_HOST_DEVICE friend bool operator==( const SecondOrderScalar& left, const SecondOrderScalar& right )
  {
    return left.value == right.value;
  }
  /** @brief Compare primal values only. @param left First operand. @param right Second operand. @return Comparison. */
  TRIBOL_HOST_DEVICE friend bool operator!=( const SecondOrderScalar& left, const SecondOrderScalar& right )
  {
    return left.value != right.value;
  }
  /** @brief Select the smaller value's derivatives, retaining the first operand at equality.
   * @param left First operand. @param right Second operand. @return Selected scalar. */
  TRIBOL_HOST_DEVICE friend SecondOrderScalar min( const SecondOrderScalar& left, const SecondOrderScalar& right )
  {
    return left.value > right.value ? right : left;
  }
  /** @brief Select the larger value's derivatives, retaining the first operand at equality.
   * @param left First operand. @param right Second operand. @return Selected scalar. */
  TRIBOL_HOST_DEVICE friend SecondOrderScalar max( const SecondOrderScalar& left, const SecondOrderScalar& right )
  {
    return left.value < right.value ? right : left;
  }
};

namespace detail {
/** @brief Read a real nodal state value. @param value Primal value. @return Unchanged value. */
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE constexpr RealT primalValue( RealT value ) { return value; }
/** @brief Read an AD value for branch/support checks. @tparam N Local variable count. @param value AD scalar.
 * @return Primal value, without differentiating the branch decision. */
template <IndexT N>
TRIBOL_HOST_DEVICE RealT primalValue( const SecondOrderScalar<N>& value )
{
  return value.value;
}
}  // namespace detail
}  // namespace tribol
#endif  // SRC_TRIBOL_COMMON_SECONDORDERSCALAR_HPP_
