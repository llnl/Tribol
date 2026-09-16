#ifndef TRIBOL_CORE_EXACTTANGENT_HPP_
#define TRIBOL_CORE_EXACTTANGENT_HPP_

#include "tribol/core/Config.hpp"

#include <cmath>

namespace tribol::linearization_detail {

struct ExactTangent {
  Real value{};
  Real tangent{};

  TRIBOL_HOST_DEVICE constexpr ExactTangent() {}
  TRIBOL_HOST_DEVICE constexpr ExactTangent( Real primal ) : value( primal ) {}
  TRIBOL_HOST_DEVICE constexpr ExactTangent( Real primal, Real directional_derivative )
      : value( primal ), tangent( directional_derivative )
  {
  }

  TRIBOL_HOST_DEVICE constexpr ExactTangent& operator+=( ExactTangent other )
  {
    value += other.value;
    tangent += other.tangent;
    return *this;
  }

  TRIBOL_HOST_DEVICE constexpr ExactTangent& operator-=( ExactTangent other )
  {
    value -= other.value;
    tangent -= other.tangent;
    return *this;
  }

  TRIBOL_HOST_DEVICE constexpr ExactTangent& operator*=( ExactTangent other )
  {
    tangent = tangent * other.value + value * other.tangent;
    value *= other.value;
    return *this;
  }

  TRIBOL_HOST_DEVICE constexpr ExactTangent& operator/=( ExactTangent other )
  {
    tangent = ( tangent * other.value - value * other.tangent ) / ( other.value * other.value );
    value /= other.value;
    return *this;
  }
};

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr ExactTangent operator+( ExactTangent left, ExactTangent right )
{
  return left += right;
}

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr ExactTangent operator-( ExactTangent left, ExactTangent right )
{
  return left -= right;
}

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr ExactTangent operator*( ExactTangent left, ExactTangent right )
{
  return left *= right;
}

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr ExactTangent operator/( ExactTangent left, ExactTangent right )
{
  return left /= right;
}

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr ExactTangent operator-( ExactTangent value )
{
  return { -value.value, -value.tangent };
}

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator<( ExactTangent left, ExactTangent right )
{
  return left.value < right.value;
}

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator<=( ExactTangent left, ExactTangent right )
{
  return left.value <= right.value;
}

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator>( ExactTangent left, ExactTangent right )
{
  return left.value > right.value;
}

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator>=( ExactTangent left, ExactTangent right )
{
  return left.value >= right.value;
}

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator==( ExactTangent left, ExactTangent right )
{
  return left.value == right.value;
}

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr Real primal( Real value ) { return value; }

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr Real primal( ExactTangent value ) { return value.value; }

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr Real tangent( Real ) { return 0.0; }

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr Real tangent( ExactTangent value ) { return value.tangent; }

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr Real absolute( Real value ) { return value < 0.0 ? -value : value; }

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr ExactTangent absolute( ExactTangent value )
{
  if ( value.value < 0.0 ) {
    return -value;
  }
  return value.value > 0.0 ? value : ExactTangent{};
}

[[nodiscard]] TRIBOL_HOST_DEVICE inline Real squareRoot( Real value )
{
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
  return ::sqrt( value );
#else
  return std::sqrt( value );
#endif
}

[[nodiscard]] TRIBOL_HOST_DEVICE inline ExactTangent squareRoot( ExactTangent value )
{
  const Real root = squareRoot( value.value );
  return { root, root > 0.0 ? value.tangent / ( 2.0 * root ) : 0.0 };
}

}  // namespace tribol::linearization_detail

#endif
