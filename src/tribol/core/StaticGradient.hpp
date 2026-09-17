#ifndef TRIBOL_CORE_STATICGRADIENT_HPP_
#define TRIBOL_CORE_STATICGRADIENT_HPP_

#include "tribol/core/ExactTangent.hpp"

#include <array>
#include <type_traits>

namespace tribol::linearization_detail {

template <typename Value, int Size>
struct StaticGradient {
  Value value{};
  std::array<Value, Size> gradient{};

  TRIBOL_HOST_DEVICE constexpr StaticGradient() = default;
  TRIBOL_HOST_DEVICE constexpr StaticGradient( const Value& input ) : value( input ) {}

  template <typename Scalar>
    requires std::is_arithmetic_v<Scalar>
  TRIBOL_HOST_DEVICE constexpr StaticGradient( Scalar input ) : value( Value{ static_cast<Real>( input ) } )
  {
  }

  [[nodiscard]] TRIBOL_HOST_DEVICE static constexpr StaticGradient variable( const Value& input, int index )
  {
    StaticGradient result{ input };
    result.gradient[index] = Value{ 1.0 };
    return result;
  }

  TRIBOL_HOST_DEVICE constexpr StaticGradient& operator+=( const StaticGradient& other )
  {
    value += other.value;
    for ( int entry = 0; entry < Size; ++entry ) {
      gradient[entry] += other.gradient[entry];
    }
    return *this;
  }

  TRIBOL_HOST_DEVICE constexpr StaticGradient& operator-=( const StaticGradient& other )
  {
    value -= other.value;
    for ( int entry = 0; entry < Size; ++entry ) {
      gradient[entry] -= other.gradient[entry];
    }
    return *this;
  }

  TRIBOL_HOST_DEVICE constexpr StaticGradient& operator*=( const StaticGradient& other )
  {
    for ( int entry = 0; entry < Size; ++entry ) {
      gradient[entry] = gradient[entry] * other.value + value * other.gradient[entry];
    }
    value *= other.value;
    return *this;
  }

  TRIBOL_HOST_DEVICE constexpr StaticGradient& operator/=( const StaticGradient& other )
  {
    const Value denominator = other.value * other.value;
    for ( int entry = 0; entry < Size; ++entry ) {
      gradient[entry] = ( gradient[entry] * other.value - value * other.gradient[entry] ) / denominator;
    }
    value /= other.value;
    return *this;
  }
};

template <typename Value, int Size>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr StaticGradient<Value, Size> operator+(
    StaticGradient<Value, Size> left, const StaticGradient<Value, Size>& right )
{
  return left += right;
}

template <typename Value, int Size>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr StaticGradient<Value, Size> operator-(
    StaticGradient<Value, Size> left, const StaticGradient<Value, Size>& right )
{
  return left -= right;
}

template <typename Value, int Size>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr StaticGradient<Value, Size> operator*(
    StaticGradient<Value, Size> left, const StaticGradient<Value, Size>& right )
{
  return left *= right;
}

template <typename Value, int Size>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr StaticGradient<Value, Size> operator/(
    StaticGradient<Value, Size> left, const StaticGradient<Value, Size>& right )
{
  return left /= right;
}

template <typename Value, int Size>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr StaticGradient<Value, Size> operator-(
    const StaticGradient<Value, Size>& input )
{
  StaticGradient<Value, Size> result{ -input.value };
  for ( int entry = 0; entry < Size; ++entry ) {
    result.gradient[entry] = -input.gradient[entry];
  }
  return result;
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr StaticGradient<Value, Size> operator+( StaticGradient<Value, Size> left,
                                                                                  Scalar right )
{
  return left += StaticGradient<Value, Size>{ right };
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr StaticGradient<Value, Size> operator+( Scalar left,
                                                                                  StaticGradient<Value, Size> right )
{
  return right += StaticGradient<Value, Size>{ left };
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr StaticGradient<Value, Size> operator-( StaticGradient<Value, Size> left,
                                                                                  Scalar right )
{
  return left -= StaticGradient<Value, Size>{ right };
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr StaticGradient<Value, Size> operator-(
    Scalar left, const StaticGradient<Value, Size>& right )
{
  return StaticGradient<Value, Size>{ left } - right;
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr StaticGradient<Value, Size> operator*( StaticGradient<Value, Size> left,
                                                                                  Scalar right )
{
  return left *= StaticGradient<Value, Size>{ right };
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr StaticGradient<Value, Size> operator*( Scalar left,
                                                                                  StaticGradient<Value, Size> right )
{
  return right *= StaticGradient<Value, Size>{ left };
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr StaticGradient<Value, Size> operator/( StaticGradient<Value, Size> left,
                                                                                  Scalar right )
{
  return left /= StaticGradient<Value, Size>{ right };
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr StaticGradient<Value, Size> operator/(
    Scalar left, const StaticGradient<Value, Size>& right )
{
  return StaticGradient<Value, Size>{ left } / right;
}

template <typename Value, int Size>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr Real primal( const StaticGradient<Value, Size>& input )
{
  return primal( input.value );
}

template <typename Value, int Size>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator<( const StaticGradient<Value, Size>& left,
                                                           const StaticGradient<Value, Size>& right )
{
  return primal( left ) < primal( right );
}

template <typename Value, int Size>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator<=( const StaticGradient<Value, Size>& left,
                                                            const StaticGradient<Value, Size>& right )
{
  return primal( left ) <= primal( right );
}

template <typename Value, int Size>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator>( const StaticGradient<Value, Size>& left,
                                                           const StaticGradient<Value, Size>& right )
{
  return primal( left ) > primal( right );
}

template <typename Value, int Size>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator>=( const StaticGradient<Value, Size>& left,
                                                            const StaticGradient<Value, Size>& right )
{
  return primal( left ) >= primal( right );
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator<( const StaticGradient<Value, Size>& left, Scalar right )
{
  return primal( left ) < static_cast<Real>( right );
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator<( Scalar left, const StaticGradient<Value, Size>& right )
{
  return static_cast<Real>( left ) < primal( right );
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator<=( const StaticGradient<Value, Size>& left, Scalar right )
{
  return primal( left ) <= static_cast<Real>( right );
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator<=( Scalar left, const StaticGradient<Value, Size>& right )
{
  return static_cast<Real>( left ) <= primal( right );
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator>( const StaticGradient<Value, Size>& left, Scalar right )
{
  return primal( left ) > static_cast<Real>( right );
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator>( Scalar left, const StaticGradient<Value, Size>& right )
{
  return static_cast<Real>( left ) > primal( right );
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator>=( const StaticGradient<Value, Size>& left, Scalar right )
{
  return primal( left ) >= static_cast<Real>( right );
}

template <typename Value, int Size, typename Scalar>
  requires std::is_arithmetic_v<Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool operator>=( Scalar left, const StaticGradient<Value, Size>& right )
{
  return static_cast<Real>( left ) >= primal( right );
}

template <typename Value, int Size>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr StaticGradient<Value, Size> absolute(
    const StaticGradient<Value, Size>& input )
{
  if ( primal( input ) < 0.0 ) {
    return -input;
  }
  return primal( input ) > 0.0 ? input : StaticGradient<Value, Size>{};
}

template <typename Value, int Size>
[[nodiscard]] TRIBOL_HOST_DEVICE inline StaticGradient<Value, Size> squareRoot(
    const StaticGradient<Value, Size>& input )
{
  StaticGradient<Value, Size> result{ squareRoot( input.value ) };
  if ( primal( result.value ) > 0.0 ) {
    const Value scale = Value{ 0.5 } / result.value;
    for ( int entry = 0; entry < Size; ++entry ) {
      result.gradient[entry] = scale * input.gradient[entry];
    }
  }
  return result;
}

}  // namespace tribol::linearization_detail

#endif
