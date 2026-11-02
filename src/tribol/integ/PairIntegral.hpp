// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Integrate scalar or fixed-size array integrands over generated element-pair quadrature points.
 */
#ifndef SRC_TRIBOL_INTEG_PAIRINTEGRAL_HPP_
#define SRC_TRIBOL_INTEG_PAIRINTEGRAL_HPP_

#include <type_traits>
#include <utility>

#include "tribol/integ/BasisFunction.hpp"
#include "tribol/integ/NodalIntegralExpressions.hpp"
#include "tribol/integ/PairQuadratureRule.hpp"

namespace tribol {

/**
 * @brief Provide both element geometries and their current quadrature-point coordinates to a pair integrand.
 * @tparam FirstElementGeometry First element geometry type.
 * @tparam SecondElementGeometry Second element geometry type.
 */
template <typename FirstElementGeometry, typename SecondElementGeometry>
struct PairQuadraturePoint {
  using FirstGeometry = FirstElementGeometry;    ///< First element geometry type.
  using SecondGeometry = SecondElementGeometry;  ///< Second element geometry type.

  const FirstGeometry& geometry1;   ///< First element geometry for the current pair calculation.
  const SecondGeometry& geometry2;  ///< Second element geometry for the current pair calculation.
  MdSpan<const RealT, FirstGeometry::parametric_dimension>
      geometry1_coords;  ///< Quadrature point in geometry1's parametric coordinates.
  MdSpan<const RealT, SecondGeometry::parametric_dimension>
      geometry2_coords;  ///< Quadrature point in geometry2's parametric coordinates.

  /**
   * @brief Map geometry1's parametric quadrature coordinate to physical space.
   * @return Physical point on geometry1.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto position1() const
    requires( FirstGeometry::parametric_dimension == 1 )
  {
    return geometry1.position( geometry1_coords[0] );
  }

  /**
   * @brief Map geometry2's parametric quadrature coordinate to physical space.
   * @return Physical point on geometry2.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto position2() const
    requires( SecondGeometry::parametric_dimension == 1 )
  {
    return geometry2.position( geometry2_coords[0] );
  }

  /**
   * @brief Evaluate a Lagrange basis at geometry1's parametric quadrature coordinates.
   * @tparam Element Contact face shape matching geometry1's parametric dimension.
   * @tparam Order Lagrange order of the basis.
   * @return Owned basis values in local-node order.
   */
  template <InterfaceElementType Element, FaceOrderType Order>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE MdArray<RealT, BasisFunction<Element, Order>::node_count> evalBasis1() const
  {
    using Basis = BasisFunction<Element, Order>;
    static_assert( Basis::parametric_dimension == FirstGeometry::parametric_dimension,
                   "Basis parametric dimension must match geometry1." );
    MdArray<RealT, Basis::node_count> values{};
    Basis::eval( geometry1_coords, values.view() );
    return values;
  }

  /**
   * @brief Evaluate a Lagrange basis at geometry2's parametric quadrature coordinates.
   * @tparam Element Contact face shape matching geometry2's parametric dimension.
   * @tparam Order Lagrange order of the basis.
   * @return Owned basis values in local-node order.
   */
  template <InterfaceElementType Element, FaceOrderType Order>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE MdArray<RealT, BasisFunction<Element, Order>::node_count> evalBasis2() const
  {
    using Basis = BasisFunction<Element, Order>;
    static_assert( Basis::parametric_dimension == SecondGeometry::parametric_dimension,
                   "Basis parametric dimension must match geometry2." );
    MdArray<RealT, Basis::node_count> values{};
    Basis::eval( geometry2_coords, values.view() );
    return values;
  }
};

/**
 * @brief Pair integrand value at one quadrature point together with branch activation.
 * @tparam Value Scalar or fixed-size array integrand type.
 */
template <typename Value>
struct QuadraturePointContribution {
  Value value{};         ///< Unweighted integrand value at this quadrature point.
  bool active{ false };  ///< Whether this point uses an active branch, even when value is zero.
};

/**
 * @brief Integrated pair value, branch activation, and quadrature generation status.
 * @tparam Value Integrated scalar or fixed-size array type.
 */
template <typename Value>
struct PairIntegralValue {
  Value value{};                         ///< Weighted sum over active quadrature points; zero on failure or empty rule.
  bool has_active_points{ false };       ///< Whether any quadrature point contributed an active value.
  QuadratureResult quadrature_result{};  ///< Generation status and point count; discard value when status fails.
};

namespace detail {

/**
 * @brief Extract the accumulated value type from a raw integrand return value or a QuadraturePointContribution.
 * @tparam ReturnValue Type returned by a point integrand callable.
 */
template <typename ReturnValue>
struct PairIntegrandTraits {
  using Value = ReturnValue;  ///< Integrated value type when every evaluated point is active.

  /**
   * @brief Treat a raw integrand value as active at the current quadrature point.
   * @param value Unweighted point integrand value.
   * @return Active contribution wrapping the point value.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static QuadraturePointContribution<Value> toContribution( const Value& value )
  {
    return { value, true };
  }
};

/**
 * @brief Specialization for integrands that explicitly report whether each quadrature point is active.
 * @tparam ValueType Underlying scalar or fixed-size array value type.
 */
template <typename ValueType>
struct PairIntegrandTraits<QuadraturePointContribution<ValueType>> {
  using Value = ValueType;  ///< Integrated value type.

  /**
   * @brief Pass an explicit point contribution through unchanged.
   * @param contribution Point integrand value and activation flag.
   * @return Unchanged contribution.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static const QuadraturePointContribution<Value>& toContribution(
      const QuadraturePointContribution<Value>& contribution )
  {
    return contribution;
  }
};

/**
 * @brief Accumulate one weighted scalar quadrature contribution.
 * @param[in,out] sum Running scalar integral.
 * @param weight Physical integration weight at the quadrature point.
 * @param value Unweighted scalar integrand value.
 */
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void addWeightedPointValue( RealT& sum, RealT weight, RealT value )
{
  sum += weight * value;
}

/**
 * @brief Accumulate one weighted fixed-size array quadrature contribution.
 * @tparam Extent Number of entries in the integrated array.
 * @param[in,out] sum Running array integral.
 * @param weight Physical integration weight at the quadrature point.
 * @param value Unweighted array integrand value.
 */
template <IndexT Extent>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE void addWeightedPointValue( MdArray<RealT, Extent>& sum, RealT weight,
                                                                    const MdArray<RealT, Extent>& value )
{
  for ( IndexT i = 0; i < Extent; ++i ) sum[i] += weight * value[i];
}

}  // namespace detail

/**
 * @brief Integrate a point callable over already generated element-pair quadrature data.
 *
 * The integrand is not called when quadrature generation failed or produced zero points. Physical integration weights
 * from the pair quadrature data are applied once to each active point value.
 * @tparam FirstGeometry First element geometry type.
 * @tparam SecondGeometry Second element geometry type.
 * @tparam Capacity Fixed point capacity of the pair quadrature storage.
 * @tparam Integrand Callable accepting PairQuadraturePoint<FirstGeometry, SecondGeometry>.
 * @param geometry1 First current element geometry.
 * @param geometry2 Second current element geometry.
 * @param quadrature Paired parametric coordinates, physical weights, and generation status.
 * @param integrand Point integrand returning a scalar/MdArray or a QuadraturePointContribution.
 * @return Integrated value, whether any point was active, and quadrature status.
 */
template <typename FirstGeometry, typename SecondGeometry, IndexT Capacity, typename Integrand>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto integratePair(
    const FirstGeometry& geometry1, const SecondGeometry& geometry2,
    const PairQuadratureData<FirstGeometry::parametric_dimension, SecondGeometry::parametric_dimension, Capacity>&
        quadrature,
    Integrand&& integrand )
{
  using Point = PairQuadraturePoint<FirstGeometry, SecondGeometry>;
  using ReturnValue = std::remove_cvref_t<decltype( integrand( std::declval<const Point&>() ) )>;
  using Traits = detail::PairIntegrandTraits<ReturnValue>;
  using Value = typename Traits::Value;

  PairIntegralValue<Value> result{};
  result.quadrature_result = { quadrature.result.status, quadrature.result.required_point_count };
  if ( quadrature.result.status != QuadratureStatus::Success || quadrature.result.required_point_count == 0 ) {
    return result;
  }

  const auto points = quadrature.getView();
  for ( IndexT q = 0; q < quadrature.result.required_point_count; ++q ) {
    const Point point{ geometry1,
                       geometry2,
                       { &points.geometry1_coords( q, 0 ), FirstGeometry::parametric_dimension },
                       { &points.geometry2_coords( q, 0 ), SecondGeometry::parametric_dimension } };
    const auto contribution = Traits::toContribution( integrand( point ) );
    if ( !contribution.active ) continue;
    result.has_active_points = true;
    detail::addWeightedPointValue( result.value, points.weights[q], contribution.value );
  }
  return result;
}

/**
 * @brief Construct element geometries, generate pair quadrature, and integrate a point callable.
 * @tparam QuadratureRule Element-pair quadrature generator.
 * @tparam Integrand Callable accepting PairQuadraturePoint for the rule's geometries.
 * @param element1_coords Current first-element coordinates.
 * @param element2_coords Current second-element coordinates.
 * @param quadrature_rule Configured pair quadrature generator.
 * @param integrand Point integrand returning a scalar/MdArray or a QuadraturePointContribution.
 * @return Integrated value, whether any point was active, and quadrature status.
 */
template <typename QuadratureRule, typename Integrand>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto integratePair(
    typename QuadratureRule::FirstGeometry::Coords element1_coords,
    typename QuadratureRule::SecondGeometry::Coords element2_coords, const QuadratureRule& quadrature_rule,
    Integrand&& integrand )
{
  const typename QuadratureRule::FirstGeometry geometry1( element1_coords );
  const typename QuadratureRule::SecondGeometry geometry2( element2_coords );
  return integratePair( geometry1, geometry2, quadrature_rule.generate( geometry1, geometry2 ),
                        std::forward<Integrand>( integrand ) );
}

/**
 * @brief Split packed pair coordinates into both element geometries, generate quadrature, and integrate.
 * @tparam QuadratureRule Element-pair quadrature generator.
 * @tparam Integrand Callable accepting PairQuadraturePoint for the rule's geometries.
 * @param coordinates Packed coordinates with element1 followed by element2.
 * @param quadrature_rule Configured pair quadrature generator.
 * @param integrand Point integrand returning a scalar/MdArray or a QuadraturePointContribution.
 * @return Integrated value, whether any point was active, and quadrature status.
 */
template <typename QuadratureRule, typename Integrand>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto integratePair(
    MdSpan<const RealT, QuadratureRule::FirstGeometry::Coords::size() + QuadratureRule::SecondGeometry::Coords::size()>
        coordinates,
    const QuadratureRule& quadrature_rule, Integrand&& integrand )
{
  constexpr IndexT first_dof_count = QuadratureRule::FirstGeometry::Coords::size();
  constexpr IndexT second_dof_count = QuadratureRule::SecondGeometry::Coords::size();
  return integratePair( { coordinates.data(), first_dof_count },
                        { coordinates.data() + first_dof_count, second_dof_count }, quadrature_rule,
                        std::forward<Integrand>( integrand ) );
}

/** @brief Tag base class enabling point-integrand arithmetic and clamping operators. */
struct PairPointIntegrandTag {};

/**
 * @brief Evaluate a Lagrange basis on geometry1 at a pair quadrature point.
 * @tparam Element Contact face shape matching geometry1's parametric dimension.
 * @tparam Order Lagrange order of the basis.
 */
template <InterfaceElementType Element = EDGE, FaceOrderType Order = LINEAR_LAGRANGE>
struct NodalBasis1 : PairPointIntegrandTag {
  static constexpr IndexT node_count = BasisFunction<Element, Order>::node_count;  ///< Basis values per element.

  /**
   * @brief Evaluate basis functions on geometry1 at the current quadrature point.
   * @tparam Point PairQuadraturePoint type.
   * @param point Current pair quadrature point.
   * @return Owned array of basis values in local-node order.
   */
  template <typename Point>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE MdArray<RealT, node_count> operator()( const Point& point ) const
  {
    return point.template evalBasis1<Element, Order>();
  }
};

/** @brief Linear Lagrange basis functor on the first (non-mortar) edge at a pair quadrature point. */
inline constexpr NodalBasis1<EDGE, LINEAR_LAGRANGE> nodal_basis{};

namespace detail {

/** @brief Detect whether T is an MdArray specialization. */
template <typename T>
inline constexpr bool is_md_array_v = false;

template <typename T, IndexT... Extents>
inline constexpr bool is_md_array_v<MdArray<T, Extents...>> = true;

/**
 * @brief Point integrand that activates only when the underlying scalar integrand is at most an upper bound.
 * @tparam Integrand Scalar point integrand callable.
 */
template <typename Integrand>
struct ClampedPointIntegrand : PairPointIntegrandTag {
  Integrand integrand;       ///< Underlying point integrand.
  RealT upper_bound{ 0.0 };  ///< Maximum active value.

  /**
   * @brief Evaluate and clamp the point integrand, marking values above upper_bound inactive.
   * @tparam Point PairQuadraturePoint type.
   * @param point Current pair quadrature point.
   * @return Active point contribution when at most upper_bound; inactive zero otherwise.
   */
  template <typename Point>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadraturePointContribution<RealT> operator()( const Point& point ) const
  {
    const auto contribution = PairIntegrandTraits<decltype( integrand( point ) )>::toContribution( integrand( point ) );
    if ( !contribution.active || contribution.value > upper_bound ) return {};
    return { contribution.value, true };
  }
};

/**
 * @brief Point integrand scaled by a constant scalar factor.
 * @tparam Integrand Point integrand callable.
 */
template <typename Integrand>
struct ScaledPointIntegrand : PairPointIntegrandTag {
  RealT scale{ 1.0 };   ///< Scalar multiplier.
  Integrand integrand;  ///< Underlying point integrand.

  /**
   * @brief Evaluate the scaled point integrand, preserving explicit point activation when present.
   * @tparam Point PairQuadraturePoint type.
   * @param point Current pair quadrature point.
   * @return Scaled point value or QuadraturePointContribution.
   */
  template <typename Point>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto operator()( const Point& point ) const
  {
    using RawType = std::remove_cvref_t<decltype( integrand( point ) )>;
    if constexpr ( std::same_as<RawType, QuadraturePointContribution<typename PairIntegrandTraits<RawType>::Value>> ) {
      const auto contribution = integrand( point );
      using Value = typename PairIntegrandTraits<RawType>::Value;
      if ( !contribution.active ) return QuadraturePointContribution<Value>{};
      return QuadraturePointContribution<Value>{ scale * contribution.value, true };
    } else {
      return scale * integrand( point );
    }
  }
};

/**
 * @brief Pointwise product of two point integrands.
 * @tparam Left Left point integrand callable.
 * @tparam Right Right point integrand callable.
 */
template <typename Left, typename Right>
struct ProductPointIntegrand : PairPointIntegrandTag {
  Left left;    ///< Left point integrand.
  Right right;  ///< Right point integrand.

  /**
   * @brief Multiply both point integrands at the current quadrature point, preserving branch activation.
   * @tparam Point PairQuadraturePoint type.
   * @param point Current pair quadrature point.
   * @return Product value or QuadraturePointContribution.
   */
  template <typename Point>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto operator()( const Point& point ) const
  {
    using LeftRaw = std::remove_cvref_t<decltype( left( point ) )>;
    using RightRaw = std::remove_cvref_t<decltype( right( point ) )>;
    using LeftTraits = PairIntegrandTraits<LeftRaw>;
    using RightTraits = PairIntegrandTraits<RightRaw>;
    constexpr bool has_contribution = std::same_as<LeftRaw, QuadraturePointContribution<typename LeftTraits::Value>> ||
                                      std::same_as<RightRaw, QuadraturePointContribution<typename RightTraits::Value>>;
    if constexpr ( has_contribution ) {
      const auto left_c = LeftTraits::toContribution( left( point ) );
      const auto right_c = RightTraits::toContribution( right( point ) );
      using Value = std::remove_cvref_t<decltype( left_c.value * right_c.value )>;
      if ( !left_c.active || !right_c.active ) return QuadraturePointContribution<Value>{};
      return QuadraturePointContribution<Value>{ left_c.value * right_c.value, true };
    } else {
      return left( point ) * right( point );
    }
  }
};

}  // namespace detail

/**
 * @brief Clamp a scalar point integrand to at most an upper bound, marking points above the bound inactive.
 * @tparam Integrand Point integrand callable type.
 * @param integrand Scalar point integrand.
 * @param upper_bound Maximum active value (typically 0.0 for compressive gaps).
 * @return Point integrand returning an active QuadraturePointContribution when at most upper_bound.
 */
template <typename Integrand>
  requires( !std::is_arithmetic_v<std::remove_cvref_t<Integrand>> &&
            !detail::is_md_array_v<std::remove_cvref_t<Integrand>> &&
            !std::derived_from<std::remove_cvref_t<Integrand>, NodalFieldExpressionTag> )
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE constexpr detail::ClampedPointIntegrand<std::remove_cvref_t<Integrand>> min(
    Integrand&& integrand, RealT upper_bound )
{
  return { {}, std::forward<Integrand>( integrand ), upper_bound };
}

/**
 * @brief Scale a point integrand by a scalar on the left.
 * @tparam Integrand Point integrand derived from PairPointIntegrandTag.
 * @param scale Scalar multiplier.
 * @param integrand Point integrand.
 * @return Scaled point integrand.
 */
template <typename Integrand>
  requires std::derived_from<std::remove_cvref_t<Integrand>, PairPointIntegrandTag>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE constexpr detail::ScaledPointIntegrand<std::remove_cvref_t<Integrand>>
operator*( RealT scale, Integrand&& integrand )
{
  return { {}, scale, std::forward<Integrand>( integrand ) };
}

/**
 * @brief Scale a point integrand by a scalar on the right.
 * @tparam Integrand Point integrand derived from PairPointIntegrandTag.
 * @param integrand Point integrand.
 * @param scale Scalar multiplier.
 * @return Scaled point integrand.
 */
template <typename Integrand>
  requires std::derived_from<std::remove_cvref_t<Integrand>, PairPointIntegrandTag>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE constexpr detail::ScaledPointIntegrand<std::remove_cvref_t<Integrand>>
operator*( Integrand&& integrand, RealT scale )
{
  return { {}, scale, std::forward<Integrand>( integrand ) };
}

/**
 * @brief Multiply two point integrands pointwise at each quadrature point.
 * @tparam Left Left point integrand callable.
 * @tparam Right Right point integrand callable.
 * @param left Left point integrand.
 * @param right Right point integrand.
 * @return Pointwise product integrand.
 */
template <typename Left, typename Right>
  requires( ( std::derived_from<std::remove_cvref_t<Left>, PairPointIntegrandTag> ||
              std::derived_from<std::remove_cvref_t<Right>, PairPointIntegrandTag> ) &&
            !std::is_arithmetic_v<std::remove_cvref_t<Left>> && !std::is_arithmetic_v<std::remove_cvref_t<Right>> &&
            !detail::is_md_array_v<std::remove_cvref_t<Left>> && !detail::is_md_array_v<std::remove_cvref_t<Right>> )
TRIBOL_HOST_DEVICE
    TRIBOL_ENZYME_INLINE constexpr detail::ProductPointIntegrand<std::remove_cvref_t<Left>, std::remove_cvref_t<Right>>
    operator*( Left&& left, Right&& right )
{
  return { {}, std::forward<Left>( left ), std::forward<Right>( right ) };
}

/**
 * @brief Bound element-pair integration rule that evaluates point integrands on a pair's generated quadrature points.
 * @tparam QuadratureRule Element-pair quadrature generator.
 */
template <typename QuadratureRule>
struct PairIntegrationRule {
  using FirstGeometry = typename QuadratureRule::FirstGeometry;               ///< First element geometry type.
  using SecondGeometry = typename QuadratureRule::SecondGeometry;             ///< Second element geometry type.
  static constexpr IndexT first_dof_count = FirstGeometry::Coords::size();    ///< First-element coordinate DOFs.
  static constexpr IndexT second_dof_count = SecondGeometry::Coords::size();  ///< Second-element coordinate DOFs.
  static constexpr IndexT dof_count = first_dof_count + second_dof_count;     ///< Pair coordinate DOFs.

  FirstGeometry geometry1;                         ///< First current element geometry.
  SecondGeometry geometry2;                        ///< Second current element geometry.
  typename QuadratureRule::QuadratureData points;  ///< Generated pair quadrature points and weights.
  mutable QuadratureResult quadrature_result{};    ///< Generation status and point count from the latest integral.
  mutable bool has_active_points{ false };         ///< Whether any evaluated integral had active quadrature points.

  /**
   * @brief Construct both element geometries and generate pair quadrature once for the current pair coordinates.
   * @param coordinates Packed pair coordinates with element1 followed by element2.
   * @param quadrature_rule Configured pair quadrature generator.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairIntegrationRule( MdSpan<const RealT, dof_count> coordinates,
                                                               const QuadratureRule& quadrature_rule )
      : geometry1( { coordinates.data(), first_dof_count } ),
        geometry2( { coordinates.data() + first_dof_count, second_dof_count } ),
        points( quadrature_rule.generate( geometry1, geometry2 ) ),
        quadrature_result{ points.result.status, points.result.required_point_count }
  {
  }

  /**
   * @brief Integrate a point integrand over the generated pair quadrature points.
   * @tparam Integrand Callable accepting PairQuadraturePoint<FirstGeometry, SecondGeometry>.
   * @param integrand Point integrand.
   * @return Integrated scalar or fixed-size array value.
   */
  template <typename Integrand>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto integrate( Integrand&& integrand ) const
  {
    const auto result = integratePair( geometry1, geometry2, points, std::forward<Integrand>( integrand ) );
    quadrature_result = result.quadrature_result;
    has_active_points = has_active_points || result.has_active_points;
    return result.value;
  }
};

/**
 * @brief Integrate a point integrand using the supplied pair or nodal integration rule context.
 * @tparam Integrand Point integrand callable.
 * @tparam IntegrationRule Integration rule context providing integrate(integrand).
 * @param integrand Point integrand expression (e.g., 0.5 * penalty * gap * gap or nodal_basis * gap).
 * @param integration_rule Integration rule context.
 * @return Integrated scalar or array value.
 */
template <typename Integrand, typename IntegrationRule>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto integral( Integrand&& integrand, const IntegrationRule& integration_rule )
{
  return integration_rule.integrate( std::forward<Integrand>( integrand ) );
}

}  // namespace tribol

#endif  // SRC_TRIBOL_INTEG_PAIRINTEGRAL_HPP_
