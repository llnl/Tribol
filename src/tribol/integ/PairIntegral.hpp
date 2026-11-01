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

}  // namespace tribol

#endif  // SRC_TRIBOL_INTEG_PAIRINTEGRAL_HPP_
