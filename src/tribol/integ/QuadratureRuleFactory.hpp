// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Polynomial-order selection and configured construction of compile-time quadrature rule types.
 */

#ifndef SRC_TRIBOL_INTEG_QUADRATURERULEFACTORY_HPP_
#define SRC_TRIBOL_INTEG_QUADRATURERULEFACTORY_HPP_

#include <stdexcept>
#include <utility>

#include "tribol/integ/EdgeQuadratureRule.hpp"

namespace tribol {
namespace detail {

/**
 * @brief Select the first rule whose exact degree meets the requested order.
 *
 * @throws std::invalid_argument No supported rule covers the requested order.
 *
 * @tparam QuadratureRule First supported rule, with its exact polynomial degree.
 * @tparam OtherQuadratureRules Remaining rule types in increasing degree order.
 * @tparam Create Callable with `operator()<SelectedRule>()` and a common return type for all rules.
 * @param order Requested reference polynomial degree.
 * @param create Callback invoked with the selected rule type.
 * @return Callback result.
 */
template <typename QuadratureRule, typename... OtherQuadratureRules, typename Create>
decltype( auto ) createWithQuadratureRuleImpl( int order, Create&& create )
{
  if ( order <= QuadratureRule::polynomial_order ) {
    return std::forward<Create>( create ).template operator()<QuadratureRule>();
  }
  if constexpr ( sizeof...( OtherQuadratureRules ) > 0 ) {
    return createWithQuadratureRuleImpl<OtherQuadratureRules...>( order, std::forward<Create>( create ) );
  } else {
    throw std::invalid_argument( "Requested quadrature order exceeds the supported quadrature rules." );
  }
}

}  // namespace detail

/**
 * @brief Select from explicitly supplied rule types in increasing polynomial degree order.
 *
 * @throws std::invalid_argument The order is negative or exceeds every supported rule.
 * Selection runs on the host; the selected type can be used in device and differentiated kernels.
 *
 * @tparam QuadratureRule First supported rule, with its exact polynomial degree.
 * @tparam OtherQuadratureRules Remaining rule types in increasing degree order.
 * @tparam Create Callable with `operator()<SelectedRule>()` and a common return type for all rules.
 * @param order Requested nonnegative reference polynomial degree.
 * @param create Callback invoked once as `operator()<SelectedRule>()`.
 * @return Callback result.
 */
template <typename QuadratureRule, typename... OtherQuadratureRules, typename Create>
decltype( auto ) createWithQuadratureRule( int order, Create&& create )
{
  if ( order < 0 ) throw std::invalid_argument( "Quadrature order must be nonnegative." );
  return detail::createWithQuadratureRuleImpl<QuadratureRule, OtherQuadratureRules...>(
      order, std::forward<Create>( create ) );
}

namespace detail {

/**
 * @brief Select a rule from a contiguous compile-time point-count range.
 *
 * @throws std::invalid_argument The order is negative or exceeds every supported rule.
 *
 * @tparam QuadratureRuleFamily Template accepting the number of points per base rule.
 * @tparam FirstPointCount First supported positive point count.
 * @tparam LastPointCount Last supported point count, inclusive.
 * @tparam Create Templated callback accepting the selected rule type.
 * @param order Requested nonnegative reference polynomial degree.
 * @param create Callback invoked once with the selected rule type.
 * @return Callback result.
 */
template <template <int> class QuadratureRuleFamily, int FirstPointCount, int LastPointCount, typename Create>
decltype( auto ) createWithQuadratureRuleRange( int order, Create&& create )
{
  static_assert( FirstPointCount > 0 && LastPointCount >= FirstPointCount );
  return [&]<int... Offsets>( std::integer_sequence<int, Offsets...> ) -> decltype( auto ) {
    return createWithQuadratureRule<QuadratureRuleFamily<FirstPointCount + Offsets>...>(
        order, std::forward<Create>( create ) );
  }( std::make_integer_sequence<int, LastPointCount - FirstPointCount + 1>{} );
}

/**
 * @brief Bind a projection type while leaving the Gauss point count selectable.
 *
 * @tparam Projection Projection value type retained by every rule in the family.
 */
template <typename Projection>
struct ProjectedEdgeRuleFamily {
  /**
   * @brief Rule type for one supported point count.
   *
   * @tparam NumPoints Number of Gauss points per base rule.
   */
  template <int NumPoints>
  using Rule = EdgeQuadratureRule<NumPoints, Projection>;
};

}  // namespace detail

/**
 * @brief Select a built-in unprojected edge rule by polynomial order.
 *
 * @throws std::invalid_argument The order is negative or greater than five.
 *
 * @tparam Create Callable invoked as `operator()<SelectedRule>()`.
 * @param order Requested degree 0-5, selecting one, two, or three Gauss points.
 * @param create Callback invoked once with the selected rule type.
 * @return Callback result.
 */
template <typename Create>
decltype( auto ) createWithQuadratureRule( int order, Create&& create )
{
  return detail::createWithQuadratureRuleRange<EdgeQuadratureRule, 1, 3>( order, std::forward<Create>( create ) );
}
/**
 * @brief Select an edge rule and construct it with the supplied projection.
 *
 * @throws std::invalid_argument The order is negative or greater than five.
 *
 * @tparam Projection Projection value type retained by the constructed rule.
 * @tparam Create Callable accepting the constructed rule by value.
 * @param order Requested degree 0-5, selecting one, two, or three Gauss points.
 * @param projection Configuration copied into the selected rule.
 * @param create Callback invoked once with the configured rule.
 * @return Callback result.
 */
template <typename Projection, typename Create>
decltype( auto ) createWithQuadratureRule( int order, const Projection& projection, Create&& create )
{
  return detail::createWithQuadratureRuleRange<detail::ProjectedEdgeRuleFamily<Projection>::template Rule, 1, 3>(
      order, [&]<typename QuadratureRule>() -> decltype( auto ) {
        return std::forward<Create>( create )( QuadratureRule{ projection } );
      } );
}

}  // namespace tribol

#endif
