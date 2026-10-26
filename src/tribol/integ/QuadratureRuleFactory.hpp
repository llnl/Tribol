// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_QUADRATURERULEFACTORY_HPP_
#define SRC_TRIBOL_INTEG_QUADRATURERULEFACTORY_HPP_

#include <stdexcept>
#include <utility>

#include "tribol/integ/EdgeQuadratureRule.hpp"

namespace tribol {
namespace detail {

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
 * @brief Select a rule type by polynomial order and pass it to a templated factory callback
 *
 * Supply rule types from least to most accurate. Each type exposes polynomial_order, the reference polynomial degree
 * it integrates exactly. The first adequate type is passed to create.template operator()<QuadratureRule>(). The
 * callback runs once and must return the same type for every supported rule, such as unique_ptr<ContactFormulation>.
 * Negative or unsupported orders throw invalid_argument without calling the callback. Selection runs on the host;
 * the selected type can then be used in device or differentiated kernels without a runtime rule selection there.
 */
template <typename QuadratureRule, typename... OtherQuadratureRules, typename Create>
decltype( auto ) createWithQuadratureRule( int order, Create&& create )
{
  if ( order < 0 ) throw std::invalid_argument( "Quadrature order must be nonnegative." );
  return detail::createWithQuadratureRuleImpl<QuadratureRule, OtherQuadratureRules...>(
      order, std::forward<Create>( create ) );
}

namespace detail {

/** @brief Generate rule types for a contiguous point-count range before selecting by polynomial order */
template <template <int> class QuadratureRuleFamily, int FirstPointCount, int LastPointCount, typename Create>
decltype( auto ) createWithQuadratureRuleRange( int order, Create&& create )
{
  static_assert( FirstPointCount > 0 && LastPointCount >= FirstPointCount );
  return [&]<int... Offsets>( std::integer_sequence<int, Offsets...> ) -> decltype( auto ) {
    return createWithQuadratureRule<QuadratureRuleFamily<FirstPointCount + Offsets>...>(
        order, std::forward<Create>( create ) );
  }( std::make_integer_sequence<int, LastPointCount - FirstPointCount + 1>{} );
}

}  // namespace detail

/**
 * @brief Select a built-in edge Gauss rule by polynomial order and invoke the construction callback
 *
 * Orders 0-5 select one, two, or three Gauss-Legendre points. The factory generates the supported types from a
 * compile-time point-count range, so callers provide only the requested order and a templated callback. The
 * explicit-type overload also supports families with noncontiguous point counts. Interval selection and target
 * projection are independent of the selected point count.
 */
template <typename Create>
decltype( auto ) createWithQuadratureRule( int order, Create&& create )
{
  return detail::createWithQuadratureRuleRange<EdgeQuadratureRule, 1, 3>( order, std::forward<Create>( create ) );
}

}  // namespace tribol

#endif
