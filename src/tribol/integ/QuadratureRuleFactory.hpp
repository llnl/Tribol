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

#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/integ/PairQuadratureRule.hpp"

namespace tribol {

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
  using OnePointRule = EdgeQuadratureRuleGenerator<1>;
  using TwoPointRule = EdgeQuadratureRuleGenerator<2>;
  using ThreePointRule = EdgeQuadratureRuleGenerator<3>;

  if ( order < 0 ) {
    throw std::invalid_argument( "Quadrature order must be nonnegative." );
  }
  if ( order <= OnePointRule::polynomial_order ) {
    return std::forward<Create>( create ).template operator()<OnePointRule>();
  }
  if ( order <= TwoPointRule::polynomial_order ) {
    return std::forward<Create>( create ).template operator()<TwoPointRule>();
  }
  if ( order <= ThreePointRule::polynomial_order ) {
    return std::forward<Create>( create ).template operator()<ThreePointRule>();
  }
  throw std::invalid_argument( "Requested quadrature order exceeds the supported quadrature rules." );
}

/**
 * @brief Compose a supplied mapping with the edge rule generator selected by polynomial order.
 * @throws std::invalid_argument The order is negative or greater than five.
 * @tparam Mapping Forward and reverse mapping retained by the pair rule.
 * @tparam Create Callable accepting the composed rule by value.
 * @param order Requested degree 0-5, selecting one, two, or three Gauss points.
 * @param mapping Configuration copied into the composed rule.
 * @param create Callback invoked once with the configured pair rule.
 * @return Callback result.
 */
template <typename Mapping, typename Create>
decltype( auto ) createWithQuadratureRule( int order, const Mapping& mapping, Create&& create )
{
  return createWithQuadratureRule( order, [&]<typename IntegrationSurfaceRuleGenerator>() -> decltype( auto ) {
    return std::forward<Create>( create )( PairQuadratureRule<Mapping, IntegrationSurfaceRuleGenerator>{ mapping } );
  } );
}

}  // namespace tribol

#endif
