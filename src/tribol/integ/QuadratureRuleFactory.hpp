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
 * @brief Create a pair quadrature rule with the edge generator selected by polynomial order.
 * @throws std::invalid_argument The order is negative or unsupported by the available edge rules.
 * @tparam Mapping Forward and reverse mapping retained by the pair rule.
 * @tparam Create Callable accepting the composed rule by value.
 * @param order Requested nonnegative polynomial degree.
 * @param mapping Mapping object copied into the pair rule and used to construct the integration surface and map its
 * points to both elements.
 * @param create Callback invoked once with the selected pair rule.
 * @return Callback result.
 */
template <typename Mapping, typename Create>
decltype( auto ) createWithQuadratureRule( int order, const Mapping& mapping, Create&& create )
{
  using OnePointRule = PairQuadratureRule<Mapping, EdgeQuadratureRuleGenerator<1>>;
  using TwoPointRule = PairQuadratureRule<Mapping, EdgeQuadratureRuleGenerator<2>>;
  using ThreePointRule = PairQuadratureRule<Mapping, EdgeQuadratureRuleGenerator<3>>;

  if ( order < 0 ) {
    throw std::invalid_argument( "Quadrature order must be nonnegative." );
  }
  if ( order <= OnePointRule::polynomial_order ) {
    return std::forward<Create>( create )( OnePointRule{ mapping } );
  }
  if ( order <= TwoPointRule::polynomial_order ) {
    return std::forward<Create>( create )( TwoPointRule{ mapping } );
  }
  if ( order <= ThreePointRule::polynomial_order ) {
    return std::forward<Create>( create )( ThreePointRule{ mapping } );
  }
  throw std::invalid_argument( "Requested quadrature order exceeds the supported quadrature rules." );
}

}  // namespace tribol

#endif
