// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Polynomial-order selection and construction of configured quadrature rule generators.
 */

#ifndef SRC_TRIBOL_INTEG_QUADRATURERULEGENERATORFACTORY_HPP_
#define SRC_TRIBOL_INTEG_QUADRATURERULEGENERATORFACTORY_HPP_

#include <stdexcept>
#include <utility>

#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"

namespace tribol {

/**
 * @brief Create a pair quadrature rule generator with the edge generator selected by polynomial order.
 * @throws std::invalid_argument The order is negative or unsupported by the available edge rules.
 * @tparam Mapping Forward and reverse mapping retained by the pair rule generator.
 * @tparam Create Callable accepting the composed pair rule generator by value.
 * @param order Requested nonnegative polynomial degree.
 * @param mapping Mapping object copied into the pair rule generator and used to construct the integration surface and
 * map its points to both elements.
 * @param create Callback invoked once with the selected pair rule generator; no quadrature points are generated here.
 * @return Callback result.
 */
template <typename Mapping, typename Create>
decltype( auto ) createWithQuadratureRuleGenerator( int order, const Mapping& mapping, Create&& create )
{
  using OnePointRuleGenerator = PairQuadratureRuleGenerator<Mapping, EdgeQuadratureRuleGenerator<1>>;
  using TwoPointRuleGenerator = PairQuadratureRuleGenerator<Mapping, EdgeQuadratureRuleGenerator<2>>;
  using ThreePointRuleGenerator = PairQuadratureRuleGenerator<Mapping, EdgeQuadratureRuleGenerator<3>>;

  if ( order < 0 ) {
    throw std::invalid_argument( "Quadrature order must be nonnegative." );
  }
  if ( order <= OnePointRuleGenerator::polynomial_order ) {
    return std::forward<Create>( create )( OnePointRuleGenerator{ mapping } );
  }
  if ( order <= TwoPointRuleGenerator::polynomial_order ) {
    return std::forward<Create>( create )( TwoPointRuleGenerator{ mapping } );
  }
  if ( order <= ThreePointRuleGenerator::polynomial_order ) {
    return std::forward<Create>( create )( ThreePointRuleGenerator{ mapping } );
  }
  throw std::invalid_argument( "Requested quadrature order exceeds the supported quadrature rules." );
}

}  // namespace tribol

#endif
