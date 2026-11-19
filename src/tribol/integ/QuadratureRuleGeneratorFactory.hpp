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

#include <utility>

#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "axom/slic/interface/slic.hpp"
#include "axom/slic/interface/slic_macros.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"

namespace tribol {

/**
 * @brief Create a pair quadrature rule generator with the rule generator selected by polynomial order.
 * @note Reports a SLIC error if the order is negative or unsupported by the available rules. Execution continues past
 * the error only when SLIC aborts are disabled; the nearest available rule is then used.
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

  // Execution continues past an unsupported-order error only when SLIC aborts are disabled; the nearest rule is used.
  SLIC_ERROR_IF( order < 0, "Quadrature order must be nonnegative." );
  if ( order <= OnePointRuleGenerator::polynomial_order ) {
    return std::forward<Create>( create )( OnePointRuleGenerator{ mapping } );
  }
  if ( order <= TwoPointRuleGenerator::polynomial_order ) {
    return std::forward<Create>( create )( TwoPointRuleGenerator{ mapping } );
  }
  SLIC_ERROR_IF( order > ThreePointRuleGenerator::polynomial_order,
                 "Requested quadrature order exceeds the supported quadrature rules." );
  return std::forward<Create>( create )( ThreePointRuleGenerator{ mapping } );
}

}  // namespace tribol

#endif
