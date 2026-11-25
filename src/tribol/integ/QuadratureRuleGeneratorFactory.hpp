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
#include "tribol/integ/PolygonQuadratureRuleGenerator.hpp"
#include "axom/slic/interface/slic.hpp"
#include "axom/slic/interface/slic_macros.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"

namespace tribol {

inline constexpr int minimum_supported_quadrature_order = 0;  ///< Smallest accepted polynomial order.
inline constexpr int maximum_supported_quadrature_order =
    EdgeQuadratureRuleGenerator<3>::polynomial_order;  ///< Largest order supported by the available edge/polygon rules.

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
  // Execution continues past an unsupported-order error only when SLIC aborts are disabled; the nearest rule is used.
  SLIC_ERROR_IF( order < minimum_supported_quadrature_order, "Quadrature order must be nonnegative." );
  if constexpr ( Mapping::Geometry1::parametric_dim == 1 ) {
    using OnePointRuleGenerator = PairQuadratureRuleGenerator<Mapping, EdgeQuadratureRuleGenerator<1>>;
    using TwoPointRuleGenerator = PairQuadratureRuleGenerator<Mapping, EdgeQuadratureRuleGenerator<2>>;
    using ThreePointRuleGenerator = PairQuadratureRuleGenerator<Mapping, EdgeQuadratureRuleGenerator<3>>;

    if ( order <= OnePointRuleGenerator::polynomial_order ) {
      return std::forward<Create>( create )( OnePointRuleGenerator{ mapping } );
    }
    if ( order <= TwoPointRuleGenerator::polynomial_order ) {
      return std::forward<Create>( create )( TwoPointRuleGenerator{ mapping } );
    }
    SLIC_ERROR_IF( order > maximum_supported_quadrature_order,
                   "Requested quadrature order exceeds the supported quadrature rules." );
    return std::forward<Create>( create )( ThreePointRuleGenerator{ mapping } );
  } else {
    static_assert( Mapping::Geometry1::parametric_dim == 2 );
    using CentroidRuleGenerator = PairQuadratureRuleGenerator<Mapping, PolygonCentroidQuadratureRuleGenerator>;
    using ThreePointTriRuleGenerator = PairQuadratureRuleGenerator<Mapping, PolygonQuadratureRuleGenerator<3>>;
    using SixPointTriRuleGenerator = PairQuadratureRuleGenerator<Mapping, PolygonQuadratureRuleGenerator<6>>;
    using SevenPointTriRuleGenerator = PairQuadratureRuleGenerator<Mapping, PolygonQuadratureRuleGenerator<7>>;

    if ( order <= CentroidRuleGenerator::polynomial_order ) {
      return std::forward<Create>( create )( CentroidRuleGenerator{ mapping } );
    }
    if ( order <= ThreePointTriRuleGenerator::polynomial_order ) {
      return std::forward<Create>( create )( ThreePointTriRuleGenerator{ mapping } );
    }
    if ( order <= SixPointTriRuleGenerator::polynomial_order ) {
      return std::forward<Create>( create )( SixPointTriRuleGenerator{ mapping } );
    }
    SLIC_ERROR_IF( order > maximum_supported_quadrature_order,
                   "Requested quadrature order exceeds the supported quadrature rules." );
    return std::forward<Create>( create )( SevenPointTriRuleGenerator{ mapping } );
  }
}

}  // namespace tribol

#endif
