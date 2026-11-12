// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Generate pair quadrature using fixed integration-surface points for derivative tests. */
#ifndef SRC_TESTS_FROZENPAIRQUADRATURERULEGENERATOR_HPP_
#define SRC_TESTS_FROZENPAIRQUADRATURERULEGENERATOR_HPP_

#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdSpan.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {
namespace detail {

/**
 * @brief Generate pair rules with fixed reference points and weights and current-geometry projections and physical
 * Jacobians.
 * @tparam PairRuleGenerator Original pair generator providing integration-surface generation and reverse mapping.
 */
template <typename PairRuleGenerator>
class FrozenPairQuadratureRuleGenerator {
 public:
  using FirstGeometry = typename PairRuleGenerator::FirstGeometry;    ///< First element geometry.
  using SecondGeometry = typename PairRuleGenerator::SecondGeometry;  ///< Second element geometry.
  static constexpr IndexT coordinate_count =
      FirstGeometry::Coords::size() + SecondGeometry::Coords::size();  ///< Coordinates in the complete pair.

  /**
   * @brief Generate and own a fixed reference rule from the supplied pair coordinates.
   * @param rule_generator Pair quadrature rule generator copied into this object.
   * @param coordinates Current coordinates ordered by element, vertex, and spatial component; not retained.
   * @throws std::invalid_argument Pair geometry is invalid; no generator is constructed.
   * @throws std::length_error Reference quadrature capacity is insufficient; no generator is constructed.
   */
  FrozenPairQuadratureRuleGenerator( const PairRuleGenerator& rule_generator,
                                     MdSpan<const RealT, coordinate_count> coordinates )
      : rule_generator_( rule_generator ),
        reference_rule_( rule_generator_.generateIntegrationSurfaceRule(
            { coordinates.data(), FirstGeometry::Coords::size() },
            { coordinates.data() + FirstGeometry::Coords::size(), SecondGeometry::Coords::size() } ) )
  {
    if ( reference_rule_.result.status != QuadratureStatus::Success ) {
      reportQuadratureFailure( reference_rule_.result.status );
    }
  }

  /**
   * @brief Reverse-map the stored reference points using current element geometry.
   * @param geometry1 Current first element.
   * @param geometry2 Current second element.
   * @return Paired coordinates, physical weights, and generation status. Invalid geometry reports zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto generate( const FirstGeometry& geometry1,
                                                         const SecondGeometry& geometry2 ) const
  {
    return rule_generator_.mapIntegrationSurfaceRuleToElements( geometry1, geometry2, reference_rule_.getView() );
  }

 private:
  PairRuleGenerator rule_generator_;                                   ///< Owned reverse-mapping configuration.
  typename PairRuleGenerator::IntegrationSurfaceRule reference_rule_;  ///< Owned fixed reference points and weights.
};

}  // namespace detail
}  // namespace tribol

#endif  // SRC_TESTS_FROZENPAIRQUADRATURERULEGENERATOR_HPP_
