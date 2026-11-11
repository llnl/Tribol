// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Hold integration-surface points fixed for pair-function derivative tests. */
#ifndef SRC_TESTS_FROZENPAIRQUADRATURERULE_HPP_
#define SRC_TESTS_FROZENPAIRQUADRATURERULE_HPP_

#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdSpan.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {
namespace detail {

/**
 * @brief Freeze reference points and weights while retaining current-geometry projections and physical Jacobians.
 * @tparam QuadratureRule Original pair rule providing integration-surface generation and reverse mapping.
 */
template <typename QuadratureRule>
class FrozenPairQuadratureRule {
 public:
  using FirstGeometry = typename QuadratureRule::FirstGeometry;    ///< First element geometry.
  using SecondGeometry = typename QuadratureRule::SecondGeometry;  ///< Second element geometry.
  static constexpr IndexT coordinate_count =
      FirstGeometry::Coords::size() + SecondGeometry::Coords::size();  ///< Coordinates in the complete pair.

  /**
   * @brief Generate and own a fixed reference rule from the supplied pair coordinates.
   * @param rule Pair quadrature configuration copied into this object.
   * @param coordinates Current coordinates ordered by element, vertex, and spatial component; not retained.
   * @throws std::invalid_argument Pair geometry is invalid; no rule is constructed.
   * @throws std::length_error Reference quadrature capacity is insufficient; no rule is constructed.
   */
  FrozenPairQuadratureRule( const QuadratureRule& rule, MdSpan<const RealT, coordinate_count> coordinates )
      : rule_( rule ),
        reference_( rule_.generateIntegrationSurfaceRule(
            { coordinates.data(), FirstGeometry::Coords::size() },
            { coordinates.data() + FirstGeometry::Coords::size(), SecondGeometry::Coords::size() } ) )
  {
    if ( reference_.result.status != QuadratureStatus::Success ) {
      reportQuadratureFailure( reference_.result.status );
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
    return rule_.mapIntegrationSurfaceRuleToElements( geometry1, geometry2, reference_.getView() );
  }

 private:
  QuadratureRule rule_;                                        ///< Owned reverse-mapping configuration.
  typename QuadratureRule::IntegrationSurfaceRule reference_;  ///< Owned fixed reference points and weights.
};

}  // namespace detail
}  // namespace tribol

#endif  // SRC_TESTS_FROZENPAIRQUADRATURERULE_HPP_
