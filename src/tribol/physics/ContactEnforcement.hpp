// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Fixed contact penalty stiffness and nodal enforcement values. */
#ifndef SRC_TRIBOL_PHYSICS_CONTACTENFORCEMENT_HPP_
#define SRC_TRIBOL_PHYSICS_CONTACTENFORCEMENT_HPP_

#include <cmath>
#include <stdexcept>
#include <variant>

#include "tribol/config.hpp"
#include "tribol/common/BasicTypes.hpp"

namespace tribol {
#ifdef BUILD_REDECOMP

/** @brief Select nodal Lagrange multiplier enforcement without penalty parameters. */
struct LagrangeMultiplierEnforcement {};

namespace detail {
/**
 * @brief Check an effective penalty before storing it in a formulation.
 * @param stiffness Combined mesh stiffness.
 * @return The supplied positive, finite stiffness.
 * @throws std::invalid_argument Stiffness is nonpositive or nonfinite; no value is returned.
 */
inline RealT checkedPenaltyStiffness( RealT stiffness )
{
  if ( !std::isfinite( stiffness ) || stiffness <= 0.0 ) {
    throw std::invalid_argument( "Contact penalty stiffness must be positive and finite." );
  }
  return stiffness;
}
}  // namespace detail

/** @brief Select nodal penalty enforcement with an effective stiffness fixed at construction. */
class PenaltyEnforcement {
 public:
  /**
   * @brief Store a fully configured effective penalty.
   * @param stiffness Combined mesh stiffness.
   * @throws std::invalid_argument Stiffness is nonpositive or nonfinite; no enforcement value is constructed.
   */
  explicit PenaltyEnforcement( RealT stiffness ) : stiffness_( detail::checkedPenaltyStiffness( stiffness ) ) {}

  /** @brief Read the fixed effective stiffness. @return Positive, finite stiffness supplied at construction. */
  RealT stiffness() const { return stiffness_; }

 private:
  const RealT stiffness_;  ///< Effective stiffness, unchanged by later updates to registered mesh data.
};

/** @brief Runtime selection between fully configured nodal penalty and Lagrange multiplier equations. */
using NodalContactEnforcement = std::variant<PenaltyEnforcement, LagrangeMultiplierEnforcement>;

#endif  // BUILD_REDECOMP
}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_CONTACTENFORCEMENT_HPP_
