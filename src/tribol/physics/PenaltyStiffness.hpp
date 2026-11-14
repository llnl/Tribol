// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Combine two supplied mesh penalty stiffnesses independently of contact geometry and enforcement.
 */
#ifndef SRC_TRIBOL_PHYSICS_PENALTYSTIFFNESS_HPP_
#define SRC_TRIBOL_PHYSICS_PENALTYSTIFFNESS_HPP_

#include <cmath>
#include <stdexcept>

#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Enzyme.hpp"

namespace tribol {
namespace detail {
/**
 * @brief Check a mesh or effective penalty before constructing a formulation.
 * @param stiffness Mesh or effective stiffness.
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

/** @brief Combine the two mesh stiffnesses using their arithmetic mean. */
struct AveragePenaltyStiffness {
  /**
   * @brief Return the average stiffness without applying geometry or mesh penalty scales.
   * @param stiffness1 First supplied stiffness.
   * @param stiffness2 Second supplied stiffness.
   * @pre Both stiffnesses are positive and finite; the caller validates its inputs and applies any mesh scales.
   * @return 0.5 * (stiffness1 + stiffness2). This operation performs no validation or allocation.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT operator()( RealT stiffness1, RealT stiffness2 ) const
  {
    return 0.5 * ( stiffness1 + stiffness2 );
  }
};

/** @brief Combine the two mesh stiffnesses as springs in series. */
struct SeriesPenaltyStiffness {
  /**
   * @brief Return the series stiffness without applying geometry or mesh penalty scales.
   * @param stiffness1 First supplied stiffness.
   * @param stiffness2 Second supplied stiffness.
   * @pre Both stiffnesses are positive and finite; the caller validates its inputs and applies any mesh scales.
   * @return stiffness1 * stiffness2 / (stiffness1 + stiffness2). No validation or allocation is performed.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT operator()( RealT stiffness1, RealT stiffness2 ) const
  {
    return stiffness1 * stiffness2 / ( stiffness1 + stiffness2 );
  }
};

}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_PENALTYSTIFFNESS_HPP_
