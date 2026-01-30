// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_INTEGRATIONRULE_HPP_
#define SRC_TRIBOL_INTEG_INTEGRATIONRULE_HPP_

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

namespace tribol {

/**
 * @brief Abstract base class for contact integration rules.
 *
 * This class defines the interface for integration rules used in contact
 * formulations. It serves as a blueprint for concrete integration strategies.
 *
 * @tparam PointwiseGapAndNormal Policy type for gap and normal computation
 * @tparam Dim Spatial dimensions of the integration rule (usually the dimension of the local surface coordinates)
 */
template <typename PointwiseGapAndNormal, int Dim>
class IntegrationRule {
 public:
  /**
   * @brief Virtual destructor to make the class polymorphic and abstract
   */
  virtual ~IntegrationRule() {};

  /**
   * @brief Identifies interface pairs that are in contact
   *
   * @param pairs Set of candidate interface pairs
   * @param check_level Level of geometric checks to perform
   * @param gap_method Method object for gap/normal calculations
   */
  virtual void findPairsInContact( ArrayT<InterfacePair>&& pairs, int check_level,
                                   PointwiseGapAndNormal& gap_method ) = 0;

  /**
   * @brief Updates the integration rule (e.g., computes quadrature points and weights)
   *
   * @tparam PointwiseGapAndNormal Policy type for gap and normal computation
   * @param gap_method Method object for gap/normal calculations
   */
  virtual void updateRule( PointwiseGapAndNormal& gap_method ) = 0;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_INTEG_INTEGRATIONRULE_HPP_ */
