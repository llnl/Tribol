// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_INTEGRATIONRULE_HPP_
#define SRC_TRIBOL_INTEG_INTEGRATIONRULE_HPP_

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

namespace tribol {

struct IntegrationPointAndWeight {
  Array1D<RealT> point1_;
  Array1D<RealT> point2_;
  RealT weight_;
};

/**
 * @brief Struct to hold integration points and weights for a single contact pair.
 *
 * Stores integration points on both surfaces of the contact interface.
 */
struct IntegrationPoints {
  InterfacePair pair_;
  Array1D<IntegrationPointAndWeight> points_;
};

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

  /**
   * @brief Returns the computed integration points.
   */
  virtual const ArrayT<IntegrationPoints>& getRule() const = 0;

  /**
   * @brief Returns the active interface pairs.
   */
  virtual const ArrayT<InterfacePair>& getPairs() const = 0;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_INTEG_INTEGRATIONRULE_HPP_ */
