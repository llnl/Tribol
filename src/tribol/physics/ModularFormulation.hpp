// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_MODULARFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_MODULARFORMULATION_HPP_

#include "tribol/physics/ContactFormulation.hpp"

namespace tribol {

template <typename IntegrationRule, typename PointwiseGapAndNormal, typename ForceAndGapMethod>
class ModularFormulation : public ContactFormulation {
 public:
  ModularFormulation( IntegrationRule&& integration_rule, PointwiseGapAndNormal&& pointwise_gap_and_normal,
                      ForceAndGapMethod&& force_and_gap_method )
      : integration_rule_( std::move( integration_rule ) ),
        pointwise_gap_and_normal_( std::move( pointwise_gap_and_normal ) ),
        force_and_gap_method_( std::move( force_and_gap_method ) )
  {
  }

  /**
   * @brief Sets the initial set of candidate interface pairs
   *
   * @param pairs View of the coarse-binned interface pairs
   * @param check_level In general, higher values mean more checks and 0 means don't do checks. See
   * IntegrationRule::findPairsInContact() for details.
   */
  void setInterfacePairs( ArrayT<InterfacePair>&& pairs, int check_level ) override;

  /**
   * @brief Updates the integration rule
   *
   * Determines overlapping contact pairs and computes necessary integration data (e.g. quadrature points, weights).
   *
   * @note Requires setInterfacePairs() to be called first.
   */
  void updateIntegrationRule() override;

  /**
   * @brief Updates nodal gaps
   *
   * @note Requires updateIntegrationRule() to be called first.
   */
  void updateNodalGaps() override;

  /**
   * @brief Updates nodal forces/residual
   *
   * @note Requires updateNodalGaps() to be called first.
   */
  void updateNodalForces() override;

  /**
   * @brief Computes the maximum allowable timestep for the formulation
   *
   * @return maximum allowable timestep
   */
  RealT computeTimeStep() override;

#ifdef BUILD_REDECOMP
  /**
   * @brief Adds computed forces to the provided MFEM vector
   *
   * @param [in,out] forces MFEM vector to add forces to
   *
   * @note Requires updateNodalForces() to be called first.
   */
  void getMfemForce( mfem::Vector& forces ) const override;

  /**
   * @brief Populates the provided MFEM vector with gap values
   *
   * Resizes the vector if necessary, zeros it out, and sets gap values.
   *
   * @param [out] gaps MFEM vector to store gaps in
   *
   * @note Requires updateNodalGaps() to be called first.
   */
  void getMfemGap( mfem::Vector& gaps ) const override;

  /**
   * @brief Returns a reference to the MFEM pressure grid function
   *
   * @return mfem::ParGridFunction& Reference to the pressure grid function
   */
  mfem::ParGridFunction& getMfemPressure() override;

  /**
   * @brief Get the derivative of force with respect to displacement
   *
   * @return Unique pointer to MFEM HypreParMatrix
   *
   * @note Requires updateNodalForces() to be called first.
   */
  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDx() const = 0;

  /**
   * @brief Get the derivative of gap with respect to displacement
   *
   * @return Unique pointer to MFEM HypreParMatrix
   *
   * @note Requires updateNodalGaps() to be called first.
   */
  std::unique_ptr<mfem::HypreParMatrix> getMfemDgDx() const override;

  /**
   * @brief Get the derivative of force with respect to pressure
   *
   * @return Unique pointer to mfem::HypreParMatrix
   *
   * @note Requires updateNodalForces() to be called first.
   */
  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp() const override;
#endif

 private:
  IntegrationRule integration_rule_;
  PointwiseGapAndNormal pointwise_gap_and_normal_;
  ForceAndGapMethod force_and_gap_method_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_MODULARFORMULATION_HPP_ */
