// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_FORCEANDGAPMETHOD_HPP_
#define SRC_TRIBOL_PHYSICS_FORCEANDGAPMETHOD_HPP_

#include "tribol/common/BasicTypes.hpp"

#include <memory>

// Forward declarations for MFEM types
namespace mfem {
class Vector;
class HypreParMatrix;
class ParGridFunction;
}  // namespace mfem

namespace tribol {

/**
 * @brief Abstract base class for force and gap calculation methods
 *
 * @tparam IntegrationRule Policy type for integration
 * @tparam PointwiseGapAndNormal Policy type for gap/normal computation
 */
template <typename IntegrationRule, typename PointwiseGapAndNormal>
class ForceAndGapMethod {
 public:
  virtual ~ForceAndGapMethod() = default;

  /**
   * @brief Updates nodal gaps
   *
   * @param integration_rule Integration rule object
   * @param gap_method Pointwise gap and normal object
   */
  virtual void updateNodalGaps( IntegrationRule& integration_rule, PointwiseGapAndNormal& gap_method ) = 0;

  /**
   * @brief Updates nodal forces
   *
   * @param integration_rule Integration rule object
   * @param gap_method Pointwise gap and normal object
   */
  virtual void updateNodalForces( IntegrationRule& integration_rule, PointwiseGapAndNormal& gap_method ) = 0;

  /**
   * @brief Computes the maximum allowable timestep
   *
   * @param integration_rule Integration rule object
   * @param gap_method Pointwise gap and normal object
   * @return maximum allowable timestep
   */
  virtual RealT computeTimeStep( IntegrationRule& integration_rule, PointwiseGapAndNormal& gap_method ) = 0;

#ifdef BUILD_REDECOMP
  /**
   * @brief Adds computed forces to the provided MFEM vector
   *
   * @param [in,out] forces MFEM vector
   */
  virtual void getMfemForce( mfem::Vector& forces ) const = 0;

  /**
   * @brief Populates the provided MFEM vector with gap values
   *
   * @param [out] gaps MFEM vector
   */
  virtual void getMfemGap( mfem::Vector& gaps ) const = 0;

  /**
   * @brief Returns a reference to the MFEM pressure grid function
   *
   * @return mfem::ParGridFunction&
   */
  virtual mfem::ParGridFunction& getMfemPressure() = 0;

  /**
   * @brief Get the derivative of force with respect to displacement
   *
   * @return Unique pointer to MFEM HypreParMatrix
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> getMfemDfDx() const = 0;

  /**
   * @brief Get the derivative of gap with respect to displacement
   *
   * @return Unique pointer to MFEM HypreParMatrix
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> getMfemDgDx() const = 0;

  /**
   * @brief Get the derivative of force with respect to pressure
   *
   * @return Unique pointer to mfem::HypreParMatrix
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp() const = 0;
#endif
};

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_FORCEANDGAPMETHOD_HPP_ */
