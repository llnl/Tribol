// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_

#include "tribol/config.hpp"

#include "tribol/common/Parameters.hpp"
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

#include <memory>

// Forward declarations for MFEM types
namespace mfem {
class Vector;
class HypreParMatrix;
class ParGridFunction;
}  // namespace mfem

namespace tribol {

// Forward declaration
class MethodData;

/*!
 * \brief Base class for contact formulations.
 *
 * This class provides a polymorphic interface for contact algorithms,
 * allowing for modular implementation of new physics and formulations.
 */
class ContactFormulation {
 public:
  /**
   * @brief Virtual destructor
   */
  virtual ~ContactFormulation() = default;

  /**
   * @brief Sets the initial set of candidate interface pairs
   *
   * @param pairs View of the coarse-binned interface pairs
   * @param check_level In general, higher values mean more checks and 0 means don't do checks. See specific methods for
   * details.
   */
  virtual void setInterfacePairs( ArrayT<InterfacePair>&& pairs, int check_level ) = 0;

  /**
   * @brief Updates the integration rule
   *
   * Determines overlapping contact pairs and computes necessary integration data (e.g. quadrature points, weights).
   *
   * @note Requires setInterfacePairs() to be called first.
   */
  virtual void updateIntegrationRule() = 0;

  /**
   * @brief Updates nodal gaps
   *
   * @note Requires updateIntegrationRule() to be called first.
   */
  virtual void updateNodalGaps() = 0;

  /**
   * @brief Updates nodal forces/residual
   *
   * @note Requires updateNodalGaps() to be called first.
   */
  virtual void updateNodalForces() = 0;

  /**
   * @brief Computes the maximum allowable timestep for the formulation
   *
   * @return maximum allowable timestep
   */
  virtual RealT computeTimeStep() = 0;

  /**
   * @brief Returns the energy stored by the contact constraints (if supported by the method)
   *
   * @note Requires updateNodalForces() to be called first.
   *
   * @return contact energy
   */
  virtual RealT getEnergy() const = 0;

#ifdef BUILD_REDECOMP
  /**
   * @brief Returns t-dof vector of forces on parent mesh
   *
   * @note Requires updateNodalForces() to be called first.
   */
  virtual const mfem::HypreParVector& getMfemForce() const = 0;

  /**
   * @brief Returns t-dof vector of gaps on submesh
   *
   * @note Requires updateNodalGaps() to be called first.
   */
  virtual const mfem::HypreParVector& getMfemGap() const = 0;

  /**
   * @brief Returns a reference to the MFEM pressure t-dof vector
   *
   * @return Reference to the pressure t-dof vector
   */
  virtual mfem::HypreParVector& getMfemPressure() = 0;

  /**
   * @brief Get the derivative of force with respect to displacement
   *
   * @return Unique pointer to MFEM HypreParMatrix
   *
   * @note Requires updateNodalForces() to be called first.
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> getMfemDfDx() const = 0;

  /**
   * @brief Get the derivative of gap with respect to displacement
   *
   * @return Unique pointer to MFEM HypreParMatrix
   *
   * @note Requires updateNodalGaps() to be called first.
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> getMfemDgDx() const = 0;

  /**
   * @brief Get the derivative of force with respect to pressure
   *
   * @return Unique pointer to mfem::HypreParMatrix
   *
   * @note Requires updateNodalForces() to be called first.
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp() const = 0;

  virtual void evaluateContactResidual( const mfem::HypreParVector& lambda,
                                        mfem::HypreParVector& r_force,
                                        mfem::HypreParVector& r_gap ) = 0;

  virtual void evaluateContactJacobian( const mfem::HypreParVector& lambda,
                                        std::unique_ptr<mfem::HypreParMatrix>& df_du,
                                        std::unique_ptr<mfem::HypreParMatrix>& df_dlambda ) = 0;
#endif
};

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_ */