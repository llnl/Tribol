// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/// @file
/// Evaluate each contact method from complete current inputs without caller-managed preparation phases.

#ifndef SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_

#include "tribol/config.hpp"
#include <stdexcept>

#include "tribol/common/Parameters.hpp"
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

#include <memory>

// Forward declarations for MFEM types
namespace mfem {
class Vector;
class HypreParMatrix;
class HypreParVector;
class ParGridFunction;
}  // namespace mfem

namespace tribol {

// Forward declaration
class MethodData;
class MeshData;

/*!
 * \brief Base class for contact formulations.
 *
 * This class provides a polymorphic interface for contact algorithms, allowing for modular implementation of new
 * physics and formulations.
 *
 * Evaluation receives current inputs together so callers do not need a separate preparation sequence.
 * Solver-owned multipliers remain inputs; result accessors expose the last completed evaluation.
 */
class ContactFormulation {
 public:
  /**
   * @brief Virtual destructor
   */
  virtual ~ContactFormulation() = default;

  /**
   * @brief Evaluate gaps, forces, energy, and Jacobians from the current inputs.
   *
   * Meshes, candidate pairs, settings, and multiplier inputs are supplied together.
   * Input views are used only during this call. Preparation is internal to evaluation. The optional multiplier vector
   * is solver-owned input, never modified by the formulation. Result accessors describe the last completed evaluation.
   */
  virtual void evaluate( MeshData& mesh1, MeshData& mesh2, ArrayViewT<const InterfacePair> pairs, int cycle,
                         const Parameters& parameters, const mfem::HypreParVector* multipliers = nullptr ) = 0;

  /**
   * @brief Reports if formulation has a maximum allowable timestep calculation
   *
   * @return true if formulation has a timestep calculation available; false otherwise
   */
  virtual bool hasTimeStepCalculation() = 0;

  /**
   * @brief Computes the maximum allowable timestep for the formulation
   *
   * @return maximum allowable timestep
   */
  virtual RealT computeTimeStep() { return 1.0; };

  /**
   * @brief Returns the energy stored by the contact constraints (if supported by the method)
   *
   * @note Requires evaluate() to be called first.
   *
   * @return contact energy
   */
  virtual RealT getEnergy() const = 0;

  /**
   * @brief Update constant penalty stiffness
   *
   * Used to dynamically update the constant penalty stiffness after the formulation has been instantiated.
   */
  virtual void updateConstantPenaltyStiffness( double /*mesh1_penalty*/, double /*mesh2_penalty*/ ) {}

#ifdef BUILD_REDECOMP
  /**
   * @brief Returns t-dof vector of forces on parent mesh
   *
   * @note Requires evaluate() to be called first.
   */
  virtual const mfem::HypreParVector& getMfemForce() const
  {
    SLIC_ERROR_ROOT( "getMfemForce() is not supported by this formulation." );
    throw std::runtime_error( "Not supported" );
  }

  /**
   * @brief Returns t-dof vector of gaps on submesh
   *
   * @note Requires evaluate() to be called first.
   */
  virtual const mfem::HypreParVector& getMfemGap() const
  {
    SLIC_ERROR_ROOT( "getMfemGap() is not supported by this formulation." );
    throw std::runtime_error( "Not supported" );
  }

  /**
   * @brief Get the derivative of force with respect to displacement
   *
   * @return Unique pointer to MFEM HypreParMatrix
   *
   * @note Requires evaluate() to be called first.
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> getMfemDfDx() const
  {
    SLIC_ERROR_ROOT( "getMfemDfDx() is not supported by this formulation." );
    return nullptr;
  }

  /**
   * @brief Get the derivative of the gap constraint with respect to displacement
   *
   * @return Unique pointer to MFEM HypreParMatrix
   *
   * @note Requires evaluate() to be called first.
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> getMfemDgDx() const
  {
    SLIC_ERROR_ROOT( "getMfemDgDx() is not supported by this formulation." );
    return nullptr;
  }

  /**
   * @brief Get the derivative of force with respect to the dual variable
   *
   * @return Unique pointer to mfem::HypreParMatrix
   *
   * @note Requires evaluate() to be called first.
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp() const
  {
    SLIC_ERROR_ROOT( "getMfemDfDp() is not supported by this formulation." );
    return nullptr;
  }

#endif
};

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_ */
