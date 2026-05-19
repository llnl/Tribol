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
class MeshData;

/*!
 * \brief Base class for contact formulations.
 *
 * This class provides a polymorphic interface for contact algorithms, allowing for modular implementation of new
 * physics and formulations.
 *
 * NOTE (EBC): This class is still a work-in-progress. It's eventual design should incorporate specific choices for
 * coarse binning and take in the relevant input data in the constructor directly (i.e. Tribol mesh data, mfem meshes,
 * etc.)
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
   * @note Many formulations require setInterfacePairs() to be called first.
   * Some formulations may treat this as a no-op.
   */
  virtual void updateIntegrationRule() = 0;

  /**
   * @brief Updates nodal gaps
   *
   * @note Many formulations require updateIntegrationRule() to be called first.
   * Some formulations assemble gaps directly from the stored interface pairs.
   */
  virtual void updateNodalGaps() = 0;

  /**
   * @brief Updates nodal forces/residual
   *
   * @note Many formulations require updateNodalGaps() to be called first.
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

  /**
   * @brief Update mesh references used by the formulation (optional)
   *
   * Some workflows re-register meshes each cycle (e.g. MFEM + redecomp), which can replace the underlying MeshData
   * instances. Formulations that cache mesh pointers/references can override this to refresh their internal handles.
   *
   * @note Called by CouplingScheme during initialization each update cycle.
   */
  virtual void updateMeshes( MeshData& /*mesh1*/, MeshData& /*mesh2*/ ) {}

  /**
   * @brief Update penalty parameters
   *
   * Used to dynamically update penalty stiffness after the formulation has been instantiated.
   */
  virtual void updatePenaltyParameters( bool /*use_penalty*/, double /*k*/ ) {}

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
   * @brief Returns a reference to the MFEM dual t-dof vector
   *
   * @return Reference to the dual t-dof vector (e.g. pressure in penalty mode, or Lagrange multiplier in LM mode)
   *
   * TODO: specify what mesh object this is define on.
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
   * @brief Get the derivative of the gap constraint with respect to displacement
   *
   * @return Unique pointer to MFEM HypreParMatrix
   *
   * @note Requires updateNodalGaps() to be called first.
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> getMfemDgDx() const = 0;

  /**
   * @brief Get the derivative of force with respect to the dual variable
   *
   * @return Unique pointer to mfem::HypreParMatrix
   *
   * @note Requires updateNodalForces() to be called first.
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp() const = 0;

#endif
};

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_ */

