// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

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
 * Formulations own their search lifecycle. Current adapters may still use a CouplingScheme as a compatibility context,
 * but search implementations accept mesh and execution data directly.
 */
class ContactFormulation {
 public:
  /**
   * @brief Virtual destructor
   */
  virtual ~ContactFormulation() = 0;

  /**
   * @brief Optionally sets externally supplied interface pairs
   *
   * Surface formulations may override this compatibility hook. Formulations
   * using another contact representation do not need to implement it.
   *
   * @param pairs Coarse candidate interface pairs
   * @param check_level In general, higher values mean more checks and 0 means don't do checks. See specific methods for
   * details.
   */
  virtual void setInterfacePairs( ArrayT<InterfacePair>&& /*pairs*/, int /*check_level*/ )
  {
    SLIC_ERROR_ROOT( "setInterfacePairs() is not supported by this formulation." );
  }

  /**
   * @brief Updates the formulation-specific contact search
   *
   * Formulations without a search may use the default no-op implementation.
   */
  virtual void updateSearch() {}

  /**
   * @brief Updates the integration rule
   *
   * Determines overlapping contact pairs and computes necessary integration data (e.g. quadrature points, weights).
   *
   * @note Formulations are responsible for updating their search state first.
   */
  virtual void updateIntegrationRule() {}

  /**
   * @brief Updates nodal gaps
   *
   * @note Many formulations require updateIntegrationRule() to be called first.
   * Some formulations assemble gaps directly from the stored interface pairs.
   */
  virtual void updateNodalGaps() {}

  /**
   * @brief Updates nodal forces/residual
   *
   * @note Many formulations require updateNodalGaps() to be called first.
   */
  virtual void updateNodalForces() {}

  /**
   * @brief Reports if formulation has a maximum allowable timestep calculation
   *
   * @return true if formulation has a timestep calculation available; false otherwise
   */
  virtual bool hasTimeStepCalculation() { return false; }

  /**
   * @brief Computes the maximum allowable timestep for the formulation
   *
   * @return maximum allowable timestep
   */
  virtual RealT computeTimeStep() { return 1.0; };

  /**
   * @brief Returns the energy stored by the contact constraints (if supported by the method)
   *
   * @note Requires updateNodalForces() to be called first.
   *
   * @return contact energy
   */
  virtual RealT getEnergy() const
  {
    SLIC_ERROR_ROOT( "getEnergy() is not implemented by this formulation." );
    throw std::runtime_error( "Not supported" );
  }

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
   * @brief Update constant penalty stiffness
   *
   * Used to dynamically update the constant penalty stiffness after the formulation has been instantiated.
   */
  virtual void updateConstantPenaltyStiffness( double /*mesh1_penalty*/, double /*mesh2_penalty*/ ) {}

#ifdef BUILD_REDECOMP
  /**
   * @brief Returns t-dof vector of forces on parent mesh
   *
   * @note Requires updateNodalForces() to be called first.
   */
  virtual const mfem::HypreParVector& getMfemForce() const
  {
    SLIC_ERROR_ROOT( "getMfemForce() is not supported by this formulation." );
    throw std::runtime_error( "Not supported" );
  }

  /**
   * @brief Returns t-dof vector of gaps on submesh
   *
   * @note Requires updateNodalGaps() to be called first.
   */
  virtual const mfem::HypreParVector& getMfemGap() const
  {
    SLIC_ERROR_ROOT( "getMfemGap() is not supported by this formulation." );
    throw std::runtime_error( "Not supported" );
  }

  /**
   * @brief Returns a reference to the MFEM dual t-dof vector on the submesh
   *
   * @return Reference to the dual t-dof vector (e.g. pressure in penalty mode, or Lagrange multiplier in LM mode)
   */
  virtual mfem::HypreParVector& getMfemPressure()
  {
    SLIC_ERROR_ROOT( "getMfemPressure() is not supported by this formulation." );
    throw std::runtime_error( "Not supported" );
  }

  /**
   * @brief Get the derivative of force with respect to displacement
   *
   * @return Unique pointer to MFEM HypreParMatrix
   *
   * @note Requires updateNodalForces() to be called first.
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
   * @note Requires updateNodalGaps() to be called first.
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
   * @note Requires updateNodalForces() to be called first.
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp() const
  {
    SLIC_ERROR_ROOT( "getMfemDfDp() is not supported by this formulation." );
    return nullptr;
  }

#endif
};

inline ContactFormulation::~ContactFormulation() = default;

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_ */
