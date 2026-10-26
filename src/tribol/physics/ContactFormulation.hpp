// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_

#include "tribol/config.hpp"
#include <stdexcept>

#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"

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
 * This class provides a polymorphic interface for contact algorithms, allowing for modular implementation of new
 * physics and formulations. Mortar formulations use mesh 1 as the non-mortar integration surface and mesh 2 as the
 * mortar projection surface.
 *
 * Each stage receives current non-owning mesh views. Implementations retain their method and search configuration and
 * compute missing prerequisites so callers can request a stage without first calling earlier stages. The caller owns
 * the mesh arrays and must keep them alive for the duration of each request.
 */
class ContactFormulation {
 public:
  /**
   * @brief Virtual destructor
   */
  virtual ~ContactFormulation() = default;

  /**
   * @brief Updates the integration rule
   *
   * Determines overlapping contact pairs and computes necessary integration data (e.g. quadrature points, weights).
   * Implementations compute any missing pair data needed by this request. Some formulations may treat this as a no-op.
   *
   * @param mesh1 Current first mesh in Tribol coupling-scheme order
   * @param mesh2 Current second mesh in Tribol coupling-scheme order
   */
  virtual void updateIntegrationRule( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) = 0;

  /**
   * @brief Updates nodal gaps
   *
   * Implementations compute any missing pair or integration data needed by this request.
   *
   * @param mesh1 Current first mesh in Tribol coupling-scheme order
   * @param mesh2 Current second mesh in Tribol coupling-scheme order
   */
  virtual void updateNodalGaps( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) = 0;

  /**
   * @brief Updates nodal forces/residual
   *
   * Implementations compute any missing pair, integration, or gap data needed by this request.
   *
   * @param mesh1 Current first mesh in Tribol coupling-scheme order
   * @param mesh2 Current second mesh in Tribol coupling-scheme order
   */
  virtual void updateNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) = 0;

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

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_ */
