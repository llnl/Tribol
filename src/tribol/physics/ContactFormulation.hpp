// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_

#include "tribol/common/Parameters.hpp"
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"

#include <memory>

// Forward declarations for MFEM types
namespace mfem {
class Vector;
class HypreParMatrix;
class ParGridFunction;
}

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
   * @brief Constructor
   *
   * @param mesh1 Reference to the first mesh
   * @param mesh2 Reference to the second mesh
   */
  ContactFormulation( MeshData& mesh1, MeshData& mesh2 )
      : mesh1_( mesh1 ), mesh2_( mesh2 )
  {
  }

  /**
   * @brief Virtual destructor
   */
  virtual ~ContactFormulation() = default;

  /**
   * @brief Checks if the necessary data is defined on the meshes
   *
   * @note This is a static interface method that should be implemented by
   *       derived classes to validate input before instantiation.
   *
   * @param mesh1 Reference to the first mesh
   * @param mesh2 Reference to the second mesh
   * @param params Coupling scheme parameters
   * @return 0 if valid, non-zero error code otherwise
   */
  static int checkData( MeshData& mesh1, MeshData& mesh2, const Parameters& params );

  /**
   * @brief Sets the initial set of candidate interface pairs
   *
   * @param pairs View of the coarse-binned interface pairs
   */
  virtual void setInterfacePairs( ArrayViewT<InterfacePair> pairs ) = 0;

  /**
   * @brief Updates the integration rule
   *
   * Determines the active set of contact pairs and computes necessary
   * integration data (e.g. quadrature points, weights).
   */
  virtual void updateIntegrationRule() = 0;

  /**
   * @brief Updates nodal gaps
   *
   * @note Requires initialize() to be called first to register meshes.
   */
  virtual void updateNodalGaps() = 0;

  /**
   * @brief Updates nodal forces/residual
   *
   * @note Requires initialize() to be called first to register meshes.
   */
  virtual void updateNodalForces() = 0;

  /**
   * @brief Updates nodal energies
   *
   * @note Requires initialize() to be called first to register meshes.
   */
  virtual void updateNodalEnergies() = 0;

  /**
   * @brief Computes the maximum allowable timestep for the formulation
   *
   * @return maximum allowable timestep
   */
  virtual RealT computeTimeStep() = 0;

  /**
   * @brief Get read-only view of computed nodal gaps
   *
   * @return ArrayViewT<const RealT> View of gaps
   */
  virtual ArrayViewT<const RealT> getGaps() const = 0;

  /**
   * @brief Get read-only view of computed nodal forces
   *
   * @return ArrayViewT<const RealT> View of forces
   */
  virtual ArrayViewT<const RealT> getForces() const = 0;

  /**
   * @brief Get read-only view of pressures
   *
   * @return ArrayViewT<const RealT> View of pressures
   */
  virtual ArrayViewT<const RealT> getPressure() const = 0;

  /**
   * @brief Get pointer to Jacobian data
   *
   * @return MethodData* Pointer to method data containing Jacobian
   */
  virtual MethodData* getJacobian() const = 0;

#ifdef BUILD_REDECOMP
  /**
   * @brief Adds computed forces to the provided MFEM vector
   *
   * @param [in,out] forces MFEM vector to add forces to
   */
  virtual void getMfemForces( mfem::Vector& forces ) const = 0;

  /**
   * @brief Populates the provided MFEM vector with gap values
   *
   * Resizes the vector if necessary, zeros it out, and sets gap values.
   *
   * @param [out] gaps MFEM vector to store gaps in
   */
  virtual void getMfemGap( mfem::Vector& gaps ) const = 0;

  /**
   * @brief Returns a reference to the MFEM pressure grid function
   *
   * @return mfem::ParGridFunction& Reference to the pressure grid function
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
   * @return Unique pointer to MFEM HypreParMatrix
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp() const = 0;
#endif

 protected:
  MeshData& mesh1_;        ///< Reference to the first mesh
  MeshData& mesh2_;        ///< Reference to the second mesh

};

} // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_ */
