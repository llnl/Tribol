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
   * @brief Returns formulation-owned interface pairs when the formulation stores them
   *
   * Some formulations take ownership of the candidate pairs from the coupling scheme. Diagnostics can use this accessor
   * to report the pair set currently used by the formulation.
   *
   * @return Pointer to stored interface pairs, or nullptr if the formulation does not store them
   */
  virtual const ArrayT<InterfacePair>* getStoredInterfacePairs() const { return nullptr; }

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

  /**
   * @brief Update ENERGY_MORTAR normal-mode parameters on formulations that support them
   */
  virtual void updateEnergyMortarNormalMode( EnergyMortarNormalMode /*normal_mode*/, bool /*projection_smoothing*/ ) {}

  /**
   * @brief Update ENERGY_MORTAR projection smoothing curve on formulations that support it
   */
  virtual void updateEnergyMortarProjectionSmoothingCurve( EnergyMortarProjectionSmoothingCurve /*curve*/ ) {}

  /**
   * @brief Update ENERGY_MORTAR quadrature differentiation mode on formulations that support it
   */
  virtual void updateEnergyMortarEnzymeQuadrature( bool /*enabled*/ ) {}

  /**
   * @brief Update ENERGY_MORTAR integration-measure differentiation mode on formulations that support it
   */
  virtual void updateEnergyMortarFixedIntegrationJacobian( bool /*enabled*/ ) {}

  /**
   * @brief Update ENERGY_MORTAR active-set smoothing gap on formulations that support it
   */
  virtual void updateEnergyMortarH1ActiveSetSmoothing( RealT /*gap_transition*/ ) {}

  /**
   * @brief Update ENERGY_MORTAR QP penalty derivative-blending transition gap on formulations that support it
   */
  virtual void updateEnergyMortarQpDerivativeBlendGap( RealT gap_transition )
  {
    updateEnergyMortarQpDerivativeBlendGapRange( 0.0, gap_transition );
  }

  /**
   * @brief Update ENERGY_MORTAR QP penalty derivative-blending transition range on formulations that support it
   */
  virtual void updateEnergyMortarQpDerivativeBlendGapRange( RealT /*min_gap*/, RealT /*max_gap*/ ) {}

  /**
   * @brief Update ENERGY_MORTAR fixed full-path QP derivative-blending weight on formulations that support it
   */
  virtual void updateEnergyMortarQpDerivativeBlendWeight( RealT /*weight*/ ) {}

  /**
   * @brief Update whether gap-based QP derivative-blending weight is differentiated with Enzyme
   */
  virtual void updateEnergyMortarQpDerivativeBlendEnzymeGapWeight( bool /*enabled*/ ) {}

  /**
   * @brief Update whether ENERGY_MORTAR QP derivative blending uses cached simplified-path integration data
   */
  virtual void updateEnergyMortarQpFrozenIntegration( bool /*enabled*/ ) {}
  virtual void updateEnergyMortarReferenceGeometry( bool /*enabled*/ ) {}

  /**
   * @brief Cache ENERGY_MORTAR QP integration data for the current interface pairs and coordinates
   */
  virtual void updateEnergyMortarQpFrozenIntegrationData() {}

  /**
   * @brief Update ENERGY_MORTAR penalty mode on formulations that support it
   */
  virtual void updateEnergyMortarPenaltyMode( EnergyMortarPenaltyMode /*mode*/ ) {}

  /**
   * @brief Update ENERGY_MORTAR nodal-energy basis on formulations that support it
   */
  virtual void updateEnergyMortarNodalEnergyBasis( EnergyMortarNodalEnergyBasis /*basis*/ ) {}

  /**
   * @brief Update ENERGY_MORTAR eta gap scaling on formulations that support it
   */
  virtual void updateEnergyMortarEtaGapScaling( bool /*enabled*/ ) {}

  /**
   * @brief Update ENERGY_MORTAR eta angle smoothing on formulations that support it
   */
  virtual void updateEnergyMortarEtaAngleSmoothing( bool /*enabled*/ ) {}

  /**
   * @brief Update ENERGY_MORTAR eta angle-smoothing start angle on formulations that support it
   */
  virtual void updateEnergyMortarEtaAngleSmoothingStart( RealT /*start_angle*/ ) {}

  /**
   * @brief Update ENERGY_MORTAR nodal-energy angle smoothing on formulations that support it
   */
  virtual void updateEnergyMortarNodalEnergyAngleSmoothing( bool /*enabled*/ ) {}

  /**
   * @brief Update residual-gap offset on formulations that cache contact parameters
   */
  virtual void updateResidualGap( RealT /*residual_gap*/ ) {}

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
   * @brief Returns a nodal normal visualization field, when supported by the formulation
   *
   * @return Pointer to a vector grid function on the parent-linked boundary submesh, or nullptr if unsupported
   */
  virtual mfem::ParGridFunction* getMfemNodalNormal() { return nullptr; }

  /**
   * @brief Returns the average QP penalty residual gap, when supported by the formulation
   *
   * @return Average over active quadrature-point penalty face pairs from the last force update
   */
  virtual RealT getEnergyMortarQpResidualGapAverage() const { return 0.0; }

  /**
   * @brief Returns QP penalty diagnostics, when supported by the formulation
   *
   * @return Diagnostics collected during the last force update
   */
  virtual EnergyMortarQpDiagnostics getEnergyMortarQpDiagnostics() const { return {}; }

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
