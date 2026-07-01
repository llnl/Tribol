// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTARADAPTER_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTARADAPTER_HPP_
#include "tribol/config.hpp"

#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/EnergyMortar.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/common/Parameters.hpp"

#include "mfem.hpp"

#include <memory>

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

/**
 * @brief ContactFormulation adapter for the ENERGY_MORTAR method
 *
 * This class wraps the ENERGY_MORTAR implementation and exposes a ContactFormulation interface compatible with
 * MFEM/redecomp workflows.
 *
 * In penalty mode, the dual field is interpreted as pressure `p = k * (g_tilde / A)`. In Lagrange multiplier (LM) mode,
 * the dual field is interpreted as the multiplier vector `lambda` and the formulation computes `f = G^T * lambda` where
 * `G = d(g_tilde)/dx`.
 */
class EnergyMortarAdapter : public ContactFormulation {
 public:
  /**
   * @brief Construct a new EnergyMortarAdapter
   *
   * @param mesh_data MFEM mesh data for the parent/primary variables
   * @param submesh_data MFEM submesh data for the dual variables (pressure/gap/LM)
   * @param jac_data MFEM Jacobian transfer data
   * @param k Penalty stiffness
   * @param delta Smoothing length
   * @param N Quadrature order
   * @param enzyme_quadrature If true, use Enzyme-assisted quadrature
   * @param use_penalty If true, interpret the dual field as pressure; otherwise interpret it as a Lagrange multiplier
   * vector (LM mode)
   *
   * @note The ENERGY_MORTAR implementation follows the literature convention of integrating on a non-mortar side and
   * mapping to a mortar side. To maintain that convention within Tribol, the adapter may internally flip mesh roles
   * relative to the order of the meshes provided here.
   */
  EnergyMortarAdapter(
      MfemMeshData& mesh_data, MfemSubmeshData& submesh_data, MfemJacobianData& jac_data, double k, double delta, int N,
      bool enzyme_quadrature, bool use_penalty = true,
      EnergyMortarNormalMode normal_mode = EnergyMortarNormalMode::ELEMENT_NORMAL, bool projection_smoothing = true,
      double h1_active_set_smoothing_gap = 0.0,
      EnergyMortarPenaltyMode penalty_mode = EnergyMortarPenaltyMode::NODAL_GAP,
      EnergyMortarNodalEnergyBasis nodal_energy_basis = EnergyMortarNodalEnergyBasis::CUBIC_SPLINE,
      bool nodal_energy_angle_smoothing = true, RealT residual_gap = 0.0 );

  /**
   * @brief Default destructor
   */
  virtual ~EnergyMortarAdapter() = default;

  /**
   * @brief Set interface pairs identified in coarse binning for this update
   *
   * @param pairs Candidate interface pairs (ownership transferred)
   * @param check_level Level of validation to apply to the pairs (currently unused)
   */
  void setInterfacePairs( ArrayT<InterfacePair>&& pairs, int check_level ) override;

  /**
   * @brief Update internal integration rule data
   *
   * @note This is currently a no-op. ENERGY_MORTAR directly assembles gap/force quantities over the stored interface
   * pairs in updateNodalGaps() and updateNodalForces().
   */
  void updateIntegrationRule() override;

  /**
   * @brief Assemble nodal gaps and their derivatives
   *
   * In penalty mode this assembles both `g_tilde` and `g = g_tilde / A`. In LM mode the constraint is `g_tilde = 0` and
   * derivatives are taken with respect to `g_tilde`.
   */
  void updateNodalGaps() override;

  /**
   * @brief Assemble nodal forces/residual and Jacobian contributions
   *
   * In penalty mode this computes pressure from the current gap and penalty stiffness. In LM mode this interprets the
   * stored pressure vector as the Lagrange multiplier vector `lambda`.
   */
  void updateNodalForces() override;

  /**
   * @brief Reports if formulation has a maximum allowable timestep calculation
   *
   * @return false
   */
  bool hasTimeStepCalculation() override { return false; }

  /**
   * @brief Compute the allowable timestep for this formulation
   *
   * @return Maximum allowable timestep
   *
   * @note This is currently a placeholder and returns a constant value.
   */
  RealT computeTimeStep() override;

  /**
   * @brief Get energy stored by contact constraints
   *
   * @return Contact energy
   */
  RealT getEnergy() const override { return energy_; }

  /**
   * @brief Refresh mesh handles used by this formulation
   *
   * Some workflows re-register meshes each cycle (e.g. MFEM + redecomp), potentially replacing the underlying MeshData
   * instances in MeshManager. This method updates the adapter's cached mesh pointers so subsequent assembly uses valid
   * mesh views.
   *
   * @param mesh1 First Tribol mesh for the coupling scheme
   * @param mesh2 Second Tribol mesh for the coupling scheme
   */
  void updateMeshes( MeshData& mesh1, MeshData& mesh2 ) override;

  /**
   * @brief Update penalty parameters
   *
   * @param use_penalty If true, the formulation is configured for penalty mode.
   * @param k Penalty stiffness.
   */
  void updatePenaltyParameters( bool use_penalty, double k ) override;

  /**
   * @brief Update normal-mode parameters and rebuild the element evaluator
   *
   * @param normal_mode Normal field used by EnergyMortar
   * @param projection_smoothing If true, apply projection-bound smoothing
   */
  void updateEnergyMortarNormalMode( EnergyMortarNormalMode normal_mode, bool projection_smoothing ) override;

  /**
   * @brief Update active-set smoothing transition gap
   *
   * @param gap_transition Positive gap transition width; disabled when <= 0
   */
  void updateEnergyMortarH1ActiveSetSmoothing( RealT gap_transition ) override;

  /**
   * @brief Update penalty enforcement mode
   *
   * @param mode Penalty mode
   */
  void updateEnergyMortarPenaltyMode( EnergyMortarPenaltyMode mode ) override;

  /**
   * @brief Update nodal-energy basis
   *
   * @param basis Basis used by NODAL_ENERGY mode
   */
  void updateEnergyMortarNodalEnergyBasis( EnergyMortarNodalEnergyBasis basis ) override;

  /**
   * @brief Update nodal-energy angle smoothing
   *
   * @param enabled True to apply 80-to-90 degree angle smoothing
   */
  void updateEnergyMortarNodalEnergyAngleSmoothing( bool enabled ) override;

  /**
   * @brief Update residual-gap offset
   *
   * @param residual_gap User-defined residual gap offset
   */
  void updateResidualGap( RealT residual_gap ) override;

#ifdef BUILD_REDECOMP
  /**
   * @brief Return the parent true-dof force vector
   *
   * @return Reference to the force vector on the parent mesh true-dofs
   *
   * @note Requires updateNodalForces() to be called first.
   */
  const mfem::HypreParVector& getMfemForce() const override { return force_vec_.get(); }

  /**
   * @brief Return the submesh true-dof gap vector
   *
   * @return Reference to the gap vector on the submesh true-dofs
   *
   * @note Requires updateNodalGaps() to be called first.
   */
  const mfem::HypreParVector& getMfemGap() const override;

  /**
   * @brief Return a reference to the dual (pressure/LM) true-dof vector
   *
   * @return Reference to the pressure vector (penalty mode) or multiplier vector (LM mode)
   */
  mfem::HypreParVector& getMfemPressure() override { return pressure_vec_.get(); }

  /**
   * @brief Return the recovered H1 nodal normal field on the contact submesh
   *
   * @return Pointer to the submesh normal grid function, or nullptr when H1 normal mode is inactive
   */
  mfem::ParGridFunction* getMfemNodalNormal() override;

  /**
   * @brief Return df/dx for the assembled contact force
   *
   * @return Unique pointer to `mfem::HypreParMatrix` holding df/dx
   *
   * @note Requires updateNodalForces() to be called first.
   * @note Ownership of the internally stored matrix is transferred; repeated calls without recomputing will return
   * null/empty data.
   */
  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDx() const override;

  /**
   * @brief Return d(g_tilde)/dx for the assembled gap
   *
   * @return Unique pointer to `mfem::HypreParMatrix` holding d(g_tilde)/dx
   *
   * @note Requires updateNodalGaps() to be called first.
   * @note Ownership of the internally stored matrix is transferred; repeated calls without recomputing will return
   * null/empty data.
   */
  std::unique_ptr<mfem::HypreParMatrix> getMfemDgDx() const override;

  /**
   * @brief Return df/dp (penalty) or df/dlambda (LM)
   *
   * @return Unique pointer to `mfem::HypreParMatrix` holding df/dp or df/dlambda
   *
   * @note Requires updateNodalForces() to be called first.
   * @note In penalty mode this returns nullptr.
   */
  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp() const override;

#endif

 private:
  /**
   * @brief Assemble force and Jacobian for quadrature-point penalty enforcement
   */
  void updateQuadraturePointPenaltyForces();

  /**
   * @brief Controls penalty vs. Lagrange multiplier (LM) mode
   *
   * - true: penalty mode, dual vector is pressure
   * - false: LM mode, dual vector is the multiplier `lambda`
   */
  bool use_penalty_;

  /**
   * @brief Tolerance used to avoid division by zero for area-weighted quantities
   */
  double area_tol_{ 1.0e-14 };

  /**
   * @brief If true, treat contact as tied (no opening) for gap filtering logic
   */
  // TODO: avoid duplication of coupling scheme data that drives algorithmic combinatorics
  bool tied_contact_ = false;

  /**
   * @brief MFEM mesh data for the parent/primary variables (coords/response)
   */
  MfemMeshData& mesh_data_;

  /**
   * @brief MFEM submesh data for the dual field (pressure/gap/LM)
   */
  MfemSubmeshData& submesh_data_;

  /**
   * @brief MFEM Jacobian transfer data used to assemble global derivatives
   */
  MfemJacobianData& jac_data_;

  // NOTE: mesh1 maps to mesh2_ and mesh2 maps to mesh1_. This is to keep
  // consistent with mesh1_ being non-mortar and mesh2_ being mortar as is
  // typical in the literature, but different from Tribol convention.

  /**
   * @brief Pointer to the non-mortar side mesh used by ENERGY_MORTAR
   *
   * @note This is refreshed via updateMeshes() to handle mesh re-registration.
   */
  MeshData* mesh1_ = nullptr;

  /**
   * @brief Pointer to the mortar side mesh used by ENERGY_MORTAR
   *
   * @note This is refreshed via updateMeshes() to handle mesh re-registration.
   */
  MeshData* mesh2_ = nullptr;

  /**
   * @brief Contact parameters (penalty, smoothing, quadrature)
   */
  ContactParams params_;

  /**
   * @brief Evaluator implementing ENERGY_MORTAR element-level computations
   */
  std::unique_ptr<EnergyMortarCalculator> evaluator_;

  // Stored InterfacePairs

  /**
   * @brief Interface pairs used for the current update
   */
  ArrayT<InterfacePair> pairs_;

  // These store the assembled nodal values

  /**
   * @brief Unnormalized gap vector g_tilde on the dual (pressure) true-dofs
   */
  shared::ParVector g_tilde_vec_;

  /**
   * @brief Tributary area vector A on the dual (pressure) true-dofs
   */
  shared::ParVector A_vec_;

  /**
   * @brief Normalized gap vector g = g_tilde / A on the dual (pressure) true-dofs
   */
  shared::ParVector gap_vec_;

  /**
   * @brief Derivative d(g_tilde)/dx as a (dual rows) x (parent displacement cols) matrix
   */
  mutable shared::ParSparseMat dg_tilde_dx_;

  /**
   * @brief Derivative dA/dx as a (dual rows) x (parent displacement cols) matrix
   */
  shared::ParSparseMat dA_dx_;

  /**
   * @brief Dual true-dof vector (pressure = k * (g_tilde / A) in penalty mode, lambda in LM mode)
   */
  shared::ParVector pressure_vec_;

  /**
   * @brief Recovered H1 nodal normal field on the redecomp mesh
   */
  std::unique_ptr<mfem::GridFunction> redecomp_nodal_normal_;

  /**
   * @brief Recovered H1 nodal normal field transferred to the parent-linked boundary submesh
   */
  mfem::ParGridFunction submesh_nodal_normal_;

  /**
   * @brief Contact constraint energy associated with the current state
   */
  RealT energy_;

  /**
   * @brief Parent true-dof force vector assembled from contact contributions
   */
  shared::ParVector force_vec_;

  /**
   * @brief Derivative df/dx assembled on parent displacement true-dofs
   */
  mutable shared::ParSparseMat df_dx_;

  /**
   * @brief Assemble the LM second-derivative df/dx contribution
   *
   * Assembles `df/dx = lambda · d²(g_tilde)/dx²`.
   *
   * @param redecomp_lambda Lagrange multiplier on the redecomp mesh
   * @return Assembled df/dx contribution on parent true-dofs
   */
  shared::ParSparseMat computeDfDxSecondDerivativesLM( const mfem::GridFunction& redecomp_lambda );

  /**
   * @brief Assemble the penalty second-derivative df/dx contribution
   *
   * Assembles the second-derivative penalty contribution:
   * `p · d²(g_tilde)/dx² - (g_tilde p / A) · d²A/dx²`.
   *
   * @param redecomp_pressure Pressure field on the redecomp mesh
   * @param redecomp_g_tilde g_tilde on the redecomp mesh
   * @param redecomp_A Area weighting A on the redecomp mesh
   * @return Assembled df/dx contribution on parent true-dofs
   */
  shared::ParSparseMat computeDfDxSecondDerivativesPenalty( const mfem::GridFunction& redecomp_pressure,
                                                            const mfem::GridFunction& redecomp_g_tilde,
                                                            const mfem::GridFunction& redecomp_A );
};

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_ENERGYMORTARADAPTER_HPP_ */
