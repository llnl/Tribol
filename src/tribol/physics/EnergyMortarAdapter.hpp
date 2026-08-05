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

template <template <typename> class EnforcementPolicy>
class EnergyMortarAdapter;

template <typename Adapter>
class NodalGapEnforcement : public ContactFormulation {
 protected:
  shared::ParVector g_tilde_vec_;
  shared::ParVector A_vec_;
  shared::ParVector gap_vec_;
  mutable shared::ParSparseMat dg_tilde_dx_;
  shared::ParSparseMat dA_dx_;
  shared::ParVector pressure_vec_;

  void init( Adapter* adapter );
  static void updateNodalGaps( Adapter* adapter );
  static void updateNodalForces( Adapter* adapter );

  static shared::ParSparseMat computeDfDxSecondDerivativesLM( Adapter* adapter,
                                                              const mfem::GridFunction& redecomp_lambda );
  static shared::ParSparseMat computeDfDxSecondDerivativesPenalty( Adapter* adapter,
                                                                   const mfem::GridFunction& redecomp_pressure,
                                                                   const mfem::GridFunction& redecomp_g_tilde,
                                                                   const mfem::GridFunction& redecomp_A );

 public:
  const mfem::HypreParVector& getMfemGap() const override
  {
    return static_cast<const Adapter*>( this )->use_penalty_ ? gap_vec_.get() : g_tilde_vec_.get();
  }
  mfem::HypreParVector& getMfemPressure() override { return pressure_vec_.get(); }
  std::unique_ptr<mfem::HypreParMatrix> getMfemDgDx() const override
  {
    return std::unique_ptr<mfem::HypreParMatrix>( dg_tilde_dx_.release() );
  }
  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp() const override;
};

template <typename Adapter>
class QuadraturePointGapEnforcement : public ContactFormulation {
 protected:
  void init( Adapter* /*adapter*/ ) {}
  static void updateNodalGaps( Adapter* /*adapter*/ ) {}
  static void updateNodalForces( Adapter* adapter );
};

template <template <typename> class EnforcementPolicy>
class EnergyMortarAdapter : public EnforcementPolicy<EnergyMortarAdapter<EnforcementPolicy>> {
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
  EnergyMortarAdapter( MfemMeshData& mesh_data, MfemSubmeshData& submesh_data, MfemJacobianData& jac_data, double k,
                       double delta, int N, bool enzyme_quadrature, bool use_penalty = true );

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
  void updateNodalGaps() override { EnforcementPolicy<EnergyMortarAdapter>::updateNodalGaps( this ); }

  /**
   * @brief Assemble nodal forces/residual and Jacobian contributions
   *
   * In penalty mode this computes pressure from the current gap and penalty stiffness. In LM mode this interprets the
   * stored pressure vector as the Lagrange multiplier vector `lambda`.
   */
  void updateNodalForces() override { EnforcementPolicy<EnergyMortarAdapter>::updateNodalForces( this ); }

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
   * @brief Update constant penalty stiffness
   *
   * @param mesh1_penalty Penalty stiffness for mesh 1.
   * @param mesh2_penalty Penalty stiffness for mesh 2.
   */
  void updateConstantPenaltyStiffness( double mesh1_penalty, double mesh2_penalty ) override;

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
   * @brief Return df/dx for the assembled contact force
   *
   * @return Unique pointer to `mfem::HypreParMatrix` holding df/dx
   *
   * @note Requires updateNodalForces() to be called first.
   * @note Ownership of the internally stored matrix is transferred; repeated calls without recomputing will return
   * null/empty data.
   */
  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDx() const override;

#endif

  friend EnforcementPolicy<EnergyMortarAdapter>;

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
};

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_ENERGYMORTARADAPTER_HPP_ */
