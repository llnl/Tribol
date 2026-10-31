// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Shared host MFEM assembly for independent energy-defined and force-defined element-pair contributions.
 */
#ifndef SRC_TRIBOL_PHYSICS_PAIRCONTACTFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_PAIRCONTACTFORMULATION_HPP_

#include <utility>

#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/CommonPlane.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"
#include "tribol/physics/PairEvaluation.hpp"
#include "tribol/physics/PenaltyStiffness.hpp"
#include "tribol/search/CoarseSearch.hpp"
#include "tribol/search/RefinedSearch.hpp"
#include "tribol/mesh/MfemData.hpp"

namespace tribol {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )

/**
 * @brief Configure quadrature-point Energy Mortar energy and its default derivatives.
 * @tparam QuadratureRule Smoothed projected-edge pair rule.
 * @tparam PenaltyStiffness Callable combining two supplied penalties; defaults to their arithmetic mean.
 */
template <typename QuadratureRule, typename PenaltyStiffness = AveragePenaltyStiffness>
class EnergyMortarPairMethod {
 public:
  using Definition = EnergyMortarEnergy<QuadratureRule>;         ///< Energy with fixed penalty and rule configuration.
  using FirstGeometry = typename QuadratureRule::FirstGeometry;  ///< First-element coordinate layout.
  using SecondGeometry = typename QuadratureRule::SecondGeometry;  ///< Second-element coordinate layout.
  static constexpr bool has_energy = true;  ///< Assembled results include energy, including for empty contact.
  /**
   * @brief Store rule configuration and the fixed gap offset; retain no geometry or calculated fields.
   * @param quadrature_rule Owned pair rule.
   * @param residual_gap Offset subtracted from the geometric gap.
   * @param penalty_enabled Whether kinematic penalty enforcement has been configured.
   * @param penalty_stiffness Owned stiffness combination, applied to the registered constant mesh penalties.
   */
  EnergyMortarPairMethod( QuadratureRule quadrature_rule, RealT residual_gap, bool penalty_enabled,
                          PenaltyStiffness penalty_stiffness = {} )
      : quadrature_rule_( std::move( quadrature_rule ) ),
        residual_gap_( residual_gap ),
        penalty_enabled_( penalty_enabled ),
        penalty_stiffness_( std::move( penalty_stiffness ) )
  {
  }
  /**
   * @brief Build an energy definition using the selected combination of current constant mesh penalties.
   * @param mesh_data Registered MFEM penalty settings.
   * @param mesh1 Current first mesh, checked for linear two-dimensional edges.
   * @param mesh2 Current second mesh, checked for linear two-dimensional edges.
   * @return Fixed inputs shared by all pairs in this request; physical geometry is supplied during evaluation.
   * @throws std::invalid_argument Unsupported mesh geometry or penalty enforcement has not been configured.
   */
  Definition prepare( const MfemMeshData& mesh_data, MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const;
  /** @brief Select compressive contact, including zero gap. @param primal Primal energy, activation, and status.
   * @return Whether derivatives and assembly are needed. Quadrature status must be checked first. */
  static bool isActive( const EnergyMortarEnergyValue& primal ) { return primal.has_active_qp; }
  /**
   * @brief Reuse primal energy while differentiating the complete geometry-dependent calculation.
   * @param coordinates Current pair coordinates.
   * @param definition Fixed method inputs for this request.
   * @param primal Successful, active primal calculation for these coordinates.
   * @return Energy, residual, and Jacobian in owning arrays.
   */
  static PairEnergyDerivatives<Definition::dof_count> differentiate(
      MdSpan<const RealT, Definition::dof_count> coordinates, const Definition& definition,
      const EnergyMortarEnergyValue& primal )
  {
    return EnergyPairEvaluator<Definition>::evaluate( coordinates, definition, primal.energy );
  }

 private:
  QuadratureRule quadrature_rule_;      ///< Owned mapping and base-rule configuration.
  RealT residual_gap_;                  ///< Geometry-independent gap offset.
  bool penalty_enabled_;                ///< Whether this configuration can calculate penalty forces.
  PenaltyStiffness penalty_stiffness_;  ///< Owned stiffness combination; retains no mesh data.
};

/**
 * @brief Configure Common Plane's direct penalty residual and its default derivatives.
 * @tparam BaseQuadratureRule Gauss rule on the common-line interval.
 * @tparam PenaltyStiffness Callable combining two supplied penalties; defaults to springs in series.
 */
template <typename BaseQuadratureRule, typename PenaltyStiffness = SeriesPenaltyStiffness>
class CommonPlanePairMethod {
 public:
  using Definition = CommonPlaneForce<BaseQuadratureRule>;  ///< Force definition differentiated as a residual.
  using FirstGeometry = LinearEdgeGeometry;                 ///< First-element coordinate layout.
  using SecondGeometry = LinearEdgeGeometry;                ///< Second-element coordinate layout.
  static constexpr bool has_energy = false;                 ///< Force-defined results leave energy absent.
  /**
   * @brief Store the gap offset and stiffness combination.
   * @param residual_gap Nonnegative finite offset subtracted before activation.
   * @param penalty_stiffness Owned combination, applied after the two mesh penalty scales.
   */
  explicit CommonPlanePairMethod( RealT residual_gap, PenaltyStiffness penalty_stiffness = {} )
      : residual_gap_( residual_gap ), penalty_stiffness_( std::move( penalty_stiffness ) )
  {
  }
  /**
   * @brief Build a residual definition by combining current mesh penalties after applying their mesh scales.
   * @param mesh_data Parent transfer data, unused by this method's penalty calculation.
   * @param mesh1 Current first linear edge mesh and its constant penalty data.
   * @param mesh2 Current second linear edge mesh and its constant penalty data.
   * @return Fixed inputs for this request; invalid geometry types or penalties throw std::invalid_argument.
   */
  Definition prepare( const MfemMeshData& mesh_data, MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const;
  /** @brief Select compressive contact, including zero gap. @param primal Primal residual, activation, and status.
   * @return Whether derivatives and assembly are needed. Quadrature status must be checked first. */
  static bool isActive( const CommonPlaneResidualData& primal ) { return primal.has_active_points; }
  /**
   * @brief Reuse the primal residual while differentiating current geometry and quadrature.
   * @param coordinates Current pair coordinates.
   * @param definition Fixed method inputs for this request.
   * @param primal Successful, active primal calculation for these coordinates.
   * @return Residual and generally nonsymmetric Jacobian in owning arrays.
   */
  static PairResidualJacobian<Definition::dof_count> differentiate(
      MdSpan<const RealT, Definition::dof_count> coordinates, const Definition& definition,
      const CommonPlaneResidualData& primal )
  {
    return ForcePairEvaluator<Definition>::evaluate( coordinates, definition, primal.residual.view() );
  }

 private:
  RealT residual_gap_;                  ///< Geometry-independent gap offset.
  PenaltyStiffness penalty_stiffness_;  ///< Owned stiffness combination; retains no mesh data.
};

/**
 * @brief Search current surfaces and assemble independent element-pair residuals and Jacobians on parent true DOFs.
 *
 * No geometry, quadrature, or calculated fields are retained. Method preparation uses current settings on each request.
 * @tparam PairMethod Prepares a definition, selects active primal results, and calculates coordinate derivatives.
 */
template <typename PairMethod>
class PairContactFormulation : public ContactFormulation {
 public:
  /**
   * @brief Borrow parent transfer data and own search and method configuration.
   * @param mesh_data Displacement data; must outlive this formulation.
   * @param jacobian_data Displacement Jacobian transfer; must outlive this formulation.
   * @param coarse_search Non-null owned candidate search.
   * @param refined_search Non-null owned pair acceptance checks.
   * @param method Owned configuration; no coordinate views or calculated fields.
   */
  PairContactFormulation( MfemMeshData& mesh_data, MfemJacobianData& jacobian_data,
                          std::unique_ptr<CoarseSearch> coarse_search, std::unique_ptr<RefinedSearch> refined_search,
                          PairMethod method );
  /**
   * @brief Return the assembled residual and Jacobian, with energy only for an energy-defined method.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned force and df_dx. Nodal gap, pressure, multiplier derivatives, and timestep are absent.
   * @throws std::invalid_argument Invalid method inputs or pair geometry.
   * @throws std::length_error Insufficient quadrature storage or unrepresentable assembly size.
   */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override;

 private:
  MfemMeshData& mesh_data_;                        ///< Borrowed displacement and decomposition data.
  MfemJacobianData& jacobian_data_;                ///< Borrowed parent Jacobian transfer, with no dual path required.
  std::unique_ptr<CoarseSearch> coarse_search_;    ///< Owned spatial search.
  std::unique_ptr<RefinedSearch> refined_search_;  ///< Owned pair acceptance checks.
  PairMethod method_;                              ///< Owned method configuration.
};
#endif
}  // namespace tribol
#endif
