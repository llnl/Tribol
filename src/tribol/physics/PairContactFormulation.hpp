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

#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/CommonPlane.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"
#include "tribol/physics/PairMethod.hpp"
#include "tribol/physics/PairMethodInputs.hpp"
#include "tribol/physics/PenaltyStiffness.hpp"
#include "tribol/search/CoarseSearch.hpp"
#include "tribol/search/RefinedSearch.hpp"
#include "tribol/mesh/MfemData.hpp"

namespace tribol {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )

/**
 * @brief Energy Mortar defaults: energy derivatives and current registered constant penalties.
 * @tparam QuadratureRule Smoothed projected-edge pair rule.
 * @tparam PenaltyStiffness Combination of registered mesh constants; defaults to their arithmetic mean.
 */
template <typename QuadratureRule, typename PenaltyStiffness = AveragePenaltyStiffness>
using EnergyMortarPairMethod =
    EnergyPairMethod<EnergyMortarEnergy<QuadratureRule>, RegisteredPenaltyInputs<PenaltyStiffness>>;

/**
 * @brief Common Plane defaults: residual derivatives and current scaled mesh penalties.
 * @tparam BaseQuadratureRule Gauss rule on the common-line interval.
 * @tparam PenaltyStiffness Combination of scaled mesh penalties; defaults to springs in series.
 */
template <typename BaseQuadratureRule, typename PenaltyStiffness = SeriesPenaltyStiffness>
using CommonPlanePairMethod =
    ForcePairMethod<CommonPlaneForce<BaseQuadratureRule>, ScaledMeshPenaltyInputs<PenaltyStiffness>>;

/**
 * @brief Search current surfaces and assemble independent element-pair residuals and Jacobians on parent true DOFs.
 *
 * No geometry, quadrature, or calculated fields are retained. Method preparation uses current settings on each request.
 * @tparam PairMethod Prepares a definition and calculates coordinate derivatives; primal results expose activation.
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
