// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief MFEM assembly of direct common-plane penalty residuals and Enzyme Jacobians.
 */
#ifndef SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_

#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/CommonPlane.hpp"
#include "tribol/search/CoarseSearch.hpp"
#include "tribol/search/RefinedSearch.hpp"
#include "tribol/mesh/MfemData.hpp"

namespace tribol {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )

/**
 * @brief Assemble a force-defined, frictionless common-plane method on linear 2D MFEM surfaces.
 *
 * Returned vectors use the contact residual sign, opposite the physical forces in the legacy Common Plane response.
 * Only assembled outputs and configuration are retained; current coordinates and pair quadrature are local to updates.
 * @tparam BaseQuadratureRule Gauss rule on each common-line overlap.
 */
template <typename BaseQuadratureRule>
class CommonPlaneFormulation : public ContactFormulation {
 public:
  /**
   * @brief Borrow displacement transfer data and take ownership of the searches.
   * @param mesh_data Borrowed MFEM displacement data; must outlive this formulation.
   * @param jacobian_data Borrowed parent-only Jacobian transfer; must outlive this formulation.
   * @param coarse_search Non-null coarse search, transferred to the formulation.
   * @param refined_search Non-null edge acceptance checks, transferred to the formulation.
   * @param residual_gap Nonnegative gap offset subtracted before activation.
   */
  CommonPlaneFormulation( MfemMeshData& mesh_data, MfemJacobianData& jacobian_data,
                          std::unique_ptr<CoarseSearch> coarse_search, std::unique_ptr<RefinedSearch> refined_search,
                          RealT residual_gap );

  /**
   * @brief Report that this quadrature-point method has no nodal gap field.
   * @param mesh1 Current first mesh; not retained.
   * @param mesh2 Current second mesh; not retained.
   * @throws std::logic_error Always; no nodal gap field is defined.
   */
  void updateNodalGaps( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) override;

  /**
   * @brief Search current geometry and assemble residual and Jacobian on parent true DOFs.
   * @param mesh1 Current first mesh with constant penalty data.
   * @param mesh2 Current second mesh with constant penalty data.
   * @throws std::invalid_argument Geometry or penalty data is invalid.
   */
  void updateNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) override;

  /** @brief Report no timestep calculation. @return False. */
  bool hasTimeStepCalculation() override { return false; }
  /** @brief Report that a force-defined method has no energy output. @throws std::logic_error Always. @return Never. */
  RealT getEnergy() const override;
  /** @brief Borrow the latest assembled contact residual. @return Parent true-DOF residual after updateNodalForces().
   */
  const mfem::HypreParVector& getMfemForce() const override { return residual_.get(); }
  /**
   * @brief Transfer ownership of the latest assembled residual Jacobian.
   * @return Parent true-DOF matrix; repeated calls without updating return null.
   */
  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDx() const override
  {
    return std::unique_ptr<mfem::HypreParMatrix>( jacobian_.release() );
  }

 private:
  MfemMeshData& mesh_data_;                                          ///< Borrowed displacement and decomposition data.
  MfemJacobianData& jacobian_data_;                                  ///< Borrowed displacement-only matrix transfer.
  std::unique_ptr<CoarseSearch> coarse_search_;                      ///< Owned spatial candidate search.
  std::unique_ptr<RefinedSearch> refined_search_;                    ///< Owned topology and orientation checks.
  CommonPlaneQuadratureRule<BaseQuadratureRule> quadrature_rule_{};  ///< Owned pair-generator configuration.
  RealT residual_gap_;                     ///< Gap offset held constant during differentiation.
  shared::ParVector residual_;             ///< Latest parent true-DOF contact residual.
  mutable shared::ParSparseMat jacobian_;  ///< Latest parent true-DOF Jacobian, transferable to the caller.
};
#endif
}  // namespace tribol
#endif
