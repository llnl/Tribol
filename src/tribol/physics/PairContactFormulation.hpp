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
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/physics/PairMethod.hpp"
#include "tribol/physics/PairMethodInputs.hpp"
#include "tribol/physics/PenaltyStiffness.hpp"
#include "tribol/search/CoarseSearch.hpp"
#include "tribol/search/RefinedSearch.hpp"
#include "tribol/mesh/MfemNodalAssembler.hpp"

namespace tribol {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )

/**
 * @brief Energy Mortar defaults: energy derivatives and current registered constant penalties.
 * @tparam QuadratureRule Smoothed projected-edge pair rule.
 * @tparam PenaltyStiffness Combination of registered mesh constants; defaults to their arithmetic mean.
 * @tparam GapDefinition Owned geometric or offset gap; the factory selects the offset wrapper.
 */
template <typename QuadratureRule, typename PenaltyStiffness = AveragePenaltyStiffness,
          typename GapDefinition = OffsetGap<OpposingNormalGap>>
using EnergyMortarPairMethod =
    EnergyPairMethod<EnergyMortarEnergy<QuadratureRule, GapDefinition>, RegisteredPenaltyInputs<PenaltyStiffness>>;

/**
 * @brief Nodal Energy Mortar defaults: nodal gap/area integrals, nodal energy derivatives, and registered penalties.
 * @tparam QuadratureRule Smoothed projected-edge pair rule.
 * @tparam PenaltyStiffness Combination of registered mesh constants; defaults to their arithmetic mean.
 * @tparam GapDefinition Owned geometric or offset gap; the factory selects the offset wrapper.
 */
template <typename QuadratureRule, typename PenaltyStiffness = AveragePenaltyStiffness,
          typename GapDefinition = OffsetGap<OpposingNormalGap>>
using EnergyMortarNodalPairMethod =
    NodalPairMethod<EnergyMortarNodalEnergy<QuadratureRule, GapDefinition>, RegisteredPenaltyInputs<PenaltyStiffness>>;

/**
 * @brief Common Plane defaults: residual derivatives and current scaled mesh penalties.
 * @tparam BaseQuadratureRule Gauss rule on the common-line interval.
 * @tparam PenaltyStiffness Combination of scaled mesh penalties; defaults to springs in series.
 * @tparam GapDefinition Owned geometric or offset gap; the factory selects the offset wrapper.
 */
template <typename BaseQuadratureRule, typename PenaltyStiffness = SeriesPenaltyStiffness,
          typename GapDefinition = OffsetGap<CommonPlaneGap>>
using CommonPlanePairMethod =
    ForcePairMethod<CommonPlaneForce<BaseQuadratureRule, GapDefinition>, ScaledMeshPenaltyInputs<PenaltyStiffness>>;

/**
 * @brief Search current surfaces and assemble element-pair or nodal residuals and Jacobians on MFEM true DOFs.
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
   * @brief Borrow parent and dual submesh transfer data and own search and nodal method configuration.
   * @param mesh_data Displacement data; must outlive this formulation.
   * @param submesh_data Dual submesh pressure/gap data; must outlive this formulation.
   * @param jacobian_data Displacement and dual Jacobian transfer; must outlive this formulation.
   * @param coarse_search Non-null owned candidate search.
   * @param refined_search Non-null owned pair acceptance checks.
   * @param method Owned nodal method configuration; no coordinate views or calculated fields.
   */
  PairContactFormulation( MfemMeshData& mesh_data, MfemSubmeshData& submesh_data, MfemJacobianData& jacobian_data,
                          std::unique_ptr<CoarseSearch> coarse_search, std::unique_ptr<RefinedSearch> refined_search,
                          PairMethod method );

  /**
   * @brief Return assembled nodal gaps and their coordinate Jacobian for nodal methods.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned gap and dg_dx on dual submesh and parent true DOFs.
   * @throws std::logic_error The method does not use nodal submesh assembly.
   */
  ContactFormulationResult computeNodalGaps( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override;

  /**
   * @brief Return the assembled residual and Jacobian, with energy for energy-defined methods and nodal fields for
   * nodal methods.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned force and df_dx, plus gap, dg_dx, pressure, and optional df_dp for nodal methods.
   * @throws std::invalid_argument Invalid method inputs or pair geometry.
   * @throws std::length_error Insufficient quadrature storage or unrepresentable assembly size.
   */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override;

 private:
  using Definition = typename PairMethod::Definition;         ///< Prepared local method definition.
  using Coordinates = MdArray<RealT, Definition::dof_count>;  ///< Pair coordinates gathered on the stack.
  /** @brief Search current meshes. @param mesh1 First mesh. @param mesh2 Second mesh.
   * @return Owned geometry-approved pairs, including pairs with no quadrature overlap. */
  ArrayT<InterfacePair> findPairs( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const;
  /** @brief Gather current pair coordinates. @param pair Element IDs. @param mesh1 First mesh.
   * @param mesh2 Second mesh. @return Coordinates in element/node/component order. */
  static Coordinates getPairCoordinates( InterfacePair pair, MeshData::Viewer mesh1, MeshData::Viewer mesh2 );
  /** @brief Assemble nodal quantities before activation. @param definition Nodal integral calculation.
   * @param mesh1 First mesh. @param mesh2 Second mesh. @param pairs Geometry-approved pairs.
   * @param[in,out] assembler Empty nodal assembler. @return Unclamped gap/area integrals and gradients. */
  MfemNodalQuantityData assembleNodalQuantities( const Definition& definition, MeshData::Viewer mesh1,
                                                 MeshData::Viewer mesh2, ArrayViewT<const InterfacePair> pairs,
                                                 MfemNodalAssembler& assembler ) const;
  /** @brief Transfer gap outputs without retaining calculated fields. @param[in,out] quantities Owned nodal data.
   * @return Normalized penalty gap or weighted LM gap, with weighted-gap coordinate Jacobian. */
  ContactFormulationResult takeGapResult( MfemNodalQuantityData& quantities ) const;
  /** @brief Evaluate explicit nodal energy derivatives and assemble their coordinate derivatives.
   * @param definition Nodal integral calculation. @param mesh1 First mesh. @param mesh2 Second mesh.
   * @param pairs Geometry-approved pairs. @return Owned nodal contact outputs and global energy. */
  ContactFormulationResult assembleNodalForces( const Definition& definition, MeshData::Viewer mesh1,
                                                MeshData::Viewer mesh2, ArrayViewT<const InterfacePair> pairs ) const;

  MfemMeshData& mesh_data_;                        ///< Borrowed parent displacement and decomposition data.
  MfemJacobianData& jacobian_data_;                ///< Borrowed coordinate and optional dual transfers.
  MfemSubmeshData* submesh_data_ = nullptr;        ///< Borrowed dual data for nodal methods only.
  std::unique_ptr<CoarseSearch> coarse_search_;    ///< Owned spatial search.
  std::unique_ptr<RefinedSearch> refined_search_;  ///< Owned pair acceptance checks.
  PairMethod method_;                              ///< Owned method configuration.
};
#endif
}  // namespace tribol
#endif
