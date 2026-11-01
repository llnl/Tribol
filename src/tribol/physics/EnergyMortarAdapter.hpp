// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTARADAPTER_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTARADAPTER_HPP_
#include "tribol/config.hpp"
#include "tribol/common/BasicTypes.hpp"

#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/EnergyMortar.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/mesh/MfemNodalAssembler.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/search/CoarseSearch.hpp"
#include "tribol/search/RefinedSearch.hpp"

#include "mfem.hpp"

#include <memory>

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

/**
 * @brief Assemble nodal Energy Mortar integrals, penalty or LM forces, and their derivatives through MFEM.
 *
 * Penalty pressure is k * (g_tilde / A). LM enforcement uses the caller's multiplier field and the weighted gap.
 * Geometry, pairs, and calculated fields are local to each request.
 * @tparam QuadratureRule Element-pair quadrature generator for the nodal integrals.
 */
template <typename QuadratureRule>
class EnergyMortarAdapter : public ContactFormulation {
 public:
  using Calculator = EnergyMortarCalculator<QuadratureRule>;  ///< Coordinate-based local evaluator.

  /**
   * @brief Calculate nodal gaps and their coordinate derivative.
   * @param mesh1 Current non-mortar mesh.
   * @param mesh2 Current mortar mesh.
   * @return Owned gap and dg_dx; other outputs are absent.
   */
  ContactFormulationResult computeNodalGaps( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override;

  /**
   * @brief Calculate nodal forces, energy, gaps, pressure, and derivatives.
   * @param mesh1 Current non-mortar mesh.
   * @param mesh2 Current mortar mesh.
   * @return Owned nodal outputs; df_dp is present only for LM enforcement. No timestep is calculated.
   */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override;

  /**
   * @brief Construct a new EnergyMortarAdapter
   *
   * @param mesh_data MFEM mesh data for the parent/primary variables
   * @param submesh_data MFEM submesh data for the dual variables (pressure/gap/LM)
   * @param jac_data MFEM Jacobian transfer data
   * @param coarse_search Coarse search whose ownership is transferred to this formulation; must be non-null
   * @param refined_search Refined search whose ownership is transferred to this formulation; must be non-null
   * @param k Penalty stiffness
   * @param quadrature_rule Quadrature generator owned by this adapter
   * @param enzyme_quadrature If true, differentiate the reference-rule construction with respect to geometry
   * @param use_penalty If true, interpret the dual field as pressure; otherwise interpret it as a Lagrange multiplier
   * vector (LM mode)
   * @param residual_gap Nonnegative gap offset subtracted from the kinematic gap. Positive values enforce separation
   *        between the contact surfaces.
   *
   * @note Mesh 1 is the non-mortar integration side; mesh 2 is the mortar projection side. Mesh views, candidate
   * pairs, and MFEM element maps all use this order.
   */
  EnergyMortarAdapter( MfemMeshData& mesh_data, MfemSubmeshData& submesh_data, MfemJacobianData& jac_data,
                       std::unique_ptr<CoarseSearch> coarse_search, std::unique_ptr<RefinedSearch> refined_search,
                       RealT k, QuadratureRule quadrature_rule, bool enzyme_quadrature, bool use_penalty = true,
                       RealT residual_gap = 0.0 );

  /**
   * @brief Default destructor
   */
  virtual ~EnergyMortarAdapter() = default;

 private:
  /**
   * @brief Accumulate pair nodal gaps, areas, and first derivatives into the supplied nodal assembler.
   * @param mesh1 Current non-mortar mesh.
   * @param mesh2 Current mortar mesh.
   * @param pairs Geometry-approved pairs for the current calculation.
   * @param[in,out] assembler Nodal assembler receiving pair quantities and gradients.
   * @return Assembled dual true-DOF vectors and (dual, parent) true-DOF Jacobian matrices.
   */
  MfemNodalQuantityData assembleNodalQuantities( MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                 ArrayViewT<const InterfacePair> pairs,
                                                 MfemNodalAssembler& assembler ) const;

  /**
   * @brief Move the requested gap and weighted-gap derivative into the returned outputs.
   * @param quantities Assembled nodal fields whose gap and derivative are transferred.
   * @return Gap and dg_dx; other outputs are absent.
   */
  ContactFormulationResult takeGapResult( MfemNodalQuantityData& quantities ) const;

  /**
   * @brief Gather coordinates once for all calculations on the current element pair.
   * @param pair Element IDs in the supplied meshes.
   * @param mesh1 Current non-mortar mesh.
   * @param mesh2 Current mortar mesh.
   * @return Owning coordinates ordered by element, node, and spatial component.
   */
  static typename Calculator::Coordinates getPairCoordinates( const InterfacePair& pair, MeshData::Viewer mesh1,
                                                              MeshData::Viewer mesh2 );

  /**
   * @brief Find geometry-approved pairs for the current meshes
   *
   * Coarse search finds spatial candidates. Refined search then removes exact self-pairs, optionally removes
   * shared-node pairs, and rejects edges without nondegenerate opposing normals. Quadrature generation determines the
   * contributing interval within each pair calculation.
   *
   * @param mesh1 Current non-mortar mesh.
   * @param mesh2 Current mortar mesh.
   * @return Owned accepted pairs in coarse-search order.
   */
  ArrayT<InterfacePair> findCandidatePairs( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const;

  /**
   * @brief Controls penalty vs. Lagrange multiplier (LM) mode
   *
   * - true: penalty mode, dual vector is pressure
   * - false: LM mode, dual vector is the multiplier `lambda`
   */
  bool use_penalty_;

  /** @brief Coarse search supplied by the caller and owned by this formulation */
  std::unique_ptr<CoarseSearch> coarse_search_;

  /** @brief Refined search supplied by the caller and owned by this formulation */
  std::unique_ptr<RefinedSearch> refined_search_;

  /** @brief Value-owned generator configured by the factory; current geometry is supplied at each call */
  QuadratureRule quadrature_rule_;

  /**
   * @brief Tolerance used to avoid division by zero for area-weighted quantities
   */
  RealT area_tol_{ 1.0e-14 };

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

  /**
   * @brief Penalty stiffness for nodal penalty enforcement
   */
  RealT penalty_;

  /**
   * @brief Residual gap and quadrature-derivative settings
   */
  ContactParams params_;

  /**
   * @brief Evaluator holding a reference to quadrature_rule_, which is initialized before evaluator_
   */
  EnergyMortarCalculator<QuadratureRule> evaluator_;
};

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_ENERGYMORTARADAPTER_HPP_ */
