// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Direct nodal and quadrature-point Energy Mortar contact formulations. */
#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTAR_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTAR_HPP_

#include "tribol/config.hpp"
#include "tribol/assembly/ContactEvaluation.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/search/PairListBuilder.hpp"

namespace tribol {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )

/** @brief Quadrature-point Energy Mortar pair search followed by scalar-energy evaluation and assembly. */
class QuadraturePointEnergyMortar final : public ContactFormulation {
 public:
  /** @brief Own the configured scalar-energy evaluation and pair search.
   * @param evaluation Non-null evaluation owning its pair function and referencing transfer data, which must outlive
   * this formulation. @param pair_builder Owned pair search.
   * @throws std::invalid_argument The evaluation is null; no formulation is constructed. */
  QuadraturePointEnergyMortar( std::unique_ptr<ContactEvaluation> evaluation, PairListBuilder pair_builder );

  /** @brief Evaluate scalar pair energies and assemble configured force outputs.
   * @param mesh1 Current first mesh. @param mesh2 Current second mesh.
   * @return Owned energy, force, and any configured force Jacobian. */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override;

 private:
  std::unique_ptr<ContactEvaluation> evaluation_;  ///< Owned energy calculation and its transfer-data references.
  PairListBuilder pair_builder_;                   ///< Owned pair search.
};

/** @brief Nodal Energy Mortar pair search followed by the configured penalty or LM evaluation. */
class NodalEnergyMortar final : public ContactFormulation {
 public:
  /** @brief Own the configured nodal evaluation and pair search.
   * @param evaluation Non-null evaluation owning its pair function and enforcement and referencing transfer data,
   * which must outlive this formulation. @param pair_builder Owned pair search.
   * @throws std::invalid_argument The evaluation is null; no formulation is constructed. */
  NodalEnergyMortar( std::unique_ptr<ContactEvaluation> evaluation, PairListBuilder pair_builder );

  /** @brief Find pairs and calculate the configured nodal gaps.
   * @param mesh1 Current first mesh. @param mesh2 Current second mesh.
   * @return Owned gap fields and their available coordinate derivatives. */
  ContactFormulationResult computeNodalGaps( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override;

  /** @brief Find pairs and calculate the configured nodal enforcement outputs.
   * @param mesh1 Current first mesh. @param mesh2 Current second mesh.
   * @return Owned gap, pressure, energy, force, and derivative fields provided by the evaluation. */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override;

 private:
  std::unique_ptr<ContactEvaluation> evaluation_;  ///< Owned nodal calculation and its transfer-data references.
  PairListBuilder pair_builder_;                   ///< Owned pair search.
};
#endif
}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_ENERGYMORTAR_HPP_
