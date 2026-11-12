// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Direct nodal and quadrature-point Energy Mortar contact formulations. */
#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTAR_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTAR_HPP_

#include "tribol/config.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/geom/OpposingNormalGap.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/physics/ContactEnforcement.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"
#include "tribol/search/PairListBuilder.hpp"

namespace tribol {

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
/**
 * @brief Energy Mortar with independent pair energies evaluated at quadrature points.
 * @tparam QuadratureRule Smoothed projected-edge pair quadrature.
 * @tparam Exec Concrete execution mode for evaluation and numerical assembly.
 */
template <typename QuadratureRule, ExecutionMode Exec>
class QuadraturePointEnergyMortar final : public ContactFormulation {
 public:
  using PairFunction = EnergyMortarEnergy<QuadratureRule, OffsetGap<OpposingNormalGap>>;  ///< Pair energy function.

  /**
   * @brief Borrow transfer data and own search and formulation configuration.
   * @param mesh_data Parent displacement transfer data.
   * @param jacobian_data Parent Jacobian transfer data.
   * @param pair_builder Owned pair search composition.
   * @param quadrature Owned pair quadrature.
   * @param gap Offset opposing-normal gap.
   * @param penalty Effective stiffness fixed for the lifetime of this formulation.
   * @throws std::invalid_argument Penalty is nonpositive or nonfinite; no formulation is constructed.
   */
  QuadraturePointEnergyMortar( MfemMeshData& mesh_data, MfemJacobianData& jacobian_data, PairListBuilder pair_builder,
                               QuadratureRule quadrature, OffsetGap<OpposingNormalGap> gap, RealT penalty )
      : mesh_data_( mesh_data ),
        jacobian_data_( jacobian_data ),
        pair_builder_( std::move( pair_builder ) ),
        function_( std::move( quadrature ), detail::checkedPenaltyStiffness( penalty ), std::move( gap ) )
  {
  }

  /** @brief Compute force, Jacobian, and energy. @param mesh1 First mesh. @param mesh2 Second mesh.
   * @return Caller-owned outputs. */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override;

 private:
  MfemMeshData& mesh_data_;          ///< Borrowed displacement transfer data.
  MfemJacobianData& jacobian_data_;  ///< Borrowed Jacobian transfer data.
  PairListBuilder pair_builder_;     ///< Owned search composition.
  const PairFunction function_;      ///< Owned pair quadrature, fixed stiffness, and gap definition.
};

/**
 * @brief Energy Mortar assembled from nodal weighted gaps and tributary areas.
 * @tparam QuadratureRule Smoothed projected-edge pair quadrature.
 * @tparam Exec Concrete execution mode for evaluation and numerical assembly.
 */
template <typename QuadratureRule, ExecutionMode Exec>
class NodalEnergyMortar final : public ContactFormulation {
 public:
  using PairFunction = EnergyMortarNodalEnergy<QuadratureRule, OffsetGap<OpposingNormalGap>>;  ///< Pair integrals.
  static constexpr RealT area_tolerance = 1.0e-14;  ///< Minimum supported tributary-area magnitude.

  /**
   * @brief Borrow parent/dual transfer data and own search and formulation configuration.
   * @param mesh_data Parent displacement transfer data.
   * @param submesh_data Dual pressure/gap transfer data.
   * @param jacobian_data Parent and dual Jacobian transfer data.
   * @param pair_builder Owned pair search composition.
   * @param quadrature Owned pair quadrature.
   * @param gap Offset opposing-normal gap.
   * @param enforcement Explicit penalty or Lagrange multiplier enforcement.
   */
  NodalEnergyMortar( MfemMeshData& mesh_data, MfemSubmeshData& submesh_data, MfemJacobianData& jacobian_data,
                     PairListBuilder pair_builder, QuadratureRule quadrature, OffsetGap<OpposingNormalGap> gap,
                     NodalContactEnforcement enforcement )
      : mesh_data_( mesh_data ),
        submesh_data_( submesh_data ),
        jacobian_data_( jacobian_data ),
        pair_builder_( std::move( pair_builder ) ),
        function_( std::move( quadrature ), std::move( gap ) ),
        enforcement_( std::move( enforcement ) )
  {
  }

  /** @brief Compute nodal gaps and their coordinate derivative. @param mesh1 First mesh. @param mesh2 Second mesh.
   * @return Caller-owned gap outputs. */
  ContactFormulationResult computeNodalGaps( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override;

  /** @brief Compute nodal enforcement outputs. @param mesh1 First mesh. @param mesh2 Second mesh.
   * @return Caller-owned gap, pressure, force, energy, and derivatives. */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override;

 private:
  /**
   * @brief Calculate nodal penalty forces and derivatives using the fixed stiffness.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @param stiffness Positive, finite effective penalty stiffness.
   * @return Owned normalized gaps, pressure, energy, force, and derivatives; df_dp is absent.
   * @throws std::invalid_argument Geometry or assembly dimensions are invalid; no result is returned.
   * @throws std::length_error Quadrature capacity or assembly storage is insufficient; no result is returned.
   */
  ContactFormulationResult computePenaltyForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                 RealT stiffness ) const;

  /**
   * @brief Calculate forces and derivatives from the current nodal Lagrange multipliers.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned weighted gaps, pressure, energy, force, coordinate derivatives, and df_dp.
   * @throws std::invalid_argument Geometry, assembly dimensions, or multiplier size are invalid;
   * no result is returned.
   * @throws std::length_error Quadrature capacity or assembly storage is insufficient; no result is returned.
   */
  ContactFormulationResult computeLagrangeMultiplierForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const;

  MfemMeshData& mesh_data_;                    ///< Borrowed displacement transfer data.
  MfemSubmeshData& submesh_data_;              ///< Borrowed dual transfer data.
  MfemJacobianData& jacobian_data_;            ///< Borrowed Jacobian transfer data.
  PairListBuilder pair_builder_;               ///< Owned search composition.
  const PairFunction function_;                ///< Owned pair quadrature and gap definition.
  const NodalContactEnforcement enforcement_;  ///< Explicit nodal enforcement equation.
};
#endif

}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_ENERGYMORTAR_HPP_
