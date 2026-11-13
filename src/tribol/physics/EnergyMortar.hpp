// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Direct nodal and quadrature-point Energy Mortar contact formulations. */
#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTAR_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTAR_HPP_

#include "tribol/config.hpp"
#include "tribol/assembly/PairEvaluationAssembly.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/EnergyMortarEvaluation.hpp"
#include "tribol/search/PairListBuilder.hpp"
#include <utility>

namespace tribol {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )

/**
 * @brief Quadrature-point Energy Mortar with energy, force, and force-Jacobian assembly.
 * @tparam Exec Concrete execution mode for search, evaluation, and assembly kernels.
 * @tparam PairFunction Fixed scalar pair-energy function.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
class QuadraturePointEnergyMortar final : public ContactFormulation {
 public:
  /**
   * @brief Own the pair energy and search and reference vector and matrix transfer data.
   * @param mesh_data Displacement transfer data; must outlive this formulation.
   * @param jacobian_data Coordinate-matrix transfer data; must outlive this formulation.
   * @param pair_function Configured scalar pair-energy function, retained by value.
   * @param pair_builder Owned pair search.
   */
  QuadraturePointEnergyMortar( const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data,
                               PairFunction pair_function, PairListBuilder pair_builder )
      : mesh_data_( mesh_data ),
        jacobian_data_( jacobian_data ),
        pair_function_( std::move( pair_function ) ),
        pair_builder_( std::move( pair_builder ) )
  {
  }

  /**
   * @brief Evaluate pair energies and assemble energy, force, and coordinate Jacobian.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned energy, force, and df_dx; other outputs are absent.
   */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override
  {
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    return evaluateAndAssemblePairEnergyJacobian<Exec>( pairs, mesh1, mesh2, mesh_data_, jacobian_data_,
                                                        pair_function_ );
  }

 private:
  const MfemMeshData& mesh_data_;          ///< Displacement transfer data referenced for this formulation's lifetime.
  const MfemJacobianData& jacobian_data_;  ///< Matrix transfer data referenced for this formulation's lifetime.
  const PairFunction pair_function_;       ///< Owned, fixed scalar pair-energy function.
  PairListBuilder pair_builder_;           ///< Owned pair search.
};

/**
 * @brief Nodal penalty Energy Mortar with normalized compressive gaps and penalty forces.
 * @tparam Exec Concrete execution mode for search, evaluation, and assembly kernels.
 * @tparam PairFunction Pair function returning weighted-gap integrals followed by tributary areas.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
class NodalPenaltyEnergyMortar final : public ContactFormulation {
 public:
  /**
   * @brief Own the nodal pair function, penalty enforcement, and search and reference required transfer data.
   * @param mesh_data Displacement transfer data; must outlive this formulation.
   * @param submesh_data Pressure and gap transfer data; must outlive this formulation.
   * @param jacobian_data Parent and scalar-field Jacobian transfer data; must outlive this formulation.
   * @param pair_function Configured nodal-integral function, retained by value.
   * @param penalty Fixed positive, finite penalty enforcement.
   * @param pair_builder Owned pair search.
   */
  NodalPenaltyEnergyMortar( const MfemMeshData& mesh_data, const MfemSubmeshData& submesh_data,
                            const MfemJacobianData& jacobian_data, PairFunction pair_function,
                            PenaltyEnforcement penalty, PairListBuilder pair_builder )
      : mesh_data_( mesh_data ),
        submesh_data_( submesh_data ),
        jacobian_data_( jacobian_data ),
        pair_function_( std::move( pair_function ) ),
        penalty_( std::move( penalty ) ),
        pair_builder_( std::move( pair_builder ) )
  {
  }

  /**
   * @brief Calculate normalized compressive nodal gaps and their coordinate Jacobian.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned gap and dg_dx; other outputs are absent.
   */
  ContactFormulationResult computeNodalGaps( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override
  {
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    return evaluateAndAssembleNodalPenaltyGaps<Exec>( pairs, mesh1, mesh2, mesh_data_, submesh_data_, jacobian_data_,
                                                      pair_function_ );
  }

  /**
   * @brief Calculate nodal penalty energy, gaps, pressure, forces, and coordinate derivatives.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned energy, gap, pressure, force, dg_dx, and df_dx; df_dp is absent.
   */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override
  {
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    return evaluateAndAssembleNodalPenalty<Exec>( pairs, mesh1, mesh2, mesh_data_, submesh_data_, jacobian_data_,
                                                  pair_function_, penalty_ );
  }

 private:
  const MfemMeshData& mesh_data_;          ///< Displacement transfer data referenced for this formulation's lifetime.
  const MfemSubmeshData& submesh_data_;    ///< Scalar transfer data referenced for this formulation's lifetime.
  const MfemJacobianData& jacobian_data_;  ///< Matrix transfer data referenced for this formulation's lifetime.
  const PairFunction pair_function_;       ///< Owned, fixed nodal-integral function.
  const PenaltyEnforcement penalty_;       ///< Fixed nodal penalty enforcement.
  PairListBuilder pair_builder_;           ///< Owned pair search.
};

/**
 * @brief Nodal Lagrange multiplier Energy Mortar with signed weighted gaps and mixed derivatives.
 * @tparam Exec Concrete execution mode for search, evaluation, and assembly kernels.
 * @tparam PairFunction Pair function returning weighted-gap integrals followed by tributary areas.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
class NodalLagrangeMultiplierEnergyMortar final : public ContactFormulation {
 public:
  /**
   * @brief Own the nodal pair function and search and reference required transfer data.
   * @param mesh_data Displacement transfer data; must outlive this formulation.
   * @param submesh_data Pressure and gap transfer data; must outlive this formulation.
   * @param jacobian_data Parent and scalar-field Jacobian transfer data; must outlive this formulation.
   * @param pair_function Configured nodal-integral function, retained by value.
   * @param pair_builder Owned pair search.
   */
  NodalLagrangeMultiplierEnergyMortar( const MfemMeshData& mesh_data, const MfemSubmeshData& submesh_data,
                                       const MfemJacobianData& jacobian_data, PairFunction pair_function,
                                       PairListBuilder pair_builder )
      : mesh_data_( mesh_data ),
        submesh_data_( submesh_data ),
        jacobian_data_( jacobian_data ),
        pair_function_( std::move( pair_function ) ),
        pair_builder_( std::move( pair_builder ) )
  {
  }

  /**
   * @brief Calculate signed weighted nodal gaps and their coordinate Jacobian.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned gap and dg_dx; other outputs are absent.
   */
  ContactFormulationResult computeNodalGaps( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override
  {
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    return evaluateAndAssembleNodalLagrangeMultiplierGaps<Exec>( pairs, mesh1, mesh2, mesh_data_, submesh_data_,
                                                                 jacobian_data_, pair_function_ );
  }

  /**
   * @brief Calculate nodal LM energy, gaps, pressure, forces, and mixed derivatives.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned energy, gap, pressure, force, dg_dx, df_dx, and df_dp.
   */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override
  {
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    return evaluateAndAssembleNodalLagrangeMultiplier<Exec>( pairs, mesh1, mesh2, mesh_data_, submesh_data_,
                                                             jacobian_data_, pair_function_ );
  }

 private:
  const MfemMeshData& mesh_data_;          ///< Displacement transfer data referenced for this formulation's lifetime.
  const MfemSubmeshData& submesh_data_;    ///< Scalar transfer data referenced for this formulation's lifetime.
  const MfemJacobianData& jacobian_data_;  ///< Matrix transfer data referenced for this formulation's lifetime.
  const PairFunction pair_function_;       ///< Owned, fixed nodal-integral function.
  PairListBuilder pair_builder_;           ///< Owned pair search.
};

#endif
}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_ENERGYMORTAR_HPP_
