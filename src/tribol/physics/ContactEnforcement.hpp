// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Explicit contact enforcement values and their current penalty inputs. */
#ifndef SRC_TRIBOL_PHYSICS_CONTACTENFORCEMENT_HPP_
#define SRC_TRIBOL_PHYSICS_CONTACTENFORCEMENT_HPP_

#include <cmath>
#include <stdexcept>
#include <utility>
#include <variant>

#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/physics/PenaltyStiffness.hpp"

namespace tribol {
#ifdef BUILD_REDECOMP

/** @brief Select nodal Lagrange multiplier enforcement without penalty parameters. */
struct LagrangeMultiplierEnforcement {};

/** @brief Preserve an invalid quadrature-point configuration until a calculation reports the missing penalty. */
struct UnconfiguredPenaltyEnforcement {
  /**
   * @brief Report that quadrature-point Energy Mortar requires penalty enforcement.
   * @param mesh_data Current MFEM mesh data.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @throws std::invalid_argument Always.
   */
  RealT prepare( [[maybe_unused]] const MfemMeshData& mesh_data, [[maybe_unused]] MeshData::Viewer mesh1,
                 [[maybe_unused]] MeshData::Viewer mesh2 ) const
  {
    throw std::invalid_argument( "Quadrature-point Energy Mortar requires configured penalty enforcement." );
  }
};

/**
 * @brief Read registered MFEM constant penalties on every request.
 * @tparam PenaltyStiffness Callable combining the two registered constants; no mesh scales are applied.
 */
template <typename PenaltyStiffness = AveragePenaltyStiffness>
class RegisteredPenaltyInputs {
 public:
  /** @brief Store the stiffness combination. @param combination Owned stiffness combination. */
  explicit RegisteredPenaltyInputs( PenaltyStiffness combination = {} ) : combination_( std::move( combination ) ) {}

  /**
   * @brief Read and combine current registered constant penalties.
   * @param mesh_data Registered constants, which can change between requests.
   * @param mesh1 Current first mesh, unused by this input provider.
   * @param mesh2 Current second mesh, unused by this input provider.
   * @return Combined stiffness, or fallback stiffness 1000.0 if either constant is absent.
   */
  RealT prepare( const MfemMeshData& mesh_data, [[maybe_unused]] MeshData::Viewer mesh1,
                 [[maybe_unused]] MeshData::Viewer mesh2 ) const
  {
    constexpr RealT default_penalty = 1000.0;
    const auto* penalty1 = mesh_data.GetMesh1KinematicConstantPenalty();
    const auto* penalty2 = mesh_data.GetMesh2KinematicConstantPenalty();
    return penalty1 && penalty2 ? combination_( *penalty1, *penalty2 ) : default_penalty;
  }

 private:
  PenaltyStiffness combination_;  ///< Owned combination; no mesh data is retained.
};

/** @brief Select penalty enforcement and own its current-input reader. @tparam Inputs Penalty input provider. */
template <typename Inputs>
struct PenaltyEnforcement {
  Inputs inputs{};  ///< Current penalty input provider.

  /**
   * @brief Prepare current penalty parameters.
   * @param mesh_data Current MFEM mesh data.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Current effective stiffness.
   */
  RealT prepare( const MfemMeshData& mesh_data, MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
  {
    return inputs.prepare( mesh_data, mesh1, mesh2 );
  }
};

/** @brief Registered penalty enforcement value. @tparam PenaltyStiffness Stiffness combination. */
template <typename PenaltyStiffness = AveragePenaltyStiffness>
using RegisteredPenaltyEnforcement = PenaltyEnforcement<RegisteredPenaltyInputs<PenaltyStiffness>>;

/** @brief Runtime selection between the two supported nodal enforcement equations. */
template <typename PenaltyStiffness = AveragePenaltyStiffness>
using NodalContactEnforcement =
    std::variant<RegisteredPenaltyEnforcement<PenaltyStiffness>, LagrangeMultiplierEnforcement>;

/** @brief Runtime selection between valid and missing quadrature-point penalty configuration. */
template <typename PenaltyStiffness = AveragePenaltyStiffness>
using QuadraturePointContactEnforcement =
    std::variant<RegisteredPenaltyEnforcement<PenaltyStiffness>, UnconfiguredPenaltyEnforcement>;

/**
 * @brief Read mesh penalties, apply current mesh scales, and validate them for each request.
 * @tparam PenaltyStiffness Callable combining the two scaled penalties.
 */
template <typename PenaltyStiffness = SeriesPenaltyStiffness>
class ScaledMeshPenaltyInputs {
 public:
  /** @brief Store the stiffness combination. @param combination Owned stiffness combination. */
  explicit ScaledMeshPenaltyInputs( PenaltyStiffness combination = {} ) : combination_( std::move( combination ) ) {}

  /**
   * @brief Validate and combine current positive finite scaled penalties.
   * @param mesh_data MFEM parent data, unused by this input provider.
   * @param mesh1 First mesh with current constant penalty and scale.
   * @param mesh2 Second mesh with current constant penalty and scale.
   * @return Combined scaled stiffness.
   * @throws std::invalid_argument A scaled penalty is invalid; no stiffness is returned.
   */
  RealT prepare( [[maybe_unused]] const MfemMeshData& mesh_data, MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
  {
    const RealT penalty1 = mesh1.getElementData().m_penalty_stiffness * mesh1.getElementData().m_penalty_scale;
    const RealT penalty2 = mesh2.getElementData().m_penalty_stiffness * mesh2.getElementData().m_penalty_scale;
    if ( !std::isfinite( penalty1 ) || !std::isfinite( penalty2 ) || penalty1 <= 0.0 || penalty2 <= 0.0 ) {
      throw std::invalid_argument( "Scaled mesh inputs require positive finite penalties." );
    }
    return combination_( penalty1, penalty2 );
  }

 private:
  PenaltyStiffness combination_;  ///< Owned combination; no mesh data is retained.
};

#endif  // BUILD_REDECOMP
}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_CONTACTENFORCEMENT_HPP_
