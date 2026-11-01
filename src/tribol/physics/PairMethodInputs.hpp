// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Read current penalty inputs and validate fixed parameters before preparing a pair definition.
 */
#ifndef SRC_TRIBOL_PHYSICS_PAIRMETHODINPUTS_HPP_
#define SRC_TRIBOL_PHYSICS_PAIRMETHODINPUTS_HPP_

#include <cmath>
#include <stdexcept>
#include <utility>
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/physics/PenaltyStiffness.hpp"

namespace tribol {
#ifdef BUILD_REDECOMP

/** @brief Fixed scalar parameters passed to an energy or force definition for the current request. */
struct PairDefinitionParameters {
  RealT penalty;       ///< Combined penalty stiffness.
  RealT residual_gap;  ///< Geometry-independent gap offset.
};

/**
 * @brief Read registered MFEM constant penalties on every request, with a fallback stiffness when absent.
 * @tparam PenaltyStiffness Callable combining the two registered constants; no mesh scales are applied.
 */
template <typename PenaltyStiffness = AveragePenaltyStiffness>
class RegisteredPenaltyInputs {
 public:
  /**
   * @brief Store penalty selection and combination configuration; retain no data pointers.
   * @param enabled Whether kinematic penalty enforcement has been configured.
   * @param combination Owned stiffness combination.
   */
  explicit RegisteredPenaltyInputs( bool enabled, PenaltyStiffness combination = {} )
      : enabled_( enabled ), combination_( std::move( combination ) )
  {
  }

  /**
   * @brief Report whether kinematic penalty enforcement is enabled.
   * @return True when penalty enforcement is active; false when Lagrange multiplier enforcement is active.
   */
  bool isEnabled() const noexcept { return enabled_; }

  /**
   * @brief Read current registered constants and pass the supplied gap offset through unchanged.
   * @param mesh_data Registered constants, which can change between requests.
   * @param mesh1 Current first mesh, unused by this input provider.
   * @param mesh2 Current second mesh, unused by this input provider.
   * @param residual_gap Fixed gap offset; no additional validation is applied here.
   * @return Combined constants, or fallback stiffness 1000.0 if either constant is absent, and the supplied gap offset.
   * @throws std::invalid_argument Penalty enforcement has not been configured; no parameters are returned.
   */
  PairDefinitionParameters prepare( const MfemMeshData& mesh_data, [[maybe_unused]] MeshData::Viewer mesh1,
                                    [[maybe_unused]] MeshData::Viewer mesh2, RealT residual_gap ) const
  {
    if ( !enabled_ ) throw std::invalid_argument( "Registered penalty inputs require configured penalty enforcement." );
    constexpr RealT default_penalty = 1000.0;
    const auto* penalty1 = mesh_data.GetMesh1KinematicConstantPenalty();
    const auto* penalty2 = mesh_data.GetMesh2KinematicConstantPenalty();
    return { penalty1 && penalty2 ? combination_( *penalty1, *penalty2 ) : default_penalty, residual_gap };
  }

 private:
  bool enabled_;                  ///< Whether this configuration can prepare penalty parameters.
  PenaltyStiffness combination_;  ///< Owned combination; no mesh data is retained.
};

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
   * @brief Combine current positive finite scaled penalties after checking the fixed gap offset.
   * @param mesh_data MFEM parent data, unused by this input provider.
   * @param mesh1 First mesh with current constant penalty and scale.
   * @param mesh2 Second mesh with current constant penalty and scale.
   * @param residual_gap Nonnegative finite offset subtracted from the geometric gap.
   * @return Combined scaled stiffness and gap offset.
   * @throws std::invalid_argument A scaled penalty or gap offset is invalid; no parameters are returned.
   */
  PairDefinitionParameters prepare( [[maybe_unused]] const MfemMeshData& mesh_data, MeshData::Viewer mesh1,
                                    MeshData::Viewer mesh2, RealT residual_gap ) const
  {
    const RealT penalty1 = mesh1.getElementData().m_penalty_stiffness * mesh1.getElementData().m_penalty_scale;
    const RealT penalty2 = mesh2.getElementData().m_penalty_stiffness * mesh2.getElementData().m_penalty_scale;
    if ( !std::isfinite( penalty1 ) || !std::isfinite( penalty2 ) || penalty1 <= 0.0 || penalty2 <= 0.0 ||
         !std::isfinite( residual_gap ) || residual_gap < 0.0 ) {
      throw std::invalid_argument(
          "Scaled mesh inputs require positive finite penalties and a nonnegative finite gap offset." );
    }
    return { combination_( penalty1, penalty2 ), residual_gap };
  }

 private:
  PenaltyStiffness combination_;  ///< Owned combination; no mesh data is retained.
};
#endif  // BUILD_REDECOMP
}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_PAIRMETHODINPUTS_HPP_
