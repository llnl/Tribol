// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Data structures, field validation helpers, and pure-math timestep criteria for contact methods.
 */
#ifndef SRC_TRIBOL_PHYSICS_TIMESTEP_TIMESTEPCRITERIA_HPP_
#define SRC_TRIBOL_PHYSICS_TIMESTEP_TIMESTEPCRITERIA_HPP_

#include <cmath>

#include "axom/core/NumericLimits.hpp"
#include "axom/core/utilities/Utilities.hpp"
#include "axom/primal/geometry/Point.hpp"
#include "axom/primal/geometry/Vector.hpp"
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Atomics.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/utils/Math.hpp"

namespace tribol {

/**
 * @brief Configuration passed by value into timestep criteria on host or device.
 */
struct TimestepSettings {
  TimestepCriteria active_criteria{};                 ///< Active timestep criteria selected at runtime.
  RealT pen_frac{ 0.3 };                              ///< Max allowable interpenetration fraction of element thickness.
  RealT explicit_integrator_stability_factor{ 2.0 };  ///< Explicit integrator stability factor (2.0 for central diff).
  CommT communicator{ TRIBOL_COMM_WORLD };            ///< Communicator used for collective timestep reductions.
  bool use_mpi{ false };                              ///< True to perform MPI_Allreduce across communicator.
};

/**
 * @brief Point-level geometric and kinematic inputs for penetration-fraction timestep criteria.
 *
 * 2D callers leave the z-component of points, velocities, and contact_normal at 0.0.
 */
struct TimestepPointData {
  axom::primal::Point<RealT, 3> point1{};           ///< Physical point on geometry1.
  axom::primal::Point<RealT, 3> point2{};           ///< Physical point on geometry2.
  axom::primal::Vector<RealT, 3> vel1{};            ///< Interpolated velocity at point1.
  axom::primal::Vector<RealT, 3> vel2{};            ///< Interpolated velocity at point2.
  axom::primal::Vector<RealT, 3> contact_normal{};  ///< Unit contact normal pointing from geometry1 toward geometry2.
  RealT thickness1{ 0.0 };                          ///< Volume element thickness associated with geometry1.
  RealT thickness2{ 0.0 };                          ///< Volume element thickness associated with geometry2.
  bool in_contact{ false };                         ///< True if the point is currently in compressive contact.
};

/**
 * @brief Pair-level inputs for the pair-local spring-mass CFL criterion.
 */
struct TimestepPairSpringMassData {
  RealT inv_mass1{ 0.0 };                   ///< Effective normal-direction inverse mass of geometry1's element.
  RealT inv_mass2{ 0.0 };                   ///< Effective normal-direction inverse mass of geometry2's element.
  RealT penalty_stiffness_per_area{ 0.0 };  ///< Effective normal penalty stiffness per unit overlap measure.
  RealT overlap_measure{ 0.0 };             ///< Active overlap length (2D) or area (3D).
  bool in_contact{ false };                 ///< True if the pair has active compressive contact.
};

/**
 * @brief Point-level inputs for accumulating assembled nodal penalty CFL row sums.
 */
struct TimestepNodalPenaltyPointData {
  static constexpr int max_spatial_dim = 3;  ///< Maximum supported spatial dimension.
  static constexpr int max_size_basis = 9;   ///< Maximum basis size per surface element (up to Q2 quad).

  RealT integration_weight{ 0.0 };                  ///< Physical integration weight (length in 2D, area in 3D).
  RealT penalty_stiffness{ 0.0 };                   ///< Normal penalty stiffness per unit overlap measure.
  RealT normal_rate_penalty{ 0.0 };                 ///< Active normal gap-rate penalty coefficient (0 when inactive).
  RealT tangential_viscous_coeff{ 0.0 };            ///< Tangential damping coefficient applied without overlap weight.
  axom::primal::Vector<RealT, 3> contact_normal{};  ///< Unit contact normal at the integration point.
  int spatial_dim{ 3 };                             ///< Spatial dimension (2 or 3).
  int size_basis[2]{ 0, 0 };                        ///< Basis size on geometry1 and geometry2.
  RealT basis_values[2][max_size_basis]{};          ///< Basis function values at the integration point.
  RealT inverse_mass[2][max_size_basis][max_spatial_dim]{};  ///< Component-wise inverse diagonal mass per basis.
  IndexT dof_indices[2][max_size_basis][max_spatial_dim]{};  ///< Local accumulation DOF index per basis/dimension.
};

namespace detail {

/**
 * @brief Check whether a scalar inverse mass entry is finite and nonnegative.
 * @param value Inverse mass entry.
 * @return True if value is nonnegative and finite.
 */
TRIBOL_HOST_DEVICE inline bool isValidInverseMass( RealT value )
{
  return value >= 0.0 && value == value && value <= axom::numeric_limits<RealT>::max();
}

}  // namespace detail

/**
 * @brief Compute the effective normal-direction inverse mass of a surface element from component-wise nodal inverse
 * masses.
 *
 * For each node a of the element, the normal-direction inverse mass is sum_d n_d^2 * m^{-1}_{a,d}. If any node is
 * constrained in the normal direction (normal inverse mass is zero), the element has infinite effective mass in that
 * direction and this function returns 0.0. Otherwise it returns 1 / sum_a (1 / m^{-1}_{a,n}).
 *
 * @tparam MeshViewer MeshData::Viewer-compatible view providing numberOfNodesPerElement(), getGlobalNodeId(), and
 * getInverseMass().
 * @param mesh Surface mesh viewer with inverse diagonal mass registered.
 * @param element_id Surface element index.
 * @param contact_normal Unit contact normal (first spatial_dim entries used).
 * @param spatial_dim Spatial dimension (2 or 3).
 * @param[out] invalid_mass Set to true if any encountered inverse mass entry is negative or nonfinite.
 * @return Effective normal-direction inverse mass of the surface element, or 0.0 if rigid/constrained.
 */
template <typename MeshViewer>
TRIBOL_HOST_DEVICE inline RealT computeSurfaceElementNormalInverseMass(
    const MeshViewer& mesh, IndexT element_id, const axom::primal::Vector<RealT, 3>& contact_normal, int spatial_dim,
    bool& invalid_mass )
{
  const IndexT num_nodes = mesh.numberOfNodesPerElement();
  RealT total_mass = 0.0;
  bool constrained = false;
  for ( IndexT node = 0; node < num_nodes; ++node ) {
    const IndexT node_id = mesh.getGlobalNodeId( element_id, node );
    RealT normal_inv_mass = 0.0;
    for ( int d = 0; d < spatial_dim; ++d ) {
      const RealT inv_m = mesh.getInverseMass( node_id, d );
      if ( !detail::isValidInverseMass( inv_m ) ) {
        invalid_mass = true;
        return 0.0;
      }
      normal_inv_mass += contact_normal[d] * contact_normal[d] * inv_m;
    }
    if ( normal_inv_mass == 0.0 ) {
      // A constrained node makes the effective mass infinite. Still validate the other nodes' inverse masses.
      constrained = true;
    } else {
      total_mass += 1.0 / normal_inv_mass;
    }
  }
  return !constrained && total_mass > 0.0 ? 1.0 / total_mass : 0.0;
}

/**
 * @brief Limit timestep when current interpenetration exceeds pen_frac * min(thickness1, thickness2).
 */
struct CurrentGapTimestepCriterion {
  /**
   * @brief Evaluate the current-gap penetration criterion at one point using relative velocity along contact_normal.
   * @param point Geometric and kinematic state at the evaluation point.
   * @param settings Timestep settings supplying pen_frac.
   * @return Allowable timestep, or +infinity if this criterion does not vote.
   */
  TRIBOL_HOST_DEVICE static RealT evaluate( const TimestepPointData& point, const TimestepSettings& settings )
  {
    const RealT inf = axom::numeric_limits<RealT>::infinity();
    if ( !point.in_contact ) {
      return inf;
    }

    const RealT closing_speed = -( point.vel1 - point.vel2 ).dot( point.contact_normal );
    if ( closing_speed <= zero_div_tol ) {
      return inf;
    }

    const RealT gap = axom::primal::Vector<RealT, 3>( point.point2, point.point1 ).dot( point.contact_normal );
    const RealT max_delta = settings.pen_frac * axom::utilities::min( point.thickness1, point.thickness2 );
    if ( -gap <= max_delta ) {
      return inf;
    }

    const RealT dt = max_delta / closing_speed;
    return dt > 0.0 ? dt : inf;
  }
};

/**
 * @brief Limit timestep when velocity-projected gap over current_dt exceeds pen_frac * min(thickness1, thickness2).
 */
struct VelocityProjectionTimestepCriterion {
  /**
   * @brief Evaluate the velocity-projection criterion at one point using relative velocity along contact_normal.
   * @param point Geometric and kinematic state at the evaluation point.
   * @param settings Timestep settings supplying pen_frac.
   * @param current_dt Incoming simulation timestep for velocity projection (<= 0 disables this criterion).
   * @return Allowable timestep, or +infinity if this criterion does not vote.
   */
  TRIBOL_HOST_DEVICE static RealT evaluate( const TimestepPointData& point, const TimestepSettings& settings,
                                            RealT current_dt )
  {
    const RealT inf = axom::numeric_limits<RealT>::infinity();
    if ( current_dt <= 0.0 ) {
      return inf;
    }

    const RealT closing_speed = -( point.vel1 - point.vel2 ).dot( point.contact_normal );
    if ( closing_speed <= zero_div_tol ) {
      return inf;
    }

    const RealT gap = axom::primal::Vector<RealT, 3>( point.point2, point.point1 ).dot( point.contact_normal );
    const RealT projected_gap = gap - current_dt * closing_speed;
    const RealT max_delta = settings.pen_frac * axom::utilities::min( point.thickness1, point.thickness2 );
    if ( -projected_gap <= max_delta ) {
      return inf;
    }

    const RealT dt = max_delta / closing_speed;
    return dt > 0.0 ? dt : inf;
  }
};

/**
 * @brief Pair-local spring-mass CFL criterion using face effective normal inverse masses and overlap measure.
 *
 * Computes dt = C_stab / sqrt(k_pen * A_overlap * (inv_mass1 + inv_mass2)).
 */
struct PairSpringMassCflTimestepCriterion {
  /**
   * @brief Evaluate the pair-local spring-mass CFL timestep limit.
   * @param pair Pair-level inverse masses, penalty stiffness per area, and overlap measure.
   * @param settings Timestep settings supplying explicit_integrator_stability_factor.
   * @return Allowable timestep, or +infinity if this criterion does not vote.
   */
  TRIBOL_HOST_DEVICE static RealT evaluate( const TimestepPairSpringMassData& pair, const TimestepSettings& settings )
  {
    const RealT inf = axom::numeric_limits<RealT>::infinity();
    if ( !pair.in_contact || pair.penalty_stiffness_per_area <= 0.0 || pair.overlap_measure <= 0.0 ) {
      return inf;
    }
    const RealT effective_inv_mass = pair.inv_mass1 + pair.inv_mass2;
    if ( effective_inv_mass <= 0.0 ) {
      return inf;
    }
    const RealT stiffness_bound = pair.penalty_stiffness_per_area * pair.overlap_measure * effective_inv_mass;
    const RealT dt_cfl = settings.explicit_integrator_stability_factor / std::sqrt( stiffness_bound );
    return ( std::isfinite( dt_cfl ) && dt_cfl > 0.0 ) ? dt_cfl : inf;
  }
};

/**
 * @brief Assembled nodal penalty CFL criterion bounding M^{-1/2} K M^{-1/2} and M^{-1/2} C M^{-1/2} via row sums
 * (Gershgorin's circle theorem).
 */
struct NodalPenaltyCflTimestepCriterion {
  /**
   * @brief Accumulate one active integration point's mass-normalized row-sum contributions into DOF buffers.
   * @param point Integration point weights, penalties, normal, basis values, inverse masses, and DOF indices.
   * @param stiffness_row_sums Per-DOF stiffness row-sum accumulator (length >= max touched DOF + 1).
   * @param damping_row_sums Optional per-DOF damping row-sum accumulator (may be empty if undamped).
   * @param invalid_mass Single-element flag set to 1 if an invalid inverse mass or out-of-range DOF index is seen.
   */
  TRIBOL_HOST_DEVICE static void accumulatePoint( const TimestepNodalPenaltyPointData& point,
                                                  ArrayViewT<RealT> stiffness_row_sums,
                                                  ArrayViewT<RealT> damping_row_sums, ArrayViewT<int> invalid_mass )
  {
    if ( point.integration_weight <= 0.0 ) {
      return;
    }

    const bool accumulate_damping =
        !damping_row_sums.empty() && ( point.normal_rate_penalty > 0.0 || point.tangential_viscous_coeff > 0.0 );
    const bool include_tangential = accumulate_damping && point.tangential_viscous_coeff > 0.0;

    RealT sqrt_inv_mass[2][TimestepNodalPenaltyPointData::max_size_basis]
                       [TimestepNodalPenaltyPointData::max_spatial_dim]{};
    RealT normal_abs_sum = 0.0;
    RealT tangential_abs_sums[TimestepNodalPenaltyPointData::max_spatial_dim]{ 0.0, 0.0, 0.0 };

    for ( int surface = 0; surface < 2; ++surface ) {
      for ( int basis_index = 0; basis_index < point.size_basis[surface]; ++basis_index ) {
        const RealT abs_basis = std::abs( point.basis_values[surface][basis_index] );
        for ( int d = 0; d < point.spatial_dim; ++d ) {
          const RealT inv_m = point.inverse_mass[surface][basis_index][d];
          if ( !detail::isValidInverseMass( inv_m ) ) {
            tribol::atomicMax( &invalid_mass[0], 1 );
            return;
          }
          const RealT sqrt_inv_m = std::sqrt( inv_m );
          sqrt_inv_mass[surface][basis_index][d] = sqrt_inv_m;
          normal_abs_sum += abs_basis * std::abs( point.contact_normal[d] ) * sqrt_inv_m;

          if ( include_tangential ) {
            for ( int target_d = 0; target_d < point.spatial_dim; ++target_d ) {
              const RealT tangent_proj =
                  ( target_d == d ? 1.0 : 0.0 ) - point.contact_normal[target_d] * point.contact_normal[d];
              tangential_abs_sums[target_d] += abs_basis * sqrt_inv_m * std::abs( tangent_proj );
            }
          }
        }
      }
    }

    const RealT stiffness_scale = point.integration_weight * point.penalty_stiffness;
    const RealT normal_damping_scale = point.integration_weight * point.normal_rate_penalty;
    const IndexT stiffness_size = stiffness_row_sums.size();

    for ( int surface = 0; surface < 2; ++surface ) {
      for ( int basis_index = 0; basis_index < point.size_basis[surface]; ++basis_index ) {
        const RealT abs_basis = std::abs( point.basis_values[surface][basis_index] );
        for ( int d = 0; d < point.spatial_dim; ++d ) {
          const RealT sqrt_inv_m = sqrt_inv_mass[surface][basis_index][d];
          if ( sqrt_inv_m == 0.0 ) {
            continue;
          }
          const IndexT dof = point.dof_indices[surface][basis_index][d];
          if ( dof < 0 || dof >= stiffness_size ) {
            tribol::atomicMax( &invalid_mass[0], 1 );
            return;
          }
          const RealT abs_normal_coeff = abs_basis * std::abs( point.contact_normal[d] ) * sqrt_inv_m;
          tribol::atomicAdd( &stiffness_row_sums[dof], stiffness_scale * abs_normal_coeff * normal_abs_sum );
          if ( accumulate_damping ) {
            const RealT damping_contrib =
                normal_damping_scale * abs_normal_coeff * normal_abs_sum +
                ( include_tangential ? point.tangential_viscous_coeff * abs_basis * sqrt_inv_m * tangential_abs_sums[d]
                                     : 0.0 );
            if ( damping_contrib > 0.0 ) {
              tribol::atomicAdd( &damping_row_sums[dof], damping_contrib );
            }
          }
        }
      }
    }
  }

  /**
   * @brief Compute the stability timestep from global stiffness and damping bounds.
   * @param stiffness_bound Conservative upper bound on lambda_max(M^{-1/2} K M^{-1/2}).
   * @param damping_bound Conservative upper bound on lambda_max(M^{-1/2} C M^{-1/2}).
   * @param settings Timestep settings supplying explicit_integrator_stability_factor.
   * @return Allowable timestep, or +infinity when the damped frequency bound is zero.
   */
  TRIBOL_HOST_DEVICE static RealT finalizeFromBounds( RealT stiffness_bound, RealT damping_bound,
                                                      const TimestepSettings& settings )
  {
    const RealT damped_frequency_bound =
        std::sqrt( stiffness_bound + 0.25 * damping_bound * damping_bound ) + 0.5 * damping_bound;
    return damped_frequency_bound > 0.0 ? settings.explicit_integrator_stability_factor / damped_frequency_bound
                                        : axom::numeric_limits<RealT>::infinity();
  }
};

}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_TIMESTEP_TIMESTEPCRITERIA_HPP_
