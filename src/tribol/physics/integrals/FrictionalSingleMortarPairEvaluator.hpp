// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Contract integrated mortar weights into frictional single mortar kinematic gaps, residuals, and partials.
 */
#ifndef SRC_TRIBOL_PHYSICS_INTEGRALS_FRICTIONALSINGLEMORTARPAIREVALUATOR_HPP_
#define SRC_TRIBOL_PHYSICS_INTEGRALS_FRICTIONALSINGLEMORTARPAIREVALUATOR_HPP_

#include "tribol/assembly/EnzymePairDerivative.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/QuadratureCommon.hpp"
#include "tribol/physics/integrals/PairIntegrals.hpp"

namespace tribol::detail {

/**
 * @brief Evaluate frictional single mortar 3D kinematic gaps, residuals, and partial derivatives from mortar weights.
 *
 * Uses a global Cartesian 3D Lagrange multiplier lambda_a at each geometry1 node a. The 3D kinematic vector is
 *   k_a = g_{N,a} * n_a + dg_{T,a} = dg_a + (n_a . g_a^n) * n_a,
 * where g_a^n = -sum_b D_{ab} x_b^{(1),n} + sum_c M_{ac} x_c^{(2),n} and
 * dg_a = -sum_b D_{ab} (x_b^{(1)} - x_b^{(1),n}) + sum_c M_{ac} (x_c^{(2)} - x_c^{(2),n}).
 * Its normal projection n_a . k_a equals the weighted normal gap g_{N,a}, and its tangential projection
 * (I - n_a x n_a) k_a equals the weighted tangential incremental slip dg_{T,a}.
 *
 * All quadrature integration is delegated to PairIntegral with MortarWeightsIntegrand, and all coordinate
 * differentiation of the mortar weights is delegated to EnzymePairDerivative.
 *
 * @tparam RuleGenerator Configured generator for the full projected overlap.
 */
template <typename RuleGenerator>
class FrictionalSingleMortarPairEvaluator {
  using Geometry1 = typename RuleGenerator::Geometry1;  ///< Non-mortar surface element geometry.
  using Geometry2 = typename RuleGenerator::Geometry2;  ///< Mortar surface element geometry.
  using Basis1 = typename Geometry1::Basis;             ///< Displacement and multiplier basis on geometry1.
  using Basis2 = typename Geometry2::Basis;             ///< Displacement basis on geometry2.

 public:
  /// Pair integral computing contiguous geometry1 and geometry2 mortar weight matrices.
  using WeightsIntegral = PairIntegral<RuleGenerator, MortarWeightsIntegrand<Basis1, Basis1, Basis2>>;
  /// Contiguous mortar weight result returned by WeightsIntegral.
  using Weights = typename WeightsIntegral::Result;
  static constexpr IndexT num_nodes1 = Geometry1::num_nodes;                    ///< Number of geometry1 nodes.
  static constexpr IndexT num_nodes2 = Geometry2::num_nodes;                    ///< Number of geometry2 nodes.
  static constexpr IndexT spatial_dim = Geometry1::spatial_dim;                 ///< Number of spatial coordinates.
  static constexpr IndexT size_geom1_coords = num_nodes1 * spatial_dim;         ///< Geometry1 coordinate/dual extent.
  static constexpr IndexT size_geom2_coords = num_nodes2 * spatial_dim;         ///< Geometry2 coordinate extent.
  static constexpr IndexT size_coords = size_geom1_coords + size_geom2_coords;  ///< Total pair coordinate extent.
  using Coords1 = typename Geometry1::Coords;                                   ///< Geometry1 coordinates/normals view.
  using Coords2 = typename Geometry2::Coords;                                   ///< Geometry2 coordinates view.
  using Multipliers = Coords1;                                                  ///< Geometry1 3D multipliers [node][d].
  using FrozenGapJacobian = MdArray<RealT, size_geom1_coords, size_coords>;     ///< 3D dual rows and coord columns.

  /** @brief Pair 3D kinematic gaps (normal gap + tangential incremental slip) and frozen-geometry Jacobian. */
  struct GapEvaluation {
    MdArray<RealT, size_geom1_coords> kinematic_gap{};  ///< 3D kinematic gap k_a in [node * spatial_dim + d] order.
    FrozenGapJacobian frozen_gap_jacobian{};            ///< Coordinate derivative holding weights and normals fixed.
    QuadratureMetadata metadata{};                      ///< Status and point count; invalid/empty rules give zeros.
  };

  /** @brief Pair 3D kinematic gaps, frozen-geometry Jacobian, and nodal residuals. */
  struct ResidualEvaluation {
    GapEvaluation gap{};                      ///< 3D kinematic gaps, frozen gap Jacobian, and quadrature metadata.
    MdArray<RealT, size_coords> residuals{};  ///< Nodal residuals ordered as geometry1 followed by geometry2.
  };

#ifdef TRIBOL_USE_ENZYME
  /** @brief 3D kinematic gaps together with coordinate and normal partials from one differentiated weight call. */
  struct GapEvaluationAndPartials {
    GapEvaluation evaluation{};                                     ///< Primal 3D kinematic gaps and frozen Jacobian.
    MdArray<RealT, size_geom1_coords, size_coords> gap_dx{};        ///< Gap coordinate partials with fixed normals.
    MdArray<RealT, size_geom1_coords, size_geom1_coords> gap_dn{};  ///< Gap normal partials with fixed coordinates.
  };

  /** @brief 3D kinematic gaps, residuals, and their coordinate and normal partials from one differentiated call. */
  struct ResidualEvaluationAndPartials {
    ResidualEvaluation evaluation{};                                ///< Primal gaps, residuals, and frozen Jacobian.
    MdArray<RealT, size_geom1_coords, size_coords> gap_dx{};        ///< Gap coordinate partials with fixed normals.
    MdArray<RealT, size_geom1_coords, size_geom1_coords> gap_dn{};  ///< Gap normal partials with fixed coordinates.
    MdArray<RealT, size_coords, size_coords> residual_dx{};  ///< Residual coordinate partials (independent of n).
  };
#endif

  /**
   * @brief Construct the underlying mortar weights integral from the supplied rule generator.
   * @param rule_generator Generates paired quadrature from current coordinates.
   */
  TRIBOL_HOST_DEVICE explicit FrictionalSingleMortarPairEvaluator( RuleGenerator rule_generator )
      : weights_integral_( std::move( rule_generator ) )
  {
  }

  /**
   * @brief Return the underlying mortar weights pair integral.
   * @return Configured WeightsIntegral.
   */
  TRIBOL_HOST_DEVICE const WeightsIntegral& weightsIntegral() const { return weights_integral_; }

  /**
   * @brief Integrate mortar weights and evaluate 3D kinematic gaps and the frozen gap Jacobian.
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param prev_coords1 Previous converged geometry1 coordinates.
   * @param prev_coords2 Previous converged geometry2 coordinates.
   * @param normals1 Geometry1 nodal normals.
   * @return 3D kinematic gaps, frozen gap Jacobian, and quadrature metadata.
   */
  TRIBOL_HOST_DEVICE GapEvaluation evaluateGaps( Coords1 coords1, Coords2 coords2, Coords1 prev_coords1,
                                                 Coords2 prev_coords2, Coords1 normals1 ) const
  {
    return evaluateGapsWithWeights( coords1, coords2, prev_coords1, prev_coords2, normals1,
                                    weights_integral_.evaluate( coords1, coords2 ) );
  }

  /**
   * @brief Integrate mortar weights and evaluate 3D kinematic gaps, residuals, and the frozen gap Jacobian.
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param prev_coords1 Previous converged geometry1 coordinates.
   * @param prev_coords2 Previous converged geometry2 coordinates.
   * @param normals1 Geometry1 nodal normals.
   * @param multipliers1 Geometry1 global Cartesian 3D Lagrange multipliers.
   * @return 3D kinematic gaps, residuals, frozen gap Jacobian, and quadrature metadata.
   */
  TRIBOL_HOST_DEVICE ResidualEvaluation evaluateResiduals( Coords1 coords1, Coords2 coords2, Coords1 prev_coords1,
                                                           Coords2 prev_coords2, Coords1 normals1,
                                                           Multipliers multipliers1 ) const
  {
    return evaluateResidualsWithWeights( coords1, coords2, prev_coords1, prev_coords2, normals1, multipliers1,
                                         weights_integral_.evaluate( coords1, coords2 ) );
  }

  /**
   * @brief Evaluate 3D kinematic gaps and the frozen gap Jacobian from previously integrated weights.
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param prev_coords1 Previous converged geometry1 coordinates.
   * @param prev_coords2 Previous converged geometry2 coordinates.
   * @param normals1 Geometry1 nodal normals.
   * @param weights Integrated mortar weight matrices for this pair.
   * @return 3D kinematic gaps, frozen-geometry gap Jacobian (whose transpose is df_dp), and metadata.
   */
  TRIBOL_HOST_DEVICE static GapEvaluation evaluateGapsWithWeights( Coords1 coords1, Coords2 coords2,
                                                                   Coords1 prev_coords1, Coords2 prev_coords2,
                                                                   Coords1 normals1, const Weights& weights )
  {
    GapEvaluation result{};
    result.metadata = weights.quadrature_metadata;
    if ( result.metadata.status != QuadratureStatus::Success || !weights.has_active_points ) {
      return result;
    }
    for ( IndexT multiplier_basis_index = 0; multiplier_basis_index < num_nodes1; ++multiplier_basis_index ) {
      RealT normal_dot_prev_gap = 0.0;
      for ( IndexT basis_index = 0; basis_index < num_nodes1; ++basis_index ) {
        const RealT geom1_weight = weights.geometry1( multiplier_basis_index, basis_index );
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          const IndexT row = multiplier_basis_index * spatial_dim + d;
          const IndexT column = basis_index * spatial_dim + d;
          result.frozen_gap_jacobian( row, column ) = -geom1_weight;
          result.kinematic_gap[row] -= geom1_weight * ( coords1( basis_index, d ) - prev_coords1( basis_index, d ) );
          normal_dot_prev_gap -= geom1_weight * normals1( multiplier_basis_index, d ) * prev_coords1( basis_index, d );
        }
      }
      for ( IndexT basis_index = 0; basis_index < num_nodes2; ++basis_index ) {
        const RealT geom2_weight = weights.geometry2( multiplier_basis_index, basis_index );
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          const IndexT row = multiplier_basis_index * spatial_dim + d;
          const IndexT column = size_geom1_coords + basis_index * spatial_dim + d;
          result.frozen_gap_jacobian( row, column ) = geom2_weight;
          result.kinematic_gap[row] += geom2_weight * ( coords2( basis_index, d ) - prev_coords2( basis_index, d ) );
          normal_dot_prev_gap += geom2_weight * normals1( multiplier_basis_index, d ) * prev_coords2( basis_index, d );
        }
      }
      for ( IndexT d = 0; d < spatial_dim; ++d ) {
        result.kinematic_gap[multiplier_basis_index * spatial_dim + d] +=
            normal_dot_prev_gap * normals1( multiplier_basis_index, d );
      }
    }
    return result;
  }

  /**
   * @brief Evaluate 3D kinematic gaps, residuals, and the frozen gap Jacobian from previously integrated weights.
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param prev_coords1 Previous converged geometry1 coordinates.
   * @param prev_coords2 Previous converged geometry2 coordinates.
   * @param normals1 Geometry1 nodal normals.
   * @param multipliers1 Geometry1 global Cartesian 3D Lagrange multipliers.
   * @param weights Integrated mortar weight matrices for this pair.
   * @return 3D kinematic gaps, residuals, frozen-geometry gap Jacobian, and metadata.
   */
  TRIBOL_HOST_DEVICE static ResidualEvaluation evaluateResidualsWithWeights( Coords1 coords1, Coords2 coords2,
                                                                             Coords1 prev_coords1, Coords2 prev_coords2,
                                                                             Coords1 normals1, Multipliers multipliers1,
                                                                             const Weights& weights )
  {
    ResidualEvaluation result{};
    result.gap = evaluateGapsWithWeights( coords1, coords2, prev_coords1, prev_coords2, normals1, weights );
    if ( result.gap.metadata.status != QuadratureStatus::Success || !weights.has_active_points ) {
      return result;
    }
    for ( IndexT multiplier_basis_index = 0; multiplier_basis_index < num_nodes1; ++multiplier_basis_index ) {
      for ( IndexT basis_index = 0; basis_index < num_nodes1; ++basis_index ) {
        const RealT geom1_weight = weights.geometry1( multiplier_basis_index, basis_index );
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          result.residuals[basis_index * spatial_dim + d] -= geom1_weight * multipliers1( multiplier_basis_index, d );
        }
      }
      for ( IndexT basis_index = 0; basis_index < num_nodes2; ++basis_index ) {
        const RealT geom2_weight = weights.geometry2( multiplier_basis_index, basis_index );
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          result.residuals[size_geom1_coords + basis_index * spatial_dim + d] +=
              geom2_weight * multipliers1( multiplier_basis_index, d );
        }
      }
    }
    return result;
  }

#ifdef TRIBOL_USE_ENZYME
  /**
   * @brief Differentiate mortar weights once with EnzymePairDerivative and contract into 3D gap values and partials.
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param prev_coords1 Previous converged geometry1 coordinates.
   * @param prev_coords2 Previous converged geometry2 coordinates.
   * @param normals1 Geometry1 nodal normals.
   * @return Primal 3D kinematic gaps, frozen gap Jacobian, and gap partials with respect to coordinates and normals.
   */
  TRIBOL_HOST_DEVICE GapEvaluationAndPartials evaluateAndDifferentiateGaps( Coords1 coords1, Coords2 coords2,
                                                                            Coords1 prev_coords1, Coords2 prev_coords2,
                                                                            Coords1 normals1 ) const
  {
    GapEvaluationAndPartials result{};
    const auto differentiated =
        EnzymePairDerivative<WeightsIntegral>::valuesAndJacobian( coords1, coords2, weights_integral_ );
    const auto& weights = differentiated.pair_value;
    result.evaluation = evaluateGapsWithWeights( coords1, coords2, prev_coords1, prev_coords2, normals1, weights );
    if ( result.evaluation.metadata.status != QuadratureStatus::Success || !weights.has_active_points ) {
      return result;
    }
    contractGapPartials( coords1, coords2, prev_coords1, prev_coords2, normals1, weights, differentiated.derivatives,
                         result.evaluation, result.gap_dx, result.gap_dn );
    return result;
  }

  /**
   * @brief Differentiate mortar weights once with EnzymePairDerivative and contract into gap and residual partials.
   *
   * Because the global Cartesian vector-LM residual is independent of nodal normals, residual_dx comes purely from
   * differentiating the mortar weights and residual_dn is identically zero.
   *
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param prev_coords1 Previous converged geometry1 coordinates.
   * @param prev_coords2 Previous converged geometry2 coordinates.
   * @param normals1 Geometry1 nodal normals.
   * @param multipliers1 Fixed global Cartesian 3D Lagrange multipliers.
   * @return Primal 3D kinematic gaps, residuals, frozen gap Jacobian, gap coordinate/normal partials, and residual
   * coordinate partials.
   */
  TRIBOL_HOST_DEVICE ResidualEvaluationAndPartials evaluateAndDifferentiateResiduals( Coords1 coords1, Coords2 coords2,
                                                                                      Coords1 prev_coords1,
                                                                                      Coords2 prev_coords2,
                                                                                      Coords1 normals1,
                                                                                      Multipliers multipliers1 ) const
  {
    ResidualEvaluationAndPartials result{};
    const auto differentiated =
        EnzymePairDerivative<WeightsIntegral>::valuesAndJacobian( coords1, coords2, weights_integral_ );
    const auto& weights = differentiated.pair_value;
    const auto& weight_jacobian = differentiated.derivatives;
    result.evaluation =
        evaluateResidualsWithWeights( coords1, coords2, prev_coords1, prev_coords2, normals1, multipliers1, weights );
    if ( result.evaluation.gap.metadata.status != QuadratureStatus::Success || !weights.has_active_points ) {
      return result;
    }
    contractGapPartials( coords1, coords2, prev_coords1, prev_coords2, normals1, weights, weight_jacobian,
                         result.evaluation.gap, result.gap_dx, result.gap_dn );

    // Residual coordinate partials (df / dx); independent of nodal normals because multipliers are global Cartesian.
    for ( IndexT multiplier_basis_index = 0; multiplier_basis_index < num_nodes1; ++multiplier_basis_index ) {
      for ( IndexT basis_index = 0; basis_index < num_nodes1; ++basis_index ) {
        const IndexT weight_row = Weights::geometry1Index( multiplier_basis_index, basis_index );
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          const IndexT residual_row = basis_index * spatial_dim + d;
          const RealT lambda = multipliers1( multiplier_basis_index, d );
          for ( IndexT column = 0; column < size_coords; ++column ) {
            result.residual_dx( residual_row, column ) -= lambda * weight_jacobian.jacobian( weight_row, column );
          }
        }
      }
      for ( IndexT basis_index = 0; basis_index < num_nodes2; ++basis_index ) {
        const IndexT weight_row = Weights::geometry2Index( multiplier_basis_index, basis_index );
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          const IndexT residual_row = size_geom1_coords + basis_index * spatial_dim + d;
          const RealT lambda = multipliers1( multiplier_basis_index, d );
          for ( IndexT column = 0; column < size_coords; ++column ) {
            result.residual_dx( residual_row, column ) += lambda * weight_jacobian.jacobian( weight_row, column );
          }
        }
      }
    }
    return result;
  }
#endif

 private:
#ifdef TRIBOL_USE_ENZYME
  /**
   * @brief Contract mortar weight derivatives into fixed-normal coordinate partials and normal partials of k_a.
   * @tparam WeightJacobian Derivative container returned by EnzymePairDerivative<WeightsIntegral>.
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param prev_coords1 Previous converged geometry1 coordinates.
   * @param prev_coords2 Previous converged geometry2 coordinates.
   * @param normals1 Geometry1 nodal normals.
   * @param weights Integrated mortar weights.
   * @param weight_jacobian Coordinate Jacobian of the integrated mortar weights.
   * @param gap_eval Primal gap evaluation supplying the frozen gap Jacobian.
   * @param gap_dx Output gap coordinate partials with fixed nodal normals.
   * @param gap_dn Output gap normal partials with fixed coordinates.
   */
  template <typename WeightJacobian>
  TRIBOL_HOST_DEVICE static void contractGapPartials( Coords1 coords1, Coords2 coords2, Coords1 prev_coords1,
                                                      Coords2 prev_coords2, Coords1 normals1, const Weights& weights,
                                                      const WeightJacobian& weight_jacobian,
                                                      const GapEvaluation& gap_eval,
                                                      MdArray<RealT, size_geom1_coords, size_coords>& gap_dx,
                                                      MdArray<RealT, size_geom1_coords, size_geom1_coords>& gap_dn )
  {
    for ( IndexT multiplier_basis_index = 0; multiplier_basis_index < num_nodes1; ++multiplier_basis_index ) {
      for ( IndexT d = 0; d < spatial_dim; ++d ) {
        const IndexT row = multiplier_basis_index * spatial_dim + d;
        for ( IndexT column = 0; column < size_coords; ++column ) {
          gap_dx( row, column ) = gap_eval.frozen_gap_jacobian( row, column );
        }
      }

      MdArray<RealT, spatial_dim> prev_gap{};
      for ( IndexT basis_index = 0; basis_index < num_nodes1; ++basis_index ) {
        const RealT geom1_weight = weights.geometry1( multiplier_basis_index, basis_index );
        const IndexT weight_row = Weights::geometry1Index( multiplier_basis_index, basis_index );
        RealT normal_dot_prev1 = 0.0;
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          prev_gap[d] -= geom1_weight * prev_coords1( basis_index, d );
          normal_dot_prev1 += normals1( multiplier_basis_index, d ) * prev_coords1( basis_index, d );
        }
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          const IndexT row = multiplier_basis_index * spatial_dim + d;
          const RealT coeff = ( coords1( basis_index, d ) - prev_coords1( basis_index, d ) ) +
                              normal_dot_prev1 * normals1( multiplier_basis_index, d );
          for ( IndexT column = 0; column < size_coords; ++column ) {
            gap_dx( row, column ) -= coeff * weight_jacobian.jacobian( weight_row, column );
          }
        }
      }

      for ( IndexT basis_index = 0; basis_index < num_nodes2; ++basis_index ) {
        const RealT geom2_weight = weights.geometry2( multiplier_basis_index, basis_index );
        const IndexT weight_row = Weights::geometry2Index( multiplier_basis_index, basis_index );
        RealT normal_dot_prev2 = 0.0;
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          prev_gap[d] += geom2_weight * prev_coords2( basis_index, d );
          normal_dot_prev2 += normals1( multiplier_basis_index, d ) * prev_coords2( basis_index, d );
        }
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          const IndexT row = multiplier_basis_index * spatial_dim + d;
          const RealT coeff = ( coords2( basis_index, d ) - prev_coords2( basis_index, d ) ) +
                              normal_dot_prev2 * normals1( multiplier_basis_index, d );
          for ( IndexT column = 0; column < size_coords; ++column ) {
            gap_dx( row, column ) += coeff * weight_jacobian.jacobian( weight_row, column );
          }
        }
      }

      // dk_{a,d} / dn_{a,e} = (n_a . g_a^n) delta_{de} + n_{a,d} g_{a,e}^n.
      RealT normal_dot_prev_gap = 0.0;
      for ( IndexT d = 0; d < spatial_dim; ++d ) {
        normal_dot_prev_gap += normals1( multiplier_basis_index, d ) * prev_gap[d];
      }
      for ( IndexT d = 0; d < spatial_dim; ++d ) {
        const IndexT row = multiplier_basis_index * spatial_dim + d;
        for ( IndexT e = 0; e < spatial_dim; ++e ) {
          const IndexT col = multiplier_basis_index * spatial_dim + e;
          gap_dn( row, col ) = normals1( multiplier_basis_index, d ) * prev_gap[e];
        }
        gap_dn( row, row ) += normal_dot_prev_gap;
      }
    }
  }
#endif

  WeightsIntegral weights_integral_;  ///< Underlying mortar weight pair integral.
};

}  // namespace tribol::detail

#endif  // SRC_TRIBOL_PHYSICS_INTEGRALS_FRICTIONALSINGLEMORTARPAIREVALUATOR_HPP_
