// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Contract integrated mortar weights into single mortar weighted gaps, residuals, and coordinate/normal
 * partials.
 */
#ifndef SRC_TRIBOL_PHYSICS_INTEGRALS_SINGLEMORTARPAIREVALUATOR_HPP_
#define SRC_TRIBOL_PHYSICS_INTEGRALS_SINGLEMORTARPAIREVALUATOR_HPP_

#include "tribol/assembly/EnzymePairDerivative.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/QuadratureCommon.hpp"
#include "tribol/physics/integrals/PairIntegrals.hpp"

namespace tribol::detail {

/**
 * @brief Evaluate single mortar gaps, residuals, and partial derivatives from integrated mortar weights.
 *
 * All quadrature integration is delegated to PairIntegral with MortarWeightsIntegrand, and all coordinate
 * differentiation of the mortar weights is delegated to EnzymePairDerivative. Derivatives with respect to nodal
 * normals and nodal pressures are computed algebraically from the integrated weights.
 *
 * @tparam RuleGenerator Configured generator for the full projected overlap.
 */
template <typename RuleGenerator>
class SingleMortarPairEvaluator {
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
  static constexpr IndexT size_geom1_coords = num_nodes1 * spatial_dim;         ///< Geometry1 coordinate extent.
  static constexpr IndexT size_geom2_coords = num_nodes2 * spatial_dim;         ///< Geometry2 coordinate extent.
  static constexpr IndexT size_coords = size_geom1_coords + size_geom2_coords;  ///< Total pair coordinate extent.
  using Coords1 = typename Geometry1::Coords;                                   ///< Geometry1 coordinates view.
  using Coords2 = typename Geometry2::Coords;                                   ///< Geometry2 coordinates view.
  using Pressures = MdSpan<const RealT, num_nodes1>;                            ///< Geometry1 nodal pressures.
  using FrozenGapJacobian = MdArray<RealT, num_nodes1, size_coords>;            ///< Gap rows and coordinate columns.

  /** @brief Pair weighted gaps and frozen-geometry gap Jacobian. */
  struct GapEvaluation {
    MdArray<RealT, num_nodes1> gaps{};        ///< Weighted gaps at geometry1 nodes.
    FrozenGapJacobian frozen_gap_jacobian{};  ///< Gap coordinate derivative holding weights and normals fixed.
    QuadratureMetadata metadata{};            ///< Status and point count. Invalid and empty rules give zero entries.
  };

  /** @brief Pair weighted gaps, frozen-geometry gap Jacobian, and nodal residuals. */
  struct ResidualEvaluation {
    GapEvaluation gap{};                      ///< Weighted gaps, frozen gap Jacobian, and quadrature metadata.
    MdArray<RealT, size_coords> residuals{};  ///< Nodal residuals ordered as geometry1 followed by geometry2.
  };

#ifdef TRIBOL_USE_ENZYME
  /** @brief Weighted gaps together with coordinate and normal partial derivatives from one differentiated call. */
  struct GapEvaluationAndPartials {
    GapEvaluation evaluation{};                              ///< Primal weighted gaps and frozen gap Jacobian.
    MdArray<RealT, num_nodes1, size_coords> gap_dx{};        ///< Gap coordinate partials with fixed nodal normals.
    MdArray<RealT, num_nodes1, size_geom1_coords> gap_dn{};  ///< Gap normal partials with fixed coordinates.
  };

  /** @brief Gaps, residuals, and their coordinate and normal partials from one differentiated call. */
  struct ResidualEvaluationAndPartials {
    ResidualEvaluation evaluation{};                               ///< Primal gaps, residuals, and frozen Jacobian.
    MdArray<RealT, num_nodes1, size_coords> gap_dx{};              ///< Gap coordinate partials with fixed normals.
    MdArray<RealT, num_nodes1, size_geom1_coords> gap_dn{};        ///< Gap normal partials with fixed coordinates.
    MdArray<RealT, size_coords, size_coords> residual_dx{};        ///< Residual coordinate partials (fixed normals).
    MdArray<RealT, size_coords, size_geom1_coords> residual_dn{};  ///< Residual normal partials (fixed coordinates).
  };
#endif

  /**
   * @brief Construct the underlying mortar weights integral from the supplied rule generator.
   * @param rule_generator Generates paired quadrature from current coordinates.
   */
  TRIBOL_HOST_DEVICE explicit SingleMortarPairEvaluator( RuleGenerator rule_generator )
      : weights_integral_( std::move( rule_generator ) )
  {
  }

  /**
   * @brief Return the underlying mortar weights pair integral.
   * @return Configured WeightsIntegral.
   */
  TRIBOL_HOST_DEVICE const WeightsIntegral& weightsIntegral() const { return weights_integral_; }

  /**
   * @brief Integrate mortar weights and evaluate weighted gaps and the frozen gap Jacobian.
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param normals1 Geometry1 nodal normals.
   * @return Weighted gaps, frozen gap Jacobian, and quadrature metadata.
   */
  TRIBOL_HOST_DEVICE GapEvaluation evaluateGaps( Coords1 coords1, Coords2 coords2, Coords1 normals1 ) const
  {
    return evaluateGapsWithWeights( coords1, coords2, normals1, weights_integral_.evaluate( coords1, coords2 ) );
  }

  /**
   * @brief Integrate mortar weights and evaluate weighted gaps, residuals, and the frozen gap Jacobian.
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param normals1 Geometry1 nodal normals.
   * @param pressures1 Geometry1 nodal pressures.
   * @return Weighted gaps, residuals, frozen gap Jacobian, and quadrature metadata.
   */
  TRIBOL_HOST_DEVICE ResidualEvaluation evaluateResiduals( Coords1 coords1, Coords2 coords2, Coords1 normals1,
                                                           Pressures pressures1 ) const
  {
    return evaluateResidualsWithWeights( coords1, coords2, normals1, pressures1,
                                         weights_integral_.evaluate( coords1, coords2 ) );
  }

  /**
   * @brief Evaluate weighted gaps and the frozen gap Jacobian from previously integrated weights.
   * @param coords1 Geometry1 coordinates used in the gap calculation.
   * @param coords2 Geometry2 coordinates used in the gap calculation.
   * @param normals1 Geometry1 nodal normals.
   * @param weights Integrated mortar weight matrices for this pair.
   * @return Weighted gaps, frozen-geometry gap Jacobian (whose transpose is df_dp), and metadata.
   */
  TRIBOL_HOST_DEVICE static GapEvaluation evaluateGapsWithWeights( Coords1 coords1, Coords2 coords2, Coords1 normals1,
                                                                   const Weights& weights )
  {
    GapEvaluation result{};
    result.metadata = weights.quadrature_metadata;
    if ( result.metadata.status != QuadratureStatus::Success || !weights.has_active_points ) {
      return result;
    }
    for ( IndexT multiplier_basis_index = 0; multiplier_basis_index < num_nodes1; ++multiplier_basis_index ) {
      for ( IndexT basis_index = 0; basis_index < num_nodes1; ++basis_index ) {
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          const RealT entry =
              -weights.geometry1( multiplier_basis_index, basis_index ) * normals1( multiplier_basis_index, d );
          const IndexT column = basis_index * spatial_dim + d;
          result.frozen_gap_jacobian( multiplier_basis_index, column ) = entry;
          result.gaps[multiplier_basis_index] += entry * coords1( basis_index, d );
        }
      }
      for ( IndexT basis_index = 0; basis_index < num_nodes2; ++basis_index ) {
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          const RealT entry =
              weights.geometry2( multiplier_basis_index, basis_index ) * normals1( multiplier_basis_index, d );
          const IndexT column = size_geom1_coords + basis_index * spatial_dim + d;
          result.frozen_gap_jacobian( multiplier_basis_index, column ) = entry;
          result.gaps[multiplier_basis_index] += entry * coords2( basis_index, d );
        }
      }
    }
    return result;
  }

  /**
   * @brief Evaluate weighted gaps, residuals, and the frozen gap Jacobian from previously integrated weights.
   * @param coords1 Geometry1 coordinates used in the gap calculation.
   * @param coords2 Geometry2 coordinates used in the gap calculation.
   * @param normals1 Geometry1 nodal normals.
   * @param pressures1 Geometry1 nodal pressures.
   * @param weights Integrated mortar weight matrices for this pair.
   * @return Weighted gaps, residuals, frozen-geometry gap Jacobian, and metadata.
   */
  TRIBOL_HOST_DEVICE static ResidualEvaluation evaluateResidualsWithWeights( Coords1 coords1, Coords2 coords2,
                                                                             Coords1 normals1, Pressures pressures1,
                                                                             const Weights& weights )
  {
    ResidualEvaluation result{};
    result.gap = evaluateGapsWithWeights( coords1, coords2, normals1, weights );
    if ( result.gap.metadata.status != QuadratureStatus::Success || !weights.has_active_points ) {
      return result;
    }
    for ( IndexT multiplier_basis_index = 0; multiplier_basis_index < num_nodes1; ++multiplier_basis_index ) {
      const RealT pressure = pressures1[multiplier_basis_index];
      for ( IndexT column = 0; column < size_coords; ++column ) {
        result.residuals[column] += result.gap.frozen_gap_jacobian( multiplier_basis_index, column ) * pressure;
      }
    }
    return result;
  }

#ifdef TRIBOL_USE_ENZYME
  /**
   * @brief Differentiate mortar weights once with EnzymePairDerivative and contract into gap values and partials.
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param normals1 Geometry1 nodal normals.
   * @return Primal weighted gaps, frozen gap Jacobian, and gap partials with respect to coordinates and normals.
   */
  TRIBOL_HOST_DEVICE GapEvaluationAndPartials evaluateAndDifferentiateGaps( Coords1 coords1, Coords2 coords2,
                                                                            Coords1 normals1 ) const
  {
    GapEvaluationAndPartials result{};
    const auto differentiated =
        EnzymePairDerivative<WeightsIntegral>::valuesAndJacobian( coords1, coords2, weights_integral_ );
    const auto& weights = differentiated.pair_value;
    result.evaluation = evaluateGapsWithWeights( coords1, coords2, normals1, weights );
    if ( result.evaluation.metadata.status != QuadratureStatus::Success || !weights.has_active_points ) {
      return result;
    }
    contractGapPartials( coords1, coords2, normals1, weights, differentiated.derivatives, result.evaluation,
                         result.gap_dx, result.gap_dn );
    return result;
  }

  /**
   * @brief Differentiate mortar weights once with EnzymePairDerivative and contract into gap and residual partials.
   *
   * At fixed normals, dg_dx equals frozen_gap_jacobian plus the weight-derivative contraction, and df_dx comes purely
   * from the weight-derivative contraction. Derivatives with respect to normals1 are computed algebraically from the
   * integrated mortar weights.
   *
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param normals1 Geometry1 nodal normals.
   * @param pressures1 Fixed nodal pressure values.
   * @return Primal gaps, residuals, frozen gap Jacobian, and coordinate and normal partials for both gaps and
   * residuals.
   */
  TRIBOL_HOST_DEVICE ResidualEvaluationAndPartials evaluateAndDifferentiateResiduals( Coords1 coords1, Coords2 coords2,
                                                                                      Coords1 normals1,
                                                                                      Pressures pressures1 ) const
  {
    ResidualEvaluationAndPartials result{};
    const auto differentiated =
        EnzymePairDerivative<WeightsIntegral>::valuesAndJacobian( coords1, coords2, weights_integral_ );
    const auto& weights = differentiated.pair_value;
    const auto& weight_jacobian = differentiated.derivatives;
    result.evaluation = evaluateResidualsWithWeights( coords1, coords2, normals1, pressures1, weights );
    if ( result.evaluation.gap.metadata.status != QuadratureStatus::Success || !weights.has_active_points ) {
      return result;
    }
    contractGapPartials( coords1, coords2, normals1, weights, weight_jacobian, result.evaluation.gap, result.gap_dx,
                         result.gap_dn );

    // Residual partials with respect to normals1 (df / dn1) and fixed-normal coordinate partials (df / dx).
    for ( IndexT multiplier_basis_index = 0; multiplier_basis_index < num_nodes1; ++multiplier_basis_index ) {
      const RealT pressure = pressures1[multiplier_basis_index];
      for ( IndexT basis_index = 0; basis_index < num_nodes1; ++basis_index ) {
        const RealT geom1_weight = weights.geometry1( multiplier_basis_index, basis_index );
        const IndexT weight_row = Weights::geometry1Index( multiplier_basis_index, basis_index );
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          const IndexT residual_row = basis_index * spatial_dim + d;
          const IndexT normal_column = multiplier_basis_index * spatial_dim + d;
          result.residual_dn( residual_row, normal_column ) -= pressure * geom1_weight;
          const RealT pressure_normal = pressure * normals1( multiplier_basis_index, d );
          for ( IndexT column = 0; column < size_coords; ++column ) {
            result.residual_dx( residual_row, column ) -=
                pressure_normal * weight_jacobian.jacobian( weight_row, column );
          }
        }
      }
      for ( IndexT basis_index = 0; basis_index < num_nodes2; ++basis_index ) {
        const RealT geom2_weight = weights.geometry2( multiplier_basis_index, basis_index );
        const IndexT weight_row = Weights::geometry2Index( multiplier_basis_index, basis_index );
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          const IndexT residual_row = size_geom1_coords + basis_index * spatial_dim + d;
          const IndexT normal_column = multiplier_basis_index * spatial_dim + d;
          result.residual_dn( residual_row, normal_column ) += pressure * geom2_weight;
          const RealT pressure_normal = pressure * normals1( multiplier_basis_index, d );
          for ( IndexT column = 0; column < size_coords; ++column ) {
            result.residual_dx( residual_row, column ) +=
                pressure_normal * weight_jacobian.jacobian( weight_row, column );
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
   * @brief Contract mortar weight derivatives into fixed-normal coordinate partials and normal partials of the gap.
   * @tparam WeightJacobian Derivative container returned by EnzymePairDerivative<WeightsIntegral>.
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param normals1 Geometry1 nodal normals.
   * @param weights Integrated mortar weights.
   * @param weight_jacobian Coordinate Jacobian of the integrated mortar weights.
   * @param gap_eval Primal gap evaluation supplying the frozen gap Jacobian.
   * @param gap_dx Output gap coordinate partials with fixed nodal normals.
   * @param gap_dn Output gap normal partials with fixed coordinates.
   */
  template <typename WeightJacobian>
  TRIBOL_HOST_DEVICE static void contractGapPartials( Coords1 coords1, Coords2 coords2, Coords1 normals1,
                                                      const Weights& weights, const WeightJacobian& weight_jacobian,
                                                      const GapEvaluation& gap_eval,
                                                      MdArray<RealT, num_nodes1, size_coords>& gap_dx,
                                                      MdArray<RealT, num_nodes1, size_geom1_coords>& gap_dn )
  {
    for ( IndexT multiplier_basis_index = 0; multiplier_basis_index < num_nodes1; ++multiplier_basis_index ) {
      for ( IndexT column = 0; column < size_coords; ++column ) {
        gap_dx( multiplier_basis_index, column ) = gap_eval.frozen_gap_jacobian( multiplier_basis_index, column );
      }
      for ( IndexT basis_index = 0; basis_index < num_nodes1; ++basis_index ) {
        const RealT geom1_weight = weights.geometry1( multiplier_basis_index, basis_index );
        const IndexT weight_row = Weights::geometry1Index( multiplier_basis_index, basis_index );
        RealT normal_dot_coord1 = 0.0;
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          gap_dn( multiplier_basis_index, multiplier_basis_index * spatial_dim + d ) -=
              geom1_weight * coords1( basis_index, d );
          normal_dot_coord1 += normals1( multiplier_basis_index, d ) * coords1( basis_index, d );
        }
        for ( IndexT column = 0; column < size_coords; ++column ) {
          gap_dx( multiplier_basis_index, column ) -=
              normal_dot_coord1 * weight_jacobian.jacobian( weight_row, column );
        }
      }
      for ( IndexT basis_index = 0; basis_index < num_nodes2; ++basis_index ) {
        const RealT geom2_weight = weights.geometry2( multiplier_basis_index, basis_index );
        const IndexT weight_row = Weights::geometry2Index( multiplier_basis_index, basis_index );
        RealT normal_dot_coord2 = 0.0;
        for ( IndexT d = 0; d < spatial_dim; ++d ) {
          gap_dn( multiplier_basis_index, multiplier_basis_index * spatial_dim + d ) +=
              geom2_weight * coords2( basis_index, d );
          normal_dot_coord2 += normals1( multiplier_basis_index, d ) * coords2( basis_index, d );
        }
        for ( IndexT column = 0; column < size_coords; ++column ) {
          gap_dx( multiplier_basis_index, column ) +=
              normal_dot_coord2 * weight_jacobian.jacobian( weight_row, column );
        }
      }
    }
  }
#endif

  WeightsIntegral weights_integral_;  ///< Underlying mortar weight pair integral.
};

}  // namespace tribol::detail

#endif  // SRC_TRIBOL_PHYSICS_INTEGRALS_SINGLEMORTARPAIREVALUATOR_HPP_
