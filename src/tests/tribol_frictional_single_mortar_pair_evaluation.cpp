// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Check frictional single mortar 3D kinematic gaps, residuals, and pair derivatives on each backend. */
#include <cmath>
#include <limits>

#include "ExecutionTest.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/geom/LinearQuadGeometry.hpp"
#include "tribol/geom/LinearTriangleGeometry.hpp"
#include "tribol/geom/MortarFaceMapping.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"
#include "tribol/integ/PolygonQuadratureRuleGenerator.hpp"
#include "tribol/physics/integrals/FrictionalSingleMortarPairEvaluator.hpp"
#include "tribol/physics/integrals/SingleMortarPairEvaluator.hpp"

namespace {
using namespace tribol;

template <typename Geometry>
using MortarRule = PairQuadratureRuleGenerator<detail::MortarFaceMapping<Geometry>, PolygonQuadratureRuleGenerator<6>>;

/**
 * @brief Track the largest derivative error, including nonfinite differences.
 * @param maximum Largest error seen so far.
 * @param difference Difference between numerical and calculated derivatives.
 * @return Updated maximum, or infinity for a nonfinite difference.
 */
TRIBOL_HOST_DEVICE RealT largestDifference( RealT maximum, RealT difference )
{
  return std::isfinite( difference ) ? axom::utilities::max( maximum, std::abs( difference ) )
                                     : std::numeric_limits<RealT>::infinity();
}

/** @brief Store one pair calculation in execution-accessible memory.
 * @tparam Result Type written to the output.
 * @tparam Evaluate Callable supplying the result.
 */
template <typename Result, typename Evaluate>
struct StorePairResult {
  ArrayViewT<Result> output;  ///< View of the single output slot.
  Evaluate evaluate;          ///< Pair calculation.

  /** @brief Write the result. @param index Output slot. */
  TRIBOL_HOST_DEVICE void operator()( IndexT index ) const { output[index] = evaluate(); }
};

/** @brief Comparison between frictional and frictionless normal projections plus rigid tangential slip. */
struct DecompositionResult {
  RealT max_normal_gap_error;       ///< Max |n_a . k_a - g_frictionless_a|.
  RealT max_tangential_slip_error;  ///< Max |(I - n_a x n_a) k_a - expected_slip_a|.
  RealT max_residual_equiv_error;   ///< Max |f_frictional(p*n) - f_frictionless(p)|.
  QuadratureMetadata metadata;      ///< Rule generation status and point count.
};

/** @brief Net residual and finite-difference derivative errors for a warped quad pair. */
struct QuadDerivativeResult {
  MdArray<RealT, LinearQuadGeometry::spatial_dim> net_residual;  ///< Sum of residuals on both elements.
  RealT frozen_gap_error;                                        ///< Largest frozen kinematic-gap Jacobian error.
  RealT multiplier_error;  ///< Largest residual derivative error with respect to 3D multipliers.
  RealT coordinate_error;  ///< Largest exact derivative error with respect to current coordinates.
  RealT normal_error;      ///< Largest exact derivative error with respect to nodal normals.
  IndexT num_points;       ///< Number of generated quadrature points.
};

class FrictionalSingleMortarPairTest : public detail::ExecutionTest {
 protected:
  /**
   * @brief Run a pair calculation on the selected backend.
   * @tparam Result Values to copy back to the host.
   * @tparam Evaluate Callable whose inputs are accessible to the selected backend.
   * @param evaluate Pair calculation.
   * @return Host copy of the result.
   */
  template <typename Result, typename Evaluate>
  Result evaluateOnBackend( Evaluate evaluate )
  {
    ArrayT<Result> result( 1, 1, allocatorId() );
    forAllExec( GetParam(), 1, StorePairResult<Result, Evaluate>{ result.view(), evaluate } );
    const ArrayT<Result, 1, MemorySpace::Host> host_result( result );
    return host_result[0];
  }
};

// On a matching triangle pair with pure rigid tangential translation of geometry2 and uniform opening separation,
// n . k must match the frictionless weighted normal gap, tangential projection must equal area_node * slip, and
// setting lambda_a = p_a * n_a must reproduce the frictionless residual identically.
TEST_P( FrictionalSingleMortarPairTest, normal_and_tangential_decomposition_and_equivalence )
{
  using FrictionalEval = detail::FrictionalSingleMortarPairEvaluator<MortarRule<LinearTriangleGeometry>>;
  using FrictionlessEval = detail::SingleMortarPairEvaluator<MortarRule<LinearTriangleGeometry>>;
  using Row = MdArray<RealT, 3>;
  using Coords = MdArray<RealT, FrictionalEval::num_nodes1, FrictionalEval::spatial_dim>;

  constexpr RealT separation = 0.2;
  constexpr RealT slip_x = 0.05;
  constexpr RealT slip_y = -0.03;
  const Coords coords1{ Row{ 0.0, 0.0, 0.0 }, Row{ 1.0, 0.0, 0.0 }, Row{ 0.0, 1.0, 0.0 } };
  const Coords prev_coords1 = coords1;
  const Coords coords2{ Row{ 0.0, 0.0, separation }, Row{ 0.0, 1.0, separation }, Row{ 1.0, 0.0, separation } };
  const Coords prev_coords2{ Row{ -slip_x, -slip_y, separation }, Row{ -slip_x, 1.0 - slip_y, separation },
                             Row{ 1.0 - slip_x, -slip_y, separation } };
  const Row unit_z{ 0.0, 0.0, 1.0 };
  const Coords normals{ unit_z, unit_z, unit_z };
  const MdArray<RealT, FrictionalEval::num_nodes1> scalar_pressures{ 1.5, 2.0, 2.5 };
  const Coords vector_multipliers{ Row{ 0.0, 0.0, 1.5 }, Row{ 0.0, 0.0, 2.0 }, Row{ 0.0, 0.0, 2.5 } };

  const auto result = evaluateOnBackend<DecompositionResult>( [=] TRIBOL_HOST_DEVICE {
    const FrictionalEval frictional( MortarRule<LinearTriangleGeometry>{} );
    const FrictionlessEval frictionless( MortarRule<LinearTriangleGeometry>{} );
    const auto f_res = frictional.evaluateResiduals( coords1.view(), coords2.view(), prev_coords1.view(),
                                                     prev_coords2.view(), normals.view(), vector_multipliers.view() );
    const auto fl_res =
        frictionless.evaluateResiduals( coords1.view(), coords2.view(), normals.view(), scalar_pressures.view() );
    DecompositionResult out{};
    out.metadata = f_res.gap.metadata;
    constexpr RealT node_area = 0.5 / 3.0;
    for ( IndexT node = 0; node < FrictionalEval::num_nodes1; ++node ) {
      const RealT kx = f_res.gap.kinematic_gap[node * 3 + 0];
      const RealT ky = f_res.gap.kinematic_gap[node * 3 + 1];
      const RealT kz = f_res.gap.kinematic_gap[node * 3 + 2];
      out.max_normal_gap_error = largestDifference( out.max_normal_gap_error, kz - fl_res.gap.gaps[node] );
      out.max_tangential_slip_error = largestDifference( out.max_tangential_slip_error, kx - slip_x * node_area );
      out.max_tangential_slip_error = largestDifference( out.max_tangential_slip_error, ky - slip_y * node_area );
    }
    for ( IndexT i = 0; i < FrictionalEval::size_coords; ++i ) {
      out.max_residual_equiv_error =
          largestDifference( out.max_residual_equiv_error, f_res.residuals[i] - fl_res.residuals[i] );
    }
    return out;
  } );

  constexpr RealT tolerance = 1.0e-13;
  ASSERT_EQ( result.metadata.status, QuadratureStatus::Success );
  EXPECT_LT( result.max_normal_gap_error, tolerance );
  EXPECT_LT( result.max_tangential_slip_error, tolerance );
  EXPECT_LT( result.max_residual_equiv_error, tolerance );
}

// On a partially overlapping warped quad pair with nonzero displacement increment, distinct nodal normals, and
// arbitrary 3D multipliers, verify residual force balance and centered finite differences for all frozen and exact
// coordinate, multiplier, and normal partials.
TEST_P( FrictionalSingleMortarPairTest, residual_balance_and_derivatives )
{
  using Evaluator = detail::FrictionalSingleMortarPairEvaluator<MortarRule<LinearQuadGeometry>>;
  using Row = MdArray<RealT, Evaluator::spatial_dim>;
  using Coords = MdArray<RealT, Evaluator::num_nodes1, Evaluator::spatial_dim>;

  const Coords coords1{ Row{ 0.0, 0.0, 0.0 }, Row{ 1.4, 0.0, 0.09 }, Row{ 1.35, 1.2, -0.04 }, Row{ 0.0, 1.1, 0.08 } };
  const Coords coords2{ Row{ -0.32, 0.27, 0.05 }, Row{ -0.19, 0.98, 0.02 }, Row{ 0.93, 0.86, 0.11 },
                        Row{ 0.89, 0.22, 0.07 } };
  const Coords prev_coords1{ Row{ -0.03, 0.02, -0.01 }, Row{ 1.38, -0.01, 0.07 }, Row{ 1.31, 1.18, -0.05 },
                             Row{ -0.02, 1.07, 0.06 } };
  const Coords prev_coords2{ Row{ -0.36, 0.24, 0.06 }, Row{ -0.22, 0.94, 0.03 }, Row{ 0.88, 0.83, 0.12 },
                             Row{ 0.84, 0.19, 0.08 } };
  const Coords normals{ Row{ -0.08, 0.02, 1.0 }, Row{ 0.04, 0.1, 0.99 }, Row{ 0.09, -0.04, 0.98 },
                        Row{ 0.0, 0.04, 1.01 } };
  const Coords multipliers{ Row{ 0.2, -0.15, 0.8 }, Row{ -0.3, 0.4, 1.1 }, Row{ 0.1, 0.25, 0.6 },
                            Row{ -0.15, -0.2, 0.9 } };
  constexpr RealT step = 1.0e-6;

  const auto result = evaluateOnBackend<QuadDerivativeResult>( [=] TRIBOL_HOST_DEVICE {
    const Evaluator evaluator( MortarRule<LinearQuadGeometry>{} );
    const auto weights = evaluator.weightsIntegral().evaluate( coords1.view(), coords2.view() );
    const auto base =
        evaluator.evaluateResidualsWithWeights( coords1.view(), coords2.view(), prev_coords1.view(),
                                                prev_coords2.view(), normals.view(), multipliers.view(), weights );
    QuadDerivativeResult out{};
    out.num_points = base.gap.metadata.num_required_points;
    for ( IndexT row = 0; row < Evaluator::size_coords; ++row ) {
      out.net_residual[row % Evaluator::spatial_dim] += base.residuals[row];
    }
    for ( IndexT column = 0; column < Evaluator::size_geom1_coords; ++column ) {
      auto plus = multipliers, minus = multipliers;
      plus[column] += step;
      minus[column] -= step;
      const auto p =
          evaluator.evaluateResidualsWithWeights( coords1.view(), coords2.view(), prev_coords1.view(),
                                                  prev_coords2.view(), normals.view(), plus.view(), weights );
      const auto m =
          evaluator.evaluateResidualsWithWeights( coords1.view(), coords2.view(), prev_coords1.view(),
                                                  prev_coords2.view(), normals.view(), minus.view(), weights );
      for ( IndexT row = 0; row < Evaluator::size_coords; ++row ) {
        const RealT derivative = ( p.residuals[row] - m.residuals[row] ) / ( 2.0 * step );
        out.multiplier_error =
            largestDifference( out.multiplier_error, derivative - base.gap.frozen_gap_jacobian( column, row ) );
      }
    }
#ifdef TRIBOL_USE_ENZYME
    const auto partials = evaluator.evaluateAndDifferentiateResiduals(
        coords1.view(), coords2.view(), prev_coords1.view(), prev_coords2.view(), normals.view(), multipliers.view() );
#endif
    for ( IndexT column = 0; column < Evaluator::size_coords; ++column ) {
      auto plus1 = coords1, minus1 = coords1, plus2 = coords2, minus2 = coords2;
      if ( column < Evaluator::size_geom1_coords ) {
        plus1[column] += step;
        minus1[column] -= step;
      } else {
        plus2[column - Evaluator::size_geom1_coords] += step;
        minus2[column - Evaluator::size_geom1_coords] -= step;
      }
      const auto p = evaluator.evaluateGapsWithWeights( plus1.view(), plus2.view(), prev_coords1.view(),
                                                        prev_coords2.view(), normals.view(), weights );
      const auto m = evaluator.evaluateGapsWithWeights( minus1.view(), minus2.view(), prev_coords1.view(),
                                                        prev_coords2.view(), normals.view(), weights );
      for ( IndexT row = 0; row < Evaluator::size_geom1_coords; ++row ) {
        const RealT derivative = ( p.kinematic_gap[row] - m.kinematic_gap[row] ) / ( 2.0 * step );
        out.frozen_gap_error =
            largestDifference( out.frozen_gap_error, derivative - base.gap.frozen_gap_jacobian( row, column ) );
      }
#ifdef TRIBOL_USE_ENZYME
      const auto exact_plus = evaluator.evaluateResiduals( plus1.view(), plus2.view(), prev_coords1.view(),
                                                           prev_coords2.view(), normals.view(), multipliers.view() );
      const auto exact_minus = evaluator.evaluateResiduals( minus1.view(), minus2.view(), prev_coords1.view(),
                                                            prev_coords2.view(), normals.view(), multipliers.view() );
      for ( IndexT row = 0; row < Evaluator::size_geom1_coords; ++row ) {
        const RealT derivative =
            ( exact_plus.gap.kinematic_gap[row] - exact_minus.gap.kinematic_gap[row] ) / ( 2.0 * step );
        out.coordinate_error = largestDifference( out.coordinate_error, derivative - partials.gap_dx( row, column ) );
      }
      for ( IndexT row = 0; row < Evaluator::size_coords; ++row ) {
        const RealT derivative = ( exact_plus.residuals[row] - exact_minus.residuals[row] ) / ( 2.0 * step );
        out.coordinate_error =
            largestDifference( out.coordinate_error, derivative - partials.residual_dx( row, column ) );
      }
#endif
    }
#ifdef TRIBOL_USE_ENZYME
    for ( IndexT column = 0; column < Evaluator::size_geom1_coords; ++column ) {
      auto plus = normals, minus = normals;
      plus[column] += step;
      minus[column] -= step;
      const auto p = evaluator.evaluateResiduals( coords1.view(), coords2.view(), prev_coords1.view(),
                                                  prev_coords2.view(), plus.view(), multipliers.view() );
      const auto m = evaluator.evaluateResiduals( coords1.view(), coords2.view(), prev_coords1.view(),
                                                  prev_coords2.view(), minus.view(), multipliers.view() );
      for ( IndexT row = 0; row < Evaluator::size_geom1_coords; ++row ) {
        const RealT derivative = ( p.gap.kinematic_gap[row] - m.gap.kinematic_gap[row] ) / ( 2.0 * step );
        out.normal_error = largestDifference( out.normal_error, derivative - partials.gap_dn( row, column ) );
      }
    }
#endif
    return out;
  } );

  ASSERT_GT( result.num_points, 0 );
  constexpr RealT balance_tolerance = 1.0e-13;
  constexpr RealT derivative_tolerance = 1.0e-8;
  for ( IndexT d = 0; d < Evaluator::spatial_dim; ++d ) {
    EXPECT_NEAR( result.net_residual[d], 0.0, balance_tolerance );
  }
  EXPECT_LT( result.frozen_gap_error, derivative_tolerance );
  EXPECT_LT( result.multiplier_error, derivative_tolerance );
#ifdef TRIBOL_USE_ENZYME
  EXPECT_LT( result.coordinate_error, derivative_tolerance );
  EXPECT_LT( result.normal_error, derivative_tolerance );
#endif
}

INSTANTIATE_TEST_SUITE_P( backends, FrictionalSingleMortarPairTest, detail::enabledExecutionModes() );
}  // namespace

int main( int argc, char** argv )
{
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
