// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Check single mortar quadrature, nodal residuals, and pair derivatives on each enabled backend. */
#include <limits>

#include "ExecutionTest.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/geom/EdgeAveragedNodalNormalKernel.hpp"
#include "tribol/geom/LinearQuadGeometry.hpp"
#include "tribol/geom/LinearTriangleGeometry.hpp"
#include "tribol/geom/MortarFaceMapping.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"
#include "tribol/integ/PolygonQuadratureRuleGenerator.hpp"
#include "tribol/physics/integrals/SingleMortarPairEvaluator.hpp"

namespace {
using namespace tribol;

template <typename Geometry>
using MortarRule = PairQuadratureRuleGenerator<detail::MortarFaceMapping<Geometry>, PolygonQuadratureRuleGenerator<6>>;

/**
 * @brief Track the largest derivative error, including nonfinite differences.
 * @param maximum Largest error seen so far.
 * @param difference Difference between the numerical and calculated derivatives.
 * @return Updated maximum, or infinity for a nonfinite difference.
 */
TRIBOL_HOST_DEVICE RealT largestDifference( RealT maximum, RealT difference )
{
  return std::isfinite( difference ) ? axom::utilities::max( maximum, std::abs( difference ) )
                                     : std::numeric_limits<RealT>::infinity();
}

/** @brief Store one pair calculation in execution-accessible memory.
 * @tparam Result Type written to the output.
 * @tparam Evaluate Callable that supplies a result.
 */
template <typename Result, typename Evaluate>
struct StorePairResult {
  ArrayViewT<Result> output;  ///< View of the single output slot.
  Evaluate evaluate;          ///< Pair calculation.

  /** @brief Write the result. @param index Output slot. */
  TRIBOL_HOST_DEVICE void operator()( IndexT index ) const { output[index] = evaluate(); }
};

/** @brief Integrated triangle area and nodal values for a pair with a uniform gap. */
struct TrianglePairResult {
  detail::SingleMortarPairEvaluator<MortarRule<LinearTriangleGeometry>>::ResidualEvaluation values;  ///< Pair values.
  RealT area;  ///< Integrated area.
};

/** @brief Projected area, coordinate error, and statuses for the tested configurations. */
struct QuadProjectionResult {
  RealT area;                         ///< Physical integration weights summed over the overlap.
  RealT max_projection_error;         ///< Largest in-plane position error after mapping to the elements.
  QuadratureMetadata full;            ///< Warped element with full projected overlap.
  QuadratureMetadata disjoint;        ///< Disjoint projections.
  QuadratureMetadata touching;        ///< Projections sharing an edge.
  QuadratureMetadata same_direction;  ///< Elements with parallel normals pointing in the same direction.
  QuadratureMetadata invalid;         ///< Collapsed mortar element.
};

/** @brief Net residual and finite-difference errors for a warped pair. */
struct QuadDerivativeResult {
  MdArray<RealT, LinearQuadGeometry::spatial_dim> net_residual;  ///< Sum of the residuals on both elements.
  RealT frozen_gap_error;                                        ///< Largest frozen weighted-gap derivative error.
  RealT pressure_error;    ///< Largest residual derivative error with respect to pressure.
  RealT weight_error;      ///< Largest mortar weight Jacobian error with respect to coordinates.
  RealT coordinate_error;  ///< Largest exact derivative error with respect to coordinates.
  RealT normal_error;      ///< Largest exact derivative error with respect to nodal normals.
  IndexT num_points;       ///< Number of generated quadrature points.
};

class SingleMortarPairTest : public detail::ExecutionTest {
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

// Matching triangle projections integrate a uniform positive opening gap and give opposite residuals on the two sides.
TEST_P( SingleMortarPairTest, triangle_area_and_signed_values )
{
  using Evaluator = detail::SingleMortarPairEvaluator<MortarRule<LinearTriangleGeometry>>;
  using Row = MdArray<RealT, 3>;
  using Coords = MdArray<RealT, Evaluator::num_nodes1, Evaluator::spatial_dim>;
  const Coords coords1{ Row{ 0.0, 0.0, 0.0 }, Row{ 1.0, 0.0, 0.0 }, Row{ 0.0, 1.0, 0.0 } };
  constexpr RealT separation = 0.2;
  const Coords coords2{ Row{ 0.0, 0.0, separation }, Row{ 0.0, 1.0, separation }, Row{ 1.0, 0.0, separation } };
  const Row normal{ 0.0, 0.0, 1.0 };
  const Coords normals{ normal, normal, normal };
  constexpr RealT pressure = 2.0;
  const MdArray<RealT, Evaluator::num_nodes1> pressures{ pressure, pressure, pressure };
  const auto result = evaluateOnBackend<TrianglePairResult>( [=] TRIBOL_HOST_DEVICE {
    const Evaluator evaluator( MortarRule<LinearTriangleGeometry>{} );
    const auto weights = evaluator.weightsIntegral().evaluate( coords1.view(), coords2.view() );
    RealT area = 0.0;
    for ( IndexT row = 0; row < Evaluator::num_nodes1; ++row ) {
      for ( IndexT column = 0; column < Evaluator::num_nodes1; ++column ) {
        area += weights.geometry1( row, column );
      }
    }
    return TrianglePairResult{ evaluator.evaluateResidualsWithWeights( coords1.view(), coords2.view(), normals.view(),
                                                                       pressures.view(), weights ),
                               area };
  } );
  constexpr RealT expected_area = 0.5;
  constexpr RealT tolerance = 1.0e-13;
  ASSERT_EQ( result.values.gap.metadata.status, QuadratureStatus::Success );
  EXPECT_NEAR( result.area, expected_area, tolerance );
  for ( IndexT node = 0; node < Evaluator::num_nodes1; ++node ) {
    const RealT node_area = expected_area / Evaluator::num_nodes1;
    const IndexT z_residual = node * Evaluator::spatial_dim + 2;
    EXPECT_NEAR( result.values.gap.gaps[node], separation * node_area, tolerance );
    EXPECT_NEAR( result.values.residuals[z_residual], -pressure * node_area, tolerance );
    EXPECT_NEAR( result.values.residuals[z_residual + Evaluator::size_geom1_coords], pressure * node_area, tolerance );
  }
}

// The warped mortar element crosses the plane. Every projected quadrature point must be kept, including opening points.
TEST_P( SingleMortarPairTest, full_overlap_and_projection_status )
{
  using Geometry = LinearQuadGeometry;
  using Generator = MortarRule<Geometry>;
  using Row = MdArray<RealT, Geometry::spatial_dim>;
  using Coords = MdArray<RealT, Geometry::num_nodes, Geometry::spatial_dim>;
  const Coords coords1{ Row{ 0.0, 0.0, 0.0 }, Row{ 1.0, 0.0, 0.0 }, Row{ 1.0, 1.0, 0.0 }, Row{ 0.0, 1.0, 0.0 } };
  const Coords coords2{ Row{ 0.0, 0.0, 0.2 }, Row{ 0.0, 1.0, -0.2 }, Row{ 1.0, 1.0, 0.1 }, Row{ 1.0, 0.0, -0.3 } };
  const auto result = evaluateOnBackend<QuadProjectionResult>( [=] TRIBOL_HOST_DEVICE {
    const Generator generator{};
    const Geometry geometry1( coords1.view() );
    const Geometry geometry2( coords2.view() );
    const auto rule = generator.generate( geometry1, geometry2 );
    QuadProjectionResult result{};
    result.full = rule.metadata;
    for ( IndexT point = 0; point < rule.metadata.num_required_points; ++point ) {
      const auto a = geometry1.position( MdSpan<const RealT, Geometry::parametric_dim>(
          &rule.geom1_parametric_coords( point, 0 ), Geometry::parametric_dim ) );
      const auto b = geometry2.position( MdSpan<const RealT, Geometry::parametric_dim>(
          &rule.geom2_parametric_coords( point, 0 ), Geometry::parametric_dim ) );
      result.area += rule.weights[point];
      result.max_projection_error =
          axom::utilities::max( result.max_projection_error, std::abs( a[0] - b[0] ) + std::abs( a[1] - b[1] ) );
    }
    auto shifted = coords2;
    for ( IndexT node = 0; node < Geometry::num_nodes; ++node ) {
      shifted( node, 0 ) += 1.0;
    }
    result.touching = generator.generate( geometry1, Geometry( shifted.view() ) ).metadata;
    for ( IndexT node = 0; node < Geometry::num_nodes; ++node ) {
      shifted( node, 0 ) += 1.0;
    }
    result.disjoint = generator.generate( geometry1, Geometry( shifted.view() ) ).metadata;
    result.same_direction = generator.generate( geometry1, geometry1 ).metadata;
    const Coords collapsed{};
    result.invalid = generator.generate( geometry1, Geometry( collapsed.view() ) ).metadata;
    return result;
  } );
  constexpr RealT tolerance = 1.0e-12;
  ASSERT_EQ( result.full.status, QuadratureStatus::Success );
  EXPECT_NEAR( result.area, 1.0, tolerance );
  EXPECT_LT( result.max_projection_error, tolerance );
  for ( const auto status : { result.disjoint, result.touching, result.same_direction } ) {
    EXPECT_EQ( status.status, QuadratureStatus::Success );
    EXPECT_EQ( status.num_required_points, 0 );
  }
  EXPECT_EQ( result.invalid.status, QuadratureStatus::InvalidGeometry );
}

// A partially overlapping warped pair has different nodal normals, so using the wrong normal on either side changes
// the net residual. The perturbations keep the intersections on the same edges.
TEST_P( SingleMortarPairTest, residual_balance_and_frozen_derivatives )
{
  using Evaluator = detail::SingleMortarPairEvaluator<MortarRule<LinearQuadGeometry>>;
  using Row = MdArray<RealT, Evaluator::spatial_dim>;
  using Coords = MdArray<RealT, Evaluator::num_nodes1, Evaluator::spatial_dim>;
  const Coords coords1{ Row{ 0.0, 0.0, 0.0 }, Row{ 1.4, 0.0, 0.09 }, Row{ 1.35, 1.2, -0.04 }, Row{ 0.0, 1.1, 0.08 } };
  const Coords coords2{ Row{ -0.32, 0.27, 0.05 }, Row{ -0.19, 0.98, 0.02 }, Row{ 0.93, 0.86, 0.11 },
                        Row{ 0.89, 0.22, 0.07 } };
  const Coords normals{ Row{ -0.08, 0.02, 1.0 }, Row{ 0.04, 0.1, 0.99 }, Row{ 0.09, -0.04, 0.98 },
                        Row{ 0.0, 0.04, 1.01 } };
  const MdArray<RealT, Evaluator::num_nodes1> pressures{ 0.3, 1.0, 0.6, 0.9 };
  constexpr RealT step = 1.0e-6;
  const auto result = evaluateOnBackend<QuadDerivativeResult>( [=] TRIBOL_HOST_DEVICE {
    const Evaluator evaluator( MortarRule<LinearQuadGeometry>{} );
    const auto weights = evaluator.weightsIntegral().evaluate( coords1.view(), coords2.view() );
    const auto base = evaluator.evaluateResidualsWithWeights( coords1.view(), coords2.view(), normals.view(),
                                                              pressures.view(), weights );
    QuadDerivativeResult result{};
    result.num_points = base.gap.metadata.num_required_points;
    for ( IndexT row = 0; row < Evaluator::size_coords; ++row ) {
      result.net_residual[row % Evaluator::spatial_dim] += base.residuals[row];
    }
    for ( IndexT column = 0; column < Evaluator::num_nodes1; ++column ) {
      auto plus = pressures, minus = pressures;
      plus[column] += step;
      minus[column] -= step;
      const auto p = evaluator.evaluateResidualsWithWeights( coords1.view(), coords2.view(), normals.view(),
                                                             plus.view(), weights );
      const auto m = evaluator.evaluateResidualsWithWeights( coords1.view(), coords2.view(), normals.view(),
                                                             minus.view(), weights );
      for ( IndexT row = 0; row < Evaluator::size_coords; ++row ) {
        const RealT derivative = ( p.residuals[row] - m.residuals[row] ) / ( 2.0 * step );
        result.pressure_error =
            largestDifference( result.pressure_error, derivative - base.gap.frozen_gap_jacobian( column, row ) );
      }
    }
#ifdef TRIBOL_USE_ENZYME
    const auto weight_diff = EnzymePairDerivative<typename Evaluator::WeightsIntegral>::valuesAndJacobian(
        coords1.view(), coords2.view(), evaluator.weightsIntegral() );
    const auto partials =
        evaluator.evaluateAndDifferentiateResiduals( coords1.view(), coords2.view(), normals.view(), pressures.view() );
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
      const auto p = evaluator.evaluateGapsWithWeights( plus1.view(), plus2.view(), normals.view(), weights );
      const auto m = evaluator.evaluateGapsWithWeights( minus1.view(), minus2.view(), normals.view(), weights );
      for ( IndexT row = 0; row < Evaluator::num_nodes1; ++row ) {
        const RealT derivative = ( p.gaps[row] - m.gaps[row] ) / ( 2.0 * step );
        result.frozen_gap_error =
            largestDifference( result.frozen_gap_error, derivative - base.gap.frozen_gap_jacobian( row, column ) );
      }
#ifdef TRIBOL_USE_ENZYME
      const auto weights_plus = evaluator.weightsIntegral().evaluate( plus1.view(), plus2.view() );
      const auto weights_minus = evaluator.weightsIntegral().evaluate( minus1.view(), minus2.view() );
      const auto w_plus_vals = weights_plus.values();
      const auto w_minus_vals = weights_minus.values();
      for ( IndexT row = 0; row < Evaluator::Weights::size_values; ++row ) {
        const RealT weight_derivative = ( w_plus_vals[row] - w_minus_vals[row] ) / ( 2.0 * step );
        result.weight_error = largestDifference( result.weight_error,
                                                 weight_derivative - weight_diff.derivatives.jacobian( row, column ) );
      }
      const auto exact_plus =
          evaluator.evaluateResiduals( plus1.view(), plus2.view(), normals.view(), pressures.view() );
      const auto exact_minus =
          evaluator.evaluateResiduals( minus1.view(), minus2.view(), normals.view(), pressures.view() );
      for ( IndexT row = 0; row < Evaluator::num_nodes1; ++row ) {
        const RealT derivative = ( exact_plus.gap.gaps[row] - exact_minus.gap.gaps[row] ) / ( 2.0 * step );
        result.coordinate_error =
            largestDifference( result.coordinate_error, derivative - partials.gap_dx( row, column ) );
      }
      for ( IndexT row = 0; row < Evaluator::size_coords; ++row ) {
        const RealT derivative = ( exact_plus.residuals[row] - exact_minus.residuals[row] ) / ( 2.0 * step );
        result.coordinate_error =
            largestDifference( result.coordinate_error, derivative - partials.residual_dx( row, column ) );
      }
#endif
    }
#ifdef TRIBOL_USE_ENZYME
    for ( IndexT column = 0; column < Evaluator::size_geom1_coords; ++column ) {
      auto plus = normals, minus = normals;
      plus[column] += step;
      minus[column] -= step;
      const auto p = evaluator.evaluateResiduals( coords1.view(), coords2.view(), plus.view(), pressures.view() );
      const auto m = evaluator.evaluateResiduals( coords1.view(), coords2.view(), minus.view(), pressures.view() );
      for ( IndexT row = 0; row < Evaluator::num_nodes1; ++row ) {
        const RealT derivative = ( p.gap.gaps[row] - m.gap.gaps[row] ) / ( 2.0 * step );
        result.normal_error = largestDifference( result.normal_error, derivative - partials.gap_dn( row, column ) );
      }
      for ( IndexT row = 0; row < Evaluator::size_coords; ++row ) {
        const RealT derivative = ( p.residuals[row] - m.residuals[row] ) / ( 2.0 * step );
        result.normal_error =
            largestDifference( result.normal_error, derivative - partials.residual_dn( row, column ) );
      }
    }
#endif
    return result;
  } );

  ASSERT_GT( result.num_points, 0 );
  constexpr RealT balance_tolerance = 1.0e-13;
  constexpr RealT derivative_tolerance = 1.0e-8;
  for ( IndexT d = 0; d < Evaluator::spatial_dim; ++d ) {
    EXPECT_NEAR( result.net_residual[d], 0.0, balance_tolerance );
  }
  EXPECT_LT( result.frozen_gap_error, derivative_tolerance );
  EXPECT_LT( result.pressure_error, derivative_tolerance );
#ifdef TRIBOL_USE_ENZYME
  EXPECT_LT( result.weight_error, derivative_tolerance );
  EXPECT_LT( result.coordinate_error, derivative_tolerance );
  EXPECT_LT( result.normal_error, derivative_tolerance );
#endif
}

// A collapsed reference element has no usable corners. Its normal calculation must return finite zeros.
TEST_P( SingleMortarPairTest, degenerate_reference_corners )
{
  using Kernel = detail::EdgeAveragedNodalNormalKernel<LinearTriangleGeometry::num_nodes>;
  using Row = MdArray<RealT, Kernel::spatial_dim>;
  const Kernel::NodalVectors reference{ Row{ 0.0, 0.0, 0.0 }, Row{ 1.0, 0.0, 0.0 }, Row{ 2.0, 0.0, 0.0 } };
  const Kernel::NodalVectors current{ Row{ 0.0, 0.0, 0.0 }, Row{ 1.0, 0.0, 0.0 }, Row{ 0.0, 1.0, 0.0 } };
  const auto normals = evaluateOnBackend<Kernel::NodalVectors>(
      [=] TRIBOL_HOST_DEVICE { return Kernel::evaluate( current.view(), reference.view() ); } );
  for ( IndexT row = 0; row < Kernel::size_coords; ++row ) {
    EXPECT_EQ( normals[row], 0.0 );
  }
#ifdef TRIBOL_USE_ENZYME
  const auto jacobian = evaluateOnBackend<Kernel::Jacobian>(
      [=] TRIBOL_HOST_DEVICE { return Kernel::differentiate( current.view(), reference.view() ); } );
  for ( IndexT entry = 0; entry < jacobian.size(); ++entry ) {
    EXPECT_EQ( jacobian[entry], 0.0 );
  }
#endif
}

INSTANTIATE_TEST_SUITE_P( backends, SingleMortarPairTest, detail::enabledExecutionModes() );
}  // namespace

int main( int argc, char** argv )
{
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
