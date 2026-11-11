// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Test common-line quadrature and direct penalty residuals on every enabled backend. Host Enzyme checks compare
 * the full pair Jacobian with finite differences; MFEM assembly and selection have separate integration tests.
 * Crossing edges check clipped interval bounds, point locations, and uniform centroid pressure for every base rule.
 */
#include <cmath>
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "ExecutionTest.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/physics/CommonPlane.hpp"
#include "tribol/physics/CommonPlaneFormulation.hpp"

namespace {
using namespace tribol;
constexpr RealT tolerance = 1.0e-12;  ///< Roundoff allowance for geometry and exact polynomial integrals.
using Coordinates = MdArray<RealT, LinearEdgeGeometry::vertex_count, LinearEdgeGeometry::spatial_dimension>;

// Common Plane exposes the same values and quadrature state required by the shared pair derivative calculations.
static_assert( PairFunctionType<CommonPlanePairFunction<EdgeQuadratureRuleGenerator<1>>> );

/**
 * @brief Copy two edge-coordinate arrays into the pair function's input order.
 * @param coordinates1 First edge's vertex coordinates.
 * @param coordinates2 Second edge's vertex coordinates.
 * @return Owned coordinates ordered by edge, vertex, and component.
 */
TRIBOL_HOST_DEVICE MdArray<RealT, 2 * Coordinates::size()> packPairCoordinates(
    LinearEdgeGeometry::Coords coordinates1, LinearEdgeGeometry::Coords coordinates2 )
{
  MdArray<RealT, 2 * Coordinates::size()> coordinates{};
  for ( int coordinate = 0; coordinate < Coordinates::size(); ++coordinate ) {
    coordinates[coordinate] = coordinates1[coordinate];
    coordinates[Coordinates::size() + coordinate] = coordinates2[coordinate];
  }
  return coordinates;
}

/** @brief Execute local geometry and residual calculations using backend-accessible storage. */
class CommonPlaneTest : public tribol::detail::ExecutionTest {
 protected:
  /** @brief Check exact parallel-edge loads and activation. @tparam Count Gauss point count. */
  template <int Count>
  void checkParallelResiduals()
  {
    struct Case {
      RealT separation;        ///< Height of the second edge; positive means penetration for these orientations.
      RealT residual_gap;      ///< Separation offset subtracted before activation.
      RealT expected_first_y;  ///< Expected residual at either first-edge node.
      bool active;             ///< Whether the nonpositive gap branch is taken.
    };
    // Unequal mesh springs combine to stiffness 1.5, rather than their arithmetic average 4.0.
    const Case cases[] = {
        { 0.1, 0.0, -0.15, true }, { -0.1, 0.0, 0.0, false }, { 0.0, 0.0, 0.0, true }, { -0.05, 0.1, -0.075, true } };
    for ( const auto test : cases ) {
      SCOPED_TRACE( test.separation );
      ArrayT<PairResidualValue<2 * Coordinates::size()>> results( 1, 1, allocatorId() );
      const auto output = results.view();
      forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
        const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.0 } };
        const Coordinates edge2{ MdArray<RealT, 2>{ 2.0, test.separation }, MdArray<RealT, 2>{ 0.0, test.separation } };
        const PairQuadratureRule<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<Count>> rule;
        const CommonPlanePairFunction<EdgeQuadratureRuleGenerator<Count>, OffsetGap<CommonPlaneGap>> function{
            rule, ComputePenaltyStiffnessPerArea( 2.0, 6.0 ), OffsetGap{ CommonPlaneGap{}, test.residual_gap } };
        const auto coordinates = packPairCoordinates( edge1.view(), edge2.view() );
        output[0] = function.evaluate( coordinates.view() );
      } );
      const ArrayT<PairResidualValue<2 * Coordinates::size()>, 1, MemorySpace::Host> host( results );
      ASSERT_EQ( host[0].quadrature_result.status, QuadratureStatus::Success );
      EXPECT_EQ( host[0].quadrature_result.required_point_count, Count );
      EXPECT_EQ( host[0].has_active_points, test.active );
      for ( int node = 0; node < BasisFunction<EDGE, LINEAR_LAGRANGE>::node_count; ++node ) {
        EXPECT_NEAR( host[0].residual[2 * node], 0.0, tolerance );
        EXPECT_NEAR( host[0].residual[2 * node + 1], test.expected_first_y, tolerance );
        EXPECT_NEAR( host[0].residual[2 * ( node + 2 )], 0.0, tolerance );
        EXPECT_NEAR( host[0].residual[2 * ( node + 2 ) + 1], -test.expected_first_y, tolerance );
      }
    }
  }

  /** @brief Check partial overlap, reverse mapping, and physical line measure. @tparam Count Gauss point count. */
  template <int Count>
  void checkQuadrature()
  {
    enum Column
    {
      Parent1,
      Parent2,
      Weight,
      Point1X,
      Point2X,
      ColumnCount
    };
    ArrayT<RealT, 2> records( axom::StackArray<axom::IndexType, 2>{ Count, ColumnCount }, allocatorId() );
    const auto output = records.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
      const Coordinates edge2{ MdArray<RealT, 2>{ 3.0, 0.2 }, MdArray<RealT, 2>{ 1.0, 0.2 } };
      const LinearEdgeGeometry geometry1( edge1.view() ), geometry2( edge2.view() );
      const PairQuadratureRule<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<Count>> rule;
      const auto generated = rule.generate( geometry1, geometry2 );
      const auto points = generated.getView();
      for ( int q = 0; q < Count; ++q ) {
        output( q, Parent1 ) = points.geometry1_coords( q, 0 );
        output( q, Parent2 ) = points.geometry2_coords( q, 0 );
        output( q, Weight ) = points.weights[q];
        output( q, Point1X ) = geometry1.position( output( q, Parent1 ) )[0];
        output( q, Point2X ) = geometry2.position( output( q, Parent2 ) )[0];
      }
    } );
    const ArrayT<RealT, 2, MemorySpace::Host> values( records );
    RealT measure = 0.0, second_moment = 0.0;
    for ( int q = 0; q < Count; ++q ) {
      EXPECT_GE( values( q, Parent1 ), -0.5 );
      EXPECT_LE( values( q, Parent1 ), 0.5 );
      EXPECT_NEAR( values( q, Point1X ), values( q, Point2X ), tolerance );
      EXPECT_NEAR( values( q, Parent2 ), 2.0 - values( q, Point1X ), tolerance );
      EXPECT_GT( values( q, Weight ), 0.0 );
      measure += values( q, Weight );
      second_moment += values( q, Weight ) * values( q, Point1X ) * values( q, Point1X );
    }
    EXPECT_NEAR( measure, 2.0, tolerance );
    if constexpr ( Count >= 2 ) {
      EXPECT_NEAR( second_moment, 26.0 / 3.0, tolerance );
    }
  }

  /** @brief Check crossing-edge intervals and centroid-pressure loads. @tparam Count Configured Gauss point count. */
  template <int Count>
  void checkInterpenetratingOverlap()
  {
    struct Case {
      RealT slope;          ///< Slope of edge1; edge2 has the opposite slope.
      RealT separation;     ///< Height of edge2 at x=0.0.
      RealT residual_gap;   ///< Offset applied to pressure after clipping the geometric interval.
      RealT lower;          ///< Exact lower endpoint of the penetrating interval.
      RealT upper;          ///< Exact upper endpoint of the penetrating interval.
      RealT total_force_y;  ///< Exact edge1 load for unit penalty and uniform centroid pressure.
    };
    // The first two full-overlap midpoints are open. The third is closed, and the last applies a pressure offset.
    const Case cases[] = { { 0.25, -0.25, 0.0, -1.0, -0.5, -0.0625 },
                           { -0.25, -0.25, 0.0, 0.5, 1.0, -0.0625 },
                           { 0.25, 0.25, 0.0, -1.0, 0.5, -0.5625 },
                           { 0.25, -0.25, 0.1, -1.0, -0.5, -0.1125 } };
    struct Sample {
      EdgeQuadratureInterval interval;                      ///< Calculated geometric bounds.
      PairResidualValue<2 * Coordinates::size()> residual;  ///< Pair loads and quadrature status.
      MdArray<RealT, Count> x1{};                           ///< Point positions on edge1.
      MdArray<RealT, Count> x2{};                           ///< Point positions on edge2.
      MdArray<RealT, Count> gaps{};                         ///< Unshifted point gaps.
      MdArray<RealT, Count> weights{};                      ///< Physical line weights.
    };
    for ( const auto test : cases ) {
      SCOPED_TRACE( testing::Message() << "points=" << Count << ", slope=" << test.slope << ", separation="
                                       << test.separation << ", residual_gap=" << test.residual_gap );
      ArrayT<Sample> samples( 1, 1, allocatorId() );
      const auto output = samples.view();
      forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
        const Coordinates edge1{ MdArray<RealT, 2>{ -1.0, -test.slope }, MdArray<RealT, 2>{ 1.0, test.slope } };
        const Coordinates edge2{ MdArray<RealT, 2>{ 1.0, test.separation - test.slope },
                                 MdArray<RealT, 2>{ -1.0, test.separation + test.slope } };
        const LinearEdgeGeometry geometry1( edge1.view() ), geometry2( edge2.view() );
        const PairQuadratureRule<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<Count>> rule;
        const auto points = rule.generate( geometry1, geometry2 );
        const auto line = CommonPlaneEdgeMapping::computeCommonPlaneLine( geometry1, geometry2 );
        output[0].interval = CommonPlaneEdgeMapping{}.makeIntegrationSurface( geometry1, geometry2 );
        const CommonPlanePairFunction<EdgeQuadratureRuleGenerator<Count>, OffsetGap<CommonPlaneGap>> function{
            rule, 1.0, OffsetGap{ CommonPlaneGap{}, test.residual_gap } };
        const auto coordinates = packPairCoordinates( edge1.view(), edge2.view() );
        output[0].residual = function.evaluate( coordinates.view() );
        for ( int point = 0; point < Count; ++point ) {
          const auto point1 = geometry1.position( points.geometry1_coords( point, 0 ) );
          const auto point2 = geometry2.position( points.geometry2_coords( point, 0 ) );
          output[0].x1[point] = point1[0];
          output[0].x2[point] = point2[0];
          output[0].gaps[point] = CommonPlaneGap{}( point1, point2, line );
          output[0].weights[point] = points.weights[point];
        }
      } );
      const ArrayT<Sample, 1, MemorySpace::Host> host( samples );
      const auto& sample = host[0];
      ASSERT_TRUE( sample.interval.valid );
      EXPECT_NEAR( sample.interval.bounds[0], test.lower, tolerance );
      EXPECT_NEAR( sample.interval.bounds[1], test.upper, tolerance );
      ASSERT_EQ( sample.residual.quadrature_result.status, QuadratureStatus::Success );
      EXPECT_EQ( sample.residual.quadrature_result.required_point_count, Count );
      EXPECT_TRUE( sample.residual.has_active_points );
      RealT measure = 0.0;
      RealT first_moment = 0.0;
      for ( int point = 0; point < Count; ++point ) {
        EXPECT_GT( sample.x1[point], test.lower );
        EXPECT_LT( sample.x1[point], test.upper );
        EXPECT_NEAR( sample.x1[point], sample.x2[point], tolerance );
        EXPECT_LT( sample.gaps[point], 0.0 );
        EXPECT_GT( sample.weights[point], 0.0 );
        measure += sample.weights[point];
        first_moment += sample.weights[point] * sample.x1[point];
      }
      const RealT center = 0.5 * ( test.lower + test.upper );
      EXPECT_NEAR( measure, test.upper - test.lower, tolerance );
      EXPECT_NEAR( first_moment / measure, center, tolerance );
      const MdArray<RealT, 4> loads{
          0.5 * ( 1.0 - center ) * test.total_force_y, 0.5 * ( 1.0 + center ) * test.total_force_y,
          -0.5 * ( 1.0 + center ) * test.total_force_y, -0.5 * ( 1.0 - center ) * test.total_force_y };
      for ( int node = 0; node < loads.size(); ++node ) {
        EXPECT_NEAR( sample.residual.residual[2 * node], 0.0, tolerance );
        EXPECT_NEAR( sample.residual.residual[2 * node + 1], loads[node], tolerance );
      }
    }
  }
};

// Each supported rule must integrate the physical overlap without multiplying by either edge Jacobian again.
TEST_P( CommonPlaneTest, maps_partial_overlap_and_physical_weights )
{
  checkQuadrature<1>();
  checkQuadrature<2>();
  checkQuadrature<3>();
}

// Penetration, opening, residual-gap shifts, and zero-gap activation use the same law for every supported rule.
TEST_P( CommonPlaneTest, integrates_parallel_edge_residuals )
{
  checkParallelResiduals<1>();
  checkParallelResiduals<2>();
  checkParallelResiduals<3>();
}

// A rotated 3-4-5 edge retains the physical length and balances both total force and moment.
TEST_P( CommonPlaneTest, balances_rotated_pair_forces_and_moments )
{
  ArrayT<RealT> records( 2 * Coordinates::size(), 2 * Coordinates::size(), allocatorId() );
  const auto output = records.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const Coordinates edge1{ MdArray<RealT, 2>{ 1.0, 2.0 }, MdArray<RealT, 2>{ 4.0, 6.0 } };
    const Coordinates edge2{ MdArray<RealT, 2>{ 3.8, 6.15 }, MdArray<RealT, 2>{ 0.8, 2.15 } };
    const PairQuadratureRule<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<3>> rule;
    const CommonPlanePairFunction<EdgeQuadratureRuleGenerator<3>> function{ rule, 2.0 };
    const auto coordinates = packPairCoordinates( edge1.view(), edge2.view() );
    const auto result = function.evaluate( coordinates.view() );
    for ( int i = 0; i < 2 * Coordinates::size(); ++i ) {
      output[i] = result.residual[i];
    }
  } );
  const ArrayT<RealT, 1, MemorySpace::Host> values( records );
  const RealT x[] = { 1.0, 4.0, 3.8, 0.8 }, y[] = { 2.0, 6.0, 6.15, 2.15 };
  RealT force_x = 0.0, force_y = 0.0, moment = 0.0;
  for ( int node = 0; node < 4; ++node ) {
    force_x += values[2 * node];
    force_y += values[2 * node + 1];
    moment += x[node] * values[2 * node + 1] - y[node] * values[2 * node];
  }
  EXPECT_NEAR( values[0], 1.0, tolerance );
  EXPECT_NEAR( values[1], -0.75, tolerance );
  EXPECT_NEAR( force_x, 0.0, tolerance );
  EXPECT_NEAR( force_y, 0.0, tolerance );
  EXPECT_NEAR( moment, 0.0, tolerance );
}

// Failed geometry is distinct from valid empty overlap, and insufficient capacity must leave all storage untouched.
TEST_P( CommonPlaneTest, reports_geometry_and_capacity_without_writing )
{
  enum Result
  {
    InvalidFirst,
    InvalidSecond,
    Disjoint,
    Cooriented,
    Perpendicular,
    Overflow,
    ResultCount
  };
  constexpr RealT unwritten = -7.0;
  ArrayT<QuadratureResult> records( ResultCount, ResultCount, allocatorId() );
  ArrayT<RealT> storage( 4, 4, allocatorId() );
  const auto output = records.view();
  const auto values = storage.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.0 } };
    const Coordinates edge2{ MdArray<RealT, 2>{ 2.0, 0.1 }, MdArray<RealT, 2>{ 0.0, 0.1 } };
    const Coordinates invalid{};
    const Coordinates disjoint{ MdArray<RealT, 2>{ 5.0, 0.1 }, MdArray<RealT, 2>{ 3.0, 0.1 } };
    const Coordinates cooriented{ MdArray<RealT, 2>{ 0.0, 0.1 }, MdArray<RealT, 2>{ 2.0, 0.1 } };
    const Coordinates perpendicular{ MdArray<RealT, 2>{ 1.0, -1.0 }, MdArray<RealT, 2>{ 1.0, 1.0 } };
    const PairQuadratureRule<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<3>> rule;
    output[InvalidFirst] = rule.generate( invalid.view(), edge2.view() ).result;
    output[InvalidSecond] = rule.generate( edge1.view(), invalid.view() ).result;
    output[Disjoint] = rule.generate( edge1.view(), disjoint.view() ).result;
    output[Cooriented] = rule.generate( edge1.view(), cooriented.view() ).result;
    output[Perpendicular] = rule.generate( edge1.view(), perpendicular.view() ).result;
    for ( int i = 0; i < values.size(); ++i ) {
      values[i] = unwritten;
    }
    const PairQuadratureView<1, 1> bounded{
        { values.data() + 1, { 2, 1 } }, { values.data() + 1, { 2, 1 } }, { values.data() + 1, 2 } };
    output[Overflow] = rule.generate( edge1.view(), edge2.view(), bounded );
  } );
  const ArrayT<QuadratureResult, 1, MemorySpace::Host> results( records );
  const ArrayT<RealT, 1, MemorySpace::Host> host( storage );
  for ( int invalid : { InvalidFirst, InvalidSecond } ) {
    EXPECT_EQ( results[invalid].status, QuadratureStatus::InvalidGeometry );
    EXPECT_EQ( results[invalid].required_point_count, 0 );
  }
  for ( int empty : { Disjoint, Cooriented, Perpendicular } ) {
    EXPECT_EQ( results[empty].status, QuadratureStatus::Success );
    EXPECT_EQ( results[empty].required_point_count, 0 );
  }
  EXPECT_EQ( results[Overflow].status, QuadratureStatus::InsufficientCapacity );
  EXPECT_EQ( results[Overflow].required_point_count, 3 );
  for ( const auto value : host ) {
    EXPECT_EQ( value, unwritten );
  }
}

// Each base rule integrates only the penetrating interval and uses the same uniform centroid pressure.
TEST_P( CommonPlaneTest, integrates_interpenetrating_overlap_with_configured_rule )
{
  checkInterpenetratingOverlap<1>();
  checkInterpenetratingOverlap<2>();
  checkInterpenetratingOverlap<3>();
}
INSTANTIATE_TEST_SUITE_P( execution, CommonPlaneTest, tribol::detail::enabledExecutionModes() );

#ifdef TRIBOL_USE_ENZYME
/** @brief Compare active-geometry Jacobians with central differences. @tparam Count Gauss point count. */
template <int Count>
void checkJacobian()
{
  using Function = CommonPlanePairFunction<EdgeQuadratureRuleGenerator<Count>, OffsetGap<CommonPlaneGap>>;
  const PairQuadratureRule<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<Count>> rule;
  const Function function{ rule, 2.5, OffsetGap{ CommonPlaneGap{}, 0.03 } };
  constexpr RealT step =
      2.0e-7;  // Small enough to resolve derivatives without changing the active or overlap branches.
  const Coordinates cases1[] = { Coordinates{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.2 } },
                                 Coordinates{ MdArray<RealT, 2>{ -1.0, 0.0 }, MdArray<RealT, 2>{ 1.0, 0.0 } } };
  const Coordinates cases2[] = { Coordinates{ MdArray<RealT, 2>{ 1.8, 0.35 }, MdArray<RealT, 2>{ 0.2, 0.1 } },
                                 Coordinates{ MdArray<RealT, 2>{ 0.8, 0.2 }, MdArray<RealT, 2>{ -0.8, -0.2 } } };
  for ( int geometry = 0; geometry < 2; ++geometry ) {
    const auto coordinates = packPairCoordinates( cases1[geometry].view(), cases2[geometry].view() );
    const auto expected = function.evaluate( coordinates.view() );
    const auto result = EnzymePairDerivative<Function>::valuesAndJacobian( coordinates.view(), function );
    ASSERT_EQ( result.quadrature_result.status, QuadratureStatus::Success );
    EXPECT_EQ( result.quadrature_result.required_point_count, Count );
    EXPECT_TRUE( result.has_active_points );
    for ( IndexT row = 0; row < Function::value_count; ++row ) {
      EXPECT_DOUBLE_EQ( result.values[row], expected.residual[row] );
    }
    for ( int column = 0; column < Function::dof_count; ++column ) {
      auto plus_coordinates = coordinates;
      auto minus_coordinates = coordinates;
      plus_coordinates[column] += step;
      minus_coordinates[column] -= step;
      const auto plus = function.evaluate( plus_coordinates.view() );
      const auto minus = function.evaluate( minus_coordinates.view() );
      ASSERT_EQ( plus.quadrature_result.status, QuadratureStatus::Success );
      ASSERT_EQ( minus.quadrature_result.status, QuadratureStatus::Success );
      for ( int row = 0; row < Function::value_count; ++row ) {
        EXPECT_NEAR( result.jacobian( row, column ), ( plus.residual[row] - minus.residual[row] ) / ( 2.0 * step ),
                     2.0e-7 )
            << "points=" << Count << ", geometry=" << geometry << ", row=" << row << ", column=" << column;
      }
    }
  }
  const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.0 } };
  const Coordinates edge2{ MdArray<RealT, 2>{ 2.0, 0.0 }, MdArray<RealT, 2>{ 0.0, 0.0 } };
  using UnshiftedFunction = CommonPlanePairFunction<EdgeQuadratureRuleGenerator<Count>>;
  const UnshiftedFunction unshifted{ rule, 1.0 };
  const auto coordinates = packPairCoordinates( edge1.view(), edge2.view() );
  const auto zero = unshifted.evaluate( coordinates.view() );
  EXPECT_TRUE( zero.has_active_points );
  const auto derivatives = EnzymePairDerivative<UnshiftedFunction>::valuesAndJacobian( coordinates.view(), unshifted );
  EXPECT_TRUE( derivatives.has_active_points );
  for ( IndexT row = 0; row < UnshiftedFunction::value_count; ++row ) {
    EXPECT_EQ( derivatives.values[row], 0.0 );
  }
  EXPECT_GT( derivatives.jacobian( 1, 1 ), 0.0 );
  const Coordinates invalid{};
  const auto invalid_coordinates = packPairCoordinates( invalid.view(), edge2.view() );
  const auto failure =
      EnzymePairDerivative<UnshiftedFunction>::valuesAndJacobian( invalid_coordinates.view(), unshifted );
  EXPECT_EQ( failure.quadrature_result.status, QuadratureStatus::InvalidGeometry );
  EXPECT_EQ( failure.quadrature_result.required_point_count, 0 );
  EXPECT_FALSE( failure.has_active_points );
  for ( int row = 0; row < failure.values.size(); ++row ) {
    EXPECT_EQ( failure.values[row], 0.0 );
    for ( IndexT column = 0; column < UnshiftedFunction::dof_count; ++column ) {
      EXPECT_EQ( failure.jacobian( row, column ), 0.0 );
    }
  }
}

// All reference counts differentiate current normals, interval bounds, edge lengths, and the interpenetration boundary
// and centroid pressure.
TEST( CommonPlaneJacobian, matches_finite_differences )
{
  checkJacobian<1>();
  checkJacobian<2>();
  checkJacobian<3>();
}
#endif
}  // namespace

/** @brief Initialize logging and run local common-plane tests. @param argc Argument count. @param argv Arguments.
 * @return Test status. */
int main( int argc, char** argv )
{
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
