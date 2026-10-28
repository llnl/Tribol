// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Tests base and paired edge quadrature on every enabled backend: polynomial accuracy, interval selection,
 * bounded writes, invalid geometry, and frozen reference points mapped to changed geometry. Factory tests check order
 * selection.
 */

#include <cmath>
#include <stdexcept>
#include <utility>
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "ExecutionTest.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/integ/EdgeQuadratureRule.hpp"
#include "tribol/integ/PairQuadratureRule.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/integ/QuadratureRuleFactory.hpp"

namespace {
using namespace tribol;
constexpr double tolerance = 1.0e-12;  ///< Absolute tolerance for coordinates, weights, and polynomial integrals.

/** @brief Run local rule calculations in memory accessible to the selected backend. */
class EdgeQuadratureTest : public detail::ExecutionTest {
 protected:
  /**
   * @brief Integrate each supported monomial on the reference interval using all three point counts.
   * @tparam Count Number of Gauss points.
   */
  template <int Count>
  void checkPolynomialIntegrals()
  {
    using Rule = EdgeQuadratureRule<Count>;
    constexpr int degree_count = Rule::polynomial_order + 1;
    ArrayT<RealT> integrals( degree_count, degree_count, allocatorId() );
    ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
    const auto output = integrals.view();
    const auto status = results.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      const auto rule = Rule{}.generate();
      status[0] = rule.result;
      const auto points = rule.getView();
      for ( int degree = 0; degree < degree_count; ++degree ) {
        output[degree] = 0.0;
        for ( int q = 0; q < Count; ++q ) {
          RealT monomial = 1.0;
          for ( int power = 0; power < degree; ++power ) monomial *= points.parent_coordinates( q, 0 );
          output[degree] += monomial * points.reference_weights[q];
        }
      }
    } );
    const ArrayT<RealT, 1, MemorySpace::Host> values( integrals );
    const ArrayT<QuadratureResult, 1, MemorySpace::Host> statuses( results );
    ASSERT_EQ( statuses[0].status, QuadratureStatus::Success );
    ASSERT_EQ( statuses[0].required_point_count, Count );
    for ( int degree = 0; degree < degree_count; ++degree ) {
      EXPECT_NEAR( values[degree], ( degree % 2 == 0 ? 2.0 / ( degree + 1 ) : 0.0 ), tolerance );
    }
  }

  /**
   * @brief Check source coordinates and measure for full, partial, smoothed, and disjoint intervals.
   * @tparam Count Number of Gauss points.
   */
  template <int Count>
  void checkIntervals()
  {
    struct IntervalCase {
      RealT target_right;    ///< First target endpoint, for a target with reversed orientation.
      RealT target_left;     ///< Second target endpoint.
      RealT smoothing;       ///< Source-length fraction used to smooth interval endpoints.
      RealT expected_lower;  ///< Expected source parent lower bound.
      RealT expected_upper;  ///< Expected source parent upper bound.
    };
    // The source spans x=[0,4]. The smoothed case extends slightly past the right endpoint of that source.
    const IntervalCase cases[] = { { 4.0, 0.0, 0.0, -1.0, 1.0 },   { 3.0, 1.0, 0.0, -0.5, 0.5 },
                                   { 4.4, 2.0, 0.1, 0.0, 1.0 },    { 4.0, 0.0, 0.1, -0.95, 0.95 },
                                   { 4.6, 4.2, 0.1, 0.9875, 1.0 }, { 8.0, 6.0, 0.1, 1.0, 1.0 } };
    for ( const auto test : cases ) {
      SCOPED_TRACE( test.target_left );
      ArrayT<RealT, 2> records( axom::StackArray<axom::IndexType, 2>{ Count, 7 }, allocatorId() );
      ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
      const auto output = records.view();
      const auto status = results.view();
      forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
        const MdArray<RealT, 2, 2> source{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
        const MdArray<RealT, 2, 2> target{ MdArray<RealT, 2>{ test.target_right, 0.2 },
                                           MdArray<RealT, 2>{ test.target_left, 0.2 } };
        const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<Count>> generator{
            SmoothedProjectedEdgeMapping{ test.smoothing } };
        const auto rule = generator.generate( source.view(), target.view() );
        status[0] = rule.result;
        const auto points = rule.getView();
        for ( IndexT q = 0; q < rule.result.required_point_count; ++q ) {
          const auto physical1 = LinearEdgeGeometry::position( source.view(), points.mesh1_parent_coordinates( q, 0 ) );
          const auto physical2 = LinearEdgeGeometry::position( target.view(), points.mesh2_parent_coordinates( q, 0 ) );
          output( q, 0 ) = points.mesh1_parent_coordinates( q, 0 );
          output( q, 1 ) = physical1[0];
          output( q, 2 ) = physical1[1];
          output( q, 3 ) = points.physical_weights[q];
          output( q, 4 ) = points.mesh2_parent_coordinates( q, 0 );
          output( q, 5 ) = physical2[0];
          output( q, 6 ) = physical2[1];
        }
      } );
      const ArrayT<QuadratureResult, 1, MemorySpace::Host> statuses( results );
      const ArrayT<RealT, 2, MemorySpace::Host> values( records );
      ASSERT_EQ( statuses[0].status, QuadratureStatus::Success );
      const bool empty = test.expected_lower == test.expected_upper;
      ASSERT_EQ( statuses[0].required_point_count, empty ? 0 : Count );
      RealT measure = 0.0;
      for ( IndexT q = 0; q < statuses[0].required_point_count; ++q ) {
        EXPECT_GE( values( q, 0 ), test.expected_lower );
        EXPECT_LE( values( q, 0 ), test.expected_upper );
        EXPECT_NEAR( values( q, 1 ), 2.0 * ( 1.0 + values( q, 0 ) ), tolerance );
        EXPECT_EQ( values( q, 2 ), 0.0 );
        EXPECT_NEAR( values( q, 4 ),
                     2.0 * ( values( q, 1 ) - test.target_right ) / ( test.target_left - test.target_right ) - 1.0,
                     tolerance );
        EXPECT_NEAR( values( q, 5 ), values( q, 1 ), tolerance );
        EXPECT_NEAR( values( q, 6 ), 0.2, tolerance );
        measure += values( q, 3 );
      }
      EXPECT_NEAR( measure, 2.0 * ( test.expected_upper - test.expected_lower ), tolerance );
    }
  }
};

// The base rule integrates its supported polynomial degrees without consulting either element.
TEST_P( EdgeQuadratureTest, integrates_supported_polynomial_degrees )
{
  checkPolynomialIntegrals<1>();
  checkPolynomialIntegrals<2>();
  checkPolynomialIntegrals<3>();
}

// Overlap and smoothing select the integration interval; a disjoint pair has a valid zero-point rule.
TEST_P( EdgeQuadratureTest, selects_source_interval_from_projection )
{
  checkIntervals<1>();
  checkIntervals<2>();
  checkIntervals<3>();
}

// Invalid edges report failure; insufficient storage reports the full count without writing outside its slice.
TEST_P( EdgeQuadratureTest, rejects_invalid_geometry_and_insufficient_storage )
{
  constexpr int point_count = 3;
  constexpr RealT unwritten = -7.0;
  ArrayT<RealT> storage( point_count, point_count, allocatorId() );
  storage.fill( unwritten );
  ArrayT<QuadratureResult> results( 3, 3, allocatorId() );
  const auto output = storage.view();
  const auto status = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<point_count>> generator{};
    const MdArray<RealT, 2, 2> source{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } },
        target{ MdArray<RealT, 2>{ 4.0, 0.2 }, MdArray<RealT, 2>{ 0.0, 0.2 } };
    const MdArray<RealT, 2, 2> degenerate{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 0.0, 0.0 } };
    status[0] = generator.generate( degenerate.view(), target.view() ).result;
    status[1] = generator.generate( source.view(), degenerate.view() ).result;
    // A capacity of one has a guard entry on either side of its weight array.
    MdArray<RealT, 1, 1> parent;
    MdArray<RealT, 1, 1> target_parent;
    const PairQuadratureView<1, 1> bounded{
        { parent.data(), { 1, 1 } }, { target_parent.data(), { 1, 1 } }, { output.data() + 1, 1 } };
    status[2] = generator.generate( source.view(), target.view(), bounded );
  } );
  const ArrayT<QuadratureResult, 1, MemorySpace::Host> statuses( results );
  for ( int invalid : { 0, 1 } ) {
    EXPECT_EQ( statuses[invalid].status, QuadratureStatus::InvalidGeometry );
    EXPECT_EQ( statuses[invalid].required_point_count, 0 );
  }
  EXPECT_EQ( statuses[2].status, QuadratureStatus::InsufficientCapacity );
  EXPECT_EQ( statuses[2].required_point_count, point_count );
  const ArrayT<RealT, 1, MemorySpace::Host> values( storage );
  EXPECT_EQ( values[0], unwritten );
  EXPECT_EQ( values[point_count - 1], unwritten );
}

// Frozen parent coordinates remain fixed while the physical source Jacobian follows the changed edge length.
TEST_P( EdgeQuadratureTest, maps_frozen_reference_points_to_current_geometry )
{
  ArrayT<RealT> records( 2, 2, allocatorId() );
  const auto output = records.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<3>> generator{};
    const auto reference = EdgeQuadratureRule<3>{}.generate();
    const MdArray<RealT, 2, 2> source{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } },
        stretched{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 8.0, 0.0 } };
    const auto original = generator.mapReferenceQuadrature( source.view(), source.view(), reference.getView() );
    const auto changed = generator.mapReferenceQuadrature( stretched.view(), source.view(), reference.getView() );
    output[0] = changed.mesh1_parent_coordinates( 0, 0 ) - original.mesh1_parent_coordinates( 0, 0 );
    output[1] = changed.physical_weights[0] / original.physical_weights[0];
  } );
  const ArrayT<RealT, 1, MemorySpace::Host> values( records );
  EXPECT_EQ( values[0], 0.0 );
  EXPECT_NEAR( values[1], 2.0, tolerance );
}

// A tilted mortar edge must receive points projected along its normal; its length must not rescale source weights.
TEST_P( EdgeQuadratureTest, maps_both_parent_coordinates_and_preserves_source_measure )
{
  constexpr int count = 3;
  constexpr int columns = 5;
  ArrayT<RealT, 2> records( axom::StackArray<axom::IndexType, 2>{ count, columns }, allocatorId() );
  ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
  const auto output = records.view();
  const auto status = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    // Source is x=[0,4], y=0. Target runs from (3,2) to (1,0); the target has a different length and direction.
    const MdArray<RealT, 2, 2> source{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
    const MdArray<RealT, 2, 2> target{ MdArray<RealT, 2>{ 3.0, 2.0 }, MdArray<RealT, 2>{ 1.0, 0.0 } };
    const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<count>> generator{
        SmoothedProjectedEdgeMapping{ 0.0 } };
    // Use a full source interval so this check isolates reverse mapping from overlap selection.
    const auto reference = EdgeQuadratureRule<count>{}.generate();
    const auto rule = generator.mapReferenceQuadrature( source.view(), target.view(), reference.getView() );
    status[0] = rule.result;
    const auto points = rule.getView();
    for ( int q = 0; q < count; ++q ) {
      const auto physical1 = LinearEdgeGeometry::position( source.view(), points.mesh1_parent_coordinates( q, 0 ) );
      const auto physical2 = LinearEdgeGeometry::position( target.view(), points.mesh2_parent_coordinates( q, 0 ) );
      output( q, 0 ) = physical1[0];
      output( q, 1 ) = physical2[0];
      output( q, 2 ) = physical2[1];
      output( q, 3 ) = points.physical_weights[q];
      output( q, 4 ) = points.mesh2_parent_coordinates( q, 0 );
    }
  } );
  const ArrayT<QuadratureResult, 1, MemorySpace::Host> statuses( results );
  ASSERT_EQ( statuses[0].status, QuadratureStatus::Success );
  ASSERT_EQ( statuses[0].required_point_count, count );
  const ArrayT<RealT, 2, MemorySpace::Host> values( records );
  RealT measure = 0.0, integral = 0.0;
  for ( int q = 0; q < count; ++q ) {
    const RealT source_x = values( q, 0 );
    // The target line is y=x-1. Its normal projection of (source_x,0) is ((source_x+1)/2,(source_x-1)/2).
    EXPECT_NEAR( values( q, 1 ), 0.5 * ( source_x + 1.0 ), tolerance );
    EXPECT_NEAR( values( q, 2 ), 0.5 * ( source_x - 1.0 ), tolerance );
    EXPECT_NEAR( values( q, 4 ), 1.5 - 0.5 * source_x, tolerance );
    measure += values( q, 3 );
    integral += source_x * values( q, 1 ) * values( q, 3 );
  }
  EXPECT_NEAR( measure, 4.0, tolerance );
  // Integral from 0 to 4 of x * (x+1)/2 dx = 44/3, using field values on both elements.
  EXPECT_NEAR( integral, 44.0 / 3.0, tolerance );
}

// Frozen source points still require reverse mapping onto the updated target, even when the source does not change.
TEST_P( EdgeQuadratureTest, remaps_frozen_points_to_updated_target )
{
  ArrayT<RealT> records( 3, 3, allocatorId() );
  const auto output = records.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const MdArray<RealT, 2, 2> source{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
    const MdArray<RealT, 2, 2> target{ MdArray<RealT, 2>{ 4.0, 0.2 }, MdArray<RealT, 2>{ 0.0, 0.2 } };
    const MdArray<RealT, 2, 2> translated{ MdArray<RealT, 2>{ 5.0, 0.2 }, MdArray<RealT, 2>{ 1.0, 0.2 } };
    const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<1>> generator{};
    const auto reference = generator.generateReferenceQuadrature( source.view(), target.view() );
    const auto original = generator.mapReferenceQuadrature( source.view(), target.view(), reference.getView() );
    const auto changed = generator.mapReferenceQuadrature( source.view(), translated.view(), reference.getView() );
    output[0] = changed.mesh1_parent_coordinates( 0, 0 ) - original.mesh1_parent_coordinates( 0, 0 );
    output[1] = changed.mesh2_parent_coordinates( 0, 0 ) - original.mesh2_parent_coordinates( 0, 0 );
    output[2] = changed.physical_weights[0] - original.physical_weights[0];
  } );
  const ArrayT<RealT, 1, MemorySpace::Host> values( records );
  EXPECT_EQ( values[0], 0.0 );
  EXPECT_NEAR( values[1], 0.5, tolerance );
  EXPECT_EQ( values[2], 0.0 );
}

/**
 * @brief Test a mapping whose integration coordinates are physical x, rather than either element's parent coordinates.
 *
 * Inputs are horizontal straight edges with mesh1 ordered left-to-right and mesh2 ordered right-to-left. Their common
 * interval can lie on a plane between the edges; reverse mapping gives a point on each edge with the same x coordinate.
 */
struct ParallelEdgeMapping {
  using Mesh1Geometry = LinearEdgeGeometry;  ///< First straight-edge geometry.
  using Mesh2Geometry = LinearEdgeGeometry;  ///< Second straight-edge geometry.

  /**
   * @brief Find the common physical-x interval of the two horizontal edges.
   * @param mesh1 Left-to-right first edge.
   * @param mesh2 Right-to-left second edge.
   * @return Common interval in physical x; nonincreasing bounds give zero measure.
   */
  TRIBOL_HOST_DEVICE EdgeQuadratureInterval makeIntegrationSurface( LinearEdgeGeometry::Coordinates mesh1,
                                                                    LinearEdgeGeometry::Coordinates mesh2 ) const
  {
    const RealT left = mesh1( 0, 0 ) > mesh2( 1, 0 ) ? mesh1( 0, 0 ) : mesh2( 1, 0 );
    const RealT right = mesh1( 1, 0 ) < mesh2( 0, 0 ) ? mesh1( 1, 0 ) : mesh2( 0, 0 );
    return { MdArray<RealT, 2>{ left, right }, true };
  }

  /**
   * @brief Map physical-x integration points onto each horizontal edge.
   * @param mesh1 First edge endpoints.
   * @param mesh2 Second edge endpoints.
   * @param reference Points and weights expressed in physical x.
   * @param [out] output Parent coordinates on both elements and physical weights.
   * @return Success with initialized count, or InsufficientCapacity with the full count and untouched output.
   */
  TRIBOL_HOST_DEVICE QuadratureResult mapToElements( LinearEdgeGeometry::Coordinates mesh1,
                                                     LinearEdgeGeometry::Coordinates mesh2,
                                                     ReferenceQuadratureView<1> reference,
                                                     PairQuadratureView<1, 1> output ) const
  {
    const IndexT count = reference.numberOfPoints();
    if ( output.capacity() < count ) return { QuadratureStatus::InsufficientCapacity, count };
    for ( IndexT q = 0; q < count; ++q ) {
      const RealT x = reference.parent_coordinates( q, 0 );
      output.mesh1_parent_coordinates( q, 0 ) = 2.0 * ( x - mesh1( 0, 0 ) ) / ( mesh1( 1, 0 ) - mesh1( 0, 0 ) ) - 1.0;
      output.mesh2_parent_coordinates( q, 0 ) = 2.0 * ( x - mesh2( 0, 0 ) ) / ( mesh2( 1, 0 ) - mesh2( 0, 0 ) ) - 1.0;
      output.physical_weights[q] = reference.reference_weights[q];
    }
    return { QuadratureStatus::Success, count };
  }
};

// The same base Gauss rule must also work when integration points are expressed independently of both elements.
TEST_P( EdgeQuadratureTest, composes_with_a_different_integration_mapping )
{
  ArrayT<RealT> records( 3, 3, allocatorId() );
  const auto output = records.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const MdArray<RealT, 2, 2> source{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
    const MdArray<RealT, 2, 2> target{ MdArray<RealT, 2>{ 3.0, 0.2 }, MdArray<RealT, 2>{ 1.0, 0.2 } };
    const PairQuadratureRule<ParallelEdgeMapping, EdgeQuadratureRule<1>> generator{};
    const auto rule = generator.generate( source.view(), target.view() );
    const auto points = rule.getView();
    output[0] = points.mesh1_parent_coordinates( 0, 0 );
    output[1] = points.mesh2_parent_coordinates( 0, 0 );
    output[2] = points.physical_weights[0];
  } );
  const ArrayT<RealT, 1, MemorySpace::Host> values( records );
  // The common interval is x=[1,3], with midpoint x=2: parent zero on both elements and physical measure two.
  EXPECT_NEAR( values[0], 0.0, tolerance );
  EXPECT_NEAR( values[1], 0.0, tolerance );
  EXPECT_NEAR( values[2], 2.0, tolerance );
}

INSTANTIATE_TEST_SUITE_P( execution, EdgeQuadratureTest, detail::enabledExecutionModes() );

// Polynomial order selects the smallest registered rule that integrates that degree.
TEST( QuadratureRuleFactory, selects_registered_rule_by_polynomial_order )
{
  int calls = 0;
  const auto create = [&]<typename QuadratureRule>() {
    ++calls;
    return QuadratureRule::points_per_base_rule;
  };
  const std::pair<int, int> orders_and_points[] = { { 0, 1 }, { 1, 1 }, { 2, 2 }, { 3, 2 }, { 4, 3 }, { 5, 3 } };
  for ( const auto& [order, expected_points] : orders_and_points ) {
    const int before = calls;
    const auto selected = createWithQuadratureRule( order, create );
    EXPECT_EQ( selected, expected_points );
    EXPECT_EQ( calls, before + 1 );
  }
  for ( const int invalid : { -1, 6 } ) {
    const int before = calls;
    EXPECT_THROW( createWithQuadratureRule( invalid, create ), std::invalid_argument );
    EXPECT_EQ( calls, before );
  }
}

/**
 * @brief Describe synthetic surface rules for degree-based factory selection.
 *
 * @tparam Degree Exactly integrated reference polynomial degree.
 * @tparam Points Points in the synthetic base rule.
 */
template <int Degree, int Points>
struct SurfaceQuadratureRuleMetadata {
  static constexpr int polynomial_order = Degree;      ///< Declared exact polynomial degree.
  static constexpr int points_per_base_rule = Points;  ///< Declared points per base rule.
};

/**
 * @brief Surface rules can have different degree-to-point mappings; selection must use polynomial degree, not point
 * count.
 */
TEST( QuadratureRuleFactory, supports_surface_rule_families )
{
  using LinearTriangle = SurfaceQuadratureRuleMetadata<1, 1>;
  using QuadraticTriangle = SurfaceQuadratureRuleMetadata<2, 3>;
  using FourthDegreeTriangle = SurfaceQuadratureRuleMetadata<4, 6>;
  const auto create = []<typename QuadratureRule>() { return QuadratureRule::points_per_base_rule; };
  const std::pair<int, int> orders_and_points[] = { { 0, 1 }, { 1, 1 }, { 2, 3 }, { 3, 6 }, { 4, 6 } };
  for ( const auto& [order, expected_points] : orders_and_points ) {
    const auto selected =
        createWithQuadratureRule<LinearTriangle, QuadraticTriangle, FourthDegreeTriangle>( order, create );
    EXPECT_EQ( selected, expected_points );
  }
}

}  // namespace

/**
 * @brief Initialize logging and run portable quadrature tests.
 *
 * @param [in,out] argc Command-line argument count; Google Test may consume arguments.
 * @param [in,out] argv Command-line argument strings.
 * @return Google Test exit status.
 */
int main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
