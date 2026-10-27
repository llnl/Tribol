// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Tests local edge quadrature on every enabled backend: polynomial accuracy, interval selection, bounded
 * writes, invalid geometry, and frozen reference points mapped to changed geometry. Factory tests check order
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
#include "tribol/integ/QuadratureRuleFactory.hpp"

namespace {
using namespace tribol;
constexpr double tolerance = 1.0e-12;  ///< Absolute tolerance for coordinates, weights, and polynomial integrals.

/** @brief Run local rule calculations in memory accessible to the selected backend. */
class EdgeQuadratureTest : public detail::ExecutionTest {
 protected:
  /**
   * @brief Integrate each supported monomial on an edge of length four, using all three point counts.
   * @tparam Count Number of Gauss points.
   */
  template <int Count>
  void checkPolynomialIntegrals()
  {
    using Rule = EdgeQuadratureRule<Count>;
    constexpr RealT edge_length = 4.0;
    constexpr int degree_count = Rule::polynomial_order + 1;
    ArrayT<RealT> integrals( degree_count, degree_count, allocatorId() );
    ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
    const auto output = integrals.view();
    const auto status = results.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      const MdArray<RealT, 2, 2> endpoints{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ edge_length, 0.0 } };
      const auto rule = Rule{}.template generate<RealT>( endpoints.view() );
      status[0] = rule.result;
      const auto points = rule.getView();
      for ( int degree = 0; degree < degree_count; ++degree ) {
        output[degree] = 0.0;
        for ( int q = 0; q < Count; ++q ) {
          RealT monomial = 1.0;
          for ( int power = 0; power < degree; ++power ) monomial *= points.physical_coordinates( q, 0 );
          output[degree] += monomial * points.physical_weights[q];
        }
      }
    } );
    const ArrayT<RealT, 1, MemorySpace::Host> values( integrals );
    const ArrayT<QuadratureResult, 1, MemorySpace::Host> statuses( results );
    ASSERT_EQ( statuses[0].status, QuadratureStatus::Success );
    ASSERT_EQ( statuses[0].required_point_count, Count );
    for ( int degree = 0; degree < degree_count; ++degree ) {
      EXPECT_NEAR( values[degree], std::pow( edge_length, degree + 1 ) / ( degree + 1 ), tolerance );
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
      ArrayT<RealT, 2> records( axom::StackArray<axom::IndexType, 2>{ Count, 4 }, allocatorId() );
      ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
      const auto output = records.view();
      const auto status = results.view();
      forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
        const MdArray<RealT, 2, 2> source{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
        const MdArray<RealT, 2, 2> target{ MdArray<RealT, 2>{ test.target_right, 0.2 },
                                           MdArray<RealT, 2>{ test.target_left, 0.2 } };
        const EdgeQuadratureRule<Count, SmoothedEdgeProjection> generator{ SmoothedEdgeProjection{ test.smoothing } };
        const auto rule = generator.template generate<RealT>( source.view(), target.view() );
        status[0] = rule.result;
        const auto points = rule.getView();
        for ( IndexT q = 0; q < rule.result.required_point_count; ++q ) {
          output( q, 0 ) = points.parent_coordinates( q, 0 );
          output( q, 1 ) = points.physical_coordinates( q, 0 );
          output( q, 2 ) = points.physical_coordinates( q, 1 );
          output( q, 3 ) = points.physical_weights[q];
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
        measure += values( q, 3 );
      }
      EXPECT_NEAR( measure, 2.0 * ( test.expected_upper - test.expected_lower ), tolerance );
    }
  }
};

// Physical weights must include the source Jacobian exactly once for all supported polynomial degrees.
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
    const EdgeQuadratureRule<point_count, SmoothedEdgeProjection> generator{};
    const MdArray<RealT, 2, 2> source{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } },
        target{ MdArray<RealT, 2>{ 4.0, 0.2 }, MdArray<RealT, 2>{ 0.0, 0.2 } };
    const MdArray<RealT, 2, 2> degenerate{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 0.0, 0.0 } };
    status[0] = generator.template generate<RealT>( degenerate.view(), target.view() ).result;
    status[1] = generator.template generate<RealT>( source.view(), degenerate.view() ).result;
    // A capacity of one has a guard entry on either side of its weight array.
    MdArray<RealT, 1, 1> parent;
    MdArray<RealT, 1, 2> physical;
    const SurfaceQuadratureView<RealT, 1, 2> bounded{
        { parent.data(), { 1, 1 } }, { physical.data(), { 1, 2 } }, { output.data() + 1, 1 } };
    status[2] = generator.template generate<RealT>( source.view(), target.view(), bounded );
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
    const EdgeQuadratureRule<3> generator{};
    const auto reference = generator.template generateReferenceQuadrature<RealT>();
    const MdArray<RealT, 2, 2> source{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } },
        stretched{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 8.0, 0.0 } };
    const auto original = generator.template mapReferenceQuadrature<RealT>( source.view(), reference.getView() );
    const auto changed = generator.template mapReferenceQuadrature<RealT>( stretched.view(), reference.getView() );
    output[0] = changed.parent_coordinates( 0, 0 ) - original.parent_coordinates( 0, 0 );
    output[1] = changed.physical_weights[0] / original.physical_weights[0];
  } );
  const ArrayT<RealT, 1, MemorySpace::Host> values( records );
  EXPECT_EQ( values[0], 0.0 );
  EXPECT_NEAR( values[1], 2.0, tolerance );
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
  for ( const auto [order, expected_points] : orders_and_points ) {
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
  for ( const auto [order, expected_points] : orders_and_points ) {
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
