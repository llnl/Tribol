// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Tests base and paired edge quadrature on every enabled backend: polynomial accuracy, interval selection,
 * bounded writes, invalid geometry, owned edge geometry, and frozen reference points mapped to changed geometry.
 * Pair generation preserves base-rule failures. Factory tests check order selection.
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
class EdgeQuadratureTest : public tribol::detail::ExecutionTest {
 protected:
  /**
   * @brief Check the Count-point Gauss-Legendre rule against exact monomial integrals on [-1, 1].
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
    // Use an independent analytical reference: integrating xi^degree from -1 to 1 gives
    // [xi^(degree+1)/(degree+1)] at the endpoints, hence zero for odd degrees and 2/(degree+1) for even degrees.
    // The expected value depends only on degree, not on the generated points or weights. Check every degree through
    // 2*Count-1 against this reference with the absolute error tolerance defined above (1e-12).
    for ( int degree = 0; degree < degree_count; ++degree ) {
      EXPECT_NEAR( values[degree], ( degree % 2 == 0 ? 2.0 / ( degree + 1 ) : 0.0 ), tolerance );
    }
  }

  /**
   * @brief Check edge1 coordinates and measure for full, partial, smoothed, and disjoint intervals.
   * @tparam Count Number of Gauss points.
   */
  template <int Count>
  void checkIntervals()
  {
    struct IntervalCase {
      const char* label;     ///< Description shown in assertion failures for this interval case.
      RealT edge2_right;     ///< First edge2 endpoint, for a edge2 with reversed orientation.
      RealT edge2_left;      ///< Second edge2 endpoint.
      RealT smoothing;       ///< Edge1-length fraction used to smooth interval endpoints.
      RealT expected_lower;  ///< Expected edge1 parent lower bound.
      RealT expected_upper;  ///< Expected edge1 parent upper bound.
    };
    // Edge1 spans x=[0,4]; edge2 endpoints are listed right-to-left to give the opposite edge orientation.
    const IntervalCase test_cases[] = {
        { "Full overlap without smoothing", 4.0, 0.0, 0.0, -1.0, 1.0 },
        { "Interior overlap without smoothing", 3.0, 1.0, 0.0, -0.5, 0.5 },
        { "Partial overlap extending past the right endpoint with smoothing", 4.4, 2.0, 0.1, 0.0, 1.0 },
        { "Full overlap with smoothed endpoints", 4.0, 0.0, 0.1, -0.95, 0.95 },
        { "Disjoint intervals with a right-end smoothing contribution", 4.6, 4.2, 0.1, 0.9875, 1.0 },
        { "Disjoint intervals beyond smoothing support", 8.0, 6.0, 0.1, 1.0, 1.0 } };
    for ( const auto& test_case : test_cases ) {
      SCOPED_TRACE( testing::Message() << test_case.label << ", Gauss points=" << Count );
      // Each records row stores one quadrature point. Columns are:
      // 0: edge1 parent coordinate; 1-2: edge1 physical (x,y); 3: physical integration weight;
      // 4: edge2 parent coordinate; 5-6: edge2 physical (x,y).
      ArrayT<RealT, 2> records( axom::StackArray<axom::IndexType, 2>{ Count, 7 }, allocatorId() );
      // One result for the entire generated rule: its status and required point count, rather than per-point data.
      ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
      // The kernel writes through these views; host copies below make the data available for assertions.
      const auto records_view = records.view();
      const auto results_view = results.view();
      forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
        const MdArray<RealT, 2, 2> edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
        const MdArray<RealT, 2, 2> edge2{ MdArray<RealT, 2>{ test_case.edge2_right, 0.2 },
                                          MdArray<RealT, 2>{ test_case.edge2_left, 0.2 } };
        const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<Count>> generator{
            SmoothedProjectedEdgeMapping{ test_case.smoothing } };
        const auto rule = generator.generate( edge1.view(), edge2.view() );
        results_view[0] = rule.result;
        const auto points = rule.getView();
        for ( IndexT q = 0; q < rule.result.required_point_count; ++q ) {
          const auto physical1 =
              LinearEdgeGeometry::position( edge1.view(), points.geometry1_parent_coordinates( q, 0 ) );
          const auto physical2 =
              LinearEdgeGeometry::position( edge2.view(), points.geometry2_parent_coordinates( q, 0 ) );
          records_view( q, 0 ) = points.geometry1_parent_coordinates( q, 0 );
          records_view( q, 1 ) = physical1[0];
          records_view( q, 2 ) = physical1[1];
          records_view( q, 3 ) = points.physical_weights[q];
          records_view( q, 4 ) = points.geometry2_parent_coordinates( q, 0 );
          records_view( q, 5 ) = physical2[0];
          records_view( q, 6 ) = physical2[1];
        }
      } );
      const ArrayT<QuadratureResult, 1, MemorySpace::Host> host_results( results );
      const ArrayT<RealT, 2, MemorySpace::Host> host_records( records );
      // Every case, including zero overlap, must produce a valid rule.
      ASSERT_EQ( host_results[0].status, QuadratureStatus::Success );
      const bool empty = test_case.expected_lower == test_case.expected_upper;
      // Empty intervals emit no points; nonempty intervals use the selected Gauss point count.
      ASSERT_EQ( host_results[0].required_point_count, empty ? 0 : Count );
      RealT measure = 0.0;
      for ( IndexT q = 0; q < host_results[0].required_point_count; ++q ) {
        // No point may fall below the expected edge1 parent interval.
        EXPECT_GE( host_records( q, 0 ), test_case.expected_lower );
        // No point may fall above the expected edge1 parent interval.
        EXPECT_LE( host_records( q, 0 ), test_case.expected_upper );
        // Mapping [-1,1] onto edge1's x=[0,4] gives x=2*(1+xi).
        EXPECT_NEAR( host_records( q, 1 ), 2.0 * ( 1.0 + host_records( q, 0 ) ), tolerance );
        // All mapped edge1 points lie on its horizontal edge at y=0.
        EXPECT_EQ( host_records( q, 2 ), 0.0 );
        // Inverting edge2's right-to-left x mapping must recover its reported parent coordinate.
        EXPECT_NEAR(
            host_records( q, 4 ),
            2.0 * ( host_records( q, 1 ) - test_case.edge2_right ) / ( test_case.edge2_left - test_case.edge2_right ) -
                1.0,
            tolerance );
        // Normal projection between these parallel edges preserves the physical x coordinate.
        EXPECT_NEAR( host_records( q, 5 ), host_records( q, 1 ), tolerance );
        // All mapped edge2 points lie on its horizontal edge at y=0.2.
        EXPECT_NEAR( host_records( q, 6 ), 0.2, tolerance );
        measure += host_records( q, 3 );
      }
      // Integrating 1 must give the physical interval length: parent interval width times edge1's Jacobian of 2.
      EXPECT_NEAR( measure, 2.0 * ( test_case.expected_upper - test_case.expected_lower ), tolerance );
    }
  }
};

// LinearEdgeGeometry owns an endpoint snapshot used for interpolation, projection, normals, length, and Jacobian.
TEST_P( EdgeQuadratureTest, owns_coordinate_snapshot )
{
  enum Result
  {
    MidpointX,
    MidpointY,
    ProjectedParent,
    ExtrapolatedParent,
    NormalX,
    NormalY,
    Length,
    Jacobian,
    ResultCount
  };
  ArrayT<RealT> records( ResultCount, ResultCount, allocatorId() );
  const auto output = records.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    MdArray<RealT, 2, 2> coordinates{ MdArray<RealT, 2>{ 1.0, 2.0 }, MdArray<RealT, 2>{ 4.0, 6.0 } };
    const LinearEdgeGeometry edge( coordinates.view() );
    // Deliberately change the caller's input array to verify that edge copied the endpoints rather than borrowing
    // them. Its private endpoint snapshot and cached edge quantities must still describe the original edge.
    coordinates( 0, 0 ) = -5.0;
    coordinates( 1, 1 ) = 8.0;
    const auto midpoint = edge.position( 0.0 );
    const auto normal = edge.normal();
    output[MidpointX] = midpoint[0];
    output[MidpointY] = midpoint[1];
    // Offset the original midpoint by one unit normal; its orthogonal projection still has parent coordinate zero.
    output[ProjectedParent] = edge.parentCoordinate( LinearEdgeGeometry::Point{ 3.3, 3.4 } );
    output[ExtrapolatedParent] = edge.parentCoordinate( edge.position( 1.5 ) );
    output[NormalX] = normal[0];
    output[NormalY] = normal[1];
    output[Length] = edge.length();
    output[Jacobian] = edge.jacobian();
  } );
  const ArrayT<RealT, 1, MemorySpace::Host> values( records );
  EXPECT_NEAR( values[MidpointX], 2.5, tolerance );
  EXPECT_NEAR( values[MidpointY], 4.0, tolerance );
  EXPECT_NEAR( values[ProjectedParent], 0.0, tolerance );
  EXPECT_NEAR( values[ExtrapolatedParent], 1.5, tolerance );
  EXPECT_NEAR( values[NormalX], 0.8, tolerance );
  EXPECT_NEAR( values[NormalY], -0.6, tolerance );
  EXPECT_NEAR( values[Length], 5.0, tolerance );
  EXPECT_NEAR( values[Jacobian], 2.5, tolerance );
}

// The base rule integrates its supported polynomial degrees without knowledge of either element.
TEST_P( EdgeQuadratureTest, integrates_supported_polynomial_degrees )
{
  checkPolynomialIntegrals<1>();
  checkPolynomialIntegrals<2>();
  checkPolynomialIntegrals<3>();
}

// Overlap and smoothing select the integration interval. A disjoint pair has no overlap between edge1's interval and
// edge2's projected interval; it has a valid zero-point rule when the separation is beyond the smoothing support.
TEST_P( EdgeQuadratureTest, selects_first_edge_interval_from_projection )
{
  checkIntervals<1>();
  checkIntervals<2>();
  checkIntervals<3>();
}

// Edges tilted relative to each other require projection along edge2's normal, rather than edge1's normal.
TEST_P( EdgeQuadratureTest, projects_oblique_edges_to_parent_bounds )
{
  ArrayT<RealT> bounds( 2, 2, allocatorId() );
  const auto output = bounds.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    // Edge1 is the steeper integration edge, directed up-right from (1,2) to (4,6).
    const MdArray<RealT, 2, 2> coordinates1{ MdArray<RealT, 2>{ 1.0, 2.0 }, MdArray<RealT, 2>{ 4.0, 6.0 } };
    // Edge2 crosses edge1, running down-left from (4,5) to (2,4); its normal points along (-1,2).
    const MdArray<RealT, 2, 2> coordinates2{ MdArray<RealT, 2>{ 4.0, 5.0 }, MdArray<RealT, 2>{ 2.0, 4.0 } };
    const auto interval = SmoothedProjectedEdgeMapping{ 0.0 }.makeIntegrationSurface(
        LinearEdgeGeometry( coordinates1.view() ), LinearEdgeGeometry( coordinates2.view() ) );
    output[0] = interval.bounds[0];
    output[1] = interval.bounds[1];
  } );
  const ArrayT<RealT, 1, MemorySpace::Host> values( bounds );
  // Projection along (-1,2) meets edge1 at fractions 0.4 and 0.9, giving parent coordinates -0.2 and 0.8.
  EXPECT_NEAR( values[0], -0.2, tolerance );
  EXPECT_NEAR( values[1], 0.8, tolerance );
}

// The edges are perpendicular. The lines through edge2's endpoints along its normal are parallel to edge1's line
// and never intersect it.
// Orthogonal projection is the fallback; both endpoints map to the same edge1 point, giving a zero-point rule.
TEST_P( EdgeQuadratureTest, keeps_endpoint_fallback_for_parallel_projection_lines )
{
  ArrayT<RealT> bounds( 2, 2, allocatorId() );
  ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
  const auto output = bounds.view();
  const auto status = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    // The edges are perpendicular: edge1 is vertical at x=0 and edge2 is horizontal at y=1. They cross at (0,1).
    // Projection along edge2's vertical normal follows x=2 and x=-2: distinct lines parallel to edge1 at x=0.
    const MdArray<RealT, 2, 2> edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 0.0, 4.0 } };
    const MdArray<RealT, 2, 2> edge2{ MdArray<RealT, 2>{ 2.0, 1.0 }, MdArray<RealT, 2>{ -2.0, 1.0 } };
    const SmoothedProjectedEdgeMapping mapping{ 0.0 };
    const auto interval =
        mapping.makeIntegrationSurface( LinearEdgeGeometry( edge1.view() ), LinearEdgeGeometry( edge2.view() ) );
    output[0] = interval.bounds[0];
    output[1] = interval.bounds[1];
    const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<3>> rule{ mapping };
    status[0] = rule.generate( edge1.view(), edge2.view() ).result;
  } );
  const ArrayT<RealT, 1, MemorySpace::Host> host_bounds( bounds );
  const ArrayT<QuadratureResult, 1, MemorySpace::Host> host_result( results );
  // Orthogonal fallback maps both edge2 endpoints to (0,1), one quarter along edge1: parent coordinate -0.5.
  EXPECT_NEAR( host_bounds[0], -0.5, tolerance );
  EXPECT_NEAR( host_bounds[1], -0.5, tolerance );
  EXPECT_EQ( host_result[0].status, QuadratureStatus::Success );
  EXPECT_EQ( host_result[0].required_point_count, 0 );
}

// Reference-point generation must skip edges whose normals point in the same general direction.
TEST_P( EdgeQuadratureTest, reference_generation_skips_edges_with_same_facing_normals )
{
  constexpr int point_count = 3;
  // Edge1 runs left-to-right at y=0. Edge2 also runs left-to-right, either parallel to edge1 or tilted upward.
  // Their x intervals overlap, but their normals point in the same general direction in both cases.
  for ( const RealT rise : { 0.0, 0.5 } ) {
    SCOPED_TRACE( testing::Message() << "edge2 endpoint rise=" << rise );
    // Store the operation's status and point count in memory accessible to the selected execution backend.
    ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
    const auto results_view = results.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      const MdArray<RealT, 2, 2> coordinates1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
      const MdArray<RealT, 2, 2> coordinates2{ MdArray<RealT, 2>{ 0.0, 0.2 }, MdArray<RealT, 2>{ 4.0, 0.2 + rise } };
      const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<point_count>> rule{};
      results_view[0] = rule.generateReferenceQuadrature( coordinates1.view(), coordinates2.view() ).result;
    } );
    // Copy the result to the host so assertions also work when the calculation ran on a GPU.
    const ArrayT<QuadratureResult, 1, MemorySpace::Host> host_results( results );
    // Skipping a same-facing pair is a valid outcome, not a geometry error.
    EXPECT_EQ( host_results[0].status, QuadratureStatus::Success );
    // No reference points should be generated for this pair.
    EXPECT_EQ( host_results[0].required_point_count, 0 );
  }
}

// Direct generation must skip same-facing edges before requiring or writing any caller-provided point storage.
TEST_P( EdgeQuadratureTest, direct_generation_skips_same_facing_edges_without_writing_points )
{
  constexpr int point_count = 3;
  constexpr RealT initial_storage_value = -7.0;
  // Test both parallel and tilted edges above edge1, with both edges ordered left-to-right.
  for ( const RealT rise : { 0.0, 0.5 } ) {
    SCOPED_TRACE( testing::Message() << "edge2 endpoint rise=" << rise );
    ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
    const auto results_view = results.view();
    // This single number lets us detect a write to output storage that should never be used for a skipped pair.
    ArrayT<RealT> point_storage( 1, 1, allocatorId() );
    point_storage.fill( initial_storage_value );
    const auto point_storage_view = point_storage.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      const MdArray<RealT, 2, 2> coordinates1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
      const MdArray<RealT, 2, 2> coordinates2{ MdArray<RealT, 2>{ 0.0, 0.2 }, MdArray<RealT, 2>{ 4.0, 0.2 + rise } };
      const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<point_count>> rule{};
      // Use zero capacity to check that this pair is skipped before any point storage is required: the result must be
      // success with zero points, not an insufficient-capacity error. With room for three points, the pair should
      // still be skipped and storage left unchanged; that case would require allocating space for all three points.
      // Here all coordinate and weight pointers refer to the same marked storage value, which must remain unchanged.
      const PairQuadratureView<1, 1> zero_capacity_output{ { point_storage_view.data(), { 0, 1 } },
                                                           { point_storage_view.data(), { 0, 1 } },
                                                           { point_storage_view.data(), 0 } };
      results_view[0] = rule.generate( coordinates1.view(), coordinates2.view(), zero_capacity_output );
    } );
    // Read back the operation result and storage value, including when generation ran on a GPU.
    const ArrayT<QuadratureResult, 1, MemorySpace::Host> host_results( results );
    const ArrayT<RealT, 1, MemorySpace::Host> host_point_storage( point_storage );
    // A skipped pair needs no storage, so zero capacity must not cause an insufficient-capacity error.
    EXPECT_EQ( host_results[0].status, QuadratureStatus::Success );
    // The caller should be told that no points are needed.
    EXPECT_EQ( host_results[0].required_point_count, 0 );
    // Skipping the pair must leave the caller's output storage unchanged.
    EXPECT_EQ( host_point_storage[0], initial_storage_value );
  }
}

// Mapping an existing reference rule must also skip edges whose current normals point in the same general direction.
TEST_P( EdgeQuadratureTest, reference_mapping_skips_edges_with_same_facing_normals )
{
  constexpr int point_count = 3;
  // Test both parallel and tilted edges above edge1, with both edges ordered left-to-right.
  for ( const RealT rise : { 0.0, 0.5 } ) {
    SCOPED_TRACE( testing::Message() << "edge2 endpoint rise=" << rise );
    ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
    const auto results_view = results.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      const MdArray<RealT, 2, 2> coordinates1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
      const MdArray<RealT, 2, 2> coordinates2{ MdArray<RealT, 2>{ 0.0, 0.2 }, MdArray<RealT, 2>{ 4.0, 0.2 + rise } };
      const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<point_count>> rule{};
      // Start with three reference points on [-1,1], generated independently of the two edges.
      const auto reference = EdgeQuadratureRule<point_count>{}.generate();
      results_view[0] =
          rule.mapReferenceQuadrature( coordinates1.view(), coordinates2.view(), reference.getView() ).result;
    } );
    // Copy the result to the host for assertions on every execution backend.
    const ArrayT<QuadratureResult, 1, MemorySpace::Host> host_results( results );
    // Rejecting this pair's orientation is a valid outcome even though reference points already exist.
    EXPECT_EQ( host_results[0].status, QuadratureStatus::Success );
    // None of the existing reference points should become paired quadrature points.
    EXPECT_EQ( host_results[0].required_point_count, 0 );
  }
}

// A zero-length edge must be rejected whether it is the first or second input.
TEST_P( EdgeQuadratureTest, rejects_collapsed_edges )
{
  constexpr int point_count = 3;
  // Make two generate() calls: case 0 replaces edge1 with collapsed_edge while keeping edge2 valid; case 1 keeps
  // edge1 valid and replaces edge2 with collapsed_edge. Store their status and point count in results[0] and
  // results[1]. case_count sizes this results array; it is the number of input cases, not the number of quadrature
  // points.
  constexpr int case_count = 2;
  ArrayT<QuadratureResult> results( case_count, case_count, allocatorId() );
  const auto results_view = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<point_count>> generator{};
    // Start with parallel edges spanning x=[0,4], separated by 0.2, with opposing normals.
    const MdArray<RealT, 2, 2> edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } },
        edge2{ MdArray<RealT, 2>{ 4.0, 0.2 }, MdArray<RealT, 2>{ 0.0, 0.2 } };
    // Coincident endpoints define a zero-length edge.
    const MdArray<RealT, 2, 2> collapsed_edge{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 0.0, 0.0 } };
    results_view[0] = generator.generate( collapsed_edge.view(), edge2.view() ).result;
    results_view[1] = generator.generate( edge1.view(), collapsed_edge.view() ).result;
  } );
  // Read the results on the host, including when generation ran on a GPU.
  const ArrayT<QuadratureResult, 1, MemorySpace::Host> host_results( results );
  for ( const int invalid_case : { 0, 1 } ) {
    SCOPED_TRACE( invalid_case == 0 ? "collapsed edge1" : "collapsed edge2" );
    EXPECT_EQ( host_results[invalid_case].status, QuadratureStatus::InvalidGeometry );
    // Invalid geometry must not request any quadrature points.
    EXPECT_EQ( host_results[invalid_case].required_point_count, 0 );
  }
}

// A valid pair requiring three points must report insufficient capacity when output has room for only one point.
TEST_P( EdgeQuadratureTest, reports_insufficient_capacity_without_writing_past_output )
{
  constexpr int point_count = 3;
  constexpr int output_capacity = 1;
  constexpr int guard_count = 2;
  constexpr int weight_storage_size = output_capacity + guard_count;
  constexpr RealT initial_storage_value = -7.0;
  // Weight storage layout: [left guard, space for one weight, right guard]. Only the middle entry belongs to output.
  // Both guards start at a known value so the assertions can detect writes immediately outside that output space.
  ArrayT<RealT> weight_storage( weight_storage_size, weight_storage_size, allocatorId() );
  weight_storage.fill( initial_storage_value );
  // Store the status and required point count for this generation call.
  ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
  const auto weight_storage_view = weight_storage.view();
  const auto results_view = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<point_count>> generator{};
    // The valid pair consists of parallel edges spanning x=[0,4], separated by 0.2, with opposing normals.
    const MdArray<RealT, 2, 2> edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } },
        edge2{ MdArray<RealT, 2>{ 4.0, 0.2 }, MdArray<RealT, 2>{ 0.0, 0.2 } };
    // For the valid pair, provide parent-coordinate arrays and weight storage for only one of the three points.
    MdArray<RealT, output_capacity, 1> edge1_parent_storage;
    MdArray<RealT, output_capacity, 1> edge2_parent_storage;
    const PairQuadratureView<1, 1> limited_output{ { edge1_parent_storage.data(), { output_capacity, 1 } },
                                                   { edge2_parent_storage.data(), { output_capacity, 1 } },
                                                   { weight_storage_view.data() + 1, output_capacity } };
    results_view[0] = generator.generate( edge1.view(), edge2.view(), limited_output );
  } );
  // Copy the operation results and weight storage to the host, including when generation ran on a GPU.
  const ArrayT<QuadratureResult, 1, MemorySpace::Host> host_results( results );
  const ArrayT<RealT, 1, MemorySpace::Host> host_weight_storage( weight_storage );
  // The geometry is valid, but room for one point is insufficient for the selected three-point rule.
  EXPECT_EQ( host_results[0].status, QuadratureStatus::InsufficientCapacity );
  // Report the full required count so the caller knows how much storage is needed.
  EXPECT_EQ( host_results[0].required_point_count, point_count );
  // The left guard entry must remain unchanged: no write before the output space.
  EXPECT_EQ( host_weight_storage[0], initial_storage_value );
  // The right guard entry must remain unchanged: no write past the output space.
  EXPECT_EQ( host_weight_storage[weight_storage_size - 1], initial_storage_value );
}

// Both generate() and mapReferenceQuadrature() reject an edge shorter than the minimum length.
// An edge exactly at the minimum is accepted, provided the other edge also meets the minimum.
TEST_P( EdgeQuadratureTest, applies_same_minimum_length_to_both_edges )
{
  constexpr RealT minimum_length = 1.0e-14;
  constexpr int point_count = 1;
  constexpr int operation_count = 2;
  struct LengthCase {
    const char* label;   ///< Description shown in assertion failures for this pair of edge lengths.
    RealT edge1_length;  ///< Length of the first edge.
    RealT edge2_length;  ///< Length of the second edge.
    bool valid;          ///< Whether both lengths meet the accepted minimum.
  };
  const LengthCase test_cases[] = {
      { "Edge1 shorter than the minimum", 0.5 * minimum_length, 1.0, false },
      { "Edge2 shorter than the minimum", 1.0, 0.5 * minimum_length, false },
      { "Both edges exactly at the minimum", minimum_length, minimum_length, true },
      { "Only edge1 exactly at the minimum", minimum_length, 1.0, true },
      { "Only edge2 exactly at the minimum", 1.0, minimum_length, true },
      { "Both edges longer than the minimum", 2.0 * minimum_length, 2.0 * minimum_length, true } };
  for ( const auto& test_case : test_cases ) {
    SCOPED_TRACE( testing::Message() << test_case.label << ", edge1 length=" << test_case.edge1_length
                                     << ", edge2 length=" << test_case.edge2_length );
    // Check each pair of lengths with two operations: [0] generate a complete rule, [1] map existing reference points.
    // Each results entry stores that operation's status and required point count.
    ArrayT<QuadratureResult> results( operation_count, operation_count, allocatorId() );
    const auto results_view = results.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      // Both edges extend rightward from x=0 and are separated vertically by 0.2. Edge2's endpoint order is reversed
      // so their normals oppose. Their lengths vary independently, but their projected intervals always overlap.
      const MdArray<RealT, 2, 2> coordinates1{ MdArray<RealT, 2>{ 0.0, 0.0 },
                                               MdArray<RealT, 2>{ test_case.edge1_length, 0.0 } };
      const MdArray<RealT, 2, 2> coordinates2{ MdArray<RealT, 2>{ test_case.edge2_length, 0.2 },
                                               MdArray<RealT, 2>{ 0.0, 0.2 } };
      // Disable endpoint smoothing to isolate the length check from smoothing effects on the overlap interval.
      const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<point_count>> rule{
          SmoothedProjectedEdgeMapping{ 0.0 } };
      // This path chooses the overlap interval and generates its quadrature points.
      results_view[0] = rule.generate( coordinates1.view(), coordinates2.view() ).result;
      // This path starts with one reference point on [-1,1] and maps it without choosing an overlap interval again.
      const auto reference = EdgeQuadratureRule<point_count>{}.generate();
      results_view[1] =
          rule.mapReferenceQuadrature( coordinates1.view(), coordinates2.view(), reference.getView() ).result;
    } );
    // Read both results on the host, including when the calculations ran on a GPU.
    const ArrayT<QuadratureResult, 1, MemorySpace::Host> host_results( results );
    for ( int operation = 0; operation < operation_count; ++operation ) {
      SCOPED_TRACE( operation == 0 ? "full point generation" : "mapping existing reference points" );
      const auto& result = host_results[operation];
      // Both operations must reject either edge below the minimum, while accepting lengths exactly at the minimum.
      EXPECT_EQ( result.status, test_case.valid ? QuadratureStatus::Success : QuadratureStatus::InvalidGeometry );
      // Valid pairs produce the selected one-point rule; invalid geometry produces no points.
      EXPECT_EQ( result.required_point_count, test_case.valid ? point_count : 0 );
    }
  }
}

// Map the same reference rule onto the first edge before and after doubling its length, without recomputing overlap.
// The supplied normalized coordinates stay unchanged; the physical weights must scale with the new edge length.
TEST_P( EdgeQuadratureTest, remaps_reference_points_after_first_edge_doubles_in_length )
{
  constexpr int point_count = 3;
  constexpr int measurement_count = 2;
  // Allocate two values to compare the first quadrature point before and after doubling the first edge's length:
  // slot 0: change in normalized edge coordinate; slot 1: new physical weight / original physical weight.
  ArrayT<RealT> records( measurement_count, measurement_count, allocatorId() );
  const auto records_view = records.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<point_count>> generator{};
    // Use [-1,1] in both calls. Although the second edge covers less of the lengthened first edge,
    // mapReferenceQuadrature() only maps supplied points; it does not select a new integration interval.
    const EdgeQuadratureInterval full_reference_interval{ MdArray<RealT, 2>{ -1.0, 1.0 }, true };
    const auto reference_points = EdgeQuadratureRule<point_count>{}.generate( full_reference_interval );
    // The first edge element lies at y=0 and doubles in length from 4 to 8. The second stays at y=0.2, ordered
    // right-to-left. Each array supplies the two endpoints of one edge element.
    const MdArray<RealT, 2, 2> original_edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
    const MdArray<RealT, 2, 2> lengthened_edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 8.0, 0.0 } };
    const MdArray<RealT, 2, 2> edge2{ MdArray<RealT, 2>{ 4.0, 0.2 }, MdArray<RealT, 2>{ 0.0, 0.2 } };
    const auto original_quadrature =
        generator.mapReferenceQuadrature( original_edge1.view(), edge2.view(), reference_points.getView() );
    const auto stretched_quadrature =
        generator.mapReferenceQuadrature( lengthened_edge1.view(), edge2.view(), reference_points.getView() );
    records_view[0] = stretched_quadrature.geometry1_parent_coordinates( 0, 0 ) -
                      original_quadrature.geometry1_parent_coordinates( 0, 0 );
    records_view[1] = stretched_quadrature.physical_weights[0] / original_quadrature.physical_weights[0];
  } );
  // Read the measurements on the host, including when the mapping ran on a GPU.
  const ArrayT<RealT, 1, MemorySpace::Host> host_records( records );
  // Reusing the reference rule must preserve its edge1 parent coordinate exactly.
  EXPECT_EQ( host_records[0], 0.0 );
  // Physical weight = reference weight * (first edge length / 2), so doubling that element's length doubles the weight.
  EXPECT_NEAR( host_records[1], 2.0, tolerance );
}

// Map reference points to edge1, then project their physical positions onto the tilted edge2 along edge2's normal.
TEST_P( EdgeQuadratureTest, maps_reference_points_to_both_tilted_edges )
{
  constexpr int point_count = 3;
  constexpr int column_count = 4;
  // One records row per point. Columns: 0: edge1 physical x; 1-2: edge2 physical (x,y); 3: edge2 parent coordinate.
  ArrayT<RealT, 2> records( axom::StackArray<axom::IndexType, 2>{ point_count, column_count }, allocatorId() );
  // The operation's status and point count are stored separately from the per-point coordinates.
  ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
  const auto records_view = records.view();
  const auto results_view = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    // Edge1 runs rightward along x=[0,4], y=0. Edge2 runs down-left from (3,2) to (1,0), on the line y=x-1.
    const MdArray<RealT, 2, 2> edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
    const MdArray<RealT, 2, 2> edge2{ MdArray<RealT, 2>{ 3.0, 2.0 }, MdArray<RealT, 2>{ 1.0, 0.0 } };
    const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<point_count>> generator{
        SmoothedProjectedEdgeMapping{ 0.0 } };
    // Supply points on the full edge1 reference interval [-1,1] to test mapping without computing overlap bounds.
    const auto reference_points = EdgeQuadratureRule<point_count>{}.generate();
    const auto mapped_quadrature =
        generator.mapReferenceQuadrature( edge1.view(), edge2.view(), reference_points.getView() );
    results_view[0] = mapped_quadrature.result;
    const auto points = mapped_quadrature.getView();
    for ( int q = 0; q < mapped_quadrature.result.required_point_count; ++q ) {
      const auto physical1 = LinearEdgeGeometry::position( edge1.view(), points.geometry1_parent_coordinates( q, 0 ) );
      const auto physical2 = LinearEdgeGeometry::position( edge2.view(), points.geometry2_parent_coordinates( q, 0 ) );
      records_view( q, 0 ) = physical1[0];
      records_view( q, 1 ) = physical2[0];
      records_view( q, 2 ) = physical2[1];
      records_view( q, 3 ) = points.geometry2_parent_coordinates( q, 0 );
    }
  } );
  // Copy coordinates and operation results to the host for assertions on every execution backend.
  const ArrayT<QuadratureResult, 1, MemorySpace::Host> host_results( results );
  const ArrayT<RealT, 2, MemorySpace::Host> host_records( records );
  // This pair has valid geometry and opposing normals, so mapping must succeed.
  ASSERT_EQ( host_results[0].status, QuadratureStatus::Success );
  // All three supplied reference points must be mapped.
  ASSERT_EQ( host_results[0].required_point_count, point_count );
  for ( int q = 0; q < point_count; ++q ) {
    SCOPED_TRACE( testing::Message() << "quadrature point=" << q );
    const RealT edge1_x = host_records( q, 0 );
    // Projection of (x,0) along edge2's normal (-1,1) intersects y=x-1 at ((x+1)/2,(x-1)/2).
    EXPECT_NEAR( host_records( q, 1 ), 0.5 * ( edge1_x + 1.0 ), tolerance );
    // The projected point's y coordinate comes from the same analytical line intersection.
    EXPECT_NEAR( host_records( q, 2 ), 0.5 * ( edge1_x - 1.0 ), tolerance );
    // Edge2's x coordinate is 2-xi2. Substituting (x+1)/2 gives its parent coordinate xi2=1.5-x/2.
    EXPECT_NEAR( host_records( q, 3 ), 1.5 - 0.5 * edge1_x, tolerance );
  }
}

// Reference weights are scaled by the Jacobian of the chosen integration surface. This mapping integrates on the
// first edge, so projecting evaluation points onto the second edge must not change that scaling.
TEST_P( EdgeQuadratureTest, uses_first_edge_length_for_physical_weights )
{
  constexpr int point_count = 3;
  constexpr int column_count = 3;
  // One records row per point. Columns: 0: edge1 physical x; 1: projected edge2 physical x; 2: physical weight.
  ArrayT<RealT, 2> records( axom::StackArray<axom::IndexType, 2>{ point_count, column_count }, allocatorId() );
  ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
  const auto records_view = records.view();
  const auto results_view = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    // Edge1 is horizontal with length 4. Edge2 runs from (3,2) to (1,0), with length sqrt(8) and line y=x-1.
    const MdArray<RealT, 2, 2> edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
    const MdArray<RealT, 2, 2> edge2{ MdArray<RealT, 2>{ 3.0, 2.0 }, MdArray<RealT, 2>{ 1.0, 0.0 } };
    const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<point_count>> generator{
        SmoothedProjectedEdgeMapping{ 0.0 } };
    // Integrate over all of edge1, using a supplied reference rule rather than selecting an overlap interval.
    const auto reference_points = EdgeQuadratureRule<point_count>{}.generate();
    const auto mapped_quadrature =
        generator.mapReferenceQuadrature( edge1.view(), edge2.view(), reference_points.getView() );
    results_view[0] = mapped_quadrature.result;
    const auto points = mapped_quadrature.getView();
    for ( int q = 0; q < mapped_quadrature.result.required_point_count; ++q ) {
      const auto physical1 = LinearEdgeGeometry::position( edge1.view(), points.geometry1_parent_coordinates( q, 0 ) );
      const auto physical2 = LinearEdgeGeometry::position( edge2.view(), points.geometry2_parent_coordinates( q, 0 ) );
      records_view( q, 0 ) = physical1[0];
      records_view( q, 1 ) = physical2[0];
      records_view( q, 2 ) = points.physical_weights[q];
    }
  } );
  // Read per-point data and the generation result on the host, including for GPU execution.
  const ArrayT<QuadratureResult, 1, MemorySpace::Host> host_results( results );
  const ArrayT<RealT, 2, MemorySpace::Host> host_records( records );
  // Only consume the point data if mapping succeeded.
  ASSERT_EQ( host_results[0].status, QuadratureStatus::Success );
  // The integration uses all three supplied Gauss points.
  ASSERT_EQ( host_results[0].required_point_count, point_count );
  RealT integrated_length = 0.0;
  RealT integrated_field_product = 0.0;
  for ( int q = 0; q < point_count; ++q ) {
    const RealT edge1_x = host_records( q, 0 );
    const RealT edge2_x = host_records( q, 1 );
    const RealT physical_weight = host_records( q, 2 );
    integrated_length += physical_weight;
    integrated_field_product += edge1_x * edge2_x * physical_weight;
  }
  // The integral of 1 over edge1 must equal its known length, 4, independently of edge2's length.
  EXPECT_NEAR( integrated_length, 4.0, tolerance );
  // On edge2, projection gives x2=(x1+1)/2. Thus the product of the two x fields integrates as
  // integral_0^4 x*(x+1)/2 dx = [x^3/6+x^2/4]_0^4 = 44/3, an analytical reference independent of quadrature weights.
  EXPECT_NEAR( integrated_field_product, 44.0 / 3.0, tolerance );
}

// Reuse edge1's reference points after moving edge2. Mapping must update edge2's parent coordinates even though the
// reference rule and edge1 have not changed.
TEST_P( EdgeQuadratureTest, remaps_reference_points_after_second_edge_moves )
{
  constexpr int point_count = 1;
  constexpr int measurement_count = 3;
  // records[0]: change in edge1 parent coordinate; [1]: change in edge2 parent coordinate; [2]: change in physical
  // weight.
  ArrayT<RealT> records( measurement_count, measurement_count, allocatorId() );
  const auto records_view = records.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    // Edge1 stays at x=[0,4], y=0. Edge2 stays horizontal at y=0.2 but moves one unit right, from x=[0,4] to [1,5].
    // Edge2 is ordered right-to-left so its normal opposes edge1's normal.
    const MdArray<RealT, 2, 2> edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
    const MdArray<RealT, 2, 2> original_edge2{ MdArray<RealT, 2>{ 4.0, 0.2 }, MdArray<RealT, 2>{ 0.0, 0.2 } };
    const MdArray<RealT, 2, 2> translated_edge2{ MdArray<RealT, 2>{ 5.0, 0.2 }, MdArray<RealT, 2>{ 1.0, 0.2 } };
    const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRule<point_count>> generator{};
    // Select the reference interval once, before edge2 moves, and reuse those points and weights in both calls.
    const auto reference_points = generator.generateReferenceQuadrature( edge1.view(), original_edge2.view() );
    const auto original_quadrature =
        generator.mapReferenceQuadrature( edge1.view(), original_edge2.view(), reference_points.getView() );
    const auto translated_quadrature =
        generator.mapReferenceQuadrature( edge1.view(), translated_edge2.view(), reference_points.getView() );
    records_view[0] = translated_quadrature.geometry1_parent_coordinates( 0, 0 ) -
                      original_quadrature.geometry1_parent_coordinates( 0, 0 );
    records_view[1] = translated_quadrature.geometry2_parent_coordinates( 0, 0 ) -
                      original_quadrature.geometry2_parent_coordinates( 0, 0 );
    records_view[2] = translated_quadrature.physical_weights[0] - original_quadrature.physical_weights[0];
  } );
  // Read the before/after differences on the host for assertions on every execution backend.
  const ArrayT<RealT, 1, MemorySpace::Host> host_records( records );
  // Edge1 and its supplied reference point did not move, so its parent coordinate must not change.
  EXPECT_EQ( host_records[0], 0.0 );
  // Edge2's reversed orientation gives an increase of 2*(translation/length)=2*(1/4)=0.5 in its parent coordinate.
  EXPECT_NEAR( host_records[1], 0.5, tolerance );
  // Physical weights depend on the unchanged reference weights and edge1 length, so moving edge2 must not change them.
  EXPECT_EQ( host_records[2], 0.0 );
}

/**
 * @brief Test-only mapping that stores physical x coordinates and weights for integration with respect to dx.
 *
 * Inputs are horizontal straight edges with edge1 ordered left-to-right and edge2 ordered right-to-left. The
 * integration interval is the overlap of their physical x ranges. Each integration point maps to a point on each edge
 * at the same x. This tests that PairQuadratureRule can use integration coordinates other than an element's [-1,1]
 * parent coordinate.
 */
struct ParallelEdgeMapping {
  using FirstGeometry = LinearEdgeGeometry;   ///< First straight-edge geometry.
  using SecondGeometry = LinearEdgeGeometry;  ///< Second straight-edge geometry.

  /**
   * @brief Find the common physical-x interval of the two horizontal edges.
   * @param edge1 Left-to-right first edge.
   * @param edge2 Right-to-left second edge.
   * @return Common interval in physical x; nonincreasing bounds give zero measure.
   */
  TRIBOL_HOST_DEVICE EdgeQuadratureInterval makeIntegrationSurface( const LinearEdgeGeometry& edge1,
                                                                    const LinearEdgeGeometry& edge2 ) const
  {
    // The overlap starts at the larger left endpoint and ends at the smaller right endpoint.
    const RealT left = edge1.origin()[0] > edge2.position( 1.0 )[0] ? edge1.origin()[0] : edge2.position( 1.0 )[0];
    const RealT right = edge1.position( 1.0 )[0] < edge2.origin()[0] ? edge1.position( 1.0 )[0] : edge2.origin()[0];
    return { MdArray<RealT, 2>{ left, right }, true };
  }

  /**
   * @brief Map physical-x integration points onto each horizontal edge.
   * @param edge1 First edge endpoints.
   * @param edge2 Second edge endpoints.
   * @param reference_points Points and weights expressed in physical x.
   * @param [out] paired_points Parent coordinates on both elements and physical weights.
   * @return Success with initialized count, or InsufficientCapacity with the full count and untouched output.
   */
  TRIBOL_HOST_DEVICE QuadratureResult mapToElements( const LinearEdgeGeometry& edge1, const LinearEdgeGeometry& edge2,
                                                     ReferenceQuadratureView<1> reference_points,
                                                     PairQuadratureView<1, 1> paired_points ) const
  {
    const IndexT point_count = reference_points.numberOfPoints();
    // Check capacity before writing any coordinate or weight.
    if ( paired_points.capacity() < point_count ) return { QuadratureStatus::InsufficientCapacity, point_count };
    for ( IndexT q = 0; q < point_count; ++q ) {
      // This mapping stores physical x in the field named parent_coordinates; y=0 is sufficient to find x on each
      // horizontal edge because parentCoordinate() projects the point orthogonally onto the edge's line.
      const RealT x = reference_points.parent_coordinates( q, 0 );
      const LinearEdgeGeometry::Point point{ x, 0.0 };
      paired_points.geometry1_parent_coordinates( q, 0 ) = edge1.parentCoordinate( point );
      paired_points.geometry2_parent_coordinates( q, 0 ) = edge2.parentCoordinate( point );
      // The base rule already scaled weights to the physical x interval; no edge-length factor is needed here.
      paired_points.physical_weights[q] = reference_points.reference_weights[q];
    }
    return { QuadratureStatus::Success, point_count };
  }
};

/** @brief Test-only rule that deliberately reports insufficient capacity instead of generating points. */
struct UndersizedEdgeQuadratureRule : EdgeQuadratureRule<1> {
  static constexpr IndexT required_points = points_per_base_rule + 1;  ///< Complete rule exceeds capacity by one point.

  /**
   * @brief Report the storage needed without exposing incomplete points.
   * @param interval Integration bounds; this test rule reports insufficient capacity for every interval.
   * @return Unpopulated reference quadrature with InsufficientCapacity and the full required count.
   */
  TRIBOL_HOST_DEVICE ReferenceQuadrature generate( [[maybe_unused]] EdgeQuadratureInterval interval ) const
  {
    ReferenceQuadrature output{};
    output.result = { QuadratureStatus::InsufficientCapacity, required_points };
    return output;
  }
};

// PairQuadratureRule must pass through a failure from the base rule, including the required count, without mapping
// incomplete points or writing the caller's output storage.
TEST_P( EdgeQuadratureTest, preserves_base_rule_failure_and_required_count )
{
  constexpr RealT initial_storage_value = -1.0;
  // Mark one output value so an unexpected write can be detected after the base rule reports its failure.
  ArrayT<RealT> point_storage( 1, 1, allocatorId() );
  point_storage.fill( initial_storage_value );
  // Store the status and required count returned by the pair generator.
  ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
  const auto point_storage_view = point_storage.view();
  const auto results_view = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    // Valid, parallel edges span x=[0,1] with opposing normals; geometry should not cause the failure.
    const MdArray<RealT, 2, 2> coordinates1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 1.0, 0.0 } };
    const MdArray<RealT, 2, 2> coordinates2{ MdArray<RealT, 2>{ 1.0, 0.2 }, MdArray<RealT, 2>{ 0.0, 0.2 } };
    // All output views have zero capacity and point to the marked value. The base rule's failure must be returned
    // before mapping needs these views. Its reported count comes from the base rule, not from this zero capacity.
    const PairQuadratureView<1, 1> zero_capacity_output{ { point_storage_view.data(), { 0, 1 } },
                                                         { point_storage_view.data(), { 0, 1 } },
                                                         { point_storage_view.data(), 0 } };
    // The test-only base rule has room for one point but always reports that it needs two; it produces no points.
    const PairQuadratureRule<SmoothedProjectedEdgeMapping, UndersizedEdgeQuadratureRule> rule{};
    results_view[0] = rule.generate( coordinates1.view(), coordinates2.view(), zero_capacity_output );
  } );
  // Read the operation result and marked storage on the host for assertions on every execution backend.
  const ArrayT<QuadratureResult, 1, MemorySpace::Host> host_results( results );
  const ArrayT<RealT, 1, MemorySpace::Host> host_point_storage( point_storage );
  // The pair generator must preserve the base rule's failure status.
  EXPECT_EQ( host_results[0].status, QuadratureStatus::InsufficientCapacity );
  // It must also preserve the base rule's requested count of two points.
  EXPECT_EQ( host_results[0].required_point_count, UndersizedEdgeQuadratureRule::required_points );
  // A failed base rule must not lead to a write to paired-point output.
  EXPECT_EQ( host_point_storage[0], initial_storage_value );
}

// Use the same Gauss rule with a mapping that integrates in physical x rather than edge1's parent coordinate.
TEST_P( EdgeQuadratureTest, composes_with_a_different_integration_mapping )
{
  constexpr int point_count = 1;
  constexpr int measurement_count = 3;
  // records[0]: edge1 parent coordinate; [1]: edge2 parent coordinate; [2]: physical integration weight.
  ArrayT<RealT> records( measurement_count, measurement_count, allocatorId() );
  ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
  const auto records_view = records.view();
  const auto results_view = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    // Edge1 spans x=[0,4] at y=0. Edge2 spans x=[1,3] at y=0.2, ordered right-to-left. Their x overlap is [1,3].
    const MdArray<RealT, 2, 2> edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
    const MdArray<RealT, 2, 2> edge2{ MdArray<RealT, 2>{ 3.0, 0.2 }, MdArray<RealT, 2>{ 1.0, 0.2 } };
    // ParallelEdgeMapping supplies physical bounds [1,3]; the one-point Gauss rule places its point at x=2.
    const PairQuadratureRule<ParallelEdgeMapping, EdgeQuadratureRule<point_count>> generator{};
    const auto generated_quadrature = generator.generate( edge1.view(), edge2.view() );
    results_view[0] = generated_quadrature.result;
    const auto points = generated_quadrature.getView();
    if ( generated_quadrature.result.status == QuadratureStatus::Success &&
         generated_quadrature.result.required_point_count == point_count ) {
      records_view[0] = points.geometry1_parent_coordinates( 0, 0 );
      records_view[1] = points.geometry2_parent_coordinates( 0, 0 );
      records_view[2] = points.physical_weights[0];
    }
  } );
  // Read the generated point and operation result on the host, including when generation ran on a GPU.
  const ArrayT<QuadratureResult, 1, MemorySpace::Host> host_results( results );
  const ArrayT<RealT, 1, MemorySpace::Host> host_records( records );
  // The alternative mapping must compose successfully with the existing Gauss rule.
  ASSERT_EQ( host_results[0].status, QuadratureStatus::Success );
  // The nonempty interval must produce the selected one-point rule.
  ASSERT_EQ( host_results[0].required_point_count, point_count );
  // Physical x=2 is the midpoint of edge1's [0,4] interval, hence edge1 parent coordinate zero.
  EXPECT_NEAR( host_records[0], 0.0, tolerance );
  // Physical x=2 is also the midpoint of edge2's reversed [3,1] interval, hence edge2 parent coordinate zero.
  EXPECT_NEAR( host_records[1], 0.0, tolerance );
  // The integral of 1 over physical x=[1,3] is 2; no extra edge1 or edge2 Jacobian may rescale that weight.
  EXPECT_NEAR( host_records[2], 2.0, tolerance );
}

INSTANTIATE_TEST_SUITE_P( execution, EdgeQuadratureTest, tribol::detail::enabledExecutionModes() );

// Requesting a polynomial degree must select the smallest Gauss rule that integrates all polynomials of that degree.
TEST( QuadratureRuleFactory, selects_registered_rule_by_polynomial_order )
{
  int creation_count = 0;
  // The factory passes its selected rule type to this callback. Returning its point count identifies that type
  // without generating any quadrature points; the counter checks how often the callback is invoked.
  const auto identify_selected_rule = [&]<typename QuadratureRule>() {
    ++creation_count;
    return QuadratureRule::points_per_base_rule;
  };
  // Each case is {requested polynomial degree, expected point count}. N-point Gauss rules are exact through 2*N-1,
  // so degrees 0-1 need one point, 2-3 need two, and 4-5 need three.
  const std::pair<int, int> test_cases[] = { { 0, 1 }, { 1, 1 }, { 2, 2 }, { 3, 2 }, { 4, 3 }, { 5, 3 } };
  for ( const auto& [requested_order, expected_point_count] : test_cases ) {
    SCOPED_TRACE( testing::Message() << "requested polynomial degree=" << requested_order );
    const int previous_creation_count = creation_count;
    const auto selected_point_count = createWithQuadratureRule( requested_order, identify_selected_rule );
    // Compare the selected type's point count with the independently specified degree-to-point table above.
    EXPECT_EQ( selected_point_count, expected_point_count );
    // One valid request must invoke the creation callback exactly once.
    EXPECT_EQ( creation_count, previous_creation_count + 1 );
  }
}

// Unsupported degrees must be rejected before the factory asks the caller to create a rule.
TEST( QuadratureRuleFactory, rejects_unsupported_orders_without_creating_a_rule )
{
  int creation_count = 0;
  const auto identify_selected_rule = [&]<typename QuadratureRule>() {
    ++creation_count;
    return QuadratureRule::points_per_base_rule;
  };
  // The registered rules support degrees 0 through 5. Check one request below and one above that range.
  for ( const int requested_order : { -1, 6 } ) {
    SCOPED_TRACE( testing::Message() << "unsupported polynomial degree=" << requested_order );
    const int previous_creation_count = creation_count;
    // Invalid requests must report an argument error instead of silently selecting another degree.
    EXPECT_THROW( createWithQuadratureRule( requested_order, identify_selected_rule ), std::invalid_argument );
    // Rejection must happen before any creation callback runs.
    EXPECT_EQ( creation_count, previous_creation_count );
  }
}

/**
 * @brief Test-only rule descriptions used to check factory selection; these types generate no points or weights.
 *
 * @tparam Degree Exactly integrated reference polynomial degree.
 * @tparam Points Points in the synthetic base rule.
 */
template <int Degree, int Points>
struct TestQuadratureRuleMetadata {
  static constexpr int polynomial_order = Degree;      ///< Declared exact polynomial degree.
  static constexpr int points_per_base_rule = Points;  ///< Declared points per base rule.
};

/**
 * @brief Check factory selection using custom rule descriptions with gaps between supported polynomial degrees.
 * These types provide only degree and point-count metadata; they do not implement quadrature generation.
 */
TEST( QuadratureRuleFactory, selects_custom_rule_types_by_polynomial_degree )
{
  using DegreeOneRule = TestQuadratureRuleMetadata<1, 1>;
  using DegreeTwoRule = TestQuadratureRuleMetadata<2, 3>;
  using DegreeFourRule = TestQuadratureRuleMetadata<4, 6>;
  // Return the selected type's point count so the assertion can identify which description the factory chose.
  const auto identify_selected_rule = []<typename QuadratureRule>() { return QuadratureRule::points_per_base_rule; };
  // Each case is {requested degree, expected point count}. There is no degree-3 rule in this test family, so a
  // degree-3 request must select the degree-4 description (six points), not assume the edge-rule point-count formula.
  const std::pair<int, int> test_cases[] = { { 0, 1 }, { 1, 1 }, { 2, 3 }, { 3, 6 }, { 4, 6 } };
  for ( const auto& [requested_order, expected_point_count] : test_cases ) {
    SCOPED_TRACE( testing::Message() << "requested polynomial degree=" << requested_order );
    const auto selected_point_count = createWithQuadratureRule<DegreeOneRule, DegreeTwoRule, DegreeFourRule>(
        requested_order, identify_selected_rule );
    // The selected description must be the first in this family that meets or exceeds the requested degree.
    EXPECT_EQ( selected_point_count, expected_point_count );
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
