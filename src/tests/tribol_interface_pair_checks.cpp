// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Test individual element-pair checks for shared nodes, edge normals, projection bounds, and support overlap.
 *
 * Geometry cases run directly on the host. Projection and support checks also run inside a loop on every enabled
 * execution backend. Filtering whole candidate arrays is covered in tribol_refined_search.cpp.
 */

#include <array>

#include "ExecutionTest.hpp"

#include "axom/slic/core/SimpleLogger.hpp"

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/search/InterfacePairChecks.hpp"

namespace {

constexpr tribol::IndexT shared_mesh_id = 11;
constexpr tribol::IndexT opposing_mesh_id = 12;
constexpr tribol::IndexT nodes_per_edge = 2;
constexpr tribol::RealT edge_gap = 0.2;
constexpr tribol::RealT comparison_tolerance = 1.0e-14;

TEST( InterfacePairChecks, identifies_same_and_topologically_adjacent_elements )
{
  // Auto-contact must reject an element paired with itself and distinct elements that meet at any finite-element node.
  constexpr std::array<tribol::IndexT, 6> connectivity{ 0, 1, 1, 2, 3, 4 };
  constexpr std::array<tribol::RealT, 5> x{ 0.0, 1.0, 2.0, 3.0, 4.0 };
  constexpr std::array<tribol::RealT, 5> y{ 0.0, 0.0, 0.0, 0.0, 0.0 };
  tribol::MeshData mesh( shared_mesh_id, connectivity.size() / nodes_per_edge, x.size(), connectivity.data(),
                         tribol::EDGE, tribol::LINEAR_LAGRANGE, x.data(), y.data(), nullptr,
                         tribol::MemorySpace::Host );
  const auto mesh_view = mesh.getView();

  EXPECT_TRUE( tribol::InterfacePairChecks::isSameElementPair( mesh_view, 0, mesh_view, 0 ) );
  EXPECT_FALSE( tribol::InterfacePairChecks::isSameElementPair( mesh_view, 0, mesh_view, 1 ) );
  EXPECT_TRUE( tribol::InterfacePairChecks::elementsShareNode( mesh_view, 0, mesh_view, 1 ) );
  EXPECT_FALSE( tribol::InterfacePairChecks::elementsShareNode( mesh_view, 0, mesh_view, 2 ) );
}

TEST( InterfacePairChecks, requires_opposing_nondegenerate_edge_normals )
{
  // Endpoint ordering defines each edge normal, so reversing the upper edge makes the two valid edges oppose one
  // another.
  constexpr std::array<tribol::IndexT, 2> connectivity{ 0, 1 };
  constexpr std::array<tribol::RealT, 2> lower_x{ 0.0, 1.0 };
  constexpr std::array<tribol::RealT, 2> lower_y{ 0.0, 0.0 };
  constexpr std::array<tribol::RealT, 2> upper_reversed_x{ 1.0, 0.0 };
  constexpr std::array<tribol::RealT, 2> upper_y{ 0.1, 0.1 };
  constexpr std::array<tribol::RealT, 2> upper_forward_x{ 0.0, 1.0 };
  constexpr std::array<tribol::RealT, 2> degenerate_x{ 0.0, 0.0 };
  constexpr std::array<tribol::RealT, 2> degenerate_y{ 0.0, 0.0 };

  tribol::MeshData lower( shared_mesh_id, 1, lower_x.size(), connectivity.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE,
                          lower_x.data(), lower_y.data(), nullptr, tribol::MemorySpace::Host );
  tribol::MeshData upper_reversed( opposing_mesh_id, 1, upper_reversed_x.size(), connectivity.data(), tribol::EDGE,
                                   tribol::LINEAR_LAGRANGE, upper_reversed_x.data(), upper_y.data(), nullptr,
                                   tribol::MemorySpace::Host );
  tribol::MeshData upper_forward( opposing_mesh_id, 1, upper_forward_x.size(), connectivity.data(), tribol::EDGE,
                                  tribol::LINEAR_LAGRANGE, upper_forward_x.data(), upper_y.data(), nullptr,
                                  tribol::MemorySpace::Host );
  tribol::MeshData degenerate( opposing_mesh_id, 1, degenerate_x.size(), connectivity.data(), tribol::EDGE,
                               tribol::LINEAR_LAGRANGE, degenerate_x.data(), degenerate_y.data(), nullptr,
                               tribol::MemorySpace::Host );

  EXPECT_TRUE(
      tribol::InterfacePairChecks::linearEdgeNormalsAreOpposing( lower.getView(), 0, upper_reversed.getView(), 0 ) );
  EXPECT_FALSE(
      tribol::InterfacePairChecks::linearEdgeNormalsAreOpposing( lower.getView(), 0, upper_forward.getView(), 0 ) );
  EXPECT_FALSE(
      tribol::InterfacePairChecks::linearEdgeNormalsAreOpposing( lower.getView(), 0, degenerate.getView(), 0 ) );
}

// Check normalized projection bounds for overlapping, contained, separated, and covering edges.
TEST( InterfacePairChecks, computes_normalized_projection_intervals )
{
  // These are physical (x, y) edge endpoints, not quadrature points. The integration edge receives the projection.
  // On this edge, x=0 maps to -1 and x=2 maps to +1, so the normalized coordinate is x-1.
  const tribol::RealT integration_edge_start[2]{ 0.0, 0.0 };
  const tribol::RealT integration_edge_end[2]{ 2.0, 0.0 };
  // Each projected edge is horizontal at y=edge_gap. Projection along its normal moves its endpoints vertically.
  struct ProjectionCase {
    const char* description;
    std::array<tribol::RealT, 2> projected_edge_start;
    std::array<tribol::RealT, 2> projected_edge_end;
    std::array<tribol::RealT, 2> expected_normalized_bounds;
  };

  const std::array cases{
      ProjectionCase{ .description = "Right overlap: keep the extension beyond the right endpoint",
                      .projected_edge_start = { 1.0, edge_gap },
                      .projected_edge_end = { 3.0, edge_gap },
                      .expected_normalized_bounds = { 0.0, 2.0 } },
      ProjectionCase{
          .description = "Full containment: the shorter projected edge is centered inside the integration edge",
          .projected_edge_start = { 0.5, edge_gap },
          .projected_edge_end = { 1.5, edge_gap },
          .expected_normalized_bounds = { -0.5, 0.5 } },
      ProjectionCase{
          // This helper only computes projection geometry. Refined search retains separated intervals, and
          // EnergyMortar later assigns them zero integration measure.
          .description = "Complete separation: projection is valid even when both bounds lie beyond the edge",
          .projected_edge_start = { 3.0, edge_gap },
          .projected_edge_end = { 4.0, edge_gap },
          .expected_normalized_bounds = { 2.0, 3.0 } },
      ProjectionCase{ .description = "Full coverage: keep the extensions beyond both endpoints",
                      .projected_edge_start = { -1.0, edge_gap },
                      .projected_edge_end = { 4.0, edge_gap },
                      .expected_normalized_bounds = { -2.0, 3.0 } },
      ProjectionCase{
          .description = "Asymmetric left overlap: most of the projected edge lies outside the integration edge",
          .projected_edge_start = { -0.5, edge_gap },
          .projected_edge_end = { 0.25, edge_gap },
          .expected_normalized_bounds = { -1.5, -0.75 } } };

  for ( const auto& test_case : cases ) {
    SCOPED_TRACE( test_case.description );
    const auto projection = tribol::InterfacePairChecks::computeLinearEdgeProjectionBounds(
        integration_edge_start, integration_edge_end, test_case.projected_edge_start.data(),
        test_case.projected_edge_end.data() );

    ASSERT_TRUE( projection.valid );
    EXPECT_NEAR( projection.bounds[0], test_case.expected_normalized_bounds[0], comparison_tolerance );
    EXPECT_NEAR( projection.bounds[1], test_case.expected_normalized_bounds[1], comparison_tolerance );
  }
}

TEST( InterfacePairChecks, rejects_degenerate_projection_edges )
{
  // A collapsed edge returns an invalid result with zero bounds. EnergyMortar uses these bounds as a zero-length
  // integration interval, so the result must contain defined values even when validity is ignored.
  const tribol::RealT valid_point0[2]{ 0.0, 0.0 };
  const tribol::RealT valid_point1[2]{ 1.0, 0.0 };
  const tribol::RealT collapsed_point[2]{ 0.5, 0.1 };
  const auto collapsed_projected = tribol::InterfacePairChecks::computeLinearEdgeProjectionBounds(
      valid_point0, valid_point1, collapsed_point, collapsed_point );
  EXPECT_FALSE( collapsed_projected.valid );
  EXPECT_EQ( collapsed_projected.bounds[0], 0.0 );
  EXPECT_EQ( collapsed_projected.bounds[1], 0.0 );

  const auto collapsed_integration = tribol::InterfacePairChecks::computeLinearEdgeProjectionBounds(
      collapsed_point, collapsed_point, valid_point0, valid_point1 );
  EXPECT_FALSE( collapsed_integration.valid );
  EXPECT_EQ( collapsed_integration.bounds[0], 0.0 );
  EXPECT_EQ( collapsed_integration.bounds[1], 0.0 );
}

TEST( InterfacePairChecks, requires_positive_overlap_with_smoothed_support )
{
  // A smoothing fraction of 0.1 extends normalized support from [-1, 1] to [-1.2, 1.2] without accepting endpoint
  // touch.
  constexpr tribol::RealT smoothing_fraction = 0.1;
  const tribol::RealT ordinary_overlap[2]{ 0.5, 1.5 };
  // Normalized coordinates span [-1, 1], so the edge-length fraction doubles in this coordinate system.
  constexpr tribol::RealT support_end = 1.0 + 2.0 * smoothing_fraction;
  const tribol::RealT smoothing_only_overlap[2]{ 1.0 + smoothing_fraction, support_end - 0.5 * smoothing_fraction };
  const tribol::RealT separated_interval[2]{ support_end + smoothing_fraction, 2.0 };
  const tribol::RealT zero_measure_touch[2]{ support_end, 2.0 };

  EXPECT_TRUE(
      tribol::InterfacePairChecks::projectionIntervalOverlapsSmoothedSupport( ordinary_overlap, smoothing_fraction ) );
  EXPECT_TRUE( tribol::InterfacePairChecks::projectionIntervalOverlapsSmoothedSupport( smoothing_only_overlap,
                                                                                       smoothing_fraction ) );
  EXPECT_FALSE( tribol::InterfacePairChecks::projectionIntervalOverlapsSmoothedSupport( separated_interval,
                                                                                        smoothing_fraction ) );
  EXPECT_FALSE( tribol::InterfacePairChecks::projectionIntervalOverlapsSmoothedSupport( zero_measure_touch,
                                                                                        smoothing_fraction ) );
}

// Check that the mesh overload uses mesh 1 (non-mortar) as the integration edge, matching ENERGY_MORTAR.
TEST( InterfacePairChecks, matches_energy_mortar_integration_edge )
{
  constexpr tribol::IndexT nonmortar_mesh_id = 0;
  constexpr tribol::IndexT mortar_mesh_id = 1;
  constexpr std::array<tribol::IndexT, 2> connectivity{ 0, 1 };

  // The non-mortar edge runs from (0, 0) to (2, 0). Its normalized coordinate maps x=0 to -1 and x=2 to +1.
  constexpr std::array<tribol::RealT, 2> nonmortar_x{ 0.0, 2.0 };
  constexpr std::array<tribol::RealT, 2> nonmortar_y{ 0.0, 0.0 };
  tribol::MeshData nonmortar_mesh( nonmortar_mesh_id, 1, nonmortar_x.size(), connectivity.data(), tribol::EDGE,
                                   tribol::LINEAR_LAGRANGE, nonmortar_x.data(), nonmortar_y.data(), nullptr,
                                   tribol::MemorySpace::Host );

  // The mortar edge runs from (0.5, 0.2) to (1.5, 0.2). Projection onto the non-mortar edge gives bounds [-0.5, 0.5].
  constexpr std::array<tribol::RealT, 2> mortar_x{ 0.5, 1.5 };
  constexpr std::array<tribol::RealT, 2> mortar_y{ 0.2, 0.2 };
  tribol::MeshData mortar_mesh( mortar_mesh_id, 1, mortar_x.size(), connectivity.data(), tribol::EDGE,
                                tribol::LINEAR_LAGRANGE, mortar_x.data(), mortar_y.data(), nullptr,
                                tribol::MemorySpace::Host );
  constexpr tribol::RealT expected_normalized_lower_bound = -0.5;
  constexpr tribol::RealT expected_normalized_upper_bound = 0.5;

  const auto projection = tribol::InterfacePairChecks::computeLinearEdgeProjectionBounds( nonmortar_mesh.getView(), 0,
                                                                                          mortar_mesh.getView(), 0 );

  ASSERT_TRUE( projection.valid );
  EXPECT_NEAR( projection.bounds[0], expected_normalized_lower_bound, comparison_tolerance );
  EXPECT_NEAR( projection.bounds[1], expected_normalized_upper_bound, comparison_tolerance );
}

class InterfacePairChecksExecutionTest : public tribol::detail::ExecutionTest {};

// Check projection bounds and support overlap inside a loop on each enabled CPU or GPU backend.
TEST_P( InterfacePairChecksExecutionTest, evaluates_projection_support )
{
  constexpr tribol::IndexT spatial_dimension = 2;
  constexpr tribol::IndexT projected_edge_offset = nodes_per_edge * spatial_dimension;
  constexpr tribol::IndexT projection_bound_count = 2;
  constexpr tribol::IndexT overlap_result_index = projection_bound_count;
  constexpr tribol::IndexT result_count = projection_bound_count + 1;
  constexpr tribol::RealT no_smoothing = 0.0;
  // A centered edge covering half of [0, 2] projects to [-0.5, 0.5] in normalized coordinates.
  const auto input_coordinates =
      copyToExecutionSpace<tribol::RealT>( { 0.0, 0.0, 2.0, 0.0, 0.5, edge_gap, 1.5, edge_gap } );
  tribol::ArrayT<tribol::RealT> output( result_count, result_count, allocatorId() );
  const auto coordinates = input_coordinates.view();
  auto results = output.view();

  tribol::forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( tribol::IndexT ) {
    const auto projection = tribol::InterfacePairChecks::computeLinearEdgeProjectionBounds(
        coordinates.data(), coordinates.data() + spatial_dimension, coordinates.data() + projected_edge_offset,
        coordinates.data() + projected_edge_offset + spatial_dimension );
    results[0] = projection.bounds[0];
    results[1] = projection.bounds[1];
    results[overlap_result_index] =
        projection.valid && tribol::InterfacePairChecks::projectionIntervalOverlapsSmoothedSupport(
                                projection.bounds.view(), no_smoothing );
  } );

  tribol::ArrayT<tribol::RealT, 1, tribol::MemorySpace::Host> host_results( output );
  EXPECT_NEAR( host_results[0], -0.5, comparison_tolerance );
  EXPECT_NEAR( host_results[1], 0.5, comparison_tolerance );
  EXPECT_EQ( host_results[overlap_result_index], 1.0 );
}

INSTANTIATE_TEST_SUITE_P( execution, InterfacePairChecksExecutionTest, tribol::detail::enabledExecutionModes() );

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
