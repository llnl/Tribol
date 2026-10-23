// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <array>

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/physics/InterfacePairChecks.hpp"

namespace {

constexpr tribol::IndexT shared_mesh_id = 11;
constexpr tribol::IndexT opposing_mesh_id = 12;
constexpr tribol::RealT comparison_tolerance = 1.0e-14;

TEST( InterfacePairChecks, identifies_same_and_topologically_adjacent_elements )
{
  // Auto-contact must reject an element paired with itself and distinct elements that meet at any finite-element node.
  constexpr std::array<tribol::IndexT, 6> connectivity{ 0, 1, 1, 2, 3, 4 };
  constexpr std::array<tribol::RealT, 5> x{ 0.0, 1.0, 2.0, 3.0, 4.0 };
  constexpr std::array<tribol::RealT, 5> y{ 0.0, 0.0, 0.0, 0.0, 0.0 };
  tribol::MeshData mesh( shared_mesh_id, 3, x.size(), connectivity.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE,
                         x.data(), y.data(), nullptr, tribol::MemorySpace::Host );
  const auto mesh_view = mesh.getView();

  EXPECT_TRUE( tribol::InterfacePairChecks::IsSameElementPair( mesh_view, 0, mesh_view, 0 ) );
  EXPECT_FALSE( tribol::InterfacePairChecks::IsSameElementPair( mesh_view, 0, mesh_view, 1 ) );
  EXPECT_TRUE( tribol::InterfacePairChecks::ElementsShareNode( mesh_view, 0, mesh_view, 1 ) );
  EXPECT_FALSE( tribol::InterfacePairChecks::ElementsShareNode( mesh_view, 0, mesh_view, 2 ) );
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
      tribol::InterfacePairChecks::LinearEdgeNormalsAreOpposing( lower.getView(), 0, upper_reversed.getView(), 0 ) );
  EXPECT_FALSE(
      tribol::InterfacePairChecks::LinearEdgeNormalsAreOpposing( lower.getView(), 0, upper_forward.getView(), 0 ) );
  EXPECT_FALSE(
      tribol::InterfacePairChecks::LinearEdgeNormalsAreOpposing( lower.getView(), 0, degenerate.getView(), 0 ) );
}

TEST( InterfacePairChecks, computes_normalized_projection_intervals )
{
  // These intervals cover partial overlap, containment, complete separation, unequal lengths, and asymmetric placement.
  const tribol::RealT integration_point0[2]{ 0.0, 0.0 };
  const tribol::RealT integration_point1[2]{ 2.0, 0.0 };

  const auto expect_projection = [&]( const std::array<tribol::RealT, 2>& projected_point0,
                                      const std::array<tribol::RealT, 2>& projected_point1,
                                      const std::array<tribol::RealT, 2>& expected_bounds ) {
    tribol::RealT projection_bounds[2];
    ASSERT_TRUE( tribol::InterfacePairChecks::ComputeLinearEdgeProjectionBounds(
        integration_point0, integration_point1, projected_point0.data(), projected_point1.data(),
        tribol::MdSpan<tribol::RealT, 2>{ projection_bounds } ) );
    EXPECT_NEAR( projection_bounds[0], expected_bounds[0], comparison_tolerance );
    EXPECT_NEAR( projection_bounds[1], expected_bounds[1], comparison_tolerance );
  };

  expect_projection( { 1.0, 0.2 }, { 3.0, 0.2 }, { 0.0, 2.0 } );
  expect_projection( { 0.5, 0.2 }, { 1.5, 0.2 }, { -0.5, 0.5 } );
  expect_projection( { 3.0, 0.2 }, { 4.0, 0.2 }, { 2.0, 3.0 } );
  expect_projection( { -1.0, 0.2 }, { 4.0, 0.2 }, { -2.0, 3.0 } );
  expect_projection( { -0.5, 0.2 }, { 0.25, 0.2 }, { -1.5, -0.75 } );
}

TEST( InterfacePairChecks, rejects_degenerate_projection_edges )
{
  // A collapsed source or target edge cannot define the projection used to construct an integration interval.
  const tribol::RealT valid_point0[2]{ 0.0, 0.0 };
  const tribol::RealT valid_point1[2]{ 1.0, 0.0 };
  const tribol::RealT collapsed_point[2]{ 0.5, 0.1 };
  tribol::RealT projection_bounds[2]{ 3.0, 4.0 };

  EXPECT_FALSE( tribol::InterfacePairChecks::ComputeLinearEdgeProjectionBounds(
      valid_point0, valid_point1, collapsed_point, collapsed_point,
      tribol::MdSpan<tribol::RealT, 2>{ projection_bounds } ) );
  EXPECT_EQ( projection_bounds[0], 0.0 );
  EXPECT_EQ( projection_bounds[1], 0.0 );

  projection_bounds[0] = 3.0;
  projection_bounds[1] = 4.0;
  EXPECT_FALSE( tribol::InterfacePairChecks::ComputeLinearEdgeProjectionBounds(
      collapsed_point, collapsed_point, valid_point0, valid_point1,
      tribol::MdSpan<tribol::RealT, 2>{ projection_bounds } ) );
  EXPECT_EQ( projection_bounds[0], 0.0 );
  EXPECT_EQ( projection_bounds[1], 0.0 );
}

TEST( InterfacePairChecks, requires_positive_overlap_with_smoothed_support )
{
  // A smoothing fraction of 0.1 extends normalized support from [-1, 1] to [-1.2, 1.2] without accepting endpoint
  // touch.
  constexpr tribol::RealT smoothing_fraction = 0.1;
  const tribol::RealT ordinary_overlap[2]{ 0.5, 1.5 };
  const tribol::RealT smoothing_only_overlap[2]{ 1.1, 1.15 };
  const tribol::RealT separated_interval[2]{ 1.3, 2.0 };
  const tribol::RealT zero_measure_touch[2]{ 1.2, 2.0 };

  EXPECT_TRUE(
      tribol::InterfacePairChecks::ProjectionIntervalOverlapsSmoothedSupport( ordinary_overlap, smoothing_fraction ) );
  EXPECT_TRUE( tribol::InterfacePairChecks::ProjectionIntervalOverlapsSmoothedSupport( smoothing_only_overlap,
                                                                                       smoothing_fraction ) );
  EXPECT_FALSE( tribol::InterfacePairChecks::ProjectionIntervalOverlapsSmoothedSupport( separated_interval,
                                                                                        smoothing_fraction ) );
  EXPECT_FALSE( tribol::InterfacePairChecks::ProjectionIntervalOverlapsSmoothedSupport( zero_measure_touch,
                                                                                        smoothing_fraction ) );
}

TEST( InterfacePairChecks, matches_energy_mortar_flipped_integration_edge )
{
  // Tribol stores mesh 1 first, while EnergyMortar integrates on mesh 2. The mesh overload must match the raw endpoint
  // calculation after those roles are reversed.
  constexpr std::array<tribol::IndexT, 2> connectivity{ 0, 1 };
  constexpr std::array<tribol::RealT, 2> projected_x{ 0.5, 1.5 };
  constexpr std::array<tribol::RealT, 2> projected_y{ 0.2, 0.2 };
  constexpr std::array<tribol::RealT, 2> integration_x{ 0.0, 2.0 };
  constexpr std::array<tribol::RealT, 2> integration_y{ 0.0, 0.0 };
  tribol::MeshData projected_mesh( shared_mesh_id, 1, projected_x.size(), connectivity.data(), tribol::EDGE,
                                   tribol::LINEAR_LAGRANGE, projected_x.data(), projected_y.data(), nullptr,
                                   tribol::MemorySpace::Host );
  tribol::MeshData integration_mesh( opposing_mesh_id, 1, integration_x.size(), connectivity.data(), tribol::EDGE,
                                     tribol::LINEAR_LAGRANGE, integration_x.data(), integration_y.data(), nullptr,
                                     tribol::MemorySpace::Host );

  tribol::RealT mesh_bounds[2];
  tribol::RealT endpoint_bounds[2];
  const tribol::RealT integration_point0[2]{ 0.0, 0.0 };
  const tribol::RealT integration_point1[2]{ 2.0, 0.0 };
  const tribol::RealT projected_point0[2]{ 0.5, 0.2 };
  const tribol::RealT projected_point1[2]{ 1.5, 0.2 };
  ASSERT_TRUE( tribol::InterfacePairChecks::ComputeLinearEdgeProjectionBounds(
      integration_mesh.getView(), 0, projected_mesh.getView(), 0, mesh_bounds ) );
  ASSERT_TRUE( tribol::InterfacePairChecks::ComputeLinearEdgeProjectionBounds(
      integration_point0, integration_point1, projected_point0, projected_point1,
      tribol::MdSpan<tribol::RealT, 2>{ endpoint_bounds } ) );

  EXPECT_NEAR( mesh_bounds[0], -0.5, comparison_tolerance );
  EXPECT_NEAR( mesh_bounds[1], 0.5, comparison_tolerance );
  EXPECT_NEAR( mesh_bounds[0], endpoint_bounds[0], comparison_tolerance );
  EXPECT_NEAR( mesh_bounds[1], endpoint_bounds[1], comparison_tolerance );
}

#if defined( TRIBOL_USE_CUDA ) || defined( TRIBOL_USE_HIP )
TEST( InterfacePairChecks, evaluates_projection_support_on_device )
{
  // Projection and support checks must compile and execute in the same device loop used to test coarse candidates.
  const int host_allocator_id = tribol::getResourceAllocatorID( tribol::MemorySpace::Host );
  const int device_allocator_id = tribol::getResourceAllocatorID( tribol::MemorySpace::Device );
  tribol::ArrayT<tribol::RealT> host_coordinates( { 0.0, 0.0, 2.0, 0.0, 0.5, 0.2, 1.5, 0.2 }, host_allocator_id );
  tribol::ArrayT<tribol::RealT> device_coordinates( host_coordinates, device_allocator_id );
  tribol::ArrayT<tribol::RealT> device_results( 3, 3, device_allocator_id );
  auto coordinates = device_coordinates.view();
  auto results = device_results.view();

#if defined( TRIBOL_USE_CUDA )
  constexpr tribol::ExecutionMode device_execution_mode = tribol::ExecutionMode::Cuda;
#else
  constexpr tribol::ExecutionMode device_execution_mode = tribol::ExecutionMode::Hip;
#endif
  tribol::forAllExec( device_execution_mode, 1, [=] TRIBOL_HOST_DEVICE( tribol::IndexT ) {
    tribol::RealT projection_bounds[2];
    const bool has_projection = tribol::InterfacePairChecks::ComputeLinearEdgeProjectionBounds(
        coordinates.data(), coordinates.data() + 2, coordinates.data() + 4, coordinates.data() + 6,
        tribol::MdSpan<tribol::RealT, 2>{ projection_bounds } );
    results[0] = projection_bounds[0];
    results[1] = projection_bounds[1];
    results[2] = has_projection &&
                 tribol::InterfacePairChecks::ProjectionIntervalOverlapsSmoothedSupport( projection_bounds, 0.0 );
  } );

  tribol::ArrayT<tribol::RealT, 1, tribol::MemorySpace::Host> host_results( device_results );
  EXPECT_NEAR( host_results[0], -0.5, comparison_tolerance );
  EXPECT_NEAR( host_results[1], 0.5, comparison_tolerance );
  EXPECT_EQ( host_results[2], 1.0 );
}
#endif

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
