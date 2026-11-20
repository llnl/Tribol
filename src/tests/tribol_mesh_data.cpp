// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <array>

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"

#include "tribol/geom/ElementNormal.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace {

constexpr tribol::IndexT mesh_id = 17;
constexpr tribol::IndexT one_element = 1;
constexpr tribol::IndexT first_element_id = 0;
constexpr tribol::IndexT linear_edge_node_count = 2;
constexpr tribol::IndexT quadratic_edge_node_count = 3;
constexpr tribol::IndexT quadratic_quad_node_count = 9;
constexpr int edge_dim = 2;
constexpr int quad_dim = 3;
constexpr int x_component = 0;
constexpr int y_component = 1;
constexpr int z_component = 2;
constexpr bool higher_order_face_data_is_supported = false;
constexpr bool derived_face_data_is_populated = false;
using LinearEdgeCoordinates = tribol::MdArray<tribol::RealT, linear_edge_node_count, edge_dim>;

void ExpectCoordinatesEqual( const LinearEdgeCoordinates& actual, const LinearEdgeCoordinates& expected )
{
  for ( tribol::IndexT node{}; node < linear_edge_node_count; ++node ) {
    for ( int component{}; component < edge_dim; ++component ) {
      EXPECT_DOUBLE_EQ( actual( node, component ), expected( node, component ) );
    }
  }
}

void ExpectDerivedFaceDataIsEmpty( tribol::MeshData& mesh )
{
  auto mesh_view = mesh.getView();
  EXPECT_EQ( mesh_view.hasElementCentroids(), derived_face_data_is_populated );
  EXPECT_EQ( mesh_view.hasElementNormals(), derived_face_data_is_populated );
  EXPECT_EQ( mesh_view.hasFaceRadius(), derived_face_data_is_populated );
  EXPECT_EQ( mesh_view.hasElementAreas(), derived_face_data_is_populated );
}

TEST( MeshData, stores_linear_edge_data )
{
  constexpr std::array<tribol::IndexT, linear_edge_node_count> connectivity{ 0, 1 };
  constexpr std::array<tribol::RealT, linear_edge_node_count> x{ 0.0, 1.0 };
  constexpr std::array<tribol::RealT, linear_edge_node_count> y{ 0.0, 0.0 };

  tribol::MeshData mesh( mesh_id, one_element, linear_edge_node_count, connectivity.data(), tribol::EDGE,
                         tribol::LINEAR_LAGRANGE, x.data(), y.data(), nullptr, tribol::MemorySpace::Host );
  auto mesh_view = mesh.getView();

  EXPECT_EQ( mesh.getElementType(), tribol::EDGE );
  EXPECT_EQ( mesh.getFaceOrder(), tribol::LINEAR_LAGRANGE );
  EXPECT_EQ( mesh.numberOfNodesPerElement(), linear_edge_node_count );
  EXPECT_EQ( mesh_view.getFaceOrder(), tribol::LINEAR_LAGRANGE );
  EXPECT_EQ( mesh_view.numberOfNodesPerElement(), linear_edge_node_count );
}

// This test ensures getFaceCoords() follows each element's local connectivity order and returns a coordinate snapshot
// that remains valid after the caller changes the original nodal arrays.
TEST( MeshData, returns_owned_face_coordinates_in_local_node_order )
{
  constexpr tribol::IndexT num_elements = 2;
  constexpr tribol::IndexT num_nodes = 4;
  constexpr std::array<tribol::IndexT, num_elements * linear_edge_node_count> connectivity{ 3, 1, 2, 0 };
  std::array<tribol::RealT, num_nodes> x{ 1.0, 3.0, -2.0, 0.0 };
  std::array<tribol::RealT, num_nodes> y{ 1.0, 4.0, 1.0, 0.0 };
  // The connectivity defines one edge from node 3 at (0,0) to node 1 at (3,4), and another from node 2 at (-2,1) to
  // node 0 at (1,1). The nonsequential node IDs make local ordering visible in the returned coordinates.
  tribol::MeshData mesh( mesh_id, num_elements, num_nodes, connectivity.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE,
                         x.data(), y.data(), nullptr, tribol::MemorySpace::Host );
  const auto mesh_view = mesh.getView();

  const auto first_face = mesh_view.getFaceCoords<LinearEdgeCoordinates>( 0 );
  const auto second_face = mesh_view.getFaceCoords<LinearEdgeCoordinates>( 1 );
  const LinearEdgeCoordinates expected_first_face{ tribol::MdArray<tribol::RealT, edge_dim>{ 0.0, 0.0 },
                                                   tribol::MdArray<tribol::RealT, edge_dim>{ 3.0, 4.0 } };
  const LinearEdgeCoordinates expected_second_face{ tribol::MdArray<tribol::RealT, edge_dim>{ -2.0, 1.0 },
                                                    tribol::MdArray<tribol::RealT, edge_dim>{ 1.0, 1.0 } };
  ExpectCoordinatesEqual( first_face, expected_first_face );
  ExpectCoordinatesEqual( second_face, expected_second_face );

  // Changing the caller-owned mesh arrays must not change coordinates returned by an earlier call.
  x[3] = 2.0;
  x[1] = 5.0;
  y[3] = -1.0;
  y[1] = 3.0;
  const LinearEdgeCoordinates expected_updated_first_face{ tribol::MdArray<tribol::RealT, edge_dim>{ 2.0, -1.0 },
                                                           tribol::MdArray<tribol::RealT, edge_dim>{ 5.0, 3.0 } };
  ExpectCoordinatesEqual( first_face, expected_first_face );
  ExpectCoordinatesEqual( mesh_view.getFaceCoords<LinearEdgeCoordinates>( 0 ), expected_updated_first_face );
}

TEST( MeshData, stores_quadratic_edge_data )
{
  // The midpoint follows the two edge vertices in the quadratic basis order.
  constexpr std::array<tribol::IndexT, quadratic_edge_node_count> connectivity{ 0, 1, 2 };
  constexpr std::array<tribol::RealT, quadratic_edge_node_count> x{ 0.0, 1.0, 0.5 };
  constexpr std::array<tribol::RealT, quadratic_edge_node_count> y{ 0.0, 0.0, 0.25 };
  constexpr std::array<tribol::RealT, quadratic_edge_node_count> velocity_x{ 1.0, 2.0, 3.0 };
  constexpr std::array<tribol::RealT, quadratic_edge_node_count> velocity_y{ 3.0, 2.0, 1.0 };

  tribol::MeshData mesh( mesh_id, one_element, quadratic_edge_node_count, connectivity.data(), tribol::EDGE,
                         tribol::QUADRATIC_LAGRANGE, x.data(), y.data(), nullptr, tribol::MemorySpace::Host );
  mesh.setVelocity( velocity_x.data(), velocity_y.data(), nullptr );
  auto mesh_view = mesh.getView();

  EXPECT_EQ( mesh.getElementType(), tribol::EDGE );
  EXPECT_EQ( mesh.getFaceOrder(), tribol::QUADRATIC_LAGRANGE );
  EXPECT_EQ( mesh.spatialDim(), edge_dim );
  EXPECT_EQ( mesh.numberOfElements(), one_element );
  EXPECT_EQ( mesh.numberOfNodes(), quadratic_edge_node_count );
  EXPECT_EQ( mesh.numberOfNodesPerElement(), quadratic_edge_node_count );
  EXPECT_EQ( mesh_view.getFaceOrder(), tribol::QUADRATIC_LAGRANGE );

  std::array<tribol::RealT, edge_dim * quadratic_edge_node_count> face_coordinates{};
  std::array<tribol::RealT, edge_dim * quadratic_edge_node_count> face_velocities{};
  mesh_view.getFaceCoords( first_element_id, face_coordinates.data() );
  mesh_view.getFaceVelocities( first_element_id, face_velocities.data() );
  for ( tribol::IndexT local_node{}; local_node < quadratic_edge_node_count; ++local_node ) {
    EXPECT_EQ( mesh.getGlobalNodeId( first_element_id, local_node ), connectivity[local_node] );
    EXPECT_DOUBLE_EQ( face_coordinates[edge_dim * local_node + x_component], x[local_node] );
    EXPECT_DOUBLE_EQ( face_coordinates[edge_dim * local_node + y_component], y[local_node] );
    EXPECT_DOUBLE_EQ( face_velocities[edge_dim * local_node + x_component], velocity_x[local_node] );
    EXPECT_DOUBLE_EQ( face_velocities[edge_dim * local_node + y_component], velocity_y[local_node] );
  }

  const auto surface_node_ids = mesh.sortSurfaceNodeIds();
  ASSERT_EQ( surface_node_ids.size(), quadratic_edge_node_count );
  for ( tribol::IndexT local_node{}; local_node < quadratic_edge_node_count; ++local_node ) {
    EXPECT_EQ( surface_node_ids[local_node], local_node );
  }

  tribol::PalletAvgNormal element_normal;
  EXPECT_EQ( mesh.computeFaceData( tribol::ExecutionMode::Sequential, element_normal ),
             higher_order_face_data_is_supported );
  ExpectDerivedFaceDataIsEmpty( mesh );
}

TEST( MeshData, stores_quadratic_quad_data )
{
  // The four edge nodes and center node follow the four corner vertices.
  constexpr std::array<tribol::IndexT, quadratic_quad_node_count> connectivity{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
  constexpr std::array<tribol::RealT, quadratic_quad_node_count> x{ 0.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5 };
  constexpr std::array<tribol::RealT, quadratic_quad_node_count> y{ 0.0, 0.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.5 };
  constexpr std::array<tribol::RealT, quadratic_quad_node_count> z{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  constexpr std::array<tribol::RealT, quadratic_quad_node_count> velocity_x{ 1.0, 1.0, 1.0, 1.0, 1.0,
                                                                             1.0, 1.0, 1.0, 1.0 };
  constexpr std::array<tribol::RealT, quadratic_quad_node_count> velocity_y{ 2.0, 2.0, 2.0, 2.0, 2.0,
                                                                             2.0, 2.0, 2.0, 2.0 };
  constexpr std::array<tribol::RealT, quadratic_quad_node_count> velocity_z{ 3.0, 3.0, 3.0, 3.0, 3.0,
                                                                             3.0, 3.0, 3.0, 3.0 };

  tribol::MeshData mesh( mesh_id, one_element, quadratic_quad_node_count, connectivity.data(), tribol::QUAD,
                         tribol::QUADRATIC_LAGRANGE, x.data(), y.data(), z.data(), tribol::MemorySpace::Host );
  mesh.setVelocity( velocity_x.data(), velocity_y.data(), velocity_z.data() );
  auto mesh_view = mesh.getView();

  EXPECT_EQ( mesh.getElementType(), tribol::QUAD );
  EXPECT_EQ( mesh.getFaceOrder(), tribol::QUADRATIC_LAGRANGE );
  EXPECT_EQ( mesh.spatialDim(), quad_dim );
  EXPECT_EQ( mesh.numberOfElements(), one_element );
  EXPECT_EQ( mesh.numberOfNodes(), quadratic_quad_node_count );
  EXPECT_EQ( mesh.numberOfNodesPerElement(), quadratic_quad_node_count );
  EXPECT_EQ( mesh_view.getFaceOrder(), tribol::QUADRATIC_LAGRANGE );

  std::array<tribol::RealT, quad_dim * quadratic_quad_node_count> face_coordinates{};
  std::array<tribol::RealT, quad_dim * quadratic_quad_node_count> face_velocities{};
  mesh_view.getFaceCoords( first_element_id, face_coordinates.data() );
  mesh_view.getFaceVelocities( first_element_id, face_velocities.data() );
  for ( tribol::IndexT local_node{}; local_node < quadratic_quad_node_count; ++local_node ) {
    EXPECT_EQ( mesh_view.getGlobalNodeId( first_element_id, local_node ), connectivity[local_node] );
    EXPECT_DOUBLE_EQ( face_coordinates[quad_dim * local_node + x_component], x[local_node] );
    EXPECT_DOUBLE_EQ( face_coordinates[quad_dim * local_node + y_component], y[local_node] );
    EXPECT_DOUBLE_EQ( face_coordinates[quad_dim * local_node + z_component], z[local_node] );
    EXPECT_DOUBLE_EQ( face_velocities[quad_dim * local_node + x_component], velocity_x[local_node] );
    EXPECT_DOUBLE_EQ( face_velocities[quad_dim * local_node + y_component], velocity_y[local_node] );
    EXPECT_DOUBLE_EQ( face_velocities[quad_dim * local_node + z_component], velocity_z[local_node] );
  }

  const auto surface_node_ids = mesh.sortSurfaceNodeIds();
  ASSERT_EQ( surface_node_ids.size(), quadratic_quad_node_count );
  for ( tribol::IndexT local_node{}; local_node < quadratic_quad_node_count; ++local_node ) {
    EXPECT_EQ( surface_node_ids[local_node], local_node );
  }

  tribol::PalletAvgNormal element_normal;
  EXPECT_EQ( mesh.computeFaceData( tribol::ExecutionMode::Sequential, element_normal ),
             higher_order_face_data_is_supported );
  ExpectDerivedFaceDataIsEmpty( mesh );
}

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
