// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
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
constexpr tribol::IndexT first_node_id = 0;
constexpr tribol::IndexT second_node_id = 1;
constexpr tribol::IndexT third_node_id = 2;
constexpr tribol::IndexT fourth_node_id = 3;
constexpr tribol::IndexT fifth_node_id = 4;
constexpr tribol::IndexT sixth_node_id = 5;
constexpr tribol::IndexT seventh_node_id = 6;
constexpr tribol::IndexT eighth_node_id = 7;
constexpr tribol::IndexT ninth_node_id = 8;
constexpr tribol::IndexT first_local_node = 0;
constexpr tribol::IndexT linear_edge_node_count = 2;
constexpr tribol::IndexT quadratic_edge_node_count = 3;
constexpr tribol::IndexT quadratic_quad_node_count = 9;
constexpr int edge_dimension = 2;
constexpr int quad_dimension = 3;
constexpr int x_component = 0;
constexpr int y_component = 1;
constexpr int z_component = 2;
constexpr tribol::RealT zero = 0.0;
constexpr tribol::RealT quarter = 0.25;
constexpr tribol::RealT half = 0.5;
constexpr tribol::RealT one = 1.0;
constexpr tribol::RealT two = 2.0;
constexpr tribol::RealT three = 3.0;
constexpr bool higher_order_face_data_is_supported = false;
constexpr bool derived_face_data_is_populated = false;

void ExpectDerivedFaceDataIsEmpty( tribol::MeshData& mesh )
{
  auto mesh_view = mesh.getView();
  EXPECT_EQ( mesh_view.hasElementCentroids(), derived_face_data_is_populated );
  EXPECT_EQ( mesh_view.hasElementNormals(), derived_face_data_is_populated );
  EXPECT_EQ( mesh_view.hasFaceRadius(), derived_face_data_is_populated );
  EXPECT_EQ( mesh_view.hasElementAreas(), derived_face_data_is_populated );
}

TEST( MeshData, default_constructor_uses_linear_order )
{
  constexpr std::array<tribol::IndexT, linear_edge_node_count> connectivity{ first_node_id, second_node_id };
  constexpr std::array<tribol::RealT, linear_edge_node_count> x{ zero, one };
  constexpr std::array<tribol::RealT, linear_edge_node_count> y{ zero, zero };

  tribol::MeshData mesh( mesh_id, one_element, linear_edge_node_count, connectivity.data(), tribol::LINEAR_EDGE,
                         x.data(), y.data(), nullptr, tribol::MemorySpace::Host );
  auto mesh_view = mesh.getView();

  EXPECT_EQ( mesh.getElementType(), tribol::LINEAR_EDGE );
  EXPECT_EQ( mesh.getFaceOrder(), tribol::LINEAR );
  EXPECT_EQ( mesh.numberOfNodesPerElement(), linear_edge_node_count );
  EXPECT_EQ( mesh_view.getFaceOrder(), tribol::LINEAR );
  EXPECT_EQ( mesh_view.numberOfNodesPerElement(), linear_edge_node_count );
}

TEST( MeshData, stores_quadratic_edge_data )
{
  // The midpoint follows the two edge vertices in the quadratic basis order.
  constexpr std::array<tribol::IndexT, quadratic_edge_node_count> connectivity{ first_node_id, second_node_id,
                                                                                third_node_id };
  constexpr std::array<tribol::RealT, quadratic_edge_node_count> x{ zero, one, half };
  constexpr std::array<tribol::RealT, quadratic_edge_node_count> y{ zero, zero, quarter };
  constexpr std::array<tribol::RealT, quadratic_edge_node_count> velocity_x{ one, two, three };
  constexpr std::array<tribol::RealT, quadratic_edge_node_count> velocity_y{ three, two, one };

  tribol::MeshData mesh( mesh_id, one_element, quadratic_edge_node_count, connectivity.data(), tribol::LINEAR_EDGE,
                         tribol::QUADRATIC_LAGRANGE, x.data(), y.data(), nullptr, tribol::MemorySpace::Host );
  mesh.setVelocity( velocity_x.data(), velocity_y.data(), nullptr );
  auto mesh_view = mesh.getView();

  EXPECT_EQ( mesh.getElementType(), tribol::LINEAR_EDGE );
  EXPECT_EQ( mesh.getFaceOrder(), tribol::QUADRATIC_LAGRANGE );
  EXPECT_EQ( mesh.spatialDimension(), edge_dimension );
  EXPECT_EQ( mesh.numberOfElements(), one_element );
  EXPECT_EQ( mesh.numberOfNodes(), quadratic_edge_node_count );
  EXPECT_EQ( mesh.numberOfNodesPerElement(), quadratic_edge_node_count );
  EXPECT_EQ( mesh_view.getFaceOrder(), tribol::QUADRATIC_LAGRANGE );

  std::array<tribol::RealT, edge_dimension * quadratic_edge_node_count> face_coordinates{};
  std::array<tribol::RealT, edge_dimension * quadratic_edge_node_count> face_velocities{};
  mesh_view.getFaceCoords( first_element_id, face_coordinates.data() );
  mesh_view.getFaceVelocities( first_element_id, face_velocities.data() );
  for ( tribol::IndexT local_node = first_local_node; local_node < quadratic_edge_node_count; ++local_node ) {
    EXPECT_EQ( mesh.getGlobalNodeId( first_element_id, local_node ), connectivity[local_node] );
    EXPECT_DOUBLE_EQ( face_coordinates[edge_dimension * local_node + x_component], x[local_node] );
    EXPECT_DOUBLE_EQ( face_coordinates[edge_dimension * local_node + y_component], y[local_node] );
    EXPECT_DOUBLE_EQ( face_velocities[edge_dimension * local_node + x_component], velocity_x[local_node] );
    EXPECT_DOUBLE_EQ( face_velocities[edge_dimension * local_node + y_component], velocity_y[local_node] );
  }

  const auto surface_node_ids = mesh.sortSurfaceNodeIds();
  ASSERT_EQ( surface_node_ids.size(), quadratic_edge_node_count );
  for ( tribol::IndexT local_node = first_local_node; local_node < quadratic_edge_node_count; ++local_node ) {
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
  constexpr std::array<tribol::IndexT, quadratic_quad_node_count> connectivity{
      first_node_id, second_node_id,  third_node_id,  fourth_node_id, fifth_node_id,
      sixth_node_id, seventh_node_id, eighth_node_id, ninth_node_id };
  constexpr std::array<tribol::RealT, quadratic_quad_node_count> x{ zero, one, one, zero, half, one, half, zero, half };
  constexpr std::array<tribol::RealT, quadratic_quad_node_count> y{ zero, zero, one, one, zero, half, one, half, half };
  constexpr std::array<tribol::RealT, quadratic_quad_node_count> z{ zero, zero, zero, zero, zero,
                                                                    zero, zero, zero, zero };
  constexpr std::array<tribol::RealT, quadratic_quad_node_count> velocity_x{ one, one, one, one, one,
                                                                             one, one, one, one };
  constexpr std::array<tribol::RealT, quadratic_quad_node_count> velocity_y{ two, two, two, two, two,
                                                                             two, two, two, two };
  constexpr std::array<tribol::RealT, quadratic_quad_node_count> velocity_z{ three, three, three, three, three,
                                                                             three, three, three, three };

  tribol::MeshData mesh( mesh_id, one_element, quadratic_quad_node_count, connectivity.data(), tribol::LINEAR_QUAD,
                         tribol::QUADRATIC_LAGRANGE, x.data(), y.data(), z.data(), tribol::MemorySpace::Host );
  mesh.setVelocity( velocity_x.data(), velocity_y.data(), velocity_z.data() );
  auto mesh_view = mesh.getView();

  EXPECT_EQ( mesh.getElementType(), tribol::LINEAR_QUAD );
  EXPECT_EQ( mesh.getFaceOrder(), tribol::QUADRATIC_LAGRANGE );
  EXPECT_EQ( mesh.spatialDimension(), quad_dimension );
  EXPECT_EQ( mesh.numberOfElements(), one_element );
  EXPECT_EQ( mesh.numberOfNodes(), quadratic_quad_node_count );
  EXPECT_EQ( mesh.numberOfNodesPerElement(), quadratic_quad_node_count );
  EXPECT_EQ( mesh_view.getFaceOrder(), tribol::QUADRATIC_LAGRANGE );

  std::array<tribol::RealT, quad_dimension * quadratic_quad_node_count> face_coordinates{};
  std::array<tribol::RealT, quad_dimension * quadratic_quad_node_count> face_velocities{};
  mesh_view.getFaceCoords( first_element_id, face_coordinates.data() );
  mesh_view.getFaceVelocities( first_element_id, face_velocities.data() );
  for ( tribol::IndexT local_node = first_local_node; local_node < quadratic_quad_node_count; ++local_node ) {
    EXPECT_EQ( mesh_view.getGlobalNodeId( first_element_id, local_node ), connectivity[local_node] );
    EXPECT_DOUBLE_EQ( face_coordinates[quad_dimension * local_node + x_component], x[local_node] );
    EXPECT_DOUBLE_EQ( face_coordinates[quad_dimension * local_node + y_component], y[local_node] );
    EXPECT_DOUBLE_EQ( face_coordinates[quad_dimension * local_node + z_component], z[local_node] );
    EXPECT_DOUBLE_EQ( face_velocities[quad_dimension * local_node + x_component], velocity_x[local_node] );
    EXPECT_DOUBLE_EQ( face_velocities[quad_dimension * local_node + y_component], velocity_y[local_node] );
    EXPECT_DOUBLE_EQ( face_velocities[quad_dimension * local_node + z_component], velocity_z[local_node] );
  }

  const auto surface_node_ids = mesh.sortSurfaceNodeIds();
  ASSERT_EQ( surface_node_ids.size(), quadratic_quad_node_count );
  for ( tribol::IndexT local_node = first_local_node; local_node < quadratic_quad_node_count; ++local_node ) {
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
