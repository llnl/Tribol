// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Test same-element detection, shared-node detection, and linear-edge normal orientation.
 *
 * These tests call each check directly on host mesh views. Filtering whole candidate arrays and testing enabled
 * execution backends are covered in tribol_refined_search.cpp.
 */

#include <array>

#include "gtest/gtest.h"

#include "axom/slic/core/SimpleLogger.hpp"

#include "tribol/mesh/MeshData.hpp"
#include "tribol/search/InterfacePairChecks.hpp"

namespace {

constexpr tribol::IndexT shared_mesh_id = 11;
constexpr tribol::IndexT opposing_mesh_id = 12;
constexpr tribol::IndexT nodes_per_edge = 2;

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

TEST( InterfacePairChecks, accepts_nondegenerate_edge_normals_separated_by_at_least_90_degrees )
{
  // Endpoint ordering defines each edge normal. Reversing the upper edge gives a 180-degree separation, while the
  // vertical edge gives the 90-degree cutoff used by geomFilter().
  constexpr std::array<tribol::IndexT, 2> connectivity{ 0, 1 };
  constexpr std::array<tribol::RealT, 2> lower_x{ 0.0, 1.0 };
  constexpr std::array<tribol::RealT, 2> lower_y{ 0.0, 0.0 };
  constexpr std::array<tribol::RealT, 2> upper_reversed_x{ 1.0, 0.0 };
  constexpr std::array<tribol::RealT, 2> upper_y{ 0.1, 0.1 };
  constexpr std::array<tribol::RealT, 2> upper_forward_x{ 0.0, 1.0 };
  constexpr std::array<tribol::RealT, 2> perpendicular_x{ 0.5, 0.5 };
  constexpr std::array<tribol::RealT, 2> perpendicular_y{ 0.0, 1.0 };
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
  tribol::MeshData perpendicular( opposing_mesh_id, 1, perpendicular_x.size(), connectivity.data(), tribol::EDGE,
                                  tribol::LINEAR_LAGRANGE, perpendicular_x.data(), perpendicular_y.data(), nullptr,
                                  tribol::MemorySpace::Host );
  tribol::MeshData degenerate( opposing_mesh_id, 1, degenerate_x.size(), connectivity.data(), tribol::EDGE,
                               tribol::LINEAR_LAGRANGE, degenerate_x.data(), degenerate_y.data(), nullptr,
                               tribol::MemorySpace::Host );

  EXPECT_TRUE(
      tribol::InterfacePairChecks::linearEdgeNormalsAreOpposing( lower.getView(), 0, upper_reversed.getView(), 0 ) );
  EXPECT_TRUE(
      tribol::InterfacePairChecks::linearEdgeNormalsAreOpposing( lower.getView(), 0, perpendicular.getView(), 0 ) );
  EXPECT_FALSE(
      tribol::InterfacePairChecks::linearEdgeNormalsAreOpposing( lower.getView(), 0, upper_forward.getView(), 0 ) );
  EXPECT_FALSE(
      tribol::InterfacePairChecks::linearEdgeNormalsAreOpposing( lower.getView(), 0, degenerate.getView(), 0 ) );
}

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
