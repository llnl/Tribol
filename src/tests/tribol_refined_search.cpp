// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Test ProjectedEdgeSearch candidate-pair filtering.
 *
 * These tests check that filtering removes exact self-pairs, optionally removes shared-node pairs, and rejects edges
 * without nondegenerate opposing normals. After filtering, the tests verify that the input candidates remain unchanged
 * and retained pairs are returned in a separate array. They also cover deferred overlap calculation, updated
 * coordinates, and empty results.
 * Direct tests of same-element detection, shared-node detection, and opposing edge normals are in
 * tribol_interface_pair_checks.cpp.
 * Tests run with sequential execution and each enabled OpenMP, CUDA, or HIP backend. GPU cases use device arrays,
 * and assertions read copies on the host. Results may list accepted pairs in any order.
 */

#include <memory>
#include <set>
#include <utility>

#include "ExecutionTest.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/search/ProjectedEdgeSearch.hpp"

namespace {

constexpr tribol::IndexT integration_mesh_id = 0;
constexpr tribol::IndexT projected_mesh_id = 1;
constexpr tribol::IndexT nodes_per_edge = 2;
constexpr tribol::RealT edge_gap = 0.1;

using PairIds = std::set<std::pair<tribol::IndexT, tribol::IndexT>>;

PairIds pairIds( const tribol::ArrayT<tribol::InterfacePair>& pairs )
{
  const tribol::ArrayT<tribol::InterfacePair, 1, tribol::MemorySpace::Host> host_pairs( pairs );
  PairIds ids;
  for ( const auto& pair : host_pairs ) ids.emplace( pair.element_id1_, pair.element_id2_ );
  EXPECT_EQ( ids.size(), pairs.size() );
  return ids;
}

class RefinedSearchTest : public tribol::detail::ExecutionTest {};

// Call ProjectedEdgeSearch through a RefinedSearch pointer and check that it removes the pair whose normals point in
// the same direction and returns the other two pairs.
// Filtering must leave the input unchanged, and later input changes must not change the returned pairs.
TEST_P( RefinedSearchTest, filters_and_compacts_without_consuming_candidates )
{
  // Mesh 1 has two unit edges; the opposing edges lie strictly inside them.
  const auto x = copyToExecutionSpace<tribol::RealT>( { 0.0, 1.0, 2.0, 0.8, 0.2, 1.8, 1.2 } );
  const auto y = copyToExecutionSpace<tribol::RealT>( { 0.0, 0.0, 0.0, edge_gap, edge_gap, edge_gap, edge_gap } );
  const auto integration_conn = copyToExecutionSpace<tribol::IndexT>( { 0, 1, 1, 2 } );
  const auto projected_conn = copyToExecutionSpace<tribol::IndexT>( { 3, 4, 4, 3, 5, 6 } );
  tribol::MeshData integration( integration_mesh_id, integration_conn.size() / nodes_per_edge, x.size(),
                                integration_conn.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE, x.data(), y.data(),
                                nullptr, memorySpace() );
  tribol::MeshData projected( projected_mesh_id, projected_conn.size() / nodes_per_edge, x.size(),
                              projected_conn.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE, x.data(), y.data(), nullptr,
                              memorySpace() );
  // The first pair has normals pointing in the same direction and is rejected. The other two pairs are accepted.
  auto candidates = copyToExecutionSpace<tribol::InterfacePair>( { { 0, 1 }, { 1, 2 }, { 0, 0 } } );
  const tribol::ArrayT<tribol::InterfacePair, 1, tribol::MemorySpace::Host> original( candidates );
  std::unique_ptr<tribol::RefinedSearch> search = std::make_unique<tribol::ProjectedEdgeSearch>( GetParam(), false );
  const auto accepted = search->filterCandidatePairs( candidates, integration.getView(), projected.getView() );
  EXPECT_EQ( pairIds( accepted ), ( PairIds{ { 0, 0 }, { 1, 2 } } ) );
  EXPECT_EQ( accepted.getAllocatorID(), allocatorId() );
  const tribol::ArrayT<tribol::InterfacePair, 1, tribol::MemorySpace::Host> unchanged( candidates );
  ASSERT_EQ( unchanged.size(), original.size() );
  for ( tribol::IndexT i = 0; i < unchanged.size(); ++i ) {
    EXPECT_EQ( unchanged[i].element_id1_, original[i].element_id1_ );
    EXPECT_EQ( unchanged[i].element_id2_, original[i].element_id2_ );
  }
  candidates.fill( tribol::InterfacePair{ 1, 0 } );
  EXPECT_EQ( pairIds( accepted ), ( PairIds{ { 0, 0 }, { 1, 2 } } ) );
}

// Self-pairs are always rejected; the shared-node setting controls only the pair that reuses endpoint IDs.
TEST_P( RefinedSearchTest, applies_shared_node_setting )
{
  // The third edge uses distinct node IDs even though its endpoints lie inside the first edge.
  const auto x = copyToExecutionSpace<tribol::RealT>( { 0.0, 1.0, 0.75, 0.25 } );
  const auto y = copyToExecutionSpace<tribol::RealT>( { 0.0, 0.0, 0.0, 0.0 } );
  const auto conn = copyToExecutionSpace<tribol::IndexT>( { 0, 1, 1, 0, 2, 3 } );
  tribol::MeshData mesh( integration_mesh_id, conn.size() / nodes_per_edge, x.size(), conn.data(), tribol::EDGE,
                         tribol::LINEAR_LAGRANGE, x.data(), y.data(), nullptr, memorySpace() );
  // Pair 0 repeats an element, pair 1 shares endpoint IDs, and pair 2 uses distinct endpoint IDs.
  const auto candidates = copyToExecutionSpace<tribol::InterfacePair>( { { 0, 0 }, { 0, 1 }, { 0, 2 } } );
  tribol::ProjectedEdgeSearch allow_shared( GetParam(), false );
  EXPECT_EQ( pairIds( allow_shared.filterCandidatePairs( candidates, mesh.getView(), mesh.getView() ) ),
             ( PairIds{ { 0, 1 }, { 0, 2 } } ) );
  tribol::ProjectedEdgeSearch reject_shared( GetParam(), true );
  EXPECT_EQ( pairIds( reject_shared.filterCandidatePairs( candidates, mesh.getView(), mesh.getView() ) ),
             ( PairIds{ { 0, 2 } } ) );
}

// Reversing the target orientation must affect the next search, and empty input must return no pairs.
// Previously returned pairs must remain valid after another search and after the search object is destroyed.
TEST_P( RefinedSearchTest, empty_results_and_current_mesh_views )
{
  const auto x1 = copyToExecutionSpace<tribol::RealT>( { 0.0, 1.0 } );
  const auto y1 = copyToExecutionSpace<tribol::RealT>( { 0.0, 0.0 } );
  const auto x2 = copyToExecutionSpace<tribol::RealT>( { 1.0, 0.0 } );
  const auto near_y = copyToExecutionSpace<tribol::RealT>( { edge_gap, edge_gap } );
  const auto reversed_x = copyToExecutionSpace<tribol::RealT>( { 0.0, 1.0 } );
  const auto conn = copyToExecutionSpace<tribol::IndexT>( { 0, 1 } );
  tribol::MeshData mesh1( integration_mesh_id, conn.size() / nodes_per_edge, x1.size(), conn.data(), tribol::EDGE,
                          tribol::LINEAR_LAGRANGE, x1.data(), y1.data(), nullptr, memorySpace() );
  tribol::MeshData mesh2( projected_mesh_id, conn.size() / nodes_per_edge, x2.size(), conn.data(), tribol::EDGE,
                          tribol::LINEAR_LAGRANGE, x2.data(), near_y.data(), nullptr, memorySpace() );
  const auto candidates = copyToExecutionSpace<tribol::InterfacePair>( { { 0, 0 } } );
  auto search = std::make_unique<tribol::ProjectedEdgeSearch>( GetParam(), false );
  auto retained = search->filterCandidatePairs( candidates, mesh1.getView(), mesh2.getView() );
  ASSERT_EQ( retained.size(), candidates.size() );
  mesh2.setPosition( reversed_x.data(), near_y.data(), nullptr );
  EXPECT_TRUE( search->filterCandidatePairs( candidates, mesh1.getView(), mesh2.getView() ).empty() );
  tribol::ArrayT<tribol::InterfacePair> empty( 0, 0, allocatorId() );
  auto result = search->filterCandidatePairs( empty, mesh1.getView(), mesh2.getView() );
  EXPECT_TRUE( result.empty() );
  EXPECT_EQ( result.getAllocatorID(), allocatorId() );
  search.reset();
  EXPECT_EQ( pairIds( retained ), ( PairIds{ { 0, 0 } } ) );
}

// Opposing edges with disjoint projected intervals remain candidates for quadrature generation.
TEST_P( RefinedSearchTest, retains_pairs_without_projected_overlap )
{
  const auto x1 = copyToExecutionSpace<tribol::RealT>( { 0.0, 1.0 } );
  const auto y1 = copyToExecutionSpace<tribol::RealT>( { 0.0, 0.0 } );
  const auto x2 = copyToExecutionSpace<tribol::RealT>( { 3.0, 2.0 } );
  const auto y2 = copyToExecutionSpace<tribol::RealT>( { edge_gap, edge_gap } );
  const auto conn = copyToExecutionSpace<tribol::IndexT>( { 0, 1 } );
  tribol::MeshData mesh1( integration_mesh_id, 1, x1.size(), conn.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE,
                          x1.data(), y1.data(), nullptr, memorySpace() );
  tribol::MeshData mesh2( projected_mesh_id, 1, x2.size(), conn.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE,
                          x2.data(), y2.data(), nullptr, memorySpace() );
  const auto candidates = copyToExecutionSpace<tribol::InterfacePair>( { { 0, 0 } } );
  tribol::ProjectedEdgeSearch search( GetParam(), false );
  EXPECT_EQ( pairIds( search.filterCandidatePairs( candidates, mesh1.getView(), mesh2.getView() ) ),
             ( PairIds{ { 0, 0 } } ) );
}

INSTANTIATE_TEST_SUITE_P( execution, RefinedSearchTest, tribol::detail::enabledExecutionModes() );

}  // namespace

int main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
