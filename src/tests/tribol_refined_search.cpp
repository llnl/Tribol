// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Test how ProjectedEdgeSearch filters candidate pairs and returns the accepted pairs.
 *
 * These tests check that filtering keeps only accepted pairs, leaves the input unchanged, and returns a separate array.
 * They also check the shared-node and smoothing settings, updated mesh coordinates, and empty inputs and results.
 * Individual geometry checks are tested in tribol_interface_pair_checks.cpp.
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

// Call ProjectedEdgeSearch through a RefinedSearch pointer and check that only accepted pairs are returned.
// Filtering must leave the input unchanged, and later input changes must not change the returned pairs.
TEST_P( RefinedSearchTest, filters_and_compacts_without_consuming_candidates )
{
  const auto x = copyToExecutionSpace<tribol::RealT>( { 0, 1, 2, 0.8, 0.2, 1.8, 1.2 } );
  const auto y = copyToExecutionSpace<tribol::RealT>( { 0, 0, 0, 0.1, 0.1, 0.1, 0.1 } );
  const auto integration_conn = copyToExecutionSpace<tribol::IndexT>( { 0, 1, 1, 2 } );
  const auto projected_conn = copyToExecutionSpace<tribol::IndexT>( { 3, 4, 4, 3, 5, 6 } );
  tribol::MeshData integration( 17, 2, 7, integration_conn.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE, x.data(),
                                y.data(), nullptr, memorySpace() );
  tribol::MeshData projected( 29, 3, 7, projected_conn.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE, x.data(),
                              y.data(), nullptr, memorySpace() );
  // The first pair has normals pointing in the same direction and is rejected. The other two pairs are accepted.
  auto candidates = copyToExecutionSpace<tribol::InterfacePair>( { { 0, 1 }, { 1, 2 }, { 0, 0 } } );
  const tribol::ArrayT<tribol::InterfacePair, 1, tribol::MemorySpace::Host> original( candidates );
  std::unique_ptr<tribol::RefinedSearch> search =
      std::make_unique<tribol::ProjectedEdgeSearch>( GetParam(), 0.0, false );
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

// Check that the shared-node setting rejects pairs sharing nodes and smoothing admits pairs just beyond the edge.
// An element paired with itself must always be rejected.
TEST_P( RefinedSearchTest, applies_shared_node_and_support_settings )
{
  const auto x = copyToExecutionSpace<tribol::RealT>( { 0, 1, 1.15, 1.05 } );
  const auto y = copyToExecutionSpace<tribol::RealT>( { 0, 0, 0, 0 } );
  const auto conn = copyToExecutionSpace<tribol::IndexT>( { 0, 1, 1, 0, 2, 3 } );
  tribol::MeshData mesh( 17, 3, 4, conn.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE, x.data(), y.data(), nullptr,
                         memorySpace() );
  // Pair 0 uses the same element twice, pair 1 shares nodes, and pair 2 overlaps only when smoothing is enabled.
  const auto candidates = copyToExecutionSpace<tribol::InterfacePair>( { { 0, 0 }, { 0, 1 }, { 0, 2 } } );
  tribol::ProjectedEdgeSearch expanded( GetParam(), 0.1, false );
  EXPECT_EQ( pairIds( expanded.filterCandidatePairs( candidates, mesh.getView(), mesh.getView() ) ),
             ( PairIds{ { 0, 1 }, { 0, 2 } } ) );
  tribol::ProjectedEdgeSearch reject_shared( GetParam(), 0.1, true );
  EXPECT_EQ( pairIds( reject_shared.filterCandidatePairs( candidates, mesh.getView(), mesh.getView() ) ),
             ( PairIds{ { 0, 2 } } ) );
  tribol::ProjectedEdgeSearch unexpanded( GetParam(), 0.0, false );
  EXPECT_EQ( pairIds( unexpanded.filterCandidatePairs( candidates, mesh.getView(), mesh.getView() ) ),
             ( PairIds{ { 0, 1 } } ) );
}

// Check that moving an edge changes the next result and that empty input returns no pairs.
// Previously returned pairs must remain valid after another search and after the search object is destroyed.
TEST_P( RefinedSearchTest, empty_results_and_current_mesh_views )
{
  const auto x1 = copyToExecutionSpace<tribol::RealT>( { 0, 1 } );
  const auto y1 = copyToExecutionSpace<tribol::RealT>( { 0, 0 } );
  const auto x2 = copyToExecutionSpace<tribol::RealT>( { 1, 0 } );
  const auto near_y = copyToExecutionSpace<tribol::RealT>( { 0.1, 0.1 } );
  const auto far_x = copyToExecutionSpace<tribol::RealT>( { 10, 9 } );
  const auto conn = copyToExecutionSpace<tribol::IndexT>( { 0, 1 } );
  tribol::MeshData mesh1( 17, 1, 2, conn.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE, x1.data(), y1.data(), nullptr,
                          memorySpace() );
  tribol::MeshData mesh2( 29, 1, 2, conn.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE, x2.data(), near_y.data(),
                          nullptr, memorySpace() );
  const auto candidates = copyToExecutionSpace<tribol::InterfacePair>( { { 0, 0 } } );
  auto search = std::make_unique<tribol::ProjectedEdgeSearch>( GetParam(), 0.1, false );
  auto retained = search->filterCandidatePairs( candidates, mesh1.getView(), mesh2.getView() );
  ASSERT_EQ( retained.size(), 1 );
  mesh2.setPosition( far_x.data(), near_y.data(), nullptr );
  EXPECT_TRUE( search->filterCandidatePairs( candidates, mesh1.getView(), mesh2.getView() ).empty() );
  tribol::ArrayT<tribol::InterfacePair> empty( 0, 0, allocatorId() );
  auto result = search->filterCandidatePairs( empty, mesh1.getView(), mesh2.getView() );
  EXPECT_TRUE( result.empty() );
  EXPECT_EQ( result.getAllocatorID(), allocatorId() );
  search.reset();
  EXPECT_EQ( pairIds( retained ), ( PairIds{ { 0, 0 } } ) );
}

INSTANTIATE_TEST_SUITE_P( execution, RefinedSearchTest, tribol::detail::enabledExecutionModes() );

}  // namespace

int main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
