// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Test refined-search orchestration; predicate geometry is covered in tribol_interface_pair_checks.cpp.
 *
 * filters_and_compacts_without_consuming_candidates checks polymorphic dispatch, accepted-pair compaction, allocator
 * preservation, unchanged input candidates, and independence from candidate storage.
 * applies_shared_node_and_support_settings checks that strategy settings reach the kernel and self-pairs stay rejected.
 * empty_results_and_current_mesh_views checks empty input, all-rejected output, fresh mesh views, and results that
 * outlive their strategy.
 * Every test runs with sequential execution and, when enabled, OpenMP; accepted-pair order is unspecified.
 */

#include <memory>
#include <set>
#include <utility>

#include <gtest/gtest.h>
#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/search/ProjectedEdgeSearch.hpp"

namespace {

using PairIds = std::set<std::pair<tribol::IndexT, tribol::IndexT>>;
const int host_allocator = tribol::getResourceAllocatorID( tribol::MemorySpace::Host );

PairIds pairIds( const tribol::ArrayT<tribol::InterfacePair>& pairs )
{
  PairIds ids;
  for ( const auto& pair : pairs ) ids.emplace( pair.element_id1_, pair.element_id2_ );
  EXPECT_EQ( ids.size(), pairs.size() );
  return ids;
}

class RefinedSearchTest : public testing::TestWithParam<tribol::ExecutionMode> {};

// Verify that base-interface filtering preserves candidates and returns a compact result independent of their storage.
TEST_P( RefinedSearchTest, filters_and_compacts_without_consuming_candidates )
{
  const tribol::RealT x[] = { 0, 1, 2, 0.8, 0.2, 1.8, 1.2 };
  const tribol::RealT y[] = { 0, 0, 0, 0.1, 0.1, 0.1, 0.1 };
  const tribol::IndexT integration_conn[] = { 0, 1, 1, 2 };
  const tribol::IndexT projected_conn[] = { 3, 4, 4, 3, 5, 6 };
  tribol::MeshData integration( 17, 2, 7, integration_conn, tribol::EDGE, tribol::LINEAR_LAGRANGE, x, y, nullptr,
                                tribol::MemorySpace::Host );
  tribol::MeshData projected( 29, 3, 7, projected_conn, tribol::EDGE, tribol::LINEAR_LAGRANGE, x, y, nullptr,
                              tribol::MemorySpace::Host );
  // The first pair is co-oriented; the other two must survive with their original element IDs.
  tribol::ArrayT<tribol::InterfacePair> candidates( { { 0, 1 }, { 1, 2 }, { 0, 0 } }, host_allocator );
  const tribol::ArrayT<tribol::InterfacePair> original( candidates );
  std::unique_ptr<tribol::RefinedSearch> search =
      std::make_unique<tribol::ProjectedEdgeSearch>( GetParam(), 0.0, false );
  const auto accepted = search->filterCandidatePairs( candidates, integration.getView(), projected.getView() );
  EXPECT_EQ( pairIds( accepted ), ( PairIds{ { 0, 0 }, { 1, 2 } } ) );
  EXPECT_EQ( accepted.getAllocatorID(), host_allocator );
  ASSERT_EQ( candidates.size(), original.size() );
  for ( tribol::IndexT i = 0; i < candidates.size(); ++i ) {
    EXPECT_EQ( candidates[i].element_id1_, original[i].element_id1_ );
    EXPECT_EQ( candidates[i].element_id2_, original[i].element_id2_ );
  }
  candidates.fill( tribol::InterfacePair{ 1, 0 } );
  EXPECT_EQ( pairIds( accepted ), ( PairIds{ { 0, 0 }, { 1, 2 } } ) );
}

// Verify that shared-node rejection and support expansion change the accepted set without ever admitting a self-pair.
TEST_P( RefinedSearchTest, applies_shared_node_and_support_settings )
{
  const tribol::RealT x[] = { 0, 1, 1.15, 1.05 }, y[] = { 0, 0, 0, 0 };
  const tribol::IndexT conn[] = { 0, 1, 1, 0, 2, 3 };
  tribol::MeshData mesh( 17, 3, 4, conn, tribol::EDGE, tribol::LINEAR_LAGRANGE, x, y, nullptr,
                         tribol::MemorySpace::Host );
  // These candidates are respectively a self-pair, a shared-node pair, and a pair overlapping only expanded support.
  const tribol::ArrayT<tribol::InterfacePair> candidates( { { 0, 0 }, { 0, 1 }, { 0, 2 } }, host_allocator );
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

// Verify that a strategy reads replacement mesh arrays, handles empty results, and returns results that outlive it.
TEST_P( RefinedSearchTest, empty_results_and_current_mesh_views )
{
  const tribol::RealT x1[] = { 0, 1 }, y1[] = { 0, 0 };
  const tribol::RealT x2[] = { 1, 0 }, near_y[] = { 0.1, 0.1 }, far_x[] = { 10, 9 };
  const tribol::IndexT conn[] = { 0, 1 };
  tribol::MeshData mesh1( 17, 1, 2, conn, tribol::EDGE, tribol::LINEAR_LAGRANGE, x1, y1, nullptr,
                          tribol::MemorySpace::Host );
  tribol::MeshData mesh2( 29, 1, 2, conn, tribol::EDGE, tribol::LINEAR_LAGRANGE, x2, near_y, nullptr,
                          tribol::MemorySpace::Host );
  tribol::ArrayT<tribol::InterfacePair> candidates( { { 0, 0 } }, host_allocator );
  auto search = std::make_unique<tribol::ProjectedEdgeSearch>( GetParam(), 0.1, false );
  auto retained = search->filterCandidatePairs( candidates, mesh1.getView(), mesh2.getView() );
  ASSERT_EQ( retained.size(), 1 );
  mesh2.setPosition( far_x, near_y, nullptr );
  EXPECT_TRUE( search->filterCandidatePairs( candidates, mesh1.getView(), mesh2.getView() ).empty() );
  tribol::ArrayT<tribol::InterfacePair> empty( 0, 0, host_allocator );
  auto result = search->filterCandidatePairs( empty, mesh1.getView(), mesh2.getView() );
  EXPECT_TRUE( result.empty() );
  EXPECT_EQ( result.getAllocatorID(), host_allocator );
  search.reset();
  EXPECT_EQ( pairIds( retained ), ( PairIds{ { 0, 0 } } ) );
}

INSTANTIATE_TEST_SUITE_P( execution, RefinedSearchTest,
                          testing::Values( tribol::ExecutionMode::Sequential
#ifdef TRIBOL_USE_OPENMP
                                           ,
                                           tribol::ExecutionMode::OpenMP
#endif
                                           ) );

}  // namespace

int main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
