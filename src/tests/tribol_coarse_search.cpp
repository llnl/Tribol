// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <array>
#include <set>
#include <utility>

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/search/BvhSearch.hpp"
#include "tribol/search/GridSearch.hpp"
#include "tribol/utils/DataManager.hpp"

namespace {

using CandidateIds = std::set<std::pair<tribol::IndexT, tribol::IndexT>>;

constexpr tribol::IndexT mesh1_id = 17;
constexpr tribol::IndexT mesh2_id = 29;
constexpr tribol::RealT proximity_scale_for_raised_nodes = 0.2;
constexpr tribol::RealT no_residual_gap = 0.0;

CandidateIds GetCandidateIds( const tribol::ArrayT<tribol::InterfacePair>& candidate_pairs )
{
  CandidateIds candidate_ids;
  for ( const auto& pair : candidate_pairs ) {
    candidate_ids.emplace( pair.element_id1_, pair.element_id2_ );
  }
  return candidate_ids;
}

template <typename Search>
CandidateIds FindCandidateIds( tribol::MeshData& mesh1, tribol::MeshData& mesh2 )
{
  Search search( proximity_scale_for_raised_nodes, no_residual_gap );
  return GetCandidateIds( search.findCandidatePairs( mesh1.getView(), mesh2.getView() ) );
}

template <int SpatialDimension>
void ExpectBothSearchesFindOnlyFirstQuery( tribol::MeshData& mesh1, tribol::MeshData& mesh2 )
{
  const CandidateIds expected_candidates{ { 0, 0 } };

  // Grid and BVH may enumerate intersections differently, so compare element IDs without imposing an output order.
  EXPECT_EQ( FindCandidateIds<tribol::GridSearch<SpatialDimension>>( mesh1, mesh2 ), expected_candidates );
  EXPECT_EQ(
      ( FindCandidateIds<tribol::BvhSearch<SpatialDimension, tribol::ExecutionMode::Sequential>>( mesh1, mesh2 ) ),
      expected_candidates );
}

template <int SpatialDimension>
void ExpectBothSearchesFindNoCandidates( tribol::MeshData& mesh1, tribol::MeshData& mesh2 )
{
  EXPECT_TRUE( FindCandidateIds<tribol::GridSearch<SpatialDimension>>( mesh1, mesh2 ).empty() );
  EXPECT_TRUE(
      ( FindCandidateIds<tribol::BvhSearch<SpatialDimension, tribol::ExecutionMode::Sequential>>( mesh1, mesh2 )
            .empty() ) );
}

template <typename Search>
CandidateIds FindCandidateIds( const Search& search, tribol::MeshData& mesh1, tribol::MeshData& mesh2 )
{
  return GetCandidateIds( search.findCandidatePairs( mesh1.getView(), mesh2.getView() ) );
}

TEST( CoarseSearch, quadratic_edge_nodes_contribute_to_bounds )
{
  // Mesh 2 has one query near the raised midpoint and one spatially distant query. The coarse bounds must include the
  // midpoint to find the nearby query without also finding the distant query.
  constexpr std::array<tribol::IndexT, 2> endpoint_connectivity{ 0, 1 };
  constexpr std::array<tribol::IndexT, 3> mesh1_connectivity{ 0, 1, 2 };
  constexpr std::array<tribol::RealT, 3> mesh1_x{ 0.0, 1.0, 0.5 };
  constexpr std::array<tribol::RealT, 3> mesh1_y{ 0.0, 0.0, 2.0 };

  constexpr std::array<tribol::IndexT, 6> mesh2_connectivity{ 0, 1, 2, 3, 4, 5 };
  constexpr std::array<tribol::RealT, 6> mesh2_x{ 0.4, 0.6, 0.5, 4.0, 5.0, 4.5 };
  constexpr std::array<tribol::RealT, 6> mesh2_y{ 2.1, 2.1, 2.15, 5.0, 5.0, 5.2 };

  tribol::MeshData mesh1( mesh1_id, 1, mesh1_x.size(), mesh1_connectivity.data(), tribol::EDGE,
                          tribol::QUADRATIC_LAGRANGE, mesh1_x.data(), mesh1_y.data(), nullptr,
                          tribol::MemorySpace::Host );
  tribol::MeshData mesh2( mesh2_id, 2, mesh2_x.size(), mesh2_connectivity.data(), tribol::EDGE,
                          tribol::QUADRATIC_LAGRANGE, mesh2_x.data(), mesh2_y.data(), nullptr,
                          tribol::MemorySpace::Host );

  tribol::MeshData endpoint_only_mesh1( mesh1_id, 1, endpoint_connectivity.size(), endpoint_connectivity.data(),
                                        tribol::EDGE, tribol::LINEAR_LAGRANGE, mesh1_x.data(), mesh1_y.data(), nullptr,
                                        tribol::MemorySpace::Host );

  // The endpoint-only control remains on y=0 and cannot reach the nearby query. Finding that query only after adding
  // the raised midpoint proves that both searches include the quadratic node in their element bounds.
  ExpectBothSearchesFindNoCandidates<2>( endpoint_only_mesh1, mesh2 );

  // MeshData rejects computeFaceData() for quadratic geometry because its derived quantities use linear face formulas.
  // Grid and BVH must therefore build bounds from connectivity and nodal coordinates without those derived arrays.
  EXPECT_FALSE( mesh1.getView().hasElementCentroids() );
  EXPECT_FALSE( mesh1.getView().hasElementAreas() );
  EXPECT_FALSE( mesh1.getView().hasElementNormals() );
  EXPECT_FALSE( mesh1.getView().hasFaceRadius() );
  ExpectBothSearchesFindOnlyFirstQuery<2>( mesh1, mesh2 );
}

TEST( CoarseSearch, quadratic_quad_nodes_contribute_to_bounds )
{
  // Mesh 1 is a unit quadrilateral in z=0 with its center node raised to z=2. Mesh 2 has a small query centered over
  // that node at z=2.1 and a second query around (4.5, 4.5, 5). The bounds must find only the first query.
  constexpr std::array<tribol::IndexT, 4> corner_connectivity{ 0, 1, 2, 3 };
  constexpr std::array<tribol::IndexT, 9> mesh1_connectivity{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
  constexpr std::array<tribol::RealT, 9> mesh1_x{ 0.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5 };
  constexpr std::array<tribol::RealT, 9> mesh1_y{ 0.0, 0.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.5 };
  constexpr std::array<tribol::RealT, 9> mesh1_z{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0 };

  constexpr std::array<tribol::IndexT, 18> mesh2_connectivity{ 0, 1,  2,  3,  4,  5,  6,  7,  8,
                                                               9, 10, 11, 12, 13, 14, 15, 16, 17 };
  constexpr std::array<tribol::RealT, 18> mesh2_x{ 0.4, 0.6, 0.6, 0.4, 0.5, 0.6, 0.5, 0.4, 0.5,
                                                   4.0, 5.0, 5.0, 4.0, 4.5, 5.0, 4.5, 4.0, 4.5 };
  constexpr std::array<tribol::RealT, 18> mesh2_y{ 0.4, 0.4, 0.6, 0.6, 0.4, 0.5, 0.6, 0.5, 0.5,
                                                   4.0, 4.0, 5.0, 5.0, 4.0, 4.5, 5.0, 4.5, 4.5 };
  constexpr std::array<tribol::RealT, 18> mesh2_z{ 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1,
                                                   5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 };

  tribol::MeshData mesh1( mesh1_id, 1, mesh1_x.size(), mesh1_connectivity.data(), tribol::QUAD,
                          tribol::QUADRATIC_LAGRANGE, mesh1_x.data(), mesh1_y.data(), mesh1_z.data(),
                          tribol::MemorySpace::Host );
  tribol::MeshData mesh2( mesh2_id, 2, mesh2_x.size(), mesh2_connectivity.data(), tribol::QUAD,
                          tribol::QUADRATIC_LAGRANGE, mesh2_x.data(), mesh2_y.data(), mesh2_z.data(),
                          tribol::MemorySpace::Host );

  tribol::MeshData corner_only_mesh1( mesh1_id, 1, corner_connectivity.size(), corner_connectivity.data(),
                                      tribol::QUAD, tribol::LINEAR_LAGRANGE, mesh1_x.data(), mesh1_y.data(),
                                      mesh1_z.data(), tribol::MemorySpace::Host );

  // The corner-only control remains in the z=0 plane and cannot reach the nearby query. Finding that query only after
  // adding the raised center node proves that both searches include the quadratic node in their element bounds.
  ExpectBothSearchesFindNoCandidates<3>( corner_only_mesh1, mesh2 );

  // Higher-order MeshData does not compute the linear face data below. Both searches must build bounds directly from
  // all nodal coordinates so quadratic faces can participate without calling computeFaceData().
  EXPECT_FALSE( mesh1.getView().hasElementCentroids() );
  EXPECT_FALSE( mesh1.getView().hasElementNormals() );
  EXPECT_FALSE( mesh1.getView().hasFaceRadius() );
  ExpectBothSearchesFindOnlyFirstQuery<3>( mesh1, mesh2 );
}

TEST( CoarseSearch, either_empty_input_returns_no_candidates )
{
  constexpr std::array<tribol::IndexT, 3> connectivity{ 0, 1, 2 };
  constexpr std::array<tribol::RealT, 3> x{ 0.0, 1.0, 0.5 };
  constexpr std::array<tribol::RealT, 3> y{ 0.0, 0.0, 0.25 };

  tribol::MeshData empty_mesh( mesh1_id, 0, 0, nullptr, tribol::EDGE, tribol::QUADRATIC_LAGRANGE, nullptr, nullptr,
                               nullptr, tribol::MemorySpace::Host );
  tribol::MeshData populated_mesh( mesh2_id, 1, x.size(), connectivity.data(), tribol::EDGE, tribol::QUADRATIC_LAGRANGE,
                                   x.data(), y.data(), nullptr, tribol::MemorySpace::Host );

  // Either empty input prevents a spatial index from producing pairs, regardless of which mesh is queried.
  ExpectBothSearchesFindNoCandidates<2>( empty_mesh, populated_mesh );
  ExpectBothSearchesFindNoCandidates<2>( populated_mesh, empty_mesh );
}

TEST( CoarseSearch, each_query_uses_current_mesh_views )
{
  constexpr std::array<tribol::IndexT, 3> connectivity{ 0, 1, 2 };
  constexpr std::array<tribol::RealT, 3> mesh1_x{ 0.0, 1.0, 0.5 };
  constexpr std::array<tribol::RealT, 3> mesh1_y{ 0.0, 0.0, 0.25 };
  constexpr std::array<tribol::RealT, 3> mesh2_x{ 0.0, 1.0, 0.5 };
  constexpr std::array<tribol::RealT, 3> nearby_mesh2_y{ 0.1, 0.1, 0.35 };
  constexpr std::array<tribol::RealT, 3> distant_mesh2_y{ 5.0, 5.0, 5.0 };

  tribol::MeshData mesh1( mesh1_id, 1, mesh1_x.size(), connectivity.data(), tribol::EDGE, tribol::QUADRATIC_LAGRANGE,
                          mesh1_x.data(), mesh1_y.data(), nullptr, tribol::MemorySpace::Host );
  tribol::MeshData nearby_mesh2( mesh2_id, 1, mesh2_x.size(), connectivity.data(), tribol::EDGE,
                                 tribol::QUADRATIC_LAGRANGE, mesh2_x.data(), nearby_mesh2_y.data(), nullptr,
                                 tribol::MemorySpace::Host );
  tribol::MeshData distant_mesh2( mesh2_id, 1, mesh2_x.size(), connectivity.data(), tribol::EDGE,
                                  tribol::QUADRATIC_LAGRANGE, mesh2_x.data(), distant_mesh2_y.data(), nullptr,
                                  tribol::MemorySpace::Host );
  const CandidateIds expected_candidate{ { 0, 0 } };

  // The same search object must accept replacement mesh arrays on each call rather than retaining stale views.
  tribol::GridSearch<2> grid_search( proximity_scale_for_raised_nodes, no_residual_gap );
  EXPECT_EQ( FindCandidateIds( grid_search, mesh1, nearby_mesh2 ), expected_candidate );
  EXPECT_TRUE( FindCandidateIds( grid_search, mesh1, distant_mesh2 ).empty() );

  tribol::BvhSearch<2, tribol::ExecutionMode::Sequential> bvh_search( proximity_scale_for_raised_nodes,
                                                                      no_residual_gap );
  EXPECT_EQ( FindCandidateIds( bvh_search, mesh1, nearby_mesh2 ), expected_candidate );
  EXPECT_TRUE( FindCandidateIds( bvh_search, mesh1, distant_mesh2 ).empty() );
}

void ExpectLegacySearchRejectsCoarseCandidate( tribol::BinningMethod binning_method )
{
  constexpr std::array<tribol::IndexT, 2> connectivity{ 0, 1 };
  constexpr std::array<tribol::RealT, 2> x{ 0.0, 1.0 };
  constexpr std::array<tribol::RealT, 2> y{ 0.0, 0.0 };
  std::array<tribol::RealT, 2> response_x{};
  std::array<tribol::RealT, 2> response_y{};

  tribol::finalize();
  tribol::registerMesh( mesh1_id, 1, x.size(), connectivity.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE, x.data(),
                        y.data(), nullptr, tribol::MemorySpace::Host );
  tribol::registerMesh( mesh2_id, 1, x.size(), connectivity.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE, x.data(),
                        y.data(), nullptr, tribol::MemorySpace::Host );
  tribol::registerNodalResponse( mesh1_id, response_x.data(), response_y.data(), nullptr );
  tribol::registerNodalResponse( mesh2_id, response_x.data(), response_y.data(), nullptr );
  tribol::setKinematicConstantPenalty( mesh1_id, 1.0 );
  tribol::setKinematicConstantPenalty( mesh2_id, 1.0 );

  constexpr tribol::IndexT coupling_scheme_id = 41;
  tribol::registerCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                                  tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, binning_method,
                                  tribol::ExecutionMode::Sequential );
  tribol::setPenaltyOptions( coupling_scheme_id, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  auto& mesh_manager = tribol::MeshManager::getInstance();
  const CandidateIds expected_coarse_candidate{ { 0, 0 } };
  if ( binning_method == tribol::BINNING_GRID ) {
    EXPECT_EQ( FindCandidateIds<tribol::GridSearch<2>>( mesh_manager.at( mesh1_id ), mesh_manager.at( mesh2_id ) ),
               expected_coarse_candidate );
  } else {
    EXPECT_EQ( ( FindCandidateIds<tribol::BvhSearch<2, tribol::ExecutionMode::Sequential>>(
                   mesh_manager.at( mesh1_id ), mesh_manager.at( mesh2_id ) ) ),
               expected_coarse_candidate );
  }

  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  ASSERT_TRUE( coupling_scheme.init() );
  coupling_scheme.performBinning();

  // The coincident boxes overlap, but matching face normals are invalid for legacy surface contact. This check confirms
  // that InterfacePairFinder still applies geomFilter() after the standalone search returns its coarse candidate.
  EXPECT_TRUE( coupling_scheme.getInterfacePairs().empty() );
  tribol::finalize();
}

TEST( CoarseSearch, legacy_interface_pair_finder_retains_geometry_filter )
{
  // Exercise both adapters because GridSearch and BvhSearch now share the same legacy filtering stage.
  ExpectLegacySearchRejectsCoarseCandidate( tribol::BINNING_GRID );
  ExpectLegacySearchRejectsCoarseCandidate( tribol::BINNING_BVH );
}

TEST( CoarseSearch, same_mesh_search_deduplicates_pairs )
{
  // When searching a mesh against itself, each intersecting pair must appear in only one mirrored direction.
  constexpr std::array<tribol::IndexT, 4> connectivity{ 0, 1, 1, 2 };
  constexpr std::array<tribol::RealT, 3> x{ 0.0, 1.0, 2.0 };
  constexpr std::array<tribol::RealT, 3> y{ 0.0, 0.0, 0.0 };

  tribol::MeshData mesh( mesh1_id, 2, x.size(), connectivity.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE, x.data(),
                         y.data(), nullptr, tribol::MemorySpace::Host );

  tribol::GridSearch<2> grid_search( proximity_scale_for_raised_nodes, no_residual_gap );
  tribol::BvhSearch<2, tribol::ExecutionMode::Sequential> bvh_search( proximity_scale_for_raised_nodes,
                                                                      no_residual_gap );
  const auto grid_candidates = grid_search.findCandidatePairs( mesh.getView(), mesh.getView() );
  const auto bvh_candidates = bvh_search.findCandidatePairs( mesh.getView(), mesh.getView() );
  const CandidateIds expected_candidates{ { 0, 0 }, { 1, 0 }, { 1, 1 } };

  // Comparing raw counts with the unique ID set ensures neither search returns an identical pair more than once.
  EXPECT_EQ( grid_candidates.size(), expected_candidates.size() );
  EXPECT_EQ( bvh_candidates.size(), expected_candidates.size() );
  EXPECT_EQ( GetCandidateIds( grid_candidates ), expected_candidates );
  EXPECT_EQ( GetCandidateIds( bvh_candidates ), expected_candidates );
}

TEST( CoarseSearch, flat_elements_avoid_division_by_zero )
{
  // A valid line element has zero range normal to the line, which must still produce a valid grid resolution.
  constexpr std::array<tribol::IndexT, 2> connectivity{ 0, 1 };
  constexpr std::array<tribol::RealT, 2> x{ 0.0, 1.0 };
  constexpr std::array<tribol::RealT, 2> y{ 0.0, 0.0 };

  tribol::MeshData mesh( mesh1_id, 1, x.size(), connectivity.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE, x.data(),
                         y.data(), nullptr, tribol::MemorySpace::Host );

  tribol::GridSearch<2> grid_search( 0.0, 0.0 );
  const auto candidates = grid_search.findCandidatePairs( mesh.getView(), mesh.getView() );
  EXPECT_EQ( candidates.size(), 1 );
}

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
