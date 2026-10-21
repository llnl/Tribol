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

template <typename Search>
CandidateIds FindCandidateIds( tribol::MeshData& mesh1, tribol::MeshData& mesh2 )
{
  Search search( mesh1.getView(), mesh2.getView(), proximity_scale_for_raised_nodes, no_residual_gap );
  search.initialize();

  CandidateIds candidate_ids;
  for ( const auto& pair : search.findCandidatePairs() ) {
    candidate_ids.emplace( pair.m_element_id1, pair.m_element_id2 );
  }
  return candidate_ids;
}

template <int SpatialDimension>
void ExpectBothSearchesFindOnlyFirstQuery( tribol::MeshData& mesh1, tribol::MeshData& mesh2 )
{
  const CandidateIds expected_candidates{ { 0, 0 } };

  // Grid and BVH may enumerate intersections differently, so compare element IDs without imposing an output order.
  EXPECT_EQ( FindCandidateIds<tribol::GridSearch<SpatialDimension>>( mesh1, mesh2 ), expected_candidates );
  EXPECT_EQ( ( FindCandidateIds<tribol::BvhSearch<SpatialDimension, axom::SEQ_EXEC>>( mesh1, mesh2 ) ),
             expected_candidates );
}

TEST( CoarseSearch, quadratic_edge_nodes_contribute_to_bounds )
{
  // The raised midpoint is the only mesh-1 node near the first query edge. Finding that query confirms that all three
  // finite-element nodes contribute to the bound; excluding the distant query confirms that the bound remains local.
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

  // Coarse search must use connectivity and coordinates directly because higher-order MeshData intentionally has no
  // derived centroid, radius, area, or normal data.
  EXPECT_FALSE( mesh1.getView().hasElementCentroids() );
  EXPECT_FALSE( mesh1.getView().hasElementNormals() );
  EXPECT_FALSE( mesh1.getView().hasFaceRadius() );
  ExpectBothSearchesFindOnlyFirstQuery<2>( mesh1, mesh2 );
}

TEST( CoarseSearch, quadratic_quad_nodes_contribute_to_bounds )
{
  // The raised center node is the only mesh-1 node near the first query quadrilateral. The coarse bounds must include
  // that node while still excluding a second quadrilateral that is spatially distant.
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

  // No method-independent face normal exists for this higher-order data, so BVH must use the same nodal inputs as Grid.
  EXPECT_FALSE( mesh1.getView().hasElementCentroids() );
  EXPECT_FALSE( mesh1.getView().hasElementNormals() );
  EXPECT_FALSE( mesh1.getView().hasFaceRadius() );
  ExpectBothSearchesFindOnlyFirstQuery<3>( mesh1, mesh2 );
}

TEST( CoarseSearch, empty_meshes_return_no_candidates )
{
  constexpr std::array<tribol::IndexT, 3> connectivity{ 0, 1, 2 };
  constexpr std::array<tribol::RealT, 3> x{ 0.0, 1.0, 0.5 };
  constexpr std::array<tribol::RealT, 3> y{ 0.0, 0.0, 0.25 };

  tribol::MeshData empty_mesh( mesh1_id, 0, 0, nullptr, tribol::EDGE, tribol::QUADRATIC_LAGRANGE, nullptr, nullptr,
                               nullptr, tribol::MemorySpace::Host );
  tribol::MeshData populated_mesh( mesh2_id, 1, x.size(), connectivity.data(), tribol::EDGE, tribol::QUADRATIC_LAGRANGE,
                                   x.data(), y.data(), nullptr, tribol::MemorySpace::Host );

  // Either empty input prevents a spatial index from producing pairs, regardless of which mesh is queried.
  EXPECT_TRUE( FindCandidateIds<tribol::GridSearch<2>>( empty_mesh, populated_mesh ).empty() );
  EXPECT_TRUE( FindCandidateIds<tribol::GridSearch<2>>( populated_mesh, empty_mesh ).empty() );
  EXPECT_TRUE( ( FindCandidateIds<tribol::BvhSearch<2, axom::SEQ_EXEC>>( empty_mesh, populated_mesh ).empty() ) );
  EXPECT_TRUE( ( FindCandidateIds<tribol::BvhSearch<2, axom::SEQ_EXEC>>( populated_mesh, empty_mesh ).empty() ) );
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
    EXPECT_EQ( ( FindCandidateIds<tribol::BvhSearch<2, axom::SEQ_EXEC>>( mesh_manager.at( mesh1_id ),
                                                                         mesh_manager.at( mesh2_id ) ) ),
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

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
