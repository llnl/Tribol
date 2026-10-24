// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <array>

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/search/BvhSearch.hpp"
#include "tribol/search/GridSearch.hpp"
#include "tribol/utils/DataManager.hpp"

namespace {

constexpr tribol::IndexT mesh1_id = 17;
constexpr tribol::IndexT mesh2_id = 29;
constexpr tribol::IndexT coupling_scheme_id = 41;
constexpr tribol::RealT binning_proximity_scale = 0.2;
constexpr tribol::RealT no_residual_gap = 0.0;

void ExpectOnlyRegisteredPair( const tribol::ArrayT<tribol::InterfacePair>& candidate_pairs )
{
  ASSERT_EQ( candidate_pairs.size(), 1 );
  EXPECT_EQ( candidate_pairs[0].element_id1_, 0 );
  EXPECT_EQ( candidate_pairs[0].element_id2_, 0 );
}

void ExpectInterfacePairFinderFiltersSpatialCandidate( tribol::BinningMethod binning_method )
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

  tribol::registerCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                                  tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, binning_method,
                                  tribol::ExecutionMode::Sequential );
  tribol::setPenaltyOptions( coupling_scheme_id, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  auto& mesh_manager = tribol::MeshManager::getInstance();
  if ( binning_method == tribol::BINNING_GRID ) {
    tribol::GridSearch<2> search( binning_proximity_scale, no_residual_gap );
    ExpectOnlyRegisteredPair(
        search.findCandidatePairs( mesh_manager.at( mesh1_id ).getView(), mesh_manager.at( mesh2_id ).getView() ) );
  } else {
    tribol::BvhSearch<2, tribol::ExecutionMode::Sequential> search( binning_proximity_scale, no_residual_gap );
    ExpectOnlyRegisteredPair(
        search.findCandidatePairs( mesh_manager.at( mesh1_id ).getView(), mesh_manager.at( mesh2_id ).getView() ) );
  }

  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  ASSERT_TRUE( coupling_scheme.init() );
  coupling_scheme.performBinning();

  // Identical co-oriented edges have overlapping bounds, so coarse search finds pair { 0, 0 }. InterfacePairFinder
  // must then remove that pair because geomFilter() requires opposing face normals.
  EXPECT_TRUE( coupling_scheme.getInterfacePairs().empty() );
  tribol::finalize();
}

TEST( InterfacePairFinder, applies_geometry_filter_after_coarse_search )
{
  // Exercise both adapters because GridSearch and BvhSearch now share the same legacy filtering stage.
  ExpectInterfacePairFinderFiltersSpatialCandidate( tribol::BINNING_GRID );
  ExpectInterfacePairFinderFiltersSpatialCandidate( tribol::BINNING_BVH );
}

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
