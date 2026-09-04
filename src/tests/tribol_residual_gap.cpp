// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// gtest includes
#include "gtest/gtest.h"

// Tribol includes
#include "tribol/interface/tribol.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/CouplingScheme.hpp"

using RealT = tribol::RealT;

class ResidualGapTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override { tribol::finalize(); }
};

class ResidualGapBinningTest : public ResidualGapTest, public ::testing::WithParamInterface<tribol::BinningMethod> {};

TEST_F( ResidualGapTest, common_plane_residual_gap )
{
  // Setup two 3D quads with a 0.1 gap
  // Quad 1: z = 0, x in [0, 1], y in [0, 1]
  // Quad 2: z = 0.1, x in [0, 1], y in [0, 1]

  constexpr int numVerts = 4;
  constexpr int numElems = 1;
  RealT x1[numVerts] = { 0.0, 1.0, 1.0, 0.0 };
  RealT y1[numVerts] = { 0.0, 0.0, 1.0, 1.0 };
  RealT z1[numVerts] = { 0.0, 0.0, 0.0, 0.0 };

  RealT x2[numVerts] = { 0.0, 1.0, 1.0, 0.0 };
  RealT y2[numVerts] = { 0.0, 0.0, 1.0, 1.0 };
  RealT z2[numVerts] = { 0.1, 0.1, 0.1, 0.1 };

  // Connectivity for 3D quads
  tribol::IndexT conn1[numVerts] = { 0, 1, 2, 3 };
  // Quad 2 normal points -z: 0-3-2-1 is CCW viewed from -z.
  tribol::IndexT conn2[numVerts] = { 0, 3, 2, 1 };

  tribol::registerMesh( 0, numElems, numVerts, &conn1[0], static_cast<int>( tribol::LINEAR_QUAD ), &x1[0], &y1[0],
                        &z1[0], tribol::MemorySpace::Host );
  tribol::registerMesh( 1, numElems, numVerts, &conn2[0], static_cast<int>( tribol::LINEAR_QUAD ), &x2[0], &y2[0],
                        &z2[0], tribol::MemorySpace::Host );

  RealT fx1[numVerts] = { 0., 0., 0., 0. };
  RealT fy1[numVerts] = { 0., 0., 0., 0. };
  RealT fz1[numVerts] = { 0., 0., 0., 0. };
  RealT fx2[numVerts] = { 0., 0., 0., 0. };
  RealT fy2[numVerts] = { 0., 0., 0., 0. };
  RealT fz2[numVerts] = { 0., 0., 0., 0. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], &fz1[0] );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], &fz2[0] );

  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  // Default residual_gap is 0.0. Gap is 0.1, so no contact.
  RealT dt = 1.0;
  tribol::update( 1, 1.0, dt );

  // there still should be an active pair even with no contact...
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( 0 );
  EXPECT_EQ( 1, cs.getNumActivePairs() );

  // ...but the pair should have zero force
  for ( int i = 0; i < numVerts; ++i ) {
    EXPECT_NEAR( fz1[i], 0.0, 1.e-10 );
    EXPECT_NEAR( fz2[i], 0.0, 1.e-10 );
  }

  // Set residual_gap to 0.15. Now gap (0.1) < residual_gap (0.15), so contact.
  RealT residual_gap = 0.15;
  tribol::setResidualGap( 0, residual_gap );

  tribol::update( 2, 2.0, dt );

  EXPECT_EQ( 1, cs.getNumActivePairs() );

  // Check forces.
  // actual gap g = 0.1.
  // pressure p = (g - residual_gap) * effective_penalty = (0.1 - 0.15) * 0.5 = -0.025
  // Force on mesh 1 nodes: repulsion points DOWN (-z).
  // Total force = p * A = -0.025 * 1.0 = -0.025.
  // Distributed to 4 nodes: -0.025 / 4 = -0.00625.

  for ( int i = 0; i < numVerts; ++i ) {
    EXPECT_NEAR( fz1[i], -0.00625, 1.e-10 );
    EXPECT_NEAR( fz2[i], 0.00625, 1.e-10 );
  }
}

TEST_F( ResidualGapTest, mortar_residual_gap )
{
  // Setup two 3D quads with a 0.1 gap
  constexpr int numVerts = 4;
  constexpr int numElems = 1;
  RealT x1[numVerts] = { 0.0, 1.0, 1.0, 0.0 };
  RealT y1[numVerts] = { 0.0, 0.0, 1.0, 1.0 };
  RealT z1[numVerts] = { 0.0, 0.0, 0.0, 0.0 };

  RealT x2[numVerts] = { 0.0, 1.0, 1.0, 0.0 };
  RealT y2[numVerts] = { 0.0, 0.0, 1.0, 1.0 };
  RealT z2[numVerts] = { 0.1, 0.1, 0.1, 0.1 };

  tribol::IndexT conn1[numVerts] = { 0, 1, 2, 3 };
  tribol::IndexT conn2[numVerts] = { 0, 3, 2, 1 };

  tribol::registerMesh( 0, numElems, numVerts, &conn1[0], (int)( tribol::LINEAR_QUAD ), &x1[0], &y1[0], &z1[0],
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, numElems, numVerts, &conn2[0], (int)( tribol::LINEAR_QUAD ), &x2[0], &y2[0], &z2[0],
                        tribol::MemorySpace::Host );

  RealT fx1[numVerts] = { 0., 0., 0., 0. };
  RealT fy1[numVerts] = { 0., 0., 0., 0. };
  RealT fz1[numVerts] = { 0., 0., 0., 0. };
  RealT fx2[numVerts] = { 0., 0., 0., 0. };
  RealT fy2[numVerts] = { 0., 0., 0., 0. };
  RealT fz2[numVerts] = { 0., 0., 0., 0. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], &fz1[0] );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], &fz2[0] );

  RealT gaps[numVerts] = { 0., 0., 0., 0. };
  RealT pressures[numVerts] = { 0., 0., 0., 0. };
  tribol::registerMortarGaps( 1, gaps );
  tribol::registerMortarPressures( 1, pressures );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::SINGLE_MORTAR,
                                  tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setLagrangeMultiplierOptions( 0, tribol::ImplicitEvalMode::MORTAR_RESIDUAL );

  // Default residual_gap is 0.0. Gap is 0.1.
  RealT dt = 1.0;
  tribol::update( 1, 1.0, dt );

  auto& cs = tribol::CouplingSchemeManager::getInstance().at( 0 );
  EXPECT_EQ( 1, cs.getNumActivePairs() );

  // Nodal gaps in mortar are integrated: 0.1 * 0.25 = 0.025
  for ( int i = 0; i < numVerts; ++i ) {
    EXPECT_NEAR( gaps[i], 0.025, 1.e-10 );
  }

  // Set residual_gap to 0.15.
  RealT residual_gap = 0.15;
  tribol::setResidualGap( 0, residual_gap );

  // Clear gaps to avoid accumulation in test
  for ( int i = 0; i < numVerts; ++i ) gaps[i] = 0.0;

  tribol::update( 2, 2.0, dt );

  // Kinematic gap is (0.1 - 0.15) = -0.05. Integrated: -0.05 * 0.25 = -0.0125
  for ( int i = 0; i < numVerts; ++i ) {
    EXPECT_NEAR( gaps[i], -0.0125, 1.e-10 );
  }
}

TEST_P( ResidualGapBinningTest, residual_gap_expands_search_bounds )
{
  constexpr int numVerts = 4;
  constexpr int numElems = 1;
  constexpr RealT geometric_gap = 4.1;
  constexpr RealT residual_gap = 4.2;

  RealT x1[numVerts] = { 0.0, 1.0, 1.0, 0.0 };
  RealT y1[numVerts] = { 0.0, 0.0, 1.0, 1.0 };
  RealT z1[numVerts] = { 0.0, 0.0, 0.0, 0.0 };

  RealT x2[numVerts] = { 0.0, 1.0, 1.0, 0.0 };
  RealT y2[numVerts] = { 0.0, 0.0, 1.0, 1.0 };
  RealT z2[numVerts] = { geometric_gap, geometric_gap, geometric_gap, geometric_gap };

  tribol::IndexT conn1[numVerts] = { 0, 1, 2, 3 };
  tribol::IndexT conn2[numVerts] = { 0, 3, 2, 1 };

  tribol::registerMesh( 0, numElems, numVerts, conn1, static_cast<int>( tribol::LINEAR_QUAD ), x1, y1, z1,
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, numElems, numVerts, conn2, static_cast<int>( tribol::LINEAR_QUAD ), x2, y2, z2,
                        tribol::MemorySpace::Host );

  RealT fx1[numVerts] = { 0.0, 0.0, 0.0, 0.0 };
  RealT fy1[numVerts] = { 0.0, 0.0, 0.0, 0.0 };
  RealT fz1[numVerts] = { 0.0, 0.0, 0.0, 0.0 };
  RealT fx2[numVerts] = { 0.0, 0.0, 0.0, 0.0 };
  RealT fy2[numVerts] = { 0.0, 0.0, 0.0, 0.0 };
  RealT fz2[numVerts] = { 0.0, 0.0, 0.0, 0.0 };

  tribol::registerNodalResponse( 0, fx1, fy1, fz1 );
  tribol::registerNodalResponse( 1, fx2, fy2, fz2 );
  tribol::setKinematicConstantPenalty( 0, 1.0 );
  tribol::setKinematicConstantPenalty( 1, 1.0 );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, GetParam(),
                                  tribol::ExecutionMode::Sequential );
  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setBinningProximityScale( 0, 2.0 );
  tribol::setResidualGap( 0, residual_gap );

  RealT dt = 1.0;
  tribol::update( 1, 1.0, dt );

  auto& cs = tribol::CouplingSchemeManager::getInstance().at( 0 );
  EXPECT_EQ( 1, cs.getNumActivePairs() );
}

INSTANTIATE_TEST_SUITE_P( tribol, ResidualGapBinningTest,
                          testing::Values( tribol::BINNING_GRID, tribol::BINNING_BVH ) );

int main( int argc, char* argv[] )
{
  int result = 0;
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  result = RUN_ALL_TESTS();
  return result;
}
