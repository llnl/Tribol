// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/interface/tribol.hpp"
#include "tribol/utils/TestUtils.hpp"
#include "tribol/utils/Math.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/geom/CompGeom.hpp"

// Axom includes
#include "axom/slic.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// gtest includes
#include "gtest/gtest.h"

// c++ includes
#include <cmath>  // std::abs, std::cos, std::sin
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>

using RealT = tribol::RealT;

/*!
 * Test fixture class with some setup necessary to test
 * the computational geometry. This test does not have a specific
 * check that will make it pass or fail (yet), instead this is
 * simply used to drive the computational geometry engine
 * and to interrogate SLIC output printed to screen
 */
class CompGeomTest : public ::testing::Test {
 public:
  tribol::TestMesh m_mesh;

 protected:
  void SetUp() override {}

  void TearDown() override
  {
    // call clear() on mesh object to be safe
    this->m_mesh.clear();
  }

 protected:
};

// TESTS
TEST_F( CompGeomTest, common_plane_full_interpen_no_overlap )
{
  // this is a configuration from testing that is/was producing an overlap for
  // non-overlapping edges, which in turn produced negative basis function
  // evaluations
  constexpr int dim = 2;
  constexpr int numVerts = 2;
  RealT xy1[dim * numVerts];
  RealT xy2[dim * numVerts];

  // this geometry has two faces that have "passed through" one another, but
  // don't have a positive area of overlap.
  xy1[0] = 0.324552;
  xy1[1] = 0.625596;
  xy1[2] = 0.16206;
  xy1[3] = 0.722646;

  xy2[0] = 4.59227e-17;
  xy2[1] = 0.752178;
  xy2[2] = 0.161705;
  xy2[3] = 0.72276;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  for ( int i = 0; i < numVerts; ++i ) {
    x1[i] = xy1[i * dim];
    y1[i] = xy1[i * dim + 1];
    x2[i] = xy2[i * dim];
    y2[i] = xy2[i * dim + 1];
  }

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  tribol::registerMesh( 0, 1, 2, &conn1[0], (int)( tribol::LINEAR_EDGE ), &x1[0], &y1[0], nullptr,
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, 1, 2, &conn2[0], (int)( tribol::LINEAR_EDGE ), &x2[0], &y2[0], nullptr,
                        tribol::MemorySpace::Host );

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], nullptr );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], nullptr );

  tribol::setKinematicConstantPenalty( 0, 1. );
  tribol::setKinematicConstantPenalty( 1, 1. );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setContactAreaFrac( 0, 1.e-4 );

  RealT dt = 1.;
  int update_err = tribol::update( 1, 1., dt );

  EXPECT_EQ( update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 0, couplingScheme->getNumActivePairs() );
}

TEST_F( CompGeomTest, common_plane_coincident_vertices_full_overlap )
{
  constexpr int dim = 2;
  constexpr int numVerts = 2;
  RealT xy1[dim * numVerts];
  RealT xy2[dim * numVerts];

  // this geometry is in contact with coincident vertices when
  // projected onto the common plane.
  xy1[0] = 1.0;
  xy1[1] = 0.0;
  xy1[2] = 0.0;
  xy1[3] = 0.0;

  xy2[0] = 1.e-12;
  xy2[1] = -0.1;
  xy2[2] = 0.999999;
  xy2[3] = -0.1;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  for ( int i = 0; i < numVerts; ++i ) {
    x1[i] = xy1[i * dim];
    y1[i] = xy1[i * dim + 1];
    x2[i] = xy2[i * dim];
    y2[i] = xy2[i * dim + 1];
  }

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  tribol::registerMesh( 0, 1, 2, &conn1[0], (int)( tribol::LINEAR_EDGE ), &x1[0], &y1[0], nullptr,
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, 1, 2, &conn2[0], (int)( tribol::LINEAR_EDGE ), &x2[0], &y2[0], nullptr,
                        tribol::MemorySpace::Host );

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], nullptr );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], nullptr );

  tribol::setKinematicConstantPenalty( 0, 1. );
  tribol::setKinematicConstantPenalty( 1, 1. );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setContactAreaFrac( 0, 1.e-4 );

  RealT dt = 1.;
  int update_err = tribol::update( 1, 1., dt );

  EXPECT_EQ( update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = static_cast<const tribol::CommonPlanePair&>( couplingScheme->getContactPlanePair( 0 ) );
  RealT diff = std::abs( ( xy2[2] - xy2[0] ) - plane.m_area );
  //std::cout << "coincident vertices full overlap gap: " << plane.m_gap << std::endl;
  //std::cout << "in contact" << plane.m_inContact << std::endl;
  EXPECT_LT( diff, 1.e-10 );

  //const auto mesh1 = const_cast<tribol::CouplingScheme*>( couplingScheme )->getMesh1().getView();
  //const auto mesh2 = const_cast<tribol::CouplingScheme*>( couplingScheme )->getMesh2().getView();
  //std::cout << "force1 node 0: " << mesh1.getResponse()[0][0] << ", " << mesh1.getResponse()[1][0] << std::endl;
  //std::cout << "force1 node 1: " << mesh1.getResponse()[0][1] << ", " << mesh1.getResponse()[1][1] << std::endl;

  //std::cout << "force2 node 0: " << mesh2.getResponse()[0][0] << ", " << mesh2.getResponse()[1][0] << std::endl;
  //std::cout << "force2 node 1: " << mesh2.getResponse()[0][1] << ", " << mesh2.getResponse()[1][1] << std::endl;
}

TEST_F( CompGeomTest, common_plane_conforming_separation )
{
  constexpr int dim = 2;
  constexpr int numVerts = 2;
  RealT xy1[dim * numVerts];
  RealT xy2[dim * numVerts];

  // this geometry is in contact with coincident vertices when
  // projected onto the common plane.
  xy1[0] = 1.0;
  xy1[1] = 0.0;
  xy1[2] = 0.0;
  xy1[3] = 0.0;

  xy2[0] = 0.;
  xy2[1] = 0.1;
  xy2[2] = 1.;
  xy2[3] = 0.1;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  for ( int i = 0; i < numVerts; ++i ) {
    x1[i] = xy1[i * dim];
    y1[i] = xy1[i * dim + 1];
    x2[i] = xy2[i * dim];
    y2[i] = xy2[i * dim + 1];
  }

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  tribol::registerMesh( 0, 1, 2, &conn1[0], (int)( tribol::LINEAR_EDGE ), &x1[0], &y1[0], nullptr,
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, 1, 2, &conn2[0], (int)( tribol::LINEAR_EDGE ), &x2[0], &y2[0], nullptr,
                        tribol::MemorySpace::Host );

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], nullptr );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], nullptr );

  tribol::setKinematicConstantPenalty( 0, 1. );
  tribol::setKinematicConstantPenalty( 1, 1. );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setContactAreaFrac( 0, 1.e-4 );

  RealT dt = 1.;
  int update_err = tribol::update( 1, 1., dt );

  EXPECT_EQ( update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = static_cast<const tribol::CommonPlanePair&>( couplingScheme->getContactPlanePair( 0 ) );
  RealT gap_diff = std::abs( 0.1 - plane.m_gap );
  //std::cout << "conforming separation gap: " << plane.m_gap << std::endl;
  //std::cout << "in contact" << plane.m_inContact << std::endl;
  EXPECT_LT( gap_diff, 1.e-10 );
  //const auto mesh1 = const_cast<tribol::CouplingScheme*>( couplingScheme )->getMesh1().getView();
  //const auto mesh2 = const_cast<tribol::CouplingScheme*>( couplingScheme )->getMesh2().getView();
  //std::cout << "force1 node 0: " << mesh1.getResponse()[0][0] << ", " << mesh1.getResponse()[1][0] << std::endl;
  //std::cout << "force1 node 1: " << mesh1.getResponse()[0][1] << ", " << mesh1.getResponse()[1][1] << std::endl;

  //std::cout << "force1 node 0: " << mesh2.getResponse()[0][0] << ", " << mesh2.getResponse()[1][0] << std::endl;
  //std::cout << "force1 node 1: " << mesh2.getResponse()[0][1] << ", " << mesh2.getResponse()[1][1] << std::endl;
}

TEST_F( CompGeomTest, common_plane_coincident_vertex_no_overlap )
{
  constexpr int dim = 2;
  constexpr int numVerts = 2;
  RealT xy1[dim * numVerts];
  RealT xy2[dim * numVerts];

  // this geometry has a pair of 'nearly' coincident vertices that should
  // produce NO positive area of overlap. Note: the actual overlap is
  // less than the contact area fraction set by tribol::setContactAreaFrac() below.
  xy1[0] = 1.0;
  xy1[1] = 0.0;
  xy1[2] = 0.0;
  xy1[3] = 0.0;

  xy2[0] = -1.;
  xy2[1] = -0.1;
  xy2[2] = 1.e-8;
  xy2[3] = -0.1;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  for ( int i = 0; i < numVerts; ++i ) {
    x1[i] = xy1[i * dim];
    y1[i] = xy1[i * dim + 1];
    x2[i] = xy2[i * dim];
    y2[i] = xy2[i * dim + 1];
  }

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  tribol::registerMesh( 0, 1, 2, &conn1[0], (int)( tribol::LINEAR_EDGE ), &x1[0], &y1[0], nullptr,
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, 1, 2, &conn2[0], (int)( tribol::LINEAR_EDGE ), &x2[0], &y2[0], nullptr,
                        tribol::MemorySpace::Host );

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], nullptr );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], nullptr );

  tribol::setKinematicConstantPenalty( 0, 1. );
  tribol::setKinematicConstantPenalty( 1, 1. );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setContactAreaFrac( 0, 1.e-4 );

  RealT dt = 1.;
  int update_err = tribol::update( 1, 1., dt );

  EXPECT_EQ( update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 0, couplingScheme->getNumActivePairs() );
}

TEST_F( CompGeomTest, common_plane_nearly_coincident_vertex_positive_overlap )
{
  constexpr int dim = 2;
  constexpr int numVerts = 2;
  RealT xy1[dim * numVerts];
  RealT xy2[dim * numVerts];

  // This geometry has a face-pair with a set of 'nearly' coincident vertices, but a
  // positive area of overlap that is greater than the contact area frac used in
  // computing an overlap area threshold. As a result, this face-pair should be
  // in contact
  xy1[0] = 1.0;
  xy1[1] = 0.0;
  xy1[2] = 0.0;
  xy1[3] = 0.0;

  xy2[0] = -1.;
  xy2[1] = -0.1;
  xy2[2] = 1.e-4;
  xy2[3] = -0.1;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  for ( int i = 0; i < numVerts; ++i ) {
    x1[i] = xy1[i * dim];
    y1[i] = xy1[i * dim + 1];
    x2[i] = xy2[i * dim];
    y2[i] = xy2[i * dim + 1];
  }

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  tribol::registerMesh( 0, 1, 2, &conn1[0], (int)( tribol::LINEAR_EDGE ), &x1[0], &y1[0], nullptr,
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, 1, 2, &conn2[0], (int)( tribol::LINEAR_EDGE ), &x2[0], &y2[0], nullptr,
                        tribol::MemorySpace::Host );

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], nullptr );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], nullptr );

  tribol::setKinematicConstantPenalty( 0, 1. );
  tribol::setKinematicConstantPenalty( 1, 1. );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setContactAreaFrac( 0, 1.e-12 );

  RealT dt = 1.;
  int update_err = tribol::update( 1, 1., dt );

  EXPECT_EQ( update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = static_cast<const tribol::CommonPlanePair&>( couplingScheme->getContactPlanePair( 0 ) );
  RealT diff = std::abs( ( xy2[2] - xy1[2] ) - plane.m_area );
  EXPECT_LT( diff, 1.e-10 );
  //const auto mesh1 = const_cast<tribol::CouplingScheme*>( couplingScheme )->getMesh1().getView();
  //const auto mesh2 = const_cast<tribol::CouplingScheme*>( couplingScheme )->getMesh2().getView();
  //std::cout << "force1 node 0: " << mesh1.getResponse()[0][0] << ", " << mesh1.getResponse()[1][0] << std::endl;
  //std::cout << "force1 node 1: " << mesh1.getResponse()[0][1] << ", " << mesh1.getResponse()[1][1] << std::endl;

  //std::cout << "force1 node 0: " << mesh2.getResponse()[0][0] << ", " << mesh2.getResponse()[1][0] << std::endl;
  //std::cout << "force1 node 1: " << mesh2.getResponse()[0][1] << ", " << mesh2.getResponse()[1][1] << std::endl;

  //std::cout << "f1 node 0: " << fx1[0] << ", " << fy1[0] << std::endl;
  //std::cout << "f1 node 1: " << fx1[1] << ", " << fy1[1] << std::endl;

  //std::cout << "f2 node 0: " << fx2[0] << ", " << fy2[0] << std::endl;
  //std::cout << "f2 node 1: " << fx2[1] << ", " << fy2[1] << std::endl;
}

TEST_F( CompGeomTest, common_plane_interpen_check_1 )
{
  // This tests checks the interpen overlap code path for a symmetric X-like interface pair
  // configuration
  //
  //                *
  //             *
  //    ------o------
  //       *
  //    *
  //
  constexpr int dim = 2;
  constexpr int numVerts = 2;
  RealT xy1[dim * numVerts];
  RealT xy2[dim * numVerts];

  xy1[0] = 1.0;
  xy1[1] = 0.0;
  xy1[2] = 0.0;
  xy1[3] = 0.0;

  xy2[0] = 0.;
  xy2[1] = -0.1;
  xy2[2] = 1.;
  xy2[3] = 0.1;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  for ( int i = 0; i < numVerts; ++i ) {
    x1[i] = xy1[i * dim];
    y1[i] = xy1[i * dim + 1];
    x2[i] = xy2[i * dim];
    y2[i] = xy2[i * dim + 1];
  }

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  tribol::registerMesh( 0, 1, 2, &conn1[0], (int)( tribol::LINEAR_EDGE ), &x1[0], &y1[0], nullptr,
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, 1, 2, &conn2[0], (int)( tribol::LINEAR_EDGE ), &x2[0], &y2[0], nullptr,
                        tribol::MemorySpace::Host );

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], nullptr );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], nullptr );

  tribol::setKinematicConstantPenalty( 0, 1. );
  tribol::setKinematicConstantPenalty( 1, 1. );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setContactAreaFrac( 0, 1.e-12 );

  RealT dt = 1.;
  int update_err = tribol::update( 1, 1., dt );

  EXPECT_EQ( update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = static_cast<const tribol::CommonPlanePair&>( couplingScheme->getContactPlanePair( 0 ) );

  EXPECT_EQ( plane.m_fullOverlap, false );
  RealT length = xy1[0] - xy1[2];
  RealT height = xy2[3] - xy2[1];
  RealT theta = std::atan( height / length );
  RealT half_theta = 0.5 * theta;
  RealT computed_area = 0.5 * length * std::cos( half_theta );
  RealT diff = std::abs( computed_area - plane.m_area );
  EXPECT_LT( diff, 1.e-10 );
  //std::cout << "contact interpen 1 gap: " << plane.m_gap << std::endl;
  //std::cout << "in contact" << plane.m_inContact << std::endl;
  //const auto mesh1 = const_cast<tribol::CouplingScheme*>( couplingScheme )->getMesh1().getView();
  //const auto mesh2 = const_cast<tribol::CouplingScheme*>( couplingScheme )->getMesh2().getView();
  //std::cout << "force1 node 0: " << mesh1.getResponse()[0][0] << ", " << mesh1.getResponse()[1][0] << std::endl;
  //std::cout << "force1 node 1: " << mesh1.getResponse()[0][1] << ", " << mesh1.getResponse()[1][1] << std::endl;

  //std::cout << "force1 node 0: " << mesh2.getResponse()[0][0] << ", " << mesh2.getResponse()[1][0] << std::endl;
  //std::cout << "force1 node 1: " << mesh2.getResponse()[0][1] << ", " << mesh2.getResponse()[1][1] << std::endl;
}

TEST_F( CompGeomTest, common_plane_interpen_check_2 )
{
  // This tests checks the interpen overlap code path for an unsymmetric X-like interface pair
  // configuration
  //                 *
  //              * 
  //    --------o----
  //          *
  //        *
  //
  constexpr int dim = 2;
  constexpr int numVerts = 2;
  RealT xy1[dim * numVerts];
  RealT xy2[dim * numVerts];

  xy1[0] = 1.0;
  xy1[1] = 0.0;
  xy1[2] = 0.0;
  xy1[3] = 0.0;

  xy2[0] = 0.;
  xy2[1] = -0.1;
  xy2[2] = 1.;
  xy2[3] = 0.1;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  RealT x_shift = 0.25;
  for ( int i = 0; i < numVerts; ++i ) {
    x1[i] = xy1[i * dim];
    y1[i] = xy1[i * dim + 1];
    x2[i] = xy2[i * dim] + x_shift;
    y2[i] = xy2[i * dim + 1];
  }

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  tribol::registerMesh( 0, 1, 2, &conn1[0], (int)( tribol::LINEAR_EDGE ), &x1[0], &y1[0], nullptr,
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, 1, 2, &conn2[0], (int)( tribol::LINEAR_EDGE ), &x2[0], &y2[0], nullptr,
                        tribol::MemorySpace::Host );

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], nullptr );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], nullptr );

  tribol::setKinematicConstantPenalty( 0, 1. );
  tribol::setKinematicConstantPenalty( 1, 1. );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setContactAreaFrac( 0, 1.e-12 );

  RealT dt = 1.;
  int update_err = tribol::update( 1, 1., dt );

  EXPECT_EQ( update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = static_cast<const tribol::CommonPlanePair&>( couplingScheme->getContactPlanePair( 0 ) );

  EXPECT_EQ( plane.m_fullOverlap, false );
  RealT length = 0.75 - (xy2[0] + x_shift);
  RealT height = 0. - xy2[1];
  RealT theta = std::atan( height / length );
  RealT hyp = length / std::cos(theta); 
  RealT half_theta = 0.5 * theta;
  RealT computed_area = hyp * std::cos( half_theta );
  RealT diff = std::abs( computed_area - plane.m_area );
  EXPECT_LT( diff, 1.e-10 );

  //std::cout << "contact interpen 2 gap: " << plane.m_gap << std::endl;
  //std::cout << "in contact" << plane.m_inContact << std::endl;
  //const auto mesh1 = const_cast<tribol::CouplingScheme*>( couplingScheme )->getMesh1().getView();
  //const auto mesh2 = const_cast<tribol::CouplingScheme*>( couplingScheme )->getMesh2().getView();
  //std::cout << "force1 node 0: " << mesh1.getResponse()[0][0] << ", " << mesh1.getResponse()[1][0] << std::endl;
  //std::cout << "force1 node 1: " << mesh1.getResponse()[0][1] << ", " << mesh1.getResponse()[1][1] << std::endl;

  //std::cout << "force1 node 0: " << mesh2.getResponse()[0][0] << ", " << mesh2.getResponse()[1][0] << std::endl;
  //std::cout << "force1 node 1: " << mesh2.getResponse()[0][1] << ", " << mesh2.getResponse()[1][1] << std::endl;
}

TEST_F( CompGeomTest, common_plane_interpen_check_3 )
{
  // This tests checks the interpen overlap code path for an X-like interface pair
  // configuration where there IS interpenetration, but one edge intersects
  // at the vertex of the other edge, which should trigger a full overlap calculation
  //
  //                     *
  //                  *
  //    ------------o
  //             *
  //          *
  //
  constexpr int dim = 2;
  constexpr int numVerts = 2;
  RealT xy1[dim * numVerts];
  RealT xy2[dim * numVerts];

  xy1[0] = 1.0;
  xy1[1] = 0.0;
  xy1[2] = 0.0;
  xy1[3] = 0.0;

  xy2[0] = 0.5;
  xy2[1] = -0.5;
  xy2[2] = 1.5;
  xy2[3] = 0.5;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  for ( int i = 0; i < numVerts; ++i ) {
    x1[i] = xy1[i * dim];
    y1[i] = xy1[i * dim + 1];
    x2[i] = xy2[i * dim];
    y2[i] = xy2[i * dim + 1];
  }

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  tribol::registerMesh( 0, 1, 2, &conn1[0], (int)( tribol::LINEAR_EDGE ), &x1[0], &y1[0], nullptr,
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, 1, 2, &conn2[0], (int)( tribol::LINEAR_EDGE ), &x2[0], &y2[0], nullptr,
                        tribol::MemorySpace::Host );

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], nullptr );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], nullptr );

  tribol::setKinematicConstantPenalty( 0, 1. );
  tribol::setKinematicConstantPenalty( 1, 1. );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setContactAreaFrac( 0, 1.e-12 );

  RealT dt = 1.;
  int update_err = tribol::update( 1, 1., dt );

  EXPECT_EQ( update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = static_cast<const tribol::CommonPlanePair&>( couplingScheme->getContactPlanePair( 0 ) );

  EXPECT_EQ( plane.m_fullOverlap, true );
  RealT h = 0.5 / std::cos( 45 * M_PI / 180 );
  RealT h_bar = h * std::cos( 45 * M_PI / 180 / 2 );
  RealT computed_area = h_bar;
  RealT diff = std::abs( computed_area - plane.m_area );
  EXPECT_LT( diff, 1.e-10 );
}

TEST_F( CompGeomTest, 2d_projections_1 )
{
  constexpr int dim = 2;
  constexpr int numVerts = 2;
  RealT xy1[dim * numVerts];
  RealT xy2[dim * numVerts];

  // this geometry should be in contact, testing projections
  xy1[0] = 0.75;
  xy1[1] = 0.;
  xy1[2] = 0.727322;
  xy1[3] = 0.183039;

  xy2[0] = 0.72705;
  xy2[1] = 0.182971;
  xy2[2] = 0.75;
  xy2[3] = 0.;

  // compute face normal
  RealT faceNormal1[dim];
  RealT faceNormal2[dim];

  RealT lambdaX1 = xy1[2] - xy1[0];
  RealT lambdaY1 = xy1[3] - xy1[1];

  faceNormal1[0] = lambdaY1;
  faceNormal1[1] = -lambdaX1;

  RealT lambdaX2 = xy2[2] - xy2[0];
  RealT lambdaY2 = xy2[3] - xy2[1];

  faceNormal2[0] = lambdaY2;
  faceNormal2[1] = -lambdaX2;

  RealT cxf1[3] = { 0., 0., 0. };
  RealT cxf2[3] = { 0., 0., 0. };

  tribol::VertexAvgCentroid( xy1, dim, numVerts, cxf1[0], cxf1[1], cxf1[2] );
  tribol::VertexAvgCentroid( xy2, dim, numVerts, cxf2[0], cxf2[1], cxf2[2] );

  // average the vertex averaged centroids of each face to get a pretty good
  // estimate of the common plane centroid
  RealT cx[dim];
  cx[0] = 0.5 * ( cxf1[0] + cxf2[0] );
  cx[1] = 0.5 * ( cxf1[1] + cxf2[1] );

  RealT cxProj1[3] = { 0., 0., 0. };
  RealT cxProj2[3] = { 0., 0., 0. };

  tribol::ProjectPointToSegment( cx[0], cx[1], faceNormal1[0], faceNormal1[1], cxf1[0], cxf1[1], cxProj1[0],
                                 cxProj1[1] );
  tribol::ProjectPointToSegment( cx[0], cx[1], faceNormal2[0], faceNormal2[1], cxf2[0], cxf2[1], cxProj2[0],
                                 cxProj2[1] );

  RealT diffx1 = std::abs( cxProj1[0] - 0.738595 );
  RealT diffy1 = std::abs( cxProj1[1] - 0.0915028 );
  RealT diffx2 = std::abs( cxProj2[0] - 0.738591 );
  RealT diffy2 = std::abs( cxProj2[1] - 0.0915022 );
  EXPECT_LE( diffx1, 1.e-6 );
  EXPECT_LE( diffy1, 1.e-6 );
  EXPECT_LE( diffx2, 1.e-6 );
  EXPECT_LE( diffy2, 1.e-6 );

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  for ( int i = 0; i < numVerts; ++i ) {
    x1[i] = xy1[i * dim];
    y1[i] = xy1[i * dim + 1];
    x2[i] = xy2[i * dim];
    y2[i] = xy2[i * dim + 1];
  }

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  tribol::registerMesh( 0, 1, 2, &conn1[0], (int)( tribol::LINEAR_EDGE ), &x1[0], &y1[0], nullptr,
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, 1, 2, &conn2[0], (int)( tribol::LINEAR_EDGE ), &x2[0], &y2[0], nullptr,
                        tribol::MemorySpace::Host );

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], nullptr );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], nullptr );

  tribol::setKinematicConstantPenalty( 0, 1. );
  tribol::setKinematicConstantPenalty( 1, 1. );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setContactAreaFrac( 0, 1.e-4 );

  RealT dt = 1.;
  int update_err = tribol::update( 1, 1., dt );

  EXPECT_EQ( update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );
}

TEST_F( CompGeomTest, 2d_projections_2 )
{
  constexpr int dim = 2;
  constexpr int numVerts = 2;
  RealT xy1[dim * numVerts];
  RealT xy2[dim * numVerts];

  // face coordinates from testing
  xy1[0] = 0.75;
  xy1[1] = 0.;
  xy1[2] = 0.727322;
  xy1[3] = 0.183039;

  xy2[0] = 0.727322;
  xy2[1] = 0.183039;
  xy2[2] = 0.75;
  xy2[3] = 0.;

  // compute face normal
  RealT faceNormal1[dim];
  RealT faceNormal2[dim];

  RealT lambdaX1 = xy1[2] - xy1[0];
  RealT lambdaY1 = xy1[3] - xy1[1];

  faceNormal1[0] = lambdaY1;
  faceNormal1[1] = -lambdaX1;

  RealT lambdaX2 = xy2[2] - xy2[0];
  RealT lambdaY2 = xy2[3] - xy2[1];

  faceNormal2[0] = lambdaY2;
  faceNormal2[1] = -lambdaX2;

  RealT cxf1[3] = { 0., 0., 0. };
  RealT cxf2[3] = { 0., 0., 0. };

  tribol::VertexAvgCentroid( xy1, dim, numVerts, cxf1[0], cxf1[1], cxf1[2] );
  tribol::VertexAvgCentroid( xy2, dim, numVerts, cxf2[0], cxf2[1], cxf2[2] );

  // average the vertex averaged centroids of each face to get a pretty good
  // estimate of the common plane centroid
  RealT cx[dim];
  cx[0] = 0.5 * ( cxf1[0] + cxf2[0] );
  cx[1] = 0.5 * ( cxf1[1] + cxf2[1] );

  RealT cxProj1[3] = { 0., 0., 0. };
  RealT cxProj2[3] = { 0., 0., 0. };

  tribol::ProjectPointToSegment( cx[0], cx[1], faceNormal1[0], faceNormal1[1], cxf1[0], cxf1[1], cxProj1[0],
                                 cxProj1[1] );
  tribol::ProjectPointToSegment( cx[0], cx[1], faceNormal2[0], faceNormal2[1], cxf2[0], cxf2[1], cxProj2[0],
                                 cxProj2[1] );

  RealT diffx1 = std::abs( cxProj1[0] - cx[0] );
  RealT diffy1 = std::abs( cxProj1[1] - cx[1] );
  RealT diffx2 = std::abs( cxProj2[0] - cx[0] );
  RealT diffy2 = std::abs( cxProj2[1] - cx[1] );
  EXPECT_LE( diffx1, 1.e-6 );
  EXPECT_LE( diffy1, 1.e-6 );
  EXPECT_LE( diffx2, 1.e-6 );
  EXPECT_LE( diffy2, 1.e-6 );
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

  axom::slic::SimpleLogger logger;                 // create & initialize logger,
  tribol::SimpleMPIWrapper wrapper( argc, argv );  // initialize and finalize MPI, when applicable

  result = RUN_ALL_TESTS();

  return result;
}
