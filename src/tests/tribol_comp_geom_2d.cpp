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
 protected:
  void SetUp() override {}

  void SetupAndUpdate2DHostProblem( tribol::IndexT* conn_1, RealT* x_1, RealT* y_1, RealT* fx_1, RealT* fy_1,
                                    RealT penalty_1, tribol::IndexT* conn_2, RealT* x_2, RealT* y_2, RealT* fx_2,
                                    RealT* fy_2, RealT penalty_2, tribol::ContactMode c_mode,
                                    tribol::ContactCase c_case, tribol::ContactMethod c_method,
                                    tribol::ContactModel c_model, tribol::EnforcementMethod c_enfrc,
                                    tribol::BinningMethod c_binning,
                                    tribol::KinematicPenaltyCalculation c_penalty_calc )
  {
    tribol::registerMesh( 0, 1, 2, conn_1, (int)( tribol::LINEAR_EDGE ), x_1, y_1, nullptr, tribol::MemorySpace::Host );
    tribol::registerMesh( 1, 1, 2, conn_2, (int)( tribol::LINEAR_EDGE ), x_2, y_2, nullptr, tribol::MemorySpace::Host );

    tribol::registerNodalResponse( 0, fx_1, fy_1, nullptr );
    tribol::registerNodalResponse( 1, fx_2, fy_2, nullptr );

    tribol::setKinematicConstantPenalty( 0, penalty_1 );
    tribol::setKinematicConstantPenalty( 1, penalty_2 );

    tribol::registerCouplingScheme( 0, 0, 1, c_mode, c_case, c_method, c_model, c_enfrc, c_binning,
                                    tribol::ExecutionMode::Sequential );

    tribol::setPenaltyOptions( 0, tribol::KINEMATIC, c_penalty_calc );
    tribol::setContactAreaFrac( 0, 1.e-4 );

    RealT dt = 1.;
    int update_err = tribol::update( 1, 1., dt );

    EXPECT_EQ( update_err, 0 );
  }

  void TearDown() override {}

 protected:
};

// TESTS
TEST_F( CompGeomTest, common_plane_full_interpen_no_overlap )
{
  // this is a configuration from testing that is/was producing an overlap for
  // non-overlapping edges, which in turn produced negative basis function
  // evaluations
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  // this geometry has two faces that have "passed through" one another, but
  // don't have a positive area of overlap.
  x1[0] = 0.324552;
  x1[1] = 0.16206;
  y1[0] = 0.625596;
  y1[1] = 0.722646;

  x2[0] = 4.59227e-17;
  x2[1] = 0.161705;
  y2[0] = 0.752178;
  y2[1] = 0.72276;

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.0;
  RealT penalty2 = 1.0;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 0, couplingScheme->getNumActivePairs() );
}

TEST_F( CompGeomTest, common_plane_coincident_vertices_full_overlap )
{
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  // this geometry is in contact with coincident vertices when
  // projected onto the common plane. Note, the coincident vertices
  // are up to some tolerance in the CG routines. Here, we make the
  // x-components for edge 2 vertices "nearly" 0 and "nearly" 1. In fact,
  // x-coord for vertex 1 of edge 2 is slightly MORE than 0 and x-coord
  // for vertex 2 of edge 2 is slight LESS than 1.0, which means that
  // edge 2 technically lies entirely inside edge 1 (up to some tolerance).
  x1[0] = 1.0;
  x1[1] = 0.0;
  y1[0] = 0.0;
  y1[1] = 0.0;

  x2[0] = 1.e-12;
  x2[1] = 0.999999;
  y2[0] = -0.1;
  y2[1] = -0.1;

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.;
  RealT penalty2 = 1.;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = couplingScheme->getCompGeom().getCommonPlane( 0 );
  // check overlap area. The analytic overlap is the difference in the x-coords
  // of edge 2. See note at vertex coordinate initialization at the top of this
  // test regarding edge 2 lying inside edge 1 (up to some tolerance).
  EXPECT_NEAR( plane.m_area, x2[1] - x2[0], 1.e-10 );
}

TEST_F( CompGeomTest, common_plane_conforming_separation )
{
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  // this geometry is in contact with coincident vertices when
  // projected onto the common plane.
  x1[0] = 1.0;
  x1[1] = 0.0;
  y1[0] = 0.0;
  y1[1] = 0.0;

  x2[0] = 0.;
  x2[1] = 1.;
  y2[0] = 0.1;
  y2[1] = 0.1;

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.;
  RealT penalty2 = 1.;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = couplingScheme->getCompGeom().getCommonPlane( 0 );
  EXPECT_NEAR( y2[0], plane.m_gap, 1.e-10 );
}

TEST_F( CompGeomTest, common_plane_coincident_vertex_no_overlap )
{
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  // this geometry has a pair of 'nearly' coincident vertices that should
  // produce NO positive area of overlap. Note: the actual overlap is
  // less than the contact area fraction set by tribol::setContactAreaFrac() below.
  x1[0] = 1.0;
  x1[1] = 0.0;
  y1[0] = 0.0;
  y1[1] = 0.0;

  x2[0] = -1.;
  x2[1] = 1.e-8;
  y2[0] = -0.1;
  y2[1] = -0.1;

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.;
  RealT penalty2 = 1.;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 0, couplingScheme->getNumActivePairs() );
}

TEST_F( CompGeomTest, common_plane_nearly_coincident_vertex_positive_overlap )
{
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  // This geometry has a face-pair with a set of 'nearly' coincident vertices, but a
  // positive area of overlap that is greater than the contact area frac used in
  // computing an overlap area threshold. As a result, this face-pair should be
  // in contact
  x1[0] = 1.0;
  x1[1] = 0.0;
  y1[0] = 0.0;
  y1[1] = 0.0;

  x2[0] = -1.;
  x2[1] = 1.e-4;
  y2[0] = -0.1;
  y2[1] = -0.1;

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.;
  RealT penalty2 = 1.;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = couplingScheme->getCompGeom().getCommonPlane( 0 );
  EXPECT_NEAR( plane.m_area, x2[1] - x1[1], 1.e-10 );
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
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  x1[0] = 1.0;
  x1[1] = 0.0;
  y1[0] = 0.0;
  y1[1] = 0.0;

  x2[0] = 0.;
  x2[1] = 1.;
  y2[0] = -0.1;
  y2[1] = 0.1;

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.;
  RealT penalty2 = 1.;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = couplingScheme->getCompGeom().getCommonPlane( 0 );

  EXPECT_EQ( plane.m_fullOverlap, false );

  // compute the length and height of the interpen portion of edge 2
  RealT length = x1[0] - x1[1];
  RealT height = y2[1] - y2[0];

  // compute the angle between the interpen portions of the edges
  RealT theta = std::atan( height / length );
  RealT half_theta = 0.5 * theta;

  // compute and check the overlap area
  RealT computed_area = 0.5 * length * std::cos( half_theta );
  EXPECT_NEAR( plane.m_area, computed_area, 1.e-10 );
}

TEST_F( CompGeomTest, common_plane_interpen_check_2 )
{
  // This tests checks the interpen overlap code path for a symmetric X-like interface pair
  // configuration where the rotated edge is rotated by some +/- epsilon in the y-direction
  //
  //                *
  //             *  epsilon
  //    ------o------
  //       *
  //    *
  //
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  x1[0] = 1.0;
  x1[1] = 0.0;
  y1[0] = 0.0;
  y1[1] = 0.0;

  // NOTE: interpen overlap is still detected with epsilon = 1.e-12; If decreased to 1.e-15
  // then a full overlap is detected as the two faces are numerically on top of one another
  RealT epsilon = 1.e-12;
  x2[0] = 0.;
  x2[1] = 1.;
  y2[0] = 0. - epsilon;
  y2[1] = 0. + epsilon;

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.;
  RealT penalty2 = 1.;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = couplingScheme->getCompGeom().getCommonPlane( 0 );

  EXPECT_EQ( plane.m_fullOverlap, false );

  // compute the length and height of the interpen portion of edge 2
  RealT length = x1[0] - x1[1];
  RealT height = y2[1] - y2[0];

  // compute the angle between the interpen portions of the edges
  RealT theta = std::atan( height / length );
  RealT half_theta = 0.5 * theta;

  // compute and check the overlap area
  RealT computed_area = 0.5 * length * std::cos( half_theta );
  EXPECT_NEAR( plane.m_area, computed_area, 1.e-10 );
}

TEST_F( CompGeomTest, common_plane_interpen_check_3 )
{
  // This tests checks the interpen overlap code path for an unsymmetric X-like interface pair
  // configuration
  //                 *
  //              *
  //    --------o----
  //          *
  //        *
  //
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  x1[0] = 1.0;
  x1[1] = 0.0;
  y1[0] = 0.0;
  y1[1] = 0.0;

  RealT x_shift = 0.25;
  x2[0] = 0. + x_shift;
  x2[1] = 1. + x_shift;
  y2[0] = -0.1;
  y2[1] = 0.1;

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.;
  RealT penalty2 = 1.;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = couplingScheme->getCompGeom().getCommonPlane( 0 );

  EXPECT_EQ( plane.m_fullOverlap, false );

  // compute the point at which edge 2 intersects edge 1
  RealT intercept_point = 0.5 * ( x1[0] - x1[1] ) + x_shift;

  // compute the height and length of the interpen portion of edge 2
  RealT length = intercept_point - x2[0];
  RealT height = y1[0] - y2[0];

  // compute the angle between the interpen portions of edge 1 and 2
  RealT theta = std::atan( height / length );

  // compute and check the overlap area
  RealT hyp = length / std::cos( theta );
  RealT half_theta = 0.5 * theta;
  RealT computed_area = hyp * std::cos( half_theta );
  EXPECT_NEAR( plane.m_area, computed_area, 1.e-10 );
}

TEST_F( CompGeomTest, common_plane_interpen_check_4 )
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
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  x1[0] = 1.0;
  x1[1] = 0.0;
  y1[0] = 0.0;
  y1[1] = 0.0;

  x2[0] = 0.5;
  x2[1] = 1.5;
  y2[0] = -0.5;
  y2[1] = 0.5;

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.;
  RealT penalty2 = 1.;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = couplingScheme->getCompGeom().getCommonPlane( 0 );

  EXPECT_EQ( plane.m_fullOverlap, true );

  // compute and check the overlap area

  // compute the hypotenuse (i.e. length of interpen portion of edge 2)
  RealT h = 0.5 / std::cos( 45 * M_PI / 180 );

  // compute the length as projected onto the common plane
  RealT h_bar = h * std::cos( 45 * M_PI / 180 / 2 );

  // check the overlap area
  RealT computed_area = h_bar;
  EXPECT_NEAR( plane.m_area, computed_area, 1.e-10 );
}

TEST_F( CompGeomTest, common_plane_interpen_check_5 )
{
  // This tests a configuration where there is interpenetration with an edge-edge
  // intersection that lies at a vertex on each edge. This configuration will trigger
  // a full overlap calculation based on the current 2D cg design
  //
  //
  //    ------------o
  //             *
  //          *
  //       *
  //    *
  //
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  x1[0] = 1.0;
  x1[1] = 0.0;
  y1[0] = 0.0;
  y1[1] = 0.0;

  x2[0] = 0.;
  x2[1] = 1.0;
  y2[0] = -0.1;
  y2[1] = 0.;

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.;
  RealT penalty2 = 1.;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = couplingScheme->getCompGeom().getCommonPlane( 0 );

  EXPECT_EQ( plane.m_fullOverlap, true );

  // compute and check the overlap area
  RealT length = x2[1] - x2[0];
  RealT height = y2[1] - y2[0];

  // compute the angle between the interpen portions of the edges
  RealT theta = std::atan( height / length );
  RealT half_theta = 0.5 * theta;

  RealT computed_area = length * std::cos( half_theta );
  EXPECT_NEAR( plane.m_area, computed_area, 1.e-10 );
}

TEST_F( CompGeomTest, common_plane_interpen_check_6 )
{
  // This tests a configuration where there is interpenetration with an edge-edge
  // intersection that lies NEARLY at a vertex on each edge, but outside of the
  // length tolerance that would collapse the intersection point to the vertex
  // and trigger a full overlap; thus, we have an interpen overlap.
  //
  //
  //                 * epsilon
  //    -----------o-
  //             *
  //          *
  //       *
  //    *
  //
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  x1[0] = 1.0;
  x1[1] = 0.0;
  y1[0] = 0.0;
  y1[1] = 0.0;

  RealT epsilon = 1.e-7;
  x2[0] = 0.;
  x2[1] = 1. + epsilon;
  y2[0] = -0.1;
  y2[1] = 0. + epsilon;

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.;
  RealT penalty2 = 1.;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = couplingScheme->getCompGeom().getCommonPlane( 0 );

  EXPECT_EQ( plane.m_fullOverlap, false );

  // compute and check the overlap area
  RealT length = x2[1] - x2[0];
  RealT height = y2[1] - y2[0];

  // compute the angle between the interpen portions of the edges
  RealT theta = std::atan( height / length );
  RealT half_theta = 0.5 * theta;

  RealT computed_area = length * std::cos( half_theta );
  EXPECT_NEAR( plane.m_area, computed_area, 1.e-6 );
}

TEST_F( CompGeomTest, common_plane_interpen_check_7 )
{
  // This tests a configuration where there is interpenetration with an edge-edge
  // intersection that lies NEARLY at a vertex on each edge. This configuration will trigger
  // an full overlap calculation if the intersection point is within some length tolerance
  // of the vertex. Here, we are close enough that the point is "collapsed" to the vertex,
  // which in turn triggers a full overlap
  //
  //                 * epsilon
  //    -----------o-
  //             *
  //          *
  //       *
  //    *
  //
  //
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  x1[0] = 1.0;
  x1[1] = 0.0;
  y1[0] = 0.0;
  y1[1] = 0.0;

  RealT epsilon = 1.e-12;
  x2[0] = 0.;
  x2[1] = 1. - epsilon;
  y2[0] = -0.1;
  y2[1] = 0. + epsilon;

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.;
  RealT penalty2 = 1.;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = couplingScheme->getCompGeom().getCommonPlane( 0 );

  EXPECT_EQ( plane.m_fullOverlap, true );

  // compute and check the overlap area
  RealT length = x2[1] - x2[0];
  RealT height = y2[1] - y2[0];

  // compute the angle between the interpen portions of the edges
  RealT theta = std::atan( height / length );
  RealT half_theta = 0.5 * theta;

  RealT computed_area = length * std::cos( half_theta );
  EXPECT_NEAR( plane.m_area, computed_area, 1.e-6 );
}

TEST_F( CompGeomTest, common_plane_interpen_check_8 )
{
  // This test checks two faces that are in separation up to some epsilon.
  // There should be one active pair due to proximity, but not in contact
  //
  // **************
  // -------------- epsilon
  //
  //
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  x1[0] = 1.0;
  x1[1] = 0.0;
  y1[0] = 0.0;
  y1[1] = 0.0;

  // Note: an epsilon even of 1.e-15 will not register the pair as in contact
  RealT epsilon = 1.e-15;
  x2[0] = 0.;
  x2[1] = 1.;
  y2[0] = 0. + epsilon;
  y2[1] = 0. + epsilon;

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.;
  RealT penalty2 = 1.;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );
  auto& plane = couplingScheme->getCompGeom().getCommonPlane( 0 );
  EXPECT_EQ( plane.m_inContact, false );
}

TEST_F( CompGeomTest, common_plane_interpen_check_9 )
{
  // This test checks two faces that are in full interpen up to some epsilon.
  //
  // -------------- epsilon
  // **************
  //
  //
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  x1[0] = 1.0;
  x1[1] = 0.0;
  y1[0] = 0.0;
  y1[1] = 0.0;

  // NOTE: the CG will still detect this pair as in contact with an epsilon of 1.e-12. If I go to 1.e-15,
  // for example, then the pair will be seen as perfectly on top of one another and not marked as in contact
  RealT epsilon = 1.e-12;
  x2[0] = 0.;
  x2[1] = 1.;
  y2[0] = 0. - epsilon;
  y2[1] = 0. - epsilon;

  tribol::IndexT conn1[2] = { 0, 1 };
  tribol::IndexT conn2[2] = { 0, 1 };

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.;
  RealT penalty2 = 1.;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );
  auto& plane = couplingScheme->getCompGeom().getCommonPlane( 0 );
  EXPECT_EQ( plane.m_inContact, true );
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

  EXPECT_NEAR( cxProj1[0], 0.7385950, 1.e-6 );
  EXPECT_NEAR( cxProj1[1], 0.0915028, 1.e-6 );
  EXPECT_NEAR( cxProj2[0], 0.7385910, 1.e-6 );
  EXPECT_NEAR( cxProj2[1], 0.0915022, 1.e-6 );

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

  RealT fx1[2] = { 0., 0. };
  RealT fy1[2] = { 0., 0. };
  RealT fx2[2] = { 0., 0. };
  RealT fy2[2] = { 0., 0. };

  RealT penalty1 = 1.;
  RealT penalty2 = 1.;

  SetupAndUpdate2DHostProblem( &conn1[0], &x1[0], &y1[0], &fx1[0], &fy1[0], penalty1, &conn2[0], &x2[0], &y2[0],
                               &fx2[0], &fy2[0], penalty2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                               tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                               tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  auto& plane = couplingScheme->getCompGeom().getCommonPlane( 0 );

  EXPECT_EQ( plane.m_fullOverlap, true );
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

  EXPECT_NEAR( cxProj1[0], cx[0], 1.e-6 );
  EXPECT_NEAR( cxProj1[1], cx[1], 1.e-6 );
  EXPECT_NEAR( cxProj2[0], cx[0], 1.e-6 );
  EXPECT_NEAR( cxProj2[1], cx[1], 1.e-6 );
}

TEST_F( CompGeomTest, point_in_edge_1 )
{
  constexpr int numVerts = 2;
  RealT x[numVerts];
  RealT y[numVerts];
  RealT xp, yp;

  x[0] = 0.;
  x[1] = 1.5;

  y[0] = 0.75;
  y[1] = 0.75;

  // point at one vertex
  xp = 0.;
  yp = 0.75;
  EXPECT_EQ( tribol::IsPointInEdge( x, y, xp, yp ), true );

  // point at other vertex
  xp = 1.5;
  yp = 0.75;
  EXPECT_EQ( tribol::IsPointInEdge( x, y, xp, yp ), true );

  // point somewhere inside but away from both edge vertices
  xp = 0.85;
  yp = 0.75;
  EXPECT_EQ( tribol::IsPointInEdge( x, y, xp, yp ), true );

  // point just barely inside the first vertex
  RealT epsilon = 1.e-12;
  xp = 0. + epsilon;
  yp = 0.75;
  EXPECT_EQ( tribol::IsPointInEdge( x, y, xp, yp ), true );

  // point just barely inside the second
  xp = 1.5 - epsilon;
  yp = 0.75;
  EXPECT_EQ( tribol::IsPointInEdge( x, y, xp, yp ), true );

  // point just barely outside first vertex
  xp = 0. - epsilon;
  yp = 0.75;
  EXPECT_EQ( tribol::IsPointInEdge( x, y, xp, yp ), false );

  // point just barely outside second vertex
  xp = 1.5 + epsilon;
  yp = 0.75;
  EXPECT_EQ( tribol::IsPointInEdge( x, y, xp, yp ), false );

  // point not even close
  xp = 2.5;
  yp = 0.75;
  EXPECT_EQ( tribol::IsPointInEdge( x, y, xp, yp ), false );
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
