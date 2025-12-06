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

#define _USE_MATH_DEFINES
#include <cmath>  // std::abs, std::cos, std::sin

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
  bool m_isElementThicknessRegistered{ false };

  int setupAndUpdateAutoCommonPlane( const int meshId, const int csId, const int numNodes, const int numCells,
                                     RealT& dt )
  {
    // if element thickness is not registered by test, then register dummy
    // element thickness in order to use auto contact
    tribol::Array1D<tribol::RealT> element_thickness( numCells );
    if ( !m_isElementThicknessRegistered ) {
      for ( int i = 0; i < numCells; ++i ) {
        element_thickness[i] = 1.0;
      }
      tribol::registerRealElementField( meshId, tribol::ELEMENT_THICKNESS, &element_thickness[0] );
    }

    RealT *fx, *fy, *fz;
    tribol::allocRealArray( &fx, numNodes, 0. );
    tribol::allocRealArray( &fy, numNodes, 0. );
    tribol::allocRealArray( &fz, numNodes, 0. );

    tribol::registerNodalResponse( meshId, fx, fy, fz );

    RealT *vx, *vy, *vz;
    tribol::allocRealArray( &vx, numNodes, 0. );
    tribol::allocRealArray( &vy, numNodes, 0. );
    tribol::allocRealArray( &vz, numNodes, 0. );

    tribol::registerNodalVelocities( meshId, vx, vy, vz );

    tribol::registerCouplingScheme( csId, meshId, meshId, tribol::SURFACE_TO_SURFACE, tribol::AUTO,
                                    tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
                                    tribol::BINNING_CARTESIAN_PRODUCT, tribol::ExecutionMode::Sequential );

    RealT max_interpen_frac = 1.0;
    tribol::setAutoContactPenScale( csId, max_interpen_frac );

    tribol::setPenaltyOptions( csId, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT, tribol::NO_RATE_PENALTY );

    tribol::setKinematicConstantPenalty( meshId, 1.0 );

    return tribol::update( 1, 1., dt );

    delete[] fx;
    delete[] fy;
    delete[] fz;
    delete[] vx;
    delete[] vy;
    delete[] vz;
  }

 protected:
  void SetUp() override {}

  void TearDown() override
  {
    // call clear() on mesh object to be safe
    this->m_mesh.clear();
  }

 protected:
};

TEST_F( CompGeomTest, common_plane_single_element_full_overlap_check_1 )
{
  // mesh bounding box with 0.1 interpenetration gap. The contact faces
  // just have a y-shift and overlap will have nodes that lie on edge
  // segments of the opposing face
  int numElems1 = 1;
  int nElemsXM = numElems1;
  int nElemsYM = 1;
  int nElemsZM = numElems1;

  int numElems2 = 1;
  int nElemsXS = numElems2;
  int nElemsYS = 1;
  int nElemsZS = numElems2;

  int userSpecifiedNumOverlaps = 1;

  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.05;

  RealT x_min2 = 0.;
  RealT y_min2 = 0.5;  // perform 0.5 shift in y direction
  RealT z_min2 = 0.95;
  RealT x_max2 = 1.0;
  RealT y_max2 = y_min2 + 1.0;
  RealT z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  parameters.penalty_ratio = false;
  parameters.const_penalty = 1.0;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  RealT computed_gap = -( z_max1 - z_min2 );
  EXPECT_NEAR( plane.m_gap, computed_gap, 1.e-8 );
  EXPECT_NEAR( plane.m_area, 0.5, 1.e-10 );

  tribol::finalize();
}

TEST_F( CompGeomTest, common_plane_single_element_full_overlap_check_2 )
{
  // mesh bounding box with 0.1 interpenetration gap. The faces will
  // have an x and y shift such that no nodes overlap with nodes/segments of
  // the opposing face
  int numElems1 = 1;
  int nElemsXM = numElems1;
  int nElemsYM = 1;
  int nElemsZM = numElems1;

  int numElems2 = 1;
  int nElemsXS = numElems2;
  int nElemsYS = 1;
  int nElemsZS = numElems2;

  int userSpecifiedNumOverlaps = 1;

  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.05;

  RealT x_min2 = -0.9;  // x-shift
  RealT y_min2 = -0.9;  // y-shift
  RealT z_min2 = 0.95;
  RealT x_max2 = 1.0 + x_min2;
  RealT y_max2 = y_min2 + 1.0;
  RealT z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  parameters.penalty_ratio = false;
  parameters.const_penalty = 1.0;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  RealT computed_gap = -( z_max1 - z_min2 );
  EXPECT_NEAR( plane.m_gap, computed_gap, 1.e-8 );
  EXPECT_NEAR( plane.m_area, ( x_max2 - x_min1 ) * ( y_max2 - y_min1 ), 1.e-10 );

  tribol::finalize();
}

TEST_F( CompGeomTest, common_plane_single_element_full_separation_check_1 )
{
  // mesh bounding box with 0.1 separation gap and no x/y shift
  int numElems1 = 1;
  int nElemsXM = numElems1;
  int nElemsYM = 1;
  int nElemsZM = numElems1;

  int numElems2 = 1;
  int nElemsXS = numElems2;
  int nElemsYS = 1;
  int nElemsZS = numElems2;

  int userSpecifiedNumOverlaps = 1;

  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.0;

  RealT x_min2 = 0.;
  RealT y_min2 = 0.;
  RealT z_min2 = 1.1;
  RealT x_max2 = 1.0;
  RealT y_max2 = 1.0;
  RealT z_max2 = 2.1;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  parameters.penalty_ratio = false;
  parameters.const_penalty = 1.0;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  RealT computed_gap = -( z_max1 - z_min2 );
  EXPECT_NEAR( plane.m_gap, computed_gap, 1.e-8 );

  EXPECT_NEAR( plane.m_area, ( x_max2 - x_min1 ) * ( y_max2 - y_min1 ), 1.e-10 );

  tribol::finalize();
}

TEST_F( CompGeomTest, common_plane_single_element_full_separation_check_2 )
{
  // mesh bounding box with 0.1 separation gap. The faces will
  // have an x and y shift
  int numElems1 = 1;
  int nElemsXM = numElems1;
  int nElemsYM = 1;
  int nElemsZM = numElems1;

  int numElems2 = 1;
  int nElemsXS = numElems2;
  int nElemsYS = 1;
  int nElemsZS = numElems2;

  int userSpecifiedNumOverlaps = 1;

  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.0;

  RealT x_min2 = -0.9;  // x-shift
  RealT y_min2 = -0.9;  // y-shift
  RealT z_min2 = 1.1;
  RealT x_max2 = 1.0 + x_min2;
  RealT y_max2 = y_min2 + 1.0;
  RealT z_max2 = 2.1;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  parameters.penalty_ratio = false;
  parameters.const_penalty = 1.0;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  RealT computed_gap = -( z_max1 - z_min2 );
  EXPECT_NEAR( plane.m_gap, computed_gap, 1.e-8 );

  EXPECT_NEAR( plane.m_area, ( x_max2 - x_min1 ) * ( y_max2 - y_min1 ), 1.e-10 );

  tribol::finalize();
}

TEST_F( CompGeomTest, common_plane_single_element_interpen_check_1 )
{
  // This test checks the interpen overlap case where each faces has
  // TWO line segments that intersect segments on the opposing face.
  // This can occur when the faces have identical dimensions in one
  // coordinate direction

  // The bottom block of this two-block problem is rotated clockwise
  // about the y-axis 45 degrees; then the top block is shifted such that the
  // centroids of the two contact surfaces are coincident in space.

  // An orthogonal edge on views of this interaction is something like:

  //       *            * * * * * * * *
  //        *           *             *
  //         *          *             *
  //    -------------   o-------------o
  //           *        *             *
  //            *       *             *
  //             *      * * * * * * * *

  int numElems1 = 1;
  int nElemsXM = numElems1;
  int nElemsYM = 1;
  int nElemsZM = numElems1;

  int numElems2 = 1;
  int nElemsXS = numElems2;
  int nElemsYS = 1;
  int nElemsZS = numElems2;

  int userSpecifiedNumOverlaps = 1;

  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.0;

  RealT x_min2 = 0.;
  RealT y_min2 = 0.;
  RealT z_min2 = 1.0;
  RealT x_max2 = 1.0;
  RealT y_max2 = 1.0;
  RealT z_max2 = 2.1;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // rotate the first face into the desired configuration
  RealT theta_y = 45.;
  m_mesh.rotateContactMesh( 0, 0., theta_y, 0. );

  // translate the second face into the desired configuration
  RealT shiftx = ( 0.7071 - 0.5 ) + 0.5 / 1.41421356;
  RealT shiftz = ( 1.0 - 0.7071 ) + 0.5 / 1.41421356;
  m_mesh.translateContactMesh( 1, shiftx, 0, -shiftz );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  parameters.penalty_ratio = false;
  parameters.const_penalty = 1.0;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  // compute length of interpen portion and the overlap centroid for gap calc.
  RealT hypotenuse = 0.5;
  RealT overlap_gap_point = 0.5 * hypotenuse * std::cos( 0.5 * theta_y * M_PI / 180 );

  // compute and check the gap
  RealT gap_computed = -2. * overlap_gap_point * std::tan( 0.5 * theta_y * M_PI / 180 );
  EXPECT_NEAR( plane.m_gap, gap_computed, 1.e-5 );

  // check the overlap area
  EXPECT_NEAR( plane.m_area, 2. * overlap_gap_point, 1.e-5 );

  EXPECT_EQ( plane.m_fullOverlap, false );
  EXPECT_EQ( plane.m_numPolyVert, 4 );

  tribol::finalize();
}

TEST_F( CompGeomTest, common_plane_single_element_interpen_check_2 )
{
  // This test checks where one face has two line-plane intersections inside
  // the opposing face and the other does not. Specifically this test has
  // one face that is smaller than the other and then rotated similar to
  // interpen_check_1 above, but the intersection points lie inside the
  // other face and not on its outer segments. The two orthogonal edge-on
  // views of the interaction are:
  //
  //               *           * * * * * *
  //             *             *         *
  //           *               *         *
  //  -------o--------       --o---------o--
  //       *                   *         *
  //     *                     *         *
  //   *                       * * * * * *
  //
  //

  constexpr int numVerts = 4;
  constexpr int numCells = 2;
  constexpr int lengthNodalData = numCells * numVerts;
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  // coordinates for face 1
  x[0] = 0.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = 0.;

  y[0] = 0.;
  y[1] = 0.;
  y[2] = 1.;
  y[3] = 1.;

  z[0] = 0.;
  z[1] = 0.;
  z[2] = 0.;
  z[3] = 0.;

  // coordinates for face 2
  RealT fortyfive = 45 * M_PI / 180;
  x[4] = 1.0 / 3.0;
  y[4] = 0.25;
  z[4] = -0.25;

  x[5] = x[4];
  y[5] = y[4] + 0.5;
  z[5] = z[4];

  x[6] = x[5];
  y[6] = y[5];
  z[6] = 1.0;

  x[7] = x[4];
  y[7] = y[4];
  z[7] = z[6];

  // rotate 45 degrees about the y-axis
  RealT x_shift = x[4];
  RealT z_shift = z[4];
  for ( int i = numVerts; i < lengthNodalData; ++i ) {
    x[i] = x[i] - x_shift;
    z[i] = z[i] - z_shift;
    RealT x_rot = x[i] * std::cos( fortyfive ) + z[i] * std::sin( fortyfive );
    RealT z_rot = x[i] * -std::sin( fortyfive ) + z[i] * std::cos( fortyfive );
    x[i] = x_rot + x_shift;
    z[i] = z_rot + z_shift;
  }

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  RealT dt = 1.0;
  int err = setupAndUpdateAutoCommonPlane( mesh_id, 0, lengthNodalData, numCells, dt );

  EXPECT_EQ( err, 0 );
  EXPECT_EQ( dt, 1.0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  // compute the "length" of the interpen portion of face 2
  RealT h = 0.25 / std::cos( fortyfive );

  // compute that "length" as projected onto the common plane
  RealT h_bar = h * std::cos( 0.5 * fortyfive );

  // compute and check the overlap area
  RealT A = h_bar * 0.5;
  EXPECT_NEAR( plane.m_area, A, 1.e-8 );

  RealT computed_gap = -h_bar * std::tan( 0.5 * fortyfive );
  EXPECT_NEAR( plane.m_gap, computed_gap, 1.e-6 );

  EXPECT_EQ( plane.m_fullOverlap, false );
  EXPECT_EQ( plane.m_numPolyVert, 4 );
}

TEST_F( CompGeomTest, common_plane_single_element_interpen_check_3 )
{
  // This test checks where one face has two line-plane intersections inside
  // the opposing face and the other does not. Specifically this test has
  // one face that is smaller than the other and then rotated similar to
  // interpen_check_1 above, but the intersection points lie inside the
  // other face and not on its outer segments. The two orthogonal edge-on
  // views of the interaction are:
  //
  //               *               * * * * * *
  //             *                 *         *
  //           *                   *         *
  //  -------o--------       ------o-----    *
  //       *                       *         *
  //     *                         *         *
  //   *                           * * * * * *
  //
  //

  constexpr int numVerts = 4;
  constexpr int numCells = 2;
  constexpr int lengthNodalData = numCells * numVerts;
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  // coordinates for face 1
  x[0] = 0.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = 0.;

  y[0] = 0.;
  y[1] = 0.;
  y[2] = 1.;
  y[3] = 1.;

  z[0] = 0.;
  z[1] = 0.;
  z[2] = 0.;
  z[3] = 0.;

  // coordinates for face 2
  x[4] = 1.0 / 3.0;
  y[4] = -0.5;
  z[4] = -0.25;

  x[5] = x[4];
  y[5] = 0.5;
  z[5] = z[4];

  x[6] = x[5];
  y[6] = y[5];
  z[6] = 1.0;

  x[7] = x[4];
  y[7] = y[4];
  z[7] = z[6];

  // rotate 45 degrees about the y-axis
  RealT x_shift = x[4];
  RealT z_shift = z[4];
  RealT fortyfive = 45 * M_PI / 180;
  for ( int i = numVerts; i < lengthNodalData; ++i ) {
    x[i] = x[i] - x_shift;
    z[i] = z[i] - z_shift;
    RealT x_rot = x[i] * std::cos( fortyfive ) + z[i] * std::sin( fortyfive );
    RealT z_rot = x[i] * -std::sin( fortyfive ) + z[i] * std::cos( fortyfive );
    x[i] = x_rot + x_shift;
    z[i] = z_rot + z_shift;
  }

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  RealT dt = 1.0;
  int err = setupAndUpdateAutoCommonPlane( mesh_id, 0, lengthNodalData, numCells, dt );

  EXPECT_EQ( err, 0 );
  EXPECT_EQ( dt, 1.0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  // compute the geometric quantities of the intepren portion of face 2
  // in order to compute the overlap area
  RealT h = 0.25 / std::cos( fortyfive );
  RealT h_bar = h * std::cos( 0.5 * fortyfive );
  RealT A = h_bar * 0.5;

  // check the overlap area
  EXPECT_NEAR( plane.m_area, A, 1.e-8 );

  // compute and check the gap
  RealT computed_gap = -h_bar * std::tan( 0.5 * fortyfive );
  EXPECT_NEAR( plane.m_gap, computed_gap, 1.e-6 );

  EXPECT_EQ( plane.m_fullOverlap, false );
  EXPECT_EQ( plane.m_numPolyVert, 4 );
}

TEST_F( CompGeomTest, common_plane_single_element_interpen_check_3_warped )
{
  // This test checks where one face has two line-plane intersections inside
  // the opposing face and the other does not. Specifically this test has
  // one face that is smaller than the other and then rotated similar to
  // interpen_check_1 above, but the intersection points lie inside the
  // other face and not on its outer segments. Furthermore, the two faces
  // are warped, but in a way that the projections and resulting overlap area
  // should be the same as the non-warped version of this test above. The two
  // orthogonal edge-on views of the interaction are (not showing warpedness
  // for clarity):
  //
  //               *               * * * * * *
  //             *                 *         *
  //           *                   *         *
  //  -------o--------       ------o-----    *
  //       *                       *         *
  //     *                         *         *
  //   *                           * * * * * *
  //
  //

  constexpr int numVerts = 4;
  constexpr int numCells = 2;
  constexpr int lengthNodalData = numCells * numVerts;
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  // coordinates for face 1
  x[0] = 0.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = 0.;

  y[0] = 0.;
  y[1] = 0.;
  y[2] = 1.;
  y[3] = 1.;

  z[0] = -0.25;
  z[1] = 0.25;
  z[2] = -0.25;
  z[3] = 0.25;

  // coordinates for face 2
  x[4] = 1.0 / 3.0;
  y[4] = -0.5;
  z[4] = -0.25;

  x[5] = x[4];
  y[5] = 0.5;
  z[5] = z[4];

  x[6] = x[5];
  y[6] = y[5];
  z[6] = 1.0;

  x[7] = x[4];
  y[7] = y[4];
  z[7] = z[6];

  // rotate face 2 45 degrees about the y-axis
  RealT x_shift = x[4];
  RealT z_shift = z[4];
  RealT fortyfive = 45 * M_PI / 180;
  for ( int i = numVerts; i < lengthNodalData; ++i ) {
    x[i] = x[i] - x_shift;
    z[i] = z[i] - z_shift;
    RealT x_rot = x[i] * std::cos( fortyfive ) + z[i] * std::sin( fortyfive );
    RealT z_rot = x[i] * -std::sin( fortyfive ) + z[i] * std::cos( fortyfive );
    x[i] = x_rot + x_shift;
    z[i] = z_rot + z_shift;
  }

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  RealT dt = 1.0;
  int err = setupAndUpdateAutoCommonPlane( mesh_id, 0, lengthNodalData, numCells, dt );

  EXPECT_EQ( err, 0 );
  EXPECT_EQ( dt, 1.0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  // compute the geometric quantities of the intepren portion of face 2
  // in order to compute the overlap area
  RealT h = 0.25 / std::cos( fortyfive );
  RealT h_bar = h * std::cos( 0.5 * fortyfive );
  RealT A = h_bar * 0.5;

  // check the overlap area
  EXPECT_NEAR( plane.m_area, A, 1.e-8 );

  // compute and check the gap
  RealT computed_gap = -h_bar * std::tan( 0.5 * fortyfive );
  EXPECT_NEAR( plane.m_gap, computed_gap, 1.e-6 );

  EXPECT_EQ( plane.m_fullOverlap, false );
  EXPECT_EQ( plane.m_numPolyVert, 4 );
}

TEST_F( CompGeomTest, common_plane_single_element_interpen_check_4 )
{
  // This tests checks the case where one face has two line-plane intersections
  // that are interior to the other face (one fully interior and the other lying on
  // the edge of the other face), and the other face has one line-plane
  // intersection that is interior to the other face, which lies on the edge of
  // the other face.
  //
  // The two orthogonal edge-on views of the interaction are:
  //
  //               *               * * * *
  //             *                 *     *
  //           *                   *     *
  //  -------o--------       ------o-----o
  //       *                       *     *
  //     *                         *     *
  //   *                           * * * *
  //
  //

  constexpr int numVerts = 4;
  constexpr int numCells = 2;
  constexpr int lengthNodalData = numCells * numVerts;
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  // coordinates for face 1
  x[0] = 0.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = 0.;

  y[0] = 0.;
  y[1] = 0.;
  y[2] = 1.;
  y[3] = 1.;

  z[0] = 0.;
  z[1] = 0.;
  z[2] = 0.;
  z[3] = 0.;

  // coordinates for face 2
  x[4] = 1.0 / 3.0;
  y[4] = 0.;
  z[4] = -0.25;

  x[5] = x[4];
  y[5] = 0.5;
  z[5] = z[4];

  x[6] = x[5];
  y[6] = y[5];
  z[6] = 1.0;

  x[7] = x[4];
  y[7] = y[4];
  z[7] = z[6];

  // rotate 45 degrees about the y-axis
  RealT x_shift = x[4];
  RealT z_shift = z[4];
  RealT fortyfive = 45 * M_PI / 180;
  for ( int i = numVerts; i < lengthNodalData; ++i ) {
    x[i] = x[i] - x_shift;
    z[i] = z[i] - z_shift;
    RealT x_rot = x[i] * std::cos( fortyfive ) + z[i] * std::sin( fortyfive );
    RealT z_rot = x[i] * -std::sin( fortyfive ) + z[i] * std::cos( fortyfive );
    x[i] = x_rot + x_shift;
    z[i] = z_rot + z_shift;
  }

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  RealT dt = 1.0;
  int err = setupAndUpdateAutoCommonPlane( mesh_id, 0, lengthNodalData, numCells, dt );

  EXPECT_EQ( err, 0 );
  EXPECT_EQ( dt, 1.0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  // compute the geometric quantities of the intepren portion of face 2
  // in order to compute the overlap area
  RealT h = 0.25 / std::cos( fortyfive );
  RealT h_bar = h * std::cos( 0.5 * fortyfive );
  RealT A = h_bar * 0.5;

  EXPECT_NEAR( plane.m_area, A, 1.e-8 );

  // compute and check the gap
  RealT computed_gap = -h_bar * std::tan( 0.5 * fortyfive );
  EXPECT_NEAR( plane.m_gap, computed_gap, 1.e-6 );

  EXPECT_EQ( plane.m_fullOverlap, false );
  EXPECT_EQ( plane.m_numPolyVert, 4 );
}

TEST_F( CompGeomTest, common_plane_single_element_interpen_check_5 )
{
  // This test checks the case where one face has two-line plane intersections
  // inside the opposing face and the opposing face has zero that lie inside the
  // other face.
  //
  // Specifically, this test checks the interaction between a flat face and a rotated
  // face where one node interpenetrates the opposing flat face. This
  // forms a triangular intersection. An edge on view looks like:
  //
  //              *
  //            *   *
  //          *       *
  //        *           *
  //      *               *
  //    ----o-----------o----
  //          *       *
  //            *   *
  //              *
  //
  //  where the rotated face is also rotated into the page 30 degrees. The
  //  intersection points are marked as "o".
  //
  constexpr int numVerts = 4;
  constexpr int numCells = 2;
  constexpr int lengthNodalData = numCells * numVerts;
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  // coordinates for face 1
  x[0] = 0.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = 0.;

  y[0] = 0.;
  y[1] = 0.;
  y[2] = 1.;
  y[3] = 1.;

  z[0] = 0.;
  z[1] = 0.;
  z[2] = 0.;
  z[3] = 0.;

  // coordinates for face 2
  RealT thirty = 30 * M_PI / 180;
  RealT fortyfive = 45 * M_PI / 180;
  x[4] = 1.0 / 3.0;
  y[4] = 0.5;
  z[4] = -0.25;

  x[5] = x[4];
  y[5] = y[4] + 0.5 * std::tan( fortyfive );
  z[5] = 0.25;

  x[6] = x[4];
  y[6] = y[4];
  z[6] = 0.75;

  x[7] = x[5];
  y[7] = y[4] - 0.5 * std::tan( fortyfive );
  z[7] = 0.25;

  // rotate 30 degrees about the y-axis
  RealT x_shift = x[4];
  RealT z_shift = z[4];
  for ( int i = numVerts; i < lengthNodalData; ++i ) {
    x[i] = x[i] - x_shift;
    z[i] = z[i] - z_shift;
    RealT x_rot = x[i] * std::cos( thirty ) + z[i] * std::sin( thirty );
    RealT z_rot = x[i] * -std::sin( thirty ) + z[i] * std::cos( thirty );
    x[i] = x_rot + x_shift;
    z[i] = z_rot + z_shift;
  }

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  RealT dt = 1.0;
  int err = setupAndUpdateAutoCommonPlane( mesh_id, 0, lengthNodalData, numCells, dt );

  EXPECT_EQ( err, 0 );
  EXPECT_EQ( dt, 1.0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  // compute the geometric quantities of interest in order to compute the area of overlap and gap
  // Note: this is somewhat involved and difficult to make super clear. Talk to SRW if you need
  // more details
  RealT third = 1.0 / 3.0;
  RealT h = 0.25;                        // height of initial interpen before face rotation
  RealT h_bar = h / std::cos( thirty );  // "height" of interpen after rotation
  // RealT w = h * std::tan(fortyfive);  // keep for reference
  RealT h_bar_bar = h_bar * std::cos( thirty );  // "height" of interpen on common plane
  RealT w_bar = h_bar * std::tan( fortyfive );   // width of triangular interpen after rotation
  // RealT A = w * h; // keep for reference
  // RealT A_bar = w_bar * h_bar; // keep for reference
  RealT A_bar_bar = h_bar_bar * w_bar;  // area of triangular interpen on common plane
  RealT computed_gap = -2 * ( third * h_bar_bar ) * std::tan( thirty );

  // check the area
  EXPECT_NEAR( plane.m_area, A_bar_bar, 1.e-8 );

  // check the gap
  EXPECT_NEAR( plane.m_gap, computed_gap, 1.e-6 );

  EXPECT_EQ( plane.m_fullOverlap, false );
  EXPECT_EQ( plane.m_numPolyVert, 3 );
}

TEST_F( CompGeomTest, common_plane_single_element_interpen_check_6 )
{
  // This test has a configuration where one face has "inside" intersection
  // points with the other face exactly at two of the first face's vertices
  // An edge on view looks like:
  //
  //              *
  //            *   *
  //          *       *
  //        *           *
  //    --o---------------o----
  //        *           *
  //          *       *
  //            *   *
  //              *
  //
  //  where the rotated face is also rotated into the page 30 degrees. The
  //  intersection points are marked as "o".
  //
  constexpr int numVerts = 4;
  constexpr int numCells = 2;
  constexpr int lengthNodalData = numCells * numVerts;
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  // coordinates for face 1
  x[0] = -1.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = -1.;

  y[0] = -1.;
  y[1] = -1.;
  y[2] = 1.;
  y[3] = 1.;

  z[0] = 0.;
  z[1] = 0.;
  z[2] = 0.;
  z[3] = 0.;

  // coordinates for face 2
  RealT thirty = 30 * M_PI / 180;
  x[4] = 0.;
  y[4] = 0.;
  z[4] = -0.45;

  x[5] = x[4];
  y[5] = y[4] - z[4];
  z[5] = 0.;

  x[6] = x[4];
  y[6] = y[4];
  z[6] = -z[4];

  x[7] = x[5];
  y[7] = y[4] + z[4];
  z[7] = 0.;

  // rotate 30 degrees about the y-axis
  // RealT x_shift = x[4];
  // RealT z_shift = z[4];
  for ( int i = numVerts; i < lengthNodalData; ++i ) {
    // x[i] = x[i] - x_shift;
    // z[i] = z[i] - z_shift;
    RealT x_rot = x[i] * std::cos( thirty ) + z[i] * std::sin( thirty );
    RealT z_rot = x[i] * -std::sin( thirty ) + z[i] * std::cos( thirty );
    x[i] = x_rot;  //+ x_shift;
    z[i] = z_rot;  //+ z_shift;
  }

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  RealT dt = 1.0;
  int err = setupAndUpdateAutoCommonPlane( mesh_id, 0, lengthNodalData, numCells, dt );

  EXPECT_EQ( err, 0 );
  EXPECT_EQ( dt, 1.0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  // compute the geometric quantities of interest in order to compute the area of overlap and gap
  // Note: this is somewhat involved and difficult to make super clear. Talk to SRW if you need
  // more details
  RealT third = 1.0 / 3.0;
  RealT h = 0.45;                        // height of initial interpen before face rotation
  RealT h_bar = h * std::cos( thirty );  // "height" of interpen on common plane
  RealT w_bar = 0.45;                    // width of triangular interpen after rotation
  RealT A_bar = h_bar * w_bar;           // area of triangular interpen on common plane
  RealT computed_gap = -2 * ( third * h_bar ) * std::tan( thirty );

  // check the area
  EXPECT_NEAR( plane.m_area, A_bar, 1.e-8 );

  // check the gap
  EXPECT_NEAR( plane.m_gap, computed_gap, 1.e-6 );

  EXPECT_EQ( plane.m_fullOverlap, false );
  EXPECT_EQ( plane.m_numPolyVert, 3 );
}

TEST_F( CompGeomTest, common_plane_single_element_interpen_check_7 )
{
  // This test has a configuration where one face has "inside" intersection
  // points with the other face at one vertex and some epsilon from another vertex.
  // An edge on view looks like:
  //
  //                *
  //              *   *
  //            *       *
  //          *           *
  //        *               *
  //    --o---------------o----
  //        *           *
  //          *       *
  //            *   *
  //              *
  //
  //  where the rotated face is also rotated into the page 30 degrees. The
  //  intersection points are marked as "o".
  //
  constexpr int numVerts = 4;
  constexpr int numCells = 2;
  constexpr int lengthNodalData = numCells * numVerts;
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  // coordinates for face 1
  x[0] = -1.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = -1.;

  y[0] = -1.;
  y[1] = -1.;
  y[2] = 1.;
  y[3] = 1.;

  z[0] = 0.;
  z[1] = 0.;
  z[2] = 0.;
  z[3] = 0.;

  // coordinates for face 2
  RealT epsilon = 1.e-15;
  RealT thirty = 30 * M_PI / 180;
  x[4] = 0.;
  y[4] = 0.;
  z[4] = -0.45;

  x[5] = x[4];
  y[5] = y[4] - z[4];
  z[5] = 0.;

  x[6] = x[4];
  y[6] = y[4];
  z[6] = -z[4];

  x[7] = x[5];
  y[7] = y[4] + z[4];
  z[7] = epsilon;

  // rotate 30 degrees about the y-axis
  // RealT x_shift = x[4];
  // RealT z_shift = z[4];
  for ( int i = numVerts; i < lengthNodalData; ++i ) {
    // x[i] = x[i] - x_shift;
    // z[i] = z[i] - z_shift;
    RealT x_rot = x[i] * std::cos( thirty ) + z[i] * std::sin( thirty );
    RealT z_rot = x[i] * -std::sin( thirty ) + z[i] * std::cos( thirty );
    x[i] = x_rot;  //+ x_shift;
    z[i] = z_rot;  //+ z_shift;
  }

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  RealT dt = 1.0;
  int err = setupAndUpdateAutoCommonPlane( mesh_id, 0, lengthNodalData, numCells, dt );

  EXPECT_EQ( err, 0 );
  EXPECT_EQ( dt, 1.0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  // compute the geometric quantities of interest in order to compute the area of overlap and gap
  // Note: this is somewhat involved and difficult to make super clear. Talk to SRW if you need
  // more details
  RealT third = 1.0 / 3.0;
  RealT h = 0.45;                        // height of initial interpen before face rotation
  RealT h_bar = h * std::cos( thirty );  // "height" of interpen on common plane
  RealT w_bar = 0.45;                    // width of triangular interpen after rotation
  RealT A_bar = h_bar * w_bar;           // area of triangular interpen on common plane
  RealT computed_gap = -2 * ( third * h_bar ) * std::tan( thirty );

  // check the area
  EXPECT_NEAR( plane.m_area, A_bar, 1.e-8 );

  // check the gap
  EXPECT_NEAR( plane.m_gap, computed_gap, 1.e-6 );

  EXPECT_EQ( plane.m_fullOverlap, false );
  EXPECT_EQ( plane.m_numPolyVert, 3 );
}

TEST_F( CompGeomTest, common_plane_single_element_interpen_check_8 )
{
  // This test has a x-configuration that should trigger an interpen calculation.
  // The x-configuration is created by some very small epsilon shift in the z-coordinates
  // of the second face. An edge-on view of this configuration is:
  //
  //   *
  //   -------*--------
  //                  *
  //

  constexpr int numVerts = 4;
  constexpr int numCells = 2;
  constexpr int lengthNodalData = numCells * numVerts;
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  // coordinates for face 1
  x[0] = -1.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = -1.;

  y[0] = -1.;
  y[1] = -1.;
  y[2] = 1.;
  y[3] = 1.;

  z[0] = 0.;
  z[1] = 0.;
  z[2] = 0.;
  z[3] = 0.;

  // coordinates for face 2; rotate the face about the y axis by some epsilon shift in the z-direction
  RealT epsilon = 1.e-15;
  x[4] = x[3];
  x[5] = x[2];
  x[6] = x[1];
  x[7] = x[0];

  y[4] = y[3];
  y[5] = y[2];
  y[6] = y[1];
  y[7] = y[0];

  z[4] = z[3] + epsilon;
  z[5] = z[2] - epsilon;
  z[6] = z[1] - epsilon;
  z[7] = z[0] + epsilon;

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  RealT dt = 1.0;
  int err = setupAndUpdateAutoCommonPlane( mesh_id, 0, lengthNodalData, numCells, dt );

  EXPECT_EQ( err, 0 );
  EXPECT_EQ( dt, 1.0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  EXPECT_EQ( plane.m_fullOverlap, false );
  EXPECT_EQ( plane.m_numPolyVert, 4 );

  // check the area
  EXPECT_NEAR( plane.m_area, 2.0, 1.e-6 );

  // check the gap
  EXPECT_NEAR( plane.m_gap, epsilon / 2., 1.e-6 );
}

TEST_F( CompGeomTest, common_plane_single_element_interpen_check_9 )
{
  // This test checks to see if the geomFilter Check #5 ultimately returns
  // a contact candidate for a convex quad close to the shape below. The other
  // face is a square quad and the overlap should be a full overlap

  //
  //   *           *
  //   **        **
  //    * *   *  *
  //     *  *  *
  //      *  *
  //       *
  //

  constexpr int numVerts = 4;
  constexpr int numCells = 2;
  constexpr int lengthNodalData = numCells * numVerts;
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  // coordinates for face 1
  x[0] = -1.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = -1.;

  y[0] = -1.;
  y[1] = -1.;
  y[2] = 1.;
  y[3] = 1.;

  z[0] = 0.;
  z[1] = 0.;
  z[2] = 0.;
  z[3] = 0.;

  // coordinates for face 2
  x[4] = 0.;
  x[5] = -0.5;
  x[6] = 0.;
  x[7] = 0.5;

  y[4] = -0.75;
  y[5] = 0.5;
  y[6] = 0.;
  y[7] = 0.5;

  z[4] = 0.;
  z[5] = 0.;
  z[6] = 0.;
  z[7] = 0.;

  // check that non-convex areas are computed correctly
  EXPECT_EQ( tribol::Area2DPolygon( &x[4], &y[4], numVerts ), 0.375 );

  // rotate second face 30 degrees about the y-axis
  RealT thirty = 30 * M_PI / 180;
  for ( int i = numVerts; i < lengthNodalData; ++i ) {
    x[i] = x[i];
    z[i] = z[i];
    RealT x_rot = x[i] * std::cos( thirty ) + z[i] * std::sin( thirty );
    RealT z_rot = x[i] * -std::sin( thirty ) + z[i] * std::cos( thirty );
    x[i] = x_rot;
    z[i] = z_rot;
  }

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  RealT dt = 1.0;
  int err = setupAndUpdateAutoCommonPlane( mesh_id, 0, lengthNodalData, numCells, dt );

  EXPECT_EQ( err, 0 );
  EXPECT_EQ( dt, 1.0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  EXPECT_EQ( plane.m_fullOverlap, true );
  EXPECT_EQ( plane.m_numPolyVert, 4 );

  // check the area
  EXPECT_NEAR( plane.m_area, 0.36222218, 1.e-6 );

  // check the gap
  EXPECT_NEAR( plane.m_gap, 0., 1.e-6 );
}

TEST_F( CompGeomTest, common_plane_single_element_interpen_check_10 )
{
  // This test checks the case where one face has two-line plane intersections
  // inside the opposing face and the opposing face has zero that lie inside the
  // other face.
  //
  // Specifically, this test checks the interaction between a flat face and a rotated
  // face where three nodes interpenetrates the opposing flat face. This
  // forms an intersection with 5 vertices. An edge on view looks like:
  //
  //
  //              *
  //            *   *
  //          *       *
  //    ----o-----------o----
  //      *               *
  //        *           *
  //          *       *
  //            *   *
  //              *
  //
  //  where the rotated face is also rotated into the page 30 degrees. The
  //  intersection points are marked as "o".
  //
  constexpr int numVerts = 4;
  constexpr int numCells = 2;
  constexpr int lengthNodalData = numCells * numVerts;
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  // coordinates for face 1
  x[0] = 0.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = 0.;

  y[0] = 0.;
  y[1] = 0.;
  y[2] = 1.;
  y[3] = 1.;

  z[0] = 0.;
  z[1] = 0.;
  z[2] = 0.;
  z[3] = 0.;

  // coordinates for face 2
  RealT thirty = 30 * M_PI / 180;
  RealT fortyfive = 45 * M_PI / 180;
  RealT third = 1.0 / 3.0;
  // shift the x-coord so when we lower second face it is still within
  // full coverage of the first face when projected to common plane
  RealT small_x_shift = 0.1;
  x[4] = third + small_x_shift;
  y[4] = 0.5;
  z[4] = -0.25;

  x[5] = x[4];
  y[5] = y[4] + 0.5 * std::tan( fortyfive );
  z[5] = 0.25;

  x[6] = x[4];
  y[6] = y[4];
  z[6] = 0.75;

  x[7] = x[5];
  y[7] = y[4] - 0.5 * std::tan( fortyfive );
  z[7] = 0.25;

  RealT side1 = tribol::magnitude( x[5] - x[4], y[5] - y[4], z[5] - z[4] );
  RealT side2 = tribol::magnitude( x[6] - x[5], y[6] - y[5], z[6] - z[5] );
  RealT face_2_area = side1 * side2;

  // rotate 30 degrees about the y-axis
  RealT x_shift = x[4];
  RealT z_shift = z[4];
  for ( int i = numVerts; i < lengthNodalData; ++i ) {
    x[i] = x[i] - x_shift;
    z[i] = z[i] - z_shift;
    RealT x_rot = x[i] * std::cos( thirty ) + z[i] * std::sin( thirty );
    RealT z_rot = x[i] * -std::sin( thirty ) + z[i] * std::cos( thirty );
    x[i] = x_rot + x_shift;
    z[i] = z_rot + z_shift;
  }

  // now shift the vertices down so that three vertices of the rotated
  // face interpenetrate the flat face such that the portion not interpenetrating
  // is of equal size to the original interpenetrating triangle (coords at this point
  // in the test and same final coords as interpen_check_4 above).
  RealT new_z_shift = z[6] - 0.25;
  for ( int i = numVerts; i < lengthNodalData; ++i ) {
    z[i] = z[i] - new_z_shift;
  }

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  RealT dt = 1.0;
  int err = setupAndUpdateAutoCommonPlane( mesh_id, 0, lengthNodalData, numCells, dt );

  EXPECT_EQ( err, 0 );
  EXPECT_EQ( dt, 1.0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  // see interpen_check_4 for meaning of these variables. These now define
  // the triangular portion of face 2 that is NOT interpenetrating face 1
  RealT h = 0.25;
  RealT h_bar = h / std::cos( thirty );
  // RealT w = h * std::tan(fortyfive);
  RealT h_bar_bar = h_bar * std::cos( thirty );
  RealT w_bar = h_bar * std::tan( fortyfive );
  // RealT A = w * h;
  RealT A_bar = w_bar * h_bar;
  RealT A_bar_bar = h_bar_bar * w_bar;

  RealT projection_ratio = A_bar_bar / A_bar;
  RealT A_bar_bar_new = projection_ratio * ( face_2_area - A_bar );

  // the gap is not computed easily so use the area and the number
  // of overlap vertices as a stand in for correct computations as
  // the gap calculation is verified in other tests
  EXPECT_NEAR( plane.m_area, A_bar_bar_new, 1.e-8 );
  EXPECT_EQ( plane.m_numPolyVert, 5 );

  EXPECT_EQ( plane.m_fullOverlap, false );
  EXPECT_EQ( plane.m_numPolyVert, 5 );
}

TEST_F( CompGeomTest, common_plane_perfect_conforming_full_overlap )
{
  int numElems1 = 3;
  int nElemsXM = numElems1;
  int nElemsYM = 3;
  int nElemsZM = numElems1;

  int numElems2 = 3;
  int nElemsXS = numElems2;
  int nElemsYS = 3;
  int nElemsZS = numElems2;

  int userSpecifiedNumOverlaps = 9;

  // mesh bounding box with 0.1 interpenetration gap
  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.05;

  RealT x_min2 = 0.;
  RealT y_min2 = 0.;
  RealT z_min2 = 0.95;
  RealT x_max2 = 1.0;
  RealT y_max2 = 1.0;
  RealT z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  parameters.penalty_ratio = false;
  parameters.const_penalty = 1.0;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  EXPECT_EQ( plane.m_fullOverlap, true );

  RealT len = 1.0 / numElems1;
  EXPECT_NEAR( plane.m_area, len * len, 1.e-8 );
  EXPECT_EQ( plane.m_numPolyVert, 4 );

  tribol::finalize();
}

TEST_F( CompGeomTest, common_plane_xy_shift_full_overlap )
{
  int numElems1 = 3;
  int nElemsXM = numElems1;
  int nElemsYM = 3;
  int nElemsZM = numElems1;

  int numElems2 = 3;
  int nElemsXS = numElems2;
  int nElemsYS = 3;
  int nElemsZS = numElems2;

  int userSpecifiedNumOverlaps = 25;

  // mesh bounding box with 0.1 interpenetration gap
  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.05;

  RealT x_min2 = -0.1;
  RealT y_min2 = 0.0001;
  RealT z_min2 = 0.95;
  RealT x_max2 = 0.9;
  RealT y_max2 = 1.0001;
  RealT z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  parameters.penalty_ratio = false;
  parameters.const_penalty = 1.0;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );

  tribol::finalize();
}

TEST_F( CompGeomTest, common_plane_rotated_face_full_overlap )
{
  int numElems1 = 1;
  int nElemsXM = numElems1;
  int nElemsYM = numElems1;
  int nElemsZM = numElems1;

  int numElems2 = 1;
  int nElemsXS = numElems2;
  int nElemsYS = numElems2;
  int nElemsZS = numElems2;

  int userSpecifiedNumOverlaps = 1;

  // mesh bounding box with 0.1 interpenetration gap
  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.05;

  RealT x_min2 = 0.;
  RealT y_min2 = 0.;
  RealT z_min2 = 0.95;
  RealT x_max2 = 1.0;
  RealT y_max2 = 1.0;
  RealT z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );
  // rotate mesh 0 45 degrees
  RealT theta_z = 45.;
  RealT x_shift = 0.5;
  RealT y_shift = 0.5;
  m_mesh.rotateContactMesh( 0, 0., 0, theta_z, x_shift, y_shift );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  parameters.penalty_ratio = false;
  parameters.const_penalty = 1.0;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );
  EXPECT_EQ( plane.m_numPolyVert, 8 );

  tribol::finalize();
}

TEST_F( CompGeomTest, common_plane_host_code_test )
{
  // This test uses face coordinates from host-code testing that at one point
  // returned an inverted overlapping polygon. This bug has since been fixed
  // and this configuration will test a once problematic interface pair.
  constexpr int numVerts = 4;
  constexpr int numCells = 2;
  constexpr int lengthNodalData = numCells * numVerts;
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  // coordinates for face 1
  x[0] = 9.39102308674e-05;
  x[1] = 9.574507947643143e-05;
  x[2] = 0.00010052443467949201;
  x[3] = 9.719026465741154e-05;

  y[0] = 0.4166824769718296;
  y[1] = 0.4722432987882217;
  y[2] = 0.47223943800552937;
  y[3] = 0.4166796856918463;

  z[0] = 0.33334123166705104;
  z[1] = 0.33334216189246335;
  z[2] = 0.41667946161766184;
  z[3] = 0.4166796814171943;

  // coordinates for face 2
  x[4] = -9.39102308408737e-05;
  x[5] = -9.71902646081789e-05;
  x[6] = -0.00010052443461445242;
  x[7] = -9.574507943875333e-05;

  y[4] = 0.4166824769718633;
  y[5] = 0.4166796856918866;
  y[6] = 0.4722394380055709;
  y[7] = 0.4722432987882541;

  z[4] = 0.3333412316670809;
  z[5] = 0.4166796814172351;
  z[6] = 0.41667946161770675;
  z[7] = 0.3333421618924954;

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  RealT dt = 1.0;
  int err = setupAndUpdateAutoCommonPlane( mesh_id, 0, lengthNodalData, numCells, dt );

  EXPECT_EQ( err, 0 );
  EXPECT_EQ( dt, 1.0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  EXPECT_NEAR( plane.m_area, 0.00463028, 1.e-8 );
  EXPECT_NEAR( plane.m_gap, -0.000193685, 1.e-8 );
}

TEST_F( CompGeomTest, single_mortar_check_1 )
{
  int nMortarElems = 4;
  int nElemsXM = nMortarElems;
  int nElemsYM = nMortarElems;
  int nElemsZM = nMortarElems;

  int nNonmortarElems = 5;
  int nElemsXS = nNonmortarElems;
  int nElemsYS = nNonmortarElems;
  int nElemsZS = nNonmortarElems;

  int userSpecifiedNumOverlaps = 64;

  // mesh bounding box with 0.1 interpenetration gap
  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.05;

  RealT x_min2 = 0.;
  RealT y_min2 = 0.;
  RealT z_min2 = 0.95;
  RealT x_max2 = 1.;
  RealT y_max2 = 1.;
  RealT z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::SINGLE_MORTAR, tribol::LAGRANGE_MULTIPLIER, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );

  tribol::finalize();
}

TEST_F( CompGeomTest, single_mortar_check_2 )
{
  // This checks a rotated X-like configuration of the two contact surfaces
  int nMortarElems = 1;
  int nElemsXM = nMortarElems;
  int nElemsYM = 1;
  int nElemsZM = nMortarElems;

  int nNonmortarElems = 1;
  int nElemsXS = nNonmortarElems;
  int nElemsYS = 1;
  int nElemsZS = nNonmortarElems;

  int userSpecifiedNumOverlaps = 1;

  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.0;

  RealT x_min2 = 0.;
  RealT y_min2 = 0.;
  RealT z_min2 = 1.0;
  RealT x_max2 = 1.0;
  RealT y_max2 = 1.0;
  RealT z_max2 = 2.1;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  RealT theta_y = 45.;
  m_mesh.rotateContactMesh( 0, 0., theta_y, 0. );

  RealT shiftx = ( 0.7071 - 0.5 ) + 0.5 / 1.41421356;
  RealT shiftz = ( 1.0 - 0.7071 ) + 0.5 / 1.41421356;
  m_mesh.translateContactMesh( 1, shiftx, 0, -shiftz );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::SINGLE_MORTAR, tribol::LAGRANGE_MULTIPLIER, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );

  auto& comp_geom = couplingScheme->getCompGeom();
  auto& plane = comp_geom.getMortarPlane( 0 );

  EXPECT_NEAR( plane.m_area, std::cos( 45. * M_PI / 180. ), 1.e-8 );

  tribol::finalize();
}

TEST_F( CompGeomTest, aligned_mortar_check_1 )
{
  int nMortarElems = 4;
  int nElemsXM = nMortarElems;
  int nElemsYM = nMortarElems;
  int nElemsZM = nMortarElems;

  int nNonmortarElems = 4;
  int nElemsXS = nNonmortarElems;
  int nElemsYS = nNonmortarElems;
  int nElemsZS = nNonmortarElems;

  int userSpecifiedNumOverlaps = 16;

  // mesh bounding box with 0.1 interpenetration gap
  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.05;

  RealT x_min2 = 0.;
  RealT y_min2 = 0.;
  RealT z_min2 = 0.95;
  RealT x_max2 = 1.;
  RealT y_max2 = 1.;
  RealT z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now

  int test_mesh_update_err =
      this->m_mesh.tribolSetupAndUpdate( tribol::ALIGNED_MORTAR, tribol::LAGRANGE_MULTIPLIER, tribol::FRICTIONLESS,
                                         tribol::NO_SLIDING, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );

  tribol::finalize();
}

TEST_F( CompGeomTest, poly_area_centroid_1 )
{
  // This test checks the area centroid calculation
  // vs. the vertex average centroid calculation for a
  // rectangular quadrilateral. The expectation is that
  // the results are the same
  constexpr int dim = 3;
  constexpr int numVerts = 4;
  RealT x[dim * numVerts];

  for ( int i = 0; i < dim * numVerts; ++i ) {
    x[i] = 0.;
  }

  // setup some quadrilateral coordinates
  x[0] = -0.5;
  x[dim * 1] = 0.5;
  x[dim * 2] = 0.5;
  x[dim * 3] = -0.5;

  x[1] = -0.5;
  x[dim * 1 + 1] = -0.5;
  x[dim * 2 + 1] = 0.5;
  x[dim * 3 + 1] = 0.5;

  x[2] = 0.1;
  x[dim * 1 + 2] = 0.1;
  x[dim * 2 + 2] = 0.1;
  x[dim * 3 + 2] = 0.1;

  RealT cX_avg, cY_avg, cZ_avg;
  RealT cX_area, cY_area, cZ_area;

  tribol::VertexAvgCentroid( x, dim, numVerts, cX_avg, cY_avg, cZ_avg );
  tribol::PolyAreaCentroid( x, dim, numVerts, cX_area, cY_area, cZ_area );

  RealT diff[3]{ 0., 0., 0. };

  diff[0] = std::abs( cX_avg - cX_area );
  diff[1] = std::abs( cY_avg - cY_area );
  diff[2] = std::abs( cZ_avg - cZ_area );

  RealT diff_mag = tribol::magnitude( diff[0], diff[1], diff[2] );

  RealT tol = 1.e-5;
  EXPECT_LE( diff_mag, tol );
}

TEST_F( CompGeomTest, poly_area_centroid_2 )
{
  // This test checks the area centroid calculation
  // the centroid calculation for a non-self-intersecting,
  // closed polygon
  constexpr int dim = 3;
  constexpr int numVerts = 4;
  RealT x[numVerts];
  RealT y[numVerts];
  RealT z[numVerts];

  for ( int i = 0; i < numVerts; ++i ) {
    x[i] = 0.;
    y[i] = 0.;
    z[i] = 0.;
  }

  // setup some quadrilateral coordinates
  x[0] = -0.515;
  x[1] = 0.54;
  x[2] = 0.65;
  x[3] = -0.524;

  y[0] = -0.5;
  y[1] = -0.5;
  y[2] = 0.5;
  y[3] = 0.5;

  z[0] = 0.1;
  z[1] = 0.1;
  z[2] = 0.1;
  z[3] = 0.1;

  // create stacked array of coordinates
  RealT x_bar[dim * numVerts];
  for ( int i = 0; i < numVerts; ++i ) {
    x_bar[dim * i] = x[i];
    x_bar[dim * i + 1] = y[i];
    x_bar[dim * i + 2] = z[i];
  }

  RealT cX_area, cY_area, cZ_area;
  RealT cX_poly, cY_poly, cZ_poly;

  tribol::PolyAreaCentroid( x_bar, dim, numVerts, cX_area, cY_area, cZ_area );
  tribol::PolyCentroid( x, y, numVerts, cX_poly, cY_poly );

  cZ_poly = z[0];

  RealT diff[3]{ 0., 0., 0. };

  diff[0] = std::abs( cX_poly - cX_area );
  diff[1] = std::abs( cY_poly - cY_area );
  diff[2] = std::abs( cZ_poly - cZ_area );

  RealT diff_mag = tribol::magnitude( diff[0], diff[1], diff[2] );

  RealT tol = 1.e-5;
  EXPECT_LE( diff_mag, tol );
}

TEST_F( CompGeomTest, codirectional_normals_3d )
{
  // this test ensures that faces in a given face-pair with nearly co-directional
  // normals is not actually included as a contact candidate
  constexpr int numVerts = 4;
  constexpr int numCells = 2;
  constexpr int lengthNodalData = numCells * numVerts;
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  // coordinates for face 1
  x[0] = 0.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = 0.;

  y[0] = 0.;
  y[1] = 0.;
  y[2] = 1.;
  y[3] = 1.;

  z[0] = 0.;
  z[1] = 0.;
  z[2] = 0.;
  z[3] = 0.;

  // coordinates for face 2
  x[4] = 0.;
  x[5] = 1.;
  x[6] = 1.;
  x[7] = 0.;

  y[4] = 0.;
  y[5] = 0.;
  y[6] = 1.;
  y[7] = 1.;

  // amount of interpenetration in the z-direction
  z[4] = -0.300001;
  z[5] = -0.300001;
  z[6] = -0.300001;
  z[7] = -0.300001;

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  RealT dt = 1.0;
  int err = setupAndUpdateAutoCommonPlane( mesh_id, 0, lengthNodalData, numCells, dt );

  EXPECT_EQ( err, 0 );
  EXPECT_EQ( dt, 1.0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 0 );
}

TEST_F( CompGeomTest, auto_contact_lt_max_interpen )
{
  // This test uses auto-contact and checks that the face-pair
  // is included as a contact candidate, and is in fact in contact
  // when the interpenetration is less than the maximum allowable
  // for auto contact
  constexpr int numVerts = 4;
  constexpr int numCells = 2;
  constexpr int lengthNodalData = numCells * numVerts;
  RealT element_thickness[numCells];
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  for ( int i = 0; i < numCells; ++i ) {
    element_thickness[i] = 1.0;
  }

  // coordinates for face 1
  x[0] = 0.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = 0.;

  y[0] = 0.;
  y[1] = 0.;
  y[2] = 1.;
  y[3] = 1.;

  z[0] = 0.;
  z[1] = 0.;
  z[2] = 0.;
  z[3] = 0.;

  // coordinates for face 2
  x[4] = 0.;
  x[5] = 1.;
  x[6] = 1.;
  x[7] = 0.;

  y[4] = 0.;
  y[5] = 0.;
  y[6] = 1.;
  y[7] = 1.;

  // amount of interpenetration in the z-direction
  RealT max_interpen_frac = 1.0;
  RealT test_ratio = 0.90;  // fraction of max interpen frac used for this test
  z[4] = -test_ratio * max_interpen_frac * element_thickness[1];
  z[5] = -test_ratio * max_interpen_frac * element_thickness[1];
  z[6] = -test_ratio * max_interpen_frac * element_thickness[1];
  z[7] = -test_ratio * max_interpen_frac * element_thickness[1];

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 7, 6, 5 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  tribol::registerRealElementField( mesh_id, tribol::ELEMENT_THICKNESS, &element_thickness[0] );
  m_isElementThicknessRegistered = true;

  RealT dt = 1.0;
  int err = setupAndUpdateAutoCommonPlane( mesh_id, 0, lengthNodalData, numCells, dt );

  EXPECT_EQ( err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );
}

TEST_F( CompGeomTest, auto_contact_gt_max_interpen )
{
  // This test uses auto-contact and checks that the face-pair
  // is included as a contact candidate, and is in fact in contact
  // when the interpenetration is less than the maximum allowable
  // for auto contact
  constexpr int numVerts = 4;
  constexpr int numCells = 2;
  constexpr int lengthNodalData = numCells * numVerts;
  RealT element_thickness[numCells];
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  for ( int i = 0; i < numCells; ++i ) {
    element_thickness[i] = 1.0;
  }

  // coordinates for face 1
  x[0] = 0.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = 0.;

  y[0] = 0.;
  y[1] = 0.;
  y[2] = 1.;
  y[3] = 1.;

  z[0] = 0.;
  z[1] = 0.;
  z[2] = 0.;
  z[3] = 0.;

  // coordinates for face 2
  x[4] = 0.;
  x[5] = 1.;
  x[6] = 1.;
  x[7] = 0.;

  y[4] = 0.;
  y[5] = 0.;
  y[6] = 1.;
  y[7] = 1.;

  // amount of interpenetration in the z-direction
  RealT max_interpen_frac = 1.0;
  RealT test_ratio = 1.01;  // fraction of max interpen frac used for this test
  z[4] = -test_ratio * max_interpen_frac * element_thickness[1];
  z[5] = -test_ratio * max_interpen_frac * element_thickness[1];
  z[6] = -test_ratio * max_interpen_frac * element_thickness[1];
  z[7] = -test_ratio * max_interpen_frac * element_thickness[1];

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 7, 6, 5 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  tribol::registerRealElementField( mesh_id, tribol::ELEMENT_THICKNESS, &element_thickness[0] );
  m_isElementThicknessRegistered = true;

  RealT dt = 1.0;
  int err = setupAndUpdateAutoCommonPlane( mesh_id, 0, lengthNodalData, numCells, dt );

  EXPECT_EQ( err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 0 );
}

TEST_F( CompGeomTest, compute_local_basis_1 )
{
  // Test the calculation of a local basis on a plane with normal <1,1,1> that passes
  // through the origin (0,0,0)
  RealT nx = 1.;
  RealT ny = 1.;
  RealT nz = 1.;

  // make normal vector unit
  RealT inv_mag = 1. / tribol::magnitude( nx, ny, nz );
  nx *= inv_mag;
  ny *= inv_mag;
  nz *= inv_mag;

  // declare local basis vector components
  RealT e1x = 0.;
  RealT e1y = 0.;
  RealT e1z = 0.;

  RealT e2x = 0.;
  RealT e2y = 0.;
  RealT e2z = 0.;

  tribol::ComputeLocalBasis( nx, ny, nz, e1x, e1y, e1z, e2x, e2y, e2z );

  EXPECT_NEAR( tribol::magnitude( nx, ny, nz ), 1.0, 1.e-12 );
  EXPECT_NEAR( tribol::magnitude( e1x, e1y, e1z ), 1.0, 1.e-12 );
  EXPECT_NEAR( tribol::magnitude( e2x, e2y, e2z ), 1.0, 1.e-12 );
  EXPECT_NEAR( tribol::dotProd( e1x, e1y, e1z, e2x, e2y, e2z ), 0.0, 1.e-12 );
  EXPECT_NEAR( tribol::dotProd( e1x, e1y, e1z, nx, ny, nz ), 0.0, 1.e-12 );
  EXPECT_NEAR( tribol::dotProd( e2x, e2y, e2z, nx, ny, nz ), 0.0, 1.e-12 );
}

TEST_F( CompGeomTest, compute_local_basis_2 )
{
  RealT nx = 1.;
  RealT ny = 0.;
  RealT nz = 0.;

  // declare local basis vector components
  RealT e1x = 0.;
  RealT e1y = 0.;
  RealT e1z = 0.;

  RealT e2x = 0.;
  RealT e2y = 0.;
  RealT e2z = 0.;

  tribol::ComputeLocalBasis( nx, ny, nz, e1x, e1y, e1z, e2x, e2y, e2z );

  EXPECT_NEAR( tribol::magnitude( nx, ny, nz ), 1.0, 1.e-12 );
  EXPECT_NEAR( tribol::magnitude( e1x, e1y, e1z ), 1.0, 1.e-12 );
  EXPECT_NEAR( tribol::magnitude( e2x, e2y, e2z ), 1.0, 1.e-12 );
  EXPECT_NEAR( tribol::dotProd( e1x, e1y, e1z, e2x, e2y, e2z ), 0.0, 1.e-12 );
  EXPECT_NEAR( tribol::dotProd( e1x, e1y, e1z, nx, ny, nz ), 0.0, 1.e-12 );
  EXPECT_NEAR( tribol::dotProd( e2x, e2y, e2z, nx, ny, nz ), 0.0, 1.e-12 );
}

TEST_F( CompGeomTest, compute_local_basis_3 )
{
  RealT nx = 0.;
  RealT ny = 1.;
  RealT nz = 0.;

  // declare local basis vector components
  RealT e1x = 0.;
  RealT e1y = 0.;
  RealT e1z = 0.;

  RealT e2x = 0.;
  RealT e2y = 0.;
  RealT e2z = 0.;

  tribol::ComputeLocalBasis( nx, ny, nz, e1x, e1y, e1z, e2x, e2y, e2z );

  EXPECT_NEAR( tribol::magnitude( nx, ny, nz ), 1.0, 1.e-12 );
  EXPECT_NEAR( tribol::magnitude( e1x, e1y, e1z ), 1.0, 1.e-12 );
  EXPECT_NEAR( tribol::magnitude( e2x, e2y, e2z ), 1.0, 1.e-12 );
  EXPECT_NEAR( tribol::dotProd( e1x, e1y, e1z, e2x, e2y, e2z ), 0.0, 1.e-12 );
  EXPECT_NEAR( tribol::dotProd( e1x, e1y, e1z, nx, ny, nz ), 0.0, 1.e-12 );
  EXPECT_NEAR( tribol::dotProd( e2x, e2y, e2z, nx, ny, nz ), 0.0, 1.e-12 );
}

TEST_F( CompGeomTest, compute_local_basis_4 )
{
  RealT nx = 0.;
  RealT ny = 0.;
  RealT nz = 1.;

  // declare local basis vector components
  RealT e1x = 0.;
  RealT e1y = 0.;
  RealT e1z = 0.;

  RealT e2x = 0.;
  RealT e2y = 0.;
  RealT e2z = 0.;

  tribol::ComputeLocalBasis( nx, ny, nz, e1x, e1y, e1z, e2x, e2y, e2z );

  EXPECT_NEAR( tribol::magnitude( nx, ny, nz ), 1.0, 1.e-12 );
  EXPECT_NEAR( tribol::magnitude( e1x, e1y, e1z ), 1.0, 1.e-12 );
  EXPECT_NEAR( tribol::magnitude( e2x, e2y, e2z ), 1.0, 1.e-12 );
  EXPECT_NEAR( tribol::dotProd( e1x, e1y, e1z, e2x, e2y, e2z ), 0.0, 1.e-12 );
  EXPECT_NEAR( tribol::dotProd( e1x, e1y, e1z, nx, ny, nz ), 0.0, 1.e-12 );
  EXPECT_NEAR( tribol::dotProd( e2x, e2y, e2z, nx, ny, nz ), 0.0, 1.e-12 );
}

TEST_F( CompGeomTest, poly_reorder_convex_1 )
{
  constexpr int numOverlapVerts = 5;
  RealT x[numOverlapVerts];
  RealT y[numOverlapVerts];
  RealT z[numOverlapVerts];
  int new_ids[numOverlapVerts];

  // overlap coords with out of order vertices; that is, the ordering
  // does not march around the convex hull. In particular, they will
  // be reordered and in a CCW orientation. This test's input order is
  // (0,3,4,2,1) and we expect (0,1,2,3,4) output ordering
  x[0] = 0.;
  x[3] = 1.;
  x[4] = 1.5;
  x[2] = 1.25;
  x[1] = 0.5;

  y[0] = 0.;
  y[3] = -0.1;
  y[4] = 0.25;
  y[2] = 0.75;
  y[1] = 0.5;

  // dummy z-coords; ensure (x,y) in same plane; used in call to PolyReorderWithNorma()
  z[0] = 0.;
  z[1] = 0.;
  z[2] = 0.;
  z[3] = 0.;
  z[4] = 0.;

  SLIC_INFO( "Testing PolyReorderConvex()." );
  tribol::PolyReorderConvex( &x[0], &y[0], &new_ids[0], numOverlapVerts );

  EXPECT_EQ( x[0], 0. );
  EXPECT_EQ( x[1], 1. );
  EXPECT_EQ( x[2], 1.5 );
  EXPECT_EQ( x[3], 1.25 );
  EXPECT_EQ( x[4], 0.5 );

  EXPECT_EQ( y[0], 0. );
  EXPECT_EQ( y[1], -0.1 );
  EXPECT_EQ( y[2], 0.25 );
  EXPECT_EQ( y[3], 0.75 );
  EXPECT_EQ( y[4], 0.5 );

  // now test reordering with normal where the overlap vertices are now already
  // ordered correctly
  SLIC_INFO( "Testing PolyReorderWithNormal() already with CCW vertex ordering." );
  tribol::PolyReorderWithNormal( &x[0], &y[0], &z[0], numOverlapVerts, 0., 0., 1. );

  EXPECT_EQ( x[0], 0. );
  EXPECT_EQ( x[1], 1. );
  EXPECT_EQ( x[2], 1.5 );
  EXPECT_EQ( x[3], 1.25 );
  EXPECT_EQ( x[4], 0.5 );

  EXPECT_EQ( y[0], 0. );
  EXPECT_EQ( y[1], -0.1 );
  EXPECT_EQ( y[2], 0.25 );
  EXPECT_EQ( y[3], 0.75 );
  EXPECT_EQ( y[4], 0.5 );

  // now test reordering with normal where the overlap vertices need to be reordered
  // in CW orientation
  SLIC_INFO( "Testing PolyReorderWithNormal() to reorder in CW vertex ordering." );
  tribol::PolyReorderWithNormal( &x[0], &y[0], &z[0], numOverlapVerts, 0., 0., -1. );

  EXPECT_EQ( x[0], 0. );
  EXPECT_EQ( x[4], 1. );
  EXPECT_EQ( x[3], 1.5 );
  EXPECT_EQ( x[2], 1.25 );
  EXPECT_EQ( x[1], 0.5 );

  EXPECT_EQ( y[0], 0. );
  EXPECT_EQ( y[4], -0.1 );
  EXPECT_EQ( y[3], 0.25 );
  EXPECT_EQ( y[2], 0.75 );
  EXPECT_EQ( y[1], 0.5 );

  // now test reordering the vertices BACK TO CCW orientation
  SLIC_INFO( "Testing PolyReorderWithNormal() to reorder BACK to CCW vertex ordering." );
  tribol::PolyReorderWithNormal( &x[0], &y[0], &z[0], numOverlapVerts, 0., 0., 1. );

  EXPECT_EQ( x[0], 0. );
  EXPECT_EQ( x[1], 1. );
  EXPECT_EQ( x[2], 1.5 );
  EXPECT_EQ( x[3], 1.25 );
  EXPECT_EQ( x[4], 0.5 );

  EXPECT_EQ( y[0], 0. );
  EXPECT_EQ( y[1], -0.1 );
  EXPECT_EQ( y[2], 0.25 );
  EXPECT_EQ( y[3], 0.75 );
  EXPECT_EQ( y[4], 0.5 );
}

TEST_F( CompGeomTest, project_point_to_plane )
{
  // Test the point projection onto a plane with normal <1,1,1> that passes
  // through the origin (0,0,0)
  RealT nx = 1.;
  RealT ny = 1.;
  RealT nz = 1.;

  // make normal vector unit
  RealT inv_mag = 1. / tribol::magnitude( nx, ny, nz );
  nx *= inv_mag;
  ny *= inv_mag;
  nz *= inv_mag;

  RealT cx = 0.;
  RealT cy = 0.;
  RealT cz = 0.;

  RealT px = 1.;
  RealT py = 1.;
  RealT pz = 1.;

  RealT proj_px = 0;
  RealT proj_py = 0;
  RealT proj_pz = 0;

  tribol::ProjectPointToPlane( px, py, pz, nx, ny, nz, cx, cy, cz, proj_px, proj_py, proj_pz );

  EXPECT_NEAR( proj_px, 0., 1.e-12 );
  EXPECT_NEAR( proj_py, 0., 1.e-12 );
  EXPECT_NEAR( proj_pz, 0., 1.e-12 );

  // now test the projection routine for a point that is on the plane AND coincident with the centroid
  // point that defines the plane
  px = 0.;
  py = 0.;
  pz = 0.;

  tribol::ProjectPointToPlane( px, py, pz, nx, ny, nz, cx, cy, cz, proj_px, proj_py, proj_pz );

  EXPECT_NEAR( proj_px, 0., 1.e-12 );
  EXPECT_NEAR( proj_py, 0., 1.e-12 );
  EXPECT_NEAR( proj_pz, 0., 1.e-12 );
}

TEST_F( CompGeomTest, point_in_face_1 )
{
  constexpr int numOverlapVerts = 5;
  RealT x[numOverlapVerts];
  RealT y[numOverlapVerts];
  RealT xp, yp;

  x[0] = 0.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = 1.;
  x[4] = 0.;

  y[0] = 0.;
  y[1] = 0.;
  y[2] = 0.25;
  y[3] = 0.75;
  y[4] = 0.75;

  RealT xc = 0.;
  RealT yc = 0.;
  RealT zc = 0.;
  tribol::VertexAvgCentroid( x, y, nullptr, numOverlapVerts, xc, yc, zc );

  // point at the first vertex exactly
  xp = 0.;
  yp = 0.;
  EXPECT_EQ( tribol::Point2DInFace( xp, yp, x, y, xc, yc, numOverlapVerts ), true );

  // point at another vertex exactly
  xp = 1.0;
  yp = 0.25;
  EXPECT_EQ( tribol::Point2DInFace( xp, yp, x, y, xc, yc, numOverlapVerts ), true );

  // x-coord just barely off the face
  xp = 1.0000001;
  yp = 0.5;
  EXPECT_EQ( tribol::Point2DInFace( xp, yp, x, y, xc, yc, numOverlapVerts ), false );

  // x-coord just barely inside the face
  xp = 0.9999999;
  yp = 0.5;
  EXPECT_EQ( tribol::Point2DInFace( xp, yp, x, y, xc, yc, numOverlapVerts ), true );

  // x and y-coords just barely inside the face
  xp = 0.999999999999;
  yp = 0.749999999999;
  EXPECT_EQ( tribol::Point2DInFace( xp, yp, x, y, xc, yc, numOverlapVerts ), true );

  // x and y-coords just barely outside the face; NOTE: past 12 digits will typically result
  // in the point being picked up as in the face.
  xp = 1.000000000001;
  yp = 0.750000000001;
  EXPECT_EQ( tribol::Point2DInFace( xp, yp, x, y, xc, yc, numOverlapVerts ), false );
}

TEST_F( CompGeomTest, line_plane_intersection_1 )
{
  // this tests edge cases where a line has a vertex that lies just outside, just on, and then
  // just through the plane
  RealT interx, intery, interz;
  bool in_plane = false;

  // define a plane with normal in the z-direction
  RealT nx = 0.;
  RealT ny = 0.;
  RealT nz = 1.;

  // define plane reference point as origin
  RealT cx = 0.;
  RealT cy = 0.;
  RealT cz = 0.;

  RealT xa, ya, za;
  RealT xb, yb, zb;

  // line passes through plane
  xa = 0.;
  ya = 0.;
  za = -1;
  xb = 0.;
  yb = 0.;
  zb = 1.;
  EXPECT_EQ(
      tribol::LinePlaneIntersection( xa, ya, za, xb, yb, zb, cx, cy, cz, nx, ny, nz, interx, intery, interz, in_plane ),
      true );

  // second vertex is on plane
  xa = 0.;
  ya = 0.;
  za = -1;
  xb = 0.;
  yb = 0.;
  zb = 0.;
  EXPECT_EQ(
      tribol::LinePlaneIntersection( xa, ya, za, xb, yb, zb, cx, cy, cz, nx, ny, nz, interx, intery, interz, in_plane ),
      true );

  // first vertex is on plane
  xa = 0.;
  ya = 0.;
  za = 0.;
  xb = 0.;
  yb = 0.;
  zb = 1.0;
  EXPECT_EQ(
      tribol::LinePlaneIntersection( xa, ya, za, xb, yb, zb, cx, cy, cz, nx, ny, nz, interx, intery, interz, in_plane ),
      true );

  // second vertex just barely passes through the plane
  xa = 0.;
  ya = 0.;
  za = -1;
  xb = 0.;
  yb = 0.;
  zb = 0.000000000001;
  EXPECT_EQ(
      tribol::LinePlaneIntersection( xa, ya, za, xb, yb, zb, cx, cy, cz, nx, ny, nz, interx, intery, interz, in_plane ),
      true );

  // second vertex just barely outside of passing through plane
  xa = 0.;
  ya = 0.;
  za = -1.;
  xb = 0.;
  yb = 0.;
  zb = -0.000000000001;
  EXPECT_EQ(
      tribol::LinePlaneIntersection( xa, ya, za, xb, yb, zb, cx, cy, cz, nx, ny, nz, interx, intery, interz, in_plane ),
      false );

  // the edge lies in the plane. We expect the in_plane boolean to be true, but there
  // to be no line-plane intersection points
  xa = 0.;
  ya = 0.;
  za = 0.;
  xb = 1.;
  yb = 0.;
  zb = 0.;
  EXPECT_EQ(
      tribol::LinePlaneIntersection( xa, ya, za, xb, yb, zb, cx, cy, cz, nx, ny, nz, interx, intery, interz, in_plane ),
      false );
  EXPECT_EQ( in_plane, true );

  // the edge lies in the plane. We expect the in_plane boolean to be true, but there
  // to be no line-plane intersection points
  xa = 0.;
  ya = 0.;
  za = -0.00000001;
  xb = 1.;
  yb = 0.;
  zb = -0.00000001;
  EXPECT_EQ(
      tribol::LinePlaneIntersection( xa, ya, za, xb, yb, zb, cx, cy, cz, nx, ny, nz, interx, intery, interz, in_plane ),
      false );
  EXPECT_EQ( in_plane, true );
}

TEST_F( CompGeomTest, convexity_1 )
{
  //
  //   *           *
  //   **        **
  //    * *   *  *
  //     *  *  *
  //      *  *
  //       *
  //
  constexpr int numVerts = 4;
  RealT x[numVerts];
  RealT y[numVerts];

  x[0] = 0.;
  x[1] = -0.5;
  x[2] = 0.;
  x[3] = 0.5;

  y[0] = -0.75;
  y[1] = 0.5;
  y[2] = 0.;
  y[3] = 0.5;

  EXPECT_EQ( tribol::IsConvex( x, y, numVerts ), false );
}

TEST_F( CompGeomTest, convexity_2 )
{
  //
  //
  //    *       *
  //
  //
  //
  //    *       *
  //
  constexpr int numVerts = 4;
  RealT x[numVerts];
  RealT y[numVerts];

  x[0] = 0.;
  x[1] = 1.;
  x[2] = 1.;
  x[3] = 0.;

  y[0] = 0.;
  y[1] = 0.;
  y[2] = 1.;
  y[3] = 1.;

  EXPECT_EQ( tribol::IsConvex( x, y, numVerts ), true );
}

TEST_F( CompGeomTest, convexity_3 )
{
  //
  //
  //                 *
  //
  //
  //    *       *         *
  //
  constexpr int numVerts = 4;
  RealT x[numVerts];
  RealT y[numVerts];

  x[0] = 0.;
  x[1] = 1.;
  x[2] = 2.;
  x[3] = 1.5;

  y[0] = 0.;
  y[1] = 0.;
  y[2] = 0.;
  y[3] = 1.;

  EXPECT_EQ( tribol::IsConvex( x, y, numVerts ), true );
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
