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

 protected:
  void SetUp() override {}

  void TearDown() override
  {
    // call clear() on mesh object to be safe
    this->m_mesh.clear();
  }

 protected:
};

//TEST_F( CompGeomTest, common_plane_single_element_full_overlap_check_1 )
//{
//  // mesh bounding box with 0.1 interpenetration gap. The contact faces
//  // just have a y-shift and overlap will have nodes that lie on edge 
//  // segments of the opposing face
//  int nMortarElems = 1;
//  int nElemsXM = nMortarElems;
//  int nElemsYM = 1;
//  int nElemsZM = nMortarElems;
//
//  int nNonmortarElems = 1;
//  int nElemsXS = nNonmortarElems;
//  int nElemsYS = 1;
//  int nElemsZS = nNonmortarElems;
//
//  int userSpecifiedNumOverlaps = 1;
//
//  RealT x_min1 = 0.;
//  RealT y_min1 = 0.;
//  RealT z_min1 = 0.;
//  RealT x_max1 = 1.;
//  RealT y_max1 = 1.;
//  RealT z_max1 = 1.05;
//
//  RealT x_min2 = 0.;
//  RealT y_min2 = 0.5; // perform 0.5 shift in y direction
//  RealT z_min2 = 0.95;
//  RealT x_max2 = 1.0;
//  RealT y_max2 = y_min2 + 1.0;
//  RealT z_max2 = 2.;
//
//  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
//                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
//                                    0. );
//
//  // call tribol setup and update
//  tribol::TestControlParameters parameters;  // struct does not hold info right now
//  parameters.penalty_ratio = false;
//  parameters.const_penalty = 1.0;
//
//  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
//      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );
//
//  EXPECT_EQ( test_mesh_update_err, 0 );
//
//  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();
//
//  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
//
//  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );
//
//  auto& comp_geom= couplingScheme->getCompGeom();
//  auto& plane = comp_geom.getCommonPlane( 0 );
//  
//  RealT computed_gap = -(z_max1 - z_min2);
//  RealT gap_diff = std::abs(plane.m_gap - computed_gap);
//  EXPECT_LE( gap_diff, 1.e-8 );
//  
//  RealT area_diff = std::abs(plane.m_area - 0.5);
//  EXPECT_LE( area_diff, 1.e-10 );
//
//  tribol::finalize();
//}
//
//TEST_F( CompGeomTest, common_plane_single_element_full_overlap_check_2 )
//{
//  // mesh bounding box with 0.1 interpenetration gap. The faces will
//  // have an x and y shift such that no nodes overlap with nodes/segments of
//  // the opposing face
//  int nMortarElems = 1;
//  int nElemsXM = nMortarElems;
//  int nElemsYM = 1;
//  int nElemsZM = nMortarElems;
//
//  int nNonmortarElems = 1;
//  int nElemsXS = nNonmortarElems;
//  int nElemsYS = 1;
//  int nElemsZS = nNonmortarElems;
//
//  int userSpecifiedNumOverlaps = 1;
//
//  RealT x_min1 = 0.;
//  RealT y_min1 = 0.;
//  RealT z_min1 = 0.;
//  RealT x_max1 = 1.;
//  RealT y_max1 = 1.;
//  RealT z_max1 = 1.05;
//
//  RealT x_min2 = -0.9; // x-shift
//  RealT y_min2 = -0.9; // y-shift 
//  RealT z_min2 = 0.95;
//  RealT x_max2 = 1.0 + x_min2;
//  RealT y_max2 = y_min2 + 1.0;
//  RealT z_max2 = 2.;
//
//  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
//                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
//                                    0. );
//
//  // call tribol setup and update
//  tribol::TestControlParameters parameters;  // struct does not hold info right now
//  parameters.penalty_ratio = false;
//  parameters.const_penalty = 1.0;
//
//  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
//      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );
//
//  EXPECT_EQ( test_mesh_update_err, 0 );
//
//  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();
//
//  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
//
//  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );
//
//  auto& comp_geom= couplingScheme->getCompGeom();
//  auto& plane = comp_geom.getCommonPlane( 0 );
//  
//  RealT computed_gap = -(z_max1 - z_min2);
//  RealT gap_diff = std::abs(plane.m_gap - computed_gap);
//  EXPECT_LE( gap_diff, 1.e-8 );
//
//  RealT area_diff = std::abs( plane.m_area - (x_max2 - x_min1) * (y_max2 - y_min1) );
//  EXPECT_LE( area_diff, 1.e-10 );
//
//  tribol::finalize();
//}
//
//TEST_F( CompGeomTest, common_plane_single_element_full_separation_check_1 )
//{
//  // mesh bounding box with 0.1 separation gap and no x/y shift
//  int nMortarElems = 1;
//  int nElemsXM = nMortarElems;
//  int nElemsYM = 1;
//  int nElemsZM = nMortarElems;
//
//  int nNonmortarElems = 1;
//  int nElemsXS = nNonmortarElems;
//  int nElemsYS = 1;
//  int nElemsZS = nNonmortarElems;
//
//  int userSpecifiedNumOverlaps = 1;
//
//  RealT x_min1 = 0.;
//  RealT y_min1 = 0.;
//  RealT z_min1 = 0.;
//  RealT x_max1 = 1.;
//  RealT y_max1 = 1.;
//  RealT z_max1 = 1.0;
//
//  RealT x_min2 = 0.;
//  RealT y_min2 = 0.; 
//  RealT z_min2 = 1.1;
//  RealT x_max2 = 1.0;
//  RealT y_max2 = 1.0;
//  RealT z_max2 = 2.1;
//
//  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
//                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
//                                    0. );
//
//  // call tribol setup and update
//  tribol::TestControlParameters parameters;  // struct does not hold info right now
//  parameters.penalty_ratio = false;
//  parameters.const_penalty = 1.0;
//
//  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
//      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );
//
//  EXPECT_EQ( test_mesh_update_err, 0 );
//
//  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();
//
//  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
//
//  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );
//
//  auto& comp_geom= couplingScheme->getCompGeom();
//  auto& plane = comp_geom.getCommonPlane( 0 );
//  
//  RealT computed_gap = -(z_max1 - z_min2);
//  RealT gap_diff = std::abs(plane.m_gap - computed_gap);
//  EXPECT_LE( gap_diff, 1.e-8 );
//
//  RealT area_diff = std::abs( plane.m_area - (x_max2 - x_min1) * (y_max2 - y_min1) );
//  EXPECT_LE( area_diff, 1.e-10 );
//
//  tribol::finalize();
//}
//
//TEST_F( CompGeomTest, common_plane_single_element_full_separation_check_2 )
//{
//  // mesh bounding box with 0.1 separation gap. The faces will
//  // have an x and y shift
//  int nMortarElems = 1;
//  int nElemsXM = nMortarElems;
//  int nElemsYM = 1;
//  int nElemsZM = nMortarElems;
//
//  int nNonmortarElems = 1;
//  int nElemsXS = nNonmortarElems;
//  int nElemsYS = 1;
//  int nElemsZS = nNonmortarElems;
//
//  int userSpecifiedNumOverlaps = 1;
//
//  RealT x_min1 = 0.;
//  RealT y_min1 = 0.;
//  RealT z_min1 = 0.;
//  RealT x_max1 = 1.;
//  RealT y_max1 = 1.;
//  RealT z_max1 = 1.0;
//
//  RealT x_min2 = -0.9; // x-shift
//  RealT y_min2 = -0.9; // y-shift 
//  RealT z_min2 = 1.1;
//  RealT x_max2 = 1.0 + x_min2;
//  RealT y_max2 = y_min2 + 1.0;
//  RealT z_max2 = 2.1;
//
//  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
//                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
//                                    0. );
//
//  // call tribol setup and update
//  tribol::TestControlParameters parameters;  // struct does not hold info right now
//  parameters.penalty_ratio = false;
//  parameters.const_penalty = 1.0;
//
//  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
//      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );
//
//  EXPECT_EQ( test_mesh_update_err, 0 );
//
//  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();
//
//  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
//
//  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );
//
//  auto& comp_geom= couplingScheme->getCompGeom();
//  auto& plane = comp_geom.getCommonPlane( 0 );
//  
//  RealT computed_gap = -(z_max1 - z_min2);
//  RealT gap_diff = std::abs(plane.m_gap - computed_gap);
//  EXPECT_LE( gap_diff, 1.e-8 );
//
//  RealT area_diff = std::abs( plane.m_area - (x_max2 - x_min1) * (y_max2 - y_min1) );
//  EXPECT_LE( area_diff, 1.e-10 );
//
//  tribol::finalize();
//}
//
//TEST_F( CompGeomTest, common_plane_single_element_interpen_check_1 )
//{
//  // This test checks the interpen overlap case where each faces has
//  // TWO line segments that intersect segments on the opposing face.
//  // This can occur when the faces have identical dimensions in one
//  // coordinate direction
//
//  // The bottom block of this two-block problem is rotated clockwise
//  // about the y-axis 45 degrees; then the top block is shifted such that the
//  // centroids of the two contact surfaces are coincident in space. 
//
//  // An orthogonal edge on views of this interaction is something like:
//
//         *            * * * * * * * *
//          *           *             *
//           *          *             *
//      -------------   o-------------o
//             *        *             *
//              *       *             *
//               *      * * * * * * * *
//
//  int nMortarElems = 1;
//  int nElemsXM = nMortarElems;
//  int nElemsYM = 1;
//  int nElemsZM = nMortarElems;
//
//  int nNonmortarElems = 1;
//  int nElemsXS = nNonmortarElems;
//  int nElemsYS = 1;
//  int nElemsZS = nNonmortarElems;
//
//  int userSpecifiedNumOverlaps = 1;
//
//  RealT x_min1 = 0.;
//  RealT y_min1 = 0.;
//  RealT z_min1 = 0.;
//  RealT x_max1 = 1.;
//  RealT y_max1 = 1.;
//  RealT z_max1 = 1.0;
//
//  RealT x_min2 = 0.;
//  RealT y_min2 = 0.; 
//  RealT z_min2 = 1.0;
//  RealT x_max2 = 1.0;
//  RealT y_max2 = 1.0;
//  RealT z_max2 = 2.1;
//
//  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
//                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
//                                    0. );
//  
//  RealT theta_y = 45.;
//  m_mesh.rotateContactMesh( 0, 0., theta_y, 0. );
//
//  RealT shiftx = ( 0.7071 - 0.5 ) + 0.5 / 1.41421356;
//  RealT shiftz = ( 1.0 - 0.7071 ) + 0.5 / 1.41421356;
//  m_mesh.translateContactMesh( 1, shiftx, 0, -shiftz );
//
//  // call tribol setup and update
//  tribol::TestControlParameters parameters;  // struct does not hold info right now
//  parameters.penalty_ratio = false;
//  parameters.const_penalty = 1.0;
//
//  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
//      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );
//
//  EXPECT_EQ( test_mesh_update_err, 0 );
//
//  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();
//
//  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
//
//  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );
//
//  auto& comp_geom= couplingScheme->getCompGeom();
//  auto& plane = comp_geom.getCommonPlane( 0 );
//
//  std::cout << "overlap area: " << plane.m_area << std::endl;;
//
//  RealT hypotenuse = 0.5;
//  RealT overlap_gap_point = 0.5 * hypotenuse * std::cos(0.5*theta_y * M_PI / 180);
//
//  std::cout << "overlap_gap_point: " << overlap_gap_point << std::endl;
//  std::cout << "2x overlap_gap_point: " << 2*overlap_gap_point << std::endl;
//  RealT gap_computed = -2. * overlap_gap_point * std::tan(0.5*theta_y * M_PI / 180);
//  std::cout << "gap_computed: " << gap_computed << std::endl;
//  std::cout << "plane.m_gap: " << plane.m_gap << std::endl;
//  RealT gap_diff = std::abs(plane.m_gap - gap_computed);
//  EXPECT_LE( gap_diff, 1.e-5 );
//
//  RealT area_diff = std::abs( plane.m_area - 2.*overlap_gap_point );
//  EXPECT_LE( area_diff, 1.e-5 );
//
//  tribol::finalize();
//}

//TEST_F( CompGeomTest, common_plane_single_element_interpen_check_2 )
//{
//  // This test checks where one face has two line-plane intersections inside
//  // the opposing face and the other does not. Specifically this test has
//  // one face that is smaller than the other and then rotated similar to 
//  // interpen_check_1 above, but the intersection points lie inside the
//  // other face and not on its outer segments. The two orthogonal edge-on
//  // views of the interaction are:
//  //
//  //               *           * * * * * *
//  //             *             *         *
//  //           *               *         *
//  //  -------o--------       --o---------o--
//  //       *                   *         *
//  //     *                     *         *
//  //   *                       * * * * * *
//  //
//  //
//
//  constexpr int numVerts = 4;
//  constexpr int numCells = 2;
//  constexpr int lengthNodalData = numCells * numVerts;
//  RealT element_thickness[numCells];
//  RealT x[lengthNodalData];
//  RealT y[lengthNodalData];
//  RealT z[lengthNodalData];
//
//  for ( int i = 0; i < numCells; ++i ) {
//    element_thickness[i] = 10.0;
//  }
//
//  // coordinates for face 1
//  x[0] = 0.;
//  x[1] = 1.;
//  x[2] = 1.;
//  x[3] = 0.;
//
//  y[0] = 0.;
//  y[1] = 0.;
//  y[2] = 1.;
//  y[3] = 1.;
//
//  z[0] = 0.;
//  z[1] = 0.;
//  z[2] = 0.;
//  z[3] = 0.;
//
//  // coordinates for face 2
//  RealT fortyfive = 45 * M_PI/180;
//  x[4] = 0.33;
//  y[4] = 0.25;
//  z[4] = -0.25;
//
//  x[5] = x[4];
//  y[5] = y[4] + 0.5;
//  z[5] = z[4]; 
//
//  x[6] = x[5];
//  y[6] = y[5];
//  z[6] = 1.0;
//
//  x[7] = x[4];
//  y[7] = y[4];
//  z[7] = z[6]; 
//
//  // rotate 45 degrees about the y-axis
//  RealT x_shift = x[4];
//  RealT z_shift = z[4];
//  for (int i=numVerts; i<lengthNodalData; ++i) {
//    x[i] = x[i] - x_shift;
//    z[i] = z[i] - z_shift;
//    RealT x_rot = x[i] * std::cos(fortyfive) + z[i] * std::sin(fortyfive);
//    RealT z_rot = x[i] * -std::sin(fortyfive) + z[i] * std::cos(fortyfive);
//    x[i] = x_rot + x_shift;
//    z[i] = z_rot + z_shift;
//  }
//
//  // Debug
//  for (int i=0; i<numVerts; ++i) {
//    std::cout << x[i] << " " << y[i] << " " << z[i] << std::endl;
//  }
//  for (int i=numVerts; i<lengthNodalData; ++i) {
//    std::cout << x[i] << " " << y[i] << " " << z[i] << std::endl;
//  }
//
//  // register contact mesh
//  tribol::IndexT mesh_id = 0;
//  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
//  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
//                        tribol::MemorySpace::Host );
//
//  RealT* fx;
//  RealT* fy;
//  RealT* fz;
//  tribol::allocRealArray( &fx, lengthNodalData, 0. );
//  tribol::allocRealArray( &fy, lengthNodalData, 0. );
//  tribol::allocRealArray( &fz, lengthNodalData, 0. );
//
//  tribol::registerNodalResponse( mesh_id, fx, fy, fz );
//
//  RealT* vx;
//  RealT* vy;
//  RealT* vz;
//  RealT vel0 = -1.;
//  tribol::allocRealArray( &vx, lengthNodalData, 0. );
//  tribol::allocRealArray( &vy, lengthNodalData, 0. );
//  tribol::allocRealArray( &vz, lengthNodalData, vel0 );
//
//  // set second face to impacting velocity
//  RealT vel2 = 1.0;
//  vz[4] = vel2;
//  vz[5] = vel2;
//  vz[6] = vel2;
//  vz[7] = vel2;
//
//  tribol::registerNodalVelocities( mesh_id, vx, vy, vz );
//
//  // register element thickness for use with auto contact
//  tribol::registerRealElementField( mesh_id, tribol::ELEMENT_THICKNESS, &element_thickness[0] );
//
//  int csIndex = 0;
//  tribol::registerCouplingScheme( csIndex, mesh_id, mesh_id, tribol::SURFACE_TO_SURFACE, tribol::AUTO,
//                                  tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
//                                  tribol::BINNING_CARTESIAN_PRODUCT, tribol::ExecutionMode::Sequential );
//
//  RealT max_interpen_frac = 1.0;
//  tribol::setAutoContactPenScale( csIndex, max_interpen_frac );
//
//  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT, tribol::NO_RATE_PENALTY );
//
//  tribol::setKinematicConstantPenalty( mesh_id, 1.0 );
//
//  RealT dt = 1.0;
//  int err = tribol::update( 1, 1., dt );
//
//  EXPECT_EQ( err, 0 );
//  EXPECT_EQ( dt, 1.0 );
//
//  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();
//
//  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
//
//  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );
//
//  auto& comp_geom= couplingScheme->getCompGeom();
//  auto& plane = comp_geom.getCommonPlane( 0 );
//
//  std::cout << "overlap area: " << plane.m_area << std::endl;;
//
//  std::cout << "num_poly_vert: " << plane.m_numPolyVert << std::endl;
//
//  std::cout << "plane.m_gap: " << plane.m_gap << std::endl;
//
//  RealT h = 0.25 / std::cos(fortyfive);
//  RealT h_bar = h * std::cos(0.5*fortyfive);
//  RealT A = h_bar * 0.5;
//  std::cout << "A: " << A << std::endl;
//  RealT area_diff = std::abs( A - plane.m_area );
//  EXPECT_LE( area_diff, 1.e-8 );
//  
//  RealT computed_gap = -h_bar * std::tan(0.5*fortyfive);
//  std::cout << "computed_gap: " << computed_gap << std::endl;
//  RealT gap_diff = std::abs( plane.m_gap - computed_gap );
//  EXPECT_LE( gap_diff, 1.e-6 );
//
//  delete[] fx;
//  delete[] fy;
//  delete[] fz;
//  delete[] vx;
//  delete[] vy;
//  delete[] vz;
//}

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
  RealT element_thickness[numCells];
  RealT x[lengthNodalData];
  RealT y[lengthNodalData];
  RealT z[lengthNodalData];

  for ( int i = 0; i < numCells; ++i ) {
    element_thickness[i] = 10.0;
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
  RealT fortyfive = 45 * M_PI/180;
  x[4] = 0.33;
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
  for (int i=numVerts; i<lengthNodalData; ++i) {
    x[i] = x[i] - x_shift;
    z[i] = z[i] - z_shift;
    RealT x_rot = x[i] * std::cos(fortyfive) + z[i] * std::sin(fortyfive);
    RealT z_rot = x[i] * -std::sin(fortyfive) + z[i] * std::cos(fortyfive);
    x[i] = x_rot + x_shift;
    z[i] = z_rot + z_shift;
  }

  // Debug
  for (int i=0; i<numVerts; ++i) {
    std::cout << x[i] << " " << y[i] << " " << z[i] << std::endl;
  }
  for (int i=numVerts; i<lengthNodalData; ++i) {
    std::cout << x[i] << " " << y[i] << " " << z[i] << std::endl;
  }

  // register contact mesh
  tribol::IndexT mesh_id = 0;
  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
                        tribol::MemorySpace::Host );

  RealT* fx;
  RealT* fy;
  RealT* fz;
  tribol::allocRealArray( &fx, lengthNodalData, 0. );
  tribol::allocRealArray( &fy, lengthNodalData, 0. );
  tribol::allocRealArray( &fz, lengthNodalData, 0. );

  tribol::registerNodalResponse( mesh_id, fx, fy, fz );

  RealT* vx;
  RealT* vy;
  RealT* vz;
  RealT vel0 = -1.;
  tribol::allocRealArray( &vx, lengthNodalData, 0. );
  tribol::allocRealArray( &vy, lengthNodalData, 0. );
  tribol::allocRealArray( &vz, lengthNodalData, vel0 );

  // set second face to impacting velocity
  RealT vel2 = 1.0;
  vz[4] = vel2;
  vz[5] = vel2;
  vz[6] = vel2;
  vz[7] = vel2;

  tribol::registerNodalVelocities( mesh_id, vx, vy, vz );

  // register element thickness for use with auto contact
  tribol::registerRealElementField( mesh_id, tribol::ELEMENT_THICKNESS, &element_thickness[0] );

  int csIndex = 0;
  tribol::registerCouplingScheme( csIndex, mesh_id, mesh_id, tribol::SURFACE_TO_SURFACE, tribol::AUTO,
                                  tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
                                  tribol::BINNING_CARTESIAN_PRODUCT, tribol::ExecutionMode::Sequential );

  RealT max_interpen_frac = 1.0;
  tribol::setAutoContactPenScale( csIndex, max_interpen_frac );

  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT, tribol::NO_RATE_PENALTY );

  tribol::setKinematicConstantPenalty( mesh_id, 1.0 );

  RealT dt = 1.0;
  int err = tribol::update( 1, 1., dt );

  EXPECT_EQ( err, 0 );
  EXPECT_EQ( dt, 1.0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );

  auto& comp_geom= couplingScheme->getCompGeom();
  auto& plane = comp_geom.getCommonPlane( 0 );

  std::cout << "overlap area: " << plane.m_area << std::endl;;

  std::cout << "num_poly_vert: " << plane.m_numPolyVert << std::endl;

  std::cout << "plane.m_gap: " << plane.m_gap << std::endl;

  RealT h = 0.25 / std::cos(fortyfive);
  RealT h_bar = h * std::cos(0.5*fortyfive);
  RealT A = h_bar * 0.5;
  std::cout << "A: " << A << std::endl;
  RealT area_diff = std::abs( A - plane.m_area );
  EXPECT_LE( area_diff, 1.e-8 );
  
  RealT computed_gap = -h_bar * std::tan(0.5*fortyfive);
  std::cout << "computed_gap: " << computed_gap << std::endl;
  RealT gap_diff = std::abs( plane.m_gap - computed_gap );
  EXPECT_LE( gap_diff, 1.e-6 );

  delete[] fx;
  delete[] fy;
  delete[] fz;
  delete[] vx;
  delete[] vy;
  delete[] vz;
}

//TEST_F( CompGeomTest, common_plane_single_element_interpen_check_4 )
//{
//  // This test checks the case where one face has two-line plane intersections
//  // inside the opposing face and the opposing face has zero that lie inside the
//  // other face.
//  //
//  // Specifically, this test checks the interaction between a flat face and a rotated
//  // face where one node interpenetrates the opposing flat face. This
//  // forms a triangular intersection. An edge on view looks like:
//  //
//  //              *
//  //            *   *
//  //          *       *
//  //        *           *
//  //      *               *
//  //    ----o-----------o----
//  //          *       *
//  //            *   *
//  //              *
//  //
//  //  where the rotated face is also rotated into the page 30 degrees. The
//  //  intersection points are marked as "o".
//  //
//  constexpr int numVerts = 4;
//  constexpr int numCells = 2;
//  constexpr int lengthNodalData = numCells * numVerts;
//  RealT element_thickness[numCells];
//  RealT x[lengthNodalData];
//  RealT y[lengthNodalData];
//  RealT z[lengthNodalData];
//
//  for ( int i = 0; i < numCells; ++i ) {
//    element_thickness[i] = 10.0;
//  }
//
//  // coordinates for face 1
//  x[0] = 0.;
//  x[1] = 1.;
//  x[2] = 1.;
//  x[3] = 0.;
//
//  y[0] = 0.;
//  y[1] = 0.;
//  y[2] = 1.;
//  y[3] = 1.;
//
//  z[0] = 0.;
//  z[1] = 0.;
//  z[2] = 0.;
//  z[3] = 0.;
//
//  // coordinates for face 2
//  RealT thirty = 30 * M_PI/180;
//  RealT fortyfive = 45 * M_PI/180;
//  x[4] = 0.33;
//  y[4] = 0.5;
//  z[4] = -0.25;
//
//  x[5] = x[4];
//  y[5] = y[4] + 0.5 * std::tan(fortyfive);
//  z[5] = 0.25; 
//
//  x[6] = x[4];
//  y[6] = y[4];
//  z[6] = 0.75;
//
//  x[7] = x[5];
//  y[7] = y[4] - 0.5 * std::tan(fortyfive);
//  z[7] = 0.25;
//
//  // rotate 30 degrees about the y-axis
//  RealT x_shift = x[4];
//  RealT z_shift = z[4];
//  for (int i=numVerts; i<lengthNodalData; ++i) {
//    x[i] = x[i] - x_shift;
//    z[i] = z[i] - z_shift;
//    RealT x_rot = x[i] * std::cos(thirty) + z[i] * std::sin(thirty);
//    RealT z_rot = x[i] * -std::sin(thirty) + z[i] * std::cos(thirty);
//    x[i] = x_rot + x_shift;
//    z[i] = z_rot + z_shift;
//  }
//
//  // Debug
//  for (int i=0; i<numVerts; ++i) {
//    std::cout << x[i] << " " << y[i] << " " << z[i] << std::endl;
//  }
//  for (int i=numVerts; i<lengthNodalData; ++i) {
//    std::cout << x[i] << " " << y[i] << " " << z[i] << std::endl;
//  }
//
//  // register contact mesh
//  tribol::IndexT mesh_id = 0;
//  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
//  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
//                        tribol::MemorySpace::Host );
//
//  RealT* fx;
//  RealT* fy;
//  RealT* fz;
//  tribol::allocRealArray( &fx, lengthNodalData, 0. );
//  tribol::allocRealArray( &fy, lengthNodalData, 0. );
//  tribol::allocRealArray( &fz, lengthNodalData, 0. );
//
//  tribol::registerNodalResponse( mesh_id, fx, fy, fz );
//
//  RealT* vx;
//  RealT* vy;
//  RealT* vz;
//  RealT vel0 = -1.;
//  tribol::allocRealArray( &vx, lengthNodalData, 0. );
//  tribol::allocRealArray( &vy, lengthNodalData, 0. );
//  tribol::allocRealArray( &vz, lengthNodalData, vel0 );
//
//  // set second face to impacting velocity
//  RealT vel2 = 1.0;
//  vz[4] = vel2;
//  vz[5] = vel2;
//  vz[6] = vel2;
//  vz[7] = vel2;
//
//  tribol::registerNodalVelocities( mesh_id, vx, vy, vz );
//
//  // register element thickness for use with auto contact
//  tribol::registerRealElementField( mesh_id, tribol::ELEMENT_THICKNESS, &element_thickness[0] );
//
//  int csIndex = 0;
//  tribol::registerCouplingScheme( csIndex, mesh_id, mesh_id, tribol::SURFACE_TO_SURFACE, tribol::AUTO,
//                                  tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
//                                  tribol::BINNING_CARTESIAN_PRODUCT, tribol::ExecutionMode::Sequential );
//
//  RealT max_interpen_frac = 1.0;
//  tribol::setAutoContactPenScale( csIndex, max_interpen_frac );
//
//  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT, tribol::NO_RATE_PENALTY );
//
//  tribol::setKinematicConstantPenalty( mesh_id, 1.0 );
//
//  RealT dt = 1.0;
//  int err = tribol::update( 1, 1., dt );
//
//  EXPECT_EQ( err, 0 );
//  EXPECT_EQ( dt, 1.0 );
//
//  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();
//
//  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
//
//  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );
//
//  auto& comp_geom= couplingScheme->getCompGeom();
//  auto& plane = comp_geom.getCommonPlane( 0 );
//
//  std::cout << "overlap area: " << plane.m_area << std::endl;;
//
//  std::cout << "num_poly_vert: " << plane.m_numPolyVert << std::endl;
//
//  std::cout << "plane.m_gap: " << plane.m_gap << std::endl;
//
//  RealT h = 0.25;
//  RealT h_bar = h / std::cos(thirty);
//  RealT w = h * std::tan(fortyfive);
//  RealT h_bar_bar = h_bar * std::cos(thirty);
//  RealT w_bar = h_bar * std::tan(fortyfive);
//  RealT A = w * h;
//  RealT A_bar = w_bar * h_bar;
//  RealT A_bar_bar = h_bar_bar * w_bar;
//
//  //std::cout << "heights (h, h_bar, h_bar_bar): " << h << ", " << h_bar << ", " << h_bar_bar << std::endl;
//  //std::cout << "areas (A, A_bar, A_bar_bar): " << A << ", " << A_bar << ", " << A_bar_bar << std::endl;
//
//  RealT computed_gap = -2*(0.33333*h_bar_bar) * std::tan(thirty);
//  std::cout << "computed_gap: " << computed_gap << std::endl;
//
//  std::cout << "computed overlap area: " << A_bar_bar << std::endl;
//
//  RealT area_diff = std::abs( A_bar_bar - plane.m_area );
//  EXPECT_LE( area_diff, 1.e-8 );
//  
//  RealT gap_diff = std::abs( plane.m_gap - computed_gap );
//  EXPECT_LE( gap_diff, 1.e-6 );
//
//  delete[] fx;
//  delete[] fy;
//  delete[] fz;
//  delete[] vx;
//  delete[] vy;
//  delete[] vz;
//}
//
//TEST_F( CompGeomTest, common_plane_perfect_conforming_full_overlap )
//{
//  int nMortarElems = 3;
//  int nElemsXM = nMortarElems;
//  int nElemsYM = 3;
//  int nElemsZM = nMortarElems;
//
//  int nNonmortarElems = 3;
//  int nElemsXS = nNonmortarElems;
//  int nElemsYS = 3;
//  int nElemsZS = nNonmortarElems;
//
//  int userSpecifiedNumOverlaps = 9;
//
//  // mesh bounding box with 0.1 interpenetration gap
//  RealT x_min1 = 0.;
//  RealT y_min1 = 0.;
//  RealT z_min1 = 0.;
//  RealT x_max1 = 1.;
//  RealT y_max1 = 1.;
//  RealT z_max1 = 1.05;
//
//  RealT x_min2 = 0.;
//  RealT y_min2 = 0.;
//  RealT z_min2 = 0.95;
//  RealT x_max2 = 1.0;
//  RealT y_max2 = 1.0;
//  RealT z_max2 = 2.;
//
//  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
//                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
//                                    0. );
//
//  // call tribol setup and update
//  tribol::TestControlParameters parameters;  // struct does not hold info right now
//  parameters.penalty_ratio = false;
//  parameters.const_penalty = 1.0;
//
//  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
//      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );
//
//  EXPECT_EQ( test_mesh_update_err, 0 );
//
//  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();
//
//  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
//
//  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );
//
//  tribol::finalize();
//}
//
//TEST_F( CompGeomTest, common_plane_xy_shift_full_overlap )
//{
//  int nMortarElems = 3;
//  int nElemsXM = nMortarElems;
//  int nElemsYM = 3;
//  int nElemsZM = nMortarElems;
//
//  int nNonmortarElems = 3;
//  int nElemsXS = nNonmortarElems;
//  int nElemsYS = 3;
//  int nElemsZS = nNonmortarElems;
//
//  int userSpecifiedNumOverlaps = 25;
//
//  // mesh bounding box with 0.1 interpenetration gap
//  RealT x_min1 = 0.;
//  RealT y_min1 = 0.;
//  RealT z_min1 = 0.;
//  RealT x_max1 = 1.;
//  RealT y_max1 = 1.;
//  RealT z_max1 = 1.05;
//
//  RealT x_min2 = -0.1;
//  RealT y_min2 = 0.0001;
//  RealT z_min2 = 0.95;
//  RealT x_max2 = 0.9;
//  RealT y_max2 = 1.0001;
//  RealT z_max2 = 2.;
//
//  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
//                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
//                                    0. );
//
//  // call tribol setup and update
//  tribol::TestControlParameters parameters;  // struct does not hold info right now
//  parameters.penalty_ratio = false;
//  parameters.const_penalty = 1.0;
//
//  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
//      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );
//
//  EXPECT_EQ( test_mesh_update_err, 0 );
//
//  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();
//
//  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
//
//  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );
//
//  tribol::finalize();
//}
//
//TEST_F( CompGeomTest, single_mortar_check )
//{
//  int nMortarElems = 4;
//  int nElemsXM = nMortarElems;
//  int nElemsYM = nMortarElems;
//  int nElemsZM = nMortarElems;
//
//  int nNonmortarElems = 5;
//  int nElemsXS = nNonmortarElems;
//  int nElemsYS = nNonmortarElems;
//  int nElemsZS = nNonmortarElems;
//
//  int userSpecifiedNumOverlaps = 64;
//
//  // mesh bounding box with 0.1 interpenetration gap
//  RealT x_min1 = 0.;
//  RealT y_min1 = 0.;
//  RealT z_min1 = 0.;
//  RealT x_max1 = 1.;
//  RealT y_max1 = 1.;
//  RealT z_max1 = 1.05;
//
//  RealT x_min2 = 0.;
//  RealT y_min2 = 0.;
//  RealT z_min2 = 0.95;
//  RealT x_max2 = 1.;
//  RealT y_max2 = 1.;
//  RealT z_max2 = 2.;
//
//  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
//                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
//                                    0. );
//
//  // call tribol setup and update
//  tribol::TestControlParameters parameters;  // struct does not hold info right now
//
//  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
//      tribol::SINGLE_MORTAR, tribol::LAGRANGE_MULTIPLIER, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );
//
//  EXPECT_EQ( test_mesh_update_err, 0 );
//
//  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();
//
//  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
//
//  EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );
//
//  tribol::finalize();
//}
//
//TEST_F( CompGeomTest, poly_area_centroid_1 )
//{
//  // This test checks the area centroid calculation
//  // vs. the vertex average centroid calculation for a
//  // rectangular quadrilateral. The expectation is that
//  // the results are the same
//  constexpr int dim = 3;
//  constexpr int numVerts = 4;
//  RealT x[dim * numVerts];
//
//  for ( int i = 0; i < dim * numVerts; ++i ) {
//    x[i] = 0.;
//  }
//
//  // setup some quadrilateral coordinates
//  x[0] = -0.5;
//  x[dim * 1] = 0.5;
//  x[dim * 2] = 0.5;
//  x[dim * 3] = -0.5;
//
//  x[1] = -0.5;
//  x[dim * 1 + 1] = -0.5;
//  x[dim * 2 + 1] = 0.5;
//  x[dim * 3 + 1] = 0.5;
//
//  x[2] = 0.1;
//  x[dim * 1 + 2] = 0.1;
//  x[dim * 2 + 2] = 0.1;
//  x[dim * 3 + 2] = 0.1;
//
//  RealT cX_avg, cY_avg, cZ_avg;
//  RealT cX_area, cY_area, cZ_area;
//
//  tribol::VertexAvgCentroid( x, dim, numVerts, cX_avg, cY_avg, cZ_avg );
//  tribol::PolyAreaCentroid( x, dim, numVerts, cX_area, cY_area, cZ_area );
//
//  RealT diff[3]{ 0., 0., 0. };
//
//  diff[0] = std::abs( cX_avg - cX_area );
//  diff[1] = std::abs( cY_avg - cY_area );
//  diff[2] = std::abs( cZ_avg - cZ_area );
//
//  RealT diff_mag = tribol::magnitude( diff[0], diff[1], diff[2] );
//
//  RealT tol = 1.e-5;
//  EXPECT_LE( diff_mag, tol );
//}
//
//TEST_F( CompGeomTest, poly_area_centroid_2 )
//{
//  // This test checks the area centroid calculation
//  // the centroid calculation for a non-self-intersecting,
//  // closed polygon
//  constexpr int dim = 3;
//  constexpr int numVerts = 4;
//  RealT x[numVerts];
//  RealT y[numVerts];
//  RealT z[numVerts];
//
//  for ( int i = 0; i < numVerts; ++i ) {
//    x[i] = 0.;
//    y[i] = 0.;
//    z[i] = 0.;
//  }
//
//  // setup some quadrilateral coordinates
//  x[0] = -0.515;
//  x[1] = 0.54;
//  x[2] = 0.65;
//  x[3] = -0.524;
//
//  y[0] = -0.5;
//  y[1] = -0.5;
//  y[2] = 0.5;
//  y[3] = 0.5;
//
//  z[0] = 0.1;
//  z[1] = 0.1;
//  z[2] = 0.1;
//  z[3] = 0.1;
//
//  // create stacked array of coordinates
//  RealT x_bar[dim * numVerts];
//  for ( int i = 0; i < numVerts; ++i ) {
//    x_bar[dim * i] = x[i];
//    x_bar[dim * i + 1] = y[i];
//    x_bar[dim * i + 2] = z[i];
//  }
//
//  RealT cX_area, cY_area, cZ_area;
//  RealT cX_poly, cY_poly, cZ_poly;
//
//  tribol::PolyAreaCentroid( x_bar, dim, numVerts, cX_area, cY_area, cZ_area );
//  tribol::PolyCentroid( x, y, numVerts, cX_poly, cY_poly );
//
//  cZ_poly = z[0];
//
//  RealT diff[3]{ 0., 0., 0. };
//
//  diff[0] = std::abs( cX_poly - cX_area );
//  diff[1] = std::abs( cY_poly - cY_area );
//  diff[2] = std::abs( cZ_poly - cZ_area );
//
//  RealT diff_mag = tribol::magnitude( diff[0], diff[1], diff[2] );
//
//  RealT tol = 1.e-5;
//  EXPECT_LE( diff_mag, tol );
//}
////
////TEST_F( CompGeomTest, codirectional_normals_3d )
////{
////  // this test ensures that faces in a given face-pair with nearly co-directional
////  // normals is not actually included as a contact candidate
////  constexpr int numVerts = 4;
////  constexpr int numCells = 2;
////  constexpr int lengthNodalData = numCells * numVerts;
////  RealT element_thickness[numCells];
////  RealT x[lengthNodalData];
////  RealT y[lengthNodalData];
////  RealT z[lengthNodalData];
////
////  for ( int i = 0; i < numCells; ++i ) {
////    element_thickness[i] = 1.0;
////  }
////
////  // coordinates for face 1
////  x[0] = 0.;
////  x[1] = 1.;
////  x[2] = 1.;
////  x[3] = 0.;
////
////  y[0] = 0.;
////  y[1] = 0.;
////  y[2] = 1.;
////  y[3] = 1.;
////
////  z[0] = 0.;
////  z[1] = 0.;
////  z[2] = 0.;
////  z[3] = 0.;
////
////  // coordinates for face 2
////  x[4] = 0.;
////  x[5] = 1.;
////  x[6] = 1.;
////  x[7] = 0.;
////
////  y[4] = 0.;
////  y[5] = 0.;
////  y[6] = 1.;
////  y[7] = 1.;
////
////  // amount of interpenetration in the z-direction
////  z[4] = -0.300001 * element_thickness[1];
////  z[5] = -0.300001 * element_thickness[1];
////  z[6] = -0.300001 * element_thickness[1];
////  z[7] = -0.300001 * element_thickness[1];
////
////  // register contact mesh
////  tribol::IndexT mesh_id = 0;
////  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };  // hard coded for a two face problem
////  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
////                        tribol::MemorySpace::Host );
////
////  RealT* fx;
////  RealT* fy;
////  RealT* fz;
////  tribol::allocRealArray( &fx, lengthNodalData, 0. );
////  tribol::allocRealArray( &fy, lengthNodalData, 0. );
////  tribol::allocRealArray( &fz, lengthNodalData, 0. );
////
////  tribol::registerNodalResponse( mesh_id, fx, fy, fz );
////
////  RealT* vx;
////  RealT* vy;
////  RealT* vz;
////  RealT vel0 = -1.e-15;
////  tribol::allocRealArray( &vx, lengthNodalData, 0. );
////  tribol::allocRealArray( &vy, lengthNodalData, 0. );
////  tribol::allocRealArray( &vz, lengthNodalData, vel0 );
////
////  // set second face to impacting velocity
////  RealT vel2 = 1.e-15;
////  vz[4] = vel2;
////  vz[5] = vel2;
////  vz[6] = vel2;
////  vz[7] = vel2;
////
////  tribol::registerNodalVelocities( mesh_id, vx, vy, vz );
////
////  RealT bulk_mod[2] = { 1.0, 1.0 };
////  tribol::registerRealElementField( mesh_id, tribol::BULK_MODULUS, bulk_mod );
////  tribol::registerRealElementField( mesh_id, tribol::ELEMENT_THICKNESS, element_thickness );
////
////  int csIndex = 0;
////  tribol::registerCouplingScheme( csIndex, mesh_id, mesh_id, tribol::SURFACE_TO_SURFACE, tribol::AUTO,
////                                  tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
////                                  tribol::BINNING_CARTESIAN_PRODUCT, tribol::ExecutionMode::Sequential );
////
////  tribol::enableTimestepVote( csIndex, true );
////
////  tribol::setLoggingLevel( csIndex, tribol::TRIBOL_DEBUG );
////
////  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_ELEMENT, tribol::NO_RATE_PENALTY );
////
////  RealT dt = 1.0;
////  int err = tribol::update( 1, 1., dt );
////
////  EXPECT_EQ( err, 0 );
////  EXPECT_EQ( dt, 1.0 );
////
////  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();
////
////  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
////
////  EXPECT_EQ( couplingScheme->getNumActivePairs(), 0 );
////
////  delete[] fx;
////  delete[] fy;
////  delete[] fz;
////  delete[] vx;
////  delete[] vy;
////  delete[] vz;
////}
////
//TEST_F( CompGeomTest, auto_contact_lt_max_interpen )
//{
//  // This test uses auto-contact and checks that the face-pair
//  // is included as a conatct candidate, and is in fact in contact
//  // when the interpenetration is less than the maximum allowable
//  // for auto contact
//  constexpr int numVerts = 4;
//  constexpr int numCells = 2;
//  constexpr int lengthNodalData = numCells * numVerts;
//  RealT element_thickness[numCells];
//  RealT x[lengthNodalData];
//  RealT y[lengthNodalData];
//  RealT z[lengthNodalData];
//
//  for ( int i = 0; i < numCells; ++i ) {
//    element_thickness[i] = 1.0;
//  }
//
//  // coordinates for face 1
//  x[0] = 0.;
//  x[1] = 1.;
//  x[2] = 1.;
//  x[3] = 0.;
//
//  y[0] = 0.;
//  y[1] = 0.;
//  y[2] = 1.;
//  y[3] = 1.;
//
//  z[0] = 0.;
//  z[1] = 0.;
//  z[2] = 0.;
//  z[3] = 0.;
//
//  // coordinates for face 2
//  x[4] = 0.;
//  x[5] = 1.;
//  x[6] = 1.;
//  x[7] = 0.;
//
//  y[4] = 0.;
//  y[5] = 0.;
//  y[6] = 1.;
//  y[7] = 1.;
//
//  // amount of interpenetration in the z-direction
//  RealT max_interpen_frac = 1.0;
//  RealT test_ratio = 0.90;  // fraction of max interpen frac used for this test
//  z[4] = -test_ratio * max_interpen_frac * element_thickness[1];
//  z[5] = -test_ratio * max_interpen_frac * element_thickness[1];
//  z[6] = -test_ratio * max_interpen_frac * element_thickness[1];
//  z[7] = -test_ratio * max_interpen_frac * element_thickness[1];
//
//  // register contact mesh
//  tribol::IndexT mesh_id = 0;
//  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 7, 6, 5 };  // hard coded for a two face problem
//  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
//                        tribol::MemorySpace::Host );
//
//  RealT* fx;
//  RealT* fy;
//  RealT* fz;
//  tribol::allocRealArray( &fx, lengthNodalData, 0. );
//  tribol::allocRealArray( &fy, lengthNodalData, 0. );
//  tribol::allocRealArray( &fz, lengthNodalData, 0. );
//
//  tribol::registerNodalResponse( mesh_id, fx, fy, fz );
//
//  RealT* vx;
//  RealT* vy;
//  RealT* vz;
//  RealT vel0 = -1.;
//  tribol::allocRealArray( &vx, lengthNodalData, 0. );
//  tribol::allocRealArray( &vy, lengthNodalData, 0. );
//  tribol::allocRealArray( &vz, lengthNodalData, vel0 );
//
//  // set second face to impacting velocity
//  RealT vel2 = 1.0;
//  vz[4] = vel2;
//  vz[5] = vel2;
//  vz[6] = vel2;
//  vz[7] = vel2;
//
//  tribol::registerNodalVelocities( mesh_id, vx, vy, vz );
//
//  // register element thickness for use with auto contact
//  tribol::registerRealElementField( mesh_id, tribol::ELEMENT_THICKNESS, &element_thickness[0] );
//
//  int csIndex = 0;
//  tribol::registerCouplingScheme( csIndex, mesh_id, mesh_id, tribol::SURFACE_TO_SURFACE, tribol::AUTO,
//                                  tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
//                                  tribol::BINNING_CARTESIAN_PRODUCT, tribol::ExecutionMode::Sequential );
//
//  tribol::setAutoContactPenScale( csIndex, max_interpen_frac );
//
//  tribol::enableTimestepVote( csIndex, true );
//
//  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT, tribol::NO_RATE_PENALTY );
//
//  tribol::setKinematicConstantPenalty( mesh_id, 1.0 );
//
//  RealT dt = 1.0;
//  int err = tribol::update( 1, 1., dt );
//
//  EXPECT_EQ( err, 0 );
//
//  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();
//
//  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
//
//  EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );
//
//  delete[] fx;
//  delete[] fy;
//  delete[] fz;
//  delete[] vx;
//  delete[] vy;
//  delete[] vz;
//}
//
//TEST_F( CompGeomTest, auto_contact_gt_max_interpen )
//{
//  // This test uses auto-contact and checks that the face-pair
//  // is included as a contact candidate, and is in fact in contact
//  // when the interpenetration is less than the maximum allowable
//  // for auto contact
//  constexpr int numVerts = 4;
//  constexpr int numCells = 2;
//  constexpr int lengthNodalData = numCells * numVerts;
//  RealT element_thickness[numCells];
//  RealT x[lengthNodalData];
//  RealT y[lengthNodalData];
//  RealT z[lengthNodalData];
//
//  for ( int i = 0; i < numCells; ++i ) {
//    element_thickness[i] = 1.0;
//  }
//
//  // coordinates for face 1
//  x[0] = 0.;
//  x[1] = 1.;
//  x[2] = 1.;
//  x[3] = 0.;
//
//  y[0] = 0.;
//  y[1] = 0.;
//  y[2] = 1.;
//  y[3] = 1.;
//
//  z[0] = 0.;
//  z[1] = 0.;
//  z[2] = 0.;
//  z[3] = 0.;
//
//  // coordinates for face 2
//  x[4] = 0.;
//  x[5] = 1.;
//  x[6] = 1.;
//  x[7] = 0.;
//
//  y[4] = 0.;
//  y[5] = 0.;
//  y[6] = 1.;
//  y[7] = 1.;
//
//  // amount of interpenetration in the z-direction
//  RealT max_interpen_frac = 1.0;
//  RealT test_ratio = 1.01;  // fraction of max interpen frac used for this test
//  z[4] = -test_ratio * max_interpen_frac * element_thickness[1];
//  z[5] = -test_ratio * max_interpen_frac * element_thickness[1];
//  z[6] = -test_ratio * max_interpen_frac * element_thickness[1];
//  z[7] = -test_ratio * max_interpen_frac * element_thickness[1];
//
//  // register contact mesh
//  tribol::IndexT mesh_id = 0;
//  tribol::IndexT conn[8] = { 0, 1, 2, 3, 4, 7, 6, 5 };  // hard coded for a two face problem
//  tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)( tribol::LINEAR_QUAD ), &x[0], &y[0], &z[0],
//                        tribol::MemorySpace::Host );
//
//  RealT* fx;
//  RealT* fy;
//  RealT* fz;
//  tribol::allocRealArray( &fx, lengthNodalData, 0. );
//  tribol::allocRealArray( &fy, lengthNodalData, 0. );
//  tribol::allocRealArray( &fz, lengthNodalData, 0. );
//
//  tribol::registerNodalResponse( mesh_id, fx, fy, fz );
//
//  RealT* vx;
//  RealT* vy;
//  RealT* vz;
//  RealT vel0 = -1.;
//  tribol::allocRealArray( &vx, lengthNodalData, 0. );
//  tribol::allocRealArray( &vy, lengthNodalData, 0. );
//  tribol::allocRealArray( &vz, lengthNodalData, vel0 );
//
//  // set second face to impacting velocity
//  RealT vel2 = 1.0;
//  vz[4] = vel2;
//  vz[5] = vel2;
//  vz[6] = vel2;
//  vz[7] = vel2;
//
//  tribol::registerNodalVelocities( mesh_id, vx, vy, vz );
//
//  // register element thickness for use with auto contact
//  tribol::registerRealElementField( mesh_id, tribol::ELEMENT_THICKNESS, &element_thickness[0] );
//
//  int csIndex = 0;
//  tribol::registerCouplingScheme( csIndex, mesh_id, mesh_id, tribol::SURFACE_TO_SURFACE, tribol::AUTO,
//                                  tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
//                                  tribol::BINNING_CARTESIAN_PRODUCT, tribol::ExecutionMode::Sequential );
//
//  tribol::setAutoContactPenScale( csIndex, max_interpen_frac );
//
//  tribol::enableTimestepVote( csIndex, true );
//
//  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT, tribol::NO_RATE_PENALTY );
//
//  tribol::setKinematicConstantPenalty( mesh_id, 1.0 );
//
//  RealT dt = 1.0;
//  int err = tribol::update( 1, 1., dt );
//
//  EXPECT_EQ( err, 0 );
//
//  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();
//
//  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
//
//  EXPECT_EQ( couplingScheme->getNumActivePairs(), 0 );
//
//  delete[] fx;
//  delete[] fy;
//  delete[] fz;
//  delete[] vx;
//  delete[] vy;
//  delete[] vz;
//}

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
