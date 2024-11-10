// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
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

// Axom includes
#include "axom/slic.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// gtest includes
#include "gtest/gtest.h"

// c++ includes
#include <cmath> // std::abs, std::cos, std::sin
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
class CompGeomTest : public ::testing::Test
{
   
public:

   tribol::TestMesh m_mesh;

protected:

   void SetUp() override
   {
   }

   void TearDown() override
   {
      // call clear() on mesh object to be safe
      this->m_mesh.clear();
   }

protected:

};

TEST_F( CompGeomTest, common_plane_check )
{
   int nMortarElems = 3; 
   int nElemsXM = nMortarElems;
   int nElemsYM = 3;
   int nElemsZM = nMortarElems;

   int nNonmortarElems = 3; 
   int nElemsXS = nNonmortarElems;
   int nElemsYS = 3;
   int nElemsZS = nNonmortarElems;

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
   RealT x_max2 = 1.1;
   RealT y_max2 = 0.9999;
   RealT z_max2 = 2.;

   this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM,
                                     x_min1, y_min1, z_min1,
                                     x_max1, y_max1, z_max1,
                                     nElemsXS, nElemsYS, nElemsZS,
                                     x_min2, y_min2, z_min2,
                                     x_max2, y_max2, z_max2,
                                     0., 0. );

   // call tribol setup and update
   tribol::TestControlParameters parameters; // struct does not hold info right now
   parameters.penalty_ratio = false;
   parameters.const_penalty = 1.0;

   int test_mesh_update_err = 
      this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, 
                                         tribol::FRICTIONLESS, tribol::NO_CASE, true, parameters );
 
   EXPECT_EQ( test_mesh_update_err, 0 );

   tribol::CouplingSchemeManager& couplingSchemeManager = 
         tribol::CouplingSchemeManager::getInstance();
  
   tribol::CouplingScheme* couplingScheme = 
      &couplingSchemeManager.at( 0 );

   EXPECT_EQ( userSpecifiedNumOverlaps, couplingScheme->getNumActivePairs() );

   tribol::finalize();
}

TEST_F( CompGeomTest, single_mortar_check )
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

   this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM,
                                     x_min1, y_min1, z_min1,
                                     x_max1, y_max1, z_max1,
                                     nElemsXS, nElemsYS, nElemsZS,
                                     x_min2, y_min2, z_min2,
                                     x_max2, y_max2, z_max2,
                                     0., 0. );

   // call tribol setup and update
   tribol::TestControlParameters parameters; // struct does not hold info right now

   int test_mesh_update_err = 
      this->m_mesh.tribolSetupAndUpdate( tribol::SINGLE_MORTAR, tribol::LAGRANGE_MULTIPLIER, 
                                         tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

   EXPECT_EQ( test_mesh_update_err, 0 );

   tribol::CouplingSchemeManager& couplingSchemeManager = 
         tribol::CouplingSchemeManager::getInstance();
  
   tribol::CouplingScheme* couplingScheme = 
      &couplingSchemeManager.at( 0 );

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
   RealT x[ dim * numVerts ];

   for (int i=0; i<dim*numVerts; ++i)
   {
      x[i] = 0.;
   }

   // setup some quadrilateral coordinates 
   x[0] = -0.5;
   x[dim*1] =  0.5;
   x[dim*2] =  0.5;
   x[dim*3] = -0.5;

   x[1] = -0.5;
   x[dim*1+1] = -0.5;
   x[dim*2+1] = 0.5;
   x[dim*3+1] = 0.5;

   x[2] = 0.1;
   x[dim*1+2] = 0.1;
   x[dim*2+2] = 0.1; 
   x[dim*3+2] = 0.1;

   RealT cX_avg, cY_avg, cZ_avg;
   RealT cX_area, cY_area, cZ_area;

   tribol::VertexAvgCentroid( x, dim, numVerts, cX_avg, cY_avg, cZ_avg );
   tribol::PolyAreaCentroid( x, dim, numVerts, cX_area, cY_area, cZ_area );

   RealT diff[3] { 0., 0., 0. };

   diff[0] = std::abs(cX_avg - cX_area);
   diff[1] = std::abs(cY_avg - cY_area);
   diff[2] = std::abs(cZ_avg - cZ_area);

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
   RealT x[ numVerts ];
   RealT y[ numVerts ];
   RealT z[ numVerts ];

   for (int i=0; i<numVerts; ++i)
   {
      x[i] = 0.;
      y[i] = 0.;
      z[i] = 0.;
   }

   // setup some quadrilateral coordinates 
   x[0] = -0.515;
   x[1] =  0.54;
   x[2] =  0.65;
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
   RealT x_bar[ dim*numVerts];
   for (int i=0; i<numVerts; ++i)
   {
      x_bar[dim*i]   = x[i];
      x_bar[dim*i+1] = y[i];
      x_bar[dim*i+2] = z[i];
   }

   RealT cX_area, cY_area, cZ_area;
   RealT cX_poly, cY_poly, cZ_poly;

   tribol::PolyAreaCentroid( x_bar, dim, numVerts, cX_area, cY_area, cZ_area );
   tribol::PolyCentroid( x, y, numVerts, cX_poly, cY_poly );

   cZ_poly = z[0];

   RealT diff[3] { 0., 0., 0. };

   diff[0] = std::abs(cX_poly - cX_area);
   diff[1] = std::abs(cY_poly - cY_area);
   diff[2] = std::abs(cZ_poly - cZ_area);

   RealT diff_mag = tribol::magnitude( diff[0], diff[1], diff[2] );

   RealT tol = 1.e-5;
   EXPECT_LE( diff_mag, tol );
}

TEST_F( CompGeomTest, 2d_projections_1 )
{
   constexpr int dim = 2;
   constexpr int numVerts = 2;
   RealT xy1[dim*numVerts];
   RealT xy2[dim*numVerts];

// Notice how the face vertices are flipped between register mesh and the segment basis eval!

//registerMesh() face 1 x: 1, 0.75, 0.744548
//registerMesh() face 1 y: 1, 0, 0.0902704
//registerMesh() face 1 x: 2, 0.744548, 0.75
//registerMesh() face 1 y: 2, 0.0902704, 0
//(x0,y0) and (x1,y1): (0.744548, 0.0902704), (0.75, 0).
//(px,py): (0.735935, 0.136655)
//SegmentBasis: phi is 1.51907 not between 0. and 1.
//(x0,y0) and (x1,y1): (0.75, 0), (0.744548, 0.0902704).
//(px,py): (0.735935, 0.136655)
//SegmentBasis: phi is 1.51907 not between 0. and 1.

   // TODO update to use these face coordinates from testing
//   xy1[0] = 0.75;
//   xy1[1] = 0. ;
//   xy1[2] = 0.744548; 
//   xy1[3] = 0.0902704;
//
//   xy2[0] = 0.744548;
//   xy2[1] = 0.0902704;
//   xy2[2] = 0.75;
//   xy2[3] = 0.;

   // this geometry should be in contact
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

   RealT lambdaX1 = xy1[2]-xy1[0];
   RealT lambdaY1 = xy1[3]-xy1[1];

   faceNormal1[0] = lambdaY1;
   faceNormal1[1] = -lambdaX1;

   RealT lambdaX2 = xy2[2]-xy2[0];
   RealT lambdaY2 = xy2[3]-xy2[1];

   faceNormal2[0] = lambdaY2;
   faceNormal2[1] = -lambdaX2;

   RealT cxf1[3] = {0., 0., 0.};
   RealT cxf2[3] = {0., 0., 0.};

   tribol::VertexAvgCentroid( xy1, dim, numVerts, cxf1[0], cxf1[1], cxf1[2] );
   tribol::VertexAvgCentroid( xy2, dim, numVerts, cxf2[0], cxf2[1], cxf2[2] );

   // average the vertex averaged centroids of each face to get a pretty good 
   // estimate of the common plane centroid
   RealT cx[dim];
   cx[0] = 0.5*(cxf1[0] + cxf2[0]);
   cx[1] = 0.5*(cxf1[1] + cxf2[1]);

   RealT cxProj1[3] = {0., 0., 0.}; 
   RealT cxProj2[3] = {0., 0., 0.}; 

   tribol::ProjectPointToSegment( cx[0], cx[1], faceNormal1[0], faceNormal1[1], 
                                  cxf1[0], cxf1[1], cxProj1[0], cxProj1[1] );
   tribol::ProjectPointToSegment( cx[0], cx[1], faceNormal2[0], faceNormal2[1], 
                                  cxf2[0], cxf2[1], cxProj2[0], cxProj2[1] );

   RealT diffx1 = std::abs(cxProj1[0] - 0.738595);
   RealT diffy1 = std::abs(cxProj1[1] - 0.0915028);
   RealT diffx2 = std::abs(cxProj2[0] - 0.738591);
   RealT diffy2 = std::abs(cxProj2[1] - 0.0915022);
   EXPECT_LE(diffx1, 1.e-6);  
   EXPECT_LE(diffy1, 1.e-6); 
   EXPECT_LE(diffx2, 1.e-6); 
   EXPECT_LE(diffy2, 1.e-6); 

   RealT x1[numVerts];
   RealT y1[numVerts];
   RealT x2[numVerts];
   RealT y2[numVerts];

   for (int i=0; i<numVerts; ++i)
   {
      x1[i] = xy1[i*dim];
      y1[i] = xy1[i*dim+1];
      x2[i] = xy2[i*dim];
      y2[i] = xy2[i*dim+1];
   }

   tribol::IndexT conn1[2] = {0,1};
   tribol::IndexT conn2[2] = {0,1};

   tribol::registerMesh( 0, 1, 2, &conn1[0], (int)(tribol::LINEAR_EDGE), &x1[0], &y1[0], nullptr, tribol::MemorySpace::Host );
   tribol::registerMesh( 1, 1, 2, &conn2[0], (int)(tribol::LINEAR_EDGE), &x2[0], &y2[0], nullptr, tribol::MemorySpace::Host );

   RealT fx1[2] = {0., 0.};
   RealT fy1[2] = {0., 0.};
   RealT fx2[2] = {0., 0.};
   RealT fy2[2] = {0., 0.};

   tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], nullptr );
   tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], nullptr );

   tribol::setKinematicConstantPenalty( 0, 1. );
   tribol::setKinematicConstantPenalty( 1, 1. );

   tribol::registerCouplingScheme( 0, 0, 1,
                                   tribol::SURFACE_TO_SURFACE,
                                   tribol::NO_CASE,
                                   tribol::COMMON_PLANE,
                                   tribol::FRICTIONLESS,
                                   tribol::PENALTY,
                                   tribol::BINNING_GRID,
                                   tribol::ExecutionMode::Sequential );

   tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
   tribol::setContactAreaFrac( 0, 1.e-4 );

   // TODO check penetration and overlap tolerance with what is being used in host-code

   RealT dt = 1.;
   int update_err = tribol::update( 1, 1., dt );

   EXPECT_EQ( update_err, 0 );

}

TEST_F( CompGeomTest, 2d_projections_2 )
{
   constexpr int dim = 2;
   constexpr int numVerts = 2;
   RealT xy1[dim*numVerts];
   RealT xy2[dim*numVerts];

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

   RealT lambdaX1 = xy1[2]-xy1[0];
   RealT lambdaY1 = xy1[3]-xy1[1];

   faceNormal1[0] = lambdaY1;
   faceNormal1[1] = -lambdaX1;

   RealT lambdaX2 = xy2[2]-xy2[0];
   RealT lambdaY2 = xy2[3]-xy2[1];

   faceNormal2[0] = lambdaY2;
   faceNormal2[1] = -lambdaX2;

   RealT cxf1[3] = {0., 0., 0.};
   RealT cxf2[3] = {0., 0., 0.};

   tribol::VertexAvgCentroid( xy1, dim, numVerts, cxf1[0], cxf1[1], cxf1[2] );
   tribol::VertexAvgCentroid( xy2, dim, numVerts, cxf2[0], cxf2[1], cxf2[2] );

   // average the vertex averaged centroids of each face to get a pretty good 
   // estimate of the common plane centroid
   RealT cx[dim];
   cx[0] = 0.5*(cxf1[0] + cxf2[0]);
   cx[1] = 0.5*(cxf1[1] + cxf2[1]);

   RealT cxProj1[3] = {0., 0., 0.}; 
   RealT cxProj2[3] = {0., 0., 0.}; 

   tribol::ProjectPointToSegment( cx[0], cx[1], faceNormal1[0], faceNormal1[1], 
                                  cxf1[0], cxf1[1], cxProj1[0], cxProj1[1] );
   tribol::ProjectPointToSegment( cx[0], cx[1], faceNormal2[0], faceNormal2[1], 
                                  cxf2[0], cxf2[1], cxProj2[0], cxProj2[1] );

   RealT diffx1 = std::abs(cxProj1[0] - cx[0]); 
   RealT diffy1 = std::abs(cxProj1[1] - cx[1]);
   RealT diffx2 = std::abs(cxProj2[0] - cx[0]); 
   RealT diffy2 = std::abs(cxProj2[1] - cx[1]); 
   EXPECT_LE(diffx1, 1.e-6);  
   EXPECT_LE(diffy1, 1.e-6); 
   EXPECT_LE(diffx2, 1.e-6); 
   EXPECT_LE(diffy2, 1.e-6); 
}

TEST_F( CompGeomTest, codirectional_normals_3d )
{
   // this test ensures that faces in a given face-pair with nearly co-directional 
   // normals is not actually included as a contact candidate
   constexpr int numVerts = 4;
   constexpr int numCells = 2;
   constexpr int lengthNodalData = numCells * numVerts;
   RealT element_thickness[numCells];
   RealT x[lengthNodalData];
   RealT y[lengthNodalData];
   RealT z[lengthNodalData];

   for (int i=0; i<numCells; ++i)
   {
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
   z[4] = -0.300001*element_thickness[1];
   z[5] = -0.300001*element_thickness[1];
   z[6] = -0.300001*element_thickness[1];
   z[7] = -0.300001*element_thickness[1];

   // register contact mesh
   tribol::IndexT mesh_id = 0;
   tribol::IndexT conn[8] = {0,1,2,3,4,5,6,7}; // hard coded for a two face problem
   tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)(tribol::LINEAR_QUAD), &x[0], &y[0], &z[0], tribol::MemorySpace::Host );

   RealT *fx;
   RealT *fy;
   RealT *fz; 
   tribol::allocRealArray( &fx, lengthNodalData, 0. );
   tribol::allocRealArray( &fy, lengthNodalData, 0. );
   tribol::allocRealArray( &fz, lengthNodalData, 0. );

   tribol::registerNodalResponse( mesh_id, fx, fy, fz );

   RealT *vx;
   RealT *vy;
   RealT *vz; 
   RealT vel0 = -1.e-15;
   tribol::allocRealArray( &vx, lengthNodalData, 0. );
   tribol::allocRealArray( &vy, lengthNodalData, 0. );
   tribol::allocRealArray( &vz, lengthNodalData, vel0 );

   // set second face to impacting velocity
   RealT vel2 = 1.e-15;
   vz[4] = vel2;
   vz[5] = vel2;
   vz[6] = vel2;
   vz[7] = vel2;

   tribol::registerNodalVelocities( mesh_id, vx, vy, vz );

   RealT bulk_mod[2] = {1.0, 1.0};
   tribol::registerRealElementField( mesh_id, tribol::BULK_MODULUS, bulk_mod );
   tribol::registerRealElementField( mesh_id, tribol::ELEMENT_THICKNESS, element_thickness );

   int csIndex = 0;
   tribol::registerCouplingScheme( csIndex, mesh_id, mesh_id,
                                   tribol::SURFACE_TO_SURFACE,
                                   tribol::AUTO,
                                   tribol::COMMON_PLANE,
                                   tribol::FRICTIONLESS,
                                   tribol::PENALTY,
                                   tribol::BINNING_CARTESIAN_PRODUCT,
                                   tribol::ExecutionMode::Sequential );

   tribol::enableTimestepVote( csIndex, true );

   tribol::setLoggingLevel( csIndex, tribol::TRIBOL_DEBUG );

   tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_ELEMENT, tribol::NO_RATE_PENALTY );

   RealT dt = 1.0;
   int err = tribol::update( 1, 1., dt );
 
   EXPECT_EQ( err, 0 );
   EXPECT_EQ( dt, 1.0 );

   tribol::CouplingSchemeManager& couplingSchemeManager = 
         tribol::CouplingSchemeManager::getInstance();
  
   tribol::CouplingScheme* couplingScheme = 
      &couplingSchemeManager.at( 0 );

   EXPECT_EQ( couplingScheme->getNumActivePairs(), 0 );

   delete [] fx;
   delete [] fy;
   delete [] fz;
   delete [] vx;
   delete [] vy;
   delete [] vz;
}

TEST_F( CompGeomTest, auto_contact_lt_max_interpen )
{
   // This test uses auto-contact and checks that the face-pair
   // is included as a conatct candidate, and is in fact in contact 
   // when the interpenetration is less than the maximum allowable
   // for auto contact
   constexpr int numVerts = 4;
   constexpr int numCells = 2;
   constexpr int lengthNodalData = numCells * numVerts;
   RealT element_thickness[numCells];
   RealT x[lengthNodalData];
   RealT y[lengthNodalData];
   RealT z[lengthNodalData];

   for (int i=0; i<numCells; ++i)
   {
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
   RealT test_ratio = 0.90; // fraction of max interpen frac used for this test
   z[4] = -test_ratio * max_interpen_frac * element_thickness[1]; 
   z[5] = -test_ratio * max_interpen_frac * element_thickness[1]; 
   z[6] = -test_ratio * max_interpen_frac * element_thickness[1]; 
   z[7] = -test_ratio * max_interpen_frac * element_thickness[1]; 

   // register contact mesh
   tribol::IndexT mesh_id = 0;
   tribol::IndexT conn[8] = {0,1,2,3,4,7,6,5}; // hard coded for a two face problem
   tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)(tribol::LINEAR_QUAD), &x[0], &y[0], &z[0], tribol::MemorySpace::Host );

   RealT *fx;
   RealT *fy;
   RealT *fz; 
   tribol::allocRealArray( &fx, lengthNodalData, 0. );
   tribol::allocRealArray( &fy, lengthNodalData, 0. );
   tribol::allocRealArray( &fz, lengthNodalData, 0. );

   tribol::registerNodalResponse( mesh_id, fx, fy, fz );

   RealT *vx;
   RealT *vy;
   RealT *vz; 
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
   tribol::registerCouplingScheme( csIndex, mesh_id, mesh_id,
                                   tribol::SURFACE_TO_SURFACE,
                                   tribol::AUTO,
                                   tribol::COMMON_PLANE,
                                   tribol::FRICTIONLESS,
                                   tribol::PENALTY,
                                   tribol::BINNING_CARTESIAN_PRODUCT,
                                   tribol::ExecutionMode::Sequential );
                                   
   tribol::setAutoContactPenScale( csIndex, max_interpen_frac );

   tribol::enableTimestepVote( csIndex, true );

   tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT, tribol::NO_RATE_PENALTY );

   tribol::setKinematicConstantPenalty( mesh_id, 1.0 );

   RealT dt = 1.0;
   int err = tribol::update( 1, 1., dt );
 
   EXPECT_EQ( err, 0 );

   tribol::CouplingSchemeManager& couplingSchemeManager = 
         tribol::CouplingSchemeManager::getInstance();
  
   tribol::CouplingScheme* couplingScheme = 
      &couplingSchemeManager.at( 0 );

   EXPECT_EQ( couplingScheme->getNumActivePairs(), 1 );

   delete [] fx;
   delete [] fy;
   delete [] fz;
   delete [] vx;
   delete [] vy;
   delete [] vz;
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

   for (int i=0; i<numCells; ++i)
   {
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
   RealT test_ratio = 1.01; // fraction of max interpen frac used for this test
   z[4] = -test_ratio * max_interpen_frac * element_thickness[1]; 
   z[5] = -test_ratio * max_interpen_frac * element_thickness[1]; 
   z[6] = -test_ratio * max_interpen_frac * element_thickness[1]; 
   z[7] = -test_ratio * max_interpen_frac * element_thickness[1]; 

   // register contact mesh
   tribol::IndexT mesh_id = 0;
   tribol::IndexT conn[8] = {0,1,2,3,4,7,6,5}; // hard coded for a two face problem
   tribol::registerMesh( mesh_id, numCells, lengthNodalData, &conn[0], (int)(tribol::LINEAR_QUAD), &x[0], &y[0], &z[0], tribol::MemorySpace::Host );

   RealT *fx;
   RealT *fy;
   RealT *fz; 
   tribol::allocRealArray( &fx, lengthNodalData, 0. );
   tribol::allocRealArray( &fy, lengthNodalData, 0. );
   tribol::allocRealArray( &fz, lengthNodalData, 0. );

   tribol::registerNodalResponse( mesh_id, fx, fy, fz );

   RealT *vx;
   RealT *vy;
   RealT *vz; 
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
   tribol::registerCouplingScheme( csIndex, mesh_id, mesh_id,
                                   tribol::SURFACE_TO_SURFACE,
                                   tribol::AUTO,
                                   tribol::COMMON_PLANE,
                                   tribol::FRICTIONLESS,
                                   tribol::PENALTY,
                                   tribol::BINNING_CARTESIAN_PRODUCT,
                                   tribol::ExecutionMode::Sequential );

   tribol::setAutoContactPenScale( csIndex, max_interpen_frac );

   tribol::enableTimestepVote( csIndex, true );

   tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT, tribol::NO_RATE_PENALTY );

   tribol::setKinematicConstantPenalty( mesh_id, 1.0 );

   RealT dt = 1.0;
   int err = tribol::update( 1, 1., dt );
 
   EXPECT_EQ( err, 0 );

   tribol::CouplingSchemeManager& couplingSchemeManager = 
         tribol::CouplingSchemeManager::getInstance();
  
   tribol::CouplingScheme* couplingScheme = 
      &couplingSchemeManager.at( 0 );

   EXPECT_EQ( couplingScheme->getNumActivePairs(), 0 );

   delete [] fx;
   delete [] fy;
   delete [] fz;
   delete [] vx;
   delete [] vy;
   delete [] vz;
}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();         // initialize umpire's ResouceManager
#endif

  axom::slic::SimpleLogger logger;                // create & initialize logger,
  tribol::SimpleMPIWrapper wrapper(argc, argv);   // initialize and finalize MPI, when applicable

  result = RUN_ALL_TESTS();

  return result;
}
