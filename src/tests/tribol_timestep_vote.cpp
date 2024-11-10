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
#include "tribol/physics/CommonPlane.hpp"
#include "tribol/geom/GeomUtilities.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// Axom includes
#include "axom/slic.hpp"

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
 * the COMMON_PLANE + PENALTY implementation with registered 
 * velocities and timestep vote
 */
class CommonPlaneTest : public ::testing::Test
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

TEST_F( CommonPlaneTest, zero_velocity_small_gap )
{
   // This test uses a small gap with zero velocity to test both the 
   // timestep vote gap check and velocity check with no resulting 
   // change to dt
   this->m_mesh.mortarMeshId = 0;
   this->m_mesh.nonmortarMeshId = 1;

   int nMortarElems = 4;
   int nElemsXM = nMortarElems;
   int nElemsYM = nMortarElems;
   int nElemsZM = nMortarElems;

   int nNonmortarElems = 5; 
   int nElemsXS = nNonmortarElems;
   int nElemsYS = nNonmortarElems;
   int nElemsZS = nNonmortarElems;

   // mesh bounding box with 'small' (<30% element thickness) interpenetration gap
   RealT x_min1 = 0.;
   RealT y_min1 = 0.;
   RealT z_min1 = 0.; 
   RealT x_max1 = 1.;
   RealT y_max1 = 1.;
   RealT z_max1 = 1.005;

   RealT x_min2 = 0.;
   RealT y_min2 = 0.;
   RealT z_min2 = 0.95;
   RealT x_max2 = 1.;
   RealT y_max2 = 1.;
   RealT z_max2 = 2.;

   // compute element thickness for each block
   RealT element_thickness1 = (z_max1 - z_min1) / nElemsZM;
   RealT element_thickness2 = (z_max2 - z_min2) / nElemsZS;

   // setup mesh
   this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM,
                                     x_min1, y_min1, z_min1,
                                     x_max1, y_max1, z_max1,
                                     nElemsXS, nElemsYS, nElemsZS,
                                     x_min2, y_min2, z_min2,
                                     x_max2, y_max2, z_max2,
                                     0., 0. );

   // specify dt and component velocities for each block.
   // Large velocity in the z-direction will incite a change in the 
   // timestep. This velocity is computed on the high side using the 
   // hardcoded rule that one face cannot interpen the other 
   // exceeding 30% of the other's element thickness.
   RealT dt = 1.0;
   RealT bulk_mod1 = 1.0; // something simple
   RealT bulk_mod2 = 1.0; 
   RealT velX1 = 0.;
   RealT velY1 = 0.;
   RealT velZ1 = 0.; 
   RealT velX2 = 0.;
   RealT velY2 = 0.;
   RealT velZ2 = 0.; 

   this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
   this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId,  velX2, velY2, -velZ2 ); 
   
   // allocate and set element thickness and bulk modulus
   this->m_mesh.allocateAndSetElementThickness( m_mesh.mortarMeshId, element_thickness1 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.mortarMeshId, bulk_mod1 );
   this->m_mesh.allocateAndSetElementThickness( m_mesh.nonmortarMeshId, element_thickness2 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.nonmortarMeshId, bulk_mod2 );

   // call tribol setup and update
   tribol::TestControlParameters parameters; 
   parameters.penalty_ratio = true;
   parameters.const_penalty = 0.75;
   parameters.dt = dt;
   parameters.enable_timestep_vote = true;
   parameters.timestep_pen_frac = 0.3;

   int test_mesh_update_err = 
      this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, 
                                         tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

   EXPECT_EQ( test_mesh_update_err, 0 );
   EXPECT_EQ( parameters.dt, dt );

   tribol::finalize();
}

TEST_F( CommonPlaneTest, large_velocity_small_gap_dt_not_enabled )
{
   // This test has a small gap and large interpen velocity, but does not
   // call the API function to return a timestep vote. As such, we expect
   // the timestep to be unchanged
   this->m_mesh.mortarMeshId = 0;
   this->m_mesh.nonmortarMeshId = 1;

   int nMortarElems = 4;
   int nElemsXM = nMortarElems;
   int nElemsYM = nMortarElems;
   int nElemsZM = nMortarElems;

   int nNonmortarElems = 5; 
   int nElemsXS = nNonmortarElems;
   int nElemsYS = nNonmortarElems;
   int nElemsZS = nNonmortarElems;

   // mesh bounding box with 'small' (0.055) interpenetration gap
   RealT x_min1 = 0.;
   RealT y_min1 = 0.;
   RealT z_min1 = 0.; 
   RealT x_max1 = 1.;
   RealT y_max1 = 1.;
   RealT z_max1 = 1.005;

   RealT x_min2 = 0.;
   RealT y_min2 = 0.;
   RealT z_min2 = 0.95;
   RealT x_max2 = 1.;
   RealT y_max2 = 1.;
   RealT z_max2 = 2.;

   // compute element thickness for each block
   RealT element_thickness1 = (z_max1 - z_min1) / nElemsZM;
   RealT element_thickness2 = (z_max2 - z_min2) / nElemsZS;

   // setup mesh
   this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM,
                                     x_min1, y_min1, z_min1,
                                     x_max1, y_max1, z_max1,
                                     nElemsXS, nElemsYS, nElemsZS,
                                     x_min2, y_min2, z_min2,
                                     x_max2, y_max2, z_max2,
                                     0., 0. );

   // specify dt and component velocities for each block.
   // Large velocity in the z-direction will incite a change in the 
   // timestep. This velocity is computed on the high side using the 
   // hardcoded rule that one face cannot interpen the other 
   // exceeding 30% of the other's element thickness.
   RealT dt = 1.0;
   RealT bulk_mod1 = 1.0;
   RealT bulk_mod2 = 1.0;
   RealT vel_factor = 100.;
   RealT velX1 = 0.;
   RealT velY1 = 0.;
   RealT velZ1 = vel_factor * 0.3 * element_thickness1 / dt;
   RealT velX2 = 0.;
   RealT velY2 = 0.;
   RealT velZ2 = vel_factor * 0.3 * element_thickness2 / dt;

   this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
   this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId,  velX2, velY2, -velZ2 ); 
   
   // allocate and set element thickness and bulk modulus
   this->m_mesh.allocateAndSetElementThickness( m_mesh.mortarMeshId, element_thickness1 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.mortarMeshId, bulk_mod1 );
   this->m_mesh.allocateAndSetElementThickness( m_mesh.nonmortarMeshId, element_thickness2 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.nonmortarMeshId, bulk_mod2 );

   // call tribol setup and update
   tribol::TestControlParameters parameters; 
   parameters.penalty_ratio = true;
   parameters.const_penalty = 0.75;
   parameters.dt = dt;
   parameters.enable_timestep_vote = false;
   parameters.timestep_pen_frac = 0.3;

   int test_mesh_update_err = 
      this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, 
                                         tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

   EXPECT_EQ( test_mesh_update_err, 0 );
   EXPECT_EQ( parameters.dt, dt);


   tribol::finalize();
}

TEST_F( CommonPlaneTest, numerically_zero_velocity_small_gap )
{
   // this test is meant to test tolerancing in the timestep calculation 
   // when numerically zero velocities are present. This test caught a 
   // negative dt estimate bug that has since been fixed.
   //
   // Additionally, this test checks the case with 'zero' velocity and
   // a small gap resulting in no timestep change
   this->m_mesh.mortarMeshId = 0;
   this->m_mesh.nonmortarMeshId = 1;

   int nMortarElems = 4;
   int nElemsXM = nMortarElems;
   int nElemsYM = nMortarElems;
   int nElemsZM = nMortarElems;

   int nNonmortarElems = 5; 
   int nElemsXS = nNonmortarElems;
   int nElemsYS = nNonmortarElems;
   int nElemsZS = nNonmortarElems;

   // mesh bounding box with 'small' (0.055) interpenetration gap
   RealT x_min1 = 0.;
   RealT y_min1 = 0.;
   RealT z_min1 = 0.; 
   RealT x_max1 = 1.;
   RealT y_max1 = 1.;
   RealT z_max1 = 1.005;

   RealT x_min2 = 0.;
   RealT y_min2 = 0.;
   RealT z_min2 = 0.95;
   RealT x_max2 = 1.;
   RealT y_max2 = 1.;
   RealT z_max2 = 2.;

   // compute element thickness for each block
   RealT element_thickness1 = (z_max1 - z_min1) / nElemsZM;
   RealT element_thickness2 = (z_max2 - z_min2) / nElemsZS;

   // setup mesh
   this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM,
                                     x_min1, y_min1, z_min1,
                                     x_max1, y_max1, z_max1,
                                     nElemsXS, nElemsYS, nElemsZS,
                                     x_min2, y_min2, z_min2,
                                     x_max2, y_max2, z_max2,
                                     0., 0. );

   // specify dt and component velocities for each block.
   // Large velocity in the z-direction will incite a change in the 
   // timestep. This velocity is computed on the high side using the 
   // hardcoded rule that one face cannot interpen the other 
   // exceeding 30% of the other's element thickness.
   RealT dt = 1.0;
   RealT bulk_mod1 = 1.0; // something simple
   RealT bulk_mod2 = 1.0; 
   RealT velX1 = 1.e-12;
   RealT velY1 = 1.e-12;
   RealT velZ1 = 1.e-12; 
   RealT velX2 = -1.e-12;
   RealT velY2 = -1.e-12;
   RealT velZ2 = -1.e-12; 

   this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
   this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId,  velX2, velY2, -velZ2 ); 
   
   // allocate and set element thickness and bulk modulus
   this->m_mesh.allocateAndSetElementThickness( m_mesh.mortarMeshId, element_thickness1 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.mortarMeshId, bulk_mod1 );
   this->m_mesh.allocateAndSetElementThickness( m_mesh.nonmortarMeshId, element_thickness2 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.nonmortarMeshId, bulk_mod2 );

   // call tribol setup and update
   tribol::TestControlParameters parameters; 
   parameters.penalty_ratio = true;
   parameters.const_penalty = 0.75;
   parameters.dt = dt;
   parameters.enable_timestep_vote = true;
   parameters.timestep_pen_frac = 0.3;

   int test_mesh_update_err = 
      this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, 
                                         tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

   EXPECT_EQ( test_mesh_update_err, 0 );

   // expect no change in dt
   EXPECT_EQ( parameters.dt, dt );

   tribol::finalize();
}

TEST_F( CommonPlaneTest, zero_velocity_large_gap )
{
   // This is a tricky test where even in the presence of a large gap, there
   // is no change to the timestep because with zero velocity, no reduction in the 
   // timestep via a contact timestep vote could reduce the amount of interpen
   this->m_mesh.mortarMeshId = 0;
   this->m_mesh.nonmortarMeshId = 1;

   int nMortarElems = 4;
   int nElemsXM = nMortarElems;
   int nElemsYM = nMortarElems;
   int nElemsZM = nMortarElems;

   int nNonmortarElems = 4;
   int nElemsXS = nNonmortarElems;
   int nElemsYS = nNonmortarElems;
   int nElemsZS = nNonmortarElems;

   // mesh bounding box with >= 30% element thickness interpenetration gap
   RealT interpen_gap = 0.3 * 1.0 / nMortarElems;
   RealT x_min1 = 0.;
   RealT y_min1 = 0.;
   RealT z_min1 = 0.; 
   RealT x_max1 = 1.;
   RealT y_max1 = 1.;
   RealT z_max1 = 1.0 + 1.05*interpen_gap;

   RealT x_min2 = 0.;
   RealT y_min2 = 0.;
   RealT z_min2 = 1.0 - 1.05*interpen_gap;
   RealT x_max2 = 1.;
   RealT y_max2 = 1.;
   RealT z_max2 = 2.;

   // compute element thickness for each block
   RealT element_thickness1 = (z_max1 - z_min1) / nElemsZM;
   RealT element_thickness2 = (z_max2 - z_min2) / nElemsZS;

   // setup mesh
   this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM,
                                     x_min1, y_min1, z_min1,
                                     x_max1, y_max1, z_max1,
                                     nElemsXS, nElemsYS, nElemsZS,
                                     x_min2, y_min2, z_min2,
                                     x_max2, y_max2, z_max2,
                                     0., 0. );

   // specify dt and component velocities for each block.
   // Large velocity in the z-direction will incite a change in the 
   // timestep. This velocity is computed on the high side using the 
   // hardcoded rule that one face cannot interpen the other 
   // exceeding 30% of the other's element thickness.
   RealT dt = 1.0;
   RealT bulk_mod1 = 1.0;
   RealT bulk_mod2 = 1.0;
   RealT velX1 = 0.;
   RealT velY1 = 0.;
   RealT velZ1 = 0.; 
   RealT velX2 = 0.;
   RealT velY2 = 0.;
   RealT velZ2 = 0.; 

   this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
   this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId,  velX2, velY2, -velZ2 ); 
   
   // allocate and set element thickness and bulk modulus
   this->m_mesh.allocateAndSetElementThickness( m_mesh.mortarMeshId, element_thickness1 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.mortarMeshId, bulk_mod1 );
   this->m_mesh.allocateAndSetElementThickness( m_mesh.nonmortarMeshId, element_thickness2 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.nonmortarMeshId, bulk_mod2 );

   // call tribol setup and update
   tribol::TestControlParameters parameters; 
   parameters.penalty_ratio = true;
   parameters.const_penalty = 0.75;
   parameters.dt = dt;
   parameters.enable_timestep_vote = true;
   parameters.timestep_pen_frac = 0.3;

   int test_mesh_update_err = 
      this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, 
                                         tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

   EXPECT_EQ( test_mesh_update_err, 0 );
   // note that with very small velocity, the dt estimate will be 
   // very large, and won't change the timestep, even if the gap is large. This 
   // allows for a soft contact response in the presence of a small velocity that
   // won't actually correct too much interpen with a contact dt vote
   EXPECT_EQ( parameters.dt, dt );

   tribol::finalize();
}

TEST_F( CommonPlaneTest, large_velocity_small_gap )
{
   // This test does call the timestep enabling API function and uses a large velocity
   // and small gap to check that the timestep velocity check triggers a change in dt
   this->m_mesh.mortarMeshId = 0;
   this->m_mesh.nonmortarMeshId = 1;

   int nMortarElems = 4;
   int nElemsXM = nMortarElems;
   int nElemsYM = nMortarElems;
   int nElemsZM = nMortarElems;

   int nNonmortarElems = 4;
   int nElemsXS = nNonmortarElems;
   int nElemsYS = nNonmortarElems;
   int nElemsZS = nNonmortarElems;

   // mesh bounding box with 'small' (0.055) interpenetration gap
   RealT x_min1 = 0.;
   RealT y_min1 = 0.;
   RealT z_min1 = 0.; 
   RealT x_max1 = 1.;
   RealT y_max1 = 1.;
   RealT z_max1 = 1.005;

   RealT x_min2 = 0.;
   RealT y_min2 = 0.;
   RealT z_min2 = 0.95;
   RealT x_max2 = 1.;
   RealT y_max2 = 1.;
   RealT z_max2 = 2.;

   // compute element thickness for each block
   RealT element_thickness1 = (z_max1 - z_min1) / nElemsZM;
   RealT element_thickness2 = (z_max2 - z_min2) / nElemsZS;

   // setup mesh
   this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM,
                                     x_min1, y_min1, z_min1,
                                     x_max1, y_max1, z_max1,
                                     nElemsXS, nElemsYS, nElemsZS,
                                     x_min2, y_min2, z_min2,
                                     x_max2, y_max2, z_max2,
                                     0., 0. );

   // specify dt and component velocities for each block.
   // Large velocity in the z-direction will incite a change in the 
   // timestep. This velocity is computed on the high side using the 
   // hardcoded rule that one face cannot interpen the other 
   // exceeding 30% of the other's element thickness.
   RealT dt = 1.0;
   RealT bulk_mod1 = 1.0;
   RealT bulk_mod2 = 1.0;
   RealT vel_factor = 100.;
   RealT velX1 = 0.;
   RealT velY1 = 0.;
   RealT velZ1 = vel_factor * 0.3 * element_thickness1 / dt;
   RealT velX2 = 0.;
   RealT velY2 = 0.;
   RealT velZ2 = vel_factor * 0.3 * element_thickness2 / dt;

   this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
   this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId,  velX2, velY2, -velZ2 ); 
   
   // allocate and set element thickness and bulk modulus
   this->m_mesh.allocateAndSetElementThickness( m_mesh.mortarMeshId, element_thickness1 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.mortarMeshId, bulk_mod1 );
   this->m_mesh.allocateAndSetElementThickness( m_mesh.nonmortarMeshId, element_thickness2 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.nonmortarMeshId, bulk_mod2 );

   // call tribol setup and update
   tribol::TestControlParameters parameters; 
   parameters.penalty_ratio = true;
   parameters.const_penalty = 0.75;
   parameters.dt = dt;
   parameters.enable_timestep_vote = true;
   parameters.timestep_pen_frac = 0.3;

   int test_mesh_update_err = 
      this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, 
                                         tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

   EXPECT_EQ( test_mesh_update_err, 0 );

   RealT dt_vote = parameters.timestep_pen_frac*element_thickness1/velZ1;
   RealT dt_diff = std::abs(parameters.dt - dt_vote);
   RealT dt_tol = 1.e-8;
   EXPECT_LT( dt_diff, dt_tol );

   tribol::finalize();
}

TEST_F( CommonPlaneTest, large_velocity_large_gap )
{
   // This test uses a large velocity AND large gap to ensure that 
   // the minimum of both timestep checks modify the dt. If the gap
   // governs, then a finite velocity means that a gap dt can control
   this->m_mesh.mortarMeshId = 0;
   this->m_mesh.nonmortarMeshId = 1;

   int nMortarElems = 4;
   int nElemsXM = nMortarElems;
   int nElemsYM = nMortarElems;
   int nElemsZM = nMortarElems;

   int nNonmortarElems = 4;
   int nElemsXS = nNonmortarElems;
   int nElemsYS = nNonmortarElems;
   int nElemsZS = nNonmortarElems;

   // mesh bounding box with >= 30% element thickness interpenetration gap
   RealT interpen_gap = 0.3 * 1.0 / nMortarElems;
   RealT x_min1 = 0.;
   RealT y_min1 = 0.;
   RealT z_min1 = 0.; 
   RealT x_max1 = 1.;
   RealT y_max1 = 1.;
   RealT z_max1 = 1.0 + 1.05*interpen_gap;

   RealT x_min2 = 0.;
   RealT y_min2 = 0.;
   RealT z_min2 = 1.0 - 1.05*interpen_gap; 
   RealT x_max2 = 1.;
   RealT y_max2 = 1.;
   RealT z_max2 = 2.;

   // compute element thickness for each block
   RealT element_thickness1 = (z_max1 - z_min1) / nElemsZM;
   RealT element_thickness2 = (z_max2 - z_min2) / nElemsZS;

   // setup mesh
   this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM,
                                     x_min1, y_min1, z_min1,
                                     x_max1, y_max1, z_max1,
                                     nElemsXS, nElemsYS, nElemsZS,
                                     x_min2, y_min2, z_min2,
                                     x_max2, y_max2, z_max2,
                                     0., 0. );

   // specify dt and component velocities for each block.
   // Large velocity in the z-direction will incite a change in the 
   // timestep. This velocity is computed on the high side using the 
   // hardcoded rule that one face cannot interpen the other 
   // exceeding 30% of the other's element thickness.
   RealT dt = 1.0;
   RealT bulk_mod1 = 1.0;
   RealT bulk_mod2 = 1.0;
   RealT vel_factor = 100.;
   RealT velX1 = 0.;
   RealT velY1 = 0.;
   RealT velZ1 = vel_factor * 0.3 * element_thickness1 / dt;
   RealT velX2 = 0.;
   RealT velY2 = 0.;
   RealT velZ2 = vel_factor * 0.3 * element_thickness2 / dt;

   this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
   this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId,  velX2, velY2, -velZ2 ); 
   
   // allocate and set element thickness and bulk modulus
   this->m_mesh.allocateAndSetElementThickness( m_mesh.mortarMeshId, element_thickness1 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.mortarMeshId, bulk_mod1 );
   this->m_mesh.allocateAndSetElementThickness( m_mesh.nonmortarMeshId, element_thickness2 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.nonmortarMeshId, bulk_mod2 );

   // call tribol setup and update
   tribol::TestControlParameters parameters; 
   parameters.penalty_ratio = true;
   parameters.const_penalty = 0.75;
   parameters.dt = dt;
   parameters.enable_timestep_vote = true;
   parameters.timestep_pen_frac = 0.3;

   int test_mesh_update_err = 
      this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, 
                                         tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

   EXPECT_EQ( test_mesh_update_err, 0 );

   RealT dt_vote = parameters.timestep_pen_frac*element_thickness1/velZ1;
   RealT dt_diff = std::abs(parameters.dt - dt_vote);
   RealT dt_tol = 1.e-8;
   EXPECT_LT( dt_diff, dt_tol );

   tribol::finalize();
}

TEST_F( CommonPlaneTest, separation_velocity_small_gap )
{
   // This test makes sure that there is no change to dt in 
   // the presence of a separation velocity with small gap
   this->m_mesh.mortarMeshId = 0;
   this->m_mesh.nonmortarMeshId = 1;

   int nMortarElems = 4;
   int nElemsXM = nMortarElems;
   int nElemsYM = nMortarElems;
   int nElemsZM = nMortarElems;

   int nNonmortarElems = 5; 
   int nElemsXS = nNonmortarElems;
   int nElemsYS = nNonmortarElems;
   int nElemsZS = nNonmortarElems;

   // mesh bounding box with 0.055 interpenetration gap
   RealT x_min1 = 0.;
   RealT y_min1 = 0.;
   RealT z_min1 = 0.; 
   RealT x_max1 = 1.;
   RealT y_max1 = 1.;
   RealT z_max1 = 1.005;

   RealT x_min2 = 0.;
   RealT y_min2 = 0.;
   RealT z_min2 = 0.95;
   RealT x_max2 = 1.;
   RealT y_max2 = 1.;
   RealT z_max2 = 2.;

   // compute element thickness for each block
   RealT element_thickness1 = (z_max1 - z_min1) / nElemsZM;
   RealT element_thickness2 = (z_max2 - z_min2) / nElemsZS;

   // setup mesh
   this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM,
                                     x_min1, y_min1, z_min1,
                                     x_max1, y_max1, z_max1,
                                     nElemsXS, nElemsYS, nElemsZS,
                                     x_min2, y_min2, z_min2,
                                     x_max2, y_max2, z_max2,
                                     0., 0. );

   // specify dt and component velocities for each block.
   // Large velocity in the z-direction will incite a change in the 
   // timestep. This velocity is computed on the high side using the 
   // hardcoded rule that one face cannot interpen the other 
   // exceeding 30% of the other's element thickness.
   RealT dt = 1.0;
   RealT bulk_mod1 = 1.0;
   RealT bulk_mod2 = 1.0;
   RealT vel_factor = 100.;
   RealT velX1 = 0.;
   RealT velY1 = 0.;
   RealT velZ1 = vel_factor * 0.3 * element_thickness1 / dt;
   RealT velX2 = 0.;
   RealT velY2 = 0.;
   RealT velZ2 = vel_factor * 0.3 * element_thickness2 / dt;

   this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, -velZ1 );
   this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId,  velX2, velY2, velZ2 ); 
   
   // allocate and set element thickness and bulk modulus
   this->m_mesh.allocateAndSetElementThickness( m_mesh.mortarMeshId, element_thickness1 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.mortarMeshId, bulk_mod1 );
   this->m_mesh.allocateAndSetElementThickness( m_mesh.nonmortarMeshId, element_thickness2 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.nonmortarMeshId, bulk_mod2 );

   // call tribol setup and update
   tribol::TestControlParameters parameters; 
   parameters.penalty_ratio = true;
   parameters.const_penalty = 0.75;
   parameters.dt = dt;
   parameters.enable_timestep_vote = true;
   parameters.timestep_pen_frac = 0.3;

   int test_mesh_update_err = 
      this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, 
                                         tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

   EXPECT_EQ( test_mesh_update_err, 0 );

   // no change in dt because of separation velocities
   EXPECT_EQ( parameters.dt, dt );

   tribol::finalize();
}

TEST_F( CommonPlaneTest, large_velocity_large_separation )
{
   // This test uses two blocks with a large initial separation, and an interpen
   // velocity small enough that it should not trigger a timestep vote from the 
   // velocity projection check
   this->m_mesh.mortarMeshId = 0;
   this->m_mesh.nonmortarMeshId = 1;

   int nMortarElems = 4;
   int nElemsXM = nMortarElems;
   int nElemsYM = nMortarElems;
   int nElemsZM = nMortarElems;

   int nNonmortarElems = 5; 
   int nElemsXS = nNonmortarElems;
   int nElemsYS = nNonmortarElems;
   int nElemsZS = nNonmortarElems;

   // mesh bounding box with initial separation
   RealT x_min1 = 0.;
   RealT y_min1 = 0.;
   RealT z_min1 = 0.; 
   RealT x_max1 = 1.;
   RealT y_max1 = 1.;
   RealT z_max1 = 1.;

   RealT x_min2 = 0.;
   RealT y_min2 = 0.;
   RealT z_min2 = 2.1;
   RealT x_max2 = 1.;
   RealT y_max2 = 1.;
   RealT z_max2 = 3.1;

   // compute element thickness for each block
   RealT element_thickness1 = (z_max1 - z_min1) / nElemsZM;
   RealT element_thickness2 = (z_max2 - z_min2) / nElemsZS;

   // setup mesh
   this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM,
                                     x_min1, y_min1, z_min1,
                                     x_max1, y_max1, z_max1,
                                     nElemsXS, nElemsYS, nElemsZS,
                                     x_min2, y_min2, z_min2,
                                     x_max2, y_max2, z_max2,
                                     0., 0. );

   // specify dt and component velocities for each block.
   // Large velocity in the z-direction will incite a change in the 
   // timestep. This velocity is computed on the high side using the 
   // hardcoded rule that one face cannot interpen the other 
   // exceeding 30% of the other's element thickness.
   RealT dt = 1.0;
   RealT bulk_mod1 = 1.0;
   RealT bulk_mod2 = 1.0;
   RealT vel_factor = 10.;
   RealT velX1 = 0.;
   RealT velY1 = 0.;
   RealT velZ1 = vel_factor * 0.3 * element_thickness1 / dt;
   RealT velX2 = 0.;
   RealT velY2 = 0.;
   RealT velZ2 = vel_factor * 0.3 * element_thickness2 / dt;

   this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
   this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId,  velX2, velY2, -velZ2 ); 
   
   // allocate and set element thickness and bulk modulus
   this->m_mesh.allocateAndSetElementThickness( m_mesh.mortarMeshId, element_thickness1 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.mortarMeshId, bulk_mod1 );
   this->m_mesh.allocateAndSetElementThickness( m_mesh.nonmortarMeshId, element_thickness2 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.nonmortarMeshId, bulk_mod2 );

   // call tribol setup and update
   tribol::TestControlParameters parameters; 
   parameters.penalty_ratio = true;
   parameters.const_penalty = 0.75;
   parameters.dt = dt;
   parameters.enable_timestep_vote = true;
   parameters.timestep_pen_frac = 0.3;

   int test_mesh_update_err = 
      this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, 
                                         tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

   EXPECT_EQ( test_mesh_update_err, 0 );

   // no change in dt due to separation velocities
   EXPECT_EQ( parameters.dt, dt );

   tribol::finalize();
}

TEST_F( CommonPlaneTest, large_velocity_small_separation )
{
   // this test checks the two blocks with a small initial separation 
   // and a large velocity. This should trigger a velocity projection 
   // timestep vote with the face pairs marked as contact candidates
   this->m_mesh.mortarMeshId = 0;
   this->m_mesh.nonmortarMeshId = 1;

   int nMortarElems = 4;
   int nElemsXM = nMortarElems;
   int nElemsYM = nMortarElems;
   int nElemsZM = nMortarElems;

   int nNonmortarElems = 4;
   int nElemsXS = nNonmortarElems;
   int nElemsYS = nNonmortarElems;
   int nElemsZS = nNonmortarElems;

   // mesh bounding box with initial separation
   RealT x_min1 = 0.;
   RealT y_min1 = 0.;
   RealT z_min1 = 0.; 
   RealT x_max1 = 1.;
   RealT y_max1 = 1.;
   RealT z_max1 = 1.;

   RealT x_min2 = 0.;
   RealT y_min2 = 0.;
   RealT z_min2 = 1.0001;
   RealT x_max2 = 1.;
   RealT y_max2 = 1.;
   RealT z_max2 = 2.0001;

   // compute element thickness for each block
   RealT element_thickness1 = (z_max1 - z_min1) / nElemsZM;
   RealT element_thickness2 = (z_max2 - z_min2) / nElemsZS;

   // setup mesh
   this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM,
                                     x_min1, y_min1, z_min1,
                                     x_max1, y_max1, z_max1,
                                     nElemsXS, nElemsYS, nElemsZS,
                                     x_min2, y_min2, z_min2,
                                     x_max2, y_max2, z_max2,
                                     0., 0. );

   // specify dt and component velocities for each block.
   // Large velocity in the z-direction will incite a change in the 
   // timestep. This velocity is computed on the high side using the 
   // hardcoded rule that one face cannot interpen the other 
   // exceeding 30% of the other's element thickness.
   RealT dt = 1.0;
   RealT bulk_mod1 = 1.0;
   RealT bulk_mod2 = 1.0;
   RealT vel_factor = 1.e6;
   RealT velX1 = 0.;
   RealT velY1 = 0.;
   RealT velZ1 = vel_factor * 0.3 * element_thickness1 / dt;
   RealT velX2 = 0.;
   RealT velY2 = 0.;
   RealT velZ2 = vel_factor * 0.3 * element_thickness2 / dt;

   this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
   this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId,  velX2, velY2, -velZ2 ); 
   
   // allocate and set element thickness and bulk modulus
   this->m_mesh.allocateAndSetElementThickness( m_mesh.mortarMeshId, element_thickness1 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.mortarMeshId, bulk_mod1 );
   this->m_mesh.allocateAndSetElementThickness( m_mesh.nonmortarMeshId, element_thickness2 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.nonmortarMeshId, bulk_mod2 );

   // call tribol setup and update
   tribol::TestControlParameters parameters; 
   parameters.penalty_ratio = true;
   parameters.const_penalty = 0.75;
   parameters.dt = dt;
   parameters.enable_timestep_vote = true;
   parameters.timestep_pen_frac = 0.3;

   int test_mesh_update_err = 
      this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, 
                                         tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

   EXPECT_EQ( test_mesh_update_err, 0 );
   RealT dt_vote = parameters.timestep_pen_frac*element_thickness1/velZ1;
   RealT dt_diff = std::abs(parameters.dt - dt_vote);
   RealT dt_tol = 1.e-8;
   EXPECT_LT( dt_diff, dt_tol );

   tribol::finalize();
}

TEST_F( CommonPlaneTest, large_velocity_small_separation_set_alpha )
{
   // this test checks the two blocks with a small initial separation 
   // and a large velocity. This should trigger a velocity projection 
   // timestep vote with the face pairs marked as contact candidates
   this->m_mesh.mortarMeshId = 0;
   this->m_mesh.nonmortarMeshId = 1;

   int nMortarElems = 4;
   int nElemsXM = nMortarElems;
   int nElemsYM = nMortarElems;
   int nElemsZM = nMortarElems;

   int nNonmortarElems = 4;
   int nElemsXS = nNonmortarElems;
   int nElemsYS = nNonmortarElems;
   int nElemsZS = nNonmortarElems;

   // mesh bounding box with initial separation
   RealT x_min1 = 0.;
   RealT y_min1 = 0.;
   RealT z_min1 = 0.; 
   RealT x_max1 = 1.;
   RealT y_max1 = 1.;
   RealT z_max1 = 1.;

   RealT x_min2 = 0.;
   RealT y_min2 = 0.;
   RealT z_min2 = 1.0001;
   RealT x_max2 = 1.;
   RealT y_max2 = 1.;
   RealT z_max2 = 2.0001;

   // compute element thickness for each block
   RealT element_thickness1 = (z_max1 - z_min1) / nElemsZM;
   RealT element_thickness2 = (z_max2 - z_min2) / nElemsZS;

   // setup mesh
   this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM,
                                     x_min1, y_min1, z_min1,
                                     x_max1, y_max1, z_max1,
                                     nElemsXS, nElemsYS, nElemsZS,
                                     x_min2, y_min2, z_min2,
                                     x_max2, y_max2, z_max2,
                                     0., 0. );

   // specify dt and component velocities for each block.
   // Large velocity in the z-direction will incite a change in the 
   // timestep. This velocity is computed on the high side using the 
   // hardcoded rule that one face cannot interpen the other 
   // exceeding 30% of the other's element thickness.
   RealT dt = 1.0;
   RealT bulk_mod1 = 1.0;
   RealT bulk_mod2 = 1.0;
   RealT vel_factor = 1.e6;
   RealT velX1 = 0.;
   RealT velY1 = 0.;
   RealT velZ1 = vel_factor * 0.3 * element_thickness1 / dt;
   RealT velX2 = 0.;
   RealT velY2 = 0.;
   RealT velZ2 = vel_factor * 0.3 * element_thickness2 / dt;

   this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
   this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId,  velX2, velY2, -velZ2 ); 
   
   // allocate and set element thickness and bulk modulus
   this->m_mesh.allocateAndSetElementThickness( m_mesh.mortarMeshId, element_thickness1 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.mortarMeshId, bulk_mod1 );
   this->m_mesh.allocateAndSetElementThickness( m_mesh.nonmortarMeshId, element_thickness2 );
   this->m_mesh.allocateAndSetBulkModulus( m_mesh.nonmortarMeshId, bulk_mod2 );

   // call tribol setup and update
   tribol::TestControlParameters parameters; 
   parameters.penalty_ratio = true;
   parameters.const_penalty = 0.75;
   parameters.dt = dt;
   parameters.enable_timestep_vote = true;
   parameters.timestep_pen_frac = 0.3;
   parameters.timestep_scale = 0.5;

   int test_mesh_update_err = 
      this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, 
                                         tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

   EXPECT_EQ( test_mesh_update_err, 0 );
   RealT dt_vote = parameters.timestep_scale*parameters.timestep_pen_frac*element_thickness1/velZ1;
   RealT dt_diff = std::abs(parameters.dt - dt_vote);
   RealT dt_tol = 1.e-8;
   EXPECT_LT( dt_diff, dt_tol );

   tribol::finalize();
}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
