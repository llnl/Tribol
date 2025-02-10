// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/types.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/utils/TestUtils.hpp"
#include "tribol/utils/Math.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/CouplingSchemeManager.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MeshManager.hpp"
#include "tribol/physics/CommonPlane.hpp"
#include "tribol/geom/GeomUtilities.hpp"

// Axom includes
#include "axom/slic.hpp"

// gtest includes
#include "gtest/gtest.h"

// c++ includes
#include <cmath>  // std::abs, std::cos, std::sin
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>

using real = tribol::real;

/*!
 * Test fixture class with some setup necessary to test
 * the COMMON_PLANE + PENALTY implementation with registered
 * velocities and timestep vote
 */
class CommonPlaneTest : public ::testing::Test {
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

TEST_F( CommonPlaneTest, zero_velocity_small_gap )
{
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
  real x_min1 = 0.;
  real y_min1 = 0.;
  real z_min1 = 0.;
  real x_max1 = 1.;
  real y_max1 = 1.;
  real z_max1 = 1.005;

  real x_min2 = 0.;
  real y_min2 = 0.;
  real z_min2 = 0.95;
  real x_max2 = 1.;
  real y_max2 = 1.;
  real z_max2 = 2.;

  // compute element thickness for each block
  real element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  real element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // specify dt and component velocities for each block.
  // Large velocity in the z-direction will incite a change in the
  // timestep. This velocity is computed on the high side using the
  // hardcoded rule that one face cannot interpen the other
  // exceeding 30% of the other's element thickness.
  real dt = 1.e-3;
  real bulk_mod1 = 1.0;  // something simple
  real bulk_mod2 = 1.0;
  real velX1 = 0.;
  real velY1 = 0.;
  real velZ1 = 0.;
  real velX2 = 0.;
  real velY2 = 0.;
  real velZ2 = 0.;

  this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
  this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId, velX2, velY2, -velZ2 );

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

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY,
                                                                tribol::FRICTIONLESS, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );
  EXPECT_EQ( parameters.dt, dt );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, zero_velocity_large_gap )
{
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

  // mesh bounding box with 0.1 interpenetration gap
  real x_min1 = 0.;
  real y_min1 = 0.;
  real z_min1 = 0.;
  real x_max1 = 1.;
  real y_max1 = 1.;
  real z_max1 = 1.05;

  real x_min2 = 0.;
  real y_min2 = 0.;
  real z_min2 = 0.95;
  real x_max2 = 1.;
  real y_max2 = 1.;
  real z_max2 = 2.;

  // compute element thickness for each block
  real element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  real element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // specify dt and component velocities for each block.
  // Large velocity in the z-direction will incite a change in the
  // timestep. This velocity is computed on the high side using the
  // hardcoded rule that one face cannot interpen the other
  // exceeding 30% of the other's element thickness.
  real dt = 1.e-3;
  real bulk_mod1 = 1.0;
  real bulk_mod2 = 1.0;
  real velX1 = 0.;
  real velY1 = 0.;
  real velZ1 = 0.;
  real velX2 = 0.;
  real velY2 = 0.;
  real velZ2 = 0.;

  this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
  this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId, velX2, velY2, -velZ2 );

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

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY,
                                                                tribol::FRICTIONLESS, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );
  // note that with very small velocity, the dt estimate will be
  // very large.
  EXPECT_EQ( parameters.dt, dt );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, large_velocity_small_gap )
{
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
  real x_min1 = 0.;
  real y_min1 = 0.;
  real z_min1 = 0.;
  real x_max1 = 1.;
  real y_max1 = 1.;
  real z_max1 = 1.005;

  real x_min2 = 0.;
  real y_min2 = 0.;
  real z_min2 = 0.95;
  real x_max2 = 1.;
  real y_max2 = 1.;
  real z_max2 = 2.;

  // compute element thickness for each block
  real element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  real element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // specify dt and component velocities for each block.
  // Large velocity in the z-direction will incite a change in the
  // timestep. This velocity is computed on the high side using the
  // hardcoded rule that one face cannot interpen the other
  // exceeding 30% of the other's element thickness.
  real dt = 1.e-3;
  real bulk_mod1 = 1.0;
  real bulk_mod2 = 1.0;
  real vel_factor = 100.;
  real velX1 = 0.;
  real velY1 = 0.;
  real velZ1 = vel_factor * 0.3 * element_thickness1 / dt;
  real velX2 = 0.;
  real velY2 = 0.;
  real velZ2 = vel_factor * 0.3 * element_thickness2 / dt;

  this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
  this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId, velX2, velY2, -velZ2 );

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

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY,
                                                                tribol::FRICTIONLESS, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  real dt_diff = std::abs( parameters.dt - 0.000747973 );
  real dt_tol = 1.e-8;
  EXPECT_LT( dt_diff, dt_tol );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, large_velocity_large_gap )
{
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

  // mesh bounding box with 0.1 interpenetration gap
  real x_min1 = 0.;
  real y_min1 = 0.;
  real z_min1 = 0.;
  real x_max1 = 1.;
  real y_max1 = 1.;
  real z_max1 = 1.05;

  real x_min2 = 0.;
  real y_min2 = 0.;
  real z_min2 = 0.95;
  real x_max2 = 1.;
  real y_max2 = 1.;
  real z_max2 = 2.;

  // compute element thickness for each block
  real element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  real element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // specify dt and component velocities for each block.
  // Large velocity in the z-direction will incite a change in the
  // timestep. This velocity is computed on the high side using the
  // hardcoded rule that one face cannot interpen the other
  // exceeding 30% of the other's element thickness.
  real dt = 1.e-3;
  real bulk_mod1 = 1.0;
  real bulk_mod2 = 1.0;
  real vel_factor = 100.;
  real velX1 = 0.;
  real velY1 = 0.;
  real velZ1 = vel_factor * 0.3 * element_thickness1 / dt;
  real velX2 = 0.;
  real velY2 = 0.;
  real velZ2 = vel_factor * 0.3 * element_thickness2 / dt;

  this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
  this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId, velX2, velY2, -velZ2 );

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

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY,
                                                                tribol::FRICTIONLESS, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  real dt_diff = std::abs( parameters.dt - 2.02381e-06 );
  real dt_tol = 1.e-8;
  EXPECT_LT( dt_diff, dt_tol );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, separation_velocity_small_gap )
{
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
  real x_min1 = 0.;
  real y_min1 = 0.;
  real z_min1 = 0.;
  real x_max1 = 1.;
  real y_max1 = 1.;
  real z_max1 = 1.005;

  real x_min2 = 0.;
  real y_min2 = 0.;
  real z_min2 = 0.95;
  real x_max2 = 1.;
  real y_max2 = 1.;
  real z_max2 = 2.;

  // compute element thickness for each block
  real element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  real element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // specify dt and component velocities for each block.
  // Large velocity in the z-direction will incite a change in the
  // timestep. This velocity is computed on the high side using the
  // hardcoded rule that one face cannot interpen the other
  // exceeding 30% of the other's element thickness.
  real dt = 1.e-3;
  real bulk_mod1 = 1.0;
  real bulk_mod2 = 1.0;
  real vel_factor = 100.;
  real velX1 = 0.;
  real velY1 = 0.;
  real velZ1 = vel_factor * 0.3 * element_thickness1 / dt;
  real velX2 = 0.;
  real velY2 = 0.;
  real velZ2 = vel_factor * 0.3 * element_thickness2 / dt;

  this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, -velZ1 );
  this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId, velX2, velY2, velZ2 );

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

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY,
                                                                tribol::FRICTIONLESS, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );
  EXPECT_EQ( parameters.dt, dt );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, large_velocity_separation )
{
  // this test has two blocks has two blocks with initial separation
  // and checks to make sure that the timestep is not altered.
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
  real x_min1 = 0.;
  real y_min1 = 0.;
  real z_min1 = 0.;
  real x_max1 = 1.;
  real y_max1 = 1.;
  real z_max1 = 1.;

  real x_min2 = 0.;
  real y_min2 = 0.;
  real z_min2 = 2.1;
  real x_max2 = 1.;
  real y_max2 = 1.;
  real z_max2 = 3.1;

  // compute element thickness for each block
  real element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  real element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // specify dt and component velocities for each block.
  // Large velocity in the z-direction will incite a change in the
  // timestep. This velocity is computed on the high side using the
  // hardcoded rule that one face cannot interpen the other
  // exceeding 30% of the other's element thickness.
  real dt = 1.e-3;
  real bulk_mod1 = 1.0;
  real bulk_mod2 = 1.0;
  real vel_factor = 10.;
  real velX1 = 0.;
  real velY1 = 0.;
  real velZ1 = vel_factor * 0.3 * element_thickness1 / dt;
  real velX2 = 0.;
  real velY2 = 0.;
  real velZ2 = vel_factor * 0.3 * element_thickness2 / dt;

  this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
  this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId, velX2, velY2, -velZ2 );

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

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY,
                                                                tribol::FRICTIONLESS, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );
  EXPECT_EQ( parameters.dt, dt );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, large_velocity_small_separation )
{
  // this test checks the two blocks with a small initial separation
  // and a large velocity. This should trigger a velocity projection
  // timestep vote
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
  real x_min1 = 0.;
  real y_min1 = 0.;
  real z_min1 = 0.;
  real x_max1 = 1.;
  real y_max1 = 1.;
  real z_max1 = 1.;

  real x_min2 = 0.;
  real y_min2 = 0.;
  real z_min2 = 1.0001;
  real x_max2 = 1.;
  real y_max2 = 1.;
  real z_max2 = 2.0001;

  // compute element thickness for each block
  real element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  real element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // specify dt and component velocities for each block.
  // Large velocity in the z-direction will incite a change in the
  // timestep. This velocity is computed on the high side using the
  // hardcoded rule that one face cannot interpen the other
  // exceeding 30% of the other's element thickness.
  real dt = 1.e-3;
  real bulk_mod1 = 1.0;
  real bulk_mod2 = 1.0;
  real vel_factor = 1.e6;
  real velX1 = 0.;
  real velY1 = 0.;
  real velZ1 = vel_factor * 0.3 * element_thickness1 / dt;
  real velX2 = 0.;
  real velY2 = 0.;
  real velZ2 = vel_factor * 0.3 * element_thickness2 / dt;

  this->m_mesh.allocateAndSetVelocities( m_mesh.mortarMeshId, velX1, velY1, velZ1 );
  this->m_mesh.allocateAndSetVelocities( m_mesh.nonmortarMeshId, velX2, velY2, -velZ2 );

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

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY,
                                                                tribol::FRICTIONLESS, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );
  real dt_diff = std::abs( parameters.dt - 0.000749999 );
  real dt_tol = 1.e-8;
  EXPECT_LT( dt_diff, dt_tol );

  tribol::finalize();
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
