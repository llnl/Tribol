// Copyright (c) Lawrence Livermore National Security, LLC and
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
#include "SlicErrorTest.hpp"

// c++ includes
#include <cmath>  // std::abs, std::cos, std::sin
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>

using RealT = tribol::RealT;

namespace {

constexpr tribol::TimestepCriteria penetration_only_criteria{
    .current_gap = true, .velocity_projection = true, .nodal_penalty_cfl = false, .pair_spring_mass_cfl = false };
constexpr tribol::TimestepCriteria nodal_penalty_cfl_only_criteria{
    .current_gap = false, .velocity_projection = false, .nodal_penalty_cfl = true, .pair_spring_mass_cfl = false };
constexpr tribol::TimestepCriteria pair_spring_mass_cfl_only_criteria{
    .current_gap = false, .velocity_projection = false, .nodal_penalty_cfl = false, .pair_spring_mass_cfl = true };

}  // namespace

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
  RealT element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  RealT element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // specify dt and component velocities for each block.
  // Large velocity in the z-direction will incite a change in the
  // timestep. This velocity is computed on the high side using the
  // hardcoded rule that one face cannot interpen the other
  // exceeding 30% of the other's element thickness.
  RealT dt = 1.0;
  RealT bulk_mod1 = 1.0;  // something simple
  RealT bulk_mod2 = 1.0;
  RealT velX1 = 0.;
  RealT velY1 = 0.;
  RealT velZ1 = 0.;
  RealT velX2 = 0.;
  RealT velY2 = 0.;
  RealT velZ2 = 0.;

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
  parameters.enable_timestep_vote = true;
  parameters.timestep_pen_frac = 0.3;
  parameters.timestep_criteria = penetration_only_criteria;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );
  EXPECT_TRUE( std::isinf( parameters.dt ) );

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
  RealT element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  RealT element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

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
  parameters.enable_timestep_vote = false;
  parameters.timestep_pen_frac = 0.3;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );
  EXPECT_EQ( parameters.dt, dt );

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
  RealT element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  RealT element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // specify dt and component velocities for each block.
  // Large velocity in the z-direction will incite a change in the
  // timestep. This velocity is computed on the high side using the
  // hardcoded rule that one face cannot interpen the other
  // exceeding 30% of the other's element thickness.
  RealT dt = 1.0;
  RealT bulk_mod1 = 1.0;  // something simple
  RealT bulk_mod2 = 1.0;
  RealT velX1 = 1.e-12;
  RealT velY1 = 1.e-12;
  RealT velZ1 = 1.e-12;
  RealT velX2 = -1.e-12;
  RealT velY2 = -1.e-12;
  RealT velZ2 = -1.e-12;

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
  parameters.enable_timestep_vote = true;
  parameters.timestep_pen_frac = 0.3;
  parameters.timestep_criteria = penetration_only_criteria;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  // expect infinite dt when no criterion votes
  EXPECT_TRUE( std::isinf( parameters.dt ) );

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
  RealT z_max1 = 1.0 + 1.05 * interpen_gap;

  RealT x_min2 = 0.;
  RealT y_min2 = 0.;
  RealT z_min2 = 1.0 - 1.05 * interpen_gap;
  RealT x_max2 = 1.;
  RealT y_max2 = 1.;
  RealT z_max2 = 2.;

  // compute element thickness for each block
  RealT element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  RealT element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

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
  parameters.enable_timestep_vote = true;
  parameters.timestep_pen_frac = 0.3;
  parameters.timestep_criteria = penetration_only_criteria;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );
  // With zero relative closing speed, no penetration criterion votes (+infinity).
  EXPECT_TRUE( std::isinf( parameters.dt ) );

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
  RealT element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  RealT element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

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
  parameters.enable_timestep_vote = true;
  parameters.timestep_pen_frac = 0.3;
  parameters.timestep_criteria = penetration_only_criteria;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  RealT dt_vote = parameters.timestep_pen_frac * std::min( element_thickness1, element_thickness2 ) / ( velZ1 + velZ2 );
  RealT dt_tol = 1.e-8;
  EXPECT_NEAR( parameters.dt, dt_vote, dt_tol );

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
  RealT z_max1 = 1.0 + 1.05 * interpen_gap;

  RealT x_min2 = 0.;
  RealT y_min2 = 0.;
  RealT z_min2 = 1.0 - 1.05 * interpen_gap;
  RealT x_max2 = 1.;
  RealT y_max2 = 1.;
  RealT z_max2 = 2.;

  // compute element thickness for each block
  RealT element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  RealT element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

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
  parameters.enable_timestep_vote = true;
  parameters.timestep_pen_frac = 0.3;
  parameters.timestep_criteria = penetration_only_criteria;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  RealT dt_vote = parameters.timestep_pen_frac * std::min( element_thickness1, element_thickness2 ) / ( velZ1 + velZ2 );
  RealT dt_tol = 1.e-8;
  EXPECT_NEAR( parameters.dt, dt_vote, dt_tol );

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
  RealT element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  RealT element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

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
  parameters.enable_timestep_vote = true;
  parameters.timestep_pen_frac = 0.3;
  parameters.timestep_criteria = penetration_only_criteria;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  // infinite dt because separation velocities trigger no criterion
  EXPECT_TRUE( std::isinf( parameters.dt ) );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, large_velocity_large_separation )
{
  // This test uses two blocks with a large initial separation, but within
  // the binning proximity, and a small enough velocity that a timestep vote
  // should not be triggered
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
  RealT element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  RealT element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // specify dt and component velocities for each block.
  // The velocity in the z-direction is such that there is interpenetration
  // given the initial separation, but does not exceed the 30% element thickness that
  // would trigger a timestep vote. Note, each z-dir velocity is multiplied by 0.5
  // to account for the combined velocity of the two blocks coming toward one another.
  RealT dt = 1.0;
  RealT bulk_mod1 = 1.0;
  RealT bulk_mod2 = 1.0;
  RealT vel_factor = 1.0;
  RealT velX1 = 0.;
  RealT velY1 = 0.;
  RealT velZ1 = 0.5 * vel_factor * ( z_min2 - z_max1 + 0.25 * element_thickness1 ) / dt;
  RealT velX2 = 0.;
  RealT velY2 = 0.;
  RealT velZ2 = 0.5 * vel_factor * ( z_min2 - z_max1 + 0.25 * element_thickness2 ) / dt;

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
  parameters.enable_timestep_vote = true;
  parameters.timestep_pen_frac = 0.3;
  parameters.timestep_criteria = penetration_only_criteria;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  // infinite dt because projected gap does not exceed max_delta
  EXPECT_TRUE( std::isinf( parameters.dt ) );

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
  RealT element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  RealT element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  // setup mesh
  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

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
  parameters.enable_timestep_vote = true;
  parameters.timestep_pen_frac = 0.3;
  parameters.timestep_criteria = penetration_only_criteria;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );
  RealT dt_vote = parameters.timestep_pen_frac * std::min( element_thickness1, element_thickness2 ) / ( velZ1 + velZ2 );
  RealT dt_tol = 1.e-8;
  EXPECT_NEAR( parameters.dt, dt_vote, dt_tol );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, cfl_nodal_penalty_only_without_velocity )
{
  // Enabling only nodal_penalty_cfl must compute the analytical CFL timestep without registering velocities or
  // element thicknesses.
  this->m_mesh.mortarMeshId = 0;
  this->m_mesh.nonmortarMeshId = 1;

  // Single conforming 1x1x1 hex on each block with overlap area = 1.0 and small interpenetration.
  this->m_mesh.setupContactMeshHex( 1, 1, 1, 0., 0., 0., 1., 1., 1.01, 1, 1, 1, 0., 0., 0.99, 1., 1., 2.0, 0., 0. );

  // Uniform inverse mass = 0.25 on both meshes (sqrt(inv_m) = 0.5).
  this->m_mesh.allocateAndSetNodalInverseMass( m_mesh.mortarMeshId, 0.25, 0.25, 0.25 );
  this->m_mesh.allocateAndSetNodalInverseMass( m_mesh.nonmortarMeshId, 0.25, 0.25, 0.25 );

  tribol::TestControlParameters parameters;
  parameters.penalty_ratio = false;
  parameters.const_penalty = 8.0;  // Series penalty = 8 * 8 / (8 + 8) = 4.0.
  parameters.dt = 0.5;             // Smaller than dt_cfl to verify dt_vote is returned unclamped.
  parameters.enable_timestep_vote = true;
  parameters.timestep_criteria = nodal_penalty_cfl_only_criteria;
  parameters.explicit_integrator_stability_factor = 2.0;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );
  // Single-point centroid on a 4-node quad has phi_a = 0.25 on each of the 4 nodes of both faces.
  // normal_abs_sum = 8 * (0.25 * 0.5) = 1.0.
  // Each z-DOF row sum = (area=1.0 * k_pen=4.0) * (0.25 * 0.5) * 1.0 = 0.5.
  // dt_cfl = 2.0 / sqrt(0.5) = 2 * sqrt(2).
  EXPECT_NEAR( parameters.dt, 2.0 / std::sqrt( 0.5 ), 1.0e-10 );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, cfl_pair_spring_mass_asymmetric_and_rigid )
{
  // Enabling only pair_spring_mass_cfl must match C_stab / sqrt(k_pen * A * (inv_m1 + inv_m2)).
  this->m_mesh.mortarMeshId = 0;
  this->m_mesh.nonmortarMeshId = 1;

  this->m_mesh.setupContactMeshHex( 1, 1, 1, 0., 0., 0., 1., 1., 1.01, 1, 1, 1, 0., 0., 0.99, 1., 1., 2.0, 0., 0. );

  // 4 nodes per face: mesh1 has inv_m = 1.0 per node -> face mass = 4.0 -> face inv_m1 = 0.25.
  // mesh2 is rigid (inv_m = 0.0 -> face inv_m2 = 0.0).
  this->m_mesh.allocateAndSetNodalInverseMass( m_mesh.mortarMeshId, 1.0, 1.0, 1.0 );
  this->m_mesh.allocateAndSetNodalInverseMass( m_mesh.nonmortarMeshId, 0.0, 0.0, 0.0 );

  tribol::TestControlParameters parameters;
  parameters.penalty_ratio = false;
  parameters.const_penalty = 8.0;  // Series penalty = 4.0.
  parameters.dt = 10.0;
  parameters.enable_timestep_vote = true;
  parameters.timestep_criteria = pair_spring_mass_cfl_only_criteria;
  parameters.explicit_integrator_stability_factor = 2.0;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );
  // stiffness_bound = 4.0 * 1.0 * (0.25 + 0.0) = 1.0 -> dt = 2.0 / sqrt(1.0) = 2.0.
  EXPECT_NEAR( parameters.dt, 2.0, 1.0e-10 );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, missing_inverse_mass_reports_slic_error )
{
  // Enabling nodal_penalty_cfl without registering inverse mass must report a SLIC error rather than silently skipping.
  this->m_mesh.mortarMeshId = 0;
  this->m_mesh.nonmortarMeshId = 1;

  this->m_mesh.setupContactMeshHex( 1, 1, 1, 0., 0., 0., 1., 1., 1.01, 1, 1, 1, 0., 0., 0.99, 1., 1., 2.0, 0., 0. );

  tribol::TestControlParameters parameters;
  parameters.penalty_ratio = false;
  parameters.const_penalty = 8.0;
  parameters.dt = 1.0;
  parameters.enable_timestep_vote = true;
  parameters.timestep_criteria = nodal_penalty_cfl_only_criteria;

  EXPECT_SLIC_ERROR( this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS,
                                                        tribol::NO_CASE, false, parameters ) );

  tribol::finalize();
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
