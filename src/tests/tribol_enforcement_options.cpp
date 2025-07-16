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

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

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

using RealT = tribol::RealT;

/*!
 * Test fixture class to test valid enforcement options
 * These tests have to be written and run with registering
 * a valid coupling scheme with valid meshes.
 *
 */
class EnforcementOptionsTest : public ::testing::Test {
 public:
 protected:
  void SetUp() override
  {
    // no-op
  }

  // Setup boiler plate data and register mesh, nodal response, and coupling scheme
  void SetupTest( tribol::TestMesh* mesh )
  {
    ////////////////////////////////////////////////
    // setup simple non-null contacting test mesh //
    ////////////////////////////////////////////////
    mesh->mortarMeshId = 0;
    mesh->nonmortarMeshId = 1;

    int nMortarElems = 1;
    int nElemsXM = nMortarElems;
    int nElemsYM = nMortarElems;
    int nElemsZM = nMortarElems;

    int nNonmortarElems = 1;
    int nElemsXS = nNonmortarElems;
    int nElemsYS = nNonmortarElems;
    int nElemsZS = nNonmortarElems;

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

    mesh->setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1, nElemsXS,
                               nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0., 0. );

    // register meshes
    tribol::registerMesh( mesh->mortarMeshId, mesh->numMortarFaces, mesh->numTotalNodes, mesh->faceConn1, 3, mesh->x,
                          mesh->y, mesh->z, tribol::MemorySpace::Host );

    tribol::registerMesh( mesh->nonmortarMeshId, mesh->numNonmortarFaces, mesh->numTotalNodes, mesh->faceConn2, 3,
                          mesh->x, mesh->y, mesh->z, tribol::MemorySpace::Host );

    // register nodal responses (i.e. nodal forces)
    tribol::allocRealArray( &mesh->fx1, mesh->numTotalNodes, 0. );
    tribol::allocRealArray( &mesh->fy1, mesh->numTotalNodes, 0. );
    tribol::allocRealArray( &mesh->fz1, mesh->numTotalNodes, 0. );
    tribol::allocRealArray( &mesh->fx2, mesh->numTotalNodes, 0. );
    tribol::allocRealArray( &mesh->fy2, mesh->numTotalNodes, 0. );
    tribol::allocRealArray( &mesh->fz2, mesh->numTotalNodes, 0. );

    tribol::registerNodalResponse( mesh->mortarMeshId, mesh->fx1, mesh->fy1, mesh->fz1 );

    tribol::registerNodalResponse( mesh->nonmortarMeshId, mesh->fx2, mesh->fy2, mesh->fz2 );

    // allocate velocity arrays on test mesh
    RealT velX1 = 0.;
    RealT velY1 = 0.;
    RealT velZ1 = -1.;
    RealT velX2 = 0.;
    RealT velY2 = 0.;
    RealT velZ2 = 1.;
    mesh->allocateAndSetVelocities( mesh->mortarMeshId, velX1, velY1, velZ1 );
    mesh->allocateAndSetVelocities( mesh->nonmortarMeshId, velX2, velY2, velZ2 );

    // register velocities with Tribol
    tribol::registerNodalVelocities( mesh->mortarMeshId, mesh->vx1, mesh->vy1, mesh->vz1 );
    tribol::registerNodalVelocities( mesh->nonmortarMeshId, mesh->vx2, mesh->vy2, mesh->vz2 );

    // register the coupling scheme
    const int csIndex = 0;
    tribol::registerCouplingScheme( csIndex, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                    tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                    tribol::ExecutionMode::Sequential );
  }

  void TearDown() override { clear(); }

  void clear()
  {
    // no-op
  }
};

// TESTS1
// Coupling schemes with errors
TEST_F( EnforcementOptionsTest, penalty_kinematic_constant_error )
{
  // Setup boiler plate test data etc.
  tribol::TestMesh* mesh = new tribol::TestMesh();
  SetupTest( mesh );

  // set penalty data so coupling scheme initialization passes
  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  // incorrectly set penalty options with KINEMATIC_ELEMENT instead of the set KINEMATIC_CONSTANT
  int csIndex = 0;
  tribol::KinematicPenaltyCalculation wrong_calculation = tribol::KINEMATIC_ELEMENT;
  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, wrong_calculation );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( csIndex );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();

  delete mesh;
}

TEST_F( EnforcementOptionsTest, penalty_kinematic_element_error )
{
  // Setup boiler plate test data etc.
  tribol::TestMesh* mesh = new tribol::TestMesh();
  SetupTest( mesh );

  // set penalty data so coupling scheme initialization passes
  RealT* bulk_modulus_1;
  RealT* bulk_modulus_2;
  RealT* element_thickness_1;
  RealT* element_thickness_2;

  tribol::allocRealArray( &bulk_modulus_1, mesh->numMortarFaces, 0. );
  tribol::allocRealArray( &bulk_modulus_2, mesh->numNonmortarFaces, 0. );
  tribol::allocRealArray( &element_thickness_1, mesh->numMortarFaces, 0. );
  tribol::allocRealArray( &element_thickness_2, mesh->numNonmortarFaces, 0. );

  tribol::setKinematicElementPenalty( mesh->mortarMeshId, bulk_modulus_1, element_thickness_1 );
  tribol::setKinematicElementPenalty( mesh->nonmortarMeshId, bulk_modulus_2, element_thickness_2 );

  // 'incorrectly' set penalty options with KINEMATIC_CONSTANT instead of the set KINEMATIC_ELEMENT
  int csIndex = 0;
  tribol::KinematicPenaltyCalculation wrong_calculation = tribol::KINEMATIC_CONSTANT;
  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, wrong_calculation );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( csIndex );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();

  delete bulk_modulus_1;
  delete bulk_modulus_2;
  delete element_thickness_1;
  delete element_thickness_2;
  delete mesh;
}

TEST_F( EnforcementOptionsTest, penalty_kinematic_constant_rate_constant_error )
{
  // Setup boiler plate test data etc.
  tribol::TestMesh* mesh = new tribol::TestMesh();
  SetupTest( mesh );

  // set penalty data so coupling scheme initialization passes
  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  RealT rate_penalty = 0.5;
  tribol::setRateConstantPenalty( 0, rate_penalty );
  tribol::setRateConstantPenalty( 1, rate_penalty );

  // incorrectly set penalty options with RATE_PERCENT instead of the set RATE_CONSTANT
  int csIndex = 0;
  tribol::RatePenaltyCalculation wrong_calculation = tribol::RATE_PERCENT;
  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC_AND_RATE, tribol::KINEMATIC_CONSTANT, wrong_calculation );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( csIndex );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();

  delete mesh;
}

TEST_F( EnforcementOptionsTest, penalty_kinematic_constant_rate_percent_error_1 )
{
  // Setup boiler plate test data etc.
  tribol::TestMesh* mesh = new tribol::TestMesh();
  SetupTest( mesh );

  // set penalty data so coupling scheme initialization passes
  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  RealT rate_percent = 0.5;
  tribol::setRatePercentPenalty( 0, rate_percent );
  tribol::setRatePercentPenalty( 1, rate_percent );

  // incorrectly set penalty options with RATE_CONSTANT instead of the set RATE_PERCENT
  int csIndex = 0;
  tribol::RatePenaltyCalculation wrong_calculation = tribol::RATE_CONSTANT;
  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC_AND_RATE, tribol::KINEMATIC_CONSTANT, wrong_calculation );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( csIndex );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();

  delete mesh;
}

TEST_F( EnforcementOptionsTest, penalty_kinematic_constant_rate_percent_error_2 )
{
  // Setup boiler plate test data etc.
  tribol::TestMesh* mesh = new tribol::TestMesh();
  SetupTest( mesh );

  // set penalty data so coupling scheme initialization passes
  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  // incorrectly set the rate_percent value outside of acceptable bounds
  RealT rate_percent = -0.01;
  tribol::setRatePercentPenalty( 0, rate_percent );
  tribol::setRatePercentPenalty( 1, rate_percent );

  // (correctly) set penalty options with KINEMATIC_CONSTANT and RATE_PERCENT
  int csIndex = 0;
  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC_AND_RATE, tribol::KINEMATIC_CONSTANT, tribol::RATE_PERCENT );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( csIndex );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();

  delete mesh;
}

// TESTS2
// Coupling schemes with no errors
TEST_F( EnforcementOptionsTest, penalty_kinematic_constant_pass )
{
  // Setup boiler plate test data etc.
  tribol::TestMesh* mesh = new tribol::TestMesh();
  SetupTest( mesh );

  // set penalty data so coupling scheme initialization passes
  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  // set penalty options with KINEMATIC_CONSTANT
  int csIndex = 0;
  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( csIndex );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, true );

  tribol::finalize();

  delete mesh;
}

TEST_F( EnforcementOptionsTest, penalty_kinematic_element_pass )
{
  // Setup boiler plate test data etc.
  tribol::TestMesh* mesh = new tribol::TestMesh();
  SetupTest( mesh );

  // set penalty data so coupling scheme initialization passes
  RealT* bulk_modulus_1;
  RealT* bulk_modulus_2;
  RealT* element_thickness_1;
  RealT* element_thickness_2;

  tribol::allocRealArray( &bulk_modulus_1, mesh->numMortarFaces, 1. );
  tribol::allocRealArray( &bulk_modulus_2, mesh->numNonmortarFaces, 1. );
  tribol::allocRealArray( &element_thickness_1, mesh->numMortarFaces, 1. );
  tribol::allocRealArray( &element_thickness_2, mesh->numNonmortarFaces, 1. );

  tribol::setKinematicElementPenalty( mesh->mortarMeshId, bulk_modulus_1, element_thickness_1 );
  tribol::setKinematicElementPenalty( mesh->nonmortarMeshId, bulk_modulus_2, element_thickness_2 );

  // set penalty options with KINEMATIC_ELEMENT
  int csIndex = 0;
  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_ELEMENT );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( csIndex );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, true );

  tribol::finalize();

  delete bulk_modulus_1;
  delete bulk_modulus_2;
  delete element_thickness_1;
  delete element_thickness_2;
  delete mesh;
}

TEST_F( EnforcementOptionsTest, penalty_kinematic_element_invalid_element_input )
{
  // Setup boiler plate test data etc.
  tribol::TestMesh* mesh = new tribol::TestMesh();
  SetupTest( mesh );

  // set penalty data so coupling scheme initialization passes
  RealT* bulk_modulus_1;
  RealT* bulk_modulus_2;
  RealT* element_thickness_1;
  RealT* element_thickness_2;

  tribol::allocRealArray( &bulk_modulus_1, mesh->numMortarFaces, 0. );
  tribol::allocRealArray( &bulk_modulus_2, mesh->numNonmortarFaces, 0. );
  tribol::allocRealArray( &element_thickness_1, mesh->numMortarFaces, 0. );
  tribol::allocRealArray( &element_thickness_2, mesh->numNonmortarFaces, 0. );

  tribol::setKinematicElementPenalty( mesh->mortarMeshId, bulk_modulus_1, element_thickness_1 );
  tribol::setKinematicElementPenalty( mesh->nonmortarMeshId, bulk_modulus_2, element_thickness_2 );

  // set penalty options with KINEMATIC_ELEMENT
  int csIndex = 0;
  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_ELEMENT );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( csIndex );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();

  delete bulk_modulus_1;
  delete bulk_modulus_2;
  delete element_thickness_1;
  delete element_thickness_2;
  delete mesh;
}

TEST_F( EnforcementOptionsTest, penalty_kinematic_constant_rate_constant_pass )
{
  // Setup boiler plate test data etc.
  tribol::TestMesh* mesh = new tribol::TestMesh();
  SetupTest( mesh );

  // set penalty data so coupling scheme initialization passes
  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  RealT rate_penalty = 0.5;
  tribol::setRateConstantPenalty( 0, rate_penalty );
  tribol::setRateConstantPenalty( 1, rate_penalty );

  // set penalty options with KINEMATIC_CONSTANT and RATE_CONSTANT
  int csIndex = 0;
  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC_AND_RATE, tribol::KINEMATIC_CONSTANT, tribol::RATE_CONSTANT );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( csIndex );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, true );

  tribol::finalize();

  delete mesh;
}

TEST_F( EnforcementOptionsTest, penalty_kinematic_constant_rate_percent_pass )
{
  // Setup boiler plate test data etc.
  tribol::TestMesh* mesh = new tribol::TestMesh();
  SetupTest( mesh );

  // set penalty data so coupling scheme initialization passes
  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  RealT rate_percent = 0.5;
  tribol::setRatePercentPenalty( 0, rate_percent );
  tribol::setRatePercentPenalty( 1, rate_percent );

  // set penalty options with KINEMATIC_CONSTANT and RATE_PERCENT
  int csIndex = 0;
  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC_AND_RATE, tribol::KINEMATIC_CONSTANT, tribol::RATE_PERCENT );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( csIndex );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, true );

  tribol::finalize();

  delete mesh;
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

  axom::slic::SimpleLogger logger;  // create & initialize logger,

  result = RUN_ALL_TESTS();

  return result;
}
