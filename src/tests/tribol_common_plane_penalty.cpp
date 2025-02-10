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

void compareGaps( tribol::CouplingScheme const* cs, real gap, const real tol, const char* gapType )
{
  tribol::ContactPlaneManager& cpManager = tribol::ContactPlaneManager::getInstance();
  tribol::InterfacePairs const* const pairs = cs->getInterfacePairs();
  tribol::IndexType const numPairs = pairs->getNumPairs();

  int cpID = 0;
  for ( tribol::IndexType kp = 0; kp < numPairs; ++kp ) {
    tribol::InterfacePair pair = pairs->getInterfacePair( kp );

    if ( !pair.isContactCandidate ) {
      continue;
    }

    real my_gap = 0.;
    if ( std::strcmp( gapType, "kinematic_penetration" ) == 0 || std::strcmp( gapType, "kinematic_separation" ) == 0 ) {
      my_gap = cpManager.m_gap[cpID];
    } else {
      my_gap = cpManager.m_velGap[cpID];
    }

    double gap_tol = cs->getGapTol( pair.pairIndex1, pair.pairIndex2 );

    // check gap sense
    if ( std::strcmp( gapType, "kinematic_penetration" ) == 0 || std::strcmp( gapType, "rate_penetration" ) == 0 ) {
      // check that g < gap_tol (interpenetration)
      EXPECT_LE( my_gap, gap_tol );
    } else if ( std::strcmp( gapType, "kinematic_separation" ) == 0 ||
                std::strcmp( gapType, "rate_separation" ) == 0 ) {
      // check that g > gap_tol (separation)
      EXPECT_GE( my_gap, gap_tol );
    } else {
      SLIC_ERROR( "compareGaps: invalid gapType. "
                  << "Acceptable types are 'kinematic_penetration', 'kinematic_separation', "
                  << "'rate_penetration' or 'rate_separation'." );
      ;
    }

    // check diffs
    real diff = std::abs( my_gap - gap );
    EXPECT_LE( diff, tol );

    ++cpID;
  }
}  // end compareGaps()

void checkMeshPenalties( tribol::CouplingScheme const* cs, const real penalty, const real tol, const char* penaltyType )
{
  tribol::IndexType const meshId1 = cs->getMeshId1();
  tribol::IndexType const meshId2 = cs->getMeshId2();

  tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
  tribol::MeshData& mesh1 = meshManager.GetMeshInstance( meshId1 );
  tribol::MeshData& mesh2 = meshManager.GetMeshInstance( meshId2 );

  if ( std::strcmp( penaltyType, "constant" ) == 0 ) {
    real penalty_diff_1 = std::abs( mesh1.m_elemData.m_penalty_stiffness - penalty );
    real penalty_diff_2 = std::abs( mesh2.m_elemData.m_penalty_stiffness - penalty );
    EXPECT_LE( penalty_diff_1, tol );
    EXPECT_LE( penalty_diff_2, tol );
  } else if ( std::strcmp( penaltyType, "face" ) == 0 ) {
    // no-op, the face-based penalty is checked in a call to tribol::update()
  } else if ( std::strcmp( penaltyType, "constant_rate" ) == 0 ) {
    real penalty_diff_1 = std::abs( mesh1.m_elemData.m_rate_penalty_stiffness - penalty );
    real penalty_diff_2 = std::abs( mesh2.m_elemData.m_rate_penalty_stiffness - penalty );
    EXPECT_LE( penalty_diff_1, tol );
    EXPECT_LE( penalty_diff_2, tol );
  } else if ( std::strcmp( penaltyType, "percent_rate" ) == 0 ) {
    real penalty1 = mesh1.m_elemData.m_rate_percent_stiffness * mesh1.m_elemData.m_penalty_stiffness;
    real penalty2 = mesh2.m_elemData.m_rate_percent_stiffness * mesh2.m_elemData.m_penalty_stiffness;
    real penalty_diff_1 = std::abs( penalty1 - penalty );
    real penalty_diff_2 = std::abs( penalty2 - penalty );
    EXPECT_LE( penalty_diff_1, tol );
    EXPECT_LE( penalty_diff_2, tol );
  } else {
    SLIC_ERROR( "checkMeshPenalties: invalid penaltyType. "
                << "only 'constant', 'face', 'constant_rate', or 'percent_rate' accepted. " );
  }

}  // end checkMeshPenalties()

void checkPressures( tribol::CouplingScheme const* cs, real pressure, const real tol,
                     const char* pressureType = "kinematic" )
{
  tribol::ContactPlaneManager& cpManager = tribol::ContactPlaneManager::getInstance();
  tribol::InterfacePairs const* const pairs = cs->getInterfacePairs();
  tribol::IndexType const numPairs = pairs->getNumPairs();

  int cpID = 0;
  for ( tribol::IndexType kp = 0; kp < numPairs; ++kp ) {
    tribol::InterfacePair pair = pairs->getInterfacePair( kp );

    if ( !pair.isContactCandidate ) {
      continue;
    }

    real my_pressure = 0.;
    if ( std::strcmp( pressureType, "rate" ) == 0 ) {
      my_pressure = cpManager.m_ratePressure[cpID];
    } else if ( std::strcmp( pressureType, "kinematic" ) == 0 ) {
      my_pressure = cpManager.m_pressure[cpID];
    } else {
      SLIC_ERROR( "checkPressures(): invalid pressure type. Supported types are "
                  << "'kinematic' or 'rate'." );
    }

    // check diffs
    real press_diff = std::abs( my_pressure - pressure );
    EXPECT_LE( press_diff, tol );

    ++cpID;
  }
}  // end checkPressures()

// problem specific routine to check the sense of the force. Note, this
// routine makes implicit use of the knowledge that the outward facing
// surface unit normals are in the +/- z-direction for mesh 1 and
// mesh 2, respectively. This is not a general routine for general
// mesh configurations.
void checkForceSense( tribol::CouplingScheme const* cs, bool isTied = false )
{
  tribol::IndexType const meshId1 = cs->getMeshId1();
  tribol::IndexType const meshId2 = cs->getMeshId2();

  tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
  tribol::MeshData& mesh1 = meshManager.GetMeshInstance( meshId1 );
  tribol::MeshData& mesh2 = meshManager.GetMeshInstance( meshId2 );

  for ( int i = 0; i < 2; ++i )  // loop over meshes
  {
    tribol::MeshData& mesh = ( i == 0 ) ? mesh1 : mesh2;

    // loop over faces and nodes
    for ( tribol::IndexType kf = 0; kf < mesh.m_numCells; ++kf ) {
      for ( tribol::IndexType a = 0; a < mesh.m_numNodesPerCell; ++a ) {
        int idx = mesh.m_numNodesPerCell * kf + a;
        int node_id = mesh.m_connectivity[idx];
        real force_mag = tribol::dotProd( mesh.m_forceX[node_id], mesh.m_forceY[node_id], mesh.m_forceZ[node_id],
                                          mesh.m_nX[kf], mesh.m_nY[kf], mesh.m_nZ[kf] );
        if ( !isTied ) {
          // <= catches interpenetration AND separation
          EXPECT_LE( force_mag, 0. );
        } else {
          // no-op, TIED is a special case where we support
          // all force 'senses' (i.e. tension and compression)
        }
      }
    }
  }
}  // end checkForceSense()

/*!
 * Test fixture class with some setup necessary to test
 * the COMMON_PLANE + PENALTY implementation
 */
class CommonPlaneTest : public ::testing::Test {
 public:
  tribol::TestMesh m_mesh;

 protected:
  void SetUp() override {}

  void TearDown() override { this->m_mesh.clear(); }

 protected:
};

TEST_F( CommonPlaneTest, penetration_gap_check )
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

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  parameters.dt = 1.e-3;
  parameters.penalty_ratio = false;
  parameters.const_penalty = 1.0;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY,
                                                                tribol::FRICTIONLESS, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = couplingSchemeManager.getCoupling( 0 );

  real gap = z_min2 - z_max1;

  compareGaps( couplingScheme, gap, 1.E-8, "kinematic_penetration" );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, separation_gap_check )
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

  // mesh bounding box with 0.1 separation gap
  real x_min1 = 0.;
  real y_min1 = 0.;
  real z_min1 = 0.;
  real x_max1 = 1.;
  real y_max1 = 1.;
  real z_max1 = 1.;

  real x_min2 = 0.;
  real y_min2 = 0.;
  real z_min2 = 1.1;
  real x_max2 = 1.;
  real y_max2 = 1.;
  real z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  parameters.penalty_ratio = false;
  parameters.const_penalty = 1.0;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY,
                                                                tribol::FRICTIONLESS, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = couplingSchemeManager.getCoupling( 0 );

  real gap = z_min2 - z_max1;

  compareGaps( couplingScheme, gap, 1.E-8, "kinematic_separation" );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, constant_penalty_check )
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

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  parameters.penalty_ratio = false;
  parameters.const_penalty = 0.75;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY,
                                                                tribol::FRICTIONLESS, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = couplingSchemeManager.getCoupling( 0 );

  // check mesh penalties
  checkMeshPenalties( couplingScheme, parameters.const_penalty, 1.E-8, "constant" );

  // check the pressures
  real gap = z_min2 - z_max1;
  real pressure = tribol::ComputePenaltyStiffnessPerArea( parameters.const_penalty, parameters.const_penalty ) * gap;
  checkPressures( couplingScheme, pressure, 1.E-8 );
  checkForceSense( couplingScheme );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, element_penalty_check )
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

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

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
  parameters.contact_pen_frac = 0.29;
  parameters.penalty_ratio = true;
  parameters.const_penalty = 0.75;
  parameters.dt = dt;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY,
                                                                tribol::FRICTIONLESS, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = couplingSchemeManager.getCoupling( 0 );

  checkMeshPenalties( couplingScheme, parameters.const_penalty, 1.E-8, "face" );

  /////////////////////////
  // check the pressures //
  /////////////////////////
  real gap = z_min2 - z_max1;

  // this uses the same face-springs-in-parallel calculation as the common plane + penalty method: K1/t_1 * K2/t_2 /
  // (K1/t_1 + K2/t_2)
  real pressure = ( bulk_mod1 / element_thickness1 * bulk_mod2 / element_thickness2 ) /
                  ( bulk_mod1 / element_thickness1 + bulk_mod2 / element_thickness2 ) * gap;
  checkPressures( couplingScheme, pressure, 1.E-8 );
  checkForceSense( couplingScheme );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, tied_contact_check )
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

  // mesh bounding box with 0.1 separation gap
  real x_min1 = 0.;
  real y_min1 = 0.;
  real z_min1 = 0.;
  real x_max1 = 1.;
  real y_max1 = 1.;
  real z_max1 = 1.;

  real x_min2 = 0.;
  real y_min2 = 0.;
  real z_min2 = 1.01;
  real x_max2 = 1.;
  real y_max2 = 1.;
  real z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  parameters.penalty_ratio = false;
  parameters.const_penalty = 0.25;

  int test_mesh_update_err =
      this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, tribol::TIED, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = couplingSchemeManager.getCoupling( 0 );

  // check the pressures
  real gap = z_min2 - z_max1;
  real pressure = tribol::ComputePenaltyStiffnessPerArea( parameters.const_penalty, parameters.const_penalty ) * gap;
  checkPressures( couplingScheme, pressure, 1.E-8 );
  checkForceSense( couplingScheme, true );

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
