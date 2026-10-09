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
#include "tribol/physics/CommonPlane.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/geom/CompGeom.hpp"

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

void compareGaps( tribol::CouplingScheme const* cs, RealT gap, const RealT tol, const char* gapType )
{
  tribol::IndexT const numPairs = cs->getNumActivePairs();
  // TODO: get rid of the const cast if we can
  const auto cs_view = const_cast<tribol::CouplingScheme*>( cs )->getView();

  for ( tribol::IndexT cpID = 0; cpID < numPairs; ++cpID ) {
    auto& plane = cs->getCompGeom().getCommonPlane( cpID );

    RealT my_gap = 0.;
    if ( std::strcmp( gapType, "kinematic_penetration" ) == 0 || std::strcmp( gapType, "kinematic_separation" ) == 0 ) {
      my_gap = plane.m_gap;
    } else {
      my_gap = plane.m_velGap;
    }

    RealT gap_tol = cs_view.getGapTol( plane.getCpElementId1(), plane.getCpElementId2() );

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
    EXPECT_NEAR( my_gap, gap, tol );
  }
}  // end compareGaps()

void checkMeshPenalties( tribol::CouplingScheme const* cs, const RealT penalty, const RealT tol,
                         const char* penaltyType )
{
  tribol::IndexT const meshId1 = cs->getMeshId1();
  tribol::IndexT const meshId2 = cs->getMeshId2();

  tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
  tribol::MeshData& mesh1 = meshManager.at( meshId1 );
  tribol::MeshData& mesh2 = meshManager.at( meshId2 );

  if ( std::strcmp( penaltyType, "constant" ) == 0 ) {
    RealT penalty_diff_1 = tribol::abs_val_diff( mesh1.getElementData().m_penalty_stiffness, penalty );
    RealT penalty_diff_2 = tribol::abs_val_diff( mesh2.getElementData().m_penalty_stiffness, penalty );
    EXPECT_LE( penalty_diff_1, tol );
    EXPECT_LE( penalty_diff_2, tol );
  } else if ( std::strcmp( penaltyType, "face" ) == 0 ) {
    // no-op, the face-based penalty is checked in a call to tribol::update()
  } else if ( std::strcmp( penaltyType, "constant_rate" ) == 0 ) {
    RealT penalty_diff_1 = tribol::abs_val_diff( mesh1.getElementData().m_rate_penalty_stiffness, penalty );
    RealT penalty_diff_2 = tribol::abs_val_diff( mesh2.getElementData().m_rate_penalty_stiffness, penalty );
    EXPECT_LE( penalty_diff_1, tol );
    EXPECT_LE( penalty_diff_2, tol );
  } else if ( std::strcmp( penaltyType, "percent_rate" ) == 0 ) {
    RealT penalty1 = mesh1.getElementData().m_rate_percent_stiffness * mesh1.getElementData().m_penalty_stiffness;
    RealT penalty2 = mesh2.getElementData().m_rate_percent_stiffness * mesh2.getElementData().m_penalty_stiffness;
    RealT penalty_diff_1 = tribol::abs_val_diff( penalty1, penalty );
    RealT penalty_diff_2 = tribol::abs_val_diff( penalty2, penalty );
    EXPECT_LE( penalty_diff_1, tol );
    EXPECT_LE( penalty_diff_2, tol );
  } else {
    SLIC_ERROR( "checkMeshPenalties: invalid penaltyType. "
                << "only 'constant', 'face', 'constant_rate', or 'percent_rate' accepted. " );
  }

}  // end checkMeshPenalties()

void checkPressures( tribol::CouplingScheme const* cs, RealT pressure, const RealT tol,
                     const char* pressureType = "kinematic" )
{
  tribol::IndexT const numPairs = cs->getNumActivePairs();

  for ( tribol::IndexT cpID = 0; cpID < numPairs; ++cpID ) {
    auto& plane = cs->getCompGeom().getCommonPlane( cpID );

    RealT my_pressure = 0.;
    if ( std::strcmp( pressureType, "rate" ) == 0 ) {
      my_pressure = plane.m_ratePressure;
    } else if ( std::strcmp( pressureType, "kinematic" ) == 0 ) {
      my_pressure = plane.m_pressure;
    } else {
      SLIC_ERROR( "checkPressures(): invalid pressure type. Supported types are " << "'kinematic' or 'rate'." );
    }

    // check diffs
    EXPECT_NEAR( my_pressure, pressure, tol );
  }
}  // end checkPressures()

// problem specific routine to check the sense of the force. Note, this
// routine makes implicit use of the knowledge that the outward facing
// surface unit normals are in the +/- z-direction for mesh 1 and
// mesh 2, respectively. This is not a general routine for general
// mesh configurations.
void checkForceSense( tribol::CouplingScheme const* cs, bool isTied = false )
{
  // TODO: get rid of const cast (if we can)
  const auto mesh1 = const_cast<tribol::CouplingScheme*>( cs )->getMesh1().getView();
  const auto mesh2 = const_cast<tribol::CouplingScheme*>( cs )->getMesh2().getView();

  for ( int i = 0; i < 2; ++i )  // loop over meshes
  {
    auto& mesh = ( i == 0 ) ? mesh1 : mesh2;

    // loop over faces and nodes
    for ( tribol::IndexT kf = 0; kf < mesh.numberOfElements(); ++kf ) {
      for ( tribol::IndexT a = 0; a < mesh.numberOfNodesPerElement(); ++a ) {
        int node_id = mesh.getGlobalNodeId( kf, a );
        RealT force_mag = 0.;
        if ( mesh1.spatialDimension() == 3 ) {
          force_mag = tribol::dotProd( mesh.getResponse()[0][node_id], mesh.getResponse()[1][node_id],
                                       mesh.getResponse()[2][node_id], mesh.getElementNormals()[0][kf],
                                       mesh.getElementNormals()[1][kf], mesh.getElementNormals()[2][kf] );
        } else {
          force_mag = tribol::dotProd( mesh.getResponse()[0][node_id], mesh.getResponse()[1][node_id], 0.,
                                       mesh.getElementNormals()[0][kf], mesh.getElementNormals()[1][kf], 0. );
        }
        if ( !isTied ) {
          // <= catches interpenetration AND separation
          // EXPECT_LE( force_mag, -1.e-6 );
          EXPECT_LE( force_mag, 0. );
        } else {
          // no-op, TIED_NORMAL is a special case where we support
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
};

struct WarpedQuadForceResult {
  int err{ -1 };
  RealT gap{ 0. };
  RealT total_abs_force{ 0. };
  RealT total_force_z{ 0. };
};

WarpedQuadForceResult runWarpedQuadForceCase( tribol::PolyInteg rule )
{
  constexpr int numVerts = 4;

  // Mesh 1 is a planar unit quad at z=0. Mesh 2 fully overlaps it in x-y, but is warped
  // so that only one corner penetrates the plane.
  RealT x1[numVerts] = { 0.0, 1.0, 1.0, 0.0 };
  RealT y1[numVerts] = { 0.0, 0.0, 1.0, 1.0 };
  RealT z1[numVerts] = { 0.0, 0.0, 0.0, 0.0 };

  RealT x2[numVerts] = { 0.0, 0.0, 1.0, 1.0 };
  RealT y2[numVerts] = { 0.0, 1.0, 1.0, 0.0 };
  RealT z2[numVerts] = { -0.20, 1.00, 1.00, 1.00 };

  tribol::IndexT conn1[numVerts] = { 0, 1, 2, 3 };
  tribol::IndexT conn2[numVerts] = { 0, 1, 2, 3 };

  tribol::registerMesh( 0, 1, numVerts, &conn1[0], (int)( tribol::LINEAR_QUAD ), &x1[0], &y1[0], &z1[0],
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, 1, numVerts, &conn2[0], (int)( tribol::LINEAR_QUAD ), &x2[0], &y2[0], &z2[0],
                        tribol::MemorySpace::Host );

  RealT fx1[numVerts] = { 0., 0., 0., 0. };
  RealT fy1[numVerts] = { 0., 0., 0., 0. };
  RealT fz1[numVerts] = { 0., 0., 0., 0. };
  RealT fx2[numVerts] = { 0., 0., 0., 0. };
  RealT fy2[numVerts] = { 0., 0., 0., 0. };
  RealT fz2[numVerts] = { 0., 0., 0., 0. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], &fz1[0] );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], &fz2[0] );

  tribol::setKinematicConstantPenalty( 0, 1.0 );
  tribol::setKinematicConstantPenalty( 1, 1.0 );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  constexpr tribol::IndexT couplingSchemeId = 0;
  constexpr int quadratureOrder = 3;
  tribol::setPenaltyOptions( couplingSchemeId, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setCommonPlaneIntegrationOptions( couplingSchemeId, rule, quadratureOrder );
  tribol::setContactAreaFrac( couplingSchemeId, 1.e-12 );

  WarpedQuadForceResult result;
  RealT dt = 1.;
  result.err = tribol::update( 1, 1., dt );

  auto& cs = tribol::CouplingSchemeManager::getInstance().at( 0 );
  EXPECT_EQ( 1, cs.getNumActivePairs() );
  result.gap = cs.getCompGeom().getCommonPlane( 0 ).m_gap;

  for ( int i = 0; i < numVerts; ++i ) {
    result.total_abs_force += std::abs( fx1[i] ) + std::abs( fy1[i] ) + std::abs( fz1[i] );
    result.total_abs_force += std::abs( fx2[i] ) + std::abs( fy2[i] ) + std::abs( fz2[i] );
    result.total_force_z += fz1[i] + fz2[i];
  }

  tribol::finalize();
  return result;
}

struct EdgeLocalContactForceResult {
  int err{ -1 };
  tribol::IndexT num_active_pairs{ 0 };
  RealT gap{ 0. };
  RealT total_abs_force{ 0. };
  RealT mesh1_node_force_norm[2] = { 0., 0. };
};

EdgeLocalContactForceResult runEdgeLocalContactCase( tribol::PolyInteg rule )
{
  constexpr int numVerts = 2;

  // Mesh 1 is a horizontal unit edge. Mesh 2 fully overlaps it in x, but is tilted so the
  // penetration varies from 0.2 at node 0 to 0.05 at node 1.
  RealT x1[numVerts] = { 1.0, 0.0 };
  RealT y1[numVerts] = { 0.0, 0.0 };

  RealT x2[numVerts] = { 0.0, 1.0 };
  RealT y2[numVerts] = { -0.2, -0.05 };

  tribol::IndexT conn1[numVerts] = { 0, 1 };
  tribol::IndexT conn2[numVerts] = { 0, 1 };

  tribol::registerMesh( 0, 1, numVerts, &conn1[0], (int)( tribol::LINEAR_EDGE ), &x1[0], &y1[0], nullptr,
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, 1, numVerts, &conn2[0], (int)( tribol::LINEAR_EDGE ), &x2[0], &y2[0], nullptr,
                        tribol::MemorySpace::Host );

  RealT fx1[numVerts] = { 0., 0. };
  RealT fy1[numVerts] = { 0., 0. };
  RealT fx2[numVerts] = { 0., 0. };
  RealT fy2[numVerts] = { 0., 0. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], nullptr );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], nullptr );

  tribol::setKinematicConstantPenalty( 0, 1. );
  tribol::setKinematicConstantPenalty( 1, 1. );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  constexpr tribol::IndexT couplingSchemeId = 0;
  constexpr int quadratureOrder = 4;
  tribol::setPenaltyOptions( couplingSchemeId, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setCommonPlaneIntegrationOptions( couplingSchemeId, rule, quadratureOrder );
  tribol::setContactAreaFrac( couplingSchemeId, 1.e-12 );

  EdgeLocalContactForceResult result;
  RealT dt = 1.;
  result.err = tribol::update( 1, 1., dt );

  tribol::CouplingScheme* couplingScheme = &tribol::CouplingSchemeManager::getInstance().at( 0 );
  result.num_active_pairs = couplingScheme->getNumActivePairs();
  if ( result.num_active_pairs > 0 ) {
    result.gap = couplingScheme->getCompGeom().getCommonPlane( 0 ).m_gap;
  }

  for ( int i = 0; i < numVerts; ++i ) {
    result.total_abs_force += std::abs( fx1[i] ) + std::abs( fy1[i] ) + std::abs( fx2[i] ) + std::abs( fy2[i] );
    result.mesh1_node_force_norm[i] = tribol::magnitude( fx1[i], fy1[i] );
  }

  tribol::finalize();

  return result;
}

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
  parameters.dt = 1.e-3;
  parameters.penalty_ratio = false;
  parameters.const_penalty = 1.0;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  RealT gap = z_min2 - z_max1;

  compareGaps( couplingScheme, gap, 1.E-8, "kinematic_penetration" );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, multipoint_quad_execution )
{
  // Planar quad meshes with full face overlap and uniform 0.1 interpenetration. Multi-point
  // integration should reproduce the same gap and valid force sense as the single-point rule.
  this->m_mesh.mortarMeshId = 0;
  this->m_mesh.nonmortarMeshId = 1;

  const int nElems = 2;
  const RealT x_min1 = 0.;
  const RealT y_min1 = 0.;
  const RealT z_min1 = 0.;
  const RealT x_max1 = 1.;
  const RealT y_max1 = 1.;
  const RealT z_max1 = 1.05;

  const RealT x_min2 = 0.;
  const RealT y_min2 = 0.;
  const RealT z_min2 = 0.95;
  const RealT x_max2 = 1.;
  const RealT y_max2 = 1.;
  const RealT z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElems, nElems, nElems, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1, nElems,
                                    nElems, nElems, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0., 0. );

  tribol::TestControlParameters parameters;
  parameters.dt = 1.e-3;
  parameters.const_penalty = 1.0;
  parameters.common_plane_rule = tribol::MULTI_POINT;
  parameters.common_plane_quadrature_order = 3;

  int err = this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS,
                                               tribol::NO_CASE, false, parameters );

  EXPECT_EQ( err, 0 );

  tribol::CouplingScheme* couplingScheme = &tribol::CouplingSchemeManager::getInstance().at( 0 );
  compareGaps( couplingScheme, z_min2 - z_max1, 1.E-8, "kinematic_penetration" );
  checkForceSense( couplingScheme );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, multipoint_triangle_execution )
{
  // Planar tet-surface triangle meshes with full overlap and uniform 0.1 interpenetration.
  // This exercises triangle overlap integration in the CommonPlane penalty path.
  this->m_mesh.mortarMeshId = 0;
  this->m_mesh.nonmortarMeshId = 1;

  const int nElems = 2;
  const RealT x_min1 = 0.;
  const RealT y_min1 = 0.;
  const RealT z_min1 = 0.;
  const RealT x_max1 = 1.;
  const RealT y_max1 = 1.;
  const RealT z_max1 = 1.05;

  const RealT x_min2 = 0.;
  const RealT y_min2 = 0.;
  const RealT z_min2 = 0.95;
  const RealT x_max2 = 1.;
  const RealT y_max2 = 1.;
  const RealT z_max2 = 2.;

  this->m_mesh.setupContactMeshTet( nElems, nElems, nElems, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1, nElems,
                                    nElems, nElems, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0., 0. );

  tribol::TestControlParameters parameters;
  parameters.dt = 1.e-3;
  parameters.const_penalty = 1.0;
  parameters.common_plane_rule = tribol::MULTI_POINT;
  parameters.common_plane_quadrature_order = 3;

  int err = this->m_mesh.tribolSetupAndUpdate( tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS,
                                               tribol::NO_CASE, false, parameters );

  EXPECT_EQ( err, 0 );

  tribol::CouplingScheme* couplingScheme = &tribol::CouplingSchemeManager::getInstance().at( 0 );
  compareGaps( couplingScheme, z_min2 - z_max1, 1.E-8, "kinematic_penetration" );
  checkForceSense( couplingScheme );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, multipoint_warped_quad_local_contact )
{
  // Single-point integration at the overlap centroid misses the local warped-quad penetration.
  // Full triangle-decomposition quadrature samples the penetrating region and generates force.
  const auto single_point = runWarpedQuadForceCase( tribol::SINGLE_POINT );
  const auto multi_point = runWarpedQuadForceCase( tribol::MULTI_POINT );

  EXPECT_EQ( single_point.err, 0 );
  EXPECT_EQ( multi_point.err, 0 );

  constexpr RealT expected_gap = 0.5852486869304208;
  EXPECT_NEAR( single_point.gap, expected_gap, 1.e-12 );
  EXPECT_NEAR( multi_point.gap, expected_gap, 1.e-12 );

  EXPECT_NEAR( single_point.total_abs_force, 0., 1.e-12 );
  EXPECT_NEAR( multi_point.total_abs_force, 8.20970073925438e-4, 1.e-12 );
  EXPECT_NEAR( multi_point.total_force_z, 0., 1.e-12 );
}

TEST_F( CommonPlaneTest, multipoint_edge_local_contact )
{
  // Both rules detect the tilted full-overlap edge contact, but multi-point integration resolves
  // the nonuniform penetration and therefore produces a larger nodal force imbalance.
  const auto single_point = runEdgeLocalContactCase( tribol::SINGLE_POINT );
  const auto multi_point = runEdgeLocalContactCase( tribol::MULTI_POINT );

  EXPECT_EQ( single_point.err, 0 );
  EXPECT_EQ( multi_point.err, 0 );

  EXPECT_EQ( single_point.num_active_pairs, 1 );
  EXPECT_EQ( multi_point.num_active_pairs, 1 );

  constexpr RealT expected_gap = -0.12423774246760939;
  EXPECT_NEAR( single_point.gap, expected_gap, 1.e-12 );
  EXPECT_NEAR( multi_point.gap, expected_gap, 1.e-12 );

  EXPECT_NEAR( single_point.total_abs_force, 0.13126963259866961, 1.e-12 );
  EXPECT_NEAR( multi_point.total_abs_force, 0.13227012255210607, 1.e-12 );

  // Imbalance is the difference between the force norms at the two nodes of mesh 1.
  const RealT single_point_imbalance =
      std::abs( single_point.mesh1_node_force_norm[0] - single_point.mesh1_node_force_norm[1] );
  const RealT multi_point_imbalance =
      std::abs( multi_point.mesh1_node_force_norm[0] - multi_point.mesh1_node_force_norm[1] );

  EXPECT_NEAR( single_point_imbalance, 3.37547013399012e-4, 1.e-12 );
  EXPECT_NEAR( multi_point_imbalance, 1.2454070813893655e-2, 1.e-12 );
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
  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.;

  RealT x_min2 = 0.;
  RealT y_min2 = 0.;
  RealT z_min2 = 1.1;
  RealT x_max2 = 1.;
  RealT y_max2 = 1.;
  RealT z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  parameters.penalty_ratio = false;
  parameters.const_penalty = 1.0;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  RealT gap = z_min2 - z_max1;

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
  parameters.penalty_ratio = false;
  parameters.const_penalty = 0.75;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  // check mesh penalties
  checkMeshPenalties( couplingScheme, parameters.const_penalty, 1.E-8, "constant" );

  // check the pressures
  RealT gap = z_min2 - z_max1;
  RealT pressure = tribol::ComputePenaltyStiffnessPerArea( parameters.const_penalty, parameters.const_penalty ) * gap;
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

  // compute element thickness for each block
  RealT element_thickness1 = ( z_max1 - z_min1 ) / nElemsZM;
  RealT element_thickness2 = ( z_max2 - z_min2 ) / nElemsZS;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  RealT dt = 1.e-3;
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

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::NO_CASE, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  checkMeshPenalties( couplingScheme, parameters.const_penalty, 1.E-8, "face" );

  /////////////////////////
  // check the pressures //
  /////////////////////////
  RealT gap = z_min2 - z_max1;

  // this uses the same face-springs-in-parallel calculation as the common plane + penalty method: K1/t_1 * K2/t_2 /
  // (K1/t_1 + K2/t_2)
  RealT pressure = ( bulk_mod1 / element_thickness1 * bulk_mod2 / element_thickness2 ) /
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
  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.;

  RealT x_min2 = 0.;
  RealT y_min2 = 0.;
  RealT z_min2 = 1.01;
  RealT x_max2 = 1.;
  RealT y_max2 = 1.;
  RealT z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  parameters.penalty_ratio = false;
  parameters.const_penalty = 0.25;

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate(
      tribol::COMMON_PLANE, tribol::PENALTY, tribol::FRICTIONLESS, tribol::TIED_NORMAL, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  // check the pressures
  RealT gap = z_min2 - z_max1;
  RealT pressure = tribol::ComputePenaltyStiffnessPerArea( parameters.const_penalty, parameters.const_penalty ) * gap;
  checkPressures( couplingScheme, pressure, 1.E-8 );
  checkForceSense( couplingScheme, true );

  tribol::finalize();
}

TEST_F( CommonPlaneTest, common_plane_2d_interpen_check )
{
  // This test checks the forces and gaps of an interpen overlap configuration in 2D
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

  checkForceSense( couplingScheme );

  // compute the point at which edge 2 intersects edge 1.
  // then compute the length and height of the interpen portion of edge 2
  RealT intercept_point = 0.5 * ( x1[0] - x1[1] ) + x_shift;
  RealT length = intercept_point - x2[0];
  RealT height = y1[0] - y2[0];

  // compute the angle between the interpen portions of edge 1 and 2
  RealT theta = std::atan( height / length );

  // compute the projected area of overlap
  RealT hyp = length / std::cos( theta );
  RealT half_theta = 0.5 * theta;
  RealT computed_area = hyp * std::cos( half_theta );

  // compute and check the gap at the overlap area centroid
  RealT half_area = 0.5 * computed_area;
  RealT half_gap = half_area * std::tan( half_theta );
  RealT gap = -2. * half_gap;
  compareGaps( couplingScheme, gap, 1.E-8, "kinematic_penetration" );
}

TEST_F( CommonPlaneTest, common_plane_viscous_tangential_2d )
{
  // This test has two edges with initial full interpen and tangential velocity,
  // which will trigger a viscous force term
  constexpr int numVerts = 2;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];

  x1[0] = 1.0;
  x1[1] = 0.0;
  y1[0] = 0.0;
  y1[1] = 0.0;

  RealT gap = 0.01;
  x2[0] = 0.0;
  x2[1] = 1.0;
  y2[0] = -gap;
  y2[1] = -gap;

  tribol::IndexT conn1[numVerts] = { 0, 1 };
  tribol::IndexT conn2[numVerts] = { 0, 1 };

  tribol::registerMesh( 0, 1, numVerts, &conn1[0], (int)( tribol::LINEAR_EDGE ), &x1[0], &y1[0], nullptr,
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, 1, numVerts, &conn2[0], (int)( tribol::LINEAR_EDGE ), &x2[0], &y2[0], nullptr,
                        tribol::MemorySpace::Host );

  RealT fx1[numVerts] = { 0., 0. };
  RealT fy1[numVerts] = { 0., 0. };
  RealT fx2[numVerts] = { 0., 0. };
  RealT fy2[numVerts] = { 0., 0. };

  // set nodal velocities in <1,1> and <-1,-1> directions
  RealT vx1[numVerts] = { 1., 1. };
  RealT vy1[numVerts] = { 1., 1. };
  RealT vx2[numVerts] = { -1., -1. };
  RealT vy2[numVerts] = { -1., -1. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], nullptr );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], nullptr );

  tribol::registerNodalVelocities( 0, &vx1[0], &vy1[0], nullptr );
  tribol::registerNodalVelocities( 1, &vx2[0], &vy2[0], nullptr );

  tribol::setKinematicConstantPenalty( 0, 1. );
  tribol::setKinematicConstantPenalty( 1, 1. );

  RealT visc_coeff = 0.5;
  tribol::registerRealElementField( 0, tribol::VISCOUS_DAMPING_COEFF, &visc_coeff );
  tribol::registerRealElementField( 1, tribol::VISCOUS_DAMPING_COEFF, &visc_coeff );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::VISCOUS_TANGENTIAL, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setContactAreaFrac( 0, 1.e-12 );

  RealT dt = 1.;
  int update_err = tribol::update( 1, 1., dt );

  EXPECT_EQ( update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  // check forces. The in-plane tangential force is the relative tangential velocity gap multiplied
  // by the viscous coefficient then evenly distributed to the two nodes, while the normal force
  // corresponds to two springs in parallel with the specified constant penalty stiffness, then multiplied
  // by the gap and then divided amongst the edge nodes
  RealT force_y = 0.5 * gap / numVerts;
  RealT force_x = visc_coeff * ( vx1[0] - vx2[0] ) / numVerts;
  // The analytic force above assumes the nominal edge directions, while the implementation uses
  // the CommonPlane normal/tangent for the slightly tilted contact pair.
  constexpr RealT tol = 5.e-5;
  for ( int i = 0; i < numVerts; ++i ) {
    EXPECT_NEAR( fx1[i], -force_x, tol );
    EXPECT_NEAR( fy1[i], -force_y, tol );
    EXPECT_NEAR( fx2[i], force_x, tol );
    EXPECT_NEAR( fy2[i], force_y, tol );
  }
}

TEST_F( CommonPlaneTest, common_plane_viscous_tangential_3d )
{
  // this test has two faces with some initial full interpen and a tangential velocity
  // that will trigger a viscous force term.
  constexpr int numVerts = 4;

  RealT x1[numVerts];
  RealT y1[numVerts];
  RealT z1[numVerts];
  RealT x2[numVerts];
  RealT y2[numVerts];
  RealT z2[numVerts];

  // first face's coords
  x1[0] = 0.0;
  y1[0] = 0.0;
  z1[0] = 0.0;

  x1[1] = 1.0;
  y1[1] = 0.0;
  z1[1] = 0.0;

  x1[2] = 1.0;
  y1[2] = 1.0;
  z1[2] = 0.0;

  x1[3] = 0.0;
  y1[3] = 1.0;
  z1[3] = 0.0;

  // second faces coords
  RealT gap = 0.01;
  x2[0] = 0.0;
  y2[0] = 0.0;
  z2[0] = -gap;

  x2[1] = 0.0;
  y2[1] = 1.0;
  z2[1] = -gap;

  x2[2] = 1.0;
  y2[2] = 1.0;
  z2[2] = -gap;

  x2[3] = 1.0;
  y2[3] = 0.0;
  z2[3] = -gap;

  tribol::IndexT conn1[numVerts] = { 0, 1, 2, 3 };
  tribol::IndexT conn2[numVerts] = { 0, 1, 2, 3 };

  tribol::registerMesh( 0, 1, numVerts, &conn1[0], (int)( tribol::LINEAR_QUAD ), &x1[0], &y1[0], &z1[0],
                        tribol::MemorySpace::Host );
  tribol::registerMesh( 1, 1, numVerts, &conn2[0], (int)( tribol::LINEAR_QUAD ), &x2[0], &y2[0], &z2[0],
                        tribol::MemorySpace::Host );

  RealT fx1[numVerts] = { 0., 0., 0., 0. };
  RealT fy1[numVerts] = { 0., 0., 0., 0. };
  RealT fz1[numVerts] = { 0., 0., 0., 0. };

  RealT fx2[numVerts] = { 0., 0., 0., 0. };
  RealT fy2[numVerts] = { 0., 0., 0., 0. };
  RealT fz2[numVerts] = { 0., 0., 0., 0. };

  // set nodal velocities in <2,2,2> and <-2,-2,-2> directions
  RealT vx1[numVerts] = { 2., 2., 2., 2. };
  RealT vy1[numVerts] = { 2., 2., 2., 2. };
  RealT vz1[numVerts] = { 2., 2., 2., 2. };
  RealT vx2[numVerts] = { -2., -2., -2., -2. };
  RealT vy2[numVerts] = { -2., -2., -2., -2. };
  RealT vz2[numVerts] = { -2., -2., -2., -2. };

  tribol::registerNodalResponse( 0, &fx1[0], &fy1[0], &fz1[0] );
  tribol::registerNodalResponse( 1, &fx2[0], &fy2[0], &fz2[0] );

  tribol::registerNodalVelocities( 0, &vx1[0], &vy1[0], &vz1[0] );
  tribol::registerNodalVelocities( 1, &vx2[0], &vy2[0], &vz2[0] );

  tribol::setKinematicConstantPenalty( 0, 1. );
  tribol::setKinematicConstantPenalty( 1, 1. );

  RealT visc_coeff = 0.5;
  tribol::registerRealElementField( 0, tribol::VISCOUS_DAMPING_COEFF, &visc_coeff );
  tribol::registerRealElementField( 1, tribol::VISCOUS_DAMPING_COEFF, &visc_coeff );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::VISCOUS_TANGENTIAL, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setContactAreaFrac( 0, 1.e-12 );

  RealT dt = 1.;
  int update_err = tribol::update( 1, 1., dt );

  EXPECT_EQ( update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );

  EXPECT_EQ( 1, couplingScheme->getNumActivePairs() );

  // check forces. The in plane force will be the relative velocity multiplied by the
  // viscous coefficient then distributed amongst the nodes, while the normal force
  // is the two springs in parallel times the gap and then distributed to the nodes.
  RealT force_z = 0.5 * gap / numVerts;
  RealT force_x_and_y = visc_coeff * ( vx1[0] - vx2[0] ) / numVerts;
  for ( int i = 0; i < numVerts; ++i ) {
    EXPECT_NEAR( fx1[i], -force_x_and_y, 1.e-10 );
    EXPECT_NEAR( fy1[i], -force_x_and_y, 1.e-10 );
    EXPECT_NEAR( fz1[i], -force_z, 1.e-10 );

    EXPECT_NEAR( fx2[i], force_x_and_y, 1.e-10 );
    EXPECT_NEAR( fy2[i], force_x_and_y, 1.e-10 );
    EXPECT_NEAR( fz2[i], force_z, 1.e-10 );
  }
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
