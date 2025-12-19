// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/interface/tribol.hpp"
#include "tribol/utils/TestUtils.hpp"
#include "tribol/utils/Math.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/CouplingScheme.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// gtest includes
#include "gtest/gtest.h"

using RealT = tribol::RealT;

/*!
 * Test fixture class to test valid coupling schemes.
 * These tests only call the coupling scheme constructor
 * and then the initialization scheme, which tests
 * for valid coupling schemes. These tests are designed
 * to check for valid and invalid schemes.
 *
 * NOTE: tests will need to be update/removed as more
 *       capabilities come online
 */
class CouplingSchemeTest : public ::testing::Test {
 public:
  int m_numCells;
  int m_lengthNodalData;
  tribol::IndexT* m_connectivity{ nullptr };
  int m_elementType;
  RealT* m_x{ nullptr };
  RealT* m_y{ nullptr };
  RealT* m_z{ nullptr };
  RealT* m_fx{ nullptr };
  RealT* m_fy{ nullptr };
  RealT* m_fz{ nullptr };

 protected:
  void SetUp() override {}

  void TearDown() override { clear(); }

  void registerDummy2DMesh( int mesh_id, int numCells = 1, bool set_response = true )
  {
    // Single element meshes are usesd in these tests out of
    // simplicity.
    m_numCells = ( numCells >= 1 ) ? 1 : 0;
    m_lengthNodalData = 4;                         // Quad 4
    m_elementType = (int)( tribol::LINEAR_EDGE );  // 1D edge for 2D mesh

    if ( m_numCells > 0 ) {
      m_connectivity = new tribol::IndexT[m_lengthNodalData];
      m_x = new RealT[m_lengthNodalData];
      m_y = new RealT[m_lengthNodalData];
      if ( set_response ) {
        m_fx = new RealT[m_lengthNodalData];
        m_fy = new RealT[m_lengthNodalData];
      }

      for ( int i = 0; i < m_lengthNodalData; ++i ) {
        m_connectivity[i] = i;
      }
      // unit square
      m_x[0] = 0.;
      m_y[0] = 0.;
      m_x[1] = 1.;
      m_y[1] = 0.;
      m_x[2] = 1.;
      m_y[2] = 1.;
      m_x[3] = 0.;
      m_y[3] = 1.;

      if ( set_response ) {
        for ( int i = 0; i < m_lengthNodalData; ++i ) {
          m_fx[i] = 0;
          m_fy[i] = 0;
        }
      }
    }

    tribol::registerMesh( mesh_id, m_numCells, m_lengthNodalData, m_connectivity, m_elementType, m_x, m_y, m_z,
                          tribol::MemorySpace::Host );

    tribol::registerNodalResponse( mesh_id, m_fx, m_fy, m_fz );
  }

  void registerDummy3DMesh( int mesh_id, int numCells = 1, bool set_response = true )
  {
    // Single element meshes are usesd in these tests out of
    // simplicity.
    m_numCells = ( numCells >= 1 ) ? 1 : 0;
    m_lengthNodalData = 8;                         // Hex 8
    m_elementType = (int)( tribol::LINEAR_QUAD );  // 2D face in 3D mesh

    if ( m_numCells > 0 ) {
      m_connectivity = new tribol::IndexT[m_lengthNodalData];
      m_x = new RealT[m_lengthNodalData];
      m_y = new RealT[m_lengthNodalData];
      m_z = new RealT[m_lengthNodalData];
      if ( set_response ) {
        m_fx = new RealT[m_lengthNodalData];
        m_fy = new RealT[m_lengthNodalData];
        m_fz = new RealT[m_lengthNodalData];
      }

      for ( int i = 0; i < m_lengthNodalData; ++i ) {
        m_connectivity[i] = i;
      }
      // unit cube
      m_x[0] = 0.;
      m_y[0] = 0.;
      m_z[0] = 0.;
      m_x[1] = 1.;
      m_y[1] = 0.;
      m_z[1] = 0.;
      m_x[2] = 1.;
      m_y[2] = 1.;
      m_z[2] = 0.;
      m_x[3] = 0.;
      m_y[3] = 1.;
      m_z[3] = 0.;
      m_x[4] = 0.;
      m_y[4] = 0.;
      m_z[4] = 1.;
      m_x[5] = 1.;
      m_y[5] = 0.;
      m_z[5] = 1.;
      m_x[6] = 1.;
      m_y[6] = 1.;
      m_z[6] = 1.;
      m_x[7] = 0.;
      m_y[7] = 1.;
      m_z[7] = 1.;

      if ( set_response ) {
        for ( int i = 0; i < m_lengthNodalData; ++i ) {
          m_fx[i] = 0;
          m_fy[i] = 0;
          m_fz[i] = 0;
        }
      }
    }

    tribol::registerMesh( mesh_id, m_numCells, m_lengthNodalData, m_connectivity, m_elementType, m_x, m_y, m_z,
                          tribol::MemorySpace::Host );

    tribol::registerNodalResponse( mesh_id, m_fx, m_fy, m_fz );
  }

  void clear()
  {
    if ( m_connectivity != nullptr ) {
      delete[] m_connectivity;
      m_connectivity = nullptr;
    }
    if ( m_x != nullptr ) {
      delete[] m_x;
      m_x = nullptr;
    }
    if ( m_y != nullptr ) {
      delete[] m_y;
      m_y = nullptr;
    }
    if ( m_z != nullptr ) {
      delete[] m_z;
      m_z = nullptr;
    }
    if ( m_fx != nullptr ) {
      delete[] m_fx;
      m_fx = nullptr;
    }
    if ( m_fy != nullptr ) {
      delete[] m_fy;
      m_fy = nullptr;
    }
    if ( m_fz != nullptr ) {
      delete[] m_fz;
      m_fz = nullptr;
    }
  }
};

TEST_F( CouplingSchemeTest, single_mortar_2D )
{
  // expect the coupling scheme to fail because 2D
  // is not yet implemented
  registerDummy2DMesh( 0 );
  registerDummy2DMesh( 1 );

  // register dummy nodal fields so error doesn't return from field
  // registration
  tribol::Array1D<RealT> gaps( this->m_lengthNodalData );
  tribol::Array1D<RealT> pressures( this->m_lengthNodalData );

  tribol::registerMortarGaps( 1, &gaps[0] );
  tribol::registerMortarPressures( 1, &pressures[0] );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::SINGLE_MORTAR,
                                  tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setLagrangeMultiplierOptions( 0, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN,
                                        tribol::SparseMode::MFEM_LINKED_LIST );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, aligned_mortar_2D )
{
  registerDummy2DMesh( 0 );
  registerDummy2DMesh( 1 );

  // register dummy nodal fields so error doesn't return from field
  // registration
  tribol::Array1D<RealT> gaps( this->m_lengthNodalData );
  tribol::Array1D<RealT> pressures( this->m_lengthNodalData );

  tribol::registerMortarGaps( 1, &gaps[0] );
  tribol::registerMortarPressures( 1, &pressures[0] );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::ALIGNED_MORTAR,
                                  tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setLagrangeMultiplierOptions( 0, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN,
                                        tribol::SparseMode::MFEM_LINKED_LIST );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, mortar_weights_2D )
{
  registerDummy2DMesh( 0 );
  registerDummy2DMesh( 1 );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::MORTAR_WEIGHTS,
                                  tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setLagrangeMultiplierOptions( 0, tribol::ImplicitEvalMode::MORTAR_WEIGHTS_EVAL,
                                        tribol::SparseMode::MFEM_LINKED_LIST );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, single_mortar_3D_penalty )
{
  registerDummy3DMesh( 0 );
  registerDummy3DMesh( 1 );

  // register dummy nodal fields so error doesn't return from field
  // registration
  tribol::Array1D<RealT> gaps( this->m_lengthNodalData );
  tribol::Array1D<RealT> pressures( this->m_lengthNodalData );

  tribol::registerMortarGaps( 1, &gaps[0] );
  tribol::registerMortarPressures( 1, &pressures[0] );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::SINGLE_MORTAR,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, common_plane_lagrange_multiplier )
{
  registerDummy3DMesh( 0 );
  registerDummy3DMesh( 1 );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setLagrangeMultiplierOptions( 0, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN,
                                        tribol::SparseMode::MFEM_LINKED_LIST );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );
}

TEST_F( CouplingSchemeTest, mortar_no_nodal_gaps_or_pressures )
{
  registerDummy3DMesh( 0 );
  registerDummy3DMesh( 1 );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::SINGLE_MORTAR,
                                  tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setLagrangeMultiplierOptions( 0, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN,
                                        tribol::SparseMode::MFEM_LINKED_LIST );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, mortar_tied )
{
  registerDummy3DMesh( 0 );
  registerDummy3DMesh( 1 );

  // register dummy nodal fields so error doesn't return from field
  // registration
  tribol::Array1D<RealT> gaps( this->m_lengthNodalData );
  tribol::Array1D<RealT> pressures( this->m_lengthNodalData );

  tribol::registerMortarGaps( 1, &gaps[0] );
  tribol::registerMortarPressures( 1, &pressures[0] );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::SINGLE_MORTAR,
                                  tribol::TIED_NORMAL, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setLagrangeMultiplierOptions( 0, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN,
                                        tribol::SparseMode::MFEM_LINKED_LIST );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, mortar_coulomb )
{
  registerDummy3DMesh( 0 );
  registerDummy3DMesh( 1 );

  // register dummy nodal fields so error doesn't return from field
  // registration
  tribol::Array1D<RealT> gaps( this->m_lengthNodalData );
  tribol::Array1D<RealT> pressures( this->m_lengthNodalData );

  tribol::registerMortarGaps( 1, &gaps[0] );
  tribol::registerMortarPressures( 1, &pressures[0] );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::SINGLE_MORTAR,
                                  tribol::FRICTION_COULOMB, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setLagrangeMultiplierOptions( 0, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN,
                                        tribol::SparseMode::MFEM_LINKED_LIST );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, common_plane_tied )
{
  registerDummy3DMesh( 0 );
  registerDummy3DMesh( 1 );

  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::TIED_NORMAL, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, true );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, common_plane_coulomb )
{
  registerDummy3DMesh( 0 );
  registerDummy3DMesh( 1 );

  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTION_COULOMB, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, non_null_to_null_meshes )
{
  ////////////////////////////////////////////////
  // setup simple non-null contacting test mesh //
  ////////////////////////////////////////////////
  tribol::TestMesh mesh;
  mesh.mortarMeshId = 0;
  mesh.nonmortarMeshId = 1;

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

  mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1, nElemsXS,
                            nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0., 0. );

  // register meshes
  tribol::registerMesh( mesh.mortarMeshId, mesh.numMortarFaces, mesh.numTotalNodes, mesh.faceConn1,
                        (int)( tribol::LINEAR_QUAD ), mesh.x, mesh.y, mesh.z, tribol::MemorySpace::Host );

  tribol::registerMesh( mesh.nonmortarMeshId, mesh.numNonmortarFaces, mesh.numTotalNodes, mesh.faceConn2,
                        (int)( tribol::LINEAR_QUAD ), mesh.x, mesh.y, mesh.z, tribol::MemorySpace::Host );

  // set penalty data so coupling scheme initialization passes
  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  // register nodal responses (i.e. nodal forces)
  tribol::allocRealArray( &mesh.fx1, mesh.numTotalNodes, 0. );
  tribol::allocRealArray( &mesh.fy1, mesh.numTotalNodes, 0. );
  tribol::allocRealArray( &mesh.fz1, mesh.numTotalNodes, 0. );
  tribol::allocRealArray( &mesh.fx2, mesh.numTotalNodes, 0. );
  tribol::allocRealArray( &mesh.fy2, mesh.numTotalNodes, 0. );
  tribol::allocRealArray( &mesh.fz2, mesh.numTotalNodes, 0. );

  tribol::registerNodalResponse( mesh.mortarMeshId, mesh.fx1, mesh.fy1, mesh.fz1 );

  tribol::registerNodalResponse( mesh.nonmortarMeshId, mesh.fx2, mesh.fy2, mesh.fz2 );

  // register the coupling scheme
  const int csIndex = 0;
  tribol::registerCouplingScheme( csIndex, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  // call update so binning on coupling scheme is performed.
  RealT dt = 1.0;
  EXPECT_EQ( tribol::update( 1, 1., dt ), 0 );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* cs_non_null = &csManager.at( csIndex );

  // check that the total number of nodes in the coupling scheme are
  // the two 8 node hexes in 3D
  EXPECT_EQ( cs_non_null->getNumTotalNodes(), 16 );  // hard coded for simplicity

  // check that there is a single active pair
  EXPECT_EQ( cs_non_null->getNumActivePairs(), 1 );

  ///////////////////////////////////////////////////////////////
  // Re-register Data With Same Mesh Ids and CouplingScheme Id //
  ///////////////////////////////////////////////////////////////

  // register same mesh IDs as NULL meshes
  int elementType = (int)( tribol::LINEAR_QUAD );
  tribol::registerMesh( 0, 0, 0, nullptr, elementType, nullptr, nullptr, nullptr, tribol::MemorySpace::Host );
  tribol::registerMesh( 1, 0, 0, nullptr, elementType, nullptr, nullptr, nullptr, tribol::MemorySpace::Host );

  // set penalty data for valid coupling scheme with penalty enforcement.
  // Previous meshes and mesh associated penalty data is overwritten with
  // registerMesh() calls above
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  // RE-register coupling scheme 0 with null-meshes with same IDs as before
  tribol::registerCouplingScheme( csIndex, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  tribol::CouplingScheme* cs_null = &csManager.at( csIndex );

  // check that total number of nodes on the coupling scheme is zero from null meshes
  EXPECT_EQ( cs_null->getNumTotalNodes(), 0 );

  // check that the number of active pairs is zero from initialization and not
  // carried over from previous coupling scheme registration with non-null meshes
  EXPECT_EQ( cs_null->getNumActivePairs(), 0 );

  // call cs_null->init() to make sure the nullMeshes boolean is correctly set
  cs_null->init();
  EXPECT_EQ( cs_null->nullMeshes(), true );

  // call update() to make sure there is a no-op for this coupling scheme
  EXPECT_EQ( tribol::update( 1, 1., dt ), 0 );
  EXPECT_EQ( cs_null->getNumActivePairs(), 0 );

  // check InterfacePairs data
  EXPECT_EQ( cs_null->getInterfacePairs().size(), 0 );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, invalid_mesh_in_coupling_scheme )
{
  // register meshes
  int id1 = 0;
  int id2 = 1;
  int numFaces1 = 1;
  int numFaces2 = 1;
  constexpr int numTotalNodes1 = 2;
  constexpr int numTotalNodes2 = 4;
  tribol::IndexT faceConn1[numTotalNodes1] = { 0, 1 };        // dummy triangle connectivity (invalid)
  tribol::IndexT faceConn2[numTotalNodes2] = { 4, 5, 6, 7 };  // dummy quadrilateral connectivity (valid)
  RealT x1[numTotalNodes1] = { 0., 0.5 };
  RealT y1[numTotalNodes1] = { 0., 0. };
  RealT z1[numTotalNodes1] = { 0., 0. };
  RealT x2[numTotalNodes2] = { 0., 0.5, 0.5, 0. };
  RealT y2[numTotalNodes2] = { 0., 0., 0.5, 0.5 };
  RealT z2[numTotalNodes2] = { -0.1, -0.1, -0.1, -0.1 };
  RealT fx1[numTotalNodes1], fy1[numTotalNodes1], fz1[numTotalNodes1];
  RealT fx2[numTotalNodes2], fy2[numTotalNodes2], fz2[numTotalNodes2];

  tribol::initRealArray( &fx1[0], numTotalNodes1, 0. );
  tribol::initRealArray( &fy1[0], numTotalNodes1, 0. );
  tribol::initRealArray( &fz1[0], numTotalNodes1, 0. );

  tribol::initRealArray( &fx2[0], numTotalNodes2, 0. );
  tribol::initRealArray( &fy2[0], numTotalNodes2, 0. );
  tribol::initRealArray( &fz2[0], numTotalNodes2, 0. );

  tribol::registerMesh( id1, numFaces1, numTotalNodes1, &faceConn1[0], (int)( tribol::LINEAR_EDGE ), &x1[0], &y1[0],
                        &z1[0], tribol::MemorySpace::Host );

  tribol::registerMesh( id2, numFaces2, numTotalNodes2, &faceConn2[0], (int)( tribol::LINEAR_QUAD ), &x2[0], &y2[0],
                        &z2[0], tribol::MemorySpace::Host );

  // set penalty data so coupling scheme initialization passes
  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  tribol::registerNodalResponse( id1, &fx1[0], &fy1[0], &fz1[0] );

  tribol::registerNodalResponse( id2, &fx2[0], &fy2[0], &fz2[0] );

  // register the coupling scheme
  const int csIndex = 0;
  tribol::registerCouplingScheme( csIndex, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, finalize )
{
  ////////////////////////////////////////////////
  // setup simple non-null contacting test mesh //
  ////////////////////////////////////////////////
  tribol::TestMesh mesh;
  mesh.mortarMeshId = 0;
  mesh.nonmortarMeshId = 1;

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

  mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1, nElemsXS,
                            nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0., 0. );

  // register meshes
  tribol::registerMesh( mesh.mortarMeshId, mesh.numMortarFaces, mesh.numTotalNodes, mesh.faceConn1,
                        (int)( tribol::LINEAR_QUAD ), mesh.x, mesh.y, mesh.z, tribol::MemorySpace::Host );

  tribol::registerMesh( mesh.nonmortarMeshId, mesh.numNonmortarFaces, mesh.numTotalNodes, mesh.faceConn2,
                        (int)( tribol::LINEAR_QUAD ), mesh.x, mesh.y, mesh.z, tribol::MemorySpace::Host );

  // set penalty data so coupling scheme initialization passes
  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  // register nodal responses (i.e. nodal forces)
  tribol::allocRealArray( &mesh.fx1, mesh.numTotalNodes, 0. );
  tribol::allocRealArray( &mesh.fy1, mesh.numTotalNodes, 0. );
  tribol::allocRealArray( &mesh.fz1, mesh.numTotalNodes, 0. );
  tribol::allocRealArray( &mesh.fx2, mesh.numTotalNodes, 0. );
  tribol::allocRealArray( &mesh.fy2, mesh.numTotalNodes, 0. );
  tribol::allocRealArray( &mesh.fz2, mesh.numTotalNodes, 0. );

  tribol::registerNodalResponse( mesh.mortarMeshId, mesh.fx1, mesh.fy1, mesh.fz1 );

  tribol::registerNodalResponse( mesh.nonmortarMeshId, mesh.fx2, mesh.fy2, mesh.fz2 );

  // register the coupling scheme
  const int csIndex = 0;
  tribol::registerCouplingScheme( csIndex, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( csIndex, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  // call update so binning on coupling scheme is performed.
  RealT dt = 1.0;
  EXPECT_EQ( tribol::update( 1, 1., dt ), 0 );

  tribol::finalize();

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  EXPECT_EQ( csManager.findData( csIndex ), nullptr );
}

TEST_F( CouplingSchemeTest, null_velocity_kinematic_penalty )
{
  registerDummy3DMesh( 0 );
  registerDummy3DMesh( 1 );

  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  // register null nodal velocity pointers. The coupling scheme
  // should initialize correctly for kinematic penalty only.
  RealT* v_x{ nullptr };
  RealT* v_y{ nullptr };
  RealT* v_z{ nullptr };
  tribol::registerNodalVelocities( 0, v_x, v_y, v_z );
  tribol::registerNodalVelocities( 1, v_x, v_y, v_z );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, true );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, null_velocity_kinematic_and_rate_penalty )
{
  registerDummy3DMesh( 0 );
  registerDummy3DMesh( 1 );

  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC_AND_RATE, tribol::KINEMATIC_CONSTANT );

  // register null nodal velocity pointers. The coupling scheme
  // should NOT initialize correctly for kinematic-and-rate penalty.
  RealT* v_x{ nullptr };
  RealT* v_y{ nullptr };
  RealT* v_z{ nullptr };
  tribol::registerNodalVelocities( 0, v_x, v_y, v_z );
  tribol::registerNodalVelocities( 1, v_x, v_y, v_z );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, mortar_weights_null_response_pointers )
{
  bool setResponse = false;
  int numCells = 1;
  registerDummy3DMesh( 0, numCells, setResponse );
  registerDummy3DMesh( 1, numCells, setResponse );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::MORTAR_WEIGHTS,
                                  tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setLagrangeMultiplierOptions( 0, tribol::ImplicitEvalMode::MORTAR_WEIGHTS_EVAL,
                                        tribol::SparseMode::MFEM_LINKED_LIST );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, true );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, single_mortar_null_response_pointers )
{
  bool setResponse = false;
  int numCells = 1;
  registerDummy3DMesh( 0, numCells, setResponse );
  registerDummy3DMesh( 1, numCells, setResponse );

  tribol::Array1D<RealT> gaps( this->m_lengthNodalData );
  tribol::Array1D<RealT> pressures( this->m_lengthNodalData );

  tribol::registerMortarGaps( 1, &gaps[0] );
  tribol::registerMortarPressures( 1, &pressures[0] );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::SINGLE_MORTAR,
                                  tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setLagrangeMultiplierOptions( 0, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN,
                                        tribol::SparseMode::MFEM_LINKED_LIST );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, common_plane_null_response_pointers )
{
  int numCells = 1;
  bool setResponse = false;
  registerDummy3DMesh( 0, numCells, setResponse );
  registerDummy3DMesh( 1, numCells, setResponse );

  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, null_mesh_with_null_pointers )
{
  int numCells = 0;
  bool setResponse = false;
  registerDummy3DMesh( 0, numCells, setResponse );
  registerDummy3DMesh( 1, numCells, setResponse );

  RealT penalty = 1.0;
  tribol::setKinematicConstantPenalty( 0, penalty );
  tribol::setKinematicConstantPenalty( 1, penalty );

  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  // register null nodal velocity pointers. The coupling scheme
  // should NOT initialize correctly for kinematic-and-rate penalty.
  RealT* v_x{ nullptr };
  RealT* v_y{ nullptr };
  RealT* v_z{ nullptr };
  tribol::registerNodalVelocities( 0, v_x, v_y, v_z );
  tribol::registerNodalVelocities( 1, v_x, v_y, v_z );

  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( 0 );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, true );
  EXPECT_EQ( scheme->nullMeshes(), true );

  // check the InterfacePairs member class on the coupling scheme
  EXPECT_EQ( scheme->getInterfacePairs().size(), 0 );

  tribol::finalize();
}

TEST_F( CouplingSchemeTest, auto_common_plane_no_element_thickness )
{
  tribol::IndexT mesh_id = 0;
  int csId = 0;
  registerDummy3DMesh( mesh_id );

  tribol::registerCouplingScheme( csId, mesh_id, mesh_id, tribol::SURFACE_TO_SURFACE, tribol::AUTO,
                                  tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setKinematicConstantPenalty( mesh_id, 1.0 );

  tribol::setPenaltyOptions( csId, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( csId );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, false );
}

TEST_F( CouplingSchemeTest, auto_common_plane_with_element_thickness )
{
  tribol::IndexT mesh_id = 0;
  int numElements = 1;
  int csId = 0;
  registerDummy3DMesh( mesh_id, numElements );

  tribol::registerCouplingScheme( csId, mesh_id, mesh_id, tribol::SURFACE_TO_SURFACE, tribol::AUTO,
                                  tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );

  tribol::setKinematicConstantPenalty( mesh_id, 1.0 );

  tribol::setPenaltyOptions( csId, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );

  RealT element_thick = 1.0;
  tribol::registerRealElementField( mesh_id, tribol::ELEMENT_THICKNESS, &element_thick );

  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();
  tribol::CouplingScheme* scheme = &csManager.at( csId );
  bool isInit = scheme->init();

  EXPECT_EQ( isInit, true );
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
