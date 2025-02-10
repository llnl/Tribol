// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/types.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MeshManager.hpp"
#include "tribol/physics/Mortar.hpp"
#include "tribol/physics/AlignedMortar.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/utils/TestUtils.hpp"

// Axom includes
#include "axom/slic.hpp"

// gtest includes
#include "gtest/gtest.h"

// c++ includes
#include <cmath>  // std::abs

using real = tribol::real;

/*!
 * Test fixture class with some setup necessary to compute
 * the mortar contact forces between two parallel, but
 * misaligned faces
 */
class MortarForceTest : public ::testing::Test {
 public:
  int numNodes;
  int numFaces;
  int numNodesPerFace;
  int numOverlapNodes;
  int dim;

  real* getXCoords() { return x; }

  real* getYCoords() { return y; }

  real* getZCoords() { return z; }

  real* getXOverlapCoords() { return xOverlap; }

  real* getYOverlapCoords() { return yOverlap; }

  real* getZOverlapCoords() { return zOverlap; }

  void checkMortarForces( int* conn1, int* conn2, tribol::ContactMethod method )
  {
    // grab coordinate data
    real* x = this->x;
    real* y = this->y;
    real* z = this->z;

    // register the mesh with tribol
    int cellType = static_cast<int>( tribol::UNDEFINED_ELEMENT );
    switch ( this->numNodesPerFace ) {
      case 4: {
        cellType = (int)( tribol::LINEAR_QUAD );
        break;
      }
      default: {
        SLIC_ERROR( "checkMortarForces: number of nodes per face not equal to 4." );
      }
    }

    const int mortarMeshId = 0;
    const int nonmortarMeshId = 1;

    // initialize tribol
    tribol::CommType problem_comm = TRIBOL_COMM_WORLD;
    tribol::initialize( dim, problem_comm );

    // register mesh
    tribol::registerMesh( mortarMeshId, 1, this->numNodes, conn1, cellType, x, y, z );
    tribol::registerMesh( nonmortarMeshId, 1, this->numNodes, conn2, cellType, x, y, z );

    // register nodal forces
    real *fx1, *fy1, *fz1;
    real *fx2, *fy2, *fz2;

    real forceX1[this->numNodes];
    real forceY1[this->numNodes];
    real forceZ1[this->numNodes];

    real forceX2[this->numNodes];
    real forceY2[this->numNodes];
    real forceZ2[this->numNodes];

    fx1 = forceX1;
    fy1 = forceY1;
    fz1 = forceZ1;

    fx2 = forceX2;
    fy2 = forceY2;
    fz2 = forceZ2;

    // initialize force arrays
    for ( int i = 0; i < this->numNodes; ++i ) {
      fx1[i] = 0.;
      fy1[i] = 0.;
      fz1[i] = 0.;
      fx2[i] = 0.;
      fy2[i] = 0.;
      fz2[i] = 0.;
    }

    tribol::registerNodalResponse( mortarMeshId, fx1, fy1, fz1 );
    tribol::registerNodalResponse( nonmortarMeshId, fx2, fy2, fz2 );

    // register nodal pressure and nodal gap array for the nonmortar mesh
    real *gaps, *pressures;

    gaps = new real[this->numNodes];
    pressures = new real[this->numNodes];

    // initialize gaps and pressures. Initialize all
    // nonmortar pressures to 1.0
    for ( int i = 0; i < this->numNodes; ++i ) {
      gaps[i] = 0.;
      pressures[i] = 1.;
    }

    // register nodal gaps and pressure arrays
    tribol::registerMortarGaps( nonmortarMeshId, gaps );
    tribol::registerMortarPressures( nonmortarMeshId, pressures );

    // register coupling scheme
    const int csIndex = 0;
    tribol::registerCouplingScheme( csIndex, mortarMeshId, nonmortarMeshId, tribol::SURFACE_TO_SURFACE, tribol::AUTO,
                                    method, tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER );

    tribol::setLagrangeMultiplierOptions( csIndex, tribol::ImplicitEvalMode::MORTAR_RESIDUAL,
                                          tribol::SparseMode::MFEM_LINKED_LIST );

    // call tribol update
    double dt = 1.0;
    int tribol_update_err = tribol::update( 1, 1., dt );

    EXPECT_EQ( tribol_update_err, 0 );

    // diagnostics
    tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
    tribol::MeshData& mortarMesh = meshManager.GetMeshInstance( mortarMeshId );
    tribol::MeshData& nonmortarMesh = meshManager.GetMeshInstance( nonmortarMeshId );

    // compute the sum of the nodal forces
    real fx1Sum = 0.;
    real fy1Sum = 0.;
    real fz1Sum = 0.;

    real fx2Sum = 0.;
    real fy2Sum = 0.;
    real fz2Sum = 0.;
    for ( int i = 0; i < this->numNodesPerFace; ++i ) {
      int nonmortarNodeId = nonmortarMesh.getFaceNodeId( 0, i );
      int mortarNodeId = mortarMesh.getFaceNodeId( 0, i );

      fx2Sum += nonmortarMesh.m_forceX[nonmortarNodeId];
      fy2Sum += nonmortarMesh.m_forceY[nonmortarNodeId];
      fz2Sum += nonmortarMesh.m_forceZ[nonmortarNodeId];

      fx1Sum += mortarMesh.m_forceX[mortarNodeId];
      fy1Sum += mortarMesh.m_forceY[mortarNodeId];
      fz1Sum += mortarMesh.m_forceZ[mortarNodeId];
    }

    // sum nonmortar pressure
    real pSum = 0.;
    for ( int i = 0; i < this->numNodesPerFace; ++i ) {
      int nonmortarNodeId = nonmortarMesh.getFaceNodeId( 0, i );
      pSum += nonmortarMesh.m_nodalFields.m_node_pressure[nonmortarNodeId];
    }

    real diffX1 = std::abs( fx1Sum ) - std::abs( pSum );
    real diffY1 = std::abs( fy1Sum ) - std::abs( pSum );
    real diffZ1 = std::abs( fz1Sum ) - std::abs( pSum );

    real diffX2 = std::abs( fx2Sum ) - std::abs( pSum );
    real diffY2 = std::abs( fy2Sum ) - std::abs( pSum );
    real diffZ2 = std::abs( fz2Sum ) - std::abs( pSum );

    real tol = 1.e-8;
    EXPECT_LE( diffX1, tol );
    EXPECT_LE( diffY1, tol );
    EXPECT_LE( diffZ1, tol );
    EXPECT_LE( diffX2, tol );
    EXPECT_LE( diffY2, tol );
    EXPECT_LE( diffZ2, tol );

    // finalize
    tribol::finalize();

    delete[] gaps;
    delete[] pressures;

  }  // end checkMortarForces()

 protected:
  void SetUp() override
  {
    this->numNodes = 8;
    this->numFaces = 2;
    this->numNodesPerFace = 4;
    this->numOverlapNodes = 4;
    this->dim = 3;

    if ( this->x == nullptr ) {
      this->x = new real[this->numNodes];
    } else {
      delete[] this->x;
      this->x = new real[this->numNodes];
    }

    if ( this->y == nullptr ) {
      this->y = new real[this->numNodes];
    } else {
      delete[] this->y;
      this->y = new real[this->numNodes];
    }

    if ( this->z == nullptr ) {
      this->z = new real[this->numNodes];
    } else {
      delete[] this->z;
      this->z = new real[this->numNodes];
    }

    if ( this->xOverlap == nullptr ) {
      this->xOverlap = new real[this->numOverlapNodes];
    } else {
      delete[] this->xOverlap;
      this->xOverlap = new real[this->numOverlapNodes];
    }

    if ( this->yOverlap == nullptr ) {
      this->yOverlap = new real[this->numOverlapNodes];
    } else {
      delete[] this->yOverlap;
      this->yOverlap = new real[this->numOverlapNodes];
    }
    if ( this->zOverlap == nullptr ) {
      this->zOverlap = new real[this->numOverlapNodes];
    } else {
      delete[] this->zOverlap;
      this->zOverlap = new real[this->numOverlapNodes];
    }
  }

  void TearDown() override
  {
    if ( this->x != nullptr ) {
      delete[] this->x;
      this->x = nullptr;
    }
    if ( this->y != nullptr ) {
      delete[] this->y;
      this->y = nullptr;
    }
    if ( this->z != nullptr ) {
      delete[] this->z;
      this->z = nullptr;
    }
    if ( this->xOverlap != nullptr ) {
      delete[] this->xOverlap;
      this->xOverlap = nullptr;
    }
    if ( this->yOverlap != nullptr ) {
      delete[] this->yOverlap;
      this->yOverlap = nullptr;
    }
    if ( this->zOverlap != nullptr ) {
      delete[] this->zOverlap;
      this->zOverlap = nullptr;
    }
  }

 protected:
  real* x{ nullptr };
  real* y{ nullptr };
  real* z{ nullptr };

  real* xOverlap{ nullptr };
  real* yOverlap{ nullptr };
  real* zOverlap{ nullptr };
};

TEST_F( MortarForceTest, parallel_misaligned )
{
  real* x = this->getXCoords();
  real* y = this->getYCoords();
  real* z = this->getZCoords();

  real* xOvrlp = this->getXOverlapCoords();
  real* yOvrlp = this->getYOverlapCoords();
  real* zOvrlp = this->getZOverlapCoords();

  x[0] = -1.;
  x[1] = -1.;
  x[2] = 1.;
  x[3] = 1.;

  y[0] = 1.;
  y[1] = -1.;
  y[2] = -1.;
  y[3] = 1.;

  z[0] = 0.1;
  z[1] = 0.1;
  z[2] = 0.1;
  z[3] = 0.1;

  x[4] = 0.;
  x[5] = 2.;
  x[6] = 2.;
  x[7] = 0.;

  y[4] = 0.;
  y[5] = 0.;
  y[6] = -2.;
  y[7] = -2.;

  z[4] = 0;
  z[5] = 0;
  z[6] = 0;
  z[7] = 0;

  xOvrlp[0] = 0.;
  xOvrlp[1] = 0.;
  xOvrlp[2] = 1.;
  xOvrlp[3] = 1.;

  yOvrlp[0] = 0.;
  yOvrlp[1] = -1.;
  yOvrlp[2] = -1.;
  yOvrlp[3] = 0.;

  zOvrlp[0] = 0.1;
  zOvrlp[1] = 0.1;
  zOvrlp[2] = 0.1;
  zOvrlp[3] = 0.1;

  // register a tribol mesh for computing mortar gaps
  int numNodesPerFace = 4;
  int conn1[numNodesPerFace];
  int conn2[numNodesPerFace];

  for ( int i = 0; i < numNodesPerFace; ++i ) {
    conn1[i] = i;
    conn2[i] = numNodesPerFace + i;
  }

  this->checkMortarForces( &conn1[0], &conn2[0], tribol::SINGLE_MORTAR );
}

TEST_F( MortarForceTest, parallel_aligned )
{
  real* x = this->getXCoords();
  real* y = this->getYCoords();
  real* z = this->getZCoords();

  real* xOvrlp = this->getXOverlapCoords();
  real* yOvrlp = this->getYOverlapCoords();
  real* zOvrlp = this->getZOverlapCoords();

  x[0] = -1.;
  x[1] = -1.;
  x[2] = 1.;
  x[3] = 1.;

  y[0] = 1.;
  y[1] = -1.;
  y[2] = -1.;
  y[3] = 1.;

  z[0] = 0.1;
  z[1] = 0.1;
  z[2] = 0.1;
  z[3] = 0.1;

  x[4] = -1.;
  x[5] = 1.;
  x[6] = 1.;
  x[7] = -1.;

  y[4] = 1.;
  y[5] = 1.;
  y[6] = -1.;
  y[7] = -1.;

  z[4] = 0;
  z[5] = 0;
  z[6] = 0;
  z[7] = 0;

  xOvrlp[0] = x[0];
  xOvrlp[1] = x[1];
  xOvrlp[2] = x[2];
  xOvrlp[3] = x[3];

  yOvrlp[0] = y[0];
  yOvrlp[1] = y[1];
  yOvrlp[2] = y[2];
  yOvrlp[3] = y[3];

  zOvrlp[0] = z[0];
  zOvrlp[1] = z[1];
  zOvrlp[2] = z[2];
  zOvrlp[3] = z[3];

  // register a tribol mesh for computing mortar gaps
  int numNodesPerFace = 4;
  int conn1[numNodesPerFace];
  int conn2[numNodesPerFace];

  for ( int i = 0; i < numNodesPerFace; ++i ) {
    conn1[i] = i;
    conn2[i] = numNodesPerFace + i;
  }

  this->checkMortarForces( &conn1[0], &conn2[0], tribol::SINGLE_MORTAR );
}

TEST_F( MortarForceTest, non_parallel_misaligned )
{
  real* x = this->getXCoords();
  real* y = this->getYCoords();
  real* z = this->getZCoords();

  real* xOvrlp = this->getXOverlapCoords();
  real* yOvrlp = this->getYOverlapCoords();
  real* zOvrlp = this->getZOverlapCoords();

  x[0] = -1.;
  x[1] = -1.;
  x[2] = 1.;
  x[3] = 1.;

  y[0] = 1.;
  y[1] = -1.;
  y[2] = -1.;
  y[3] = 1.;

  z[0] = 0.1;
  z[1] = 0.1;
  z[2] = 0.1;
  z[3] = 0.1;

  x[4] = 0.;
  x[5] = 2.;
  x[6] = 2.;
  x[7] = 0.;

  y[4] = 0.;
  y[5] = 0.;
  y[6] = -2.;
  y[7] = -2.;

  z[4] = 0.05;
  z[5] = 0.05;
  z[6] = 0;
  z[7] = 0;

  xOvrlp[0] = 0.;
  xOvrlp[1] = 0.;
  xOvrlp[2] = 1.;
  xOvrlp[3] = 1.;

  yOvrlp[0] = 0.;
  yOvrlp[1] = -1.;
  yOvrlp[2] = -1.;
  yOvrlp[3] = 0.;

  zOvrlp[0] = 0.1;
  zOvrlp[1] = 0.1;
  zOvrlp[2] = 0.1;
  zOvrlp[3] = 0.1;

  // register a tribol mesh for computing mortar gaps
  int numNodesPerFace = 4;
  int conn1[numNodesPerFace];
  int conn2[numNodesPerFace];

  for ( int i = 0; i < numNodesPerFace; ++i ) {
    conn1[i] = i;
    conn2[i] = numNodesPerFace + i;
  }

  this->checkMortarForces( &conn1[0], &conn2[0], tribol::SINGLE_MORTAR );
}

TEST_F( MortarForceTest, non_parallel_aligned )
{
  real* x = this->getXCoords();
  real* y = this->getYCoords();
  real* z = this->getZCoords();

  real* xOvrlp = this->getXOverlapCoords();
  real* yOvrlp = this->getYOverlapCoords();
  real* zOvrlp = this->getZOverlapCoords();

  x[0] = -1.;
  x[1] = -1.;
  x[2] = 1.;
  x[3] = 1.;

  y[0] = 1.;
  y[1] = -1.;
  y[2] = -1.;
  y[3] = 1.;

  z[0] = 0.1;
  z[1] = 0.1;
  z[2] = 0.1;
  z[3] = 0.1;

  x[4] = -1.;
  x[5] = 1.;
  x[6] = 1.;
  x[7] = -1.;

  y[4] = 1.;
  y[5] = 1.;
  y[6] = -1.;
  y[7] = -1.;

  z[4] = 0.05;
  z[5] = 0.05;
  z[6] = 0;
  z[7] = 0;

  xOvrlp[0] = x[0];
  xOvrlp[1] = x[1];
  xOvrlp[2] = x[2];
  xOvrlp[3] = x[3];

  yOvrlp[0] = y[0];
  yOvrlp[1] = y[1];
  yOvrlp[2] = y[2];
  yOvrlp[3] = y[3];

  zOvrlp[0] = z[0];
  zOvrlp[1] = z[1];
  zOvrlp[2] = z[2];
  zOvrlp[3] = z[3];

  // register a tribol mesh for computing mortar gaps
  int numNodesPerFace = 4;
  int conn1[numNodesPerFace];
  int conn2[numNodesPerFace];

  for ( int i = 0; i < numNodesPerFace; ++i ) {
    conn1[i] = i;
    conn2[i] = numNodesPerFace + i;
  }

  this->checkMortarForces( &conn1[0], &conn2[0], tribol::SINGLE_MORTAR );
}

TEST_F( MortarForceTest, parallel_simple_aligned )
{
  real* x = this->getXCoords();
  real* y = this->getYCoords();
  real* z = this->getZCoords();

  real* xOvrlp = this->getXOverlapCoords();
  real* yOvrlp = this->getYOverlapCoords();
  real* zOvrlp = this->getZOverlapCoords();

  x[0] = -1.;
  x[1] = -1.;
  x[2] = 1.;
  x[3] = 1.;

  y[0] = 1.;
  y[1] = -1.;
  y[2] = -1.;
  y[3] = 1.;

  z[0] = 0.1;
  z[1] = 0.1;
  z[2] = 0.1;
  z[3] = 0.1;

  x[4] = -1.;
  x[5] = 1.;
  x[6] = 1.;
  x[7] = -1.;

  y[4] = 1.;
  y[5] = 1.;
  y[6] = -1.;
  y[7] = -1.;

  z[4] = 0.;
  z[5] = 0.;
  z[6] = 0.;
  z[7] = 0.;

  xOvrlp[0] = x[0];
  xOvrlp[1] = x[1];
  xOvrlp[2] = x[2];
  xOvrlp[3] = x[3];

  yOvrlp[0] = y[0];
  yOvrlp[1] = y[1];
  yOvrlp[2] = y[2];
  yOvrlp[3] = y[3];

  zOvrlp[0] = z[0];
  zOvrlp[1] = z[1];
  zOvrlp[2] = z[2];
  zOvrlp[3] = z[3];

  // register a tribol mesh for computing mortar gaps
  int numNodesPerFace = 4;
  int conn1[numNodesPerFace];
  int conn2[numNodesPerFace];

  for ( int i = 0; i < numNodesPerFace; ++i ) {
    conn1[i] = i;
    conn2[i] = numNodesPerFace + i;
  }

  this->checkMortarForces( &conn1[0], &conn2[0], tribol::ALIGNED_MORTAR );
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;  // create & initialize logger,

  result = RUN_ALL_TESTS();

  return result;
}
