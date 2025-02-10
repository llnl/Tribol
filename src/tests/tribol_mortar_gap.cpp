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
#include "tribol/mesh/MethodCouplingData.hpp"
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
 * the mortar gap between two parallel, but misaligned faces
 */
class MortarGapTest : public ::testing::Test {
 public:
  int numNodes;
  int numFaces;
  int numNodesPerFace;
  int numOverlapNodes;
  int dim;

  real* getXCoords( int id )
  {
    if ( id == 0 ) {
      return x1;
    } else {
      return x2;
    }
  }

  real* getYCoords( int id )
  {
    if ( id == 0 ) {
      return y1;
    } else {
      return y2;
    }
  }

  real* getZCoords( int id )
  {
    if ( id == 0 ) {
      return z1;
    } else {
      return z2;
    }
  }

  real* getXOverlapCoords() { return xOverlap; }

  real* getYOverlapCoords() { return yOverlap; }

  real* getZOverlapCoords() { return zOverlap; }

  void checkMortarGaps( int* conn1, int* conn2, tribol::ContactMethod method )
  {
    // declare arrays to hold stacked coordinates for each
    // face used in initializing a SurfaceContactElem struct
    real xyz1[this->dim * this->numNodesPerFace];
    real xyz2[this->dim * this->numNodesPerFace];

    // declare array to hold overlap vertices used for
    // initializing a SurfaceContactElem struct
    real xyzOverlap[this->dim * this->numOverlapNodes];

    // assign pointers to arrays
    real* xy1 = xyz1;
    real* xy2 = xyz2;
    real* xyOverlap = xyzOverlap;

    // grab coordinate data
    real* x1 = this->x1;
    real* y1 = this->y1;
    real* z1 = this->z1;
    real* x2 = this->x2;
    real* y2 = this->y2;
    real* z2 = this->z2;
    real* xo = this->xOverlap;
    real* yo = this->yOverlap;
    real* zo = this->zOverlap;

    // generate stacked coordinate array
    for ( int j = 0; j < this->numNodesPerFace; ++j ) {
      for ( int k = 0; k < this->dim; ++k ) {
        int id = this->dim * j + k;
        // int id2 = this->dim * this->numNodesPerFace + id;
        switch ( k ) {
          case 0:
            xy1[id] = x1[j];
            xy2[id] = x2[j];
            xyOverlap[id] = xo[j];
            break;
          case 1:
            xy1[id] = y1[j];
            xy2[id] = y2[j];
            xyOverlap[id] = yo[j];
            break;
          case 2:
            xy1[id] = z1[j];
            xy2[id] = z2[j];
            xyOverlap[id] = zo[j];
            break;
        }  // end switch
      }    // end loop over dimension
    }      // end loop over nodes

    // register the mesh with tribol
    int cellType = static_cast<int>( tribol::UNDEFINED_ELEMENT );
    switch ( this->numNodesPerFace ) {
      case 4: {
        cellType = (int)( tribol::LINEAR_QUAD );
        break;
      }
      default: {
        SLIC_ERROR( "checkMortarWts: number of nodes per face not equal to 4." );
      }
    }

    int dim = 3;
    tribol::CommType problem_comm = TRIBOL_COMM_WORLD;
    tribol::initialize( dim, problem_comm );

    const int mortarMeshId = 0;
    const int nonmortarMeshId = 1;

    tribol::registerMesh( mortarMeshId, 1, this->numNodesPerFace, conn1, cellType, x1, y1, z1 );
    tribol::registerMesh( nonmortarMeshId, 1, this->numNodesPerFace, conn2, cellType, x2, y2, z2 );

    // get instance of meshes to compute face data required for other calculations
    tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
    tribol::MeshData& mortarMesh = meshManager.GetMeshInstance( mortarMeshId );
    tribol::MeshData& nonmortarMesh = meshManager.GetMeshInstance( nonmortarMeshId );

    mortarMesh.computeFaceData( dim );
    nonmortarMesh.computeFaceData( dim );

    real* gaps;
    int size = 2 * this->numNodesPerFace;
    gaps = new real[size];

    for ( int i = 0; i < size; ++i ) {
      gaps[i] = 0.;
    }

    tribol::registerMortarGaps( nonmortarMeshId, gaps );

    // instantiate SurfaceContactElem struct. Note, this object is instantiated
    // using face 1, face 2, and the set overlap polygon. Note, the mesh ids are set
    // equal to 0, and the face ids are 0 and 1, respectively.
    tribol::SurfaceContactElem elem( this->dim, xy1, xy2, xyOverlap, this->numNodesPerFace, this->numOverlapNodes,
                                     mortarMeshId, nonmortarMeshId, 0, 0 );

    // compute the mortar weights to be stored on
    // the surface contact element struct.
    switch ( method ) {
      case tribol::SINGLE_MORTAR:
        tribol::ComputeMortarWeights( elem );
        break;
      case tribol::ALIGNED_MORTAR:
        tribol::ComputeAlignedMortarWeights( elem );
        break;
      default:
        SLIC_ERROR( "Unsupported contact method" );
        break;
    }

    nonmortarMesh.computeNodalNormals( this->dim );

    switch ( method ) {
      case tribol::SINGLE_MORTAR:
        tribol::ComputeNodalGap<tribol::SINGLE_MORTAR>( elem );
        break;
      case tribol::ALIGNED_MORTAR:
        tribol::ComputeNodalGap<tribol::ALIGNED_MORTAR>( elem );
        break;
      default:
        SLIC_ERROR( "Unsupported contact method" );
        break;
    }

    tribol::finalize();
  }

 protected:
  void SetUp() override
  {
    this->numNodes = 8;
    this->numFaces = 2;
    this->numNodesPerFace = 4;
    this->numOverlapNodes = 4;
    this->dim = 3;

    if ( this->x1 == nullptr ) {
      this->x1 = new real[this->numNodes];
    } else {
      delete[] this->x1;
      this->x1 = new real[this->numNodes];
    }

    if ( this->x2 == nullptr ) {
      this->x2 = new real[this->numNodes];
    } else {
      delete[] this->x2;
      this->x2 = new real[this->numNodes];
    }

    if ( this->y1 == nullptr ) {
      this->y1 = new real[this->numNodes];
    } else {
      delete[] this->y1;
      this->y1 = new real[this->numNodes];
    }

    if ( this->y2 == nullptr ) {
      this->y2 = new real[this->numNodes];
    } else {
      delete[] this->y2;
      this->y2 = new real[this->numNodes];
    }

    if ( this->z1 == nullptr ) {
      this->z1 = new real[this->numNodes];
    } else {
      delete[] this->z1;
      this->z1 = new real[this->numNodes];
    }

    if ( this->z2 == nullptr ) {
      this->z2 = new real[this->numNodes];
    } else {
      delete[] this->z2;
      this->z2 = new real[this->numNodes];
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
    if ( this->x1 != nullptr ) {
      delete[] this->x1;
      this->x1 = nullptr;
    }
    if ( this->x2 != nullptr ) {
      delete[] this->x2;
      this->x2 = nullptr;
    }
    if ( this->y1 != nullptr ) {
      delete[] this->y1;
      this->y1 = nullptr;
    }
    if ( this->y2 != nullptr ) {
      delete[] this->y2;
      this->y2 = nullptr;
    }
    if ( this->z1 != nullptr ) {
      delete[] this->z1;
      this->z1 = nullptr;
    }
    if ( this->z2 != nullptr ) {
      delete[] this->z2;
      this->z2 = nullptr;
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
  real* x1{ nullptr };
  real* y1{ nullptr };
  real* z1{ nullptr };

  real* x2{ nullptr };
  real* y2{ nullptr };
  real* z2{ nullptr };

  real* xOverlap{ nullptr };
  real* yOverlap{ nullptr };
  real* zOverlap{ nullptr };
};

TEST_F( MortarGapTest, parallel_misaligned )
{
  real* x1 = this->getXCoords( 0 );
  real* y1 = this->getYCoords( 0 );
  real* z1 = this->getZCoords( 0 );

  real* x2 = this->getXCoords( 1 );
  real* y2 = this->getYCoords( 1 );
  real* z2 = this->getZCoords( 1 );

  real* xOvrlp = this->getXOverlapCoords();
  real* yOvrlp = this->getYOverlapCoords();
  real* zOvrlp = this->getZOverlapCoords();

  x1[0] = -1.;
  x1[1] = -1.;
  x1[2] = 1.;
  x1[3] = 1.;

  y1[0] = 1.;
  y1[1] = -1.;
  y1[2] = -1.;
  y1[3] = 1.;

  z1[0] = 0.1;
  z1[1] = 0.1;
  z1[2] = 0.1;
  z1[3] = 0.1;

  x2[0] = 0.;
  x2[1] = 2.;
  x2[2] = 2.;
  x2[3] = 0.;

  y2[0] = 0.;
  y2[1] = 0.;
  y2[2] = -2.;
  y2[3] = -2.;

  z2[0] = 0;
  z2[1] = 0;
  z2[2] = 0;
  z2[3] = 0;

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
    conn2[i] = i;
  }

  this->checkMortarGaps( &conn1[0], &conn2[0], tribol::SINGLE_MORTAR );

  tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
  tribol::MeshData& nonmortarMesh = meshManager.GetMeshInstance( 1 );

  // compute the sum of the nodal gaps
  real gap = 0.;
  for ( int i = 0; i < numNodesPerFace; ++i ) {
    gap += nonmortarMesh.m_nodalFields.m_node_gap[i];
  }

  // note the face-gap of 0.1 is hard coded based on the
  // hard-coded face coordinates in this test
  real gapDiff = std::abs( 0.1 + gap );

  real tol = 1.e-8;
  EXPECT_LE( gapDiff, tol );

  if ( nonmortarMesh.m_nodalFields.m_node_gap != nullptr ) {
    delete[] nonmortarMesh.m_nodalFields.m_node_gap;
    nonmortarMesh.m_nodalFields.m_node_gap = nullptr;
  }
}

TEST_F( MortarGapTest, parallel_aligned )
{
  real* x1 = this->getXCoords( 0 );
  real* y1 = this->getYCoords( 0 );
  real* z1 = this->getZCoords( 0 );

  real* x2 = this->getXCoords( 1 );
  real* y2 = this->getYCoords( 1 );
  real* z2 = this->getZCoords( 1 );

  real* xOvrlp = this->getXOverlapCoords();
  real* yOvrlp = this->getYOverlapCoords();
  real* zOvrlp = this->getZOverlapCoords();

  x1[0] = 0.;  //-1.;
  x1[1] = 0.;  //-1.;
  x1[2] = 1.;
  x1[3] = 1.;

  y1[0] = 1.;
  y1[1] = 0.;  // -1.;
  y1[2] = 0.;  // -1.;
  y1[3] = 1.;

  z1[0] = 0.1;
  z1[1] = 0.1;
  z1[2] = 0.1;
  z1[3] = 0.1;

  x2[0] = 0.;  // -1.;
  x2[1] = 1.;
  x2[2] = 1.;
  x2[3] = 0.;  //-1.;

  y2[0] = 1.;
  y2[1] = 1.;
  y2[2] = 0.;  // -1.;
  y2[3] = 0.;  //-1.;

  z2[0] = 0;
  z2[1] = 0;
  z2[2] = 0;
  z2[3] = 0;

  xOvrlp[0] = x1[0];
  xOvrlp[1] = x1[1];
  xOvrlp[2] = x1[2];
  xOvrlp[3] = x1[3];

  yOvrlp[0] = y1[0];
  yOvrlp[1] = y1[1];
  yOvrlp[2] = y1[2];
  yOvrlp[3] = y1[3];

  zOvrlp[0] = z1[0];
  zOvrlp[1] = z1[1];
  zOvrlp[2] = z1[2];
  zOvrlp[3] = z1[3];

  // register a tribol mesh for computing mortar gaps
  int numNodesPerFace = 4;
  int conn1[numNodesPerFace];
  int conn2[numNodesPerFace];

  for ( int i = 0; i < numNodesPerFace; ++i ) {
    conn1[i] = i;
    conn2[i] = i;
  }

  this->checkMortarGaps( &conn1[0], &conn2[0], tribol::SINGLE_MORTAR );

  tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
  tribol::MeshData& nonmortarMesh = meshManager.GetMeshInstance( 1 );

  // compute the sum of the nodal gaps
  real gap = 0.;
  for ( int i = 0; i < numNodesPerFace; ++i ) {
    gap += nonmortarMesh.m_nodalFields.m_node_gap[i];
  }

  // note the face-gap of 0.1 is hard coded based on the
  // hard-coded face coordinates in this test
  real gapDiff = std::abs( 0.1 + gap );

  real tol = 1.e-8;
  EXPECT_LE( gapDiff, tol );

  if ( nonmortarMesh.m_nodalFields.m_node_gap != nullptr ) {
    delete[] nonmortarMesh.m_nodalFields.m_node_gap;
    nonmortarMesh.m_nodalFields.m_node_gap = nullptr;
  }
}

TEST_F( MortarGapTest, parallel_simple_aligned )
{
  real* x1 = this->getXCoords( 0 );
  real* y1 = this->getYCoords( 0 );
  real* z1 = this->getZCoords( 0 );

  real* x2 = this->getXCoords( 1 );
  real* y2 = this->getYCoords( 1 );
  real* z2 = this->getZCoords( 1 );

  real* xOvrlp = this->getXOverlapCoords();
  real* yOvrlp = this->getYOverlapCoords();
  real* zOvrlp = this->getZOverlapCoords();

  x1[0] = -1.;
  x1[1] = -1.;
  x1[2] = 1.;
  x1[3] = 1.;

  y1[0] = 1.;
  y1[1] = -1.;
  y1[2] = -1.;
  y1[3] = 1.;

  z1[0] = 0.1;
  z1[1] = 0.1;
  z1[2] = 0.2;
  z1[3] = 0.1;

  x2[0] = -1.;
  x2[1] = 1.;
  x2[2] = 1.;
  x2[3] = -1.;

  y2[0] = 1.;
  y2[1] = 1.;
  y2[2] = -1.;
  y2[3] = -1.;

  z2[0] = 0;
  z2[1] = 0;
  z2[2] = 0;
  z2[3] = 0;

  xOvrlp[0] = x1[0];
  xOvrlp[1] = x1[1];
  xOvrlp[2] = x1[2];
  xOvrlp[3] = x1[3];

  yOvrlp[0] = y1[0];
  yOvrlp[1] = y1[1];
  yOvrlp[2] = y1[2];
  yOvrlp[3] = y1[3];

  zOvrlp[0] = z1[0];
  zOvrlp[1] = z1[1];
  zOvrlp[2] = z1[2];
  zOvrlp[3] = z1[3];

  // register a tribol mesh for computing mortar gaps
  int numNodesPerFace = 4;
  int conn1[numNodesPerFace];
  int conn2[numNodesPerFace];

  for ( int i = 0; i < numNodesPerFace; ++i ) {
    conn1[i] = i;
    conn2[i] = i;
  }

  this->checkMortarGaps( &conn1[0], &conn2[0], tribol::ALIGNED_MORTAR );

  tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
  tribol::MeshData& nonmortarMesh = meshManager.GetMeshInstance( 1 );

  // compute the sum of the nodal gaps
  real gap = 0.;
  real gapTest = 0;
  for ( int i = 0; i < numNodesPerFace; ++i ) {
    gap += nonmortarMesh.m_nodalFields.m_node_gap[i];
    gapTest += z1[i] - z2[i];
  }

  // note the face-gap of 0.1 is hard coded based on the
  // hard-coded face coordinates in this test
  real gapDiff = std::abs( gapTest + gap );

  real tol = 1.e-8;
  EXPECT_LE( gapDiff, tol );

  if ( nonmortarMesh.m_nodalFields.m_node_gap != nullptr ) {
    delete[] nonmortarMesh.m_nodalFields.m_node_gap;
    nonmortarMesh.m_nodalFields.m_node_gap = nullptr;
  }
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
