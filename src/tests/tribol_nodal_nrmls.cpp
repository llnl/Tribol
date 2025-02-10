// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/types.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MeshManager.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/utils/Math.hpp"

// Axom includes
#include "axom/slic.hpp"

// gtest includes
#include "gtest/gtest.h"

// c++ includes
#include <cmath>  // std::abs

using real = tribol::real;

/*!
 * Test fixture class with some setup for registering a mesh
 * and computing nodally averaged normals as used in mortar methods
 */
class NodalNormalTest : public ::testing::Test {
 public:
  int numNodes;
  int dim;

  void computeNodalNormals( int cell_type, real const* const x, real const* const y, real const* const z,
                            int const* const conn, int const numCells, int const numNodes, int const dim )
  {
    // register the mesh with tribol
    const int meshId = 0;
    tribol::registerMesh( meshId, numCells, numNodes, conn, cell_type, x, y, z );

    // get instance of mesh in order to compute nodally averaged normals
    tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
    tribol::MeshData& mesh = meshManager.GetMeshInstance( meshId );

    // compute the face data for this mesh
    mesh.computeFaceData( dim );

    mesh.computeNodalNormals( dim );

    return;
  }

 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
};

TEST_F( NodalNormalTest, two_quad_inverted_v )
{
  //
  // This test includes two, four node quadrilaterals oriented
  // in an inverted V shape that is symmetric about the Z axis.
  // The two faces are oriented at a 45 degree angle with respect
  // to the Z axis.
  //
  int numFaces = 2;
  int numNodesPerFace = 4;
  int cellType = (int)( tribol::LINEAR_QUAD );
  int conn[numFaces * numNodesPerFace];

  // setup connectivity for the two faces ensuring nodes
  // are ordered consistent with an outward unit normal
  for ( int i = 0; i < numNodesPerFace; ++i ) {
    conn[i] = i;
  }

  conn[4] = 0;
  conn[5] = 5;
  conn[6] = 4;
  conn[7] = 1;

  // setup the nodal coordinates of the mesh
  real x[numNodesPerFace + 2];
  real y[numNodesPerFace + 2];
  real z[numNodesPerFace + 2];

  x[0] = 0.;
  x[1] = 0.;
  x[2] = -1.;
  x[3] = -1.;
  x[4] = 1.;
  x[5] = 1.;

  y[0] = 0.;
  y[1] = 1.;
  y[2] = 1.;
  y[3] = 0.;
  y[4] = 1.;
  y[5] = 0.;

  z[0] = 0.;
  z[1] = 0.;
  z[2] = -1.;
  z[3] = -1.;
  z[4] = -1;
  z[5] = -1.;

  // compute the nodal normals
  computeNodalNormals( cellType, &x[0], &y[0], &z[0], &conn[0], numFaces, 6, 3 );

  tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
  tribol::MeshData& mesh = meshManager.GetMeshInstance( 0 );

  // check each normal...hard coded
  real n1check = tribol::magnitude( mesh.m_node_nX[0] - 0., mesh.m_node_nY[0] - 0., mesh.m_node_nZ[0] - 1. );
  real n2check = tribol::magnitude( mesh.m_node_nX[1] - 0., mesh.m_node_nY[1] - 0., mesh.m_node_nZ[1] - 1. );
  real n3check = tribol::magnitude( mesh.m_node_nX[2] - ( -1. / std::sqrt( 2. ) ), mesh.m_node_nY[2] - 0.,
                                    mesh.m_node_nZ[2] - ( 1. / std::sqrt( 2. ) ) );
  real n4check = tribol::magnitude( mesh.m_node_nX[3] - ( -1. / std::sqrt( 2. ) ), mesh.m_node_nY[3] - 0.,
                                    mesh.m_node_nZ[3] - ( 1. / std::sqrt( 2. ) ) );
  real n5check = tribol::magnitude( mesh.m_node_nX[4] - ( 1. / std::sqrt( 2. ) ), mesh.m_node_nY[4] - 0.,
                                    mesh.m_node_nZ[4] - ( 1. / std::sqrt( 2. ) ) );
  real n6check = tribol::magnitude( mesh.m_node_nX[4] - ( 1. / std::sqrt( 2. ) ), mesh.m_node_nY[4] - 0.,
                                    mesh.m_node_nZ[4] - ( 1. / std::sqrt( 2. ) ) );

  real tol = 1.e-12;

  EXPECT_LE( n1check, tol );
  EXPECT_LE( n2check, tol );
  EXPECT_LE( n3check, tol );
  EXPECT_LE( n4check, tol );
  EXPECT_LE( n5check, tol );
  EXPECT_LE( n6check, tol );
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
