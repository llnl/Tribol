// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/utils/Math.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// Axom includes
#include "axom/slic.hpp"

// gtest includes
#include "gtest/gtest.h"

// c++ includes
#include <cmath> // std::abs

using RealT = tribol::RealT;

/*!
 * Test fixture class with some setup for registering a mesh 
 * and computing nodally averaged normals as used in mortar methods
 */
class NodalNormalTest : public ::testing::Test
{
   
public:
   int numNodes;
   int dim;

   void computeNodalNormals( int cell_type,
                             RealT const * const x, 
                             RealT const * const y, 
                             RealT const * const z,
                             int const * const conn,
                             int const numCells,
                             int const numNodes,
                             int const dim )
   {
      // register the mesh with tribol
      const tribol::IndexT mesh_id = 0;
      tribol::registerMesh( mesh_id, numCells, numNodes, 
                            conn, cell_type, x, y, z, tribol::MemorySpace::Host );


      // get instance of mesh in order to compute nodally averaged normals
      tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
      tribol::MeshData& mesh = meshManager.at( mesh_id );

      // compute the face data for this mesh
      mesh.computeFaceData(tribol::ExecutionMode::Sequential);

      mesh.computeNodalNormals( dim );

      return;
   }

protected:

   void SetUp() override
   {
   }

   void TearDown() override
   {
   }

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
   int cellType = (int)(tribol::LINEAR_QUAD);
   int conn[ numFaces * numNodesPerFace ];

   // setup connectivity for the two faces ensuring nodes 
   // are ordered consistent with an outward unit normal
   for (int i=0; i<numNodesPerFace; ++i)
   {
      conn[i] = i;
   }

   conn[4] = 0;
   conn[5] = 5;
   conn[6] = 4;
   conn[7] = 1;

   // setup the nodal coordinates of the mesh
   RealT x[numNodesPerFace + 2]; 
   RealT y[numNodesPerFace + 2]; 
   RealT z[numNodesPerFace + 2]; 

   x[0] =  0.;
   x[1] =  0.;
   x[2] = -1.;
   x[3] = -1.;
   x[4] =  1.;
   x[5] =  1.;

   y[0] = 0.;
   y[1] = 1.;
   y[2] = 1.; 
   y[3] = 0.;
   y[4] = 1.;
   y[5] = 0.;

   z[0] =  0.;
   z[1] =  0.;
   z[2] = -1.; 
   z[3] = -1.;
   z[4] = -1;
   z[5] = -1.;

   // compute the nodal normals
   computeNodalNormals( cellType, &x[0], &y[0], &z[0], &conn[0], 
                        numFaces, 6, 3 );

   tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
   auto mesh = meshManager.at( 0 ).getView();

   // check each normal...hard coded
   RealT n1check = tribol::magnitude( mesh.getNodalNormals()[0][0] - 0., 
                                     mesh.getNodalNormals()[1][0] - 0., 
                                     mesh.getNodalNormals()[2][0] - 1. );
   RealT n2check = tribol::magnitude( mesh.getNodalNormals()[0][1] - 0., 
                                     mesh.getNodalNormals()[1][1] - 0., 
                                     mesh.getNodalNormals()[2][1] - 1. );
   RealT n3check = tribol::magnitude( mesh.getNodalNormals()[0][2] - (-1./std::sqrt(2.)), 
                                     mesh.getNodalNormals()[1][2] - 0., 
                                     mesh.getNodalNormals()[2][2] - (1./std::sqrt(2.)) );
   RealT n4check = tribol::magnitude( mesh.getNodalNormals()[0][3] - (-1./std::sqrt(2.)), 
                                     mesh.getNodalNormals()[1][3] - 0., 
                                     mesh.getNodalNormals()[2][3] - (1./std::sqrt(2.)) );
   RealT n5check = tribol::magnitude( mesh.getNodalNormals()[0][4] - (1./std::sqrt(2.)), 
                                     mesh.getNodalNormals()[1][4] - 0., 
                                     mesh.getNodalNormals()[2][4] - (1./std::sqrt(2.)) );
   RealT n6check = tribol::magnitude( mesh.getNodalNormals()[0][4] - (1./std::sqrt(2.)), 
                                     mesh.getNodalNormals()[1][4] - 0., 
                                     mesh.getNodalNormals()[2][4] - (1./std::sqrt(2.)) );

   RealT tol = 1.e-12;

   EXPECT_LE( n1check, tol );
   EXPECT_LE( n2check, tol );
   EXPECT_LE( n3check, tol );
   EXPECT_LE( n4check, tol );
   EXPECT_LE( n5check, tol );
   EXPECT_LE( n6check, tol );
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
