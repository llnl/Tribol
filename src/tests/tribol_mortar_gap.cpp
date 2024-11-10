// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/physics/Mortar.hpp"
#include "tribol/physics/AlignedMortar.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/utils/TestUtils.hpp"

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
 * Test fixture class with some setup necessary to compute
 * the mortar gap between two parallel, but misaligned faces
 */
class MortarGapTest : public ::testing::Test
{

public:
   int numNodes;
   int numFaces;
   int numNodesPerFace;
   int numOverlapNodes;
   int dim;

   tribol::ArrayT<RealT> gaps;

   RealT* getXCoords( int id )
   {
      if (id == 0)
      {
         return x1;
      }
      else
      {
         return x2;
      }
   }

   RealT* getYCoords( int id )
   {
      if (id == 0)
      {
         return y1;
      }
      else
      {
         return y2;
      }
   }

   RealT* getZCoords( int id )
   {
      if (id == 0)
      {
         return z1;
      }
      else
      {
         return z2;
      }
   }

   RealT* getXOverlapCoords()
   {
      return xOverlap;
   }

   RealT* getYOverlapCoords()
   {
      return yOverlap;
   }

   RealT* getZOverlapCoords()
   {
      return zOverlap;
   }

   void checkMortarGaps( int * conn1,
                         int * conn2,
                         tribol::ContactMethod method )
   {
      // declare arrays to hold stacked coordinates for each
      // face used in initializing a SurfaceContactElem struct
      RealT xyz1[ this->dim * this->numNodesPerFace ];
      RealT xyz2[ this->dim * this->numNodesPerFace ];

      // declare array to hold overlap vertices used for
      // initializing a SurfaceContactElem struct
      RealT xyzOverlap[ this->dim * this->numOverlapNodes ];

      // assign pointers to arrays
      RealT* xy1 = xyz1;
      RealT* xy2 = xyz2;
      RealT* xyOverlap = xyzOverlap;

      // grab coordinate data
      RealT * x1 = this->x1;
      RealT * y1 = this->y1;
      RealT * z1 = this->z1;
      RealT * x2 = this->x2;
      RealT * y2 = this->y2;
      RealT * z2 = this->z2;
      RealT * xo = this->xOverlap;
      RealT * yo = this->yOverlap;
      RealT * zo = this->zOverlap;

      // generate stacked coordinate array
      for (int j=0; j<this->numNodesPerFace; ++j)
      {
         for (int k=0; k<this->dim; ++k)
         {
            int id = this->dim * j + k;
            //int id2 = this->dim * this->numNodesPerFace + id;
            switch (k)
            {
               case 0:
                  xy1[ id ] = x1[ j ];
                  xy2[ id ] = x2[ j ];
                  xyOverlap[ id ] = xo[ j ];
                  break;
               case 1:
                  xy1[ id ] = y1[ j ];
                  xy2[ id ] = y2[ j ];
                  xyOverlap[ id ] = yo[ j ];
                  break;
               case 2:
                  xy1[ id ] = z1[ j ];
                  xy2[ id ] = z2[ j ];
                  xyOverlap[ id ] = zo[ j ];
                  break;
            } // end switch
         } // end loop over dimension
      } // end loop over nodes

      // register the mesh with tribol
      int cellType = static_cast<int>(tribol::UNDEFINED_ELEMENT);
      switch (this->numNodesPerFace)
      {
         case 4:
         {
            cellType = (int)(tribol::LINEAR_QUAD);
            break;
         }
         default:
         {
            SLIC_ERROR("checkMortarWts: number of nodes per face not equal to 4.");
         }
      }

      const int mortarMeshId = 0;
      const int nonmortarMeshId = 1;

      tribol::registerMesh( mortarMeshId, 1,
                            this->numNodesPerFace,
                            conn1, cellType,
                            x1, y1, z1, tribol::MemorySpace::Host );
      tribol::registerMesh( nonmortarMeshId, 1,
                            this->numNodesPerFace,
                            conn2, cellType,
                            x2, y2, z2, tribol::MemorySpace::Host );

      // get instance of meshes to compute face data required for other calculations
      tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
      tribol::MeshData& mortarMesh = meshManager.at( mortarMeshId );
      tribol::MeshData& nonmortarMesh = meshManager.at( nonmortarMeshId );

      mortarMesh.computeFaceData(tribol::ExecutionMode::Sequential);
      nonmortarMesh.computeFaceData(tribol::ExecutionMode::Sequential);

      gaps.clear();
      int size = 2*this->numNodesPerFace;
      gaps.resize(size);

      for (int i=0; i<size; ++i)
      {
         gaps[i] = 0.;
      }

      tribol::registerMortarGaps( nonmortarMeshId, gaps.data() );

      nonmortarMesh.computeNodalNormals( this->dim );

      auto mortarView = mortarMesh.getView();
      auto nonmortarView = nonmortarMesh.getView();

      // instantiate SurfaceContactElem struct. Note, this object is instantiated
      // using face 1, face 2, and the set overlap polygon. Note, the mesh ids are set
      // equal to 0, and the face ids are 0 and 1, respectively.
      tribol::SurfaceContactElem elem ( this->dim, xy1, xy2, xyOverlap,
                                        this->numNodesPerFace, this->numOverlapNodes,
                                        &mortarView, &nonmortarView, 0, 0);

      // compute the mortar weights to be stored on
      // the surface contact element struct.
      switch (method)
      {
      case tribol::SINGLE_MORTAR:
         tribol::ComputeMortarWeights( elem );
         break;
      case tribol::ALIGNED_MORTAR:
         tribol::ComputeAlignedMortarWeights( elem );
         break;
      default:
         SLIC_ERROR("Unsupported contact method");
         break;
      }

      switch (method)
      {
      case tribol::SINGLE_MORTAR:
         tribol::ComputeNodalGap< tribol::SINGLE_MORTAR >( elem );
         break;
      case tribol::ALIGNED_MORTAR:
         tribol::ComputeNodalGap< tribol::ALIGNED_MORTAR >( elem );
         break;
      default:
         SLIC_ERROR("Unsupported contact method");
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

      if (this->x1 == nullptr)
      {
         this->x1 = new RealT [this->numNodes];
      }
      else
      {
         delete [] this->x1;
         this->x1 = new RealT [this->numNodes];
      }

      if (this->x2 == nullptr)
      {
         this->x2 = new RealT [this->numNodes];
      }
      else
      {
         delete [] this->x2;
         this->x2 = new RealT [this->numNodes];
      }

      if (this->y1 == nullptr)
      {
         this->y1 = new RealT [this->numNodes];
      }
      else
      {
         delete [] this->y1;
         this->y1 = new RealT [this->numNodes];
      }

      if (this->y2 == nullptr)
      {
         this->y2 = new RealT [this->numNodes];
      }
      else
      {
         delete [] this->y2;
         this->y2 = new RealT [this->numNodes];
      }

      if (this->z1 == nullptr)
      {
         this->z1 = new RealT [this->numNodes];
      }
      else
      {
         delete [] this->z1;
         this->z1 = new RealT [this->numNodes];
      }

      if (this->z2 == nullptr)
      {
         this->z2 = new RealT [this->numNodes];
      }
      else
      {
         delete [] this->z2;
         this->z2 = new RealT [this->numNodes];
      }

      if (this->xOverlap == nullptr)
      {
         this->xOverlap = new RealT [this->numOverlapNodes];
      }
      else
      {
         delete [] this->xOverlap;
         this->xOverlap = new RealT [this->numOverlapNodes];
      }

      if (this->yOverlap == nullptr)
      {
         this->yOverlap = new RealT [this->numOverlapNodes];
      }
      else
      {
         delete [] this->yOverlap;
         this->yOverlap = new RealT [this->numOverlapNodes];
      }
      if (this->zOverlap == nullptr)
      {
         this->zOverlap = new RealT [this->numOverlapNodes];
      }
      else
      {
         delete [] this->zOverlap;
         this->zOverlap = new RealT [this->numOverlapNodes];
      }
   }

   void TearDown() override
   {
      if (this->x1 != nullptr)
      {
         delete [] this->x1;
         this->x1 = nullptr;
      }
      if (this->x2 != nullptr)
      {
         delete [] this->x2;
         this->x2 = nullptr;
      }
      if (this->y1 != nullptr)
      {
         delete [] this->y1;
         this->y1 = nullptr;
      }
      if (this->y2 != nullptr)
      {
         delete [] this->y2;
         this->y2 = nullptr;
      }
      if (this->z1 != nullptr)
      {
         delete [] this->z1;
         this->z1 = nullptr;
      }
      if (this->z2 != nullptr)
      {
         delete [] this->z2;
         this->z2 = nullptr;
      }
      if (this->xOverlap != nullptr)
      {
         delete [] this->xOverlap;
         this->xOverlap = nullptr;
      }
      if (this->yOverlap != nullptr)
      {
         delete [] this->yOverlap;
         this->yOverlap = nullptr;
      }
      if (this->zOverlap != nullptr)
      {
         delete [] this->zOverlap;
         this->zOverlap = nullptr;
      }
   }

protected:

   RealT* x1 {nullptr};
   RealT* y1 {nullptr};
   RealT* z1 {nullptr};

   RealT* x2 {nullptr};
   RealT* y2 {nullptr};
   RealT* z2 {nullptr};

   RealT* xOverlap {nullptr};
   RealT* yOverlap {nullptr};
   RealT* zOverlap {nullptr};

};

TEST_F( MortarGapTest, parallel_misaligned )
{

   RealT* x1 = this->getXCoords(0);
   RealT* y1 = this->getYCoords(0);
   RealT* z1 = this->getZCoords(0);

   RealT* x2 = this->getXCoords(1);
   RealT* y2 = this->getYCoords(1);
   RealT* z2 = this->getZCoords(1);

   RealT* xOvrlp = this->getXOverlapCoords();
   RealT* yOvrlp = this->getYOverlapCoords();
   RealT* zOvrlp = this->getZOverlapCoords();

   x1[0] = -1.;
   x1[1] = -1.;
   x1[2] =  1.;
   x1[3] =  1.;

   y1[0] =  1.;
   y1[1] = -1.;
   y1[2] = -1.;
   y1[3] =  1.;

   z1[0] = 0.1;
   z1[1] = 0.1;
   z1[2] = 0.1;
   z1[3] = 0.1;

   x2[0] =  0.;
   x2[1] =  2.;
   x2[2] =  2.;
   x2[3] =  0.;

   y2[0] =  0.;
   y2[1] =  0.;
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
   int conn1[ numNodesPerFace ];
   int conn2[ numNodesPerFace ];

   for (int i=0; i<numNodesPerFace; ++i)
   {
      conn1[i] = i;
      conn2[i] = i;
   }

   this->checkMortarGaps( &conn1[0], &conn2[0], tribol::SINGLE_MORTAR );

   tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
   tribol::MeshData& nonmortarMesh = meshManager.at( 1 );

   // compute the sum of the nodal gaps
   RealT gap = 0.;
   for (int i=0; i<numNodesPerFace; ++i)
   {
      gap += nonmortarMesh.getNodalFields().m_node_gap[i];
   }

   // note the face-gap of 0.1 is hard coded based on the
   // hard-coded face coordinates in this test
   RealT gapDiff = std::abs(0.1 + gap);

   RealT tol = 1.e-8;
   EXPECT_LE( gapDiff, tol );
}

TEST_F( MortarGapTest, parallel_aligned )
{

   RealT* x1 = this->getXCoords(0);
   RealT* y1 = this->getYCoords(0);
   RealT* z1 = this->getZCoords(0);

   RealT* x2 = this->getXCoords(1);
   RealT* y2 = this->getYCoords(1);
   RealT* z2 = this->getZCoords(1);

   RealT* xOvrlp = this->getXOverlapCoords();
   RealT* yOvrlp = this->getYOverlapCoords();
   RealT* zOvrlp = this->getZOverlapCoords();

   x1[0] = 0.; //-1.;
   x1[1] = 0.; //-1.;
   x1[2] =  1.;
   x1[3] =  1.;

   y1[0] =  1.;
   y1[1] = 0.; // -1.;
   y1[2] = 0.; // -1.;
   y1[3] =  1.;

   z1[0] = 0.1;
   z1[1] = 0.1;
   z1[2] = 0.1;
   z1[3] = 0.1;

   x2[0] = 0.; // -1.;
   x2[1] =  1.;
   x2[2] =  1.;
   x2[3] = 0.; //-1.;

   y2[0] =  1.;
   y2[1] =  1.;
   y2[2] = 0.; // -1.;
   y2[3] = 0.; //-1.;

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
   int conn1[ numNodesPerFace ];
   int conn2[ numNodesPerFace ];

   for (int i=0; i<numNodesPerFace; ++i)
   {
      conn1[i] = i;
      conn2[i] = i;
   }

   this->checkMortarGaps( &conn1[0], &conn2[0], tribol::SINGLE_MORTAR );

   tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
   tribol::MeshData& nonmortarMesh = meshManager.at( 1 );

   // compute the sum of the nodal gaps
   RealT gap = 0.;
   for (int i=0; i<numNodesPerFace; ++i)
   {
      gap += nonmortarMesh.getNodalFields().m_node_gap[i];
   }

   // note the face-gap of 0.1 is hard coded based on the
   // hard-coded face coordinates in this test
   RealT gapDiff = std::abs(0.1 + gap);

   RealT tol = 1.e-8;
   EXPECT_LE( gapDiff, tol );
}

TEST_F( MortarGapTest, parallel_simple_aligned )
{

   RealT* x1 = this->getXCoords(0);
   RealT* y1 = this->getYCoords(0);
   RealT* z1 = this->getZCoords(0);

   RealT* x2 = this->getXCoords(1);
   RealT* y2 = this->getYCoords(1);
   RealT* z2 = this->getZCoords(1);

   RealT* xOvrlp = this->getXOverlapCoords();
   RealT* yOvrlp = this->getYOverlapCoords();
   RealT* zOvrlp = this->getZOverlapCoords();

   x1[0] = -1.;
   x1[1] = -1.;
   x1[2] =  1.;
   x1[3] =  1.;

   y1[0] =  1.;
   y1[1] = -1.;
   y1[2] = -1.;
   y1[3] =  1.;

   z1[0] = 0.1;
   z1[1] = 0.1;
   z1[2] = 0.2;
   z1[3] = 0.1;

   x2[0] = -1.;
   x2[1] =  1.;
   x2[2] =  1.;
   x2[3] = -1.;

   y2[0] =  1.;
   y2[1] =  1.;
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
   int conn1[ numNodesPerFace ];
   int conn2[ numNodesPerFace ];

   for (int i=0; i<numNodesPerFace; ++i)
   {
      conn1[i] = i;
      conn2[i] = i;
   }

   this->checkMortarGaps( &conn1[0], &conn2[0], tribol::ALIGNED_MORTAR );

   tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
   tribol::MeshData& nonmortarMesh = meshManager.at( 1 );

   // compute the sum of the nodal gaps
   RealT gap = 0.;
   RealT gapTest = 0;
   for (int i=0; i<numNodesPerFace; ++i)
   {
      gap += nonmortarMesh.getNodalFields().m_node_gap[i];
      gapTest += z1[i] - z2[i];
   }

   // note the face-gap of 0.1 is hard coded based on the
   // hard-coded face coordinates in this test
   RealT gapDiff = std::abs(gapTest + gap);

   RealT tol = 1.e-8;
   EXPECT_LE( gapDiff, tol );
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
