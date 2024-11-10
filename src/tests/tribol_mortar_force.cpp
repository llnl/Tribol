// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/interface/tribol.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MeshData.hpp"
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
 * the mortar contact forces between two parallel, but 
 * misaligned faces
 */
class MortarForceTest : public ::testing::Test
{
   
public:
   int numNodes;
   int numFaces;
   int numNodesPerFace;
   int numOverlapNodes;
   int dim;

   RealT* getXCoords( )
   {
      return x;
   }

   RealT* getYCoords( )
   {
      return y;
   }

   RealT* getZCoords( )
   {
      return z;
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

   void checkMortarForces( int * conn1,
                           int * conn2,
                           tribol::ContactMethod method )
   {
      // grab coordinate data
      RealT * x = this->x;
      RealT * y = this->y;
      RealT * z = this->z;

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
            SLIC_ERROR("checkMortarForces: number of nodes per face not equal to 4.");
         } 
      }

      const int mortarMeshId = 0;
      const int nonmortarMeshId = 1;

      // register mesh
      tribol::registerMesh( mortarMeshId, 1, 
                            this->numNodes,
                            conn1, cellType, 
                            x, y, z, tribol::MemorySpace::Host );
      tribol::registerMesh( nonmortarMeshId, 1, 
                            this->numNodes,
                            conn2, cellType, 
                            x, y, z, tribol::MemorySpace::Host );

      // register nodal forces
      RealT *fx1, *fy1, *fz1;
      RealT *fx2, *fy2, *fz2;

      RealT forceX1[ this->numNodes ];
      RealT forceY1[ this->numNodes ];
      RealT forceZ1[ this->numNodes ];     

      RealT forceX2[ this->numNodes ];
      RealT forceY2[ this->numNodes ];
      RealT forceZ2[ this->numNodes ];     

      fx1 = forceX1; 
      fy1 = forceY1;
      fz1 = forceZ1; 

      fx2 = forceX2;
      fy2 = forceY2;
      fz2 = forceZ2;

      // initialize force arrays
      for (int i=0; i<this->numNodes; ++i)
      {
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
      RealT *gaps, *pressures;

      gaps = new RealT [ this->numNodes ];
      pressures = new RealT [ this->numNodes ];

      // initialize gaps and pressures. Initialize all 
      // nonmortar pressures to 1.0
      for (int i=0; i<this->numNodes; ++i)
      {
         gaps[i] = 0.;
         pressures[i] = 1.;
      }

      // register nodal gaps and pressure arrays
      tribol::registerMortarGaps( nonmortarMeshId, gaps );
      tribol::registerMortarPressures( nonmortarMeshId, pressures );

      // register coupling scheme
      const int csIndex = 0;
      tribol::registerCouplingScheme( csIndex,
                                      mortarMeshId,
                                      nonmortarMeshId,
                                      tribol::SURFACE_TO_SURFACE,
                                      tribol::NO_CASE,
                                      method,
                                      tribol::FRICTIONLESS,
                                      tribol::LAGRANGE_MULTIPLIER,
                                      tribol::DEFAULT_BINNING_METHOD,
                                      tribol::ExecutionMode::Sequential );
    
      tribol::setLagrangeMultiplierOptions( csIndex, tribol::ImplicitEvalMode::MORTAR_RESIDUAL,
                                            tribol::SparseMode::MFEM_LINKED_LIST );

      // call tribol update
      RealT dt = 1.0;
      int tribol_update_err = tribol::update( 1, 1., dt );

      EXPECT_EQ( tribol_update_err, 0 );      

      // diagnostics
      auto& cs = tribol::CouplingSchemeManager::getInstance().at(csIndex);
      const auto mortarMesh = cs.getMesh1().getView();
      const auto nonmortarMesh = cs.getMesh2().getView();

      // compute the sum of the nodal forces
      RealT fx1Sum = 0.;
      RealT fy1Sum = 0.;
      RealT fz1Sum = 0.;

      RealT fx2Sum = 0.;
      RealT fy2Sum = 0.;
      RealT fz2Sum = 0.;
      for (int i=0; i<this->numNodesPerFace; ++i)
      {
         int nonmortarNodeId = nonmortarMesh.getGlobalNodeId( 0, i );
         int mortarNodeId = mortarMesh.getGlobalNodeId( 0, i );

         fx2Sum += nonmortarMesh.getResponse()[0][ nonmortarNodeId ];
         fy2Sum += nonmortarMesh.getResponse()[1][ nonmortarNodeId ];
         fz2Sum += nonmortarMesh.getResponse()[2][ nonmortarNodeId ];

         fx1Sum += mortarMesh.getResponse()[0][ mortarNodeId ];
         fy1Sum += mortarMesh.getResponse()[1][ mortarNodeId ];
         fz1Sum += mortarMesh.getResponse()[2][ mortarNodeId ];
      }

      // sum nonmortar pressure
      RealT pSum = 0.;
      for (int i=0; i<this->numNodesPerFace; ++i)
      {
         int nonmortarNodeId = nonmortarMesh.getGlobalNodeId( 0, i );
         pSum += nonmortarMesh.getNodalFields().m_node_pressure[ nonmortarNodeId ];
      }

      RealT diffX1 = std::abs(fx1Sum) - std::abs(pSum);
      RealT diffY1 = std::abs(fy1Sum) - std::abs(pSum);
      RealT diffZ1 = std::abs(fz1Sum) - std::abs(pSum);

      RealT diffX2 = std::abs(fx2Sum) - std::abs(pSum);
      RealT diffY2 = std::abs(fy2Sum) - std::abs(pSum);
      RealT diffZ2 = std::abs(fz2Sum) - std::abs(pSum);

      RealT tol = 1.e-8;
      EXPECT_LE( diffX1, tol);
      EXPECT_LE( diffY1, tol);
      EXPECT_LE( diffZ1, tol);
      EXPECT_LE( diffX2, tol);
      EXPECT_LE( diffY2, tol);
      EXPECT_LE( diffZ2, tol);

      // finalize
      tribol::finalize();

      delete [] gaps;
      delete [] pressures;

   } // end checkMortarForces()

protected:

   void SetUp() override
   {
      this->numNodes = 8;
      this->numFaces = 2;
      this->numNodesPerFace = 4;
      this->numOverlapNodes = 4;
      this->dim = 3;

      if (this->x == nullptr)
      {
         this->x = new RealT [this->numNodes];
      }
      else
      {
         delete [] this->x;
         this->x = new RealT [this->numNodes];
      }

      if (this->y == nullptr)
      {
         this->y = new RealT [this->numNodes];
      }
      else
      {
         delete [] this->y;
         this->y = new RealT [this->numNodes];
      }

      if (this->z == nullptr)
      {
         this->z = new RealT [this->numNodes];
      }
      else
      {
         delete [] this->z;
         this->z = new RealT [this->numNodes];
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
      if (this->x != nullptr)
      {
         delete [] this->x;
         this->x = nullptr;
      }
      if (this->y != nullptr)
      {
         delete [] this->y;
         this->y = nullptr;
      }
      if (this->z != nullptr)
      {
         delete [] this->z;
         this->z = nullptr;
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

   RealT* x {nullptr};
   RealT* y {nullptr};
   RealT* z {nullptr};

   RealT* xOverlap {nullptr};
   RealT* yOverlap {nullptr};
   RealT* zOverlap {nullptr};

};

TEST_F( MortarForceTest, parallel_misaligned )
{

   RealT* x = this->getXCoords();
   RealT* y = this->getYCoords();
   RealT* z = this->getZCoords();

   RealT* xOvrlp = this->getXOverlapCoords();
   RealT* yOvrlp = this->getYOverlapCoords();
   RealT* zOvrlp = this->getZOverlapCoords();

   x[0] = -1.;
   x[1] = -1.;
   x[2] =  1.;
   x[3] =  1.;

   y[0] =  1.;
   y[1] = -1.;
   y[2] = -1.;
   y[3] =  1.;

   z[0] = 0.1;
   z[1] = 0.1;
   z[2] = 0.1; 
   z[3] = 0.1;

   x[4] =  0.;
   x[5] =  2.;
   x[6] =  2.;
   x[7] =  0.;

   y[4] =  0.;
   y[5] =  0.;
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
   int conn1[ numNodesPerFace ];
   int conn2[ numNodesPerFace ];
  
   for (int i=0; i<numNodesPerFace; ++i)
   {
      conn1[i] = i;
      conn2[i] = numNodesPerFace + i;
   }

   this->checkMortarForces( &conn1[0], &conn2[0], tribol::SINGLE_MORTAR );

}

TEST_F( MortarForceTest, parallel_aligned )
{

   RealT* x = this->getXCoords();
   RealT* y = this->getYCoords();
   RealT* z = this->getZCoords();

   RealT* xOvrlp = this->getXOverlapCoords();
   RealT* yOvrlp = this->getYOverlapCoords();
   RealT* zOvrlp = this->getZOverlapCoords();

   x[0] = -1.;
   x[1] = -1.;
   x[2] =  1.;
   x[3] =  1.;

   y[0] =  1.;
   y[1] = -1.;
   y[2] = -1.;
   y[3] =  1.;

   z[0] = 0.1;
   z[1] = 0.1;
   z[2] = 0.1; 
   z[3] = 0.1;

   x[4] = -1.;
   x[5] =  1.;
   x[6] =  1.;
   x[7] = -1.;

   y[4] =  1.;
   y[5] =  1.;
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
   int conn1[ numNodesPerFace ];
   int conn2[ numNodesPerFace ];
  
   for (int i=0; i<numNodesPerFace; ++i)
   {
      conn1[i] = i;
      conn2[i] = numNodesPerFace + i;
   }

   this->checkMortarForces( &conn1[0], &conn2[0], tribol::SINGLE_MORTAR );

}

TEST_F( MortarForceTest, non_parallel_misaligned )
{

   RealT* x = this->getXCoords();
   RealT* y = this->getYCoords();
   RealT* z = this->getZCoords();

   RealT* xOvrlp = this->getXOverlapCoords();
   RealT* yOvrlp = this->getYOverlapCoords();
   RealT* zOvrlp = this->getZOverlapCoords();

   x[0] = -1.;
   x[1] = -1.;
   x[2] =  1.;
   x[3] =  1.;

   y[0] =  1.;
   y[1] = -1.;
   y[2] = -1.;
   y[3] =  1.;

   z[0] = 0.1;
   z[1] = 0.1;
   z[2] = 0.1; 
   z[3] = 0.1;

   x[4] =  0.;
   x[5] =  2.;
   x[6] =  2.;
   x[7] =  0.;

   y[4] =  0.;
   y[5] =  0.;
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
   int conn1[ numNodesPerFace ];
   int conn2[ numNodesPerFace ];
  
   for (int i=0; i<numNodesPerFace; ++i)
   {
      conn1[i] = i;
      conn2[i] = numNodesPerFace + i;
   }

   this->checkMortarForces( &conn1[0], &conn2[0], tribol::SINGLE_MORTAR );

}

TEST_F( MortarForceTest, non_parallel_aligned )
{

   RealT* x = this->getXCoords();
   RealT* y = this->getYCoords();
   RealT* z = this->getZCoords();

   RealT* xOvrlp = this->getXOverlapCoords();
   RealT* yOvrlp = this->getYOverlapCoords();
   RealT* zOvrlp = this->getZOverlapCoords();

   x[0] = -1.;
   x[1] = -1.;
   x[2] =  1.;
   x[3] =  1.;

   y[0] =  1.;
   y[1] = -1.;
   y[2] = -1.;
   y[3] =  1.;

   z[0] = 0.1;
   z[1] = 0.1;
   z[2] = 0.1; 
   z[3] = 0.1;

   x[4] = -1.;
   x[5] =  1.;
   x[6] =  1.;
   x[7] = -1.;

   y[4] =  1.;
   y[5] =  1.;
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
   int conn1[ numNodesPerFace ];
   int conn2[ numNodesPerFace ];
  
   for (int i=0; i<numNodesPerFace; ++i)
   {
      conn1[i] = i;
      conn2[i] = numNodesPerFace + i;
   }

   this->checkMortarForces( &conn1[0], &conn2[0], tribol::SINGLE_MORTAR );

}

TEST_F( MortarForceTest, parallel_simple_aligned )
{

   RealT* x = this->getXCoords();
   RealT* y = this->getYCoords();
   RealT* z = this->getZCoords();

   RealT* xOvrlp = this->getXOverlapCoords();
   RealT* yOvrlp = this->getYOverlapCoords();
   RealT* zOvrlp = this->getZOverlapCoords();

   x[0] = -1.;
   x[1] = -1.;
   x[2] =  1.;
   x[3] =  1.;

   y[0] =  1.;
   y[1] = -1.;
   y[2] = -1.;
   y[3] =  1.;

   z[0] = 0.1;
   z[1] = 0.1;
   z[2] = 0.1; 
   z[3] = 0.1;

   x[4] = -1.;
   x[5] =  1.;
   x[6] =  1.;
   x[7] = -1.;

   y[4] =  1.;
   y[5] =  1.;
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
   int conn1[ numNodesPerFace ];
   int conn2[ numNodesPerFace ];
  
   for (int i=0; i<numNodesPerFace; ++i)
   {
      conn1[i] = i;
      conn2[i] = numNodesPerFace + i;
   }

   this->checkMortarForces( &conn1[0], &conn2[0], tribol::ALIGNED_MORTAR );

}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

  axom::slic::SimpleLogger logger;                // create & initialize logger,

  result = RUN_ALL_TESTS();

  return result;
}
