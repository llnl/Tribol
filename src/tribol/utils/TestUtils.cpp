// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/utils/TestUtils.hpp"
#include "tribol/utils/Math.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/interface/simple_tribol.hpp"

#define _USE_MATH_DEFINES
#include <cmath> // std::abs, std::cos, std::sin

// AXOM includes
#include "axom/core.hpp"
#include "axom/slic.hpp"


namespace tribol
{

////////////////////////////////
//                            //
// Class function definitions //
//                            //
////////////////////////////////
TestMesh::TestMesh()
   : mfem_mesh            (nullptr)
   , mortarMeshId         ( 0 )
   , nonmortarMeshId      ( 0 )
   , numTotalNodes        ( 0 )
   , numMortarNodes       ( 0 )
   , numNonmortarNodes    ( 0 )
   , numTotalElements     ( 0 )
   , numMortarElements    ( 0 )
   , numNonmortarElements ( 0 )
   , numTotalFaces        ( 0 )
   , numMortarFaces       ( 0 )
   , numNonmortarFaces    ( 0 )
   , numNodesPerFace      ( 0 )
   , numNodesPerElement   ( 0 ) 
   , dim                  ( 3 ) // no 2D support
   , dirNodesX1 ( nullptr )
   , dirNodesY1 ( nullptr )
   , dirNodesZ1 ( nullptr )
   , iDirValX1  ( nullptr )
   , iDirValY1  ( nullptr )
   , iDirValZ1  ( nullptr )
   , dirNodesX2 ( nullptr )
   , dirNodesY2 ( nullptr )
   , dirNodesZ2 ( nullptr )
   , iDirValX2  ( nullptr )
   , iDirValY2  ( nullptr )
   , iDirValZ2  ( nullptr )
   , presDofs1  ( nullptr )
   , presDofs2  ( nullptr )
   , faceConn1  ( nullptr )
   , faceConn2  ( nullptr )
   , elConn1    ( nullptr )
   , elConn2    ( nullptr )
   , fx1        ( nullptr )
   , fy1        ( nullptr )
   , fz1        ( nullptr )
   , fx2        ( nullptr )
   , fy2        ( nullptr )
   , fz2        ( nullptr )
   , vx1        ( nullptr )
   , vy1        ( nullptr )
   , vz1        ( nullptr )
   , vx2        ( nullptr )
   , vy2        ( nullptr )
   , vz2        ( nullptr )
   , x          ( nullptr )
   , y          ( nullptr )
   , z          ( nullptr )
   , I          ( nullptr )
   , J          ( nullptr )
   , vals       ( nullptr )
   , gaps       ( nullptr )
   , pressures  ( nullptr )
   , mortar_bulk_mod          ( nullptr )
   , mortar_element_thickness ( nullptr )
   , nonmortar_bulk_mod           ( nullptr )
   , nonmortar_element_thickness  ( nullptr )
   , registered_velocities1 (false)
   , registered_velocities2 (false)
{ }

//------------------------------------------------------------------------------
TestMesh::~TestMesh()
{
   this->clear();
}

//------------------------------------------------------------------------------
void TestMesh::clear( bool keepCoords )
{
   if (this->mfem_mesh != nullptr)
   {
      delete this->mfem_mesh;
      this->mfem_mesh = nullptr;
   }

   // coordinates
   if (!keepCoords)
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
   }
   // boundary condition nodes side 1
   if (this->dirNodesX1 != nullptr)
   {
      delete [] this->dirNodesX1;
      this->dirNodesX1 = nullptr;
   }
   if (this->dirNodesY1 != nullptr)
   {
      delete [] this->dirNodesY1;
      this->dirNodesY1 = nullptr;
   }
   if (this->dirNodesZ1 != nullptr)
   {
      delete [] this->dirNodesZ1;
      this->dirNodesZ1 = nullptr;
   }
   // boundary condition values side 1
   if (this->iDirValX1 != nullptr)
   {
      delete [] this->iDirValX1;
      this->iDirValX1 = nullptr;
   }
   if (this->iDirValY1 != nullptr)
   {
      delete [] this->iDirValY1;
      this->iDirValY1 = nullptr;
   }
   if (this->iDirValZ1 != nullptr)
   {
      delete [] this->iDirValZ1;
      this->iDirValZ1 = nullptr;
   }
   // boundary condition nodes side 2
   if (this->dirNodesX2 != nullptr)
   {
      delete [] this->dirNodesX2;
      this->dirNodesX2 = nullptr;
   }
   if (this->dirNodesY2 != nullptr)
   {
      delete [] this->dirNodesY2;
      this->dirNodesY2 = nullptr;
   }
   if (this->dirNodesZ2 != nullptr)
   {
      delete [] this->dirNodesZ2;
      this->dirNodesZ2 = nullptr;
   }
   // boundary condition values side 2
   if (this->iDirValX2 != nullptr)
   {
      delete [] this->iDirValX2;
      this->iDirValX2 = nullptr;
   }
   if (this->iDirValY2 != nullptr)
   {
      delete [] this->iDirValY2;
      this->iDirValY2 = nullptr;
   }
   if (this->iDirValZ2 != nullptr)
   {
      delete [] this->iDirValZ2;
      this->iDirValZ2 = nullptr;
   }
   // face element connectivity
   if (this->faceConn1 != nullptr)
   {
      delete [] this->faceConn1;
      this->faceConn1 = nullptr;
   }
   if (this->faceConn2 != nullptr)
   {
      delete [] this->faceConn2;
      this->faceConn2 = nullptr;
   }
   // volume element connectivity
   if (this->elConn1 != nullptr)
   {
      delete [] this->elConn1;
      this->elConn1 = nullptr;
   }
   if (this->elConn2 != nullptr)
   {
      delete [] this->elConn2;
      this->elConn2 = nullptr;
   }
   // pressure dofs
   if (this->presDofs1 != nullptr)
   {
      delete [] this->presDofs1;
      this->presDofs1 = nullptr;
   }
   if (this->presDofs2 != nullptr)
   {
      delete [] this->presDofs2;
      this->presDofs2 = nullptr;
   }
   // nodal forces side 1
   if (this->fx1 != nullptr)
   {
      delete [] this->fx1;
      this->fx1 = nullptr;
   }
   if (this->fy1 != nullptr)
   {
      delete [] this->fy1;
      this->fy1 = nullptr;
   }
   if (this->fz1 != nullptr)
   {
      delete [] this->fz1;
      this->fz1 = nullptr;
   }
   // nodal forces side 2
   if (this->fx2 != nullptr)
   {
      delete [] this->fx2;
      this->fx2 = nullptr;
   }
   if (this->fy2 != nullptr)
   {
      delete [] this->fy2;
      this->fy2 = nullptr;
   }
   if (this->fz2 != nullptr)
   {
      delete [] this->fz2;
      this->fz2 = nullptr;
   }
   // nodal velocities side 1
   if (this->vx1 != nullptr)
   {
      delete [] this->vx1;
      this->vx1 = nullptr;
   }
   if (this->vy1 != nullptr)
   {
      delete [] this->vy1;
      this->vy1 = nullptr;
   }
   if (this->vz1 != nullptr)
   {
      delete [] this->vz1;
      this->vz1 = nullptr;
   }
   // nodal velocities side 2
   if (this->vx2 != nullptr)
   {
      delete [] this->vx2;
      this->vx2 = nullptr;
   }
   if (this->vy2 != nullptr)
   {
      delete [] this->vy2;
      this->vy2 = nullptr;
   }
   if (this->vz2 != nullptr)
   {
      delete [] this->vz2;
      this->vz2 = nullptr;
   }
   // nodal gaps and pressures
   if (this->gaps != nullptr)
   {
      delete [] this->gaps;
      this->gaps = nullptr;
   }
   if (this->pressures != nullptr)
   {
      delete [] this->pressures;
      this->pressures = nullptr;
   }
   // sparse matrix data
   if (this->I != nullptr)
   {
      delete [] this->I;
      this->I = nullptr;
   }
   if (this->J != nullptr)
   {
      delete [] this->J;
      this->J = nullptr;
   }
   if (this->vals != nullptr)
   {
      delete [] this->vals;
      this->vals = nullptr;
   }

   // penalty data
   if (this->mortar_bulk_mod != nullptr)
   {
      delete [] this->mortar_bulk_mod;
      this->mortar_bulk_mod = nullptr;
   }
   if (this->nonmortar_bulk_mod != nullptr)
   {
      delete [] this->nonmortar_bulk_mod;
      this->nonmortar_bulk_mod = nullptr;
   }
   if (this->mortar_element_thickness != nullptr)
   {
      delete [] this->mortar_element_thickness;
      this->mortar_element_thickness = nullptr;
   }
   if (this->nonmortar_element_thickness != nullptr)
   {
      delete [] this->nonmortar_element_thickness;
      this->nonmortar_element_thickness = nullptr;
   }

} // end TestMesh::clear()

//------------------------------------------------------------------------------
void TestMesh::setupContactMeshHex( int numElemsX1, int numElemsY1, int numElemsZ1, 
                                    RealT xMin1, RealT yMin1, RealT zMin1,
                                    RealT xMax1, RealT yMax1, RealT zMax1,
                                    int numElemsX2, int numElemsY2, int numElemsZ2,
                                    RealT xMin2, RealT yMin2, RealT zMin2, 
                                    RealT xMax2, RealT yMax2, RealT zMax2,
                                    RealT thetaMortar, RealT thetaNonmortar )
{
   // NOTE: ONLY CONTACT INTERACTIONS IN THE Z-DIRECTION ARE SUPPORTED 
   // AT THE MOMENT.

   // allocate mesh data arrays
   this->cellType = (int)(tribol::LINEAR_QUAD);
   this->numNodesPerElement = 8; // hard coded for hex8 elements
   this->numNodesPerFace = 4; // hard code for quad4 faces
   int numElementsBlock1, numNodesBlock1;
   int numElementsBlock2, numNodesBlock2;

   this->mortarMeshId = 0;
   this->nonmortarMeshId = 1;

   // ASSUMING contact is in the Z-Direction, check to make sure that the 
   // gap of interpenetration is not greater than either block's element 
   // dimension
   RealT h1 = (zMax1 - zMin1) / numElemsZ1;
   RealT h2 = (zMax2 - zMin2) / numElemsZ2;
   RealT mesh_gap = zMin2 - zMax1;

   SLIC_ERROR_IF( mesh_gap < -h1, "TestMesh::setupContactMeshHex(): " << 
                  "Initial mesh configuration has a gap greater than the " << 
                  "element thickness in block 1.");

   SLIC_ERROR_IF( mesh_gap < -h2, "TestMesh::setupContactMeshHex(): " << 
                  "Initial mesh configuration has a gap greater than the " << 
                  "element thickness in block 2.");

   numElementsBlock1       = numElemsX1 * numElemsY1 * numElemsZ1;
   numNodesBlock1          = (numElemsX1+1) * (numElemsY1+1) * (numElemsZ1+1);
   this->numMortarFaces    = numElemsX1 * numElemsY1;
   numElementsBlock2       = numElemsX2 * numElemsY2 * numElemsZ2;
   numNodesBlock2          = (numElemsX2+1) * (numElemsY2+1) * (numElemsZ2+1);
   this->numNonmortarFaces = numElemsX2 * numElemsY2;

   this->numMortarNodes           = numNodesBlock1;
   this->numNonmortarNodes        = numNodesBlock2;
   this->numNonmortarSurfaceNodes = (numElemsX2+1) * (numElemsY2+1);
   this->numTotalNodes            = numNodesBlock1 + numNodesBlock2;
   this->numMortarElements        = numElementsBlock1;
   this->numNonmortarElements     = numElementsBlock2;
   this->numTotalElements         = numElementsBlock1 + numElementsBlock2;
   this->numTotalFaces            = this->numNonmortarFaces + this->numMortarFaces;

   this->elConn1   = new int[ this->numNodesPerElement * this->numMortarElements ];
   this->elConn2   = new int[ this->numNodesPerElement * this->numNonmortarElements ];
   this->faceConn1 = new int[ this->numNodesPerFace * this->numMortarFaces ];
   this->faceConn2 = new int[ this->numNodesPerFace * this->numNonmortarFaces ];
   this->x = new RealT[ this->numTotalNodes ];
   this->y = new RealT[ this->numTotalNodes ];
   this->z = new RealT[ this->numTotalNodes ];

   // setup mesh nodal coordinate arrays
   int ndOffset;
   int numElemsX, numElemsY, numElemsZ;
   RealT xMax, yMax, zMax, xMin, yMin, zMin;
   RealT theta;
   int * elConn, * faceConn;
   for ( int iblk = 0; iblk<2; ++iblk)
   {
      if ( iblk == 0)
      {
         ndOffset = 0;
         numElemsX = numElemsX1;
         numElemsY = numElemsY1;
         numElemsZ = numElemsZ1;
         xMin = xMin1;
         yMin = yMin1;
         zMin = zMin1;
         xMax = xMax1;
         yMax = yMax1;
         zMax = zMax1;
         elConn = this->elConn1;
         faceConn = this->faceConn1;
         theta = thetaMortar;
      }
      else
      {
         ndOffset = numNodesBlock1;
         numElemsX = numElemsX2;
         numElemsY = numElemsY2;
         numElemsZ = numElemsZ2;
         xMin = xMin2;
         yMin = yMin2;
         zMin = zMin2;
         xMax = xMax2;
         yMax = yMax2;
         zMax = zMax2;
         elConn = this->elConn2;
         faceConn = this->faceConn2;
         theta = thetaNonmortar;
      }

      int numNodesX = numElemsX + 1;
      int numNodesY = numElemsY + 1;
      int numNodesZ = numElemsZ + 1;
      RealT hx = (xMax - xMin) / numElemsX;
      RealT hy = (yMax - yMin) / numElemsY;
      RealT hz = (zMax - zMin) / numElemsZ;

      // compute mesh coordinates
      int ctr = 0;
      for (int k=0; k<numNodesZ; ++k)
      {
         for (int j=0; j<numNodesY; ++j)
         {
            for (int i=0; i<numNodesX; ++i)
            {
               int idx = ndOffset + ctr;
               RealT xyz[3];

               // compute coordinates
               xyz[0] = xMin + i * hx;
               xyz[1] = yMin + j * hy;
               xyz[2] = zMin + k * hz;

               // for non-boundary faces, apply coordinate rotation about z-axis

               bool rotationY = (j>0 && j<(numNodesY-1)) ? true : false;
               bool rotationX = (i>0 && i<(numNodesX-1)) ? true : false;

               if (rotationX && rotationY)
               {
                  RealT rot0 = std::cos(theta * M_PI/180);
                  RealT rot1 = -std::sin(theta * M_PI/180);
                  RealT rot2 = -rot1;
                  RealT rot3 = rot0;
                  
                  RealT x_temp = xyz[0]*rot0 + xyz[1]*rot1;
                  RealT y_temp = xyz[0]*rot2 + xyz[1]*rot3;
                  RealT z_temp = xyz[2];

                  xyz[0] = x_temp;
                  xyz[1] = y_temp;
                  xyz[2] = z_temp;
               }
         
               this->x[ idx ] = xyz[0];
               this->y[ idx ] = xyz[1];
               this->z[ idx ] = xyz[2];
               ++ctr;
            } // end loop over x nodes
         } // end loop over y nodes
      } // end loop over z nodes

      // populate element connectivity arrays
      ctr = 0;
      for (int k=0; k<numElemsZ; ++k)
      {
         for (int j=0; j<numElemsY; ++j)
         {
            for (int i=0; i<numElemsX; ++i)
            {
               int zIdOffset = k * (numNodesX * numNodesY);
               int yIdOffset = j * numNodesX;
               int icr = ndOffset + zIdOffset + yIdOffset + i;
               elConn[ this->numNodesPerElement * ctr ]    = icr;
               elConn[ this->numNodesPerElement * ctr + 1] = icr + 1;
               elConn[ this->numNodesPerElement * ctr + 2] = icr + numNodesX + 1;
               elConn[ this->numNodesPerElement * ctr + 3] = icr + numNodesX;

               int lclZOffset = icr + numNodesX * numNodesY;
               elConn[ this->numNodesPerElement * ctr + 4] = lclZOffset;
               elConn[ this->numNodesPerElement * ctr + 5] = lclZOffset + 1;
               elConn[ this->numNodesPerElement * ctr + 6] = lclZOffset + numNodesX + 1;
               elConn[ this->numNodesPerElement * ctr + 7] = lclZOffset + numNodesX;
               ++ctr;

            } // end loop over x elements
         } // end loop over y elements
      } // end loop over z elements

      // populate contact surface connectivity
      // Note: element connectivity is not necessarily consistent with outward unit normal at 
      //       contact faces
      ctr = 0;
      for (int j=0; j<numElemsY; ++j)
      { 
         for (int i=0; i<numElemsX; ++i)
         {
            if (ndOffset == 0)
            {
               int yIdOffset = j * numNodesX;
               int icr = (numElemsX+1)*(numElemsY+1)*(numElemsZ) + yIdOffset + i; // top surface
               faceConn[ this->numNodesPerFace * ctr ]     = icr;
               faceConn[ this->numNodesPerFace * ctr + 1 ] = icr + 1;
               faceConn[ this->numNodesPerFace * ctr + 2 ] = icr + numNodesX + 1;
               faceConn[ this->numNodesPerFace * ctr + 3 ] = icr + numNodesX;
               ++ctr;
            }
            if (ndOffset != 0) // reorient nonmortar face connectivity per outward unit normal requirement
            {
               int yIdOffset = j * numNodesX;
               int icr = ndOffset + yIdOffset + i; // bottom surface
               faceConn[ this->numNodesPerFace * ctr ]     = icr;
               faceConn[ this->numNodesPerFace * ctr + 1 ] = icr + numNodesX;
               faceConn[ this->numNodesPerFace * ctr + 2 ] = icr + numNodesX + 1;
               faceConn[ this->numNodesPerFace * ctr + 3 ] = icr + 1;
               ++ctr;
            }
         }
      }

   } // end loop over blocks

   this->mesh_constructed = true;

} // end setupContactMeshHex()

//------------------------------------------------------------------------------
void TestMesh::setupContactMeshTet( int numElemsX1, int numElemsY1, int numElemsZ1,
                                    RealT xMin1, RealT yMin1, RealT zMin1,
                                    RealT xMax1, RealT yMax1, RealT zMax1,
                                    int numElemsX2, int numElemsY2, int numElemsZ2,
                                    RealT xMin2, RealT yMin2, RealT zMin2,
                                    RealT xMax2, RealT yMax2, RealT zMax2,
                                    RealT thetaMortar, RealT thetaNonmortar )
{
   // NOTE: ONLY CONTACT INTERACTIONS IN THE Z-DIRECTION ARE SUPPORTED 
   // AT THE MOMENT.

   // Construct hex mesh first
   setupContactMeshHex( numElemsX1, numElemsY1, numElemsZ1,
                        xMin1, yMin1, zMin1,
                        xMax1, yMax1, zMax1,
                        numElemsX2, numElemsY2, numElemsZ2,
                        xMin2, yMin2, zMin2,
                        xMax2, yMax2, zMax2,
                        thetaMortar, thetaNonmortar );

   // allocate temporary storage for tet mesh data while pulling 
   // from hex mesh data
   int numTetsPerHex          = 6;
   this->mesh_constructed     = false; // will reset after tet mesh has been constructed
   this->cellType = (int)(tribol::LINEAR_TRIANGLE);
   this->numNodesPerFace      = 3; // linear triangle faces
   this->numNodesPerElement   = 4; // linear four node tets
   this->numTotalElements     = numTetsPerHex * this->numTotalElements; // 6 tets per hex
   this->numMortarElements    = numTetsPerHex * numElemsX1 * numElemsY1 * numElemsZ1; 
   this->numNonmortarElements = numTetsPerHex * numElemsX2 * numElemsY2 * numElemsZ2;
   this->numMortarFaces       = 2 * numElemsX1 * numElemsY1; // only on contact surface
   this->numNonmortarFaces    = 2 * numElemsX2 * numElemsY2; // only on contact surface
   this->numMortarNodes       = (numElemsX1+1) * (numElemsY1+1) * (numElemsZ1+1); 
   this->numNonmortarNodes    = (numElemsX2+1) * (numElemsY2+1) * (numElemsZ2+1);
   this->numTotalNodes        = this->numMortarNodes + this->numNonmortarNodes; 
   this->numTotalFaces        = this->numMortarFaces + this->numNonmortarFaces; 

   // clear prior mesh, but keep same nodal coordinate arrays
   // Note, at this point nodal data arrays should not have been allocated 
   // on the new hex mesh, but this makes sure of that
   bool keep_coords = true;
   this->clear(keep_coords);

   allocIntArray( &this->faceConn1, this->numNodesPerFace*this->numMortarFaces, -1 );
   allocIntArray( &this->faceConn2, this->numNodesPerFace*this->numNonmortarFaces, -1 ); 
   allocIntArray( &this->elConn1, this->numNodesPerElement*this->numMortarElements, -1 );
   allocIntArray( &this->elConn2, this->numNodesPerElement*this->numNonmortarElements, -1 );
   
   // tet element connectivity
   int ndOffset;
   int numElemsX, numElemsY, numElemsZ;
   int * elConn, * faceConn;
   for ( int iblk = 0; iblk<2; ++iblk)
   {
      if ( iblk == 0 )
      {
         ndOffset  = 0;
         numElemsX = numElemsX1;
         numElemsY = numElemsY1;
         numElemsZ = numElemsZ1;
         elConn    = this->elConn1;
         faceConn  = this->faceConn1;
      }
      else
      {
         ndOffset  = this->numMortarNodes;
         numElemsX = numElemsX2;
         numElemsY = numElemsY2;
         numElemsZ = numElemsZ2;
         elConn    = this->elConn2;
         faceConn  = this->faceConn2;
      }

      int numNodesX = numElemsX + 1;
      int numNodesY = numElemsY + 1;

      // populate element connectivity arrays
      int ctr = 0;
      for (int k=0; k<numElemsZ; ++k) // loop over row in z-direction
      {
         for (int j=0; j<numElemsY; ++j) // loop over row in y-direction
         {
            for (int i=0; i<numElemsX; ++i) // loop over elements in x-direction
            {
               // same node ids and offsets as hex mesh
               int zIdOffset = k * (numNodesX * numNodesY); // offset for z-direction node ids
               int yIdOffset = j * numNodesX; // offset for y-direction node ids
               int icr = ndOffset + zIdOffset + yIdOffset + i; // first node starting counter
               int lclZOffset = icr + numNodesX * numNodesY; // local element z-offset for top nodes of hex

               // write out the 6 tet elements per this hex  // code for each local hex node
               // local node numbering for each hex          // 1) icr;
               // 1)  1, 5, 6, 4                             // 2) icr + 1;
               // 2)  1, 6, 2, 4                             // 3) icr + numNodesX + 1;
               // 3)  3, 2, 6, 4                             // 4) icr + numNodesX;
               // 4)  3, 6, 7, 4                             // 5) lclZOffset;
               // 5)  8, 7, 6, 4                             // 6) lclZOffset + 1;
               // 6)  8, 6, 5, 4                             // 7) lclZOffset + numNodesX + 1;
               //                                            // 8) lclZOffset + numNodesX;

               // local tet element 1
               elConn[ this->numNodesPerElement * ctr ]    = icr;
               elConn[ this->numNodesPerElement * ctr + 1] = lclZOffset;
               elConn[ this->numNodesPerElement * ctr + 2] = lclZOffset + 1;
               elConn[ this->numNodesPerElement * ctr + 3] = icr + numNodesX;
               ++ctr; // incrememnt for next local tet

               // local tet element 2 
               elConn[ this->numNodesPerElement * ctr ]    = icr;
               elConn[ this->numNodesPerElement * ctr + 1] = lclZOffset + 1;
               elConn[ this->numNodesPerElement * ctr + 2] = icr + 1;
               elConn[ this->numNodesPerElement * ctr + 3] = icr + numNodesX;
               ++ctr; // incrememnt for next local tet

               // local tet element 3
               elConn[ this->numNodesPerElement * ctr ]    = icr + numNodesX + 1;
               elConn[ this->numNodesPerElement * ctr + 1] = icr + 1;
               elConn[ this->numNodesPerElement * ctr + 2] = lclZOffset + 1;
               elConn[ this->numNodesPerElement * ctr + 3] = icr + numNodesX;
               ++ctr; // incrememnt for next local tet
 
               // local tet element 4
               elConn[ this->numNodesPerElement * ctr ]    = icr + numNodesX + 1;
               elConn[ this->numNodesPerElement * ctr + 1] = lclZOffset + 1; 
               elConn[ this->numNodesPerElement * ctr + 2] = lclZOffset + numNodesX + 1; 
               elConn[ this->numNodesPerElement * ctr + 3] = icr + numNodesX;
               ++ctr; // incrememnt for next local tet

               // local tet element 5 
               elConn[ this->numNodesPerElement * ctr ]    = lclZOffset + numNodesX;
               elConn[ this->numNodesPerElement * ctr + 1] = lclZOffset + numNodesX + 1;
               elConn[ this->numNodesPerElement * ctr + 2] = lclZOffset + 1;
               elConn[ this->numNodesPerElement * ctr + 3] = icr + numNodesX;
               ++ctr; // incrememnt for next local tet

               // local tet element 6
               elConn[ this->numNodesPerElement * ctr ]    = lclZOffset + numNodesX;
               elConn[ this->numNodesPerElement * ctr + 1] = lclZOffset + 1;
               elConn[ this->numNodesPerElement * ctr + 2] = lclZOffset;
               elConn[ this->numNodesPerElement * ctr + 3] = icr + numNodesX;
               ++ctr; // incrememnt for next local tet
               
            } // end loop over x elements
         } // end loop over y elements
      } // end loop over z elements

      // populate contact surface connectivity
      // Note: element connectivity is not necessarily consistent with outward unit normal at 
      //       contact faces
      ctr = 0;
      for (int j=0; j<numElemsY; ++j)
      { 
         for (int i=0; i<numElemsX; ++i)
         {
            if (ndOffset == 0) // top surface of mortar bottom block
            {
               int yIdOffset = j * numNodesX;
               int icr = (numElemsX+1)*(numElemsY+1)*(numElemsZ) + yIdOffset + i; // top surface node id

               // local tet face connectivity for top surface of bottom block
               // 2 tet faces per hex face   // code for each local hex node
               // local el 1) 6, 7, 8        5) icr
               // local el 2) 5, 6, 8        6) icr + 1
               //                            7) icr + numNodesX + 1
               //                            8) icr + numNodesX
               
               // local tet face 1
               faceConn[ this->numNodesPerFace * ctr ]     = icr + 1;
               faceConn[ this->numNodesPerFace * ctr + 1 ] = icr + numNodesX + 1;
               faceConn[ this->numNodesPerFace * ctr + 2 ] = icr + numNodesX;
               ++ctr; // increment counter to next tet face

               // local tet face 2
               faceConn[ this->numNodesPerFace * ctr ]     = icr;
               faceConn[ this->numNodesPerFace * ctr + 1 ] = icr + 1;
               faceConn[ this->numNodesPerFace * ctr + 2 ] = icr + numNodesX;
               ++ctr; // increment counter to next tet face

            }
            // bottom surface of nonmortar top block
            // reorient nonmortar face connectivity per outward unit normal requirement
            if (ndOffset != 0) 
            {
               int yIdOffset = j * numNodesX;
               int icr = ndOffset + yIdOffset + i; // bottom surface node id

               // local tet face connectivity for top block
               // 2 tet faces per hex face   // code for each local hex node
               // local el 1) 1, 4, 2        1) icr
               // local el 2) 2, 4, 3        2) icr + 1;
               //                            3) icr + numNodesX + 1
               //                            4) icr + numNodesX; 

               // local tet face 1
               faceConn[ this->numNodesPerFace * ctr ]     = icr;
               faceConn[ this->numNodesPerFace * ctr + 1 ] = icr + numNodesX;
               faceConn[ this->numNodesPerFace * ctr + 2 ] = icr + 1; 
               ++ctr; // increment counter to next tet face

               // local tet face 2
               faceConn[ this->numNodesPerFace * ctr ]     = icr + 1;
               faceConn[ this->numNodesPerFace * ctr + 1 ] = icr + numNodesX;
               faceConn[ this->numNodesPerFace * ctr + 2 ] = icr + numNodesX + 1;
               ++ctr; // increment counter to next tet face

            }
         } // end loop over x-direction elements
      } // end loop over y-direction elements

   } // end for loop over blocks
   
   this->mesh_constructed = true;

} // end setupContactMeshTet()
//------------------------------------------------------------------------------
void TestMesh::allocateAndSetVelocities( IndexT mesh_id, RealT valX, RealT valY, RealT valZ )
{
   // Check that mesh ids are not the same. The TestMesh class was built around 
   // testing the mortar method with Lagrange multiplier enforcement, which does not 
   // support auto contact.
   SLIC_ERROR_IF( this->mortarMeshId == this->nonmortarMeshId, 
                  "TestMesh::allocateAndSetVelocities(): please set unique " << 
                  "mortarMeshId and nonmortarMeshId prior to calling this routine.");

   // check to see if pointers have been set
   bool deleteVels = false;
   if (mesh_id == this->mortarMeshId)
   {
      if (this->vx1 != nullptr)
      {
         delete [] this->vx1;
         deleteVels = true;
      }
      if (this->vy1 != nullptr)
      {
         delete [] this->vy1;
         deleteVels = true;
      }
      if (this->vz1 != nullptr)
      {
         delete [] this->vz1;
         deleteVels = true;
      }

      allocRealArray( &this->vx1, this->numTotalNodes, valX );
      allocRealArray( &this->vy1, this->numTotalNodes, valY );
      allocRealArray( &this->vz1, this->numTotalNodes, valZ );

      registered_velocities1 = true;
   }
   else if (mesh_id == this->nonmortarMeshId)
   {
      if (this->vx2 != nullptr)
      {
         delete [] this->vx2;
         deleteVels = true;
      }
      if (this->vy2 != nullptr)
      {
         delete [] this->vy2;
         deleteVels = true;
      }
      if (this->vz2 != nullptr)
      {
         delete [] this->vz2;
         deleteVels = true;
      }

      allocRealArray( &this->vx2, this->numTotalNodes, valX );
      allocRealArray( &this->vy2, this->numTotalNodes, valY );
      allocRealArray( &this->vz2, this->numTotalNodes, valZ );

      registered_velocities2 = true;
   }
   else
   {
      SLIC_ERROR( "TestMesh::allocateAndSetVelocities(): " << 
                  "not a valid mesh id." );
   }

   SLIC_DEBUG_IF( deleteVels, "TestMesh::allocateAndSetVelocities(): " << 
                       "a velocity array has been deleted and reallocated." );

} // end TestMesh::allocateAndSetVelocities()

//------------------------------------------------------------------------------
void TestMesh::allocateAndSetBulkModulus( IndexT mesh_id, RealT val )
{
   // Check that mesh ids are the same. The TestMesh class was built around 
   // testing the mortar method with Lagrange multiplier enforcement, which does 
   // not support auto contact.
   SLIC_ERROR_IF( this->mortarMeshId == this->nonmortarMeshId, 
                  "TestMesh::allocateAndSetVelocities(): please set unique " << 
                  "mortarMeshId and nonmortarMeshId prior to calling this routine.");

   // check to see if pointers have been set
   bool deleteData = false;
   if (mesh_id == this->mortarMeshId)
   {
      if (this->mortar_bulk_mod != nullptr)
      {
         delete [] this->mortar_bulk_mod;
         deleteData = true;
      }

      allocRealArray( &this->mortar_bulk_mod, this->numMortarFaces, val );
   }
   else if (mesh_id == this->nonmortarMeshId)
   {
      if (this->nonmortar_bulk_mod != nullptr)
      {
         delete [] this->nonmortar_bulk_mod;
         deleteData = true;
      }

      allocRealArray( &this->nonmortar_bulk_mod, this->numNonmortarFaces, val );
   }
   else
   {
      SLIC_ERROR( "TestMesh::allocateAndSetBulkModulus(): " << 
                  "not a valid mesh id." );
   }

   SLIC_DEBUG_IF( deleteData, "TestMesh::allocateAndSetBulkModulus(): " << 
                  "a bulk modulus array has been deleted and reallocated." );

} // end TestMesh::allocateAndSetBulkModulus()

//------------------------------------------------------------------------------
void TestMesh::allocateAndSetElementThickness( IndexT mesh_id, RealT t )
{
   // check to see if pointers have been set
   bool deleteData = false; 
   if (mesh_id == this->mortarMeshId)
   {
      if (this->mortar_element_thickness != nullptr)
      {
         delete [] this->mortar_element_thickness;
         deleteData = true;
      }

      allocRealArray( &this->mortar_element_thickness, this->numMortarFaces, t );
   }
   else if (mesh_id == this->nonmortarMeshId)
   {
      if (this->nonmortar_element_thickness != nullptr)
      {
         delete [] this->nonmortar_element_thickness;
         deleteData = true;
      }

      allocRealArray( &this->nonmortar_element_thickness, this->numNonmortarFaces, t );
   }
   else
   {
      SLIC_ERROR( "TestMesh::allocateAndSetElementThickness(): " << 
                  "not a valid mesh id." );
   }

   SLIC_DEBUG_IF( deleteData, "TestMesh::allocateAndSetElementThickness(): " << 
                  "an element thickness array has been deleted and reallocated." );

} // end TestMesh::allocateAndSetElementThickness()

//------------------------------------------------------------------------------
int TestMesh::simpleTribolSetupAndUpdate( ContactMethod method,
                                          EnforcementMethod TRIBOL_UNUSED_PARAM(enforcement),
                                          ContactModel TRIBOL_UNUSED_PARAM(model),
                                          ContactCase TRIBOL_UNUSED_PARAM(contact_case),
                                          bool TRIBOL_UNUSED_PARAM(visualization),
                                          TestControlParameters& params )
{
   SLIC_ERROR_IF( !this->mesh_constructed, "TestMesh::simpleTribolSetupAndUpdate(): " <<
                  "must construct hex or tet mesh prior to calling this routine." );

   // grab coordinate data
   RealT * x = this->x;
   RealT * y = this->y;
   RealT * z = this->z;

   switch (method)
   {
      case SINGLE_MORTAR:
      case ALIGNED_MORTAR:
      {
         // allocate gaps and pressures with length of total mesh to allow use 
         // of global connectivity for indexing
         allocRealArray( &this->gaps, this->numTotalNodes, 0. );
         allocRealArray( &this->pressures, this->numTotalNodes, 1. );
         break;
      }
      case MORTAR_WEIGHTS:
      {
         allocRealArray( &this->gaps, this->numTotalNodes, 0. );
         this->pressures = nullptr;
         break;
      }
      default:
      {
         // no-op
         break;
      }
   } // end switch on method
 
   const RealT area_frac = 1.e-03;

   SimpleCouplingSetup( this->dim,
                        this->cellType,
                        method,
                        this->numMortarFaces,
                        this->numTotalNodes,
                        this->faceConn1,
                        x, y, z,
                        this->numNonmortarFaces,
                        this->numTotalNodes,
                        this->faceConn2,
                        x, y, z,
                        area_frac,
                        this->gaps, 
                        this->pressures);
 
   int err = Update( params.dt );

   return err;

} // end simpleTribolSetupAndUpdate()

//------------------------------------------------------------------------------
int TestMesh::tribolSetupAndUpdate( ContactMethod method,
                                    EnforcementMethod enforcement,
                                    ContactModel model,
                                    ContactCase contact_case,
                                    bool visualization,
                                    TestControlParameters& params )
{
   // grab coordinate data
   RealT * x = this->x;
   RealT * y = this->y;
   RealT * z = this->z;

   // register mesh. Note: Tribol will still work with global integer 
   // ids for the connectivity and array lengths of numTotalNodes. 
   registerMesh( this->mortarMeshId, this->numMortarFaces, 
                 this->numTotalNodes,
                 this->faceConn1, this->cellType, x, y, z, MemorySpace::Host );
   registerMesh( this->nonmortarMeshId, this->numNonmortarFaces, 
                 this->numTotalNodes,
                 this->faceConn2, this->cellType, x, y, z, MemorySpace::Host );

   // register nodal forces. Note, I was getting a seg fault when 
   // registering the same pointer to a single set of force arrays 
   // for both calls to tribol::registerNodalResponse(). As a result,
   // I created two sets of nodal force arrays with their own pointers 
   // to the data that are registered with tribol and there is no longer
   // a seg fault.
   allocRealArray( &this->fx1, this->numTotalNodes, 0. );
   allocRealArray( &this->fy1, this->numTotalNodes, 0. );
   allocRealArray( &this->fz1, this->numTotalNodes, 0. );
   allocRealArray( &this->fx2, this->numTotalNodes, 0. );
   allocRealArray( &this->fy2, this->numTotalNodes, 0. );
   allocRealArray( &this->fz2, this->numTotalNodes, 0. );

   registerNodalResponse( this->mortarMeshId, 
                          this->fx1, 
                          this->fy1, 
                          this->fz1 );

   registerNodalResponse( this->nonmortarMeshId, 
                          this->fx2, 
                          this->fy2, 
                          this->fz2 );

   // register nodal velocities
   if (registered_velocities1)
   {
      registerNodalVelocities( this->mortarMeshId,
                               this->vx1, 
                               this->vy1,
                               this->vz1 );
   }

   if (registered_velocities2)
   {
      registerNodalVelocities( this->nonmortarMeshId,
                               this->vx2, 
                               this->vy2,
                               this->vz2 );
   }

   // register nodal pressure and nodal gap array for the nonmortar mesh
   // for mortar based methods
   switch (method)
   {
      case SINGLE_MORTAR:
      case ALIGNED_MORTAR:
      {
         allocRealArray( &this->gaps, this->numTotalNodes, 0. );
         allocRealArray( &this->pressures, this->numTotalNodes, 1. );

         // register nodal gaps and pressures
         registerMortarGaps( this->nonmortarMeshId, this->gaps );
         registerMortarPressures( this->nonmortarMeshId, this->pressures );
         break;
      }
      case MORTAR_WEIGHTS:
      {
         allocRealArray( &this->gaps, this->numTotalNodes, 0. );
         this->pressures = nullptr;

         registerMortarGaps( this->nonmortarMeshId, this->gaps );
         break;
      }
      default:
      {
         // no-op
      }
   } // end switch on method

   // if enforcement is penalty, register penalty parameters
   if (enforcement == PENALTY)
   {
      if (!params.penalty_ratio)
      {
         setKinematicConstantPenalty( this->mortarMeshId, params.const_penalty );
         setKinematicConstantPenalty( this->nonmortarMeshId, params.const_penalty );
      }
      else
      {
         if (this->mortar_bulk_mod == nullptr)
         {
            SLIC_DEBUG_ROOT( "TestMesh::tribolSetupAndUpdate(): " <<
                             "mortar_bulk_mod not set; registering default value." );
            this->mortar_bulk_mod = new RealT[ this->numMortarFaces ];
            for (int i=0; i<this->numMortarFaces; ++i)
            {
               this->mortar_bulk_mod[i] = params.const_penalty; // non-physical for testing 
            }
         }
         if (this->mortar_element_thickness == nullptr)
         {
            SLIC_DEBUG_ROOT( "TestMesh::tribolSetupAndUpdate(): " <<
                             "mortar_element_thickness not set; registering default value." );
            this->mortar_element_thickness = new RealT[ this->numMortarFaces ];
            for (int i=0; i<this->numMortarFaces; ++i)
            {
               this->mortar_element_thickness[i] = 1.; // non-physical for testing
            }
         }

         if (this->nonmortar_bulk_mod == nullptr)
         { 
            SLIC_DEBUG_ROOT( "TestMesh::tribolSetupAndUpdate(): " <<
                             "nonmortar_bulk_mod not set; registering default value." );
            this->nonmortar_bulk_mod = new RealT[ this->numNonmortarFaces ];
            for (int i=0; i<this->numNonmortarFaces; ++i)
            {
               this->nonmortar_bulk_mod[i] = params.const_penalty; // non-physical for testing
            }
         }
         if (this->nonmortar_element_thickness == nullptr)
         {  
            SLIC_DEBUG_ROOT( "TestMesh::tribolSetupAndUpdate(): " <<
                             "nonmortar_element_thickness not set; registering default value." );
            this->nonmortar_element_thickness = new RealT[ this->numNonmortarFaces ];
            for (int i=0; i<this->numNonmortarFaces; ++i)
            {
               this->nonmortar_element_thickness[i] = 1.; // non-physical for testing
            }
         }

         // register mortar penalty data
         registerRealElementField( this->mortarMeshId, BULK_MODULUS, 
                                   this->mortar_bulk_mod );
         registerRealElementField( this->mortarMeshId, ELEMENT_THICKNESS,
                                   this->mortar_element_thickness );
  
         // register nonmortar penalty data
         registerRealElementField( this->nonmortarMeshId, BULK_MODULUS, 
                                   this->nonmortar_bulk_mod );
         registerRealElementField( this->nonmortarMeshId, ELEMENT_THICKNESS,
                                   this->nonmortar_element_thickness );
      } // end if-penalty-ratio

      // check for gap rate penalty enforcement
      if (params.constant_rate_penalty)
      {
         setRateConstantPenalty( this->mortarMeshId, params.constant_rate_penalty );
         setRateConstantPenalty( this->nonmortarMeshId, params.constant_rate_penalty );
      }
      else if (params.percent_rate_penalty)
      {
         setRatePercentPenalty( this->mortarMeshId, params.rate_penalty_ratio );
         setRatePercentPenalty( this->nonmortarMeshId,  params.rate_penalty_ratio );
      }

   } // end if-penalty

   // register coupling scheme
   const int csIndex = 0;
   registerCouplingScheme( csIndex,
                           this->mortarMeshId,
                           this->nonmortarMeshId,
                           SURFACE_TO_SURFACE,
                           contact_case,
                           method,
                           model,
                           enforcement,
                           BINNING_GRID,
                           ExecutionMode::Sequential );

   enableTimestepVote( csIndex, params.enable_timestep_vote );
   setTimestepPenFrac( csIndex, params.timestep_pen_frac );
   setTimestepScale( csIndex, params.timestep_scale );

   // if enforcement is penalty, register penalty parameters
   if (enforcement == PENALTY)
   {
      // set the penetration fraction for timestep votes computed with penalty enforcements
      setAutoContactPenScale( csIndex, params.auto_contact_pen_frac );

   } // end if-penalty

   setContactAreaFrac( csIndex, 1.e-6 );

   if (visualization)
   {
      setPlotCycleIncrement( csIndex, 1 );
   }

   setLoggingLevel(csIndex, TRIBOL_WARNING);

   if (method == COMMON_PLANE && enforcement == PENALTY)
   {
      PenaltyConstraintType constraint_type = (params.constant_rate_penalty || params.percent_rate_penalty) 
                                            ? KINEMATIC_AND_RATE : KINEMATIC; 
      KinematicPenaltyCalculation pen_calc = (params.penalty_ratio) ? KINEMATIC_ELEMENT : KINEMATIC_CONSTANT;
 
      RatePenaltyCalculation rate_calc; 
      rate_calc = NO_RATE_PENALTY;
      if (params.constant_rate_penalty)
      {
         rate_calc = RATE_CONSTANT;
      }
      else if (params.percent_rate_penalty)
      {
         rate_calc = RATE_PERCENT;
      }

      // set penalty options after registering coupling scheme
      setPenaltyOptions( csIndex, constraint_type, pen_calc, rate_calc );

   }
   else if ((method == SINGLE_MORTAR || method == ALIGNED_MORTAR) && enforcement == LAGRANGE_MULTIPLIER)
   {
      // note, eval modes and sparse modes not exposed in the interface to this class
      setLagrangeMultiplierOptions( csIndex, ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN, 
                                    SparseMode::MFEM_LINKED_LIST );
   }
   else if (method == MORTAR_WEIGHTS)
   {
      // note, eval modes and sparse modes not exposed in the interface to this class
      setLagrangeMultiplierOptions( csIndex, ImplicitEvalMode::MORTAR_WEIGHTS_EVAL, 
                                    SparseMode::MFEM_LINKED_LIST ); 
   }

   // call tribol update()
   int err = update( 1, 1., params.dt );

   // print mesh to vtk
   if (visualization)
   {
      testMeshToVtk( "", 1, 1 );
   }

   return err;

} // end tribolSetupAndUpdate()
      
//------------------------------------------------------------------------------
void TestMesh::setupPatchTestDirichletBCs( IndexT mesh_id, 
                                           int numElemsX, 
                                           int numElemsY, 
                                           int numElemsZ, 
                                           int nodeIdOffset, 
                                           bool inHomogeneousGap, 
                                           RealT inHomogeneousZVal )
{
   SLIC_ERROR_IF( !this->mesh_constructed, "TestMesh::setupPatchTestDirichletBCs(): " << 
                  "mesh must be constructed prior to calling this routine." );

   bool mortar = false;
   if (mesh_id == this->mortarMeshId)
   {
      mortar = true;
   }

   // This routine sets up x,y, and z-component Dirichlet BCs on 
   // three faces of each block specifically for the classical contact 
   // patch test problem
   //
   // NOTE: the number of nodes in the x, y and z-directions is the same 
   // for a hex mesh and tet mesh
   int numNodesX = numElemsX + 1;
   int numNodesY = numElemsY + 1;
   int numNodesZ = numElemsZ + 1;

   int numNodes = numNodesX * numNodesY * numNodesZ;

   int* nodeIdsX, *nodeIdsY, *nodeIdsZ;
   RealT* valX, *valY, *valZ;
   if (mortar) // mortar BCs
   {
      this->dirNodesX1 = new int[ numNodes ];
      this->dirNodesY1 = new int[ numNodes ];
      this->dirNodesZ1 = new int[ numNodes ];
      nodeIdsX = this->dirNodesX1;
      nodeIdsY = this->dirNodesY1;
      nodeIdsZ = this->dirNodesZ1;

      this->iDirValX1 = new RealT[ numNodes ];
      this->iDirValY1 = new RealT[ numNodes ];
      this->iDirValZ1 = new RealT[ numNodes ];
      valX     = this->iDirValX1;
      valY     = this->iDirValY1;
      valZ     = this->iDirValZ1;
   }
   else // nonmortar BCs
   {
      this->dirNodesX2 = new int[ numNodes ];
      this->dirNodesY2 = new int[ numNodes ];
      this->dirNodesZ2 = new int[ numNodes ];
      nodeIdsX = this->dirNodesX2;
      nodeIdsY = this->dirNodesY2;
      nodeIdsZ = this->dirNodesZ2;
   
      this->iDirValX2 = new RealT[ numNodes ];
      this->iDirValY2 = new RealT[ numNodes ];
      this->iDirValZ2 = new RealT[ numNodes ];
      valX     = this->iDirValX2;
      valY     = this->iDirValY2;
      valZ     = this->iDirValZ2;
   }

   // initialize arrays
   initIntArray( nodeIdsX, numNodes, -1 ); 
   initIntArray( nodeIdsY, numNodes, -1 ); 
   initIntArray( nodeIdsZ, numNodes, -1 ); 
   initRealArray( valX, numNodes, 0. );
   initRealArray( valY, numNodes, 0. );
   initRealArray( valZ, numNodes, 0. );

   // setup the boundary conditions for mortar (bottom) block
   if (mortar) 
   {
      // setup BCs in the x-direction
      int ctr = 0;
      for (int k=0; k<numNodesZ; ++k)
      {
         for (int j=0; j<numNodesY; ++j)
         {
            int yOffset = j * numNodesX;
            int zOffset = numNodesX * numNodesY * k;
            int id = yOffset + zOffset;
            nodeIdsX[ ctr ] = id;
            ++ctr;
         }
      }

      // setupBCs in the y-dirction
      ctr = 0;
      for (int k=0; k<numNodesZ; ++k)
      {
         for (int i=0; i<numNodesX; ++i)
         {
            int zOffset = numNodesX * numNodesY * k;
            int id = zOffset + i;
            nodeIdsY[ ctr ] = id;
            ++ctr;
         }
      }

      // setupBCs in the z-dirction
      ctr = 0;
      for (int j=0; j<numNodesY; ++j)
      {
         for (int i=0; i<numNodesX; ++i)
         {
            nodeIdsZ[ ctr ] = ctr;
            ++ctr;
         }
      }
      if (inHomogeneousGap)
      {
         int icr = 0;
         for (int j=0; j<numNodesY; ++j)
         {
            for (int i=0; i<numNodesX; ++i)
            {
               int offset = numNodesX * numNodesY * (numNodesZ - 1);
               nodeIdsZ[ ctr ] = offset + icr; 
               valZ[ ctr ] = inHomogeneousZVal;
               ++ctr;
               ++icr;
            }
         }
      }
   }
   else // BCs for nonmortar (top) block 
   {
      // setup BCs in the x-direction
      int ctr = 0;
      for (int k=0; k<numNodesZ; ++k)
      {
         for (int j=0; j<numNodesY; ++j)
         {
            int yOffset = j * numNodesX;
            int zOffset = numNodesX * numNodesY * k;
            int id = nodeIdOffset + yOffset + zOffset;
            nodeIdsX[ ctr ] = id;
            ++ctr;
         }
      }

      // setupBCs in the y-dirction
      ctr = 0;
      for (int k=0; k<numNodesZ; ++k)
      {
         for (int i=0; i<numNodesX; ++i)
         {
            int zOffset = numNodesX * numNodesY * k;
            int id = nodeIdOffset + zOffset + i;
            nodeIdsY[ ctr ] = id;
            ++ctr;
         }
      }

      // setupBCs in the z-direction; Check for inHomogeneous gap 
      // first in order to keep ordering of nodes sequential and 
      // and increasing
      ctr = 0;
      if (inHomogeneousGap)
      {
         for (int j=0; j<numNodesY; ++j)
         {
            for (int i=0; i<numNodesX; ++i)
            {
               int offset = nodeIdOffset;
               nodeIdsZ[ ctr ] = offset + ctr;
               valZ[ ctr ] = inHomogeneousZVal;
               ++ctr;
            }
         }
      }
      int icr = 0;
      for (int j=0; j<numNodesY; ++j)
      {
         for (int i=0; i<numNodesX; ++i)
         {
            int offset = nodeIdOffset + numNodesX * numNodesY * (numNodesZ - 1);
            nodeIdsZ[ ctr ] = offset + icr;
            ++ctr;
            ++icr;
         }
      }
   }

   return;
} // end setupPatchTestDirichletBCs()

//------------------------------------------------------------------------------
void TestMesh::setupPatchTestPressureDofs( IndexT mesh_id,
                                           int numElemsX,
                                           int numElemsY,
                                           int numElemsZ,
                                           int nodeIdOffset,
                                           bool contact )
{
   SLIC_ERROR_IF( !this->mesh_constructed, "TestMesh::setupPatchTestPressureDofs(): " << 
                  "mesh must be constructed prior to calling this routine." );

   bool mortar = false;
   if (mesh_id == this->mortarMeshId)
   {
      mortar = true;
   }

   // this routine hard codes nonmortar pressure dofs for the bottom surface 
   // of the top (nonmortar) block
   int numNodes = (numElemsX+1) * (numElemsY+1) * (numElemsZ+1);
   int * presDofs;
  
   if (mortar) // mortar dofs
   {
      this->presDofs1 = new int[ numNodes ]; 
      presDofs = this->presDofs1; 
   }
   else // nonmortar pressure dofs
   {
      this->presDofs2 = new int[ numNodes ]; 
      presDofs = this->presDofs2;
   }

   for (int i=0; i<numNodes; ++i)
   {
      presDofs[ i ] = -1;
   }

   if (contact)
   {
      int numSurfaceNodes = (numElemsX+1) * (numElemsY+1);
      for (int i=0; i<numSurfaceNodes; ++i)
      {
         presDofs[ i ] = nodeIdOffset + i;
      }
   }
} // end setupPatchTestPressureDofs()

//------------------------------------------------------------------------------
void TestMesh::setupMfemMesh( bool fix_orientation )
{
   SLIC_ERROR_IF( !this->mesh_constructed, "TestMesh::setupMfemMesh(): " << 
                  "test mesh must be constructed prior to calling this routine." );

   SLIC_ERROR_IF( this->dim != 3, "TestMesh::setupMfemMesh(): Mfem meshes of dimension, " << 
                  this->dim << ", are not supported at this time." );

   // construct new mfem mesh
   if (this->mfem_mesh != nullptr)
   {
      this->mfem_mesh->Clear();
   }

   this->mfem_mesh = new mfem::Mesh( this->dim,
                                     this->numTotalNodes,
                                     this->numTotalElements );

   // add mortar elements and vertices. Not sure if order of adding 
   // elements matters, but adding vertices should probably correspond 
   // to the global contiguous id system
   int mConn[ this->numNodesPerElement ];
   for (int iel=0; iel<this->numMortarElements; ++iel)
   {
      for (int idx=0; idx<this->numNodesPerElement; ++idx)
      {
         int index = iel * this->numNodesPerElement + idx;
         mConn[ idx ] = this->elConn1[ index ];
      }
      switch (this->cellType)
      {
         case LINEAR_TRIANGLE:
         {
            this->mfem_mesh->AddTet( &mConn[0] );
            break;
         }
         case LINEAR_QUAD:
         {
            this->mfem_mesh->AddHex( &mConn[0] );
            break;
         }
         default:
         {
            SLIC_ERROR("Element type not supported for creating mfem mesh from test mesh.");
         }
      } // end switch on surface element type
   } // end loop over mortar elements

   for (int i=0; i<this->numMortarNodes; ++i)
   {
      RealT vert[3] = {0., 0., 0.}; 
      vert[ 0 ] = this->x[ i ];
      vert[ 1 ] = this->y[ i ];
      vert[ 2 ] = this->z[ i ];

      this->mfem_mesh->AddVertex( &vert[0] );
   } 

   // add nonmortar elements and vertices. Not sure if order of adding 
   // elements matters, but adding vertices should probably correspond 
   // to the global contiguous id system
   int sConn[ this->numNodesPerElement ]; 
   for (int iel=0; iel<this->numNonmortarElements; ++iel)
   {
      for (int idx=0; idx<this->numNodesPerElement; ++idx)
      {
         int index = iel * this->numNodesPerElement + idx;
         sConn[ idx ] = this->elConn2[ index ];
      }
      switch (this->cellType)
      {
         case LINEAR_TRIANGLE:
         {
            this->mfem_mesh->AddTet( &sConn[0] );
            break;
         }
         case LINEAR_QUAD:
         {
            this->mfem_mesh->AddHex( &sConn[0] );
            break;
         }
         default:
         {
            SLIC_ERROR("Element type not supported for creating mfem mesh from test mesh.");
         }
      } // end switch on surface element type
   } // end loop over nonmortar elements

   for (int i=0; i<this->numNonmortarNodes; ++i)
   { 
      RealT vert[3] = {0., 0., 0.}; 
      int offset = this->numMortarNodes;
      vert[ 0 ] = this->x[ offset + i ];
      vert[ 1 ] = this->y[ offset + i ];
      vert[ 2 ] = this->z[ offset + i ];

      this->mfem_mesh->AddVertex( &vert[0] );
   }

   int gen_edges = 0; // mfem default
   int refine = 0; // mfem default
   switch (this->cellType)
   {
      case LINEAR_TRIANGLE:
      {
         this->mfem_mesh->FinalizeTetMesh( gen_edges, refine, fix_orientation );
         break;
      }
      case LINEAR_QUAD:
      {
         this->mfem_mesh->FinalizeHexMesh( gen_edges, refine, fix_orientation );
         break;
      }
      default:
      {
         // no-op
         break;
      }
   } // end switch on surface cell type

} // end setupMfemMesh()


//------------------------------------------------------------------------------
void TestMesh::computeEquilibriumJacobian( mfem::SparseMatrix* A,
                                           RealT const nu, RealT const youngs )
{
   SLIC_ERROR_IF( this->mfem_mesh == nullptr, 
                  "TestMesh::computeEquilibriumJacobian(): must call setupMfemMesh() " << 
                  "prior to calling this routine." );

   // define the FE collection and finite element space
   mfem::FiniteElementSpace * fes { nullptr };
   int order = 1; // hard coded for linear elements
   mfem::H1_FECollection fe_coll( order, this->dim );
   fes = new mfem::FiniteElementSpace( this->mfem_mesh, &fe_coll, this->dim );
   
   // compute 1st Lame parameter from Youngs modulus and Poisson's ratio 
   // (required for MFEM integrator)
   RealT lambda_val { (youngs * nu) / ((1. + nu) * (1. - 2. * nu)) }; 

   // compute shear modulus from Young's Modulus and Poisson's ratio
   // (required for MFEM integrator)
   RealT mu_val { youngs / (2.*(1.+nu)) }; 
   mfem::ConstantCoefficient lambda(lambda_val);
   mfem::ConstantCoefficient mu(mu_val);

   // instantiate elasticity integrator
   mfem::ElasticityIntegrator elastInteg( lambda, mu );

   // compute element contributions on the mesh and sum them into the 
   // input sparse matrix
   computeElementJacobianContributions( A, &elastInteg, fes );

   delete fes;

} // end TestMesh::computeEquilibriumJacobian()

//------------------------------------------------------------------------------
void TestMesh::computeEquilibriumJacobian( mfem::SparseMatrix * const A, 
                                           mfem::ElasticityIntegrator * eInteg,
                                           mfem::FiniteElementSpace * fes )
{
   // compute element contributions on the mesh and sum them into the 
   // input sparse matrix
   computeElementJacobianContributions( A, eInteg, fes );

} // end TestMesh::computeEquilibriumJacobian()

//------------------------------------------------------------------------------
void TestMesh::computeElementJacobianContributions( mfem::SparseMatrix * const A,
                                                    mfem::ElasticityIntegrator * eInt,
                                                    mfem::FiniteElementSpace * fe_space,
                                                    bool matrixDebug )
{
   SLIC_ERROR_IF( A == nullptr, "TestMesh::computeElementJacobianContributions(): " <<
                  "input pointer to sparse matrix is null." );
   SLIC_ERROR_IF( eInt == nullptr, "TestMesh::computeElementJacobianContributions(): " <<
                  "input pointer to elasticity integrator is null." );
   SLIC_ERROR_IF( fe_space == nullptr, "TestMesh::computeElementJacobianContributions(): " <<
                  "input pointer to finite element space is null." );

   mfem::ElementTransformation *T;
   mfem::DenseMatrix elmat;
   const mfem::FiniteElement *fe; 

   // loop over finite elements
   for (int idel=0; idel<fe_space->GetNE(); ++idel)
   {
      fe = fe_space->GetFE(idel);
      T = fe_space->GetElementTransformation(idel);

      eInt->AssembleElementMatrix( *fe, *T, elmat );
 
      if (matrixDebug)
      {
         // print out elmat for debugging
         std::ofstream matrix;
         matrix.setf(std::ios::scientific);
         matrix.precision(2);
         std::ostringstream suffix_matrix;
         suffix_matrix << "el_" << idel << ".txt";
         matrix.open("matrix_" + suffix_matrix.str());

         int numRows = elmat.NumRows();
         int numCols = elmat.NumCols();
         for (int i=0; i<numRows; ++i)
         {
            for (int j=0; j<numCols; ++j)
            {
               RealT val = elmat(i,j);
               matrix << val << "  ";
            }
            matrix << "\n";
         }

         matrix.close();
      }

      // sum contributions into global jacobian input/output argument
      int * el_conn;
      for (int i=0; i<fe->GetDof(); ++i)
      {
         for (int j=0; j<fe->GetDof(); ++j)
         {
            // get global indices
            if ( idel < this->numMortarElements )
            {
               el_conn = this->elConn1 + idel * fe->GetDof();
            }
            else 
            {
               el_conn = this->elConn2 + (idel - this->numMortarElements) * fe->GetDof();
            } 

            int glb_i = fe->GetDim() * el_conn[i];
            int glb_j = fe->GetDim() * el_conn[j];
 
            // add the 3x3 block for the (i,j) pair
            for (int m=0; m<fe->GetDim(); ++m)
            {
               for (int n=0; n<fe->GetDim(); ++n)
               {
                  // get local element indices
                  int el_m = fe->GetDof() * m;
                  int el_n = fe->GetDof() * n;

                  // keep in mind the local element stiffness matrix is stacked, x, then y, and 
                  // then z contributions
                  A->Add(glb_i+m, glb_j+n, elmat(el_m+i,el_n+j));
               }
            }

         } // end loop over column dofs
      } // end loop over row dofs
   } // end of loop over elements
} // end TestMesh::computeElementJacobianContributions()

//------------------------------------------------------------------------------
void TestMesh::tribolMatrixToSystemMatrix( mfem::DenseMatrix * const ATribol,
                                           mfem::SparseMatrix * const ASystem )
{

   SLIC_ERROR_IF( ATribol == nullptr, "TestMesh::tribolMatrixToSystemMatrix(): " 
                  << "ATribol pointer is null." );

   SLIC_ERROR_IF( ATribol == nullptr, "TestMesh::tribolMatrixToSystemMatrix(): " 
                  << "ASystem pointer is null." );

   int solveSize = this->dim * this->numTotalNodes + this->numNonmortarSurfaceNodes;

   // initialize matrix
   for (int i=0; i<solveSize; ++i)
   {
      for (int j=0; j<solveSize; ++j)
      {
         ASystem->Add(i,j, 0.);
      }
   } // end of matrix initialization

   ////////////////////////////
   // assemble matrix blocks //
   ////////////////////////////

   // compose upper diagonal block
   for (int i=0; i<this->dim * this->numTotalNodes; ++i)
   {
      for (int j=0; j<this->dim * this->numTotalNodes; ++j)
      {
         ASystem->Add(i,j, (*ATribol)(i,j));
      }
   } // end of upper diagonal block loop

   // compose off-diagonal blocks
   for (int i=0; i<this->numNonmortarSurfaceNodes; ++i)
   {
      int newOffset = this->dim * this->numTotalNodes;
      int oldOffset = this->dim * this->numTotalNodes + 
                      this->numMortarNodes;
      int col_id = newOffset + i;
      int row_id = col_id;

      for (int j=0; j<this->dim * this->numTotalNodes; ++j)
      {
         // assemble upper-right diagonal block first
         ASystem->Set( j, col_id, (*ATribol)( j, oldOffset + i ));

         // assemble lower left diagonal block 
         ASystem->Set( row_id, j, (*ATribol)( oldOffset + i, j ));
      }
   } // end of off-diagonal block loop

   // KEEP this as debug code for unit tests, but in general 
   // there should be no contributions
   // TEST: compose bottom-diagonal block diagonal elements
//   for (int i=0; i<this->m_mesh.numNonmortarSurfaceNodes; ++i)
//   {
//      int newOffset = this->m_mesh.dim * this->m_mesh.numTotalNodes;
//      int oldOffset = this->m_mesh.dim * this->m_mesh.numTotalNodes + 
//                      this->m_mesh.numMortarNodes;
//      int idx = newOffset + i;
//         ASystem->Add(idx,idx, (*ATribol)( oldOffset + i, oldOffset + i ));
//   }

} // end TestMesh::tribolMatrixToSystemMatrix()

//------------------------------------------------------------------------------
void TestMesh::getGapEvals( RealT * const v )
{
   SLIC_ERROR_IF( v == nullptr, "TestMesh::getGapEvals(): input pointer is null." );
   int presDofCtr = 0;
   for (int a=0; a<this->numNonmortarNodes; ++a)
   {
      // pressure dofs
      int presOffset = this->dim * this->numTotalNodes;
      if ( this->presDofs2[a] >= 0) // note: pressure dofs ordered sequentially
      {
         v[ presOffset + presDofCtr ] = 
             -this->gaps[ this->numMortarNodes + a ]; // we have negative rhs
         ++presDofCtr;
      }
   }
} // end TestMesh::getGapEvals()

//------------------------------------------------------------------------------
void TestMesh::enforceDirichletBCs( mfem::SparseMatrix * const A,
                                    mfem::Vector * const b,
                                    bool contact )
{
   SLIC_ERROR_IF( A == nullptr, "TestMesh::enforceDirichletBCs(): " << 
                  "input pointer to sparse matrix is null." );
   SLIC_ERROR_IF( b == nullptr, "TestMesh::enforceDirichletBCs(): " << 
                  "input pointer to rhs vector, b, is null." );

   int * dirBCX, * dirBCY, * dirBCZ, * presDofs;
   RealT *dirValX, *dirValY, *dirValZ;
   int numBlkNodes;

   int presCtr = 0;
   // loop over each mesh block and then over the number of block nodes, 
   // enforcing BCs
   for (int iblk=0; iblk<2; ++iblk)
   {
      // point to mortar data if iblk == 0
      if (iblk == 0)
      {
         dirBCX      = this->dirNodesX1;
         dirBCY      = this->dirNodesY1;
         dirBCZ      = this->dirNodesZ1; 
         dirValX     = this->iDirValX1;
         dirValY     = this->iDirValY1;
         dirValZ     = this->iDirValZ1;
         presDofs    = this->presDofs1;
         numBlkNodes = this->numMortarNodes;
      }
      // point to nonmortar data if iblk != 0
      else
      {
         dirBCX      = this->dirNodesX2;
         dirBCY      = this->dirNodesY2;
         dirBCZ      = this->dirNodesZ2;
         dirValX     = this->iDirValX2;
         dirValY     = this->iDirValY2;
         dirValZ     = this->iDirValZ2;
         presDofs    = this->presDofs2;
         numBlkNodes = this->numNonmortarNodes;
      }

      // loop over all block nodes. 
      for (int m=0; m<numBlkNodes; ++m)
      {
         // row ids for homogeneous dirichlet BCs
         if (dirBCX[m] >= 0)
         {
            int bcRowIdX = this->dim * dirBCX[m];
            RealT dirBCValX = dirValX[m];
            A->EliminateRowCol( bcRowIdX, dirBCValX, *b );
         }

         if (dirBCY[m] >= 0)
         {
            int bcRowIdY = this->dim * dirBCY[m]+1;
            RealT dirBCValY = dirValY[m];
            A->EliminateRowCol( bcRowIdY, dirBCValY, *b );
         }

         if (dirBCZ[m] >= 0)
         {
            int bcRowIdZ = this->dim * dirBCZ[m]+2;
            RealT dirBCValZ = dirValZ[m];
            A->EliminateRowCol( bcRowIdZ, dirBCValZ, *b );
         }

         // KEEP THIS CODE - this is used to test pressure rows etc.
         // TEST: row ids for pressure rows where we will specify 
         // homogeneous dirichlet BCs for the pressure Lagrange multipliers
         if (!contact)
         {
            if (presDofs[m] >= 0) 
            {
               // the following index assumes that all pressure dofs, which have 
               // contiguous integer Ids, are going to be zeroed out
               int testPresRowId = this->dim * this->numTotalNodes 
                                   + presCtr;
 
               if (testPresRowId > A->NumRows())
               {
                  SLIC_ERROR("Pressure dof > number of matrix rows.");
               }
               A->EliminateRowCol( testPresRowId, 0., *b );
               ++presCtr;
            }
         } // end if (!contact)

      } // end loop over number of nodes
   } // end loop over mesh blocks
} // end TestMesh::enforceDirichletBCs()

//------------------------------------------------------------------------------
void TestMesh::testMeshToVtk( const std::string& dir, int cycle, RealT time )
{
   std::ostringstream suffix_mesh;
   suffix_mesh << std::setfill('0') << std::setw(7) << cycle << ".vtk";
   std::string f_name = axom::utilities::filesystem::joinPath(
      dir, "meshTest3D_" + suffix_mesh.str() ); 
  
   std::ofstream mesh;
   mesh.setf(std::ios::scientific);
   mesh.open(f_name.c_str());
    
   mesh << "# vtk DataFile Version 3.0\n";
   mesh << "vtk output\n";
   mesh << "ASCII\n";
   mesh << "DATASET UNSTRUCTURED_GRID\n";

   // Add the cycle and time to FieldData
   mesh << "FIELD FieldData 2\n";
   mesh << "TIME 1 1 RealT\n";
   mesh << time << "\n";
   mesh << "CYCLE 1 1 int\n";
   mesh << cycle << "\n";

   // print mesh vertices
   mesh << "POINTS " << this->numTotalNodes << " float\n";
   for (int i=0; i<this->numTotalNodes; ++i)
   {
      mesh << this->x[i] << " " 
           << this->y[i] << " " 
           << this->z[i] << std::endl;
   }

   // print mesh element connectivity
   mesh << "CELLS " << this->numTotalElements << " " 
        << this->numTotalElements+this->numTotalElements*this->numNodesPerElement << std::endl;

   for (int i=0; i<this->numMortarElements; ++i)
   {
      mesh << this->numNodesPerElement; 
      for (int a=0; a<this->numNodesPerElement; ++a)
      {
         int id = this->numNodesPerElement * i + a;
         mesh << " " << elConn1[id];
      }
      mesh << std::endl;
   }

   for (int i=0; i<this->numNonmortarElements; ++i)
   {
      mesh << this->numNodesPerElement; 
      for (int a=0; a<this->numNodesPerElement; ++a)
      {
         int id = this->numNodesPerElement * i + a;
         mesh << " " << elConn2[id];
      }
      mesh << std::endl;
   }

   // specify integer id for each cell type. 
   mesh << "CELL_TYPES " << this->numTotalElements << std::endl;

   int element_id;
   switch (this->numNodesPerElement)
   {
      case 8:
         element_id = 12; // vtk 8-node hexahedron
         break;
      case 4:
         element_id = 10; // vtk 4-node tetra
         break;
      default :
         SLIC_ERROR("TestMesh::testMeshToVtk(): element type not supported by vtk.");
   }

   for (int i=0; i<this->numTotalElements; ++i)
   {
      mesh << element_id << std::endl;
   }

   mesh.close();

} // end TestMesh::testMeshToVtk()

//------------------------------------------------------------------------------
} // end of namespace "tribol"

namespace mfem_ext
{

CentralDiffSolver::CentralDiffSolver(const mfem::Array<int>& bc_vdofs_)
:  bc_vdofs { bc_vdofs_ },
   first_step { true }
{}

void CentralDiffSolver::Step(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt)
{
   // acceleration at t
   f->SetTime(t);
   if (first_step)
   {
      accel.SetSize(x.Size());
      // update acceleration given displacement, velocity using linked
      // SecondOrderTimeDependentOperator (set using Init() method)
      f->Mult(x, dxdt, accel);
      first_step = false;
   }

   // velocity at t + dt/2
   dxdt.Add(0.5*dt, accel);

   // set homogeneous velocity BC at t + dt/2
   SetHomogeneousBC(dxdt);

   // set displacement at t + dt
   x.Add(dt, dxdt);

   // acceleration at t + dt
   f->SetTime(t + dt);
   // update acceleration given displacement, velocity using linked
   // SecondOrderTimeDependentOperator (set using Init() method)
   f->Mult(x, dxdt, accel);

   // velocity at t + dt
   dxdt.Add(0.5*dt, accel);
}

void CentralDiffSolver::SetHomogeneousBC(mfem::Vector& dxdt) const
{
   for (auto bc_vdof : bc_vdofs)
   {
      dxdt[bc_vdof] = 0.0;
   }
}

#ifdef TRIBOL_USE_MPI
ExplicitMechanics::ExplicitMechanics(
   mfem::ParFiniteElementSpace& fespace, 
   mfem::Coefficient& rho,
   mfem::Coefficient& lambda,
   mfem::Coefficient& mu
)
:  f_ext { &fespace },
   elasticity { &fespace },
   inv_lumped_mass { fespace.GetVSize() }
{
   // create inverse lumped mass matrix; store as a vector
   mfem::ParBilinearForm mass { &fespace };
   mass.AddDomainIntegrator(new mfem::VectorMassIntegrator(rho));
   mass.Assemble();
   mfem::Vector ones {fespace.GetVSize()};
   ones = 1.0;
   mass.SpMat().Mult(ones, inv_lumped_mass);
   mfem::Vector mass_true(fespace.GetTrueVSize());
   const Operator& P = *fespace.GetProlongationMatrix();
   P.MultTranspose(inv_lumped_mass, mass_true);
   for (int i {0}; i < mass_true.Size(); ++i)
   {
      mass_true[i] = 1.0 / mass_true[i];
   }
   P.Mult(mass_true, inv_lumped_mass);

   // create elasticity stiffness matrix
   elasticity.AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda, mu));
   elasticity.Assemble();
}

void ExplicitMechanics::Mult(
   const mfem::Vector& u,
   const mfem::Vector& AXOM_UNUSED_PARAM(dudt),
   mfem::Vector& a
) const
{
   mfem::Vector f { u.Size() };
   f = 0.0;

   mfem::Vector f_int { f.Size() };
   elasticity.Mult(u, f_int);
   f.Add(-1.0, f_int);

   f.Add(1.0, f_ext);

   // sum forces over ranks
   auto& fespace = *elasticity.ParFESpace();
   const Operator& P = *fespace.GetProlongationMatrix();
   mfem::Vector f_true {fespace.GetTrueVSize()};
   P.MultTranspose(f, f_true);
   P.Mult(f_true, f);

   for (int i {0}; i < inv_lumped_mass.Size(); ++i)
   {
      a[i] = inv_lumped_mass[i] * f[i];
   }
}
#endif

} // end of namespace "mfem_ext"
