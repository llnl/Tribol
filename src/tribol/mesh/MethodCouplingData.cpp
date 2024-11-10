// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

// Axom includes
#include "axom/slic.hpp"

#include <iostream> 
#include <sstream>
#include <iomanip>
#include <fstream>

namespace tribol
{

////////////////////////////////////////////////
//                                            //
// Routines for the SurfaceContactElem struct //
//                                            //
////////////////////////////////////////////////
void SurfaceContactElem::allocateMortarWts( )
{
   // We store wts, n_aa and n_ab, which are nonmortar/nonmortar and nonmortar/mortar 
   // products of shape functions
   this->numWts = 2 * this->numFaceVert * this->numFaceVert;
   if (this->mortarWts != nullptr)
   {
      delete [] this->mortarWts;
      this->mortarWts = new RealT [ this->numWts ];
   }
   else
   {
      this->mortarWts = new RealT [ this->numWts ];
   }

   this->initializeMortarWts();
}

//------------------------------------------------------------------------------
void SurfaceContactElem::initializeMortarWts( ) 
{
   for (int i=0; i<this->numWts; ++i)
   {
      mortarWts[i] = 0.;
   }
}

//------------------------------------------------------------------------------
void SurfaceContactElem::allocateBlockJ( EnforcementMethod enf )
{
   // number of element displacement degrees of freedom
   int nPrimal = this->dim * this->numFaceVert;
   for (int i{}; i < 2; ++i)
   {
      for (int j{}; j < 2; ++j)
      {
         this->blockJ(i, j) = DeviceArray2D<RealT>(nPrimal, nPrimal);
         this->blockJ(i, j).fill(0.0);
      }
   }

   if (enf == LAGRANGE_MULTIPLIER)
   {
      // number of element Lagrange multiplier degrees of freedom
      int nDual = this->numFaceVert;
      for (int i{}; i < 2; ++i)
      {
         this->blockJ(i, 2) = DeviceArray2D<RealT>(nPrimal, nDual);
         this->blockJ(i, 2).fill(0.0);
         // transpose
         this->blockJ(2, i) = DeviceArray2D<RealT>(nDual, nPrimal);
         this->blockJ(2, i).fill(0.0);
      }
      this->blockJ(2, 2) = DeviceArray2D<RealT>(nDual, nDual);
      this->blockJ(2, 2).fill(0.0);
   }
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void SurfaceContactElem::deallocateElem( )
{
    if (this->mortarWts != nullptr)
    {
       delete [] this->mortarWts;
       this->mortarWts = nullptr;
    }
}

//------------------------------------------------------------------------------
RealT SurfaceContactElem::getMortarNonmortarWt( const int a, const int b )
{
   // note: the mortar wts are stored in a stacked array with nonmortar-nonmortar 
   //       wts followed by mortar-nonmortar weights in mortar-nonmortar ordering

   int mortarNonmortarId = this->numFaceVert * this->numFaceVert + 
                       this->numFaceVert * a + b;

   RealT wt = this->mortarWts[ mortarNonmortarId ];
   return wt;
}

//------------------------------------------------------------------------------
RealT SurfaceContactElem::getNonmortarMortarWt( const int a, const int b )
{
   // note: the mortar wts are stored in a stacked array with nonmortar-nonmortar 
   //       wts followed by mortar-nonmortar weights in mortar-nonmortar ordering.
   //       Therefore, the nonmortarMortar id is the transpose of how it is 
   //       stored

   int nonmortarMortarId = this->numFaceVert * this->numFaceVert + 
                       this->numFaceVert * b + a;

   RealT wt = this->mortarWts[ nonmortarMortarId ];
   return wt;
}

//------------------------------------------------------------------------------
RealT SurfaceContactElem::getNonmortarNonmortarWt( const int a, const int b )
{
   // note: the mortar wts are stored in a stacked array with nonmortar-nonmortar 
   //       wts followed by mortar-nonmortar weights in mortar-nonmortar ordering

   int nonmortarNonmortarId = this->numFaceVert * a + b;

   RealT wt = this->mortarWts[ nonmortarNonmortarId ];
   return wt;
}

//------------------------------------------------------------------------------
int SurfaceContactElem::getJacobianIndex( JacBlock block,
                                          const int a, 
                                          const int b ) const
{
   if (block == JguBlock)
   {
      return a + this->numFaceVert * b;
   }

   if (block == JrpBlock)
   {
      return a + this->dim * this->numFaceVert * b;
   }

   return -1;
}

//------------------------------------------------------------------------------
int SurfaceContactElem::getJacobianDimOffset( JacBlock block ) const
{
   if (block == JguBlock)
   {
      return this->numFaceVert * this->numFaceVert;
   }

   if (block == JrpBlock)
   {
      return this->numFaceVert;
   }

   return -1;
}

//------------------------------------------------------------------------------
MethodData::MethodData()
:  m_blockJElemIds (
      static_cast<IndexT>(BlockSpace::NUM_BLOCK_SPACES),
      static_cast<IndexT>(BlockSpace::NUM_BLOCK_SPACES)
   ) ,
   m_blockJ (
      static_cast<IndexT>(BlockSpace::NUM_BLOCK_SPACES),
      static_cast<IndexT>(BlockSpace::NUM_BLOCK_SPACES)
   )
{
   m_blockJ.shrink();
}

//------------------------------------------------------------------------------
void MethodData::reserveBlockJ( 
   ArrayT<BlockSpace>&& blockJSpaces, 
   int nPairs
)
{
   IndexT pairFraction = static_cast<IndexT>(0.5 * nPairs);

   m_blockJSpaces = std::move(blockJSpaces);
   m_blockJElemIds.fill(ArrayT<int>(0, 0));
   for (auto blockJSpace : m_blockJSpaces)
   {
      m_blockJElemIds[static_cast<IndexT>(blockJSpace)].reserve(pairFraction);
   }
   m_blockJ.fill(ArrayT<mfem::DenseMatrix>(0, 0));
   for (auto blockJSpaceI : m_blockJSpaces)
   {
      for (auto blockJSpaceJ : m_blockJSpaces)
      {
         m_blockJ(
            static_cast<IndexT>(blockJSpaceI),
            static_cast<IndexT>(blockJSpaceJ)
         ).reserve(pairFraction);
      }
   }
}

//------------------------------------------------------------------------------
void MethodData::storeElemBlockJ( 
   ArrayT<int>&& blockJElemIds,
   const StackArray<DeviceArray2D<RealT>, 9>& blockJ
)
{
   SLIC_ASSERT_MSG(blockJElemIds.size() == getNSpaces(),
      "Number of element ID vectors does not match the number of Jacobian spaces.");
   for (IndexT i{}; i < getNSpaces(); ++i)
   {
      IndexT blockIdxI = static_cast<IndexT>(m_blockJSpaces[i]);
      m_blockJElemIds[blockIdxI].push_back(blockJElemIds[i]);
      for (IndexT j{}; j < getNSpaces(); ++j)
      {
         IndexT blockIdxJ = static_cast<IndexT>(m_blockJSpaces[j]);
         // convert to mfem::DenseMatrix
         auto& block = blockJ(i, j);
         // this DenseMatrix is a "view" of block
         mfem::DenseMatrix block_densemat(block.data(), block.height(), block.width());
         // deep copy should happen here
         m_blockJ(blockIdxI, blockIdxJ).push_back(block_densemat);
      }
   }
} 


///////////////////////////////////////
//                                   //
// Routines for the MortarData class //
//                                   //
///////////////////////////////////////

// Constructor
MortarData::MortarData()
{
  // set null pointers
  this->m_smat       = nullptr;
}

//------------------------------------------------------------------------------
// Destructor
MortarData::~MortarData()
{
  if (this->m_smat != nullptr)
  {
     delete this->m_smat;
     this->m_smat = nullptr;
  }
}

//------------------------------------------------------------------------------
void MortarData::assembleJacobian( SurfaceContactElem & elem, SparseMode s_mode ) const
{
   (void) s_mode; // will be used at a later point
   // grab the two meshes in this coupling scheme
   auto& mortarMesh = *elem.m_mesh1;
   auto& nonmortarMesh  = *elem.m_mesh2;

   // compute the pressure dof offset. 
   // Recall that the "equilibrium" 
   // block is the problem_dimension x total_number_of_coupling_scheme_nodes,
   // which is the sum of mortar and nonmortar mesh nodes registered 
   // by the host code. The node ids between the two are assumed to 
   // be unique and contiguous. There is space to store a pressure 
   // dof for ALL nonmortar AND mortar nodes (done for ease of indexing 
   // using connectivity arrays registered by the host code), but we 
   // will pass back an array indicating the active pressure dofs for 
   // both mortar and nonmortar meshes in the case of LM implementations

   // TODO: fix tests where we now assume the number of nodes on the 
   // registered nonmortar and mortar mesh is ONLY the number of nonmortar 
   // nodes and mortar nodes, respectively.
   int numNodes = this->m_numTotalNodes;
   int presOffset = elem.dim * numNodes;

   // loop over contact element nodes and assemble the four block 
   // contributions stored in arrays on the SurfaceContactElem struct

   // loop over face nodes "a" (general loop, don't distinguish mortar/nonmortar 
   // as that changes for Jrp and Jgu contributions)
   for (int a = 0; a<elem.numFaceVert; ++a)
   {
      // get mortar and nonmortar node ids from connectivity
      int mortarNodeIdA = mortarMesh.getGlobalNodeId( elem.faceId1, a );
      int nonmortarNodeIdA  = nonmortarMesh.getGlobalNodeId( elem.faceId2, a );

      // loop over face nodes "b" (general loop, don't distinguish mortar/nonmortar 
      // as that changes for Jrp and Jgu contributions)
      for (int b = 0; b<elem.numFaceVert; ++b)
      {
         // get nonmortar node id from connectivity
         int nonmortarNodeIdB = nonmortarMesh.getGlobalNodeId( elem.faceId2, b );

         // We don't exclude nonmortar nodes that are in separation. NOTE: Per 
         // testing, we include ALL nonmortar contributions for faces that 
         // have positive areas of overlap regardless of gap evaluation. Contact 
         // activity is determined from gaps AND pressure solution per KKT constraint 
         // equations

         /////////////////////////////////////////////////////////////////
         // Assemble Jrp contributions (upper-right off-diagonal block) //
         /////////////////////////////////////////////////////////////////

         // get local ids into contact element Jacobian arrays
         int localId = elem.getJacobianIndex(SurfaceContactElem::JrpBlock, a, b );

         int i = elem.dim * mortarNodeIdA; // mortar row contributions
         int j = presOffset + nonmortarNodeIdB; // nonmortar column contributions always
         int dimOffset = elem.getJacobianDimOffset(SurfaceContactElem::JrpBlock);

         // Add() will "set" if a nonzero entry hasn't been 
         // introduced at the (i,j) element
         // Mortar-Lagrange multiplier block (0, 2)
         this->m_smat->Add( i, j, elem.blockJ(
            static_cast<IndexT>(BlockSpace::MORTAR),
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER)
         )[ localId ]);
         this->m_smat->Add( i+1, j, elem.blockJ(
            static_cast<IndexT>(BlockSpace::MORTAR),
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER)
         )[ localId + dimOffset ]);
         this->m_smat->Add( i+2, j, elem.blockJ(
            static_cast<IndexT>(BlockSpace::MORTAR),
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER)
         )[ localId + 2*dimOffset ]); // assume 3D for now

         // Nonmortar-Lagrange Multiplier block (1, 2)
         i = elem.dim * nonmortarNodeIdA; // nonmortar row contributions
         this->m_smat->Add( i, j, elem.blockJ(
            static_cast<IndexT>(BlockSpace::NONMORTAR),
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER)
         )[ localId  ]);
         this->m_smat->Add( i+1, j, elem.blockJ(
            static_cast<IndexT>(BlockSpace::NONMORTAR),
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER)
         )[ localId + dimOffset ]);
         this->m_smat->Add( i+2, j, elem.blockJ(
            static_cast<IndexT>(BlockSpace::NONMORTAR),
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER)
         )[ localId + 2*dimOffset ]); // assume 3D for now

         ////////////////////////////////////////////////////////////////
         // Assemble Jgu contributions (lower-left off-diagonal block) //
         ////////////////////////////////////////////////////////////////

         // note b and a are swapped.  index a loops over displacement DOFs (i.e. the
         // columns of Jgu) and index b loops over pressure DOFs (rows of Jgu)
         localId = elem.getJacobianIndex(SurfaceContactElem::JguBlock, b, a );

         i = presOffset + nonmortarNodeIdB; // nonmortar rows always
         j = elem.dim * mortarNodeIdA; // mortar column contributions
         dimOffset = elem.getJacobianDimOffset(SurfaceContactElem::JguBlock);

         // Lagrange-multiplier-mortar block (2, 0)
         this->m_smat->Add( i, j, elem.blockJ(
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER),
            static_cast<IndexT>(BlockSpace::MORTAR)
         )[ localId ]);
         this->m_smat->Add( i, j+1, elem.blockJ(
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER),
            static_cast<IndexT>(BlockSpace::MORTAR)
         )[ localId + dimOffset ]);
         this->m_smat->Add( i, j+2, elem.blockJ(
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER),
            static_cast<IndexT>(BlockSpace::MORTAR)
         )[ localId + 2*dimOffset ]); // assume 3D for now

         // Lagrange multiplier-nonmortar block (2, 1)
         j = elem.dim * nonmortarNodeIdA; // nonmortar column contributions
         this->m_smat->Add( i, j, elem.blockJ(
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER),
            static_cast<IndexT>(BlockSpace::NONMORTAR)
         )[ localId  ]);
         this->m_smat->Add( i, j+1, elem.blockJ(
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER),
            static_cast<IndexT>(BlockSpace::NONMORTAR)
         )[ localId + dimOffset ]);
         this->m_smat->Add( i, j+2, elem.blockJ(
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER),
            static_cast<IndexT>(BlockSpace::NONMORTAR)
         )[ localId + 2*dimOffset ]); // assume 3D for now

         //////////////////////////////////////////////////
         // Assemble Jru contributions (matrix 11-block) //
         //////////////////////////////////////////////////

         // TODO implement the assembly procedure if we want to add a consistent
         // Jacobian

         //////////////////////////////////////////////////
         // Assemble Jgp contributions (matrix 22-block) //
         //////////////////////////////////////////////////
            
         // TODO implement the assembly procedure if we want to add a consistent
         // Jacobian
         
      } // end loop over b nodes
   } // end loop over a nodes

   return;
} // end of MortarData::assembleJacobian() 

//------------------------------------------------------------------------------
void MortarData::assembleMortarWts( SurfaceContactElem & elem, SparseMode s_mode ) const
{
   // Note: check for active gaps here
   // grab the two meshes in this coupling scheme
   auto& mortarMesh = *elem.m_mesh1;
   auto& nonmortarMesh  = *elem.m_mesh2;

   // Note: The node ids between the two are assumed to 
   // be unique and contiguous using the int ids in the 
   // mortar and nonmortar mesh connectivity arrays with the number 
   // of nodes specified for each mesh being the number of mortar
   // and the number of nonmortar nodes for each mesh, respectively.
   //int numNodes = this->m_numTotalNodes;

   // loop over contact element nodes and assemble the local weight 
   // contributions

   // loop over nonmortar ROWS 
   for (int a = 0; a<elem.numFaceVert; ++a)
   {
      // weights will be mortar/nonmortar (a,b) or nonmortar/nonmortar (a,b)

      // get nonmortar node ROW id from connectivity
      int nonmortarNodeIdA = nonmortarMesh.getGlobalNodeId( elem.faceId2, a );

      // We include ALL nonmortar nodes, even if the nodal gap is in separation. 
      // NOTE: Per testing, we include ALL nonmortar node 
      // contributions for faces that have positive areas of overlap per the 
      // geometric filtering. Contact activity is not determined per 
      // gap interrogation, but rather gap AND pressure solution. For MORTAR_WEIGHTS, 
      // we just pass back all nonmortar node weights.

      // loop over mortar and nonmortar columns  
      for (int b = 0; b<elem.numFaceVert; ++b)
      {
         // get mortar COL id based on connectivity array
         int mortarNodeIdB = mortarMesh.getGlobalNodeId( elem.faceId1, b );

         // get nonmortar COL id based on connectivity array
         int nonmortarNodeIdB  = nonmortarMesh.getGlobalNodeId( elem.faceId2, b );

         // Add() will "set" if a nonzero entry hasn't been 
         // introduced at the (i,j) element
         if (s_mode == SparseMode::MFEM_LINKED_LIST)
         {
            // note: if we are here, the mfem sparse object, m_smat, has been allocated
            this->m_smat->Add( nonmortarNodeIdA, mortarNodeIdB, elem.getNonmortarMortarWt( a, b ) );
            this->m_smat->Add( nonmortarNodeIdA, nonmortarNodeIdB, elem.getNonmortarNonmortarWt( a, b ) );
         }
         else if (s_mode == SparseMode::MFEM_INDEX_SET)
         {
            SLIC_ERROR("MortarData::assembleMortarWts() MFEM_INDEX_SET " << 
                       "not implemented."); 
         }

      } // end loop over b nodes (columns)
   } // end loop over a nodes (rows)

   return;
} // end of MortarData::assembleMortarWts()

//------------------------------------------------------------------------------
void MortarData::getCSRArrays( int** I, int** J, RealT** vals,
                               int* n_offsets, int* n_nonzero)
{
   if (this->m_smat == nullptr)
   {
      SLIC_ERROR("getCSRArrays: method data get routine has m_smat == nullptr.");
   }

   this->m_smat->Finalize();
   *I = this->m_smat->GetI();
   *J = this->m_smat->GetJ();
   *vals = this->m_smat->GetData();

   // Set the sizes
   if(n_offsets != nullptr)
   {
      *n_offsets = this->m_smat->GetMemoryI().Capacity();
   }
   if(n_nonzero != nullptr)
   {
      *n_nonzero = this->m_smat->GetMemoryJ().Capacity();
   }

   return;
}

//------------------------------------------------------------------------------

} // end of namespace tribol
