// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MeshManager.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

// Axom includes
#include "axom/slic.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>

namespace tribol {

////////////////////////////////////////////////
//                                            //
// Routines for the SurfaceContactElem struct //
//                                            //
////////////////////////////////////////////////
void SurfaceContactElem::allocateMortarWts()
{
  // We store wts, n_aa and n_ab, which are nonmortar/nonmortar and nonmortar/mortar
  // products of shape functions
  this->numWts = 2 * this->numFaceVert * this->numFaceVert;
  if ( this->mortarWts != nullptr ) {
    delete[] this->mortarWts;
    this->mortarWts = new real[this->numWts];
  } else {
    this->mortarWts = new real[this->numWts];
  }

  this->initializeMortarWts();
}

//------------------------------------------------------------------------------
void SurfaceContactElem::initializeMortarWts()
{
  for ( int i = 0; i < this->numWts; ++i ) {
    mortarWts[i] = 0.;
  }
}

//------------------------------------------------------------------------------
void SurfaceContactElem::allocateBlockJ( EnforcementMethod enf )
{
  if ( enf == LAGRANGE_MULTIPLIER ) {
    this->blockJ.resize( 3, 3 );
  } else {
    this->blockJ.resize( 2, 2 );
  }

  // number of element displacement degrees of freedom
  integer nPrimal = this->dim * this->numFaceVert;
  for ( int i{}; i < 2; ++i ) {
    for ( int j{}; j < 2; ++j ) {
      blockJ( i, j ).SetSize( nPrimal, nPrimal );
    }
  }

  if ( enf == LAGRANGE_MULTIPLIER ) {
    // number of element Lagrange multiplier degrees of freedom
    integer nDual = this->numFaceVert;
    for ( int i{}; i < 2; ++i ) {
      blockJ( 2, i ).SetSize( nDual, nPrimal );
      // transpose
      blockJ( i, 2 ).SetSize( nPrimal, nDual );
    }
    blockJ( 2, 2 ).SetSize( nDual, nDual );
  }
}

//------------------------------------------------------------------------------
void SurfaceContactElem::deallocateElem()
{
  if ( this->mortarWts != nullptr ) {
    delete[] this->mortarWts;
    this->mortarWts = nullptr;
  }
}

//------------------------------------------------------------------------------
real SurfaceContactElem::getMortarNonmortarWt( const int a, const int b )
{
  // note: the mortar wts are stored in a stacked array with nonmortar-nonmortar
  //       wts followed by mortar-nonmortar weights in mortar-nonmortar ordering

  int mortarNonmortarId = this->numFaceVert * this->numFaceVert + this->numFaceVert * a + b;

  real wt = this->mortarWts[mortarNonmortarId];
  return wt;
}

//------------------------------------------------------------------------------
real SurfaceContactElem::getNonmortarMortarWt( const int a, const int b )
{
  // note: the mortar wts are stored in a stacked array with nonmortar-nonmortar
  //       wts followed by mortar-nonmortar weights in mortar-nonmortar ordering.
  //       Therefore, the nonmortarMortar id is the transpose of how it is
  //       stored

  int nonmortarMortarId = this->numFaceVert * this->numFaceVert + this->numFaceVert * b + a;

  real wt = this->mortarWts[nonmortarMortarId];
  return wt;
}

//------------------------------------------------------------------------------
real SurfaceContactElem::getNonmortarNonmortarWt( const int a, const int b )
{
  // note: the mortar wts are stored in a stacked array with nonmortar-nonmortar
  //       wts followed by mortar-nonmortar weights in mortar-nonmortar ordering

  int nonmortarNonmortarId = this->numFaceVert * a + b;

  real wt = this->mortarWts[nonmortarNonmortarId];
  return wt;
}

//------------------------------------------------------------------------------
int SurfaceContactElem::getJacobianIndex( JacBlock block, const int a, const int b ) const
{
  if ( block == JguBlock ) {
    return a + this->numFaceVert * b;
  }

  if ( block == JrpBlock ) {
    return a + this->dim * this->numFaceVert * b;
  }

  return -1;
}

//------------------------------------------------------------------------------
int SurfaceContactElem::getJacobianDimOffset( JacBlock block ) const
{
  if ( block == JguBlock ) {
    return this->numFaceVert * this->numFaceVert;
  }

  if ( block == JrpBlock ) {
    return this->numFaceVert;
  }

  return -1;
}

//------------------------------------------------------------------------------
MethodData::MethodData()
    : m_blockJElemIds( static_cast<axom::IndexType>( BlockSpace::NUM_BLOCK_SPACES ),
                       static_cast<axom::IndexType>( BlockSpace::NUM_BLOCK_SPACES ) ),
      m_blockJ( static_cast<axom::IndexType>( BlockSpace::NUM_BLOCK_SPACES ),
                static_cast<axom::IndexType>( BlockSpace::NUM_BLOCK_SPACES ) )
{
  m_blockJ.shrink();
}

//------------------------------------------------------------------------------
void MethodData::reserveBlockJ( axom::Array<BlockSpace>&& blockJSpaces, int nPairs )
{
  axom::IndexType pairFraction = static_cast<axom::IndexType>( 0.5 * nPairs );

  m_blockJSpaces = std::move( blockJSpaces );
  m_blockJElemIds.fill( axom::Array<integer>( 0, 0 ) );
  for ( auto blockJSpace : m_blockJSpaces ) {
    m_blockJElemIds[static_cast<axom::IndexType>( blockJSpace )].reserve( pairFraction );
  }
  m_blockJ.fill( axom::Array<mfem::DenseMatrix>( 0, 0 ) );
  for ( auto blockJSpaceI : m_blockJSpaces ) {
    for ( auto blockJSpaceJ : m_blockJSpaces ) {
      m_blockJ( static_cast<axom::IndexType>( blockJSpaceI ), static_cast<axom::IndexType>( blockJSpaceJ ) )
          .reserve( pairFraction );
    }
  }
}

//------------------------------------------------------------------------------
void MethodData::storeElemBlockJ( axom::Array<integer>&& blockJElemIds, axom::Array<mfem::DenseMatrix, 2>& blockJ )
{
  SLIC_ASSERT_MSG( blockJElemIds.size() == getNSpaces(),
                   "Number of element ID vectors does not match the number of Jacobian spaces." );
  SLIC_ASSERT_MSG( blockJ.shape()[0] == getNSpaces(),
                   "Number of rows in blockJ does not match the number of Jacobian spaces." );
  SLIC_ASSERT_MSG( blockJ.shape()[1] == getNSpaces(),
                   "Number of columns in blockJ does not match the number of Jacobian spaces." );
  for ( axom::IndexType i{}; i < getNSpaces(); ++i ) {
    axom::IndexType blockIdxI = static_cast<axom::IndexType>( m_blockJSpaces[i] );
    m_blockJElemIds[blockIdxI].push_back( blockJElemIds[i] );
    for ( axom::IndexType j{}; j < getNSpaces(); ++j ) {
      axom::IndexType blockIdxJ = static_cast<axom::IndexType>( m_blockJSpaces[j] );
      m_blockJ( blockIdxI, blockIdxJ ).push_back( blockJ( i, j ) );
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
  this->m_smat = nullptr;
}

//------------------------------------------------------------------------------
// Destructor
MortarData::~MortarData()
{
  if ( this->m_smat != nullptr ) {
    delete this->m_smat;
    this->m_smat = nullptr;
  }
}

//------------------------------------------------------------------------------
void MortarData::assembleJacobian( SurfaceContactElem& elem, SparseMode s_mode ) const
{
  (void)s_mode;  // will be used at a later point
  // grab the two meshes in this coupling scheme
  MeshManager& meshManager = MeshManager::getInstance();

  IndexType const mortarId = elem.meshId1;
  IndexType const nonmortarId = elem.meshId2;

  MeshData& mortarMesh = meshManager.GetMeshInstance( mortarId );
  MeshData& nonmortarMesh = meshManager.GetMeshInstance( nonmortarId );

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
  for ( int a = 0; a < elem.numFaceVert; ++a ) {
    // get mortar and nonmortar node ids from connectivity
    int mortarNodeIdA = mortarMesh.getFaceNodeId( elem.faceId1, a );
    int nonmortarNodeIdA = nonmortarMesh.getFaceNodeId( elem.faceId2, a );

    // loop over face nodes "b" (general loop, don't distinguish mortar/nonmortar
    // as that changes for Jrp and Jgu contributions)
    for ( int b = 0; b < elem.numFaceVert; ++b ) {
      // get nonmortar node id from connectivity
      int nonmortarNodeIdB = nonmortarMesh.getFaceNodeId( elem.faceId2, b );

      // We don't exclude nonmortar nodes that are in separation. NOTE: Per
      // testing, we include ALL nonmortar contributions for faces that
      // have positive areas of overlap regardless of gap evaluation. Contact
      // activity is determined from gaps AND pressure solution per KKT constraint
      // equations

      /////////////////////////////////////////////////////////////////
      // Assemble Jrp contributions (upper-right off-diagonal block) //
      /////////////////////////////////////////////////////////////////

      // get local ids into contact element Jacobian arrays
      int localId = elem.getJacobianIndex( SurfaceContactElem::JrpBlock, a, b );

      int i = elem.dim * mortarNodeIdA;       // mortar row contributions
      int j = presOffset + nonmortarNodeIdB;  // nonmortar column contributions always
      int dimOffset = elem.getJacobianDimOffset( SurfaceContactElem::JrpBlock );

      // Add() will "set" if a nonzero entry hasn't been
      // introduced at the (i,j) element
      // Mortar-Lagrange multiplier block (0, 2)
      this->m_smat->Add( i, j,
                         elem.blockJ( static_cast<axom::IndexType>( BlockSpace::MORTAR ),
                                      static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ) )
                             .Data()[localId] );
      this->m_smat->Add( i + 1, j,
                         elem.blockJ( static_cast<axom::IndexType>( BlockSpace::MORTAR ),
                                      static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ) )
                             .Data()[localId + dimOffset] );
      this->m_smat->Add( i + 2, j,
                         elem.blockJ( static_cast<axom::IndexType>( BlockSpace::MORTAR ),
                                      static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ) )
                             .Data()[localId + 2 * dimOffset] );  // assume 3D for now

      // Nonmortar-Lagrange Multiplier block (1, 2)
      i = elem.dim * nonmortarNodeIdA;  // nonmortar row contributions
      this->m_smat->Add( i, j,
                         elem.blockJ( static_cast<axom::IndexType>( BlockSpace::NONMORTAR ),
                                      static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ) )
                             .Data()[localId] );
      this->m_smat->Add( i + 1, j,
                         elem.blockJ( static_cast<axom::IndexType>( BlockSpace::NONMORTAR ),
                                      static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ) )
                             .Data()[localId + dimOffset] );
      this->m_smat->Add( i + 2, j,
                         elem.blockJ( static_cast<axom::IndexType>( BlockSpace::NONMORTAR ),
                                      static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ) )
                             .Data()[localId + 2 * dimOffset] );  // assume 3D for now

      ////////////////////////////////////////////////////////////////
      // Assemble Jgu contributions (lower-left off-diagonal block) //
      ////////////////////////////////////////////////////////////////

      // note b and a are swapped.  index a loops over displacement DOFs (i.e. the
      // columns of Jgu) and index b loops over pressure DOFs (rows of Jgu)
      localId = elem.getJacobianIndex( SurfaceContactElem::JguBlock, b, a );

      i = presOffset + nonmortarNodeIdB;  // nonmortar rows always
      j = elem.dim * mortarNodeIdA;       // mortar column contributions
      dimOffset = elem.getJacobianDimOffset( SurfaceContactElem::JguBlock );

      // Lagrange-multiplier-mortar block (2, 0)
      this->m_smat->Add( i, j,
                         elem.blockJ( static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ),
                                      static_cast<axom::IndexType>( BlockSpace::MORTAR ) )
                             .Data()[localId] );
      this->m_smat->Add( i, j + 1,
                         elem.blockJ( static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ),
                                      static_cast<axom::IndexType>( BlockSpace::MORTAR ) )
                             .Data()[localId + dimOffset] );
      this->m_smat->Add( i, j + 2,
                         elem.blockJ( static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ),
                                      static_cast<axom::IndexType>( BlockSpace::MORTAR ) )
                             .Data()[localId + 2 * dimOffset] );  // assume 3D for now

      // Lagrange multiplier-nonmortar block (2, 1)
      j = elem.dim * nonmortarNodeIdA;  // nonmortar column contributions
      this->m_smat->Add( i, j,
                         elem.blockJ( static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ),
                                      static_cast<axom::IndexType>( BlockSpace::NONMORTAR ) )
                             .Data()[localId] );
      this->m_smat->Add( i, j + 1,
                         elem.blockJ( static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ),
                                      static_cast<axom::IndexType>( BlockSpace::NONMORTAR ) )
                             .Data()[localId + dimOffset] );
      this->m_smat->Add( i, j + 2,
                         elem.blockJ( static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ),
                                      static_cast<axom::IndexType>( BlockSpace::NONMORTAR ) )
                             .Data()[localId + 2 * dimOffset] );  // assume 3D for now

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

    }  // end loop over b nodes
  }    // end loop over a nodes

  return;
}  // end of MortarData::assembleJacobian()

//------------------------------------------------------------------------------
void MortarData::assembleMortarWts( SurfaceContactElem& elem, SparseMode s_mode ) const
{
  // Note: check for active gaps here
  // grab the two meshes in this coupling scheme
  MeshManager& meshManager = MeshManager::getInstance();

  IndexType const mortarId = elem.meshId1;
  IndexType const nonmortarId = elem.meshId2;

  MeshData& mortarMesh = meshManager.GetMeshInstance( mortarId );
  MeshData& nonmortarMesh = meshManager.GetMeshInstance( nonmortarId );

  // Note: The node ids between the two are assumed to
  // be unique and contiguous using the integer ids in the
  // mortar and nonmortar mesh connectivity arrays with the number
  // of nodes specified for each mesh being the number of mortar
  // and the number of nonmortar nodes for each mesh, respectively.
  // int numNodes = this->m_numTotalNodes;

  // loop over contact element nodes and assemble the local weight
  // contributions

  // loop over nonmortar ROWS
  for ( int a = 0; a < elem.numFaceVert; ++a ) {
    // weights will be mortar/nonmortar (a,b) or nonmortar/nonmortar (a,b)

    // get nonmortar node ROW id from connectivity
    int nonmortarNodeIdA = nonmortarMesh.getFaceNodeId( elem.faceId2, a );

    // We include ALL nonmortar nodes, even if the nodal gap is in separation.
    // NOTE: Per testing, we include ALL nonmortar node
    // contributions for faces that have positive areas of overlap per the
    // geometric filtering. Contact activity is not determined per
    // gap interrogation, but rather gap AND pressure solution. For MORTAR_WEIGHTS,
    // we just pass back all nonmortar node weights.

    // loop over mortar and nonmortar columns
    for ( int b = 0; b < elem.numFaceVert; ++b ) {
      // get mortar COL id based on connectivity array
      int mortarNodeIdB = mortarMesh.getFaceNodeId( elem.faceId1, b );

      // get nonmortar COL id based on connectivity array
      int nonmortarNodeIdB = nonmortarMesh.getFaceNodeId( elem.faceId2, b );

      // Add() will "set" if a nonzero entry hasn't been
      // introduced at the (i,j) element
      if ( s_mode == SparseMode::MFEM_LINKED_LIST ) {
        // note: if we are here, the mfem sparse object, m_smat, has been allocated
        this->m_smat->Add( nonmortarNodeIdA, mortarNodeIdB, elem.getNonmortarMortarWt( a, b ) );
        this->m_smat->Add( nonmortarNodeIdA, nonmortarNodeIdB, elem.getNonmortarNonmortarWt( a, b ) );
      } else if ( s_mode == SparseMode::MFEM_INDEX_SET ) {
        SLIC_ERROR( "MortarData::assembleMortarWts() MFEM_INDEX_SET "
                    << "not implemented." );
      }

    }  // end loop over b nodes (columns)
  }    // end loop over a nodes (rows)

  return;
}  // end of MortarData::assembleMortarWts()

//------------------------------------------------------------------------------
void MortarData::getCSRArrays( int** I, int** J, real** vals, int* n_offsets, int* n_nonzero )
{
  if ( this->m_smat == nullptr ) {
    SLIC_ERROR( "getCSRArrays: method data get routine has m_smat == nullptr." );
  }

  this->m_smat->Finalize();
  *I = this->m_smat->GetI();
  *J = this->m_smat->GetJ();
  *vals = this->m_smat->GetData();

  // Set the sizes
  if ( n_offsets != nullptr ) {
    *n_offsets = this->m_smat->GetMemoryI().Capacity();
  }
  if ( n_nonzero != nullptr ) {
    *n_nonzero = this->m_smat->GetMemoryJ().Capacity();
  }

  return;
}

//------------------------------------------------------------------------------

}  // end of namespace tribol
