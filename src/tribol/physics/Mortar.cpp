// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "Mortar.hpp"

#include "tribol/types.hpp"
#include "tribol/mesh/MeshManager.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/CouplingSchemeManager.hpp"
#include "tribol/geom/ContactPlane.hpp"
#include "tribol/geom/ContactPlaneManager.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/integ/Integration.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/utils/ContactPlaneOutput.hpp"
#include "tribol/utils/Math.hpp"

// Axom includes
#include "axom/slic.hpp"

#include <iostream>

namespace tribol {

void ComputeMortarWeights( SurfaceContactElem& elem )
{
  // instantiate integration object
  IntegPts integ;

  // Debug: leave code in for now to call Gauss quadrature on triangle rule
  GaussPolyIntTri( elem, integ, 3 );

  // call Taylor-Wingate-Bos integation rule. NOTE: this is not
  // working. The correct gaps are not being computed.
  //   TWBPolyInt( elem, integ, 3 );

  // get individual arrays of coordinates for each face
  real x1[elem.numFaceVert];
  real y1[elem.numFaceVert];
  real z1[elem.numFaceVert];
  real x2[elem.numFaceVert];
  real y2[elem.numFaceVert];
  real z2[elem.numFaceVert];

  for ( int i = 0; i < elem.numFaceVert; ++i ) {
    x1[i] = elem.faceCoords1[elem.dim * i];
    y1[i] = elem.faceCoords1[elem.dim * i + 1];
    z1[i] = elem.faceCoords1[elem.dim * i + 2];
    x2[i] = elem.faceCoords2[elem.dim * i];
    y2[i] = elem.faceCoords2[elem.dim * i + 1];
    z2[i] = elem.faceCoords2[elem.dim * i + 2];
  }

  // allocate mortar weights array on SurfaceContactElem object. This routine
  // also initializes the array
  elem.allocateMortarWts();

  real phiNonmortarA, phiNonmortarB, phiMortarA;

  // loop over number of nodes on the nonmortar or mortar depending on whether forming
  // nonmortar/nonmortar or mortar/nonmortar weights

  for ( int a = 0; a < elem.numFaceVert; ++a ) {
    // loop over number of nodes on nonmortar side
    for ( int b = 0; b < elem.numFaceVert; ++b ) {
      // set nonmortar/nonmortar and mortar/nonmortar ids...Don't change these ids
      int nonmortarNonmortarId = elem.numFaceVert * a + b;
      int mortarNonmortarId = elem.numFaceVert * elem.numFaceVert + elem.numFaceVert * a + b;

      // loop over number of integration points
      for ( int ip = 0; ip < integ.numIPs; ++ip ) {
        // The integration method for computing weights uses
        // the inverse isoparametric mapping of a current configuration
        // integration point (as projected onto the current configuration
        // face) to obtain a (xi,eta) coordinate pair in parent space
        // for the evaluation of Lagrange shape functions
        real xp[3] = { integ.xy[elem.dim * ip], integ.xy[elem.dim * ip + 1], integ.xy[elem.dim * ip + 2] };
        real xi[2] = { 0., 0. };

        InvIso( xp, x1, y1, z1, elem.numFaceVert, xi );
        LinIsoQuadShapeFunc( xi[0], xi[1], a, phiMortarA );

        InvIso( xp, x2, y2, z2, elem.numFaceVert, xi );
        LinIsoQuadShapeFunc( xi[0], xi[1], a, phiNonmortarA );
        LinIsoQuadShapeFunc( xi[0], xi[1], b, phiNonmortarB );

        SLIC_ERROR_IF( nonmortarNonmortarId > elem.numWts || mortarNonmortarId > elem.numWts,
                       "ComputeMortarWts: integer ids for weights exceed elem.numWts" );

        // compute nonmortar/nonmortar mortar weight
        elem.mortarWts[nonmortarNonmortarId] += integ.wts[ip] * phiNonmortarA * phiNonmortarB;

        // compute mortar/nonmortar mortar weight
        elem.mortarWts[mortarNonmortarId] += integ.wts[ip] * phiMortarA * phiNonmortarB;

      }  // end loop over integration points

    }  // end loop over nodes on side 2

  }  // end loop over nodes on side 1

}  // end ComputeMortarWeights()

//------------------------------------------------------------------------------
template <>
void ComputeNodalGap<SINGLE_MORTAR>( SurfaceContactElem& elem )
{
  // check to make sure mortar weights have been computed locally
  // for the SurfaceContactElem object
  SLIC_ERROR_IF( elem.mortarWts == nullptr,
                 "ComputeNodalGap< SINGLE_MORTAR >: compute local weights on input struct first." );

  // get mesh instance to store gaps on mesh data object
  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& nonmortarMesh = meshManager.GetMeshInstance( elem.meshId2 );
  IndexType const* const nonmortarConn = nonmortarMesh.m_connectivity;

  // will populate local gaps on nonmortar face on nonmortar mesh data object
  SLIC_ERROR_IF( nonmortarMesh.m_nodalFields.m_node_gap == nullptr,
                 "ComputeNodalGap< SINGLE_MORTAR >: allocate gaps on mesh data object." );

  SLIC_ERROR_IF( nonmortarMesh.m_node_nX == nullptr || nonmortarMesh.m_node_nY == nullptr,
                 "ComputeNodalGap< SINGLE_MORTAR >: allocate and compute nodal normals on mesh data object." );

  // compute gap contributions associated with face 2 on the SurfaceContactElem
  // (i.e. nonmortar surface)

  // loop over nodes on nonmortar side
  for ( int a = 0; a < elem.numFaceVert; ++a ) {
    // initialize gap1 and gap2 terms
    real g1 = 0.;
    real g2 = 0.;

    // get global nonmortar node number from connectivity
    real nrml_a[elem.dim];
    int glbId = nonmortarConn[elem.numFaceVert * elem.faceId2 + a];
    nrml_a[0] = nonmortarMesh.m_node_nX[glbId];
    nrml_a[1] = nonmortarMesh.m_node_nY[glbId];
    if ( elem.dim == 3 ) {
      nrml_a[2] = nonmortarMesh.m_node_nZ[glbId];
    }

    // sum contributions from both sides
    for ( int b = 0; b < elem.numFaceVert; ++b ) {
      // compute nonmortar-mortar and nonmortar-nonmortar ids. Note, n_ab is
      // the stored mortar weight. For mortar-nonmortar mortar weights,
      // a = mortar node and b = nonmortar node, BUT FOR THE GAP COMPUTATION,
      // THE SUM OF MORTAR WEIGHTS IS ACTUALLY OVER SHAPE FUNCTIONS
      // DEFINED AT NODE "b", SO WE NEED TO USE (n_ab)^T.
      real nab_1 = elem.getNonmortarMortarWt( a, b );     // nonmortar-mortar weight
      real nab_2 = elem.getNonmortarNonmortarWt( a, b );  // nonmortar-nonmortar weight

      g1 += dotProd( &nrml_a[0], &elem.faceCoords1[elem.dim * b], elem.dim ) * nab_1;
      g2 += dotProd( &nrml_a[0], &elem.faceCoords2[elem.dim * b], elem.dim ) * nab_2;
    }

    // store local gap
    nonmortarMesh.m_nodalFields.m_node_gap[glbId] += ( g1 - g2 );

  }  // end a-loop over nonmortar nodes

}  // end ComputeNodalGap<>()

//------------------------------------------------------------------------------
void ComputeSingleMortarGaps( CouplingScheme const* cs )
{
  InterfacePairs const* const pairs = cs->getInterfacePairs();
  IndexType const numPairs = pairs->getNumPairs();

  MeshManager& meshManager = MeshManager::getInstance();
  ContactPlaneManager& cpManager = ContactPlaneManager::getInstance();
  parameters_t& parameters = parameters_t::getInstance();
  integer const dim = parameters.dimension;

  ////////////////////////////////////////////////////////////////////////
  //
  // Grab pointers to mesh data
  //
  ////////////////////////////////////////////////////////////////////////
  IndexType const mortarId = cs->getMeshId1();
  IndexType const nonmortarId = cs->getMeshId2();

  MeshData& mortarMesh = meshManager.GetMeshInstance( mortarId );
  MeshData& nonmortarMesh = meshManager.GetMeshInstance( nonmortarId );
  IndexType const numNodesPerFace = mortarMesh.m_numNodesPerCell;

  real const* const x1 = mortarMesh.m_positionX;
  real const* const y1 = mortarMesh.m_positionY;
  real const* const z1 = mortarMesh.m_positionZ;
  IndexType const* const mortarConn = mortarMesh.m_connectivity;

  real const* const x2 = nonmortarMesh.m_positionX;
  real const* const y2 = nonmortarMesh.m_positionY;
  real const* const z2 = nonmortarMesh.m_positionZ;
  IndexType const* nonmortarConn = nonmortarMesh.m_connectivity;

  // compute nodal normals (do this outside the element loop)
  // Note, this is guarded against zero element meshes
  nonmortarMesh.computeNodalNormals( dim );

  // declare local variables to hold face nodal coordinates
  // and overlap vertex coordinates
  IndexType size = dim * numNodesPerFace;
  real mortarX[size];
  real nonmortarX[size];

  // projected coords
  real mortarX_bar[size];
  real nonmortarX_bar[size];
  real* overlapX;

  ////////////////////////////////////////////////////////////////////
  // compute nonmortar gaps to determine active set of contact dofs //
  ////////////////////////////////////////////////////////////////////
  int cpID = 0;
  for ( IndexType kp = 0; kp < numPairs; ++kp ) {
    InterfacePair pair = pairs->getInterfacePair( kp );

    if ( !pair.isContactCandidate ) {
      continue;
    }

    // get pair indices
    IndexType index1 = pair.pairIndex1;
    IndexType index2 = pair.pairIndex2;

    // populate the current configuration nodal coordinates for the
    // two faces
    for ( int i = 0; i < numNodesPerFace; ++i ) {
      int id = dim * i;
      IndexType mortar_id = mortarConn[numNodesPerFace * index1 + i];
      IndexType nonmortar_id = nonmortarConn[numNodesPerFace * index2 + i];

      mortarX[id] = x1[mortar_id];
      mortarX[id + 1] = y1[mortar_id];
      mortarX[id + 2] = z1[mortar_id];
      nonmortarX[id] = x2[nonmortar_id];
      nonmortarX[id + 1] = y2[nonmortar_id];
      nonmortarX[id + 2] = z2[nonmortar_id];

      // arrays for projected coords
      mortarX_bar[id] = x1[mortar_id];
      mortarX_bar[id + 1] = y1[mortar_id];
      mortarX_bar[id + 2] = z1[mortar_id];
      nonmortarX_bar[id] = x2[nonmortar_id];
      nonmortarX_bar[id + 1] = y2[nonmortar_id];
      nonmortarX_bar[id + 2] = z2[nonmortar_id];
    }

    overlapX = new real[dim * cpManager.m_numPolyVert[cpID]];
    initRealArray( overlapX, dim * cpManager.m_numPolyVert[cpID], 0. );

    // get projected face coordinates
    cpManager.getProjectedFaceCoords( cpID, 0, &mortarX_bar[0] );     // face 0 = first face
    cpManager.getProjectedFaceCoords( cpID, 1, &nonmortarX_bar[0] );  // face 1 = second face

    // construct array of polygon overlap vertex coordinates
    cpManager.getContactPlaneOverlapVerts( cpID, cpManager.m_numPolyVert[cpID], &overlapX[0] );

    // instantiate contact surface element for purposes of computing
    // mortar weights. Note, this uses projected face coords
    SurfaceContactElem elem( dim, &mortarX_bar[0], &nonmortarX_bar[0], &overlapX[0], numNodesPerFace,
                             cpManager.m_numPolyVert[cpID], mortarId, nonmortarId, index1, index2 );

    // compute the mortar weights to be stored on the surface
    // contact element struct. This must be done prior to computing nodal gaps
    elem.overlapArea = cpManager.m_area[cpID];
    ComputeMortarWeights( elem );

    // compute mortar gaps. Note, we have to now use current configuration
    // nodal coordinates on the contact element
    elem.faceCoords1 = &mortarX[0];
    elem.faceCoords2 = &nonmortarX[0];

    ComputeNodalGap<SINGLE_MORTAR>( elem );

    // TODO: fix this to register the actual number of active nonmortar gaps.
    // This is not the appropriate data structure to put this information in
    // as the SurfaceContactElem goes out of scope when we exit the loop.
    // HAVE TO set the number of active constraints. For now set to
    // all nonmortar face nodes.
    elem.numActiveGaps = numNodesPerFace;

    ++cpID;

    delete[] overlapX;

  }  // end loop over pairs to compute nodal gaps

  nonmortarMesh.m_nodalFields.m_isGapComputed = true;

}  // end ComputeSingleMortarGaps()

//------------------------------------------------------------------------------
template <>
int ApplyNormal<SINGLE_MORTAR, LAGRANGE_MULTIPLIER>( CouplingScheme const* cs )
{
  InterfacePairs const* const pairs = cs->getInterfacePairs();
  IndexType const numPairs = pairs->getNumPairs();

  MeshManager& meshManager = MeshManager::getInstance();
  ContactPlaneManager& cpManager = ContactPlaneManager::getInstance();
  parameters_t& parameters = parameters_t::getInstance();
  integer const dim = parameters.dimension;

  ////////////////////////////////
  //                            //
  // Grab pointers to mesh data //
  //                            //
  ////////////////////////////////
  IndexType const mortarId = cs->getMeshId1();
  IndexType const nonmortarId = cs->getMeshId2();

  MeshData& mortarMesh = meshManager.GetMeshInstance( mortarId );
  MeshData& nonmortarMesh = meshManager.GetMeshInstance( nonmortarId );
  IndexType const numNodesPerFace = mortarMesh.m_numNodesPerCell;

  real const* const x1 = mortarMesh.m_positionX;
  real const* const y1 = mortarMesh.m_positionY;
  real const* const z1 = mortarMesh.m_positionZ;
  real* const fx1 = mortarMesh.m_forceX;
  real* const fy1 = mortarMesh.m_forceY;
  real* const fz1 = mortarMesh.m_forceZ;
  IndexType const* const mortarConn = mortarMesh.m_connectivity;

  real const* const x2 = nonmortarMesh.m_positionX;
  real const* const y2 = nonmortarMesh.m_positionY;
  real const* const z2 = nonmortarMesh.m_positionZ;
  real* const fx2 = nonmortarMesh.m_forceX;
  real* const fy2 = nonmortarMesh.m_forceY;
  real* const fz2 = nonmortarMesh.m_forceZ;
  IndexType const* nonmortarConn = nonmortarMesh.m_connectivity;

  // projected coords
  real mortarX_bar[dim * numNodesPerFace];
  real nonmortarX_bar[dim * numNodesPerFace];
  real* overlapX;  // [dim * cpManager.m_numPolyVert[cpID]];

  ///////////////////////////////////////////////////////
  //                                                   //
  //            compute single mortar gaps             //
  //                                                   //
  // Note, this routine is guarded against null meshes //
  ///////////////////////////////////////////////////////
  ComputeSingleMortarGaps( cs );

  int numTotalNodes = cs->getNumTotalNodes();
  int numRows = dim * numTotalNodes + numTotalNodes;
  const EnforcementOptions& enforcement_options = const_cast<EnforcementOptions&>( cs->getEnforcementOptions() );
  const LagrangeMultiplierImplicitOptions& lm_options = enforcement_options.lm_implicit_options;
  if ( !cs->nullMeshes() ) {
    if ( lm_options.sparse_mode == SparseMode::MFEM_ELEMENT_DENSE ) {
      static_cast<MortarData*>( cs->getMethodData() )
          ->reserveBlockJ( { BlockSpace::MORTAR, BlockSpace::NONMORTAR, BlockSpace::LAGRANGE_MULTIPLIER }, numPairs );
    } else if ( lm_options.sparse_mode == SparseMode::MFEM_INDEX_SET ||
                lm_options.sparse_mode == SparseMode::MFEM_LINKED_LIST ) {
      static_cast<MortarData*>( cs->getMethodData() )->allocateMfemSparseMatrix( numRows );
    } else {
      SLIC_WARNING( "Unsupported Jacobian storage method." );
      return 1;
    }
  }

  ////////////////////////////////////////////////////////////////
  //                                                            //
  // compute equilibrium residual and/or Jacobian contributions //
  //                                                            //
  ////////////////////////////////////////////////////////////////
  int cpID = 0;
  for ( IndexType kp = 0; kp < numPairs; ++kp ) {
    InterfacePair pair = pairs->getInterfacePair( kp );

    if ( !pair.isContactCandidate ) {
      continue;
    }

    // get pair indices
    IndexType index1 = pair.pairIndex1;
    IndexType index2 = pair.pairIndex2;

    // populate the current configuration nodal coordinates for the
    // two faces
    for ( int i = 0; i < numNodesPerFace; ++i ) {
      int id = dim * i;
      IndexType mortar_id = mortarConn[numNodesPerFace * index1 + i];
      IndexType nonmortar_id = nonmortarConn[numNodesPerFace * index2 + i];
      // arrays for projected coords
      mortarX_bar[id] = x1[mortar_id];
      mortarX_bar[id + 1] = y1[mortar_id];
      mortarX_bar[id + 2] = z1[mortar_id];
      nonmortarX_bar[id] = x2[nonmortar_id];
      nonmortarX_bar[id + 1] = y2[nonmortar_id];
      nonmortarX_bar[id + 2] = z2[nonmortar_id];
    }

    overlapX = new real[dim * cpManager.m_numPolyVert[cpID]];
    initRealArray( overlapX, dim * cpManager.m_numPolyVert[cpID], 0. );

    // get projected face coordinates
    cpManager.getProjectedFaceCoords( cpID, 0, &mortarX_bar[0] );     // face 0 = first face
    cpManager.getProjectedFaceCoords( cpID, 1, &nonmortarX_bar[0] );  // face 1 = second face

    // construct array of polygon overlap vertex coordinates
    cpManager.getContactPlaneOverlapVerts( cpID, cpManager.m_numPolyVert[cpID], &overlapX[0] );

    // instantiate contact surface element for purposes of computing
    // mortar weights. Note, this uses projected face coords
    SurfaceContactElem elem( dim, &mortarX_bar[0], &nonmortarX_bar[0], &overlapX[0], numNodesPerFace,
                             cpManager.m_numPolyVert[cpID], mortarId, nonmortarId, index1, index2 );

    //////////////////////////////////
    // compute equilibrium residual //
    //////////////////////////////////

    // compute mortar weight
    elem.overlapArea = cpManager.m_area[cpID];
    ComputeMortarWeights( elem );

    // TODO fix this. This may not be required.
    // HAVE TO set the number of active constraints. For now set to
    // all nonmortar face nodes.
    elem.numActiveGaps = numNodesPerFace;

    // loop over face nodes (BOTH MORTAR and NONMORTAR
    // contributions)
    for ( int a = 0; a < numNodesPerFace; ++a ) {
      int mortarIdA = mortarConn[index1 * numNodesPerFace + a];
      int nonmortarIdA = nonmortarConn[index2 * numNodesPerFace + a];

      // inner loop over NONMORTAR nodes
      for ( int b = 0; b < numNodesPerFace; ++b ) {
        int nonmortarIdB = nonmortarConn[index2 * numNodesPerFace + b];

        // We include all nonmortar nodes even if nodal gap is in separation.
        // NOTE: Per testing, we include ALL nonmortar nodes
        // in the computation after the geometric filtering and judge contact
        // activity based on the gap AND the pressure solution

        real forceX = nonmortarMesh.m_nodalFields.m_node_pressure[nonmortarIdB] * nonmortarMesh.m_node_nX[nonmortarIdB];
        real forceY = nonmortarMesh.m_nodalFields.m_node_pressure[nonmortarIdB] * nonmortarMesh.m_node_nY[nonmortarIdB];
        real forceZ = nonmortarMesh.m_nodalFields.m_node_pressure[nonmortarIdB] * nonmortarMesh.m_node_nZ[nonmortarIdB];

        // contact nodal force is the interpolated force using mortar
        // weights n_ab, where "a" is mortar or nonmortar node and "b" is
        // nonmortar node.
        fx1[mortarIdA] += forceX * elem.getMortarNonmortarWt( a, b );
        fy1[mortarIdA] += forceY * elem.getMortarNonmortarWt( a, b );
        fz1[mortarIdA] += forceZ * elem.getMortarNonmortarWt( a, b );

        fx2[nonmortarIdA] -= forceX * elem.getNonmortarNonmortarWt( a, b );
        fy2[nonmortarIdA] -= forceY * elem.getNonmortarNonmortarWt( a, b );
        fz2[nonmortarIdA] -= forceZ * elem.getNonmortarNonmortarWt( a, b );

      }  // end inner loop over nonmortar nodes

    }  // end outer loop over nonmortar and mortar nodes

    //////////////////////////////////////////////////////////
    // compute tangent stiffness contributions if requested //
    //////////////////////////////////////////////////////////
    if ( lm_options.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN ||
         lm_options.eval_mode == ImplicitEvalMode::MORTAR_JACOBIAN ) {
      ComputeSingleMortarJacobian( elem );
      if ( lm_options.sparse_mode == SparseMode::MFEM_ELEMENT_DENSE ) {
        static_cast<MortarData*>( cs->getMethodData() )
            ->storeElemBlockJ( { elem.faceId1, elem.faceId2, elem.faceId2 }, elem.blockJ );
      } else if ( lm_options.sparse_mode == SparseMode::MFEM_INDEX_SET ||
                  lm_options.sparse_mode == SparseMode::MFEM_LINKED_LIST ) {
        static_cast<MortarData*>( cs->getMethodData() )->assembleJacobian( elem, lm_options.sparse_mode );
      } else {
        SLIC_WARNING( "Unsupported Jacobian storage method." );
        return 1;
      }
    }

    ++cpID;

    delete[] overlapX;

  }  // end of loop over interface pairs computing residual/Jacobian contributions

  return 0;

}  // end ApplyNormal<>()

//------------------------------------------------------------------------------
template <>
void ComputeResidualJacobian<SINGLE_MORTAR, PRIMAL>( SurfaceContactElem& TRIBOL_UNUSED_PARAM( elem ) )
{
  // There is no Jacobian contribution for this block. Be safe and zero out...
  return;
}

//------------------------------------------------------------------------------
template <>
void ComputeResidualJacobian<SINGLE_MORTAR, DUAL>( SurfaceContactElem& elem )
{
  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& nonmortarMesh = meshManager.GetMeshInstance( elem.meshId2 );
  IndexType const* const nonmortarConn = nonmortarMesh.m_connectivity;

  // loop over "a" nodes accumulating sums of mortar/nonmortar
  // and nonmortar/nonmortar weights
  for ( int a = 0; a < elem.numFaceVert; ++a ) {
    // single loop over "b" nodes accumulating sums of
    // mortar(a)/nonmortar(b) and nonmortar(a)/nonmortar(b) weights
    for ( int b = 0; b < elem.numFaceVert; ++b ) {
      // get global nonmortar node id to index into nodal normals on
      // nonmortar mesh
      real nrml_b[elem.dim];
      int glbId = nonmortarConn[elem.numFaceVert * elem.faceId2 + b];

      // We assemble ALL nonmortar node contributions, even if gap is in separation.
      // NOTE: Per testing, we compute ALL nonmortar nodes
      // for faces that have positive areas of overlap after the geometric
      // filtering and use the gap AND the pressure solution to determine
      // contact activity

      nrml_b[0] = nonmortarMesh.m_node_nX[glbId];
      nrml_b[1] = nonmortarMesh.m_node_nY[glbId];
      if ( elem.dim == 3 ) {
        nrml_b[2] = nonmortarMesh.m_node_nZ[glbId];
      }

      // get mortar-nonmortar and nonmortar-nonmortar mortar weights
      real n_mortar_b = elem.getMortarNonmortarWt( a, b );  // mortar-nonmortar weight
      real n_nonmortar_b =
          elem.getNonmortarNonmortarWt( a, b );  // nonmortar-nonmortar weight, note negative in formulation

      // fill Jrp element-pair Jacobian blocks
      // Fill block (0, 2)
      int elem_xdof = elem.getJacobianIndex( SurfaceContactElem::JrpBlock, a, b );
      int dim_offset = elem.getJacobianDimOffset( SurfaceContactElem::JrpBlock );
      elem.blockJ( static_cast<axom::IndexType>( BlockSpace::MORTAR ),
                   static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ) )
          .Data()[elem_xdof] += nrml_b[0] * n_mortar_b;
      elem.blockJ( static_cast<axom::IndexType>( BlockSpace::MORTAR ),
                   static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ) )
          .Data()[elem_xdof + dim_offset] += nrml_b[1] * n_mortar_b;
      elem.blockJ( static_cast<axom::IndexType>( BlockSpace::MORTAR ),
                   static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ) )
          .Data()[elem_xdof + 2 * dim_offset] += nrml_b[2] * n_mortar_b;

      // Fill block (1, 2)
      elem.blockJ( static_cast<axom::IndexType>( BlockSpace::NONMORTAR ),
                   static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ) )
          .Data()[elem_xdof] -= nrml_b[0] * n_nonmortar_b;
      elem.blockJ( static_cast<axom::IndexType>( BlockSpace::NONMORTAR ),
                   static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ) )
          .Data()[elem_xdof + dim_offset] -= nrml_b[1] * n_nonmortar_b;
      elem.blockJ( static_cast<axom::IndexType>( BlockSpace::NONMORTAR ),
                   static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ) )
          .Data()[elem_xdof + 2 * dim_offset] -= nrml_b[2] * n_nonmortar_b;

    }  // end loop over b nodes

  }  // end loop over a nodes

  return;
}  // end ComputeResidualJacobian<>()

//------------------------------------------------------------------------------
template <>
void ComputeConstraintJacobian<SINGLE_MORTAR, PRIMAL>( SurfaceContactElem& elem )
{
  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& nonmortarMesh = meshManager.GetMeshInstance( elem.meshId2 );
  IndexType const* const nonmortarConn = nonmortarMesh.m_connectivity;

  // loop over nonmortar nodes for which we are accumulating Jacobian
  // contributions
  for ( int a = 0; a < elem.numFaceVert; ++a ) {
    // get global nonmortar node id to index into nodal normals on
    // nonmortar mesh
    real nrml_a[elem.dim];
    int glbId = nonmortarConn[elem.numFaceVert * elem.faceId2 + a];

    // We assemble ALL nonmortar node contributions even if gap is in separation.
    // NOTE: Per mortar method testing we compute ALL nonmortar node
    // contributions for faces that have positive areas of overlap per the
    // geometric filtering. Contact activity is judged based on gaps AND
    // the pressure solution.

    nrml_a[0] = nonmortarMesh.m_node_nX[glbId];
    nrml_a[1] = nonmortarMesh.m_node_nY[glbId];
    if ( elem.dim == 3 ) {
      nrml_a[2] = nonmortarMesh.m_node_nZ[glbId];
    }

    // single loop over "b" nodes accumulating sums of
    // nonmortar(a)/mortar(b) and nonmortar(a)/nonmortar(b) weights
    for ( int b = 0; b < elem.numFaceVert; ++b ) {
      // get nonmortar-mortar and nonmortar-nonmortar mortar weights
      real n_mortar_a = elem.getNonmortarMortarWt( a, b );  // nonmortar-mortar weight
      real n_nonmortar_a =
          elem.getNonmortarNonmortarWt( a, b );  // nonmortar-nonmortar weight, note negative in formulation

      // fill Jgu element-pair Jacobian blocks
      // Fill block (2, 0)
      int dim_offset = elem.getJacobianDimOffset( SurfaceContactElem::JguBlock );
      int elem_xdof = elem.getJacobianIndex( SurfaceContactElem::JguBlock, a, b );
      elem.blockJ( static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<axom::IndexType>( BlockSpace::MORTAR ) )
          .Data()[elem_xdof] += nrml_a[0] * n_mortar_a;
      elem.blockJ( static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<axom::IndexType>( BlockSpace::MORTAR ) )
          .Data()[elem_xdof + dim_offset] += nrml_a[1] * n_mortar_a;
      elem.blockJ( static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<axom::IndexType>( BlockSpace::MORTAR ) )
          .Data()[elem_xdof + 2 * dim_offset] += nrml_a[2] * n_mortar_a;

      // Fill block (2, 1)
      elem.blockJ( static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<axom::IndexType>( BlockSpace::NONMORTAR ) )
          .Data()[elem_xdof] -= nrml_a[0] * n_nonmortar_a;
      elem.blockJ( static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<axom::IndexType>( BlockSpace::NONMORTAR ) )
          .Data()[elem_xdof + dim_offset] -= nrml_a[1] * n_nonmortar_a;
      elem.blockJ( static_cast<axom::IndexType>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<axom::IndexType>( BlockSpace::NONMORTAR ) )
          .Data()[elem_xdof + 2 * dim_offset] -= nrml_a[2] * n_nonmortar_a;

    }  // end loop over b nodes

  }  // end loop over a nodes

  return;
}  // end ComputeConstraintJacobian

//------------------------------------------------------------------------------
template <>
void ComputeConstraintJacobian<SINGLE_MORTAR, DUAL>( SurfaceContactElem& TRIBOL_UNUSED_PARAM( elem ) )
{
  // unless we end up solving the complementarity equation, there is
  // no Jacobian contribtion for this block. Zero out to be safe...
  return;
}

//------------------------------------------------------------------------------
void ComputeSingleMortarJacobian( SurfaceContactElem& elem )
{
  elem.allocateBlockJ( LAGRANGE_MULTIPLIER );

  ComputeResidualJacobian<SINGLE_MORTAR, PRIMAL>( elem );

  ComputeResidualJacobian<SINGLE_MORTAR, DUAL>( elem );

  ComputeConstraintJacobian<SINGLE_MORTAR, PRIMAL>( elem );

  ComputeConstraintJacobian<SINGLE_MORTAR, DUAL>( elem );

  // Optionally print contact element matrix. Keep commented out here.
  // elem.printBlockJMatrix();

  return;
}

//------------------------------------------------------------------------------
template <>
int GetMethodData<MORTAR_WEIGHTS>( CouplingScheme const* cs )
{
  InterfacePairs const* const pairs = cs->getInterfacePairs();
  IndexType const numPairs = pairs->getNumPairs();

  ContactPlaneManager& cpManager = ContactPlaneManager::getInstance();
  parameters_t& parameters = parameters_t::getInstance();
  integer const dim = parameters.dimension;

  IndexType const mortarId = cs->getMeshId1();
  IndexType const nonmortarId = cs->getMeshId2();
  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& mortarMesh = meshManager.GetMeshInstance( mortarId );
  IndexType const numNodesPerFace = mortarMesh.m_numNodesPerCell;

  ////////////////////////////////
  //                            //
  // compute single mortar gaps //
  //                            //
  ////////////////////////////////
  ComputeSingleMortarGaps( cs );

  int numRows = cs->getNumTotalNodes();
  static_cast<MortarData*>( cs->getMethodData() )->allocateMfemSparseMatrix( numRows );

  //////////////////////////////////////////////
  //                                          //
  // aggregate data to compute mortar weights //
  //                                          //
  //////////////////////////////////////////////

  // declare local variables to hold face nodal coordinates
  // and overlap vertex coordinates
  IndexType size = dim * numNodesPerFace;

  // projected coords
  real mortarX_bar[size];
  real nonmortarX_bar[size];
  initRealArray( &mortarX_bar[0], size, 0. );
  initRealArray( &nonmortarX_bar[0], size, 0. );

  real* overlapX;  // [dim * cpManager.m_numPolyVert[cpID]];

  int cpID = 0;
  for ( IndexType kp = 0; kp < numPairs; ++kp ) {
    InterfacePair pair = pairs->getInterfacePair( kp );

    if ( !pair.isContactCandidate ) {
      continue;
    }

    // get pair indices
    IndexType index1 = pair.pairIndex1;
    IndexType index2 = pair.pairIndex2;

    overlapX = new real[dim * cpManager.m_numPolyVert[cpID]];
    initRealArray( overlapX, dim * cpManager.m_numPolyVert[cpID], 0. );

    // get projected face coordinates
    cpManager.getProjectedFaceCoords( cpID, 0, &mortarX_bar[0] );     // face 0 = first face
    cpManager.getProjectedFaceCoords( cpID, 1, &nonmortarX_bar[0] );  // face 1 = second face

    // construct array of polygon overlap vertex coordinates
    cpManager.getContactPlaneOverlapVerts( cpID, cpManager.m_numPolyVert[cpID], &overlapX[0] );

    // instantiate contact surface element for purposes of computing
    // mortar weights. Note, this uses projected face coords
    SurfaceContactElem elem( dim, &mortarX_bar[0], &nonmortarX_bar[0], &overlapX[0], numNodesPerFace,
                             cpManager.m_numPolyVert[cpID], mortarId, nonmortarId, index1, index2 );

    // compute the mortar weights to be stored on the surface
    // contact element struct. This must be done prior to computing nodal gaps
    elem.overlapArea = cpManager.m_area[cpID];

    ComputeMortarWeights( elem );

    elem.numActiveGaps = numNodesPerFace;

    // assemble mortar weight contributions sum_alpha int_alpha phi_a phi_b da.
    // Note: active nonmortar nodes (i.e. active gaps) are checked in this routine.
    const EnforcementOptions& enforcement_options = const_cast<EnforcementOptions&>( cs->getEnforcementOptions() );
    const SparseMode sparse_mode = enforcement_options.lm_implicit_options.sparse_mode;
    if ( sparse_mode == SparseMode::MFEM_ELEMENT_DENSE ) {
      SLIC_WARNING( "GetMethodData<MORTAR_WEIGHTS>() MFEM_ELEMENT_DENSE "
                    << "Unassembled element dense matrix output not implemented." );
      return 1;
    }
    static_cast<MortarData*>( cs->getMethodData() )->assembleMortarWts( elem, sparse_mode );

    ++cpID;

    delete[] overlapX;

  }  // end loop over pairs to assemble mortar weights

  return 0;

}  // end GetMethodData< MORTAR_WEIGHTS >()

//------------------------------------------------------------------------------

}  // end namespace tribol
