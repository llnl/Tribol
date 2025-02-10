// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "AlignedMortar.hpp"

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

#include <fstream>
#include <iostream>
#include <iomanip>

namespace tribol {

void ComputeAlignedMortarWeights( SurfaceContactElem& elem )
{
  // NOTE: The aligned case may not need to make use of the
  // full-up mortar weights.

  // instantiate integration object
  IntegPts integ;

  // call Gauss quadrature integration rule on quads
  GaussPolyIntQuad( elem, integ, 2 );

  // allocate mortar weights array on SurfaceContactElem object. This routine
  // also initializes the array
  elem.allocateMortarWts();

  real phiNonmortarA, phiNonmortarB, phiMortarA;

  // loop over nodes "a", where node "a" can be a nonmortar node or a mortar node
  for ( int a = 0; a < elem.numFaceVert; ++a ) {
    // loop over nonmortar nodes
    for ( int b = 0; b < elem.numFaceVert; ++b ) {
      // loop over number of integration points
      for ( int ip = 0; ip < integ.numIPs; ++ip ) {
        real xi[2];
        xi[0] = integ.xy[integ.ipDim * ip];
        xi[1] = integ.xy[integ.ipDim * ip + 1];
        LinIsoQuadShapeFunc( xi[0], xi[1], a, phiMortarA );
        LinIsoQuadShapeFunc( xi[0], xi[1], a, phiNonmortarA );
        LinIsoQuadShapeFunc( xi[0], xi[1], b, phiNonmortarB );

        // set nonmortar/nonmortar and nonmortar/mortar ids
        int nonmortarNonmortarId = elem.numFaceVert * a + b;
        int mortarNonmortarId = elem.numFaceVert * elem.numFaceVert + elem.numFaceVert * a + b;

        SLIC_ERROR_IF( nonmortarNonmortarId > elem.numWts || mortarNonmortarId > elem.numWts,
                       "ComputeAlignedMortarWeights: integer ids for weights exceed elem.numWts" );

        // compute nonmortar/nonmortar mortar weight
        elem.mortarWts[nonmortarNonmortarId] += integ.wts[ip] * phiNonmortarA * phiNonmortarB;

        // compute nonmortar/mortar mortar weight
        elem.mortarWts[mortarNonmortarId] += integ.wts[ip] * phiMortarA * phiNonmortarB;

      }  // end loop over integration points
    }    // end loop over nodes on side 2
  }      // end loop over nodes on side 1

}  // end ComputeAlignedMortarWeights()

//------------------------------------------------------------------------------
template <>
void ComputeNodalGap<ALIGNED_MORTAR>( SurfaceContactElem& elem )
{
  // get mesh instance to store gaps on mesh data object
  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& nonmortarMesh = meshManager.GetMeshInstance( elem.meshId2 );
  IndexType const* const nonmortarConn = nonmortarMesh.m_connectivity;

  SLIC_ERROR_IF( nonmortarMesh.m_nodalFields.m_node_gap == nullptr,
                 "ComputeNodalGap< ALIGNED_MORTAR >: allocate gaps on mesh data object." );

  // compute gap contributions associated with face 2 on the SurfaceContactElem
  // (i.e. nonmortar surface)

  // set the distance magnitude tolerance as the longest edge of
  // the mortar face
  real magTol;
  real magTest = 0.;
  for ( int k = 0; k < elem.numFaceVert; ++k ) {
    int idPlus = ( k == ( elem.numFaceVert - 1 ) ) ? 0 : k + 1;
    real dx = elem.faceCoords1[elem.dim * idPlus] - elem.faceCoords1[elem.dim * k];
    real dy = elem.faceCoords1[elem.dim * idPlus + 1] - elem.faceCoords1[elem.dim * k + 1];
    real dz = elem.faceCoords1[elem.dim * idPlus + 2] - elem.faceCoords1[elem.dim * k + 2];

    real mag = magnitude( dx, dy, dz );

    magTol = ( mag > magTest ) ? mag : magTest;
    magTest = mag;
  }

  // loop over nodes on nonmortar side
  for ( int a = 0; a < elem.numFaceVert; ++a ) {
    // get global nonmortar node number from connectivity
    real nrml_a[elem.dim];
    int glbId = nonmortarConn[elem.numFaceVert * elem.faceId2 + a];
    nrml_a[0] = nonmortarMesh.m_node_nX[glbId];
    nrml_a[1] = nonmortarMesh.m_node_nY[glbId];
    if ( elem.dim == 3 ) {
      nrml_a[2] = nonmortarMesh.m_node_nZ[glbId];
    }

    //////////////////////////////////////////////
    // determine which mortar node is aligned with
    // nonmortar node "a"
    //////////////////////////////////////////////
    int mortarNodeId;
    real v[3] = { 0., 0., 0. };
    real magTest = magTol;
    // loop over nodes on the mortar side
    for ( int b = 0; b < elem.numFaceVert; ++b ) {
      v[0] = elem.faceCoords1[elem.dim * b] - elem.faceCoords2[elem.dim * a];
      v[1] = elem.faceCoords1[elem.dim * b + 1] - elem.faceCoords2[elem.dim * a + 1];
      v[2] = elem.faceCoords1[elem.dim * b + 2] - elem.faceCoords2[elem.dim * a + 2];

      real magV = magnitude( v[0], v[1], v[2] );

      if ( magV < magTest ) {
        mortarNodeId = b;
        magTest = magV;
      }
    }

    // store local gap
    v[0] = elem.faceCoords1[elem.dim * mortarNodeId] - elem.faceCoords2[elem.dim * a];
    v[1] = elem.faceCoords1[elem.dim * mortarNodeId + 1] - elem.faceCoords2[elem.dim * a + 1];
    v[2] = elem.faceCoords1[elem.dim * mortarNodeId + 2] - elem.faceCoords2[elem.dim * a + 2];

    nonmortarMesh.m_nodalFields.m_node_gap[glbId] += dotProd( &v[0], &nrml_a[0], elem.dim );
  }

}  // end of ComputeNodalGap<>()

//------------------------------------------------------------------------------
void ComputeAlignedMortarGaps( CouplingScheme const* cs )
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
  // This routine is guarded against a null mesh
  nonmortarMesh.computeNodalNormals( dim );

  // declare local variables to hold face nodal coordinates
  // and overlap vertex coordinates
  real mortarX[dim * numNodesPerFace];
  real nonmortarX[dim * numNodesPerFace];
  real* overlapX;  // [dim * cpManager.m_numPolyVert[cpID]];

  ////////////////////////////
  // compute nonmortar gaps //
  ////////////////////////////
  int cpID = 0;
  for ( IndexType kp = 0; kp < numPairs; ++kp ) {
    InterfacePair pair = pairs->getInterfacePair( kp );

    if ( !pair.isContactCandidate ) {
      continue;
    }

    // get pair indices
    IndexType index1 = pair.pairIndex1;
    IndexType index2 = pair.pairIndex2;

    // populate nodal coordinates array. For the aligned mortar
    // gap equations, the nodal coordinates are NOT to be projected
    // onto the common plane, since the aligned mortar gap
    // calculation uses the current configuration nodal coordinates
    // themselves
    for ( int i = 0; i < numNodesPerFace; ++i ) {
      int id = dim * i;
      mortarX[id] = x1[mortarConn[numNodesPerFace * index1 + i]];
      mortarX[id + 1] = y1[mortarConn[numNodesPerFace * index1 + i]];
      mortarX[id + 2] = z1[mortarConn[numNodesPerFace * index1 + i]];
      nonmortarX[id] = x2[nonmortarConn[numNodesPerFace * index2 + i]];
      nonmortarX[id + 1] = y2[nonmortarConn[numNodesPerFace * index2 + i]];
      nonmortarX[id + 2] = z2[nonmortarConn[numNodesPerFace * index2 + i]];
    }

    overlapX = new real[dim * cpManager.m_numPolyVert[cpID]];
    initRealArray( overlapX, dim * cpManager.m_numPolyVert[cpID], 0. );

    // construct array of polygon overlap vertex coordinates
    cpManager.getContactPlaneOverlapVerts( cpID, cpManager.m_numPolyVert[cpID], &overlapX[0] );

    // instantiate SurfaceContactElem struct. Note, this is done with
    // the projected area of overlap, but with the actual current
    // configuration face coordinates. We need the current
    // configuration face coordinates here in order to correctly
    // compute the mortar gaps.
    SurfaceContactElem elem_for_gap( dim, &mortarX[0], &nonmortarX[0], &overlapX[0], numNodesPerFace,
                                     cpManager.m_numPolyVert[cpID], mortarId, nonmortarId, index1, index2 );

    /////////////////////////
    // compute mortar gaps //
    /////////////////////////
    ComputeNodalGap<ALIGNED_MORTAR>( elem_for_gap );

    // HAVE TO set the number of active constraints. For now set to
    // all nonmortar face nodes.
    elem_for_gap.numActiveGaps = numNodesPerFace;

    ++cpID;

    delete[] overlapX;

  }  // end loop over pairs for nonmortar gap calculations

  nonmortarMesh.m_nodalFields.m_isGapComputed = true;

}  // end ComputeAlignedMortarGaps()

//------------------------------------------------------------------------------
template <>
int ApplyNormal<ALIGNED_MORTAR, LAGRANGE_MULTIPLIER>( CouplingScheme const* cs )
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

  ///////////////////////////////////////////////////////
  //                                                   //
  //            compute single mortar gaps             //
  //                                                   //
  // Note, this routine is guarded against a null mesh //
  ///////////////////////////////////////////////////////
  ComputeAlignedMortarGaps( cs );

  int numTotalNodes;
  int numRows;
  if ( !cs->nullMeshes() ) {
    numTotalNodes = cs->getNumTotalNodes();
    numRows = dim * numTotalNodes + numTotalNodes;
    static_cast<MortarData*>( cs->getMethodData() )->allocateMfemSparseMatrix( numRows );
  }

  ////////////////////////////////////////////////////////////////
  // compute equilibrium residual and/or Jacobian contributions //
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

    //////////////////////////////////
    // compute equilibrium residual //
    //////////////////////////////////

    // loop over face nodes (BOTH MORTAR and NONMORTAR
    // contributions). NOTE: mortar weights aren't required for
    // the residual contributions for ALIGNED_MORTAR. As a result,
    // a SurfaceContactElem is not required
    for ( int a = 0; a < numNodesPerFace; ++a ) {
      int mortarIdA = mortarConn[index1 * numNodesPerFace + a];
      int nonmortarIdA = nonmortarConn[index2 * numNodesPerFace + a];

      // Note, we assemble residual for all nonmortar nodes, even if the gap
      // is in separation. NOTE: Per testing, we
      // assemble ALL nonmortar node contributions for faces with positive
      // areas of overlap that have passed the geometric filtering. We judge
      // contact activity by gaps AND the pressure solution.

      // note: we don't have to interpolate the nodal pressure for mortar and nonmortar
      // sides for the aligned mortar case (i.e. no interpolation necessary for coincident
      // mortar and nonmortar nodes).
      real forceX = nonmortarMesh.m_nodalFields.m_node_pressure[nonmortarIdA] * nonmortarMesh.m_node_nX[nonmortarIdA];
      real forceY = nonmortarMesh.m_nodalFields.m_node_pressure[nonmortarIdA] * nonmortarMesh.m_node_nY[nonmortarIdA];
      real forceZ = nonmortarMesh.m_nodalFields.m_node_pressure[nonmortarIdA] * nonmortarMesh.m_node_nZ[nonmortarIdA];

      fx2[nonmortarIdA] -= forceX;
      fy2[nonmortarIdA] -= forceY;
      fz2[nonmortarIdA] -= forceZ;

      fx1[mortarIdA] += forceX;
      fy1[mortarIdA] += forceY;
      fz1[mortarIdA] += forceZ;

    }  // end outer loop over nonmortar and mortar nodes

    /////////////////////////////////////////////////////////////
    // compute tangent stiffness contributions; Note, the      //
    // Jacobian of the weak form contact nodal force term      //
    // with respect to the primal kinematic field is 0. There  //
    // are only contributions from the Jacobian of the contact //
    // nodal force term wrt nodal pressures and the Jacobian   //
    // of the gap constraints wrt the primal kinematic field   //
    /////////////////////////////////////////////////////////////

    const EnforcementOptions& enforcement_options = const_cast<EnforcementOptions&>( cs->getEnforcementOptions() );
    const LagrangeMultiplierImplicitOptions& lm_options = enforcement_options.lm_implicit_options;
    if ( lm_options.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN ||
         lm_options.eval_mode == ImplicitEvalMode::MORTAR_JACOBIAN ) {
      // declare local variables to hold face nodal coordinates
      // and overlap vertex coordinates
      real mortarX[dim * numNodesPerFace];
      real nonmortarX[dim * numNodesPerFace];
      real* overlapX;  // [dim * cpManager.m_numPolyVert[cpID]];

      // populate nodal coordinates array. For the aligned mortar
      // gap equations, the nodal coordinates are NOT to be projected
      // onto the common plane, since the aligned mortar gap
      // calculation uses the current configuration nodal coordinates
      // themselves
      for ( int i = 0; i < numNodesPerFace; ++i ) {
        int id = dim * i;
        mortarX[id] = x1[mortarConn[numNodesPerFace * index1 + i]];
        mortarX[id + 1] = y1[mortarConn[numNodesPerFace * index1 + i]];
        mortarX[id + 2] = z1[mortarConn[numNodesPerFace * index1 + i]];
        nonmortarX[id] = x2[nonmortarConn[numNodesPerFace * index2 + i]];
        nonmortarX[id + 1] = y2[nonmortarConn[numNodesPerFace * index2 + i]];
        nonmortarX[id + 2] = z2[nonmortarConn[numNodesPerFace * index2 + i]];
      }

      overlapX = new real[dim * cpManager.m_numPolyVert[cpID]];
      initRealArray( overlapX, dim * cpManager.m_numPolyVert[cpID], 0. );

      // construct array of polygon overlap vertex coordinates
      cpManager.getContactPlaneOverlapVerts( cpID, cpManager.m_numPolyVert[cpID], &overlapX[0] );

      // get projected face coordinates for computing mortar weights and
      // Jacobian contributions
      cpManager.getProjectedFaceCoords( cpID, 0, &mortarX[0] );     // face 0 = first face
      cpManager.getProjectedFaceCoords( cpID, 1, &nonmortarX[0] );  // face 1 = second face

      // instantiate a new surface contact element with projected face
      // coordinates
      SurfaceContactElem elem_for_jac( dim, &mortarX[0], &nonmortarX[0], &overlapX[0], numNodesPerFace,
                                       cpManager.m_numPolyVert[cpID], mortarId, nonmortarId, index1, index2 );

      // HAVE TO set the number of active constraints. For now set to
      // all nonmortar face nodes.
      elem_for_jac.numActiveGaps = numNodesPerFace;

      // we need mortar weights for Jacobian contribution calculations
      ComputeAlignedMortarWeights( elem_for_jac );
      ComputeAlignedMortarJacobian( elem_for_jac );
      static_cast<MortarData*>( cs->getMethodData() )->assembleJacobian( elem_for_jac, lm_options.sparse_mode );

      delete[] overlapX;

      ++cpID;

    }  // end if-jacobian related output

  }  // end of loop over interface pairs computing residual/Jacobian contributions

  return 0;

}  // end ApplyNormal<>()

//------------------------------------------------------------------------------
template <>
void ComputeResidualJacobian<ALIGNED_MORTAR, PRIMAL>( SurfaceContactElem& TRIBOL_UNUSED_PARAM( elem ) )
{
  // There is no Jacobian contribution for this block. Be safe and zero out...
  return;
}

//------------------------------------------------------------------------------
template <>
void ComputeResidualJacobian<ALIGNED_MORTAR, DUAL>( SurfaceContactElem& elem )
{
  ComputeResidualJacobian<SINGLE_MORTAR, DUAL>( elem );
}  // end ComputeResidualJacobian<>()

//------------------------------------------------------------------------------
template <>
void ComputeConstraintJacobian<ALIGNED_MORTAR, PRIMAL>( SurfaceContactElem& elem )
{
  ComputeConstraintJacobian<SINGLE_MORTAR, PRIMAL>( elem );
}  // end ComputeConstraintJacobian

//------------------------------------------------------------------------------
template <>
void ComputeConstraintJacobian<ALIGNED_MORTAR, DUAL>( SurfaceContactElem& TRIBOL_UNUSED_PARAM( elem ) )
{
  // unless we end up solving the complementarity equation, there is
  // no Jacobian contribtion for this block. Zero out to be safe...
  return;
}

//------------------------------------------------------------------------------
void ComputeAlignedMortarJacobian( SurfaceContactElem& elem )
{
  elem.allocateBlockJ( LAGRANGE_MULTIPLIER );

  ComputeResidualJacobian<ALIGNED_MORTAR, PRIMAL>( elem );

  ComputeResidualJacobian<ALIGNED_MORTAR, DUAL>( elem );

  ComputeConstraintJacobian<ALIGNED_MORTAR, PRIMAL>( elem );

  ComputeConstraintJacobian<ALIGNED_MORTAR, DUAL>( elem );

  return;
}

//------------------------------------------------------------------------------

}  // namespace tribol
