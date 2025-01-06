// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "Mortar.hpp"

#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/geom/ContactPlane.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/integ/Integration.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/utils/ContactPlaneOutput.hpp"
#include "tribol/utils/Math.hpp"
#include "tribol/utils/Algorithm.hpp"

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
  RealT x1[elem.numFaceVert];
  RealT y1[elem.numFaceVert];
  RealT z1[elem.numFaceVert];
  RealT x2[elem.numFaceVert];
  RealT y2[elem.numFaceVert];
  RealT z2[elem.numFaceVert];

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

  RealT phiNonmortarA, phiNonmortarB, phiMortarA;

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
        RealT xp[3] = { integ.xy[elem.dim * ip], integ.xy[elem.dim * ip + 1], integ.xy[elem.dim * ip + 2] };
        RealT xi[2] = { 0., 0. };

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
  auto& nonmortarMesh = *elem.m_mesh2;
  IndexT const* const nonmortarConn = nonmortarMesh.getConnectivity().data();

  // will populate local gaps on nonmortar face on nonmortar mesh data object
  SLIC_ERROR_IF( nonmortarMesh.getNodalFields().m_node_gap.empty(),
                 "ComputeNodalGap< SINGLE_MORTAR >: allocate gaps on mesh data object." );

  SLIC_ERROR_IF( !nonmortarMesh.hasNodalNormals(),
                 "ComputeNodalGap< SINGLE_MORTAR >: allocate and compute nodal normals on mesh data object." );

  // compute gap contributions associated with face 2 on the SurfaceContactElem
  // (i.e. nonmortar surface)

  // loop over nodes on nonmortar side
  for ( int a = 0; a < elem.numFaceVert; ++a ) {
    // initialize gap1 and gap2 terms
    RealT g1 = 0.;
    RealT g2 = 0.;

    // get global nonmortar node number from connectivity
    RealT nrml_a[elem.dim];
    int glbId = nonmortarConn[elem.numFaceVert * elem.faceId2 + a];
    nrml_a[0] = nonmortarMesh.getNodalNormals()[0][glbId];
    nrml_a[1] = nonmortarMesh.getNodalNormals()[1][glbId];
    if ( elem.dim == 3 ) {
      nrml_a[2] = nonmortarMesh.getNodalNormals()[2][glbId];
    }

    // sum contributions from both sides
    for ( int b = 0; b < elem.numFaceVert; ++b ) {
      // compute nonmortar-mortar and nonmortar-nonmortar ids. Note, n_ab is
      // the stored mortar weight. For mortar-nonmortar mortar weights,
      // a = mortar node and b = nonmortar node, BUT FOR THE GAP COMPUTATION,
      // THE SUM OF MORTAR WEIGHTS IS ACTUALLY OVER SHAPE FUNCTIONS
      // DEFINED AT NODE "b", SO WE NEED TO USE (n_ab)^T.
      RealT nab_1 = elem.getNonmortarMortarWt( a, b );     // nonmortar-mortar weight
      RealT nab_2 = elem.getNonmortarNonmortarWt( a, b );  // nonmortar-nonmortar weight

      g1 += dotProd( &nrml_a[0], &elem.faceCoords1[elem.dim * b], elem.dim ) * nab_1;
      g2 += dotProd( &nrml_a[0], &elem.faceCoords2[elem.dim * b], elem.dim ) * nab_2;
    }

    // store local gap
    nonmortarMesh.getNodalFields().m_node_gap[glbId] += ( g1 - g2 );

  }  // end a-loop over nonmortar nodes

}  // end ComputeNodalGap<>()

//------------------------------------------------------------------------------
void ComputeSingleMortarGaps( CouplingScheme* cs )
{
  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& nonmortarMeshData = meshManager.at( cs->getMeshId2() );
  // compute nodal normals (do this outside the element loop)
  // Note, this is guarded against zero element meshes
  int const dim = cs->spatialDimension();
  nonmortarMeshData.computeNodalNormals( dim );

  auto pairs = cs->getInterfacePairs();
  const IndexT numPairs = pairs.size();
  auto planes = cs->get3DContactPlanes();

  ////////////////////////////////////////////////////////////////////////
  //
  // Grab mesh views
  //
  ////////////////////////////////////////////////////////////////////////
  auto mortarMesh = cs->getMesh1().getView();
  auto nonmortarMesh = cs->getMesh2().getView();

  IndexT const numNodesPerFace = mortarMesh.numberOfNodesPerElement();

  RealT const* const x1 = mortarMesh.getPosition()[0].data();
  RealT const* const y1 = mortarMesh.getPosition()[1].data();
  RealT const* const z1 = mortarMesh.getPosition()[2].data();
  IndexT const* const mortarConn = mortarMesh.getConnectivity().data();

  RealT const* const x2 = nonmortarMesh.getPosition()[0].data();
  RealT const* const y2 = nonmortarMesh.getPosition()[1].data();
  RealT const* const z2 = nonmortarMesh.getPosition()[2].data();
  IndexT const* nonmortarConn = nonmortarMesh.getConnectivity().data();

  // declare local variables to hold face nodal coordinates
  // and overlap vertex coordinates
  IndexT size = dim * numNodesPerFace;
  RealT mortarX[size];
  RealT nonmortarX[size];

  ////////////////////////////////////////////////////////////////////
  // compute nonmortar gaps to determine active set of contact dofs //
  ////////////////////////////////////////////////////////////////////
  int cpID = 0;
  for ( IndexT kp = 0; kp < numPairs; ++kp ) {
    auto& pair = pairs[kp];

    if ( !pair.m_is_contact_candidate ) {
      continue;
    }

    auto& plane = planes[cpID];

    // get pair indices
    IndexT index1 = pair.m_element_id1;
    IndexT index2 = pair.m_element_id2;

    // populate the current configuration nodal coordinates for the
    // two faces
    for ( int i = 0; i < numNodesPerFace; ++i ) {
      int id = dim * i;
      IndexT mortar_id = mortarConn[numNodesPerFace * index1 + i];
      IndexT nonmortar_id = nonmortarConn[numNodesPerFace * index2 + i];

      mortarX[id] = x1[mortar_id];
      mortarX[id + 1] = y1[mortar_id];
      mortarX[id + 2] = z1[mortar_id];
      nonmortarX[id] = x2[nonmortar_id];
      nonmortarX[id + 1] = y2[nonmortar_id];
      nonmortarX[id + 2] = z2[nonmortar_id];
    }

    // get projected face coordinates
    // stores projected coordinates in row-major format
    ArrayT<RealT, 2> mortarX_bar( numNodesPerFace, dim );
    ArrayT<RealT, 2> nonmortarX_bar( numNodesPerFace, dim );
    // stores projected coordinates in column-major format
    ArrayT<RealT, 2> mortarX_barT( dim, numNodesPerFace );
    ArrayT<RealT, 2> nonmortarX_barT( dim, numNodesPerFace );
    ProjectFaceNodesToPlane( mortarMesh, index1, plane.m_nX, plane.m_nY, plane.m_nZ, plane.m_cX, plane.m_cY, plane.m_cZ,
                             &mortarX_barT( 0, 0 ), &mortarX_barT( 1, 0 ), &mortarX_barT( 2, 0 ) );
    ProjectFaceNodesToPlane( nonmortarMesh, index2, plane.m_nX, plane.m_nY, plane.m_nZ, plane.m_cX, plane.m_cY,
                             plane.m_cZ, &nonmortarX_barT( 0, 0 ), &nonmortarX_barT( 1, 0 ), &nonmortarX_barT( 2, 0 ) );
    // populate row-major projected coordinates for the purpose of sending to
    // the SurfaceContactElem struct
    algorithm::transpose<MemorySpace::Dynamic>( mortarX_barT, mortarX_bar );
    algorithm::transpose<MemorySpace::Dynamic>( nonmortarX_barT, nonmortarX_bar );

    // construct array of polygon overlap vertex coordinates
    ArrayT<RealT, 2> overlapX( plane.m_numPolyVert, dim );
    for ( IndexT i{ 0 }; i < plane.m_numPolyVert; ++i ) {
      overlapX( i, 0 ) = plane.m_polyX[i];
      overlapX( i, 1 ) = plane.m_polyY[i];
      overlapX( i, 2 ) = plane.m_polyZ[i];
    }

    // instantiate contact surface element for purposes of computing
    // mortar weights. Note, this uses projected face coords
    SurfaceContactElem elem( dim, mortarX_bar.data(), nonmortarX_bar.data(), overlapX.data(), numNodesPerFace,
                             plane.m_numPolyVert, &mortarMesh, &nonmortarMesh, index1, index2 );

    // compute the mortar weights to be stored on the surface
    // contact element struct. This must be done prior to computing nodal gaps
    elem.overlapArea = plane.m_area;
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

  }  // end loop over pairs to compute nodal gaps

}  // end ComputeSingleMortarGaps()

//------------------------------------------------------------------------------
template <>
int ApplyNormal<SINGLE_MORTAR, LAGRANGE_MULTIPLIER>( CouplingScheme* cs )
{
  ///////////////////////////////////////////////////////
  //                                                   //
  //            compute single mortar gaps             //
  //                                                   //
  // Note, this routine is guarded against null meshes //
  ///////////////////////////////////////////////////////
  ComputeSingleMortarGaps( cs );

  auto pairs = cs->getInterfacePairs();
  const IndexT numPairs = pairs.size();
  auto planes = cs->get3DContactPlanes();

  int const dim = cs->spatialDimension();

  ////////////////////////////////////////////////////////////////////////
  //
  // Grab mesh views
  //
  ////////////////////////////////////////////////////////////////////////
  auto mortarMesh = cs->getMesh1().getView();
  auto nonmortarMesh = cs->getMesh2().getView();

  IndexT const numNodesPerFace = mortarMesh.numberOfNodesPerElement();

  RealT* const fx1 = mortarMesh.getResponse()[0].data();
  RealT* const fy1 = mortarMesh.getResponse()[1].data();
  RealT* const fz1 = mortarMesh.getResponse()[2].data();
  IndexT const* const mortarConn = mortarMesh.getConnectivity().data();

  RealT* const fx2 = nonmortarMesh.getResponse()[0].data();
  RealT* const fy2 = nonmortarMesh.getResponse()[1].data();
  RealT* const fz2 = nonmortarMesh.getResponse()[2].data();
  IndexT const* nonmortarConn = nonmortarMesh.getConnectivity().data();

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
  for ( IndexT kp = 0; kp < numPairs; ++kp ) {
    auto& pair = pairs[kp];

    if ( !pair.m_is_contact_candidate ) {
      continue;
    }

    auto& plane = planes[cpID];

    // get pair indices
    IndexT index1 = pair.m_element_id1;
    IndexT index2 = pair.m_element_id2;

    // get projected face coordinates
    // stores projected coordinates in row-major format
    ArrayT<RealT, 2> mortarX_bar( numNodesPerFace, dim );
    ArrayT<RealT, 2> nonmortarX_bar( numNodesPerFace, dim );
    // stores projected coordinates in column-major format
    ArrayT<RealT, 2> mortarX_barT( dim, numNodesPerFace );
    ArrayT<RealT, 2> nonmortarX_barT( dim, numNodesPerFace );
    ProjectFaceNodesToPlane( mortarMesh, index1, plane.m_nX, plane.m_nY, plane.m_nZ, plane.m_cX, plane.m_cY, plane.m_cZ,
                             &mortarX_barT( 0, 0 ), &mortarX_barT( 1, 0 ), &mortarX_barT( 2, 0 ) );
    ProjectFaceNodesToPlane( nonmortarMesh, index2, plane.m_nX, plane.m_nY, plane.m_nZ, plane.m_cX, plane.m_cY,
                             plane.m_cZ, &nonmortarX_barT( 0, 0 ), &nonmortarX_barT( 1, 0 ), &nonmortarX_barT( 2, 0 ) );
    // populate row-major projected coordinates for the purpose of sending to
    // the SurfaceContactElem struct
    algorithm::transpose<MemorySpace::Dynamic>( mortarX_barT, mortarX_bar );
    algorithm::transpose<MemorySpace::Dynamic>( nonmortarX_barT, nonmortarX_bar );

    // construct array of polygon overlap vertex coordinates
    // TODO: get rid of this copy
    ArrayT<RealT, 2> overlapX( plane.m_numPolyVert, dim );
    for ( IndexT i{ 0 }; i < plane.m_numPolyVert; ++i ) {
      overlapX( i, 0 ) = plane.m_polyX[i];
      overlapX( i, 1 ) = plane.m_polyY[i];
      overlapX( i, 2 ) = plane.m_polyZ[i];
    }

    // instantiate contact surface element for purposes of computing
    // mortar weights. Note, this uses projected face coords
    SurfaceContactElem elem( dim, mortarX_bar.data(), nonmortarX_bar.data(), overlapX.data(), numNodesPerFace,
                             plane.m_numPolyVert, &mortarMesh, &nonmortarMesh, index1, index2 );

    //////////////////////////////////
    // compute equilibrium residual //
    //////////////////////////////////

    // compute mortar weight
    elem.overlapArea = plane.m_area;
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

        RealT forceX = nonmortarMesh.getNodalFields().m_node_pressure[nonmortarIdB] *
                       nonmortarMesh.getNodalNormals()[0][nonmortarIdB];
        RealT forceY = nonmortarMesh.getNodalFields().m_node_pressure[nonmortarIdB] *
                       nonmortarMesh.getNodalNormals()[1][nonmortarIdB];
        RealT forceZ = nonmortarMesh.getNodalFields().m_node_pressure[nonmortarIdB] *
                       nonmortarMesh.getNodalNormals()[2][nonmortarIdB];

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
  auto& nonmortarMesh = *elem.m_mesh2;
  IndexT const* const nonmortarConn = nonmortarMesh.getConnectivity().data();

  // loop over "a" nodes accumulating sums of mortar/nonmortar
  // and nonmortar/nonmortar weights
  for ( int a = 0; a < elem.numFaceVert; ++a ) {
    // single loop over "b" nodes accumulating sums of
    // mortar(a)/nonmortar(b) and nonmortar(a)/nonmortar(b) weights
    for ( int b = 0; b < elem.numFaceVert; ++b ) {
      // get global nonmortar node id to index into nodal normals on
      // nonmortar mesh
      RealT nrml_b[elem.dim];
      int glbId = nonmortarConn[elem.numFaceVert * elem.faceId2 + b];

      // We assemble ALL nonmortar node contributions, even if gap is in separation.
      // NOTE: Per testing, we compute ALL nonmortar nodes
      // for faces that have positive areas of overlap after the geometric
      // filtering and use the gap AND the pressure solution to determine
      // contact activity

      nrml_b[0] = nonmortarMesh.getNodalNormals()[0][glbId];
      nrml_b[1] = nonmortarMesh.getNodalNormals()[1][glbId];
      if ( elem.dim == 3 ) {
        nrml_b[2] = nonmortarMesh.getNodalNormals()[2][glbId];
      }

      // get mortar-nonmortar and nonmortar-nonmortar mortar weights
      RealT n_mortar_b = elem.getMortarNonmortarWt( a, b );  // mortar-nonmortar weight
      RealT n_nonmortar_b =
          elem.getNonmortarNonmortarWt( a, b );  // nonmortar-nonmortar weight, note negative in formulation

      // fill Jrp element-pair Jacobian blocks
      // Fill block (0, 2)
      int elem_xdof = elem.getJacobianIndex( SurfaceContactElem::JrpBlock, a, b );
      int dim_offset = elem.getJacobianDimOffset( SurfaceContactElem::JrpBlock );
      elem.blockJ( static_cast<IndexT>( BlockSpace::MORTAR ),
                   static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ) )[elem_xdof] += nrml_b[0] * n_mortar_b;
      elem.blockJ( static_cast<IndexT>( BlockSpace::MORTAR ),
                   static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ) )[elem_xdof + dim_offset] +=
          nrml_b[1] * n_mortar_b;
      elem.blockJ( static_cast<IndexT>( BlockSpace::MORTAR ),
                   static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ) )[elem_xdof + 2 * dim_offset] +=
          nrml_b[2] * n_mortar_b;

      // Fill block (1, 2)
      elem.blockJ( static_cast<IndexT>( BlockSpace::NONMORTAR ),
                   static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ) )[elem_xdof] -= nrml_b[0] * n_nonmortar_b;
      elem.blockJ( static_cast<IndexT>( BlockSpace::NONMORTAR ),
                   static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ) )[elem_xdof + dim_offset] -=
          nrml_b[1] * n_nonmortar_b;
      elem.blockJ( static_cast<IndexT>( BlockSpace::NONMORTAR ),
                   static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ) )[elem_xdof + 2 * dim_offset] -=
          nrml_b[2] * n_nonmortar_b;

    }  // end loop over b nodes

  }  // end loop over a nodes

  return;
}  // end ComputeResidualJacobian<>()

//------------------------------------------------------------------------------
template <>
void ComputeConstraintJacobian<SINGLE_MORTAR, PRIMAL>( SurfaceContactElem& elem )
{
  auto& nonmortarMesh = *elem.m_mesh2;
  IndexT const* const nonmortarConn = nonmortarMesh.getConnectivity().data();

  // loop over nonmortar nodes for which we are accumulating Jacobian
  // contributions
  for ( int a = 0; a < elem.numFaceVert; ++a ) {
    // get global nonmortar node id to index into nodal normals on
    // nonmortar mesh
    RealT nrml_a[elem.dim];
    int glbId = nonmortarConn[elem.numFaceVert * elem.faceId2 + a];

    // We assemble ALL nonmortar node contributions even if gap is in separation.
    // NOTE: Per mortar method testing we compute ALL nonmortar node
    // contributions for faces that have positive areas of overlap per the
    // geometric filtering. Contact activity is judged based on gaps AND
    // the pressure solution.

    nrml_a[0] = nonmortarMesh.getNodalNormals()[0][glbId];
    nrml_a[1] = nonmortarMesh.getNodalNormals()[1][glbId];
    if ( elem.dim == 3 ) {
      nrml_a[2] = nonmortarMesh.getNodalNormals()[2][glbId];
    }

    // single loop over "b" nodes accumulating sums of
    // nonmortar(a)/mortar(b) and nonmortar(a)/nonmortar(b) weights
    for ( int b = 0; b < elem.numFaceVert; ++b ) {
      // get nonmortar-mortar and nonmortar-nonmortar mortar weights
      RealT n_mortar_a = elem.getNonmortarMortarWt( a, b );  // nonmortar-mortar weight
      RealT n_nonmortar_a =
          elem.getNonmortarNonmortarWt( a, b );  // nonmortar-nonmortar weight, note negative in formulation

      // fill Jgu element-pair Jacobian blocks
      // Fill block (2, 0)
      int dim_offset = elem.getJacobianDimOffset( SurfaceContactElem::JguBlock );
      int elem_xdof = elem.getJacobianIndex( SurfaceContactElem::JguBlock, a, b );
      elem.blockJ( static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<IndexT>( BlockSpace::MORTAR ) )[elem_xdof] += nrml_a[0] * n_mortar_a;
      elem.blockJ( static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<IndexT>( BlockSpace::MORTAR ) )[elem_xdof + dim_offset] += nrml_a[1] * n_mortar_a;
      elem.blockJ( static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<IndexT>( BlockSpace::MORTAR ) )[elem_xdof + 2 * dim_offset] += nrml_a[2] * n_mortar_a;

      // Fill block (2, 1)
      elem.blockJ( static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<IndexT>( BlockSpace::NONMORTAR ) )[elem_xdof] -= nrml_a[0] * n_nonmortar_a;
      elem.blockJ( static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<IndexT>( BlockSpace::NONMORTAR ) )[elem_xdof + dim_offset] -= nrml_a[1] * n_nonmortar_a;
      elem.blockJ( static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<IndexT>( BlockSpace::NONMORTAR ) )[elem_xdof + 2 * dim_offset] -=
          nrml_a[2] * n_nonmortar_a;

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
int GetMethodData<MORTAR_WEIGHTS>( CouplingScheme* cs )
{
  ////////////////////////////////
  //                            //
  // compute single mortar gaps //
  //                            //
  ////////////////////////////////
  ComputeSingleMortarGaps( cs );

  auto pairs = cs->getInterfacePairs();
  IndexT const numPairs = pairs.size();
  auto planes = cs->get3DContactPlanes();

  const int dim = cs->spatialDimension();

  auto mortarMesh = cs->getMesh1().getView();
  auto nonmortarMesh = cs->getMesh2().getView();
  IndexT const numNodesPerFace = mortarMesh.numberOfNodesPerElement();

  int numRows = cs->getNumTotalNodes();
  static_cast<MortarData*>( cs->getMethodData() )->allocateMfemSparseMatrix( numRows );

  //////////////////////////////////////////////
  //                                          //
  // aggregate data to compute mortar weights //
  //                                          //
  //////////////////////////////////////////////

  int cpID = 0;
  for ( IndexT kp = 0; kp < numPairs; ++kp ) {
    InterfacePair pair = pairs[kp];

    if ( !pair.m_is_contact_candidate ) {
      continue;
    }

    auto& plane = planes[cpID];

    // get pair indices
    IndexT index1 = pair.m_element_id1;
    IndexT index2 = pair.m_element_id2;

    // get projected face coordinates
    // stores projected coordinates in row-major format
    ArrayT<RealT, 2> mortarX_bar( numNodesPerFace, dim );
    ArrayT<RealT, 2> nonmortarX_bar( numNodesPerFace, dim );
    // stores projected coordinates in column-major format
    ArrayT<RealT, 2> mortarX_barT( dim, numNodesPerFace );
    ArrayT<RealT, 2> nonmortarX_barT( dim, numNodesPerFace );
    ProjectFaceNodesToPlane( mortarMesh, index1, plane.m_nX, plane.m_nY, plane.m_nZ, plane.m_cX, plane.m_cY, plane.m_cZ,
                             &mortarX_barT( 0, 0 ), &mortarX_barT( 1, 0 ), &mortarX_barT( 2, 0 ) );
    ProjectFaceNodesToPlane( nonmortarMesh, index2, plane.m_nX, plane.m_nY, plane.m_nZ, plane.m_cX, plane.m_cY,
                             plane.m_cZ, &nonmortarX_barT( 0, 0 ), &nonmortarX_barT( 1, 0 ), &nonmortarX_barT( 2, 0 ) );
    // populate row-major projected coordinates for the purpose of sending to
    // the SurfaceContactElem struct
    algorithm::transpose<MemorySpace::Dynamic>( mortarX_barT, mortarX_bar );
    algorithm::transpose<MemorySpace::Dynamic>( nonmortarX_barT, nonmortarX_bar );

    // construct array of polygon overlap vertex coordinates
    ArrayT<RealT, 2> overlapX( plane.m_numPolyVert, dim );
    for ( IndexT i{ 0 }; i < plane.m_numPolyVert; ++i ) {
      overlapX( i, 0 ) = plane.m_polyX[i];
      overlapX( i, 1 ) = plane.m_polyY[i];
      overlapX( i, 2 ) = plane.m_polyZ[i];
    }

    // instantiate contact surface element for purposes of computing
    // mortar weights. Note, this uses projected face coords
    SurfaceContactElem elem( dim, mortarX_bar.data(), nonmortarX_bar.data(), overlapX.data(), numNodesPerFace,
                             plane.m_numPolyVert, &mortarMesh, &nonmortarMesh, index1, index2 );

    // compute the mortar weights to be stored on the surface
    // contact element struct. This must be done prior to computing nodal gaps
    elem.overlapArea = plane.m_area;

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

  }  // end loop over pairs to assemble mortar weights

  return 0;

}  // end GetMethodData< MORTAR_WEIGHTS >()

//------------------------------------------------------------------------------

}  // end namespace tribol
