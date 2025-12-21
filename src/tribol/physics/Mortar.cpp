// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "Mortar.hpp"

#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/geom/CompGeom.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/geom/NodalNormal.hpp"
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/integ/Integration.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/utils/Math.hpp"

// Axom includes
#include "axom/slic.hpp"

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
  Array1D<RealT> x1( elem.numFaceVert );
  Array1D<RealT> y1( elem.numFaceVert );
  Array1D<RealT> z1( elem.numFaceVert );
  Array1D<RealT> x2( elem.numFaceVert );
  Array1D<RealT> y2( elem.numFaceVert );
  Array1D<RealT> z2( elem.numFaceVert );

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

        InvIso( xp, x1.data(), y1.data(), z1.data(), elem.numFaceVert, xi );
        if ( elem.numFaceVert == 4 ) {
          LinIsoQuadShapeFunc( xi[0], xi[1], a, phiMortarA );
        } else if ( elem.numFaceVert == 3 ) {
          LinIsoTriShapeFunc( xi[0], xi[1], a, phiMortarA );
        }

        InvIso( xp, x2.data(), y2.data(), z2.data(), elem.numFaceVert, xi );
        if ( elem.numFaceVert == 4 ) {
          LinIsoQuadShapeFunc( xi[0], xi[1], a, phiNonmortarA );
          LinIsoQuadShapeFunc( xi[0], xi[1], b, phiNonmortarB );
        } else if ( elem.numFaceVert == 3 ) {
          LinIsoTriShapeFunc( xi[0], xi[1], a, phiNonmortarA );
          LinIsoTriShapeFunc( xi[0], xi[1], b, phiNonmortarB );
        }

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
    Array1D<RealT> nrml_a( elem.dim );
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

      g1 += dotProd( nrml_a.data(), &elem.faceCoords1[elem.dim * b], elem.dim ) * nab_1;
      g2 += dotProd( nrml_a.data(), &elem.faceCoords2[elem.dim * b], elem.dim ) * nab_2;
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
  ElementAvgNodalNormal normal_method;
  normal_method.Compute( nonmortarMeshData );

  auto pairs = cs->getInterfacePairs();
  const IndexT numPairs = pairs.size();

  ////////////////////////////////////////////////////////////////////////
  //
  // Grab mesh views
  //
  ////////////////////////////////////////////////////////////////////////
  auto mortarMesh = cs->getMesh1().getView();
  auto nonmortarMesh = cs->getMesh2().getView();
  IndexT const numNodesPerFace = mortarMesh.numberOfNodesPerElement();

  // declare local variables to hold face nodal coordinates
  // and overlap vertex coordinates
  int const dim = cs->spatialDimension();
  Array2D<RealT> mortarX( numNodesPerFace, dim );
  Array2D<RealT> nonmortarX( numNodesPerFace, dim );
  Array2D<RealT> mortarX_bar( numNodesPerFace, dim );
  Array2D<RealT> nonmortarX_bar( numNodesPerFace, dim );

  ////////////////////////////////////////////////////////////////////
  // compute nonmortar gaps to determine active set of contact dofs //
  ////////////////////////////////////////////////////////////////////
  int cpID = 0;
  for ( IndexT kp = 0; kp < numPairs; ++kp ) {
    auto& pair = pairs[kp];

    if ( !pair.m_is_contact_candidate ) {
      continue;
    }

    auto& cg_pairs = cs->getCompGeom();
    auto& plane = cg_pairs.getMortarPlane( cpID );

    Array2D<RealT> overlapX( plane.m_numPolyVert, dim );

    // get pair indices
    IndexT index1 = pair.m_element_id1;
    IndexT index2 = pair.m_element_id2;

    // populate the current configuration nodal coordinates for the
    // two faces; stored on the contact plane object
    plane.getFace1Coords( mortarX.data(), numNodesPerFace );
    plane.getFace2Coords( nonmortarX.data(), numNodesPerFace );

    // get face coordinates projected onto contact plane
    plane.getFace1ProjectedCoords( mortarX_bar.data(), numNodesPerFace );
    plane.getFace2ProjectedCoords( nonmortarX_bar.data(), numNodesPerFace );

    // get overlap vertices
    plane.getOverlapVertices( overlapX.data() );

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
    elem.faceCoords1 = mortarX.data();
    elem.faceCoords2 = nonmortarX.data();

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
#ifdef TRIBOL_USE_ENZYME
  if ( cs->isEnzymeEnabled() ) {
    return ApplyNormalEnzyme( cs );
  }
#endif
  ///////////////////////////////////////////////////////
  //                                                   //
  //            compute single mortar gaps             //
  //                                                   //
  // Note, this routine is guarded against null meshes //
  ///////////////////////////////////////////////////////
  ComputeSingleMortarGaps( cs );

  auto pairs = cs->getInterfacePairs();
  const IndexT numPairs = pairs.size();

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

  // declare local variables to hold projected face nodal coordinates
  Array2D<RealT> mortarX_bar( numNodesPerFace, dim );
  Array2D<RealT> nonmortarX_bar( numNodesPerFace, dim );

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

    auto& cg_pairs = cs->getCompGeom();
    auto& plane = cg_pairs.getMortarPlane( cpID );

    Array2D<RealT> overlapX( plane.m_numPolyVert, dim );

    // get pair indices
    IndexT index1 = pair.m_element_id1;
    IndexT index2 = pair.m_element_id2;

    // get face coordinates projected onto contact plane
    plane.getFace1ProjectedCoords( mortarX_bar.data(), numNodesPerFace );
    plane.getFace2ProjectedCoords( nonmortarX_bar.data(), numNodesPerFace );

    // get overlap coords
    plane.getOverlapVertices( overlapX.data() );

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
      Array1D<RealT> nrml_b( elem.dim );
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
      elem.blockJ( static_cast<IndexT>( BlockSpace::MORTAR ), static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ) )
          .data()[elem_xdof] += nrml_b[0] * n_mortar_b;
      elem.blockJ( static_cast<IndexT>( BlockSpace::MORTAR ), static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ) )
          .data()[elem_xdof + dim_offset] += nrml_b[1] * n_mortar_b;
      elem.blockJ( static_cast<IndexT>( BlockSpace::MORTAR ), static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ) )
          .data()[elem_xdof + 2 * dim_offset] += nrml_b[2] * n_mortar_b;

      // Fill block (1, 2)
      elem.blockJ( static_cast<IndexT>( BlockSpace::NONMORTAR ),
                   static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ) )
          .data()[elem_xdof] -= nrml_b[0] * n_nonmortar_b;
      elem.blockJ( static_cast<IndexT>( BlockSpace::NONMORTAR ),
                   static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ) )
          .data()[elem_xdof + dim_offset] -= nrml_b[1] * n_nonmortar_b;
      elem.blockJ( static_cast<IndexT>( BlockSpace::NONMORTAR ),
                   static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ) )
          .data()[elem_xdof + 2 * dim_offset] -= nrml_b[2] * n_nonmortar_b;

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
    Array1D<RealT> nrml_a( elem.dim );
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
      elem.blockJ( static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ), static_cast<IndexT>( BlockSpace::MORTAR ) )
          .data()[elem_xdof] += nrml_a[0] * n_mortar_a;
      elem.blockJ( static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ), static_cast<IndexT>( BlockSpace::MORTAR ) )
          .data()[elem_xdof + dim_offset] += nrml_a[1] * n_mortar_a;
      elem.blockJ( static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ), static_cast<IndexT>( BlockSpace::MORTAR ) )
          .data()[elem_xdof + 2 * dim_offset] += nrml_a[2] * n_mortar_a;

      // Fill block (2, 1)
      elem.blockJ( static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<IndexT>( BlockSpace::NONMORTAR ) )
          .data()[elem_xdof] -= nrml_a[0] * n_nonmortar_a;
      elem.blockJ( static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<IndexT>( BlockSpace::NONMORTAR ) )
          .data()[elem_xdof + dim_offset] -= nrml_a[1] * n_nonmortar_a;
      elem.blockJ( static_cast<IndexT>( BlockSpace::LAGRANGE_MULTIPLIER ),
                   static_cast<IndexT>( BlockSpace::NONMORTAR ) )
          .data()[elem_xdof + 2 * dim_offset] -= nrml_a[2] * n_nonmortar_a;

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

#ifdef TRIBOL_USE_ENZYME

//------------------------------------------------------------------------------
int ApplyNormalEnzyme( CouplingScheme* cs )
{
  auto& comp_geom = cs->getCompGeom();
  int num_active_pairs = cs->getNumActivePairs();
  auto& lm_opts = cs->getEnforcementOptions().lm_implicit_options;
  if ( lm_opts.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN ||
       lm_opts.eval_mode == ImplicitEvalMode::MORTAR_JACOBIAN ) {
    if ( lm_opts.sparse_mode == SparseMode::MFEM_ELEMENT_DENSE ) {
      cs->getMethodData()->reserveBlockJ(
          { BlockSpace::NONMORTAR, BlockSpace::MORTAR, BlockSpace::LAGRANGE_MULTIPLIER }, num_active_pairs );
      cs->createNodalNormalJacobianData();
      cs->getDfDnMethodData()->reserveBlockJ(
          { BlockSpace::NONMORTAR, BlockSpace::MORTAR, BlockSpace::LAGRANGE_MULTIPLIER }, num_active_pairs );
      cs->getDnDxMethodData()->reserveBlockJ( { BlockSpace::NONMORTAR }, cs->getMesh2().numberOfElements() );
    } else {
      SLIC_WARNING( "Unsupported Jacobian storage method." );
      return 1;
    }
  }
  // convention: 1 = nonmortar
  //             2 = mortar
  // This follows the defs used in Puso and Laursen (2004), but is switched from the rest of Tribol. Sticking to the
  // Puso and Laursen notation here so it's easier to track.
  EdgeAvgNodalNormal normal_method;
  normal_method.Compute( cs->getMesh2(), cs->getDnDxMethodData() );
  auto mesh1 = cs->getMesh2().getView();  // switched from tribol convention
  auto mesh2 = cs->getMesh1().getView();  // switched from tribol convention
  int size1 = mesh1.numberOfNodesPerElement();
  int size2 = mesh2.numberOfNodesPerElement();

  for ( auto& plane : comp_geom.getMortarPlanePairs() ) {
    int elem1 = plane.getCpElementId2();  // switched from tribol convention
    // NOTE: mfem::DenseMatrix data is stored by nodes instead of by vdim
    RealT x1[12];
    RealT n1[12];
    RealT f1[12];
    RealT p1[4];
    RealT g1[4];
    for ( int i{ 0 }; i < size1; ++i ) {
      int node_id = mesh1.getGlobalNodeId( elem1, i );
      for ( int d{ 0 }; d < 3; ++d ) {
        x1[d * size1 + i] = mesh1.getPosition()[d][node_id];
        n1[d * size1 + i] = mesh1.getNodalNormals()( d, node_id );
        f1[d * size1 + i] = 0.0;
      }
      p1[i] = mesh1.getNodalFields().m_node_pressure[node_id];
      g1[i] = 0.0;
    }
    int elem2 = plane.getCpElementId1();  // switched from tribol convention
    RealT x2[12];
    RealT f2[12];
    for ( int i{ 0 }; i < size2; ++i ) {
      int node_id = mesh2.getGlobalNodeId( elem2, i );
      for ( int d{ 0 }; d < 3; ++d ) {
        x2[d * size2 + i] = mesh2.getPosition()[d][node_id];
        f2[d * size2 + i] = 0.0;
      }
    }
    if ( lm_opts.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN ||
         lm_opts.eval_mode == ImplicitEvalMode::MORTAR_JACOBIAN ) {
      StackArray<DeviceArray2D<RealT>, 9> blockJ_n( 3 );
      int n_disp[2] = { size1 * 3, size2 * 3 };
      for ( int i{ 0 }; i < 2; ++i ) {
        blockJ_n( i, 0 ) = DeviceArray2D<RealT>( n_disp[0], n_disp[0] );
        blockJ_n( i, 0 ).fill( 0.0 );
      }
      int n_multipliers = size1;
      blockJ_n( 2, 0 ) = DeviceArray2D<RealT>( n_multipliers, n_disp[0] );
      blockJ_n( 2, 0 ).fill( 0.0 );

      StackArray<DeviceArray2D<RealT>, 9> blockJ( 3 );
      for ( int i{}; i < 2; ++i ) {
        for ( int j{}; j < 2; ++j ) {
          blockJ( i, j ) = DeviceArray2D<RealT>( n_disp[i], n_disp[j] );
          blockJ( i, j ).fill( 0.0 );
        }
      }
      for ( int i{}; i < 2; ++i ) {
        blockJ( i, 2 ) = DeviceArray2D<RealT>( n_disp[i], n_multipliers );
        blockJ( i, 2 ).fill( 0.0 );
        // transpose
        blockJ( 2, i ) = DeviceArray2D<RealT>( n_multipliers, n_disp[i] );
        blockJ( 2, i ).fill( 0.0 );
      }
      blockJ( 2, 2 ) = DeviceArray2D<RealT>( n_multipliers, n_multipliers );
      blockJ( 2, 2 ).fill( 0.0 );

      // This function also computes the residual contributions
      ComputeMortarJacobianEnzyme( x1, n1, p1, f1, blockJ( 0, 0 ).data(), blockJ( 0, 1 ).data(),
                                   blockJ_n( 0, 0 ).data(), blockJ( 0, 2 ).data(), g1, blockJ( 2, 0 ).data(),
                                   blockJ( 2, 1 ).data(), blockJ_n( 2, 0 ).data(), size1, x2, f2, blockJ( 1, 0 ).data(),
                                   blockJ( 1, 1 ).data(), blockJ_n( 1, 0 ).data(), blockJ( 1, 2 ).data(), size2 );

      if ( lm_opts.sparse_mode == SparseMode::MFEM_ELEMENT_DENSE ) {
        cs->getMethodData()->storeElemBlockJ( { elem1, elem2, elem1 }, blockJ );
        cs->getDfDnMethodData()->storeElemBlockJ( { elem1, elem2, elem1 }, blockJ_n );
      } else {
        SLIC_WARNING( "Unsupported Jacobian storage method." );
        return 1;
      }
    } else if ( lm_opts.eval_mode == ImplicitEvalMode::MORTAR_GAP ||
                lm_opts.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL ) {
      ComputeMortarForceEnzyme( x1, n1, p1, f1, g1, size1, x2, f2, size2 );
    }
    for ( int i{ 0 }; i < size1; ++i ) {
      int node_id = mesh1.getGlobalNodeId( elem1, i );
      for ( int d{ 0 }; d < 3; ++d ) {
        mesh1.getResponse()[d][node_id] += f1[d * size1 + i];
      }
      mesh1.getNodalFields().m_node_gap[node_id] += g1[i];
    }
    for ( int i{ 0 }; i < size2; ++i ) {
      int node_id = mesh2.getGlobalNodeId( elem2, i );
      for ( int d{ 0 }; d < 3; ++d ) {
        mesh2.getResponse()[d][node_id] += f2[d * size2 + i];
      }
    }
  }

  return 0;
}

//------------------------------------------------------------------------------
void ComputeMortarForceEnzyme( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* g1, int size1,
                               const RealT* x2, RealT* f2, int size2 )
{
  // convention: elem1 = nonmortar element
  //             elem2 = mortar element
  constexpr int max_mortar_mat_size = 4 * 4;
  RealT mortar_mat1[max_mortar_mat_size];
  int mortar_mat1_size = size1 * size1;
  for ( int i{ 0 }; i < mortar_mat1_size; ++i ) {
    mortar_mat1[i] = 0.0;
  }
  RealT mortar_mat2[max_mortar_mat_size];
  int mortar_mat2_size = size1 * size2;
  for ( int i{ 0 }; i < mortar_mat2_size; ++i ) {
    mortar_mat2[i] = 0.0;
  }
  // get point x0 (geometric center of elem1)
  RealT x0[3] = { 0.0, 0.0, 0.0 };
  for ( int i{ 0 }; i < size1; ++i ) {
    for ( int d{ 0 }; d < 3; ++d ) {
      x0[d] += x1[d * size1 + i] / static_cast<RealT>( size1 );
    }
  }

  // get vector n (normal of elem1) = de1 x de2
  // clang-format off
  RealT de1[3] = { 0.0, 0.0, 0.0 };
  RealT de2[3] = { 0.0, 0.0, 0.0 };
  if ( size1 == 4 ) {
    de1[0] = -0.25*x1[0] + 0.25*x1[1] + 0.25*x1[2] - 0.25*x1[3];
    de1[1] = -0.25*x1[4] + 0.25*x1[5] + 0.25*x1[6] - 0.25*x1[7];
    de1[2] = -0.25*x1[8] + 0.25*x1[9] + 0.25*x1[10] - 0.25*x1[11];
    de2[0] = -0.25*x1[0] - 0.25*x1[1] + 0.25*x1[2] + 0.25*x1[3];
    de2[1] = -0.25*x1[4] - 0.25*x1[5] + 0.25*x1[6] + 0.25*x1[7];
    de2[2] = -0.25*x1[8] - 0.25*x1[9] + 0.25*x1[10] + 0.25*x1[11];
  } else if ( size1 == 3 ) {
    de1[0] = x1[1] - x1[0];
    de1[1] = x1[4] - x1[3];
    de1[2] = x1[7] - x1[6];
    de2[0] = x1[2] - x1[0];
    de2[1] = x1[5] - x1[3];
    de2[2] = x1[8] - x1[6];
  }
  RealT n[3] = {
    de1[1]*de2[2] - de1[2]*de2[1],
    de1[2]*de2[0] - de1[0]*de2[2],
    de1[0]*de2[1] - de1[1]*de2[0]
  };
  // clang-format on
  RealT n_mag = std::sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] );
  for ( int d{ 0 }; d < 3; ++d ) {
    n[d] /= n_mag;
  }

  // x1t = x1 coordinates projected to plane p (def'd by x0 and n) but in 3d
  constexpr int max_coord_size = 4 * 3;
  RealT x1t[max_coord_size];
  for ( int i{ 0 }; i < size1; ++i ) {
    RealT x1diff_mag = 0.0;
    for ( int d{ 0 }; d < 3; ++d ) {
      x1diff_mag += n[d] * ( x1[size1 * d + i] - x0[d] );
    }
    for ( int d{ 0 }; d < 3; ++d ) {
      x1t[size1 * d + i] = x1[size1 * d + i] - n[d] * x1diff_mag;
    }
  }
  // x2t = x2 coordinates projected to plane p but in 3d
  RealT x2t[max_coord_size];
  for ( int i{ 0 }; i < size2; ++i ) {
    RealT x2diff_mag = 0.0;
    for ( int d{ 0 }; d < 3; ++d ) {
      x2diff_mag += n[d] * ( x2[size2 * d + i] - x0[d] );
    }
    for ( int d{ 0 }; d < 3; ++d ) {
      x2t[size2 * d + i] = x2[size2 * d + i] - n[d] * x2diff_mag;
    }
  }
  // Tribol's clipping algorithm
  // create a local basis; e1 is a unit vector aligned with the first edge in element 1
  // clang-format off
   RealT e1[3] = {
      x1t[0*size1 + 1] - x1t[0*size1 + 0],
      x1t[1*size1 + 1] - x1t[1*size1 + 0],
      x1t[2*size1 + 1] - x1t[2*size1 + 0]
   };
  // clang-format on
  RealT e1_mag = std::sqrt( e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2] );
  for ( int d{ 0 }; d < 3; ++d ) {
    e1[d] /= e1_mag;
  }
  // e2 is a unit vector = n x e1
  // clang-format off
   RealT e2[3] = {
      n[1]*e1[2] - n[2]*e1[1],
      n[2]*e1[0] - n[0]*e1[2],
      n[0]*e1[1] - n[1]*e1[0]
   };
  // clang-format on
  RealT x1t_2d[4];
  RealT y1t_2d[4];
  PlaneTo2DCoords( x1t, x0, e1, e2, x1t_2d, y1t_2d, size1 );
  RealT x2t_2d[4];
  RealT y2t_2d[4];
  PlaneTo2DCoords( x2t, x0, e1, e2, x2t_2d, y2t_2d, size2 );
  // coordinates need to be CCW for both faces. the call to ElemReverse() will reverse the projected 2d coordinates of
  // element 2, which are in clockwise direction
  RealT x2t_2d_rev[4];
  RealT y2t_2d_rev[4];
  for ( int i{ 0 }; i < size2; ++i ) {
    x2t_2d_rev[i] = x2t_2d[i];
    y2t_2d_rev[i] = y2t_2d[i];
  }
  ElemReverse( x2t_2d_rev, y2t_2d_rev, size2 );
  RealT xti_2d[8];
  RealT yti_2d[8];
  int overlap_poly_size = 0;
  Intersection2DPolygonEnzyme( x1t_2d, y1t_2d, size1, x2t_2d_rev, y2t_2d_rev, size2, 1.0e-8, 1.0e-8, xti_2d, yti_2d,
                               &overlap_poly_size );
  RealT overlap_poly_area = Area2DPolygon( xti_2d, yti_2d, overlap_poly_size );
  if ( overlap_poly_area <= 0.0 ) {
    return;
  }

  // Integrate mortar matrix over the polygon
  // 1. get base triangle integration rule
  RealT base_rule_2d[12];
  RealT base_weights[6];
  {
    RealT wt1 = 0.109951743655322;
    RealT wt2 = 0.223381589678011;
    base_weights[0] = wt1;
    base_weights[1] = wt1;
    base_weights[2] = wt1;
    base_weights[3] = wt2;
    base_weights[4] = wt2;
    base_weights[5] = wt2;
    RealT base_x1 = 0.091576213509771;
    RealT base_x2 = 0.816847572980459;
    RealT base_x3 = 0.108103018168070;
    RealT base_x4 = 0.445948490915965;
    base_rule_2d[0] = base_x1;
    base_rule_2d[1] = base_x1;
    base_rule_2d[2] = base_x2;
    base_rule_2d[3] = base_x1;
    base_rule_2d[4] = base_x1;
    base_rule_2d[5] = base_x2;
    base_rule_2d[6] = base_x3;
    base_rule_2d[7] = base_x4;
    base_rule_2d[8] = base_x4;
    base_rule_2d[9] = base_x3;
    base_rule_2d[10] = base_x4;
    base_rule_2d[11] = base_x4;
  }

  // 2. build the sub-triangles
  // vert0 = centroid of overlap polygon; this will be used as the first vertex of the sub-triangles
  RealT tri_0[2];
  PolyCentroid( xti_2d, yti_2d, overlap_poly_size, tri_0[0], tri_0[1] );
  for ( int i{ 0 }; i < overlap_poly_size; ++i ) {
    int idx1 = i;
    int idx2 = ( i + 1 ) % overlap_poly_size;
    RealT tri_1[2] = { xti_2d[idx1], yti_2d[idx1] };
    RealT tri_2[2] = { xti_2d[idx2], yti_2d[idx2] };
    RealT side1[2] = { tri_2[0] - tri_1[0], tri_2[1] - tri_1[1] };
    RealT side2[2] = { tri_0[0] - tri_1[0], tri_0[1] - tri_1[1] };
    RealT area = 0.5 * ( side1[0] * side2[1] - side1[1] * side2[0] );

    // the sub-triangle is inverted.  likely something went wrong with CG.  don't try to integrate over it.
    if ( area <= 0.0 ) {
      continue;
    }

    for ( int j{ 0 }; j < 6; ++j ) {
      RealT tri_xi[2] = { base_rule_2d[j * 2 + 0], base_rule_2d[j * 2 + 1] };
      RealT tri_phi[3] = { 0.0, 0.0, 0.0 };
      LinIsoTriShapeFunc( tri_xi, tri_phi );
      RealT tri_quad_pt[2] = { tri_phi[0] * tri_0[0] + tri_phi[1] * tri_1[0] + tri_phi[2] * tri_2[0],
                               tri_phi[0] * tri_0[1] + tri_phi[1] * tri_1[1] + tri_phi[2] * tri_2[1] };

      // 3. map sub-triangle coordinate to nonmortar and mortar coordinates
      // NOTE: we ideally want to do this in 2d, but there are finite differencing errors when we do
      RealT tri_quad_pt_3d[3] = { 0.0, 0.0, 0.0 };
      Coords2DToPlane( tri_quad_pt, tri_quad_pt + 1, x0, e1, e2, tri_quad_pt_3d, 1 );
      RealT xi1[2] = { 0.0, 0.0 };
      InvIso( tri_quad_pt_3d, x1t, x1t + size1, x1t + 2 * size1, size1, xi1 );
      RealT xi2[2] = { 0.0, 0.0 };
      InvIso( tri_quad_pt_3d, x2t, x2t + size2, x2t + 2 * size2, size2, xi2 );

      RealT quad_wt = base_weights[j] * area;

      // 4. Evaluate mortar matrix (nonmortar/nonmortar contribs)
      // NOTE: Nonstandard node numbering with InvIso and LinIsoQuadShapeFunc
      for ( int k{ 0 }; k < size1; ++k ) {
        RealT phiA = 0.0;
        if ( size1 == 4 ) {
          LinIsoQuadShapeFunc( xi1[0], xi1[1], k, phiA );
        } else if ( size1 == 3 ) {
          LinIsoTriShapeFunc( xi1[0], xi1[1], k, phiA );
        }
        for ( int l{ 0 }; l < size1; ++l ) {
          RealT phiB = 0.0;
          if ( size1 == 4 ) {
            LinIsoQuadShapeFunc( xi1[0], xi1[1], l, phiB );
          } else if ( size1 == 3 ) {
            LinIsoTriShapeFunc( xi1[0], xi1[1], l, phiB );
          }
          mortar_mat1[k * size1 + l] += phiA * phiB * quad_wt;
        }
      }

      // 5. Evaluate mortar matrix (nonmortar/mortar contribs)
      for ( int k{ 0 }; k < size1; ++k ) {
        RealT phiA = 0.0;
        if ( size1 == 4 ) {
          LinIsoQuadShapeFunc( xi1[0], xi1[1], k, phiA );
        } else if ( size1 == 3 ) {
          LinIsoTriShapeFunc( xi1[0], xi1[1], k, phiA );
        }
        for ( int l{ 0 }; l < size2; ++l ) {
          RealT phiB = 0.0;
          if ( size2 == 4 ) {
            LinIsoQuadShapeFunc( xi2[0], xi2[1], l, phiB );
          } else if ( size2 == 3 ) {
            LinIsoTriShapeFunc( xi2[0], xi2[1], l, phiB );
          }
          mortar_mat2[k * size2 + l] += phiA * phiB * quad_wt;
        }
      }
    }
  }

  // compute gaps
  for ( int i{ 0 }; i < size1; ++i ) {
    g1[i] = 0.0;
    RealT gap_v[3] = { 0.0, 0.0, 0.0 };
    for ( int j{ 0 }; j < size1; ++j ) {
      for ( int d{ 0 }; d < 3; ++d ) {
        gap_v[d] -= mortar_mat1[i * size1 + j] * x1[d * size1 + j];
      }
    }
    for ( int j{ 0 }; j < size2; ++j ) {
      for ( int d{ 0 }; d < 3; ++d ) {
        gap_v[d] += mortar_mat2[i * size2 + j] * x2[d * size2 + j];
      }
    }
    for ( int d{ 0 }; d < 3; ++d ) {
      g1[i] += n1[d * size1 + i] * gap_v[d];
    }
  }

  // compute nonmortar force contributions
  for ( int i{ 0 }; i < size1; ++i ) {
    for ( int d{ 0 }; d < 3; ++d ) {
      f1[d * size1 + i] = 0.0;
    }
    for ( int j{ 0 }; j < size1; ++j ) {
      for ( int d{ 0 }; d < 3; ++d ) {
        f1[d * size1 + i] -= p1[j] * n1[d * size1 + i] * mortar_mat1[j * size1 + i];
      }
    }
  }

  // compute mortar force contributions
  for ( int i{ 0 }; i < size2; ++i ) {
    for ( int d{ 0 }; d < 3; ++d ) {
      f2[d * size2 + i] = 0.0;
    }
    for ( int j{ 0 }; j < size1; ++j ) {
      for ( int d{ 0 }; d < 3; ++d ) {
        f2[d * size2 + i] += p1[j] * n1[d * size1 + i] * mortar_mat2[j * size2 + i];
      }
    }
  }
}

//------------------------------------------------------------------------------
void ComputeMortarJacobianEnzyme( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* df1dx1,
                                  RealT* df1dx2, RealT* df1dn1, RealT* df1dp1, RealT* g1, RealT* dg1dx1, RealT* dg1dx2,
                                  RealT* dg1dn1, int size1, const RealT* x2, RealT* f2, RealT* df2dx1, RealT* df2dx2,
                                  RealT* df2dn1, RealT* df2dp1, int size2 )
{
  RealT x1_dot[12] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  for ( int i{ 0 }; i < size1 * 3; ++i ) {
    x1_dot[i] = 1.0;
    // clang-format off
      __enzyme_fwddiff<void>((void*)ComputeMortarForceEnzyme,
         TRIBOL_ENZYME_DUP, x1, x1_dot,
         TRIBOL_ENZYME_CONST, n1,
         TRIBOL_ENZYME_CONST, p1,
         TRIBOL_ENZYME_DUP, f1, &df1dx1[size1*3*i],
         TRIBOL_ENZYME_DUP, g1, &dg1dx1[size1*i],
         TRIBOL_ENZYME_CONST, size1,
         TRIBOL_ENZYME_CONST, x2,
         TRIBOL_ENZYME_DUP, f2, &df2dx1[size1*3*i],
         TRIBOL_ENZYME_CONST, size2);
    // clang-format on
    x1_dot[i] = 0.0;
  }
  RealT n1_dot[12] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  for ( int i{ 0 }; i < size1 * 3; ++i ) {
    n1_dot[i] = 1.0;
    // clang-format off
      __enzyme_fwddiff<void>((void*)ComputeMortarForceEnzyme,
         TRIBOL_ENZYME_CONST, x1,
         TRIBOL_ENZYME_DUP, n1, n1_dot,
         TRIBOL_ENZYME_CONST, p1,
         TRIBOL_ENZYME_DUP, f1, &df1dn1[size1*3*i],
         TRIBOL_ENZYME_DUP, g1, &dg1dn1[size1*i],
         TRIBOL_ENZYME_CONST, size1,
         TRIBOL_ENZYME_CONST, x2,
         TRIBOL_ENZYME_DUP, f2, &df2dn1[size1*3*i],
         TRIBOL_ENZYME_CONST, size2);
    // clang-format on
    n1_dot[i] = 0.0;
  }
  RealT p1_dot[4] = { 0.0, 0.0, 0.0, 0.0 };
  for ( int i{ 0 }; i < size1; ++i ) {
    p1_dot[i] = 1.0;
    // clang-format off
      __enzyme_fwddiff<void>((void*)ComputeMortarForceEnzyme,
         TRIBOL_ENZYME_CONST, x1,
         TRIBOL_ENZYME_CONST, n1,
         TRIBOL_ENZYME_DUP, p1, p1_dot,
         TRIBOL_ENZYME_DUP, f1, &df1dp1[size1*3*i],
         TRIBOL_ENZYME_CONST, g1,
         TRIBOL_ENZYME_CONST, size1,
         TRIBOL_ENZYME_CONST, x2,
         TRIBOL_ENZYME_DUP, f2, &df2dp1[size1*3*i],
         TRIBOL_ENZYME_CONST, size2);
    // clang-format on
    p1_dot[i] = 0.0;
  }
  RealT x2_dot[12] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  for ( int i{ 0 }; i < size2 * 3; ++i ) {
    x2_dot[i] = 1.0;
    // clang-format off
      __enzyme_fwddiff<void>((void*)ComputeMortarForceEnzyme,
         TRIBOL_ENZYME_CONST, x1,
         TRIBOL_ENZYME_CONST, n1,
         TRIBOL_ENZYME_CONST, p1,
         TRIBOL_ENZYME_DUP, f1, &df1dx2[size2*3*i],
         TRIBOL_ENZYME_DUP, g1, &dg1dx2[size2*i],
         TRIBOL_ENZYME_CONST, size1,
         TRIBOL_ENZYME_DUP, x2, x2_dot,
         TRIBOL_ENZYME_DUP, f2, &df2dx2[size2*3*i],
         TRIBOL_ENZYME_CONST, size2);
    // clang-format on
    x2_dot[i] = 0.0;
  }
}
#endif

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

  const int dim = cs->spatialDimension();

  auto mortarMesh = cs->getMesh1().getView();
  auto nonmortarMesh = cs->getMesh2().getView();
  IndexT const numNodesPerFace = mortarMesh.numberOfNodesPerElement();

  Array2D<RealT> mortarX_bar( numNodesPerFace, dim );
  Array2D<RealT> nonmortarX_bar( numNodesPerFace, dim );

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

    auto& cg_pairs = cs->getCompGeom();
    auto& plane = cg_pairs.getMortarPlane( cpID );

    Array2D<RealT> overlapX( plane.m_numPolyVert, dim );

    // get pair indices
    IndexT index1 = pair.m_element_id1;
    IndexT index2 = pair.m_element_id2;

    // get face coordinates projected onto contact plane
    plane.getFace1ProjectedCoords( mortarX_bar.data(), numNodesPerFace );
    plane.getFace2ProjectedCoords( nonmortarX_bar.data(), numNodesPerFace );

    // construct array of polygon overlap vertex coordinates
    plane.getOverlapVertices( overlapX.data() );

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
