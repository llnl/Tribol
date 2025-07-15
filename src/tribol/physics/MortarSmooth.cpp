// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "MortarSmooth.hpp"

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
#include <iomanip>

#ifdef TRIBOL_USE_ENZYME
#include "tribol/geom/Normal.hpp"
#include "tribol/common/Enzyme.hpp"
#endif

namespace tribol {

//------------------------------------------------------------------------------
template <>
void ComputeNodalGap<SMOOTH_MORTAR>( SurfaceContactElem& elem )
{
  // check to make sure mortar weights have been computed locally
  // for the SurfaceContactElem object
  SLIC_ERROR_IF( elem.mortarWts == nullptr,
                 "ComputeNodalGap< SMOOTH_MORTAR >: compute local weights on input struct first." );

  // get mesh instance to store gaps on mesh data object
  auto& nonmortarMesh = *elem.m_mesh2;
  IndexT const* const nonmortarConn = nonmortarMesh.getConnectivity().data();

  // will populate local gaps on nonmortar face on nonmortar mesh data object
  SLIC_ERROR_IF( nonmortarMesh.getNodalFields().m_node_gap.empty(),
                 "ComputeNodalGap< SMOOTH_MORTAR >: allocate gaps on mesh data object." );

  SLIC_ERROR_IF( !nonmortarMesh.hasNodalNormals(),
                 "ComputeNodalGap< SMOOTH_MORTAR >: allocate and compute nodal normals on mesh data object." );

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
void ComputeSmoothMortarGaps( CouplingScheme* cs )
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
    // ComputeMortarWeights( elem );

    // compute mortar gaps. Note, we have to now use current configuration
    // nodal coordinates on the contact element
    elem.faceCoords1 = &mortarX[0];
    elem.faceCoords2 = &nonmortarX[0];

    ComputeNodalGap<SMOOTH_MORTAR>( elem );

    // TODO: fix this to register the actual number of active nonmortar gaps.
    // This is not the appropriate data structure to put this information in
    // as the SurfaceContactElem goes out of scope when we exit the loop.
    // HAVE TO set the number of active constraints. For now set to
    // all nonmortar face nodes.
    elem.numActiveGaps = numNodesPerFace;

    ++cpID;

  }  // end loop over pairs to compute nodal gaps

}  // end ComputeSmoothMortarGaps()

//------------------------------------------------------------------------------
template <>
int ApplyNormal<SMOOTH_MORTAR, LAGRANGE_MULTIPLIER>( CouplingScheme* cs )
{
#ifdef TRIBOL_USE_ENZYME
  printf("enzyme enabled\n");
  //if ( cs->isEnzymeEnabled() ) {
  return ApplySmoothNormalEnzyme( cs );
  //}
#endif
  ///////////////////////////////////////////////////////
  //                                                   //
  //            compute smooth mortar gaps             //
  //                                                   //
  // Note, this routine is guarded against null meshes //
  ///////////////////////////////////////////////////////
  ComputeSmoothMortarGaps( cs );

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
      ComputeSmoothMortarJacobian( elem );
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
void ComputeResidualJacobian<SMOOTH_MORTAR, PRIMAL>( SurfaceContactElem& TRIBOL_UNUSED_PARAM( elem ) )
{
  // There is no Jacobian contribution for this block. Be safe and zero out...
  return;
}

//------------------------------------------------------------------------------
template <>
void ComputeResidualJacobian<SMOOTH_MORTAR, DUAL>( SurfaceContactElem& elem )
{
  auto& nonmortarMesh = *elem.m_mesh2;
  IndexT const* const nonmortarConn = nonmortarMesh.getConnectivity().data();

  // loop over "a" nodes accumulating sums of mortar/nonmortar
  // and nonmortar/nonmortar weights
  for ( int a = 0; a < elem.numFaceVert; ++a ) {
    // smooth loop over "b" nodes accumulating sums of
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
void ComputeConstraintJacobian<SMOOTH_MORTAR, PRIMAL>( SurfaceContactElem& elem )
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

    // smooth loop over "b" nodes accumulating sums of
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
void ComputeConstraintJacobian<SMOOTH_MORTAR, DUAL>( SurfaceContactElem& TRIBOL_UNUSED_PARAM( elem ) )
{
  // unless we end up solving the complementarity equation, there is
  // no Jacobian contribtion for this block. Zero out to be safe...
  return;
}

//------------------------------------------------------------------------------
void ComputeSmoothMortarJacobian( SurfaceContactElem& elem )
{
  elem.allocateBlockJ( LAGRANGE_MULTIPLIER );

  ComputeResidualJacobian<SMOOTH_MORTAR, PRIMAL>( elem );

  ComputeResidualJacobian<SMOOTH_MORTAR, DUAL>( elem );

  ComputeConstraintJacobian<SMOOTH_MORTAR, PRIMAL>( elem );

  ComputeConstraintJacobian<SMOOTH_MORTAR, DUAL>( elem );

  // Optionally print contact element matrix. Keep commented out here.
  // elem.printBlockJMatrix();

  return;
}

#ifdef TRIBOL_USE_ENZYME

//------------------------------------------------------------------------------
int ApplySmoothNormalEnzyme( CouplingScheme* cs )
{
  printf("smoothed enzyme\n");
  exit(1);
  auto planes_view = cs->get3DContactPlanes().view();
  auto& lm_opts = cs->getEnforcementOptions().lm_implicit_options;
  bool compute_jacobian = false;
  if ( lm_opts.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN ||
       lm_opts.eval_mode == ImplicitEvalMode::MORTAR_JACOBIAN ) {
    if ( lm_opts.sparse_mode == SparseMode::MFEM_ELEMENT_DENSE ) {
      cs->getMethodData()->reserveBlockJ(
          { BlockSpace::NONMORTAR, BlockSpace::MORTAR, BlockSpace::LAGRANGE_MULTIPLIER }, planes_view.size() );
      cs->createNormalJacobian();
      cs->getdnMethodData()->reserveBlockJ(
          { BlockSpace::NONMORTAR, BlockSpace::MORTAR, BlockSpace::LAGRANGE_MULTIPLIER }, planes_view.size() );
      compute_jacobian = true;
    } else {
      SLIC_WARNING( "Unsupported Jacobian storage method." );
      return 1;
    }
  }
  // convention: 1 = nonmortar
  //             2 = mortar
  cs->createNodalNormal( std::make_unique<VertexAvgNormal>( compute_jacobian ) );
  cs->getNodalNormal()->Compute( cs->getMesh2() );
  auto mesh1 = cs->getMesh2().getView();  // switched from tribol convention
  auto mesh2 = cs->getMesh1().getView();  // switched from tribol convention
  int size1 = mesh1.numberOfNodesPerElement();
  int size2 = mesh2.numberOfNodesPerElement();

  for ( auto& plane : planes_view ) {
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
      constexpr int n_disp = 12;
      for ( int i{ 0 }; i < 2; ++i ) {
        blockJ_n( i, 0 ) = DeviceArray2D<RealT>( n_disp, n_disp );
        blockJ_n( i, 0 ).fill( 0.0 );
      }
      constexpr int n_multipliers = 4;
      blockJ_n( 2, 0 ) = DeviceArray2D<RealT>( n_multipliers, n_disp );
      blockJ_n( 2, 0 ).fill( 0.0 );

      StackArray<DeviceArray2D<RealT>, 9> blockJ( 3 );
      for ( int i{}; i < 2; ++i ) {
        for ( int j{}; j < 2; ++j ) {
          blockJ( i, j ) = DeviceArray2D<RealT>( n_disp, n_disp );
          blockJ( i, j ).fill( 0.0 );
        }
      }
      for ( int i{}; i < 2; ++i ) {
        blockJ( i, 2 ) = DeviceArray2D<RealT>( n_disp, n_multipliers );
        blockJ( i, 2 ).fill( 0.0 );
        // transpose
        blockJ( 2, i ) = DeviceArray2D<RealT>( n_multipliers, n_disp );
        blockJ( 2, i ).fill( 0.0 );
      }
      blockJ( 2, 2 ) = DeviceArray2D<RealT>( n_multipliers, n_multipliers );
      blockJ( 2, 2 ).fill( 0.0 );

      ComputeSmoothMortarJacobianEnzyme( x1, n1, p1, f1, blockJ( 0, 0 ).data(), blockJ( 0, 1 ).data(),
                                   blockJ_n( 0, 0 ).data(), blockJ( 0, 2 ).data(), g1, blockJ( 2, 0 ).data(),
                                   blockJ( 2, 1 ).data(), blockJ_n( 2, 0 ).data(), size1, x2, f2, blockJ( 1, 0 ).data(),
                                   blockJ( 1, 1 ).data(), blockJ_n( 1, 0 ).data(), blockJ( 1, 2 ).data(), size2 );

      if ( lm_opts.sparse_mode == SparseMode::MFEM_ELEMENT_DENSE ) {
        cs->getMethodData()->storeElemBlockJ( { elem1, elem2, elem1 }, blockJ );
        cs->getdnMethodData()->storeElemBlockJ( { elem1, elem2, elem1 }, blockJ_n );
      } else {
        SLIC_WARNING( "Unsupported Jacobian storage method." );
        return 1;
      }
    } else if ( lm_opts.eval_mode == ImplicitEvalMode::MORTAR_GAP ||
                lm_opts.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL ) {
      ComputeSmoothMortarForceEnzyme( x1, n1, p1, f1, g1, size1, x2, f2, size2 );
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
void find_normal(const RealT* coord1, const RealT* coord2, RealT* normal) {
    RealT dx = coord2[0] - coord1[0];
    RealT dy = coord2[1] - coord1[1];
    RealT len = std::sqrt(dy * dy + dx * dx);
    dx /= len;
    dy /= len;
    normal[0] = dy;
    normal[1] = -dx;
}

 void determine_legendre_nodes(int N, double* N_i) {
    if (N==1) {
       N_i[0] = 0.0; 
    }
    else if(N==2) {
        N_i[0] = -1 / std::sqrt(3);
        N_i[1] = 1 / std::sqrt(3);
    }
    else if(N==3) {
        N_i[0] = -std::sqrt(3.0/5.0);
        N_i[1] = 0.0;
        N_i[2] = std::sqrt(3.0/5.0);
    }
    else {
        N_i[0] = -1.0 * std::sqrt((15 + 2 * std::sqrt(30)) / 35);
        N_i[1] = -1.0 * std::sqrt((15 - 2 * std::sqrt(30)) / 35);
        N_i[2] = -std::sqrt((15 - 2 * std::sqrt(30)) / 35);
        N_i[4] = -std::sqrt((15 + 2 * std::sqrt(30)) / 35);
    }
 }

 void determine_legendre_weights(int N, double* W) {
    if (N == 1) {
        W[0] = 2.0;
    }
    else if(N == 2) {
        W[0] = 1.0;
        W[1] = 1.0;
    }
    else if (N == 3) {
        W[0] = 5.0 / 9.0;
        W[1] = 8.0 / 9.0;
        W[2] = 5.0 / 9.0;
    }
    else {
        W[0] = (18 - std::sqrt(30)) / 36.0;
        W[1] = (18 + std::sqrt(30)) / 36.0;
        W[2] = (18 + std::sqrt(30)) / 36.0;
        W[3] = (18 - std::sqrt(30)) / 36.0;
    }
 }


void iso_map(const RealT* coord1, const RealT* coord2, RealT xi, RealT* mapped_coord){
    double N1 = 0.5 - xi;
    double N2 = 0.5 + xi;
    mapped_coord[0] = N1 * coord1[0] + N2 * coord2[0];
    mapped_coord[1] =  N1 * coord1[1] + N2 * coord2[1];
}

bool segmentsIntersect(const RealT A0[2], const RealT A1[2],
                       const RealT B0[2], const RealT B1[2],
                       RealT intersection[2]) {
    auto cross = [](RealT x0, RealT y0, RealT x1, RealT y1) {
        return x0 * y1 - y0 * x1;
    };

    RealT dxA = A1[0] - A0[0], dyA = A1[1] - A0[1];
    RealT dxB = B1[0] - B0[0], dyB = B1[1] - B0[1];
    RealT dxAB = B0[0] - A0[0], dyAB = B0[1] - A0[1];

    RealT denom = cross(dxA, dyA, dxB, dyB);
    RealT numeA = cross(dxAB, dyAB, dxB, dyB);
    RealT numeB = cross(dxAB, dyAB, dxA, dyA);

    // Collinear or parallel
    if (std::abs(denom) < 1e-12) {
        if (std::abs(numeA) > 1e-12 || std::abs(numeB) > 1e-12)
            return false; // Parallel, not collinear

        // Collinear: check for overlap
        auto between = [](RealT a, RealT b, RealT c) {
            return std::min(a, b) <= c && c <= std::max(a, b);
        };

        // Check if endpoints overlap
        for (int i = 0; i < 2; ++i) {
            if (between(A0[0], A1[0], B0[0]) && between(A0[1], A1[1], B0[1])) {
                intersection[0] = B0[0];
                intersection[1] = B0[1];
                return true;
            }
            if (between(A0[0], A1[0], B1[0]) && between(A0[1], A1[1], B1[1])) {
                intersection[0] = B1[0];
                intersection[1] = B1[1];
                return true;
            }
            if (between(B0[0], B1[0], A0[0]) && between(B0[1], B1[1], A0[1])) {
                intersection[0] = A0[0];
                intersection[1] = A0[1];
                return true;
            }
            if (between(B0[0], B1[0], A1[0]) && between(B0[1], B1[1], A1[1])) {
                intersection[0] = A1[0];
                intersection[1] = A1[1];
                return true;
            }
        }
        // Overlap but not at a single point
        return false;
    }


    RealT ua = numeA / denom;
    RealT ub = numeB / denom;

    if (ua >= 0.0 && ua <= 1.0 && ub >= 0.0 && ub <= 1.0) {
        intersection[0] = A0[0] + ua * dxA;
        intersection[1] = A0[1] + ua * dyA;
        return true;
    }
    return false;
}


void find_intersection(const RealT* A0, const RealT* A1, const RealT* p, const RealT* nB, RealT* intersection) {
    RealT tA[2] = {A1[0] - A0[0], A1[1] - A0[1] };
    RealT d[2] = {p[0] - A0[0], p[1] - A0[1]};

    RealT det = tA[0] * nB[1] - tA[1] * nB[0];

    if(std::abs(det) < 1e-12) {
        intersection[0] = p[0];
        intersection[1] = p[1];
    }

    RealT inv_det = 1.0 / det;

    RealT alpha = (d[0] * nB[1] - d[1] * nB[0]) * inv_det;

    if (alpha < 0.0) alpha = 0.0;
    if (alpha > 1.0) alpha = 1.0;

    intersection[0] = (A0[0] + alpha * tA[0]);
    intersection[1] = A0[1]  + alpha * tA[1];

}


void get_projections(const RealT* A0, const RealT* A1, const RealT* B0, const RealT* B1, RealT* projections, RealT del) {
    RealT nA[2] = {0.0};
    RealT nB[2] = {0.0}; 
    find_normal(A0, A1, nA);
    find_normal(B0, B1, nB);
    RealT end_points[2] = {-0.5, 0.5}; 
    for (int i = 0; i < 2; ++i) {
        RealT p[2] = {0.0};

        RealT intersection[2] = {0.0};
        RealT seg_intersection[2] = {0.0};
        iso_map(B0, B1, end_points[i], p);
        find_intersection(A0, A1, p, nB, intersection);

        RealT dx = A1[0] - A0[0];
        RealT dy = A1[1] - A0[1];
        RealT len2 = dx*dx + dy*dy;
        RealT xiA = ((intersection[0] - A0[0]) * dx + (intersection[1] - A0[1]) * dy) / len2;
        RealT nB_unit[2] = { nB[0], nB[1] };
        RealT norm = std::sqrt(nB_unit[0]*nB_unit[0] + nB_unit[1]*nB_unit[1]);
        nB_unit[0] /= norm;
        nB_unit[1] /= norm;

        RealT dx_gap = intersection[0] - p[0];
        RealT dy_gap = intersection[1] - p[1];
        RealT gap = dx_gap * nB_unit[0] + dy_gap * nB_unit[1];

        if(segmentsIntersect(A0, A1, B0, B1, seg_intersection) &&  gap > 0.0) {

                xiA = ((seg_intersection[0] - A0[0]) * dx + (seg_intersection[1] - A0[1]) * dy) / len2;
                if (xiA < del) { 
                  xiA = del;
                }

        }
        xiA = xiA - 0.5;
        projections[i] = xiA;
    }
}


void compute_integration_bounds(const RealT* projections, RealT* integration_bounds) {
    RealT xi_min = projections[0];
    RealT xi_max = projections[0];
    for (int i = 0; i < 2; ++i) {
        if (xi_min > projections[i]) {
            xi_min = projections[i];
        }
        if(xi_max < projections[i]) {
            xi_max = projections[i]; 
        }

    }

    if (xi_max < -0.5) {
        xi_max = -0.5;
    }
    if(xi_min > 0.5) {
        xi_min  = 0.5;
    }
    if (xi_min < -0.5) { 
        xi_min = -0.5;
    }
    if (xi_max > 0.5) {
        xi_max = 0.5;
    }

    integration_bounds[0] = xi_min;
    integration_bounds[1] = xi_max;
}

void modify_bounds(const RealT* integration_bounds, RealT del, RealT* modified_bounds) {
    RealT xi = 0.0;

    RealT int_bound[2] = {0.0};
    for(int i = 0; i < 2; ++i) {
        int_bound[i] = integration_bounds[i];
    }
    for (int i = 0; i < 2; ++i) {
        RealT xi_hat = 0.0;
        xi = int_bound[i] + 0.5;
        if (0.0 - del <= xi && xi <= del) {
            xi_hat = (1.0/(4*del)) * (xi*xi) + 0.5 * xi + del/4.0;
        }
        else if((1.0 - del) <= xi && xi <= 1.0 + del) {
        RealT b = -1.0/(4.0*del);
        RealT c = 0.5 + 1.0/(2.0*del);
        RealT d = 1.0 - del + (1.0/(4.0*del)) * pow(1.0-del, 2) - 0.5*(1.0-del) - (1.0-del)/(2.0*del);

        xi_hat = b*xi*xi + c*xi + d;
        }
        else if(del <= xi && xi <= (1.0 - del)) { 
            xi_hat = xi;
        }
        else{ 
            std::cerr << "Xi did not fall in an expected range for modifying bounds" << std::endl;
        }
        modified_bounds[i] = xi_hat - 0.5;
    }
}

void modify_bounds_for_weight(const RealT* integration_bounds, RealT del, RealT* modified_bounds) {
    RealT xi = 0.0;
    for (int i = 0; i < 2; ++i) {
        RealT xi_hat = 0.0;
        xi = integration_bounds[i] + 0.5;

        if (xi < std::abs(1e-10)) {
            xi = 0.0;
        }
        if (0.0 <= xi && xi <= del) {
            xi_hat = (xi * xi) / (2.0 * del * (1.0 - del));
        }
        else if((1.0 - del) <= xi && xi <= 1.0) {
            xi_hat =  1.0 -(((1.0- xi) * (1.0 - xi)) / (2 * del * (1.0 - del)));
        }
        else if(del <= xi && xi <= (1.0 - del)) { 
            xi_hat = ((2.0 * xi) - del) / (2.0 * (1.0 - del));
        }
        else{ 
            std::cerr << "Xi did not fall in an expected range for modifying bounds" << std::endl;
        }
        modified_bounds[i] = xi_hat - 0.5;
    }
}


void compute_quadrature_point(const RealT* integration_bounds, const RealT* A0, const RealT* A1, int N, RealT* quad_points) {
    RealT eta_values[N];
    determine_legendre_nodes(N, eta_values);

    for (int i = 0; i < N; ++i) {
        eta_values[i] *= 0.5;
    }

    RealT xi_min = integration_bounds[0];
    RealT xi_max = integration_bounds[1];

    for ( int i = 0; i < N; ++i) {
        RealT xi_i = 0.5 * (xi_max + xi_min) + eta_values[i] * (xi_max - xi_min); 
        RealT mapped_coords[2] = {0.0, 0.0};
        iso_map(A0, A1, xi_i, mapped_coords);
        quad_points[2 * i] = mapped_coords[0];
        quad_points[2 * i + 1] = mapped_coords[1];   
    }     
}

void assign_weights(const RealT* integration_bounds, int N, RealT* weights) {
    RealT ref_weights[N];
    determine_legendre_weights(N, ref_weights);
    RealT J = 0.0;
    RealT xi_min = integration_bounds[0];
    RealT xi_max = integration_bounds[1];
    
    J = 0.5 * (xi_max - xi_min);

    for( int i = 0; i < N; ++i) {
        weights[i] = ref_weights[i] * J;
    }
}

RealT compute_gap(const RealT* p, const RealT* B0, const RealT* B1, const RealT* nB) {
    RealT nB_orig[2] = {nB[0], nB[1]};

    RealT len = std::sqrt(nB[0] * nB[0] + nB[1] * nB[1]);
    nB_orig[0] /= len;
    nB_orig[1] /= len;
    RealT intersection[2] = {0.0};
    find_intersection(B0, B1, p, nB_orig, intersection);


    RealT dx = intersection[0] - p[0];
    RealT dy = intersection[1] - p[1];

    RealT gap = dx * nB_orig[0] + dy * nB_orig[1];
    return gap;
}

RealT compute_modified_gap(RealT gap, RealT* nA, RealT* nB) {
    RealT dot = nA[0] * nB[0] + nA[1] * nB[1];
    RealT eta = (dot < 0) ? -dot:0.0;
    return gap * eta;
}

RealT compute_contact_potential(RealT gap, RealT k1, RealT k2) {
    if (gap < 1e-10) {
        return 0;
    }
    RealT pot = k1 * (gap * gap) - k2 * (gap * gap * gap);
    return pot;
}




void ComputeSmoothMortarEnergyEnzyme(const RealT* coords, RealT del, RealT k1, RealT k2, int N, RealT lenA, RealT lenB, RealT* energy) {
    RealT A0[2] = {coords[0], coords[1]};
    RealT A1[2] = {coords[2], coords[3]};
    RealT B0[2] = {coords[4], coords[5]};
    RealT B1[2] = {coords[6], coords[7]};


    RealT AC[2] = {0.5 * (A0[0]+A1[0]), 0.5*(A0[1]+A1[1])};
    RealT AR[2] = {0.5 * (A0[0]-A1[0]), 0.5*(A0[1]-A1[1])};
    RealT normAR = std::sqrt(AR[0]*AR[0] + AR[1]*AR[1]);

    RealT BC[2] = {0.5 * (B0[0]+B1[0]), 0.5*(B0[1]+B1[1])};
    RealT BR[2] = {0.5 * (B0[0]-B1[0]), 0.5*(B0[1]-B1[1])};
    RealT normBR = std::sqrt(BR[0]*BR[0] + BR[1]*BR[1]);

    A0[0] = AC[0] + AR[0] * lenA * 0.5 / normAR;
    A0[1] = AC[1] + AR[1] * lenA * 0.5 / normAR;

    A1[0] = AC[0] - AR[0] * lenA * 0.5 / normAR;
    A1[1] = AC[1] - AR[1] * lenA * 0.5 / normAR;

    B0[0] = BC[0] + BR[0] * lenB * 0.5 / normBR;
    B0[1] = BC[1] + BR[1] * lenB * 0.5 / normBR;;

    B1[0] = BC[0] - BR[0] * lenB * 0.5 / normBR;;
    B1[1] = BC[1] - BR[1] * lenB * 0.5 / normBR;;

    RealT nA[2] = {0.0};
    RealT nB[2] = {0.0};
    find_normal(A0, A1, nA);
    find_normal(B0, B1, nB);

    RealT dot_product = nA[0] * nB[0] + nA[1] * nB[1];

    if (std::abs(dot_product) < 1e-10) {
        *energy = 0;
    }

    else{

    RealT projections[2];
    get_projections(A0, A1, B0, B1, projections);

    RealT integration_bounds[2];
    compute_integration_bounds(projections, integration_bounds);

    RealT modified_bounds[2];
    modify_bounds(integration_bounds, del, modified_bounds);

    RealT modified_bounds_w[2];
    modify_bounds_for_weight(integration_bounds, del, modified_bounds_w); 

    RealT quad_points[2 * N];
    compute_quadrature_point(modified_bounds, A0, A1, N, quad_points);

    RealT weights[N];
    assign_weights(modified_bounds_w, N, weights);

    *energy = 0.0;
    for(int i = 0; i < N; ++i) {
        RealT p[2] = {quad_points[2 * i], quad_points[2 * i + 1]};
        RealT gap = compute_gap(p, B0, B1, nB);
        RealT smooth_gap = compute_modified_gap(gap, nA, nB);
        RealT potential = compute_contact_potential(smooth_gap, k1, k2);
        *energy +=  weights[i] * potential;
    }
    *energy *= lenA * 0.5;

}}


// void ComputeSmoothMortarEnergyEnzyme( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* g1, int size1,
//                                      const RealT* x2, RealT* f2, int size2 )

// {

// }


//--------------------------------------------------------------------------------









//--------------------------------------------------------------------------------





void ComputeSmoothMortarForceEnzyme(RealT* coords, RealT del, RealT k1, RealT k2, int N, RealT lenA, RealT lenB, RealT* dE_dX) 
{
    double dcoords[8] = {0.0};
    double E = 0.0;
    double dE = 1.0;
    __enzyme_autodiff<void>( (void*) ComputeSmoothMortarEnergyEnzyme, enzyme_dup, coords, dcoords, enzyme_const, del, enzyme_const, k1, enzyme_const, k2, enzyme_const, N, enzyme_const, lenA, enzyme_const, lenB,enzyme_dup, &E, &dE);

    for(int i = 0; i < 8; ++i) {
        dE_dX[i] = -dcoords[i];
    }
}


// void ComputeSmoothMortarForceEnzyme( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* g1, int size1,
//                                      const RealT* x2, RealT* f2, int size2 )
// {






//------------------------------------------------------------------------------

void ComputeSmoothMortarJacobianEnzyme(RealT* coords, RealT del, RealT k1, RealT k2, int N, RealT lenA, RealT lenB, RealT* d2E_d2X) {
    RealT dE[8] = {0.0};
    RealT d2E[8] = {0.0};
    for(int i = 0; i < 8; ++i) {
        RealT d2coords[8] = {0.0};
        d2coords[i] = 1.0;
        RealT d2k1 = 0.0;
        RealT d2del = 0.0;
        RealT d2k2 = 0.0;
        RealT d2lenA = 0.0;
        RealT d2lenA = 0.0;
        __enzyme_fwddiff<void>( (void*) ComputeSmoothMortarForceEnzyme, coords, d2coords, del, d2del, k1, d2k1, k2, d2k2, N, lenA, d2lenA, lenB, d2lenB, &dE, &d2E);
        for(int j = 0; j < 8; ++j) {
            d2E_d2X[8 * i + j] = d2E[j];
        }

    }
}

// void ComputeSmoothMortarJacobianEnzyme( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* df1dx1,
//                                         RealT* df1dx2, RealT* df1dn1, RealT* df1dp1, RealT* g1, RealT* dg1dx1, RealT* dg1dx2,
//                                         RealT* dg1dn1, int size1, const RealT* x2, RealT* f2, RealT* df2dx1, RealT* df2dx2,
//                                         RealT* df2dn1, RealT* df2dp1, int size2 )
// {
// }
#endif

//------------------------------------------------------------------------------

}  // end namespace tribol
