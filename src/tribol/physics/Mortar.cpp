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
#include "axom/primal.hpp"

#include <iostream>

#ifdef TRIBOL_USE_ENZYME
#include "tribol/common/Enzyme.hpp"
#endif

namespace tribol
{

void ComputeMortarWeights( SurfaceContactElem & elem )
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

   for (int i=0; i<elem.numFaceVert; ++i)
   {
      x1[i] = elem.faceCoords1[elem.dim*i];
      y1[i] = elem.faceCoords1[elem.dim*i+1];
      z1[i] = elem.faceCoords1[elem.dim*i+2];
      x2[i] = elem.faceCoords2[elem.dim*i];
      y2[i] = elem.faceCoords2[elem.dim*i+1];
      z2[i] = elem.faceCoords2[elem.dim*i+2];
   }

   // allocate mortar weights array on SurfaceContactElem object. This routine 
   // also initializes the array
   elem.allocateMortarWts();

   RealT phiNonmortarA, phiNonmortarB, phiMortarA;

   // loop over number of nodes on the nonmortar or mortar depending on whether forming 
   // nonmortar/nonmortar or mortar/nonmortar weights
  
   for (int a=0; a<elem.numFaceVert; ++a)
   {
      // loop over number of nodes on nonmortar side
      for (int b=0; b<elem.numFaceVert; ++b)
      {
         // set nonmortar/nonmortar and mortar/nonmortar ids...Don't change these ids
         int nonmortarNonmortarId = elem.numFaceVert * a + b;
         int mortarNonmortarId = elem.numFaceVert * elem.numFaceVert + elem.numFaceVert * a + b;

         // loop over number of integration points
         for (int ip=0; ip<integ.numIPs; ++ip)
         {
            // The integration method for computing weights uses 
            // the inverse isoparametric mapping of a current configuration 
            // integration point (as projected onto the current configuration 
            // face) to obtain a (xi,eta) coordinate pair in parent space 
            // for the evaluation of Lagrange shape functions
            RealT xp[3] = { integ.xy[elem.dim*ip], integ.xy[elem.dim*ip+1], integ.xy[elem.dim*ip+2] };
            RealT xi[2] = { 0., 0. };

            InvIso( xp, x1, y1, z1, elem.numFaceVert, xi );
            LinIsoQuadShapeFunc( xi[0], xi[1], a, phiMortarA );

            InvIso( xp, x2, y2, z2, elem.numFaceVert, xi );
            LinIsoQuadShapeFunc( xi[0], xi[1], a, phiNonmortarA );
            LinIsoQuadShapeFunc( xi[0], xi[1], b, phiNonmortarB );

            SLIC_ERROR_IF(nonmortarNonmortarId > elem.numWts || mortarNonmortarId > elem.numWts,
                          "ComputeMortarWts: integer ids for weights exceed elem.numWts");

            // compute nonmortar/nonmortar mortar weight
            elem.mortarWts[ nonmortarNonmortarId ]  += integ.wts[ip] * phiNonmortarA * phiNonmortarB;


            // compute mortar/nonmortar mortar weight
            elem.mortarWts[ mortarNonmortarId ] += integ.wts[ip] * phiMortarA * phiNonmortarB;

         } // end loop over integration points

      } // end loop over nodes on side 2

   } // end loop over nodes on side 1
   
} // end ComputeMortarWeights()

//------------------------------------------------------------------------------
template< >
void ComputeNodalGap< SINGLE_MORTAR >( SurfaceContactElem & elem )
{
   // check to make sure mortar weights have been computed locally 
   // for the SurfaceContactElem object
   SLIC_ERROR_IF(elem.mortarWts==nullptr, "ComputeNodalGap< SINGLE_MORTAR >: compute local weights on input struct first.");

   // get mesh instance to store gaps on mesh data object
   auto& nonmortarMesh = *elem.m_mesh2;
   IndexT const * const nonmortarConn = nonmortarMesh.getConnectivity().data();

   // will populate local gaps on nonmortar face on nonmortar mesh data object
   SLIC_ERROR_IF(nonmortarMesh.getNodalFields().m_node_gap.empty(),
                 "ComputeNodalGap< SINGLE_MORTAR >: allocate gaps on mesh data object."); 

   SLIC_ERROR_IF(!nonmortarMesh.hasNodalNormals(),
                 "ComputeNodalGap< SINGLE_MORTAR >: allocate and compute nodal normals on mesh data object.");   

   // compute gap contributions associated with face 2 on the SurfaceContactElem 
   // (i.e. nonmortar surface)

   // loop over nodes on nonmortar side
   for (int a=0; a<elem.numFaceVert; ++a)
   {
      // initialize gap1 and gap2 terms
      RealT g1 = 0.;
      RealT g2 = 0.;

      // get global nonmortar node number from connectivity
      RealT nrml_a[elem.dim];
      int glbId = nonmortarConn[ elem.numFaceVert * elem.faceId2 + a ];
      nrml_a[0] = nonmortarMesh.getNodalNormals()[0][ glbId ];
      nrml_a[1] = nonmortarMesh.getNodalNormals()[1][ glbId ];
      if (elem.dim == 3 )
      {
         nrml_a[2] = nonmortarMesh.getNodalNormals()[2][ glbId ];
      }

      // sum contributions from both sides
      for (int b=0; b<elem.numFaceVert; ++b)
      {
         // compute nonmortar-mortar and nonmortar-nonmortar ids. Note, n_ab is 
         // the stored mortar weight. For mortar-nonmortar mortar weights, 
         // a = mortar node and b = nonmortar node, BUT FOR THE GAP COMPUTATION,
         // THE SUM OF MORTAR WEIGHTS IS ACTUALLY OVER SHAPE FUNCTIONS 
         // DEFINED AT NODE "b", SO WE NEED TO USE (n_ab)^T.
         RealT nab_1 = elem.getNonmortarMortarWt( a, b ); // nonmortar-mortar weight
         RealT nab_2 = elem.getNonmortarNonmortarWt( a, b ); // nonmortar-nonmortar weight

         g1 += dotProd( &nrml_a[0], &elem.faceCoords1[ elem.dim * b ], elem.dim ) *
               nab_1;
         g2 += dotProd( &nrml_a[0], &elem.faceCoords2[ elem.dim * b ], elem.dim ) * 
               nab_2;
      }

      // store local gap
      nonmortarMesh.getNodalFields().m_node_gap[ glbId ] += (g1-g2);

   } // end a-loop over nonmortar nodes

} // end ComputeNodalGap<>()

//------------------------------------------------------------------------------
void ComputeSingleMortarGaps( const CouplingScheme* cs )
{
   MeshManager& meshManager = MeshManager::getInstance();
   MeshData& nonmortarMeshBase = meshManager.at( cs->getMeshId2() );
   // compute nodal normals (do this outside the element loop)
   // Note, this is guarded against zero element meshes
   int const dim = cs->spatialDimension();
   nonmortarMeshBase.computeNodalNormals( dim );
   // mesh data has changed.  update coupling scheme mesh view
   // TODO: get rid of the const cast
   auto cs_mutable = const_cast<CouplingScheme*>(cs);
   cs_mutable->updateMeshViews();

   auto pairs = cs->getInterfacePairsView();
   const IndexT numPairs = pairs.size();
   auto planes = cs->get3DContactPlanesView();

   ////////////////////////////////////////////////////////////////////////
   //
   // Grab pointers to mesh data
   //
   ////////////////////////////////////////////////////////////////////////
   auto& mortarMesh = cs->getMesh1();
   auto& nonmortarMesh = cs->getMesh2();

   IndexT const numNodesPerFace = mortarMesh.numberOfNodesPerElement();

   RealT const * const x1 = mortarMesh.getPosition()[0].data();
   RealT const * const y1 = mortarMesh.getPosition()[1].data(); 
   RealT const * const z1 = mortarMesh.getPosition()[2].data(); 
   IndexT const * const mortarConn= mortarMesh.getConnectivity().data();

   RealT const * const x2 = nonmortarMesh.getPosition()[0].data(); 
   RealT const * const y2 = nonmortarMesh.getPosition()[1].data();
   RealT const * const z2 = nonmortarMesh.getPosition()[2].data();
   IndexT const * nonmortarConn = nonmortarMesh.getConnectivity().data();

   // declare local variables to hold face nodal coordinates
   // and overlap vertex coordinates
   IndexT size = dim * numNodesPerFace;
   RealT mortarX[ size ];
   RealT nonmortarX[ size ];

   ////////////////////////////////////////////////////////////////////
   // compute nonmortar gaps to determine active set of contact dofs //
   ////////////////////////////////////////////////////////////////////
   int cpID = 0;
   for (IndexT kp = 0; kp < numPairs; ++kp)
   {
      auto& pair = pairs[kp];

      if (!pair.m_is_contact_candidate)
      {
         continue;
      }

      auto& plane = planes[cpID];

      // get pair indices
      IndexT index1 = pair.m_element_id1;
      IndexT index2 = pair.m_element_id2;

      // populate the current configuration nodal coordinates for the 
      // two faces
      for (int i=0; i<numNodesPerFace; ++i)
      {
         int id = dim * i;
         IndexT mortar_id = mortarConn[ numNodesPerFace * index1 + i ];
         IndexT nonmortar_id  = nonmortarConn[ numNodesPerFace * index2 + i ];

         mortarX[ id ]   = x1[ mortar_id ];
         mortarX[ id+1 ] = y1[ mortar_id ];
         mortarX[ id+2 ] = z1[ mortar_id ];
         nonmortarX[ id ]   = x2[ nonmortar_id ];
         nonmortarX[ id+1 ] = y2[ nonmortar_id ];
         nonmortarX[ id+2 ] = z2[ nonmortar_id ];
      }

      // get projected face coordinates
      ArrayT<RealT, 2> mortarX_bar(numNodesPerFace, dim);
      ArrayT<RealT, 2> nonmortarX_bar(numNodesPerFace, dim);
      ArrayT<RealT, 2> mortarX_barT(dim, numNodesPerFace);
      ArrayT<RealT, 2> nonmortarX_barT(dim, numNodesPerFace);
      ProjectFaceNodesToPlane( mortarMesh, index1, 
                               plane.m_nX, plane.m_nY, plane.m_nZ,
                               plane.m_cX, plane.m_cY, plane.m_cZ,
                               &mortarX_barT(0, 0), 
                               &mortarX_barT(1, 0), 
                               &mortarX_barT(2, 0) );
      ProjectFaceNodesToPlane( nonmortarMesh, index2, 
                               plane.m_nX, plane.m_nY, plane.m_nZ,
                               plane.m_cX, plane.m_cY, plane.m_cZ,
                               &nonmortarX_barT(0, 0), 
                               &nonmortarX_barT(1, 0), 
                               &nonmortarX_barT(2, 0) );
      algorithm::transpose<MemorySpace::Dynamic>(mortarX_barT, mortarX_bar);
      algorithm::transpose<MemorySpace::Dynamic>(nonmortarX_barT, nonmortarX_bar);

      // construct array of polygon overlap vertex coordinates
      ArrayT<RealT, 2> overlapX(plane.m_numPolyVert, dim);
      for (IndexT i{0}; i < plane.m_numPolyVert; ++i)
      {
        overlapX(i, 0) = plane.m_polyX[i];
        overlapX(i, 1) = plane.m_polyY[i];
        overlapX(i, 2) = plane.m_polyZ[i];
      }

      // instantiate contact surface element for purposes of computing 
      // mortar weights. Note, this uses projected face coords
      SurfaceContactElem elem( dim, mortarX_bar.data(), nonmortarX_bar.data(), 
                               overlapX.data(),
                               numNodesPerFace, 
                               plane.m_numPolyVert,
                               &mortarMesh, &nonmortarMesh, index1, index2 );

      // compute the mortar weights to be stored on the surface 
      // contact element struct. This must be done prior to computing nodal gaps
      elem.overlapArea = plane.m_area;
      ComputeMortarWeights( elem );

      // compute mortar gaps. Note, we have to now use current configuration
      // nodal coordinates on the contact element
      elem.faceCoords1 = &mortarX[0];
      elem.faceCoords2 = &nonmortarX[0];

      ComputeNodalGap< SINGLE_MORTAR >( elem );

      // TODO: fix this to register the actual number of active nonmortar gaps.
      // This is not the appropriate data structure to put this information in 
      // as the SurfaceContactElem goes out of scope when we exit the loop.
      // HAVE TO set the number of active constraints. For now set to 
      // all nonmortar face nodes.
      elem.numActiveGaps = numNodesPerFace;

      ++cpID;

   } // end loop over pairs to compute nodal gaps

} // end ComputeSingleMortarGaps()

//------------------------------------------------------------------------------
template< >
int ApplyNormal< SINGLE_MORTAR, LAGRANGE_MULTIPLIER >( CouplingScheme* cs )
{
   ///////////////////////////////////////////////////////
   //                                                   //
   //            compute single mortar gaps             //
   //                                                   //
   // Note, this routine is guarded against null meshes //
   ///////////////////////////////////////////////////////
   ComputeSingleMortarGaps( cs );

   auto pairs = cs->getInterfacePairsView();
   const IndexT numPairs = pairs.size();
   auto planes = cs->get3DContactPlanesView();

   int const dim = cs->spatialDimension();

   ////////////////////////////////////////////////////////////////////////
   //
   // Grab pointers to mesh data
   //
   ////////////////////////////////////////////////////////////////////////
   auto& mortarMesh = cs->getMesh1();
   auto& nonmortarMesh = cs->getMesh2();

   IndexT const numNodesPerFace = mortarMesh.numberOfNodesPerElement();

   RealT * const fx1 = mortarMesh.getResponse()[0].data();
   RealT * const fy1 = mortarMesh.getResponse()[1].data(); 
   RealT * const fz1 = mortarMesh.getResponse()[2].data(); 
   IndexT const * const mortarConn= mortarMesh.getConnectivity().data();

   RealT * const fx2 = nonmortarMesh.getResponse()[0].data(); 
   RealT * const fy2 = nonmortarMesh.getResponse()[1].data();
   RealT * const fz2 = nonmortarMesh.getResponse()[2].data();
   IndexT const * nonmortarConn = nonmortarMesh.getConnectivity().data();

   int numTotalNodes = cs->getNumTotalNodes();
   int numRows = dim * numTotalNodes + numTotalNodes;
   const EnforcementOptions& enforcement_options = const_cast<EnforcementOptions&>(cs->getEnforcementOptions());
   const LagrangeMultiplierImplicitOptions& lm_options  = enforcement_options.lm_implicit_options;
   if (!cs->nullMeshes())
   {
      if ( lm_options.sparse_mode == SparseMode::MFEM_ELEMENT_DENSE )
      {
         static_cast<MortarData*>( cs->getMethodData() )->reserveBlockJ( 
            {BlockSpace::MORTAR, BlockSpace::NONMORTAR, BlockSpace::LAGRANGE_MULTIPLIER},
            numPairs
         );
      }
      else if ( lm_options.sparse_mode == SparseMode::MFEM_INDEX_SET || 
                lm_options.sparse_mode == SparseMode::MFEM_LINKED_LIST )
      {
         static_cast<MortarData*>( cs->getMethodData() )->allocateMfemSparseMatrix( numRows );
      }
      else
      {
         SLIC_WARNING("Unsupported Jacobian storage method.");
         return 1;
      }
   }

   ////////////////////////////////////////////////////////////////
   //                                                            //
   // compute equilibrium residual and/or Jacobian contributions //
   //                                                            //
   ////////////////////////////////////////////////////////////////
   int cpID = 0;
   for (IndexT kp = 0; kp < numPairs; ++kp)
   {
      auto& pair = pairs[kp];

      if (!pair.m_is_contact_candidate)
      {
         continue;
      }

      auto& plane = planes[cpID];

      // get pair indices
      IndexT index1 = pair.m_element_id1;
      IndexT index2 = pair.m_element_id2;

      // get projected face coordinates
      ArrayT<RealT, 2> mortarX_bar(numNodesPerFace, dim);
      ArrayT<RealT, 2> nonmortarX_bar(numNodesPerFace, dim);
      ArrayT<RealT, 2> mortarX_barT(dim, numNodesPerFace);
      ArrayT<RealT, 2> nonmortarX_barT(dim, numNodesPerFace);
      ProjectFaceNodesToPlane( mortarMesh, index1, 
                               plane.m_nX, plane.m_nY, plane.m_nZ,
                               plane.m_cX, plane.m_cY, plane.m_cZ,
                               &mortarX_barT(0, 0), 
                               &mortarX_barT(1, 0), 
                               &mortarX_barT(2, 0) );
      ProjectFaceNodesToPlane( nonmortarMesh, index2, 
                               plane.m_nX, plane.m_nY, plane.m_nZ,
                               plane.m_cX, plane.m_cY, plane.m_cZ,
                               &nonmortarX_barT(0, 0), 
                               &nonmortarX_barT(1, 0), 
                               &nonmortarX_barT(2, 0) );
      algorithm::transpose<MemorySpace::Dynamic>(mortarX_barT, mortarX_bar);
      algorithm::transpose<MemorySpace::Dynamic>(nonmortarX_barT, nonmortarX_bar);

      // construct array of polygon overlap vertex coordinates
      ArrayT<RealT, 2> overlapX(plane.m_numPolyVert, dim);
      for (IndexT i{0}; i < plane.m_numPolyVert; ++i)
      {
        overlapX(i, 0) = plane.m_polyX[i];
        overlapX(i, 1) = plane.m_polyY[i];
        overlapX(i, 2) = plane.m_polyZ[i];
      }

      // instantiate contact surface element for purposes of computing 
      // mortar weights. Note, this uses projected face coords
      SurfaceContactElem elem( dim, mortarX_bar.data(), nonmortarX_bar.data(), 
                               overlapX.data(),
                               numNodesPerFace, 
                               plane.m_numPolyVert,
                               &mortarMesh, &nonmortarMesh, index1, index2 );

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
      for (int a=0; a<numNodesPerFace; ++a)
      {
         int mortarIdA = mortarConn[ index1 * numNodesPerFace + a];
         int nonmortarIdA = nonmortarConn[ index2 * numNodesPerFace + a ];

         // inner loop over NONMORTAR nodes
         for (int b=0; b<numNodesPerFace; ++b)
         {
            int nonmortarIdB = nonmortarConn[ index2 * numNodesPerFace + b ];

            // We include all nonmortar nodes even if nodal gap is in separation. 
            // NOTE: Per testing, we include ALL nonmortar nodes 
            // in the computation after the geometric filtering and judge contact 
            // activity based on the gap AND the pressure solution

            RealT forceX = nonmortarMesh.getNodalFields().m_node_pressure[ nonmortarIdB ] * 
                          nonmortarMesh.getNodalNormals()[0][ nonmortarIdB ];
            RealT forceY = nonmortarMesh.getNodalFields().m_node_pressure[ nonmortarIdB ] * 
                          nonmortarMesh.getNodalNormals()[1][ nonmortarIdB ];
            RealT forceZ = nonmortarMesh.getNodalFields().m_node_pressure[ nonmortarIdB ] * 
                          nonmortarMesh.getNodalNormals()[2][ nonmortarIdB ];

            // contact nodal force is the interpolated force using mortar 
            // weights n_ab, where "a" is mortar or nonmortar node and "b" is 
            // nonmortar node.
            fx1[ mortarIdA ] += forceX * elem.getMortarNonmortarWt( a, b );
            fy1[ mortarIdA ] += forceY * elem.getMortarNonmortarWt( a, b ); 
            fz1[ mortarIdA ] += forceZ * elem.getMortarNonmortarWt( a, b ); 

            fx2[ nonmortarIdA ]  -= forceX * elem.getNonmortarNonmortarWt( a, b );
            fy2[ nonmortarIdA ]  -= forceY * elem.getNonmortarNonmortarWt( a, b );
            fz2[ nonmortarIdA ]  -= forceZ * elem.getNonmortarNonmortarWt( a, b );

         } // end inner loop over nonmortar nodes

      } // end outer loop over nonmortar and mortar nodes

      //////////////////////////////////////////////////////////
      // compute tangent stiffness contributions if requested //
      //////////////////////////////////////////////////////////
      if ( lm_options.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN ||
           lm_options.eval_mode == ImplicitEvalMode::MORTAR_JACOBIAN )
      {
         ComputeSingleMortarJacobian( elem );
         if ( lm_options.sparse_mode == SparseMode::MFEM_ELEMENT_DENSE )
         {
            static_cast<MortarData*>( cs->getMethodData() )->storeElemBlockJ(
               {elem.faceId1, elem.faceId2, elem.faceId2},
               elem.blockJ
            );
         }
         else if ( lm_options.sparse_mode == SparseMode::MFEM_INDEX_SET || 
                   lm_options.sparse_mode == SparseMode::MFEM_LINKED_LIST )
         {
            static_cast<MortarData*>( cs->getMethodData() )->assembleJacobian( elem, lm_options.sparse_mode );
         }
         else
         {
            SLIC_WARNING("Unsupported Jacobian storage method.");
            return 1;
         }
         
      }

      ++cpID;

   } // end of loop over interface pairs computing residual/Jacobian contributions

   return 0;

} // end ApplyNormal<>()

//------------------------------------------------------------------------------
template< >
void ComputeResidualJacobian< SINGLE_MORTAR, PRIMAL >( SurfaceContactElem & TRIBOL_UNUSED_PARAM(elem) )
{
   // There is no Jacobian contribution for this block. Be safe and zero out...
   return;
}

//------------------------------------------------------------------------------
template< >
void ComputeResidualJacobian< SINGLE_MORTAR, DUAL >( SurfaceContactElem & elem )
{
   auto& nonmortarMesh = *elem.m_mesh2;
   IndexT const * const nonmortarConn = nonmortarMesh.getConnectivity().data();

   // loop over "a" nodes accumulating sums of mortar/nonmortar 
   // and nonmortar/nonmortar weights
   for (int a = 0; a<elem.numFaceVert; ++a)
   {
      // single loop over "b" nodes accumulating sums of 
      // mortar(a)/nonmortar(b) and nonmortar(a)/nonmortar(b) weights
      for (int b = 0; b<elem.numFaceVert; ++b)
      {
         // get global nonmortar node id to index into nodal normals on 
         // nonmortar mesh
         RealT nrml_b[elem.dim];
         int glbId = nonmortarConn[ elem.numFaceVert * elem.faceId2 + b ];

         // We assemble ALL nonmortar node contributions, even if gap is in separation.
         // NOTE: Per testing, we compute ALL nonmortar nodes 
         // for faces that have positive areas of overlap after the geometric 
         // filtering and use the gap AND the pressure solution to determine 
         // contact activity

         nrml_b[0] = nonmortarMesh.getNodalNormals()[0][ glbId ];
         nrml_b[1] = nonmortarMesh.getNodalNormals()[1][ glbId ];
         if (elem.dim == 3 )
         {
            nrml_b[2] = nonmortarMesh.getNodalNormals()[2][ glbId ];
         }

         // get mortar-nonmortar and nonmortar-nonmortar mortar weights
         RealT n_mortar_b = elem.getMortarNonmortarWt( a, b ); // mortar-nonmortar weight
         RealT n_nonmortar_b  = elem.getNonmortarNonmortarWt( a, b ); // nonmortar-nonmortar weight, note negative in formulation
         
         // fill Jrp element-pair Jacobian blocks
         // Fill block (0, 2)
         int elem_xdof = elem.getJacobianIndex(SurfaceContactElem::JrpBlock, a, b );
         int dim_offset = elem.getJacobianDimOffset(SurfaceContactElem::JrpBlock);
         elem.blockJ(
            static_cast<IndexT>(BlockSpace::MORTAR),
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER)
         )[ elem_xdof ]                += nrml_b[0] * n_mortar_b;
         elem.blockJ(
            static_cast<IndexT>(BlockSpace::MORTAR),
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER)
         )[ elem_xdof + dim_offset ]   += nrml_b[1] * n_mortar_b;
         elem.blockJ(
            static_cast<IndexT>(BlockSpace::MORTAR),
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER)
         )[ elem_xdof + 2*dim_offset ] += nrml_b[2] * n_mortar_b;

         // Fill block (1, 2)
         elem.blockJ(
            static_cast<IndexT>(BlockSpace::NONMORTAR),
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER)
         )[ elem_xdof ]                -= nrml_b[0] * n_nonmortar_b;
         elem.blockJ(
            static_cast<IndexT>(BlockSpace::NONMORTAR),
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER)
         )[ elem_xdof + dim_offset ]   -= nrml_b[1] * n_nonmortar_b;
         elem.blockJ(
            static_cast<IndexT>(BlockSpace::NONMORTAR),
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER)
         )[ elem_xdof + 2*dim_offset ] -= nrml_b[2] * n_nonmortar_b;

      } // end loop over b nodes

   } // end loop over a nodes

   return;
} // end ComputeResidualJacobian<>()

//------------------------------------------------------------------------------
template< >
void ComputeConstraintJacobian< SINGLE_MORTAR, PRIMAL >( SurfaceContactElem & elem )
{
   auto& nonmortarMesh = *elem.m_mesh2;
   IndexT const * const nonmortarConn = nonmortarMesh.getConnectivity().data();

   // loop over nonmortar nodes for which we are accumulating Jacobian 
   // contributions
   for (int a = 0; a<elem.numFaceVert; ++a)
   {
      // get global nonmortar node id to index into nodal normals on 
      // nonmortar mesh
      RealT nrml_a[elem.dim];
      int glbId = nonmortarConn[ elem.numFaceVert * elem.faceId2 + a ];

      // We assemble ALL nonmortar node contributions even if gap is in separation.
      // NOTE: Per mortar method testing we compute ALL nonmortar node 
      // contributions for faces that have positive areas of overlap per the 
      // geometric filtering. Contact activity is judged based on gaps AND 
      // the pressure solution.

      nrml_a[0] = nonmortarMesh.getNodalNormals()[0][ glbId ];
      nrml_a[1] = nonmortarMesh.getNodalNormals()[1][ glbId ];
      if (elem.dim == 3 )
      {
         nrml_a[2] = nonmortarMesh.getNodalNormals()[2][ glbId ];
      }

      // single loop over "b" nodes accumulating sums of 
      // nonmortar(a)/mortar(b) and nonmortar(a)/nonmortar(b) weights
      for (int b = 0; b<elem.numFaceVert; ++b)
      {
         // get nonmortar-mortar and nonmortar-nonmortar mortar weights
         RealT n_mortar_a = elem.getNonmortarMortarWt( a, b ); // nonmortar-mortar weight
         RealT n_nonmortar_a  = elem.getNonmortarNonmortarWt( a, b ); // nonmortar-nonmortar weight, note negative in formulation

         // fill Jgu element-pair Jacobian blocks
         // Fill block (2, 0)
         int dim_offset = elem.getJacobianDimOffset(SurfaceContactElem::JguBlock);
         int elem_xdof = elem.getJacobianIndex(SurfaceContactElem::JguBlock, a, b );
         elem.blockJ(
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER),
            static_cast<IndexT>(BlockSpace::MORTAR)
         )[ elem_xdof ]                += nrml_a[0] * n_mortar_a;
         elem.blockJ(
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER),
            static_cast<IndexT>(BlockSpace::MORTAR)
         )[ elem_xdof + dim_offset ]   += nrml_a[1] * n_mortar_a;
         elem.blockJ(
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER),
            static_cast<IndexT>(BlockSpace::MORTAR)
         )[ elem_xdof + 2*dim_offset ] += nrml_a[2] * n_mortar_a;

         // Fill block (2, 1)
         elem.blockJ(
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER),
            static_cast<IndexT>(BlockSpace::NONMORTAR)
         )[ elem_xdof ]                -= nrml_a[0] * n_nonmortar_a;
         elem.blockJ(
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER),
            static_cast<IndexT>(BlockSpace::NONMORTAR)
         )[ elem_xdof + dim_offset ]   -= nrml_a[1] * n_nonmortar_a;
         elem.blockJ(
            static_cast<IndexT>(BlockSpace::LAGRANGE_MULTIPLIER),
            static_cast<IndexT>(BlockSpace::NONMORTAR)
         )[ elem_xdof + 2*dim_offset ] -= nrml_a[2] * n_nonmortar_a;

      } // end loop over b nodes

   } // end loop over a nodes

   return;
} // end ComputeConstraintJacobian

//------------------------------------------------------------------------------
template< >
void ComputeConstraintJacobian< SINGLE_MORTAR, DUAL >( SurfaceContactElem& TRIBOL_UNUSED_PARAM(elem) )
{
   // unless we end up solving the complementarity equation, there is 
   // no Jacobian contribtion for this block. Zero out to be safe...
   return;
}

//------------------------------------------------------------------------------
void ComputeSingleMortarJacobian( SurfaceContactElem & elem )
{
   elem.allocateBlockJ( LAGRANGE_MULTIPLIER );

   ComputeResidualJacobian  < SINGLE_MORTAR, PRIMAL >( elem );

   ComputeResidualJacobian  < SINGLE_MORTAR, DUAL   >( elem );

   ComputeConstraintJacobian< SINGLE_MORTAR, PRIMAL >( elem );

   ComputeConstraintJacobian< SINGLE_MORTAR, DUAL   >( elem );

   // Optionally print contact element matrix. Keep commented out here.
   //elem.printBlockJMatrix();

   return;

}

#ifdef TRIBOL_USE_ENZYME

void LinearQuadBasis(const RealT* xi, RealT* phi)
{
  phi[0] = 0.25*(1 - xi[0])*(1 - xi[1]);
  phi[1] = 0.25*(1 + xi[0])*(1 - xi[1]);
  phi[2] = 0.25*(1 + xi[0])*(1 + xi[1]);
  phi[3] = 0.25*(1 - xi[0])*(1 + xi[1]);
}

void LinearQuadBasisDeriv(const RealT* xi, RealT* phi, RealT* dphi_dxi, RealT* dphi_deta)
{
  double xi_dot[2] = {1.0, 0.0};
  __enzyme_fwddiff<void>((void*)LinearQuadBasis,
    xi, xi_dot,
    phi, dphi_dxi);
  xi_dot[0] = 0.0;
  xi_dot[1] = 1.0;
  __enzyme_fwddiff<void>((void*)LinearQuadBasis,
    xi, xi_dot,
    phi, dphi_deta);
}

// enum class ClippingState
// {
//    Unknown,
//    Inside,
//    Outside,
//    OnBoundary
// };

// enum class ClippingVertexType
// {
//    Mortar,
//    Nonmortar,
//    EdgeEdge
// };

// void CheckAndAddVertex(const RealT* vert, RealT* coords, ClippingVertexType vert_type, ClippingVertexType* coords_type, int vert_id, int* coords_id, int& num_coords)
// {
//    constexpr RealT dist_tol = 1.0e-13;
//    if (num_coords > 0 && (
//       std::abs(coords[(num_coords-1)*3 + 0] - vert[0]) > dist_tol ||
//       std::abs(coords[(num_coords-1)*3 + 1] - vert[1]) > dist_tol ||
//       std::abs(coords[(num_coords-1)*3 + 2] - vert[2]) > dist_tol
//    ))
//    {
//       for (int d{0}; d < 3; ++d)
//       {
//          coords[num_coords*3 + d] = vert[d];
//       }
//       coords_type[num_coords] = vert_type;
//       coords_id[num_coords] = vert_id;
//       ++num_coords;
//    }
// }

void PlaneTo2DCoords(const RealT* x, const RealT* x0, const RealT* e1, const RealT* e2, 
                     RealT* xp, RealT* yp, int num_coords)
{
   for (int i{0}; i < num_coords; ++i)
   {
      xp[i] = 0.0;
      yp[i] = 0.0;

      for (int d{0}; d < 3; ++d)
      {
         RealT v_d = x[3*i + d] - x0[d];
         xp[i] += v_d*e1[d];
         yp[i] += v_d*e2[d];
      }
   }
}

void Coords2DToPlane(const RealT* xp, const RealT* yp, const RealT* x0, 
                     const RealT* e1, const RealT* e2, 
                     RealT* x, int num_coords)
{
   for (int i{0}; i < num_coords; ++i)
   {
      for (int d{0}; d < 3; ++d)
      {
         x[i*3 + d] = x0[d] + xp[i]*e1[d] + yp[i]*e2[d];
      }
   }
}

void ComputeMortarForceEnzyme( const RealT* x1, const RealT* n1, const RealT* p1, 
                               RealT* f1, RealT* g1, int size1,
                               const RealT* x2, 
                               RealT* f2, int size2 )
{
   // convention: elem1 = nonmortar element
   //             elem2 = mortar element
  //  // TODO: set this based on double/float precision
  //  constexpr RealT dist_tol = 1.0e-13;
  //  // cos(pi/2 + angle_tol) = tolerance for aligned edges
  //  constexpr RealT angle_tol = 1.0e-8;
   constexpr int max_mortar_mat_size = 4 * 4;
   RealT mortar_mat1[max_mortar_mat_size];
   int mortar_mat1_size = size1 * size1;
   for (int i{0}; i < mortar_mat1_size; ++i)
   {
      mortar_mat1[i] = 0.0;
   }
   RealT mortar_mat2[max_mortar_mat_size];
   int mortar_mat2_size = size1 * size2;
   for (int i{0}; i < mortar_mat2_size; ++i)
   {
      mortar_mat2[i] = 0.0;
   }
   // get point x0 (geometric center of elem1)
   RealT x0[3] = {0.0, 0.0, 0.0};
   for (int i{0}; i < size1; ++i)
   {
      for (int d{0}; d < 3; ++d)
      {
         x0[d] += x1[i*3 + d] / static_cast<RealT>(size1);
      }
   }
   // get vector n (normal of elem1)
   RealT n[3] = {0.0, 0.0, 0.0};
   for (int i{0}; i < size1; ++i)
   {
      for (int d{0}; d < 3; ++d)
      {
         n[d] += n1[i*3 + d];
      }
   }
   RealT n_mag = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
   for (int d{0}; d < 3; ++d)
   {
      n[d] /= n_mag;
   }
   // x1t = x1 projected to plane p (def'd by x0 and n)
   constexpr int max_coord_size = 4*3;
   RealT x1t[max_coord_size];
   for (int i{0}; i < size1; ++i)
   {
      RealT x1diff_mag = 0.0;
      for (int d{0}; d < 3; ++d)
      {
         x1diff_mag += n[d] * (x1[i*3 + d] - x0[d]);
      }
      for (int d{0}; d < 3; ++d)
      {
         x1t[i*3 + d] = x1[i*3 + d] - n[d]*x1diff_mag;
      }
   }
   // x2t = x2 projected to plane p
   RealT x2t[max_coord_size];
   for (int i{0}; i < size2; ++i)
   {
      RealT x2diff_mag = 0.0;
      for (int d{0}; d < 3; ++d)
      {
         x2diff_mag += n[d] * (x2[i*3 + d] - x0[d]);
      }
      for (int d{0}; d < 3; ++d)
      {
         x2t[i*3 + d] = x2[i*3 + d] - n[d]*x2diff_mag;
      }
   }
  //  // n2_in = inward normal unit vector for edges of elem2
  //  axom::StackArray<axom::primal::Vector<RealT, 3>, 4> n2_in;
  //  for (int i{0}; i < size2; ++i)
  //  {
  //     int vert1 = i;
  //     int vert2 = (i+1) % size2;
  //     // p = vector along edge
  //     axom::primal::Vector<RealT, 3> p({
  //        x2t[vert2*3 + 0] - x2t[vert1*3 + 0],
  //        x2t[vert2*3 + 1] - x2t[vert1*3 + 1],
  //        x2t[vert2*3 + 2] - x2t[vert1*3 + 2]
  //     });
  //     n2_in[i] = p.cross_product(p, n);
  //     n2_in[i] = n2_in[i].unitVector();
  //  }
  //  // Line clipping algorithm (Cyrus-Beck-like)
  //  constexpr int max_int_size = 8*3;
  //  // xti = intersection polygon
  //  RealT xti[max_int_size];
  //  ClippingVertexType xti_type[8];
  //  int xti_id[8];
  //  int poly_vert_ct = 0;
  //  ClippingState poly_status = ClippingState::Unknown;
  //  // loop over vertices and edges of nonmortar element
  //  for (int i{0}; i < size1; ++i)
  //  {
  //     // check if nonmortar vertex is inside/outside/on edge of mortar element
  //     ClippingState vert_status = ClippingState::Inside;
  //     for (int j{0}; j < size2; ++j)
  //     {
  //        // x1_to_edge = vector from edge vert1 to elem1 vertex
  //        axom::primal::Vector<RealT, 3> x1_to_edge({
  //           x1t[i*3 + 0] - x2t[j*3 + 0],
  //           x1t[i*3 + 1] - x2t[j*3 + 1],
  //           x1t[i*3 + 2] - x2t[j*3 + 2]
  //        });
  //        RealT x1_dot_n2 = x1_to_edge.dot(n2_in[j]);
  //        if (x1_dot_n2 < -dist_tol) // TODO: investigate scaling dist_tol based on edge size/element size
  //        {
  //           vert_status = ClippingState::Outside;
  //           break;
  //        }
  //        else if (std::abs(x1_dot_n2) < dist_tol)
  //        {
  //           vert_status = ClippingState::OnBoundary;
  //        }
  //     }
  //     if (vert_status == ClippingState::Inside)
  //     {
  //       // if vertex is inside, add to list
  //        SLIC_ERROR_IF(poly_status == ClippingState::Outside, "Potentially missed edge/edge intersection.");
  //        poly_status = ClippingState::Inside;
  //        CheckAndAddVertex(&x1t[i*3], xti, ClippingVertexType::Nonmortar, xti_type, i, xti_id, poly_vert_ct);
  //     }
  //     else if (vert_status == ClippingState::OnBoundary)
  //     {
  //        if (poly_status == ClippingState::Inside)
  //        {
  //           // vertex is inside if we are inside and on boundary; add to list
  //           poly_status = ClippingState::OnBoundary;
  //           CheckAndAddVertex(&x1t[i*3], xti, ClippingVertexType::Nonmortar, xti_type, i, xti_id, poly_vert_ct);
  //        }
  //        else if (poly_status == ClippingState::Unknown)
  //        {
  //           // unknown if we are inside or outside yet. add point for now, but don't change poly_status
  //           CheckAndAddVertex(&x1t[i*3], xti, ClippingVertexType::Nonmortar, xti_type, i, xti_id, poly_vert_ct);
  //        }
  //     }
  //     else // vertex is outside
  //     {
  //        SLIC_ERROR_IF(poly_status == ClippingState::Inside, "Potentially missed edge/edge intersection.");
  //        if (poly_status == ClippingState::Unknown)
  //        {
  //           // we know we are outside now. reset poly_vert_ct so existing unknown points are removed.
  //           poly_vert_ct = 0;
  //        }
  //        poly_status = ClippingState::Outside;
  //     }
  //     int vert1 = i;
  //     int vert2 = (i+1) % size1;
  //     // p = vector along edge
  //     axom::primal::Vector<RealT, 3> p({
  //        x1t[vert2*3 + 0] - x1t[vert1*3 + 0],
  //        x1t[vert2*3 + 1] - x1t[vert1*3 + 1],
  //        x1t[vert2*3 + 2] - x1t[vert1*3 + 2]
  //     });
  //     RealT pnorm_inv = 1.0 / p.norm();
  //     // t_vals = parameter values of the intersection point (initialized to -1.0)
  //     RealT t_vals[4] = {-1.0, -1.0, -1.0, -1.0};
  //     int edge_ids[4] = {-1, -1, -1, -1};
  //     int edge_ct = 0;
  //     // get intersection points of nonmortar edge and mortar element
  //     for (int j{0}; j < size2; ++j)
  //     {
  //        // denominator of intersection point calc
  //        RealT den = p.dot(n2_in[j]);
  //        // den_norm = normalized denominator
  //        RealT den_norm = den * pnorm_inv;
  //        if (std::abs(den_norm) > angle_tol)
  //        {
  //           // edges are not aligned, find intersection point
  //           // x1_to_edge = vector from edge vert1 to elem1 vertex
  //           axom::primal::Vector<RealT, 3> x1_to_edge({
  //              x2t[j*3 + 0] - x1t[i*3 + 0],
  //              x2t[j*3 + 1] - x1t[i*3 + 1],
  //              x2t[j*3 + 2] - x1t[i*3 + 2]
  //           });
  //           RealT t_val = x1_to_edge.dot(n2_in[j]) / den;
  //           if (t_val < (1.0 + dist_tol*pnorm_inv) && t_val > -dist_tol*pnorm_inv)
  //           {
  //              // check for duplicated t values
  //              bool repeated_t_val = false;
  //              for (int k{0}; k < edge_ct; ++k)
  //              {
  //                 if (abs(t_val - t_vals[k]) < dist_tol*pnorm_inv)
  //                 {
  //                    repeated_t_val = true;
  //                    break;
  //                 }
  //              }
  //              if (!repeated_t_val)
  //              {
  //                 SLIC_ERROR_IF(edge_ct == 2, "More than 2 edge intersections. Are the projected polygons convex?");
  //                 // intersection is over line segment
  //                 t_vals[edge_ct] = t_val;
  //                 edge_ids[edge_ct] = j;
  //                 ++edge_ct;
  //              }
  //           }
  //        }
  //     }
  //     // sort the t_vals, smallest is closest to the vertex
  //     // max number of unique intersections should be 2
  //     if (edge_ct == 2 && t_vals[1] < t_vals[0])
  //     {
  //        t_vals[2] = t_vals[0]
  //     }
  //  }
  //  // all elem1 coords are outside elem2 and there are no edge/edge
  //  // intersections. is elem2 inside elem1?

  //  // all elem1 coords are on the boundary of elem2
   // Tribol's clipping algorithm
   // create a local basis
   RealT e1[3] = {
      x1t[0] - x0[0],
      x1t[1] - x0[1],
      x1t[2] - x0[2]
   };
   RealT e1_mag = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
   for (int d{0}; d < 3; ++d)
   {
      e1[d] /= e1_mag;
   }
   RealT e2[3] = {
      n[1]*e1[2] - n[2]*e1[1],
      n[2]*e1[0] - n[0]*e1[2],
      n[0]*e1[1] - n[1]*e1[0]
   };
   RealT x1t_2d[4];
   RealT y1t_2d[4];
   PlaneTo2DCoords(x1t, x0, e1, e2, x1t_2d, y1t_2d, size1);
   RealT x2t_2d[4];
   RealT y2t_2d[4];
   PlaneTo2DCoords(x2t, x0, e1, e2, x2t_2d, y2t_2d, size1);
   PolyReverse(x2t_2d, y2t_2d, size2);
   RealT xti_2d[8];
   RealT yti_2d[8];
   int overlap_poly_size = 0;
   RealT overlap_poly_area = 0.0;
   Intersection2DPolygon(x1t_2d, y1t_2d, size1, x2t_2d, y2t_2d, size2, 
                         1.0e-8, 1.0e-8, xti_2d, yti_2d, overlap_poly_size, overlap_poly_area);
   RealT xti[8*3];
   Coords2DToPlane(xti_2d, yti_2d, x0, e1, e2, xti, overlap_poly_size);

   std::cout << "Overlap coords:" << std::endl;
   for (int i{0}; i < overlap_poly_size; ++i)
   {
      std::cout << xti[i*3+0] << ", " << xti[i*3+1] << ", " << xti[i*3+2] << std::endl;
   }
   
   // some Tribol calls require x, y, z component vectors of projected coords
   RealT x1t_comp[4];
   RealT y1t_comp[4];
   RealT z1t_comp[4];
   for (int i{0}; i < size1; ++i)
   {
      x1t_comp[i] = x1t[i*3 + 0];
      y1t_comp[i] = x1t[i*3 + 1];
      z1t_comp[i] = x1t[i*3 + 2];
   }
   RealT x2t_comp[4];
   RealT y2t_comp[4];
   RealT z2t_comp[4];
   for (int i{0}; i < size2; ++i)
   {
      x2t_comp[i] = x2t[i*3 + 0];
      y2t_comp[i] = x2t[i*3 + 1];
      z2t_comp[i] = x2t[i*3 + 2];
   }

   // Create integration rule over polygon
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
      base_rule_2d[0]  = base_x1;
      base_rule_2d[1]  = base_x1;
      base_rule_2d[2]  = base_x2;
      base_rule_2d[3]  = base_x1;
      base_rule_2d[4]  = base_x1;
      base_rule_2d[5]  = base_x2;
      base_rule_2d[6]  = base_x3;
      base_rule_2d[7]  = base_x4;
      base_rule_2d[8]  = base_x4;
      base_rule_2d[9]  = base_x3;
      base_rule_2d[10] = base_x4;
      base_rule_2d[11] = base_x4;
   }

   // 2. find centroid of the polygon
   RealT xci[3];
   PolyAreaCentroid(xti, 3, overlap_poly_size, xci[0], xci[1], xci[2]);

   // 3. build sub-triangles
   for (int i{0}; i < overlap_poly_size; ++i)
   {
      int idx1 = i;
      int idx2 = (i + 1) % overlap_poly_size;
      RealT vert1[3] = {
         xti[idx1*3 + 0], xti[idx1*3 + 1], xti[idx1*3 + 2]
      };
      RealT vert2[3] = {
         xti[idx2*3 + 0], xti[idx2*3 + 1], xti[idx2*3 + 2]
      };
      RealT side1[3] = {
        vert2[0] - vert1[0], vert2[1] - vert1[1], vert2[2] - vert1[2]
      };
      RealT side2[3] = {
        xci[0] - vert1[0], xci[1] - vert1[1], xci[2] - vert1[2]
      };
      RealT area_vec[3] = {
         side1[1]*side2[2] - side1[2]*side2[1],
         side1[2]*side2[0] - side1[0]*side2[2],
         side1[0]*side2[1] - side1[1]*side2[0]
      };
      RealT area = 0.5*std::sqrt(area_vec[0]*area_vec[0] + area_vec[1]*area_vec[1] + area_vec[2]*area_vec[2]);

      // 4. map integration points and weights to sub-triangle
      for (int j{0}; j < 6; ++j)
      {
         // obtain shape function evaluations at (xi,eta)
         RealT xi[2] = { base_rule_2d[j*2 + 0], base_rule_2d[j*2 + 1] };
         RealT phi[3] = { 0., 0., 0. };
         LinIsoTriShapeFunc( xi[0], xi[1], 0, phi[0] );
         LinIsoTriShapeFunc( xi[0], xi[1], 1, phi[1] );
         LinIsoTriShapeFunc( xi[0], xi[1], 2, phi[2] );

         RealT quad_pt[3];
         for (int d{0}; d < 3; ++d)
         {
            quad_pt[d] = vert1[d] * phi[0] + vert2[d] * phi[1] + xci[d] * phi[2];
         }
         RealT quad_wt = base_weights[j] * area;

         // 5. map sub-triangle point to nonmortar and mortar surfaces
         RealT xi1[2];
         InvIso(quad_pt, x1t_comp, y1t_comp, z1t_comp, size1, xi1);
         RealT xi2[2];
         InvIso(quad_pt, x2t_comp, y2t_comp, z2t_comp, size2, xi2);

         // 6. Evaluate mortar matrix (nonmortar/nonmortar contribs)
         for (int k{0}; k < size1; ++k)
         {
            RealT phiA;
            // NOTE: this limits this routine to quads
            LinIsoQuadShapeFunc(xi1[0], xi1[1], k, phiA);
            for (int l{0}; l < size1; ++l)
            {
              RealT phiB;
              // NOTE: this limits this routine to quads
              LinIsoQuadShapeFunc(xi1[0], xi1[1], l, phiB);
              mortar_mat1[k*size1 + l] += phiA * phiB * quad_wt;
            }
         }

         // 7. Evaluate mortar matrix (nonmortar/mortar contribs)
         for (int k{0}; k < size1; ++k)
         {
            RealT phiA;
            // NOTE: this limits this routine to quads
            LinIsoQuadShapeFunc(xi1[0], xi1[1], k, phiA);
            for (int l{0}; l < size2; ++l)
            {
              RealT phiB;
              // NOTE: this limits this routine to quads
              LinIsoQuadShapeFunc(xi2[0], xi2[1], l, phiB);
              mortar_mat2[k*size2 + l] += phiA * phiB * quad_wt;
            }
         }
      }
   }

   // compute gaps
   for (int i{0}; i < size1; ++i)
   {
      g1[i] = 0.0;
      RealT gap_v[3] = {0.0, 0.0, 0.0};
      for (int j{0}; j < size1; ++j)
      {
         for (int d{0}; d < 3; ++d)
         {
            gap_v[d] += mortar_mat1[i*size1 + j]*x1[j*3 + d];
         }
      }
      for (int j{0}; j < size2; ++j)
      {
         for (int d{0}; d < 3; ++d)
         {
            gap_v[d] -= mortar_mat2[i*size2 + j]*x2[j*3 + d];
         }
      }
      for (int d{0}; d < 3; ++d)
      {
         g1[i] += n1[i*size1 + d] * gap_v[d];
      }
   }

   // compute nonmortar force contributions
   for (int i{0}; i < size1; ++i)
   {
      for (int d{0}; d < 3; ++d)
      {
         f1[i*3 + d] = 0.0;
      }
      for (int j{0}; j < size1; ++j)
      {
         for (int d{0}; d < 3; ++d)
         {
            f1[i*3 + d] += p1[j] * n1[j*3 + d] * mortar_mat1[j*size1 + i];
         }
      }
   }

   // compute mortar force contributions
   for (int i{0}; i < size2; ++i)
   {
      for (int d{0}; d < 3; ++d)
      {
         f2[i*3 + d] = 0.0;
      }
      for (int j{0}; j < size1; ++j)
      {
         for (int d{0}; d < 3; ++d)
         {
            f2[i*3 + d] -= p1[j] * n1[j*3 + d] * mortar_mat2[j*size2 + i];
         }
      }
   }
}

void ComputeMortarJacobianEnzyme( const RealT* x1, const RealT* n1, const RealT* p1,
                                  RealT* f1, RealT* df1dx1, RealT* df1dx2, RealT* df1dp1,
                                  RealT* g1, RealT* dg1dx1, RealT* dg1dx2, int size1,
                                  const RealT* x2,
                                  RealT* f2, RealT* df2dx1, RealT* df2dx2, RealT* df2dp1,
                                  int size2 )
{
   RealT x1_dot[12] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
   __enzyme_fwddiff<void>((void*)ComputeMortarForceEnzyme,
      enzyme_dup, x1, x1_dot,
      enzyme_const, n1,
      enzyme_const, p1,
      enzyme_dup, f1, df1dx1,
      enzyme_dup, g1, dg1dx1,
      enzyme_const, size1,
      enzyme_const, x2,
      enzyme_dup, f2, df2dx1,
      enzyme_const, size2);
   
}
#endif

//------------------------------------------------------------------------------
template< >
int GetMethodData< MORTAR_WEIGHTS >( CouplingScheme const * cs )
{
   ////////////////////////////////
   //                            //
   // compute single mortar gaps //
   //                            //
   ////////////////////////////////
   ComputeSingleMortarGaps( cs );
   
   auto pairs = cs->getInterfacePairsView();
   IndexT const numPairs = pairs.size();
   auto planes = cs->get3DContactPlanesView();

   const int dim = cs->spatialDimension();

   auto& mortarMesh = cs->getMesh1();
   auto& nonmortarMesh = cs->getMesh2();
   IndexT const numNodesPerFace = mortarMesh.numberOfNodesPerElement();

   int numRows = cs->getNumTotalNodes();
   static_cast<MortarData*>( cs->getMethodData() )->allocateMfemSparseMatrix( numRows );

   //////////////////////////////////////////////
   //                                          //
   // aggregate data to compute mortar weights //
   //                                          //
   //////////////////////////////////////////////

   int cpID = 0;
   for (IndexT kp = 0; kp < numPairs; ++kp)
   {
      InterfacePair pair = pairs[kp];

      if (!pair.m_is_contact_candidate)
      {
         continue;
      }

      auto& plane = planes[cpID];

      // get pair indices
      IndexT index1 = pair.m_element_id1;
      IndexT index2 = pair.m_element_id2;

      // get projected face coordinates
      ArrayT<RealT, 2> mortarX_bar(numNodesPerFace, dim);
      ArrayT<RealT, 2> nonmortarX_bar(numNodesPerFace, dim);
      ArrayT<RealT, 2> mortarX_barT(dim, numNodesPerFace);
      ArrayT<RealT, 2> nonmortarX_barT(dim, numNodesPerFace);
      ProjectFaceNodesToPlane( mortarMesh, index1, 
                               plane.m_nX, plane.m_nY, plane.m_nZ,
                               plane.m_cX, plane.m_cY, plane.m_cZ,
                               &mortarX_barT(0, 0), 
                               &mortarX_barT(1, 0), 
                               &mortarX_barT(2, 0) );
      ProjectFaceNodesToPlane( nonmortarMesh, index2, 
                               plane.m_nX, plane.m_nY, plane.m_nZ,
                               plane.m_cX, plane.m_cY, plane.m_cZ,
                               &nonmortarX_barT(0, 0), 
                               &nonmortarX_barT(1, 0), 
                               &nonmortarX_barT(2, 0) );
      algorithm::transpose<MemorySpace::Dynamic>(mortarX_barT, mortarX_bar);
      algorithm::transpose<MemorySpace::Dynamic>(nonmortarX_barT, nonmortarX_bar);

      // construct array of polygon overlap vertex coordinates
      ArrayT<RealT, 2> overlapX(plane.m_numPolyVert, dim);
      for (IndexT i{0}; i < plane.m_numPolyVert; ++i)
      {
        overlapX(i, 0) = plane.m_polyX[i];
        overlapX(i, 1) = plane.m_polyY[i];
        overlapX(i, 2) = plane.m_polyZ[i];
      }

      // instantiate contact surface element for purposes of computing 
      // mortar weights. Note, this uses projected face coords
      SurfaceContactElem elem( dim, mortarX_bar.data(), nonmortarX_bar.data(), 
                               overlapX.data(),
                               numNodesPerFace, 
                               plane.m_numPolyVert,
                               &mortarMesh, &nonmortarMesh, index1, index2 );

      // compute the mortar weights to be stored on the surface 
      // contact element struct. This must be done prior to computing nodal gaps
      elem.overlapArea = plane.m_area;

      ComputeMortarWeights( elem );

      elem.numActiveGaps = numNodesPerFace;

      // assemble mortar weight contributions sum_alpha int_alpha phi_a phi_b da.
      // Note: active nonmortar nodes (i.e. active gaps) are checked in this routine.
      const EnforcementOptions& enforcement_options = const_cast<EnforcementOptions&>(cs->getEnforcementOptions());
      const SparseMode sparse_mode = enforcement_options.lm_implicit_options.sparse_mode;
      if (sparse_mode == SparseMode::MFEM_ELEMENT_DENSE)
      {
        SLIC_WARNING( "GetMethodData<MORTAR_WEIGHTS>() MFEM_ELEMENT_DENSE " << 
                      "Unassembled element dense matrix output not implemented." );
        return 1;
      }
      static_cast<MortarData*>( cs->getMethodData() )->assembleMortarWts( elem, sparse_mode );

      ++cpID;

   } // end loop over pairs to assemble mortar weights

   return 0;

} // end GetMethodData< MORTAR_WEIGHTS >()

//------------------------------------------------------------------------------

} // end namespace tribol
