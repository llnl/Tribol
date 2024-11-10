// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "AlignedMortar.hpp"

#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/geom/ContactPlane.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/integ/Integration.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/utils/ContactPlaneOutput.hpp"
#include "tribol/utils/Algorithm.hpp"
#include "tribol/utils/Math.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>

namespace tribol
{

void ComputeAlignedMortarWeights( SurfaceContactElem & elem )
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

   RealT phiNonmortarA, phiNonmortarB, phiMortarA;

   // loop over nodes "a", where node "a" can be a nonmortar node or a mortar node
   for (int a=0; a<elem.numFaceVert; ++a)
   {
      // loop over nonmortar nodes
      for (int b=0; b<elem.numFaceVert; ++b)
      {

         // loop over number of integration points
         for (int ip=0; ip<integ.numIPs; ++ip)
         {

            RealT xi[2]; 
            xi[0] = integ.xy[ integ.ipDim * ip ];
            xi[1] = integ.xy[ integ.ipDim * ip + 1 ];
            LinIsoQuadShapeFunc( xi[0], xi[1], a, phiMortarA );
            LinIsoQuadShapeFunc( xi[0], xi[1], a, phiNonmortarA );
            LinIsoQuadShapeFunc( xi[0], xi[1], b, phiNonmortarB );

            // set nonmortar/nonmortar and nonmortar/mortar ids
            int nonmortarNonmortarId = elem.numFaceVert * a + b;
            int mortarNonmortarId = elem.numFaceVert * elem.numFaceVert + 
                                elem.numFaceVert * a + b;
 
            SLIC_ERROR_IF(nonmortarNonmortarId > elem.numWts || mortarNonmortarId > elem.numWts,
                          "ComputeAlignedMortarWeights: integer ids for weights exceed elem.numWts");

            // compute nonmortar/nonmortar mortar weight
            elem.mortarWts[ nonmortarNonmortarId ]  += integ.wts[ip] * phiNonmortarA * phiNonmortarB;
 
            // compute nonmortar/mortar mortar weight
            elem.mortarWts[ mortarNonmortarId ] += integ.wts[ip] * phiMortarA * phiNonmortarB;
 
         } // end loop over integration points
      } // end loop over nodes on side 2
   } // end loop over nodes on side 1
   
} // end ComputeAlignedMortarWeights()

//------------------------------------------------------------------------------
template< >
void ComputeNodalGap< ALIGNED_MORTAR >( SurfaceContactElem & elem )
{
   // get pointer to mesh view to store gaps on mesh data object
   auto& nonmortarMesh = *elem.m_mesh2;
   const IndexT * const nonmortarConn = nonmortarMesh.getConnectivity().data();

   SLIC_ERROR_IF(nonmortarMesh.getNodalFields().m_node_gap.empty(), 
                 "ComputeNodalGap< ALIGNED_MORTAR >: allocate gaps on mesh data object.");

   // compute gap contributions associated with face 2 on the SurfaceContactElem 
   // (i.e. nonmortar surface)

   // set the distance magnitude tolerance as the longest edge of 
   // the mortar face
   RealT magTol;
   RealT magTest = 0.;
   for (int k=0; k<elem.numFaceVert; ++k)
   {
      int idPlus = (k == (elem.numFaceVert-1)) ? 0 : k+1;
      RealT dx = elem.faceCoords1[ elem.dim * idPlus ] - elem.faceCoords1[ elem.dim * k ];
      RealT dy = elem.faceCoords1[ elem.dim * idPlus + 1 ] - elem.faceCoords1[ elem.dim * k + 1 ];
      RealT dz = elem.faceCoords1[ elem.dim * idPlus + 2 ] - elem.faceCoords1[ elem.dim * k + 2 ];

      RealT mag = magnitude( dx, dy, dz );

      magTol = (mag > magTest) ? mag : magTest;
      magTest = mag;
   }

   // loop over nodes on nonmortar side
   for (int a=0; a<elem.numFaceVert; ++a)
   {

      // get global nonmortar node number from connectivity
      RealT nrml_a[elem.dim];
      int glbId = nonmortarConn[ elem.numFaceVert * elem.faceId2 + a ];
      nrml_a[0] = nonmortarMesh.getNodalNormals()[0][ glbId ];
      nrml_a[1] = nonmortarMesh.getNodalNormals()[1][ glbId ];
      if (elem.dim == 3 )
      {
         nrml_a[2] = nonmortarMesh.getNodalNormals()[2][ glbId ];
      }

      //////////////////////////////////////////////
      // determine which mortar node is aligned with 
      // nonmortar node "a" 
      //////////////////////////////////////////////
      int mortarNodeId;
      RealT v[3] = {0., 0., 0.};
      RealT magTest = magTol; 
      // loop over nodes on the mortar side 
      for (int b=0; b<elem.numFaceVert; ++b)
      {
         v[0] = elem.faceCoords1[ elem.dim * b ] - elem.faceCoords2[ elem.dim * a ];
         v[1] = elem.faceCoords1[ elem.dim * b + 1 ] - elem.faceCoords2[ elem.dim * a + 1 ];
         v[2] = elem.faceCoords1[ elem.dim * b + 2 ] - elem.faceCoords2[ elem.dim * a + 2 ];

         RealT magV = magnitude( v[0], v[1], v[2] );

         if (magV < magTest)
         {
            mortarNodeId = b;
            magTest = magV;
         }
      }
   
      // store local gap
      v[0] = elem.faceCoords1[ elem.dim * mortarNodeId ] - elem.faceCoords2[ elem.dim * a ] ;
      v[1] = elem.faceCoords1[ elem.dim * mortarNodeId + 1 ] - elem.faceCoords2[ elem.dim * a + 1 ] ;
      v[2] = elem.faceCoords1[ elem.dim * mortarNodeId + 2 ] - elem.faceCoords2[ elem.dim * a + 2 ] ;

      
      nonmortarMesh.getNodalFields().m_node_gap[ glbId ] += dotProd( &v[0], &nrml_a[0], elem.dim );

   }

} // end of ComputeNodalGap<>()

//------------------------------------------------------------------------------
void ComputeAlignedMortarGaps( CouplingScheme* cs )
{
   MeshManager& meshManager = MeshManager::getInstance();
   MeshData& nonmortarMeshData = meshManager.at( cs->getMeshId2() );
   int const dim = cs->spatialDimension();
   // compute nodal normals (do this outside the element loop)
   // This routine is guarded against a null mesh
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

   const IndexT numNodesPerFace = mortarMesh.numberOfNodesPerElement();

   RealT const * const x1 = mortarMesh.getPosition()[0].data();
   RealT const * const y1 = mortarMesh.getPosition()[1].data();
   RealT const * const z1 = mortarMesh.getPosition()[2].data();
   const IndexT * const mortarConn= mortarMesh.getConnectivity().data();

   RealT const * const x2 = nonmortarMesh.getPosition()[0].data();
   RealT const * const y2 = nonmortarMesh.getPosition()[1].data();
   RealT const * const z2 = nonmortarMesh.getPosition()[2].data();
   const IndexT * nonmortarConn = nonmortarMesh.getConnectivity().data();

 
   // declare local variables to hold face nodal coordinates
   // and overlap vertex coordinates
   RealT mortarX[ dim * numNodesPerFace ];
   RealT nonmortarX[ dim * numNodesPerFace ];

   ////////////////////////////
   // compute nonmortar gaps //
   ////////////////////////////
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

      // populate nodal coordinates array. For the aligned mortar 
      // gap equations, the nodal coordinates are NOT to be projected 
      // onto the common plane, since the aligned mortar gap 
      // calculation uses the current configuration nodal coordinates 
      // themselves
      for (int i=0; i<numNodesPerFace; ++i)
      {
         int id = dim * i;
         mortarX[ id ]   = x1[ mortarConn[ numNodesPerFace * index1 + i ] ];
         mortarX[ id+1 ] = y1[ mortarConn[ numNodesPerFace * index1 + i ] ];
         mortarX[ id+2 ] = z1[ mortarConn[ numNodesPerFace * index1 + i ] ];
         nonmortarX[ id ]   = x2[ nonmortarConn[ numNodesPerFace * index2 + i ] ];
         nonmortarX[ id+1 ] = y2[ nonmortarConn[ numNodesPerFace * index2 + i ] ];
         nonmortarX[ id+2 ] = z2[ nonmortarConn[ numNodesPerFace * index2 + i ] ];
      }

      // construct array of polygon overlap vertex coordinates
      ArrayT<RealT, 2> overlapX(plane.m_numPolyVert, dim);
      for (IndexT i{0}; i < plane.m_numPolyVert; ++i)
      {
        overlapX(i, 0) = plane.m_polyX[i];
        overlapX(i, 1) = plane.m_polyY[i];
        overlapX(i, 2) = plane.m_polyZ[i];
      }

      // instantiate SurfaceContactElem struct. Note, this is done with 
      // the projected area of overlap, but with the actual current 
      // configuration face coordinates. We need the current 
      // configuration face coordinates here in order to correctly 
      // compute the mortar gaps.
      SurfaceContactElem elem_for_gap( dim, mortarX, nonmortarX, 
                                       overlapX.data(),
                                       numNodesPerFace, 
                                       plane.m_numPolyVert,
                                       &mortarMesh, &nonmortarMesh, index1, index2 );

      /////////////////////////
      // compute mortar gaps //
      /////////////////////////
      ComputeNodalGap< ALIGNED_MORTAR >( elem_for_gap );

      // HAVE TO set the number of active constraints. For now set to 
      // all nonmortar face nodes.
      elem_for_gap.numActiveGaps = numNodesPerFace;

      ++cpID;

   } // end loop over pairs for nonmortar gap calculations

} // end ComputeAlignedMortarGaps()

//------------------------------------------------------------------------------
template< >
int ApplyNormal< ALIGNED_MORTAR, LAGRANGE_MULTIPLIER >( CouplingScheme* cs )
{
   ///////////////////////////////////////////////////////
   //                                                   //
   //            compute single mortar gaps             //
   //                                                   //
   // Note, this routine is guarded against a null mesh //
   ///////////////////////////////////////////////////////
   ComputeAlignedMortarGaps( cs );
   
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

   const IndexT numNodesPerFace = mortarMesh.numberOfNodesPerElement();

   RealT * const fx1 = mortarMesh.getResponse()[0].data();
   RealT * const fy1 = mortarMesh.getResponse()[1].data();
   RealT * const fz1 = mortarMesh.getResponse()[2].data();
   const IndexT * const mortarConn= mortarMesh.getConnectivity().data();

   RealT * const fx2 = nonmortarMesh.getResponse()[0].data();
   RealT * const fy2 = nonmortarMesh.getResponse()[1].data();
   RealT * const fz2 = nonmortarMesh.getResponse()[2].data();
   const IndexT * nonmortarConn = nonmortarMesh.getConnectivity().data();

   int numTotalNodes;
   int numRows;
   if (!cs->nullMeshes())
   {
      numTotalNodes = cs->getNumTotalNodes();
      numRows = dim * numTotalNodes + numTotalNodes;
      static_cast<MortarData*>( cs->getMethodData() )->allocateMfemSparseMatrix( numRows );
   }

   ////////////////////////////////////////////////////////////////
   // compute equilibrium residual and/or Jacobian contributions //
   ////////////////////////////////////////////////////////////////
   int cpID = 0;
   for (IndexT kp = 0; kp < numPairs; ++kp)
   {
      auto& pair = pairs[kp];

      if (!pair.m_is_contact_candidate)
      {
         continue;
      }

      // get pair indices
      IndexT index1 = pair.m_element_id1;
      IndexT index2 = pair.m_element_id2;

      //////////////////////////////////
      // compute equilibrium residual //
      //////////////////////////////////

      // loop over face nodes (BOTH MORTAR and NONMORTAR 
      // contributions). NOTE: mortar weights aren't required for 
      // the residual contributions for ALIGNED_MORTAR. As a result, 
      // a SurfaceContactElem is not required
      for (int a=0; a<numNodesPerFace; ++a)
      {
         int mortarIdA = mortarConn[ index1 * numNodesPerFace + a];
         int nonmortarIdA = nonmortarConn[ index2 * numNodesPerFace + a ];

         // Note, we assemble residual for all nonmortar nodes, even if the gap 
         // is in separation. NOTE: Per testing, we 
         // assemble ALL nonmortar node contributions for faces with positive 
         // areas of overlap that have passed the geometric filtering. We judge 
         // contact activity by gaps AND the pressure solution.

         // note: we don't have to interpolate the nodal pressure for mortar and nonmortar 
         // sides for the aligned mortar case (i.e. no interpolation necessary for coincident 
         // mortar and nonmortar nodes).
         RealT forceX = nonmortarMesh.getNodalFields().m_node_pressure[ nonmortarIdA ] * 
                       nonmortarMesh.getNodalNormals()[0][ nonmortarIdA ];
         RealT forceY = nonmortarMesh.getNodalFields().m_node_pressure[ nonmortarIdA ] *
                       nonmortarMesh.getNodalNormals()[1][ nonmortarIdA ];
         RealT forceZ = nonmortarMesh.getNodalFields().m_node_pressure[ nonmortarIdA ] * 
                       nonmortarMesh.getNodalNormals()[2][ nonmortarIdA ];

         fx2[ nonmortarIdA ]  -= forceX;
         fy2[ nonmortarIdA ]  -= forceY;
         fz2[ nonmortarIdA ]  -= forceZ;

         fx1[ mortarIdA ] += forceX;
         fy1[ mortarIdA ] += forceY;
         fz1[ mortarIdA ] += forceZ;

      } // end outer loop over nonmortar and mortar nodes

      /////////////////////////////////////////////////////////////
      // compute tangent stiffness contributions; Note, the      //
      // Jacobian of the weak form contact nodal force term      //
      // with respect to the primal kinematic field is 0. There  //
      // are only contributions from the Jacobian of the contact //
      // nodal force term wrt nodal pressures and the Jacobian   // 
      // of the gap constraints wrt the primal kinematic field   //
      /////////////////////////////////////////////////////////////

      const EnforcementOptions& enforcement_options = const_cast<EnforcementOptions&>(cs->getEnforcementOptions());
      const LagrangeMultiplierImplicitOptions& lm_options = enforcement_options.lm_implicit_options;
      if ( lm_options.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN ||
           lm_options.eval_mode == ImplicitEvalMode::MORTAR_JACOBIAN )
      {
         auto& plane = planes[cpID];

         // stores projected coordinates in row-major format
         ArrayT<RealT, 2> mortarX(numNodesPerFace, dim);
         ArrayT<RealT, 2> nonmortarX(numNodesPerFace, dim);
         // stores projected coordinates in column-major format
         ArrayT<RealT, 2> mortarXT(dim, numNodesPerFace);
         ArrayT<RealT, 2> nonmortarXT(dim, numNodesPerFace);
         ProjectFaceNodesToPlane( mortarMesh, index1, 
                                  plane.m_nX, plane.m_nY, plane.m_nZ,
                                  plane.m_cX, plane.m_cY, plane.m_cZ,
                                  &mortarXT(0, 0), 
                                  &mortarXT(1, 0), 
                                  &mortarXT(2, 0) );
         ProjectFaceNodesToPlane( nonmortarMesh, index2, 
                                  plane.m_nX, plane.m_nY, plane.m_nZ,
                                  plane.m_cX, plane.m_cY, plane.m_cZ,
                                  &nonmortarXT(0, 0), 
                                  &nonmortarXT(1, 0), 
                                  &nonmortarXT(2, 0) );
         // populate row-major projected coordinates for the purpose of sending to
         // the SurfaceContactElem struct
         algorithm::transpose<MemorySpace::Dynamic>(mortarXT, mortarX);
         algorithm::transpose<MemorySpace::Dynamic>(nonmortarXT, nonmortarX);

         // construct array of polygon overlap vertex coordinates
         ArrayT<RealT, 2> overlapX(plane.m_numPolyVert, dim);
         for (IndexT i{0}; i < plane.m_numPolyVert; ++i)
         {
            overlapX(i, 0) = plane.m_polyX[i];
            overlapX(i, 1) = plane.m_polyY[i];
            overlapX(i, 2) = plane.m_polyZ[i];
         }

         // instantiate a new surface contact element with projected face 
         // coordinates
         SurfaceContactElem elem_for_jac( dim, mortarX.data(), nonmortarX.data(), 
                                          overlapX.data(),
                                          numNodesPerFace, 
                                          plane.m_numPolyVert,
                                          &mortarMesh, &nonmortarMesh, index1, index2 );

         // HAVE TO set the number of active constraints. For now set to 
         // all nonmortar face nodes.
         elem_for_jac.numActiveGaps = numNodesPerFace; 

         // we need mortar weights for Jacobian contribution calculations
         ComputeAlignedMortarWeights( elem_for_jac );
         ComputeAlignedMortarJacobian( elem_for_jac );
         static_cast<MortarData*>( cs->getMethodData() )->assembleJacobian( elem_for_jac, lm_options.sparse_mode );

         ++cpID;

      } // end if-jacobian related output

   } // end of loop over interface pairs computing residual/Jacobian contributions

   return 0;

} // end ApplyNormal<>()

//------------------------------------------------------------------------------
template< >
void ComputeResidualJacobian< ALIGNED_MORTAR, PRIMAL >( SurfaceContactElem & TRIBOL_UNUSED_PARAM(elem) )
{
   // There is no Jacobian contribution for this block. Be safe and zero out...
   return;
}

//------------------------------------------------------------------------------
template< >
void ComputeResidualJacobian< ALIGNED_MORTAR, DUAL >( SurfaceContactElem & elem )
{
   ComputeResidualJacobian< SINGLE_MORTAR, DUAL >( elem );
} // end ComputeResidualJacobian<>()

//------------------------------------------------------------------------------
template< >
void ComputeConstraintJacobian< ALIGNED_MORTAR, PRIMAL >( SurfaceContactElem & elem )
{
   ComputeConstraintJacobian< SINGLE_MORTAR, PRIMAL >( elem );
} // end ComputeConstraintJacobian

//------------------------------------------------------------------------------
template< >
void ComputeConstraintJacobian< ALIGNED_MORTAR, DUAL >( SurfaceContactElem & TRIBOL_UNUSED_PARAM(elem) )
{
   // unless we end up solving the complementarity equation, there is 
   // no Jacobian contribtion for this block. Zero out to be safe...
   return;
}

//------------------------------------------------------------------------------
void ComputeAlignedMortarJacobian( SurfaceContactElem & elem )
{
   elem.allocateBlockJ( LAGRANGE_MULTIPLIER );

   ComputeResidualJacobian  < ALIGNED_MORTAR, PRIMAL >( elem );

   ComputeResidualJacobian  < ALIGNED_MORTAR, DUAL   >( elem );

   ComputeConstraintJacobian< ALIGNED_MORTAR, PRIMAL >( elem );

   ComputeConstraintJacobian< ALIGNED_MORTAR, DUAL   >( elem );

   return;

}

//------------------------------------------------------------------------------

} // end namespace Tribol
