// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "ContactPlane.hpp"
#include "GeomUtilities.hpp"
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/utils/Math.hpp"

#include "axom/core.hpp" 
#include "axom/slic.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <fstream>

namespace tribol
{

//------------------------------------------------------------------------------
// free functions
//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError CheckInterfacePair( InterfacePair& pair,
                                                     const MeshData::Viewer& mesh1,
                                                     const MeshData::Viewer& mesh2,
                                                     const Parameters& params,
                                                     ContactMethod cMethod,
                                                     ContactCase TRIBOL_UNUSED_PARAM(cCase),
                                                     bool& isInteracting,
                                                     ArrayViewT<ContactPlane2D>& planes_2d,
                                                     ArrayViewT<ContactPlane3D>& planes_3d,
                                                     IndexT* plane_ct )
{
  isInteracting = false;

  // note: will likely need the ContactCase for specialized 
  // geometry check(s)/routine(s)

  switch (cMethod)
  {
    case SINGLE_MORTAR:
    case MORTAR_WEIGHTS:
    case COMMON_PLANE:
    {
      // set whether full overlap is to be used or not. Note: SINGLE_MORTAR and 
      // MORTAR_WEIGHTS drop into this 'case', so the method still needs to be checked
      const bool full = (cMethod == COMMON_PLANE) ? false : true;
      const bool interpenOverlap = (full) ? false : true;
      const bool intermediatePlane = (cMethod == COMMON_PLANE) ? true : false;

      // Perform contact plane specific computations (2D and 3D)
      if (mesh1.spatialDimension() == 3)
      {

        ContactPlane3D cpTemp( &pair, params.overlap_area_frac, interpenOverlap, intermediatePlane);
        FaceGeomError face_err = CheckFacePair( cpTemp, mesh1, mesh2, params, full );

#ifdef TRIBOL_USE_HOST
        SLIC_DEBUG("face_err: " << face_err );
#endif

        if (face_err != NO_FACE_GEOM_ERROR)
        {
          isInteracting = false;
        }
        else if (cpTemp.m_inContact)
        {
#ifdef TRIBOL_USE_RAJA
          auto idx = RAJA::atomicInc<RAJA::auto_atomic>(plane_ct);
#else
          auto idx = *plane_ct;
          ++(*plane_ct);
#endif
          planes_3d[idx] = std::move(cpTemp);
          isInteracting = true;
        }
        else
        {
          isInteracting = false;
        }
        return face_err;
      }
      else 
      {
        ContactPlane2D cpTemp( &pair, params.overlap_area_frac, interpenOverlap, intermediatePlane);
        FaceGeomError edge_err = CheckEdgePair( cpTemp, mesh1, mesh2, params, full );

        if (edge_err != NO_FACE_GEOM_ERROR)
        {
          isInteracting = false;
        }
        else if (cpTemp.m_inContact)
        {
#ifdef TRIBOL_USE_RAJA
          auto idx = RAJA::atomicInc<RAJA::auto_atomic>(plane_ct);
#else
          auto idx = *plane_ct;
          ++(*plane_ct);
#endif
          planes_2d[idx] = std::move(cpTemp);
          isInteracting = true;
        }
        else
        {
          isInteracting = false;
        }
        return edge_err;
      }
      break;
    }

    case ALIGNED_MORTAR:
    {
      // Note: this is checked by CouplingScheme::isValidMethod()
      if (mesh1.spatialDimension() == 3)
      {
        // TODO refactor to make consistent with CheckFacePair, SRW
        ContactPlane3D cpTemp = CheckAlignedFacePair( pair, mesh1, mesh2, params );

        if (cpTemp.m_inContact)
        {
#ifdef TRIBOL_USE_RAJA
          auto idx = RAJA::atomicInc<RAJA::auto_atomic>(plane_ct);
#else
          auto idx = (*plane_ct);
          ++(*plane_ct);
#endif
          planes_3d[idx] = std::move(cpTemp);
          isInteracting = true;
        }
        else
        {
          isInteracting = false;
        }
        return NO_FACE_GEOM_ERROR;
      }
      else
      {
        // Note: this is checked by CouplingScheme::isValidMethod()
        // SLIC_ERROR_IF(true, "2D ALIGNED_MORTAR not yet implemented.");
      }
      break;
    }

    default:
    {
      // don't do anything
      break;
    }
  }

  return NO_FACE_GEOM_ERROR; // quiet compiler

} // end CheckInterfacePair()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE bool FaceInterCheck( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2, 
                     int fId1, int fId2, RealT tol, bool& allVerts )
{
   bool check = false;
   allVerts = false;

   // loop over vertices on face 2
   int k = 0;
   for (int i=0; i<mesh2.numberOfNodesPerElement(); ++i) 
   {
      // get ith face 2 node id
      const int f2NodeId = mesh2.getGlobalNodeId(fId2, i);

      // compute components of vector between face 1 center and face 2 vertex
      RealT vX = mesh1.getElementCentroids()[0][fId1] - mesh2.getPosition()[0][f2NodeId];
      RealT vY = mesh1.getElementCentroids()[1][fId1] - mesh2.getPosition()[1][f2NodeId];
      RealT vZ = mesh1.getElementCentroids()[2][fId1] - mesh2.getPosition()[2][f2NodeId];

      // project the vector onto face 1 normal
      RealT proj = vX * mesh1.getElementNormals()[0][fId1] + vY * mesh1.getElementNormals()[1][fId1] 
                + vZ * mesh1.getElementNormals()[2][fId1];

      // if a node of face 2 is on the other side of the plane defined by face 1 the 
      // projection will be positive. If a node on face 2 lies on face 1 the projection 
      // will be zero. If a node lies just outside of face 1 then the projection will 
      // be a small negative number.
      if (proj > -tol)
      {
         check = true;
         ++k;
      }

   } // end loop over nodes

   // check to see if all nodes are on the other side
   if (k == mesh2.numberOfNodesPerElement())
   {
      allVerts = true;
   }

   // at this point, if the check is false then there is deemed no interaction. The 
   // faces are on the non-contacting sides of one another. Return.
   if (check == false) 
   {
      return check;
   }

   // Added 10/19/18 by SRW to catch the case where one face is entirely on the other 
   // side of the other (for which ordering of face 1 vs. 2 in this routine matters) 
   // and may not cross the contact plane (adjusted later outside of this routine). 
   // For the cases where check == true, we know that some of the vertices of face 
   // 1 pass through the plane defined by face 2. We want to know if they all do and 
   // trigger the allVerts boolean, which will later trigger the correct full 
   // overlap computation

   // loop over vertices on face 1
   k = 0;
   for (int i=0; i<mesh1.numberOfNodesPerElement(); ++i) 
   {
      // get ith face 1 node id
      const int f1NodeId = mesh1.getGlobalNodeId(fId1, i);
  
      // compute the components of vector between face 2 center and face 1 vertex
      RealT vX = mesh2.getElementCentroids()[0][fId2] - mesh1.getPosition()[0][f1NodeId];
      RealT vY = mesh2.getElementCentroids()[1][fId2] - mesh1.getPosition()[1][f1NodeId];
      RealT vZ = mesh2.getElementCentroids()[2][fId2] - mesh1.getPosition()[2][f1NodeId];

      // project the vector onto face 2 normal
      RealT proj = vX * mesh2.getElementNormals()[0][fId2] + vY * mesh2.getElementNormals()[1][fId2] 
                + vZ * mesh2.getElementNormals()[2][fId2];

      // if a node of face 1 is on the other side of the plane defined by face 2 the 
      // projection will be positive. If a node on face 1 lies on face 2 the projection 
      // will be zero. If a node lies just outside of face 2 then the projection will 
      // be a small negative number.
      if (proj > -tol) {
         check = true; // check will be true from the first loop
         ++k;
      }

   } // end loop over nodes

   // check to see if all nodes are on the other side
   if (k == mesh1.numberOfNodesPerElement()) 
   {
      allVerts = true;
   }

   return check;

} // end FaceInterCheck()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE bool EdgeInterCheck( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2, 
                     int eId1, int eId2, RealT tol, bool& allVerts )
{
   bool check = false;
   allVerts = false;

   // loop over vertices on edge 2
   int k = 0;
   for (int i=0; i<mesh2.numberOfNodesPerElement(); ++i)
   {
      // get edge 2 ith vertex id
      const int e2vId = mesh2.getGlobalNodeId( eId2, i );
   
      // compute components of vector between edge 1 center and edge 2 vertex
      RealT vX = mesh1.getElementCentroids()[0][eId1] - mesh2.getPosition()[0][e2vId];
      RealT vY = mesh1.getElementCentroids()[1][eId1] - mesh2.getPosition()[1][e2vId];

      // project the vector onto edge1 normal
      RealT proj = vX * mesh1.getElementNormals()[0][eId1] + vY * mesh1.getElementNormals()[1][eId1];

      // check projection against tolerance
      if (proj > -tol)
      {
         check = true;
         ++k;
      }
   } // end loop over edge2 vertices

   // check to see if all vertices are on the other side
   if (k == mesh2.numberOfNodesPerElement()) allVerts = true;

   if (check == false) return check;

   // loop over vertices on edge 1 to catch the case where edge 1 lies
   // entirely on the other side of edge 2 triggering a full overlap 
   // computation
   k = 0;
   for (int i=0; i<mesh1.numberOfNodesPerElement(); ++i)
   {
      // get edge 1 ith vertex id
      const int e1vId = mesh1.getGlobalNodeId( eId1, i );
 
      // compute components of vector between edge 2 center and edge 1 vertex
      RealT vX = mesh2.getElementCentroids()[0][eId2] - mesh1.getPosition()[0][e1vId];
      RealT vY = mesh2.getElementCentroids()[1][eId2] - mesh1.getPosition()[1][e1vId];

      // project the vector onto edge2 normal
      RealT proj = vX * mesh2.getElementNormals()[0][eId2] + vY * mesh2.getElementNormals()[1][eId2];

      // check projection against tolerance
      if (proj > -tol)
      {
         check = true; // check will be true from the first loop
         ++k;
      }
   } // end loop over edge1 vertices

   // check to see if all vertices are on the other side
   if (k == mesh1.numberOfNodesPerElement()) allVerts = true;

   return check;

} // end EdgeInterCheck()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE bool ExceedsMaxAutoInterpen( const MeshData::Viewer& mesh1,
                                                const MeshData::Viewer& mesh2,
                                                const int faceId1,
                                                const int faceId2,
                                                const Parameters& params,
                                                const RealT gap )
{
   if (params.auto_contact_check)
   {
      RealT max_interpen = -1. * params.auto_contact_pen_frac * 
                      axom::utilities::min( mesh1.getElementData().m_thickness[ faceId1 ],
                                            mesh2.getElementData().m_thickness[ faceId2 ] );
      if (gap < max_interpen)
      {
        return true;
      }
   }
   return false;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ProjectFaceNodesToPlane( const MeshData::Viewer& mesh, int faceId, 
                                                 RealT nrmlX, RealT nrmlY, RealT nrmlZ,
                                                 RealT cX, RealT cY, RealT cZ,
                                                 RealT* pX, RealT* pY, RealT* pZ )
{

   // loop over nodes and project onto the plane defined by the point-normal 
   // input arguments
   for (int i=0; i<mesh.numberOfNodesPerElement(); ++i) {
      const int nodeId = mesh.getGlobalNodeId(faceId, i);
      ProjectPointToPlane( mesh.getPosition()[0][nodeId], mesh.getPosition()[1][nodeId], 
                           mesh.getPosition()[2][nodeId], nrmlX, nrmlY, nrmlZ, 
                           cX, cY, cZ, pX[i], pY[i], pZ[i] );
   }
   
   return;

} // end EdgeInterCheck()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ProjectEdgeNodesToSegment( const MeshData::Viewer& mesh, int edgeId, 
                                                   RealT nrmlX, RealT nrmlY, RealT cX, 
                                                   RealT cY, RealT* pX, RealT* pY )
{
   for (int i=0; i<mesh.numberOfNodesPerElement(); ++i)
   {
      const int nodeId = mesh.getGlobalNodeId(edgeId, i);
      ProjectPointToSegment( mesh.getPosition()[0][nodeId], mesh.getPosition()[1][nodeId], 
                             nrmlX, nrmlY, cX, cY, pX[i], pY[i] );
   }
   
   return;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE ContactPlane::ContactPlane( InterfacePair* pair, 
                                               RealT areaFrac, 
                                               bool interpenOverlap, 
                                               bool interPlane,
                                               int dim )
  : m_pair( pair )
  , m_dim( dim )
  , m_intermediatePlane( interPlane )
  , m_interpenOverlap( interpenOverlap )
  , m_cX( 0.0 )
  , m_cY( 0.0 )
  , m_cZ( 0.0 )
  , m_cXf1( 0.0 )
  , m_cYf1( 0.0 )
  , m_cZf1( 0.0 )
  , m_cXf2( 0.0 )
  , m_cYf2( 0.0 )
  , m_cZf2( 0.0 )
  , m_numInterpenPoly1Vert( 0 )
  , m_numInterpenPoly2Vert( 0 )
  , m_nX( 0.0 )
  , m_nY( 0.0 )
  , m_nZ( 0.0 )
  , m_gap( 0.0 )
  , m_gapTol( 0.0 )
  , m_areaFrac( areaFrac )
  , m_areaMin( 0.0 )
  , m_area( 0.0 )
  , m_interpenArea( 0.0 )
  , m_velGap( 0.0 )
  , m_ratePressure( 0.0 )
  , m_pressure( 0.0 )
{}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE ContactPlane::ContactPlane()
  : ContactPlane(nullptr, 0.0, true, false, 0)
{}

//-----------------------------------------------------------------------------
// 3D contact plane member functions
//-----------------------------------------------------------------------------
TRIBOL_HOST_DEVICE ContactPlane3D::ContactPlane3D( InterfacePair* pair,
                                                   RealT areaFrac, 
                                                   bool interpenOverlap, 
                                                   bool interPlane )
  : ContactPlane( pair, areaFrac, interpenOverlap, interPlane, 3 )
  , m_e1X( 0.0 )
  , m_e1Y( 0.0 )
  , m_e1Z( 0.0 )
  , m_e2X( 0.0 )
  , m_e2Y( 0.0 )
  , m_e2Z( 0.0 )
  , m_numPolyVert( 0 )
  , m_overlapCX( 0.0 )
  , m_overlapCY( 0.0 )
{}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE ContactPlane3D::ContactPlane3D()
  : ContactPlane3D( nullptr, 0.0, true, false )
{}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError CheckFacePair( ContactPlane3D& cp,
                                                const MeshData::Viewer& mesh1,
                                                const MeshData::Viewer& mesh2,
                                                const Parameters& params,
                                                bool fullOverlap )
{
   // Note: Checks #1-#4 are done in the binning

   // alias variables off the InterfacePair
   IndexT element_id1 = cp.getCpElementId1();
   IndexT element_id2 = cp.getCpElementId2();

   // set overlap booleans based on input arguments and contact method
   bool interpenOverlap = (!fullOverlap) ? true : false;

   // CHECK #5: check if the nodes of face2 interpenetrate the 
   // plane defined by face1 AND vice-versa. For proximate faces 
   // that pass check #3 this check may easily indicate that the faces 
   // do in fact intersect. 
   RealT separationTol = params.gap_separation_ratio * 
                        axom::utilities::max( mesh1.getFaceRadius()[ element_id1 ], 
                                              mesh2.getFaceRadius()[ element_id2 ] );
   bool all = false;
   bool ls = FaceInterCheck( mesh1, mesh2, element_id1, element_id2, separationTol, all );
   if (!ls) {
      cp.m_inContact = false;
      return NO_FACE_GEOM_ERROR;
   }

   // if all vertices of one face lie on the other side of the other face, per 
   // FaceInterCheck computation, then use the full projection.
   if (all)
   {
      fullOverlap = true;
      interpenOverlap = false;
      cp.m_interpenOverlap = interpenOverlap;
   }

   // CHECK #6: check if the two faces overlap in a projected sense.
   // To do this check we need to use the contact plane object, which will 
   // have its own local basis that needs to be defined

   // compute cp normal
   cp.computeNormal(mesh1, mesh2);

   // compute cp centroid
   cp.computePlanePoint(mesh1, mesh2);

   // project face nodes onto contact plane. Still do this for mortar. 
   // The mortar face may not be exactly planar so we still need to project 
   // the nodes onto the contact plane, which is defined by average normal of the 
   // nonmortar face.
   constexpr int max_nodes_per_elem = 4;
   RealT projX1[ max_nodes_per_elem ];
   RealT projY1[ max_nodes_per_elem ];
   RealT projZ1[ max_nodes_per_elem ];
   RealT projX2[ max_nodes_per_elem ];
   RealT projY2[ max_nodes_per_elem ];
   RealT projZ2[ max_nodes_per_elem ];
    
   ProjectFaceNodesToPlane( mesh1, element_id1, cp.m_nX, cp.m_nY, cp.m_nZ, 
                            cp.m_cX, cp.m_cY, cp.m_cZ, 
                            &projX1[0], &projY1[0], &projZ1[0] );
   ProjectFaceNodesToPlane( mesh2, element_id2, cp.m_nX, cp.m_nY, cp.m_nZ, 
                            cp.m_cX, cp.m_cY, cp.m_cZ,
                            &projX2[0], &projY2[0], &projZ2[0] );

   // compute cp local coordinate basis
   cp.computeLocalBasis( mesh1 );

   // project the projected global nodal coordinates onto local 
   // contact plane 2D coordinate system. 
   RealT projeX1[ max_nodes_per_elem ];
   RealT projeY1[ max_nodes_per_elem ];
   RealT projeX2[ max_nodes_per_elem ];
   RealT projeY2[ max_nodes_per_elem ];

   cp.globalTo2DLocalCoords( &projX1[0], &projY1[0], &projZ1[0], 
                             &projeX1[0], &projeY1[0], 
                             mesh1.numberOfNodesPerElement() );
   cp.globalTo2DLocalCoords( &projX2[0], &projY2[0], &projZ2[0], 
                             &projeX2[0], &projeY2[0], 
                             mesh2.numberOfNodesPerElement() );

   // compute the overlap area of the two faces. Note, this is the full,
   // but cheaper, overlap computation. This is suitable enough to 
   // compare to a minimum area tolerance, and in general the full 
   // overlap area will be bigger than the interpenetration overlap 
   // area case
   cp.checkPolyOverlap( mesh1, mesh2, &projeX1[0], &projeY1[0], 
                        &projeX2[0], &projeY2[0], 0 );
  
   // compute the overlap area tolerance
   cp.computeAreaTol(mesh1, mesh2, params);

   if (cp.m_area == 0. || cp.m_area < cp.m_areaMin)
   {
      cp.m_inContact = false;
      return NO_FACE_GEOM_ERROR;
   }

   // CHECK #7: compute the required intersection with overlap polygon vertices 
   // and compute the actual mean average gap between the two faces 
   // This can either be a fully integrated gap computation or a single 
   // integration point computation.

   // compute full projected overlap
   if (fullOverlap)
   {
      // compute the full intersection polygon vertex coordinates
      RealT* X1 = &projeX1[0];
      RealT* Y1 = &projeY1[0];
      RealT* X2 = &projeX2[0];
      RealT* Y2 = &projeY2[0];
 
      // assuming each face's vertices are ordered WRT that face's outward unit normal,
      // reorder face 2 vertices to be consistent with face 1. DO NOT CALL POLYREORDER() 
      // to do this. // TODO debug this; this may affect calculations later on. We may 
      // have to unreverse the ordering.
      ElemReverse( X2, Y2, mesh2.numberOfNodesPerElement() );

      // compute intersection polygon and area. Note, the polygon centroid 
      // is stored from the previous intersection calc that just computes 
      // area and local centroid
      RealT pos_tol = params.len_collapse_ratio * 
                     axom::utilities::max( mesh1.getFaceRadius()[ element_id1 ], 
                                           mesh2.getFaceRadius()[ element_id2 ] );
      RealT len_tol = pos_tol;
      FaceGeomError inter_err = Intersection2DPolygon( X1, Y1, mesh1.numberOfNodesPerElement(),
                                                       X2, Y2, mesh2.numberOfNodesPerElement(),
                                                       pos_tol, len_tol, cp.m_polyLocX, 
                                                       cp.m_polyLocY, cp.m_numPolyVert, 
                                                       cp.m_area, false ); 

      if (inter_err != NO_FACE_GEOM_ERROR)
      {
         cp.m_inContact = false;
         return inter_err;
      }

      if (cp.m_area < cp.m_areaMin)
      {
         cp.m_inContact = false;
         return NO_FACE_GEOM_ERROR; 
      }

   } // end if (fullOverlap)

   // compute projected overlap of interpenetrating portion of faces
   if (interpenOverlap)
   {
      // computing the interpenetration area of overlap requires relocating the 
      // centroid using the centroid computed in the check #5 calculations. This 
      // is not necessary for mortar formulations as the mortar plane is correctly 
      // located.
      cp.planePointAndCentroidGap( mesh1, mesh2, 2. * 
         axom::utilities::max( mesh1.getFaceRadius()[ element_id1 ], 
                               mesh2.getFaceRadius()[ element_id2 ] ));

      bool interpen = false;
      FaceGeomError interpen_err = cp.computeLocalInterpenOverlap( 
        mesh1, mesh2, params, interpen ); // same for mortar
      if (interpen_err != NO_FACE_GEOM_ERROR)
      {
         cp.m_inContact = false;
         return interpen_err;
      }
      else if (!interpen) 
      {
         cp.m_inContact = false;
         return NO_FACE_GEOM_ERROR;
      }

      // check new area to area tol
      if (cp.m_interpenArea == 0 || cp.m_interpenArea < cp.m_areaMin) 
      { 
         cp.m_inContact = false;
         return NO_FACE_GEOM_ERROR;
      }

      // reassign area based on possible modification to the actual 
      // intersection polygon.
      cp.m_area = cp.m_interpenArea;

      // compute the local vertex averaged centroid of overlapping polygon
      RealT z;
      VertexAvgCentroid( cp.m_polyLocX, cp.m_polyLocY, nullptr, 
                         cp.m_numPolyVert, cp.m_overlapCX, 
                         cp.m_overlapCY, z );

   } // end if (interpenOverlap)

   // handle the case where the actual polygon with connectivity 
   // and computed vertex coordinates becomes degenerate due to 
   // either position tolerances (segment-segment intersections) 
   // or length tolerances (intersecting polygon segment lengths)
   if (cp.m_numPolyVert < 3) 
   {
#ifdef TRIBOL_USE_HOST
      SLIC_DEBUG( "degenerate polygon intersection detected.\n" );
#endif
      cp.m_inContact = false;
      return DEGENERATE_OVERLAP;
   }

   // Tranform local vertex coordinates to global coordinates for the 
   // current projection of the polygonal overlap
   for (int i=0; i<cp.m_numPolyVert; ++i)
   {
      cp.m_polyX[i] = 0.0;
      cp.m_polyY[i] = 0.0;
      cp.m_polyZ[i] = 0.0;

      cp.local2DToGlobalCoords( cp.m_polyLocX[i], cp.m_polyLocY[i], 
                                cp.m_polyX[i], cp.m_polyY[i], 
                                cp.m_polyZ[i] );
   }

   // check polygonal vertex ordering with common plane normal
   PolyReorderWithNormal( cp.m_polyX, cp.m_polyY, cp.m_polyZ, cp.m_numPolyVert,
                          cp.m_nX, cp.m_nY, cp.m_nZ );

   // transform local interpenetration overlaps to global coords for the 
   // current polygonal overlap
   if (interpenOverlap)
   {
      for (int i=0; i<cp.m_numInterpenPoly1Vert; ++i)
      {
         Local2DToGlobalCoords( cp.m_interpenPoly1X[i], 
                                cp.m_interpenPoly1Y[i], 
                                cp.m_e1X, cp.m_e1Y, cp.m_e1Z,
                                cp.m_e2X, cp.m_e2Y, cp.m_e2Z, 
                                cp.m_cX, cp.m_cY, cp.m_cZ,
                                cp.m_interpenG1X[i], cp.m_interpenG1Y[i],
                                cp.m_interpenG1Z[i] );

      }

      for (int i=0; i<cp.m_numInterpenPoly2Vert; ++i)
      {
         Local2DToGlobalCoords( cp.m_interpenPoly2X[i], 
                                cp.m_interpenPoly2Y[i], 
                                cp.m_e1X, cp.m_e1Y, cp.m_e1Z,
                                cp.m_e2X, cp.m_e2Y, cp.m_e2Z, 
                                cp.m_cX, cp.m_cY, cp.m_cZ,
                                cp.m_interpenG2X[i], cp.m_interpenG2Y[i],
                                cp.m_interpenG2Z[i] );

      }
   }

   // Now that all local-to-global projections have occurred,
   // relocate the contact plane based on the most up-to-date 
   // contact plane centroid and recompute the gap. For interpenOverlap, 
   // the contact plane is updated and this just amounts to a gap 
   // computation. For the fullOverlap case, this may relocate 
   // the contact plane in space. For Mortar methods this routine 
   // only computes the gap based on the current plane point and 
   // normal.
   // 
   // Warning:
   // Make sure that any local to global transformations have 
   // occurred prior to this call. This does not need to be done 
   // for mortar methods. We should just do a gap computation if 
   // needed. 
   cp.planePointAndCentroidGap( mesh1, mesh2, 2. * 
      axom::utilities::max( mesh1.getFaceRadius()[ element_id1 ], 
                            mesh2.getFaceRadius()[ element_id2 ] ));

   // The gap tolerance allows separation up to the separation ratio of the 
   // largest face-radius. This is conservative and allows for possible 
   // over-inclusion. This is done for the mortar method per testing.
   cp.m_gapTol = params.gap_separation_ratio * 
                 axom::utilities::max( mesh1.getFaceRadius()[ element_id1 ], 
                                       mesh2.getFaceRadius()[ element_id2 ] );

   if (cp.m_gap > cp.m_gapTol)
   {
      cp.m_inContact = false;
      return NO_FACE_GEOM_ERROR;
   }

   // for auto-contact, remove contact candidacy for full-overlap 
   // face-pairs with interpenetration exceeding contact penetration fraction. 
   // Note, this check is solely meant to exclude face-pairs composed of faces 
   // on opposite sides of thin structures/plates
   //
   // Recall that interpen gaps are negative
   if (fullOverlap)
   {
      if (ExceedsMaxAutoInterpen( mesh1, mesh2, element_id1, element_id2, 
                                  params, cp.m_gap ))
      {
         cp.m_inContact = false;
         return NO_FACE_GEOM_ERROR;
      }
   }
   
   // if fullOverlap is used, REPROJECT the overlapping polygon 
   // onto the new contact plane 
   if (fullOverlap)
   {
      for (int i=0; i<cp.m_numPolyVert; ++i)
      {
         ProjectPointToPlane( cp.m_polyX[i], cp.m_polyY[i], cp.m_polyZ[i], 
                              cp.m_nX, cp.m_nY, cp.m_nZ,
                              cp.m_cX, cp.m_cY, cp.m_cZ,
                              cp.m_polyX[i], cp.m_polyY[i], cp.m_polyZ[i] );
      }
   }

   cp.m_inContact = true;
   return NO_FACE_GEOM_ERROR;

} // end CheckFacePair()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlane::planePointAndCentroidGap( const MeshData::Viewer& m1,
                                                                const MeshData::Viewer& m2,
                                                                RealT scale )
{
   // project the overlap centroid back to each face using a 
   // line-plane intersection method
   RealT xc1 = 0.;
   RealT yc1 = 0.;
   RealT zc1 = 0.;
   RealT xc2 = 0.;
   RealT yc2 = 0.;
   RealT zc2 = 0.;

   RealT xcg = m_cX;
   RealT ycg = m_cY;
   RealT zcg = 0.0;

   // first project the projected area of overlap's centroid in local 
   // coordinates to global coordinates
   if (m_dim == 3)
   {
      auto& cp3 = static_cast<ContactPlane3D&>(*this);
      auto xloc = cp3.m_overlapCX;
      auto yloc = cp3.m_overlapCY;
      xcg = xloc * cp3.m_e1X + yloc * cp3.m_e2X +cp3.m_cX;
      ycg = xloc * cp3.m_e1Y + yloc * cp3.m_e2Y +cp3.m_cY;
      zcg = xloc * cp3.m_e1Z + yloc * cp3.m_e2Z +cp3.m_cZ;
   }

   // find where the overlap centroid (plane point) intersects each face

   // set the line segment's first vertex at the contact plane centroid scaled 
   // in the direction opposite the contact plane normal

   RealT xA = xcg + m_nX * scale;
   RealT yA = ycg + m_nY * scale;
   RealT zA = 0.0;
   if (m_dim == 3)
   {
      zA = zcg + m_nZ * scale;
   }

   // use the contact plane normal as the segment directional vector scale in 
   // the direction of the contact plane
   RealT xB = xcg - m_nX * scale;
   RealT yB = ycg - m_nY * scale;
   RealT zB = 0.0;
   if (m_dim == 3)
   {
      zB = zcg - m_nZ * scale;
   }
   
   auto fId1 = getCpElementId1();
   auto fId2 = getCpElementId2();
   RealT c1_z = 0.0;
   RealT n1_z = 0.0;
   RealT c2_z = 0.0;
   RealT n2_z = 0.0;
   if (m_dim == 3)
   {
    c1_z = m1.getElementCentroids()[2][fId1];
    n1_z = m1.getElementNormals()[2][fId1];
    c2_z = m2.getElementCentroids()[2][fId2];
    n2_z = m2.getElementNormals()[2][fId2];
   }
   bool inPlane = false;
   LinePlaneIntersection( xA, yA, zA, xB, yB, zB,
                          m1.getElementCentroids()[0][fId1], m1.getElementCentroids()[1][fId1], c1_z,
                          m1.getElementNormals()[0][fId1], m1.getElementNormals()[1][fId1], n1_z,
                          xc1, yc1, zc1, inPlane );

   LinePlaneIntersection( xA, yA, zA, xB, yB, zB,
                          m2.getElementCentroids()[0][fId2], m2.getElementCentroids()[1][fId2], c2_z,
                          m2.getElementNormals()[0][fId2], m2.getElementNormals()[1][fId2], n2_z,
                          xc2, yc2, zc2, inPlane );

   // for intermediate, or common plane methods, average the two contact plane 
   // centroid-to-plane intersections and use this as the new point data for the 
   // contact plane (do not do for mortar methods, or is redundant).
   if ( m_dim == 2 || m_intermediatePlane )
   {
      m_cX = 0.5 * (xc1 + xc2);
      m_cY = 0.5 * (yc1 + yc2);
      m_cZ = 0.5 * (zc1 + zc2); 
   }

   // compute normal gap magnitude (x1 - x2 for positive gap in separation 
   // and negative gap in penetration)
   m_gap = (xc1 - xc2) * m_nX + (yc1 - yc2) * m_nY + (zc1 - zc2) * m_nZ;

   // store the two face points corresponding to the contact plane centroid projection/intersection

   m_cXf1 = xc1;
   m_cYf1 = yc1;
   m_cZf1 = zc1;

   m_cXf2 = xc2;
   m_cYf2 = yc2;
   m_cZf2 = zc2;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE ContactPlane3D CheckAlignedFacePair( InterfacePair& pair,
                                                        const MeshData::Viewer& mesh1,
                                                        const MeshData::Viewer& mesh2,
                                                        const Parameters& params )
{
   // Note: Checks #1-#4 are done in the binning

   // get fraction of largest face we keep for overlap area
   RealT areaFrac = params.overlap_area_frac;

   // alias variables off the InterfacePair
   IndexT element_id1 = pair.m_element_id1;
   IndexT element_id2 = pair.m_element_id2;

   // instantiate temporary contact plane to be returned by this routine
   bool interpenOverlap = false;
   bool intermediatePlane = false;
   ContactPlane3D cp( &pair, areaFrac, interpenOverlap, intermediatePlane);

   // TODO should probably stay consistent with the mortar convention and change 
   // the plane point and normal to the nonmortar surface. These calculations are only 
   // geometry calculations intended to determine if the face-pair should be included 
   // so there isn't much consequence to choosing until we talk about integration. 
   // If the mortar data is switched to nonmortar data, the calculations must be chased 
   // through to make sure contacting face pairs are included.
  
   // set the common plane "point" to the mortar face vertex averaged centroid
   cp.m_cX = mesh1.getElementCentroids()[0][ element_id1 ];
   cp.m_cY = mesh1.getElementCentroids()[1][ element_id1 ];
   cp.m_cZ = mesh1.getElementCentroids()[2][ element_id1 ];

   // set the common plane "normal" to the mortar outward unit normal
   cp.m_nX = mesh1.getElementNormals()[0][ element_id1 ];
   cp.m_nY = mesh1.getElementNormals()[1][ element_id1 ];
   cp.m_nZ = mesh1.getElementNormals()[2][ element_id1 ];

   // set the gap tolerance inclusive for separation up to m_gapTol
   cp.m_gapTol = params.gap_separation_ratio * 
                 axom::utilities::max( mesh1.getFaceRadius()[ element_id1 ],
                                       mesh2.getFaceRadius()[ element_id2 ] );
   
   // set the area fraction
   cp.m_areaFrac = params.overlap_area_frac;

   // set the minimum area
   cp.m_areaMin = cp.m_areaFrac * 
                  axom::utilities::min( mesh1.getElementAreas()[ element_id1 ], 
                  mesh2.getElementAreas()[ element_id2 ] );

   // compute the vector centroid gap and scalar centroid gap to 
   // check the alignment criterion AND gap
   RealT gapVecX =  mesh2.getElementCentroids()[0][element_id2] - mesh1.getElementCentroids()[0][element_id1]; 
   RealT gapVecY =  mesh2.getElementCentroids()[1][element_id2] - mesh1.getElementCentroids()[1][element_id1];
   RealT gapVecZ =  mesh2.getElementCentroids()[2][element_id2] - mesh1.getElementCentroids()[2][element_id1];

   RealT scalarGap = ( mesh2.getElementCentroids()[0][element_id2] - mesh1.getElementCentroids()[0][element_id1] ) * cp.m_nX + 
                    ( mesh2.getElementCentroids()[1][element_id2] - mesh1.getElementCentroids()[1][element_id1] ) * cp.m_nY +
                    ( mesh2.getElementCentroids()[2][element_id2] - mesh1.getElementCentroids()[2][element_id1] ) * cp.m_nZ;

   RealT gapVecMag = magnitude( gapVecX, gapVecY, gapVecZ );

   if (gapVecMag > 1.1*std::abs(scalarGap)) 
   {
      cp.m_inContact = false;
      return cp;
   } 

   // perform gap check
   if (scalarGap > cp.m_gapTol)
   {
      cp.m_inContact = false;
      return cp;
   }

   // for auto-contact, remove contact candidacy for face-pairs with 
   // interpenetration exceeding contact penetration fraction. 
   // Note, this check is solely meant to exclude face-pairs composed of faces 
   // on opposite sides of thin structures/plates
   //
   // Recall that interpen gaps are negative
   if (ExceedsMaxAutoInterpen( mesh1, mesh2, element_id1, element_id2, 
                               params, scalarGap ))
   {
      cp.m_inContact = false;
      return cp;
   }

   // if we are here we have contact between two aligned faces
   cp.m_numPolyVert = mesh1.numberOfNodesPerElement();

   for (int a=0; a<cp.m_numPolyVert; ++a)
   {
      int id = mesh1.getGlobalNodeId(element_id1, a);
      cp.m_polyX[a] = mesh1.getPosition()[0][id];
      cp.m_polyY[a] = mesh1.getPosition()[1][id]; 
      cp.m_polyZ[a] = mesh1.getPosition()[2][id];
   }

   // compute vertex averaged centroid
   VertexAvgCentroid( &cp.m_polyX[0], &cp.m_polyY[0], &cp.m_polyZ[0],
                      cp.m_numPolyVert, cp.m_cX, cp.m_cY, cp.m_cZ );

   cp.m_gap = scalarGap;
   cp.m_area = mesh1.getElementAreas()[element_id1];

   cp.m_inContact = true;
   return cp;

} // end CheckAlignedFacePair()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlane3D::computeNormal( const MeshData::Viewer& m1,
                                                       const MeshData::Viewer& m2 )
{
   IndexT fId1 = m_pair->m_element_id1;
   IndexT fId2 = m_pair->m_element_id2;

   if (m_intermediatePlane)
   {
      // INTERMEDIATE (I.E. COMMON) PLANE normal calculation:
      // compute the cp normal as the average of the two face normals, and in 
      // the direction such that the dot product between the cp normal and 
      // the normal of face 2 is positive. This is the default method of 
      // computing the cp normal
      m_nX = 0.5 * ( m2.getElementNormals()[0][ fId2 ] - m1.getElementNormals()[0][ fId1 ] );
      m_nY = 0.5 * ( m2.getElementNormals()[1][ fId2 ] - m1.getElementNormals()[1][ fId1 ] );
      m_nZ = 0.5 * ( m2.getElementNormals()[2][ fId2 ] - m1.getElementNormals()[2][ fId1 ] );
   }
   else // for mortar
   {
      // the projection plane is the nonmortar (i.e. mesh id 2) surface so 
      // we use the outward normal for face 2 on mesh 2 
      m_nX = m2.getElementNormals()[0][ fId2 ];
      m_nY = m2.getElementNormals()[1][ fId2 ];
      m_nZ = m2.getElementNormals()[2][ fId2 ];
   }

   // normalize the cp normal
   RealT mag = magnitude( m_nX, m_nY, m_nZ );
   RealT invMag = 1.0 / mag;

   m_nX *= invMag;
   m_nY *= invMag;
   m_nZ *= invMag;

   return;

} // end ContactPlane3D::computeNormal()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlane3D::computePlanePoint( const MeshData::Viewer& m1,
                                                           const MeshData::Viewer& m2 )
{
   // compute the cp centroid as the average of the two face's centers. 
   // This is the default method of computing the cp centroid
   IndexT fId1 = m_pair->m_element_id1;
   IndexT fId2 = m_pair->m_element_id2;

   // INTERMEDIATE (I.E. COMMON) PLANE point calculation:
   // average two face vertex averaged centroids
   if (m_intermediatePlane)
   {
      m_cX = 0.5 * ( m1.getElementCentroids()[0][fId1] + m2.getElementCentroids()[0][fId2] );
      m_cY = 0.5 * ( m1.getElementCentroids()[1][fId1] + m2.getElementCentroids()[1][fId2] );
      m_cZ = 0.5 * ( m1.getElementCentroids()[2][fId1] + m2.getElementCentroids()[2][fId2] );
   }
   // ELSE: MORTAR calculation using the vertex averaged 
   // centroid of the nonmortar face
   else
   {
      m_cX = m2.getElementCentroids()[0][ fId2 ];
      m_cY = m2.getElementCentroids()[1][ fId2 ];
      m_cZ = m2.getElementCentroids()[2][ fId2 ];
   }

   return;

} // end ContactPlane3D::computePlanePoint()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlane3D::computeLocalBasis( const MeshData::Viewer& m1 )
{
   // somewhat arbitrarily set the first local basis vector to be 
   // between contact plane centroid and first node on first face as 
   // projected onto the contact plane
   const int nodeId = m1.getGlobalNodeId(m_pair->m_element_id1, 0);

   // project to plane
   RealT pX, pY, pZ;
   ProjectPointToPlane( m1.getPosition()[0][nodeId],
                        m1.getPosition()[1][nodeId],
                        m1.getPosition()[2][nodeId],
                        m_nX, m_nY, m_nZ, m_cX,
                        m_cY, m_cZ, pX, pY, pZ );

   // check the square of the magnitude of the first basis vector to 
   // catch the case where pX = m_cX and so on.
   RealT sqrMag = m_e1X * m_e1X + m_e1Y * m_e1Y + m_e1Z * m_e1Z;

   if (sqrMag < 1.E-12) // note: tolerance on the square of the magnitude
   {
      // translate projected first node by face radius
      RealT radius = m1.getFaceRadius()[ m_pair->m_element_id1 ];
      RealT scale = 1.0 * radius;
   
      RealT pNewX = pX + scale;
      RealT pNewY = pY + scale;
      RealT pNewZ = pZ + scale;

      // project point onto contact plane
      ProjectPointToPlane( pNewX, pNewY, pNewZ,
                           m_nX, m_nY, m_nZ,
                           m_cX, m_cY, m_cZ,
                           pX, pY, pZ );

      m_e1X = pX - m_cX;
      m_e1Y = pY - m_cY;
      m_e1Z = pZ - m_cZ;
   }

   // recompute the magnitude
   RealT mag = magnitude( m_e1X, m_e1Y, m_e1Z );
   RealT invMag = 1.0 / mag;

   // normalize the first basis vector 
   m_e1X *= invMag;
   m_e1Y *= invMag;
   m_e1Z *= invMag;

   // compute the second, and orthogonal, in-plane basis vector as the 
   // cross product between the cp normal and e1. This will be unit because 
   // the cp normal and e1 are unit.
   m_e2X = 0.0;
   m_e2Y = 0.0;
   m_e2Z = 0.0;

   m_e2X += (m_nY * m_e1Z) - (m_nZ * m_e1Y);
   m_e2Y += (m_nZ * m_e1X) - (m_nX * m_e1Z);
   m_e2Z += (m_nX * m_e1Y) - (m_nY * m_e1X);

   // normalize second vector
   mag = magnitude( m_e2X, m_e2Y, m_e2Z );
   invMag = 1.0 / mag;
  
   m_e2X *= invMag;
   m_e2Y *= invMag;
   m_e2Z *= invMag;

   return;

} // end ContactPlane3D::computeLocalBasis()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlane3D::globalTo2DLocalCoords( RealT* pX, RealT* pY, 
                                                               RealT* pZ, RealT* pLX, 
                                                               RealT* pLY, int size )
{

   // loop over projected nodes
   for (int i=0; i<size; ++i) {

      // compute the vector between the point on the plane and the contact plane point
      RealT vX = pX[i] - m_cX;
      RealT vY = pY[i] - m_cY;
      RealT vZ = pZ[i] - m_cZ;

      // project this vector onto the {e1,e2} local basis. This vector is 
      // in the plane so the out-of-plane component should be zero.
      pLX[i] = vX * m_e1X + vY * m_e1Y + vZ * m_e1Z; // projection onto e1
      pLY[i] = vX * m_e2X + vY * m_e2Y + vZ * m_e2Z; // projection onto e2
    
   }

   return;

} // end ContactPlane3D::globalTo2DLocalCoords()

//------------------------------------------------------------------------------
void ContactPlane3D::globalTo2DLocalCoords( RealT pX, RealT pY, RealT pZ,
                                            RealT& pLX, RealT& pLY, int TRIBOL_UNUSED_PARAM(size) )
{
   // compute the vector between the point on the plane and the contact plane point
   RealT vX = pX - m_cX;
   RealT vY = pY - m_cY;
   RealT vZ = pZ - m_cZ;

   // project this vector onto the {e1,e2} local basis. This vector is 
   // in the plane so the out-of-plane component should be zero.
   pLX = vX*m_e1X + vY*m_e1Y + vZ*m_e1Z; // projection onto e1
   pLY = vX*m_e2X + vY*m_e2Y + vZ*m_e2Z; // projection onto e2

   return;

} // end ContactPlane3D::globalTo2DLocalCoords()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlane3D::computeAreaTol( const MeshData::Viewer& m1,
                                                        const MeshData::Viewer& m2,
                                                        const Parameters& params )
{
   if (m_areaFrac < params.overlap_area_frac ) {
#ifdef TRIBOL_USE_HOST
      SLIC_DEBUG( "ContactPlane3D::computeAreaTol() the overlap area fraction too small or negative; " << 
                  "setting to overlap_area_frac parameter." );
#endif
      m_areaFrac = params.overlap_area_frac;
   }

   m_areaMin = m_areaFrac * 
               axom::utilities::min( m1.getElementAreas()[ m_pair->m_element_id1 ], 
                                     m2.getElementAreas()[ m_pair->m_element_id2 ] );

   return;

} // end ContactPlane3D::computeAreaTol()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlane3D::checkPolyOverlap( const MeshData::Viewer& m1,
                                                          const MeshData::Viewer& m2,
                                                          RealT* projLocX1, RealT* projLocY1, 
                                                          RealT* projLocX2, RealT* projLocY2, 
                                                          const int isym )
{
   // change the vertex ordering of one of the faces so that the two match
   constexpr int max_nodes_per_elem = 4;
   RealT x2Temp[ max_nodes_per_elem ];
   RealT y2Temp[ max_nodes_per_elem ];

   // set first vertex coordinates the same
   x2Temp[0] = projLocX2[0];
   y2Temp[0] = projLocY2[0];

   // reorder
   int k = 1;
   for (int i=(m2.numberOfNodesPerElement()-1); i>0; --i)
   {
      x2Temp[k] = projLocX2[i];
      y2Temp[k] = projLocY2[i];
      ++k;
   }

   PolyInterYCentroid( m1.numberOfNodesPerElement(), projLocX1, projLocY1, m2.numberOfNodesPerElement(), 
                       x2Temp, y2Temp, isym, m_area, m_overlapCY );
   PolyInterYCentroid( m1.numberOfNodesPerElement(), projLocY1, projLocX1, m2.numberOfNodesPerElement(), 
                       y2Temp, x2Temp, isym, m_area, m_overlapCX );

   return;

} // end ContactPlane3D::checkPolyOverlap()

//------------------------------------------------------------------------------
void ContactPlane3D::computeIntegralGap()
{
   // TODO implement this routine

   // this may be contact method specific

   return;

} // end ContactPlane3D::computeIntegralGap()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlane3D::local2DToGlobalCoords( RealT xloc, RealT yloc, 
                                                               RealT& xg, RealT& yg, RealT& zg )
{
   // This projection takes the two input local vector components and uses 
   // them as coefficients in a linear combination of local basis vectors. 
   // This gives a 3-vector with origin at the contact plane centroid.
   RealT vx = xloc * m_e1X + yloc * m_e2X;
   RealT vy = xloc * m_e1Y + yloc * m_e2Y;
   RealT vz = xloc * m_e1Z + yloc * m_e2Z;

   // the vector in the global coordinate system requires the addition of the 
   // contact plane point vector (in global Cartesian basis) to the previously 
   // computed vector
   xg = vx + m_cX;
   yg = vy + m_cY;
   zg = vz + m_cZ;

   return;

} // end ContactPlane3D::local2DToGlobalCoords()

//------------------------------------------------------------------------------
void ContactPlane3D::centroidGap( const MeshData::Viewer& m1,
                                  const MeshData::Viewer& m2,
                                  RealT scale )
{
   // project the overlap centroid back to each face using a 
   // line-plane intersection method
   RealT xc1 = 0.;
   RealT yc1 = 0.;
   RealT zc1 = 0.;
   RealT xc2 = 0.;
   RealT yc2 = 0.;
   RealT zc2 = 0.;

   RealT xcg = 0.;
   RealT ycg = 0.;
   RealT zcg = 0.;

   // first project the projected area of overlap's centroid in local 
   // coordinates to global coordinates
   local2DToGlobalCoords(m_overlapCX, m_overlapCY, xcg, ycg, zcg);

   // find where the overlap centroid (plane point) intersects each face

   // set the line segment's first vertex at the contact plane centroid scaled 
   // in the direction opposite the contact plane normal
   RealT xA = xcg + m_nX * scale;
   RealT yA = ycg + m_nY * scale;
   RealT zA = zcg + m_nZ * scale;

   // use the contact plane normal as the segment directional vector scale in 
   // the direction of the contact plane
   RealT xB = xcg - m_nX * scale;
   RealT yB = ycg - m_nY * scale;
   RealT zB = zcg - m_nZ * scale;

   bool inPlane = false;
   IndexT fId1 = m_pair->m_element_id1;
   IndexT fId2 = m_pair->m_element_id2;

   bool intersect1 = LinePlaneIntersection( xA, yA, zA, xB, yB, zB,
                                            m1.getElementCentroids()[0][fId1], m1.getElementCentroids()[1][fId1], m1.getElementCentroids()[2][fId1],
                                            m1.getElementNormals()[0][fId1], m1.getElementNormals()[1][fId1], m1.getElementNormals()[2][fId1],
                                            xc1, yc1, zc1, inPlane );

   bool intersect2 = LinePlaneIntersection( xA, yA, zA, xB, yB, zB,
                                            m2.getElementCentroids()[0][fId2], m2.getElementCentroids()[1][fId2], m2.getElementCentroids()[2][fId2],
                                            m2.getElementNormals()[0][fId2], m2.getElementNormals()[1][fId2], m2.getElementNormals()[2][fId2],
                                            xc2, yc2, zc2, inPlane );
   TRIBOL_UNUSED_VAR(intersect1); // We don't currently use these bool variabeles
   TRIBOL_UNUSED_VAR(intersect2); // but the above function calls modify some parameters


   // compute normal gap magnitude (x1 - x2 for positive gap in separation 
   // and negative gap in penetration)
   m_gap = (xc1 - xc2) * m_nX + (yc1 - yc2) * m_nY + (zc1 - zc2) * m_nZ;

   // store the two face points corresponding to the contact plane centroid projection/intersection
   m_cXf1 = xc1;
   m_cYf1 = yc1;
   m_cZf1 = zc1;

   m_cXf2 = xc2;
   m_cYf2 = yc2;
   m_cZf2 = zc2;

   return;

} // end ContactPlane3D::centroidGap()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError ContactPlane3D::computeLocalInterpenOverlap(
  const MeshData::Viewer& m1, const MeshData::Viewer& m2,  
  const Parameters& params, bool& interpen )
{
   // for each face, loop over current configuration segments and 
   // determine the two (there should be at most two, or in the odd 
   // case zero if one plane lies completely on one side of the 
   // contact plane) that intersect the contact plane. These two new 
   // vertices in addition to the face vertices that "penetrate" the 
   // contact plane define the two new polygons whose intersection we 
   // seek.

   interpen = false;

   RealT xInter[4];
   RealT yInter[4];
   RealT zInter[4];
   bool inPlane = false;
   int numV[2];
   
   // set up vertex id arrays to indicate which vertices pass through
   // contact plane
   StackArrayT<const MeshData::Viewer*, 2> mesh({&m1, &m2});
   constexpr int max_nodes_per_elem = 4;
   int interpenVertex1[ max_nodes_per_elem ];
   int interpenVertex2[ max_nodes_per_elem ];

   StackArrayT<IndexT, 2> element_id({getCpElementId1(), getCpElementId2()});

   for (int i=0; i<2; ++i) // loop over two constituent faces
   {
      // initialize output intersection point array 
      xInter[ 2*i   ] = 0.;
      xInter[ 2*i+1 ] = 0.;
      yInter[ 2*i   ] = 0.;
      yInter[ 2*i+1 ] = 0.;
      zInter[ 2*i   ] = 0.;
      zInter[ 2*i+1 ] = 0.;
      
      // declare array to hold vertex id for all vertices that interpenetrate 
      // the contact plane
      int interpenVertex[ max_nodes_per_elem ];

      int k = 0;
      for (int j=0; j<mesh[i]->numberOfNodesPerElement(); ++j) // loop over face segments
      {
         // initialize current entry in the vertex id list
         interpenVertex[j] = -1;

         // determine segment vertex ids
         int ja = j;
         int jb = (j == (mesh[i]->numberOfNodesPerElement()-1)) ? 0 : (j+1);

         const int& fNodeIdA = mesh[i]->getGlobalNodeId(element_id[i], ja);
         const RealT& x1 = mesh[i]->getPosition()[0][fNodeIdA];
         const RealT& y1 = mesh[i]->getPosition()[1][fNodeIdA];
         const RealT& z1 = mesh[i]->getPosition()[2][fNodeIdA];

         const int& fNodeIdB = mesh[i]->getGlobalNodeId(element_id[i], jb);
         const RealT& x2 = mesh[i]->getPosition()[0][fNodeIdB];
         const RealT& y2 = mesh[i]->getPosition()[1][fNodeIdB];
         const RealT& z2 = mesh[i]->getPosition()[2][fNodeIdB]; 

         if (k > 2)
         {
#ifdef TRIBOL_USE_HOST
            SLIC_DEBUG("ContactPlane3D::computeInterpenOverlap(): too many segment-plane intersections; " << 
                       "check for degenerate face " << m_pair->m_element_id1 << "on mesh " << mesh[i]->meshId() << ".");
#endif
            interpen = false;
            return DEGENERATE_OVERLAP;
         }

         // call segment-to-plane intersection routine
         if (k < 2) // we haven't found both intersection points yet
         {
            bool inter = LinePlaneIntersection( x1, y1, z1, x2, y2, z2,
                                                m_cX, m_cY, m_cZ,
                                                m_nX, m_nY, m_nZ,
                                                xInter[2*i+k], yInter[2*i+k], 
                                                zInter[2*i+k], inPlane );

            if (inter) ++k;
         }

         // we now have the two vertices for the ith face that represent segment-plane intersections.
         // Now determine the existing current configuration face vertices that lie 
         // "on the other side" of the contact plane.
         RealT vX = x1 - m_cX;
         RealT vY = y1 - m_cY; 
         RealT vZ = z1 - m_cZ; 

         // project the vector onto the contact plane normal
         RealT proj = vX*m_nX + vY*m_nY + vZ*m_nZ;
       
         // if the projection for face 1 vertices is positive then that vertex crosses 
         // (i.e. interpenetrates) the contact plane. if the projection for face 2 vertices
         // is negative then that vertex crosses the contact plane
         interpenVertex[ja] = (i == 0 && proj > 0) ? ja : -1;
         interpenVertex[ja] = (i == 1 && proj < 0) ? ja : interpenVertex[ja];

      } // end loop over nodes

      // if we haven't found intersection points, the planes are either separated or coplanar.
      // return

      if (k < 2) 
      {
         interpen = false;
         return NO_FACE_GEOM_ERROR;
      }
  
      // count the number of vertices for the clipped portion of the i^th face that 
      // interpenetrates the contact plane.
      numV[i] = k;
      for (int vid=0; vid<mesh[i]->numberOfNodesPerElement(); ++vid)
      {
         if (interpenVertex[vid] == vid) ++numV[i];

         // populate the face specific id array
         if (i == 0)
         {
            interpenVertex1[vid] = interpenVertex[vid];
         }
         else
         {
            interpenVertex2[vid] = interpenVertex[vid];
         }
      }

   } // end loop over faces

   // allocate arrays to store new clipped vertices for the current face
   constexpr int max_nodes_per_overlap = 8;
   RealT cfx1[ max_nodes_per_overlap ]; // cfx = clipped face x-coordinate
   RealT cfy1[ max_nodes_per_overlap ];
   RealT cfz1[ max_nodes_per_overlap ];

   RealT cfx2[ max_nodes_per_overlap ]; // cfx = clipped face x-coordinate
   RealT cfy2[ max_nodes_per_overlap ];
   RealT cfz2[ max_nodes_per_overlap ];

   // populate arrays
   for (int m=0; m<2; ++m) // populate segment-contact-plane intersection vertices
   {
      cfx1[ m ] = xInter[ m ];
      cfy1[ m ] = yInter[ m ];
      cfz1[ m ] = zInter[ m ];

      cfx2[ m ] = xInter[ 2+m ];
      cfy2[ m ] = yInter[ 2+m ];
      cfz2[ m ] = zInter[ 2+m ];
   }

   // populate the face 1 vertices that cross the contact plane
   int k = 2;
   for (int m=0; m<mesh[0]->numberOfNodesPerElement(); ++m) 
   {
      if (interpenVertex1[m] != -1)
      {
         int fNodeId = mesh[0]->getGlobalNodeId(getCpElementId1(), interpenVertex1[m]);
         cfx1[ k ] = mesh[0]->getPosition()[0][ fNodeId ];
         cfy1[ k ] = mesh[0]->getPosition()[1][ fNodeId ];
         cfz1[ k ] = mesh[0]->getPosition()[2][ fNodeId ];
         ++k;
      }
   }

   // populate the face 2 vertices that cross the contact plane
   k = 2;
   for (int m=0; m<mesh[1]->numberOfNodesPerElement(); ++m) 
   {
      if (interpenVertex2[m] != -1)
      {
         int fNodeId = mesh[1]->getGlobalNodeId(getCpElementId2(), interpenVertex2[m]);
         cfx2[ k ] = mesh[1]->getPosition()[0][ fNodeId ];
         cfy2[ k ] = mesh[1]->getPosition()[1][ fNodeId ];
         cfz2[ k ] = mesh[1]->getPosition()[2][ fNodeId ];
         ++k;
      }
   }

   // declare projected coordinate arrays
   RealT cfx1_proj[ max_nodes_per_overlap ];
   RealT cfy1_proj[ max_nodes_per_overlap ];
   RealT cfz1_proj[ max_nodes_per_overlap ];

   RealT cfx2_proj[ max_nodes_per_overlap ];
   RealT cfy2_proj[ max_nodes_per_overlap ];
   RealT cfz2_proj[ max_nodes_per_overlap ];

   // project clipped face coordinates to contact plane
   for (int i=0; i<numV[0]; ++i)
   {
      ProjectPointToPlane( cfx1[i], cfy1[i], cfz1[i],
                           m_nX, m_nY, m_nZ,
                           m_cX, m_cY, m_cZ,
                           cfx1_proj[i], cfy1_proj[i], cfz1_proj[i] );
   }

   for (int i=0; i<numV[1]; ++i)
   {
      ProjectPointToPlane( cfx2[i], cfy2[i], cfz2[i],
                           m_nX, m_nY, m_nZ,
                           m_cX, m_cY, m_cZ,
                           cfx2_proj[i], cfy2_proj[i], cfz2_proj[i] ); 
   }
 
   // declare local coordinate pointers
   RealT cfx1_loc[ max_nodes_per_overlap ];
   RealT cfy1_loc[ max_nodes_per_overlap ];

   RealT cfx2_loc[ max_nodes_per_overlap ];
   RealT cfy2_loc[ max_nodes_per_overlap ];

   // convert global coords to local contact plane coordinates
   GlobalTo2DLocalCoords( cfx1_proj, cfy1_proj, cfz1_proj,
                          m_e1X, m_e1Y, m_e1Z,
                          m_e2X, m_e2Y, m_e2Z,
                          m_cX, m_cY, m_cZ,
                          cfx1_loc, cfy1_loc, numV[0] ); 

   GlobalTo2DLocalCoords( cfx2_proj, cfy2_proj, cfz2_proj,
                          m_e1X, m_e1Y, m_e1Z,
                          m_e2X, m_e2Y, m_e2Z,
                          m_cX, m_cY, m_cZ,
                          cfx2_loc, cfy2_loc, numV[1] );

   // reorder potentially unordered set of vertices
   PolyReorder( cfx1_loc, cfy1_loc, numV[0] );
   PolyReorder( cfx2_loc, cfy2_loc, numV[1] ); 

   // call intersection routine to get intersecting polygon
   RealT pos_tol = params.len_collapse_ratio * 
                  axom::utilities::max( mesh[0]->getFaceRadius()[ getCpElementId1() ], 
                                        mesh[1]->getFaceRadius()[ getCpElementId2() ] );
   RealT len_tol = pos_tol;
   FaceGeomError inter_err = Intersection2DPolygon( cfx1_loc, cfy1_loc, numV[0],
                                                    cfx2_loc, cfy2_loc, numV[1],
                                                    pos_tol, len_tol, m_polyLocX,
                                                    m_polyLocY, m_numPolyVert,
                                                    m_interpenArea, true );

   if (inter_err != NO_FACE_GEOM_ERROR)
   {
      interpen = false;
      return inter_err;
   }

   // store the local intersection polygons on the contact plane object, 
   // primarily for visualization

   m_numInterpenPoly1Vert = numV[0];
   m_numInterpenPoly2Vert = numV[1];

   for (int i=0; i<numV[0]; ++i)
   {
      m_interpenPoly1X[i] = cfx1_loc[i];
      m_interpenPoly1Y[i] = cfy1_loc[i];
   }

   for (int i=0; i<numV[1]; ++i)
   {
      m_interpenPoly2X[i] = cfx2_loc[i];
      m_interpenPoly2Y[i] = cfy2_loc[i];
   }

   interpen = true;
   return NO_FACE_GEOM_ERROR;
 
} // end ContactPlane3D::computeLocalInterpenOverlap()

//-----------------------------------------------------------------------------
// Contact Plane 2D routines
//-----------------------------------------------------------------------------
TRIBOL_HOST_DEVICE ContactPlane2D::ContactPlane2D( InterfacePair* pair,
                                                   RealT lenFrac,
                                                   bool interpenOverlap,
                                                   bool interPlane )
  : ContactPlane( pair, lenFrac, interpenOverlap, interPlane, 2 )
{
   for (int i=0; i<2; ++i)
   {
      m_segX[i] = 0.0;
      m_segY[i] = 0.0;
   }
} // end ContactPlane2D::ContactPlane2D()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE ContactPlane2D::ContactPlane2D()
  : ContactPlane2D( nullptr, 0.0, true, false )
{}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError CheckEdgePair( ContactPlane2D& cp,
                                                const MeshData::Viewer& mesh1,
                                                const MeshData::Viewer& mesh2,
                                                const Parameters& params,
                                                bool fullOverlap )
{
   // Note: Checks #1-#4 are done in the binning

   // alias variables off the InterfacePair
   IndexT edgeId1 = cp.getCpElementId1();
   IndexT edgeId2 = cp.getCpElementId2();

   // instantiate temporary contact plane to be returned by this routine
   bool interpenOverlap = (!fullOverlap) ? true : false;

   // CHECK #5: check if edge2 vertices have passed through edge1, and 
   // vice-versa. If both vertices have done so for either edge, we trigger 
   // a fullOverlap computation, but we don't know if there is a positive 
   // length of overlap and will have to construct the contact plane 
   // (contact segment) and perform this check. Note, this tolerance is 
   // inclusive up to a separation of a fraction of the edge-radius.
   // This is done for the mortar method per 3D testing.
   RealT separationTol = params.gap_separation_ratio * 
                        axom::utilities::max( mesh1.getFaceRadius()[ edgeId1 ], 
                                              mesh2.getFaceRadius()[ edgeId2 ] );
   bool all = false;
   bool ls = EdgeInterCheck( mesh1, mesh2, edgeId1, edgeId2, separationTol, all );
   if (!ls) 
   {
      cp.m_inContact = false;
      return NO_FACE_GEOM_ERROR;
   }

   // if all the vertices lie on the other side of edge1, then use full 
   // projection
   if (all)
   {
      fullOverlap = true;
      interpenOverlap = false;
      cp.m_interpenOverlap = interpenOverlap;
   }

   // CHECK #6: compute the projected length of overlap on the contact plane.
   // At this point the edges are proximate and likely have a positive 
   // projected length of overlap.

   cp.computeNormal(mesh1, mesh2);
   cp.computePlanePoint(mesh1, mesh2);

   // project each edge's nodes onto the contact segment.
   constexpr int max_nodes_per_elem = 2;
   RealT projX1[max_nodes_per_elem];
   RealT projY1[max_nodes_per_elem];
   RealT projX2[max_nodes_per_elem];
   RealT projY2[max_nodes_per_elem];

   ProjectEdgeNodesToSegment( mesh1, edgeId1, cp.m_nX, cp.m_nY,
                              cp.m_cX, cp.m_cY, &projX1[0], &projY1[0] );
   ProjectEdgeNodesToSegment( mesh2, edgeId2, cp.m_nX, cp.m_nY,
                              cp.m_cX, cp.m_cY, &projX2[0], &projY2[0] );

   // compute the full overlap. Even if we are using the interpenetration 
   // overlap, we have to compute the full overlap in order to properly 
   // locate the contact plane (segment) for the interpenetration calculation
   cp.checkSegOverlap( mesh1, mesh2, params, &projX1[0], &projY1[0], &projX2[0], &projY2[0], 
                       mesh1.numberOfNodesPerElement(), mesh2.numberOfNodesPerElement() );

   // compute the overlap length tolerance
   cp.computeAreaTol( mesh1, mesh2, params ); 

   // check the contact plane length against the minimum length. 
   // In general the interpen length is going to be less than 
   // the full overlap length so we can do this check prior to 
   // any interpenetration overlap calculation
   if (cp.m_area < cp.m_areaMin)
   {
      cp.m_inContact = false;
      return NO_FACE_GEOM_ERROR;
   }

   if (interpenOverlap)
   {
      // properly locate the contact plane (segment)
      cp.planePointAndCentroidGap( mesh1, mesh2, 2. * 
         axom::utilities::max( mesh1.getFaceRadius()[ edgeId1 ], 
                               mesh2.getFaceRadius()[ edgeId2 ] )); 
      bool interpen = false;
      FaceGeomError interpen_err = cp.computeLocalInterpenOverlap(
        mesh1, mesh2, params, interpen );
      if (interpen_err != NO_FACE_GEOM_ERROR)
      {
         cp.m_inContact = false;
         return interpen_err;
      }
      else if (!interpen) 
      {
         cp.m_inContact = false;
         return NO_FACE_GEOM_ERROR;
      }
   }

   // recompute the plane point and centroid gap. For the full overlap 
   // the centroid (i.e. plane point) of the contact plane has been modified 
   // based on the computed segment overlap. This routine will relocate the 
   // contact plane and compute the centroid gap. For the interpenetration 
   // overlap, this ought to only amount to a centroid gap calculation as the 
   // contact plane was properly located wrt the two edges, but the contact 
   // plane point moved (in-contact segment) due to the interpen overlap 
   // segment calc
   cp.planePointAndCentroidGap( mesh1, mesh2, 2. * 
      axom::utilities::max( mesh1.getFaceRadius()[ edgeId1 ], 
                            mesh2.getFaceRadius()[ edgeId2 ] )); 

   // Per 3D mortar testing, allow for separation up to the edge-radius
   cp.m_gapTol = params.gap_separation_ratio * 
                 axom::utilities::max( mesh1.getFaceRadius()[ edgeId1 ], 
                                       mesh2.getFaceRadius()[ edgeId2 ] );
   if (cp.m_gap > cp.m_gapTol)
   {
      cp.m_inContact = false;
      return NO_FACE_GEOM_ERROR;
   }

   // for auto-contact, remove contact candidacy for full-overlap 
   // face-pairs with interpenetration exceeding contact penetration fraction. 
   // Note, this check is solely meant to exclude face-pairs composed of faces 
   // on opposite sides of thin structures/plates
   //
   // Recall that interpen gaps are negative
   if (fullOverlap)
   {
      if (ExceedsMaxAutoInterpen( mesh1, mesh2, edgeId1, edgeId2, 
                                  params, cp.m_gap ))
      {
         cp.m_inContact = false;
         return NO_FACE_GEOM_ERROR;
      }
   }

   // for the full overlap case we need to project the overlap segment
   // onto the updated contact plane
   if (fullOverlap)
   {
      // allocate dummy space for the interpen topology so adding the 
      // contact plane to the contact plane manager doesn't seg fault.
      // Fix this later...
      cp.m_numInterpenPoly1Vert = 2;
      cp.m_numInterpenPoly2Vert = 2;

      for (int i=0; i<2; ++i)
      {
         RealT xproj, yproj;
         ProjectPointToSegment( cp.m_segX[i], cp.m_segY[i], 
                                cp.m_nX, cp.m_nY, 
                                cp.m_cX, cp.m_cY, xproj, yproj );
         cp.m_segX[i] = xproj;
         cp.m_segY[i] = yproj;

         // set the interpen vertices to the full overlap vertices
         cp.m_interpenG1X[i] = 0.0;
         cp.m_interpenG1Y[i] = 0.0;
         cp.m_interpenG2X[i] = 0.0;
         cp.m_interpenG2Y[i] = 0.0;
      }
  
   }

   cp.m_inContact = true;
   return NO_FACE_GEOM_ERROR;

} // end CheckEdgePair()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlane2D::computeNormal( const MeshData::Viewer& m1, 
                                                       const MeshData::Viewer& m2 )
{
   if (m_intermediatePlane)
   {
      // COMMON_PLANE normal calculation:
      // compute the cp normal as the average of the two face normals, and in 
      // the direction such that the dot product between the cp normal and 
      // the normal of face 2 is positive.
      m_nX = 0.5 * (m2.getElementNormals()[0][ m_pair->m_element_id2 ] - m1.getElementNormals()[0][ m_pair->m_element_id1 ]);
      m_nY = 0.5 * (m2.getElementNormals()[1][ m_pair->m_element_id2 ] - m1.getElementNormals()[1][ m_pair->m_element_id1 ]);
      m_nZ = 0.0; // zero out the third component of the normal
   }
   else
   {
      // MORTAR normal calculation. This is the normal of the nonmortar surface
      m_nX = m2.getElementNormals()[0][ m_pair->m_element_id2 ];
      m_nY = m2.getElementNormals()[1][ m_pair->m_element_id2 ];
      m_nZ = 0.;
   }

   // normalize the cp normal
   RealT mag;
   RealT invMag;

   mag = magnitude( m_nX, m_nY );
   invMag = 1.0 / mag;

   m_nX *= invMag;
   m_nY *= invMag;

   return;

} // end ContactPlane2D::computeNormal()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlane2D::computePlanePoint( const MeshData::Viewer& m1,
                                                           const MeshData::Viewer& m2 )
{
   // compute the cp centroid as the average of 
   // the two face's centers. This is the default 
   // method of compute the cp centroid
   m_cX = 0.5 * ( m1.getElementCentroids()[0][m_pair->m_element_id1] + m2.getElementCentroids()[0][m_pair->m_element_id2] );
   m_cY = 0.5 * ( m1.getElementCentroids()[1][m_pair->m_element_id1] + m2.getElementCentroids()[1][m_pair->m_element_id2] );
   m_cZ = 0.0;
   return;

} // end ContactPlane2D::computePlanePoint()

//------------------------------------------------------------------------------
void ContactPlane2D::computeIntegralGap()
{
   // TODO implement this routine
   // This will be contact method dependent
   return;

} // end ContactPlane2D::computeIntegralGap()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlane2D::computeAreaTol( const MeshData::Viewer& m1,
                                                        const MeshData::Viewer& m2,
                                                        const Parameters& params )

{
   if (m_areaFrac < params.overlap_area_frac)
   {
#ifdef TRIBOL_USE_HOST
      SLIC_DEBUG( "ContactPlane2D::computeAreaTol() the overlap area fraction too small or negative; " << 
                  "setting to overlap_area_frac parameter." );
#endif
      m_areaFrac = params.overlap_area_frac;
   }

   m_areaMin = m_areaFrac * 
               axom::utilities::min( m1.getElementAreas()[ m_pair->m_element_id1 ], 
                                     m2.getElementAreas()[ m_pair->m_element_id2 ] );
   return;

} // ContactPlane2D::computeAreaTol()

//------------------------------------------------------------------------------
void ContactPlane2D::centroidGap( const MeshData::Viewer& m1,
                                  const MeshData::Viewer& m2,
                                  RealT scale )
{
   // project the overlap centroid, which is taken to be the point data 
   // (i.e. centroid) of the contact plane, back to each edge using the 
   // line-plane intersection method where each edge is imagined to be 
   // within a plane defined by the edge's centroid and normal
   RealT xc1 = 0.;
   RealT yc1 = 0.;
   RealT zc1 = 0.;
   RealT xc2 = 0.;
   RealT yc2 = 0.;
   RealT zc2 = 0.;

   // find where the overlap centroid (plane point) intersects each face.
   // The following variables store the end vertices of 
   // a fictitious line co-directional with the contact plane normal 
   // passing through each edge

   // set the line segment's first vertex at the contact plane centroid, 
   // scaled in the direction opposite the contact plane normal
   RealT xA = m_cX + m_nX * scale;
   RealT yA = m_cY + m_nY * scale;
   RealT zA = 0.0;

   // use the contact plane normal as the directional vector scale 
   // in the direction of the contact plane
   RealT xB = m_cX - m_nX * scale;
   RealT yB = m_cY - m_nY * scale;
   RealT zB = 0.0;

   bool inPlane = false;
   IndexT fId1 = m_pair->m_element_id1;
   IndexT fId2 = m_pair->m_element_id2;
   bool intersect1 = LinePlaneIntersection( xA, yA, zA, xB, yB, zB,
                                            m1.getElementCentroids()[0][fId1], m1.getElementCentroids()[1][fId1], 0.0,
                                            m1.getElementNormals()[0][fId1], m1.getElementNormals()[1][fId1], 0.0,
                                            xc1, yc1, zc1, inPlane );

   bool intersect2 = LinePlaneIntersection( xA, yA, zA, xB, yB, zB,
                                            m2.getElementCentroids()[0][fId2], m2.getElementCentroids()[1][fId2], 0.0,
                                            m2.getElementNormals()[0][fId2], m2.getElementNormals()[1][fId2], 0.0,
                                            xc2, yc2, zc2, inPlane );
   TRIBOL_UNUSED_VAR(intersect1); // We don't currently use these bool variabeles
   TRIBOL_UNUSED_VAR(intersect2); // but the above function calls modify some parameters

   // compute the normal gap magnitude (x1 - x2 for positive gap in separation 
   // and negative gap in penetration).
   m_gap = (xc1 - xc2) * m_nX + (yc1 - yc2) * m_nY;

   // store the two edge points corresponding to the contact plane centroid 
   // projection/intersection
   m_cXf1 = xc1;
   m_cYf1 = yc1;
   m_cZf1 = 0.0;

   m_cXf2 = xc2;
   m_cYf2 = yc2;
   m_cZf2 = 0.0;

   return;

} // end ContactPlane2D::centroidGap()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlane2D::checkSegOverlap( const MeshData::Viewer& m1,
                                                         const MeshData::Viewer& m2,
                                                         const Parameters& params,
                                                         const RealT* const pX1, const RealT* const pY1, 
                                                         const RealT* const pX2, const RealT* const pY2, 
                                                         const int nV1, const int nV2 )
{
   // TODO: Re-write in a way where the assert isn't needed
#ifdef TRIBOL_USE_CUDA
   assert(nV1 == 2);
   assert(nV2 == 2);
#else
   SLIC_ASSERT( nV1 == 2 );
   SLIC_ASSERT( nV2 == 2 );
#endif

   // get edge ids
   int e1Id = getCpElementId1();
   int e2Id = getCpElementId2();

   //
   // perform the all-in-1 check
   //

   // define the edge 1 non-unit directional vector between vertices 
   // 2 and 1
   RealT lvx1 = pX1[1] - pX1[0];
   RealT lvy1 = pY1[1] - pY1[0];

   // compute vector between each edge 2 vertex and vertex 1 on edge 1.
   // Then dot that vector with the directional vector of edge 1 to see 
   // if they are codirectional. If so, check, that this vector length is 
   // less than edge 1 length indicating that the vertex lies within edge 1
   RealT projTol = params.projection_ratio * 
                  axom::utilities::max( m1.getFaceRadius()[ e1Id ], 
                                        m2.getFaceRadius()[ e2Id ] );
   RealT vLenTol = projTol;
   int inter2 = 0;
   int twoInOneId = -1;
   for (int i=0; i<nV2; ++i)
   {
      RealT vx = pX2[i] - pX1[0];
      RealT vy = pY2[i] - pY1[0]; 

      // compute projection onto edge 1 directional vector
      RealT proj = vx * lvx1 + vy * lvy1;

      // compute length of <vx,vy>; if vLen < some tolerance we have a 
      // coincident node
      RealT vLen = magnitude( vx, vy );

      if (vLen < vLenTol) // coincident vertex
      {
         twoInOneId = i;
         ++inter2;
      }
      else if (proj > projTol && vLen <= m1.getElementAreas()[e1Id]) // interior vertex
      {
         twoInOneId = i;
         ++inter2;
      }
   }

   // if both vertices pass the above criteria than 2 is in 1
   if (inter2 == 2) 
   {
      // set the contact plane (segment) length
      m_area = m2.getElementAreas()[e2Id];

      // set the vertices of the overlap segment
      m_segX[0] = pX2[0];
      m_segY[0] = pY2[0];
  
      m_segX[1] = pX2[1];
      m_segY[1] = pY2[1];

      // relocate the centroid within the currently defined contact 
      // segment
      m_cX = 0.5 * (m_segX[0] + m_segX[1]);
      m_cY = 0.5 * (m_segY[0] + m_segY[1]);
      m_cZ = 0.0;
      return;
   }

   //
   // perform the all-in-2 check
   //

   // define the edge 2 non-unit directional vector between vertices 
   // 2 and 1
   RealT lvx2 = pX2[1] - pX2[0];
   RealT lvy2 = pY2[1] - pY2[0];

   // compute vector between each edge 1 vertex and vertex 1 on edge 2.
   // Then dot that vector with the directional vector of edge 2 to see 
   // if they are codirectional. If so, check, that this vector length is 
   // less than edge 2 length indicating that the vertex is within edge 2
   int inter1 = 0;
   int oneInTwoId = -1;
   for (int i=0; i<nV1; ++i)
   {
      RealT vx = pX1[i] - pX2[0];
      RealT vy = pY1[i] - pY2[0]; 

      // compute projection onto edge 2 directional vector
      RealT proj = vx * lvx2 + vy * lvy2;

      // compute length of <vx,vy>
      RealT vLen = magnitude( vx, vy );

      if (vLen < vLenTol) // coincident vertex
      {
         oneInTwoId = i;
         ++inter1;
      }
      else if (proj > projTol && vLen <= m2.getElementAreas()[e2Id]) // interior vertex
      {
         oneInTwoId = i;
         ++inter1;
      }
   }

   // if both vertices pass the above criteria then 1 is in 2.
   if (inter1 == 2)
   {
      // set the contact plane (segment) length
      m_area = m1.getElementAreas()[e1Id];

      // set the overlap segment vertices on the contact plane object
      m_segX[0] = pX1[0];
      m_segY[0] = pY1[0];
  
      m_segX[1] = pX1[1];
      m_segY[1] = pY1[1];
  
      // relocate the centroid within the currently defined contact 
      // segment
      m_cX = 0.5 * (m_segX[0] + m_segX[1]);
      m_cY = 0.5 * (m_segY[0] + m_segY[1]);
      m_cZ = 0.0;
      return;
   }

   // if inter1 == 0 and inter2 == 0 then there is no overlap
   if (inter1 == 0 && inter2 == 0)
   {
      m_area = 0.0;
      m_cX = m_cY = m_cZ = 0.0;
      return;
   }

   // there is a chance that oneInTowId or twoInOneId is not actually set, 
   // in which case we don't have an overlap. 
   if (oneInTwoId == -1 || twoInOneId == -1)
   {
      m_area = 0.0;
      m_cX = m_cY = m_cZ = 0.0;
      return;
   }

   // if we are here, we have ruled out all-in-1 and all-in-2 overlaps, 
   // and non-overlapping edges, but have the case where edge 1 and 
   // edge 2 overlap some finite distance that is less than either of their 
   // lengths. We have vertex information from the all-in-one checks 
   // indicating which vertices on one edge are within the other edge

   // set the segment vertices
   m_segX[0] = pX1[ oneInTwoId ];
   m_segY[0] = pY1[ oneInTwoId ];
   m_segX[1] = pX2[ twoInOneId ];
   m_segY[1] = pY2[ twoInOneId ];

   // compute vector between "inter"-vertices
   RealT vecX = m_segX[1] - m_segX[0];
   RealT vecY = m_segY[1] - m_segY[0];

   // compute the length of the overlapping segment
   m_area = magnitude( vecX, vecY );

   // compute the overlap centroid
   m_cX = 0.5 * (m_segX[0] + m_segX[1]);
   m_cY = 0.5 * (m_segY[0] + m_segY[1]);
   m_cZ = 0.0;

   return;

} // end ContactPlane2D::checkSegOverlap()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError ContactPlane2D::computeLocalInterpenOverlap(
      const MeshData::Viewer& m1, const MeshData::Viewer& m2, 
      const Parameters& params, bool& interpen )
{
   //
   // Note: the contact plane has to be properly located prior to calling 
   // this routine. 
   //

   interpen = false;

   // all edge-edge interactions suitable for an interpenetration overlap 
   // calculation are edges that intersect at a single point
   int edgeId1 = getCpElementId1();
   int edgeId2 = getCpElementId2();
   int nodeA1 = m1.getGlobalNodeId( edgeId1, 0 );
   int nodeB1 = m1.getGlobalNodeId( edgeId1, 1 );
   int nodeA2 = m2.getGlobalNodeId( edgeId2, 0 );
   int nodeB2 = m2.getGlobalNodeId( edgeId2, 1 );

   RealT xposA1 = m1.getPosition()[0][ nodeA1 ];
   RealT yposA1 = m1.getPosition()[1][ nodeA1 ];
   RealT xposB1 = m1.getPosition()[0][ nodeB1 ];
   RealT yposB1 = m1.getPosition()[1][ nodeB1 ];

   RealT xposA2 = m2.getPosition()[0][ nodeA2 ];
   RealT yposA2 = m2.getPosition()[1][ nodeA2 ];
   RealT xposB2 = m2.getPosition()[0][ nodeB2 ];
   RealT yposB2 = m2.getPosition()[1][ nodeB2 ];

   RealT xInter, yInter;
   bool duplicatePoint = false;

   // check if the segments intersect
   RealT len_tol = params.len_collapse_ratio * 
                  axom::utilities::max( m1.getFaceRadius()[ edgeId1 ], 
                                        m2.getFaceRadius()[ edgeId2 ] );

   bool edgeIntersect = SegmentIntersection2D( xposA1, yposA1, xposB1, yposB1,
                                               xposA2, yposA2, xposB2, yposB2,
                                               nullptr, xInter, yInter, 
                                               duplicatePoint, len_tol );
   
   // check to make sure the edges are actually intersecting. Note 
   // that an intersection point within the specified tolerance of 
   // an edge vertex is collapsed to that vertex point and duplicatePoint 
   // is marked true, but the SegmentIntersection2D returns false
   if (!edgeIntersect && !duplicatePoint)
   {
      m_interpenArea = 0.0;
      interpen = false;
      return NO_FACE_GEOM_ERROR;
   }

   // check if a duplicate point (i.e. vertex) was registered. 
   // That is, if the intersection point is at an edge vertex, 
   // in which case we don't register the interaction
   if (duplicatePoint)
   {
      m_interpenArea = 0.0;
      interpen = false;
      return NO_FACE_GEOM_ERROR;
   }
   
   // project unique intersection point to the contact plane. 
   // The contact plane should have been properly located prior 
   // to this subroutine, in which case the intersection point lies 
   // on the contact plane (segment). We can still do this projection 
   // to be safe and the routine will handle a point that is already 
   // on the plane
   RealT xInterProj, yInterProj;
   ProjectPointToSegment( xInter, yInter, m_nX, m_nY, 
                          m_cX, m_cY, xInterProj, yInterProj );

   // now isolate which vertex on edge 1 and which vertex on edge 2 lie 
   // on the "wrong" side of the contact plane. 

   // define vectors between an edge vertex and the contact plane centroid;
   int interId1 = -1;
   int interId2 = -1;
   int k = 0;
   for (int i=0; i<m1.numberOfNodesPerElement(); ++i)
   {
      int nodeId1 = m1.getGlobalNodeId( edgeId1, i );
      int nodeId2 = m2.getGlobalNodeId( edgeId2, i );
      RealT lvx1 = m1.getPosition()[0][ nodeId1 ] - m_cX;
      RealT lvy1 = m1.getPosition()[1][ nodeId1 ] - m_cY;
      RealT lvx2 = m2.getPosition()[0][ nodeId2 ] - m_cX;
      RealT lvy2 = m2.getPosition()[1][ nodeId2 ] - m_cY;

      // dot each vector with the contact plane normal
      RealT proj1 = lvx1 * m_nX + lvy1 * m_nY;
      RealT proj2 = lvx2 * m_nX + lvy2 * m_nY;

      // check the projection to detect interpenetration and 
      // mark the node id if true
      if (proj1 > 0.0) 
      {
         interId1 = i; 
         ++k;
      }
      if (proj2 < 0.0) 
      {    
         interId2 = i; 
         ++k;
      }
   }

   // Debug check the number of interpenetrating vertices
   if (k > 2)
   {
#ifdef TRIBOL_USE_HOST
      SLIC_DEBUG("ContactPlane2D::computeLocalInterpenOverlap() more than 2 interpenetrating vertices detected; " << 
                 "check for degenerate geometry for edges (" << edgeId1 << ", " << edgeId2 << ") on meshes (" << 
                 m1.meshId() << ", " << m2.meshId() << ").");
#endif
      interpen = false;
      return DEGENERATE_OVERLAP;
   }

   // now that we have marked the interpenetrating vertex of each edge, 
   // compute the distance between the interpenetrating vertex and the 
   // edge intersection point
   int nodeInter1 = m1.getGlobalNodeId( edgeId1, interId1 );
   int nodeInter2 = m2.getGlobalNodeId( edgeId2, interId2 ); 

   RealT vix1 = m1.getPosition()[0][ nodeInter1 ] - xInterProj;
   RealT viy1 = m1.getPosition()[1][ nodeInter1 ] - yInterProj;
   RealT vix2 = m2.getPosition()[0][ nodeInter2 ] - xInterProj;
   RealT viy2 = m2.getPosition()[1][ nodeInter2 ] - yInterProj;

   // determine magnitude of each vector
   RealT mag1 = magnitude( vix1, viy1 );
   RealT mag2 = magnitude( vix2, viy2 );

   // the interpenetration overlap length is the minimum of the above 
   // vectors
   m_interpenArea = (mag1 <= mag2) ? mag1 : mag2;

   if (m_interpenArea > m_areaMin) 
   {
      // determine the edge vertex that forms the overlap segment along 
      // with the intersection point previously computed
      RealT vx1 = (mag1 <= mag2) ? m1.getPosition()[0][ nodeInter1 ] 
                                : m2.getPosition()[0][ nodeInter2 ];

      RealT vy1 = (mag1 <= mag2) ? m1.getPosition()[1][ nodeInter1 ]
                                : m2.getPosition()[1][ nodeInter2 ];
     
      RealT vx2 = xInterProj;
      RealT vy2 = yInterProj;

      // allocate space to store the interpen vertices for visualization
      // (stored on contact plane base class)
      m_numInterpenPoly1Vert = 2;
      m_numInterpenPoly2Vert = 2;

      m_interpenG1X[0] = vix1;
      m_interpenG1Y[0] = viy1;
      m_interpenG1X[1] = xInter;
      m_interpenG1Y[1] = yInter;
   
      m_interpenG2X[0] = vix2;
      m_interpenG2Y[0] = viy2;
      m_interpenG2X[1] = xInter;
      m_interpenG2Y[1] = yInter;

      // project these points to the contact plane
      ProjectPointToSegment(vx1, vy1, m_nX, m_nY, m_cX, m_cY, 
                            m_segX[0], m_segY[0]);
                             
      ProjectPointToSegment(vx2, vy2, m_nX, m_nY, m_cX, m_cY, 
                            m_segX[1], m_segY[1]);

      // compute the new contact plane overlap centroid (segment point)
      m_cX = 0.5 * (m_segX[0] + m_segX[1]);
      m_cY = 0.5 * (m_segY[0] + m_segY[1]);

      interpen = true;
      return NO_FACE_GEOM_ERROR;
   }

   interpen = false;
   return NO_FACE_GEOM_ERROR;

} // end ContactPlane2D::computeLocalInterpenOverlap()

//------------------------------------------------------------------------------

} // end of namespace
