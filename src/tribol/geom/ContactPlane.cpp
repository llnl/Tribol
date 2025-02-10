// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/types.hpp"
#include "ContactPlane.hpp"
#include "ContactPlaneManager.hpp"
#include "GeomUtilities.hpp"
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

namespace tribol {

//------------------------------------------------------------------------------
// free functions
//------------------------------------------------------------------------------
FaceGeomError CheckInterfacePair( InterfacePair& pair, ContactMethod const cMethod,
                                  ContactCase const TRIBOL_UNUSED_PARAM( cCase ), bool& isInteracting )
{
  isInteracting = false;

  // note: will likely need the ContactCase for specialized
  // geometry check(s)/routine(s)

  switch ( cMethod ) {
    case SINGLE_MORTAR:
    case MORTAR_WEIGHTS:
    case COMMON_PLANE: {
      // get instance of global parameters
      parameters_t& params = parameters_t::getInstance();

      // get instance of contact plane manager to store all contact plane data per cycle
      ContactPlaneManager& cpMgr = ContactPlaneManager::getInstance();

      // set whether full overlap is to be used or not. Note: SINGLE_MORTAR and
      // MORTAR_WEIGHTS drop into this 'case', so the method still needs to be checked
      const bool full = ( cMethod == COMMON_PLANE ) ? false : true;
      const bool interpenOverlap = ( full ) ? false : true;
      const bool intermediatePlane = ( cMethod == COMMON_PLANE ) ? true : false;
      real lenFrac = params.overlap_area_frac;
      real areaFrac = lenFrac;

      // Perform contact plane specific computations (2D and 3D)
      if ( params.dimension == 3 ) {
        ContactPlane3D cpTemp( pair, areaFrac, interpenOverlap, intermediatePlane, 3 );
        FaceGeomError face_err = CheckFacePair( pair, full, cpTemp );

        if ( face_err != NO_FACE_GEOM_ERROR ) {
          isInteracting = false;
        } else if ( cpTemp.m_inContact ) {
          cpMgr.addContactPlane( cpTemp );
          isInteracting = true;
        } else {
          isInteracting = false;
        }
        return face_err;
      } else {
        ContactPlane2D cpTemp( pair, lenFrac, interpenOverlap, intermediatePlane, 2 );
        FaceGeomError edge_err = CheckEdgePair( pair, full, cpTemp );

        if ( edge_err != NO_FACE_GEOM_ERROR ) {
          isInteracting = false;
        } else if ( cpTemp.m_inContact ) {
          cpMgr.addContactPlane( cpTemp );
          isInteracting = true;
        } else {
          isInteracting = false;
        }
        return edge_err;
      }
      break;
    }

    case ALIGNED_MORTAR: {
      // get instance of global parameters
      parameters_t& params = parameters_t::getInstance();

      // get instance of contact plane manager to store all contact plane data per cycle
      ContactPlaneManager& cpMgr = ContactPlaneManager::getInstance();

      if ( params.dimension == 3 ) {
        // TODO refactor to make consistent with CheckFacePair, SRW
        ContactPlane3D cpTemp = CheckAlignedFacePair( pair );

        if ( cpTemp.m_inContact ) {
          cpMgr.addContactPlane( cpTemp );
          isInteracting = true;
        } else {
          isInteracting = false;
        }
        return NO_FACE_GEOM_ERROR;
      } else {
        SLIC_ERROR( "2D ALIGNED_MORTAR not yet implemented." );
      }
      break;
    }

    default: {
      // don't do anything
      break;
    }
  }

  return NO_FACE_GEOM_ERROR;  // quiet compiler

}  // end CheckInterfacePair()

//------------------------------------------------------------------------------
bool FaceInterCheck( const MeshData& meshDat1, const MeshData& meshDat2, int fId1, int fId2, real tol, bool& allVerts )
{
  bool check = false;
  allVerts = false;

  // loop over vertices on face 2
  int k = 0;
  for ( int i = 0; i < meshDat2.m_numNodesPerCell; ++i ) {
    // get ith face 2 node id
    const int f2NodeId = meshDat2.getFaceNodeId( fId2, i );

    // compute components of vector between face 1 center and face 2 vertex
    real vX = meshDat1.m_cX[fId1] - meshDat2.m_positionX[f2NodeId];
    real vY = meshDat1.m_cY[fId1] - meshDat2.m_positionY[f2NodeId];
    real vZ = meshDat1.m_cZ[fId1] - meshDat2.m_positionZ[f2NodeId];

    // project the vector onto face 1 normal
    real proj = vX * meshDat1.m_nX[fId1] + vY * meshDat1.m_nY[fId1] + vZ * meshDat1.m_nZ[fId1];

    // if a node of face 2 is on the other side of the plane defined by face 1 the
    // projection will be positive. If a node on face 2 lies on face 1 the projection
    // will be zero. If a node lies just outside of face 1 then the projection will
    // be a small negative number.
    if ( proj > -tol ) {
      check = true;
      ++k;
    }

  }  // end loop over nodes

  // check to see if all nodes are on the other side
  if ( k == meshDat2.m_numNodesPerCell ) {
    allVerts = true;
  }

  // at this point, if the check is false then there is deemed no interaction. The
  // faces are on the non-contacting sides of one another. Return.
  if ( check == false ) {
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
  for ( int i = 0; i < meshDat1.m_numNodesPerCell; ++i ) {
    // get ith face 1 node id
    const int f1NodeId = meshDat1.getFaceNodeId( fId1, i );

    // compute the components of vector between face 2 center and face 1 vertex
    real vX = meshDat2.m_cX[fId2] - meshDat1.m_positionX[f1NodeId];
    real vY = meshDat2.m_cY[fId2] - meshDat1.m_positionY[f1NodeId];
    real vZ = meshDat2.m_cZ[fId2] - meshDat1.m_positionZ[f1NodeId];

    // project the vector onto face 2 normal
    real proj = vX * meshDat2.m_nX[fId2] + vY * meshDat2.m_nY[fId2] + vZ * meshDat2.m_nZ[fId2];

    // if a node of face 1 is on the other side of the plane defined by face 2 the
    // projection will be positive. If a node on face 1 lies on face 2 the projection
    // will be zero. If a node lies just outside of face 2 then the projection will
    // be a small negative number.
    if ( proj > -tol ) {
      check = true;  // check will be true from the first loop
      ++k;
    }

  }  // end loop over nodes

  // check to see if all nodes are on the other side
  if ( k == meshDat1.m_numNodesPerCell ) {
    allVerts = true;
  }

  return check;

}  // end FaceInterCheck()

//------------------------------------------------------------------------------
bool EdgeInterCheck( const MeshData& meshDat1, const MeshData& meshDat2, int eId1, int eId2, real tol, bool& allVerts )
{
  bool check = false;
  allVerts = false;

  // loop over vertices on edge 2
  int k = 0;
  for ( int i = 0; i < meshDat2.m_numNodesPerCell; ++i ) {
    // get edge 2 ith vertex id
    const int e2vId = meshDat2.getFaceNodeId( eId2, i );

    // compute components of vector between edge 1 center and edge 2 vertex
    real vX = meshDat1.m_cX[eId1] - meshDat2.m_positionX[e2vId];
    real vY = meshDat1.m_cY[eId1] - meshDat2.m_positionY[e2vId];

    // project the vector onto edge1 normal
    real proj = vX * meshDat1.m_nX[eId1] + vY * meshDat1.m_nY[eId1];

    // check projection against tolerance
    if ( proj > -tol ) {
      check = true;
      ++k;
    }
  }  // end loop over edge2 vertices

  // check to see if all vertices are on the other side
  if ( k == meshDat2.m_numNodesPerCell ) allVerts = true;

  if ( check == false ) return check;

  // loop over vertices on edge 1 to catch the case where edge 1 lies
  // entirely on the other side of edge 2 triggering a full overlap
  // computation
  k = 0;
  for ( int i = 0; i < meshDat1.m_numNodesPerCell; ++i ) {
    // get edge 1 ith vertex id
    const int e1vId = meshDat1.getFaceNodeId( eId1, i );

    // compute components of vector between edge 2 center and edge 1 vertex
    real vX = meshDat2.m_cX[eId2] - meshDat1.m_positionX[e1vId];
    real vY = meshDat2.m_cY[eId2] - meshDat1.m_positionY[e1vId];

    // project the vector onto edge2 normal
    real proj = vX * meshDat2.m_nX[eId2] + vY * meshDat2.m_nY[eId2];

    // check projection against tolerance
    if ( proj > -tol ) {
      check = true;  // check will be true from the first loop
      ++k;
    }
  }  // end loop over edge1 vertices

  // check to see if all vertices are on the other side
  if ( k == meshDat1.m_numNodesPerCell ) allVerts = true;

  return check;

}  // end EdgeInterCheck()

//------------------------------------------------------------------------------
void ProjectFaceNodesToPlane( const MeshData& mesh, int faceId, real nrmlX, real nrmlY, real nrmlZ, real cX, real cY,
                              real cZ, real* RESTRICT pX, real* RESTRICT pY, real* RESTRICT pZ )
{
  // loop over nodes and project onto the plane defined by the point-normal
  // input arguments
  for ( int i = 0; i < mesh.m_numNodesPerCell; ++i ) {
    const int nodeId = mesh.getFaceNodeId( faceId, i );
    ProjectPointToPlane( mesh.m_positionX[nodeId], mesh.m_positionY[nodeId], mesh.m_positionZ[nodeId], nrmlX, nrmlY,
                         nrmlZ, cX, cY, cZ, pX[i], pY[i], pZ[i] );
  }

  return;

}  // end EdgeInterCheck()

//------------------------------------------------------------------------------
void ProjectEdgeNodesToSegment( const MeshData& mesh, int edgeId, real nrmlX, real nrmlY, real cX, real cY,
                                real* RESTRICT pX, real* RESTRICT pY )
{
  for ( int i = 0; i < mesh.m_numNodesPerCell; ++i ) {
    const int nodeId = mesh.getFaceNodeId( edgeId, i );
    ProjectPointToSegment( mesh.m_positionX[nodeId], mesh.m_positionY[nodeId], nrmlX, nrmlY, cX, cY, pX[i], pY[i] );
  }

  return;
}

//-----------------------------------------------------------------------------
// 3D contact plane member functions
//-----------------------------------------------------------------------------
ContactPlane3D::ContactPlane3D( InterfacePair& pair, real areaFrac, bool interpenOverlap, bool interPlane,
                                int dimension )
{
  m_numFaces = 0;
  dim = dimension;

  m_pair.meshId1 = pair.meshId1;
  m_pair.pairType1 = pair.pairType1;
  m_pair.pairIndex1 = pair.pairIndex1;

  m_pair.meshId2 = pair.meshId2;
  m_pair.pairType2 = pair.pairType2;
  m_pair.pairIndex2 = pair.pairIndex2;
  m_pair.pairId = pair.pairId;

  m_pair.isContactCandidate = pair.isContactCandidate;

  m_intermediatePlane = ( interPlane ) ? true : false;

  m_inContact = false;
  m_interpenOverlap = interpenOverlap;

  m_cX = 0.0;
  m_cY = 0.0;
  m_cZ = 0.0;

  m_cXf1 = 0.0;
  m_cYf1 = 0.0;
  m_cZf1 = 0.0;

  m_cXf2 = 0.0;
  m_cYf2 = 0.0;
  m_cZf2 = 0.0;

  m_nX = 0.0;
  m_nY = 0.0;
  m_nZ = 0.0;

  m_gap = 0.0;
  m_gapTol = 0.0;

  m_e1X = 0.0;
  m_e1Y = 0.0;
  m_e1Z = 0.0;

  m_e2X = 0.0;
  m_e2Y = 0.0;
  m_e2Z = 0.0;

  m_overlapCX = 0.0;
  m_overlapCY = 0.0;

  m_numPolyVert = 0;
  m_polyX = nullptr;
  m_polyY = nullptr;
  m_polyZ = nullptr;
  m_polyLocX = nullptr;
  m_polyLocY = nullptr;

  m_numInterpenPoly1Vert = 0;
  m_interpenPoly1X = nullptr;
  m_interpenPoly1Y = nullptr;

  m_numInterpenPoly2Vert = 0;
  m_interpenPoly2X = nullptr;
  m_interpenPoly2Y = nullptr;

  m_interpenG1X = nullptr;
  m_interpenG1Y = nullptr;
  m_interpenG1Z = nullptr;
  m_interpenG2X = nullptr;
  m_interpenG2Y = nullptr;
  m_interpenG2Z = nullptr;

  m_areaFrac = areaFrac;
  m_areaMin = 0.0;
  m_area = 0.0;
  m_interpenArea = 0.0;

}  // end ContactPlane3D::ContactPlane3D()

//------------------------------------------------------------------------------
ContactPlane3D::ContactPlane3D()
{
  m_numFaces = 0;
  dim = -1;

  m_inContact = false;
  m_interpenOverlap = true;

  m_cX = 0.0;
  m_cY = 0.0;
  m_cZ = 0.0;

  m_cXf1 = 0.0;
  m_cYf1 = 0.0;
  m_cZf1 = 0.0;

  m_cXf2 = 0.0;
  m_cYf2 = 0.0;
  m_cZf2 = 0.0;

  m_nX = 0.0;
  m_nY = 0.0;
  m_nZ = 0.0;

  m_gap = 0.0;
  m_gapTol = 0.0;

  m_e1X = 0.0;
  m_e1Y = 0.0;
  m_e1Z = 0.0;

  m_e2X = 0.0;
  m_e2Y = 0.0;
  m_e2Z = 0.0;

  m_overlapCX = 0.0;
  m_overlapCY = 0.0;

  m_numPolyVert = 0;
  m_polyX = nullptr;
  m_polyY = nullptr;
  m_polyZ = nullptr;
  m_polyLocX = nullptr;
  m_polyLocY = nullptr;

  m_numInterpenPoly1Vert = 0;
  m_interpenPoly1X = nullptr;
  m_interpenPoly1Y = nullptr;

  m_numInterpenPoly2Vert = 0;
  m_interpenPoly2X = nullptr;
  m_interpenPoly2Y = nullptr;

  m_interpenG1X = nullptr;
  m_interpenG1Y = nullptr;
  m_interpenG1Z = nullptr;

  m_interpenG2X = nullptr;
  m_interpenG2Y = nullptr;
  m_interpenG2Z = nullptr;

  m_areaFrac = 0.0;
  m_areaMin = 0.0;
  m_area = 0.0;
  m_interpenArea = 0.0;

}  // end ContactPlane3D::ContactPlane3D()

//------------------------------------------------------------------------------
FaceGeomError CheckFacePair( InterfacePair& pair, bool fullOverlap, ContactPlane3D& cp )
{
  // Note: Checks #1-#4 are done in the binning

  // get instance of global parameters
  parameters_t& params = parameters_t::getInstance();

  // alias variables off the InterfacePair.
  IndexType& meshId1 = pair.meshId1;
  IndexType& meshId2 = pair.meshId2;
  IndexType& faceId1 = pair.pairIndex1;
  IndexType& faceId2 = pair.pairIndex2;

  // get instance of mesh manager
  MeshManager& meshManager = MeshManager::getInstance();

  // get instance of mesh data
  MeshData& mesh1 = meshManager.GetMeshInstance( meshId1 );
  MeshData& mesh2 = meshManager.GetMeshInstance( meshId2 );

  // set overlap booleans based on input arguments and contact method
  bool interpenOverlap = ( !fullOverlap ) ? true : false;

  // CHECK #5: check if the nodes of face2 interpenetrate the
  // plane defined by face1 AND vice-versa. For proximate faces
  // that pass check #3 this check may easily indicate that the faces
  // do in fact intersect. Note, the separation tolerance is 1/2 the longest
  // dimension of the face element. A larger tolerance allowing for separation
  // is per testing of the mortar method.
  real separationTol =
      params.gap_separation_ratio * axom::utilities::max( mesh1.m_faceRadius[faceId1], mesh2.m_faceRadius[faceId2] );
  bool all = false;
  bool ls = FaceInterCheck( mesh1, mesh2, faceId1, faceId2, separationTol, all );
  if ( !ls ) {
    cp.m_inContact = false;
    return NO_FACE_GEOM_ERROR;
  }

  // if all vertices of one face lie on the other side of the other face, per
  // FaceInterCheck computation, then use the full projection.
  if ( all ) {
    fullOverlap = true;
    interpenOverlap = false;
    cp.m_interpenOverlap = interpenOverlap;
  }

  // CHECK #6: check if the two faces overlap in a projected sense.
  // To do this check we need to use the contact plane object, which will
  // have its own local basis that needs to be defined

  // compute cp normal and centroid
  cp.computeNormal();      // this routine handles common plane AND mortar
  cp.computePlanePoint();  // this routine handles common plane AND mortar

  // project face nodes onto contact plane. Still do this for mortar.
  // The mortar face may not be exactly planar so we still need to project
  // the nodes onto the contact plane, which is defined by average normal of the
  // nonmortar face.
  real projX1[mesh1.m_numNodesPerCell];
  real projY1[mesh1.m_numNodesPerCell];
  real projZ1[mesh1.m_numNodesPerCell];
  real projX2[mesh2.m_numNodesPerCell];
  real projY2[mesh2.m_numNodesPerCell];
  real projZ2[mesh2.m_numNodesPerCell];

  ProjectFaceNodesToPlane( mesh1, faceId1, cp.m_nX, cp.m_nY, cp.m_nZ, cp.m_cX, cp.m_cY, cp.m_cZ, &projX1[0], &projY1[0],
                           &projZ1[0] );
  ProjectFaceNodesToPlane( mesh2, faceId2, cp.m_nX, cp.m_nY, cp.m_nZ, cp.m_cX, cp.m_cY, cp.m_cZ, &projX2[0], &projY2[0],
                           &projZ2[0] );

  // compute cp local coordinate basis
  cp.computeLocalBasis();

  // project the projected global nodal coordinates onto local
  // contact plane 2D coordinate system.
  real projeX1[mesh1.m_numNodesPerCell];
  real projeY1[mesh1.m_numNodesPerCell];
  real projeX2[mesh2.m_numNodesPerCell];
  real projeY2[mesh2.m_numNodesPerCell];

  cp.globalTo2DLocalCoords( &projX1[0], &projY1[0], &projZ1[0], &projeX1[0], &projeY1[0], mesh1.m_numNodesPerCell );
  cp.globalTo2DLocalCoords( &projX2[0], &projY2[0], &projZ2[0], &projeX2[0], &projeY2[0], mesh2.m_numNodesPerCell );

  // compute the overlap area of the two faces. Note, this is the full,
  // but cheaper, overlap computation. This is suitable enough to
  // compare to a minimum area tolerance, and in general the full
  // overlap area will be bigger than the interpenetration overlap
  // area case
  cp.checkPolyOverlap( &projeX1[0], &projeY1[0], &projeX2[0], &projeY2[0], 0 );

  // compute the overlap area tolerance
  cp.computeAreaTol();

  if ( cp.m_area == 0. || cp.m_area < cp.m_areaMin ) {
    cp.m_inContact = false;
    return NO_FACE_GEOM_ERROR;
  }

  // CHECK #7: compute the required intersection with overlap polygon vertices
  // and compute the actual mean average gap between the two faces
  // This can either be a fully integrated gap computation or a single
  // integration point computation.

  // compute full projected overlap
  if ( fullOverlap ) {
    // compute the full intersection polygon vertex coordinates
    real* X1 = &projeX1[0];
    real* Y1 = &projeY1[0];
    real* X2 = &projeX2[0];
    real* Y2 = &projeY2[0];

    // assuming each face's vertices are ordered WRT that face's outward unit normal,
    // reorder face 2 vertices to be consistent with face 1. DO NOT CALL POLYREORDER()
    // to do this.
    PolyReverse( X2, Y2, mesh2.m_numNodesPerCell );

    // compute intersection polygon and area. Note, the polygon centroid
    // is stored from the previous intersection calc that just computes
    // area and local centroid
    real pos_tol =
        params.len_collapse_ratio * axom::utilities::max( mesh1.m_faceRadius[faceId1], mesh2.m_faceRadius[faceId2] );
    real len_tol = pos_tol;
    FaceGeomError inter_err =
        Intersection2DPolygon( X1, Y1, mesh1.m_numNodesPerCell, X2, Y2, mesh2.m_numNodesPerCell, pos_tol, len_tol,
                               &cp.m_polyLocX, &cp.m_polyLocY, cp.m_numPolyVert, cp.m_area, false );

    if ( inter_err != NO_FACE_GEOM_ERROR ) {
      cp.m_inContact = false;
      return inter_err;
    }

    if ( cp.m_area < cp.m_areaMin ) {
      cp.m_inContact = false;
      return NO_FACE_GEOM_ERROR;
    }

  }  // end if (fullOverlap)

  // compute projected overlap of interpenetrating portion of faces
  if ( interpenOverlap ) {
    // computing the interpenetration area of overlap requires relocating the
    // centroid using the centroid computed in the check #5 calculations. This
    // is not necessary for mortar formulations as the mortar plane is correctly
    // located.
    cp.planePointAndCentroidGap( 2. *
                                 axom::utilities::max( mesh1.m_faceRadius[faceId1], mesh2.m_faceRadius[faceId2] ) );

    bool interpen = false;
    FaceGeomError interpen_err = cp.computeLocalInterpenOverlap( interpen );  // same for mortar
    if ( interpen_err != NO_FACE_GEOM_ERROR ) {
      cp.m_inContact = false;
      return interpen_err;
    } else if ( !interpen ) {
      cp.m_inContact = false;
      return NO_FACE_GEOM_ERROR;
    }

    // check new area to area tol
    if ( cp.m_interpenArea == 0 || cp.m_interpenArea < cp.m_areaMin ) {
      cp.m_inContact = false;
      return NO_FACE_GEOM_ERROR;
    }

    // reassign area based on possible modification to the actual
    // intersection polygon.
    cp.m_area = cp.m_interpenArea;

    // compute the local vertex averaged centroid of overlapping polygon
    real z;
    VertexAvgCentroid( cp.m_polyLocX, cp.m_polyLocY, nullptr, cp.m_numPolyVert, cp.m_overlapCX, cp.m_overlapCY, z );

  }  // end if (interpenOverlap)

  // check to make sure the pointers to the polygonal overlap vertices
  // have been set
  if ( cp.m_polyLocX == nullptr || cp.m_polyLocY == nullptr ) {
    SLIC_ERROR( "cp.m_polyLocX or cp.m_polyLocY not allocated" );
  }

  // handle the case where the actual polygon with connectivity
  // and computed vertex coordinates becomes degenerate due to
  // either position tolerances (segment-segment intersections)
  // or length tolerances (intersecting polygon segment lengths)
  if ( cp.m_numPolyVert < 3 ) {
    SLIC_DEBUG( "degenerate polygon intersection detected.\n" );
    cp.m_inContact = false;
    return DEGENERATE_OVERLAP;
  }

  // Tranform local vertex coordinates to global coordinates for the
  // current projection of the polygonal overlap
  cp.m_polyX = new real[cp.m_numPolyVert];
  cp.m_polyY = new real[cp.m_numPolyVert];
  cp.m_polyZ = new real[cp.m_numPolyVert];

  for ( int i = 0; i < cp.m_numPolyVert; ++i ) {
    cp.m_polyX[i] = 0.0;
    cp.m_polyY[i] = 0.0;
    cp.m_polyZ[i] = 0.0;

    cp.local2DToGlobalCoords( cp.m_polyLocX[i], cp.m_polyLocY[i], cp.m_polyX[i], cp.m_polyY[i], cp.m_polyZ[i] );
  }

  // check polygonal vertex ordering with common plane normal
  PolyReorderWithNormal( cp.m_polyX, cp.m_polyY, cp.m_polyZ, cp.m_numPolyVert, cp.m_nX, cp.m_nY, cp.m_nZ );

  // transform local interpenetration overlaps to global coords for the
  // current polygonal overlap
  if ( interpenOverlap ) {
    cp.m_interpenG1X = new real[cp.m_numInterpenPoly1Vert];
    cp.m_interpenG1Y = new real[cp.m_numInterpenPoly1Vert];
    cp.m_interpenG1Z = new real[cp.m_numInterpenPoly1Vert];
    for ( int i = 0; i < cp.m_numInterpenPoly1Vert; ++i ) {
      Local2DToGlobalCoords( cp.m_interpenPoly1X[i], cp.m_interpenPoly1Y[i], cp.m_e1X, cp.m_e1Y, cp.m_e1Z, cp.m_e2X,
                             cp.m_e2Y, cp.m_e2Z, cp.m_cX, cp.m_cY, cp.m_cZ, cp.m_interpenG1X[i], cp.m_interpenG1Y[i],
                             cp.m_interpenG1Z[i] );
    }

    cp.m_interpenG2X = new real[cp.m_numInterpenPoly2Vert];
    cp.m_interpenG2Y = new real[cp.m_numInterpenPoly2Vert];
    cp.m_interpenG2Z = new real[cp.m_numInterpenPoly2Vert];
    for ( int i = 0; i < cp.m_numInterpenPoly2Vert; ++i ) {
      Local2DToGlobalCoords( cp.m_interpenPoly2X[i], cp.m_interpenPoly2Y[i], cp.m_e1X, cp.m_e1Y, cp.m_e1Z, cp.m_e2X,
                             cp.m_e2Y, cp.m_e2Z, cp.m_cX, cp.m_cY, cp.m_cZ, cp.m_interpenG2X[i], cp.m_interpenG2Y[i],
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
  cp.planePointAndCentroidGap( 2. * axom::utilities::max( mesh1.m_faceRadius[faceId1], mesh2.m_faceRadius[faceId2] ) );

  // The gap tolerance allows separation up to 50% the largest face-radius.
  // This is conservative and allows for possible over-inclusion. This is done
  // for the mortar method per testing.
  cp.m_gapTol =
      params.gap_separation_ratio * axom::utilities::max( mesh1.m_faceRadius[faceId1], mesh2.m_faceRadius[faceId2] );
  if ( cp.m_gap > cp.m_gapTol ) {
    cp.m_inContact = false;
    return NO_FACE_GEOM_ERROR;
  }

  // if fullOverlap is used, REPROJECT the overlapping polygon
  // onto the new contact plane
  if ( fullOverlap ) {
    for ( int i = 0; i < cp.m_numPolyVert; ++i ) {
      ProjectPointToPlane( cp.m_polyX[i], cp.m_polyY[i], cp.m_polyZ[i], cp.m_nX, cp.m_nY, cp.m_nZ, cp.m_cX, cp.m_cY,
                           cp.m_cZ, cp.m_polyX[i], cp.m_polyY[i], cp.m_polyZ[i] );
    }
  }

  cp.m_inContact = true;
  return NO_FACE_GEOM_ERROR;

}  // end CheckFacePair()

//------------------------------------------------------------------------------
ContactPlane3D CheckAlignedFacePair( InterfacePair& pair )
{
  // Note: Checks #1-#4 are done in the binning

  // get instance of global parameters
  parameters_t& params = parameters_t::getInstance();

  // get fraction of largest face we keep for overlap area
  real areaFrac = params.overlap_area_frac;

  // alias variables off the InterfacePair.
  IndexType& meshId1 = pair.meshId1;
  IndexType& meshId2 = pair.meshId2;
  IndexType& faceId1 = pair.pairIndex1;
  IndexType& faceId2 = pair.pairIndex2;

  // get instance of mesh manager
  MeshManager& meshManager = MeshManager::getInstance();

  // get instance of mesh data
  MeshData& mesh1 = meshManager.GetMeshInstance( meshId1 );
  MeshData& mesh2 = meshManager.GetMeshInstance( meshId2 );

  // instantiate temporary contact plane to be returned by this routine
  bool interpenOverlap = false;
  bool intermediatePlane = false;
  ContactPlane3D cp( pair, areaFrac, interpenOverlap, intermediatePlane, 3 );

  // TODO should probably stay consistent with the mortar convention and change
  // the plane point and normal to the nonmortar surface. These calculations are only
  // geometry calculations intended to determine if the face-pair should be included
  // so there isn't much consequence to choosing until we talk about integration.
  // If the mortar data is switched to nonmortar data, the calculations must be chased
  // through to make sure contacting face pairs are included.

  // set the common plane "point" to the mortar face vertex averaged centroid
  cp.m_cX = mesh1.m_cX[faceId1];
  cp.m_cY = mesh1.m_cY[faceId1];
  cp.m_cZ = mesh1.m_cZ[faceId1];

  // set the common plane "normal" to the mortar outward unit normal
  cp.m_nX = mesh1.m_nX[faceId1];
  cp.m_nY = mesh1.m_nY[faceId1];
  cp.m_nZ = mesh1.m_nZ[faceId1];

  // set the gap tolerance inclusive for separation up to m_gapTol
  cp.m_gapTol =
      params.gap_separation_ratio * axom::utilities::max( mesh1.m_faceRadius[faceId1], mesh2.m_faceRadius[faceId2] );

  // set the area fraction
  cp.m_areaFrac = params.overlap_area_frac;

  // set the minimum area
  cp.m_areaMin = cp.m_areaFrac * axom::utilities::min( mesh1.m_area[faceId1], mesh2.m_area[faceId2] );

  // compute the vector centroid gap and scalar centroid gap to
  // check the alignment criterion AND gap
  real gapVecX = mesh2.m_cX[faceId2] - mesh1.m_cX[faceId1];
  real gapVecY = mesh2.m_cY[faceId2] - mesh1.m_cY[faceId1];
  real gapVecZ = mesh2.m_cZ[faceId2] - mesh1.m_cZ[faceId1];

  real scalarGap = ( mesh2.m_cX[faceId2] - mesh1.m_cX[faceId1] ) * cp.m_nX +
                   ( mesh2.m_cY[faceId2] - mesh1.m_cY[faceId1] ) * cp.m_nY +
                   ( mesh2.m_cZ[faceId2] - mesh1.m_cZ[faceId1] ) * cp.m_nZ;

  real gapVecMag = magnitude( gapVecX, gapVecY, gapVecZ );

  if ( gapVecMag > 1.1 * std::abs( scalarGap ) ) {
    cp.m_inContact = false;
    return cp;
  }

  // perform gap check
  if ( scalarGap > cp.m_gapTol ) {
    cp.m_inContact = false;
    return cp;
  }

  // if we are here we have contact between two aligned faces
  cp.m_numPolyVert = mesh1.m_numNodesPerCell;
  cp.m_polyX = new real[cp.m_numPolyVert];
  cp.m_polyY = new real[cp.m_numPolyVert];
  cp.m_polyZ = new real[cp.m_numPolyVert];

  for ( int a = 0; a < cp.m_numPolyVert; ++a ) {
    int id = mesh1.m_connectivity[mesh1.m_numNodesPerCell * faceId1 + a];
    cp.m_polyX[a] = mesh1.m_positionX[id];
    cp.m_polyY[a] = mesh1.m_positionY[id];
    cp.m_polyZ[a] = mesh1.m_positionZ[id];
  }

  // compute vertex averaged centroid
  VertexAvgCentroid( &cp.m_polyX[0], &cp.m_polyY[0], &cp.m_polyZ[0], cp.m_numPolyVert, cp.m_cX, cp.m_cY, cp.m_cZ );

  cp.m_gap = scalarGap;
  cp.m_area = mesh1.m_area[faceId1];

  cp.m_inContact = true;
  return cp;

}  // end CheckAlignedFacePair()

//------------------------------------------------------------------------------
void ContactPlane3D::computeNormal()
{
  // get the mesh data
  MeshData& m1 = getCpMeshData( m_pair.meshId1 );
  MeshData& m2 = getCpMeshData( m_pair.meshId2 );

  IndexType fId1 = m_pair.pairIndex1;
  IndexType fId2 = m_pair.pairIndex2;

  if ( m_intermediatePlane ) {
    // INTERMEDIATE (I.E. COMMON) PLANE normal calculation:
    // compute the cp normal as the average of the two face normals, and in
    // the direction such that the dot product between the cp normal and
    // the normal of face 2 is positive. This is the default method of
    // computing the cp normal
    m_nX = 0.5 * ( m2.m_nX[fId2] - m1.m_nX[fId1] );
    m_nY = 0.5 * ( m2.m_nY[fId2] - m1.m_nY[fId1] );
    m_nZ = 0.5 * ( m2.m_nZ[fId2] - m1.m_nZ[fId1] );
  } else  // for mortar
  {
    // the projection plane is the nonmortar (i.e. mesh id 2) surface so
    // we use the outward normal for face 2 on mesh 2
    m_nX = m2.m_nX[fId2];
    m_nY = m2.m_nY[fId2];
    m_nZ = m2.m_nZ[fId2];
  }

  // normalize the cp normal
  real mag = magnitude( m_nX, m_nY, m_nZ );
  real invMag = 1.0 / mag;

  m_nX *= invMag;
  m_nY *= invMag;
  m_nZ *= invMag;

  return;

}  // end ContactPlane3D::computeNormal()

//------------------------------------------------------------------------------
void ContactPlane3D::computePlanePoint()
{
  // get the mesh data
  MeshData& m1 = getCpMeshData( m_pair.meshId1 );
  MeshData& m2 = getCpMeshData( m_pair.meshId2 );

  // compute the cp centroid as the average of the two face's centers.
  // This is the default method of computing the cp centroid
  IndexType fId1 = m_pair.pairIndex1;
  IndexType fId2 = m_pair.pairIndex2;

  // INTERMEDIATE (I.E. COMMON) PLANE point calculation:
  // average two face vertex averaged centroids
  if ( m_intermediatePlane ) {
    m_cX = 0.5 * ( m1.m_cX[fId1] + m2.m_cX[fId2] );
    m_cY = 0.5 * ( m1.m_cY[fId1] + m2.m_cY[fId2] );
    m_cZ = 0.5 * ( m1.m_cZ[fId1] + m2.m_cZ[fId2] );
  }
  // ELSE: MORTAR calculation using the vertex averaged
  // centroid of the nonmortar face
  else {
    m_cX = m2.m_cX[fId2];
    m_cY = m2.m_cY[fId2];
    m_cZ = m2.m_cZ[fId2];
  }

  return;

}  // end ContactPlane3D::computePlanePoint()

//------------------------------------------------------------------------------
void ContactPlane3D::computeLocalBasis()
{
  // get the mesh data associated with the first face
  MeshData& m1 = getCpMeshData( m_pair.meshId1 );

  // somewhat arbitrarily set the first local basis vector to be
  // between contact plane centroid and first node on first face as
  // projected onto the contact plane
  const int nodeId = m1.getFaceNodeId( m_pair.pairIndex1, 0 );

  // project to plane
  real pX, pY, pZ;
  ProjectPointToPlane( m1.m_positionX[nodeId], m1.m_positionY[nodeId], m1.m_positionZ[nodeId], m_nX, m_nY, m_nZ, m_cX,
                       m_cY, m_cZ, pX, pY, pZ );

  // check the square of the magnitude of the first basis vector to
  // catch the case where pX = m_cX and so on.
  real sqrMag = m_e1X * m_e1X + m_e1Y * m_e1Y + m_e1Z * m_e1Z;

  if ( sqrMag < 1.E-12 )  // note: tolerance on the square of the magnitude
  {
    // translate projected first node by face radius
    real radius = m1.m_faceRadius[m_pair.pairIndex1];
    real scale = 1.0 * radius;

    real pNewX = pX + scale;
    real pNewY = pY + scale;
    real pNewZ = pZ + scale;

    // project point onto contact plane
    ProjectPointToPlane( pNewX, pNewY, pNewZ, m_nX, m_nY, m_nZ, m_cX, m_cY, m_cZ, pX, pY, pZ );

    m_e1X = pX - m_cX;
    m_e1Y = pY - m_cY;
    m_e1Z = pZ - m_cZ;
  }

  // recompute the magnitude
  real mag = magnitude( m_e1X, m_e1Y, m_e1Z );
  real invMag = 1.0 / mag;

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

  m_e2X += ( m_nY * m_e1Z ) - ( m_nZ * m_e1Y );
  m_e2Y += ( m_nZ * m_e1X ) - ( m_nX * m_e1Z );
  m_e2Z += ( m_nX * m_e1Y ) - ( m_nY * m_e1X );

  // normalize second vector
  mag = magnitude( m_e2X, m_e2Y, m_e2Z );
  invMag = 1.0 / mag;

  m_e2X *= invMag;
  m_e2Y *= invMag;
  m_e2Z *= invMag;

  return;

}  // end ContactPlane3D::computeLocalBasis()

//------------------------------------------------------------------------------
void ContactPlane3D::globalTo2DLocalCoords( real* RESTRICT pX, real* RESTRICT pY, real* RESTRICT pZ, real* RESTRICT pLX,
                                            real* RESTRICT pLY, int size )
{
  // loop over projected nodes
  for ( int i = 0; i < size; ++i ) {
    // compute the vector between the point on the plane and the contact plane point
    real vX = pX[i] - m_cX;
    real vY = pY[i] - m_cY;
    real vZ = pZ[i] - m_cZ;

    // project this vector onto the {e1,e2} local basis. This vector is
    // in the plane so the out-of-plane component should be zero.
    pLX[i] = vX * m_e1X + vY * m_e1Y + vZ * m_e1Z;  // projection onto e1
    pLY[i] = vX * m_e2X + vY * m_e2Y + vZ * m_e2Z;  // projection onto e2
  }

  return;

}  // end ContactPlane3D::globalTo2DLocalCoords()

//------------------------------------------------------------------------------
void ContactPlane3D::globalTo2DLocalCoords( real pX, real pY, real pZ, real& pLX, real& pLY,
                                            int TRIBOL_UNUSED_PARAM( size ) )
{
  // compute the vector between the point on the plane and the contact plane point
  real vX = pX - m_cX;
  real vY = pY - m_cY;
  real vZ = pZ - m_cZ;

  // project this vector onto the {e1,e2} local basis. This vector is
  // in the plane so the out-of-plane component should be zero.
  pLX = vX * m_e1X + vY * m_e1Y + vZ * m_e1Z;  // projection onto e1
  pLY = vX * m_e2X + vY * m_e2Y + vZ * m_e2Z;  // projection onto e2

  return;

}  // end ContactPlane3D::globalTo2DLocalCoords()

//------------------------------------------------------------------------------
void ContactPlane3D::computeAreaTol()
{
  parameters_t& parameters = parameters_t::getInstance();

  if ( m_areaFrac < parameters.overlap_area_frac ) {
    SLIC_DEBUG( "ContactPlane3D::computeAreaTol() the overlap area fraction too small or negative; "
                << "setting to overlap_area_frac parameter." );
    m_areaFrac = parameters.overlap_area_frac;
  }

  MeshData& mesh1 = getCpMeshData( m_pair.meshId1 );
  MeshData& mesh2 = getCpMeshData( m_pair.meshId2 );

  m_areaMin = m_areaFrac * axom::utilities::min( mesh1.m_area[m_pair.pairIndex1], mesh2.m_area[m_pair.pairIndex2] );

  return;

}  // end ContactPlane3D::computeAreaTol()

//------------------------------------------------------------------------------
void ContactPlane3D::checkPolyOverlap( real* RESTRICT projLocX1, real* RESTRICT projLocY1, real* RESTRICT projLocX2,
                                       real* RESTRICT projLocY2, const int isym )
{
  MeshData& mesh1 = getCpMeshData( m_pair.meshId1 );
  MeshData& mesh2 = getCpMeshData( m_pair.meshId2 );

  // change the vertex ordering of one of the faces so that the two match
  real x2Temp[mesh2.m_numNodesPerCell];
  real y2Temp[mesh2.m_numNodesPerCell];

  // set first vertex coordinates the same
  x2Temp[0] = projLocX2[0];
  y2Temp[0] = projLocY2[0];

  // reorder
  int k = 1;
  for ( int i = ( mesh2.m_numNodesPerCell - 1 ); i > 0; --i ) {
    x2Temp[k] = projLocX2[i];
    y2Temp[k] = projLocY2[i];
    ++k;
  }

  PolyInterYCentroid( mesh1.m_numNodesPerCell, projLocX1, projLocY1, mesh2.m_numNodesPerCell, x2Temp, y2Temp, isym,
                      m_area, m_overlapCY );
  PolyInterYCentroid( mesh1.m_numNodesPerCell, projLocY1, projLocX1, mesh2.m_numNodesPerCell, y2Temp, x2Temp, isym,
                      m_area, m_overlapCX );

  return;

}  // end ContactPlane3D::checkPolyOverlap()

//------------------------------------------------------------------------------
void ContactPlane3D::computeIntegralGap()
{
  // TODO implement this routine

  // this may be contact method specific

  return;

}  // end ContactPlane3D::computeIntegralGap()

//------------------------------------------------------------------------------
void ContactPlane3D::local2DToGlobalCoords( real xloc, real yloc, real& xg, real& yg, real& zg )
{
  // This projection takes the two input local vector components and uses
  // them as coefficients in a linear combination of local basis vectors.
  // This gives a 3-vector with origin at the contact plane centroid.
  real vx = xloc * m_e1X + yloc * m_e2X;
  real vy = xloc * m_e1Y + yloc * m_e2Y;
  real vz = xloc * m_e1Z + yloc * m_e2Z;

  // the vector in the global coordinate system requires the addition of the
  // contact plane point vector (in global Cartesian basis) to the previously
  // computed vector
  xg = vx + m_cX;
  yg = vy + m_cY;
  zg = vz + m_cZ;

  return;

}  // end ContactPlane3D::local2DToGlobalCoords()

//------------------------------------------------------------------------------
void ContactPlane3D::planePointAndCentroidGap( real scale )
{
  // project the overlap centroid back to each face using a
  // line-plane intersection method
  real xc1 = 0.;
  real yc1 = 0.;
  real zc1 = 0.;
  real xc2 = 0.;
  real yc2 = 0.;
  real zc2 = 0.;

  real xcg = 0.;
  real ycg = 0.;
  real zcg = 0.;

  // first project the projected area of overlap's centroid in local
  // coordinates to global coordinates
  local2DToGlobalCoords( m_overlapCX, m_overlapCY, xcg, ycg, zcg );

  // find where the overlap centroid (plane point) intersects each face

  // set the line segment's first vertex at the contact plane centroid scaled
  // in the direction opposite the contact plane normal

  real xA = xcg + m_nX * scale;
  real yA = ycg + m_nY * scale;
  real zA = zcg + m_nZ * scale;

  // use the contact plane normal as the segment directional vector scale in
  // the direction of the contact plane
  real xB = xcg - m_nX * scale;
  real yB = ycg - m_nY * scale;
  real zB = zcg - m_nZ * scale;

  MeshData& m1 = getCpMeshData( m_pair.meshId1 );
  MeshData& m2 = getCpMeshData( m_pair.meshId2 );
  bool inPlane = false;
  IndexType fId1 = m_pair.pairIndex1;
  IndexType fId2 = m_pair.pairIndex2;
  bool intersect1 = LinePlaneIntersection( xA, yA, zA, xB, yB, zB, m1.m_cX[fId1], m1.m_cY[fId1], m1.m_cZ[fId1],
                                           m1.m_nX[fId1], m1.m_nY[fId1], m1.m_nZ[fId1], xc1, yc1, zc1, inPlane );

  bool intersect2 = LinePlaneIntersection( xA, yA, zA, xB, yB, zB, m2.m_cX[fId2], m2.m_cY[fId2], m2.m_cZ[fId2],
                                           m2.m_nX[fId2], m2.m_nY[fId2], m2.m_nZ[fId2], xc2, yc2, zc2, inPlane );
  TRIBOL_UNUSED_VAR( intersect1 );  // We don't currently use these bool variabeles
  TRIBOL_UNUSED_VAR( intersect2 );  // but the above function calls modify some parameters

  // for intermediate, or common plane methods, average the two contact plane
  // centroid-to-plane intersections and use this as the new point data for the
  // contact plane (do not do for mortar methods, or is redundant).
  if ( m_intermediatePlane ) {
    m_cX = 0.5 * ( xc1 + xc2 );
    m_cY = 0.5 * ( yc1 + yc2 );
    m_cZ = 0.5 * ( zc1 + zc2 );
  }

  // compute normal gap magnitude (x1 - x2 for positive gap in separation
  // and negative gap in penetration)
  m_gap = ( xc1 - xc2 ) * m_nX + ( yc1 - yc2 ) * m_nY + ( zc1 - zc2 ) * m_nZ;

  // store the two face points corresponding to the contact plane centroid projection/intersection

  m_cXf1 = xc1;
  m_cYf1 = yc1;
  m_cZf1 = zc1;

  m_cXf2 = xc2;
  m_cYf2 = yc2;
  m_cZf2 = zc2;

  return;
}  // end ContactPlane3D::planePointAndCentroidGap()

//------------------------------------------------------------------------------
void ContactPlane3D::centroidGap( real scale )
{
  // project the overlap centroid back to each face using a
  // line-plane intersection method
  real xc1 = 0.;
  real yc1 = 0.;
  real zc1 = 0.;
  real xc2 = 0.;
  real yc2 = 0.;
  real zc2 = 0.;

  real xcg = 0.;
  real ycg = 0.;
  real zcg = 0.;

  // first project the projected area of overlap's centroid in local
  // coordinates to global coordinates
  local2DToGlobalCoords( m_overlapCX, m_overlapCY, xcg, ycg, zcg );

  // find where the overlap centroid (plane point) intersects each face

  // set the line segment's first vertex at the contact plane centroid scaled
  // in the direction opposite the contact plane normal
  real xA = xcg + m_nX * scale;
  real yA = ycg + m_nY * scale;
  real zA = zcg + m_nZ * scale;

  // use the contact plane normal as the segment directional vector scale in
  // the direction of the contact plane
  real xB = xcg - m_nX * scale;
  real yB = ycg - m_nY * scale;
  real zB = zcg - m_nZ * scale;

  MeshData& m1 = getCpMeshData( m_pair.meshId1 );
  MeshData& m2 = getCpMeshData( m_pair.meshId2 );
  bool inPlane = false;
  IndexType fId1 = m_pair.pairIndex1;
  IndexType fId2 = m_pair.pairIndex2;

  bool intersect1 = LinePlaneIntersection( xA, yA, zA, xB, yB, zB, m1.m_cX[fId1], m1.m_cY[fId1], m1.m_cZ[fId1],
                                           m1.m_nX[fId1], m1.m_nY[fId1], m1.m_nZ[fId1], xc1, yc1, zc1, inPlane );

  bool intersect2 = LinePlaneIntersection( xA, yA, zA, xB, yB, zB, m2.m_cX[fId2], m2.m_cY[fId2], m2.m_cZ[fId2],
                                           m2.m_nX[fId2], m2.m_nY[fId2], m2.m_nZ[fId2], xc2, yc2, zc2, inPlane );
  TRIBOL_UNUSED_VAR( intersect1 );  // We don't currently use these bool variabeles
  TRIBOL_UNUSED_VAR( intersect2 );  // but the above function calls modify some parameters

  // compute normal gap magnitude (x1 - x2 for positive gap in separation
  // and negative gap in penetration)
  m_gap = ( xc1 - xc2 ) * m_nX + ( yc1 - yc2 ) * m_nY + ( zc1 - zc2 ) * m_nZ;

  // store the two face points corresponding to the contact plane centroid projection/intersection
  m_cXf1 = xc1;
  m_cYf1 = yc1;
  m_cZf1 = zc1;

  m_cXf2 = xc2;
  m_cYf2 = yc2;
  m_cZf2 = zc2;

  return;

}  // end ContactPlane3D::centroidGap()

//------------------------------------------------------------------------------
FaceGeomError ContactPlane3D::computeLocalInterpenOverlap( bool& interpen )
{
  // for each face, loop over current configuration segments and
  // determine the two (there should be at most two, or in the odd
  // case zero if one plane lies completely on one side of the
  // contact plane) that intersect the contact plane. These two new
  // vertices in addition to the face vertices that "penetrate" the
  // contact plane define the two new polygons whose intersection we
  // seek.

  interpen = false;
  parameters_t& parameters = parameters_t::getInstance();

  real xInter[4];
  real yInter[4];
  real zInter[4];
  bool inPlane = false;
  int numV[2];

  // set up vertex id arrays to indicate which vertices pass through
  // contact plane
  MeshData& mesh1 = getCpMeshData( m_pair.meshId1 );
  int interpenVertex1[mesh1.m_numNodesPerCell];

  MeshData& mesh2 = getCpMeshData( m_pair.meshId2 );
  int interpenVertex2[mesh2.m_numNodesPerCell];

  for ( int i = 0; i < 2; ++i )  // loop over two constituent faces
  {
    // initialize output intersection point array
    xInter[2 * i] = 0.;
    xInter[2 * i + 1] = 0.;
    yInter[2 * i] = 0.;
    yInter[2 * i + 1] = 0.;
    zInter[2 * i] = 0.;
    zInter[2 * i + 1] = 0.;

    int fId;
    int meshId;

    if ( i == 0 ) {
      meshId = m_pair.meshId1;
      fId = m_pair.pairIndex1;
    } else {
      meshId = m_pair.meshId2;
      fId = m_pair.pairIndex2;
    }

    MeshData& mesh = getCpMeshData( meshId );

    // declare array to hold vertex id for all vertices that interpenetrate
    // the contact plane
    int interpenVertex[mesh.m_numNodesPerCell];

    int k = 0;
    for ( int j = 0; j < mesh.m_numNodesPerCell; ++j )  // loop over face segments
    {
      // initialize current entry in the vertex id list
      interpenVertex[j] = -1;

      // determine segment vertex ids
      int ja = j;
      int jb = ( j == ( mesh.m_numNodesPerCell - 1 ) ) ? 0 : ( j + 1 );

      const int& fNodeIdA = mesh.getFaceNodeId( fId, ja );
      const real& x1 = mesh.m_positionX[fNodeIdA];
      const real& y1 = mesh.m_positionY[fNodeIdA];
      const real& z1 = mesh.m_positionZ[fNodeIdA];

      const int& fNodeIdB = mesh.getFaceNodeId( fId, jb );
      const real& x2 = mesh.m_positionX[fNodeIdB];
      const real& y2 = mesh.m_positionY[fNodeIdB];
      const real& z2 = mesh.m_positionZ[fNodeIdB];

      meshId = m_pair.meshId1;
      fId = m_pair.pairIndex1;
      if ( k > 2 ) {
        SLIC_DEBUG( "ContactPlane3D::computeInterpenOverlap(): too many segment-plane intersections; "
                    << "check for degenerate face " << fId << "on mesh " << meshId << "." );
        interpen = false;
        return DEGENERATE_OVERLAP;
      }

      // call segment-to-plane intersection routine
      if ( k < 2 )  // we haven't found both intersection points yet
      {
        bool inter = LinePlaneIntersection( x1, y1, z1, x2, y2, z2, m_cX, m_cY, m_cZ, m_nX, m_nY, m_nZ,
                                            xInter[2 * i + k], yInter[2 * i + k], zInter[2 * i + k], inPlane );

        if ( inter ) ++k;
      }

      // we now have the two vertices for the ith face that represent segment-plane intersections.
      // Now determine the existing current configuration face vertices that lie
      // "on the other side" of the contact plane.
      real vX = x1 - m_cX;
      real vY = y1 - m_cY;
      real vZ = z1 - m_cZ;

      // project the vector onto the contact plane normal
      real proj = vX * m_nX + vY * m_nY + vZ * m_nZ;

      // if the projection for face 1 vertices is positive then that vertex crosses
      // (i.e. interpenetrates) the contact plane. if the projection for face 2 vertices
      // is negative then that vertex crosses the contact plane
      interpenVertex[ja] = ( i == 0 && proj > 0 ) ? ja : -1;
      interpenVertex[ja] = ( i == 1 && proj < 0 ) ? ja : interpenVertex[ja];

    }  // end loop over nodes

    // if we haven't found intersection points, the planes are either separated or coplanar.
    // return

    if ( k < 2 ) {
      interpen = false;
      return NO_FACE_GEOM_ERROR;
    }

    // count the number of vertices for the clipped portion of the i^th face that
    // interpenetrates the contact plane.
    numV[i] = k;
    for ( int vid = 0; vid < mesh.m_numNodesPerCell; ++vid ) {
      if ( interpenVertex[vid] == vid ) ++numV[i];

      // populate the face specific id array
      if ( i == 0 ) {
        interpenVertex1[vid] = interpenVertex[vid];
      } else {
        interpenVertex2[vid] = interpenVertex[vid];
      }
    }

  }  // end loop over faces

  // allocate arrays to store new clipped vertices for the current face
  real cfx1[numV[0]];  // cfx = clipped face x-coordinate
  real cfy1[numV[0]];
  real cfz1[numV[0]];

  real cfx2[numV[1]];  // cfx = clipped face x-coordinate
  real cfy2[numV[1]];
  real cfz2[numV[1]];

  // populate arrays
  for ( int m = 0; m < 2; ++m )  // populate segment-contact-plane intersection vertices
  {
    cfx1[m] = xInter[m];
    cfy1[m] = yInter[m];
    cfz1[m] = zInter[m];

    cfx2[m] = xInter[2 + m];
    cfy2[m] = yInter[2 + m];
    cfz2[m] = zInter[2 + m];
  }

  // populate the face 1 vertices that cross the contact plane
  int k = 2;
  for ( int m = 0; m < mesh1.m_numNodesPerCell; ++m ) {
    if ( interpenVertex1[m] != -1 ) {
      int fNodeId = mesh1.getFaceNodeId( m_pair.pairIndex1, interpenVertex1[m] );
      cfx1[k] = mesh1.m_positionX[fNodeId];
      cfy1[k] = mesh1.m_positionY[fNodeId];
      cfz1[k] = mesh1.m_positionZ[fNodeId];
      ++k;
    }
  }

  // populate the face 2 vertices that cross the contact plane
  k = 2;
  for ( int m = 0; m < mesh2.m_numNodesPerCell; ++m ) {
    if ( interpenVertex2[m] != -1 ) {
      int fNodeId = mesh2.getFaceNodeId( m_pair.pairIndex2, interpenVertex2[m] );
      cfx2[k] = mesh2.m_positionX[fNodeId];
      cfy2[k] = mesh2.m_positionY[fNodeId];
      cfz2[k] = mesh2.m_positionZ[fNodeId];
      ++k;
    }
  }

  // declare projected coordinate arrays
  real cfx1_proj[numV[0]];
  real cfy1_proj[numV[0]];
  real cfz1_proj[numV[0]];

  real cfx2_proj[numV[1]];
  real cfy2_proj[numV[1]];
  real cfz2_proj[numV[1]];

  // project clipped face coordinates to contact plane
  for ( int i = 0; i < numV[0]; ++i ) {
    ProjectPointToPlane( cfx1[i], cfy1[i], cfz1[i], m_nX, m_nY, m_nZ, m_cX, m_cY, m_cZ, cfx1_proj[i], cfy1_proj[i],
                         cfz1_proj[i] );
  }

  for ( int i = 0; i < numV[1]; ++i ) {
    ProjectPointToPlane( cfx2[i], cfy2[i], cfz2[i], m_nX, m_nY, m_nZ, m_cX, m_cY, m_cZ, cfx2_proj[i], cfy2_proj[i],
                         cfz2_proj[i] );
  }

  // declare local coordinate pointers
  real cfx1_loc[numV[0]];
  real cfy1_loc[numV[0]];

  real cfx2_loc[numV[1]];
  real cfy2_loc[numV[1]];

  // convert global coords to local contact plane coordinates
  GlobalTo2DLocalCoords( cfx1_proj, cfy1_proj, cfz1_proj, m_e1X, m_e1Y, m_e1Z, m_e2X, m_e2Y, m_e2Z, m_cX, m_cY, m_cZ,
                         cfx1_loc, cfy1_loc, numV[0] );

  GlobalTo2DLocalCoords( cfx2_proj, cfy2_proj, cfz2_proj, m_e1X, m_e1Y, m_e1Z, m_e2X, m_e2Y, m_e2Z, m_cX, m_cY, m_cZ,
                         cfx2_loc, cfy2_loc, numV[1] );

  // reorder potentially unordered set of vertices
  PolyReorder( cfx1_loc, cfy1_loc, numV[0] );
  PolyReorder( cfx2_loc, cfy2_loc, numV[1] );

  // call intersection routine to get intersecting polygon
  real pos_tol = parameters.len_collapse_ratio *
                 axom::utilities::max( mesh1.m_faceRadius[m_pair.pairIndex1], mesh2.m_faceRadius[m_pair.pairIndex2] );
  real len_tol = pos_tol;
  FaceGeomError inter_err =
      Intersection2DPolygon( cfx1_loc, cfy1_loc, numV[0], cfx2_loc, cfy2_loc, numV[1], pos_tol, len_tol, &m_polyLocX,
                             &m_polyLocY, m_numPolyVert, m_interpenArea, true );

  if ( inter_err != NO_FACE_GEOM_ERROR ) {
    interpen = false;
    return inter_err;
  }

  // store the local intersection polygons on the contact plane object,
  // primarily for visualization
  m_interpenPoly1X = new real[numV[0]];
  m_interpenPoly1Y = new real[numV[0]];
  m_interpenPoly2X = new real[numV[1]];
  m_interpenPoly2Y = new real[numV[1]];

  m_numInterpenPoly1Vert = numV[0];
  m_numInterpenPoly2Vert = numV[1];

  for ( int i = 0; i < numV[0]; ++i ) {
    m_interpenPoly1X[i] = cfx1_loc[i];
    m_interpenPoly1Y[i] = cfy1_loc[i];
  }

  for ( int i = 0; i < numV[1]; ++i ) {
    m_interpenPoly2X[i] = cfx2_loc[i];
    m_interpenPoly2Y[i] = cfy2_loc[i];
  }

  interpen = true;
  return NO_FACE_GEOM_ERROR;

}  // end ContactPlane3D::computeLocalInterpenOverlap()

//------------------------------------------------------------------------------
void ContactPlane3D::copyContactPlane( ContactPlane* RESTRICT cPlane )
{
  ContactPlane3D* cp = dynamic_cast<ContactPlane3D*>( cPlane );

  int meshId1 = cp->getCpMeshId( 1 );
  int meshId2 = cp->getCpMeshId( 2 );

  cp->setCpMeshId( 1, meshId1 );
  cp->setCpMeshId( 2, meshId2 );

  int fId1 = cp->getCpFaceId( 1 );
  int fId2 = cp->getCpFaceId( 2 );

  cp->setCpFaceId( 1, fId1 );
  cp->setCpFaceId( 2, fId2 );

  int dim = cp->getDim();
  int numFaces = cp->getCpNumFaces();

  cp->setCpNumFaces( numFaces );
  cp->setDim( dim );

  // set all other member variables
  m_inContact = cp->m_inContact;
  m_interpenOverlap = cp->m_interpenOverlap;

  m_cX = cp->m_cX;
  m_cY = cp->m_cY;
  m_cZ = cp->m_cZ;

  m_cXf1 = cp->m_cXf1;
  m_cYf1 = cp->m_cYf1;
  m_cZf1 = cp->m_cZf1;

  m_cXf2 = cp->m_cXf2;
  m_cYf2 = cp->m_cYf2;
  m_cZf2 = cp->m_cZf2;

  m_e1X = cp->m_e1X;
  m_e1Y = cp->m_e1Y;
  m_e1Z = cp->m_e1Z;

  m_e2X = cp->m_e2X;
  m_e2Y = cp->m_e2Y;
  m_e2Z = cp->m_e2Z;

  m_nX = cp->m_nX;
  m_nY = cp->m_nY;
  m_nZ = cp->m_nZ;

  m_numPolyVert = cp->m_numPolyVert;

  m_numInterpenPoly1Vert = cp->m_numInterpenPoly1Vert;
  m_numInterpenPoly2Vert = cp->m_numInterpenPoly2Vert;

  m_overlapCX = cp->m_overlapCX;
  m_overlapCY = cp->m_overlapCY;

  m_gap = cp->m_gap;
  m_gapTol = cp->m_gapTol;

  m_areaFrac = cp->m_areaFrac;
  m_areaMin = cp->m_areaMin;
  m_area = cp->m_area;
  m_interpenArea = cp->m_interpenArea;

  // allocate new memory for pointers and copy entries
  if ( cp->m_polyLocX != nullptr && cp->m_polyLocY != nullptr ) {
    m_polyLocX = new real[m_numPolyVert];
    m_polyLocY = new real[m_numPolyVert];
  }

  if ( cp->m_polyX != nullptr ) m_polyX = new real[m_numPolyVert];
  if ( cp->m_polyY != nullptr ) m_polyY = new real[m_numPolyVert];
  if ( cp->m_polyZ != nullptr ) m_polyZ = new real[m_numPolyVert];

  for ( int i = 0; i < m_numPolyVert; ++i ) {
    if ( m_polyLocX != nullptr ) m_polyLocX[i] = cp->m_polyLocX[i];
    if ( m_polyLocY != nullptr ) m_polyLocY[i] = cp->m_polyLocY[i];

    if ( m_polyX != nullptr ) m_polyX[i] = cp->m_polyX[i];
    if ( m_polyY != nullptr ) m_polyY[i] = cp->m_polyY[i];
    if ( m_polyZ != nullptr ) m_polyZ[i] = cp->m_polyZ[i];
  }

  if ( cp->m_interpenPoly1X != nullptr && cp->m_interpenPoly1Y != nullptr ) {
    m_interpenPoly1X = new real[m_numInterpenPoly1Vert];
    m_interpenPoly1Y = new real[m_numInterpenPoly1Vert];

    for ( int i = 0; i < m_numInterpenPoly1Vert; ++i ) {
      m_interpenPoly1X[i] = cp->m_interpenPoly1X[i];
      m_interpenPoly1Y[i] = cp->m_interpenPoly1Y[i];
    }
  }

  if ( cp->m_interpenPoly2X != nullptr && cp->m_interpenPoly2Y != nullptr ) {
    m_interpenPoly2X = new real[m_numInterpenPoly2Vert];
    m_interpenPoly2Y = new real[m_numInterpenPoly2Vert];

    for ( int i = 0; i < m_numInterpenPoly2Vert; ++i ) {
      m_interpenPoly2X[i] = cp->m_interpenPoly2X[i];
      m_interpenPoly2Y[i] = cp->m_interpenPoly2Y[i];
    }
  }

  if ( cp->m_interpenG1X != nullptr && cp->m_interpenG1Y != nullptr && cp->m_interpenG1Z != nullptr ) {
    m_interpenG1X = new real[m_numInterpenPoly1Vert];
    m_interpenG1Y = new real[m_numInterpenPoly1Vert];
    m_interpenG1Z = new real[m_numInterpenPoly1Vert];

    for ( int i = 0; i < m_numInterpenPoly1Vert; ++i ) {
      m_interpenG1X[i] = cp->m_interpenG1X[i];
      m_interpenG1Y[i] = cp->m_interpenG1Y[i];
      m_interpenG1Z[i] = cp->m_interpenG1Z[i];
    }
  }

  if ( cp->m_interpenG2X != nullptr && cp->m_interpenG2Y != nullptr && cp->m_interpenG2Z != nullptr ) {
    m_interpenG2X = new real[m_numInterpenPoly2Vert];
    m_interpenG2Y = new real[m_numInterpenPoly2Vert];
    m_interpenG2Z = new real[m_numInterpenPoly2Vert];

    for ( int i = 0; i < m_numInterpenPoly2Vert; ++i ) {
      m_interpenG2X[i] = cp->m_interpenG2X[i];
      m_interpenG2Y[i] = cp->m_interpenG2Y[i];
      m_interpenG2Z[i] = cp->m_interpenG2Z[i];
    }
  }

  return;

}  // end ContactPlane3D::copyContactPlane()

//-----------------------------------------------------------------------------
// Contact Plane 2D routines
//-----------------------------------------------------------------------------
ContactPlane2D::ContactPlane2D( InterfacePair& pair, real lenFrac, bool interpenOverlap, bool interPlane,
                                int dimension )
{
  m_numFaces = 0;
  dim = dimension;

  m_pair.meshId1 = pair.meshId1;
  m_pair.pairType1 = pair.pairType1;
  m_pair.pairIndex1 = pair.pairIndex1;

  m_pair.meshId2 = pair.meshId2;
  m_pair.pairType2 = pair.pairType2;
  m_pair.pairIndex2 = pair.pairIndex2;
  m_pair.pairId = pair.pairId;

  m_pair.isContactCandidate = pair.isContactCandidate;

  m_inContact = false;
  m_interpenOverlap = interpenOverlap;
  m_intermediatePlane = ( interPlane ) ? true : false;

  m_cX = 0.0;
  m_cY = 0.0;
  m_cZ = 0.0;

  m_cXf1 = 0.0;
  m_cYf1 = 0.0;
  m_cZf1 = 0.0;

  m_cXf2 = 0.0;
  m_cYf2 = 0.0;
  m_cZf2 = 0.0;

  m_nX = 0.0;
  m_nY = 0.0;
  m_nZ = 0.0;

  m_numInterpenPoly1Vert = 0;
  m_interpenG1X = nullptr;
  m_interpenG1Y = nullptr;
  m_interpenG1Z = nullptr;

  m_numInterpenPoly2Vert = 0;
  m_interpenG2X = nullptr;
  m_interpenG2Y = nullptr;
  m_interpenG2Z = nullptr;

  m_gap = 0.0;
  m_gapTol = 0.0;

  m_areaFrac = lenFrac;
  m_areaMin = 0.0;
  m_area = 0.0;
  m_interpenArea = 0.0;

  for ( int i = 0; i < 2; ++i ) {
    m_segX[i] = 0.0;
    m_segY[i] = 0.0;
  }

  return;

}  // end ContactPlane2D::ContactPlane2D()

//------------------------------------------------------------------------------
ContactPlane2D::ContactPlane2D()
{
  m_numFaces = 0;
  dim = -1;

  m_inContact = false;
  m_interpenOverlap = false;

  m_cX = 0.0;
  m_cY = 0.0;
  m_cZ = 0.0;

  m_cXf1 = 0.0;
  m_cYf1 = 0.0;
  m_cZf1 = 0.0;

  m_cXf2 = 0.0;
  m_cYf2 = 0.0;
  m_cZf2 = 0.0;

  m_nX = 0.0;
  m_nY = 0.0;
  m_nZ = 0.0;

  m_numInterpenPoly1Vert = 0;
  m_interpenG1X = nullptr;
  m_interpenG1Y = nullptr;
  m_interpenG1Z = nullptr;

  m_numInterpenPoly2Vert = 0;
  m_interpenG2X = nullptr;
  m_interpenG2Y = nullptr;
  m_interpenG2Z = nullptr;

  m_gap = 0.0;
  m_gapTol = 0.0;

  m_areaFrac = 0.0;
  m_areaMin = 0.0;
  m_area = 0.0;
  m_interpenArea = 0.0;

  for ( int i = 0; i < 2; ++i ) {
    m_segX[i] = 0.0;
    m_segY[i] = 0.0;
  }

  return;

}  // end ContactPlane2D::ContactPlane2D()

//------------------------------------------------------------------------------
FaceGeomError CheckEdgePair( InterfacePair& pair, bool fullOverlap, ContactPlane2D& cp )
{
  // Note: Checks #1-#4 are done in the binning

  // get instance of global parameters
  parameters_t& params = parameters_t::getInstance();

  // alias variables off the InterfacePair
  IndexType& meshId1 = pair.meshId1;
  IndexType& meshId2 = pair.meshId2;
  IndexType& edgeId1 = pair.pairIndex1;
  IndexType& edgeId2 = pair.pairIndex2;

  // get instance of mesh manager
  MeshManager& meshManager = MeshManager::getInstance();

  // get instance of mesh data
  MeshData& mesh1 = meshManager.GetMeshInstance( meshId1 );
  MeshData& mesh2 = meshManager.GetMeshInstance( meshId2 );

  // instantiate temporary contact plane to be returned by this routine
  bool interpenOverlap = ( !fullOverlap ) ? true : false;

  // CHECK #5: check if edge2 vertices have passed through edge1, and
  // vice-versa. If both vertices have done so for either edge, we trigger
  // a fullOverlap computation, but we don't know if there is a positive
  // length of overlap and will have to construct the contact plane
  // (contact segment) and perform this check. Note, this tolerance is
  // inclusive up to a separation of the edge-radius.
  // This is done for the mortar method per 3D testing.
  real separationTol =
      params.gap_separation_ratio * axom::utilities::max( mesh1.m_faceRadius[edgeId1], mesh2.m_faceRadius[edgeId2] );
  bool all = false;
  bool ls = EdgeInterCheck( mesh1, mesh2, edgeId1, edgeId2, separationTol, all );
  if ( !ls ) {
    cp.m_inContact = false;
    return NO_FACE_GEOM_ERROR;
  }

  // if all the vertices lie on the other side of edge1, then use full
  // projection
  if ( all ) {
    fullOverlap = true;
    interpenOverlap = false;
    cp.m_interpenOverlap = interpenOverlap;
  }

  // CHECK #6: compute the projected length of overlap on the contact plane.
  // At this point the edges are proximate and likely have a positive
  // projected length of overlap.

  cp.computeNormal();
  cp.computePlanePoint();

  // project each edge's nodes onto the contact segment.
  real projX1[mesh1.m_numNodesPerCell];
  real projY1[mesh1.m_numNodesPerCell];
  real projX2[mesh2.m_numNodesPerCell];
  real projY2[mesh2.m_numNodesPerCell];

  ProjectEdgeNodesToSegment( mesh1, edgeId1, cp.m_nX, cp.m_nY, cp.m_cX, cp.m_cY, &projX1[0], &projY1[0] );
  ProjectEdgeNodesToSegment( mesh2, edgeId2, cp.m_nX, cp.m_nY, cp.m_cX, cp.m_cY, &projX2[0], &projY2[0] );

  // compute the full overlap. Even if we are using the interpenetration
  // overlap, we have to compute the full overlap in order to properly
  // locate the contact plane (segment) for the interpenetration calculation
  cp.checkSegOverlap( &projX1[0], &projY1[0], &projX2[0], &projY2[0], mesh1.m_numNodesPerCell,
                      mesh2.m_numNodesPerCell );

  // compute the overlap length tolerance
  cp.computeAreaTol();

  // check the contact plane length against the minimum length.
  // In general the interpen length is going to be less than
  // the full overlap length so we can do this check prior to
  // any interpenetration overlap calculation
  if ( cp.m_area < cp.m_areaMin ) {
    cp.m_inContact = false;
    return NO_FACE_GEOM_ERROR;
  }

  if ( interpenOverlap ) {
    // properly locate the contact plane (segment)
    cp.planePointAndCentroidGap( 2. *
                                 axom::utilities::max( mesh1.m_faceRadius[edgeId1], mesh2.m_faceRadius[edgeId2] ) );
    bool interpen = false;
    FaceGeomError interpen_err = cp.computeLocalInterpenOverlap( interpen );
    if ( interpen_err != NO_FACE_GEOM_ERROR ) {
      cp.m_inContact = false;
      return interpen_err;
    } else if ( !interpen ) {
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
  cp.planePointAndCentroidGap( 2. * axom::utilities::max( mesh1.m_faceRadius[edgeId1], mesh2.m_faceRadius[edgeId2] ) );

  // Per 3D mortar testing, allow for separation up to the edge-radius
  cp.m_gapTol =
      params.gap_separation_ratio * axom::utilities::max( mesh1.m_faceRadius[edgeId1], mesh2.m_faceRadius[edgeId2] );
  if ( cp.m_gap > cp.m_gapTol ) {
    cp.m_inContact = false;
    return NO_FACE_GEOM_ERROR;
  }

  // for the full overlap case we need to project the overlap segment
  // onto the updated contact plane
  if ( fullOverlap ) {
    // allocate dummy space for the interpen topology so adding the
    // contact plane to the contact plane manager doesn't seg fault.
    // Fix this later...
    cp.m_numInterpenPoly1Vert = 2;
    cp.m_numInterpenPoly2Vert = 2;
    cp.m_interpenG1X = new real[cp.m_numInterpenPoly1Vert];
    cp.m_interpenG1Y = new real[cp.m_numInterpenPoly1Vert];
    cp.m_interpenG2X = new real[cp.m_numInterpenPoly2Vert];
    cp.m_interpenG2Y = new real[cp.m_numInterpenPoly2Vert];

    cp.m_interpenG1Z = nullptr;
    cp.m_interpenG2Z = nullptr;

    for ( int i = 0; i < 2; ++i ) {
      real xproj, yproj;
      ProjectPointToSegment( cp.m_segX[i], cp.m_segY[i], cp.m_nX, cp.m_nY, cp.m_cX, cp.m_cY, xproj, yproj );
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

}  // end CheckEdgePair()

//------------------------------------------------------------------------------
void ContactPlane2D::computeNormal()
{
  // get the mesh data
  MeshData& m1 = getCpMeshData( m_pair.meshId1 );
  MeshData& m2 = getCpMeshData( m_pair.meshId2 );

  if ( m_intermediatePlane ) {
    // COMMON_PLANE normal calculation:
    // compute the cp normal as the average of the two face normals, and in
    // the direction such that the dot product between the cp normal and
    // the normal of face 2 is positive.
    m_nX = 0.5 * ( m2.m_nX[m_pair.pairIndex2] - m1.m_nX[m_pair.pairIndex1] );
    m_nY = 0.5 * ( m2.m_nY[m_pair.pairIndex2] - m1.m_nY[m_pair.pairIndex1] );
    m_nZ = 0.0;  // zero out the third component of the normal
  } else {
    // MORTAR normal calculation. This is the normal of the nonmortar surface
    m_nX = m2.m_nX[m_pair.pairIndex2];
    m_nY = m2.m_nY[m_pair.pairIndex2];
    m_nZ = 0.;
  }

  // normalize the cp normal
  real mag;
  real invMag;

  mag = magnitude( m_nX, m_nY );
  invMag = 1.0 / mag;

  m_nX *= invMag;
  m_nY *= invMag;

  return;

}  // end ContactPlane2D::computeNormal()

//------------------------------------------------------------------------------
void ContactPlane2D::computePlanePoint()
{
  // get the mesh data
  MeshData& m1 = getCpMeshData( m_pair.meshId1 );
  MeshData& m2 = getCpMeshData( m_pair.meshId2 );

  // compute the cp centroid as the average of
  // the two face's centers. This is the default
  // method of compute the cp centroid
  m_cX = 0.5 * ( m1.m_cX[m_pair.pairIndex1] + m2.m_cX[m_pair.pairIndex2] );
  m_cY = 0.5 * ( m1.m_cY[m_pair.pairIndex1] + m2.m_cY[m_pair.pairIndex2] );
  m_cZ = 0.0;
  return;

}  // end ContactPlane2D::computePlanePoint()

//------------------------------------------------------------------------------
void ContactPlane2D::computeIntegralGap()
{
  // TODO implement this routine
  // This will be contact method dependent
  return;

}  // end ContactPlane2D::computeIntegralGap()

//------------------------------------------------------------------------------
void ContactPlane2D::computeAreaTol()
{
  // Note: this code is the same as for ContactPlane3D, but maintain separate
  // routine for 2D treatment.
  parameters_t& parameters = parameters_t::getInstance();

  if ( m_areaFrac < parameters.overlap_area_frac ) {
    SLIC_DEBUG( "ContactPlane2D::computeAreaTol() the overlap area fraction too small or negative; "
                << "setting to overlap_area_frac parameter." );
    m_areaFrac = parameters.overlap_area_frac;
  }

  MeshData& mesh1 = getCpMeshData( m_pair.meshId1 );
  MeshData& mesh2 = getCpMeshData( m_pair.meshId2 );

  m_areaMin = m_areaFrac * axom::utilities::min( mesh1.m_area[m_pair.pairIndex1], mesh2.m_area[m_pair.pairIndex2] );
  return;

}  // ContactPlane2D::computeAreaTol()

//------------------------------------------------------------------------------
FaceGeomError ContactPlane2D::computeLocalInterpenOverlap( bool& interpen )
{
  //
  // Note: the contact plane has to be properly located prior to calling
  // this routine.
  //

  interpen = false;
  parameters_t& parameters = parameters_t::getInstance();

  // all edge-edge interactions suitable for an interpenetration overlap
  // calculation are edges that intersect at a single point
  MeshData& mesh1 = getCpMeshData( getCpMeshId( 1 ) );
  MeshData& mesh2 = getCpMeshData( getCpMeshId( 2 ) );
  int edgeId1 = getCpFaceId( 1 );
  int edgeId2 = getCpFaceId( 2 );
  int nodeA1 = mesh1.getFaceNodeId( edgeId1, 0 );
  int nodeB1 = mesh1.getFaceNodeId( edgeId1, 1 );
  int nodeA2 = mesh2.getFaceNodeId( edgeId2, 0 );
  int nodeB2 = mesh2.getFaceNodeId( edgeId2, 1 );

  real xposA1 = mesh1.m_positionX[nodeA1];
  real yposA1 = mesh1.m_positionY[nodeA1];
  real xposB1 = mesh1.m_positionX[nodeB1];
  real yposB1 = mesh1.m_positionY[nodeB1];

  real xposA2 = mesh2.m_positionX[nodeA2];
  real yposA2 = mesh2.m_positionY[nodeA2];
  real xposB2 = mesh2.m_positionX[nodeB2];
  real yposB2 = mesh2.m_positionY[nodeB2];

  real xInter, yInter;
  bool duplicatePoint = false;

  // check if the segments intersect
  real len_tol =
      parameters.len_collapse_ratio * axom::utilities::max( mesh1.m_faceRadius[edgeId1], mesh2.m_faceRadius[edgeId2] );

  bool edgeIntersect = SegmentIntersection2D( xposA1, yposA1, xposB1, yposB1, xposA2, yposA2, xposB2, yposB2, nullptr,
                                              xInter, yInter, duplicatePoint, len_tol );

  // check to make sure the edges are actually intersecting. Note
  // that an intersection point within the specified tolerance of
  // an edge vertex is collapsed to that vertex point and duplicatePoint
  // is marked true, but the SegmentIntersection2D returns false
  if ( !edgeIntersect && !duplicatePoint ) {
    m_interpenArea = 0.0;
    interpen = false;
    return NO_FACE_GEOM_ERROR;
  }

  // check if a duplicate point (i.e. vertex) was registered.
  // That is, if the intersection point is at an edge vertex,
  // in which case we don't register the interaction
  if ( duplicatePoint ) {
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
  real xInterProj, yInterProj;
  ProjectPointToSegment( xInter, yInter, m_nX, m_nY, m_cX, m_cY, xInterProj, yInterProj );

  // now isolate which vertex on edge 1 and which vertex on edge 2 lie
  // on the "wrong" side of the contact plane.

  // define vectors between an edge vertex and the contact plane centroid;
  int interId1 = -1;
  int interId2 = -1;
  int k = 0;
  for ( int i = 0; i < mesh1.m_numNodesPerCell; ++i ) {
    int nodeId1 = mesh1.getFaceNodeId( edgeId1, i );
    int nodeId2 = mesh2.getFaceNodeId( edgeId2, i );
    real lvx1 = mesh1.m_positionX[nodeId1] - m_cX;
    real lvy1 = mesh1.m_positionY[nodeId1] - m_cY;
    real lvx2 = mesh2.m_positionX[nodeId2] - m_cX;
    real lvy2 = mesh2.m_positionY[nodeId2] - m_cY;

    // dot each vector with the contact plane normal
    real proj1 = lvx1 * m_nX + lvy1 * m_nY;
    real proj2 = lvx2 * m_nX + lvy2 * m_nY;

    // check the projection to detect interpenetration and
    // mark the node id if true
    if ( proj1 > 0.0 ) {
      interId1 = i;
      ++k;
    }
    if ( proj2 < 0.0 ) {
      interId2 = i;
      ++k;
    }
  }

  // Debug check the number of interpenetrating vertices
  if ( k > 2 ) {
    SLIC_DEBUG( "ContactPlane2D::computeLocalInterpenOverlap() more than 2 interpenetrating vertices detected; "
                << "check for degenerate geometry for edges (" << edgeId1 << ", " << edgeId2 << ") on meshes ("
                << mesh1.m_meshId << ", " << mesh2.m_meshId << ")." );
    interpen = false;
    return DEGENERATE_OVERLAP;
  }

  // now that we have marked the interpenetrating vertex of each edge,
  // compute the distance between the interpenetrating vertex and the
  // edge intersection point
  int nodeInter1 = mesh1.getFaceNodeId( edgeId1, interId1 );
  int nodeInter2 = mesh2.getFaceNodeId( edgeId2, interId2 );

  real vix1 = mesh1.m_positionX[nodeInter1] - xInterProj;
  real viy1 = mesh1.m_positionY[nodeInter1] - yInterProj;
  real vix2 = mesh2.m_positionX[nodeInter2] - xInterProj;
  real viy2 = mesh2.m_positionY[nodeInter2] - yInterProj;

  // determine magnitude of each vector
  real mag1 = magnitude( vix1, viy1 );
  real mag2 = magnitude( vix2, viy2 );

  // the interpenetration overlap length is the minimum of the above
  // vectors
  m_interpenArea = ( mag1 <= mag2 ) ? mag1 : mag2;

  if ( m_interpenArea > m_areaMin ) {
    // determine the edge vertex that forms the overlap segment along
    // with the intersection point previously computed
    real vx1 = ( mag1 <= mag2 ) ? mesh1.m_positionX[nodeInter1] : mesh2.m_positionX[nodeInter2];

    real vy1 = ( mag1 <= mag2 ) ? mesh1.m_positionY[nodeInter1] : mesh2.m_positionY[nodeInter2];

    real vx2 = xInterProj;
    real vy2 = yInterProj;

    // allocate space to store the interpen vertices for visualization
    // (stored on contact plane base class)
    m_numInterpenPoly1Vert = 2;
    m_numInterpenPoly2Vert = 2;
    m_interpenG1X = new real[2];
    m_interpenG1Y = new real[2];
    m_interpenG2X = new real[2];
    m_interpenG2Y = new real[2];

    m_interpenG1Z = nullptr;
    m_interpenG2Z = nullptr;

    m_interpenG1X[0] = vix1;
    m_interpenG1Y[0] = viy1;
    m_interpenG1X[1] = xInter;
    m_interpenG1Y[1] = yInter;

    m_interpenG2X[0] = vix2;
    m_interpenG2Y[0] = viy2;
    m_interpenG2X[1] = xInter;
    m_interpenG2Y[1] = yInter;

    // project these points to the contact plane
    ProjectPointToSegment( vx1, vy1, m_nX, m_nY, m_cX, m_cY, m_segX[0], m_segY[0] );

    ProjectPointToSegment( vx2, vy2, m_nX, m_nY, m_cX, m_cY, m_segX[1], m_segY[1] );

    // compute the new contact plane overlap centroid (segment point)
    m_cX = 0.5 * ( m_segX[0] + m_segX[1] );
    m_cY = 0.5 * ( m_segY[0] + m_segY[1] );

    interpen = true;
    return NO_FACE_GEOM_ERROR;
  }

  m_interpenG1X = nullptr;
  m_interpenG1Y = nullptr;
  m_interpenG2X = nullptr;
  m_interpenG2Y = nullptr;
  m_interpenG1Z = nullptr;
  m_interpenG2Z = nullptr;

  interpen = false;
  return NO_FACE_GEOM_ERROR;

}  // end ContactPlane2D::computeLocalInterpenOverlap()

//------------------------------------------------------------------------------
void ContactPlane2D::centroidGap( real scale )
{
  // project the overlap centroid, which is taken to be the point data
  // (i.e. centroid) of the contact plane, back to each edge using the
  // line-plane intersection method where each edge is imagined to be
  // within a plane defined by the edge's centroid and normal
  real xc1 = 0.;
  real yc1 = 0.;
  real zc1 = 0.;
  real xc2 = 0.;
  real yc2 = 0.;
  real zc2 = 0.;

  // find where the overlap centroid (plane point) intersects each face.
  // The following variables store the end vertices of
  // a fictitious line co-directional with the contact plane normal
  // passing through each edge

  // set the line segment's first vertex at the contact plane centroid,
  // scaled in the direction opposite the contact plane normal
  real xA = m_cX + m_nX * scale;
  real yA = m_cY + m_nY * scale;
  real zA = 0.0;

  // use the contact plane normal as the directional vector scale
  // in the direction of the contact plane
  real xB = m_cX - m_nX * scale;
  real yB = m_cY - m_nY * scale;
  real zB = 0.0;

  // get mesh data
  MeshData& m1 = getCpMeshData( m_pair.meshId1 );
  MeshData& m2 = getCpMeshData( m_pair.meshId2 );
  bool inPlane = false;
  IndexType fId1 = m_pair.pairIndex1;
  IndexType fId2 = m_pair.pairIndex2;
  bool intersect1 = LinePlaneIntersection( xA, yA, zA, xB, yB, zB, m1.m_cX[fId1], m1.m_cY[fId1], 0.0, m1.m_nX[fId1],
                                           m1.m_nY[fId1], 0.0, xc1, yc1, zc1, inPlane );

  bool intersect2 = LinePlaneIntersection( xA, yA, zA, xB, yB, zB, m2.m_cX[fId2], m2.m_cY[fId2], 0.0, m2.m_nX[fId2],
                                           m2.m_nY[fId2], 0.0, xc2, yc2, zc2, inPlane );
  TRIBOL_UNUSED_VAR( intersect1 );  // We don't currently use these bool variabeles
  TRIBOL_UNUSED_VAR( intersect2 );  // but the above function calls modify some parameters

  // compute the normal gap magnitude (x1 - x2 for positive gap in separation
  // and negative gap in penetration).
  m_gap = ( xc1 - xc2 ) * m_nX + ( yc1 - yc2 ) * m_nY;

  // store the two edge points corresponding to the contact plane centroid
  // projection/intersection
  m_cXf1 = xc1;
  m_cYf1 = yc1;
  m_cZf1 = 0.0;

  m_cXf2 = xc2;
  m_cYf2 = yc2;
  m_cZf2 = 0.0;

  return;

}  // end ContactPlane2D::centroidGap()

//------------------------------------------------------------------------------
void ContactPlane2D::planePointAndCentroidGap( real scale )
{
  // project the overlap centroid, which is taken to be the point data
  // (i.e. centroid) of the contact plane, back to each edge using the
  // line-plane intersection method where each edge is imagined to be
  // within a plane defined by the edge's centroid and normal
  real xc1 = 0.;
  real yc1 = 0.;
  real zc1 = 0.;
  real xc2 = 0.;
  real yc2 = 0.;
  real zc2 = 0.;

  // find where the overlap centroid (plane point) intersects each face.
  // The following variables store the end vertices of
  // a fictitious line co-directional with the contact plane normal
  // passing through each edge

  // set the line segment's first vertex at the contact plane centroid,
  // scaled in the direction opposite the contact plane normal
  real xA = m_cX + m_nX * scale;
  real yA = m_cY + m_nY * scale;
  real zA = 0.0;

  // use the contact plane normal as the directional vector scale
  // in the direction of the contact plane
  real xB = m_cX - m_nX * scale;
  real yB = m_cY - m_nY * scale;
  real zB = 0.0;

  // get mesh data
  MeshData& m1 = getCpMeshData( m_pair.meshId1 );
  MeshData& m2 = getCpMeshData( m_pair.meshId2 );
  bool inPlane = false;
  IndexType fId1 = m_pair.pairIndex1;
  IndexType fId2 = m_pair.pairIndex2;
  bool intersect1 = LinePlaneIntersection( xA, yA, zA, xB, yB, zB, m1.m_cX[fId1], m1.m_cY[fId1], 0.0, m1.m_nX[fId1],
                                           m1.m_nY[fId1], 0.0, xc1, yc1, zc1, inPlane );

  bool intersect2 = LinePlaneIntersection( xA, yA, zA, xB, yB, zB, m2.m_cX[fId2], m2.m_cY[fId2], 0.0, m2.m_nX[fId2],
                                           m2.m_nY[fId2], 0.0, xc2, yc2, zc2, inPlane );
  TRIBOL_UNUSED_VAR( intersect1 );  // We don't currently use these bool variabeles
  TRIBOL_UNUSED_VAR( intersect2 );  // but the above function calls modify some parameters

  // average the two contact plane centroid-to-plane intersections and
  // use this as the new point data for the contact plane (do not do
  // for mortar methods, or may be redundant).
  m_cX = 0.5 * ( xc1 + xc2 );
  m_cY = 0.5 * ( yc1 + yc2 );
  m_cZ = 0.0;

  // compute the normal gap magnitude (x1 - x2 for positive gap in separation
  // and negative gap in penetration).
  m_gap = ( xc1 - xc2 ) * m_nX + ( yc1 - yc2 ) * m_nY;

  // store the two edge points corresponding to the contact plane centroid
  // projection/intersection
  m_cXf1 = xc1;
  m_cYf1 = yc1;
  m_cZf1 = 0.0;

  m_cXf2 = xc2;
  m_cYf2 = yc2;
  m_cZf2 = 0.0;

  return;

}  // end ContactPlane2D::planePointAndCentroidGap()

//------------------------------------------------------------------------------
void ContactPlane2D::copyContactPlane( ContactPlane* RESTRICT cPlane )
{
  ContactPlane2D* cp = dynamic_cast<ContactPlane2D*>( cPlane );

  int meshId1 = cp->getCpMeshId( 1 );
  int meshId2 = cp->getCpMeshId( 2 );

  cp->setCpMeshId( 1, meshId1 );
  cp->setCpMeshId( 2, meshId2 );

  int fId1 = cp->getCpFaceId( 1 );
  int fId2 = cp->getCpFaceId( 2 );

  cp->setCpFaceId( 1, fId1 );
  cp->setCpFaceId( 2, fId2 );

  int dim = cp->getDim();
  int numFaces = cp->getCpNumFaces();

  cp->setCpNumFaces( numFaces );
  cp->setDim( dim );

  // set all other member variables
  m_inContact = cp->m_inContact;
  m_interpenOverlap = cp->m_interpenOverlap;

  m_cX = cp->m_cX;
  m_cY = cp->m_cY;
  m_cZ = cp->m_cZ;

  m_cXf1 = cp->m_cXf1;
  m_cYf1 = cp->m_cYf1;
  m_cZf1 = cp->m_cZf1;

  m_cXf2 = cp->m_cXf2;
  m_cYf2 = cp->m_cYf2;
  m_cZf2 = cp->m_cZf2;

  m_nX = cp->m_nX;
  m_nY = cp->m_nY;
  m_nZ = cp->m_nZ;

  m_numInterpenPoly1Vert = cp->m_numInterpenPoly1Vert;
  m_numInterpenPoly2Vert = cp->m_numInterpenPoly2Vert;

  for ( int i = 0; i < 2; ++i ) {
    m_segX[i] = cp->m_segX[i];
    m_segY[i] = cp->m_segY[i];
  }

  m_gap = cp->m_gap;
  m_gapTol = cp->m_gapTol;

  m_areaFrac = cp->m_areaFrac;
  m_areaMin = cp->m_areaMin;
  m_area = cp->m_area;
  m_interpenArea = cp->m_interpenArea;

  // allocate new memory for pointers and copy entries
  if ( cp->m_interpenG1X != nullptr && cp->m_interpenG1Y != nullptr && cp->m_interpenG1Z != nullptr ) {
    m_interpenG1X = new real[m_numInterpenPoly1Vert];
    m_interpenG1Y = new real[m_numInterpenPoly1Vert];
    m_interpenG1Z = new real[m_numInterpenPoly1Vert];

    for ( int i = 0; i < m_numInterpenPoly1Vert; ++i ) {
      m_interpenG1X[i] = cp->m_interpenG1X[i];
      m_interpenG1Y[i] = cp->m_interpenG1Y[i];
      m_interpenG1Z[i] = cp->m_interpenG1Z[i];
    }
  }

  if ( cp->m_interpenG2X != nullptr && cp->m_interpenG2Y != nullptr && cp->m_interpenG2Z != nullptr ) {
    m_interpenG2X = new real[m_numInterpenPoly2Vert];
    m_interpenG2Y = new real[m_numInterpenPoly2Vert];
    m_interpenG2Z = new real[m_numInterpenPoly2Vert];

    for ( int i = 0; i < m_numInterpenPoly2Vert; ++i ) {
      m_interpenG2X[i] = cp->m_interpenG2X[i];
      m_interpenG2Y[i] = cp->m_interpenG2Y[i];
      m_interpenG2Z[i] = cp->m_interpenG2Z[i];
    }
  }

  return;

}  // end ContactPlane2D::copyContactPlane()

//------------------------------------------------------------------------------
void ContactPlane2D::checkSegOverlap( const real* const RESTRICT pX1, const real* const RESTRICT pY1,
                                      const real* const RESTRICT pX2, const real* const RESTRICT pY2, const int nV1,
                                      const int nV2 )
{
  parameters_t& parameters = parameters_t::getInstance();

  SLIC_ASSERT( nV1 == 2 );
  SLIC_ASSERT( nV2 == 2 );

  // get mesh ids
  int mesh1Id = getCpMeshId( 1 );
  int mesh2Id = getCpMeshId( 2 );

  // get mesh data
  MeshData& mesh1 = getCpMeshData( mesh1Id );
  MeshData& mesh2 = getCpMeshData( mesh2Id );

  // get edge ids
  int e1Id = getCpFaceId( 1 );
  int e2Id = getCpFaceId( 2 );

  //
  // perform the all-in-1 check
  //

  // define the edge 1 non-unit directional vector between vertices
  // 2 and 1
  real lvx1 = pX1[1] - pX1[0];
  real lvy1 = pY1[1] - pY1[0];

  // compute vector between each edge 2 vertex and vertex 1 on edge 1.
  // Then dot that vector with the directional vector of edge 1 to see
  // if they are codirectional. If so, check, that this vector length is
  // less than edge 1 length indicating that the vertex lies within edge 1
  real projTol =
      parameters.projection_ratio * axom::utilities::max( mesh1.m_faceRadius[e1Id], mesh2.m_faceRadius[e2Id] );
  real vLenTol = projTol;
  int inter2 = 0;
  int twoInOneId = -1;
  for ( int i = 0; i < nV2; ++i ) {
    real vx = pX2[i] - pX1[0];
    real vy = pY2[i] - pY1[0];

    // compute projection onto edge 1 directional vector
    real proj = vx * lvx1 + vy * lvy1;

    // compute length of <vx,vy>; if vLen < some tolerance we have a
    // coincident node
    real vLen = magnitude( vx, vy );

    if ( vLen < vLenTol )  // coincident vertex
    {
      twoInOneId = i;
      ++inter2;
    } else if ( proj > projTol && vLen <= mesh1.m_area[e1Id] )  // interior vertex
    {
      twoInOneId = i;
      ++inter2;
    }
  }

  // if both vertices pass the above criteria than 2 is in 1
  if ( inter2 == 2 ) {
    // set the contact plane (segment) length
    m_area = mesh2.m_area[e2Id];

    // set the vertices of the overlap segment
    m_segX[0] = pX2[0];
    m_segY[0] = pY2[0];

    m_segX[1] = pX2[1];
    m_segY[1] = pY2[1];

    // relocate the centroid within the currently defined contact
    // segment
    m_cX = 0.5 * ( m_segX[0] + m_segX[1] );
    m_cY = 0.5 * ( m_segY[0] + m_segY[1] );
    m_cZ = 0.0;
    return;
  }

  //
  // perform the all-in-2 check
  //

  // define the edge 2 non-unit directional vector between vertices
  // 2 and 1
  real lvx2 = pX2[1] - pX2[0];
  real lvy2 = pY2[1] - pY2[0];

  // compute vector between each edge 1 vertex and vertex 1 on edge 2.
  // Then dot that vector with the directional vector of edge 2 to see
  // if they are codirectional. If so, check, that this vector length is
  // less than edge 2 length indicating that the vertex is within edge 2
  int inter1 = 0;
  int oneInTwoId = -1;
  for ( int i = 0; i < nV1; ++i ) {
    real vx = pX1[i] - pX2[0];
    real vy = pY1[i] - pY2[0];

    // compute projection onto edge 2 directional vector
    real proj = vx * lvx2 + vy * lvy2;

    // compute length of <vx,vy>
    real vLen = magnitude( vx, vy );

    if ( vLen < vLenTol )  // coincident vertex
    {
      oneInTwoId = i;
      ++inter1;
    } else if ( proj > projTol && vLen <= mesh2.m_area[e2Id] )  // interior vertex
    {
      oneInTwoId = i;
      ++inter1;
    }
  }

  // if both vertices pass the above criteria then 1 is in 2.
  if ( inter1 == 2 ) {
    // set the contact plane (segment) length
    m_area = mesh1.m_area[e1Id];

    // set the overlap segment vertices on the contact plane object
    m_segX[0] = pX1[0];
    m_segY[0] = pY1[0];

    m_segX[1] = pX1[1];
    m_segY[1] = pY1[1];

    // relocate the centroid within the currently defined contact
    // segment
    m_cX = 0.5 * ( m_segX[0] + m_segX[1] );
    m_cY = 0.5 * ( m_segY[0] + m_segY[1] );
    m_cZ = 0.0;
    return;
  }

  // if inter1 == 0 and inter2 == 0 then there is no overlap
  if ( inter1 == 0 && inter2 == 0 ) {
    m_area = 0.0;
    m_cX = m_cY = m_cZ = 0.0;
    return;
  }

  // there is a chance that oneInTowId or twoInOneId is not actually set,
  // in which case we don't have an overlap.
  if ( oneInTwoId == -1 || twoInOneId == -1 ) {
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
  m_segX[0] = pX1[oneInTwoId];
  m_segY[0] = pY1[oneInTwoId];
  m_segX[1] = pX2[twoInOneId];
  m_segY[1] = pY2[twoInOneId];

  // compute vector between "inter"-vertices
  real vecX = m_segX[1] - m_segX[0];
  real vecY = m_segY[1] - m_segY[0];

  // compute the length of the overlapping segment
  m_area = magnitude( vecX, vecY );

  // compute the overlap centroid
  m_cX = 0.5 * ( m_segX[0] + m_segX[1] );
  m_cY = 0.5 * ( m_segY[0] + m_segY[1] );
  m_cZ = 0.0;

  return;

}  // end ContactPlane2D::checkSegOverlap()

//------------------------------------------------------------------------------

}  // namespace tribol
