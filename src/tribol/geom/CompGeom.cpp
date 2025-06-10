// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "CompGeom.hpp"
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

namespace tribol {

//------------------------------------------------------------------------------
// free functions
//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError CheckInterfacePair( InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                     const MeshData::Viewer& mesh2, const Parameters& params,
                                                     ContactMethod cMethod, ContactCase TRIBOL_UNUSED_PARAM( cCase ),
                                                     bool& isInteracting, CompGeom::Viewer& cg, IndexT* plane_ct )
{
  isInteracting = false;
  inContact = false;
  FaceGeomError face_err;
  ContactPlane* my_plane;

  // note: will likely need the ContactCase for specialized
  // geometry check(s)/routine(s)

  switch ( cMethod ) {
    case MORTAR_WEIGHTS:
    case SINGLE_MORTAR: {
      MortarPlanePair mortar_plane( &pair, params, mesh1.spatialDimension() );
      face_err = mortar_plane.checkInterfacePair( mesh1, mesh2 ); // TODO SRW fix/write this routine
      inContact = mortar_plane.m_inContact;
      my_plane = &mortar_plane;
      break;
    }
    case COMMON_PLANE: {
      CommonPlanePair common_plane( &pair, params, mesh1.spatialDimension() );
      face_err = common_plane.checkInterfacePair( mesh1, mesh2 );
      inContact = common_plane.m_inContact;
      my_plane = &common_plane;
      break;
    }
    case ALIGNED_MORTAR: {
      AlignedMortarPlanePair aligned_mortar_plane(&pair, params, mesh1.spatialDimension() );
      face_err = aligned_mortar_plane.checkInterfacePair( mesh1, mesh2 );
      inContact = aligned_mortar_plane.m_inContact;
      my_plane = &aligned_mortar_plane;
      break;
    }
    default: {
      // don't do anything
      break;
    }
  } // end switch

  // check errors and contact status and add ContactPlanePair accordingly
  if ( face_err != NO_FACE_GEOM_ERROR ) {
    isInteracting = false;
#ifdef TRIBOL_USE_HOST
    SLIC_DEBUG( "face_err: " << face_err );
#endif
  } else if ( inContact ) {
#ifdef TRIBOL_USE_RAJA
    auto idx = RAJA::atomicInc<RAJA::auto_atomic>( plane_ct );
#else
    auto idx = *plane_ct;
    ++( *plane_ct );
#endif
    cg.addContactPlane( *my_plane, idx, cMethod );
    isInteracting = true;
  } else {
    isInteracting = false;
  }

  return face_err;

}  // end CheckInterfacePair()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE bool FullFaceCheck( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2, int fId1,
                                       int fId2 )
{
  // loop over vertices on face 2
  int k = 0;
  for ( int i = 0; i < mesh2.numberOfNodesPerElement(); ++i ) {
    // get ith face 2 node id
    const int f2NodeId = mesh2.getGlobalNodeId( fId2, i );

    // compute components of vector between face 1 center and face 2 vertex
    RealT vX = mesh1.getElementCentroids()[0][fId1] - mesh2.getPosition()[0][f2NodeId];
    RealT vY = mesh1.getElementCentroids()[1][fId1] - mesh2.getPosition()[1][f2NodeId];
    RealT vZ = mesh1.getElementCentroids()[2][fId1] - mesh2.getPosition()[2][f2NodeId];

    // project the vector onto face 1 normal
    RealT proj = vX * mesh1.getElementNormals()[0][fId1] + vY * mesh1.getElementNormals()[1][fId1] +
                 vZ * mesh1.getElementNormals()[2][fId1];

    // if a node of face 2 is on the other side of the plane defined by face 1 the
    // projection will be positive. If a node on face 2 lies on face 1 the projection
    // will be zero. If a node lies just outside of face 1 then the projection will
    // be a small negative number. Note: using a small negative number will result in
    // nodes with a very small amount of separation contributing to the node count,
    // which may result in a full overlap calculation. This should have a negligible
    // effect on the resulting overlap area and no negative affect on the contact
    // behavior.
    if ( proj >= 0. ) {
      ++k;
    }

  }  // end loop over nodes

  // check to see:
  // 1) all nodes are on the other side triggering a full overlap calc with gap contraint violation
  //    provided there is a positive area of overlap (checked elsewhere)
  // 2) zero nodes are on the other side, triggering a full overlap for a separation configuration
  //    in which case the gap constraint check in the physics will ensure no force contribution
  if ( k == mesh2.numberOfNodesPerElement() || k == 0 ) {
    return true;
  }

  // Added 10/19/18 by SRW - If we are here then:
  // 1) if some of face 2 nodes lie on the other side of the PLANE defined by face 1,
  //    then it is possible that all of the nodes on face 1 lie on the other side of the plane
  //    defined by face 2 (see check below).
  // 2) we still have to perform the check below to know if not all nodes on face 1 pass
  //    through the plane defined by face 2. This would indicate a face-intersection and an
  //    interpen overlap calculation is used, not a full overlap calculation

  // loop over vertices on face 1
  k = 0;
  for ( int i = 0; i < mesh1.numberOfNodesPerElement(); ++i ) {
    // get ith face 1 node id
    const int f1NodeId = mesh1.getGlobalNodeId( fId1, i );

    // compute the components of vector between face 2 center and face 1 vertex
    RealT vX = mesh2.getElementCentroids()[0][fId2] - mesh1.getPosition()[0][f1NodeId];
    RealT vY = mesh2.getElementCentroids()[1][fId2] - mesh1.getPosition()[1][f1NodeId];
    RealT vZ = mesh2.getElementCentroids()[2][fId2] - mesh1.getPosition()[2][f1NodeId];

    // project the vector onto face 2 normal
    RealT proj = vX * mesh2.getElementNormals()[0][fId2] + vY * mesh2.getElementNormals()[1][fId2] +
                 vZ * mesh2.getElementNormals()[2][fId2];

    // if a node of face 1 is on the other side of the plane defined by face 2 the
    // projection will be positive. If a node on face 1 lies on face 2 the projection
    // will be zero. If a node lies just outside of face 2 then the projection will
    // be a small negative number.
    if ( proj >= 0. ) {
      ++k;
    }

  }  // end loop over nodes

  // check to see:
  // 1) all nodes are on the other side triggering a full overlap calc with gap contraint violation
  //    provided there is a positive area of overlap (checked elsewhere)
  // 2) zero nodes are on the other side, triggering a full overlap for a separation configuration
  //    in which case the gap constraint check in the physics will ensure no force contribution
  if ( k == mesh1.numberOfNodesPerElement() || k == 0 ) {
    return true;
  }

  return false;

}  // end FullFaceCheck()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE bool FullEdgeCheck( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2, int eId1,
                                       int eId2 )
{
  // loop over vertices on edge 2
  int k = 0;
  for ( int i = 0; i < mesh2.numberOfNodesPerElement(); ++i ) {
    // get edge 2 ith vertex id
    const int e2vId = mesh2.getGlobalNodeId( eId2, i );

    // compute components of vector between edge 1 center and edge 2 vertex
    RealT vX = mesh1.getElementCentroids()[0][eId1] - mesh2.getPosition()[0][e2vId];
    RealT vY = mesh1.getElementCentroids()[1][eId1] - mesh2.getPosition()[1][e2vId];

    // project the vector onto edge1 normal
    RealT proj = vX * mesh1.getElementNormals()[0][eId1] + vY * mesh1.getElementNormals()[1][eId1];

    // check projection against tolerance
    if ( proj >= 0. ) {
      ++k;
    }
  }  // end loop over edge2 vertices

  // check to see if all vertices are on the other side of this edge in either
  // an interpen sense w.r.t. the plane defined by the other edge or a separation sense
  if ( k == mesh2.numberOfNodesPerElement() || k == 0 ) {
    return true;
  }

  // loop over vertices on edge 1 to catch the case where edge 1 lies
  // entirely on the other side of edge 2 triggering a full overlap
  // computation
  k = 0;
  for ( int i = 0; i < mesh1.numberOfNodesPerElement(); ++i ) {
    // get edge 1 ith vertex id
    const int e1vId = mesh1.getGlobalNodeId( eId1, i );

    // compute components of vector between edge 2 center and edge 1 vertex
    RealT vX = mesh2.getElementCentroids()[0][eId2] - mesh1.getPosition()[0][e1vId];
    RealT vY = mesh2.getElementCentroids()[1][eId2] - mesh1.getPosition()[1][e1vId];

    // project the vector onto edge2 normal
    RealT proj = vX * mesh2.getElementNormals()[0][eId2] + vY * mesh2.getElementNormals()[1][eId2];

    // check projection against tolerance
    if ( proj >= 0. ) {
      ++k;
    }
  }  // end loop over edge1 vertices

  // check to see if all vertices are on the other side of this edge in either
  // an interpen sense w.r.t. the plane defined by the other edge or a separation sense
  if ( k == mesh1.numberOfNodesPerElement() || k == 0 ) {
    return true;
  }

  return false;

}  // end FullEdgeCheck()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE bool ExceedsMaxAutoInterpen( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                                const int faceId1, const int faceId2, const Parameters& params,
                                                const RealT gap )
{
  if ( params.auto_contact_check ) {
    RealT max_interpen = -1. * params.auto_contact_pen_frac *
                         axom::utilities::min( mesh1.getElementData().m_thickness[faceId1],
                                               mesh2.getElementData().m_thickness[faceId2] );
    if ( gap < max_interpen ) {
      return true;
    }
  }
  return false;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE ContactPlanePair::ContactPlanePair( InterfacePair* pair, Parameters& params, const int dim )
    : CompGeomPair( pair, params, dim ),
      m_inContact( false ),
      m_gap( 0.0 ),
      m_gapTol( 0.0 ),
      m_e1X( 0.0 ),
      m_e1Y( 0.0 ),
      m_e1Z( 0.0 ),
      m_e2X( 0.0 ),
      m_e2Y( 0.0 ),
      m_e2Z( 0.0 ),
      m_cX( 0.0 ),
      m_cY( 0.0 ),
      m_cZ( 0.0 ),
      m_overlapCX( 0.0 ),
      m_overlapCY( 0.0 ),
      m_cXf1( 0.0 ),
      m_cYf1( 0.0 ),
      m_cZf1( 0.0 ),
      m_cXf2( 0.0 ),
      m_cYf2( 0.0 ),
      m_cZf2( 0.0 ),
      m_nX( 0.0 ),
      m_nY( 0.0 ),
      m_nZ( 0.0 ),
      m_numPolyVert( 0 ),
      m_areaFrac( params.overlap_area_frac ),
      m_areaMin( 0.0 ),
      m_area( 0.0 )
{
   for (int i=0; i<max_nodes_per_overlap; ++i) {
     m_polyX[i] = 0.;
     m_polyY[i] = 0.;
     m_polyZ[i] = 0.;
     m_polyLocX[i] = 0.;
     m_polyLocY[i] = 0.;
   }
}

//------------------------------------------------------------------------------
// Common Plane Routines
//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE CommonPlanePair::CommonPlanePair( InterfacePair* pair, Parameters& params, const int dim )
    : ContactPlanePair( pair, params, dim ),
      m_numInterpenPoly1Vert( 0 ),
      m_numInterpenPoly2Vert( 0 ),
      m_velGap( 0.0 ),
      m_ratePressure ( 0.0 ),
      m_pressure( 0.0 )
{
  for (int i=0; i<max_nodes_per_intersection; ++i) {
    m_interpenG1X[i] = 0.;
    m_interpenG1Y[i] = 0.;
    m_interpenG1Z[i] = 0.;

    m_interpenG2X[i] = 0.;
    m_interpenG2Y[i] = 0.;
    m_interpenG2Z[i] = 0.;
  }
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError CommonPlanePair::checkInterfacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 )
{
   if (m_dim == 2) {
     return this->checkEdgePair( mesh1, mesh2 );
   } else {
     return this->checkFacePair( mesh1, mesh2 );
   }
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError CommonPlanePair::checkFacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 )
{
  // Note: Checks #1-#5 are done in the binning; see geomFilter()

  // alias variables off the InterfacePair
  IndexT element_id1 = this->getCpElementId1();
  IndexT element_id2 = this->getCpElementId2();

  ////////////////////////////
  // Planar Face Projection //
  ////////////////////////////

  // Project faces (potentially warped) onto their 'average' face-planes.
  // These are the 'prime' coordinates and ensure that our cg is working on
  // truly planar 4 node quadrilaterals
  constexpr int max_dim = 3;
  constexpr int max_nodes_per_elem = 4;
  RealT x1_prime[max_nodes_per_elem];
  RealT y1_prime[max_nodes_per_elem];
  RealT z1_prime[max_nodes_per_elem];
  RealT x2_prime[max_nodes_per_elem];
  RealT y2_prime[max_nodes_per_elem];
  RealT z2_prime[max_nodes_per_elem];

  // get face normmals
  RealT fn1[max_dim], fn2[max_dim];
  mesh1.getFaceNormal( element_id1, fn1 );
  mesh2.getFaceNormal( element_id2, fn2 );

  // get face centroids
  RealT cx1[max_dim], cx2[max_dim];
  mesh1.getFaceCentroid( element_id1, cx1 );
  mesh2.getFaceCentroid( element_id2, cx2 );

  // project face vertices onto FACE-PLANE defined by face centroid-normal
  ProjectFaceNodesToPlane( mesh1, element_id1, fn1[0], fn1[1], fn1[2], cx1[0], cx1[1], cx1[2], &x1_prime[0],
                           &y1_prime[0], &z1_prime[0] );
  ProjectFaceNodesToPlane( mesh2, element_id2, fn2[0], fn2[1], fn2[2], cx2[0], cx2[1], cx2[2], &x2_prime[0],
                           &y2_prime[0], &z2_prime[0] );

  // CHECK #6: check if the two faces overlap in a projected sense.
  // To do this check we need to use the contact plane object, which will
  // have its own local basis that needs to be defined

  ////////////////////////////////////////////////
  // Compute Common Plane Overlap with Vertices //
  ////////////////////////////////////////////////

  // compute common plane normal, centroid, local basis and area tolerance
  this->computeNormal( mesh1, mesh2 );
  this->computePlanePoint( mesh1, mesh2 );
  this->computeLocalBasis( mesh1 );
  this->computeAreaTol( mesh1, mesh2, params );

  // the contact plane has to be properly located prior to computing the interpen overlap
  this->resetPlanePointAndCentroidGap( mesh1, mesh2 );
  FaceGeomError interpen_err = this->computeOverlap3D( &x1_prime[0], &y1_prime[0], &z1_prime[0],
                                                       &x2_prime[0], &y2_prime[0], &z2_prime[0],
                                                       mesh1, mesh2, params );

  if ( interpen_err != NO_FACE_GEOM_ERROR ) {
    this->m_inContact = false;
    return interpen_err;
  }

  this->m_inContact = true;
  return NO_FACE_GEOM_ERROR;

} // end CommonPlanePair::checkFacePair()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError CommonPlanePair::checkEdgePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 )
{
  // Note: Checks #1-#5 are done in the binning

  // alias variables off the InterfacePair
  IndexT edgeId1 = this->getCpElementId1();
  IndexT edgeId2 = this->getCpElementId2();

  // CHECK #6: compute the projected length of overlap on the contact plane.
  // At this point the edges are proximate and likely have a positive
  // projected length of overlap.

  // compute common-plane point-normal data. At this point we don't know where to properly
  // locate the common plane centroid so we just take the average of the two face centroids
  this->computeNormal( mesh1, mesh2 );
  this->computePlanePoint( mesh1, mesh2 );
  this->computeAreaTol( mesh1, mesh2, params );

  FaceGeomError interpen_err = this->computeOverlap2D( mesh1, mesh2, params );
  if ( interpen_err != NO_FACE_GEOM_ERROR ) {
    this->m_inContact = false;
    return interpen_err;
  }

  this->m_inContact = true;
  return NO_FACE_GEOM_ERROR;

}  // end CommonPlanePair::checkEdgePair()

//------------------------------------------------------------------------------
// Mortar Plane Routines
//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE MortarPlanePair::MortarPlanePair( InterfacePair* pair, Parameters& params, const int dim )
    : ContactPlanePair( pair, params, dim )
{
  // no-op
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError MortarPlanePair::checkInterfacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 )
{
  if (m_dim == 3) {
    this->checkFacePair( mesh1, mesh2 );
  } // note mortar not implemented in 2D
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError MortarPlanePair::checkFacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 )
{
  // Note: Checks #1-#5 are done in the binning

  // alias variables off the InterfacePair
  IndexT element_id1 = this->getCpElementId1();
  IndexT element_id2 = this->getCpElementId2();

  ////////////////////////////
  // Planar Face Projection //
  ////////////////////////////

  // Project faces (potentially warped) onto their 'average' face-planes.
  // These are the 'prime' coordinates and ensure that our cg is working on
  // truly planar 4 node quadrilaterals
  constexpr int max_dim = 3;
  constexpr int max_nodes_per_elem = 4;
  RealT x1_prime[max_nodes_per_elem];
  RealT y1_prime[max_nodes_per_elem];
  RealT z1_prime[max_nodes_per_elem];
  RealT x2_prime[max_nodes_per_elem];
  RealT y2_prime[max_nodes_per_elem];
  RealT z2_prime[max_nodes_per_elem];

  // get face normmals
  RealT fn1[max_dim], fn2[max_dim];
  mesh1.getFaceNormal( element_id1, fn1 );
  mesh2.getFaceNormal( element_id2, fn2 );

  // get face centroids
  RealT cx1[max_dim], cx2[max_dim];
  mesh1.getFaceCentroid( element_id1, cx1 );
  mesh2.getFaceCentroid( element_id2, cx2 );

  // project face vertices onto FACE-PLANE defined by face centroid-normal
  ProjectFaceNodesToPlane( mesh1, element_id1, fn1[0], fn1[1], fn1[2], cx1[0], cx1[1], cx1[2], &x1_prime[0],
                           &y1_prime[0], &z1_prime[0] );
  ProjectFaceNodesToPlane( mesh2, element_id2, fn2[0], fn2[1], fn2[2], cx2[0], cx2[1], cx2[2], &x2_prime[0],
                           &y2_prime[0], &z2_prime[0] );
  
  ////////////////////////////////////////////////
  // Compute Mortar Plane Overlap with Vertices //
  ////////////////////////////////////////////////

  // compute common plane normal, centroid, local basis and area tolerance
  this->computeNormal( mesh1, mesh2 );
  this->computePlanePoint( mesh1, mesh2 );
  this->computeLocalBasis( mesh1 );
  this->computeAreaTol( mesh1, mesh2, params );

  // the contact plane has to be properly located prior to computing the interpen overlap
  FaceGeomError interpen_err = this->computeOverlap3D( &x1_prime[0], &y1_prime[0], &z1_prime[0],
                                                       &x2_prime[0], &y2_prime[0], &z2_prime[0],
                                                       mesh1, mesh2, params );

  if ( interpen_err != NO_FACE_GEOM_ERROR ) {
    this->m_inContact = false;
    return interpen_err;
  }

  this->m_inContact = true;
  return NO_FACE_GEOM_ERROR;

} // end MortarPlanePair::checkFacePair()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError MortarPlanePair::computeOverlap3D( const RealT* x1, const RealT* y1, const RealT* z1,
                                                                    const RealT* x2, const RealT* y2, const RealT* z2,
                                                                    const MeshData::Viewer& m1,
                                                                    const MeshData::Viewer& m2,
                                                                    const Parameters& params )
{
  
  IndexT element_id1 = this->getCpElementId1();
  IndexT element_id2 = this->getCpElementId2();

  // project face nodes onto contact plane.
  constexpr int max_nodes_per_elem = 4;
  RealT x1_bar[max_nodes_per_elem];
  RealT y1_bar[max_nodes_per_elem];
  RealT z1_bar[max_nodes_per_elem];
  RealT x2_bar[max_nodes_per_elem];
  RealT y2_bar[max_nodes_per_elem];
  RealT z2_bar[max_nodes_per_elem];

  ProjectFaceNodesToPlane( m1, element_id1, this->m_nX, this->m_nY, this->m_nZ, this->m_cX, this->m_cY, this->m_cZ,
                           &x1_bar[0], &y1_bar[0], &z1_bar[0] );
  ProjectFaceNodesToPlane( m2, element_id2, this->m_nX, this->m_nY, this->m_nZ, this->m_cX, this->m_cY, this->m_cZ,
                           &x2_bar[0], &y2_bar[0], &z2_bar[0] );

  // project the projected global nodal coordinates onto local
  // contact plane 2D coordinate system.
  RealT x1_bar_local[max_nodes_per_elem];
  RealT y1_bar_local[max_nodes_per_elem];
  RealT x2_bar_local[max_nodes_per_elem];
  RealT y2_bar_local[max_nodes_per_elem];

  this->globalTo2DLocalCoords( &x1_bar[0], &y1_bar[0], &z1_bar[0], &x1_bar_local[0], &y1_bar_local[0],
                               mesh1.numberOfNodesPerElement() );
  this->globalTo2DLocalCoords( &x2_bar[0], &y2_bar[0], &z2_bar[0], &x2_bar_local[0], &y2_bar_local[0],
                               mesh2.numberOfNodesPerElement() );

  // compute the full intersection polygon vertex coordinates
  RealT* X1 = &x1_bar_local[0];
  RealT* Y1 = &y1_bar_local[0];
  RealT* X2 = &x2_bar_local[0];
  RealT* Y2 = &y2_bar_local[0];

  // assuming each face's vertices are ordered WRT that face's outward unit normal,
  // reorder face 2 vertices to be consistent with face 1. DO NOT CALL POLYREORDER()
  // to do this. // TODO debug this; this may affect calculations later on. We may
  // have to unreverse the ordering.
  ElemReverse( X2, Y2, mesh2.numberOfNodesPerElement() );

  // compute intersection polygon and area.
  RealT pos_tol = params.len_collapse_ratio *
                  axom::utilities::max( mesh1.getFaceRadius()[element_id1], mesh2.getFaceRadius()[element_id2] );
  RealT len_tol = pos_tol;
  FaceGeomError inter_err =
      Intersection2DPolygon( X1, Y1, mesh1.numberOfNodesPerElement(), X2, Y2, mesh2.numberOfNodesPerElement(),
                             pos_tol, len_tol, this->m_polyLocX, this->m_polyLocY, this->m_numPolyVert, this->m_area, false );

  if ( inter_err != NO_FACE_GEOM_ERROR ) {
    return inter_err;
  }

  // check overlap area to area tol
  if ( m_area < m_areaMin ) {
    return NO_OVERLAP;
  }

  // compute the local vertex averaged centroid of overlapping polygon
  RealT cZ;  // dummy z component for call to routine
  VertexAvgCentroid( this->m_polyLocX, this->m_polyLocY, nullptr, this->m_numPolyVert, this->m_overlapCX, this->m_overlapCY, cZ );

  // handle the case where the actual polygon with connectivity
  // and computed vertex coordinates becomes degenerate due to
  // either position tolerances (segment-segment intersections)
  // or length tolerances (intersecting polygon segment lengths)
  if ( this->m_numPolyVert < 3 ) {
#ifdef TRIBOL_USE_HOST
    SLIC_DEBUG( "degenerate polygon intersection detected.\n" );
#endif
    return DEGENERATE_OVERLAP;
  }

  // Transform local vertex coordinates to global coordinates for the
  // current projection of the polygonal overlap
  for ( int i = 0; i < this->m_numPolyVert; ++i ) {
    this->m_polyX[i] = 0.0;
    this->m_polyY[i] = 0.0;
    this->m_polyZ[i] = 0.0;

    this->local2DToGlobalCoords( this->m_polyLocX[i], this->m_polyLocY[i], this->m_polyX[i], this->m_polyY[i], this->m_polyZ[i] );
  }

  // check polygonal vertex ordering with mortar plane normal
  PolyReorderWithNormal( this->m_polyX, this->m_polyY, this->m_polyZ, this->m_numPolyVert, this->m_nX, this->m_nY, this->m_nZ );

  return NO_FACE_GEOM_ERROR;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError MortarPlanePair::checkEdgePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 )
{
  // no-op; implement when 2D mortar is implemented
}

//------------------------------------------------------------------------------
// Aligned Mortar Plane Routines
//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE AlignedMortarPlanePair::AlignedMortarPlanePair( InterfacePair* pair, Parameters& params, const int dim )
    : ContactPlanePair( pair, params, dim )
{
  // no-op
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError AlignedMortarPlanePair::checkInterfacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 )
{
  if (m_dim == 3) {
     this->checkFacePair( mesh1, mesh2 );
  }

  // NOTE the coupling scheme initialization will error out on host if aligned mortar
  // is trying to be used with dim = 2.
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError AlignedMortarPlanePair::checkFacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 )
{
  // Note: Checks #1-#5 are done in the binning

  // alias variables off the InterfacePair
  IndexT element_id1 = this->getCpElementId1();
  IndexT element_id2 = this->getCpElementId2();

  // Project faces (potentially warped) onto their 'average' face-planes.
  // These are the 'prime' coordinates and ensure that our cg is working on
  // truly planar 4 node quadrilaterals
  constexpr int max_dim = 3;
  constexpr int max_nodes_per_elem = 4;
  RealT x1_prime[max_nodes_per_elem];
  RealT y1_prime[max_nodes_per_elem];
  RealT z1_prime[max_nodes_per_elem];
  RealT x2_prime[max_nodes_per_elem];
  RealT y2_prime[max_nodes_per_elem];
  RealT z2_prime[max_nodes_per_elem];

  // get face normmals
  RealT fn1[max_dim], fn2[max_dim];
  mesh1.getFaceNormal( element_id1, fn1 );
  mesh2.getFaceNormal( element_id2, fn2 );

  // get face centroids
  RealT cx1[max_dim], cx2[max_dim];
  mesh1.getFaceCentroid( element_id1, cx1 );
  mesh2.getFaceCentroid( element_id2, cx2 );

  // project face vertices onto FACE-PLANE defined by face centroid-normal
  ProjectFaceNodesToPlane( mesh1, element_id1, fn1[0], fn1[1], fn1[2], cx1[0], cx1[1], cx1[2], &x1_prime[0],
                           &y1_prime[0], &z1_prime[0] );
  ProjectFaceNodesToPlane( mesh2, element_id2, fn2[0], fn2[1], fn2[2], cx2[0], cx2[1], cx2[2], &x2_prime[0],
                           &y2_prime[0], &z2_prime[0] );

  // Check #6 see if the two faces are aligned; hence, overlap area being face area

  // compute common plane normal, centroid, local basis and area tolerance
  this->computeNormal( mesh1, mesh2 );
  this->computePlanePoint( mesh1, mesh2 );
  this->computeAreaTol( mesh1, mesh2, params );
  FaceGeomError interpen_err = this->computeOverlap3D( &x1_prime[0], &y1_prime[0], &z1_prime[0],
                                                       &x2_prime[0], &y2_prime[0], &z2_prime[0],
                                                       mesh1, mesh2, params );

  if ( interpen_err != NO_FACE_GEOM_ERROR ) {
    this->m_inContact = false;
    return interpen_err;
  }

  this->m_inContact = true;
  return NO_FACE_GEOM_ERROR;

}  // end AlignedMortarPlanePair::checkFacePair()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError AlignedMortarPlanePair::checkEdgePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 )
{
  // no-op; implement when 2D aligned mortar is implemented
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError AlignedMortarPlanePair::computeOverlap3D( const RealT* x1, const RealT* y1, const RealT* z1,
                                                                           const RealT* x2, const RealT* y2, const RealT* z2,
                                                                           const MeshData::Viewer& m1,
                                                                           const MeshData::Viewer& m2,
                                                                           const Parameters& params )
{
  IndexT element_id1 = this->getCpElementId1();
  IndexT element_id2 = this->getCpElementId2();

  // Compute face centroids using projected coordinates passed in
  RealT cx1, cy1, cz1;
  RealT cx2, cy2, cz2;
  VertexAvgCentroid( x1, y1, z1, m1.numberOfNodesPerElement(), cx1, cy1, cz1 );
  VertexAvgCentroid( x2, y2, z2, m2.numberOfNodesPerElement(), cx2, cy2, cz2 );

  // compute the gap vector between face centroids. Then, project the gap vector
  // on the contact plane unit normal. The magnitude of the gap vector should be very
  // close to the projected gap vector for aligned faces
  RealT gapVecX = cx2 - cx1;
  RealT gapVecY = cy2 - cy1;
  RealT gapVecZ = cz2 - cz1;

  RealT scalarGap = gapVecX * this->m_nX + gapVecY * this->m_nY + gapVecZ * this->m_nZ;

  RealT gapVecMag = magnitude( gapVecX, gapVecY, gapVecZ );

  if ( gapVecMag > 1.1 * std::abs( scalarGap ) ) {
    return NO_OVERLAP;
  }

  // if we are here we have contact between two aligned faces; per mortar method take
  // the non-mortar (mesh2) face as the contact plane. For Aligned mortar we can use
  // face 2 as the overlap plane and overlap
  this->m_numPolyVert = m2.numberOfNodesPerElement();
  for ( int a = 0; a < m2.numberOfNodesPerElement(); ++a ) {
    this->m_polyX[a] = x2[a];
    this->m_polyY[a] = y2[a];
    this->m_polyZ[a] = z2[a];
  }

  // compute the local vertex averaged centroid of overlapping polygon
  RealT cZ;  // dummy z component for call to routine
  VertexAvgCentroid( this->m_polyLocX, this->m_polyLocY, nullptr, this->m_numPolyVert, this->m_overlapCX, this->m_overlapCY, cZ );

  m_gap = scalarGap;
  m_area = mesh2.getElementAreas()[element_id2];

  return NO_FACE_GEOM_ERROR;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void CommonPlanePair::resetPlanePointAndCentroidGap( const MeshData::Viewer& m1, const MeshData::Viewer& m2 )
{
  // project the overlap centroid back to each face using a
  // line-plane intersection method
  RealT xc1 = 0.;
  RealT yc1 = 0.;
  RealT zc1 = 0.;
  RealT xc2 = 0.;
  RealT yc2 = 0.;
  RealT zc2 = 0.;

  RealT xcg = m_cX; // for 2D just set to the overlap centroid
  RealT ycg = m_cY; // same
  RealT zcg = 0.0;

  // first project the projected area of overlap's centroid in local
  // coordinates to global coordinates
  if ( m_dim == 3 ) {
    auto xloc = this->m_overlapCX; // the local coords of the overlap centroid are computed in computeOverlap3D()
    auto yloc = this->m_overlapCY;
    xcg = xloc * this->m_e1X + yloc * this->m_e2X + this->m_cX;
    ycg = xloc * this->m_e1Y + yloc * this->m_e2Y + this->m_cY;
    zcg = xloc * this->m_e1Z + yloc * this->m_e2Z + this->m_cZ;
  }

  // find where the overlap centroid (plane point) intersects each face
  auto fId1 = getCpElementId1();
  auto fId2 = getCpElementId2();
  RealT c1_z = 0.0;
  RealT n1_z = 0.0;
  RealT c2_z = 0.0;
  RealT n2_z = 0.0;
  if ( m_dim == 3 ) {
    c1_z = m1.getElementCentroids()[2][fId1];
    n1_z = m1.getElementNormals()[2][fId1];
    c2_z = m2.getElementCentroids()[2][fId2];
    n2_z = m2.getElementNormals()[2][fId2];
  }
  ProjectPointToPlane( xcg, ycg, zcg, m1.getElementNormals()[0][fId1], m1.getElementNormals()[1][fId1], n1_z,
                       m1.getElementCentroids()[0][fId1], m1.getElementCentroids()[1][fId1], c1_z, xc1, yc1, zc1 );
  ProjectPointToPlane( xcg, ycg, zcg, m2.getElementNormals()[0][fId2], m2.getElementNormals()[1][fId2], n2_z,
                       m2.getElementCentroids()[0][fId2], m2.getElementCentroids()[1][fId2], c2_z, xc2, yc2, zc2 );

  // reset point-location of common plane as average of overlap centroid projected to each face
  m_cX = 0.5 * ( xc1 + xc2 );
  m_cY = 0.5 * ( yc1 + yc2 );
  m_cZ = 0.5 * ( zc1 + zc2 );

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
} // end CommonPlanePair::resetPlanePointAndCentroidGap()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void CommonPlanePair::computeNormal( const MeshData::Viewer& m1, const MeshData::Viewer& m2 )
{
  IndexT fId1 = m_pair->m_element_id1;
  IndexT fId2 = m_pair->m_element_id2;
  m_nZ = 0.0;

  // INTERMEDIATE (I.E. COMMON) PLANE normal calculation:
  // compute the cp normal as the average of the two face normals, and in
  // the direction such that the dot product between the cp normal and
  // the normal of face 2 is positive. This is the default method of
  // computing the cp normal
  m_nX = 0.5 * ( m2.getElementNormals()[0][fId2] - m1.getElementNormals()[0][fId1] );
  m_nY = 0.5 * ( m2.getElementNormals()[1][fId2] - m1.getElementNormals()[1][fId1] );

  if (m_dim == 3) {
    m_nZ = 0.5 * ( m2.getElementNormals()[2][fId2] - m1.getElementNormals()[2][fId1] );
  }

  // normalize the cp normal
  RealT mag = magnitude( m_nX, m_nY, m_nZ );
  RealT invMag = 1.0 / mag;

  m_nX *= invMag;
  m_nY *= invMag;
  m_nZ *= invMag;

  return;

}  // end CommonPlanePair::computeNormal()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void MortarPlanePair::computeNormal( const MeshData::Viewer& m1, const MeshData::Viewer& m2 )
{
  IndexT fId1 = m_pair->m_element_id1;
  IndexT fId2 = m_pair->m_element_id2;
  m_nZ = 0.0;

  // the projection plane is the nonmortar (i.e. mesh id 2) surface so
  // we use the outward normal for face 2 on mesh 2
  m_nX = m2.getElementNormals()[0][fId2];
  m_nY = m2.getElementNormals()[1][fId2];

  if (m_dim == 3) {
    m_nZ = m2.getElementNormals()[2][fId2];
  }

  // normalize the cp normal
  RealT mag = magnitude( m_nX, m_nY, m_nZ );
  RealT invMag = 1.0 / mag;

  m_nX *= invMag;
  m_nY *= invMag;
  m_nZ *= invMag;

  return;

}  // end MortarPlanePair::computeNormal()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void AlignedMortarPlanePair::computeNormal( const MeshData::Viewer& m1, const MeshData::Viewer& m2 )
{
  IndexT fId1 = m_pair->m_element_id1;
  IndexT fId2 = m_pair->m_element_id2;
  m_nZ = 0.0;

  // the projection plane is the nonmortar (i.e. mesh id 2) surface so
  // we use the outward normal for face 2 on mesh 2
  m_nX = m2.getElementNormals()[0][fId2];
  m_nY = m2.getElementNormals()[1][fId2];

  if (m_dim == 3) {
    m_nZ = m2.getElementNormals()[2][fId2];
  }

  // normalize the cp normal
  RealT mag = magnitude( m_nX, m_nY, m_nZ );
  RealT invMag = 1.0 / mag;

  m_nX *= invMag;
  m_nY *= invMag;
  m_nZ *= invMag;

  return;

}  // end AlignedMortarPlanePair::computeNormal()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void CommonPlanePair::computePlanePoint( const MeshData::Viewer& m1, const MeshData::Viewer& m2 )
{
  // compute the cp centroid as the average of the two face's centers.
  // This is the default method of computing the cp centroid
  IndexT fId1 = m_pair->m_element_id1;
  IndexT fId2 = m_pair->m_element_id2;

  // INTERMEDIATE (I.E. COMMON) PLANE point calculation:
  // average two face vertex averaged centroids
  m_cX = 0.5 * ( m1.getElementCentroids()[0][fId1] + m2.getElementCentroids()[0][fId2] );
  m_cY = 0.5 * ( m1.getElementCentroids()[1][fId1] + m2.getElementCentroids()[1][fId2] );

  if (m_dim == 3) {
    m_cZ = 0.5 * ( m1.getElementCentroids()[2][fId1] + m2.getElementCentroids()[2][fId2] );
  }

  return;

}  // end CommonPlanePair::computePlanePoint()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void MortarPlanePair::computePlanePoint( const MeshData::Viewer& m1, const MeshData::Viewer& m2 )
{
  // compute the cp centroid as the average of the two face's centers.
  // This is the default method of computing the cp centroid
  IndexT fId1 = m_pair->m_element_id1;
  IndexT fId2 = m_pair->m_element_id2;

  // MORTAR calculation using the vertex averaged
  // centroid of the nonmortar face
  m_cX = m2.getElementCentroids()[0][fId2];
  m_cY = m2.getElementCentroids()[1][fId2];

  if (m_dim == 3) {
    m_cZ = m2.getElementCentroids()[2][fId2];
  }

  return;

}  // end MortarPlanePair::computePlanePoint()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void AlignedMortarPlanePair::computePlanePoint( const MeshData::Viewer& m1, const MeshData::Viewer& m2 )
{
  // compute the cp centroid as the average of the two face's centers.
  // This is the default method of computing the cp centroid
  IndexT fId1 = m_pair->m_element_id1;
  IndexT fId2 = m_pair->m_element_id2;

  // set plane centroid to mesh 2 face centroid (i.e. non-mortar)
  m_cX = m2.getElementCentroids()[0][fId2];
  m_cY = m2.getElementCentroids()[1][fId2];

  if (m_dim == 3) {
    m_cZ = m2.getElementCentroids()[2][fId2];
  }

  return;

}  // end AlignedMortarPlanePair::computePlanePoint()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlanePair::computeLocalBasis( const MeshData::Viewer& m1 )
{
  // somewhat arbitrarily set the first local basis vector to be
  // between contact plane centroid and first node on first face as
  // projected onto the contact plane
  const int nodeId = m1.getGlobalNodeId( m_pair->m_element_id1, 0 );

  // project to plane
  RealT pX, pY, pZ;
  ProjectPointToPlane( m1.getPosition()[0][nodeId], m1.getPosition()[1][nodeId], m1.getPosition()[2][nodeId], m_nX,
                       m_nY, m_nZ, m_cX, m_cY, m_cZ, pX, pY, pZ );

  // compute first basis vector between projected node and overlap centroid
  m_e1X = pX - m_cX;
  m_e1Y = pY - m_cY;
  m_e1Z = pZ - m_cZ;

  // check the square of the magnitude of the first basis vector to
  // catch the case where pX = m_cX and so on.
  RealT sqrMag = m_e1X * m_e1X + m_e1Y * m_e1Y + m_e1Z * m_e1Z;

  if ( sqrMag < 1.E-12 )  // note: tolerance on the square of the magnitude
  {
    // translate projected first node by face radius
    RealT radius = m1.getFaceRadius()[m_pair->m_element_id1];
    RealT scale = 1.0 * radius;

    RealT pNewX = pX + scale;
    RealT pNewY = pY + scale;
    RealT pNewZ = pZ + scale;

    // project point onto contact plane
    ProjectPointToPlane( pNewX, pNewY, pNewZ, m_nX, m_nY, m_nZ, m_cX, m_cY, m_cZ, pX, pY, pZ );

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

}  // end CommonPlanePair::computeLocalBasis()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlanePair::globalTo2DLocalCoords( RealT* pX, RealT* pY, RealT* pZ, RealT* pLX, RealT* pLY,
                                                                 int size )
{
  // loop over projected nodes
  for ( int i = 0; i < size; ++i ) {
    // compute the vector between the point on the plane and the contact plane point
    RealT vX = pX[i] - m_cX;
    RealT vY = pY[i] - m_cY;
    RealT vZ = pZ[i] - m_cZ;

    // project this vector onto the {e1,e2} local basis. This vector is
    // in the plane so the out-of-plane component should be zero.
    pLX[i] = vX * m_e1X + vY * m_e1Y + vZ * m_e1Z;  // projection onto e1
    pLY[i] = vX * m_e2X + vY * m_e2Y + vZ * m_e2Z;  // projection onto e2
  }

  return;

}  // end ContactPlanePair::globalTo2DLocalCoords()

//------------------------------------------------------------------------------
void ContactPlanePair::globalTo2DLocalCoords( RealT pX, RealT pY, RealT pZ, RealT& pLX, RealT& pLY,
                                              int TRIBOL_UNUSED_PARAM( size ) )
{
  // compute the vector between the point on the plane and the contact plane point
  RealT vX = pX - m_cX;
  RealT vY = pY - m_cY;
  RealT vZ = pZ - m_cZ;

  // project this vector onto the {e1,e2} local basis. This vector is
  // in the plane so the out-of-plane component should be zero.
  pLX = vX * m_e1X + vY * m_e1Y + vZ * m_e1Z;  // projection onto e1
  pLY = vX * m_e2X + vY * m_e2Y + vZ * m_e2Z;  // projection onto e2

  return;

}  // end ContactPlanePair::globalTo2DLocalCoords()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlanePair::computeAreaTol( const MeshData::Viewer& m1, const MeshData::Viewer& m2,
                                                          const Parameters& params )
{
  if ( m_areaFrac < params.overlap_area_frac ) {
#ifdef TRIBOL_USE_HOST
    SLIC_DEBUG( "CommonPlanePair::computeAreaTol() the overlap area fraction too small or negative; "
                << "setting to overlap_area_frac parameter." );
#endif
    m_areaFrac = params.overlap_area_frac;
  }

  m_areaMin = m_areaFrac * axom::utilities::min( m1.getElementAreas()[ this->getCpElementId1() ],
                                                 m2.getElementAreas()[ this->getCpElementId2() ] );

  return;

}  // end ContactPlanePair::computeAreaTol()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlanePair::checkPolyOverlap( const MeshData::Viewer& m1, const MeshData::Viewer& m2,
                                                            RealT* projLocX1, RealT* projLocY1, RealT* projLocX2,
                                                            RealT* projLocY2, const int isym )
{
  // change the vertex ordering of one of the faces so that the two match
  constexpr int max_nodes_per_elem = 4;
  RealT x2Temp[max_nodes_per_elem];
  RealT y2Temp[max_nodes_per_elem];

  // set first vertex coordinates the same
  x2Temp[0] = projLocX2[0];
  y2Temp[0] = projLocY2[0];

  // reorder
  int k = 1;
  for ( int i = ( m2.numberOfNodesPerElement() - 1 ); i > 0; --i ) {
    x2Temp[k] = projLocX2[i];
    y2Temp[k] = projLocY2[i];
    ++k;
  }

  PolyInterYCentroid( m1.numberOfNodesPerElement(), projLocX1, projLocY1, m2.numberOfNodesPerElement(), x2Temp, y2Temp,
                      isym, m_area, m_overlapCY );
  PolyInterYCentroid( m1.numberOfNodesPerElement(), projLocY1, projLocX1, m2.numberOfNodesPerElement(), y2Temp, x2Temp,
                      isym, m_area, m_overlapCX );

  return;

}  // end ContactPlanePair::checkPolyOverlap()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlanePair::local2DToGlobalCoords( RealT xloc, RealT yloc, RealT& xg, RealT& yg, RealT& zg )
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

}  // end ContactPlanePair::local2DToGlobalCoords()

//------------------------------------------------------------------------------
void CommonPlanePair::centroidGap( const MeshData::Viewer& m1, const MeshData::Viewer& m2, RealT scale )
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

  // in 3D, first project the projected area of overlap's centroid in local
  // coordinates to global coordinates
  if (m_dim == 3) {
    local2DToGlobalCoords( m_overlapCX, m_overlapCY, xcg, ycg, zcg );
  }
  else {
    xcg = m_cX;
    ycg = m_cY;
    zcg = 0.;
  }

  // find where the overlap centroid (plane point) intersects each face

  // set the line segment's first vertex at the contact plane centroid scaled
  // in the direction opposite the contact plane normal
  RealT xA = xcg + m_nX * scale;
  RealT yA = ycg + m_nY * scale;
  RealT zA = 0.;

  if (m_dim == 3) {
    zA = zcg + m_nZ * scale;
  }

  // use the contact plane normal as the segment directional vector scale in
  // the direction of the contact plane
  RealT xB = xcg - m_nX * scale;
  RealT yB = ycg - m_nY * scale;
  RealT zB = 0.;

  if (m_dim == 3) {
    zB = zcg - m_nZ * scale;
  }

  bool inPlane = false;
  IndexT fId1 = m_pair->m_element_id1;
  IndexT fId2 = m_pair->m_element_id2;

  // get face normmals
  constexpr int max_dim = 3;
  RealT fn1[max_dim], fn2[max_dim];
  m1.getFaceNormal( fId1, fn1 );
  m2.getFaceNormal( fId2, fn2 );

  // get face centroids
  RealT cx1[max_dim], cx2[max_dim];
  m1.getFaceCentroid( fId1, cx1 );
  m2.getFaceCentroid( fId2, cx2 );

  // initialize to zero so that z component will be set to zero for dim = 2
  for ( int i=0; i<max_dim; ++i) {
    fn1[i] = 0.;
    fn2[i] = 0.;
    cx1[i] = 0.;
    cx2[i] = 0.;
  }

  bool intersect1 = LinePlaneIntersection( xA, yA, zA, xB, yB, zB,
                                           cx1[0], cx1[1], cx1[2],
                                           fn1[0], fn1[1], fn1[2],
                                           xc1, yc1, zc1, inPlane );

  bool intersect1 = LinePlaneIntersection( xA, yA, zA, xB, yB, zB,
                                           cx2[0], cx2[1], cx2[2],
                                           fn2[0], fn2[1], fn2[2],
                                           xc2, yc2, zc2, inPlane );

  TRIBOL_UNUSED_VAR( intersect1 );  // We don't currently use these bool variabeles
  TRIBOL_UNUSED_VAR( intersect2 );  // but the above function calls modify some parameters

  // compute normal gap magnitude (x1 - x2 for positive gap in separation
  // and negative gap in penetration)
  m_gap = ( xc1 - xc2 ) * m_nX + ( yc1 - yc2 ) * m_nY;

  if (m_dim == 3) {
    m_gap += (zc1 - zc2) * m_nZ;
  }

  // store the two face points corresponding to the contact plane centroid projection/intersection
  m_cXf1 = xc1;
  m_cYf1 = yc1;
  m_cZf1 = zc1;

  m_cXf2 = xc2;
  m_cYf2 = yc2;
  m_cZf2 = zc2;

  return;

}  // end CommonPlanePair::centroidGap()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError CommonPlanePair::computeOverlap3D( const RealT* x1, const RealT* y1, const RealT* z1,
                                                                    const RealT* x2, const RealT* y2, const RealT* z2,
                                                                    const MeshData::Viewer& m1,
                                                                    const MeshData::Viewer& m2,
                                                                    const Parameters& params )
{
  // TODO SRW determine if we can pass pointers to a device callable function

  // outer loop over faces, inner loop over nodes/segments and determine
  // how many 1) line-plane intersections there are (there should at most be
  // two for interesection polygons or zero for fully separated or fully 
  // interpenetrated faces) and then 2) number of nodes one the current face
  // that cross the plane defined by the other face.
  
  // arrays to hold the maximum line-plane intersections for both faces.
  // Note, convex planar quadrilaterals can only intesect the common
  // plane at most in two places for each face.
  constexpr int max_nodes_per_elem = 4;
  constexpr int max_dim = 3;
  RealT xInter[ max_nodes_per_elem ];
  RealT yInter[ max_nodes_per_elem ];
  RealT zInter[ max_nodes_per_elem ];

  for (int i=0; i<max_nodes_per_elem; ++i) {
     xInter[i] = 0.;
     yInter[i] = 0.;
     zInter[i] = 0.;
  }

  bool inPlane = false;
  int numV[2] = {0, 0};

  // set up vertex id arrays to indicate which face vertices pass through
  // contact plane (i.e. lie on the other side)
  StackArrayT<const MeshData::Viewer*, 2> mesh( { &m1, &m2 } );
  int interpenVertex1[max_nodes_per_elem];
  int interpenVertex2[max_nodes_per_elem];

  for (int i=0; i<max_nodes_per_elem; ++i) {
    interpenVertex1[i] = -1;
    interpenVertex2[i] = -1;
  }

  StackArrayT<IndexT, 2> element_id( { getCpElementId1(), getCpElementId2() } );
  StackArrayT<IndexT, 2> num_intersections( {0, 0} );
  StackArrayT<IndexT, 2> num_intersections_inside( {0, 0} );
  StackArrayT<IndexT, 2> num_nodes_otherside( {0, 0} );

  for ( int i = 0; i < 2; ++i )  // loop over two constituent faces
  {
    // declare array to hold vertex id for all vertices that interpenetrate
    // the contact plane. At most, all nodes pass through common plane for
    // the current face
    int interpenVertex[max_nodes_per_elem];

    // point to the correct current face coordinates
    const RealT* x, y, z;
    const RealT* x_other, y_other, z_other;
    if (i==0) {
       x = x1;
       y = y1;
       z = z1;
       x_other = x2;
       y_other = y2;
       z_other = z2;
    } else {
       x = x2;
       y = y2;
       z = z2;
       x_other = x1;
       y_other = y1;
       z_other = z1;
    }

    // get the other face normal and centroid for line-face-plane intersections
    RealT fn[max_dim], cx[max_dim];
    RealT num_nodes_other;
    if (i==0) {
      mesh[1].getFaceNormal( element_id[1], fn );
      mesh[1].getFaceCentroid( element_id[1], cx );
      num_nodes_other = mesh[1].numberOfNodesPerElement();
    } else {
      mesh[0].getFaceNormal( element_id[0], fn );
      mesh[0].getFaceCentroid( element_id[0], cx );
      num_nodes_other = mesh[0].numberOfNodesPerElement();
    }

    int k = 0;
    int k_inside = 0;
    int k_otherside = 0;
    for ( int j = 0; j < mesh[i]->numberOfNodesPerElement(); ++j )  // loop over face segments
    {
      // determine local segment vertex ids
      int ja = j;
      int jb = ( j == ( mesh[i]->numberOfNodesPerElement() - 1 ) ) ? 0 : ( j + 1 );

      // initialize current entry in the vertex id list
      interpenVertex[ja] = -1;

      // first and second nodes of the current segment
      const RealT xa = x[ja]; 
      const RealT ya = y[ja]; 
      const RealT za = z[ja]; 

      const RealT xb = x[jb]; 
      const RealT yb = y[jb]; 
      const RealT zb = z[jb]; 

      if ( k > 2 ) { // at most we can have two segment-plane intersections for a single planar, convex face
#ifdef TRIBOL_USE_HOST
        SLIC_DEBUG( "CommonPlanePair::computeOverlap3D(): too many segment-face intersections; "
                    << "check for degenerate face " << m_pair->m_element_id1 << "on mesh " << mesh[i]->meshId()
                    << "." );
#endif
        return DEGENERATE_OVERLAP;
      }

      // call segment-to-plane intersection routine
      if ( k < 2 )  // we haven't found both intersection points yet
      {
        // compute the current face's current segment-to-plane intersection using the other face's point-normal data
        // TODO SRW check that this routine doesn't scale the line segment that it is truly a line-to-plane intersection calc
        bool inter = LinePlaneIntersection( xa, ya, za, xb, yb, zb, cx[0], cx[1], cx[2], fn[0], fn[1], fn[2],
                                            xInter[2 * i + k], yInter[2 * i + k], zInter[2 * i + k], inPlane );

        if ( inter ) {
          // check to see if the line-plane intersection point lies inside the other planar face

          RealT x_other_local[ max_nodes_per_elem ];
          RealT y_other_local[ max_nodes_per_elem ];
          Plane3DTo2D( &x_other[0], &y_other[0], &z_other[0], fn[0], fn[1], fn[2], cx[0], cx[1], cx[2],
                       num_nodes_other, &x1_other_local[0], &y1_other_local[1] );
          
          // get the local coordinates of the current intersection point
          RealT xInter_local, yInter_local;
          Point3DTo2D( xInter[2*i+k], yInter[2*i+k], zInter[2*i+k], fn[0], fn[1], fn[2], cx[0], cx[1], cx[2], xInter_local, yInter_local );

          // get the local coordinates of the other face's centroid
          RealT cx_other_local, cy_other_local;
          RealT cz = 0.; // dummy arg.
          //Point3DTo2D( cx[0], cx[1], cx[2], fn[0], fn[1], fn[2], cx[0], cx[1], cx[2], cx_other_local, cy_other_local );
          VertexAvgCentroid( &x_other_local[0], &y_other_local[0], nullptr, num_nodes_other, cx_other_local, cy_other_local, cz );

          // check if local intersection point lies inside other face
          bool check = Point2DInFace( xInter_local, yInter_local, &x_other_local[0], &y_other_local[0], cx_other_local, cy_other_local, num_nodes_other );
          // if intersection point lies in other face then increment intersection counter
          if (check) {
            ++k_inside;
          }
   
          // we still want to increment the intersection counter expecting up to 2 line-plane intersections
          // even if the point is not inside
          ++k;
                                 
        } // end if (inter)
      } // end if (k<2)

      // Secondly: check the current face's current node to see if it lies on the other side of the other face.
      // do this even if we don't ultimately have an interpen overlap calc.
      RealT vX = xa - cx[0];
      RealT vY = ya - cx[1];
      RealT vZ = za - cx[2];

      // project the vector onto the opposing face's normal
      RealT proj = vX * fn[0] + vY * fn[1] + vZ * fn[2];

      // if the projection for face 1 vertices is positive then that vertex crosses
      // (i.e. interpenetrates) the other face's plane. if the projection for face 2 vertices
      // is negative then that vertex crosses the other face's plane 
      interpenVertex[ja] = ( i == 0 && proj > 0 ) ? ja : -1;
      interpenVertex[ja] = ( i == 1 && proj < 0 ) ? ja : interpenVertex[ja];

      if (interpenVertex[ja] != -1) {
        ++k_otherside;
      }

    }  // end loop over nodes

    // count the number of vertices (intersection points and interpen points) for the clipped
    // portion of the i^th face that interpenetrates the opposing face.
    numV[i] = k; // could be zero intersection points
    for ( int vid = 0; vid < mesh[i]->numberOfNodesPerElement(); ++vid ) {

      // increment total vertex counter if ids match
      if ( interpenVertex[vid] == vid ) ++numV[i];

      // populate the face specific id array
      if ( i == 0 ) {
        interpenVertex1[vid] = interpenVertex[vid];
      } else {
        interpenVertex2[vid] = interpenVertex[vid];
      }
    }

    // set face specific intersection point count
    num_intersections[i] = k;
    num_intersections_inside[i] = k_inside;
    num_nodes_otherside[i] = k_otherside;

  }  // end loop over faces

  // we come into this routine with full overlap calculation set to true. Here, we need
  // to determine if we need to switch to interpen overlap calc. This is cleaner logic
  // than assuming interpen and switching to full because it only checks interior intersection
  // points. The criterion for intersection and thus the interpen overlap calc for two
  // planar quadrilaterals is:
  //
  // 1) each face has one intersection point that lies INSIDE the other face, OR
  // 2) one face has two intersection points that lie INSIDE the other face and the
  //    other face has zero intersection points that lie INSIDE its opposing face
  //
  // Note: still double check degenerate face-interaction vertex counts and in the case
  //       that one of the criterion above switched to the interpen overlap calc, return
  //       the calc to full overlap for robustness
  if (num_intersections_inside[0] == 1 && num_intersections_inside[1] == 1) {
    m_fullOverlap = false;
  } else if (num_intersections_inside[0] == 2 && num_intersections_inside[1] == 0) {
    m_fullOverlap = false;
  } else if (num_intersections_inside[0] == 0 && num_intersections_inside[1] == 2) {
    m_fullOverlap = false;
  }

  // reset degenerate intersections back to full overlap
  if (numV[0] < 3 || numV[1] < 3) {
    m_fullOverlap = true;
  }

  // if full overlap then reset overlap-face vertex count
  if (m_fullOverlap) {
    numV[0] = mesh[0]->numberOfNodesPerElement();
    numV[1] = mesh[1]->numberOfNodesPerElement();
  }

  // allocate arrays to store the vertices for clipped or full face used either
  // in the interpen or full overlap calc
  constexpr int max_nodes_per_overlap = 8; // TODO confirm that this number may be 5
  RealT cfx1[max_nodes_per_overlap];  // cfx = clipped face x-coordinate
  RealT cfy1[max_nodes_per_overlap];
  RealT cfz1[max_nodes_per_overlap];

  RealT cfx2[max_nodes_per_overlap];  // cfx = clipped face x-coordinate
  RealT cfy2[max_nodes_per_overlap];
  RealT cfz2[max_nodes_per_overlap];

  if (!m_fullOverlap) {
    // populate segment-contact-plane intersection vertices
    for ( int m = 0; m < num_intersections[0]; ++m )
    {
      cfx1[m] = xInter[m];
      cfy1[m] = yInter[m];
      cfz1[m] = zInter[m];
    }
    for ( int n = 0; n < num_intersections[1]; ++n )
    {
      cfx2[n] = xInter[num_intersections[0] + n];
      cfy2[n] = yInter[num_intersections[0] + n];
      cfz2[n] = zInter[num_intersections[0] + n];
    }

    // populate the face 1 vertices that cross the contact plane
    int k = num_intersections[0];
    for ( int m = 0; m < mesh[0]->numberOfNodesPerElement(); ++m ) {
      if ( interpenVertex1[m] != -1 ) {
        int fNodeId = mesh[0]->getGlobalNodeId( element_id[0], interpenVertex1[m] );
        cfx1[k] = mesh[0]->getPosition()[0][fNodeId];
        cfy1[k] = mesh[0]->getPosition()[1][fNodeId];
        cfz1[k] = mesh[0]->getPosition()[2][fNodeId];
        ++k;
      }
    }

    // populate the face 2 vertices that cross the contact plane
    k = num_intersections[1];
    for ( int n = 0; n < mesh[1]->numberOfNodesPerElement(); ++n ) {
      if ( interpenVertex2[n] != -1 ) {
        int fNodeId = mesh[1]->getGlobalNodeId( element_id[1], interpenVertex2[n] );
        cfx2[k] = mesh[1]->getPosition()[0][fNodeId];
        cfy2[k] = mesh[1]->getPosition()[1][fNodeId];
        cfz2[k] = mesh[1]->getPosition()[2][fNodeId];
        ++k;
      }
    }

  } // end if (m_fullOverlap)
  else { // populate the face vertex array with the face coordinates themselves
    // face 1
    for ( int m = 0; m < mesh[0]->numberOfNodesPerElement(); ++m ) {
        int fNodeId = mesh[0]->getGlobalNodeId( element_id[0], m );
        cfx1[m] = mesh[0]->getPosition()[0][fNodeId];
        cfy1[m] = mesh[0]->getPosition()[1][fNodeId];
        cfz1[m] = mesh[0]->getPosition()[2][fNodeId];
    }
   
    // face 2
    for ( int n = 0; n < mesh[1]->numberOfNodesPerElement(); ++n ) {
        int fNodeId = mesh[1]->getGlobalNodeId( element_id[1], n );
        cfx1[n] = mesh[1]->getPosition()[0][fNodeId];
        cfy1[n] = mesh[1]->getPosition()[1][fNodeId];
        cfz1[n] = mesh[1]->getPosition()[2][fNodeId];
    }
  } // end else (m_fullOverlap)

  // declare projected coordinate arrays
  RealT cfx1_proj[max_nodes_per_overlap];
  RealT cfy1_proj[max_nodes_per_overlap];
  RealT cfz1_proj[max_nodes_per_overlap];

  RealT cfx2_proj[max_nodes_per_overlap];
  RealT cfy2_proj[max_nodes_per_overlap];
  RealT cfz2_proj[max_nodes_per_overlap];

  // project overlap-calc face coordinates to contact plane
  for ( int i = 0; i < numV[0]; ++i ) {
    ProjectPointToPlane( cfx1[i], cfy1[i], cfz1[i], m_nX, m_nY, m_nZ, m_cX, m_cY, m_cZ, cfx1_proj[i], cfy1_proj[i],
                         cfz1_proj[i] );
  }

  for ( int i = 0; i < numV[1]; ++i ) {
    ProjectPointToPlane( cfx2[i], cfy2[i], cfz2[i], m_nX, m_nY, m_nZ, m_cX, m_cY, m_cZ, cfx2_proj[i], cfy2_proj[i],
                         cfz2_proj[i] );
  }

  // declare local coordinate pointers
  RealT cfx1_loc[max_nodes_per_overlap];
  RealT cfy1_loc[max_nodes_per_overlap];

  RealT cfx2_loc[max_nodes_per_overlap];
  RealT cfy2_loc[max_nodes_per_overlap];

  // convert global coords to local contact plane coordinates
  GlobalTo2DLocalCoords( cfx1_proj, cfy1_proj, cfz1_proj, m_e1X, m_e1Y, m_e1Z, m_e2X, m_e2Y, m_e2Z, m_cX, m_cY, m_cZ,
                         cfx1_loc, cfy1_loc, numV[0] );

  GlobalTo2DLocalCoords( cfx2_proj, cfy2_proj, cfz2_proj, m_e1X, m_e1Y, m_e1Z, m_e2X, m_e2Y, m_e2Z, m_cX, m_cY, m_cZ,
                         cfx2_loc, cfy2_loc, numV[1] );

  // reorder potentially unordered set of vertices for interpen calcs
  // Note: this routine will order both sets of vertices in counter clockwise orientation.
  //       Intersection2DPolygon() assumes consistent ordering between faces
  if (!m_fullOverlap) {
    PolyReorder( cfx1_loc, cfy1_loc, nullptr, numV[0] );
    PolyReorder( cfx2_loc, cfy2_loc, nullptr, numV[1] );
  } else { // use ElemReverse() per original implementation for full overlaps
    ElemReverse( cfx2_loc, cfy1_loc, numV[1] );
  }

  // call intersection routine to get intersecting polygon
  RealT pos_tol = params.len_collapse_ratio * axom::utilities::max( mesh[0]->getFaceRadius()[ element_id[0] ],
                                                                    mesh[1]->getFaceRadius()[ element_id[1] ] );
  RealT len_tol = pos_tol;
  FaceGeomError inter_err =
      Intersection2DPolygon( cfx1_loc, cfy1_loc, numV[0], cfx2_loc, cfy2_loc, numV[1], pos_tol, len_tol, m_polyLocX,
                             m_polyLocY, m_numPolyVert, m_area, true );

  if ( inter_err != NO_FACE_GEOM_ERROR ) {
    return inter_err;
  }

  // check overlap area to area tol
  if ( m_area < m_areaMin ) {
    return NO_OVERLAP;
  }

  // handle the case where the actual polygon with connectivity
  // and computed vertex coordinates becomes degenerate due to
  // either position tolerances (segment-segment intersections)
  // or length tolerances (intersecting polygon segment lengths)
  if ( m_numPolyVert < 3 ) {
#ifdef TRIBOL_USE_HOST
    SLIC_DEBUG( "degenerate polygon intersection detected.\n" );
#endif
    return DEGENERATE_OVERLAP;
  }

  // compute the local vertex averaged centroid of overlapping polygon
  RealT cZ;  // dummy z component for call to routine
  VertexAvgCentroid( m_polyLocX, m_polyLocY, nullptr, m_numPolyVert, m_overlapCX, m_overlapCY, cZ );

  // Transform local vertex coordinates to global coordinates for the
  // current projection of the polygonal overlap
  for ( int i = 0; i < m_numPolyVert; ++i ) {
    m_polyX[i] = 0.0;
    m_polyY[i] = 0.0;
    m_polyZ[i] = 0.0;

    local2DToGlobalCoords( m_polyLocX[i], m_polyLocY[i], m_polyX[i], m_polyY[i], m_polyZ[i] );
  }

  // check polygonal vertex ordering with common plane normal
  PolyReorderWithNormal( m_polyX, m_polyY, m_polyZ, m_numPolyVert, m_nX, m_nY, m_nZ );

  // store the local intersection polygons on the contact plane object,
  // Note: we don't have to fix the ordering of the vertices consistent with the face's
  //       outward unit normal since this data is just for visualization, not physics
  //       calculations

  m_numInterpenPoly1Vert = numV[0];
  m_numInterpenPoly2Vert = numV[1];

  // transform local interpenetration overlaps to global coords for the
  // current polygonal overlap
  for ( int i = 0; i < m_numInterpenPoly1Vert; ++i ) {
    Local2DToGlobalCoords( cfx1_loc[i], cfy1_loc[i], m_e1X, m_e1Y, m_e1Z, m_e2X,
                           m_e2Y, m_e2Z, m_cX, m_cY, m_cZ, m_interpenG1X[i], m_interpenG1Y[i],
                           m_interpenG1Z[i] );
  }

  for ( int i = 0; i < m_numInterpenPoly2Vert; ++i ) {
    Local2DToGlobalCoords( cfx2_loc[i], cfy2_loc[i], m_e1X, m_e1Y, m_e1Z, m_e2X,
                           m_e2Y, m_e2Z, m_cX, m_cY, m_cZ, m_interpenG2X[i], m_interpenG2Y[i],
                           m_interpenG2Z[i] );
  }

  // Now that all local-to-global projections have occurred,
  // relocate the contact plane based on the most up-to-date
  // contact plane centroid and recompute the gap. For interpenOverlap,
  // the contact plane is updated and this just amounts to a gap
  // computation. For the fullOverlap case, this may relocate
  // the contact plane in space.
  //
  // Warning:
  // Make sure that any local to global transformations have
  // occurred prior to this call.
  this->resetPlanePointAndCentroidGap( m1, m2 );
    
  // for auto-contact, remove contact candidacy for full-overlap
  // face-pairs with interpenetration exceeding contact penetration fraction.
  // Note, this check is solely meant to exclude face-pairs composed of faces
  // on opposite sides of thin structures/plates
  //
  // Note: Interpen overlaps can not occur between faces on opposing sides of thin structures
  //       without element inversion. Also, if a thin body is in self-contact we can't distinguish
  //       between opposing faces where each face is on one side of the thin structure, or a body
  //       in self-contact where true contact pairs have passed through one another beyond the
  //       interpenetration limit. In the latter case, we will simply flag these pairs and they
  //       will have to lose contact.
  //
  // Recall that interpen gaps are negative
  if ( m_fullOverlap ) {
    if ( ExceedsMaxAutoInterpen( m1, m2, element_id[0], element_id[1], params, m_gap ) ) {
      return EXCEEDS_AUTO_CONTACT_LENGTH_SCALE;
    }
  }

  // REPROJECT the overlapping polygon onto the new contact plane
  for ( int i = 0; i < m_numPolyVert; ++i ) {
    ProjectPointToPlane( m_polyX[i], m_polyY[i], m_polyZ[i], m_nX, m_nY, m_nZ, m_cX, m_cY,
                         m_cZ, m_polyX[i], m_polyY[i], m_polyZ[i] );
  }
  
  // REPROJECT the global coordinates of the interpenetrating polygon as projection onto the common plane
  for ( int i = 0; i < m_numInterpenPoly1Vert; ++i ) {
    ProjectPointToPlane( m_interpenG1X[i], m_interpenG1Y[i], m_interpenG1Z[i], m_nX, m_nY, m_nZ, m_cX, m_cY,
                         m_cZ, m_interpenG1X[i], m_interpenG1Y[i], m_interpenG1Z[i] );
  }

  for ( int i = 0; i < m_numInterpenPoly2Vert; ++i ) {
    ProjectPointToPlane( m_interpenG2X[i], m_interpenG2Y[i], m_interpenG2Z[i], m_nX, m_nY, m_nZ, m_cX, m_cY,
                         m_cZ, m_interpenG2X[i], m_interpenG2Y[i], m_interpenG2Z[i] );
  }

  return NO_FACE_GEOM_ERROR;

}  // end CommonPlanePair::computeOverlap3D()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomError CommonPlanePair::computeOverlap2D( const MeshData::Viewer& m1,
                                                                    const MeshData::Viewer& m2,
                                                                    const Parameters& params )
{
  // all edge-edge interactions suitable for an interpenetration overlap
  // calculation are edges that intersect at a single point
  int edgeId1 = getCpElementId1();
  int edgeId2 = getCpElementId2();

  constexpr int max_dim = 2;
  RealT cx1[max_dim], cx2[max_dim];
  RealT fn1[max_dim], fn2[max_dim];
  m1.getFaceCentroid( edgeId1, cx1 );
  m2.getFaceCentroid( edgeId2, cx2 );
  m1.getFaceNormal( edgeId1, fn1 );
  m2.getFaceNormal( edgeId2, fn2 );

  int nodeA1 = m1.getGlobalNodeId( edgeId1, 0 );
  int nodeB1 = m1.getGlobalNodeId( edgeId1, 1 );
  int nodeA2 = m2.getGlobalNodeId( edgeId2, 0 );
  int nodeB2 = m2.getGlobalNodeId( edgeId2, 1 );

  RealT xposA1 = m1.getPosition()[0][nodeA1];
  RealT yposA1 = m1.getPosition()[1][nodeA1];
  RealT xposB1 = m1.getPosition()[0][nodeB1];
  RealT yposB1 = m1.getPosition()[1][nodeB1];

  RealT xposA2 = m2.getPosition()[0][nodeA2];
  RealT yposA2 = m2.getPosition()[1][nodeA2];
  RealT xposB2 = m2.getPosition()[0][nodeB2];
  RealT yposB2 = m2.getPosition()[1][nodeB2];

  RealT xInter, yInter;
  bool duplicatePoint = false;

  // check if the segments intersect thus triggering an interpen overlap calc
  RealT len_tol =
      params.len_collapse_ratio * axom::utilities::max( m1.getFaceRadius()[edgeId1], m2.getFaceRadius()[edgeId2] );

  bool edgeIntersect = SegmentIntersection2D( xposA1, yposA1, xposB1, yposB1, xposA2, yposA2, xposB2, yposB2, nullptr,
                                              xInter, yInter, duplicatePoint, len_tol );

  // if there is no edge-edge intersection point or the intersection
  // point exists at an edge vertex, then switch to full overlap
  if (!edgeIntersect || duplicatePoint) {
    m_fullOverlap = true;
  }

  RealT x1_to_project[ max_dim ], y1_to_project[ max_dim ];
  RealT x2_to_project[ max_dim ], y2_to_project[ max_dim ];

  // project unique intersection point to the contact plane.
  // The contact plane should have been properly located prior
  // to this subroutine, in which case the intersection point lies
  // on the contact plane (segment). We can still do this projection
  // to be safe and the routine will handle a point that is already
  // on the plane
  //RealT xInterProj, yInterProj;
  //ProjectPointToSegment( xInter, yInter, m_nX, m_nY, m_cX, m_cY, xInterProj, yInterProj );

  if ( !m_fullOverlap ) {
    // now isolate which vertex on edge 1 and which vertex on edge 2 lie
    // on the "wrong" side of the contact plane.

    // define vectors between each edge's vertices and the OTHER edge's centroid
    int interId1 = -1;
    int interId2 = -1;
    int k1 = 0;
    int k2 = 0;
    for ( int i = 0; i < m1.numberOfNodesPerElement(); ++i ) {
      int nodeId1 = m1.getGlobalNodeId( edgeId1, i );
      int nodeId2 = m2.getGlobalNodeId( edgeId2, i );
      RealT lvx1 = m1.getPosition()[0][nodeId1] - cx2[0];
      RealT lvy1 = m1.getPosition()[1][nodeId1] - cx2[1];
      RealT lvx2 = m2.getPosition()[0][nodeId2] - cx1[0];
      RealT lvy2 = m2.getPosition()[1][nodeId2] - cx1[1];

      // dot each vector with the OTHER edge's normal
      RealT proj1 = lvx1 * fn2[0] + lvy1 * fn2[1];
      RealT proj2 = lvx2 * fn1[0] + lvy2 * fn1[1];

      // check the projection to detect interpenetration and
      // mark the node id if true
      if ( proj1 > 0.0 ) {
        interId1 = i;
        ++k1;
      }
      if ( proj2 < 0.0 ) {
        interId2 = i;
        ++k2;
      }
    } // end loop over nodes

    // Debug check the number of interpenetrating vertices
    if ( k1 > 1 || k2 > 1 ) {
#ifdef TRIBOL_USE_HOST
      SLIC_DEBUG( "CommonPlanePair::computeOverlap2D() more than 2 interpenetrating vertices detected; "
                  << "check for degenerate geometry for edges (" << edgeId1 << ", " << edgeId2 << ") on meshes ("
                  << m1.meshId() << ", " << m2.meshId() << ")." );
#endif
      return DEGENERATE_OVERLAP;
    }

    // populate arrays holding the interpenetrating edge portions to be
    // used in computing the overlap
    int nodeInter1 = m1.getGlobalNodeId( edgeId1, interId1 );
    x1_to_project[0] = m1.getPosition()[0][nodeInter1];
    x1_to_project[1] = xInter;
    y1_to_project[0] = m1.getPosition()[1][nodeInter1];
    y1_to_project[1] = yInter;

    int nodeInter2 = m2.getGlobalNodeId( edgeId2, interId2 );
    x2_to_project[0] = m2.getPosition()[0][nodeInter2];
    x2_to_project[1] = xInter;
    y2_to_project[0] = m2.getPosition()[1][nodeInter2];
    y2_to_project[1] = yInter;

    // compute minimum interpenetrating edge-portion length. When projected
    // onto the common plane this minimum length will correspond to the overlap
    RealT vix1 = x1_to_project[1] - x1_to_project[0];
    RealT viy1 = y1_to_project[1] - y1_to_project[0];
    RealT vix2 = x2_to_project[1] - x2_to_project[0];
    RealT viy2 = y2_to_project[1] - y2_to_project[0];

    // determine magnitude of each vector
    RealT mag1 = magnitude( vix1, viy1 );
    RealT mag2 = magnitude( vix2, viy2 );

    // the interpenetration overlap length is the minimum of the above
    // vectors
    m_area = ( mag1 <= mag2 ) ? mag1 : mag2;

    // determine the edge vertex that forms the overlap segment along
    // with the intersection point previously computed
    RealT vx1 = ( mag1 <= mag2 ) ? m1.getPosition()[0][nodeInter1] : m2.getPosition()[0][nodeInter2];

    RealT vy1 = ( mag1 <= mag2 ) ? m1.getPosition()[1][nodeInter1] : m2.getPosition()[1][nodeInter2];

    RealT vx2 = xInter;
    RealT vy2 = yInter;

    // allocate space to store the interpen vertices for visualization
    m_numInterpenPoly1Vert = 2;
    m_numInterpenPoly2Vert = 2;

    m_interpenG1X[0] = x1_to_project[0];
    m_interpenG1Y[0] = y1_to_project[0];
    m_interpenG1X[1] = xInter;
    m_interpenG1Y[1] = yInter;

    m_interpenG2X[0] = x2_to_project[0];
    m_interpenG2Y[0] = y2_to_project[0];
    m_interpenG2X[1] = xInter;
    m_interpenG2Y[1] = yInter;

    // project the node and intersection vertex associated with the shortest interpenetrating
    // edge-portion to the common plane
    ProjectPointToSegment( vx1, vy1, m_nX, m_nY, m_cX, m_cY, m_segX[0], m_segY[0] );
    ProjectPointToSegment( vx2, vy2, m_nX, m_nY, m_cX, m_cY, m_segX[1], m_segY[1] );

    // compute the overlap area
    m_area = magnitude( m_segX[1] - m_segX[0], m_segY[1] - m_segY[0] );

    if (m_area < m_areaMin) {
      return NO_OVERLAP;
    }

    // compute the new contact plane overlap centroid (segment point)
    m_cX = 0.5 * ( m_segX[0] + m_segX[1] );
    m_cY = 0.5 * ( m_segY[0] + m_segY[1] );
    m_cZ = 0.;

    this->resetPlanePointAndCentroidGap( m1, m2 );

    // reproject the overlap points to the new common plane
    ProjectPointToSegment( m_segX[0], m_segY[0], m_nX, m_nY, m_cX, m_cY, m_segX[0], m_segY[0] );
    ProjectPointToSegment( m_segX[1], m_segY[1], m_nX, m_nY, m_cX, m_cY, m_segX[1], m_segY[1] );

    // reproject the global intepen face vertices onto the new contact plane 
    ProjectPointToSegment( m_interpenG1X[0], m_interpenG1Y[0], m_nX, m_nY, m_cX, m_cY, m_interpenG1X[0], m_interpenG1Y[0] );
    ProjectPointToSegment( m_interpenG2X[0], m_interpenG2Y[0], m_nX, m_nY, m_cX, m_cY, m_interpenG2X[0], m_interpenG2Y[0] );

  } else if (m_fullOverlap) { // end if (!m_fullOverlap)
    
    RealT projX1[ max_dim ], projY1[ max_dim ];
    RealT projX2[ max_dim ], projY2[ max_dim ];

    ProjectEdgeNodesToSegment( mesh1, edgeId1, this->m_nX, this->m_nY, this->m_cX, this->m_cY, &projX1[0], &projY1[0] );
    ProjectEdgeNodesToSegment( mesh2, edgeId2, this->m_nX, this->m_nY, this->m_cX, this->m_cY, &projX2[0], &projY2[0] );
    FaceGeomError segError = CheckSegOverlap( &projX1[0], &projY1[0], &projX2[0], &projY2[0], mesh1.numberOfNodesPerElement(),
                                              mesh2.numberOfNodesPerElement(), &m_segX[0], &m_segY[0], m_area );

    if (segError != NO_FACE_GEOM_ERROR) {
       return segError;
    }

    if (m_area < m_areaMin) {
      return NO_OVERLAP;
    }

    // compute the new contact plane overlap centroid (segment point)
    m_cX = 0.5 * ( m_segX[0] + m_segX[1] );
    m_cY = 0.5 * ( m_segY[0] + m_segY[1] );
    m_cZ = 0.;

    this->resetPlanePointAndCentroidGap( m1, m2 );

    // reproject the overlap points to the new common plane
    ProjectPointToSegment( m_segX[0], m_segY[0], m_nX, m_nY, m_cX, m_cY, m_segX[0], m_segY[0] );
    ProjectPointToSegment( m_segX[1], m_segY[1], m_nX, m_nY, m_cX, m_cY, m_segX[1], m_segY[1] );
  
    // for auto-contact, remove contact candidacy for full-overlap
    // face-pairs with interpenetration exceeding contact penetration fraction.
    // Note, this check is solely meant to exclude face-pairs composed of faces
    // on opposite sides of thin structures/plates
    //
    // Recall that interpen gaps are negative
    if ( ExceedsMaxAutoInterpen( m1, m2, edgeId1, edgeId2, params, this->m_gap ) ) {
      return EXCEEDS_AUTO_CONTACT_LENGTH_SCALE;
    }

    // initialize un-used interpen coordinates to work with visualization
    // TODO SRW confirm if this is needed
    this->m_numInterpenPoly1Vert = 2;
    this->m_numInterpenPoly2Vert = 2;

    for ( int i = 0; i < 2; ++i ) {
      // set the interpen vertices to the full overlap vertices
      this->m_interpenG1X[i] = 0.0;
      this->m_interpenG1Y[i] = 0.0;
      this->m_interpenG2X[i] = 0.0;
      this->m_interpenG2Y[i] = 0.0;
    }

  } // end if (m_fullOverlap)

  return NO_FACE_GEOM_ERROR;

}  // end CommonPlanePair::computeOverlap2D()

//------------------------------------------------------------------------------

}  // namespace tribol
