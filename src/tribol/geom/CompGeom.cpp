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
template <typename T>
TRIBOL_HOST_DEVICE FaceGeomException CheckInterfacePairByMethod(
    InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2, const Parameters& params,
    ContactCase TRIBOL_UNUSED_PARAM( cCase ), bool& isInteracting, CompGeom::Viewer& cg, IndexT* plane_ct )
{
  auto dim = mesh1.spatialDimension();
  T my_plane( &pair, params, dim );
  FaceGeomException face_err = NO_FACE_GEOM_EXCEPTION;
  if ( dim == 3 ) {
    face_err = my_plane.checkFacePair( mesh1, mesh2 );
  } else {
    face_err = my_plane.checkEdgePair( mesh1, mesh2 );
  }

  if ( face_err != NO_FACE_GEOM_EXCEPTION ) {
    isInteracting = false;
#ifdef TRIBOL_USE_HOST
    SLIC_DEBUG( "face_err: " << face_err );
#endif
  } else if ( my_plane.m_inContact ) {
#ifdef TRIBOL_USE_RAJA
    auto idx = RAJA::atomicInc<RAJA::auto_atomic>( plane_ct );
#else
    auto idx = *plane_ct;
    ++( *plane_ct );
#endif
    cg.getPlane<T>( idx ) = my_plane;
    isInteracting = true;
  } else {
    isInteracting = false;
  }

  return face_err;
}

TRIBOL_HOST_DEVICE FaceGeomException CheckInterfacePair( InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                         const MeshData::Viewer& mesh2, const Parameters& params,
                                                         ContactMethod cMethod, ContactCase cCase, bool& isInteracting,
                                                         CompGeom::Viewer& cg, IndexT* plane_ct )
{
  FaceGeomException face_err = NO_FACE_GEOM_EXCEPTION;

  // note: will likely need the ContactCase for specialized
  // geometry check(s)/routine(s)

  switch ( cMethod ) {
    case MORTAR_WEIGHTS:
    case SINGLE_MORTAR: {
      face_err =
          CheckInterfacePairByMethod<MortarPlanePair>( pair, mesh1, mesh2, params, cCase, isInteracting, cg, plane_ct );
      break;
    }
    case COMMON_PLANE: {
      face_err =
          CheckInterfacePairByMethod<CommonPlanePair>( pair, mesh1, mesh2, params, cCase, isInteracting, cg, plane_ct );
      break;
    }
    case ALIGNED_MORTAR: {
      face_err = CheckInterfacePairByMethod<AlignedMortarPlanePair>( pair, mesh1, mesh2, params, cCase, isInteracting,
                                                                     cg, plane_ct );
      break;
    }
    default: {
      // don't do anything
      break;
    }
  }  // end switch

  return face_err;

}  // end CheckInterfacePair()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE bool CommonPlanePair::exceedsMaxAutoInterpen( const MeshData::Viewer& mesh1,
                                                                 const MeshData::Viewer& mesh2, const int faceId1,
                                                                 const int faceId2, const Parameters& params,
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
TRIBOL_HOST_DEVICE ContactPlanePair::ContactPlanePair( InterfacePair* pair, const Parameters& params, const int dim )
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
  for ( int i = 0; i < max_nodes_per_overlap; ++i ) {
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
TRIBOL_HOST_DEVICE CommonPlanePair::CommonPlanePair( InterfacePair* pair, const Parameters& params, const int dim )
    : ContactPlanePair( pair, params, dim ),
      m_numInterpenPoly1Vert( 0 ),
      m_numInterpenPoly2Vert( 0 ),
      m_velGap( 0.0 ),
      m_ratePressure( 0.0 ),
      m_pressure( 0.0 )
{
  for ( int i = 0; i < max_nodes_per_intersection; ++i ) {
    m_interpenG1X[i] = 0.;
    m_interpenG1Y[i] = 0.;
    m_interpenG1Z[i] = 0.;

    m_interpenG2X[i] = 0.;
    m_interpenG2Y[i] = 0.;
    m_interpenG2Z[i] = 0.;
  }
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomException CommonPlanePair::checkFacePair( const MeshData::Viewer& mesh1,
                                                                     const MeshData::Viewer& mesh2 )
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

  // get face normals
  constexpr int max_dim = 3;
  RealT fn1[max_dim], fn2[max_dim];
  mesh1.getFaceNormal( element_id1, fn1 );
  mesh2.getFaceNormal( element_id2, fn2 );

  // get face centroids
  RealT cx1[max_dim], cx2[max_dim];
  mesh1.getFaceCentroid( element_id1, cx1 );
  mesh2.getFaceCentroid( element_id2, cx2 );

  // project face vertices onto FACE-PLANE defined by face centroid-normal
  ProjectFaceNodesToPlane( mesh1, element_id1, fn1[0], fn1[1], fn1[2], cx1[0], cx1[1], cx1[2], &m_x1_prime[0],
                           &m_y1_prime[0], &m_z1_prime[0] );
  ProjectFaceNodesToPlane( mesh2, element_id2, fn2[0], fn2[1], fn2[2], cx2[0], cx2[1], cx2[2], &m_x2_prime[0],
                           &m_y2_prime[0], &m_z2_prime[0] );

  // CHECK #6: check if the two faces overlap in a projected sense.
  // To do this check we need to use the contact plane object, which will
  // have its own local basis that needs to be defined

  ////////////////////////////////////////////////
  // Compute Common Plane Overlap with Vertices //
  ////////////////////////////////////////////////

  // compute common plane normal, centroid, local basis and area tolerance
  computePlaneData( mesh1, mesh2 );

  // mark the convexity of each face
  constexpr int max_nodes = 4;
  RealT x1_loc[max_nodes];
  RealT y1_loc[max_nodes];
  RealT x2_loc[max_nodes];
  RealT y2_loc[max_nodes];
  GlobalTo2DLocalCoords( &m_x1_prime[0], &m_y1_prime[0], &m_z1_prime[0], m_e1X, m_e1Y, m_e1Z, m_e2X, m_e2Y, m_e2Z, m_cX,
                         m_cY, m_cZ, &x1_loc[0], &y1_loc[0], mesh1.numberOfNodesPerElement() );
  GlobalTo2DLocalCoords( &m_x2_prime[0], &m_y2_prime[0], &m_z2_prime[0], m_e1X, m_e1Y, m_e1Z, m_e2X, m_e2Y, m_e2Z, m_cX,
                         m_cY, m_cZ, &x2_loc[0], &y2_loc[0], mesh2.numberOfNodesPerElement() );

  m_face1_convex = IsConvex( x1_loc, y1_loc, mesh1.numberOfNodesPerElement() );
  m_face2_convex = IsConvex( x2_loc, y2_loc, mesh2.numberOfNodesPerElement() );

  // explicitly call compute overlap routine for common plane so CUDA can determine stack size
  FaceGeomException interpen_err = CommonPlanePair::computeOverlap3D(
      &m_x1_prime[0], &m_y1_prime[0], &m_z1_prime[0], &m_x2_prime[0], &m_y2_prime[0], &m_z2_prime[0], mesh1, mesh2 );

  if ( interpen_err != NO_FACE_GEOM_EXCEPTION ) {
    this->m_inContact = false;
    return interpen_err;
  }

  this->m_inContact = true;
  return NO_FACE_GEOM_EXCEPTION;

}  // end CommonPlanePair::checkFacePair()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomException CommonPlanePair::checkEdgePair( const MeshData::Viewer& mesh1,
                                                                     const MeshData::Viewer& mesh2 )
{
  // Note: Checks #1-#5 are done in the binning

  IndexT element_id1 = this->getCpElementId1();
  IndexT element_id2 = this->getCpElementId2();

  // set face "prime" coordinates to current configuration coordinates so we can pull these
  // off the plane object later
  for ( int i = 0; i < mesh1.numberOfNodesPerElement(); ++i ) {
    const int nodeId = mesh1.getGlobalNodeId( element_id1, i );
    m_x1_prime[i] = mesh1.getPosition()[0][nodeId];
    m_y1_prime[i] = mesh1.getPosition()[1][nodeId];
  }

  for ( int i = 0; i < mesh2.numberOfNodesPerElement(); ++i ) {
    const int nodeId = mesh2.getGlobalNodeId( element_id2, i );
    m_x2_prime[i] = mesh2.getPosition()[0][nodeId];
    m_y2_prime[i] = mesh2.getPosition()[1][nodeId];
  }

  // CHECK #6: compute the projected length of overlap on the contact plane.
  // At this point the edges are proximate and likely have a positive
  // projected length of overlap.

  // compute common-plane point-normal data. At this point we don't know where to properly
  // locate the common plane centroid so we just take the average of the two face centroids
  computePlaneData( mesh1, mesh2 );

  // explicitly call compute overlap routine for common plane so CUDA can determine stack size
  FaceGeomException interpen_err = CommonPlanePair::computeOverlap2D( mesh1, mesh2 );
  if ( interpen_err != NO_FACE_GEOM_EXCEPTION ) {
    this->m_inContact = false;
    return interpen_err;
  }

  this->m_inContact = true;
  return NO_FACE_GEOM_EXCEPTION;

}  // end CommonPlanePair::checkEdgePair()

//------------------------------------------------------------------------------
// Mortar Plane Routines
//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE MortarPlanePair::MortarPlanePair( InterfacePair* pair, const Parameters& params, const int dim )
    : ContactPlanePair( pair, params, dim )
{
  // no-op
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomException MortarPlanePair::checkFacePair( const MeshData::Viewer& mesh1,
                                                                     const MeshData::Viewer& mesh2 )
{
  // Note: Checks #1-#5 are done in the binning

  // alias variables off the InterfacePair
  IndexT element_id1 = this->getCpElementId1();
  IndexT element_id2 = this->getCpElementId2();

  ////////////////////////////
  // Planar Face Projection //
  ////////////////////////////

  // set face "prime" coordinates to current configuration coordinates for full overlap methods
  for ( int i = 0; i < mesh1.numberOfNodesPerElement(); ++i ) {
    const int nodeId = mesh1.getGlobalNodeId( element_id1, i );
    m_x1_prime[i] = mesh1.getPosition()[0][nodeId];
    m_y1_prime[i] = mesh1.getPosition()[1][nodeId];
    m_z1_prime[i] = mesh1.getPosition()[2][nodeId];
  }

  for ( int i = 0; i < mesh2.numberOfNodesPerElement(); ++i ) {
    const int nodeId = mesh2.getGlobalNodeId( element_id2, i );
    m_x2_prime[i] = mesh2.getPosition()[0][nodeId];
    m_y2_prime[i] = mesh2.getPosition()[1][nodeId];
    m_z2_prime[i] = mesh2.getPosition()[2][nodeId];
  }

  ////////////////////////////////////////////////
  // Compute Mortar Plane Overlap with Vertices //
  ////////////////////////////////////////////////

  // compute common plane normal, centroid, local basis and area tolerance
  computePlaneData( mesh1, mesh2 );

  // the contact plane has to be properly located prior to computing the interpen overlap
  FaceGeomException interpen_err = this->computeOverlap3D(
      &m_x1_prime[0], &m_y1_prime[0], &m_z1_prime[0], &m_x2_prime[0], &m_y2_prime[0], &m_z2_prime[0], mesh1, mesh2 );

  if ( interpen_err != NO_FACE_GEOM_EXCEPTION ) {
    this->m_inContact = false;
    return interpen_err;
  }

  this->m_inContact = true;
  return NO_FACE_GEOM_EXCEPTION;

}  // end MortarPlanePair::checkFacePair()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomException MortarPlanePair::computeOverlap3D( const RealT* x1, const RealT* y1,
                                                                        const RealT* z1, const RealT* x2,
                                                                        const RealT* y2, const RealT* z2,
                                                                        const MeshData::Viewer& m1,
                                                                        const MeshData::Viewer& m2 )
{
  IndexT element_id1 = this->getCpElementId1();
  IndexT element_id2 = this->getCpElementId2();

  // project face vertex coordinates to contact plane
  ProjectPointsToPlane( x1, y1, z1, this->m_nX, this->m_nY, this->m_nZ, this->m_cX, this->m_cY, this->m_cZ,
                        &m_x1_bar[0], &m_y1_bar[0], &m_z1_bar[0], m1.numberOfNodesPerElement() );
  ProjectPointsToPlane( x2, y2, z2, this->m_nX, this->m_nY, this->m_nZ, this->m_cX, this->m_cY, this->m_cZ,
                        &m_x2_bar[0], &m_y2_bar[0], &m_z2_bar[0], m2.numberOfNodesPerElement() );

  // project the projected global nodal coordinates onto local
  // contact plane 2D coordinate system.
  constexpr int max_nodes_per_elem = 4;
  RealT x1_bar_local[max_nodes_per_elem];
  RealT y1_bar_local[max_nodes_per_elem];
  RealT x2_bar_local[max_nodes_per_elem];
  RealT y2_bar_local[max_nodes_per_elem];

  this->globalTo2DLocalCoords( &m_x1_bar[0], &m_y1_bar[0], &m_z1_bar[0], &x1_bar_local[0], &y1_bar_local[0],
                               m1.numberOfNodesPerElement() );
  this->globalTo2DLocalCoords( &m_x2_bar[0], &m_y2_bar[0], &m_z2_bar[0], &x2_bar_local[0], &y2_bar_local[0],
                               m2.numberOfNodesPerElement() );

  // compute the full intersection polygon vertex coordinates
  RealT* X1 = &x1_bar_local[0];
  RealT* Y1 = &y1_bar_local[0];
  RealT* X2 = &x2_bar_local[0];
  RealT* Y2 = &y2_bar_local[0];

  // assuming each face's vertices are ordered WRT that face's outward unit normal,
  // reorder face 2 vertices to be consistent with face 1. DO NOT CALL POLYREORDER()
  // to do this.
  ElemReverse( X2, Y2, m2.numberOfNodesPerElement() );

  // compute intersection polygon and area.
  RealT pos_tol = this->m_params.len_collapse_ratio *
                  axom::utilities::max( m1.getFaceRadius()[element_id1], m2.getFaceRadius()[element_id2] );
  RealT len_tol = pos_tol;
  FaceGeomException inter_err =
      Intersection2DPolygon( X1, Y1, m1.numberOfNodesPerElement(), X2, Y2, m2.numberOfNodesPerElement(), pos_tol,
                             len_tol, this->m_polyLocX, this->m_polyLocY, this->m_numPolyVert, this->m_area, false );

  if ( inter_err != NO_FACE_GEOM_EXCEPTION ) {
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

    this->local2DToGlobalCoords( this->m_polyLocX[i], this->m_polyLocY[i], this->m_polyX[i], this->m_polyY[i],
                                 this->m_polyZ[i] );
  }

  // compute the vertex averaged centroid of overlapping polygon
  VertexAvgCentroid( this->m_polyX, this->m_polyY, this->m_polyZ, this->m_numPolyVert, this->m_cX, this->m_cY,
                     this->m_cZ );

  // check polygonal vertex ordering with mortar plane normal
  PolyReorderWithNormal( this->m_polyX, this->m_polyY, this->m_polyZ, this->m_numPolyVert, this->m_nX, this->m_nY,
                         this->m_nZ );

  return NO_FACE_GEOM_EXCEPTION;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomException MortarPlanePair::checkEdgePair(
    const MeshData::Viewer& TRIBOL_UNUSED_PARAM( mesh1 ), const MeshData::Viewer& TRIBOL_UNUSED_PARAM( mesh2 ) )
{
  // no-op; implement when 2D mortar is implemented
  return NO_FACE_GEOM_EXCEPTION;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomException MortarPlanePair::computeOverlap2D(
    const MeshData::Viewer& TRIBOL_UNUSED_PARAM( m1 ), const MeshData::Viewer& TRIBOL_UNUSED_PARAM( m2 ) )
{
  // no-op
  return NO_FACE_GEOM_EXCEPTION;
}

//------------------------------------------------------------------------------
// Aligned Mortar Plane Routines
//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE AlignedMortarPlanePair::AlignedMortarPlanePair( InterfacePair* pair, const Parameters& params,
                                                                   const int dim )
    : ContactPlanePair( pair, params, dim )
{
  // no-op
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomException AlignedMortarPlanePair::checkFacePair( const MeshData::Viewer& mesh1,
                                                                            const MeshData::Viewer& mesh2 )
{
  // Note: Checks #1-#5 are done in the binning

  // alias variables off the InterfacePair
  IndexT element_id1 = this->getCpElementId1();
  IndexT element_id2 = this->getCpElementId2();

  // set face "prime" coordinates to current configuration coordinates for full overlap methods
  for ( int i = 0; i < mesh1.numberOfNodesPerElement(); ++i ) {
    const int nodeId = mesh1.getGlobalNodeId( element_id1, i );
    m_x1_prime[i] = mesh1.getPosition()[0][nodeId];
    m_y1_prime[i] = mesh1.getPosition()[1][nodeId];
    m_z1_prime[i] = mesh1.getPosition()[2][nodeId];
  }

  for ( int i = 0; i < mesh2.numberOfNodesPerElement(); ++i ) {
    const int nodeId = mesh2.getGlobalNodeId( element_id2, i );
    m_x2_prime[i] = mesh2.getPosition()[0][nodeId];
    m_y2_prime[i] = mesh2.getPosition()[1][nodeId];
    m_z2_prime[i] = mesh2.getPosition()[2][nodeId];
  }

  // Check #6 see if the two faces are aligned; hence, overlap area being face area

  // compute common plane normal, centroid, local basis and area tolerance
  computePlaneData( mesh1, mesh2 );
  FaceGeomException interpen_err = this->computeOverlap3D(
      &m_x1_prime[0], &m_y1_prime[0], &m_z1_prime[0], &m_x2_prime[0], &m_y2_prime[0], &m_z2_prime[0], mesh1, mesh2 );

  if ( interpen_err != NO_FACE_GEOM_EXCEPTION ) {
    this->m_inContact = false;
    return interpen_err;
  }

  this->m_inContact = true;
  return NO_FACE_GEOM_EXCEPTION;

}  // end AlignedMortarPlanePair::checkFacePair()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomException AlignedMortarPlanePair::checkEdgePair(
    const MeshData::Viewer& TRIBOL_UNUSED_PARAM( mesh1 ), const MeshData::Viewer& TRIBOL_UNUSED_PARAM( mesh2 ) )
{
  // no-op; implement when 2D aligned mortar is implemented
  return NO_FACE_GEOM_EXCEPTION;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomException AlignedMortarPlanePair::computeOverlap3D( const RealT* x1, const RealT* y1,
                                                                               const RealT* z1, const RealT* x2,
                                                                               const RealT* y2, const RealT* z2,
                                                                               const MeshData::Viewer& m1,
                                                                               const MeshData::Viewer& m2 )
{
  IndexT element_id2 = this->getCpElementId2();

  // project face vertex coordinates to contact plane
  ProjectPointsToPlane( x1, y1, z1, this->m_nX, this->m_nY, this->m_nZ, this->m_cX, this->m_cY, this->m_cZ,
                        &m_x1_bar[0], &m_y1_bar[0], &m_z1_bar[0], m1.numberOfNodesPerElement() );
  ProjectPointsToPlane( x2, y2, z2, this->m_nX, this->m_nY, this->m_nZ, this->m_cX, this->m_cY, this->m_cZ,
                        &m_x2_bar[0], &m_y2_bar[0], &m_z2_bar[0], m2.numberOfNodesPerElement() );

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
  VertexAvgCentroid( this->m_polyX, this->m_polyY, this->m_polyZ, this->m_numPolyVert, this->m_cX, this->m_cY,
                     this->m_cZ );

  m_gap = scalarGap;
  m_area = m2.getElementAreas()[element_id2];

  return NO_FACE_GEOM_EXCEPTION;
}  // end AlignedMortarPlanePair::computeOverlap3D()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomException AlignedMortarPlanePair::computeOverlap2D(
    const MeshData::Viewer& TRIBOL_UNUSED_PARAM( m1 ), const MeshData::Viewer& TRIBOL_UNUSED_PARAM( m2 ) )
{
  // no-op
  return NO_FACE_GEOM_EXCEPTION;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void CommonPlanePair::resetPlanePointAndCentroidGap( const MeshData::Viewer& m1,
                                                                        const MeshData::Viewer& m2 )
{
  // reset the common plane centroid based on the overlapping polygon vertices.
  // In 3D use the area centroid, in 2D use the vertex avg. centroid
  if ( m_dim == 3 ) {
    // construct array of polygon overlap vertex coordinates
    constexpr int max_dim = 3;
    constexpr int max_nodes_per_overlap = 10;
    RealT xVert[max_dim * max_nodes_per_overlap];

    for ( IndexT j{ 0 }; j < this->m_numPolyVert; ++j ) {
      xVert[m_dim * j] = this->m_polyX[j];
      xVert[m_dim * j + 1] = this->m_polyY[j];
      xVert[m_dim * j + 2] = this->m_polyZ[j];
    }

    PolyAreaCentroid( &xVert[0], this->m_dim, this->m_numPolyVert, this->m_cX, this->m_cY, this->m_cZ );

  } else {
    // compute the new contact plane overlap centroid (segment point)
    this->m_cX = 0.5 * ( this->m_polyX[0] + this->m_polyX[1] );
    this->m_cY = 0.5 * ( this->m_polyY[0] + this->m_polyY[1] );
    this->m_cZ = 0.;
  }

  RealT radius1 = m1.getFaceRadius()[this->getCpElementId1()];
  RealT radius2 = m2.getFaceRadius()[this->getCpElementId2()];
  RealT radius = ( radius1 > radius2 ) ? radius1 : radius2;

  // scale the centroidGap projections using the updated effective binning scale
  // times the max face radius premultiplied by a safety factor to ensure
  // we find the overlap-centroid-to-face intersection
  RealT fs = 10.;

  // I can get binning proximity off the parameters, but the effective binning takes
  // into account LOR factor that lives on the coupling scheme. Effective binning scale
  // is likely what we want to use here to be consistent with binning? DO NOT set binning
  // proximity on parameters to effective. This resulted in failed mfem tests.
  //
  // To work around this just increase the safety factor (fs) used
  RealT scale = fs * this->m_params.binning_proximity_scale * radius;

  // compute the gap at the overlap centroid as projected onto each face
  // in the direction of the common plane normal
  this->centroidGap( m1, m2, scale );

  // reset point-location of common plane along the direction of the common plane normal
  // as average of overlap centroid projected to each face
  m_cX = 0.5 * ( m_cXf1 + m_cXf2 );
  m_cY = 0.5 * ( m_cYf1 + m_cYf2 );
  m_cZ = 0.5 * ( m_cZf1 + m_cZf2 );

}  // end CommonPlanePair::resetPlanePointAndCentroidGap()

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

  if ( m_dim == 3 ) {
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
TRIBOL_HOST_DEVICE void MortarPlanePair::computeNormal( const MeshData::Viewer& TRIBOL_UNUSED_PARAM( m1 ),
                                                        const MeshData::Viewer& m2 )
{
  // mesh id 2 is the projection/mortar plane
  IndexT fId2 = m_pair->m_element_id2;
  m_nZ = 0.0;

  // the projection plane is the nonmortar (i.e. mesh id 2) surface so
  // we use the outward normal for face 2 on mesh 2
  m_nX = m2.getElementNormals()[0][fId2];
  m_nY = m2.getElementNormals()[1][fId2];

  if ( m_dim == 3 ) {
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
TRIBOL_HOST_DEVICE void AlignedMortarPlanePair::computeNormal( const MeshData::Viewer& TRIBOL_UNUSED_PARAM( m1 ),
                                                               const MeshData::Viewer& m2 )
{
  // side 2 is the mortar/projection plane
  IndexT fId2 = m_pair->m_element_id2;
  m_nZ = 0.0;

  // the projection plane is the nonmortar (i.e. mesh id 2) surface so
  // we use the outward normal for face 2 on mesh 2
  m_nX = m2.getElementNormals()[0][fId2];
  m_nY = m2.getElementNormals()[1][fId2];

  if ( m_dim == 3 ) {
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

  if ( m_dim == 3 ) {
    m_cZ = 0.5 * ( m1.getElementCentroids()[2][fId1] + m2.getElementCentroids()[2][fId2] );
  }

  return;

}  // end CommonPlanePair::computePlanePoint()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void MortarPlanePair::computePlanePoint( const MeshData::Viewer& TRIBOL_UNUSED_PARAM( m1 ),
                                                            const MeshData::Viewer& m2 )
{
  // take side 2 as the projection plane
  IndexT fId2 = m_pair->m_element_id2;

  // MORTAR calculation using the vertex averaged
  // centroid of the nonmortar face
  m_cX = m2.getElementCentroids()[0][fId2];
  m_cY = m2.getElementCentroids()[1][fId2];
  m_cZ = 0.;

  if ( m_dim == 3 ) {
    m_cZ = m2.getElementCentroids()[2][fId2];
  }

  return;

}  // end MortarPlanePair::computePlanePoint()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void AlignedMortarPlanePair::computePlanePoint( const MeshData::Viewer& TRIBOL_UNUSED_PARAM( m1 ),
                                                                   const MeshData::Viewer& m2 )
{
  // take side 2 as the projection plane
  IndexT fId2 = m_pair->m_element_id2;

  // set plane centroid to mesh 2 face centroid (i.e. non-mortar)
  m_cX = m2.getElementCentroids()[0][fId2];
  m_cY = m2.getElementCentroids()[1][fId2];

  if ( m_dim == 3 ) {
    m_cZ = m2.getElementCentroids()[2][fId2];
  }

  return;

}  // end AlignedMortarPlanePair::computePlanePoint()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlanePair::getFace1Coords( RealT* x1, int num_coords ) const
{
  if ( m_dim == 2 ) {
    x1[0] = m_x1_prime[0];
    x1[1] = m_y1_prime[0];
    x1[2] = m_x1_prime[1];
    x1[3] = m_y1_prime[1];
  } else {
    for ( int i = 0; i < num_coords; ++i ) {
      x1[m_dim * i] = m_x1_prime[i];
      x1[m_dim * i + 1] = m_y1_prime[i];
      x1[m_dim * i + 2] = m_z1_prime[i];
    }
  }
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlanePair::getFace2Coords( RealT* x2, int num_coords ) const
{
  if ( m_dim == 2 ) {
    x2[0] = m_x2_prime[0];
    x2[1] = m_y2_prime[0];
    x2[2] = m_x2_prime[1];
    x2[3] = m_y2_prime[1];
  } else {
    for ( int i = 0; i < num_coords; ++i ) {
      x2[m_dim * i] = m_x2_prime[i];
      x2[m_dim * i + 1] = m_y2_prime[i];
      x2[m_dim * i + 2] = m_z2_prime[i];
    }
  }
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlanePair::getFace1ProjectedCoords( RealT* x1_proj, int num_coords ) const
{
  if ( m_dim == 2 ) {
    x1_proj[0] = m_x1_bar[0];
    x1_proj[1] = m_y1_bar[0];
    x1_proj[2] = m_x1_bar[1];
    x1_proj[3] = m_y1_bar[1];
  } else {
    for ( int i = 0; i < num_coords; ++i ) {
      x1_proj[m_dim * i] = m_x1_bar[i];
      x1_proj[m_dim * i + 1] = m_y1_bar[i];
      x1_proj[m_dim * i + 2] = m_z1_bar[i];
    }
  }
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlanePair::getFace2ProjectedCoords( RealT* x2_proj, int num_coords ) const
{
  if ( m_dim == 2 ) {
    x2_proj[0] = m_x2_bar[0];
    x2_proj[1] = m_y2_bar[0];
    x2_proj[2] = m_x2_bar[1];
    x2_proj[3] = m_y2_bar[1];
  } else {
    for ( int i = 0; i < num_coords; ++i ) {
      x2_proj[m_dim * i] = m_x2_bar[i];
      x2_proj[m_dim * i + 1] = m_y2_bar[i];
      x2_proj[m_dim * i + 2] = m_z2_bar[i];
    }
  }
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlanePair::getOverlapVertices( RealT* overlap_verts ) const
{
#ifdef TRIBOL_USE_HOST
  SLIC_ERROR_IF( m_dim == 2 && m_numPolyVert != 2,
                 "ContactPlanePair::getOverlapVertices(): " << "number of overlap vertices not equal to 2" );
  SLIC_ERROR_IF( m_dim == 3 && m_numPolyVert < 3,
                 "ContactPlanePair::getOverlapVertices(): " << "number of overlap vertices < 3" );
#endif

  for ( int i = 0; i < m_numPolyVert; ++i ) {
    overlap_verts[m_dim * i] = m_polyX[i];
    overlap_verts[m_dim * i + 1] = m_polyY[i];
    if ( m_dim == 3 ) {
      overlap_verts[m_dim * i + 2] = m_polyZ[i];
    }
  }
}
//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlanePair::computeLocalBasis()
{
  ComputeLocalBasis( m_nX, m_nY, m_nZ, m_e1X, m_e1Y, m_e1Z, m_e2X, m_e2Y, m_e2Z );
  return;

}  // end ContactPlanePair::computeLocalBasis()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlanePair::globalTo2DLocalCoords( const RealT* pX, const RealT* pY, const RealT* pZ,
                                                                 RealT* pLX, RealT* pLY, int size )
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
TRIBOL_HOST_DEVICE void ContactPlanePair::computeAreaTol( const MeshData::Viewer& m1, const MeshData::Viewer& m2 )
{
  if ( m_areaFrac < this->m_params.overlap_area_frac ) {
#ifdef TRIBOL_USE_HOST
    SLIC_DEBUG( "CommonPlanePair::computeAreaTol() the overlap area fraction too small or negative; "
                << "setting to overlap_area_frac parameter." );
#endif
    m_areaFrac = this->m_params.overlap_area_frac;
  }

  m_areaMin = m_areaFrac * axom::utilities::min( m1.getElementAreas()[this->getCpElementId1()],
                                                 m2.getElementAreas()[this->getCpElementId2()] );

  return;

}  // end ContactPlanePair::computeAreaTol()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ContactPlanePair::local2DToGlobalCoords( RealT xloc, RealT yloc, RealT& xg, RealT& yg,
                                                                 RealT& zg )
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
TRIBOL_HOST_DEVICE void CommonPlanePair::centroidGap( const MeshData::Viewer& m1, const MeshData::Viewer& m2,
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
  RealT zcg = 0.;
  if ( m_dim == 3 ) {
    zcg = m_cZ;
  }

  // find where the overlap centroid (plane point) intersects each face

  // set the line segment's first vertex at the contact plane centroid scaled
  // in the direction opposite the contact plane normal
  RealT xA = xcg + m_nX * scale;  // flipped the sign
  RealT yA = ycg + m_nY * scale;
  RealT zA = 0.;

  if ( m_dim == 3 ) {
    zA = zcg + m_nZ * scale;
  }

  // use the contact plane normal as the segment directional vector scale in
  // the direction of the contact plane
  RealT xB = xcg - m_nX * scale;
  RealT yB = ycg - m_nY * scale;
  RealT zB = 0.;

  if ( m_dim == 3 ) {
    zB = zcg - m_nZ * scale;
  }

  bool is_parallel = false;
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

  RealT fn1z = 0.;
  RealT cx1z = 0.;
  RealT fn2z = 0.;
  RealT cx2z = 0.;

  if ( m_dim == 3 ) {
    fn1z = fn1[2];
    cx1z = cx1[2];
    fn2z = fn2[2];
    cx2z = cx2[2];
  }

  // fine line-plane intersection with average face planes, which is consistent with using the prime coords
  // for each face
  LinePlaneIntersection( xA, yA, zA, xB, yB, zB, cx1[0], cx1[1], cx1z, fn1[0], fn1[1], fn1z, xc1, yc1, zc1,
                         is_parallel );

  LinePlaneIntersection( xA, yA, zA, xB, yB, zB, cx2[0], cx2[1], cx2z, fn2[0], fn2[1], fn2z, xc2, yc2, zc2,
                         is_parallel );

  // compute normal gap magnitude (x1 - x2 for positive gap in separation
  // and negative gap in penetration)
  m_gap = ( xc1 - xc2 ) * m_nX + ( yc1 - yc2 ) * m_nY;

  if ( m_dim == 3 ) {
    m_gap += ( zc1 - zc2 ) * m_nZ;
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
TRIBOL_HOST_DEVICE FaceGeomException CommonPlanePair::computeOverlap3D( const RealT* x1, const RealT* y1,
                                                                        const RealT* z1, const RealT* x2,
                                                                        const RealT* y2, const RealT* z2,
                                                                        const MeshData::Viewer& m1,
                                                                        const MeshData::Viewer& m2 )
{
  // outer loop over faces, inner loop over nodes/segments and determine
  // how many 1) line-plane intersections there are (there should at most be
  // two for intersection polygons or zero for fully separated or fully
  // interpenetrated faces) and then 2) number of nodes on the current face
  // that cross the plane defined by the other face.

  // arrays to hold the maximum line-plane intersections for both faces.
  // Note, convex planar quadrilaterals can only intesect the common
  // plane at most in two places for each face.
  constexpr int max_nodes_per_elem = 4;
  constexpr int max_dim = 3;
  RealT xInter[max_nodes_per_elem];
  RealT yInter[max_nodes_per_elem];
  RealT zInter[max_nodes_per_elem];

  for ( int i = 0; i < max_nodes_per_elem; ++i ) {
    xInter[i] = 0.;
    yInter[i] = 0.;
    zInter[i] = 0.;
  }

  bool is_parallel = false;
  int numV[2] = { 0, 0 };

  // set up vertex id arrays to indicate which face vertices pass through
  // contact plane (i.e. lie on the other side)
  StackArrayT<const MeshData::Viewer*, 2> mesh( { &m1, &m2 } );
  int interpenVertex1[max_nodes_per_elem];
  int interpenVertex2[max_nodes_per_elem];

  for ( int i = 0; i < max_nodes_per_elem; ++i ) {
    interpenVertex1[i] = -1;
    interpenVertex2[i] = -1;
  }

  StackArrayT<IndexT, 2> element_id( { getCpElementId1(), getCpElementId2() } );
  StackArrayT<IndexT, 2> num_intersections( { 0, 0 } );
  StackArrayT<IndexT, 2> num_intersections_inside( { 0, 0 } );
  StackArrayT<IndexT, 2> num_nodes_otherside( { 0, 0 } );

  // compute interpen data for convex face-pairs only
  if ( m_face1_convex && m_face2_convex ) {
    for ( int i = 0; i < 2; ++i )  // loop over two constituent faces
    {
      // declare array to hold vertex id for all vertices that interpenetrate
      // the contact plane. At most, all nodes pass through common plane for
      // the current face
      int interpenVertex[max_nodes_per_elem];

      // point to the correct current face coordinates
      const RealT *x, *y, *z;
      const RealT *x_other, *y_other, *z_other;
      if ( i == 0 ) {
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
      if ( i == 0 ) {
        mesh[1]->getFaceNormal( element_id[1], fn );
        mesh[1]->getFaceCentroid( element_id[1], cx );
        num_nodes_other = mesh[1]->numberOfNodesPerElement();
      } else {
        mesh[0]->getFaceNormal( element_id[0], fn );
        mesh[0]->getFaceCentroid( element_id[0], cx );
        num_nodes_other = mesh[0]->numberOfNodesPerElement();
      }

      int k = 0;
      int k_inside = 0;
      int k_otherside = 0;
      for ( int j = 0; j < mesh[i]->numberOfNodesPerElement(); ++j )  // loop over face segments
      {
        bool intersection_is_node = false;

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

        // check for the case k > 2. This is a 'breaking' assumption in the algorithm. Two planar quadrilaterals
        // can intersect the plane defined by the other fast AT MOST in two locations. This check points out
        // unanticipated degenerate cases or bugs. Here, we error out for further investigation
        if ( k > 2 ) {
#ifdef TRIBOL_USE_HOST
          // Debug print faces to screen to catch unforeseen degenerate face configurations etc.
          SLIC_WARNING( "Degenerate face configuration detected with number of line-plane intersections > 2." );
          SLIC_INFO( "Planar coordinates for face 1 in CommonPlanePair::computeOverlap3D(): " );
          for ( int a = 0; a < mesh[0]->numberOfNodesPerElement(); ++a ) {
            std::cout << x1[a] << ", " << y1[a] << ", " << z1[a] << std::endl;
          }

          SLIC_INFO( "Planar coordinates for face 2 in CommonPlanePair::computeOverlap3D(): " );
          for ( int b = 0; b < mesh[1]->numberOfNodesPerElement(); ++b ) {
            std::cout << x2[b] << ", " << y2[b] << ", " << z2[b] << std::endl;
          }

          SLIC_ERROR( "CommonPlanePair::computeOverlap3D(): too many segment-face intersections; "
                      << "check for degenerate face " << m_pair->m_element_id1 << "on mesh " << mesh[i]->meshId()
                      << "." );
#endif
          return DEGENERATE_OVERLAP;
        }

        // call segment-to-plane intersection routine
        if ( k < 2 )  // we haven't found both intersection points yet
        {
          // compute the current face's current segment-to-plane intersection using the other face's point-normal data
          bool inter = LinePlaneIntersection( xa, ya, za, xb, yb, zb, cx[0], cx[1], cx[2], fn[0], fn[1], fn[2],
                                              xInter[2 * i + k], yInter[2 * i + k], zInter[2 * i + k], is_parallel );

          // check for duplicate intersection points. This can arise when the intersection points occur at one face's
          // vertices. Then, there are two edge segments that share each vertex, which would register a total of 4
          // line-plane intersections, with two duplications.
          // TODO verify the use/value of the tolerance for duplicate points. Tolerancing up to 1.e-12 may be ok
          // NOTE: this would be a very specific case such that a departure from the tolerance likely won't trigger
          // some other edge case. Two intersection points less than the tolerance from one another would only arise
          // if two edges of a four node quad form a very acute angle that also interpenetrates the opposing face.
          if ( inter ) {
            for ( int a = ( 2 * i + k ); a > 2 * i; --a ) {
              if ( magnitude( xInter[a] - xInter[a - 1], yInter[a] - yInter[a - 1], zInter[a] - zInter[a - 1] ) <
                   1.e-10 ) {
                inter = false;  // we already have the point
              }
            }

            if ( magnitude( xInter[2 * i + k] - xa, yInter[2 * i + k] - ya, zInter[2 * i + k] - za ) < 1.e-10 ) {
              intersection_is_node = true;
            }
          }

          if ( inter ) {
            // check to see if the line-plane intersection point lies inside the other planar face
            RealT x_other_local[max_nodes_per_elem];
            RealT y_other_local[max_nodes_per_elem];
            Points3DTo2D( &x_other[0], &y_other[0], &z_other[0], fn[0], fn[1], fn[2], cx[0], cx[1], cx[2],
                          num_nodes_other, &x_other_local[0], &y_other_local[0] );

            // get the local coordinates of the current intersection point
            RealT xInter_local, yInter_local;
            Points3DTo2D( &xInter[2 * i + k], &yInter[2 * i + k], &zInter[2 * i + k], fn[0], fn[1], fn[2], cx[0], cx[1],
                          cx[2], 1.0, &xInter_local, &yInter_local );

            // get the local coordinates of the other face's centroid
            RealT cx_other_local, cy_other_local;
            RealT cz = 0.;  // dummy arg.
            VertexAvgCentroid( &x_other_local[0], &y_other_local[0], nullptr, num_nodes_other, cx_other_local,
                               cy_other_local, cz );

            // check if local intersection point lies inside other face
            bool check = Point2DInFace( xInter_local, yInter_local, &x_other_local[0], &y_other_local[0],
                                        cx_other_local, cy_other_local, num_nodes_other );

            // if intersection point lies in other face then increment intersection counter
            if ( check ) {
              ++k_inside;
            }

            // we still want to increment the intersection counter expecting up to 2 line-plane intersections
            // even if the point is not inside
            ++k;

          }  // end if (inter)
        }  // end if (k<2)

        // Secondly: check the current face's current node to see if it lies on the other side of the other face.
        // do this even if we don't ultimately have an interpen overlap calc.
        if ( intersection_is_node == false ) {
          RealT vX = xa - cx[0];
          RealT vY = ya - cx[1];
          RealT vZ = za - cx[2];

          // project the vector onto the opposing face's normal
          RealT proj = vX * fn[0] + vY * fn[1] + vZ * fn[2];

          // check for negative projections meaning a node on one face crosses
          // the plane defined by the other face
          interpenVertex[ja] = ( i == 0 && proj < 0. ) ? ja : -1;
          interpenVertex[ja] = ( i == 1 && proj < 0. ) ? ja : interpenVertex[ja];

          if ( interpenVertex[ja] != -1 ) {
            ++k_otherside;
          }
        }

      }  // end loop over nodes

      // count the number of vertices (intersection points and interpen points) for the clipped
      // portion of the i^th face that interpenetrates the opposing face.
      numV[i] = k;  // could be zero intersection points
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
  }  // end if-convex

  // we come into this routine with full overlap calculation set to true. Here, we need
  // to determine if we need to switch to interpen overlap calc. This is cleaner logic
  // than assuming interpen and switching to full because it only checks interior intersection
  // points. The criterion for intersection and thus the interpen overlap calc for two
  // planar quadrilaterals is:
  //
  // 1) each face has only one line-plane intersection point with the opposing face and that point
  //    lies INSIDE that opposing face, OR
  // 2) one face has two intersection points that lie INSIDE the other face and the
  //    other face has zero intersection points that lie INSIDE its opposing face
  // 3) each face has two intersection points that lie INSIDE the other face; this occurs when
  //    one face's two line-plane intersections occur at edge segments of the other face (and vice versa).
  // 4) one face has two line-plane intersections that are INSIDE the other face, and the other face
  //    has only ONE line-plane intersection that lies INSIDE the opposing face. This occurs when the first
  //    face has one intersection point fully interior to the opposing face, and the other intersects the
  //    face at a vertex or edge. Then, the second face will have a line-plane intersection point through
  //    one of the first face's edges or vertices.
  //
  // Note: still double check degenerate face-interaction vertex counts and in the case
  //       that one of the criterion above switched to the interpen overlap calc, return
  //       the calc to full overlap for robustness
  if ( num_intersections_inside[0] == 1 && num_intersections_inside[1] == 1 ) {
    m_fullOverlap = false;
  } else if ( num_intersections_inside[0] == 2 && num_intersections_inside[1] == 0 ) {
    m_fullOverlap = false;
  } else if ( num_intersections_inside[0] == 0 && num_intersections_inside[1] == 2 ) {
    m_fullOverlap = false;
  } else if ( num_intersections_inside[0] == 2 && num_intersections_inside[1] == 2 ) {
    m_fullOverlap = false;
  } else if ( num_intersections_inside[0] == 2 && num_intersections_inside[1] == 1 ) {
    m_fullOverlap = false;
  } else if ( num_intersections_inside[0] == 1 && num_intersections_inside[1] == 2 ) {
    m_fullOverlap = false;
  }

#ifdef TRIBOL_USE_HOST
  SLIC_ERROR_IF( ( !m_face1_convex || !m_face2_convex ) && !m_fullOverlap,
                 "Must switch to full overlap for non-convex faces!" );
#endif

  // allocate arrays to store the vertices for clipped or full face used either
  // in the interpen or full overlap calc
  constexpr int max_nodes_per_clipped_face = 5;  // max five nodes for clipped face of 4 node planar quad
  RealT cfx1[max_nodes_per_clipped_face];        // cfx = clipped face x-coordinate
  RealT cfy1[max_nodes_per_clipped_face];
  RealT cfz1[max_nodes_per_clipped_face];

  RealT cfx2[max_nodes_per_clipped_face];  // cfx = clipped face x-coordinate
  RealT cfy2[max_nodes_per_clipped_face];
  RealT cfz2[max_nodes_per_clipped_face];

  FaceGeomException overlap_error = NO_FACE_GEOM_EXCEPTION;
  if ( !m_fullOverlap ) {
    // populate segment-contact-plane intersection vertices
    for ( int m = 0; m < num_intersections[0]; ++m ) {
      cfx1[m] = xInter[m];
      cfy1[m] = yInter[m];
      cfz1[m] = zInter[m];
    }
    for ( int n = 0; n < num_intersections[1]; ++n ) {
      cfx2[n] = xInter[num_intersections[0] + n];
      cfy2[n] = yInter[num_intersections[0] + n];
      cfz2[n] = zInter[num_intersections[0] + n];
    }

    // populate the face 1 vertices that cross the contact plane
    int k = num_intersections[0];
    for ( int m = 0; m < mesh[0]->numberOfNodesPerElement(); ++m ) {
      if ( interpenVertex1[m] != -1 ) {
        // set nodal coordinates to the "prime" coords sent into this routine
        // associated with the average face plane
        cfx1[k] = x1[interpenVertex1[m]];
        cfy1[k] = y1[interpenVertex1[m]];
        cfz1[k] = z1[interpenVertex1[m]];
        ++k;
      }
    }

    // populate the face 2 vertices that cross the contact plane
    k = num_intersections[1];
    for ( int n = 0; n < mesh[1]->numberOfNodesPerElement(); ++n ) {
      if ( interpenVertex2[n] != -1 ) {
        // set nodal coordinates to the "prime" coords sent into this routine
        // associated with the average face plane
        cfx2[k] = x2[interpenVertex2[n]];
        cfy2[k] = y2[interpenVertex2[n]];
        cfz2[k] = z2[interpenVertex2[n]];
        ++k;
      }
    }

    // project face coordinates onto common plane and compute overlap
    FaceGeomException interpen_error = this->projectPointsAndComputeOverlap(
        &cfx1[0], &cfy1[0], &cfz1[0], &cfx2[0], &cfy2[0], &cfz2[0], numV[0], numV[1], m1, m2 );
    overlap_error = interpen_error;

    // geomFilter() checks to see if at least one vertex of one face lies in the other as a proxy
    // for a positive area of overlap. If the interpen overlap calculation returns no overlap then
    // it could be that the interpenetrating portion of one face lies nearly outside the other face
    // resulting in an overlap area less than the min. In this case, we could miss the full overlap
    // portion that is in separation. Ideally, we account for this separated portion for the timestep
    // vote by computing a full overlap area.
    if ( overlap_error == NO_OVERLAP ) {
      m_fullOverlap = true;
    } else if ( overlap_error != NO_FACE_GEOM_EXCEPTION ) {
      return overlap_error;
    }

  }  // end if (!m_fullOverlap)

  if ( m_fullOverlap ) {  // populate the face vertex array with the face coordinates themselves
    numV[0] = mesh[0]->numberOfNodesPerElement();
    numV[1] = mesh[1]->numberOfNodesPerElement();
    // face 1
    for ( int m = 0; m < mesh[0]->numberOfNodesPerElement(); ++m ) {
      // use face averaged "prime" coordinates for consistency
      cfx1[m] = x1[m];
      cfy1[m] = y1[m];
      cfz1[m] = z1[m];
    }

    // face 2
    for ( int n = 0; n < mesh[1]->numberOfNodesPerElement(); ++n ) {
      // use face averaged "prime" coordinates for consistency
      cfx2[n] = x2[n];
      cfy2[n] = y2[n];
      cfz2[n] = z2[n];
    }

    FaceGeomException full_error = this->projectPointsAndComputeOverlap( &cfx1[0], &cfy1[0], &cfz1[0], &cfx2[0],
                                                                         &cfy2[0], &cfz2[0], numV[0], numV[1], m1, m2 );

    overlap_error = full_error;

    if ( overlap_error != NO_FACE_GEOM_EXCEPTION ) {
      return overlap_error;
    }

  }  // end else (m_fullOverlap)

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

  // Now that all local-to-global projections have occurred,
  // relocate the contact plane using the area centroid calculation,
  // then compute the gap, and then locate the area centroid equidistant
  // between each face.
  //
  // Warning:
  // Make sure that any local to global transformations have
  // occurred prior to this call.
  this->resetPlanePointAndCentroidGap( m1, m2 );

  // REPROJECT the overlapping polygon onto the new contact plane
  for ( int i = 0; i < m_numPolyVert; ++i ) {
    ProjectPointToPlane( m_polyX[i], m_polyY[i], m_polyZ[i], m_nX, m_nY, m_nZ, m_cX, m_cY, m_cZ, m_polyX[i], m_polyY[i],
                         m_polyZ[i] );
  }

  // REPROJECT the global coordinates of the interpenetrating polygon as projection onto the common plane
  for ( int i = 0; i < m_numInterpenPoly1Vert; ++i ) {
    ProjectPointToPlane( m_interpenG1X[i], m_interpenG1Y[i], m_interpenG1Z[i], m_nX, m_nY, m_nZ, m_cX, m_cY, m_cZ,
                         m_interpenG1X[i], m_interpenG1Y[i], m_interpenG1Z[i] );
  }

  for ( int i = 0; i < m_numInterpenPoly2Vert; ++i ) {
    ProjectPointToPlane( m_interpenG2X[i], m_interpenG2Y[i], m_interpenG2Z[i], m_nX, m_nY, m_nZ, m_cX, m_cY, m_cZ,
                         m_interpenG2X[i], m_interpenG2Y[i], m_interpenG2Z[i] );
  }

  // Project averaged face coordinates sent into this routine to the common plane and store
  for ( int i = 0; i < mesh[0]->numberOfNodesPerElement(); ++i ) {
    ProjectPointToPlane( x1[i], y1[i], z1[i], m_nX, m_nY, m_nZ, m_cX, m_cY, m_cZ, m_x1_bar[i], m_y1_bar[i],
                         m_z1_bar[i] );
  }

  for ( int i = 0; i < mesh[1]->numberOfNodesPerElement(); ++i ) {
    ProjectPointToPlane( x2[i], y2[i], z2[i], m_nX, m_nY, m_nZ, m_cX, m_cY, m_cZ, m_x2_bar[i], m_y2_bar[i],
                         m_z2_bar[i] );
  }

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
    if ( this->exceedsMaxAutoInterpen( m1, m2, element_id[0], element_id[1], this->m_params, m_gap ) ) {
      return EXCEEDS_AUTO_CONTACT_LENGTH_SCALE;
    }
  }

  return NO_FACE_GEOM_EXCEPTION;

}  // end CommonPlanePair::computeOverlap3D()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomException CommonPlanePair::projectPointsAndComputeOverlap(
    RealT const* const fx1, RealT const* const fy1, RealT const* const fz1, RealT const* const fx2,
    RealT const* const fy2, RealT const* const fz2, const int num_vert_1, const int num_vert_2,
    const MeshData::Viewer& m1, const MeshData::Viewer& m2 )
{
  // sanity check
#ifdef TRIBOL_USE_HOST
  if ( m_fullOverlap ) {
    if ( num_vert_1 != m1.numberOfNodesPerElement() || num_vert_2 != m2.numberOfNodesPerElement() ) {
      SLIC_ERROR( "CommonPlanePair::projectPointsAndComputeOverlap(): full overlap requires "
                  << "input number of vertices to match number of nodes per element." );
    }
  }
#endif

  IndexT element_id1 = this->getCpElementId1();
  IndexT element_id2 = this->getCpElementId2();

  constexpr int max_nodes_per_clipped_face = 5;
  RealT cfx1_proj[max_nodes_per_clipped_face];
  RealT cfy1_proj[max_nodes_per_clipped_face];
  RealT cfz1_proj[max_nodes_per_clipped_face];

  RealT cfx2_proj[max_nodes_per_clipped_face];
  RealT cfy2_proj[max_nodes_per_clipped_face];
  RealT cfz2_proj[max_nodes_per_clipped_face];

  // project overlap-calc face coordinates to contact plane as currently located
  for ( int i = 0; i < num_vert_1; ++i ) {
    ProjectPointToPlane( fx1[i], fy1[i], fz1[i], m_nX, m_nY, m_nZ, m_cX, m_cY, m_cZ, cfx1_proj[i], cfy1_proj[i],
                         cfz1_proj[i] );
  }

  for ( int i = 0; i < num_vert_2; ++i ) {
    ProjectPointToPlane( fx2[i], fy2[i], fz2[i], m_nX, m_nY, m_nZ, m_cX, m_cY, m_cZ, cfx2_proj[i], cfy2_proj[i],
                         cfz2_proj[i] );
  }

  // declare local coordinate pointers
  RealT cfx1_loc[max_nodes_per_clipped_face];
  RealT cfy1_loc[max_nodes_per_clipped_face];

  RealT cfx2_loc[max_nodes_per_clipped_face];
  RealT cfy2_loc[max_nodes_per_clipped_face];

  // convert global coords to local contact plane coordinates
  GlobalTo2DLocalCoords( &cfx1_proj[0], &cfy1_proj[0], &cfz1_proj[0], m_e1X, m_e1Y, m_e1Z, m_e2X, m_e2Y, m_e2Z, m_cX,
                         m_cY, m_cZ, &cfx1_loc[0], &cfy1_loc[0], num_vert_1 );

  GlobalTo2DLocalCoords( &cfx2_proj[0], &cfy2_proj[0], &cfz2_proj[0], m_e1X, m_e1Y, m_e1Z, m_e2X, m_e2Y, m_e2Z, m_cX,
                         m_cY, m_cZ, &cfx2_loc[0], &cfy2_loc[0], num_vert_2 );

  // reorder potentially unordered set of vertices for interpen calcs
  // Note: this routine will order both sets of vertices in counter clockwise orientation.
  //       Intersection2DPolygon() assumes consistent ordering between faces
  if ( !m_fullOverlap ) {
    PolyReorderConvex( cfx1_loc, cfy1_loc, nullptr, num_vert_1 );
    PolyReorderConvex( cfx2_loc, cfy2_loc, nullptr, num_vert_2 );
  } else {  // use ElemReverse() per original implementation for full overlaps
    ElemReverse( cfx2_loc, cfy2_loc, num_vert_2 );
  }

  // call intersection routine to get intersecting polygon
  RealT pos_tol = this->m_params.len_collapse_ratio *
                  axom::utilities::max( m1.getFaceRadius()[element_id1], m2.getFaceRadius()[element_id2] );
  RealT len_tol = pos_tol;
  FaceGeomException inter_err =
      Intersection2DPolygon( cfx1_loc, cfy1_loc, num_vert_1, cfx2_loc, cfy2_loc, num_vert_2, pos_tol, len_tol,
                             m_polyLocX, m_polyLocY, m_numPolyVert, m_area, false );

  if ( inter_err != NO_FACE_GEOM_EXCEPTION ) {
    return inter_err;
  }

  // check overlap area to area tol
  if ( m_area < m_areaMin ) {
    return NO_OVERLAP;
  }

  // transform local interpenetration overlaps to global coords for the
  // current polygonal overlap
  for ( int i = 0; i < num_vert_1; ++i ) {
    Local2DToGlobalCoords( cfx1_loc[i], cfy1_loc[i], m_e1X, m_e1Y, m_e1Z, m_e2X, m_e2Y, m_e2Z, m_cX, m_cY, m_cZ,
                           m_interpenG1X[i], m_interpenG1Y[i], m_interpenG1Z[i] );
  }

  for ( int i = 0; i < num_vert_2; ++i ) {
    Local2DToGlobalCoords( cfx2_loc[i], cfy2_loc[i], m_e1X, m_e1Y, m_e1Z, m_e2X, m_e2Y, m_e2Z, m_cX, m_cY, m_cZ,
                           m_interpenG2X[i], m_interpenG2Y[i], m_interpenG2Z[i] );
  }

  return NO_FACE_GEOM_EXCEPTION;

}  // end CommonPlanePair::projectPointsAndComputeOverlap()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomException CommonPlanePair::computeOverlap2D( const MeshData::Viewer& m1,
                                                                        const MeshData::Viewer& m2 )
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

  RealT xposA1 = this->m_x1_prime[0];
  RealT yposA1 = this->m_y1_prime[0];
  RealT xposB1 = this->m_x1_prime[1];
  RealT yposB1 = this->m_y1_prime[1];

  RealT xposA2 = this->m_x2_prime[0];
  RealT yposA2 = this->m_y2_prime[0];
  RealT xposB2 = this->m_x2_prime[1];
  RealT yposB2 = this->m_y2_prime[1];

  RealT xInter, yInter;
  bool duplicatePoint = false;

  // check if the segments intersect thus triggering an interpen overlap calc
  RealT len_tol = this->m_params.len_collapse_ratio *
                  axom::utilities::max( m1.getFaceRadius()[edgeId1], m2.getFaceRadius()[edgeId2] );

  bool edgeIntersect = SegmentIntersection2D( xposA1, yposA1, xposB1, yposB1, xposA2, yposA2, xposB2, yposB2, nullptr,
                                              xInter, yInter, duplicatePoint, len_tol );

  // if there is no edge-edge intersection point or the intersection
  // point exists at an edge vertex, then switch to full overlap
  if ( !edgeIntersect || duplicatePoint ) {
    m_fullOverlap = true;
  } else {
    m_fullOverlap = false;
  }

  RealT x1_to_project[max_dim], y1_to_project[max_dim];
  RealT x2_to_project[max_dim], y2_to_project[max_dim];

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
      if ( proj1 < 0.0 ) {
        interId1 = i;
        ++k1;
      }
      if ( proj2 < 0.0 ) {
        interId2 = i;
        ++k2;
      }
    }  // end loop over nodes

    // Debug check the number of interpenetrating vertices. We should never get here, but
    // let's check for a design breaking case
    if ( k1 > 1 || k2 > 1 ) {
#ifdef TRIBOL_USE_HOST
      SLIC_INFO(
          "CommonPlanePair::computeOverlap2D() with more than one intersection point. Offending edge1 vertices are: "
          << xposA1 << ", " << yposA1 << " and " << xposB1 << ", " << yposB1 << "." );
      SLIC_INFO(
          "CommonPlanePair::computeOverlap2D() with more than one intersection point. Offending edge2 vertices are: "
          << xposA2 << ", " << yposA2 << " and " << xposB2 << ", " << yposB2 << "." );

      SLIC_ERROR( "CommonPlanePair::computeOverlap2D() more than 2 interpenetrating vertices detected; "
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
    ProjectPointToSegment( vx1, vy1, m_nX, m_nY, m_cX, m_cY, m_polyX[0], m_polyY[0] );
    ProjectPointToSegment( vx2, vy2, m_nX, m_nY, m_cX, m_cY, m_polyX[1], m_polyY[1] );

    // compute the overlap area
    m_area = magnitude( m_polyX[1] - m_polyX[0], m_polyY[1] - m_polyY[0] );

    if ( m_area < m_areaMin ) {
      return NO_OVERLAP;
    } else {
      this->m_numPolyVert = 2;
    }

    this->resetPlanePointAndCentroidGap( m1, m2 );

    // reproject the overlap points to the new common plane
    ProjectPointToSegment( m_polyX[0], m_polyY[0], m_nX, m_nY, m_cX, m_cY, m_polyX[0], m_polyY[0] );
    ProjectPointToSegment( m_polyX[1], m_polyY[1], m_nX, m_nY, m_cX, m_cY, m_polyX[1], m_polyY[1] );

    // reproject the global intepen face vertices onto the new contact plane
    ProjectPointToSegment( m_interpenG1X[0], m_interpenG1Y[0], m_nX, m_nY, m_cX, m_cY, m_interpenG1X[0],
                           m_interpenG1Y[0] );
    ProjectPointToSegment( m_interpenG1X[1], m_interpenG1Y[1], m_nX, m_nY, m_cX, m_cY, m_interpenG1X[1],
                           m_interpenG1Y[1] );
    ProjectPointToSegment( m_interpenG2X[0], m_interpenG2Y[0], m_nX, m_nY, m_cX, m_cY, m_interpenG2X[0],
                           m_interpenG2Y[0] );
    ProjectPointToSegment( m_interpenG2X[1], m_interpenG2Y[1], m_nX, m_nY, m_cX, m_cY, m_interpenG2X[1],
                           m_interpenG2Y[1] );

  } else if ( m_fullOverlap ) {  // end if (!m_fullOverlap)

    RealT projX1[max_dim], projY1[max_dim];
    RealT projX2[max_dim], projY2[max_dim];

    ProjectEdgeNodesToSegment( m1, edgeId1, this->m_nX, this->m_nY, this->m_cX, this->m_cY, &projX1[0], &projY1[0] );
    ProjectEdgeNodesToSegment( m2, edgeId2, this->m_nX, this->m_nY, this->m_cX, this->m_cY, &projX2[0], &projY2[0] );
    FaceGeomException segError =
        CheckSegOverlap( &projX1[0], &projY1[0], &projX2[0], &projY2[0], m1.numberOfNodesPerElement(),
                         m2.numberOfNodesPerElement(), &m_polyX[0], &m_polyY[0], m_area );

    if ( segError != NO_FACE_GEOM_EXCEPTION ) {
      return segError;
    }

    if ( m_area < m_areaMin ) {
      return NO_OVERLAP;
    } else {
      this->m_numPolyVert = 2;
    }

    this->resetPlanePointAndCentroidGap( m1, m2 );

    // reproject the overlap points to the new common plane
    ProjectPointToSegment( m_polyX[0], m_polyY[0], m_nX, m_nY, m_cX, m_cY, m_polyX[0], m_polyY[0] );
    ProjectPointToSegment( m_polyX[1], m_polyY[1], m_nX, m_nY, m_cX, m_cY, m_polyX[1], m_polyY[1] );

    // for auto-contact, remove contact candidacy for full-overlap
    // face-pairs with interpenetration exceeding contact penetration fraction.
    // Note, this check is solely meant to exclude face-pairs composed of faces
    // on opposite sides of thin structures/plates
    //
    // Recall that interpen gaps are negative
    if ( this->exceedsMaxAutoInterpen( m1, m2, edgeId1, edgeId2, this->m_params, this->m_gap ) ) {
      return EXCEEDS_AUTO_CONTACT_LENGTH_SCALE;
    }

    // initialize un-used interpen coordinates to work with visualization
    this->m_numInterpenPoly1Vert = 2;
    this->m_numInterpenPoly2Vert = 2;

    for ( int i = 0; i < 2; ++i ) {
      // set the interpen vertices to the full overlap vertices
      this->m_interpenG1X[i] = 0.0;
      this->m_interpenG1Y[i] = 0.0;
      this->m_interpenG2X[i] = 0.0;
      this->m_interpenG2Y[i] = 0.0;
    }

  }  // end if (m_fullOverlap)

  // Project edge coordinates to the common plane and store
  for ( int i = 0; i < m1.numberOfNodesPerElement(); ++i ) {
    ProjectPointToSegment( m_x1_prime[i], m_y1_prime[i], m_nX, m_nY, m_cX, m_cY, m_x1_bar[i], m_y1_bar[i] );
  }

  for ( int i = 0; i < m2.numberOfNodesPerElement(); ++i ) {
    ProjectPointToSegment( m_x2_prime[i], m_y2_prime[i], m_nX, m_nY, m_cX, m_cY, m_x2_bar[i], m_y2_bar[i] );
  }

  return NO_FACE_GEOM_EXCEPTION;

}  // end CommonPlanePair::computeOverlap2D()

//------------------------------------------------------------------------------
template <>
TRIBOL_HOST_DEVICE CommonPlanePair& CompGeom::Viewer::getPlane<CommonPlanePair>( int id )
{
  return m_common_plane_pairs[id];
};

template <>
TRIBOL_HOST_DEVICE MortarPlanePair& CompGeom::Viewer::getPlane<MortarPlanePair>( int id )
{
  return m_mortar_plane_pairs[id];
};

template <>
TRIBOL_HOST_DEVICE AlignedMortarPlanePair& CompGeom::Viewer::getPlane<AlignedMortarPlanePair>( int id )
{
  return m_aligned_mortar_plane_pairs[id];
};

}  // namespace tribol
