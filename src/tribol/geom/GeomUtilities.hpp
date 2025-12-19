// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_GEOM_GEOMUTILITIES_HPP_
#define SRC_TRIBOL_GEOM_GEOMUTILITIES_HPP_

#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/utils/Math.hpp"

namespace tribol {

/*!
 *
 * \brief computes a local basis on the plane defined by the given unit normal
 *
 * \param [in] nx x-component of plane normal
 * \param [in] ny y-component of plane normal
 * \param [in] nz z-component of plane normal
 * \param [out] e1x x-component of first basis vector
 * \param [out] e1y y-component of first basis vector
 * \param [out] e1z z-component of first basis vector
 * \param [out] e2x x-component of second basis vector
 * \param [out] e2y y-component of second basis vector
 * \param [out] e2z z-component of second basis vector
 *
 */
TRIBOL_HOST_DEVICE void ComputeLocalBasis( RealT nx, RealT ny, RealT nz, RealT& e1x, RealT& e1y, RealT& e1z, RealT& e2x,
                                           RealT& e2y, RealT& e2z );

/*!
 *
 * \brief projects all the nodes (vertices) of a given FE face to a
 *  specified plane
 *
 * \param [in] mesh mesh data viewer
 * \param [in] faceId id for given face
 * \param [in] nrmlX x component of plane's unit normal
 * \param [in] nrmlY y component of plane's unit normal
 * \param [in] nrmlZ z component of plane's unit normal
 * \param [in] cX x coordinate of reference point on the plane
 * \param [in] cY y coordinate of reference point on the plane
 * \param [in] cZ z coordinate of reference point on the plane
 * \param [out] pX array of x coordinates of projected nodes
 * \param [out] pY array of y coordinates of projected nodes
 * \param [out] pZ array of z coordinates of projected nodes
 *
 * \pre length(pX), length(pY), length(pZ) >= number of nodes on face
 */
TRIBOL_HOST_DEVICE void ProjectFaceNodesToPlane( const MeshData::Viewer& mesh, int faceId, RealT nrmlX, RealT nrmlY,
                                                 RealT nrmlZ, RealT cX, RealT cY, RealT cZ, RealT* pX, RealT* pY,
                                                 RealT* pZ );

/*!
 *
 * \brief projects nodes belonging to a surface edge to a contact segment
 *
 * \param [in] mesh mesh data viewer
 * \param [in] edgeId edge id
 * \param [in] nrmlX x-component of the contact segment normal
 * \param [in] nrmlY y-component of the contact segment normal
 * \param [in] cX x-coordinate of a point on the contact segment
 * \param [in] cY y-coordinate of a point on the contact segment
 * \param [out] pX pointer to array of projected nodal x-coordinates
 * \param [out] pY pointer to array of projected nodal y-coordinates
 *
 */
TRIBOL_HOST_DEVICE void ProjectEdgeNodesToSegment( const MeshData::Viewer& mesh, int edgeId, RealT nrmlX, RealT nrmlY,
                                                   RealT cX, RealT cY, RealT* pX, RealT* pY );

/*!
 *
 * \brief Projects a point in 3-space to a plane.
 *
 * General method to project a point to a plane based on point normal data for that
 * plane and the input point in three dimensions.
 *
 * \param [in] x coordinate of point to be projected
 * \param [in] y coordinate of point to be projected
 * \param [in] z coordinate of point to be projected
 * \param [in] nx x component of unit normal defining plane
 * \param [in] ny y component of unit normal defining plane
 * \param [in] nz z component of unit normal defining plane
 * \param [in] ox x coordinate of reference point on plane
 * \param [in] oy y coordinate of reference point on plane
 * \param [in] oz z coordinate of reference point on plane
 *
 * \param [out] px x coordinate of projected point
 * \param [out] py y coordinate of projected point
 * \param [out] pz z coordinate of projected point
 *
 */
TRIBOL_HOST_DEVICE void ProjectPointToPlane( const RealT x, const RealT y, const RealT z, const RealT nx,
                                             const RealT ny, const RealT nz, const RealT ox, const RealT oy,
                                             const RealT oz, RealT& px, RealT& py, RealT& pz );

/*!
 *
 * \brief Projects an array of points in 3-space to a plane.
 *
 * General method to project a collection of points to a plane based on point normal data for that
 * plane and the input points in three dimensions.
 *
 * \param [in] x coordinates of points to be projected
 * \param [in] y coordinates of points to be projected
 * \param [in] z coordinates of points to be projected
 * \param [in] nx x component of unit normal defining plane
 * \param [in] ny y component of unit normal defining plane
 * \param [in] nz z component of unit normal defining plane
 * \param [in] ox x coordinate of reference point on plane
 * \param [in] oy y coordinate of reference point on plane
 * \param [in] oz z coordinate of reference point on plane
 *
 * \param [out] px x coordinates of projected point
 * \param [out] py y coordinates of projected point
 * \param [out] pz z coordinates of projected point
 *
 * \param [in] num_points number of points to be projected
 *
 */
TRIBOL_HOST_DEVICE void ProjectPointsToPlane( const RealT* x, const RealT* y, const RealT* z, const RealT nx,
                                              const RealT ny, const RealT nz, const RealT ox, const RealT oy,
                                              const RealT oz, RealT* px, RealT* py, RealT* pz, const int num_points );

/*!
 *
 * \brief Projects an array of points in 3-space to a plane.
 *
 * General method to project a point to a plane based on an origin point and basis vectors for that plane and the array
 * of 3D input points
 *
 * \param [in] x array of x coordinates of points to be projected, length = 3*num_coords,
 *               [x0, ..., xn, y0, ..., yn, z0, ..., zn]
 * \param [in] x0 Origin point on plane, length = 3, [x0, y0, z0]
 * \param [in] e1 First basis vector of the plane, length = 3, [e1x, e1y, e1z]
 * \param [in] e2 Second basis vector of the plane, length = 3, [e2x, e2y, e2z]
 * \param [out] xp array of x coordinates of projected points, length = num_coords, [xp0, ..., xpn]
 * \param [out] yp array of y coordinates of projected points, length = num_coords, [yp0, ..., ypn]
 * \param [in] num_coords number of coordinates to be projected
 *
 */
inline void PlaneTo2DCoords( const RealT* x, const RealT* x0, const RealT* e1, const RealT* e2, RealT* xp, RealT* yp,
                             int num_coords )
{
  for ( int i{ 0 }; i < num_coords; ++i ) {
    xp[i] = 0.0;
    yp[i] = 0.0;

    for ( int d{ 0 }; d < 3; ++d ) {
      RealT v_d = x[d * num_coords + i] - x0[d];
      xp[i] += v_d * e1[d];
      yp[i] += v_d * e2[d];
    }
  }
}

/*!
 *
 * \brief Converts an array of points in a local 2D coordinate system to a point in the global 3D coordinate system
 *
 * \param [in] xp array of x coordinates of points in local coordinate system, length = num_coords, [xp0, ..., xpn]
 * \param [in] yp array of y coordinates of points in local coordinate system, length = num_coords, [yp0, ..., ypn]
 * \param [in] x0 Origin point on plane, length = 3, [x0, y0, z0]
 * \param [in] e1 First basis vector of the plane, length = 3, [e1x, e1y, e1z]
 * \param [in] e2 Second basis vector of the plane, length = 3, [e2x, e2y, e2z]
 * \param [in,out] x array of x coordinates of projected points, length = 3*num_coords,
 *                   [x0, ..., xn, y0, ..., yn, z0, ..., zn]
 * \param [in] num_coords number of coordinates to be projected
 *
 */
inline void Coords2DToPlane( const RealT* xp, const RealT* yp, const RealT* x0, const RealT* e1, const RealT* e2,
                             RealT* x, int num_coords )
{
  for ( int i{ 0 }; i < num_coords; ++i ) {
    for ( int d{ 0 }; d < 3; ++d ) {
      x[d * num_coords + i] = x0[d] + xp[i] * e1[d] + yp[i] * e2[d];
    }
  }
}

/*!
 *
 * \brief Projects a point in 2D space to a segment
 *
 * \param [in] x coordinate of point to be projected
 * \param [in] y coordinate of point to be projected
 * \param [in] nx x component of unit normal defining segment
 * \param [in] ny y component of unit normal defining segment
 * \param [in] ox x coordinate of reference point on segment
 * \param [in] oy y coordinate of reference point on segment
 *
 * \param [out] px x coordinate of projected point
 * \param [out] py y coordinate of projected point
 *
 */
TRIBOL_HOST_DEVICE void ProjectPointToSegment( const RealT x, const RealT y, const RealT nx, const RealT ny,
                                               const RealT ox, const RealT oy, RealT& px, RealT& py );

/*
 *
 * \brief Method to find the intersection area between two polygons and
 *  the local y-coordinate of the centroid
 *
 * \param [in] namax number of vertices in polygon a
 * \param [in] xa array of local x coordinates of polygon a vertices
 * \param [in] ya array of local y coordinates of polygon b vertices
 * \param [in] nbmax number of vertices in polygon b
 * \param [in] xb array of local x coordintes of polygon b
 * \param [in] yb array of local y coordinates of polygon b
 * \param [in] isym 0 for planar symmetry, 1 for axial symmetry
 * \param [out] area intersection polygon's area
 * \param [out] ycent local y centroid coordinate
 * \pre length(xa), length(ya) >= namax
 * \pre length(xb), length(yb) >= nbmax
 *
 * \note method to determine area of overlap of two polygons that lie on the same plane
 *  and local centroid y-coordinate. Swap input (xa,ya)->(ya,xa) and (xb,yb)->(yb,xb)
 *  to get centroid x-coordinate. This is the FULL overlap calculation.
 */
TRIBOL_HOST_DEVICE void PolyInterYCentroid( const int namax, const RealT* const xa, const RealT* const ya,
                                            const int nbmax, const RealT* const xb, const RealT* const yb,
                                            const int isym, RealT& area, RealT& ycent );

/*!
 *
 * \brief converts a point in a local 2D coordinate system to a
 *  point in the global 3D coordinate system
 *
 * \param [in] xloc local x coordinate in (e1,e2) frame
 * \param [in] yloc local y coordinate in (e1,e2) frame
 * \param [in] e1X x component of first local basis vector
 * \param [in] e1Y y component of first local basis vector
 * \param [in] e1Z z component of first local basis vector
 * \param [in] e2X x component of second local basis vector
 * \param [in] e2Y y component of second local basis vector
 * \param [in] e2Z z component of second local basis vector
 * \param [in] cX global x coordinate of local basis shift
 * \param [in] cY global y coordinate of local basis shift
 * \param [in] cZ global z coordinate of local basis shift
 * \param [out] xg global x coordinate
 * \param [out] yg global y coordinate
 * \param [out] zg global z coordinate
 *
 * \note this is used to convert a point on a plane in a local
 *  2D coordinate basis to a point in the 3D global coordinate system
 *
 */
TRIBOL_HOST_DEVICE void Local2DToGlobalCoords( RealT xloc, RealT yloc, RealT e1X, RealT e1Y, RealT e1Z, RealT e2X,
                                               RealT e2Y, RealT e2Z, RealT cX, RealT cY, RealT cZ, RealT& xg, RealT& yg,
                                               RealT& zg );

/*!
 *
 * \brief converts an array of points in the global coordinate system to a 2D
 *  local basis
 *
 * \param [in] pX array of x coordinates of points in global coordinate system
 * \param [in] pY array of y coordinates of points in global coordinate system
 * \param [in] pZ array of z coordinates of points in global coordinate system
 * \param [in] e1X x component of local e1 basis vector
 * \param [in] e1Y y component of local e1 basis vector
 * \param [in] e1Z z component of local e1 basis vector
 * \param [in] e2X x component of local e2 basis vector
 * \param [in] e2Y y component of local e2 basis vector
 * \param [in] e2Z z component of local e2 basis vector
 * \param [in] cX global x coordinate of local basis shift
 * \param [in] cY global y coordinate of local basis shift
 * \param [in] cZ global z coordinate of local basis shift
 * \param [out] pLX array of local x coordinates of input points
 * \param [out] pLY array of local y coordinates of input points
 *
 * \pre length(pX) >= size
 * \pre length(pY) >= size
 * \pre length(pZ) >= size
 * \pre length(pLX) >= size
 * \pre length(pLY) >= size
 *
 * \note this assumes that the point lies in the plane defined by the
 *  2D local basis vectors.
 */
TRIBOL_HOST_DEVICE void GlobalTo2DLocalCoords( const RealT* const pX, const RealT* const pY, const RealT* const pZ,
                                               RealT e1X, RealT e1Y, RealT e1Z, RealT e2X, RealT e2Y, RealT e2Z,
                                               RealT cX, RealT cY, RealT cZ, RealT* const pLX, RealT* const pLY,
                                               int size );

/*!
 *
 * \brief converts a point in the global coordinate system to a 2D
 *  local basis
 *
 * \param [in] pX x coordinate of point in global coordinate system
 * \param [in] pY y coordinate of point in global coordinate system
 * \param [in] pZ z coordinate of point in global coordinate system
 * \param [in] e1X x component of local e1 basis vector
 * \param [in] e1Y y component of local e1 basis vector
 * \param [in] e1Z z component of local e1 basis vector
 * \param [in] e2X x component of local e2 basis vector
 * \param [in] e2Y y component of local e2 basis vector
 * \param [in] e2Z z component of local e2 basis vector
 * \param [in] cX global x coordinate of local basis shift
 * \param [in] cY global y coordinate of local basis shift
 * \param [in] cZ global z coordinate of local basis shift
 * \param [out] pLX local x coordinate of input point
 * \param [out] pLY local y coordinate of input point
 *
 * \note this assumes that the point lies in the plane defined by the
 *  2D local basis vectors.
 */
TRIBOL_HOST_DEVICE void GlobalTo2DLocalCoords( RealT pX, RealT pY, RealT pZ, RealT e1X, RealT e1Y, RealT e1Z, RealT e2X,
                                               RealT e2Y, RealT e2Z, RealT cX, RealT cY, RealT cZ, RealT& pLX,
                                               RealT& pLY );
/*!
 *
 * \brief computes the vertex averaged centroid of a point set
 *
 * \param [in] x array of x coordinates for point set
 * \param [in] y array of y coordinates for point set
 * \param [in] z array of z coordinates for point set
 * \param [in] numVert number of points in point set
 * \param [out] cX x coordinate of vertex averaged centroid
 * \param [out] cY y coordinate of vertex averaged centroid
 * \param [out] cZ z coordinate of vertex averaged centroid
 *
 * \return true if calculation successful, false if an error occurred
 *
 * \pre length(x) >= numVert
 * \pre length(y) >= numVert
 * \pre length(z) >= numVert
 *
 */
TRIBOL_HOST_DEVICE inline bool VertexAvgCentroid( const RealT* const x, const RealT* const y, const RealT* const z,
                                                  const int numVert, RealT& cX, RealT& cY, RealT& cZ )
{
#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
  SLIC_ERROR_IF( numVert == 0, "VertexAvgCentroid: numVert = 0." );
#endif
  if ( numVert == 0 ) {
    return false;
  }

  // (re)initialize the input/output centroid components
  cX = 0.0;
  cY = 0.0;
  cZ = 0.0;

  // loop over nodes adding the position components
  RealT fac = 1.0 / numVert;
  for ( int i = 0; i < numVert; ++i ) {
    cX += x[i];
    cY += y[i];
    if ( z != nullptr ) {
      cZ += z[i];
    }
  }

  // divide by the number of nodes to compute average
  cX *= fac;
  cY *= fac;
  cZ *= fac;

  return true;
}

/*!
 *
 * \brief computes the vertex averaged centroid of a point set
 *
 * \param [in] x array of stacked coordinates for point set
 * \param [in] dim 2D or 3D coordinate dimension
 * \param [in] numVert number of points in point set
 * \param [out] cX x coordinate of vertex averaged centroid
 * \param [out] cY y coordinate of vertex averaged centroid
 * \param [out] cZ z coordinate of vertex averaged centroid
 *
 * \return true if calculation successful, false if an error occurred
 *
 * \pre length(x) >= numVert
 *
 */
TRIBOL_HOST_DEVICE bool VertexAvgCentroid( const RealT* const x, const int dim, const int numVert, RealT& cX, RealT& cY,
                                           RealT& cZ );

/*!
 *
 * \brief computes the area centroid of a polygon
 *
 * \param [in] x array of stacked coordinates for point set
 * \param [in] dim 2D or 3D coordinate dimension
 * \param [in] numVert number of points in point set
 * \param [out] cX x coordinate of vertex averaged centroid
 * \param [out] cY y coordinate of vertex averaged centroid
 * \param [out] cZ z coordinate of vertex averaged centroid
 *
 * \return true if calculation successful, false if an error occurred
 *
 * \pre length(x) >= numVert
 *
 */
TRIBOL_HOST_DEVICE bool PolyAreaCentroid( const RealT* const x, const int dim, const int numVert, RealT& cX, RealT& cY,
                                          RealT& cZ );

/*!
 *
 * \brief computes the centroid of the polygon
 *
 * \param [in] x array of x-coordinates for point set
 * \param [in] y array of y-coordinates for point set
 * \param [in] numVert number of points in point set
 * \param [out] cX x coordinate of vertex averaged centroid
 * \param [out] cY y coordinate of vertex averaged centroid
 *
 * \pre length(x) >= numVert
 *
 */
inline void PolyCentroid( const RealT* const x, const RealT* const y, const int numVert, RealT& cX, RealT& cY )
{
#ifndef TRIBOL_USE_ENZYME
  SLIC_ERROR_IF( numVert == 0, "PolyAreaCentroid: numVert = 0." );
#endif

  // (re)initialize the input/output centroid components
  cX = 0.0;
  cY = 0.0;

  RealT area = 0.;

  for ( int i = 0; i < numVert; ++i ) {
    int i_plus_one = ( i + 1 ) % numVert;
    cX += ( x[i] + x[i_plus_one] ) * ( x[i] * y[i_plus_one] - x[i_plus_one] * y[i] );
    cY += ( y[i] + y[i_plus_one] ) * ( x[i] * y[i_plus_one] - x[i_plus_one] * y[i] );
    area += ( x[i] * y[i_plus_one] - x[i_plus_one] * y[i] );
  }

  area *= 1. / 2.;

  RealT fac = 1. / ( 6. * area );
  cX *= fac;
  cY *= fac;
}

/*!
 *
 * \brief check to confirm orientation of polygon vertices are counter clockwise (CCW)
 *
 * \param [in] x array of local x coordinates
 * \param [in] y array of local y coordinates
 * \param [in] numVertex number of vertices
 *
 * \return true if CCW orientation, false otherwise
 *
 */
TRIBOL_HOST_DEVICE inline bool CheckPolyOrientation( const RealT* const x, const RealT* const y, const int numVertex )
{
  bool check = true;
  for ( int i = 0; i < numVertex; ++i ) {
    // determine vertex indices of the segment
    int ia = i;
    int ib = ( i == ( numVertex - 1 ) ) ? 0 : ( i + 1 );

    // compute segment vector
    RealT lambdaX = x[ib] - x[ia];
    RealT lambdaY = y[ib] - y[ia];

    // determine segment normal
    RealT nrmlx = -lambdaY;
    RealT nrmly = lambdaX;

    // compute vertex-averaged centroid
    RealT* z = nullptr;
    RealT xc, yc, zc;
    VertexAvgCentroid( x, y, z, numVertex, xc, yc, zc );

    // compute vector between centroid and first vertex of current segment
    RealT vx = xc - x[ia];
    RealT vy = yc - y[ia];

    // compute dot product between segment normal and centroid-to-vertex vector.
    // the normal points inward toward the centroid
    RealT prod = vx * nrmlx + vy * nrmly;

    if ( prod < 0. )  // don't keep checking
    {
      check = false;
      break;
    }
  }
  return check;
}

/*!
 *
 * \brief check to see if a point in a local 2D coordinate system lies in a triangle
 *
 * \param [in] xp local x coordinate of point
 * \param [in] yp local y coordinate of point
 * \param [in] xTri array of local x coordinates of triangle
 * \param [in] yTri array of local y coordinates of triangle
 * \param [in] tol "fuzz" length: how far outside the triangle a point can be to still be considered inside
 *
 * \return true if the point is inside the triangle, false otherwise
 *
 * \pre length(xTri), length(yTri) >= 3
 *
 * \note this routine finds the two barycentric coordinates of the triangle and
 *  determines if those coordinates are inside or out
 *  (http://blackpawn.com/texts/pointinpoly/default.html);
 */
TRIBOL_HOST_DEVICE inline bool Point2DInTri( const RealT xp, const RealT yp, const RealT* const xTri,
                                             const RealT* const yTri, RealT tol = 1.0e-12 )
{
  bool inside = false;

  // compute coordinate basis between the 1-2 and 1-3 vertices
  RealT e1x = xTri[1] - xTri[0];
  RealT e1y = yTri[1] - yTri[0];

  RealT e2x = xTri[2] - xTri[0];
  RealT e2y = yTri[2] - yTri[0];

  // compute vector components of vector between point and first vertex
  RealT p1x = xp - xTri[0];
  RealT p1y = yp - yTri[0];

  // compute dot products (e1,e1), (e1,e2), (e2,e2), (p1,e1), and (p1,e2)
  RealT e11 = e1x * e1x + e1y * e1y;
  RealT e12 = e1x * e2x + e1y * e2y;
  RealT e22 = e2x * e2x + e2y * e2y;
  RealT p1e1 = p1x * e1x + p1y * e1y;
  RealT p1e2 = p1x * e2x + p1y * e2y;

  // compute the inverse determinant
  RealT invDet = 1.0 / ( e11 * e22 - e12 * e12 );

  // compute 2 local barycentric coordinates
  RealT u = invDet * ( e22 * p1e1 - e12 * p1e2 );
  RealT v = invDet * ( e11 * p1e2 - e12 * p1e1 );

  // check if point is inside the triangle within a tolerance
  if ( ( u >= -tol ) && ( u <= 1. ) && ( v >= -tol ) && ( v <= 1. ) && ( u + v <= 1.0 ) ) {
    inside = true;
  }

  return inside;
}

/*!
 *
 * \brief check to see if a point is in a convex polygonal face
 *
 * \param [in] xPoint local x coordinate of point to be checked
 * \param [in] yPoint local y coordinate of point to be checked
 * \param [in] xPoly array of local x coordinates of polygon
 * \param [in] yPoly array of local y coordinates of polygon
 * \param [in] xC local x coordinate of vertex averaged centroid
 * \param [in] yC local y coordinate of vertex averaged centroid
 * \param [in] numPolyVert number of polygon vertices
 * \param [in] tol "fuzz" length: how far outside the face a point can be to still be considered inside
 *
 * \return true if the point is in the face, false otherwise.
 *
 * \pre length(xPoly), length(yPoly) >= numPolyVert
 *
 * \note this routine assumes a star convex polygon. It starts by checking
 *  which polygon vertex the input point is closest to, defined as the reference
 *  vertex. Two triangles are then constructed, each using the reference vertex as
 *  the first point, the second vertex is the vertex belonging to one of two edge
 *  segments that share the reference vertex, and the third vertex is the input
 *  vertex averaged centroid. This routine then calls a routine to check if the
 *  point lies in either of those two triangles.
 */
TRIBOL_HOST_DEVICE inline bool Point2DInFace( const RealT xPoint, const RealT yPoint, const RealT* const xPoly,
                                              const RealT* const yPoly, const RealT xC, const RealT yC,
                                              const int numPolyVert, RealT tol = 1.0e-12 )
{
#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
  SLIC_ERROR_IF( numPolyVert < 3, "Point2DInFace: number of face vertices is less than 3" );

  SLIC_ERROR_IF( xPoly == nullptr || yPoly == nullptr, "Point2DInFace: input pointer not set" );
#endif

  // if face is triangle (numPolyVert), call Point2DInTri once
  if ( numPolyVert == 3 ) {
    return Point2DInTri( xPoint, yPoint, xPoly, yPoly, tol );
  }

  // loop over triangles and determine if point is inside
  bool tri = false;
  for ( int i = 0; i < numPolyVert; ++i ) {
    RealT xTri[3];
    RealT yTri[3];

    // construct polygon using i^th segment vertices and face centroid
    xTri[0] = xPoly[i];
    yTri[0] = yPoly[i];

    xTri[1] = ( i == ( numPolyVert - 1 ) ) ? xPoly[0] : xPoly[i + 1];
    yTri[1] = ( i == ( numPolyVert - 1 ) ) ? yPoly[0] : yPoly[i + 1];

    // last vertex of the triangle is the vertex averaged centroid of the polygonal face
    xTri[2] = xC;
    yTri[2] = yC;

    // call Point2DInTri for each triangle
    // NOTE (EBC): we should probably "round" the corners around the tolerance so there aren't weird spikes in the
    // inclusion polygon
    tri = Point2DInTri( xPoint, yPoint, xTri, yTri, tol );

    if ( tri ) {
      return true;
    }
  }
  return false;
}

/*!
 * \brief computes the area of a polygon using the shoelace formula (i.e. Gauss's area formula)
 *
 * \param [in] x array of local x coordinates of polygon vertices
 * \param [in] y array of local y coordinates of polygon vertices
 * \param [in] numPolyVert number of polygon vertices
 *
 * \return area of polygon
 *
 * \note this works for convex and non-conves, though non-self-intersecting polygons
 */
TRIBOL_HOST_DEVICE inline RealT Area2DPolygon( const RealT* const x, const RealT* const y, const int numPolyVert )
{
  RealT area = 0.;

  for ( int i = 0; i < numPolyVert; ++i ) {
    // determine vertex indices of the segment
    int ia = i;
    int ib = ( i == ( numPolyVert - 1 ) ) ? 0 : ( i + 1 );

    area += x[ia] * y[ib] - y[ia] * x[ib];
  }
  return std::abs( 0.5 * area );
}

/*!
 *
 * \brief computes a segment-segment intersection in a way specific to the tribol
 *  polygon-polygon intersection calculation
 *
 * \param [in] xA1 local x coordinate of first vertex (A) of segment 1
 * \param [in] yA1 local y coordinate of first vertex (A) of segment 1
 * \param [in] xB1 local x coordinate of second vertex (B) of segment 1
 * \param [in] yB1 local y coordinate of second vertex (B) of segment 1
 * \param [in] xA2 local x coordinate of first vertex (A) of segment 2
 * \param [in] yA2 local y coordinate of first vertex (A) of segment 2
 * \param [in] xB2 local x coordinate of second vertex (B) of segment 2
 * \param [in] yB2 local y coordinate of second vertex (B) of segment 2
 * \param [in] interior array where each element is set to true if the associated
 *             vertex is interior to the polygon to which the other segment belongs
 * \param [out] x local coordinate of the intersection point
 * \param [out] y local coordinate of the intersection point
 * \param [out] duplicate true if intersection point is computed as duplicate polygon
 *                 intersection point
 * \param [in] tol length tolerance for collapsing intersection points to interior points
 *
 * \return true if the segments intersect at a non-duplicate point, false otherwise
 *
 * \note this routine returns true or false for an intersection point that we are going
 *  to use to define an overlapping polygon. In this sense, this is a routine specific to
 *  the tribol problem. If this routine returns false, but the boolean duplicate is true,
 *  then the x,y coordinates are the coordinates of a segment vertex that is interior
 *  to one of the polygons that is the solution to the intersection problem. The solution
 *  may arise due to the the segments intersecting at a vertex tagged as interior to one
 *  of the polygons, or due to collapsing the true intersection point to an interior
 *  vertex based on the position tolerance input argument. For intersection points that
 *  are within the position tolerance to a non-interior segment vertex, nothing is done
 *  because we want to retain this intersection point. Collapsing intersection points to
 *  vertices tagged as interior to one of the polygons may render a degenerate overlap
 *  polygon and must be checked.
 */
TRIBOL_HOST_DEVICE inline bool SegmentIntersection2D( RealT xA1, RealT yA1, RealT xB1, RealT yB1, RealT xA2, RealT yA2,
                                                      RealT xB2, RealT yB2, const bool* interior, RealT& x, RealT& y,
                                                      bool& duplicate, RealT tol )
{
  // note 1: this routine computes a unique segment-segment intersection, where two
  // segments are assumed to intersect at a single point. A segment-segment overlap
  // is a different computation and is not accounted for here. In the context of the
  // use of this routine in the tribol polygon-polygon intersection calculation,
  // two overlapping segments will have already registered the vertices that form
  // the bounds of the overlapping length as vertices interior to the other polygon
  // and therefore will be in the list of overlapping polygon vertices prior to this
  // routine.
  //
  // note 2: any segment-segment intersection that occurs at a vertex of either segment
  // will pass back the intersection coordinates, but will note a duplicate vertex.
  // This is because that any vertex of polygon A that lies on a segment of polygon B
  // will be caught and registered as a vertex interior to the other polygon and will
  // be in the list of overlapping polygon vertices prior to calling this routine.

  // compute segment vectors
  RealT lambdaX1 = xB1 - xA1;
  RealT lambdaY1 = yB1 - yA1;

  RealT lambdaX2 = xB2 - xA2;
  RealT lambdaY2 = yB2 - yA2;

  RealT seg1Mag = magnitude( lambdaX1, lambdaY1 );
  RealT seg2Mag = magnitude( lambdaX2, lambdaY2 );

  // compute determinant of the lambda matrix, [ -lx1 -ly1, lx2 ly2 ]
  RealT det = -lambdaX1 * lambdaY2 + lambdaX2 * lambdaY1;

  // return false if det = 0. Check for numerically zero determinant
  // nearly colinear edges will have det ~= 0.
  RealT detTol = 1.E-12;
  if ( det > -detTol && det < detTol ) {
    x = 0.;
    y = 0.;
    duplicate = false;
    return false;
  }

  // compute intersection
  RealT invDet = 1.0 / det;
  RealT rX = xA1 - xA2;
  RealT rY = yA1 - yA2;
  RealT tA = invDet * ( rX * lambdaY2 - rY * lambdaX2 );
  RealT tB = invDet * ( rX * lambdaY1 - rY * lambdaX1 );

  // if tA and tB don't lie between [0,1] then return false.
  if ( ( tA < 0. || tA > 1. ) || ( tB < 0. || tB > 1. ) ) {
    // no intersection
    x = 0.;
    y = 0.;
    duplicate = false;
    return false;
  }

  // TODO refine how these debug calculations are guarded
  {
    // debug check to make sure the intersection coordinates derived from
    // each segment equation (scaled with tA and tB) are the same to some
    // tolerance
    RealT xTest1 = xA1 + lambdaX1 * tA;
    RealT yTest1 = yA1 + lambdaY1 * tA;
    RealT xTest2 = xA2 + lambdaX2 * tB;
    RealT yTest2 = yA2 + lambdaY2 * tB;

    RealT xDiff = xTest1 - xTest2;
    RealT yDiff = yTest1 - yTest2;

    // make sure the differences are positive
    xDiff = ( xDiff < 0. ) ? -1.0 * xDiff : xDiff;
    yDiff = ( yDiff < 0. ) ? -1.0 * yDiff : yDiff;

#if defined( TRIBOL_DEBUG ) && defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
    RealT diffTol = 1.0E-3;
    SLIC_DEBUG_IF( xDiff > diffTol || yDiff > diffTol,
                   "SegmentIntersection2D(): Intersection coordinates are not equally derived." );
#endif
  }

  // if we get here then it means we have an intersection point.
  // Find the minimum distance of the intersection point to any of the segment
  // vertices.
  x = xA1 + lambdaX1 * tA;
  y = yA1 + lambdaY1 * tA;

  // for convenience, define an array of pointers that point to the
  // input coordinates
  RealT xVert[4];
  RealT yVert[4];

  xVert[0] = xA1;
  xVert[1] = xB1;
  xVert[2] = xA2;
  xVert[3] = xB2;

  yVert[0] = yA1;
  yVert[1] = yB1;
  yVert[2] = yA2;
  yVert[3] = yB2;

  RealT distX[4];
  RealT distY[4];
  RealT distMag[4];

  for ( int i = 0; i < 4; ++i ) {
    distX[i] = x - xVert[i];
    distY[i] = y - yVert[i];
    distMag[i] = magnitude( distX[i], distY[i] );
  }

  RealT distMin = distMag[0];
  int idMin = 0;
  RealT xMinVert = xVert[0];
  RealT yMinVert = yVert[0];

  for ( int i = 1; i < 4; ++i ) {
    if ( distMag[i] < distMin ) {
      distMin = distMag[i];
      idMin = i;
      xMinVert = xVert[i];
      yMinVert = yVert[i];
    }
  }

  // check to see if the minimum distance is less than the position tolerance for
  // the segments
  RealT distRatio = ( idMin == 0 || idMin == 1 ) ? ( distMin / seg1Mag ) : ( distMin / seg2Mag );

  // if the distRatio is less than the tolerance, or percentage cutoff of the original
  // segment that we would like to keep, then check to see if the segment vertex closest
  // to the computed intersection point is an interior point. If this is true, then collapse
  // the computed intersection point to the interior point and mark the duplicate boolean.
  // Also do this for the argument, interior, set to nullptr
  if ( distRatio < tol ) {
    if ( interior == nullptr ) {
      x = xMinVert;
      y = yMinVert;
      duplicate = true;
      return false;
    } else if ( interior[idMin] ) {
      x = xMinVert;
      y = yMinVert;
      duplicate = true;
      return false;
    }
  }

  // if we are here we are ready to return the true intersection point
  duplicate = false;
  return true;
}

/*!
 *
 * \brief reorders a set of unordered vertices associated with a star convex polygon in
 *        counter clockwise orientation
 *
 * \param [in,out] x array of local x vertex coordinates
 * \param [in,out] y array of local y vertex coordinates
 * \param [in,out] newIDs array of vertex IDs in output polygon that correspond to input polygon vertices
 * \param [in] numPoints number of vertices
 *
 * \return true if calculation successful, false if an error occurred
 *
 * \pre length(x), length(y) >= numPoints
 *
 * \note This routine takes the unordered set of vertex coordinates of a star convex
 *  polygon and orders the vertices in counter-clockwise orientation.
 */
TRIBOL_HOST_DEVICE inline bool PolyReorderConvex( RealT* x, RealT* y, int* newIDs, int numPoints )
{
  if ( numPoints < 3 ) {
#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
    SLIC_DEBUG( "PolyReorderConvex: numPoints (" << numPoints << ") < 3." );
#endif
    return false;
  }

  RealT xC, yC, zC;
  RealT* z = nullptr;
  constexpr int max_nodes_per_overlap = 5 * 2;  // 5 max verts for a given interpen face-portion

#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
  SLIC_ERROR_IF( numPoints > max_nodes_per_overlap, "PolyReorderConvex: numPoints exceed maximum "
                                                        << "expected per overlap (" << max_nodes_per_overlap << ")." );
#endif

  constexpr int max_proj_nodes = max_nodes_per_overlap - 2;
  RealT proj[max_proj_nodes];

  int local_newIDs[max_nodes_per_overlap];
  if ( !newIDs ) {
    newIDs = local_newIDs;
  }

  // initialize newIDs array to local ordering, 0,1,2,...,numPoints-1
  for ( int i = 0; i < numPoints; ++i ) {
    newIDs[i] = i;
  }

  // compute vertex averaged centroid of input overlap vertices (local coordinates with dummy z args)
  VertexAvgCentroid( x, y, z, numPoints, xC, yC, zC );

  // using the FIRST index into the x,y vertex coordinate arrays as
  // the first vertex of the soon-to-be ordered list of vertices, determine
  // the NEXT vertex that will comprise the only the FIRST segment in a counter
  // clockwise ordering of vertices
  newIDs[0] = 0;
  for ( int j = newIDs[1]; j < numPoints; ++j ) {
    // determine current segment vector and normal
    RealT lambdaX = x[j] - x[newIDs[0]];
    RealT lambdaY = y[j] - y[newIDs[0]];
    RealT nrmlx = -lambdaY;
    RealT nrmly = lambdaX;

    // project all segment vectors between all OTHER vertices and newIDs[0] onto the current
    // segment vector's normal. There will always be numPoints-2 projections
    int pk = 0;                              // projection counter
    for ( int k = 0; k < numPoints; ++k ) {  // loop over all segments
      if ( k != newIDs[0] && k != j ) {      // pick off segments that are NOT the current segment
        proj[pk] = ( x[k] - x[newIDs[0]] ) * nrmlx + ( y[k] - y[newIDs[0]] ) * nrmly;
        ++pk;
      }
    }

    // check if all points are on one side of line defined by segment
    // (pk at this point should be equal to numPoints - 2)
    bool neg = false;
    bool pos = false;
    for ( int ip = 0; ip < pk; ++ip ) {
      if ( neg ) {  // if neg is previously set to true, keep it true
        neg = true;
      } else if ( !neg ) {
        neg = ( proj[ip] < 0. ) ? true : false;
      }

      if ( pos ) {  // if pos is previously set to true, keep it true
        pos = true;
      } else if ( !pos ) {
        pos = ( proj[ip] > 0. ) ? true : false;
      }

      // if at least one projection is negative and one positive then the
      // current vertex of the current segment vector is not the properly
      // ordered next vertex
      if ( neg && pos ) {
        break;
      }
    }

    // if one of the booleans is false then all points are on one side
    // of line defined by i-j segment.
    if ( !neg || !pos ) {
      // check the orientation of the nodes to make sure we have the correct
      // one of two segments that will pass the previous test.
      // Check the dot product between the current segment normal and the vector
      // between the centroid and first (0th) vertex
      RealT vx = xC - x[newIDs[0]];
      RealT vy = yC - y[newIDs[0]];

      RealT prod = nrmlx * vx + nrmly * vy;

      // check if the two vertices are a segment on the convex hull and oriented CCW.
      // CCW orientation has prod > 0
      if ( prod > 0 ) {
        // set newIDs[1] to the current vertex where newIDs[1] and newIDs[0] form the
        // first segment vector on the convex hull; then, swap ids
        int oldID1 = newIDs[1];
        newIDs[1] = j;
        newIDs[j] = oldID1;
        break;
      }
    }

  }  // end loop over j

  // given the first segment vector on the convex hull, determine the rest of the vertex ordering
  //
  // compute the current reference segment vector between currently ordered vertices. At first, this is simply
  // taken as the first segment vector determined above. Then, loop over remaining unorderd vertices and compute
  // the link vector between that unordered vertex and the first vertex in the reference segment vector. These
  // two vectors share that vertex as a common origin. Then, compute the angle between the link vector and the
  // current reference vector. The link vector with the smallest angle gives us the next vertex in the ordered set
  //
  // Note: increment to (numPoints - 3) as as the (number_of_remaining_vertices-1) where the last vertex
  // will automatically
  for ( int i = 0; i < ( numPoints - 3 ); ++i ) {
    RealT refMag, linkMag;

    // compute current ordered reference vector;
    RealT refx, refy;
    refx = x[newIDs[i + 1]] - x[newIDs[i]];
    refy = y[newIDs[i + 1]] - y[newIDs[i]];
    refMag = magnitude( refx, refy );

    //      SLIC_ERROR_IF(refMag < 1.E-12, "PolyReorderConvex: reference segment for link vector check is nearly zero
    //      length");

    // loop over link vectors of unassigned vertices
    int jID = -1;
    RealT cosThetaMax = -1.;  // this handles angles up to 180 degrees. Any greater and the polygon is not convex
    RealT cosTheta;
    int nextVertexID = 2 + i;
    for ( int j = nextVertexID; j < numPoints; ++j ) {
      RealT lx, ly;

      lx = x[newIDs[j]] - x[newIDs[i]];
      ly = y[newIDs[j]] - y[newIDs[i]];
      linkMag = magnitude( lx, ly );

      cosTheta = ( lx * refx + ly * refy ) / ( refMag * linkMag );
      if ( cosTheta > cosThetaMax ) {
        cosThetaMax = cosTheta;
        jID = j;
      }

    }  // end loop over j

    // we have found the minimum angle between remaining segment vectors and the corresponding local vertex id.
    // swap ids
    if ( jID > -1 ) {
      int swapID = newIDs[nextVertexID];
      newIDs[nextVertexID] = newIDs[jID];
      newIDs[jID] = swapID;
    }

  }  // end loop over i

  // reorder x and y coordinate arrays based on newIDs id-array
  RealT xtemp[max_nodes_per_overlap];
  RealT ytemp[max_nodes_per_overlap];
  for ( int i = 0; i < numPoints; ++i ) {
    xtemp[i] = x[i];
    ytemp[i] = y[i];
  }

  for ( int i = 0; i < numPoints; ++i ) {
    x[i] = xtemp[newIDs[i]];
    y[i] = ytemp[newIDs[i]];
  }

  return true;
}

/*!
 *
 * \brief checks polygon segments to collapse any segments less than input tolerance
 *
 * \param [in] x array of local x coordinates of polygon vertices
 * \param [in] y array of local y coordinates of polygon vertices
 * \param [in] numPoints number of polygon vertices
 * \param [in] tol edge segment tolerance
 * \param [out] xnew array of new x coordinates
 * \param [out] ynew array of new y coordinates
 * \param [out] newIDs array of vertex IDs in new polygon that correspond to input vertices
 * \param [out] numNewPoints number of new points
 *
 * \return 0 if no exception, >0 a face geom exception
 *
 * \pre length(x), length(y) >= numPoints
 *
 * /note this routine checks the overlapping polygon segments. We may have two adjacent
 *  intersection points (vertices derived from two different segment-segment
 *  intersection calculations) that produce a very small polygon edge. We may want
 *  to collapse these segments. Multiple collapses may produce a degenerate polygon,
 *  which needs to be checked. For cases where there is no collapse of segments, then
 *  xnew and ynew values are set to x and y, respectively, and numNewPoints
 *  equals numPoints.
 */
TRIBOL_HOST_DEVICE inline FaceGeomException CheckPolySegs( const RealT* x, const RealT* y, int numPoints, RealT tol,
                                                           RealT* xnew, RealT* ynew, int* newIDs, int& numNewPoints )
{
  constexpr int max_nodes_per_overlap = 5 * 2;  // max five interpen vertices in a single cut face
  int local_newIDs[max_nodes_per_overlap];
  if ( !newIDs ) {
    newIDs = local_newIDs;
  }

  // set newIDs[i] to original local ordering
  for ( int i = 0; i < numPoints; ++i ) {
    newIDs[i] = i;
  }

  for ( int i = 0; i < numPoints; ++i ) {
    // determine vertex indices of the segment
    int ia = i;
    int ib = ( i == ( numPoints - 1 ) ) ? 0 : ( i + 1 );

    // compute segment vector magnitude
    RealT lambdaX = x[ib] - x[ia];
    RealT lambdaY = y[ib] - y[ia];
    RealT lambdaMag = magnitude( lambdaX, lambdaY );

    // check segment length against tolerance
    if ( lambdaMag < tol ) {
      // collapse second vertex to the first vertex of the current segment
      newIDs[ib] = i;
    }
  }

  // determine the number of new points
  numNewPoints = 0;
  for ( int i = 0; i < numPoints; ++i ) {
    if ( newIDs[i] == i ) {
      ++numNewPoints;
    }
  }

  // check to make sure numNewPoints >= 3 for valid overlap polygons prior
  // to memory allocation
  if ( numNewPoints < 3 ) {
    // return and degenerated polygon will be skipped over.
    return NO_FACE_GEOM_EXCEPTION;
  }

  // set the coordinates in xnew and ynew
  int k = 0;
  for ( int i = 0; i < numPoints; ++i ) {
    if ( newIDs[i] == i ) {
      if ( k > numNewPoints ) {
#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
        SLIC_DEBUG( "checkPolySegs(): index into polyX/polyY exceeds allocated space" );
#endif
        return FACE_VERTEX_INDEX_EXCEEDS_OVERLAP_VERTICES;
      }

      xnew[k] = x[i];
      ynew[k] = y[i];
      ++k;
    }
  }

  return NO_FACE_GEOM_EXCEPTION;
}

/*!
 * \brief Defines the vertex type in an intersection polygon
 */
enum class OverlapVertexType
{
  A,        ///! vertex in polygon A
  B,        ///! vertex in polygon B
  EdgeEdge  ///! vertex at intersection of edges of polygons A and B
};

/*!
 *
 * \brief computes the hard intersection between two coplanar polygons
 *
 * \param [in] xA array of local x coordinates of polygon A
 * \param [in] yA array of local y coordinates of polygon A
 * \param [in] numVertexA number of vertices in polygon A
 * \param [in] xB array of local x coordinates of polygon B
 * \param [in] yB array of local y coordinates of polygon B
 * \param [in] numVertexB number of vertices in polygon B
 * \param [in] posTol position tolerance to collapse segment-segment intersection points
 * \param [in] lenTol length tolerance to collapse short intersection edges
 * \param [out] polyX array of x coordinates of intersection polygon
 * \param [out] polyY array of y coordinates of intersection polygon
 * \param [out] numPolyVert number of vertices in intersection polygon
 * \param [out] area intersection polygon area
 * \param [in] orientCheck checks if vertices of each polygon are oriented correctly
 * \param [out] vertType classification of each vertex in the intersection polygon. optional. use nullptr if not
 * needed.
 * \param [out] edgeA associated vertex or edge on polygon A for each vertex in the intersection polygon.
 * optional. use nullptr if not needed.
 * \param [out] edgeB associated vertex or edge on polygon B for each vertex in the intersection polygon.
 * optional. use nullptr if not needed.
 *
 * \return 0 if no exception, >0 a face geom exception
 *
 * \pre length(xA), length(yA) >= numVertexA
 * \pre length(xB), length(yB) >= numVertexB
 *
 * \note polyX and polyY must be pre-allocated and sized to the maximum number
 * of points for the intersection polygon
 *
 */
TRIBOL_HOST_DEVICE inline FaceGeomException Intersection2DPolygon(
    const RealT* xA, const RealT* yA, int numVertexA, const RealT* xB, const RealT* yB, int numVertexB, RealT posTol,
    RealT lenTol, RealT* polyX, RealT* polyY, int& numPolyVert, RealT& area, bool orientCheck = true,
    OverlapVertexType* vertType = nullptr, int* edgeA = nullptr, int* edgeB = nullptr )
{
  // for tribol, if you have called this routine it is because a positive area of
  // overlap between two polygons (faces) exists. This routine does not perform a
  // "proximity" check to determine if the faces are "close enough" to proceed with
  // the full calculation. This can and probably should be added.

  // check numVertexA and numVertexB to make sure they are 3 (triangle) or more
  if ( numVertexA < 3 || numVertexB < 3 ) {
#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
    SLIC_DEBUG( "Intersection2DPolygon(): one or more degenerate faces with < 3 vertices." );
#endif
    area = 0.0;
    return INVALID_FACE_INPUT;
  }

  // check right hand rule ordering of polygon vertices.
  // Note 1: This check is consistent with the ordering that comes from PolyReorderConvex()
  // of two faces with unordered vertices.
  // Note 2: Intersection2DPolygon doesn't require consistent face vertex orientation
  // between faces, as long as each are 'ordered' (CW or CCW).
  if ( orientCheck ) {
    bool orientA = CheckPolyOrientation( xA, yA, numVertexA );
    bool orientB = CheckPolyOrientation( xB, yB, numVertexB );

    if ( !orientA || !orientB ) {
#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
      SLIC_DEBUG( "Intersection2DPolygon(): check face orientations for face A." );
#endif
      return FACE_ORIENTATION;
    }
  }

  // maximum number of vertices for potentially clipped four node quads (for use later)
  constexpr int max_nodes_per_element = 5;

  // allocate an array to hold ids of interior vertices
  int interiorVAId[max_nodes_per_element];
  int interiorVBId[max_nodes_per_element];

  // initialize all entries in interior vertex array to -1
  initIntArray( &interiorVAId[0], numVertexA, -1 );
  initIntArray( &interiorVBId[0], numVertexB, -1 );

  // precompute the vertex averaged centroids for both polygons.
  RealT xCA = 0.0;
  RealT yCA = 0.0;
  RealT xCB = 0.0;
  RealT yCB = 0.0;
  RealT zC = 0.0;  // not required, only used as dummy argument in centroid routine

  VertexAvgCentroid( xA, yA, nullptr, numVertexA, xCA, yCA, zC );
  VertexAvgCentroid( xB, yB, nullptr, numVertexB, xCB, yCB, zC );

  // check to see if any of polygon A's vertices are in polygon B, and vice-versa. Track
  // which vertices are interior to the other polygon. Keep in mind that vertex
  // coordinates are local 2D coordinates.
  int numVAI = 0;
  int numVBI = 0;

  // check A in B
  for ( int i = 0; i < numVertexA; ++i ) {
    if ( Point2DInFace( xA[i], yA[i], xB, yB, xCB, yCB, numVertexB ) ) {
      // interior A in B
      interiorVAId[i] = i;
      ++numVAI;
    }
  }

  // check to see if ALL of A is in B; then A is the overlapping polygon.
  if ( numVAI == numVertexA ) {
    numPolyVert = numVertexA;
    for ( int i = 0; i < numVertexA; ++i ) {
      polyX[i] = xA[i];
      polyY[i] = yA[i];
      if ( vertType ) {
        vertType[i] = OverlapVertexType::A;
      }
      // set all edgeA to polygon A vertex IDs
      if ( edgeA ) {
        edgeA[i] = i;
      }
      // set all edgeB to -1 since all vertices are on polygon A
      if ( edgeB ) {
        edgeB[i] = -1;
      }
    }
    area = Area2DPolygon( polyX, polyY, numVertexA );
    return NO_FACE_GEOM_EXCEPTION;
  }

  // check B in A
  for ( int i = 0; i < numVertexB; ++i ) {
    if ( Point2DInFace( xB[i], yB[i], xA, yA, xCA, yCA, numVertexA ) ) {
      // interior B in A
      interiorVBId[i] = i;
      ++numVBI;
    }
  }

  // check to see if ALL of B is in A; then B is the overlapping polygon.
  if ( numVBI == numVertexB ) {
    numPolyVert = numVertexB;
    for ( int i = 0; i < numVertexB; ++i ) {
      polyX[i] = xB[i];
      polyY[i] = yB[i];
      if ( vertType ) {
        vertType[i] = OverlapVertexType::B;
      }
      // set all edgeA to -1 since all vertices are on polygon B
      if ( edgeA ) {
        edgeA[i] = -1;
      }
      // set all edgeB to polygon B vertex IDs
      if ( edgeB ) {
        edgeB[i] = i;
      }
    }
    area = Area2DPolygon( polyX, polyY, numVertexB );
    return NO_FACE_GEOM_EXCEPTION;
  }

  // check for coincident interior vertices. That is, a vertex on A interior to
  // B occupies the same point in space as a vertex on B interior to A. This is
  // O(n^2), but the number of interior vertices is anticipated to be small
  // if we are at this location in the routine
  for ( int i = 0; i < numVertexA; ++i ) {
    if ( interiorVAId[i] != -1 ) {
      for ( int j = 0; j < numVertexB; ++j ) {
        if ( interiorVBId[j] != -1 ) {
          // compute the distance between interior vertices
          RealT distX = xA[i] - xB[j];
          RealT distY = yA[i] - yB[j];
          RealT distMag = magnitude( distX, distY );
          if ( distMag < 1.E-15 ) {
            // remove the interior designation for the vertex in polygon B
            //                 SLIC_DEBUG( "Removing duplicate interior vertex id: " << j << ".\n" );
            interiorVBId[j] = -1;
            numVBI -= 1;
          }
        }
      }
    }
  }

  // determine the maximum number of intersection points

  // allocate space to store the segment-segment intersection vertex coords.
  // and a boolean array to indicate intersecting pairs
  constexpr int max_intersections = max_nodes_per_element * max_nodes_per_element;
  RealT interX[max_intersections];
  RealT interY[max_intersections];
  bool intersect[max_intersections];
  int edgeATemp[max_intersections];
  int edgeBTemp[max_intersections];
  bool dupl;  // boolean to indicate a segment-segment intersection that
              // duplicates an existing interior vertex.
  bool interior[4];

  // initialize the interX and interY entries
  initRealArray( interX, max_intersections, 0. );
  initRealArray( interY, max_intersections, 0. );
  initBoolArray( intersect, max_intersections, false );
  initIntArray( edgeATemp, max_intersections, 0 );
  initIntArray( edgeBTemp, max_intersections, 0 );
  dupl = false;

  // loop over segment-segment intersections to find the rest of the
  // intersection vertices. This is O(n^2), but segments defined by two
  // nodes interior to the other polygon will be skipped. This will catch
  // outlier cases.
  int interId = 0;

  // loop over A segments
  for ( int ia = 0; ia < numVertexA; ++ia ) {
    int vAID1 = ia;
    int vAID2 = ( ia == ( numVertexA - 1 ) ) ? 0 : ( ia + 1 );

    // set boolean indicating which nodes on segment A are interior
    interior[0] = ( interiorVAId[vAID1] != -1 ) ? true : false;
    interior[1] = ( interiorVAId[vAID2] != -1 ) ? true : false;
    //      bool checkA = (interior[0] == -1 && interior[1] == -1) ? true : false;
    bool checkA = true;

    // loop over B segments
    for ( int jb = 0; jb < numVertexB; ++jb ) {
      int vBID1 = jb;
      int vBID2 = ( jb == ( numVertexB - 1 ) ) ? 0 : ( jb + 1 );
      interior[2] = ( interiorVBId[vBID1] != -1 ) ? true : false;
      interior[3] = ( interiorVBId[vBID2] != -1 ) ? true : false;
      //         bool checkB = (interior[2] == -1 && interior[3] == -1) ? true : false;
      bool checkB = true;

      // if both segments are not defined by nodes interior to the other polygon
      // UPDATE: just check all segment-segment intersections for robustness
      if ( checkA && checkB ) {
        if ( interId >= max_intersections ) {
#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
          SLIC_DEBUG( "Intersection2DPolygon: number of segment/segment intersections exceeds precomputed maximum; "
                      << "check for degenerate overlap." );
#endif
          return DEGENERATE_OVERLAP;
        }

        intersect[interId] =
            SegmentIntersection2D( xA[vAID1], yA[vAID1], xA[vAID2], yA[vAID2], xB[vBID1], yB[vBID1], xB[vBID2],
                                   yB[vBID2], interior, interX[interId], interY[interId], dupl, posTol );
        if ( intersect[interId] ) {
          edgeATemp[interId] = ia;
          edgeBTemp[interId] = jb;
          ++interId;  // increment intersection counter for segments that intersect
        }
      }
    }  // end loop over A segments
  }  // end loop over B segments

  // count the number of segment-segment intersections
  int numSegInter = 0;
  for ( int i = 0; i < interId; ++i ) {
    if ( intersect[i] ) ++numSegInter;
  }

  // add check for case where there are no interior vertices or
  // intersection vertices
  if ( numSegInter == 0 && numVBI == 0 && numVAI == 0 ) {
    area = 0.0;
    return NO_OVERLAP;
  }

  // allocate temp intersection polygon vertex coordinate arrays to consist
  // of segment-segment intersections and number of interior points in A and B
  numPolyVert = numSegInter + numVAI + numVBI;
  // maximum number of vertices between the two polygons.  assumes convex elements.
  constexpr int max_nodes_per_overlap = 2 * max_nodes_per_element;
  constexpr int max_identified_points = max_nodes_per_overlap + 2 * max_nodes_per_element;
  RealT polyXTemp[max_identified_points];
  RealT polyYTemp[max_identified_points];
  OverlapVertexType vertTypeTemp[max_identified_points];

  // fill polyXTemp and polyYTemp with the intersection points
  int k = 0;
  for ( int i = 0; i < interId; ++i ) {
    if ( intersect[i] ) {
      polyXTemp[k] = interX[i];
      polyYTemp[k] = interY[i];
      vertTypeTemp[k] = OverlapVertexType::EdgeEdge;
      edgeATemp[k] = edgeATemp[i];
      edgeBTemp[k] = edgeBTemp[i];
      ++k;
    }
  }

  // fill polyX and polyY with the vertices on A that lie in B
  for ( int i = 0; i < numVertexA; ++i ) {
    if ( interiorVAId[i] != -1 ) {
      // debug
      if ( k > max_identified_points ) {
#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
        SLIC_DEBUG( "Intersection2DPolygon(): number of A vertices interior to B "
                    << "polygon exceeds total number of overlap vertices. Check interior vertex id values." );
#endif
        return FACE_VERTEX_INDEX_EXCEEDS_OVERLAP_VERTICES;
      }

      polyXTemp[k] = xA[i];
      polyYTemp[k] = yA[i];
      vertTypeTemp[k] = OverlapVertexType::A;
      edgeATemp[k] = i;
      edgeBTemp[k] = -1;
      ++k;
    }
  }

  for ( int i = 0; i < numVertexB; ++i ) {
    if ( interiorVBId[i] != -1 ) {
      // debug
      if ( k > max_identified_points ) {
#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
        SLIC_DEBUG( "Intersection2DPolygon(): number of B vertices interior to A "
                    << "polygon exceeds total number of overlap vertices. Check interior vertex id values." );
#endif
        return FACE_VERTEX_INDEX_EXCEEDS_OVERLAP_VERTICES;
      }

      polyXTemp[k] = xB[i];
      polyYTemp[k] = yB[i];
      vertTypeTemp[k] = OverlapVertexType::B;
      edgeATemp[k] = -1;
      edgeBTemp[k] = i;
      ++k;
    }
  }

  // reorder the unordered vertices and check segment length against tolerance for edge collapse.
  // Only do this for overlaps with 3 or more vertices. We skip any overlap that degenerates to <3 vertices
  if ( numPolyVert > 2 ) {
    // order the unordered vertices (in counter clockwise fashion)
    int vertIdx[max_intersections];
    initIntArray( vertIdx, max_intersections, 0 );
    PolyReorderConvex( &polyXTemp[0], &polyYTemp[0], &vertIdx[0], numPolyVert );

    OverlapVertexType vertTypeTemp2[max_identified_points];
    int edgeATemp2[max_intersections];
    int edgeBTemp2[max_intersections];
    for ( int i = 0; i < numPolyVert; ++i ) {
      vertTypeTemp2[i] = vertTypeTemp[vertIdx[i]];
      edgeATemp2[i] = edgeATemp[vertIdx[i]];
      edgeBTemp2[i] = edgeBTemp[vertIdx[i]];
    }

    // check length of segs against tolerance and collapse short segments if necessary
    // This is where polyX and polyY get allocated for any overlap that remains with
    // > 3 vertices
    int numFinalVert = 0;

    FaceGeomException segErr =
        CheckPolySegs( polyXTemp, polyYTemp, numPolyVert, lenTol, polyX, polyY, vertIdx, numFinalVert );
    for ( int i = 0; i < numFinalVert; ++i ) {
      if ( vertType ) {
        vertType[i] = vertTypeTemp2[vertIdx[i]];
      }
      if ( edgeA ) {
        edgeA[i] = edgeATemp2[vertIdx[i]];
      }
      if ( edgeB ) {
        edgeB[i] = edgeBTemp2[vertIdx[i]];
      }
    }

    numPolyVert = numFinalVert;

    // check for an error in the segment check routine
    if ( segErr != 0 ) {
      return segErr;
    }

    // check to see if the overlap was degenerated to have 2 or less vertices.
    if ( numFinalVert < 3 ) {
      numPolyVert = 0;
      area = 0.0;
      return NO_OVERLAP;  // punt on degenerated or collapsed overlaps
    }
  } else {
    numPolyVert = 0;
    area = 0.0;
    return NO_OVERLAP;  // don't return error here. We should tolerate 'collapsed' (zero area) overlaps
  }

  // compute the area of the polygon
  area = Area2DPolygon( polyX, polyY, numPolyVert );

  return NO_FACE_GEOM_EXCEPTION;
}

#ifdef TRIBOL_USE_ENZYME

/*!
 *
 * \brief Enzyme wrapper for Intersection2DPolygon
 *
 * \param [in] xA array of local x coordinates of polygon A
 * \param [in] yA array of local y coordinates of polygon A
 * \param [in] numVertexA number of vertices in polygon A
 * \param [in] xB array of local x coordinates of polygon B
 * \param [in] yB array of local y coordinates of polygon B
 * \param [in] numVertexB number of vertices in polygon B
 * \param [in] posTol position tolerance to collapse segment-segment intersection points
 * \param [in] lenTol length tolerance to collapse short intersection edges
 * \param [out] polyX array of x coordinates of intersection polygon
 * \param [out] polyY array of y coordinates of intersection polygon
 * \param [out] numPolyVert number of vertices in intersection polygon
 *
 * \return 0 if no exception, >0 a face geom exception
 *
 * \pre length(xA), length(yA) >= numVertexA
 * \pre length(xB), length(yB) >= numVertexB
 *
 * \note polyX and polyY must be pre-allocated and sized to the maximum number
 * of points for the intersection polygon
 *
 */
inline FaceGeomException Intersection2DPolygonEnzyme( const RealT* xA, const RealT* yA, int numVertexA, const RealT* xB,
                                                      const RealT* yB, int numVertexB, RealT posTol, RealT lenTol,
                                                      RealT* polyX, RealT* polyY, int* numPolyVert )
{
  double area = 0.0;
  constexpr bool orientCheck = true;
  return Intersection2DPolygon( xA, yA, numVertexA, xB, yB, numVertexB, posTol, lenTol, polyX, polyY, *numPolyVert,
                                area, orientCheck );
}

#endif

/*!
 *
 * \brief computes the segment overlap between two linear edges projected onto the same 2D plane
 *
 * \param [in] pX1 x-coordinates of edge 1 as projected onto a common plane
 * \param [in] pY1 y-coordinates of edge 1 as projected onto a common plane
 * \param [in] pX2 x-coordinates of edge 2 as projected onto a common plane
 * \param [in] pY2 y-coordinates of edge 2 as projected onto a common plane
 * \param [in] nV1 number of vertices on edge 1
 * \param [in] nV2 number of vertices on edge 2
 * \param [out] overlapX pointer to x coordinates of overlapping segment
 * \param [out] overlapY pointer to y coordinates of overlapping segment
 * \param [out] area overlap area/length
 *
 * \return 0 if no exception, >0 if a geom exception
 *
 * \pre project each edge to a common 2D plane
 */
TRIBOL_HOST_DEVICE FaceGeomException CheckSegOverlap( const RealT* const pX1, const RealT* const pY1,
                                                      const RealT* const pX2, const RealT* const pY2, const int nV1,
                                                      const int nV2, RealT* overlapX, RealT* overlapY, RealT& area );

/*!
 * \brief computes the area of a triangle given 3D vertex coordinates
 *
 * \param [in] x array of x coordinates of the three vertices
 * \param [in] y array of y coordinates of the three vertices
 * \param [in] z array of z coordinates of the three vertices
 *
 * \return area of triangle
 *
 */
TRIBOL_HOST_DEVICE RealT Area3DTri( const RealT* const x, const RealT* const y, const RealT* const z );

/*!
 *
 * \brief reverses ordering of polygon vertices
 *
 * \param [in,out] x array of local x vertex coordinates
 * \param [in,out] y array of local y vertex coordinates
 * \param [in] numPoints number of vertices
 *
 * \pre length(x), length(y) >= numPoints
 *
 */
TRIBOL_HOST_DEVICE inline void ElemReverse( RealT* const x, RealT* const y, const int numPoints )
{
  constexpr int max_nodes_per_elem = 4;
  RealT xtemp[max_nodes_per_elem];
  RealT ytemp[max_nodes_per_elem];
  for ( int i = 0; i < numPoints; ++i ) {
    xtemp[i] = x[i];
    ytemp[i] = y[i];
  }

  int k = 1;
  for ( int i = ( numPoints - 1 ); i > 0; --i ) {
    x[k] = xtemp[i];
    y[k] = ytemp[i];
    ++k;
  }
}

/*!
 *
 * \brief reorders, if necessary, an ordered set of polygonal vertices such that the
 *        ordering obeys the right hand rule per the provided normal
 *
 * \param [in,out] x array of global x vertex coordinates
 * \param [in,out] y array of global y vertex coordinates
 * \param [in,out] z array of global z vertex coordinates
 * \param [in] numPoints number of vertices
 * \param [in] nX x-component of the polygon's normal
 * \param [in] nY y-component of the polygon's normal
 * \param [in] nZ z-component of the polygon's normal
 *
 * \pre length(x), length(y), length(z) >= numPoints
 *
 */
TRIBOL_HOST_DEVICE void PolyReorderWithNormal( RealT* const x, RealT* const y, RealT* const z, const int numPoints,
                                               const RealT nX, const RealT nY, const RealT nZ );

/*!
 * \brief computes the intersection point between a line and plane
 *
 * \param[in] xA x-coordinate of segment's first vertex
 * \param[in] yA y-coordinate of segment's first vertex
 * \param[in] zA z-coordinate of segment's first vertex
 * \param[in] xB x-coordinate of segment's second vertex
 * \param[in] yB y-coordinate of segment's second vertex
 * \param[in] zB z-coordinate of segment's second vertex
 * \param[in] xP x-coordinate of reference point on plane
 * \param[in] yP y-coordinate of reference point on plane
 * \param[in] zP z-coordinate of reference point on plane
 * \param[in] nX x-component of plane's unit normal
 * \param[in] nY y-component of plane's unit normal
 * \param[in] nZ z-component of plane's unit normal
 * \param[out] x x-coordinate of intersection point
 * \param[out] y y-coordinate of intersection point
 * \param[out] z z-coordinate of intersection point
 * \param[out] isParallel true if segment lies parallel to plane
 *
 * \note isParallel is true if the line is parallel, but not in plane, or if it is parallel and in plane
 *
 */
TRIBOL_HOST_DEVICE bool LinePlaneIntersection( const RealT xA, const RealT yA, const RealT zA, const RealT xB,
                                               const RealT yB, const RealT zB, const RealT xP, const RealT yP,
                                               const RealT zP, const RealT nX, const RealT nY, const RealT nZ, RealT& x,
                                               RealT& y, RealT& z, bool& isParallel );

/*!
 * \brief computes the line segment that is the intersection between two
 *        planes.
 *
 * \param[in] x1 x-coordinate of reference point on plane 1
 * \param[in] y1 y-coordinate of reference point on plane 1
 * \param[in] z1 z-coordinate of reference point on plane 1
 * \param[in] x2 x-coordinate of reference point on plane 2
 * \param[in] y2 y-coordinate of reference point on plane 2
 * \param[in] z2 z-coordinate of reference point on plane 2
 * \param[in] nX1 x-component of plane 1's unit normal
 * \param[in] nY1 y-component of plane 1's unit normal
 * \param[in] nZ1 z-component of plane 1's unit normal
 * \param[in] nX2 x-component of plane 2's unit normal
 * \param[in] nY2 y-component of plane 2's unit normal
 * \param[in] nZ2 z-component of plane 2's unit normal
 * \param[out] x x-component of point on intersection line
 * \param[out] y y-component of point on intersection line
 * \param[out] z z-component of point on intersection line
 *
 * \note the line segment is described by the output point (x,y,z), which
 * locates the segment vector, which is n1 x n2, where n1 is the
 * unit normal of plane 1 and n2 is the unit normal of plane 2.
 * Internal to this routine, this point is the intersection of a third
 * plane where a point on this plane is described in terms of a linear
 * combination of the two original plane's unit normals. This coordinate
 * description is plugged into the equation describing each of the two
 * original planes and the two parameters scaling the third plane's
 * linear combination are solved for. The three planes intersect at
 * this point, which is along the line segment of the line of intersection
 * of the first two planes.  Where this point lies on the intersection line
 * segment is controlled by each plane's input reference points.
 *
 */
bool PlanePlaneIntersection( const RealT x1, const RealT y1, const RealT z1, const RealT x2, const RealT y2,
                             const RealT z2, const RealT nX1, const RealT nY1, const RealT nZ1, const RealT nX2,
                             const RealT nY2, const RealT nZ2, RealT& x, RealT& y, RealT& z );

/*!
 *
 * \brief reverses order of 2D vertex coordinates to counter-clockwise orientation
 *
 * \param [in] x x-component coordinates
 * \param [in] y y-component coordinates
 * \param [out] xTemp reordered x-component coordinates
 * \param [out] yTemp reordered y-component coordinates
 * \param [in] numVert number of vertices
 *
 * \pre this routine assumes that the original coordinates are in clockwise ordering
 *
 */
void Vertex2DOrderToCCW( const RealT* const x, const RealT* const y, RealT* xTemp, RealT* yTemp, const int numVert );

/*!
 *
 * \brief Converts a set of planar 3D vertex coordinates to 2D
 *
 * \param [in] x pointer to x-component coordinates
 * \param [in] y pointer to y-component coordinates
 * \param [in] z pointer to z-component coordinates
 * \param [in] nx x-component of plane normal
 * \param [in] ny y-component of plane normal
 * \param [in] nz z-component of plane normal
 * \param [in] cx x-component of plane centroid
 * \param [in] cy y-component of plane centroid
 * \param [in] cz z-component of plane centroid
 * \param [in] num_verts number of vertices in polygon
 * \param [out] x_loc pointer to local x-coordinates
 * \param [out] y_loc pointer to local y-coordinates
 *
 * \pre x_loc and y_loc point to pre-allocated memory of length num_verts
 *
 * \note the local basis used in this routine is from ComputeLocalBasis() using the same point-normal data
 *       passed to this routine
 *
 */
TRIBOL_HOST_DEVICE void Points3DTo2D( const RealT* const x, const RealT* const y, const RealT* const z, const RealT nx,
                                      const RealT ny, const RealT nz, const RealT cx, const RealT cy, const RealT cz,
                                      const int num_verts, RealT* x_loc, RealT* y_loc );

/*!
 *
 * \brief Checks if the given point lies inside an edge
 *
 * \param [in] x to x-component coordinates of the edge's two vertices
 * \param [in] y to y-component coordinates of the edge's two vertices
 * \param [in] xp x-coordinate of the point in question
 * \param [in] yp y-coordinate of the point in question
 * \param [in] fuzz_factor percent of edge length to include in query
 *
 * \return true if the point lies inside the edge (or coincident with edge vertices).
 *
 * \pre (xp,yp) to be coliniear with edge defined by (x,y)
 *
 * \note the fuzz_factor is 0.0 by default, which will not include vertices that lie just outside the edge up to
 *       some fuzz. If a user wants a proximity query, they can increase the fuzz factor.
 *
 */
TRIBOL_HOST_DEVICE bool IsPointInEdge( const RealT* const x, const RealT* const y, RealT xp, RealT yp,
                                       RealT fuzz_factor = 0.0 );

/*!
 * \brief Check whether two polygons (faces) have a positive area of overlap
 *
 * \note Wrapper routine that calls the polygon intersection routine. That routine
 *  does not return vertices, just overlap area. This is the FULL overlap calculation.
 *
 * \param [in] num_nodes_1 number of nodes on first polygon
 * \param [in] num_nodes_2 number of nodes on second polygon
 * \param [in] projLocX1 2D x-coordinates of projected element 1 vertices
 * \param [in] projLocY1 2D y-coordinates of projected element 1 vertices
 * \param [in] projLocX2 2D x-coordinates of projected element 2 vertices
 * \param [in] projLocY2 2D y-coordinates of projected element 2 vertices
 * \param [out] area area of overlap
 * \param [in] isym 0 for planar symmetry, 1 for axial symmetry
 */
TRIBOL_HOST_DEVICE void CheckPolyOverlap( const int num_nodes_1, const int num_nodes_2, RealT* projLocX1,
                                          RealT* projLocY1, RealT* projLocX2, RealT* projLocY2, RealT& area,
                                          const int isym );

/*!
 * \brief This routine is used to check to see if two faces/edges overlap
 *        as projected to a (d-1) - dimensional hyperplane
 *
 * \param [in] x1 pointer to x coords for first face
 * \param [in] y1 pointer to y coords for first face
 * \param [in] z1 pointer to z coords for first face
 * \param [in] x1 pointer to x coords for second face
 * \param [in] y1 pointer to y coords for second face
 * \param [in] z1 pointer to z coords for second face
 * \param [in] n pointer to the contact plane normal
 * \param [in] c pointer to the contact plane centroid
 * \param [in] numNodesFace1 number of nodes on face 1 (4 maximum)
 * \param [in] numNodesFace2 number of nodes on face 2 (4 maximum)
 * \param [in] dim problem dimension
 *
 * \return true if the faces/edges overlap; otherwise false
 *
 * \note this routine is only to check whether two edges or faces overlap as projected onto a
 *       common or intermediate contact plane (i.e. a (d-1) - dimensional hyperplane.
 */
TRIBOL_HOST_DEVICE bool IsOverlappingOnPlane( const RealT* const x1, const RealT* const y1, const RealT* const z1,
                                              const RealT* const x2, const RealT* const y2, const RealT* const z2,
                                              const RealT* const n, const RealT* const c, const int numNodesFace1,
                                              const int numNodesFace2, const int dim );

/*!
 * \brief check if the planar polygon is convex
 *
 * \param [in] x array of local x coordinates of polygon vertices
 * \param [in] y array of local y coordinates of polygon vertices
 * \param [in] numPolyVert number of polygon vertices
 *
 * \return true if convex, false otherwise
 *
 * \note this routine does not check for self-intersecting polygons
 *
 */
TRIBOL_HOST_DEVICE bool IsConvex( const RealT* const x, const RealT* const y, const int numPolyVert );

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_GEOMUTILITIES_HPP_ */
