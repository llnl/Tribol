// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_GEOM_GEOMUTILITIES_HPP_
#define SRC_GEOM_GEOMUTILITIES_HPP_

#include "tribol/types.hpp"
#include "tribol/common/Parameters.hpp"

namespace tribol {

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
 * \param [in,out] px x coordinate of projected point
 * \param [in,out] py y coordinate of projected point
 * \param [in,out] pz z coordinate of projected point
 *
 */
void ProjectPointToPlane( const real x, const real y, const real z, const real nx, const real ny, const real nz,
                          const real ox, const real oy, const real oz, real& px, real& py, real& pz );

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
 * \param [in,out] px x coordinate of projected point
 * \param [in,out] py y coordinate of projected point
 *
 */
void ProjectPointToSegment( const real x, const real y, const real nx, const real ny, const real ox, const real oy,
                            real& px, real& py );

/*!
 *
 * \brief Method to find the intersection area between two polygons and
 *  the local y-coordinate of the centroid
 * \note taken from Ben Liu's implementation of
 *  Eric Herbold's routine in ALE3D.
 *
 * \param [in] namax number of vertices in polygon a
 * \param [in] xa array of local x coordinates of polygon a vertices
 * \param [in] ya array of local y coordinates of polygon b vertices
 * \param [in] nbmax number of vertices in polygon b
 * \param [in] xb array of local x coordintes of polygon b
 * \param [in] yb array of local y coordinates of polygon b
 * \param [in] isym 0 for planar symmetry, 1 for axial symmetry
 * \param [in,out] area intersection polygon's area
 * \param [in,out] ycent local y centroid coordinate
 * \pre length(xa), length(ya) >= namax
 * \pre length(xb), length(yb) >= nbmax
 *
 * \note method to determine area of overlap of two polygons and local
 *  centroid y-coordinate. Swap input (xa,ya)->(ya,xa) and (xb,yb)->(yb,xb)
 *  to get centroid x-coordinate.
 */
void PolyInterYCentroid( const int namax, const real* const RESTRICT xa, const real* const RESTRICT ya, const int nbmax,
                         const real* const RESTRICT xb, const real* const RESTRICT yb, const int isym, real& area,
                         real& ycent );

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
 * \param [in,out] xg global x coordinate
 * \param [in,out] yg global y coordinate
 * \param [in,out] zg global z coordinate
 *
 * \note this is used to convert a point on a plane in a local
 *  2D coordinate basis to a point in the 3D global coordinate system
 *
 */
void Local2DToGlobalCoords( real xloc, real yloc, real e1X, real e1Y, real e1Z, real e2X, real e2Y, real e2Z, real cX,
                            real cY, real cZ, real& xg, real& yg, real& zg );

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
 * \param [in,out] pLX array of local x coordinates of input points
 * \param [in,out] pLY array of local y coordinates of input points
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
void GlobalTo2DLocalCoords( const real* const RESTRICT pX, const real* const RESTRICT pY, const real* const RESTRICT pZ,
                            real e1X, real e1Y, real e1Z, real e2X, real e2Y, real e2Z, real cX, real cY, real cZ,
                            real* const RESTRICT pLX, real* const RESTRICT pLY, int size );

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
 * \param [in,out] pLX local x coordinate of input point
 * \param [in,out] pLY local y coordinate of input point
 *
 * \note this assumes that the point lies in the plane defined by the
 *  2D local basis vectors.
 */
void GlobalTo2DLocalCoords( real pX, real pY, real pZ, real e1X, real e1Y, real e1Z, real e2X, real e2Y, real e2Z,
                            real cX, real cY, real cZ, real& pLX, real& pLY );
/*!
 *
 * \brief computes the vertex averaged centroid of a point set
 *
 * \param [in] x array of x coordinates for point set
 * \param [in] y array of y coordinates for point set
 * \param [in] z array of z coordinates for point set
 * \param [in] numVert number of points in point set
 * \param [in,out] cX x coordinate of vertex averaged centroid
 * \param [in,out] cY y coordinate of vertex averaged centroid
 * \param [in,out] cZ z coordinate of vertex averaged centroid
 *
 * \pre length(x) >= numVert
 * \pre length(y) >= numVert
 * \pre length(z) >= numVert
 *
 */
void VertexAvgCentroid( const real* const RESTRICT x, const real* const RESTRICT y, const real* const RESTRICT z,
                        const int numVert, real& cX, real& cY, real& cZ );

/*!
 *
 * \brief computes the vertex averaged centroid of a point set
 *
 * \param [in] x array of stacked coordinates for point set
 * \param [in] dim 2D or 3D coordinate dimension
 * \param [in] numVert number of points in point set
 * \param [in,out] cX x coordinate of vertex averaged centroid
 * \param [in,out] cY y coordinate of vertex averaged centroid
 * \param [in,out] cZ z coordinate of vertex averaged centroid
 *
 * \pre length(x) >= numVert
 *
 */
void VertexAvgCentroid( const real* const RESTRICT x, const int dim, const int numVert, real& cX, real& cY, real& cZ );

/*!
 *
 * \brief computes the area centroid of a polygon
 *
 * \param [in] x array of stacked coordinates for point set
 * \param [in] dim 2D or 3D coordinate dimension
 * \param [in] numVert number of points in point set
 * \param [in,out] cX x coordinate of vertex averaged centroid
 * \param [in,out] cY y coordinate of vertex averaged centroid
 * \param [in,out] cZ z coordinate of vertex averaged centroid
 *
 * \pre length(x) >= numVert
 *
 */
void PolyAreaCentroid( const real* const RESTRICT x, const int dim, const int numVert, real& cX, real& cY, real& cZ );

/*!
 *
 * \brief computes the centroid of the polygon
 *
 * \param [in] x array of x-coordinates for point set
 * \param [in] y array of y-coordinates for point set
 * \param [in] numVert number of points in point set
 * \param [in,out] cX x coordinate of vertex averaged centroid
 * \param [in,out] cY y coordinate of vertex averaged centroid
 *
 * \pre length(x) >= numVert
 *
 */
void PolyCentroid( const real* const RESTRICT x, const real* const RESTRICT y, const int numVert, real& cX, real& cY );

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
 * \param [in,out] polyX array of x coordinates of intersection polygon
 * \param [in,out] polyY array of y coordinates of intersection polygon
 * \param [in,out] numPolyVert number of vertices in intersection polygon
 * \param [in,out] area intersection polygon area
 *
 * \return 0 if no error, >0 a face geom error
 *
 * \pre length(xA), length(yA) >= numVertexA
 * \pre length(xB), length(yB) >= numVertexB
 *
 * \note this routine will allocate new dynamic memory for the polyX and polyY arrays
 *  as the length of these arrays is not known a priori. One must take care to delete
 *  this memory when necessary.
 *
 */
FaceGeomError Intersection2DPolygon( const real* const RESTRICT xA, const real* const RESTRICT yA, const int numVertexA,
                                     const real* const RESTRICT xB, const real* const RESTRICT yB, const int numVertexB,
                                     real posTol, real lenTol, real* RESTRICT* RESTRICT polyX,
                                     real* RESTRICT* RESTRICT polyY, int& numPolyVert, real& area,
                                     bool orientCheck = true );

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
bool CheckPolyOrientation( const real* const RESTRICT x, const real* const RESTRICT y, const int numVertex );

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
bool Point2DInFace( const real xPoint, const real yPoint, const real* const RESTRICT xPoly,
                    const real* const RESTRICT yPoly, const real xC, const real yC, const int numPolyVert );

/*!
 *
 * \brief check to see if a point in a local 2D coordinate system lies in a triangle
 *
 * \param [in] xp local x coordinate of point
 * \param [in] yp local y coordinate of point
 * \param [in] xTri array of local x coordinates of triangle
 * \param [in] yTri array of local y coordinates of triangle
 *
 * \return true if the point is inside the triangle, false otherwise
 *
 * \pre length(xTri), length(yTri) >= 3
 *
 * \note this routine finds the two barycentric coordinates of the triangle and
 *  determines if those coordinates are inside or out
 *  (http://blackpawn.com/texts/pointinpoly/default.html);
 */
bool Point2DInTri( const real xp, const real yp, const real* const RESTRICT xTri, const real* const RESTRICT yTri );

/*!
 * \brief computes the area of a polygon
 *
 * \param [in] x array of local x coordinates of polygon vertices
 * \param [in] y array of local y coordinates of polygon vertices
 * \param [in] numPolyVert number of polygon vertices
 *
 * \return area of polygon
 *
 * \note breaks the polygon into triangles and sums the areas of the triangles
 */
real Area2DPolygon( const real* const RESTRICT x, const real* const RESTRICT y, const int numPolyVert );

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
real Area3DTri( const real* const RESTRICT x, const real* const RESTRICT y, const real* const RESTRICT z );

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
 * \param [in,out] x local coordinate of the intersection point
 * \param [in,out] y local coordinate of the intersection point
 * \param [in,out] duplicate true if intersection point is computed as duplicate polygon
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
bool SegmentIntersection2D( const real xA1, const real yA1, const real xB1, const real yB1, const real xA2,
                            const real yA2, const real xB2, const real yB2, const bool* const RESTRICT interior,
                            real& x, real& y, bool& duplicate, const real tol );

/*!
 *
 * \brief checks polygon segments to collapse any segments less than input tolerance
 *
 * \param [in] x array of local x coordinates of polygon vertices
 * \param [in] y array of local y coordinates of polygon vertices
 * \param [in] numPoints number of polygon vertices
 * \param [in] tol edge segment tolerance
 * \param [in,out] xnew array of new x coordinates
 * \param [in,out] ynew array of new y coordinates
 * \param [in,out] numNewPoints number of new points
 *
 * \return 0 if no error, >0 a face geom error
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
FaceGeomError CheckPolySegs( const real* const RESTRICT x, const real* const RESTRICT y, const int numPoints,
                             const real tol, real* RESTRICT* RESTRICT xnew, real* RESTRICT* RESTRICT ynew,
                             int& numNewPoints );

/*!
 *
 * \brief reorders a set of unordered vertices associated with a star convex polygon in
 *        counter clockwise orientation
 *
 * \param [in,out] x array of local x vertex coordinates
 * \param [in,out] y array of local y vertex coordinates
 * \param [in] numPoints number of vertices
 *
 * \pre length(x), length(y) >= numPoints
 *
 * \note This routine takes the unordered set of vertex coordinates of a star convex
 *  polygon and orders the vertices in counter-clockwise orientation.
 */
void PolyReorder( real* const RESTRICT x, real* const RESTRICT y, const int numPoints );

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
void PolyReverse( real* const RESTRICT x, real* const RESTRICT y, const int numPoints );

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
void PolyReorderWithNormal( real* const RESTRICT x, real* const RESTRICT y, real* const RESTRICT z, const int numPoints,
                            const real nX, const real nY, const real nZ );

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
 * \param[in,out] x x-coordainte of intersection point
 * \param[in,out] y y-coordinate of intersection point
 * \param[in,out] z z-coordainte of intersection point
 * \param[in,out] inPlane true if segment lies in the plane
 *
 */
bool LinePlaneIntersection( const real xA, const real yA, const real zA, const real xB, const real yB, const real zB,
                            const real xP, const real yP, const real zP, const real nX, const real nY, const real nZ,
                            real& x, real& y, real& z, bool& inPlane );

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
 * \param[in,out] x x-component of point on intersection line
 * \param[in,out] y y-component of point on intersection line
 * \param[in,out] z z-component of point on intersection line
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
bool PlanePlaneIntersection( const real x1, const real y1, const real z1, const real x2, const real y2, const real z2,
                             const real nX1, const real nY1, const real nZ1, const real nX2, const real nY2,
                             const real nZ2, real& x, real& y, real& z );

/*!
 *
 * \brief reverses order of 2D vertex coordinates to counter-clockwise orientation
 *
 * \param [in] x x-component coordinates
 * \param [in] y y-component coordinates
 * \param [in,out] xTemp reordered x-component coordinates
 * \param [in,out] yTemp reordered y-component coordinates
 * \param [in] numVert number of vertices
 *
 * \pre this routine assumes that the original coordinates are in clockwise ordering
 *
 */
void Vertex2DOrderToCCW( const real* const RESTRICT x, const real* const RESTRICT y, real* RESTRICT xTemp,
                         real* RESTRICT yTemp, const int numVert );

}  // namespace tribol

#endif /* SRC_GEOM_GEOMUTILITIES_HPP_ */
