#include <iostream>
#include <cmath>

double magnitude( double const vx, double const vy )
{
   return std::sqrt(vx * vx + vy * vy);
}

bool SegmentIntersection2D( double xA1, double yA1, double xB1, double yB1,
                            double xA2, double yA2, double xB2, double yB2,
                            const bool* interior, double& x, double& y, 
                            bool& duplicate, double tol )
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
   double lambdaX1 = xB1 - xA1;
   double lambdaY1 = yB1 - yA1;

   double lambdaX2 = xB2 - xA2;
   double lambdaY2 = yB2 - yA2;

   double seg1Mag = magnitude( lambdaX1, lambdaY1 );
   double seg2Mag = magnitude( lambdaX2, lambdaY2 );

   // compute determinant of the lambda matrix, [ -lx1 -ly1, lx2 ly2 ]
   double det = -lambdaX1 * lambdaY2 + lambdaX2 * lambdaY1;

   // return false if det = 0. Check for numerically zero determinant
   double detTol = 1.E-12;
   if (det > -detTol && det < detTol)
   {
      x = 0.;
      y = 0.;
      duplicate = false;
      return false;
   }

   // compute intersection
   double invDet = 1.0 / det;
   double rX = xA1 - xA2;
   double rY = yA1 - yA2;
   double tA = invDet * (rX * lambdaY2 - rY * lambdaX2);
   double tB = invDet * (rX * lambdaY1 - rY * lambdaX1);

   // if tA and tB don't lie between [0,1] then return false.
   if ((tA < 0. || tA > 1.) || (tB < 0. || tB > 1.))
   {
      // no intersection
      x = 0.;
      y = 0.;
      duplicate = false;
      return false;
   }

   // if we get here then it means we have an intersection point.
   // Find the minimum distance of the intersection point to any of the segment 
   // vertices. 
   x = xA1 + lambdaX1 * tA;
   y = yA1 + lambdaY1 * tA;

   // for convenience, define an array of pointers that point to the 
   // input coordinates
   double xVert[4];
   double yVert[4];
   
   xVert[0] = xA1;
   xVert[1] = xB1;
   xVert[2] = xA2;
   xVert[3] = xB2;

   yVert[0] = yA1;
   yVert[1] = yB1;
   yVert[2] = yA2;
   yVert[3] = yB2;

   double distX[4];
   double distY[4];
   double distMag[4];

   for (int i=0; i<4; ++i)
   {
      distX[i] = x - xVert[i];
      distY[i] = y - yVert[i];
      distMag[i] = magnitude( distX[i], distY[i] );
   }

   double distMin = (seg1Mag > seg2Mag) ? seg1Mag: seg2Mag;
   int idMin;
   double xMinVert;
   double yMinVert;

   for (int i=0; i<4; ++i)
   {
      if (distMag[i] < distMin)
      {
         distMin = distMag[i];
         idMin = i;
         xMinVert = xVert[i];
         yMinVert = yVert[i];
      }
   }

   // check to see if the minimum distance is less than the position tolerance for 
   // the segments
   double distRatio = (idMin == 0 || idMin == 1) ? (distMin / seg1Mag) : (distMin / seg2Mag);

   // if the distRatio is less than the tolerance, or percentage cutoff of the original 
   // segment that we would like to keep, then check to see if the segment vertex closest 
   // to the computed intersection point is an interior point. If this is true, then collapse
   // the computed intersection point to the interior point and mark the duplicate boolean.
   // Also do this for the argument, interior, set to nullptr
   if (distRatio < tol)
   {
      if (interior == nullptr || interior[idMin])
      {
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

bool VertexAvgCentroid( const double* const x, 
                        const double* const y, 
                        const double* const z, 
                        const int numVert,
                        double& cX, double& cY, double& cZ )
{
   if (numVert == 0)
   {
      return false;
   }

   // (re)initialize the input/output centroid components
   cX = 0.0;
   cY = 0.0;
   cZ = 0.0;

   // loop over nodes adding the position components
   double fac = 1.0 / numVert;
   for (int i=0; i<numVert; ++i) {
      cX += x[i];
      cY += y[i];
      if (z != nullptr)
      {
         cZ += z[i]; 
      }
   }

   // divide by the number of nodes to compute average
   cX *= fac;
   cY *= fac;
   cZ *= fac;

   return true;

}

bool CheckPolyOrientation( const double* const x, 
                           const double* const y, 
                           const int numVertex )
{
   bool check = true;
   for (int i=0; i<numVertex; ++i)
   {
      // determine vertex indices of the segment
      int ia = i;
      int ib = (i == (numVertex-1)) ? 0 : (i+1);

      // compute segment vector
      double lambdaX = x[ib] - x[ia];
      double lambdaY = y[ib] - y[ia];
    
      // determine segment normal 
      double nrmlx = -lambdaY;
      double nrmly = lambdaX;

      // compute vertex-averaged centroid
      double* z = nullptr;
      double xc, yc, zc;
      VertexAvgCentroid( x, y, z, numVertex, xc, yc, zc );

      // compute vector between centroid and first vertex of current segment
      double vx = xc - x[ia];
      double vy = yc - y[ia];

      // compute dot product between segment normal and centroid-to-vertex vector.
      // the normal points inward toward the centroid
      double prod = vx * nrmlx + vy * nrmly;

      if (prod < 0.) // don't keep checking
      {
         check = false;
         return check;
      }
   }
   return check; // should equal true if here.

}

bool Point2DInTri( const double xp, const double yp, 
                   const double* const xTri, 
                   const double* const yTri )
{
   bool inside = false;

   // compute coordinate basis between the 1-2 and 1-3 vertices
   double e1x = xTri[1] - xTri[0];
   double e1y = yTri[1] - yTri[0];

   double e2x = xTri[2] - xTri[0];
   double e2y = yTri[2] - yTri[0];

   // compute vector components of vector between point and first vertex
   double p1x = xp - xTri[0];
   double p1y = yp - yTri[0];

   // compute dot products (e1,e1), (e1,e2), (e2,e2), (p1,e1), and (p1,e2)
   double e11 =  e1x * e1x + e1y * e1y;
   double e12 =  e1x * e2x + e1y * e2y;
   double e22 =  e2x * e2x + e2y * e2y;
   double p1e1 = p1x * e1x + p1y * e1y;
   double p1e2 = p1x * e2x + p1y * e2y;

   // compute the inverse determinant
   double invDet = 1.0 / (e11 * e22 - e12 * e12);

   // compute 2 local barycentric coordinates
   double u = invDet * (e22 * p1e1 - e12 * p1e2);
   double v = invDet * (e11 * p1e2 - e12 * p1e1); 
   
   // u or v may be negative, but numerically zero. Address this
   u = (std::abs(u) < 1.e-12) ? 0.0 : u;
   v = (std::abs(v) < 1.e-12) ? 0.0 : v;

   if ((u >= 0) && (v >= 0) && (u + v <= 1))
   {
      inside = true;
   }

   return inside;

}

bool Point2DInFace( const double xPoint, const double yPoint, 
                    const double* const xPoly, 
                    const double* const yPoly,
                    const double xC, const double yC, 
                    const int numPolyVert )
{

   // if face is triangle (numPolyVert), call Point2DInTri once
   if (numPolyVert == 3)
   {
      return Point2DInTri( xPoint, yPoint, xPoly, yPoly );
   }

   // loop over triangles and determine if point is inside
   bool tri = false;
   for (int i=0; i<numPolyVert; ++i)
   {

      double xTri[3];
      double yTri[3];

      // construct polygon using i^th segment vertices and face centroid
      xTri[0] = xPoly[i];
      yTri[0] = yPoly[i];

      xTri[1] = (i == (numPolyVert-1)) ? xPoly[0] : xPoly[i+1];
      yTri[1] = (i == (numPolyVert-1)) ? yPoly[0] : yPoly[i+1];

      // last vertex of the triangle is the vertex averaged centroid of the polygonal face
      xTri[2] = xC;
      yTri[2] = yC;

      // call Point2DInTri for each triangle
      tri = Point2DInTri( xPoint, yPoint, xTri, yTri );

      if (tri) 
      {
         return true;
      }
   }
   return false;

}

double Area2DPolygon( const double* const x, 
                      const double* const y, 
                      const int numPolyVert )
{

   double area = 0.;

   // compute vertex-averaged centroid to construct a triangle between segment 
   // vertices and centroid
   double* z = nullptr;
   double xc, yc, zc;
   VertexAvgCentroid(x, y, z, numPolyVert, xc, yc, zc);

   for (int i=0; i<numPolyVert; ++i)
   {
      // determine vertex indices of the segment
      int ia = i;
      int ib = (i == (numPolyVert-1)) ? 0 : (i+1);

      area += std::abs( 0.5 * (x[ia]*(y[ib]-yc) + 
                               x[ib]*(yc-y[ia]) + 
                               xc*(y[ia]-y[ib])));
   }
   return area;

}

enum class OverlapVertexType
{
  A,
  B,
  EdgeEdge
};

bool PolyReorder( double* x, double* y, OverlapVertexType* vertType, int numPoints )
{

   if (numPoints<3)
   {
#if defined(TRIBOL_USE_HOST) && !defined(TRIBOL_USE_ENZYME)
      SLIC_DEBUG("PolyReorder: numPoints (" << numPoints << ") < 3.");
#endif
      return false;
   }

   double xC, yC, zC;
   double * z = nullptr;
   constexpr int max_nodes_per_overlap = 8 + 2*4;
   double proj [max_nodes_per_overlap - 2];

   int newIDs[ max_nodes_per_overlap ];

   // initialize newIDs array to local ordering, 0,1,2,...,numPoints-1
   for (int i=0; i<numPoints; ++i)
   {
      newIDs[i] = i;
   }

   // compute vertex averaged centroid, in local coordinates
   VertexAvgCentroid( x, y, z, numPoints, xC, yC, zC );

   // using the first index into the x,y vertex coordinate arrays as 
   // the first vertex of the soon-to-be ordered list of vertices, determine 
   // the next vertex that will comprise the first segment in a counter 
   // clockwise ordering of vertices
   int id1 = -1;
   int id0 = 0;
   newIDs[0] = id0;
  
   for (int j=1; j<numPoints; ++j)
   {
      // determine segment vector and normal
      double lambdaX = x[j] - x[id0];
      double lambdaY = y[j] - y[id0];
      double nrmlx = -lambdaY;
      double nrmly = lambdaX;

      // project vectors that span from each point, except j,k, to first vertex (id0), onto the 
      // segment normal. There will always be numPoints-2 projections
      int pk = 0;
      for (int k=0; k<numPoints; ++k)
      {
         if (k != id0 && k != j)
         {
            proj[pk] = (x[k]-x[id0]) * nrmlx + (y[k]-y[id0]) * nrmly;
            ++pk;
         }
      }

      // check if all points are on one side of line defined by segment
      // (pk at this point should be equal to numPoints - 2)
      bool neg = false;
      bool pos = false;
      for (int ip=0; ip<pk; ++ip)
      {
         if (neg)
         {
            neg = true;
         }
         else if (!neg)
         {
            neg = (proj[ip] < 0.) ? true : false;
         }

         if (pos)
         {
            pos = true;
         }
         else if (!pos)
         {
            pos = (proj[ip] > 0.) ? true : false;
         }

         if (neg && pos)
         {
            break;
         }
      }

      // if one of the booleans is false then all points are on one side 
      // of line defined by i-j segment.
      if (!neg || !pos)
      {
         // check the orientation of the nodes to make sure we have the correct 
         // one of two segments that will pass the previous test.
         // Check the dot product between the normal and the vector
         // between the centroid and first (0th) vertex
         double vx = xC - x[id0];
         double vy = yC - y[id0];

         double prod = nrmlx * vx + nrmly * vy;

         // check if the two vertices are a segment on the convex hull and oriented CCW.
         // CCW orientation has prod > 0
         if (prod > 0) 
         {
            id1 = j;  
            break;
         }
      }

   } // end loop over j

   // swap ids
   if (id1 != -1)
   {
      newIDs[1] = id1;
      newIDs[id1] = 1;
   }

   // given the first (current) reference segment, compute the link vector between the jth vertex 
   // (j cannot be a vertex belonging to the reference segment) and the first vertex of 
   // the given reference segment. The next reference segment is between the second vertex of 
   // the current reference segment and the jth vertex whose link vector has the smallest 
   // dot product with the current reference segment.

   for (int i=0; i<(numPoints-3); ++i) // increment to (numPoints - 3) or (numPoints - 2)?
   {
      int jID;
      double cosThetaMax = -1.; // this handles angles up to 180 degrees. Not possible for convex polygons
      double cosTheta;
      double refMag, linkMag;

      // compute reference vector;
      double refx, refy;
      refx = x[newIDs[i+1]] - x[newIDs[i]];
      refy = y[newIDs[i+1]] - y[newIDs[i]];
      refMag = magnitude( refx, refy );

//      SLIC_ERROR_IF(refMag < 1.E-12, "PolyReorder: reference segment for link vector check is nearly zero length");

      // loop over link vectors of unassigned vertices
      int nextVertexID = 2+i;
      for (int j=nextVertexID; j<numPoints; ++j)
      {
         double lx, ly;

         lx = x[newIDs[j]] - x[newIDs[i]];
         ly = y[newIDs[j]] - y[newIDs[i]];
         linkMag = magnitude( lx, ly );

         cosTheta = ( lx * refx + ly * refy ) / (refMag * linkMag);
         if (cosTheta > cosThetaMax)
         {
            cosThetaMax = cosTheta;
            jID = j;
         } 
           
      } // end loop over j

      // we have found the minimum angle and the corresponding local vertex id.
      // swap ids
      int swapID = newIDs[ nextVertexID ];
      newIDs[ nextVertexID ] = newIDs[jID];
      newIDs[jID] = swapID;

   } // end loop over i

   // reorder x and y coordinate arrays based on newIDs id-array
   double xtemp[ max_nodes_per_overlap ];
   double ytemp[ max_nodes_per_overlap ];
   OverlapVertexType vertTypeTemp[ max_nodes_per_overlap ];
   for (int i=0; i<numPoints; ++i)
   {
      xtemp[i] = x[i];
      ytemp[i] = y[i];
      vertTypeTemp[i] = vertType[i];
   }

   for (int i=0; i<numPoints; ++i)
   {
      x[i] = xtemp[ newIDs[i] ];
      y[i] = ytemp[ newIDs[i] ];
      if (vertType)
      {
         vertType[i] = vertTypeTemp[ newIDs[i] ];
      }
   }

   return true;

}

enum FaceGeomError
{
   NO_FACE_GEOM_ERROR,                         ///! No face geometry error
   FACE_ORIENTATION,                           ///! Face vertices not ordered consistent with outward unit normal
   INVALID_FACE_INPUT,                         ///! Invalid input
   DEGENERATE_OVERLAP,                         ///! Issues with overlap calculation resulting in degenerate overlap
   FACE_VERTEX_INDEX_EXCEEDS_OVERLAP_VERTICES, ///! Very specific debug indexing error where face vertex count exceeds overlap vertex count in cg routine
   NUM_FACE_GEOM_ERRORS
};

FaceGeomError CheckPolySegs( const double* x, const double* y,
                             const OverlapVertexType* vertType,
                             int numPoints, double tol, 
                             double* xnew, double* ynew,
                             OverlapVertexType* vertTypeNew,
                             int& numNewPoints )
{
   constexpr int max_nodes_per_overlap = 8;
   double newIDs[ max_nodes_per_overlap ];

   // set newIDs[i] to original local ordering
   for (int i=0; i<numPoints; ++i)
   {
      newIDs[i] = i;
   }

   for (int i=0; i<numPoints; ++i)
   {
      // determine vertex indices of the segment
      int ia = i;
      int ib = (i == (numPoints-1)) ? 0 : (i+1);

      // compute segment vector magnitude
      double lambdaX = x[ib] - x[ia];
      double lambdaY = y[ib] - y[ia];
      double lambdaMag = magnitude( lambdaX, lambdaY );
     
      // check segment length against tolerance
      if (lambdaMag < tol)
      {
         // collapse second vertex to the first vertex of the current segment
         newIDs[ib] = i;
      }
   }

   // determine the number of new points
   numNewPoints = 0;
   for (int i=0; i<numPoints; ++i)
   {
      if (newIDs[i] == i)
      {
         ++numNewPoints;
      }
   }

   // check to make sure numNewPoints >= 3 for valid overlap polygons prior
   // to memory allocation
   if (numNewPoints < 3)
   {
      // return and degenerated polygon will be skipped over. 
      return NO_FACE_GEOM_ERROR;
   }
   
   // set the coordinates in xnew and ynew 
   int k = 0;
   for (int i=0; i<numPoints; ++i)
   {
      if (newIDs[i] == i)
      {
         if (k > numNewPoints)
         {
#if defined(TRIBOL_USE_HOST) && !defined(TRIBOL_USE_ENZYME)
            SLIC_DEBUG("checkPolySegs(): index into polyX/polyY exceeds allocated space");
#endif
            return FACE_VERTEX_INDEX_EXCEEDS_OVERLAP_VERTICES;
         }

         xnew[k] = x[i];
         ynew[k] = y[i];
         if (vertTypeNew)
         {
            vertTypeNew[k] = vertType[i];
         }
         ++k;
      }
   }

   return NO_FACE_GEOM_ERROR;

}

void initRealArray( double * arr, int length, double init_val )
{
   for (int i=0; i<length; ++i)
   {
      arr[i] = init_val;
   }
} 

void initIntArray( int * arr, int length, int init_val )
{
   for (int i=0; i<length; ++i)
   {
      arr[i] = init_val;
   }
}

void initBoolArray( bool * arr, int length, bool init_val )
{
   for (int i=0; i<length; ++i)
   {
      arr[i] = init_val;
   }
} 

FaceGeomError Intersection2DPolygon( const double* const xA, 
                                     const double* const yA, 
                                     const int numVertexA, 
                                     const double* const xB, 
                                     const double* const yB, 
                                     const int numVertexB,
                                     double posTol, double lenTol, 
                                     double* polyX, 
                                     double* polyY,
                                     OverlapVertexType* vertType,
                                     int& numPolyVert, double& area, bool orientCheck )
{
   // for tribol, if you have called this routine it is because a positive area of 
   // overlap between two polygons (faces) exists. This routine does not perform a 
   // "proximity" check to determine if the faces are "close enough" to proceed with 
   // the full calculation. This can and probably should be added.

   // check numVertexA and numVertexB to make sure they are 3 (triangle) or more
   if (numVertexA < 3 || numVertexB < 3) 
   {
#if defined(TRIBOL_USE_HOST) && !defined(TRIBOL_USE_ENZYME)
      SLIC_DEBUG( "Intersection2DPolygon(): one or more degenerate faces with < 3 vertices." );
#endif
      area = 0.0;
      return INVALID_FACE_INPUT; 
   }

   // check right hand rule ordering of polygon vertices. 
   // Note 1: This check is consistent with the ordering that comes from PolyReorder() 
   // of two faces with unordered vertices. 
   // Note 2: Intersection2DPolygon doesn't require consistent face vertex orientation
   // between faces, as long as each are 'ordered' (CW or CCW).
   if (orientCheck)
   {
      bool orientA = CheckPolyOrientation( xA, yA, numVertexA );
      bool orientB = CheckPolyOrientation( xB, yB, numVertexB );

      if (!orientA || !orientB)
      {
#if defined(TRIBOL_USE_HOST) && !defined(TRIBOL_USE_ENZYME)
         SLIC_DEBUG( "Intersection2DPolygon(): check face orientations for face A." );
#endif
         return FACE_ORIENTATION;
      }
   }

   // maximum number of vertices (for use later)
   constexpr int max_nodes_per_element = 4;

   // allocate an array to hold ids of interior vertices
   int interiorVAId[ max_nodes_per_element ];
   int interiorVBId[ max_nodes_per_element ];

   // initialize all entries in interior vertex array to -1
   initIntArray( &interiorVAId[0], numVertexA, -1 );
   initIntArray( &interiorVBId[0], numVertexB, -1 );

   // precompute the vertex averaged centroids for both polygons. 
   double xCA = 0.0;
   double yCA = 0.0;
   double xCB = 0.0;
   double yCB = 0.0;
   double zC = 0.0; // not required, only used as dummy argument in centroid routine

   VertexAvgCentroid( xA, yA, nullptr, numVertexA, xCA, yCA, zC );
   VertexAvgCentroid( xB, yB, nullptr, numVertexB, xCB, yCB, zC );

   // check to see if any of polygon A's vertices are in polygon B, and vice-versa. Track
   // which vertices are interior to the other polygon. Keep in mind that vertex 
   // coordinates are local 2D coordinates.
   int numVAI = 0;
   int numVBI = 0;

   // check A in B
   for (int i=0; i<numVertexA; ++i)
   {
      if (Point2DInFace( xA[i], yA[i], xB, yB, xCB, yCB, numVertexB ))
      {
         // interior A in B
         interiorVAId[i] = i;
         ++numVAI; 
      }
   }

   // check to see if ALL of A is in B; then A is the overlapping polygon.
   if (numVAI == numVertexA)
   {
      numPolyVert = numVertexA;
      for (int i=0; i<numVertexA; ++i)
      {
         polyX[i] = xA[i];
         polyY[i] = yA[i];
         if (vertType)
         {
            vertType[i] = OverlapVertexType::A;
         }
      }
      area = Area2DPolygon( polyX, polyY, numVertexA );
      return NO_FACE_GEOM_ERROR;
   }

   // check B in A
   for (int i=0; i<numVertexB; ++i) 
   {
      if (Point2DInFace( xB[i], yB[i], xA, yA, xCA, yCA, numVertexA) )
      {
         // interior B in A
         interiorVBId[i] = i;
         ++numVBI;
      }
   }

   // check to see if ALL of B is in A; then B is the overlapping polygon.
   if (numVBI == numVertexB)
   {
      numPolyVert = numVertexB;
      for (int i=0; i<numVertexB; ++i)
      {
         polyX[i] = xB[i];
         polyY[i] = yB[i];
         if (vertType)
         {
            vertType[i] = OverlapVertexType::B;
         }
      }
      area = Area2DPolygon( polyX, polyY, numVertexB );
      return NO_FACE_GEOM_ERROR;
   }

   // check for coincident interior vertices. That is, a vertex on A interior to 
   // B occupies the same point in space as a vertex on B interior to A. This is 
   // O(n^2), but the number of interior vertices is anticipated to be small
   // if we are at this location in the routine
   for (int i=0; i<numVertexA; ++i)
   {
      if (interiorVAId[i] != -1)
      {
         for (int j=0; j<numVertexB; ++j)
         {
            if (interiorVBId[j] != -1)
            {
              // compute the distance between interior vertices
              double distX = xA[i] - xB[j];
              double distY = yA[i] - yB[j];
              double distMag = magnitude( distX, distY );
              if (distMag < 1.E-15)
              {
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
   constexpr int max_intersections = max_nodes_per_element*max_nodes_per_element;
   double interX[ max_intersections ];
   double interY[ max_intersections ];
   bool intersect[ max_intersections ];
   bool dupl; // boolean to indicate a segment-segment intersection that 
              // duplicates an existing interior vertex.
   bool interior [4];

   // initialize the interX and interY entries
   initRealArray( interX, max_intersections, 0. );
   initRealArray( interY, max_intersections, 0. );
   initBoolArray( intersect, max_intersections, false );
   dupl = false;

   // loop over segment-segment intersections to find the rest of the 
   // intersecting vertices. This is O(n^2), but segments defined by two 
   // nodes interior to the other polygon will be skipped. This will catch 
   // outlier cases.
   int interId = 0;

   // loop over A segments
   for (int ia=0; ia<numVertexA; ++ia)
   {
      int vAID1 = ia;
      int vAID2 = (ia == (numVertexA-1)) ? 0 : (ia + 1);
      
      // set boolean indicating which nodes on segment A are interior
      interior[0] = (interiorVAId[vAID1] != -1) ? true : false; 
      interior[1] = (interiorVAId[vAID2] != -1) ? true : false;
//      bool checkA = (interior[0] == -1 && interior[1] == -1) ? true : false;
      bool checkA = true;

      // loop over B segments
      for (int jb=0; jb<numVertexB; ++jb)
      {
         int vBID1 = jb;
         int vBID2 = (jb == (numVertexB-1)) ? 0 : (jb + 1);
         interior[2] = (interiorVBId[vBID1] != -1) ? true : false; 
         interior[3] = (interiorVBId[vBID2] != -1) ? true : false;
//         bool checkB = (interior[2] == -1 && interior[3] == -1) ? true : false;
         bool checkB = true;

         // if both segments are not defined by nodes interior to the other polygon
         if (checkA && checkB) 
         {
            if (interId >= max_intersections) 
            {
#if defined(TRIBOL_USE_HOST) && !defined(TRIBOL_USE_ENZYME)
               SLIC_DEBUG("Intersection2DPolygon: number of segment/segment intersections exceeds precomputed maximum; " << 
                          "check for degenerate overlap.");
#endif
               return DEGENERATE_OVERLAP;
            }

            intersect[interId] = SegmentIntersection2D( xA[vAID1], yA[vAID1], xA[vAID2], yA[vAID2],
                                                        xB[vBID1], yB[vBID1], xB[vBID2], yB[vBID2],
                                                        interior, interX[interId], interY[interId], 
                                                        dupl, posTol );
            ++interId;
         }
      } // end loop over A segments
   }  // end loop over B segments

   // count the number of segment-segment intersections
   int numSegInter = 0;
   for (int i=0; i<interId; ++i)
   {
      if (intersect[i]) ++numSegInter; 
   }

   // add check for case where there are no interior vertices or 
   // intersection vertices
   if (numSegInter == 0 && numVBI == 0 && numVAI == 0)
   {
      area = 0.0;
      return NO_FACE_GEOM_ERROR;
   }

   // allocate temp intersection polygon vertex coordinate arrays to consist 
   // of segment-segment intersections and number of interior points in A and B
   numPolyVert = numSegInter + numVAI + numVBI;
   // maximum number of vertices between the two polygons.  assumes convex elements.
   constexpr int max_nodes_per_overlap = 2*max_nodes_per_element;
   constexpr int max_identified_points = max_nodes_per_overlap + 2*max_nodes_per_element;
   double polyXTemp[ max_identified_points ];
   double polyYTemp[ max_identified_points ];
   OverlapVertexType vertTypeTemp[ max_identified_points ];

   // fill polyXTemp and polyYTemp with the intersection points
   int k = 0;
   for (int i=0; i<interId; ++i) 
   {
      if (intersect[i]) 
      {
         polyXTemp[k] = interX[i];
         polyYTemp[k] = interY[i];
         vertTypeTemp[k] = OverlapVertexType::EdgeEdge;
         ++k;
      }
   }

   // fill polyX and polyY with the vertices on A that lie in B
   for (int i=0; i<numVertexA; ++i)
   {
      if (interiorVAId[i] != -1)
      {
         // debug
         if (k > max_identified_points)
         {
#if defined(TRIBOL_USE_HOST) && !defined(TRIBOL_USE_ENZYME)
            SLIC_DEBUG("Intersection2DPolygon(): number of A vertices interior to B " << 
                       "polygon exceeds total number of overlap vertices. Check interior vertex id values.");
#endif
            return FACE_VERTEX_INDEX_EXCEEDS_OVERLAP_VERTICES;
         }

         polyXTemp[k] = xA[i];
         polyYTemp[k] = yA[i];
         vertTypeTemp[k] = OverlapVertexType::A;
         ++k;
      }
   }

   for (int i=0; i<numVertexB; ++i)
   {
      if (interiorVBId[i] != -1)
      {
         // debug
         if (k > max_identified_points)
         {
            return FACE_VERTEX_INDEX_EXCEEDS_OVERLAP_VERTICES;
         }

         polyXTemp[k] = xB[i];
         polyYTemp[k] = yB[i];
         vertTypeTemp[k] = OverlapVertexType::B;
         ++k;
      }
   }

   // reorder the unordered vertices and check segment length against tolerance for edge collapse.
   // Only do this for overlaps with 3 or more vertices. We skip any overlap that degenerates to <3 vertices
   if (numPolyVert>2)
   {
      // order the unordered vertices (in counter clockwise fashion)
      PolyReorder( polyXTemp, polyYTemp, vertTypeTemp, numPolyVert );

      // check length of segs against tolerance and collapse short segments if necessary
      // This is where polyX and polyY get allocated for any overlap that remains with 
      // > 3 vertices
      int numFinalVert = 0; 

      FaceGeomError segErr = CheckPolySegs( polyXTemp, polyYTemp, vertTypeTemp, numPolyVert, 
                                            lenTol, polyX, polyY, vertType, numFinalVert );

      // check for an error in the segment check routine
      if (segErr != 0)
      {
         return segErr;
      }
 
      // check to see if the overlap was degenerated to have 2 or less vertices.
      if (numFinalVert<3)
      {
         area = 0.0;
         return NO_FACE_GEOM_ERROR; // punt on degenerated or collapsed overlaps
      }

      numPolyVert = numFinalVert;
   }
   else
   {
      area = 0.0;
      return NO_FACE_GEOM_ERROR; // don't return error here. We should tolerate 'collapsed' (zero area) overlaps
   }

   // compute the area of the polygon
   area = Area2DPolygon( polyX, polyY, numPolyVert );

   return NO_FACE_GEOM_ERROR;

}

template <typename return_type, typename... Args>
return_type __enzyme_fwddiff(Args...);

extern int enzyme_dup;
extern int enzyme_const;

bool dSegmentIntersection2D(double xA1, double yA1, double xB1, double yB1,
                            double xA2, double yA2, double xB2, double yB2,
                            const bool* interior, double& x, double& y, 
                            bool& duplicate, double tol) {
    double dxdxvert{0.0};
    double dydxvert{0.0};
    auto is_intersected = __enzyme_fwddiff<bool>((void*)SegmentIntersection2D,
        enzyme_dup, xA1, 1.0,
        enzyme_const, yA1,
        enzyme_const, xB1,
        enzyme_const, yB1,
        enzyme_const, xA2,
        enzyme_const, yA2,
        enzyme_const, xB2,
        enzyme_const, yB2,
        enzyme_const, interior,
        enzyme_dup, &x, &dxdxvert,
        enzyme_dup, &y, &dydxvert,
        enzyme_const, &duplicate,
        enzyme_const, tol);
    std::cout << "Intersection point = (" << x << ", " << y << ")" << std::endl;
    std::cout << "dxdxA1 = (" << dxdxvert << ", " << dydxvert << ")" << std::endl;
    __enzyme_fwddiff<bool>((void*)SegmentIntersection2D,
        enzyme_const, xA1,
        enzyme_dup, yA1, 1.0,
        enzyme_const, xB1,
        enzyme_const, yB1,
        enzyme_const, xA2,
        enzyme_const, yA2,
        enzyme_const, xB2,
        enzyme_const, yB2,
        enzyme_const, interior,
        enzyme_dup, &x, &dxdxvert,
        enzyme_dup, &y, &dydxvert,
        enzyme_const, &duplicate,
        enzyme_const, tol);
    std::cout << "dxdyA1 = (" << dxdxvert << ", " << dydxvert << ")" << std::endl;
    __enzyme_fwddiff<bool>((void*)SegmentIntersection2D,
        enzyme_const, xA1,
        enzyme_const, yA1,
        enzyme_dup, xB1, 1.0,
        enzyme_const, yB1,
        enzyme_const, xA2,
        enzyme_const, yA2,
        enzyme_const, xB2,
        enzyme_const, yB2,
        enzyme_const, interior,
        enzyme_dup, &x, &dxdxvert,
        enzyme_dup, &y, &dydxvert,
        enzyme_const, &duplicate,
        enzyme_const, tol);
    std::cout << "dxdxB1 = (" << dxdxvert << ", " << dydxvert << ")" << std::endl;
    __enzyme_fwddiff<bool>((void*)SegmentIntersection2D,
        enzyme_const, xA1,
        enzyme_const, yA1,
        enzyme_const, xB1,
        enzyme_dup, yB1, 1.0,
        enzyme_const, xA2,
        enzyme_const, yA2,
        enzyme_const, xB2,
        enzyme_const, yB2,
        enzyme_const, interior,
        enzyme_dup, &x, &dxdxvert,
        enzyme_dup, &y, &dydxvert,
        enzyme_const, &duplicate,
        enzyme_const, tol);
    std::cout << "dxdyB1 = (" << dxdxvert << ", " << dydxvert << ")" << std::endl;
    __enzyme_fwddiff<bool>((void*)SegmentIntersection2D,
        enzyme_const, xA1,
        enzyme_const, yA1,
        enzyme_const, xB1,
        enzyme_const, yB1,
        enzyme_dup, xA2, 1.0,
        enzyme_const, yA2,
        enzyme_const, xB2,
        enzyme_const, yB2,
        enzyme_const, interior,
        enzyme_dup, &x, &dxdxvert,
        enzyme_dup, &y, &dydxvert,
        enzyme_const, &duplicate,
        enzyme_const, tol);
    std::cout << "dxdxA2 = (" << dxdxvert << ", " << dydxvert << ")" << std::endl;
    __enzyme_fwddiff<bool>((void*)SegmentIntersection2D,
        enzyme_const, xA1,
        enzyme_const, yA1,
        enzyme_const, xB1,
        enzyme_const, yB1,
        enzyme_const, xA2,
        enzyme_dup, yA2, 1.0,
        enzyme_const, xB2,
        enzyme_const, yB2,
        enzyme_const, interior,
        enzyme_dup, &x, &dxdxvert,
        enzyme_dup, &y, &dydxvert,
        enzyme_const, &duplicate,
        enzyme_const, tol);
    std::cout << "dxdyA2 = (" << dxdxvert << ", " << dydxvert << ")" << std::endl;
    __enzyme_fwddiff<bool>((void*)SegmentIntersection2D,
        enzyme_const, xA1,
        enzyme_const, yA1,
        enzyme_const, xB1,
        enzyme_const, yB1,
        enzyme_const, xA2,
        enzyme_const, yA2,
        enzyme_dup, xB2, 1.0,
        enzyme_const, yB2,
        enzyme_const, interior,
        enzyme_dup, &x, &dxdxvert,
        enzyme_dup, &y, &dydxvert,
        enzyme_const, &duplicate,
        enzyme_const, tol);
    std::cout << "dxdxB2 = (" << dxdxvert << ", " << dydxvert << ")" << std::endl;
    __enzyme_fwddiff<bool>((void*)SegmentIntersection2D,
        enzyme_const, xA1,
        enzyme_const, yA1,
        enzyme_const, xB1,
        enzyme_const, yB1,
        enzyme_const, xA2,
        enzyme_const, yA2,
        enzyme_const, xB2,
        enzyme_dup, yB2, 1.0,
        enzyme_const, interior,
        enzyme_dup, &x, &dxdxvert,
        enzyme_dup, &y, &dydxvert,
        enzyme_const, &duplicate,
        enzyme_const, tol);
    std::cout << "dxdyB2 = (" << dxdxvert << ", " << dydxvert << ")" << std::endl;
    return is_intersected;
}

int main() {

    constexpr auto delta_ = 1.0e-8;
    constexpr auto pos_tol = 10.0 * delta_;
    constexpr auto len_tol = 10.0 * delta_; 
    constexpr auto offset = 0.2;
    // shift node 3 in a little to prevent edge class change with FD
    double x1[8] = { 0.0, 1.0, 1.0 - pos_tol, 0.0,
                    0.0, 0.0, 1.0 - pos_tol, 1.0 };
    double x2[8] = { 0.0, 1.0, 1.0, 0.0,
                    0.0-offset, 0.0+offset, 1.0, 1.0 };
    double xi[16];
    initRealArray(xi, 16, 0.0);
    OverlapVertexType type[8];
    auto num_poly_verts = 0;
    double area = 0.0;
    double x_dot[4] = {0.0, 0.0, 0.0, 0.0};
    double dxidx1[16*8];
    double dxidx2[16*8];
    for (int i{0}; i < 16*8; ++i)
    {
      dxidx1[i] = 0.0;
      dxidx2[i] = 0.0;
    }

    for (int i{0}; i < 4; ++i)
    {
      x_dot[i] = 1.0;
      __enzyme_fwddiff<void>((void*)Intersection2DPolygon,
        enzyme_dup, x1, x_dot,
        enzyme_const, x1 + 4,
        enzyme_const, 4,
        enzyme_const, x2,
        enzyme_const, x2 + 4,
        enzyme_const, 4,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx1 + 16*i,
        enzyme_dup, xi + 8, dxidx1 + 16*i + 8,
        enzyme_const, type,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true
      );
      __enzyme_fwddiff<void>((void*)Intersection2DPolygon,
        enzyme_const, x1,
        enzyme_dup, x1 + 4, x_dot,
        enzyme_const, 4,
        enzyme_const, x2,
        enzyme_const, x2 + 4,
        enzyme_const, 4,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx1 + 16*(4 + i),
        enzyme_dup, xi + 8, dxidx1 + 16*(4 + i) + 8,
        enzyme_const, type,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true
      );
      __enzyme_fwddiff<void>((void*)Intersection2DPolygon,
        enzyme_const, x1,
        enzyme_const, x1 + 4,
        enzyme_const, 4,
        enzyme_dup, x2, x_dot,
        enzyme_const, x2 + 4,
        enzyme_const, 4,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx2 + 16*i,
        enzyme_dup, xi + 8, dxidx2 + 16*i + 8,
        enzyme_const, type,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true
      );
      __enzyme_fwddiff<void>((void*)Intersection2DPolygon,
        enzyme_const, x1,
        enzyme_const, x1 + 4,
        enzyme_const, 4,
        enzyme_const, x2,
        enzyme_dup, x2 + 4, x_dot,
        enzyme_const, 4,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx2 + 16*(4 + i),
        enzyme_dup, xi + 8, dxidx2 + 16*(4 + i) + 8,
        enzyme_const, type,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true
      );
      x_dot[i] = 0.0;
    }

    std::cout << "dxi/dx1 nonzero values:" << std::endl;
    for (int j{0}; j < 8; ++j)
    {
        for (int i{0}; i < 16; ++i)
        {
            auto idx = j*16 + i;
            if (std::abs(dxidx1[idx]) > 1.0e-15)
            {
                std::cout << "  (" << i << ", " << j << ") = " << dxidx1[idx] << std::endl;
            }
        }
    }

    std::cout << "dxi/dx2 nonzero values:" << std::endl;
    for (int j{0}; j < 8; ++j)
    {
        for (int i{0}; i < 16; ++i)
        {
            auto idx = j*16 + i;
            if (std::abs(dxidx2[idx]) > 1.0e-15)
            {
                std::cout << "  (" << i << ", " << j << ") = " << dxidx2[idx] << std::endl;
            }
        }
    }
}
