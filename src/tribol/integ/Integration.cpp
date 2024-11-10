// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// tribol includes
#include "Integration.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/utils/Math.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/geom/GeomUtilities.hpp"

// axom includes
#include "axom/slic.hpp" 

// C++ includes
#include <cmath>

namespace tribol
{

template< >
TRIBOL_HOST_DEVICE void EvalWeakFormIntegral< COMMON_PLANE, SINGLE_POINT >
                         ( SurfaceContactElem const & elem,
                           RealT * const integ1,
                           RealT * const integ2 )
{
   // compute the area centroid of the overlap polygon,
   // which serves as the single integration point
   RealT cx[3] = {0., 0., 0.};
   if (elem.dim == 2)
   {
      VertexAvgCentroid( elem.overlapCoords, elem.dim, elem.numPolyVert,
                         cx[0], cx[1], cx[2] );
   } else {
      PolyAreaCentroid( elem.overlapCoords, elem.dim, elem.numPolyVert,
                        cx[0], cx[1], cx[2] );
   }

   ///////////////////////////////////////////////
   //
   //project overlap polygon centroid to each face
   //
   ///////////////////////////////////////////////
   RealT cxProj1[3]  = { 0., 0., 0. }; // overlap centroid projected to face 1
   RealT cxProj2[3]  = { 0., 0., 0. }; // overlap centroid projected to face 2
   RealT cxf1[3] = { 0., 0., 0. }; // vertex avg. centroid of face 1
   RealT cxf2[3] = { 0., 0., 0. }; // vertex avg. centroid of face 2

   // compute vertex averaged centroid of each face for the point-normal data
   VertexAvgCentroid( elem.faceCoords1, elem.dim, elem.numFaceVert,
                      cxf1[0], cxf1[1], cxf1[2] );
   VertexAvgCentroid( elem.faceCoords2, elem.dim, elem.numFaceVert,
                      cxf2[0], cxf2[1], cxf2[2] );

   // project the overlap centroid to each face
   if (elem.dim == 3)
   {
      // TODO we should probably project in the direction of the overlap normal
      // As the faces become less coplanar we lose conservation of angular momentum
      ProjectPointToPlane( cx[0], cx[1], cx[2],
                           elem.faceNormal1[0],
                           elem.faceNormal1[1],
                           elem.faceNormal1[2],
                           cxf1[0], cxf1[1], cxf1[2],
                           cxProj1[0], cxProj1[1], cxProj1[2] ); 

      ProjectPointToPlane( cx[0], cx[1], cx[2],
                           elem.faceNormal2[0],
                           elem.faceNormal2[1],
                           elem.faceNormal2[2],
                           cxf2[0], cxf2[1], cxf2[2],
                           cxProj2[0], cxProj2[1], cxProj2[2] ); 
   } 
   else
   {
      // TODO we should probably project in the direction of the overlap normal
      // As the faces become less coplanar we lose conservation of angular momentum
      ProjectPointToSegment( cx[0], cx[1], 
                             elem.faceNormal1[0],
                             elem.faceNormal1[1],
                             cxf1[0], cxf1[1],
                             cxProj1[0], cxProj1[1] );
      ProjectPointToSegment( cx[0], cx[1], 
                             elem.faceNormal2[0],
                             elem.faceNormal2[1],
                             cxf2[0], cxf2[1],
                             cxProj2[0], cxProj2[1] );
   }

   // loop over nodes and compute nodal force integral 
   // contributions
   for (int a=0; a<elem.numFaceVert; ++a)
   {

      EvalBasis( elem.faceCoords1, cxProj1[0], cxProj1[1], cxProj1[2],
                 elem.numFaceVert, a, integ1[a] );
      EvalBasis( elem.faceCoords2, cxProj2[0], cxProj2[1], cxProj2[2],
                 elem.numFaceVert, a, integ2[a] );
   }

   return;
}

//------------------------------------------------------------------------------
void TWBPolyInt( SurfaceContactElem const & elem,
                 IntegPts & integ,
                 int k )
{
   // check that the order, k, is either 2 or 3
   if (k != 2 && k !=3)
   {
      SLIC_ERROR("TWBPolyInt: input argument, k, must be 2 or 3.");
      return;
   }

   // determine number of TWB integration points for current overlap
   int numTotalPoints, numTriPoints;
   numTotalPoints = NumTWBPointsPoly( elem, k );
   numTriPoints   = NumTWBPointsPerTri( k );

   integ.initialize( 3, numTotalPoints );

   // declare local array to hold barycentric coordinates for each 
   // triangle
   RealT bary[ elem.dim * numTriPoints ];

   switch (k) 
   {
      case 2:

         for (int i=0; i<numTotalPoints; ++i)
         {
            integ.wts[i] = 0.6666666666667;
         }

         // first barycentric point
         bary[0] = 0.1666666666667;
         bary[1] = 0.6666666666667;
         bary[2] = 1. - bary[0] - bary[1];
         // second barycentric point
         bary[3] = 0.6666666666667;
         bary[4] = 0.1666666666667;
         bary[5] = 1. - bary[3] - bary[4];
         // third barycentric point
         bary[6] = 0.1666666666667;
         bary[7] = 0.1666666666667;
         bary[8] = 1. - bary[6] - bary[7];
         break;

      case 3:

         // populate first three wts for first triangle
         for (int i=0; i<3; ++i)
         {
            integ.wts[i] = 0.2199034873106;
         }

         // populate second three wts to complete first triangle
         for (int i=3; i<6; ++i)
         {
            integ.wts[i] = 0.4467631793560;
         }

         // reproduce first six wts for the rest of the triangles
         for (int i=1; i<elem.numPolyVert; ++i)
         {
            for (int j=0; j<6; ++j)
            {
               integ.wts[6*i+j] = integ.wts[j];
            }
         }

         // first barycentric point
         bary[0] = 0.0915762135098;
         bary[1] = bary[0];
         bary[2] = 1. - bary[0] - bary[1];
         // second barycentric point
         bary[3] = 0.8168475729805; 
         bary[4] = bary[0]; 
         bary[5] = 1. - bary[3] - bary[4]; 
         // third barycentric point
         bary[6] = bary[0]; 
         bary[7] = bary[3]; 
         bary[8] = 1. - bary[6] - bary[7]; 
         // fourth barycentric point
         bary[9] = 0.1081030181681;
         bary[10] = 0.4459484909160;
         bary[11] = 1. - bary[9] - bary[10];
         // fifth barycentric point
         bary[12] = bary[10];
         bary[13] = bary[9];
         bary[14] = 1. - bary[12] - bary[13];
         // sixth barycentric point
         bary[15] = bary[10];
         bary[16] = bary[10];
         bary[17] = 1. - bary[15] - bary[16];
         break;
   }

   // compute the vertex averaged centroid of the overlap polygon. Note 
   // that the coordinates of the overlap polygon are always assumed to be 
   // 3D
   RealT xc[elem.dim];
   for (int i=0; i<elem.dim; ++i)
   {
      xc[i] = 0.;
   }

   for (int i=0; i<elem.numPolyVert; ++i)
   {
      xc[0] += elem.overlapCoords[ elem.dim * i ];
      xc[1] += elem.overlapCoords[ elem.dim * i + 1 ];
      xc[2] += elem.overlapCoords[ elem.dim * i + 2 ];
   }

   xc[0] /= elem.numPolyVert;
   xc[1] /= elem.numPolyVert;
   xc[2] /= elem.numPolyVert;

   // populate the xy coordinate array of the 3D coordinates of the 
   // integration points based on the TWB barycenter data for the 
   // given integration order

   RealT vx, vy, vz, a, b, c, p, area;
   int kmax = elem.numPolyVert - 1;
   for (int k=0; k<kmax; ++k)
   {
      vx = elem.overlapCoords[ elem.dim * (k+1) ] -
           elem.overlapCoords[ elem.dim * k ];
      vy = elem.overlapCoords[ elem.dim * (k+1) + 1 ] - 
           elem.overlapCoords[ elem.dim * k + 1 ];
      vz = elem.overlapCoords[ elem.dim * (k+1) + 2 ] - 
           elem.overlapCoords[ elem.dim * k + 2 ];
      a  = magnitude( vx, vy, vz );
      vx = xc[0] - elem.overlapCoords[ elem.dim * k ];
      vy = xc[1] - elem.overlapCoords[ elem.dim * k + 1 ];
      vz = xc[2] - elem.overlapCoords[ elem.dim * k + 2 ];
      b = magnitude( vx, vy, vz );
      vx = xc[0] - elem.overlapCoords[ elem.dim * (k+1) ];
      vy = xc[1] - elem.overlapCoords[ elem.dim * (k+1) + 1 ];
      vz = xc[2] - elem.overlapCoords[ elem.dim * (k+1) + 2 ];
      c = magnitude( vx, vy, vz );
      p = 0.5 * (a + b + c);
      RealT area = sqrt(p * (p - a) * (p - b) * (p - c));

      for (int m=0; m<numTriPoints; ++m)
      {
         integ.wts[ numTriPoints * k + m ] *= 0.5 * area;
         integ.xy[ (elem.dim * numTriPoints) * k + (elem.dim * m) ] = 
                  bary[ elem.dim * m ] * elem.overlapCoords[ elem.dim * k ] + 
                  bary[ elem.dim * m + 1 ] * elem.overlapCoords[ elem.dim * (k+1) ] + 
                  bary[ elem.dim * m + 2 ] * xc[0];
         integ.xy[ (elem.dim * numTriPoints) * k + (elem.dim * m) + 1 ] = 
                  bary[ elem.dim * m ] * elem.overlapCoords[ elem.dim * k + 1 ] + 
                  bary[ elem.dim * m + 1 ] * elem.overlapCoords[ elem.dim * (k+1) + 1 ] + 
                  bary[ elem.dim * m + 2 ] * xc[1];
         integ.xy[ (elem.dim * numTriPoints) * k + (elem.dim * m) + 2 ] = 
                  bary[ elem.dim * m ] * elem.overlapCoords[ elem.dim * k + 2 ] + 
                  bary[ elem.dim * m + 1 ] * elem.overlapCoords[ elem.dim * (k+1) + 2 ] + 
                  bary[ elem.dim * m + 2 ] * xc[2];
      } // end loop over number of points per triangle
   } // end loop over (n-1) number of triangles

   // populate last triangle's integration point coordinates
   vx = elem.overlapCoords[ elem.dim * (elem.numPolyVert - 1) ] - 
        elem.overlapCoords[0];
   vy = elem.overlapCoords[ elem.dim * (elem.numPolyVert - 1) + 1 ] - 
        elem.overlapCoords[1];
   vz = elem.overlapCoords[ elem.dim * (elem.numPolyVert - 1) + 2 ] -
        elem.overlapCoords[2];
   a = magnitude(vx, vy, vz );
   vx = xc[0] - elem.overlapCoords[ elem.dim * (elem.numPolyVert - 1) ];
   vy = xc[1] - elem.overlapCoords[ elem.dim * (elem.numPolyVert - 1) + 1 ];
   vz = xc[2] - elem.overlapCoords[ elem.dim * (elem.numPolyVert - 1) + 2 ];
   b = magnitude(vx, vy, vz );
   vx = xc[0] - elem.overlapCoords[0];
   vy = xc[1] - elem.overlapCoords[1];
   vz = xc[2] - elem.overlapCoords[2];
   c = magnitude(vx, vy, vz );
   p = 0.5 * (a + b + c);
   area = sqrt(p * (p - a) * (p - b) * (p - c));

   for (int i=0; i<numTriPoints; ++i)
   {
      integ.wts[ numTriPoints * (elem.numPolyVert - 1) + i ] *= 0.5 * area;
      integ.xy[ elem.dim * numTriPoints * (elem.numPolyVert - 1) + (elem.dim * i) ] =
               bary[ elem.dim * i ] * elem.overlapCoords[ elem.dim * (elem.numPolyVert - 1) ] + 
               bary[ elem.dim * i + 1 ] * elem.overlapCoords[0] + 
               bary[ elem.dim * i + 2 ] * xc[0]; 
      integ.xy[ elem.dim * numTriPoints * (elem.numPolyVert - 1) + (elem.dim * i) + 1 ] =
               bary[ elem.dim * i ] * elem.overlapCoords[ elem.dim * (elem.numPolyVert - 1) + 1 ] + 
               bary[ elem.dim * i + 1 ] * elem.overlapCoords[1] + 
               bary[ elem.dim * i + 2 ] * xc[1]; 
      integ.xy[ elem.dim * numTriPoints * (elem.numPolyVert - 1) + (elem.dim * i) + 2 ] = 
               bary[ elem.dim * i ] * elem.overlapCoords[ elem.dim * (elem.numPolyVert - 1) + 2 ] + 
               bary[ elem.dim * i + 1 ] * elem.overlapCoords[2] + 
               bary[ elem.dim * i + 2 ] * xc[2]; 
   } // end loop over numTriPoints for last triangle
   return;
}

//------------------------------------------------------------------------------
int NumTWBPointsPoly( SurfaceContactElem const & elem,
                      int k )
{

   // get the number of integration points per triangle per integration rule 
   // order k
   int numPoints = NumTWBPointsPerTri( k ); 

   // note the number of triangles is the same as the number of 
   // vertices of the overlapping polygon.  
   return numPoints * elem.numPolyVert; 

}

//------------------------------------------------------------------------------
int NumTWBPointsPerTri( int order )
{
   switch (order)
   {
      case 2:
         return 3;
      case 3:
         return 6;
      default:
         SLIC_ERROR("NumTWBPoints: integration rule order not supported.");
         break;
   }

   return 0;
}

//------------------------------------------------------------------------------
void GaussPolyIntTri( SurfaceContactElem const & elem,
                      IntegPts & integ,
                      int k )
{
   // determine the number of integration points per triangle in the decomposed 
   // polygon and the total number of integration points on the polygon
   int numTriPoints, numTotalPoints;
   switch (k)
   {
      case 2:
         numTriPoints = 3;
         numTotalPoints = numTriPoints * elem.numPolyVert; 
         break;
      case 3:
         // don't do anything, default to case 4
      case 4:
         numTriPoints = 6;
         numTotalPoints = numTriPoints * elem.numPolyVert;
         break;
      default:
         SLIC_ERROR("GaussPolyIntTri: only Gauss integration of order 2-4 is implemented.");
         return;
   }

   int parentDim = 2;

   integ.initialize( 3, numTotalPoints );

   // populate wts array and set parent space coordinates of 
   // integration points on triangle
   RealT* coords;
   switch (k)
   {
      case 2:
         for (int i=0; i<numTotalPoints; ++i)
         {
            integ.wts[i] = 0.3333333333;
         }
         coords = new RealT[6];
         coords[0] = 0.1666666667;
         coords[1] = 0.1666666667;
         coords[2] = 0.6666666667;
         coords[3] = 0.1666666667;
         coords[4] = 0.1666666667;
         coords[5] = 0.6666666667;
         break;
      case 3:
      case 4:
         RealT wt1 = 0.109951743655322;
         RealT wt2 = 0.223381589678011;
         for (int i=0; i<elem.numPolyVert; ++i)
         {
            integ.wts[numTriPoints * i] = wt1;
            integ.wts[numTriPoints * i + 1] = wt1;
            integ.wts[numTriPoints * i + 2] = wt1;
            integ.wts[numTriPoints * i + 3] = wt2;
            integ.wts[numTriPoints * i + 4] = wt2;
            integ.wts[numTriPoints * i + 5] = wt2; 
         }
         RealT x1 = 0.091576213509771;
         RealT x2 = 0.816847572980459;
         RealT x3 = 0.108103018168070;
         RealT x4 = 0.445948490915965;
         coords = new RealT[12];
         coords[0]  = x1;
         coords[1]  = x1;
         coords[2]  = x2;
         coords[3]  = x1;
         coords[4]  = x1;
         coords[5]  = x2;
         coords[6]  = x3;
         coords[7]  = x4;
         coords[8]  = x4;
         coords[9]  = x3;
         coords[10] = x4;
         coords[11] = x4;
         break;
   }
 
   // compute area centroid of polygon
   RealT xTri[3] = { 0., 0., 0. };
   RealT yTri[3] = { 0., 0., 0. };
   RealT zTri[3] = { 0., 0., 0. };
   PolyAreaCentroid(elem.overlapCoords, elem.dim, elem.numPolyVert, 
                    xTri[2], yTri[2], zTri[2] );

   // populate xy array
   for (int j=0; j<elem.numPolyVert; ++j)
   {
      // group triangle coordinates
      int triId = j;
      int triIdPlusOne = (j == (elem.numPolyVert - 1)) ? 0 : triId+1;
      xTri[0] = elem.overlapCoords[ elem.dim * triId ];
      yTri[0] = elem.overlapCoords[ elem.dim * triId + 1 ];
      zTri[0] = elem.overlapCoords[ elem.dim * triId + 2 ];
      xTri[1] = elem.overlapCoords[ elem.dim * triIdPlusOne ];
      yTri[1] = elem.overlapCoords[ elem.dim * triIdPlusOne + 1 ];
      zTri[1] = elem.overlapCoords[ elem.dim * triIdPlusOne + 2 ];

      // compute area of triangle
      RealT area = Area3DTri( xTri, yTri, zTri );

      for (int k=0; k<numTriPoints; ++k)
      {
         // NOTE: Per Puso 2004, the sum over integration point 
         // evaluations per pallet are multiplied by the pallet area. 
         //
         // multiply the integration point weights by the 
         // triangle area (note: this is specific to how integrals 
         // are computed on polygonal overlaps for Contact)
         integ.wts[ numTriPoints * j + k ] *= area;

         // group parent space ip coordinates
         RealT xi[2];
         xi[0] = coords[parentDim * k];
         xi[1] = coords[parentDim * k + 1];

         // forward map parent space ip coords to physical space
         RealT x[3];
         FwdMapLinTri( xi, xTri, yTri, zTri, x );

         integ.xy[ ((integ.ipDim) * numTriPoints) * j + (integ.ipDim * k) ] = 
             x[0];
         integ.xy[ ((integ.ipDim) * numTriPoints) * j + (integ.ipDim * k) + 1 ] = 
             x[1];
         integ.xy[ ((integ.ipDim) * numTriPoints) * j + (integ.ipDim * k) + 2 ] = 
             x[2];
      } // end loop over number of ips per triangle
   } // end loop over triangles

   delete [] coords;
   
}

//------------------------------------------------------------------------------
void GaussPolyIntQuad( SurfaceContactElem const & TRIBOL_UNUSED_PARAM(elem),
                       IntegPts & integ,
                       int k )
{
   // determine the number of integration points per quad 
   int numQuadPoints;
   switch (k)
   {
      case 2:
         numQuadPoints = 4;
         break;
      case 3:
         numQuadPoints = 9;
         break;
      case 4:
         numQuadPoints = 16;
         break;
      case 5:
         numQuadPoints = 25;
         break;
      default:
         SLIC_ERROR("GaussPolyIntQuad: only Gauss integration of order 2-5 is implemented.");
         return;
   }

   int parentDim = 2;

   // initialize integration rule struct. Note, use the parentDim = 2 in the 
   // initialization as we want to house all 2D quadrature points in parent space. 
   // This is different than TWB or the triangular decomposition methods that 
   // house quadrature points in 3D physical space (the integration rule using 
   // the triangular decomposition of the polygon forward maps parent space 
   // IP coordinates to physical space. Only later do we inverse map those 
   // to the quad 4 of interest).
   integ.initialize( parentDim, numQuadPoints );

   // populate wts array and set parent space coordinates of 
   // integration points on triangle
   switch (k)
   {
      case 2:
      {
         for (int i=0; i<numQuadPoints; ++i)
         {
            integ.wts[i] = 1.;
         }
//         RealT inv_root_3 = 1./std::sqrt(3.);
         RealT inv_root_3 = 1./std::sqrt(3.);

         // integration points ordered counter-clockwise 
         integ.xy[0] = -inv_root_3;
         integ.xy[1] = -inv_root_3;
         integ.xy[2] =  inv_root_3;
         integ.xy[3] = -inv_root_3;
         integ.xy[4] =  inv_root_3;
         integ.xy[5] =  inv_root_3;
         integ.xy[6] = -inv_root_3;
         integ.xy[7] =  inv_root_3;
         break;
      }
      case 3:
      {
         RealT five_nine = 5./9.;
         RealT eight_nine = 8./9.;

         // integration points ordered left to right, bottom to top
         integ.wts[0] = five_nine * five_nine;
         integ.wts[1] = eight_nine * five_nine;
         integ.wts[2] = five_nine * five_nine;
         integ.wts[3] = five_nine * eight_nine;
         integ.wts[4] = eight_nine * eight_nine;
         integ.wts[5] = five_nine * eight_nine;
         integ.wts[6] = five_nine * five_nine; 
         integ.wts[7] = five_nine * eight_nine;
         integ.wts[8] = five_nine * five_nine;

         RealT x1 = std::sqrt(3./5.);
         RealT x2 = 0.;
         integ.xy[0]  = -x1; 
         integ.xy[1]  = -x1; 
         integ.xy[2]  =  x2; 
         integ.xy[3]  = -x1; 
         integ.xy[4]  =  x1; 
         integ.xy[5]  = -x1; 

         integ.xy[6]  = -x1; 
         integ.xy[7]  =  x2; 
         integ.xy[8]  =  x2; 
         integ.xy[9]  =  x2; 
         integ.xy[10] =  x1; 
         integ.xy[11] =  x2; 

         integ.xy[12] = -x1; 
         integ.xy[13] =  x1; 
         integ.xy[14] =  x2; 
         integ.xy[15] =  x1; 
         integ.xy[16] =  x1; 
         integ.xy[17] =  x1; 
         break;
      }
      case 4:
      {
         RealT wt1 = (18. - std::sqrt(30.)) / 36.;
         RealT wt2 = (18. + std::sqrt(30.)) / 36.;

         // integration points ordered bottom to top, left to right.
         // Note, this is different that the third order rule
         integ.wts[0]  = wt1 * wt1;
         integ.wts[1]  = wt1 * wt2;
         integ.wts[2]  = wt1 * wt2;
         integ.wts[3]  = wt1 * wt1;
         integ.wts[4]  = wt2 * wt1;
         integ.wts[5]  = wt2 * wt2;
         integ.wts[6]  = wt2 * wt2;
         integ.wts[7]  = wt2 * wt1;
         integ.wts[8]  = wt2 * wt1;
         integ.wts[9]  = wt2 * wt2;
         integ.wts[10] = wt2 * wt2;
         integ.wts[11] = wt2 * wt1;
         integ.wts[12] = wt1 * wt1;
         integ.wts[13] = wt1 * wt2;
         integ.wts[14] = wt1 * wt2;
         integ.wts[15] = wt1 * wt1;

//         RealT x1 = std::sqrt((15. + 2. * std::sqrt(30.)) / 35.);
//         RealT x2 = std::sqrt((15. - 2. * std::sqrt(30.)) / 35.);
         RealT x1 = std::sqrt(3./7. + 2./7. * std::sqrt(6./5.));
         RealT x2 = std::sqrt(3./7. - 2./7. * std::sqrt(6./5.));

         integ.xy[0]  = -x1;
         integ.xy[1]  = -x1;
         integ.xy[2]  = -x1;
         integ.xy[3]  = -x2; 
         integ.xy[4]  = -x1;
         integ.xy[5]  =  x2;
         integ.xy[6]  = -x1;
         integ.xy[7]  =  x1;

         integ.xy[8]  = -x2;
         integ.xy[9]  = -x1;
         integ.xy[10] = -x2;
         integ.xy[11] = -x2;
         integ.xy[12] = -x2;
         integ.xy[13] =  x2;
         integ.xy[14] = -x2;
         integ.xy[15] =  x1;

         integ.xy[16] =  x2;
         integ.xy[17] = -x1;
         integ.xy[18] =  x2;
         integ.xy[19] = -x2;
         integ.xy[20] =  x2;
         integ.xy[21] =  x2;
         integ.xy[22] =  x2;
         integ.xy[23] =  x1;

         integ.xy[24] =  x1;
         integ.xy[25] = -x1;
         integ.xy[26] =  x1;
         integ.xy[27] = -x2;
         integ.xy[28] =  x1;
         integ.xy[29] =  x2;
         integ.xy[30] =  x1;
         integ.xy[31] =  x1;
         break;
      }
      case 5:
      {
         RealT wt1 = 1. / 900. * (322. - 13. * std::sqrt(70.));
         RealT wt2 = 1. / 900. * (322. + 13. * std::sqrt(70.));
         RealT wt3 = 128. / 225.;
     
         // points are ordered bottom to top, left to right
         integ.wts[0] = wt1 * wt1;
         integ.wts[1] = wt1 * wt2;
         integ.wts[2] = wt1 * wt3;
         integ.wts[3] = wt1 * wt2;
         integ.wts[4] = wt1 * wt1;
         integ.wts[5] = wt2 * wt1;
         integ.wts[6] = wt2 * wt2;
         integ.wts[7] = wt2 * wt3;
         integ.wts[8] = wt2 * wt2;
         integ.wts[9] = wt2 * wt1;
         integ.wts[10] = wt3 * wt1;
         integ.wts[11] = wt3 * wt2;
         integ.wts[12] = wt3 * wt3;
         integ.wts[13] = wt3 * wt2;
         integ.wts[14] = wt3 * wt1;
         integ.wts[15] = wt2 * wt1;
         integ.wts[16] = wt2 * wt2;
         integ.wts[17] = wt2 * wt3;
         integ.wts[18] = wt2 * wt2;
         integ.wts[19] = wt2 * wt1;
         integ.wts[20] = wt1 * wt1;
         integ.wts[21] = wt1 * wt2;
         integ.wts[22] = wt1 * wt3;
         integ.wts[23] = wt1 * wt2;
         integ.wts[24] = wt1 * wt1;

         RealT x1 = 1./3. * std::sqrt(5. + 2. * std::sqrt(10./7.)) ;
         RealT x2 = 1./3. * std::sqrt(5. - 2. * std::sqrt(10./7.)) ;
         RealT x3 = 0.;

         integ.xy[0] = -x1;
         integ.xy[1] = -x1;
         integ.xy[2] = -x1;
         integ.xy[3] = -x2;
         integ.xy[4] = -x1;
         integ.xy[5] = x3;
         integ.xy[6] = -x1;
         integ.xy[7] = x2;
         integ.xy[8] = -x1;
         integ.xy[9] = x1;

         integ.xy[10] = -x2;
         integ.xy[11] = -x1;
         integ.xy[12] = -x2;
         integ.xy[13] = -x2;
         integ.xy[14] = -x2;
         integ.xy[15] = x3;
         integ.xy[16] = -x2;
         integ.xy[17] = x2;
         integ.xy[18] = -x2; 
         integ.xy[19] = x1;

         integ.xy[20] = x3;
         integ.xy[21] = -x1;
         integ.xy[22] = x3;
         integ.xy[23] = -x2;
         integ.xy[24] = x3;
         integ.xy[25] = x3;
         integ.xy[26] = x3;
         integ.xy[27] = x2;
         integ.xy[28] = x3;
         integ.xy[29] = x1;

         integ.xy[30] = x2;
         integ.xy[31] = -x1;
         integ.xy[32] = x2;
         integ.xy[33] = -x2;
         integ.xy[34] = x2;
         integ.xy[35] = x3;
         integ.xy[36] = x2;
         integ.xy[37] = x2;
         integ.xy[38] = x2;
         integ.xy[39] = x1;

         integ.xy[40] = x1;
         integ.xy[41] = -x1;
         integ.xy[42] = x1;
         integ.xy[43] = -x2;
         integ.xy[44] = x1;
         integ.xy[45] = x3;
         integ.xy[46] = x1;
         integ.xy[47] = x2;
         integ.xy[48] = x1;
         integ.xy[49] = x1;
  
         break;
      }
   }

   return;
}

//------------------------------------------------------------------------------

} // end namespace tribol
