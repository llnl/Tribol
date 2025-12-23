// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "GeomUtilities.hpp"
#include "CompGeom.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/utils/Math.hpp"

#ifdef TRIBOL_USE_ENZYME
#include "mfem/general/enzyme.hpp"
#endif

#include "axom/core.hpp"
#include "axom/slic.hpp"

#include <float.h>
#include <cmath>

namespace tribol {

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ComputeLocalBasis( RealT nx, RealT ny, RealT nz, RealT& e1x, RealT& e1y, RealT& e1z, RealT& e2x,
                                           RealT& e2y, RealT& e2z )
{
  constexpr int max_dim = 3;
  RealT a[max_dim];
  for ( int i = 0; i < max_dim; ++i ) {
    a[i] = 0.;
  }

  // define a vector non-parallel to the input unit normal. Do so by
  // finding the smallest unit normal component and define a corresponding
  // vector in that direction
  if ( std::abs( nx ) <= std::abs( ny ) && std::abs( nx ) <= std::abs( nz ) ) {
    a[0] = 1.0;
  } else if ( std::abs( ny ) <= std::abs( nx ) && std::abs( ny ) <= std::abs( nz ) ) {
    a[1] = 1.0;
  } else if ( std::abs( nz ) <= std::abs( nx ) && std::abs( nz ) <= std::abs( ny ) ) {
    a[2] = 1.0;
  }

  // compute the first basis vector as a x n / ||a x n||
  crossProd( a[0], a[1], a[2], nx, ny, nz, e1x, e1y, e1z );
  RealT a_cross_n_mag = magnitude( e1x, e1y, e1z );
  RealT inv_a_cross_n_mag = 1.0 / a_cross_n_mag;
  e1x *= inv_a_cross_n_mag;
  e1y *= inv_a_cross_n_mag;
  e1z *= inv_a_cross_n_mag;

  // now compute the second basis vector as n x e1
  crossProd( nx, ny, nz, e1x, e1y, e1z, e2x, e2y, e2z );
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ProjectFaceNodesToPlane( const MeshData::Viewer& mesh, int faceId, RealT nrmlX, RealT nrmlY,
                                                 RealT nrmlZ, RealT cX, RealT cY, RealT cZ, RealT* pX, RealT* pY,
                                                 RealT* pZ )
{
  // loop over nodes and project onto the plane defined by the point-normal
  // input arguments
  for ( int i = 0; i < mesh.numberOfNodesPerElement(); ++i ) {
    const int nodeId = mesh.getGlobalNodeId( faceId, i );
    ProjectPointToPlane( mesh.getPosition()[0][nodeId], mesh.getPosition()[1][nodeId], mesh.getPosition()[2][nodeId],
                         nrmlX, nrmlY, nrmlZ, cX, cY, cZ, pX[i], pY[i], pZ[i] );
  }

  return;

}  // end ProjectFaceNodesToPlane()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ProjectEdgeNodesToSegment( const MeshData::Viewer& mesh, int edgeId, RealT nrmlX, RealT nrmlY,
                                                   RealT cX, RealT cY, RealT* pX, RealT* pY )
{
  for ( int i = 0; i < mesh.numberOfNodesPerElement(); ++i ) {
    const int nodeId = mesh.getGlobalNodeId( edgeId, i );
    ProjectPointToSegment( mesh.getPosition()[0][nodeId], mesh.getPosition()[1][nodeId], nrmlX, nrmlY, cX, cY, pX[i],
                           pY[i] );
  }

  return;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ProjectPointToPlane( const RealT x, const RealT y, const RealT z, const RealT nx,
                                             const RealT ny, const RealT nz, const RealT ox, const RealT oy,
                                             const RealT oz, RealT& px, RealT& py, RealT& pz )
{
  // compute the vector from input point to be projected to
  // the origin point on the plane
  RealT vx = x - ox;
  RealT vy = y - oy;
  RealT vz = z - oz;

  // compute the projection onto the plane normal
  RealT dist = vx * nx + vy * ny + vz * nz;

  // compute the projected coordinates of the input point
  px = x - dist * nx;
  py = y - dist * ny;
  pz = z - dist * nz;

  return;

}  // end ProjectPointToPlane()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ProjectPointsToPlane( const RealT* x, const RealT* y, const RealT* z, const RealT nx,
                                              const RealT ny, const RealT nz, const RealT ox, const RealT oy,
                                              const RealT oz, RealT* px, RealT* py, RealT* pz, const int num_points )
{
  for ( int i = 0; i < num_points; ++i ) {
    ProjectPointToPlane( x[i], y[i], z[i], nx, ny, nz, ox, oy, oz, px[i], py[i], pz[i] );
  }
}  // end ProjectPointsToPlane()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void ProjectPointToSegment( const RealT x, const RealT y, const RealT nx, const RealT ny,
                                               const RealT ox, const RealT oy, RealT& px, RealT& py )
{
  // compute the vector from input point to be projected to
  // the origin point on the plane
  RealT vx = x - ox;
  RealT vy = y - oy;

  // compute the projection onto the plane normal
  RealT dist = vx * nx + vy * ny;

  // compute the projected coordinates of the input point
  px = x - dist * nx;
  py = y - dist * ny;

  return;

}  // end ProjectPointToSegment()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void PolyInterYCentroid( const int namax, const RealT* const xa, const RealT* const ya,
                                            const int nbmax, const RealT* const xb, const RealT* const yb,
                                            const int isym, RealT& area, RealT& ycent )
{
  RealT vol;

  // calculate origin shift to avoid roundoff errors
  // TODO figure out numeric limits associated with RealT that also works on device
  RealT xorg = FLT_MAX;
  RealT yorg = FLT_MAX;
  RealT xa_min = FLT_MAX;
  RealT xa_max = -FLT_MAX;
  RealT ya_min = FLT_MAX;
  RealT ya_max = -FLT_MAX;
  RealT xb_min = FLT_MAX;
  RealT xb_max = -FLT_MAX;
  RealT yb_min = FLT_MAX;
  RealT yb_max = -FLT_MAX;

  RealT qy = 0.0;

  if ( nbmax < 1 || namax < 1 ) {
    area = 0.0;
    vol = 0.0;
    ycent = 0.0;
    return;
  }

  for ( int na = 0; na < namax; ++na ) {
    if ( xa[na] < xa_min ) {
      xa_min = xa[na];
    }
    if ( ya[na] < ya_min ) {
      ya_min = ya[na];
    }
    if ( xa[na] > xa_max ) {
      xa_max = xa[na];
    }
    if ( ya[na] > ya_max ) {
      ya_max = ya[na];
    }
    xorg = axom::utilities::min( xorg, xa[na] );
    yorg = axom::utilities::min( yorg, ya[na] );
  }
  for ( int nb = 0; nb < nbmax; ++nb ) {
    if ( xb[nb] < xb_min ) {
      xb_min = xb[nb];
    }
    if ( yb[nb] < yb_min ) {
      yb_min = yb[nb];
    }
    if ( xb[nb] > xb_max ) {
      xb_max = xb[nb];
    }
    if ( yb[nb] > yb_max ) {
      yb_max = yb[nb];
    }
    xorg = axom::utilities::min( xorg, xb[nb] );
    yorg = axom::utilities::min( yorg, yb[nb] );
  }
  if ( isym == 1 ) {
    yorg = axom::utilities::max( yorg, 0.0 );
  }

  area = 0.0;
  vol = 0.0;
  ycent = 0.0;
  if ( xa_min > xb_max ) {
    return;
  }
  if ( xb_min > xa_max ) {
    return;
  }
  if ( ya_min > yb_max ) {
    return;
  }
  if ( yb_min > ya_max ) {
    return;
  }

  // loop over faces of polygon a
  for ( int na = 0; na < namax; ++na ) {
    int nap = ( na + 1 ) % namax;
    RealT xa1 = xa[na] - xorg;
    RealT ya1 = ya[na] - yorg;
    RealT xa2 = xa[nap] - xorg;
    RealT ya2 = ya[nap] - yorg;
    if ( isym == 1 ) {
      if ( ya[na] < 0.0 && ya[nap] < 0.0 ) {
        continue;
      }
      if ( ya[na] < 0.0 ) {
        if ( ya1 != ya2 ) {
          xa1 = xa1 - ( ya1 + yorg ) * ( xa2 - xa1 ) / ( ya2 - ya1 );
        }
        ya1 = -yorg;
      } else if ( ya[nap] < 0.0 ) {
        if ( ya1 != ya2 ) {
          xa2 = xa2 - ( ya2 + yorg ) * ( xa1 - xa2 ) / ( ya1 - ya2 );
        }
        ya2 = -yorg;
      }
    }
    RealT dxa = xa2 - xa1;
    if ( dxa == 0.0 ) {
      continue;
    }
    RealT dya = ya2 - ya1;
    RealT slopea = dya / dxa;

    // loop over faces of polygon b
    for ( int nb = 0; nb < nbmax; ++nb ) {
      int nbp = ( nb + 1 ) % nbmax;
      RealT xb1 = xb[nb] - xorg;
      RealT yb1 = yb[nb] - yorg;
      RealT xb2 = xb[nbp] - xorg;
      RealT yb2 = yb[nbp] - yorg;
      if ( isym == 1 ) {
        if ( yb[nb] < 0.0 && yb[nbp] < 0.0 ) {
          continue;
        }
        if ( yb[nb] < 0.0 ) {
          if ( yb1 != yb2 ) {
            xb1 = xb1 - ( yb1 + yorg ) * ( xb2 - xb1 ) / ( yb2 - yb1 );
          }
          yb1 = -yorg;
        } else if ( yb[nbp] < 0.0 ) {
          if ( yb1 != yb2 ) {
            xb2 = xb2 - ( yb2 + yorg ) * ( xb1 - xb2 ) / ( yb1 - yb2 );
          }
          yb2 = -yorg;
        }
      }
      RealT dxb = xb2 - xb1;
      if ( dxb == 0.0 ) {
        continue;
      }
      RealT dyb = yb2 - yb1;
      RealT slopeb = dyb / dxb;

      // determine sign of volume of intersection
      RealT s = dxa * dxb;

      // calculate left and right coordinates of overlap
      RealT xl = axom::utilities::max( axom::utilities::min( xa1, xa2 ), axom::utilities::min( xb1, xb2 ) );
      RealT xr = axom::utilities::min( axom::utilities::max( xa1, xa2 ), axom::utilities::max( xb1, xb2 ) );
      if ( xl >= xr ) {
        continue;
      }
      RealT yla = ya1 + ( xl - xa1 ) * slopea;
      RealT ylb = yb1 + ( xl - xb1 ) * slopeb;
      RealT yra = ya1 + ( xr - xa1 ) * slopea;
      RealT yrb = yb1 + ( xr - xb1 ) * slopeb;
      RealT yl = axom::utilities::min( yla, ylb );
      RealT yr = axom::utilities::min( yra, yrb );

      RealT area1;
      RealT qy1;
      RealT ym;

      // check if lines intersect
      RealT dslope = slopea - slopeb;
      if ( dslope != 0.0 ) {
        RealT xm = ( yb1 - ya1 + slopea * xa1 - slopeb * xb1 ) / dslope;
        ym = ya1 + slopea * ( xm - xa1 );
        if ( xm > xl && xm < xr ) {
          // lines intersect, case ii
          area1 = 0.5 * copysign( ( yl + ym ) * ( xm - xl ), s );
          RealT area2 = 0.5 * copysign( ( ym + yr ) * ( xr - xm ), s );
          area = area + area1 + area2;

          if ( yl + ym > 0 ) {
            qy1 = 1.0 / 3.0 * ( ym + yl * yl / ( yl + ym ) ) * area1;
            qy = qy + qy1;
          }
          if ( ym + yr > 0 ) {
            RealT qy2 = 1.0 / 3.0 * ( yr + ym * ym / ( ym + yr ) ) * area2;
            qy = qy + qy2;
          }

          if ( isym == 1 ) {
            yl = yl + yorg;
            ym = ym + yorg;
            yr = yr + yorg;
            vol = vol + copysign( ( xm - xl ) * ( yl * yl + yl * ym + ym * ym ) +
                                      ( xr - xm ) * ( ym * ym + ym * yr + yr * yr ),
                                  s ) /
                            3.0;
          }
          continue;
        }
      }

      // lines do not intersect, case i
      area1 = 0.5 * copysign( ( xr - xl ) * ( yr + yl ), s );
      area = area + area1;
      if ( yl + yr > 0 ) {
        qy1 = 1. / 3.0 * ( yr + yl * yl / ( yl + yr ) ) * area1;
        qy = qy + qy1;
      }

      if ( isym == 1 ) {
        yl = yl + yorg;
        ym = ym + yorg;
        yr = yr + yorg;
        vol = vol + copysign( ( xr - xl ) * ( yl * yl + yl * yr + yr * yr ), s ) / 3.0;
      }
    }
  }

  if ( area != 0.0 ) {
    ycent = qy / area + yorg;
  }

  if ( isym == 0 ) {
    vol = area;
  }

  return;

}  // end PolyInterYCentroid()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void Local2DToGlobalCoords( RealT xloc, RealT yloc, RealT e1X, RealT e1Y, RealT e1Z, RealT e2X,
                                               RealT e2Y, RealT e2Z, RealT cX, RealT cY, RealT cZ, RealT& xg, RealT& yg,
                                               RealT& zg )
{
  // This projection takes the two input local vector components and uses
  // them as coefficients in a linear combination of local basis vectors.
  // This gives a 3-vector with origin at the common plane centroid.
  RealT vx = xloc * e1X + yloc * e2X;
  RealT vy = xloc * e1Y + yloc * e2Y;
  RealT vz = xloc * e1Z + yloc * e2Z;

  // the vector in the global coordinate system requires the addition of the
  // plane point vector (global Cartesian coordinate basis) to the previously
  // computed vector
  xg = vx + cX;
  yg = vy + cY;
  zg = vz + cZ;

  return;

}  // end Local2DToGlobalCoords()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void GlobalTo2DLocalCoords( const RealT* const pX, const RealT* const pY, const RealT* const pZ,
                                               RealT e1X, RealT e1Y, RealT e1Z, RealT e2X, RealT e2Y, RealT e2Z,
                                               RealT cX, RealT cY, RealT cZ, RealT* const pLX, RealT* const pLY,
                                               int size )
{
#ifdef TRIBOL_USE_HOST
  SLIC_ERROR_IF( size > 0 && ( pLX == nullptr || pLY == nullptr ),
                 "GlobalTo2DLocalCoords: local coordinate pointers are null" );
#endif

  // loop over projected nodes
  for ( int i = 0; i < size; ++i ) {
    // compute the vector between the point on the plane and the input plane point
    RealT vX = pX[i] - cX;
    RealT vY = pY[i] - cY;
    RealT vZ = pZ[i] - cZ;

    // project this vector onto the {e1,e2} local basis. This vector is
    // in the plane so the out-of-plane component should be zero.
    pLX[i] = vX * e1X + vY * e1Y + vZ * e1Z;  // projection onto e1
    pLY[i] = vX * e2X + vY * e2Y + vZ * e2Z;  // projection onto e2
  }

  return;

}  // end GlobalTo2DLocalCoords()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void GlobalTo2DLocalCoords( RealT pX, RealT pY, RealT pZ, RealT e1X, RealT e1Y, RealT e1Z, RealT e2X,
                                               RealT e2Y, RealT e2Z, RealT cX, RealT cY, RealT cZ, RealT& pLX,
                                               RealT& pLY )
{
  // compute the vector between the point on the plane and the input plane point
  RealT vX = pX - cX;
  RealT vY = pY - cY;
  RealT vZ = pZ - cZ;

  // project this vector onto the {e1,e2} local basis. This vector is
  // in the plane so the out-of-plane component should be zero.
  pLX = vX * e1X + vY * e1Y + vZ * e1Z;  // projection onto e1
  pLY = vX * e2X + vY * e2Y + vZ * e2Z;  // projection onto e2

  return;

}  // end GlobalTo2DLocalCoords()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE bool VertexAvgCentroid( const RealT* const x, const int dim, const int numVert, RealT& cX, RealT& cY,
                                           RealT& cZ )
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
    cX += x[dim * i];
    cY += x[dim * i + 1];
    if ( dim > 2 ) {
      cZ += x[dim * i + 2];
    }
  }

  // divide by the number of nodes to compute average
  cX *= fac;
  cY *= fac;
  cZ *= fac;

  return true;

}  // end VertexAvgCentroid()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE bool PolyAreaCentroid( const RealT* const x, const int dim, const int numVert, RealT& cX, RealT& cY,
                                          RealT& cZ )
{
#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
  SLIC_ERROR_IF( dim != 3, "PolyAreaCentroid: Only compatible with dim = 3." );
  SLIC_ERROR_IF( numVert == 0, "PolyAreaCentroid: numVert = 0." );
#endif
  if ( numVert == 0 ) {
    return false;
  }

  // (re)initialize the input/output centroid components
  cX = 0.0;
  cY = 0.0;
  cZ = 0.0;

  // compute the vertex average centroid of the polygon in
  // order to break it up into triangles
  RealT cX_poly, cY_poly, cZ_poly;
  VertexAvgCentroid( x, dim, numVert, cX_poly, cY_poly, cZ_poly );

  // loop over triangles formed from adjacent polygon vertices
  // and the vertex averaged centroid
  RealT xTri[3] = { 0., 0., 0. };
  RealT yTri[3] = { 0., 0., 0. };
  RealT zTri[3] = { 0., 0., 0. };

  // assign all of the last triangle coordinates to the
  // polygon's vertex average centroid
  xTri[2] = cX_poly;
  yTri[2] = cY_poly;
  zTri[2] = cZ_poly;
  RealT area_sum = 0.;
  for ( int i = 0; i < numVert; ++i )  // loop over triangles
  {
    // group triangle coordinates
    int triId = i;
    int triIdPlusOne = ( i == ( numVert - 1 ) ) ? 0 : triId + 1;
    xTri[0] = x[dim * triId];
    yTri[0] = x[dim * triId + 1];
    zTri[0] = x[dim * triId + 2];
    xTri[1] = x[dim * triIdPlusOne];
    yTri[1] = x[dim * triIdPlusOne + 1];
    zTri[1] = x[dim * triIdPlusOne + 2];

    // compute the area of the triangle
    RealT area_tri = Area3DTri( xTri, yTri, zTri );
    area_sum += area_tri;

    // compute the vertex average centroid of the triangle
    RealT cX_tri, cY_tri, cZ_tri;
    VertexAvgCentroid( &xTri[0], &yTri[0], &zTri[0], 3, cX_tri, cY_tri, cZ_tri );

    cX += cX_tri * area_tri;
    cY += cY_tri * area_tri;
    cZ += cZ_tri * area_tri;
  }

  cX /= area_sum;
  cY /= area_sum;
  cZ /= area_sum;

  return true;

}  // end PolyAreaCentroid()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE FaceGeomException CheckSegOverlap( const RealT* const pX1, const RealT* const pY1,
                                                      const RealT* const pX2, const RealT* const pY2, const int nV1,
                                                      const int nV2, RealT* overlapX, RealT* overlapY, RealT& area )
{
  // TODO: Re-write in a way where the assert isn't needed
#ifdef TRIBOL_USE_CUDA
  assert( nV1 == 2 );
  assert( nV2 == 2 );
#else
  SLIC_ASSERT( nV1 == 2 );
  SLIC_ASSERT( nV2 == 2 );
#endif

  // define the edge 1 non-unit directional vector between vertices
  // 2 and 1
  RealT lvx1 = pX1[1] - pX1[0];
  RealT lvy1 = pY1[1] - pY1[0];

  RealT e1_len = magnitude( lvx1, lvy1 );

  // define the edge 2 non-unit directional vector between vertices
  // 2 and 1
  RealT lvx2 = pX2[1] - pX2[0];
  RealT lvy2 = pY2[1] - pY2[0];

  RealT e2_len = magnitude( lvx2, lvy2 );

  //
  // perform the all-in-1 check
  //

  // compute vector between each edge 2 vertex and vertex 1 on edge 1.
  // Then dot that vector with the directional vector of edge 1 to see
  // if they are codirectional (projection > 0 indicating edge 2 vertex
  // lies within or beyond edge 1. If so, check, that this vector length is
  // less than edge 1 length indicating that the vertex lies within edge 1
  int inter2 = 0;
  int twoInOneId = -1;
  for ( int i = 0; i < nV2; ++i ) {
    RealT vx = pX2[i] - pX1[0];
    RealT vy = pY2[i] - pY1[0];

    // compute projection onto edge 1 directional vector. (Positive if codirectional,
    // negative otherwise. Only positive projections will be potential overlap vertex candidates
    RealT proj = vx * lvx1 + vy * lvy1;

    // compute length of <vx,vy>; if vLen < some tolerance we have a
    // coincident node
    RealT vLen = magnitude( vx, vy );

    // check for >= 0 projections and vector lengths <= edge 1 length. This
    // indicates an edge 2 vertex interior to edge 1, or coincident vertices in
    // the case of projection = 0 or vector length is equal to edge 1 length
    if ( proj >= 0 && vLen <= e1_len )  // interior vertex
    {
      twoInOneId = i;
      ++inter2;
    }
  }

  // if both vertices pass the above criteria than 2 is in 1
  if ( inter2 == 2 ) {
    // set the contact plane (segment) length
    area = e2_len;

    // set the vertices of the overlap segment
    overlapX[0] = pX2[0];
    overlapY[0] = pY2[0];

    overlapX[1] = pX2[1];
    overlapY[1] = pY2[1];

    return NO_FACE_GEOM_EXCEPTION;
  }

  //
  // perform the all-in-2 check
  //

  // compute vector between each edge 1 vertex and vertex 1 on edge 2.
  // Then dot that vector with the directional vector of edge 2 to see
  // if they are codirectional. If so, check, that this vector length is
  // less than edge 2 length indicating that the vertex is within edge 2
  int inter1 = 0;
  int oneInTwoId = -1;
  for ( int i = 0; i < nV1; ++i ) {
    RealT vx = pX1[i] - pX2[0];
    RealT vy = pY1[i] - pY2[0];

    // compute projection onto edge 2 directional vector
    RealT proj = vx * lvx2 + vy * lvy2;

    // compute length of <vx,vy>
    RealT vLen = magnitude( vx, vy );

    // check for >= 0 projections and vector lengths <= edge 2 length. This
    // indicates an edge 1 vertex interior to edge 2 or is coincident if the
    // projection is zero or vector length is equal to edge 2 length
    if ( proj >= 0. && vLen <= e2_len )  // interior vertex
    {
      oneInTwoId = i;
      ++inter1;
    }
  }

  // if both vertices pass the above criteria then 1 is in 2.
  if ( inter1 == 2 ) {
    // set the contact plane (segment) length
    area = e1_len;

    // set the overlap segment vertices on the contact plane object
    overlapX[0] = pX1[0];
    overlapY[0] = pY1[0];

    overlapX[1] = pX1[1];
    overlapY[1] = pY1[1];

    return NO_FACE_GEOM_EXCEPTION;
  }

  // if inter1 == 0 and inter2 == 0 then there is no overlap
  if ( inter1 == 0 && inter2 == 0 ) {
    area = 0.0;
    return NO_OVERLAP;
  }

  // there is a chance that oneInTowId or twoInOneId is not actually set,
  // in which case we don't have an overlap.
  if ( oneInTwoId == -1 || twoInOneId == -1 ) {
    area = 0.0;
    return NO_OVERLAP;
  }

  // if we are here, we have ruled out all-in-1 and all-in-2 overlaps,
  // and non-overlapping edges, but have the case where edge 1 and
  // edge 2 overlap some finite distance that is less than either of their
  // lengths. We have vertex information from the all-in-one checks
  // indicating which vertices on one edge are within the other edge

  // set the segment vertices
  overlapX[0] = pX1[oneInTwoId];
  overlapY[0] = pY1[oneInTwoId];
  overlapX[1] = pX2[twoInOneId];
  overlapY[1] = pY2[twoInOneId];

  // compute vector between "inter"-vertices
  RealT vecX = overlapX[1] - overlapX[0];
  RealT vecY = overlapY[1] - overlapY[0];

  // compute the length of the overlapping segment
  area = magnitude( vecX, vecY );

  return NO_FACE_GEOM_EXCEPTION;

}  // end CommonPlanePair::checkSegOverlap()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE RealT Area3DTri( const RealT* const x, const RealT* const y, const RealT* const z )
{
  RealT u[3] = { x[1] - x[0], y[1] - y[0], z[1] - z[0] };
  RealT v[3] = { x[2] - x[0], y[2] - y[0], z[2] - z[0] };

  return std::abs( 1. / 2. * magCrossProd( u, v ) );

}  // end Area3DTri()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void PolyReorderWithNormal( RealT* const x, RealT* const y, RealT* const z, const int numPoints,
                                               const RealT nX, const RealT nY, const RealT nZ )
{
  if ( numPoints < 3 ) {
#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
    SLIC_DEBUG( "PolyReorderWithNormal(): numPoints (" << numPoints << ") < 3." );
#endif
    return;
  }

  constexpr int max_nodes_per_overlap = 5 * 2;  // max face polygon for interpen can be 5

#if defined( TRIBOL_USE_HOST )
  SLIC_ERROR_IF( numPoints > max_nodes_per_overlap, "PolyReorderWithNormal: numPoints exceed maximum "
                                                        << "expected per overlap (" << max_nodes_per_overlap << ")." );
#endif

  // form link vectors between second and first vertex and third and first
  // vertex
  RealT lv10X = x[1] - x[0];
  RealT lv10Y = y[1] - y[0];
  RealT lv10Z = z[1] - z[0];

  RealT lv20X = x[2] - x[0];
  RealT lv20Y = y[2] - y[0];
  RealT lv20Z = z[2] - z[0];

  // take the cross product of the vectors to get the normal
  RealT pNrmlX, pNrmlY, pNrmlZ;
  crossProd( lv10X, lv10Y, lv10Z, lv20X, lv20Y, lv20Z, pNrmlX, pNrmlY, pNrmlZ );

  // dot the computed plane normal based on vertex ordering with the
  // input normal
  RealT v = dotProd( pNrmlX, pNrmlY, pNrmlZ, nX, nY, nZ );

  // check to see if v is negative. If so, reorient the vertices
  if ( v < 0. ) {
    RealT xTemp[max_nodes_per_overlap];
    RealT yTemp[max_nodes_per_overlap];
    RealT zTemp[max_nodes_per_overlap];

    xTemp[0] = x[0];
    yTemp[0] = y[0];
    zTemp[0] = z[0];

    for ( int i = 1; i < numPoints; ++i ) {
      xTemp[i] = x[numPoints - i];
      yTemp[i] = y[numPoints - i];
      zTemp[i] = z[numPoints - i];
    }

    for ( int i = 0; i < numPoints; ++i ) {
      x[i] = xTemp[i];
      y[i] = yTemp[i];
      z[i] = zTemp[i];
    }
  }

  return;

}  // end PolyReorderWithNormal()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE bool LinePlaneIntersection( const RealT xA, const RealT yA, const RealT zA, const RealT xB,
                                               const RealT yB, const RealT zB, const RealT xP, const RealT yP,
                                               const RealT zP, const RealT nX, const RealT nY, const RealT nZ, RealT& x,
                                               RealT& y, RealT& z, bool& isParallel )
{
  // compute segment vector
  RealT lambdaX = xB - xA;
  RealT lambdaY = yB - yA;
  RealT lambdaZ = zB - zA;

  // check dot product with plane normal
  RealT prod = lambdaX * nX + lambdaY * nY + lambdaZ * nZ;

  if ( prod == 0. )  // line lies in plane or parallel to plane
  {
    x = 0.;
    y = 0.;
    z = 0.;
    isParallel = true;
    return false;
  }

  // compute vector difference between point on plane
  // and first vertex on segment
  RealT vX = xP - xA;
  RealT vY = yP - yA;
  RealT vZ = zP - zA;

  // compute dot product between <vX, vY, vZ> and the plane normal
  RealT prodV = vX * nX + vY * nY + vZ * nZ;

  // compute the line segment parameter, t, and check to see if it is
  // between 0 and 1, inclusive
  RealT t = prodV / prod;

  if ( t >= 0 && t <= 1 ) {
    x = xA + lambdaX * t;
    y = yA + lambdaY * t;
    z = zA + lambdaZ * t;
    isParallel = false;
    return true;
  } else {
    x = 0.;
    y = 0.;
    z = 0.;
    isParallel = false;
    return false;
  }

}  // end LinePlaneIntersection()

//------------------------------------------------------------------------------
bool PlanePlaneIntersection( const RealT x1, const RealT y1, const RealT z1, const RealT x2, const RealT y2,
                             const RealT z2, const RealT nX1, const RealT nY1, const RealT nZ1, const RealT nX2,
                             const RealT nY2, const RealT nZ2, RealT& x, RealT& y, RealT& z )
{
  // note: this routine has not been tested

  // check dot product between two normals for coplanarity
  RealT coProd = nX1 * nX2 + nY1 * nY2 + nZ1 * nZ2;

  if ( axom::utilities::isNearlyEqual( coProd, 1.0, 1.e-8 ) ) {
    x = 0.;
    y = 0.;
    z = 0.;
    return false;
  }

  // compute dot products between each plane's reference point and the normal
  RealT prod1 = nX1 * x1 + nY1 * y1 + nZ1 * z1;
  RealT prod2 = nX2 * x2 + nY2 * y2 + nZ2 * z2;

  // form matrix of dot products between normals
  RealT A11 = nX1 * nX1 + nY1 * nY1 + nZ1 * nZ1;
  RealT A12 = nX1 * nX2 + nY1 * nY2 + nZ1 * nZ2;
  RealT A22 = nX2 * nX2 + nY2 * nY2 + nZ2 * nZ2;

  // form determinant and inverse determinant of 2x2 matrix
  RealT detA = A11 * A22 - A12 * A12;
  RealT invDetA = 1.0 / detA;

  // form inverse matrix components
  RealT invA11 = A22;
  RealT invA12 = -A12;
  RealT invA22 = A11;

  // compute two parameters for point on line of intersection
  RealT s1 = invDetA * ( prod1 * invA11 + prod2 * invA12 );
  RealT s2 = invDetA * ( prod1 * invA12 + prod2 * invA22 );

  // compute the point on the line of intersection
  x = s1 * nX1 + s2 * nX2;
  y = s1 * nY1 + s2 * nY2;
  z = s1 * nZ1 + s2 * nZ2;

  return true;

}  // end PlanePlaneIntersection()

//------------------------------------------------------------------------------
void Vertex2DOrderToCCW( const RealT* const x, const RealT* const y, RealT* xTemp, RealT* yTemp, const int numVert )
{
  if ( numVert <= 0 ) {
    SLIC_DEBUG( "Vertex2DOrderToCCW: numVert <= 0; returning." );
    return;
  }

  SLIC_ERROR_IF( x == nullptr || y == nullptr || xTemp == nullptr || yTemp == nullptr,
                 "Vertex2DOrderToCCW: must set pointers prior to call to routine." );

  xTemp[0] = x[0];
  yTemp[0] = y[0];

  int k = 1;
  for ( int i = numVert; i > 0; --i ) {
    xTemp[k] = x[i];
    yTemp[k] = y[i];
    ++k;
  }

  return;

}  // end Vertex2DOrderToCCW()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void Points3DTo2D( const RealT* const x, const RealT* const y, const RealT* const z, const RealT nx,
                                      const RealT ny, const RealT nz, const RealT cx, const RealT cy, const RealT cz,
                                      const int num_verts, RealT* x_loc, RealT* y_loc )
{
  RealT e1x, e1y, e1z;
  RealT e2x, e2y, e2z;

  ComputeLocalBasis( nx, ny, nz, e1x, e1y, e1z, e2x, e2y, e2z );
  GlobalTo2DLocalCoords( x, y, z, e1x, e1y, e1z, e2x, e2y, e2z, cx, cy, cz, x_loc, y_loc, num_verts );
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE bool IsPointInEdge( const RealT* const x, const RealT* const y, RealT xp, RealT yp,
                                       RealT fuzz_factor )
{
  RealT xmax, xmin, ymax, ymin;
  if ( x[0] > x[1] ) {
    xmax = x[0];
    xmin = x[1];
  } else {
    xmax = x[1];
    xmin = x[0];
  }

  if ( y[0] > y[1] ) {
    ymax = y[0];
    ymin = y[1];
  } else {
    ymax = y[1];
    ymin = y[0];
  }

  // add fuzz to catch nearly coincident vertices
  RealT l = magnitude( x[1] - x[0], y[1] - y[0] );  // edge length
  RealT fuzz = fuzz_factor * l;

  if ( xp <= ( xmax + fuzz ) && xp >= ( xmin - fuzz ) && yp <= ( ymax + fuzz ) && yp >= ( ymin - fuzz ) ) {
    return true;
  }
  return false;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void CheckPolyOverlap( const int num_nodes_1, const int num_nodes_2, RealT* projLocX1,
                                          RealT* projLocY1, RealT* projLocX2, RealT* projLocY2, RealT& area,
                                          const int isym )
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
  for ( int i = ( num_nodes_2 - 1 ); i > 0; --i ) {
    x2Temp[k] = projLocX2[i];
    y2Temp[k] = projLocY2[i];
    ++k;
  }

  RealT cy;
  PolyInterYCentroid( num_nodes_1, projLocX1, projLocY1, num_nodes_2, x2Temp, y2Temp, isym, area, cy );
  // PolyInterYCentroid( num_nodes_1, projLocY1, projLocX1, num_nodes_2, y2Temp, x2Temp,
  //                     isym, area, cx );

  return;

}  // end CheckPolyOverlap()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE bool IsOverlappingOnPlane( const RealT* const x1, const RealT* const y1, const RealT* const z1,
                                              const RealT* const x2, const RealT* const y2, const RealT* const z2,
                                              const RealT* const n, const RealT* const c, const int numNodesFace1,
                                              const int numNodesFace2, const int dim )
{
  constexpr int max_nodes_per_face = 4;

  if ( dim == 3 ) {
    RealT x1_bar[max_nodes_per_face];
    RealT y1_bar[max_nodes_per_face];
    RealT z1_bar[max_nodes_per_face];
    RealT x2_bar[max_nodes_per_face];
    RealT y2_bar[max_nodes_per_face];
    RealT z2_bar[max_nodes_per_face];

    // project vertices to plane
    ProjectPointsToPlane( x1, y1, z1, n[0], n[1], n[2], c[0], c[1], c[2], &x1_bar[0], &y1_bar[0], &z1_bar[0],
                          numNodesFace1 );
    ProjectPointsToPlane( x2, y2, z2, n[0], n[1], n[2], c[0], c[1], c[2], &x2_bar[0], &y2_bar[0], &z2_bar[0],
                          numNodesFace2 );

    RealT x1_bar_local[max_nodes_per_face];
    RealT y1_bar_local[max_nodes_per_face];
    RealT x2_bar_local[max_nodes_per_face];
    RealT y2_bar_local[max_nodes_per_face];

    // 3D coordinates to local 2D coordinates
    Points3DTo2D( &x1_bar[0], &y1_bar[0], &z1_bar[0], n[0], n[1], n[2], c[0], c[1], c[2], numNodesFace1,
                  &x1_bar_local[0], &y1_bar_local[0] );
    Points3DTo2D( &x2_bar[0], &y2_bar[0], &z2_bar[0], n[0], n[1], n[2], c[0], c[1], c[2], numNodesFace2,
                  &x2_bar_local[0], &y2_bar_local[0] );

    RealT area;
    CheckPolyOverlap( numNodesFace1, numNodesFace2, &x1_bar_local[0], &y1_bar_local[0], &x2_bar_local[0],
                      &y2_bar_local[0], area, 0 );

    if ( area < 1.e-15 ) {
      return false;
    }
    // end dim == 3
  } else {
    RealT projX1[max_nodes_per_face];
    RealT projY1[max_nodes_per_face];
    RealT projX2[max_nodes_per_face];
    RealT projY2[max_nodes_per_face];

    // project edge nodes to plane
    for ( int i = 0; i < numNodesFace1; ++i ) {
      ProjectPointToSegment( x1[i], y1[i], n[0], n[1], c[0], c[1], projX1[i], projY1[i] );
    }

    for ( int i = 0; i < numNodesFace2; ++i ) {
      ProjectPointToSegment( x2[i], y2[i], n[0], n[1], c[0], c[1], projX2[i], projY2[i] );
    }

    // check if either of edge 1's projected vertices are inside projected edge 2
    bool vert1_inside2 = IsPointInEdge( &projX2[0], &projY2[0], projX1[0], projY1[0], 1.e-12 );
    bool vert2_inside2 = IsPointInEdge( &projX2[0], &projY2[0], projX1[1], projY1[1], 1.e-12 );

    // now, check if either of edge 2's projected vertices are inside projected edge 1
    // note, if we just checked for 1 in 2, then if 2 lies entirely within 1 we would have missed that
    bool vert1_inside1 = IsPointInEdge( &projX1[0], &projY1[0], projX2[0], projY2[0], 1.e-12 );
    bool vert2_inside1 = IsPointInEdge( &projX1[0], &projY1[0], projX2[1], projY2[1], 1.e-12 );

    // return false if none of the vertices lie inside the other edge
    if ( !vert1_inside2 && !vert2_inside2 ) {
      if ( !vert1_inside1 && !vert2_inside1 ) {
        return false;
      }
    }

  }  // end dim == 2

  return true;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE bool IsConvex( const RealT* const x, const RealT* const y, const int numPolyVert )
{
  if ( numPolyVert < 4 ) {  // triangles are convex
    return true;
  }

  // for each vertex B, form the two segments AB and BC using the triple (A,B,C).
  // Take the cross product of the two segments. For a convex polygon the cross
  // products are all the same sign (positive or negative depending on how the polygon
  // "turns"). Cross products == 0 can be ignored. This indicates colinear segments
  // and does not help in determining the turning of the polygon
  bool pos = false;
  bool neg = false;
  for ( int i = 0; i < numPolyVert; ++i ) {
    RealT ax = x[( i + 1 ) % numPolyVert] - x[i];
    RealT ay = y[( i + 1 ) % numPolyVert] - y[i];

    RealT bx = x[( i + 2 ) % numPolyVert] - x[( i + 1 ) % numPolyVert];
    RealT by = y[( i + 2 ) % numPolyVert] - y[( i + 1 ) % numPolyVert];

    RealT cross = ax * by - ay * bx;

    // check for strict positivity or negativity of the cross product.
    // Again, cross == 0 can be ignored indicating colinear segments,
    // which does not break convexity, but does not indicate how the
    // polygon turns.
    if ( cross > 0 ) {
      pos = true;
    } else if ( cross < 0 ) {
      neg = true;
    }
  }

  if ( pos && neg ) {
    return false;
  } else {
    return true;
  }

}  // end IsConvex()
//------------------------------------------------------------------------------

}  // end namespace tribol
