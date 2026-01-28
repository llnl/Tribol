// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "GeomUtilities.hpp"

#include "axom/slic.hpp"

namespace tribol {

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

}  // end namespace tribol
