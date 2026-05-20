// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_INTEGRATION_HPP_
#define SRC_TRIBOL_INTEG_INTEGRATION_HPP_

#include "tribol/common/Parameters.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"

namespace tribol {

// forward declaration
struct SurfaceContactElem;

/// struct to hold 2D or 3D integration point coordinates and
//  weights for integration on a face-face overlapping
//  convex polygon. This struct is quadrature rule agnostic.
struct IntegPts {
  /// IntegPts constructor
  IntegPts( int numPoints,  ///< [in] Number of integration points
            int IPDim       ///< [in] dimension of integration point coordinates
            )
      : numIPs( numPoints ), ipDim( IPDim )
  {
    xy = new RealT[IPDim * numPoints];
    wts = new RealT[numPoints];
  }

  /// IntegPts overloaded constructor
  IntegPts() : numIPs( 0 ), xy( nullptr ), wts( nullptr ) {}

  /// Destructor
  ~IntegPts()
  {
    if ( xy != nullptr ) {
      delete[] xy;
      xy = nullptr;
    }
    if ( wts != nullptr ) {
      delete[] wts;
      wts = nullptr;
    }
  }

  /// Initialization function
  void initialize( int const dim, int const numTotalIPs )
  {
    this->ipDim = dim;
    this->numIPs = numTotalIPs;
    if ( this->xy == nullptr ) {
      this->xy = new RealT[dim * numTotalIPs];
    } else {
      delete[] this->xy;
      this->xy = new RealT[dim * numTotalIPs];
    }
    if ( this->wts == nullptr ) {
      this->wts = new RealT[numTotalIPs];
    } else {
      delete[] this->wts;
      this->wts = new RealT[numTotalIPs];
    }
  }

  // member variables
  int numIPs;  ///< number of integration points on entire overlap
  int ipDim;   ///< coordinate dimension of the integration points
  RealT* xy;   ///< coordinates of ALL integration points
  RealT* wts;  ///< integration point weights
};

/*!
 *
 * \brief Templated function with explicit specialization evaluating the
 *        weak form contact integral, typically involving the integration
 *        of shape functions or product of shape functions over contact
 *        overlap patches for surface-to-surface contact methods.
 *
 * \param [in] elem surface contact element struct
 * \param [out] integ1 scalar integral evaluation for face 1 at node nodeEvalId
 * \param [out] integ2 scalar integral evaluation for face 2 at node nodeEvalId
 *
 * \pre The local node id, nodeEvalId, ranges from 0-3 for a four node quad face.
 *
 */
template <ContactMethod M, PolyInteg I>
TRIBOL_HOST_DEVICE inline void EvalWeakFormIntegral( SurfaceContactElem const& elem, RealT* const integ1,
                                                     RealT* const integ2 );

/*!
 *
 * \brief Populates the integration points and weights on the IntegPts object
 *        for all integration points per Taylor-Wingate-Bos integration rule
 *        of order k.
 *
 * \note Integration per M. Taylor, B. Wingate, L. Bos. Several new quadrature
 *       formulas for polynomial integration in the triangle.
 *       arXiv:math/0501496, 2007.
 *
 * \param [in] elem SurfaceContactElem object containing dimension and overlap vertices
 * \param [in,out] integ IntegPts object holding integration points and weights
 * \param [in] k order of TWB integration
 *
 * \pre order 2 <= k <= 3
 * \pre integ IntegPts object can be instantiated with no-op constructor. This routine
 *            will allocate and populate necessary data.
 *
 */
void TWBPolyInt( SurfaceContactElem const& elem, IntegPts& integ, int k );

/*!
 *
 * \brief Populates the integration points and weights on the IntegPts object
 *        for all integration points per symmetric Gauss integration rule
 *        of order k on triangles
 *
 * \param [in] elem SurfaceContactElem object containing dimension and overlap vertices
 * \param [in,out] integ IntegPts object holding integration points and weights
 * \param [in] k order of integration
 *
 * \pre order 2 <= k <= 3
 * \pre integ IntegPts object can be instantiated with no-op constructor. This routine
 *            will allocate and populate necessary data.
 *
 */
void GaussPolyIntTri( SurfaceContactElem const& elem, IntegPts& integ, int k );

/*!
 *
 * \brief Populates the integration points and weights on the IntegPts object
 *        for all integration points per symmetric Gauss integration rule
 *        of order k on quadrilaterals
 *
 * \param [in] elem SurfaceContactElem object containing dimension and overlap vertices
 * \param [in,out] integ IntegPts object holding integration points and weights
 * \param [in] k order of integration
 *
 * \pre order 2 <= k <= 3
 * \pre integ IntegPts object can be instantiated with no-op constructor. This routine
 *            will allocate and populate necessary data.
 *
 */

void GaussPolyIntQuad( SurfaceContactElem const& elem, IntegPts& integ, int k );
/*!
 *
 * \brief returns the number of TWB integration points for polygonal overlap
 *        for integration rule of order k
 *
 * \param [in] elem SurfaceContactElem object containing dimension and overlap vertices
 * \param [in] k order of TWB integration
 *
 * \pre order 2 <= k <= 3
 *
 */
int NumTWBPointsPoly( SurfaceContactElem const& elem, int k );

/*!
 *
 * \brief returns the number of TWB integration points on a triangle per
 *        the integration rule of order k
 *
 * \param [in] order order of polynomial that TWB integration rule will exactly integrate
 *
 * \pre order 2 <= k <= 3
 *
 */
int NumTWBPointsPerTri( int order );

//-----------------------------------------------------------------------------
// Implementations
//-----------------------------------------------------------------------------

TRIBOL_HOST_DEVICE inline void GetCommonPlaneOverlapCentroid( SurfaceContactElem const& elem, RealT cx[3] )
{
  cx[0] = 0.;
  cx[1] = 0.;
  cx[2] = 0.;

  if ( elem.dim == 2 ) {
    VertexAvgCentroid( elem.overlapCoords, elem.dim, elem.numPolyVert, cx[0], cx[1], cx[2] );
  } else {
    PolyAreaCentroid( elem.overlapCoords, elem.dim, elem.numPolyVert, cx[0], cx[1], cx[2] );
  }
}

TRIBOL_HOST_DEVICE inline void AccumulateCommonPlaneIntegralAtPoint( SurfaceContactElem const& elem, const RealT x[3],
                                                                     const RealT wt, RealT* const integ1,
                                                                     RealT* const integ2 )
{
  for ( int a = 0; a < elem.numFaceVert; ++a ) {
    RealT phi1 = 0.;
    RealT phi2 = 0.;
    EvalBasisOnPhysicalFace( elem.faceCoords1, x[0], x[1], x[2], elem.numFaceVert, a, phi1 );
    EvalBasisOnPhysicalFace( elem.faceCoords2, x[0], x[1], x[2], elem.numFaceVert, a, phi2 );
    integ1[a] += wt * phi1;
    integ2[a] += wt * phi2;
  }
}

TRIBOL_HOST_DEVICE inline int GetCommonPlaneTriangleRule( int order, RealT* wts, RealT* coords )
{
  switch ( order ) {
    case 2:
      wts[0] = 0.3333333333;
      wts[1] = 0.3333333333;
      wts[2] = 0.3333333333;

      coords[0] = 0.1666666667;
      coords[1] = 0.1666666667;
      coords[2] = 0.6666666667;
      coords[3] = 0.1666666667;
      coords[4] = 0.1666666667;
      coords[5] = 0.6666666667;
      return 3;
    case 3:
    case 4: {
      constexpr RealT wt1 = 0.109951743655322;
      constexpr RealT wt2 = 0.223381589678011;
      wts[0] = wt1;
      wts[1] = wt1;
      wts[2] = wt1;
      wts[3] = wt2;
      wts[4] = wt2;
      wts[5] = wt2;

      constexpr RealT x1 = 0.091576213509771;
      constexpr RealT x2 = 0.816847572980459;
      constexpr RealT x3 = 0.108103018168070;
      constexpr RealT x4 = 0.445948490915965;
      coords[0] = x1;
      coords[1] = x1;
      coords[2] = x2;
      coords[3] = x1;
      coords[4] = x1;
      coords[5] = x2;
      coords[6] = x3;
      coords[7] = x4;
      coords[8] = x4;
      coords[9] = x3;
      coords[10] = x4;
      coords[11] = x4;
      return 6;
    }
    default:
#ifdef TRIBOL_USE_HOST
      SLIC_ERROR( "GetCommonPlaneTriangleRule(): only Gauss integration of order 2-4 is implemented." );
#endif
      return 0;
  }
}

TRIBOL_HOST_DEVICE inline void EvalWeakFormIntegralCommonPlaneFullTri( SurfaceContactElem const& elem,
                                                                       const int tri_order, RealT* const integ1,
                                                                       RealT* const integ2 )
{
  if ( elem.dim != 3 ) {
    RealT cx[3] = { 0., 0., 0. };
    GetCommonPlaneOverlapCentroid( elem, cx );
    AccumulateCommonPlaneIntegralAtPoint( elem, cx, 1.0, integ1, integ2 );
    return;
  }

  constexpr int max_qpts = 6;
  RealT rule_wts[max_qpts] = { 0., 0., 0., 0., 0., 0. };
  RealT rule_coords[2 * max_qpts] = { 0. };
  const int num_qpts = GetCommonPlaneTriangleRule( tri_order, rule_wts, rule_coords );

  RealT centroid[3];
  GetCommonPlaneOverlapCentroid( elem, centroid );

  RealT xTri[3];
  RealT yTri[3];
  RealT zTri[3];

  for ( int j = 0; j < elem.numPolyVert; ++j ) {
    const int next = ( j == elem.numPolyVert - 1 ) ? 0 : j + 1;
    xTri[0] = elem.overlapCoords[elem.dim * j];
    yTri[0] = elem.overlapCoords[elem.dim * j + 1];
    zTri[0] = elem.overlapCoords[elem.dim * j + 2];
    xTri[1] = elem.overlapCoords[elem.dim * next];
    yTri[1] = elem.overlapCoords[elem.dim * next + 1];
    zTri[1] = elem.overlapCoords[elem.dim * next + 2];
    xTri[2] = centroid[0];
    yTri[2] = centroid[1];
    zTri[2] = centroid[2];

    const RealT area = Area3DTri( xTri, yTri, zTri );
    if ( area <= 0. ) {
      continue;
    }

    for ( int qp = 0; qp < num_qpts; ++qp ) {
      const RealT xi = rule_coords[2 * qp];
      const RealT eta = rule_coords[2 * qp + 1];
      const RealT n0 = 1. - xi - eta;
      RealT x[3];
      x[0] = n0 * xTri[0] + xi * xTri[1] + eta * xTri[2];
      x[1] = n0 * yTri[0] + xi * yTri[1] + eta * yTri[2];
      x[2] = n0 * zTri[0] + xi * zTri[1] + eta * zTri[2];
      AccumulateCommonPlaneIntegralAtPoint( elem, x, area * rule_wts[qp], integ1, integ2 );
    }
  }
}

TRIBOL_HOST_DEVICE inline void EvalWeakFormIntegralCommonPlane( SurfaceContactElem const& elem, const PolyInteg rule,
                                                                const int tri_order, RealT* const integ1,
                                                                RealT* const integ2 )
{
  switch ( rule ) {
    case SINGLE_POINT: {
      RealT cx[3] = { 0., 0., 0. };
      GetCommonPlaneOverlapCentroid( elem, cx );
      AccumulateCommonPlaneIntegralAtPoint( elem, cx, 1.0, integ1, integ2 );
      break;
    }
    case FULL_TRI_DECOMP:
      EvalWeakFormIntegralCommonPlaneFullTri( elem, tri_order, integ1, integ2 );
      break;
    default:
#ifdef TRIBOL_USE_HOST
      SLIC_ERROR( "EvalWeakFormIntegralCommonPlane(): unsupported polygon integration rule." );
#endif
      break;
  }
}

template <>
TRIBOL_HOST_DEVICE inline void EvalWeakFormIntegral<COMMON_PLANE, SINGLE_POINT>( SurfaceContactElem const& elem,
                                                                                 RealT* const integ1,
                                                                                 RealT* const integ2 )
{
  RealT cx[3] = { 0., 0., 0. };
  GetCommonPlaneOverlapCentroid( elem, cx );
  AccumulateCommonPlaneIntegralAtPoint( elem, cx, 1.0, integ1, integ2 );
}

}  // end namespace tribol
#endif /* SRC_TRIBOL_INTEG_INTEGRATION_HPP_ */
