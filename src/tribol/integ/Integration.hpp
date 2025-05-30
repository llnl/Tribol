// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_INTEGRATION_HPP_
#define SRC_TRIBOL_INTEG_INTEGRATION_HPP_

#include "tribol/common/Parameters.hpp"

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
TRIBOL_HOST_DEVICE void EvalWeakFormIntegral( SurfaceContactElem const& elem, RealT* const integ1,
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

}  // end namespace tribol
#endif /* SRC_TRIBOL_INTEG_INTEGRATION_HPP_ */
