// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_INTEG_FE_HPP_
#define SRC_INTEG_FE_HPP_

#include "tribol/common/Parameters.hpp"

namespace tribol
{

/*!
 *
 * \brief wrapper routine for evaluation of 2D or 3D shape functions on projected 
 *        surface element topologies
 *
 * \param [in] x pointer to array of stacked (xyz) coordinates for 2D or 3D surface face/edge vertices
 * \param [in] pX x-coordinate of point at which to evaluate shape function
 * \param [in] pX y-coordinate of point at which to evaluate shape function
 * \param [in] pZ z-coordinate of point at which to evaluate shape function (3D only)
 * \param [in] order_type the order of the Lagrangian finite element
 * \param [in] basis_type either current configuration physical or canonical reference basis
 * \param [in] dim the dimension of the overall contact problem
 * \param [in] galerkinDim the vector-dim of the Galerkin coefficients
 * \param [in] nodeVals the nodal values for the Galerkin approximation
 * \param [in,out] galerkinVal the Galerkin approximation
 *
 * \pre z is nullptr for 2D
 *
 */
TRIBOL_HOST_DEVICE void GalerkinEval( const RealT* const x, 
                                      const RealT pX, const RealT pY, const RealT pZ, 
                                      FaceOrderType order_type, BasisEvalType basis_type, 
                                      int dim, int galerkinDim, RealT* nodeVals, RealT* galerkinVal );

/*!
 *
 * \brief wrapper routine for evaluation of 2D or 3D shape functions on projected 
 *        surface element topologies
 *
 * \param [in] x pointer to array of stacked (xyz) coordinates for 2D or 3D surface face/edge vertices
 * \param [in] pX x-coordinate of point at which to evaluate shape function
 * \param [in] pX y-coordinate of point at which to evaluate shape function
 * \param [in] pZ z-coordinate of point at which to evaluate shape function (3D only)
 * \param [in] numPoints number of vertices in x,y,z arrays
 * \param [in] vertexId node id whose shape function is to be evaluated
 * \param [in,out] phi shape function evaluation
 *
 * \pre z is nullptr for 2D
 *
 */
TRIBOL_HOST_DEVICE void EvalBasis( const RealT* const x, 
                                   const RealT pX, const RealT pY, const RealT pZ, 
                                   const int numPoints, const int vertexId, 
                                   RealT& phi );

/*!
 *
 * \brief evaluates Wachspress basis functions on 4-node quadrilateral faces
 *
 * \param [in] x pointer to array of stacked (xyz) coordinates of quad's vertices
 * \param [in] pX x-coordinate of point at which to evaluate shape function
 * \param [in] pX y-coordinate of point at which to evaluate shape function
 * \param [in] pZ x-coordinate of point at which to evaluate shape function
 * \param [in] numPoints number of vertices in x,y,z arrays
 * \param [in] vertexId node id whose shape function is to be evaluated
 * \param [in,out] phi shape function evaluation
 *
 * \pre Input argument x is expected to be full 3D coordinate array
 *
 * \note This is implicitly a 3D routine
 *
 */
TRIBOL_HOST_DEVICE void WachspressBasis( const RealT* const x, 
                                         const RealT pX, const RealT pY, const RealT pZ, 
                                         const int numPoints, const int vertexId, RealT& phi );

/*!
 *
 * \brief evaluates standard linear shape functions on 2-node segments
 *
 * \param [in] x pointer to array of stacked (xy) coordinates for 2D segment
 * \param [in] pX x-coordinate of point at which to evaluate shape function
 * \param [in] pX y-coordinate of point at which to evaluate shape function
 * \param [in] vertexId node id whose shape function is to be evaluated
 * \param [in,out] phi shape function evaluation
 *
 * \note This is implicitly a 2D routine
 *
 */
TRIBOL_HOST_DEVICE void SegmentBasis( const RealT* const x, 
                                      const RealT pX, const RealT pY,
                                      const int vertexId, RealT& phi );

/*!
 *
 * \brief performs the inverse isoparametric mapping to obtain a (xi,eta) 
 *        coordinate in parent space associated with a point in physical space
 *
 * \param [in] x array of (x,y,z) coordinates of a point in physical space
 * \param [in] xA pointer to array of stacked nodal x-coordinates
 * \param [in] yA pointer to array of stacked nodal y-coordinates
 * \param [in] zA pointer to array of stacked nodal z-coordinates
 * \param [in] numNodes number of nodes for a given finite element
 * \param [in,out] xi (xi,eta) coordinates in parent space 
 *
 * \pre xA, yA, and zA are pointer to arrays of length, numNodes
 *
 * \note This routine works in 2D or 3D. In 2D, zA is a nullptr and 
 *       x[2] is equal to 0.
 *
 */
void InvIso( const RealT  x[3], 
             const RealT* xA,
             const RealT* yA,
             const RealT* zA,
             const int numNodes,
             RealT  xi[2] );

/*!
 *
 * \brief performs a foward linear map for a linear, four node quadrilateral
 *
 * \param [in] xi (xi,eta) point in parent space
 * \param [in] xa nodal x-coordinates
 * \param [in] ya nodal y-coordinates
 * \param [in] za nodal z-coordinates
 * \param [in,out] x corresponding point in physical space
 *
 *
 */
void FwdMapLinQuad( const RealT xi[2],
                    RealT xa[4],
                    RealT ya[4],
                    RealT za[4],
                    RealT x[3] );

/*!
 *
 * \brief performs a foward linear map for a linear, three node triangle 
 *
 * \param [in] xi (xi,eta) point in parent space
 * \param [in] xa nodal x-coordinates
 * \param [in] ya nodal y-coordinates
 * \param [in] za nodal z-coordinates
 * \param [in,out] x corresponding point in physical space
 *
 *
 */
void FwdMapLinTri( const RealT xi[2],
                   RealT xa[3],
                   RealT ya[3],
                   RealT za[3],
                   RealT x[3] );

/*!
 *
 * \brief returns the shape function at node a for a linear,
 *        four node isoparametric quadrilateral evaluated at (xi,eta)
 *
 * \param [in] xi first parent coordinate of evaluation point
 * \param [in] eta second parent coordinate of evaluation point
 * \param [in] a node id of shape function
 * \param [in,out] phi shape function evaluation
 *
 * \pre input argument, a, ranges from 0-3.
 *
 *
 */
void LinIsoQuadShapeFunc( const RealT xi, 
                          const RealT eta,
                          const int a,
                          RealT& phi );

/*!
 *
 * \brief returns the shape function at node a for a linear,
 *        three node isoparametric triangle evaluated at (xi,eta)
 *
 * \param [in] xi first parent coordinate of evaluation point
 * \param [in] eta second parent coordinate of evaluation point
 * \param [in] a node id of shape function
 * \param [in,out] phi shape function evaluation
 *
 * \pre input argument, a, ranges from 0-2.
 *
 * \note this routine uses shape functions generated from 
 *       collapsing a four node quadrilateral. The parent coordinates 
 *       of each node are as follows (-1,-1), (1,-1), (0,1).
 *
 */
void LinIsoTriShapeFunc( const RealT xi, 
                         const RealT eta,
                         const int a,
                         RealT& phi );

/*!
 *
 * \brief returns the determinant of the Jacobian for a four node 
 *        quadrilateral
 *
 * \param [in] xi first parent coordinate of evaluation point
 * \param [in] eta second parent coordinate of evaluation point
 * \param [in] x pointer to stacked array of nodal coordinates
 * \param [in] dim dimension of the coordinate data
 * \param [in,out] detJ determinant of the Jacobian of transformation
 *
 * \note The input argument x may be stacked 2D or 3D coordinates, 
 *       indicated by dim, respectively.
 *       This routine ignores the z-dimension, assuming that the 
 *       four node quad is planar, which is the case for contact 
 *       integrals
 *
 */
void DetJQuad( const RealT xi,
               const RealT eta,
               const RealT* x,
               const int dim,
               RealT& detJ );

} // end of namespace "tribol"

#endif /* SRC_INTEG_FE_HPP_ */
