// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_FE_HPP_
#define SRC_TRIBOL_INTEG_FE_HPP_

#include <cmath>

#include "axom/slic.hpp"

#include "tribol/common/Parameters.hpp"

namespace tribol {

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
TRIBOL_HOST_DEVICE void GalerkinEval( const RealT* const x, const RealT pX, const RealT pY, const RealT pZ,
                                      FaceOrderType order_type, BasisEvalType basis_type, int dim, int galerkinDim,
                                      RealT* nodeVals, RealT* galerkinVal );

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
TRIBOL_HOST_DEVICE void EvalBasis( const RealT* const x, const RealT pX, const RealT pY, const RealT pZ,
                                   const int numPoints, const int vertexId, RealT& phi );

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
TRIBOL_HOST_DEVICE void WachspressBasis( const RealT* const x, const RealT pX, const RealT pY, const RealT pZ,
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
TRIBOL_HOST_DEVICE void SegmentBasis( const RealT* const x, const RealT pX, const RealT pY, const int vertexId,
                                      RealT& phi );

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
inline void InvIso( const RealT x[3], const RealT* xA, const RealT* yA, const RealT* zA, const int numNodes,
                    RealT xi[2] )
{
  if ( numNodes == 4 ) {
    constexpr int kmax = 15;
    constexpr RealT xtol = 1.E-12;

    RealT x_sol[2] = { 0., 0. };

    // derivatives of the Jacobian wrt (xi,eta)
    RealT djde_11 = 0.;
    RealT djde_x_12 = 0.25 * ( xA[0] - xA[1] + xA[2] - xA[3] );
    RealT djde_y_12 = 0.25 * ( yA[0] - yA[1] + yA[2] - yA[3] );
    RealT djde_z_12 = 0.25 * ( zA[0] - zA[1] + zA[2] - zA[3] );
    RealT djde_22 = 0.;

    // loop over newton iterations
    int k = 0;
    for ( ; k < kmax; ++k ) {
      // evaluate Jacobian
      RealT j_x_1 = 0.25 * ( xA[0] * ( 1. + x_sol[1] ) - xA[1] * ( 1. + x_sol[1] ) - xA[2] * ( 1. - x_sol[1] ) +
                             xA[3] * ( 1. - x_sol[1] ) );

      RealT j_y_1 = 0.25 * ( yA[0] * ( 1. + x_sol[1] ) - yA[1] * ( 1. + x_sol[1] ) - yA[2] * ( 1. - x_sol[1] ) +
                             yA[3] * ( 1. - x_sol[1] ) );

      RealT j_z_1 = 0.25 * ( zA[0] * ( 1. + x_sol[1] ) - zA[1] * ( 1. + x_sol[1] ) - zA[2] * ( 1. - x_sol[1] ) +
                             zA[3] * ( 1. - x_sol[1] ) );

      RealT j_x_2 = 0.25 * ( xA[0] * ( 1. + x_sol[0] ) + xA[1] * ( 1. - x_sol[0] ) - xA[2] * ( 1. - x_sol[0] ) -
                             xA[3] * ( 1. + x_sol[0] ) );

      RealT j_y_2 = 0.25 * ( yA[0] * ( 1. + x_sol[0] ) + yA[1] * ( 1. - x_sol[0] ) - yA[2] * ( 1. - x_sol[0] ) -
                             yA[3] * ( 1. + x_sol[0] ) );

      RealT j_z_2 = 0.25 * ( zA[0] * ( 1. + x_sol[0] ) + zA[1] * ( 1. - x_sol[0] ) - zA[2] * ( 1. - x_sol[0] ) -
                             zA[3] * ( 1. + x_sol[0] ) );

      // evaluate the residual
      RealT f_x =
          x[0] -
          0.25 * ( ( 1. + x_sol[0] ) * ( 1. + x_sol[1] ) * xA[0] + ( 1. - x_sol[0] ) * ( 1. + x_sol[1] ) * xA[1] +
                   ( 1. - x_sol[0] ) * ( 1. - x_sol[1] ) * xA[2] + ( 1. + x_sol[0] ) * ( 1. - x_sol[1] ) * xA[3] );

      RealT f_y =
          x[1] -
          0.25 * ( ( 1. + x_sol[0] ) * ( 1. + x_sol[1] ) * yA[0] + ( 1. - x_sol[0] ) * ( 1. + x_sol[1] ) * yA[1] +
                   ( 1. - x_sol[0] ) * ( 1. - x_sol[1] ) * yA[2] + ( 1. + x_sol[0] ) * ( 1. - x_sol[1] ) * yA[3] );

      RealT f_z =
          x[2] -
          0.25 * ( ( 1. + x_sol[0] ) * ( 1. + x_sol[1] ) * zA[0] + ( 1. - x_sol[0] ) * ( 1. + x_sol[1] ) * zA[1] +
                   ( 1. - x_sol[0] ) * ( 1. - x_sol[1] ) * zA[2] + ( 1. + x_sol[0] ) * ( 1. - x_sol[1] ) * zA[3] );

      // compute J' * J
      RealT JTJ_11 = j_x_1 * j_x_1 + j_y_1 * j_y_1 + j_z_1 * j_z_1;
      RealT JTJ_12 = j_x_1 * j_x_2 + j_y_1 * j_y_2 + j_z_1 * j_z_2;
      // RealT JTJ_21 = JTJ_12;
      RealT JTJ_22 = j_x_2 * j_x_2 + j_y_2 * j_y_2 + j_z_2 * j_z_2;
      ;

      // compute J' * F
      RealT JTF_1 = j_x_1 * f_x + j_y_1 * f_y + j_z_1 * f_z;
      RealT JTF_2 = j_x_2 * f_x + j_y_2 * f_y + j_z_2 * f_z;

      // for first few steps don't do exact Newton.
      RealT cm_11 = JTJ_11;  //- (djde_11 * f_x + djde_11 * f_y + djde_11 * f_z);
      RealT cm_12 = JTJ_12;  //- (djde_x_12 * f_x + djde_y_12 * f_y + djde_z_12 * f_z);
      RealT cm_21 = cm_12;
      RealT cm_22 = JTJ_22;  //- (djde_22 * f_x + djde_22 * f_y + djde_22 * f_z);

      // do exact Newton for steps beyond first few
      if ( k > 2 )  // set to 2 per mortar method testing
      {
        cm_11 += -( djde_11 * f_x + djde_11 * f_y + djde_11 * f_z );
        cm_12 += -( djde_x_12 * f_x + djde_y_12 * f_y + djde_z_12 * f_z );
        cm_21 = cm_12;
        cm_22 += -( djde_22 * f_x + djde_22 * f_y + djde_22 * f_z );
      }

      RealT detI = 1. / ( cm_11 * cm_22 - cm_12 * cm_21 );

      RealT cmi_11 = cm_22 * detI;
      RealT cmi_22 = cm_11 * detI;
      RealT cmi_12 = -cm_12 * detI;
      RealT cmi_21 = -cm_21 * detI;

      RealT dxi_1 = cmi_11 * JTF_1 + cmi_12 * JTF_2;
      RealT dxi_2 = cmi_21 * JTF_1 + cmi_22 * JTF_2;

      x_sol[0] += dxi_1;
      x_sol[1] += dxi_2;

      RealT abs_dxi_1 = std::abs( dxi_1 );
      RealT abs_dxi_2 = std::abs( dxi_2 );

      if ( abs_dxi_1 <= xtol && abs_dxi_2 <= xtol ) {
        xi[0] = x_sol[0];
        xi[1] = x_sol[1];

        //       check to make sure point is inside isoparametric quad_wt
#if !defined( TRIBOL_USE_ENZYME )
        bool in_quad = true;
#endif
        if ( std::abs( xi[0] ) > 1. || std::abs( xi[1] ) > 1. ) {
          if ( std::abs( xi[0] ) > 1. + 100. * xtol ||
               std::abs( xi[1] ) > 1. + 100. * xtol )  // should have some tolerance dependent conv tol?
          {
#if !defined( TRIBOL_USE_ENZYME )
            in_quad = false;
#endif
          } else {
            xi[0] = std::min( xi[0], 1. );
            xi[1] = std::min( xi[1], 1. );
            xi[0] = std::max( xi[0], -1. );
            xi[1] = std::max( xi[1], -1. );
          }
        }

#if !defined( TRIBOL_USE_ENZYME )
        SLIC_WARNING_IF( !in_quad, "InvIso(): (xi,eta) coordinate does not lie inside isoparametric quad." );
#endif

        break;
      }
    }

#if !defined( TRIBOL_USE_ENZYME )
    SLIC_ERROR_IF( k == kmax, "InvIso: Newtons method did not converge." );
#endif

  } else if ( numNodes == 3 ) {
    // use area (barycentric) coords to get xi, eta
    RealT a[3] = { xA[1] - xA[0], yA[1] - yA[0], zA[1] - zA[0] };
    RealT b[3] = { xA[2] - xA[0], yA[2] - yA[0], zA[2] - zA[0] };
    RealT a_cr_b[3] = { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
    RealT area = std::sqrt( a_cr_b[0] * a_cr_b[0] + a_cr_b[1] * a_cr_b[1] + a_cr_b[2] * a_cr_b[2] );
    RealT c[3] = { x[0] - xA[0], x[1] - yA[0], x[2] - zA[0] };

    RealT c_cr_b[3] = { c[1] * b[2] - c[2] * b[1], c[2] * b[0] - c[0] * b[2], c[0] * b[1] - c[1] * b[0] };
    RealT xi_area = std::sqrt( c_cr_b[0] * c_cr_b[0] + c_cr_b[1] * c_cr_b[1] + c_cr_b[2] * c_cr_b[2] );
    xi[0] = xi_area / area;

    RealT a_cr_c[3] = { a[1] * c[2] - a[2] * c[1], a[2] * c[0] - a[0] * c[2], a[0] * c[1] - a[1] * c[0] };
    RealT eta_area = std::sqrt( a_cr_c[0] * a_cr_c[0] + a_cr_c[1] * a_cr_c[1] + a_cr_c[2] * a_cr_c[2] );
    xi[1] = eta_area / area;
  } else {
#if !defined( TRIBOL_USE_ENZYME )
    SLIC_ERROR( "Invalid number of nodes: " << numNodes );
#endif
  }
}

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
void FwdMapLinTri( const RealT xi[2], RealT xa[3], RealT ya[3], RealT za[3], RealT x[3] );

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
void FwdMapLinQuad( const RealT xi[2], RealT xa[4], RealT ya[4], RealT za[4], RealT x[3] );

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
inline void LinIsoTriShapeFunc( const RealT xi, const RealT eta, const int a, RealT& phi )
{
  switch ( a ) {
    case 0:
      phi = 1 - xi - eta;
      break;
    case 1:
      phi = xi;
      break;
    case 2:
      phi = eta;
      break;
    default:
#if !defined( TRIBOL_USE_ENZYME )
      SLIC_ERROR( "LinIsoTriShapeFunc: node id is not between 0 and 2." );
#endif
      break;
  }

  return;
}

/*!
 *
 * \brief returns the shape functions for a three node isoparametric triangle evaluated at (xi[0], xi[1])
 *
 * \param [in] xi array of length 2 holding parent coordinates
 * \param [in,out] phi shape function evaluation (array of length 3)
 */
inline void LinIsoTriShapeFunc( const RealT* xi, RealT* phi )
{
  phi[0] = 1.0 - xi[0] - xi[1];
  phi[1] = xi[0];
  phi[2] = xi[1];
}

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
inline void LinIsoQuadShapeFunc( const RealT xi, const RealT eta, const int a, RealT& phi )
{
  RealT xi_node, eta_node;
  switch ( a ) {
    case 0:
      xi_node = 1.;
      eta_node = 1.;
      break;
    case 1:
      xi_node = -1.;
      eta_node = 1.;
      break;
    case 2:
      xi_node = -1.;
      eta_node = -1.;
      break;
    case 3:
      xi_node = 1.;
      eta_node = -1.;
      break;
    default:
#if !defined( TRIBOL_USE_ENZYME )
      SLIC_ERROR( "LinIsoQuadShapeFunc: node id is not between 0 and 3." );
#endif
      return;
  }

  phi = 0.25 * ( 1. + xi_node * xi ) * ( 1. + eta_node * eta );

#if !defined( TRIBOL_USE_ENZYME )
  SLIC_ERROR_IF( phi > 1.0 || phi < 0.0, "LinIsoQuadShapeFunc: phi is " << phi << " not between 0. and 1." );
#endif

  return;
}

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
void DetJQuad( const RealT xi, const RealT eta, const RealT* x, const int dim, RealT& detJ );

}  // namespace tribol

#endif /* SRC_TRIBOL_INTEG_FE_HPP_ */
