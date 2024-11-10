// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/utils/Math.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/geom/GeomUtilities.hpp"

#include "axom/slic.hpp" 

#include <algorithm>
#include <cmath>

namespace tribol
{

TRIBOL_HOST_DEVICE int GetNumFaceNodes( int dim, FaceOrderType order_type )
{
   // SRW consider consolidating this to take a tribol topology and 
   // order for consistency
   int numNodes = 0;
   switch (order_type)
   {
      case LINEAR :
         numNodes = (dim == 3) ? 4 : 2; // segments and quads
         break;
      default :
#ifdef TRIBOL_USE_HOST
         SLIC_ERROR("GetNumFaceNodes(): order_type not supported.");
#endif
         break;
   } 
   return numNodes;
}

TRIBOL_HOST_DEVICE void GalerkinEval( const RealT* const x, 
                                      const RealT pX, const RealT pY, const RealT pZ, 
                                      FaceOrderType order_type, BasisEvalType basis_type, 
                                      int dim, int galerkinDim, RealT* nodeVals, RealT* galerkinVal )
{
#ifdef TRIBOL_USE_HOST
   SLIC_ERROR_IF(x==nullptr, "GalerkinEval(): input pointer, x, is NULL.");
   SLIC_ERROR_IF(nodeVals==nullptr, "GalerkinEval(): input pointer, nodeVals, is NULL.");
   SLIC_ERROR_IF(galerkinVal==nullptr, "GalerkinEval(): input/output pointer, galerkinVal, is NULL.");
   SLIC_ERROR_IF(galerkinDim<1, "GalerkinEval(): scalar approximations not yet supported." );
#endif

   int numNodes = GetNumFaceNodes( dim, order_type );
   switch (basis_type)
   {
      case PHYSICAL :
         for (int nd=0; nd<numNodes; ++nd)
         {
            RealT phi = 0.;
            EvalBasis( x, pX, pY, pZ, numNodes, nd, phi );
            for (int i=0; i<galerkinDim; ++i)
            {
               galerkinVal[i] += nodeVals[i+nd*galerkinDim] * phi;
            } 
         }
         break;
      default :
#ifdef TRIBOL_USE_HOST
         SLIC_ERROR( "GalerkinEval(): basis_type = PARENT not yet supported." );
#endif
         break;
   }
}

TRIBOL_HOST_DEVICE void EvalBasis( const RealT* const x, 
                                   const RealT pX, const RealT pY, const RealT pZ, 
                                   const int numPoints, const int vertexId, 
                                   RealT& phi )
{
   if (numPoints > 2)
   {
      WachspressBasis( x, pX, pY, pZ, numPoints, vertexId, phi );
   }
   else if (numPoints == 2)
   {
      SegmentBasis( x, pX, pY, numPoints, vertexId, phi );
   }
   else
   {
#ifdef TRIBOL_USE_HOST
      SLIC_ERROR("EvalBasis: invalid numPoints argument.");
#endif
   }
   return;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void SegmentBasis( const RealT* const x, 
                                      const RealT pX, const RealT pY,
                                      const int numPoints, const int vertexId, 
                                      RealT& phi )
{
#ifdef TRIBOL_USE_HOST
    SLIC_ERROR_IF(numPoints != 2, "SegmentBasis: numPoints is " << numPoints <<
                  " but should be 2.");

    // note, vertexId is the index, 0 or 1.
    SLIC_ERROR_IF(vertexId > numPoints-1, "SegmentBasis: vertexId is " << vertexId << 
                  " but should be 0 or 1.");
#endif

   // compute length of segment
   RealT vx = x[numPoints*1] - x[numPoints*0];
   RealT vy = x[numPoints*1+1] - x[numPoints*0+1];
   RealT lambda = magnitude( vx, vy );

   // compute the magnitude of the vector <pX,pY> - <x[vertexId],y[vertexId]>
   RealT wx = pX - x[ numPoints*vertexId ];
   RealT wy = pY - x[ numPoints*vertexId+1 ];

   RealT magW = magnitude( wx, wy );

   phi = 1.0 / lambda * (lambda - magW); // this calculation is inverted, (phi_a is actually phi_b and vice versa)

   // TODO verify this code as a bugfix to fix flipping of nodes a and b
   // when evaluating basis. Suppress error for now.
   //if (std::abs(lambda-magW)/lambda < 1.E-2)
   //{
   //   phi=1.;
   //}
   //else if (magW<1.e-5)
   //{
   //   phi=0.;
   //}
   //else
   //{
   //   //phi = 1.0 / lambda * (lambda - magW); // this calculation is inverted, (phi_a is actually phi_b and vice versa)
   //                                           // this will shift nodal contributions over one node
   //   phi = 1.0 / lambda * magW;
   //}

   //if (phi > 1.0 || phi < 0.0)
   //{
   //   SLIC_INFO("(x0,y0) and (x1,y1): " << "(" << x[0] << ", " << x[1] << "), " << "(" << x[2] << ", " << x[3] << ").");
   //   SLIC_INFO("(px,py): " << "(" << pX << ", " << pY << ")");
   //}
   //SLIC_ERROR_IF(phi > 1.0 || phi < 0.0, "SegmentBasis: phi is " << phi << " not between 0. and 1." );

   return;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void WachspressBasis( const RealT* const x, 
                                         const RealT pX, const RealT pY, const RealT pZ, 
                                         const int numPoints, const int vertexId, RealT& phi )
{
#ifdef TRIBOL_USE_HOST
    SLIC_ERROR_IF(numPoints<3, "WachspressBasis: numPoints < 3.");
#endif

   // first compute the areas of all the triangles formed by the i-1,i,i+1 vertices.
   // These consist of all the numerators in the Wachspress formulation
   // NOTE: this limits the routine to 4 noded quadrilaterals
   constexpr int max_nodes_per_elem = 4;
   RealT triVertArea[ max_nodes_per_elem ];
   for (int i=0; i<numPoints; ++i)
   {
      // determine the i-1, i, i+1 vertices
      int vId = i; 
      int vIdMinus = (vId == 0) ? (numPoints-1) : (vId-1);
      int vIdPlus = (vId == (numPoints-1)) ? 0 : (vId+1);

      // construct segment between i-1,i and i-1,i+1
      RealT vx = x[ 3*vId ]   - x[ 3*vIdMinus ];
      RealT vy = x[ 3*vId+1 ] - x[ 3*vIdMinus+1 ];
      RealT vz = x[ 3*vId+2 ] - x[ 3*vIdMinus+2 ];

      RealT wx = x[ 3*vIdPlus ] -   x[ 3*vIdMinus ];
      RealT wy = x[ 3*vIdPlus+1 ] - x[ 3*vIdMinus+1 ];
      RealT wz = x[ 3*vIdPlus+2 ] - x[ 3*vIdMinus+2 ];

      // take the cross product between v and w to get the normal, and then obtain the 
      // area from the normal's magnitude
      RealT nX = (vy * wz) - (vz * wy);
      RealT nY = (vz * wx) - (vx * wz);
      RealT nZ = (vx * wy) - (vy * wx);
  
      triVertArea[i] = 0.5 * magnitude( nX, nY, nZ );

   }

   // second, compute the areas of all triangles formed using edge segment vertices
   // and the specified interior point (pX,pY,pZ)
   RealT triPointArea[ max_nodes_per_elem ];
   for (int i=0; i<numPoints; ++i)
   {
      // determine the i,i+1 edge segment 
      int vId = i;
      int vIdPlus = (vId == (numPoints-1)) ? 0 : (vId+1);

      // construct segments between i+1,i and p,i 
      RealT vx = x[ 3*vIdPlus ]   - x[ 3*vId ];
      RealT vy = x[ 3*vIdPlus+1 ] - x[ 3*vId+1 ];
      RealT vz = x[ 3*vIdPlus+2 ] - x[ 3*vId+2 ];

      RealT wx = pX - x[ 3*vId ];
      RealT wy = pY - x[ 3*vId+1 ];
      RealT wz = pZ - x[ 3*vId+2 ];

      // take the cross product between v and w to get the normal, and then obtain the 
      // area from the normal's magnitude
      RealT nX = (vy * wz) - (vz * wy);
      RealT nY = (vz * wx) - (vx * wz);
      RealT nZ = (vx * wy) - (vy * wx);
  
      triPointArea[i] = 0.5 * magnitude( nX, nY, nZ );

   }

   // third, compute all of the weights per Wachspress formulation
   RealT weight[ max_nodes_per_elem ];
   RealT myWeight;
   RealT weightSum = 0.;
   for (int i=0; i<numPoints; ++i)
   {
      int vId = i; 
      int vIdMinus = (vId == 0) ? (numPoints-1) : (vId-1);

      weight[vId] = triVertArea[vId] / (triPointArea[vIdMinus] * triPointArea[vId]);

      weightSum += weight[vId];

      if (i == vertexId) 
      {
         myWeight = weight[vId];
      }
      
   }

   phi = myWeight / weightSum;

#ifdef TRIBOL_USE_HOST
   if (phi <= 0. || phi > 1.)
   {
      SLIC_ERROR("Wachspress Basis: phi is not between 0 and 1.");
   }
#endif

   return;
}

//------------------------------------------------------------------------------
void InvIso( const RealT  x[3], 
             const RealT* xA,
             const RealT* yA,
             const RealT* zA,
             const int numNodes,
             RealT  xi[2] )
{

#if !defined(TRIBOL_USE_ENZYME)
   SLIC_ERROR_IF(numNodes!=4, "InvIso: routine only for 4 node quads.");
#endif

   bool convrg = false;
   int kmax = 15;
   RealT xtol = 1.E-12;

   RealT x_sol[2] = {0., 0.};

   // derivatives of the Jacobian wrt (xi,eta)
   RealT djde_11   = 0.;
   RealT djde_x_12 = 0.25 * (xA[0] - xA[1] + xA[2] - xA[3]);
   RealT djde_y_12 = 0.25 * (yA[0] - yA[1] + yA[2] - yA[3]);
   RealT djde_z_12 = 0.25 * (zA[0] - zA[1] + zA[2] - zA[3]);
   RealT djde_22   = 0.;

   // loop over newton iterations
   for (int k = 0; k < kmax; ++k)
   {
      // evaluate Jacobian
      RealT j_x_1 = 0.25 * (xA[0] * (1. + x_sol[1]) - xA[1] * 
                   (1. + x_sol[1]) - xA[2] * (1. - x_sol[1]) + 
                   xA[3] * (1. - x_sol[1]));

      RealT j_y_1 = 0.25 * (yA[0] * (1. + x_sol[1]) - yA[1] * 
                   (1. + x_sol[1]) - yA[2] * (1. - x_sol[1]) + 
                   yA[3] * (1. - x_sol[1]));

      RealT j_z_1 = 0.25 * (zA[0] * (1. + x_sol[1]) - zA[1] * 
                   (1. + x_sol[1]) - zA[2] * (1. - x_sol[1]) + 
                   zA[3] * (1. - x_sol[1]));

      RealT j_x_2 = 0.25 * (xA[0] * (1. + x_sol[0]) + xA[1] * 
                   (1. - x_sol[0]) - xA[2] * (1. - x_sol[0]) - 
                   xA[3] * (1. + x_sol[0]));

      RealT j_y_2 = 0.25 * (yA[0] * (1. + x_sol[0]) + yA[1] * 
                   (1. - x_sol[0]) - yA[2] * (1. - x_sol[0]) - 
                   yA[3] * (1. + x_sol[0]));

      RealT j_z_2 = 0.25 * (zA[0] * (1. + x_sol[0]) + zA[1] * 
                   (1. - x_sol[0]) - zA[2] * (1. - x_sol[0]) - 
                   zA[3] * (1. + x_sol[0]));

      // evaluate the residual
      RealT f_x = x[0] - 0.25 * ((1. + x_sol[0]) * (1. + x_sol[1]) * xA[0]
                 + (1. - x_sol[0]) * (1. + x_sol[1]) * xA[1]
                 + (1. - x_sol[0]) * (1. - x_sol[1]) * xA[2]
                 + (1. + x_sol[0]) * (1. - x_sol[1]) * xA[3]);

      RealT f_y = x[1] - 0.25 * ((1. + x_sol[0]) * (1. + x_sol[1]) * yA[0]
                 + (1. - x_sol[0]) * (1. + x_sol[1]) * yA[1]
                 + (1. - x_sol[0]) * (1. - x_sol[1]) * yA[2]
                 + (1. + x_sol[0]) * (1. - x_sol[1]) * yA[3]);

      RealT f_z = x[2] - 0.25 * ((1. + x_sol[0]) * (1. + x_sol[1]) * zA[0]
                 + (1. - x_sol[0]) * (1. + x_sol[1]) * zA[1]
                 + (1. - x_sol[0]) * (1. - x_sol[1]) * zA[2]
                 + (1. + x_sol[0]) * (1. - x_sol[1]) * zA[3]);

      // compute J' * J
      RealT JTJ_11 = j_x_1 * j_x_1 + j_y_1 * j_y_1 + j_z_1 * j_z_1;
      RealT JTJ_12 = j_x_1 * j_x_2 + j_y_1 * j_y_2 + j_z_1 * j_z_2;
      //RealT JTJ_21 = JTJ_12;
      RealT JTJ_22 = j_x_2 * j_x_2 + j_y_2 * j_y_2 + j_z_2 * j_z_2;;

      // compute J' * F
      RealT JTF_1 = j_x_1 * f_x + j_y_1 * f_y + j_z_1 * f_z;
      RealT JTF_2 = j_x_2 * f_x + j_y_2 * f_y + j_z_2 * f_z;

      // for first few steps don't do exact Newton.
      RealT cm_11 = JTJ_11; //- (djde_11 * f_x + djde_11 * f_y + djde_11 * f_z);
      RealT cm_12 = JTJ_12; //- (djde_x_12 * f_x + djde_y_12 * f_y + djde_z_12 * f_z);
      RealT cm_21 = cm_12;
      RealT cm_22 = JTJ_22; //- (djde_22 * f_x + djde_22 * f_y + djde_22 * f_z);

      // do exact Newton for steps beyond first few
      if (k > 2)  // set to 2 per mortar method testing 
      {
       cm_11 += - (djde_11 * f_x + djde_11 * f_y + djde_11 * f_z);
       cm_12 += - (djde_x_12 * f_x + djde_y_12 * f_y + djde_z_12 * f_z);
       cm_21 = cm_12;
       cm_22 += - (djde_22 * f_x + djde_22 * f_y + djde_22 * f_z);
      }

      RealT detI = 1. / (cm_11 * cm_22 - cm_12 * cm_21);

      RealT cmi_11 = cm_22 * detI;
      RealT cmi_22 = cm_11 * detI;
      RealT cmi_12 = -cm_12 * detI;
      RealT cmi_21 = -cm_21 * detI;

      RealT dxi_1 = cmi_11 * JTF_1 + cmi_12 * JTF_2;
      RealT dxi_2 = cmi_21 * JTF_1 + cmi_22 * JTF_2;

      x_sol[0] += dxi_1;
      x_sol[1] += dxi_2;

      RealT abs_dxi_1 = std::abs(dxi_1);
      RealT abs_dxi_2 = std::abs(dxi_2);

      if (abs_dxi_1 <= xtol && abs_dxi_2 <= xtol)
      {
         convrg = true;
         xi[0] = x_sol[0];
         xi[1] = x_sol[1];

//       check to make sure point is inside isoparametric quad
         bool in_quad = true;
         if(std::abs(xi[0]) > 1. || std::abs(xi[1]) > 1.) 
         {
           if(std::abs(xi[0]) > 1.+100*xtol || std::abs(xi[1]) > 1.+100*xtol) // should have some tolerance dependent conv tol?
           {
              in_quad = false;
           }
           else 
           {
           xi[0] = std::min(xi[0],1.);
           xi[1] = std::min(xi[1],1.);
           xi[0] = std::max(xi[0],-1.);
           xi[1] = std::max(xi[1],-1.);
           }
         }

#if !defined(TRIBOL_USE_ENZYME)
         SLIC_ERROR_IF(!in_quad, "InvIso(): (xi,eta) coordinate does not lie " << 
                       "inside isoparametric quad.");
#endif

         return; 
      }

   }

#if !defined(TRIBOL_USE_ENZYME)
   SLIC_ERROR_IF(!convrg, "InvIso: Newtons method did not converge.");
#endif

   return;
}

//------------------------------------------------------------------------------
void FwdMapLinQuad( const RealT xi[2],
                    RealT xa[4],
                    RealT ya[4],
                    RealT za[4],
                    RealT x[3] )
{
   // initialize output array
   initRealArray( &x[0], 3, 0. );

   // obtain shape function evaluations at (xi,eta)
   RealT phi[4] = { 0., 0., 0., 0. };
   LinIsoQuadShapeFunc( xi[0], xi[1], 0, phi[0] );
   LinIsoQuadShapeFunc( xi[0], xi[1], 1, phi[1] );
   LinIsoQuadShapeFunc( xi[0], xi[1], 2, phi[2] );
   LinIsoQuadShapeFunc( xi[0], xi[1], 3, phi[3] );

   for (int j=0; j<4; ++j)
   {
      x[0] += xa[j] * phi[j];
      x[1] += ya[j] * phi[j];
      x[2] += za[j] * phi[j];
   }
   return;
}

//------------------------------------------------------------------------------
void FwdMapLinTri( const RealT xi[2],
                   RealT xa[3],
                   RealT ya[3],
                   RealT za[3],
                   RealT x[3] )
{
   // initialize output array
   initRealArray( &x[0], 3, 0. );

   // obtain shape function evaluations at (xi,eta)
   RealT phi[3] = { 0., 0., 0. };
   LinIsoTriShapeFunc( xi[0], xi[1], 0, phi[0] );
   LinIsoTriShapeFunc( xi[0], xi[1], 1, phi[1] );
   LinIsoTriShapeFunc( xi[0], xi[1], 2, phi[2] );

   for (int j=0; j<3; ++j)
   {
      x[0] += xa[j] * phi[j];
      x[1] += ya[j] * phi[j];
      x[2] += za[j] * phi[j];
   }
   return;
}

//------------------------------------------------------------------------------
void LinIsoTriShapeFunc( const RealT xi, 
                         const RealT eta,
                         const int a,
                         RealT& phi )
{
   switch (a)
   {
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
#if !defined(TRIBOL_USE_ENZYME)
         SLIC_ERROR("LinIsoTriShapeFunc: node id is not between 0 and 2.");
#endif
         break;
   }

   return;
}

//------------------------------------------------------------------------------
void LinIsoQuadShapeFunc( const RealT xi, 
                          const RealT eta,
                          const int a,
                          RealT& phi )
{
   RealT xi_node, eta_node;
   switch (a)
   {
      case 0:
         xi_node  = 1.;
         eta_node = 1.;
         break;
      case 1:
         xi_node  = -1.;
         eta_node = 1.;
         break;
      case 2:
         xi_node  = -1.;
         eta_node = -1.;
         break;
      case 3:
         xi_node  = 1.;
         eta_node = -1.;
         break;
      default:
#if !defined(TRIBOL_USE_ENZYME)
         SLIC_ERROR("LinIsoQuadShapeFunc: node id is not between 0 and 3.");
#endif
         return;
   }

   phi = 0.25 * (1. + xi_node * xi) * ( 1. + eta_node * eta);

#if !defined(TRIBOL_USE_ENZYME)
   SLIC_ERROR_IF(phi > 1.0 || phi < 0.0, "LinIsoQuadShapeFunc: phi is " << phi << " not between 0. and 1." );
#endif

   return;
}

//------------------------------------------------------------------------------
void DetJQuad( const RealT xi,
               const RealT eta,
               const RealT* x,
               const int dim,
               RealT& detJ )
{
   
   RealT J[4] = { 0., 0., 0., 0. }; // column major ordering

   // loop over nodes
   for (int a=0; a<4; ++a)
   {
      // determine (xi,eta) coord of node a
      RealT xi_node, eta_node;
      switch (a)
      {
         case 0:
            xi_node  = 1.;
            eta_node = 1.;
            break;
         case 1:
            xi_node  = -1.;
            eta_node = 1.;
            break;
         case 2:
            xi_node  = -1.;
            eta_node = -1.;
            break;
         case 3:
            xi_node  = 1.;
            eta_node = -1.;
            break;
      }
   
      // loop over 2D coords
      for (int j=0; j<2; ++j)
      {
         J[0+j] += 0.25 * x[dim*a+j] * xi_node * (1. + eta_node * eta);
         J[2+j] += 0.25 * x[dim*a+j] * eta_node * (1. + xi_node * xi); 
      }
   }

   detJ = J[0] * J[3] - J[2] * J[1];

   // this is a hack, but I can't guarantee the correct orientation of the 
   // overlap vertices with respect to some global notion of up. This 
   // calculation will calculate the correct absolute value.
   detJ = (detJ <= 0) ? -detJ : detJ;

   return;
}

//------------------------------------------------------------------------------

} // end of namespace "tribol"
