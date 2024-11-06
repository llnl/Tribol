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

int GetNumFaceNodes( int dim, FaceOrderType order_type )
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
         SLIC_ERROR("GetNumFaceNodes(): order_type not supported.");
         break;
   } 
   return numNodes;
}

void GalerkinEval( const real* const RESTRICT x, 
                   const real pX, const real pY, const real pZ, 
                   FaceOrderType order_type, BasisEvalType basis_type, 
                   int dim, int galerkinDim, real* nodeVals, real* galerkinVal )
{
   SLIC_ERROR_IF(x==nullptr, "GalerkinEval(): input pointer, x, is NULL.");
   SLIC_ERROR_IF(nodeVals==nullptr, "GalerkinEval(): input pointer, nodeVals, is NULL.");
   SLIC_ERROR_IF(galerkinVal==nullptr, "GalerkinEval(): input/output pointer, galerkinVal, is NULL.");
   SLIC_ERROR_IF(galerkinDim<1, "GalerkinEval(): scalar approximations not yet supported." );

   int numNodes = GetNumFaceNodes( dim, order_type );
   switch (basis_type)
   {
      case PHYSICAL :
         for (int nd=0; nd<numNodes; ++nd)
         {
            real phi = 0.;
            EvalBasis( x, pX, pY, pZ, numNodes, nd, phi );
            for (int i=0; i<galerkinDim; ++i)
            {
               galerkinVal[i] += nodeVals[i+nd*galerkinDim] * phi;
            } 
         }
         break;
      default :
         SLIC_ERROR( "GalerkinEval(): basis_type = PARENT not yet supported." );
   }
}

void EvalBasis( const real* const RESTRICT x, 
                const real pX, const real pY, const real pZ, 
                const int numPoints, const int vertexId, 
                real& phi )
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
      SLIC_ERROR("EvalBasis: invalid numPoints argument.");
   }
   return;
}

//------------------------------------------------------------------------------
void SegmentBasis( const real* const RESTRICT x, 
                   const real pX, const real pY,
                   const int numPoints, const int vertexId, 
                   real& phi )
{
   SLIC_ERROR_IF(numPoints != 2, "SegmentBasis: numPoints is " << numPoints <<
                 " but should be 2.");

   // note, vertexId is the index, 0 or 1.
   SLIC_ERROR_IF(vertexId > numPoints-1, "SegmentBasis: vertexId is " << vertexId << 
                 " but should be 0 or 1.");

   // compute length of segment
   real vx = x[numPoints*1] - x[numPoints*0];
   real vy = x[numPoints*1+1] - x[numPoints*0+1];
   real lambda = magnitude( vx, vy );

   // compute the magnitude of the vector <pX,pY> - <x[vertexId],y[vertexId]>
   real wx = pX - x[ numPoints*vertexId ];
   real wy = pY - x[ numPoints*vertexId+1 ];

   real magW = magnitude( wx, wy );

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
void WachspressBasis( const real* const RESTRICT x, 
                      const real pX, const real pY, const real pZ, 
                      const int numPoints, const int vertexId, real& phi )
{
   SLIC_ERROR_IF(numPoints<3, "WachspressBasis: numPoints < 3.");

   // first compute the areas of all the triangles formed by the i-1,i,i+1 vertices.
   // These consist of all the numerators in the Wachspress formulation
   real triVertArea[ numPoints ];
   for (int i=0; i<numPoints; ++i)
   {
      // determine the i-1, i, i+1 vertices
      int vId = i; 
      int vIdMinus = (vId == 0) ? (numPoints-1) : (vId-1);
      int vIdPlus = (vId == (numPoints-1)) ? 0 : (vId+1);

      // construct segment between i-1,i and i-1,i+1
      real vx = x[ 3*vId ]   - x[ 3*vIdMinus ];
      real vy = x[ 3*vId+1 ] - x[ 3*vIdMinus+1 ];
      real vz = x[ 3*vId+2 ] - x[ 3*vIdMinus+2 ];

      real wx = x[ 3*vIdPlus ] -   x[ 3*vIdMinus ];
      real wy = x[ 3*vIdPlus+1 ] - x[ 3*vIdMinus+1 ];
      real wz = x[ 3*vIdPlus+2 ] - x[ 3*vIdMinus+2 ];

      // take the cross product between v and w to get the normal, and then obtain the 
      // area from the normal's magnitude
      real nX = (vy * wz) - (vz * wy);
      real nY = (vz * wx) - (vx * wz);
      real nZ = (vx * wy) - (vy * wx);
  
      triVertArea[i] = 0.5 * magnitude( nX, nY, nZ );

   }

   // second, compute the areas of all triangles formed using edge segment vertices
   // and the specified interior point (pX,pY,pZ)
   real triPointArea[ numPoints ];
   for (int i=0; i<numPoints; ++i)
   {
      // determine the i,i+1 edge segment 
      int vId = i;
      int vIdPlus = (vId == (numPoints-1)) ? 0 : (vId+1);

      // construct segments between i+1,i and p,i 
      real vx = x[ 3*vIdPlus ]   - x[ 3*vId ];
      real vy = x[ 3*vIdPlus+1 ] - x[ 3*vId+1 ];
      real vz = x[ 3*vIdPlus+2 ] - x[ 3*vId+2 ];

      real wx = pX - x[ 3*vId ];
      real wy = pY - x[ 3*vId+1 ];
      real wz = pZ - x[ 3*vId+2 ];

      // take the cross product between v and w to get the normal, and then obtain the 
      // area from the normal's magnitude
      real nX = (vy * wz) - (vz * wy);
      real nY = (vz * wx) - (vx * wz);
      real nZ = (vx * wy) - (vy * wx);
  
      triPointArea[i] = 0.5 * magnitude( nX, nY, nZ );

   }

   // third, compute all of the weights per Wachspress formulation
   real weight[ numPoints ];
   real myWeight;
   real weightSum = 0.;
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

   if (phi <= 0. || phi > 1.)
   {
      SLIC_ERROR("Wachspress Basis: phi is not between 0 and 1.");
   }

   return;
}

//------------------------------------------------------------------------------
bool InvIso( const real  x[3], 
             const real* xA,
             const real* yA,
             const real* zA,
             const int numNodes,
             real  xi[2] )
{

   SLIC_ERROR_IF(numNodes!=4, "InvIso: routine only for 4 node quads.");

   bool convrg = false;
   int kmax = 15;
   real xtol = 1.E-12;

   real x_sol[2] = {0., 0.};

   // derivatives of the Jacobian wrt (xi,eta)
   real djde_11   = 0.;
   real djde_x_12 = 0.25 * (xA[0] - xA[1] + xA[2] - xA[3]);
   real djde_y_12 = 0.25 * (yA[0] - yA[1] + yA[2] - yA[3]);
   real djde_z_12 = 0.25 * (zA[0] - zA[1] + zA[2] - zA[3]);
   real djde_22   = 0.;

   // loop over newton iterations
   for (int k = 0; k < kmax; ++k)
   {
      // evaluate Jacobian
      real j_x_1 = 0.25 * (xA[0] * (1. + x_sol[1]) - xA[1] * 
                   (1. + x_sol[1]) - xA[2] * (1. - x_sol[1]) + 
                   xA[3] * (1. - x_sol[1]));

      real j_y_1 = 0.25 * (yA[0] * (1. + x_sol[1]) - yA[1] * 
                   (1. + x_sol[1]) - yA[2] * (1. - x_sol[1]) + 
                   yA[3] * (1. - x_sol[1]));

      real j_z_1 = 0.25 * (zA[0] * (1. + x_sol[1]) - zA[1] * 
                   (1. + x_sol[1]) - zA[2] * (1. - x_sol[1]) + 
                   zA[3] * (1. - x_sol[1]));

      real j_x_2 = 0.25 * (xA[0] * (1. + x_sol[0]) + xA[1] * 
                   (1. - x_sol[0]) - xA[2] * (1. - x_sol[0]) - 
                   xA[3] * (1. + x_sol[0]));

      real j_y_2 = 0.25 * (yA[0] * (1. + x_sol[0]) + yA[1] * 
                   (1. - x_sol[0]) - yA[2] * (1. - x_sol[0]) - 
                   yA[3] * (1. + x_sol[0]));

      real j_z_2 = 0.25 * (zA[0] * (1. + x_sol[0]) + zA[1] * 
                   (1. - x_sol[0]) - zA[2] * (1. - x_sol[0]) - 
                   zA[3] * (1. + x_sol[0]));

      // evaluate the residual
      real f_x = x[0] - 0.25 * ((1. + x_sol[0]) * (1. + x_sol[1]) * xA[0]
                 + (1. - x_sol[0]) * (1. + x_sol[1]) * xA[1]
                 + (1. - x_sol[0]) * (1. - x_sol[1]) * xA[2]
                 + (1. + x_sol[0]) * (1. - x_sol[1]) * xA[3]);

      real f_y = x[1] - 0.25 * ((1. + x_sol[0]) * (1. + x_sol[1]) * yA[0]
                 + (1. - x_sol[0]) * (1. + x_sol[1]) * yA[1]
                 + (1. - x_sol[0]) * (1. - x_sol[1]) * yA[2]
                 + (1. + x_sol[0]) * (1. - x_sol[1]) * yA[3]);

      real f_z = x[2] - 0.25 * ((1. + x_sol[0]) * (1. + x_sol[1]) * zA[0]
                 + (1. - x_sol[0]) * (1. + x_sol[1]) * zA[1]
                 + (1. - x_sol[0]) * (1. - x_sol[1]) * zA[2]
                 + (1. + x_sol[0]) * (1. - x_sol[1]) * zA[3]);

      // compute J' * J
      real JTJ_11 = j_x_1 * j_x_1 + j_y_1 * j_y_1 + j_z_1 * j_z_1;
      real JTJ_12 = j_x_1 * j_x_2 + j_y_1 * j_y_2 + j_z_1 * j_z_2;
      //real JTJ_21 = JTJ_12;
      real JTJ_22 = j_x_2 * j_x_2 + j_y_2 * j_y_2 + j_z_2 * j_z_2;;

      // compute J' * F
      real JTF_1 = j_x_1 * f_x + j_y_1 * f_y + j_z_1 * f_z;
      real JTF_2 = j_x_2 * f_x + j_y_2 * f_y + j_z_2 * f_z;

      // for first few steps don't do exact Newton.
      real cm_11 = JTJ_11; //- (djde_11 * f_x + djde_11 * f_y + djde_11 * f_z);
      real cm_12 = JTJ_12; //- (djde_x_12 * f_x + djde_y_12 * f_y + djde_z_12 * f_z);
      real cm_21 = cm_12;
      real cm_22 = JTJ_22; //- (djde_22 * f_x + djde_22 * f_y + djde_22 * f_z);

      // do exact Newton for steps beyond first few
      if (k > 2)  // set to 2 per mortar method testing 
      {
       cm_11 += - (djde_11 * f_x + djde_11 * f_y + djde_11 * f_z);
       cm_12 += - (djde_x_12 * f_x + djde_y_12 * f_y + djde_z_12 * f_z);
       cm_21 = cm_12;
       cm_22 += - (djde_22 * f_x + djde_22 * f_y + djde_22 * f_z);
      }

      real detI = 1. / (cm_11 * cm_22 - cm_12 * cm_21);

      real cmi_11 = cm_22 * detI;
      real cmi_22 = cm_11 * detI;
      real cmi_12 = -cm_12 * detI;
      real cmi_21 = -cm_21 * detI;

      real dxi_1 = cmi_11 * JTF_1 + cmi_12 * JTF_2;
      real dxi_2 = cmi_21 * JTF_1 + cmi_22 * JTF_2;

      x_sol[0] += dxi_1;
      x_sol[1] += dxi_2;

      real abs_dxi_1 = std::abs(dxi_1);
      real abs_dxi_2 = std::abs(dxi_2);

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

         SLIC_WARNING_IF(!in_quad, "InvIso(): (xi,eta) coordinate does not lie " << 
                       "inside isoparametric quad.");
         if (!in_quad) {
            return false;
         }

         return true; 
      }

   }

   SLIC_WARNING_IF(!convrg, "InvIso: Newtons method did not converge.");
   if (!convrg) {
      return false;
   }

   return true;
}

//------------------------------------------------------------------------------
void FwdMapLinQuad( const real xi[2],
                    real xa[4],
                    real ya[4],
                    real za[4],
                    real x[3] )
{
   // initialize output array
   initRealArray( &x[0], 3, 0. );

   // obtain shape function evaluations at (xi,eta)
   real phi[4] = { 0., 0., 0., 0. };
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
void FwdMapLinTri( const real xi[2],
                   real xa[3],
                   real ya[3],
                   real za[3],
                   real x[3] )
{
   // initialize output array
   initRealArray( &x[0], 3, 0. );

   // obtain shape function evaluations at (xi,eta)
   real phi[3] = { 0., 0., 0. };
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
void LinIsoTriShapeFunc( const real xi, 
                         const real eta,
                         const int a,
                         real& phi )
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
         SLIC_ERROR("LinIsoTriShapeFunc: node id is not between 0 and 2.");
         break;
   }

   return;
}

//------------------------------------------------------------------------------
void LinIsoQuadShapeFunc( const real xi, 
                          const real eta,
                          const int a,
                          real& phi )
{
   real xi_node, eta_node;
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
         SLIC_ERROR("LinIsoQuadShapeFunc: node id is not between 0 and 3.");
         return;
   }

   phi = 0.25 * (1. + xi_node * xi) * ( 1. + eta_node * eta);

   SLIC_ERROR_IF(phi > 1.0 || phi < 0.0, "LinIsoQuadShapeFunc: phi is " << phi << " not between 0. and 1." );

   return;
}

//------------------------------------------------------------------------------
void DetJQuad( const real xi,
               const real eta,
               const real* x,
               const int dim,
               real& detJ )
{
   
   real J[4] = { 0., 0., 0., 0. }; // column major ordering

   // loop over nodes
   for (int a=0; a<4; ++a)
   {
      // determine (xi,eta) coord of node a
      real xi_node, eta_node;
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
