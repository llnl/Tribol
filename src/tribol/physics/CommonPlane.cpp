// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "CommonPlane.hpp"

#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/geom/ContactPlane.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/integ/Integration.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/utils/ContactPlaneOutput.hpp"
#include "tribol/utils/Math.hpp"

namespace tribol
{

TRIBOL_HOST_DEVICE RealT ComputePenaltyStiffnessPerArea( const RealT K1_over_t1,
                                                         const RealT K2_over_t2 )
{
   // compute face-pair specific penalty stiffness per unit area.
   // Note: This assumes that each face has a spring stiffness 
   // equal to that side's material Bulk modulus, K, over the 
   // thickness of the volume element to which that face belongs, 
   // times the overlap area. That is, K1_over_t1 * A and K2_over_t2 * A. We 
   // then assume the two springs are in series and compute an 
   // equivalent spring stiffness as, 
   // k_eq = A*(K1_over_t1)*(K2_over_t2) / ((K1_over_t1)+(K2_over_t2). 
   // Note, the host code registers each face's (K/t) as a penalty scale.
   //
   // UNITS: we multiply k_eq above by the overlap area A, to get a 
   // stiffness per unit area. This will make the force calculations 
   // commensurate with the previous calculations using only the 
   // constant registered penalty scale.

   return K1_over_t1 * K2_over_t2 / (K1_over_t1 + K2_over_t2);

} // end ComputePenaltyStiffnessPerArea

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE RealT ComputeGapRatePressure( ContactPlane& plane,
                                                 const MeshData::Viewer& m1,
                                                 const MeshData::Viewer& m2,
                                                 RealT element_penalty,
                                                 RatePenaltyCalculation rate_calc )
{
   auto fId1 = plane.getCpElementId1();
   auto fId2 = plane.getCpElementId2();

   const auto dim = plane.m_dim;

   // compute the correct rate_penalty
   RealT rate_penalty = 0.;
   switch (rate_calc)
   {
      case NO_RATE_PENALTY:
      {
         return 0.;
      }
      case RATE_CONSTANT:
      {
         rate_penalty = 0.5 * (m1.getElementData().m_rate_penalty_stiffness + 
                               m2.getElementData().m_rate_penalty_stiffness);
         break;
      }
      case RATE_PERCENT:
      {
         rate_penalty = element_penalty * 0.5 * 
                        (m1.getElementData().m_rate_percent_stiffness + 
                         m2.getElementData().m_rate_percent_stiffness);
         break;
      }
      default:
         // no-op, quiet compiler
         break;
   } // end switch on rate_calc

   // compute the velocity gap and pressure contribution
   constexpr int max_dim = 3;
   constexpr int max_nodes_per_elem = 4;

   StackArrayT<RealT, max_dim * max_nodes_per_elem> x1;
   StackArrayT<RealT, max_dim * max_nodes_per_elem> v1;
   m1.getFaceCoords( fId1, x1 );
   m1.getFaceVelocities( fId1, v1 );

   StackArrayT<RealT, max_dim * max_nodes_per_elem> x2;
   StackArrayT<RealT, max_dim * max_nodes_per_elem> v2;
   m2.getFaceCoords( fId2, x2 );
   m2.getFaceVelocities( fId2, v2 );

   //////////////////////////////////////////////////////////
   // compute velocity Galerkin approximation at projected // 
   // overlap centroid                                     //
   //////////////////////////////////////////////////////////
   RealT vel_f1[max_dim];
   RealT vel_f2[max_dim];
   initRealArray( vel_f1, dim, 0. );
   initRealArray( vel_f2, dim, 0. );

   // interpolate nodal velocity at overlap centroid as projected 
   // onto face 1
   RealT cXf1 = plane.m_cXf1;
   RealT cYf1 = plane.m_cYf1;
   RealT cZf1 = (dim == 3) ? plane.m_cZf1 : 0.;
   GalerkinEval( x1, cXf1, cYf1, cZf1,
                 LINEAR, PHYSICAL, dim, dim, 
                 v1, vel_f1 );

   // interpolate nodal velocity at overlap centroid as projected 
   // onto face 2
   RealT cXf2 = plane.m_cXf2;
   RealT cYf2 = plane.m_cYf2;
   RealT cZf2 = (dim == 3) ? plane.m_cZf2 : 0.;
   GalerkinEval( x2, cXf2, cYf2, cZf2,
                 LINEAR, PHYSICAL, dim, dim, 
                 v2, vel_f2 );

   // compute velocity gap vector
   RealT velGap[max_dim];
   velGap[0] = vel_f1[0] - vel_f2[0];
   velGap[1] = vel_f1[1] - vel_f2[1];
   if (dim == 3)
   {
      velGap[2] = vel_f1[2] - vel_f2[2];
   }

   // compute velocity gap scalar
   plane.m_velGap = 0.;
   plane.m_velGap += velGap[0] * plane.m_nX;
   plane.m_velGap += velGap[1] * plane.m_nY;
   if (dim == 3)
   {
      plane.m_velGap += velGap[2] * plane.m_nZ;
   }

   // check the gap rate sense. 
   // (v1-v2) * \nu < 0 : velocities lead to more interpenetration;
   // note, \nu is in direction of face_2 outward unit normal
   // TODO consider a velocity gap tolerance. Checking this against 
   // 0. actually smoothed out contact behavior in contact problem 1
   // for certain percent rate penalties.
   if (plane.m_velGap <= 0.) // TODO do we want = or just <?
   {
      plane.m_ratePressure = plane.m_velGap * rate_penalty;
      return plane.m_ratePressure;
   } // end if-check on velocity gap

   return 0.;

} // end ComputeGapRatePressure()

//------------------------------------------------------------------------------
template< >
int ApplyNormal< COMMON_PLANE, PENALTY >( CouplingScheme* cs )
{
   ///////////////////////////////
   // loop over interface pairs //
   ///////////////////////////////
   ArrayT<int> err_data({0}, cs->getAllocatorId());
   ArrayViewT<int> err = err_data;
   ArrayT<bool> neg_thickness_data({false}, cs->getAllocatorId());
   ArrayViewT<bool> neg_thickness = neg_thickness_data;
   auto cs_view = cs->getView();
   const auto num_pairs = cs->getNumActivePairs();
   forAllExec(cs->getExecutionMode(), num_pairs,
      [cs_view, err, neg_thickness] TRIBOL_HOST_DEVICE (IndexT i)
      {
         auto& plane = cs_view.getContactPlane(i);

         auto& mesh1 = cs_view.getMesh1View();
         auto& mesh2 = cs_view.getMesh2View();

         // get pair indices
         IndexT index1 = plane.getCpElementId1();
         IndexT index2 = plane.getCpElementId2();
        
         RealT gap = plane.m_gap;
         RealT A = plane.m_area; // face-pair overlap area

        //  don't proceed for gaps that don't violate the constraints. This check 
        //  allows for numerically zero interpenetration.
         RealT gap_tol = cs_view.getGapTol( index1, index2 );

        if ( gap > gap_tol )
        {
          // We are here if we have a pair that passes ALL geometric 
          // filter checks, BUT does not actually violate this method's 
          // gap constraint.
          plane.m_inContact = false;
          return;
        }

        // debug force sums
        // RealT dbg_sum_force1 {0.};
        // RealT dbg_sum_force2 {0.};
        /////////////////////////////////////////////
        // kinematic penalty stiffness calculation //
        /////////////////////////////////////////////
        RealT penalty_stiff_per_area {0.};
        auto& enforcement_options = cs_view.getEnforcementOptions();
        const PenaltyEnforcementOptions& pen_enfrc_options = enforcement_options.penalty_options;
        RealT pen_scale1 = mesh1.getElementData().m_penalty_scale;
        RealT pen_scale2 = mesh2.getElementData().m_penalty_scale;
        switch (pen_enfrc_options.kinematic_calculation)
        {
          case KINEMATIC_CONSTANT: 
          {
              // pre-multiply each spring stiffness by each mesh's penalty scale
              auto stiffness1 = pen_scale1 * mesh1.getElementData().m_penalty_stiffness;
              auto stiffness2 = pen_scale2 * mesh2.getElementData().m_penalty_stiffness;
              // compute the equivalent contact penalty spring stiffness per area
              penalty_stiff_per_area  = ComputePenaltyStiffnessPerArea( stiffness1, stiffness2 );
              break;
          }
          case KINEMATIC_ELEMENT:
          {
              // add tiny_length to element thickness to avoid division by zero
              auto t1 = mesh1.getElementData().m_thickness[ index1 ] + pen_enfrc_options.tiny_length;
              auto t2 = mesh2.getElementData().m_thickness[ index2 ] + pen_enfrc_options.tiny_length;

              if (t1 < 0. || t2 < 0.)
              {
                neg_thickness[0] = true;
                err[0] = 1;
              }

              // compute each element spring stiffness. Pre-multiply the material modulus 
              // (i.e. material stiffness) by each mesh's penalty scale
              auto stiffness1 = pen_scale1 * mesh1.getElementData().m_mat_mod[ index1 ] / t1;
              auto stiffness2 = pen_scale2 * mesh2.getElementData().m_mat_mod[ index2 ] / t2;
              // compute the equivalent contact penalty spring stiffness per area
              penalty_stiff_per_area = ComputePenaltyStiffnessPerArea( stiffness1, stiffness2 );
              break;
          }
          default:
              // no-op, quiet compiler
              break;
        } // end switch on kinematic penalty calculation option

        ////////////////////////////////////////////////////
        // Compute contact pressure(s) on current overlap // 
        ////////////////////////////////////////////////////

        // compute total pressure based on constraint type
        RealT totalPressure = 0.;
        plane.m_pressure = gap * penalty_stiff_per_area; // kinematic contribution
        switch(pen_enfrc_options.constraint_type)
        {
          case KINEMATIC_AND_RATE:
          {
              // kinematic contribution
              totalPressure += plane.m_pressure;
              // add gap-rate contribution
              totalPressure += 
                ComputeGapRatePressure( plane, mesh1, mesh2, penalty_stiff_per_area,
                                        pen_enfrc_options.rate_calculation );
              break;
          }
          case KINEMATIC:
              // kinematic gap pressure contribution  only
              totalPressure += plane.m_pressure;
              break;
          default:
              // no-op
              break;
        } // end switch on registered penalty enforcement option

        // debug prints. Comment out for now, but keep for future common plane 
        // debugging
  //         SLIC_DEBUG("gap: " << gap);
  //         SLIC_DEBUG("area: " << A);
  //         SLIC_DEBUG("penalty stiffness: " << penalty_stiff_per_area);
  //         SLIC_DEBUG("pressure: " << cpManager.m_pressure[ cpID ]);

        ///////////////////////////////////////////
        // create surface contact element struct //
        ///////////////////////////////////////////

        // construct array of nodal coordinates
        constexpr int max_dim = 3;
        constexpr int max_nodes_per_face = 4;
        constexpr int max_nodes_per_overlap = 8;
        RealT xf1[ max_dim * max_nodes_per_face ];
        RealT xf2[ max_dim * max_nodes_per_face ];
        RealT xVert[ max_dim * max_nodes_per_overlap ];  
        int dim = cs_view.spatialDimension();
        int num_nodes_per_face = mesh1.numberOfNodesPerElement();
        initRealArray( xf1, dim * num_nodes_per_face, 0. );
        initRealArray( xf2, dim * num_nodes_per_face, 0. );
        // initialize assuming 2d
        auto xVert_size = 4;
        auto numPolyVert = 2;
        // update if we are in 3d
        if (dim == 3)
        {
          auto& cp3 = static_cast<ContactPlane3D&>(plane);
          numPolyVert = cp3.m_numPolyVert;
          xVert_size = 3 * numPolyVert;
        }
        initRealArray( xVert, xVert_size, 0. );

  //      // get projected face coordinates
  //      cpManager.getProjectedFaceCoords( cpID, 0, &xf1[0] ); // face 0 = first face
  //      cpManager.getProjectedFaceCoords( cpID, 1, &xf2[0] ); // face 1 = second face

        // get current configuration, physical coordinates of each face
        mesh1.getFaceCoords( index1, &xf1[0] );
        mesh2.getFaceCoords( index2, &xf2[0] );

        // construct array of polygon overlap vertex coordinates
        if (dim == 2)
        {
          auto& cp2 = static_cast<ContactPlane2D&>(plane);
          for (IndexT j{0}; j < numPolyVert; ++j)
          {
            xVert[dim*j] = cp2.m_segX[j];
            xVert[dim*j + 1] = cp2.m_segY[j];
          }
        }
        else
        {
          auto& cp3 = static_cast<ContactPlane3D&>(plane);
          for (IndexT j{0}; j < numPolyVert; ++j)
          {
            xVert[dim*j] = cp3.m_polyX[j];
            xVert[dim*j + 1] = cp3.m_polyY[j];
            xVert[dim*j + 2] = cp3.m_polyZ[j];
          }
        }

        // instantiate surface contact element struct. Note, this is done with current 
        // configuration face coordinates (i.e. NOT on the contact plane) and overlap 
        // coordinates ON the contact plane. The surface contact element does not need 
        // to be used this way, but the developer should do the book-keeping.
        SurfaceContactElem cntctElem( dim, xf1, xf2, xVert,
                                      num_nodes_per_face, numPolyVert,
                                      &mesh1, &mesh2, index1, index2 );

        // set SurfaceContactElem face normals and overlap normal
        RealT faceNormal1[max_dim];
        RealT faceNormal2[max_dim];
        RealT overlapNormal[max_dim];

        mesh1.getFaceNormal( index1, faceNormal1 );
        mesh2.getFaceNormal( index2, faceNormal2 );
        overlapNormal[0] = plane.m_nX;
        overlapNormal[1] = plane.m_nY;
        if (dim == 3)
        {
          overlapNormal[2] = plane.m_nZ;
        }

        cntctElem.faceNormal1 = faceNormal1;
        cntctElem.faceNormal2 = faceNormal2;
        cntctElem.overlapNormal = overlapNormal;
        cntctElem.overlapArea   = plane.m_area;

        // create arrays to hold nodal residual weak form integral evaluations
        RealT phi1[max_nodes_per_face];
        RealT phi2[max_nodes_per_face];
        initRealArray( phi1, num_nodes_per_face, 0. );
        initRealArray( phi2, num_nodes_per_face, 0. );

        ////////////////////////////////////////////////////////////////////////
        // Integration of contact integrals: integral of shape functions over //
        // contact overlap patch                                              //
        ////////////////////////////////////////////////////////////////////////
        EvalWeakFormIntegral< COMMON_PLANE, SINGLE_POINT >
                            ( cntctElem, phi1, phi2 );

        ///////////////////////////////////////////////////////////////////////
        // Computation of full contact nodal force contributions             //
        // (i.e. premultiplication of contact integrals by normal component, //
        //  contact pressure, and overlap area)                              //
        ///////////////////////////////////////////////////////////////////////

        // RealT phi_sum_1 = 0.;
        // RealT phi_sum_2 = 0.;

        // compute contact force (spring force)
        RealT contact_force = totalPressure * A;
        RealT force_x = overlapNormal[0] * contact_force;
        RealT force_y = overlapNormal[1] * contact_force;
        RealT force_z = 0.;
        if (dim == 3)
        {
          force_z = overlapNormal[2] * contact_force;
        }

        //////////////////////////////////////////////////////
        // loop over nodes and compute contact nodal forces //
        //////////////////////////////////////////////////////
        for( IndexT a=0 ; a < num_nodes_per_face ; ++a )
        {

          IndexT node0 = mesh1.getGlobalNodeId(index1, a);
          IndexT node1 = mesh2.getGlobalNodeId(index2, a);

          // if (logLevel == TRIBOL_DEBUG)
          // {
          //   phi_sum_1 += phi1[a];
          //   phi_sum_2 += phi2[a];
          // }
  
          const RealT nodal_force_x1 = force_x * phi1[a];
          const RealT nodal_force_y1 = force_y * phi1[a];
          const RealT nodal_force_z1 = force_z * phi1[a];

          const RealT nodal_force_x2 = force_x * phi2[a];
          const RealT nodal_force_y2 = force_y * phi2[a];
          const RealT nodal_force_z2 = force_z * phi2[a];

          // if (logLevel == TRIBOL_DEBUG)
          // {
          //   dbg_sum_force1 += magnitude( nodal_force_x1, 
          //                                 nodal_force_y1, 
          //                                 nodal_force_z1 );
          //   dbg_sum_force2 += magnitude( nodal_force_x2,
          //                                 nodal_force_y2,
          //                                 nodal_force_z2 );
          // }

          // accumulate contributions in host code's registered nodal force arrays
#ifdef TRIBOL_USE_RAJA
          RAJA::atomicAdd<RAJA::auto_atomic>(&mesh1.getResponse()[0][node0], -nodal_force_x1);
          RAJA::atomicAdd<RAJA::auto_atomic>(&mesh2.getResponse()[0][node1],  nodal_force_x2);

          RAJA::atomicAdd<RAJA::auto_atomic>(&mesh1.getResponse()[1][node0], -nodal_force_y1);
          RAJA::atomicAdd<RAJA::auto_atomic>(&mesh2.getResponse()[1][node1],  nodal_force_y2);

          // there is no z component for 2D
          if (dim == 3)
          {
            RAJA::atomicAdd<RAJA::auto_atomic>(&mesh1.getResponse()[2][node0], -nodal_force_z1);
            RAJA::atomicAdd<RAJA::auto_atomic>(&mesh2.getResponse()[2][node1],  nodal_force_z2);
          }
#else
          mesh1.getResponse()[0][node0] -= nodal_force_x1;
          mesh2.getResponse()[0][node1] += nodal_force_x2;

          mesh1.getResponse()[1][node0] -= nodal_force_y1;
          mesh2.getResponse()[1][node1] += nodal_force_y2;

          // there is no z component for 2D
          if (dim == 3)
          {
            mesh1.getResponse()[2][node0] -= nodal_force_z1;
            mesh2.getResponse()[2][node1] += nodal_force_z2;
          }
#endif
        } // end for loop over face nodes

        // comment out debug logs; too much output during tests. Keep for easy 
        // debugging if needed
        //SLIC_DEBUG("force sum, side 1, pair " << kp << ": " << -dbg_sum_force1 );
        //SLIC_DEBUG("force sum, side 2, pair " << kp << ": " << dbg_sum_force2 );
        //SLIC_DEBUG("phi 1 sum: " << phi_sum_1 );
        //SLIC_DEBUG("phi 2 sum: " << phi_sum_2 );
      }
   );
  
   ArrayT<bool, 1, MemorySpace::Host> neg_thickness_host(neg_thickness_data);
   SLIC_DEBUG_IF(neg_thickness_host[0], "ApplyNormal<COMMON_PLANE, PENALTY>: negative element thicknesses encountered.");

   ArrayT<int, 1, MemorySpace::Host> err_host(err_data);
   return err_host[0];

} // end ApplyNormal<COMMON_PLANE, PENALTY>()

}
