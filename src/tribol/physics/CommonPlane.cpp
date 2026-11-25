// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "CommonPlane.hpp"

#include "tribol/common/LoopExec.hpp"
#include "tribol/common/Atomics.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/geom/CompGeom.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/integ/Integration.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/utils/Math.hpp"

namespace tribol {

TRIBOL_HOST_DEVICE RealT ComputeGapRatePressure( CommonPlanePair& plane, const MeshData::Viewer& m1,
                                                 const MeshData::Viewer& m2, RealT element_penalty,
                                                 RatePenaltyCalculation rate_calc )
{
  auto fId1 = plane.getCpElementId1();
  auto fId2 = plane.getCpElementId2();

  const auto dim = plane.m_dim;

  if ( rate_calc == NO_RATE_PENALTY ) {
    return 0.0;
  }
  const RealT rate_penalty = detail::computeCommonPlaneRatePenalty( m1, m2, element_penalty, rate_calc );

  // compute the velocity gap and pressure contribution
  constexpr int max_dim = 3;
  constexpr int max_nodes_per_elem = 4;

  StackArrayT<RealT, max_dim * max_nodes_per_elem> x1;
  StackArrayT<RealT, max_dim * max_nodes_per_elem> v1;
  auto numNodesPerFace1 = m1.numberOfNodesPerElement();
  plane.getFace1Coords( x1, numNodesPerFace1 );  // get avg face coords off the contact plane
  m1.getFaceVelocities( fId1, v1 );

  StackArrayT<RealT, max_dim * max_nodes_per_elem> x2;
  StackArrayT<RealT, max_dim * max_nodes_per_elem> v2;
  auto numNodesPerFace2 = m2.numberOfNodesPerElement();
  plane.getFace2Coords( x2, numNodesPerFace2 );  // get avg face coords off the contact plane
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
  RealT cZf1 = ( dim == 3 ) ? plane.m_cZf1 : 0.;
  GalerkinEval( x1, cXf1, cYf1, cZf1, LINEAR_LAGRANGE, PHYSICAL, dim, dim, v1, vel_f1 );

  // interpolate nodal velocity at overlap centroid as projected
  // onto face 2
  RealT cXf2 = plane.m_cXf2;
  RealT cYf2 = plane.m_cYf2;
  RealT cZf2 = ( dim == 3 ) ? plane.m_cZf2 : 0.;
  GalerkinEval( x2, cXf2, cYf2, cZf2, LINEAR_LAGRANGE, PHYSICAL, dim, dim, v2, vel_f2 );

  // compute velocity gap vector
  RealT velGap[max_dim];
  velGap[0] = vel_f1[0] - vel_f2[0];
  velGap[1] = vel_f1[1] - vel_f2[1];
  if ( dim == 3 ) {
    velGap[2] = vel_f1[2] - vel_f2[2];
  }

  // compute velocity gap scalar
  plane.m_velGap = 0.;
  plane.m_velGap += velGap[0] * plane.m_nX;
  plane.m_velGap += velGap[1] * plane.m_nY;
  if ( dim == 3 ) {
    plane.m_velGap += velGap[2] * plane.m_nZ;
  }

  // check the gap rate sense.
  // (v1-v2) * \nu < 0 : velocities lead to more interpenetration;
  // note, \nu is in direction of face_2 outward unit normal
  // TODO consider a velocity gap tolerance. Checking this against
  // 0. actually smoothed out contact behavior in contact problem 1
  // for certain percent rate penalties.
  if ( plane.m_velGap <= 0. )  // TODO do we want = or just <?
  {
    plane.m_ratePressure = plane.m_velGap * rate_penalty;
    return plane.m_ratePressure;
  }  // end if-check on velocity gap

  return 0.;

}  // end ComputeGapRatePressure()

//------------------------------------------------------------------------------
template <>
int ApplyNormal<COMMON_PLANE, PENALTY>( CouplingScheme* cs )
{
  ///////////////////////////////
  // loop over interface pairs //
  ///////////////////////////////
  ArrayT<int> err_data( { 0 }, cs->getAllocatorId() );
  ArrayViewT<int> err = err_data;
  ArrayT<bool> neg_thickness_data( { false }, cs->getAllocatorId() );
  ArrayViewT<bool> neg_thickness = neg_thickness_data;
  auto cs_view = cs->getView();
  const auto num_pairs = cs->getNumActivePairs();
  forAllExec( cs->getExecutionMode(), num_pairs, [cs_view, err, neg_thickness] TRIBOL_HOST_DEVICE( IndexT i ) {
    auto& cg_view = cs_view.getCompGeomView();
    auto& plane = cg_view.getCommonPlane( i );

    auto& mesh1 = cs_view.getMesh1View();
    auto& mesh2 = cs_view.getMesh2View();

    // get pair indices
    IndexT index1 = plane.getCpElementId1();
    IndexT index2 = plane.getCpElementId2();

    RealT gap = plane.m_gap;
    RealT A = plane.m_area;  // face-pair overlap area

    //  don't proceed for gaps that don't violate the constraints. This check
    //  allows for numerically zero interpenetration.
    RealT gap_tol = cs_view.getGapTol( index1, index2 );

    if ( gap > gap_tol ) {
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
    RealT penalty_stiff_per_area{ 0. };
    auto& enforcement_options = cs_view.getEnforcementOptions();
    const PenaltyEnforcementOptions& pen_enfrc_options = enforcement_options.penalty_options;
    if ( !detail::computeCommonPlanePenaltyStiffnessPerArea( mesh1, mesh2, pen_enfrc_options, index1, index2,
                                                             penalty_stiff_per_area ) ) {
      neg_thickness[0] = true;
      err[0] = 1;
    }

    ////////////////////////////////////////////////////
    // Compute contact pressure(s) on current overlap //
    ////////////////////////////////////////////////////

    // compute total pressure based on constraint type
    RealT totalPressure = 0.;
    RealT residual_gap = cs_view.getParameters().residual_gap;
    plane.m_pressure = ( gap - residual_gap ) * penalty_stiff_per_area;  // kinematic contribution
    switch ( pen_enfrc_options.constraint_type ) {
      case KINEMATIC_AND_RATE: {
        // kinematic contribution
        totalPressure += plane.m_pressure;
        // add gap-rate contribution
        totalPressure +=
            ComputeGapRatePressure( plane, mesh1, mesh2, penalty_stiff_per_area, pen_enfrc_options.rate_calculation );
        break;
      }
      case KINEMATIC:
        // kinematic gap pressure contribution  only
        totalPressure += plane.m_pressure;
        break;
      default:
        // no-op
        break;
    }  // end switch on registered penalty enforcement option

    // debug prints. Comment out for now, but keep for future common plane
    // debugging
    //         SLIC_DEBUG("gap: " << gap);
    //         SLIC_DEBUG("area: " << A);
    //         SLIC_DEBUG("penalty stiffness: " << penalty_stiff_per_area);
    //         SLIC_DEBUG("pressure: " << cpManager.m_pressure[ cpID ]);

    const int dim = cs_view.spatialDim();
    const IndexT num_nodes_per_face = mesh1.numberOfNodesPerElement();
    const auto basis = detail::computeCommonPlaneForceBasis( plane, mesh1, mesh2 );
    const RealT contact_force = totalPressure * A;
    const RealT force_x = plane.m_nX * contact_force;
    const RealT force_y = plane.m_nY * contact_force;
    const RealT force_z = dim == 3 ? plane.m_nZ * contact_force : 0.0;

    //////////////////////////////////////////////////////
    // loop over nodes and compute contact nodal forces //
    //////////////////////////////////////////////////////
    for ( IndexT a = 0; a < num_nodes_per_face; ++a ) {
      IndexT node0 = mesh1.getGlobalNodeId( index1, a );
      IndexT node1 = mesh2.getGlobalNodeId( index2, a );

      // if (logLevel == TRIBOL_DEBUG)
      // {
      //   phi_sum_1 += basis.geometry1[a];
      //   phi_sum_2 += basis.geometry2[a];
      // }

      const RealT nodal_force_x1 = force_x * basis.geometry1[a];
      const RealT nodal_force_y1 = force_y * basis.geometry1[a];
      const RealT nodal_force_z1 = force_z * basis.geometry1[a];

      const RealT nodal_force_x2 = force_x * basis.geometry2[a];
      const RealT nodal_force_y2 = force_y * basis.geometry2[a];
      const RealT nodal_force_z2 = force_z * basis.geometry2[a];

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
      tribol::atomicAdd( &mesh1.getResponse()[0][node0], -nodal_force_x1 );
      tribol::atomicAdd( &mesh2.getResponse()[0][node1], nodal_force_x2 );

      tribol::atomicAdd( &mesh1.getResponse()[1][node0], -nodal_force_y1 );
      tribol::atomicAdd( &mesh2.getResponse()[1][node1], nodal_force_y2 );

      // there is no z component for 2D
      if ( dim == 3 ) {
        tribol::atomicAdd( &mesh1.getResponse()[2][node0], -nodal_force_z1 );
        tribol::atomicAdd( &mesh2.getResponse()[2][node1], nodal_force_z2 );
      }
    }  // end for loop over face nodes

    // comment out debug logs; too much output during tests. Keep for easy
    // debugging if needed
    // SLIC_DEBUG("force sum, side 1, pair " << kp << ": " << -dbg_sum_force1 );
    // SLIC_DEBUG("force sum, side 2, pair " << kp << ": " << dbg_sum_force2 );
    // SLIC_DEBUG("phi 1 sum: " << phi_sum_1 );
    // SLIC_DEBUG("phi 2 sum: " << phi_sum_2 );
  } );

  ArrayT<bool, 1, MemorySpace::Host> neg_thickness_host( neg_thickness_data );
  SLIC_DEBUG_IF( neg_thickness_host[0],
                 "ApplyNormal<COMMON_PLANE, PENALTY>: negative element thicknesses encountered." );

  ArrayT<int, 1, MemorySpace::Host> err_host( err_data );
  return err_host[0];

}  // end ApplyNormal<COMMON_PLANE, PENALTY>()

//------------------------------------------------------------------------------
template <>
int ApplyTangential<COMMON_PLANE, PENALTY, VISCOUS_TANGENTIAL>( CouplingScheme* cs )
{
  ///////////////////////////////
  // loop over interface pairs //
  ///////////////////////////////
  auto cs_view = cs->getView();
  const auto num_pairs = cs->getNumActivePairs();
  forAllExec( cs->getExecutionMode(), num_pairs, [cs_view] TRIBOL_HOST_DEVICE( IndexT i ) {
    auto& cg_view = cs_view.getCompGeomView();
    auto& plane = cg_view.getCommonPlane( i );

    if ( !plane.m_inContact ) {
      return;
    }

    const auto dim = plane.m_dim;
    auto& mesh1 = cs_view.getMesh1View();
    auto& mesh2 = cs_view.getMesh2View();

    // get pair indices
    IndexT index1 = plane.getCpElementId1();
    IndexT index2 = plane.getCpElementId2();

    // compute the velocity gap and pressure contribution
    constexpr int max_dim = 3;
    constexpr int max_nodes_per_elem = 4;

    StackArrayT<RealT, max_dim * max_nodes_per_elem> x1;
    StackArrayT<RealT, max_dim * max_nodes_per_elem> v1;
    auto numNodesPerFace1 = mesh1.numberOfNodesPerElement();
    plane.getFace1Coords( x1, numNodesPerFace1 );  // get avg face coords off the contact plane
    mesh1.getFaceVelocities( index1, v1 );

    StackArrayT<RealT, max_dim * max_nodes_per_elem> x2;
    StackArrayT<RealT, max_dim * max_nodes_per_elem> v2;
    auto numNodesPerFace2 = mesh2.numberOfNodesPerElement();
    plane.getFace2Coords( x2, numNodesPerFace2 );  // get avg face coords off the contact plane
    mesh2.getFaceVelocities( index2, v2 );

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
    RealT cZf1 = ( dim == 3 ) ? plane.m_cZf1 : 0.;
    GalerkinEval( x1, cXf1, cYf1, cZf1, LINEAR_LAGRANGE, PHYSICAL, dim, dim, v1, vel_f1 );

    // interpolate nodal velocity at overlap centroid as projected
    // onto face 2
    RealT cXf2 = plane.m_cXf2;
    RealT cYf2 = plane.m_cYf2;
    RealT cZf2 = ( dim == 3 ) ? plane.m_cZf2 : 0.;
    GalerkinEval( x2, cXf2, cYf2, cZf2, LINEAR_LAGRANGE, PHYSICAL, dim, dim, v2, vel_f2 );

    // compute velocity gap vector
    RealT velGap[max_dim];
    velGap[0] = vel_f1[0] - vel_f2[0];
    velGap[1] = vel_f1[1] - vel_f2[1];
    if ( dim == 3 ) {
      velGap[2] = vel_f1[2] - vel_f2[2];
    }

    // subtract off the common-plane normal component of the velocity gap
    RealT velGap_dot_n = velGap[0] * plane.m_nX + velGap[1] * plane.m_nY;
    if ( dim == 3 ) {
      velGap_dot_n += velGap[2] * plane.m_nZ;
    }
    RealT velGapTan[max_dim];
    velGapTan[0] = velGap[0] - velGap_dot_n * plane.m_nX;
    velGapTan[1] = velGap[1] - velGap_dot_n * plane.m_nY;
    if ( dim == 3 ) {
      velGapTan[2] = velGap[2] - velGap_dot_n * plane.m_nZ;
    }

    const auto basis = detail::computeCommonPlaneForceBasis( plane, mesh1, mesh2 );

    /////////////////////////////////////////////////////
    // Computation of tangential viscous damping force //
    /////////////////////////////////////////////////////
    const RealT visc = detail::computeCommonPlaneTangentialDamping( mesh1, mesh2 );
    RealT force_x = visc * velGapTan[0];
    RealT force_y = visc * velGapTan[1];
    RealT force_z = 0.;
    if ( dim == 3 ) {
      force_z = visc * velGapTan[2];
    }

    //////////////////////////////////////////////////////
    // loop over nodes and compute contact nodal forces //
    //////////////////////////////////////////////////////
    for ( IndexT a = 0; a < numNodesPerFace1; ++a ) {
      IndexT node0 = mesh1.getGlobalNodeId( index1, a );
      IndexT node1 = mesh2.getGlobalNodeId( index2, a );

      const RealT nodal_force_x1 = force_x * basis.geometry1[a];
      const RealT nodal_force_y1 = force_y * basis.geometry1[a];
      const RealT nodal_force_z1 = force_z * basis.geometry1[a];

      const RealT nodal_force_x2 = force_x * basis.geometry2[a];
      const RealT nodal_force_y2 = force_y * basis.geometry2[a];
      const RealT nodal_force_z2 = force_z * basis.geometry2[a];

      // accumulate contributions in host code's registered nodal force arrays
      tribol::atomicAdd( &mesh1.getResponse()[0][node0], -nodal_force_x1 );
      tribol::atomicAdd( &mesh2.getResponse()[0][node1], nodal_force_x2 );

      tribol::atomicAdd( &mesh1.getResponse()[1][node0], -nodal_force_y1 );
      tribol::atomicAdd( &mesh2.getResponse()[1][node1], nodal_force_y2 );

      // there is no z component for 2D
      if ( dim == 3 ) {
        tribol::atomicAdd( &mesh1.getResponse()[2][node0], -nodal_force_z1 );
        tribol::atomicAdd( &mesh2.getResponse()[2][node1], nodal_force_z2 );
      }
    }  // end for loop over face nodes
  } );

  return 0;
}
//------------------------------------------------------------------------------

}  // namespace tribol
