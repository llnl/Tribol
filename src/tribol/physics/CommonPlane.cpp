// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "CommonPlane.hpp"

#include "tribol/common/LoopExec.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/geom/CompGeom.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/integ/Integration.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/utils/Math.hpp"

namespace tribol {

namespace {

constexpr int max_dim = 3;
constexpr int max_nodes_per_face = 4;
constexpr int max_nodes_per_overlap = 10;

TRIBOL_HOST_DEVICE inline RealT ComputeRatePenalty( const MeshData::Viewer& m1, const MeshData::Viewer& m2,
                                                    RealT element_penalty, RatePenaltyCalculation rate_calc )
{
  switch ( rate_calc ) {
    case NO_RATE_PENALTY: {
      return 0.;
    }
    case RATE_CONSTANT: {
      return 0.5 * ( m1.getElementData().m_rate_penalty_stiffness + m2.getElementData().m_rate_penalty_stiffness );
    }
    case RATE_PERCENT: {
      return element_penalty * 0.5 *
             ( m1.getElementData().m_rate_percent_stiffness + m2.getElementData().m_rate_percent_stiffness );
    }
    default:
      return 0.;
  }
}

TRIBOL_HOST_DEVICE inline void EvalLinearFaceAtPoint( const RealT* face_coords, const int num_nodes,
                                                      const RealT x_query[3], RealT x_face[3], RealT* phi,
                                                      const int value_dim = 0, const RealT* nodal_vals = nullptr,
                                                      RealT* values = nullptr )
{
  RealT xA[max_nodes_per_face] = { 0., 0., 0., 0. };
  RealT yA[max_nodes_per_face] = { 0., 0., 0., 0. };
  RealT zA[max_nodes_per_face] = { 0., 0., 0., 0. };
  for ( int i = 0; i < num_nodes; ++i ) {
    xA[i] = face_coords[3 * i];
    yA[i] = face_coords[3 * i + 1];
    zA[i] = face_coords[3 * i + 2];
  }

  RealT xi[2] = { 0., 0. };
  InvIso( x_query, xA, yA, zA, num_nodes, xi );

  initRealArray( x_face, max_dim, 0. );
  initRealArray( phi, num_nodes, 0. );
  if ( values != nullptr ) {
    initRealArray( values, value_dim, 0. );
  }

  for ( int a = 0; a < num_nodes; ++a ) {
    if ( num_nodes == 4 ) {
      LinIsoQuadShapeFunc( xi[0], xi[1], a, phi[a] );
    } else {
      LinIsoTriShapeFunc( xi[0], xi[1], a, phi[a] );
    }

    x_face[0] += xA[a] * phi[a];
    x_face[1] += yA[a] * phi[a];
    x_face[2] += zA[a] * phi[a];

    if ( values != nullptr ) {
      for ( int i = 0; i < value_dim; ++i ) {
        values[i] += nodal_vals[i + a * value_dim] * phi[a];
      }
    }
  }
}

TRIBOL_HOST_DEVICE inline void AccumulateContactForce( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                                       const IndexT index1, const IndexT index2, const int dim,
                                                       const int num_nodes_per_face, const RealT force_x,
                                                       const RealT force_y, const RealT force_z, const RealT* phi1,
                                                       const RealT* phi2 )
{
  for ( IndexT a = 0; a < num_nodes_per_face; ++a ) {
    IndexT node0 = mesh1.getGlobalNodeId( index1, a );
    IndexT node1 = mesh2.getGlobalNodeId( index2, a );

    const RealT nodal_force_x1 = force_x * phi1[a];
    const RealT nodal_force_y1 = force_y * phi1[a];
    const RealT nodal_force_z1 = force_z * phi1[a];

    const RealT nodal_force_x2 = force_x * phi2[a];
    const RealT nodal_force_y2 = force_y * phi2[a];
    const RealT nodal_force_z2 = force_z * phi2[a];

#ifdef TRIBOL_USE_RAJA
    RAJA::atomicAdd<RAJA::auto_atomic>( &mesh1.getResponse()[0][node0], -nodal_force_x1 );
    RAJA::atomicAdd<RAJA::auto_atomic>( &mesh2.getResponse()[0][node1], nodal_force_x2 );

    RAJA::atomicAdd<RAJA::auto_atomic>( &mesh1.getResponse()[1][node0], -nodal_force_y1 );
    RAJA::atomicAdd<RAJA::auto_atomic>( &mesh2.getResponse()[1][node1], nodal_force_y2 );

    if ( dim == 3 ) {
      RAJA::atomicAdd<RAJA::auto_atomic>( &mesh1.getResponse()[2][node0], -nodal_force_z1 );
      RAJA::atomicAdd<RAJA::auto_atomic>( &mesh2.getResponse()[2][node1], nodal_force_z2 );
    }
#else
    mesh1.getResponse()[0][node0] -= nodal_force_x1;
    mesh2.getResponse()[0][node1] += nodal_force_x2;

    mesh1.getResponse()[1][node0] -= nodal_force_y1;
    mesh2.getResponse()[1][node1] += nodal_force_y2;

    if ( dim == 3 ) {
      mesh1.getResponse()[2][node0] -= nodal_force_z1;
      mesh2.getResponse()[2][node1] += nodal_force_z2;
    }
#endif
  }  // end switch on rate_calc
}

}  // namespace

TRIBOL_HOST_DEVICE RealT ComputeGapRatePressure( CommonPlanePair& plane, const MeshData::Viewer& m1,
                                                 const MeshData::Viewer& m2, RealT element_penalty,
                                                 RatePenaltyCalculation rate_calc )
{
  auto fId1 = plane.getCpElementId1();
  auto fId2 = plane.getCpElementId2();

  const auto dim = plane.m_dim;
  const RealT rate_penalty = ComputeRatePenalty( m1, m2, element_penalty, rate_calc );
  if ( rate_penalty == 0. ) {
    return 0.;
  }

  // compute the velocity gap and pressure contribution
  StackArrayT<RealT, max_dim * max_nodes_per_face> x1;
  StackArrayT<RealT, max_dim * max_nodes_per_face> v1;
  auto numNodesPerFace1 = m1.numberOfNodesPerElement();
  plane.getFace1Coords( x1, numNodesPerFace1 );  // get avg face coords off the contact plane
  m1.getFaceVelocities( fId1, v1 );

  StackArrayT<RealT, max_dim * max_nodes_per_face> x2;
  StackArrayT<RealT, max_dim * max_nodes_per_face> v2;
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
  GalerkinEvalOnPhysicalFace( x1, cXf1, cYf1, cZf1, numNodesPerFace1, dim, v1, vel_f1 );

  // interpolate nodal velocity at overlap centroid as projected
  // onto face 2
  RealT cXf2 = plane.m_cXf2;
  RealT cYf2 = plane.m_cYf2;
  RealT cZf2 = ( dim == 3 ) ? plane.m_cZf2 : 0.;
  GalerkinEvalOnPhysicalFace( x2, cXf2, cYf2, cZf2, numNodesPerFace2, dim, v2, vel_f2 );

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

    // debug force sums
    // RealT dbg_sum_force1 {0.};
    // RealT dbg_sum_force2 {0.};
    /////////////////////////////////////////////
    // kinematic penalty stiffness calculation //
    /////////////////////////////////////////////
    RealT penalty_stiff_per_area{ 0. };
    auto& enforcement_options = cs_view.getEnforcementOptions();
    const PenaltyEnforcementOptions& pen_enfrc_options = enforcement_options.penalty_options;
    const bool use_full_tri = pen_enfrc_options.common_plane_rule == FULL_TRI_DECOMP && cs_view.spatialDimension() == 3;
    if ( !use_full_tri && gap > gap_tol ) {
      // We are here if we have a pair that passes ALL geometric
      // filter checks, BUT does not actually violate this method's
      // gap constraint.
      plane.m_inContact = false;
      return;
    }

    RealT pen_scale1 = mesh1.getElementData().m_penalty_scale;
    RealT pen_scale2 = mesh2.getElementData().m_penalty_scale;
    switch ( pen_enfrc_options.kinematic_calculation ) {
      case KINEMATIC_CONSTANT: {
        // pre-multiply each spring stiffness by each mesh's penalty scale
        auto stiffness1 = pen_scale1 * mesh1.getElementData().m_penalty_stiffness;
        auto stiffness2 = pen_scale2 * mesh2.getElementData().m_penalty_stiffness;
        // compute the equivalent contact penalty spring stiffness per area
        penalty_stiff_per_area = ComputePenaltyStiffnessPerArea( stiffness1, stiffness2 );
        break;
      }
      case KINEMATIC_ELEMENT: {
        // add tiny_length to element thickness to avoid division by zero
        auto t1 = mesh1.getElementData().m_thickness[index1] + pen_enfrc_options.tiny_length;
        auto t2 = mesh2.getElementData().m_thickness[index2] + pen_enfrc_options.tiny_length;

        if ( t1 < 0. || t2 < 0. ) {
          neg_thickness[0] = true;
          err[0] = 1;
        }

        // compute each element spring stiffness. Pre-multiply the material modulus
        // (i.e. material stiffness) by each mesh's penalty scale
        auto stiffness1 = pen_scale1 * mesh1.getElementData().m_mat_mod[index1] / t1;
        auto stiffness2 = pen_scale2 * mesh2.getElementData().m_mat_mod[index2] / t2;
        // compute the equivalent contact penalty spring stiffness per area
        penalty_stiff_per_area = ComputePenaltyStiffnessPerArea( stiffness1, stiffness2 );
        break;
      }
      default:
        // no-op, quiet compiler
        break;
    }  // end switch on kinematic penalty calculation option

    ////////////////////////////////////////////////////
    // Compute contact pressure(s) on current overlap //
    ////////////////////////////////////////////////////

    // compute total pressure based on constraint type
    RealT totalPressure = 0.;
    plane.m_pressure = gap * penalty_stiff_per_area;  // kinematic contribution
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

    ///////////////////////////////////////////
    // create surface contact element struct //
    ///////////////////////////////////////////

    // construct array of nodal coordinates
    RealT xf1[max_dim * max_nodes_per_face];
    RealT xf2[max_dim * max_nodes_per_face];
    RealT xVert[max_dim * max_nodes_per_overlap];
    int dim = cs_view.spatialDimension();
    int num_nodes_per_face = mesh1.numberOfNodesPerElement();
    initRealArray( xf1, dim * num_nodes_per_face, 0. );
    initRealArray( xf2, dim * num_nodes_per_face, 0. );
    // initialize assuming 2d
    auto xVert_size = 4;
    auto numPolyVert = 2;
    // update if we are in 3d
    if ( dim == 3 ) {
      numPolyVert = plane.m_numPolyVert;
      xVert_size = 3 * numPolyVert;
    }
    initRealArray( xVert, xVert_size, 0. );

    // get current configuration, physical coordinates of each face
    plane.getFace1Coords( &xf1[0], num_nodes_per_face );
    plane.getFace2Coords( &xf2[0], num_nodes_per_face );

    // construct array of polygon overlap vertex coordinates
    plane.getOverlapVertices( &xVert[0] );

    // instantiate surface contact element struct. Note, this is done with current
    // configuration face coordinates (i.e. NOT on the contact plane) and overlap
    // coordinates ON the contact plane. The surface contact element does not need
    // to be used this way, but the developer should do the book-keeping.
    SurfaceContactElem cntctElem( dim, xf1, xf2, xVert, num_nodes_per_face, numPolyVert, &mesh1, &mesh2, index1,
                                  index2 );

    // set SurfaceContactElem face normals and overlap normal
    RealT faceNormal1[max_dim];
    RealT faceNormal2[max_dim];
    RealT overlapNormal[max_dim];

    mesh1.getFaceNormal( index1, faceNormal1 );
    mesh2.getFaceNormal( index2, faceNormal2 );
    overlapNormal[0] = plane.m_nX;
    overlapNormal[1] = plane.m_nY;
    if ( dim == 3 ) {
      overlapNormal[2] = plane.m_nZ;
    }

    cntctElem.faceNormal1 = faceNormal1;
    cntctElem.faceNormal2 = faceNormal2;
    cntctElem.overlapNormal = overlapNormal;
    cntctElem.overlapArea = plane.m_area;

    // create arrays to hold nodal residual weak form integral evaluations
    RealT phi1[max_nodes_per_face];
    RealT phi2[max_nodes_per_face];
    initRealArray( phi1, num_nodes_per_face, 0. );
    initRealArray( phi2, num_nodes_per_face, 0. );

    ////////////////////////////////////////////////////////////////////////
    // Integration of contact integrals: integral of shape functions over //
    // contact overlap patch                                              //
    ////////////////////////////////////////////////////////////////////////
    EvalWeakFormIntegralCommonPlane( cntctElem, pen_enfrc_options.common_plane_rule,
                                     pen_enfrc_options.common_plane_triangle_order, phi1, phi2 );

    if ( use_full_tri ) {
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_xf1;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_xf2;
      mesh1.getFaceCoords( index1, actual_xf1 );
      mesh2.getFaceCoords( index2, actual_xf2 );

      const bool use_rate = pen_enfrc_options.constraint_type == KINEMATIC_AND_RATE;
      const RealT rate_penalty =
          use_rate ? ComputeRatePenalty( mesh1, mesh2, penalty_stiff_per_area, pen_enfrc_options.rate_calculation ) : 0.;

      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_vf1;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_vf2;
      if ( use_rate ) {
        mesh1.getFaceVelocities( index1, actual_vf1 );
        mesh2.getFaceVelocities( index2, actual_vf2 );
      }

      constexpr int max_qpts = 6;
      RealT rule_wts[max_qpts] = { 0., 0., 0., 0., 0., 0. };
      RealT rule_coords[2 * max_qpts] = { 0. };
      const int num_qpts = GetCommonPlaneTriangleRule( pen_enfrc_options.common_plane_triangle_order, rule_wts, rule_coords );

      RealT centroid[3];
      GetCommonPlaneOverlapCentroid( cntctElem, centroid );

      RealT xTri[3];
      RealT yTri[3];
      RealT zTri[3];
      bool has_contact = false;

      for ( int j = 0; j < numPolyVert; ++j ) {
        const int next = ( j == numPolyVert - 1 ) ? 0 : j + 1;
        xTri[0] = xVert[dim * j];
        yTri[0] = xVert[dim * j + 1];
        zTri[0] = xVert[dim * j + 2];
        xTri[1] = xVert[dim * next];
        yTri[1] = xVert[dim * next + 1];
        zTri[1] = xVert[dim * next + 2];
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
          RealT x_q[3];
          x_q[0] = n0 * xTri[0] + xi * xTri[1] + eta * xTri[2];
          x_q[1] = n0 * yTri[0] + xi * yTri[1] + eta * yTri[2];
          x_q[2] = n0 * zTri[0] + xi * zTri[1] + eta * zTri[2];

          RealT phi_q1[max_nodes_per_face] = { 0., 0., 0., 0. };
          RealT phi_q2[max_nodes_per_face] = { 0., 0., 0., 0. };
          RealT x_qf1[max_dim];
          RealT x_qf2[max_dim];
          RealT vel_q1[max_dim] = { 0., 0., 0. };
          RealT vel_q2[max_dim] = { 0., 0., 0. };

          EvalLinearFaceAtPoint( actual_xf1, num_nodes_per_face, x_q, x_qf1, phi_q1, use_rate ? dim : 0,
                                 use_rate ? &actual_vf1[0] : nullptr, use_rate ? vel_q1 : nullptr );
          EvalLinearFaceAtPoint( actual_xf2, num_nodes_per_face, x_q, x_qf2, phi_q2, use_rate ? dim : 0,
                                 use_rate ? &actual_vf2[0] : nullptr, use_rate ? vel_q2 : nullptr );

          RealT local_gap = ( x_qf1[0] - x_qf2[0] ) * overlapNormal[0] + ( x_qf1[1] - x_qf2[1] ) * overlapNormal[1] +
                            ( x_qf1[2] - x_qf2[2] ) * overlapNormal[2];
          if ( local_gap > gap_tol ) {
            continue;
          }

          has_contact = true;

          RealT local_pressure = local_gap * penalty_stiff_per_area;
          if ( use_rate && rate_penalty > 0. ) {
            RealT local_vel_gap =
                ( vel_q1[0] - vel_q2[0] ) * overlapNormal[0] + ( vel_q1[1] - vel_q2[1] ) * overlapNormal[1] +
                ( vel_q1[2] - vel_q2[2] ) * overlapNormal[2];
            if ( local_vel_gap <= 0. ) {
              local_pressure += local_vel_gap * rate_penalty;
            }
          }

          const RealT weighted_force = area * rule_wts[qp] * local_pressure;
          const RealT force_x = overlapNormal[0] * weighted_force;
          const RealT force_y = overlapNormal[1] * weighted_force;
          const RealT force_z = overlapNormal[2] * weighted_force;

          AccumulateContactForce( mesh1, mesh2, index1, index2, dim, num_nodes_per_face, force_x, force_y, force_z,
                                  phi_q1, phi_q2 );
        }
      }

      plane.m_inContact = has_contact;
      return;
    }

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
    if ( dim == 3 ) {
      force_z = overlapNormal[2] * contact_force;
    }

    AccumulateContactForce( mesh1, mesh2, index1, index2, dim, num_nodes_per_face, force_x, force_y, force_z, phi1, phi2 );

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
    const PenaltyEnforcementOptions& pen_enfrc_options = cs_view.getEnforcementOptions().penalty_options;

    // get pair indices
    IndexT index1 = plane.getCpElementId1();
    IndexT index2 = plane.getCpElementId2();

    const bool use_full_tri = pen_enfrc_options.common_plane_rule == FULL_TRI_DECOMP && dim == 3;

    // compute the velocity gap and pressure contribution
    StackArrayT<RealT, max_dim * max_nodes_per_face> x1;
    StackArrayT<RealT, max_dim * max_nodes_per_face> v1;
    auto numNodesPerFace1 = mesh1.numberOfNodesPerElement();
    plane.getFace1Coords( x1, numNodesPerFace1 );  // get avg face coords off the contact plane
    mesh1.getFaceVelocities( index1, v1 );

    StackArrayT<RealT, max_dim * max_nodes_per_face> x2;
    StackArrayT<RealT, max_dim * max_nodes_per_face> v2;
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
    GalerkinEvalOnPhysicalFace( x1, cXf1, cYf1, cZf1, numNodesPerFace1, dim, v1, vel_f1 );

    // interpolate nodal velocity at overlap centroid as projected
    // onto face 2
    RealT cXf2 = plane.m_cXf2;
    RealT cYf2 = plane.m_cYf2;
    RealT cZf2 = ( dim == 3 ) ? plane.m_cZf2 : 0.;
    GalerkinEvalOnPhysicalFace( x2, cXf2, cYf2, cZf2, numNodesPerFace2, dim, v2, vel_f2 );

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

    // setup the contact element struct for purposes of evaluating basis functions on overlap
    // initialize assuming 2d
    RealT xVert[max_dim * max_nodes_per_overlap];
    auto xVert_size = 4;
    auto numPolyVert = 2;
    // update if we are in 3d
    if ( dim == 3 ) {
      numPolyVert = plane.m_numPolyVert;
      xVert_size = 3 * numPolyVert;
    }
    initRealArray( xVert, xVert_size, 0. );

    // construct array of polygon overlap vertex coordinates
    plane.getOverlapVertices( &xVert[0] );

    // instantiate surface contact element struct. Note, this is done with current
    // configuration face coordinates (i.e. NOT on the contact plane) and overlap
    // coordinates ON the contact plane. The surface contact element does not need
    // to be used this way, but the developer should do the book-keeping.
    SurfaceContactElem cntctElem( dim, x1, x2, xVert, numNodesPerFace1, numPolyVert, &mesh1, &mesh2, index1, index2 );

    // set SurfaceContactElem face normals and overlap normal
    RealT faceNormal1[max_dim];
    RealT faceNormal2[max_dim];
    RealT overlapNormal[max_dim];

    mesh1.getFaceNormal( index1, faceNormal1 );
    mesh2.getFaceNormal( index2, faceNormal2 );
    overlapNormal[0] = plane.m_nX;
    overlapNormal[1] = plane.m_nY;
    if ( dim == 3 ) {
      overlapNormal[2] = plane.m_nZ;
    }

    cntctElem.faceNormal1 = faceNormal1;
    cntctElem.faceNormal2 = faceNormal2;
    cntctElem.overlapNormal = overlapNormal;
    cntctElem.overlapArea = plane.m_area;

    // create arrays to hold nodal residual weak form integral evaluations
    RealT phi1[max_nodes_per_face];
    RealT phi2[max_nodes_per_face];
    initRealArray( phi1, numNodesPerFace1, 0. );
    initRealArray( phi2, numNodesPerFace2, 0. );

    ////////////////////////////////////////////////////////////////////////
    // Integration of contact integrals: integral of shape functions over //
    // contact overlap patch                                              //
    ////////////////////////////////////////////////////////////////////////
    EvalWeakFormIntegralCommonPlane( cntctElem, pen_enfrc_options.common_plane_rule,
                                     pen_enfrc_options.common_plane_triangle_order, phi1, phi2 );

    if ( use_full_tri ) {
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_xf1;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_xf2;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_vf1;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_vf2;
      mesh1.getFaceCoords( index1, actual_xf1 );
      mesh2.getFaceCoords( index2, actual_xf2 );
      mesh1.getFaceVelocities( index1, actual_vf1 );
      mesh2.getFaceVelocities( index2, actual_vf2 );

      constexpr int max_qpts = 6;
      RealT rule_wts[max_qpts] = { 0., 0., 0., 0., 0., 0. };
      RealT rule_coords[2 * max_qpts] = { 0. };
      const int num_qpts = GetCommonPlaneTriangleRule( pen_enfrc_options.common_plane_triangle_order, rule_wts, rule_coords );

      RealT centroid[3];
      GetCommonPlaneOverlapCentroid( cntctElem, centroid );

      RealT xTri[3];
      RealT yTri[3];
      RealT zTri[3];
      const RealT visc = 0.5 *
                         ( mesh1.getElementData().m_viscous_damping_coeff + mesh2.getElementData().m_viscous_damping_coeff );

      for ( int j = 0; j < numPolyVert; ++j ) {
        const int next = ( j == numPolyVert - 1 ) ? 0 : j + 1;
        xTri[0] = xVert[dim * j];
        yTri[0] = xVert[dim * j + 1];
        zTri[0] = xVert[dim * j + 2];
        xTri[1] = xVert[dim * next];
        yTri[1] = xVert[dim * next + 1];
        zTri[1] = xVert[dim * next + 2];
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
          RealT x_q[3];
          x_q[0] = n0 * xTri[0] + xi * xTri[1] + eta * xTri[2];
          x_q[1] = n0 * yTri[0] + xi * yTri[1] + eta * yTri[2];
          x_q[2] = n0 * zTri[0] + xi * zTri[1] + eta * zTri[2];

          RealT phi_q1[max_nodes_per_face] = { 0., 0., 0., 0. };
          RealT phi_q2[max_nodes_per_face] = { 0., 0., 0., 0. };
          RealT x_qf1[max_dim];
          RealT x_qf2[max_dim];
          RealT vel_q1[max_dim];
          RealT vel_q2[max_dim];

          EvalLinearFaceAtPoint( actual_xf1, numNodesPerFace1, x_q, x_qf1, phi_q1, dim, &actual_vf1[0], vel_q1 );
          EvalLinearFaceAtPoint( actual_xf2, numNodesPerFace2, x_q, x_qf2, phi_q2, dim, &actual_vf2[0], vel_q2 );

          RealT local_gap = ( x_qf1[0] - x_qf2[0] ) * overlapNormal[0] + ( x_qf1[1] - x_qf2[1] ) * overlapNormal[1] +
                            ( x_qf1[2] - x_qf2[2] ) * overlapNormal[2];
          RealT gap_tol = cs_view.getGapTol( index1, index2 );
          if ( local_gap > gap_tol ) {
            continue;
          }

          RealT velGap_q[max_dim];
          velGap_q[0] = vel_q1[0] - vel_q2[0];
          velGap_q[1] = vel_q1[1] - vel_q2[1];
          velGap_q[2] = vel_q1[2] - vel_q2[2];

          RealT velGap_dot_n = velGap_q[0] * overlapNormal[0] + velGap_q[1] * overlapNormal[1] + velGap_q[2] * overlapNormal[2];
          RealT velGapTan[max_dim];
          velGapTan[0] = velGap_q[0] - velGap_dot_n * overlapNormal[0];
          velGapTan[1] = velGap_q[1] - velGap_dot_n * overlapNormal[1];
          velGapTan[2] = velGap_q[2] - velGap_dot_n * overlapNormal[2];

          const RealT weighted_force = area * rule_wts[qp] * visc;
          const RealT force_x = weighted_force * velGapTan[0];
          const RealT force_y = weighted_force * velGapTan[1];
          const RealT force_z = weighted_force * velGapTan[2];

          AccumulateContactForce( mesh1, mesh2, index1, index2, dim, numNodesPerFace1, force_x, force_y, force_z,
                                  phi_q1, phi_q2 );
        }
      }

      return;
    }

    /////////////////////////////////////////////////////
    // Computation of tangential viscous damping force //
    /////////////////////////////////////////////////////
    RealT visc =
        0.5 * ( mesh1.getElementData().m_viscous_damping_coeff + mesh2.getElementData().m_viscous_damping_coeff );
    RealT force_x = visc * velGapTan[0];
    RealT force_y = visc * velGapTan[1];
    RealT force_z = 0.;
    if ( dim == 3 ) {
      force_z = visc * velGapTan[2];
    }

    AccumulateContactForce( mesh1, mesh2, index1, index2, dim, numNodesPerFace1, force_x, force_y, force_z, phi1, phi2 );
  } );

  return 0;
}
//------------------------------------------------------------------------------

}  // namespace tribol
