// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
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

TRIBOL_HOST_DEVICE inline bool Solve3x3( const RealT A[3][3], const RealT b[3], RealT x[3] )
{
  const RealT detA = A[0][0] * ( A[1][1] * A[2][2] - A[1][2] * A[2][1] ) -
                     A[0][1] * ( A[1][0] * A[2][2] - A[1][2] * A[2][0] ) +
                     A[0][2] * ( A[1][0] * A[2][1] - A[1][1] * A[2][0] );
  constexpr RealT det_tol = 1.e-15;
  if ( std::abs( detA ) <= det_tol ) {
    return false;
  }

  const RealT inv_detA = 1. / detA;
  x[0] = inv_detA * ( b[0] * ( A[1][1] * A[2][2] - A[1][2] * A[2][1] ) - A[0][1] * ( b[1] * A[2][2] - A[1][2] * b[2] ) +
                      A[0][2] * ( b[1] * A[2][1] - A[1][1] * b[2] ) );
  x[1] = inv_detA * ( A[0][0] * ( b[1] * A[2][2] - A[1][2] * b[2] ) - b[0] * ( A[1][0] * A[2][2] - A[1][2] * A[2][0] ) +
                      A[0][2] * ( A[1][0] * b[2] - b[1] * A[2][0] ) );
  x[2] = inv_detA * ( A[0][0] * ( A[1][1] * b[2] - b[1] * A[2][1] ) - A[0][1] * ( A[1][0] * b[2] - b[1] * A[2][0] ) +
                      b[0] * ( A[1][0] * A[2][1] - A[1][1] * A[2][0] ) );
  return true;
}

TRIBOL_HOST_DEVICE inline void AccumulateFaceInterpolation( const RealT* face_coords, const int num_nodes,
                                                            const RealT* phi, RealT x_face[3], const int value_dim = 0,
                                                            const RealT* nodal_vals = nullptr, RealT* values = nullptr )
{
  initRealArray( x_face, max_dim, 0. );
  if ( values != nullptr ) {
    initRealArray( values, value_dim, 0. );
  }

  for ( int a = 0; a < num_nodes; ++a ) {
    x_face[0] += face_coords[3 * a] * phi[a];
    x_face[1] += face_coords[3 * a + 1] * phi[a];
    x_face[2] += face_coords[3 * a + 2] * phi[a];

    if ( values != nullptr ) {
      for ( int i = 0; i < value_dim; ++i ) {
        values[i] += nodal_vals[i + a * value_dim] * phi[a];
      }
    }
  }
}

TRIBOL_HOST_DEVICE inline bool EvalLinearFaceAtProjectedPoint( const RealT* face_coords, const int num_nodes,
                                                               const RealT x_query[3], const RealT projection_dir[3],
                                                               RealT x_face[3], RealT* phi, const int value_dim = 0,
                                                               const RealT* nodal_vals = nullptr,
                                                               RealT* values = nullptr )
{
  // CommonPlane quadrature points lie on the overlap polygon. Evaluate the face
  // fields at the corresponding on-face point found by projecting along the
  // common-plane normal rather than by an off-surface closest-point inverse map.
  initRealArray( phi, num_nodes, 0. );

  if ( num_nodes == 3 ) {
    const RealT x0[3] = { face_coords[0], face_coords[1], face_coords[2] };
    const RealT e1[3] = { face_coords[3] - x0[0], face_coords[4] - x0[1], face_coords[5] - x0[2] };
    const RealT e2[3] = { face_coords[6] - x0[0], face_coords[7] - x0[1], face_coords[8] - x0[2] };

    RealT n_face[3];
    crossProd( e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], n_face[0], n_face[1], n_face[2] );

    const RealT denom =
        dotProd( n_face[0], n_face[1], n_face[2], projection_dir[0], projection_dir[1], projection_dir[2] );
    constexpr RealT parallel_tol = 1.e-14;
    if ( std::abs( denom ) <= parallel_tol ) {
      return false;
    }

    const RealT dx0 = x0[0] - x_query[0];
    const RealT dy0 = x0[1] - x_query[1];
    const RealT dz0 = x0[2] - x_query[2];
    const RealT step = dotProd( n_face[0], n_face[1], n_face[2], dx0, dy0, dz0 ) / denom;

    x_face[0] = x_query[0] + step * projection_dir[0];
    x_face[1] = x_query[1] + step * projection_dir[1];
    x_face[2] = x_query[2] + step * projection_dir[2];

    const RealT r[3] = { x_face[0] - x0[0], x_face[1] - x0[1], x_face[2] - x0[2] };
    const RealT gram11 = dotProd( e1[0], e1[1], e1[2], e1[0], e1[1], e1[2] );
    const RealT gram12 = dotProd( e1[0], e1[1], e1[2], e2[0], e2[1], e2[2] );
    const RealT gram22 = dotProd( e2[0], e2[1], e2[2], e2[0], e2[1], e2[2] );
    const RealT rhs1 = dotProd( e1[0], e1[1], e1[2], r[0], r[1], r[2] );
    const RealT rhs2 = dotProd( e2[0], e2[1], e2[2], r[0], r[1], r[2] );

    const RealT gram_det = gram11 * gram22 - gram12 * gram12;
    constexpr RealT gram_tol = 1.e-15;
    if ( std::abs( gram_det ) <= gram_tol ) {
      return false;
    }

    const RealT xi = ( gram22 * rhs1 - gram12 * rhs2 ) / gram_det;
    const RealT eta = ( gram11 * rhs2 - gram12 * rhs1 ) / gram_det;
    phi[0] = 1. - xi - eta;
    phi[1] = xi;
    phi[2] = eta;
  } else if ( num_nodes == 4 ) {
    constexpr int max_iter = 25;
    constexpr RealT step_tol = 1.e-12;
    constexpr RealT residual_tol = 1.e-12;
    constexpr RealT xi_tol = 1.e-8;

    RealT xi = 0.;
    RealT eta = 0.;
    RealT phi0[max_nodes_per_face] = { 0.25, 0.25, 0.25, 0.25 };
    RealT x_center[3];
    AccumulateFaceInterpolation( face_coords, num_nodes, phi0, x_center );
    RealT s = ( x_center[0] - x_query[0] ) * projection_dir[0] + ( x_center[1] - x_query[1] ) * projection_dir[1] +
              ( x_center[2] - x_query[2] ) * projection_dir[2];
    bool converged = false;

    for ( int iter = 0; iter < max_iter; ++iter ) {
      const RealT xi_node[4] = { 1., -1., -1., 1. };
      const RealT eta_node[4] = { 1., 1., -1., -1. };
      RealT dxdxi[3] = { 0., 0., 0. };
      RealT dxdeta[3] = { 0., 0., 0. };
      initRealArray( phi, num_nodes, 0. );
      initRealArray( x_face, max_dim, 0. );

      for ( int a = 0; a < num_nodes; ++a ) {
        phi[a] = 0.25 * ( 1. + xi_node[a] * xi ) * ( 1. + eta_node[a] * eta );
        const RealT dphi_dxi = 0.25 * xi_node[a] * ( 1. + eta_node[a] * eta );
        const RealT dphi_deta = 0.25 * eta_node[a] * ( 1. + xi_node[a] * xi );

        const RealT xa = face_coords[3 * a];
        const RealT ya = face_coords[3 * a + 1];
        const RealT za = face_coords[3 * a + 2];

        x_face[0] += xa * phi[a];
        x_face[1] += ya * phi[a];
        x_face[2] += za * phi[a];

        dxdxi[0] += xa * dphi_dxi;
        dxdxi[1] += ya * dphi_dxi;
        dxdxi[2] += za * dphi_dxi;

        dxdeta[0] += xa * dphi_deta;
        dxdeta[1] += ya * dphi_deta;
        dxdeta[2] += za * dphi_deta;
      }

      RealT residual[3] = { x_face[0] - x_query[0] - s * projection_dir[0],
                            x_face[1] - x_query[1] - s * projection_dir[1],
                            x_face[2] - x_query[2] - s * projection_dir[2] };
      const RealT residual_norm = magnitude( residual[0], residual[1], residual[2] );
      if ( residual_norm <= residual_tol ) {
        converged = true;
        break;
      }

      RealT J[3][3] = { { dxdxi[0], dxdeta[0], -projection_dir[0] },
                        { dxdxi[1], dxdeta[1], -projection_dir[1] },
                        { dxdxi[2], dxdeta[2], -projection_dir[2] } };
      RealT rhs[3] = { -residual[0], -residual[1], -residual[2] };
      RealT delta[3];
      if ( !Solve3x3( J, rhs, delta ) ) {
        return false;
      }

      xi += delta[0];
      eta += delta[1];
      s += delta[2];

      const RealT step_norm = magnitude( delta[0], delta[1], delta[2] );
      if ( step_norm <= step_tol ) {
        converged = true;
        break;
      }
    }

    if ( !converged || xi < -1. - xi_tol || xi > 1. + xi_tol || eta < -1. - xi_tol || eta > 1. + xi_tol ) {
      return false;
    }

    initRealArray( phi, num_nodes, 0. );
    phi[0] = 0.25 * ( 1. + xi ) * ( 1. + eta );
    phi[1] = 0.25 * ( 1. - xi ) * ( 1. + eta );
    phi[2] = 0.25 * ( 1. - xi ) * ( 1. - eta );
    phi[3] = 0.25 * ( 1. + xi ) * ( 1. - eta );
    AccumulateFaceInterpolation( face_coords, num_nodes, phi, x_face );

    const RealT line_residual = magnitude( x_face[0] - x_query[0], x_face[1] - x_query[1], x_face[2] - x_query[2] );
    const RealT normal_step = ( x_face[0] - x_query[0] ) * projection_dir[0] +
                              ( x_face[1] - x_query[1] ) * projection_dir[1] +
                              ( x_face[2] - x_query[2] ) * projection_dir[2];
    const RealT projection_residual = magnitude( x_face[0] - x_query[0] - normal_step * projection_dir[0],
                                                 x_face[1] - x_query[1] - normal_step * projection_dir[1],
                                                 x_face[2] - x_query[2] - normal_step * projection_dir[2] );
    if ( line_residual > 0. && projection_residual > residual_tol * line_residual ) {
      return false;
    }
  } else {
    return false;
  }

  if ( values != nullptr ) {
    initRealArray( values, value_dim, 0. );
    for ( int a = 0; a < num_nodes; ++a ) {
      for ( int i = 0; i < value_dim; ++i ) {
        values[i] += nodal_vals[i + a * value_dim] * phi[a];
      }
    }
  }

  return true;
}

TRIBOL_HOST_DEVICE inline bool EvalLinearEdgeAtProjectedPoint( const RealT* edge_coords, const RealT x_query[2],
                                                               const RealT projection_dir[2], RealT x_edge[3],
                                                               RealT* phi, const int value_dim = 0,
                                                               const RealT* nodal_vals = nullptr,
                                                               RealT* values = nullptr )
{
  const RealT ax = edge_coords[0];
  const RealT ay = edge_coords[1];
  const RealT bx = edge_coords[2];
  const RealT by = edge_coords[3];
  const RealT ex = bx - ax;
  const RealT ey = by - ay;

  const RealT det = projection_dir[0] * ey - ex * projection_dir[1];
  constexpr RealT det_tol = 1.e-14;
  if ( std::abs( det ) <= det_tol ) {
    return false;
  }

  const RealT rhs_x = x_query[0] - ax;
  const RealT rhs_y = x_query[1] - ay;
  RealT edge_param = ( projection_dir[0] * rhs_y - projection_dir[1] * rhs_x ) / det;

  constexpr RealT edge_tol = 1.e-8;
  if ( edge_param < -edge_tol || edge_param > 1. + edge_tol ) {
    return false;
  }
  edge_param = std::max( 0., std::min( 1., edge_param ) );

  phi[0] = 1. - edge_param;
  phi[1] = edge_param;

  x_edge[0] = ax + edge_param * ex;
  x_edge[1] = ay + edge_param * ey;
  x_edge[2] = 0.;

  if ( values != nullptr ) {
    initRealArray( values, value_dim, 0. );
    for ( int a = 0; a < 2; ++a ) {
      for ( int i = 0; i < value_dim; ++i ) {
        values[i] += nodal_vals[i + a * value_dim] * phi[a];
      }
    }
  }

  return true;
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
  }  // end switch on rate_calc
}

TRIBOL_HOST_DEVICE inline RealT ComputeInverseEffectiveMass( const MeshData::Viewer& mesh1,
                                                             const MeshData::Viewer& mesh2, IndexT index1,
                                                             IndexT index2, int dim, int num_nodes_per_face,
                                                             const RealT* normal,
                                                             const RealT* phi1, const RealT* phi2 )
{
  RealT inverse_effective_mass = 0.;
  for ( int a = 0; a < num_nodes_per_face; ++a ) {
    const IndexT node1 = mesh1.getGlobalNodeId( index1, a );
    const IndexT node2 = mesh2.getGlobalNodeId( index2, a );
    for ( int d = 0; d < dim; ++d ) {
      const RealT constraint1 = phi1[a] * normal[d];
      const RealT constraint2 = phi2[a] * normal[d];
      inverse_effective_mass += constraint1 * constraint1 * mesh1.getInverseMass( node1, d );
      inverse_effective_mass += constraint2 * constraint2 * mesh2.getInverseMass( node2, d );
    }
  }
  return inverse_effective_mass;
}

TRIBOL_HOST_DEVICE inline RealT ComputePredictorTargetVelocity( RealT gap, RealT dt )
{
  return dt > 0. && gap > 0. ? -gap / dt : 0.;
}

TRIBOL_HOST_DEVICE inline RealT ComputePredictorPressure( const MeshData::Viewer& mesh1,
                                                          const MeshData::Viewer& mesh2, IndexT index1,
                                                          IndexT index2, int dim, int num_nodes_per_face,
                                                          const RealT* normal, const RealT* phi1, const RealT* phi2,
                                                          RealT gap, RealT velocity_gap,
                                                          RealT quadrature_measure, RealT dt,
                                                          RealT relaxation )
{
  if ( dt <= 0. || quadrature_measure <= 0. ) {
    return 0.;
  }
  const RealT inverse_effective_mass =
      ComputeInverseEffectiveMass( mesh1, mesh2, index1, index2, dim, num_nodes_per_face, normal, phi1, phi2 );
  if ( inverse_effective_mass <= 0. ) {
    return 0.;
  }

  const RealT target_velocity = ComputePredictorTargetVelocity( gap, dt );
  if ( velocity_gap >= target_velocity ) {
    return 0.;
  }

  const RealT impulse = relaxation * ( target_velocity - velocity_gap ) / inverse_effective_mass;
  return -impulse / ( quadrature_measure * dt );
}

TRIBOL_HOST_DEVICE inline void AccumulateConstraintRowBounds(
    const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2, IndexT index1, IndexT index2, int dim,
    int num_nodes_per_face, const RealT* normal, const RealT* phi1, const RealT* phi2, RealT quadrature_measure,
    RealT penalty_stiffness, bool include_predictor, ArrayViewT<RealT> predictor_rows1,
    ArrayViewT<RealT> predictor_rows2, ArrayViewT<RealT> stiffness_rows1, ArrayViewT<RealT> stiffness_rows2 )
{
  const RealT inverse_effective_mass =
      ComputeInverseEffectiveMass( mesh1, mesh2, index1, index2, dim, num_nodes_per_face, normal, phi1, phi2 );
  if ( inverse_effective_mass <= 0. ) {
    return;
  }

  const RealT inv_sqrt_effective = 1. / sqrt( inverse_effective_mass );
  RealT predictor_l1 = 0.;
  RealT stiffness_l1 = 0.;
  for ( int a = 0; a < num_nodes_per_face; ++a ) {
    const IndexT node1 = mesh1.getGlobalNodeId( index1, a );
    const IndexT node2 = mesh2.getGlobalNodeId( index2, a );
    for ( int d = 0; d < dim; ++d ) {
      const RealT sqrt_inv_mass1 = sqrt( mesh1.getInverseMass( node1, d ) );
      const RealT sqrt_inv_mass2 = sqrt( mesh2.getInverseMass( node2, d ) );
      predictor_l1 += std::abs( phi1[a] * normal[d] * sqrt_inv_mass1 * inv_sqrt_effective );
      predictor_l1 += std::abs( phi2[a] * normal[d] * sqrt_inv_mass2 * inv_sqrt_effective );
      stiffness_l1 += std::abs( phi1[a] * normal[d] * sqrt_inv_mass1 );
      stiffness_l1 += std::abs( phi2[a] * normal[d] * sqrt_inv_mass2 );
    }
  }

  const RealT stiffness_scale = penalty_stiffness * quadrature_measure;
  for ( int a = 0; a < num_nodes_per_face; ++a ) {
    const IndexT node1 = mesh1.getGlobalNodeId( index1, a );
    const IndexT node2 = mesh2.getGlobalNodeId( index2, a );
    for ( int d = 0; d < dim; ++d ) {
      const RealT sqrt_inv_mass1 = sqrt( mesh1.getInverseMass( node1, d ) );
      const RealT sqrt_inv_mass2 = sqrt( mesh2.getInverseMass( node2, d ) );
      const RealT predictor_h1 = std::abs( phi1[a] * normal[d] * sqrt_inv_mass1 * inv_sqrt_effective );
      const RealT predictor_h2 = std::abs( phi2[a] * normal[d] * sqrt_inv_mass2 * inv_sqrt_effective );
      const RealT stiffness_h1 = std::abs( phi1[a] * normal[d] * sqrt_inv_mass1 );
      const RealT stiffness_h2 = std::abs( phi2[a] * normal[d] * sqrt_inv_mass2 );
      if ( include_predictor ) {
        tribol::atomicAdd( &predictor_rows1[node1 * dim + d], predictor_h1 * predictor_l1 );
        tribol::atomicAdd( &predictor_rows2[node2 * dim + d], predictor_h2 * predictor_l1 );
      }
      tribol::atomicAdd( &stiffness_rows1[node1 * dim + d], stiffness_scale * stiffness_h1 * stiffness_l1 );
      tribol::atomicAdd( &stiffness_rows2[node2 * dim + d], stiffness_scale * stiffness_h2 * stiffness_l1 );
    }
  }
}

TRIBOL_HOST_DEVICE inline void AccumulateForceDiagnostics(
    bool predictor_active, RealT quadrature_measure, RealT penalty_pressure, RealT predictor_pressure,
    RealT applied_pressure, ArrayViewT<int> predictor_counts, ArrayViewT<RealT> integrated_forces )
{
  if ( predictor_active ) {
    tribol::atomicInc( &predictor_counts[0] );
    if ( predictor_pressure < penalty_pressure ) {
      tribol::atomicInc( &predictor_counts[1] );
    }
  }

  const RealT penalty_force = penalty_pressure < 0. ? -quadrature_measure * penalty_pressure : 0.;
  const RealT predictor_force = predictor_pressure < 0. ? -quadrature_measure * predictor_pressure : 0.;
  const RealT applied_force = applied_pressure < 0. ? -quadrature_measure * applied_pressure : 0.;
  tribol::atomicAdd( &integrated_forces[0], penalty_force );
  tribol::atomicAdd( &integrated_forces[1], predictor_force );
  tribol::atomicAdd( &integrated_forces[2], applied_force );
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
  const auto& penalty_options = cs->getEnforcementOptions().penalty_options;
  const bool dissipative = penalty_options.constraint_type == KINEMATIC_AND_DISSIPATIVE;
  const bool multipoint = penalty_options.common_plane_rule == MULTI_POINT;

  if ( dissipative ) {
    SLIC_ERROR_ROOT_IF( !multipoint,
                        "Dissipative common-plane enforcement requires MULTI_POINT integration." );
    SLIC_ERROR_ROOT_IF( !cs->getMesh1().hasInverseMass() || !cs->getMesh2().hasInverseMass(),
                        "Dissipative common-plane enforcement requires registered inverse nodal masses." );
  }

  RealT predictor_relaxation = 1.;
  if ( cs->getMesh1().hasInverseMass() && cs->getMesh2().hasInverseMass() ) {
    const int dim = cs->spatialDimension();
    ArrayT<RealT> predictor_rows1( cs->getMesh1().numberOfNodes() * dim, cs->getMesh1().numberOfNodes() * dim,
                                   cs->getAllocatorId() );
    ArrayT<RealT> predictor_rows2( cs->getMesh2().numberOfNodes() * dim, cs->getMesh2().numberOfNodes() * dim,
                                   cs->getAllocatorId() );
    ArrayT<RealT> stiffness_rows1( cs->getMesh1().numberOfNodes() * dim, cs->getMesh1().numberOfNodes() * dim,
                                   cs->getAllocatorId() );
    ArrayT<RealT> stiffness_rows2( cs->getMesh2().numberOfNodes() * dim, cs->getMesh2().numberOfNodes() * dim,
                                   cs->getAllocatorId() );
    predictor_rows1.fill( 0. );
    predictor_rows2.fill( 0. );
    stiffness_rows1.fill( 0. );
    stiffness_rows2.fill( 0. );
    auto predictor_view1 = predictor_rows1.view();
    auto predictor_view2 = predictor_rows2.view();
    auto stiffness_view1 = stiffness_rows1.view();
    auto stiffness_view2 = stiffness_rows2.view();
    const RealT stage_dt = cs->getCurrentTimeStep();

    forAllExec( cs->getExecutionMode(), num_pairs,
                [cs_view, predictor_view1, predictor_view2, stiffness_view1, stiffness_view2, dissipative,
                 stage_dt] TRIBOL_HOST_DEVICE( IndexT i ) {
                  auto& plane = cs_view.getCompGeomView().getCommonPlane( i );
                  const auto& mesh1 = cs_view.getMesh1View();
                  const auto& mesh2 = cs_view.getMesh2View();
                  const IndexT index1 = plane.getCpElementId1();
                  const IndexT index2 = plane.getCpElementId2();
                  const int local_dim = cs_view.spatialDimension();
                  const int num_nodes_per_face = mesh1.numberOfNodesPerElement();
                  const auto& options = cs_view.getEnforcementOptions().penalty_options;

                  RealT penalty_stiffness = 0.;
                  const RealT scale1 = mesh1.getElementData().m_penalty_scale;
                  const RealT scale2 = mesh2.getElementData().m_penalty_scale;
                  if ( options.kinematic_calculation == KINEMATIC_CONSTANT ) {
                    penalty_stiffness = ComputePenaltyStiffnessPerArea(
                        scale1 * mesh1.getElementData().m_penalty_stiffness,
                        scale2 * mesh2.getElementData().m_penalty_stiffness );
                  } else if ( options.kinematic_calculation == KINEMATIC_ELEMENT ) {
                    const RealT thickness1 = mesh1.getElementData().m_thickness[index1] + options.tiny_length;
                    const RealT thickness2 = mesh2.getElementData().m_thickness[index2] + options.tiny_length;
                    if ( thickness1 <= 0. || thickness2 <= 0. ) {
                      return;
                    }
                    penalty_stiffness = ComputePenaltyStiffnessPerArea(
                        scale1 * mesh1.getElementData().m_mat_mod[index1] / thickness1,
                        scale2 * mesh2.getElementData().m_mat_mod[index2] / thickness2 );
                  }

                  StackArrayT<RealT, max_dim * max_nodes_per_face> face1;
                  StackArrayT<RealT, max_dim * max_nodes_per_face> face2;
                  StackArrayT<RealT, max_dim * max_nodes_per_face> velocity1;
                  StackArrayT<RealT, max_dim * max_nodes_per_face> velocity2;
                  mesh1.getFaceCoords( index1, face1 );
                  mesh2.getFaceCoords( index2, face2 );
                  if ( dissipative ) {
                    mesh1.getFaceVelocities( index1, velocity1 );
                    mesh2.getFaceVelocities( index2, velocity2 );
                  }

                  RealT normal[max_dim] = { plane.m_nX, plane.m_nY, plane.m_nZ };
                  RealT overlap_vertices[max_dim * max_nodes_per_overlap] = { 0. };
                  plane.getOverlapVertices( overlap_vertices );
                  const RealT gap_tolerance = cs_view.getGapTol( index1, index2 );

                  if ( options.common_plane_rule != MULTI_POINT ) {
                    RealT projected_face1[max_dim * max_nodes_per_face] = { 0. };
                    RealT projected_face2[max_dim * max_nodes_per_face] = { 0. };
                    plane.getFace1Coords( projected_face1, num_nodes_per_face );
                    plane.getFace2Coords( projected_face2, num_nodes_per_face );
                    const int num_vertices = local_dim == 3 ? plane.m_numPolyVert : 2;
                    SurfaceContactElem contact_element( local_dim, projected_face1, projected_face2, overlap_vertices,
                                                        num_nodes_per_face, num_vertices, &mesh1, &mesh2, index1, index2 );
                    RealT phi1[max_nodes_per_face] = { 0. };
                    RealT phi2[max_nodes_per_face] = { 0. };
                    EvalWeakFormIntegralCommonPlane( contact_element, options.common_plane_rule,
                                                     options.common_plane_quadrature_order, phi1, phi2 );
                    if ( plane.m_gap <= gap_tolerance ) {
                      AccumulateConstraintRowBounds(
                          mesh1, mesh2, index1, index2, local_dim, num_nodes_per_face, normal, phi1, phi2, plane.m_area,
                          penalty_stiffness, false, predictor_view1, predictor_view2, stiffness_view1, stiffness_view2 );
                    }
                    return;
                  }

                  if ( local_dim == 3 ) {
                    RealT rule_weights[max_symmetric_triangle_qpts] = { 0. };
                    RealT rule_coordinates[2 * max_symmetric_triangle_qpts] = { 0. };
                    const int num_qpts =
                        GetCommonPlaneTriangleRule( options.common_plane_quadrature_order, rule_weights, rule_coordinates );
                    RealT face_coords1[max_dim * max_nodes_per_face] = { 0. };
                    RealT face_coords2[max_dim * max_nodes_per_face] = { 0. };
                    plane.getFace1Coords( face_coords1, num_nodes_per_face );
                    plane.getFace2Coords( face_coords2, num_nodes_per_face );
                    const int num_vertices = plane.m_numPolyVert;
                    SurfaceContactElem contact_element( local_dim, face_coords1, face_coords2, overlap_vertices,
                                                        num_nodes_per_face, num_vertices, &mesh1, &mesh2, index1, index2 );
                    RealT centroid[3];
                    GetCommonPlaneOverlapCentroid( contact_element, centroid );
                    for ( int vertex = 0; vertex < num_vertices; ++vertex ) {
                      const int next = vertex == num_vertices - 1 ? 0 : vertex + 1;
                      RealT triangle_x[3] = { overlap_vertices[3 * vertex], overlap_vertices[3 * next], centroid[0] };
                      RealT triangle_y[3] = { overlap_vertices[3 * vertex + 1], overlap_vertices[3 * next + 1],
                                              centroid[1] };
                      RealT triangle_z[3] = { overlap_vertices[3 * vertex + 2], overlap_vertices[3 * next + 2],
                                              centroid[2] };
                      const RealT area = Area3DTri( triangle_x, triangle_y, triangle_z );
                      if ( area <= 0. ) {
                        continue;
                      }
                      for ( int qp = 0; qp < num_qpts; ++qp ) {
                        const RealT xi = rule_coordinates[2 * qp];
                        const RealT eta = rule_coordinates[2 * qp + 1];
                        const RealT n0 = 1. - xi - eta;
                        RealT x_q[3] = { n0 * triangle_x[0] + xi * triangle_x[1] + eta * triangle_x[2],
                                         n0 * triangle_y[0] + xi * triangle_y[1] + eta * triangle_y[2],
                                         n0 * triangle_z[0] + xi * triangle_z[1] + eta * triangle_z[2] };
                        RealT phi1[max_nodes_per_face] = { 0. };
                        RealT phi2[max_nodes_per_face] = { 0. };
                        RealT x_face1[max_dim], x_face2[max_dim];
                        RealT vel1[max_dim] = { 0. }, vel2[max_dim] = { 0. };
                        if ( !EvalLinearFaceAtProjectedPoint(
                                 face1, num_nodes_per_face, x_q, normal, x_face1, phi1,
                                 dissipative ? local_dim : 0, dissipative ? &velocity1[0] : nullptr,
                                 dissipative ? vel1 : nullptr ) ||
                             !EvalLinearFaceAtProjectedPoint(
                                 face2, num_nodes_per_face, x_q, normal, x_face2, phi2,
                                 dissipative ? local_dim : 0, dissipative ? &velocity2[0] : nullptr,
                                 dissipative ? vel2 : nullptr ) ) {
                          continue;
                        }
                        const RealT gap = ( x_face1[0] - x_face2[0] ) * normal[0] +
                                          ( x_face1[1] - x_face2[1] ) * normal[1] +
                                          ( x_face1[2] - x_face2[2] ) * normal[2];
                        const RealT velocity_gap = ( vel1[0] - vel2[0] ) * normal[0] +
                                                   ( vel1[1] - vel2[1] ) * normal[1] +
                                                   ( vel1[2] - vel2[2] ) * normal[2];
                        const RealT target_velocity = ComputePredictorTargetVelocity( gap, stage_dt );
                        const bool predictor_active =
                            dissipative && stage_dt > 0. && velocity_gap < target_velocity;
                        if ( gap <= gap_tolerance || predictor_active ) {
                          AccumulateConstraintRowBounds(
                              mesh1, mesh2, index1, index2, local_dim, num_nodes_per_face, normal, phi1, phi2,
                              area * rule_weights[qp], penalty_stiffness, predictor_active, predictor_view1,
                              predictor_view2, stiffness_view1, stiffness_view2 );
                        }
                      }
                    }
                  } else {
                    RealT rule_weights[max_segment_gauss_legendre_qpts] = { 0. };
                    RealT rule_coordinates[max_segment_gauss_legendre_qpts] = { 0. };
                    const int num_qpts =
                        GetCommonPlaneSegmentRule( options.common_plane_quadrature_order, rule_weights, rule_coordinates );
                    const RealT x0 = overlap_vertices[0];
                    const RealT y0 = overlap_vertices[1];
                    const RealT x1 = overlap_vertices[2];
                    const RealT y1 = overlap_vertices[3];
                    const RealT length = magnitude( x1 - x0, y1 - y0 );
                    for ( int qp = 0; qp < num_qpts; ++qp ) {
                      const RealT s = rule_coordinates[qp];
                      RealT x_q[2] = { ( 1. - s ) * x0 + s * x1, ( 1. - s ) * y0 + s * y1 };
                      RealT phi1[max_nodes_per_face] = { 0. };
                      RealT phi2[max_nodes_per_face] = { 0. };
                      RealT x_face1[max_dim], x_face2[max_dim];
                      RealT vel1[max_dim] = { 0. }, vel2[max_dim] = { 0. };
                      if ( !EvalLinearEdgeAtProjectedPoint(
                               face1, x_q, normal, x_face1, phi1, dissipative ? local_dim : 0,
                               dissipative ? &velocity1[0] : nullptr, dissipative ? vel1 : nullptr ) ||
                           !EvalLinearEdgeAtProjectedPoint(
                               face2, x_q, normal, x_face2, phi2, dissipative ? local_dim : 0,
                               dissipative ? &velocity2[0] : nullptr, dissipative ? vel2 : nullptr ) ) {
                        continue;
                      }
                      const RealT gap =
                          ( x_face1[0] - x_face2[0] ) * normal[0] + ( x_face1[1] - x_face2[1] ) * normal[1];
                      const RealT velocity_gap =
                          ( vel1[0] - vel2[0] ) * normal[0] + ( vel1[1] - vel2[1] ) * normal[1];
                      const RealT target_velocity = ComputePredictorTargetVelocity( gap, stage_dt );
                      const bool predictor_active = dissipative && stage_dt > 0. && velocity_gap < target_velocity;
                      if ( gap <= gap_tolerance || predictor_active ) {
                        AccumulateConstraintRowBounds(
                            mesh1, mesh2, index1, index2, local_dim, num_nodes_per_face, normal, phi1, phi2,
                            length * rule_weights[qp], penalty_stiffness, predictor_active, predictor_view1,
                            predictor_view2, stiffness_view1, stiffness_view2 );
                      }
                    }
                  }
                } );

    ArrayT<RealT> maxima_data( { 0., 0. }, cs->getAllocatorId() );
    auto maxima = maxima_data.view();
    forAllExec( cs->getExecutionMode(), predictor_rows1.size(),
                [predictor_view1, stiffness_view1, maxima] TRIBOL_HOST_DEVICE( IndexT i ) {
                  tribol::atomicMax( &maxima[0], predictor_view1[i] );
                  tribol::atomicMax( &maxima[1], stiffness_view1[i] );
                } );
    forAllExec( cs->getExecutionMode(), predictor_rows2.size(),
                [predictor_view2, stiffness_view2, maxima] TRIBOL_HOST_DEVICE( IndexT i ) {
                  tribol::atomicMax( &maxima[0], predictor_view2[i] );
                  tribol::atomicMax( &maxima[1], stiffness_view2[i] );
                } );
    ArrayT<RealT, 1, MemorySpace::Host> maxima_host( maxima_data );
#ifdef TRIBOL_USE_MPI
    int mpi_initialized = 0;
    MPI_Initialized( &mpi_initialized );
    if ( mpi_initialized ) {
      MPI_Allreduce( MPI_IN_PLACE, maxima_host.data(), 2, MPI_DOUBLE, MPI_SUM, cs->getProblemComm() );
    }
#endif
    const RealT predictor_bound = maxima_host[0] > 0. ? maxima_host[0] : 1.;
    predictor_relaxation = std::min( 1., penalty_options.predictor_relaxation_scale / predictor_bound );
    const RealT stiffness_bound = maxima_host[1];
    const RealT stability_dt = stiffness_bound > 0.
        ? 2. * penalty_options.penalty_stability_scale / sqrt( stiffness_bound )
        : std::numeric_limits<RealT>::infinity();
    cs->setPredictorDiagnostics( predictor_bound, predictor_relaxation );
    cs->setPenaltyStabilityTimeStep( stability_dt );
  }

  const RealT stage_dt = cs->getCurrentTimeStep();
  ArrayT<int> predictor_counts_data( { 0, 0 }, cs->getAllocatorId() );
  ArrayT<RealT> integrated_forces_data( { 0., 0., 0. }, cs->getAllocatorId() );
  auto predictor_counts = predictor_counts_data.view();
  auto integrated_forces = integrated_forces_data.view();
  forAllExec( cs->getExecutionMode(), num_pairs,
              [cs_view, err, neg_thickness, predictor_relaxation, stage_dt, predictor_counts,
               integrated_forces] TRIBOL_HOST_DEVICE( IndexT i ) {
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
    const bool use_multi_point = pen_enfrc_options.common_plane_rule == MULTI_POINT;
    if ( !use_multi_point && gap > gap_tol ) {
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
      case KINEMATIC_AND_DISSIPATIVE:
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

    if ( use_multi_point ) {
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_xf1;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_xf2;
      mesh1.getFaceCoords( index1, actual_xf1 );
      mesh2.getFaceCoords( index2, actual_xf2 );

      const bool use_rate = pen_enfrc_options.constraint_type == KINEMATIC_AND_RATE;
      const bool use_dissipative = pen_enfrc_options.constraint_type == KINEMATIC_AND_DISSIPATIVE;
      const bool use_velocity = use_rate || use_dissipative;
      const RealT rate_penalty =
          use_rate ? ComputeRatePenalty( mesh1, mesh2, penalty_stiff_per_area, pen_enfrc_options.rate_calculation )
                   : 0.;

      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_vf1;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_vf2;
      if ( use_velocity ) {
        mesh1.getFaceVelocities( index1, actual_vf1 );
        mesh2.getFaceVelocities( index2, actual_vf2 );
      }

      constexpr int max_qpts = max_symmetric_triangle_qpts;
      RealT rule_wts[max_qpts] = { 0. };
      RealT rule_coords[2 * max_qpts] = { 0. };
      bool has_contact = false;

      if ( dim == 3 ) {
        const int num_qpts =
            GetCommonPlaneTriangleRule( pen_enfrc_options.common_plane_quadrature_order, rule_wts, rule_coords );

        RealT centroid[3];
        GetCommonPlaneOverlapCentroid( cntctElem, centroid );

        RealT xTri[3];
        RealT yTri[3];
        RealT zTri[3];

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

            const bool mapped_face1 = EvalLinearFaceAtProjectedPoint(
                actual_xf1, num_nodes_per_face, x_q, overlapNormal, x_qf1, phi_q1, use_velocity ? dim : 0,
                use_velocity ? &actual_vf1[0] : nullptr, use_velocity ? vel_q1 : nullptr );
            const bool mapped_face2 = EvalLinearFaceAtProjectedPoint(
                actual_xf2, num_nodes_per_face, x_q, overlapNormal, x_qf2, phi_q2, use_velocity ? dim : 0,
                use_velocity ? &actual_vf2[0] : nullptr, use_velocity ? vel_q2 : nullptr );
            if ( !mapped_face1 || !mapped_face2 ) {
              continue;
            }

            RealT local_gap = ( x_qf1[0] - x_qf2[0] ) * overlapNormal[0] + ( x_qf1[1] - x_qf2[1] ) * overlapNormal[1] +
                              ( x_qf1[2] - x_qf2[2] ) * overlapNormal[2];
            RealT local_vel_gap = 0.;
            if ( use_velocity ) {
              local_vel_gap = ( vel_q1[0] - vel_q2[0] ) * overlapNormal[0] +
                              ( vel_q1[1] - vel_q2[1] ) * overlapNormal[1] +
                              ( vel_q1[2] - vel_q2[2] ) * overlapNormal[2];
            }
            const RealT target_velocity = ComputePredictorTargetVelocity( local_gap, stage_dt );
            const bool predictor_active = use_dissipative && stage_dt > 0. && local_vel_gap < target_velocity;
            if ( local_gap > gap_tol && !predictor_active ) {
              continue;
            }

            has_contact = true;

            const RealT penalty_pressure = local_gap <= gap_tol ? local_gap * penalty_stiff_per_area : 0.;
            RealT local_pressure = penalty_pressure;
            if ( use_rate && rate_penalty > 0. ) {
              if ( local_vel_gap <= 0. ) {
                local_pressure += local_vel_gap * rate_penalty;
              }
            }

            const RealT quadrature_measure = area * rule_wts[qp];
            RealT predictor_pressure = 0.;
            if ( use_dissipative ) {
              predictor_pressure = ComputePredictorPressure(
                  mesh1, mesh2, index1, index2, dim, num_nodes_per_face, overlapNormal, phi_q1, phi_q2, local_gap,
                  local_vel_gap, quadrature_measure, stage_dt, predictor_relaxation );
              local_pressure = std::min( local_pressure, predictor_pressure );
            }
            AccumulateForceDiagnostics( predictor_active, quadrature_measure, penalty_pressure, predictor_pressure,
                                        local_pressure, predictor_counts, integrated_forces );

            const RealT weighted_force = quadrature_measure * local_pressure;
            const RealT force_x = overlapNormal[0] * weighted_force;
            const RealT force_y = overlapNormal[1] * weighted_force;
            const RealT force_z = overlapNormal[2] * weighted_force;

            AccumulateContactForce( mesh1, mesh2, index1, index2, dim, num_nodes_per_face, force_x, force_y, force_z,
                                    phi_q1, phi_q2 );
          }
        }
      } else {
        RealT segment_rule_wts[max_segment_gauss_legendre_qpts] = { 0. };
        RealT segment_rule_coords[max_segment_gauss_legendre_qpts] = { 0. };
        const int num_qpts = GetCommonPlaneSegmentRule( pen_enfrc_options.common_plane_quadrature_order,
                                                        segment_rule_wts, segment_rule_coords );
        const RealT x0 = xVert[0];
        const RealT y0 = xVert[1];
        const RealT x1 = xVert[2];
        const RealT y1 = xVert[3];
        const RealT length = magnitude( x1 - x0, y1 - y0 );

        for ( int qp = 0; qp < num_qpts; ++qp ) {
          const RealT s = segment_rule_coords[qp];
          const RealT one_minus_s = 1. - s;
          RealT x_q[2] = { one_minus_s * x0 + s * x1, one_minus_s * y0 + s * y1 };

          RealT phi_q1[max_nodes_per_face] = { 0., 0., 0., 0. };
          RealT phi_q2[max_nodes_per_face] = { 0., 0., 0., 0. };
          RealT x_qf1[max_dim];
          RealT x_qf2[max_dim];
          RealT vel_q1[max_dim] = { 0., 0., 0. };
          RealT vel_q2[max_dim] = { 0., 0., 0. };

          const bool mapped_face1 =
              EvalLinearEdgeAtProjectedPoint( actual_xf1, x_q, overlapNormal, x_qf1, phi_q1, use_velocity ? dim : 0,
                                              use_velocity ? &actual_vf1[0] : nullptr, use_velocity ? vel_q1 : nullptr );
          const bool mapped_face2 =
              EvalLinearEdgeAtProjectedPoint( actual_xf2, x_q, overlapNormal, x_qf2, phi_q2, use_velocity ? dim : 0,
                                              use_velocity ? &actual_vf2[0] : nullptr, use_velocity ? vel_q2 : nullptr );
          if ( !mapped_face1 || !mapped_face2 ) {
            continue;
          }

          RealT local_gap = ( x_qf1[0] - x_qf2[0] ) * overlapNormal[0] + ( x_qf1[1] - x_qf2[1] ) * overlapNormal[1];
          RealT local_vel_gap = 0.;
          if ( use_velocity ) {
            local_vel_gap =
                ( vel_q1[0] - vel_q2[0] ) * overlapNormal[0] + ( vel_q1[1] - vel_q2[1] ) * overlapNormal[1];
          }
          const RealT target_velocity = ComputePredictorTargetVelocity( local_gap, stage_dt );
          const bool predictor_active = use_dissipative && stage_dt > 0. && local_vel_gap < target_velocity;
          if ( local_gap > gap_tol && !predictor_active ) {
            continue;
          }

          has_contact = true;

          const RealT penalty_pressure = local_gap <= gap_tol ? local_gap * penalty_stiff_per_area : 0.;
          RealT local_pressure = penalty_pressure;
          if ( use_rate && rate_penalty > 0. ) {
            if ( local_vel_gap <= 0. ) {
              local_pressure += local_vel_gap * rate_penalty;
            }
          }

          const RealT quadrature_measure = length * segment_rule_wts[qp];
          RealT predictor_pressure = 0.;
          if ( use_dissipative ) {
            predictor_pressure = ComputePredictorPressure(
                mesh1, mesh2, index1, index2, dim, num_nodes_per_face, overlapNormal, phi_q1, phi_q2, local_gap,
                local_vel_gap, quadrature_measure, stage_dt, predictor_relaxation );
            local_pressure = std::min( local_pressure, predictor_pressure );
          }
          AccumulateForceDiagnostics( predictor_active, quadrature_measure, penalty_pressure, predictor_pressure,
                                      local_pressure, predictor_counts, integrated_forces );

          const RealT weighted_force = quadrature_measure * local_pressure;
          const RealT force_x = overlapNormal[0] * weighted_force;
          const RealT force_y = overlapNormal[1] * weighted_force;

          AccumulateContactForce( mesh1, mesh2, index1, index2, dim, num_nodes_per_face, force_x, force_y, 0., phi_q1,
                                  phi_q2 );
        }
      }

      plane.m_inContact = has_contact;
      return;
    }

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
                                     pen_enfrc_options.common_plane_quadrature_order, phi1, phi2 );

    ///////////////////////////////////////////////////////////////////////
    // Computation of full contact nodal force contributions             //
    // (i.e. premultiplication of contact integrals by normal component, //
    //  contact pressure, and overlap area)                              //
    ///////////////////////////////////////////////////////////////////////

    // RealT phi_sum_1 = 0.;
    // RealT phi_sum_2 = 0.;

    // compute contact force (spring force)
    AccumulateForceDiagnostics( false, A, plane.m_pressure, 0., totalPressure, predictor_counts, integrated_forces );
    RealT contact_force = totalPressure * A;

    RealT force_x = overlapNormal[0] * contact_force;
    RealT force_y = overlapNormal[1] * contact_force;
    RealT force_z = 0.;
    if ( dim == 3 ) {
      force_z = overlapNormal[2] * contact_force;
    }

    AccumulateContactForce( mesh1, mesh2, index1, index2, dim, num_nodes_per_face, force_x, force_y, force_z, phi1,
                            phi2 );

    // comment out debug logs; too much output during tests. Keep for easy
    // debugging if needed
    // SLIC_DEBUG("force sum, side 1, pair " << kp << ": " << -dbg_sum_force1 );
    // SLIC_DEBUG("force sum, side 2, pair " << kp << ": " << dbg_sum_force2 );
    // SLIC_DEBUG("phi 1 sum: " << phi_sum_1 );
    // SLIC_DEBUG("phi 2 sum: " << phi_sum_2 );
  } );

  ArrayT<int, 1, MemorySpace::Host> predictor_counts_host( predictor_counts_data );
  ArrayT<RealT, 1, MemorySpace::Host> integrated_forces_host( integrated_forces_data );
#ifdef TRIBOL_USE_MPI
  int mpi_initialized = 0;
  MPI_Initialized( &mpi_initialized );
  if ( mpi_initialized ) {
    MPI_Allreduce( MPI_IN_PLACE, predictor_counts_host.data(), 2, MPI_INT, MPI_SUM, cs->getProblemComm() );
    MPI_Allreduce( MPI_IN_PLACE, integrated_forces_host.data(), 3, MPI_DOUBLE, MPI_SUM, cs->getProblemComm() );
  }
#endif
  cs->setPredictorForceDiagnostics( predictor_counts_host[0], predictor_counts_host[1], integrated_forces_host[0],
                                    integrated_forces_host[1], integrated_forces_host[2] );

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

    const bool use_multi_point = pen_enfrc_options.common_plane_rule == MULTI_POINT;

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

    if ( use_multi_point ) {
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_xf1;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_xf2;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_vf1;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_vf2;
      mesh1.getFaceCoords( index1, actual_xf1 );
      mesh2.getFaceCoords( index2, actual_xf2 );
      mesh1.getFaceVelocities( index1, actual_vf1 );
      mesh2.getFaceVelocities( index2, actual_vf2 );

      constexpr int max_qpts = max_symmetric_triangle_qpts;
      RealT rule_wts[max_qpts] = { 0. };
      RealT rule_coords[2 * max_qpts] = { 0. };
      const RealT visc =
          0.5 * ( mesh1.getElementData().m_viscous_damping_coeff + mesh2.getElementData().m_viscous_damping_coeff );

      if ( dim == 3 ) {
        const int num_qpts =
            GetCommonPlaneTriangleRule( pen_enfrc_options.common_plane_quadrature_order, rule_wts, rule_coords );

        RealT centroid[3];
        GetCommonPlaneOverlapCentroid( cntctElem, centroid );

        RealT xTri[3];
        RealT yTri[3];
        RealT zTri[3];

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

            const bool mapped_face1 = EvalLinearFaceAtProjectedPoint( actual_xf1, numNodesPerFace1, x_q, overlapNormal,
                                                                      x_qf1, phi_q1, dim, &actual_vf1[0], vel_q1 );
            const bool mapped_face2 = EvalLinearFaceAtProjectedPoint( actual_xf2, numNodesPerFace2, x_q, overlapNormal,
                                                                      x_qf2, phi_q2, dim, &actual_vf2[0], vel_q2 );
            if ( !mapped_face1 || !mapped_face2 ) {
              continue;
            }

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

            RealT velGap_dot_n =
                velGap_q[0] * overlapNormal[0] + velGap_q[1] * overlapNormal[1] + velGap_q[2] * overlapNormal[2];
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
      } else {
        RealT segment_rule_wts[max_segment_gauss_legendre_qpts] = { 0. };
        RealT segment_rule_coords[max_segment_gauss_legendre_qpts] = { 0. };
        const int num_qpts = GetCommonPlaneSegmentRule( pen_enfrc_options.common_plane_quadrature_order,
                                                        segment_rule_wts, segment_rule_coords );
        const RealT x0 = xVert[0];
        const RealT y0 = xVert[1];
        const RealT x1_seg = xVert[2];
        const RealT y1_seg = xVert[3];
        const RealT length = magnitude( x1_seg - x0, y1_seg - y0 );

        for ( int qp = 0; qp < num_qpts; ++qp ) {
          const RealT s = segment_rule_coords[qp];
          const RealT one_minus_s = 1. - s;
          RealT x_q[2] = { one_minus_s * x0 + s * x1_seg, one_minus_s * y0 + s * y1_seg };

          RealT phi_q1[max_nodes_per_face] = { 0., 0., 0., 0. };
          RealT phi_q2[max_nodes_per_face] = { 0., 0., 0., 0. };
          RealT x_qf1[max_dim];
          RealT x_qf2[max_dim];
          RealT vel_q1[max_dim] = { 0., 0., 0. };
          RealT vel_q2[max_dim] = { 0., 0., 0. };

          const bool mapped_face1 = EvalLinearEdgeAtProjectedPoint( actual_xf1, x_q, overlapNormal, x_qf1, phi_q1, dim,
                                                                    &actual_vf1[0], vel_q1 );
          const bool mapped_face2 = EvalLinearEdgeAtProjectedPoint( actual_xf2, x_q, overlapNormal, x_qf2, phi_q2, dim,
                                                                    &actual_vf2[0], vel_q2 );
          if ( !mapped_face1 || !mapped_face2 ) {
            continue;
          }

          RealT local_gap = ( x_qf1[0] - x_qf2[0] ) * overlapNormal[0] + ( x_qf1[1] - x_qf2[1] ) * overlapNormal[1];
          RealT gap_tol = cs_view.getGapTol( index1, index2 );
          if ( local_gap > gap_tol ) {
            continue;
          }

          RealT velGap_q[max_dim];
          velGap_q[0] = vel_q1[0] - vel_q2[0];
          velGap_q[1] = vel_q1[1] - vel_q2[1];

          RealT velGap_dot_n = velGap_q[0] * overlapNormal[0] + velGap_q[1] * overlapNormal[1];
          RealT velGapTan_q[max_dim] = { velGap_q[0] - velGap_dot_n * overlapNormal[0],
                                         velGap_q[1] - velGap_dot_n * overlapNormal[1], 0. };

          const RealT weighted_force = length * segment_rule_wts[qp] * visc;
          const RealT force_x = weighted_force * velGapTan_q[0];
          const RealT force_y = weighted_force * velGapTan_q[1];

          AccumulateContactForce( mesh1, mesh2, index1, index2, dim, numNodesPerFace1, force_x, force_y, 0., phi_q1,
                                  phi_q2 );
        }
      }

      return;
    }

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
                                     pen_enfrc_options.common_plane_quadrature_order, phi1, phi2 );

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

    AccumulateContactForce( mesh1, mesh2, index1, index2, dim, numNodesPerFace1, force_x, force_y, force_z, phi1,
                            phi2 );
  } );

  return 0;
}
//------------------------------------------------------------------------------

}  // namespace tribol
