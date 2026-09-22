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

#include <cmath>
#include <limits>

#ifdef TRIBOL_USE_MPI
#include <mpi.h>
#endif

namespace tribol {

namespace {

constexpr int max_dim = 3;
constexpr int max_nodes_per_face = ParentFaceData::max_parent_face_nodes;

/**
 * @brief Compute the equivalent normal rate-penalty coefficient for a face pair.
 *
 * @param first_mesh First contact surface mesh
 * @param second_mesh Second contact surface mesh
 * @param penalty_stiffness Equivalent kinematic penalty stiffness
 * @param rate_calculation Registered rate-penalty calculation
 * @return Equivalent normal rate coefficient per unit overlap measure
 */
TRIBOL_HOST_DEVICE inline RealT ComputeRatePenalty( const MeshData::Viewer& first_mesh,
                                                    const MeshData::Viewer& second_mesh, RealT penalty_stiffness,
                                                    RatePenaltyCalculation rate_calculation )
{
  switch ( rate_calculation ) {
    case NO_RATE_PENALTY: {
      return 0.;
    }
    case RATE_CONSTANT: {
      return 0.5 * ( first_mesh.getElementData().m_rate_penalty_stiffness +
                     second_mesh.getElementData().m_rate_penalty_stiffness );
    }
    case RATE_PERCENT: {
      return penalty_stiffness * 0.5 *
             ( first_mesh.getElementData().m_rate_percent_stiffness +
               second_mesh.getElementData().m_rate_percent_stiffness );
    }
    default:
      return 0.;
  }
}

/**
 * @brief Select multipoint integration for one CommonPlane face pair.
 *
 * Explicit user selections are preserved. Automatic selection retains the
 * legacy single-point rule for ordinary linear Tribol meshes and selects
 * multipoint integration whenever native MFEM parent-face data are available.
 *
 * @param penalty_options CommonPlane penalty options
 * @param first_mesh First contact surface mesh
 * @param second_mesh Second contact surface mesh
 * @return true when the overlap must use multipoint integration
 */
TRIBOL_HOST_DEVICE inline bool UseMultipleIntegrationPoints( const PenaltyEnforcementOptions& penalty_options,
                                                             const MeshData::Viewer& first_mesh,
                                                             const MeshData::Viewer& second_mesh )
{
  if ( penalty_options.common_plane_rule == MULTI_POINT ) {
    return true;
  }
  if ( penalty_options.common_plane_rule == SINGLE_POINT ) {
    return false;
  }
  return first_mesh.hasParentFaceFields() || second_mesh.hasParentFaceFields();
}

/**
 * @brief Select an order-aware CommonPlane quadrature order for one face pair.
 *
 * @param penalty_options CommonPlane penalty options
 * @param first_mesh First contact surface mesh
 * @param second_mesh Second contact surface mesh
 * @param first_face_id First Tribol face identifier
 * @param second_face_id Second Tribol face identifier
 * @return Triangle or segment quadrature order in the supported range [2,10]
 */
TRIBOL_HOST_DEVICE inline int GetCommonPlaneQuadratureOrder( const PenaltyEnforcementOptions& penalty_options,
                                                             const MeshData::Viewer& first_mesh,
                                                             const MeshData::Viewer& second_mesh, IndexT first_face_id,
                                                             IndexT second_face_id )
{
  if ( penalty_options.common_plane_rule != AUTO_INTEGRATION ) {
    return penalty_options.common_plane_quadrature_order;
  }

  int parent_order = 1;
  if ( first_mesh.hasParentFaceFields() ) {
    parent_order = first_mesh.getParentFaceData().m_parent_face_orders[first_face_id];
  }
  if ( second_mesh.hasParentFaceFields() &&
       second_mesh.getParentFaceData().m_parent_face_orders[second_face_id] > parent_order ) {
    parent_order = second_mesh.getParentFaceData().m_parent_face_orders[second_face_id];
  }
  const int requested_order = 2 * parent_order;
  return requested_order < 2 ? 2 : requested_order > 10 ? 10 : requested_order;
}

/**
 * @brief Solve a three-by-three linear system using Cramer's rule.
 *
 * @param matrix System matrix
 * @param right_hand_side System right-hand-side vector
 * @param solution Computed solution vector
 * @return true when the matrix determinant is sufficiently far from zero
 */
TRIBOL_HOST_DEVICE inline bool SolveThreeByThreeLinearSystem( const RealT matrix[3][3], const RealT right_hand_side[3],
                                                              RealT solution[3] )
{
  const RealT determinant = matrix[0][0] * ( matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1] ) -
                            matrix[0][1] * ( matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0] ) +
                            matrix[0][2] * ( matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0] );
  constexpr RealT determinant_tolerance = 1.e-15;
  if ( std::abs( determinant ) <= determinant_tolerance ) {
    return false;
  }

  const RealT inverse_determinant = 1. / determinant;
  solution[0] = inverse_determinant *
                ( right_hand_side[0] * ( matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1] ) -
                  matrix[0][1] * ( right_hand_side[1] * matrix[2][2] - matrix[1][2] * right_hand_side[2] ) +
                  matrix[0][2] * ( right_hand_side[1] * matrix[2][1] - matrix[1][1] * right_hand_side[2] ) );
  solution[1] = inverse_determinant *
                ( matrix[0][0] * ( right_hand_side[1] * matrix[2][2] - matrix[1][2] * right_hand_side[2] ) -
                  right_hand_side[0] * ( matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0] ) +
                  matrix[0][2] * ( matrix[1][0] * right_hand_side[2] - right_hand_side[1] * matrix[2][0] ) );
  solution[2] =
      inverse_determinant * ( matrix[0][0] * ( matrix[1][1] * right_hand_side[2] - right_hand_side[1] * matrix[2][1] ) -
                              matrix[0][1] * ( matrix[1][0] * right_hand_side[2] - right_hand_side[1] * matrix[2][0] ) +
                              right_hand_side[0] * ( matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0] ) );
  return true;
}

/**
 * @brief Evaluate physical position and an optional nodal field from basis values.
 *
 * @param face_coordinates Node-major physical face coordinates
 * @param number_of_nodes Number of face nodes
 * @param spatial_dimension Number of coordinate components per face node
 * @param basis_values Basis values at the evaluation point
 * @param face_position Evaluated physical position
 * @param value_dimension Number of components in the optional nodal field
 * @param nodal_values Optional node-major field values
 * @param interpolated_values Optional evaluated field values
 */
TRIBOL_HOST_DEVICE inline void EvaluateFaceFieldsFromBasis( const RealT* face_coordinates, int number_of_nodes,
                                                            int spatial_dimension, const RealT* basis_values,
                                                            RealT face_position[3], int value_dimension = 0,
                                                            const RealT* nodal_values = nullptr,
                                                            RealT* interpolated_values = nullptr )
{
  initRealArray( face_position, max_dim, 0. );
  if ( interpolated_values != nullptr ) {
    initRealArray( interpolated_values, value_dimension, 0. );
  }

  for ( int node = 0; node < number_of_nodes; ++node ) {
    for ( int component = 0; component < spatial_dimension; ++component ) {
      face_position[component] += face_coordinates[spatial_dimension * node + component] * basis_values[node];
    }

    if ( interpolated_values != nullptr ) {
      for ( int component = 0; component < value_dimension; ++component ) {
        interpolated_values[component] += nodal_values[component + node * value_dimension] * basis_values[node];
      }
    }
  }
}

/**
 * @brief Evaluate a linear face at a point projected along the CommonPlane normal.
 *
 * @param face_coordinates Node-major physical face coordinates
 * @param number_of_nodes Number of linear triangle or quadrilateral nodes
 * @param query_point Physical CommonPlane point to project
 * @param projection_direction Direction used by the projection equation
 * @param face_position Evaluated physical position on the face
 * @param basis_values Evaluated linear face basis values
 * @param field_dimension Number of components in the optional nodal field
 * @param nodal_values Optional node-major field values
 * @param interpolated_values Optional evaluated field values
 * @param lor_reference_coordinates Optional evaluated LOR reference coordinates
 * @return true when the projection converges inside the face tolerance
 */
TRIBOL_HOST_DEVICE inline bool EvaluateLinearFaceAtProjectedPoint(
    const RealT* face_coordinates, const int number_of_nodes, const RealT query_point[3],
    const RealT projection_direction[3], RealT face_position[3], RealT* basis_values, const int field_dimension = 0,
    const RealT* nodal_values = nullptr, RealT* interpolated_values = nullptr,
    RealT* lor_reference_coordinates = nullptr )
{
  // CommonPlane quadrature points lie on the overlap polygon. Evaluate the face
  // fields at the corresponding on-face point found by projecting along the
  // common-plane normal rather than by an off-surface closest-point inverse map.
  initRealArray( basis_values, number_of_nodes, 0. );

  if ( number_of_nodes == 3 ) {
    const RealT first_vertex[3] = { face_coordinates[0], face_coordinates[1], face_coordinates[2] };
    const RealT first_edge[3] = { face_coordinates[3] - first_vertex[0], face_coordinates[4] - first_vertex[1],
                                  face_coordinates[5] - first_vertex[2] };
    const RealT second_edge[3] = { face_coordinates[6] - first_vertex[0], face_coordinates[7] - first_vertex[1],
                                   face_coordinates[8] - first_vertex[2] };

    RealT face_normal[3];
    crossProd( first_edge[0], first_edge[1], first_edge[2], second_edge[0], second_edge[1], second_edge[2],
               face_normal[0], face_normal[1], face_normal[2] );

    const RealT projection_denominator =
        dotProd( face_normal[0], face_normal[1], face_normal[2], projection_direction[0], projection_direction[1],
                 projection_direction[2] );
    constexpr RealT parallel_tolerance = 1.e-14;
    if ( std::abs( projection_denominator ) <= parallel_tolerance ) {
      return false;
    }

    const RealT query_to_first_vertex[3] = { first_vertex[0] - query_point[0], first_vertex[1] - query_point[1],
                                             first_vertex[2] - query_point[2] };
    const RealT projection_distance = dotProd( face_normal[0], face_normal[1], face_normal[2], query_to_first_vertex[0],
                                               query_to_first_vertex[1], query_to_first_vertex[2] ) /
                                      projection_denominator;

    face_position[0] = query_point[0] + projection_distance * projection_direction[0];
    face_position[1] = query_point[1] + projection_distance * projection_direction[1];
    face_position[2] = query_point[2] + projection_distance * projection_direction[2];

    const RealT first_vertex_to_position[3] = { face_position[0] - first_vertex[0], face_position[1] - first_vertex[1],
                                                face_position[2] - first_vertex[2] };
    const RealT first_edge_squared_norm =
        dotProd( first_edge[0], first_edge[1], first_edge[2], first_edge[0], first_edge[1], first_edge[2] );
    const RealT edge_inner_product =
        dotProd( first_edge[0], first_edge[1], first_edge[2], second_edge[0], second_edge[1], second_edge[2] );
    const RealT second_edge_squared_norm =
        dotProd( second_edge[0], second_edge[1], second_edge[2], second_edge[0], second_edge[1], second_edge[2] );
    const RealT first_edge_position_projection =
        dotProd( first_edge[0], first_edge[1], first_edge[2], first_vertex_to_position[0], first_vertex_to_position[1],
                 first_vertex_to_position[2] );
    const RealT second_edge_position_projection =
        dotProd( second_edge[0], second_edge[1], second_edge[2], first_vertex_to_position[0],
                 first_vertex_to_position[1], first_vertex_to_position[2] );

    const RealT gram_determinant =
        first_edge_squared_norm * second_edge_squared_norm - edge_inner_product * edge_inner_product;
    constexpr RealT gram_tolerance = 1.e-15;
    if ( std::abs( gram_determinant ) <= gram_tolerance ) {
      return false;
    }

    const RealT xi = ( second_edge_squared_norm * first_edge_position_projection -
                       edge_inner_product * second_edge_position_projection ) /
                     gram_determinant;
    const RealT eta = ( first_edge_squared_norm * second_edge_position_projection -
                        edge_inner_product * first_edge_position_projection ) /
                      gram_determinant;
    if ( lor_reference_coordinates != nullptr ) {
      lor_reference_coordinates[0] = xi;
      lor_reference_coordinates[1] = eta;
    }
    basis_values[0] = 1. - xi - eta;
    basis_values[1] = xi;
    basis_values[2] = eta;
  } else if ( number_of_nodes == 4 ) {
    constexpr int maximum_iterations = 25;
    constexpr RealT step_tolerance = 1.e-12;
    constexpr RealT residual_tolerance = 1.e-12;
    constexpr RealT reference_coordinate_tolerance = 1.e-8;

    RealT xi = 0.;
    RealT eta = 0.;
    RealT center_basis_values[max_nodes_per_face] = { 0.25, 0.25, 0.25, 0.25 };
    RealT center_position[3];
    EvaluateFaceFieldsFromBasis( face_coordinates, number_of_nodes, max_dim, center_basis_values, center_position );
    RealT projection_distance = ( center_position[0] - query_point[0] ) * projection_direction[0] +
                                ( center_position[1] - query_point[1] ) * projection_direction[1] +
                                ( center_position[2] - query_point[2] ) * projection_direction[2];
    bool converged = false;

    for ( int iteration = 0; iteration < maximum_iterations; ++iteration ) {
      const RealT xi_node_signs[4] = { 1., -1., -1., 1. };
      const RealT eta_node_signs[4] = { 1., 1., -1., -1. };
      RealT position_derivative_xi[3] = { 0., 0., 0. };
      RealT position_derivative_eta[3] = { 0., 0., 0. };
      initRealArray( basis_values, number_of_nodes, 0. );
      initRealArray( face_position, max_dim, 0. );

      for ( int node_index = 0; node_index < number_of_nodes; ++node_index ) {
        basis_values[node_index] =
            0.25 * ( 1. + xi_node_signs[node_index] * xi ) * ( 1. + eta_node_signs[node_index] * eta );
        const RealT basis_derivative_xi = 0.25 * xi_node_signs[node_index] * ( 1. + eta_node_signs[node_index] * eta );
        const RealT basis_derivative_eta = 0.25 * eta_node_signs[node_index] * ( 1. + xi_node_signs[node_index] * xi );

        const RealT node_x = face_coordinates[3 * node_index];
        const RealT node_y = face_coordinates[3 * node_index + 1];
        const RealT node_z = face_coordinates[3 * node_index + 2];

        face_position[0] += node_x * basis_values[node_index];
        face_position[1] += node_y * basis_values[node_index];
        face_position[2] += node_z * basis_values[node_index];

        position_derivative_xi[0] += node_x * basis_derivative_xi;
        position_derivative_xi[1] += node_y * basis_derivative_xi;
        position_derivative_xi[2] += node_z * basis_derivative_xi;

        position_derivative_eta[0] += node_x * basis_derivative_eta;
        position_derivative_eta[1] += node_y * basis_derivative_eta;
        position_derivative_eta[2] += node_z * basis_derivative_eta;
      }

      RealT residual[3] = { face_position[0] - query_point[0] - projection_distance * projection_direction[0],
                            face_position[1] - query_point[1] - projection_distance * projection_direction[1],
                            face_position[2] - query_point[2] - projection_distance * projection_direction[2] };
      const RealT residual_norm = magnitude( residual[0], residual[1], residual[2] );
      if ( residual_norm <= residual_tolerance ) {
        converged = true;
        break;
      }

      RealT jacobian[3][3] = { { position_derivative_xi[0], position_derivative_eta[0], -projection_direction[0] },
                               { position_derivative_xi[1], position_derivative_eta[1], -projection_direction[1] },
                               { position_derivative_xi[2], position_derivative_eta[2], -projection_direction[2] } };
      RealT newton_right_hand_side[3] = { -residual[0], -residual[1], -residual[2] };
      RealT newton_increment[3];
      if ( !SolveThreeByThreeLinearSystem( jacobian, newton_right_hand_side, newton_increment ) ) {
        return false;
      }

      xi += newton_increment[0];
      eta += newton_increment[1];
      projection_distance += newton_increment[2];

      const RealT step_norm = magnitude( newton_increment[0], newton_increment[1], newton_increment[2] );
      if ( step_norm <= step_tolerance ) {
        converged = true;
        break;
      }
    }

    if ( !converged || xi < -1. - reference_coordinate_tolerance || xi > 1. + reference_coordinate_tolerance ||
         eta < -1. - reference_coordinate_tolerance || eta > 1. + reference_coordinate_tolerance ) {
      return false;
    }

    initRealArray( basis_values, number_of_nodes, 0. );
    basis_values[0] = 0.25 * ( 1. + xi ) * ( 1. + eta );
    basis_values[1] = 0.25 * ( 1. - xi ) * ( 1. + eta );
    basis_values[2] = 0.25 * ( 1. - xi ) * ( 1. - eta );
    basis_values[3] = 0.25 * ( 1. + xi ) * ( 1. - eta );
    if ( lor_reference_coordinates != nullptr ) {
      // MFEM's square reference element is [0,1]^2. The Newton solve above
      // uses the equivalent [-1,1]^2 coordinates aligned with Tribol's local
      // quadrilateral vertex ordering.
      lor_reference_coordinates[0] = 0.5 * ( 1.0 - xi );
      lor_reference_coordinates[1] = 0.5 * ( 1.0 - eta );
    }
    EvaluateFaceFieldsFromBasis( face_coordinates, number_of_nodes, max_dim, basis_values, face_position );

    const RealT normal_distance = ( face_position[0] - query_point[0] ) * projection_direction[0] +
                                  ( face_position[1] - query_point[1] ) * projection_direction[1] +
                                  ( face_position[2] - query_point[2] ) * projection_direction[2];
    const RealT projection_residual =
        magnitude( face_position[0] - query_point[0] - normal_distance * projection_direction[0],
                   face_position[1] - query_point[1] - normal_distance * projection_direction[1],
                   face_position[2] - query_point[2] - normal_distance * projection_direction[2] );
    if ( projection_residual > residual_tolerance ) {
      return false;
    }
  } else {
    return false;
  }

  if ( interpolated_values != nullptr ) {
    initRealArray( interpolated_values, field_dimension, 0. );
    for ( int node_index = 0; node_index < number_of_nodes; ++node_index ) {
      for ( int component = 0; component < field_dimension; ++component ) {
        interpolated_values[component] +=
            nodal_values[component + node_index * field_dimension] * basis_values[node_index];
      }
    }
  }

  return true;
}

/**
 * @brief Evaluate a linear edge at a point projected along the CommonPlane normal.
 *
 * @param edge_coordinates Node-major physical edge coordinates
 * @param query_point Physical CommonPlane point to project
 * @param projection_direction Direction used by the projection equation
 * @param edge_position Evaluated physical position on the edge
 * @param basis_values Evaluated linear edge basis values
 * @param field_dimension Number of components in the optional nodal field
 * @param nodal_values Optional node-major field values
 * @param interpolated_values Optional evaluated field values
 * @param lor_reference_coordinates Optional evaluated LOR reference coordinate
 * @return true when the projected point lies inside the edge tolerance
 */
TRIBOL_HOST_DEVICE inline bool EvaluateLinearEdgeAtProjectedPoint(
    const RealT* edge_coordinates, const RealT query_point[2], const RealT projection_direction[2],
    RealT edge_position[3], RealT* basis_values, const int field_dimension = 0, const RealT* nodal_values = nullptr,
    RealT* interpolated_values = nullptr, RealT* lor_reference_coordinates = nullptr )
{
  const RealT first_endpoint_x = edge_coordinates[0];
  const RealT first_endpoint_y = edge_coordinates[1];
  const RealT second_endpoint_x = edge_coordinates[2];
  const RealT second_endpoint_y = edge_coordinates[3];
  const RealT edge_direction_x = second_endpoint_x - first_endpoint_x;
  const RealT edge_direction_y = second_endpoint_y - first_endpoint_y;

  const RealT projection_determinant =
      projection_direction[0] * edge_direction_y - edge_direction_x * projection_direction[1];
  constexpr RealT determinant_tolerance = 1.e-14;
  if ( std::abs( projection_determinant ) <= determinant_tolerance ) {
    return false;
  }

  const RealT query_offset_x = query_point[0] - first_endpoint_x;
  const RealT query_offset_y = query_point[1] - first_endpoint_y;
  RealT edge_parameter =
      ( projection_direction[0] * query_offset_y - projection_direction[1] * query_offset_x ) / projection_determinant;

  constexpr RealT edge_tolerance = 1.e-8;
  if ( edge_parameter < -edge_tolerance || edge_parameter > 1. + edge_tolerance ) {
    return false;
  }
  edge_parameter = std::max( 0., std::min( 1., edge_parameter ) );
  if ( lor_reference_coordinates != nullptr ) {
    lor_reference_coordinates[0] = edge_parameter;
  }

  basis_values[0] = 1. - edge_parameter;
  basis_values[1] = edge_parameter;

  edge_position[0] = first_endpoint_x + edge_parameter * edge_direction_x;
  edge_position[1] = first_endpoint_y + edge_parameter * edge_direction_y;
  edge_position[2] = 0.;

  if ( interpolated_values != nullptr ) {
    initRealArray( interpolated_values, field_dimension, 0. );
    for ( int node_index = 0; node_index < 2; ++node_index ) {
      for ( int component = 0; component < field_dimension; ++component ) {
        interpolated_values[component] +=
            nodal_values[component + node_index * field_dimension] * basis_values[node_index];
      }
    }
  }

  return true;
}

/**
 * @brief Evaluate native parent-face fields at a projected CommonPlane point.
 *
 * The physical point is first projected to the linear LOR face. Its LOR
 * reference coordinates are mapped through the stored subfacet provenance,
 * after which the native parent basis evaluates position and optional velocity.
 *
 * @param mesh Contact surface mesh
 * @param face_id Tribol LOR face identifier
 * @param lor_face_coordinates Physical coordinates of the LOR face vertices
 * @param physical_point CommonPlane quadrature point
 * @param projection_direction CommonPlane projection direction
 * @param evaluate_velocity Whether velocity values are required
 * @param parent_basis_values Native parent-face basis values
 * @param parent_position Native parent-face position at the mapped point
 * @param parent_velocity Native parent-face velocity at the mapped point
 * @param mapped_parent_reference_coordinates Native parent reference coordinates at the mapped point
 * @return zero when both LOR and native parent mappings succeed; nonzero otherwise
 */
TRIBOL_HOST_DEVICE inline int EvaluateParentFaceAtProjectedPoint(
    const MeshData::Viewer& mesh, IndexT face_id, const RealT* lor_face_coordinates, const RealT* physical_point,
    const RealT* projection_direction, bool evaluate_velocity, RealT* parent_basis_values, RealT* parent_position,
    RealT* parent_velocity, RealT* mapped_parent_reference_coordinates )
{
  RealT lor_basis_values[4] = { 0.0, 0.0, 0.0, 0.0 };
  RealT projected_lor_position[3] = { 0.0, 0.0, 0.0 };
  RealT lor_reference_coordinates[2] = { 0.0, 0.0 };
  const bool mapped_to_lor =
      mesh.spatialDimension() == 2
          ? EvaluateLinearEdgeAtProjectedPoint( lor_face_coordinates, physical_point, projection_direction,
                                                projected_lor_position, lor_basis_values, 0, nullptr, nullptr,
                                                lor_reference_coordinates )
          : EvaluateLinearFaceAtProjectedPoint( lor_face_coordinates, mesh.numberOfNodesPerElement(), physical_point,
                                                projection_direction, projected_lor_position, lor_basis_values, 0,
                                                nullptr, nullptr, lor_reference_coordinates );
  if ( !mapped_to_lor ) {
    return 1;
  }

  RealT parent_reference_coordinates[2] = { 0.0, 0.0 };
  if ( !mesh.mapToParentReference( face_id, lor_reference_coordinates, parent_reference_coordinates ) ) {
    return 2;
  }
  if ( !mesh.evaluateParentFaceBasis( face_id, parent_reference_coordinates, parent_basis_values ) ) {
    return 3;
  }

  mapped_parent_reference_coordinates[0] = parent_reference_coordinates[0];
  mapped_parent_reference_coordinates[1] = parent_reference_coordinates[1];

  mesh.evaluateParentFaceFields( face_id, parent_basis_values, parent_position,
                                 evaluate_velocity ? parent_velocity : nullptr );
  return 0;
}

/**
 * @brief Scatter an equal-and-opposite force through linear contact-face bases.
 *
 * @param first_mesh First contact surface mesh
 * @param second_mesh Second contact surface mesh
 * @param first_face_id First contact-surface face identifier
 * @param second_face_id Second contact-surface face identifier
 * @param spatial_dimension Contact problem dimension
 * @param number_of_nodes_per_face Number of nodes on each linear face
 * @param force_x Integrated x force applied to the second face
 * @param force_y Integrated y force applied to the second face
 * @param force_z Integrated z force applied to the second face
 * @param first_basis_values Basis values on the first face
 * @param second_basis_values Basis values on the second face
 */
TRIBOL_HOST_DEVICE inline void AccumulateContactForce(
    const MeshData::Viewer& first_mesh, const MeshData::Viewer& second_mesh, const IndexT first_face_id,
    const IndexT second_face_id, const int spatial_dimension, const int number_of_nodes_per_face, const RealT force_x,
    const RealT force_y, const RealT force_z, const RealT* first_basis_values, const RealT* second_basis_values )
{
  for ( IndexT basis_index = 0; basis_index < number_of_nodes_per_face; ++basis_index ) {
    const IndexT first_node_id = first_mesh.getGlobalNodeId( first_face_id, basis_index );
    const IndexT second_node_id = second_mesh.getGlobalNodeId( second_face_id, basis_index );

    const RealT first_nodal_force_x = force_x * first_basis_values[basis_index];
    const RealT first_nodal_force_y = force_y * first_basis_values[basis_index];
    const RealT first_nodal_force_z = force_z * first_basis_values[basis_index];

    const RealT second_nodal_force_x = force_x * second_basis_values[basis_index];
    const RealT second_nodal_force_y = force_y * second_basis_values[basis_index];
    const RealT second_nodal_force_z = force_z * second_basis_values[basis_index];

    tribol::atomicAdd( &first_mesh.getResponse()[0][first_node_id], -first_nodal_force_x );
    tribol::atomicAdd( &second_mesh.getResponse()[0][second_node_id], second_nodal_force_x );

    tribol::atomicAdd( &first_mesh.getResponse()[1][first_node_id], -first_nodal_force_y );
    tribol::atomicAdd( &second_mesh.getResponse()[1][second_node_id], second_nodal_force_y );

    if ( spatial_dimension == 3 ) {
      tribol::atomicAdd( &first_mesh.getResponse()[2][first_node_id], -first_nodal_force_z );
      tribol::atomicAdd( &second_mesh.getResponse()[2][second_node_id], second_nodal_force_z );
    }
  }
}

/**
 * @brief Scatter an equal-and-opposite force through native parent-face bases.
 *
 * @param mesh1 First contact surface mesh
 * @param mesh2 Second contact surface mesh
 * @param face_id1 First Tribol face identifier
 * @param face_id2 Second Tribol face identifier
 * @param force Force applied to the second face
 * @param parent_basis_values1 Native basis values on the first face
 * @param parent_basis_values2 Native basis values on the second face
 */
TRIBOL_HOST_DEVICE inline void AccumulateParentContactForce( const MeshData::Viewer& mesh1,
                                                             const MeshData::Viewer& mesh2, IndexT face_id1,
                                                             IndexT face_id2, const RealT* force,
                                                             const RealT* parent_basis_values1,
                                                             const RealT* parent_basis_values2 )
{
  RealT opposite_force[max_dim] = { -force[0], -force[1], -force[2] };
  mesh1.addParentFaceResponse( face_id1, parent_basis_values1, opposite_force );
  mesh2.addParentFaceResponse( face_id2, parent_basis_values2, force );
}

/**
 * @brief Compute the CommonPlane penalty stiffness for one LOR face pair.
 *
 * @param first_mesh First contact surface mesh
 * @param second_mesh Second contact surface mesh
 * @param first_face_id First Tribol face identifier
 * @param second_face_id Second Tribol face identifier
 * @param penalty_options Registered penalty options
 * @param penalty_stiffness Computed equivalent stiffness per unit overlap measure
 * @return true when the registered penalty data are physically admissible
 */
TRIBOL_HOST_DEVICE inline bool ComputePairPenaltyStiffness( const MeshData::Viewer& first_mesh,
                                                            const MeshData::Viewer& second_mesh, IndexT first_face_id,
                                                            IndexT second_face_id,
                                                            const PenaltyEnforcementOptions& penalty_options,
                                                            RealT& penalty_stiffness )
{
  const RealT first_penalty_scale = first_mesh.getElementData().m_penalty_scale;
  const RealT second_penalty_scale = second_mesh.getElementData().m_penalty_scale;

  switch ( penalty_options.kinematic_calculation ) {
    case KINEMATIC_CONSTANT: {
      const RealT first_stiffness = first_penalty_scale * first_mesh.getElementData().m_penalty_stiffness;
      const RealT second_stiffness = second_penalty_scale * second_mesh.getElementData().m_penalty_stiffness;
      penalty_stiffness = ComputePenaltyStiffnessPerArea( first_stiffness, second_stiffness );
      return true;
    }
    case KINEMATIC_ELEMENT: {
      const RealT first_thickness =
          first_mesh.getElementData().m_thickness[first_face_id] + penalty_options.tiny_length;
      const RealT second_thickness =
          second_mesh.getElementData().m_thickness[second_face_id] + penalty_options.tiny_length;
      if ( first_thickness <= 0.0 || second_thickness <= 0.0 ) {
        return false;
      }
      const RealT first_stiffness =
          first_penalty_scale * first_mesh.getElementData().m_mat_mod[first_face_id] / first_thickness;
      const RealT second_stiffness =
          second_penalty_scale * second_mesh.getElementData().m_mat_mod[second_face_id] / second_thickness;
      penalty_stiffness = ComputePenaltyStiffnessPerArea( first_stiffness, second_stiffness );
      return true;
    }
    default:
      penalty_stiffness = 0.0;
      return false;
  }
}

/**
 * @brief Store one fully evaluated CommonPlane quadrature row.
 *
 * The helper maps the physical integration point to each LOR face, maps the
 * resulting reference coordinates to native parent space when available, and
 * stores all field and basis values needed by downstream explicit operators.
 *
 * @param rows Writable CommonPlane row-batch view
 * @param row_id Stable row slot assigned to this integration point
 * @param contact_pair_id Active CommonPlane pair identifier
 * @param first_mesh First contact surface mesh
 * @param second_mesh Second contact surface mesh
 * @param first_face_id First Tribol face identifier
 * @param second_face_id Second Tribol face identifier
 * @param first_face_coordinates First physical LOR face coordinates
 * @param second_face_coordinates Second physical LOR face coordinates
 * @param first_projected_face_coordinates First LOR face projected onto the CommonPlane
 * @param second_projected_face_coordinates Second LOR face projected onto the CommonPlane
 * @param first_face_velocities First LOR face velocities when requested
 * @param second_face_velocities Second LOR face velocities when requested
 * @param integration_point Physical CommonPlane integration point
 * @param normal Consistently oriented CommonPlane unit normal
 * @param integration_weight Physical integration measure for this row
 * @param penalty_stiffness Equivalent normal penalty stiffness per unit measure
 * @param rate_penalty_coefficient Normal rate-penalty coefficient per unit measure
 * @param tangential_viscous_coefficient Tangential viscous coefficient per unit measure
 * @param gap_tolerance Normal contact activation tolerance
 * @param evaluate_velocity Whether velocity fields are required by any consumer
 * @param use_parent_fields Whether fields and responses use native parent faces
 * @return zero when both face mappings and field evaluations succeed; nonzero otherwise
 */
TRIBOL_HOST_DEVICE inline int StoreCommonPlaneContactRow(
    CommonPlaneContactData::Viewer rows, IndexT row_id, IndexT contact_pair_id, const MeshData::Viewer& first_mesh,
    const MeshData::Viewer& second_mesh, IndexT first_face_id, IndexT second_face_id,
    const RealT* first_face_coordinates, const RealT* second_face_coordinates, const RealT* first_face_velocities,
    const RealT* second_face_velocities, const RealT* first_projected_face_coordinates,
    const RealT* second_projected_face_coordinates, const RealT* integration_point, const RealT* normal,
    RealT integration_weight, RealT penalty_stiffness, RealT rate_penalty_coefficient,
    RealT tangential_viscous_coefficient, RealT gap_tolerance, bool evaluate_velocity, bool use_parent_fields )
{
  RealT first_basis_values[max_nodes_per_face] = { 0.0 };
  RealT second_basis_values[max_nodes_per_face] = { 0.0 };
  RealT first_position[max_dim] = { 0.0, 0.0, 0.0 };
  RealT second_position[max_dim] = { 0.0, 0.0, 0.0 };
  RealT first_velocity[max_dim] = { 0.0, 0.0, 0.0 };
  RealT second_velocity[max_dim] = { 0.0, 0.0, 0.0 };
  RealT first_reference_coordinates[ParentFaceData::max_reference_dimension] = { 0.0, 0.0 };
  RealT second_reference_coordinates[ParentFaceData::max_reference_dimension] = { 0.0, 0.0 };

  const int spatial_dimension = rows.spatial_dimension;
  const int first_basis_count = use_parent_fields ? first_mesh.getParentFaceData().m_parent_node_counts[first_face_id]
                                                  : first_mesh.numberOfNodesPerElement();
  const int second_basis_count = use_parent_fields
                                     ? second_mesh.getParentFaceData().m_parent_node_counts[second_face_id]
                                     : second_mesh.numberOfNodesPerElement();

  int first_mapping_status = 0;
  int second_mapping_status = 0;
  if ( use_parent_fields ) {
    first_mapping_status = EvaluateParentFaceAtProjectedPoint(
        first_mesh, first_face_id, first_projected_face_coordinates, integration_point, normal, evaluate_velocity,
        first_basis_values, first_position, first_velocity, first_reference_coordinates );
    second_mapping_status = EvaluateParentFaceAtProjectedPoint(
        second_mesh, second_face_id, second_projected_face_coordinates, integration_point, normal, evaluate_velocity,
        second_basis_values, second_position, second_velocity, second_reference_coordinates );
  } else if ( spatial_dimension == 2 ) {
    first_mapping_status =
        EvaluateLinearEdgeAtProjectedPoint( first_projected_face_coordinates, integration_point, normal, first_position,
                                            first_basis_values, 0, nullptr, nullptr, first_reference_coordinates )
            ? 0
            : 1;
    second_mapping_status = EvaluateLinearEdgeAtProjectedPoint( second_projected_face_coordinates, integration_point,
                                                                normal, second_position, second_basis_values, 0,
                                                                nullptr, nullptr, second_reference_coordinates )
                                ? 0
                                : 1;
  } else {
    first_mapping_status = EvaluateLinearFaceAtProjectedPoint(
                               first_projected_face_coordinates, first_basis_count, integration_point, normal,
                               first_position, first_basis_values, 0, nullptr, nullptr, first_reference_coordinates )
                               ? 0
                               : 1;
    second_mapping_status =
        EvaluateLinearFaceAtProjectedPoint( second_projected_face_coordinates, second_basis_count, integration_point,
                                            normal, second_position, second_basis_values, 0, nullptr, nullptr,
                                            second_reference_coordinates )
            ? 0
            : 1;
  }

  if ( first_mapping_status != 0 || second_mapping_status != 0 ) {
    return first_mapping_status != 0 ? first_mapping_status : 10 + second_mapping_status;
  }

  if ( !use_parent_fields ) {
    EvaluateFaceFieldsFromBasis( first_face_coordinates, first_basis_count, spatial_dimension, first_basis_values,
                                 first_position, evaluate_velocity ? spatial_dimension : 0,
                                 evaluate_velocity ? first_face_velocities : nullptr,
                                 evaluate_velocity ? first_velocity : nullptr );
    EvaluateFaceFieldsFromBasis( second_face_coordinates, second_basis_count, spatial_dimension, second_basis_values,
                                 second_position, evaluate_velocity ? spatial_dimension : 0,
                                 evaluate_velocity ? second_face_velocities : nullptr,
                                 evaluate_velocity ? second_velocity : nullptr );
  }

  rows.row_is_valid[row_id] = 1;
  rows.contact_pair_ids[row_id] = contact_pair_id;
  rows.first_face_ids[row_id] = first_face_id;
  rows.second_face_ids[row_id] = second_face_id;
  rows.first_basis_counts[row_id] = first_basis_count;
  rows.second_basis_counts[row_id] = second_basis_count;
  rows.row_uses_parent_fields[row_id] = use_parent_fields ? 1 : 0;
  rows.integration_weights[row_id] = integration_weight;
  rows.penalty_stiffnesses[row_id] = penalty_stiffness;
  rows.rate_penalty_coefficients[row_id] = rate_penalty_coefficient;
  rows.tangential_viscous_coefficients[row_id] = tangential_viscous_coefficient;

  if ( use_parent_fields ) {
    const ParentFaceData& first_parent_data = first_mesh.getParentFaceData();
    const ParentFaceData& second_parent_data = second_mesh.getParentFaceData();
    rows.first_parent_face_ids[row_id] = first_parent_data.m_parent_face_ids[first_face_id];
    rows.second_parent_face_ids[row_id] = second_parent_data.m_parent_face_ids[second_face_id];
    rows.first_parent_face_owner_ranks[row_id] = first_parent_data.m_parent_face_owner_ranks[first_face_id];
    rows.second_parent_face_owner_ranks[row_id] = second_parent_data.m_parent_face_owner_ranks[second_face_id];
  } else {
    rows.first_parent_face_ids[row_id] = first_face_id;
    rows.second_parent_face_ids[row_id] = second_face_id;
  }

  RealT normal_gap = 0.0;
  RealT normal_velocity_gap = 0.0;
  for ( int component = 0; component < max_dim; ++component ) {
    rows.integration_points( row_id, component ) = component < spatial_dimension ? integration_point[component] : 0.0;
    rows.first_positions( row_id, component ) = first_position[component];
    rows.second_positions( row_id, component ) = second_position[component];
    rows.first_velocities( row_id, component ) = first_velocity[component];
    rows.second_velocities( row_id, component ) = second_velocity[component];
    rows.normals( row_id, component ) = component < spatial_dimension ? normal[component] : 0.0;
    if ( component < spatial_dimension ) {
      normal_gap += ( first_position[component] - second_position[component] ) * normal[component];
      normal_velocity_gap += ( first_velocity[component] - second_velocity[component] ) * normal[component];
    }
  }
  for ( int reference_component = 0; reference_component < ParentFaceData::max_reference_dimension;
        ++reference_component ) {
    rows.first_parent_reference_coordinates( row_id, reference_component ) =
        first_reference_coordinates[reference_component];
    rows.second_parent_reference_coordinates( row_id, reference_component ) =
        second_reference_coordinates[reference_component];
  }
  for ( int basis_index = 0; basis_index < first_basis_count; ++basis_index ) {
    rows.first_basis_values( row_id, basis_index ) = first_basis_values[basis_index];
  }
  for ( int basis_index = 0; basis_index < second_basis_count; ++basis_index ) {
    rows.second_basis_values( row_id, basis_index ) = second_basis_values[basis_index];
  }

  rows.gaps[row_id] = normal_gap;
  rows.normal_velocity_gaps[row_id] = normal_velocity_gap;
  rows.row_is_active[row_id] = normal_gap <= gap_tolerance ? 1 : 0;
  return 0;
}

/**
 * @brief Scatter one vector force from a shared CommonPlane row.
 *
 * @param rows CommonPlane row-batch view
 * @param row_id Row carrying basis and face identifiers
 * @param first_mesh First contact surface mesh
 * @param second_mesh Second contact surface mesh
 * @param force_on_second_face Integrated force applied to the second face
 */
TRIBOL_HOST_DEVICE inline void ScatterCommonPlaneRowForce( const CommonPlaneContactData::Viewer& rows, IndexT row_id,
                                                           const MeshData::Viewer& first_mesh,
                                                           const MeshData::Viewer& second_mesh,
                                                           const RealT* force_on_second_face )
{
  const RealT* first_basis_values = &rows.first_basis_values( row_id, 0 );
  const RealT* second_basis_values = &rows.second_basis_values( row_id, 0 );
  if ( rows.row_uses_parent_fields[row_id] != 0 ) {
    AccumulateParentContactForce( first_mesh, second_mesh, rows.first_face_ids[row_id], rows.second_face_ids[row_id],
                                  force_on_second_face, first_basis_values, second_basis_values );
    return;
  }

  AccumulateContactForce( first_mesh, second_mesh, rows.first_face_ids[row_id], rows.second_face_ids[row_id],
                          rows.spatial_dimension, rows.first_basis_counts[row_id], force_on_second_face[0],
                          force_on_second_face[1], force_on_second_face[2], first_basis_values, second_basis_values );
}

/**
 * @brief Return one inverse mass used by a CommonPlane quadrature row.
 *
 * @param mesh Contact surface containing the field data
 * @param face_id Tribol face identifier stored by the row
 * @param basis_index Local face-basis index
 * @param component Vector component
 * @param uses_parent_fields Whether the row evaluates native parent fields
 * @return Component-wise inverse diagonal mass
 */
TRIBOL_HOST_DEVICE inline RealT GetCommonPlaneRowInverseMass( const MeshData::Viewer& mesh, IndexT face_id,
                                                              int basis_index, int component, bool uses_parent_fields )
{
  if ( uses_parent_fields ) {
    const int field_index = basis_index * mesh.spatialDimension() + component;
    return mesh.getParentFaceData().m_parent_inverse_masses( face_id, field_index );
  }
  return mesh.getInverseMass( mesh.getGlobalNodeId( face_id, basis_index ), component );
}

/**
 * @brief Return the local accumulation index for one CommonPlane row degree of freedom.
 *
 * Native parent rows use the parent-mesh vector degree-of-freedom identifier.
 * Ordinary Tribol meshes use component-interleaved nodal identifiers. Parent
 * identifiers from different source ranks may share an accumulation slot;
 * this only increases the resulting conservative bound.
 *
 * @param mesh Contact surface containing the field data
 * @param face_id Tribol face identifier stored by the row
 * @param basis_index Local face-basis index
 * @param component Vector component
 * @param uses_parent_fields Whether the row evaluates native parent fields
 * @return Nonnegative local accumulation index
 */
TRIBOL_HOST_DEVICE inline IndexT GetCommonPlaneRowDofIndex( const MeshData::Viewer& mesh, IndexT face_id,
                                                            int basis_index, int component, bool uses_parent_fields )
{
  if ( uses_parent_fields ) {
    const int field_index = basis_index * mesh.spatialDimension() + component;
    return mesh.getParentFaceData().m_parent_vector_dof_ids( face_id, field_index );
  }
  return mesh.getGlobalNodeId( face_id, basis_index ) * mesh.spatialDimension() + component;
}

/**
 * @brief Return the number of accumulation slots needed by one contact surface.
 *
 * @param mesh Contact surface whose mass-normalized rows will be accumulated
 * @return Number of local vector degree-of-freedom slots
 */
inline IndexT GetCommonPlaneRowDofCount( const MeshData& mesh )
{
  return mesh.hasParentFaceFields() ? mesh.getParentFaceData().m_parent_vector_dof_count
                                    : mesh.numberOfNodes() * mesh.spatialDimension();
}

}  // namespace

/**
 * @brief Build the shared CommonPlane quadrature row batch for one update.
 *
 * One execution thread owns each accepted overlap cell. Rows are written into
 * a deterministic pair-local range, which keeps generation device-resident and
 * gives every row a stable identity without a host prefix sum.
 *
 * @param cs CommonPlane coupling scheme
 * @return zero on success and nonzero when row generation detects invalid data
 */
int BuildCommonPlaneContactRows( CouplingScheme* cs )
{
  auto* common_plane_data = static_cast<CommonPlaneContactData*>( cs->getMethodData() );
  SLIC_ERROR_ROOT_IF( common_plane_data == nullptr,
                      "BuildCommonPlaneContactRows(): CommonPlane row storage is unavailable." );

  const IndexT number_of_pairs = cs->getNumActivePairs();
  const int spatial_dimension = cs->spatialDimension();
  common_plane_data->resize( number_of_pairs, spatial_dimension, cs->getAllocatorId() );
  if ( number_of_pairs == 0 ) {
    return 0;
  }

  const CommonPlaneContactData::Viewer rows = common_plane_data->getView();
  const CouplingScheme::Viewer coupling_scheme = cs->getView();
  const bool tangential_velocity_is_required = cs->getContactModel() == VISCOUS_TANGENTIAL;
  Array1D<int> evaluation_error_data( { 0 }, cs->getAllocatorId() );
  Array1DView<int> evaluation_error = evaluation_error_data.view();
  forAllExec(
      cs->getExecutionMode(), number_of_pairs,
      [rows, coupling_scheme, spatial_dimension, tangential_velocity_is_required,
       evaluation_error] TRIBOL_HOST_DEVICE( IndexT contact_pair_id ) {
        CommonPlanePair& common_plane = coupling_scheme.getCompGeomView().getCommonPlane( contact_pair_id );
        const MeshData::Viewer& first_mesh = coupling_scheme.getMesh1View();
        const MeshData::Viewer& second_mesh = coupling_scheme.getMesh2View();
        const PenaltyEnforcementOptions& penalty_options = coupling_scheme.getEnforcementOptions().penalty_options;
        const IndexT first_face_id = common_plane.getCpElementId1();
        const IndexT second_face_id = common_plane.getCpElementId2();
        const IndexT first_row_id = rows.pairRowOffset( contact_pair_id );

        rows.pair_row_counts[contact_pair_id] = 0;
        common_plane.m_inContact = false;
        common_plane.m_pressure = 0.0;
        common_plane.m_ratePressure = 0.0;
        common_plane.m_velGap = 0.0;

        const bool first_face_uses_parent_fields = first_mesh.hasParentFaceFields();
        const bool second_face_uses_parent_fields = second_mesh.hasParentFaceFields();
        if ( first_face_uses_parent_fields != second_face_uses_parent_fields ) {
          rows.pair_evaluation_statuses[contact_pair_id] =
              static_cast<int>( CommonPlanePairEvaluationStatus::INVALID_PARENT_DATA );
          tribol::atomicMax( &evaluation_error[0], 1 );
          return;
        }
        const bool use_parent_fields = first_face_uses_parent_fields && second_face_uses_parent_fields;
        const bool normal_rate_velocity_is_required = penalty_options.constraint_type == KINEMATIC_AND_RATE;
        const bool evaluate_velocity = normal_rate_velocity_is_required || tangential_velocity_is_required;
        const bool first_velocity_is_available =
            use_parent_fields ? first_mesh.hasParentFaceVelocity() : first_mesh.hasVelocity();
        const bool second_velocity_is_available =
            use_parent_fields ? second_mesh.hasParentFaceVelocity() : second_mesh.hasVelocity();
        if ( evaluate_velocity && ( !first_velocity_is_available || !second_velocity_is_available ) ) {
          rows.pair_evaluation_statuses[contact_pair_id] =
              static_cast<int>( CommonPlanePairEvaluationStatus::INVALID_PARENT_DATA );
          tribol::atomicMax( &evaluation_error[0], 1 );
          return;
        }

        RealT common_plane_normal[max_dim] = { common_plane.m_nX, common_plane.m_nY,
                                               spatial_dimension == 3 ? common_plane.m_nZ : 0.0 };
        const RealT normal_magnitude =
            spatial_dimension == 3 ? magnitude( common_plane_normal[0], common_plane_normal[1], common_plane_normal[2] )
                                   : magnitude( common_plane_normal[0], common_plane_normal[1] );
        if ( normal_magnitude <= 1.e-12 || std::abs( normal_magnitude - 1.0 ) > 1.e-6 ) {
          rows.pair_evaluation_statuses[contact_pair_id] =
              static_cast<int>( CommonPlanePairEvaluationStatus::INCONSISTENT_NORMAL );
          tribol::atomicMax( &evaluation_error[0], 1 );
          return;
        }

        RealT penalty_stiffness = 0.0;
        if ( !ComputePairPenaltyStiffness( first_mesh, second_mesh, first_face_id, second_face_id, penalty_options,
                                           penalty_stiffness ) ) {
          rows.pair_evaluation_statuses[contact_pair_id] =
              static_cast<int>( CommonPlanePairEvaluationStatus::INVALID_PARENT_DATA );
          tribol::atomicMax( &evaluation_error[0], 1 );
          return;
        }
        const RealT rate_penalty_coefficient =
            normal_rate_velocity_is_required
                ? ComputeRatePenalty( first_mesh, second_mesh, penalty_stiffness, penalty_options.rate_calculation )
                : 0.0;
        const RealT tangential_viscous_coefficient =
            tangential_velocity_is_required ? 0.5 * ( first_mesh.getElementData().m_viscous_damping_coeff +
                                                      second_mesh.getElementData().m_viscous_damping_coeff )
                                            : 0.0;
        const RealT gap_tolerance = coupling_scheme.getGapTol( first_face_id, second_face_id );

        StackArrayT<RealT, max_dim * max_nodes_per_face> first_face_coordinates;
        StackArrayT<RealT, max_dim * max_nodes_per_face> second_face_coordinates;
        StackArrayT<RealT, max_dim * max_nodes_per_face> first_projected_face_coordinates;
        StackArrayT<RealT, max_dim * max_nodes_per_face> second_projected_face_coordinates;
        StackArrayT<RealT, max_dim * max_nodes_per_face> first_face_velocities;
        StackArrayT<RealT, max_dim * max_nodes_per_face> second_face_velocities;
        first_mesh.getFaceCoords( first_face_id, first_face_coordinates );
        second_mesh.getFaceCoords( second_face_id, second_face_coordinates );
        common_plane.getFace1ProjectedCoords( first_projected_face_coordinates, first_mesh.numberOfNodesPerElement() );
        common_plane.getFace2ProjectedCoords( second_projected_face_coordinates,
                                              second_mesh.numberOfNodesPerElement() );
        if ( evaluate_velocity && !use_parent_fields ) {
          first_mesh.getFaceVelocities( first_face_id, first_face_velocities );
          second_mesh.getFaceVelocities( second_face_id, second_face_velocities );
        }

        RealT overlap_vertices[max_dim * CommonPlaneContactData::maximum_overlap_vertices] = { 0.0 };
        common_plane.getOverlapVertices( overlap_vertices );
        const int number_of_overlap_vertices = spatial_dimension == 2 ? 2 : common_plane.m_numPolyVert;
        const bool use_multiple_points = UseMultipleIntegrationPoints( penalty_options, first_mesh, second_mesh );
        const int quadrature_order =
            GetCommonPlaneQuadratureOrder( penalty_options, first_mesh, second_mesh, first_face_id, second_face_id );
        int generated_row_count = 0;
        bool has_active_row = false;
        bool mapping_failed = false;
        RealT integrated_measure = 0.0;

        if ( !use_multiple_points ) {
          const RealT integration_point[max_dim] = { common_plane.m_cX, common_plane.m_cY,
                                                     spatial_dimension == 3 ? common_plane.m_cZ : 0.0 };
          const int row_evaluation_status = StoreCommonPlaneContactRow(
              rows, first_row_id, contact_pair_id, first_mesh, second_mesh, first_face_id, second_face_id,
              first_face_coordinates, second_face_coordinates, first_face_velocities, second_face_velocities,
              first_projected_face_coordinates, second_projected_face_coordinates, integration_point,
              common_plane_normal, common_plane.m_area, penalty_stiffness, rate_penalty_coefficient,
              tangential_viscous_coefficient, gap_tolerance, evaluate_velocity, use_parent_fields );
          mapping_failed = row_evaluation_status != 0;
          if ( row_evaluation_status == 0 ) {
            generated_row_count = 1;
            integrated_measure = common_plane.m_area;
            has_active_row = rows.row_is_active[first_row_id] != 0;
          }
        } else if ( spatial_dimension == 2 ) {
          RealT quadrature_weights[max_segment_gauss_legendre_qpts] = { 0.0 };
          RealT quadrature_coordinates[max_segment_gauss_legendre_qpts] = { 0.0 };
          const int number_of_quadrature_points =
              GetCommonPlaneSegmentRule( quadrature_order, quadrature_weights, quadrature_coordinates );
          const RealT first_endpoint_x = overlap_vertices[0];
          const RealT first_endpoint_y = overlap_vertices[1];
          const RealT second_endpoint_x = overlap_vertices[2];
          const RealT second_endpoint_y = overlap_vertices[3];
          const RealT overlap_length =
              magnitude( second_endpoint_x - first_endpoint_x, second_endpoint_y - first_endpoint_y );

          for ( int quadrature_point = 0; quadrature_point < number_of_quadrature_points; ++quadrature_point ) {
            const RealT segment_coordinate = quadrature_coordinates[quadrature_point];
            const RealT integration_point[max_dim] = {
                ( 1.0 - segment_coordinate ) * first_endpoint_x + segment_coordinate * second_endpoint_x,
                ( 1.0 - segment_coordinate ) * first_endpoint_y + segment_coordinate * second_endpoint_y, 0.0 };
            const RealT integration_weight = overlap_length * quadrature_weights[quadrature_point];
            const IndexT row_id = first_row_id + generated_row_count;
            const int row_evaluation_status = StoreCommonPlaneContactRow(
                rows, row_id, contact_pair_id, first_mesh, second_mesh, first_face_id, second_face_id,
                first_face_coordinates, second_face_coordinates, first_face_velocities, second_face_velocities,
                first_projected_face_coordinates, second_projected_face_coordinates, integration_point,
                common_plane_normal, integration_weight, penalty_stiffness, rate_penalty_coefficient,
                tangential_viscous_coefficient, gap_tolerance, evaluate_velocity, use_parent_fields );
            if ( row_evaluation_status != 0 ) {
              mapping_failed = true;
              break;
            }
            integrated_measure += integration_weight;
            has_active_row = has_active_row || rows.row_is_active[row_id] != 0;
            ++generated_row_count;
          }
        } else {
          RealT quadrature_weights[max_symmetric_triangle_qpts] = { 0.0 };
          RealT quadrature_coordinates[2 * max_symmetric_triangle_qpts] = { 0.0 };
          const int number_of_quadrature_points =
              GetCommonPlaneTriangleRule( quadrature_order, quadrature_weights, quadrature_coordinates );
          const RealT overlap_centroid[max_dim] = { common_plane.m_cX, common_plane.m_cY, common_plane.m_cZ };

          // CommonPlane overlap polygons are convex and consistently ordered.
          // A fan about the polygon centroid therefore creates nonoverlapping
          // integration triangles while preserving the LOR overlap measure.
          for ( int overlap_vertex = 0; overlap_vertex < number_of_overlap_vertices; ++overlap_vertex ) {
            const int next_overlap_vertex = overlap_vertex + 1 == number_of_overlap_vertices ? 0 : overlap_vertex + 1;
            RealT triangle_x[3] = { overlap_vertices[spatial_dimension * overlap_vertex],
                                    overlap_vertices[spatial_dimension * next_overlap_vertex], overlap_centroid[0] };
            RealT triangle_y[3] = { overlap_vertices[spatial_dimension * overlap_vertex + 1],
                                    overlap_vertices[spatial_dimension * next_overlap_vertex + 1],
                                    overlap_centroid[1] };
            RealT triangle_z[3] = { overlap_vertices[spatial_dimension * overlap_vertex + 2],
                                    overlap_vertices[spatial_dimension * next_overlap_vertex + 2],
                                    overlap_centroid[2] };
            const RealT triangle_area = Area3DTri( triangle_x, triangle_y, triangle_z );
            if ( triangle_area <= 0.0 ) {
              continue;
            }

            for ( int quadrature_point = 0; quadrature_point < number_of_quadrature_points; ++quadrature_point ) {
              const RealT first_triangle_coordinate = quadrature_coordinates[2 * quadrature_point];
              const RealT second_triangle_coordinate = quadrature_coordinates[2 * quadrature_point + 1];
              const RealT centroid_coordinate = 1.0 - first_triangle_coordinate - second_triangle_coordinate;
              const RealT integration_point[max_dim] = {
                  centroid_coordinate * triangle_x[0] + first_triangle_coordinate * triangle_x[1] +
                      second_triangle_coordinate * triangle_x[2],
                  centroid_coordinate * triangle_y[0] + first_triangle_coordinate * triangle_y[1] +
                      second_triangle_coordinate * triangle_y[2],
                  centroid_coordinate * triangle_z[0] + first_triangle_coordinate * triangle_z[1] +
                      second_triangle_coordinate * triangle_z[2] };
              const RealT integration_weight = triangle_area * quadrature_weights[quadrature_point];
              const IndexT row_id = first_row_id + generated_row_count;
              const int row_evaluation_status = StoreCommonPlaneContactRow(
                  rows, row_id, contact_pair_id, first_mesh, second_mesh, first_face_id, second_face_id,
                  first_face_coordinates, second_face_coordinates, first_face_velocities, second_face_velocities,
                  first_projected_face_coordinates, second_projected_face_coordinates, integration_point,
                  common_plane_normal, integration_weight, penalty_stiffness, rate_penalty_coefficient,
                  tangential_viscous_coefficient, gap_tolerance, evaluate_velocity, use_parent_fields );
              if ( row_evaluation_status != 0 ) {
                mapping_failed = true;
                break;
              }
              integrated_measure += integration_weight;
              has_active_row = has_active_row || rows.row_is_active[row_id] != 0;
              ++generated_row_count;
            }
            if ( mapping_failed ) {
              break;
            }
          }
        }

        if ( mapping_failed ) {
          for ( int generated_row = 0; generated_row < generated_row_count; ++generated_row ) {
            rows.row_is_valid[first_row_id + generated_row] = 0;
            rows.row_is_active[first_row_id + generated_row] = 0;
          }
          rows.pair_evaluation_statuses[contact_pair_id] =
              static_cast<int>( CommonPlanePairEvaluationStatus::INVALID_PARENT_MAPPING );
          tribol::atomicMax( &evaluation_error[0], 1 );
          return;
        }
        if ( integrated_measure <= 0.0 || generated_row_count == 0 ) {
          rows.pair_evaluation_statuses[contact_pair_id] =
              static_cast<int>( CommonPlanePairEvaluationStatus::DEGENERATE_OVERLAP );
          tribol::atomicMax( &evaluation_error[0], 1 );
          return;
        }

        rows.pair_row_counts[contact_pair_id] = generated_row_count;
        rows.pair_evaluation_statuses[contact_pair_id] = static_cast<int>( CommonPlanePairEvaluationStatus::VALID );
        common_plane.m_inContact = has_active_row;
      } );

  Array1D<int, MemorySpace::Host> evaluation_error_host( evaluation_error_data );
  if ( evaluation_error_host[0] != 0 ) {
    Array1D<int, MemorySpace::Host> pair_statuses_host( common_plane_data->getPairEvaluationStatuses() );
    for ( IndexT contact_pair_id = 0; contact_pair_id < number_of_pairs; ++contact_pair_id ) {
      if ( pair_statuses_host[contact_pair_id] != static_cast<int>( CommonPlanePairEvaluationStatus::VALID ) ) {
        SLIC_DEBUG( "BuildCommonPlaneContactRows(): row generation failed for active pair "
                    << contact_pair_id << " with status " << pair_statuses_host[contact_pair_id] << "." );
      }
    }
  }
  return evaluation_error_host[0];
}

//------------------------------------------------------------------------------
int ComputeCommonPlanePenaltyStabilityTimeStep( CouplingScheme* cs, RealT& timestep )
{
  auto* common_plane_data = static_cast<CommonPlaneContactData*>( cs->getMethodData() );
  SLIC_ERROR_ROOT_IF( common_plane_data == nullptr,
                      "ComputeCommonPlanePenaltyStabilityTimeStep(): CommonPlane row storage is unavailable." );

  const CommonPlaneContactData::Viewer rows = common_plane_data->getView();
  Array1D<IndexT> active_row_count_data( { 0 }, cs->getAllocatorId() );
  Array1DView<IndexT> active_row_count = active_row_count_data.view();
  forAllExec( cs->getExecutionMode(), rows.row_capacity, [rows, active_row_count] TRIBOL_HOST_DEVICE( IndexT row_id ) {
    if ( rows.row_is_valid[row_id] != 0 && rows.row_is_active[row_id] != 0 ) {
      tribol::atomicInc( &active_row_count[0] );
    }
  } );

  Array1D<IndexT, MemorySpace::Host> active_row_count_host( active_row_count_data );
  IndexT global_active_row_count = active_row_count_host[0];
#ifdef TRIBOL_USE_MPI
  int mpi_initialized = 0;
  MPI_Initialized( &mpi_initialized );
  if ( mpi_initialized ) {
    MPI_Allreduce( MPI_IN_PLACE, &global_active_row_count, 1, MPI_INT, MPI_SUM, cs->getProblemComm() );
  }
#endif

  if ( global_active_row_count == 0 ) {
    cs->setExplicitPenaltyStabilityData( std::numeric_limits<RealT>::infinity(), 0.0, 0.0 );
    return 0;
  }

  const PenaltyEnforcementOptions& penalty_options = cs->getEnforcementOptions().penalty_options;
  const MeshData& first_mesh_data = cs->getMesh1();
  const MeshData& second_mesh_data = cs->getMesh2();
  const bool first_mass_is_available = first_mesh_data.hasParentFaceFields()
                                           ? first_mesh_data.getParentFaceData().hasParentInverseMass()
                                           : first_mesh_data.hasInverseMass();
  const bool second_mass_is_available = second_mesh_data.hasParentFaceFields()
                                            ? second_mesh_data.getParentFaceData().hasParentInverseMass()
                                            : second_mesh_data.hasInverseMass();
  int invalid_configuration =
      ( active_row_count_host[0] > 0 && ( !first_mass_is_available || !second_mass_is_available ) ) ||
              !penalty_options.explicit_integrator_stability_factor_set
          ? 1
          : 0;
#ifdef TRIBOL_USE_MPI
  if ( mpi_initialized ) {
    MPI_Allreduce( MPI_IN_PLACE, &invalid_configuration, 1, MPI_INT, MPI_MAX, cs->getProblemComm() );
  }
#endif
  if ( invalid_configuration != 0 ) {
    SLIC_WARNING_ROOT(
        "ComputeCommonPlanePenaltyStabilityTimeStep(): active explicit penalty contact requires component-wise "
        "inverse diagonal mass and an explicit-integrator stability factor." );
    cs->setExplicitPenaltyStabilityData( -1.0, 0.0, 0.0 );
    timestep = -1.0;
    return 1;
  }

  const IndexT first_dof_count = GetCommonPlaneRowDofCount( first_mesh_data );
  const IndexT second_dof_count = GetCommonPlaneRowDofCount( second_mesh_data );
  const bool rows_use_parent_fields = first_mesh_data.hasParentFaceFields();
  const bool meshes_share_dof_numbering = rows_use_parent_fields || cs->getMeshId1() == cs->getMeshId2();
  const IndexT second_dof_offset = meshes_share_dof_numbering ? 0 : first_dof_count;
  const IndexT row_sum_count = axom::utilities::max(
      static_cast<IndexT>( 1 ), meshes_share_dof_numbering ? axom::utilities::max( first_dof_count, second_dof_count )
                                                           : first_dof_count + second_dof_count );
  Array1D<RealT> stiffness_row_sums_data( row_sum_count, row_sum_count, cs->getAllocatorId() );
  Array1D<RealT> damping_row_sums_data( row_sum_count, row_sum_count, cs->getAllocatorId() );
  stiffness_row_sums_data.fill( 0.0 );
  damping_row_sums_data.fill( 0.0 );
  Array1DView<RealT> stiffness_row_sums = stiffness_row_sums_data.view();
  Array1DView<RealT> damping_row_sums = damping_row_sums_data.view();
  Array1D<int> invalid_mass_data( { 0 }, cs->getAllocatorId() );
  Array1DView<int> invalid_mass = invalid_mass_data.view();

  const CouplingScheme::Viewer coupling_scheme = cs->getView();
  const PenaltyConstraintType constraint_type = penalty_options.constraint_type;
  const bool include_tangential_damping = cs->getContactModel() == VISCOUS_TANGENTIAL;
  forAllExec(
      cs->getExecutionMode(), rows.row_capacity,
      [rows, coupling_scheme, constraint_type, include_tangential_damping, second_dof_offset, row_sum_count,
       stiffness_row_sums, damping_row_sums, invalid_mass] TRIBOL_HOST_DEVICE( IndexT row_id ) {
        if ( rows.row_is_valid[row_id] == 0 || rows.row_is_active[row_id] == 0 ) {
          return;
        }

        const MeshData::Viewer& first_mesh = coupling_scheme.getMesh1View();
        const MeshData::Viewer& second_mesh = coupling_scheme.getMesh2View();
        const bool uses_parent_fields = rows.row_uses_parent_fields[row_id] != 0;
        const IndexT first_face_id = rows.first_face_ids[row_id];
        const IndexT second_face_id = rows.second_face_ids[row_id];
        const int first_basis_count = rows.first_basis_counts[row_id];
        const int second_basis_count = rows.second_basis_counts[row_id];

        StackArrayT<RealT, 2 * max_nodes_per_face * max_dim> square_root_inverse_masses;
        RealT normal_constraint_absolute_sum = 0.0;
        RealT tangential_constraint_absolute_sums[max_dim] = { 0.0, 0.0, 0.0 };

        // First form the absolute row sums for the rank-one normal operator and
        // the tangential projection operator. Signed higher-order basis values
        // are retained until the absolute-value bound is formed here.
        for ( int surface_index = 0; surface_index < 2; ++surface_index ) {
          const MeshData::Viewer& mesh = surface_index == 0 ? first_mesh : second_mesh;
          const IndexT face_id = surface_index == 0 ? first_face_id : second_face_id;
          const int basis_count = surface_index == 0 ? first_basis_count : second_basis_count;
          const Array2DView<RealT>& basis_values =
              surface_index == 0 ? rows.first_basis_values : rows.second_basis_values;
          const int surface_offset = surface_index * max_nodes_per_face * max_dim;
          for ( int basis_index = 0; basis_index < basis_count; ++basis_index ) {
            const RealT absolute_basis_value = std::abs( basis_values( row_id, basis_index ) );
            for ( int component = 0; component < rows.spatial_dimension; ++component ) {
              const RealT inverse_mass =
                  GetCommonPlaneRowInverseMass( mesh, face_id, basis_index, component, uses_parent_fields );
              if ( inverse_mass < 0.0 || inverse_mass != inverse_mass ||
                   inverse_mass > std::numeric_limits<RealT>::max() ) {
                tribol::atomicMax( &invalid_mass[0], 1 );
                return;
              }
              const RealT square_root_inverse_mass = std::sqrt( inverse_mass );
              square_root_inverse_masses[surface_offset + basis_index * max_dim + component] = square_root_inverse_mass;
              normal_constraint_absolute_sum +=
                  absolute_basis_value * std::abs( rows.normals( row_id, component ) ) * square_root_inverse_mass;

              if ( include_tangential_damping && rows.tangential_viscous_coefficients[row_id] > 0.0 ) {
                for ( int target_component = 0; target_component < rows.spatial_dimension; ++target_component ) {
                  const RealT tangential_projection =
                      ( target_component == component ? 1.0 : 0.0 ) -
                      rows.normals( row_id, target_component ) * rows.normals( row_id, component );
                  tangential_constraint_absolute_sums[target_component] +=
                      absolute_basis_value * square_root_inverse_mass * std::abs( tangential_projection );
                }
              }
            }
          }
        }

        const RealT stiffness_scale = rows.integration_weights[row_id] * rows.penalty_stiffnesses[row_id];
        const RealT normal_damping_scale =
            constraint_type == KINEMATIC_AND_RATE && rows.normal_velocity_gaps[row_id] <= 0.0
                ? rows.integration_weights[row_id] * rows.rate_penalty_coefficients[row_id]
                : 0.0;
        const RealT tangential_damping_scale =
            include_tangential_damping ? rows.integration_weights[row_id] * rows.tangential_viscous_coefficients[row_id]
                                       : 0.0;

        // Accumulate all contact rows that touch a degree of freedom before
        // taking the local maximum. Reusing source-local identifiers across
        // different MPI owners can only overestimate this absolute row sum;
        // summing rank-local maxima below remains conservative for shared DOFs.
        for ( int surface_index = 0; surface_index < 2; ++surface_index ) {
          const MeshData::Viewer& mesh = surface_index == 0 ? first_mesh : second_mesh;
          const IndexT face_id = surface_index == 0 ? first_face_id : second_face_id;
          const int basis_count = surface_index == 0 ? first_basis_count : second_basis_count;
          const Array2DView<RealT>& basis_values =
              surface_index == 0 ? rows.first_basis_values : rows.second_basis_values;
          const int surface_offset = surface_index * max_nodes_per_face * max_dim;
          for ( int basis_index = 0; basis_index < basis_count; ++basis_index ) {
            const RealT absolute_basis_value = std::abs( basis_values( row_id, basis_index ) );
            for ( int component = 0; component < rows.spatial_dimension; ++component ) {
              const RealT square_root_inverse_mass =
                  square_root_inverse_masses[surface_offset + basis_index * max_dim + component];
              if ( square_root_inverse_mass == 0.0 ) {
                continue;
              }
              const IndexT degree_of_freedom =
                  GetCommonPlaneRowDofIndex( mesh, face_id, basis_index, component, uses_parent_fields ) +
                  ( surface_index == 1 ? second_dof_offset : 0 );
              if ( degree_of_freedom < 0 || degree_of_freedom >= row_sum_count ) {
                tribol::atomicMax( &invalid_mass[0], 1 );
                return;
              }

              const RealT absolute_normal_coefficient =
                  absolute_basis_value * std::abs( rows.normals( row_id, component ) ) * square_root_inverse_mass;
              tribol::atomicAdd( &stiffness_row_sums[degree_of_freedom],
                                 stiffness_scale * absolute_normal_coefficient * normal_constraint_absolute_sum );
              tribol::atomicAdd( &damping_row_sums[degree_of_freedom],
                                 normal_damping_scale * absolute_normal_coefficient * normal_constraint_absolute_sum +
                                     tangential_damping_scale * absolute_basis_value * square_root_inverse_mass *
                                         tangential_constraint_absolute_sums[component] );
            }
          }
        }
      } );

  Array1D<RealT> local_bounds_data( { 0.0, 0.0 }, cs->getAllocatorId() );
  Array1DView<RealT> local_bounds = local_bounds_data.view();
  forAllExec( cs->getExecutionMode(), row_sum_count,
              [stiffness_row_sums, damping_row_sums, local_bounds] TRIBOL_HOST_DEVICE( IndexT degree_of_freedom ) {
                tribol::atomicMax( &local_bounds[0], stiffness_row_sums[degree_of_freedom] );
                tribol::atomicMax( &local_bounds[1], damping_row_sums[degree_of_freedom] );
              } );

  Array1D<int, MemorySpace::Host> invalid_mass_host( invalid_mass_data );
  ArrayT<RealT, 1, MemorySpace::Host> bounds_host( local_bounds_data );
#ifdef TRIBOL_USE_MPI
  if ( mpi_initialized ) {
    MPI_Allreduce( MPI_IN_PLACE, invalid_mass_host.data(), 1, MPI_INT, MPI_MAX, cs->getProblemComm() );
    MPI_Allreduce( MPI_IN_PLACE, bounds_host.data(), 2, MPI_DOUBLE, MPI_SUM, cs->getProblemComm() );
  }
#endif
  if ( invalid_mass_host[0] != 0 ) {
    SLIC_WARNING_ROOT( "ComputeCommonPlanePenaltyStabilityTimeStep(): inverse diagonal mass data are invalid." );
    cs->setExplicitPenaltyStabilityData( -1.0, 0.0, 0.0 );
    timestep = -1.0;
    return 1;
  }

  const RealT stiffness_bound = bounds_host[0];
  const RealT damping_bound = bounds_host[1];
  const RealT damped_frequency_bound =
      std::sqrt( stiffness_bound + 0.25 * damping_bound * damping_bound ) + 0.5 * damping_bound;
  const RealT stability_timestep = damped_frequency_bound > 0.0
                                       ? penalty_options.explicit_integrator_stability_factor / damped_frequency_bound
                                       : std::numeric_limits<RealT>::infinity();
  cs->setExplicitPenaltyStabilityData( stability_timestep, stiffness_bound, damping_bound );
  timestep = axom::utilities::min( timestep, stability_timestep );
  return 0;
}

//------------------------------------------------------------------------------
template <>
int ApplyNormal<COMMON_PLANE, PENALTY>( CouplingScheme* cs )
{
  auto* common_plane_data = static_cast<CommonPlaneContactData*>( cs->getMethodData() );
  SLIC_ERROR_ROOT_IF( common_plane_data == nullptr,
                      "ApplyNormal<COMMON_PLANE, PENALTY>(): CommonPlane row storage is unavailable." );

  const int row_build_error = BuildCommonPlaneContactRows( cs );
  if ( row_build_error != 0 ) {
    return row_build_error;
  }

  const CommonPlaneContactData::Viewer rows = common_plane_data->getView();
  const CouplingScheme::Viewer coupling_scheme = cs->getView();
  const RealT residual_gap = coupling_scheme.getParameters().residual_gap;
  const PenaltyConstraintType constraint_type = coupling_scheme.getEnforcementOptions().penalty_options.constraint_type;

  // These per-pair reductions preserve the existing contact-plane diagnostics,
  // but derive them from the same pointwise values used by the force scatter.
  Array1D<RealT> integrated_kinematic_pressure( rows.number_of_pairs, rows.number_of_pairs, cs->getAllocatorId() );
  Array1D<RealT> integrated_rate_pressure( rows.number_of_pairs, rows.number_of_pairs, cs->getAllocatorId() );
  Array1D<RealT> integrated_normal_velocity( rows.number_of_pairs, rows.number_of_pairs, cs->getAllocatorId() );
  Array1D<RealT> active_measure( rows.number_of_pairs, rows.number_of_pairs, cs->getAllocatorId() );
  integrated_kinematic_pressure.fill( 0.0 );
  integrated_rate_pressure.fill( 0.0 );
  integrated_normal_velocity.fill( 0.0 );
  active_measure.fill( 0.0 );

  const Array1DView<RealT> integrated_kinematic_pressure_view = integrated_kinematic_pressure.view();
  const Array1DView<RealT> integrated_rate_pressure_view = integrated_rate_pressure.view();
  const Array1DView<RealT> integrated_normal_velocity_view = integrated_normal_velocity.view();
  const Array1DView<RealT> active_measure_view = active_measure.view();

  forAllExec( cs->getExecutionMode(), rows.row_capacity,
              [rows, coupling_scheme, residual_gap, constraint_type, integrated_kinematic_pressure_view,
               integrated_rate_pressure_view, integrated_normal_velocity_view,
               active_measure_view] TRIBOL_HOST_DEVICE( IndexT row_id ) {
                if ( rows.row_is_valid[row_id] == 0 || rows.row_is_active[row_id] == 0 ) {
                  return;
                }

                const RealT kinematic_pressure =
                    ( rows.gaps[row_id] - residual_gap ) * rows.penalty_stiffnesses[row_id];
                RealT rate_pressure = 0.0;
                if ( constraint_type == KINEMATIC_AND_RATE && rows.normal_velocity_gaps[row_id] <= 0.0 ) {
                  rate_pressure = rows.normal_velocity_gaps[row_id] * rows.rate_penalty_coefficients[row_id];
                }
                const RealT applied_pressure = kinematic_pressure + rate_pressure;
                const RealT weighted_pressure = rows.integration_weights[row_id] * applied_pressure;
                RealT force_on_second_face[max_dim] = { 0.0, 0.0, 0.0 };
                for ( int component = 0; component < rows.spatial_dimension; ++component ) {
                  force_on_second_face[component] = rows.normals( row_id, component ) * weighted_pressure;
                }

                ScatterCommonPlaneRowForce( rows, row_id, coupling_scheme.getMesh1View(),
                                            coupling_scheme.getMesh2View(), force_on_second_face );

                const IndexT contact_pair_id = rows.contact_pair_ids[row_id];
                tribol::atomicAdd( &integrated_kinematic_pressure_view[contact_pair_id],
                                   rows.integration_weights[row_id] * kinematic_pressure );
                tribol::atomicAdd( &integrated_rate_pressure_view[contact_pair_id],
                                   rows.integration_weights[row_id] * rate_pressure );
                tribol::atomicAdd( &integrated_normal_velocity_view[contact_pair_id],
                                   rows.integration_weights[row_id] * rows.normal_velocity_gaps[row_id] );
                tribol::atomicAdd( &active_measure_view[contact_pair_id], rows.integration_weights[row_id] );
              } );

  forAllExec( cs->getExecutionMode(), rows.number_of_pairs,
              [coupling_scheme, integrated_kinematic_pressure_view, integrated_rate_pressure_view,
               integrated_normal_velocity_view, active_measure_view] TRIBOL_HOST_DEVICE( IndexT contact_pair_id ) {
                CommonPlanePair& common_plane = coupling_scheme.getCompGeomView().getCommonPlane( contact_pair_id );
                const RealT measure = active_measure_view[contact_pair_id];
                if ( measure > 0.0 ) {
                  common_plane.m_pressure = integrated_kinematic_pressure_view[contact_pair_id] / measure;
                  common_plane.m_ratePressure = integrated_rate_pressure_view[contact_pair_id] / measure;
                  common_plane.m_velGap = integrated_normal_velocity_view[contact_pair_id] / measure;
                } else {
                  common_plane.m_pressure = 0.0;
                  common_plane.m_ratePressure = 0.0;
                  common_plane.m_velGap = 0.0;
                }
              } );

  return 0;
}

//------------------------------------------------------------------------------
template <>
int ApplyTangential<COMMON_PLANE, PENALTY, VISCOUS_TANGENTIAL>( CouplingScheme* cs )
{
  auto* common_plane_data = static_cast<CommonPlaneContactData*>( cs->getMethodData() );
  SLIC_ERROR_ROOT_IF( common_plane_data == nullptr,
                      "ApplyTangential<COMMON_PLANE, PENALTY, VISCOUS_TANGENTIAL>(): CommonPlane row storage is "
                      "unavailable." );

  const CommonPlaneContactData::Viewer rows = common_plane_data->getView();
  const CouplingScheme::Viewer coupling_scheme = cs->getView();
  forAllExec( cs->getExecutionMode(), rows.row_capacity, [rows, coupling_scheme] TRIBOL_HOST_DEVICE( IndexT row_id ) {
    if ( rows.row_is_valid[row_id] == 0 || rows.row_is_active[row_id] == 0 ) {
      return;
    }

    RealT tangential_velocity[max_dim] = { 0.0, 0.0, 0.0 };
    for ( int component = 0; component < rows.spatial_dimension; ++component ) {
      const RealT relative_velocity =
          rows.first_velocities( row_id, component ) - rows.second_velocities( row_id, component );
      tangential_velocity[component] =
          relative_velocity - rows.normal_velocity_gaps[row_id] * rows.normals( row_id, component );
    }

    const RealT force_scale = rows.integration_weights[row_id] * rows.tangential_viscous_coefficients[row_id];
    RealT force_on_second_face[max_dim] = { force_scale * tangential_velocity[0], force_scale * tangential_velocity[1],
                                            force_scale * tangential_velocity[2] };
    ScatterCommonPlaneRowForce( rows, row_id, coupling_scheme.getMesh1View(), coupling_scheme.getMesh2View(),
                                force_on_second_face );
  } );

  return 0;
}
//------------------------------------------------------------------------------

}  // namespace tribol
