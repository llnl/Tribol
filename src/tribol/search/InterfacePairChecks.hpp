// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_INTERFACEPAIRCHECKS_HPP_
#define SRC_TRIBOL_SEARCH_INTERFACEPAIRCHECKS_HPP_

#include <cmath>

#include "tribol/config.hpp"

#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

/** @brief Normalized projection bounds; invalid edges return zero bounds and valid == false */
template <typename Value>
struct EdgeProjection {
  /** @brief Increasing bounds in the integration edge's normalized coordinate; zero when invalid */
  MdArray<Value, 2> bounds{ Value{}, Value{} };
  /** @brief Whether both edges are nondegenerate; overlap with the integration edge is checked separately */
  bool valid{ false };
};

/**
 * @brief Provides allocation-free checks that contact formulations can compose when filtering coarse candidates
 *
 * Each method states one geometric or topological condition instead of imposing one filter policy on every contact
 * method. The methods use only caller-owned mesh data and fixed-size local values, so they can run inside host or
 * device candidate loops.
 */
struct InterfacePairChecks {
 public:
  /** @brief Return whether both IDs identify the same element on the same mesh */
  TRIBOL_HOST_DEVICE static bool isSameElementPair( const MeshData::Viewer& mesh1, IndexT element_id1,
                                                    const MeshData::Viewer& mesh2, IndexT element_id2 )
  {
    return mesh1.meshId() == mesh2.meshId() && element_id1 == element_id2;
  }

  /** @brief Return whether two elements contain any common finite-element node */
  TRIBOL_HOST_DEVICE static bool elementsShareNode( const MeshData::Viewer& mesh1, IndexT element_id1,
                                                    const MeshData::Viewer& mesh2, IndexT element_id2 )
  {
    for ( IndexT local_node1 = 0; local_node1 < mesh1.numberOfNodesPerElement(); ++local_node1 ) {
      const IndexT node1 = mesh1.getGlobalNodeId( element_id1, local_node1 );
      for ( IndexT local_node2 = 0; local_node2 < mesh2.numberOfNodesPerElement(); ++local_node2 ) {
        if ( node1 == mesh2.getGlobalNodeId( element_id2, local_node2 ) ) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * @brief Return whether two linear-edge normals point into opposing half-planes
   *
   * Normals are computed directly from endpoint coordinates so formulations do not require MeshData::computeFaceData()
   * before filtering candidates. Degenerate edges do not define an orientation and return false.
   */
  TRIBOL_HOST_DEVICE static bool linearEdgeNormalsAreOpposing( const MeshData::Viewer& mesh1, IndexT element_id1,
                                                               const MeshData::Viewer& mesh2, IndexT element_id2 )
  {
    RealT edge1_point0[2];
    RealT edge1_point1[2];
    RealT edge2_point0[2];
    RealT edge2_point1[2];
    getLinearEdgeEndpoints( mesh1, element_id1, edge1_point0, edge1_point1 );
    getLinearEdgeEndpoints( mesh2, element_id2, edge2_point0, edge2_point1 );

    RealT edge1_normal[2];
    RealT edge2_normal[2];
    if ( !computeLinearEdgeNormal( edge1_point0, edge1_point1, edge1_normal ) ||
         !computeLinearEdgeNormal( edge2_point0, edge2_point1, edge2_normal ) ) {
      return false;
    }
    return edge1_normal[0] * edge2_normal[0] + edge1_normal[1] * edge2_normal[1] < 0.0;
  }

  /**
   * @brief Project a linear edge onto an integration edge and return normalized bounds on the integration edge
   *
   * The projected edge normal defines the projection direction. The returned interval uses the integration edge's
   * normalized `[-1, 1]` coordinate and may extend outside the finite integration edge. When a projection direction is
   * nearly parallel to the integration edge, each projected endpoint falls back to its original position, matching the
   * existing EnergyMortar construction.
   *
   * @param integration_point0 First endpoint of the integration edge
   * @param integration_point1 Second endpoint of the integration edge
   * @param projected_point0 First endpoint of the projected edge
   * @param projected_point1 Second endpoint of the projected edge
   * @tparam Value Coordinate value type; EnergyMortar uses double even when Tribol configures RealT differently
   * @return Increasing bounds with valid == true for nondegenerate edges; otherwise zero bounds and valid == false
   */
  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static EdgeProjection<Value> computeLinearEdgeProjectionBounds(
      const Value* integration_point0, const Value* integration_point1, const Value* projected_point0,
      const Value* projected_point1 )
  {
    Value projected_normal[2];
    if ( !computeLinearEdgeNormal( projected_point0, projected_point1, projected_normal ) ) {
      return {};
    }

    const Value integration_x = integration_point1[0] - integration_point0[0];
    const Value integration_y = integration_point1[1] - integration_point0[1];
    const Value integration_length_squared = integration_x * integration_x + integration_y * integration_y;
    const Value minimum_edge_length = static_cast<Value>( minimum_edge_length_ );
    if ( integration_length_squared <= minimum_edge_length * minimum_edge_length ) {
      return {};
    }

    Value projected_intersection0[2];
    findLineIntersection( integration_point0, integration_point1, projected_point0, projected_normal,
                          projected_intersection0 );
    const Value integration_fraction0 = ( ( projected_intersection0[0] - integration_point0[0] ) * integration_x +
                                          ( projected_intersection0[1] - integration_point0[1] ) * integration_y ) /
                                        integration_length_squared;

    Value projected_intersection1[2];
    findLineIntersection( integration_point0, integration_point1, projected_point1, projected_normal,
                          projected_intersection1 );
    const Value integration_fraction1 = ( ( projected_intersection1[0] - integration_point0[0] ) * integration_x +
                                          ( projected_intersection1[1] - integration_point0[1] ) * integration_y ) /
                                        integration_length_squared;

    const Value coordinate0 = 2.0 * integration_fraction0 - 1.0;
    const Value coordinate1 = 2.0 * integration_fraction1 - 1.0;
    return { MdArray<Value, 2>{ coordinate0 < coordinate1 ? coordinate0 : coordinate1,
                                coordinate0 < coordinate1 ? coordinate1 : coordinate0 },
             true };
  }

  /**
   * @brief Project one mesh edge onto another without requiring callers to gather endpoint coordinates
   *
   * @param integration_mesh Mesh containing the edge whose normalized coordinate receives the projected interval
   * @param integration_element Element ID of the integration edge
   * @param projected_mesh Mesh containing the edge projected along its own normal
   * @param projected_element Element ID of the projected edge
   * @return Increasing bounds with valid == true for nondegenerate edges; otherwise zero bounds and valid == false
   */
  TRIBOL_HOST_DEVICE static EdgeProjection<RealT> computeLinearEdgeProjectionBounds(
      const MeshData::Viewer& integration_mesh, IndexT integration_element, const MeshData::Viewer& projected_mesh,
      IndexT projected_element )
  {
    RealT integration_point0[2];
    RealT integration_point1[2];
    RealT projected_point0[2];
    RealT projected_point1[2];
    getLinearEdgeEndpoints( integration_mesh, integration_element, integration_point0, integration_point1 );
    getLinearEdgeEndpoints( projected_mesh, projected_element, projected_point0, projected_point1 );
    return computeLinearEdgeProjectionBounds( integration_point0, integration_point1, projected_point0,
                                              projected_point1 );
  }

  /**
   * @brief Return whether a projected interval has positive measure inside an edge's smoothed support
   *
   * EnergyMortar expresses smoothing as an edge-length fraction. Normalized `[-1, 1]` coordinates double that fraction,
   * so the supported interval is `[-1 - 2 * smoothing_fraction, 1 + 2 * smoothing_fraction]`. Merely touching a support
   * endpoint produces no integration measure and returns false.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static bool projectionIntervalOverlapsSmoothedSupport(
      MdSpan<const RealT, 2> projection_bounds, RealT smoothing_fraction )
  {
    const RealT support_extension = 2.0 * smoothing_fraction;
    return projection_bounds[1] > -1.0 - support_extension && projection_bounds[0] < 1.0 + support_extension;
  }

 private:
  InterfacePairChecks() = delete;

  inline static constexpr RealT minimum_edge_length_ = 1.0e-14;
  inline static constexpr RealT minimum_projection_determinant_ = 1.0e-12;

  TRIBOL_HOST_DEVICE static void getLinearEdgeEndpoints( const MeshData::Viewer& mesh, IndexT element_id, RealT* point0,
                                                         RealT* point1 )
  {
    const IndexT node0 = mesh.getGlobalNodeId( element_id, 0 );
    const IndexT node1 = mesh.getGlobalNodeId( element_id, 1 );
    point0[0] = mesh.getPosition()[0][node0];
    point0[1] = mesh.getPosition()[1][node0];
    point1[0] = mesh.getPosition()[0][node1];
    point1[1] = mesh.getPosition()[1][node1];
  }

  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static bool computeLinearEdgeNormal( const Value* point0, const Value* point1,
                                                                               Value* normal )
  {
    const Value edge_x = point1[0] - point0[0];
    const Value edge_y = point1[1] - point0[1];
    const Value edge_length = std::sqrt( edge_x * edge_x + edge_y * edge_y );
    if ( edge_length < static_cast<Value>( minimum_edge_length_ ) ) {
      normal[0] = Value{};
      normal[1] = Value{};
      return false;
    }
    normal[0] = edge_y / edge_length;
    normal[1] = -edge_x / edge_length;
    return true;
  }

  template <typename Value>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void findLineIntersection( const Value* line_point0,
                                                                            const Value* line_point1,
                                                                            const Value* projected_point,
                                                                            const Value* projection_direction,
                                                                            Value* intersection )
  {
    const Value line_x = line_point1[0] - line_point0[0];
    const Value line_y = line_point1[1] - line_point0[1];
    const Value point_x = projected_point[0] - line_point0[0];
    const Value point_y = projected_point[1] - line_point0[1];
    const Value determinant = line_x * projection_direction[1] - line_y * projection_direction[0];
    if ( std::abs( determinant ) < static_cast<Value>( minimum_projection_determinant_ ) ) {
      intersection[0] = projected_point[0];
      intersection[1] = projected_point[1];
      return;
    }

    const Value line_fraction = ( point_x * projection_direction[1] - point_y * projection_direction[0] ) / determinant;
    intersection[0] = line_point0[0] + line_fraction * line_x;
    intersection[1] = line_point0[1] + line_fraction * line_y;
  }
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_INTERFACEPAIRCHECKS_HPP_ */
