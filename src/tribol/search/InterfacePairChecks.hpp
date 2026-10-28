// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_INTERFACEPAIRCHECKS_HPP_
#define SRC_TRIBOL_SEARCH_INTERFACEPAIRCHECKS_HPP_

#include <cmath>

#include "tribol/config.hpp"

#include "tribol/mesh/MeshData.hpp"

namespace tribol {

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

 private:
  /** @brief Use the static checks without constructing an instance. */
  InterfacePairChecks() = delete;

  inline static constexpr RealT minimum_edge_length_ = 1.0e-14;  ///< Edge-length degeneracy threshold.

  /**
   * @brief Read the first two finite-element nodes as straight-edge endpoints.
   * @param mesh Mesh view containing the edge.
   * @param element_id Element whose endpoint coordinates are read.
   * @param[out] point0 First endpoint, with two physical components.
   * @param[out] point1 Second endpoint, with two physical components.
   */
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

  /**
   * @brief Calculate the unit normal (dy, -dx) from ordered endpoints.
   * @param point0 First endpoint, with two physical components.
   * @param point1 Second endpoint, with two physical components.
   * @param[out] normal Unit normal, or zero for a degenerate edge.
   * @return Whether the edge meets the minimum physical length.
   */
  TRIBOL_HOST_DEVICE static bool computeLinearEdgeNormal( const RealT* point0, const RealT* point1, RealT* normal )
  {
    const RealT edge_x = point1[0] - point0[0];
    const RealT edge_y = point1[1] - point0[1];
    const RealT edge_length = std::sqrt( edge_x * edge_x + edge_y * edge_y );
    if ( edge_length < minimum_edge_length_ ) {
      normal[0] = RealT{};
      normal[1] = RealT{};
      return false;
    }
    normal[0] = edge_y / edge_length;
    normal[1] = -edge_x / edge_length;
    return true;
  }
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_INTERFACEPAIRCHECKS_HPP_ */
