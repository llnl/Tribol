// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_INTERFACEPAIRCHECKS_HPP_
#define SRC_TRIBOL_SEARCH_INTERFACEPAIRCHECKS_HPP_

#include "axom/primal.hpp"

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
   * @brief Return whether two linear-edge normals are separated by at least 90 degrees
   *
   * Normals are computed directly from endpoint coordinates so formulations do not require MeshData::computeFaceData()
   * before filtering candidates. Degenerate edges do not define an orientation and return false.
   */
  TRIBOL_HOST_DEVICE static bool linearEdgeNormalsAreOpposing( const MeshData::Viewer& mesh1, IndexT element_id1,
                                                               const MeshData::Viewer& mesh2, IndexT element_id2 )
  {
    const Vector2D edge1 = getLinearEdgeVector( mesh1, element_id1 );
    const Vector2D edge2 = getLinearEdgeVector( mesh2, element_id2 );

    const RealT minimum_edge_length_squared = minimum_edge_length_ * minimum_edge_length_;
    if ( edge1.squared_norm() < minimum_edge_length_squared || edge2.squared_norm() < minimum_edge_length_squared ) {
      return false;
    }

    // Rotating both directed edge vectors 90 degrees produces their normal vectors and preserves their dot product.
    // Normalizing those vectors only divides the dot product by two positive edge lengths, so it cannot change the
    // sign. The edge dot product therefore identifies opposing normals without constructing or normalizing them.
    return edge1.dot( edge2 ) <= 0.0;
  }

  /**
   * @brief Return whether two linear 3D surface-element normals are separated by at least 90 degrees.
   *
   * Normals are computed directly from vertex coordinates so formulations do not require MeshData::computeFaceData()
   * before filtering candidates. Supports 3-node triangles and 4-node quadrilaterals; degenerate faces return false.
   *
   * @param mesh1 First surface mesh view.
   * @param element_id1 Surface element index on mesh1.
   * @param mesh2 Second surface mesh view.
   * @param element_id2 Surface element index on mesh2.
   * @return True when both faces have nondegenerate normals with dot product <= 0.
   */
  TRIBOL_HOST_DEVICE static bool linearFaceNormalsAreOpposing( const MeshData::Viewer& mesh1, IndexT element_id1,
                                                               const MeshData::Viewer& mesh2, IndexT element_id2 )
  {
    const Vector3D normal1 = getLinearFaceNormalVector( mesh1, element_id1 );
    const Vector3D normal2 = getLinearFaceNormalVector( mesh2, element_id2 );
    const RealT min_area_sq = minimum_face_twice_area_ * minimum_face_twice_area_;
    if ( normal1.squared_norm() < min_area_sq || normal2.squared_norm() < min_area_sq ) {
      return false;
    }
    return normal1.dot( normal2 ) <= 0.0;
  }

 private:
  /** @brief Use the static checks without constructing an instance. */
  InterfacePairChecks() = delete;

  using Point2D = axom::primal::Point<RealT, 2>;
  using Vector2D = axom::primal::Vector<RealT, 2>;
  using Point3D = axom::primal::Point<RealT, 3>;
  using Vector3D = axom::primal::Vector<RealT, 3>;

  inline static constexpr RealT minimum_edge_length_ = 1.0e-14;      ///< Edge-length degeneracy threshold.
  inline static constexpr RealT minimum_face_twice_area_ = 1.0e-28;  ///< Face twice-area degeneracy threshold.

  /**
   * @brief Return the directed edge from the first finite-element node to the second.
   * @param mesh Mesh view containing the edge.
   * @param element_id Element whose edge vector is read.
   */
  TRIBOL_HOST_DEVICE static Vector2D getLinearEdgeVector( const MeshData::Viewer& mesh, IndexT element_id )
  {
    const IndexT node0 = mesh.getGlobalNodeId( element_id, 0 );
    const IndexT node1 = mesh.getGlobalNodeId( element_id, 1 );
    const Point2D point0{ mesh.getPosition()[0][node0], mesh.getPosition()[1][node0] };
    const Point2D point1{ mesh.getPosition()[0][node1], mesh.getPosition()[1][node1] };
    return Vector2D( point0, point1 );
  }

  /**
   * @brief Return an unnormalized outward normal vector for a 3-node triangle or 4-node quadrilateral.
   * @param mesh Mesh view containing the 3D surface element.
   * @param element_id Element whose vertex coordinates are read.
   * @return Twice-area outward normal vector, or zero if the element has unsupported node count.
   */
  TRIBOL_HOST_DEVICE static Vector3D getLinearFaceNormalVector( const MeshData::Viewer& mesh, IndexT element_id )
  {
    const IndexT num_nodes = mesh.numberOfNodesPerElement();
    if ( num_nodes == 3 ) {
      const IndexT n0 = mesh.getGlobalNodeId( element_id, 0 );
      const IndexT n1 = mesh.getGlobalNodeId( element_id, 1 );
      const IndexT n2 = mesh.getGlobalNodeId( element_id, 2 );
      const Point3D p0{ mesh.getPosition()[0][n0], mesh.getPosition()[1][n0], mesh.getPosition()[2][n0] };
      const Point3D p1{ mesh.getPosition()[0][n1], mesh.getPosition()[1][n1], mesh.getPosition()[2][n1] };
      const Point3D p2{ mesh.getPosition()[0][n2], mesh.getPosition()[1][n2], mesh.getPosition()[2][n2] };
      return Vector3D::cross_product( Vector3D( p0, p1 ), Vector3D( p0, p2 ) );
    }
    if ( num_nodes == 4 ) {
      const IndexT n0 = mesh.getGlobalNodeId( element_id, 0 );
      const IndexT n1 = mesh.getGlobalNodeId( element_id, 1 );
      const IndexT n2 = mesh.getGlobalNodeId( element_id, 2 );
      const IndexT n3 = mesh.getGlobalNodeId( element_id, 3 );
      const Point3D p0{ mesh.getPosition()[0][n0], mesh.getPosition()[1][n0], mesh.getPosition()[2][n0] };
      const Point3D p1{ mesh.getPosition()[0][n1], mesh.getPosition()[1][n1], mesh.getPosition()[2][n1] };
      const Point3D p2{ mesh.getPosition()[0][n2], mesh.getPosition()[1][n2], mesh.getPosition()[2][n2] };
      const Point3D p3{ mesh.getPosition()[0][n3], mesh.getPosition()[1][n3], mesh.getPosition()[2][n3] };
      return Vector3D::cross_product( Vector3D( p0, p2 ), Vector3D( p1, p3 ) );
    }
    return Vector3D{ 0.0, 0.0, 0.0 };
  }
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_INTERFACEPAIRCHECKS_HPP_ */
