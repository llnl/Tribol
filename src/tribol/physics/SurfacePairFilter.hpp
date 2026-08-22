// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_SURFACEPAIRFILTER_HPP_
#define SRC_TRIBOL_PHYSICS_SURFACEPAIRFILTER_HPP_

#include "tribol/common/Parameters.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/utils/Math.hpp"

namespace tribol {

/** @brief Named geometric checks that surface-contact methods may compose. */
class SurfacePairFilter {
 public:
  /** @brief Disable construction of this static utility class. */
  SurfacePairFilter() = delete;

  /**
   * @brief Check that a pair does not identify the same element of one mesh.
   *
   * Equal element indices on different meshes are considered distinct.
   *
   * @param [in] mesh1 View of the first surface mesh.
   * @param [in] mesh2 View of the second surface mesh.
   * @param [in] element_id1 Element index in `mesh1`.
   * @param [in] element_id2 Element index in `mesh2`.
   * @return `true` when the pair identifies distinct elements; otherwise `false`.
   */
  TRIBOL_HOST_DEVICE static bool areDistinctElements( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                                      IndexT element_id1, IndexT element_id2 )
  {
    return mesh1.meshId() != mesh2.meshId() || element_id1 != element_id2;
  }

  /**
   * @brief Check that two elements do not reference a common global node.
   *
   * This check is typically selected for auto-contact to exclude adjacent surface elements. The caller decides whether
   * that policy is appropriate.
   *
   * @param [in] mesh1 View of the first surface mesh.
   * @param [in] mesh2 View of the second surface mesh.
   * @param [in] element_id1 Element index in `mesh1`.
   * @param [in] element_id2 Element index in `mesh2`.
   * @return `true` when the elements share no global node identifiers; otherwise `false`.
   */
  TRIBOL_HOST_DEVICE static bool haveNoSharedNodes( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                                    IndexT element_id1, IndexT element_id2 )
  {
    for ( IndexT i{ 0 }; i < mesh1.numberOfNodesPerElement(); ++i ) {
      const int node1 = mesh1.getGlobalNodeId( element_id1, i );
      for ( IndexT j{ 0 }; j < mesh2.numberOfNodesPerElement(); ++j ) {
        if ( node1 == mesh2.getGlobalNodeId( element_id2, j ) ) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * @brief Check that the element normals are opposing or orthogonal.
   *
   * The check accepts a pair when the normal dot product is less than or equal to zero and assumes element normals have
   * already been computed.
   *
   * @param [in] mesh1 View of the first surface mesh.
   * @param [in] mesh2 View of the second surface mesh.
   * @param [in] element_id1 Element index in `mesh1`.
   * @param [in] element_id2 Element index in `mesh2`.
   * @return `true` when the normal dot product is nonpositive; otherwise `false`.
   */
  TRIBOL_HOST_DEVICE static bool haveOpposingNormals( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                                      IndexT element_id1, IndexT element_id2 )
  {
    RealT normal_dot = 0.0;
    for ( int d{ 0 }; d < mesh1.spatialDimension(); ++d ) {
      normal_dot += mesh1.getElementNormals()[d][element_id1] * mesh2.getElementNormals()[d][element_id2];
    }
    return normal_dot <= 0.0;
  }

  /**
   * @brief Check centroid separation against a size-scaled proximity distance.
   *
   * In 3D, the limit is the proximity scale times the sum of face radii. In 2D, it is the scale times the sum of half
   * edge lengths. Conforming contact reduces the resulting limit to five percent of that value.
   *
   * @param [in] mesh1 View of the first surface mesh.
   * @param [in] mesh2 View of the second surface mesh.
   * @param [in] element_id1 Element index in `mesh1`.
   * @param [in] element_id2 Element index in `mesh2`.
   * @param [in] proximity_scale Multiplier applied to the combined element size.
   * @param [in] contact_mode Contact mode controlling the conforming-distance reduction.
   * @return `true` when the centroid separation does not exceed the scaled distance; otherwise `false`.
   */
  TRIBOL_HOST_DEVICE static bool areWithinCentroidProximity( const MeshData::Viewer& mesh1,
                                                             const MeshData::Viewer& mesh2, IndexT element_id1,
                                                             IndexT element_id2, RealT proximity_scale,
                                                             ContactMode contact_mode )
  {
    RealT max_distance;
    RealT distance;
    if ( mesh1.spatialDimension() == 3 ) {
      max_distance = proximity_scale * ( mesh1.getFaceRadius()[element_id1] + mesh2.getFaceRadius()[element_id2] );
      const RealT dx = mesh2.getElementCentroids()[0][element_id2] - mesh1.getElementCentroids()[0][element_id1];
      const RealT dy = mesh2.getElementCentroids()[1][element_id2] - mesh1.getElementCentroids()[1][element_id1];
      const RealT dz = mesh2.getElementCentroids()[2][element_id2] - mesh1.getElementCentroids()[2][element_id1];
      distance = magnitude( dx, dy, dz );
    } else {
      max_distance =
          proximity_scale * ( 0.5 * mesh1.getElementAreas()[element_id1] + 0.5 * mesh2.getElementAreas()[element_id2] );
      const RealT dx = mesh2.getElementCentroids()[0][element_id2] - mesh1.getElementCentroids()[0][element_id1];
      const RealT dy = mesh2.getElementCentroids()[1][element_id2] - mesh1.getElementCentroids()[1][element_id1];
      distance = magnitude( dx, dy );
    }

    if ( contact_mode == SURFACE_TO_SURFACE_CONFORMING ) {
      max_distance *= 0.05;
    }
    return distance <= max_distance;
  }

  /**
   * @brief Check projected overlap on the plane of the second element.
   *
   * Both elements are projected orthogonally onto the plane defined by the second element's normal and centroid. Thus,
   * the projection direction is parallel to the second element's normal. The check passes when the projected edges or
   * faces have positive overlap according to IsOverlappingOnPlane().
   *
   * @param [in] mesh1 View of the first surface mesh.
   * @param [in] mesh2 View of the second surface mesh whose plane defines the projection target.
   * @param [in] element_id1 Element index in `mesh1`.
   * @param [in] element_id2 Element index in `mesh2`.
   * @return `true` when the projected elements have positive overlap; otherwise `false`.
   *
   * @pre Each surface element contains at most four nodes.
   */
  TRIBOL_HOST_DEVICE static bool haveProjectedOverlapOnSecondElementPlane( const MeshData::Viewer& mesh1,
                                                                           const MeshData::Viewer& mesh2,
                                                                           IndexT element_id1, IndexT element_id2 )
  {
    constexpr int max_dim = 3;
    constexpr int max_nodes_per_element = 4;

    RealT x1[max_nodes_per_element]{};
    RealT y1[max_nodes_per_element]{};
    RealT z1[max_nodes_per_element]{};
    RealT x2[max_nodes_per_element]{};
    RealT y2[max_nodes_per_element]{};
    RealT z2[max_nodes_per_element]{};
    gatherElementCoordinateComponents( mesh1, element_id1, x1, y1, z1 );
    gatherElementCoordinateComponents( mesh2, element_id2, x2, y2, z2 );

    RealT normal[max_dim]{};
    RealT centroid[max_dim]{};
    mesh2.getFaceNormal( element_id2, normal );
    mesh2.getFaceCentroid( element_id2, centroid );

    return IsOverlappingOnPlane( x1, y1, z1, x2, y2, z2, normal, centroid, mesh1.numberOfNodesPerElement(),
                                 mesh2.numberOfNodesPerElement(), mesh1.spatialDimension() );
  }

  /**
   * @brief Check projected overlap on an intermediate element plane.
   *
   * The plane centroid is midway between the element centroids and its normal is the normalized half-difference of the
   * element normals. Both elements are projected orthogonally onto this intermediate plane, parallel to its normal. In
   * 3D, nodes of each potentially non-planar face are first projected orthogonally to that face's own average plane,
   * parallel to that face's normal.
   *
   * @param [in] mesh1 View of the first surface mesh.
   * @param [in] mesh2 View of the second surface mesh.
   * @param [in] element_id1 Element index in `mesh1`.
   * @param [in] element_id2 Element index in `mesh2`.
   * @return `true` when the projected elements have positive overlap; otherwise `false`.
   *
   * @pre Each surface element contains at most four nodes.
   * @pre The element normals are not co-oriented; callers typically establish this with haveOpposingNormals().
   */
  TRIBOL_HOST_DEVICE static bool haveProjectedOverlapOnIntermediatePlane( const MeshData::Viewer& mesh1,
                                                                          const MeshData::Viewer& mesh2,
                                                                          IndexT element_id1, IndexT element_id2 )
  {
    constexpr int max_dim = 3;
    constexpr int max_nodes_per_element = 4;

    const int dim = mesh1.spatialDimension();
    RealT normal1[max_dim]{};
    RealT centroid1[max_dim]{};
    mesh1.getFaceNormal( element_id1, normal1 );
    mesh1.getFaceCentroid( element_id1, centroid1 );

    RealT normal2[max_dim]{};
    RealT centroid2[max_dim]{};
    mesh2.getFaceNormal( element_id2, normal2 );
    mesh2.getFaceCentroid( element_id2, centroid2 );

    RealT normal[max_dim]{};
    RealT centroid[max_dim]{};
    for ( int i{ 0 }; i < dim; ++i ) {
      normal[i] = 0.5 * ( normal2[i] - normal1[i] );
      centroid[i] = 0.5 * ( centroid1[i] + centroid2[i] );
    }

    const RealT normal_magnitude =
        dim == 3 ? magnitude( normal[0], normal[1], normal[2] ) : magnitude( normal[0], normal[1], 0.0 );
    const RealT inverse_normal_magnitude = 1.0 / normal_magnitude;
    for ( int i{ 0 }; i < dim; ++i ) {
      normal[i] *= inverse_normal_magnitude;
    }

    RealT x1[max_nodes_per_element]{};
    RealT y1[max_nodes_per_element]{};
    RealT z1[max_nodes_per_element]{};
    RealT x2[max_nodes_per_element]{};
    RealT y2[max_nodes_per_element]{};
    RealT z2[max_nodes_per_element]{};
    if ( dim == 3 ) {
      ProjectFaceNodesToPlane( mesh1, element_id1, normal1[0], normal1[1], normal1[2], centroid1[0], centroid1[1],
                               centroid1[2], x1, y1, z1 );
      ProjectFaceNodesToPlane( mesh2, element_id2, normal2[0], normal2[1], normal2[2], centroid2[0], centroid2[1],
                               centroid2[2], x2, y2, z2 );
    } else {
      gatherElementCoordinateComponents( mesh1, element_id1, x1, y1, z1 );
      gatherElementCoordinateComponents( mesh2, element_id2, x2, y2, z2 );
    }

    return IsOverlappingOnPlane( x1, y1, z1, x2, y2, z2, normal, centroid, mesh1.numberOfNodesPerElement(),
                                 mesh2.numberOfNodesPerElement(), dim );
  }

 private:
  /**
   * @brief Gather the Cartesian coordinates of an element's nodes.
   *
   * @param [in] mesh View of the surface mesh.
   * @param [in] element_id Element index in `mesh`.
   * @param [out] x Array receiving nodal x-coordinates.
   * @param [out] y Array receiving nodal y-coordinates.
   * @param [out] z Array receiving nodal z-coordinates in 3D; unchanged in 2D.
   *
   * @pre Each output array has space for `mesh.numberOfNodesPerElement()` values.
   */
  TRIBOL_HOST_DEVICE static void gatherElementCoordinateComponents( const MeshData::Viewer& mesh, IndexT element_id,
                                                                    RealT* x, RealT* y, RealT* z )
  {
    for ( IndexT i{ 0 }; i < mesh.numberOfNodesPerElement(); ++i ) {
      const IndexT node_id = mesh.getGlobalNodeId( element_id, i );
      x[i] = mesh.getPosition()[0][node_id];
      y[i] = mesh.getPosition()[1][node_id];
      if ( mesh.spatialDimension() == 3 ) {
        z[i] = mesh.getPosition()[2][node_id];
      }
    }
  }
};

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_SURFACEPAIRFILTER_HPP_ */
