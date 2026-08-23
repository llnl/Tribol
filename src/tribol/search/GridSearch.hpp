// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_GRID_SEARCH_HPP_
#define SRC_TRIBOL_SEARCH_GRID_SEARCH_HPP_

#include "axom/primal.hpp"
#include "axom/slic.hpp"
#include "axom/spin.hpp"

#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"

#include <cmath>

namespace tribol {

/**
 * @brief Finds coarse candidate pairs with an Axom implicit grid
 *
 * The search inserts inflated bounding boxes for the first mesh into a spatial
 * index, then queries the index with each inflated second-mesh element box.
 * Grid search currently executes sequentially.
 *
 * @tparam Dimension Spatial dimension of the mesh coordinates.
 */
template <int Dimension>
class GridSearch {
  static_assert( Dimension == 2 || Dimension == 3, "GridSearch supports only two- and three-dimensional meshes." );

 public:
  /** Type returned by findPairs(). */
  using PairRange = ArrayT<ElementPair>;

  /**
   * @brief Construct a grid-search policy
   *
   * @param [in] proximity_scale Element-size multiplier used to inflate each
   * element box.
   * @param [in] allocator_id Allocator used for explicitly stored result pairs.
   */
  GridSearch( RealT proximity_scale, int allocator_id )
      : proximity_scale_( proximity_scale ), allocator_id_( allocator_id )
  {
  }

  /**
   * @brief Build and query an implicit grid for two meshes
   *
   * Empty input meshes return an empty pair array.
   *
   * @param [in] mesh1 Mesh whose element boxes are inserted into the grid.
   * @param [in] mesh2 Mesh whose element boxes query the grid.
   * @return Candidate pairs stored with the configured allocator.
   */
  PairRange findPairs( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) const
  {
    PairRange pairs( 0, 0, allocator_id_ );

    if ( mesh1.numberOfElements() == 0 || mesh2.numberOfElements() == 0 ) {
      return pairs;
    }

    // TODO: Determine whether this tolerance should scale with the mesh.
    const RealT bbox_tolerance = 1e-6;

    // Cache first-mesh element boxes for grid construction and insertion.
    ArrayT<SpatialBoundingBox> mesh_bboxes1;
    mesh_bboxes1.reserve( mesh1.numberOfElements() );
    for ( int i = 0; i < mesh1.numberOfElements(); ++i ) {
      mesh_bboxes1.emplace_back( elementBoundingBox( mesh1, i ) );
    }

    // Estimate a grid resolution from the average inflated element-box extent.
    SpaceVec ranges;
    SpatialBoundingBox grid_bbox;
    for ( int i = 0; i < mesh1.numberOfElements(); ++i ) {
      auto& bbox = mesh_bboxes1[i];
      inflateBBox( bbox, proximity_scale_ );
      ranges += bbox.range();
      grid_bbox.addBox( bbox );
    }

    // Keep inserted boxes away from the grid boundary despite roundoff.
    grid_bbox.scale( 1 + bbox_tolerance );
    ranges /= static_cast<double>( mesh1.numberOfElements() );

    typename ImplicitGridType::GridCell resolution;
    SpaceVec bbox_range = grid_bbox.range();
    const RealT scale_factor = 0.5;  // TODO: Determine whether this factor should be mesh-dependent.
    for ( int i = 0; i < Dimension; ++i ) {
      resolution[i] = static_cast<IndexT>( std::ceil( scale_factor * bbox_range[i] / ranges[i] ) );
    }

    ImplicitGridType grid;
    grid.initialize( grid_bbox, &resolution, mesh1.numberOfElements() );
    for ( int i = 0; i < mesh1.numberOfElements(); ++i ) {
      grid.insert( mesh_bboxes1[i], i );
    }

    SLIC_DEBUG( "Implicit Grid info: " << "\n Mesh 1 bounding box (inflated): " << grid_bbox
                                       << "\n Avg range: " << ranges << "\n Computed resolution: " << resolution );

    SpatialBoundingBox bbox2;
    for ( int i = 0; i < mesh2.numberOfElements(); ++i ) {
      bbox2.addBox( elementBoundingBox( mesh2, i ) );
    }
    SLIC_DEBUG( "Mesh 2 bounding box is: " << bbox2 );

    using BitsetType = typename ImplicitGridType::BitsetType;
    for ( int element_id2 = 0; element_id2 < mesh2.numberOfElements(); ++element_id2 ) {
      SpatialBoundingBox bbox = elementBoundingBox( mesh2, element_id2 );
      inflateBBox( bbox, proximity_scale_ );

      auto candidate_bits = grid.getCandidates( bbox );
      for ( IndexT element_id1 = candidate_bits.find_first(); element_id1 != BitsetType::npos;
            element_id1 = candidate_bits.find_next( element_id1 ) ) {
        if ( mesh1.meshId() == mesh2.meshId() && element_id1 < element_id2 ) {
          continue;
        }

        // TODO: Reject candidates whose boxes share grid cells but do not overlap.
        pairs.emplace_back( element_id1, element_id2 );
      }
    }

    return pairs;
  }

 private:
  /** Axis-aligned bounding-box type used by the grid. */
  using BBox = axom::primal::BoundingBox<RealT, Dimension>;

  /** Point type used to construct element bounding boxes. */
  using PointT = axom::primal::Point<RealT, Dimension>;

  /** Axom implicit-grid type used by this search policy. */
  using ImplicitGridType = axom::spin::ImplicitGrid<Dimension, axom::SEQ_EXEC, int>;

  /** Coordinate-vector type used by the implicit grid. */
  using SpaceVec = typename ImplicitGridType::SpaceVec;

  /** Bounding-box type expected by the implicit grid. */
  using SpatialBoundingBox = typename ImplicitGridType::SpatialBoundingBox;

  /**
   * @brief Compute the axis-aligned bounding box of one mesh element
   *
   * @param [in] mesh Mesh containing the element.
   * @param [in] element_id Element index in @p mesh.
   * @return Minimal axis-aligned box containing the element nodes.
   */
  static BBox elementBoundingBox( const MeshData::Viewer& mesh, IndexT element_id )
  {
    // Support Axom releases before and after NumericArray moved into primal.
    using namespace axom;
    using namespace axom::primal;

    BBox box;
    for ( int i = 0; i < mesh.numberOfNodesPerElement(); ++i ) {
      NumericArray<RealT, Dimension> vertex;
      auto node_id = mesh.getGlobalNodeId( element_id, i );
      for ( int dimension = 0; dimension < Dimension; ++dimension ) {
        vertex[dimension] = mesh.getPosition()[dimension][node_id];
      }
      box.addPoint( PointT( vertex ) );
    }
    return box;
  }

  /**
   * @brief Expand a box isotropically based on its longest dimension
   *
   * @param [in,out] bbox Box to expand.
   * @param [in] range_multiplier Multiplier applied to the box's longest
   * extent to obtain the expansion distance.
   */
  static void inflateBBox( SpatialBoundingBox& bbox, RealT range_multiplier )
  {
    int dimension = bbox.getLongestDimension();
    const RealT expansion_factor = range_multiplier * bbox.range()[dimension];
    bbox.expand( expansion_factor );
  }

  /** Element-size multiplier used to inflate element boxes. */
  RealT proximity_scale_;

  /** Allocator used for explicitly stored result pairs. */
  int allocator_id_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_GRID_SEARCH_HPP_ */
