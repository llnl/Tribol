// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_BVHSEARCH_HPP_
#define SRC_TRIBOL_SEARCH_BVHSEARCH_HPP_

#include <cmath>

#include "axom/core/execution/for_all.hpp"
#include "axom/primal.hpp"
#include "axom/slic.hpp"
#include "axom/spin.hpp"

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/utils/Algorithm.hpp"

namespace tribol {

/**
 * @brief Uses a bounding volume hierarchy to find spatially proximate element pairs
 *
 * BvhSearch performs only coarse spatial binning. It intentionally does not apply method-specific normal, overlap, or
 * contact checks. Callers decide which returned candidates are appropriate for their contact formulation.
 *
 * The search stores non-owning mesh views. The MeshData objects and their caller-owned arrays must remain valid for the
 * lifetime of this object. Each query builds and releases its BVH and bounding boxes so moving coordinates cannot leave
 * stale spatial data on the search object.
 *
 * @tparam SpatialDimension Spatial dimension of the mesh coordinates
 * @tparam ExecSpace Axom execution space used to construct and query bounding boxes
 */
template <int SpatialDimension, typename ExecSpace>
class BvhSearch {
 public:
  using Bvh = axom::spin::BVH<SpatialDimension, ExecSpace, RealT>;
  using BoundingBox = typename Bvh::BoxType;
  using Point = axom::primal::Point<RealT, SpatialDimension>;

  /**
   * @brief Store the meshes and proximity values used by each one-shot BVH search
   *
   * @param mesh1 Elements inserted into the BVH
   * @param mesh2 Elements used to query the BVH
   * @param binning_proximity_scale Multiplier applied to each element's maximum nodal radius
   * @param residual_gap Additional physical distance added to each bounding-box expansion
   */
  BvhSearch( MeshData::Viewer mesh1, MeshData::Viewer mesh2, RealT binning_proximity_scale, RealT residual_gap )
      : mesh1_( mesh1 ),
        mesh2_( mesh2 ),
        binning_proximity_scale_( binning_proximity_scale ),
        residual_gap_( residual_gap )
  {
    SLIC_ASSERT_MSG( mesh1_.getAllocatorId() == mesh2_.getAllocatorId(),
                     "BvhSearch meshes must use the same allocator." );
  }

  /**
   * @brief Build a BVH and return pairs whose expanded bounding boxes intersect
   *
   * Bounding-box candidates can still include self-pairs, shared-node pairs, co-oriented faces, and pairs that fail a
   * contact method's overlap checks. Keeping those decisions outside this class allows each formulation to apply the
   * geometry checks it requires.
   */
  ArrayT<InterfacePair> findCandidatePairs() const
  {
    ArrayT<InterfacePair> candidate_pairs( 0, 0, mesh1_.getAllocatorId() );
    if ( mesh1_.numberOfElements() == 0 || mesh2_.numberOfElements() == 0 ) {
      return candidate_pairs;
    }

    // Search storage is local because moving meshes require the BVH and bounds to be rebuilt for every query.
    ArrayT<BoundingBox> mesh1_bounding_boxes( axom::ArrayOptions::Uninitialized{}, mesh1_.numberOfElements(),
                                              mesh1_.numberOfElements(), mesh1_.getAllocatorId() );
    ArrayT<BoundingBox> mesh2_bounding_boxes( axom::ArrayOptions::Uninitialized{}, mesh2_.numberOfElements(),
                                              mesh2_.numberOfElements(), mesh2_.getAllocatorId() );
    ArrayT<IndexT> candidates( axom::ArrayOptions::Uninitialized{}, 0, 0, mesh1_.getAllocatorId() );
    ArrayT<IndexT> offsets( axom::ArrayOptions::Uninitialized{}, mesh2_.numberOfElements(), mesh2_.numberOfElements(),
                            mesh2_.getAllocatorId() );
    ArrayT<IndexT> counts( axom::ArrayOptions::Uninitialized{}, mesh2_.numberOfElements(), mesh2_.numberOfElements(),
                           mesh2_.getAllocatorId() );

    buildMeshBoundingBoxes( mesh1_bounding_boxes, mesh1_ );
    buildMeshBoundingBoxes( mesh2_bounding_boxes, mesh2_ );

    Bvh bvh;
    bvh.setAllocatorID( mesh1_.getAllocatorId() );
    bvh.initialize( mesh1_bounding_boxes.view(), mesh1_bounding_boxes.size() );
    bvh.findBoundingBoxes( offsets.view(), counts.view(), candidates, mesh2_.numberOfElements(),
                           mesh2_bounding_boxes.view() );

    candidate_pairs = ArrayT<InterfacePair>( axom::ArrayOptions::Uninitialized{}, candidates.size(), candidates.size(),
                                             mesh1_.getAllocatorId() );
    const auto candidate_ids = candidates.view();
    const auto candidate_offsets = offsets.view();
    const auto candidate_counts = counts.view();
    auto pairs = candidate_pairs.view();
    axom::for_all<ExecSpace>(
        candidates.size(), [candidate_ids, candidate_offsets, candidate_counts, pairs] TRIBOL_HOST_DEVICE( IndexT i ) {
          const IndexT mesh1_element = candidate_ids[i];
          const IndexT mesh2_element = algorithm::binarySearch( candidate_offsets, candidate_counts, i );
          pairs[i] = InterfacePair{ mesh1_element, mesh2_element };
        } );

    return candidate_pairs;
  }

 private:
  /**
   * @brief Bound every finite-element node and expand by a node-based element size
   *
   * Quadratic nodes participate in both the box and nodal-radius calculations. Assuming the curved element remains in
   * their convex hull, the unexpanded box contains its geometry. Isotropic expansion avoids requiring a face normal,
   * which belongs to the formulation-specific geometry rather than coarse search.
   */
  void buildMeshBoundingBoxes( ArrayT<BoundingBox>& bounding_boxes, const MeshData::Viewer& mesh ) const
  {
    auto boxes = bounding_boxes.view();
    const RealT binning_proximity_scale = binning_proximity_scale_;
    const RealT residual_gap = residual_gap_;
    axom::for_all<ExecSpace>( mesh.numberOfElements(), [mesh, boxes, binning_proximity_scale,
                                                        residual_gap] TRIBOL_HOST_DEVICE( IndexT element ) {
      BoundingBox bounding_box;
      Point average_position;
      for ( int dimension = 0; dimension < SpatialDimension; ++dimension ) {
        average_position[dimension] = 0.0;
      }

      const IndexT node_count = mesh.numberOfNodesPerElement();
      for ( IndexT local_node = 0; local_node < node_count; ++local_node ) {
        const IndexT node = mesh.getGlobalNodeId( element, local_node );
        Point position;
        for ( int dimension = 0; dimension < SpatialDimension; ++dimension ) {
          position[dimension] = mesh.getPosition()[dimension][node];
          average_position[dimension] += position[dimension];
        }
        bounding_box.addPoint( position );
      }

      for ( int dimension = 0; dimension < SpatialDimension; ++dimension ) {
        average_position[dimension] /= static_cast<RealT>( node_count );
      }

      RealT squared_nodal_radius = 0.0;
      for ( IndexT local_node = 0; local_node < node_count; ++local_node ) {
        const IndexT node = mesh.getGlobalNodeId( element, local_node );
        RealT squared_distance = 0.0;
        for ( int dimension = 0; dimension < SpatialDimension; ++dimension ) {
          const RealT distance = mesh.getPosition()[dimension][node] - average_position[dimension];
          squared_distance += distance * distance;
        }
        squared_nodal_radius = squared_distance > squared_nodal_radius ? squared_distance : squared_nodal_radius;
      }

      const RealT expansion = binning_proximity_scale * std::sqrt( squared_nodal_radius ) + residual_gap;
      bounding_box.expand( expansion );
      boxes[element] = bounding_box;
    } );
  }

  MeshData::Viewer mesh1_;
  MeshData::Viewer mesh2_;
  RealT binning_proximity_scale_;
  RealT residual_gap_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_BVHSEARCH_HPP_ */
