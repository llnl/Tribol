// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_BVHSEARCH_HPP_
#define SRC_TRIBOL_SEARCH_BVHSEARCH_HPP_

#include <cmath>

#include "axom/primal.hpp"
#include "axom/slic.hpp"
#include "axom/spin.hpp"

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Atomics.hpp"
#include "tribol/common/ExecModel.hpp"
#include "tribol/common/LoopExec.hpp"
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
 * Each call accepts current mesh views, then builds and releases its BVH and bounding boxes. The search object does not
 * retain views that may become stale when a caller replaces mesh arrays.
 *
 * @tparam SpatialDimension Spatial dimension of the mesh coordinates
 * @tparam ExecMode Tribol execution mode used to construct and query bounding boxes
 */
template <int SpatialDimension, ExecutionMode ExecMode>
class BvhSearch {
 public:
  using AxomExecSpace = typename toAxomExecSpace<ExecMode>::type;
  using Bvh = axom::spin::BVH<SpatialDimension, AxomExecSpace, RealT>;
  using BoundingBox = typename Bvh::BoxType;
  using Point = axom::primal::Point<RealT, SpatialDimension>;

  /**
   * @brief Store the proximity values applied by each BVH search
   *
   * @param binning_proximity_scale Multiplier applied to each element's maximum nodal radius
   * @param residual_gap Additional physical distance added to each bounding-box expansion
   */
  BvhSearch( RealT binning_proximity_scale, RealT residual_gap )
      : binning_proximity_scale_( binning_proximity_scale ), residual_gap_( residual_gap )
  {
  }

  /**
   * @brief Build a BVH and return pairs whose expanded bounding boxes intersect
   *
   * Bounding-box candidates can still include shared-node pairs, co-oriented faces, and pairs that fail a contact
   * method's overlap checks. Same-mesh searches remove exact self-pairs and retain one direction for mirrored pairs.
   * Keeping the remaining decisions outside this class allows each formulation to apply the geometry checks it
   * requires.
   *
   * @param mesh1 Current elements inserted into the BVH
   * @param mesh2 Current elements used to query the BVH
   */
  ArrayT<InterfacePair> findCandidatePairs( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
  {
    SLIC_ASSERT_MSG( mesh1.getAllocatorId() == mesh2.getAllocatorId(),
                     "BvhSearch meshes must use the same allocator." );
    ArrayT<InterfacePair> candidate_pairs( 0, 0, mesh1.getAllocatorId() );
    if ( mesh1.numberOfElements() == 0 || mesh2.numberOfElements() == 0 ) {
      return candidate_pairs;
    }

    // Search storage is local because moving meshes require the BVH and bounds to be rebuilt for every query.
    ArrayT<BoundingBox> mesh1_bounding_boxes( axom::ArrayOptions::Uninitialized{}, mesh1.numberOfElements(),
                                              mesh1.numberOfElements(), mesh1.getAllocatorId() );
    ArrayT<BoundingBox> mesh2_bounding_boxes( axom::ArrayOptions::Uninitialized{}, mesh2.numberOfElements(),
                                              mesh2.numberOfElements(), mesh2.getAllocatorId() );
    ArrayT<IndexT> candidates( axom::ArrayOptions::Uninitialized{}, 0, 0, mesh1.getAllocatorId() );
    ArrayT<IndexT> offsets( axom::ArrayOptions::Uninitialized{}, mesh2.numberOfElements(), mesh2.numberOfElements(),
                            mesh2.getAllocatorId() );
    ArrayT<IndexT> counts( axom::ArrayOptions::Uninitialized{}, mesh2.numberOfElements(), mesh2.numberOfElements(),
                           mesh2.getAllocatorId() );

    buildMeshBoundingBoxes( mesh1_bounding_boxes, mesh1 );
    buildMeshBoundingBoxes( mesh2_bounding_boxes, mesh2 );

    Bvh bvh;
    bvh.setAllocatorID( mesh1.getAllocatorId() );
    bvh.initialize( mesh1_bounding_boxes.view(), mesh1_bounding_boxes.size() );
    bvh.findBoundingBoxes( offsets.view(), counts.view(), candidates, mesh2.numberOfElements(),
                           mesh2_bounding_boxes.view() );

    const auto candidate_ids = candidates.view();
    const auto candidate_offsets = offsets.view();
    const auto candidate_counts = counts.view();
    const bool same_mesh = mesh1.meshId() == mesh2.meshId();

    if ( !same_mesh ) {
      candidate_pairs = ArrayT<InterfacePair>( axom::ArrayOptions::Uninitialized{}, candidates.size(),
                                               candidates.size(), mesh1.getAllocatorId() );
      auto pairs = candidate_pairs.view();
      // This loop must finish before its local BVH query arrays are released at the end of the call.
      forAllExec<ExecMode, false>( candidates.size(), [candidate_ids, candidate_offsets, candidate_counts,
                                                       pairs] TRIBOL_HOST_DEVICE( IndexT candidate ) {
        const IndexT mesh1_element = candidate_ids[candidate];
        const IndexT mesh2_element = algorithm::binarySearch( candidate_offsets, candidate_counts, candidate );
        pairs[candidate] = InterfacePair{ mesh1_element, mesh2_element };
      } );
    } else {
      ArrayT<IndexT> retained_pair_count_data( { 0 }, mesh1.getAllocatorId() );
      auto retained_pair_count = retained_pair_count_data.view();

      // A same-mesh search visits both pair directions. Exact self-pairs cannot represent contact, so retain only one
      // direction of each pair with different element IDs.
      forAllExec<ExecMode, false>( candidates.size(), [candidate_ids, candidate_offsets, candidate_counts,
                                                       retained_pair_count] TRIBOL_HOST_DEVICE( IndexT candidate ) {
        const IndexT mesh1_element = candidate_ids[candidate];
        const IndexT mesh2_element = algorithm::binarySearch( candidate_offsets, candidate_counts, candidate );
        if ( mesh1_element > mesh2_element ) {
          tribol::atomicInc( retained_pair_count.data() );
        }
      } );

      ArrayT<IndexT, 1, MemorySpace::Host> retained_pair_count_host( retained_pair_count_data );
      candidate_pairs = ArrayT<InterfacePair>( axom::ArrayOptions::Uninitialized{}, retained_pair_count_host[0],
                                               retained_pair_count_host[0], mesh1.getAllocatorId() );
      retained_pair_count_data.fill( 0 );
      auto pairs = candidate_pairs.view();

      // Repeating the inexpensive ID comparison avoids retaining a candidate-sized Boolean array between kernels.
      forAllExec<ExecMode, false>(
          candidates.size(), [candidate_ids, candidate_offsets, candidate_counts, retained_pair_count,
                              pairs] TRIBOL_HOST_DEVICE( IndexT candidate ) {
            const IndexT mesh1_element = candidate_ids[candidate];
            const IndexT mesh2_element = algorithm::binarySearch( candidate_offsets, candidate_counts, candidate );
            if ( mesh1_element <= mesh2_element ) {
              return;
            }
            const IndexT retained_pair_index = tribol::atomicInc( retained_pair_count.data() );
            pairs[retained_pair_index] = InterfacePair{ mesh1_element, mesh2_element };
          } );
    }

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
    // The BVH reads these boxes immediately after this function returns, so the loop must complete before returning.
    forAllExec<ExecMode, false>( mesh.numberOfElements(), [mesh, boxes, binning_proximity_scale,
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

  RealT binning_proximity_scale_;
  RealT residual_gap_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_BVHSEARCH_HPP_ */
