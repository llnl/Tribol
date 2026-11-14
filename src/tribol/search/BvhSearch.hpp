// Copyright (c) Lawrence Livermore National Security, LLC and
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

#include "tribol/search/CoarseSearch.hpp"

namespace tribol {

/**
 * @brief Uses a bounding volume hierarchy to find spatially proximate element pairs
 *
 * BvhSearch performs only coarse spatial binning. It intentionally does not apply method-specific normal, overlap, or
 * contact checks. Callers decide which returned candidates are appropriate for their contact formulation.
 *
 * The nodal bounding boxes assume each element's geometry lies within the convex hull of its nodes. This assumption
 * may fail for higher-order (e.g. quadratic) meshes, so the boxes may not enclose the full element geometry.
 *
 * Each call accepts current mesh views, then builds and releases its BVH and bounding boxes. The search object does not
 * retain views that may become stale when a caller replaces mesh arrays.
 *
 * @tparam SpatialDim Spatial dimension of the mesh coordinates
 * @tparam Exec Tribol execution mode used to construct and query bounding boxes
 */
template <int SpatialDim, ExecutionMode Exec>
class BvhSearch : public CoarseSearch {
 public:
  using AxomExecSpace = typename toAxomExecSpace<Exec>::type;
  using Bvh = axom::spin::BVH<SpatialDim, AxomExecSpace, RealT>;
  using BoundingBox = typename Bvh::BoxType;
  using Point = axom::primal::Point<RealT, SpatialDim>;

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
  ArrayT<InterfacePair> findCandidatePairs( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override
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
      forAllExec<Exec, false>( candidates.size(), [candidate_ids, candidate_offsets, candidate_counts,
                                                   pairs] TRIBOL_HOST_DEVICE( IndexT candidate ) {
        const IndexT mesh1_element = candidate_ids[candidate];
        const IndexT mesh2_element = algorithm::binarySearch( candidate_offsets, candidate_counts, candidate );
        pairs[candidate] = InterfacePair{ mesh1_element, mesh2_element };
      } );
    } else {
      ArrayT<IndexT> num_retained_pairs_data( { 0 }, mesh1.getAllocatorId() );
      auto num_retained_pairs = num_retained_pairs_data.view();

      // A same-mesh search visits both pair directions. Exact self-pairs cannot represent contact, so retain only one
      // direction of each pair with different element IDs.
      forAllExec<Exec, false>( candidates.size(), [candidate_ids, candidate_offsets, candidate_counts,
                                                   num_retained_pairs] TRIBOL_HOST_DEVICE( IndexT candidate ) {
        const IndexT mesh1_element = candidate_ids[candidate];
        const IndexT mesh2_element = algorithm::binarySearch( candidate_offsets, candidate_counts, candidate );
        if ( mesh1_element > mesh2_element ) {
          tribol::atomicInc( num_retained_pairs.data() );
        }
      } );

      ArrayT<IndexT, 1, MemorySpace::Host> num_retained_pairs_host( num_retained_pairs_data );
      candidate_pairs = ArrayT<InterfacePair>( axom::ArrayOptions::Uninitialized{}, num_retained_pairs_host[0],
                                               num_retained_pairs_host[0], mesh1.getAllocatorId() );
      num_retained_pairs_data.fill( 0 );
      auto pairs = candidate_pairs.view();

      // Repeating the inexpensive ID comparison avoids retaining a candidate-sized Boolean array between kernels.
      forAllExec<Exec, false>( candidates.size(), [candidate_ids, candidate_offsets, candidate_counts,
                                                   num_retained_pairs, pairs] TRIBOL_HOST_DEVICE( IndexT candidate ) {
        const IndexT mesh1_element = candidate_ids[candidate];
        const IndexT mesh2_element = algorithm::binarySearch( candidate_offsets, candidate_counts, candidate );
        if ( mesh1_element <= mesh2_element ) {
          return;
        }
        const IndexT retained_pair_index = tribol::atomicInc( num_retained_pairs.data() );
        pairs[retained_pair_index] = InterfacePair{ mesh1_element, mesh2_element };
      } );
    }

    return candidate_pairs;
  }

 public:
  /**
   * @brief Bound every finite-element node and expand by a node-based element size
   *
   * Quadratic nodes participate in both the box and nodal-radius calculations. Assuming the curved element remains in
   * their convex hull, the unexpanded box contains its geometry. Isotropic expansion avoids requiring a face normal,
   * which belongs to the formulation-specific geometry rather than coarse search.
   *
   * NVCC requires the enclosing function of an extended device lambda to be public, so this implementation helper is
   * visible even though callers only need @ref findCandidatePairs.
   *
   * @param bounding_boxes Preallocated output with one entry per mesh element.
   * @param mesh Current mesh coordinates and connectivity.
   */
  void buildMeshBoundingBoxes( ArrayT<BoundingBox>& bounding_boxes, const MeshData::Viewer& mesh ) const
  {
    auto boxes = bounding_boxes.view();
    const RealT binning_proximity_scale = binning_proximity_scale_;
    const RealT residual_gap = residual_gap_;
    // The BVH reads these boxes immediately after this function returns, so the loop must complete before returning.
    forAllExec<Exec, false>( mesh.numberOfElements(), [mesh, boxes, binning_proximity_scale,
                                                       residual_gap] TRIBOL_HOST_DEVICE( IndexT element ) {
      BoundingBox bounding_box;
      Point average_position;
      for ( int dim = 0; dim < SpatialDim; ++dim ) {
        average_position[dim] = 0.0;
      }

      const IndexT num_nodes_per_element = mesh.numberOfNodesPerElement();
      for ( IndexT local_node = 0; local_node < num_nodes_per_element; ++local_node ) {
        const IndexT node = mesh.getGlobalNodeId( element, local_node );
        Point position;
        for ( int dim = 0; dim < SpatialDim; ++dim ) {
          position[dim] = mesh.getPosition()[dim][node];
          average_position[dim] += position[dim];
        }
        bounding_box.addPoint( position );
      }

      for ( int dim = 0; dim < SpatialDim; ++dim ) {
        average_position[dim] /= static_cast<RealT>( num_nodes_per_element );
      }

      RealT squared_nodal_radius = 0.0;
      for ( IndexT local_node = 0; local_node < num_nodes_per_element; ++local_node ) {
        const IndexT node = mesh.getGlobalNodeId( element, local_node );
        RealT squared_distance = 0.0;
        for ( int dim = 0; dim < SpatialDim; ++dim ) {
          const RealT distance = mesh.getPosition()[dim][node] - average_position[dim];
          squared_distance += distance * distance;
        }
        squared_nodal_radius = squared_distance > squared_nodal_radius ? squared_distance : squared_nodal_radius;
      }

      const RealT expansion = binning_proximity_scale * std::sqrt( squared_nodal_radius ) + residual_gap;
      bounding_box.expand( expansion );
      boxes[element] = bounding_box;
    } );
  }

 private:
  RealT binning_proximity_scale_;
  RealT residual_gap_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_BVHSEARCH_HPP_ */
