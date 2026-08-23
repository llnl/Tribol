// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_BVH_SEARCH_HPP_
#define SRC_TRIBOL_SEARCH_BVH_SEARCH_HPP_

#include "axom/core/execution/for_all.hpp"
#include "axom/primal.hpp"
#include "axom/spin.hpp"

#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/utils/Algorithm.hpp"

#include <utility>

namespace tribol {

/**
 * @brief Finds coarse candidate pairs with an Axom bounding-volume hierarchy
 *
 * The search builds a BVH over expanded first-mesh element boxes and queries
 * it with expanded second-mesh element boxes.
 *
 * @tparam Dimension Spatial dimension of the mesh coordinates.
 * @tparam ExecSpace Axom execution space used to build and query the BVH.
 */
template <int Dimension, typename ExecSpace>
class BvhSearch {
  static_assert( Dimension == 2 || Dimension == 3, "BvhSearch supports only two- and three-dimensional meshes." );
  static_assert( axom::execution_space<ExecSpace>::valid(), "BvhSearch requires a valid Axom execution space." );
  static_assert( !axom::execution_space<ExecSpace>::async(),
                 "BvhSearch requires synchronous execution because search work arrays are local." );

 public:
  /** Type returned by findPairs(). */
  using PairRange = ArrayT<ElementPair>;

  /**
   * @brief Construct a BVH-search policy
   *
   * @param [in] proximity_scale Element-size multiplier used to expand element
   * boxes along their face normals.
   * @param [in] allocator_id Allocator used for search work arrays and result
   * pairs.
   */
  BvhSearch( RealT proximity_scale, int allocator_id )
      : allocator_id_( allocator_id ), proximity_scale_( proximity_scale )
  {
  }

  /**
   * @brief Build and query a BVH for two meshes
   *
   * @param [in] mesh1 Mesh whose element boxes populate the BVH.
   * @param [in] mesh2 Mesh whose element boxes query the BVH.
   * @return Candidate pairs stored with the configured allocator.
   */
  PairRange findPairs( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) const
  {
    if ( mesh1.numberOfElements() == 0 || mesh2.numberOfElements() == 0 ) {
      return PairRange( 0, 0, allocator_id_ );
    }

    ArrayT<BoxT> boxes1( axom::ArrayOptions::Uninitialized{}, mesh1.numberOfElements(), mesh1.numberOfElements(),
                         allocator_id_ );
    ArrayT<BoxT> boxes2( axom::ArrayOptions::Uninitialized{}, mesh2.numberOfElements(), mesh2.numberOfElements(),
                         allocator_id_ );
    buildMeshBBoxes( boxes1, mesh1, proximity_scale_ );
    buildMeshBBoxes( boxes2, mesh2, proximity_scale_ );

    BVHT bvh;
    bvh.setAllocatorID( allocator_id_ );
    bvh.initialize( boxes1.view(), boxes1.size() );

    ArrayT<IndexT> candidates( axom::ArrayOptions::Uninitialized{}, 0, 0, allocator_id_ );
    ArrayT<IndexT> offsets( axom::ArrayOptions::Uninitialized{}, mesh2.numberOfElements(), mesh2.numberOfElements(),
                            allocator_id_ );
    ArrayT<IndexT> counts( axom::ArrayOptions::Uninitialized{}, mesh2.numberOfElements(), mesh2.numberOfElements(),
                           allocator_id_ );
    auto counts_view = counts.view();
    auto offsets_view = offsets.view();
    bvh.findBoundingBoxes( offsets_view, counts_view, candidates, mesh2.numberOfElements(), boxes2.view() );

    auto candidates_view = candidates.view();
    PairRange pairs( candidates.size(), candidates.size(), allocator_id_ );
    auto pairs_view = pairs.view();
    axom::for_all<ExecSpace>( candidates.size(),
                              [candidates_view, offsets_view, counts_view, pairs_view] TRIBOL_HOST_DEVICE( IndexT i ) {
                                auto mesh1_element = candidates_view[i];
                                auto mesh2_element = algorithm::binarySearch( offsets_view, counts_view, i );
                                pairs_view[i] = ElementPair( mesh1_element, mesh2_element );
                              } );
    return pairs;
  }

 private:
  /** Axom BVH type used by this search policy. */
  using BVHT = axom::spin::BVH<Dimension, ExecSpace, RealT>;

  /** Bounding-box type expected by the BVH. */
  using BoxT = typename BVHT::BoxType;

  /** Point type used to construct element bounding boxes. */
  using PointT = axom::primal::Point<RealT, Dimension>;

  /** Ray type used for face-normal box expansion. */
  using RayT = axom::primal::Ray<RealT, Dimension>;

  /** Vector type used for face normals. */
  using VectorT = axom::primal::Vector<RealT, Dimension>;

  /**
   * @brief Build proximity-expanded element boxes for one mesh
   *
   * @param [out] boxes Preallocated output array with one box per element.
   * @param [in] mesh Mesh whose element boxes are built.
   * @param [in] binning_proximity Multiplier applied to each element's face
   * radius to obtain the normal expansion distance.
   */
  void buildMeshBBoxes( ArrayT<BoxT>& boxes, const MeshData::Viewer& mesh, RealT binning_proximity ) const
  {
    auto boxes_view = boxes.view();
    axom::for_all<ExecSpace>( mesh.numberOfElements(),
                              [mesh, boxes_view, binning_proximity] TRIBOL_HOST_DEVICE( IndexT element_id ) {
                                BoxT box;
                                auto num_nodes_per_element = mesh.numberOfNodesPerElement();
                                for ( IndexT node = 0; node < num_nodes_per_element; ++node ) {
                                  IndexT node_id = mesh.getGlobalNodeId( element_id, node );
                                  PointT position;
                                  for ( int dimension = 0; dimension < Dimension; ++dimension ) {
                                    position[dimension] = mesh.getPosition()[dimension][node_id];
                                  }
                                  box.addPoint( position );
                                }

                                RealT normal_data[3];
                                mesh.getFaceNormal( element_id, normal_data );
                                VectorT face_normal( normal_data );
                                RealT face_radius = mesh.getFaceRadius()[element_id];
                                expandBBoxNormal( box, face_normal, binning_proximity * face_radius );
                                boxes_view[element_id] = std::move( box );
                              } );
  }

  /**
   * @brief Expand a box in both directions along a face normal
   *
   * @param [in,out] bbox Box to expand.
   * @param [in] face_normal Unit face normal defining the expansion direction.
   * @param [in] face_radius Distance added in each normal direction.
   */
  TRIBOL_HOST_DEVICE static void expandBBoxNormal( BoxT& bbox, const VectorT& face_normal, RealT face_radius )
  {
    PointT centroid = bbox.getCentroid();
    RayT outward_ray( centroid, face_normal );
    VectorT inward_normal( face_normal );
    inward_normal *= -1.0;
    RayT inward_ray( centroid, inward_normal );
    bbox.addPoint( outward_ray.at( face_radius ) );
    bbox.addPoint( inward_ray.at( face_radius ) );
  }

  /** Allocator for search work arrays and explicitly stored result pairs. */
  int allocator_id_;

  /** Element-size multiplier used for face-normal box expansion. */
  RealT proximity_scale_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_BVH_SEARCH_HPP_ */
