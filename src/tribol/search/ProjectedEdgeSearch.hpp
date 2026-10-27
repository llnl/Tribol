// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_PROJECTEDEDGESEARCH_HPP_
#define SRC_TRIBOL_SEARCH_PROJECTEDEDGESEARCH_HPP_

#include "tribol/search/RefinedSearch.hpp"
#include "tribol/geom/EdgeProjection.hpp"
#include "tribol/common/Atomics.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/search/InterfacePairChecks.hpp"

namespace tribol {

/**
 * @brief Refine two-dimensional linear-edge candidates using opposing normals and projected support
 *
 * Rejects exact self-pairs, optionally rejects shared-node pairs, and requires nondegenerate opposing edge normals.
 * Projection decides whether a pair contributes to its integration support. Its configuration is retained by value;
 * mesh coordinates are supplied at each call.
 * Geometry checks use the first two connectivity entries as edge endpoints; they do not account for curvature from
 * higher-order nodes. Shared-node rejection compares all finite-element node IDs and requires a common node-numbering
 * space. Candidates and both meshes must use the same allocator, accessible from the selected execution mode.
 * No face-data preprocessing is required. Temporary storage scales with the candidate count, not all possible pairs.
 *
 * @tparam Projection Value type providing mayContribute() for source and target endpoint coordinates.
 */
template <typename Projection>
class ProjectedEdgeSearch final : public RefinedSearch {
 public:
  /**
   * @brief Retain execution settings and a value copy of the projection.
   *
   * @param execution_mode Enabled backend used for pair checks and returned-pair packing.
   * @param projection Configured support test for source and target edges.
   * @param checks_shared_nodes Whether to reject pairs sharing finite-element nodes.
   */
  ProjectedEdgeSearch( ExecutionMode execution_mode, Projection projection, bool checks_shared_nodes )
      : execution_mode_( execution_mode ), projection_( projection ), checks_shared_nodes_( checks_shared_nodes )
  {
  }

  /**
   * @brief Return accepted pairs without changing candidate or mesh arrays.
   *
   * @param candidate_pairs Coarse candidate pairs in mesh1/mesh2 order.
   * @param mesh1 Current source mesh view.
   * @param mesh2 Current target mesh view.
   * @return Owned accepted pairs in the input allocator; their order is unspecified.
   */
  ArrayT<InterfacePair> filterCandidatePairs( const ArrayT<InterfacePair>& candidate_pairs, MeshData::Viewer mesh1,
                                              MeshData::Viewer mesh2 ) const override;

 private:
  const ExecutionMode execution_mode_;  ///< Backend for candidate checks and packing.
  const Projection projection_;         ///< Owned projection configuration; no mesh storage is retained.
  const bool checks_shared_nodes_;      ///< Whether shared-node pairs are rejected.
};

template <typename Projection>
ArrayT<InterfacePair> ProjectedEdgeSearch<Projection>::filterCandidatePairs(
    const ArrayT<InterfacePair>& candidate_pairs, MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  SLIC_ERROR_ROOT_IF( mesh1.spatialDimension() != 2 || mesh2.spatialDimension() != 2,
                      "Projected-edge search requires two-dimensional meshes." );
  SLIC_ASSERT_MSG(
      mesh1.getAllocatorId() == mesh2.getAllocatorId() && candidate_pairs.getAllocatorID() == mesh1.getAllocatorId(),
      "Refined-search meshes and candidates must use the same allocator." );
  const IndexT candidate_count = candidate_pairs.size();
  const int allocator_id = mesh1.getAllocatorId();
  ArrayT<bool> accepted( axom::ArrayOptions::Uninitialized{}, candidate_count, candidate_count, allocator_id );
  ArrayT<IndexT> accepted_count_data( { 0 }, allocator_id );

  const auto candidates = candidate_pairs.view();
  auto accepted_values = accepted.view();
  auto accepted_count = accepted_count_data.view();
  const bool checks_shared_nodes = checks_shared_nodes_;
  const Projection projection = projection_;

  // Save each geometric decision so compaction does not repeat the checks.
  forAllExec( execution_mode_, candidate_count,
              [mesh1, mesh2, candidates, accepted_values, accepted_count, checks_shared_nodes,
               projection] TRIBOL_HOST_DEVICE( IndexT candidate_index ) {
                const InterfacePair& pair = candidates[candidate_index];
                bool accepts_pair =
                    !InterfacePairChecks::isSameElementPair( mesh1, pair.element_id1_, mesh2, pair.element_id2_ );
                if ( accepts_pair && checks_shared_nodes ) {
                  accepts_pair =
                      !InterfacePairChecks::elementsShareNode( mesh1, pair.element_id1_, mesh2, pair.element_id2_ );
                }
                if ( accepts_pair ) {
                  accepts_pair = InterfacePairChecks::linearEdgeNormalsAreOpposing( mesh1, pair.element_id1_, mesh2,
                                                                                    pair.element_id2_ );
                }

                if ( accepts_pair ) {
                  MdArray<RealT, 2, 2> source, target;
                  for ( int endpoint = 0; endpoint < 2; ++endpoint ) {
                    const IndexT source_node = mesh1.getGlobalNodeId( pair.element_id1_, endpoint );
                    const IndexT target_node = mesh2.getGlobalNodeId( pair.element_id2_, endpoint );
                    for ( int component = 0; component < 2; ++component ) {
                      source( endpoint, component ) = mesh1.getPosition()[component][source_node];
                      target( endpoint, component ) = mesh2.getPosition()[component][target_node];
                    }
                  }
                  accepts_pair = projection.template mayContribute<RealT>( source.view(), target.view() );
                }

                accepted_values[candidate_index] = accepts_pair;
                if ( accepts_pair ) {
                  tribol::atomicInc( accepted_count.data() );
                }
              } );

  ArrayT<IndexT, 1, MemorySpace::Host> accepted_count_host( accepted_count_data );
  ArrayT<InterfacePair> accepted_pairs( axom::ArrayOptions::Uninitialized{}, accepted_count_host[0],
                                        accepted_count_host[0], allocator_id );

  accepted_count_data.fill( 0 );
  auto accepted_pairs_view = accepted_pairs.view();
  // Synchronous execution keeps candidates and scratch arrays alive until all accepted pairs have been copied.
  forAllExec(
      execution_mode_, candidate_count,
      [candidates, accepted_values, accepted_count, accepted_pairs_view] TRIBOL_HOST_DEVICE( IndexT candidate_index ) {
        if ( !accepted_values[candidate_index] ) {
          return;
        }
        const IndexT accepted_index = tribol::atomicInc( accepted_count.data() );
        accepted_pairs_view[accepted_index] = candidates[candidate_index];
      } );

  SLIC_INFO( "Projected-edge search accepted " << accepted_pairs.size() << " of " << candidate_count
                                               << " spatial candidate pairs." );
  return accepted_pairs;
}

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_PROJECTEDEDGESEARCH_HPP_ */
