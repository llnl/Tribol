// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/search/ProjectedEdgeSearch.hpp"

#include "axom/slic.hpp"
#include "tribol/common/Atomics.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/search/InterfacePairChecks.hpp"

namespace tribol {

ArrayT<InterfacePair> ProjectedEdgeSearch::filterCandidatePairs( const ArrayT<InterfacePair>& candidate_pairs,
                                                                 MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
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
  const RealT smoothing_fraction = smoothing_fraction_;

  // Save each geometric decision so compaction does not repeat the checks.
  forAllExec( execution_mode_, candidate_count,
              [mesh1, mesh2, candidates, accepted_values, accepted_count, checks_shared_nodes,
               smoothing_fraction] TRIBOL_HOST_DEVICE( IndexT candidate_index ) {
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
                  const auto projection = InterfacePairChecks::computeLinearEdgeProjectionBounds(
                      mesh1, pair.element_id1_, mesh2, pair.element_id2_ );
                  accepts_pair = projection.valid && InterfacePairChecks::projectionIntervalOverlapsSmoothedSupport(
                                                         projection.bounds.view(), smoothing_fraction );
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
