// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Filter edge candidates by topology and orientation without calculating projected overlap.
 */

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
  ArrayT<bool> passes_checks( axom::ArrayOptions::Uninitialized{}, candidate_count, candidate_count, allocator_id );
  ArrayT<IndexT> retained_count_data( { 0 }, allocator_id );

  const auto candidates = candidate_pairs.view();
  auto passes_checks_values = passes_checks.view();
  auto retained_count = retained_count_data.view();
  const bool checks_shared_nodes = checks_shared_nodes_;

  // Save each result so returning retained candidates does not repeat the checks.
  forAllExec( execution_mode_, candidate_count,
              [mesh1, mesh2, candidates, passes_checks_values, retained_count,
               checks_shared_nodes] TRIBOL_HOST_DEVICE( IndexT candidate_index ) {
                const InterfacePair& pair = candidates[candidate_index];
                bool pair_passes_checks =
                    !InterfacePairChecks::isSameElementPair( mesh1, pair.element_id1_, mesh2, pair.element_id2_ );
                if ( pair_passes_checks && checks_shared_nodes ) {
                  pair_passes_checks =
                      !InterfacePairChecks::elementsShareNode( mesh1, pair.element_id1_, mesh2, pair.element_id2_ );
                }
                if ( pair_passes_checks ) {
                  pair_passes_checks = InterfacePairChecks::linearEdgeNormalsAreOpposing( mesh1, pair.element_id1_,
                                                                                          mesh2, pair.element_id2_ );
                }

                passes_checks_values[candidate_index] = pair_passes_checks;
                if ( pair_passes_checks ) {
                  tribol::atomicInc( retained_count.data() );
                }
              } );

  ArrayT<IndexT, 1, MemorySpace::Host> retained_count_host( retained_count_data );
  ArrayT<InterfacePair> retained_pairs( axom::ArrayOptions::Uninitialized{}, retained_count_host[0],
                                        retained_count_host[0], allocator_id );

  retained_count_data.fill( 0 );
  auto retained_pairs_view = retained_pairs.view();
  // Synchronous execution keeps candidates and scratch arrays alive until all retained pairs have been copied.
  forAllExec( execution_mode_, candidate_count,
              [candidates, passes_checks_values, retained_count,
               retained_pairs_view] TRIBOL_HOST_DEVICE( IndexT candidate_index ) {
                if ( !passes_checks_values[candidate_index] ) {
                  return;
                }
                const IndexT retained_index = tribol::atomicInc( retained_count.data() );
                retained_pairs_view[retained_index] = candidates[candidate_index];
              } );

  SLIC_INFO( "Projected-edge search retained " << retained_pairs.size() << " of " << candidate_count
                                               << " spatial candidate pairs." );
  return retained_pairs;
}

}  // namespace tribol
