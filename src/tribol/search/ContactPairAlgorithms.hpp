// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_CONTACT_PAIR_ALGORITHMS_HPP_
#define SRC_TRIBOL_SEARCH_CONTACT_PAIR_ALGORITHMS_HPP_

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Atomics.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

namespace tribol {

/**
 * @brief Compact accepted contact pairs without candidate-sized temporary storage
 *
 * The predicate is evaluated once to count accepted pairs and again to scatter
 * them into the output array. Parallel execution does not preserve pair order.
 * Existing output capacity is reused when its allocator matches allocator_id.
 *
 * @tparam PairView Random-access candidate-pair view
 * @tparam Predicate Device-copyable predicate accepting an ElementPair
 * @tparam OutputPair Pair type constructible from two element indices
 * @param [in] candidate_pairs Candidate pairs to filter
 * @param [in] predicate Returns true for accepted candidate pairs
 * @param [in] execution_mode Execution mode for counting and scattering
 * @param [in] allocator_id Allocator used for counters and output storage
 * @param [out] output_pairs Compact array of accepted pairs
 */
template <typename PairView, typename Predicate, typename OutputPair>
void compactContactPairs( PairView candidate_pairs, Predicate predicate, ExecutionMode execution_mode, int allocator_id,
                          ArrayT<OutputPair>& output_pairs )
{
  if ( output_pairs.getAllocatorID() != allocator_id ) {
    output_pairs = ArrayT<OutputPair>( 0, 0, allocator_id );
  }

  ArrayT<IndexT> count_data( 1, 1, allocator_id );
  count_data.fill( 0 );
  auto count = count_data.view();

  forAllExec<false>( execution_mode, candidate_pairs.size(),
                     [candidate_pairs, predicate, count] TRIBOL_HOST_DEVICE( IndexT i ) mutable {
                       if ( predicate( candidate_pairs[i] ) ) {
                         tribol::atomicInc( count.data() );
                       }
                     } );

  ArrayT<IndexT, 1, MemorySpace::Host> host_count( count_data );
  output_pairs.resize( host_count[0] );
  count_data.fill( 0 );
  auto output = output_pairs.view();

  forAllExec<false>( execution_mode, candidate_pairs.size(),
                     [candidate_pairs, predicate, count, output] TRIBOL_HOST_DEVICE( IndexT i ) mutable {
                       const auto pair = candidate_pairs[i];
                       if ( !predicate( pair ) ) {
                         return;
                       }
                       const auto output_index = tribol::atomicInc( count.data() );
                       output[output_index] = OutputPair( pair.element_id1, pair.element_id2 );
                     } );
}

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_CONTACT_PAIR_ALGORITHMS_HPP_ */
