// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/mesh/ProcessingUnit.hpp"

namespace tribol {

template <typename BinningT, typename MethodT>
ProcessingUnit<BinningT, MethodT>::ProcessingUnit( const MeshData& mesh1, const MeshData& mesh2,
                                                   const Parameters& params )
    : method_( mesh1, mesh2, params ), pair_finder_( mesh1, mesh2, params )
{
}

template <typename BinningT, typename MethodT>
auto ProcessingUnit<BinningT, MethodT>::findPairs()
{
  auto pair_candidates = pair_finder_.getPairCandidates();
  return method_.findPairs( pair_candidates );
}

template <typename BinningT, typename MethodT>
template <typename PairListT>
void ProcessingUnit<BinningT, MethodT>::applyPhysics( const PairListT& pair_list )
{
  method_.applyPhysics( pair_list );
}

template <typename BinningT, typename MethodT>
RealT ProcessingUnit<BinningT, MethodT>::computeTimestep()
{
  return method_.computeTimestep();
}

template <typename BinningT, typename MethodT>
void ProcessingUnit<BinningT, MethodT>::writeOutput()
{
  method_.writeOutput();
}

} /* namespace tribol */
