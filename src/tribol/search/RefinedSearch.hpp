// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_REFINEDSEARCH_HPP_
#define SRC_TRIBOL_SEARCH_REFINEDSEARCH_HPP_

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

/** @brief Runtime interface for refining coarse spatial candidates into method-approved element pairs */
class RefinedSearch {
 public:
  virtual ~RefinedSearch() = default;

  /**
   * @brief Filter current candidates and return an owned array of accepted pairs
   *
   * Candidates and mesh arrays are borrowed for this call and remain unchanged. Implementations retain configuration,
   * not mesh views or search results. Returned pair order is unspecified; element IDs preserve the input mesh order.
   * Candidate element IDs must be valid for the corresponding meshes. Implementations finish all work using borrowed
   * data before returning, so callers may immediately replace that data or destroy the strategy.
   */
  virtual ArrayT<InterfacePair> filterCandidatePairs( const ArrayT<InterfacePair>& candidate_pairs,
                                                      MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const = 0;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_REFINEDSEARCH_HPP_ */
