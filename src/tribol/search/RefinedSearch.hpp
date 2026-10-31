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

/** @brief Runtime interface for applying method-specific checks to coarse spatial candidates */
class RefinedSearch {
 public:
  virtual ~RefinedSearch() = default;

  /**
   * @brief Filter current candidates and return those that pass the implementation's checks
   *
   * Returned pairs remain candidates; implementations are not required to compute or store overlap geometry.
   * Candidate and mesh arrays belong to the caller and remain unchanged. Implementations retain configuration,
   * not mesh views or search results. Returned pair order is unspecified; element IDs preserve the input mesh order.
   * Candidate element IDs must be valid for the corresponding meshes. Implementations finish all work using the input
   * arrays before returning, so callers may immediately replace those arrays or destroy the strategy.
   */
  virtual ArrayT<InterfacePair> filterCandidatePairs( const ArrayT<InterfacePair>& candidate_pairs,
                                                      MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const = 0;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_REFINEDSEARCH_HPP_ */
