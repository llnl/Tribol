// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_COARSESEARCH_HPP_
#define SRC_TRIBOL_SEARCH_COARSESEARCH_HPP_

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

/**
 * @brief Runtime interface for spatial candidate searches
 *
 * Searches return spatial candidates without formulation-specific filtering. Each call takes current mesh views so
 * callers can reuse a search object after replacing mesh arrays.
 */
class CoarseSearch {
 public:
  virtual ~CoarseSearch() = default;

  /** @brief Return spatially proximate element pairs from the current mesh views */
  virtual ArrayT<InterfacePair> findCandidatePairs( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const = 0;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_COARSESEARCH_HPP_ */
