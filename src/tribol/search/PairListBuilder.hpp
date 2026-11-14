// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Composition of coarse and refined searches for current contact pairs. */
#ifndef SRC_TRIBOL_SEARCH_PAIRLISTBUILDER_HPP_
#define SRC_TRIBOL_SEARCH_PAIRLISTBUILDER_HPP_

#include <memory>
#include <stdexcept>
#include "tribol/search/CoarseSearch.hpp"
#include "tribol/search/RefinedSearch.hpp"

namespace tribol {

/** @brief Own search strategies and build a fresh pair list from current mesh views. */
class PairListBuilder {
 public:
  /**
   * @brief Take ownership of both search stages.
   * @param coarse_search Spatial candidate search; must be non-null.
   * @param refined_search Method-independent edge checks; must be non-null.
   */
  PairListBuilder( std::unique_ptr<CoarseSearch> coarse_search, std::unique_ptr<RefinedSearch> refined_search )
      : coarse_search_( std::move( coarse_search ) ), refined_search_( std::move( refined_search ) )
  {
    if ( !coarse_search_ || !refined_search_ ) {
      throw std::invalid_argument( "PairListBuilder requires both search stages." );
    }
  }

  /**
   * @brief Find and filter pairs using current coordinates.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned pairs in execution-accessible memory.
   */
  ArrayT<InterfacePair> build( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
  {
    auto candidates = coarse_search_->findCandidatePairs( mesh1, mesh2 );
    return refined_search_->filterCandidatePairs( candidates, mesh1, mesh2 );
  }

 private:
  std::unique_ptr<CoarseSearch> coarse_search_;    ///< Candidate search configuration.
  std::unique_ptr<RefinedSearch> refined_search_;  ///< Pair acceptance configuration.
};

}  // namespace tribol

#endif  // SRC_TRIBOL_SEARCH_PAIRLISTBUILDER_HPP_
