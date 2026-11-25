// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Refine three-dimensional surface-element candidates using topology and opposing face normals.
 */
#ifndef SRC_TRIBOL_SEARCH_PROJECTEDFACESEARCH_HPP_
#define SRC_TRIBOL_SEARCH_PROJECTEDFACESEARCH_HPP_

#include "tribol/search/RefinedSearch.hpp"

namespace tribol {

/**
 * @brief Refine three-dimensional surface-element candidates using topology and opposing normals.
 *
 * Rejects exact self-pairs, optionally rejects shared-node pairs, and requires nondegenerate opposing face normals for
 * linear triangles and quadrilaterals. Projected overlap is left to quadrature generation. Higher-order faces defer
 * orientation and overlap checks to subdivided quadrature because vertex planes can disagree with local curved
 * geometry.
 */
class ProjectedFaceSearch final : public RefinedSearch {
 public:
  /**
   * @brief Retain execution settings and the shared-node rejection setting.
   * @param execution_mode Enabled backend used for pair checks and returned-pair packing.
   * @param checks_shared_nodes Whether to reject pairs sharing finite-element nodes.
   */
  ProjectedFaceSearch( ExecutionMode execution_mode, bool checks_shared_nodes )
      : execution_mode_( execution_mode ), checks_shared_nodes_( checks_shared_nodes )
  {
  }

  /**
   * @brief Return candidates that pass topology and orientation checks without changing the inputs.
   * @param candidate_pairs Coarse candidate pairs in mesh1/mesh2 order.
   * @param mesh1 Current first mesh view.
   * @param mesh2 Current second mesh view.
   * @return Owned retained candidates in the input allocator; their order is unspecified.
   */
  ArrayT<InterfacePair> filterCandidatePairs( const ArrayT<InterfacePair>& candidate_pairs, MeshData::Viewer mesh1,
                                              MeshData::Viewer mesh2 ) const override;

 private:
  const ExecutionMode execution_mode_;  ///< Backend for candidate checks and packing.
  const bool checks_shared_nodes_;      ///< Whether shared-node pairs are rejected.
};

}  // namespace tribol

#endif  // SRC_TRIBOL_SEARCH_PROJECTEDFACESEARCH_HPP_
