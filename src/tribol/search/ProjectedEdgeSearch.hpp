// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_PROJECTEDEDGESEARCH_HPP_
#define SRC_TRIBOL_SEARCH_PROJECTEDEDGESEARCH_HPP_

#include "tribol/search/RefinedSearch.hpp"

namespace tribol {

/**
 * @brief Refine two-dimensional edge candidates using topology and opposing normals
 *
 * Rejects exact self-pairs, optionally rejects shared-node pairs, and requires nondegenerate opposing edge normals.
 * Projected overlap is left to quadrature generation. Mesh coordinates are supplied at each call.
 * Geometry checks use the first two connectivity entries as edge endpoints; they do not account for curvature from
 * higher-order nodes. Shared-node rejection compares all finite-element node IDs and requires a common node-numbering
 * space. Candidates and both meshes must use the same allocator, accessible from the selected execution mode.
 * No face-data preprocessing is required. Temporary storage scales with the candidate count, not all possible pairs.
 */
class ProjectedEdgeSearch final : public RefinedSearch {
 public:
  /**
   * @brief Retain execution settings and the shared-node rejection setting.
   *
   * @param execution_mode Enabled backend used for pair checks and returned-pair packing.
   * @param checks_shared_nodes Whether to reject pairs sharing finite-element nodes.
   */
  ProjectedEdgeSearch( ExecutionMode execution_mode, bool checks_shared_nodes )
      : execution_mode_( execution_mode ), checks_shared_nodes_( checks_shared_nodes )
  {
  }

  /**
   * @brief Return candidates that pass topology and orientation checks without changing the inputs.
   *
   * @param candidate_pairs Coarse candidate pairs in mesh1/mesh2 order.
   * @param mesh1 Current source mesh view.
   * @param mesh2 Current target mesh view.
   * @return Owned retained candidates in the input allocator; their order is unspecified.
   */
  ArrayT<InterfacePair> filterCandidatePairs( const ArrayT<InterfacePair>& candidate_pairs, MeshData::Viewer mesh1,
                                              MeshData::Viewer mesh2 ) const override;

 private:
  const ExecutionMode execution_mode_;  ///< Backend for candidate checks and packing.
  const bool checks_shared_nodes_;      ///< Whether shared-node pairs are rejected.
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_PROJECTEDEDGESEARCH_HPP_ */
