// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_PROJECTEDEDGESEARCH_HPP_
#define SRC_TRIBOL_SEARCH_PROJECTEDEDGESEARCH_HPP_

#include "tribol/search/RefinedSearch.hpp"

namespace tribol {

/**
 * @brief Refine two-dimensional linear-edge candidates using opposing normals and projected support
 *
 * Rejects exact self-pairs, optionally rejects shared-node pairs, and requires nondegenerate opposing edge normals.
 * Mesh 1 is the integration edge; mesh 2 is projected along its normal onto mesh 1. The projected interval must have
 * positive overlap with mesh 1's support, expanded at each end by smoothing_fraction times its length.
 * Geometry checks use the first two connectivity entries as edge endpoints; they do not account for curvature from
 * higher-order nodes. Shared-node rejection compares all finite-element node IDs and requires a common node-numbering
 * space. Candidates and both meshes must use the same allocator, accessible from the selected execution mode.
 * No face-data preprocessing is required. Temporary storage scales with the candidate count, not all possible pairs.
 */
class ProjectedEdgeSearch final : public RefinedSearch {
 public:
  /** @brief Store execution mode, support expansion as an edge-length fraction, and the shared-node rejection policy */
  ProjectedEdgeSearch( ExecutionMode execution_mode, RealT smoothing_fraction, bool checks_shared_nodes )
      : execution_mode_( execution_mode ),
        smoothing_fraction_( smoothing_fraction ),
        checks_shared_nodes_( checks_shared_nodes )
  {
  }

  ArrayT<InterfacePair> filterCandidatePairs( const ArrayT<InterfacePair>& candidate_pairs, MeshData::Viewer mesh1,
                                              MeshData::Viewer mesh2 ) const override;

 private:
  const ExecutionMode execution_mode_;
  const RealT smoothing_fraction_;
  const bool checks_shared_nodes_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_PROJECTEDEDGESEARCH_HPP_ */
