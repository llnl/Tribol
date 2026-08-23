// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_INTERFACE_PAIR_FINDER_HPP_
#define SRC_TRIBOL_SEARCH_INTERFACE_PAIR_FINDER_HPP_

#include "tribol/common/Parameters.hpp"
#include "tribol/common/ExecModel.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

namespace tribol {

/**
 * @brief Configures and runs coarse contact-pair searches between two meshes
 *
 * This compatibility adapter maps the legacy BinningMethod enum to a public coarse-search policy. New contact
 * formulations may compose directly with CartesianProductSearch, GridSearch, BvhSearch, or another compatible search
 * type to retain its concrete pair-range type.
 *
 * This class stores only search configuration. It does not own or retain the meshes supplied to findInterfacePairs().
 */
class InterfacePairFinder {
 public:
  /**
   * @brief Construct a coarse contact-pair finder
   *
   * @param [in] binning_method Search strategy used to generate candidate pairs.
   * @param [in] execution_mode Execution mode used by the search strategy.
   * @param [in] allocator_id Allocator used for search work arrays and explicitly stored result pairs.
   * @param [in] proximity_scale Element-size multiplier used to inflate search bounds.
   *
   * @note Device execution with @c BINNING_GRID is normalized to @c BINNING_BVH because grid search is not supported on
   * devices.
   */
  InterfacePairFinder( BinningMethod binning_method, ExecutionMode execution_mode, int allocator_id,
                       RealT proximity_scale );

  /**
   * @brief Find coarse candidate pairs between the provided meshes
   *
   * The meshes are used only for this search and are not retained. Grid and BVH searches return owning explicit pair
   * storage allocated with the configured allocator. Cartesian-product search returns a lazy view and does not
   * explicitly store the generated pairs.
   *
   * @param [in] mesh1 First contact mesh. Spatial searches index its elements.
   * @param [in] mesh2 Second contact mesh. Spatial searches query its elements against the first mesh.
   * @return Owning explicit pair storage or a lazy Cartesian-product pair view.
   */
  ContactPairRange findInterfacePairs( MeshData& mesh1, MeshData& mesh2 ) const;

  /**
   * @brief Get the effective coarse-search strategy
   *
   * @return Configured binning method after any device compatibility normalization performed by the constructor.
   */
  BinningMethod getBinningMethod() const { return binning_method_; }

 private:
  /** Coarse-search strategy used to generate candidate pairs. */
  BinningMethod binning_method_;

  /** Execution mode used by the selected search strategy. */
  ExecutionMode execution_mode_;

  /** Allocator for search work arrays and explicitly stored result pairs. */
  int allocator_id_;

  /** Element-size multiplier used to inflate coarse-search bounds. */
  RealT proximity_scale_;
};

}  // end namespace tribol

#endif /* SRC_TRIBOL_SEARCH_INTERFACE_PAIR_FINDER_HPP_ */
