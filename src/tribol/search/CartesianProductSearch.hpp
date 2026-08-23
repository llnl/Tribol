// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_CARTESIAN_PRODUCT_SEARCH_HPP_
#define SRC_TRIBOL_SEARCH_CARTESIAN_PRODUCT_SEARCH_HPP_

#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

/**
 * @brief Generates every possible element pair without explicit storage
 *
 * The returned range computes each pair from its index. Same-mesh searches
 * return one ordering of every pair of distinct elements.
 */
class CartesianProductSearch {
 public:
  /** Type returned by findPairs(). */
  using PairRange = CartesianPairView;

  /**
   * @brief Generate a lazy Cartesian product of two meshes
   *
   * @param [in] mesh1 Mesh providing the first element index in each pair.
   * @param [in] mesh2 Mesh providing the second element index in each pair.
   * @return Device-copyable random-access view of the Cartesian product.
   */
  PairRange findPairs( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) const
  {
    return PairRange( mesh1.numberOfElements(), mesh2.numberOfElements(), mesh1.meshId() == mesh2.meshId() );
  }
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_CARTESIAN_PRODUCT_SEARCH_HPP_ */
