// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_PARTITION_PARTITIONELEMENTS_HPP_
#define SRC_REDECOMP_PARTITION_PARTITIONELEMENTS_HPP_

#include "redecomp/partition/PartitionEntity.hpp"

namespace redecomp {

/**
 * @brief PartitionEntity class for elements
 *
 * @tparam NDIMS number of dimensions
 */
template <int NDIMS>
class PartitionElements : public PartitionEntity<NDIMS> {
 public:
  /**
   * @brief Returns lists of entity coordinates (points) from the par_meshes, sorted by mesh
   *
   * @param par_meshes Input meshes
   * @return Vector of array of points from each entity on each mesh
   */
  std::vector<axom::Array<Point<NDIMS>>> EntityCoordinates(
      const std::vector<const mfem::ParMesh*>& par_meshes ) const override;
};

using PartitionElements2D = PartitionElements<2>;
using PartitionElements3D = PartitionElements<3>;

}  // end namespace redecomp

#endif /* SRC_REDECOMP_PARTITION_PARTITIONELEMENTS_HPP_ */
