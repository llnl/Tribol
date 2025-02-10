// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_PARTITION_PARTITIONENTITY_HPP_
#define SRC_REDECOMP_PARTITION_PARTITIONENTITY_HPP_

#include <vector>

#include "mfem.hpp"

#include "redecomp/common/TypeDefs.hpp"

namespace redecomp {

/**
 * @brief PartitionEntity interface class
 *
 * This class establishes an interface for providing a list of coordinates given
 * a vector of ParMeshes.  Derived classes will determine what the coordinates
 * represent.  For instance, the PartitionElements class returns approximate
 * element centroids as its list of coordinates.
 *
 * @tparam NDIMS number of dimensions
 */
template <int NDIMS>
class PartitionEntity {
 public:
  /**
   * @brief Returns lists of entity coordinates (points) from the par_meshes, sorted by mesh
   *
   * @param par_meshes Input meshes
   * @return Vector of array of points from each entity on each mesh
   */
  virtual std::vector<axom::Array<Point<NDIMS>>> EntityCoordinates(
      const std::vector<const mfem::ParMesh*>& par_meshes ) const = 0;

  /**
   * @brief Destroy the PartitionEntity object
   */
  virtual ~PartitionEntity() = default;
};

using PartitionEntity2D = PartitionEntity<2>;
using PartitionEntity3D = PartitionEntity<3>;

}  // end namespace redecomp

#endif /* SRC_REDECOMP_PARTITION_PARTITIONENTITY_HPP_ */
