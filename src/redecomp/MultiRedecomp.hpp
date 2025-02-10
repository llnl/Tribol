// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_MULTIREDECOMP_HPP_
#define SRC_REDECOMP_MULTIREDECOMP_HPP_

#include <vector>

#include "mfem.hpp"

#include "redecomp/RedecompMesh.hpp"

namespace redecomp {

class MultiRedecomp {
 public:
  /**
   * @brief List of partitioning methods verified to work with MultiRedecomp
   */
  enum PartitionType
  {
    RCB
  };

  /**
   * @brief Construct a new MultiRedecomp object
   *
   * @note This constructor builds the Partitioner object based on the method
   * passed. If no method is passed, a RCB Partitioner is constructed.
   *
   * @param method The method of redecomposition
   * @param ghost_len_multiplier Multiplier for the ghost element layer (base
   * value set by the approximate largest element size)
   */
  MultiRedecomp( int dim, MPI_Comm comm, PartitionType method = RCB, double ghost_len_multiplier = 1.25 );

  /**
   * @brief Construct a new MultiRedecomp object
   *
   * @note This constructor requires the Partitioner object passed directly to
   * it. This can be used to customize the Partitioner used (for example, with
   * non-default options with RCB or for a user-defined Partitioner).
   *
   * @param partitioner Partitioning object used to define redecomposition
   * @param ghost_len_multiplier Multiplier for the ghost element layer (base
   * value set by the approximate largest element size)
   */
  MultiRedecomp( std::unique_ptr<const Partitioner> partitioner, double ghost_len_multiplier = 1.25 );

  /**
   * @brief Get a vector of Redecomp meshes associated with MultiRedecomp
   *
   * @note This method can be used to move RedecompMeshes out of the
   * MultiRedecomp object. Once the RedecompMeshes are moved, they can no longer
   * be accessed in the MultiRedecomp object. If you wish to access a
   * RedecompMesh without taking ownership, the getRedecompMesh() method can be
   * used.
   *
   * @return Vector of Redecomp unique pointers
   */
  std::vector<std::unique_ptr<RedecompMesh>> createRedecompMeshes( const std::vector<const mfem::ParMesh*>& parents );

 private:
  /**
   * @brief Partitioning method
   */
  std::unique_ptr<const Partitioner> partitioner_;

  /**
   * @brief Multiplier of max element length to include in ghost regions of the
   * RedecompMeshes
   */
  double ghost_len_multiplier_;
};

}  // namespace redecomp

#endif /* SRC_REDECOMP_MULTIREDECOMP_HPP_ */
