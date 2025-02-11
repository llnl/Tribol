// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_PARTITION_PARTITIONER_HPP_
#define SRC_REDECOMP_PARTITION_PARTITIONER_HPP_

#include <vector>

#include "mfem.hpp"

#include "redecomp/common/TypeDefs.hpp"

namespace redecomp {

/**
 * @brief Partitioner interface class
 */
class Partitioner {
 public:
  /**
   * @brief Partitions entities in all par_meshes into n_parts pieces
   *
   * @param n_parts Number of subdomains to cut par_mesh entities into
   * @param par_meshes Original meshes
   * @param ghost_size Entities within ghost_size distance from the edge of each subdomain are included in the subdomain
   * as a ghost entity
   * @return vector of EntityIndexByRank; lists of entities and ghost entities sorted by each subdomain
   */
  virtual std::vector<EntityIndexByRank> generatePartitioning( int n_parts,
                                                               const std::vector<const mfem::ParMesh*>& par_meshes,
                                                               double ghost_size ) const = 0;

  /**
   * @brief Returns the MPIUtility associated with the Partitioner
   *
   * @return MPIUtility pointer
   */
  virtual const MPIUtility& getMPIUtility() const = 0;

  /**
   * @brief Destroy the Partitioner object
   */
  virtual ~Partitioner() = default;
};

template <int NDIMS>
class PartitionEntity;

template <int NDIMS>
class PartitionMethod;

/**
 * @brief Dimension-specific partitioner class
 *
 * @tparam NDIMS number of dimensions
 */
template <int NDIMS>
class PartitionerByDim : public Partitioner {
 public:
  /**
   * @brief Construct a new PartitionerByDim object
   *
   * @param partition_entity Pointer to PartitionEntity member
   * @param partition_method Pointer to a PartitionMethod member
   */
  PartitionerByDim( std::unique_ptr<const PartitionEntity<NDIMS>> partition_entity,
                    std::unique_ptr<const PartitionMethod<NDIMS>> partition_method );

  /**
   * @brief Returns the MPIUtility associated with the Partitioner
   *
   * @return MPIUtility pointer
   */
  const MPIUtility& getMPIUtility() const override;

  /**
   * @brief Returns a list of entity ids on each rank/subdomain
   *
   * @param n_parts Number of subdomains to cut par_mesh entities into
   * @param par_meshes Original meshes
   * @param ghost_size Entities within ghost_size distance from the edge of each
   * subdomain are included in the subdomain as a ghost entity
   * @return vector of EntityIndexByRank; lists of entities and ghost entities
   * sorted by each subdomain
   */
  std::vector<EntityIndexByRank> generatePartitioning( int n_parts, const std::vector<const mfem::ParMesh*>& par_meshes,
                                                       double ghost_size ) const override;

  /**
   * @brief Get the PartitionEntity object
   *
   * @return PartitionEntity pointer
   */
  const PartitionEntity<NDIMS>* getPartitionEntity() const;

  /**
   * @brief Get the PartitionMethod object
   *
   * @return PartitionMethod pointer
   */
  const PartitionMethod<NDIMS>* getPartitionMethod() const;

 private:
  /**
   * @brief Owned pointer to a PartitionEntity object
   *
   * The PartitionEntity object determines the list of coordinates that are
   * returned given an mfem::ParMesh.  For instance, the PartitionElements
   * derived class returns approximate element centroids as its list of
   * coordinates.
   *
   */
  const std::unique_ptr<const PartitionEntity<NDIMS>> partition_entity_;

  /**
   * @brief Owned pointer to a PartitionMethod object
   *
   * The PartitionMethod object takes a list of coordinates on each rank then
   * re-balances them based on the method specified in the derived class.  The
   * RCB class, for instance, re-balances points using a recursive coordinate
   * bisection algorithm.
   *
   */
  const std::unique_ptr<const PartitionMethod<NDIMS>> partition_method_;
};

using Partitioner2D = PartitionerByDim<2>;
using Partitioner3D = PartitionerByDim<3>;

}  // end namespace redecomp

#endif /* SRC_REDECOMP_PARTITION_PARTITIONER_HPP_ */
