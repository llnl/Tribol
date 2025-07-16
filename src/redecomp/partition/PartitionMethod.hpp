// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_PARTITION_PARTITIONMETHOD_HPP_
#define SRC_REDECOMP_PARTITION_PARTITIONMETHOD_HPP_

#include "redecomp/common/TypeDefs.hpp"

namespace redecomp {

/**
 * @brief PartitionMethod base class
 *
 * The PartitionMethod class is designed to take a list of coordinates on each
 * rank and produce a re-balancing of the coordinates over the MPI ranks based
 * on the method in the derived class.  For instance, RCB re-balances the
 * coordinates using the recursive coordinate bisection technique.  The list of
 * coordinates can be obtained from a PartitionEntity object.
 *
 * @tparam NDIMS number of dimensions
 */
template <int NDIMS>
class PartitionMethod {
 public:
  /**
   * @brief Returns a list of entity ids on each rank/subdomain determined by
   * the partitioning method
   *
   * @param n_parts Number of subdomains to cut the list of coords into
   * @param coords_by_mesh List of on-rank points (one point-per-entity, sorted
   * by entity id) to subdivide sorted by mesh
   * @param ghost_size Sets length to include entities as ghost on each
   * subdomain
   * @return List of entity ids and ghost information on each subdomain sorted
   * by mesh
   */
  virtual std::vector<EntityIndexByRank> generatePartitioning(
      int n_parts, const std::vector<axom::Array<Point<NDIMS>>>& coords_by_mesh, double ghost_size ) const = 0;

  /**
   * @brief Returns the MPIUtility
   *
   * @return MPIUtility reference
   */
  const MPIUtility& getMPIUtility() const;

  /**
   * @brief Destroy the PartitionMethod object
   */
  virtual ~PartitionMethod() = default;

 protected:
  /**
   * @brief Construct a new PartitionMethod object
   *
   * @param comm MPI_Comm to used to build MPIUtility
   */
  PartitionMethod( const MPI_Comm& comm );

 private:
  /**
   * @brief MPIUtility used to hold MPI communication patterns used in redecomp
   */
  const MPIUtility mpi_;
};

using PartitionMethod2D = PartitionMethod<2>;
using PartitionMethod3D = PartitionMethod<3>;

}  // end namespace redecomp

#endif /* SRC_REDECOMP_PARTITION_PARTITIONMETHOD_HPP_ */
