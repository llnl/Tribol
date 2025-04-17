// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_PARTITION_RCB_HPP_
#define SRC_REDECOMP_PARTITION_RCB_HPP_

#include <utility>

#include "axom/core.hpp"
#include "axom/primal.hpp"

#include "redecomp/partition/PartitionMethod.hpp"
#include "redecomp/utils/BisecTree.hpp"

namespace redecomp {

/**
 * @brief Stores information about the partition in BisecTree
 *
 * @tparam NDIMS number of dimensions
 */
template <int NDIMS>
struct RCBInfo {
  /**
   * @brief Desired fraction of entities in the partition
   */
  double desired_frac_;

  /**
   * @brief Actual fraction of entities in the partition
   */
  double actual_frac_;

  /**
   * @brief Bounding box defining the bounds of the partition
   */
  BoundingBox<NDIMS> bbox_;

  /**
   * @brief Bounding box defining the bounds of the partition including ghost regions
   */
  BoundingBox<NDIMS> ghost_bbox_;

  /**
   * @brief List of bounding boxes that are adjacent to this bounding box
   */
  axom::Array<int> neighbor_bboxes_;
};

/**
 * @brief Recursive coordinate bisection implementation
 *
 * @tparam NDIMS number of dimensions
 *
 * Recursive coordinate bisection creates a suitable subdivision of the domain
 * along the coordinate axes.  Cuts which minimize the length of the bisection
 * axis are preferred, with the assumption that these cuts reduce the number of
 * entities at the interface and therefore reduce the amount of inter-processor
 * communication.  Cut location iterations are chosen based on the bisection
 * method, which is well-suited since it reduces communication and relatively
 * few iterations are needed given a sufficiently large max_out_of_balance_.
 * See Hendrickson and Devine (2000), Comput Methods Appl Mech Eng.
 */
template <int NDIMS>
class RCB : public PartitionMethod<NDIMS> {
 public:
  /**
   * @brief Construct a new RCB object
   *
   * @param comm MPI_Comm for the partitioning
   * @param max_out_of_balance Allowable deviation from desired fraction of entities on each partition
   * @param n_try_new_axis Number of attempted cuts with no change in entity counts before trying a new axis
   */
  RCB( const MPI_Comm& comm, double max_out_of_balance = 0.1, int n_try_new_axis = 10 );

  /**
   * @brief Build entity partitioning using recursive coordinate bisection
   *
   * @param n_parts Number of subdomains to cut the list of coords into
   * @param coords_by_mesh List of on-rank points to subdivide sorted by mesh
   * @param ghost_len Sets length to include entities as ghost on each subdomain
   * @return List of points and ghost entities on each subdomain sorted by mesh
   */
  std::vector<EntityIndexByRank> generatePartitioning( int n_parts,
                                                       const std::vector<axom::Array<Point<NDIMS>>>& coords_by_mesh,
                                                       double ghost_len ) const override;

 private:
  /**
   * @brief Builds a binary bisection tree of domain cut locations (all ranks)
   *
   * @param n_parts Number of subdomains
   * @param coords_by_mesh Entity coordinates sorted by mesh (on-rank)
   * @param ghost_len Sets length to include entities as ghost on each subdomain
   * @return BisecTree of bounding boxes defining subdomains and ghost subdomains
   */
  BisecTree<RCBInfo<NDIMS>> BuildProblemTree( int n_parts, const std::vector<axom::Array<Point<NDIMS>>>& coords_by_mesh,
                                              double ghost_len ) const;

  /**
   * @brief Bounding box of coords on all ranks
   *
   * @param coords_by_mesh On-rank coords sorted by mesh
   * @return BoundingBox<NDIMS> Bounding box of coords on all ranks
   */
  BoundingBox<NDIMS> DomainBoundingBox( const std::vector<axom::Array<Point<NDIMS>>>& coords_by_mesh ) const;

  /**
   * @brief Counts the entities across all ranks in the two given bounding boxes
   *
   * @param left_bbox First (left) bounding box
   * @param right_bbox Second (right) bounding box
   * @param coords_by_mesh On-rank entity coordinates sorted by mesh
   * @return std::pair<int, int> Entity counts in each bounding box
   */
  std::pair<int, int> CountEntities( const BoundingBox<NDIMS>& left_bbox, const BoundingBox<NDIMS>& right_bbox,
                                     const std::vector<axom::Array<Point<NDIMS>>>& coords_by_mesh ) const;

  /**
   * @brief Find the domain at the base of the BisecTree the coord belongs to
   *
   * @param problem_tree BisecTree holding tree of bounding boxes of each domain
   * @param coord Entity coordinate to test
   * @return int Domain index
   */
  int DetermineDomain( const BisecTree<RCBInfo<NDIMS>>& problem_tree, const Point<NDIMS>& coord ) const;

  /**
   * @brief Sums number of entities over all processors
   *
   * @param n_local_ents Number of on-rank entities
   * @return int Total number of entities
   */
  int TotalEntities( int n_local_ents ) const;

  /**
   * @brief Tolerance for out-of-balance loading (default = 0.1, i.e. 10%)
   */
  double max_out_of_balance_;

  /**
   * @brief Number of cuts to try before switching to a new axis (default = 5).  Designed to guard against cases where
   * e.g. all elements are aligned along an axis.
   */
  int n_try_new_axis_;
};

using RCB2D = RCB<2>;
using RCB3D = RCB<3>;

}  // end namespace redecomp

#endif /* SRC_REDECOMP_PARTITION_RCB_HPP_ */
