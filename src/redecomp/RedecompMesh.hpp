// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_REDECOMPMESH_HPP_
#define SRC_REDECOMP_REDECOMPMESH_HPP_

#include "mfem.hpp"

#include "axom/core.hpp"

#include "redecomp/common/TypeDefs.hpp"

namespace redecomp {

class Partitioner;

/**
 * @brief Redecomposes an mfem::ParMesh using a specified method
 *
 * Takes an mfem::ParMesh and redecomposes it across MPI ranks by the method
 * specified in the constructor.  Also creates a list of elements transferred
 * between meshes that is used to transfer functions defined on the mesh.
 * RedecompMesh has a notion of ghost elements, which generally are elements
 * which have data transferred to them from the parent mfem::ParMesh, but whose
 * data are not returned when transferred back to the mfem::ParMesh.
 *
 * @note Though RedecompMesh is distributed across ranks, the RedecompMesh on
 * each rank is an independent, serial mesh that derives from mfem::Mesh.
 */
class RedecompMesh : public mfem::Mesh {
 public:
  /**
   * @brief List of partitioning methods verified to work with RedecompMesh
   */
  enum PartitionType
  {
    RCB
  };

  /**
   * @brief Construct a new RedecompMesh object
   *
   * @note This constructor builds the Partitioner object based on the method passed. If no method is passed, a RCB
   * Partitioner is constructed.
   *
   * @param parent The mfem::ParMesh that will be redecomposed
   * @param method The method of redecomposition (optional)
   * @param n_ranks Sets number of MPI ranks to build RedecompMesh on (default = 0). Set this to zero to have this
   * automatically determined.
   */
  RedecompMesh( const mfem::ParMesh& parent, PartitionType method = RCB, int n_ranks = 0 );

  /**
   * @brief Construct a new RedecompMesh object
   *
   * @note This constructor builds the Partitioner object based on the method passed. If no method is passed, a RCB
   * Partitioner is constructed.
   *
   * @param parent The mfem::ParMesh that will be redecomposed
   * @param ghost_length Size of layer of un-owned ghost elements to include around the edge of the on-rank domain
   * @param method The method of redecomposition (optional)
   * @param n_ranks Sets number of MPI ranks to build RedecompMesh on (default = 0). Set this to zero to have this
   * automatically determined.
   */
  RedecompMesh( const mfem::ParMesh& parent, double ghost_length, PartitionType method = RCB, int n_ranks = 0 );

  /**
   * @brief Construct a new RedecompMesh object
   *
   * @note This constructor requires the Partitioner object passed directly to
   * it. This can be used to customize the Partitioner used (for example, with
   * non-default options with RCB or for a user-defined Partitioner).
   *
   * @param parent The mfem::ParMesh that will be redecomposed
   * @param partitioner Partitioning object used to define redecomposition
   * @param n_ranks Sets number of MPI ranks to build RedecompMesh on (default = 0). Set this to zero to have this
   * automatically determined.
   */
  RedecompMesh( const mfem::ParMesh& parent, std::unique_ptr<const Partitioner> partitioner, int n_ranks = 0 );

  /**
   * @brief Construct a new RedecompMesh object
   *
   * @note This constructor requires the Partitioner object passed directly to
   * it. This can be used to customize the Partitioner used (for example, with
   * non-default options with RCB or for a user-defined Partitioner).
   *
   * @param parent The mfem::ParMesh that will be redecomposed
   * @param ghost_length Size of layer of un-owned ghost elements to include around the edge of the on-rank domain
   * @param partitioner Partitioning object used to define redecomposition
   * @param n_ranks Sets number of MPI ranks to build RedecompMesh on (default = 0). Set this to zero to have this
   * automatically determined.
   */
  RedecompMesh( const mfem::ParMesh& parent, double ghost_length, std::unique_ptr<const Partitioner> partitioner,
                int n_ranks = 0 );

  /**
   * @brief Construct a new RedecompMesh object
   *
   * @param parent The mfem::ParMesh that will be redecomposed
   * @param p2r_elems List of local parent element ids to put on each
   * RedecompMesh rank
   */
  RedecompMesh( const mfem::ParMesh& parent, EntityIndexByRank&& p2r_elems );

  /**
   * @brief Get the parent mfem::ParMesh
   *
   * @return const mfem::ParMesh*
   */
  const mfem::ParMesh& getParent() const { return parent_; }

  /**
   * @brief Return the MPIUtility
   *
   * @return const MPIUtility&
   */
  const MPIUtility& getMPIUtility() const { return mpi_; }

  /**
   * @brief Get the list of local parent elements that belong on each rank of
   * RedecompMesh
   *
   * @return const EntityIndexByRank&
   */
  const EntityIndexByRank& getParentToRedecompElems() const { return p2r_elems_; }

  /**
   * @brief Get Redecomp mesh element offsets denoting parent elements on each
   * rank
   *
   * Redecomp element IDs are ordered by parent rank; therefore, only an element
   * offset is required to identify the parent rank of a given redecomp element
   * id.  The first redecomp element on parent rank m is given by entry m of the
   * array.
   *
   * @return const axom::Array<int>&
   */
  const axom::Array<int>& getRedecompToParentElemOffsets() const { return r2p_elem_offsets_; }

  /**
   * @brief Get a list of redecomp ghost elements on each parent rank
   *
   * @return const MPIArray<int>&
   */
  const MPIArray<int>& getRedecompToParentGhostElems() const { return r2p_ghost_elems_; }

  /**
   * @brief Computes the largest element length in terms of stretch at the
   * element centroids
   *
   * @note This can underestimate the largest element size if stretch is variable over the element.
   *
   * @param parent The mfem::ParMesh containing the elements
   * @param mpi MPI utility containing the communicator of the parent mesh
   *
   * @return Largest element length in terms of stretch component at element centroid
   */
  static double MaxElementSize( const mfem::ParMesh& parent, const MPIUtility& mpi );

 private:
  /**
   * @brief Computes a default ghost element length: 1.25 * max element size
   *
   * @param parent The mfem::ParMesh containing the elements
   *
   * @return Default ghost element lemgth
   */
  double DefaultGhostLength( const mfem::ParMesh& parent ) const;

  /**
   * @brief Builds list of parent elements to be transfered to Redecomp ranks
   *
   * @param partitioner Method of partitioning the elements
   * @param n_parts Number of parts to partition the mesh into
   * @param ghost_length Size of layer of un-owned ghost elements to include around the edge of the on-rank domain
   * @return List of parent element IDs and ghost elements sorted by Redecomp rank
   */
  EntityIndexByRank BuildP2RElementList( const Partitioner& partitioner, int n_parts, double ghost_length ) const;

  /**
   * @brief Builds the Redecomp mesh and inverse element transfer list
   */
  void BuildRedecomp();

  /**
   * @brief Linked parent mfem::ParMesh
   */
  const mfem::ParMesh& parent_;

  /**
   * @brief MPI utility for the Redecomp MPI_Comm
   */
  MPIUtility mpi_;

  /**
   * @brief List of parent elements that belong on each RedecompMesh rank
   */
  EntityIndexByRank p2r_elems_;

  /**
   * @brief Range of redecomp elements that belong on each parent rank
   */
  axom::Array<int> r2p_elem_offsets_;

  /**
   * @brief Ghost redecomp elements sorted by parent rank
   */
  MPIArray<int> r2p_ghost_elems_;
};

}  // namespace redecomp

#endif /* SRC_REDECOMP_REDECOMPMESH_HPP_ */
