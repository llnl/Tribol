// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_TRANSFER_TRANSFERBYNODES_HPP_
#define SRC_REDECOMP_TRANSFER_TRANSFERBYNODES_HPP_

#include "mfem.hpp"

#include "redecomp/transfer/GridFnTransfer.hpp"
#include "redecomp/common/TypeDefs.hpp"

namespace redecomp {

class RedecompMesh;

class MPIUtility;

/**
 * @brief GridFnTransfer method using nodes
 *
 * This method of GridFnTransfer can be used to speed repeated transfer of H1
 * fields from a parent mesh to a Redecomp mesh.  Speed gains are obtained by
 * creating lists of nodes to transfer from parent to Redecomp and vice versa.
 * This eliminates repeated transfer of nodes on inter-element boundaries, which
 * occur with the related TransferByElements class.  However, creating these
 * maps is a non-trivial operation, and therefore this class should only be used
 * if multiple transfers between parent and RedecompMesh are anticipated.
 */
class TransferByNodes : public GridFnTransfer {
 public:
  /**
   * @brief Construct a new TransferByNodes object
   *
   * @param parent_fes ParFiniteElementSpace constructed from parent mesh
   * identified in redecomp_fes
   * @param redecomp_fes FiniteElementSpace constructed from RedecompMesh
   */
  TransferByNodes( const mfem::ParFiniteElementSpace& parent_fes, const mfem::FiniteElementSpace& redecomp_fes );
  /**
   * @brief Construct a new TransferByNodes object (used by RedecompMesh)
   *
   * @note This constructor is used in the RedecompMesh constructor to assist
   * with transferring and creating vertices on the RedecompMesh. Since this
   * constructor does not create a usable TransferByNodes object, end user
   * utility is limited.
   *
   * @param parent_fes ParFiniteElementSpace constructed from parent mesh
   * identified in redecomp
   * @param redecomp RedecompMesh pointer
   */
  TransferByNodes( const mfem::ParFiniteElementSpace& parent_fes, const RedecompMesh& redecomp );

  /**
   * @brief Copies parent-based mfem::ParGridFunction values to a
   * RedecompMesh-based mfem::GridFunction
   *
   * @param src A parent ParGridFunction to be copied to corresponding redecomp
   * GridFunction (dst)
   * @param dst A redecomp GridFunction which receives values from a parent
   * ParGridFunction (src)
   */
  void TransferToSerial( const mfem::ParGridFunction& src, mfem::GridFunction& dst ) const override;

  /**
   * @brief Copies RedecompMesh-based mfem::GridFunction values to a
   * parent-based mfem::ParGridFunction
   *
   * @param src A redecomp GridFunction to be copied to corresponding parent
   * ParGridFunction (dst)
   * @param dst A parent ParGridFunction which receives values from a redecomp
   * GridFunction (src)
   */
  void TransferToParallel( const mfem::GridFunction& src, mfem::ParGridFunction& dst ) const override;

  /**
   * @brief Determine list of parent nodes to send to RedecompMesh
   *
   * @param use_global_ids Should global IDs be used on the parent mesh?
   * @return EntityIndexByRank List of parent node ids (with ghost information)
   * which belong on each RedecompMesh rank
   */
  EntityIndexByRank P2RNodeList( bool use_global_ids );

 private:
  /**
   * @brief Determine list of RedecompMesh nodes to send to parent mesh
   *
   * @return EntityIndexByRank List of RedecompMesh node ids (with ghost
   * information) which belong on each parent rank
   */
  EntityIndexByRank R2PNodeList();

  /**
   * @brief ParFiniteElementSpace constructed from mesh identified as parent in redecomp_
   */
  const mfem::ParFiniteElementSpace* parent_fes_;

  /**
   * @brief FiniteElementSpace constructed from redecomp_
   */
  const mfem::FiniteElementSpace* redecomp_fes_;

  /**
   * @brief Redecomp mesh
   */
  const RedecompMesh* redecomp_;

  /**
   * @brief List of parent nodes (with ghost information) to send to RedecompMesh
   */
  EntityIndexByRank p2r_nodes_;

  /**
   * @brief List of Redecomp nodes (with ghost information) to send to parent mesh
   */
  EntityIndexByRank r2p_nodes_;
};

}  // end namespace redecomp

#endif /* SRC_REDECOMP_TRANSFER_TRANSFERBYNODES_HPP_ */
