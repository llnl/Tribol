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
   * @brief Select one return-eligible copy of an already-assembled redecomp dual
   *
   * @param src Redecomp linear form with matching return-eligible copies
   * @param dst Parent parallel linear form receiving the designated copy
   */
  void CollapseReplicatedDualToParallel( const mfem::LinearForm& src, mfem::ParLinearForm& dst ) const override;

  /**
   * @brief Build parent DOF lists for each destination redecomp rank
   *
   * @param use_global_ids Should global IDs be used on the parent mesh?
   * @return Parent DOFs and ghost flags grouped by destination redecomp rank
   */
  EntityIndexByRank BuildParentToRedecompNodeList( bool use_global_ids );

 private:
  /**
   * @brief Build redecomp DOF lists for each destination parent rank
   *
   * @return Redecomp DOFs and ghost flags grouped by destination parent rank
   */
  EntityIndexByRank BuildRedecompToParentNodeList();

  /**
   * @brief Build maps that select one return-eligible redecomp copy for each parent DOF
   */
  void BuildDesignatedReturnNodeLists();

  /**
   * @brief Select return-eligible redecomp values for parent local DOFs
   *
   * @param src Redecomp field data
   * @param dst Parent field data
   * @param keep_only_parent_owned_dofs If true, zero shared parent copies not owned by this rank
   * @param field_name Name included in debug diagnostics
   */
  void TransferDesignatedValuesToParent( const mfem::Vector& src, mfem::Vector& dst,
                                         bool keep_only_parent_owned_dofs, const char* field_name ) const;

  /**
   * @brief Verify that every return-eligible redecomp copy matches the designated parent value
   *
   * @param src Redecomp field data
   * @param selected_parent_values Parent values produced by the designated-copy transfer
   * @param field_name Name included in diagnostics
   */
  void CheckReturnEligibleCopies( const mfem::Vector& src, const mfem::Vector& selected_parent_values,
                                  const char* field_name ) const;

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
   * @brief Parent scalar DOFs and ghost flags grouped by destination redecomp rank
   */
  EntityIndexByRank parent_to_redecomp_nodes_;

  /**
   * @brief Redecomp scalar DOFs and ghost flags grouped by destination parent rank
   */
  EntityIndexByRank redecomp_to_parent_nodes_;

  /**
   * @brief Redecomp scalar DOFs that supply designated values to each parent rank
   */
  MPIArray<int> redecomp_dofs_supplying_designated_values_;

  /**
   * @brief Parent scalar DOFs receiving designated values from each redecomp rank
   */
  MPIArray<int> parent_dofs_receiving_designated_values_;
};

}  // end namespace redecomp

#endif /* SRC_REDECOMP_TRANSFER_TRANSFERBYNODES_HPP_ */
