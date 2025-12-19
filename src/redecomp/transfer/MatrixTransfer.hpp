// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_TRANSFER_MATRIXTRANSFER_HPP_
#define SRC_REDECOMP_TRANSFER_MATRIXTRANSFER_HPP_

#include "mfem.hpp"

#include "redecomp/common/TypeDefs.hpp"

namespace redecomp {

class RedecompMesh;

/**
 * @brief Transfers element matrices from RedecompMesh to mfem::ParMesh
 *
 * This class enables transferring element mfem::DenseMatrix's on a RedecompMesh
 * to a mfem::HypreParMatrix on a parent mfem::ParMesh. The primary method which
 * this is accomplished is TransferToParallel(), which takes element-level
 * DenseMatrix contributions on mfem::FiniteElementSpaces on a RedecompMesh,
 * then returns a mfem::HypreParMatrix on the ldofs or tdofs of
 * mfem::ParFiniteElementSpaces on the parent mfem::ParMesh. Alternatively, a
 * two-stage transfer process is available. First, TransferToParallel() creates
 * an un-finalized mfem::SparseMatrix with ldofs on the rows and global ldofs on
 * the columns. This mfem::SparseMatrix is designed to be passed to a
 * mfem::HypreParMatrix constructor (done through the ConvertToHypreParMatrix()
 * method).  The two-stage process allows easier manipulation of matrix
 * contributions before the matrix is finalized.  Both square and rectangular
 * matrices are supported, necessitating test and trial finite element spaces in
 * the constructor.
 */
class MatrixTransfer {
 public:
  /**
   * @brief Construct a new Matrix Transfer object
   *
   * @param parent_test_fes Test finite element space on parent mfem::ParMesh
   * @param parent_trial_fes Trial finite element space on parent mfem::ParMesh
   * @param redecomp_test_fes Test finite element space on RedecompMesh
   * @param redecomp_trial_fes Trial finite element space on RedecompMesh
   */
  MatrixTransfer( const mfem::ParFiniteElementSpace& parent_test_fes,
                  const mfem::ParFiniteElementSpace& parent_trial_fes,
                  const mfem::FiniteElementSpace& redecomp_test_fes,
                  const mfem::FiniteElementSpace& redecomp_trial_fes );

  /**
   * @brief Transfers element RedecompMesh matrices to parent mfem::ParMesh
   *
   * @param test_elem_idx List of element IDs on the redecomp test space
   * @param trial_elem_idx List of element IDs on the redecomp trial space
   * @param src List of element-level dense matrices from the redecomp mesh
   * @param parallel_assemble Performs parallel assembly (transforms to tdofs)
   * on the HypreParMatrix if true, returns ldofs otherwise
   * @return Owned pointer to mfem::HypreParMatrix on the parent mesh
   *
   * @note Parallel assembly (e.g. mfem::RAP()) must be done on the
   * HypreParMatrix returned to transform it to the t-dofs if parallel_assemble
   * is false.
   */
  std::unique_ptr<mfem::HypreParMatrix> TransferToParallel( const axom::Array<int>& test_elem_idx,
                                                            const axom::Array<int>& trial_elem_idx,
                                                            const axom::Array<mfem::DenseMatrix>& src_elem_mat,
                                                            bool parallel_assemble = true ) const;

  /**
   * @brief Transfers element RedecompMesh matrices to parent mfem::ParMesh
   *
   * @param test_elem_idx List of element IDs on the redecomp test space
   * @param trial_elem_idx List of element IDs on the redecomp trial space
   * @param src List of element-level dense matrices from the redecomp mesh
   * @return mfem::SparseMatrix on the parent mesh (ldofs on the rows, global
   * ldofs on the columns) in linked-list format
   *
   * @note Linked-list format is returned here to enable additional matrix
   * contributions to be added before Finalize() is called on the SparseMatrix
   */
  mfem::SparseMatrix TransferToParallelSparse( const axom::Array<int>& test_elem_idx,
                                               const axom::Array<int>& trial_elem_idx,
                                               const axom::Array<mfem::DenseMatrix>& src_elem_mat ) const;

  /**
   * @brief Converts SparseMatrix from TransferToParallelSparse to HypreParMatrix
   *
   * @param sparse Finalized mfem::SparseMatrix returned from TransferToParallel
   * @param parallel_assemble Performs parallel assembly (transforms to tdofs)
   * on the HypreParMatrix if true, returns ldofs otherwise
   * @return Owned pointer to mfem::HypreParMatrix on the parent mesh
   *
   * @note Parallel assembly (e.g. mfem::RAP()) must be done on the
   * HypreParMatrix returned to transform it to the t-dofs if parallel_assemble
   * is false.
   */
  std::unique_ptr<mfem::HypreParMatrix> ConvertToHypreParMatrix( mfem::SparseMatrix& sparse,
                                                                 bool parallel_assemble = true ) const;

 private:
  /**
   * @brief Returns a map of the corresponding parent rank for a given redecomp index
   *
   * @param redecomp Redecomp mesh holding elements whose parent rank should be mapped
   * @param mark_ghost If true, mark ghost element ranks as -1
   * @return axom::Array<int> Map indicating parent rank for given redecomp element id
   */
  axom::Array<int> buildRedecomp2ParentElemRank( const RedecompMesh& redecomp, bool mark_ghost );

  /**
   * @brief List of entries in test_elem_idx that belong on each parent test space rank
   *
   * @param test_elem_idx List of element IDs on the redecomp test space
   * @return MPIArray<int>
   */
  MPIArray<int> buildSendArrayIDs( const axom::Array<int>& test_elem_idx ) const;

  /**
   * @brief Number of matrix entries to be sent to each parent test space rank
   *
   * @param test_elem_idx List of element IDs on the redecomp test space
   * @param trial_elem_idx List of element IDs on the redecomp trial space
   * @return axom::Array<int>
   */
  axom::Array<int> buildSendNumMatEntries( const axom::Array<int>& test_elem_idx,
                                           const axom::Array<int>& trial_elem_idx ) const;

  /**
   * @brief Number of test and trial vdofs received from test space redecomp ranks
   *
   * @param test_elem_idx List of element IDs on the redecomp test space
   * @param trial_elem_idx List of element IDs on the redecomp trial space
   * @return MPIArray<int, 2>
   */
  MPIArray<int, 2> buildRecvMatSizes( const axom::Array<int>& test_elem_idx,
                                      const axom::Array<int>& trial_elem_idx ) const;

  /**
   * @brief List of test element offsets received from test space redecomp ranks
   *
   * @param test_redecomp Redecomp mesh of the test space
   * @param test_elem_idx List of element IDs on the redecomp test space
   * @return MPIArray<int>
   */
  MPIArray<int> buildRecvTestElemOffsets( const RedecompMesh& test_redecomp,
                                          const axom::Array<int>& test_elem_idx ) const;

  /**
   * @brief List of trial element global vdofs corresponding to the element matrix
   *        entries received from redecomp ranks
   *
   * @param trial_redecomp Redecomp mesh of the trial space
   * @param test_elem_idx List of element IDs on the redecomp test space
   * @param trial_elem_idx List of element IDs on the redecomp trial space
   * @return MPIArray<HYPRE_BigInt>
   */
  MPIArray<HYPRE_BigInt> buildRecvTrialElemDofs( const RedecompMesh& trial_redecomp,
                                                 const axom::Array<int>& test_elem_idx,
                                                 const axom::Array<int>& trial_elem_idx ) const;

  /**
   * @brief Returns MPIUtility pointer for the MatrixTransfer object
   *
   * @return const MPIUtility*
   */
  const MPIUtility& getMPIUtility() const;

  /**
   * @brief Test finite element space on the parent mfem::ParMesh
   */
  const mfem::ParFiniteElementSpace& parent_test_fes_;

  /**
   * @brief Trial finite element space on the parent mfem::ParMesh
   */
  const mfem::ParFiniteElementSpace& parent_trial_fes_;

  /**
   * @brief Test finite element space on the RedecompMesh
   */
  const mfem::FiniteElementSpace& redecomp_test_fes_;

  /**
   * @brief Trial finite element space on the RedecompMesh
   */
  const mfem::FiniteElementSpace& redecomp_trial_fes_;

  /**
   * @brief Parent rank of given test redecomp element id
   *
   * Note: r2p = redecomp to parent
   */
  axom::Array<int> test_r2p_elem_rank_;

  /**
   * @brief Parent rank of given trial redecomp element id
   *
   * Note: r2p = redecomp to parent
   */
  axom::Array<int> trial_r2p_elem_rank_;
};

}  // end namespace redecomp

#endif /* SRC_REDECOMP_TRANSFER_MATRIXTRANSFER_HPP_ */
