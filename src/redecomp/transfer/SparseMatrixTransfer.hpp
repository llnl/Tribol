// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_TRANSFER_SPARSEMATRIXTRANSFER_HPP_
#define SRC_REDECOMP_TRANSFER_SPARSEMATRIXTRANSFER_HPP_

#include <memory>

#include "mfem.hpp"

#include "axom/core.hpp"

namespace redecomp {

class MPIUtility;
class RedecompMesh;

/**
 * @brief Transfers sparse matrices on FiniteElementSpaces from RedecompMesh to mfem::ParMesh
 *
 * This class enables transferring an mfem::SparseMatrix whose rows and columns are linked to mfem::FiniteElementSpaces
 * on a RedecompMesh to a mfem::HypreParMatrix whose rows and columns are linked to mfem::ParFiniteElementSpaces on an
 * mfem::ParMesh.  Since the row and column space can be different (i.e. a rectangular matrix, in general), both spaces
 * must be identified in the constructor.
 *
 * The class currently supports order 0 L2 fields (one value per element).
 */
class SparseMatrixTransfer {
 public:
  /**
   * @brief Construct a new SparseMatrixTransfer object
   *
   * @param parent_test_space Test finite element space on parent mfem::ParMesh
   * @param parent_trial_space Trial finite element space on parent mfem::ParMesh
   * @param redecomp_test_space Test finite element space on RedecompMesh
   * @param redecomp_trial_space Trial finite element space on RedecompMesh
   */
  SparseMatrixTransfer( const mfem::ParFiniteElementSpace& parent_test_space,
                        const mfem::ParFiniteElementSpace& parent_trial_space,
                        const mfem::FiniteElementSpace& redecomp_test_space,
                        const mfem::FiniteElementSpace& redecomp_trial_space );

  /**
   * @brief Transfers RedecompMesh sparse matrix to parent mfem::ParMesh hypre par matrix
   *
   * @param src Sparse matrix whose rows and columns are linked to test and trial FiniteElementSpaces on the
   * RedecompMesh
   * @return Owned pointer to mfem::HypreParMatrix on the parent mesh
   *
   * @note The HypreParMatrix is on the global dofs.  Apply mfem::RAP to reduce to tdofs.
   */
  std::unique_ptr<mfem::HypreParMatrix> TransferToParallel( const mfem::SparseMatrix& src ) const;

 private:
  /**
   * @brief Get the RedecompMesh from the finite element space
   *
   * This method verifies the finite element space does have a RedecompMesh before returning. Finite element spaces with
   * no RedecompMesh trigger a SLIC_ERROR macro.
   *
   * @param fe_space Finite element space linked to a RedecompMesh
   * @return const RedecompMesh& Redecomp mesh associated with finite element space
   */
  static const RedecompMesh& getRedecompMesh( const mfem::FiniteElementSpace& fe_space );

  /**
   * @brief Build vector of parent element offsets for each MPI rank
   *
   * @param redecomp_mesh Offset vector is computed on the parent-linked mesh of this mesh
   * @return Array of element offsets for each MPI rank
   */
  static axom::Array<HYPRE_BigInt> BuildParentElementRankOffsets( const RedecompMesh& redecomp_mesh );

  /**
   * @brief Returns MPIUtility pointer for the MatrixTransfer object
   *
   * @return const MPIUtility*
   */
  const MPIUtility& getMPIUtility() const;

  /**
   * @brief Test finite element space on the parent mfem::ParMesh
   */
  const mfem::ParFiniteElementSpace& parent_test_space_;

  /**
   * @brief Trial finite element space on the parent mfem::ParMesh
   */
  const mfem::ParFiniteElementSpace& parent_trial_space_;

  /**
   * @brief Test finite element space on the RedecompMesh
   */
  const mfem::FiniteElementSpace& redecomp_test_space_;

  /**
   * @brief Trial finite element space on the RedecompMesh
   */
  const mfem::FiniteElementSpace& redecomp_trial_space_;

  /**
   * @brief Test RedecompMesh
   */
  const RedecompMesh& redecomp_test_mesh_;

  /**
   * @brief Trial RedecompMesh
   */
  const RedecompMesh& redecomp_trial_mesh_;
};

}  // end namespace redecomp

#endif /* SRC_REDECOMP_TRANSFER_SPARSEMATRIXTRANSFER_HPP_ */
