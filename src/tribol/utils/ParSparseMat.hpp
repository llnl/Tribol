// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_UTILS_PARSPARSEMAT_HPP_
#define SRC_TRIBOL_UTILS_PARSPARSEMAT_HPP_

#include "tribol/config.hpp"
#include "mfem.hpp"

#include <memory>
#include <utility>

namespace tribol {

/**
 * @brief Wrapper class for mfem::HypreParMatrix to provide convenience operators
 *
 * This class owns a mfem::HypreParMatrix via a unique_ptr and adds support for
 * algebraic operations like addition and scalar multiplication.
 */
class ParSparseMat {
 public:
  /**
   * @brief Construct from a mfem::HypreParMatrix pointer and take ownership
   *
   * @param mat Pointer to the mfem HypreParMatrix
   */
  explicit ParSparseMat( mfem::HypreParMatrix* mat );

  /**
   * @brief Construct from a mfem::HypreParMatrix pointer and take ownership
   *
   * @param mat Pointer to the mfem HypreParMatrix
   */
  explicit ParSparseMat( std::unique_ptr<mfem::HypreParMatrix> mat );

  /**
   * @brief Construct from MPI communicator, global size, row_starts, and mfem::SparseMatrix rvalue
   *
   * @param comm MPI communicator
   * @param glob_size Global number of rows (and columns)
   * @param row_starts Global row partitioning
   * @param diag Local diagonal block SparseMatrix (rvalue)
   *
   * @note The HypreParMatrix will take ownership of the I, J, and Data from diag.
   */
  ParSparseMat( MPI_Comm comm, HYPRE_BigInt glob_size, HYPRE_BigInt* row_starts, mfem::SparseMatrix&& diag );

  /// Template constructor forwarding arguments to mfem::HypreParMatrix constructor
  template <typename... Args>
  explicit ParSparseMat( Args&&... args )
      : m_mat( std::make_unique<mfem::HypreParMatrix>( std::forward<Args>( args )... ) )
  {
  }

  /// Move constructor
  ParSparseMat( ParSparseMat&& other ) = default;

  /// Move assignment
  ParSparseMat& operator=( ParSparseMat&& other ) = default;

  // Disable copy constructor and assignment
  ParSparseMat( const ParSparseMat& ) = delete;
  ParSparseMat& operator=( const ParSparseMat& ) = delete;

  /**
   * @brief Access the underlying mfem::HypreParMatrix
   */
  mfem::HypreParMatrix& get() { return *m_mat; }

  /**
   * @brief Access the underlying mfem::HypreParMatrix (const)
   */
  const mfem::HypreParMatrix& get() const { return *m_mat; }

  /**
   * @brief Access underlying matrix members via arrow operator
   */
  mfem::HypreParMatrix* operator->() { return m_mat.get(); }

  /**
   * @brief Access underlying matrix members via arrow operator (const)
   */
  const mfem::HypreParMatrix* operator->() const { return m_mat.get(); }

  /**
   * @brief Access and release ownership of the HypreParMatrix pointer. The caller is now resposible for releasing the
   * memory.
   */
  mfem::HypreParMatrix* release() { return m_mat.release(); }

  /**
   * @brief Matrix addition: returns A + B
   */
  ParSparseMat operator+( const ParSparseMat& other ) const;

  /**
   * @brief Matrix in-place addition: A += B
   */
  ParSparseMat& operator+=( const ParSparseMat& other );

  /**
   * @brief Matrix subtraction: returns A - B
   */
  ParSparseMat operator-( const ParSparseMat& other ) const;

  /**
   * @brief Matrix in-place subtraction: A -= B
   */
  ParSparseMat& operator-=( const ParSparseMat& other );

  /**
   * @brief Matrix scalar multiplication: returns s * A
   */
  ParSparseMat operator*( double s ) const;

  /**
   * @brief Matrix multiplication: returns A * B
   */
  ParSparseMat operator*( const ParSparseMat& other ) const;

  /**
   * @brief Matrix in-place multiplication: A *= B
   */
  ParSparseMat& operator*=( const ParSparseMat& other );

  /**
   * @brief Matrix-vector multiplication: returns y = A * x
   */
  mfem::Vector operator*( const mfem::Vector& x ) const;

  /**
   * @brief Returns the transpose of the matrix
   */
  ParSparseMat transpose() const;

  /**
   * @brief Returns the square of the matrix (A * A)
   */
  ParSparseMat square() const;

  /**
   * @brief Returns P^T * A * P
   */
  ParSparseMat RAP( const ParSparseMat& P ) const;

  /**
   * @brief Returns R * A * P
   */
  static ParSparseMat RAP( const ParSparseMat& R, const ParSparseMat& A, const ParSparseMat& P );

  /**
   * @brief Returns a diagonal matrix with the given diagonal value
   *
   * @param comm MPI communicator
   * @param global_size Global size of the matrix (rows and columns)
   * @param row_starts Row partitioning (global offsets)
   * @param diag_val Value for the diagonal entries
   * @param ordered_zero_val_rows Sorted array of local row indices that should be zero (inactive). Defaults to empty.
   * @return ParSparseMat The constructed diagonal matrix
   */
  static ParSparseMat diagonalMatrix( MPI_Comm comm, HYPRE_BigInt global_size,
                                      const mfem::Array<HYPRE_BigInt>& row_starts, double diag_val,
                                      const mfem::Array<int>& ordered_zero_val_rows = mfem::Array<int>() );

  /**
   * @brief Eliminates the rows from the matrix
   *
   * @param rows Array of rows to eliminate
   */
  void EliminateRows( const mfem::Array<int>& rows );

  /**
   * @brief Eliminates the columns from the matrix
   *
   * @param cols Array of columns to eliminate
   */
  ParSparseMat EliminateCols( const mfem::Array<int>& cols );

  /**
   * @brief Scalar-Matrix multiplication: returns s * A
   */
  friend ParSparseMat operator*( double s, const ParSparseMat& mat );

  /**
   * @brief Vector-Matrix multiplication: returns y = x^T * A (computed as A^T * x)
   */
  friend mfem::Vector operator*( const mfem::Vector& x, const ParSparseMat& mat );

 private:
  std::unique_ptr<mfem::HypreParMatrix> m_mat;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_UTILS_PARSPARSEMAT_HPP_ */