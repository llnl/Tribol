// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_SHARED_MATH_PARSPARSEMAT_HPP_
#define SRC_SHARED_MATH_PARSPARSEMAT_HPP_

#include "shared/config.hpp"

#include <memory>

#include "mfem.hpp"

#include "shared/common/BasicTypes.hpp"
#include "shared/common/ExecModel.hpp"
#include "shared/math/ParVector.hpp"

#ifdef TRIBOL_USE_MPI
#include <HYPRE_utilities.h>
#endif

namespace shared {

#ifdef TRIBOL_USE_MPI

class ParSparseMat;

/**
 * @brief Non-owning view of a mfem::HypreParMatrix
 *
 * This class holds a raw pointer to a mfem::HypreParMatrix and provides algebraic operations.
 * It does not manage the lifetime of the matrix.
 */
class ParSparseMatView {
 public:
  /**
   * @brief Construct from a mfem::HypreParMatrix pointer
   *
   * @param mat Pointer to the mfem HypreParMatrix
   */
  ParSparseMatView( mfem::HypreParMatrix* mat );

  virtual ~ParSparseMatView() = default;

  /**
   * @brief Access the underlying mfem::HypreParMatrix
   */
  mfem::HypreParMatrix& get() { return *mat_; }

  /**
   * @brief Access the underlying mfem::HypreParMatrix (const)
   */
  const mfem::HypreParMatrix& get() const { return *mat_; }

  /**
   * @brief Access underlying matrix members via arrow operator
   */
  mfem::HypreParMatrix* operator->() { return mat_; }

  /**
   * @brief Access underlying matrix members via arrow operator (const)
   */
  const mfem::HypreParMatrix* operator->() const { return mat_; }

  /**
   * @brief Returns the number of local rows
   */
  int Height() const { return mat_->Height(); }

  /**
   * @brief Returns the number of local columns
   */
  int Width() const { return mat_->Width(); }

  /**
   * @brief Matrix addition: returns A + B
   */
  friend ParSparseMat operator+( const ParSparseMatView& lhs, const ParSparseMatView& rhs );

  /**
   * @brief Matrix subtraction: returns A - B
   */
  friend ParSparseMat operator-( const ParSparseMatView& lhs, const ParSparseMatView& rhs );

  /**
   * @brief Matrix scalar multiplication: returns s * A
   */
  ParSparseMat operator*( double s ) const;

  /**
   * @brief Matrix multiplication: returns A * B
   */
  friend ParSparseMat operator*( const ParSparseMatView& lhs, const ParSparseMatView& rhs );

  /**
   * @brief Matrix-vector multiplication: returns y = A * x
   */
  ParVector operator*( const ParVectorView& x ) const;

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
  ParSparseMat RAP( const ParSparseMatView& P ) const;

  /**
   * @brief Returns P^T * A * P
   */
  static ParSparseMat RAP( const ParSparseMatView& A, const ParSparseMatView& P );

  /**
   * @brief Returns Rt^T * A * P
   */
  static ParSparseMat RAP( const ParSparseMatView& Rt, const ParSparseMatView& A, const ParSparseMatView& P );

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
   * @brief Eliminates the rows and columns from the matrix
   *
   * @param cols Array of rows and columns to eliminate
   */
  ParSparseMat EliminateRowsCols( const mfem::Array<int>& rows_cols );

  /**
   * @brief Scalar-Matrix multiplication: returns s * A
   */
  friend ParSparseMat operator*( double s, const ParSparseMatView& mat );

  /**
   * @brief Vector-Matrix multiplication: returns y = x^T * A (computed as A^T * x)
   */
  friend ParVector operator*( const ParVectorView& x, const ParSparseMatView& mat );

 protected:
  static ParSparseMat add( RealT alpha, const ParSparseMatView& A, RealT beta, const ParSparseMatView& B );

  /**
   * @brief Helper to invoke a Hypre method with a specific memory location
   */
  template <MemorySpace MSPACE, typename F>
  static auto invokeHypreMethod( F&& f )
  {
    HYPRE_MemoryLocation old_hypre_mem_location;
    HYPRE_GetMemoryLocation( &old_hypre_mem_location );

    if constexpr ( MSPACE == MemorySpace::Host ) {
      HYPRE_SetMemoryLocation( HYPRE_MEMORY_HOST );
    }

    if constexpr ( std::is_same_v<decltype( f() ), void> ) {
      f();
      HYPRE_SetMemoryLocation( old_hypre_mem_location );
    } else {
      auto result = f();
      if constexpr ( std::is_same_v<decltype( result ), mfem::HypreParMatrix*> ) {
        if ( result ) {
          if constexpr ( MSPACE == MemorySpace::Host ) {
            constexpr int hypre_owned_host_arrays = -1;
            result->SetOwnerFlags( hypre_owned_host_arrays, hypre_owned_host_arrays, hypre_owned_host_arrays );
          }
        }
      }
      HYPRE_SetMemoryLocation( old_hypre_mem_location );
      return result;
    }
  }

  /**
   * @brief Creates a mfem::HypreParMatrix with a specific memory location and sets owner flags
   *
   * @tparam MSPACE Memory space to use
   * @tparam F Lambda type
   * @param f Lambda that returns a mfem::HypreParMatrix*
   * @return mfem::HypreParMatrix* The created matrix
   */
  template <MemorySpace MSPACE, typename F, std::enable_if_t<std::is_invocable_v<F>, int> = 0>
  static auto createHypreParMatrix( F&& f )
  {
    return invokeHypreMethod<MSPACE>( std::forward<F>( f ) );
  }

  /**
   * @brief Creates a mfem::HypreParMatrix with a specific memory location and sets owner flags
   *
   * @tparam MSPACE Memory space to use
   * @tparam Args Constructor argument types
   * @param args Constructor arguments
   * @return mfem::HypreParMatrix* The created matrix
   */
  template <MemorySpace MSPACE, typename... Args>
  static mfem::HypreParMatrix* createHypreParMatrix( Args&&... args )
  {
    return invokeHypreMethod<MSPACE>( [&]() { return new mfem::HypreParMatrix( std::forward<Args>( args )... ); } );
  }

  mfem::HypreParMatrix* mat_;
};

/**
 * @brief Wrapper class for mfem::HypreParMatrix to provide convenience operators
 *
 * This class owns a mfem::HypreParMatrix via a unique_ptr and adds support for
 * algebraic operations like addition and scalar multiplication.
 */
class ParSparseMat : public ParSparseMatView {
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

  /**
   * @brief Construct from MPI communicator, global size, row/column starts, and mfem::SparseMatrix rvalue
   *
   * @param comm MPI communicator
   * @param global_num_rows Global number of rows
   * @param global_num_cols Global number of columns
   * @param row_starts Global row partitioning
   * @param col_starts Global column partitioning
   * @param diag Local diagonal block SparseMatrix (rvalue)
   *
   * @note The HypreParMatrix will take ownership of the I, J, and Data from diag.
   */
  ParSparseMat( MPI_Comm comm, HYPRE_BigInt global_num_rows, HYPRE_BigInt global_num_cols, HYPRE_BigInt* row_starts,
                HYPRE_BigInt* col_starts, mfem::SparseMatrix&& diag );

  /// Template constructor forwarding arguments to mfem::HypreParMatrix constructor
  template <typename... Args>
  explicit ParSparseMat( Args&&... args )
      : ParSparseMatView( nullptr ),
        owned_mat_( createHypreParMatrix<MemorySpace::Host>( std::forward<Args>( args )... ) )
  {
    mat_ = owned_mat_.get();
  }

  /// Move constructor
  ParSparseMat( ParSparseMat&& other ) noexcept;

  /// Move assignment
  ParSparseMat& operator=( ParSparseMat&& other ) noexcept;

  // Disable copy constructor and assignment
  ParSparseMat( const ParSparseMat& ) = delete;
  ParSparseMat& operator=( const ParSparseMat& ) = delete;

  /**
   * @brief Access and release ownership of the HypreParMatrix pointer. The caller is now resposible for releasing the
   * memory.
   */
  mfem::HypreParMatrix* release();

  /**
   * @brief Matrix in-place addition: A += B
   */
  ParSparseMat& operator+=( const ParSparseMatView& other );

  /**
   * @brief Matrix in-place subtraction: A -= B
   */
  ParSparseMat& operator-=( const ParSparseMatView& other );

  /**
   * @brief Matrix in-place multiplication: A *= B
   */
  ParSparseMat& operator*=( const ParSparseMatView& other );

  /**
   * @brief Returns a diagonal matrix with the given diagonal value
   *
   * @param comm MPI communicator
   * @param global_size Global size of the matrix (rows and columns)
   * @param row_starts Row partitioning (global offsets)
   * @param diag_val Value for the diagonal entries
   * @param ordered_rows Sorted array of local row indices. Defaults to empty.
   * @param skip_rows If true (default), ordered_rows are skipped (zero entries). If false, ordered_rows are the only
   * entries.
   * @return ParSparseMat The constructed diagonal matrix
   */
  static ParSparseMat diagonalMatrix( MPI_Comm comm, HYPRE_BigInt global_size,
                                      const mfem::Array<HYPRE_BigInt>& row_starts, double diag_val,
                                      const mfem::Array<int>& ordered_rows = mfem::Array<int>(),
                                      bool skip_rows = true );

  /**
   * @brief Returns a diagonal matrix with the given diagonal value
   *
   * @param comm MPI communicator
   * @param global_size Global size of the matrix (rows and columns)
   * @param row_starts Row partitioning (global offsets)
   * @param diag_val Value for the diagonal entries
   * @param ordered_rows Sorted array of local row indices. Defaults to empty.
   * @param skip_rows If true (default), ordered_rows are skipped (zero entries). If false, ordered_rows are the only
   * entries.
   * @return ParSparseMat The constructed diagonal matrix
   */
  static ParSparseMat diagonalMatrix( MPI_Comm comm, HYPRE_BigInt global_size, HYPRE_BigInt* row_starts,
                                      double diag_val, const mfem::Array<int>& ordered_rows = mfem::Array<int>(),
                                      bool skip_rows = true );

  /**
   * @brief Returns a diagonal matrix with the values from the given vector on the diagonal
   *
   * @param comm MPI communicator
   * @param global_size Global size of the matrix (rows and columns)
   * @param row_starts Row partitioning (global offsets)
   * @param diag_vals Vector containing the values for the diagonal entries. Size must match local rows.
   * @return ParSparseMat The constructed diagonal matrix
   */
  static ParSparseMat diagonalMatrix( MPI_Comm comm, HYPRE_BigInt global_size, HYPRE_BigInt* row_starts,
                                      const mfem::Vector& diag_vals );

 private:
  std::unique_ptr<mfem::HypreParMatrix> owned_mat_;
};

#endif  // #ifdef TRIBOL_USE_MPI

}  // namespace shared

#endif /* SRC_SHARED_MATH_PARSPARSEMAT_HPP_ */
