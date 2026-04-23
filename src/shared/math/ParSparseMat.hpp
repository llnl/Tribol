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

  /**
   * @brief Destroy the view without deleting the wrapped matrix
   */
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
  int height() const { return mat_->Height(); }

  /**
   * @brief Returns the number of local columns
   */
  int width() const { return mat_->Width(); }

  /**
   * @brief Returns lhs + rhs
   */
  friend ParSparseMat operator+( const ParSparseMatView& lhs, const ParSparseMatView& rhs );

  /**
   * @brief Returns lhs - rhs
   */
  friend ParSparseMat operator-( const ParSparseMatView& lhs, const ParSparseMatView& rhs );

  /**
   * @brief Returns this scaled by s
   */
  ParSparseMat operator*( double s ) const;

  /**
   * @brief Returns lhs * rhs
   */
  friend ParSparseMat operator*( const ParSparseMatView& lhs, const ParSparseMatView& rhs );

  /**
   * @brief Returns this * x
   */
  ParVector operator*( const ParVectorView& x ) const;

  /**
   * @brief Returns the transpose of the matrix
   */
  ParSparseMat transpose() const;

  /**
   * @brief Returns this squared
   */
  ParSparseMat square() const;

  /**
   * @brief Returns P^T * this * P
   */
  ParSparseMat rap( const ParSparseMatView& P ) const;

  /**
   * @brief Returns P^T * A * P
   */
  static ParSparseMat rap( const ParSparseMatView& A, const ParSparseMatView& P );

  /**
   * @brief Returns Rt^T * A * P
   */
  static ParSparseMat rap( const ParSparseMatView& Rt, const ParSparseMatView& A, const ParSparseMatView& P );

  /**
   * @brief Eliminates chosen rows from the matrix
   *
   * @param rows Array of rows to eliminate
   */
  void eliminateRows( const mfem::Array<int>& rows );

  /**
   * @brief Eliminates chosen columns from the matrix
   *
   * @param cols Array of columns to eliminate
   */
  ParSparseMat eliminateCols( const mfem::Array<int>& cols );

  /**
   * @brief Returns s * mat
   */
  friend ParSparseMat operator*( double s, const ParSparseMatView& mat );

  /**
   * @brief Returns x^T * mat, computed as mat^T * x
   */
  friend ParVector operator*( const ParVectorView& x, const ParSparseMatView& mat );

 protected:
  /**
   * @brief Force a wrapped Hypre matrix to use valid host storage
   *
   * All shared::ParSparseMat operations are currently host-only. This helper may update MFEM's memory residency state
   * but does not change matrix values.
   */
  static void ensureHostMemory( mfem::HypreParMatrix* mat );

  /**
   * @brief Force a wrapped view operand to use valid host storage
   */
  static void ensureHostMemory( const ParSparseMatView& mat );

  /**
   * @brief Returns alpha * A + beta * B
   */
  static ParSparseMat add( RealT alpha, const ParSparseMatView& A, RealT beta, const ParSparseMatView& B );

  /**
   * @brief Temporarily switch Hypre to the requested memory location, invoke f, then restore the previous location
   *
   * When f returns a host mfem::HypreParMatrix*, this helper also updates the returned matrix owner flags so MFEM can
   * release the host arrays correctly.
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
            ensureHostMemory( result );
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

  /**
   * @brief Raw pointer to the wrapped parallel matrix
   */
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
   * The "diag" matrix here is the on-rank (diagonal) CSR block used by HYPRE's ParCSRMatrix representation:
   * it stores only entries whose column indices fall within this rank's locally-owned column range.
   * In other words, `diag` represents the block of the global matrix on rows owned by this rank and columns
   * owned by this rank. Any off-rank couplings belong to the ParCSR "offd" block and are not provided through
   * this argument.
   *
   * The column indices in `diag` are local to this rank's diagonal block (i.e., in `[0, local_num_cols)`),
   * consistent with MFEM's `mfem::HypreParMatrix` constructor that takes a `mfem::SparseMatrix* diag`.
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
   * This is the diagonal (on-rank) CSR block of the HYPRE ParCSR matrix: rows owned by this rank, and columns
   * owned by this rank (as defined by `col_starts`). Entries that couple to off-rank columns are part of the
   * ParCSR "offd" block and are not included in `diag`.
   *
   * The column indices in `diag` are local to this rank's diagonal block (i.e., in `[0, local_num_cols)`).
   *
   * @note The HypreParMatrix will take ownership of the I, J, and Data from diag.
   */
  ParSparseMat( MPI_Comm comm, HYPRE_BigInt global_num_rows, HYPRE_BigInt global_num_cols, HYPRE_BigInt* row_starts,
                HYPRE_BigInt* col_starts, mfem::SparseMatrix&& diag );

  /**
   * @brief Forward arguments to a mfem::HypreParMatrix constructor and take ownership
   *
   * @tparam Args Constructor argument types
   * @param args Constructor arguments
   */
  template <typename... Args>
  explicit ParSparseMat( Args&&... args )
      : ParSparseMatView( nullptr ),
        owned_mat_( createHypreParMatrix<MemorySpace::Host>( std::forward<Args>( args )... ) )
  {
    mat_ = owned_mat_.get();
    ensureHostMemory( mat_ );
  }

  /**
   * @brief Move construct from another owned parallel matrix
   *
   * @param other Source matrix to move from
   */
  ParSparseMat( ParSparseMat&& other ) noexcept;

  /**
   * @brief Move assign from another owned parallel matrix
   *
   * @param other Source matrix to move from
   * @return ParSparseMat& Reference to this
   */
  ParSparseMat& operator=( ParSparseMat&& other ) noexcept;

  /**
   * @brief Copy construction is disabled
   */
  ParSparseMat( const ParSparseMat& ) = delete;
  /**
   * @brief Copy assignment is disabled
   */
  ParSparseMat& operator=( const ParSparseMat& ) = delete;

  /**
   * @brief Access and release ownership of the HypreParMatrix pointer. The caller is now responsible for releasing the
   * memory.
   */
  mfem::HypreParMatrix* release();

  /**
   * @brief Add other into this in place
   */
  ParSparseMat& operator+=( const ParSparseMatView& other );

  /**
   * @brief Subtract other from this in place
   */
  ParSparseMat& operator-=( const ParSparseMatView& other );

  /**
   * @brief Right-multiply this by other in place
   */
  ParSparseMat& operator*=( const ParSparseMatView& other );

  /**
   * @brief Returns a diagonal matrix with the given diagonal value
   *
   * @param comm MPI communicator
   * @param global_size Global size of the matrix (rows and columns)
   * @param row_starts Row partitioning (global offsets)
   * @param diag_val Value for the diagonal entries
   * @param ordered_rows Sorted local row indices used to select which diagonal entries are omitted or retained
   * @param skip_rows If true (default), rows listed in ordered_rows are omitted and all other local diagonal entries
   * are retained. If false, only rows listed in ordered_rows receive diagonal entries.
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
   * @param ordered_rows Sorted local row indices used to select which diagonal entries are omitted or retained
   * @param skip_rows If true (default), rows listed in ordered_rows are omitted and all other local diagonal entries
   * are retained. If false, only rows listed in ordered_rows receive diagonal entries.
   * @return ParSparseMat The constructed diagonal matrix
   */
  static ParSparseMat diagonalMatrix( MPI_Comm comm, HYPRE_BigInt global_size, HYPRE_BigInt* row_starts,
                                      double diag_val, const mfem::Array<int>& ordered_rows = mfem::Array<int>(),
                                      bool skip_rows = true );

 private:
  /**
   * @brief Owning storage for the wrapped parallel matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> owned_mat_;
};

#endif  // #ifdef TRIBOL_USE_MPI

}  // namespace shared

#endif /* SRC_SHARED_MATH_PARSPARSEMAT_HPP_ */
