// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_SHARED_MATH_PARVECTOR_HPP_
#define SRC_SHARED_MATH_PARVECTOR_HPP_

#include "shared/config.hpp"

#include <memory>
#include <utility>

#include "mfem.hpp"

namespace shared {

#ifdef TRIBOL_USE_MPI

class ParVector;

/**
 * @brief Non-owning view of a mfem::HypreParVector
 *
 * This class holds a raw pointer to a mfem::HypreParVector and provides algebraic operations.
 * It does not manage the lifetime of the vector.
 */
class ParVectorView {
 public:
  /**
   * @brief Construct from a mfem::HypreParVector pointer
   *
   * @param vec Pointer to the mfem HypreParVector
   */
  ParVectorView( mfem::HypreParVector* vec );

  /**
   * @brief Destroy the view without deleting the wrapped vector
   */
  virtual ~ParVectorView() = default;

  /**
   * @brief Access the underlying mfem::HypreParVector
   */
  mfem::HypreParVector& get() { return *vec_; }

  /**
   * @brief Access the underlying mfem::HypreParVector (const)
   */
  const mfem::HypreParVector& get() const { return *vec_; }

  /**
   * @brief Access underlying vector members via arrow operator
   */
  mfem::HypreParVector* operator->() { return vec_; }

  /**
   * @brief Access underlying vector members via arrow operator (const)
   */
  const mfem::HypreParVector* operator->() const { return vec_; }

  /**
   * @brief Access local vector entry
   */
  mfem::real_t& operator[]( int i ) { return ( *vec_ )[i]; }

  /**
   * @brief Access local vector entry (const)
   */
  const mfem::real_t& operator[]( int i ) const { return ( *vec_ )[i]; }

  /**
   * @brief Sets all entries of the vector to the given value
   *
   * @param val Value to set
   */
  void fill( double val ) { *vec_ = val; }

  /**
   * @brief Returns the local size of the vector
   */
  int size() const { return vec_->Size(); }

  /**
   * @brief Returns the maximum value in the vector
   */
  mfem::real_t max() const { return vec_->Max(); }

  /**
   * @brief Returns the minimum value in the vector
   */
  mfem::real_t min() const { return vec_->Min(); }

  /**
   * @brief Returns the dot product with another vector
   */
  mfem::real_t dot( const ParVectorView& other ) const;

  /**
   * @brief Returns the component-wise product of this and other
   */
  ParVector multiply( const ParVectorView& other ) const;

  /**
   * @brief Returns the component-wise quotient of this divided by other
   *
   * @param tol Sets a tolerance to prevent division by zero
   */
  ParVector divide( const ParVectorView& other, mfem::real_t tol = 1.0e-14 ) const;

  /**
   * @brief Returns the component-wise inverse of this
   *
   * @param tol Sets a tolerance to prevent division by zero
   */
  ParVector inverse( mfem::real_t tol = 1.0e-14 ) const;

  /**
   * @brief Returns lhs + rhs
   */
  friend ParVector operator+( const ParVectorView& lhs, const ParVectorView& rhs );

  /**
   * @brief Returns lhs - rhs
   */
  friend ParVector operator-( const ParVectorView& lhs, const ParVectorView& rhs );

  /**
   * @brief Returns this scaled by s
   */
  ParVector operator*( double s ) const;

  /**
   * @brief Returns s * vec
   */
  friend ParVector operator*( double s, const ParVectorView& vec );

 protected:
  /**
   * @brief Raw pointer to the wrapped parallel vector
   */
  mfem::HypreParVector* vec_;
};

/**
 * @brief Wrapper class for mfem::HypreParVector to provide convenience operators
 *
 * This class owns a mfem::HypreParVector via a unique_ptr and adds support for
 * algebraic operations.
 */
class ParVector : public ParVectorView {
 public:
  /**
   * @brief Construct from a mfem::HypreParVector pointer and take ownership
   *
   * @param vec Pointer to the mfem HypreParVector
   */
  explicit ParVector( mfem::HypreParVector* vec );

  /**
   * @brief Construct from a mfem::HypreParVector pointer and take ownership
   *
   * @param vec Pointer to the mfem HypreParVector
   */
  explicit ParVector( std::unique_ptr<mfem::HypreParVector> vec );

  /**
   * @brief Forward arguments to a mfem::HypreParVector constructor and take ownership
   *
   * @tparam Args Constructor argument types
   * @param args Constructor arguments
   */
  template <typename... Args>
  explicit ParVector( Args&&... args ) : ParVectorView( nullptr ), owned_vec_( nullptr )
  {
    // ParVector is host-only for now.
    HYPRE_MemoryLocation old_hypre_mem_location;
    HYPRE_GetMemoryLocation( &old_hypre_mem_location );
    HYPRE_SetMemoryLocation( HYPRE_MEMORY_HOST );
    owned_vec_ = std::make_unique<mfem::HypreParVector>( std::forward<Args>( args )... );
    // Return hypre's memory location to what it was before
    HYPRE_SetMemoryLocation( old_hypre_mem_location );
    vec_ = owned_vec_.get();
  }

  /**
   * @brief Move construct from another owned parallel vector
   *
   * @param other Source vector to move from
   */
  ParVector( ParVector&& other ) noexcept;

  /**
   * @brief Move assign from another owned parallel vector
   *
   * @param other Source vector to move from
   * @return ParVector& Reference to this
   */
  ParVector& operator=( ParVector&& other ) noexcept;

  /**
   * @brief Copy construct from another owned parallel vector
   *
   * @param other Source vector to copy
   */
  ParVector( const ParVector& other );

  /**
   * @brief Copy construct from a non-const vector
   *
   * @param other Source vector to copy
   */
  ParVector( ParVector& other ) : ParVector( static_cast<const ParVector&>( other ) ) {}

  /**
   * @brief Copy assign from another owned parallel vector
   *
   * @param other Source vector to copy
   * @return ParVector& Reference to this
   */
  ParVector& operator=( const ParVector& other );

  /**
   * @brief Access and release ownership of the HypreParVector pointer. The caller is now responsible for releasing the
   * memory.
   */
  mfem::HypreParVector* release();

  /**
   * @brief Add other into this in place
   */
  ParVector& operator+=( const ParVectorView& other );

  /**
   * @brief Subtract other from this in place
   */
  ParVector& operator-=( const ParVectorView& other );

  /**
   * @brief Scale this by s in place
   */
  ParVector& operator*=( double s );

  /**
   * @brief Multiply this by other component-wise in place
   */
  ParVector& multiplyInPlace( const ParVectorView& other );

  /**
   * @brief Divide this by other component-wise in place
   *
   * @param tol Sets a tolerance to prevent division by zero
   */
  ParVector& divideInPlace( const ParVectorView& other, mfem::real_t tol = 1.0e-14 );

  /**
   * @brief Replace this with its component-wise inverse
   *
   * @param tol Sets a tolerance to prevent division by zero
   */
  ParVector& inverseInPlace( mfem::real_t tol = 1.0e-14 );

 private:
  /**
   * @brief Owning storage for the wrapped parallel vector
   */
  std::unique_ptr<mfem::HypreParVector> owned_vec_;
};

#endif  // #ifdef TRIBOL_USE_MPI

}  // namespace shared

#endif /* SRC_SHARED_MATH_PARVECTOR_HPP_ */
