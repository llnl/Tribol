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
  void Fill( double val ) { *vec_ = val; }

  /**
   * @brief Returns the local size of the vector
   */
  int Size() const { return vec_->Size(); }

  /**
   * @brief Returns the maximum value in the vector
   */
  mfem::real_t Max() const { return vec_->Max(); }

  /**
   * @brief Returns the minimum value in the vector
   */
  mfem::real_t Min() const { return vec_->Min(); }

  /**
   * @brief Component-wise multiplication: returns z[i] = x[i] * y[i]
   */
  ParVector multiply( const ParVectorView& other ) const;

  /**
   * @brief Component-wise division: returns z[i] = x[i] / y[i]
   */
  ParVector divide( const ParVectorView& other ) const;

  /**
   * @brief Vector addition: returns x + y
   */
  friend ParVector operator+( const ParVectorView& lhs, const ParVectorView& rhs );

  /**
   * @brief Vector subtraction: returns x - y
   */
  friend ParVector operator-( const ParVectorView& lhs, const ParVectorView& rhs );

  /**
   * @brief Vector scalar multiplication: returns s * x
   */
  ParVector operator*( double s ) const;

  /**
   * @brief Scalar-Vector multiplication: returns s * x
   */
  friend ParVector operator*( double s, const ParVectorView& vec );

 protected:
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

  /// Template constructor forwarding arguments to mfem::HypreParVector constructor
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

  /// Move constructor
  ParVector( ParVector&& other ) noexcept;

  /// Move assignment
  ParVector& operator=( ParVector&& other ) noexcept;

  /// Copy constructor
  ParVector( const ParVector& other );

  /// Copy constructor (non-const; prevents the template constructor from trying to match this case)
  ParVector( ParVector& other ) : ParVector( static_cast<const ParVector&>( other ) ) {}

  /// Copy assignment
  ParVector& operator=( const ParVector& other );

  /**
   * @brief Access and release ownership of the HypreParVector pointer. The caller is now responsible for releasing the
   * memory.
   */
  mfem::HypreParVector* release();

  /**
   * @brief Vector in-place addition: x += y
   */
  ParVector& operator+=( const ParVectorView& other );

  /**
   * @brief Vector in-place subtraction: x -= y
   */
  ParVector& operator-=( const ParVectorView& other );

  /**
   * @brief Vector in-place multiplication: x *= s
   */
  ParVector& operator*=( double s );

  /**
   * @brief Component-wise in-place multiplication: x[i] *= y[i]
   */
  ParVector& multiplyInPlace( const ParVectorView& other );

  /**
   * @brief Component-wise in-place division: x[i] /= y[i]
   */
  ParVector& divideInPlace( const ParVectorView& other );

 private:
  std::unique_ptr<mfem::HypreParVector> owned_vec_;
};

#endif  // #ifdef TRIBOL_USE_MPI

}  // namespace shared

#endif /* SRC_SHARED_MATH_PARVECTOR_HPP_ */
