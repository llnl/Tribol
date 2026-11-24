// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_SHARED_MATH_PARVECTOR_HPP_
#define SRC_SHARED_MATH_PARVECTOR_HPP_

#include "shared/config.hpp"
#include "shared/common/BasicTypes.hpp"

#include <memory>
#include <utility>

#include "mfem.hpp"

namespace shared {

#ifdef TRIBOL_USE_MPI

class ParVector;

/** @brief Preferred HYPRE allocation and operation location for an owned parallel vector. */
enum class ParVectorMemory
{
  Host,
  Device
};

namespace detail {
/** @brief Temporarily select the HYPRE allocation location required by one vector operation. */
class HypreMemoryLocationGuard {
 public:
  explicit HypreMemoryLocationGuard( ParVectorMemory memory_space )
  {
    HYPRE_GetMemoryLocation( &previous_location_ );
    HYPRE_SetMemoryLocation( memory_space == ParVectorMemory::Device ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST );
  }

  ~HypreMemoryLocationGuard() { HYPRE_SetMemoryLocation( previous_location_ ); }

  HypreMemoryLocationGuard( const HypreMemoryLocationGuard& ) = delete;
  HypreMemoryLocationGuard& operator=( const HypreMemoryLocationGuard& ) = delete;

 private:
  HYPRE_MemoryLocation previous_location_;
};
}  // namespace detail

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
  void fill( RealT val ) { *vec_ = val; }

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
  ParVector operator*( RealT s ) const;

  /**
   * @brief Returns s * vec
   */
  friend ParVector operator*( RealT s, const ParVectorView& vec );

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
   * @brief Copy an MFEM parallel vector while preserving its preferred host or device location.
   * @param vec Source vector.
   */
  explicit ParVector( const mfem::HypreParVector& vec );

  /** @copydoc ParVector(const mfem::HypreParVector&) */
  explicit ParVector( mfem::HypreParVector& vec ) : ParVector( static_cast<const mfem::HypreParVector&>( vec ) ) {}

  /**
   * @brief Forward arguments to an MFEM parallel-vector constructor in the requested memory space.
   * @tparam Args MFEM constructor argument types.
   * @param memory_space Preferred HYPRE allocation and operation location.
   * @param args MFEM constructor arguments.
   */
  template <typename... Args>
  ParVector( ParVectorMemory memory_space, Args&&... args ) : ParVectorView( nullptr ), owned_vec_( nullptr )
  {
    detail::HypreMemoryLocationGuard memory_guard( memory_space );
    owned_vec_ = std::make_unique<mfem::HypreParVector>( std::forward<Args>( args )... );
    owned_vec_->UseDevice( memory_space == ParVectorMemory::Device );
    vec_ = owned_vec_.get();
  }

  /**
   * @brief Forward arguments to a host MFEM parallel-vector constructor and take ownership.
   *
   * @tparam Args Constructor argument types
   * @param args Constructor arguments
   */
  template <typename... Args>
  explicit ParVector( Args&&... args ) : ParVector( ParVectorMemory::Host, std::forward<Args>( args )... )
  {
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
   * @brief Return the vector's preferred operation location.
   * @return Device when MFEM prefers device access; otherwise Host.
   */
  ParVectorMemory memorySpace() const { return vec_->UseDevice() ? ParVectorMemory::Device : ParVectorMemory::Host; }

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
  ParVector& operator*=( RealT s );

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
