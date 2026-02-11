// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/utils/ParVector.hpp"

#include <_hypre_parcsr_mv.h>

#include "axom/slic.hpp"

#include "tribol/common/BasicTypes.hpp"

namespace tribol {

ParVectorView::ParVectorView( mfem::HypreParVector* vec ) : vec_( vec )
{
}

ParVector::ParVector( mfem::HypreParVector* vec ) : ParVectorView( vec ), owned_vec_( vec ) {}

ParVector::ParVector( std::unique_ptr<mfem::HypreParVector> vec )
    : ParVectorView( vec.get() ), owned_vec_( std::move( vec ) )
{
}

ParVector::ParVector( ParVector&& other ) noexcept
    : ParVectorView( other.owned_vec_.get() ), owned_vec_( std::move( other.owned_vec_ ) )
{
  other.vec_ = nullptr;
}

ParVector& ParVector::operator=( ParVector&& other ) noexcept
{
  if ( this != &other ) {
    owned_vec_ = std::move( other.owned_vec_ );
    vec_ = owned_vec_.get();
    other.vec_ = nullptr;
  }
  return *this;
}

ParVector::ParVector( const ParVector& other ) : ParVectorView( nullptr ), owned_vec_( nullptr )
{
  HYPRE_MemoryLocation old_hypre_mem_location;
  HYPRE_GetMemoryLocation( &old_hypre_mem_location );
  HYPRE_SetMemoryLocation( HYPRE_MEMORY_HOST );
  owned_vec_ = std::make_unique<mfem::HypreParVector>( *other.vec_ );
  HYPRE_SetMemoryLocation( old_hypre_mem_location );
  vec_ = owned_vec_.get();
}

ParVector& ParVector::operator=( const ParVector& other )
{
  if ( this != &other ) {
    HYPRE_MemoryLocation old_hypre_mem_location;
    HYPRE_GetMemoryLocation( &old_hypre_mem_location );
    HYPRE_SetMemoryLocation( HYPRE_MEMORY_HOST );
    owned_vec_ = std::make_unique<mfem::HypreParVector>( *other.vec_ );
    HYPRE_SetMemoryLocation( old_hypre_mem_location );
    vec_ = owned_vec_.get();
  }
  return *this;
}

mfem::HypreParVector* ParVector::release()
{
  vec_ = nullptr;
  return owned_vec_.release();
}

ParVector operator+( const ParVectorView& lhs, const ParVectorView& rhs )
{
  ParVector result( lhs.get() );
  result.get().Add( 1.0, rhs.get() );
  return result;
}

ParVector operator-( const ParVectorView& lhs, const ParVectorView& rhs )
{
  ParVector result( lhs.get() );
  result.get().Add( -1.0, rhs.get() );
  return result;
}

ParVector ParVectorView::operator*( double s ) const
{
  ParVector result( *vec_ );
  result.get() *= s;
  return result;
}

ParVector operator*( double s, const ParVectorView& vec ) { return vec * s; }

ParVector& ParVector::operator+=( const ParVectorView& other )
{
  vec_->Add( 1.0, other.get() );
  return *this;
}

ParVector& ParVector::operator-=( const ParVectorView& other )
{
  vec_->Add( -1.0, other.get() );
  return *this;
}

ParVector& ParVector::operator*=( double s )
{
  *vec_ *= s;
  return *this;
}

ParVector ParVectorView::multiply( const ParVectorView& other ) const
{
  ParVector result( *vec_ );
  result.multiplyInPlace( other );
  return result;
}

ParVector ParVectorView::divide( const ParVectorView& other ) const
{
  ParVector result( *vec_ );
  result.divideInPlace( other );
  return result;
}

ParVector& ParVector::multiplyInPlace( const ParVectorView& other )
{
  SLIC_ASSERT( vec_->Size() == other.get().Size() );
  int n = vec_->Size();
  if ( n > 0 ) {
    bool use_device = vec_->UseDevice() || other.get().UseDevice();
    auto d_vec = vec_->ReadWrite( use_device );
    auto d_other = other.get().Read( use_device );
    mfem::forall_switch( use_device, n, [=] TRIBOL_HOST_DEVICE( int i ) { d_vec[i] *= d_other[i]; } );
  }
  return *this;
}

ParVector& ParVector::divideInPlace( const ParVectorView& other )
{
  SLIC_ASSERT( vec_->Size() == other.get().Size() );
  int n = vec_->Size();
  if ( n > 0 ) {
    bool use_device = vec_->UseDevice() || other.get().UseDevice();
    auto d_vec = vec_->ReadWrite( use_device );
    auto d_other = other.get().Read( use_device );
    mfem::forall_switch( use_device, n, [=] TRIBOL_HOST_DEVICE( int i ) { d_vec[i] /= d_other[i]; } );
  }
  return *this;
}

}  // namespace tribol
