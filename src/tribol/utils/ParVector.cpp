// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/utils/ParVector.hpp"

#include "axom/slic.hpp"

namespace tribol {

ParVector::ParVector( mfem::HypreParVector* vec ) : ParVectorView( vec ), m_owned_vec( vec ) {}

ParVector::ParVector( std::unique_ptr<mfem::HypreParVector> vec )
    : ParVectorView( vec.get() ), m_owned_vec( std::move( vec ) )
{
}

ParVector::ParVector( ParVector&& other ) noexcept
    : ParVectorView( other.m_owned_vec.get() ), m_owned_vec( std::move( other.m_owned_vec ) )
{
  other.m_vec = nullptr;
}

ParVector& ParVector::operator=( ParVector&& other ) noexcept
{
  if ( this != &other ) {
    m_owned_vec = std::move( other.m_owned_vec );
    m_vec = m_owned_vec.get();
    other.m_vec = nullptr;
  }
  return *this;
}

ParVector::ParVector( const ParVector& other )
    : ParVectorView( nullptr ), m_owned_vec( std::make_unique<mfem::HypreParVector>( *other.m_vec ) )
{
  m_vec = m_owned_vec.get();
}

ParVector& ParVector::operator=( const ParVector& other )
{
  if ( this != &other ) {
    m_owned_vec = std::make_unique<mfem::HypreParVector>( *other.m_vec );
    m_vec = m_owned_vec.get();
  }
  return *this;
}

mfem::HypreParVector* ParVector::release()
{
  m_vec = nullptr;
  return m_owned_vec.release();
}

ParVector operator+( const ParVectorView& lhs, const ParVectorView& rhs )
{
  ParVector result( new mfem::HypreParVector( lhs.get() ) );
  result.get().Add( 1.0, rhs.get() );
  return result;
}

ParVector operator-( const ParVectorView& lhs, const ParVectorView& rhs )
{
  ParVector result( new mfem::HypreParVector( lhs.get() ) );
  result.get().Add( -1.0, rhs.get() );
  return result;
}

ParVector ParVectorView::operator*( double s ) const
{
  ParVector result( new mfem::HypreParVector( *m_vec ) );
  result.get() *= s;
  return result;
}

ParVector operator*( double s, const ParVectorView& vec ) { return vec * s; }

ParVector& ParVector::operator+=( const ParVectorView& other )
{
  m_vec->Add( 1.0, other.get() );
  return *this;
}

ParVector& ParVector::operator-=( const ParVectorView& other )
{
  m_vec->Add( -1.0, other.get() );
  return *this;
}

ParVector& ParVector::operator*=( double s )
{
  *m_vec *= s;
  return *this;
}

ParVector ParVectorView::multiply( const ParVectorView& other ) const
{
  ParVector result( new mfem::HypreParVector( *m_vec ) );
  result.multiply( other );
  return result;
}

ParVector ParVectorView::divide( const ParVectorView& other ) const
{
  ParVector result( new mfem::HypreParVector( *m_vec ) );
  result.divide( other );
  return result;
}

ParVector& ParVector::multiply( const ParVectorView& other )
{
  SLIC_ASSERT( m_vec->Size() == other.get().Size() );
  int n = m_vec->Size();
  if ( n > 0 ) {
    bool use_device = m_vec->UseDevice() || other.get().UseDevice();
    auto d_vec = m_vec->ReadWrite( use_device );
    auto d_other = other.get().Read( use_device );
    mfem::forall_switch( use_device, n, [=] MFEM_DEVICE( int i ) { d_vec[i] *= d_other[i]; } );
  }
  return *this;
}

ParVector& ParVector::divide( const ParVectorView& other )
{
  SLIC_ASSERT( m_vec->Size() == other.get().Size() );
  int n = m_vec->Size();
  if ( n > 0 ) {
    bool use_device = m_vec->UseDevice() || other.get().UseDevice();
    auto d_vec = m_vec->ReadWrite( use_device );
    auto d_other = other.get().Read( use_device );
    mfem::forall_switch( use_device, n, [=] MFEM_DEVICE( int i ) { d_vec[i] /= d_other[i]; } );
  }
  return *this;
}

}  // namespace tribol
