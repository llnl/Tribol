// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/utils/ParSparseMat.hpp"

namespace tribol {

ParSparseMat::ParSparseMat( mfem::HypreParMatrix* mat ) : m_mat( mat ) {}

ParSparseMat::ParSparseMat( std::unique_ptr<mfem::HypreParMatrix> mat ) : m_mat( std::move( mat ) ) {}

ParSparseMat::ParSparseMat( MPI_Comm comm, HYPRE_BigInt glob_size, HYPRE_BigInt* row_starts, mfem::SparseMatrix&& diag )
{
  m_mat = std::make_unique<mfem::HypreParMatrix>( comm, glob_size, row_starts, &diag );
  diag.GetMemoryI().ClearOwnerFlags();
  diag.GetMemoryJ().ClearOwnerFlags();
  diag.GetMemoryData().ClearOwnerFlags();
  m_mat->SetOwnerFlags( -1, m_mat->OwnsOffd(), m_mat->OwnsColMap() );
}

ParSparseMat ParSparseMat::operator+( const ParSparseMat& other ) const
{
  mfem::HypreParMatrix* result = mfem::Add( 1.0, *m_mat, 1.0, *other.m_mat );
  return ParSparseMat( result );
}

ParSparseMat& ParSparseMat::operator+=( const ParSparseMat& other )
{
  *this = *this + other;
  return *this;
}

ParSparseMat ParSparseMat::operator-( const ParSparseMat& other ) const
{
  mfem::HypreParMatrix* result = mfem::Add( 1.0, *m_mat, -1.0, *other.m_mat );
  return ParSparseMat( result );
}

ParSparseMat& ParSparseMat::operator-=( const ParSparseMat& other )
{
  *this = *this - other;
  return *this;
}

ParSparseMat ParSparseMat::operator*( double s ) const
{
  mfem::HypreParMatrix* result = mfem::Add( s, *m_mat, 0.0, *m_mat );
  return ParSparseMat( result );
}

ParSparseMat ParSparseMat::operator*( const ParSparseMat& other ) const
{
  mfem::HypreParMatrix* result = mfem::ParMult( m_mat.get(), other.m_mat.get() );
  result->CopyRowStarts();
  result->CopyColStarts();
  return ParSparseMat( result );
}

ParSparseMat& ParSparseMat::operator*=( const ParSparseMat& other )
{
  *this = *this * other;
  return *this;
}

mfem::Vector ParSparseMat::operator*( const mfem::Vector& x ) const
{
  mfem::Vector y( m_mat->Height() );
  m_mat->Mult( x, y );
  return y;
}

ParSparseMat ParSparseMat::transpose() const { return ParSparseMat( m_mat->Transpose() ); }

ParSparseMat ParSparseMat::square() const { return *this * *this; }

ParSparseMat ParSparseMat::RAP( const ParSparseMat& P ) const
{
  return ParSparseMat( mfem::RAP( m_mat.get(), P.m_mat.get() ) );
}

ParSparseMat ParSparseMat::RAP( const ParSparseMat& R, const ParSparseMat& A, const ParSparseMat& P )
{
  return ParSparseMat( mfem::RAP( R.m_mat.get(), A.m_mat.get(), P.m_mat.get() ) );
}

ParSparseMat ParSparseMat::diagonalMatrix( MPI_Comm comm, HYPRE_BigInt global_size,
                                           const mfem::Array<HYPRE_BigInt>& row_starts, double diag_val,
                                           const mfem::Array<int>& ordered_zero_val_rows )
{
  int num_local_rows = 0;
  if ( HYPRE_AssumedPartitionCheck() ) {
    num_local_rows = row_starts[1] - row_starts[0];
  } else {
    int rank;
    MPI_Comm_rank( comm, &rank );
    num_local_rows = row_starts[rank + 1] - row_starts[rank];
  }
  int num_zero_val_rows = ordered_zero_val_rows.Size();
  int num_nonzero_val_rows = num_local_rows - num_zero_val_rows;
  mfem::Array<int> rows( num_local_rows + 1 );
  mfem::Array<int> cols( num_nonzero_val_rows );
  rows = 0;
  int zero_row_ct = 0;
  for ( int i{ 0 }; i < num_local_rows; ++i ) {
    if ( zero_row_ct < num_zero_val_rows && ordered_zero_val_rows[zero_row_ct] != i ) {
      ++zero_row_ct;
    } else {
      cols[i - zero_row_ct] = i;
    }
    rows[i + 1] = i + 1 - zero_row_ct;
  }
  rows.GetMemory().SetHostPtrOwner( false );
  cols.GetMemory().SetHostPtrOwner( false );
  mfem::Vector vals( num_nonzero_val_rows );
  vals = diag_val;
  vals.GetMemory().SetHostPtrOwner( false );
  mfem::SparseMatrix inactive_diag( rows.GetData(), cols.GetData(), vals.GetData(), num_local_rows, num_local_rows,
                                    false, false, true );
  // if the size of vals is zero, SparseMatrix creates its own memory which it owns.  explicitly prevent this...
  inactive_diag.SetDataOwner( false );
  // copy row_starts to a new array
  mfem::Array<HYPRE_BigInt> row_starts_copy = row_starts;
  auto mat = std::make_unique<mfem::HypreParMatrix>( comm, global_size, row_starts_copy, &inactive_diag );
  mat->CopyRowStarts();
  mat->SetOwnerFlags( -1, mat->OwnsOffd(), mat->OwnsColMap() );
  return ParSparseMat( std::move( mat ) );
}

void ParSparseMat::EliminateRows( const mfem::Array<int>& rows ) { m_mat->EliminateRows( rows ); }

ParSparseMat ParSparseMat::EliminateCols( const mfem::Array<int>& cols )
{
  return ParSparseMat( m_mat->EliminateCols( cols ) );
}

ParSparseMat operator*( double s, const ParSparseMat& mat ) { return mat * s; }

mfem::Vector operator*( const mfem::Vector& x, const ParSparseMat& mat )
{
  mfem::Vector y( mat.get().Width() );
  mat.get().MultTranspose( x, y );
  return y;
}

}  // namespace tribol