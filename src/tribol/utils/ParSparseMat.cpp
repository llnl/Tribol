// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/utils/ParSparseMat.hpp"

namespace tribol {

// ParSparseMatView implementations

ParSparseMat operator+( const ParSparseMatView& lhs, const ParSparseMatView& rhs )
{
  mfem::HypreParMatrix* result = mfem::Add( 1.0, *lhs.m_mat, 1.0, *rhs.m_mat );
  return ParSparseMat( result );
}

ParSparseMat operator-( const ParSparseMatView& lhs, const ParSparseMatView& rhs )
{
  mfem::HypreParMatrix* result = mfem::Add( 1.0, *lhs.m_mat, -1.0, *rhs.m_mat );
  return ParSparseMat( result );
}

ParSparseMat ParSparseMatView::operator*( double s ) const
{
  mfem::HypreParMatrix* result = mfem::Add( s, *m_mat, 0.0, *m_mat );
  return ParSparseMat( result );
}

ParSparseMat operator*( const ParSparseMatView& lhs, const ParSparseMatView& rhs )
{
  mfem::HypreParMatrix* result = mfem::ParMult( lhs.m_mat, rhs.m_mat );
  result->CopyRowStarts();
  result->CopyColStarts();
  return ParSparseMat( result );
}

mfem::Vector ParSparseMatView::operator*( const mfem::Vector& x ) const
{
  mfem::Vector y( m_mat->Height() );
  m_mat->Mult( x, y );
  return y;
}

ParSparseMat ParSparseMatView::transpose() const { return ParSparseMat( m_mat->Transpose() ); }

ParSparseMat ParSparseMatView::square() const { return *this * *this; }

ParSparseMat ParSparseMatView::RAP( const ParSparseMatView& P ) const
{
  return ParSparseMat( mfem::RAP( m_mat, P.m_mat ) );
}

ParSparseMat ParSparseMatView::RAP( const ParSparseMatView& A, const ParSparseMatView& P )
{
  return ParSparseMat( mfem::RAP( A.m_mat, P.m_mat ) );
}

ParSparseMat ParSparseMatView::RAP( const ParSparseMatView& R, const ParSparseMatView& A, const ParSparseMatView& P )
{
  return ParSparseMat( mfem::RAP( R.m_mat, A.m_mat, P.m_mat ) );
}

void ParSparseMatView::EliminateRows( const mfem::Array<int>& rows ) { m_mat->EliminateRows( rows ); }

ParSparseMat ParSparseMatView::EliminateCols( const mfem::Array<int>& cols )
{
  return ParSparseMat( m_mat->EliminateCols( cols ) );
}

ParSparseMat operator*( double s, const ParSparseMatView& mat ) { return mat * s; }

mfem::Vector operator*( const mfem::Vector& x, const ParSparseMatView& mat )
{
  mfem::Vector y( mat.m_mat->Width() );
  mat.m_mat->MultTranspose( x, y );
  return y;
}

// ParSparseMat implementations

ParSparseMat::ParSparseMat( mfem::HypreParMatrix* mat ) : ParSparseMatView( mat ), m_owned_mat( mat ) {}

ParSparseMat::ParSparseMat( std::unique_ptr<mfem::HypreParMatrix> mat )
    : ParSparseMatView( mat.get() ), m_owned_mat( std::move( mat ) )
{
}

ParSparseMat::ParSparseMat( MPI_Comm comm, HYPRE_BigInt glob_size, HYPRE_BigInt* row_starts,
                            mfem::SparseMatrix&& diag )
    : ParSparseMatView( nullptr )
{
  m_owned_mat = std::make_unique<mfem::HypreParMatrix>( comm, glob_size, row_starts, &diag );
  m_mat = m_owned_mat.get();
  diag.GetMemoryI().ClearOwnerFlags();
  diag.GetMemoryJ().ClearOwnerFlags();
  diag.GetMemoryData().ClearOwnerFlags();
  m_owned_mat->SetOwnerFlags( -1, m_owned_mat->OwnsOffd(), m_owned_mat->OwnsColMap() );
}

ParSparseMat::ParSparseMat( ParSparseMat&& other ) noexcept
    : ParSparseMatView( other.m_owned_mat.get() ), m_owned_mat( std::move( other.m_owned_mat ) )
{
  other.m_mat = nullptr;
}

ParSparseMat& ParSparseMat::operator=( ParSparseMat&& other ) noexcept
{
  if ( this != &other ) {
    m_owned_mat = std::move( other.m_owned_mat );
    m_mat = m_owned_mat.get();
    other.m_mat = nullptr;
  }
  return *this;
}

mfem::HypreParMatrix* ParSparseMat::release()
{
  m_mat = nullptr;
  return m_owned_mat.release();
}

ParSparseMat& ParSparseMat::operator+=( const ParSparseMatView& other )
{
  *this = *this + other;
  return *this;
}

ParSparseMat& ParSparseMat::operator-=( const ParSparseMatView& other )
{
  *this = *this - other;
  return *this;
}

ParSparseMat& ParSparseMat::operator*=( const ParSparseMatView& other )
{
  *this = *this * other;
  return *this;
}

ParSparseMat ParSparseMat::diagonalMatrix( MPI_Comm comm, HYPRE_BigInt global_size,
                                           const mfem::Array<HYPRE_BigInt>& row_starts, double diag_val,
                                           const mfem::Array<int>& ordered_rows, bool skip_rows )
{
  int num_local_rows = 0;
  if ( HYPRE_AssumedPartitionCheck() ) {
    num_local_rows = static_cast<int>( row_starts[1] - row_starts[0] );
  } else {
    int rank;
    MPI_Comm_rank( comm, &rank );
    num_local_rows = static_cast<int>( row_starts[rank + 1] - row_starts[rank] );
  }

  int num_ordered_rows = ordered_rows.Size();
  int num_diag_entries = skip_rows ? ( num_local_rows - num_ordered_rows ) : num_ordered_rows;

  mfem::Array<int> rows( num_local_rows + 1 );
  mfem::Array<int> cols( num_diag_entries );
  rows[0] = 0;

  int diag_entry_ct = 0;
  int ordered_idx = 0;
  for ( int i{ 0 }; i < num_local_rows; ++i ) {
    bool is_ordered = ( ordered_idx < num_ordered_rows && ordered_rows[ordered_idx] == i );
    bool add_entry = skip_rows ? !is_ordered : is_ordered;

    if ( add_entry ) {
      cols[diag_entry_ct] = i;
      ++diag_entry_ct;
    }
    rows[i + 1] = diag_entry_ct;

    if ( is_ordered ) {
      ++ordered_idx;
    }
  }

  rows.GetMemory().SetHostPtrOwner( false );
  cols.GetMemory().SetHostPtrOwner( false );
  mfem::Vector vals( num_diag_entries );
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

ParSparseMat ParSparseMat::diagonalMatrix( MPI_Comm comm, HYPRE_BigInt global_size, HYPRE_BigInt* row_starts,
                                           double diag_val, const mfem::Array<int>& ordered_rows, bool skip_rows )
{
  int num_procs;
  MPI_Comm_size( comm, &num_procs );
  int n_row_starts = HYPRE_AssumedPartitionCheck() ? 3 : num_procs + 1;
  mfem::Array<HYPRE_BigInt> row_starts_array( row_starts, n_row_starts );
  return diagonalMatrix( comm, global_size, row_starts_array, diag_val, ordered_rows, skip_rows );
}

}  // namespace tribol