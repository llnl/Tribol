// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/utils/ParSparseMat.hpp"
#include <HYPRE_utilities.h>

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

ParSparseMat::ParSparseMat( MPI_Comm comm, HYPRE_BigInt glob_size, HYPRE_BigInt* row_starts, mfem::SparseMatrix&& diag )
    : ParSparseMatView( nullptr )
{
  // ParSparseMat is host only now. Make sure CSR data is copied on host in the constructor.
  HYPRE_MemoryLocation old_hypre_mem_location;
  HYPRE_GetMemoryLocation(&old_hypre_mem_location);
  HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
  m_owned_mat = std::make_unique<mfem::HypreParMatrix>( comm, glob_size, row_starts, &diag );
  m_mat = m_owned_mat.get();
  diag.GetMemoryI().ClearOwnerFlags();
  diag.GetMemoryJ().ClearOwnerFlags();
  diag.GetMemoryData().ClearOwnerFlags();
  auto mfem_owned_arrays = 3;
  m_owned_mat->SetOwnerFlags( mfem_owned_arrays, m_owned_mat->OwnsOffd(), m_owned_mat->OwnsColMap() );
  // Return hypre's memory location to what it was before
  HYPRE_SetMemoryLocation(old_hypre_mem_location);
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

  // NOTE: mfem::HypreParMatrix(MPI_Comm, HYPRE_BigInt, HYPRE_BigInt*, SparseMatrix*) does not take ownership
  // of the provided CSR arrays (see MFEM docs). To avoid dangling pointers and allocator mismatches, build
  // the ParCSR data using the HypreParMatrix constructor that takes ownership of raw arrays allocated with
  // new[].

  const int num_ordered_rows = ordered_rows.Size();
  if ( num_local_rows < 0 ) {
    num_local_rows = 0;
  }

  // Count selected diagonal entries in a first pass (do not rely on ordered_rows being unique/in-range).
  HYPRE_Int num_diag_entries = 0;
  int ordered_idx = 0;
  for ( int i = 0; i < num_local_rows; ++i ) {
    while ( ordered_idx < num_ordered_rows && ordered_rows[ordered_idx] < i ) {
      ++ordered_idx;
    }
    const bool is_in_list = ( ordered_idx < num_ordered_rows && ordered_rows[ordered_idx] == i );
    const bool add_entry = skip_rows ? !is_in_list : is_in_list;
    if ( add_entry ) {
      ++num_diag_entries;
    }
  }

  auto* diag_i = new HYPRE_Int[num_local_rows + 1];
  auto* diag_j = ( num_diag_entries > 0 ) ? new HYPRE_Int[num_diag_entries] : nullptr;
  auto* diag_data = ( num_diag_entries > 0 ) ? new mfem::real_t[num_diag_entries] : nullptr;

  // No off-diagonal entries for a purely diagonal matrix.
  auto* offd_i = new HYPRE_Int[num_local_rows + 1];
  auto* offd_j = static_cast<HYPRE_Int*>( nullptr );
  auto* offd_data = static_cast<mfem::real_t*>( nullptr );
  auto* offd_col_map = static_cast<HYPRE_BigInt*>( nullptr );

  diag_i[0] = 0;
  for ( int i = 0; i < num_local_rows + 1; ++i ) {
    offd_i[i] = 0;
  }

  HYPRE_Int diag_entry_ct = 0;
  ordered_idx = 0;
  for ( int i = 0; i < num_local_rows; ++i ) {
    while ( ordered_idx < num_ordered_rows && ordered_rows[ordered_idx] < i ) {
      ++ordered_idx;
    }
    const bool is_in_list = ( ordered_idx < num_ordered_rows && ordered_rows[ordered_idx] == i );
    const bool add_entry = skip_rows ? !is_in_list : is_in_list;

    if ( add_entry ) {
      diag_j[diag_entry_ct] = static_cast<HYPRE_Int>( i );
      diag_data[diag_entry_ct] = diag_val;
      ++diag_entry_ct;
    }
    diag_i[i + 1] = diag_entry_ct;
  }

  // copy row_starts to a new array
  mfem::Array<HYPRE_BigInt> row_starts_copy = row_starts;
  auto diag_hpm = std::make_unique<mfem::HypreParMatrix>( comm, global_size, global_size, row_starts_copy.GetData(),
                                                          row_starts_copy.GetData(), diag_i, diag_j, diag_data, offd_i,
                                                          offd_j, offd_data, 0, offd_col_map, true );
  diag_hpm->CopyRowStarts();
  diag_hpm->CopyColStarts();
  auto mfem_owned_arrays = 3;
  diag_hpm->SetOwnerFlags( mfem_owned_arrays, diag_hpm->OwnsOffd(), diag_hpm->OwnsColMap() );
  return ParSparseMat( std::move( diag_hpm ) );
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
