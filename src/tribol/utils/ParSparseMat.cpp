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
  mfem::HypreParMatrix* result = mfem::Add( 1.0, *lhs.mat_, 1.0, *rhs.mat_ );
  return ParSparseMat( result );
}

ParSparseMat operator-( const ParSparseMatView& lhs, const ParSparseMatView& rhs )
{
  mfem::HypreParMatrix* result = mfem::Add( 1.0, *lhs.mat_, -1.0, *rhs.mat_ );
  return ParSparseMat( result );
}

ParSparseMat ParSparseMatView::operator*( double s ) const
{
  mfem::HypreParMatrix* result = mfem::Add( s, *mat_, 0.0, *mat_ );
  return ParSparseMat( result );
}

ParSparseMat operator*( const ParSparseMatView& lhs, const ParSparseMatView& rhs )
{
  mfem::HypreParMatrix* result = mfem::ParMult( lhs.mat_, rhs.mat_ );
  result->CopyRowStarts();
  result->CopyColStarts();
  return ParSparseMat( result );
}

ParVector ParSparseMatView::operator*( const ParVectorView& x ) const
{
  ParVector y( *mat_ );
  mat_->Mult( const_cast<mfem::HypreParVector&>( x.get() ), y.get() );
  return y;
}

ParSparseMat ParSparseMatView::transpose() const { return ParSparseMat( mat_->Transpose() ); }

ParSparseMat ParSparseMatView::square() const { return *this * *this; }

ParSparseMat ParSparseMatView::RAP( const ParSparseMatView& P ) const
{
  return ParSparseMat( mfem::RAP( mat_, P.mat_ ) );
}

ParSparseMat ParSparseMatView::RAP( const ParSparseMatView& A, const ParSparseMatView& P )
{
  return ParSparseMat( mfem::RAP( A.mat_, P.mat_ ) );
}

ParSparseMat ParSparseMatView::RAP( const ParSparseMatView& R, const ParSparseMatView& A, const ParSparseMatView& P )
{
  return ParSparseMat( mfem::RAP( R.mat_, A.mat_, P.mat_ ) );
}

void ParSparseMatView::EliminateRows( const mfem::Array<int>& rows ) { mat_->EliminateRows( rows ); }

ParSparseMat ParSparseMatView::EliminateCols( const mfem::Array<int>& cols )
{
  return ParSparseMat( mat_->EliminateCols( cols ) );
}

ParSparseMat operator*( double s, const ParSparseMatView& mat ) { return mat * s; }

ParVector operator*( const ParVectorView& x, const ParSparseMatView& mat )
{
  ParVector y( *mat.mat_, 1 );
  mat.mat_->MultTranspose( const_cast<mfem::HypreParVector&>( x.get() ), y.get() );
  return y;
}

// ParSparseMat implementations

ParSparseMat::ParSparseMat( mfem::HypreParMatrix* mat ) : ParSparseMatView( mat ), owned_mat_( mat ) {}

ParSparseMat::ParSparseMat( std::unique_ptr<mfem::HypreParMatrix> mat )
    : ParSparseMatView( mat.get() ), owned_mat_( std::move( mat ) )
{
}

ParSparseMat::ParSparseMat( MPI_Comm comm, HYPRE_BigInt glob_size, HYPRE_BigInt* row_starts, mfem::SparseMatrix&& diag )
    : ParSparseMatView( nullptr )
{
  // ParSparseMat is host only now. Make sure CSR data is copied on host in the constructor.
  HYPRE_MemoryLocation old_hypre_mem_location;
  HYPRE_GetMemoryLocation( &old_hypre_mem_location );
  HYPRE_SetMemoryLocation( HYPRE_MEMORY_HOST );
  owned_mat_ = std::make_unique<mfem::HypreParMatrix>( comm, glob_size, row_starts, &diag );
  mat_ = owned_mat_.get();
  diag.GetMemoryI().ClearOwnerFlags();
  diag.GetMemoryJ().ClearOwnerFlags();
  diag.GetMemoryData().ClearOwnerFlags();
  auto mfem_owned_arrays = 3;
  owned_mat_->SetOwnerFlags( mfem_owned_arrays, owned_mat_->OwnsOffd(), owned_mat_->OwnsColMap() );
  // Return hypre's memory location to what it was before
  HYPRE_SetMemoryLocation( old_hypre_mem_location );
}

ParSparseMat::ParSparseMat( ParSparseMat&& other ) noexcept
    : ParSparseMatView( other.owned_mat_.get() ), owned_mat_( std::move( other.owned_mat_ ) )
{
  other.mat_ = nullptr;
}

ParSparseMat& ParSparseMat::operator=( ParSparseMat&& other ) noexcept
{
  if ( this != &other ) {
    owned_mat_ = std::move( other.owned_mat_ );
    mat_ = owned_mat_.get();
    other.mat_ = nullptr;
  }
  return *this;
}

mfem::HypreParMatrix* ParSparseMat::release()
{
  mat_ = nullptr;
  return owned_mat_.release();
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
