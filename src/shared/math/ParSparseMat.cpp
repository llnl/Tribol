// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "shared/math/ParSparseMat.hpp"

#include <_hypre_parcsr_mv.h>

#include "axom/slic.hpp"

namespace shared {

#ifdef TRIBOL_USE_MPI

// ParSparseMatView implementations

ParSparseMatView::ParSparseMatView( mfem::HypreParMatrix* mat ) : mat_( mat ) {}

ParSparseMat operator+( const ParSparseMatView& lhs, const ParSparseMatView& rhs )
{
  return ParSparseMatView::add( 1.0, lhs, 1.0, rhs );
}

ParSparseMat operator-( const ParSparseMatView& lhs, const ParSparseMatView& rhs )
{
  return ParSparseMatView::add( 1.0, lhs, -1.0, rhs );
}

ParSparseMat ParSparseMatView::operator*( double s ) const { return add( s, *this, 0.0, *this ); }

ParSparseMat operator*( const ParSparseMatView& lhs, const ParSparseMatView& rhs )
{
  return ParSparseMat( ParSparseMatView::createHypreParMatrix<MemorySpace::Host>(
      [&]() { return mfem::ParMult( lhs.mat_, rhs.mat_, true ); } ) );
}

ParVector ParSparseMatView::operator*( const ParVectorView& x ) const
{
  ParVector y( *mat_, 1 );
  invokeHypreMethod<MemorySpace::Host>(
      [&]() { mat_->Mult( const_cast<mfem::HypreParVector&>( x.get() ), y.get() ); } );
  return y;
}

ParSparseMat ParSparseMatView::transpose() const
{
  return ParSparseMat( createHypreParMatrix<MemorySpace::Host>( [&]() { return mat_->Transpose(); } ) );
}

ParSparseMat ParSparseMatView::square() const { return *this * *this; }

ParSparseMat ParSparseMatView::RAP( const ParSparseMatView& P ) const
{
  return ParSparseMat( createHypreParMatrix<MemorySpace::Host>( [&]() {
    mat_->HostRead();
    P->HostRead();
    return mfem::RAP( mat_, P.mat_ );
  } ) );
}

ParSparseMat ParSparseMatView::RAP( const ParSparseMatView& A, const ParSparseMatView& P )
{
  return ParSparseMat( createHypreParMatrix<MemorySpace::Host>( [&]() {
    A->HostRead();
    P->HostRead();
    return mfem::RAP( A.mat_, P.mat_ );
  } ) );
}

ParSparseMat ParSparseMatView::RAP( const ParSparseMatView& Rt, const ParSparseMatView& A, const ParSparseMatView& P )
{
  return ParSparseMat( createHypreParMatrix<MemorySpace::Host>( [&]() {
    Rt->HostRead();
    A->HostRead();
    P->HostRead();
    return mfem::RAP( Rt.mat_, A.mat_, P.mat_ );
  } ) );
}

void ParSparseMatView::EliminateRows( const mfem::Array<int>& rows )
{
  invokeHypreMethod<MemorySpace::Host>( [&]() { mat_->EliminateRows( rows ); } );
}

ParSparseMat ParSparseMatView::EliminateCols( const mfem::Array<int>& cols )
{
  return ParSparseMat( createHypreParMatrix<MemorySpace::Host>( [&]() { return mat_->EliminateCols( cols ); } ) );
}

ParSparseMat operator*( double s, const ParSparseMatView& mat ) { return mat * s; }

ParVector operator*( const ParVectorView& x, const ParSparseMatView& mat )
{
  ParVector y( *mat.mat_, 0 );
  ParSparseMatView::invokeHypreMethod<MemorySpace::Host>(
      [&]() { mat.mat_->MultTranspose( const_cast<mfem::HypreParVector&>( x.get() ), y.get() ); } );
  return y;
}

ParSparseMat ParSparseMatView::add( RealT alpha, const ParSparseMatView& A, RealT beta, const ParSparseMatView& B )
{
  return ParSparseMat(
      createHypreParMatrix<MemorySpace::Host>( [&]() { return mfem::Add( alpha, A.get(), beta, B.get() ); } ) );
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
  owned_mat_.reset( createHypreParMatrix<MemorySpace::Host>(
      [&]() { return new mfem::HypreParMatrix( comm, glob_size, row_starts, &diag ); } ) );
  mat_ = owned_mat_.get();
  diag.GetMemoryI().ClearOwnerFlags();
  diag.GetMemoryJ().ClearOwnerFlags();
  diag.GetMemoryData().ClearOwnerFlags();
  // The mfem::Memory in mfem::SparseMatrix allocates using operator new [], so mark the diag memory as owned by MFEM so
  // it can be deleted correctly
  constexpr int mfem_owned_host_flag = 3;
  owned_mat_->SetOwnerFlags( mfem_owned_host_flag, owned_mat_->OwnsOffd(), owned_mat_->OwnsColMap() );
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
  auto diag_hpm = std::unique_ptr<mfem::HypreParMatrix>( createHypreParMatrix<MemorySpace::Host>(
      comm, global_size, global_size, row_starts_copy.GetData(), row_starts_copy.GetData(), diag_i, diag_j, diag_data,
      offd_i, offd_j, offd_data, 0, offd_col_map, true ) );
  diag_hpm->CopyRowStarts();
  diag_hpm->CopyColStarts();
  // We allocated memory using operator new [], so mark all memory as owned by MFEM so it can be deleted correctly
  constexpr int mfem_owned_host_flag = 3;
  diag_hpm->SetOwnerFlags( mfem_owned_host_flag, mfem_owned_host_flag, mfem_owned_host_flag );
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

ParSparseMat ParSparseMat::diagonalMatrix( MPI_Comm comm, HYPRE_BigInt global_size, HYPRE_BigInt* row_starts,
                                           const mfem::Vector& diag_vals )
{
  int num_local_rows = diag_vals.Size();

  mfem::Array<int> rows( num_local_rows + 1 );
  mfem::Array<int> cols( num_local_rows );
  rows[0] = 0;

  for ( int i = 0; i < num_local_rows; ++i ) {
    rows[i + 1] = i + 1;
    cols[i] = i;
  }

  rows.GetMemory().SetHostPtrOwner( false );
  cols.GetMemory().SetHostPtrOwner( false );

  mfem::Vector vals = diag_vals;
  vals.GetMemory().SetHostPtrOwner( false );

  mfem::SparseMatrix inactive_diag( rows.GetData(), cols.GetData(), vals.GetData(), num_local_rows, num_local_rows,
                                    false, false, true );

  return ParSparseMat( comm, global_size, row_starts, std::move( inactive_diag ) );
}

#endif  // #ifdef TRIBOL_USE_MPI

}  // namespace shared
