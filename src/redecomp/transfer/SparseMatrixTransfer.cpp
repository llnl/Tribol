// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "SparseMatrixTransfer.hpp"

#include <map>

#include "axom/slic.hpp"

#include "redecomp/utils/MPIUtility.hpp"
#include "redecomp/utils/MPIArray.hpp"
#include "redecomp/RedecompMesh.hpp"

namespace redecomp {

SparseMatrixTransfer::SparseMatrixTransfer( const mfem::ParFiniteElementSpace& parent_test_space,
                                            const mfem::ParFiniteElementSpace& parent_trial_space,
                                            const mfem::FiniteElementSpace& redecomp_test_space,
                                            const mfem::FiniteElementSpace& redecomp_trial_space )
    : parent_test_space_{ parent_test_space },
      parent_trial_space_{ parent_trial_space },
      redecomp_test_space_{ redecomp_test_space },
      redecomp_trial_space_{ redecomp_trial_space },
      redecomp_test_mesh_{ getRedecompMesh( redecomp_test_space_ ) },
      redecomp_trial_mesh_{ getRedecompMesh( redecomp_trial_space_ ) }
{
  SLIC_ERROR_ROOT_IF(
      static_cast<const mfem::Mesh*>( &redecomp_test_mesh_.getParent() ) != parent_test_space_.GetMesh(),
      "The parent test finite element space mesh must be linked to the test Redecomp mesh." );
  SLIC_ERROR_ROOT_IF(
      static_cast<const mfem::Mesh*>( &redecomp_trial_mesh_.getParent() ) != parent_trial_space_.GetMesh(),
      "The parent trial finite element space mesh must be linked to the trial Redecomp mesh." );
  SLIC_ERROR_ROOT_IF( &redecomp_test_mesh_.getMPIUtility().MPIComm() != &redecomp_trial_mesh_.getMPIUtility().MPIComm(),
                      "MPI Communicator must match in test and trial spaces." );

  // one value per element for initial implementation
  SLIC_ERROR_ROOT_IF( !parent_test_space_.IsDGSpace() || !parent_trial_space_.IsDGSpace() ||
                          !redecomp_test_space_.IsDGSpace() || !redecomp_trial_space_.IsDGSpace(),
                      "Only DG (L2) spaces are currently supported." );
  SLIC_ERROR_ROOT_IF( parent_test_space_.GetMaxElementOrder() != 0 || parent_trial_space_.GetMaxElementOrder() != 0 ||
                          redecomp_test_space_.GetMaxElementOrder() != 0 ||
                          redecomp_trial_space_.GetMaxElementOrder() != 0,
                      "Only one value per element (zero-th order) is currently supported." );
  SLIC_ERROR_ROOT_IF( parent_test_space_.GetVDim() != 1 || parent_trial_space_.GetVDim() != 1 ||
                          redecomp_test_space_.GetVDim() != 1 || redecomp_trial_space_.GetVDim() != 1,
                      "Only one value per element (zero-th order) is currently supported." );
}

std::unique_ptr<mfem::HypreParMatrix> SparseMatrixTransfer::TransferToParallel( const mfem::SparseMatrix& src ) const
{
  // check sizing on src
  SLIC_ERROR_ROOT_IF( src.Height() != redecomp_test_space_.GetVSize(),
                      "The number of rows in src must match the size of the test FiniteElementSpace." );
  SLIC_ERROR_ROOT_IF( src.Width() != redecomp_trial_space_.GetVSize(),
                      "The number of columns in src must match the size of the trial FiniteElementSpace." );

  // we want a csr matrix.  make sure src is finalized
  SLIC_ERROR_ROOT_IF( !src.Finalized(), "src must be finalized first." );

  auto my_rank = getMPIUtility().MyRank();
  auto n_ranks = getMPIUtility().NRanks();

  // The corresponding parent rank of redecomp mesh elements are ordered, i.e. the parent rank of redecomp element i-1
  // is <= the parent rank of redecomp element i. As a result, the map from redecomp elements to parent ranks only
  // stores element offsets. Elements that lie in the ghost region (ghost elements) are stored as an array of arrays,
  // with the first index corresponding to the parent rank and the nested array holding sorted ghost element indices.

  // package the entries to send to other ranks
  MPIArray<double> recv_matrix_data( &getMPIUtility() );
  MPIArray<int> recv_parent_local_row( &getMPIUtility() );
  MPIArray<int> recv_parent_local_col( &getMPIUtility() );
  MPIArray<int> recv_parent_col_ranks( &getMPIUtility() );
  {
    // values to send to parent ranks
    MPIArray<double> send_matrix_data( &getMPIUtility() );
    // indices in redecomp_mesh_.p2r_elems_ (to obtain parent element id)
    MPIArray<int> send_parent_local_row_offsets( &getMPIUtility() );
    // indices in redecomp_mesh_.p2r_elems_ (to obtain parent element id)
    MPIArray<int> send_parent_local_col_offsets( &getMPIUtility() );
    // parent rank which owns the column (trial) space element. we need this to convert offsets to parent element ids.
    MPIArray<int> send_parent_col_ranks( &getMPIUtility() );
    // guess size and preallocate
    auto approx_size = 2 * src.NumNonZeroElems() / n_ranks;
    for ( int r{ 0 }; r < n_ranks; ++r ) {
      send_matrix_data[r].reserve( approx_size );
      send_parent_local_row_offsets[r].reserve( approx_size );
      send_parent_local_col_offsets[r].reserve( approx_size );
      send_parent_col_ranks[r].reserve( approx_size );
    }

    auto src_I = src.GetI();
    auto src_J = src.GetJ();
    auto src_data = src.GetData();

    auto& test_elem_offsets = redecomp_test_mesh_.getRedecompToParentElemOffsets();
    auto& test_ghost_elems = redecomp_test_mesh_.getRedecompToParentGhostElems();
    auto& trial_elem_offsets = redecomp_trial_mesh_.getRedecompToParentElemOffsets();

    // NOTE: test space = row space
    auto test_parent_rank = 0;
    // tracks index of ghost elements for each rank
    auto test_ghost_ct = 0;
    for ( int i{ 0 }; i < src.Height(); ++i ) {
      const auto test_elem_id = i;  // assumes 1 point per element
      while ( test_elem_id == test_elem_offsets[test_parent_rank + 1] ) {
        ++test_parent_rank;
        // the rank increased, reset the ghost element index
        test_ghost_ct = 0;
      }
      // skip unowned elements
      if ( test_ghost_ct < test_ghost_elems[test_parent_rank].size() &&
           test_elem_id == test_ghost_elems[test_parent_rank][test_ghost_ct] ) {
        // the element is a ghost on this rank. increase the index to check for the next ghost (since they are sorted in
        // ascending order).
        ++test_ghost_ct;
        continue;
      }
      // at this point, we know the element in the row space is own by the redecomp mesh. loop through the column (trial
      // space) data in the SparseMatrix for this row.
      auto trial_parent_rank = 0;
      for ( int Ij{ src_I[i] }; Ij < src_I[i + 1]; ++Ij ) {
        const int j{ src_J[Ij] };
        const auto trial_elem_id = j;  // assumes 1 point per element
        // use the offset data to find the parent rank where this element is stored.
        while ( trial_elem_id >= trial_elem_offsets[trial_parent_rank + 1] ) {
          ++trial_parent_rank;
        }
        // if the entries in J aren't ordered, we might need to go backwards too
        while ( trial_parent_rank > 0 && trial_elem_id < trial_elem_offsets[trial_parent_rank] ) {
          --trial_parent_rank;
        }
        // we have everything now.  package it up into send arrays
        send_matrix_data[test_parent_rank].push_back( src_data[Ij] );
        send_parent_local_row_offsets[test_parent_rank].push_back( test_elem_id - test_elem_offsets[test_parent_rank] );
        send_parent_local_col_offsets[test_parent_rank].push_back( trial_elem_id -
                                                                   trial_elem_offsets[trial_parent_rank] );
        send_parent_col_ranks[test_parent_rank].push_back( trial_parent_rank );
      }
    }
    // convert local col offsets to parent element ids: MPI transfer to the parent rank where the element lives, then
    // use the p2r_elems_ map to convert to an element ID
    for ( int r{ 0 }; r < n_ranks; ++r ) {
      // do per-rank communication to transform local_col from offsets to element ids
      auto send_col_offsets_by_rank = MPIArray<int>( &getMPIUtility() );
      // guess size and preallocate
      auto approx_rank_size = 2 * send_parent_local_col_offsets[r].size() / n_ranks;
      for ( auto& send_col_offsets : send_col_offsets_by_rank ) {
        send_col_offsets.reserve( approx_rank_size );
      }
      for ( int i{ 0 }; i < send_parent_local_col_offsets[r].size(); ++i ) {
        send_col_offsets_by_rank[send_parent_col_ranks[r][i]].push_back( send_parent_local_col_offsets[r][i] );
      }
      auto recv_col_offsets_by_rank = MPIArray<int>( &getMPIUtility() );
      recv_col_offsets_by_rank.SendRecvArrayEach( send_col_offsets_by_rank );
      // convert (now on-rank) offsets to element ids
      for ( int trial_r{ 0 }; trial_r < n_ranks; ++trial_r ) {
        for ( auto& recv_col_offset : recv_col_offsets_by_rank[trial_r] ) {
          recv_col_offset = redecomp_trial_mesh_.getParentToRedecompElems().first[trial_r][recv_col_offset];
        }
      }
      // send back to original rank
      send_col_offsets_by_rank.SendRecvArrayEach( recv_col_offsets_by_rank );
      // fill send_parent_local_col_offsets with local element ids
      axom::Array<int> per_rank_ct( n_ranks, n_ranks );
      for ( int i{ 0 }; i < send_parent_local_col_offsets[r].size(); ++i ) {
        auto col_rank = send_parent_col_ranks[r][i];
        send_parent_local_col_offsets[r][i] = send_col_offsets_by_rank[col_rank][per_rank_ct[col_rank]++];
      }
    }
    // do MPI communication
    recv_matrix_data.SendRecvArrayEach( send_matrix_data );
    // NOTE: these hold offsets now; need to convert these to element numbers
    recv_parent_local_row.SendRecvArrayEach( send_parent_local_row_offsets );
    // NOTE: the send values were converted from offsets to element numbers in the block above
    recv_parent_local_col.SendRecvArrayEach( send_parent_local_col_offsets );
    recv_parent_col_ranks.SendRecvArrayEach( send_parent_col_ranks );
    // convert local row offsets to element numbers
    for ( int r{ 0 }; r < n_ranks; ++r ) {
      for ( auto& row_elem : recv_parent_local_row[r] ) {
        row_elem = redecomp_test_mesh_.getParentToRedecompElems().first[r][row_elem];
      }
    }
  }

  // Now we need to build the CSR data for the hypre diag and offd matrices.  diag holds data with both rows and cols
  // on-rank and offd holds data with rows on-rank and cols off-rank.  the HypreParMatrix constructor that takes
  // diagonal and offdiagonal CSR contributions is used.  see MFEM documentation and hypre documentation on the
  // hypre_ParCSRMatrix for more details.
  auto col_starts = BuildParentElementRankOffsets( redecomp_trial_mesh_ );
  // size i_diag and i_offd, number of columms per row of matrix, and build a global dof to local offd column map
  auto diag_nnz = 0;
  auto offd_nnz = 0;
  // NOTE: these will be owned by the HypreParMatrix. using mfem::Memory, the data will not be deleted upon destruction.
  mfem::Memory<HYPRE_Int> i_diag( parent_test_space_.GetVSize() + 1 );
  mfem::Memory<HYPRE_Int> i_offd( parent_test_space_.GetVSize() + 1 );
  for ( int i{ 0 }; i < parent_test_space_.GetVSize() + 1; ++i ) {
    i_diag[i] = 0;
    i_offd[i] = 0;
  }
  // maps from global j to order of appearance from received data.  this is used to map offdiagonal J values from the
  // local offdiagonal matrix to the global offdiagonal matrix.
  std::map<HYPRE_BigInt, int> cmap_j_offd;
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    for ( int i{ 0 }; i < recv_parent_col_ranks[r].size(); ++i ) {
      auto col_rank = recv_parent_col_ranks[r][i];
      if ( col_rank == my_rank ) {
        ++diag_nnz;
        ++i_diag[recv_parent_local_row[r][i] + 1];
      } else {
        ++offd_nnz;
        ++i_offd[recv_parent_local_row[r][i] + 1];
        cmap_j_offd.insert( std::make_pair( col_starts[col_rank] + recv_parent_local_col[r][i], cmap_j_offd.size() ) );
      }
    }
  }
  // sum up i_diag and i_offd.  the definition of i_diag and i_offd is complete after summation.
  for ( int i{ 0 }; i < parent_test_space_.GetVSize(); ++i ) {
    i_diag[i + 1] += i_diag[i];
    i_offd[i + 1] += i_offd[i];
  }
  SLIC_ASSERT( i_diag[parent_test_space_.GetVSize()] == diag_nnz );
  SLIC_ASSERT( i_offd[parent_test_space_.GetVSize()] == offd_nnz );
  // re-order cmap_j_offd
  {
    auto offd_ct = 0;
    for ( auto& cmap_val : cmap_j_offd ) {
      cmap_val.second = offd_ct++;
    }
  }
  // create cmap, offd column map from local to global element IDs
  mfem::Memory<HYPRE_BigInt> cmap( cmap_j_offd.size() );
  for ( auto& cmap_val : cmap_j_offd ) {
    cmap[cmap_val.second] = cmap_val.first;
  }
  // populate j_diag and j_offd, the columns for each row (row offsets given by i).  will not be sorted (i.e. larger
  // column indices may appear before smaller ones in the same row)
  axom::Array<int> i_diag_ct( parent_test_space_.GetVSize(), parent_test_space_.GetVSize() );
  axom::Array<int> i_offd_ct( parent_test_space_.GetVSize(), parent_test_space_.GetVSize() );
  mfem::Memory<HYPRE_Int> j_diag( diag_nnz );
  mfem::Memory<double> data_diag( diag_nnz );
  mfem::Memory<HYPRE_Int> j_offd( offd_nnz );
  mfem::Memory<double> data_offd( offd_nnz );
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    for ( int i{ 0 }; i < recv_parent_col_ranks[r].size(); ++i ) {
      auto curr_row = recv_parent_local_row[r][i];
      auto curr_col = recv_parent_local_col[r][i];
      auto col_rank = recv_parent_col_ranks[r][i];
      if ( col_rank == my_rank ) {
        j_diag[i_diag[curr_row] + i_diag_ct[curr_row]] = curr_col;
        data_diag[i_diag[curr_row] + i_diag_ct[curr_row]] = recv_matrix_data[r][i];
        ++i_diag_ct[curr_row];
      } else {
        j_offd[i_offd[curr_row] + i_offd_ct[curr_row]] = cmap_j_offd[col_starts[col_rank] + curr_col];
        data_offd[i_offd[curr_row] + i_offd_ct[curr_row]] = recv_matrix_data[r][i];
        ++i_offd_ct[curr_row];
      }
    }
  }
  // create hypre par matrix
  return std::make_unique<mfem::HypreParMatrix>( getMPIUtility().MPIComm(), parent_test_space_.GlobalVSize(),
                                                 parent_trial_space_.GlobalVSize(), parent_test_space_.GetDofOffsets(),
                                                 parent_trial_space_.GetDofOffsets(), i_diag, j_diag, data_diag, i_offd,
                                                 j_offd, data_offd, cmap_j_offd.size(), cmap );
}

const RedecompMesh& SparseMatrixTransfer::getRedecompMesh( const mfem::FiniteElementSpace& fe_space )
{
  auto redecomp_mesh = dynamic_cast<const RedecompMesh*>( fe_space.GetMesh() );
  SLIC_ERROR_ROOT_IF( redecomp_mesh == nullptr, "The finite element space must have a Redecomp mesh." );
  return *redecomp_mesh;
}

axom::Array<HYPRE_BigInt> SparseMatrixTransfer::BuildParentElementRankOffsets( const RedecompMesh& redecomp_mesh )
{
  auto n_ranks = redecomp_mesh.getMPIUtility().NRanks();
  auto my_rank = redecomp_mesh.getMPIUtility().MyRank();
  axom::Array<HYPRE_BigInt> elem_offsets( n_ranks + 1, n_ranks + 1 );
  elem_offsets[my_rank + 1] = redecomp_mesh.getParent().GetNE();
  redecomp_mesh.getMPIUtility().Allreduce( &elem_offsets, MPI_SUM );
  for ( int r{ 2 }; r < elem_offsets.size(); ++r ) {
    elem_offsets[r] += elem_offsets[r - 1];
  }
  return elem_offsets;
}

const MPIUtility& SparseMatrixTransfer::getMPIUtility() const { return redecomp_test_mesh_.getMPIUtility(); }

}  // end namespace redecomp
