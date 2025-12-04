// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "MatrixTransfer.hpp"

#include "axom/slic.hpp"

#include "mfem/general/forall.hpp"

#include "redecomp/RedecompMesh.hpp"

namespace redecomp {

MatrixTransfer::MatrixTransfer( const mfem::ParFiniteElementSpace& parent_test_fes,
                                const mfem::ParFiniteElementSpace& parent_trial_fes,
                                const mfem::FiniteElementSpace& redecomp_test_fes,
                                const mfem::FiniteElementSpace& redecomp_trial_fes )
    : parent_test_fes_{ parent_test_fes },
      parent_trial_fes_{ parent_trial_fes },
      redecomp_test_fes_{ redecomp_test_fes },
      redecomp_trial_fes_{ redecomp_trial_fes }
{
  auto test_redecomp = dynamic_cast<const RedecompMesh*>( redecomp_test_fes_.GetMesh() );
  auto trial_redecomp = dynamic_cast<const RedecompMesh*>( redecomp_trial_fes_.GetMesh() );
  SLIC_ERROR_ROOT_IF( test_redecomp == nullptr, "The Redecomp test finite element space must have a Redecomp mesh." );
  SLIC_ERROR_ROOT_IF( trial_redecomp == nullptr, "The Redecomp trial finite element space must have a Redecomp mesh." );
  SLIC_ERROR_ROOT_IF( &test_redecomp->getParent() != parent_test_fes_.GetParMesh(),
                      "The parent test finite element space mesh must be linked to the test Redecomp mesh." );
  SLIC_ERROR_ROOT_IF( &trial_redecomp->getParent() != parent_trial_fes_.GetParMesh(),
                      "The parent trial finite element space mesh must be linked to the trial Redecomp mesh." );
  SLIC_ERROR_ROOT_IF( &test_redecomp->getMPIUtility().MPIComm() != &trial_redecomp->getMPIUtility().MPIComm(),
                      "MPI Communicator must match in test and trial spaces." );

  trial_r2p_elem_rank_ = buildRedecomp2ParentElemRank( *trial_redecomp, false );
  test_r2p_elem_rank_ = buildRedecomp2ParentElemRank( *test_redecomp, true );
}

std::unique_ptr<mfem::HypreParMatrix> MatrixTransfer::TransferToParallel(
    const axom::Array<int>& test_elem_idx, const axom::Array<int>& trial_elem_idx,
    const axom::Array<mfem::DenseMatrix>& src_elem_mat, bool parallel_assemble ) const
{
  auto J_sparse = TransferToParallelSparse( test_elem_idx, trial_elem_idx, src_elem_mat );
  J_sparse.Finalize();
  return ConvertToHypreParMatrix( J_sparse, parallel_assemble );
}

mfem::SparseMatrix MatrixTransfer::TransferToParallelSparse( const axom::Array<int>& test_elem_idx,
                                                             const axom::Array<int>& trial_elem_idx,
                                                             const axom::Array<mfem::DenseMatrix>& src_elem_mat ) const
{
  // TODO (EBC): we need a SparseMatrix-like data structure that allows HYPRE_BigInt on the columns
  auto parentJ = mfem::SparseMatrix( parent_test_fes_.GetVSize(), parent_trial_fes_.GlobalVSize() );

  // verify inputs
  SLIC_ERROR_IF( test_elem_idx.size() != trial_elem_idx.size() || test_elem_idx.size() != src_elem_mat.size(),
                 "Element index arrays and element Jacobian contribution array must be the same size." );
  for ( int i{ 0 }; i < src_elem_mat.size(); ++i ) {
    auto test_e = test_elem_idx[i];
    auto trial_e = trial_elem_idx[i];

    SLIC_ERROR_IF( test_e < 0, "Invalid primary index value." );
    SLIC_ERROR_IF( trial_e < 0, "Invalid secondary index value." );

    auto n_test_elem_vdofs = redecomp_test_fes_.GetFE( test_e )->GetDof() * redecomp_test_fes_.GetVDim();
    auto n_trial_elem_vdofs = redecomp_trial_fes_.GetFE( trial_e )->GetDof() * redecomp_trial_fes_.GetVDim();

    SLIC_ERROR_IF( src_elem_mat[i].Height() != n_test_elem_vdofs,
                   "The number of test DOFs does not match the size of the element DenseMatrix." );
    SLIC_ERROR_IF( src_elem_mat[i].Width() != n_trial_elem_vdofs,
                   "The number of trial DOFs does not match the size of the element DenseMatrix." );
  }

  auto test_redecomp = dynamic_cast<const RedecompMesh*>( redecomp_test_fes_.GetMesh() );
  auto trial_redecomp = dynamic_cast<const RedecompMesh*>( redecomp_trial_fes_.GetMesh() );

  // List of entries in src_elem_mat that belong on each parent test space rank.
  // This is needed so we know which rank to send entries in src_elem_mat to.
  auto send_array_ids = buildSendArrayIDs( test_elem_idx );
  // Number of matrix entries to be sent to each parent test space rank.  This
  // is used to size the array of element matrix values to be sent to other ranks.
  auto send_num_mat_entries = buildSendNumMatEntries( test_elem_idx, trial_elem_idx );
  // Number of test and trial vdofs received from test space redecomp ranks.
  // This is used to determine the beginning and end of each element matrix
  // received as a single array of values and the beginning and end of vdof indices
  // received as a single array of values.
  auto recv_mat_sizes = buildRecvMatSizes( test_elem_idx, trial_elem_idx );
  // List of test element offsets received from test space redecomp ranks.  The
  // offset is used with the parent to redecomp map (and the redecomp rank
  // received from) to determine the parent element ID.  Parent element ID is
  // used to determine the test vldofs of the element matrix entries received
  // from redecomp ranks.
  auto recv_test_elem_offsets = buildRecvTestElemOffsets( *test_redecomp, test_elem_idx );
  // List of trial element global vdofs corresponding to the element matrix
  // entries received from redecomp ranks.  The second column of recv_mat_sizes
  // determines the offset for each trial element.
  auto recv_trial_elem_dofs = buildRecvTrialElemDofs( *trial_redecomp, test_elem_idx, trial_elem_idx );

  // aggregate dense matrix values, send and assemble
  getMPIUtility().SendRecvEach(
      type<axom::Array<double>>(),
      [&send_array_ids, &send_num_mat_entries, &src_elem_mat]( axom::IndexType dst ) {
        auto send_vals = axom::Array<double>( 0, send_num_mat_entries[dst] );

        for ( auto src_array_idx : send_array_ids[dst] ) {
          send_vals.append(
              axom::ArrayView<double>( src_elem_mat[src_array_idx].Data(),
                                       src_elem_mat[src_array_idx].Width() * src_elem_mat[src_array_idx].Height() ) );
        }

        return send_vals;
      },
      [this, test_redecomp, &parentJ, &recv_mat_sizes, &recv_trial_elem_dofs, &recv_test_elem_offsets](
          axom::Array<double>&& send_vals, axom::IndexType src ) {
        if ( recv_trial_elem_dofs[src].empty() ) {
          return;
        }
        auto trial_dof_ct = 0;
        auto dof_ct = 0;
        // element loop
        for ( int e{ 0 }; e < recv_test_elem_offsets[src].size(); ++e ) {
          auto test_elem_id = test_redecomp->getParentToRedecompElems().first[src][recv_test_elem_offsets[src][e]];
          auto test_elem_dofs = mfem::Array<int>();
          parent_test_fes_.GetElementVDofs( test_elem_id, test_elem_dofs );
          auto trial_elem_dofs =
              mfem::Array<HYPRE_BigInt>( &recv_trial_elem_dofs[src][trial_dof_ct], recv_mat_sizes[src]( e, 1 ) );
          // trial loop
          for ( int j{ 0 }; j < trial_elem_dofs.Size(); ++j ) {
            // test loop
            for ( int i{ 0 }; i < test_elem_dofs.Size(); ++i ) {
              parentJ.Add( test_elem_dofs[i], trial_elem_dofs[j],
                           // send_vals comes from mfem::SparseMatrix (column major)
                           send_vals[dof_ct + i + j * test_elem_dofs.Size()] );
            }
          }
          trial_dof_ct += recv_mat_sizes[src]( e, 1 );
          dof_ct += recv_mat_sizes[src]( e, 0 ) * recv_mat_sizes[src]( e, 1 );
        }
      } );

  return parentJ;
}

std::unique_ptr<mfem::HypreParMatrix> MatrixTransfer::ConvertToHypreParMatrix( mfem::SparseMatrix& sparse,
                                                                               bool parallel_assemble ) const
{
  SLIC_ERROR_IF( sparse.Height() != parent_test_fes_.GetVSize(),
                 "Height of sparse must match number of test ParFiniteElementSpace L-dofs." );
  SLIC_ERROR_IF( sparse.Width() != parent_trial_fes_.GlobalVSize(),
                 "Width of sparse must match number of trial ParFiniteElementSpace global dofs." );

  // TODO (EBC): mfem::SparseMatrix uses int for global column values; this needs to be HYPRE_BigInt to be compatible
  // with mfem::HypreParMatrix. The following copy is a hack to get this working for now, but true support for
  // HYPRE_BigInt needs a data structure for sparse that allows HYPRE_BigInt on the columns (J)
  auto* J_int = sparse.GetJ();
  auto num_J_vals = sparse.NumNonZeroElems();
  mfem::Array<HYPRE_BigInt> J_bigint( num_J_vals );
  auto* J_bigint_write = J_bigint.HostWrite();
  mfem::forall_switch( false, num_J_vals, [=] MFEM_HOST_DEVICE( int i ) { J_bigint_write[i] = J_int[i]; } );
  // update the host pointer
  J_bigint.HostRead();

  auto J_full = std::make_unique<mfem::HypreParMatrix>(
      getMPIUtility().MPIComm(), parent_test_fes_.GetVSize(), parent_test_fes_.GlobalVSize(),
      parent_trial_fes_.GlobalVSize(), sparse.GetI(), J_bigint.GetData(), sparse.GetData(),
      parent_test_fes_.GetDofOffsets(), parent_trial_fes_.GetDofOffsets() );
  if ( !parallel_assemble ) {
    return J_full;
  } else {
    auto J_true = std::unique_ptr<mfem::HypreParMatrix>(
        mfem::RAP( parent_test_fes_.Dof_TrueDof_Matrix(), J_full.get(), parent_trial_fes_.Dof_TrueDof_Matrix() ) );
    return J_true;
  }
}

axom::Array<int> MatrixTransfer::buildRedecomp2ParentElemRank( const RedecompMesh& redecomp, bool mark_ghost )
{
  auto r2p_elem_rank = axom::Array<int>( 0, redecomp.GetNE() );

  const auto& elem_offsets = redecomp.getRedecompToParentElemOffsets();
  for ( int r{ 0 }; r < getMPIUtility().NRanks(); ++r ) {
    r2p_elem_rank.insert( elem_offsets[r], elem_offsets[r + 1] - elem_offsets[r], r );
  }

  if ( mark_ghost ) {
    // mark ghost elements as rank -1
    const auto& ghost_elems = redecomp.getRedecompToParentGhostElems();
    auto ghost_ct = 0;
    auto r = 0;
    for ( int e{ 0 }; e < redecomp.GetNE(); ++e ) {
      if ( r != r2p_elem_rank[e] ) {
        r = r2p_elem_rank[e];
        ghost_ct = 0;
      }
      if ( ghost_ct < ghost_elems[r].size() && ghost_elems[r][ghost_ct] == e ) {
        r2p_elem_rank[e] = -1;
        ++ghost_ct;
      }
    }
  }

  return r2p_elem_rank;
}

MPIArray<int> MatrixTransfer::buildSendArrayIDs( const axom::Array<int>& test_elem_idx ) const
{
  auto send_array_ids = MPIArray<int>( &getMPIUtility() );

  auto n_ranks = getMPIUtility().NRanks();
  auto est_max_elems = 2 * test_elem_idx.size() / n_ranks;
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    send_array_ids[r].reserve( est_max_elems );
  }
  for ( int i{ 0 }; i < test_elem_idx.size(); ++i ) {
    auto test_e = test_elem_idx[i];
    // rank is selected by the test element space
    auto r = test_r2p_elem_rank_[test_e];
    // test element is owned by this rank (since -1 denotes a ghost element)
    if ( r != -1 ) {
      send_array_ids[r].push_back( i );
    }
  }
  for ( auto send_array_ids_rank : send_array_ids ) {
    send_array_ids_rank.shrink();
  }

  return send_array_ids;
}

axom::Array<int> MatrixTransfer::buildSendNumMatEntries( const axom::Array<int>& test_elem_idx,
                                                         const axom::Array<int>& trial_elem_idx ) const
{
  auto n_ranks = getMPIUtility().NRanks();
  auto send_num_mat_entries = axom::Array<int>( n_ranks, n_ranks );

  for ( int i{ 0 }; i < test_elem_idx.size(); ++i ) {
    auto test_e = test_elem_idx[i];
    auto trial_e = trial_elem_idx[i];
    auto n_test_elem_vdofs = redecomp_test_fes_.GetFE( test_e )->GetDof() * redecomp_test_fes_.GetVDim();
    auto n_trial_elem_vdofs = redecomp_trial_fes_.GetFE( trial_e )->GetDof() * redecomp_trial_fes_.GetVDim();
    // rank is selected by the test element space
    auto r = test_r2p_elem_rank_[test_e];
    // test element is owned by this rank (since -1 denotes a ghost element)
    if ( r != -1 ) {
      send_num_mat_entries[r] += n_test_elem_vdofs * n_trial_elem_vdofs;
    }
  }

  return send_num_mat_entries;
}

MPIArray<int, 2> MatrixTransfer::buildRecvMatSizes( const axom::Array<int>& test_elem_idx,
                                                    const axom::Array<int>& trial_elem_idx ) const
{
  auto recv_mat_sizes = MPIArray<int, 2>( &getMPIUtility() );

  // Number of test and trial vdofs for each element matrix to be sent to each
  // parent test space rank.  MPI communication is used to turn this array into
  // recv_mat_sizes.
  auto send_mat_sizes = MPIArray<int, 2>( &getMPIUtility() );
  auto n_ranks = getMPIUtility().NRanks();
  auto est_max_elems = 2 * test_elem_idx.size() / n_ranks;
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    send_mat_sizes[r].reserve( 2 * est_max_elems );
  }
  for ( int i{ 0 }; i < test_elem_idx.size(); ++i ) {
    auto test_e = test_elem_idx[i];
    auto trial_e = trial_elem_idx[i];
    auto n_test_elem_vdofs = redecomp_test_fes_.GetFE( test_e )->GetDof() * redecomp_test_fes_.GetVDim();
    auto n_trial_elem_vdofs = redecomp_trial_fes_.GetFE( trial_e )->GetDof() * redecomp_trial_fes_.GetVDim();
    // rank is selected by the test element space
    auto r = test_r2p_elem_rank_[test_e];
    // test element is owned by this rank (since -1 denotes a ghost element)
    if ( r != -1 ) {
      auto row = send_mat_sizes[r].shape()[0];
      send_mat_sizes[r].resize( row + 1, 2 );
      send_mat_sizes[r]( row, 0 ) = n_test_elem_vdofs;
      send_mat_sizes[r]( row, 1 ) = n_trial_elem_vdofs;
    }
  }
  recv_mat_sizes.SendRecvArrayEach( send_mat_sizes );

  return recv_mat_sizes;
}

MPIArray<int> MatrixTransfer::buildRecvTestElemOffsets( const RedecompMesh& test_redecomp,
                                                        const axom::Array<int>& test_elem_idx ) const
{
  auto recv_test_elem_offsets = MPIArray<int>( &getMPIUtility() );

  // List of test element offsets that will be sent to each parent test space rank.
  // MPI communication is used to turn this array into recv_test_elem_offsets.
  auto send_test_elem_offsets = MPIArray<int>( &getMPIUtility() );
  auto n_ranks = getMPIUtility().NRanks();
  auto est_max_elems = 2 * test_elem_idx.size() / n_ranks;
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    send_test_elem_offsets[r].reserve( est_max_elems );
  }
  const auto& test_r2p_elem_offsets = test_redecomp.getRedecompToParentElemOffsets();
  for ( int i{ 0 }; i < test_elem_idx.size(); ++i ) {
    auto test_e = test_elem_idx[i];
    // rank is selected by the test element space
    auto r = test_r2p_elem_rank_[test_e];
    // test element is owned by this rank (since -1 denotes a ghost element)
    if ( r != -1 ) {
      send_test_elem_offsets[r].push_back( test_e - test_r2p_elem_offsets[r] );
    }
  }
  recv_test_elem_offsets.SendRecvArrayEach( send_test_elem_offsets );

  return recv_test_elem_offsets;
}

MPIArray<HYPRE_BigInt> MatrixTransfer::buildRecvTrialElemDofs( const RedecompMesh& trial_redecomp,
                                                               const axom::Array<int>& test_elem_idx,
                                                               const axom::Array<int>& trial_elem_idx ) const
{
  auto recv_trial_elem_dofs = MPIArray<HYPRE_BigInt>( &getMPIUtility() );

  auto n_ranks = getMPIUtility().NRanks();

  // List of trial element offsets sorted by the parent test space rank and the
  // parent trial space rank it belongs to.  Used to get the trial space parent
  // vdofs (on the parent trial space rank) onto the parent test space rank.
  auto send_trial_elem_offsets = MPIArray<axom::Array<int>>( &getMPIUtility() );
  // Used to order the trial vdofs by element in the same order as received test
  // elements.  Stores the ordered trial rank of the vdofs.
  auto send_trial_elem_rank = MPIArray<int>( &getMPIUtility() );
  // Used to order the trial vdofs by element in the same order as received test
  // elements.  Stores the start index and length of the vdofs for each element.
  auto send_trial_dof_extents = MPIArray<int, 2>( &getMPIUtility() );
  // Total trial element vdofs to be sent to each parent test space rank
  auto send_trial_dof_sizes = axom::Array<int>( n_ranks, n_ranks );
  // Running count of number of DOFs in each send test/trial rank combo.  Used
  // in send_trial_dof_extents.
  auto trial_elem_offsets_dof_ct = MPIArray<int>( &getMPIUtility() );
  // List of trial element global vtdofs corresponding to the element matrix
  // entries to send to parent ranks.  The second column of send_mat_sizes
  // determines the offset for each trial element.  MPI communication is used
  // to turn this array into recv_trial_elem_dofs.
  auto send_trial_elem_dofs = MPIArray<HYPRE_BigInt>( &getMPIUtility() );

  // allocate space for the above arrays
  auto est_max_elems = 2 * test_elem_idx.size() / n_ranks;
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    send_trial_elem_offsets[r].resize( n_ranks );
    send_trial_elem_offsets[r].shrink();
    for ( auto& send_trial_elem_offsets_rank : send_trial_elem_offsets[r] ) {
      send_trial_elem_offsets_rank.reserve( est_max_elems / n_ranks );
    }
    send_trial_elem_rank[r].reserve( est_max_elems );
    send_trial_dof_extents[r].reserve( 2 * est_max_elems );
    trial_elem_offsets_dof_ct[r].resize( n_ranks );
    trial_elem_offsets_dof_ct[r].shrink();
  }

  const auto& trial_r2p_elem_offsets = trial_redecomp.getRedecompToParentElemOffsets();
  // loop over dense matrices and determine what needs to be sent where
  for ( int i{ 0 }; i < test_elem_idx.size(); ++i ) {
    auto test_e = test_elem_idx[i];
    auto trial_e = trial_elem_idx[i];

    auto n_trial_elem_vdofs = redecomp_trial_fes_.GetFE( trial_e )->GetDof() * redecomp_trial_fes_.GetVDim();

    // rank is selected by the test element space
    auto r = test_r2p_elem_rank_[test_e];
    // test element is owned by this rank (since -1 denotes a ghost element)
    if ( r != -1 ) {
      // mark trial elements rank
      auto trial_r = trial_r2p_elem_rank_[trial_e];
      send_trial_elem_offsets[r][trial_r].push_back( trial_e - trial_r2p_elem_offsets[trial_r] );
      send_trial_elem_rank[r].push_back( trial_r );
      auto row = send_trial_dof_extents[r].shape()[0];
      send_trial_dof_extents[r].resize( row + 1, 2 );
      send_trial_dof_extents[r]( row, 0 ) = trial_elem_offsets_dof_ct[r][trial_r];
      send_trial_dof_extents[r]( row, 1 ) = n_trial_elem_vdofs;
      trial_elem_offsets_dof_ct[r][trial_r] += n_trial_elem_vdofs;
      send_trial_dof_sizes[r] += n_trial_elem_vdofs;
    }
  }

  // grab test DOFs from the parent rank and return them to the redecomp rank
  auto unsorted_trial_dofs = axom::Array<axom::Array<axom::Array<HYPRE_BigInt>>>( n_ranks, n_ranks );
  for ( auto& test_dof_rank : unsorted_trial_dofs ) {
    test_dof_rank = axom::Array<axom::Array<HYPRE_BigInt>>( n_ranks, n_ranks );
  }
  const auto& trial_p2r_elems = trial_redecomp.getParentToRedecompElems();
  for ( int dst_test_r{ 0 }; dst_test_r < n_ranks; ++dst_test_r ) {
    // send parent trial element offsets to parent trial rank
    auto recv_trial_elem_offsets = MPIArray<int>( &getMPIUtility() );
    getMPIUtility().SendRecvEach(
        type<axom::Array<int>>(),
        [dst_test_r, &send_trial_elem_offsets]( axom::IndexType dst_trial_r ) {
          return send_trial_elem_offsets[dst_test_r][dst_trial_r];
        },
        [&recv_trial_elem_offsets]( axom::Array<int>&& send_vals, axom::IndexType src_trial_r ) {
          recv_trial_elem_offsets[src_trial_r] = std::move( send_vals );
        } );
    // send parent trial vdofs back to test redecomp rank
    auto first_dof = parent_trial_fes_.GetMyDofOffset();
    auto trial_dofs_by_rank = MPIArray<HYPRE_BigInt>( &getMPIUtility() );
    trial_dofs_by_rank.SendRecvEach(
        [this, &trial_p2r_elems, &recv_trial_elem_offsets, first_dof]( axom::IndexType src_trial_r ) {
          auto trial_dofs = axom::Array<HYPRE_BigInt>();

          const auto& rank_elem_offsets = recv_trial_elem_offsets[src_trial_r];
          auto n_elems = rank_elem_offsets.size();
          if ( n_elems > 0 ) {
            auto elem_id = trial_p2r_elems.first[src_trial_r][rank_elem_offsets[0]];
            auto est_n_dofs = n_elems * parent_trial_fes_.GetFE( elem_id )->GetDof() * parent_trial_fes_.GetVDim();
            trial_dofs.reserve( est_n_dofs );
            for ( auto elem_offset : rank_elem_offsets ) {
              elem_id = trial_p2r_elems.first[src_trial_r][elem_offset];
              auto n_elem_dofs = parent_trial_fes_.GetFE( elem_id )->GetDof() * parent_trial_fes_.GetVDim();
              trial_dofs.resize( trial_dofs.size() + n_elem_dofs );
              auto dof_array = mfem::Array<HYPRE_BigInt>( &trial_dofs[trial_dofs.size() - n_elem_dofs], n_elem_dofs );
              auto int_dof_array = mfem::Array<int>( n_elem_dofs );
              parent_trial_fes_.GetElementVDofs( elem_id, int_dof_array );
              for ( int i{ 0 }; i < n_elem_dofs; ++i ) {
                dof_array[i] = first_dof + static_cast<HYPRE_BigInt>( int_dof_array[i] );
              }
              // NOTE (EBC): this can probably be used when redecomp is GPU-ified
              // auto dof_array_write = dof_array.HostWrite();
              // auto int_dof_array_read = int_dof_array.HostRead();
              // mfem::forall_switch( false, n_elem_dofs,
              //                      [=] MFEM_HOST_DEVICE ( int i ) { dof_array_write[i] = first_dof +
              //                      int_dof_array_read[i]; } );
            }
          }

          return trial_dofs;
        } );
    for ( int dst_trial_r{ 0 }; dst_trial_r < n_ranks; ++dst_trial_r ) {
      unsorted_trial_dofs[dst_test_r][dst_trial_r] = std::move( trial_dofs_by_rank[dst_trial_r] );
    }
  }
  // construct trial vdof vectors for each test rank
  for ( int test_r{ 0 }; test_r < n_ranks; ++test_r ) {
    send_trial_elem_dofs[test_r].reserve( send_trial_dof_sizes[test_r] );
    auto n_trial_elems = send_trial_elem_rank[test_r].size();
    for ( int e{ 0 }; e < n_trial_elems; ++e ) {
      auto trial_elem_dofs = axom::ArrayView<HYPRE_BigInt>(
          &unsorted_trial_dofs[test_r][send_trial_elem_rank[test_r][e]][send_trial_dof_extents[test_r]( e, 0 )],
          { send_trial_dof_extents[test_r]( e, 1 ) } );
      send_trial_elem_dofs[test_r].append( trial_elem_dofs );
    }
  }
  recv_trial_elem_dofs.SendRecvArrayEach( send_trial_elem_dofs );

  return recv_trial_elem_dofs;
}

const MPIUtility& MatrixTransfer::getMPIUtility() const
{
  return static_cast<RedecompMesh*>( redecomp_test_fes_.GetMesh() )->getMPIUtility();
}

}  // end namespace redecomp
