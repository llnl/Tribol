// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "redecomp/utils/MPIUtility.hpp"

#include "redecomp/utils/ArrayUtility.hpp"

namespace redecomp {

MPIUtility::MPIUtility( const MPI_Comm& comm ) : comm_{ comm }
{
  MPI_Comm_size( comm, &n_ranks_ );
  MPI_Comm_rank( comm, &my_rank_ );
}

MPIUtility::Request::Request( std::unique_ptr<MPI_Request> request ) : request_{ std::move( request ) } {}

void MPIUtility::Request::Wait() { MPI_Wait( request_.get(), &status_ ); }

BisecTree<int> MPIUtility::BuildSendTree( int rank ) const
{
  auto send_tree = BisecTree<int>( n_ranks_ );
  auto lvl_it = --send_tree.end();
  *lvl_it = ArrayUtility::IndexArray<int>( n_ranks_ );
  while ( lvl_it != send_tree.begin() ) {
    --lvl_it;
    for ( auto node_it = send_tree.begin( lvl_it ); node_it != send_tree.end( lvl_it ); ++node_it ) {
      auto right_node_it = send_tree.right( lvl_it, node_it );
      *node_it = right_node_it.second == send_tree.end( right_node_it.first ) || *right_node_it.second != rank
                     ? *send_tree.left( lvl_it, node_it ).second
                     : rank;
    }
  }
  return send_tree;
}

template <>
MPI_Datatype MPIUtility::GetMPIType<bool>() const
{
  return MPI_C_BOOL;
}

template <>
MPI_Datatype MPIUtility::GetMPIType<double>() const
{
  return MPI_DOUBLE;
}

template <>
MPI_Datatype MPIUtility::GetMPIType<int>() const
{
  return MPI_INT;
}

template <>
MPI_Datatype MPIUtility::GetMPIType<long long>() const
{
  return MPI_LONG_LONG;
}

template <>
MPI_Datatype MPIUtility::GetMPIType<long>() const
{
  return MPI_LONG;
}

}  // end namespace redecomp
