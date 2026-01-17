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

void MPIUtility::Send( const mfem::Vector& container, int dest, int tag ) const
{
  const void* data = nullptr;
  if ( container.UseDevice() && mfem::Device::GetGPUAwareMPI() ) {
    data = container.Read();
  } else {
    data = container.HostRead();
  }
  MPI_Send( data, container.Size(), GetMPIType<mfem::real_t>(), dest, tag, comm_ );
}

std::unique_ptr<MPIUtility::Request> MPIUtility::Isend( const mfem::Vector& container, int dest, int tag ) const
{
  auto request = std::make_unique<MPI_Request>();
  const void* data = nullptr;
  if ( container.UseDevice() && mfem::Device::GetGPUAwareMPI() ) {
    data = container.Read();
  } else {
    data = container.HostRead();
  }
  MPI_Isend( data, container.Size(), GetMPIType<mfem::real_t>(), dest, tag, comm_, request.get() );
  return std::make_unique<Request>( std::move( request ) );
}

mfem::Vector MPIUtility::Recv( type<mfem::Vector>, int source, int tag, bool use_device ) const
{
  auto container = mfem::Vector();
  if ( use_device && mfem::Device::GetGPUAwareMPI() ) {
    container.UseDevice( true );
  }
  MPI_Probe( source, tag, comm_, &status_ );
  int count;
  MPI_Get_count( &status_, GetMPIType<mfem::real_t>(), &count );
  container.SetSize( count );
  void* data = nullptr;
  if ( container.UseDevice() && mfem::Device::GetGPUAwareMPI() ) {
    data = container.Write();
  } else {
    data = container.HostWrite();
  }
  MPI_Recv( data, count, GetMPIType<mfem::real_t>(), source, tag, comm_, &status_ );
  return container;
}

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

}  // end namespace redecomp