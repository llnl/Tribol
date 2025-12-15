// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_UTILS_MPIUTILITY_HPP_
#define SRC_REDECOMP_UTILS_MPIUTILITY_HPP_

#include <memory>
#include <type_traits>

#include <mpi.h>

#include "axom/core.hpp"
#include "axom/slic.hpp"

#include "redecomp/utils/BisecTree.hpp"

namespace redecomp {

template <typename T>
struct type {};

/**
 * @brief Wrapper class for MPI functions and communication patterns used in redecomp.
 */
class MPIUtility {
 public:
  /**
   * @brief Construct a new MPIUtility object
   *
   * @param comm MPI_Comm associated with the MPIUtility
   */
  MPIUtility( const MPI_Comm& comm );

  /**
   * @brief Returns the MPI communicator
   *
   * @return MPI Communicator
   */
  const MPI_Comm& MPIComm() const { return comm_; }

  /**
   * @brief Returns the MPI rank of the process
   *
   * @return MPI rank
   */
  int MyRank() const { return my_rank_; }

  /**
   * @brief Returns the total number of MPI ranks
   *
   * @return Number of MPI ranks
   */
  int NRanks() const { return n_ranks_; }

  /**
   * @brief Class to hold/query an MPI_Request
   */
  class Request {
   public:
    /**
     * @brief Construct a new Request object
     *
     * @param request MPI_Request object associated with Request
     */
    Request( std::unique_ptr<MPI_Request> request );

    /**
     * @brief Instructs the process to wait until the request completes
     */
    void Wait();

   private:
    /**
     * @brief MPI_Request object
     */
    std::unique_ptr<MPI_Request> request_;

    /**
     * @brief MPI_Status object
     */
    MPI_Status status_;
  };

  /**
   * @brief Calls MPI_Allreduce on a single value
   *
   * @tparam T Type of the local value: bool, double, int currently supported
   * @param local_value Value on rank
   * @param op MPI operation to perform while reducing
   * @return Reduced value
   */
  template <typename T>
  T AllreduceValue( T local_value, MPI_Op op ) const;

  /**
   * @brief Calls MPI_Allreduce on an array stored in container
   *
   * @tparam T Data type of the data container
   * @param container Stores the data to reduce; must have T::data() and T::size() methods
   * @param op MPI operation to perform while reducing
   */
  template <typename T>
  void Allreduce( T* container, MPI_Op op ) const;

  /**
   * @brief Calls MPI_Allreduce on an array pointed to by data
   *
   * @tparam T Data type of an element in the array: bool, double, int currently supported
   * @param data Pointer to first element in the array
   * @param size Number of elements in the array
   * @param op MPI operation to perform while reducing
   */
  template <typename T>
  void Allreduce( T* data, int size, MPI_Op op ) const;

  /**
   * @brief Calls MPI_Send on an array stored in container
   *
   * @tparam T Data type of the data container
   * @param container Stores the data to send; must have T::data() and T::size() methods
   * @param dest Rank to send the data in container to
   * @param tag MPI tag for identifying the data
   */
  template <typename T>
  void Send( const T& container, int dest, int tag = 0 ) const;

  /**
   * @brief Calls MPI_Send on an 2D array stored in an axom::Array
   *
   * @tparam T Data type of an element in the array: bool, double, int currently supported
   * @tparam Sp axom::MemorySpace of the axom::Array
   * @param container 2D axom::Array of data to send
   * @param dest Rank to send the data in container to
   * @param tag MPI tag for identifying the data
   */
  template <typename T, axom::MemorySpace Sp>
  void Send( const axom::Array<T, 2, Sp>& container, int dest, int tag = 0 ) const;

  /**
   * @brief Calls MPI_Isend (non-blocking send) on an array stored in container
   *
   * @tparam T Data type of the data container
   * @param container Stores the data to send; must have T::data() and T::size() methods
   * @param dest Rank to send the data in container to
   * @param tag MPI tag for identifying the data
   * @return Request object to track completion of the send
   */
  template <typename T>
  std::unique_ptr<Request> Isend( const T& container, int dest, int tag = 0 ) const;

  /**
   * @brief Calls MPI_Isend (non-blocking send) on a 2D axom::Array
   *
   * @tparam T Data type of an element in the array: bool, double, int currently supported
   * @tparam Sp axom::MemorySpace of the axom::Array
   * @param container 2D axom::Array of data to send
   * @param dest Rank to send the data in container to
   * @param tag MPI tag for identifying the data
   * @return Request object to track completion of the send
   */
  template <typename T, axom::MemorySpace Sp>
  std::unique_ptr<Request> Isend( const axom::Array<T, 2, Sp>& container, int dest, int tag = 0 ) const;

  /**
   * @brief Calls MPI_Recv on an array stored in container
   *
   * @tparam T Data type of the data container
   * @param source MPI rank data is coming from
   * @param tag MPI tag for identifying the data
   * @return Container of type T holding data sent
   */
  template <typename T>
  T Recv( type<T>, int source, int tag = 0 ) const;

  /**
   * @brief Calls MPI_Recv on an 2D array stored in an axom::Array
   *
   * @tparam T Data type of an element in the array: bool, double, int currently supported
   * @tparam Sp axom::MemorySpace of the axom::Array
   * @param source MPI rank data is coming from
   * @param tag MPI tag for identifying the data
   * @return 2D axom::Array holding the data sent
   */
  template <typename T, axom::MemorySpace Sp>
  axom::Array<T, 2, Sp> Recv( type<axom::Array<T, 2, Sp>>, int source, int tag = 0 ) const;

  /**
   * @brief Sends the array stored in container to all other ranks
   *
   * @tparam T Data type of the data container
   * @param container Stores the data to send; must have T::data() and T::size() methods
   */
  template <typename T>
  void SendAll( const T& container ) const;

  /**
   * @brief Receives the array sent by SendAll
   *
   * @tparam T Data type of the data container
   * @param rank Rank the data originated from
   * @return Container of type T holding data sent
   */
  template <typename T>
  T RecvSendAll( type<T>, int rank ) const;

  /**
   * @brief Sends and receives a different array to each rank
   *
   * @tparam T Data type of the data container
   * @tparam F1 Lambda with destination rank as parameter returning container type T
   * @tparam F2 Lambda with container type T and source rank parameters
   * @param build_send Builds a container of type T holding data to be sent to destination rank
   * @param process_recv Process the data received from another rank
   */
  template <typename T, typename F1, typename F2>
  void SendRecvEach( type<T>, F1&& build_send, F2&& process_recv ) const;

 private:
  /**
   * @brief MPI_Comm used to facilitate MPI communications
   */
  const MPI_Comm comm_;

  /**
   * @brief MPI rank of this process
   */
  int my_rank_;

  /**
   * @brief Total number of MPI ranks
   */
  int n_ranks_;

  /**
   * @brief MPI_Status object
   */
  mutable MPI_Status status_;

  /**
   * @brief Builds binary tree describing the sequence of MPI communication
   *
   * @param rank Rank data originates from
   * @return Tree used to describe where data is and needs to be sent in e.g. SendAll
   */
  BisecTree<int> BuildSendTree( int rank ) const;

  /**
   * @brief Get the MPI_Datatype from a type T
   *
   * @tparam T Datatype with a corresponding MPI_Datatype
   * @return MPI_Datatype corresponding to T
   */
  template <typename T>
  MPI_Datatype GetMPIDatatype( T* ) const
  {
    return GetMPIType<typename std::remove_cv<T>::type>();
  }

  /**
   * @brief Get the MPI_Datatype from a type T
   *
   * @tparam T Datatype with a corresponding MPI_Datatype
   * @return MPI_Datatype corresponding to T
   */
  template <typename T>
  MPI_Datatype GetMPIType() const;

  /**
   * @brief Sends data to other ranks once it has been received from SendAll
   *
   * @tparam T Data type of the data container
   * @param container Stores the data to send; must have T::data() and T::size() methods
   * @param send_tree Bisection tree describing where data must be sent
   * @param lvl_it Current level on the send_tree
   * @param node_it Current node on the send_tree
   */
  template <typename T>
  void SendToRest( const T& container, const BisecTree<int>& send_tree, BisecTree<int>::ConstBFSLevelIterator lvl_it,
                   BisecTree<int>::ConstBFSNodeIterator node_it ) const;
};

template <typename T>
T MPIUtility::AllreduceValue( T local_value, MPI_Op op ) const
{
  MPI_Allreduce( MPI_IN_PLACE, &local_value, 1, GetMPIType<T>(), op, comm_ );
  return local_value;
}

template <typename T>
void MPIUtility::Allreduce( T* container, MPI_Op op ) const
{
  Allreduce( container->data(), container->size(), op );
}

template <typename T>
void MPIUtility::Allreduce( T* data, int size, MPI_Op op ) const
{
  MPI_Allreduce( MPI_IN_PLACE, data, size, GetMPIDatatype( data ), op, comm_ );
}

template <typename T>
void MPIUtility::Send( const T& container, int dest, int tag ) const
{
  MPI_Send( container.data(), container.size(), GetMPIDatatype( container.data() ), dest, tag, comm_ );
}

template <typename T, axom::MemorySpace Sp>
void MPIUtility::Send( const axom::Array<T, 2, Sp>& container, int dest, int tag ) const
{
  MPI_Send( container.shape().m_data, 2, GetMPIType<axom::IndexType>(), dest, tag, comm_ );
  MPI_Send( container.data(), container.size(), GetMPIDatatype( container.data() ), dest, tag, comm_ );
}

template <typename T>
std::unique_ptr<MPIUtility::Request> MPIUtility::Isend( const T& container, int dest, int tag ) const
{
  auto request = std::make_unique<MPI_Request>();
  MPI_Isend( container.data(), container.size(), GetMPIDatatype( container.data() ), dest, tag, comm_, request.get() );
  return std::make_unique<Request>( std::move( request ) );
}

template <typename T, axom::MemorySpace Sp>
std::unique_ptr<MPIUtility::Request> MPIUtility::Isend( const axom::Array<T, 2, Sp>& container, int dest,
                                                        int tag ) const
{
  MPI_Send( container.shape().m_data, 2, GetMPIType<axom::IndexType>(), dest, tag, comm_ );
  auto request = std::make_unique<MPI_Request>();
  MPI_Isend( container.data(), container.size(), GetMPIDatatype( container.data() ), dest, tag, comm_, request.get() );
  return std::make_unique<Request>( std::move( request ) );
}

template <typename T>
T MPIUtility::Recv( type<T>, int source, int tag ) const
{
  auto container = T();
  MPI_Probe( source, tag, comm_, &status_ );
  int count;
  MPI_Get_count( &status_, GetMPIDatatype( container.data() ), &count );
  container.reserve( count );
  container.resize( count );
  MPI_Recv( container.data(), count, GetMPIDatatype( container.data() ), source, tag, comm_, &status_ );
  return container;
}

template <typename T, axom::MemorySpace Sp>
axom::Array<T, 2, Sp> MPIUtility::Recv( type<axom::Array<T, 2, Sp>>, int source, int tag ) const
{
  auto container = axom::Array<T, 2, Sp>();
  axom::StackArray<axom::IndexType, 2> dim_size;
  MPI_Recv( dim_size.m_data, 2, GetMPIType<axom::IndexType>(), source, tag, comm_, &status_ );
  container.reserve( dim_size[0] * dim_size[1] );
  container.resize( dim_size[0], dim_size[1] );
  MPI_Recv( container.data(), container.size(), GetMPIDatatype( container.data() ), source, tag, comm_, &status_ );
  return container;
}

template <typename T>
void MPIUtility::SendAll( const T& container ) const
{
  const auto send_tree = BuildSendTree( my_rank_ );
  auto lvl_it = send_tree.begin();
  SendToRest( container, send_tree, lvl_it, send_tree.begin( lvl_it ) );
}

template <typename T>
T MPIUtility::RecvSendAll( type<T>, int rank ) const
{
  SLIC_ERROR_IF( rank == my_rank_, "Send and receive rank are the same." );
  const auto send_tree = BuildSendTree( rank );
  auto lvl_it = --send_tree.end();
  auto node_it = send_tree.begin( lvl_it ) + my_rank_;
  auto it = send_tree.root( lvl_it, node_it );
  while ( *it.second == my_rank_ ) {
    lvl_it = it.first;
    node_it = it.second;
    it = send_tree.root( lvl_it, node_it );
  }
  auto container = Recv( type<T>(), *it.second );
  SendToRest( container, send_tree, lvl_it, node_it );
  return container;
}

template <typename T>
void MPIUtility::SendToRest( const T& container, const BisecTree<int>& send_tree,
                             BisecTree<int>::ConstBFSLevelIterator lvl_it,
                             BisecTree<int>::ConstBFSNodeIterator node_it ) const
{
  auto it = std::make_pair( lvl_it, node_it );
  while ( it.first != --send_tree.end() ) {
    auto left_it = send_tree.left( it );
    auto right_it = send_tree.right( it );
    if ( *left_it.second == *it.second ) {
      it = left_it;
      if ( right_it.second != send_tree.end( right_it.first ) ) {
        Send( container, *right_it.second );
      }
    } else {
      it = right_it;
      Send( container, *left_it.second );
    }
  }
}

template <typename T, typename F1, typename F2>
void MPIUtility::SendRecvEach( type<T>, F1&& build_send, F2&& process_recv ) const
{
  for ( int i{ 1 }; i < n_ranks_; ++i ) {
    // compute which rank we are sending and receiving data to
    auto dest = ( my_rank_ + i ) % n_ranks_;
    auto source = ( my_rank_ + n_ranks_ - i ) % n_ranks_;

    // build data (note data must be stored somewhere until the send completes)
    auto data = build_send( dest );

    // build and send data; return immediately
    auto request = Isend( data, dest );

    // receive data and process
    process_recv( Recv( type<T>(), source ), source );

    // wait for send to complete
    request->Wait();
  }
  // process on-rank data (no communication)
  process_recv( build_send( my_rank_ ), my_rank_ );
}

}  // end namespace redecomp

#endif /* SRC_REDECOMP_UTILS_MPIUTILITY_HPP_ */
