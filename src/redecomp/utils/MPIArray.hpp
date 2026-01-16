// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_UTILS_MPIARRAY_HPP_
#define SRC_REDECOMP_UTILS_MPIARRAY_HPP_

#include "mfem.hpp"

#include "shared/infrastructure/Profiling.hpp"

#include "redecomp/utils/MPIUtility.hpp"

namespace redecomp {

/**
 * @brief Creates and manages per-MPI-rank arrays
 *
 * @tparam T Array data type
 * @tparam ArrayType Array type on each rank
 */
template <typename T, typename ArrayType = mfem::Array<T>>
class MPIArray : public std::vector<ArrayType> {
 public:
  typedef ArrayType ArrayT;
  /**
   * @brief Construct a new MPIArray object
   *
   * @param mpi MPIUtility to define MPI_Comm for MPI operations
   * @param array Array data
   */
  MPIArray( const MPIUtility* mpi, const std::vector<ArrayType>& array ) : std::vector<ArrayType>( array ), mpi_{ mpi }
  {
  }

  /**
   * @brief Construct a new MPIArray object
   *
   * @param mpi MPIUtility to define MPI_Comm for MPI operations
   * @param array Array data
   */
  MPIArray( const MPIUtility* mpi, std::vector<ArrayType>&& array )
      : std::vector<ArrayType>( std::move( array ) ), mpi_{ mpi }
  {
  }

  /**
   * @brief Construct a new MPIArray object
   *
   * @param mpi MPIUtility to define MPI_Comm for MPI operations
   */
  MPIArray( const MPIUtility* mpi ) : MPIArray( mpi, std::vector<ArrayType>( mpi->NRanks() ) ) {}

  /**
   * @brief Construct an empty MPIArray object (note: object cannot be used)
   */
  MPIArray() = default;

  /**
   * @brief Returns the array at the given rank
   *
   * @param rank The MPI rank of the array
   * @return ArrayType reference holding array values at rank
   */
  ArrayType& at( axom::IndexType rank ) { return this->operator[]( rank ); }

  /**
   * @brief Returns the array at the given rank
   *
   * @param rank The MPI rank of the array
   * @return ArrayType reference holding array values at rank
   */
  const ArrayType& at( axom::IndexType rank ) const { return this->operator[]( rank ); }

  /**
   * @brief Sends the Array data to all other MPI ranks
   *
   * @param data Data to send to other ranks
   */
  static void SendAll( const ArrayType& data ) { data.mpi_.SendAll( data ); }

  /**
   * @brief Receive data sent from a call to MPIArray::SendAll()
   *
   * @param src The source rank of the data
   * @param use_device Whether to allocate the received array on device
   */
  void RecvSendAll( axom::IndexType src, bool use_device = false )
  {
    at( src ) = mpi_->RecvSendAll( type<ArrayType>(), src, use_device );
  }

  /**
   * @brief Sends the MPIArray data to all other MPI ranks while receiving from other ranks
   *
   * @param data Data to send to other ranks
   * @param use_device Whether to allocate the received array on device
   */
  void SendRecvArrayEach( const MPIArray& data, bool use_device = false )
  {
    mpi_->SendRecvEach(
        type<ArrayType>(), [data]( axom::IndexType dst ) { return data.at( dst ); },
        [this]( ArrayType&& recv_data, axom::IndexType src ) { at( src ) = std::move( recv_data ); }, use_device );
  }

  /**
   * @brief Create data to send to all other MPI ranks while receiving from other ranks
   *
   * @param build_send A lambda which returns an ArrayType to send to the input rank
   * @param use_device Whether to allocate the received array on device
   */
  template <typename F>
  void SendRecvEach( F&& build_send, bool use_device = false )
  {
    TRIBOL_MARK_FUNCTION;
    mpi_->SendRecvEach( type<ArrayType>(), std::forward<F>( build_send ),
                        [this]( ArrayType&& recv_data, axom::IndexType src ) { at( src ) = std::move( recv_data ); },
                        use_device );
  }

 private:
  /**
   * @brief MPIUtility associated with MPI_Comm of the MPIArray
   */
  const MPIUtility* mpi_;
};

}  // end namespace redecomp

#endif /* SRC_REDECOMP_UTILS_MPIARRAY_HPP_ */
