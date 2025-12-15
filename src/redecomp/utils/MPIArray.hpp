// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_UTILS_MPIARRAY_HPP_
#define SRC_REDECOMP_UTILS_MPIARRAY_HPP_

#include "axom/core.hpp"

#include "shared/infrastructure/Profiling.hpp"

#include "redecomp/utils/MPIUtility.hpp"

namespace redecomp {

/**
 * @brief Creates and manages per-MPI-rank axom::Arrays
 *
 * @tparam T Array data type
 * @tparam DIM Array dimension
 */
template <typename T, int DIM = 1, typename ArrayType = axom::Array<axom::Array<T, DIM>>>
class MPIArray : public ArrayType {
 public:
  /**
   * @brief Construct a new MPIArray object
   *
   * @param mpi MPIUtility to define MPI_Comm for MPI operations
   * @param array Array data
   */
  MPIArray( const MPIUtility* mpi, const ArrayType& array ) : ArrayType( array ), mpi_{ mpi }
  {
    this->reserve( mpi_->NRanks() );
    this->resize( mpi_->NRanks() );
    this->shrink();
  }

  /**
   * @brief Construct a new MPIArray object
   *
   * @param mpi MPIUtility to define MPI_Comm for MPI operations
   * @param array Array data
   */
  MPIArray( const MPIUtility* mpi, ArrayType&& array ) : ArrayType( std::move( array ) ), mpi_{ mpi }
  {
    this->reserve( mpi_->NRanks() );
    this->resize( mpi_->NRanks() );
    this->shrink();
  }

  /**
   * @brief Construct a new MPIArray object
   *
   * @param mpi MPIUtility to define MPI_Comm for MPI operations
   */
  MPIArray( const MPIUtility* mpi ) : MPIArray( mpi, ArrayType( 0, 0 ) ) {}

  /**
   * @brief Construct an empty MPIArray object (note: object cannot be used)
   */
  MPIArray() = default;

  /**
   * @brief Returns the axom::Array at the given rank
   *
   * @param rank The MPI rank of the array
   * @return axom::Array<T, DIM>& holding array values at rank
   */
  axom::Array<T, DIM>& at( axom::IndexType rank ) { return this->operator[]( rank ); }

  /**
   * @brief Returns the axom::Array at the given rank
   *
   * @param rank The MPI rank of the array
   * @return axom::Array<T, DIM>& holding array values at rank
   */
  const axom::Array<T, DIM>& at( axom::IndexType rank ) const { return this->operator[]( rank ); }

  /**
   * @brief Sends the Array data to all other MPI ranks
   *
   * @param data Data to send to other ranks
   */
  static void SendAll( const axom::Array<T, DIM>& data ) { data.mpi_.SendAll( data ); }

  /**
   * @brief Receive data sent from a call to MPIArray::SendAll()
   *
   * @param src The source rank of the data
   */
  void RecvSendAll( axom::IndexType src ) { at( src ) = mpi_->RecvSendAll( type<axom::Array<T, DIM>>(), src ); }

  /**
   * @brief Sends the MPIArray data to all other MPI ranks while receiving from other ranks
   *
   * @param data Data to send to other ranks
   */
  void SendRecvArrayEach( const MPIArray<T, DIM>& data )
  {
    mpi_->SendRecvEach(
        type<axom::Array<T, DIM>>(), [data]( axom::IndexType dst ) { return data.at( dst ); },
        [this]( axom::Array<T, DIM>&& recv_data, axom::IndexType src ) { at( src ) = std::move( recv_data ); } );
  }

  /**
   * @brief Create data to send to all other MPI ranks while receiving from other ranks
   *
   * @param build_send A lambda which returns an axom::Array<T, DIM> to send to the input rank
   */
  template <typename F>
  void SendRecvEach( F&& build_send )
  {
    TRIBOL_MARK_FUNCTION;
    mpi_->SendRecvEach(
        type<axom::Array<T, DIM>>(), std::forward<F>( build_send ),
        [this]( axom::Array<T, DIM>&& recv_data, axom::IndexType src ) { at( src ) = std::move( recv_data ); } );
  }

 private:
  /**
   * @brief MPIUtility associated with MPI_Comm of the MPIArray
   */
  const MPIUtility* mpi_;
};

}  // end namespace redecomp

#endif /* SRC_REDECOMP_UTILS_MPIARRAY_HPP_ */
