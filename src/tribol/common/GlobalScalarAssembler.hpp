// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Execution-space reduction and MPI assembly for scalar contributions. */
#ifndef SRC_TRIBOL_COMMON_GLOBALSCALARASSEMBLER_HPP_
#define SRC_TRIBOL_COMMON_GLOBALSCALARASSEMBLER_HPP_

#include <mpi.h>
#include <type_traits>

#include "tribol/common/LoopExec.hpp"
#include "tribol/common/Reductions.hpp"

namespace tribol {

/** @brief Reduce scalar contributions in their execution space and exchange only the resulting scalar. */
class GlobalScalarAssembler {
 public:
  /**
   * @brief Sum local contributions and then sum the local result over an MPI communicator.
   * @tparam Reader Device-callable `(IndexT) -> RealT` contribution reader.
   * @tparam Exec Concrete execution mode for the local reduction.
   * @param num_contributions Number of local contributions.
   * @param reader Scalar contribution reader.
   * @param communicator Communicator over which local sums are combined.
   * @return Global scalar sum.
   */
  template <ExecutionMode Exec, typename Reader>
  static RealT sum( IndexT num_contributions, Reader reader, MPI_Comm communicator )
  {
    static_assert( std::is_same_v<RealT, double>, "MPI_DOUBLE requires double-precision RealT." );
    const RealT local_sum = localSum<Exec>( num_contributions, reader );
    RealT global_sum = 0.0;
    MPI_Allreduce( &local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, communicator );
    return global_sum;
  }

  /**
   * @brief Reduce local contributions with a compile-time execution policy.
   * @tparam Exec Concrete execution policy.
   * @tparam Reader Device-callable scalar contribution reader.
   * @param num_contributions Number of local contributions.
   * @param reader Scalar contribution reader.
   * @return Local scalar sum copied from the backend reduction object.
   */
  template <ExecutionMode Exec, typename Reader>
  static RealT localSum( IndexT num_contributions, Reader reader )
  {
    ReduceSum<Exec, RealT> sum( 0.0 );
    forAllExec<Exec>( num_contributions,
                      [=] TRIBOL_HOST_DEVICE( IndexT contribution ) { sum += reader( contribution ); } );
    return sum.get();
  }
};

}  // namespace tribol

#endif  // SRC_TRIBOL_COMMON_GLOBALSCALARASSEMBLER_HPP_
