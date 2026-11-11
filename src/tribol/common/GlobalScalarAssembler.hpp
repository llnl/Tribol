// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Execution-space reduction and MPI assembly for scalar contributions. */
#ifndef SRC_TRIBOL_COMMON_GLOBALSCALARASSEMBLER_HPP_
#define SRC_TRIBOL_COMMON_GLOBALSCALARASSEMBLER_HPP_

#include <mpi.h>
#include <stdexcept>

#include "axom/core/execution/reductions.hpp"
#include "tribol/common/LoopExec.hpp"

namespace tribol {

/** @brief Reduce scalar contributions in their execution space and exchange only the resulting scalar. */
class GlobalScalarAssembler {
 public:
  /**
   * @brief Sum local contributions and then sum the local result over an MPI communicator.
   * @tparam Reader Device-callable `(IndexT) -> RealT` contribution reader.
   * @param execution_mode Execution policy for the local reduction.
   * @param contribution_count Number of local contributions.
   * @param reader Scalar contribution reader.
   * @param communicator Communicator over which local sums are combined.
   * @return Global scalar sum.
   */
  template <typename Reader>
  static RealT sum( ExecutionMode execution_mode, IndexT contribution_count, Reader reader, MPI_Comm communicator )
  {
    RealT local_sum = 0.0;
    switch ( execution_mode ) {
      case ExecutionMode::Sequential:
        local_sum = localSum<ExecutionMode::Sequential>( contribution_count, reader );
        break;
#ifdef TRIBOL_USE_OPENMP
      case ExecutionMode::OpenMP:
        local_sum = localSum<ExecutionMode::OpenMP>( contribution_count, reader );
        break;
#endif
#ifdef TRIBOL_USE_CUDA
      case ExecutionMode::Cuda:
        local_sum = localSum<ExecutionMode::Cuda>( contribution_count, reader );
        break;
#endif
#ifdef TRIBOL_USE_HIP
      case ExecutionMode::Hip:
        local_sum = localSum<ExecutionMode::Hip>( contribution_count, reader );
        break;
#endif
      default:
        throw std::invalid_argument( "Global scalar assembly requires a concrete execution mode." );
    }
    RealT global_sum = 0.0;
    MPI_Allreduce( &local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, communicator );
    return global_sum;
  }

 public:
  /**
   * @brief Reduce local contributions with a compile-time execution policy.
   * @tparam Exec Concrete execution policy.
   * @tparam Reader Device-callable scalar contribution reader.
   * @param contribution_count Number of local contributions.
   * @param reader Scalar contribution reader.
   * @return Local scalar sum copied from the backend reduction object.
   */
  template <ExecutionMode Exec, typename Reader>
  static RealT localSum( IndexT contribution_count, Reader reader )
  {
    using ExecSpace = typename toAxomExecSpace<Exec>::type;
    axom::ReduceSum<ExecSpace, RealT> sum( 0.0 );
    forAllExec<Exec>( contribution_count,
                      [=] TRIBOL_HOST_DEVICE( IndexT contribution ) { sum += reader( contribution ); } );
    return sum.get();
  }
};

}  // namespace tribol

#endif  // SRC_TRIBOL_COMMON_GLOBALSCALARASSEMBLER_HPP_
