// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TESTS_EXECUTIONTEST_HPP_
#define SRC_TESTS_EXECUTIONTEST_HPP_

#include <initializer_list>

#include <gtest/gtest.h>

#include "tribol/common/ArrayTypes.hpp"

namespace tribol {
namespace detail {

/** @brief Run the same test with each execution mode enabled in this build */
inline auto enabledExecutionModes()
{
  return testing::Values( ExecutionMode::Sequential
#ifdef TRIBOL_USE_OPENMP
                          ,
                          ExecutionMode::OpenMP
#endif
#ifdef TRIBOL_USE_CUDA
                          ,
                          ExecutionMode::Cuda
#endif
#ifdef TRIBOL_USE_HIP
                          ,
                          ExecutionMode::Hip
#endif
  );
}

/** @brief Keep test inputs in memory accessible to the selected execution mode */
class ExecutionTest : public testing::TestWithParam<ExecutionMode> {
 protected:
  MemorySpace memorySpace() const
  {
#ifdef TRIBOL_USE_UMPIRE
    if ( isOnDevice( GetParam() ) ) return MemorySpace::Device;
#endif
    return MemorySpace::Host;
  }

  int allocatorId() const { return getResourceAllocatorID( memorySpace() ); }

  /** @brief Copy values from a host initializer list into the selected memory space */
  template <typename T>
  ArrayT<T> copyToExecutionSpace( std::initializer_list<T> values ) const
  {
    const ArrayT<T> host_values( values, getResourceAllocatorID( MemorySpace::Host ) );
    return ArrayT<T>( host_values, allocatorId() );
  }
};

}  // namespace detail
}  // namespace tribol

#endif /* SRC_TESTS_EXECUTIONTEST_HPP_ */
