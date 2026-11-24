// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Parallel reductions selected by Tribol execution mode. */
#ifndef SRC_TRIBOL_COMMON_REDUCTIONS_HPP_
#define SRC_TRIBOL_COMMON_REDUCTIONS_HPP_

#include "axom/core/execution/reductions.hpp"
#include "tribol/common/LoopExec.hpp"

namespace tribol {

/**
 * @brief Find the maximum of an initial value and contributions from a parallel loop.
 *
 * Construct with an initial value, capture by value in a forAllExec<Exec>() loop, and call max(value) for each
 * contribution. After the loop completes, get() returns the maximum on the host. An empty loop returns the initial
 * value. For example, ReduceMax<Exec, int> can collect the most severe quadrature status across element pairs.
 * @tparam Exec Enabled concrete execution mode, matching the loop; Dynamic is unsupported.
 * @tparam T Scalar type being reduced.
 */
template <ExecutionMode Exec, typename T>
using ReduceMax = axom::ReduceMax<typename toAxomExecSpace<Exec>::type, T>;

/**
 * @brief Sum an initial value and contributions from a parallel loop.
 *
 * Construct with an initial value, capture by value in a forAllExec<Exec>() loop, and add contributions with +=.
 * After the loop completes, get() returns the sum on the host. An empty loop returns the initial value.
 * For example, ReduceSum<Exec, RealT> initialized to 0.0 can sum pair energies on the local MPI rank.
 * @tparam Exec Enabled concrete execution mode, matching the loop; Dynamic is unsupported.
 * @tparam T Scalar type being reduced.
 */
template <ExecutionMode Exec, typename T>
using ReduceSum = axom::ReduceSum<typename toAxomExecSpace<Exec>::type, T>;

}  // namespace tribol

#endif  // SRC_TRIBOL_COMMON_REDUCTIONS_HPP_
