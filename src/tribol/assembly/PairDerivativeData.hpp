// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Fixed-size derivative payloads and their composition with one pair-function value.
 */
#ifndef SRC_TRIBOL_ASSEMBLY_PAIRDERIVATIVEDATA_HPP_
#define SRC_TRIBOL_ASSEMBLY_PAIRDERIVATIVEDATA_HPP_

#include "tribol/common/MdArray.hpp"

namespace tribol {

/** @brief Own the coordinate Jacobian of a fixed-size pair-function value.
 * @tparam SizeValues Number of scalar function values.
 * @tparam SizeCoords Number of active coordinate entries.
 */
template <IndexT SizeValues, IndexT SizeCoords>
struct PairJacobian {
  MdArray<RealT, SizeValues, SizeCoords> jacobian{};  ///< Value rows followed by coordinate columns.
};

/** @brief Own one scalar function's coordinate Jacobian and Hessian.
 * @tparam SizeCoords Number of active coordinate entries.
 */
template <IndexT SizeCoords>
struct PairJacobianHessian {
  MdArray<RealT, SizeCoords> jacobian{};             ///< One first derivative per coordinate.
  MdArray<RealT, SizeCoords, SizeCoords> hessian{};  ///< Coordinate row and column order.
};

/** @brief Keep a pair function's domain value together with derivatives calculated from that value.
 * @tparam PairValue Pair-function result retaining its domain-specific accessors and quadrature state.
 * @tparam Derivatives Fixed-size derivative payload.
 */
template <typename PairValue, typename Derivatives>
struct DifferentiatedPairValue {
  PairValue pair_value{};     ///< Original pair-function value, quadrature metadata, and activation.
  Derivatives derivatives{};  ///< Derivatives of pair_value.values().
};

/** @brief Pair-function value and its value-by-coordinate Jacobian. */
template <typename PairValue, IndexT SizeCoords>
using PairValueJacobian = DifferentiatedPairValue<PairValue, PairJacobian<PairValue::size_values, SizeCoords>>;

/** @brief Scalar pair-function value and its coordinate Jacobian and Hessian. */
template <typename PairValue, IndexT SizeCoords>
using PairValueJacobianHessian = DifferentiatedPairValue<PairValue, PairJacobianHessian<SizeCoords>>;

}  // namespace tribol

#endif  // SRC_TRIBOL_ASSEMBLY_PAIRDERIVATIVEDATA_HPP_
