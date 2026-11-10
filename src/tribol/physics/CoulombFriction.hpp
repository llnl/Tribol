// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
// SPDX-License-Identifier: (MIT)

/** @file @brief Puso-Laursen mortar quantities and direct Coulomb multiplier enforcement. */
#ifndef SRC_TRIBOL_PHYSICS_COULOMBFRICTION_HPP_
#define SRC_TRIBOL_PHYSICS_COULOMBFRICTION_HPP_

#include "tribol/config.hpp"
#include "tribol/common/MdArray.hpp"

namespace tribol {
/** @brief Coulomb coefficient and numerical scales for weighted three-dimensional mortar constraints. */
struct CoulombFrictionOptions {
  RealT coefficient{ 0.0 };         ///< Constant isotropic friction coefficient; finite and nonnegative.
  RealT normal_scale{ 0.0 };        ///< Required positive scale: pressure divided by length cubed.
  RealT tangential_scale{ 0.0 };    ///< Required positive scale: pressure divided by length cubed.
  RealT area_tolerance{ 1.0e-14 };  ///< Minimum usable nodal overlap area, in length squared.
};
/** @brief Contact classification; successful evaluation does not imply convergence. */
enum class FrictionState
{
  Open,
  Stick,
  Slip
};

namespace detail {
#ifdef TRIBOL_USE_ENZYME
/** @brief Inputs assembled at one multiplier node; all vectors use Cartesian components. */
struct CoulombNodeInput {
  MdArray<RealT, 3> normal_sum{};   ///< Sum of current unit corner normals.
  MdArray<RealT, 3> gap_vector{};   ///< Current unshifted D*x1-M*x2.
  MdArray<RealT, 3> slip_vector{};  ///< Current D*x1_old-M*x2_old minus accepted gap vector.
  MdArray<RealT, 3> traction{};     ///< Cartesian multiplier input.
  RealT area{ 0.0 };                ///< Current nodal overlap area.
};
/** @brief Local values and derivatives; columns are normal sum, gap vector, slip vector, traction, then area. */
struct CoulombNodeResult {
  MdArray<RealT, 3> constraint{};              ///< Cartesian normal-plus-tangential residual.
  MdArray<RealT, 3> normal{};                  ///< Unit current nodal normal, or zero without overlap.
  MdArray<RealT, 3> slip{};                    ///< Equation 20 tangential slip.
  RealT pressure{ 0.0 };                       ///< Compression-positive normal multiplier.
  RealT gap{ 0.0 };                            ///< Weighted normal gap including the optional residual-gap offset.
  FrictionState state{ FrictionState::Open };  ///< Chosen contact-law branch.
  MdArray<RealT, 3, 13> derivative{};  ///< Constraint partial derivatives, holding options and accepted data fixed.
};
/**
 * @brief Evaluate the direct multiplier equations and optionally their Enzyme derivatives.
 * @param input Assembled nodal quantities.
 * @param options Validated coefficient, equation scales, and area tolerance.
 * @param residual_gap Normal gap offset in length units; zero reproduces the paper.
 * @param differentiate Whether to compute derivative.
 * @return Owned calculation; no overlap enforces zero traction. A nonzero residual is a valid calculation.
 * @throws std::invalid_argument Inputs/options are nonfinite or an overlapping node has a degenerate normal.
 */
CoulombNodeResult evaluateCoulombNode( const CoulombNodeInput& input, const CoulombFrictionOptions& options,
                                       RealT residual_gap = 0.0, bool differentiate = true );
/** @brief Unit corner-normal contributions and their coordinate derivatives for one linear face. */
struct CornerNormalResult {
  MdArray<RealT, 12> values{};          ///< Component-major nodal unit corner normals.
  MdArray<RealT, 12, 12> derivative{};  ///< Rows and columns in component-major element order.
};
/**
 * @brief Calculate the corner-normal contributions in Puso-Laursen equation 38.
 * @param coordinates Component-major face coordinates.
 * @param nodes Three or four face nodes.
 * @return Owned contributions and Enzyme derivatives, before nodal summation/normalization.
 * @throws std::invalid_argument Unsupported node count, nonfinite coordinates, or degenerate corner.
 */
CornerNormalResult computeCornerNormals( const RealT* coordinates, int nodes );
#endif

#ifdef BUILD_REDECOMP
class CoulombMortar;
#endif
}  // namespace detail
}  // namespace tribol
#endif
