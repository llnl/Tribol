// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Weight a traction by both elements' basis values and form their signed local residual blocks.
 */
#ifndef SRC_TRIBOL_INTEG_PAIRTRACTION_HPP_
#define SRC_TRIBOL_INTEG_PAIRTRACTION_HPP_

#include "axom/primal/geometry/Vector.hpp"
#include "tribol/common/MdArray.hpp"

namespace tribol {

/**
 * @brief Form basis-weighted traction on element1 and its negative on element2, in element/node/component order.
 *
 * The two bases can have different node counts. Quadrature weights, contact activation, and traction calculation
 * belong to the caller. No inputs are retained and no dynamic allocation is performed. Every returned entry is
 * initialized, including
 * zero entries for zero traction. If both bases sum to one, the two residual blocks sum to equal and opposite forces.
 * @tparam FirstBasis Fixed-size rank-one basis array or view for element1, such as MdArray or MdSpan.
 * @tparam SecondBasis Fixed-size rank-one basis array or view for element2.
 * @tparam Dimension Spatial components of the traction and each node's residual.
 * @param basis1 Basis values on element1 in finite-element node order.
 * @param basis2 Basis values on element2 in finite-element node order.
 * @param traction Traction contributing positively to element1 and negatively to element2.
 * @return Owned local residual before quadrature weighting, with element1's block followed by element2's block.
 */
template <typename FirstBasis, typename SecondBasis, int Dimension>
  requires( FirstBasis::rank() == 1 && SecondBasis::rank() == 1 )
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto pairTraction( const FirstBasis& basis1, const SecondBasis& basis2,
                                                           const axom::primal::Vector<RealT, Dimension>& traction )
{
  constexpr IndexT first_dofs = FirstBasis::size() * Dimension;
  MdArray<RealT, first_dofs + SecondBasis::size() * Dimension> residual;
  for ( IndexT node = 0; node < FirstBasis::size(); ++node ) {
    for ( int component = 0; component < Dimension; ++component ) {
      residual[node * Dimension + component] = basis1[node] * traction[component];
    }
  }
  for ( IndexT node = 0; node < SecondBasis::size(); ++node ) {
    for ( int component = 0; component < Dimension; ++component ) {
      residual[first_dofs + node * Dimension + component] = -basis2[node] * traction[component];
    }
  }
  return residual;
}

}  // namespace tribol
#endif  // SRC_TRIBOL_INTEG_PAIRTRACTION_HPP_
