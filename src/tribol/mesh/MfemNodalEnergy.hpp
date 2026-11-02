// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Differentiate nodal energies after assembling gap and area integrals on dual true degrees of freedom.
 */
#ifndef SRC_TRIBOL_MESH_MFEMNODALENERGY_HPP_
#define SRC_TRIBOL_MESH_MFEMNODALENERGY_HPP_

#include "tribol/integ/NodalIntegralExpressions.hpp"
#include "tribol/mesh/MfemNodalAssembler.hpp"
#include "tribol/physics/PairEvaluation.hpp"

namespace tribol {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
/**
 * @brief Use Enzyme to evaluate a nodal energy expression and its first and second assembled-integral derivatives.
 *
 * Values and coordinate Jacobians remain in quantities. The returned partial derivatives supply both the outer
 * chain-rule products and the weights for pair coordinate Hessians in MfemNodalAssembler. This function does not
 * generate quadrature or assemble the displacement residual and Jacobian. Energy is summed across the MPI communicator.
 * @tparam Field Contracted nodal field expression.
 * @param energy Deferred energy, including its complete scalar scaling and any borrowed independent fields.
 * @param quantities Assembled weighted-gap and area values in local dual true-DOF order.
 * @param assembler Allocates derivative vectors on the corresponding dual space.
 * @param area_tolerance Finite nonnegative support threshold for reciprocal-area energies; unsupported nodes return
 * zero energy and zero first and second partial derivatives.
 * @return Owned nodal partial derivatives and global energy; borrowed expressions and inputs are not retained.
 * @throws std::invalid_argument Local field lengths or the area tolerance are invalid.
 * @pre All ranks call collectively with integral vectors on the assembler's dual space and matching field partitions.
 */
template <typename Field>
MfemNodalDerivativeData evaluateNodalEnergy( const NodalEnergyExpression<Field>& energy,
                                             const MfemNodalQuantityData& quantities,
                                             const MfemNodalAssembler& assembler, RealT area_tolerance )
{
  using Expression = NodalEnergyExpression<Field>;
  const int num_dofs = quantities.weighted_gap.size();
  if ( quantities.area.size() != num_dofs || !std::isfinite( area_tolerance ) || area_tolerance < 0.0 ) {
    throw std::invalid_argument(
        "Nodal integrals must have matching lengths and a finite nonnegative area tolerance." );
  }
  energy.validateSize( num_dofs );
  auto data = assembler.makeDerivativeData( Expression::has_area_derivatives, Expression::has_outer_hessian );
  if ( data.pressure.size() != num_dofs ) {
    throw std::invalid_argument( "Nodal integrals must match the assembler's local dual true-DOF count." );
  }
  constexpr IndexT gap = static_cast<IndexT>( NodalIntegralQuantity::WeightedGap );
  constexpr IndexT area = static_cast<IndexT>( NodalIntegralQuantity::Area );
  RealT local_energy = 0.0;
  for ( int node = 0; node < num_dofs; ++node ) {
    const MdArray<RealT, nodal_integral_count> state{ quantities.weighted_gap[node], quantities.area[node] };
    auto definition = energy.atNode( node, area_tolerance );
    const auto evaluated = EnergyPairEvaluator<decltype( definition )>::evaluate( state.view(), definition );
    local_energy += evaluated.energy;
    data.pressure[node] = data.d_energy_d_gap[node] = evaluated.residual[gap];
    if constexpr ( Expression::has_area_derivatives ) data.d_energy_d_area[node] = evaluated.residual[area];
    if constexpr ( Expression::has_outer_hessian ) {
      data.d2_energy_dg2[node] = evaluated.jacobian( gap, gap );
      if constexpr ( Expression::has_area_derivatives ) {
        data.d2_energy_dg_da[node] = evaluated.jacobian( gap, area );
        data.d2_energy_da2[node] = evaluated.jacobian( area, area );
      }
    }
  }
  MPI_Allreduce( &local_energy, &data.energy, 1, MPI_DOUBLE, MPI_SUM, quantities.weighted_gap.get().GetComm() );
  return data;
}
#endif
}  // namespace tribol
#endif  // SRC_TRIBOL_MESH_MFEMNODALENERGY_HPP_
