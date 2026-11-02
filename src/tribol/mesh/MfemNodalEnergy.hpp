// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Differentiate nodal energies with respect to assembled field values on true degrees of freedom.
 */
#ifndef SRC_TRIBOL_MESH_MFEMNODALENERGY_HPP_
#define SRC_TRIBOL_MESH_MFEMNODALENERGY_HPP_

#include "tribol/integ/NodalIntegralExpressions.hpp"
#include "tribol/mesh/DifferentiableNodalField.hpp"
#include "tribol/physics/PairEvaluation.hpp"

namespace tribol {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
/**
 * @brief Use Enzyme to evaluate a nodewise energy and its partial derivatives with respect to assembled fields.
 *
 * Field indices identify variables; repeated expression references share the same differentiation input. Only
 * referenced first partials and upper-triangle second partials are allocated. The bound fields subsequently apply
 * coordinate chain rules, including their weighted coordinate Hessians. No geometry is generated here.
 * @tparam Field Contracted nodal field expression.
 * @param energy Deferred energy, including scaling and borrowed independent coefficients.
 * @param fields Bound values and coordinate Jacobians in expression-variable order.
 * @param area_tolerance Finite nonnegative support threshold for reciprocal-area expressions; unsupported nodes have
 * zero energy and partial derivatives.
 * @return Owned nodewise partials and globally summed energy. Inputs and expressions are not retained.
 * @throws std::invalid_argument Referenced fields, independent field lengths, or area tolerance are invalid.
 * @pre All ranks call collectively with matching expression dependencies and field MPI partitions.
 */
template <typename Field>
NodalEnergyDerivatives evaluateNodalEnergy( const NodalEnergyExpression<Field>& energy,
                                            const DifferentiableNodalFields& fields, RealT area_tolerance )
{
  using Expression = NodalEnergyExpression<Field>;
  if ( fields.size() < Expression::field_count || !std::isfinite( area_tolerance ) || area_tolerance < 0.0 ) {
    throw std::invalid_argument(
        "Nodal energy requires its referenced fields and a finite nonnegative area tolerance." );
  }
  const int num_dofs = fields.fieldSpace().GetTrueVSize();
  energy.validateSize( num_dofs );
  NodalEnergyDerivatives data;
  for ( IndexT i = 0; i < Expression::field_count; ++i ) {
    if ( !Expression::dependsOn( i ) ) continue;
    data.gradient.emplace( i, fields.makeVector() );
    if constexpr ( Expression::has_outer_hessian ) {
      for ( IndexT j = i; j < Expression::field_count; ++j ) {
        if ( Expression::dependsOn( j ) ) data.hessian.emplace( std::pair{ i, j }, fields.makeVector() );
      }
    }
  }
  RealT local_energy = 0.0;
  for ( int node = 0; node < num_dofs; ++node ) {
    auto definition = energy.atNode( node, area_tolerance );
    MdArray<RealT, decltype( definition )::dof_count> state{};
    for ( const auto& [index, partial] : data.gradient ) state[index] = fields[index].values[node];
    const auto evaluated = EnergyPairEvaluator<decltype( definition )>::evaluate( state.view(), definition );
    local_energy += evaluated.energy;
    for ( auto& [index, partial] : data.gradient ) partial[node] = evaluated.residual[index];
    for ( auto& [indices, partial] : data.hessian ) partial[node] = evaluated.jacobian( indices.first, indices.second );
  }
  MPI_Allreduce( &local_energy, &data.energy, 1, MPI_DOUBLE, MPI_SUM, fields.fieldSpace().GetComm() );
  return data;
}
#endif
}  // namespace tribol
#endif  // SRC_TRIBOL_MESH_MFEMNODALENERGY_HPP_
