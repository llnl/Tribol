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
#ifdef BUILD_REDECOMP
/**
 * @brief Customize the energy and partial derivatives of an assembled nodal expression.
 *
 * Specialize this class for an expression to supply analytical or other derivatives. Its static evaluate() returns
 * NodalEnergyDerivatives with globally summed energy, first partials keyed by field index, and the upper triangle of
 * nodewise second partials. Missing entries mean zero. Returned vectors own their data on the bound field space.
 * The default implementation uses Enzyme; specializations also work in builds without Enzyme.
 * Use evaluateNodalEnergy() to apply input validation before calling the selected implementation.
 * @tparam Expression Deferred NodalEnergyExpression type identifying the calculation.
 */
template <typename Expression>
class NodalEnergyEvaluator;

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Default assembled-field energy derivatives using Enzyme at each node.
 * @tparam Expression Deferred expression identifying referenced fields and outer Hessian requirements.
 */
template <typename Expression>
class NodalEnergyEvaluator {
 public:
  /**
   * @brief Differentiate referenced values and sum energy across the field communicator without generating geometry.
   * @param energy Deferred expression, including scaling and borrowed independent coefficients.
   * @param fields Bound values and coordinate Jacobians in expression-variable order.
   * @param area_tolerance Support threshold for reciprocal-area expressions; unsupported nodes have zero energy and
   * partials.
   * @return Owned first partials, upper-triangle second partials, and globally summed energy.
   * @pre Inputs have been validated by evaluateNodalEnergy().
   */
  static NodalEnergyDerivatives evaluate( const Expression& energy, const DifferentiableNodalFields& fields,
                                          RealT area_tolerance )
  {
    const int num_dofs = fields.fieldSpace().GetTrueVSize();
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
      for ( auto& [indices, partial] : data.hessian )
        partial[node] = evaluated.jacobian( indices.first, indices.second );
    }
    MPI_Allreduce( &local_energy, &data.energy, 1, MPI_DOUBLE, MPI_SUM, fields.fieldSpace().GetComm() );
    return data;
  }
};
#endif  // TRIBOL_USE_ENZYME

/**
 * @brief Evaluate nodal energy partials using Enzyme by default or a NodalEnergyEvaluator specialization.
 *
 * Field indices identify variables; repeated expression references share the same input. The bound fields apply
 * coordinate chain rules using the returned partials. Neither expressions nor input views are retained in results.
 * @tparam Field Contracted nodal field expression.
 * @param energy Deferred energy, including scaling and borrowed independent coefficients.
 * @param fields Bound values and coordinate Jacobians in expression-variable order.
 * @param area_tolerance Finite nonnegative support threshold for reciprocal-area expressions.
 * @return Owned nodewise partials and globally summed energy, following the selected evaluator's contract.
 * @throws std::invalid_argument Referenced fields, independent field lengths, or area tolerance are invalid.
 * Exceptions from a custom evaluator propagate to the caller.
 * @pre All ranks call collectively with matching expression dependencies and field MPI partitions.
 * Without Enzyme, a matching evaluator specialization must be provided.
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
  energy.validateSize( fields.fieldSpace().GetTrueVSize() );
  return NodalEnergyEvaluator<Expression>::evaluate( energy, fields, area_tolerance );
}
#endif
}  // namespace tribol
#endif  // SRC_TRIBOL_MESH_MFEMNODALENERGY_HPP_
