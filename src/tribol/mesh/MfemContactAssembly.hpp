// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Assemble existing pair definitions over a current contact domain with host MFEM transfers.
 */
#ifndef SRC_TRIBOL_MESH_MFEMCONTACTASSEMBLY_HPP_
#define SRC_TRIBOL_MESH_MFEMCONTACTASSEMBLY_HPP_

#include "tribol/integ/ContactDomain.hpp"
#include "tribol/mesh/MfemNodalAssembler.hpp"
#include "tribol/mesh/MfemPairAssembler.hpp"
#include "tribol/physics/PairMethod.hpp"
#include <type_traits>
#include <utility>

namespace tribol {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
namespace detail {
/**
 * @brief Evaluate nodal energy, pressure, and Enzyme partial derivatives with respect to (weighted_gap, area) at each
 * dual submesh true degree of freedom.
 * @tparam NodeDefinitionFactory Callable taking a dual true-DOF index and returning a two-DOF nodal energy definition.
 * @param assembler Nodal assembler providing dual true-DOF vector allocation.
 * @param quantities Assembled nodal weighted gaps and tributary areas on dual submesh true DOFs.
 * @param has_area_derivatives Whether the nodal definition depends on tributary area.
 * @param has_outer_hessian Whether second partial derivatives of nodal energy are nonzero.
 * @param make_node_definition Factory returning the nodal energy definition for a given dual true DOF.
 * @return Nodal energy, pressure, and first/second partial derivatives on dual submesh true DOFs.
 */
template <typename NodeDefinitionFactory>
MfemNodalDerivativeData evaluateNodalDerivatives( const MfemNodalAssembler& assembler,
                                                  const MfemNodalQuantityData& quantities, bool has_area_derivatives,
                                                  bool has_outer_hessian, NodeDefinitionFactory&& make_node_definition )
{
  auto data = assembler.makeDerivativeData( has_area_derivatives, has_outer_hessian );
  const int num_dofs = quantities.weighted_gap.size();
  RealT local_energy = 0.0;
  for ( int i = 0; i < num_dofs; ++i ) {
    const MdArray<RealT, 2> nodal_state{ quantities.weighted_gap[i], quantities.area[i] };
    const auto definition = make_node_definition( i );
    using Definition = std::decay_t<decltype( definition )>;
    const auto evaluated = EnergyPairEvaluator<Definition>::evaluate( nodal_state.view(), definition );
    local_energy += evaluated.energy;
    data.pressure[i] = evaluated.residual[0];
    data.d_energy_d_gap[i] = evaluated.residual[0];
    if ( has_area_derivatives ) {
      data.d_energy_d_area[i] = evaluated.residual[1];
    }
    if ( has_outer_hessian ) {
      data.d2_energy_dg2[i] = evaluated.jacobian( 0, 0 );
      if ( has_area_derivatives ) {
        data.d2_energy_dg_da[i] = evaluated.jacobian( 0, 1 );
        data.d2_energy_da2[i] = evaluated.jacobian( 1, 1 );
      }
    }
  }
  MPI_Allreduce( &local_energy, &data.energy, 1, MPI_DOUBLE, MPI_SUM, quantities.weighted_gap.get().GetComm() );
  return data;
}

/**
 * @brief Evaluate nodal integrals and gradients on each accepted pair and assemble them onto the dual submesh.
 * @tparam PairMethod Nodal pair method type.
 * @param definition Prepared nodal integral definition.
 * @param domain Current mesh views and geometry-approved pairs.
 * @param use_penalty Whether nodal penalty enforcement is active (clamping positive weighted gaps).
 * @param[in,out] assembler Nodal assembler accumulating quantities and gradients.
 * @return Assembled nodal weighted gaps, areas, normalized gaps, and coordinate Jacobians.
 */
template <typename PairMethod>
MfemNodalQuantityData assembleNodalQuantities(
    const typename PairMethod::Definition& definition,
    const ContactDomain<typename PairMethod::FirstGeometry, typename PairMethod::SecondGeometry>& domain,
    bool use_penalty, MfemNodalAssembler& assembler )
{
  using Definition = typename PairMethod::Definition;
  constexpr IndexT node_count = Definition::node_count;
  constexpr IndexT pair_dofs = Definition::dof_count;
  domain.forEachPair( [&]( InterfacePair pair, auto coordinates ) {
    const auto primal = definition.evaluate( coordinates );
    if ( primal.quadrature_result.status != QuadratureStatus::Success )
      reportQuadratureFailure( primal.quadrature_result.status );
    if ( !primal.has_active_points ) return;
    const auto derivatives = PairMethod::differentiateQuantities( coordinates, definition, primal );
    assembler.addPairQuantities( pair, { derivatives.values.data(), node_count },
                                 { derivatives.values.data() + node_count, node_count },
                                 { derivatives.gradients.data(), { node_count, pair_dofs } },
                                 { derivatives.gradients.data() + node_count * pair_dofs, { node_count, pair_dofs } } );
  } );
  return assembler.assembleQuantities( use_penalty, PairMethod::area_tolerance );
}

/**
 * @brief Move the selected nodal gap vector and weighted-gap Jacobian into a ContactFormulationResult.
 * @param[in,out] quantities Assembled nodal quantities whose gap and Jacobian pointers are transferred.
 * @param use_penalty If true, return the area-normalized nodal gap; otherwise return the weighted nodal gap.
 * @return Result owning gap and dg_dx.
 */
inline ContactFormulationResult takeGapResult( MfemNodalQuantityData& quantities, bool use_penalty )
{
  ContactFormulationResult result;
  result.gap.reset( use_penalty ? quantities.normalized_gap.release() : quantities.weighted_gap.release() );
  result.dg_dx.reset( quantities.weighted_gap_jacobian.release() );
  return result;
}
}  // namespace detail

/**
 * @brief Assemble independent pair residuals and Jacobians, with energy for energy-defined methods.
 *
 * Quadrature is generated by the definition from current coordinates inside each differentiated calculation.
 * @tparam PairMethod Energy or force method providing differentiate().
 * @param domain Geometry-approved pairs and current host mesh views; borrowed only during this call.
 * @param definition Prepared pair definition with fixed configuration.
 * @param[in,out] assembler Empty parent assembler sized for the domain's pair count.
 * @return Owned residual and Jacobian, with the local pair energy sum for energy-defined methods.
 * Empty or inactive domains produce allocated zero residuals and Jacobians.
 * @throws std::invalid_argument A pair has invalid quadrature geometry.
 * @throws std::length_error A pair exceeds its quadrature capacity.
 */
template <typename PairMethod>
ContactFormulationResult assemblePairContact(
    const ContactDomain<typename PairMethod::FirstGeometry, typename PairMethod::SecondGeometry>& domain,
    const typename PairMethod::Definition& definition, MfemPairAssembler& assembler )
{
  static_assert( !PairMethod::is_nodal );
  constexpr IndexT pair_dofs = PairMethod::Definition::dof_count;
  static_assert( pair_dofs ==
                 ContactDomain<typename PairMethod::FirstGeometry, typename PairMethod::SecondGeometry>::dof_count );
  RealT energy = 0.0;
  domain.forEachPair( [&]( InterfacePair pair, auto coordinates ) {
    const auto primal = definition.evaluate( coordinates );
    if ( primal.quadrature_result.status != QuadratureStatus::Success )
      reportQuadratureFailure( primal.quadrature_result.status );
    if ( !primal.has_active_points ) return;
    const auto derivatives = PairMethod::differentiate( coordinates, definition, primal );
    if constexpr ( PairMethod::has_energy ) energy += derivatives.energy;
    assembler.addPair( pair, { derivatives.residual.data(), pair_dofs },
                       { derivatives.jacobian.data(), { pair_dofs, pair_dofs } } );
  } );
  auto result = assembler.assemble();
  if constexpr ( PairMethod::has_energy ) result.energy = energy;
  return result;
}

/**
 * @brief Assemble nodal gap integrals and their coordinate Jacobian without evaluating contact forces.
 * @tparam PairMethod Nodal method defining the area tolerance and integral derivatives.
 * @param domain Geometry-approved pairs and current host mesh views.
 * @param definition Prepared nodal integral definition.
 * @param use_penalty Clamp positive assembled gaps and return normalized gaps when true; otherwise return weighted
 * gaps.
 * @param[in,out] assembler Empty nodal assembler sized for the domain's pair count.
 * @return Owned gaps and weighted-gap coordinate Jacobian; other outputs are absent.
 * @throws std::invalid_argument A pair has invalid quadrature geometry.
 * @throws std::length_error A pair exceeds its quadrature capacity.
 */
template <typename PairMethod>
ContactFormulationResult assembleNodalContactGaps(
    const ContactDomain<typename PairMethod::FirstGeometry, typename PairMethod::SecondGeometry>& domain,
    const typename PairMethod::Definition& definition, bool use_penalty, MfemNodalAssembler& assembler )
{
  static_assert( PairMethod::is_nodal );
  auto quantities = detail::assembleNodalQuantities<PairMethod>( definition, domain, use_penalty, assembler );
  return detail::takeGapResult( quantities, use_penalty );
}

/**
 * @brief Assemble nodal energy derivatives, residuals, and Jacobians using the existing penalty or LM definition.
 *
 * Nodal integrals are assembled before activation and outer energy derivatives are evaluated. Weighted pair Hessians
 * regenerate quadrature inside differentiation. Neither the domain nor this function retains calculated fields.
 * @tparam PairMethod Nodal method providing quantity derivatives, weighted-energy derivatives, and enforcement inputs.
 * @param domain Geometry-approved pairs and current host mesh views.
 * @param definition Prepared nodal integral definition.
 * @param method Fixed configuration selecting penalty or Lagrange multiplier enforcement.
 * @param mesh_data Current registered penalties, read only for penalty enforcement.
 * @param[in,out] assembler Empty nodal assembler sized for the domain's pair count.
 * @return Owned gaps, pressure, residual, Jacobians, and globally summed energy. LM results also include df_dp.
 * @throws std::invalid_argument Quadrature geometry, penalty inputs, or multiplier vector dimensions are invalid.
 * @throws std::length_error A pair exceeds its quadrature capacity.
 */
template <typename PairMethod>
ContactFormulationResult assembleNodalContactForces(
    const ContactDomain<typename PairMethod::FirstGeometry, typename PairMethod::SecondGeometry>& domain,
    const typename PairMethod::Definition& definition, const PairMethod& method, const MfemMeshData& mesh_data,
    MfemNodalAssembler& assembler )
{
  static_assert( PairMethod::is_nodal );
  using Definition = typename PairMethod::Definition;
  constexpr IndexT node_count = Definition::node_count;
  constexpr IndexT pair_dofs = Definition::dof_count;
  const bool use_penalty = method.usesPenalty();
  auto quantities = detail::assembleNodalQuantities<PairMethod>( definition, domain, use_penalty, assembler );

  using NodalPhysics = typename Definition::Physics;
  MfemNodalDerivativeData derivatives;
  if ( use_penalty ) {
    const auto parameters = method.preparePenalty( mesh_data, domain.mesh1(), domain.mesh2() );
    const NodalPenaltyStateEnergy<NodalPhysics> penalty_energy{ parameters.penalty, parameters.residual_gap };
    derivatives =
        detail::evaluateNodalDerivatives( assembler, quantities, true, true, [&]( int ) { return penalty_energy; } );
  } else {
    const auto lm_vec = assembler.getSubmeshPressureTrueDofs();
    if ( lm_vec.size() != quantities.weighted_gap.size() ) {
      throw std::invalid_argument( "Lagrange multiplier vector size must match the dual submesh true-DOF count." );
    }
    derivatives = detail::evaluateNodalDerivatives( assembler, quantities, false, false, [&]( int node ) {
      return NodalLagrangeMultiplierStateEnergy<NodalPhysics>{ lm_vec[node], 0.0 };
    } );
  }

  const auto redecomp_d_gap = assembler.transferToRedecomp( derivatives.d_energy_d_gap.get() );
  const auto redecomp_d_area = derivatives.has_area_derivatives
                                   ? assembler.transferToRedecomp( derivatives.d_energy_d_area.get() )
                                   : mfem::GridFunction{};
  domain.forEachPair( [&]( InterfacePair pair, auto coordinates ) {
    const auto conn1 = domain.mesh1().getConnectivity()( pair.element_id1_ );
    const MdArray<RealT, node_count> gap_weights{ redecomp_d_gap( conn1[0] ), redecomp_d_gap( conn1[1] ) };
    if ( use_penalty && gap_weights[0] == 0.0 && gap_weights[1] == 0.0 ) return;
    const MdArray<RealT, node_count> area_weights =
        derivatives.has_area_derivatives
            ? MdArray<RealT, node_count>{ redecomp_d_area( conn1[0] ), redecomp_d_area( conn1[1] ) }
            : MdArray<RealT, node_count>{};
    const auto weighted_definition = definition.withWeights( gap_weights, area_weights );
    const auto pair_derivatives = PairMethod::differentiateWeightedEnergy( coordinates, weighted_definition );
    assembler.addPairHessian( pair, { pair_derivatives.jacobian.data(), { pair_dofs, pair_dofs } } );
  } );

  auto force_vec = assembler.assembleForce( quantities, derivatives );
  auto df_dx = assembler.assembleJacobian( quantities, derivatives );

  ContactFormulationResult result;
  if ( !use_penalty ) {
    auto df_dp = quantities.weighted_gap_jacobian.transpose();
    result.df_dp.reset( df_dp.release() );
  }
  auto gaps = detail::takeGapResult( quantities, use_penalty );
  result.gap = std::move( gaps.gap );
  result.dg_dx = std::move( gaps.dg_dx );
  result.energy = derivatives.energy;
  result.pressure.reset( derivatives.pressure.release() );
  result.force.reset( force_vec.release() );
  result.df_dx.reset( df_dx.release() );
  return result;
}
#endif
}  // namespace tribol
#endif  // SRC_TRIBOL_MESH_MFEMCONTACTASSEMBLY_HPP_
