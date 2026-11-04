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
#include "tribol/mesh/GlobalAD.hpp"
#include "tribol/mesh/MfemPairAssembler.hpp"
#include "tribol/physics/PairMethod.hpp"
#include <utility>
#include <functional>
#include <optional>

namespace tribol {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
/**
 * @brief Borrow displacement and optional dual transfer data for a contact calculation.
 *
 * No geometry or calculated fields are retained. Referenced MFEM data must outlive this context and its evaluation.
 */
struct MfemContactAssemblyContext {
  const MfemMeshData& mesh_data;          ///< Borrowed displacement and decomposition data.
  const MfemJacobianData& jacobian_data;  ///< Borrowed coordinate and optional dual Jacobian transfers.
  std::optional<std::reference_wrapper<const MfemSubmeshData>> submesh_data;  ///< Borrowed dual data for nodal methods.
  /**
   * @brief Borrow parent displacement transfers.
   * @param mesh Displacement data.
   * @param jacobian Coordinate Jacobian transfer.
   */
  MfemContactAssemblyContext( const MfemMeshData& mesh, const MfemJacobianData& jacobian )
      : mesh_data( mesh ), jacobian_data( jacobian )
  {
  }
  /**
   * @brief Borrow parent and dual transfers.
   * @param mesh Displacement data.
   * @param submesh Dual pressure/gap data.
   * @param jacobian Coordinate and dual Jacobian transfer.
   */
  MfemContactAssemblyContext( const MfemMeshData& mesh, const MfemSubmeshData& submesh,
                              const MfemJacobianData& jacobian )
      : mesh_data( mesh ), jacobian_data( jacobian ), submesh_data( std::cref( submesh ) )
  {
  }
};

namespace detail {
/**
 * @brief Evaluate nodal integrals and gradients on each accepted pair and assemble them onto the dual submesh.
 * @tparam PairMethod Nodal pair method type.
 * @param definition Prepared nodal integral definition.
 * @param domain Current mesh views and geometry-approved pairs.
 * @param[in,out] assembler Nodal assembler accumulating quantities and gradients.
 * @return Assembled nodal weighted gaps, areas, normalized gaps, and coordinate Jacobians.
 */
template <typename PairMethod>
MfemNodalQuantityData assembleNodalQuantities(
    const typename PairMethod::Definition& definition,
    const ContactDomain<typename PairMethod::FirstGeometry, typename PairMethod::SecondGeometry>& domain,
    MfemNodalAssembler& assembler )
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
  return assembler.assembleQuantities( false, PairMethod::area_tolerance );
}

/**
 * @brief Move the selected nodal gap vector and weighted-gap Jacobian into a ContactFormulationResult.
 * @param[in,out] quantities Assembled nodal quantities whose gap and Jacobian pointers are transferred.
 * @param use_penalty If true, clamp positive gaps and their Jacobian rows, then return normalized gaps; otherwise
 * return weighted nodal gaps.
 * @return Result owning gap and dg_dx.
 */
inline ContactFormulationResult takeGapResult( MfemNodalQuantityData& quantities, bool use_penalty )
{
  if ( use_penalty ) {
    mfem::Array<int> open_nodes;
    for ( int node = 0; node < quantities.weighted_gap.size(); ++node ) {
      if ( quantities.weighted_gap[node] > 0.0 ) {
        quantities.normalized_gap[node] = 0.0;
        open_nodes.Append( node );
      }
    }
    quantities.weighted_gap_jacobian.eliminateRows( open_nodes );
  }
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
 * @return Owned residual and Jacobian, with the globally summed pair energy for energy-defined methods.
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
  if constexpr ( PairMethod::has_energy ) {
    RealT global_energy = 0.0;
    MPI_Allreduce( &energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, result.force->GetComm() );
    result.energy = global_energy;
  }
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
  auto quantities = detail::assembleNodalQuantities<PairMethod>( definition, domain, assembler );
  return detail::takeGapResult( quantities, use_penalty );
}

/**
 * @brief Assemble nodal energy derivatives, residuals, and Jacobians using the existing penalty or LM definition.
 *
 * Nodal integrals are assembled before activation and outer energy derivatives are evaluated. Weighted pair Hessians
 * regenerate quadrature inside differentiation. Neither the domain nor this function retains calculated fields.
 * @tparam PairMethod Nodal method providing quantity derivatives, weighted-energy derivatives, and enforcement inputs.
 * Definition::Physics provides penaltyEnergy() and lagrangeMultiplierEnergy() to construct
 * the deferred outer energy from named fields.
 * @param domain Geometry-approved pairs and current host mesh views.
 * @param definition Prepared nodal integral definition.
 * @param method Fixed configuration selecting penalty or Lagrange multiplier enforcement.
 * @param context Current displacement and Jacobian transfers, including registered penalties.
 * @param[in,out] assembler Empty nodal assembler sized for the domain's pair count.
 * @return Owned gaps, pressure, residual, Jacobians, and globally summed energy. LM results also include df_dp.
 * @throws std::invalid_argument Quadrature geometry, penalty inputs, or multiplier vector dimensions are invalid.
 * @throws std::length_error A pair exceeds its quadrature capacity.
 */
template <typename PairMethod>
ContactFormulationResult assembleNodalContactForces(
    const ContactDomain<typename PairMethod::FirstGeometry, typename PairMethod::SecondGeometry>& domain,
    const typename PairMethod::Definition& definition, const PairMethod& method,
    const MfemContactAssemblyContext& context, MfemNodalAssembler& assembler )
{
  static_assert( PairMethod::is_nodal );
  using Definition = typename PairMethod::Definition;
  constexpr IndexT node_count = Definition::node_count;
  constexpr IndexT pair_dofs = Definition::dof_count;
  const bool use_penalty = method.usesPenalty();
  auto quantities = detail::assembleNodalQuantities<PairMethod>( definition, domain, assembler );

  // The field Hessians share a pair pass. Each request starts fresh so repeated chain-rule assembly cannot add
  // contributions left over from an earlier set of weights.
  auto weighted_hessian = [&]( const NodalEnergyGradient& weights ) {
    constexpr IndexT gap = static_cast<IndexT>( NodalIntegralQuantity::WeightedGap );
    constexpr IndexT area = static_cast<IndexT>( NodalIntegralQuantity::Area );
    const auto transfer = [&]( IndexT index ) -> mfem::GridFunction {
      const auto entry = weights.find( index );
      if ( entry == weights.end() ) return {};
      return assembler.transferToRedecomp( entry->second.get() );
    };
    const auto redecomp_gap_weights = transfer( gap );
    const auto redecomp_area_weights = transfer( area );
    const bool has_gap_weights = weights.contains( gap );
    const bool has_area_weights = weights.contains( area );
    MfemPairAssembler hessians( context.mesh_data, context.jacobian_data, domain.mesh1(), domain.mesh2(),
                                domain.size() );
    domain.forEachPair( [&]( InterfacePair pair, auto coordinates ) {
      const auto conn1 = domain.mesh1().getConnectivity()( pair.element_id1_ );
      MdArray<RealT, node_count> gap_weights{}, area_weights{};
      bool has_weights = false;
      for ( IndexT node = 0; node < node_count; ++node ) {
        if ( has_gap_weights ) gap_weights[node] = redecomp_gap_weights( conn1[node] );
        if ( has_area_weights ) area_weights[node] = redecomp_area_weights( conn1[node] );
        has_weights = has_weights || gap_weights[node] != 0.0 || area_weights[node] != 0.0;
      }
      if ( !has_weights ) return;
      const auto weighted_definition = definition.withWeights( gap_weights, area_weights );
      const auto pair_derivatives = PairMethod::differentiateWeightedEnergy( coordinates, weighted_definition );
      hessians.addPairJacobian( pair, { pair_derivatives.jacobian.data(), { pair_dofs, pair_dofs } } );
    } );
    return hessians.assembleJacobian();
  };
  const auto fields = assembler.bindFields(
      { { quantities.weighted_gap, quantities.weighted_gap_jacobian }, { quantities.area, quantities.area_jacobian } },
      std::ref( weighted_hessian ) );
  const auto variables = makeADFields( fields );
  const auto& gap = variables[static_cast<IndexT>( NodalIntegralQuantity::WeightedGap )];
  const auto& area = variables[static_cast<IndexT>( NodalIntegralQuantity::Area )];
  const auto& physics = definition.physics();
  const auto energy = [&] {
    if ( use_penalty ) {
      const RealT penalty = method.preparePenalty( context.mesh_data, domain.mesh1(), domain.mesh2() );
      if constexpr ( requires { physics.penaltyEnergy( gap, area, penalty, PairMethod::area_tolerance ); } )
        return physics.penaltyEnergy( gap, area, penalty, PairMethod::area_tolerance );
      else
        return physics.penaltyEnergy( gap, area, penalty );
    }
    auto lm_vec = assembler.getSubmeshPressureTrueDofs();
    const auto pressure = ADVector::constant( fields.fieldSpace(), fields.coordinateSpace(), std::move( lm_vec ) );
    return physics.lagrangeMultiplierEnergy( gap, pressure );
  }();
  auto derivatives = energy.differentiate();
  auto pressure = derivatives.hasPartial( gap ) ? derivatives.releasePartial( gap ) : fields.makeVector();

  ContactFormulationResult result;
  if ( !use_penalty ) {
    auto df_dp = quantities.weighted_gap_jacobian.transpose();
    result.df_dp.reset( df_dp.release() );
  }
  auto gaps = detail::takeGapResult( quantities, use_penalty );
  result.gap = std::move( gaps.gap );
  result.dg_dx = std::move( gaps.dg_dx );
  result.energy = derivatives.value;
  result.pressure.reset( pressure.release() );
  result.force.reset( derivatives.gradient.release() );
  result.df_dx.reset( derivatives.hessian.release() );
  return result;
}
/**
 * @brief Evaluate a prepared local definition over a domain using its parent-only or nodal assembly requirements.
 *
 * Search and definition preparation remain with the caller. This entry point constructs per-call assembly storage,
 * delegates pair or nodal evaluation, and returns owned outputs without retaining the domain or expression.
 * @tparam PairMethod Energy, residual, or nodal method with a prepared Definition.
 * @param domain Geometry-approved pairs and current host mesh views.
 * @param definition Prepared local integral expression and inactive parameters.
 * @param method Method configuration and derivative selection.
 * @param context Borrowed displacement and optional dual transfer data.
 * @return Calculated contact outputs. Availability, signs, and energy reductions follow the selected method.
 * @throws std::invalid_argument Nodal transfer data, method inputs, or quadrature geometry are invalid.
 * @throws std::length_error Quadrature capacity or assembly storage is insufficient.
 */
template <typename PairMethod>
ContactFormulationResult evaluateContactForces(
    const ContactDomain<typename PairMethod::FirstGeometry, typename PairMethod::SecondGeometry>& domain,
    const typename PairMethod::Definition& definition, const PairMethod& method,
    const MfemContactAssemblyContext& context )
{
  if constexpr ( PairMethod::is_nodal ) {
    if ( !context.submesh_data ) throw std::invalid_argument( "Nodal pair contact requires dual submesh data." );
    MfemNodalAssembler assembler( context.mesh_data, context.submesh_data->get(), context.jacobian_data, domain.mesh1(),
                                  domain.mesh2(), domain.size() );
    return assembleNodalContactForces<PairMethod>( domain, definition, method, context, assembler );
  } else {
    MfemPairAssembler assembler( context.mesh_data, context.jacobian_data, domain.mesh1(), domain.mesh2(),
                                 domain.size() );
    return assemblePairContact<PairMethod>( domain, definition, assembler );
  }
}

#endif
}  // namespace tribol
#endif  // SRC_TRIBOL_MESH_MFEMCONTACTASSEMBLY_HPP_
