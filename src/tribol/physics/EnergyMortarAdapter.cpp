// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/common/BasicTypes.hpp"
#include "tribol/physics/EnergyMortarAdapter.hpp"

#include <axom/slic/interface/slic_macros.hpp>

#include <utility>

#include "tribol/mesh/MfemData.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

template <typename QuadratureRule>
EnergyMortarAdapter<QuadratureRule>::EnergyMortarAdapter( MfemMeshData& mesh_data, MfemSubmeshData& submesh_data,
                                                          MfemJacobianData& jac_data,
                                                          std::unique_ptr<CoarseSearch> coarse_search,
                                                          std::unique_ptr<RefinedSearch> refined_search, RealT k,
                                                          QuadratureRule quadrature_rule, bool enzyme_quadrature,
                                                          bool use_penalty, RealT residual_gap )
    : use_penalty_( use_penalty ),
      coarse_search_( std::move( coarse_search ) ),
      refined_search_( std::move( refined_search ) ),
      quadrature_rule_( std::move( quadrature_rule ) ),
      mesh_data_( mesh_data ),
      submesh_data_( submesh_data ),
      jac_data_( jac_data ),
      penalty_( k ),
      params_{ enzyme_quadrature, residual_gap },
      evaluator_( params_, quadrature_rule_ )
{
  SLIC_ASSERT_MSG( coarse_search_ != nullptr, "ENERGY_MORTAR requires a coarse search object." );
  SLIC_ASSERT_MSG( refined_search_ != nullptr, "ENERGY_MORTAR requires a refined search object." );
}

template <typename QuadratureRule>
typename EnergyMortarAdapter<QuadratureRule>::Calculator::Coordinates
EnergyMortarAdapter<QuadratureRule>::getPairCoordinates( const InterfacePair& pair, MeshData::Viewer mesh1,
                                                         MeshData::Viewer mesh2 )
{
  typename Calculator::Coordinates coordinates;
  mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
  mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + Calculator::element_dof_count );
  return coordinates;
}

template <typename QuadratureRule>
ArrayT<InterfacePair> EnergyMortarAdapter<QuadratureRule>::findCandidatePairs( MeshData::Viewer mesh1,
                                                                               MeshData::Viewer mesh2 ) const
{
  const auto candidates = coarse_search_->findCandidatePairs( mesh1, mesh2 );
  return refined_search_->filterCandidatePairs( candidates, mesh1, mesh2 );
}

template <typename QuadratureRule>
MfemNodalQuantityData EnergyMortarAdapter<QuadratureRule>::assembleNodalQuantities(
    MeshData::Viewer mesh1, MeshData::Viewer mesh2, ArrayViewT<const InterfacePair> pairs,
    MfemNodalAssembler& assembler ) const
{
  for ( const auto& pair : pairs ) {
    const auto coordinates = getPairCoordinates( pair, mesh1, mesh2 );
    const auto quantities = evaluator_.computeContactQuantities( coordinates.view() );
    if ( quantities.AI[0] <= 0.0 && quantities.AI[1] <= 0.0 ) {
      continue;
    }
    assembler.addPairQuantities(
        pair, { quantities.g_tilde.data(), NodalContactData::node_count },
        { quantities.AI.data(), NodalContactData::node_count },
        { quantities.gap_gradients.data(), { NodalContactData::node_count, Calculator::dof_count } },
        { quantities.area_gradients.data(), { NodalContactData::node_count, Calculator::dof_count } } );
  }
  return assembler.assembleQuantities( !tied_contact_ && use_penalty_, area_tol_ );
}

template <typename QuadratureRule>
ContactFormulationResult EnergyMortarAdapter<QuadratureRule>::takeGapResult( MfemNodalQuantityData& quantities ) const
{
  ContactFormulationResult result;
  result.gap.reset( use_penalty_ ? quantities.normalized_gap.release() : quantities.weighted_gap.release() );
  result.dg_dx.reset( quantities.weighted_gap_jacobian.release() );
  return result;
}

template <typename QuadratureRule>
ContactFormulationResult EnergyMortarAdapter<QuadratureRule>::computeNodalGaps( MeshData::Viewer mesh1,
                                                                                MeshData::Viewer mesh2 ) const
{
  const auto pairs = findCandidatePairs( mesh1, mesh2 );
  MfemNodalAssembler assembler( mesh_data_, submesh_data_, jac_data_, mesh1, mesh2, pairs.size() );
  auto quantities = assembleNodalQuantities( mesh1, mesh2, pairs.view(), assembler );
  return takeGapResult( quantities );
}

template <typename QuadratureRule>
ContactFormulationResult EnergyMortarAdapter<QuadratureRule>::computeNodalForces( MeshData::Viewer mesh1,
                                                                                  MeshData::Viewer mesh2 ) const
{
  const auto pairs = findCandidatePairs( mesh1, mesh2 );
  MfemNodalAssembler assembler( mesh_data_, submesh_data_, jac_data_, mesh1, mesh2, pairs.size() );
  auto quantities = assembleNodalQuantities( mesh1, mesh2, pairs.view(), assembler );
  shared::ParVector pressure_vec;
  shared::ParVector force_vec;
  shared::ParSparseMat df_dx;

  if ( use_penalty_ ) {
    // Penalty mode: p = k * (g_tilde / A)
    pressure_vec = penalty_ * quantities.normalized_gap;
  } else {
    // Read caller-supplied LM values from the current submesh field.
    pressure_vec = assembler.getSubmeshPressureTrueDofs();
    SLIC_ERROR_ROOT_IF( pressure_vec.size() != quantities.weighted_gap.size(),
                        "LM vector size mismatch with contact dofs (g_tilde)." );
  }

  const RealT energy = pressure_vec.dot( quantities.weighted_gap );

  if ( !use_penalty_ ) {
    // -------------------------------------------------------------------------
    // LM mode: force = G^T * lambda and df/dx = lambda · d^2(g_tilde)/dx^2
    // -------------------------------------------------------------------------
    force_vec = pressure_vec * quantities.weighted_gap_jacobian;
    const auto redecomp_lambda = assembler.transferToRedecomp( pressure_vec.get() );
    for ( const auto& pair : pairs ) {
      const auto coordinates = getPairCoordinates( pair, mesh1, mesh2 );
      const auto conn1 = mesh1.getConnectivity()( pair.element_id1_ );
      const MdArray<RealT, NodalContactData::node_count> gap_weights{ redecomp_lambda( conn1[0] ),
                                                                      redecomp_lambda( conn1[1] ) };
      const auto hessians = evaluator_.computeSecondDerivatives( coordinates.view(), false );
      assembler.addPairHessians( pair, { gap_weights.data(), NodalContactData::node_count },
                                 { hessians.gap_hessians.data(),
                                   { NodalContactData::node_count, Calculator::dof_count, Calculator::dof_count } } );
    }
    df_dx = assembler.assembleHessians();
  } else {
    // ---------------------------------------------------------------------------
    // Penalty mode: force and Jacobian include pressure/area coupling terms
    // ---------------------------------------------------------------------------
    auto k_over_a = penalty_ * quantities.area.inverse( area_tol_ );
    auto p_over_a = pressure_vec.divide( quantities.area, area_tol_ );

    shared::ParSparseMat dp_dx( quantities.weighted_gap_jacobian.get() );
    dp_dx->ScaleRows( k_over_a.get() );
    shared::ParSparseMat dp_dx_temp( quantities.area_jacobian.get() );
    dp_dx_temp->ScaleRows( p_over_a.get() );
    dp_dx -= dp_dx_temp;

    force_vec = ( pressure_vec * quantities.weighted_gap_jacobian ) + ( quantities.weighted_gap * dp_dx );

    const auto redecomp_pressure = assembler.transferToRedecomp( pressure_vec.get() );
    const auto redecomp_g_tilde = assembler.transferToRedecomp( quantities.weighted_gap.get() );
    const auto redecomp_A = assembler.transferToRedecomp( quantities.area.get() );
    for ( const auto& pair : pairs ) {
      const auto conn1 = mesh1.getConnectivity()( pair.element_id1_ );
      const MdArray<RealT, NodalContactData::node_count> gap_weights{ 2.0 * redecomp_pressure( conn1[0] ),
                                                                      2.0 * redecomp_pressure( conn1[1] ) };
      if ( gap_weights[0] == 0.0 && gap_weights[1] == 0.0 ) {
        continue;
      }
      const MdArray<RealT, NodalContactData::node_count> area_weights{
          -redecomp_g_tilde( conn1[0] ) * redecomp_pressure( conn1[0] ) / redecomp_A( conn1[0] ),
          -redecomp_g_tilde( conn1[1] ) * redecomp_pressure( conn1[1] ) / redecomp_A( conn1[1] ) };
      const auto coordinates = getPairCoordinates( pair, mesh1, mesh2 );
      const auto hessians = evaluator_.computeSecondDerivatives( coordinates.view() );
      assembler.addPairHessians( pair, { gap_weights.data(), NodalContactData::node_count },
                                 { hessians.gap_hessians.data(),
                                   { NodalContactData::node_count, Calculator::dof_count, Calculator::dof_count } },
                                 { area_weights.data(), NodalContactData::node_count },
                                 { hessians.area_hessians.data(),
                                   { NodalContactData::node_count, Calculator::dof_count, Calculator::dof_count } } );
    }
    df_dx = assembler.assembleHessians();

    auto pg2_over_asq = ( 2.0 * pressure_vec )
                            .multiplyInPlace( quantities.weighted_gap )
                            .divideInPlace( quantities.area, area_tol_ )
                            .divideInPlace( quantities.area, area_tol_ );

    auto p_over_a_diag = assembler.makeSubmeshDiagonalMatrix( p_over_a.get() );
    auto pg2_over_asq_diag = assembler.makeSubmeshDiagonalMatrix( pg2_over_asq.get() );

    df_dx -= shared::ParSparseMat::rap( quantities.weighted_gap_jacobian, p_over_a_diag, quantities.area_jacobian );
    df_dx -= shared::ParSparseMat::rap( quantities.area_jacobian, p_over_a_diag, quantities.weighted_gap_jacobian );
    df_dx += shared::ParSparseMat::rap( quantities.area_jacobian, pg2_over_asq_diag, quantities.weighted_gap_jacobian );
    df_dx += dp_dx.transpose() * quantities.weighted_gap_jacobian;
    df_dx += quantities.weighted_gap_jacobian.transpose() * dp_dx;
  }

  ContactFormulationResult result;
  if ( !use_penalty_ ) {
    auto df_dp = quantities.weighted_gap_jacobian.transpose();
    result.df_dp.reset( df_dp.release() );
  }
  auto gaps = takeGapResult( quantities );
  result.gap = std::move( gaps.gap );
  result.dg_dx = std::move( gaps.dg_dx );
  result.energy = energy;
  result.pressure.reset( pressure_vec.release() );
  result.force.reset( force_vec.release() );
  result.df_dx.reset( df_dx.release() );
  return result;
}

template class EnergyMortarAdapter<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>>;
template class EnergyMortarAdapter<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>>;
template class EnergyMortarAdapter<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>>;

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
