// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/common/BasicTypes.hpp"
#include "tribol/physics/EnergyMortarAdapter.hpp"

#include <axom/slic/interface/slic_macros.hpp>

#include <utility>

#include "tribol/mesh/MfemData.hpp"
#include "tribol/mesh/MfemPairAssembler.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"
#include "tribol/physics/PairEvaluation.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

template <template <typename> class EnforcementLocation, typename QuadratureRule>
EnergyMortarAdapter<EnforcementLocation, QuadratureRule>::EnergyMortarAdapter(
    MfemMeshData& mesh_data, MfemSubmeshData& submesh_data, MfemJacobianData& jac_data,
    std::unique_ptr<CoarseSearch> coarse_search, std::unique_ptr<RefinedSearch> refined_search, RealT k,
    QuadratureRule quadrature_rule, bool enzyme_quadrature, bool use_penalty, RealT residual_gap )
    : use_penalty_( use_penalty ),
      coarse_search_( std::move( coarse_search ) ),
      refined_search_( std::move( refined_search ) ),
      quadrature_rule_( std::move( quadrature_rule ) ),
      mesh_data_( mesh_data ),
      submesh_data_( submesh_data ),
      jac_data_( jac_data )
{
  SLIC_ASSERT_MSG( coarse_search_ != nullptr, "ENERGY_MORTAR requires a coarse search object." );
  SLIC_ASSERT_MSG( refined_search_ != nullptr, "ENERGY_MORTAR requires a refined search object." );

  params_.k = k;
  params_.enzyme_quadrature = enzyme_quadrature;
  params_.residual_gap = residual_gap;

  evaluator_ = std::make_unique<EnergyMortarCalculator<QuadratureRule>>( params_, quadrature_rule_ );
}

template <template <typename> class EnforcementLocation, typename QuadratureRule>
typename EnergyMortarAdapter<EnforcementLocation, QuadratureRule>::Calculator::Coordinates
EnergyMortarAdapter<EnforcementLocation, QuadratureRule>::getPairCoordinates( const InterfacePair& pair,
                                                                              MeshData::Viewer mesh1,
                                                                              MeshData::Viewer mesh2 )
{
  typename Calculator::Coordinates coordinates;
  mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
  mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + Calculator::element_dof_count );
  return coordinates;
}

template <template <typename> class EnforcementLocation, typename QuadratureRule>
ArrayT<InterfacePair> EnergyMortarAdapter<EnforcementLocation, QuadratureRule>::findCandidatePairs(
    MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  const auto candidates = coarse_search_->findCandidatePairs( mesh1, mesh2 );
  return refined_search_->filterCandidatePairs( candidates, mesh1, mesh2 );
}

template <typename Adapter>
typename Nodal<Adapter>::NodalData Nodal<Adapter>::computeNodalData( MeshData::Viewer mesh1,
                                                                     MeshData::Viewer mesh2 ) const
{
  const auto* adapter = static_cast<const Adapter*>( this );
  NodalData data;
  data.pairs = adapter->findCandidatePairs( mesh1, mesh2 );

  // NOTE: user should have called updateMfemParallelDecomposition() with updated coords before calling this

  // Tribol level data structures for storing gap, area, and derivatives
  mfem::GridFunction redecomp_gap( adapter->submesh_data_.GetRedecompGap().FESpace() );
  redecomp_gap = 0.0;
  mfem::GridFunction redecomp_area( redecomp_gap.FESpace() );
  redecomp_area = 0.0;

  const bool use_lor = ( adapter->mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes =
      use_lor ? *adapter->mesh_data_.GetLORMeshFESpace() : adapter->mesh_data_.GetSubmeshFESpace();
  const auto& pressure_surface_fes =
      use_lor ? *adapter->submesh_data_.GetLORMeshFESpace() : adapter->submesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *adapter->mesh_data_.GetRedecompResponse().FESpace();
  const auto& pressure_redecomp_fes = *adapter->submesh_data_.GetRedecompGap().FESpace();
  const auto& nonmortar_elem_map = adapter->mesh_data_.GetElemMap1();
  const auto& mortar_elem_map = adapter->mesh_data_.GetElemMap2();

  PackedPairJacobianContribs dg_lm_nm( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs dg_lm_m( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs dA_lm_nm( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs dA_lm_m( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );

  dg_lm_nm.reserve( data.pairs.size(), 8 );
  dg_lm_m.reserve( data.pairs.size(), 8 );
  dA_lm_nm.reserve( data.pairs.size(), 8 );
  dA_lm_m.reserve( data.pairs.size(), 8 );

  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  // Compute local contributions
  for ( const auto& pair : data.pairs ) {
    const auto coordinates = adapter->getPairCoordinates( pair, mesh1, mesh2 );
    const auto elem1 = static_cast<int>( pair.element_id1_ );
    const auto elem2 = static_cast<int>( pair.element_id2_ );

    const auto quantities = adapter->evaluator_->computeContactQuantities( coordinates.view() );

    if ( quantities.AI[0] <= 0.0 && quantities.AI[1] <= 0.0 ) {
      continue;
    }

    auto A_conn = mesh1.getConnectivity()( elem1 );

    // Add to nodes of Element A
    redecomp_gap( A_conn[0] ) += quantities.g_tilde[0];
    redecomp_gap( A_conn[1] ) += quantities.g_tilde[1];

    redecomp_area( A_conn[0] ) += quantities.AI[0];
    redecomp_area( A_conn[1] ) += quantities.AI[1];

    RealT dg_tilde_dx_blocks[2][8];
    for ( int i{ 0 }; i < 4; ++i ) {
      dg_tilde_dx_blocks[0][i * 2] = quantities.gap_gradients( 0, node_idx[i] );
      dg_tilde_dx_blocks[0][i * 2 + 1] = quantities.gap_gradients( 1, node_idx[i] );
      dg_tilde_dx_blocks[1][i * 2] = quantities.gap_gradients( 0, node_idx[i + 4] );
      dg_tilde_dx_blocks[1][i * 2 + 1] = quantities.gap_gradients( 1, node_idx[i + 4] );
    }
    dg_lm_nm.append( elem1, elem1, dg_tilde_dx_blocks[0], 8 );
    dg_lm_m.append( elem1, elem2, dg_tilde_dx_blocks[1], 8 );

    RealT dA_dx_blocks[2][8];
    for ( int i{ 0 }; i < 4; ++i ) {
      dA_dx_blocks[0][i * 2] = quantities.area_gradients( 0, node_idx[i] );
      dA_dx_blocks[0][i * 2 + 1] = quantities.area_gradients( 1, node_idx[i] );
      dA_dx_blocks[1][i * 2] = quantities.area_gradients( 0, node_idx[i + 4] );
      dA_dx_blocks[1][i * 2 + 1] = quantities.area_gradients( 1, node_idx[i + 4] );
    }
    dA_lm_nm.append( elem1, elem1, dA_dx_blocks[0], 8 );
    dA_lm_m.append( elem1, elem2, dA_dx_blocks[1], 8 );
  }

  // Move gap and area to submesh level vectors
  mfem::ParLinearForm g_tilde_linear_form(
      const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
  adapter->submesh_data_.GetPressureTransfer().RedecompToSubmesh( redecomp_gap, g_tilde_linear_form );
  auto& P_submesh = *adapter->submesh_data_.GetSubmeshFESpace().GetProlongationMatrix();
  data.g_tilde_vec =
      shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
  data.g_tilde_vec.fill( 0.0 );
  P_submesh.MultTranspose( g_tilde_linear_form, data.g_tilde_vec.get() );

  mfem::Array<int> rows_to_elim;
  if ( !adapter->tied_contact_ && adapter->use_penalty_ ) {
    rows_to_elim.Reserve( data.g_tilde_vec.size() );
    for ( int i{ 0 }; i < data.g_tilde_vec.size(); ++i ) {
      if ( data.g_tilde_vec[i] > 0.0 ) {
        data.g_tilde_vec[i] = 0.0;
        rows_to_elim.push_back( i );
      }
    }
  }

  mfem::ParLinearForm A_linear_form(
      const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
  adapter->submesh_data_.GetPressureTransfer().RedecompToSubmesh( redecomp_area, A_linear_form );
  data.A_vec =
      shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
  data.A_vec.fill( 0.0 );
  P_submesh.MultTranspose( A_linear_form, data.A_vec.get() );

  data.gap_vec = data.g_tilde_vec.divide( data.A_vec, adapter->area_tol_ );

  // Move gap and area derivatives to (pressure true-dof rows, displacement true-dof cols)
  std::vector<PackedPairJacobianContribs> dg_contribs;
  dg_contribs.reserve( 2 );
  dg_contribs.push_back( std::move( dg_lm_nm ) );
  dg_contribs.push_back( std::move( dg_lm_m ) );
  data.dg_tilde_dx = adapter->jac_data_.GetMfemJacobian(
      &adapter->submesh_data_.GetSubmeshFESpace(), adapter->mesh_data_.GetParentCoords().ParFESpace(), dg_contribs );
  if ( !adapter->tied_contact_ && adapter->use_penalty_ ) {
    // technically, we should do this on all the vectors/matrices below, but it looks like the mutliplication operators
    // below will zero them out anyway
    data.dg_tilde_dx.eliminateRows( rows_to_elim );
  }

  std::vector<PackedPairJacobianContribs> dA_contribs;
  dA_contribs.reserve( 2 );
  dA_contribs.push_back( std::move( dA_lm_nm ) );
  dA_contribs.push_back( std::move( dA_lm_m ) );
  data.dA_dx = adapter->jac_data_.GetMfemJacobian( &adapter->submesh_data_.GetSubmeshFESpace(),
                                                   adapter->mesh_data_.GetParentCoords().ParFESpace(), dA_contribs );
  return data;
}

template <typename Adapter>
ContactFormulationResult Nodal<Adapter>::takeGapResult( NodalData& data ) const
{
  const auto* adapter = static_cast<const Adapter*>( this );
  ContactFormulationResult result;
  result.gap.reset( adapter->use_penalty_ ? data.gap_vec.release() : data.g_tilde_vec.release() );
  result.dg_dx.reset( data.dg_tilde_dx.release() );
  return result;
}

template <typename Adapter>
ContactFormulationResult Nodal<Adapter>::computeNodalGaps( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  auto data = computeNodalData( mesh1, mesh2 );
  return takeGapResult( data );
}

template <typename Adapter>
ContactFormulationResult Nodal<Adapter>::computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  const auto* adapter = static_cast<const Adapter*>( this );
  auto data = computeNodalData( mesh1, mesh2 );
  shared::ParVector pressure_vec;
  shared::ParVector force_vec;
  shared::ParSparseMat df_dx;

  if ( adapter->use_penalty_ ) {
    // Penalty mode: p = k * (g_tilde / A)
    pressure_vec = adapter->params_.k * data.gap_vec;
  } else {
    // Read caller-supplied LM values from the current submesh field.
    pressure_vec =
        shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
    adapter->submesh_data_.GetSubmeshPressure().GetTrueDofs( pressure_vec.get() );
    SLIC_ERROR_ROOT_IF( pressure_vec.size() != data.g_tilde_vec.size(),
                        "LM vector size mismatch with contact dofs (g_tilde)." );
  }

  const RealT energy = pressure_vec.dot( data.g_tilde_vec );

  if ( !adapter->use_penalty_ ) {
    // -------------------------------------------------------------------------
    // LM mode: force = G^T * lambda and df/dx = lambda · d^2(g_tilde)/dx^2
    // -------------------------------------------------------------------------
    force_vec = pressure_vec * data.dg_tilde_dx;

    mfem::GridFunction redecomp_lambda( adapter->submesh_data_.GetRedecompGap() );
    mfem::ParGridFunction submesh_lambda(
        const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
    submesh_lambda.SetFromTrueDofs( pressure_vec.get() );
    adapter->submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_lambda, redecomp_lambda );

    df_dx = computeDfDxSecondDerivativesLM( adapter, mesh1, mesh2, data.pairs.view(), redecomp_lambda );
  } else {
    // ---------------------------------------------------------------------------
    // Penalty mode: force and Jacobian include pressure/area coupling terms
    // ---------------------------------------------------------------------------
    auto k_over_a = adapter->params_.k * data.A_vec.inverse( adapter->area_tol_ );
    auto p_over_a = pressure_vec.divide( data.A_vec, adapter->area_tol_ );

    shared::ParSparseMat dp_dx( data.dg_tilde_dx.get() );
    dp_dx->ScaleRows( k_over_a.get() );
    shared::ParSparseMat dp_dx_temp( data.dA_dx.get() );
    dp_dx_temp->ScaleRows( p_over_a.get() );
    dp_dx -= dp_dx_temp;

    force_vec = ( pressure_vec * data.dg_tilde_dx ) + ( data.g_tilde_vec * dp_dx );

    // TODO (EBC): Move transfer path-specific logic out of this file
    mfem::GridFunction redecomp_pressure( adapter->submesh_data_.GetRedecompGap() );
    mfem::ParGridFunction submesh_pressure(
        const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
    submesh_pressure.SetFromTrueDofs( pressure_vec.get() );
    adapter->submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_pressure, redecomp_pressure );

    mfem::GridFunction redecomp_g_tilde( adapter->submesh_data_.GetRedecompGap() );
    mfem::ParGridFunction submesh_g_tilde(
        const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
    submesh_g_tilde.SetFromTrueDofs( data.g_tilde_vec.get() );
    adapter->submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_g_tilde, redecomp_g_tilde );

    mfem::GridFunction redecomp_A( adapter->submesh_data_.GetRedecompGap() );
    mfem::ParGridFunction submesh_A(
        const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
    submesh_A.SetFromTrueDofs( data.A_vec.get() );
    adapter->submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_A, redecomp_A );

    df_dx = computeDfDxSecondDerivativesPenalty( adapter, mesh1, mesh2, data.pairs.view(), redecomp_pressure,
                                                 redecomp_g_tilde, redecomp_A );

    auto pg2_over_asq = ( 2.0 * pressure_vec )
                            .multiplyInPlace( data.g_tilde_vec )
                            .divideInPlace( data.A_vec, adapter->area_tol_ )
                            .divideInPlace( data.A_vec, adapter->area_tol_ );

    auto& submesh_fes = adapter->submesh_data_.GetSubmeshFESpace();
    auto p_over_a_diag = shared::ParSparseMat::diagonalMatrix( submesh_fes.GetComm(), submesh_fes.GlobalTrueVSize(),
                                                               submesh_fes.GetTrueDofOffsets(), p_over_a.get() );
    auto pg2_over_asq_diag = shared::ParSparseMat::diagonalMatrix(
        submesh_fes.GetComm(), submesh_fes.GlobalTrueVSize(), submesh_fes.GetTrueDofOffsets(), pg2_over_asq.get() );

    df_dx -= shared::ParSparseMat::rap( data.dg_tilde_dx, p_over_a_diag, data.dA_dx );
    df_dx -= shared::ParSparseMat::rap( data.dA_dx, p_over_a_diag, data.dg_tilde_dx );
    df_dx += shared::ParSparseMat::rap( data.dA_dx, pg2_over_asq_diag, data.dg_tilde_dx );
    df_dx += dp_dx.transpose() * data.dg_tilde_dx;
    df_dx += data.dg_tilde_dx.transpose() * dp_dx;
  }
  ContactFormulationResult result;
  if ( !adapter->use_penalty_ ) {
    auto df_dp = data.dg_tilde_dx.transpose();
    result.df_dp.reset( df_dp.release() );
  }
  auto gaps = takeGapResult( data );
  result.gap = std::move( gaps.gap );
  result.dg_dx = std::move( gaps.dg_dx );
  result.energy = energy;
  result.pressure.reset( pressure_vec.release() );
  result.force.reset( force_vec.release() );
  result.df_dx.reset( df_dx.release() );
  return result;
}

template <typename Adapter>
shared::ParSparseMat Nodal<Adapter>::computeDfDxSecondDerivativesLM( const Adapter* adapter, MeshData::Viewer mesh1,
                                                                     MeshData::Viewer mesh2,
                                                                     ArrayViewT<const InterfacePair> pairs,
                                                                     const mfem::GridFunction& redecomp_lambda )
{
  const bool use_lor = ( adapter->mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes =
      use_lor ? *adapter->mesh_data_.GetLORMeshFESpace() : adapter->mesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *adapter->mesh_data_.GetRedecompResponse().FESpace();
  const auto& nonmortar_elem_map = adapter->mesh_data_.GetElemMap1();
  const auto& mortar_elem_map = adapter->mesh_data_.GetElemMap2();

  PackedPairJacobianContribs df_nm_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_nm_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs df_m_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, mortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_m_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                     displacement_redecomp_fes, mortar_elem_map, mortar_elem_map );

  df_nm_nm.reserve( pairs.size(), 16 );
  df_nm_m.reserve( pairs.size(), 16 );
  df_m_nm.reserve( pairs.size(), 16 );
  df_m_m.reserve( pairs.size(), 16 );

  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  for ( auto& pair : pairs ) {
    const auto coordinates = adapter->getPairCoordinates( pair, mesh1, mesh2 );
    const auto elem1 = static_cast<int>( pair.element_id1_ );
    const auto node11 = mesh1.getConnectivity()( elem1, 0 );
    const auto node12 = mesh1.getConnectivity()( elem1, 1 );
    const auto elem2 = static_cast<int>( pair.element_id2_ );

    const RealT lambda1 = redecomp_lambda( node11 );
    const RealT lambda2 = redecomp_lambda( node12 );

    const auto hessians = adapter->evaluator_->computeSecondDerivatives( coordinates.view(), false );

    RealT df_dx_blocks[2][2][16];
    for ( int i{ 0 }; i < 2; ++i ) {
      for ( int j{ 0 }; j < 2; ++j ) {
        for ( int k{ 0 }; k < 4; ++k ) {
          for ( int l{ 0 }; l < 4; ++l ) {
            const auto row = node_idx[k + j * 4];
            const auto column = node_idx[l + i * 4];
            df_dx_blocks[i][j][l + k * 4] =
                lambda1 * hessians.gap_hessians( 0, row, column ) + lambda2 * hessians.gap_hessians( 1, row, column );
          }
        }
      }
    }

    df_nm_nm.append( elem1, elem1, df_dx_blocks[0][0], 16 );
    df_nm_m.append( elem1, elem2, df_dx_blocks[0][1], 16 );
    df_m_nm.append( elem2, elem1, df_dx_blocks[1][0], 16 );
    df_m_m.append( elem2, elem2, df_dx_blocks[1][1], 16 );
  }

  std::vector<PackedPairJacobianContribs> df_contribs;
  df_contribs.reserve( 4 );
  df_contribs.push_back( std::move( df_nm_nm ) );
  df_contribs.push_back( std::move( df_nm_m ) );
  df_contribs.push_back( std::move( df_m_nm ) );
  df_contribs.push_back( std::move( df_m_m ) );
  return adapter->jac_data_.GetMfemJacobian( adapter->mesh_data_.GetParentCoords().ParFESpace(),
                                             adapter->mesh_data_.GetParentCoords().ParFESpace(), df_contribs );
}

template <typename Adapter>
shared::ParSparseMat Nodal<Adapter>::computeDfDxSecondDerivativesPenalty(
    const Adapter* adapter, MeshData::Viewer mesh1, MeshData::Viewer mesh2, ArrayViewT<const InterfacePair> pairs,
    const mfem::GridFunction& redecomp_pressure, const mfem::GridFunction& redecomp_g_tilde,
    const mfem::GridFunction& redecomp_A )
{
  const bool use_lor = ( adapter->mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes =
      use_lor ? *adapter->mesh_data_.GetLORMeshFESpace() : adapter->mesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *adapter->mesh_data_.GetRedecompResponse().FESpace();
  const auto& nonmortar_elem_map = adapter->mesh_data_.GetElemMap1();
  const auto& mortar_elem_map = adapter->mesh_data_.GetElemMap2();

  PackedPairJacobianContribs df_nm_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_nm_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs df_m_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, mortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_m_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                     displacement_redecomp_fes, mortar_elem_map, mortar_elem_map );

  df_nm_nm.reserve( pairs.size(), 16 );
  df_nm_m.reserve( pairs.size(), 16 );
  df_m_nm.reserve( pairs.size(), 16 );
  df_m_m.reserve( pairs.size(), 16 );

  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  for ( auto& pair : pairs ) {
    const auto coordinates = adapter->getPairCoordinates( pair, mesh1, mesh2 );
    const auto elem1 = static_cast<int>( pair.element_id1_ );
    const auto node11 = mesh1.getConnectivity()( elem1, 0 );
    const auto node12 = mesh1.getConnectivity()( elem1, 1 );
    const auto elem2 = static_cast<int>( pair.element_id2_ );

    const RealT pressure1 = 2.0 * redecomp_pressure( node11 );
    const RealT pressure2 = 2.0 * redecomp_pressure( node12 );

    if ( pressure1 == 0.0 && pressure2 == 0.0 ) {
      continue;
    }

    const RealT g_p_ainv1 = -redecomp_g_tilde( node11 ) * redecomp_pressure( node11 ) / redecomp_A( node11 );
    const RealT g_p_ainv2 = -redecomp_g_tilde( node12 ) * redecomp_pressure( node12 ) / redecomp_A( node12 );

    const auto hessians = adapter->evaluator_->computeSecondDerivatives( coordinates.view() );

    RealT df_dx_blocks[2][2][16];
    for ( int i{ 0 }; i < 2; ++i ) {
      for ( int j{ 0 }; j < 2; ++j ) {
        for ( int k{ 0 }; k < 4; ++k ) {
          for ( int l{ 0 }; l < 4; ++l ) {
            const auto row = node_idx[k + j * 4];
            const auto column = node_idx[l + i * 4];
            df_dx_blocks[i][j][l + k * 4] = pressure1 * hessians.gap_hessians( 0, row, column ) +
                                            pressure2 * hessians.gap_hessians( 1, row, column ) +
                                            g_p_ainv1 * hessians.area_hessians( 0, row, column ) +
                                            g_p_ainv2 * hessians.area_hessians( 1, row, column );
          }
        }
      }
    }

    df_nm_nm.append( elem1, elem1, df_dx_blocks[0][0], 16 );
    df_nm_m.append( elem1, elem2, df_dx_blocks[0][1], 16 );
    df_m_nm.append( elem2, elem1, df_dx_blocks[1][0], 16 );
    df_m_m.append( elem2, elem2, df_dx_blocks[1][1], 16 );
  }

  std::vector<PackedPairJacobianContribs> df_contribs;
  df_contribs.reserve( 4 );
  df_contribs.push_back( std::move( df_nm_nm ) );
  df_contribs.push_back( std::move( df_nm_m ) );
  df_contribs.push_back( std::move( df_m_nm ) );
  df_contribs.push_back( std::move( df_m_m ) );
  return adapter->jac_data_.GetMfemJacobian( adapter->mesh_data_.GetParentCoords().ParFESpace(),
                                             adapter->mesh_data_.GetParentCoords().ParFESpace(), df_contribs );
}

template <typename Adapter>
ContactFormulationResult QuadraturePoint<Adapter>::computeNodalForces( MeshData::Viewer mesh1,
                                                                       MeshData::Viewer mesh2 ) const
{
  const auto* adapter = static_cast<const Adapter*>( this );
  const auto pairs = adapter->findCandidatePairs( mesh1, mesh2 );
  RealT energy = 0.0;

  SLIC_ERROR_ROOT_IF( !adapter->use_penalty_,
                      "ENERGY_MORTAR quadrature-point enforcement requires penalty enforcement." );

  MfemPairAssembler assembler( adapter->mesh_data_, adapter->jac_data_, mesh1, mesh2, pairs.size() );
  using Definition = EnergyMortarEnergy<decltype( adapter->quadrature_rule_ )>;
  const Definition definition{ adapter->quadrature_rule_, adapter->params_.k, adapter->params_.residual_gap };
  constexpr IndexT pair_dofs = Definition::dof_count;

  for ( const auto& pair : pairs ) {
    const auto coordinates = adapter->getPairCoordinates( pair, mesh1, mesh2 );
    const auto primal = definition.evaluate( coordinates.view() );
    if ( primal.quadrature_result.status != QuadratureStatus::Success )
      reportQuadratureFailure( primal.quadrature_result.status );

    if ( !primal.has_active_qp ) {
      continue;
    }

    const auto derivatives = EnergyPairEvaluator<Definition>::evaluate( coordinates.view(), definition, primal.energy );
    energy += derivatives.energy;

    assembler.addPair( pair, { derivatives.residual.data(), pair_dofs },
                       { derivatives.jacobian.data(), { pair_dofs, pair_dofs } } );
  }

  auto result = assembler.assemble();
  result.energy = energy;
  return result;
}

template class EnergyMortarAdapter<Nodal,
                                   PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>>;
template class EnergyMortarAdapter<QuadraturePoint,
                                   PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>>;
template class EnergyMortarAdapter<Nodal,
                                   PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>>;
template class EnergyMortarAdapter<QuadraturePoint,
                                   PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>>;
template class EnergyMortarAdapter<Nodal,
                                   PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>>;
template class EnergyMortarAdapter<QuadraturePoint,
                                   PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>>;

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
