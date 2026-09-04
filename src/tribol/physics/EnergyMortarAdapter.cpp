// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortarAdapter.hpp"
#include <axom/slic/interface/slic_macros.hpp>
#include "tribol/mesh/MfemData.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

template <template <typename> class EnforcementLocation>
EnergyMortarAdapter<EnforcementLocation>::EnergyMortarAdapter( MfemMeshData& mesh_data, MfemSubmeshData& submesh_data,
                                                               MfemJacobianData& jac_data, double k, double delta,
                                                               int N, bool enzyme_quadrature, bool use_penalty,
                                                               RealT residual_gap )
    // NOTE: mesh1 maps to mesh2_ and mesh2 maps to mesh1_. This is to keep consistent with mesh1_ being non-mortar and
    // mesh2_ being mortar as is typical in the literature, but different from Tribol convention.
    : use_penalty_( use_penalty ), mesh_data_( mesh_data ), submesh_data_( submesh_data ), jac_data_( jac_data )
{
  params_.k = k;
  params_.del = delta;
  params_.N = N;
  params_.enzyme_quadrature = enzyme_quadrature;
  params_.residual_gap = residual_gap;

  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );

  this->init( this );
}

template <typename Adapter>
void Nodal<Adapter>::init( Adapter* adapter )
{
  // Allocate the (pressure) true-dof vector early so host code can set it via tribol::getMfemContactPressure() after
  // the formulation is created. In penalty mode this is overwritten in updateNodalForces(); in LM mode it is treated as
  // the Lagrange multiplier vector (lambda).
  pressure_vec_ =
      shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
  pressure_vec_.fill( 0.0 );
}

template <template <typename> class EnforcementLocation>
void EnergyMortarAdapter<EnforcementLocation>::updateMeshes( MeshData& mesh1, MeshData& mesh2 )
{
  // Maintain the same "flipped" convention as the constructor.
  mesh1_ = &mesh2;
  mesh2_ = &mesh1;
}

template <template <typename> class EnforcementLocation>
void EnergyMortarAdapter<EnforcementLocation>::updateConstantPenaltyStiffness( double mesh1_penalty,
                                                                               double mesh2_penalty )
{
  use_penalty_ = true;
  params_.k = 0.5 * ( mesh1_penalty + mesh2_penalty );
  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );
}

template <template <typename> class EnforcementLocation>
void EnergyMortarAdapter<EnforcementLocation>::setResidualGap( RealT residual_gap )
{
  params_.residual_gap = residual_gap;
  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );
}

template <template <typename> class EnforcementLocation>
void EnergyMortarAdapter<EnforcementLocation>::setInterfacePairs( ArrayT<InterfacePair>&& pairs, int /*check_level*/ )
{
  // TODO: Consider design and how this interacts with binning and CG
  pairs_ = std::move( pairs );
}

template <template <typename> class EnforcementLocation>
void EnergyMortarAdapter<EnforcementLocation>::updateIntegrationRule()
{
  SLIC_WARNING_ROOT( "Update integration rule not implemmented for any method" );
  // TODO: break out integration rule as a separate method
}

template <typename Adapter>
void Nodal<Adapter>::updateNodalGaps()
{
  auto* adapter = static_cast<Adapter*>( this );

  // NOTE: user should have called updateMfemParallelDecomposition() with updated coords before calling this

  // Tribol level data structures for storing gap, area, and derivatives
  auto& redecomp_gap = adapter->submesh_data_.GetRedecompGap();
  mfem::GridFunction redecomp_area( redecomp_gap.FESpace() );
  redecomp_area = 0.0;

  const bool use_lor = ( adapter->mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes =
      use_lor ? *adapter->mesh_data_.GetLORMeshFESpace() : adapter->mesh_data_.GetSubmeshFESpace();
  const auto& pressure_surface_fes =
      use_lor ? *adapter->submesh_data_.GetLORMeshFESpace() : adapter->submesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *adapter->mesh_data_.GetRedecompResponse().FESpace();
  const auto& pressure_redecomp_fes = *adapter->submesh_data_.GetRedecompGap().FESpace();
  const auto& mortar_elem_map = adapter->mesh_data_.GetElemMap1();
  const auto& nonmortar_elem_map = adapter->mesh_data_.GetElemMap2();

  PackedPairJacobianContribs dg_lm_nm( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs dg_lm_m( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs dA_lm_nm( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs dA_lm_m( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );

  dg_lm_nm.reserve( adapter->pairs_.size(), 8 );
  dg_lm_m.reserve( adapter->pairs_.size(), 8 );
  dA_lm_nm.reserve( adapter->pairs_.size(), 8 );
  dA_lm_m.reserve( adapter->pairs_.size(), 8 );

  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  SLIC_ERROR_ROOT_IF( adapter->mesh1_ == nullptr || adapter->mesh2_ == nullptr, "ENERGY_MORTAR meshes not set." );
  auto mesh1_view = adapter->mesh1_->getView();
  auto mesh2_view = adapter->mesh2_->getView();

  // Compute local contributions
  for ( const auto& pair : adapter->pairs_ ) {
    // These need to be flipped, since the pairs are determined with element 1 associated with mesh 1, and we flipped
    // the mesh numbers to be consistent with the literature and since the underlying method integrates on element 1
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    double g_tilde_elem[2];
    double A_elem[2];

    adapter->evaluator_->compute_gtilde_and_area( flipped_pair, mesh1_view, mesh2_view, g_tilde_elem, A_elem );

    if ( A_elem[0] <= 0.0 && A_elem[1] <= 0.0 ) {
      continue;
    }

    auto A_conn = mesh1_view.getConnectivity()( elem1 );

    // Add to nodes of Element A
    redecomp_gap( A_conn[0] ) += g_tilde_elem[0];
    redecomp_gap( A_conn[1] ) += g_tilde_elem[1];

    redecomp_area( A_conn[0] ) += A_elem[0];
    redecomp_area( A_conn[1] ) += A_elem[1];

    // compute g_tilde first derivative
    double dg_dx_node1[8];
    double dg_dx_node2[8];
    // TODO: make grad_gtilde return directly in dg_tilde_dx_blocks format
    adapter->evaluator_->grad_gtilde( flipped_pair, mesh1_view, mesh2_view, dg_dx_node1, dg_dx_node2 );
    double dg_tilde_dx_blocks[2][8];
    for ( int i{ 0 }; i < 4; ++i ) {
      dg_tilde_dx_blocks[0][i * 2] = dg_dx_node1[node_idx[i]];
      dg_tilde_dx_blocks[0][i * 2 + 1] = dg_dx_node2[node_idx[i]];
      dg_tilde_dx_blocks[1][i * 2] = dg_dx_node1[node_idx[i + 4]];
      dg_tilde_dx_blocks[1][i * 2 + 1] = dg_dx_node2[node_idx[i + 4]];
    }
    dg_lm_nm.append( elem1, elem1, dg_tilde_dx_blocks[0], 8 );
    dg_lm_m.append( elem1, elem2, dg_tilde_dx_blocks[1], 8 );

    double dA_dx_node1[8];
    double dA_dx_node2[8];
    // TODO: make grad_trib_area return directly in dA_dx_blocks format
    adapter->evaluator_->grad_trib_area( flipped_pair, mesh1_view, mesh2_view, dA_dx_node1, dA_dx_node2 );
    double dA_dx_blocks[2][8];
    for ( int i{ 0 }; i < 4; ++i ) {
      dA_dx_blocks[0][i * 2] = dA_dx_node1[node_idx[i]];
      dA_dx_blocks[0][i * 2 + 1] = dA_dx_node2[node_idx[i]];
      dA_dx_blocks[1][i * 2] = dA_dx_node1[node_idx[i + 4]];
      dA_dx_blocks[1][i * 2 + 1] = dA_dx_node2[node_idx[i + 4]];
    }
    dA_lm_nm.append( elem1, elem1, dA_dx_blocks[0], 8 );
    dA_lm_m.append( elem1, elem2, dA_dx_blocks[1], 8 );
  }

  // Move gap and area to submesh level vectors
  mfem::ParLinearForm g_tilde_linear_form(
      const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
  adapter->submesh_data_.GetSubmeshGap( g_tilde_linear_form );
  auto& P_submesh = *adapter->submesh_data_.GetSubmeshFESpace().GetProlongationMatrix();
  adapter->g_tilde_vec_ =
      shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
  adapter->g_tilde_vec_.fill( 0.0 );
  P_submesh.MultTranspose( g_tilde_linear_form, adapter->g_tilde_vec_.get() );

  mfem::Array<int> rows_to_elim;
  if ( !adapter->tied_contact_ && adapter->use_penalty_ ) {
    rows_to_elim.Reserve( adapter->g_tilde_vec_.size() );
    for ( int i{ 0 }; i < adapter->g_tilde_vec_.size(); ++i ) {
      if ( adapter->g_tilde_vec_[i] > 0.0 ) {
        adapter->g_tilde_vec_[i] = 0.0;
        rows_to_elim.push_back( i );
      }
    }
  }

  mfem::ParLinearForm A_linear_form(
      const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
  adapter->submesh_data_.GetPressureTransfer().RedecompToSubmesh( redecomp_area, A_linear_form );
  adapter->A_vec_ =
      shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
  adapter->A_vec_.fill( 0.0 );
  P_submesh.MultTranspose( A_linear_form, adapter->A_vec_.get() );

  adapter->gap_vec_ = adapter->g_tilde_vec_.divide( adapter->A_vec_, adapter->area_tol_ );

  // Move gap and area derivatives to (pressure true-dof rows, displacement true-dof cols)
  std::vector<PackedPairJacobianContribs> dg_contribs;
  dg_contribs.reserve( 2 );
  dg_contribs.push_back( std::move( dg_lm_nm ) );
  dg_contribs.push_back( std::move( dg_lm_m ) );
  adapter->dg_tilde_dx_ = adapter->jac_data_.GetMfemJacobian(
      &adapter->submesh_data_.GetSubmeshFESpace(), adapter->mesh_data_.GetParentCoords().ParFESpace(), dg_contribs );
  if ( !adapter->tied_contact_ && adapter->use_penalty_ ) {
    // technically, we should do this on all the vectors/matrices below, but it looks like the mutliplication operators
    // below will zero them out anyway
    adapter->dg_tilde_dx_.eliminateRows( rows_to_elim );
  }

  std::vector<PackedPairJacobianContribs> dA_contribs;
  dA_contribs.reserve( 2 );
  dA_contribs.push_back( std::move( dA_lm_nm ) );
  dA_contribs.push_back( std::move( dA_lm_m ) );
  adapter->dA_dx_ = adapter->jac_data_.GetMfemJacobian(
      &adapter->submesh_data_.GetSubmeshFESpace(), adapter->mesh_data_.GetParentCoords().ParFESpace(), dA_contribs );
}

template <typename Adapter>
void Nodal<Adapter>::updateNodalForces()
{
  auto* adapter = static_cast<Adapter*>( this );

  // NOTE: user should have called updateNodalGaps() with updated coords before calling this

  if ( adapter->use_penalty_ ) {
    // Penalty mode: p = k * (g_tilde / A)
    adapter->pressure_vec_ = adapter->params_.k * adapter->gap_vec_;
  } else {
    // LM mode: adapter->pressure_vec_ is treated as the Lagrange multiplier vector (lambda)
    SLIC_ERROR_ROOT_IF( adapter->submesh_data_.GetSubmeshFESpace().GetTrueVSize() != adapter->pressure_vec_.size(),
                        "LM vector is not initialized. Call tribol::update() once to initialize the formulation." );
    SLIC_ERROR_ROOT_IF( adapter->pressure_vec_.size() != adapter->g_tilde_vec_.size(),
                        "LM vector size mismatch with contact dofs (g_tilde)." );
  }

  adapter->energy_ = adapter->pressure_vec_.dot( adapter->g_tilde_vec_ );

  if ( !adapter->use_penalty_ ) {
    // -------------------------------------------------------------------------
    // LM mode: force = G^T * lambda and df/dx = lambda · d^2(g_tilde)/dx^2
    // -------------------------------------------------------------------------
    adapter->force_vec_ = adapter->pressure_vec_ * adapter->dg_tilde_dx_;

    mfem::GridFunction redecomp_lambda( adapter->submesh_data_.GetRedecompGap() );
    mfem::ParGridFunction submesh_lambda(
        const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
    submesh_lambda.SetFromTrueDofs( adapter->pressure_vec_.get() );
    adapter->submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_lambda, redecomp_lambda );

    adapter->df_dx_ = computeDfDxSecondDerivativesLM( adapter, redecomp_lambda );
    return;
  }

  // ---------------------------------------------------------------------------
  // Penalty mode: force and Jacobian include pressure/area coupling terms
  // ---------------------------------------------------------------------------
  auto k_over_a = adapter->params_.k * adapter->A_vec_.inverse( adapter->area_tol_ );
  auto p_over_a = adapter->pressure_vec_.divide( adapter->A_vec_, adapter->area_tol_ );

  shared::ParSparseMat dp_dx( adapter->dg_tilde_dx_.get() );
  dp_dx->ScaleRows( k_over_a.get() );
  shared::ParSparseMat dp_dx_temp( adapter->dA_dx_.get() );
  dp_dx_temp->ScaleRows( p_over_a.get() );
  dp_dx -= dp_dx_temp;

  adapter->force_vec_ = ( adapter->pressure_vec_ * adapter->dg_tilde_dx_ ) + ( adapter->g_tilde_vec_ * dp_dx );

  // TODO (EBC): Move transfer path-specific logic out of this file
  mfem::GridFunction redecomp_pressure( adapter->submesh_data_.GetRedecompGap() );
  mfem::ParGridFunction submesh_pressure(
      const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
  submesh_pressure.SetFromTrueDofs( adapter->pressure_vec_.get() );
  adapter->submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_pressure, redecomp_pressure );

  mfem::GridFunction redecomp_g_tilde( adapter->submesh_data_.GetRedecompGap() );
  mfem::ParGridFunction submesh_g_tilde(
      const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
  submesh_g_tilde.SetFromTrueDofs( adapter->g_tilde_vec_.get() );
  adapter->submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_g_tilde, redecomp_g_tilde );

  mfem::GridFunction redecomp_A( adapter->submesh_data_.GetRedecompGap() );
  mfem::ParGridFunction submesh_A(
      const_cast<mfem::ParFiniteElementSpace*>( &adapter->submesh_data_.GetSubmeshFESpace() ) );
  submesh_A.SetFromTrueDofs( adapter->A_vec_.get() );
  adapter->submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_A, redecomp_A );

  adapter->df_dx_ = computeDfDxSecondDerivativesPenalty( adapter, redecomp_pressure, redecomp_g_tilde, redecomp_A );

  auto pg2_over_asq = ( 2.0 * adapter->pressure_vec_ )
                          .multiplyInPlace( adapter->g_tilde_vec_ )
                          .divideInPlace( adapter->A_vec_, adapter->area_tol_ )
                          .divideInPlace( adapter->A_vec_, adapter->area_tol_ );

  auto& submesh_fes = adapter->submesh_data_.GetSubmeshFESpace();
  auto p_over_a_diag = shared::ParSparseMat::diagonalMatrix( submesh_fes.GetComm(), submesh_fes.GlobalTrueVSize(),
                                                             submesh_fes.GetTrueDofOffsets(), p_over_a.get() );
  auto pg2_over_asq_diag = shared::ParSparseMat::diagonalMatrix( submesh_fes.GetComm(), submesh_fes.GlobalTrueVSize(),
                                                                 submesh_fes.GetTrueDofOffsets(), pg2_over_asq.get() );

  adapter->df_dx_ -= shared::ParSparseMat::rap( adapter->dg_tilde_dx_, p_over_a_diag, adapter->dA_dx_ );
  adapter->df_dx_ -= shared::ParSparseMat::rap( adapter->dA_dx_, p_over_a_diag, adapter->dg_tilde_dx_ );
  adapter->df_dx_ += shared::ParSparseMat::rap( adapter->dA_dx_, pg2_over_asq_diag, adapter->dg_tilde_dx_ );
  adapter->df_dx_ += dp_dx.transpose() * adapter->dg_tilde_dx_;
  adapter->df_dx_ += adapter->dg_tilde_dx_.transpose() * dp_dx;
}

template <template <typename> class EnforcementLocation>
RealT EnergyMortarAdapter<EnforcementLocation>::computeTimeStep()
{
  SLIC_INFO_ROOT( "computeTimestep() not implemented for EnergyMortar" );
  // TODO: implement timestep calculation
  return 1.0;
}

template <typename Adapter>
shared::ParSparseMat Nodal<Adapter>::computeDfDxSecondDerivativesLM( Adapter* adapter,
                                                                     const mfem::GridFunction& redecomp_lambda )
{
  const bool use_lor = ( adapter->mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes =
      use_lor ? *adapter->mesh_data_.GetLORMeshFESpace() : adapter->mesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *adapter->mesh_data_.GetRedecompResponse().FESpace();
  const auto& mortar_elem_map = adapter->mesh_data_.GetElemMap1();
  const auto& nonmortar_elem_map = adapter->mesh_data_.GetElemMap2();

  PackedPairJacobianContribs df_nm_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_nm_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs df_m_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, mortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_m_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                     displacement_redecomp_fes, mortar_elem_map, mortar_elem_map );

  df_nm_nm.reserve( adapter->pairs_.size(), 16 );
  df_nm_m.reserve( adapter->pairs_.size(), 16 );
  df_m_nm.reserve( adapter->pairs_.size(), 16 );
  df_m_m.reserve( adapter->pairs_.size(), 16 );

  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  SLIC_ERROR_ROOT_IF( adapter->mesh1_ == nullptr || adapter->mesh2_ == nullptr, "ENERGY_MORTAR meshes not set." );
  auto mesh1_view = adapter->mesh1_->getView();
  auto mesh2_view = adapter->mesh2_->getView();

  for ( auto& pair : adapter->pairs_ ) {
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto node11 = mesh1_view.getConnectivity()( elem1, 0 );
    const auto node12 = mesh1_view.getConnectivity()( elem1, 1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    const RealT lambda1 = redecomp_lambda( node11 );
    const RealT lambda2 = redecomp_lambda( node12 );

    double d2g_dx2_node1[64];
    double d2g_dx2_node2[64];
    adapter->evaluator_->d2_g2tilde( flipped_pair, mesh1_view, mesh2_view, d2g_dx2_node1, d2g_dx2_node2 );

    double df_dx_blocks[2][2][16];
    for ( int i{ 0 }; i < 2; ++i ) {
      for ( int j{ 0 }; j < 2; ++j ) {
        for ( int k{ 0 }; k < 4; ++k ) {
          for ( int l{ 0 }; l < 4; ++l ) {
            const auto idx = node_idx[l + i * 4] + node_idx[k + j * 4] * 8;
            df_dx_blocks[i][j][l + k * 4] = lambda1 * d2g_dx2_node1[idx] + lambda2 * d2g_dx2_node2[idx];
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
shared::ParSparseMat Nodal<Adapter>::computeDfDxSecondDerivativesPenalty( Adapter* adapter,
                                                                          const mfem::GridFunction& redecomp_pressure,
                                                                          const mfem::GridFunction& redecomp_g_tilde,
                                                                          const mfem::GridFunction& redecomp_A )
{
  const bool use_lor = ( adapter->mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes =
      use_lor ? *adapter->mesh_data_.GetLORMeshFESpace() : adapter->mesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *adapter->mesh_data_.GetRedecompResponse().FESpace();
  const auto& mortar_elem_map = adapter->mesh_data_.GetElemMap1();
  const auto& nonmortar_elem_map = adapter->mesh_data_.GetElemMap2();

  PackedPairJacobianContribs df_nm_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_nm_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs df_m_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, mortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_m_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                     displacement_redecomp_fes, mortar_elem_map, mortar_elem_map );

  df_nm_nm.reserve( adapter->pairs_.size(), 16 );
  df_nm_m.reserve( adapter->pairs_.size(), 16 );
  df_m_nm.reserve( adapter->pairs_.size(), 16 );
  df_m_m.reserve( adapter->pairs_.size(), 16 );

  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  SLIC_ERROR_ROOT_IF( adapter->mesh1_ == nullptr || adapter->mesh2_ == nullptr, "ENERGY_MORTAR meshes not set." );
  auto mesh1_view = adapter->mesh1_->getView();
  auto mesh2_view = adapter->mesh2_->getView();

  for ( auto& pair : adapter->pairs_ ) {
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto node11 = mesh1_view.getConnectivity()( elem1, 0 );
    const auto node12 = mesh1_view.getConnectivity()( elem1, 1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    const RealT pressure1 = 2.0 * redecomp_pressure( node11 );
    const RealT pressure2 = 2.0 * redecomp_pressure( node12 );

    if ( pressure1 == 0.0 && pressure2 == 0.0 ) {
      continue;
    }

    const RealT g_p_ainv1 = -redecomp_g_tilde( node11 ) * redecomp_pressure( node11 ) / redecomp_A( node11 );
    const RealT g_p_ainv2 = -redecomp_g_tilde( node12 ) * redecomp_pressure( node12 ) / redecomp_A( node12 );

    double d2g_dx2_node1[64];
    double d2g_dx2_node2[64];
    adapter->evaluator_->d2_g2tilde( flipped_pair, mesh1_view, mesh2_view, d2g_dx2_node1, d2g_dx2_node2 );

    double d2A_dx2_node1[64];
    double d2A_dx2_node2[64];
    adapter->evaluator_->compute_d2A_d2u( flipped_pair, mesh1_view, mesh2_view, d2A_dx2_node1, d2A_dx2_node2 );

    double df_dx_blocks[2][2][16];
    for ( int i{ 0 }; i < 2; ++i ) {
      for ( int j{ 0 }; j < 2; ++j ) {
        for ( int k{ 0 }; k < 4; ++k ) {
          for ( int l{ 0 }; l < 4; ++l ) {
            const auto idx = node_idx[l + i * 4] + node_idx[k + j * 4] * 8;
            df_dx_blocks[i][j][l + k * 4] = pressure1 * d2g_dx2_node1[idx] + pressure2 * d2g_dx2_node2[idx] +
                                            g_p_ainv1 * d2A_dx2_node1[idx] + g_p_ainv2 * d2A_dx2_node2[idx];
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
void QuadraturePoint<Adapter>::updateNodalForces()
{
  auto* adapter = static_cast<Adapter*>( this );

  SLIC_ERROR_ROOT_IF( !adapter->use_penalty_,
                      "ENERGY_MORTAR quadrature-point enforcement requires penalty enforcement." );

  const bool use_lor = ( adapter->mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes =
      use_lor ? *adapter->mesh_data_.GetLORMeshFESpace() : adapter->mesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *adapter->mesh_data_.GetRedecompResponse().FESpace();
  const auto& mortar_elem_map = adapter->mesh_data_.GetElemMap1();
  const auto& nonmortar_elem_map = adapter->mesh_data_.GetElemMap2();

  PackedPairJacobianContribs df_nm_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_nm_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs df_m_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, mortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_m_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                     displacement_redecomp_fes, mortar_elem_map, mortar_elem_map );

  df_nm_nm.reserve( adapter->pairs_.size(), 16 );
  df_nm_m.reserve( adapter->pairs_.size(), 16 );
  df_m_nm.reserve( adapter->pairs_.size(), 16 );
  df_m_m.reserve( adapter->pairs_.size(), 16 );

  mfem::GridFunction redecomp_force( const_cast<mfem::FiniteElementSpace*>( &displacement_redecomp_fes ) );
  redecomp_force = 0.0;
  const int scalar_size = redecomp_force.FESpace()->GetVSize() / redecomp_force.FESpace()->GetVDim();
  adapter->energy_ = 0.0;

  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  SLIC_ERROR_ROOT_IF( adapter->mesh1_ == nullptr || adapter->mesh2_ == nullptr, "ENERGY_MORTAR meshes not set." );
  auto mesh1_view = adapter->mesh1_->getView();
  auto mesh2_view = adapter->mesh2_->getView();

  for ( const auto& pair : adapter->pairs_ ) {
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );
    const auto qp_data =
        adapter->evaluator_->compute_quadrature_point_penalty_data( flipped_pair, mesh1_view, mesh2_view );

    if ( !qp_data.has_active_qp ) {
      continue;
    }

    adapter->energy_ += qp_data.energy;

    auto A_conn = mesh1_view.getConnectivity()( elem1 );
    auto B_conn = mesh2_view.getConnectivity()( elem2 );
    redecomp_force( A_conn[0] ) += qp_data.force[0];
    redecomp_force( scalar_size + A_conn[0] ) += qp_data.force[1];
    redecomp_force( A_conn[1] ) += qp_data.force[2];
    redecomp_force( scalar_size + A_conn[1] ) += qp_data.force[3];
    redecomp_force( B_conn[0] ) += qp_data.force[4];
    redecomp_force( scalar_size + B_conn[0] ) += qp_data.force[5];
    redecomp_force( B_conn[1] ) += qp_data.force[6];
    redecomp_force( scalar_size + B_conn[1] ) += qp_data.force[7];

    double df_dx_blocks[2][2][16];
    for ( int i{ 0 }; i < 2; ++i ) {
      for ( int j{ 0 }; j < 2; ++j ) {
        for ( int k{ 0 }; k < 4; ++k ) {
          for ( int l{ 0 }; l < 4; ++l ) {
            const auto idx = node_idx[l + i * 4] + node_idx[k + j * 4] * 8;
            df_dx_blocks[i][j][l + k * 4] = qp_data.stiffness[idx];
          }
        }
      }
    }

    df_nm_nm.append( elem1, elem1, df_dx_blocks[0][0], 16 );
    df_nm_m.append( elem1, elem2, df_dx_blocks[0][1], 16 );
    df_m_nm.append( elem2, elem1, df_dx_blocks[1][0], 16 );
    df_m_m.append( elem2, elem2, df_dx_blocks[1][1], 16 );
  }

  auto* parent_fes = adapter->mesh_data_.GetParentCoords().ParFESpace();
  adapter->force_vec_ = shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( parent_fes ) );
  adapter->force_vec_.fill( 0.0 );
  mfem::Vector parent_force( parent_fes->GetVSize() );
  parent_force = 0.0;
  adapter->mesh_data_.GetParentRedecompTransfer().RedecompToParent( redecomp_force, parent_force );
  parent_fes->GetProlongationMatrix()->MultTranspose( parent_force, adapter->force_vec_.get() );

  std::vector<PackedPairJacobianContribs> df_contribs;
  df_contribs.reserve( 4 );
  df_contribs.push_back( std::move( df_nm_nm ) );
  df_contribs.push_back( std::move( df_nm_m ) );
  df_contribs.push_back( std::move( df_m_nm ) );
  df_contribs.push_back( std::move( df_m_m ) );
  adapter->df_dx_ = adapter->jac_data_.GetMfemJacobian( parent_fes, parent_fes, df_contribs );
}

template <template <typename> class EnforcementLocation>
std::unique_ptr<mfem::HypreParMatrix> EnergyMortarAdapter<EnforcementLocation>::getMfemDfDx() const
{
  return std::unique_ptr<mfem::HypreParMatrix>( df_dx_.release() );
}

template <typename Adapter>
std::unique_ptr<mfem::HypreParMatrix> Nodal<Adapter>::getMfemDfDp() const
{
  if ( static_cast<const Adapter*>( this )->use_penalty_ ) {
    return nullptr;
  }
  // LM mode: df/dlambda = (d g_tilde / dx)^T
  auto df_dlambda = dg_tilde_dx_.transpose();
  return std::unique_ptr<mfem::HypreParMatrix>( df_dlambda.release() );
}

template class EnergyMortarAdapter<Nodal>;
template class EnergyMortarAdapter<QuadraturePoint>;

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
