// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortarAdapter.hpp"
#include "tribol/mesh/MfemData.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

namespace {

}  // namespace

EnergyMortarAdapter::EnergyMortarAdapter( MfemMeshData& mesh_data, MfemSubmeshData& submesh_data,
                                          MfemJacobianData& jac_data, MeshData& mesh1, MeshData& mesh2, double k,
                                          double delta, int N, bool enzyme_quadrature, bool use_penalty )
    // NOTE: mesh1 maps to mesh2_ and mesh2 maps to mesh1_. This is to keep consistent with mesh1_ being non-mortar and
    // mesh2_ being mortar as is typical in the literature, but different from Tribol convention.
    : use_penalty_( use_penalty ),
      mesh_data_( mesh_data ),
      submesh_data_( submesh_data ),
      jac_data_( jac_data ),
      mesh1_( mesh2 ),
      mesh2_( mesh1 )
{
  if ( mesh1.numberOfNodes() > 0 && mesh2.numberOfNodes() > 0 ) {
    SLIC_ERROR_ROOT_IF( mesh1.spatialDimension() != 2 || mesh2.spatialDimension() != 2,
                        "ENERGY_MORTAR requires 2D meshes." );
  }

  params_.k = k;
  params_.del = delta;
  params_.N = N;
  params_.enzyme_quadrature = enzyme_quadrature;

  evaluator_ = std::make_unique<ContactEvaluator>( params_ );
}

void EnergyMortarAdapter::setInterfacePairs( ArrayT<InterfacePair>&& pairs, int /*check_level*/ )
{
  // TODO: improved pair identification
  pairs_ = std::move( pairs );
}

void EnergyMortarAdapter::updateIntegrationRule()
{
  // TODO: break out integration rule as a separate method
}

void EnergyMortarAdapter::updateNodalGaps()
{
  // NOTE: user should have called updateMfemParallelDecomposition() with updated coords before calling this

  // Tribol level data structures for storing gap, area, and derivatives
  auto& redecomp_gap = submesh_data_.GetRedecompGap();
  mfem::GridFunction redecomp_area( redecomp_gap.FESpace() );
  redecomp_area = 0.0;

  const bool use_lor = ( mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes = use_lor ? *mesh_data_.GetLORMeshFESpace() : mesh_data_.GetSubmeshFESpace();
  const auto& pressure_surface_fes = use_lor ? *submesh_data_.GetLORMeshFESpace() : submesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *mesh_data_.GetRedecompResponse().FESpace();
  const auto& pressure_redecomp_fes = *submesh_data_.GetRedecompGap().FESpace();
  const auto& mortar_elem_map = mesh_data_.GetElemMap1();
  const auto& nonmortar_elem_map = mesh_data_.GetElemMap2();

  PackedPairJacobianContribs dg_lm_nm( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs dg_lm_m( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs dA_lm_nm( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs dA_lm_m( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );

  dg_lm_nm.reserve( pairs_.size(), 8 );
  dg_lm_m.reserve( pairs_.size(), 8 );
  dA_lm_nm.reserve( pairs_.size(), 8 );
  dA_lm_m.reserve( pairs_.size(), 8 );

  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  auto mesh1_view = mesh1_.getView();
  auto mesh2_view = mesh2_.getView();

  // Compute local contributions
  for ( const auto& pair : pairs_ ) {
    // These need to be flipped, since the pairs are determined with element 1 associated with mesh 1, and we flipped
    // the mesh numbers to be consistent with the literature and since the underlying method integrates on element 1
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    double g_tilde_elem[2];
    double A_elem[2];

    evaluator_->gtilde_and_area( flipped_pair, mesh1_view, mesh2_view, g_tilde_elem, A_elem );

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
    evaluator_->grad_gtilde( flipped_pair, mesh1_view, mesh2_view, dg_dx_node1, dg_dx_node2 );
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
    evaluator_->grad_trib_area( flipped_pair, mesh1_view, mesh2_view, dA_dx_node1, dA_dx_node2 );
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
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_data_.GetSubmeshGap( g_tilde_linear_form );
  auto& P_submesh = *submesh_data_.GetSubmeshFESpace().GetProlongationMatrix();
  g_tilde_vec_ = shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  g_tilde_vec_.fill( 0.0 );
  P_submesh.MultTranspose( g_tilde_linear_form, g_tilde_vec_.get() );

  mfem::Array<int> rows_to_elim;
  if ( !tied_contact_ && use_penalty_ ) {
    rows_to_elim.Reserve( g_tilde_vec_.size() );
    for ( int i{ 0 }; i < g_tilde_vec_.size(); ++i ) {
      if ( g_tilde_vec_[i] > 0.0 ) {
        g_tilde_vec_[i] = 0.0;
        rows_to_elim.push_back( i );
      }
    }
  }

  mfem::ParLinearForm A_linear_form( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_data_.GetPressureTransfer().RedecompToSubmesh( redecomp_area, A_linear_form );
  A_vec_ = shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  A_vec_.fill( 0.0 );
  P_submesh.MultTranspose( A_linear_form, A_vec_.get() );

  gap_vec_ = g_tilde_vec_.divide( A_vec_, area_tol_ );

  // Move gap and area derivatives to (pressure true-dof rows, displacement true-dof cols)
  std::vector<PackedPairJacobianContribs> dg_contribs;
  dg_contribs.reserve( 2 );
  dg_contribs.push_back( std::move( dg_lm_nm ) );
  dg_contribs.push_back( std::move( dg_lm_m ) );
  dg_tilde_dx_ = jac_data_.GetMfemJacobian( &submesh_data_.GetSubmeshFESpace(),
                                            mesh_data_.GetParentCoords().ParFESpace(), dg_contribs );
  if ( !tied_contact_ && use_penalty_ ) {
    // technically, we should do this on all the vectors/matrices below, but it looks like the mutliplication operators
    // below will zero them out anyway
    dg_tilde_dx_.eliminateRows( rows_to_elim );
  }

  std::vector<PackedPairJacobianContribs> dA_contribs;
  dA_contribs.reserve( 2 );
  dA_contribs.push_back( std::move( dA_lm_nm ) );
  dA_contribs.push_back( std::move( dA_lm_m ) );
  dA_dx_ = jac_data_.GetMfemJacobian( &submesh_data_.GetSubmeshFESpace(), mesh_data_.GetParentCoords().ParFESpace(),
                                      dA_contribs );
}

void EnergyMortarAdapter::updateNodalForces()
{
  // NOTE: user should have called updateNodalGaps() with updated coords before calling this

  // compute nodal pressures. these are used in the Hessian vector product below so we don't have to assemble a Hessian
  // NOTE: in general, pressure should likely be set by the host code

  pressure_vec_ = params_.k * gap_vec_;

  energy_ = pressure_vec_.dot( g_tilde_vec_ );

  auto k_over_a = params_.k * A_vec_.inverse( area_tol_ );

  auto p_over_a = pressure_vec_.divide( A_vec_, area_tol_ );

  shared::ParSparseMat dp_dx( dg_tilde_dx_.get() );
  dp_dx->ScaleRows( k_over_a.get() );
  shared::ParSparseMat dp_dx_temp( dA_dx_.get() );
  dp_dx_temp->ScaleRows( p_over_a.get() );
  dp_dx -= dp_dx_temp;

  force_vec_ = ( pressure_vec_ * dg_tilde_dx_ ) + ( g_tilde_vec_ * dp_dx );

  // TODO (EBC): Move transfer-specific logic out of this file
  const bool use_lor = ( mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes = use_lor ? *mesh_data_.GetLORMeshFESpace() : mesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *mesh_data_.GetRedecompResponse().FESpace();
  const auto& mortar_elem_map = mesh_data_.GetElemMap1();
  const auto& nonmortar_elem_map = mesh_data_.GetElemMap2();

  PackedPairJacobianContribs df_nm_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_nm_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs df_m_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, mortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_m_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                     displacement_redecomp_fes, mortar_elem_map, mortar_elem_map );

  df_nm_nm.reserve( pairs_.size(), 16 );
  df_nm_m.reserve( pairs_.size(), 16 );
  df_m_nm.reserve( pairs_.size(), 16 );
  df_m_m.reserve( pairs_.size(), 16 );

  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  mfem::GridFunction redecomp_pressure( submesh_data_.GetRedecompGap() );
  mfem::ParGridFunction submesh_pressure(
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_pressure.SetFromTrueDofs( pressure_vec_.get() );
  submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_pressure, redecomp_pressure );

  mfem::GridFunction redecomp_g_tilde( submesh_data_.GetRedecompGap() );
  mfem::ParGridFunction submesh_g_tilde(
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_g_tilde.SetFromTrueDofs( g_tilde_vec_.get() );
  submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_g_tilde, redecomp_g_tilde );

  mfem::GridFunction redecomp_A( submesh_data_.GetRedecompGap() );
  mfem::ParGridFunction submesh_A( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_A.SetFromTrueDofs( A_vec_.get() );
  submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_A, redecomp_A );

  auto mesh1_view = mesh1_.getView();
  auto mesh2_view = mesh2_.getView();

  // get pairwise action of second derivatives of gaps and pressure for stiffness contribution
  for ( auto& pair : pairs_ ) {
    // These need to be flipped, since the pairs are determined with element 1 associated with mesh 1, and we flipped
    // the mesh numbers to be consistent with the literature and since the underlying method integrates on element 1
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
    // ordering: [dg/(dx0dx0) dg/(dy0dx0) dg/(dx1dx0) ...]
    evaluator_->d2_g2tilde( flipped_pair, mesh1_view, mesh2_view, d2g_dx2_node1, d2g_dx2_node2 );

    double d2A_dx2_node1[64];
    double d2A_dx2_node2[64];
    evaluator_->compute_d2A_d2u( flipped_pair, mesh1_view, mesh2_view, d2A_dx2_node1, d2A_dx2_node2 );

    double df_dx_blocks[2][2][16];
    for ( int i{ 0 }; i < 2; ++i ) {
      for ( int j{ 0 }; j < 2; ++j ) {
        for ( int k{ 0 }; k < 4; ++k ) {
          for ( int l{ 0 }; l < 4; ++l ) {
            df_dx_blocks[i][j][l + k * 4] = pressure1 * d2g_dx2_node1[node_idx[l + i * 4] + node_idx[k + j * 4] * 8] +
                                            pressure2 * d2g_dx2_node2[node_idx[l + i * 4] + node_idx[k + j * 4] * 8] +
                                            g_p_ainv1 * d2A_dx2_node1[node_idx[l + i * 4] + node_idx[k + j * 4] * 8] +
                                            g_p_ainv2 * d2A_dx2_node2[node_idx[l + i * 4] + node_idx[k + j * 4] * 8];
          }
        }
      }
    }

    df_nm_nm.append( elem1, elem1, df_dx_blocks[0][0], 16 );
    df_nm_m.append( elem1, elem2, df_dx_blocks[0][1], 16 );
    df_m_nm.append( elem2, elem1, df_dx_blocks[1][0], 16 );
    df_m_m.append( elem2, elem2, df_dx_blocks[1][1], 16 );
  }

  // Move stiffness contributions to (parent true-dof rows, parent true-dof cols)
  std::vector<PackedPairJacobianContribs> df_contribs;
  df_contribs.reserve( 4 );
  df_contribs.push_back( std::move( df_nm_nm ) );
  df_contribs.push_back( std::move( df_nm_m ) );
  df_contribs.push_back( std::move( df_m_nm ) );
  df_contribs.push_back( std::move( df_m_m ) );
  df_dx_ = jac_data_.GetMfemJacobian( mesh_data_.GetParentCoords().ParFESpace(),
                                      mesh_data_.GetParentCoords().ParFESpace(), df_contribs );

  auto pg2_over_asq = ( 2.0 * pressure_vec_ )
                          .multiplyInPlace( g_tilde_vec_ )
                          .divideInPlace( A_vec_, area_tol_ )
                          .divideInPlace( A_vec_, area_tol_ );

  auto& submesh_fes = submesh_data_.GetSubmeshFESpace();
  auto p_over_a_diag = shared::ParSparseMat::diagonalMatrix( submesh_fes.GetComm(), submesh_fes.GlobalTrueVSize(),
                                                             submesh_fes.GetTrueDofOffsets(), p_over_a.get() );
  auto pg2_over_asq_diag = shared::ParSparseMat::diagonalMatrix( submesh_fes.GetComm(), submesh_fes.GlobalTrueVSize(),
                                                                 submesh_fes.GetTrueDofOffsets(), pg2_over_asq.get() );

  df_dx_ -= shared::ParSparseMat::rap( dg_tilde_dx_, p_over_a_diag, dA_dx_ );
  df_dx_ -= shared::ParSparseMat::rap( dA_dx_, p_over_a_diag, dg_tilde_dx_ );
  df_dx_ += shared::ParSparseMat::rap( dA_dx_, pg2_over_asq_diag, dg_tilde_dx_ );
  df_dx_ += dp_dx.transpose() * dg_tilde_dx_;
  df_dx_ += dg_tilde_dx_.transpose() * dp_dx;
}

RealT EnergyMortarAdapter::computeTimeStep()
{
  // TODO: implement timestep calculation
  return 1.0;
}

void EnergyMortarAdapter::compute_df_du_lagrange( const mfem::HypreParVector& lambda,
                                                  std::unique_ptr<mfem::HypreParMatrix>& df_du )
{
  // Convert Lambda to redecomp space for element wise access
  mfem::GridFunction redecomp_lambda( submesh_data_.GetRedecompGap() );
  mfem::ParGridFunction submesh_lambda(
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_lambda.SetFromTrueDofs( lambda );
  submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_lambda, redecomp_lambda );

  const bool use_lor = ( mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes = use_lor ? *mesh_data_.GetLORMeshFESpace() : mesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *mesh_data_.GetRedecompResponse().FESpace();
  const auto& mortar_elem_map = mesh_data_.GetElemMap1();
  const auto& nonmortar_elem_map = mesh_data_.GetElemMap2();

  PackedPairJacobianContribs df_nm_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_nm_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs df_m_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, mortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_m_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                     displacement_redecomp_fes, mortar_elem_map, mortar_elem_map );

  df_nm_nm.reserve( pairs_.size(), 16 );
  df_nm_m.reserve( pairs_.size(), 16 );
  df_m_nm.reserve( pairs_.size(), 16 );
  df_m_m.reserve( pairs_.size(), 16 );

  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  auto mesh1_view = mesh1_.getView();
  auto mesh2_view = mesh2_.getView();
  // Loop over element pairs and compute Hessian contributions
  for ( auto& pair : pairs_ ) {
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto node11 = mesh1_view.getConnectivity()( elem1, 0 );
    const auto node12 = mesh1_view.getConnectivity()( elem1, 1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    // Get lambda values at nodes (no factor of 2 in Lagrange mode)
    const RealT lambda1 = redecomp_lambda( node11 );
    const RealT lambda2 = redecomp_lambda( node12 );

    // if ( lambda1 == 0.0 && lambda2 == 0.0 ) {
    //   continue;
    // }

    // Compute Hessian of g̃
    double d2g_dx2_node1[64];
    double d2g_dx2_node2[64];
    evaluator_->d2_g2tilde( flipped_pair, mesh1_view, mesh2_view, d2g_dx2_node1, d2g_dx2_node2 );

    // Assemble df/du blocks: df/du = λ · d²g̃/du²
    double df_dx_blocks[2][2][16];
    for ( int i{ 0 }; i < 2; ++i ) {
      for ( int j{ 0 }; j < 2; ++j ) {
        for ( int k{ 0 }; k < 4; ++k ) {
          for ( int l{ 0 }; l < 4; ++l ) {
            df_dx_blocks[i][j][l + k * 4] = lambda1 * d2g_dx2_node1[node_idx[l + i * 4] + node_idx[k + j * 4] * 8] +
                                            lambda2 * d2g_dx2_node2[node_idx[l + i * 4] + node_idx[k + j * 4] * 8];
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
  auto df_dx_temp = jac_data_.GetMfemJacobian( mesh_data_.GetParentCoords().ParFESpace(),
                                               mesh_data_.GetParentCoords().ParFESpace(), df_contribs );
  df_du = std::unique_ptr<mfem::HypreParMatrix>( df_dx_temp.release() );
}

void EnergyMortarAdapter::evaluateContactResidual( const mfem::HypreParVector& lambda, mfem::HypreParVector& r_force,
                                                   mfem::HypreParVector& r_gap )
{
  SLIC_ERROR_ROOT_IF( use_penalty_, "evaluateContactResidual() should only be  called in lagrange multiplier mode" );

  SLIC_ERROR_ROOT_IF( g_tilde_vec_.size() == 0, "updateNodalGaps() must be called before evaluateContactResidual()" );

  // Force residual = r_f = lambda * dg_tilde/du
  dg_tilde_dx_->MultTranspose( lambda, r_force );

  // gap residual
  r_gap = g_tilde_vec_.get();
}

void EnergyMortarAdapter::evaluateContactJacobian( const mfem::HypreParVector& lambda,
                                                   std::unique_ptr<mfem::HypreParMatrix>& df_du,
                                                   std::unique_ptr<mfem::HypreParMatrix>& df_dlambda )
{
  SLIC_ERROR_ROOT_IF( use_penalty_, "evaluateContactJacobian() should only be called in Lagrange multiplier mode" );

  SLIC_ERROR_ROOT_IF( g_tilde_vec_.size() == 0, "updateNodalGaps() must be called before evaluateContactJacobian()" );

  // df/dlambda = dg_tilde/du:
  df_dlambda = std::unique_ptr<mfem::HypreParMatrix>( dg_tilde_dx_.release() );

  // df/du = lambda * d2g_tilde/du2
  compute_df_du_lagrange( lambda, df_du );
}

std::unique_ptr<mfem::HypreParMatrix> EnergyMortarAdapter::getMfemDfDx() const
{
  return std::unique_ptr<mfem::HypreParMatrix>( df_dx_.release() );
}

std::unique_ptr<mfem::HypreParMatrix> EnergyMortarAdapter::getMfemDgDx() const
{
  return std::unique_ptr<mfem::HypreParMatrix>( dg_tilde_dx_.release() );
}

std::unique_ptr<mfem::HypreParMatrix> EnergyMortarAdapter::getMfemDfDp() const
{
  // SLIC_ERROR_ROOT( "EnergyMortar does not support getMfemDfDp()" );
  return nullptr;
}

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
