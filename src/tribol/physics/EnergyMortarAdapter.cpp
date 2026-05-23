// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortarAdapter.hpp"
#include <cmath>
#include <axom/slic/interface/slic_macros.hpp>
#include "tribol/mesh/MfemData.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

EnergyMortarAdapter::EnergyMortarAdapter( MfemMeshData& mesh_data, MfemSubmeshData& submesh_data,
                                          MfemJacobianData& jac_data, MeshData& mesh1, MeshData& mesh2, double k,
                                          double delta, int N, bool enzyme_quadrature, bool use_penalty,
                                          SmoothingType smoothing_type,
                                          PenaltySmoothing penalty_smoothing,
                                          double penalty_smoothing_del )
    // NOTE: mesh1 maps to mesh2_ and mesh2 maps to mesh1_. This is to keep consistent with mesh1_ being non-mortar and
    // mesh2_ being mortar as is typical in the literature, but different from Tribol convention.
    : use_penalty_( use_penalty ),
      mesh_data_( mesh_data ),
      submesh_data_( submesh_data ),
      jac_data_( jac_data ),
      mesh1_( &mesh2 ),
      mesh2_( &mesh1 )
{
  if ( mesh1.numberOfNodes() > 0 && mesh2.numberOfNodes() > 0 ) {
    SLIC_ERROR_ROOT_IF( mesh1.spatialDimension() != 2 || mesh2.spatialDimension() != 2,
                        "ENERGY_MORTAR requires 2D meshes." );
  }

  params_.k = k;
  params_.del = delta;
  params_.N = N;
  params_.enzyme_quadrature = enzyme_quadrature;
  params_.smoothing_type = smoothing_type;
  params_.penalty_smoothing = penalty_smoothing;
  params_.penalty_smoothing_del = penalty_smoothing_del;

  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );

  // Allocate the (pressure) true-dof vector early so host code can set it via tribol::getMfemTDofPressure() after the
  // formulation is created. In penalty mode this is overwritten in updateNodalForces(); in LM mode it is treated as the
  // Lagrange multiplier vector (lambda).
  pressure_vec_ = shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  pressure_vec_.fill( 0.0 );
}

void EnergyMortarAdapter::updateMeshes( MeshData& mesh1, MeshData& mesh2 )
{
  // Maintain the same "flipped" convention as the constructor.
  mesh1_ = &mesh2;
  mesh2_ = &mesh1;
}

const mfem::HypreParVector& EnergyMortarAdapter::getMfemGap() const
{
  // Penalty mode uses the normalized gap g = g_tilde / A. LM mode enforces the unnormalized constraint g_tilde = 0,
  // consistent with dg/dx returned by getMfemDgDx().
  return use_penalty_ ? gap_vec_.get() : g_tilde_vec_.get();
}

void EnergyMortarAdapter::setInterfacePairs( ArrayT<InterfacePair>&& pairs, int /*check_level*/ )
{
  // TODO: Consider design and how this interacts with binning and CG
  pairs_ = std::move( pairs );
}

void EnergyMortarAdapter::updateIntegrationRule()
{
  SLIC_WARNING_ROOT( "Update integration rule not implemmented for any method" );
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

  SLIC_ERROR_ROOT_IF( mesh1_ == nullptr || mesh2_ == nullptr, "ENERGY_MORTAR meshes not set." );
  auto mesh1_view = mesh1_->getView();
  auto mesh2_view = mesh2_->getView();

  // Compute local contributions
  for ( const auto& pair : pairs_ ) {
    // These need to be flipped, since the pairs are determined with element 1 associated with mesh 1, and we flipped
    // the mesh numbers to be consistent with the literature and since the underlying method integrates on element 1
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    double g_tilde_elem[2];
    double A_elem[2];

    evaluator_->compute_gtilde_and_area( flipped_pair, mesh1_view, mesh2_view, g_tilde_elem, A_elem );

    if constexpr ( tribol::energy_mortar_debug_level >= 1 ) {
      assert( A_elem[0] >= -1e-14 && "per-pair tributary area[0] is negative" );
      assert( A_elem[1] >= -1e-14 && "per-pair tributary area[1] is negative" );
    }

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

  mfem::ParLinearForm A_linear_form( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_data_.GetPressureTransfer().RedecompToSubmesh( redecomp_area, A_linear_form );
  A_vec_ = shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  A_vec_.fill( 0.0 );
  P_submesh.MultTranspose( A_linear_form, A_vec_.get() );

  // Identify rows to eliminate: nodes with non-positive area have no contact
  // contribution.  For non-tied penalty contact with Hard penalty smoothing,
  // also eliminate nodes with positive g_tilde (opening gap).
  mfem::Array<int> rows_to_elim;
  rows_to_elim.Reserve( g_tilde_vec_.size() );
  for ( int i{ 0 }; i < g_tilde_vec_.size(); ++i ) {
    if ( A_vec_[i] <= 0.0 ) {
      g_tilde_vec_[i] = 0.0;
      A_vec_[i] = 0.0;
      rows_to_elim.push_back( i );
    } else if ( !tied_contact_ && use_penalty_ && params_.penalty_smoothing == PenaltySmoothing::Hard &&
                g_tilde_vec_[i] > 0.0 ) {
      g_tilde_vec_[i] = 0.0;
      rows_to_elim.push_back( i );
    }
  }

  // Regularized area: A_reg = A + eps prevents 1/A blow-up near separation
  // while leaving the solution unchanged when A >> eps.
  A_reg_vec_ = shared::ParVector( A_vec_ );
  for ( int i{ 0 }; i < A_reg_vec_.size(); ++i ) {
    A_reg_vec_[i] += area_reg_;
  }

  gap_vec_ = g_tilde_vec_.divide( A_reg_vec_, area_tol_ );

  // Compute the penalty ramp H(g), H'(g), H''(g) per node.
  gap_orig_vec_ = shared::ParVector( gap_vec_ );
  h_prime_vec_ = shared::ParVector( gap_vec_ );
  h_double_prime_vec_ = shared::ParVector( gap_vec_ );

  const bool apply_penalty_ramp = !tied_contact_ && use_penalty_;
  const double del = params_.penalty_smoothing_del;

  for ( int i{ 0 }; i < gap_vec_.size(); ++i ) {
    if ( apply_penalty_ramp && params_.penalty_smoothing == PenaltySmoothing::Smooth ) {
      const auto ramp = ContactSmoothing::penalty_ramp( gap_orig_vec_[i], del, params_.penalty_smoothing );
      gap_vec_[i] = ramp.value;
      h_prime_vec_[i] = ramp.first_derivative;
      h_double_prime_vec_[i] = ramp.second_derivative;
    } else {
      h_prime_vec_[i] = 1.0;
      h_double_prime_vec_[i] = 0.0;
    }
  }

  // Move gap and area derivatives to (pressure true-dof rows, displacement true-dof cols)
  std::vector<PackedPairJacobianContribs> dg_contribs;
  dg_contribs.reserve( 2 );
  dg_contribs.push_back( std::move( dg_lm_nm ) );
  dg_contribs.push_back( std::move( dg_lm_m ) );
  dg_tilde_dx_ = jac_data_.GetMfemJacobian( &submesh_data_.GetSubmeshFESpace(),
                                            mesh_data_.GetParentCoords().ParFESpace(), dg_contribs );
  if ( rows_to_elim.Size() > 0 ) {
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

  if ( use_penalty_ ) {
    // Penalty mode: p = k * H(g) where g = g_tilde / A_reg, H is the penalty ramp
    pressure_vec_ = params_.k * gap_vec_;
  } else {
    // LM mode: pressure_vec_ is treated as the Lagrange multiplier vector (lambda)
    SLIC_ERROR_ROOT_IF( pressure_vec_.size() == 0,
                        "LM vector is not initialized. Call tribol::update() once to initialize the formulation." );
    SLIC_ERROR_ROOT_IF( pressure_vec_.size() != g_tilde_vec_.size(),
                        "LM vector size mismatch with contact dofs (g_tilde)." );
  }

  energy_ = pressure_vec_.dot( g_tilde_vec_ );

  if ( !use_penalty_ ) {
    // -------------------------------------------------------------------------
    // LM mode: force = G^T * lambda and df/dx = lambda · d^2(g_tilde)/dx^2
    // -------------------------------------------------------------------------
    force_vec_ = pressure_vec_ * dg_tilde_dx_;

    mfem::GridFunction redecomp_lambda( submesh_data_.GetRedecompGap() );
    mfem::ParGridFunction submesh_lambda(
        const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
    submesh_lambda.SetFromTrueDofs( pressure_vec_.get() );
    submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_lambda, redecomp_lambda );

    df_dx_ = computeDfDxSecondDerivativesLM( redecomp_lambda );
    return;
  }

  // ---------------------------------------------------------------------------
  // Penalty mode with generic penalty ramp H(g).
  //
  // Energy:  E = k * H(g) * g_tilde   (per node, g = g_tilde / A_reg)
  //
  // Smoothed gap:   h  = H(g)      = gap_vec_
  // Original gap:   g  = g_orig    = gap_orig_vec_
  // Ramp deriv:     h' = H'(g)     = h_prime_vec_
  // Ramp 2nd deriv: h''= H''(g)    = h_double_prime_vec_
  //
  // Pressure: p = k * h
  //
  // dp/dx = k * h' / A_reg * (Gt - g * A')   [chain rule through H(g)]
  //
  // Force: f = p * Gt + g_tilde * dp/dx
  //          = k*(h + h'*g) * Gt - k*h'*g² * A'
  //
  // Jacobian (RAP terms from d(dp/dx diagonal)/dx):
  //   rap(Gt, diag(k*g*h''/A), Gt)                          [new: zero for Hard]
  //   - rap(Gt, diag(k*g*(h'+g*h'')/A), A')
  //   - rap(A', diag(k*g*(h'+g*h'')/A), Gt)
  //   + rap(A', diag(k*g²*(2h'+g*h'')/A), A')
  //   + dp_dx^T * Gt + Gt^T * dp_dx
  //
  // Second derivatives:
  //   k*(h + h'*g) * d²g_tilde/dx²  -  k*h'*g² * d²A/dx²
  // ---------------------------------------------------------------------------
  auto A_reg_inv = A_reg_vec_.inverse( area_tol_ );

  // dp/dx = k * h' / A_reg * (Gt - g_orig * A')
  shared::ParSparseMat dp_dx( dg_tilde_dx_.get() );
  shared::ParSparseMat g_dA( dA_dx_.get() );
  g_dA->ScaleRows( gap_orig_vec_.get() );
  dp_dx -= g_dA;
  auto kh_prime_over_a_reg = params_.k * h_prime_vec_.multiply( A_reg_inv );
  dp_dx->ScaleRows( kh_prime_over_a_reg.get() );

  force_vec_ = ( pressure_vec_ * dg_tilde_dx_ ) + ( g_tilde_vec_ * dp_dx );

  // Transfer per-node second-derivative coefficients to redecomp
  //   coeff_gt = k * (h + h'*g)  =  k * alpha
  //   coeff_A  = -k * h' * g²
  auto alpha = gap_vec_ + h_prime_vec_.multiply( gap_orig_vec_ );
  auto coeff_gt_vec = params_.k * alpha;
  auto coeff_A_vec = -params_.k * h_prime_vec_.multiply( gap_orig_vec_ ).multiplyInPlace( gap_orig_vec_ );

  mfem::GridFunction redecomp_coeff_gt( submesh_data_.GetRedecompGap() );
  mfem::ParGridFunction submesh_coeff_gt(
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_coeff_gt.SetFromTrueDofs( coeff_gt_vec.get() );
  submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_coeff_gt, redecomp_coeff_gt );

  mfem::GridFunction redecomp_coeff_A( submesh_data_.GetRedecompGap() );
  mfem::ParGridFunction submesh_coeff_A(
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_coeff_A.SetFromTrueDofs( coeff_A_vec.get() );
  submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_coeff_A, redecomp_coeff_A );

  df_dx_ = computeDfDxSecondDerivativesPenalty( redecomp_coeff_gt, redecomp_coeff_A );

  // RAP Jacobian cross terms
  auto& submesh_fes = submesh_data_.GetSubmeshFESpace();

  // d1 = k * g * h'' / A_reg  (for Gt-Gt term, zero in Hard/full-contact)
  auto d1_vec = params_.k * gap_orig_vec_.multiply( h_double_prime_vec_ ).divideInPlace( A_reg_vec_, area_tol_ );
  // d2 = k * g * (h' + g*h'') / A_reg  (for Gt-A' cross terms)
  auto h_prime_plus_g_h_double_prime = h_prime_vec_ + gap_orig_vec_.multiply( h_double_prime_vec_ );
  auto d2_vec = params_.k * gap_orig_vec_.multiply( h_prime_plus_g_h_double_prime ).divideInPlace( A_reg_vec_, area_tol_ );
  // d4 = k * g² * (2h' + g*h'') / A_reg  (for A'-A' term)
  auto two_h_prime_plus_g_h_double_prime = 2.0 * h_prime_vec_ + gap_orig_vec_.multiply( h_double_prime_vec_ );
  auto g_sq = gap_orig_vec_.multiply( gap_orig_vec_ );
  auto d4_vec = params_.k * g_sq.multiplyInPlace( two_h_prime_plus_g_h_double_prime )
                    .divideInPlace( A_reg_vec_, area_tol_ );

  auto d1_diag = shared::ParSparseMat::diagonalMatrix( submesh_fes.GetComm(), submesh_fes.GlobalTrueVSize(),
                                                        submesh_fes.GetTrueDofOffsets(), d1_vec.get() );
  auto d2_diag = shared::ParSparseMat::diagonalMatrix( submesh_fes.GetComm(), submesh_fes.GlobalTrueVSize(),
                                                        submesh_fes.GetTrueDofOffsets(), d2_vec.get() );
  auto d4_diag = shared::ParSparseMat::diagonalMatrix( submesh_fes.GetComm(), submesh_fes.GlobalTrueVSize(),
                                                        submesh_fes.GetTrueDofOffsets(), d4_vec.get() );

  auto rap_gg = shared::ParSparseMat::rap( dg_tilde_dx_, d1_diag, dg_tilde_dx_ );
  auto rap_ga = shared::ParSparseMat::rap( dg_tilde_dx_, d2_diag, dA_dx_ );
  auto rap_ag = shared::ParSparseMat::rap( dA_dx_, d2_diag, dg_tilde_dx_ );
  auto rap_aa = shared::ParSparseMat::rap( dA_dx_, d4_diag, dA_dx_ );
  auto dp_g = dp_dx.transpose() * dg_tilde_dx_;
  auto g_dp = dg_tilde_dx_.transpose() * dp_dx;

  df_dx_ += rap_gg;
  df_dx_ -= rap_ga;
  df_dx_ -= rap_ag;
  df_dx_ += rap_aa;
  df_dx_ += dp_g;
  df_dx_ += g_dp;
}

RealT EnergyMortarAdapter::computeTimeStep()
{
  // TODO: implement timestep calculation
  return 1.0;
}

shared::ParSparseMat EnergyMortarAdapter::computeDfDxSecondDerivativesLM( const mfem::GridFunction& redecomp_lambda )
{
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

  SLIC_ERROR_ROOT_IF( mesh1_ == nullptr || mesh2_ == nullptr, "ENERGY_MORTAR meshes not set." );
  auto mesh1_view = mesh1_->getView();
  auto mesh2_view = mesh2_->getView();

  for ( auto& pair : pairs_ ) {
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto node11 = mesh1_view.getConnectivity()( elem1, 0 );
    const auto node12 = mesh1_view.getConnectivity()( elem1, 1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    const RealT lambda1 = redecomp_lambda( node11 );
    const RealT lambda2 = redecomp_lambda( node12 );

    double d2g_dx2_node1[64];
    double d2g_dx2_node2[64];
    evaluator_->d2_g2tilde( flipped_pair, mesh1_view, mesh2_view, d2g_dx2_node1, d2g_dx2_node2 );

    double df_dx_blocks[2][2][16];
    for ( int i{ 0 }; i < 2; ++i ) {
      for ( int j{ 0 }; j < 2; ++j ) {
        for ( int k{ 0 }; k < 4; ++k ) {
          for ( int l{ 0 }; l < 4; ++l ) {
            const auto idx = node_idx[l + i * 4] * 8 + node_idx[k + j * 4];
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
  return jac_data_.GetMfemJacobian( mesh_data_.GetParentCoords().ParFESpace(),
                                    mesh_data_.GetParentCoords().ParFESpace(), df_contribs );
}

shared::ParSparseMat EnergyMortarAdapter::computeDfDxSecondDerivativesPenalty(
    const mfem::GridFunction& redecomp_coeff_gt, const mfem::GridFunction& redecomp_coeff_A )
{
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

  SLIC_ERROR_ROOT_IF( mesh1_ == nullptr || mesh2_ == nullptr, "ENERGY_MORTAR meshes not set." );
  auto mesh1_view = mesh1_->getView();
  auto mesh2_view = mesh2_->getView();

  for ( auto& pair : pairs_ ) {
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto node11 = mesh1_view.getConnectivity()( elem1, 0 );
    const auto node12 = mesh1_view.getConnectivity()( elem1, 1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    const RealT cgt1 = redecomp_coeff_gt( node11 );
    const RealT cgt2 = redecomp_coeff_gt( node12 );

    const RealT cA1 = redecomp_coeff_A( node11 );
    const RealT cA2 = redecomp_coeff_A( node12 );

    if ( cgt1 == 0.0 && cgt2 == 0.0 && cA1 == 0.0 && cA2 == 0.0 ) {
      continue;
    }

    double d2g_dx2_node1[64];
    double d2g_dx2_node2[64];
    evaluator_->d2_g2tilde( flipped_pair, mesh1_view, mesh2_view, d2g_dx2_node1, d2g_dx2_node2 );

    double d2A_dx2_node1[64];
    double d2A_dx2_node2[64];
    evaluator_->compute_d2A_d2u( flipped_pair, mesh1_view, mesh2_view, d2A_dx2_node1, d2A_dx2_node2 );

    double df_dx_blocks[2][2][16];
    for ( int i{ 0 }; i < 2; ++i ) {
      for ( int j{ 0 }; j < 2; ++j ) {
        for ( int k{ 0 }; k < 4; ++k ) {
          for ( int l{ 0 }; l < 4; ++l ) {
            const auto idx = node_idx[l + i * 4] * 8 + node_idx[k + j * 4];
            const double value = cgt1 * d2g_dx2_node1[idx] + cgt2 * d2g_dx2_node2[idx] +
                                 cA1 * d2A_dx2_node1[idx] + cA2 * d2A_dx2_node2[idx];
            df_dx_blocks[i][j][l + k * 4] = value;
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
  return jac_data_.GetMfemJacobian( mesh_data_.GetParentCoords().ParFESpace(),
                                    mesh_data_.GetParentCoords().ParFESpace(), df_contribs );
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
  if ( use_penalty_ ) {
    return nullptr;
  }
  // TODO (EBC): figure out better lifetime for this. if you called getMfemDgDx() before this, then dg_tilde_dx_ will be
  // null.
  // LM mode: df/dlambda = (d g_tilde / dx)^T
  auto df_dlambda = dg_tilde_dx_.transpose();
  return std::unique_ptr<mfem::HypreParMatrix>( df_dlambda.release() );
}

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
