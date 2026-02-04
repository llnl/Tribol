// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/NewMethodAdapter.hpp"

namespace tribol {

NewMethodAdapter::NewMethodAdapter( MfemMeshData& mfem_data, MfemSubmeshData& submesh_data, MfemJacobianData& jac_data,
                                    MeshData& mesh1, MeshData& mesh2, double k, double delta, int N )
    : mfem_data_( mfem_data ), submesh_data_( submesh_data ), jac_data_( jac_data ), mesh1_( mesh1 ), mesh2_( mesh2 )
{
  params_.k = k;
  params_.del = delta;
  params_.N = N;
  evaluator_ = std::make_unique<ContactEvaluator>( params_ );
}

void NewMethodAdapter::setInterfacePairs( ArrayT<InterfacePair>&& pairs, int /*check_level*/ )
{
  // TODO: improved pair identification
  pairs_ = std::move( pairs );
}

void NewMethodAdapter::updateIntegrationRule()
{
  // TODO: break out integration rule as a separate method
}

void NewMethodAdapter::updateNodalGaps()
{
  // NOTE: user should have called updateMfemParallelDecomposition() with updated coords before calling this

  // Tribol level data structures for storing gap, area, and derivatives
  auto redecomp_gap = submesh_data_.GetRedecompGap();
  mfem::GridFunction redecomp_area( redecomp_gap.FESpace() );
  redecomp_area = 0.0;
  MethodData dg_tilde_dx;
  dg_tilde_dx.reserveBlockJ( { BlockSpace::NONMORTAR, BlockSpace::MORTAR, BlockSpace::LAGRANGE_MULTIPLIER },
                             pairs_.size() );
  MethodData dA_dx;
  dA_dx.reserveBlockJ( { BlockSpace::NONMORTAR, BlockSpace::MORTAR, BlockSpace::LAGRANGE_MULTIPLIER }, pairs_.size() );

  auto mesh1_view = mesh1_.getView();
  auto mesh2_view = mesh2_.getView();

  // Compute local contributions
  for ( const auto& pair : pairs_ ) {
    const auto elem1 = static_cast<int>( pair.m_element_id1 );
    const auto elem2 = static_cast<int>( pair.m_element_id2 );

    double g_tilde_elem[2];
    double A_elem[2];

    evaluator_->gtilde_and_area( pair, mesh1_view, mesh2_view, g_tilde_elem, A_elem );

    if ( A_elem[0] <= 0.0 && A_elem[1] <= 0.0 ) {
      continue;
    }

    auto A_conn = mesh1_view.getConnectivity()( elem1 );

    // Add to nodes of Element A
    redecomp_gap[A_conn[0]] += g_tilde_elem[0];
    redecomp_gap[A_conn[1]] += g_tilde_elem[1];

    redecomp_area[A_conn[0]] += A_elem[0];
    redecomp_area[A_conn[1]] += A_elem[1];

    // compute g_tilde first derivative
    double dg_dx_node1[8];
    double dg_dx_node2[8];
    evaluator_->grad_gtilde( pair, mesh1_view, mesh2_view, dg_dx_node1, dg_dx_node2 );
    StackArray<DeviceArray2D<RealT>, 9> dg_tilde_dx_block( 3 );
    dg_tilde_dx_block( 2, 0 ) = DeviceArray2D<RealT>( 2, 4 );
    dg_tilde_dx_block( 2, 0 ).fill( 0.0 );
    dg_tilde_dx_block( 2, 1 ) = DeviceArray2D<RealT>( 2, 4 );
    dg_tilde_dx_block( 2, 1 ).fill( 0.0 );
    for ( int i{ 0 }; i < 4; ++i ) {
      dg_tilde_dx_block( 2, 0 )( 0, i ) = dg_dx_node1[i];
    }
    for ( int i{ 0 }; i < 4; ++i ) {
      dg_tilde_dx_block( 2, 0 )( 1, i ) = dg_dx_node2[i];
    }
    for ( int i{ 0 }; i < 4; ++i ) {
      dg_tilde_dx_block( 2, 1 )( 0, i ) = dg_dx_node1[i + 4];
    }
    for ( int i{ 0 }; i < 4; ++i ) {
      dg_tilde_dx_block( 2, 1 )( 1, i ) = dg_dx_node2[i + 4];
    }
    dg_tilde_dx.storeElemBlockJ( { elem1, elem2, elem1 }, dg_tilde_dx_block );

    // compute area first derivative
    double dA_dx_node1[8];
    double dA_dx_node2[8];
    evaluator_->grad_trib_area( pair, mesh1_view, mesh2_view, dA_dx_node1, dA_dx_node2 );
    StackArray<DeviceArray2D<RealT>, 9> dA_dx_block( 3 );
    dA_dx_block( 2, 0 ) = DeviceArray2D<RealT>( 2, 4 );
    dA_dx_block( 2, 0 ).fill( 0.0 );
    dA_dx_block( 2, 1 ) = DeviceArray2D<RealT>( 2, 4 );
    dA_dx_block( 2, 1 ).fill( 0.0 );
    for ( int i{ 0 }; i < 4; ++i ) {
      dA_dx_block( 2, 0 )( 0, i ) = dA_dx_node1[i];
    }
    for ( int i{ 0 }; i < 4; ++i ) {
      dA_dx_block( 2, 0 )( 1, i ) = dA_dx_node2[i];
    }
    for ( int i{ 0 }; i < 4; ++i ) {
      dA_dx_block( 2, 1 )( 0, i ) = dA_dx_node1[i + 4];
    }
    for ( int i{ 0 }; i < 4; ++i ) {
      dA_dx_block( 2, 1 )( 1, i ) = dA_dx_node2[i + 4];
    }
    dA_dx.storeElemBlockJ( { elem1, elem2, elem1 }, dA_dx_block );
  }

  // Move gap and area to submesh level vectors
  mfem::ParLinearForm g_tilde_linear_form(
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_data_.GetSubmeshGap( g_tilde_linear_form );
  auto& P_submesh = *submesh_data_.GetSubmeshFESpace().GetProlongationMatrix();
  g_tilde_vec_ = mfem::HypreParVector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  g_tilde_vec_ = 0.0;
  P_submesh.MultTranspose( g_tilde_linear_form, g_tilde_vec_ );

  mfem::ParLinearForm A_linear_form( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_data_.GetPressureTransfer().RedecompToSubmesh( redecomp_area, A_linear_form );
  A_vec_ = mfem::HypreParVector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  A_vec_ = 0.0;
  P_submesh.MultTranspose( A_linear_form, A_vec_ );

  // Move gap and area derivatives to HypreParMatrix (submesh rows, parent mesh cols)
  const std::vector<std::pair<int, BlockSpace>> all_info{
      { 0, BlockSpace::NONMORTAR }, { 0, BlockSpace::MORTAR }, { 1, BlockSpace::LAGRANGE_MULTIPLIER } };
  auto dg_tilde_dx_block = jac_data_.GetMfemBlockJacobian( dg_tilde_dx, all_info, all_info );
  dg_tilde_dx_block->owns_blocks = false;
  dg_tilde_dx_ = ParSparseMat( static_cast<mfem::HypreParMatrix*>( &dg_tilde_dx_block->GetBlock( 1, 0 ) ) );
  delete &dg_tilde_dx_block->GetBlock( 0, 0 );
  delete &dg_tilde_dx_block->GetBlock( 0, 1 );
  delete &dg_tilde_dx_block->GetBlock( 1, 1 );

  auto dA_dx_block = jac_data_.GetMfemBlockJacobian( dA_dx, all_info, all_info );
  dA_dx_block->owns_blocks = false;
  dA_dx_ = ParSparseMat( static_cast<mfem::HypreParMatrix*>( &dA_dx_block->GetBlock( 1, 0 ) ) );
  delete &dA_dx_block->GetBlock( 0, 0 );
  delete &dA_dx_block->GetBlock( 0, 1 );
  delete &dA_dx_block->GetBlock( 1, 1 );
}

void NewMethodAdapter::updateNodalForces()
{
  // NOTE: user should have called updateNodalGaps() with updated coords before calling this

  // compute nodal pressures. these are used in the Hessian vector product below so we don't have to assemble a Hessian
  // NOTE: in general, pressure should likely be set by the host code
  pressure_vec_.SetSize( g_tilde_vec_.Size() );
  pressure_vec_ = 0.0;
  for ( int i{ 0 }; i < pressure_vec_.Size(); ++i ) {
    if ( A_vec_[i] > 1.0e-14 && g_tilde_vec_[i] <= 0.0 ) {
      pressure_vec_[i] = params_.k * g_tilde_vec_[i] / A_vec_[i];
    }
  }

  mfem::HypreParVector k_over_a( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  k_over_a = 0.0;
  for ( int i{ 0 }; i < k_over_a.Size(); ++i ) {
    if ( A_vec_[i] > 1.0e-14 ) {
      k_over_a[i] = params_.k / A_vec_[i];
    }
  }

  mfem::HypreParVector p_over_a( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  p_over_a = 0.0;
  for ( int i{ 0 }; i < p_over_a.Size(); ++i ) {
    if ( A_vec_[i] > 1.0e-14 ) {
      p_over_a[i] = pressure_vec_[i] / A_vec_[i];
    }
  }

  ParSparseMat dp_dx( *dg_tilde_dx_.get() );
  dp_dx->ScaleRows( k_over_a );
  ParSparseMat dp_dx_temp( *dA_dx_.get() );
  dp_dx_temp->ScaleRows( p_over_a );
  dp_dx -= dp_dx_temp;

  force_vec_ = pressure_vec_ * dg_tilde_dx_ - g_tilde_vec_ * dp_dx;

  MethodData df_dx_data;
  df_dx_data.reserveBlockJ( { BlockSpace::NONMORTAR, BlockSpace::MORTAR }, pairs_.size() );

  auto mesh1_view = mesh1_.getView();
  auto mesh2_view = mesh2_.getView();

  // get pairwise action of second derivatives of gaps and pressure for stiffness contribution
  for ( auto& pair : pairs_ ) {
    const auto elem1 = static_cast<int>( pair.m_element_id1 );
    const auto node11 = mesh1_view.getConnectivity()( elem1, 0 );
    const auto node12 = mesh1_view.getConnectivity()( elem1, 1 );
    const auto elem2 = static_cast<int>( pair.m_element_id2 );

    const RealT pressure1 = 2.0 * pressure_vec_[node11];
    const RealT pressure2 = 2.0 * pressure_vec_[node12];

    if ( pressure1 == 0.0 && pressure2 == 0.0 ) {
      continue;
    }

    const RealT g_p_ainv1 = -g_tilde_vec_[node11] * pressure_vec_[node11] / A_vec_[node11];
    const RealT g_p_ainv2 = -g_tilde_vec_[node12] * pressure_vec_[node12] / A_vec_[node12];

    double df_dx_node1[64];
    double df_dx_node2[64];
    // ordering: [dg/(dx0dx0) dg/(dy0dx0) dg/(dx1dx0) ...]
    evaluator_->d2_g2tilde( pair, mesh1_view, mesh2_view, df_dx_node1, df_dx_node2 );
    StackArray<DeviceArray2D<RealT>, 9> df_dx_block( 2 );
    df_dx_block( 0, 0 ) = DeviceArray2D<RealT>( 4, 4 );
    df_dx_block( 0, 0 ).fill( 0.0 );
    df_dx_block( 0, 1 ) = DeviceArray2D<RealT>( 4, 4 );
    df_dx_block( 0, 1 ).fill( 0.0 );
    df_dx_block( 1, 0 ) = DeviceArray2D<RealT>( 4, 4 );
    df_dx_block( 1, 0 ).fill( 0.0 );
    df_dx_block( 1, 1 ) = DeviceArray2D<RealT>( 4, 4 );
    df_dx_block( 1, 1 ).fill( 0.0 );
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 0, 0 )( i, j ) = pressure1 * df_dx_node1[i + j * 8] + pressure2 * df_dx_node2[i + j * 8];
      }
    }
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 0, 1 )( i, j ) =
            pressure1 * df_dx_node1[i + ( j + 4 ) * 8] + pressure2 * df_dx_node2[i + ( j + 4 ) * 8];
      }
    }
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 1, 0 )( i, j ) = pressure1 * df_dx_node1[i + 4 + j * 8] + pressure2 * df_dx_node2[i + 4 + j * 8];
      }
    }
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 1, 1 )( i, j ) =
            pressure1 * df_dx_node1[i + 4 + ( j + 4 ) * 8] + pressure2 * df_dx_node2[i + 4 + ( j + 4 ) * 8];
      }
    }
    evaluator_->compute_d2A_d2u( pair, mesh1_view, mesh2_view, df_dx_node1, df_dx_node2 );
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 0, 0 )( i, j ) += g_p_ainv1 * df_dx_node1[i + j * 8] + g_p_ainv2 * df_dx_node2[i + j * 8];
      }
    }
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 0, 1 )( i, j ) +=
            g_p_ainv1 * df_dx_node1[i + ( j + 4 ) * 8] + g_p_ainv2 * df_dx_node2[i + ( j + 4 ) * 8];
      }
    }
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 1, 0 )( i, j ) += g_p_ainv1 * df_dx_node1[i + 4 + j * 8] + g_p_ainv2 * df_dx_node2[i + 4 + j * 8];
      }
    }
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 1, 1 )( i, j ) +=
            g_p_ainv1 * df_dx_node1[i + 4 + ( j + 4 ) * 8] + g_p_ainv2 * df_dx_node2[i + 4 + ( j + 4 ) * 8];
      }
    }
    df_dx_data.storeElemBlockJ( { elem1, elem2 }, df_dx_block );
  }

  // Move gap and area derivatives to HypreParMatrix (submesh rows, parent mesh cols)
  const std::vector<std::pair<int, BlockSpace>> all_info{ { 0, BlockSpace::NONMORTAR }, { 0, BlockSpace::MORTAR } };
  auto df_dx_block = jac_data_.GetMfemBlockJacobian( df_dx_data, all_info, all_info );
  df_dx_block->owns_blocks = false;
  df_dx_ = ParSparseMat( static_cast<mfem::HypreParMatrix*>( &df_dx_block->GetBlock( 0, 0 ) ) );

  mfem::HypreParVector pg2_over_asq( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  pg2_over_asq = 0.0;
  for ( int i{ 0 }; i < pg2_over_asq.Size(); ++i ) {
    if ( A_vec_[i] > 1.0e-14 ) {
      pg2_over_asq[i] = 2.0 * pressure_vec_[i] * g_tilde_vec_[i] / ( A_vec_[i] * A_vec_[i] );
    }
  }

  auto& parent_fes = *mfem_data_.GetParentCoords().ParFESpace();
  auto p_over_a_diag = ParSparseMat::diagonalMatrix( parent_fes.GetComm(), parent_fes.GlobalTrueVSize(),
                                                     parent_fes.GetTrueDofOffsets(), p_over_a );
  auto pg2_over_asq_diag = ParSparseMat::diagonalMatrix( parent_fes.GetComm(), parent_fes.GlobalTrueVSize(),
                                                         parent_fes.GetTrueDofOffsets(), pg2_over_asq );

  df_dx_ -= ParSparseMat::RAP( dg_tilde_dx_, p_over_a_diag, dA_dx_ );
  df_dx_ -= ParSparseMat::RAP( dA_dx_, p_over_a_diag, dg_tilde_dx_ );
  df_dx_ += ParSparseMat::RAP( dA_dx_, pg2_over_asq_diag, dg_tilde_dx_ );
  df_dx_ += dp_dx.transpose() * dg_tilde_dx_;
  df_dx_ += dg_tilde_dx_.transpose() * dp_dx;
}

RealT NewMethodAdapter::computeTimeStep()
{
  // TODO: implement timestep calculation
  return 1.0;
}

void NewMethodAdapter::getMfemForce( mfem::Vector& forces ) const { forces = force_vec_; }

void NewMethodAdapter::getMfemGap( mfem::Vector& gaps ) const
{
  gaps.SetSize( g_tilde_vec_.Size() );

  for ( int i = 0; i < gaps.Size(); ++i ) {
    if ( A_vec_[i] > 1.0e-14 )
      gaps[i] = g_tilde_vec_[i] / A_vec_[i];
    else
      gaps[i] = 0.0;
  }
}

mfem::ParGridFunction& NewMethodAdapter::getMfemPressure()
{
  auto& pressure = submesh_data_.GetSubmeshPressure();
  pressure.SetFromTrueDofs( pressure_vec_ );
  return pressure;
}

std::unique_ptr<mfem::HypreParMatrix> NewMethodAdapter::getMfemDfDx() const
{
  return std::unique_ptr<mfem::HypreParMatrix>( df_dx_.release() );
}

std::unique_ptr<mfem::HypreParMatrix> NewMethodAdapter::getMfemDgDx() const
{
  return std::unique_ptr<mfem::HypreParMatrix>( dg_tilde_dx_.release() );
}

std::unique_ptr<mfem::HypreParMatrix> NewMethodAdapter::getMfemDfDp() const { return nullptr; }

}  // namespace tribol
