// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/NewMethodAdapter.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

static std::vector<ComputedElementData> convertMethodData( const MethodData& method_data,
                                                           const std::vector<BlockSpace>& row_spaces,
                                                           const std::vector<BlockSpace>& col_spaces )
{
  std::vector<ComputedElementData> contributions;
  for ( auto rs : row_spaces ) {
    for ( auto cs : col_spaces ) {
      const auto& J_block = method_data.getBlockJ()( static_cast<int>( rs ), static_cast<int>( cs ) );
      if ( J_block.size() == 0 ) {
        continue;
      }

      ComputedElementData data;
      data.row_space = rs;
      data.col_space = cs;

      const auto& row_ids = method_data.getBlockJElementIds()[static_cast<int>( rs )];
      const auto& col_ids = method_data.getBlockJElementIds()[static_cast<int>( cs )];

      data.row_elem_ids.append( axom::ArrayView<const int>( row_ids.data(), row_ids.size() ) );
      data.col_elem_ids.append( axom::ArrayView<const int>( col_ids.data(), col_ids.size() ) );

      int n_elems = J_block.size();
      int n_rows = J_block[0].Height();
      int n_cols = J_block[0].Width();
      data.jacobian_data.resize( n_elems * n_rows * n_cols );
      data.jacobian_offsets.resize( n_elems );
      for ( int k = 0; k < n_elems; ++k ) {
        data.jacobian_offsets[k] = k * n_rows * n_cols;
        std::copy( J_block[k].GetData(), J_block[k].GetData() + n_rows * n_cols,
                   data.jacobian_data.data() + data.jacobian_offsets[k] );
      }
      contributions.push_back( std::move( data ) );
    }
  }
  return contributions;
}

NewMethodAdapter::NewMethodAdapter( MfemSubmeshData& submesh_data, MfemJacobianData& jac_data, MeshData& mesh1,
                                    MeshData& mesh2, double k, double delta, int N )
    // NOTE: mesh1 maps to mesh2_ and mesh2 maps to mesh1_. This is to keep consistent with mesh1_ being non-mortar and
    // mesh2_ being mortar as is typical in the literature, but different from Tribol convention.
    : submesh_data_( submesh_data ), jac_data_( jac_data ), mesh1_( mesh2 ), mesh2_( mesh1 )
{
  if ( mesh1.numberOfNodes() > 0 && mesh2.numberOfNodes() > 0 ) {
    SLIC_ERROR_ROOT_IF( mesh1.spatialDimension() != 2 || mesh2.spatialDimension() != 2,
                        "ENERGY_MORTAR requires 2D meshes." );
  }

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
  auto& redecomp_gap = submesh_data_.GetRedecompGap();
  mfem::GridFunction redecomp_area( redecomp_gap.FESpace() );
  redecomp_area = 0.0;
  MethodData dg_tilde_dx;
  dg_tilde_dx.reserveBlockJ( { BlockSpace::NONMORTAR, BlockSpace::MORTAR, BlockSpace::LAGRANGE_MULTIPLIER },
                             pairs_.size() );
  MethodData dA_dx;
  dA_dx.reserveBlockJ( { BlockSpace::NONMORTAR, BlockSpace::MORTAR, BlockSpace::LAGRANGE_MULTIPLIER }, pairs_.size() );
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
    evaluator_->grad_gtilde( flipped_pair, mesh1_view, mesh2_view, dg_dx_node1, dg_dx_node2 );
    StackArray<DeviceArray2D<RealT>, 9> dg_tilde_dx_block( 3 );
    dg_tilde_dx_block( 2, 0 ) = DeviceArray2D<RealT>( 2, 4 );
    dg_tilde_dx_block( 2, 0 ).fill( 0.0 );
    dg_tilde_dx_block( 2, 1 ) = DeviceArray2D<RealT>( 2, 4 );
    dg_tilde_dx_block( 2, 1 ).fill( 0.0 );
    for ( int i{ 0 }; i < 4; ++i ) {
      dg_tilde_dx_block( 2, 0 )( 0, i ) = dg_dx_node1[node_idx[i]];
    }
    for ( int i{ 0 }; i < 4; ++i ) {
      dg_tilde_dx_block( 2, 0 )( 1, i ) = dg_dx_node2[node_idx[i]];
    }
    for ( int i{ 0 }; i < 4; ++i ) {
      dg_tilde_dx_block( 2, 1 )( 0, i ) = dg_dx_node1[node_idx[i + 4]];
    }
    for ( int i{ 0 }; i < 4; ++i ) {
      dg_tilde_dx_block( 2, 1 )( 1, i ) = dg_dx_node2[node_idx[i + 4]];
    }
    dg_tilde_dx.storeElemBlockJ( { elem1, elem2, elem1 }, dg_tilde_dx_block );

    // compute area first derivative
    double dA_dx_node1[8];
    double dA_dx_node2[8];
    evaluator_->grad_trib_area( flipped_pair, mesh1_view, mesh2_view, dA_dx_node1, dA_dx_node2 );
    StackArray<DeviceArray2D<RealT>, 9> dA_dx_block( 3 );
    dA_dx_block( 2, 0 ) = DeviceArray2D<RealT>( 2, 4 );
    dA_dx_block( 2, 0 ).fill( 0.0 );
    dA_dx_block( 2, 1 ) = DeviceArray2D<RealT>( 2, 4 );
    dA_dx_block( 2, 1 ).fill( 0.0 );
    for ( int i{ 0 }; i < 4; ++i ) {
      dA_dx_block( 2, 0 )( 0, i ) = dA_dx_node1[node_idx[i]];
    }
    for ( int i{ 0 }; i < 4; ++i ) {
      dA_dx_block( 2, 0 )( 1, i ) = dA_dx_node2[node_idx[i]];
    }
    for ( int i{ 0 }; i < 4; ++i ) {
      dA_dx_block( 2, 1 )( 0, i ) = dA_dx_node1[node_idx[i + 4]];
    }
    for ( int i{ 0 }; i < 4; ++i ) {
      dA_dx_block( 2, 1 )( 1, i ) = dA_dx_node2[node_idx[i + 4]];
    }
    dA_dx.storeElemBlockJ( { elem1, elem2, elem1 }, dA_dx_block );
  }

  // Move gap and area to submesh level vectors
  mfem::ParLinearForm g_tilde_linear_form(
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_data_.GetSubmeshGap( g_tilde_linear_form );
  auto& P_submesh = *submesh_data_.GetSubmeshFESpace().GetProlongationMatrix();
  g_tilde_vec_ = shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  g_tilde_vec_.Fill( 0.0 );
  P_submesh.MultTranspose( g_tilde_linear_form, g_tilde_vec_.get() );

  mfem::Array<int> rows_to_elim;
  if ( !tied_contact_ ) {
    rows_to_elim.Reserve( g_tilde_vec_.Size() );
    for ( int i{ 0 }; i < g_tilde_vec_.Size(); ++i ) {
      if ( g_tilde_vec_[i] > 0.0 ) {
        g_tilde_vec_[i] = 0.0;
        rows_to_elim.push_back( i );
      }
    }
  }

  mfem::ParLinearForm A_linear_form( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_data_.GetPressureTransfer().RedecompToSubmesh( redecomp_area, A_linear_form );
  A_vec_ = shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  A_vec_.Fill( 0.0 );
  P_submesh.MultTranspose( A_linear_form, A_vec_.get() );

  gap_vec_ = g_tilde_vec_.divide( A_vec_, area_tol_ );

  // Move gap and area derivatives to HypreParMatrix (submesh rows, parent mesh cols)
  dg_tilde_dx_ = jac_data_.GetMfemJacobian(
      convertMethodData( dg_tilde_dx, { BlockSpace::LAGRANGE_MULTIPLIER }, { BlockSpace::NONMORTAR, BlockSpace::MORTAR } ) );
  if ( !tied_contact_ ) {
    // technically, we should do this on all the vectors/matrices below, but it looks like the mutliplication operators
    // below will zero them out anyway
    dg_tilde_dx_.EliminateRows( rows_to_elim );
  }

  dA_dx_ = jac_data_.GetMfemJacobian(
      convertMethodData( dA_dx, { BlockSpace::LAGRANGE_MULTIPLIER }, { BlockSpace::NONMORTAR, BlockSpace::MORTAR } ) );
}

void NewMethodAdapter::updateNodalForces()
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

  MethodData df_dx_data;
  df_dx_data.reserveBlockJ( { BlockSpace::NONMORTAR, BlockSpace::MORTAR }, pairs_.size() );
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

    double df_dx_node1[64];
    double df_dx_node2[64];
    // ordering: [dg/(dx0dx0) dg/(dy0dx0) dg/(dx1dx0) ...]
    evaluator_->d2_g2tilde( flipped_pair, mesh1_view, mesh2_view, df_dx_node1, df_dx_node2 );
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
        df_dx_block( 0, 0 )( i, j ) = pressure1 * df_dx_node1[node_idx[i] + node_idx[j] * 8] +
                                      pressure2 * df_dx_node2[node_idx[i] + node_idx[j] * 8];
      }
    }
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 0, 1 )( i, j ) = pressure1 * df_dx_node1[node_idx[i] + node_idx[j + 4] * 8] +
                                      pressure2 * df_dx_node2[node_idx[i] + node_idx[j + 4] * 8];
      }
    }
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 1, 0 )( i, j ) = pressure1 * df_dx_node1[node_idx[i + 4] + node_idx[j] * 8] +
                                      pressure2 * df_dx_node2[node_idx[i + 4] + node_idx[j] * 8];
      }
    }
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 1, 1 )( i, j ) = pressure1 * df_dx_node1[node_idx[i + 4] + node_idx[j + 4] * 8] +
                                      pressure2 * df_dx_node2[node_idx[i + 4] + node_idx[j + 4] * 8];
      }
    }
    evaluator_->compute_d2A_d2u( flipped_pair, mesh1_view, mesh2_view, df_dx_node1, df_dx_node2 );
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 0, 0 )( i, j ) += g_p_ainv1 * df_dx_node1[node_idx[i] + node_idx[j] * 8] +
                                       g_p_ainv2 * df_dx_node2[node_idx[i] + node_idx[j] * 8];
      }
    }
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 0, 1 )( i, j ) += g_p_ainv1 * df_dx_node1[node_idx[i] + node_idx[j + 4] * 8] +
                                       g_p_ainv2 * df_dx_node2[node_idx[i] + node_idx[j + 4] * 8];
      }
    }
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 1, 0 )( i, j ) += g_p_ainv1 * df_dx_node1[node_idx[i + 4] + node_idx[j] * 8] +
                                       g_p_ainv2 * df_dx_node2[node_idx[i + 4] + node_idx[j] * 8];
      }
    }
    for ( int j{ 0 }; j < 4; ++j ) {
      for ( int i{ 0 }; i < 4; ++i ) {
        df_dx_block( 1, 1 )( i, j ) += g_p_ainv1 * df_dx_node1[node_idx[i + 4] + node_idx[j + 4] * 8] +
                                       g_p_ainv2 * df_dx_node2[node_idx[i + 4] + node_idx[j + 4] * 8];
      }
    }
    df_dx_data.storeElemBlockJ( { elem1, elem2 }, df_dx_block );
  }

  // Move gap and area derivatives to HypreParMatrix (submesh rows, parent mesh cols)
  df_dx_ = jac_data_.GetMfemJacobian( convertMethodData(
      df_dx_data, { BlockSpace::NONMORTAR, BlockSpace::MORTAR }, { BlockSpace::NONMORTAR, BlockSpace::MORTAR } ) );

  auto pg2_over_asq = ( 2.0 * pressure_vec_ )
                          .multiplyInPlace( g_tilde_vec_ )
                          .divideInPlace( A_vec_, area_tol_ )
                          .divideInPlace( A_vec_, area_tol_ );

  auto& submesh_fes = submesh_data_.GetSubmeshFESpace();
  auto p_over_a_diag = shared::ParSparseMat::diagonalMatrix( submesh_fes.GetComm(), submesh_fes.GlobalTrueVSize(),
                                                             submesh_fes.GetTrueDofOffsets(), p_over_a.get() );
  auto pg2_over_asq_diag = shared::ParSparseMat::diagonalMatrix( submesh_fes.GetComm(), submesh_fes.GlobalTrueVSize(),
                                                                 submesh_fes.GetTrueDofOffsets(), pg2_over_asq.get() );

  df_dx_ -= shared::ParSparseMat::RAP( dg_tilde_dx_, p_over_a_diag, dA_dx_ );
  df_dx_ -= shared::ParSparseMat::RAP( dA_dx_, p_over_a_diag, dg_tilde_dx_ );
  df_dx_ += shared::ParSparseMat::RAP( dA_dx_, pg2_over_asq_diag, dg_tilde_dx_ );
  df_dx_ += dp_dx.transpose() * dg_tilde_dx_;
  df_dx_ += dg_tilde_dx_.transpose() * dp_dx;
}

RealT NewMethodAdapter::computeTimeStep()
{
  // TODO: implement timestep calculation
  return 1.0;
}

std::unique_ptr<mfem::HypreParMatrix> NewMethodAdapter::getMfemDfDx() const
{
  return std::unique_ptr<mfem::HypreParMatrix>( df_dx_.release() );
}

std::unique_ptr<mfem::HypreParMatrix> NewMethodAdapter::getMfemDgDx() const
{
  return std::unique_ptr<mfem::HypreParMatrix>( dg_tilde_dx_.release() );
}

std::unique_ptr<mfem::HypreParMatrix> NewMethodAdapter::getMfemDfDp() const
{
  SLIC_ERROR_ROOT( "NewMethod does not support getMfemDfDp()" );
  return nullptr;
}

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
