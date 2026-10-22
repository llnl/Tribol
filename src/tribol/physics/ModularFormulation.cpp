// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/ModularFormulation.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/geom/GeomUtilities.hpp"

#include <cmath>
#include <algorithm>
#include <cassert>

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

namespace {

struct Gparams {
  int N;
  const double* qp;
  const double* w;
  const double* x2;
};

void find_normal( const double* coord1, const double* coord2, double* normal )
{
  double dx = coord2[0] - coord1[0];
  double dy = coord2[1] - coord1[1];
  double len = std::sqrt( dy * dy + dx * dx );
  dx /= len;
  dy /= len;
  normal[0] = dy;
  normal[1] = -dx;
}

void determine_legendre_nodes( int N, std::vector<double>& x )
{
  x.resize( N );
  if ( N == 1 ) {
    x[0] = 0.0;
  } else if ( N == 2 ) {
    const double a = 1.0 / std::sqrt( 3.0 );
    x[0] = -a;
    x[1] = a;
  } else if ( N == 3 ) {
    const double a = std::sqrt( 3.0 / 5.0 );
    x[0] = -a;
    x[1] = 0.0;
    x[2] = a;
  } else if ( N == 4 ) {
    const double a = std::sqrt( ( 3.0 - 2.0 * std::sqrt( 6.0 / 5.0 ) ) / 7.0 );
    const double b = std::sqrt( ( 3.0 + 2.0 * std::sqrt( 6.0 / 5.0 ) ) / 7.0 );
    x[0] = -b;
    x[1] = -a;
    x[2] = a;
    x[3] = b;
  } else {
    assert( false && "Unsupported quadrature order" );
  }
}

void determine_legendre_weights( int N, std::vector<double>& W )
{
  W.resize( N );
  if ( N == 1 ) {
    W[0] = 2.0;
  } else if ( N == 2 ) {
    W[0] = 1.0;
    W[1] = 1.0;
  } else if ( N == 3 ) {
    W[0] = 5.0 / 9.0;
    W[1] = 8.0 / 9.0;
    W[2] = 5.0 / 9.0;
  } else {
    W[0] = ( 18 - std::sqrt( 30 ) ) / 36.0;
    W[1] = ( 18 + std::sqrt( 30 ) ) / 36.0;
    W[2] = ( 18 + std::sqrt( 30 ) ) / 36.0;
    W[3] = ( 18 - std::sqrt( 30 ) ) / 36.0;
  }
}

void iso_map( const double* coord1, const double* coord2, double xi, double* mapped_coord )
{
  double N1 = 0.5 - xi;
  double N2 = 0.5 + xi;
  mapped_coord[0] = N1 * coord1[0] + N2 * coord2[0];
  mapped_coord[1] = N1 * coord1[1] + N2 * coord2[1];
}

inline void endpoints( const MeshData::Viewer& mesh, int elem_id, double P0[2], double P1[2] )
{
  double P0_P1[4];
  mesh.getFaceCoords( elem_id, P0_P1 );
  P0[0] = P0_P1[0];
  P0[1] = P0_P1[1];
  P1[0] = P0_P1[2];
  P1[1] = P0_P1[3];
}

void find_intersection( const double* A0, const double* A1, const double* p, const double* nB, double* intersection )
{
  const double tA[2] = { A1[0] - A0[0], A1[1] - A0[1] };
  const double d[2] = { p[0] - A0[0], p[1] - A0[1] };

  const double nlen = std::sqrt( nB[0] * nB[0] + nB[1] * nB[1] );
  if ( nlen < 1e-14 ) {
    intersection[0] = p[0];
    intersection[1] = p[1];
    return;
  }
  const double n[2] = { nB[0] / nlen, nB[1] / nlen };

  const double det = tA[0] * n[1] - tA[1] * n[0];

  if ( std::abs( det ) < 1e-12 ) {
    intersection[0] = p[0];
    intersection[1] = p[1];
    return;
  }

  const double inv_det = 1.0 / det;
  double alpha = ( d[0] * n[1] - d[1] * n[0] ) * inv_det;

  intersection[0] = A0[0] + alpha * tA[0];
  intersection[1] = A0[1] + alpha * tA[1];
}

void get_projections( const double* A0, const double* A1, const double* B0, const double* B1, double* projections )
{
  double nB[2] = { 0.0, 0.0 };
  find_normal( B0, B1, nB );

  const double dxA = A1[0] - A0[0];
  const double dyA = A1[1] - A0[1];
  const double len2A = dxA * dxA + dyA * dyA;

  const double* B_endpoints[2] = { B0, B1 };

  double xi0 = 0.0, xi1 = 0.0;
  for ( int i = 0; i < 2; ++i ) {
    double q[2] = { 0.0, 0.0 };
    find_intersection( A0, A1, B_endpoints[i], nB, q );

    const double alphaA = ( ( q[0] - A0[0] ) * dxA + ( q[1] - A0[1] ) * dyA ) / len2A;
    const double xiA = alphaA - 0.5;

    if ( i == 0 )
      xi0 = xiA;
    else
      xi1 = xiA;
  }

  double xi_min = std::min( xi0, xi1 );
  double xi_max = std::max( xi0, xi1 );

  projections[0] = xi_min;
  projections[1] = xi_max;
}

void gtilde_kernel( const double* x, const Gparams* gp, double* g_tilde_out, double* A_out )
{
  const double A0[2] = { x[0], x[1] };
  const double A1[2] = { x[2], x[3] };
  const double B0[2] = { x[4], x[5] };
  const double B1[2] = { x[6], x[7] };

  const double J = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );
  const double J_ref = J;

  double nB[2];
  find_normal( B0, B1, nB );

  double nA[2];
  find_normal( A0, A1, nA );
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  double eta = ( dot < 0 ) ? dot : 0.0;

  double g1 = 0.0, g2 = 0.0;
  double AI_1 = 0.0, AI_2 = 0.0;

  for ( int i = 0; i < gp->N; ++i ) {
    const double xiA = gp->qp[i];
    const double w = gp->w[i];

    const double N1 = 0.5 - xiA;
    const double N2 = 0.5 + xiA;

    double x1[2];
    iso_map( A0, A1, xiA, x1 );

    double x2[2];
    find_intersection( B0, B1, x1, nB, x2 );

    const double dx = x1[0] - x2[0];
    const double dy = x1[1] - x2[1];

    const double gn = -( dx * nB[0] + dy * nB[1] );
    const double g = gn * eta;

    g1 += w * N1 * g * J;
    g2 += w * N2 * g * J;

    AI_1 += w * N1 * J_ref;
    AI_2 += w * N2 * J_ref;
  }

  g_tilde_out[0] = g1;
  g_tilde_out[1] = g2;

  A_out[0] = AI_1;
  A_out[1] = AI_2;
}

static void gtilde1_out( const double* x, const void* gp_void, double* out )
{
  const Gparams* gp = static_cast<const Gparams*>( gp_void );
  double gt[2];
  double A_out[2];
  gtilde_kernel( x, gp, gt, A_out );
  *out = gt[0];
}

static void gtilde2_out( const double* x, const void* gp_void, double* out )
{
  const Gparams* gp = static_cast<const Gparams*>( gp_void );
  double gt[2];
  double A_out[2];
  gtilde_kernel( x, gp, gt, A_out );
  *out = gt[1];
}

static void A1_out( const double* x, const void* gp_void, double* out )
{
  const Gparams* gp = static_cast<const Gparams*>( gp_void );
  double gt[2];
  double A_out[2];
  gtilde_kernel( x, gp, gt, A_out );
  *out = A_out[0];
}

static void A2_out( const double* x, const void* gp_void, double* out )
{
  const Gparams* gp = static_cast<const Gparams*>( gp_void );
  double gt[2];
  double A_out[2];
  gtilde_kernel( x, gp, gt, A_out );
  *out = A_out[1];
}

void grad_gtilde1( const double* x, const Gparams* gp, double* dgt1_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;
  __enzyme_autodiff<void>( (void*)gtilde1_out, enzyme_dup, x, dx, enzyme_const, (const void*)gp, enzyme_dup, &out,
                           &dout );
  for ( int i = 0; i < 8; ++i ) {
    dgt1_du[i] = dx[i];
  }
}

void grad_gtilde2( const double* x, const Gparams* gp, double* dgt2_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;
  __enzyme_autodiff<void>( (void*)gtilde2_out, enzyme_dup, x, dx, enzyme_const, (const void*)gp, enzyme_dup, &out,
                           &dout );
  for ( int i = 0; i < 8; ++i ) {
    dgt2_du[i] = dx[i];
  }
}

void grad_A1( const double* x, const Gparams* gp, double* dA1_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;
  __enzyme_autodiff<void>( (void*)A1_out, enzyme_dup, x, dx, enzyme_const, (const void*)gp, enzyme_dup, &out, &dout );
  for ( int i = 0; i < 8; ++i ) {
    dA1_du[i] = dx[i];
  }
}

void grad_A2( const double* x, const Gparams* gp, double* dA2_du )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;
  __enzyme_autodiff<void>( (void*)A2_out, enzyme_dup, x, dx, enzyme_const, (const void*)gp, enzyme_dup, &out, &dout );
  for ( int i = 0; i < 8; ++i ) {
    dA2_du[i] = dx[i];
  }
}

void d2gtilde1( const double* x, const Gparams* gp, double* H1 )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;
    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };
    __enzyme_fwddiff<void>( (void*)grad_gtilde1, x, dx, enzyme_const, (const void*)gp, enzyme_dup, grad, dgrad );
    for ( int row = 0; row < 8; ++row ) {
      H1[row * 8 + col] = dgrad[row];
    }
  }
}

void d2gtilde2( const double* x, const Gparams* gp, double* H2 )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;
    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };
    __enzyme_fwddiff<void>( (void*)grad_gtilde2, x, dx, enzyme_const, (const void*)gp, enzyme_dup, grad, dgrad );
    for ( int row = 0; row < 8; ++row ) {
      H2[row * 8 + col] = dgrad[row];
    }
  }
}

void get_d2A1( const double* x, const Gparams* gp, double* H1 )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;
    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };
    __enzyme_fwddiff<void>( (void*)grad_A1, x, dx, enzyme_const, (const void*)gp, enzyme_dup, grad, dgrad );
    for ( int row = 0; row < 8; ++row ) {
      H1[row * 8 + col] = dgrad[row];
    }
  }
}

void get_d2A2( const double* x, const Gparams* gp, double* H1 )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;
    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };
    __enzyme_fwddiff<void>( (void*)grad_A2, x, dx, enzyme_const, (const void*)gp, enzyme_dup, grad, dgrad );
    for ( int row = 0; row < 8; ++row ) {
      H1[row * 8 + col] = dgrad[row];
    }
  }
}

}  // namespace

ModularFormulation::ModularFormulation( MfemSubmeshData& submesh_data, MfemJacobianData& jac_data, MeshData& mesh1,
                                        MeshData& mesh2, double k, double delta, int N )
    : submesh_data_( submesh_data ), jac_data_( jac_data ), mesh1_( mesh2 ), mesh2_( mesh1 )
{
  if ( mesh1.numberOfNodes() > 0 && mesh2.numberOfNodes() > 0 ) {
    SLIC_ERROR_ROOT_IF( mesh1.spatialDimension() != 2 || mesh2.spatialDimension() != 2,
                        "MODULAR_FORMULATION requires 2D meshes." );
  }

  params_.k = k;
  params_.del = delta;
  params_.N = N;
}

void ModularFormulation::setInterfacePairs( ArrayT<InterfacePair>&& pairs, int /*check_level*/ )
{
  pairs_ = std::move( pairs );
}

void ModularFormulation::updateIntegrationRule()
{
}

void ModularFormulation::updateNodalGaps()
{
  auto& redecomp_gap = submesh_data_.GetRedecompGap();
  mfem::GridFunction redecomp_area( redecomp_gap.FESpace() );
  redecomp_area = 0.0;

  JacobianContributions dg_tilde_dx_contribs( { { BlockSpace::LAGRANGE_MULTIPLIER, BlockSpace::NONMORTAR },
                                                { BlockSpace::LAGRANGE_MULTIPLIER, BlockSpace::MORTAR } } );
  JacobianContributions dA_dx_contribs( { { BlockSpace::LAGRANGE_MULTIPLIER, BlockSpace::NONMORTAR },
                                          { BlockSpace::LAGRANGE_MULTIPLIER, BlockSpace::MORTAR } } );

  dg_tilde_dx_contribs.reserve( pairs_.size(), 8 );
  dA_dx_contribs.reserve( pairs_.size(), 8 );

  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  auto mesh1_view = mesh1_.getView();
  auto mesh2_view = mesh2_.getView();

  for ( const auto& pair : pairs_ ) {
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    double g_tilde_elem[2];
    double A_elem[2];

    this->gtilde_and_area( flipped_pair, mesh1_view, mesh2_view, g_tilde_elem, A_elem );

    if ( A_elem[0] <= 0.0 && A_elem[1] <= 0.0 ) {
      continue;
    }

    auto A_conn = mesh1_view.getConnectivity()( elem1 );

    redecomp_gap( A_conn[0] ) += g_tilde_elem[0];
    redecomp_gap( A_conn[1] ) += g_tilde_elem[1];

    redecomp_area( A_conn[0] ) += A_elem[0];
    redecomp_area( A_conn[1] ) += A_elem[1];

    double dg_dx_node1[8];
    double dg_dx_node2[8];
    this->grad_gtilde( flipped_pair, mesh1_view, mesh2_view, dg_dx_node1, dg_dx_node2 );
    double dg_tilde_dx_blocks[2][8];
    for ( int i{ 0 }; i < 4; ++i ) {
      dg_tilde_dx_blocks[0][i * 2] = dg_dx_node1[node_idx[i]];
      dg_tilde_dx_blocks[0][i * 2 + 1] = dg_dx_node2[node_idx[i]];
      dg_tilde_dx_blocks[1][i * 2] = dg_dx_node1[node_idx[i + 4]];
      dg_tilde_dx_blocks[1][i * 2 + 1] = dg_dx_node2[node_idx[i + 4]];
    }
    dg_tilde_dx_contribs.push_back( 0, elem1, elem1, dg_tilde_dx_blocks[0], 8 );
    dg_tilde_dx_contribs.push_back( 1, elem1, elem2, dg_tilde_dx_blocks[1], 8 );

    double dA_dx_node1[8];
    double dA_dx_node2[8];
    this->grad_trib_area( flipped_pair, mesh1_view, mesh2_view, dA_dx_node1, dA_dx_node2 );
    double dA_dx_blocks[2][8];
    for ( int i{ 0 }; i < 4; ++i ) {
      dA_dx_blocks[0][i * 2] = dA_dx_node1[node_idx[i]];
      dA_dx_blocks[0][i * 2 + 1] = dA_dx_node2[node_idx[i]];
      dA_dx_blocks[1][i * 2] = dA_dx_node1[node_idx[i + 4]];
      dA_dx_blocks[1][i * 2 + 1] = dA_dx_node2[node_idx[i + 4]];
    }
    dA_dx_contribs.push_back( 0, elem1, elem1, dA_dx_blocks[0], 8 );
    dA_dx_contribs.push_back( 1, elem1, elem2, dA_dx_blocks[1], 8 );
  }

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

  dg_tilde_dx_ = jac_data_.GetMfemJacobian( dg_tilde_dx_contribs.get() );
  if ( !tied_contact_ ) {
    dg_tilde_dx_.EliminateRows( rows_to_elim );
  }

  dA_dx_ = jac_data_.GetMfemJacobian( dA_dx_contribs.get() );
}

void ModularFormulation::updateNodalForces()
{
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

  JacobianContributions df_dx_contribs( { { BlockSpace::NONMORTAR, BlockSpace::NONMORTAR },
                                          { BlockSpace::NONMORTAR, BlockSpace::MORTAR },
                                          { BlockSpace::MORTAR, BlockSpace::NONMORTAR },
                                          { BlockSpace::MORTAR, BlockSpace::MORTAR } } );

  df_dx_contribs.reserve( pairs_.size(), 16 );

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

  for ( auto& pair : pairs_ ) {
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto node11 = mesh1_view.getConnectivity()( elem1, 0 );
    const auto node12 = mesh1_view.getConnectivity()( elem1, 1 );

    const RealT pressure1 = 2.0 * redecomp_pressure( node11 );
    const RealT pressure2 = 2.0 * redecomp_pressure( node12 );

    if ( pressure1 == 0.0 && pressure2 == 0.0 ) {
      continue;
    }

    const RealT g_p_ainv1 = -redecomp_g_tilde( node11 ) * redecomp_pressure( node11 ) / redecomp_A( node11 );
    const RealT g_p_ainv2 = -redecomp_g_tilde( node12 ) * redecomp_pressure( node12 ) / redecomp_A( node12 );

    double d2g_dx2_node1[64];
    double d2g_dx2_node2[64];
    this->d2_g2tilde( flipped_pair, mesh1_view, mesh2_view, d2g_dx2_node1, d2g_dx2_node2 );

    double d2A_dx2_node1[64];
    double d2A_dx2_node2[64];
    this->compute_d2A_d2u( flipped_pair, mesh1_view, mesh2_view, d2A_dx2_node1, d2A_dx2_node2 );

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

    df_dx_contribs.push_back( 0, elem1, elem1, df_dx_blocks[0][0], 16 );
    df_dx_contribs.push_back( 1, elem1, elem2, df_dx_blocks[0][1], 16 );
    df_dx_contribs.push_back( 2, elem2, elem1, df_dx_blocks[1][0], 16 );
    df_dx_contribs.push_back( 3, elem2, elem2, df_dx_blocks[1][1], 16 );
  }

  df_dx_ = jac_data_.GetMfemJacobian( df_dx_contribs.get() );

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

RealT ModularFormulation::computeTimeStep()
{
  return 1.0;
}

std::unique_ptr<mfem::HypreParMatrix> ModularFormulation::getMfemDfDx() const
{
  return std::unique_ptr<mfem::HypreParMatrix>( df_dx_.release() );
}

std::unique_ptr<mfem::HypreParMatrix> ModularFormulation::getMfemDgDx() const
{
  return std::unique_ptr<mfem::HypreParMatrix>( dg_tilde_dx_.release() );
}

std::unique_ptr<mfem::HypreParMatrix> ModularFormulation::getMfemDfDp() const
{
  SLIC_ERROR_ROOT( "ModularFormulation does not support getMfemDfDp()" );
  return nullptr;
}

// --- Physics Implementation ---

std::array<double, 2> ModularFormulation::projections( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                       const MeshData::Viewer& mesh2 ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );
  double projs[2];
  get_projections( A0, A1, B0, B1, projs );
  return { projs[0], projs[1] };
}

std::array<double, 2> ModularFormulation::bounds_from_projections( const std::array<double, 2>& proj ) const
{
  double xi_min = std::min( proj[0], proj[1] );
  double xi_max = std::max( proj[0], proj[1] );
  const double del = params_.del;
  if ( xi_max < -0.5 - del ) xi_max = -0.5 - del;
  if ( xi_min > 0.5 + del ) xi_min = 0.5 + del;
  if ( xi_min < -0.5 - del ) xi_min = -0.5 - del;
  if ( xi_max > 0.5 + del ) xi_max = 0.5 + del;
  return { xi_min, xi_max };
}

std::array<double, 2> ModularFormulation::smooth_bounds( const std::array<double, 2>& bounds ) const
{
  std::array<double, 2> smooth_bounds;
  const double del = params_.del;
  for ( int i = 0; i < 2; ++i ) {
    double xi = bounds[i] + 0.5;
    double xi_hat = 0.0;
    if ( 0.0 - del <= xi && xi <= del ) {
      xi_hat = ( 1.0 / ( 4 * del ) ) * ( xi * xi ) + 0.5 * xi + del / 4.0;
    } else if ( ( 1.0 - del ) <= xi && xi <= 1.0 + del ) {
      double b = -1.0 / ( 4.0 * del );
      double c = 0.5 + 1.0 / ( 2.0 * del );
      double d = 1.0 - del + ( 1.0 / ( 4.0 * del ) ) * std::pow( 1.0 - del, 2 ) - 0.5 * ( 1.0 - del ) -
                 ( 1.0 - del ) / ( 2.0 * del );
      xi_hat = b * xi * xi + c * xi + d;
    } else if ( del <= xi && xi <= ( 1.0 - del ) ) {
      xi_hat = xi;
    }
    smooth_bounds[i] = xi_hat - 0.5;
  }
  return smooth_bounds;
}

ModularFormulation::QuadPoints ModularFormulation::compute_quadrature( const std::array<double, 2>& xi_bounds ) const
{
  const int N = params_.N;
  QuadPoints out;
  out.qp.resize( N );
  out.w.resize( N );
  std::vector<double> qpoints( N );
  std::vector<double> weights( N );
  determine_legendre_nodes( N, qpoints );
  determine_legendre_weights( N, weights );
  const double xi_min = xi_bounds[0];
  const double xi_max = xi_bounds[1];
  const double J = 0.5 * ( xi_max - xi_min );
  for ( int i = 0; i < N; ++i ) {
    out.qp[i] = 0.5 * ( xi_max - xi_min ) * qpoints[i] + 0.5 * ( xi_max + xi_min );
    out.w[i] = weights[i] * J;
  }
  return out;
}

double ModularFormulation::gap( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                double xiA ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );
  double nA[2], nB[2];
  find_normal( A0, A1, nA );
  find_normal( B0, B1, nB );
  double x1[2];
  iso_map( A0, A1, xiA, x1 );
  double x2[2];
  find_intersection( B0, B1, x1, nB, x2 );
  double dx = x1[0] - x2[0];
  double dy = x1[1] - x2[1];
  double gn = -( dx * nB[0] + dy * nB[1] );
  double dot = nB[0] * nA[0] + nB[1] * nA[1];
  double eta = ( dot < 0 ) ? dot : 0.0;
  return gn * eta;
}

ModularFormulation::NodalContactData ModularFormulation::compute_nodal_contact_data( const InterfacePair& pair,
                                                                                      const MeshData::Viewer& mesh1,
                                                                                      const MeshData::Viewer& mesh2 ) const
{
  double A0[2], A1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  double J = std::sqrt( std::pow( A1[0] - A0[0], 2 ) + std::pow( A1[1] - A0[1], 2 ) );
  double J_ref = J;

  auto projs = projections( pair, mesh1, mesh2 );
  auto bounds = bounds_from_projections( projs );
  auto smooth_bounds = smooth_bounds( bounds );
  auto qp = compute_quadrature( smooth_bounds );

  double g_tilde1 = 0.0, g_tilde2 = 0.0, AI_1 = 0.0, AI_2 = 0.0;

  for ( size_t i = 0; i < qp.qp.size(); ++i ) {
    double xiA = qp.qp[i];
    double w = qp.w[i];
    double N1 = 0.5 - xiA, N2 = 0.5 + xiA;
    double gn = gap( pair, mesh1, mesh2, xiA );
    g_tilde1 += w * N1 * gn * J;
    g_tilde2 += w * N2 * gn * J;
    AI_1 += w * N1 * J_ref;
    AI_2 += w * N2 * J_ref;
  }

  NodalContactData contact_data;
  contact_data.AI = { AI_1, AI_2 };
  contact_data.g_tilde = { g_tilde1, g_tilde2 };
  return contact_data;
}

std::array<double, 2> ModularFormulation::compute_pressures( const NodalContactData& ncd ) const
{
  double g1 = ncd.g_tilde[0] / ncd.AI[0];
  double g2 = ncd.g_tilde[1] / ncd.AI[1];
  double p1 = ( g1 < 0.0 ) ? params_.k * g1 : 0.0;
  double p2 = ( g2 < 0.0 ) ? params_.k * g2 : 0.0;
  std::array<double, 2> pressures = { p1, p2 };
  for ( int i = 0; i < 2; ++i ) {
    if ( ncd.AI[i] < 1e-12 ) pressures[i] = 0.0;
  }
  return pressures;
}

void ModularFormulation::gtilde_and_area( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                          const MeshData::Viewer& mesh2, double gtilde[2], double area[2] ) const
{
  auto ncd = compute_nodal_contact_data( pair, mesh1, mesh2 );
  gtilde[0] = ncd.g_tilde[0];
  gtilde[1] = ncd.g_tilde[1];
  area[0] = ncd.AI[0];
  area[1] = ncd.AI[1];
}

void ModularFormulation::grad_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                      const MeshData::Viewer& mesh2, double dgt1_dx[8], double dgt2_dx[8] ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );
  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  auto projs = projections( pair, mesh1, mesh2 );
  auto bounds = bounds_from_projections( projs );
  auto s_bounds = smooth_bounds( bounds );
  auto qp = compute_quadrature( s_bounds );

  const int N = static_cast<int>( qp.qp.size() );
  std::vector<double> x2( 2 * N );
  double nB[2];
  find_normal( B0, B1, nB );
  for ( int i = 0; i < N; ++i ) {
    double x1[2];
    iso_map( A0, A1, qp.qp[i], x1 );
    double x2_i[2];
    find_intersection( B0, B1, x1, nB, x2_i );
    x2[2 * i] = x2_i[0];
    x2[2 * i + 1] = x2_i[1];
  }

  Gparams gp;
  gp.N = N;
  gp.qp = qp.qp.data();
  gp.w = qp.w.data();
  gp.x2 = x2.data();

  grad_gtilde1( x, &gp, dgt1_dx );
  grad_gtilde2( x, &gp, dgt2_dx );
}

void ModularFormulation::grad_trib_area( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                         const MeshData::Viewer& mesh2, double dA1_dx[8], double dA2_dx[8] ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );
  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  auto projs = projections( pair, mesh1, mesh2 );
  auto bounds = bounds_from_projections( projs );
  auto s_bounds = smooth_bounds( bounds );
  auto qp = compute_quadrature( s_bounds );

  const int N = static_cast<int>( qp.qp.size() );
  std::vector<double> x2( 2 * N );
  double nB[2];
  find_normal( B0, B1, nB );
  for ( int i = 0; i < N; ++i ) {
    double x1[2];
    iso_map( A0, A1, qp.qp[i], x1 );
    double x2_i[2];
    find_intersection( B0, B1, x1, nB, x2_i );
    x2[2 * i] = x2_i[0];
    x2[2 * i + 1] = x2_i[1];
  }

  Gparams gp;
  gp.N = N;
  gp.qp = qp.qp.data();
  gp.w = qp.w.data();
  gp.x2 = x2.data();

  grad_A1( x, &gp, dA1_dx );
  grad_A2( x, &gp, dA2_dx );
}

void ModularFormulation::d2_g2tilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                     const MeshData::Viewer& mesh2, double H1[64], double H2[64] ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );
  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  auto projs = projections( pair, mesh1, mesh2 );
  auto bounds = bounds_from_projections( projs );
  auto s_bounds = smooth_bounds( bounds );
  auto qp = compute_quadrature( s_bounds );

  const int N = static_cast<int>( qp.qp.size() );
  std::vector<double> x2( 2 * N );
  double nB[2];
  find_normal( B0, B1, nB );
  for ( int i = 0; i < N; ++i ) {
    double x1[2];
    iso_map( A0, A1, qp.qp[i], x1 );
    double x2_i[2];
    find_intersection( B0, B1, x1, nB, x2_i );
    x2[2 * i] = x2_i[0];
    x2[2 * i + 1] = x2_i[1];
  }

  Gparams gp;
  gp.N = N;
  gp.qp = qp.qp.data();
  gp.w = qp.w.data();
  gp.x2 = x2.data();

  d2gtilde1( x, &gp, H1 );
  d2gtilde2( x, &gp, H2 );
}

void ModularFormulation::compute_d2A_d2u( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                          const MeshData::Viewer& mesh2, double d2A1[64], double d2A2[64] ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );
  double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  auto projs = projections( pair, mesh1, mesh2 );
  auto bounds = bounds_from_projections( projs );
  auto s_bounds = smooth_bounds( bounds );
  auto qp = compute_quadrature( s_bounds );

  const int N = static_cast<int>( qp.qp.size() );
  std::vector<double> x2( 2 * N );
  double nB[2];
  find_normal( B0, B1, nB );
  for ( int i = 0; i < N; ++i ) {
    double x1[2];
    iso_map( A0, A1, qp.qp[i], x1 );
    double x2_i[2];
    find_intersection( B0, B1, x1, nB, x2_i );
    x2[2 * i] = x2_i[0];
    x2[2 * i + 1] = x2_i[1];
  }

  Gparams gp;
  gp.N = N;
  gp.qp = qp.qp.data();
  gp.w = qp.w.data();
  gp.x2 = x2.data();

  get_d2A1( x, &gp, d2A1 );
  get_d2A2( x, &gp, d2A2 );
}

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
