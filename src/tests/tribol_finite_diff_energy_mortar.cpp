// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <cmath>
#include <set>
#include "/usr/workspace/lutz23/Smith/repo/tribol/src/tribol/physics/new_method.hpp"
// #include <tribol/src/tribol/physics/new_method.hpp>
// #include "tribol/physics/new_method.hpp"
#include <gtest/gtest.h>

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

#include "mfem.hpp"

#include "axom/CLI11.hpp"
#include "axom/slic.hpp"

#include "shared/mesh/MeshBuilder.hpp"
#include "redecomp/redecomp.hpp"

#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"

namespace tribol {

static ContactSmoothing smoother( ContactParams{} );

inline void endpoints( const MeshData::Viewer& mesh, int elem_id, double P0[2], double P1[2] )
{
  double P0_P1[4];
  mesh.getFaceCoords( elem_id, P0_P1 );
  P0[0] = P0_P1[0];
  P0[1] = P0_P1[1];
  P1[0] = P0_P1[2];
  P1[1] = P0_P1[3];
}

std::pair<double, double> ContactEvaluator::eval_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                         const MeshData::Viewer& mesh2 ) const
{
  NodalContactData ncd = compute_nodal_contact_data( pair, mesh1, mesh2 );
  // double gt1 = ncd.g_tilde[0];
  // double gt2 = ncd.g_tilde[1];
  double A1 = ncd.g_tilde[0];
  double A2 = ncd.g_tilde[1];

  return { A1, A2 };
}

FiniteDiffResult ContactEvaluator::validate_g_tilde( const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
                                                     double epsilon ) const
{
  FiniteDiffResult result;

  auto viewer1 = mesh1.getView();
  auto viewer2 = mesh2.getView();

  auto projs0 = projections( pair, viewer1, viewer2 );
  auto bounds0 = smoother_.bounds_from_projections( projs0, smoother.get_del() );
  auto smooth_bounds0 = smoother_.smooth_bounds( bounds0, smoother.get_del() );
  QuadPoints qp0;
  if ( !p_.enzyme_quadrature ) {
    qp0 = compute_quadrature( smooth_bounds0 );
  }

  auto [g1_base, g2_base] = eval_gtilde( pair, viewer1, viewer2 );

  result.g_tilde1_baseline = g1_base;
  result.g_tilde2_baseline = g2_base;


  auto A_conn = viewer1.getConnectivity()( static_cast<std::size_t>( pair.m_element_id1 ) );
  auto B_conn = viewer2.getConnectivity()( static_cast<std::size_t>( pair.m_element_id2 ) );


  result.node_ids = { static_cast<int>( A_conn[0] ), static_cast<int>( A_conn[1] ), static_cast<int>( B_conn[0] ),
                      static_cast<int>( B_conn[1] ) };

  const int num_dofs = 8;
  result.fd_gradient_g1.resize( num_dofs );
  result.fd_gradient_g2.resize( num_dofs );
  result.analytical_gradient_g1.resize( num_dofs );
  result.analytical_gradient_g2.resize( num_dofs );

  // ===== ANALYTICAL GRADIENTS =====
  double dgt1_dx[8] = { 0.0 };
  double dgt2_dx[8] = { 0.0 };
  grad_gtilde( pair, viewer1, viewer2, dgt1_dx, dgt2_dx );
  for ( size_t i = 0; i < 8; ++i ) {
    result.analytical_gradient_g1[i] = dgt1_dx[i];
    result.analytical_gradient_g2[i] = dgt2_dx[i];
  }


  // ===== ORIGINAL COORDS =====
  const IndexT num_nodes1 = mesh1.numberOfNodes();
  const std::size_t n1 = static_cast<std::size_t>( num_nodes1 );

  std::vector<RealT> x1_orig( n1 ), y1_orig( n1 );
  {
    auto pos = mesh1.getView().getPosition();
    for ( IndexT i = 0; i < num_nodes1; ++i ) {
      const std::size_t iu = static_cast<std::size_t>( i );
      x1_orig[iu] = pos[0][i];
      y1_orig[iu] = pos[1][i];
    }
  }

  IndexT num_nodes2 = mesh2.numberOfNodes();
  const std::size_t n2 = static_cast<std::size_t>( num_nodes2 );

  std::vector<RealT> x2_orig( n2 ), y2_orig( n2 );
  {
    auto pos = mesh2.getView().getPosition();
    for ( int i = 0; i < num_nodes2; ++i ) {
      const std::size_t iu = static_cast<std::size_t>( i );
      x2_orig[iu] = pos[0][i];
      y2_orig[iu] = pos[1][i];
    }
  }

  auto eval = [&]( const MeshData::Viewer& v1, const MeshData::Viewer& v2 ) -> std::pair<double, double> {
    return p_.enzyme_quadrature ? eval_gtilde( pair, v1, v2 ) : eval_gtilde_fixed_qp( pair, v1, v2, qp0 );
  };

  // ===== FINITE DIFFERENCE GRADIENTS =====
  size_t dof_idx = 0;

  // A nodes → perturb mesh1
  for ( int k = 0; k < 2; ++k ) {
    const int local_node = A_conn[k];

    // x perturbation
    {
      auto x_pert = x1_orig;
      x_pert[static_cast<std::size_t>( local_node )] += epsilon;
      mesh1.setPosition( x_pert.data(), y1_orig.data(), nullptr );
      auto [g1p, g2p] = eval( mesh1.getView(), mesh2.getView() );

      x_pert[static_cast<std::size_t>( local_node )] = x1_orig[static_cast<std::size_t>( local_node )] - epsilon;
      mesh1.setPosition( x_pert.data(), y1_orig.data(), nullptr );
      auto [g1m, g2m] = eval( mesh1.getView(), mesh2.getView() );

      mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
      result.fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
      result.fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
      dof_idx++;
    }

    // y perturbation
    {
      auto y_pert = y1_orig;
      y_pert[static_cast<std::size_t>( local_node )] += epsilon;
      mesh1.setPosition( x1_orig.data(), y_pert.data(), nullptr );
      auto [g1p, g2p] = eval( mesh1.getView(), mesh2.getView() );

      y_pert[static_cast<std::size_t>( local_node )] = y1_orig[static_cast<std::size_t>( local_node )] - epsilon;
      mesh1.setPosition( x1_orig.data(), y_pert.data(), nullptr );
      auto [g1m, g2m] = eval( mesh1.getView(), mesh2.getView() );

      mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
      result.fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
      result.fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
      dof_idx++;
    }
  }

  // B nodes → perturb mesh2
  for ( int k = 0; k < 2; ++k ) {
    const int local_node = B_conn[k];

    // x perturbation
    {
      auto x_pert = x2_orig;
      x_pert[static_cast<std::size_t>( local_node )] += epsilon;
      mesh2.setPosition( x_pert.data(), y2_orig.data(), nullptr );
      auto [g1p, g2p] = eval( mesh1.getView(), mesh2.getView() );

      x_pert[static_cast<std::size_t>( local_node )] = x2_orig[static_cast<std::size_t>( local_node )] - epsilon;
      mesh2.setPosition( x_pert.data(), y2_orig.data(), nullptr );
      auto [g1m, g2m] = eval( mesh1.getView(), mesh2.getView() );

      mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );
      result.fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
      result.fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
      dof_idx++;
    }

    // y perturbation
    {
      auto y_pert = y2_orig;
      y_pert[static_cast<std::size_t>( local_node )] += epsilon;
      mesh2.setPosition( x2_orig.data(), y_pert.data(), nullptr );
      auto [g1p, g2p] = eval( mesh1.getView(), mesh2.getView() );

      y_pert[static_cast<std::size_t>( local_node )] = y2_orig[static_cast<std::size_t>( local_node )] - epsilon;
      mesh2.setPosition( x2_orig.data(), y_pert.data(), nullptr );
      auto [g1m, g2m] = eval( mesh1.getView(), mesh2.getView() );

      mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );
      result.fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
      result.fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
      dof_idx++;
    }
  }

  return result;
}

std::pair<double, double> ContactEvaluator::eval_gtilde_fixed_qp( const InterfacePair& pair,
                                                                  const MeshData::Viewer& mesh1,
                                                                  const MeshData::Viewer& mesh2,
                                                                  const QuadPoints& qp_fixed ) const
{
  double A0[2], A1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );

  const double J = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  double gt1 = 0.0, gt2 = 0.0;

  for ( size_t i = 0; i < qp_fixed.qp.size(); ++i ) {
    const double xiA = qp_fixed.qp[i];
    const double w = qp_fixed.w[i];

    const double N1 = 0.5 - xiA;
    const double N2 = 0.5 + xiA;

    const double gn = gap( pair, mesh1, mesh2, xiA );  


    gt1 += w * N1 * J * gn;
    gt2 += w * N2 * J * gn;
  }

  return { gt1, gt2 };
}








FiniteDiffResult ContactEvaluator::validate_hessian( const InterfacePair& pair,
                                                     MeshData& mesh1,
                                                     MeshData& mesh2,
                                                     double epsilon ) const
{
  FiniteDiffResult result;

  auto viewer1 = mesh1.getView();
  auto viewer2 = mesh2.getView();

  double hess1[64] = { 0.0 };
  double hess2[64] = { 0.0 };

  const int ndof = 8;
  result.fd_gradient_g1.assign( ndof * ndof, 0.0 );
  result.fd_gradient_g2.assign( ndof * ndof, 0.0 );

  auto A_conn = viewer1.getConnectivity()( static_cast<std::size_t>( pair.m_element_id1 ) );
  auto B_conn = viewer2.getConnectivity()( static_cast<std::size_t>( pair.m_element_id2 ) );

  result.node_ids = {
    static_cast<int>( A_conn[0] ),
    static_cast<int>( A_conn[1] ),
    static_cast<int>( B_conn[0] ),
    static_cast<int>( B_conn[1] )
  };

  // analytical Hessian
  d2_g2tilde( pair, viewer1, viewer2, hess1, hess2 );
  result.analytical_gradient_g1.assign( hess1, hess1 + 64 );
  result.analytical_gradient_g2.assign( hess2, hess2 + 64 );

  // ===== ORIGINAL COORDS =====
  const IndexT num_nodes1 = mesh1.numberOfNodes();
  const std::size_t n1 = static_cast<std::size_t>( num_nodes1 );
  std::vector<RealT> x1_orig( n1 ), y1_orig( n1 );
  {
    auto pos = mesh1.getView().getPosition();
    for ( IndexT i = 0; i < num_nodes1; ++i ) {
      const std::size_t iu = static_cast<std::size_t>( i );
      x1_orig[iu] = pos[0][i];
      y1_orig[iu] = pos[1][i];
    }
  }

  const IndexT num_nodes2 = mesh2.numberOfNodes();
  const std::size_t n2 = static_cast<std::size_t>( num_nodes2 );
  std::vector<RealT> x2_orig( n2 ), y2_orig( n2 );
  {
    auto pos = mesh2.getView().getPosition();
    for ( IndexT i = 0; i < num_nodes2; ++i ) {
      const std::size_t iu = static_cast<std::size_t>( i );
      x2_orig[iu] = pos[0][i];
      y2_orig[iu] = pos[1][i];
    }
  }

  // ===== FIXED QUADRATURE FOR enzyme_quadrature = false =====
  QuadPoints qp0;
  if ( !p_.enzyme_quadrature ) {
    auto projs0 = projections( pair, viewer1, viewer2 );
    auto bounds0 = smoother_.bounds_from_projections( projs0, smoother.get_del() );
    auto smooth_bounds0 = smoother_.smooth_bounds( bounds0, smoother.get_del() );
    qp0 = compute_quadrature( smooth_bounds0 );
  }

  auto eval_from_offsets = [&]( const std::array<double, 8>& du ) -> std::pair<double, double> {
    auto x1 = x1_orig;
    auto y1 = y1_orig;
    auto x2 = x2_orig;
    auto y2 = y2_orig;

    x1[static_cast<std::size_t>( A_conn[0] )] += du[0];
    y1[static_cast<std::size_t>( A_conn[0] )] += du[1];
    x1[static_cast<std::size_t>( A_conn[1] )] += du[2];
    y1[static_cast<std::size_t>( A_conn[1] )] += du[3];

    x2[static_cast<std::size_t>( B_conn[0] )] += du[4];
    y2[static_cast<std::size_t>( B_conn[0] )] += du[5];
    x2[static_cast<std::size_t>( B_conn[1] )] += du[6];
    y2[static_cast<std::size_t>( B_conn[1] )] += du[7];

    mesh1.setPosition( x1.data(), y1.data(), nullptr );
    mesh2.setPosition( x2.data(), y2.data(), nullptr );

    if ( p_.enzyme_quadrature ) {
      return eval_gtilde( pair, mesh1.getView(), mesh2.getView() );
    } else {
      return eval_gtilde_fixed_qp( pair, mesh1.getView(), mesh2.getView(), qp0 );
    }
  };

  const std::array<double, 8> zero = { 0., 0., 0., 0., 0., 0., 0., 0. };
  const auto [g10, g20] = eval_from_offsets( zero );

  // ===== FD HESSIAN =====
  for ( size_t i = 0; i < ndof; ++i ) {
    for ( size_t j = 0; j < ndof; ++j ) {
      const std::size_t idx = static_cast<std::size_t>( i * ndof + j );

      if ( i == j ) {
        std::array<double, 8> up = zero;
        std::array<double, 8> um = zero;
        up[i] += epsilon;
        um[i] -= epsilon;

        const auto [g1p, g2p] = eval_from_offsets( up );
        const auto [g1m, g2m] = eval_from_offsets( um );

        result.fd_gradient_g1[idx] = ( g1p - 2.0 * g10 + g1m ) / ( epsilon * epsilon );
        result.fd_gradient_g2[idx] = ( g2p - 2.0 * g20 + g2m ) / ( epsilon * epsilon );
      } else {
        std::array<double, 8> upp = zero;
        std::array<double, 8> upm = zero;
        std::array<double, 8> ump = zero;
        std::array<double, 8> umm = zero;

        upp[i] += epsilon; upp[j] += epsilon;
        upm[i] += epsilon; upm[j] -= epsilon;
        ump[i] -= epsilon; ump[j] += epsilon;
        umm[i] -= epsilon; umm[j] -= epsilon;

        const auto [g1pp, g2pp] = eval_from_offsets( upp );
        const auto [g1pm, g2pm] = eval_from_offsets( upm );
        const auto [g1mp, g2mp] = eval_from_offsets( ump );
        const auto [g1mm, g2mm] = eval_from_offsets( umm );

        result.fd_gradient_g1[idx] = ( g1pp - g1pm - g1mp + g1mm ) / ( 4.0 * epsilon * epsilon );
        result.fd_gradient_g2[idx] = ( g2pp - g2pm - g2mp + g2mm ) / ( 4.0 * epsilon * epsilon );
      }
    }
  }

  mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
  mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );

  return result;
}




TEST( GradientCheck, GtildeFDvsAD )
{
  // ── Geometry: two facing LINEAR_EDGE segments ────────────────────────────
  // Segment A: (0,0) -> (1,0)
  // Segment B: (0.2, 0.5) -> (0.8, 0.5)

  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };  // reversed so nA points toward B
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.5, 0.5 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  InterfacePair pair( 0, 0 );

  // ── Evaluator setup ──────────────────────────────────────────────────────
  ContactParams params;
  params.del = 0.1;                 // smoothing m
  params.k = 1.0;                   // penalty stiffness
  params.N = 3;                     // quadrature points
  params.enzyme_quadrature = true;  // use the non-Enzyme quadrature path

  ContactEvaluator evaluator( params );

  // ── Run validation ───────────────────────────────────────────────────────
  const double epsilon = 1e-7;

  RealT y1_plus[2] = { epsilon, 0.0 };
  RealT y1_minus[2] = { -epsilon, 0.0 };
  mesh1.setPosition( x1, y1_plus, nullptr );
  auto [gp1, gp2] = evaluator.eval_gtilde( pair, mesh1.getView(), mesh2.getView() );
  mesh1.setPosition( x1, y1_minus, nullptr );
  auto [gm1, gm2] = evaluator.eval_gtilde( pair, mesh1.getView(), mesh2.getView() );
  mesh1.setPosition( x1, y1, nullptr );
  std::cout << "Manual FD DOF1: " << ( gp1 - gm1 ) / ( 2 * epsilon ) << ", " << ( gp2 - gm2 ) / ( 2 * epsilon ) << "\n";
  auto result = evaluator.validate_g_tilde( pair, mesh1, mesh2, epsilon );

  // ── Compare ──────────────────────────────────────────────────────────────
  const double tol = 1e-6;
  const std::size_t num_dofs = result.node_ids.size() * 2;

  ASSERT_EQ( result.fd_gradient_g1.size(), result.analytical_gradient_g1.size() );
  ASSERT_EQ( result.fd_gradient_g2.size(), result.analytical_gradient_g2.size() );

  for ( size_t i = 0; i < num_dofs; ++i ) {
    EXPECT_NEAR( result.fd_gradient_g1[i], result.analytical_gradient_g1[i], tol )
        << "gtilde1 mismatch at DOF [" << i << "]"
        << "  node=" << result.node_ids[i / 2] << "  dir=" << ( i % 2 == 0 ? "x" : "y" )
        << "  FD=" << result.fd_gradient_g1[i] << "  AD=" << result.analytical_gradient_g1[i];

    EXPECT_NEAR( result.fd_gradient_g2[i], result.analytical_gradient_g2[i], tol )
        << "gtilde2 mismatch at DOF [" << i << "]"
        << "  node=" << result.node_ids[i / 2] << "  dir=" << ( i % 2 == 0 ? "x" : "y" )
        << "  FD=" << result.fd_gradient_g2[i] << "  AD=" << result.analytical_gradient_g2[i];
  }
}

TEST( HessianCheck, GtildeFDvsAD )
{
  // ── Geometry: two facing LINEAR_EDGE segments ────────────────────────────
  // Segment A: (0,0) -> (1,0)
  // Segment B: (0.2, 0.5) -> (0.8, 0.5)

  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.5, 0.5 };

  IndexT conn1[2] = { 1, 0 };
  IndexT conn2[2] = { 0, 1 };

  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  InterfacePair pair( 0, 0 );

  // ── Evaluator setup ──────────────────────────────────────────────────────
  ContactParams params;
  params.del = 0.1;
  params.k = 1.0;
  params.N = 4;
  params.enzyme_quadrature = true;

  ContactEvaluator evaluator( params );

  // ── Run validation ───────────────────────────────────────────────────────
  const double epsilon = 1e-5;  
  auto result = evaluator.validate_hessian( pair, mesh1, mesh2, epsilon );

  // ── Compare ──────────────────────────────────────────────────────────────
  const double tol = 1e-4; 
  const int ndof = 8;

  ASSERT_EQ( result.fd_gradient_g1.size(), 64u );
  ASSERT_EQ( result.fd_gradient_g2.size(), 64u );
  ASSERT_EQ( result.analytical_gradient_g1.size(), 64u );
  ASSERT_EQ( result.analytical_gradient_g2.size(), 64u );

  for ( size_t row = 0; row < ndof; ++row ) {
    for ( size_t col = 0; col < ndof; ++col ) {
      size_t idx = row * ndof + col;

      EXPECT_NEAR( result.fd_gradient_g1[idx], result.analytical_gradient_g1[idx], tol )
          << "Hessian g1 mismatch at [row=" << row << ", col=" << col << "]"
          << "  FD=" << result.fd_gradient_g1[idx] << "  AD=" << result.analytical_gradient_g1[idx];

      EXPECT_NEAR( result.fd_gradient_g2[idx], result.analytical_gradient_g2[idx], tol )
          << "Hessian g2 mismatch at [row=" << row << ", col=" << col << "]"
          << "  FD=" << result.fd_gradient_g2[idx] << "  AD=" << result.analytical_gradient_g2[idx];
    }
  } 
}

}  // namespace tribol