// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <cmath>
#include <set>
#include "tribol/physics/EnergyMortar.hpp"
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

TEST( GradientCheck, GtildeFDvsAD )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };  
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.5, 0.5 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  InterfacePair pair( 0, 0 );

  ContactParams params_;
  params_.del = 0.1;                 
  params_.k = 1.0;                   
  params_.N = 3;                     
  params_.enzyme_quadrature = true;  

  EnergyMortarCalculator evaluator_( params_ );

  const double epsilon = 1e-7;

  double g_tilde[2], area[2];
  double dgt1_dx[8] = { 0.0 }, dgt2_dx[8] = { 0.0 };
  double dA1_dx[8] = { 0.0 }, dA2_dx[8] = { 0.0 };
  evaluator_.computeContactQuantities( pair, mesh1.getView(), mesh2.getView(), g_tilde, area, dgt1_dx, dgt2_dx, dA1_dx, dA2_dx );

  auto eval_gtilde = [&]( const MeshData::Viewer& v1, const MeshData::Viewer& v2 ) -> std::pair<double, double> {
    double gt[2], a[2], dgt1[8], dgt2[8], da1[8], da2[8];
    evaluator_.computeContactQuantities( pair, v1, v2, gt, a, dgt1, dgt2, da1, da2 );
    return { gt[0], gt[1] };
  };

  std::vector<double> fd_gradient_g1( 8, 0.0 );
  std::vector<double> fd_gradient_g2( 8, 0.0 );

  int A_conn[2] = { static_cast<int>(conn1[0]), static_cast<int>(conn1[1]) };
  int B_conn[2] = { static_cast<int>(conn2[0]), static_cast<int>(conn2[1]) };

  size_t dof_idx = 0;

  auto x1_orig = std::vector<RealT>( x1, x1 + 2 );
  auto y1_orig = std::vector<RealT>( y1, y1 + 2 );
  auto x2_orig = std::vector<RealT>( x2, x2 + 2 );
  auto y2_orig = std::vector<RealT>( y2, y2 + 2 );

  // A nodes 
  for ( int k = 0; k < 2; ++k ) {
    const int local_node = A_conn[k];

    // x pert
    {
      auto x_pert = x1_orig;
      x_pert[static_cast<std::size_t>( local_node )] += epsilon;
      mesh1.setPosition( x_pert.data(), y1_orig.data(), nullptr );
      auto [g1p, g2p] = eval_gtilde( mesh1.getView(), mesh2.getView() );

      x_pert[static_cast<std::size_t>( local_node )] = x1_orig[static_cast<std::size_t>( local_node )] - epsilon;
      mesh1.setPosition( x_pert.data(), y1_orig.data(), nullptr );
      auto [g1m, g2m] = eval_gtilde( mesh1.getView(), mesh2.getView() );

      mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
      fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
      fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
      dof_idx++;
    }

    // y pert
    {
      auto y_pert = y1_orig;
      y_pert[static_cast<std::size_t>( local_node )] += epsilon;
      mesh1.setPosition( x1_orig.data(), y_pert.data(), nullptr );
      auto [g1p, g2p] = eval_gtilde( mesh1.getView(), mesh2.getView() );

      y_pert[static_cast<std::size_t>( local_node )] = y1_orig[static_cast<std::size_t>( local_node )] - epsilon;
      mesh1.setPosition( x1_orig.data(), y_pert.data(), nullptr );
      auto [g1m, g2m] = eval_gtilde( mesh1.getView(), mesh2.getView() );

      mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
      fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
      fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
      dof_idx++;
    }
  }

  // B nodes
  for ( int k = 0; k < 2; ++k ) {
    const int local_node = B_conn[k];

    // x pert
    {
      auto x_pert = x2_orig;
      x_pert[static_cast<std::size_t>( local_node )] += epsilon;
      mesh2.setPosition( x_pert.data(), y2_orig.data(), nullptr );
      auto [g1p, g2p] = eval_gtilde( mesh1.getView(), mesh2.getView() );

      x_pert[static_cast<std::size_t>( local_node )] = x2_orig[static_cast<std::size_t>( local_node )] - epsilon;
      mesh2.setPosition( x_pert.data(), y2_orig.data(), nullptr );
      auto [g1m, g2m] = eval_gtilde( mesh1.getView(), mesh2.getView() );

      mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );
      fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
      fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
      dof_idx++;
    }

    // y pert
    {
      auto y_pert = y2_orig;
      y_pert[static_cast<std::size_t>( local_node )] += epsilon;
      mesh2.setPosition( x2_orig.data(), y_pert.data(), nullptr );
      auto [g1p, g2p] = eval_gtilde( mesh1.getView(), mesh2.getView() );

      y_pert[static_cast<std::size_t>( local_node )] = y2_orig[static_cast<std::size_t>( local_node )] - epsilon;
      mesh2.setPosition( x2_orig.data(), y_pert.data(), nullptr );
      auto [g1m, g2m] = eval_gtilde( mesh1.getView(), mesh2.getView() );

      mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );
      fd_gradient_g1[dof_idx] = ( g1p - g1m ) / ( 2.0 * epsilon );
      fd_gradient_g2[dof_idx] = ( g2p - g2m ) / ( 2.0 * epsilon );
      dof_idx++;
    }
  }

  const double tol = 1e-6;
  for ( size_t i = 0; i < 8; ++i ) {
    EXPECT_NEAR( fd_gradient_g1[i], dgt1_dx[i], tol ) << "gtilde1 mismatch at DOF [" << i << "]";
    EXPECT_NEAR( fd_gradient_g2[i], dgt2_dx[i], tol ) << "gtilde2 mismatch at DOF [" << i << "]";
  }
}

TEST( HessianCheck, GtildeFDvsAD )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.5, 0.5 };

  IndexT conn1[2] = { 1, 0 };
  IndexT conn2[2] = { 0, 1 };

  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  InterfacePair pair( 0, 0 );

  ContactParams params_;
  params_.del = 0.1;
  params_.k = 1.0;
  params_.N = 3;
  params_.enzyme_quadrature = true;

  EnergyMortarCalculator evaluator_( params_ );

  const double epsilon = 1e-5;
  
  double hess1[64] = { 0.0 };
  double hess2[64] = { 0.0 };
  double hessA1[64] = { 0.0 };
  double hessA2[64] = { 0.0 };
  
  evaluator_.computeSecondDerivatives( pair, mesh1.getView(), mesh2.getView(), hess1, hess2, hessA1, hessA2 );
  
  auto x1_orig = std::vector<RealT>( x1, x1 + 2 );
  auto y1_orig = std::vector<RealT>( y1, y1 + 2 );
  auto x2_orig = std::vector<RealT>( x2, x2 + 2 );
  auto y2_orig = std::vector<RealT>( y2, y2 + 2 );

  int A_conn[2] = { static_cast<int>(conn1[0]), static_cast<int>(conn1[1]) };
  int B_conn[2] = { static_cast<int>(conn2[0]), static_cast<int>(conn2[1]) };

  auto eval_from_offsets = [&]( const std::array<double, 8>& du ) -> std::pair<double, double> {
    auto x1_t = x1_orig;
    auto y1_t = y1_orig;
    auto x2_t = x2_orig;
    auto y2_t = y2_orig;

    x1_t[static_cast<std::size_t>( A_conn[0] )] += du[0];
    y1_t[static_cast<std::size_t>( A_conn[0] )] += du[1];
    x1_t[static_cast<std::size_t>( A_conn[1] )] += du[2];
    y1_t[static_cast<std::size_t>( A_conn[1] )] += du[3];

    x2_t[static_cast<std::size_t>( B_conn[0] )] += du[4];
    y2_t[static_cast<std::size_t>( B_conn[0] )] += du[5];
    x2_t[static_cast<std::size_t>( B_conn[1] )] += du[6];
    y2_t[static_cast<std::size_t>( B_conn[1] )] += du[7];

    mesh1.setPosition( x1_t.data(), y1_t.data(), nullptr );
    mesh2.setPosition( x2_t.data(), y2_t.data(), nullptr );

    double gt[2], a[2], dgt1[8], dgt2[8], da1[8], da2[8];
    evaluator_.computeContactQuantities( pair, mesh1.getView(), mesh2.getView(), gt, a, dgt1, dgt2, da1, da2 );
    return { gt[0], gt[1] };
  };

  const std::array<double, 8> zero = { 0., 0., 0., 0., 0., 0., 0., 0. };
  const auto [g10, g20] = eval_from_offsets( zero );
  
  std::vector<double> fd_gradient_g1(64, 0.0);
  std::vector<double> fd_gradient_g2(64, 0.0);
  
  const int ndof = 8;
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

        fd_gradient_g1[idx] = ( g1p - 2.0 * g10 + g1m ) / ( epsilon * epsilon );
        fd_gradient_g2[idx] = ( g2p - 2.0 * g20 + g2m ) / ( epsilon * epsilon );
      } else {
        std::array<double, 8> upp = zero;
        std::array<double, 8> upm = zero;
        std::array<double, 8> ump = zero;
        std::array<double, 8> umm = zero;

        upp[i] += epsilon;
        upp[j] += epsilon;
        upm[i] += epsilon;
        upm[j] -= epsilon;
        ump[i] -= epsilon;
        ump[j] += epsilon;
        umm[i] -= epsilon;
        umm[j] -= epsilon;

        const auto [g1pp, g2pp] = eval_from_offsets( upp );
        const auto [g1pm, g2pm] = eval_from_offsets( upm );
        const auto [g1mp, g2mp] = eval_from_offsets( ump );
        const auto [g1mm, g2mm] = eval_from_offsets( umm );

        fd_gradient_g1[idx] = ( g1pp - g1pm - g1mp + g1mm ) / ( 4.0 * epsilon * epsilon );
        fd_gradient_g2[idx] = ( g2pp - g2pm - g2mp + g2mm ) / ( 4.0 * epsilon * epsilon );
      }
    }
  }

  mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
  mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );

  const double tol = 1e-4;

  for ( size_t row = 0; row < ndof; ++row ) {
    for ( size_t col = 0; col < ndof; ++col ) {
      size_t idx = row * ndof + col;

      EXPECT_NEAR( fd_gradient_g1[idx], hess1[idx], tol )
          << "Hessian g1 mismatch at [row=" << row << ", col=" << col << "]";

      EXPECT_NEAR( fd_gradient_g2[idx], hess2[idx], tol )
          << "Hessian g2 mismatch at [row=" << row << ", col=" << col << "]";
    }
  }
}

}  // namespace tribol