// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <cmath>
#include <vector>
#include <array>
#include "tribol/physics/EnergyMortar.hpp"
#include <gtest/gtest.h>

#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"

#ifdef TRIBOL_USE_ENZYME

namespace tribol {

TEST( GradientCheck, GtildeAndAreaFDvsAD )
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

  ContactParams params;
  params.del = 0.1;
  params.k = 1.0;
  params.N = 3;
  params.enzyme_quadrature = true;

  EnergyMortarCalculator evaluator( params );

  const double epsilon = 1e-7;

  double g_tilde[2], area[2];
  double dgt1_dx[8] = { 0.0 }, dgt2_dx[8] = { 0.0 };
  double dA1_dx[8] = { 0.0 }, dA2_dx[8] = { 0.0 };
  evaluator.computeContactQuantities( pair, mesh1.getView(), mesh2.getView(), g_tilde, area, dgt1_dx, dgt2_dx, dA1_dx,
                                      dA2_dx );

  auto eval_quantities = [&]( const MeshData::Viewer& v1, const MeshData::Viewer& v2 ) {
    double gt[2], a[2], dgt1[8], dgt2[8], da1[8], da2[8];
    evaluator.computeContactQuantities( pair, v1, v2, gt, a, dgt1, dgt2, da1, da2 );
    return std::array<double, 4>{ gt[0], gt[1], a[0], a[1] };
  };

  std::vector<double> fd_gradient_g1( 8, 0.0 );
  std::vector<double> fd_gradient_g2( 8, 0.0 );
  std::vector<double> fd_gradient_A1( 8, 0.0 );
  std::vector<double> fd_gradient_A2( 8, 0.0 );

  int A_conn[2] = { static_cast<int>( conn1[0] ), static_cast<int>( conn1[1] ) };
  int B_conn[2] = { static_cast<int>( conn2[0] ), static_cast<int>( conn2[1] ) };

  size_t dof_idx = 0;

  auto x1_orig = std::vector<RealT>( x1, x1 + 2 );
  auto y1_orig = std::vector<RealT>( y1, y1 + 2 );
  auto x2_orig = std::vector<RealT>( x2, x2 + 2 );
  auto y2_orig = std::vector<RealT>( y2, y2 + 2 );

  // A nodes
  for ( int k = 0; k < 2; ++k ) {
    const int local_node = A_conn[k];

    // x perturbation
    {
      auto x_pert = x1_orig;
      x_pert[static_cast<std::size_t>( local_node )] += epsilon;
      mesh1.setPosition( x_pert.data(), y1_orig.data(), nullptr );
      auto qp = eval_quantities( mesh1.getView(), mesh2.getView() );

      x_pert[static_cast<std::size_t>( local_node )] = x1_orig[static_cast<std::size_t>( local_node )] - epsilon;
      mesh1.setPosition( x_pert.data(), y1_orig.data(), nullptr );
      auto qm = eval_quantities( mesh1.getView(), mesh2.getView() );

      mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
      fd_gradient_g1[dof_idx] = ( qp[0] - qm[0] ) / ( 2.0 * epsilon );
      fd_gradient_g2[dof_idx] = ( qp[1] - qm[1] ) / ( 2.0 * epsilon );
      fd_gradient_A1[dof_idx] = ( qp[2] - qm[2] ) / ( 2.0 * epsilon );
      fd_gradient_A2[dof_idx] = ( qp[3] - qm[3] ) / ( 2.0 * epsilon );
      dof_idx++;
    }

    // y perturbation
    {
      auto y_pert = y1_orig;
      y_pert[static_cast<std::size_t>( local_node )] += epsilon;
      mesh1.setPosition( x1_orig.data(), y_pert.data(), nullptr );
      auto qp = eval_quantities( mesh1.getView(), mesh2.getView() );

      y_pert[static_cast<std::size_t>( local_node )] = y1_orig[static_cast<std::size_t>( local_node )] - epsilon;
      mesh1.setPosition( x1_orig.data(), y_pert.data(), nullptr );
      auto qm = eval_quantities( mesh1.getView(), mesh2.getView() );

      mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
      fd_gradient_g1[dof_idx] = ( qp[0] - qm[0] ) / ( 2.0 * epsilon );
      fd_gradient_g2[dof_idx] = ( qp[1] - qm[1] ) / ( 2.0 * epsilon );
      fd_gradient_A1[dof_idx] = ( qp[2] - qm[2] ) / ( 2.0 * epsilon );
      fd_gradient_A2[dof_idx] = ( qp[3] - qm[3] ) / ( 2.0 * epsilon );
      dof_idx++;
    }
  }

  // B nodes
  for ( int k = 0; k < 2; ++k ) {
    const int local_node = B_conn[k];

    // x perturbation
    {
      auto x_pert = x2_orig;
      x_pert[static_cast<std::size_t>( local_node )] += epsilon;
      mesh2.setPosition( x_pert.data(), y2_orig.data(), nullptr );
      auto qp = eval_quantities( mesh1.getView(), mesh2.getView() );

      x_pert[static_cast<std::size_t>( local_node )] = x2_orig[static_cast<std::size_t>( local_node )] - epsilon;
      mesh2.setPosition( x_pert.data(), y2_orig.data(), nullptr );
      auto qm = eval_quantities( mesh1.getView(), mesh2.getView() );

      mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );
      fd_gradient_g1[dof_idx] = ( qp[0] - qm[0] ) / ( 2.0 * epsilon );
      fd_gradient_g2[dof_idx] = ( qp[1] - qm[1] ) / ( 2.0 * epsilon );
      fd_gradient_A1[dof_idx] = ( qp[2] - qm[2] ) / ( 2.0 * epsilon );
      fd_gradient_A2[dof_idx] = ( qp[3] - qm[3] ) / ( 2.0 * epsilon );
      dof_idx++;
    }

    // y perturbation
    {
      auto y_pert = y2_orig;
      y_pert[static_cast<std::size_t>( local_node )] += epsilon;
      mesh2.setPosition( x2_orig.data(), y_pert.data(), nullptr );
      auto qp = eval_quantities( mesh1.getView(), mesh2.getView() );

      y_pert[static_cast<std::size_t>( local_node )] = y2_orig[static_cast<std::size_t>( local_node )] - epsilon;
      mesh2.setPosition( x2_orig.data(), y_pert.data(), nullptr );
      auto qm = eval_quantities( mesh1.getView(), mesh2.getView() );

      mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );
      fd_gradient_g1[dof_idx] = ( qp[0] - qm[0] ) / ( 2.0 * epsilon );
      fd_gradient_g2[dof_idx] = ( qp[1] - qm[1] ) / ( 2.0 * epsilon );
      fd_gradient_A1[dof_idx] = ( qp[2] - qm[2] ) / ( 2.0 * epsilon );
      fd_gradient_A2[dof_idx] = ( qp[3] - qm[3] ) / ( 2.0 * epsilon );
      dof_idx++;
    }
  }

  const double tol = 1e-6;
  for ( size_t i = 0; i < 8; ++i ) {
    EXPECT_NEAR( fd_gradient_g1[i], dgt1_dx[i], tol ) << "gtilde1 mismatch at DOF [" << i << "]";
    EXPECT_NEAR( fd_gradient_g2[i], dgt2_dx[i], tol ) << "gtilde2 mismatch at DOF [" << i << "]";
    EXPECT_NEAR( fd_gradient_A1[i], dA1_dx[i], tol ) << "area1 mismatch at DOF [" << i << "]";
    EXPECT_NEAR( fd_gradient_A2[i], dA2_dx[i], tol ) << "area2 mismatch at DOF [" << i << "]";
  }
}

TEST( HessianCheck, GtildeAndAreaFDvsAD )
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

  ContactParams params;
  params.del = 0.1;
  params.k = 1.0;
  params.N = 3;
  params.enzyme_quadrature = true;

  EnergyMortarCalculator evaluator( params );

  const double epsilon = 1e-5;

  double hess1[64] = { 0.0 };
  double hess2[64] = { 0.0 };
  double hessA1[64] = { 0.0 };
  double hessA2[64] = { 0.0 };

  evaluator.computeSecondDerivatives( pair, mesh1.getView(), mesh2.getView(), hess1, hess2, hessA1, hessA2 );

  auto x1_orig = std::vector<RealT>( x1, x1 + 2 );
  auto y1_orig = std::vector<RealT>( y1, y1 + 2 );
  auto x2_orig = std::vector<RealT>( x2, x2 + 2 );
  auto y2_orig = std::vector<RealT>( y2, y2 + 2 );

  int A_conn[2] = { static_cast<int>( conn1[0] ), static_cast<int>( conn1[1] ) };
  int B_conn[2] = { static_cast<int>( conn2[0] ), static_cast<int>( conn2[1] ) };

  auto eval_from_offsets = [&]( const std::array<double, 8>& du ) {
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
    evaluator.computeContactQuantities( pair, mesh1.getView(), mesh2.getView(), gt, a, dgt1, dgt2, da1, da2 );
    return std::array<double, 4>{ gt[0], gt[1], a[0], a[1] };
  };

  const std::array<double, 8> zero = { 0., 0., 0., 0., 0., 0., 0., 0. };
  const auto base = eval_from_offsets( zero );

  std::vector<double> fd_gradient_g1( 64, 0.0 );
  std::vector<double> fd_gradient_g2( 64, 0.0 );
  std::vector<double> fd_gradient_A1( 64, 0.0 );
  std::vector<double> fd_gradient_A2( 64, 0.0 );

  const int ndof = 8;
  for ( size_t i = 0; i < ndof; ++i ) {
    for ( size_t j = 0; j < ndof; ++j ) {
      const std::size_t idx = static_cast<std::size_t>( i * ndof + j );

      if ( i == j ) {
        std::array<double, 8> up = zero;
        std::array<double, 8> um = zero;
        up[i] += epsilon;
        um[i] -= epsilon;

        const auto qp = eval_from_offsets( up );
        const auto qm = eval_from_offsets( um );

        fd_gradient_g1[idx] = ( qp[0] - 2.0 * base[0] + qm[0] ) / ( epsilon * epsilon );
        fd_gradient_g2[idx] = ( qp[1] - 2.0 * base[1] + qm[1] ) / ( epsilon * epsilon );
        fd_gradient_A1[idx] = ( qp[2] - 2.0 * base[2] + qm[2] ) / ( epsilon * epsilon );
        fd_gradient_A2[idx] = ( qp[3] - 2.0 * base[3] + qm[3] ) / ( epsilon * epsilon );
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

        const auto qpp = eval_from_offsets( upp );
        const auto qpm = eval_from_offsets( upm );
        const auto qmp = eval_from_offsets( ump );
        const auto qmm = eval_from_offsets( umm );

        fd_gradient_g1[idx] = ( qpp[0] - qpm[0] - qmp[0] + qmm[0] ) / ( 4.0 * epsilon * epsilon );
        fd_gradient_g2[idx] = ( qpp[1] - qpm[1] - qmp[1] + qmm[1] ) / ( 4.0 * epsilon * epsilon );
        fd_gradient_A1[idx] = ( qpp[2] - qpm[2] - qmp[2] + qmm[2] ) / ( 4.0 * epsilon * epsilon );
        fd_gradient_A2[idx] = ( qpp[3] - qpm[3] - qmp[3] + qmm[3] ) / ( 4.0 * epsilon * epsilon );
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

      EXPECT_NEAR( fd_gradient_A1[idx], hessA1[idx], tol )
          << "Hessian A1 mismatch at [row=" << row << ", col=" << col << "]";

      EXPECT_NEAR( fd_gradient_A2[idx], hessA2[idx], tol )
          << "Hessian A2 mismatch at [row=" << row << ", col=" << col << "]";
    }
  }
}
}  // namespace tribol

#endif  // TRIBOL_USE_ENZYME
