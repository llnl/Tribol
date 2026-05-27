// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <cmath>
#include <array>
#include <vector>
#include "tribol/physics/EnergyAreaPenalty.hpp"
#include <gtest/gtest.h>

#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

TEST( EnergyAreaPenaltyGradientCheck, EnergyFDvsAD )
{
  // Segment A: (0,0) -> (1,0)
  // Segment B: (0.2, 0.5) -> (0.8, 0.5)
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.5, 0.5 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  InterfacePair pair( 0, 0 );

  EnergyMortarOptions opts;
  opts.del = 0.1;
  opts.k = 1.0;
  opts.N = 3;
  opts.enzyme_quadrature = true;
  opts.penalty_smoothing = PenaltySmoothing::Smooth;
  opts.penalty_smoothing_del = 1.0e-3;

  EnergyAreaPenaltyCalculator calc( opts );

  auto viewer1 = mesh1.getView();
  auto viewer2 = mesh2.getView();

  // Analytical gradient
  double dE_dx_ad[8];
  calc.grad_energy( pair, viewer1, viewer2, dE_dx_ad );

  // Finite difference gradient
  const double eps = 1.0e-7;
  double dE_dx_fd[8];

  auto A_conn = viewer1.getConnectivity()( 0 );
  auto B_conn = viewer2.getConnectivity()( 0 );

  const IndexT n1 = mesh1.numberOfNodes();
  const IndexT n2 = mesh2.numberOfNodes();

  std::vector<RealT> x1_orig( x1, x1 + n1 );
  std::vector<RealT> y1_orig( y1, y1 + n1 );
  std::vector<RealT> x2_orig( x2, x2 + n2 );
  std::vector<RealT> y2_orig( y2, y2 + n2 );

  auto eval = [&]( const std::array<double, 8>& du ) -> double {
    auto xx1 = x1_orig;
    auto yy1 = y1_orig;
    auto xx2 = x2_orig;
    auto yy2 = y2_orig;

    xx1[A_conn[0]] += du[0];
    yy1[A_conn[0]] += du[1];
    xx1[A_conn[1]] += du[2];
    yy1[A_conn[1]] += du[3];
    xx2[B_conn[0]] += du[4];
    yy2[B_conn[0]] += du[5];
    xx2[B_conn[1]] += du[6];
    yy2[B_conn[1]] += du[7];

    mesh1.setPosition( xx1.data(), yy1.data(), nullptr );
    mesh2.setPosition( xx2.data(), yy2.data(), nullptr );

    double E = calc.compute_energy( pair, mesh1.getView(), mesh2.getView() );

    mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
    mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );

    return E;
  };

  for ( int d = 0; d < 8; ++d ) {
    std::array<double, 8> up = {};
    std::array<double, 8> um = {};
    up[d] = eps;
    um[d] = -eps;
    dE_dx_fd[d] = ( eval( up ) - eval( um ) ) / ( 2.0 * eps );
  }

  const double tol = 1.0e-5;
  for ( int d = 0; d < 8; ++d ) {
    EXPECT_NEAR( dE_dx_fd[d], dE_dx_ad[d], tol )
        << "Energy gradient mismatch at DOF " << d << "  FD=" << dE_dx_fd[d] << "  AD=" << dE_dx_ad[d];
  }
}

TEST( EnergyAreaPenaltyHessianCheck, EnergyFDvsAD )
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

  EnergyMortarOptions opts;
  opts.del = 0.1;
  opts.k = 1.0;
  opts.N = 3;
  opts.enzyme_quadrature = true;
  opts.penalty_smoothing = PenaltySmoothing::Smooth;
  opts.penalty_smoothing_del = 1.0e-3;

  EnergyAreaPenaltyCalculator calc( opts );

  auto viewer1 = mesh1.getView();
  auto viewer2 = mesh2.getView();

  // Analytical Hessian
  double d2E_dx2_ad[64];
  calc.hessian_energy( pair, viewer1, viewer2, d2E_dx2_ad );

  // Finite difference Hessian via central differences on the gradient
  const double eps = 1.0e-5;
  double d2E_dx2_fd[64];

  auto A_conn = viewer1.getConnectivity()( 0 );
  auto B_conn = viewer2.getConnectivity()( 0 );

  const IndexT n1 = mesh1.numberOfNodes();
  const IndexT n2 = mesh2.numberOfNodes();

  std::vector<RealT> x1_orig( x1, x1 + n1 );
  std::vector<RealT> y1_orig( y1, y1 + n1 );
  std::vector<RealT> x2_orig( x2, x2 + n2 );
  std::vector<RealT> y2_orig( y2, y2 + n2 );

  auto eval_grad = [&]( const std::array<double, 8>& du, double* grad ) {
    auto xx1 = x1_orig;
    auto yy1 = y1_orig;
    auto xx2 = x2_orig;
    auto yy2 = y2_orig;

    xx1[A_conn[0]] += du[0];
    yy1[A_conn[0]] += du[1];
    xx1[A_conn[1]] += du[2];
    yy1[A_conn[1]] += du[3];
    xx2[B_conn[0]] += du[4];
    yy2[B_conn[0]] += du[5];
    xx2[B_conn[1]] += du[6];
    yy2[B_conn[1]] += du[7];

    mesh1.setPosition( xx1.data(), yy1.data(), nullptr );
    mesh2.setPosition( xx2.data(), yy2.data(), nullptr );

    calc.grad_energy( pair, mesh1.getView(), mesh2.getView(), grad );

    mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
    mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );
  };

  for ( int col = 0; col < 8; ++col ) {
    std::array<double, 8> up = {};
    std::array<double, 8> um = {};
    up[col] = eps;
    um[col] = -eps;

    double grad_p[8], grad_m[8];
    eval_grad( up, grad_p );
    eval_grad( um, grad_m );

    for ( int row = 0; row < 8; ++row ) {
      d2E_dx2_fd[row * 8 + col] = ( grad_p[row] - grad_m[row] ) / ( 2.0 * eps );
    }
  }

  const double tol = 1.0e-4;
  for ( int row = 0; row < 8; ++row ) {
    for ( int col = 0; col < 8; ++col ) {
      int idx = row * 8 + col;
      EXPECT_NEAR( d2E_dx2_fd[idx], d2E_dx2_ad[idx], tol )
          << "Hessian mismatch at [" << row << "," << col << "]"
          << "  FD=" << d2E_dx2_fd[idx] << "  AD=" << d2E_dx2_ad[idx];
    }
  }
}

TEST( EnergyAreaPenaltyGradientCheck, ForceIsDerivativeOfEnergy )
{
  // Verify that force = k * dE/dx matches FD of total energy k * E
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.1, 0.9 };
  RealT y2[2] = { 0.1, 0.1 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  InterfacePair pair( 0, 0 );

  EnergyMortarOptions opts;
  opts.del = 0.1;
  opts.k = 100.0;
  opts.N = 3;
  opts.enzyme_quadrature = true;
  opts.penalty_smoothing = PenaltySmoothing::Smooth;
  opts.penalty_smoothing_del = 1.0e-4;

  EnergyAreaPenaltyCalculator calc( opts );

  auto viewer1 = mesh1.getView();
  auto viewer2 = mesh2.getView();

  // Force = k * dE/dx
  double dE_dx[8];
  calc.grad_energy( pair, viewer1, viewer2, dE_dx );

  double force[8];
  for ( int i = 0; i < 8; ++i ) {
    force[i] = opts.k * dE_dx[i];
  }

  // FD of k * E
  const double eps = 1.0e-7;
  auto A_conn = viewer1.getConnectivity()( 0 );
  auto B_conn = viewer2.getConnectivity()( 0 );

  const IndexT n1 = mesh1.numberOfNodes();
  const IndexT n2 = mesh2.numberOfNodes();
  std::vector<RealT> x1_orig( x1, x1 + n1 );
  std::vector<RealT> y1_orig( y1, y1 + n1 );
  std::vector<RealT> x2_orig( x2, x2 + n2 );
  std::vector<RealT> y2_orig( y2, y2 + n2 );

  auto eval_kE = [&]( const std::array<double, 8>& du ) -> double {
    auto xx1 = x1_orig;
    auto yy1 = y1_orig;
    auto xx2 = x2_orig;
    auto yy2 = y2_orig;

    xx1[A_conn[0]] += du[0];
    yy1[A_conn[0]] += du[1];
    xx1[A_conn[1]] += du[2];
    yy1[A_conn[1]] += du[3];
    xx2[B_conn[0]] += du[4];
    yy2[B_conn[0]] += du[5];
    xx2[B_conn[1]] += du[6];
    yy2[B_conn[1]] += du[7];

    mesh1.setPosition( xx1.data(), yy1.data(), nullptr );
    mesh2.setPosition( xx2.data(), yy2.data(), nullptr );

    double E = opts.k * calc.compute_energy( pair, mesh1.getView(), mesh2.getView() );

    mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
    mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );

    return E;
  };

  double force_fd[8];
  for ( int d = 0; d < 8; ++d ) {
    std::array<double, 8> up = {};
    std::array<double, 8> um = {};
    up[d] = eps;
    um[d] = -eps;
    force_fd[d] = ( eval_kE( up ) - eval_kE( um ) ) / ( 2.0 * eps );
  }

  const double tol = 1.0e-4;
  for ( int d = 0; d < 8; ++d ) {
    EXPECT_NEAR( force_fd[d], force[d], tol )
        << "Force mismatch at DOF " << d << "  FD=" << force_fd[d] << "  AD=" << force[d];
  }
}

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
