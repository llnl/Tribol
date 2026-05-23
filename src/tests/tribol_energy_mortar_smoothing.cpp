// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <cmath>
#include <array>
#include <vector>
#include <limits>

#include <gtest/gtest.h>

#include "tribol/config.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/physics/EnergyMortar.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

#include "axom/slic.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

namespace {

enum class KernelOutputForTest
{
  GTILDE1,
  GTILDE2
};

template <KernelOutputForTest Output>
void varying_quadrature_kernel_output( const double* x, double* out )
{
  double g_tilde[2];
  double area[2];
  energy_mortar_varying_quadrature_kernel( x, 0.1, 3, SmoothingType::Hermite, g_tilde, area );
  *out = Output == KernelOutputForTest::GTILDE1 ? g_tilde[0] : g_tilde[1];
}

template <KernelOutputForTest Output>
void fixed_quadrature_kernel_output( const double* x, const Gparams* gp, double* out )
{
  double g_tilde[2];
  double area[2];
  energy_mortar_fixed_quadrature_kernel( x, *gp, g_tilde, area );
  *out = Output == KernelOutputForTest::GTILDE1 ? g_tilde[0] : g_tilde[1];
}

template <KernelOutputForTest Output>
void varying_quadrature_kernel_gradient( const double* x, double* grad )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;
  __enzyme_autodiff<void>( (void*)varying_quadrature_kernel_output<Output>, enzyme_dup, x, dx, enzyme_dup, &out,
                           &dout );
  for ( int i = 0; i < 8; ++i ) {
    grad[i] = dx[i];
  }
}

template <KernelOutputForTest Output>
void fixed_quadrature_kernel_gradient( const double* x, const Gparams* gp, double* grad )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;
  __enzyme_autodiff<void>( (void*)fixed_quadrature_kernel_output<Output>, enzyme_dup, x, dx, enzyme_const,
                           (const void*)gp, enzyme_dup, &out, &dout );
  for ( int i = 0; i < 8; ++i ) {
    grad[i] = dx[i];
  }
}

template <KernelOutputForTest Output>
void compute_varying_quadrature_hessian( const double x[8], double H[64] )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;
    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };
    __enzyme_fwddiff<void>( (void*)varying_quadrature_kernel_gradient<Output>, enzyme_dup, x, dx, enzyme_dup, grad,
                            dgrad );
    for ( int row = 0; row < 8; ++row ) {
      H[row * 8 + col] = dgrad[row];
    }
  }
}

template <KernelOutputForTest Output>
void compute_fixed_quadrature_hessian( const double x[8], const Gparams& gp, double H[64] )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;
    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };
    __enzyme_fwddiff<void>( (void*)fixed_quadrature_kernel_gradient<Output>, enzyme_dup, x, dx, enzyme_const,
                            (const void*)&gp, enzyme_dup, grad, dgrad );
    for ( int row = 0; row < 8; ++row ) {
      H[row * 8 + col] = dgrad[row];
    }
  }
}

std::pair<double, double> hessian_norms( const double H[64] )
{
  double norm_sq = 0.0;
  double skew_sq = 0.0;
  for ( int i = 0; i < 8; ++i ) {
    for ( int j = 0; j < 8; ++j ) {
      const double value = H[i * 8 + j];
      norm_sq += value * value;
      const double skew = 0.5 * ( value - H[j * 8 + i] );
      skew_sq += skew * skew;
    }
  }
  return { std::sqrt( norm_sq ), std::sqrt( skew_sq ) };
}

}  // namespace

// ============================================================================
// smooth_bounds tests
// ============================================================================

class SmoothBoundsTest : public testing::TestWithParam<SmoothingType> {};

TEST_P( SmoothBoundsTest, OutputInRange )
{
  const auto type = GetParam();
  const std::vector<double> del_values = { 0.01, 0.05, 0.1, 0.2, 0.4 };

  for ( double del : del_values ) {
    for ( int k = -20; k <= 20; ++k ) {
      double xi = k * 0.1;  // spans [-2, 2]
      std::array<double, 2> bounds = { xi, xi };
      auto result = ContactSmoothing::smooth_bounds( bounds, del, type );
      EXPECT_GE( result[0], -0.5 )
          << "del=" << del << " xi=" << xi << " type=" << static_cast<int>( type );
      EXPECT_LE( result[0], 0.5 )
          << "del=" << del << " xi=" << xi << " type=" << static_cast<int>( type );
    }
  }
}

TEST_P( SmoothBoundsTest, Monotonicity )
{
  const auto type = GetParam();
  const std::vector<double> del_values = { 0.01, 0.05, 0.1, 0.2 };

  for ( double del : del_values ) {
    double prev = -1.0;
    for ( int k = 0; k <= 1000; ++k ) {
      double xi = -0.5 + k * ( 1.0 / 1000.0 );
      std::array<double, 2> bounds = { xi, xi };
      auto result = ContactSmoothing::smooth_bounds( bounds, del, type );
      EXPECT_GE( result[0], prev - 1e-15 )
          << "Non-monotone at del=" << del << " xi=" << xi << " type=" << static_cast<int>( type );
      prev = result[0];
    }
  }
}

TEST_P( SmoothBoundsTest, EndpointValues )
{
  const auto type = GetParam();
  const double del = 0.1;
  const double tol = 1e-14;

  // f(0) = 0 in shifted coords => f(-0.5) = -0.5 in unshifted
  std::array<double, 2> bounds_min = { -0.5, -0.5 };
  auto result_min = ContactSmoothing::smooth_bounds( bounds_min, del, type );
  EXPECT_NEAR( result_min[0], -0.5, tol );

  // f(1) = 1 in shifted coords => f(0.5) = 0.5 in unshifted
  std::array<double, 2> bounds_max = { 0.5, 0.5 };
  auto result_max = ContactSmoothing::smooth_bounds( bounds_max, del, type );
  EXPECT_NEAR( result_max[0], 0.5, tol );
}

TEST_P( SmoothBoundsTest, ProjectionBoundsClampToElement )
{
  const double del = 0.1;

  auto left_separated =
      ContactSmoothing::bounds_from_projections( { -0.8, -0.7 }, del );
  EXPECT_NEAR( left_separated[0], -0.5, 1e-14 );
  EXPECT_NEAR( left_separated[1], -0.5, 1e-14 );

  auto right_separated =
      ContactSmoothing::bounds_from_projections( { 0.7, 0.8 }, del );
  EXPECT_NEAR( right_separated[0], 0.5, 1e-14 );
  EXPECT_NEAR( right_separated[1], 0.5, 1e-14 );

  auto crossing =
      ContactSmoothing::bounds_from_projections( { -0.7, 0.2 }, del );
  EXPECT_NEAR( crossing[0], -0.5, 1e-14 );
  EXPECT_NEAR( crossing[1], 0.2, 1e-14 );
}

TEST_P( SmoothBoundsTest, C1ContinuityAtJunctions )
{
  const auto type = GetParam();
  const std::vector<double> del_values = { 0.05, 0.1, 0.2 };
  const double h = 1e-7;

  for ( double del : del_values ) {
    // Check derivative continuity at xi = -0.5 + del (left junction)
    // and xi = 0.5 - del (right junction) in unshifted coords
    std::vector<double> junctions = { -0.5 + del, 0.5 - del };

    for ( double xi_j : junctions ) {
      std::array<double, 2> bp = { xi_j + h, xi_j + h };
      std::array<double, 2> bm = { xi_j - h, xi_j - h };
      auto rp = ContactSmoothing::smooth_bounds( bp, del, type );
      auto rm = ContactSmoothing::smooth_bounds( bm, del, type );

      double deriv_fwd = ( rp[0] - ContactSmoothing::smooth_bounds( { xi_j, xi_j }, del, type )[0] ) / h;
      double deriv_bwd =
          ( ContactSmoothing::smooth_bounds( { xi_j, xi_j }, del, type )[0] - rm[0] ) / h;

      EXPECT_NEAR( deriv_fwd, deriv_bwd, 1e-4 )
          << "C1 break at xi=" << xi_j << " del=" << del << " type=" << static_cast<int>( type )
          << " fwd=" << deriv_fwd << " bwd=" << deriv_bwd;
    }
  }
}

TEST_P( SmoothBoundsTest, ZeroDerivativeAtEndpoints )
{
  const auto type = GetParam();
  const double del = 0.1;
  const double h = 1e-8;

  // f'(0) = 0 in shifted coords => f'(-0.5) = 0 in unshifted
  std::array<double, 2> b0 = { -0.5, -0.5 };
  std::array<double, 2> b0h = { -0.5 + h, -0.5 + h };
  auto r0 = ContactSmoothing::smooth_bounds( b0, del, type );
  auto r0h = ContactSmoothing::smooth_bounds( b0h, del, type );
  double deriv_left = ( r0h[0] - r0[0] ) / h;
  EXPECT_NEAR( deriv_left, 0.0, 1e-5 ) << "f'(0) != 0 for type=" << static_cast<int>( type );

  // f'(1) = 0 in shifted coords => f'(0.5) = 0 in unshifted
  std::array<double, 2> b1 = { 0.5, 0.5 };
  std::array<double, 2> b1m = { 0.5 - h, 0.5 - h };
  auto r1 = ContactSmoothing::smooth_bounds( b1, del, type );
  auto r1m = ContactSmoothing::smooth_bounds( b1m, del, type );
  double deriv_right = ( r1[0] - r1m[0] ) / h;
  EXPECT_NEAR( deriv_right, 0.0, 1e-5 ) << "f'(1) != 0 for type=" << static_cast<int>( type );
}

INSTANTIATE_TEST_SUITE_P( SmoothingTypes, SmoothBoundsTest,
                          testing::Values( SmoothingType::Hermite, SmoothingType::Quadratic ),
                          []( const testing::TestParamInfo<SmoothingType>& info ) {
                            return info.param == SmoothingType::Hermite ? "Hermite" : "Quadratic";
                          } );

// ============================================================================
// penalty ramp tests
// ============================================================================

TEST( PenaltyRampTest, SmoothRampMatchesHardOutsideTransition )
{
  const double del = 0.1;

  auto full_contact = ContactSmoothing::penalty_ramp( -2.0 * del, del, PenaltySmoothing::Smooth );
  EXPECT_NEAR( full_contact.value, -2.0 * del, 1e-14 );
  EXPECT_NEAR( full_contact.first_derivative, 1.0, 1e-14 );
  EXPECT_NEAR( full_contact.second_derivative, 0.0, 1e-14 );

  auto separated = ContactSmoothing::penalty_ramp( del, del, PenaltySmoothing::Smooth );
  EXPECT_NEAR( separated.value, 0.0, 1e-14 );
  EXPECT_NEAR( separated.first_derivative, 0.0, 1e-14 );
  EXPECT_NEAR( separated.second_derivative, 0.0, 1e-14 );
}

TEST( PenaltyRampTest, SmoothRampIsC1AtTransitionEndpoints )
{
  const double del = 0.1;
  const double d2 = 0.5 * del;

  // Left junction: g = -del/2.  H = g, H' = 1.
  auto left = ContactSmoothing::penalty_ramp( -d2, del, PenaltySmoothing::Smooth );
  EXPECT_NEAR( left.value, -d2, 1e-14 );
  EXPECT_NEAR( left.first_derivative, 1.0, 1e-14 );

  // Right junction: g = +del/2.  H = 0, H' = 0.
  auto right = ContactSmoothing::penalty_ramp( d2, del, PenaltySmoothing::Smooth );
  EXPECT_NEAR( right.value, 0.0, 1e-14 );
  EXPECT_NEAR( right.first_derivative, 0.0, 1e-14 );

  // H'' is constant -1/del inside the band.
  auto mid = ContactSmoothing::penalty_ramp( 0.0, del, PenaltySmoothing::Smooth );
  EXPECT_NEAR( mid.second_derivative, -1.0 / del, 1e-14 );
}

TEST( PenaltyRampTest, SmoothRampDerivativesMatchFiniteDifference )
{
  const double del = 0.1;
  const double h = 1e-6;
  for ( double g : { -0.04, -0.02, 0.0, 0.02, 0.04 } ) {
    auto ramp = ContactSmoothing::penalty_ramp( g, del, PenaltySmoothing::Smooth );
    auto plus = ContactSmoothing::penalty_ramp( g + h, del, PenaltySmoothing::Smooth );
    auto minus = ContactSmoothing::penalty_ramp( g - h, del, PenaltySmoothing::Smooth );

    const double fd_first = ( plus.value - minus.value ) / ( 2.0 * h );
    const double fd_second = ( plus.first_derivative - minus.first_derivative ) / ( 2.0 * h );
    EXPECT_NEAR( ramp.first_derivative, fd_first, 1e-8 ) << "g=" << g;
    EXPECT_NEAR( ramp.second_derivative, fd_second, 1e-7 ) << "g=" << g;
  }
}

// ============================================================================
// Local Enzyme kernel tests
// ============================================================================

TEST( VaryingQuadratureKernelTest, FixedQuadratureHessianIsSymmetric )
{
  const double x[8] = { 0.0, 0.0, 1.0, 0.0, 0.14, 0.12, 0.92, 0.18 };
  const auto qp = EnergyMortarCalculator::compute_quadrature( { -0.42, 0.36 }, 3 );
  Gparams gp;
  for ( std::size_t i = 0; i < qp.qp.size(); ++i ) {
    gp.qp[i] = qp.qp[i];
    gp.w[i] = qp.w[i];
  }

  double H[64];
  compute_fixed_quadrature_hessian<KernelOutputForTest::GTILDE1>( x, gp, H );
  auto [norm1, skew1] = hessian_norms( H );
  ASSERT_GT( norm1, 0.0 );
  EXPECT_LT( skew1 / norm1, 1.0e-12 );

  compute_fixed_quadrature_hessian<KernelOutputForTest::GTILDE2>( x, gp, H );
  auto [norm2, skew2] = hessian_norms( H );
  ASSERT_GT( norm2, 0.0 );
  EXPECT_LT( skew2 / norm2, 1.0e-12 );
}

TEST( VaryingQuadratureKernelTest, VaryingQuadratureHessianDriverProducesFiniteValues )
{
  const double x[8] = { 0.0, 0.0, 1.0, 0.0, -0.08, 0.12, 0.26, 0.18 };

  double H[64];
  compute_varying_quadrature_hessian<KernelOutputForTest::GTILDE1>( x, H );
  auto [norm1, skew1] = hessian_norms( H );
  EXPECT_TRUE( std::isfinite( norm1 ) );
  EXPECT_TRUE( std::isfinite( skew1 ) );
  EXPECT_GT( norm1, 0.0 );

  compute_varying_quadrature_hessian<KernelOutputForTest::GTILDE2>( x, H );
  auto [norm2, skew2] = hessian_norms( H );
  EXPECT_TRUE( std::isfinite( norm2 ) );
  EXPECT_TRUE( std::isfinite( skew2 ) );
  EXPECT_GT( norm2, 0.0 );
}

TEST( VaryingQuadratureKernelTest, DISABLED_VaryingQuadratureHessianIsSymmetric )
{
  const double x[8] = { 0.0, 0.0, 1.0, 0.0, -0.08, 0.12, 0.26, 0.18 };

  double H[64];
  compute_varying_quadrature_hessian<KernelOutputForTest::GTILDE1>( x, H );
  auto [norm1, skew1] = hessian_norms( H );
  ASSERT_GT( norm1, 0.0 );
  EXPECT_LT( skew1 / norm1, 1.0e-12 );

  compute_varying_quadrature_hessian<KernelOutputForTest::GTILDE2>( x, H );
  auto [norm2, skew2] = hessian_norms( H );
  ASSERT_GT( norm2, 0.0 );
  EXPECT_LT( skew2 / norm2, 1.0e-12 );
}

// ============================================================================
// Vanishing overlap tests
// ============================================================================

class VanishingOverlapTest : public testing::TestWithParam<SmoothingType> {};

TEST_P( VanishingOverlapTest, GtildeAndAreaScaleTogether )
{
  // Two parallel edges. Edge A: (0,0)->(1,0). Edge B slides rightward
  // from full overlap toward separation. Gap is 0.1 in y.

  const auto type = GetParam();
  ContactParams params;
  params.del = 0.1;
  params.k = 1.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  params.smoothing_type = type;

  EnergyMortarCalculator evaluator( params );

  // Edge A: (0,0) -> (1,0) — nonmortar
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };  // reversed so nA points upward
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  // Edge B: slides rightward
  RealT x2[2] = { 0.0, 0.0 };
  RealT y2[2] = { 0.1, 0.1 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  InterfacePair pair( 0, 0 );

  std::vector<double> offsets = { 0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 1.0 };

  for ( double offset : offsets ) {
    x2[0] = offset;
    x2[1] = offset + 0.2;  // B is always 0.2 wide
    mesh2.setPosition( x2, y2, nullptr );

    double gtilde[2] = { 0.0 };
    double area[2] = { 0.0 };
    evaluator.compute_gtilde_and_area( pair, mesh1.getView(), mesh2.getView(), gtilde, area );

    double total_gtilde = gtilde[0] + gtilde[1];
    double total_area = area[0] + area[1];

    // Area should always be non-negative
    EXPECT_GE( area[0], -1e-15 ) << "Negative area[0] at offset=" << offset;
    EXPECT_GE( area[1], -1e-15 ) << "Negative area[1] at offset=" << offset;
    EXPECT_GE( total_area, -1e-15 ) << "Negative total area at offset=" << offset;

    // When there is meaningful overlap, g_tilde/A should be bounded
    if ( total_area > 1e-10 ) {
      double ratio = total_gtilde / total_area;
      EXPECT_FALSE( std::isnan( ratio ) ) << "NaN ratio at offset=" << offset;
      EXPECT_FALSE( std::isinf( ratio ) ) << "Inf ratio at offset=" << offset;
      EXPECT_LT( std::abs( ratio ), 10.0 ) << "Ratio blew up at offset=" << offset;
    }

    // As overlap vanishes, both g_tilde and A should approach zero
    if ( offset >= 1.0 ) {
      EXPECT_NEAR( total_area, 0.0, 1e-10 ) << "Area not zero when B is past A";
      EXPECT_NEAR( total_gtilde, 0.0, 1e-10 ) << "g_tilde not zero when B is past A";
    }
  }
}

// Disabled until this test uses the same fixed-quadrature path for the finite
// difference and analytic derivative. The current finite difference recomputes
// geometry-dependent quadrature, so it does not isolate grad_gtilde.
TEST_P( VanishingOverlapTest, DISABLED_GradientConsistency )
{
  // Verify that Enzyme gradients of g_tilde match finite differences
  // at various overlap levels.
  const auto type = GetParam();
  ContactParams params;
  params.del = 0.1;
  params.k = 1.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  params.smoothing_type = type;

  EnergyMortarCalculator evaluator( params );

  // Edge A: (0,0) -> (1,0)
  RealT x1_orig[2] = { 0.0, 1.0 };
  RealT y1_orig[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1_orig, y1_orig, nullptr, MemorySpace::Host );

  InterfacePair pair( 0, 0 );

  std::vector<double> left_positions = { 0.0, 0.5, 0.8, 0.95 };

  for ( double left : left_positions ) {
    RealT x2[2] = { left, left + 0.2 };
    RealT y2[2] = { 0.1, 0.1 };
    IndexT conn2[2] = { 0, 1 };
    MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

    // Analytical gradients
    double dgt1_dx[8] = { 0.0 };
    double dgt2_dx[8] = { 0.0 };
    evaluator.grad_gtilde( pair, mesh1.getView(), mesh2.getView(), dgt1_dx, dgt2_dx );

    // Finite-difference gradients.
    // grad_gtilde returns derivatives in the order [A0x, A0y, A1x, A1y, B0x, B0y, B1x, B1y]
    // where A0/A1 and B0/B1 follow the connectivity (not the raw array index).
    const double eps = 1e-7;
    double fd_g1[8] = { 0.0 };
    double fd_g2[8] = { 0.0 };

    // Perturb mesh1 nodes in connectivity order (A0 = conn1[0], A1 = conn1[1])
    for ( int ci = 0; ci < 2; ++ci ) {
      int node = conn1[ci];
      for ( int dim = 0; dim < 2; ++dim ) {
        int dof_idx = ci * 2 + dim;
        RealT x1p[2] = { x1_orig[0], x1_orig[1] };
        RealT y1p[2] = { y1_orig[0], y1_orig[1] };
        RealT x1m[2] = { x1_orig[0], x1_orig[1] };
        RealT y1m[2] = { y1_orig[0], y1_orig[1] };

        if ( dim == 0 ) {
          x1p[node] += eps;
          x1m[node] -= eps;
        } else {
          y1p[node] += eps;
          y1m[node] -= eps;
        }

        mesh1.setPosition( x1p, y1p, nullptr );
        double gtp[2], ap[2];
        evaluator.compute_gtilde_and_area( pair, mesh1.getView(), mesh2.getView(), gtp, ap );

        mesh1.setPosition( x1m, y1m, nullptr );
        double gtm[2], am[2];
        evaluator.compute_gtilde_and_area( pair, mesh1.getView(), mesh2.getView(), gtm, am );

        mesh1.setPosition( x1_orig, y1_orig, nullptr );

        fd_g1[dof_idx] = ( gtp[0] - gtm[0] ) / ( 2.0 * eps );
        fd_g2[dof_idx] = ( gtp[1] - gtm[1] ) / ( 2.0 * eps );
      }
    }

    // Perturb mesh2 nodes in connectivity order (B0 = conn2[0], B1 = conn2[1])
    RealT x2_orig[2] = { x2[0], x2[1] };
    RealT y2_orig[2] = { y2[0], y2[1] };
    for ( int ci = 0; ci < 2; ++ci ) {
      int node = conn2[ci];
      for ( int dim = 0; dim < 2; ++dim ) {
        int dof_idx = 4 + ci * 2 + dim;
        RealT x2p[2] = { x2_orig[0], x2_orig[1] };
        RealT y2p[2] = { y2_orig[0], y2_orig[1] };
        RealT x2m[2] = { x2_orig[0], x2_orig[1] };
        RealT y2m[2] = { y2_orig[0], y2_orig[1] };

        if ( dim == 0 ) {
          x2p[node] += eps;
          x2m[node] -= eps;
        } else {
          y2p[node] += eps;
          y2m[node] -= eps;
        }

        mesh2.setPosition( x2p, y2p, nullptr );
        double gtp[2], ap[2];
        evaluator.compute_gtilde_and_area( pair, mesh1.getView(), mesh2.getView(), gtp, ap );

        mesh2.setPosition( x2m, y2m, nullptr );
        double gtm[2], am[2];
        evaluator.compute_gtilde_and_area( pair, mesh1.getView(), mesh2.getView(), gtm, am );

        mesh2.setPosition( x2_orig, y2_orig, nullptr );

        fd_g1[dof_idx] = ( gtp[0] - gtm[0] ) / ( 2.0 * eps );
        fd_g2[dof_idx] = ( gtp[1] - gtm[1] ) / ( 2.0 * eps );
      }
    }

    const double tol = 1e-4;
    for ( int i = 0; i < 8; ++i ) {
      EXPECT_NEAR( fd_g1[i], dgt1_dx[i], tol )
          << "g1 grad mismatch at dof " << i << " left=" << left
          << " type=" << static_cast<int>( type )
          << " FD=" << fd_g1[i] << " AD=" << dgt1_dx[i];

      EXPECT_NEAR( fd_g2[i], dgt2_dx[i], tol )
          << "g2 grad mismatch at dof " << i << " left=" << left
          << " type=" << static_cast<int>( type )
          << " FD=" << fd_g2[i] << " AD=" << dgt2_dx[i];
    }
  }
}

INSTANTIATE_TEST_SUITE_P( SmoothingTypes, VanishingOverlapTest,
                          testing::Values( SmoothingType::Hermite, SmoothingType::Quadratic ),
                          []( const testing::TestParamInfo<SmoothingType>& info ) {
                            return info.param == SmoothingType::Hermite ? "Hermite" : "Quadratic";
                          } );

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
