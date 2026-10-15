// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <algorithm>
#include <cmath>
#include <numbers>
#include <set>
#include <tuple>
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
#include "tribol/geom/ElementNormal.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/search/InterfacePairFinder.hpp"

namespace tribol {

inline void endpoints( const MeshData::Viewer& mesh, int elem_id, double P0[2], double P1[2] )
{
  double P0_P1[4];
  mesh.getFaceCoords( elem_id, P0_P1 );
  P0[0] = P0_P1[0];
  P0[1] = P0_P1[1];
  P1[0] = P0_P1[2];
  P1[1] = P0_P1[3];
}

BallEndpointData endpointBallData( double neighbor_x, double neighbor_y )
{
  BallEndpointData data;
  data.weight = { 0.0, 1.0 };
  data.neighbor_coordinates = { 0.0, 0.0, neighbor_x, neighbor_y };
  return data;
}

BallEndpointData openEndBallData()
{
  BallEndpointData data;
  data.weight = { 0.0, 1.0 };
  data.is_open_endpoint[1] = true;
  return data;
}

std::pair<double, double> EnergyMortarCalculator::eval_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                               const MeshData::Viewer& mesh2 ) const
{
  double gtilde[2];
  double area[2];
  compute_gtilde_and_area( pair, mesh1, mesh2, gtilde, area );
  return { gtilde[0], gtilde[1] };
}

FiniteDiffResult EnergyMortarCalculator::validate_g_tilde( const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
                                                           double epsilon ) const
{
  FiniteDiffResult result;

  auto viewer1 = mesh1.getView();
  auto viewer2 = mesh2.getView();

  auto projs0 = projections( pair, viewer1, viewer2 );
  double bounds0[2];
  ContactSmoothing::bounds_from_projections( projs0.data(), p_.del, bounds0 );
  double smooth_bounds0[2];
  ContactSmoothing::smooth_bounds( bounds0, p_.del, smooth_bounds0 );
  QuadPoints qp0;
  if ( !p_.enzyme_quadrature ) {
    compute_quadrature( smooth_bounds0, p_.N, &qp0 );
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

std::pair<double, double> EnergyMortarCalculator::eval_gtilde_fixed_qp( const InterfacePair& pair,
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

    const double gn = compute_weighted_normal_gap( pair, mesh1, mesh2, xiA );

    gt1 += w * N1 * J * gn;
    gt2 += w * N2 * J * gn;
  }

  return { gt1, gt2 };
}

FiniteDiffResult EnergyMortarCalculator::validate_hessian( const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
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

  result.node_ids = { static_cast<int>( A_conn[0] ), static_cast<int>( A_conn[1] ), static_cast<int>( B_conn[0] ),
                      static_cast<int>( B_conn[1] ) };

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
    double bounds0[2];
    ContactSmoothing::bounds_from_projections( projs0.data(), p_.del, bounds0 );
    double smooth_bounds0[2];
    ContactSmoothing::smooth_bounds( bounds0, p_.del, smooth_bounds0 );
    compute_quadrature( smooth_bounds0, p_.N, &qp0 );
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

        result.fd_gradient_g1[idx] = ( g1pp - g1pm - g1mp + g1mm ) / ( 4.0 * epsilon * epsilon );
        result.fd_gradient_g2[idx] = ( g2pp - g2pm - g2mp + g2mm ) / ( 4.0 * epsilon * epsilon );
      }
    }
  }

  mesh1.setPosition( x1_orig.data(), y1_orig.data(), nullptr );
  mesh2.setPosition( x2_orig.data(), y2_orig.data(), nullptr );

  return result;
}

// Keep full contact weight up to the configured angle and suppress nonopposing normals.
TEST( NormalAngleSmoothingCheck, QuinticStartsAtConfiguredAngle )
{
  // Without angular attenuation, contact can retain full strength at a steep angle and then drop abruptly to zero.
  constexpr double pi = std::numbers::pi_v<double>;
  constexpr double start_angle = 0.25 * pi;
  const ContactParams default_params{};
  EXPECT_DOUBLE_EQ( default_params.normal_smoothing_start_angle, start_angle );
  EXPECT_DOUBLE_EQ( energy_mortar::default_residual_gap_ramp_angle, pi / 18.0 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( -std::cos( pi / 6.0 ), start_angle ), -1.0 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( -std::cos( start_angle ), start_angle ), -1.0 );
  EXPECT_NEAR( ContactSmoothing::normal_alignment_factor( -std::cos( 3.0 * pi / 8.0 ), start_angle ), -0.5, 1.0e-14 );
  EXPECT_NEAR( ContactSmoothing::normal_alignment_factor( -0.5, 0.0 ), -17.0 / 81.0, 1.0e-14 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( 0.0, start_angle ), 0.0 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( 0.5, start_angle ), 0.0 );
}

// Avoid force or tangent jumps where angular attenuation starts and ends.
TEST( NormalAngleSmoothingCheck, EnergyWeightHasContinuousSlopeAndCurvature )
{
  const double start = tribol::energy_mortar::default_normal_smoothing_start_angle;
  auto weight = [&]( double angle ) {
    const double factor = ContactSmoothing::normal_alignment_factor( -std::cos( angle ), start );
    return factor * factor;
  };
  constexpr double h = 1.0e-5;
  for ( double join : { start, tribol::energy_mortar::right_angle } ) {
    SCOPED_TRACE( join );
    const double center = weight( join );
    const double slope_left = ( center - weight( join - h ) ) / h;
    const double slope_right = ( weight( join + h ) - center ) / h;
    EXPECT_NEAR( slope_left, slope_right, 1.0e-7 );
    const double curvature_left = ( center - 2.0 * weight( join - h ) + weight( join - 2.0 * h ) ) / ( h * h );
    const double curvature_right = ( weight( join + 2.0 * h ) - 2.0 * weight( join + h ) + center ) / ( h * h );
    EXPECT_NEAR( curvature_left, curvature_right, 0.01 );
  }
}

// A right-angle start setting is used to disable attenuation; a bug here would still weaken oblique opposing faces.
TEST( NormalAngleSmoothingCheck, NinetyDegreesDisablesAttenuation )
{
  constexpr double right_angle = energy_mortar::right_angle;
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( -1.0, right_angle ), -1.0 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( -0.5, right_angle ), -1.0 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( -1.0e-12, right_angle ), -1.0 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( 0.0, right_angle ), 0.0 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( 0.5, right_angle ), 0.0 );
}

// Each self-contact orientation uses its own mortar normal, so an oblique pair can be rejected in one direction and
// remain eligible in the reverse direction.
TEST( EnergyMortarAutoContactCheck, UsesMortarNormalForEachDirectedPair )
{
  RealT x1[2] = { 0.5, -0.5 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  constexpr RealT tangent_x = 0.1;
  constexpr RealT tangent_y = 0.99498743710662;
  RealT x2[2] = { -0.45, -0.45 + tangent_x };
  RealT y2[2] = { -0.5 - 0.5 * tangent_y, -0.5 + 0.5 * tangent_y };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ASSERT_TRUE( mesh1.computeFaceData( ExecutionMode::Sequential, PalletAvgNormal() ) );
  ASSERT_TRUE( mesh2.computeFaceData( ExecutionMode::Sequential, PalletAvgNormal() ) );
  const RealT thickness1[1] = { 1.0 };
  const RealT thickness2[1] = { 1.0 };
  mesh1.getElementData().m_thickness = ArrayViewT<const RealT>( thickness1, 1 );
  mesh2.getElementData().m_thickness = ArrayViewT<const RealT>( thickness2, 1 );
  mesh1.getElementData().m_is_element_thickness_set = true;
  mesh2.getElementData().m_is_element_thickness_set = true;

  constexpr RealT max_penetration_fraction = 0.3;
  EXPECT_TRUE(
      detail::energyMortarExceedsMaxAutoInterpen( mesh1.getView(), mesh2.getView(), 0, 0, max_penetration_fraction ) );
  EXPECT_FALSE(
      detail::energyMortarExceedsMaxAutoInterpen( mesh2.getView(), mesh1.getView(), 0, 0, max_penetration_fraction ) );

  ContactParams params;
  params.del = 0.0;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  params.normal_smoothing_start_angle = energy_mortar::right_angle;
  params.is_auto_contact = true;
  params.auto_contact_penetration_fraction = max_penetration_fraction;
  EnergyMortarCalculator evaluator( params );
  const InterfacePair pair( 0, 0 );

  const auto rejected =
      evaluator.compute_quadrature_point_penalty_data( pair, mesh2.getView(), mesh1.getView(), nullptr );
  const auto accepted =
      evaluator.compute_quadrature_point_penalty_data( pair, mesh1.getView(), mesh2.getView(), nullptr );
  EXPECT_EQ( rejected.energy, 0.0 );
  EXPECT_GT( accepted.energy, 0.0 );
}

// Counting the residual-gap offset as penetration would incorrectly reject this shallow physical overlap.
TEST( EnergyMortarAutoContactCheck, ResidualGapDoesNotCountAsPhysicalPenetration )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 1.0, 0.0 };
  RealT y2[2] = { -0.006, -0.006 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ASSERT_TRUE( mesh1.computeFaceData( ExecutionMode::Sequential, PalletAvgNormal() ) );
  ASSERT_TRUE( mesh2.computeFaceData( ExecutionMode::Sequential, PalletAvgNormal() ) );
  const RealT thickness1[1] = { 0.02 };
  const RealT thickness2[1] = { 0.02 };
  mesh1.getElementData().m_thickness = ArrayViewT<const RealT>( thickness1, 1 );
  mesh2.getElementData().m_thickness = ArrayViewT<const RealT>( thickness2, 1 );

  constexpr RealT max_penetration_fraction = 0.95;
  EXPECT_FALSE(
      detail::energyMortarExceedsMaxAutoInterpen( mesh1.getView(), mesh2.getView(), 0, 0, max_penetration_fraction ) );
}

// Include penetration-suppression derivatives in both edge and cap response.
TEST( EnergyMortarAutoContactCheck, SuppressedEdgeAndCapDerivativesMatchFiniteDifference )
{
  RealT x1[2] = { -1.0, 0.0 }, y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData source( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );
  RealT x2[2] = { 0.15, 0.05 }, y2[2] = { 0.05, 0.05 };
  IndexT conn2[2] = { 0, 1 };
  MeshData target( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );
  const RealT thickness[1] = { 0.2 };
  for ( auto* mesh : { &source, &target } ) {
    mesh->getElementData().m_thickness = ArrayViewT<const RealT>( thickness, 1 );
    mesh->getElementData().m_is_element_thickness_set = true;
  }
  ContactParams params;
  params.is_auto_contact = true;
  params.k = 3.0;
  EnergyMortarCalculator calculator( params );
  const double residual[2] = { 0.0, 0.2 };
  const InterfacePair pair( 0, 0 );
  const auto endpoints = endpointBallData( 0.0, -1.0 );
  auto evaluate = [&]() {
    source.setPosition( x1, y1, nullptr );
    target.setPosition( x2, y2, nullptr );
    auto result = calculator.compute_ball_penalty_data( pair, source.getView(), target.getView(), endpoints, residual );
    const auto edge =
        calculator.compute_quadrature_point_penalty_data( pair, source.getView(), target.getView(), residual );
    EXPECT_GT( result.energy, 0.0 );
    EXPECT_GT( edge.energy, 0.0 );
    result.energy += edge.energy;
    for ( int row = 0; row < 8; ++row ) {
      result.force[row] += edge.force[row];
      for ( int col = 0; col < 8; ++col ) result.stiffness[12 * row + col] += edge.stiffness[8 * row + col];
    }
    return result;
  };
  const auto base = evaluate();
  constexpr double eps = 1.0e-7;
  for ( int col = 0; col < 8; ++col ) {
    RealT* coordinates = col < 4 ? ( col % 2 == 0 ? x1 : y1 ) : ( col % 2 == 0 ? x2 : y2 );
    const int node = ( col % 4 ) / 2;
    const double original = coordinates[node];
    coordinates[node] = original + eps;
    const auto plus = evaluate();
    coordinates[node] = original - eps;
    const auto minus = evaluate();
    coordinates[node] = original;
    EXPECT_NEAR( base.force[col], ( plus.energy - minus.energy ) / ( 2.0 * eps ), 1.0e-6 );
    for ( int row = 0; row < 12; ++row ) {
      EXPECT_NEAR( base.stiffness[12 * row + col], ( plus.force[row] - minus.force[row] ) / ( 2.0 * eps ), 1.0e-5 );
    }
  }
}

// Suppress nodal contact without corrupting normalized gaps or their force and tangent derivatives.
TEST( EnergyMortarAutoContactCheck, NodalSuppressionHasConsistentGapAndAreaDerivatives )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData source( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );
  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.1, 0.1 };
  IndexT conn2[2] = { 1, 0 };
  MeshData target( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );
  const RealT thickness[1] = { 0.2 };
  for ( auto* mesh : { &source, &target } ) {
    mesh->getElementData().m_thickness = ArrayViewT<const RealT>( thickness, 1 );
    mesh->getElementData().m_is_element_thickness_set = true;
  }
  ContactParams params;
  params.is_auto_contact = true;
  params.residual_gap = 0.2;
  const InterfacePair pair( 0, 0 );
  for ( bool varying_quadrature : { false, true } ) {
    params.enzyme_quadrature = varying_quadrature;
    EnergyMortarCalculator calculator( params );
    auto value = [&]( double y, double* gap, double* area ) {
      y1[0] = y1[1] = y;
      source.setPosition( x1, y1, nullptr );
      // Rigid normal translation preserves the quadrature rule, so this reference also applies to the fixed-rule path.
      calculator.compute_gtilde_and_area( pair, source.getView(), target.getView(), gap, area );
    };
    double gap_minus[2], gap_plus[2], area_minus[2], area_plus[2];
    value( -1.0e-7, gap_minus, area_minus );
    value( 1.0e-7, gap_plus, area_plus );
    y1[0] = y1[1] = 0.0;
    source.setPosition( x1, y1, nullptr );
    double dg[2][8], da[2][8];
    calculator.grad_gtilde( pair, source.getView(), target.getView(), dg[0], dg[1] );
    calculator.grad_trib_area( pair, source.getView(), target.getView(), da[0], da[1] );
    for ( int node = 0; node < 2; ++node ) {
      EXPECT_NEAR( ( gap_plus[node] - gap_minus[node] ) / 2.0e-7, dg[node][1] + dg[node][3], 1.0e-7 );
      EXPECT_NEAR( ( area_plus[node] - area_minus[node] ) / 2.0e-7, da[node][1] + da[node][3], 1.0e-7 );
    }
    double hg[2][64], ha[2][64];
    calculator.d2_g2tilde( pair, source.getView(), target.getView(), hg[0], hg[1] );
    calculator.compute_d2A_d2u( pair, source.getView(), target.getView(), ha[0], ha[1] );
    double dg_minus[2][8], dg_plus[2][8], da_minus[2][8], da_plus[2][8];
    y1[0] = y1[1] = -1.0e-7;
    source.setPosition( x1, y1, nullptr );
    calculator.grad_gtilde( pair, source.getView(), target.getView(), dg_minus[0], dg_minus[1] );
    calculator.grad_trib_area( pair, source.getView(), target.getView(), da_minus[0], da_minus[1] );
    y1[0] = y1[1] = 1.0e-7;
    source.setPosition( x1, y1, nullptr );
    calculator.grad_gtilde( pair, source.getView(), target.getView(), dg_plus[0], dg_plus[1] );
    calculator.grad_trib_area( pair, source.getView(), target.getView(), da_plus[0], da_plus[1] );
    for ( int node = 0; node < 2; ++node ) {
      for ( int row = 0; row < 8; ++row ) {
        EXPECT_NEAR( ( dg_plus[node][row] - dg_minus[node][row] ) / 2.0e-7,
                     hg[node][8 * row + 1] + hg[node][8 * row + 3], 1.0e-6 );
        EXPECT_NEAR( ( da_plus[node][row] - da_minus[node][row] ) / 2.0e-7,
                     ha[node][8 * row + 1] + ha[node][8 * row + 3], 1.0e-6 );
      }
    }
    y1[0] = y1[1] = 0.0;
    source.setPosition( x1, y1, nullptr );
    ContactParams unfiltered = params;
    unfiltered.is_auto_contact = false;
    EnergyMortarCalculator unfiltered_calculator( unfiltered );
    double gap[2], area[2], unfiltered_gap[2], unfiltered_area[2];
    calculator.compute_gtilde_and_area( pair, source.getView(), target.getView(), gap, area );
    unfiltered_calculator.compute_gtilde_and_area( pair, source.getView(), target.getView(), unfiltered_gap,
                                                   unfiltered_area );
    EXPECT_GT( area[0], 0.0 );
    EXPECT_LT( area[0], unfiltered_area[0] );
    EXPECT_NEAR( gap[0] / area[0], unfiltered_gap[0] / unfiltered_area[0], 1.0e-12 );
    y2[0] = y2[1] = params.auto_contact_penetration_fraction * thickness[0];
    target.setPosition( x2, y2, nullptr );
    calculator.compute_gtilde_and_area( pair, source.getView(), target.getView(), gap, area );
    EXPECT_DOUBLE_EQ( gap[0], 0.0 );
    EXPECT_DOUBLE_EQ( area[0], 0.0 );
    y2[0] = y2[1] = 0.1;
    target.setPosition( x2, y2, nullptr );
  }
}

// Unequal endpoint offsets can make virtual normals oppose even when the physical normals do not; using only the
// physical normals would miss this valid contact pair.
TEST( EnergyMortarResidualGapCheck, VirtualNormalsCanOpposeWhenPhysicalNormalsDoNot )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.5, 0.500001 };
  RealT y2[2] = { -0.2, 0.2 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ASSERT_TRUE( mesh1.computeFaceData( ExecutionMode::Sequential, PalletAvgNormal() ) );
  ASSERT_TRUE( mesh2.computeFaceData( ExecutionMode::Sequential, PalletAvgNormal() ) );
  const double physical_normal_dot =
      mesh1.getView().getElementNormals()[0][0] * mesh2.getView().getElementNormals()[0][0] +
      mesh1.getView().getElementNormals()[1][0] * mesh2.getView().getElementNormals()[1][0];
  ASSERT_GT( physical_normal_dot, 0.0 );

  ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  params.normal_smoothing_start_angle = energy_mortar::default_normal_smoothing_start_angle;
  EnergyMortarCalculator evaluator( params );
  const double residual[2] = { 0.0, 0.2 };
  const auto contact = evaluator.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                        mesh2.getView(), residual );
  EXPECT_TRUE( contact.has_active_qp );
  EXPECT_GT( contact.energy, 0.0 );
}

TEST( QuadraturePointPenaltyCheck, OpenGapIsInactive )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.1, 0.1 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;

  EnergyMortarCalculator evaluator( params );
  const auto result =
      evaluator.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView() );
  EXPECT_FALSE( result.has_active_qp );
  EXPECT_EQ( result.energy, 0.0 );
  EXPECT_TRUE( std::all_of( result.force.begin(), result.force.end(), []( double force ) { return force == 0.0; } ) );
  EXPECT_TRUE( std::all_of( result.stiffness.begin(), result.stiffness.end(),
                            []( double stiffness ) { return stiffness == 0.0; } ) );
}

// Reading residual-gap values from the mortar edge would activate the wrong side of this separated pair.
TEST( EnergyMortarResidualGapCheck, UsesNonmortarResidualGap )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.1, 0.1 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  params.normal_smoothing_start_angle = energy_mortar::right_angle;
  params.residual_gap = 0.15;
  EnergyMortarCalculator evaluator( params );

  const double zero_gap[2] = { 0.0, 0.0 };
  const auto inactive = evaluator.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                         mesh2.getView(), zero_gap );
  EXPECT_FALSE( inactive.has_active_qp );

  const double nonmortar_full[2] = { 0.15, 0.15 };
  const auto active = evaluator.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                       mesh2.getView(), nonmortar_full );
  EXPECT_TRUE( active.has_active_qp );
  EXPECT_GT( active.energy, 0.0 );
}

// Replacing unequal endpoint gaps with one global value would project a uniformly offset edge instead of the ramp.
TEST( EnergyMortarResidualGapCheck, NonuniformRampChangesProjectionSpatially )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.1, 0.25 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  EnergyMortarCalculator evaluator( params );

  const double zero[2] = { 0.0, 0.0 };
  const double ramp[2] = { 0.0, 0.2 };
  const double global_max[2] = { 0.2, 0.2 };
  const auto zero_projection =
      evaluator.compute_projection_bounds( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(), zero );
  const auto ramp_projection =
      evaluator.compute_projection_bounds( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(), ramp );
  const auto max_projection =
      evaluator.compute_projection_bounds( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(), global_max );

  EXPECT_TRUE( std::abs( ramp_projection[0] - zero_projection[0] ) > 1.0e-6 ||
               std::abs( ramp_projection[1] - zero_projection[1] ) > 1.0e-6 );
  EXPECT_TRUE( std::abs( ramp_projection[0] - max_projection[0] ) > 1.0e-6 ||
               std::abs( ramp_projection[1] - max_projection[1] ) > 1.0e-6 );
}

// These physical edges do not project onto each other, so using physical geometry would miss overlap created by the
// residual-gap offsets.
TEST( EnergyMortarResidualGapCheck, ObliqueVirtualEdgesCreateProjectedOverlap )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.9, 0.95 };
  RealT y2[2] = { 0.1, 0.18660254037844387 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.0;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  EnergyMortarCalculator evaluator( params );
  const double zero[2] = { 0.0, 0.0 };
  const double residual[2] = { 0.25, 0.25 };

  const auto physical =
      evaluator.compute_projection_bounds( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(), zero );
  const auto virtual_projection =
      evaluator.compute_projection_bounds( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(), residual );
  EXPECT_LT( physical[1], -0.5 );
  EXPECT_GT( virtual_projection[1], -0.5 );

  const auto inactive =
      evaluator.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(), zero );
  const auto active = evaluator.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                       mesh2.getView(), residual );
  EXPECT_FALSE( inactive.has_active_qp );
  EXPECT_TRUE( active.has_active_qp );
  EXPECT_GT( active.energy, 0.0 );
}

// Applying the endpoint ball outside its corner cone would double count contact owned by the adjacent edge.
TEST( EnergyMortarResidualGapCheck, BallCompletionOnlyOwnsCornerCone )
{
  RealT x1[2] = { -1.0, 0.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.0;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  EnergyMortarCalculator evaluator( params );
  const double residual[2] = { 0.0, 0.2 };
  const auto endpoint_data = endpointBallData( 0.0, -1.0 );

  RealT x2_inside[2] = { 0.30, 0.05 };
  RealT y2[2] = { 0.05, 0.05 };
  IndexT conn2[2] = { 0, 1 };
  MeshData inside_mesh( 1, 1, 2, conn2, LINEAR_EDGE, x2_inside, y2, nullptr, MemorySpace::Host );
  const auto inside = evaluator.compute_ball_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                           inside_mesh.getView(), endpoint_data, residual );
  EXPECT_GT( inside.energy, 0.0 );

  RealT x2_outside[2] = { -0.05, -0.30 };
  MeshData outside_mesh( 1, 1, 2, conn2, LINEAR_EDGE, x2_outside, y2, nullptr, MemorySpace::Host );
  const auto outside = evaluator.compute_ball_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                            outside_mesh.getView(), endpoint_data, residual );
  EXPECT_EQ( outside.energy, 0.0 );
  EXPECT_TRUE( std::all_of( outside.force.begin(), outside.force.end(), []( double value ) { return value == 0.0; } ) );
  EXPECT_TRUE(
      std::all_of( outside.stiffness.begin(), outside.stiffness.end(), []( double value ) { return value == 0.0; } ) );

  RealT x2_crossing[2] = { 0.10, -0.10 };
  MeshData crossing_mesh( 1, 1, 2, conn2, LINEAR_EDGE, x2_crossing, y2, nullptr, MemorySpace::Host );
  const auto crossing_ball = evaluator.compute_ball_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                  crossing_mesh.getView(), endpoint_data, residual );
  const auto crossing_mortar = evaluator.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                                crossing_mesh.getView(), residual );
  EXPECT_GT( crossing_ball.energy, 0.0 );
  EXPECT_GT( crossing_mortar.energy, 0.0 );

  RealT x2_inside_half[2] = { 0.10, 0.0 };
  MeshData inside_half_mesh( 1, 1, 2, conn2, LINEAR_EDGE, x2_inside_half, y2, nullptr, MemorySpace::Host );
  const auto inside_half = evaluator.compute_ball_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                inside_half_mesh.getView(), endpoint_data, residual );
  EXPECT_NEAR( crossing_ball.energy, inside_half.energy, 1.0e-14 );
}

// Ignoring endpoint weights would add a cap at an ineligible corner or give a partial cap its full energy.
TEST( EnergyMortarResidualGapCheck, BallCompletionHonorsEndpointWeights )
{
  RealT x1[2] = { -1.0, 0.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.30, 0.05 };
  RealT y2[2] = { 0.05, 0.05 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.0;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  EnergyMortarCalculator evaluator( params );
  const InterfacePair pair( 0, 0 );
  const double residual[2] = { 0.0, 0.2 };
  auto eligible = openEndBallData();
  auto ineligible = eligible;
  ineligible.weight = { 1.0, 0.0 };
  auto half_weight = eligible;
  half_weight.weight[1] = 0.5;

  const auto active = evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), eligible, residual );
  ASSERT_TRUE( active.has_active_qp );
  ASSERT_GT( active.energy, 0.0 );

  const auto inactive =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), ineligible, residual );
  EXPECT_FALSE( inactive.has_active_qp );
  EXPECT_DOUBLE_EQ( inactive.energy, 0.0 );

  const auto weighted =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), half_weight, residual );
  EXPECT_NEAR( weighted.energy, 0.5 * active.energy, 1.0e-14 );
}

// Without normal smoothing on the endpoint ball, it can keep exerting force after the faces become perpendicular.
TEST( EnergyMortarResidualGapCheck, BallCompletionFadesOutAsNormalsBecomePerpendicular )
{
  RealT x1[2] = { 1.0, 0.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = {};
  RealT y2[2] = {};
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.0;
  const double right_angle = energy_mortar::right_angle;
  params.normal_smoothing_start_angle = 0.5 * right_angle;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  EnergyMortarCalculator evaluator( params );
  auto unsmoothed_params = params;
  unsmoothed_params.normal_smoothing_start_angle = right_angle;
  EnergyMortarCalculator unsmoothed_evaluator( unsmoothed_params );
  const InterfacePair pair( 0, 0 );
  const double residual[2] = { 0.2, 0.2 };
  const auto endpoint_data = openEndBallData();

  // Compare against the same cone-weighted geometry with normal smoothing disabled, which isolates the normal
  // alignment factor while the target edge rotates inside the endpoint cone.
  auto ball_energy_at_angle = [&]( const EnergyMortarCalculator& calculator, double mortar_angle ) {
    constexpr double segment_length = 0.25;
    constexpr double endpoint_offset = 0.05;
    constexpr double perpendicular_offset = 0.05;
    const double tangent[2] = { std::cos( mortar_angle ), std::sin( mortar_angle ) };
    const double normal[2] = { -tangent[1], tangent[0] };
    for ( int dimension = 0; dimension < 2; ++dimension ) {
      const double endpoint = -endpoint_offset * tangent[dimension] + perpendicular_offset * normal[dimension];
      if ( dimension == 0 ) {
        x2[1] = endpoint;
        x2[0] = endpoint - segment_length * tangent[dimension];
      } else {
        y2[1] = endpoint;
        y2[0] = endpoint - segment_length * tangent[dimension];
      }
    }
    mesh2.setPosition( x2, y2, nullptr );
    return calculator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), endpoint_data, residual )
        .energy;
  };

  const double opposed_energy = ball_energy_at_angle( evaluator, 0.0 );
  ASSERT_GT( opposed_energy, 0.0 );

  const double ramp_angle = 0.75 * right_angle;
  const double ramp_alignment =
      ContactSmoothing::normal_alignment_factor( -std::cos( ramp_angle ), params.normal_smoothing_start_angle );
  const double unsmoothed_ramp_energy = ball_energy_at_angle( unsmoothed_evaluator, ramp_angle );
  EXPECT_NEAR( ball_energy_at_angle( evaluator, ramp_angle ), ramp_alignment * ramp_alignment * unsmoothed_ramp_energy,
               1.0e-12 );

  const double near_perpendicular_angle = 89.0 * right_angle / 90.0;
  const double near_perpendicular_alignment = ContactSmoothing::normal_alignment_factor(
      -std::cos( near_perpendicular_angle ), params.normal_smoothing_start_angle );
  const double near_perpendicular_energy = ball_energy_at_angle( evaluator, near_perpendicular_angle );
  const double unsmoothed_near_perpendicular_energy =
      ball_energy_at_angle( unsmoothed_evaluator, near_perpendicular_angle );
  EXPECT_NEAR( near_perpendicular_energy,
               near_perpendicular_alignment * near_perpendicular_alignment * unsmoothed_near_perpendicular_energy,
               1.0e-12 );
  EXPECT_LT( near_perpendicular_energy, ball_energy_at_angle( evaluator, ramp_angle ) );

  EXPECT_NEAR( ball_energy_at_angle( evaluator, right_angle ), 0.0, 1.0e-30 );
  EXPECT_DOUBLE_EQ( ball_energy_at_angle( evaluator, 2.0 * right_angle ), 0.0 );
}

// Splitting one mortar edge into two must not duplicate the endpoint cap or change its total energy.
TEST( EnergyMortarResidualGapCheck, BallCompletionIsInvariantToMortarRefinement )
{
  RealT x1[2] = { -1.0, 0.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT coarse_x[2] = { 0.15, 0.05 };
  RealT coarse_y[2] = { 0.05, 0.05 };
  IndexT coarse_conn[2] = { 0, 1 };
  MeshData coarse( 1, 1, 2, coarse_conn, LINEAR_EDGE, coarse_x, coarse_y, nullptr, MemorySpace::Host );

  RealT fine_x[3] = { 0.15, 0.10, 0.05 };
  RealT fine_y[3] = { 0.05, 0.05, 0.05 };
  IndexT fine_conn[4] = { 0, 1, 1, 2 };
  MeshData fine( 1, 2, 3, fine_conn, LINEAR_EDGE, fine_x, fine_y, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.0;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  EnergyMortarCalculator evaluator( params );
  const double residual[2] = { 0.0, 0.2 };
  const auto endpoint_data = openEndBallData();

  const auto coarse_data = evaluator.compute_ball_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                coarse.getView(), endpoint_data, residual );
  double refined_energy = 0.0;
  for ( int edge = 0; edge < 2; ++edge ) {
    refined_energy += evaluator
                          .compute_ball_penalty_data( InterfacePair( 0, edge ), mesh1.getView(), fine.getView(),
                                                      endpoint_data, residual )
                          .energy;
  }
  // The radial integrand is non-polynomial, so independently mapping the same three-point rule to the two refined
  // edges produces a small quadrature difference.
  EXPECT_NEAR( refined_energy, coarse_data.energy, 2.0e-6 );
}

// Without nonmortar tributary scaling, refining the source edge would increase the total endpoint-cap energy.
TEST( EnergyMortarResidualGapCheck, BallCompletionScalesWithNonmortarTributaryLength )
{
  RealT coarse_x[2] = { -1.0, 0.0 };
  RealT refined_x[2] = { -0.5, 0.0 };
  RealT nonmortar_y[2] = { 0.0, 0.0 };
  IndexT nonmortar_conn[2] = { 0, 1 };
  MeshData coarse_nonmortar( 0, 1, 2, nonmortar_conn, LINEAR_EDGE, coarse_x, nonmortar_y, nullptr, MemorySpace::Host );
  MeshData refined_nonmortar( 0, 1, 2, nonmortar_conn, LINEAR_EDGE, refined_x, nonmortar_y, nullptr,
                              MemorySpace::Host );

  RealT mortar_x[2] = { 0.30, 0.05 };
  RealT mortar_y[2] = { 0.05, 0.05 };
  IndexT mortar_conn[2] = { 0, 1 };
  MeshData mortar( 1, 1, 2, mortar_conn, LINEAR_EDGE, mortar_x, mortar_y, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.0;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  EnergyMortarCalculator evaluator( params );
  const double residual[2] = { 0.0, 0.2 };
  const auto endpoint_data = openEndBallData();
  const double coarse_energy = evaluator
                                   .compute_ball_penalty_data( InterfacePair( 0, 0 ), coarse_nonmortar.getView(),
                                                               mortar.getView(), endpoint_data, residual )
                                   .energy;
  const double refined_energy = evaluator
                                    .compute_ball_penalty_data( InterfacePair( 0, 0 ), refined_nonmortar.getView(),
                                                                mortar.getView(), endpoint_data, residual )
                                    .energy;
  ASSERT_GT( coarse_energy, 0.0 );
  EXPECT_NEAR( refined_energy, 0.5 * coarse_energy, 1.0e-14 );
}

// Moving the adjacent nonmortar node moves the corner-cone boundary; omitting that derivative gives incorrect force
// and stiffness entries.
TEST( EnergyMortarResidualGapCheck, BallCompletionConeDerivativesMatchFiniteDifference )
{
  RealT x1[2] = { -1.0, 0.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );
  // A short portion of the target edge crosses the boundary defined by the second source edge. This exercises the
  // moving cone bound and derivatives with respect to the additional source node as well as both contact edges.
  RealT x2[2] = { 0.15, 0.05 };
  RealT y2[2] = { 0.005, -0.10 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.0;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  EnergyMortarCalculator evaluator( params );
  const InterfacePair pair( 0, 0 );
  const double residual[2] = { 0.0, 0.2 };
  auto endpoint_data = endpointBallData( 0.0, -1.0 );
  const auto analytical =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), endpoint_data, residual );
  ASSERT_GT( analytical.energy, 0.0 );

  const std::array<RealT, 2> x1_orig{ x1[0], x1[1] };
  const std::array<RealT, 2> y1_orig{ y1[0], y1[1] };
  const std::array<RealT, 2> x2_orig{ x2[0], x2[1] };
  const std::array<RealT, 2> y2_orig{ y2[0], y2[1] };
  const auto neighbor_orig = endpoint_data.neighbor_coordinates;
  auto restore = [&]() {
    std::copy( x1_orig.begin(), x1_orig.end(), x1 );
    std::copy( y1_orig.begin(), y1_orig.end(), y1 );
    std::copy( x2_orig.begin(), x2_orig.end(), x2 );
    std::copy( y2_orig.begin(), y2_orig.end(), y2 );
    endpoint_data.neighbor_coordinates = neighbor_orig;
    mesh1.setPosition( x1, y1, nullptr );
    mesh2.setPosition( x2, y2, nullptr );
  };
  auto perturb = [&]( int dof, double delta ) {
    RealT* component = nullptr;
    int node = 0;
    if ( dof < 4 ) {
      node = dof / 2;
      component = dof % 2 == 0 ? x1 : y1;
      mesh1.setPosition( x1, y1, nullptr );
    } else if ( dof < 8 ) {
      node = ( dof - 4 ) / 2;
      component = dof % 2 == 0 ? x2 : y2;
    } else {
      endpoint_data.neighbor_coordinates[dof - 8] += delta;
      return;
    }
    component[node] += delta;
    mesh1.setPosition( x1, y1, nullptr );
    mesh2.setPosition( x2, y2, nullptr );
  };

  constexpr double grad_eps = 1.0e-7;
  for ( int dof = 0; dof < BallPenaltyData::num_force_dofs; ++dof ) {
    restore();
    perturb( dof, grad_eps );
    const double plus =
        evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), endpoint_data, residual ).energy;
    restore();
    perturb( dof, -grad_eps );
    const double minus =
        evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), endpoint_data, residual ).energy;
    EXPECT_NEAR( analytical.force[dof], ( plus - minus ) / ( 2.0 * grad_eps ), 1.0e-6 ) << "dof " << dof;
  }

  constexpr double hess_eps = 1.0e-6;
  for ( int col = 0; col < BallPenaltyData::num_force_dofs; ++col ) {
    restore();
    perturb( col, hess_eps );
    const auto plus =
        evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), endpoint_data, residual ).force;
    restore();
    perturb( col, -hess_eps );
    const auto minus =
        evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), endpoint_data, residual ).force;
    for ( int row = 0; row < BallPenaltyData::num_force_dofs; ++row ) {
      EXPECT_NEAR( analytical.stiffness[row * BallPenaltyData::num_force_dofs + col],
                   ( plus[row] - minus[row] ) / ( 2.0 * hess_eps ), 2.0e-4 )
          << "row " << row << ", col " << col;
    }
  }
  restore();
}

// A stale clipped interval can leave ghost energy, force, or stiffness after the target edge leaves the corner cone.
TEST( EnergyMortarResidualGapCheck, BallCompletionEnergyVanishesWithConeOverlap )
{
  RealT x1[2] = { -1.0, 0.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.01, -0.10 };
  RealT y2[2] = { 0.05, 0.05 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.0;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  EnergyMortarCalculator evaluator( params );
  const InterfacePair pair( 0, 0 );
  const double residual[2] = { 0.0, 0.2 };
  const auto endpoint_data = endpointBallData( 0.0, -1.0 );
  auto norm = []( const auto& values ) {
    double norm_sq = 0.0;
    for ( double value : values ) {
      norm_sq += value * value;
    }
    return std::sqrt( norm_sq );
  };

  const auto finite_overlap =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), endpoint_data, residual );
  ASSERT_GT( finite_overlap.energy, 0.0 );
  x2[0] = 1.0e-6;
  mesh2.setPosition( x2, y2, nullptr );
  const auto vanishing_overlap =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), endpoint_data, residual );
  ASSERT_GT( vanishing_overlap.energy, 0.0 );
  EXPECT_LT( vanishing_overlap.energy, 1.0e-2 * finite_overlap.energy );

  x2[0] = 0.0;
  mesh2.setPosition( x2, y2, nullptr );
  const auto zero_overlap =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), endpoint_data, residual );
  EXPECT_DOUBLE_EQ( zero_overlap.energy, 0.0 );
  EXPECT_DOUBLE_EQ( norm( zero_overlap.force ), 0.0 );
  EXPECT_DOUBLE_EQ( norm( zero_overlap.stiffness ), 0.0 );
}

// Incorrect smoothing derivatives can leave a finite endpoint force as the corner-cone overlap shrinks to zero.
TEST( EnergyMortarResidualGapCheck, SmoothedBallForceVanishesWithConeOverlap )
{
  RealT x1[2] = { -1.0, 0.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.01, -0.10 };
  RealT y2[2] = { 0.05, 0.05 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  EnergyMortarCalculator evaluator( params );
  const InterfacePair pair( 0, 0 );
  const double residual[2] = { 0.0, 0.2 };
  const auto endpoint_data = endpointBallData( 0.0, -1.0 );
  auto norm = []( const auto& values ) {
    double norm_sq = 0.0;
    for ( double value : values ) {
      norm_sq += value * value;
    }
    return std::sqrt( norm_sq );
  };

  const auto finite_overlap =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), endpoint_data, residual );
  ASSERT_GT( finite_overlap.energy, 0.0 );
  const double finite_force_norm = norm( finite_overlap.force );
  const double finite_stiffness_norm = norm( finite_overlap.stiffness );
  ASSERT_GT( finite_force_norm, 0.0 );
  ASSERT_GT( finite_stiffness_norm, 0.0 );

  x2[0] = 1.0e-6;
  mesh2.setPosition( x2, y2, nullptr );
  const auto vanishing_overlap =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), endpoint_data, residual );
  ASSERT_GT( vanishing_overlap.energy, 0.0 );
  EXPECT_LT( norm( vanishing_overlap.force ), 1.0e-2 * finite_force_norm );
  EXPECT_LT( norm( vanishing_overlap.stiffness ), 1.0e-2 * finite_stiffness_norm );
}

// Dependence on absolute coordinates would change endpoint-ball energy under translation and can produce unbalanced
// forces or a nonsymmetric Hessian.
TEST( EnergyMortarResidualGapCheck, BallCompletionPreservesRigidMotionsAndHasSymmetricHessian )
{
  RealT x1[2] = { -1.0, 0.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );
  RealT x2[2] = { 0.30, 0.05 };
  RealT y2[2] = { 0.05, 0.05 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.0;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  EnergyMortarCalculator evaluator( params );
  const double residual[2] = { 0.0, 0.2 };
  auto endpoint_data = endpointBallData( 0.0, -1.0 );
  const auto base = evaluator.compute_ball_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(),
                                                         endpoint_data, residual );
  ASSERT_GT( base.energy, 0.0 );

  EXPECT_NEAR( base.force[0] + base.force[2] + base.force[4] + base.force[6] + base.force[8] + base.force[10], 0.0,
               1.0e-10 );
  EXPECT_NEAR( base.force[1] + base.force[3] + base.force[5] + base.force[7] + base.force[9] + base.force[11], 0.0,
               1.0e-10 );
  for ( int row = 0; row < BallPenaltyData::num_force_dofs; ++row ) {
    for ( int col = 0; col < BallPenaltyData::num_force_dofs; ++col ) {
      EXPECT_NEAR( base.stiffness[row * BallPenaltyData::num_force_dofs + col],
                   base.stiffness[col * BallPenaltyData::num_force_dofs + row], 1.0e-10 );
    }
  }

  for ( double& x : x1 ) x += 1.25;
  for ( double& y : y1 ) y -= 0.75;
  for ( double& x : x2 ) x += 1.25;
  for ( double& y : y2 ) y -= 0.75;
  endpoint_data.neighbor_coordinates[2] += 1.25;
  endpoint_data.neighbor_coordinates[3] -= 0.75;
  mesh1.setPosition( x1, y1, nullptr );
  mesh2.setPosition( x2, y2, nullptr );
  const auto translated = evaluator.compute_ball_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(),
                                                               endpoint_data, residual );
  EXPECT_NEAR( translated.energy, base.energy, 1.0e-12 );
  for ( int i = 0; i < BallPenaltyData::num_force_dofs; ++i ) {
    EXPECT_NEAR( translated.force[i], base.force[i], 1.0e-9 );
  }
}

// The sum of edge and endpoint response must have matching limits as a target slides across the ownership boundary.
TEST( EnergyMortarResidualGapCheck, EdgeCapHandoffHasContinuousForceAndJacobian )
{
  RealT x1[2] = { -1.0, 0.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData source( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );
  RealT x2[2] = {};
  RealT y2[2] = { -0.1, -0.1 };
  IndexT conn2[2] = { 1, 0 };
  MeshData target( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );
  ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.residual_gap = 0.2;
  EnergyMortarCalculator evaluator( params );
  const auto endpoints = openEndBallData();
  const double residual[2] = { 0.2, 0.2 };
  const InterfacePair pair( 0, 0 );
  auto evaluate = [&]( double center ) {
    x2[0] = center - 0.02;
    x2[1] = center + 0.02;
    target.setPosition( x2, y2, nullptr );
    auto total = evaluator.compute_quadrature_point_penalty_data( pair, source.getView(), target.getView(), residual );
    const auto cap =
        evaluator.compute_ball_penalty_data( pair, source.getView(), target.getView(), endpoints, residual );
    total.energy += cap.energy;
    for ( int row = 0; row < 8; ++row ) {
      total.force[row] += cap.force[row];
      for ( int col = 0; col < 8; ++col ) {
        total.stiffness[row * 8 + col] += cap.stiffness[row * 12 + col];
      }
    }
    return total;
  };
  // Cover both target endpoints crossing the cone, the midpoint handoff, and both ends of edge-bound smoothing.
  for ( double handoff : { -0.12, -0.08, -0.02, 0.0, 0.02, 0.08, 0.12 } ) {
    SCOPED_TRACE( handoff );
    const auto coarse_left = evaluate( handoff - 1.0e-6 );
    const auto coarse_right = evaluate( handoff + 1.0e-6 );
    const auto left = evaluate( handoff - 1.0e-8 );
    const auto right = evaluate( handoff + 1.0e-8 );
    EXPECT_NEAR( left.energy, right.energy, 1.0e-7 );
    for ( int row = 0; row < 8; ++row ) {
      EXPECT_NEAR( left.force[row], right.force[row], 1.0e-6 );
      for ( int col = 0; col < 8; ++col ) {
        EXPECT_NEAR( left.stiffness[row * 8 + col], right.stiffness[row * 8 + col], 1.0e-5 );
        EXPECT_LE(
            std::abs( left.stiffness[row * 8 + col] - right.stiffness[row * 8 + col] ),
            0.1 * std::abs( coarse_left.stiffness[row * 8 + col] - coarse_right.stiffness[row * 8 + col] ) + 1.0e-8 );
      }
    }
    // Away from the exact join, the assembled tangent must describe the same total force being evaluated.
    for ( double side : { -1.0, 1.0 } ) {
      const double center = handoff + side * 1.0e-4;
      const auto middle = evaluate( center );
      const auto minus = evaluate( center - 1.0e-7 );
      const auto plus = evaluate( center + 1.0e-7 );
      EXPECT_NEAR( ( plus.energy - minus.energy ) / 2.0e-7, middle.force[4] + middle.force[6], 1.0e-6 );
      for ( int row = 0; row < 8; ++row ) {
        EXPECT_NEAR( ( plus.force[row] - minus.force[row] ) / 2.0e-7,
                     middle.stiffness[row * 8 + 4] + middle.stiffness[row * 8 + 6], 1.0e-5 );
      }
    }
  }
}

// Keep the assembled response continuous across angular attenuation joins for both features.
TEST( EnergyMortarResidualGapCheck, CombinedEdgeCapAngularTransitionsAreContinuous )
{
  RealT x1[2] = { -1.0, 0.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData source( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );
  RealT x2[2] = {};
  RealT y2[2] = {};
  IndexT conn2[2] = { 1, 0 };
  MeshData target( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );
  ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.residual_gap = 0.2;
  EnergyMortarCalculator evaluator( params );
  const auto endpoints = openEndBallData();
  const double residual[2] = { 0.2, 0.2 };
  const InterfacePair pair( 0, 0 );
  auto evaluate = [&]( double angle ) {
    for ( int node = 0; node < 2; ++node ) {
      const double sign = node == 0 ? -1.0 : 1.0;
      x2[node] = 0.01 + sign * 0.02 * std::cos( angle );
      y2[node] = -0.1 + sign * 0.02 * std::sin( angle );
    }
    target.setPosition( x2, y2, nullptr );
    auto result = evaluator.compute_quadrature_point_penalty_data( pair, source.getView(), target.getView(), residual );
    const auto cap =
        evaluator.compute_ball_penalty_data( pair, source.getView(), target.getView(), endpoints, residual );
    result.energy += cap.energy;
    for ( int row = 0; row < 8; ++row ) {
      result.force[row] += cap.force[row];
      for ( int col = 0; col < 8; ++col ) {
        result.stiffness[8 * row + col] += cap.stiffness[12 * row + col];
      }
    }
    return result;
  };
  for ( double angle : { params.normal_smoothing_start_angle, energy_mortar::right_angle } ) {
    SCOPED_TRACE( angle );
    const auto left = evaluate( angle - 1.0e-8 );
    const auto right = evaluate( angle + 1.0e-8 );
    EXPECT_NEAR( left.energy, right.energy, 1.0e-7 );
    for ( int row = 0; row < 8; ++row ) {
      EXPECT_NEAR( left.force[row], right.force[row], 1.0e-6 );
      for ( int col = 0; col < 8; ++col ) {
        EXPECT_NEAR( left.stiffness[8 * row + col], right.stiffness[8 * row + col], 1.0e-5 );
      }
    }
  }
}

// Match total force and tangent limits at both cap handoffs around a convex corner.
TEST( EnergyMortarResidualGapCheck, ConvexCornerEdgeCapEdgeHandoffsAreContinuous )
{
  RealT sx[3] = { -1.0, 0.0, 0.0 }, sy[3] = { 0.0, 0.0, 1.0 };
  IndexT source_conn[4] = { 0, 1, 1, 2 };
  MeshData source( 0, 2, 3, source_conn, LINEAR_EDGE, sx, sy, nullptr, MemorySpace::Host );
  RealT tx[2] = {}, ty[2] = {};
  IndexT target_conn[2] = { 1, 0 };
  MeshData target( 1, 1, 2, target_conn, LINEAR_EDGE, tx, ty, nullptr, MemorySpace::Host );
  ContactParams params;
  params.k = 3.0;
  EnergyMortarCalculator calculator( params );
  const double residual[2] = { 0.2, 0.2 };
  BallEndpointData caps[2];
  caps[0].weight = { 0.0, 0.5 };
  caps[0].neighbor_coordinates = { 0.0, 0.0, 0.0, 1.0 };
  caps[1].weight = { 0.5, 0.0 };
  caps[1].neighbor_coordinates = { -1.0, 0.0, 0.0, 0.0 };
  struct Result {
    double energy{};
    std::array<double, 10> force{};
    std::array<double, 100> stiffness{};
  };
  auto evaluate = [&]( double angle ) {
    for ( int node = 0; node < 2; ++node ) {
      const double sign = node == 0 ? -1.0 : 1.0;
      tx[node] = 0.1 * std::cos( angle ) - sign * 0.02 * std::sin( angle );
      ty[node] = 0.1 * std::sin( angle ) + sign * 0.02 * std::cos( angle );
    }
    target.setPosition( tx, ty, nullptr );
    Result total;
    for ( int edge = 0; edge < 2; ++edge ) {
      const InterfacePair pair( edge, 0 );
      const auto projected =
          calculator.compute_quadrature_point_penalty_data( pair, source.getView(), target.getView(), residual );
      const auto cap =
          calculator.compute_ball_penalty_data( pair, source.getView(), target.getView(), caps[edge], residual );
      // Global ordering: three source nodes, then the two target nodes. Each cap also carries its neighboring source
      // node.
      const int map[12] = { 2 * edge, 2 * edge + 1,      2 * edge + 2,     2 * edge + 3, 8, 9, 6, 7, 0,
                            1,        edge == 0 ? 4 : 0, edge == 0 ? 5 : 1 };
      total.energy += projected.energy + cap.energy;
      for ( int row = 0; row < 12; ++row ) {
        total.force[map[row]] += cap.force[row];
        if ( row < 8 ) total.force[map[row]] += projected.force[row];
        for ( int col = 0; col < 12; ++col ) {
          total.stiffness[10 * map[row] + map[col]] += cap.stiffness[12 * row + col];
          if ( row < 8 && col < 8 ) total.stiffness[10 * map[row] + map[col]] += projected.stiffness[8 * row + col];
        }
      }
    }
    return total;
  };
  for ( double join : { -energy_mortar::right_angle, -std::atan( 5.0 ), -0.5 * energy_mortar::right_angle,
                        -std::atan( 0.2 ), 0.0 } ) {
    SCOPED_TRACE( join );
    const auto left = evaluate( join - 1.0e-8 );
    const auto right = evaluate( join + 1.0e-8 );
    EXPECT_NEAR( left.energy, right.energy, 1.0e-7 );
    for ( int row = 0; row < 10; ++row ) {
      EXPECT_NEAR( left.force[row], right.force[row], 1.0e-6 );
      for ( int col = 0; col < 10; ++col )
        EXPECT_NEAR( left.stiffness[10 * row + col], right.stiffness[10 * row + col], 1.0e-5 );
    }
    for ( double side : { -1.0, 1.0 } ) {
      const double angle = join + side * 1.0e-4;
      const auto middle = evaluate( angle );
      const auto minus = evaluate( angle - 1.0e-7 );
      const auto plus = evaluate( angle + 1.0e-7 );
      double velocity[4];
      for ( int node = 0; node < 2; ++node ) {
        const double sign = node == 0 ? -1.0 : 1.0;
        velocity[2 * node] = -0.1 * std::sin( angle ) - sign * 0.02 * std::cos( angle );
        velocity[2 * node + 1] = 0.1 * std::cos( angle ) - sign * 0.02 * std::sin( angle );
      }
      double energy_derivative = 0.0;
      for ( int col = 0; col < 4; ++col ) energy_derivative += middle.force[6 + col] * velocity[col];
      EXPECT_NEAR( ( plus.energy - minus.energy ) / 2.0e-7, energy_derivative, 1.0e-6 );
      for ( int row = 0; row < 10; ++row ) {
        double force_derivative = 0.0;
        for ( int col = 0; col < 4; ++col ) force_derivative += middle.stiffness[10 * row + 6 + col] * velocity[col];
        EXPECT_NEAR( ( plus.force[row] - minus.force[row] ) / 2.0e-7, force_derivative, 1.0e-5 );
      }
    }
  }
}

// Missing derivatives of a virtual endpoint can create a net force or make the response change after translation.
TEST( EnergyMortarResidualGapCheck, ForceBalanceAndTranslationInvariance )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );
  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.1, 0.1 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  params.residual_gap = 0.15;
  EnergyMortarCalculator evaluator( params );
  const auto base =
      evaluator.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView() );
  EXPECT_NEAR( base.force[0] + base.force[2] + base.force[4] + base.force[6], 0.0, 1.0e-9 );
  EXPECT_NEAR( base.force[1] + base.force[3] + base.force[5] + base.force[7], 0.0, 1.0e-9 );

  for ( double& x : x1 ) x += 1.25;
  for ( double& y : y1 ) y -= 0.75;
  for ( double& x : x2 ) x += 1.25;
  for ( double& y : y2 ) y -= 0.75;
  mesh1.setPosition( x1, y1, nullptr );
  mesh2.setPosition( x2, y2, nullptr );
  const auto translated =
      evaluator.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView() );
  EXPECT_NEAR( translated.energy, base.energy, 1.0e-12 );
  for ( int i = 0; i < 8; ++i ) {
    EXPECT_NEAR( translated.force[i], base.force[i], 1.0e-9 );
  }
}

TEST( QuadraturePointPenaltyCheck, ZeroGapRetainsActiveTangent )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.0, 0.0 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;

  EnergyMortarCalculator evaluator( params );
  const auto result =
      evaluator.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView() );

  EXPECT_TRUE( result.has_active_qp );
  EXPECT_EQ( result.energy, 0.0 );
  EXPECT_TRUE( std::all_of( result.force.begin(), result.force.end(), []( double force ) { return force == 0.0; } ) );
  EXPECT_TRUE( std::any_of( result.stiffness.begin(), result.stiffness.end(),
                            []( double stiffness ) { return stiffness != 0.0; } ) );
}

TEST( EnergyMortarResidualGapCheck, AssembledGapIsShiftedByArea )
{
  // The parallel edges have a normal separation of 0.1.
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.1, 0.1 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;

  double gap_without_residual[2] = { 0.0, 0.0 };
  double tributary_area_without_residual[2] = { 0.0, 0.0 };
  EnergyMortarCalculator evaluator_without_residual( params );
  evaluator_without_residual.compute_gtilde_and_area( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(),
                                                      gap_without_residual, tributary_area_without_residual );

  params.residual_gap = 0.15;
  double gap_with_residual[2] = { 0.0, 0.0 };
  double tributary_area_with_residual[2] = { 0.0, 0.0 };
  EnergyMortarCalculator evaluator_with_residual( params );
  evaluator_with_residual.compute_gtilde_and_area( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(),
                                                   gap_with_residual, tributary_area_with_residual );

  // The residual gap shifts the gap integral without changing the projected overlap geometry.
  for ( int i = 0; i < 2; ++i ) {
    EXPECT_NEAR( tributary_area_with_residual[i], tributary_area_without_residual[i], 1.0e-14 );
    EXPECT_NEAR( gap_with_residual[i],
                 gap_without_residual[i] - params.residual_gap * tributary_area_without_residual[i], 1.0e-14 );
  }
}

// Merely subtracting the residual gap from a physical-edge integral gives the wrong tributary areas for oblique
// virtual edges.
TEST( EnergyMortarResidualGapCheck, VirtualGeometryRespectsNormalSmoothing )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  constexpr double pi = std::numbers::pi_v<double>;
  constexpr double angle = pi / 3.0;
  constexpr double separation = 0.3;
  constexpr double half_edge_length = 0.05;
  const double tangent_x = std::cos( angle );
  const double tangent_y = std::sin( angle );
  const double normal_x = tangent_y;
  const double normal_y = -tangent_x;
  const double center_x = 0.5 + separation * normal_x;
  const double center_y = separation * normal_y;
  RealT x2[2] = { center_x - half_edge_length * tangent_x, center_x + half_edge_length * tangent_x };
  RealT y2[2] = { center_y - half_edge_length * tangent_y, center_y + half_edge_length * tangent_y };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;

  double gap_without_residual[2] = { 0.0, 0.0 };
  double tributary_area_without_residual[2] = { 0.0, 0.0 };
  EnergyMortarCalculator evaluator_without_residual( params );
  evaluator_without_residual.compute_gtilde_and_area( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(),
                                                      gap_without_residual, tributary_area_without_residual );

  params.residual_gap = 0.15;
  double gap_with_residual[2] = { 0.0, 0.0 };
  double tributary_area_with_residual[2] = { 0.0, 0.0 };
  EnergyMortarCalculator evaluator_with_residual( params );
  evaluator_with_residual.compute_gtilde_and_area( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(),
                                                   gap_with_residual, tributary_area_with_residual );

  for ( int i = 0; i < 2; ++i ) {
    EXPECT_NE( tributary_area_with_residual[i], tributary_area_without_residual[i] );
    EXPECT_NE( gap_with_residual[i], gap_without_residual[i] );
  }
}

TEST( EnergyMortarResidualGapCheck, QuadraturePointOpenGapBecomesActive )
{
  // The edges have a normal separation of 0.1; a residual gap of 0.15 produces an effective gap of -0.05.
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.1, 0.1 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;

  EnergyMortarCalculator evaluator_without_residual( params );
  const auto inactive = evaluator_without_residual.compute_quadrature_point_penalty_data(
      InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView() );
  EXPECT_FALSE( inactive.has_active_qp );
  EXPECT_EQ( inactive.energy, 0.0 );
  EXPECT_TRUE(
      std::all_of( inactive.force.begin(), inactive.force.end(), []( double force ) { return force == 0.0; } ) );
  EXPECT_TRUE( std::all_of( inactive.stiffness.begin(), inactive.stiffness.end(),
                            []( double stiffness ) { return stiffness == 0.0; } ) );

  params.residual_gap = 0.15;
  EnergyMortarCalculator evaluator_with_residual( params );
  const auto active = evaluator_with_residual.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ),
                                                                                     mesh1.getView(), mesh2.getView() );
  EXPECT_TRUE( active.has_active_qp );
  EXPECT_GT( active.energy, 0.0 );
  EXPECT_TRUE( std::any_of( active.force.begin(), active.force.end(), []( double force ) { return force != 0.0; } ) );
  EXPECT_TRUE( std::any_of( active.stiffness.begin(), active.stiffness.end(),
                            []( double stiffness ) { return stiffness != 0.0; } ) );
}

class ResidualGapDerivativeCheck : public ::testing::TestWithParam<std::tuple<double, bool>> {};

TEST_P( ResidualGapDerivativeCheck, QuadraturePointPenaltyDerivativesMatchFiniteDifference )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  constexpr double pi = std::numbers::pi_v<double>;
  constexpr double angle = pi / 3.0;
  constexpr double separation = 0.3;
  constexpr double half_edge_length = 0.05;
  const double tangent_x = std::cos( angle );
  const double tangent_y = std::sin( angle );
  const double normal_x = tangent_y;
  const double normal_y = -tangent_x;
  const double center_x = 0.5 + separation * normal_x;
  const double center_y = separation * normal_y;
  RealT x2[2] = { center_x - half_edge_length * tangent_x, center_x + half_edge_length * tangent_x };
  RealT y2[2] = { center_y - half_edge_length * tangent_y, center_y + half_edge_length * tangent_y };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = std::get<1>( GetParam() );
  params.residual_gap = std::get<0>( GetParam() );

  EnergyMortarCalculator evaluator( params );
  const InterfacePair pair( 0, 0 );
  const auto analytical = evaluator.compute_quadrature_point_penalty_data( pair, mesh1.getView(), mesh2.getView() );
  ASSERT_GT( analytical.energy, 0.0 );

  const std::array<RealT, 2> x1_orig{ x1[0], x1[1] };
  const std::array<RealT, 2> y1_orig{ y1[0], y1[1] };
  const std::array<RealT, 2> x2_orig{ x2[0], x2[1] };
  const std::array<RealT, 2> y2_orig{ y2[0], y2[1] };

  auto restore = [&]() {
    x1[0] = x1_orig[0];
    x1[1] = x1_orig[1];
    y1[0] = y1_orig[0];
    y1[1] = y1_orig[1];
    x2[0] = x2_orig[0];
    x2[1] = x2_orig[1];
    y2[0] = y2_orig[0];
    y2[1] = y2_orig[1];
    mesh1.setPosition( x1, y1, nullptr );
    mesh2.setPosition( x2, y2, nullptr );
  };

  auto perturb = [&]( int dof, double delta ) {
    if ( dof < 4 ) {
      const int endpoint = dof / 2;
      const int component = dof % 2;
      const int node = conn1[endpoint];
      ( component == 0 ? x1[node] : y1[node] ) += delta;
      mesh1.setPosition( x1, y1, nullptr );
    } else {
      const int endpoint = ( dof - 4 ) / 2;
      const int component = ( dof - 4 ) % 2;
      const int node = conn2[endpoint];
      ( component == 0 ? x2[node] : y2[node] ) += delta;
      mesh2.setPosition( x2, y2, nullptr );
    }
  };

  const double gradient_eps = 1.0e-7;
  const double gradient_tol = 1.0e-6;
  for ( int dof = 0; dof < 8; ++dof ) {
    restore();
    perturb( dof, gradient_eps );
    const double energy_plus =
        evaluator.compute_quadrature_point_penalty_energy( pair, mesh1.getView(), mesh2.getView() );
    restore();
    perturb( dof, -gradient_eps );
    const double energy_minus =
        evaluator.compute_quadrature_point_penalty_energy( pair, mesh1.getView(), mesh2.getView() );
    const double fd_force = ( energy_plus - energy_minus ) / ( 2.0 * gradient_eps );
    EXPECT_NEAR( fd_force, analytical.force[dof], gradient_tol ) << "force mismatch at dof " << dof;
  }

  const double hessian_eps = 1.0e-6;
  const double hessian_tol = 1.0e-4;
  for ( int col = 0; col < 8; ++col ) {
    restore();
    perturb( col, hessian_eps );
    const auto force_plus =
        evaluator.compute_quadrature_point_penalty_data( pair, mesh1.getView(), mesh2.getView() ).force;
    restore();
    perturb( col, -hessian_eps );
    const auto force_minus =
        evaluator.compute_quadrature_point_penalty_data( pair, mesh1.getView(), mesh2.getView() ).force;
    for ( int row = 0; row < 8; ++row ) {
      const double fd_stiffness = ( force_plus[row] - force_minus[row] ) / ( 2.0 * hessian_eps );
      EXPECT_NEAR( fd_stiffness, analytical.stiffness[row * 8 + col], hessian_tol )
          << "stiffness mismatch at row " << row << ", col " << col;
    }
  }
  restore();
}

TEST_P( ResidualGapDerivativeCheck, GtildeGradientFDvsAD )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };  // reversed so nA points toward B
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
  params_.enzyme_quadrature = std::get<1>( GetParam() );
  params_.residual_gap = std::get<0>( GetParam() );

  EnergyMortarCalculator evaluator_( params_ );

  const double epsilon = 1e-7;

  RealT y1_plus[2] = { epsilon, 0.0 };
  RealT y1_minus[2] = { -epsilon, 0.0 };
  mesh1.setPosition( x1, y1_plus, nullptr );
  mesh1.setPosition( x1, y1_minus, nullptr );
  mesh1.setPosition( x1, y1, nullptr );
  auto result = evaluator_.validate_g_tilde( pair, mesh1, mesh2, epsilon );

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

TEST_P( ResidualGapDerivativeCheck, GtildeHessianFDvsAD )
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
  params_.enzyme_quadrature = std::get<1>( GetParam() );
  params_.residual_gap = std::get<0>( GetParam() );

  EnergyMortarCalculator evaluator_( params_ );

  const double epsilon = 1e-5;
  auto result = evaluator_.validate_hessian( pair, mesh1, mesh2, epsilon );

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

INSTANTIATE_TEST_SUITE_P( FixedAndGeometryDependentQuadrature, ResidualGapDerivativeCheck,
                          testing::Combine( testing::Values( 0.0, 0.15 ), testing::Bool() ) );

}  // namespace tribol
