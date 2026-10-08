// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <algorithm>
#include <cmath>
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

// static ContactSmoothing smoother( ContactParams{} );

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
  smoother_.bounds_from_projections( projs0.data(), p_.del, bounds0 );
  double smooth_bounds0[2];
  smoother_.smooth_bounds( bounds0, p_.del, smooth_bounds0 );
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
    smoother_.bounds_from_projections( projs0.data(), p_.del, bounds0 );
    double smooth_bounds0[2];
    smoother_.smooth_bounds( bounds0, p_.del, smooth_bounds0 );
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

TEST( NormalAngleSmoothingCheck, ShiftedCosineStartsAtConfiguredAngle )
{
  constexpr double pi = 3.14159265358979323846264338327950288;
  constexpr double start_angle = 0.25 * pi;
  const ContactParams default_params{};
  EXPECT_DOUBLE_EQ( default_params.normal_smoothing_start_angle, start_angle );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( -std::cos( pi / 6.0 ), start_angle ), -1.0 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( -std::cos( start_angle ), start_angle ), -1.0 );
  EXPECT_NEAR( ContactSmoothing::normal_alignment_factor( -std::cos( 3.0 * pi / 8.0 ), start_angle ),
               -1.0 / std::sqrt( 2.0 ), 1.0e-14 );
  EXPECT_NEAR( ContactSmoothing::normal_alignment_factor( -0.5, 0.0 ), -0.5, 1.0e-14 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( 0.0, start_angle ), 0.0 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( 0.5, start_angle ), 0.0 );
}

TEST( NormalAngleSmoothingCheck, NinetyDegreesDisablesAttenuation )
{
  constexpr double perpendicular = energy_mortar::perpendicular_normal_angle;
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( -1.0, perpendicular ), -1.0 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( -0.5, perpendicular ), -1.0 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( -1.0e-12, perpendicular ), -1.0 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( 0.0, perpendicular ), 0.0 );
  EXPECT_DOUBLE_EQ( ContactSmoothing::normal_alignment_factor( 0.5, perpendicular ), 0.0 );
}

TEST( EnergyMortarAutoContactCheck, RejectsWhenEitherDirectedGapExceedsPenetrationLimit )
{
  // The first normal sees excessive penetration, while the nearly perpendicular second normal sees separation. Both
  // directed gaps must independently satisfy the penetration limit for a self-contact pair to remain eligible.
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

  constexpr RealT max_penetration_fraction = 0.3;
  EXPECT_TRUE(
      detail::energyMortarExceedsMaxAutoInterpen( mesh1.getView(), mesh2.getView(), 0, 0, max_penetration_fraction ) );
  EXPECT_TRUE(
      detail::energyMortarExceedsMaxAutoInterpen( mesh2.getView(), mesh1.getView(), 0, 0, max_penetration_fraction ) );
}

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
  const double residual[4] = { 0.0, 0.2, 0.0, 0.0 };
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

TEST( EnergyMortarResidualGapCheck, DirectedPairUsesOnlyNonmortarResidualGap )
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
  params.normal_smoothing_start_angle = energy_mortar::perpendicular_normal_angle;
  params.residual_gap = 0.15;
  EnergyMortarCalculator evaluator( params );

  const double both_reduced[4] = { 0.0, 0.0, 0.0, 0.0 };
  const auto inactive = evaluator.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                         mesh2.getView(), both_reduced );
  EXPECT_FALSE( inactive.has_active_qp );

  const double mortar_only[4] = { 0.0, 0.0, 0.15, 0.15 };
  const auto still_inactive = evaluator.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                               mesh2.getView(), mortar_only );
  EXPECT_FALSE( still_inactive.has_active_qp );
  EXPECT_EQ( still_inactive.energy, 0.0 );

  const double nonmortar_full[4] = { 0.15, 0.15, 0.0, 0.0 };
  const auto active = evaluator.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                       mesh2.getView(), nonmortar_full );
  EXPECT_TRUE( active.has_active_qp );
  EXPECT_GT( active.energy, 0.0 );
}

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

  const double zero[4] = { 0.0, 0.0, 0.0, 0.0 };
  const double ramp[4] = { 0.0, 0.2, 0.0, 0.0 };
  const double mortar_ramp[4] = { 0.0, 0.0, 0.0, 0.2 };
  const double global_max[4] = { 0.2, 0.2, 0.2, 0.2 };
  const auto zero_projection =
      evaluator.compute_projection_bounds( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(), zero );
  const auto ramp_projection =
      evaluator.compute_projection_bounds( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(), ramp );
  const auto mortar_projection =
      evaluator.compute_projection_bounds( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(), mortar_ramp );
  const auto max_projection =
      evaluator.compute_projection_bounds( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(), global_max );

  EXPECT_NEAR( mortar_projection[0], zero_projection[0], 1.0e-14 );
  EXPECT_NEAR( mortar_projection[1], zero_projection[1], 1.0e-14 );
  EXPECT_TRUE( std::abs( ramp_projection[0] - max_projection[0] ) > 1.0e-6 ||
               std::abs( ramp_projection[1] - max_projection[1] ) > 1.0e-6 );
}

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
  const double zero[4] = { 0.0, 0.0, 0.0, 0.0 };
  const double residual[4] = { 0.25, 0.25, 0.25, 0.25 };

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
  const double residual[4] = { 0.0, 0.2, 0.0, 0.0 };
  const auto endpoint_data = endpointBallData( 0.0, -1.0 );

  RealT x2_inside[2] = { 0.30, 0.05 };
  RealT y2[2] = { 0.05, 0.05 };
  IndexT conn2[2] = { 0, 1 };
  MeshData inside_mesh( 1, 1, 2, conn2, LINEAR_EDGE, x2_inside, y2, nullptr, MemorySpace::Host );
  const auto inside = evaluator.compute_ball_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                           inside_mesh.getView(), residual, &endpoint_data );
  EXPECT_GT( inside.energy, 0.0 );

  RealT x2_outside[2] = { -0.05, -0.30 };
  MeshData outside_mesh( 1, 1, 2, conn2, LINEAR_EDGE, x2_outside, y2, nullptr, MemorySpace::Host );
  const auto outside = evaluator.compute_ball_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                            outside_mesh.getView(), residual, &endpoint_data );
  EXPECT_EQ( outside.energy, 0.0 );
  EXPECT_TRUE( std::all_of( outside.force.begin(), outside.force.end(), []( double value ) { return value == 0.0; } ) );
  EXPECT_TRUE(
      std::all_of( outside.stiffness.begin(), outside.stiffness.end(), []( double value ) { return value == 0.0; } ) );

  RealT x2_crossing[2] = { 0.10, -0.10 };
  MeshData crossing_mesh( 1, 1, 2, conn2, LINEAR_EDGE, x2_crossing, y2, nullptr, MemorySpace::Host );
  const auto crossing_ball = evaluator.compute_ball_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                  crossing_mesh.getView(), residual, &endpoint_data );
  const auto crossing_mortar = evaluator.compute_quadrature_point_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                                crossing_mesh.getView(), residual );
  EXPECT_GT( crossing_ball.energy, 0.0 );
  EXPECT_GT( crossing_mortar.energy, 0.0 );

  RealT x2_inside_half[2] = { 0.10, 0.0 };
  MeshData inside_half_mesh( 1, 1, 2, conn2, LINEAR_EDGE, x2_inside_half, y2, nullptr, MemorySpace::Host );
  const auto inside_half = evaluator.compute_ball_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                inside_half_mesh.getView(), residual, &endpoint_data );
  EXPECT_NEAR( crossing_ball.energy, inside_half.energy, 1.0e-14 );
}

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
  const double residual[4] = { 0.0, 0.2, 0.0, 0.0 };
  auto eligible = openEndBallData();
  auto ineligible = eligible;
  ineligible.weight = { 1.0, 0.0 };
  auto half_weight = eligible;
  half_weight.weight[1] = 0.5;

  const auto active =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual, &eligible );
  ASSERT_TRUE( active.has_active_qp );
  ASSERT_GT( active.energy, 0.0 );

  const auto missing_data = evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual );
  EXPECT_FALSE( missing_data.has_active_qp );
  EXPECT_DOUBLE_EQ( missing_data.energy, 0.0 );

  const auto inactive =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual, &ineligible );
  EXPECT_FALSE( inactive.has_active_qp );
  EXPECT_DOUBLE_EQ( inactive.energy, 0.0 );

  const auto weighted =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual, &half_weight );
  EXPECT_NEAR( weighted.energy, 0.5 * active.energy, 1.0e-14 );
}

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
  const double perpendicular_angle = energy_mortar::perpendicular_normal_angle;
  params.normal_smoothing_start_angle = 0.5 * perpendicular_angle;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  EnergyMortarCalculator evaluator( params );
  auto unsmoothed_params = params;
  unsmoothed_params.normal_smoothing_start_angle = perpendicular_angle;
  EnergyMortarCalculator unsmoothed_evaluator( unsmoothed_params );
  const InterfacePair pair( 0, 0 );
  const double residual[4] = { 0.2, 0.2, 0.0, 0.0 };
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
    return calculator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual, &endpoint_data )
        .energy;
  };

  const double opposed_energy = ball_energy_at_angle( evaluator, 0.0 );
  ASSERT_GT( opposed_energy, 0.0 );

  const double ramp_angle = 0.75 * perpendicular_angle;
  const double ramp_alignment =
      ContactSmoothing::normal_alignment_factor( -std::cos( ramp_angle ), params.normal_smoothing_start_angle );
  const double unsmoothed_ramp_energy = ball_energy_at_angle( unsmoothed_evaluator, ramp_angle );
  EXPECT_NEAR( ball_energy_at_angle( evaluator, ramp_angle ), ramp_alignment * ramp_alignment * unsmoothed_ramp_energy,
               1.0e-12 );

  const double near_perpendicular_angle = 89.0 * perpendicular_angle / 90.0;
  const double near_perpendicular_alignment = ContactSmoothing::normal_alignment_factor(
      -std::cos( near_perpendicular_angle ), params.normal_smoothing_start_angle );
  const double near_perpendicular_energy = ball_energy_at_angle( evaluator, near_perpendicular_angle );
  const double unsmoothed_near_perpendicular_energy =
      ball_energy_at_angle( unsmoothed_evaluator, near_perpendicular_angle );
  EXPECT_NEAR( near_perpendicular_energy,
               near_perpendicular_alignment * near_perpendicular_alignment * unsmoothed_near_perpendicular_energy,
               1.0e-12 );
  EXPECT_LT( near_perpendicular_energy, ball_energy_at_angle( evaluator, ramp_angle ) );

  EXPECT_NEAR( ball_energy_at_angle( evaluator, perpendicular_angle ), 0.0, 1.0e-30 );
  EXPECT_DOUBLE_EQ( ball_energy_at_angle( evaluator, 2.0 * perpendicular_angle ), 0.0 );
}

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
  const double residual[4] = { 0.0, 0.2, 0.0, 0.0 };
  const auto endpoint_data = openEndBallData();

  const auto coarse_data = evaluator.compute_ball_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(),
                                                                coarse.getView(), residual, &endpoint_data );
  double refined_energy = 0.0;
  for ( int edge = 0; edge < 2; ++edge ) {
    refined_energy += evaluator
                          .compute_ball_penalty_data( InterfacePair( 0, edge ), mesh1.getView(), fine.getView(),
                                                      residual, &endpoint_data )
                          .energy;
  }
  // The radial integrand is non-polynomial, so independently mapping the same three-point rule to the two refined
  // edges produces a small quadrature difference.
  EXPECT_NEAR( refined_energy, coarse_data.energy, 2.0e-6 );
}

TEST( EnergyMortarResidualGapCheck, BallCompletionScalesWithSlaveTributaryLength )
{
  RealT coarse_x[2] = { -1.0, 0.0 };
  RealT refined_x[2] = { -0.5, 0.0 };
  RealT slave_y[2] = { 0.0, 0.0 };
  IndexT slave_conn[2] = { 0, 1 };
  MeshData coarse_slave( 0, 1, 2, slave_conn, LINEAR_EDGE, coarse_x, slave_y, nullptr, MemorySpace::Host );
  MeshData refined_slave( 0, 1, 2, slave_conn, LINEAR_EDGE, refined_x, slave_y, nullptr, MemorySpace::Host );

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
  const double residual[4] = { 0.0, 0.2, 0.0, 0.0 };
  const auto endpoint_data = openEndBallData();
  const double coarse_energy = evaluator
                                   .compute_ball_penalty_data( InterfacePair( 0, 0 ), coarse_slave.getView(),
                                                               mortar.getView(), residual, &endpoint_data )
                                   .energy;
  const double refined_energy = evaluator
                                    .compute_ball_penalty_data( InterfacePair( 0, 0 ), refined_slave.getView(),
                                                                mortar.getView(), residual, &endpoint_data )
                                    .energy;
  ASSERT_GT( coarse_energy, 0.0 );
  EXPECT_NEAR( refined_energy, 0.5 * coarse_energy, 1.0e-14 );
}

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
  const double residual[4] = { 0.0, 0.2, 0.0, 0.0 };
  auto endpoint_data = endpointBallData( 0.0, -1.0 );
  const auto analytical =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual, &endpoint_data );
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
        evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual, &endpoint_data ).energy;
    restore();
    perturb( dof, -grad_eps );
    const double minus =
        evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual, &endpoint_data ).energy;
    EXPECT_NEAR( analytical.force[dof], ( plus - minus ) / ( 2.0 * grad_eps ), 1.0e-6 ) << "dof " << dof;
  }

  constexpr double hess_eps = 1.0e-6;
  for ( int col = 0; col < BallPenaltyData::num_force_dofs; ++col ) {
    restore();
    perturb( col, hess_eps );
    const auto plus =
        evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual, &endpoint_data ).force;
    restore();
    perturb( col, -hess_eps );
    const auto minus =
        evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual, &endpoint_data ).force;
    for ( int row = 0; row < BallPenaltyData::num_force_dofs; ++row ) {
      EXPECT_NEAR( analytical.stiffness[row * BallPenaltyData::num_force_dofs + col],
                   ( plus[row] - minus[row] ) / ( 2.0 * hess_eps ), 2.0e-4 )
          << "row " << row << ", col " << col;
    }
  }
  restore();
}

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
  const double residual[4] = { 0.0, 0.2, 0.0, 0.0 };
  const auto endpoint_data = endpointBallData( 0.0, -1.0 );
  auto norm = []( const auto& values ) {
    double norm_sq = 0.0;
    for ( double value : values ) {
      norm_sq += value * value;
    }
    return std::sqrt( norm_sq );
  };

  const auto finite_overlap =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual, &endpoint_data );
  ASSERT_GT( finite_overlap.energy, 0.0 );
  x2[0] = 1.0e-6;
  mesh2.setPosition( x2, y2, nullptr );
  const auto vanishing_overlap =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual, &endpoint_data );
  ASSERT_GT( vanishing_overlap.energy, 0.0 );
  EXPECT_LT( vanishing_overlap.energy, 1.0e-2 * finite_overlap.energy );

  x2[0] = 0.0;
  mesh2.setPosition( x2, y2, nullptr );
  const auto zero_overlap =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual, &endpoint_data );
  EXPECT_DOUBLE_EQ( zero_overlap.energy, 0.0 );
  EXPECT_DOUBLE_EQ( norm( zero_overlap.force ), 0.0 );
  EXPECT_DOUBLE_EQ( norm( zero_overlap.stiffness ), 0.0 );
}

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
  const double residual[4] = { 0.0, 0.2, 0.0, 0.0 };
  const auto endpoint_data = endpointBallData( 0.0, -1.0 );
  auto norm = []( const auto& values ) {
    double norm_sq = 0.0;
    for ( double value : values ) {
      norm_sq += value * value;
    }
    return std::sqrt( norm_sq );
  };

  const auto finite_overlap =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual, &endpoint_data );
  ASSERT_GT( finite_overlap.energy, 0.0 );
  const double finite_force_norm = norm( finite_overlap.force );
  const double finite_stiffness_norm = norm( finite_overlap.stiffness );
  ASSERT_GT( finite_force_norm, 0.0 );
  ASSERT_GT( finite_stiffness_norm, 0.0 );

  x2[0] = 1.0e-6;
  mesh2.setPosition( x2, y2, nullptr );
  const auto vanishing_overlap =
      evaluator.compute_ball_penalty_data( pair, mesh1.getView(), mesh2.getView(), residual, &endpoint_data );
  ASSERT_GT( vanishing_overlap.energy, 0.0 );
  EXPECT_LT( norm( vanishing_overlap.force ), 1.0e-2 * finite_force_norm );
  EXPECT_LT( norm( vanishing_overlap.stiffness ), 1.0e-2 * finite_stiffness_norm );
}

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
  const double residual[4] = { 0.0, 0.2, 0.0, 0.0 };
  auto endpoint_data = endpointBallData( 0.0, -1.0 );
  const auto base = evaluator.compute_ball_penalty_data( InterfacePair( 0, 0 ), mesh1.getView(), mesh2.getView(),
                                                         residual, &endpoint_data );
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
                                                               residual, &endpoint_data );
  EXPECT_NEAR( translated.energy, base.energy, 1.0e-12 );
  for ( int i = 0; i < BallPenaltyData::num_force_dofs; ++i ) {
    EXPECT_NEAR( translated.force[i], base.force[i], 1.0e-9 );
  }
}

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

TEST( EnergyMortarResidualGapCheck, VirtualGeometryRespectsNormalSmoothing )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

  constexpr double pi = 3.14159265358979323846264338327950288;
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

  // For oblique edges, offsetting the endpoints changes the projected interval and its tributary-area split as well as
  // the gap. This verifies that enforcement uses the same virtual geometry as projection instead of subtracting a gap
  // from values integrated over the physical edge.
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

  constexpr double pi = 3.14159265358979323846264338327950288;
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
  ContactParams params_;
  params_.del = 0.1;  // smoothing m
  params_.k = 1.0;    // penalty stiffness
  params_.N = 3;      // quadrature points
  params_.enzyme_quadrature = std::get<1>( GetParam() );
  params_.residual_gap = std::get<0>( GetParam() );

  EnergyMortarCalculator evaluator_( params_ );

  // ── Run validation ───────────────────────────────────────────────────────
  const double epsilon = 1e-7;

  RealT y1_plus[2] = { epsilon, 0.0 };
  RealT y1_minus[2] = { -epsilon, 0.0 };
  mesh1.setPosition( x1, y1_plus, nullptr );
  mesh1.setPosition( x1, y1_minus, nullptr );
  mesh1.setPosition( x1, y1, nullptr );
  auto result = evaluator_.validate_g_tilde( pair, mesh1, mesh2, epsilon );

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

TEST_P( ResidualGapDerivativeCheck, GtildeHessianFDvsAD )
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
  ContactParams params_;
  params_.del = 0.1;
  params_.k = 1.0;
  params_.N = 3;
  params_.enzyme_quadrature = std::get<1>( GetParam() );
  params_.residual_gap = std::get<0>( GetParam() );

  EnergyMortarCalculator evaluator_( params_ );

  // ── Run validation ───────────────────────────────────────────────────────
  const double epsilon = 1e-5;
  auto result = evaluator_.validate_hessian( pair, mesh1, mesh2, epsilon );

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

INSTANTIATE_TEST_SUITE_P( FixedAndGeometryDependentQuadrature, ResidualGapDerivativeCheck,
                          testing::Combine( testing::Values( 0.0, 0.15 ), testing::Bool() ) );

}  // namespace tribol
