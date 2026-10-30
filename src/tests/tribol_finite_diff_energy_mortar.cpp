// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <type_traits>
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
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/interface/tribol.hpp"

// Checks local Energy Mortar response and finite-difference derivatives. Selected-rule coverage includes one, two,
// and three points with frozen and varying integration-surface quadrature. Invalid geometry must report an error for
// values and derivatives; portable rule values are tested separately.
namespace tribol {

using LinearEdgeBasis = BasisFunction<EDGE, LINEAR_LAGRANGE>;
using LinearEdgePointView = MdSpan<const double, LinearEdgeBasis::spatial_dimension>;
using LinearEdgeCoords = MdArray<double, LinearEdgeBasis::node_count, LinearEdgeBasis::spatial_dimension>;

inline void endpoints( const MeshData::Viewer& mesh, int elem_id, double P0[2], double P1[2] )
{
  double P0_P1[4];
  mesh.getFaceCoords( elem_id, P0_P1 );
  P0[0] = P0_P1[0];
  P0[1] = P0_P1[1];
  P1[0] = P0_P1[2];
  P1[1] = P0_P1[3];
}

template <typename QuadratureRule>
std::pair<double, double> EnergyMortarCalculator<QuadratureRule>::eval_gtilde( const InterfacePair& pair,
                                                                               const MeshData::Viewer& mesh1,
                                                                               const MeshData::Viewer& mesh2 ) const
{
  NodalContactData ncd = compute_nodal_contact_data( pair, mesh1, mesh2 );
  // double gt1 = ncd.g_tilde[0];
  // double gt2 = ncd.g_tilde[1];
  double A1 = ncd.g_tilde[0];
  double A2 = ncd.g_tilde[1];

  return { A1, A2 };
}

template <typename QuadratureRule>
FiniteDiffResult EnergyMortarCalculator<QuadratureRule>::validate_g_tilde( const InterfacePair& pair, MeshData& mesh1,
                                                                           MeshData& mesh2, double epsilon ) const
{
  FiniteDiffResult result;

  auto viewer1 = mesh1.getView();
  auto viewer2 = mesh2.getView();

  IntegrationSurfaceRule qp0{};
  if ( !p_.enzyme_quadrature ) qp0 = generateIntegrationSurfaceRule( pair, viewer1, viewer2 );

  auto [g1_base, g2_base] = eval_gtilde( pair, viewer1, viewer2 );

  result.g_tilde1_baseline = g1_base;
  result.g_tilde2_baseline = g2_base;

  auto A_conn = viewer1.getConnectivity()( static_cast<std::size_t>( pair.element_id1_ ) );
  auto B_conn = viewer2.getConnectivity()( static_cast<std::size_t>( pair.element_id2_ ) );

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

template <typename QuadratureRule>
std::pair<double, double> EnergyMortarCalculator<QuadratureRule>::eval_gtilde_fixed_qp(
    const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
    const IntegrationSurfaceRule& qp_fixed ) const
{
  double A0[2], A1[2];
  endpoints( mesh1, pair.element_id1_, A0, A1 );

  const LinearEdgeCoords edge_coords{ LinearEdgePointView{ A0 }, LinearEdgePointView{ A1 } };
  const double J = ComputeJacobianWeight<LinearEdgeBasis>( 0.0, edge_coords.view() );

  double gt1 = 0.0, gt2 = 0.0;

  for ( IndexT i = 0; i < qp_fixed.result.required_point_count; ++i ) {
    const double xiA = qp_fixed.coords( i, 0 );
    const double w = qp_fixed.weights[i];

    double basis_values[LinearEdgeBasis::node_count];
    LinearEdgeBasis::Eval(
        MdSpan<const double, LinearEdgeBasis::parametric_dimension>{ &xiA, LinearEdgeBasis::parametric_dimension },
        basis_values );

    const double gn = compute_weighted_normal_gap( pair, mesh1, mesh2, xiA );

    gt1 += w * basis_values[0] * J * gn;
    gt2 += w * basis_values[1] * J * gn;
  }

  return { gt1, gt2 };
}

template <typename QuadratureRule>
FiniteDiffResult EnergyMortarCalculator<QuadratureRule>::validate_hessian( const InterfacePair& pair, MeshData& mesh1,
                                                                           MeshData& mesh2, double epsilon ) const
{
  FiniteDiffResult result;

  auto viewer1 = mesh1.getView();
  auto viewer2 = mesh2.getView();

  double hess1[64] = { 0.0 };
  double hess2[64] = { 0.0 };

  const int ndof = 8;
  result.fd_gradient_g1.assign( ndof * ndof, 0.0 );
  result.fd_gradient_g2.assign( ndof * ndof, 0.0 );

  auto A_conn = viewer1.getConnectivity()( static_cast<std::size_t>( pair.element_id1_ ) );
  auto B_conn = viewer2.getConnectivity()( static_cast<std::size_t>( pair.element_id2_ ) );

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

  IntegrationSurfaceRule qp0{};
  if ( !p_.enzyme_quadrature ) qp0 = generateIntegrationSurfaceRule( pair, viewer1, viewer2 );

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

TEST( QuadraturePointPenaltyCheck, OpenGapIsInactive )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.1, 0.1 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.k = 3.0;
  params.enzyme_quadrature = true;

  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  EnergyMortarCalculator evaluator( params, quadrature_rule );
  const auto result =
      evaluator.compute_quadrature_point_penalty_data( InterfacePair{ 0, 0 }, mesh1.getView(), mesh2.getView() );
  EXPECT_FALSE( result.has_active_qp );
  EXPECT_EQ( result.energy, 0.0 );
  EXPECT_TRUE( std::all_of( result.force.begin(), result.force.end(), []( double force ) { return force == 0.0; } ) );
  EXPECT_TRUE( std::all_of( result.stiffness.begin(), result.stiffness.end(),
                            []( double stiffness ) { return stiffness == 0.0; } ) );
}

TEST( QuadraturePointPenaltyCheck, ZeroGapRetainsActiveTangent )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.0, 0.0 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.k = 3.0;
  params.enzyme_quadrature = true;

  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  EnergyMortarCalculator evaluator( params, quadrature_rule );
  const auto result =
      evaluator.compute_quadrature_point_penalty_data( InterfacePair{ 0, 0 }, mesh1.getView(), mesh2.getView() );

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
  MeshData mesh1( 0, 1, 2, conn1, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.1, 0.1 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.k = 3.0;
  params.enzyme_quadrature = true;

  double gap_without_residual[2] = { 0.0, 0.0 };
  double tributary_area_without_residual[2] = { 0.0, 0.0 };
  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  EnergyMortarCalculator evaluator_without_residual( params, quadrature_rule );
  evaluator_without_residual.compute_gtilde_and_area( InterfacePair{ 0, 0 }, mesh1.getView(), mesh2.getView(),
                                                      gap_without_residual, tributary_area_without_residual );

  params.residual_gap = 0.15;
  double gap_with_residual[2] = { 0.0, 0.0 };
  double tributary_area_with_residual[2] = { 0.0, 0.0 };
  EnergyMortarCalculator evaluator_with_residual( params, quadrature_rule );
  evaluator_with_residual.compute_gtilde_and_area( InterfacePair{ 0, 0 }, mesh1.getView(), mesh2.getView(),
                                                   gap_with_residual, tributary_area_with_residual );

  // The residual gap shifts the gap integral without changing the projected overlap geometry.
  for ( int i = 0; i < 2; ++i ) {
    EXPECT_NEAR( tributary_area_with_residual[i], tributary_area_without_residual[i], 1.0e-14 );
    EXPECT_NEAR( gap_with_residual[i],
                 gap_without_residual[i] - params.residual_gap * tributary_area_without_residual[i], 1.0e-14 );
  }
}

TEST( EnergyMortarResidualGapCheck, QuadraturePointOpenGapBecomesActive )
{
  // The edges have a normal separation of 0.1; a residual gap of 0.15 produces an effective gap of -0.05.
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.1, 0.1 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.k = 3.0;
  params.enzyme_quadrature = true;

  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  EnergyMortarCalculator evaluator_without_residual( params, quadrature_rule );
  const auto inactive = evaluator_without_residual.compute_quadrature_point_penalty_data(
      InterfacePair{ 0, 0 }, mesh1.getView(), mesh2.getView() );
  EXPECT_FALSE( inactive.has_active_qp );
  EXPECT_EQ( inactive.energy, 0.0 );
  EXPECT_TRUE(
      std::all_of( inactive.force.begin(), inactive.force.end(), []( double force ) { return force == 0.0; } ) );
  EXPECT_TRUE( std::all_of( inactive.stiffness.begin(), inactive.stiffness.end(),
                            []( double stiffness ) { return stiffness == 0.0; } ) );

  params.residual_gap = 0.15;
  EnergyMortarCalculator evaluator_with_residual( params, quadrature_rule );
  const auto active = evaluator_with_residual.compute_quadrature_point_penalty_data( InterfacePair{ 0, 0 },
                                                                                     mesh1.getView(), mesh2.getView() );
  EXPECT_TRUE( active.has_active_qp );
  EXPECT_GT( active.energy, 0.0 );
  EXPECT_TRUE( std::any_of( active.force.begin(), active.force.end(), []( double force ) { return force != 0.0; } ) );
  EXPECT_TRUE( std::any_of( active.stiffness.begin(), active.stiffness.end(),
                            []( double stiffness ) { return stiffness != 0.0; } ) );
}

class ResidualGapDerivativeCheck : public ::testing::TestWithParam<double> {};

TEST_P( ResidualGapDerivativeCheck, QuadraturePointPenaltyDerivativesMatchFiniteDifference )
{
  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };
  MeshData mesh1( 0, 1, 2, conn1, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { -0.1, -0.1 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );

  ContactParams params;
  params.k = 3.0;
  params.enzyme_quadrature = true;
  params.residual_gap = GetParam();

  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  EnergyMortarCalculator evaluator( params, quadrature_rule );
  const InterfacePair pair{ 0, 0 };
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
  // ── Geometry: two facing EDGE segments ────────────────────────────
  // Segment A: (0,0) -> (1,0)
  // Segment B: (0.2, 0.5) -> (0.8, 0.5)

  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 1, 0 };  // reversed so nA points toward B
  MeshData mesh1( 0, 1, 2, conn1, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.5, 0.5 };
  IndexT conn2[2] = { 0, 1 };
  MeshData mesh2( 1, 1, 2, conn2, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );

  InterfacePair pair{ 0, 0 };

  // ── Evaluator setup ──────────────────────────────────────────────────────
  ContactParams params_;
  params_.k = 1.0;                   // penalty stiffness
  params_.enzyme_quadrature = true;  // use the non-Enzyme quadrature path
  params_.residual_gap = GetParam();

  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  EnergyMortarCalculator evaluator_( params_, quadrature_rule );

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
  // ── Geometry: two facing EDGE segments ────────────────────────────
  // Segment A: (0,0) -> (1,0)
  // Segment B: (0.2, 0.5) -> (0.8, 0.5)

  RealT x1[2] = { 0.0, 1.0 };
  RealT y1[2] = { 0.0, 0.0 };

  RealT x2[2] = { 0.2, 0.8 };
  RealT y2[2] = { 0.5, 0.5 };

  IndexT conn1[2] = { 1, 0 };
  IndexT conn2[2] = { 0, 1 };

  MeshData mesh1( 0, 1, 2, conn1, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );
  MeshData mesh2( 1, 1, 2, conn2, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );

  InterfacePair pair{ 0, 0 };

  // ── Evaluator setup ──────────────────────────────────────────────────────
  ContactParams params_;
  params_.k = 1.0;
  params_.enzyme_quadrature = true;
  params_.residual_gap = GetParam();

  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  EnergyMortarCalculator evaluator_( params_, quadrature_rule );

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

INSTANTIATE_TEST_SUITE_P( ZeroAndNonzeroResidualGap, ResidualGapDerivativeCheck, testing::Values( 0.0, 0.15 ) );

namespace {

// Disjoint projected intervals must produce no nodal contribution or active penalty points.
TEST( EnergyMortarCoordinates, disjoint_projected_intervals_have_zero_contribution )
{
  const RealT x1[] = { 0.0, 1.0 }, y1[] = { 0.0, 0.0 };
  const RealT x2[] = { 3.0, 2.0 }, y2[] = { 0.1, 0.1 };
  const IndexT connectivity[] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );
  MeshData mesh2( 1, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );
  const ContactParams params{ 1.0, true, 0.0 };
  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule;
  const EnergyMortarCalculator evaluator( params, quadrature_rule );
  NodalContactData nodal{};
  evaluator.compute_gtilde_and_area( { 0, 0 }, mesh1.getView(), mesh2.getView(), nodal.g_tilde.data(),
                                     nodal.AI.data() );
  for ( int node = 0; node < LinearEdgeBasis::node_count; ++node ) {
    EXPECT_EQ( nodal.g_tilde[node], 0.0 );
    EXPECT_EQ( nodal.AI[node], 0.0 );
  }
  const auto penalty = evaluator.compute_quadrature_point_penalty_data( { 0, 0 }, mesh1.getView(), mesh2.getView() );
  EXPECT_FALSE( penalty.has_active_qp );
  EXPECT_EQ( penalty.energy, 0.0 );
  for ( const double force : penalty.force ) EXPECT_EQ( force, 0.0 );
  for ( const double stiffness : penalty.stiffness ) EXPECT_EQ( stiffness, 0.0 );
}

// Calling the calculator without search must give no gap, area, or penalty contribution for co-oriented edges.
TEST( EnergyMortarCoordinates, cooriented_pairs_have_zero_contribution_without_search )
{
  const RealT x1[] = { 0.0, 1.0 }, y1[] = { 0.0, 0.0 };
  const RealT x2[] = { 0.0, 1.0 }, y2[] = { 0.1, 0.1 };
  const IndexT connectivity[] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );
  MeshData mesh2( 1, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );
  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule;
  // A positive residual gap must not turn an excluded pair into an active constraint.
  for ( const RealT residual_gap : { 0.0, 0.15 } ) {
    SCOPED_TRACE( residual_gap );
    const EnergyMortarCalculator calculator( ContactParams{ 1.0, true, residual_gap }, quadrature_rule );
    NodalContactData nodal{};
    calculator.compute_gtilde_and_area( { 0, 0 }, mesh1.getView(), mesh2.getView(), nodal.g_tilde.data(),
                                        nodal.AI.data() );
    for ( int node = 0; node < LinearEdgeBasis::node_count; ++node ) {
      EXPECT_EQ( nodal.g_tilde[node], 0.0 );
      EXPECT_EQ( nodal.AI[node], 0.0 );
    }
    const auto penalty = calculator.compute_quadrature_point_penalty_data( { 0, 0 }, mesh1.getView(), mesh2.getView() );
    EXPECT_FALSE( penalty.has_active_qp );
    EXPECT_EQ( penalty.energy, 0.0 );
    for ( const auto force : penalty.force ) EXPECT_EQ( force, 0.0 );
    for ( const auto stiffness : penalty.stiffness ) EXPECT_EQ( stiffness, 0.0 );
  }
}

// Either degenerate edge must report an error for nodal values, derivatives, and penalty quantities in both modes.
TEST( EnergyMortarCoordinates, rejects_degenerate_geometry_before_consuming_quadrature )
{
  const auto check = []<int PointCount>() {
    const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<PointCount>> quadrature_rule;
    constexpr int coordinate_count = QuadraturePointPenaltyData::num_force_dofs;
    constexpr int stiffness_count = QuadraturePointPenaltyData::num_stiffness_entries;
    // Test coincident endpoints and a nonzero edge below the mapping's accepted length.
    for ( const RealT edge_length : { 0.0, 5.0e-15 } ) {
      for ( const bool collapse_mesh1 : { false, true } ) {
        for ( const bool varying_quadrature : { false, true } ) {
          SCOPED_TRACE( testing::Message()
                        << "points=" << PointCount << ", length=" << edge_length
                        << ", collapse_mesh1=" << collapse_mesh1 << ", varying_quadrature=" << varying_quadrature );
          const RealT x1[] = { 0.0, collapse_mesh1 ? edge_length : 1.0 }, y1[] = { 0.0, 0.0 };
          const RealT x2[] = { collapse_mesh1 ? 1.0 : edge_length, 0.0 }, y2[] = { 0.1, 0.1 };
          const IndexT connectivity[] = { 0, 1 };
          MeshData mesh1( 0, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );
          MeshData mesh2( 1, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );
          const auto view1 = mesh1.getView();
          const auto view2 = mesh2.getView();
          const InterfacePair pair{ 0, 0 };
          const EnergyMortarCalculator calculator( ContactParams{ 1.0, varying_quadrature, 0.0 }, quadrature_rule );
          NodalContactData nodal{};
          MdArray<double, coordinate_count> gradient1{}, gradient2{};
          MdArray<double, stiffness_count> hessian1{}, hessian2{};
          EXPECT_THROW( calculator.compute_gtilde_and_area( pair, view1, view2, nodal.g_tilde.data(), nodal.AI.data() ),
                        std::invalid_argument );
          EXPECT_THROW( calculator.grad_gtilde( pair, view1, view2, gradient1.data(), gradient2.data() ),
                        std::invalid_argument );
          EXPECT_THROW( calculator.grad_trib_area( pair, view1, view2, gradient1.data(), gradient2.data() ),
                        std::invalid_argument );
          EXPECT_THROW( calculator.d2_g2tilde( pair, view1, view2, hessian1.data(), hessian2.data() ),
                        std::invalid_argument );
          EXPECT_THROW( calculator.compute_d2A_d2u( pair, view1, view2, hessian1.data(), hessian2.data() ),
                        std::invalid_argument );
          EXPECT_THROW( calculator.compute_quadrature_point_penalty_energy( pair, view1, view2 ),
                        std::invalid_argument );
          EXPECT_THROW( calculator.compute_quadrature_point_penalty_data( pair, view1, view2 ), std::invalid_argument );
        }
      }
    }
  };
  check.template operator()<1>();
  check.template operator()<2>();
  check.template operator()<3>();
}

// A short valid edge must retain its normal distance; small edge length does not imply zero separation.
TEST( EnergyMortarGap, calculates_normal_distance_for_short_valid_edges )
{
  constexpr RealT edge_length = 1.0e-13;
  constexpr RealT separation = 0.25;
  constexpr double tolerance = 1.0e-12;
  const IndexT connectivity[] = { 0, 1 };
  RealT x1[] = { 0.0, edge_length }, y1[] = { 0.0, 0.0 };
  RealT x2[] = { edge_length, 0.0 }, y2[] = { separation, separation };
  MeshData mesh1( 0, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );
  MeshData mesh2( 1, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );
  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  const EnergyMortarCalculator calculator( ContactParams{ 1.0, true, 0.0 }, quadrature_rule );
  NodalContactData response{};
  calculator.compute_gtilde_and_area( { 0, 0 }, mesh1.getView(), mesh2.getView(), response.g_tilde.data(),
                                      response.AI.data() );
  for ( int node = 0; node < LinearEdgeBasis::node_count; ++node ) {
    ASSERT_GT( response.AI[node], 0.0 );
    EXPECT_NEAR( response.g_tilde[node] / response.AI[node], -separation, tolerance );
  }
}

// Perturb both endpoints in x and y so finite differences include edge-length and overlap-bound changes.
template <int Count>
void checkSelectedRuleDerivatives( bool varying )
{
  using QuadratureRule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<Count>>;
  using Calculator = EnergyMortarCalculator<QuadratureRule>;
  static_assert( !std::is_constructible_v<Calculator, const ContactParams&, QuadratureRule&&>,
                 "A calculator must not retain a reference to a temporary rule generator." );
  constexpr int dofs = 2 * LinearEdgeBasis::node_count * LinearEdgeBasis::spatial_dimension;
  constexpr double step = 1.0e-4;
  constexpr double gradient_tolerance = 2.0e-7;
  constexpr double hessian_tolerance = 2.0e-6;
  SCOPED_TRACE( testing::Message() << "points=" << Count << ", varying=" << varying );
  RealT x1[] = { 0.0, 1.7 }, y1[] = { 0.0, 0.1 };
  RealT x2[] = { 1.6, 0.25 }, y2[] = { 0.25, 0.15 };
  const IndexT connectivity[] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );
  MeshData mesh2( 1, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );
  std::array<RealT*, dofs> coords{ &x1[0], &y1[0], &x1[1], &y1[1], &x2[0], &y2[0], &x2[1], &y2[1] };
  std::array<RealT, dofs> baseline;
  for ( int i = 0; i < dofs; ++i ) baseline[i] = *coords[i];
  const QuadratureRule quadrature_rule{ SmoothedProjectedEdgeMapping{ 0.2 } };
  const Calculator calculator( ContactParams{ 7.0, varying, 0.05 }, quadrature_rule );
  const LinearEdgeCoords initial1{ LinearEdgePointView{ baseline.data(), 2 },
                                   LinearEdgePointView{ baseline.data() + 2, 2 } };
  const LinearEdgeCoords initial2{ LinearEdgePointView{ baseline.data() + 4, 2 },
                                   LinearEdgePointView{ baseline.data() + 6, 2 } };
  const auto integration_surface_rule =
      quadrature_rule.generateIntegrationSurfaceRule( initial1.view(), initial2.view() );
  const auto evaluate = [&]() {
    std::array<double, 4> values{};
    if ( varying ) {
      calculator.compute_gtilde_and_area( { 0, 0 }, mesh1.getView(), mesh2.getView(), values.data(),
                                          values.data() + 2 );
    } else {
      const auto gaps =
          calculator.eval_gtilde_fixed_qp( { 0, 0 }, mesh1.getView(), mesh2.getView(), integration_surface_rule );
      values[0] = gaps.first;
      values[1] = gaps.second;
      const double dx = x1[1] - x1[0], dy = y1[1] - y1[0];
      const double jacobian = 0.5 * std::sqrt( dx * dx + dy * dy );
      for ( int q = 0; q < Count; ++q ) {
        values[2] +=
            0.5 * ( 1.0 - integration_surface_rule.coords( q, 0 ) ) * integration_surface_rule.weights[q] * jacobian;
        values[3] +=
            0.5 * ( 1.0 + integration_surface_rule.coords( q, 0 ) ) * integration_surface_rule.weights[q] * jacobian;
      }
    }
    return values;
  };
  std::array<std::array<double, dofs>, 4> gradient{};
  std::array<std::array<double, dofs * dofs>, 4> hessian{};
  calculator.grad_gtilde( { 0, 0 }, mesh1.getView(), mesh2.getView(), gradient[0].data(), gradient[1].data() );
  calculator.grad_trib_area( { 0, 0 }, mesh1.getView(), mesh2.getView(), gradient[2].data(), gradient[3].data() );
  calculator.d2_g2tilde( { 0, 0 }, mesh1.getView(), mesh2.getView(), hessian[0].data(), hessian[1].data() );
  calculator.compute_d2A_d2u( { 0, 0 }, mesh1.getView(), mesh2.getView(), hessian[2].data(), hessian[3].data() );
  const auto original = evaluate();
  for ( int row = 0; row < dofs; ++row ) {
    *coords[row] = baseline[row] + step;
    const auto plus = evaluate();
    *coords[row] = baseline[row] - step;
    const auto minus = evaluate();
    *coords[row] = baseline[row];
    for ( std::size_t output = 0; output < gradient.size(); ++output ) {
      EXPECT_NEAR( gradient[output][row], ( plus[output] - minus[output] ) / ( 2.0 * step ), gradient_tolerance );
      EXPECT_NEAR( hessian[output][row * dofs + row],
                   ( plus[output] - 2.0 * original[output] + minus[output] ) / ( step * step ), hessian_tolerance );
    }
    for ( int col = row + 1; col < dofs; ++col ) {
      std::array<std::array<double, 4>, 4> samples{};
      int sample = 0;
      for ( const double row_sign : { 1.0, -1.0 } ) {
        for ( const double col_sign : { 1.0, -1.0 } ) {
          *coords[row] = baseline[row] + row_sign * step;
          *coords[col] = baseline[col] + col_sign * step;
          samples[sample++] = evaluate();
        }
      }
      *coords[row] = baseline[row];
      *coords[col] = baseline[col];
      for ( std::size_t output = 0; output < hessian.size(); ++output ) {
        const double expected = ( samples[0][output] - samples[1][output] - samples[2][output] + samples[3][output] ) /
                                ( 4.0 * step * step );
        EXPECT_NEAR( hessian[output][row * dofs + col], expected, hessian_tolerance );
        EXPECT_NEAR( hessian[output][col * dofs + row], expected, hessian_tolerance );
      }
    }
  }
  if ( !varying ) return;
  // Penalty gradients and Hessians must use the same selected rule, including its geometry-dependent weights.
  const auto penalty = calculator.compute_quadrature_point_penalty_data( { 0, 0 }, mesh1.getView(), mesh2.getView() );
  ASSERT_TRUE( penalty.has_active_qp );
  for ( int col = 0; col < dofs; ++col ) {
    *coords[col] = baseline[col] + step;
    const auto plus = calculator.compute_quadrature_point_penalty_data( { 0, 0 }, mesh1.getView(), mesh2.getView() );
    *coords[col] = baseline[col] - step;
    const auto minus = calculator.compute_quadrature_point_penalty_data( { 0, 0 }, mesh1.getView(), mesh2.getView() );
    *coords[col] = baseline[col];
    EXPECT_NEAR( penalty.force[col], ( plus.energy - minus.energy ) / ( 2.0 * step ), gradient_tolerance );
    for ( int row = 0; row < dofs; ++row ) {
      EXPECT_NEAR( penalty.stiffness[row * dofs + col], ( plus.force[row] - minus.force[row] ) / ( 2.0 * step ),
                   hessian_tolerance );
    }
  }
}
}  // namespace

// All point counts must preserve nodal gap, area, and penalty derivatives when geometry and edge length change.
TEST( EnergyMortarQuadrature, selected_point_counts_preserve_gradients_and_hessians )
{
  for ( const bool varying : { false, true } ) {
    checkSelectedRuleDerivatives<1>( varying );
    checkSelectedRuleDerivatives<2>( varying );
    checkSelectedRuleDerivatives<3>( varying );
  }
}

}  // namespace tribol
