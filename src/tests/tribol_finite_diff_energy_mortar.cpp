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
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/common/BasicTypes.hpp"
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
// and three points with frozen and varying integration-surface quadrature. Perturbations change local coordinate arrays
// without modifying mesh storage. Invalid geometry must report an error for
// values and derivatives; portable rule values are tested separately.
namespace tribol {

using LinearEdgeBasis = BasisFunction<EDGE, LINEAR_LAGRANGE>;

namespace {
using TestRule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>;
using PairCoordinates = EnergyMortarCalculator<TestRule>::Coordinates;

/**
 * @brief Read the two test elements in finite-element order before evaluating their coordinates.
 * @param mesh1 First mesh containing element zero.
 * @param mesh2 Second mesh containing element zero.
 * @return Pair coordinates used by the local calculations.
 */
PairCoordinates gatherCoordinates( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 )
{
  PairCoordinates coordinates;
  mesh1.getFaceCoords( 0, coordinates.data() );
  mesh2.getFaceCoords( 0, coordinates.data() + EnergyMortarCalculator<TestRule>::element_dof_count );
  return coordinates;
}
}  // namespace

template <typename QuadratureRule>
std::pair<RealT, RealT> EnergyMortarCalculator<QuadratureRule>::eval_gtilde( CoordinateView coordinates ) const
{
  const auto value = compute_nodal_contact_data( coordinates );
  return { value.g_tilde[0], value.g_tilde[1] };
}

template <typename QuadratureRule>
FiniteDiffResult EnergyMortarCalculator<QuadratureRule>::validate_g_tilde( CoordinateView coordinates,
                                                                           RealT epsilon ) const
{
  FiniteDiffResult result;
  IntegrationSurfaceRule frozen{};
  if ( !p_.enzyme_quadrature ) frozen = generateIntegrationSurfaceRule( coordinates );
  const auto evaluate = [&]( CoordinateView current ) {
    return p_.enzyme_quadrature ? eval_gtilde( current ) : eval_gtilde_fixed_qp( current, frozen );
  };
  const auto [gap1, gap2] = evaluate( coordinates );
  result.g_tilde1_baseline = gap1;
  result.g_tilde2_baseline = gap2;
  MdArray<RealT, dof_count> gradient1{}, gradient2{};
  grad_gtilde( coordinates, gradient1.data(), gradient2.data() );
  result.analytical_gradient_g1.assign( gradient1.data(), gradient1.data() + dof_count );
  result.analytical_gradient_g2.assign( gradient2.data(), gradient2.data() + dof_count );
  result.fd_gradient_g1.resize( dof_count );
  result.fd_gradient_g2.resize( dof_count );
  for ( IndexT dof = 0; dof < dof_count; ++dof ) {
    Coordinates perturbed;
    for ( IndexT i = 0; i < dof_count; ++i ) perturbed[i] = coordinates[i];
    perturbed[dof] += epsilon;
    const auto plus = evaluate( perturbed.view() );
    perturbed[dof] = coordinates[dof] - epsilon;
    const auto minus = evaluate( perturbed.view() );
    result.fd_gradient_g1[dof] = ( plus.first - minus.first ) / ( 2.0 * epsilon );
    result.fd_gradient_g2[dof] = ( plus.second - minus.second ) / ( 2.0 * epsilon );
  }
  return result;
}

template <typename QuadratureRule>
std::pair<RealT, RealT> EnergyMortarCalculator<QuadratureRule>::eval_gtilde_fixed_qp(
    CoordinateView coordinates, const IntegrationSurfaceRule& qp_fixed ) const
{
  const LinearEdgeGeometry::Coords edge_coords{ coordinates.data(), element_dof_count };
  const RealT jacobian = computeJacobianWeight<LinearEdgeBasis>( 0.0, edge_coords );
  RealT gap1 = 0.0, gap2 = 0.0;
  for ( IndexT q = 0; q < qp_fixed.result.required_point_count; ++q ) {
    const RealT xi = qp_fixed.coords( q, 0 );
    RealT basis_values[LinearEdgeBasis::node_count];
    LinearEdgeBasis::eval(
        MdSpan<const RealT, LinearEdgeBasis::parametric_dimension>{ &xi, LinearEdgeBasis::parametric_dimension },
        basis_values );
    const RealT gap = compute_weighted_normal_gap( coordinates, xi );
    gap1 += qp_fixed.weights[q] * basis_values[0] * jacobian * gap;
    gap2 += qp_fixed.weights[q] * basis_values[1] * jacobian * gap;
  }
  return { gap1, gap2 };
}

template <typename QuadratureRule>
FiniteDiffResult EnergyMortarCalculator<QuadratureRule>::validate_hessian( CoordinateView coordinates,
                                                                           RealT epsilon ) const
{
  FiniteDiffResult result;
  MdArray<RealT, dof_count, dof_count> hessian1{}, hessian2{};
  d2_g2tilde( coordinates, hessian1.data(), hessian2.data() );
  result.analytical_gradient_g1.assign( hessian1.data(), hessian1.data() + hessian1.size() );
  result.analytical_gradient_g2.assign( hessian2.data(), hessian2.data() + hessian2.size() );
  result.fd_gradient_g1.resize( hessian1.size() );
  result.fd_gradient_g2.resize( hessian2.size() );
  IntegrationSurfaceRule frozen{};
  if ( !p_.enzyme_quadrature ) frozen = generateIntegrationSurfaceRule( coordinates );
  const auto evaluate = [&]( const Coordinates& offsets ) {
    Coordinates perturbed;
    for ( IndexT i = 0; i < dof_count; ++i ) perturbed[i] = coordinates[i] + offsets[i];
    return p_.enzyme_quadrature ? eval_gtilde( perturbed.view() ) : eval_gtilde_fixed_qp( perturbed.view(), frozen );
  };
  const Coordinates zero{};
  const auto baseline = evaluate( zero );
  for ( IndexT row = 0; row < dof_count; ++row ) {
    for ( IndexT column = 0; column < dof_count; ++column ) {
      const IndexT index = row * dof_count + column;
      if ( row == column ) {
        Coordinates plus = zero, minus = zero;
        plus[row] = epsilon;
        minus[row] = -epsilon;
        const auto value_plus = evaluate( plus );
        const auto value_minus = evaluate( minus );
        result.fd_gradient_g1[index] =
            ( value_plus.first - 2.0 * baseline.first + value_minus.first ) / ( epsilon * epsilon );
        result.fd_gradient_g2[index] =
            ( value_plus.second - 2.0 * baseline.second + value_minus.second ) / ( epsilon * epsilon );
      } else {
        Coordinates pp = zero, pm = zero, mp = zero, mm = zero;
        pp[row] = pp[column] = epsilon;
        pm[row] = epsilon;
        pm[column] = -epsilon;
        mp[row] = -epsilon;
        mp[column] = epsilon;
        mm[row] = mm[column] = -epsilon;
        const auto value_pp = evaluate( pp ), value_pm = evaluate( pm );
        const auto value_mp = evaluate( mp ), value_mm = evaluate( mm );
        result.fd_gradient_g1[index] =
            ( value_pp.first - value_pm.first - value_mp.first + value_mm.first ) / ( 4.0 * epsilon * epsilon );
        result.fd_gradient_g2[index] =
            ( value_pp.second - value_pm.second - value_mp.second + value_mm.second ) / ( 4.0 * epsilon * epsilon );
      }
    }
  }
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
  const auto coordinates = gatherCoordinates( mesh1.getView(), mesh2.getView() );

  ContactParams params;
  params.k = 3.0;
  params.enzyme_quadrature = true;

  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  EnergyMortarCalculator evaluator( params, quadrature_rule );
  const auto result = evaluator.compute_quadrature_point_penalty_data( coordinates.view() );
  EXPECT_FALSE( result.has_active_qp );
  EXPECT_EQ( result.energy, 0.0 );
  EXPECT_TRUE( std::all_of( result.force.begin(), result.force.end(), []( RealT force ) { return force == 0.0; } ) );
  EXPECT_TRUE( std::all_of( result.stiffness.begin(), result.stiffness.end(),
                            []( RealT stiffness ) { return stiffness == 0.0; } ) );
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
  const auto coordinates = gatherCoordinates( mesh1.getView(), mesh2.getView() );

  ContactParams params;
  params.k = 3.0;
  params.enzyme_quadrature = true;

  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  EnergyMortarCalculator evaluator( params, quadrature_rule );
  const auto result = evaluator.compute_quadrature_point_penalty_data( coordinates.view() );

  EXPECT_TRUE( result.has_active_qp );
  EXPECT_EQ( result.energy, 0.0 );
  EXPECT_TRUE( std::all_of( result.force.begin(), result.force.end(), []( RealT force ) { return force == 0.0; } ) );
  EXPECT_TRUE( std::any_of( result.stiffness.begin(), result.stiffness.end(),
                            []( RealT stiffness ) { return stiffness != 0.0; } ) );
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
  const auto coordinates = gatherCoordinates( mesh1.getView(), mesh2.getView() );

  ContactParams params;
  params.k = 3.0;
  params.enzyme_quadrature = true;

  RealT gap_without_residual[2] = { 0.0, 0.0 };
  RealT tributary_area_without_residual[2] = { 0.0, 0.0 };
  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  EnergyMortarCalculator evaluator_without_residual( params, quadrature_rule );
  evaluator_without_residual.compute_gtilde_and_area( coordinates.view(), gap_without_residual,
                                                      tributary_area_without_residual );

  params.residual_gap = 0.15;
  RealT gap_with_residual[2] = { 0.0, 0.0 };
  RealT tributary_area_with_residual[2] = { 0.0, 0.0 };
  EnergyMortarCalculator evaluator_with_residual( params, quadrature_rule );
  evaluator_with_residual.compute_gtilde_and_area( coordinates.view(), gap_with_residual,
                                                   tributary_area_with_residual );

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
  const auto coordinates = gatherCoordinates( mesh1.getView(), mesh2.getView() );

  ContactParams params;
  params.k = 3.0;
  params.enzyme_quadrature = true;

  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  EnergyMortarCalculator evaluator_without_residual( params, quadrature_rule );
  const auto inactive = evaluator_without_residual.compute_quadrature_point_penalty_data( coordinates.view() );
  EXPECT_FALSE( inactive.has_active_qp );
  EXPECT_EQ( inactive.energy, 0.0 );
  EXPECT_TRUE(
      std::all_of( inactive.force.begin(), inactive.force.end(), []( RealT force ) { return force == 0.0; } ) );
  EXPECT_TRUE( std::all_of( inactive.stiffness.begin(), inactive.stiffness.end(),
                            []( RealT stiffness ) { return stiffness == 0.0; } ) );

  params.residual_gap = 0.15;
  EnergyMortarCalculator evaluator_with_residual( params, quadrature_rule );
  const auto active = evaluator_with_residual.compute_quadrature_point_penalty_data( coordinates.view() );
  EXPECT_TRUE( active.has_active_qp );
  EXPECT_GT( active.energy, 0.0 );
  EXPECT_TRUE( std::any_of( active.force.begin(), active.force.end(), []( RealT force ) { return force != 0.0; } ) );
  EXPECT_TRUE( std::any_of( active.stiffness.begin(), active.stiffness.end(),
                            []( RealT stiffness ) { return stiffness != 0.0; } ) );
}

class ResidualGapDerivativeCheck : public ::testing::TestWithParam<RealT> {};

// Perturb local coordinates in finite-element order so gradients and Hessians include both edges' current geometry.
TEST_P( ResidualGapDerivativeCheck, QuadraturePointPenaltyDerivativesMatchFiniteDifference )
{
  PairCoordinates coordinates{ 1.0, 0.0, 0.0, 0.0, 0.2, -0.1, 0.8, -0.1 };
  const auto baseline = coordinates;
  const TestRule quadrature_rule{};
  const EnergyMortarCalculator evaluator( ContactParams{ 3.0, true, GetParam() }, quadrature_rule );
  const auto analytical = evaluator.compute_quadrature_point_penalty_data( coordinates.view() );
  ASSERT_GT( analytical.energy, 0.0 );
  constexpr IndexT dofs = PairCoordinates::size();
  constexpr RealT gradient_step = 1.0e-7;
  constexpr RealT gradient_tolerance = 1.0e-6;
  for ( IndexT dof = 0; dof < dofs; ++dof ) {
    coordinates[dof] = baseline[dof] + gradient_step;
    const RealT plus = evaluator.compute_quadrature_point_penalty_energy( coordinates.view() );
    coordinates[dof] = baseline[dof] - gradient_step;
    const RealT minus = evaluator.compute_quadrature_point_penalty_energy( coordinates.view() );
    coordinates[dof] = baseline[dof];
    EXPECT_NEAR( analytical.force[dof], ( plus - minus ) / ( 2.0 * gradient_step ), gradient_tolerance )
        << "force mismatch at DOF " << dof;
  }
  constexpr RealT hessian_step = 1.0e-6;
  constexpr RealT hessian_tolerance = 1.0e-4;
  for ( IndexT column = 0; column < dofs; ++column ) {
    coordinates[column] = baseline[column] + hessian_step;
    const auto plus = evaluator.compute_quadrature_point_penalty_data( coordinates.view() );
    coordinates[column] = baseline[column] - hessian_step;
    const auto minus = evaluator.compute_quadrature_point_penalty_data( coordinates.view() );
    coordinates[column] = baseline[column];
    for ( IndexT row = 0; row < dofs; ++row ) {
      EXPECT_NEAR( analytical.stiffness[row * dofs + column],
                   ( plus.force[row] - minus.force[row] ) / ( 2.0 * hessian_step ), hessian_tolerance )
          << "stiffness mismatch at row " << row << ", column " << column;
    }
  }
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
  const auto coordinates = gatherCoordinates( mesh1.getView(), mesh2.getView() );

  // ── Evaluator setup ──────────────────────────────────────────────────────
  ContactParams params_;
  params_.k = 1.0;                   // penalty stiffness
  params_.enzyme_quadrature = true;  // use the non-Enzyme quadrature path
  params_.residual_gap = GetParam();

  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  EnergyMortarCalculator evaluator_( params_, quadrature_rule );

  // ── Run validation ───────────────────────────────────────────────────────
  const RealT epsilon = 1e-7;

  auto result = evaluator_.validate_g_tilde( coordinates.view(), epsilon );

  // ── Compare ──────────────────────────────────────────────────────────────
  const RealT tol = 1e-6;
  const std::size_t num_dofs = result.fd_gradient_g1.size();

  ASSERT_EQ( result.fd_gradient_g1.size(), result.analytical_gradient_g1.size() );
  ASSERT_EQ( result.fd_gradient_g2.size(), result.analytical_gradient_g2.size() );

  for ( size_t i = 0; i < num_dofs; ++i ) {
    EXPECT_NEAR( result.fd_gradient_g1[i], result.analytical_gradient_g1[i], tol )
        << "gtilde1 mismatch at DOF [" << i << "]"
        << "  pair node=" << ( i / LinearEdgeBasis::spatial_dimension ) << "  dir=" << ( i % 2 == 0 ? "x" : "y" )
        << "  FD=" << result.fd_gradient_g1[i] << "  AD=" << result.analytical_gradient_g1[i];

    EXPECT_NEAR( result.fd_gradient_g2[i], result.analytical_gradient_g2[i], tol )
        << "gtilde2 mismatch at DOF [" << i << "]"
        << "  pair node=" << ( i / LinearEdgeBasis::spatial_dimension ) << "  dir=" << ( i % 2 == 0 ? "x" : "y" )
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
  const auto coordinates = gatherCoordinates( mesh1.getView(), mesh2.getView() );

  // ── Evaluator setup ──────────────────────────────────────────────────────
  ContactParams params_;
  params_.k = 1.0;
  params_.enzyme_quadrature = true;
  params_.residual_gap = GetParam();

  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  EnergyMortarCalculator evaluator_( params_, quadrature_rule );

  // ── Run validation ───────────────────────────────────────────────────────
  const RealT epsilon = 1e-5;
  auto result = evaluator_.validate_hessian( coordinates.view(), epsilon );

  // ── Compare ──────────────────────────────────────────────────────────────
  const RealT tol = 1e-4;
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
  const auto coordinates = gatherCoordinates( mesh1.getView(), mesh2.getView() );
  const ContactParams params{ 1.0, true, 0.0 };
  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule;
  const EnergyMortarCalculator evaluator( params, quadrature_rule );
  NodalContactData nodal{};
  evaluator.compute_gtilde_and_area( coordinates.view(), nodal.g_tilde.data(), nodal.AI.data() );
  for ( int node = 0; node < LinearEdgeBasis::node_count; ++node ) {
    EXPECT_EQ( nodal.g_tilde[node], 0.0 );
    EXPECT_EQ( nodal.AI[node], 0.0 );
  }
  const auto penalty = evaluator.compute_quadrature_point_penalty_data( coordinates.view() );
  EXPECT_FALSE( penalty.has_active_qp );
  EXPECT_EQ( penalty.energy, 0.0 );
  for ( const RealT force : penalty.force ) EXPECT_EQ( force, 0.0 );
  for ( const RealT stiffness : penalty.stiffness ) EXPECT_EQ( stiffness, 0.0 );
}

// Calling the calculator without search must give no gap, area, or penalty contribution for co-oriented edges.
TEST( EnergyMortarCoordinates, cooriented_pairs_have_zero_contribution_without_search )
{
  const RealT x1[] = { 0.0, 1.0 }, y1[] = { 0.0, 0.0 };
  const RealT x2[] = { 0.0, 1.0 }, y2[] = { 0.1, 0.1 };
  const IndexT connectivity[] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );
  MeshData mesh2( 1, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );
  const auto coordinates = gatherCoordinates( mesh1.getView(), mesh2.getView() );
  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule;
  // A positive residual gap must not turn an excluded pair into an active constraint.
  for ( const RealT residual_gap : { 0.0, 0.15 } ) {
    SCOPED_TRACE( residual_gap );
    const EnergyMortarCalculator calculator( ContactParams{ 1.0, true, residual_gap }, quadrature_rule );
    NodalContactData nodal{};
    calculator.compute_gtilde_and_area( coordinates.view(), nodal.g_tilde.data(), nodal.AI.data() );
    for ( int node = 0; node < LinearEdgeBasis::node_count; ++node ) {
      EXPECT_EQ( nodal.g_tilde[node], 0.0 );
      EXPECT_EQ( nodal.AI[node], 0.0 );
    }
    const auto penalty = calculator.compute_quadrature_point_penalty_data( coordinates.view() );
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
          const auto coordinates = gatherCoordinates( mesh1.getView(), mesh2.getView() );

          const EnergyMortarCalculator calculator( ContactParams{ 1.0, varying_quadrature, 0.0 }, quadrature_rule );
          NodalContactData nodal{};
          MdArray<RealT, coordinate_count> gradient1{}, gradient2{};
          MdArray<RealT, stiffness_count> hessian1{}, hessian2{};
          EXPECT_THROW( calculator.compute_gtilde_and_area( coordinates.view(), nodal.g_tilde.data(), nodal.AI.data() ),
                        std::invalid_argument );
          EXPECT_THROW( calculator.grad_gtilde( coordinates.view(), gradient1.data(), gradient2.data() ),
                        std::invalid_argument );
          EXPECT_THROW( calculator.grad_trib_area( coordinates.view(), gradient1.data(), gradient2.data() ),
                        std::invalid_argument );
          EXPECT_THROW( calculator.d2_g2tilde( coordinates.view(), hessian1.data(), hessian2.data() ),
                        std::invalid_argument );
          EXPECT_THROW( calculator.compute_d2A_d2u( coordinates.view(), hessian1.data(), hessian2.data() ),
                        std::invalid_argument );
          EXPECT_THROW( calculator.compute_quadrature_point_penalty_energy( coordinates.view() ),
                        std::invalid_argument );
          EXPECT_THROW( calculator.compute_quadrature_point_penalty_data( coordinates.view() ), std::invalid_argument );
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
  constexpr RealT tolerance = 1.0e-12;
  const IndexT connectivity[] = { 0, 1 };
  RealT x1[] = { 0.0, edge_length }, y1[] = { 0.0, 0.0 };
  RealT x2[] = { edge_length, 0.0 }, y2[] = { separation, separation };
  MeshData mesh1( 0, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );
  MeshData mesh2( 1, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );
  const auto coordinates = gatherCoordinates( mesh1.getView(), mesh2.getView() );
  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  const EnergyMortarCalculator calculator( ContactParams{ 1.0, true, 0.0 }, quadrature_rule );
  NodalContactData response{};
  calculator.compute_gtilde_and_area( coordinates.view(), response.g_tilde.data(), response.AI.data() );
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
  constexpr RealT step = 1.0e-4;
  constexpr RealT gradient_tolerance = 2.0e-7;
  constexpr RealT hessian_tolerance = 2.0e-6;
  SCOPED_TRACE( testing::Message() << "points=" << Count << ", varying=" << varying );
  typename Calculator::Coordinates coordinates{ 0.0, 0.0, 1.7, 0.1, 1.6, 0.25, 0.25, 0.15 };
  const auto baseline = coordinates;
  const QuadratureRule quadrature_rule{ SmoothedProjectedEdgeMapping{ 0.2 } };
  const Calculator calculator( ContactParams{ 7.0, varying, 0.05 }, quadrature_rule );
  const auto integration_surface_rule = quadrature_rule.generateIntegrationSurfaceRule(
      { coordinates.data(), Calculator::element_dof_count },
      { coordinates.data() + Calculator::element_dof_count, Calculator::element_dof_count } );
  const auto evaluate = [&]() {
    std::array<RealT, 4> values{};
    if ( varying ) {
      calculator.compute_gtilde_and_area( coordinates.view(), values.data(), values.data() + 2 );
    } else {
      const auto gaps = calculator.eval_gtilde_fixed_qp( coordinates.view(), integration_surface_rule );
      values[0] = gaps.first;
      values[1] = gaps.second;
      const RealT dx = coordinates[2] - coordinates[0], dy = coordinates[3] - coordinates[1];
      const RealT jacobian = 0.5 * std::sqrt( dx * dx + dy * dy );
      for ( int q = 0; q < Count; ++q ) {
        values[2] +=
            0.5 * ( 1.0 - integration_surface_rule.coords( q, 0 ) ) * integration_surface_rule.weights[q] * jacobian;
        values[3] +=
            0.5 * ( 1.0 + integration_surface_rule.coords( q, 0 ) ) * integration_surface_rule.weights[q] * jacobian;
      }
    }
    return values;
  };
  std::array<std::array<RealT, dofs>, 4> gradient{};
  std::array<std::array<RealT, dofs * dofs>, 4> hessian{};
  calculator.grad_gtilde( coordinates.view(), gradient[0].data(), gradient[1].data() );
  calculator.grad_trib_area( coordinates.view(), gradient[2].data(), gradient[3].data() );
  calculator.d2_g2tilde( coordinates.view(), hessian[0].data(), hessian[1].data() );
  calculator.compute_d2A_d2u( coordinates.view(), hessian[2].data(), hessian[3].data() );
  const auto original = evaluate();
  for ( int row = 0; row < dofs; ++row ) {
    coordinates[row] = baseline[row] + step;
    const auto plus = evaluate();
    coordinates[row] = baseline[row] - step;
    const auto minus = evaluate();
    coordinates[row] = baseline[row];
    for ( std::size_t output = 0; output < gradient.size(); ++output ) {
      EXPECT_NEAR( gradient[output][row], ( plus[output] - minus[output] ) / ( 2.0 * step ), gradient_tolerance );
      EXPECT_NEAR( hessian[output][row * dofs + row],
                   ( plus[output] - 2.0 * original[output] + minus[output] ) / ( step * step ), hessian_tolerance );
    }
    for ( int col = row + 1; col < dofs; ++col ) {
      std::array<std::array<RealT, 4>, 4> samples{};
      int sample = 0;
      for ( const RealT row_sign : { 1.0, -1.0 } ) {
        for ( const RealT col_sign : { 1.0, -1.0 } ) {
          coordinates[row] = baseline[row] + row_sign * step;
          coordinates[col] = baseline[col] + col_sign * step;
          samples[sample++] = evaluate();
        }
      }
      coordinates[row] = baseline[row];
      coordinates[col] = baseline[col];
      for ( std::size_t output = 0; output < hessian.size(); ++output ) {
        const RealT expected = ( samples[0][output] - samples[1][output] - samples[2][output] + samples[3][output] ) /
                               ( 4.0 * step * step );
        EXPECT_NEAR( hessian[output][row * dofs + col], expected, hessian_tolerance );
        EXPECT_NEAR( hessian[output][col * dofs + row], expected, hessian_tolerance );
      }
    }
  }
  if ( !varying ) return;
  // Penalty gradients and Hessians must use the same selected rule, including its geometry-dependent weights.
  const auto penalty = calculator.compute_quadrature_point_penalty_data( coordinates.view() );
  ASSERT_TRUE( penalty.has_active_qp );
  for ( int col = 0; col < dofs; ++col ) {
    coordinates[col] = baseline[col] + step;
    const auto plus = calculator.compute_quadrature_point_penalty_data( coordinates.view() );
    coordinates[col] = baseline[col] - step;
    const auto minus = calculator.compute_quadrature_point_penalty_data( coordinates.view() );
    coordinates[col] = baseline[col];
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
