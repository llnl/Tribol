// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <algorithm>
#include <array>
#include <stdexcept>
#include <type_traits>

#include <gtest/gtest.h>

#include "tribol/physics/EnergyMortar.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/integ/BasisFunction.hpp"

// Checks local Energy Mortar response and finite-difference derivatives. Selected-rule coverage includes one, two,
// and three points with frozen and varying integration-surface quadrature. Perturbations change local coordinate arrays
// without modifying mesh storage. Invalid geometry must report an error for values and derivatives; portable rule
// values are tested separately.
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

/**
 * @brief Check that excluded or empty-overlap pairs have no nodal contribution or derivatives.
 * @tparam Calculator Local Energy Mortar calculator.
 * @param calculator Calculator with the quadrature derivative mode under test.
 * @param coordinates Noncontributing pair geometry.
 */
template <typename Calculator>
void expectZeroNodalResponse( const Calculator& calculator, typename Calculator::CoordinateView coordinates )
{
  const auto values = calculator.computeNodalValues( coordinates );
  const auto quantities = calculator.computeContactQuantities( coordinates );
  const auto hessians = calculator.computeSecondDerivatives( coordinates );
  for ( IndexT node = 0; node < NodalContactData::node_count; ++node ) {
    EXPECT_EQ( values.g_tilde[node], 0.0 );
    EXPECT_EQ( values.AI[node], 0.0 );
    EXPECT_EQ( quantities.g_tilde[node], 0.0 );
    EXPECT_EQ( quantities.AI[node], 0.0 );
  }
  for ( IndexT entry = 0; entry < quantities.gap_gradients.size(); ++entry ) {
    EXPECT_EQ( quantities.gap_gradients[entry], 0.0 );
    EXPECT_EQ( quantities.area_gradients[entry], 0.0 );
  }
  for ( IndexT entry = 0; entry < hessians.gap_hessians.size(); ++entry ) {
    EXPECT_EQ( hessians.gap_hessians[entry], 0.0 );
    EXPECT_EQ( hessians.area_hessians[entry], 0.0 );
  }
}
}  // namespace

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

  const PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>> quadrature_rule{};
  EnergyMortarCalculator evaluator_without_residual( params, quadrature_rule );
  const auto without_residual = evaluator_without_residual.computeNodalValues( coordinates.view() );

  params.residual_gap = 0.15;
  EnergyMortarCalculator evaluator_with_residual( params, quadrature_rule );
  const auto with_residual = evaluator_with_residual.computeNodalValues( coordinates.view() );

  // The residual gap shifts the gap integral without changing the projected overlap geometry.
  for ( int i = 0; i < 2; ++i ) {
    EXPECT_NEAR( with_residual.AI[i], without_residual.AI[i], 1.0e-14 );
    EXPECT_NEAR( with_residual.g_tilde[i], without_residual.g_tilde[i] - params.residual_gap * without_residual.AI[i],
                 1.0e-14 );
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
  for ( const bool varying : { false, true } ) {
    const EnergyMortarCalculator calculator( ContactParams{ 1.0, varying, 0.0 }, quadrature_rule );
    expectZeroNodalResponse( calculator, coordinates.view() );
  }
  const EnergyMortarCalculator evaluator( params, quadrature_rule );
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
    for ( const bool varying : { false, true } ) {
      const EnergyMortarCalculator calculator( ContactParams{ 1.0, varying, residual_gap }, quadrature_rule );
      expectZeroNodalResponse( calculator, coordinates.view() );
    }
    const EnergyMortarCalculator calculator( ContactParams{ 1.0, true, residual_gap }, quadrature_rule );
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
          EXPECT_THROW( calculator.computeNodalValues( coordinates.view() ), std::invalid_argument );
          EXPECT_THROW( calculator.computeContactQuantities( coordinates.view() ), std::invalid_argument );
          EXPECT_THROW( calculator.computeSecondDerivatives( coordinates.view() ), std::invalid_argument );
          EXPECT_THROW( calculator.computeSecondDerivatives( coordinates.view(), false ), std::invalid_argument );
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
  const auto response = calculator.computeNodalValues( coordinates.view() );
  for ( int node = 0; node < LinearEdgeBasis::node_count; ++node ) {
    ASSERT_GT( response.AI[node], 0.0 );
    EXPECT_NEAR( response.g_tilde[node] / response.AI[node], -separation, tolerance );
  }
}

/**
 * @brief Check gap and area derivatives while perturbations change length, normals, and overlap bounds.
 * @tparam Count Points in each edge quadrature rule.
 * @param varying Whether to regenerate integration-surface points under coordinate perturbations.
 * @param residual_gap Fixed offset subtracted from the geometric gap.
 * @param initial_coordinates Baseline geometry, away from smoothing and contact activation transitions.
 */
template <int Count>
void checkSelectedRuleDerivatives( bool varying, RealT residual_gap, const PairCoordinates& initial_coordinates )
{
  using QuadratureRule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<Count>>;
  using Calculator = EnergyMortarCalculator<QuadratureRule>;
  static_assert( !std::is_constructible_v<Calculator, const ContactParams&, QuadratureRule&&>,
                 "A calculator must not retain a reference to a temporary rule generator." );
  constexpr IndexT dofs = Calculator::dof_count;
  constexpr RealT step = 1.0e-4;
  constexpr RealT gradient_tolerance = 2.0e-7;
  constexpr RealT hessian_tolerance = 2.0e-6;
  SCOPED_TRACE( testing::Message() << "points=" << Count << ", varying=" << varying
                                   << ", residual_gap=" << residual_gap );
  auto coordinates = initial_coordinates;
  const QuadratureRule quadrature_rule{ SmoothedProjectedEdgeMapping{ 0.2 } };
  const Calculator calculator( ContactParams{ 7.0, varying, residual_gap }, quadrature_rule );
  const auto frozen = quadrature_rule.generateIntegrationSurfaceRule(
      { coordinates.data(), Calculator::element_dof_count },
      { coordinates.data() + Calculator::element_dof_count, Calculator::element_dof_count } );
  ASSERT_EQ( frozen.result.status, QuadratureStatus::Success );
  const auto evaluate = [&]() {
    if ( varying ) return calculator.computeNodalValues( coordinates.view() );
    // Freeze only the surface points and reference weights. Current projections, normals, and length must change.
    const LinearEdgeGeometry::Coords coordinates1{ coordinates.data(), Calculator::element_dof_count };
    const LinearEdgeGeometry::Coords coordinates2{ coordinates.data() + Calculator::element_dof_count,
                                                   Calculator::element_dof_count };
    const auto quadrature =
        quadrature_rule.mapIntegrationSurfaceRuleToElements( coordinates1, coordinates2, frozen.getView() );
    EXPECT_EQ( quadrature.result.status, QuadratureStatus::Success );
    const LinearEdgeGeometry geometry1( coordinates1 ), geometry2( coordinates2 );
    const auto normal2 = geometry2.normal();
    const RealT alignment = std::min( geometry1.normal().dot( normal2 ), RealT{ 0.0 } );
    NodalContactData values{};
    for ( IndexT q = 0; q < quadrature.result.required_point_count; ++q ) {
      const RealT xi = quadrature.geometry1_coords( q, 0 );
      const auto point1 = geometry1.position( xi );
      const auto point2 = geometry2.position( quadrature.geometry2_coords( q, 0 ) );
      const RealT gap =
          ( ( point2[0] - point1[0] ) * normal2[0] + ( point2[1] - point1[1] ) * normal2[1] ) * alignment -
          residual_gap;
      for ( IndexT node = 0; node < NodalContactData::node_count; ++node ) {
        const RealT basis = 0.5 * ( 1.0 + ( node == 0 ? -xi : xi ) );
        values.g_tilde[node] += basis * quadrature.weights[q] * gap;
        values.AI[node] += basis * quadrature.weights[q];
      }
    }
    return values;
  };
  const auto quantities = calculator.computeContactQuantities( coordinates.view() );
  const auto hessians = calculator.computeSecondDerivatives( coordinates.view() );
  const auto gap_only = calculator.computeSecondDerivatives( coordinates.view(), false );
  for ( IndexT entry = 0; entry < gap_only.gap_hessians.size(); ++entry ) {
    EXPECT_DOUBLE_EQ( gap_only.gap_hessians[entry], hessians.gap_hessians[entry] );
    EXPECT_EQ( gap_only.area_hessians[entry], 0.0 );
  }
  const auto original = evaluate();
  for ( IndexT node = 0; node < NodalContactData::node_count; ++node ) {
    EXPECT_NEAR( quantities.g_tilde[node], original.g_tilde[node], gradient_tolerance );
    EXPECT_NEAR( quantities.AI[node], original.AI[node], gradient_tolerance );
  }
  for ( IndexT row = 0; row < dofs; ++row ) {
    coordinates[row] = initial_coordinates[row] + step;
    const auto plus = evaluate();
    coordinates[row] = initial_coordinates[row] - step;
    const auto minus = evaluate();
    coordinates[row] = initial_coordinates[row];
    for ( IndexT node = 0; node < NodalContactData::node_count; ++node ) {
      SCOPED_TRACE( testing::Message() << "node=" << node << ", row=" << row );
      EXPECT_NEAR( quantities.gap_gradients( node, row ), ( plus.g_tilde[node] - minus.g_tilde[node] ) / ( 2.0 * step ),
                   gradient_tolerance );
      EXPECT_NEAR( quantities.area_gradients( node, row ), ( plus.AI[node] - minus.AI[node] ) / ( 2.0 * step ),
                   gradient_tolerance );
      EXPECT_NEAR( hessians.gap_hessians( node, row, row ),
                   ( plus.g_tilde[node] - 2.0 * original.g_tilde[node] + minus.g_tilde[node] ) / ( step * step ),
                   hessian_tolerance );
      EXPECT_NEAR( hessians.area_hessians( node, row, row ),
                   ( plus.AI[node] - 2.0 * original.AI[node] + minus.AI[node] ) / ( step * step ), hessian_tolerance );
    }
    for ( IndexT col = row + 1; col < dofs; ++col ) {
      std::array<NodalContactData, 4> samples{};
      IndexT sample = 0;
      for ( const RealT row_sign : { 1.0, -1.0 } ) {
        for ( const RealT col_sign : { 1.0, -1.0 } ) {
          coordinates[row] = initial_coordinates[row] + row_sign * step;
          coordinates[col] = initial_coordinates[col] + col_sign * step;
          samples[sample++] = evaluate();
        }
      }
      coordinates[row] = initial_coordinates[row];
      coordinates[col] = initial_coordinates[col];
      for ( IndexT node = 0; node < NodalContactData::node_count; ++node ) {
        SCOPED_TRACE( testing::Message() << "node=" << node << ", row=" << row << ", column=" << col );
        const RealT expected_gap = ( samples[0].g_tilde[node] - samples[1].g_tilde[node] - samples[2].g_tilde[node] +
                                     samples[3].g_tilde[node] ) /
                                   ( 4.0 * step * step );
        const RealT expected_area =
            ( samples[0].AI[node] - samples[1].AI[node] - samples[2].AI[node] + samples[3].AI[node] ) /
            ( 4.0 * step * step );
        EXPECT_NEAR( hessians.gap_hessians( node, row, col ), expected_gap, hessian_tolerance );
        EXPECT_NEAR( hessians.gap_hessians( node, col, row ), expected_gap, hessian_tolerance );
        EXPECT_NEAR( hessians.area_hessians( node, row, col ), expected_area, hessian_tolerance );
        EXPECT_NEAR( hessians.area_hessians( node, col, row ), expected_area, hessian_tolerance );
      }
    }
  }
  if ( !varying ) return;
  // Penalty gradients and Hessians must use the same selected rule, including its geometry-dependent weights.
  const auto penalty = calculator.compute_quadrature_point_penalty_data( coordinates.view() );
  if ( !penalty.has_active_qp ) return;
  for ( IndexT col = 0; col < dofs; ++col ) {
    coordinates[col] = initial_coordinates[col] + step;
    const auto plus = calculator.compute_quadrature_point_penalty_data( coordinates.view() );
    coordinates[col] = initial_coordinates[col] - step;
    const auto minus = calculator.compute_quadrature_point_penalty_data( coordinates.view() );
    coordinates[col] = initial_coordinates[col];
    EXPECT_NEAR( penalty.force[col], ( plus.energy - minus.energy ) / ( 2.0 * step ), gradient_tolerance );
    for ( IndexT row = 0; row < dofs; ++row ) {
      EXPECT_NEAR( penalty.stiffness[row * dofs + col], ( plus.force[row] - minus.force[row] ) / ( 2.0 * step ),
                   hessian_tolerance );
    }
  }
}
}  // namespace

// Parallel and oblique edges must preserve gap and area derivatives for all point counts, modes, and gap offsets.
TEST_P( ResidualGapDerivativeCheck, NodalAndPenaltyDerivativesMatchFiniteDifference )
{
  // The parallel overlap stays outside the endpoint smoothing regions; oblique edges exercise smoothed bounds.
  const std::array<PairCoordinates, 2> geometries{ PairCoordinates{ 1.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 0.5 },
                                                   PairCoordinates{ 0.0, 0.0, 1.7, 0.1, 1.6, 0.25, 0.25, 0.15 } };
  for ( std::size_t geometry = 0; geometry < geometries.size(); ++geometry ) {
    SCOPED_TRACE( testing::Message() << "geometry=" << geometry );
    for ( const bool varying : { false, true } ) {
      checkSelectedRuleDerivatives<1>( varying, GetParam(), geometries[geometry] );
      checkSelectedRuleDerivatives<2>( varying, GetParam(), geometries[geometry] );
      checkSelectedRuleDerivatives<3>( varying, GetParam(), geometries[geometry] );
    }
  }
}

}  // namespace tribol
