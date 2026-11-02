// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <algorithm>
#include <array>
#include <stdexcept>
#include <type_traits>

#include <gtest/gtest.h>

#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/integ/PairQuadratureRule.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"
#include "tribol/physics/PairEvaluation.hpp"

// Checks local Energy Mortar response and finite-difference derivatives for quadrature-point and nodal energy
// definitions across one, two, and three Gauss points. Perturbations change local coordinate arrays without modifying
// mesh storage. Invalid geometry must report an error for values and derivatives; portable rule values are tested
// separately.
namespace tribol {

using LinearEdgeBasis = BasisFunction<EDGE, LINEAR_LAGRANGE>;

namespace {
using TestRule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>;
using TestNodalEnergy = EnergyMortarNodalEnergy<TestRule, OffsetGap<OpposingNormalGap>>;
using PairCoordinates = TestNodalEnergy::Coordinates;
using TestEnergy = EnergyMortarEnergy<TestRule, OffsetGap<OpposingNormalGap>>;
using TestEnergyEvaluator = EnergyPairEvaluator<TestEnergy>;

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
  mesh2.getFaceCoords( 0, coordinates.data() + TestNodalEnergy::element_dof_count );
  return coordinates;
}

/**
 * @brief Check that excluded or empty-overlap pairs have no nodal contribution or derivatives.
 * @tparam NodalDefinition Local nodal Energy Mortar definition.
 * @param definition Nodal definition under test.
 * @param coordinates Noncontributing pair geometry.
 */
template <typename NodalDefinition>
void expectZeroNodalResponse( const NodalDefinition& definition, typename NodalDefinition::CoordinateView coordinates )
{
  const auto primal = definition.evaluate( coordinates );
  ASSERT_EQ( primal.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_FALSE( primal.has_active_points );
  const auto values = definition.values( coordinates );
  const auto quantities = VectorPairEvaluator<NodalDefinition>::evaluateGradients( coordinates, definition );
  const auto hessians = VectorPairEvaluator<NodalDefinition>::evaluateHessians( coordinates, definition );
  for ( IndexT i = 0; i < NodalDefinition::value_count; ++i ) {
    EXPECT_EQ( primal.value[i], 0.0 );
    EXPECT_EQ( values[i], 0.0 );
    EXPECT_EQ( quantities.values[i], 0.0 );
  }
  for ( IndexT entry = 0; entry < quantities.gradients.size(); ++entry ) {
    EXPECT_EQ( quantities.gradients[entry], 0.0 );
  }
  for ( IndexT entry = 0; entry < hessians.size(); ++entry ) {
    EXPECT_EQ( hessians[entry], 0.0 );
  }
  const auto weighted = definition.withWeights( MdArray<RealT, NodalDefinition::node_count>{ 2.0, -3.0 } );
  const auto weighted_eval = EnergyPairEvaluator<std::decay_t<decltype( weighted )>>::evaluate( coordinates, weighted );
  EXPECT_EQ( weighted_eval.energy, 0.0 );
  for ( IndexT entry = 0; entry < weighted_eval.residual.size(); ++entry )
    EXPECT_EQ( weighted_eval.residual[entry], 0.0 );
  for ( IndexT entry = 0; entry < weighted_eval.jacobian.size(); ++entry )
    EXPECT_EQ( weighted_eval.jacobian[entry], 0.0 );
}
}  // namespace

// An open gap must report inactive contact and have zero energy and derivatives.
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

  const TestEnergy definition{ TestRule{}, 3.0, EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, 0.0 } } };
  const auto primal = definition.evaluate( coordinates.view() );
  ASSERT_EQ( primal.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_FALSE( primal.has_active_points );
  EXPECT_EQ( primal.energy, 0.0 );
  const auto result = TestEnergyEvaluator::evaluate( coordinates.view(), definition, primal.energy );
  EXPECT_EQ( result.energy, 0.0 );
  for ( IndexT entry = 0; entry < result.residual.size(); ++entry ) EXPECT_EQ( result.residual[entry], 0.0 );
  for ( IndexT entry = 0; entry < result.jacobian.size(); ++entry ) EXPECT_EQ( result.jacobian[entry], 0.0 );
}

// Zero gap remains active: energy and residual vanish, but the compressive-side Jacobian must remain nonzero.
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

  const TestEnergy definition{ TestRule{}, 3.0, EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, 0.0 } } };
  const auto primal = definition.evaluate( coordinates.view() );
  ASSERT_EQ( primal.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_TRUE( primal.has_active_points );
  EXPECT_EQ( primal.energy, 0.0 );
  const auto result = TestEnergyEvaluator::evaluate( coordinates.view(), definition, primal.energy );
  EXPECT_EQ( result.energy, 0.0 );
  for ( IndexT entry = 0; entry < result.residual.size(); ++entry ) EXPECT_EQ( result.residual[entry], 0.0 );
  EXPECT_TRUE( std::any_of( result.jacobian.data(), result.jacobian.data() + result.jacobian.size(),
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

  const TestRule quadrature_rule{};
  const TestNodalEnergy without_residual_def( quadrature_rule,
                                              EnergyMortarNodal{ OffsetGap{ OpposingNormalGap{}, 0.0 } } );
  const auto without_residual = without_residual_def.values( coordinates.view() );

  constexpr RealT residual_gap = 0.15;
  const TestNodalEnergy with_residual_def( quadrature_rule,
                                           EnergyMortarNodal{ OffsetGap{ OpposingNormalGap{}, residual_gap } } );
  const auto with_residual = with_residual_def.values( coordinates.view() );

  // The residual gap shifts the gap integral without changing the projected overlap geometry.
  for ( IndexT i = 0; i < TestNodalEnergy::node_count; ++i ) {
    const RealT area_without = without_residual[i + TestNodalEnergy::node_count];
    const RealT area_with = with_residual[i + TestNodalEnergy::node_count];
    EXPECT_NEAR( area_with, area_without, 1.0e-14 );
    EXPECT_NEAR( with_residual[i], without_residual[i] - residual_gap * area_without, 1.0e-14 );
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

  const TestEnergy without_residual{ TestRule{}, 3.0, EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, 0.0 } } };
  const auto inactive = without_residual.evaluate( coordinates.view() );
  ASSERT_EQ( inactive.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_FALSE( inactive.has_active_points );
  EXPECT_EQ( inactive.energy, 0.0 );
  const auto inactive_derivatives =
      TestEnergyEvaluator::evaluate( coordinates.view(), without_residual, inactive.energy );
  for ( IndexT entry = 0; entry < inactive_derivatives.residual.size(); ++entry )
    EXPECT_EQ( inactive_derivatives.residual[entry], 0.0 );
  for ( IndexT entry = 0; entry < inactive_derivatives.jacobian.size(); ++entry )
    EXPECT_EQ( inactive_derivatives.jacobian[entry], 0.0 );

  const TestEnergy with_residual{ TestRule{}, 3.0, EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, 0.15 } } };
  const auto active = with_residual.evaluate( coordinates.view() );
  ASSERT_EQ( active.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_TRUE( active.has_active_points );
  EXPECT_GT( active.energy, 0.0 );
  const auto active_derivatives = TestEnergyEvaluator::evaluate( coordinates.view(), with_residual, active.energy );
  EXPECT_TRUE( std::any_of( active_derivatives.residual.data(),
                            active_derivatives.residual.data() + active_derivatives.residual.size(),
                            []( RealT force ) { return force != 0.0; } ) );
  EXPECT_TRUE( std::any_of( active_derivatives.jacobian.data(),
                            active_derivatives.jacobian.data() + active_derivatives.jacobian.size(),
                            []( RealT stiffness ) { return stiffness != 0.0; } ) );
}

class ResidualGapDerivativeCheck : public ::testing::TestWithParam<RealT> {};

// Perturb local coordinates in finite-element order so gradients and Hessians include both edges' current geometry.
TEST_P( ResidualGapDerivativeCheck, QuadraturePointPenaltyDerivativesMatchFiniteDifference )
{
  PairCoordinates coordinates{ 1.0, 0.0, 0.0, 0.0, 0.2, -0.1, 0.8, -0.1 };
  const auto baseline = coordinates;
  const TestEnergy definition{ TestRule{}, 3.0, EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, GetParam() } } };
  const auto analytical = TestEnergyEvaluator::evaluate( coordinates.view(), definition );
  ASSERT_GT( analytical.energy, 0.0 );
  constexpr IndexT dofs = PairCoordinates::size();
  constexpr RealT gradient_step = 1.0e-7;
  constexpr RealT gradient_tolerance = 1.0e-6;
  for ( IndexT dof = 0; dof < dofs; ++dof ) {
    coordinates[dof] = baseline[dof] + gradient_step;
    const RealT plus = definition.energy( coordinates.view() );
    coordinates[dof] = baseline[dof] - gradient_step;
    const RealT minus = definition.energy( coordinates.view() );
    coordinates[dof] = baseline[dof];
    EXPECT_NEAR( analytical.residual[dof], ( plus - minus ) / ( 2.0 * gradient_step ), gradient_tolerance )
        << "force mismatch at DOF " << dof;
  }
  constexpr RealT hessian_step = 1.0e-6;
  constexpr RealT hessian_tolerance = 1.0e-4;
  for ( IndexT column = 0; column < dofs; ++column ) {
    coordinates[column] = baseline[column] + hessian_step;
    const auto plus = TestEnergyEvaluator::evaluate( coordinates.view(), definition );
    coordinates[column] = baseline[column] - hessian_step;
    const auto minus = TestEnergyEvaluator::evaluate( coordinates.view(), definition );
    coordinates[column] = baseline[column];
    for ( IndexT row = 0; row < dofs; ++row ) {
      EXPECT_NEAR( analytical.jacobian( row, column ),
                   ( plus.residual[row] - minus.residual[row] ) / ( 2.0 * hessian_step ), hessian_tolerance )
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
  const TestRule quadrature_rule;
  expectZeroNodalResponse(
      TestNodalEnergy{ quadrature_rule, EnergyMortarNodal{ OffsetGap{ OpposingNormalGap{}, 0.0 } } },
      coordinates.view() );
  const TestEnergy definition{ quadrature_rule, 1.0, EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, 0.0 } } };
  const auto primal = definition.evaluate( coordinates.view() );
  ASSERT_EQ( primal.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_EQ( primal.quadrature_result.required_point_count, 0 );
  EXPECT_FALSE( primal.has_active_points );
  const auto penalty = TestEnergyEvaluator::evaluate( coordinates.view(), definition, primal.energy );
  EXPECT_EQ( penalty.energy, 0.0 );
  for ( IndexT entry = 0; entry < penalty.residual.size(); ++entry ) EXPECT_EQ( penalty.residual[entry], 0.0 );
  for ( IndexT entry = 0; entry < penalty.jacobian.size(); ++entry ) EXPECT_EQ( penalty.jacobian[entry], 0.0 );
}

// Calling the definitions without search must give no gap, area, or penalty contribution for co-oriented edges.
TEST( EnergyMortarCoordinates, cooriented_pairs_have_zero_contribution_without_search )
{
  const RealT x1[] = { 0.0, 1.0 }, y1[] = { 0.0, 0.0 };
  const RealT x2[] = { 0.0, 1.0 }, y2[] = { 0.1, 0.1 };
  const IndexT connectivity[] = { 0, 1 };
  MeshData mesh1( 0, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );
  MeshData mesh2( 1, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );
  const auto coordinates = gatherCoordinates( mesh1.getView(), mesh2.getView() );
  const TestRule quadrature_rule;
  // A positive residual gap must not turn an excluded pair into an active constraint.
  for ( const RealT residual_gap : { 0.0, 0.15 } ) {
    SCOPED_TRACE( residual_gap );
    expectZeroNodalResponse(
        TestNodalEnergy{ quadrature_rule, EnergyMortarNodal{ OffsetGap{ OpposingNormalGap{}, residual_gap } } },
        coordinates.view() );
    const TestEnergy definition{ quadrature_rule, 1.0,
                                 EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, residual_gap } } };
    const auto primal = definition.evaluate( coordinates.view() );
    ASSERT_EQ( primal.quadrature_result.status, QuadratureStatus::Success );
    EXPECT_EQ( primal.quadrature_result.required_point_count, 0 );
    EXPECT_FALSE( primal.has_active_points );
    const auto penalty = TestEnergyEvaluator::evaluate( coordinates.view(), definition, primal.energy );
    EXPECT_EQ( penalty.energy, 0.0 );
    for ( IndexT entry = 0; entry < penalty.residual.size(); ++entry ) EXPECT_EQ( penalty.residual[entry], 0.0 );
    for ( IndexT entry = 0; entry < penalty.jacobian.size(); ++entry ) EXPECT_EQ( penalty.jacobian[entry], 0.0 );
  }
}

// Either degenerate edge must report an error for nodal values, derivatives, and penalty quantities.
TEST( EnergyMortarCoordinates, rejects_degenerate_geometry_before_consuming_quadrature )
{
  const auto check = []<int PointCount>() {
    using QuadratureRule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<PointCount>>;
    const QuadratureRule quadrature_rule;
    // Test coincident endpoints and a nonzero edge below the mapping's accepted length.
    for ( const RealT edge_length : { 0.0, 5.0e-15 } ) {
      for ( const bool collapse_mesh1 : { false, true } ) {
        SCOPED_TRACE( testing::Message() << "points=" << PointCount << ", length=" << edge_length
                                         << ", collapse_mesh1=" << collapse_mesh1 );
        const RealT x1[] = { 0.0, collapse_mesh1 ? edge_length : 1.0 }, y1[] = { 0.0, 0.0 };
        const RealT x2[] = { collapse_mesh1 ? 1.0 : edge_length, 0.0 }, y2[] = { 0.1, 0.1 };
        const IndexT connectivity[] = { 0, 1 };
        MeshData mesh1( 0, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );
        MeshData mesh2( 1, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );
        const auto coordinates = gatherCoordinates( mesh1.getView(), mesh2.getView() );

        using NodalDefinition = EnergyMortarNodalEnergy<QuadratureRule, OffsetGap<OpposingNormalGap>>;
        const NodalDefinition nodal_definition{ quadrature_rule,
                                                EnergyMortarNodal{ OffsetGap{ OpposingNormalGap{}, 0.0 } } };
        const auto nodal_primal = nodal_definition.evaluate( coordinates.view() );
        EXPECT_EQ( nodal_primal.quadrature_result.status, QuadratureStatus::InvalidGeometry );
        EXPECT_EQ( nodal_primal.quadrature_result.required_point_count, 0 );
        EXPECT_FALSE( nodal_primal.has_active_points );
        EXPECT_THROW( nodal_definition.values( coordinates.view() ), std::invalid_argument );
        EXPECT_THROW( nodal_definition.withWeights( {} ).energy( coordinates.view() ), std::invalid_argument );
        EXPECT_THROW( VectorPairEvaluator<NodalDefinition>::evaluateGradients( coordinates.view(), nodal_definition ),
                      std::invalid_argument );
        EXPECT_THROW( VectorPairEvaluator<NodalDefinition>::evaluateHessians( coordinates.view(), nodal_definition ),
                      std::invalid_argument );
        EXPECT_THROW( EnergyPairEvaluator<std::decay_t<decltype( nodal_definition.withWeights( {} ) )>>::evaluate(
                          coordinates.view(), nodal_definition.withWeights( {} ) ),
                      std::invalid_argument );

        using Definition = EnergyMortarEnergy<QuadratureRule, OffsetGap<OpposingNormalGap>>;
        const Definition definition{ quadrature_rule, 1.0,
                                     EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, 0.0 } } };
        const auto primal = definition.evaluate( coordinates.view() );
        EXPECT_EQ( primal.quadrature_result.status, QuadratureStatus::InvalidGeometry );
        EXPECT_EQ( primal.quadrature_result.required_point_count, 0 );
        EXPECT_EQ( primal.energy, 0.0 );
        EXPECT_FALSE( primal.has_active_points );
        EXPECT_THROW( definition.energy( coordinates.view() ), std::invalid_argument );
        EXPECT_THROW( EnergyPairEvaluator<Definition>::evaluate( coordinates.view(), definition ),
                      std::invalid_argument );
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
  const TestRule quadrature_rule{};
  const TestNodalEnergy nodal_definition( quadrature_rule, EnergyMortarNodal{ OffsetGap{ OpposingNormalGap{}, 0.0 } } );
  const auto response = nodal_definition.values( coordinates.view() );
  for ( int node = 0; node < LinearEdgeBasis::node_count; ++node ) {
    const RealT area = response[node + LinearEdgeBasis::node_count];
    ASSERT_GT( area, 0.0 );
    EXPECT_NEAR( response[node] / area, -separation, tolerance );
  }
}

/**
 * @brief Check nodal integral, nodal-weighted energy, and quadrature-point penalty derivatives while perturbations
 * change length, normals, and overlap bounds.
 * @tparam Count Points in each edge quadrature rule.
 * @param residual_gap Fixed offset subtracted from the geometric gap.
 * @param initial_coordinates Baseline geometry, away from smoothing and contact activation transitions.
 */
template <int Count>
void checkSelectedRuleDerivatives( RealT residual_gap, const PairCoordinates& initial_coordinates )
{
  using QuadratureRule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<Count>>;
  using NodalDefinition = EnergyMortarNodalEnergy<QuadratureRule, OffsetGap<OpposingNormalGap>>;
  constexpr IndexT node_count = NodalDefinition::node_count;
  constexpr IndexT value_count = NodalDefinition::value_count;
  constexpr IndexT dofs = NodalDefinition::dof_count;
  constexpr RealT step = 1.0e-4;
  constexpr RealT gradient_tolerance = 2.0e-7;
  constexpr RealT hessian_tolerance = 2.0e-6;
  SCOPED_TRACE( testing::Message() << "points=" << Count << ", residual_gap=" << residual_gap );
  auto coordinates = initial_coordinates;
  const QuadratureRule quadrature_rule{ SmoothedProjectedEdgeMapping{ 0.2 } };
  const NodalDefinition nodal_definition( quadrature_rule,
                                          EnergyMortarNodal{ OffsetGap{ OpposingNormalGap{}, residual_gap } } );
  const auto original = nodal_definition.values( coordinates.view() );
  const auto quantities =
      VectorPairEvaluator<NodalDefinition>::evaluateGradients( coordinates.view(), nodal_definition, original.view() );
  const auto hessians = VectorPairEvaluator<NodalDefinition>::evaluateHessians( coordinates.view(), nodal_definition );
  const auto gap_only = VectorPairEvaluator<NodalDefinition>::template evaluateHessians<node_count>( coordinates.view(),
                                                                                                     nodal_definition );
  for ( IndexT node = 0; node < node_count; ++node ) {
    for ( IndexT row = 0; row < dofs; ++row ) {
      for ( IndexT col = 0; col < dofs; ++col ) {
        EXPECT_DOUBLE_EQ( gap_only( node, row, col ), hessians( node, row, col ) );
        EXPECT_EQ( gap_only( node + node_count, row, col ), 0.0 );
      }
    }
  }
  for ( IndexT i = 0; i < value_count; ++i ) {
    EXPECT_NEAR( quantities.values[i], original[i], gradient_tolerance );
  }
  for ( IndexT row = 0; row < dofs; ++row ) {
    coordinates[row] = initial_coordinates[row] + step;
    const auto plus = nodal_definition.values( coordinates.view() );
    coordinates[row] = initial_coordinates[row] - step;
    const auto minus = nodal_definition.values( coordinates.view() );
    coordinates[row] = initial_coordinates[row];
    for ( IndexT output = 0; output < value_count; ++output ) {
      SCOPED_TRACE( testing::Message() << "output=" << output << ", row=" << row );
      EXPECT_NEAR( quantities.gradients( output, row ), ( plus[output] - minus[output] ) / ( 2.0 * step ),
                   gradient_tolerance );
      EXPECT_NEAR( hessians( output, row, row ),
                   ( plus[output] - 2.0 * original[output] + minus[output] ) / ( step * step ), hessian_tolerance );
    }
    for ( IndexT col = row + 1; col < dofs; ++col ) {
      std::array<MdArray<RealT, value_count>, 4> samples{};
      IndexT sample = 0;
      for ( const RealT row_sign : { 1.0, -1.0 } ) {
        for ( const RealT col_sign : { 1.0, -1.0 } ) {
          coordinates[row] = initial_coordinates[row] + row_sign * step;
          coordinates[col] = initial_coordinates[col] + col_sign * step;
          samples[sample++] = nodal_definition.values( coordinates.view() );
        }
      }
      coordinates[row] = initial_coordinates[row];
      coordinates[col] = initial_coordinates[col];
      for ( IndexT output = 0; output < value_count; ++output ) {
        SCOPED_TRACE( testing::Message() << "output=" << output << ", row=" << row << ", column=" << col );
        const RealT expected = ( samples[0][output] - samples[1][output] - samples[2][output] + samples[3][output] ) /
                               ( 4.0 * step * step );
        EXPECT_NEAR( hessians( output, row, col ), expected, hessian_tolerance );
        EXPECT_NEAR( hessians( output, col, row ), expected, hessian_tolerance );
      }
    }
  }

  // Differentiating energy = dot(pressure, integratePair(coordinates, rule, nodal_basis * gap)) with
  // EnergyPairEvaluator must match both finite differences and the pressure-weighted sum of nodal gap derivatives.
  const MdArray<RealT, node_count> pressure{ -2.5, 1.75 };
  const auto lm_pair_def = nodal_definition.withWeights( pressure );
  const auto lm_pair_eval =
      EnergyPairEvaluator<std::decay_t<decltype( lm_pair_def )>>::evaluate( coordinates.view(), lm_pair_def );
  EXPECT_NEAR( lm_pair_eval.energy, dot( pressure, MdSpan<const RealT, node_count>{ original.data(), node_count } ),
               1.0e-14 );
  for ( IndexT row = 0; row < dofs; ++row ) {
    const RealT expected_grad =
        pressure[0] * quantities.gradients( 0, row ) + pressure[1] * quantities.gradients( 1, row );
    EXPECT_NEAR( lm_pair_eval.residual[row], expected_grad, 1.0e-13 );
    for ( IndexT col = 0; col < dofs; ++col ) {
      const RealT expected_hess = pressure[0] * hessians( 0, row, col ) + pressure[1] * hessians( 1, row, col );
      EXPECT_NEAR( lm_pair_eval.jacobian( row, col ), expected_hess, 1.0e-12 );
    }
  }

  // Differentiating with both nodal gap and tributary-area weights must match the combined weighted Hessians.
  const MdArray<RealT, node_count> area_weight{ -0.4, -0.9 };
  const auto weighted_def = nodal_definition.withWeights( pressure, area_weight );
  const auto weighted_eval =
      EnergyPairEvaluator<std::decay_t<decltype( weighted_def )>>::evaluate( coordinates.view(), weighted_def );
  for ( IndexT row = 0; row < dofs; ++row ) {
    for ( IndexT col = 0; col < dofs; ++col ) {
      const RealT expected_hess = pressure[0] * hessians( 0, row, col ) + pressure[1] * hessians( 1, row, col ) +
                                  area_weight[0] * hessians( 2, row, col ) + area_weight[1] * hessians( 3, row, col );
      EXPECT_NEAR( weighted_eval.jacobian( row, col ), expected_hess, 1.0e-12 );
    }
  }

  // Penalty gradients and Hessians must use the same selected rule, including its geometry-dependent weights.
  using Definition = EnergyMortarEnergy<QuadratureRule, OffsetGap<OpposingNormalGap>>;
  const Definition definition{ quadrature_rule, 7.0,
                               EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, residual_gap } } };
  const auto primal = definition.evaluate( coordinates.view() );
  ASSERT_EQ( primal.quadrature_result.status, QuadratureStatus::Success );
  if ( !primal.has_active_points ) return;
  const auto penalty = EnergyPairEvaluator<Definition>::evaluate( coordinates.view(), definition, primal.energy );
  for ( IndexT col = 0; col < dofs; ++col ) {
    coordinates[col] = initial_coordinates[col] + step;
    const auto plus = EnergyPairEvaluator<Definition>::evaluate( coordinates.view(), definition );
    coordinates[col] = initial_coordinates[col] - step;
    const auto minus = EnergyPairEvaluator<Definition>::evaluate( coordinates.view(), definition );
    coordinates[col] = initial_coordinates[col];
    EXPECT_NEAR( penalty.residual[col], ( plus.energy - minus.energy ) / ( 2.0 * step ), gradient_tolerance );
    for ( IndexT row = 0; row < dofs; ++row ) {
      EXPECT_NEAR( penalty.jacobian( row, col ), ( plus.residual[row] - minus.residual[row] ) / ( 2.0 * step ),
                   hessian_tolerance );
    }
  }
}
}  // namespace

// Parallel and oblique edges must preserve gap and area derivatives for all point counts and gap offsets.
TEST_P( ResidualGapDerivativeCheck, NodalAndPenaltyDerivativesMatchFiniteDifference )
{
  // The parallel overlap stays outside the endpoint smoothing regions; oblique edges exercise smoothed bounds.
  const std::array<PairCoordinates, 2> geometries{ PairCoordinates{ 1.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 0.5 },
                                                   PairCoordinates{ 0.0, 0.0, 1.7, 0.1, 1.6, 0.25, 0.25, 0.15 } };
  for ( std::size_t geometry = 0; geometry < geometries.size(); ++geometry ) {
    SCOPED_TRACE( testing::Message() << "geometry=" << geometry );
    checkSelectedRuleDerivatives<1>( GetParam(), geometries[geometry] );
    checkSelectedRuleDerivatives<2>( GetParam(), geometries[geometry] );
    checkSelectedRuleDerivatives<3>( GetParam(), geometries[geometry] );
  }
}

// Nodal penalty and Lagrange multiplier energy definitions must return exact gradients and Hessians wrt (g_tilde, A).
TEST( NodalEnergyDefinitionsCheck, DerivativesMatchFiniteDifference )
{
  MdArray<RealT, 2> state{ -0.12, 0.45 };
  const auto baseline = state;
  const auto penalty_expression = EnergyMortarNodal{}.penaltyEnergy( nodal_weighted_gap, nodal_area, 25.0 );
  auto penalty_def = penalty_expression.atNode( 0, 1.0e-14 );
  const auto penalty_eval = EnergyPairEvaluator<decltype( penalty_def )>::evaluate( state.view(), penalty_def );
  EXPECT_NEAR( penalty_eval.residual[0], 25.0 * ( -0.12 / 0.45 ), 1.0e-14 );
  EXPECT_NEAR( penalty_eval.energy, 0.5 * penalty_eval.residual[0] * state[0], 1.0e-14 );

  constexpr RealT step = 1.0e-6;
  constexpr RealT tol = 1.0e-5;
  for ( IndexT col = 0; col < 2; ++col ) {
    state[col] = baseline[col] + step;
    const auto plus = EnergyPairEvaluator<decltype( penalty_def )>::evaluate( state.view(), penalty_def );
    state[col] = baseline[col] - step;
    const auto minus = EnergyPairEvaluator<decltype( penalty_def )>::evaluate( state.view(), penalty_def );
    state[col] = baseline[col];
    EXPECT_NEAR( penalty_eval.residual[col], ( plus.energy - minus.energy ) / ( 2.0 * step ), tol );
    for ( IndexT row = 0; row < 2; ++row ) {
      EXPECT_NEAR( penalty_eval.jacobian( row, col ), ( plus.residual[row] - minus.residual[row] ) / ( 2.0 * step ),
                   tol );
    }
  }

  const RealT pressure[]{ -3.5 };
  const auto lm_expression =
      EnergyMortarNodal{}.lagrangeMultiplierEnergy( nodal_weighted_gap, IndependentNodalField{ { pressure, 1 } } );
  auto lm_def = lm_expression.atNode( 0, 1.0e-14 );
  const MdArray<RealT, decltype( lm_def )::dof_count> lm_state{ state[0] };
  const auto lm_eval = EnergyPairEvaluator<decltype( lm_def )>::evaluate( lm_state.view(), lm_def );
  EXPECT_DOUBLE_EQ( lm_eval.energy, -3.5 * state[0] );
  EXPECT_DOUBLE_EQ( lm_eval.residual[0], -3.5 );
  for ( IndexT entry = 0; entry < lm_eval.jacobian.size(); ++entry ) {
    EXPECT_DOUBLE_EQ( lm_eval.jacobian[entry], 0.0 );
  }
}

}  // namespace tribol
