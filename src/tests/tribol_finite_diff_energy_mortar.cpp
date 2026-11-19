// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Finite-difference checks for the composed Energy Mortar pair functions. */

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/physics/QuadraturePointEnergyMortar.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/geom/OpposingNormalGap.hpp"
#include "tribol/physics/integrals/PairIntegrals.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"
#include "tribol/assembly/EnzymePairDerivative.hpp"

namespace {
using namespace tribol;

constexpr RealT finite_difference_step = 1.0e-6;  ///< Stable central-difference step for the oblique edge geometry.
constexpr RealT jacobian_tolerance = 1.0e-6;      ///< First-derivative comparison tolerance.
constexpr RealT hessian_tolerance = 2.0e-5;       ///< Second-derivative comparison tolerance.

/**
 * @brief Check one vector pair function and weighted contraction against coordinate differences.
 * @tparam Function Fixed-size Energy Mortar pair function.
 * @param function Production pair function with geometry-dependent quadrature.
 * @param coordinates Current pair coordinates in Jacobian column order, away from overlap and activation transitions.
 * @param weights Output weights used for the Hessian contraction.
 */
template <typename Function>
void checkFiniteDifferences( const Function& function, const MdArray<RealT, Function::size_coords>& coordinates,
                             const MdArray<RealT, Function::size_values>& weights )
{
  const auto base = EnzymePairDerivative<Function>::valuesAndJacobian(
      { coordinates.data(), Function::size_geom1_coords },
      { coordinates.data() + Function::size_geom1_coords, Function::size_geom2_coords }, function );
  const auto weighted = EnzymePairDerivative<Function>::weightedJacobianAndHessian(
      { coordinates.data(), Function::size_geom1_coords },
      { coordinates.data() + Function::size_geom1_coords, Function::size_geom2_coords }, function, weights.view() );
  ASSERT_EQ( base.pair_value.quadrature_metadata.status, QuadratureStatus::Success );
  ASSERT_TRUE( base.pair_value.has_active_points );
  bool has_nonzero_jacobian = false;
  bool has_nonzero_hessian = false;
  for ( IndexT row = 0; row < Function::size_values; ++row ) {
    for ( IndexT column = 0; column < Function::size_coords; ++column ) {
      has_nonzero_jacobian = has_nonzero_jacobian || base.derivatives.jacobian( row, column ) != 0.0;
    }
  }
  for ( IndexT row = 0; row < Function::size_coords; ++row ) {
    for ( IndexT column = 0; column < Function::size_coords; ++column ) {
      has_nonzero_hessian = has_nonzero_hessian || weighted.hessian( row, column ) != 0.0;
    }
  }
  ASSERT_TRUE( has_nonzero_jacobian );
  ASSERT_TRUE( has_nonzero_hessian );
  for ( IndexT column = 0; column < Function::size_coords; ++column ) {
    auto positive_coordinates = coordinates;
    auto negative_coordinates = coordinates;
    positive_coordinates[column] += finite_difference_step;
    negative_coordinates[column] -= finite_difference_step;
    const auto positive = EnzymePairDerivative<Function>::valuesAndJacobian(
        { positive_coordinates.data(), Function::size_geom1_coords },
        { positive_coordinates.data() + Function::size_geom1_coords, Function::size_geom2_coords }, function );
    const auto negative = EnzymePairDerivative<Function>::valuesAndJacobian(
        { negative_coordinates.data(), Function::size_geom1_coords },
        { negative_coordinates.data() + Function::size_geom1_coords, Function::size_geom2_coords }, function );
    ASSERT_EQ( positive.pair_value.quadrature_metadata.status, QuadratureStatus::Success );
    ASSERT_TRUE( positive.pair_value.has_active_points );
    ASSERT_EQ( negative.pair_value.quadrature_metadata.status, QuadratureStatus::Success );
    ASSERT_TRUE( negative.pair_value.has_active_points );
    const auto positive_values = positive.pair_value.values();
    const auto negative_values = negative.pair_value.values();

    for ( IndexT value = 0; value < Function::size_values; ++value ) {
      EXPECT_NEAR( base.derivatives.jacobian( value, column ),
                   ( positive_values[value] - negative_values[value] ) / ( 2.0 * finite_difference_step ),
                   jacobian_tolerance );
    }
    for ( IndexT row = 0; row < Function::size_coords; ++row ) {
      RealT positive_gradient = 0.0;
      RealT negative_gradient = 0.0;
      for ( IndexT value = 0; value < Function::size_values; ++value ) {
        positive_gradient += weights[value] * positive.derivatives.jacobian( value, row );
        negative_gradient += weights[value] * negative.derivatives.jacobian( value, row );
      }
      EXPECT_NEAR( weighted.hessian( row, column ),
                   ( positive_gradient - negative_gradient ) / ( 2.0 * finite_difference_step ), hessian_tolerance );
    }
  }
}

/** @brief Check derivatives using production quadrature generation. @tparam NumPoints Gauss point count. */
template <int NumPoints>
void checkRuleGenerator()
{
  using RuleGenerator =
      PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>;
  using Gap = OffsetGap<OpposingNormalGap>;
  constexpr RealT smoothing_fraction = 0.2;
  const RuleGenerator rule_generator{ SmoothedProjectedEdgeMapping{ smoothing_fraction } };
  const Gap gap{ {}, 0.05 };
  const PairEnergyIntegral<RuleGenerator, PenaltyEnergyDensity<Gap>> energy{ rule_generator, { 7.0, gap } };
  const PairNodalGapIntegral<RuleGenerator, Gap, BasisFunction<EDGE, LINEAR_LAGRANGE>> nodal{ rule_generator, gap };
  const MdArray<RealT, decltype( energy )::size_coords> coordinates{ 0.0, 0.0, 1.7, 0.1, 1.6, 0.25, 0.25, 0.15 };
  SCOPED_TRACE( testing::Message() << "Gauss points: " << NumPoints );
  checkFiniteDifferences( energy, coordinates, MdArray<RealT, decltype( energy )::size_values>{ 1.0 } );
  checkFiniteDifferences( nodal, coordinates, MdArray<RealT, decltype( nodal )::size_values>{ 0.3, -0.2, 0.4, 0.1 } );
}

TEST( EnergyMortarFiniteDifference, smoothing_branch_with_production_quadrature )
{
  checkRuleGenerator<1>();
  checkRuleGenerator<2>();
  checkRuleGenerator<3>();
}

TEST( EnergyMortarFiniteDifference, invalid_geometry_reports_status_before_derivatives )
{
  using RuleGenerator = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>;
  using Energy = PairEnergyIntegral<RuleGenerator, PenaltyEnergyDensity<OpposingNormalGap>>;
  const Energy energy{ RuleGenerator{}, { 1.0 } };
  const MdArray<RealT, Energy::size_coords> coordinates{ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 };
  const typename Energy::Coords1 coords1{ coordinates.data(), Energy::size_geom1_coords };
  const typename Energy::Coords2 coords2{ coordinates.data() + Energy::size_geom1_coords, Energy::size_geom2_coords };
  const auto value_jacobian = EnzymePairDerivative<Energy>::valuesAndJacobian( coords1, coords2, energy );
  const auto value_jacobian_hessian = EnzymePairDerivative<Energy>::valueJacobianAndHessian( coords1, coords2, energy );
  EXPECT_NE( value_jacobian.pair_value.quadrature_metadata.status, QuadratureStatus::Success );
  EXPECT_NE( value_jacobian_hessian.pair_value.quadrature_metadata.status, QuadratureStatus::Success );
  EXPECT_FALSE( value_jacobian.pair_value.has_active_points );
  EXPECT_FALSE( value_jacobian_hessian.pair_value.has_active_points );
  EXPECT_EQ( value_jacobian.pair_value.values()[0], 0.0 );
  EXPECT_EQ( value_jacobian_hessian.pair_value.values()[0], 0.0 );
  for ( IndexT coordinate = 0; coordinate < Energy::size_coords; ++coordinate ) {
    EXPECT_EQ( value_jacobian.derivatives.jacobian( 0, coordinate ), 0.0 );
    EXPECT_EQ( value_jacobian_hessian.derivatives.jacobian[coordinate], 0.0 );
    for ( IndexT row = 0; row < Energy::size_coords; ++row ) {
      EXPECT_EQ( value_jacobian_hessian.derivatives.hessian( row, coordinate ), 0.0 );
    }
  }
}

// A residual gap changes the gap value and contact activation without changing the integration surface.
TEST( EnergyMortarValues, residual_gap_shifts_nodal_integrals_and_activates_energy )
{
  using RuleGenerator = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>;
  using Gap = OffsetGap<OpposingNormalGap>;
  using Nodal = PairNodalGapIntegral<RuleGenerator, Gap, BasisFunction<EDGE, LINEAR_LAGRANGE>>;
  using Energy = PairEnergyIntegral<RuleGenerator, PenaltyEnergyDensity<Gap>>;
  constexpr RealT separation = 0.1;
  constexpr RealT residual_gap = 0.15;
  constexpr RealT penalty = 3.0;
  constexpr RealT comparison_tolerance = 1.0e-14;
  const MdArray<RealT, Nodal::size_coords> coordinates{ 1.0, 0.0, 0.0, 0.0, 0.2, separation, 0.8, separation };
  const typename Nodal::Coords1 coords1{ coordinates.data(), Nodal::size_geom1_coords };
  const typename Nodal::Coords2 coords2{ coordinates.data() + Nodal::size_geom1_coords, Nodal::size_geom2_coords };
  const RuleGenerator rule_generator{ SmoothedProjectedEdgeMapping{ 0.0 } };
  const Nodal without_offset{ rule_generator, Gap{ OpposingNormalGap{}, 0.0 } };
  const Nodal with_offset{ rule_generator, Gap{ OpposingNormalGap{}, residual_gap } };
  const auto baseline = without_offset.evaluate( coords1, coords2 );
  const auto shifted = with_offset.evaluate( coords1, coords2 );
  ASSERT_EQ( baseline.quadrature_metadata.status, QuadratureStatus::Success );
  ASSERT_EQ( shifted.quadrature_metadata.status, QuadratureStatus::Success );
  ASSERT_TRUE( baseline.has_active_points );
  ASSERT_TRUE( shifted.has_active_points );
  for ( IndexT node = 0; node < Nodal::num_nodes; ++node ) {
    EXPECT_NEAR( shifted.tributaryArea( node ), baseline.tributaryArea( node ), comparison_tolerance );
    EXPECT_NEAR( shifted.weightedGap( node ),
                 baseline.weightedGap( node ) - residual_gap * baseline.tributaryArea( node ), comparison_tolerance );
  }

  const Energy open_energy{ rule_generator, { penalty, Gap{ OpposingNormalGap{}, 0.0 } } };
  const Energy active_energy{ rule_generator, { penalty, Gap{ OpposingNormalGap{}, residual_gap } } };
  const auto open = open_energy.evaluate( coords1, coords2 );
  const auto active = active_energy.evaluate( coords1, coords2 );
  ASSERT_EQ( open.quadrature_metadata.status, QuadratureStatus::Success );
  ASSERT_EQ( active.quadrature_metadata.status, QuadratureStatus::Success );
  EXPECT_FALSE( open.has_active_points );
  EXPECT_EQ( open.values()[0], 0.0 );
  EXPECT_TRUE( active.has_active_points );
  EXPECT_GT( active.values()[0], 0.0 );
}

// A short nonzero edge has a small area, but its normalized gap still equals its physical normal separation.
TEST( EnergyMortarValues, preserves_normal_gap_for_very_short_valid_edge )
{
  using RuleGenerator = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>;
  using Nodal = PairNodalGapIntegral<RuleGenerator, OpposingNormalGap, BasisFunction<EDGE, LINEAR_LAGRANGE>>;
  constexpr RealT edge_length = 1.0e-13;
  constexpr RealT separation = 0.25;
  constexpr RealT comparison_tolerance = 1.0e-12;
  const MdArray<RealT, Nodal::size_coords> coordinates{ 0.0,         0.0,        edge_length, 0.0,
                                                        edge_length, separation, 0.0,         separation };
  const Nodal nodal{ RuleGenerator{ SmoothedProjectedEdgeMapping{ 0.0 } } };
  const auto result = nodal.evaluate( { coordinates.data(), Nodal::size_geom1_coords },
                                      { coordinates.data() + Nodal::size_geom1_coords, Nodal::size_geom2_coords } );
  ASSERT_EQ( result.quadrature_metadata.status, QuadratureStatus::Success );
  ASSERT_TRUE( result.has_active_points );
  for ( IndexT node = 0; node < Nodal::num_nodes; ++node ) {
    ASSERT_GT( result.tributaryArea( node ), 0.0 );
    EXPECT_NEAR( result.weightedGap( node ) / result.tributaryArea( node ), -separation, comparison_tolerance );
  }
}

}  // namespace

/** @brief Run Energy Mortar derivative checks. @param argc Argument count. @param argv Arguments. @return Test status.
 */
int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
