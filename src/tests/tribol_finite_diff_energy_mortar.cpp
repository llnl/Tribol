// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Finite-difference checks for the composed Energy Mortar pair functions. */

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"
#include "tribol/assembly/PairEvaluation.hpp"

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
void checkFiniteDifferences( const Function& function, const MdArray<RealT, Function::num_coords>& coordinates,
                             const MdArray<RealT, Function::num_values>& weights )
{
  const auto base = EnzymePairDerivative<Function>::valuesAndJacobian(
      { coordinates.data(), Function::num_geom1_coords },
      { coordinates.data() + Function::num_geom1_coords, Function::num_geom2_coords }, function );
  const auto weighted = EnzymePairDerivative<Function>::weightedJacobianAndHessian(
      { coordinates.data(), Function::num_geom1_coords },
      { coordinates.data() + Function::num_geom1_coords, Function::num_geom2_coords }, function, weights.view() );
  for ( IndexT column = 0; column < Function::num_coords; ++column ) {
    auto positive_coordinates = coordinates;
    auto negative_coordinates = coordinates;
    positive_coordinates[column] += finite_difference_step;
    negative_coordinates[column] -= finite_difference_step;
    const auto positive = EnzymePairDerivative<Function>::valuesAndJacobian(
        { positive_coordinates.data(), Function::num_geom1_coords },
        { positive_coordinates.data() + Function::num_geom1_coords, Function::num_geom2_coords }, function );
    const auto negative = EnzymePairDerivative<Function>::valuesAndJacobian(
        { negative_coordinates.data(), Function::num_geom1_coords },
        { negative_coordinates.data() + Function::num_geom1_coords, Function::num_geom2_coords }, function );

    for ( IndexT value = 0; value < Function::num_values; ++value ) {
      EXPECT_NEAR( base.jacobian( value, column ),
                   ( positive.values[value] - negative.values[value] ) / ( 2.0 * finite_difference_step ),
                   jacobian_tolerance );
    }
    for ( IndexT row = 0; row < Function::num_coords; ++row ) {
      RealT positive_gradient = 0.0;
      RealT negative_gradient = 0.0;
      for ( IndexT value = 0; value < Function::num_values; ++value ) {
        positive_gradient += weights[value] * positive.jacobian( value, row );
        negative_gradient += weights[value] * negative.jacobian( value, row );
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
  const RuleGenerator rule_generator;
  const Gap gap{ {}, 0.05 };
  const EnergyMortarEnergy<RuleGenerator, Gap> energy{ rule_generator, 7.0, gap };
  const EnergyMortarNodalEnergy<RuleGenerator, Gap> nodal{ rule_generator, gap };
  const MdArray<RealT, decltype( energy )::num_coords> coordinates{ 0.0, 0.0, 2.0, 0.15, 1.7, 0.35, 0.3, 0.15 };
  SCOPED_TRACE( testing::Message() << "Gauss points: " << NumPoints );
  checkFiniteDifferences( energy, coordinates, MdArray<RealT, decltype( energy )::num_values>{ 1.0 } );
  checkFiniteDifferences( nodal, coordinates, MdArray<RealT, decltype( nodal )::num_values>{ 0.3, -0.2, 0.4, 0.1 } );
}

TEST( EnergyMortarFiniteDifference, production_quadrature )
{
  checkRuleGenerator<1>();
  checkRuleGenerator<2>();
  checkRuleGenerator<3>();
}

TEST( EnergyMortarFiniteDifference, invalid_geometry_reports_status_before_derivatives )
{
  using RuleGenerator = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>;
  const EnergyMortarEnergy<RuleGenerator> energy{ RuleGenerator{}, 1.0 };
  const MdArray<RealT, decltype( energy )::num_coords> coordinates{ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 };
  EXPECT_NE( energy
                 .evaluate( { coordinates.data(), decltype( energy )::num_geom1_coords },
                            { coordinates.data() + decltype( energy )::num_geom1_coords,
                              decltype( energy )::num_geom2_coords } )
                 .quadrature_result.status,
             QuadratureStatus::Success );
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
