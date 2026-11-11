// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Finite-difference checks for the composed Energy Mortar pair functions. */

#include <type_traits>

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"
#include "tribol/physics/PairEvaluation.hpp"

namespace {
using namespace tribol;

constexpr RealT finite_difference_step = 1.0e-6;  ///< Stable central-difference step for the oblique edge geometry.
constexpr RealT jacobian_tolerance = 1.0e-6;      ///< First-derivative comparison tolerance.
constexpr RealT hessian_tolerance = 2.0e-5;       ///< Second-derivative comparison tolerance.

/**
 * @brief Check one vector pair function and weighted contraction against coordinate differences.
 * @tparam Function Fixed-size Energy Mortar pair function.
 * @param function Pair function with varying or frozen quadrature.
 * @param coordinates Current pair coordinates away from overlap and activation transitions.
 * @param weights Output weights used for the Hessian contraction.
 */
template <typename Function>
void checkFiniteDifferences( const Function& function, const MdArray<RealT, Function::dof_count>& coordinates,
                             const MdArray<RealT, Function::value_count>& weights )
{
  const auto base = EnzymePairDerivative<Function>::valuesAndJacobian( coordinates.view(), function );
  const auto weighted = EnzymePairDerivative<Function>::weightedHessian( coordinates.view(), function, weights.view() );
  for ( IndexT column = 0; column < Function::dof_count; ++column ) {
    auto positive_coordinates = coordinates;
    auto negative_coordinates = coordinates;
    positive_coordinates[column] += finite_difference_step;
    negative_coordinates[column] -= finite_difference_step;
    const auto positive = EnzymePairDerivative<Function>::valuesAndJacobian( positive_coordinates.view(), function );
    const auto negative = EnzymePairDerivative<Function>::valuesAndJacobian( negative_coordinates.view(), function );

    for ( IndexT value = 0; value < Function::value_count; ++value ) {
      EXPECT_NEAR( base.gradients( value, column ),
                   ( positive.values[value] - negative.values[value] ) / ( 2.0 * finite_difference_step ),
                   jacobian_tolerance );
    }
    for ( IndexT row = 0; row < Function::dof_count; ++row ) {
      RealT positive_gradient = 0.0;
      RealT negative_gradient = 0.0;
      for ( IndexT value = 0; value < Function::value_count; ++value ) {
        positive_gradient += weights[value] * positive.gradients( value, row );
        negative_gradient += weights[value] * negative.gradients( value, row );
      }
      EXPECT_NEAR( weighted.hessian( row, column ),
                   ( positive_gradient - negative_gradient ) / ( 2.0 * finite_difference_step ), hessian_tolerance );
    }
  }
}

/** @brief Check varying and frozen quadrature derivatives. @tparam PointCount Gauss point count. */
template <int PointCount>
void checkRule()
{
  using Rule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<PointCount>>;
  using Gap = OffsetGap<OpposingNormalGap>;
  const Rule rule;
  const Gap gap{ {}, 0.05 };
  const EnergyMortarEnergy<Rule, Gap> energy{ rule, 7.0, EnergyMortarPenalty{ gap } };
  const EnergyMortarNodalEnergy<Rule, Gap> nodal{ rule, EnergyMortarNodal{ gap } };
  const MdArray<RealT, decltype( energy )::dof_count> coordinates{ 0.0, 0.0, 2.0, 0.15, 1.7, 0.35, 0.3, 0.15 };
  const auto check = [&]( const auto& selected_energy, const auto& selected_nodal ) {
    using SelectedEnergy = std::remove_cvref_t<decltype( selected_energy )>;
    using SelectedNodal = std::remove_cvref_t<decltype( selected_nodal )>;
    checkFiniteDifferences( selected_energy, coordinates, MdArray<RealT, SelectedEnergy::value_count>{ 1.0 } );
    checkFiniteDifferences( selected_nodal, coordinates,
                            MdArray<RealT, SelectedNodal::value_count>{ 0.3, -0.2, 0.4, 0.1 } );
  };

  SCOPED_TRACE( testing::Message() << "Gauss points: " << PointCount );
  check( energy, nodal );
  check( energy.withFrozenQuadrature( coordinates.view() ), nodal.withFrozenQuadrature( coordinates.view() ) );
}

TEST( EnergyMortarFiniteDifference, varying_and_frozen_quadrature )
{
  checkRule<1>();
  checkRule<2>();
  checkRule<3>();
}

TEST( EnergyMortarFiniteDifference, invalid_geometry_reports_status_before_derivatives )
{
  using Rule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>;
  const EnergyMortarEnergy<Rule> energy{ Rule{}, 1.0, EnergyMortarPenalty{} };
  const MdArray<RealT, decltype( energy )::dof_count> coordinates{ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 };
  EXPECT_NE( energy.evaluate( coordinates.view() ).quadrature_result.status, QuadratureStatus::Success );
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
