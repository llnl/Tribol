// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check direct Energy Mortar energies, nodal integrals, and explicit field partials on enabled backends.
 * Host Enzyme checks compare the new pair definitions with finite differences for each Gauss count and derivative
 * mode. MFEM tests separately check assembly and shared-node normalization.
 */
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "ExecutionTest.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"

namespace {
using namespace tribol;
constexpr RealT tolerance = 1.0e-12;  ///< Roundoff allowance for primal integrals and exact partials.
using EnergyMortarDefinitionTest = detail::ExecutionTest;

/** @brief Check constant-gap integrals on an unsmoothed interval. @tparam Count Gauss point count.
 * @param mode Execution backend. @param allocator Backend-accessible allocation resource. */
template <int Count>
void checkParallelIntegrals( ExecutionMode mode, int allocator )
{
  using Rule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<Count>>;
  using Energy = EnergyMortarEnergy<Rule>;
  using Nodal = EnergyMortarNodalEnergy<Rule>;
  struct Result {
    PairEnergyValue energy;              ///< Quadrature-point energy and status.
    typename Nodal::PrimalResult nodal;  ///< Nodal integrals and status.
  };
  ArrayT<Result> results( 1, 1, allocator );
  const auto output = results.view();
  forAllExec( mode, 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const MdArray<RealT, Energy::dof_count> coordinates{ 0.0, 0.0, 2.0, 0.0, 3.0, 0.1, -1.0, 0.1 };
    const Rule rule( SmoothedProjectedEdgeMapping{ 0.0 } );
    output[0] = { Energy{ rule, 4.0 }.evaluate( coordinates.view() ), Nodal{ rule }.evaluate( coordinates.view() ) };
  } );
  const ArrayT<Result, 1, MemorySpace::Host> host( results );
  ASSERT_EQ( host[0].energy.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_EQ( host[0].energy.quadrature_result.required_point_count, Count );
  // A length-two edge with gap -0.1 has energy 0.5*4*0.1^2*2, and equal nodal areas of one.
  EXPECT_NEAR( host[0].energy.energy, 0.04, tolerance );
  EXPECT_TRUE( host[0].energy.has_active_points );
  ASSERT_EQ( host[0].nodal.quadrature_result.status, QuadratureStatus::Success );
  for ( IndexT node = 0; node < Nodal::node_count; ++node ) {
    EXPECT_NEAR( host[0].nodal.value[node], -0.1, tolerance );
    EXPECT_NEAR( host[0].nodal.value[Nodal::node_count + node], 1.0, tolerance );
  }
}

// Direct nodal and quadrature-point calculations must use physical weights exactly once for every Gauss count.
TEST_P( EnergyMortarDefinitionTest, integrates_constant_gap )
{
  checkParallelIntegrals<1>( GetParam(), allocatorId() );
  checkParallelIntegrals<2>( GetParam(), allocatorId() );
  checkParallelIntegrals<3>( GetParam(), allocatorId() );
}

// Unsupported areas and open gaps give zero partials; zero gap retains the compressive tangent.
TEST_P( EnergyMortarDefinitionTest, guards_area_before_dividing_and_keeps_zero_gap_tangent )
{
  struct Case {
    RealT gap;                           ///< Weighted nodal gap.
    RealT area;                          ///< Tributary area, including unsupported values on both sides of zero.
    EnergyMortarNodalPartials expected;  ///< Exact energy and field partials for stiffness four.
  };
  constexpr RealT area_tolerance = 1.0e-14;
  const auto cases = copyToExecutionSpace<Case>( { { -1.0, 2.0, { 1.0, -2.0, -0.5, 2.0, 1.0, 0.5 } },
                                                   { 0.0, 2.0, { 0.0, 0.0, 0.0, 2.0, 0.0, 0.0 } },
                                                   { 1.0, 2.0, {} },
                                                   { -1.0, 0.0, {} },
                                                   { -1.0, area_tolerance, {} },
                                                   { -1.0, -area_tolerance, {} } } );
  ArrayT<EnergyMortarNodalPartials> results( cases.size(), cases.size(), allocatorId() );
  const auto input = cases.view();
  const auto output = results.view();
  forAllExec( GetParam(), cases.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
    output[i] = EnergyMortarNodal<>::penaltyPartials( input[i].gap, input[i].area, 4.0, area_tolerance );
  } );
  const ArrayT<Case, 1, MemorySpace::Host> host_cases( cases );
  const ArrayT<EnergyMortarNodalPartials, 1, MemorySpace::Host> host_results( results );
  for ( IndexT i = 0; i < cases.size(); ++i ) {
    SCOPED_TRACE( i );
    const auto& actual = host_results[i];
    const auto& expected = host_cases[i].expected;
    EXPECT_NEAR( actual.energy, expected.energy, tolerance );
    EXPECT_NEAR( actual.gap, expected.gap, tolerance );
    EXPECT_NEAR( actual.area, expected.area, tolerance );
    EXPECT_NEAR( actual.gap_gap, expected.gap_gap, tolerance );
    EXPECT_NEAR( actual.gap_area, expected.gap_area, tolerance );
    EXPECT_NEAR( actual.area_area, expected.area_area, tolerance );
  }
}
INSTANTIATE_TEST_SUITE_P( backends, EnergyMortarDefinitionTest, detail::enabledExecutionModes() );

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Compare an energy definition's gradient and Hessian with directional coordinate differences.
 * @tparam Definition Scalar energy with dof_count and energy().
 * @param definition Fixed configuration, possibly holding a frozen reference rule.
 * @param coordinates Current oblique geometry, away from activation and overlap switches.
 */
template <typename Definition>
void checkEnergyDerivatives( const Definition& definition, const MdArray<RealT, Definition::dof_count>& coordinates )
{
  constexpr RealT step = 1.0e-6;
  const auto base = EnergyPairEvaluator<Definition>::evaluate( coordinates.view(), definition );
  for ( IndexT column = 0; column < Definition::dof_count; ++column ) {
    auto plus = coordinates, minus = coordinates;
    plus[column] += step;
    minus[column] -= step;
    const auto positive = EnergyPairEvaluator<Definition>::evaluate( plus.view(), definition );
    const auto negative = EnergyPairEvaluator<Definition>::evaluate( minus.view(), definition );
    EXPECT_NEAR( base.residual[column], ( positive.energy - negative.energy ) / ( 2.0 * step ), 1.0e-6 );
    for ( IndexT row = 0; row < Definition::dof_count; ++row )
      EXPECT_NEAR( base.jacobian( row, column ), ( positive.residual[row] - negative.residual[row] ) / ( 2.0 * step ),
                   2.0e-5 );
  }
}

/** @brief Check both pair definitions for varying and frozen quadrature. @tparam Count Gauss point count. */
template <int Count>
void checkRuleDerivatives()
{
  using Rule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<Count>>;
  using Gap = OffsetGap<OpposingNormalGap>;
  const Rule rule;
  const Gap gap{ {}, 0.05 };
  const EnergyMortarEnergy<Rule, Gap> energy{ rule, 7.0, EnergyMortarPenalty{ gap } };
  const EnergyMortarNodalEnergy<Rule, Gap> nodal{ rule, EnergyMortarNodal{ gap } };
  const MdArray<RealT, decltype( energy )::dof_count> coordinates{ 0.0, 0.0, 2.0, 0.15, 1.7, 0.35, 0.3, 0.15 };
  const auto check = [&]( const auto& selected_energy, const auto& selected_nodal ) {
    ASSERT_EQ( selected_nodal.evaluate( coordinates.view() ).quadrature_result.status, QuadratureStatus::Success );
    checkEnergyDerivatives( selected_energy, coordinates );
    checkEnergyDerivatives( selected_nodal.withWeights( MdArray<RealT, 2>{ 0.3, -0.2 }, MdArray<RealT, 2>{ 0.4, 0.1 } ),
                            coordinates );
    const auto derivatives = VectorPairEvaluator<std::remove_cvref_t<decltype( selected_nodal )>>::evaluateGradients(
        coordinates.view(), selected_nodal );
    constexpr RealT step = 1.0e-6;
    for ( IndexT column = 0; column < decltype( nodal )::dof_count; ++column ) {
      auto plus = coordinates, minus = coordinates;
      plus[column] += step;
      minus[column] -= step;
      const auto positive = selected_nodal.values( plus.view() ), negative = selected_nodal.values( minus.view() );
      for ( IndexT row = 0; row < decltype( nodal )::value_count; ++row )
        EXPECT_NEAR( derivatives.gradients( row, column ), ( positive[row] - negative[row] ) / ( 2.0 * step ), 1.0e-6 );
    }
  };
  SCOPED_TRACE( testing::Message() << "points=" << Count );
  {
    SCOPED_TRACE( "varying quadrature" );
    check( energy, nodal );
  }
  {
    SCOPED_TRACE( "frozen quadrature" );
    check( energy.withFrozenQuadrature( coordinates.view() ), nodal.withFrozenQuadrature( coordinates.view() ) );
  }
}

// Perturbing every coordinate changes lengths and normals as well as overlap bounds for all supported rules.
TEST( EnergyMortarDerivatives, differentiates_direct_definitions )
{
  checkRuleDerivatives<1>();
  checkRuleDerivatives<2>();
  checkRuleDerivatives<3>();
}
#endif
}  // namespace

/** @brief Run primal and optional Enzyme tests. @param argc Argument count. @param argv Arguments. @return Status. */
int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
