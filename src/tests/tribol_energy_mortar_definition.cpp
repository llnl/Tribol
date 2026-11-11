// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check direct Energy Mortar energies, nodal integrals, and explicit field partials on enabled backends.
 * Host Enzyme checks compare the new pair definitions with finite differences for each Gauss count and derivative
 * mode. The energy kernel checks active, open, zero-gap, and invalid pairs on every enabled backend, including the
 * compressive zero-gap Hessian. MFEM tests separately check assembly and shared-node normalization.
 */
#include <type_traits>

#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "ExecutionTest.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"
#include "tribol/physics/PairContribution.hpp"
#include "tribol/physics/PairEvaluation.hpp"
#include "tribol/physics/PairKernel.hpp"

namespace {
using namespace tribol;
constexpr RealT tolerance = 1.0e-12;  ///< Roundoff allowance for primal integrals and exact partials.
using EnergyMortarDefinitionTest = detail::ExecutionTest;

/** @brief Pair energy and nodal integrals written by one backend thread. @tparam Count Gauss point count. */
template <int Count>
struct ParallelIntegralResult {
  using Rule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<Count>>;
  PairEnergyValue energy;                                      ///< Quadrature-point energy and status.
  typename EnergyMortarNodalEnergy<Rule>::PrimalResult nodal;  ///< Nodal integrals and status.
};

/** @brief Nodal field inputs and exact penalty partials for one backend test case. */
struct NodalPartialsCase {
  RealT gap;                           ///< Weighted nodal gap.
  RealT area;                          ///< Tributary area, including unsupported values on both sides of zero.
  EnergyMortarNodalPartials expected;  ///< Exact energy and field partials for stiffness four.
};

/** @brief Check constant-gap integrals on an unsmoothed interval. @tparam Count Gauss point count.
 * @param mode Execution backend. @param allocator Backend-accessible allocation resource. */
template <int Count>
void checkParallelIntegrals( ExecutionMode mode, int allocator )
{
  using Rule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<Count>>;
  using Energy = EnergyMortarEnergy<Rule>;
  using Nodal = EnergyMortarNodalEnergy<Rule>;
  ArrayT<ParallelIntegralResult<Count>> results( 1, 1, allocator );
  const auto output = results.view();
  forAllExec( mode, 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const MdArray<RealT, Energy::dof_count> coordinates{ 0.0, 0.0, 2.0, 0.0, 3.0, 0.1, -1.0, 0.1 };
    const Rule rule( SmoothedProjectedEdgeMapping{ 0.0 } );
    output[0] = { Energy{ rule, 4.0 }.evaluate( coordinates.view() ), Nodal{ rule }.evaluate( coordinates.view() ) };
  } );
  const ArrayT<ParallelIntegralResult<Count>, 1, MemorySpace::Host> host( results );
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
  constexpr RealT area_tolerance = 1.0e-14;
  const auto cases = copyToExecutionSpace<NodalPartialsCase>( { { -1.0, 2.0, { 1.0, -2.0, -0.5, 2.0, 1.0, 0.5 } },
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
  const ArrayT<NodalPartialsCase, 1, MemorySpace::Host> host_cases( cases );
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

TEST_P( EnergyMortarDefinitionTest, contribution_buffers_use_backend_accessible_structure_of_arrays )
{
  constexpr IndexT pair_count = 3;
  PairValueJacobianBuffer<2, 2> values( pair_count, allocatorId() );
  ElementBlockBuffer<2, 2> blocks( pair_count, allocatorId() );
  const auto value_view = values.view();
  const auto block_view = blocks.view();
  forAllExec( GetParam(), pair_count, [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
    value_view.setValue( pair, 0, 10.0 + pair );
    value_view.setValue( pair, 1, 20.0 + pair );
    value_view.setJacobian( pair, 0, 0, 30.0 + pair );
    value_view.setJacobian( pair, 0, 1, 40.0 + pair );
    value_view.setJacobian( pair, 1, 0, 50.0 + pair );
    value_view.setJacobian( pair, 1, 1, 60.0 + pair );
    block_view.setElements( pair, pair, pair + 4 );
    for ( IndexT column = 0; column < 2; ++column ) {
      for ( IndexT row = 0; row < 2; ++row ) {
        block_view.setValue( pair, row, column, 100.0 * pair + 10.0 * column + row );
      }
    }
  } );

  const Array1D<RealT, MemorySpace::Host> host_values( values.values() );
  const Array1D<RealT, MemorySpace::Host> host_jacobians( values.jacobians() );
  const Array1D<IndexT, MemorySpace::Host> host_rows( blocks.rowElementIds() );
  const Array1D<IndexT, MemorySpace::Host> host_columns( blocks.columnElementIds() );
  const Array1D<IndexT, MemorySpace::Host> host_offsets( blocks.valueOffsets() );
  const Array1D<RealT, MemorySpace::Host> host_blocks( blocks.values() );
  for ( IndexT pair = 0; pair < pair_count; ++pair ) {
    EXPECT_DOUBLE_EQ( host_values[pair], 10.0 + pair );
    EXPECT_DOUBLE_EQ( host_values[pair_count + pair], 20.0 + pair );
    EXPECT_DOUBLE_EQ( host_jacobians[pair], 30.0 + pair );
    EXPECT_DOUBLE_EQ( host_jacobians[pair_count + pair], 40.0 + pair );
    EXPECT_DOUBLE_EQ( host_jacobians[2 * pair_count + pair], 50.0 + pair );
    EXPECT_DOUBLE_EQ( host_jacobians[3 * pair_count + pair], 60.0 + pair );
    EXPECT_EQ( host_rows[pair], pair );
    EXPECT_EQ( host_columns[pair], pair + 4 );
    EXPECT_EQ( host_offsets[pair], 4 * pair );
    for ( IndexT column = 0; column < 2; ++column ) {
      for ( IndexT row = 0; row < 2; ++row ) {
        EXPECT_DOUBLE_EQ( host_blocks[4 * pair + row + 2 * column], 100.0 * pair + 10.0 * column + row );
      }
    }
  }
}

#ifdef TRIBOL_USE_ENZYME
// A rigid vertical edge translation has analytic energy derivatives; zero gap retains the compressive tangent.
TEST_P( EnergyMortarDefinitionTest, energy_kernel_returns_values_gradients_and_hessians )
{
  using Rule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>;
  using Function = EnergyMortarEnergy<Rule>;
  const Function function{ Rule{ SmoothedProjectedEdgeMapping{ 0.0 } }, 4.0 };
  const auto x1 = copyToExecutionSpace<RealT>( { 0.0, 2.0 } );
  const auto y1 = copyToExecutionSpace<RealT>( { 0.0, 0.0 } );
  const auto conn1 = copyToExecutionSpace<IndexT>( { 0, 1 } );
  const auto x2 = copyToExecutionSpace<RealT>( { 3.0, -1.0, 3.0, -1.0, 3.0, -1.0, 0.0, 0.0 } );
  const auto y2 = copyToExecutionSpace<RealT>( { 0.1, 0.1, -0.1, -0.1, 0.0, 0.0, 0.0, 0.0 } );
  const auto conn2 = copyToExecutionSpace<IndexT>( { 0, 1, 2, 3, 4, 5, 6, 7 } );
  MeshData mesh1( 0, 1, x1.size(), conn1.data(), EDGE, LINEAR_LAGRANGE, x1.data(), y1.data(), nullptr, memorySpace() );
  MeshData mesh2( 1, 4, x2.size(), conn2.data(), EDGE, LINEAR_LAGRANGE, x2.data(), y2.data(), nullptr, memorySpace() );
  const auto pairs = copyToExecutionSpace<InterfacePair>( { { 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, 3 } } );
  PairEnergyHessianBuffer<Function::dof_count> buffer( pairs.size(), allocatorId() );
  const auto output = buffer.view();
  evaluatePairEnergiesAndHessians( GetParam(), pairs.view(), mesh1.getView(), mesh2.getView(), function, output );

  ArrayT<RealT, 2> summaries( axom::StackArray<axom::IndexType, 2>{ pairs.size(), 4 }, allocatorId() );
  const auto summary = summaries.view();
  forAllExec( GetParam(), pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
    summary( pair, 0 ) = output.energies[pair];
    summary( pair, 1 ) = output.derivatives.gradient( pair, 1 ) + output.derivatives.gradient( pair, 3 );
    summary( pair, 2 ) = 0.0;
    summary( pair, 3 ) = 0.0;
    for ( IndexT row = 0; row < Function::dof_count; ++row ) {
      summary( pair, 3 ) += std::abs( output.derivatives.gradient( pair, row ) );
      for ( IndexT column = 0; column < Function::dof_count; ++column ) {
        summary( pair, 3 ) += std::abs( output.derivatives.hessian( pair, row, column ) );
        if ( ( row == 1 || row == 3 ) && ( column == 1 || column == 3 ) ) {
          summary( pair, 2 ) += output.derivatives.hessian( pair, row, column );
        }
      }
    }
  } );
  const ArrayT<RealT, 2, MemorySpace::Host> host( summaries );
  const Array1D<QuadratureResult, MemorySpace::Host> status( buffer.quadratureResults() );
  // Length two and stiffness four give E=0.04, dE/dy=-0.8, and d2E/dy2=8 for the penetrating pair.
  EXPECT_NEAR( host( 0, 0 ), 0.04, tolerance );
  EXPECT_NEAR( host( 0, 1 ), -0.8, tolerance );
  EXPECT_NEAR( host( 0, 2 ), 8.0, tolerance );
  EXPECT_NEAR( host( 2, 0 ), 0.0, tolerance );
  EXPECT_NEAR( host( 2, 1 ), 0.0, tolerance );
  EXPECT_NEAR( host( 2, 2 ), 8.0, tolerance );
  for ( IndexT pair : { 0, 1, 2 } ) {
    EXPECT_EQ( status[pair].status, QuadratureStatus::Success );
  }
  EXPECT_EQ( status[3].status, QuadratureStatus::InvalidGeometry );
  for ( IndexT pair : { 1, 3 } ) {
    EXPECT_EQ( host( pair, 0 ), 0.0 );
    EXPECT_EQ( host( pair, 3 ), 0.0 );
  }
  EXPECT_THROW( checkPairQuadratureResults( GetParam(), buffer.quadratureResults() ), std::invalid_argument );
}

/** @brief Small fixed-size pair function used to compile and execute both Enzyme operations on every backend. */
struct PolynomialPairFunction {
  static constexpr IndexT value_count = 2;  ///< Number of outputs.
  static constexpr IndexT dof_count = 2;    ///< Number of active coordinates.

  /** @brief Evaluate two coupled polynomial outputs. @param coordinates Active coordinates. @return Function values. */
  TRIBOL_HOST_DEVICE MdArray<RealT, value_count> values( MdSpan<const RealT, dof_count> coordinates ) const
  {
    return MdArray<RealT, value_count>{ coordinates[0] * coordinates[0] + coordinates[1],
                                        coordinates[0] * coordinates[1] };
  }
};

/** @brief Device-storable output from both pair derivative operations. */
struct PolynomialDerivativeResult {
  PairVectorGradients<PolynomialPairFunction::value_count, PolynomialPairFunction::dof_count> value_jacobian;
  PairWeightedHessian<PolynomialPairFunction::dof_count> weighted_hessian;
};

TEST_P( EnergyMortarDefinitionTest, enzyme_pair_derivatives_execute_on_backend )
{
  Array1D<PolynomialDerivativeResult> results( 1, 1, allocatorId() );
  auto output = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const MdArray<RealT, PolynomialPairFunction::dof_count> coordinates{ 3.0, 5.0 };
    const MdArray<RealT, PolynomialPairFunction::value_count> weights{ 2.0, -1.0 };
    output[0].value_jacobian =
        EnzymePairDerivative<PolynomialPairFunction>::valuesAndJacobian( coordinates.view(), {} );
    output[0].weighted_hessian =
        EnzymePairDerivative<PolynomialPairFunction>::weightedHessian( coordinates.view(), {}, weights.view() );
  } );
  const Array1D<PolynomialDerivativeResult, MemorySpace::Host> host_results( results );
  const auto& value_jacobian = host_results[0].value_jacobian;
  EXPECT_DOUBLE_EQ( value_jacobian.values[0], 14.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.values[1], 15.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.gradients( 0, 0 ), 6.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.gradients( 0, 1 ), 1.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.gradients( 1, 0 ), 5.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.gradients( 1, 1 ), 3.0 );
  const auto& weighted_hessian = host_results[0].weighted_hessian;
  EXPECT_DOUBLE_EQ( weighted_hessian.gradient[0], 7.0 );
  EXPECT_DOUBLE_EQ( weighted_hessian.gradient[1], -1.0 );
  EXPECT_DOUBLE_EQ( weighted_hessian.hessian( 0, 0 ), 4.0 );
  EXPECT_DOUBLE_EQ( weighted_hessian.hessian( 0, 1 ), -1.0 );
  EXPECT_DOUBLE_EQ( weighted_hessian.hessian( 1, 0 ), -1.0 );
  EXPECT_DOUBLE_EQ( weighted_hessian.hessian( 1, 1 ), 0.0 );
}

/**
 * @brief Compare a weighted pair function's derivatives with directional coordinate differences.
 * @tparam Function Vector pair function with fixed output and coordinate counts.
 * @param function Fixed configuration, possibly holding a frozen reference rule.
 * @param coordinates Current oblique geometry, away from activation and overlap switches.
 * @param weights Scalar contraction used for the gradient and Hessian check.
 */
template <typename Function>
void checkPairDerivatives( const Function& function, const MdArray<RealT, Function::dof_count>& coordinates,
                           const MdArray<RealT, Function::value_count>& weights )
{
  constexpr RealT step = 1.0e-6;
  const auto base = EnzymePairDerivative<Function>::valuesAndJacobian( coordinates.view(), function );
  const auto weighted = EnzymePairDerivative<Function>::weightedHessian( coordinates.view(), function, weights.view() );
  for ( IndexT column = 0; column < Function::dof_count; ++column ) {
    auto plus = coordinates, minus = coordinates;
    plus[column] += step;
    minus[column] -= step;
    const auto positive = EnzymePairDerivative<Function>::valuesAndJacobian( plus.view(), function );
    const auto negative = EnzymePairDerivative<Function>::valuesAndJacobian( minus.view(), function );
    RealT positive_scalar = 0.0;
    RealT negative_scalar = 0.0;
    for ( IndexT value = 0; value < Function::value_count; ++value ) {
      positive_scalar += weights[value] * positive.values[value];
      negative_scalar += weights[value] * negative.values[value];
      EXPECT_NEAR( base.gradients( value, column ),
                   ( positive.values[value] - negative.values[value] ) / ( 2.0 * step ), 1.0e-6 );
    }
    EXPECT_NEAR( weighted.gradient[column], ( positive_scalar - negative_scalar ) / ( 2.0 * step ), 1.0e-6 );
    for ( IndexT row = 0; row < Function::dof_count; ++row ) {
      RealT positive_gradient = 0.0;
      RealT negative_gradient = 0.0;
      for ( IndexT value = 0; value < Function::value_count; ++value ) {
        positive_gradient += weights[value] * positive.gradients( value, row );
        negative_gradient += weights[value] * negative.gradients( value, row );
      }
      EXPECT_NEAR( weighted.hessian( row, column ), ( positive_gradient - negative_gradient ) / ( 2.0 * step ),
                   2.0e-5 );
    }
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
    using SelectedEnergy = std::remove_cvref_t<decltype( selected_energy )>;
    using SelectedNodal = std::remove_cvref_t<decltype( selected_nodal )>;
    ASSERT_EQ( selected_nodal.evaluate( coordinates.view() ).quadrature_result.status, QuadratureStatus::Success );
    checkPairDerivatives( selected_energy, coordinates, MdArray<RealT, SelectedEnergy::value_count>{ 1.0 } );
    checkPairDerivatives( selected_nodal, coordinates,
                          MdArray<RealT, SelectedNodal::value_count>{ 0.3, -0.2, 0.4, 0.1 } );
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
