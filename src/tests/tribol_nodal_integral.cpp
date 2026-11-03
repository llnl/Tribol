// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check named assembled-integral expressions independently of MFEM and pair geometry.
 * Portable tests evaluate penalty and LM energies on every enabled backend, including repeated fields, reordered
 * products, general arithmetic, fixed coefficients, activation, and unsupported areas. Operator AD runs on each
 * backend. Host Enzyme tests check outer gradients and Hessians analytically and by finite differences. MFEM
 * formulation tests cover coordinate derivatives and multi-rank assembly.
 */
#include <array>
#include "ExecutionTest.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/integ/NodalIntegralExpressions.hpp"
#include "tribol/physics/PairEvaluation.hpp"

namespace {
using namespace tribol;
constexpr RealT tolerance = 1.0e-12;       ///< Roundoff allowance for simple nodal energies and derivatives.
constexpr RealT area_tolerance = 1.0e-14;  ///< Support threshold used by nodal contact assembly.
constexpr RealT penalty = 4.0;             ///< Stiffness giving energy 0.16 for gap -0.2 and area 0.5.

/** @brief Run pure nodal expressions in memory accessible to each enabled backend. */
class NodalIntegralTest : public detail::ExecutionTest {};

// Gap and area are distinct named inputs; repeated gaps and reordered products must produce the same energy.
TEST_P( NodalIntegralTest, evaluates_penalty_activation_and_area_support )
{
  struct Case {
    RealT gap;       ///< Assembled weighted gap.
    RealT area;      ///< Assembled area.
    RealT expected;  ///< Expected penalty energy, including zero for unsupported nodes.
  };
  const auto cases = copyToExecutionSpace<Case>( { { -0.2, 0.5, 0.16 },
                                                   { 0.2, 0.5, 0.0 },
                                                   { 0.0, 0.5, 0.0 },
                                                   { -0.2, 0.0, 0.0 },
                                                   { 0.0, 0.0, 0.0 },
                                                   { -0.2, 0.5 * area_tolerance, 0.0 },
                                                   { -0.2, area_tolerance, 0.0 } } );
  ArrayT<MdArray<RealT, 2>> results( cases.size(), cases.size(), allocatorId() );
  const auto input = cases.view();
  const auto output = results.view();
  forAllExec( GetParam(), cases.size(), [=] TRIBOL_HOST_DEVICE( IndexT node ) {
    const auto g = min( nodal_weighted_gap, 0.0 );
    const auto energy = 0.5 * penalty * sum( g, g, 1.0 / nodal_area );
    const auto reordered = sum( 1.0 / nodal_area, g, g ) * ( 0.5 * penalty );
    const MdArray<RealT, nodal_integral_count> state{ input[node].gap, input[node].area };
    output[node][0] = energy.atNode( node, area_tolerance ).energy( state.view() );
    output[node][1] = reordered.atNode( node, area_tolerance ).energy( state.view() );
  } );
  const ArrayT<Case, 1, MemorySpace::Host> host_input( cases );
  const ArrayT<MdArray<RealT, 2>, 1, MemorySpace::Host> host( results );
  for ( IndexT node = 0; node < host.size(); ++node ) {
    SCOPED_TRACE( node );
    EXPECT_NEAR( host[node][0], host_input[node].expected, tolerance );
    EXPECT_NEAR( host[node][1], host_input[node].expected, tolerance );
  }
}

// LM values are independent coefficients; their energy needs no area and must not clamp an open weighted gap.
TEST_P( NodalIntegralTest, contracts_independent_pressure_with_gap )
{
  const auto coefficients = copyToExecutionSpace<RealT>( { -3.0, 2.0 } );
  const IndependentNodalField pressure{ coefficients.view() };
  const auto energy = dot( pressure, nodal_weighted_gap );
  static_assert( !decltype( energy )::dependsOn( 1 ) );
  static_assert( !decltype( energy )::has_outer_hessian );
  EXPECT_NO_THROW( energy.validateSize( coefficients.size() ) );
  EXPECT_THROW( energy.validateSize( coefficients.size() + 1 ), std::invalid_argument );
  ArrayT<SecondOrderScalar<1>> results( coefficients.size(), coefficients.size(), allocatorId() );
  const auto output = results.view();
  forAllExec( GetParam(), coefficients.size(), [=] TRIBOL_HOST_DEVICE( IndexT node ) {
    const MdArray<SecondOrderScalar<1>, 1> state{ SecondOrderScalar<1>::variable( 0.2, 0 ) };
    output[node] = energy.atNode( node, area_tolerance ).energy( state.view() );
  } );
  const ArrayT<SecondOrderScalar<1>, 1, MemorySpace::Host> host( results );
  EXPECT_NEAR( host[0].value, -0.6, tolerance );
  EXPECT_NEAR( host[1].value, 0.4, tolerance );
  EXPECT_EQ( host[0].gradient[0], -3.0 );
  EXPECT_EQ( host[1].gradient[0], 2.0 );
  for ( const auto& value : host ) EXPECT_EQ( value.hessian( 0, 0 ), 0.0 );
}

// Three fields use distinct indices without assigning gap or area meanings to them.
TEST_P( NodalIntegralTest, evaluates_three_indexed_fields )
{
  const auto input = copyToExecutionSpace<MdArray<RealT, 3>>( { MdArray<RealT, 3>{ 2.0, 3.0, 4.0 } } );
  ArrayT<RealT> results( input.size(), input.size(), allocatorId() );
  const auto states = input.view();
  const auto output = results.view();
  forAllExec( GetParam(), input.size(), [=] TRIBOL_HOST_DEVICE( IndexT node ) {
    const auto energy = sum( NodalFieldVariable<0>{}, NodalFieldVariable<1>{}, NodalFieldVariable<2>{} );
    output[node] = energy.atNode( node, area_tolerance ).energy( states[node].view() );
  } );
  const ArrayT<RealT, 1, MemorySpace::Host> host( results );
  EXPECT_NEAR( host[0], 24.0, tolerance );
}

// General field quotients and scalar constants must propagate derivatives through the complete nodal expression.
TEST_P( NodalIntegralTest, differentiates_general_nodal_arithmetic )
{
  using Scalar = SecondOrderScalar<2>;
  ArrayT<Scalar> results( 1, 1, allocatorId() );
  const auto output = results.view();
  const auto a = NodalFieldVariable<0>{};
  const auto b = NodalFieldVariable<1>{};
  const auto expression = sum( ( a + b ) / ( a - b ) + a / 2.0 + ( 2.0 - b ) - 1.0 + 2.0 / ( a + b ) );
  forAllExec( GetParam(), results.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
    const MdArray<Scalar, 2> state{ Scalar::variable( 3.0, 0 ), Scalar::variable( 1.0, 1 ) };
    output[i] = expression.atNode( i, area_tolerance ).energy( state.view() );
  } );
  const ArrayT<Scalar, 1, MemorySpace::Host> host( results );
  // At a=3, b=1, E=(a+b)/(a-b) + a/2 + (2-b) - 1 + 2/(a+b).
  EXPECT_NEAR( host[0].value, 4.0, tolerance );
  EXPECT_NEAR( host[0].gradient[0], -0.125, tolerance );
  EXPECT_NEAR( host[0].gradient[1], 0.375, tolerance );
  EXPECT_NEAR( host[0].hessian( 0, 0 ), 0.5625, tolerance );
  EXPECT_NEAR( host[0].hessian( 0, 1 ), -0.9375, tolerance );
  EXPECT_NEAR( host[0].hessian( 1, 0 ), -0.9375, tolerance );
  EXPECT_NEAR( host[0].hessian( 1, 1 ), 1.5625, tolerance );
#ifdef TRIBOL_USE_ENZYME
  auto definition = expression.atNode( 0, area_tolerance );
  const MdArray<RealT, 2> state{ 3.0, 1.0 };
  const auto enzyme = EnergyPairEvaluator<decltype( definition )>::evaluate( state.view(), definition );
  EXPECT_NEAR( host[0].value, enzyme.energy, tolerance );
  for ( IndexT i = 0; i < Scalar::variable_count; ++i ) {
    EXPECT_NEAR( host[0].gradient[i], enzyme.residual[i], tolerance );
    for ( IndexT j = 0; j < Scalar::variable_count; ++j )
      EXPECT_NEAR( host[0].hessian( i, j ), enzyme.jacobian( i, j ), tolerance );
  }
#endif
}

// Operator AD must retain the compressive tangent at equality and avoid division on unsupported area.
TEST_P( NodalIntegralTest, preserves_activation_and_area_support_with_operator_ad )
{
  using Scalar = SecondOrderScalar<2>;
  struct Case {
    RealT gap;            ///< Weighted gap.
    RealT area;           ///< Tributary area.
    RealT gap_curvature;  ///< Expected second derivative with respect to weighted gap.
  };
  const auto cases = copyToExecutionSpace<Case>(
      { { 0.0, 0.5, 8.0 }, { 0.2, 0.5, 0.0 }, { -0.2, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { -0.2, area_tolerance, 0.0 } } );
  ArrayT<Scalar> results( cases.size(), cases.size(), allocatorId() );
  const auto input = cases.view();
  const auto output = results.view();
  forAllExec( GetParam(), cases.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
    const auto g = min( nodal_weighted_gap, 0.0 );
    const auto expression = 0.5 * penalty * sum( g, g, 1.0 / nodal_area );
    const MdArray<Scalar, 2> state{ Scalar::variable( input[i].gap, 0 ), Scalar::variable( input[i].area, 1 ) };
    output[i] = expression.atNode( i, area_tolerance ).energy( state.view() );
  } );
  const ArrayT<Case, 1, MemorySpace::Host> expected( cases );
  const ArrayT<Scalar, 1, MemorySpace::Host> host( results );
  for ( IndexT i = 0; i < host.size(); ++i ) {
    SCOPED_TRACE( i );
    EXPECT_EQ( host[i].value, 0.0 );
    for ( IndexT j = 0; j < Scalar::variable_count; ++j ) EXPECT_EQ( host[i].gradient[j], 0.0 );
    EXPECT_NEAR( host[i].hessian( 0, 0 ), expected[i].gap_curvature, tolerance );
    EXPECT_EQ( host[i].hessian( 0, 1 ), 0.0 );
    EXPECT_EQ( host[i].hessian( 1, 0 ), 0.0 );
    EXPECT_EQ( host[i].hessian( 1, 1 ), 0.0 );
  }
}

INSTANTIATE_TEST_SUITE_P( EnabledBackends, NodalIntegralTest, detail::enabledExecutionModes() );

#ifdef TRIBOL_USE_ENZYME
// Repeated references must share one gap variable; scalar scaling must apply to energy and both derivative orders.
TEST( NodalIntegralDerivatives, penalty_partials_match_analytical_values_and_finite_differences )
{
  const auto g = min( nodal_weighted_gap, 0.0 );
  const auto expression = 0.5 * penalty * sum( g, g, 1.0 / nodal_area );
  static_assert( decltype( expression )::dependsOn( 1 ) );
  static_assert( decltype( expression )::has_outer_hessian );
  auto definition = expression.atNode( 0, area_tolerance );
  MdArray<RealT, nodal_integral_count> state{ -0.2, 0.5 };
  const auto evaluated = EnergyPairEvaluator<decltype( definition )>::evaluate( state.view(), definition );
  EXPECT_NEAR( evaluated.energy, 0.16, tolerance );
  const MdArray<RealT, nodal_integral_count> expected_gradient{ -1.6, -0.32 };
  const MdArray<RealT, nodal_integral_count, nodal_integral_count> expected_hessian{
      MdArray<RealT, nodal_integral_count>{ 8.0, 3.2 }, MdArray<RealT, nodal_integral_count>{ 3.2, 1.28 } };
  constexpr RealT step = 1.0e-6;                  ///< Central difference step away from activation/support transitions.
  constexpr RealT difference_tolerance = 1.0e-8;  ///< Difference truncation and roundoff allowance.
  for ( IndexT column = 0; column < nodal_integral_count; ++column ) {
    EXPECT_NEAR( evaluated.residual[column], expected_gradient[column], tolerance );
    auto plus_state = state, minus_state = state;
    plus_state[column] += step;
    minus_state[column] -= step;
    const auto plus = EnergyPairEvaluator<decltype( definition )>::evaluate( plus_state.view(), definition );
    const auto minus = EnergyPairEvaluator<decltype( definition )>::evaluate( minus_state.view(), definition );
    EXPECT_NEAR( evaluated.residual[column], ( plus.energy - minus.energy ) / ( 2.0 * step ), difference_tolerance );
    for ( IndexT row = 0; row < nodal_integral_count; ++row ) {
      EXPECT_NEAR( evaluated.jacobian( row, column ), expected_hessian( row, column ), tolerance );
      EXPECT_NEAR( evaluated.jacobian( row, column ), ( plus.residual[row] - minus.residual[row] ) / ( 2.0 * step ),
                   difference_tolerance );
    }
  }
}

// Zero gap retains the compressive tangent; open or unsupported nodes have zero energy and all-zero derivatives.
TEST( NodalIntegralDerivatives, preserves_zero_gap_tangent_and_zeros_unsupported_nodes )
{
  const auto g = min( nodal_weighted_gap, 0.0 );
  auto definition = ( 0.5 * penalty * sum( g, g, 1.0 / nodal_area ) ).atNode( 0, area_tolerance );
  const MdArray<RealT, nodal_integral_count> zero_gap{ 0.0, 0.5 };
  const auto zero = EnergyPairEvaluator<decltype( definition )>::evaluate( zero_gap.view(), definition );
  EXPECT_EQ( zero.energy, 0.0 );
  EXPECT_EQ( zero.residual[0], 0.0 );
  EXPECT_EQ( zero.residual[1], 0.0 );
  EXPECT_NEAR( zero.jacobian( 0, 0 ), 8.0, tolerance );
  EXPECT_EQ( zero.jacobian( 0, 1 ), 0.0 );
  EXPECT_EQ( zero.jacobian( 1, 0 ), 0.0 );
  EXPECT_EQ( zero.jacobian( 1, 1 ), 0.0 );
  for ( const auto& state : std::array<MdArray<RealT, nodal_integral_count>, 4>{
            MdArray<RealT, nodal_integral_count>{ 0.2, 0.5 }, MdArray<RealT, nodal_integral_count>{ -0.2, 0.0 },
            MdArray<RealT, nodal_integral_count>{ 0.0, 0.0 },
            MdArray<RealT, nodal_integral_count>{ -0.2, 0.5 * area_tolerance } } ) {
    const auto evaluated = EnergyPairEvaluator<decltype( definition )>::evaluate( state.view(), definition );
    EXPECT_EQ( evaluated.energy, 0.0 );
    for ( IndexT i = 0; i < evaluated.residual.size(); ++i ) EXPECT_EQ( evaluated.residual[i], 0.0 );
    for ( IndexT i = 0; i < evaluated.jacobian.size(); ++i ) EXPECT_EQ( evaluated.jacobian[i], 0.0 );
  }
}

// Independent coefficients are read for each evaluation, while coordinate derivatives hold them fixed.
TEST( NodalIntegralDerivatives, reads_current_multiplier_values_without_differentiating_them )
{
  RealT coefficients[]{ -3.0, 2.0 };
  const IndependentNodalField pressure{ { coefficients, 2 } };
  const auto expression = dot( pressure, nodal_weighted_gap );
  auto definition = expression.atNode( 1, area_tolerance );
  const MdArray<RealT, 1> state{ -0.2 };
  auto evaluated = EnergyPairEvaluator<decltype( definition )>::evaluate( state.view(), definition );
  EXPECT_NEAR( evaluated.energy, -0.4, tolerance );
  EXPECT_EQ( evaluated.residual[0], 2.0 );
  for ( IndexT i = 0; i < evaluated.jacobian.size(); ++i ) EXPECT_EQ( evaluated.jacobian[i], 0.0 );
  coefficients[1] = 4.0;
  evaluated = EnergyPairEvaluator<decltype( definition )>::evaluate( state.view(), definition );
  EXPECT_NEAR( evaluated.energy, -0.8, tolerance );
  EXPECT_EQ( evaluated.residual[0], 4.0 );
}
#endif
}  // namespace

/** @brief Run portable expression tests and host derivatives. @param argc Argument count. @param argv Arguments.
 * @return Test status. */
int main( int argc, char** argv )
{
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
