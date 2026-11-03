// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check operator-based scalar AD on every enabled execution backend without MFEM or Enzyme.
 * Analytical derivatives check products, reciprocals, scalar constants, unary signs, and mixed Hessian entries.
 * Additional tests check self-aliasing compound assignments, constant derivatives, and branch choices at equality.
 */
#include "ExecutionTest.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/common/SecondOrderScalar.hpp"

namespace {
using namespace tribol;
using Scalar = SecondOrderScalar<2>;  ///< Two independent local variables, x and y.
constexpr RealT tolerance = 1.0e-12;  ///< Roundoff allowance for binary-fraction analytical results.
/** @brief Provide execution-space memory for portable AD arithmetic checks. */
class SecondOrderScalarTest : public detail::ExecutionTest {};

// Derivatives of x²/y + 3x - 2y + 1/x check quotient curvature and cross derivatives for both signs of x.
TEST_P( SecondOrderScalarTest, differentiates_arithmetic_with_constants )
{
  struct Case {
    RealT x;      ///< First variable, nonzero for reciprocal evaluation.
    RealT y;      ///< Second variable, nonzero denominator.
    RealT value;  ///< Expected function value.
    RealT dx;     ///< Expected x derivative.
    RealT dy;     ///< Expected y derivative.
    RealT dxx;    ///< Expected x curvature.
    RealT dxy;    ///< Expected mixed curvature in either order.
    RealT dyy;    ///< Expected y curvature.
  };
  const auto cases = copyToExecutionSpace<Case>(
      { { 2.0, 4.0, -0.5, 3.75, -2.25, 0.75, -0.25, 0.125 }, { -2.0, 4.0, -13.5, 1.75, -2.25, 0.25, 0.25, 0.125 } } );
  ArrayT<Scalar> results( cases.size(), cases.size(), allocatorId() );
  const auto input = cases.view();
  const auto output = results.view();
  forAllExec( GetParam(), cases.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
    const auto x = Scalar::variable( input[i].x, 0 );
    const auto y = Scalar::variable( input[i].y, 1 );
    output[i] = x * x / y + 2.0 * x - ( -x ) - y * 2.0 + 1.0 / ( +x );
  } );
  const ArrayT<Case, 1, MemorySpace::Host> expected( cases );
  const ArrayT<Scalar, 1, MemorySpace::Host> host( results );
  for ( IndexT i = 0; i < host.size(); ++i ) {
    SCOPED_TRACE( i );
    EXPECT_NEAR( host[i].value, expected[i].value, tolerance );
    EXPECT_NEAR( host[i].gradient[0], expected[i].dx, tolerance );
    EXPECT_NEAR( host[i].gradient[1], expected[i].dy, tolerance );
    EXPECT_NEAR( host[i].hessian( 0, 0 ), expected[i].dxx, tolerance );
    EXPECT_NEAR( host[i].hessian( 0, 1 ), expected[i].dxy, tolerance );
    EXPECT_NEAR( host[i].hessian( 1, 0 ), expected[i].dxy, tolerance );
    EXPECT_NEAR( host[i].hessian( 1, 1 ), expected[i].dyy, tolerance );
  }
}

// Squaring a scalar in place must use the original value and derivatives for both operands.
TEST_P( SecondOrderScalarTest, preserves_derivatives_during_compound_assignment )
{
  ArrayT<Scalar> results( 1, 1, allocatorId() );
  const auto output = results.view();
  forAllExec( GetParam(), results.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
    auto value = Scalar::variable( 3.0, 0 );
    const auto y = Scalar::variable( 4.0, 1 );
    value += y;
    value -= 2.0;
    value *= value;
    value /= y;
    output[i] = value;
  } );
  const ArrayT<Scalar, 1, MemorySpace::Host> host( results );
  // (x + y - 2)² / y at x=3, y=4.
  EXPECT_NEAR( host[0].value, 6.25, tolerance );
  EXPECT_NEAR( host[0].gradient[0], 2.5, tolerance );
  EXPECT_NEAR( host[0].gradient[1], 0.9375, tolerance );
  EXPECT_NEAR( host[0].hessian( 0, 0 ), 0.5, tolerance );
  EXPECT_NEAR( host[0].hessian( 0, 1 ), -0.125, tolerance );
  EXPECT_NEAR( host[0].hessian( 1, 0 ), -0.125, tolerance );
  EXPECT_NEAR( host[0].hessian( 1, 1 ), 0.03125, tolerance );
}

// Equal values can have different derivatives. Min/max retain the first operand's derivatives at equality.
TEST_P( SecondOrderScalarTest, retains_selected_branch_derivatives_and_constant_zeros )
{
  constexpr IndexT result_count = 5;
  ArrayT<Scalar> results( result_count, result_count, allocatorId() );
  const auto output = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const auto x = Scalar::variable( 0.0, 0 );
    const auto y = Scalar::variable( 0.0, 1 );
    output[0] = min( x, y );
    output[1] = max( y, x );
    output[2] = min( x + 1.0, 0.0 );
    output[3] = 4.0;
    // Comparisons inspect values, even when the two gradients differ.
    output[4] = ( x == y && !( x != y ) && x <= y && x >= y && !( x < y ) && !( x > y ) ) ? 1.0 : 0.0;
  } );
  const ArrayT<Scalar, 1, MemorySpace::Host> host( results );
  EXPECT_EQ( host[0].gradient[0], 1.0 );
  EXPECT_EQ( host[0].gradient[1], 0.0 );
  EXPECT_EQ( host[1].gradient[0], 0.0 );
  EXPECT_EQ( host[1].gradient[1], 1.0 );
  EXPECT_EQ( host[2].value, 0.0 );
  EXPECT_EQ( host[3].value, 4.0 );
  EXPECT_EQ( host[4].value, 1.0 );
  for ( IndexT i = 2; i < result_count; ++i )
    for ( IndexT j = 0; j < Scalar::variable_count; ++j ) EXPECT_EQ( host[i].gradient[j], 0.0 );
  for ( const auto& result : host )
    for ( IndexT i = 0; i < result.hessian.size(); ++i ) EXPECT_EQ( result.hessian[i], 0.0 );
}

INSTANTIATE_TEST_SUITE_P( EnabledBackends, SecondOrderScalarTest, detail::enabledExecutionModes() );
}  // namespace

/** @brief Run portable AD arithmetic tests. @param argc Argument count. @param argv Arguments. @return Test status. */
int main( int argc, char** argv )
{
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
