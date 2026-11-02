// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check basis weighting and signed pair residual layout independently of contact geometry and quadrature.
 * Tests use unequal node counts and distinct traction components in 2D and 3D, check array and view inputs, and verify
 * zero traction. Every enabled backend runs the same calculations. Common Plane tests cover activation, force/moment
 * balance, quadrature, and Enzyme Jacobians through this helper.
 */
#include "ExecutionTest.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/integ/PairTraction.hpp"

namespace {
using namespace tribol;
constexpr RealT tolerance = 1.0e-12;  ///< Roundoff allowance for prescribed decimal basis weights.

/** @brief Run local traction calculations in memory accessible to each enabled backend. */
class PairTractionTest : public detail::ExecutionTest {
 protected:
  /**
   * @brief Run a calculation on the selected backend and copy its owning result to the host.
   * @tparam Calculation Callable returning a fixed-size residual array.
   * @param calculation Operation with value-owned inputs accessible to the selected backend.
   * @return Owned host residual, valid after execution storage has been released.
   */
  template <typename Calculation>
  auto evaluate( Calculation calculation ) const
  {
    using Result = decltype( calculation() );
    ArrayT<Result> result( 1, 1, allocatorId() );
    const auto output = result.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) { output[0] = calculation(); } );
    const ArrayT<Result, 1, MemorySpace::Host> host( result );
    return host[0];
  }
};

// Different basis sizes expose an incorrect second-block offset; distinct components expose swapped node/component
// order.
TEST_P( PairTractionTest, weights_two_dimensional_traction_and_preserves_zero_values )
{
  const MdArray<RealT, 2> basis1{ 0.25, 0.75 };
  const MdArray<RealT, 3> basis2{ -0.125, 0.375, 0.75 };
  const axom::primal::Vector<RealT, 2> traction{ 2.0, -4.0 };
  const auto residual = evaluate( [=] TRIBOL_HOST_DEVICE() { return pairTraction( basis1, basis2, traction ); } );
  const MdArray<RealT, basis1.size() + basis2.size(), 2> expected{
      MdArray<RealT, 2>{ 0.5, -1.0 }, MdArray<RealT, 2>{ 1.5, -3.0 }, MdArray<RealT, 2>{ 0.25, -0.5 },
      MdArray<RealT, 2>{ -0.75, 1.5 }, MdArray<RealT, 2>{ -1.5, 3.0 } };
  ASSERT_EQ( residual.size(), expected.size() );
  for ( IndexT dof = 0; dof < expected.size(); ++dof ) EXPECT_NEAR( residual[dof], expected[dof], tolerance );
  const auto zero =
      evaluate( [=] TRIBOL_HOST_DEVICE() { return pairTraction( basis1, basis2, axom::primal::Vector<RealT, 2>{} ); } );
  for ( IndexT dof = 0; dof < zero.size(); ++dof ) EXPECT_EQ( zero[dof], 0.0 );
}

// Triangle and quadrilateral basis views must produce separate 3D blocks with opposite signs and all three components.
TEST_P( PairTractionTest, weights_three_dimensional_traction_from_basis_views )
{
  const MdArray<RealT, 3> basis1{ 0.2, 0.3, 0.5 };
  const MdArray<RealT, 4> basis2{ 0.1, 0.3, 0.45, 0.15 };
  const axom::primal::Vector<RealT, 3> traction{ 2.0, -4.0, 6.0 };
  const auto residual =
      evaluate( [=] TRIBOL_HOST_DEVICE() { return pairTraction( basis1.view(), basis2.view(), traction ); } );
  const MdArray<RealT, basis1.size() + basis2.size(), 3> expected{
      MdArray<RealT, 3>{ 0.4, -0.8, 1.2 },  MdArray<RealT, 3>{ 0.6, -1.2, 1.8 },  MdArray<RealT, 3>{ 1.0, -2.0, 3.0 },
      MdArray<RealT, 3>{ -0.2, 0.4, -0.6 }, MdArray<RealT, 3>{ -0.6, 1.2, -1.8 }, MdArray<RealT, 3>{ -0.9, 1.8, -2.7 },
      MdArray<RealT, 3>{ -0.3, 0.6, -0.9 } };
  ASSERT_EQ( residual.size(), expected.size() );
  for ( IndexT dof = 0; dof < expected.size(); ++dof ) EXPECT_NEAR( residual[dof], expected[dof], tolerance );
}

INSTANTIATE_TEST_SUITE_P( EnabledBackends, PairTractionTest, detail::enabledExecutionModes() );
}  // namespace

/** @brief Run portable pair traction tests. @param argc Argument count. @param argv Arguments. @return Test status. */
int main( int argc, char** argv )
{
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
