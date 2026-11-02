// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Test common-line quadrature and direct penalty residuals on every enabled backend. Host Enzyme checks compare
 * the full pair Jacobian with finite differences; MFEM assembly and selection have separate integration tests.
 */
#include <cmath>
#include <stdexcept>
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "ExecutionTest.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/physics/CommonPlane.hpp"

namespace {
using namespace tribol;
constexpr RealT tolerance = 1.0e-12;  ///< Roundoff allowance for geometry and exact polynomial integrals.
using Coordinates = MdArray<RealT, LinearEdgeGeometry::vertex_count, LinearEdgeGeometry::spatial_dimension>;

/** @brief Execute local geometry and residual calculations using backend-accessible storage. */
class CommonPlaneTest : public tribol::detail::ExecutionTest {
 protected:
  /** @brief Check exact parallel-edge loads and activation. @tparam Count Gauss point count. */
  template <int Count>
  void checkParallelResiduals()
  {
    struct Case {
      RealT separation;        ///< Height of the second edge; positive means penetration for these orientations.
      RealT residual_gap;      ///< Separation offset subtracted before activation.
      RealT expected_first_y;  ///< Expected residual at either first-edge node.
      bool active;             ///< Whether the nonpositive gap branch is taken.
    };
    // Unequal mesh springs combine to stiffness 1.5, rather than their arithmetic average 4.0.
    const Case cases[] = {
        { 0.1, 0.0, -0.15, true }, { -0.1, 0.0, 0.0, false }, { 0.0, 0.0, 0.0, true }, { -0.05, 0.1, -0.075, true } };
    for ( const auto test : cases ) {
      SCOPED_TRACE( test.separation );
      ArrayT<CommonPlaneResidualData> results( 1, 1, allocatorId() );
      const auto output = results.view();
      forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
        const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.0 } };
        const Coordinates edge2{ MdArray<RealT, 2>{ 2.0, test.separation }, MdArray<RealT, 2>{ 0.0, test.separation } };
        const CommonPlaneQuadratureRule<EdgeQuadratureRuleGenerator<Count>> rule;
        output[0] =
            computeCommonPlaneResidual( edge1.view(), edge2.view(), rule, ComputePenaltyStiffnessPerArea( 2.0, 6.0 ),
                                        OffsetGap{ CommonPlaneGap{}, test.residual_gap } );
      } );
      const ArrayT<CommonPlaneResidualData, 1, MemorySpace::Host> host( results );
      ASSERT_EQ( host[0].quadrature_result.status, QuadratureStatus::Success );
      EXPECT_EQ( host[0].quadrature_result.required_point_count, Count );
      EXPECT_EQ( host[0].has_active_points, test.active );
      for ( int node = 0; node < BasisFunction<EDGE, LINEAR_LAGRANGE>::node_count; ++node ) {
        EXPECT_NEAR( host[0].residual[2 * node], 0.0, tolerance );
        EXPECT_NEAR( host[0].residual[2 * node + 1], test.expected_first_y, tolerance );
        EXPECT_NEAR( host[0].residual[2 * ( node + 2 )], 0.0, tolerance );
        EXPECT_NEAR( host[0].residual[2 * ( node + 2 ) + 1], -test.expected_first_y, tolerance );
      }
    }
  }

  /** @brief Check partial overlap, reverse mapping, and physical line measure. @tparam Count Gauss point count. */
  template <int Count>
  void checkQuadrature()
  {
    enum Column
    {
      Parent1,
      Parent2,
      Weight,
      Point1X,
      Point2X,
      ColumnCount
    };
    ArrayT<RealT, 2> records( axom::StackArray<axom::IndexType, 2>{ Count, ColumnCount }, allocatorId() );
    const auto output = records.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
      const Coordinates edge2{ MdArray<RealT, 2>{ 3.0, 0.2 }, MdArray<RealT, 2>{ 1.0, 0.2 } };
      const LinearEdgeGeometry geometry1( edge1.view() ), geometry2( edge2.view() );
      const CommonPlaneQuadratureRule<EdgeQuadratureRuleGenerator<Count>> rule;
      const auto generated = rule.generate( geometry1, geometry2 );
      const auto points = generated.getView();
      for ( int q = 0; q < Count; ++q ) {
        output( q, Parent1 ) = points.geometry1_coords( q, 0 );
        output( q, Parent2 ) = points.geometry2_coords( q, 0 );
        output( q, Weight ) = points.weights[q];
        output( q, Point1X ) = geometry1.position( output( q, Parent1 ) )[0];
        output( q, Point2X ) = geometry2.position( output( q, Parent2 ) )[0];
      }
    } );
    const ArrayT<RealT, 2, MemorySpace::Host> values( records );
    RealT measure = 0.0, second_moment = 0.0;
    for ( int q = 0; q < Count; ++q ) {
      EXPECT_GE( values( q, Parent1 ), -0.5 );
      EXPECT_LE( values( q, Parent1 ), 0.5 );
      EXPECT_NEAR( values( q, Point1X ), values( q, Point2X ), tolerance );
      EXPECT_NEAR( values( q, Parent2 ), 2.0 - values( q, Point1X ), tolerance );
      EXPECT_GT( values( q, Weight ), 0.0 );
      measure += values( q, Weight );
      second_moment += values( q, Weight ) * values( q, Point1X ) * values( q, Point1X );
    }
    EXPECT_NEAR( measure, 2.0, tolerance );
    if constexpr ( Count >= 2 ) EXPECT_NEAR( second_moment, 26.0 / 3.0, tolerance );
  }
};

// Each supported rule must integrate the physical overlap without multiplying by either edge Jacobian again.
TEST_P( CommonPlaneTest, maps_partial_overlap_and_physical_weights )
{
  checkQuadrature<1>();
  checkQuadrature<2>();
  checkQuadrature<3>();
}

// Penetration, opening, residual-gap shifts, and zero-gap activation use the same law for every supported rule.
TEST_P( CommonPlaneTest, integrates_parallel_edge_residuals )
{
  checkParallelResiduals<1>();
  checkParallelResiduals<2>();
  checkParallelResiduals<3>();
}

// A rotated 3-4-5 edge retains the physical length and balances both total force and moment.
TEST_P( CommonPlaneTest, balances_rotated_pair_forces_and_moments )
{
  ArrayT<RealT> records( CommonPlaneResidualData::coordinate_count, CommonPlaneResidualData::coordinate_count,
                         allocatorId() );
  const auto output = records.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const Coordinates edge1{ MdArray<RealT, 2>{ 1.0, 2.0 }, MdArray<RealT, 2>{ 4.0, 6.0 } };
    const Coordinates edge2{ MdArray<RealT, 2>{ 3.8, 6.15 }, MdArray<RealT, 2>{ 0.8, 2.15 } };
    const CommonPlaneQuadratureRule<EdgeQuadratureRuleGenerator<3>> rule;
    const auto result = computeCommonPlaneResidual( edge1.view(), edge2.view(), rule, 2.0 );
    for ( int i = 0; i < CommonPlaneResidualData::coordinate_count; ++i ) output[i] = result.residual[i];
  } );
  const ArrayT<RealT, 1, MemorySpace::Host> values( records );
  const RealT x[] = { 1.0, 4.0, 3.8, 0.8 }, y[] = { 2.0, 6.0, 6.15, 2.15 };
  RealT force_x = 0.0, force_y = 0.0, moment = 0.0;
  for ( int node = 0; node < 4; ++node ) {
    force_x += values[2 * node];
    force_y += values[2 * node + 1];
    moment += x[node] * values[2 * node + 1] - y[node] * values[2 * node];
  }
  EXPECT_NEAR( values[0], 1.0, tolerance );
  EXPECT_NEAR( values[1], -0.75, tolerance );
  EXPECT_NEAR( force_x, 0.0, tolerance );
  EXPECT_NEAR( force_y, 0.0, tolerance );
  EXPECT_NEAR( moment, 0.0, tolerance );
}

// Failed geometry is distinct from valid empty overlap, and insufficient capacity must leave all storage untouched.
TEST_P( CommonPlaneTest, reports_geometry_and_capacity_without_writing )
{
  enum Result
  {
    InvalidFirst,
    InvalidSecond,
    Disjoint,
    Cooriented,
    Perpendicular,
    Overflow,
    ResultCount
  };
  constexpr RealT unwritten = -7.0;
  ArrayT<QuadratureResult> records( ResultCount, ResultCount, allocatorId() );
  ArrayT<RealT> storage( 4, 4, allocatorId() );
  const auto output = records.view();
  const auto values = storage.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.0 } };
    const Coordinates edge2{ MdArray<RealT, 2>{ 2.0, 0.1 }, MdArray<RealT, 2>{ 0.0, 0.1 } };
    const Coordinates invalid{};
    const Coordinates disjoint{ MdArray<RealT, 2>{ 5.0, 0.1 }, MdArray<RealT, 2>{ 3.0, 0.1 } };
    const Coordinates cooriented{ MdArray<RealT, 2>{ 0.0, 0.1 }, MdArray<RealT, 2>{ 2.0, 0.1 } };
    const Coordinates perpendicular{ MdArray<RealT, 2>{ 1.0, -1.0 }, MdArray<RealT, 2>{ 1.0, 1.0 } };
    const CommonPlaneQuadratureRule<EdgeQuadratureRuleGenerator<3>> rule;
    output[InvalidFirst] = rule.generate( invalid.view(), edge2.view() ).result;
    output[InvalidSecond] = rule.generate( edge1.view(), invalid.view() ).result;
    output[Disjoint] = rule.generate( edge1.view(), disjoint.view() ).result;
    output[Cooriented] = rule.generate( edge1.view(), cooriented.view() ).result;
    output[Perpendicular] = rule.generate( edge1.view(), perpendicular.view() ).result;
    for ( int i = 0; i < values.size(); ++i ) values[i] = unwritten;
    const PairQuadratureView<1, 1> bounded{
        { values.data() + 1, { 2, 1 } }, { values.data() + 1, { 2, 1 } }, { values.data() + 1, 2 } };
    output[Overflow] = rule.generate( edge1.view(), edge2.view(), bounded );
  } );
  const ArrayT<QuadratureResult, 1, MemorySpace::Host> results( records );
  const ArrayT<RealT, 1, MemorySpace::Host> host( storage );
  for ( int invalid : { InvalidFirst, InvalidSecond } ) {
    EXPECT_EQ( results[invalid].status, QuadratureStatus::InvalidGeometry );
    EXPECT_EQ( results[invalid].required_point_count, 0 );
  }
  for ( int empty : { Disjoint, Cooriented, Perpendicular } ) {
    EXPECT_EQ( results[empty].status, QuadratureStatus::Success );
    EXPECT_EQ( results[empty].required_point_count, 0 );
  }
  EXPECT_EQ( results[Overflow].status, QuadratureStatus::InsufficientCapacity );
  EXPECT_EQ( results[Overflow].required_point_count, 3 );
  for ( const auto value : host ) EXPECT_EQ( value, unwritten );
}

// Crossing oblique edges have both open and penetrating points; activation is decided at each point.
TEST_P( CommonPlaneTest, activates_only_penetrating_quadrature_points )
{
  ArrayT<RealT> records( 3, 3, allocatorId() );
  const auto output = records.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const Coordinates edge1{ MdArray<RealT, 2>{ -1.0, 0.0 }, MdArray<RealT, 2>{ 1.0, 0.0 } };
    const Coordinates edge2{ MdArray<RealT, 2>{ 0.8, 0.2 }, MdArray<RealT, 2>{ -0.8, -0.2 } };
    const LinearEdgeGeometry geometry1( edge1.view() ), geometry2( edge2.view() );
    const CommonPlaneQuadratureRule<EdgeQuadratureRuleGenerator<2>> rule;
    const auto quadrature = rule.generate( geometry1, geometry2 );
    const auto frame = CommonPlaneEdgeMapping::makeFrame( geometry1, geometry2 );
    for ( int q = 0; q < 2; ++q ) {
      output[q] = LinearEdgeGeometry::Vector( geometry2.position( quadrature.geometry2_coords( q, 0 ) ),
                                              geometry1.position( quadrature.geometry1_coords( q, 0 ) ) )
                      .dot( frame.normal );
    }
    const auto result = computeCommonPlaneResidual( edge1.view(), edge2.view(), rule, 1.0 );
    output[2] = 0.0;
    for ( int node = 0; node < 2; ++node ) output[2] += result.residual[2 * node + 1];
  } );
  const ArrayT<RealT, 1, MemorySpace::Host> values( records );
  EXPECT_LT( values[0] * values[1], 0.0 );
  EXPECT_LT( values[2], 0.0 );
}
INSTANTIATE_TEST_SUITE_P( execution, CommonPlaneTest, tribol::detail::enabledExecutionModes() );

#ifdef TRIBOL_USE_ENZYME
/** @brief Compare active-geometry Jacobians with central differences. @tparam Count Gauss point count. */
template <int Count>
void checkJacobian()
{
  const CommonPlaneQuadratureRule<EdgeQuadratureRuleGenerator<Count>> rule;
  constexpr int dofs = CommonPlaneResidualData::coordinate_count;
  constexpr RealT step =
      2.0e-7;  // Small enough to resolve derivatives without changing the active or overlap branches.
  const Coordinates cases1[] = { Coordinates{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.2 } },
                                 Coordinates{ MdArray<RealT, 2>{ -1.0, 0.0 }, MdArray<RealT, 2>{ 1.0, 0.0 } } };
  const Coordinates cases2[] = { Coordinates{ MdArray<RealT, 2>{ 1.8, 0.35 }, MdArray<RealT, 2>{ 0.2, 0.1 } },
                                 Coordinates{ MdArray<RealT, 2>{ 0.8, 0.2 }, MdArray<RealT, 2>{ -0.8, -0.2 } } };
  for ( int geometry = 0; geometry < 2; ++geometry ) {
    const auto& edge1 = cases1[geometry];
    const auto& edge2 = cases2[geometry];
    const auto result =
        computeCommonPlanePair( edge1.view(), edge2.view(), rule, 2.5, OffsetGap{ CommonPlaneGap{}, 0.03 } );
    for ( int column = 0; column < dofs; ++column ) {
      auto plus1 = edge1, minus1 = edge1, plus2 = edge2, minus2 = edge2;
      if ( column < Coordinates::size() ) {
        plus1[column] += step;
        minus1[column] -= step;
      } else {
        plus2[column - Coordinates::size()] += step;
        minus2[column - Coordinates::size()] -= step;
      }
      const auto plus =
          computeCommonPlaneResidual( plus1.view(), plus2.view(), rule, 2.5, OffsetGap{ CommonPlaneGap{}, 0.03 } );
      const auto minus =
          computeCommonPlaneResidual( minus1.view(), minus2.view(), rule, 2.5, OffsetGap{ CommonPlaneGap{}, 0.03 } );
      for ( int row = 0; row < dofs; ++row ) {
        EXPECT_NEAR( result.jacobian( row, column ), ( plus.residual[row] - minus.residual[row] ) / ( 2.0 * step ),
                     2.0e-7 )
            << "points=" << Count << ", geometry=" << geometry << ", row=" << row << ", column=" << column;
      }
    }
  }
  const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.0 } };
  const Coordinates edge2{ MdArray<RealT, 2>{ 2.0, 0.0 }, MdArray<RealT, 2>{ 0.0, 0.0 } };
  const auto zero = computeCommonPlanePair( edge1.view(), edge2.view(), rule, 1.0 );
  EXPECT_TRUE( zero.has_active_points );
  EXPECT_GT( zero.jacobian( 1, 1 ), 0.0 );
  const Coordinates invalid{};
  EXPECT_THROW( computeCommonPlanePair( invalid.view(), edge2.view(), rule, 1.0 ), std::invalid_argument );
}

// All reference counts differentiate current normals, interval bounds, edge lengths, and mixed active-point residuals.
TEST( CommonPlaneJacobian, matches_finite_differences )
{
  checkJacobian<1>();
  checkJacobian<2>();
  checkJacobian<3>();
}
#endif
}  // namespace

/** @brief Initialize logging and run local common-plane tests. @param argc Argument count. @param argv Arguments.
 * @return Test status. */
int main( int argc, char** argv )
{
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
