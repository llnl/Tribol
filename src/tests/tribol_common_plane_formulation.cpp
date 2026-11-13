// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Test common-line quadrature and direct penalty residuals on every enabled backend. Host Enzyme checks compare
 * the full pair Jacobian with finite differences; MFEM assembly and selection have separate integration tests.
 * Crossing edges check clipped interval bounds, point locations, and uniform centroid pressure for every base rule.
 * Value-only pair kernels check activation, quadrature failures, allocator selection, and empty input on every enabled
 * execution backend. Values from an earlier call remain unchanged after another evaluation. Returned face-coordinate
 * arrays follow reordered connectivity and remain valid when mesh coordinates change; geometry copies their values.
 */
#include <cmath>
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "ExecutionTest.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/physics/CommonPlane.hpp"
#include "tribol/physics/CommonPlaneFormulation.hpp"
#include "tribol/assembly/PairKernel.hpp"

namespace {
using namespace tribol;
constexpr RealT tolerance = 1.0e-12;  ///< Roundoff allowance for geometry and exact polynomial integrals.
using Coordinates = MdArray<RealT, LinearEdgeGeometry::num_vertices, LinearEdgeGeometry::spatial_dim>;
using CommonPlaneResidualValue = CommonPlaneResidualPairFunction<EdgeQuadratureRuleGenerator<1>>::Result;

// Common Plane exposes the same values and quadrature state required by the shared pair derivative calculations.
static_assert( PairFunctionType<CommonPlaneResidualPairFunction<EdgeQuadratureRuleGenerator<1>>> );

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Copy two edge-coordinate arrays into Jacobian column order for finite differences.
 * @param coordinates1 First edge's vertex coordinates.
 * @param coordinates2 Second edge's vertex coordinates.
 * @return Owned coordinates ordered by edge, vertex, and component.
 */
TRIBOL_HOST_DEVICE MdArray<RealT, 2 * Coordinates::size()> packPairCoordinates(
    LinearEdgeGeometry::Coords coordinates1, LinearEdgeGeometry::Coords coordinates2 )
{
  MdArray<RealT, 2 * Coordinates::size()> coordinates{};
  for ( int coordinate = 0; coordinate < Coordinates::size(); ++coordinate ) {
    coordinates[coordinate] = coordinates1[coordinate];
    coordinates[Coordinates::size() + coordinate] = coordinates2[coordinate];
  }
  return coordinates;
}

#endif

/** @brief Execute local geometry and residual calculations using backend-accessible storage. */
class CommonPlaneTest : public tribol::detail::ExecutionTest {
 protected:
  /** @brief Check exact parallel-edge loads and activation. @tparam NumPoints Gauss point count. */
  template <int NumPoints>
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
      ArrayT<CommonPlaneResidualValue> results( 1, 1, allocatorId() );
      const auto output = results.view();
      forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
        const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.0 } };
        const Coordinates edge2{ MdArray<RealT, 2>{ 2.0, test.separation }, MdArray<RealT, 2>{ 0.0, test.separation } };
        const PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>
            rule_generator;
        const CommonPlaneResidualPairFunction<EdgeQuadratureRuleGenerator<NumPoints>, OffsetGap<CommonPlaneGap>>
            function{ rule_generator, ComputePenaltyStiffnessPerArea( 2.0, 6.0 ),
                      OffsetGap{ CommonPlaneGap{}, test.residual_gap } };
        output[0] = function.evaluate( edge1.view(), edge2.view() );
      } );
      const ArrayT<CommonPlaneResidualValue, 1, MemorySpace::Host> host( results );
      ASSERT_EQ( host[0].quadrature_result.status, QuadratureStatus::Success );
      EXPECT_EQ( host[0].quadrature_result.num_required_points, NumPoints );
      EXPECT_EQ( host[0].has_contributing_points, test.active );
      for ( int node = 0; node < BasisFunction<EDGE, LINEAR_LAGRANGE>::num_nodes; ++node ) {
        EXPECT_NEAR( host[0].geometry1( node, 0 ), 0.0, tolerance );
        EXPECT_NEAR( host[0].geometry1( node, 1 ), test.expected_first_y, tolerance );
        EXPECT_NEAR( host[0].geometry2( node, 0 ), 0.0, tolerance );
        EXPECT_NEAR( host[0].geometry2( node, 1 ), -test.expected_first_y, tolerance );
      }
    }
  }

  /** @brief Check partial overlap, reverse mapping, and physical line measure. @tparam NumPoints Gauss point count. */
  template <int NumPoints>
  void checkQuadrature()
  {
    enum Column
    {
      Parent1,
      Parent2,
      Weight,
      Point1X,
      Point2X,
      NumColumns
    };
    ArrayT<RealT, 2> records( axom::StackArray<axom::IndexType, 2>{ NumPoints, NumColumns }, allocatorId() );
    const auto output = records.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
      const Coordinates edge2{ MdArray<RealT, 2>{ 3.0, 0.2 }, MdArray<RealT, 2>{ 1.0, 0.2 } };
      const LinearEdgeGeometry geometry1( edge1.view() ), geometry2( edge2.view() );
      const PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>> rule_generator;
      const auto generated = rule_generator.generate( geometry1, geometry2 );
      const auto points = generated.getView();
      for ( int q = 0; q < NumPoints; ++q ) {
        output( q, Parent1 ) = points.geom1_parametric_coords( q, 0 );
        output( q, Parent2 ) = points.geom2_parametric_coords( q, 0 );
        output( q, Weight ) = points.weights[q];
        output( q, Point1X ) = geometry1.position( output( q, Parent1 ) )[0];
        output( q, Point2X ) = geometry2.position( output( q, Parent2 ) )[0];
      }
    } );
    const ArrayT<RealT, 2, MemorySpace::Host> values( records );
    RealT measure = 0.0, second_moment = 0.0;
    for ( int q = 0; q < NumPoints; ++q ) {
      EXPECT_GE( values( q, Parent1 ), -0.5 );
      EXPECT_LE( values( q, Parent1 ), 0.5 );
      EXPECT_NEAR( values( q, Point1X ), values( q, Point2X ), tolerance );
      EXPECT_NEAR( values( q, Parent2 ), 2.0 - values( q, Point1X ), tolerance );
      EXPECT_GT( values( q, Weight ), 0.0 );
      measure += values( q, Weight );
      second_moment += values( q, Weight ) * values( q, Point1X ) * values( q, Point1X );
    }
    EXPECT_NEAR( measure, 2.0, tolerance );
    if constexpr ( NumPoints >= 2 ) {
      EXPECT_NEAR( second_moment, 26.0 / 3.0, tolerance );
    }
  }

  /** @brief Check crossing-edge intervals and centroid-pressure loads. @tparam NumPoints Configured Gauss point count.
   */
  template <int NumPoints>
  void checkInterpenetratingOverlap()
  {
    struct Case {
      RealT slope;          ///< Slope of edge1; edge2 has the opposite slope.
      RealT separation;     ///< Height of edge2 at x=0.0.
      RealT residual_gap;   ///< Offset applied to pressure after clipping the geometric interval.
      RealT lower;          ///< Exact lower endpoint of the penetrating interval.
      RealT upper;          ///< Exact upper endpoint of the penetrating interval.
      RealT total_force_y;  ///< Exact edge1 load for unit penalty and uniform centroid pressure.
    };
    // The first two full-overlap midpoints are open. The third is closed, and the last applies a pressure offset.
    const Case cases[] = { { 0.25, -0.25, 0.0, -1.0, -0.5, -0.0625 },
                           { -0.25, -0.25, 0.0, 0.5, 1.0, -0.0625 },
                           { 0.25, 0.25, 0.0, -1.0, 0.5, -0.5625 },
                           { 0.25, -0.25, 0.1, -1.0, -0.5, -0.1125 } };
    struct Sample {
      EdgeQuadratureInterval interval;      ///< Calculated geometric bounds.
      CommonPlaneResidualValue residual;    ///< Pair loads and quadrature status.
      MdArray<RealT, NumPoints> x1{};       ///< Point positions on edge1.
      MdArray<RealT, NumPoints> x2{};       ///< Point positions on edge2.
      MdArray<RealT, NumPoints> gaps{};     ///< Unshifted point gaps.
      MdArray<RealT, NumPoints> weights{};  ///< Physical line weights.
    };
    for ( const auto test : cases ) {
      SCOPED_TRACE( testing::Message() << "points=" << NumPoints << ", slope=" << test.slope << ", separation="
                                       << test.separation << ", residual_gap=" << test.residual_gap );
      ArrayT<Sample> samples( 1, 1, allocatorId() );
      const auto output = samples.view();
      forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
        const Coordinates edge1{ MdArray<RealT, 2>{ -1.0, -test.slope }, MdArray<RealT, 2>{ 1.0, test.slope } };
        const Coordinates edge2{ MdArray<RealT, 2>{ 1.0, test.separation - test.slope },
                                 MdArray<RealT, 2>{ -1.0, test.separation + test.slope } };
        const LinearEdgeGeometry geometry1( edge1.view() ), geometry2( edge2.view() );
        const PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>
            rule_generator;
        const auto points = rule_generator.generate( geometry1, geometry2 );
        const auto line = CommonPlaneEdgeMapping::computeCommonPlaneLine( geometry1, geometry2 );
        output[0].interval = CommonPlaneEdgeMapping{}.makeIntegrationSurface( geometry1, geometry2 );
        const CommonPlaneResidualPairFunction<EdgeQuadratureRuleGenerator<NumPoints>, OffsetGap<CommonPlaneGap>>
            function{ rule_generator, 1.0, OffsetGap{ CommonPlaneGap{}, test.residual_gap } };
        output[0].residual = function.evaluate( edge1.view(), edge2.view() );
        for ( int point = 0; point < NumPoints; ++point ) {
          const auto point1 = geometry1.position( points.geom1_parametric_coords( point, 0 ) );
          const auto point2 = geometry2.position( points.geom2_parametric_coords( point, 0 ) );
          output[0].x1[point] = point1[0];
          output[0].x2[point] = point2[0];
          output[0].gaps[point] = CommonPlaneGap{}( point1, point2, line );
          output[0].weights[point] = points.weights[point];
        }
      } );
      const ArrayT<Sample, 1, MemorySpace::Host> host( samples );
      const auto& sample = host[0];
      ASSERT_TRUE( sample.interval.valid );
      EXPECT_NEAR( sample.interval.bounds[0], test.lower, tolerance );
      EXPECT_NEAR( sample.interval.bounds[1], test.upper, tolerance );
      ASSERT_EQ( sample.residual.quadrature_result.status, QuadratureStatus::Success );
      EXPECT_EQ( sample.residual.quadrature_result.num_required_points, NumPoints );
      EXPECT_TRUE( sample.residual.has_contributing_points );
      RealT measure = 0.0;
      RealT first_moment = 0.0;
      for ( int point = 0; point < NumPoints; ++point ) {
        EXPECT_GT( sample.x1[point], test.lower );
        EXPECT_LT( sample.x1[point], test.upper );
        EXPECT_NEAR( sample.x1[point], sample.x2[point], tolerance );
        EXPECT_LT( sample.gaps[point], 0.0 );
        EXPECT_GT( sample.weights[point], 0.0 );
        measure += sample.weights[point];
        first_moment += sample.weights[point] * sample.x1[point];
      }
      const RealT center = 0.5 * ( test.lower + test.upper );
      EXPECT_NEAR( measure, test.upper - test.lower, tolerance );
      EXPECT_NEAR( first_moment / measure, center, tolerance );
      const MdArray<RealT, 4> loads{
          0.5 * ( 1.0 - center ) * test.total_force_y, 0.5 * ( 1.0 + center ) * test.total_force_y,
          -0.5 * ( 1.0 + center ) * test.total_force_y, -0.5 * ( 1.0 - center ) * test.total_force_y };
      const auto residual_values = sample.residual.values();
      for ( int node = 0; node < loads.size(); ++node ) {
        EXPECT_NEAR( residual_values[2 * node], 0.0, tolerance );
        EXPECT_NEAR( residual_values[2 * node + 1], loads[node], tolerance );
      }
    }
  }
};

// Reordered connectivity distinguishes the two faces. Mesh and array updates must not change constructed geometry.
TEST_P( CommonPlaneTest, constructs_geometry_from_returned_face_coordinate_arrays )
{
  auto x = copyToExecutionSpace<RealT>( { 1.0, 3.0, -2.0, 0.0 } );
  auto y = copyToExecutionSpace<RealT>( { 1.0, 4.0, 1.0, 0.0 } );
  const auto connectivity = copyToExecutionSpace<IndexT>( { 3, 1, 2, 0 } );
  MeshData mesh( 0, 2, x.size(), connectivity.data(), EDGE, LINEAR_LAGRANGE, x.data(), y.data(), nullptr,
                 memorySpace() );
  const auto mesh_view = mesh.getView();
  const auto x_view = x.view();
  const auto y_view = y.view();
  ArrayT<MdArray<RealT, 4, 5>> results( 1, 1, allocatorId() );
  const auto output = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    auto coordinates = mesh_view.getFaceCoords<LinearEdgeGeometry::Coords>( 0 );
    const auto other_coordinates = mesh_view.getFaceCoords<LinearEdgeGeometry::Coords>( 1 );
    const LinearEdgeGeometry original( coordinates.view() ), other( other_coordinates.view() );
    // Translate only face zero; its returned array must retain the original coordinates.
    x_view[3] += 2.0;
    x_view[1] += 2.0;
    y_view[3] -= 1.0;
    y_view[1] -= 1.0;
    for ( IndexT i = 0; i < coordinates.size(); ++i ) {
      output[0]( 3, i ) = coordinates[i];
    }
    output[0]( 3, 4 ) = coordinates.size();
    const auto updated_coordinates = mesh_view.getFaceCoords<LinearEdgeGeometry::Coords>( 0 );
    const LinearEdgeGeometry updated( updated_coordinates.view() );
    // The geometry must retain its own values when its input array is modified as well.
    coordinates( 0, 0 ) = -3.0;
    coordinates( 1, 1 ) = -2.0;
    const LinearEdgeGeometry geometries[] = { original, other, updated };
    for ( IndexT i = 0; i < 3; ++i ) {
      const auto first = geometries[i].position( -1.0 );
      const auto second = geometries[i].position( 1.0 );
      output[0]( i, 0 ) = first[0];
      output[0]( i, 1 ) = first[1];
      output[0]( i, 2 ) = second[0];
      output[0]( i, 3 ) = second[1];
      output[0]( i, 4 ) = geometries[i].length();
    }
  } );
  const ArrayT<MdArray<RealT, 4, 5>, 1, MemorySpace::Host> host( results );
  const MdArray<RealT, 4, 5> expected{
      MdArray<RealT, 5>{ 0.0, 0.0, 3.0, 4.0, 5.0 }, MdArray<RealT, 5>{ -2.0, 1.0, 1.0, 1.0, 3.0 },
      MdArray<RealT, 5>{ 2.0, -1.0, 5.0, 3.0, 5.0 }, MdArray<RealT, 5>{ 0.0, 0.0, 3.0, 4.0, 4.0 } };
  for ( IndexT i = 0; i < expected.size(); ++i ) {
    EXPECT_DOUBLE_EQ( host[0][i], expected[i] );
  }
}

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
  ArrayT<RealT> records( 2 * Coordinates::size(), 2 * Coordinates::size(), allocatorId() );
  const auto output = records.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const Coordinates edge1{ MdArray<RealT, 2>{ 1.0, 2.0 }, MdArray<RealT, 2>{ 4.0, 6.0 } };
    const Coordinates edge2{ MdArray<RealT, 2>{ 3.8, 6.15 }, MdArray<RealT, 2>{ 0.8, 2.15 } };
    const PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<3>> rule_generator;
    const CommonPlaneResidualPairFunction<EdgeQuadratureRuleGenerator<3>> function{ rule_generator, 2.0 };
    const auto result = function.evaluate( edge1.view(), edge2.view() );
    const auto values = result.values();
    for ( int i = 0; i < 2 * Coordinates::size(); ++i ) {
      output[i] = values[i];
    }
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
    NumResults
  };
  constexpr RealT unwritten = -7.0;
  ArrayT<QuadratureResult> records( NumResults, NumResults, allocatorId() );
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
    const PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<3>> rule_generator;
    output[InvalidFirst] = rule_generator.generate( invalid.view(), edge2.view() ).result;
    output[InvalidSecond] = rule_generator.generate( edge1.view(), invalid.view() ).result;
    output[Disjoint] = rule_generator.generate( edge1.view(), disjoint.view() ).result;
    output[Cooriented] = rule_generator.generate( edge1.view(), cooriented.view() ).result;
    output[Perpendicular] = rule_generator.generate( edge1.view(), perpendicular.view() ).result;
    for ( int i = 0; i < values.size(); ++i ) {
      values[i] = unwritten;
    }
    const PairQuadratureRuleView<1, 1> bounded{
        { values.data() + 1, { 2, 1 } }, { values.data() + 1, { 2, 1 } }, { values.data() + 1, 2 } };
    output[Overflow] = rule_generator.generate( edge1.view(), edge2.view(), bounded );
  } );
  const ArrayT<QuadratureResult, 1, MemorySpace::Host> results( records );
  const ArrayT<RealT, 1, MemorySpace::Host> host( storage );
  for ( int invalid : { InvalidFirst, InvalidSecond } ) {
    EXPECT_EQ( results[invalid].status, QuadratureStatus::InvalidGeometry );
    EXPECT_EQ( results[invalid].num_required_points, 0 );
  }
  for ( int empty : { Disjoint, Cooriented, Perpendicular } ) {
    EXPECT_EQ( results[empty].status, QuadratureStatus::Success );
    EXPECT_EQ( results[empty].num_required_points, 0 );
  }
  EXPECT_EQ( results[Overflow].status, QuadratureStatus::InsufficientCapacity );
  EXPECT_EQ( results[Overflow].num_required_points, 3 );
  for ( const auto value : host ) {
    EXPECT_EQ( value, unwritten );
  }
}

// Each base rule integrates only the penetrating interval and uses the same uniform centroid pressure.
TEST_P( CommonPlaneTest, integrates_interpenetrating_overlap_with_configured_rule )
{
  checkInterpenetratingOverlap<1>();
  checkInterpenetratingOverlap<2>();
  checkInterpenetratingOverlap<3>();
}
// A force-only kernel must keep pair slots distinct and preserve active zero-gap and invalid-geometry state.
TEST_P( CommonPlaneTest, value_kernel_returns_forces_and_quadrature_without_derivatives )
{
  using Function = CommonPlaneResidualPairFunction<EdgeQuadratureRuleGenerator<2>>;
  const Function function{ {}, 1.0 };
  const auto x1 = copyToExecutionSpace<RealT>( { 0.0, 2.0 } );
  const auto y1 = copyToExecutionSpace<RealT>( { 0.0, 0.0 } );
  const auto conn1 = copyToExecutionSpace<IndexT>( { 0, 1 } );
  const auto x2 = copyToExecutionSpace<RealT>( { 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0 } );
  const auto y2 = copyToExecutionSpace<RealT>( { 0.1, 0.1, -0.1, -0.1, 0.0, 0.0, 0.0, 0.0 } );
  const auto conn2 = copyToExecutionSpace<IndexT>( { 0, 1, 2, 3, 4, 5, 6, 7 } );
  MeshData mesh1( 0, 1, x1.size(), conn1.data(), EDGE, LINEAR_LAGRANGE, x1.data(), y1.data(), nullptr, memorySpace() );
  MeshData mesh2( 1, 4, x2.size(), conn2.data(), EDGE, LINEAR_LAGRANGE, x2.data(), y2.data(), nullptr, memorySpace() );
  const auto pairs = copyToExecutionSpace<InterfacePair>( { { 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, 3 } } );
  auto buffer = dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    return evaluatePairValues<Exec>( pairs, mesh1.getView(), mesh2.getView(), function );
  } );
  EXPECT_EQ( buffer.numberOfPairs(), pairs.size() );
  EXPECT_EQ( buffer.values().getAllocatorID(), pairs.getAllocatorID() );
  EXPECT_EQ( buffer.quadratureResults().getAllocatorID(), pairs.getAllocatorID() );
  EXPECT_EQ( buffer.contributingPairs().getAllocatorID(), pairs.getAllocatorID() );
  const Array1D<RealT, MemorySpace::Host> values( buffer.values() );
  const Array1D<QuadratureResult, MemorySpace::Host> status( buffer.quadratureResults() );
  const Array1D<IndexT, MemorySpace::Host> active( buffer.contributingPairs() );
  for ( IndexT pair = 0; pair < pairs.size(); ++pair ) {
    EXPECT_EQ( status[pair].status, pair == 3 ? QuadratureStatus::InvalidGeometry : QuadratureStatus::Success );
    EXPECT_EQ( status[pair].num_required_points, pair == 3 ? 0 : 2 );
    EXPECT_EQ( active[pair], pair == 0 || pair == 2 );
    for ( IndexT row = 0; row < Function::size_values; ++row ) {
      const RealT expected = pair == 0 && row % 2 == 1 ? ( row < Function::size_geom1_coords ? -0.1 : 0.1 ) : 0.0;
      EXPECT_NEAR( values[row * pairs.size() + pair], expected, tolerance );
    }
  }
  EXPECT_THROW( dispatchExecutionMode( GetParam(),
                                       [&]<ExecutionMode Exec>() {
                                         return checkPairQuadratureResults<Exec>( buffer.quadratureResults() );
                                       } ),
                std::invalid_argument );
  // A second calculation uses different stiffness and must leave the first returned buffer unchanged.
  auto doubled = dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    return evaluatePairValues<Exec>( pairs, mesh1.getView(), mesh2.getView(), Function{ {}, 2.0 } );
  } );
  const Array1D<RealT, MemorySpace::Host> doubled_values( doubled.values() );
  const Array1D<RealT, MemorySpace::Host> unchanged( buffer.values() );
  for ( IndexT i = 0; i < values.size(); ++i ) {
    EXPECT_EQ( unchanged[i], values[i] );
    EXPECT_NEAR( doubled_values[i], 2.0 * values[i], tolerance );
  }
  const ArrayT<InterfacePair> empty_pairs( 0, 0, allocatorId() );
  auto empty = dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    return evaluatePairValues<Exec>( empty_pairs, mesh1.getView(), mesh2.getView(), function );
  } );
  EXPECT_EQ( empty.numberOfPairs(), 0 );
  EXPECT_EQ( empty.values().size(), 0 );
  EXPECT_EQ( empty.quadratureResults().size(), 0 );
  EXPECT_EQ( empty.contributingPairs().size(), 0 );
  EXPECT_NO_THROW( dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    return checkPairQuadratureResults<Exec>( empty.quadratureResults() );
  } ) );
}
INSTANTIATE_TEST_SUITE_P( execution, CommonPlaneTest, tribol::detail::enabledExecutionModes() );

#ifdef TRIBOL_USE_ENZYME
/** @brief Compare active-geometry Jacobians with central differences. @tparam NumPoints Gauss point count. */
template <int NumPoints>
void checkJacobian()
{
  using Function = CommonPlaneResidualPairFunction<EdgeQuadratureRuleGenerator<NumPoints>, OffsetGap<CommonPlaneGap>>;
  const PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>> rule_generator;
  const Function function{ rule_generator, 2.5, OffsetGap{ CommonPlaneGap{}, 0.03 } };
  constexpr RealT step =
      2.0e-7;  // Small enough to resolve derivatives without changing the active or overlap branches.
  const Coordinates cases1[] = { Coordinates{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.2 } },
                                 Coordinates{ MdArray<RealT, 2>{ -1.0, 0.0 }, MdArray<RealT, 2>{ 1.0, 0.0 } } };
  const Coordinates cases2[] = { Coordinates{ MdArray<RealT, 2>{ 1.8, 0.35 }, MdArray<RealT, 2>{ 0.2, 0.1 } },
                                 Coordinates{ MdArray<RealT, 2>{ 0.8, 0.2 }, MdArray<RealT, 2>{ -0.8, -0.2 } } };
  for ( int geometry = 0; geometry < 2; ++geometry ) {
    const auto coordinates = packPairCoordinates( cases1[geometry].view(), cases2[geometry].view() );
    const auto expected = function.evaluate( cases1[geometry].view(), cases2[geometry].view() );
    const auto result =
        EnzymePairDerivative<Function>::valuesAndJacobian( cases1[geometry].view(), cases2[geometry].view(), function );
    const auto expected_values = expected.values();
    ASSERT_EQ( result.quadrature_result.status, QuadratureStatus::Success );
    EXPECT_EQ( result.quadrature_result.num_required_points, NumPoints );
    EXPECT_TRUE( result.has_contributing_points );
    for ( IndexT row = 0; row < Function::size_values; ++row ) {
      EXPECT_DOUBLE_EQ( result.values[row], expected_values[row] );
    }
    for ( int column = 0; column < Function::size_coords; ++column ) {
      auto plus_coordinates = coordinates;
      auto minus_coordinates = coordinates;
      plus_coordinates[column] += step;
      minus_coordinates[column] -= step;
      const auto plus =
          function.evaluate( { plus_coordinates.data(), Function::size_geom1_coords },
                             { plus_coordinates.data() + Function::size_geom1_coords, Function::size_geom2_coords } );
      const auto minus =
          function.evaluate( { minus_coordinates.data(), Function::size_geom1_coords },
                             { minus_coordinates.data() + Function::size_geom1_coords, Function::size_geom2_coords } );
      ASSERT_EQ( plus.quadrature_result.status, QuadratureStatus::Success );
      ASSERT_EQ( minus.quadrature_result.status, QuadratureStatus::Success );
      const auto plus_values = plus.values();
      const auto minus_values = minus.values();
      for ( int row = 0; row < Function::size_values; ++row ) {
        EXPECT_NEAR( result.jacobian( row, column ), ( plus_values[row] - minus_values[row] ) / ( 2.0 * step ), 2.0e-7 )
            << "points=" << NumPoints << ", geometry=" << geometry << ", row=" << row << ", column=" << column;
      }
    }
  }
  const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.0 } };
  const Coordinates edge2{ MdArray<RealT, 2>{ 2.0, 0.0 }, MdArray<RealT, 2>{ 0.0, 0.0 } };
  using UnshiftedFunction = CommonPlaneResidualPairFunction<EdgeQuadratureRuleGenerator<NumPoints>>;
  const UnshiftedFunction unshifted{ rule_generator, 1.0 };
  const auto zero = unshifted.evaluate( edge1.view(), edge2.view() );
  EXPECT_TRUE( zero.has_contributing_points );
  const auto derivatives =
      EnzymePairDerivative<UnshiftedFunction>::valuesAndJacobian( edge1.view(), edge2.view(), unshifted );
  EXPECT_TRUE( derivatives.has_contributing_points );
  for ( IndexT row = 0; row < UnshiftedFunction::size_values; ++row ) {
    EXPECT_EQ( derivatives.values[row], 0.0 );
  }
  EXPECT_GT( derivatives.jacobian( 1, 1 ), 0.0 );
  const Coordinates invalid{};
  const auto failure =
      EnzymePairDerivative<UnshiftedFunction>::valuesAndJacobian( invalid.view(), edge2.view(), unshifted );
  EXPECT_EQ( failure.quadrature_result.status, QuadratureStatus::InvalidGeometry );
  EXPECT_EQ( failure.quadrature_result.num_required_points, 0 );
  EXPECT_FALSE( failure.has_contributing_points );
  for ( int row = 0; row < failure.values.size(); ++row ) {
    EXPECT_EQ( failure.values[row], 0.0 );
    for ( IndexT column = 0; column < UnshiftedFunction::size_coords; ++column ) {
      EXPECT_EQ( failure.jacobian( row, column ), 0.0 );
    }
  }
}

// All reference counts differentiate current normals, interval bounds, edge lengths, and the interpenetration boundary
// and centroid pressure.
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
