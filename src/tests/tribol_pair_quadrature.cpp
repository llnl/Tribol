// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * Tests single-source quadrature, independent target projections, paired values, and bounded writes on every enabled
 * backend. Host batch tests separately check capacity errors, packing, pair ordering, returned arrays surviving
 * temporary inputs, and factory selection by polynomial order. Geometry acceptance rules belong to
 * tribol_interface_pair_checks; these tests check generated quadrature data and its storage contract.
 */

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "ExecutionTest.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/integ/GeneratePairQuadrature.hpp"
#include "tribol/integ/QuadratureRuleFactory.hpp"
#include "tribol/integ/EdgeQuadratureRule.hpp"
#include "tribol/integ/SmoothProjectedEdgeInterval.hpp"

namespace {
using namespace tribol;
using Status = QuadratureStatus;
constexpr RealT unwritten = -7.0;
constexpr double tolerance = 1.0e-12;

struct GeneratedPoints {
  QuadratureResult result;
  ArrayT<RealT, 1, MemorySpace::Host> mesh1;
  ArrayT<RealT, 1, MemorySpace::Host> mesh2;
  ArrayT<RealT, 1, MemorySpace::Host> weights;
};

class PairQuadratureTest : public detail::ExecutionTest {
 protected:
  template <int Count>
  GeneratedPoints generate( std::initializer_list<RealT> endpoints, double smoothing, IndexT capacity )
  {
    const auto coords = copyToExecutionSpace<RealT>( endpoints );
    // A value before and after each output slice detects writes beyond the supplied capacity, including capacity zero.
    const IndexT storage_size = capacity + 2;
    ArrayT<RealT> mesh1( storage_size, storage_size, allocatorId() );
    ArrayT<RealT> mesh2( storage_size, storage_size, allocatorId() );
    ArrayT<RealT> weights( storage_size, storage_size, allocatorId() );
    mesh1.fill( unwritten );
    mesh2.fill( unwritten );
    weights.fill( unwritten );
    ArrayT<QuadratureResult> result( 1, 1, allocatorId() );
    const auto coordinate_view = coords.view();
    const axom::StackArray<axom::IndexType, 2> shape{ capacity, 1 };
    const PairQuadratureView<RealT, 1> output{ ArrayViewT<RealT, 2>{ mesh1.data() + 1, shape },
                                               ArrayViewT<RealT, 2>{ mesh2.data() + 1, shape },
                                               ArrayViewT<RealT>{ weights.data() + 1, capacity } };
    auto status = result.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      using QuadratureRule = EdgeQuadratureRule<Count>;
      const QuadratureRule quadrature_rule{};
      const SmoothProjectedEdgeInterval select_interval{ smoothing };
      const typename QuadratureRule::template Coordinates<RealT> source{ coordinate_view.data(), 4 };
      const typename QuadratureRule::template Coordinates<RealT> target{ coordinate_view.data() + 4, 4 };
      const auto interval = select_interval.template operator()<RealT>( source, target );
      const auto quadrature = quadrature_rule.template generate<RealT>( source, interval );
      status[0] = projectEdgeQuadrature<RealT>( quadrature, target, output );
    } );
    const ArrayT<QuadratureResult, 1, MemorySpace::Host> host_result( result );
    GeneratedPoints host{ host_result[0], ArrayT<RealT, 1, MemorySpace::Host>( mesh1 ),
                          ArrayT<RealT, 1, MemorySpace::Host>( mesh2 ),
                          ArrayT<RealT, 1, MemorySpace::Host>( weights ) };
    for ( const auto* values : { &host.mesh1, &host.mesh2, &host.weights } ) {
      EXPECT_EQ( ( *values )[0], unwritten );
      EXPECT_EQ( ( *values )[storage_size - 1], unwritten );
    }
    return host;
  }

  template <int Count>
  void checkSingleSurfaceAndProjections()
  {
    using QuadratureRule = EdgeQuadratureRule<Count>;
    enum Column
    {
      Parent,
      X,
      Y,
      Weight,
      Target1,
      Target2,
      IntervalParent,
      IntervalWeight,
      ColumnCount
    };
    enum ResultKind
    {
      Source,
      IntervalSource,
      FirstProjection,
      SecondProjection,
      ResultCount
    };
    constexpr RealT source_length = 4.0;
    // The targets have different lengths and orientations; neither target measure belongs in the source weights.
    const auto coordinates =
        copyToExecutionSpace<RealT>( { 0.0, 0.0, source_length, 0.0, 3.0, 0.2, 1.0, 0.2, 0.0, -0.3, 6.0, -0.3 } );
    const auto inputs = coordinates.view();
    ArrayT<RealT, 2> records( axom::StackArray<axom::IndexType, 2>{ Count, ColumnCount }, allocatorId() );
    ArrayT<QuadratureResult> results( ResultCount, ResultCount, allocatorId() );
    const auto values = records.view();
    const auto statuses = results.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      constexpr int coordinate_count = QuadratureRule::endpoint_count * QuadratureRule::spatial_dimension;
      const typename QuadratureRule::template Coordinates<RealT> surface{ inputs.data(), coordinate_count };
      const typename QuadratureRule::template Coordinates<RealT> target1{ inputs.data() + coordinate_count,
                                                                          coordinate_count };
      const typename QuadratureRule::template Coordinates<RealT> target2{ inputs.data() + 2 * coordinate_count,
                                                                          coordinate_count };
      const QuadratureRule quadrature_rule{};
      const auto source = quadrature_rule.template generate<RealT>( surface );
      // An explicitly supplied [0, 1] parent interval integrates the right half without consulting a target edge.
      const EdgeQuadratureInterval<RealT> right_half{ MdArray<RealT, 2>{ 0.0, 1.0 }, true };
      const auto partial = quadrature_rule.template generate<RealT>( surface, right_half );
      statuses[IntervalSource] = partial.result;
      MdArray<RealT, Count, 1> projected1{}, projected2{};
      const axom::StackArray<axom::IndexType, 2> shape{ Count, 1 };
      statuses[Source] = source.result;
      statuses[FirstProjection] =
          projectQuadratureToEdge( source.getView(), target1, ArrayViewT<RealT, 2>{ projected1.data(), shape } );
      statuses[SecondProjection] =
          projectQuadratureToEdge( source.getView(), target2, ArrayViewT<RealT, 2>{ projected2.data(), shape } );
      for ( int q = 0; q < Count; ++q ) {
        values( q, Parent ) = source.parent_coordinates( q, 0 );
        values( q, X ) = source.physical_coordinates( q, 0 );
        values( q, Y ) = source.physical_coordinates( q, 1 );
        values( q, Weight ) = source.physical_weights[q];
        values( q, Target1 ) = projected1( q, 0 );
        values( q, Target2 ) = projected2( q, 0 );
        values( q, IntervalParent ) = partial.parent_coordinates( q, 0 );
        values( q, IntervalWeight ) = partial.physical_weights[q];
      }
    } );
    const ArrayT<RealT, 2, MemorySpace::Host> actual( records );
    const ArrayT<QuadratureResult, 1, MemorySpace::Host> status( results );
    for ( const auto result : status ) {
      ASSERT_EQ( result.status, Status::Success );
      ASSERT_EQ( result.required_point_count, Count );
    }
    for ( int q = 0; q < Count; ++q ) {
      const RealT xi = actual( q, Parent );
      EXPECT_NEAR( actual( q, X ), source_length * ( 1.0 + xi ) / 2.0, tolerance );
      EXPECT_EQ( actual( q, Y ), 0.0 );
      EXPECT_NEAR( actual( q, Target1 ), -2.0 * xi, tolerance );
      EXPECT_NEAR( actual( q, Target2 ), ( 2.0 * xi - 1.0 ) / 3.0, tolerance );
    }
    // The full source interval must integrate every supported polynomial degree, even after both projections.
    for ( int degree = 0; degree <= QuadratureRule::polynomial_order; ++degree ) {
      double moment = 0.0;
      for ( int q = 0; q < Count; ++q ) moment += actual( q, Weight ) * std::pow( actual( q, Parent ), degree );
      const double expected = degree % 2 == 0 ? source_length / ( degree + 1 ) : 0.0;
      EXPECT_NEAR( moment, expected, tolerance ) << "points=" << Count << ", degree=" << degree;
      double partial_moment = 0.0;
      for ( int q = 0; q < Count; ++q ) {
        partial_moment += actual( q, IntervalWeight ) * std::pow( actual( q, IntervalParent ), degree );
      }
      EXPECT_NEAR( partial_moment, source_length / ( 2.0 * ( degree + 1 ) ), tolerance );
    }
  }

  template <int Count>
  void checkSourceFailures()
  {
    using QuadratureRule = EdgeQuadratureRule<Count>;
    enum ResultKind
    {
      ShortSource,
      InvalidSource,
      InvalidInterval,
      ShortProjection,
      InvalidTarget,
      ResultCount
    };
    // Surround a short output slice with sentinel points, so all three source arrays are checked for overrun.
    constexpr IndexT storage_count = Count + 2;
    constexpr IndexT short_capacity = Count - 1;
    const auto coordinates = copyToExecutionSpace<RealT>( { 0.0, 0.0, 4.0, 0.0 } );
    const auto inputs = coordinates.view();
    ArrayT<RealT, 2> parent( axom::StackArray<axom::IndexType, 2>{ storage_count, 1 }, allocatorId() );
    ArrayT<RealT, 2> physical( axom::StackArray<axom::IndexType, 2>{ storage_count, 2 }, allocatorId() );
    ArrayT<RealT> weights( storage_count, storage_count, allocatorId() );
    parent.fill( unwritten );
    physical.fill( unwritten );
    weights.fill( unwritten );
    ArrayT<QuadratureResult> results( ResultCount, ResultCount, allocatorId() );
    const auto statuses = results.view();
    const auto parent_view = parent.view();
    const auto physical_view = physical.view();
    const auto weight_view = weights.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      const typename QuadratureRule::template Coordinates<RealT> surface{ inputs.data(), 4 };
      const QuadratureRule quadrature_rule{};
      const SurfaceQuadratureView<RealT, 1, 2> output{
          ArrayViewT<RealT, 2>{ parent_view.data() + 1, axom::StackArray<axom::IndexType, 2>{ short_capacity, 1 } },
          ArrayViewT<RealT, 2>{ physical_view.data() + 2, axom::StackArray<axom::IndexType, 2>{ short_capacity, 2 } },
          ArrayViewT<RealT>{ weight_view.data() + 1, short_capacity } };
      statuses[ShortSource] = quadrature_rule.template generate<RealT>( surface, {}, output );
      MdArray<RealT, 2, 2> degenerate{};
      statuses[InvalidSource] = quadrature_rule.template generate<RealT>( degenerate.view(), {}, output );
      const EdgeQuadratureInterval<RealT> invalid_interval{ MdArray<RealT, 2>{ -1.0, 1.0 }, false };
      statuses[InvalidInterval] = quadrature_rule.template generate<RealT>( surface, invalid_interval, output );
      const auto source = quadrature_rule.template generate<RealT>( surface );
      const ArrayViewT<RealT, 2> projected{ parent_view.data() + 1,
                                            axom::StackArray<axom::IndexType, 2>{ short_capacity, 1 } };
      statuses[ShortProjection] = projectQuadratureToEdge( source.getView(), surface, projected );
      statuses[InvalidTarget] = projectQuadratureToEdge<RealT>( source.getView(), degenerate.view(), projected );
    } );
    const ArrayT<QuadratureResult, 1, MemorySpace::Host> status( results );
    for ( const auto kind : { ShortSource, ShortProjection } ) {
      EXPECT_EQ( status[kind].status, Status::InsufficientCapacity );
      EXPECT_EQ( status[kind].required_point_count, Count );
    }
    for ( const auto kind : { InvalidSource, InvalidInterval, InvalidTarget } ) {
      EXPECT_EQ( status[kind].status, Status::InvalidGeometry );
      EXPECT_EQ( status[kind].required_point_count, 0 );
    }
    const ArrayT<RealT, 2, MemorySpace::Host> parent_values( parent ), physical_values( physical );
    const ArrayT<RealT, 1, MemorySpace::Host> weight_values( weights );
    for ( IndexT q = 0; q < storage_count; ++q ) {
      EXPECT_EQ( parent_values( q, 0 ), unwritten );
      EXPECT_EQ( physical_values( q, 0 ), unwritten );
      EXPECT_EQ( physical_values( q, 1 ), unwritten );
      EXPECT_EQ( weight_values[q], unwritten );
    }
  }

  template <int Count>
  void checkMoments()
  {
    constexpr double edge_length = 4.0;
    constexpr double jacobian = edge_length / 2.0;
    constexpr double smoothed_endpoint = 0.95;
    const auto points = generate<Count>( { 0.0, 0.0, edge_length, 0.0, edge_length, 0.2, 0.0, 0.2 }, 0.1, Count );
    ASSERT_EQ( points.result.status, Status::Success );
    ASSERT_EQ( points.result.required_point_count, Count );
    for ( int q = 0; q < Count; ++q ) {
      EXPECT_GT( points.weights[q + 1], 0.0 );
      EXPECT_NEAR( points.mesh2[q + 1], -points.mesh1[q + 1], tolerance );
    }
    for ( int degree = 0; degree < 2 * Count; ++degree ) {
      double moment = 0.0;
      for ( int q = 0; q < Count; ++q ) {
        moment += points.weights[q + 1] * std::pow( points.mesh1[q + 1], degree );
      }
      const double expected =
          degree % 2 == 0 ? jacobian * 2.0 * std::pow( smoothed_endpoint, degree + 1 ) / ( degree + 1 ) : 0.0;
      EXPECT_NEAR( moment, expected, tolerance ) << "points=" << Count << ", degree=" << degree;
    }
  }

  template <int Count>
  void checkOverlapAndFailure()
  {
    // Half of the integration edge is covered, so its physical measure is two and its parent interval is [-0.5, 0.5].
    const auto partial = generate<Count>( { 0.0, 0.0, 4.0, 0.0, 3.0, 0.2, 1.0, 0.2 }, 0.0, Count );
    ASSERT_EQ( partial.result.status, Status::Success );
    ASSERT_EQ( partial.result.required_point_count, Count );
    double measure = 0.0;
    for ( int q = 0; q < Count; ++q ) {
      measure += partial.weights[q + 1];
      EXPECT_LE( std::abs( partial.mesh1[q + 1] ), 0.5 );
      EXPECT_NEAR( partial.mesh2[q + 1], -2.0 * partial.mesh1[q + 1], tolerance );
    }
    EXPECT_NEAR( measure, 2.0, tolerance );
    // This edge lies beyond mesh1 but within its smoothing support. The smoothed physical interval has length 0.05.
    const auto support = generate<Count>( { 0.0, 0.0, 4.0, 0.0, 4.3, 0.2, 4.1, 0.2 }, 0.1, Count );
    ASSERT_EQ( support.result.required_point_count, Count );
    measure = 0.0;
    for ( int q = 0; q < Count; ++q ) measure += support.weights[q + 1];
    EXPECT_NEAR( measure, 0.05, tolerance );
    // Covering the entire smoothing support must recover the full physical edge length.
    const auto extended = generate<Count>( { 0.0, 0.0, 4.0, 0.0, 5.0, 0.2, -1.0, 0.2 }, 0.1, Count );
    measure = 0.0;
    for ( int q = 0; q < Count; ++q ) measure += extended.weights[q + 1];
    EXPECT_NEAR( measure, 4.0, tolerance );
    const auto outside = generate<Count>( { 0.0, 0.0, 4.0, 0.0, 6.0, 0.2, 5.0, 0.2 }, 0.1, 0 );
    EXPECT_EQ( outside.result.status, Status::Success );
    EXPECT_EQ( outside.result.required_point_count, 0 );
    const auto collapsed = generate<Count>( { 0.0, 0.0, 4.0, 0.0, 2.0, 0.1, 2.0, 0.3 }, 0.0, Count );
    EXPECT_EQ( collapsed.result.status, Status::Success );
    EXPECT_EQ( collapsed.result.required_point_count, 0 );
    for ( const auto coords : { std::initializer_list<RealT>{ 0.0, 0.0, 0.0, 0.0, 3.0, 0.2, 1.0, 0.2 },
                                std::initializer_list<RealT>{ 0.0, 0.0, 4.0, 0.0, 2.0, 0.2, 2.0, 0.2 } } ) {
      const auto invalid = generate<Count>( coords, 0.1, Count );
      EXPECT_EQ( invalid.result.status, Status::InvalidGeometry );
      EXPECT_EQ( invalid.result.required_point_count, 0 );
    }
    const auto overflow = generate<Count>( { 0.0, 0.0, 4.0, 0.0, 4.0, 0.2, 0.0, 0.2 }, 0.1, Count - 1 );
    EXPECT_EQ( overflow.result.status, Status::InsufficientCapacity );
    EXPECT_EQ( overflow.result.required_point_count, Count );
    const auto spare = generate<Count>( { 0.0, 0.0, 4.0, 0.0, 4.0, 0.2, 0.0, 0.2 }, 0.1, Count + 1 );
    EXPECT_EQ( spare.result.required_point_count, Count );
    EXPECT_EQ( spare.weights[Count + 1], unwritten );
  }
  template <int Count>
  void checkLocalStorageAndFrozenMapping()
  {
    using QuadratureRule = EdgeQuadratureRule<Count>;
    enum Column
    {
      Parent1,
      PhysicalWeight,
      UpdatedParent2,
      UpdatedWeight,
      ReferenceWeight,
      ColumnCount
    };
    enum ResultKind
    {
      CopiedPhysical,
      UpdatedPhysical,
      FrozenReference,
      ResultCount
    };
    constexpr RealT length_scale = 2.0;
    constexpr RealT initial_jacobian = 2.0;
    const auto coordinates = copyToExecutionSpace<RealT>( { 0.0, 0.0, 4.0, 0.0, 4.0, 0.2, 0.0, 0.2 } );
    const auto inputs = coordinates.view();
    const axom::StackArray<axom::IndexType, 2> shape{ Count, ColumnCount };
    ArrayT<RealT, 2> values( shape, allocatorId() );
    ArrayT<QuadratureResult> results( ResultCount, ResultCount, allocatorId() );
    const auto output_records = values.view();
    const auto output_statuses = results.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      constexpr int coordinate_count = QuadratureRule::endpoint_count * QuadratureRule::spatial_dimension;
      const typename QuadratureRule::template Coordinates<RealT> mesh1{ inputs.data(), coordinate_count };
      const typename QuadratureRule::template Coordinates<RealT> mesh2{ inputs.data() + coordinate_count,
                                                                        coordinate_count };
      const QuadratureRule quadrature_rule{};
      const SmoothProjectedEdgeInterval select_interval{};
      const auto reference = quadrature_rule.template generateReferenceQuadrature<RealT>(
          select_interval.template operator()<RealT>( mesh1, mesh2 ) );
      auto original =
          quadrature_rule.template generate<RealT>( mesh1, select_interval.template operator()<RealT>( mesh1, mesh2 ) );
      const auto copied = original;
      original.parent_coordinates( 0, 0 ) = unwritten;
      original.physical_weights[0] = unwritten;
      const auto copied_view = copied.getView();
      MdArray<RealT, QuadratureRule::endpoint_count, QuadratureRule::spatial_dimension> changed;
      for ( int i = 0; i < coordinate_count; ++i ) changed.data()[i] = inputs[i];
      changed( 1, 0 ) *= length_scale;
      const auto updated =
          quadrature_rule.template mapReferenceQuadrature<RealT>( changed.view(), reference.getView() );
      const auto mapped = projectEdgeQuadrature<RealT>( updated, mesh2 );
      const auto mapped_view = mapped.getView();
      output_statuses[CopiedPhysical] = copied.result;
      output_statuses[UpdatedPhysical] = mapped.result;
      output_statuses[FrozenReference] = reference.result;
      for ( int q = 0; q < Count; ++q ) {
        output_records( q, Parent1 ) = copied_view.parent_coordinates( q, 0 );
        output_records( q, PhysicalWeight ) = copied_view.physical_weights[q];
        output_records( q, UpdatedParent2 ) = mapped_view.mesh2_parent_coordinates( q, 0 );
        output_records( q, UpdatedWeight ) = mapped_view.physical_weights[q];
        output_records( q, ReferenceWeight ) = reference.getView().reference_weights[q];
      }
    } );
    const ArrayT<RealT, 2, MemorySpace::Host> records( values );
    const ArrayT<QuadratureResult, 1, MemorySpace::Host> statuses( results );
    for ( const auto status : statuses ) {
      ASSERT_EQ( status.status, Status::Success );
      ASSERT_EQ( status.required_point_count, Count );
    }
    const auto bounded = generate<Count>( { 0.0, 0.0, 4.0, 0.0, 4.0, 0.2, 0.0, 0.2 }, 0.1, Count );
    for ( int q = 0; q < Count; ++q ) {
      EXPECT_EQ( records( q, Parent1 ), bounded.mesh1[q + 1] );
      EXPECT_EQ( records( q, PhysicalWeight ), bounded.weights[q + 1] );
      EXPECT_NEAR( records( q, UpdatedParent2 ), -1.0 - length_scale * bounded.mesh1[q + 1], tolerance );
      EXPECT_NEAR( records( q, UpdatedWeight ), length_scale * bounded.weights[q + 1], tolerance );
      EXPECT_NEAR( records( q, ReferenceWeight ), bounded.weights[q + 1] / initial_jacobian, tolerance );
    }
  }
};

// One ordinary Gauss rule covers the source edge and can be reused for two targets without changing its measure.
TEST_P( PairQuadratureTest, generates_one_source_rule_and_projects_to_two_targets )
{
  checkSingleSurfaceAndProjections<1>();
  checkSingleSurfaceAndProjections<2>();
  checkSingleSurfaceAndProjections<3>();
}

// Generation and projection must distinguish invalid geometry from insufficient capacity and leave buffers untouched.
TEST_P( PairQuadratureTest, rejects_invalid_source_or_target_and_short_buffers )
{
  checkSourceFailures<1>();
  checkSourceFailures<2>();
  checkSourceFailures<3>();
}

// Each point count must integrate its supported polynomial degrees with the physical edge measure included once.
TEST_P( PairQuadratureTest, integrates_moments_and_projects_to_mesh2 )
{
  checkMoments<1>();
  checkMoments<2>();
  checkMoments<3>();
}

// Distinguish zero measure from invalid geometry and verify that short output slices never overwrite their neighbors.
TEST_P( PairQuadratureTest, handles_overlap_zero_measure_and_insufficient_capacity )
{
  checkOverlapAndFailure<1>();
  checkOverlapAndFailure<2>();
  checkOverlapAndFailure<3>();
}

// Local copies must own their arrays, and frozen reference weights must acquire the updated physical Jacobian.
TEST_P( PairQuadratureTest, owns_local_rule_data_and_remaps_frozen_reference_points )
{
  checkLocalStorageAndFrozenMapping<1>();
  checkLocalStorageAndFrozenMapping<2>();
  checkLocalStorageAndFrozenMapping<3>();
}

// Copies retain their smoothing fraction, and computing another interval must not change a configured selector.
TEST_P( PairQuadratureTest, keeps_configuration_in_each_interval_selector )
{
  using QuadratureRule = EdgeQuadratureRule<3>;
  enum Measure
  {
    CopiedInterval,
    UnsmoothedInterval,
    OriginalInterval,
    MeasureCount
  };
  constexpr double smoothing_fraction = 0.2;
  constexpr RealT edge_length = 4.0;
  const QuadratureRule quadrature_rule{};
  const SmoothProjectedEdgeInterval smoothed{ smoothing_fraction };
  const SmoothProjectedEdgeInterval unsmoothed{ 0.0 };
  const auto copied = smoothed;
  const auto coordinates = copyToExecutionSpace<RealT>( { 0.0, 0.0, edge_length, 0.0, edge_length, 0.2, 0.0, 0.2 } );
  const auto inputs = coordinates.view();
  ArrayT<RealT> measures( MeasureCount, MeasureCount, allocatorId() );
  const auto output = measures.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    constexpr int coordinate_count = QuadratureRule::endpoint_count * QuadratureRule::spatial_dimension;
    const QuadratureRule::Coordinates<RealT> mesh1{ inputs.data(), coordinate_count };
    const QuadratureRule::Coordinates<RealT> mesh2{ inputs.data() + coordinate_count, coordinate_count };
    const auto first =
        quadrature_rule.template generate<RealT>( mesh1, copied.template operator()<RealT>( mesh1, mesh2 ) );
    const auto middle =
        quadrature_rule.template generate<RealT>( mesh1, unsmoothed.template operator()<RealT>( mesh1, mesh2 ) );
    const auto last =
        quadrature_rule.template generate<RealT>( mesh1, smoothed.template operator()<RealT>( mesh1, mesh2 ) );
    output[CopiedInterval] = output[UnsmoothedInterval] = output[OriginalInterval] = 0.0;
    for ( IndexT q = 0; q < first.result.required_point_count; ++q ) {
      output[CopiedInterval] += first.getView().physical_weights[q];
      output[UnsmoothedInterval] += middle.getView().physical_weights[q];
      output[OriginalInterval] += last.getView().physical_weights[q];
    }
  } );
  const ArrayT<RealT, 1, MemorySpace::Host> result( measures );
  const double expected_smoothed_length = edge_length * ( 1.0 - smoothing_fraction / 2.0 );
  EXPECT_NEAR( result[CopiedInterval], expected_smoothed_length, tolerance );
  EXPECT_NEAR( result[UnsmoothedInterval], edge_length, tolerance );
  EXPECT_NEAR( result[OriginalInterval], expected_smoothed_length, tolerance );
}

// Emit two points per region with two parent components, without implementing a surface intersection algorithm.
struct VariableSurfaceRule {
  static constexpr int parent_dimension = 2;
  static constexpr int spatial_dimension = 3;
  static constexpr int points_per_base_rule = 2;
  struct Parameters {
    IndexT region_capacity{ 1 };
  };
  TRIBOL_DEFAULT_HOST_DEVICE VariableSurfaceRule() = default;
  TRIBOL_HOST_DEVICE explicit VariableSurfaceRule( Parameters parameters ) : parameters_( parameters ) {}
  TRIBOL_HOST_DEVICE IndexT pointCapacity() const { return points_per_base_rule * parameters_.region_capacity; }
  TRIBOL_HOST_DEVICE QuadratureResult
  generate( IndexT regions, SurfaceQuadratureView<RealT, parent_dimension, spatial_dimension> output ) const
  {
    const IndexT count = regions * points_per_base_rule;
    for ( IndexT q = 0; q < count; ++q ) {
      if ( q >= output.capacity() ) continue;
      output.parent_coordinates( q, 0 ) = static_cast<RealT>( q );
      output.parent_coordinates( q, 1 ) = static_cast<RealT>( regions );
      output.physical_coordinates( q, 0 ) = static_cast<RealT>( q );
      output.physical_coordinates( q, 1 ) = static_cast<RealT>( regions );
      output.physical_coordinates( q, 2 ) = 0.0;
      output.physical_weights[q] = 1.0;
    }
    return { count > output.capacity() ? Status::InsufficientCapacity : Status::Success, count };
  }

 private:
  Parameters parameters_{};
};

// Use known target coordinates to check batch packing separately from geometry and projection algorithms.
TRIBOL_HOST_DEVICE QuadratureResult generateSurfacePair( const VariableSurfaceRule& rule, IndexT regions,
                                                         PairQuadratureView<RealT, 2> output )
{
  const IndexT count = regions * VariableSurfaceRule::points_per_base_rule;
  for ( IndexT q = 0; q < count && q < output.capacity(); ++q ) {
    for ( int d = 0; d < VariableSurfaceRule::parent_dimension; ++d ) {
      const RealT coordinate = d == 0 ? static_cast<RealT>( q ) : static_cast<RealT>( regions );
      output.mesh1_parent_coordinates( q, d ) = coordinate;
      output.mesh2_parent_coordinates( q, d ) = -coordinate;
    }
    output.physical_weights[q] = 1.0;
  }
  return { count > rule.pointCapacity() ? Status::InsufficientCapacity : Status::Success, count };
}

// Runtime-sized surface output must report the full required count after reaching capacity on every enabled backend.
TEST_P( PairQuadratureTest, supports_variable_surface_point_counts )
{
  const VariableSurfaceRule quadrature_rule{};
  constexpr IndexT regions = 3;
  constexpr IndexT capacity = VariableSurfaceRule::points_per_base_rule;
  constexpr int dimension = VariableSurfaceRule::parent_dimension;
  const axom::StackArray<axom::IndexType, 2> shape{ capacity, dimension };
  ArrayT<RealT, 2> parent( shape, allocatorId() );
  ArrayT<RealT, 2> physical( axom::StackArray<axom::IndexType, 2>{ capacity, VariableSurfaceRule::spatial_dimension },
                             allocatorId() );
  ArrayT<RealT> weights( capacity + 1, capacity + 1, allocatorId() );
  weights.fill( unwritten );
  ArrayT<QuadratureResult> results( 1, 1, allocatorId() );
  const SurfaceQuadratureView<RealT, dimension, VariableSurfaceRule::spatial_dimension> output{
      parent.view(), physical.view(), ArrayViewT<RealT>{ weights.data(), capacity } };
  auto status = results.view();
  forAllExec( GetParam(), 1,
              [=] TRIBOL_HOST_DEVICE( IndexT ) { status[0] = quadrature_rule.generate( regions, output ); } );
  const ArrayT<QuadratureResult, 1, MemorySpace::Host> result( results );
  const ArrayT<RealT, 1, MemorySpace::Host> values( weights );
  EXPECT_EQ( result[0].status, Status::InsufficientCapacity );
  EXPECT_EQ( result[0].required_point_count, regions * VariableSurfaceRule::points_per_base_rule );
  EXPECT_EQ( values[capacity], unwritten );
}

INSTANTIATE_TEST_SUITE_P( execution, PairQuadratureTest, detail::enabledExecutionModes() );

class QuadratureBatchExecutionTest : public detail::ExecutionTest {};

InterfaceQuadratureData copyQuadratureToHost( InterfaceQuadratureData& source )
{
  const int allocator = getResourceAllocatorID( MemorySpace::Host );
  ArrayT<InterfacePair> pairs( source.getInterfacePairs(), allocator );
  ArrayT<IndexT> offsets( source.getPairOffsets(), allocator );
  ArrayT<RealT, 2> mesh1( source.getMesh1ParentCoordinates(), allocator );
  ArrayT<RealT, 2> mesh2( source.getMesh2ParentCoordinates(), allocator );
  ArrayT<RealT> weights( source.getWeights(), allocator );
  return { source.getParentDimension(), std::move( pairs ), std::move( offsets ),
           std::move( mesh1 ),          std::move( mesh2 ), std::move( weights ) };
}

// Mixed region counts check unused capacity, omitted zero-point pairs, ordering, and exactly one call per pair.
TEST_P( QuadratureBatchExecutionTest, packs_variable_point_counts_without_regenerating_pairs )
{
  const auto pairs = copyToExecutionSpace<InterfacePair>( { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } } );
  const auto regions = copyToExecutionSpace<IndexT>( { 0, 1, 3, 2 } );
  ArrayT<IndexT> calls( pairs.size(), pairs.size(), allocatorId() );
  calls.fill( 0 );
  const VariableSurfaceRule quadrature_rule{ { 3 } };
  const auto region_values = regions.view();
  auto call_values = calls.view();
  auto retained =
      detail::generateQuadrature( GetParam(), quadrature_rule, pairs.view(), allocatorId(),
                                  [=] TRIBOL_HOST_DEVICE( IndexT pair, PairQuadratureView<RealT, 2> output ) {
                                    ++call_values[pair];
                                    return generateSurfacePair( quadrature_rule, region_values[pair], output );
                                  } );
  EXPECT_EQ( retained.getAllocatorId(), allocatorId() );
  const ArrayT<IndexT, 1, MemorySpace::Host> host_calls( calls );
  for ( IndexT pair = 0; pair < host_calls.size(); ++pair ) EXPECT_EQ( host_calls[pair], 1 );

  const auto device_view = retained.getView<2>();
  ArrayT<RealT> pair_measures( device_view.numberOfPairs(), device_view.numberOfPairs(), allocatorId() );
  auto measures = pair_measures.view();
  forAllExec( GetParam(), device_view.numberOfPairs(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
    RealT measure = 0.0;
    for ( IndexT point = 0; point < device_view.numberOfQuadraturePoints( pair ); ++point ) {
      measure += device_view.getQuadraturePoint( pair, point ).getWeight();
    }
    measures[pair] = measure;
  } );
  const ArrayT<RealT, 1, MemorySpace::Host> host_measures( pair_measures );
  EXPECT_EQ( host_measures[0], 2.0 );
  EXPECT_EQ( host_measures[1], 6.0 );
  EXPECT_EQ( host_measures[2], 4.0 );

  auto host = copyQuadratureToHost( retained );
  const auto actual = host.getView<2>();
  ASSERT_EQ( actual.numberOfPairs(), 3 );
  EXPECT_EQ( actual.numberOfQuadraturePoints(), 12 );
  const InterfacePair expected_pairs[] = { { 0, 1 }, { 1, 0 }, { 1, 1 } };
  const IndexT expected_regions[] = { 1, 3, 2 };
  for ( IndexT pair = 0; pair < actual.numberOfPairs(); ++pair ) {
    const IndexT region_count = expected_regions[pair];
    EXPECT_EQ( actual.getInterfacePair( pair ).element_id1_, expected_pairs[pair].element_id1_ );
    EXPECT_EQ( actual.getInterfacePair( pair ).element_id2_, expected_pairs[pair].element_id2_ );
    ASSERT_EQ( actual.numberOfQuadraturePoints( pair ), region_count * VariableSurfaceRule::points_per_base_rule );
    for ( IndexT point = 0; point < actual.numberOfQuadraturePoints( pair ); ++point ) {
      const auto quadrature_point = actual.getQuadraturePoint( pair, point );
      EXPECT_EQ( quadrature_point.getWeight(), 1.0 );
      EXPECT_EQ( quadrature_point.getMesh1ParentCoordinates()[0], static_cast<RealT>( point ) );
      EXPECT_EQ( quadrature_point.getMesh1ParentCoordinates()[1], static_cast<RealT>( region_count ) );
      EXPECT_EQ( quadrature_point.getMesh2ParentCoordinates()[0], -static_cast<RealT>( point ) );
      EXPECT_EQ( quadrature_point.getMesh2ParentCoordinates()[1], -static_cast<RealT>( region_count ) );
    }
  }
  const auto offsets = host.getPairOffsets();
  EXPECT_EQ( offsets[0], 0 );
  EXPECT_EQ( offsets[1], 2 );
  EXPECT_EQ( offsets[2], 8 );
  EXPECT_EQ( offsets[3], 12 );
}

// Insufficient capacity reports the first affected pair after each submitted pair runs exactly once.
TEST_P( QuadratureBatchExecutionTest, reports_insufficient_capacity_without_regenerating_pairs )
{
  const auto pairs = copyToExecutionSpace<InterfacePair>( { { 0, 0 }, { 1, 0 } } );
  const auto regions = copyToExecutionSpace<IndexT>( { 1, 3 } );
  ArrayT<IndexT> calls( pairs.size(), pairs.size(), allocatorId() );
  calls.fill( 0 );
  const VariableSurfaceRule quadrature_rule{};
  const auto region_values = regions.view();
  auto call_values = calls.view();
  const auto generate = [=] TRIBOL_HOST_DEVICE( IndexT pair, PairQuadratureView<RealT, 2> output ) {
    ++call_values[pair];
    return generateSurfacePair( quadrature_rule, region_values[pair], output );
  };
  try {
    detail::generateQuadrature( GetParam(), quadrature_rule, pairs.view(), allocatorId(), generate );
    FAIL() << "Insufficient point capacity must report an error.";
  } catch ( const std::length_error& error ) {
    const std::string message = error.what();
    EXPECT_NE( message.find( "pair 1 (elements 1, 0)" ), std::string::npos );
    const IndexT required_points = 3 * VariableSurfaceRule::points_per_base_rule;
    EXPECT_NE( message.find( "requires " + std::to_string( required_points ) + " points" ), std::string::npos );
    EXPECT_NE( message.find( "capacity " + std::to_string( quadrature_rule.pointCapacity() ) ), std::string::npos );
  }
  const ArrayT<IndexT, 1, MemorySpace::Host> host_calls( calls );
  EXPECT_EQ( host_calls[0], 1 );
  EXPECT_EQ( host_calls[1], 1 );
}

// Batch generation must use the configured interval selector and updated coordinates without changing earlier data.
TEST( QuadratureBatch, uses_configuration_and_current_coordinates )
{
  using QuadratureRule = EdgeQuadratureRule<3>;
  constexpr double smoothing_fraction = 0.2;
  constexpr RealT edge_length = 4.0;
  const QuadratureRule quadrature_rule{};
  const SmoothProjectedEdgeInterval select_interval{ smoothing_fraction };
  const IndexT connectivity[] = { 0, 1 };
  RealT x1[] = { 0.0, edge_length }, y1[] = { 0.0, 0.0 };
  RealT x2[] = { edge_length, 0.0 }, y2[] = { 0.2, 0.2 };
  MeshData mesh1( 0, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );
  MeshData mesh2( 1, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );
  const ArrayT<InterfacePair> pairs( { { 0, 0 } }, getResourceAllocatorID( MemorySpace::Host ) );
#ifdef TRIBOL_USE_UMPIRE
  const ArrayT<InterfacePair> mismatched_pairs( { { 0, 0 } }, axom::MALLOC_ALLOCATOR_ID );
  EXPECT_THROW( generatePairQuadrature( ExecutionMode::Sequential, quadrature_rule, mismatched_pairs, mesh1.getView(),
                                        mesh2.getView(), select_interval ),
                std::invalid_argument );
#endif
  auto first = generatePairQuadrature<ExecutionMode::Sequential>( quadrature_rule, pairs, mesh1.getView(),
                                                                  mesh2.getView(), select_interval );
  x1[1] *= 2.0;
  x2[0] *= 2.0;
  auto second = generatePairQuadrature<ExecutionMode::Sequential>( quadrature_rule, pairs, mesh1.getView(),
                                                                   mesh2.getView(), select_interval );
  const auto measure = []( const InterfaceQuadratureData& data ) {
    const auto view = data.getView<1>();
    double result = 0.0;
    for ( IndexT q = 0; q < view.numberOfQuadraturePoints( 0 ); ++q )
      result += view.getQuadraturePoint( 0, q ).getWeight();
    return result;
  };
  const double expected_length = edge_length * ( 1.0 - smoothing_fraction / 2.0 );
  EXPECT_NEAR( measure( first ), expected_length, tolerance );
  EXPECT_NEAR( measure( second ), 2.0 * expected_length, tolerance );
}

// Empty input calls no generator; invalid calculations and inconsistent counts report host exceptions.
TEST_P( QuadratureBatchExecutionTest, handles_empty_input_and_rejects_failed_generation )
{
  const VariableSurfaceRule quadrature_rule{};
  const ArrayT<InterfacePair> empty( 0, 0, allocatorId() );
  ArrayT<IndexT> calls( 1, 1, allocatorId() );
  calls.fill( 0 );
  auto call_values = calls.view();
  auto rule = detail::generateQuadrature( GetParam(), quadrature_rule, empty.view(), allocatorId(),
                                          [=] TRIBOL_HOST_DEVICE( IndexT, PairQuadratureView<RealT, 2> ) {
                                            ++call_values[0];
                                            return QuadratureResult{};
                                          } );
  const ArrayT<IndexT, 1, MemorySpace::Host> host_calls( calls );
  EXPECT_EQ( host_calls[0], 0 );
  EXPECT_EQ( rule.getAllocatorId(), allocatorId() );
  EXPECT_EQ( rule.getView().numberOfPairs(), 0 );
  const ArrayT<IndexT, 1, MemorySpace::Host> empty_offsets( rule.getPairOffsets() );
  EXPECT_EQ( empty_offsets[0], 0 );
  const auto pairs = copyToExecutionSpace<InterfacePair>( { { 0, 0 } } );
  const auto invalid = [] TRIBOL_HOST_DEVICE( IndexT, PairQuadratureView<RealT, 2> ) {
    return QuadratureResult{ Status::InvalidGeometry, 0 };
  };
  EXPECT_THROW( detail::generateQuadrature( GetParam(), quadrature_rule, pairs.view(), allocatorId(), invalid ),
                std::invalid_argument );
  const auto inconsistent = [] TRIBOL_HOST_DEVICE( IndexT, PairQuadratureView<RealT, 2> output ) {
    return QuadratureResult{ Status::Success, output.capacity() + 1 };
  };
  EXPECT_THROW( detail::generateQuadrature( GetParam(), quadrature_rule, pairs.view(), allocatorId(), inconsistent ),
                std::runtime_error );
}

// Generate concrete edge quadrature from device-resident mesh data, then consume its typed view on the device.
TEST_P( QuadratureBatchExecutionTest, generates_projected_edge_quadrature_in_execution_space )
{
  using QuadratureRule = EdgeQuadratureRule<3>;
  constexpr RealT edge_length = 4.0;
  constexpr RealT smoothing_fraction = 0.2;
  const auto connectivity = copyToExecutionSpace<IndexT>( { 0, 1 } );
  const auto x1 = copyToExecutionSpace<RealT>( { 0.0, edge_length } );
  const auto y1 = copyToExecutionSpace<RealT>( { 0.0, 0.0 } );
  const auto x2 = copyToExecutionSpace<RealT>( { edge_length, 0.0 } );
  const auto y2 = copyToExecutionSpace<RealT>( { 0.2, 0.2 } );
  MeshData mesh1( 0, 1, 2, connectivity.data(), EDGE, LINEAR_LAGRANGE, x1.data(), y1.data(), nullptr, memorySpace() );
  MeshData mesh2( 1, 1, 2, connectivity.data(), EDGE, LINEAR_LAGRANGE, x2.data(), y2.data(), nullptr, memorySpace() );
  const auto pairs = copyToExecutionSpace<InterfacePair>( { { 0, 0 } } );
  EXPECT_TRUE( executionModeUsesAllocator( GetParam(), allocatorId() ) );
#ifdef TRIBOL_USE_UMPIRE
  if ( isOnDevice( GetParam() ) ) {
    EXPECT_FALSE( executionModeUsesAllocator( GetParam(), getResourceAllocatorID( MemorySpace::Host ) ) );
  }
#endif
  auto rule = generatePairQuadrature( GetParam(), QuadratureRule{}, pairs, mesh1.getView(), mesh2.getView(),
                                      SmoothProjectedEdgeInterval{ smoothing_fraction } );
  EXPECT_EQ( rule.getAllocatorId(), allocatorId() );
  const auto view = rule.getView<1>();
  ArrayT<RealT> measure_data( 1, 1, allocatorId() );
  auto measure = measure_data.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    RealT value = 0.0;
    for ( IndexT point = 0; point < view.numberOfQuadraturePoints( 0 ); ++point ) {
      value += view.getQuadraturePoint( 0, point ).getWeight();
    }
    measure[0] = value;
  } );
  const ArrayT<RealT, 1, MemorySpace::Host> host_measure( measure_data );
  EXPECT_NEAR( host_measure[0], edge_length * ( 1.0 - smoothing_fraction / 2.0 ), tolerance );
  auto host = copyQuadratureToHost( rule );
  EXPECT_EQ( host.getView().numberOfPairs(), 1 );
  EXPECT_EQ( host.getView().numberOfQuadraturePoints( 0 ), QuadratureRule::points_per_base_rule );
}

INSTANTIATE_TEST_SUITE_P( execution, QuadratureBatchExecutionTest, detail::enabledExecutionModes() );

TEST( QuadratureBatch, rejects_unrepresentable_sizes )
{
  EXPECT_THROW( detail::checkedQuadratureProduct( std::numeric_limits<IndexT>::max(), 2 ), std::length_error );
  EXPECT_THROW( detail::checkedQuadratureSum( std::numeric_limits<IndexT>::max(), 1 ), std::length_error );
}

// The factory's registered rules must cover supported degrees and invoke the construction callback exactly once.
TEST( QuadratureRuleFactory, selects_registered_rule_by_polynomial_order )
{
  int calls = 0;
  const auto create = [&]<typename QuadratureRule>() {
    ++calls;
    return QuadratureRule::points_per_base_rule;
  };
  const std::pair<int, int> orders_and_points[] = { { 0, 1 }, { 1, 1 }, { 2, 2 }, { 3, 2 }, { 4, 3 }, { 5, 3 } };
  for ( const auto [order, expected_points] : orders_and_points ) {
    const int before = calls;
    const auto selected = createWithQuadratureRule( order, create );
    EXPECT_EQ( selected, expected_points );
    EXPECT_EQ( calls, before + 1 );
  }
  for ( const int invalid : { -1, 6 } ) {
    const int before = calls;
    EXPECT_THROW( createWithQuadratureRule( invalid, create ), std::invalid_argument );
    EXPECT_EQ( calls, before );
  }
}

// These test-only rule descriptions exercise selection without implementing triangle quadrature generation.
template <int Degree, int Points>
struct SurfaceQuadratureRuleMetadata {
  static constexpr int polynomial_order = Degree;
  static constexpr int points_per_base_rule = Points;
};

// Surface rules can have different degree-to-point mappings; selection must use polynomial degree, not point count.
TEST( QuadratureRuleFactory, supports_surface_rule_families )
{
  using LinearTriangle = SurfaceQuadratureRuleMetadata<1, 1>;
  using QuadraticTriangle = SurfaceQuadratureRuleMetadata<2, 3>;
  using FourthDegreeTriangle = SurfaceQuadratureRuleMetadata<4, 6>;
  const auto create = []<typename QuadratureRule>() { return QuadratureRule::points_per_base_rule; };
  const std::pair<int, int> orders_and_points[] = { { 0, 1 }, { 1, 1 }, { 2, 3 }, { 3, 6 }, { 4, 6 } };
  for ( const auto [order, expected_points] : orders_and_points ) {
    const auto selected =
        createWithQuadratureRule<LinearTriangle, QuadraticTriangle, FourthDegreeTriangle>( order, create );
    EXPECT_EQ( selected, expected_points );
  }
}

}  // namespace

int main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
