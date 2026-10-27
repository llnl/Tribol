// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Test quadrature storage with runtime dimensions, typed coordinate views, and basis integration.
 *
 * Basis evaluation through a typed view runs with each enabled execution backend, using device arrays on GPUs.
 */

#include <cmath>
#include <type_traits>
#include <utility>

#include "ExecutionTest.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/integ/InterfaceQuadrature.hpp"

namespace {

using tribol::InterfaceQuadratureData;
using tribol::RealT;

const int host_allocator = tribol::getResourceAllocatorID( tribol::MemorySpace::Host );

InterfaceQuadratureData makeRule( int dimension )
{
  InterfaceQuadratureData rule( dimension, 3, 3, host_allocator );
  for ( int i = 0; i < 3; ++i ) {
    rule.getInterfacePairs()[i] = { i + 4, i + 8 };
    rule.getWeights()[i] = i + 0.5;
    for ( int d = 0; d < dimension; ++d ) {
      rule.getMesh1ParentCoordinates()( i, d ) = 0.1 * ( i + d );
      rule.getMesh2ParentCoordinates()( i, d ) = -0.2 * ( i + d );
    }
  }
  auto offsets = rule.getPairOffsets();
  offsets[0] = 0;
  offsets[1] = 2;
  offsets[2] = 2;  // A pair may have no quadrature points.
  offsets[3] = 3;
  return rule;
}

TEST( InterfaceQuadratureData, runtime_dimensions_and_variable_point_ranges )
{
  for ( int dim : { 1, 2 } ) {
    auto rule = makeRule( dim );
    const auto view = rule.getView();
    EXPECT_EQ( rule.getParentDimension(), dim );
    EXPECT_EQ( rule.getAllocatorId(), host_allocator );
    EXPECT_EQ( view.getParentDimension(), dim );
    EXPECT_EQ( view.numberOfPairs(), 3 );
    EXPECT_EQ( view.numberOfQuadraturePoints(), 3 );
    EXPECT_EQ( view.numberOfQuadraturePoints( 0 ), 2 );
    EXPECT_EQ( view.numberOfQuadraturePoints( 1 ), 0 );
    EXPECT_EQ( view.numberOfQuadraturePoints( 2 ), 1 );
    EXPECT_EQ( view.getInterfacePair( 2 ).element_id1_, 6 );
    const auto point = view.getQuadraturePoint( 2, 0 );
    EXPECT_EQ( point.getMesh1ParentCoordinates().size(), dim );
    EXPECT_EQ( point.getMesh2ParentCoordinates()[dim - 1], -0.2 * ( dim + 1 ) );
    EXPECT_EQ( point.getWeight(), 2.5 );
  }
}

TEST( InterfaceQuadratureData, fixed_extent_views_feed_element_bases )
{
  using EdgeView = InterfaceQuadratureData::Viewer<1>;
  using EdgePoint = decltype( std::declval<EdgeView>().getQuadraturePoint( 0, 0 ) );
  static_assert( std::is_trivially_copyable_v<EdgeView> );
  static_assert( std::is_same_v<decltype( std::declval<EdgePoint>().getMesh1ParentCoordinates() ),
                                tribol::MdSpan<const RealT, 1>> );
  static_assert( !std::is_copy_constructible_v<InterfaceQuadratureData> );
  static_assert( std::is_move_constructible_v<InterfaceQuadratureData> );
  using Basis = tribol::BasisFunction<tribol::QUAD, tribol::QUADRATIC_LAGRANGE>;
  auto rule = makeRule( 2 );
  const auto point = rule.getView<2>().getQuadraturePoint( 0, 1 );
  RealT values[Basis::node_count];
  Basis::Eval( point.getMesh1ParentCoordinates(), tribol::MdSpan<RealT, Basis::node_count>{ values } );
  RealT sum = 0;
  for ( RealT value : values ) sum += value;
  EXPECT_NEAR( sum, 1.0, 1.e-12 );
}

TEST( InterfaceQuadratureData, empty_and_moved_rules_keep_ownership )
{
  for ( int dim : { 1, 2 } ) {
    InterfaceQuadratureData empty( dim, host_allocator );
    EXPECT_EQ( empty.getParentDimension(), dim );
    EXPECT_EQ( empty.getPairOffsets().size(), 1 );
    EXPECT_EQ( empty.getPairOffsets()[0], 0 );
    EXPECT_EQ( empty.getView().numberOfQuadraturePoints(), 0 );
    auto moved = makeRule( dim );
    auto old_view = moved.getView();
    InterfaceQuadratureData destination( std::move( moved ) );
    EXPECT_EQ( old_view.getQuadraturePoint( 0, 1 ).getWeight(), 1.5 );
    EXPECT_EQ( destination.getView().getQuadraturePoint( 0, 1 ).getWeight(), 1.5 );
    empty = std::move( destination );
    EXPECT_EQ( empty.getView().numberOfPairs(), 3 );
  }
}

TEST( InterfaceQuadratureData, adopts_completed_arrays )
{
  auto original = makeRule( 2 );
  tribol::ArrayT<tribol::InterfacePair> pairs( original.getInterfacePairs() );
  tribol::ArrayT<tribol::IndexT> offsets( original.getPairOffsets() );
  tribol::ArrayT<RealT, 2> mesh1( original.getMesh1ParentCoordinates() );
  tribol::ArrayT<RealT, 2> mesh2( original.getMesh2ParentCoordinates() );
  tribol::ArrayT<RealT> weights( original.getWeights() );
  const auto* storage = mesh1.data();
  InterfaceQuadratureData adopted( 2, std::move( pairs ), std::move( offsets ), std::move( mesh1 ), std::move( mesh2 ),
                                   std::move( weights ) );
  EXPECT_EQ( adopted.getMesh1ParentCoordinates().data(), storage );
  EXPECT_EQ( adopted.getView<2>().getQuadraturePoint( 2, 0 ).getWeight(), 2.5 );
  EXPECT_EQ( adopted.getView().numberOfQuadraturePoints( 1 ), 0 );
}

TEST( InterfaceQuadratureData, checks_dimension_shapes_and_offsets )
{
  EXPECT_DEBUG_DEATH( { InterfaceQuadratureData invalid( 3, host_allocator ); }, "dimension" );
  auto rule = makeRule( 1 );
  EXPECT_DEBUG_DEATH( { rule.getView<2>(); }, "ParentDimension" );
#ifndef NDEBUG
  rule.getPairOffsets()[2] = 4;
  EXPECT_DEATH( { rule.getView(); }, "offsets" );
  rule.getPairOffsets()[2] = 2;
  rule.getPairOffsets()[3] = 2;
  EXPECT_DEATH( { rule.getView(); }, "offsets" );
#endif
}

TEST( InterfaceQuadratureData, integrates_quadratic_edge_basis_with_physical_weights )
{
  using Basis = tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  InterfaceQuadratureData rule( 1, 1, 3, host_allocator );
  rule.getInterfacePairs()[0] = { 3, 9 };
  rule.getPairOffsets()[0] = 0;
  rule.getPairOffsets()[1] = 3;
  const RealT a = std::sqrt( 3.0 / 5.0 );
  const RealT coords[] = { -a, 0, a };
  const RealT weights[] = { 10.0 / 9, 16.0 / 9, 10.0 / 9 };  // Physical edge length 4.
  for ( int q = 0; q < 3; ++q ) {
    rule.getMesh1ParentCoordinates()( q, 0 ) = coords[q];
    rule.getMesh2ParentCoordinates()( q, 0 ) = coords[q];
    rule.getWeights()[q] = weights[q];
  }
  RealT integrals[3] = {};
  const auto view = rule.getView<1>();
  for ( int q = 0; q < 3; ++q ) {
    const auto point = view.getQuadraturePoint( 0, q );
    RealT values[3];
    Basis::Eval( point.getMesh1ParentCoordinates(), tribol::MdSpan<RealT, 3>{ values } );
    for ( int i = 0; i < 3; ++i ) integrals[i] += 0.5 * values[i] * point.getWeight();
  }
  EXPECT_NEAR( integrals[0], 1.0 / 3, 1.e-12 );
  EXPECT_NEAR( integrals[1], 1.0 / 3, 1.e-12 );
  EXPECT_NEAR( integrals[2], 4.0 / 3, 1.e-12 );
}

class InterfaceQuadratureExecutionTest : public tribol::detail::ExecutionTest {};

// Build scan offsets in the same execution space that will consume them during device packing.
TEST_P( InterfaceQuadratureExecutionTest, scans_offsets_in_execution_space )
{
  const auto counts = copyToExecutionSpace<tribol::IndexT>( { 2, 0, 3, 1 } );
  tribol::ArrayT<tribol::IndexT> offsets( counts.size(), counts.size(), allocatorId() );
  tribol::exclusiveScanExec( GetParam(), counts.view(), offsets.view() );
  const tribol::ArrayT<tribol::IndexT, 1, tribol::MemorySpace::Host> host_offsets( offsets );
  EXPECT_EQ( host_offsets[0], 0 );
  EXPECT_EQ( host_offsets[1], 2 );
  EXPECT_EQ( host_offsets[2], 2 );
  EXPECT_EQ( host_offsets[3], 5 );

  auto inplace_offsets = copyToExecutionSpace<tribol::IndexT>( { 2, 0, 3, 1 } );
  tribol::exclusiveScanInplaceExec( GetParam(), inplace_offsets.view() );
  const tribol::ArrayT<tribol::IndexT, 1, tribol::MemorySpace::Host> host_inplace_offsets( inplace_offsets );
  EXPECT_EQ( host_inplace_offsets[0], 0 );
  EXPECT_EQ( host_inplace_offsets[1], 2 );
  EXPECT_EQ( host_inplace_offsets[2], 2 );
  EXPECT_EQ( host_inplace_offsets[3], 5 );
}

// Initialize owned storage in its execution space, move it, and consume the typed view in another kernel.
TEST_P( InterfaceQuadratureExecutionTest, initializes_and_consumes_owned_storage_in_execution_space )
{
  constexpr tribol::IndexT pair_count = 2;
  constexpr tribol::IndexT point_count = 3;
  InterfaceQuadratureData rule( 1, pair_count, point_count, allocatorId() );
  auto pairs = rule.getInterfacePairs();
  auto offsets = rule.getPairOffsets();
  auto mesh1 = rule.getMesh1ParentCoordinates();
  auto mesh2 = rule.getMesh2ParentCoordinates();
  auto weights = rule.getWeights();
  tribol::forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( tribol::IndexT ) {
    pairs[0] = { 4, 8 };
    pairs[1] = { 5, 9 };
    offsets[0] = 0;
    offsets[1] = 2;
    offsets[2] = point_count;
  } );
  tribol::forAllExec( GetParam(), point_count, [=] TRIBOL_HOST_DEVICE( tribol::IndexT point ) {
    mesh1( point, 0 ) = static_cast<RealT>( point );
    mesh2( point, 0 ) = -static_cast<RealT>( point );
    weights[point] = static_cast<RealT>( point + 1 );
  } );

  const auto* storage = rule.getMesh1ParentCoordinates().data();
  InterfaceQuadratureData moved( std::move( rule ) );
  EXPECT_EQ( moved.getAllocatorId(), allocatorId() );
  EXPECT_EQ( moved.getMesh1ParentCoordinates().data(), storage );
  const auto view = moved.getView<1>();
  tribol::ArrayT<RealT> values( point_count, point_count, allocatorId() );
  auto output = values.view();
  tribol::forAllExec( GetParam(), point_count, [=] TRIBOL_HOST_DEVICE( tribol::IndexT point ) {
    const tribol::IndexT pair = point < 2 ? 0 : 1;
    const tribol::IndexT local_point = point < 2 ? point : 0;
    const auto quadrature_point = view.getQuadraturePoint( pair, local_point );
    output[point] = quadrature_point.getMesh1ParentCoordinates()[0] - quadrature_point.getMesh2ParentCoordinates()[0] +
                    quadrature_point.getWeight();
  } );
  const tribol::ArrayT<RealT, 1, tribol::MemorySpace::Host> host_values( values );
  EXPECT_EQ( host_values[0], 1.0 );
  EXPECT_EQ( host_values[1], 4.0 );
  EXPECT_EQ( host_values[2], 7.0 );

  InterfaceQuadratureData empty( 1, allocatorId() );
  EXPECT_EQ( empty.getAllocatorId(), allocatorId() );
  EXPECT_EQ( empty.getView().numberOfPairs(), 0 );
  const tribol::ArrayT<tribol::IndexT, 1, tribol::MemorySpace::Host> empty_offsets( empty.getPairOffsets() );
  EXPECT_EQ( empty_offsets[0], 0 );
}

// Evaluate the quadratic edge basis at its midpoint through a typed view and read the stored physical weight.
TEST_P( InterfaceQuadratureExecutionTest, evaluates_basis_from_view )
{
  using Basis = tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  constexpr RealT physical_weight = 1.25;
  constexpr tribol::IndexT midpoint_node = Basis::node_count - 1;
  constexpr tribol::IndexT weight_result_index = Basis::node_count;
  constexpr tribol::IndexT result_count = Basis::node_count + 1;
  constexpr double comparison_tolerance = 1.0e-12;
  auto host = makeRule( Basis::parent_dimension );
  host.getMesh1ParentCoordinates()( 1, 0 ) = 0.0;
  host.getWeights()[1] = physical_weight;
  tribol::ArrayT<tribol::InterfacePair> pairs( host.getInterfacePairs(), allocatorId() );
  tribol::ArrayT<tribol::IndexT> offsets( host.getPairOffsets(), allocatorId() );
  tribol::ArrayT<RealT, 2> mesh1( host.getMesh1ParentCoordinates(), allocatorId() );
  tribol::ArrayT<RealT, 2> mesh2( host.getMesh2ParentCoordinates(), allocatorId() );
  tribol::ArrayT<RealT> weights( host.getWeights(), allocatorId() );
  InterfaceQuadratureData rule( Basis::parent_dimension, std::move( pairs ), std::move( offsets ), std::move( mesh1 ),
                                std::move( mesh2 ), std::move( weights ) );
  const auto view = rule.getView<Basis::parent_dimension>();
  tribol::ArrayT<RealT> output( result_count, result_count, allocatorId() );
  auto values = output.view();
  tribol::forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( tribol::IndexT ) {
    const auto point = view.getQuadraturePoint( 0, 1 );
    Basis::Eval( point.getMesh1ParentCoordinates(),
                 tribol::MdSpan<RealT, Basis::node_count>{ values.data(), Basis::node_count } );
    values[weight_result_index] = point.getWeight();
  } );
  const tribol::ArrayT<RealT, 1, tribol::MemorySpace::Host> result( output );
  EXPECT_NEAR( result[0], 0.0, comparison_tolerance );
  EXPECT_NEAR( result[1], 0.0, comparison_tolerance );
  EXPECT_NEAR( result[midpoint_node], 1.0, comparison_tolerance );
  EXPECT_NEAR( result[weight_result_index], physical_weight, comparison_tolerance );
}

INSTANTIATE_TEST_SUITE_P( execution, InterfaceQuadratureExecutionTest, tribol::detail::enabledExecutionModes() );

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
