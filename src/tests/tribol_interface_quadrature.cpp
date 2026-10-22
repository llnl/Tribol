// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <cmath>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"

#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/integ/InterfaceQuadrature.hpp"

namespace {

constexpr tribol::RealT comparison_tolerance = 1.0e-12;

TEST( InterfaceQuadraturePoint, exposes_fixed_extent_parent_coordinates )
{
  // A point must expose the exact fixed extents consumed by each basis while keeping mutable and const access distinct.
  using EdgeBasis = tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  using FaceBasis = tribol::BasisFunction<tribol::QUAD, tribol::QUADRATIC_LAGRANGE>;
  using EdgePoint = tribol::InterfaceQuadraturePoint<EdgeBasis::parent_dimension>;
  using FacePoint = tribol::InterfaceQuadraturePoint<FaceBasis::parent_dimension>;
  static_assert( std::is_trivially_copyable_v<EdgePoint> );
  static_assert( std::is_trivially_copyable_v<FacePoint> );
  static_assert( sizeof( EdgePoint ) == 3 * sizeof( tribol::RealT ) );
  static_assert( sizeof( FacePoint ) == 5 * sizeof( tribol::RealT ) );
  static_assert( std::is_same_v<decltype( std::declval<EdgePoint&>().getMesh1ParentCoordinates() ),
                                tribol::MdSpan<tribol::RealT, EdgeBasis::parent_dimension>> );
  static_assert( std::is_same_v<decltype( std::declval<const EdgePoint&>().getMesh1ParentCoordinates() ),
                                tribol::MdSpan<const tribol::RealT, EdgeBasis::parent_dimension>> );
  static_assert( std::is_same_v<decltype( std::declval<FacePoint&>().getMesh2ParentCoordinates() ),
                                tribol::MdSpan<tribol::RealT, FaceBasis::parent_dimension>> );
  static_assert( std::is_same_v<decltype( std::declval<const FacePoint&>().getMesh2ParentCoordinates() ),
                                tribol::MdSpan<const tribol::RealT, FaceBasis::parent_dimension>> );

  EdgePoint edge_point{ EdgePoint::ParentCoordinates{ -0.25 }, EdgePoint::ParentCoordinates{ 0.5 }, 1.75 };
  edge_point.getMesh1ParentCoordinates()[0] = -0.5;
  const auto& read_only_edge_point = edge_point;
  EXPECT_EQ( read_only_edge_point.getMesh1ParentCoordinates()[0], -0.5 );
  EXPECT_EQ( read_only_edge_point.getMesh2ParentCoordinates()[0], 0.5 );
  EXPECT_EQ( read_only_edge_point.getWeight(), 1.75 );

  // A stored face coordinate must pass directly to a higher-order basis without an intermediate coordinate copy.
  FacePoint face_point{ FacePoint::ParentCoordinates{ 0.25, -0.5 }, FacePoint::ParentCoordinates{ -0.4, 0.2 }, 2.0 };
  tribol::MdArray<tribol::RealT, FaceBasis::node_count> basis_values;
  FaceBasis::Eval( face_point.getMesh1ParentCoordinates(), basis_values.view() );

  tribol::RealT basis_sum = 0.0;
  for ( int node = 0; node < FaceBasis::node_count; ++node ) {
    basis_sum += basis_values[node];
  }
  EXPECT_NEAR( basis_sum, 1.0, comparison_tolerance );
}

TEST( InterfaceQuadratureData, owns_pairs_and_variable_point_ranges )
{
  // CSR offsets let each interface pair own a different number of points without storing pair IDs on every point.
  using QuadratureData = tribol::InterfaceQuadratureData<1>;
  using QuadraturePoint = QuadratureData::QuadraturePoint;
  using QuadratureView = QuadratureData::Viewer;
  static_assert( std::is_same_v<decltype( std::declval<const QuadratureView&>().getInterfacePair( 0 ) ),
                                const tribol::InterfacePair&> );
  static_assert( std::is_same_v<decltype( std::declval<const QuadratureView&>().getQuadraturePoint( 0, 0 ) ),
                                const QuadraturePoint&> );
  const int host_allocator_id = tribol::getResourceAllocatorID( tribol::MemorySpace::Host );
  QuadratureData quadrature_data( 2, 3, host_allocator_id );

  auto pairs = quadrature_data.getInterfacePairs();
  pairs[0] = tribol::InterfacePair{ 4, 8 };
  pairs[1] = tribol::InterfacePair{ 7, 11 };

  auto pair_offsets = quadrature_data.getPairOffsets();
  pair_offsets[0] = 0;
  pair_offsets[1] = 2;
  pair_offsets[2] = 3;

  auto points = quadrature_data.getQuadraturePoints();
  points[0] =
      QuadraturePoint{ QuadraturePoint::ParentCoordinates{ -0.5 }, QuadraturePoint::ParentCoordinates{ -0.25 }, 0.6 };
  points[1] =
      QuadraturePoint{ QuadraturePoint::ParentCoordinates{ 0.5 }, QuadraturePoint::ParentCoordinates{ 0.25 }, 0.7 };
  points[2] =
      QuadraturePoint{ QuadraturePoint::ParentCoordinates{ 0.0 }, QuadraturePoint::ParentCoordinates{ 0.1 }, 1.2 };

  const auto view = quadrature_data.getView();
  EXPECT_EQ( view.numberOfPairs(), 2 );
  EXPECT_EQ( view.numberOfQuadraturePoints(), 3 );
  EXPECT_EQ( view.numberOfQuadraturePoints( 0 ), 2 );
  EXPECT_EQ( view.numberOfQuadraturePoints( 1 ), 1 );
  EXPECT_EQ( view.getInterfacePair( 0 ).m_element_id1, 4 );
  EXPECT_EQ( view.getInterfacePair( 1 ).m_element_id2, 11 );
  EXPECT_EQ( view.getQuadraturePoint( 0, 1 ).getMesh1ParentCoordinates()[0], 0.5 );
  EXPECT_EQ( view.getQuadraturePoint( 1, 0 ).getWeight(), 1.2 );

  // An empty rule retains the one terminal offset required by the same CSR contract used for populated rules.
  QuadratureData empty_quadrature_data( 0, 0, host_allocator_id );
  empty_quadrature_data.getPairOffsets()[0] = 0;
  const auto empty_view = empty_quadrature_data.getView();
  EXPECT_EQ( empty_view.numberOfPairs(), 0 );
  EXPECT_EQ( empty_view.numberOfQuadraturePoints(), 0 );
}

TEST( InterfaceQuadratureData, integrates_quadratic_edge_gap_like_data )
{
  // This reproduces the core data flow of an EnergyMortar edge integral without changing its differentiated kernels:
  // evaluate both bases from stored parent coordinates, map both points, and use the stored physical measure.
  using Basis = tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  using QuadratureData = tribol::InterfaceQuadratureData<Basis::parent_dimension>;
  using QuadraturePoint = QuadratureData::QuadraturePoint;
  using PhysicalPoint = tribol::MdArray<tribol::RealT, Basis::spatial_dimension>;
  using EdgeCoordinates = tribol::MdArray<tribol::RealT, Basis::node_count, Basis::spatial_dimension>;
  using CoordinateRow = tribol::MdArray<tribol::RealT, Basis::spatial_dimension>;

  constexpr tribol::IndexT integration_edge_id = 3;
  constexpr tribol::IndexT projection_edge_id = 9;
  constexpr tribol::RealT edge_gap = 0.5;
  constexpr tribol::RealT edge_length = 4.0;
  constexpr tribol::RealT endpoint_basis_integral = edge_gap * edge_length / 6.0;
  constexpr tribol::RealT midpoint_basis_integral = 2.0 * edge_gap * edge_length / 3.0;
  constexpr tribol::RealT total_gap_integral = edge_gap * edge_length;
  constexpr tribol::RealT physical_jacobian = 0.5 * edge_length;
  const tribol::RealT gauss_coordinate = std::sqrt( 3.0 / 5.0 );
  const tribol::RealT parent_coordinates[3] = { -gauss_coordinate, 0.0, gauss_coordinate };
  const tribol::RealT physical_weights[3] = { physical_jacobian * 5.0 / 9.0, physical_jacobian * 8.0 / 9.0,
                                              physical_jacobian * 5.0 / 9.0 };

  // Quadratic edge ordering is endpoint, endpoint, midpoint, matching FaceElementTraits and MeshData connectivity.
  const EdgeCoordinates mesh1_coordinates{ CoordinateRow{ 0.0, 0.0 }, CoordinateRow{ edge_length, 0.0 },
                                           CoordinateRow{ 0.5 * edge_length, 0.0 } };
  const EdgeCoordinates mesh2_coordinates{ CoordinateRow{ 0.0, edge_gap }, CoordinateRow{ edge_length, edge_gap },
                                           CoordinateRow{ 0.5 * edge_length, edge_gap } };

  const int host_allocator_id = tribol::getResourceAllocatorID( tribol::MemorySpace::Host );
  QuadratureData quadrature_data( 1, 3, host_allocator_id );
  quadrature_data.getInterfacePairs()[0] = tribol::InterfacePair{ integration_edge_id, projection_edge_id };
  auto pair_offsets = quadrature_data.getPairOffsets();
  pair_offsets[0] = 0;
  pair_offsets[1] = 3;
  auto points = quadrature_data.getQuadraturePoints();
  for ( int point = 0; point < 3; ++point ) {
    points[point] =
        QuadraturePoint{ QuadraturePoint::ParentCoordinates{ parent_coordinates[point] },
                         QuadraturePoint::ParentCoordinates{ parent_coordinates[point] }, physical_weights[point] };
  }

  tribol::RealT integrated_gap = 0.0;
  tribol::MdArray<tribol::RealT, Basis::node_count> nodal_gap_integrals;
  for ( int node = 0; node < Basis::node_count; ++node ) {
    nodal_gap_integrals[node] = 0.0;
  }

  const auto quadrature_view = quadrature_data.getView();
  for ( int point = 0; point < quadrature_view.numberOfQuadraturePoints( 0 ); ++point ) {
    const auto& quadrature_point = quadrature_view.getQuadraturePoint( 0, point );
    tribol::MdArray<tribol::RealT, Basis::node_count> mesh1_basis_values;
    tribol::MdArray<tribol::RealT, Basis::node_count> mesh2_basis_values;
    Basis::Eval( quadrature_point.getMesh1ParentCoordinates(), mesh1_basis_values.view() );
    Basis::Eval( quadrature_point.getMesh2ParentCoordinates(), mesh2_basis_values.view() );

    PhysicalPoint mesh1_point;
    PhysicalPoint mesh2_point;
    tribol::MapToPhysical<Basis>( mesh1_basis_values.view(), mesh1_coordinates.view(), mesh1_point.view() );
    tribol::MapToPhysical<Basis>( mesh2_basis_values.view(), mesh2_coordinates.view(), mesh2_point.view() );
    const tribol::RealT gap_like_value = mesh2_point[1] - mesh1_point[1];

    integrated_gap += gap_like_value * quadrature_point.getWeight();
    for ( int node = 0; node < Basis::node_count; ++node ) {
      nodal_gap_integrals[node] += mesh1_basis_values[node] * gap_like_value * quadrature_point.getWeight();
    }
  }

  EXPECT_NEAR( integrated_gap, total_gap_integral, comparison_tolerance );
  EXPECT_NEAR( nodal_gap_integrals[0], endpoint_basis_integral, comparison_tolerance );
  EXPECT_NEAR( nodal_gap_integrals[1], endpoint_basis_integral, comparison_tolerance );
  EXPECT_NEAR( nodal_gap_integrals[2], midpoint_basis_integral, comparison_tolerance );
}

#if defined( TRIBOL_USE_CUDA ) || defined( TRIBOL_USE_HIP )
TEST( InterfaceQuadratureData, evaluates_basis_from_device_view )
{
  // The rule view and each point's fixed-extent coordinate span must remain usable inside the configured device path.
  using Basis = tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  using QuadratureData = tribol::InterfaceQuadratureData<Basis::parent_dimension>;
  using QuadraturePoint = QuadratureData::QuadraturePoint;
  const int host_allocator_id = tribol::getResourceAllocatorID( tribol::MemorySpace::Host );
  const int device_allocator_id = tribol::getResourceAllocatorID( tribol::MemorySpace::Device );

  tribol::ArrayT<tribol::InterfacePair> host_pairs( { tribol::InterfacePair{ 1, 2 } }, host_allocator_id );
  tribol::ArrayT<tribol::IndexT> host_offsets( { 0, 1 }, host_allocator_id );
  tribol::ArrayT<QuadraturePoint> host_points(
      { QuadraturePoint{ QuadraturePoint::ParentCoordinates{ 0.0 }, QuadraturePoint::ParentCoordinates{ 0.5 }, 1.25 } },
      host_allocator_id );

  tribol::ArrayT<tribol::InterfacePair> device_pairs( host_pairs, device_allocator_id );
  tribol::ArrayT<tribol::IndexT> device_offsets( host_offsets, device_allocator_id );
  tribol::ArrayT<QuadraturePoint> device_points( host_points, device_allocator_id );
  QuadratureData quadrature_data( std::move( device_pairs ), std::move( device_offsets ), std::move( device_points ) );
  const auto quadrature_view = quadrature_data.getView();

  tribol::ArrayT<tribol::RealT> device_results( Basis::node_count + 1, Basis::node_count + 1, device_allocator_id );
  auto results = device_results.view();
#if defined( TRIBOL_USE_CUDA )
  constexpr tribol::ExecutionMode device_execution_mode = tribol::ExecutionMode::Cuda;
#else
  constexpr tribol::ExecutionMode device_execution_mode = tribol::ExecutionMode::Hip;
#endif
  tribol::forAllExec<false>( device_execution_mode, 1, [=] TRIBOL_HOST_DEVICE( tribol::IndexT ) {
    const auto& point = quadrature_view.getQuadraturePoint( 0, 0 );
    Basis::Eval( point.getMesh1ParentCoordinates(),
                 tribol::MdSpan<tribol::RealT, Basis::node_count>{ results.data(), Basis::node_count } );
    results[Basis::node_count] = point.getWeight();
  } );

  tribol::ArrayT<tribol::RealT, 1, tribol::MemorySpace::Host> host_results( device_results );
  EXPECT_NEAR( host_results[0], 0.0, comparison_tolerance );
  EXPECT_NEAR( host_results[1], 0.0, comparison_tolerance );
  EXPECT_NEAR( host_results[2], 1.0, comparison_tolerance );
  EXPECT_NEAR( host_results[Basis::node_count], 1.25, comparison_tolerance );
}
#endif

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
