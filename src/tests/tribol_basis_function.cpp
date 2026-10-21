// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <array>
#include <cmath>
#include <type_traits>

#include <gtest/gtest.h>
#include <mfem.hpp>

#include "axom/slic/core/SimpleLogger.hpp"

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/BasisFunction.hpp"

namespace {

constexpr tribol::RealT comparison_tolerance = 1.0e-12;
constexpr tribol::RealT finite_difference_step = 1.0e-7;
constexpr tribol::RealT finite_difference_tolerance = 1.0e-8;

// TODO: Replace this test adapter when MeshData::Viewer can expose face nodal fields through connectivity without
// copying the values into contiguous element storage.
template <tribol::IndexT ComponentCount>
struct NodalValuesThroughConnectivity {
  const tribol::RealT* component_values[ComponentCount];
  const tribol::IndexT* connectivity;

  tribol::RealT operator()( tribol::IndexT local_node, tribol::IndexT component ) const
  {
    return component_values[component][connectivity[local_node]];
  }
};

template <typename Basis>
void ExpectNodalInterpolation()
{
  for ( int evaluation_node = 0; evaluation_node < Basis::node_count; ++evaluation_node ) {
    const auto parent_coordinates = Basis::GetParentNodeCoordinates( evaluation_node );
    std::array<tribol::RealT, Basis::node_count> basis_values{};
    Basis::Eval( parent_coordinates.view(), basis_values );

    for ( int basis_node = 0; basis_node < Basis::node_count; ++basis_node ) {
      EXPECT_NEAR( basis_values[basis_node], basis_node == evaluation_node ? 1.0 : 0.0, comparison_tolerance );
    }
  }
}

template <typename Basis>
void ExpectPartitionOfUnity( const std::array<tribol::RealT, Basis::parent_dimension>& parent_coordinates )
{
  std::array<tribol::RealT, Basis::node_count> basis_values{};
  std::array<tribol::RealT, Basis::node_count * Basis::parent_dimension> parent_derivatives{};
  Basis::Eval( parent_coordinates, basis_values );
  Basis::EvalParentDerivatives( parent_coordinates, parent_derivatives );

  tribol::RealT basis_sum{};
  for ( auto value : basis_values ) {
    basis_sum += value;
  }
  EXPECT_NEAR( basis_sum, 1.0, comparison_tolerance );

  for ( int parent_component = 0; parent_component < Basis::parent_dimension; ++parent_component ) {
    tribol::RealT derivative_sum{};
    for ( int node = 0; node < Basis::node_count; ++node ) {
      derivative_sum += parent_derivatives[Basis::parent_dimension * node + parent_component];
    }
    EXPECT_NEAR( derivative_sum, 0.0, comparison_tolerance );
  }
}

template <typename Basis>
void ExpectParentDerivativesMatchFiniteDifferences(
    const std::array<tribol::RealT, Basis::parent_dimension>& parent_coordinates )
{
  std::array<tribol::RealT, Basis::node_count * Basis::parent_dimension> parent_derivatives{};
  Basis::EvalParentDerivatives( parent_coordinates, parent_derivatives );

  for ( int parent_component = 0; parent_component < Basis::parent_dimension; ++parent_component ) {
    auto lower_coordinates = parent_coordinates;
    auto upper_coordinates = parent_coordinates;
    lower_coordinates[parent_component] -= finite_difference_step;
    upper_coordinates[parent_component] += finite_difference_step;

    std::array<tribol::RealT, Basis::node_count> lower_values{};
    std::array<tribol::RealT, Basis::node_count> upper_values{};
    Basis::Eval( lower_coordinates, lower_values );
    Basis::Eval( upper_coordinates, upper_values );
    for ( int node = 0; node < Basis::node_count; ++node ) {
      const tribol::RealT finite_difference =
          ( upper_values[node] - lower_values[node] ) / ( 2.0 * finite_difference_step );
      EXPECT_NEAR( parent_derivatives[Basis::parent_dimension * node + parent_component], finite_difference,
                   finite_difference_tolerance );
    }
  }
}

template <typename Basis, typename MfemElement>
void ExpectBasisMatchesMfem( const std::array<tribol::RealT, Basis::parent_dimension>& parent_coordinates,
                             MfemElement& mfem_element )
{
  std::array<tribol::RealT, Basis::node_count> basis_values{};
  std::array<tribol::RealT, Basis::node_count * Basis::parent_dimension> parent_derivatives{};
  Basis::Eval( parent_coordinates, basis_values );
  Basis::EvalParentDerivatives( parent_coordinates, parent_derivatives );

  mfem::IntegrationPoint mfem_point;
  mfem_point.x = 0.5 * ( parent_coordinates[0] + 1.0 );
  if constexpr ( Basis::parent_dimension == 2 ) {
    mfem_point.y = 0.5 * ( parent_coordinates[1] + 1.0 );
  }
  mfem::Vector mfem_values( Basis::node_count );
  mfem::DenseMatrix mfem_derivatives( Basis::node_count, Basis::parent_dimension );
  mfem_element.CalcShape( mfem_point, mfem_values );
  mfem_element.CalcDShape( mfem_point, mfem_derivatives );

  for ( int node = 0; node < Basis::node_count; ++node ) {
    EXPECT_NEAR( basis_values[node], mfem_values[node], comparison_tolerance );
    for ( int parent_component = 0; parent_component < Basis::parent_dimension; ++parent_component ) {
      // MFEM differentiates on [0, 1]. Tribol's derivative gains a factor of one half on [-1, 1].
      EXPECT_NEAR( parent_derivatives[Basis::parent_dimension * node + parent_component],
                   0.5 * mfem_derivatives( node, parent_component ), comparison_tolerance );
    }
  }
}

TEST( BasisFunction, face_element_traits_define_connectivity_order )
{
  using LinearEdge = tribol::FaceElementTraits<tribol::EDGE, tribol::LINEAR_LAGRANGE>;
  using QuadraticEdge = tribol::FaceElementTraits<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  using LinearTriangle = tribol::FaceElementTraits<tribol::TRIANGLE, tribol::LINEAR_LAGRANGE>;
  using LinearQuad = tribol::FaceElementTraits<tribol::QUAD, tribol::LINEAR_LAGRANGE>;
  using QuadraticQuad = tribol::FaceElementTraits<tribol::QUAD, tribol::QUADRATIC_LAGRANGE>;

  static_assert( LinearEdge::node_count == 2 );
  static_assert( LinearEdge::parent_dimension == 1 );
  static_assert( LinearEdge::spatial_dimension == 2 );
  static_assert( std::is_same_v<LinearEdge::ParentCoordinates, tribol::MdArray<tribol::RealT, 1>> );
  static_assert( QuadraticEdge::node_count == 3 );
  static_assert( QuadraticEdge::parent_dimension == 1 );
  static_assert( QuadraticEdge::spatial_dimension == 2 );
  static_assert( LinearTriangle::node_count == 3 );
  static_assert( LinearTriangle::parent_dimension == 2 );
  static_assert( LinearTriangle::spatial_dimension == 3 );
  static_assert( LinearQuad::node_count == 4 );
  static_assert( LinearQuad::parent_dimension == 2 );
  static_assert( LinearQuad::spatial_dimension == 3 );
  static_assert( QuadraticQuad::node_count == 9 );
  static_assert( QuadraticQuad::parent_dimension == 2 );
  static_assert( QuadraticQuad::spatial_dimension == 3 );
  static_assert( std::is_same_v<QuadraticQuad::ParentCoordinates, tribol::MdArray<tribol::RealT, 2>> );

  constexpr std::array<tribol::RealT, LinearEdge::node_count> expected_linear_edge_nodes{ -1.0, 1.0 };
  for ( int node = 0; node < LinearEdge::node_count; ++node ) {
    const auto parent_coordinates = LinearEdge::GetParentNodeCoordinates( node );
    EXPECT_EQ( parent_coordinates[0], expected_linear_edge_nodes[node] );
  }

  constexpr std::array<tribol::RealT, QuadraticEdge::node_count> expected_edge_nodes{ -1.0, 1.0, 0.0 };
  for ( int node = 0; node < QuadraticEdge::node_count; ++node ) {
    const auto parent_coordinates = QuadraticEdge::GetParentNodeCoordinates( node );
    EXPECT_EQ( parent_coordinates[0], expected_edge_nodes[node] );
  }

  constexpr std::array<std::array<tribol::RealT, 2>, LinearTriangle::node_count> expected_triangle_nodes{
      { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 } } };
  for ( int node = 0; node < LinearTriangle::node_count; ++node ) {
    const auto parent_coordinates = LinearTriangle::GetParentNodeCoordinates( node );
    for ( int parent_component = 0; parent_component < LinearTriangle::parent_dimension; ++parent_component ) {
      EXPECT_EQ( parent_coordinates[parent_component], expected_triangle_nodes[node][parent_component] );
    }
  }

  constexpr std::array<std::array<tribol::RealT, 2>, LinearQuad::node_count> expected_linear_quad_nodes{
      { { -1.0, -1.0 }, { 1.0, -1.0 }, { 1.0, 1.0 }, { -1.0, 1.0 } } };
  for ( int node = 0; node < LinearQuad::node_count; ++node ) {
    const auto parent_coordinates = LinearQuad::GetParentNodeCoordinates( node );
    for ( int parent_component = 0; parent_component < LinearQuad::parent_dimension; ++parent_component ) {
      EXPECT_EQ( parent_coordinates[parent_component], expected_linear_quad_nodes[node][parent_component] );
    }
  }

  constexpr std::array<std::array<tribol::RealT, 2>, QuadraticQuad::node_count> expected_quad_nodes{ { { -1.0, -1.0 },
                                                                                                       { 1.0, -1.0 },
                                                                                                       { 1.0, 1.0 },
                                                                                                       { -1.0, 1.0 },
                                                                                                       { 0.0, -1.0 },
                                                                                                       { 1.0, 0.0 },
                                                                                                       { 0.0, 1.0 },
                                                                                                       { -1.0, 0.0 },
                                                                                                       { 0.0, 0.0 } } };
  for ( int node = 0; node < QuadraticQuad::node_count; ++node ) {
    const auto parent_coordinates = QuadraticQuad::GetParentNodeCoordinates( node );
    for ( int parent_component = 0; parent_component < QuadraticQuad::parent_dimension; ++parent_component ) {
      EXPECT_EQ( parent_coordinates[parent_component], expected_quad_nodes[node][parent_component] );
    }
  }

  // The output overload remains available when a caller needs to reuse existing storage.
  tribol::RealT reused_parent_coordinates[LinearTriangle::parent_dimension];
  LinearTriangle::GetParentNodeCoordinates( 2, reused_parent_coordinates );
  EXPECT_EQ( reused_parent_coordinates[0], 0.0 );
  EXPECT_EQ( reused_parent_coordinates[1], 1.0 );

  EXPECT_EQ( tribol::GetFaceNodeCount( tribol::EDGE, tribol::LINEAR_LAGRANGE ), 2 );
  EXPECT_EQ( tribol::GetFaceNodeCount( tribol::EDGE, tribol::QUADRATIC_LAGRANGE ), 3 );
  EXPECT_EQ( tribol::GetFaceNodeCount( tribol::TRIANGLE, tribol::LINEAR_LAGRANGE ), 3 );
  EXPECT_EQ( tribol::GetFaceNodeCount( tribol::QUAD, tribol::LINEAR_LAGRANGE ), 4 );
  EXPECT_EQ( tribol::GetFaceNodeCount( tribol::QUAD, tribol::QUADRATIC_LAGRANGE ), 9 );
  EXPECT_EQ( tribol::GetFaceNodeCount( tribol::TRIANGLE, tribol::QUADRATIC_LAGRANGE ), 0 );
}

TEST( BasisFunction, interpolates_nodes_and_partitions_unity )
{
  using LinearEdge = tribol::BasisFunction<tribol::EDGE, tribol::LINEAR_LAGRANGE>;
  using QuadraticEdge = tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  using LinearTriangle = tribol::BasisFunction<tribol::TRIANGLE, tribol::LINEAR_LAGRANGE>;
  using LinearQuad = tribol::BasisFunction<tribol::QUAD, tribol::LINEAR_LAGRANGE>;
  using QuadraticQuad = tribol::BasisFunction<tribol::QUAD, tribol::QUADRATIC_LAGRANGE>;

  ExpectNodalInterpolation<LinearEdge>();
  ExpectNodalInterpolation<QuadraticEdge>();
  ExpectNodalInterpolation<LinearTriangle>();
  ExpectNodalInterpolation<LinearQuad>();
  ExpectNodalInterpolation<QuadraticQuad>();

  ExpectPartitionOfUnity<LinearEdge>( { 0.27 } );
  ExpectPartitionOfUnity<QuadraticEdge>( { -0.31 } );
  ExpectPartitionOfUnity<LinearTriangle>( { 0.21, 0.37 } );
  ExpectPartitionOfUnity<LinearQuad>( { 0.19, -0.43 } );
  ExpectPartitionOfUnity<QuadraticQuad>( { -0.23, 0.41 } );
}

TEST( BasisFunction, reproduces_quadratic_polynomials )
{
  using QuadraticEdge = tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  constexpr tribol::RealT edge_parent_coordinate = 0.37;
  std::array<tribol::RealT, QuadraticEdge::node_count> edge_values{};
  std::array<tribol::RealT, QuadraticEdge::node_count> edge_derivatives{};
  QuadraticEdge::Eval( edge_parent_coordinate, edge_values );
  QuadraticEdge::EvalParentDerivatives( edge_parent_coordinate, edge_derivatives );

  tribol::RealT interpolated_edge_value{};
  tribol::RealT interpolated_edge_derivative{};
  for ( int node = 0; node < QuadraticEdge::node_count; ++node ) {
    const auto node_coordinates = QuadraticEdge::GetParentNodeCoordinates( node );
    const tribol::RealT node_coordinate = node_coordinates[0];
    const tribol::RealT nodal_value = 2.0 + 3.0 * node_coordinate + 4.0 * node_coordinate * node_coordinate;
    interpolated_edge_value += edge_values[node] * nodal_value;
    interpolated_edge_derivative += edge_derivatives[node] * nodal_value;
  }
  EXPECT_NEAR( interpolated_edge_value,
               2.0 + 3.0 * edge_parent_coordinate + 4.0 * edge_parent_coordinate * edge_parent_coordinate,
               comparison_tolerance );
  EXPECT_NEAR( interpolated_edge_derivative, 3.0 + 8.0 * edge_parent_coordinate, comparison_tolerance );

  using QuadraticQuad = tribol::BasisFunction<tribol::QUAD, tribol::QUADRATIC_LAGRANGE>;
  constexpr std::array<tribol::RealT, QuadraticQuad::parent_dimension> quad_parent_coordinates{ 0.17, -0.29 };
  std::array<tribol::RealT, QuadraticQuad::node_count> quad_values{};
  std::array<tribol::RealT, QuadraticQuad::node_count * QuadraticQuad::parent_dimension> quad_derivatives{};
  QuadraticQuad::Eval( quad_parent_coordinates, quad_values );
  QuadraticQuad::EvalParentDerivatives( quad_parent_coordinates, quad_derivatives );

  auto polynomial = []( tribol::RealT x, tribol::RealT y ) {
    return 1.0 + 2.0 * x - 3.0 * y + 4.0 * x * x + 5.0 * x * y + 6.0 * y * y + 7.0 * x * x * y + 8.0 * x * y * y +
           9.0 * x * x * y * y;
  };
  tribol::RealT interpolated_quad_value{};
  tribol::RealT interpolated_quad_dx{};
  tribol::RealT interpolated_quad_dy{};
  for ( int node = 0; node < QuadraticQuad::node_count; ++node ) {
    const auto node_coordinates = QuadraticQuad::GetParentNodeCoordinates( node );
    const tribol::RealT nodal_value = polynomial( node_coordinates[0], node_coordinates[1] );
    interpolated_quad_value += quad_values[node] * nodal_value;
    interpolated_quad_dx += quad_derivatives[QuadraticQuad::parent_dimension * node] * nodal_value;
    interpolated_quad_dy += quad_derivatives[QuadraticQuad::parent_dimension * node + 1] * nodal_value;
  }
  const tribol::RealT x = quad_parent_coordinates[0];
  const tribol::RealT y = quad_parent_coordinates[1];
  EXPECT_NEAR( interpolated_quad_value, polynomial( x, y ), comparison_tolerance );
  EXPECT_NEAR( interpolated_quad_dx, 2.0 + 8.0 * x + 5.0 * y + 14.0 * x * y + 8.0 * y * y + 18.0 * x * y * y,
               comparison_tolerance );
  EXPECT_NEAR( interpolated_quad_dy, -3.0 + 5.0 * x + 12.0 * y + 7.0 * x * x + 16.0 * x * y + 18.0 * x * x * y,
               comparison_tolerance );
}

TEST( BasisFunction, parent_derivatives_match_finite_differences )
{
  ExpectParentDerivativesMatchFiniteDifferences<tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>>(
      { 0.23 } );
  ExpectParentDerivativesMatchFiniteDifferences<tribol::BasisFunction<tribol::TRIANGLE, tribol::LINEAR_LAGRANGE>>(
      { 0.17, 0.31 } );
  ExpectParentDerivativesMatchFiniteDifferences<tribol::BasisFunction<tribol::QUAD, tribol::QUADRATIC_LAGRANGE>>(
      { -0.37, 0.28 } );
}

TEST( BasisFunction, interpolates_nodal_values_through_connectivity )
{
  using Basis = tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  constexpr tribol::IndexT field_component_count = 2;
  constexpr tribol::IndexT global_node_count = 5;
  constexpr std::array<tribol::IndexT, Basis::node_count> connectivity{ 3, 1, 4 };

  std::array<tribol::RealT, global_node_count> first_component{};
  std::array<tribol::RealT, global_node_count> second_component{};
  for ( int local_node = 0; local_node < Basis::node_count; ++local_node ) {
    const auto node_coordinates = Basis::GetParentNodeCoordinates( local_node );
    const tribol::RealT node_coordinate = node_coordinates[0];
    const tribol::IndexT global_node = connectivity[local_node];
    first_component[global_node] = 2.0 + 3.0 * node_coordinate + 4.0 * node_coordinate * node_coordinate;
    second_component[global_node] = -1.0 + 2.0 * node_coordinate - 3.0 * node_coordinate * node_coordinate;
  }
  const NodalValuesThroughConnectivity<field_component_count> nodal_values{
      { first_component.data(), second_component.data() }, connectivity.data() };

  constexpr tribol::RealT parent_coordinate = 0.25;
  tribol::MdArray<tribol::RealT, Basis::node_count> basis_values;
  Basis::Eval( parent_coordinate, basis_values.view() );
  tribol::MdArray<tribol::RealT, field_component_count> interpolated_values;
  tribol::InterpolateNodalValues<Basis, field_component_count>( basis_values.view(), nodal_values,
                                                                interpolated_values.view() );

  EXPECT_NEAR( interpolated_values[0], 2.0 + 3.0 * parent_coordinate + 4.0 * parent_coordinate * parent_coordinate,
               comparison_tolerance );
  EXPECT_NEAR( interpolated_values[1], -1.0 + 2.0 * parent_coordinate - 3.0 * parent_coordinate * parent_coordinate,
               comparison_tolerance );
}

TEST( BasisFunction, maps_affine_linear_elements )
{
  using EdgeBasis = tribol::BasisFunction<tribol::EDGE, tribol::LINEAR_LAGRANGE>;
  constexpr tribol::RealT edge_parent_coordinate = 0.25;
  constexpr std::array<tribol::RealT, EdgeBasis::node_count * EdgeBasis::spatial_dimension> edge_coordinates{
      1.0, 2.0, 5.0, 4.0 };
  std::array<tribol::RealT, EdgeBasis::node_count> edge_values{};
  std::array<tribol::RealT, EdgeBasis::node_count * EdgeBasis::parent_dimension> edge_parent_derivatives{};
  std::array<tribol::RealT, EdgeBasis::spatial_dimension> edge_point{};
  std::array<tribol::RealT, EdgeBasis::spatial_dimension * EdgeBasis::parent_dimension> edge_jacobian{};
  EdgeBasis::Eval( edge_parent_coordinate, edge_values );
  EdgeBasis::EvalParentDerivatives( edge_parent_coordinate, edge_parent_derivatives );
  tribol::MapToPhysical<EdgeBasis>( edge_values, edge_coordinates, edge_point );
  tribol::ComputeMappingJacobian<EdgeBasis>( edge_parent_derivatives, edge_coordinates, edge_jacobian );

  EXPECT_NEAR( edge_point[0], 3.5, comparison_tolerance );
  EXPECT_NEAR( edge_point[1], 3.25, comparison_tolerance );
  EXPECT_NEAR( edge_jacobian[0], 2.0, comparison_tolerance );
  EXPECT_NEAR( edge_jacobian[1], 1.0, comparison_tolerance );
  EXPECT_NEAR( tribol::ComputeJacobianWeight<EdgeBasis>( edge_jacobian ), std::sqrt( 5.0 ), comparison_tolerance );
  EXPECT_NEAR( tribol::ComputeJacobianWeight<EdgeBasis>( edge_parent_coordinate, edge_coordinates ), std::sqrt( 5.0 ),
               comparison_tolerance );

  using QuadBasis = tribol::BasisFunction<tribol::QUAD, tribol::LINEAR_LAGRANGE>;
  constexpr std::array<tribol::RealT, QuadBasis::parent_dimension> quad_parent_coordinates{ 0.2, -0.4 };
  std::array<tribol::RealT, QuadBasis::node_count * QuadBasis::spatial_dimension> quad_coordinates{};
  for ( int node = 0; node < QuadBasis::node_count; ++node ) {
    const auto node_coordinates = QuadBasis::GetParentNodeCoordinates( node );
    quad_coordinates[QuadBasis::spatial_dimension * node] = 1.0 + 2.0 * node_coordinates[0];
    quad_coordinates[QuadBasis::spatial_dimension * node + 1] = -2.0 + 3.0 * node_coordinates[1];
    quad_coordinates[QuadBasis::spatial_dimension * node + 2] = 4.0 + 0.5 * node_coordinates[0] - node_coordinates[1];
  }
  std::array<tribol::RealT, QuadBasis::node_count> quad_values{};
  std::array<tribol::RealT, QuadBasis::node_count * QuadBasis::parent_dimension> quad_parent_derivatives{};
  std::array<tribol::RealT, QuadBasis::spatial_dimension> quad_point{};
  std::array<tribol::RealT, QuadBasis::spatial_dimension * QuadBasis::parent_dimension> quad_jacobian{};
  QuadBasis::Eval( quad_parent_coordinates, quad_values );
  QuadBasis::EvalParentDerivatives( quad_parent_coordinates, quad_parent_derivatives );
  tribol::MapToPhysical<QuadBasis>( quad_values, quad_coordinates, quad_point );
  tribol::ComputeMappingJacobian<QuadBasis>( quad_parent_derivatives, quad_coordinates, quad_jacobian );

  constexpr std::array<tribol::RealT, QuadBasis::spatial_dimension> expected_quad_point{ 1.4, -3.2, 4.5 };
  constexpr std::array<tribol::RealT, QuadBasis::spatial_dimension * QuadBasis::parent_dimension>
      expected_quad_jacobian{ 2.0, 0.0, 0.0, 3.0, 0.5, -1.0 };
  for ( int component = 0; component < QuadBasis::spatial_dimension; ++component ) {
    EXPECT_NEAR( quad_point[component], expected_quad_point[component], comparison_tolerance );
  }
  for ( int entry = 0; entry < static_cast<int>( quad_jacobian.size() ); ++entry ) {
    EXPECT_NEAR( quad_jacobian[entry], expected_quad_jacobian[entry], comparison_tolerance );
  }
  EXPECT_NEAR( tribol::ComputeJacobianWeight<QuadBasis>( quad_jacobian ), 6.5, comparison_tolerance );
  EXPECT_NEAR( tribol::ComputeJacobianWeight<QuadBasis>( quad_parent_coordinates, quad_coordinates ), 6.5,
               comparison_tolerance );
}

TEST( BasisFunction, maps_curved_quadratic_edge )
{
  using Basis = tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  constexpr tribol::RealT parent_coordinate = 0.25;
  constexpr std::array<tribol::RealT, Basis::node_count * Basis::spatial_dimension> coordinates{ 0.0, 0.0, 2.0,
                                                                                                 0.0, 1.0, 1.0 };
  std::array<tribol::RealT, Basis::node_count> values{};
  std::array<tribol::RealT, Basis::node_count * Basis::parent_dimension> parent_derivatives{};
  std::array<tribol::RealT, Basis::spatial_dimension> physical_point{};
  std::array<tribol::RealT, Basis::spatial_dimension * Basis::parent_dimension> jacobian{};
  std::array<tribol::RealT, Basis::node_count * Basis::spatial_dimension> physical_derivatives{};

  Basis::Eval( parent_coordinate, values );
  Basis::EvalParentDerivatives( parent_coordinate, parent_derivatives );
  tribol::MapToPhysical<Basis>( values, coordinates, physical_point );
  tribol::ComputeMappingJacobian<Basis>( parent_derivatives, coordinates, jacobian );

  EXPECT_NEAR( physical_point[0], 1.0 + parent_coordinate, comparison_tolerance );
  EXPECT_NEAR( physical_point[1], 1.0 - parent_coordinate * parent_coordinate, comparison_tolerance );
  EXPECT_NEAR( jacobian[0], 1.0, comparison_tolerance );
  EXPECT_NEAR( jacobian[1], -2.0 * parent_coordinate, comparison_tolerance );
  EXPECT_NEAR( tribol::ComputeJacobianWeight<Basis>( jacobian ), std::sqrt( 1.25 ), comparison_tolerance );
  EXPECT_NEAR( tribol::ComputeJacobianWeight<Basis>( parent_coordinate, coordinates ), std::sqrt( 1.25 ),
               comparison_tolerance );
  ASSERT_TRUE( tribol::ComputePhysicalDerivatives<Basis>( jacobian, parent_derivatives, physical_derivatives ) );

  for ( int node = 0; node < Basis::node_count; ++node ) {
    tribol::RealT recovered_parent_derivative{};
    for ( int physical_component = 0; physical_component < Basis::spatial_dimension; ++physical_component ) {
      recovered_parent_derivative +=
          jacobian[physical_component] * physical_derivatives[Basis::spatial_dimension * node + physical_component];
    }
    EXPECT_NEAR( recovered_parent_derivative, parent_derivatives[node], comparison_tolerance );
  }
}

TEST( BasisFunction, maps_curved_quadratic_quad )
{
  using Basis = tribol::BasisFunction<tribol::QUAD, tribol::QUADRATIC_LAGRANGE>;
  constexpr std::array<tribol::RealT, Basis::parent_dimension> parent_coordinates{ 0.2, -0.3 };
  std::array<tribol::RealT, Basis::node_count * Basis::spatial_dimension> coordinates{};
  for ( int node = 0; node < Basis::node_count; ++node ) {
    const auto node_coordinates = Basis::GetParentNodeCoordinates( node );
    coordinates[Basis::spatial_dimension * node] = 1.0 + node_coordinates[0];
    coordinates[Basis::spatial_dimension * node + 1] = 2.0 + node_coordinates[1];
    coordinates[Basis::spatial_dimension * node + 2] = 0.25 * ( 1.0 - node_coordinates[0] * node_coordinates[0] ) *
                                                       ( 1.0 - node_coordinates[1] * node_coordinates[1] );
  }

  std::array<tribol::RealT, Basis::node_count> values{};
  std::array<tribol::RealT, Basis::node_count * Basis::parent_dimension> parent_derivatives{};
  std::array<tribol::RealT, Basis::spatial_dimension> physical_point{};
  std::array<tribol::RealT, Basis::spatial_dimension * Basis::parent_dimension> jacobian{};
  std::array<tribol::RealT, Basis::node_count * Basis::spatial_dimension> physical_derivatives{};
  Basis::Eval( parent_coordinates, values );
  Basis::EvalParentDerivatives( parent_coordinates, parent_derivatives );
  tribol::MapToPhysical<Basis>( values, coordinates, physical_point );
  tribol::ComputeMappingJacobian<Basis>( parent_derivatives, coordinates, jacobian );

  EXPECT_NEAR( physical_point[0], 1.2, comparison_tolerance );
  EXPECT_NEAR( physical_point[1], 1.7, comparison_tolerance );
  EXPECT_NEAR( physical_point[2], 0.2184, comparison_tolerance );
  constexpr std::array<tribol::RealT, Basis::spatial_dimension * Basis::parent_dimension> expected_jacobian{
      1.0, 0.0, 0.0, 1.0, -0.091, 0.144 };
  for ( int entry = 0; entry < static_cast<int>( jacobian.size() ); ++entry ) {
    EXPECT_NEAR( jacobian[entry], expected_jacobian[entry], comparison_tolerance );
  }
  EXPECT_NEAR( tribol::ComputeJacobianWeight<Basis>( jacobian ), std::sqrt( 1.0 + 0.091 * 0.091 + 0.144 * 0.144 ),
               comparison_tolerance );
  ASSERT_TRUE( tribol::ComputePhysicalDerivatives<Basis>( jacobian, parent_derivatives, physical_derivatives ) );

  for ( int node = 0; node < Basis::node_count; ++node ) {
    for ( int parent_component = 0; parent_component < Basis::parent_dimension; ++parent_component ) {
      tribol::RealT recovered_parent_derivative{};
      for ( int physical_component = 0; physical_component < Basis::spatial_dimension; ++physical_component ) {
        recovered_parent_derivative += jacobian[Basis::parent_dimension * physical_component + parent_component] *
                                       physical_derivatives[Basis::spatial_dimension * node + physical_component];
      }
      EXPECT_NEAR( recovered_parent_derivative, parent_derivatives[Basis::parent_dimension * node + parent_component],
                   comparison_tolerance );
    }
  }
}

TEST( BasisFunction, rejects_rank_deficient_mapping )
{
  using EdgeBasis = tribol::BasisFunction<tribol::EDGE, tribol::LINEAR_LAGRANGE>;
  constexpr std::array<tribol::RealT, EdgeBasis::node_count * EdgeBasis::spatial_dimension> edge_coordinates{
      1.0, 2.0, 1.0, 2.0 };
  constexpr tribol::RealT parent_coordinate = 0.0;
  std::array<tribol::RealT, EdgeBasis::node_count * EdgeBasis::parent_dimension> edge_parent_derivatives{};
  std::array<tribol::RealT, EdgeBasis::spatial_dimension * EdgeBasis::parent_dimension> edge_jacobian{};
  std::array<tribol::RealT, EdgeBasis::node_count * EdgeBasis::spatial_dimension> edge_physical_derivatives{};
  EdgeBasis::EvalParentDerivatives( parent_coordinate, edge_parent_derivatives );
  tribol::ComputeMappingJacobian<EdgeBasis>( edge_parent_derivatives, edge_coordinates, edge_jacobian );

  EXPECT_FALSE( tribol::ComputePhysicalDerivatives<EdgeBasis>( edge_jacobian, edge_parent_derivatives,
                                                               edge_physical_derivatives ) );
  for ( auto derivative : edge_physical_derivatives ) {
    EXPECT_EQ( derivative, 0.0 );
  }

  using QuadBasis = tribol::BasisFunction<tribol::QUAD, tribol::LINEAR_LAGRANGE>;
  constexpr std::array<tribol::RealT, QuadBasis::parent_dimension> quad_parent_coordinates{ 0.0, 0.0 };
  constexpr std::array<tribol::RealT, QuadBasis::node_count * QuadBasis::spatial_dimension> quad_coordinates{
      -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0 };
  std::array<tribol::RealT, QuadBasis::node_count * QuadBasis::parent_dimension> quad_parent_derivatives{};
  std::array<tribol::RealT, QuadBasis::spatial_dimension * QuadBasis::parent_dimension> quad_jacobian{};
  std::array<tribol::RealT, QuadBasis::node_count * QuadBasis::spatial_dimension> quad_physical_derivatives{};
  QuadBasis::EvalParentDerivatives( quad_parent_coordinates, quad_parent_derivatives );
  tribol::ComputeMappingJacobian<QuadBasis>( quad_parent_derivatives, quad_coordinates, quad_jacobian );

  EXPECT_FALSE( tribol::ComputePhysicalDerivatives<QuadBasis>( quad_jacobian, quad_parent_derivatives,
                                                               quad_physical_derivatives ) );
  for ( auto derivative : quad_physical_derivatives ) {
    EXPECT_EQ( derivative, 0.0 );
  }
}

TEST( BasisFunction, matches_mfem_h1_basis )
{
  mfem::H1_SegmentElement linear_segment( 1 );
  mfem::H1_SegmentElement quadratic_segment( 2 );
  mfem::H1_QuadrilateralElement linear_quad( 1 );
  mfem::H1_QuadrilateralElement quadratic_quad( 2 );

  ExpectBasisMatchesMfem<tribol::BasisFunction<tribol::EDGE, tribol::LINEAR_LAGRANGE>>( { 0.23 }, linear_segment );
  ExpectBasisMatchesMfem<tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>>( { -0.31 },
                                                                                           quadratic_segment );
  ExpectBasisMatchesMfem<tribol::BasisFunction<tribol::QUAD, tribol::LINEAR_LAGRANGE>>( { 0.19, -0.43 }, linear_quad );
  ExpectBasisMatchesMfem<tribol::BasisFunction<tribol::QUAD, tribol::QUADRATIC_LAGRANGE>>( { -0.23, 0.41 },
                                                                                           quadratic_quad );
}

#if defined( TRIBOL_USE_CUDA ) || defined( TRIBOL_USE_HIP )
TEST( BasisFunction, evaluates_on_device )
{
  using Basis = tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  tribol::ArrayT<tribol::RealT, 1, tribol::MemorySpace::Device> device_values( Basis::node_count + 1 );
  auto values = device_values.view();

#if defined( TRIBOL_USE_CUDA )
  constexpr tribol::ExecutionMode device_execution_mode = tribol::ExecutionMode::Cuda;
#else
  constexpr tribol::ExecutionMode device_execution_mode = tribol::ExecutionMode::Hip;
#endif
  tribol::forAllExec<false>( device_execution_mode, 1, [=] TRIBOL_HOST_DEVICE( tribol::IndexT ) {
    const auto parent_coordinates = Basis::GetParentNodeCoordinates( 2 );
    Basis::Eval( parent_coordinates.view(),
                 tribol::MdSpan<tribol::RealT, Basis::node_count>{ values.data(), Basis::node_count } );

    tribol::MdArray<tribol::RealT, Basis::node_count, 1> nodal_values;
    for ( int node = 0; node < Basis::node_count; ++node ) {
      nodal_values( node, 0 ) = static_cast<tribol::RealT>( node + 1 );
    }
    tribol::MdArray<tribol::RealT, 1> interpolated_value;
    tribol::InterpolateNodalValues<Basis, 1>(
        tribol::MdSpan<const tribol::RealT, Basis::node_count>{ values.data(), Basis::node_count }, nodal_values.view(),
        interpolated_value.view() );
    values[Basis::node_count] = interpolated_value[0];
  } );

  tribol::ArrayT<tribol::RealT, 1, tribol::MemorySpace::Host> host_values( device_values );
  EXPECT_NEAR( host_values[0], 0.0, comparison_tolerance );
  EXPECT_NEAR( host_values[1], 0.0, comparison_tolerance );
  EXPECT_NEAR( host_values[2], 1.0, comparison_tolerance );
  EXPECT_NEAR( host_values[Basis::node_count], 3.0, comparison_tolerance );
}
#endif

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
