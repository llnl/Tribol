// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check Lagrange basis values, parametric derivatives, physical mappings, and nodal interpolation.
 * Basis sizes and FE-node counts agree for the supported Lagrange elements. Interpolation checks separate the
 * number of basis functions from the number of scalar values stored at each node.
 */

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
template <tribol::IndexT NumValuesPerNode>
struct NodalValuesThroughConnectivity {
  const tribol::RealT* nodal_value_arrays[NumValuesPerNode];
  const tribol::IndexT* connectivity;

  tribol::RealT operator()( tribol::IndexT local_node, tribol::IndexT value_index ) const
  {
    return nodal_value_arrays[value_index][connectivity[local_node]];
  }
};

template <typename Basis>
void ExpectNodalInterpolation()
{
  for ( int evaluation_node = 0; evaluation_node < Basis::size_basis; ++evaluation_node ) {
    const auto parametric_coords = Basis::getParametricNodeCoords( evaluation_node );
    std::array<tribol::RealT, Basis::size_basis> basis_values{};
    Basis::eval( parametric_coords.view(), basis_values );

    for ( int basis_node = 0; basis_node < Basis::size_basis; ++basis_node ) {
      EXPECT_NEAR( basis_values[basis_node], basis_node == evaluation_node ? 1.0 : 0.0, comparison_tolerance );
    }
  }
}

template <typename Basis>
void ExpectPartitionOfUnity( const std::array<tribol::RealT, Basis::parametric_dim>& parametric_coords )
{
  std::array<tribol::RealT, Basis::size_basis> basis_values{};
  std::array<tribol::RealT, Basis::size_basis * Basis::parametric_dim> parametric_derivatives{};
  Basis::eval( parametric_coords, basis_values );
  Basis::evalParametricDerivatives( parametric_coords, parametric_derivatives );

  tribol::RealT basis_sum{};
  for ( auto value : basis_values ) {
    basis_sum += value;
  }
  EXPECT_NEAR( basis_sum, 1.0, comparison_tolerance );

  for ( int parametric_dim = 0; parametric_dim < Basis::parametric_dim; ++parametric_dim ) {
    tribol::RealT derivative_sum{};
    for ( int node = 0; node < Basis::size_basis; ++node ) {
      derivative_sum += parametric_derivatives[Basis::parametric_dim * node + parametric_dim];
    }
    EXPECT_NEAR( derivative_sum, 0.0, comparison_tolerance );
  }
}

template <typename Basis>
void ExpectParametricDerivativesMatchFiniteDifferences(
    const std::array<tribol::RealT, Basis::parametric_dim>& parametric_coords )
{
  std::array<tribol::RealT, Basis::size_basis * Basis::parametric_dim> parametric_derivatives{};
  Basis::evalParametricDerivatives( parametric_coords, parametric_derivatives );

  for ( int parametric_dim = 0; parametric_dim < Basis::parametric_dim; ++parametric_dim ) {
    auto lower_coordinates = parametric_coords;
    auto upper_coordinates = parametric_coords;
    lower_coordinates[parametric_dim] -= finite_difference_step;
    upper_coordinates[parametric_dim] += finite_difference_step;

    std::array<tribol::RealT, Basis::size_basis> lower_values{};
    std::array<tribol::RealT, Basis::size_basis> upper_values{};
    Basis::eval( lower_coordinates, lower_values );
    Basis::eval( upper_coordinates, upper_values );
    for ( int node = 0; node < Basis::size_basis; ++node ) {
      const tribol::RealT finite_difference =
          ( upper_values[node] - lower_values[node] ) / ( 2.0 * finite_difference_step );
      EXPECT_NEAR( parametric_derivatives[Basis::parametric_dim * node + parametric_dim], finite_difference,
                   finite_difference_tolerance );
    }
  }
}

template <typename Basis, typename MfemElement>
void ExpectBasisMatchesMfem( const std::array<tribol::RealT, Basis::parametric_dim>& parametric_coords,
                             MfemElement& mfem_element )
{
  std::array<tribol::RealT, Basis::size_basis> basis_values{};
  std::array<tribol::RealT, Basis::size_basis * Basis::parametric_dim> parametric_derivatives{};
  Basis::eval( parametric_coords, basis_values );
  Basis::evalParametricDerivatives( parametric_coords, parametric_derivatives );

  mfem::IntegrationPoint mfem_point;
  mfem_point.x = 0.5 * ( parametric_coords[0] + 1.0 );
  if constexpr ( Basis::parametric_dim == 2 ) {
    mfem_point.y = 0.5 * ( parametric_coords[1] + 1.0 );
  }
  mfem::Vector mfem_values( Basis::size_basis );
  mfem::DenseMatrix mfem_derivatives( Basis::size_basis, Basis::parametric_dim );
  mfem_element.CalcShape( mfem_point, mfem_values );
  mfem_element.CalcDShape( mfem_point, mfem_derivatives );

  for ( int node = 0; node < Basis::size_basis; ++node ) {
    EXPECT_NEAR( basis_values[node], mfem_values[node], comparison_tolerance );
    for ( int parametric_dim = 0; parametric_dim < Basis::parametric_dim; ++parametric_dim ) {
      // MFEM differentiates on [0, 1]. Tribol's derivative gains a factor of one half on [-1, 1].
      EXPECT_NEAR( parametric_derivatives[Basis::parametric_dim * node + parametric_dim],
                   0.5 * mfem_derivatives( node, parametric_dim ), comparison_tolerance );
    }
  }
}

TEST( BasisFunction, face_element_traits_define_connectivity_order )
{
  // FaceElementTraits is the shared contract between MeshData connectivity and basis evaluation, so each supported
  // shape and order must expose the dimensions, node count, and local parametric-node ordering expected by both.
  using LinearEdge = tribol::FaceElementTraits<tribol::EDGE, tribol::LINEAR_LAGRANGE>;
  using QuadraticEdge = tribol::FaceElementTraits<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  using LinearTriangle = tribol::FaceElementTraits<tribol::TRIANGLE, tribol::LINEAR_LAGRANGE>;
  using LinearQuad = tribol::FaceElementTraits<tribol::QUAD, tribol::LINEAR_LAGRANGE>;
  using QuadraticQuad = tribol::FaceElementTraits<tribol::QUAD, tribol::QUADRATIC_LAGRANGE>;

  // Compile-time dimensions let callers size stack storage without carrying runtime element metadata into kernels.
  static_assert( LinearEdge::num_nodes == 2 );
  static_assert( LinearEdge::parametric_dim == 1 );
  static_assert( LinearEdge::spatial_dim == 2 );
  static_assert( std::is_same_v<LinearEdge::ParametricCoords, tribol::MdArray<tribol::RealT, 1>> );
  static_assert( QuadraticEdge::num_nodes == 3 );
  static_assert( QuadraticEdge::parametric_dim == 1 );
  static_assert( QuadraticEdge::spatial_dim == 2 );
  static_assert( LinearTriangle::num_nodes == 3 );
  static_assert( LinearTriangle::parametric_dim == 2 );
  static_assert( LinearTriangle::spatial_dim == 3 );
  static_assert( LinearQuad::num_nodes == 4 );
  static_assert( LinearQuad::parametric_dim == 2 );
  static_assert( LinearQuad::spatial_dim == 3 );
  static_assert( QuadraticQuad::num_nodes == 9 );
  static_assert( QuadraticQuad::parametric_dim == 2 );
  static_assert( QuadraticQuad::spatial_dim == 3 );
  static_assert( std::is_same_v<QuadraticQuad::ParametricCoords, tribol::MdArray<tribol::RealT, 2>> );

  // These parametric coordinates make the local node index concrete and guard the MFEM-compatible connectivity
  // ordering.
  constexpr std::array<tribol::RealT, LinearEdge::num_nodes> expected_linear_edge_nodes{ -1.0, 1.0 };
  for ( int node = 0; node < LinearEdge::num_nodes; ++node ) {
    const auto parametric_coords = LinearEdge::getParametricNodeCoords( node );
    EXPECT_EQ( parametric_coords[0], expected_linear_edge_nodes[node] );
  }

  constexpr std::array<tribol::RealT, QuadraticEdge::num_nodes> expected_edge_nodes{ -1.0, 1.0, 0.0 };
  for ( int node = 0; node < QuadraticEdge::num_nodes; ++node ) {
    const auto parametric_coords = QuadraticEdge::getParametricNodeCoords( node );
    EXPECT_EQ( parametric_coords[0], expected_edge_nodes[node] );
  }

  constexpr std::array<std::array<tribol::RealT, 2>, LinearTriangle::num_nodes> expected_triangle_nodes{
      { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 } } };
  for ( int node = 0; node < LinearTriangle::num_nodes; ++node ) {
    const auto parametric_coords = LinearTriangle::getParametricNodeCoords( node );
    for ( int parametric_dim = 0; parametric_dim < LinearTriangle::parametric_dim; ++parametric_dim ) {
      EXPECT_EQ( parametric_coords[parametric_dim], expected_triangle_nodes[node][parametric_dim] );
    }
  }

  constexpr std::array<std::array<tribol::RealT, 2>, LinearQuad::num_nodes> expected_linear_quad_nodes{
      { { -1.0, -1.0 }, { 1.0, -1.0 }, { 1.0, 1.0 }, { -1.0, 1.0 } } };
  for ( int node = 0; node < LinearQuad::num_nodes; ++node ) {
    const auto parametric_coords = LinearQuad::getParametricNodeCoords( node );
    for ( int parametric_dim = 0; parametric_dim < LinearQuad::parametric_dim; ++parametric_dim ) {
      EXPECT_EQ( parametric_coords[parametric_dim], expected_linear_quad_nodes[node][parametric_dim] );
    }
  }

  constexpr std::array<std::array<tribol::RealT, 2>, QuadraticQuad::num_nodes> expected_quad_nodes{ { { -1.0, -1.0 },
                                                                                                      { 1.0, -1.0 },
                                                                                                      { 1.0, 1.0 },
                                                                                                      { -1.0, 1.0 },
                                                                                                      { 0.0, -1.0 },
                                                                                                      { 1.0, 0.0 },
                                                                                                      { 0.0, 1.0 },
                                                                                                      { -1.0, 0.0 },
                                                                                                      { 0.0, 0.0 } } };
  for ( int node = 0; node < QuadraticQuad::num_nodes; ++node ) {
    const auto parametric_coords = QuadraticQuad::getParametricNodeCoords( node );
    for ( int parametric_dim = 0; parametric_dim < QuadraticQuad::parametric_dim; ++parametric_dim ) {
      EXPECT_EQ( parametric_coords[parametric_dim], expected_quad_nodes[node][parametric_dim] );
    }
  }

  // The output overload remains available when a caller needs to reuse existing storage.
  tribol::RealT reused_parametric_coords[LinearTriangle::parametric_dim];
  LinearTriangle::getParametricNodeCoords( 2, reused_parametric_coords );
  EXPECT_EQ( reused_parametric_coords[0], 0.0 );
  EXPECT_EQ( reused_parametric_coords[1], 1.0 );

  // MeshData uses the runtime dispatcher, so it must agree with the compile-time traits and reject unsupported pairs.
  EXPECT_EQ( tribol::numberOfFaceNodes( tribol::EDGE, tribol::LINEAR_LAGRANGE ), 2 );
  EXPECT_EQ( tribol::numberOfFaceNodes( tribol::EDGE, tribol::QUADRATIC_LAGRANGE ), 3 );
  EXPECT_EQ( tribol::numberOfFaceNodes( tribol::TRIANGLE, tribol::LINEAR_LAGRANGE ), 3 );
  EXPECT_EQ( tribol::numberOfFaceNodes( tribol::QUAD, tribol::LINEAR_LAGRANGE ), 4 );
  EXPECT_EQ( tribol::numberOfFaceNodes( tribol::QUAD, tribol::QUADRATIC_LAGRANGE ), 9 );
  EXPECT_EQ( tribol::numberOfFaceNodes( tribol::TRIANGLE, tribol::QUADRATIC_LAGRANGE ), 0 );
}

TEST( BasisFunction, interpolates_nodes_and_partitions_unity )
{
  // Every supported basis must preserve nodal values exactly and preserve constant fields throughout its parametric
  // element; the derivative sums must therefore vanish as well.
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
  // Quadratic Lagrange bases must exactly reproduce every polynomial in their approximation space, including the
  // parametric derivatives needed by mapping and contact calculations.
  using QuadraticEdge = tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  constexpr tribol::RealT edge_parametric_coord = 0.37;
  std::array<tribol::RealT, QuadraticEdge::size_basis> edge_values{};
  std::array<tribol::RealT, QuadraticEdge::size_basis> edge_derivatives{};
  QuadraticEdge::eval( edge_parametric_coord, edge_values );
  QuadraticEdge::evalParametricDerivatives( edge_parametric_coord, edge_derivatives );

  tribol::RealT interpolated_edge_value{};
  tribol::RealT interpolated_edge_derivative{};
  for ( int node = 0; node < QuadraticEdge::size_basis; ++node ) {
    const auto node_coordinates = QuadraticEdge::getParametricNodeCoords( node );
    const tribol::RealT node_coordinate = node_coordinates[0];
    const tribol::RealT nodal_value = 2.0 + 3.0 * node_coordinate + 4.0 * node_coordinate * node_coordinate;
    interpolated_edge_value += edge_values[node] * nodal_value;
    interpolated_edge_derivative += edge_derivatives[node] * nodal_value;
  }
  EXPECT_NEAR( interpolated_edge_value,
               2.0 + 3.0 * edge_parametric_coord + 4.0 * edge_parametric_coord * edge_parametric_coord,
               comparison_tolerance );
  EXPECT_NEAR( interpolated_edge_derivative, 3.0 + 8.0 * edge_parametric_coord, comparison_tolerance );

  using QuadraticQuad = tribol::BasisFunction<tribol::QUAD, tribol::QUADRATIC_LAGRANGE>;
  constexpr std::array<tribol::RealT, QuadraticQuad::parametric_dim> quad_parametric_coords{ 0.17, -0.29 };
  std::array<tribol::RealT, QuadraticQuad::size_basis> quad_values{};
  std::array<tribol::RealT, QuadraticQuad::size_basis * QuadraticQuad::parametric_dim> quad_derivatives{};
  QuadraticQuad::eval( quad_parametric_coords, quad_values );
  QuadraticQuad::evalParametricDerivatives( quad_parametric_coords, quad_derivatives );

  // The mixed terms exercise the full tensor-product space rather than only independent one-dimensional quadratics.
  auto polynomial = []( tribol::RealT x, tribol::RealT y ) {
    return 1.0 + 2.0 * x - 3.0 * y + 4.0 * x * x + 5.0 * x * y + 6.0 * y * y + 7.0 * x * x * y + 8.0 * x * y * y +
           9.0 * x * x * y * y;
  };
  tribol::RealT interpolated_quad_value{};
  tribol::RealT interpolated_quad_dx{};
  tribol::RealT interpolated_quad_dy{};
  for ( int node = 0; node < QuadraticQuad::size_basis; ++node ) {
    const auto node_coordinates = QuadraticQuad::getParametricNodeCoords( node );
    const tribol::RealT nodal_value = polynomial( node_coordinates[0], node_coordinates[1] );
    interpolated_quad_value += quad_values[node] * nodal_value;
    interpolated_quad_dx += quad_derivatives[QuadraticQuad::parametric_dim * node] * nodal_value;
    interpolated_quad_dy += quad_derivatives[QuadraticQuad::parametric_dim * node + 1] * nodal_value;
  }
  const tribol::RealT x = quad_parametric_coords[0];
  const tribol::RealT y = quad_parametric_coords[1];
  EXPECT_NEAR( interpolated_quad_value, polynomial( x, y ), comparison_tolerance );
  EXPECT_NEAR( interpolated_quad_dx, 2.0 + 8.0 * x + 5.0 * y + 14.0 * x * y + 8.0 * y * y + 18.0 * x * y * y,
               comparison_tolerance );
  EXPECT_NEAR( interpolated_quad_dy, -3.0 + 5.0 * x + 12.0 * y + 7.0 * x * x + 16.0 * x * y + 18.0 * x * x * y,
               comparison_tolerance );
}

TEST( BasisFunction, parametric_derivatives_match_finite_differences )
{
  // Independent finite differences catch sign, scaling, and local-order mistakes in the analytical parametric
  // derivatives for both simplex and tensor-product elements.
  ExpectParametricDerivativesMatchFiniteDifferences<tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>>(
      { 0.23 } );
  ExpectParametricDerivativesMatchFiniteDifferences<tribol::BasisFunction<tribol::TRIANGLE, tribol::LINEAR_LAGRANGE>>(
      { 0.17, 0.31 } );
  ExpectParametricDerivativesMatchFiniteDifferences<tribol::BasisFunction<tribol::QUAD, tribol::QUADRATIC_LAGRANGE>>(
      { -0.37, 0.28 } );
}

TEST( BasisFunction, interpolates_nodal_values_through_connectivity )
{
  // Field interpolation must read spatial_dim arrays through element connectivity because MeshData does not store each
  // element's nodal values in contiguous local arrays.
  using Basis = tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  constexpr tribol::IndexT num_values_per_node = 2;
  constexpr tribol::IndexT num_global_nodes = 5;
  // A noncontiguous, nonmonotonic map detects implementations that accidentally use the local node as a global index.
  constexpr std::array<tribol::IndexT, Basis::size_basis> connectivity{ 3, 1, 4 };

  std::array<tribol::RealT, num_global_nodes> first_nodal_values{};
  std::array<tribol::RealT, num_global_nodes> second_nodal_values{};
  for ( int local_node = 0; local_node < Basis::size_basis; ++local_node ) {
    const auto node_coordinates = Basis::getParametricNodeCoords( local_node );
    const tribol::RealT node_coordinate = node_coordinates[0];
    const tribol::IndexT global_node = connectivity[local_node];
    first_nodal_values[global_node] = 2.0 + 3.0 * node_coordinate + 4.0 * node_coordinate * node_coordinate;
    second_nodal_values[global_node] = -1.0 + 2.0 * node_coordinate - 3.0 * node_coordinate * node_coordinate;
  }
  const NodalValuesThroughConnectivity<num_values_per_node> nodal_values{
      { first_nodal_values.data(), second_nodal_values.data() }, connectivity.data() };

  constexpr tribol::RealT parametric_coord = 0.25;
  tribol::MdArray<tribol::RealT, Basis::size_basis> basis_values;
  Basis::eval( parametric_coord, basis_values.view() );
  tribol::MdArray<tribol::RealT, num_values_per_node> interpolated_values;
  tribol::interpolateNodalValues<Basis, num_values_per_node>( basis_values.view(), nodal_values,
                                                              interpolated_values.view() );

  EXPECT_NEAR( interpolated_values[0], 2.0 + 3.0 * parametric_coord + 4.0 * parametric_coord * parametric_coord,
               comparison_tolerance );
  EXPECT_NEAR( interpolated_values[1], -1.0 + 2.0 * parametric_coord - 3.0 * parametric_coord * parametric_coord,
               comparison_tolerance );
}

TEST( BasisFunction, maps_affine_linear_elements )
{
  // Affine elements have constant analytical mappings, so they provide exact reference values for mapped points,
  // Jacobians, and line or surface integration weights.
  using EdgeBasis = tribol::BasisFunction<tribol::EDGE, tribol::LINEAR_LAGRANGE>;
  constexpr tribol::RealT edge_parametric_coord = 0.25;
  constexpr std::array<tribol::RealT, EdgeBasis::size_basis * EdgeBasis::spatial_dim> edge_coordinates{ 1.0, 2.0, 5.0,
                                                                                                        4.0 };
  std::array<tribol::RealT, EdgeBasis::size_basis> edge_values{};
  std::array<tribol::RealT, EdgeBasis::size_basis * EdgeBasis::parametric_dim> edge_parametric_derivatives{};
  std::array<tribol::RealT, EdgeBasis::spatial_dim> edge_point{};
  std::array<tribol::RealT, EdgeBasis::spatial_dim * EdgeBasis::parametric_dim> edge_jacobian{};
  EdgeBasis::eval( edge_parametric_coord, edge_values );
  EdgeBasis::evalParametricDerivatives( edge_parametric_coord, edge_parametric_derivatives );
  tribol::mapToPhysical<EdgeBasis>( edge_values, edge_coordinates, edge_point );
  tribol::computeMappingJacobian<EdgeBasis>( edge_parametric_derivatives, edge_coordinates, edge_jacobian );

  EXPECT_NEAR( edge_point[0], 3.5, comparison_tolerance );
  EXPECT_NEAR( edge_point[1], 3.25, comparison_tolerance );
  EXPECT_NEAR( edge_jacobian[0], 2.0, comparison_tolerance );
  EXPECT_NEAR( edge_jacobian[1], 1.0, comparison_tolerance );
  // The precomputed-Jacobian and coordinate convenience overloads must produce the same physical integration weight.
  EXPECT_NEAR( tribol::computeJacobianWeight<EdgeBasis>( edge_jacobian ), std::sqrt( 5.0 ), comparison_tolerance );
  EXPECT_NEAR( tribol::computeJacobianWeight<EdgeBasis>( edge_parametric_coord, edge_coordinates ), std::sqrt( 5.0 ),
               comparison_tolerance );

  using QuadBasis = tribol::BasisFunction<tribol::QUAD, tribol::LINEAR_LAGRANGE>;
  constexpr std::array<tribol::RealT, QuadBasis::parametric_dim> quad_parametric_coords{ 0.2, -0.4 };
  std::array<tribol::RealT, QuadBasis::size_basis * QuadBasis::spatial_dim> quad_coordinates{};
  for ( int node = 0; node < QuadBasis::size_basis; ++node ) {
    const auto node_coordinates = QuadBasis::getParametricNodeCoords( node );
    quad_coordinates[QuadBasis::spatial_dim * node] = 1.0 + 2.0 * node_coordinates[0];
    quad_coordinates[QuadBasis::spatial_dim * node + 1] = -2.0 + 3.0 * node_coordinates[1];
    quad_coordinates[QuadBasis::spatial_dim * node + 2] = 4.0 + 0.5 * node_coordinates[0] - node_coordinates[1];
  }
  std::array<tribol::RealT, QuadBasis::size_basis> quad_values{};
  std::array<tribol::RealT, QuadBasis::size_basis * QuadBasis::parametric_dim> quad_parametric_derivatives{};
  std::array<tribol::RealT, QuadBasis::spatial_dim> quad_point{};
  std::array<tribol::RealT, QuadBasis::spatial_dim * QuadBasis::parametric_dim> quad_jacobian{};
  QuadBasis::eval( quad_parametric_coords, quad_values );
  QuadBasis::evalParametricDerivatives( quad_parametric_coords, quad_parametric_derivatives );
  tribol::mapToPhysical<QuadBasis>( quad_values, quad_coordinates, quad_point );
  tribol::computeMappingJacobian<QuadBasis>( quad_parametric_derivatives, quad_coordinates, quad_jacobian );

  constexpr std::array<tribol::RealT, QuadBasis::spatial_dim> expected_quad_point{ 1.4, -3.2, 4.5 };
  constexpr std::array<tribol::RealT, QuadBasis::spatial_dim * QuadBasis::parametric_dim> expected_quad_jacobian{
      2.0, 0.0, 0.0, 3.0, 0.5, -1.0 };
  for ( int spatial_dim = 0; spatial_dim < QuadBasis::spatial_dim; ++spatial_dim ) {
    EXPECT_NEAR( quad_point[spatial_dim], expected_quad_point[spatial_dim], comparison_tolerance );
  }
  for ( int entry = 0; entry < static_cast<int>( quad_jacobian.size() ); ++entry ) {
    EXPECT_NEAR( quad_jacobian[entry], expected_quad_jacobian[entry], comparison_tolerance );
  }
  // The surface weight includes the out-of-plane slope, so this check exercises sqrt(det(J^T J)), not only x-y area.
  EXPECT_NEAR( tribol::computeJacobianWeight<QuadBasis>( quad_jacobian ), 6.5, comparison_tolerance );
  EXPECT_NEAR( tribol::computeJacobianWeight<QuadBasis>( quad_parametric_coords, quad_coordinates ), 6.5,
               comparison_tolerance );
}

TEST( BasisFunction, maps_curved_quadratic_edge )
{
  // A curved edge makes the Jacobian vary with the parametric coordinate and verifies that quadratic geometry nodes
  // affect
  // mapped points, integration weights, and tangential physical derivatives.
  using Basis = tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  constexpr tribol::RealT parametric_coord = 0.25;
  constexpr std::array<tribol::RealT, Basis::size_basis * Basis::spatial_dim> coordinates{ 0.0, 0.0, 2.0,
                                                                                           0.0, 1.0, 1.0 };
  std::array<tribol::RealT, Basis::size_basis> values{};
  std::array<tribol::RealT, Basis::size_basis * Basis::parametric_dim> parametric_derivatives{};
  std::array<tribol::RealT, Basis::spatial_dim> physical_point{};
  std::array<tribol::RealT, Basis::spatial_dim * Basis::parametric_dim> jacobian{};
  std::array<tribol::RealT, Basis::size_basis * Basis::spatial_dim> physical_derivatives{};

  Basis::eval( parametric_coord, values );
  Basis::evalParametricDerivatives( parametric_coord, parametric_derivatives );
  tribol::mapToPhysical<Basis>( values, coordinates, physical_point );
  tribol::computeMappingJacobian<Basis>( parametric_derivatives, coordinates, jacobian );

  EXPECT_NEAR( physical_point[0], 1.0 + parametric_coord, comparison_tolerance );
  EXPECT_NEAR( physical_point[1], 1.0 - parametric_coord * parametric_coord, comparison_tolerance );
  EXPECT_NEAR( jacobian[0], 1.0, comparison_tolerance );
  EXPECT_NEAR( jacobian[1], -2.0 * parametric_coord, comparison_tolerance );
  EXPECT_NEAR( tribol::computeJacobianWeight<Basis>( jacobian ), std::sqrt( 1.25 ), comparison_tolerance );
  EXPECT_NEAR( tribol::computeJacobianWeight<Basis>( parametric_coord, coordinates ), std::sqrt( 1.25 ),
               comparison_tolerance );
  ASSERT_TRUE( tribol::computePhysicalDerivatives<Basis>( jacobian, parametric_derivatives, physical_derivatives ) );

  // Physical derivatives are tangential ambient gradients, so multiplying by J must recover each parametric derivative.
  for ( int node = 0; node < Basis::size_basis; ++node ) {
    tribol::RealT recovered_parametric_derivative{};
    for ( int spatial_dim = 0; spatial_dim < Basis::spatial_dim; ++spatial_dim ) {
      recovered_parametric_derivative +=
          jacobian[spatial_dim] * physical_derivatives[Basis::spatial_dim * node + spatial_dim];
    }
    EXPECT_NEAR( recovered_parametric_derivative, parametric_derivatives[node], comparison_tolerance );
  }
}

TEST( BasisFunction, maps_curved_quadratic_quad )
{
  // A bowed quadratic face verifies the three-dimensional mapping and metric terms that an affine planar face cannot
  // exercise.
  using Basis = tribol::BasisFunction<tribol::QUAD, tribol::QUADRATIC_LAGRANGE>;
  constexpr std::array<tribol::RealT, Basis::parametric_dim> parametric_coords{ 0.2, -0.3 };
  std::array<tribol::RealT, Basis::size_basis * Basis::spatial_dim> coordinates{};
  for ( int node = 0; node < Basis::size_basis; ++node ) {
    const auto node_coordinates = Basis::getParametricNodeCoords( node );
    coordinates[Basis::spatial_dim * node] = 1.0 + node_coordinates[0];
    coordinates[Basis::spatial_dim * node + 1] = 2.0 + node_coordinates[1];
    coordinates[Basis::spatial_dim * node + 2] = 0.25 * ( 1.0 - node_coordinates[0] * node_coordinates[0] ) *
                                                 ( 1.0 - node_coordinates[1] * node_coordinates[1] );
  }

  std::array<tribol::RealT, Basis::size_basis> values{};
  std::array<tribol::RealT, Basis::size_basis * Basis::parametric_dim> parametric_derivatives{};
  std::array<tribol::RealT, Basis::spatial_dim> physical_point{};
  std::array<tribol::RealT, Basis::spatial_dim * Basis::parametric_dim> jacobian{};
  std::array<tribol::RealT, Basis::size_basis * Basis::spatial_dim> physical_derivatives{};
  Basis::eval( parametric_coords, values );
  Basis::evalParametricDerivatives( parametric_coords, parametric_derivatives );
  tribol::mapToPhysical<Basis>( values, coordinates, physical_point );
  tribol::computeMappingJacobian<Basis>( parametric_derivatives, coordinates, jacobian );

  EXPECT_NEAR( physical_point[0], 1.2, comparison_tolerance );
  EXPECT_NEAR( physical_point[1], 1.7, comparison_tolerance );
  EXPECT_NEAR( physical_point[2], 0.2184, comparison_tolerance );
  constexpr std::array<tribol::RealT, Basis::spatial_dim * Basis::parametric_dim> expected_jacobian{
      1.0, 0.0, 0.0, 1.0, -0.091, 0.144 };
  for ( int entry = 0; entry < static_cast<int>( jacobian.size() ); ++entry ) {
    EXPECT_NEAR( jacobian[entry], expected_jacobian[entry], comparison_tolerance );
  }
  EXPECT_NEAR( tribol::computeJacobianWeight<Basis>( jacobian ), std::sqrt( 1.0 + 0.091 * 0.091 + 0.144 * 0.144 ),
               comparison_tolerance );
  ASSERT_TRUE( tribol::computePhysicalDerivatives<Basis>( jacobian, parametric_derivatives, physical_derivatives ) );

  // The chain-rule identity must hold for both parametric directions after lifting derivatives into ambient space.
  for ( int node = 0; node < Basis::size_basis; ++node ) {
    for ( int parametric_dim = 0; parametric_dim < Basis::parametric_dim; ++parametric_dim ) {
      tribol::RealT recovered_parametric_derivative{};
      for ( int spatial_dim = 0; spatial_dim < Basis::spatial_dim; ++spatial_dim ) {
        recovered_parametric_derivative += jacobian[Basis::parametric_dim * spatial_dim + parametric_dim] *
                                           physical_derivatives[Basis::spatial_dim * node + spatial_dim];
      }
      EXPECT_NEAR( recovered_parametric_derivative,
                   parametric_derivatives[Basis::parametric_dim * node + parametric_dim], comparison_tolerance );
    }
  }
}

TEST( BasisFunction, rejects_rank_deficient_mapping )
{
  // Collapsed geometry has no unique tangential physical derivative, so the utility must report failure and erase the
  // entire output instead of leaving partially computed values for contact kernels to consume.
  using EdgeBasis = tribol::BasisFunction<tribol::EDGE, tribol::LINEAR_LAGRANGE>;
  constexpr std::array<tribol::RealT, EdgeBasis::size_basis * EdgeBasis::spatial_dim> edge_coordinates{ 1.0, 2.0, 1.0,
                                                                                                        2.0 };
  constexpr tribol::RealT parametric_coord = 0.0;
  std::array<tribol::RealT, EdgeBasis::size_basis * EdgeBasis::parametric_dim> edge_parametric_derivatives{};
  std::array<tribol::RealT, EdgeBasis::spatial_dim * EdgeBasis::parametric_dim> edge_jacobian{};
  std::array<tribol::RealT, EdgeBasis::size_basis * EdgeBasis::spatial_dim> edge_physical_derivatives{};
  EdgeBasis::evalParametricDerivatives( parametric_coord, edge_parametric_derivatives );
  tribol::computeMappingJacobian<EdgeBasis>( edge_parametric_derivatives, edge_coordinates, edge_jacobian );

  EXPECT_FALSE( tribol::computePhysicalDerivatives<EdgeBasis>( edge_jacobian, edge_parametric_derivatives,
                                                               edge_physical_derivatives ) );
  for ( auto derivative : edge_physical_derivatives ) {
    EXPECT_EQ( derivative, 0.0 );
  }

  using QuadBasis = tribol::BasisFunction<tribol::QUAD, tribol::LINEAR_LAGRANGE>;
  constexpr std::array<tribol::RealT, QuadBasis::parametric_dim> quad_parametric_coords{ 0.0, 0.0 };
  constexpr std::array<tribol::RealT, QuadBasis::size_basis * QuadBasis::spatial_dim> quad_coordinates{
      -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0 };
  std::array<tribol::RealT, QuadBasis::size_basis * QuadBasis::parametric_dim> quad_parametric_derivatives{};
  std::array<tribol::RealT, QuadBasis::spatial_dim * QuadBasis::parametric_dim> quad_jacobian{};
  std::array<tribol::RealT, QuadBasis::size_basis * QuadBasis::spatial_dim> quad_physical_derivatives{};
  QuadBasis::evalParametricDerivatives( quad_parametric_coords, quad_parametric_derivatives );
  tribol::computeMappingJacobian<QuadBasis>( quad_parametric_derivatives, quad_coordinates, quad_jacobian );

  EXPECT_FALSE( tribol::computePhysicalDerivatives<QuadBasis>( quad_jacobian, quad_parametric_derivatives,
                                                               quad_physical_derivatives ) );
  for ( auto derivative : quad_physical_derivatives ) {
    EXPECT_EQ( derivative, 0.0 );
  }
}

TEST( BasisFunction, matches_mfem_h1_basis )
{
  // Matching MFEM values and derivatives confirms that Tribol uses the same H1 local ordering while accounting for
  // Tribol's normalized [-1, 1] edge and quadrilateral coordinates.
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
  // Basis evaluation, fixed-size storage, and nodal interpolation must all compile and execute inside the configured
  // CUDA or HIP kernel path rather than passing only through host-compatible annotations.
  using Basis = tribol::BasisFunction<tribol::EDGE, tribol::QUADRATIC_LAGRANGE>;
  tribol::ArrayT<tribol::RealT, 1, tribol::MemorySpace::Device> device_values( Basis::size_basis + 1 );
  auto values = device_values.view();

#if defined( TRIBOL_USE_CUDA )
  constexpr tribol::ExecutionMode device_execution_mode = tribol::ExecutionMode::Cuda;
#else
  constexpr tribol::ExecutionMode device_execution_mode = tribol::ExecutionMode::Hip;
#endif
  tribol::forAllExec<false>( device_execution_mode, 1, [=] TRIBOL_HOST_DEVICE( tribol::IndexT ) {
    const auto parametric_coords = Basis::getParametricNodeCoords( 2 );
    Basis::eval( parametric_coords.view(),
                 tribol::MdSpan<tribol::RealT, Basis::size_basis>{ values.data(), Basis::size_basis } );

    tribol::MdArray<tribol::RealT, Basis::size_basis, 1> nodal_values;
    for ( int node = 0; node < Basis::size_basis; ++node ) {
      nodal_values( node, 0 ) = static_cast<tribol::RealT>( node + 1 );
    }
    tribol::MdArray<tribol::RealT, 1> interpolated_value;
    tribol::interpolateNodalValues<Basis, 1>(
        tribol::MdSpan<const tribol::RealT, Basis::size_basis>{ values.data(), Basis::size_basis }, nodal_values.view(),
        interpolated_value.view() );
    values[Basis::size_basis] = interpolated_value[0];
  } );

  // Evaluating at the midpoint node produces its Kronecker row, which then selects the third nodal field value.
  tribol::ArrayT<tribol::RealT, 1, tribol::MemorySpace::Host> host_values( device_values );
  EXPECT_NEAR( host_values[0], 0.0, comparison_tolerance );
  EXPECT_NEAR( host_values[1], 0.0, comparison_tolerance );
  EXPECT_NEAR( host_values[2], 1.0, comparison_tolerance );
  EXPECT_NEAR( host_values[Basis::size_basis], 3.0, comparison_tolerance );
}
#endif

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
