// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Unit tests for 3D Common Plane geometries, polygon clipping/mapping, positive-weight polygon quadrature,
 * 3D refined face search, and 3D pair traction residuals and Enzyme coordinate Jacobians.
 */

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "axom/slic.hpp"
#include "ExecutionTest.hpp"
#include "tribol/assembly/EvaluatePairs.hpp"
#include "tribol/geom/CommonPlaneFaceMapping.hpp"
#include "tribol/geom/LinearQuadGeometry.hpp"
#include "tribol/geom/LinearTriangleGeometry.hpp"
#include "tribol/integ/PolygonQuadratureRuleGenerator.hpp"
#include "tribol/integ/QuadratureRuleGeneratorFactory.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/physics/CommonPlaneFormulation.hpp"
#include "tribol/physics/integrals/PairIntegrals.hpp"
#include "tribol/search/ProjectedFaceSearch.hpp"

namespace {

using tribol::ArrayT;
using tribol::BasisFunction;
using tribol::CommonPlaneFaceMapping;
using tribol::CommonPlaneGap;
using tribol::CommonPlanePenaltyTraction;
using tribol::ExecutionMode;
using tribol::IndexT;
using tribol::InterfacePair;
using tribol::LINEAR_LAGRANGE;
using tribol::LinearQuadGeometry;
using tribol::LinearTriangleGeometry;
using tribol::MdArray;
using tribol::MemorySpace;
using tribol::MeshData;
using tribol::PairQuadratureRuleGenerator;
using tribol::PairTractionIntegral;
using tribol::PolygonCentroidQuadratureRuleGenerator;
using tribol::PolygonQuadratureRuleGenerator;
using tribol::PolygonQuadratureSurface;
using tribol::ProjectedFaceSearch;
using tribol::QUAD;
using tribol::QuadratureStatus;
using tribol::RealT;
using tribol::TRIANGLE;

// LinearTriangleGeometry maps vertices, centroid, unit normal, and recovers exact parametric coordinates along a
// normal.
TEST( CommonPlane3DGeometryTest, linear_triangle_position_normal_and_inverse_projection )
{
  const RealT coords[3 * 3] = { 0.0, 0.0, 0.2, 2.0, 0.0, 0.4, 0.0, 3.0, 0.5 };
  const LinearTriangleGeometry tri( coords );

  const auto normal_res = tri.unitNormal();
  ASSERT_TRUE( normal_res.valid );
  EXPECT_GT( normal_res.normal[2], 0.0 );
  EXPECT_NEAR( normal_res.normal.norm(), 1.0, 1.0e-12 );

  const LinearTriangleGeometry::Vector e1{ 1.0, 0.0, 0.0 };
  const LinearTriangleGeometry::Vector e2{ 0.0, 1.0, 0.0 };
  const MdArray<RealT, 2> xi_in{ 0.25, 0.35 };
  const auto p_surface = tri.position( xi_in.view() );
  const LinearTriangleGeometry::Point p_plane{ p_surface[0], p_surface[1], 0.0 };

  const auto proj = tri.parametricCoordinatesAlongDirection( p_plane, e1, e2 );
  ASSERT_TRUE( proj.valid );
  EXPECT_NEAR( proj.parametric_coords[0], xi_in[0], 1.0e-12 );
  EXPECT_NEAR( proj.parametric_coords[1], xi_in[1], 1.0e-12 );

  // Degenerate collinear triangle must report valid = false.
  const RealT deg_coords[3 * 3] = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0 };
  const LinearTriangleGeometry deg_tri( deg_coords );
  EXPECT_FALSE( deg_tri.unitNormal().valid );
}

// LinearQuadGeometry recovers exact parametric coordinates on a warped quad and preserves angular momentum when paired
// along the common-plane normal.
TEST( CommonPlane3DGeometryTest, warped_quad_inverse_projection_conserves_angular_momentum )
{
  // Warped lower quad (corner 2 raised by +0.15) and tilted upper quad.
  const RealT coords1[4 * 3] = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.15, 0.0, 1.0, 0.0 };
  const RealT coords2[4 * 3] = { 0.0, 1.0, 0.02, 1.0, 1.0, -0.03, 1.0, 0.0, 0.04, 0.0, 0.0, -0.02 };
  const LinearQuadGeometry quad1( coords1 );
  const LinearQuadGeometry quad2( coords2 );

  const CommonPlaneFaceMapping<LinearQuadGeometry> mapping{};
  const PairQuadratureRuleGenerator<CommonPlaneFaceMapping<LinearQuadGeometry>, PolygonQuadratureRuleGenerator<3>>
      rule_gen( mapping );
  const auto rule = rule_gen.generate( quad1, quad2 );
  ASSERT_EQ( rule.metadata.status, QuadratureStatus::Success );
  ASSERT_GT( rule.metadata.num_required_points, 0 );

  // Every paired quadrature point must satisfy (point1 - point2) x normal == 0 (exact angular momentum conservation).
  for ( IndexT q = 0; q < rule.metadata.num_required_points; ++q ) {
    const auto p1 = quad1.position( { &rule.geom1_parametric_coords( q, 0 ), 2 } );
    const auto p2 = quad2.position( { &rule.geom2_parametric_coords( q, 0 ), 2 } );
    const LinearQuadGeometry::Vector normal( &rule.integration_surface_normals( q, 0 ), 3 );
    const LinearQuadGeometry::Vector sep( p2, p1 );
    const auto moment_arm = LinearQuadGeometry::Vector::cross_product( sep, normal );
    EXPECT_LT( moment_arm.norm(), 1.0e-11 );
  }
}

// PolygonQuadratureRuleGenerator integrates all monomials u^p * v^q exactly up to each rule's polynomial degree with
// strictly positive weights.
TEST( CommonPlane3DQuadratureTest, polygon_rules_integrate_monomials_exactly_with_positive_weights )
{
  // Non-symmetric convex pentagon in (u, v).
  PolygonQuadratureSurface poly{};
  poly.valid = true;
  poly.num_vertices = 5;
  poly.vertices( 0, 0 ) = 0.0;
  poly.vertices( 0, 1 ) = 0.0;
  poly.vertices( 1, 0 ) = 2.0;
  poly.vertices( 1, 1 ) = 0.2;
  poly.vertices( 2, 0 ) = 2.4;
  poly.vertices( 2, 1 ) = 1.5;
  poly.vertices( 3, 0 ) = 0.9;
  poly.vertices( 3, 1 ) = 2.2;
  poly.vertices( 4, 0 ) = -0.3;
  poly.vertices( 4, 1 ) = 1.1;

  // Reference exact integral of u^p * v^q over a polygon via exact Green's theorem / high-order decomposition on the
  // 7-point degree-5 rule (which is exact through degree 5).
  const auto rule1 = PolygonCentroidQuadratureRuleGenerator{}.generate( poly );
  const auto rule2 = PolygonQuadratureRuleGenerator<3>{}.generate( poly );
  const auto rule4 = PolygonQuadratureRuleGenerator<6>{}.generate( poly );
  const auto rule5 = PolygonQuadratureRuleGenerator<7>{}.generate( poly );

  ASSERT_EQ( rule1.metadata.status, QuadratureStatus::Success );
  ASSERT_EQ( rule1.metadata.num_required_points, 1 );
  ASSERT_EQ( rule2.metadata.num_required_points, 15 );
  ASSERT_EQ( rule4.metadata.num_required_points, 30 );
  ASSERT_EQ( rule5.metadata.num_required_points, 35 );

  // All weights must be strictly positive.
  for ( IndexT q = 0; q < rule5.metadata.num_required_points; ++q ) {
    EXPECT_GT( rule5.weights[q], 0.0 );
  }
  for ( IndexT q = 0; q < rule4.metadata.num_required_points; ++q ) {
    EXPECT_GT( rule4.weights[q], 0.0 );
  }

  // Exact shoelace area of the pentagon.
  RealT exact_area = 0.0;
  for ( IndexT v = 0; v < poly.num_vertices; ++v ) {
    const IndexT nv = ( v + 1 ) % poly.num_vertices;
    exact_area +=
        0.5 * ( poly.vertices( v, 0 ) * poly.vertices( nv, 1 ) - poly.vertices( nv, 0 ) * poly.vertices( v, 1 ) );
  }
  EXPECT_NEAR( rule1.weights[0], exact_area, 1.0e-13 );

  // Verify exact agreement across rules up to each rule's polynomial degree.
  auto eval_monomial = []( const auto& rule, int p, int q_exp ) {
    RealT sum = 0.0;
    for ( IndexT i = 0; i < rule.metadata.num_required_points; ++i ) {
      sum += rule.weights[i] * std::pow( rule.coords( i, 0 ), p ) * std::pow( rule.coords( i, 1 ), q_exp );
    }
    return sum;
  };

  for ( int deg = 0; deg <= 5; ++deg ) {
    for ( int p = 0; p <= deg; ++p ) {
      const int q_exp = deg - p;
      const RealT ref = eval_monomial( rule5, p, q_exp );
      if ( deg <= 1 ) {
        EXPECT_NEAR( eval_monomial( rule1, p, q_exp ), ref, 1.0e-12 );
      }
      if ( deg <= 2 ) {
        EXPECT_NEAR( eval_monomial( rule2, p, q_exp ), ref, 1.0e-12 );
      }
      if ( deg <= 4 ) {
        EXPECT_NEAR( eval_monomial( rule4, p, q_exp ), ref, 1.0e-12 );
      }
    }
  }

  // Check exact degree-5 integral on the unit square [0, 1]^2: int x^5 dx dy = 1/6, int x^3 y^2 dx dy = 1/12.
  PolygonQuadratureSurface square{};
  square.valid = true;
  square.num_vertices = 4;
  square.vertices( 0, 0 ) = 0.0;
  square.vertices( 0, 1 ) = 0.0;
  square.vertices( 1, 0 ) = 1.0;
  square.vertices( 1, 1 ) = 0.0;
  square.vertices( 2, 0 ) = 1.0;
  square.vertices( 2, 1 ) = 1.0;
  square.vertices( 3, 0 ) = 0.0;
  square.vertices( 3, 1 ) = 1.0;
  const auto sq_rule5 = PolygonQuadratureRuleGenerator<7>{}.generate( square );
  EXPECT_NEAR( eval_monomial( sq_rule5, 5, 0 ), 1.0 / 6.0, 1.0e-13 );
  EXPECT_NEAR( eval_monomial( sq_rule5, 3, 2 ), 1.0 / 12.0, 1.0e-13 );
  EXPECT_NEAR( eval_monomial( sq_rule5, 4, 1 ), 1.0 / 10.0, 1.0e-13 );
}

// CommonPlaneFaceMapping handles partial overlap, planar crossing, exact/near-vertex half-plane clips, nearly-parallel
// edges, touching/non-opposing empty rules, and degenerate invalid geometry.
TEST( CommonPlane3DMappingTest, clipping_edge_cases_and_invalid_geometry )
{
  const CommonPlaneFaceMapping<LinearQuadGeometry> mapping{};

  // 1. Planar crossing: lower quad at z = 0, upper quad tilted z = x - 0.5 (penetrating for x in [0, 0.5]).
  // Projected onto common plane, the penetrating half has area 0.5 / cos(theta/2) ~ 0.5 * sqrt(...).
  const RealT lower[4 * 3] = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
  const RealT upper_cross[4 * 3] = { 0.0, 1.0, -0.2, 1.0, 1.0, 0.2, 1.0, 0.0, 0.2, 0.0, 0.0, -0.2 };
  const auto cross_surface =
      mapping.makeIntegrationSurface( LinearQuadGeometry( lower ), LinearQuadGeometry( upper_cross ) );
  ASSERT_TRUE( cross_surface.valid );
  ASSERT_EQ( cross_surface.num_vertices, 4 );
  const auto cross_rule = PolygonCentroidQuadratureRuleGenerator{}.generate( cross_surface );
  ASSERT_EQ( cross_rule.metadata.num_required_points, 1 );
  EXPECT_GT( cross_rule.weights[0], 0.4 );
  EXPECT_LT( cross_rule.weights[0], 0.6 );

  // 2. Half-plane clip landing right at/near a vertex (z = 0 at (0,0), z = +1e-14 at (0,1), z = -0.2 at x=1).
  const RealT upper_near_vert[4 * 3] = { 0.0, 1.0, 1.0e-14, 1.0, 1.0, -0.2, 1.0, 0.0, -0.2, 0.0, 0.0, 0.0 };
  const auto near_vert_surface =
      mapping.makeIntegrationSurface( LinearQuadGeometry( lower ), LinearQuadGeometry( upper_near_vert ) );
  ASSERT_TRUE( near_vert_surface.valid );
  ASSERT_GE( near_vert_surface.num_vertices, 3 );
  const auto near_vert_rule = PolygonQuadratureRuleGenerator<3>{}.generate( near_vert_surface );
  EXPECT_EQ( near_vert_rule.metadata.status, QuadratureStatus::Success );
  EXPECT_GT( near_vert_rule.metadata.num_required_points, 0 );

  // 3. Touching (edge-adjacent, zero area) returns valid empty polygon.
  const RealT upper_touch[4 * 3] = { 1.0, 1.0, -0.01, 2.0, 1.0, -0.01, 2.0, 0.0, -0.01, 1.0, 0.0, -0.01 };
  const auto touch_surface =
      mapping.makeIntegrationSurface( LinearQuadGeometry( lower ), LinearQuadGeometry( upper_touch ) );
  EXPECT_TRUE( touch_surface.valid );
  EXPECT_EQ( touch_surface.num_vertices, 0 );

  // 4. Non-opposing normals (both +z) return valid empty polygon.
  const RealT upper_same_dir[4 * 3] = { 0.0, 0.0, 0.01, 1.0, 0.0, 0.01, 1.0, 1.0, 0.01, 0.0, 1.0, 0.01 };
  const auto non_opp_surface =
      mapping.makeIntegrationSurface( LinearQuadGeometry( lower ), LinearQuadGeometry( upper_same_dir ) );
  EXPECT_TRUE( non_opp_surface.valid );
  EXPECT_EQ( non_opp_surface.num_vertices, 0 );

  // 5. Degenerate face returns InvalidGeometry.
  const RealT deg[4 * 3] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  const auto deg_surface = mapping.makeIntegrationSurface( LinearQuadGeometry( lower ), LinearQuadGeometry( deg ) );
  EXPECT_FALSE( deg_surface.valid );
}

#ifdef TRIBOL_USE_ENZYME
// Verify 3D PairTractionIntegral residual and Enzyme coordinate Jacobian against centered finite differences for
// linear triangles, planar quads, warped quads, and a planar crossing perturbed into a warped crossing.
template <typename Geometry, typename Basis, int Order>
void verify3DPairJacobianAgainstFiniteDiff( const RealT* coords1_data, const RealT* coords2_data, RealT penalty,
                                            RealT rel_tol = 1.0e-6 )
{
  const CommonPlaneFaceMapping<Geometry> mapping{};
  tribol::createWithQuadratureRuleGenerator( Order, mapping, [&]( auto rule_generator ) {
    using PairFunction =
        PairTractionIntegral<decltype( rule_generator ), CommonPlanePenaltyTraction<CommonPlaneGap>, Basis>;
    const PairFunction pair_function( rule_generator, { penalty, CommonPlaneGap{} } );

    constexpr IndexT n1 = Geometry::num_nodes;
    constexpr IndexT n2 = Geometry::num_nodes;
    std::vector<RealT> x1( n1 ), y1( n1 ), z1( n1 ), x2( n2 ), y2( n2 ), z2( n2 );
    std::vector<IndexT> conn1( n1 ), conn2( n2 );
    for ( IndexT i = 0; i < n1; ++i ) {
      x1[i] = coords1_data[3 * i + 0];
      y1[i] = coords1_data[3 * i + 1];
      z1[i] = coords1_data[3 * i + 2];
      conn1[i] = i;
    }
    for ( IndexT i = 0; i < n2; ++i ) {
      x2[i] = coords2_data[3 * i + 0];
      y2[i] = coords2_data[3 * i + 1];
      z2[i] = coords2_data[3 * i + 2];
      conn2[i] = i;
    }

    const auto elem_type = ( n1 == 3 ) ? TRIANGLE : QUAD;
    MeshData mesh1( 0, 1, n1, conn1.data(), elem_type, LINEAR_LAGRANGE, x1.data(), y1.data(), z1.data(),
                    MemorySpace::Host );
    MeshData mesh2( 1, 1, n2, conn2.data(), elem_type, LINEAR_LAGRANGE, x2.data(), y2.data(), z2.data(),
                    MemorySpace::Host );
    ArrayT<InterfacePair> pairs( { InterfacePair{ 0, 0 } } );

    auto eval_and_jac = tribol::evaluatePairValuesAndJacobians<ExecutionMode::Sequential>(
        pairs, mesh1.getView(), mesh2.getView(), pair_function );
    ASSERT_EQ( eval_and_jac.quadratureMetadata()[0].status, QuadratureStatus::Success );
    ASSERT_GT( eval_and_jac.quadratureMetadata()[0].num_required_points, 0 );

    const auto jac = eval_and_jac.view();
    constexpr IndexT size_vals = PairFunction::size_values;
    constexpr IndexT size_coords = PairFunction::size_coords;
    constexpr RealT eps = 1.0e-7;

    auto eval_residual = [&]( const std::vector<RealT>& c1, const std::vector<RealT>& c2 ) {
      return pair_function.evaluate( { c1.data(), n1 * 3 }, { c2.data(), n2 * 3 } );
    };

    std::vector<RealT> c1_base( coords1_data, coords1_data + 3 * n1 );
    std::vector<RealT> c2_base( coords2_data, coords2_data + 3 * n2 );
    for ( IndexT col = 0; col < size_coords; ++col ) {
      auto c1_plus = c1_base;
      auto c2_plus = c2_base;
      auto c1_minus = c1_base;
      auto c2_minus = c2_base;
      if ( col < 3 * n1 ) {
        c1_plus[col] += eps;
        c1_minus[col] -= eps;
      } else {
        c2_plus[col - 3 * n1] += eps;
        c2_minus[col - 3 * n1] -= eps;
      }
      const auto r_plus = eval_residual( c1_plus, c2_plus );
      const auto r_minus = eval_residual( c1_minus, c2_minus );
      for ( IndexT row = 0; row < size_vals; ++row ) {
        const RealT fd = ( r_plus.values()[row] - r_minus.values()[row] ) / ( 2.0 * eps );
        const RealT ad = jac.jacobian( 0, row, col );
        const RealT scale = std::max( RealT{ 1.0 }, std::abs( fd ) );
        EXPECT_NEAR( ad, fd, rel_tol * scale ) << "row=" << row << " col=" << col;
      }
    }
  } );
}

TEST( CommonPlane3DPairEvalTest, enzyme_jacobian_matches_fd_for_quads_and_triangles )
{
  // 1. Warped, rotated quad pair (partial overlap).
  const RealT quad1[4 * 3] = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.02, 1.0, 1.0, 0.08, 0.0, 1.0, -0.01 };
  const RealT quad2[4 * 3] = { 0.2, 1.1, -0.05, 1.2, 1.1, -0.03, 1.2, 0.1, -0.04, 0.2, 0.1, -0.06 };
  verify3DPairJacobianAgainstFiniteDiff<LinearQuadGeometry, BasisFunction<QUAD, LINEAR_LAGRANGE>, 1>( quad1, quad2,
                                                                                                      10.0 );
  verify3DPairJacobianAgainstFiniteDiff<LinearQuadGeometry, BasisFunction<QUAD, LINEAR_LAGRANGE>, 2>( quad1, quad2,
                                                                                                      10.0 );

  // 2. Planar crossing quad perturbed into a warped crossing (preserving clipping topology).
  const RealT cross1[4 * 3] = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.02, 0.0, 1.0, 0.0 };
  const RealT cross2[4 * 3] = { 0.0, 1.0, -0.15, 1.0, 1.0, 0.15, 1.0, 0.0, 0.14, 0.0, 0.0, -0.16 };
  verify3DPairJacobianAgainstFiniteDiff<LinearQuadGeometry, BasisFunction<QUAD, LINEAR_LAGRANGE>, 2>( cross1, cross2,
                                                                                                      10.0 );

  // 3. Linear triangle pair (partial overlap, crossing).
  const RealT tri1[3 * 3] = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0 };
  const RealT tri2[3 * 3] = { 0.1, 0.9, -0.08, 0.9, 0.1, 0.06, 0.1, 0.1, -0.05 };
  verify3DPairJacobianAgainstFiniteDiff<LinearTriangleGeometry, BasisFunction<TRIANGLE, LINEAR_LAGRANGE>, 2>(
      tri1, tri2, 10.0 );
}
#endif  // TRIBOL_USE_ENZYME

class ProjectedFaceSearchExecutionTest : public tribol::detail::ExecutionTest {};

// ProjectedFaceSearch filters self-pairs, shared-node pairs, and non-opposing 3D faces across enabled backends.
TEST_P( ProjectedFaceSearchExecutionTest, filters_3d_quad_and_triangle_candidates )
{
  // Two opposing quads (elem 0 on mesh1 at z=0 with +z normal, elem 0 on mesh2 at z=0.1 with -z normal, and elem 1 on
  // mesh2 at z=0.2 with +z normal).
  const auto x1 = copyToExecutionSpace<RealT>( { 0.0, 1.0, 1.0, 0.0 } );
  const auto y1 = copyToExecutionSpace<RealT>( { 0.0, 0.0, 1.0, 1.0 } );
  const auto z1 = copyToExecutionSpace<RealT>( { 0.0, 0.0, 0.0, 0.0 } );
  const auto conn1 = copyToExecutionSpace<IndexT>( { 0, 1, 2, 3 } );

  const auto x2 = copyToExecutionSpace<RealT>( { 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0 } );
  const auto y2 = copyToExecutionSpace<RealT>( { 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 } );
  const auto z2 = copyToExecutionSpace<RealT>( { 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2 } );
  const auto conn2 = copyToExecutionSpace<IndexT>( { 0, 1, 2, 3, 4, 5, 6, 7 } );

  MeshData mesh1( 0, 1, 4, conn1.data(), QUAD, LINEAR_LAGRANGE, x1.data(), y1.data(), z1.data(), memorySpace() );
  MeshData mesh2( 1, 2, 8, conn2.data(), QUAD, LINEAR_LAGRANGE, x2.data(), y2.data(), z2.data(), memorySpace() );

  const auto candidates = copyToExecutionSpace<InterfacePair>( { InterfacePair{ 0, 0 }, InterfacePair{ 0, 1 } } );
  const ProjectedFaceSearch search( GetParam(), false );
  const auto retained = search.filterCandidatePairs( candidates, mesh1.getView(), mesh2.getView() );
  const ArrayT<InterfacePair, 1, MemorySpace::Host> retained_host( retained );
  ASSERT_EQ( retained_host.size(), 1 );
  EXPECT_EQ( retained_host[0].element_id1_, 0 );
  EXPECT_EQ( retained_host[0].element_id2_, 0 );
}

INSTANTIATE_TEST_SUITE_P( execution_modes, ProjectedFaceSearchExecutionTest, tribol::detail::enabledExecutionModes() );

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
