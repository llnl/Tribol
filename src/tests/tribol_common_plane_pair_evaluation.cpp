// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Verify Common Plane pair quadrature, residuals, and automatic derivatives independently of MFEM assembly.
 *
 * Direct backend tests cover overlap mapping, physical integration weights, prescribed pressure, gap activation,
 * quadrature failures, and result ownership. Host finite differences check the complete pair residual Jacobian.
 */
#include <algorithm>
#include <cmath>
#include "gtest/gtest.h"
#include "SlicErrorTest.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "ExecutionTest.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/physics/CommonPlane.hpp"
#include "tribol/physics/CommonPlaneFormulation.hpp"
#include "tribol/geom/CommonPlaneQuadraticEdgeProjection.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/geom/QuadraticEdgeGeometry.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"
#include "tribol/integ/SubdividedPairQuadratureRuleGenerator.hpp"
#include "tribol/physics/integrals/PairIntegrals.hpp"
#include "tribol/assembly/EvaluatePairs.hpp"
#include "tribol/assembly/PairFunctionData.hpp"

namespace {
using namespace tribol;
constexpr RealT tolerance = 1.0e-12;  ///< Roundoff allowance for geometry and exact polynomial integrals.
using Coordinates = MdArray<RealT, LinearEdgeGeometry::num_vertices, LinearEdgeGeometry::spatial_dim>;
using QuadraticCoordinates = MdArray<RealT, QuadraticEdgeGeometry::num_nodes, QuadraticEdgeGeometry::spatial_dim>;
using CommonPlaneRuleGenerator = PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<1>>;
using CommonPlaneTraction = CommonPlanePenaltyTraction<CommonPlaneGap>;
using CommonPlaneResidualValue = PairTractionIntegral<CommonPlaneRuleGenerator, CommonPlaneTraction,
                                                      BasisFunction<EDGE, LINEAR_LAGRANGE>>::PairResidual;

/**
 * @brief Quadratic subdivided Common Plane rule generator with a chosen subelement point projection.
 * @tparam NumPoints Number of Gauss points on each subelement-pair overlap.
 * @tparam Projection Policy mapping paired subelement points onto both quadratic edges.
 */
template <int NumPoints, typename Projection>
using QuadraticCommonPlaneRuleGenerator = SubdividedPairQuadratureRuleGenerator<
    QuadraticEdgeGeometry, PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>,
    max_num_quadrature_subdivisions, Projection>;

/**
 * @brief Quadratic subdivided Common Plane rule generator as configured by the formulation factory.
 * @tparam NumPoints Number of Gauss points on each subelement-pair overlap.
 */
template <int NumPoints>
using ProjectedQuadraticCommonPlaneRuleGenerator =
    QuadraticCommonPlaneRuleGenerator<NumPoints, CommonPlaneQuadraticEdgeProjection>;

/**
 * @brief Quadratic Common Plane pair integral as configured by the formulation factory.
 * @tparam NumPoints Number of Gauss points on each subelement-pair overlap.
 */
template <int NumPoints>
using ProjectedQuadraticCommonPlaneIntegral =
    PairTractionIntegral<ProjectedQuadraticCommonPlaneRuleGenerator<NumPoints>, ProjectedCommonPlanePenaltyTraction,
                         BasisFunction<EDGE, QUADRATIC_LAGRANGE>>;

static_assert( PairFunctionType<ProjectedQuadraticCommonPlaneIntegral<3>> );

/** @brief Net force and net moment about the origin of one 2D pair residual. */
struct NetForceAndMoment {
  MdArray<RealT, 2> force{};   ///< Sum of nodal residual vectors on both edges.
  RealT moment = 0.0;          ///< Sum of nodal position cross nodal residual on both edges.
  RealT max_nodal_load = 0.0;  ///< Largest residual component, used to scale tolerances.
};

/**
 * @brief Sum nodal residual vectors and their moments about the origin over both quadratic edges.
 * @tparam PairResidual Pair residual with basis-indexed geometry1 and geometry2 entries.
 * @param residual Pair residual whose basis indices match the nodes of coords1 and coords2.
 * @param coords1 Nodal coordinates of geometry1.
 * @param coords2 Nodal coordinates of geometry2.
 * @return Net force, net moment, and largest residual component.
 */
template <typename PairResidual>
NetForceAndMoment computeNetForceAndMoment( const PairResidual& residual, const QuadraticCoordinates& coords1,
                                            const QuadraticCoordinates& coords2 )
{
  NetForceAndMoment result;
  for ( IndexT node = 0; node < QuadraticEdgeGeometry::num_nodes; ++node ) {
    for ( const bool first_geometry : { true, false } ) {
      const QuadraticCoordinates& coords = first_geometry ? coords1 : coords2;
      const RealT load_x = first_geometry ? residual.geometry1( node, 0 ) : residual.geometry2( node, 0 );
      const RealT load_y = first_geometry ? residual.geometry1( node, 1 ) : residual.geometry2( node, 1 );
      result.force[0] += load_x;
      result.force[1] += load_y;
      result.moment += coords( node, 0 ) * load_y - coords( node, 1 ) * load_x;
      result.max_nodal_load = std::max( { result.max_nodal_load, std::abs( load_x ), std::abs( load_y ) } );
    }
  }
  return result;
}

template <int NumPoints>
struct InterpenetratingOverlapResult {
  EdgeQuadratureInterval interval;      ///< Calculated geometric bounds.
  CommonPlaneResidualValue residual;    ///< Pair loads and quadrature status.
  MdArray<RealT, NumPoints> x1{};       ///< Point positions on edge1.
  MdArray<RealT, NumPoints> x2{};       ///< Point positions on edge2.
  MdArray<RealT, NumPoints> gaps{};     ///< Unshifted point gaps.
  MdArray<RealT, NumPoints> weights{};  ///< Physical line weights.
};

static_assert( PairTractionFunction<CommonPlaneTraction, CommonPlaneRuleGenerator> );
static_assert( !PairEnergyFunction<CommonPlaneTraction, CommonPlaneRuleGenerator> );

// Common Plane exposes the same values and quadrature state required by the shared pair derivative calculations.
static_assert( PairFunctionType<PairTractionIntegral<CommonPlaneRuleGenerator, CommonPlaneTraction,
                                                     BasisFunction<EDGE, LINEAR_LAGRANGE>>> );

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Copy two edge-coordinate arrays into Jacobian column order for finite differences.
 * @tparam NumNodes Number of nodes defining each edge.
 * @param coordinates1 First edge's vertex coordinates.
 * @param coordinates2 Second edge's vertex coordinates.
 * @return Owned coordinates ordered by edge, vertex, and component.
 */
template <IndexT NumNodes>
TRIBOL_HOST_DEVICE MdArray<RealT, 2 * NumNodes * LinearEdgeGeometry::spatial_dim> packPairCoordinates(
    MdSpan<const RealT, NumNodes, LinearEdgeGeometry::spatial_dim> coordinates1,
    MdSpan<const RealT, NumNodes, LinearEdgeGeometry::spatial_dim> coordinates2 )
{
  constexpr IndexT coordinates_per_edge = NumNodes * LinearEdgeGeometry::spatial_dim;
  MdArray<RealT, 2 * coordinates_per_edge> coordinates{};
  for ( IndexT coordinate = 0; coordinate < coordinates_per_edge; ++coordinate ) {
    coordinates[coordinate] = coordinates1[coordinate];
    coordinates[coordinates_per_edge + coordinate] = coordinates2[coordinate];
  }
  return coordinates;
}

#endif

/** @brief Execute local geometry and residual calculations using backend-accessible storage. */
class CommonPlaneTest : public tribol::detail::ExecutionTest {
 public:
  // NVCC requires functions containing extended device lambdas to have public access.
  /** @brief Check exact parallel-edge loads and activation. @tparam NumPoints Gauss point count. */
  template <int NumPoints>
  void checkParallelResiduals()
  {
    struct ParallelResidualCase {
      RealT separation;        ///< Height of the second edge; positive means penetration for these orientations.
      RealT residual_gap;      ///< Separation offset subtracted before activation.
      RealT expected_first_y;  ///< Expected residual at either first-edge node.
      bool has_active_points;  ///< Whether the nonpositive gap branch is taken.
    };
    // Unequal mesh springs combine to stiffness 1.5, rather than their arithmetic average 4.0.
    const ParallelResidualCase cases[] = {
        { 0.1, 0.0, -0.15, true }, { -0.1, 0.0, 0.0, false }, { 0.0, 0.0, 0.0, true }, { -0.05, 0.1, -0.075, true } };
    for ( const auto test_case : cases ) {
      SCOPED_TRACE( test_case.separation );
      const RealT separation = test_case.separation;
      const RealT residual_gap = test_case.residual_gap;
      ArrayT<CommonPlaneResidualValue> pair_residuals( 1, 1, allocatorId() );
      const auto pair_residuals_view = pair_residuals.view();
      forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
        const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.0 } };
        const Coordinates edge2{ MdArray<RealT, 2>{ 2.0, separation }, MdArray<RealT, 2>{ 0.0, separation } };
        const PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>
            rule_generator;
        const PairTractionIntegral<
            PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>,
            CommonPlanePenaltyTraction<OffsetGap<CommonPlaneGap>>, BasisFunction<EDGE, LINEAR_LAGRANGE>>
            pair_integral{
                rule_generator,
                { ComputePenaltyStiffnessPerArea( 2.0, 6.0 ), OffsetGap{ CommonPlaneGap{}, residual_gap } } };
        pair_residuals_view[0] = pair_integral.evaluate( edge1.view(), edge2.view() );
      } );
      const ArrayT<CommonPlaneResidualValue, 1, MemorySpace::Host> host_pair_residuals( pair_residuals );
      const auto& pair_residual = host_pair_residuals[0];
      ASSERT_EQ( pair_residual.quadrature_metadata.status, QuadratureStatus::Success );
      EXPECT_EQ( pair_residual.quadrature_metadata.num_required_points, NumPoints );
      EXPECT_EQ( pair_residual.has_active_points, test_case.has_active_points );
      for ( int node = 0; node < BasisFunction<EDGE, LINEAR_LAGRANGE>::num_nodes; ++node ) {
        EXPECT_NEAR( pair_residual.geometry1( node, 0 ), 0.0, tolerance );
        EXPECT_NEAR( pair_residual.geometry1( node, 1 ), test_case.expected_first_y, tolerance );
        EXPECT_NEAR( pair_residual.geometry2( node, 0 ), 0.0, tolerance );
        EXPECT_NEAR( pair_residual.geometry2( node, 1 ), -test_case.expected_first_y, tolerance );
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
    ArrayT<RealT, 2> quadrature_samples( axom::StackArray<axom::IndexType, 2>{ NumPoints, NumColumns }, allocatorId() );
    const auto quadrature_samples_view = quadrature_samples.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      // Oppositely directed parallel edges whose projected overlap is x in [1, 3].
      const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 4.0, 0.0 } };
      const Coordinates edge2{ MdArray<RealT, 2>{ 3.0, 0.2 }, MdArray<RealT, 2>{ 1.0, 0.2 } };
      const LinearEdgeGeometry geometry1( edge1.view() ), geometry2( edge2.view() );
      const PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>> rule_generator;
      const auto quadrature_rule = rule_generator.generate( geometry1, geometry2 );
      const auto quadrature_points = quadrature_rule.getView();
      for ( int q = 0; q < NumPoints; ++q ) {
        quadrature_samples_view( q, Parent1 ) = quadrature_points.geom1_parametric_coords( q, 0 );
        quadrature_samples_view( q, Parent2 ) = quadrature_points.geom2_parametric_coords( q, 0 );
        quadrature_samples_view( q, Weight ) = quadrature_points.weights[q];
        quadrature_samples_view( q, Point1X ) = geometry1.position( { &quadrature_samples_view( q, Parent1 ), 1 } )[0];
        quadrature_samples_view( q, Point2X ) = geometry2.position( { &quadrature_samples_view( q, Parent2 ), 1 } )[0];
      }
    } );
    const ArrayT<RealT, 2, MemorySpace::Host> host_quadrature_samples( quadrature_samples );
    RealT measure = 0.0, second_moment = 0.0;
    for ( int q = 0; q < NumPoints; ++q ) {
      EXPECT_GE( host_quadrature_samples( q, Parent1 ), -0.5 );
      EXPECT_LE( host_quadrature_samples( q, Parent1 ), 0.5 );
      EXPECT_NEAR( host_quadrature_samples( q, Point1X ), host_quadrature_samples( q, Point2X ), tolerance );
      EXPECT_NEAR( host_quadrature_samples( q, Parent2 ), 2.0 - host_quadrature_samples( q, Point1X ), tolerance );
      EXPECT_GT( host_quadrature_samples( q, Weight ), 0.0 );
      measure += host_quadrature_samples( q, Weight );
      second_moment += host_quadrature_samples( q, Weight ) * host_quadrature_samples( q, Point1X ) *
                       host_quadrature_samples( q, Point1X );
    }
    EXPECT_NEAR( measure, 2.0, tolerance );
    if constexpr ( NumPoints >= 2 ) {
      EXPECT_NEAR( second_moment, 26.0 / 3.0, tolerance );
    }
  }

  /** @brief Check crossing-edge intervals and pointwise-pressure loads. @tparam NumPoints Configured Gauss point count.
   */
  template <int NumPoints>
  void checkInterpenetratingOverlap()
  {
    struct InterpenetratingOverlapCase {
      RealT slope;          ///< Slope of edge1; edge2 has the opposite slope.
      RealT separation;     ///< Height of edge2 at x=0.0.
      RealT residual_gap;   ///< Value subtracted from the geometric gap in the pressure law after geometric clipping.
      RealT lower;          ///< Exact lower endpoint of the penetrating interval.
      RealT upper;          ///< Exact upper endpoint of the penetrating interval.
      RealT total_force_y;  ///< Exact edge1 load for unit penalty and an affine compressive gap.
    };
    // Each case spans the physical interval x in [-1, 1]. Edge1 follows y=slope*x, and the oppositely directed edge2
    // follows y=separation-slope*x. Therefore, x=0 is the midpoint of their full projected overlap.
    const InterpenetratingOverlapCase cases[] = {
        // The edges cross at x=-0.5. Their geometric gap is positive at x=0, so only the left interval is retained.
        { 0.25, -0.25, 0.0, -1.0, -0.5, -0.0625 },
        // Mirroring both slopes moves the crossing to x=0.5 and retains only the right interval.
        { -0.25, -0.25, 0.0, 0.5, 1.0, -0.0625 },
        // The edges cross at x=0.5. Their geometric gap is negative at x=0, so the retained interval includes x=0.
        { 0.25, 0.25, 0.0, -1.0, 0.5, -0.5625 },
        // The first geometry is reused, but pressure uses the geometric gap minus 0.1 on the same clipped interval.
        { 0.25, -0.25, 0.1, -1.0, -0.5, -0.1125 } };
    for ( const auto test_case : cases ) {
      SCOPED_TRACE( testing::Message() << "points=" << NumPoints << ", slope=" << test_case.slope << ", separation="
                                       << test_case.separation << ", residual_gap=" << test_case.residual_gap );
      const RealT slope = test_case.slope;
      const RealT separation = test_case.separation;
      const RealT residual_gap = test_case.residual_gap;
      ArrayT<InterpenetratingOverlapResult<NumPoints>> overlap_results( 1, 1, allocatorId() );
      const auto overlap_results_view = overlap_results.view();
      forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
        const Coordinates edge1{ MdArray<RealT, 2>{ -1.0, -slope }, MdArray<RealT, 2>{ 1.0, slope } };
        const Coordinates edge2{ MdArray<RealT, 2>{ 1.0, separation - slope },
                                 MdArray<RealT, 2>{ -1.0, separation + slope } };
        const LinearEdgeGeometry geometry1( edge1.view() ), geometry2( edge2.view() );
        const PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>
            rule_generator;
        const auto quadrature_rule = rule_generator.generate( geometry1, geometry2 );
        const auto line = CommonPlaneEdgeMapping::computeCommonPlaneLine( geometry1, geometry2 );
        overlap_results_view[0].interval = CommonPlaneEdgeMapping{}.makeIntegrationSurface( geometry1, geometry2 );
        const PairTractionIntegral<
            PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>,
            CommonPlanePenaltyTraction<OffsetGap<CommonPlaneGap>>, BasisFunction<EDGE, LINEAR_LAGRANGE>>
            pair_integral{ rule_generator, { 1.0, OffsetGap{ CommonPlaneGap{}, residual_gap } } };
        overlap_results_view[0].residual = pair_integral.evaluate( edge1.view(), edge2.view() );
        for ( int point = 0; point < NumPoints; ++point ) {
          const auto point1 = geometry1.position( { &quadrature_rule.geom1_parametric_coords( point, 0 ), 1 } );
          const auto point2 = geometry2.position( { &quadrature_rule.geom2_parametric_coords( point, 0 ), 1 } );
          overlap_results_view[0].x1[point] = point1[0];
          overlap_results_view[0].x2[point] = point2[0];
          overlap_results_view[0].gaps[point] = CommonPlaneGap{}.computeGap( point1, point2, line );
          overlap_results_view[0].weights[point] = quadrature_rule.weights[point];
        }
      } );
      const ArrayT<InterpenetratingOverlapResult<NumPoints>, 1, MemorySpace::Host> host_overlap_results(
          overlap_results );
      const auto& overlap_result = host_overlap_results[0];
      ASSERT_TRUE( overlap_result.interval.valid );
      EXPECT_NEAR( overlap_result.interval.bounds[0], test_case.lower, tolerance );
      EXPECT_NEAR( overlap_result.interval.bounds[1], test_case.upper, tolerance );
      ASSERT_EQ( overlap_result.residual.quadrature_metadata.status, QuadratureStatus::Success );
      EXPECT_EQ( overlap_result.residual.quadrature_metadata.num_required_points, NumPoints );
      EXPECT_TRUE( overlap_result.residual.has_active_points );
      RealT measure = 0.0;
      RealT first_moment = 0.0;
      for ( int point = 0; point < NumPoints; ++point ) {
        EXPECT_GT( overlap_result.x1[point], test_case.lower );
        EXPECT_LT( overlap_result.x1[point], test_case.upper );
        EXPECT_NEAR( overlap_result.x1[point], overlap_result.x2[point], tolerance );
        EXPECT_LT( overlap_result.gaps[point], 0.0 );
        EXPECT_GT( overlap_result.weights[point], 0.0 );
        measure += overlap_result.weights[point];
        first_moment += overlap_result.weights[point] * overlap_result.x1[point];
      }
      const RealT center = 0.5 * ( test_case.lower + test_case.upper );
      EXPECT_NEAR( measure, test_case.upper - test_case.lower, tolerance );
      EXPECT_NEAR( first_moment / measure, center, tolerance );
      // The one-point rule applies centroid pressure. Two or more points integrate N_i(x) * (2*slope*x-offset)
      // exactly, including the pressure variation's contribution to the first moment of the load.
      RealT load_moment = center * test_case.total_force_y;
      if constexpr ( NumPoints >= 2 ) {
        const RealT length = test_case.upper - test_case.lower;
        load_moment += test_case.slope * length * length * length / 6.0;
      }
      const MdArray<RealT, 4> expected_nodal_y_loads{
          0.5 * ( test_case.total_force_y - load_moment ), 0.5 * ( test_case.total_force_y + load_moment ),
          -0.5 * ( test_case.total_force_y + load_moment ), -0.5 * ( test_case.total_force_y - load_moment ) };
      const auto residual_values = overlap_result.residual.values();
      for ( int node = 0; node < expected_nodal_y_loads.size(); ++node ) {
        EXPECT_NEAR( residual_values[2 * node], 0.0, tolerance );
        EXPECT_NEAR( residual_values[2 * node + 1], expected_nodal_y_loads[node], tolerance );
      }
    }
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

// Rigidly rotating and translating an edge pair must preserve total force and moment balance.
TEST_P( CommonPlaneTest, balances_rotated_pair_forces_and_moments )
{
  constexpr RealT edge_length = 5.0;
  constexpr RealT edge_offset = 0.25;
  constexpr RealT rotated_edge_x_component = 3.0;
  constexpr RealT rotated_edge_y_component = 4.0;
  constexpr RealT rotation_cosine = rotated_edge_x_component / edge_length;
  constexpr RealT rotation_sine = rotated_edge_y_component / edge_length;
  constexpr RealT penalty_stiffness = 2.0;
  constexpr int pair_node_count = 2 * LinearEdgeGeometry::num_vertices;

  // Begin with two oppositely directed length-five edges separated by 0.25 in their normal direction.
  const Coordinates reference_edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ edge_length, 0.0 } };
  const Coordinates reference_edge2{ MdArray<RealT, 2>{ edge_length, edge_offset },
                                     MdArray<RealT, 2>{ 0.0, edge_offset } };
  // This matrix maps the reference edge vector (5,0) to (3,4) without changing its length.
  const MdArray<RealT, 2, 2> rotation{ MdArray<RealT, 2>{ rotation_cosine, -rotation_sine },
                                       MdArray<RealT, 2>{ rotation_sine, rotation_cosine } };
  const MdArray<RealT, 2> translation{ 1.0, 2.0 };
  const auto transform = [&]( const Coordinates& reference ) {
    Coordinates transformed{};
    for ( int node = 0; node < LinearEdgeGeometry::num_vertices; ++node ) {
      for ( int row = 0; row < LinearEdgeGeometry::spatial_dim; ++row ) {
        transformed( node, row ) = translation[row];
        for ( int column = 0; column < LinearEdgeGeometry::spatial_dim; ++column ) {
          transformed( node, row ) += rotation( row, column ) * reference( node, column );
        }
      }
    }
    return transformed;
  };
  const Coordinates edge1 = transform( reference_edge1 );
  const Coordinates edge2 = transform( reference_edge2 );

  using Integral =
      PairTractionIntegral<PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<3>>,
                           CommonPlanePenaltyTraction<CommonPlaneGap>, BasisFunction<EDGE, LINEAR_LAGRANGE>>;
  ArrayT<Integral::PairResidual> pair_residuals( 1, 1, allocatorId() );
  const auto pair_residuals_view = pair_residuals.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<3>> rule_generator;
    const Integral pair_integral{ rule_generator, { penalty_stiffness } };
    pair_residuals_view[0] = pair_integral.evaluate( edge1.view(), edge2.view() );
  } );
  const ArrayT<Integral::PairResidual, 1, MemorySpace::Host> host_pair_residuals( pair_residuals );
  const auto pair_residual_values = host_pair_residuals[0].values();
  RealT force_x = 0.0, force_y = 0.0, moment = 0.0;
  const Coordinates edges[] = { edge1, edge2 };
  for ( int pair_node = 0; pair_node < pair_node_count; ++pair_node ) {
    const int edge = pair_node / LinearEdgeGeometry::num_vertices;
    const int edge_node = pair_node % LinearEdgeGeometry::num_vertices;
    force_x += pair_residual_values[2 * pair_node];
    force_y += pair_residual_values[2 * pair_node + 1];
    moment += edges[edge]( edge_node, 0 ) * pair_residual_values[2 * pair_node + 1] -
              edges[edge]( edge_node, 1 ) * pair_residual_values[2 * pair_node];
  }
  const RealT reference_nodal_force_y = -penalty_stiffness * edge_offset * edge_length / 2.0;
  EXPECT_NEAR( pair_residual_values[0], -rotation_sine * reference_nodal_force_y, tolerance );
  EXPECT_NEAR( pair_residual_values[1], rotation_cosine * reference_nodal_force_y, tolerance );
  EXPECT_NEAR( force_x, 0.0, tolerance );
  EXPECT_NEAR( force_y, 0.0, tolerance );
  EXPECT_NEAR( moment, 0.0, tolerance );
}

// Failed geometry is distinct from valid empty overlap, and insufficient capacity must leave all storage untouched.
TEST_P( CommonPlaneTest, reports_geometry_and_capacity_without_writing )
{
  enum GenerationCase
  {
    InvalidFirstGeometry,
    InvalidSecondGeometry,
    Disjoint,
    Cooriented,
    Perpendicular,
    InsufficientCapacity,
    NumCases
  };
  constexpr RealT unwritten = -7.0;
  ArrayT<QuadratureMetadata> quadrature_metadata( NumCases, NumCases, allocatorId() );
  ArrayT<RealT> point_storage( 4, 4, allocatorId() );
  const auto quadrature_metadata_view = quadrature_metadata.view();
  const auto point_storage_view = point_storage.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    // The baseline pair is two oppositely directed, fully overlapping horizontal edges. The alternatives collapse one
    // edge, move the second edge beyond x=2, reverse it to the same direction, or rotate it perpendicular to edge1.
    const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.0 } };
    const Coordinates edge2{ MdArray<RealT, 2>{ 2.0, 0.1 }, MdArray<RealT, 2>{ 0.0, 0.1 } };
    const Coordinates invalid{};
    const Coordinates disjoint{ MdArray<RealT, 2>{ 5.0, 0.1 }, MdArray<RealT, 2>{ 3.0, 0.1 } };
    const Coordinates cooriented{ MdArray<RealT, 2>{ 0.0, 0.1 }, MdArray<RealT, 2>{ 2.0, 0.1 } };
    const Coordinates perpendicular{ MdArray<RealT, 2>{ 1.0, -1.0 }, MdArray<RealT, 2>{ 1.0, 1.0 } };
    const PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<3>> rule_generator;
    quadrature_metadata_view[InvalidFirstGeometry] =
        rule_generator.generate( LinearEdgeGeometry( invalid.view() ), LinearEdgeGeometry( edge2.view() ) ).metadata;
    quadrature_metadata_view[InvalidSecondGeometry] =
        rule_generator.generate( LinearEdgeGeometry( edge1.view() ), LinearEdgeGeometry( invalid.view() ) ).metadata;
    quadrature_metadata_view[Disjoint] =
        rule_generator.generate( LinearEdgeGeometry( edge1.view() ), LinearEdgeGeometry( disjoint.view() ) ).metadata;
    quadrature_metadata_view[Cooriented] =
        rule_generator.generate( LinearEdgeGeometry( edge1.view() ), LinearEdgeGeometry( cooriented.view() ) ).metadata;
    quadrature_metadata_view[Perpendicular] =
        rule_generator.generate( LinearEdgeGeometry( edge1.view() ), LinearEdgeGeometry( perpendicular.view() ) )
            .metadata;
    for ( int i = 0; i < point_storage_view.size(); ++i ) {
      point_storage_view[i] = unwritten;
    }
    const PairQuadratureRuleView<1, 1> limited_capacity_rule_view{ { point_storage_view.data() + 1, { 2, 1 } },
                                                                   { point_storage_view.data() + 1, { 2, 1 } },
                                                                   { point_storage_view.data() + 1, 2 } };
    quadrature_metadata_view[InsufficientCapacity] = rule_generator.generate(
        LinearEdgeGeometry( edge1.view() ), LinearEdgeGeometry( edge2.view() ), limited_capacity_rule_view );
  } );
  const ArrayT<QuadratureMetadata, 1, MemorySpace::Host> host_quadrature_metadata( quadrature_metadata );
  const ArrayT<RealT, 1, MemorySpace::Host> host_point_storage( point_storage );
  for ( int invalid_geometry : { InvalidFirstGeometry, InvalidSecondGeometry } ) {
    EXPECT_EQ( host_quadrature_metadata[invalid_geometry].status, QuadratureStatus::InvalidGeometry );
    EXPECT_EQ( host_quadrature_metadata[invalid_geometry].num_required_points, 0 );
  }
  for ( int empty_overlap : { Disjoint, Cooriented, Perpendicular } ) {
    EXPECT_EQ( host_quadrature_metadata[empty_overlap].status, QuadratureStatus::Success );
    EXPECT_EQ( host_quadrature_metadata[empty_overlap].num_required_points, 0 );
  }
  EXPECT_EQ( host_quadrature_metadata[InsufficientCapacity].status, QuadratureStatus::InsufficientCapacity );
  EXPECT_EQ( host_quadrature_metadata[InsufficientCapacity].num_required_points, 3 );
  for ( const auto value : host_point_storage ) {
    EXPECT_EQ( value, unwritten );
  }
}

// Each base rule integrates only the penetrating interval; higher-order rules resolve the varying pressure.
TEST_P( CommonPlaneTest, integrates_interpenetrating_overlap_with_configured_rule )
{
  checkInterpenetratingOverlap<1>();
  checkInterpenetratingOverlap<2>();
  checkInterpenetratingOverlap<3>();
}
/** @brief Interpolate fixed nodal pressures and integrate them even on an opening pair. */
struct NodalPressureTraction {
  MdArray<RealT, 2> pressures{ 1.0, 3.0 };  ///< Fixed nodal values on geometry2.

  /** @brief Use geometry2's unit normal for this pair.
   * @param geometry1 First geometry, unused by this prescribed traction.
   * @param geometry2 Second current geometry.
   * @return Current traction direction.
   */
  TRIBOL_HOST_DEVICE auto computePairGeometryData( const LinearEdgeGeometry& TRIBOL_UNUSED_PARAM( geometry1 ),
                                                   const LinearEdgeGeometry& geometry2 ) const
  {
    return geometry2.normal();
  }

  /** @brief Interpolate pressure from geometry2's parametric point without a gap activation check.
   * @param point1 First physical point, unused.
   * @param point2 Second physical point, unused.
   * @param parametric_coords1 First parametric point, unused.
   * @param parametric_coords2 Point used to interpolate the fixed nodal pressures.
   * @param normal Current unit traction direction.
   * @return Active, unweighted vector traction.
   */
  TRIBOL_HOST_DEVICE QuadraturePointValue<MdArray<RealT, 2>> computeTractionAtQuadraturePoint(
      const LinearEdgeGeometry::Point& TRIBOL_UNUSED_PARAM( point1 ),
      const LinearEdgeGeometry::Point& TRIBOL_UNUSED_PARAM( point2 ),
      MdSpan<const RealT, 1> TRIBOL_UNUSED_PARAM( parametric_coords1 ), MdSpan<const RealT, 1> parametric_coords2,
      const LinearEdgeGeometry::Vector& normal ) const
  {
    const RealT xi = parametric_coords2[0];
    const RealT pressure = 0.5 * ( ( 1.0 - xi ) * pressures[0] + ( 1.0 + xi ) * pressures[1] );
    return { MdArray<RealT, 2>{ pressure * normal[0], pressure * normal[1] }, true };
  }
};

// Independent residual bases and a pressure field must work without penalty-specific point activation. Geometry2
// lies below geometry1, giving an opening Common Plane gap, but the prescribed pressure still contributes.
TEST_P( CommonPlaneTest, integrates_supplied_traction_with_independent_bases )
{
  using RuleGenerator = PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<2>>;
  using Integral = PairTractionIntegral<RuleGenerator, NodalPressureTraction, BasisFunction<EDGE, LINEAR_LAGRANGE>,
                                        BasisFunction<EDGE, QUADRATIC_LAGRANGE>>;
  static_assert( PairFunctionType<Integral> );
  ArrayT<Integral::PairResidual> pair_residuals( 1, 1, allocatorId() );
  const auto pair_residuals_view = pair_residuals.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const Coordinates edge1{ MdArray<RealT, 2>{ -1.0, 0.0 }, MdArray<RealT, 2>{ 1.0, 0.0 } };
    const Coordinates edge2{ MdArray<RealT, 2>{ 1.0, -0.1 }, MdArray<RealT, 2>{ -1.0, -0.1 } };
    const Integral pair_integral{ {}, {} };
    pair_residuals_view[0] = pair_integral.evaluate( edge1.view(), edge2.view() );
  } );
  const ArrayT<Integral::PairResidual, 1, MemorySpace::Host> host_pair_residuals( pair_residuals );
  const auto& pair_residual = host_pair_residuals[0];
  ASSERT_EQ( pair_residual.quadrature_metadata.status, QuadratureStatus::Success );
  EXPECT_EQ( pair_residual.quadrature_metadata.num_required_points, 2 );
  EXPECT_TRUE( pair_residual.has_active_points );
  // Geometry1's pressure is 2-x. Geometry2's reversed coordinate gives 2+xi and endpoint/endpoint/midpoint ordering.
  const MdArray<RealT, 2> expected_geometry1_y_loads{ 7.0 / 3.0, 5.0 / 3.0 };
  const MdArray<RealT, 3> expected_geometry2_y_loads{ -1.0 / 3.0, -1.0, -8.0 / 3.0 };
  for ( IndexT node = 0; node < expected_geometry1_y_loads.size(); ++node ) {
    EXPECT_NEAR( pair_residual.geometry1( node, 0 ), 0.0, tolerance );
    EXPECT_NEAR( pair_residual.geometry1( node, 1 ), expected_geometry1_y_loads[node], tolerance );
  }
  for ( IndexT node = 0; node < expected_geometry2_y_loads.size(); ++node ) {
    EXPECT_NEAR( pair_residual.geometry2( node, 0 ), 0.0, tolerance );
    EXPECT_NEAR( pair_residual.geometry2( node, 1 ), expected_geometry2_y_loads[node], tolerance );
  }
}

// Subdivided common-plane quadrature must integrate quadratic penalty tractions onto the quadratic basis on every
// enabled execution backend.
TEST_P( CommonPlaneTest, integrates_penalty_traction_with_quadratic_geometry_and_basis )
{
  using Integral = ProjectedQuadraticCommonPlaneIntegral<1>;
  using RuleGenerator = ProjectedQuadraticCommonPlaneRuleGenerator<1>;

  ArrayT<Integral::PairResidual> pair_residuals( 1, 1, allocatorId() );
  const auto pair_residuals_view = pair_residuals.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    // Two straight, oppositely directed quadratic edges spanning x in [-1, 1] at y=0.0 and y=0.1 (penetration 0.1),
    // giving closed-form nodal loads under two 1-point subdivisions.
    const MdArray<RealT, 3, 2> edge1{ MdArray<RealT, 2>{ -1.0, 0.0 }, MdArray<RealT, 2>{ 1.0, 0.0 },
                                      MdArray<RealT, 2>{ 0.0, 0.0 } };
    const MdArray<RealT, 3, 2> edge2{ MdArray<RealT, 2>{ 1.0, 0.1 }, MdArray<RealT, 2>{ -1.0, 0.1 },
                                      MdArray<RealT, 2>{ 0.0, 0.1 } };
    const Integral pair_integral{ RuleGenerator{ default_num_quadrature_subdivisions }, { 1.0 } };
    pair_residuals_view[0] = pair_integral.evaluate( edge1.view(), edge2.view() );
  } );

  const ArrayT<Integral::PairResidual, 1, MemorySpace::Host> host_pair_residuals( pair_residuals );
  const auto& residual = host_pair_residuals[0];
  ASSERT_EQ( residual.quadrature_metadata.status, QuadratureStatus::Success );
  ASSERT_EQ( residual.quadrature_metadata.num_required_points, default_num_quadrature_subdivisions );
  ASSERT_TRUE( residual.has_active_points );
  const MdArray<RealT, 3> expected_geometry1_y{ -0.025, -0.025, -0.15 };
  for ( IndexT node = 0; node < expected_geometry1_y.size(); ++node ) {
    EXPECT_NEAR( residual.geometry1( node, 0 ), 0.0, tolerance );
    EXPECT_NEAR( residual.geometry1( node, 1 ), expected_geometry1_y[node], tolerance );
    EXPECT_NEAR( residual.geometry2( node, 0 ), 0.0, tolerance );
    EXPECT_NEAR( residual.geometry2( node, 1 ), -expected_geometry1_y[node], tolerance );
  }
}

// Curved, tilted, interpenetrating quadratic edges must produce pair residuals with zero net force and zero net moment
// for every subdivision count. The averaged-normal traction on affinely mapped points is checked to show the moment
// test detects the defect that the projected configuration removes.
TEST_P( CommonPlaneTest, conserves_linear_and_angular_momentum_on_curved_quadratic_edges )
{
  using Integral = ProjectedQuadraticCommonPlaneIntegral<3>;
  using RuleGenerator = ProjectedQuadraticCommonPlaneRuleGenerator<3>;
  using AffineRuleGenerator = QuadraticCommonPlaneRuleGenerator<3, AffineSubelementMapping>;
  using AffineIntegral = PairTractionIntegral<AffineRuleGenerator, CommonPlanePenaltyTraction<CommonPlaneGap>,
                                              BasisFunction<EDGE, QUADRATIC_LAGRANGE>>;
  // Edge1 bulges upward; edge2 is rotated by 0.15 radians and bulges downward, so the curved edges interpenetrate with
  // varying gap and the moment arm of each paired point differs from its neighbors'.
  const QuadraticCoordinates edge1{ MdArray<RealT, 2>{ 1.0, 0.0 }, MdArray<RealT, 2>{ -1.0, 0.0 },
                                    MdArray<RealT, 2>{ 0.05, 0.12 } };
  const QuadraticCoordinates edge2{ MdArray<RealT, 2>{ -0.8, -0.085 }, MdArray<RealT, 2>{ 0.98, 0.184 },
                                    MdArray<RealT, 2>{ 0.205, -0.025 } };
  constexpr RealT penalty = 1.0;
  // Exact balance leaves only roundoff relative to the largest nodal load.
  constexpr RealT relative_balance_tolerance = 1.0e-12;
  // The affine configuration's moment is a few percent of the largest nodal load at one subdivision.
  constexpr RealT minimum_detectable_relative_moment = 1.0e-4;
  for ( const int num_subdivisions : { 1, 2, max_num_quadrature_subdivisions } ) {
    SCOPED_TRACE( num_subdivisions );
    ArrayT<Integral::PairResidual> pair_residuals( 1, 1, allocatorId() );
    const auto pair_residuals_view = pair_residuals.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      const Integral pair_integral{ RuleGenerator{ num_subdivisions }, { penalty } };
      pair_residuals_view[0] = pair_integral.evaluate( edge1.view(), edge2.view() );
    } );
    const ArrayT<Integral::PairResidual, 1, MemorySpace::Host> host_pair_residuals( pair_residuals );
    const auto& residual = host_pair_residuals[0];
    ASSERT_EQ( residual.quadrature_metadata.status, QuadratureStatus::Success );
    ASSERT_TRUE( residual.has_active_points );
    const auto balance = computeNetForceAndMoment( residual, edge1, edge2 );
    ASSERT_GT( balance.max_nodal_load, 0.0 );
    const RealT balance_tolerance = relative_balance_tolerance * balance.max_nodal_load;
    EXPECT_NEAR( balance.force[0], 0.0, balance_tolerance );
    EXPECT_NEAR( balance.force[1], 0.0, balance_tolerance );
    EXPECT_NEAR( balance.moment, 0.0, balance_tolerance );

    if ( num_subdivisions == 1 ) {
      const AffineIntegral affine_integral{ AffineRuleGenerator{ num_subdivisions }, { penalty } };
      const auto affine_residual = affine_integral.evaluate( edge1.view(), edge2.view() );
      ASSERT_EQ( affine_residual.quadrature_metadata.status, QuadratureStatus::Success );
      const auto affine_balance = computeNetForceAndMoment( affine_residual, edge1, edge2 );
      EXPECT_GT( std::abs( affine_balance.moment ),
                 minimum_detectable_relative_moment * affine_balance.max_nodal_load );
    }
  }
}

// Quadratic penalty traction must compute the local gap and activation from pointwise normals, returning active
// compressive traction only when the local normals are valid, opposing, and interpenetrating.
TEST( QuadraticCommonPlaneTraction, uses_true_local_normals_for_gap_and_activation )
{
  // Baseline horizontal edge on x in [-1, 1] at y=0.0, paired at the midpoint (xi=0) against four edge2 variants:
  // interpenetrated (opposing at y=0.1), separated (opposing at y=-0.1), nonopposing (same direction at y=0.1), and
  // degenerate (both endpoints at (0, 0.1) with midpoint at (1, 0.1), giving a zero tangent at xi=0).
  const QuadraticCoordinates edge1{ MdArray<RealT, 2>{ -1.0, 0.0 }, MdArray<RealT, 2>{ 1.0, 0.0 },
                                    MdArray<RealT, 2>{ 0.0, 0.0 } };
  const QuadraticCoordinates interpenetrated_edge2{ MdArray<RealT, 2>{ 1.0, 0.1 }, MdArray<RealT, 2>{ -1.0, 0.1 },
                                                    MdArray<RealT, 2>{ 0.0, 0.1 } };
  const QuadraticCoordinates separated_edge2{ MdArray<RealT, 2>{ 1.0, -0.1 }, MdArray<RealT, 2>{ -1.0, -0.1 },
                                              MdArray<RealT, 2>{ 0.0, -0.1 } };
  const QuadraticCoordinates nonopposing_edge2{ MdArray<RealT, 2>{ -1.0, 0.1 }, MdArray<RealT, 2>{ 1.0, 0.1 },
                                                MdArray<RealT, 2>{ 0.0, 0.1 } };
  const QuadraticCoordinates zero_tangent_edge2{ MdArray<RealT, 2>{ 0.0, 0.1 }, MdArray<RealT, 2>{ 0.0, 0.1 },
                                                 MdArray<RealT, 2>{ 1.0, 0.1 } };
  const QuadraticEdgeGeometry geometry1( edge1.view() );
  const MdArray<RealT, 1> midpoint_coordinate{ 0.0 };
  const CommonPlanePenaltyTraction<CommonPlaneGap> traction{ 1.0 };
  const auto evaluateMidpointTraction = [&]( const QuadraticCoordinates& edge2 ) {
    const QuadraticEdgeGeometry geometry2( edge2.view() );
    return traction.computeTractionAtQuadraturePoint( geometry1.position( midpoint_coordinate.view() ),
                                                      geometry2.position( midpoint_coordinate.view() ),
                                                      midpoint_coordinate.view(), midpoint_coordinate.view(),
                                                      traction.computePairGeometryData( geometry1, geometry2 ) );
  };

  const auto interpenetrated = evaluateMidpointTraction( interpenetrated_edge2 );
  ASSERT_TRUE( interpenetrated.active );
  EXPECT_NEAR( interpenetrated.value[0], 0.0, tolerance );
  EXPECT_NEAR( interpenetrated.value[1], -0.1, tolerance );
  EXPECT_FALSE( evaluateMidpointTraction( separated_edge2 ).active );
  EXPECT_FALSE( evaluateMidpointTraction( nonopposing_edge2 ).active );
  EXPECT_FALSE( evaluateMidpointTraction( zero_tangent_edge2 ).active );

  // The projected traction penalizes the separation itself and uses the local normals only for activation.
  const ProjectedCommonPlanePenaltyTraction projected_traction{ 1.0 };
  const auto evaluateProjectedMidpointTraction = [&]( const QuadraticCoordinates& edge2 ) {
    const QuadraticEdgeGeometry geometry2( edge2.view() );
    return projected_traction.computeTractionAtQuadraturePoint(
        geometry1.position( midpoint_coordinate.view() ), geometry2.position( midpoint_coordinate.view() ),
        midpoint_coordinate.view(), midpoint_coordinate.view(),
        projected_traction.computePairGeometryData( geometry1, geometry2 ) );
  };
  const auto projected_interpenetrated = evaluateProjectedMidpointTraction( interpenetrated_edge2 );
  ASSERT_TRUE( projected_interpenetrated.active );
  EXPECT_NEAR( projected_interpenetrated.value[0], 0.0, tolerance );
  EXPECT_NEAR( projected_interpenetrated.value[1], -0.1, tolerance );
  EXPECT_FALSE( evaluateProjectedMidpointTraction( separated_edge2 ).active );
  EXPECT_FALSE( evaluateProjectedMidpointTraction( nonopposing_edge2 ).active );
  EXPECT_FALSE( evaluateProjectedMidpointTraction( zero_tangent_edge2 ).active );
}

// A gap offset opens the centroid while leaving one of two Gauss points compressive. Only that point contributes.
// A larger opening offset makes both points inactive and must return zero residuals.
TEST_P( CommonPlaneTest, activates_each_quadrature_point_after_gap_offset )
{
  const RealT active_x = -0.5 - 0.5 / std::sqrt( 3.0 );
  for ( const RealT residual_gap : { -0.3, -0.5 } ) {
    SCOPED_TRACE( residual_gap );
    ArrayT<CommonPlaneResidualValue> pair_residuals( 1, 1, allocatorId() );
    const auto pair_residuals_view = pair_residuals.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      const Coordinates edge1{ MdArray<RealT, 2>{ -1.0, -0.25 }, MdArray<RealT, 2>{ 1.0, 0.25 } };
      const Coordinates edge2{ MdArray<RealT, 2>{ 1.0, -0.25 }, MdArray<RealT, 2>{ -1.0, 0.25 } };
      const PairTractionIntegral<PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<2>>,
                                 CommonPlanePenaltyTraction<OffsetGap<CommonPlaneGap>>,
                                 BasisFunction<EDGE, LINEAR_LAGRANGE>>
          pair_integral{ {}, { 1.0, OffsetGap{ CommonPlaneGap{}, residual_gap } } };
      pair_residuals_view[0] = pair_integral.evaluate( edge1.view(), edge2.view() );
    } );
    const ArrayT<CommonPlaneResidualValue, 1, MemorySpace::Host> host_pair_residuals( pair_residuals );
    const auto& pair_residual = host_pair_residuals[0];
    ASSERT_EQ( pair_residual.quadrature_metadata.status, QuadratureStatus::Success );
    EXPECT_EQ( pair_residual.quadrature_metadata.num_required_points, 2 );
    const RealT gap = 0.5 * active_x - residual_gap;
    const bool has_active_point = gap <= 0.0;
    EXPECT_EQ( pair_residual.has_active_points, has_active_point );
    // The clipped interval is [-1, 0], so each Gauss weight is 0.5 and N_0(x)=(1-x)/2.
    const RealT weighted_pressure = has_active_point ? 0.5 * gap : 0.0;
    const RealT first_load = 0.5 * ( 1.0 - active_x ) * weighted_pressure;
    const RealT second_load = 0.5 * ( 1.0 + active_x ) * weighted_pressure;
    const MdArray<RealT, 4> expected_nodal_y_loads{ first_load, second_load, -second_load, -first_load };
    const auto residual_values = pair_residual.values();
    for ( int node = 0; node < expected_nodal_y_loads.size(); ++node ) {
      EXPECT_NEAR( residual_values[2 * node], 0.0, tolerance );
      EXPECT_NEAR( residual_values[2 * node + 1], expected_nodal_y_loads[node], tolerance );
    }
  }
}

/** @brief Supply a nonzero entry even for inactive or invalid pairs to check pair-output masking. */
struct UnmaskedResidualPairFunction
    : PairTractionIntegral<PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<2>>,
                           CommonPlanePenaltyTraction<CommonPlaneGap>, BasisFunction<EDGE, LINEAR_LAGRANGE>> {
  using PairTractionIntegral<PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<2>>,
                             CommonPlanePenaltyTraction<CommonPlaneGap>,
                             BasisFunction<EDGE, LINEAR_LAGRANGE>>::PairTractionIntegral;

  /** @brief Evaluate the pair and overwrite one residual entry regardless of state.
   * @param coords1 First edge coordinates. @param coords2 Second edge coordinates.
   * @return Residual with its original quadrature status and activation, and a nonzero first entry. */
  TRIBOL_HOST_DEVICE PairResidual evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    auto pair_residual =
        PairTractionIntegral<PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<2>>,
                             CommonPlanePenaltyTraction<CommonPlaneGap>,
                             BasisFunction<EDGE, LINEAR_LAGRANGE>>::evaluate( coords1, coords2 );
    pair_residual.geometry1( 0, 0 ) = 1.0;
    return pair_residual;
  }
};

// Force-only evaluation must keep pair slots distinct and preserve active zero-gap and invalid-geometry state.
TEST_P( CommonPlaneTest, value_evaluation_returns_forces_and_quadrature_without_derivatives )
{
  using PairIntegral =
      PairTractionIntegral<PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<2>>,
                           CommonPlanePenaltyTraction<CommonPlaneGap>, BasisFunction<EDGE, LINEAR_LAGRANGE>>;
  const PairIntegral pair_integral{ {}, { 1.0 } };
  // Mesh1 contains one horizontal edge at y=0. Mesh2 contains four oppositely directed candidates: a penetrating edge
  // at y=0.1, an open edge at y=-0.1, a coincident edge, and a collapsed edge with both nodes at the origin.
  const auto mesh1_x = copyToExecutionSpace<RealT>( { 0.0, 2.0 } );
  const auto mesh1_y = copyToExecutionSpace<RealT>( { 0.0, 0.0 } );
  const auto mesh1_connectivity = copyToExecutionSpace<IndexT>( { 0, 1 } );
  const auto mesh2_x = copyToExecutionSpace<RealT>( { 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0 } );
  const auto mesh2_y = copyToExecutionSpace<RealT>( { 0.1, 0.1, -0.1, -0.1, 0.0, 0.0, 0.0, 0.0 } );
  const auto mesh2_connectivity = copyToExecutionSpace<IndexT>( { 0, 1, 2, 3, 4, 5, 6, 7 } );
  MeshData mesh1( 0, 1, mesh1_x.size(), mesh1_connectivity.data(), EDGE, LINEAR_LAGRANGE, mesh1_x.data(),
                  mesh1_y.data(), nullptr, memorySpace() );
  MeshData mesh2( 1, 4, mesh2_x.size(), mesh2_connectivity.data(), EDGE, LINEAR_LAGRANGE, mesh2_x.data(),
                  mesh2_y.data(), nullptr, memorySpace() );
  const auto pair_list = copyToExecutionSpace<InterfacePair>( { { 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, 3 } } );
  auto pair_value_buffer = dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    return evaluatePairValues<Exec>( pair_list, mesh1.getView(), mesh2.getView(), pair_integral );
  } );
  EXPECT_EQ( pair_value_buffer.numberOfPairs(), pair_list.size() );
  EXPECT_EQ( pair_value_buffer.values().getAllocatorID(), pair_list.getAllocatorID() );
  EXPECT_EQ( pair_value_buffer.quadratureMetadata().getAllocatorID(), pair_list.getAllocatorID() );
  const Array1D<RealT, MemorySpace::Host> host_pair_values( pair_value_buffer.values() );
  const Array1D<QuadratureMetadata, MemorySpace::Host> host_quadrature_metadata(
      pair_value_buffer.quadratureMetadata() );
  for ( IndexT pair = 0; pair < pair_list.size(); ++pair ) {
    EXPECT_EQ( host_quadrature_metadata[pair].status,
               pair == 3 ? QuadratureStatus::InvalidGeometry : QuadratureStatus::Success );
    EXPECT_EQ( host_quadrature_metadata[pair].num_required_points, pair == 3 ? 0 : 2 );
    for ( IndexT row = 0; row < PairIntegral::size_values; ++row ) {
      const RealT expected_pair_value =
          pair == 0 && row % 2 == 1 ? ( row < PairIntegral::size_geom1_coords ? -0.1 : 0.1 ) : 0.0;
      EXPECT_NEAR( host_pair_values[row * pair_list.size() + pair], expected_pair_value, tolerance );
    }
  }
  EXPECT_SLIC_ERROR( dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    return checkPairQuadratureMetadata<Exec>( pair_value_buffer.quadratureMetadata() );
  } ) );
  // A second calculation uses different stiffness and must leave the first returned buffer unchanged.
  auto doubled_stiffness_buffer = dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    return evaluatePairValues<Exec>( pair_list, mesh1.getView(), mesh2.getView(), PairIntegral{ {}, { 2.0 } } );
  } );
  const Array1D<RealT, MemorySpace::Host> host_doubled_stiffness_values( doubled_stiffness_buffer.values() );
  const Array1D<RealT, MemorySpace::Host> host_preserved_pair_values( pair_value_buffer.values() );
  for ( IndexT i = 0; i < host_pair_values.size(); ++i ) {
    EXPECT_EQ( host_preserved_pair_values[i], host_pair_values[i] );
    EXPECT_NEAR( host_doubled_stiffness_values[i], 2.0 * host_pair_values[i], tolerance );
  }
  // Pair evaluation must zero inactive or failed outputs even when evaluate() supplies nonzero numerical entries.
  auto masked_pair_value_buffer = dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    return evaluatePairValues<Exec>( pair_list, mesh1.getView(), mesh2.getView(),
                                     UnmaskedResidualPairFunction{ {}, { 1.0 } } );
  } );
  const Array1D<RealT, MemorySpace::Host> host_masked_values( masked_pair_value_buffer.values() );
  for ( IndexT pair = 0; pair < pair_list.size(); ++pair ) {
    EXPECT_EQ( host_masked_values[pair], pair == 0 || pair == 2 ? 1.0 : 0.0 );
  }
  const ArrayT<InterfacePair> empty_pairs( 0, 0, allocatorId() );
  auto empty_pair_buffer = dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    return evaluatePairValues<Exec>( empty_pairs, mesh1.getView(), mesh2.getView(), pair_integral );
  } );
  EXPECT_EQ( empty_pair_buffer.numberOfPairs(), 0 );
  EXPECT_EQ( empty_pair_buffer.values().size(), 0 );
  EXPECT_EQ( empty_pair_buffer.quadratureMetadata().size(), 0 );
  EXPECT_NO_THROW( dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    return checkPairQuadratureMetadata<Exec>( empty_pair_buffer.quadratureMetadata() );
  } ) );
}
INSTANTIATE_TEST_SUITE_P( execution, CommonPlaneTest, tribol::detail::enabledExecutionModes() );

#ifdef TRIBOL_USE_ENZYME
/** @brief Compare primal values and all coordinate derivatives of a pair integral against central differences. */
template <typename PairIntegral, typename Coords>
void checkPairJacobian( const PairIntegral& pair_integral, const Coords& edge1, const Coords& edge2,
                        RealT derivative_tolerance )
{
  // Small enough to resolve derivatives without crossing active-gap or overlap-endpoint branches.
  constexpr RealT step = 2.0e-7;
  const auto pair_coordinates = packPairCoordinates( edge1.view(), edge2.view() );
  const auto reference_residual = pair_integral.evaluate( edge1.view(), edge2.view() );
  const auto differentiated_residual =
      EnzymePairDerivative<PairIntegral>::valuesAndJacobian( edge1.view(), edge2.view(), pair_integral );
  ASSERT_EQ( differentiated_residual.pair_value.quadrature_metadata.status, QuadratureStatus::Success );
  EXPECT_EQ( differentiated_residual.pair_value.quadrature_metadata.num_required_points,
             reference_residual.quadrature_metadata.num_required_points );
  EXPECT_EQ( differentiated_residual.pair_value.has_active_points, reference_residual.has_active_points );
  const auto reference_residual_values = reference_residual.values();
  const auto differentiated_residual_values = differentiated_residual.pair_value.values();
  for ( IndexT row = 0; row < PairIntegral::size_values; ++row ) {
    EXPECT_DOUBLE_EQ( differentiated_residual_values[row], reference_residual_values[row] );
  }
  for ( int column = 0; column < PairIntegral::size_coords; ++column ) {
    auto increased_coordinate_pair = pair_coordinates;
    auto decreased_coordinate_pair = pair_coordinates;
    increased_coordinate_pair[column] += step;
    decreased_coordinate_pair[column] -= step;
    const auto increased_coordinate_residual = pair_integral.evaluate(
        { increased_coordinate_pair.data(), PairIntegral::size_geom1_coords },
        { increased_coordinate_pair.data() + PairIntegral::size_geom1_coords, PairIntegral::size_geom2_coords } );
    const auto decreased_coordinate_residual = pair_integral.evaluate(
        { decreased_coordinate_pair.data(), PairIntegral::size_geom1_coords },
        { decreased_coordinate_pair.data() + PairIntegral::size_geom1_coords, PairIntegral::size_geom2_coords } );
    ASSERT_EQ( increased_coordinate_residual.quadrature_metadata.status, QuadratureStatus::Success );
    ASSERT_EQ( decreased_coordinate_residual.quadrature_metadata.status, QuadratureStatus::Success );
    ASSERT_EQ( increased_coordinate_residual.quadrature_metadata.num_required_points,
               differentiated_residual.pair_value.quadrature_metadata.num_required_points );
    ASSERT_EQ( decreased_coordinate_residual.quadrature_metadata.num_required_points,
               differentiated_residual.pair_value.quadrature_metadata.num_required_points );
    const auto increased_values = increased_coordinate_residual.values();
    const auto decreased_values = decreased_coordinate_residual.values();
    for ( int row = 0; row < PairIntegral::size_values; ++row ) {
      EXPECT_NEAR( differentiated_residual.derivatives.jacobian( row, column ),
                   ( increased_values[row] - decreased_values[row] ) / ( 2.0 * step ), derivative_tolerance )
          << "row=" << row << ", column=" << column;
    }
  }
}

/** @brief Compare active-geometry linear Jacobians with central differences and check zero-gap and invalid cases.
 * @tparam NumPoints Gauss point count. */
template <int NumPoints>
void checkJacobian()
{
  using PairIntegral =
      PairTractionIntegral<PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>,
                           CommonPlanePenaltyTraction<OffsetGap<CommonPlaneGap>>, BasisFunction<EDGE, LINEAR_LAGRANGE>>;
  const PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>> rule_generator;
  // The longer second edge in the offset case keeps the limiting overlap endpoint unchanged under perturbation.
  const Coordinates edge1_cases[] = { Coordinates{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.2 } },
                                      Coordinates{ MdArray<RealT, 2>{ -1.0, 0.0 }, MdArray<RealT, 2>{ 1.0, 0.0 } },
                                      Coordinates{ MdArray<RealT, 2>{ -1.0, -0.25 }, MdArray<RealT, 2>{ 1.0, 0.25 } } };
  const Coordinates edge2_cases[] = { Coordinates{ MdArray<RealT, 2>{ 1.8, 0.35 }, MdArray<RealT, 2>{ 0.2, 0.1 } },
                                      Coordinates{ MdArray<RealT, 2>{ 0.8, 0.2 }, MdArray<RealT, 2>{ -0.8, -0.2 } },
                                      Coordinates{ MdArray<RealT, 2>{ 1.2, -0.3 }, MdArray<RealT, 2>{ -1.2, 0.3 } } };
  const MdArray<RealT, 3> residual_gaps{ 0.03, 0.03, -0.3 };
  for ( int case_index = 0; case_index < residual_gaps.size(); ++case_index ) {
    SCOPED_TRACE( testing::Message() << "points=" << NumPoints << ", case=" << case_index );
    const PairIntegral pair_integral{ rule_generator,
                                      { 2.5, OffsetGap{ CommonPlaneGap{}, residual_gaps[case_index] } } };
    EXPECT_EQ( pair_integral.evaluate( edge1_cases[case_index].view(), edge2_cases[case_index].view() )
                   .quadrature_metadata.num_required_points,
               NumPoints );
    checkPairJacobian( pair_integral, edge1_cases[case_index], edge2_cases[case_index], 2.0e-7 );
  }
  const Coordinates edge1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.0 } };
  const Coordinates edge2{ MdArray<RealT, 2>{ 2.0, 0.0 }, MdArray<RealT, 2>{ 0.0, 0.0 } };
  using UnshiftedIntegral =
      PairTractionIntegral<PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>,
                           CommonPlanePenaltyTraction<CommonPlaneGap>, BasisFunction<EDGE, LINEAR_LAGRANGE>>;
  const UnshiftedIntegral unshifted_pair_integral{ rule_generator, { 1.0 } };
  const auto zero_gap_residual = unshifted_pair_integral.evaluate( edge1.view(), edge2.view() );
  EXPECT_TRUE( zero_gap_residual.has_active_points );
  const auto zero_gap_derivatives =
      EnzymePairDerivative<UnshiftedIntegral>::valuesAndJacobian( edge1.view(), edge2.view(), unshifted_pair_integral );
  EXPECT_TRUE( zero_gap_derivatives.pair_value.has_active_points );
  const auto zero_gap_residual_values = zero_gap_derivatives.pair_value.values();
  for ( IndexT row = 0; row < UnshiftedIntegral::size_values; ++row ) {
    EXPECT_EQ( zero_gap_residual_values[row], 0.0 );
  }
  EXPECT_GT( zero_gap_derivatives.derivatives.jacobian( 1, 1 ), 0.0 );
  const Coordinates invalid{};
  const auto invalid_geometry_derivatives = EnzymePairDerivative<UnshiftedIntegral>::valuesAndJacobian(
      invalid.view(), edge2.view(), unshifted_pair_integral );
  EXPECT_EQ( invalid_geometry_derivatives.pair_value.quadrature_metadata.status, QuadratureStatus::InvalidGeometry );
  EXPECT_EQ( invalid_geometry_derivatives.pair_value.quadrature_metadata.num_required_points, 0 );
  EXPECT_FALSE( invalid_geometry_derivatives.pair_value.has_active_points );
  const auto invalid_geometry_residual_values = invalid_geometry_derivatives.pair_value.values();
  for ( int row = 0; row < invalid_geometry_residual_values.size(); ++row ) {
    EXPECT_EQ( invalid_geometry_residual_values[row], 0.0 );
    for ( IndexT column = 0; column < UnshiftedIntegral::size_coords; ++column ) {
      EXPECT_EQ( invalid_geometry_derivatives.derivatives.jacobian( row, column ), 0.0 );
    }
  }
}

// All rules differentiate current normals, interval bounds, edge lengths, and pointwise pressure. A shifted gap
// checks mixed open/compressive quadrature points without crossing their activation boundaries under perturbation.
TEST( CommonPlaneJacobian, matches_finite_differences )
{
  checkJacobian<1>();
  checkJacobian<2>();
  checkJacobian<3>();

  using QuadraticPairIntegral = ProjectedQuadraticCommonPlaneIntegral<1>;
  using RuleGenerator = ProjectedQuadraticCommonPlaneRuleGenerator<1>;
  // Two curved, non-parallel, interpenetrating quadratic edges whose midpoints are bowed upward and shifted so every
  // nodal coordinate contributes to the subdivided overlap bounds, local normals, and pointwise gaps.
  const QuadraticCoordinates edge1{ MdArray<RealT, 2>{ -1.0, 0.0 }, MdArray<RealT, 2>{ 1.0, 0.1 },
                                    MdArray<RealT, 2>{ 0.0, 0.25 } };
  const QuadraticCoordinates edge2{ MdArray<RealT, 2>{ 1.3, 0.35 }, MdArray<RealT, 2>{ -1.1, 0.4 },
                                    MdArray<RealT, 2>{ 0.15, 0.55 } };
  const QuadraticPairIntegral pair_integral{ RuleGenerator{ default_num_quadrature_subdivisions }, { 2.5 } };
  ASSERT_TRUE( pair_integral.evaluate( edge1.view(), edge2.view() ).has_active_points );
  checkPairJacobian( pair_integral, edge1, edge2, 3.0e-7 );
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
