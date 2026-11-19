// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Verify Energy Mortar pair integrals and automatic derivatives independently of MFEM assembly.
 *
 * Direct backend tests cover energy and nodal-gap values, quadrature state, contact activation, derivative evaluation,
 * and pair-buffer ownership. Compile-time checks also verify the interfaces required by derivative evaluation.
 */

#include <cmath>

#include "gtest/gtest.h"
#include "SlicErrorTest.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "ExecutionTest.hpp"
#include "tribol/enforcement/PenaltyEnergy.hpp"
#include "tribol/physics/QuadraturePointEnergyMortar.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/geom/OpposingNormalGap.hpp"
#include "tribol/physics/integrals/PairIntegrals.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"
#include "tribol/assembly/ElementBlockStorage.hpp"
#include "tribol/assembly/EnzymePairDerivative.hpp"
#include "tribol/assembly/EvaluatePairs.hpp"
#include "tribol/assembly/PairBuffers.hpp"
#include "tribol/assembly/PairFunctionData.hpp"

namespace {
using namespace tribol;
constexpr RealT tolerance = 1.0e-12;  ///< Roundoff allowance for primal integrals and exact partials.
using EnergyMortarPairEvaluationTest = tribol::detail::ExecutionTest;

/** @brief Generic fixed-size values and pair state for checking the pair-function interface. @tparam SizeValues Size.
 */
template <IndexT SizeValues>
struct TestPairValue {
  static constexpr IndexT size_values = SizeValues;  ///< Number of scalar function values.
  MdArray<RealT, SizeValues> function_values{};      ///< Values without residual-layout semantics.
  QuadratureMetadata quadrature_metadata{};          ///< Quadrature metadata supplied by the test function.
  bool has_active_points = false;                    ///< Whether the test function reports active integration points.

  /** @brief Return the function value vector. @return Fixed-size test values. */
  TRIBOL_HOST_DEVICE MdArray<RealT, SizeValues> values() const { return function_values; }

  /** @brief Return writable flattened value storage for derivative tests. @return Function value storage. */
  TRIBOL_HOST_DEVICE RealT* data() { return function_values.data(); }
};

/** @brief Declaration-only coordinate pair function for checking the interface without executing it. */
struct PairFunctionCandidate {
  using Coords1 = MdSpan<const RealT, 2>;                                       ///< Two geometry1 coordinates.
  using Coords2 = MdSpan<const RealT, 2>;                                       ///< Two geometry2 coordinates.
  static constexpr IndexT size_values = 2;                                      ///< Two output values.
  static constexpr IndexT size_geom1_coords = Coords1::size();                  ///< Geometry1 coordinate count.
  static constexpr IndexT size_geom2_coords = Coords2::size();                  ///< Geometry2 coordinate count.
  static constexpr IndexT size_coords = size_geom1_coords + size_geom2_coords;  ///< Combined coordinate count.

  /** @brief Declare a const evaluation. @param coords1 Geometry1 inputs. @param coords2 Geometry2 inputs.
   * @return Values and quadrature state. */
  TestPairValue<size_values> evaluate( Coords1 coords1, Coords2 coords2 ) const;
};

/** @brief Zero outputs must be rejected before instantiating fixed-size numerical arrays. */
struct ZeroValuePairFunction : PairFunctionCandidate {
  static constexpr IndexT size_values = 0;  ///< Invalid output count.
};

/** @brief Zero coordinates must be rejected before instantiating a coordinate view. */
struct ZeroCoordinatePairFunction : PairFunctionCandidate {
  static constexpr IndexT size_coords = 0;  ///< Invalid coordinate count.
};

/** @brief Output counts must be compile-time constants. */
struct RuntimeSizePairFunction : PairFunctionCandidate {
  IndexT size_values = 2;  ///< Invalid runtime output count.
};

/** @brief The returned value array must agree with the declared output count. */
struct WrongNumValuesPairFunction : PairFunctionCandidate {
  static constexpr IndexT size_values = 1;  ///< Disagrees with the inherited two-entry result.
};

/** @brief Pair evaluation must be callable on a const function object. */
struct MutablePairFunction : PairFunctionCandidate {
  /** @brief Declare a mutable-only evaluation. @param coords1 Geometry1 inputs. @param coords2 Geometry2 inputs.
   * @return Values and quadrature state. */
  TestPairValue<size_values> evaluate( Coords1 coords1, Coords2 coords2 );
};

/** @brief An array alone supplies no quadrature status or activation. */
struct MissingStatePairFunction : PairFunctionCandidate {
  /** @brief Declare a result without quadrature state. @param coords1 Geometry1 inputs. @param coords2 Geometry2
   * inputs. @return Numerical values only.
   */
  MdArray<RealT, size_values> evaluate( Coords1 coords1, Coords2 coords2 ) const;
};

/** @brief The numerical values must be readable from a const evaluation result. */
struct MutableValuesPairFunction : PairFunctionCandidate {
  /** @brief Result that hides the const values() accessor with a mutable-only accessor. */
  struct Result : TestPairValue<size_values> {
    /** @brief Declare mutable-only value access. @return Pair values. */
    MdArray<RealT, size_values> values();
  };
  /** @brief Declare an incompatible result. @param coords1 Geometry1 inputs. @param coords2 Geometry2 inputs. @return
   * Mutable-only numerical values. */
  Result evaluate( Coords1 coords1, Coords2 coords2 ) const;
};

/** @brief Activation must have the expected bool type. */
struct WrongActivationPairFunction : PairFunctionCandidate {
  /** @brief Result with an integer flag that hides the required bool member. */
  struct Result : TestPairValue<size_values> {
    IndexT has_active_points = 0;  ///< Incompatible activation type.
  };
  /** @brief Declare an incompatible result. @param coords1 Geometry1 inputs. @param coords2 Geometry2 inputs. @return
   * Values with an integer flag. */
  Result evaluate( Coords1 coords1, Coords2 coords2 ) const;
};

/** @brief Quadrature state must have the expected QuadratureMetadata type. */
struct WrongQuadraturePairFunction : PairFunctionCandidate {
  /** @brief Result whose status member omits the required point count. */
  struct Result : TestPairValue<size_values> {
    QuadratureStatus quadrature_metadata{};  ///< Incompatible quadrature metadata type.
  };
  /** @brief Declare an incompatible result. @param coords1 Geometry1 inputs. @param coords2 Geometry2 inputs. @return
   * Values without a point count. */
  Result evaluate( Coords1 coords1, Coords2 coords2 ) const;
};

/** @brief A coordinate split must leave entries for geometry1. */
struct EmptyFirstGeometryPairFunction : PairFunctionCandidate {
  static constexpr IndexT size_geom1_coords = 0;  ///< Invalid empty geometry1.
};

/** @brief A coordinate split must leave entries for geometry2. */
struct EmptySecondGeometryPairFunction : PairFunctionCandidate {
  static constexpr IndexT size_geom2_coords = 0;  ///< Invalid empty geometry2.
};

/** @brief A value-only pair function need not expose writable numerical storage. */
struct ReadOnlyPairFunction : PairFunctionCandidate {
  /** @brief Valid pair-function result without the writable interface required by derivative providers. */
  struct Result : TestPairValue<size_values> {
    RealT* data() = delete;
  };

  /** @brief Declare a read-only evaluation. @param coords1 Geometry1 inputs. @param coords2 Geometry2 inputs.
   * @return Values and pair state without writable flattened storage. */
  Result evaluate( Coords1 coords1, Coords2 coords2 ) const;
};

// Coordinate differentiation supports functions without geometry types.
static_assert( PairFunctionType<PairFunctionCandidate> );
static_assert( DifferentiablePairFunctionType<PairFunctionCandidate> );
static_assert( PairFunctionType<ReadOnlyPairFunction> );
static_assert( !DifferentiablePairFunctionType<ReadOnlyPairFunction> );
static_assert( !PairFunctionType<ZeroValuePairFunction> );
static_assert( !PairFunctionType<ZeroCoordinatePairFunction> );
static_assert( !PairFunctionType<RuntimeSizePairFunction> );
static_assert( !PairFunctionType<WrongNumValuesPairFunction> );
static_assert( !PairFunctionType<MutablePairFunction> );
static_assert( !PairFunctionType<MissingStatePairFunction> );
static_assert( !PairFunctionType<MutableValuesPairFunction> );
static_assert( !PairFunctionType<WrongActivationPairFunction> );
static_assert( !PairFunctionType<WrongQuadraturePairFunction> );
static_assert( !PairFunctionType<EmptyFirstGeometryPairFunction> );
static_assert( !PairFunctionType<EmptySecondGeometryPairFunction> );

using ProjectedEdgeRuleGenerator =
    PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>;
using PenaltyEnergy = PenaltyEnergyDensity<OpposingNormalGap>;

static_assert( PairEnergyFunction<PenaltyEnergy, ProjectedEdgeRuleGenerator> );
static_assert( PairGapFunction<OpposingNormalGap, ProjectedEdgeRuleGenerator> );
static_assert( !PairTractionFunction<PenaltyEnergy, ProjectedEdgeRuleGenerator> );
static_assert( !PairEnergyFunction<OpposingNormalGap, ProjectedEdgeRuleGenerator> );

// Energy and nodal-integral functions accept two coordinate views with fixed sizes.
static_assert( PairFunctionType<PairEnergyIntegral<ProjectedEdgeRuleGenerator, PenaltyEnergy>> );
static_assert( PairFunctionType<PairNodalGapIntegral<ProjectedEdgeRuleGenerator, OpposingNormalGap,
                                                     BasisFunction<EDGE, LINEAR_LAGRANGE>>> );

/** @brief One-coordinate geometry caching a nonlinear coordinate expression. */
struct SquaredCoordinateGeometry {
  using Coords = MdSpan<const RealT, 1>;  ///< One coordinate input.
  RealT squared;                          ///< Cached coordinate square.
  /** @brief Cache the squared coordinate. @param coordinates Single coordinate. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE explicit SquaredCoordinateGeometry( Coords coordinates )
      : squared( coordinates[0] * coordinates[0] )
  {
  }
};

/** @brief Two-coordinate geometry caching a product. */
struct CoordinateProductGeometry {
  using Coords = MdSpan<const RealT, 2>;  ///< Two coordinate inputs.
  RealT product;                          ///< Cached coordinate product.
  /** @brief Cache the coordinate product. @param coordinates Two coordinates. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE explicit CoordinateProductGeometry( Coords coordinates )
      : product( coordinates[0] * coordinates[1] )
  {
  }
};

/** @brief Scalar pair function using distinct geometry types and unequal coordinate counts. */
struct CachedGeometryPairFunction {
  using Geometry1 = SquaredCoordinateGeometry;                                  ///< Single-coordinate geometry.
  using Geometry2 = CoordinateProductGeometry;                                  ///< Two-coordinate geometry.
  using Coords1 = Geometry1::Coords;                                            ///< Geometry1 coordinate view.
  using Coords2 = Geometry2::Coords;                                            ///< Geometry2 coordinate view.
  static constexpr IndexT size_values = 1;                                      ///< One scalar value.
  static constexpr IndexT size_geom1_coords = Coords1::size();                  ///< Geometry1 coordinate count.
  static constexpr IndexT size_geom2_coords = Coords2::size();                  ///< Geometry2 coordinate count.
  static constexpr IndexT size_coords = size_geom1_coords + size_geom2_coords;  ///< Combined coordinate count.

  /** @brief Multiply the cached expressions. @param coords1 Coordinate to square. @param coords2 Coordinates to
   * multiply.
   * @return Active value with successful one-point quadrature state. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairEnergyValue evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    const Geometry1 geometry1( coords1 );
    const Geometry2 geometry2( coords2 );
    return { geometry1.squared * geometry2.product, true, { QuadratureStatus::Success, 1 } };
  }
};
static_assert( PairFunctionType<CachedGeometryPairFunction> );

/** @brief Pair energy and nodal integrals written by one backend thread. */
struct ParallelIntegralResult {
  PairEnergyValue energy;      ///< Quadrature-point energy and status.
  PairNodalGapValue<2> nodal;  ///< Nodal integrals and status.
};

/** @brief Nodal field inputs and exact penalty partials for one backend test case. */
struct NodalPartialsCase {
  RealT weighted_gap;    ///< Weighted nodal weighted_gap.
  RealT tributary_area;  ///< Tributary tributary_area, including unsupported values on both sides of zero.
  EnergyMortarNodalPenaltyPartials expected;  ///< Exact energy and field partials for stiffness four.
};

/** @brief Check constant-gap integrals on an unsmoothed interval. @tparam NumPoints Gauss point count.
 * @param mode Execution backend. @param allocator Backend-accessible allocation resource. */
template <int NumPoints>
void checkParallelIntegrals( ExecutionMode mode, int allocator )
{
  using RuleGenerator =
      PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>;
  using Energy = PairEnergyIntegral<RuleGenerator, PenaltyEnergyDensity<OpposingNormalGap>>;
  using Nodal = PairNodalGapIntegral<RuleGenerator, OpposingNormalGap, BasisFunction<EDGE, LINEAR_LAGRANGE>>;
  ArrayT<ParallelIntegralResult> integral_results( 1, 1, allocator );
  const auto integral_result_view = integral_results.view();
  forAllExec( mode, 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    // A length-two lower edge is covered by a longer, oppositely directed upper edge at a constant separation of 0.1.
    const MdArray<RealT, 2, 2> coordinates1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.0 } };
    const MdArray<RealT, 2, 2> coordinates2{ MdArray<RealT, 2>{ 3.0, 0.1 }, MdArray<RealT, 2>{ -1.0, 0.1 } };
    const RuleGenerator rule_generator( SmoothedProjectedEdgeMapping{ 0.0 } );
    integral_result_view[0] = { Energy{ rule_generator, { 4.0 } }.evaluate( coordinates1.view(), coordinates2.view() ),
                                Nodal{ rule_generator }.evaluate( coordinates1.view(), coordinates2.view() ) };
  } );
  const ArrayT<ParallelIntegralResult, 1, MemorySpace::Host> host_integral_results( integral_results );
  ASSERT_EQ( host_integral_results[0].energy.quadrature_metadata.status, QuadratureStatus::Success );
  EXPECT_EQ( host_integral_results[0].energy.quadrature_metadata.num_required_points, NumPoints );
  // A length-two edge with gap -0.1 has energy 0.5*4*0.1^2*2, and equal nodal areas of one.
  EXPECT_NEAR( host_integral_results[0].energy.energy, 0.04, tolerance );
  EXPECT_TRUE( host_integral_results[0].energy.has_active_points );
  ASSERT_EQ( host_integral_results[0].nodal.quadrature_metadata.status, QuadratureStatus::Success );
  for ( IndexT node = 0; node < Nodal::num_nodes; ++node ) {
    EXPECT_NEAR( host_integral_results[0].nodal.weightedGap( node ), -0.1, tolerance );
    EXPECT_NEAR( host_integral_results[0].nodal.tributaryArea( node ), 1.0, tolerance );
  }
}

// Direct nodal and quadrature-point calculations must use physical weights exactly once for every Gauss count.
TEST_P( EnergyMortarPairEvaluationTest, integrates_constant_gap )
{
  checkParallelIntegrals<1>( GetParam(), allocatorId() );
  checkParallelIntegrals<2>( GetParam(), allocatorId() );
  checkParallelIntegrals<3>( GetParam(), allocatorId() );
}

// A residual gap changes the gap value and contact activation without changing the integration surface.
TEST( EnergyMortarPairValues, residual_gap_shifts_nodal_integrals_and_activates_energy )
{
  using RuleGenerator = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>;
  using Gap = OffsetGap<OpposingNormalGap>;
  using Nodal = PairNodalGapIntegral<RuleGenerator, Gap, BasisFunction<EDGE, LINEAR_LAGRANGE>>;
  using Energy = PairEnergyIntegral<RuleGenerator, PenaltyEnergyDensity<Gap>>;
  constexpr RealT separation = 0.1;
  constexpr RealT residual_gap = 0.15;
  constexpr RealT penalty = 3.0;
  constexpr RealT comparison_tolerance = 1.0e-14;
  // Oppositely directed horizontal edges overlap from x=0.2 to x=0.8 and are separated by 0.1. The 0.15 residual gap
  // therefore changes the open geometric gap into an active penalty gap without changing the overlap interval.
  const MdArray<RealT, Nodal::size_coords> coordinates{ 1.0, 0.0, 0.0, 0.0, 0.2, separation, 0.8, separation };
  const typename Nodal::Coords1 coords1{ coordinates.data(), Nodal::size_geom1_coords };
  const typename Nodal::Coords2 coords2{ coordinates.data() + Nodal::size_geom1_coords, Nodal::size_geom2_coords };
  const RuleGenerator rule_generator{ SmoothedProjectedEdgeMapping{ 0.0 } };
  const Nodal without_offset{ rule_generator, Gap{ OpposingNormalGap{}, 0.0 } };
  const Nodal with_offset{ rule_generator, Gap{ OpposingNormalGap{}, residual_gap } };
  const auto baseline = without_offset.evaluate( coords1, coords2 );
  const auto shifted = with_offset.evaluate( coords1, coords2 );
  ASSERT_EQ( baseline.quadrature_metadata.status, QuadratureStatus::Success );
  ASSERT_EQ( shifted.quadrature_metadata.status, QuadratureStatus::Success );
  ASSERT_TRUE( baseline.has_active_points );
  ASSERT_TRUE( shifted.has_active_points );
  for ( IndexT node = 0; node < Nodal::num_nodes; ++node ) {
    EXPECT_NEAR( shifted.tributaryArea( node ), baseline.tributaryArea( node ), comparison_tolerance );
    EXPECT_NEAR( shifted.weightedGap( node ),
                 baseline.weightedGap( node ) - residual_gap * baseline.tributaryArea( node ), comparison_tolerance );
  }

  const Energy open_energy{ rule_generator, { penalty, Gap{ OpposingNormalGap{}, 0.0 } } };
  const Energy active_energy{ rule_generator, { penalty, Gap{ OpposingNormalGap{}, residual_gap } } };
  const auto open = open_energy.evaluate( coords1, coords2 );
  const auto active = active_energy.evaluate( coords1, coords2 );
  ASSERT_EQ( open.quadrature_metadata.status, QuadratureStatus::Success );
  ASSERT_EQ( active.quadrature_metadata.status, QuadratureStatus::Success );
  EXPECT_FALSE( open.has_active_points );
  EXPECT_EQ( open.values()[0], 0.0 );
  EXPECT_TRUE( active.has_active_points );
  EXPECT_GT( active.values()[0], 0.0 );
}

// A short nonzero edge has a small area, but its normalized gap still equals its physical normal separation.
TEST( EnergyMortarPairValues, preserves_normal_gap_for_very_short_valid_edge )
{
  using RuleGenerator = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>;
  using Nodal = PairNodalGapIntegral<RuleGenerator, OpposingNormalGap, BasisFunction<EDGE, LINEAR_LAGRANGE>>;
  constexpr RealT edge_length = 1.0e-13;
  constexpr RealT separation = 0.25;
  constexpr RealT comparison_tolerance = 1.0e-12;
  // Two oppositely directed horizontal edges have the same very short length and a fixed normal separation.
  const MdArray<RealT, Nodal::size_coords> coordinates{ 0.0,         0.0,        edge_length, 0.0,
                                                        edge_length, separation, 0.0,         separation };
  const Nodal nodal{ RuleGenerator{ SmoothedProjectedEdgeMapping{ 0.0 } } };
  const auto result = nodal.evaluate( { coordinates.data(), Nodal::size_geom1_coords },
                                      { coordinates.data() + Nodal::size_geom1_coords, Nodal::size_geom2_coords } );
  ASSERT_EQ( result.quadrature_metadata.status, QuadratureStatus::Success );
  ASSERT_TRUE( result.has_active_points );
  for ( IndexT node = 0; node < Nodal::num_nodes; ++node ) {
    ASSERT_GT( result.tributaryArea( node ), 0.0 );
    EXPECT_NEAR( result.weightedGap( node ) / result.tributaryArea( node ), -separation, comparison_tolerance );
  }
}

// Zero area and open gaps give zero partials; tiny supported areas remain scale-independent.
TEST_P( EnergyMortarPairEvaluationTest, guards_zero_area_and_keeps_small_area_and_zero_gap_behavior )
{
  constexpr RealT penalty_stiffness = 4.0;
  constexpr RealT small_area = 1.0e-16;
  constexpr RealT small_weighted_gap = -0.5 * small_area;
  const auto cases = copyToExecutionSpace<NodalPartialsCase>(
      { { -1.0, 2.0, { 1.0, -2.0, -0.5, 2.0, 1.0, 0.5 } },
        { 0.0, 2.0, { 0.0, 0.0, 0.0, 2.0, 0.0, 0.0 } },
        { 1.0, 2.0, {} },
        { -1.0, 0.0, {} },
        { small_weighted_gap,
          small_area,
          { 0.5 * small_area, -2.0, -0.5, 4.0 / small_area, 2.0 / small_area, 1.0 / small_area } } } );
  ArrayT<EnergyMortarNodalPenaltyPartials> partial_results( cases.size(), cases.size(), allocatorId() );
  const auto case_view = cases.view();
  const auto partial_result_view = partial_results.view();
  forAllExec( GetParam(), cases.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
    partial_result_view[i] = EnergyMortarNodalPenaltyPartials::evaluate(
        case_view[i].weighted_gap, case_view[i].tributary_area, penalty_stiffness );
  } );
  const ArrayT<NodalPartialsCase, 1, MemorySpace::Host> host_cases( cases );
  const ArrayT<EnergyMortarNodalPenaltyPartials, 1, MemorySpace::Host> host_results( partial_results );
  for ( IndexT i = 0; i < cases.size(); ++i ) {
    SCOPED_TRACE( i );
    const auto& actual = host_results[i];
    const auto& expected = host_cases[i].expected;
    const auto expect_near = [&]( RealT value, RealT reference ) {
      EXPECT_NEAR( value, reference, tolerance * ( 1.0 + std::abs( reference ) ) );
    };
    expect_near( actual.energy, expected.energy );
    expect_near( actual.denergy_dweighted_gap, expected.denergy_dweighted_gap );
    expect_near( actual.denergy_dtributary_area, expected.denergy_dtributary_area );
    expect_near( actual.d2energy_dweighted_gap2, expected.d2energy_dweighted_gap2 );
    expect_near( actual.d2energy_dweighted_gap_dtributary_area, expected.d2energy_dweighted_gap_dtributary_area );
    expect_near( actual.d2energy_dtributary_area2, expected.d2energy_dtributary_area2 );
  }
}

// This test ensures pair values, Jacobians, element IDs, and matrix blocks retain their documented structure-of-arrays
// layout in every enabled memory space.
TEST_P( EnergyMortarPairEvaluationTest, contribution_buffers_use_backend_accessible_structure_of_arrays )
{
  constexpr IndexT num_pairs = 3;
  PairValueJacobianBuffer<2, 2> values( num_pairs, allocatorId() );
  ElementBlockStorage blocks( num_pairs, 2 * 2, allocatorId() );
  const auto value_jacobian_view = values.view();
  const auto block_view = blocks.view<2, 2>();
  forAllExec( GetParam(), num_pairs, [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
    value_jacobian_view.setValue( pair, 0, 10.0 + pair );
    value_jacobian_view.setValue( pair, 1, 20.0 + pair );
    value_jacobian_view.setJacobian( pair, 0, 0, 30.0 + pair );
    value_jacobian_view.setJacobian( pair, 0, 1, 40.0 + pair );
    value_jacobian_view.setJacobian( pair, 1, 0, 50.0 + pair );
    value_jacobian_view.setJacobian( pair, 1, 1, 60.0 + pair );
    block_view.setElements( pair, pair, pair + 4 );
    for ( IndexT column = 0; column < 2; ++column ) {
      for ( IndexT row = 0; row < 2; ++row ) {
        block_view.setValue( pair, row, column, 100.0 * pair + 10.0 * column + row );
      }
    }
  } );

  const Array1D<RealT, MemorySpace::Host> host_values( values.values() );
  const Array1D<RealT, MemorySpace::Host> host_jacobians( values.jacobians() );
  const Array1D<IndexT, MemorySpace::Host> host_rows( blocks.rowElementIds() );
  const Array1D<IndexT, MemorySpace::Host> host_columns( blocks.columnElementIds() );
  const Array1D<IndexT, MemorySpace::Host> host_offsets( blocks.valueOffsets() );
  const Array1D<RealT, MemorySpace::Host> host_blocks( blocks.values() );
  for ( IndexT pair = 0; pair < num_pairs; ++pair ) {
    EXPECT_DOUBLE_EQ( host_values[pair], 10.0 + pair );
    EXPECT_DOUBLE_EQ( host_values[num_pairs + pair], 20.0 + pair );
    EXPECT_DOUBLE_EQ( host_jacobians[pair], 30.0 + pair );
    EXPECT_DOUBLE_EQ( host_jacobians[num_pairs + pair], 40.0 + pair );
    EXPECT_DOUBLE_EQ( host_jacobians[2 * num_pairs + pair], 50.0 + pair );
    EXPECT_DOUBLE_EQ( host_jacobians[3 * num_pairs + pair], 60.0 + pair );
    EXPECT_EQ( host_rows[pair], pair );
    EXPECT_EQ( host_columns[pair], pair + 4 );
    EXPECT_EQ( host_offsets[pair], 4 * pair );
    for ( IndexT column = 0; column < 2; ++column ) {
      for ( IndexT row = 0; row < 2; ++row ) {
        EXPECT_DOUBLE_EQ( host_blocks[4 * pair + row + 2 * column], 100.0 * pair + 10.0 * column + row );
      }
    }
  }
}

#ifdef TRIBOL_USE_ENZYME

/**
 * @brief Check one pair function and weighted contraction against coordinate differences.
 * @tparam Function Fixed-size Energy Mortar pair function.
 * @param function Production pair function with geometry-dependent quadrature.
 * @param coordinates Current pair coordinates in Jacobian column order, away from overlap and activation transitions.
 * @param weights Output weights used for the Hessian contraction.
 */
template <typename Function>
void checkFiniteDifferences( const Function& function, const MdArray<RealT, Function::size_coords>& coordinates,
                             const MdArray<RealT, Function::size_values>& weights )
{
  constexpr RealT finite_difference_step = 1.0e-6;
  constexpr RealT jacobian_tolerance = 1.0e-6;
  constexpr RealT hessian_tolerance = 2.0e-5;
  const auto base = EnzymePairDerivative<Function>::valuesAndJacobian(
      { coordinates.data(), Function::size_geom1_coords },
      { coordinates.data() + Function::size_geom1_coords, Function::size_geom2_coords }, function );
  const auto weighted = EnzymePairDerivative<Function>::weightedJacobianAndHessian(
      { coordinates.data(), Function::size_geom1_coords },
      { coordinates.data() + Function::size_geom1_coords, Function::size_geom2_coords }, function, weights.view() );
  ASSERT_EQ( base.pair_value.quadrature_metadata.status, QuadratureStatus::Success );
  ASSERT_TRUE( base.pair_value.has_active_points );
  bool has_nonzero_jacobian = false;
  bool has_nonzero_hessian = false;
  for ( IndexT row = 0; row < Function::size_values; ++row ) {
    for ( IndexT column = 0; column < Function::size_coords; ++column ) {
      has_nonzero_jacobian = has_nonzero_jacobian || base.derivatives.jacobian( row, column ) != 0.0;
    }
  }
  for ( IndexT row = 0; row < Function::size_coords; ++row ) {
    for ( IndexT column = 0; column < Function::size_coords; ++column ) {
      has_nonzero_hessian = has_nonzero_hessian || weighted.hessian( row, column ) != 0.0;
    }
  }
  ASSERT_TRUE( has_nonzero_jacobian );
  ASSERT_TRUE( has_nonzero_hessian );
  for ( IndexT column = 0; column < Function::size_coords; ++column ) {
    auto positive_coordinates = coordinates;
    auto negative_coordinates = coordinates;
    positive_coordinates[column] += finite_difference_step;
    negative_coordinates[column] -= finite_difference_step;
    const auto positive = EnzymePairDerivative<Function>::valuesAndJacobian(
        { positive_coordinates.data(), Function::size_geom1_coords },
        { positive_coordinates.data() + Function::size_geom1_coords, Function::size_geom2_coords }, function );
    const auto negative = EnzymePairDerivative<Function>::valuesAndJacobian(
        { negative_coordinates.data(), Function::size_geom1_coords },
        { negative_coordinates.data() + Function::size_geom1_coords, Function::size_geom2_coords }, function );
    ASSERT_EQ( positive.pair_value.quadrature_metadata.status, QuadratureStatus::Success );
    ASSERT_TRUE( positive.pair_value.has_active_points );
    ASSERT_EQ( negative.pair_value.quadrature_metadata.status, QuadratureStatus::Success );
    ASSERT_TRUE( negative.pair_value.has_active_points );
    const auto positive_values = positive.pair_value.values();
    const auto negative_values = negative.pair_value.values();

    for ( IndexT value = 0; value < Function::size_values; ++value ) {
      EXPECT_NEAR( base.derivatives.jacobian( value, column ),
                   ( positive_values[value] - negative_values[value] ) / ( 2.0 * finite_difference_step ),
                   jacobian_tolerance );
    }
    for ( IndexT row = 0; row < Function::size_coords; ++row ) {
      RealT positive_gradient = 0.0;
      RealT negative_gradient = 0.0;
      for ( IndexT value = 0; value < Function::size_values; ++value ) {
        positive_gradient += weights[value] * positive.derivatives.jacobian( value, row );
        negative_gradient += weights[value] * negative.derivatives.jacobian( value, row );
      }
      EXPECT_NEAR( weighted.hessian( row, column ),
                   ( positive_gradient - negative_gradient ) / ( 2.0 * finite_difference_step ), hessian_tolerance );
    }
  }
}

/** @brief Check derivatives using production quadrature generation. @tparam NumPoints Gauss point count. */
template <int NumPoints>
void checkRuleGeneratorFiniteDifferences()
{
  using RuleGenerator =
      PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>;
  using Gap = OffsetGap<OpposingNormalGap>;
  constexpr RealT smoothing_fraction = 0.2;
  const RuleGenerator rule_generator{ SmoothedProjectedEdgeMapping{ smoothing_fraction } };
  const Gap gap{ {}, 0.05 };
  const PairEnergyIntegral<RuleGenerator, PenaltyEnergyDensity<Gap>> energy{ rule_generator, { 7.0, gap } };
  const PairNodalGapIntegral<RuleGenerator, Gap, BasisFunction<EDGE, LINEAR_LAGRANGE>> nodal{ rule_generator, gap };
  // The two oblique, oppositely directed edges overlap only through their interiors, placing both projected endpoints
  // inside the smoothing regions rather than on an exact endpoint branch.
  const MdArray<RealT, decltype( energy )::size_coords> coordinates{ 0.0, 0.0, 1.7, 0.1, 1.6, 0.25, 0.25, 0.15 };
  SCOPED_TRACE( testing::Message() << "Gauss points: " << NumPoints );
  checkFiniteDifferences( energy, coordinates, MdArray<RealT, decltype( energy )::size_values>{ 1.0 } );
  checkFiniteDifferences( nodal, coordinates, MdArray<RealT, decltype( nodal )::size_values>{ 0.3, -0.2, 0.4, 0.1 } );
}

// Each production Gauss rule must differentiate energy and nodal integrals through the smoothed overlap mapping.
TEST( EnergyMortarPairFiniteDifference, smoothing_branch_with_production_quadrature )
{
  checkRuleGeneratorFiniteDifferences<1>();
  checkRuleGeneratorFiniteDifferences<2>();
  checkRuleGeneratorFiniteDifferences<3>();
}

// Different geometry types must retain all cached coordinate dependencies through both orders of AD.
TEST_P( EnergyMortarPairEvaluationTest, differentiates_geometry_construction_with_unequal_coordinate_counts )
{
  using Function = CachedGeometryPairFunction;
  using DerivativeResult = PairValueJacobianHessian<PairFunctionValue<Function>, Function::size_coords>;
  ArrayT<DerivativeResult> derivative_results( 1, 1, allocatorId() );
  const auto derivative_result_view = derivative_results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const MdArray<RealT, Function::size_geom1_coords> coords1{ 2.0 };
    const MdArray<RealT, Function::size_geom2_coords> coords2{ 3.0, 4.0 };
    derivative_result_view[0] =
        EnzymePairDerivative<Function>::valueJacobianAndHessian( coords1.view(), coords2.view(), Function{} );
  } );
  const ArrayT<DerivativeResult, 1, MemorySpace::Host> host_derivative_results( derivative_results );
  const auto& derivative_result = host_derivative_results[0];
  ASSERT_EQ( derivative_result.pair_value.quadrature_metadata.status, QuadratureStatus::Success );
  EXPECT_EQ( derivative_result.pair_value.quadrature_metadata.num_required_points, 1 );
  EXPECT_TRUE( derivative_result.pair_value.has_active_points );
  EXPECT_DOUBLE_EQ( derivative_result.pair_value.energy, 48.0 );
  // For f(x,y,z)=x^2*y*z, both cached expressions contribute to the mixed second derivatives.
  const MdArray<RealT, Function::size_coords> jacobian{ 48.0, 16.0, 12.0 };
  const MdArray<RealT, Function::size_coords, Function::size_coords> hessian{
      MdArray<RealT, Function::size_coords>{ 24.0, 16.0, 12.0 },
      MdArray<RealT, Function::size_coords>{ 16.0, 0.0, 4.0 },
      MdArray<RealT, Function::size_coords>{ 12.0, 4.0, 0.0 } };
  for ( IndexT row = 0; row < Function::size_coords; ++row ) {
    EXPECT_DOUBLE_EQ( derivative_result.derivatives.jacobian[row], jacobian[row] );
    for ( IndexT column = 0; column < Function::size_coords; ++column ) {
      EXPECT_DOUBLE_EQ( derivative_result.derivatives.hessian( row, column ), hessian( row, column ) );
    }
  }
}

/** @brief Check whether mesh pair evaluation accepts a scalar pair function. @tparam Function Candidate pair function.
 */
template <typename Function>
concept CanEvaluateScalarPairs =
    requires { &evaluatePairValuesJacobiansAndHessians<ExecutionMode::Sequential, Function>; };

// Scalar pair evaluation accepts energy outputs and rejects vector nodal integrals at its interface.
static_assert(
    CanEvaluateScalarPairs<
        PairEnergyIntegral<PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>,
                           PenaltyEnergyDensity<OpposingNormalGap>>> );
static_assert(
    !CanEvaluateScalarPairs<
        PairNodalGapIntegral<PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>,
                             OpposingNormalGap, BasisFunction<EDGE, LINEAR_LAGRANGE>>> );

// Both evaluations must return values and quadrature status for active, open, zero-gap, and invalid pairs. A rigid
// vertical edge translation has analytic derivatives; zero gap retains activation and the compressive Hessian.
TEST_P( EnergyMortarPairEvaluationTest, energy_evaluation_returns_values_jacobians_and_hessians )
{
  using RuleGenerator = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>;
  using Function = PairEnergyIntegral<RuleGenerator, PenaltyEnergyDensity<OpposingNormalGap>>;
  const Function function{ RuleGenerator{ SmoothedProjectedEdgeMapping{ 0.0 } }, { 4.0 } };
  // Mesh1 contains one horizontal edge at y=0. Mesh2 contains four longer, oppositely directed candidates: a
  // penetrating edge at y=0.1, an open edge at y=-0.1, a coincident edge, and a collapsed edge at the origin.
  const auto x1 = copyToExecutionSpace<RealT>( { 0.0, 2.0 } );
  const auto y1 = copyToExecutionSpace<RealT>( { 0.0, 0.0 } );
  const auto conn1 = copyToExecutionSpace<IndexT>( { 0, 1 } );
  const auto x2 = copyToExecutionSpace<RealT>( { 3.0, -1.0, 3.0, -1.0, 3.0, -1.0, 0.0, 0.0 } );
  const auto y2 = copyToExecutionSpace<RealT>( { 0.1, 0.1, -0.1, -0.1, 0.0, 0.0, 0.0, 0.0 } );
  const auto conn2 = copyToExecutionSpace<IndexT>( { 0, 1, 2, 3, 4, 5, 6, 7 } );
  MeshData mesh1( 0, 1, x1.size(), conn1.data(), EDGE, LINEAR_LAGRANGE, x1.data(), y1.data(), nullptr, memorySpace() );
  MeshData mesh2( 1, 4, x2.size(), conn2.data(), EDGE, LINEAR_LAGRANGE, x2.data(), y2.data(), nullptr, memorySpace() );
  const auto pairs = copyToExecutionSpace<InterfacePair>( { { 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, 3 } } );
  auto buffer = dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    return evaluatePairValuesJacobiansAndHessians<Exec>( pairs, mesh1.getView(), mesh2.getView(), function );
  } );
  const auto contributions_view = buffer.view();
  auto value_jacobian = dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    return evaluatePairValuesAndJacobians<Exec>( pairs, mesh1.getView(), mesh2.getView(), function );
  } );
  EXPECT_EQ( contributions_view.derivatives.num_pairs, pairs.size() );
  EXPECT_EQ( buffer.quadratureMetadata().getAllocatorID(), pairs.getAllocatorID() );
  EXPECT_EQ( value_jacobian.numberOfPairs(), pairs.size() );
  EXPECT_EQ( value_jacobian.values().getAllocatorID(), pairs.getAllocatorID() );
  EXPECT_EQ( value_jacobian.jacobians().getAllocatorID(), pairs.getAllocatorID() );

  constexpr IndexT energy_column = 0;
  constexpr IndexT vertical_jacobian_column = 1;
  constexpr IndexT vertical_hessian_column = 2;
  constexpr IndexT derivative_norm_column = 3;
  constexpr IndexT num_summary_columns = 4;
  ArrayT<RealT, 2> derivative_summaries( axom::StackArray<axom::IndexType, 2>{ pairs.size(), num_summary_columns },
                                         allocatorId() );
  const auto derivative_summary_view = derivative_summaries.view();
  forAllExec( GetParam(), pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
    derivative_summary_view( pair, energy_column ) = contributions_view.value( pair );
    derivative_summary_view( pair, vertical_jacobian_column ) =
        contributions_view.jacobian( pair, 1 ) + contributions_view.jacobian( pair, 3 );
    derivative_summary_view( pair, vertical_hessian_column ) = 0.0;
    derivative_summary_view( pair, derivative_norm_column ) = 0.0;
    for ( IndexT row = 0; row < Function::size_coords; ++row ) {
      derivative_summary_view( pair, derivative_norm_column ) += std::abs( contributions_view.jacobian( pair, row ) );
      for ( IndexT column = 0; column < Function::size_coords; ++column ) {
        derivative_summary_view( pair, derivative_norm_column ) +=
            std::abs( contributions_view.hessian( pair, row, column ) );
        if ( ( row == 1 || row == 3 ) && ( column == 1 || column == 3 ) ) {
          derivative_summary_view( pair, vertical_hessian_column ) += contributions_view.hessian( pair, row, column );
        }
      }
    }
  } );
  const ArrayT<RealT, 2, MemorySpace::Host> host_derivative_summaries( derivative_summaries );
  const Array1D<QuadratureMetadata, MemorySpace::Host> derivative_metadata( buffer.quadratureMetadata() );
  const Array1D<RealT, MemorySpace::Host> values( value_jacobian.values() );
  const Array1D<RealT, MemorySpace::Host> jacobians( value_jacobian.jacobians() );
  const Array1D<QuadratureMetadata, MemorySpace::Host> jacobian_metadata( value_jacobian.quadratureMetadata() );
  for ( IndexT pair = 0; pair < pairs.size(); ++pair ) {
    EXPECT_NEAR( values[pair], host_derivative_summaries( pair, energy_column ), tolerance );
    EXPECT_NEAR( jacobians[pairs.size() + pair] + jacobians[3 * pairs.size() + pair],
                 host_derivative_summaries( pair, vertical_jacobian_column ), tolerance );
    EXPECT_EQ( jacobian_metadata[pair].status, derivative_metadata[pair].status );
    EXPECT_EQ( jacobian_metadata[pair].num_required_points, pair == 3 ? 0 : 2 );
  }
  // Length two and stiffness four give E=0.04, dE/dy=-0.8, and d2E/dy2=8 for the penetrating pair.
  EXPECT_NEAR( host_derivative_summaries( 0, energy_column ), 0.04, tolerance );
  EXPECT_NEAR( host_derivative_summaries( 0, vertical_jacobian_column ), -0.8, tolerance );
  EXPECT_NEAR( host_derivative_summaries( 0, vertical_hessian_column ), 8.0, tolerance );
  EXPECT_NEAR( host_derivative_summaries( 2, energy_column ), 0.0, tolerance );
  EXPECT_NEAR( host_derivative_summaries( 2, vertical_jacobian_column ), 0.0, tolerance );
  EXPECT_NEAR( host_derivative_summaries( 2, vertical_hessian_column ), 8.0, tolerance );
  for ( IndexT pair : { 0, 1, 2 } ) {
    EXPECT_EQ( derivative_metadata[pair].status, QuadratureStatus::Success );
  }
  EXPECT_EQ( derivative_metadata[3].status, QuadratureStatus::InvalidGeometry );
  for ( IndexT pair : { 1, 3 } ) {
    EXPECT_EQ( host_derivative_summaries( pair, energy_column ), 0.0 );
    EXPECT_EQ( host_derivative_summaries( pair, derivative_norm_column ), 0.0 );
    for ( IndexT coordinate = 0; coordinate < Function::size_coords; ++coordinate ) {
      EXPECT_EQ( jacobians[coordinate * pairs.size() + pair], 0.0 );
    }
  }
  EXPECT_SLIC_ERROR( dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    return checkPairQuadratureMetadata<Exec>( buffer.quadratureMetadata() );
  } ) );

  // Read derivatives after destroying the input pair list. Reversed element IDs and unequal weights check pair slots.
  auto weighted = dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    const auto weighted_pairs = copyToExecutionSpace<InterfacePair>( { { 0, 2 }, { 0, 0 } } );
    return evaluatePairWeightedJacobiansAndHessians<Exec>(
        weighted_pairs, mesh1.getView(), mesh2.getView(), function,
        [] TRIBOL_HOST_DEVICE( IndexT pair ) { return MdArray<RealT, Function::size_values>{ pair + 2.0 }; } );
  } );
  constexpr IndexT num_weighted_pairs = 2;
  EXPECT_EQ( weighted.numberOfPairs(), num_weighted_pairs );
  EXPECT_EQ( weighted.jacobians().getAllocatorID(), pairs.getAllocatorID() );
  EXPECT_EQ( weighted.hessians().getAllocatorID(), pairs.getAllocatorID() );
  const Array1D<RealT, MemorySpace::Host> weighted_jacobians( weighted.jacobians() );
  const Array1D<RealT, MemorySpace::Host> weighted_hessians( weighted.hessians() );
  for ( IndexT pair = 0; pair < num_weighted_pairs; ++pair ) {
    const IndexT original_pair = pair == 0 ? 2 : 0;
    const RealT weight = pair + 2.0;
    for ( IndexT coordinate = 0; coordinate < Function::size_coords; ++coordinate ) {
      EXPECT_NEAR( weighted_jacobians[coordinate * num_weighted_pairs + pair],
                   weight * jacobians[coordinate * pairs.size() + original_pair], tolerance );
    }
    RealT vertical_hessian = 0.0;
    for ( IndexT row : { 1, 3 } ) {
      for ( IndexT column : { 1, 3 } ) {
        vertical_hessian += weighted_hessians[( row * Function::size_coords + column ) * num_weighted_pairs + pair];
      }
    }
    EXPECT_NEAR( vertical_hessian, weight * host_derivative_summaries( original_pair, vertical_hessian_column ),
                 tolerance );
  }
}

/** @brief Small fixed-size pair function used to compile and execute both Enzyme operations on every backend. */
struct PolynomialPairFunction {
  using Coords1 = MdSpan<const RealT, 1>;                       ///< First scalar input.
  using Coords2 = MdSpan<const RealT, 1>;                       ///< Second scalar input.
  static constexpr IndexT size_geom1_coords = Coords1::size();  ///< Geometry1 coordinate count.
  static constexpr IndexT size_geom2_coords = Coords2::size();  ///< Geometry2 coordinate count.
  static constexpr IndexT size_values = 2;                      ///< Number of outputs.
  static constexpr IndexT size_coords = 2;                      ///< Number of active coordinates.
  IndexT* num_evaluations = nullptr;                            ///< Optional counter for complete function evaluations.

  /** @brief Evaluate two coupled polynomial outputs. @param coords1 First scalar input. @param coords2 Second scalar
   * input.
   * @return Function values with successful, active quadrature state. */
  TRIBOL_HOST_DEVICE TestPairValue<size_values> evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    if ( num_evaluations != nullptr ) {
      ++*num_evaluations;
    }
    return { MdArray<RealT, size_values>{ coords1[0] * coords1[0] + coords2[0], coords1[0] * coords2[0] },
             { QuadratureStatus::Success, 2 },
             true };
  }
};

/** @brief Device-storable output from both pair derivative operations. */
struct PolynomialDerivativeResult {
  PairValueJacobian<PairFunctionValue<PolynomialPairFunction>, PolynomialPairFunction::size_coords> value_jacobian;
  PairJacobianHessian<PolynomialPairFunction::size_coords> weighted_hessian;
  IndexT num_evaluations = 0;           ///< Complete evaluations used for values and Jacobian.
  IndexT num_weighted_evaluations = 0;  ///< Complete evaluations used for the weighted Jacobian and Hessian.
};

// Jacobian and Hessian AD calls must also supply their primal outputs without separate evaluations.
TEST_P( EnergyMortarPairEvaluationTest, enzyme_pair_derivatives_execute_on_backend )
{
  Array1D<PolynomialDerivativeResult> derivative_results( 1, 1, allocatorId() );
  auto derivative_result_view = derivative_results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const MdArray<RealT, 1> coords1{ 3.0 }, coords2{ 5.0 };
    const MdArray<RealT, PolynomialPairFunction::size_values> weights{ 2.0, -1.0 };
    IndexT num_evaluations = 0;
    derivative_result_view[0].value_jacobian = EnzymePairDerivative<PolynomialPairFunction>::valuesAndJacobian(
        coords1.view(), coords2.view(), PolynomialPairFunction{ &num_evaluations } );
    derivative_result_view[0].num_evaluations = num_evaluations;
    IndexT num_weighted_evaluations = 0;
    derivative_result_view[0].weighted_hessian =
        EnzymePairDerivative<PolynomialPairFunction>::weightedJacobianAndHessian(
            coords1.view(), coords2.view(), PolynomialPairFunction{ &num_weighted_evaluations }, weights.view() );
    derivative_result_view[0].num_weighted_evaluations = num_weighted_evaluations;
  } );
  const Array1D<PolynomialDerivativeResult, MemorySpace::Host> host_results( derivative_results );
  const auto& value_jacobian = host_results[0].value_jacobian;
  EXPECT_EQ( host_results[0].num_evaluations, PolynomialPairFunction::size_coords );
  EXPECT_EQ( value_jacobian.pair_value.quadrature_metadata.status, QuadratureStatus::Success );
  EXPECT_EQ( value_jacobian.pair_value.quadrature_metadata.num_required_points, 2 );
  EXPECT_TRUE( value_jacobian.pair_value.has_active_points );
  const auto values = value_jacobian.pair_value.values();
  EXPECT_DOUBLE_EQ( values[0], 14.0 );
  EXPECT_DOUBLE_EQ( values[1], 15.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.derivatives.jacobian( 0, 0 ), 6.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.derivatives.jacobian( 0, 1 ), 1.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.derivatives.jacobian( 1, 0 ), 5.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.derivatives.jacobian( 1, 1 ), 3.0 );
  EXPECT_EQ( host_results[0].num_weighted_evaluations, PolynomialPairFunction::size_coords );
  const auto& weighted_hessian = host_results[0].weighted_hessian;
  EXPECT_DOUBLE_EQ( weighted_hessian.jacobian[0], 7.0 );
  EXPECT_DOUBLE_EQ( weighted_hessian.jacobian[1], -1.0 );
  EXPECT_DOUBLE_EQ( weighted_hessian.hessian( 0, 0 ), 4.0 );
  EXPECT_DOUBLE_EQ( weighted_hessian.hessian( 0, 1 ), -1.0 );
  EXPECT_DOUBLE_EQ( weighted_hessian.hessian( 1, 0 ), -1.0 );
  EXPECT_DOUBLE_EQ( weighted_hessian.hessian( 1, 1 ), 0.0 );
}

/** @brief Scalar polynomial with a counter to detect separate value or Jacobian evaluations. */
struct ScalarPolynomialPairFunction {
  using Coords1 = MdSpan<const RealT, 1>;                       ///< First scalar input.
  using Coords2 = MdSpan<const RealT, 1>;                       ///< Second scalar input.
  static constexpr IndexT size_geom1_coords = Coords1::size();  ///< Geometry1 coordinate count.
  static constexpr IndexT size_geom2_coords = Coords2::size();  ///< Geometry2 coordinate count.
  static constexpr IndexT size_values = 1;                      ///< One scalar output.
  static constexpr IndexT size_coords = 2;                      ///< Two coordinate entries.
  IndexT* num_evaluations;  ///< Number of evaluate() calls, in backend-accessible memory.
  bool active;              ///< Activation reported with the polynomial value.
  QuadratureStatus status;  ///< Quadrature status reported with the polynomial value.

  /** @brief Evaluate x0^2*x1 with the configured status. @param coords1 First scalar input. @param coords2 Second
   * scalar input.
   * @return Scalar value, activation, and status; numerical outputs are retained even for inactive or failed inputs
   * so the test can check that differentiation masks them. */
  TRIBOL_HOST_DEVICE PairEnergyValue evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    ++*num_evaluations;
    return {
        coords1[0] * coords1[0] * coords2[0], active, { status, status == QuadratureStatus::InvalidGeometry ? 0 : 2 } };
  }
};

// Both derivative operations must retain quadrature state, stop on failure or inactivity, and keep active zero values.
TEST_P( EnergyMortarPairEvaluationTest, derivatives_reuse_ad_values_and_state )
{
  using Function = ScalarPolynomialPairFunction;
  /** @brief Polynomial inputs and reported quadrature state for one backend test case. */
  struct DerivativeCase {
    MdArray<RealT, Function::size_coords> coordinates;  ///< Polynomial inputs.
    bool active;                                        ///< Reported activation.
    QuadratureStatus status;                            ///< Reported quadrature status.
  };
  /** @brief Derivative results and evaluation counts for both operations on the same input. */
  struct DerivativeCaseResult {
    PairValueJacobian<PairFunctionValue<Function>, Function::size_coords>
        values_and_jacobian;  ///< Values and Jacobian.
    PairValueJacobianHessian<PairFunctionValue<Function>, Function::size_coords>
        value_jacobian_hessian;           ///< Value, Jacobian, and Hessian.
    IndexT num_jacobian_evaluations = 0;  ///< Forward AD evaluations.
    IndexT num_hessian_evaluations = 0;   ///< Forward-over-reverse AD evaluations.
  };
  const auto derivative_cases = copyToExecutionSpace<DerivativeCase>( {
      { MdArray<RealT, Function::size_coords>{ 3.0, 5.0 }, true, QuadratureStatus::Success },
      { MdArray<RealT, Function::size_coords>{ 0.0, 5.0 }, true, QuadratureStatus::Success },
      { MdArray<RealT, Function::size_coords>{ 3.0, 5.0 }, false, QuadratureStatus::Success },
      { MdArray<RealT, Function::size_coords>{ 3.0, 5.0 }, true, QuadratureStatus::InvalidGeometry },
      { MdArray<RealT, Function::size_coords>{ 3.0, 5.0 }, true, QuadratureStatus::InsufficientCapacity },
  } );
  ArrayT<DerivativeCaseResult> derivative_results( derivative_cases.size(), derivative_cases.size(), allocatorId() );
  const auto derivative_case_view = derivative_cases.view();
  const auto derivative_result_view = derivative_results.view();
  forAllExec( GetParam(), derivative_cases.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
    DerivativeCaseResult derivative_result{};
    const Function jacobian_function{ &derivative_result.num_jacobian_evaluations, derivative_case_view[i].active,
                                      derivative_case_view[i].status };
    const Function hessian_function{ &derivative_result.num_hessian_evaluations, derivative_case_view[i].active,
                                     derivative_case_view[i].status };
    const Function::Coords1 coords1{ derivative_case_view[i].coordinates.data(), Function::size_geom1_coords };
    const Function::Coords2 coords2{ derivative_case_view[i].coordinates.data() + Function::size_geom1_coords,
                                     Function::size_geom2_coords };
    derivative_result.values_and_jacobian =
        EnzymePairDerivative<Function>::valuesAndJacobian( coords1, coords2, jacobian_function );
    derivative_result.value_jacobian_hessian =
        EnzymePairDerivative<Function>::valueJacobianAndHessian( coords1, coords2, hessian_function );
    derivative_result_view[i] = derivative_result;
  } );
  const ArrayT<DerivativeCase, 1, MemorySpace::Host> host_derivative_cases( derivative_cases );
  const ArrayT<DerivativeCaseResult, 1, MemorySpace::Host> host_derivative_results( derivative_results );
  for ( IndexT i = 0; i < derivative_cases.size(); ++i ) {
    SCOPED_TRACE( i );
    const auto& derivative_result = host_derivative_results[i];
    const bool valid_active =
        host_derivative_cases[i].active && host_derivative_cases[i].status == QuadratureStatus::Success;
    const RealT x = valid_active ? host_derivative_cases[i].coordinates[0] : 0.0;
    const RealT y = valid_active ? host_derivative_cases[i].coordinates[1] : 0.0;
    EXPECT_EQ( derivative_result.values_and_jacobian.pair_value.quadrature_metadata.status,
               host_derivative_cases[i].status );
    EXPECT_EQ( derivative_result.value_jacobian_hessian.pair_value.quadrature_metadata.status,
               host_derivative_cases[i].status );
    EXPECT_EQ( derivative_result.values_and_jacobian.pair_value.quadrature_metadata.num_required_points,
               i == 3 ? 0 : 2 );
    EXPECT_EQ( derivative_result.value_jacobian_hessian.pair_value.quadrature_metadata.num_required_points,
               i == 3 ? 0 : 2 );
    EXPECT_EQ( derivative_result.values_and_jacobian.pair_value.has_active_points, host_derivative_cases[i].active );
    EXPECT_EQ( derivative_result.value_jacobian_hessian.pair_value.has_active_points, host_derivative_cases[i].active );
    EXPECT_EQ( derivative_result.num_jacobian_evaluations, valid_active ? Function::size_coords : 1 );
    EXPECT_EQ( derivative_result.num_hessian_evaluations, valid_active ? Function::size_coords : 1 );
    EXPECT_DOUBLE_EQ( derivative_result.values_and_jacobian.pair_value.energy, x * x * y );
    EXPECT_DOUBLE_EQ( derivative_result.value_jacobian_hessian.pair_value.energy, x * x * y );
    EXPECT_DOUBLE_EQ( derivative_result.values_and_jacobian.derivatives.jacobian( 0, 0 ), 2.0 * x * y );
    EXPECT_DOUBLE_EQ( derivative_result.values_and_jacobian.derivatives.jacobian( 0, 1 ), x * x );
    EXPECT_DOUBLE_EQ( derivative_result.value_jacobian_hessian.derivatives.jacobian[0], 2.0 * x * y );
    EXPECT_DOUBLE_EQ( derivative_result.value_jacobian_hessian.derivatives.jacobian[1], x * x );
    EXPECT_DOUBLE_EQ( derivative_result.value_jacobian_hessian.derivatives.hessian( 0, 0 ), 2.0 * y );
    EXPECT_DOUBLE_EQ( derivative_result.value_jacobian_hessian.derivatives.hessian( 0, 1 ), 2.0 * x );
    EXPECT_DOUBLE_EQ( derivative_result.value_jacobian_hessian.derivatives.hessian( 1, 0 ), 2.0 * x );
    EXPECT_DOUBLE_EQ( derivative_result.value_jacobian_hessian.derivatives.hessian( 1, 1 ), 0.0 );
  }
}

/** @brief Count evaluations of the vector function (x^2*y, x*y). */
struct VectorPolynomialPairFunction : ScalarPolynomialPairFunction {
  static constexpr IndexT size_values = 2;  ///< Two polynomial outputs.

  /** @brief Evaluate the polynomials. @param coords1 Scalar x. @param coords2 Scalar y.
   * @return Polynomial values with successful active quadrature state. */
  TRIBOL_HOST_DEVICE PairResidualValue<1, 1, 1> evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    const auto scalar = ScalarPolynomialPairFunction::evaluate( coords1, coords2 );
    PairResidualValue<1, 1, 1> result;
    result.geometry1( 0, 0 ) = scalar.energy;
    result.geometry2( 0, 0 ) = coords1[0] * coords2[0];
    result.has_active_points = scalar.has_active_points;
    result.quadrature_metadata = scalar.quadrature_metadata;
    return result;
  }
};

// Open nodal contact still needs signed gaps, areas, and their derivatives before enforcement is applied.
TEST_P( EnergyMortarPairEvaluationTest, open_nodal_integrals_keep_values_and_jacobians )
{
  using RuleGenerator = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>;
  using Function = PairNodalGapIntegral<RuleGenerator, OpposingNormalGap, BasisFunction<EDGE, LINEAR_LAGRANGE>>;
  using PairNodalGap = Function::PairNodalGap;
  using DerivativeResult = PairValueJacobian<PairNodalGap, Function::size_coords>;
  ArrayT<DerivativeResult> derivative_results( 1, 1, allocatorId() );
  const auto derivative_result_view = derivative_results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const MdArray<RealT, 2, 2> coords1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.0 } };
    const MdArray<RealT, 2, 2> coords2{ MdArray<RealT, 2>{ 3.0, -0.1 }, MdArray<RealT, 2>{ -1.0, -0.1 } };
    const Function function{ RuleGenerator{ SmoothedProjectedEdgeMapping{ 0.0 } } };
    derivative_result_view[0] =
        EnzymePairDerivative<Function>::valuesAndJacobian( coords1.view(), coords2.view(), function );
  } );
  const ArrayT<DerivativeResult, 1, MemorySpace::Host> host_derivative_results( derivative_results );
  const auto& derivative_result = host_derivative_results[0];
  ASSERT_EQ( derivative_result.pair_value.quadrature_metadata.status, QuadratureStatus::Success );
  EXPECT_TRUE( derivative_result.pair_value.has_active_points );
  for ( IndexT node = 0; node < Function::num_nodes; ++node ) {
    const IndexT gap_row = PairNodalGap::weightedGapIndex( node );
    const IndexT area_row = PairNodalGap::tributaryAreaIndex( node );
    EXPECT_NEAR( derivative_result.pair_value.weightedGap( node ), 0.1, tolerance );
    EXPECT_NEAR( derivative_result.pair_value.tributaryArea( node ), 1.0, tolerance );
    // Translating geometry2 upward decreases each nodal gap while leaving its area unchanged.
    EXPECT_NEAR(
        derivative_result.derivatives.jacobian( gap_row, 5 ) + derivative_result.derivatives.jacobian( gap_row, 7 ),
        -1.0, tolerance );
    EXPECT_NEAR(
        derivative_result.derivatives.jacobian( area_row, 5 ) + derivative_result.derivatives.jacobian( area_row, 7 ),
        0.0, tolerance );
  }
}

// Only all-zero weights skip AD; a nonzero weight in either output must retain the contracted Jacobian and Hessian.
TEST_P( EnergyMortarPairEvaluationTest, zero_weights_skip_derivatives )
{
  using Function = VectorPolynomialPairFunction;
  const auto weights = copyToExecutionSpace<MdArray<RealT, Function::size_values>>(
      { MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 0.0, 2.0 }, MdArray<RealT, 2>{ 2.0, 0.0 },
        MdArray<RealT, 2>{ 2.0, -1.0 } } );
  /** @brief Weighted derivative outputs and the number of function evaluations they required. */
  struct Result {
    PairJacobianHessian<Function::size_coords> derivatives;  ///< Contracted Jacobian and Hessian.
    IndexT num_evaluations = 0;                              ///< Calls through AD.
  };
  ArrayT<Result> derivative_results( weights.size(), weights.size(), allocatorId() );
  const auto weight_view = weights.view();
  const auto derivative_result_view = derivative_results.view();
  forAllExec( GetParam(), weights.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
    Result derivative_result{};
    const Function function{ { &derivative_result.num_evaluations, true, QuadratureStatus::Success } };
    const MdArray<RealT, 1> coords1{ 3.0 }, coords2{ 5.0 };
    derivative_result.derivatives = EnzymePairDerivative<Function>::weightedJacobianAndHessian(
        coords1.view(), coords2.view(), function, weight_view[i].view() );
    derivative_result_view[i] = derivative_result;
  } );
  const ArrayT<MdArray<RealT, Function::size_values>, 1, MemorySpace::Host> host_weights( weights );
  const ArrayT<Result, 1, MemorySpace::Host> host_derivative_results( derivative_results );
  for ( IndexT i = 0; i < weights.size(); ++i ) {
    SCOPED_TRACE( i );
    const auto& derivative_result = host_derivative_results[i];
    const auto& weights_for_case = host_weights[i];
    EXPECT_EQ( derivative_result.num_evaluations, i == 0 ? 0 : Function::size_coords );
    EXPECT_DOUBLE_EQ( derivative_result.derivatives.jacobian[0],
                      30.0 * weights_for_case[0] + 5.0 * weights_for_case[1] );
    EXPECT_DOUBLE_EQ( derivative_result.derivatives.jacobian[1],
                      9.0 * weights_for_case[0] + 3.0 * weights_for_case[1] );
    EXPECT_DOUBLE_EQ( derivative_result.derivatives.hessian( 0, 0 ), 10.0 * weights_for_case[0] );
    EXPECT_DOUBLE_EQ( derivative_result.derivatives.hessian( 0, 1 ), 6.0 * weights_for_case[0] + weights_for_case[1] );
    EXPECT_DOUBLE_EQ( derivative_result.derivatives.hessian( 1, 0 ), 6.0 * weights_for_case[0] + weights_for_case[1] );
    EXPECT_DOUBLE_EQ( derivative_result.derivatives.hessian( 1, 1 ), 0.0 );
  }
}

INSTANTIATE_TEST_SUITE_P( backends, EnergyMortarPairEvaluationTest, tribol::detail::enabledExecutionModes() );

#endif
}  // namespace

/** @brief Run primal and optional Enzyme tests. @param argc Argument count. @param argv Arguments. @return Status. */
int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int test_status = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_status;
}
