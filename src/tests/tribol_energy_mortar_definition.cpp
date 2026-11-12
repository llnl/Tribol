// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check direct Energy Mortar energies, nodal integrals, and explicit field partials on enabled backends.
 * Host Enzyme checks compare the new pair definitions with finite differences for each Gauss count and derivative
 * mode. The energy kernel checks active, open, zero-gap, and invalid pairs on every enabled backend, including the
 * compressive zero-gap Hessian. A pair with distinct geometry types and unequal coordinate counts checks that AD
 * includes nonlinear quantities cached during geometry construction. Forward AD checks values and quadrature state
 * without an extra value evaluation. Compile-time checks accept the supported pair interfaces and reject invalid sizes,
 * results, and coordinate splits. Returned buffers retain the pair allocator and remain valid after the input pair list
 * is destroyed. Weighted Jacobian/Hessian kernels check reordered pairs with unequal weights. MFEM tests separately
 * check assembly and shared-node normalization.
 */
#include <type_traits>

#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "ExecutionTest.hpp"
#include "FrozenPairQuadratureRuleGenerator.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"
#include "tribol/assembly/PairContribution.hpp"
#include "tribol/assembly/PairEvaluation.hpp"
#include "tribol/assembly/PairKernel.hpp"

namespace {
using namespace tribol;
constexpr RealT tolerance = 1.0e-12;  ///< Roundoff allowance for primal integrals and exact partials.
using EnergyMortarDefinitionTest = detail::ExecutionTest;

/** @brief Declaration-only pair function for checking the evaluation interface without executing it. */
struct CoordinateFunctionCandidate {
  static constexpr IndexT num_values = 2;  ///< Two output values.
  static constexpr IndexT num_coords = 4;  ///< Four coordinate entries.

  /** @brief Declare a const evaluation. @param coordinates Pair inputs. @return Values and quadrature state. */
  PairResidualValue<num_values> evaluate( MdSpan<const RealT, num_coords> coordinates ) const;
};

/** @brief Geometry pair function for checking mesh-kernel inputs without executing it. */
struct GeometryPairFunctionCandidate {
  using FirstGeometry = LinearEdgeGeometry;                                   ///< Geometry1 type.
  using SecondGeometry = LinearEdgeGeometry;                                  ///< Geometry2 type.
  static constexpr IndexT num_values = 2;                                     ///< Two output values.
  static constexpr IndexT num_geom1_coords = FirstGeometry::Coords::size();   ///< Geometry1 coordinate count.
  static constexpr IndexT num_geom2_coords = SecondGeometry::Coords::size();  ///< Geometry2 coordinate count.
  static constexpr IndexT num_coords = num_geom1_coords + num_geom2_coords;   ///< Combined coordinate count.

  /** @brief Supply an empty evaluation. @param geometry1 First edge. @param geometry2 Second edge.
   * @return Zero values and empty, inactive quadrature state. */
  PairResidualValue<num_values> evaluate( [[maybe_unused]] const FirstGeometry& geometry1,
                                          [[maybe_unused]] const SecondGeometry& geometry2 ) const
  {
    return {};
  }
};

/** @brief Zero outputs must be rejected before instantiating fixed-size numerical arrays. */
struct ZeroValuePairFunction : CoordinateFunctionCandidate {
  static constexpr IndexT num_values = 0;  ///< Invalid output count.
};

/** @brief Zero coordinates must be rejected before instantiating a coordinate view. */
struct ZeroCoordinatePairFunction : CoordinateFunctionCandidate {
  static constexpr IndexT num_coords = 0;  ///< Invalid coordinate count.
};

/** @brief Output counts must be compile-time constants. */
struct RuntimeSizePairFunction : CoordinateFunctionCandidate {
  IndexT num_values = 2;  ///< Invalid runtime output count.
};

/** @brief The returned value array must agree with the declared output count. */
struct WrongNumValuesPairFunction : CoordinateFunctionCandidate {
  static constexpr IndexT num_values = 1;  ///< Disagrees with the inherited two-entry result.
};

/** @brief Pair evaluation must be callable on a const function object. */
struct MutablePairFunction : CoordinateFunctionCandidate {
  /** @brief Declare a mutable-only evaluation. @param coordinates Pair inputs. @return Values and quadrature state. */
  PairResidualValue<num_values> evaluate( MdSpan<const RealT, num_coords> coordinates );
};

/** @brief An array alone supplies no quadrature status or activation. */
struct MissingStatePairFunction : CoordinateFunctionCandidate {
  /** @brief Declare a result without quadrature state. @param coordinates Pair inputs. @return Numerical values only.
   */
  MdArray<RealT, num_values> evaluate( MdSpan<const RealT, num_coords> coordinates ) const;
};

/** @brief The numerical values must be readable from a const evaluation result. */
struct MutableValuesPairFunction : CoordinateFunctionCandidate {
  /** @brief Result that hides the const values() accessor with a mutable-only accessor. */
  struct Result : PairResidualValue<num_values> {
    /** @brief Declare mutable-only value access. @return Pair values. */
    MdArray<RealT, num_values> values();
  };
  /** @brief Declare an incompatible result. @param coordinates Pair inputs. @return Mutable-only numerical values. */
  Result evaluate( MdSpan<const RealT, num_coords> coordinates ) const;
};

/** @brief Activation must have the expected bool type. */
struct WrongActivationPairFunction : CoordinateFunctionCandidate {
  /** @brief Result with an integer flag that hides the required bool member. */
  struct Result : PairResidualValue<num_values> {
    IndexT has_active_points = 0;  ///< Incompatible activation type.
  };
  /** @brief Declare an incompatible result. @param coordinates Pair inputs. @return Values with an integer flag. */
  Result evaluate( MdSpan<const RealT, num_coords> coordinates ) const;
};

/** @brief Quadrature state must have the expected QuadratureResult type. */
struct WrongQuadraturePairFunction : CoordinateFunctionCandidate {
  /** @brief Result whose status member omits the required point count. */
  struct Result : PairResidualValue<num_values> {
    QuadratureStatus quadrature_result{};  ///< Incompatible quadrature result type.
  };
  /** @brief Declare an incompatible result. @param coordinates Pair inputs. @return Values without a point count. */
  Result evaluate( MdSpan<const RealT, num_coords> coordinates ) const;
};

/** @brief A coordinate split must leave entries for geometry1. */
struct EmptyFirstGeometryPairFunction : GeometryPairFunctionCandidate {
  static constexpr IndexT num_geom1_coords = 0;  ///< Invalid empty geometry1.
};

/** @brief A coordinate split must leave entries for geometry2. */
struct EmptySecondGeometryPairFunction : GeometryPairFunctionCandidate {
  static constexpr IndexT num_geom2_coords = 0;  ///< Invalid empty geometry2.
};

// Coordinate differentiation remains available for functions without geometry types.
static_assert( CoordinateFunctionType<CoordinateFunctionCandidate> );
static_assert( !PairFunctionType<CoordinateFunctionCandidate> );
static_assert( PairFunctionType<GeometryPairFunctionCandidate> );
static_assert( CoordinateFunctionType<PairCoordinateFunction<GeometryPairFunctionCandidate>> );
static_assert( !CoordinateFunctionType<ZeroValuePairFunction> );
static_assert( !CoordinateFunctionType<ZeroCoordinatePairFunction> );
static_assert( !CoordinateFunctionType<RuntimeSizePairFunction> );
static_assert( !CoordinateFunctionType<WrongNumValuesPairFunction> );
static_assert( !CoordinateFunctionType<MutablePairFunction> );
static_assert( !CoordinateFunctionType<MissingStatePairFunction> );
static_assert( !CoordinateFunctionType<MutableValuesPairFunction> );
static_assert( !CoordinateFunctionType<WrongActivationPairFunction> );
static_assert( !CoordinateFunctionType<WrongQuadraturePairFunction> );
static_assert( !PairFunctionType<EmptyFirstGeometryPairFunction> );
static_assert( !PairFunctionType<EmptySecondGeometryPairFunction> );

// Energy and nodal-integral functions accept two geometries with fixed coordinate counts.
static_assert( PairFunctionType<EnergyMortarEnergy<
                   PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>>> );
static_assert( PairFunctionType<EnergyMortarNodalEnergy<
                   PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>>> );

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
  using FirstGeometry = SquaredCoordinateGeometry;                            ///< Single-coordinate geometry.
  using SecondGeometry = CoordinateProductGeometry;                           ///< Two-coordinate geometry.
  static constexpr IndexT num_values = 1;                                     ///< One scalar value.
  static constexpr IndexT num_geom1_coords = FirstGeometry::Coords::size();   ///< Geometry1 coordinate count.
  static constexpr IndexT num_geom2_coords = SecondGeometry::Coords::size();  ///< Geometry2 coordinate count.
  static constexpr IndexT num_coords = num_geom1_coords + num_geom2_coords;   ///< Combined coordinate count.

  /** @brief Multiply the cached expressions. @param geometry1 Squared coordinate. @param geometry2 Coordinate product.
   * @return Active value with successful one-point quadrature state. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairEnergyValue evaluate( const FirstGeometry& geometry1,
                                                                    const SecondGeometry& geometry2 ) const
  {
    return { geometry1.squared * geometry2.product, true, { QuadratureStatus::Success, 1 } };
  }
};
static_assert( PairFunctionType<CachedGeometryPairFunction> );
static_assert( CoordinateFunctionType<PairCoordinateFunction<CachedGeometryPairFunction>> );

/** @brief Pair energy and nodal integrals written by one backend thread. @tparam NumPoints Gauss point count. */
template <int NumPoints>
struct ParallelIntegralResult {
  using RuleGenerator =
      PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>;
  PairEnergyValue energy;                                                     ///< Quadrature-point energy and status.
  typename EnergyMortarNodalEnergy<RuleGenerator>::NodalIntegralValue nodal;  ///< Nodal integrals and status.
};

/** @brief Nodal field inputs and exact penalty partials for one backend test case. */
struct NodalPartialsCase {
  RealT gap;                           ///< Weighted nodal gap.
  RealT area;                          ///< Tributary area, including unsupported values on both sides of zero.
  EnergyMortarNodalPartials expected;  ///< Exact energy and field partials for stiffness four.
};

/** @brief Check constant-gap integrals on an unsmoothed interval. @tparam NumPoints Gauss point count.
 * @param mode Execution backend. @param allocator Backend-accessible allocation resource. */
template <int NumPoints>
void checkParallelIntegrals( ExecutionMode mode, int allocator )
{
  using RuleGenerator =
      PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>;
  using Energy = EnergyMortarEnergy<RuleGenerator>;
  using Nodal = EnergyMortarNodalEnergy<RuleGenerator>;
  ArrayT<ParallelIntegralResult<NumPoints>> results( 1, 1, allocator );
  const auto output = results.view();
  forAllExec( mode, 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const MdArray<RealT, 2, 2> coordinates1{ MdArray<RealT, 2>{ 0.0, 0.0 }, MdArray<RealT, 2>{ 2.0, 0.0 } };
    const MdArray<RealT, 2, 2> coordinates2{ MdArray<RealT, 2>{ 3.0, 0.1 }, MdArray<RealT, 2>{ -1.0, 0.1 } };
    const LinearEdgeGeometry geometry1( coordinates1.view() ), geometry2( coordinates2.view() );
    const RuleGenerator rule_generator( SmoothedProjectedEdgeMapping{ 0.0 } );
    output[0] = { Energy{ rule_generator, 4.0 }.evaluate( geometry1, geometry2 ),
                  Nodal{ rule_generator }.evaluate( geometry1, geometry2 ) };
  } );
  const ArrayT<ParallelIntegralResult<NumPoints>, 1, MemorySpace::Host> host( results );
  ASSERT_EQ( host[0].energy.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_EQ( host[0].energy.quadrature_result.num_required_points, NumPoints );
  // A length-two edge with gap -0.1 has energy 0.5*4*0.1^2*2, and equal nodal areas of one.
  EXPECT_NEAR( host[0].energy.energy, 0.04, tolerance );
  EXPECT_TRUE( host[0].energy.has_active_points );
  ASSERT_EQ( host[0].nodal.quadrature_result.status, QuadratureStatus::Success );
  for ( IndexT node = 0; node < Nodal::num_nodes; ++node ) {
    EXPECT_NEAR( host[0].nodal.value[node], -0.1, tolerance );
    EXPECT_NEAR( host[0].nodal.value[Nodal::num_nodes + node], 1.0, tolerance );
  }
}

// Direct nodal and quadrature-point calculations must use physical weights exactly once for every Gauss count.
TEST_P( EnergyMortarDefinitionTest, integrates_constant_gap )
{
  checkParallelIntegrals<1>( GetParam(), allocatorId() );
  checkParallelIntegrals<2>( GetParam(), allocatorId() );
  checkParallelIntegrals<3>( GetParam(), allocatorId() );
}

// Unsupported areas and open gaps give zero partials; zero gap retains the compressive tangent.
TEST_P( EnergyMortarDefinitionTest, guards_area_before_dividing_and_keeps_zero_gap_tangent )
{
  constexpr RealT area_tolerance = 1.0e-14;
  const auto cases = copyToExecutionSpace<NodalPartialsCase>( { { -1.0, 2.0, { 1.0, -2.0, -0.5, 2.0, 1.0, 0.5 } },
                                                                { 0.0, 2.0, { 0.0, 0.0, 0.0, 2.0, 0.0, 0.0 } },
                                                                { 1.0, 2.0, {} },
                                                                { -1.0, 0.0, {} },
                                                                { -1.0, area_tolerance, {} },
                                                                { -1.0, -area_tolerance, {} } } );
  ArrayT<EnergyMortarNodalPartials> results( cases.size(), cases.size(), allocatorId() );
  const auto input = cases.view();
  const auto output = results.view();
  forAllExec( GetParam(), cases.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
    output[i] = EnergyMortarNodalPartials::evaluate( input[i].gap, input[i].area, 4.0, area_tolerance );
  } );
  const ArrayT<NodalPartialsCase, 1, MemorySpace::Host> host_cases( cases );
  const ArrayT<EnergyMortarNodalPartials, 1, MemorySpace::Host> host_results( results );
  for ( IndexT i = 0; i < cases.size(); ++i ) {
    SCOPED_TRACE( i );
    const auto& actual = host_results[i];
    const auto& expected = host_cases[i].expected;
    EXPECT_NEAR( actual.energy, expected.energy, tolerance );
    EXPECT_NEAR( actual.gap, expected.gap, tolerance );
    EXPECT_NEAR( actual.area, expected.area, tolerance );
    EXPECT_NEAR( actual.gap_gap, expected.gap_gap, tolerance );
    EXPECT_NEAR( actual.gap_area, expected.gap_area, tolerance );
    EXPECT_NEAR( actual.area_area, expected.area_area, tolerance );
  }
}
INSTANTIATE_TEST_SUITE_P( backends, EnergyMortarDefinitionTest, detail::enabledExecutionModes() );

TEST_P( EnergyMortarDefinitionTest, contribution_buffers_use_backend_accessible_structure_of_arrays )
{
  constexpr IndexT num_pairs = 3;
  PairValueJacobianBuffer<2, 2> values( num_pairs, allocatorId() );
  ElementBlockBuffer<2, 2> blocks( num_pairs, allocatorId() );
  const auto value_jacobian_view = values.view();
  const auto block_view = blocks.view();
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
// Different geometry types must retain all cached coordinate dependencies through both orders of AD.
TEST_P( EnergyMortarDefinitionTest, differentiates_geometry_construction_with_unequal_coordinate_counts )
{
  using Function = PairCoordinateFunction<CachedGeometryPairFunction>;
  ArrayT<PairValueJacobianHessian<Function::num_coords>> results( 1, 1, allocatorId() );
  const auto output = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const MdArray<RealT, Function::num_coords> coordinates{ 2.0, 3.0, 4.0 };
    output[0] = EnzymePairDerivative<Function>::valueJacobianAndHessian( coordinates.view(),
                                                                         Function{ CachedGeometryPairFunction{} } );
  } );
  const ArrayT<PairValueJacobianHessian<Function::num_coords>, 1, MemorySpace::Host> host( results );
  const auto& result = host[0];
  ASSERT_EQ( result.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_EQ( result.quadrature_result.num_required_points, 1 );
  EXPECT_TRUE( result.has_active_points );
  EXPECT_DOUBLE_EQ( result.value, 48.0 );
  // For f(x,y,z)=x^2*y*z, both cached expressions contribute to the mixed second derivatives.
  const MdArray<RealT, Function::num_coords> jacobian{ 48.0, 16.0, 12.0 };
  const MdArray<RealT, Function::num_coords, Function::num_coords> hessian{
      MdArray<RealT, Function::num_coords>{ 24.0, 16.0, 12.0 }, MdArray<RealT, Function::num_coords>{ 16.0, 0.0, 4.0 },
      MdArray<RealT, Function::num_coords>{ 12.0, 4.0, 0.0 } };
  for ( IndexT row = 0; row < Function::num_coords; ++row ) {
    EXPECT_DOUBLE_EQ( result.jacobian[row], jacobian[row] );
    for ( IndexT column = 0; column < Function::num_coords; ++column ) {
      EXPECT_DOUBLE_EQ( result.hessian( row, column ), hessian( row, column ) );
    }
  }
}

/** @brief Check whether a mesh kernel accepts a scalar pair function. @tparam Function Candidate pair function. */
template <typename Function>
concept CanEvaluateScalarPairKernel =
    requires { &evaluatePairValuesJacobiansAndHessians<ExecutionMode::Sequential, Function>; };

// The scalar kernel accepts energy outputs and rejects vector nodal integrals at its interface.
static_assert( CanEvaluateScalarPairKernel<EnergyMortarEnergy<
                   PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>>> );
static_assert( !CanEvaluateScalarPairKernel<EnergyMortarNodalEnergy<
                   PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>>> );

// Both kernels must return values and quadrature status for active, open, zero-gap, and invalid pairs. A rigid vertical
// edge translation has analytic derivatives; zero gap retains activation and the compressive Hessian.
TEST_P( EnergyMortarDefinitionTest, energy_kernels_return_values_jacobians_and_hessians )
{
  using RuleGenerator = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>;
  using Function = EnergyMortarEnergy<RuleGenerator>;
  const Function function{ RuleGenerator{ SmoothedProjectedEdgeMapping{ 0.0 } }, 4.0 };
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
  EXPECT_EQ( buffer.quadratureResults().getAllocatorID(), pairs.getAllocatorID() );
  EXPECT_EQ( value_jacobian.numberOfPairs(), pairs.size() );
  EXPECT_EQ( value_jacobian.values().getAllocatorID(), pairs.getAllocatorID() );
  EXPECT_EQ( value_jacobian.jacobians().getAllocatorID(), pairs.getAllocatorID() );

  ArrayT<RealT, 2> summaries( axom::StackArray<axom::IndexType, 2>{ pairs.size(), 4 }, allocatorId() );
  const auto summary = summaries.view();
  forAllExec( GetParam(), pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
    summary( pair, 0 ) = contributions_view.values[pair];
    summary( pair, 1 ) =
        contributions_view.derivatives.jacobian( pair, 1 ) + contributions_view.derivatives.jacobian( pair, 3 );
    summary( pair, 2 ) = 0.0;
    summary( pair, 3 ) = 0.0;
    for ( IndexT row = 0; row < Function::num_coords; ++row ) {
      summary( pair, 3 ) += std::abs( contributions_view.derivatives.jacobian( pair, row ) );
      for ( IndexT column = 0; column < Function::num_coords; ++column ) {
        summary( pair, 3 ) += std::abs( contributions_view.derivatives.hessian( pair, row, column ) );
        if ( ( row == 1 || row == 3 ) && ( column == 1 || column == 3 ) ) {
          summary( pair, 2 ) += contributions_view.derivatives.hessian( pair, row, column );
        }
      }
    }
  } );
  const ArrayT<RealT, 2, MemorySpace::Host> host( summaries );
  const Array1D<QuadratureResult, MemorySpace::Host> status( buffer.quadratureResults() );
  const Array1D<RealT, MemorySpace::Host> values( value_jacobian.values() );
  const Array1D<RealT, MemorySpace::Host> jacobians( value_jacobian.jacobians() );
  const Array1D<QuadratureResult, MemorySpace::Host> jacobian_status( value_jacobian.quadratureResults() );
  const Array1D<IndexT, MemorySpace::Host> active( value_jacobian.activePoints() );
  for ( IndexT pair = 0; pair < pairs.size(); ++pair ) {
    EXPECT_NEAR( values[pair], host( pair, 0 ), tolerance );
    EXPECT_NEAR( jacobians[pairs.size() + pair] + jacobians[3 * pairs.size() + pair], host( pair, 1 ), tolerance );
    EXPECT_EQ( jacobian_status[pair].status, status[pair].status );
    EXPECT_EQ( jacobian_status[pair].num_required_points, pair == 3 ? 0 : 2 );
    EXPECT_EQ( active[pair], pair == 0 || pair == 2 );
  }
  // Length two and stiffness four give E=0.04, dE/dy=-0.8, and d2E/dy2=8 for the penetrating pair.
  EXPECT_NEAR( host( 0, 0 ), 0.04, tolerance );
  EXPECT_NEAR( host( 0, 1 ), -0.8, tolerance );
  EXPECT_NEAR( host( 0, 2 ), 8.0, tolerance );
  EXPECT_NEAR( host( 2, 0 ), 0.0, tolerance );
  EXPECT_NEAR( host( 2, 1 ), 0.0, tolerance );
  EXPECT_NEAR( host( 2, 2 ), 8.0, tolerance );
  for ( IndexT pair : { 0, 1, 2 } ) {
    EXPECT_EQ( status[pair].status, QuadratureStatus::Success );
  }
  EXPECT_EQ( status[3].status, QuadratureStatus::InvalidGeometry );
  for ( IndexT pair : { 1, 3 } ) {
    EXPECT_EQ( host( pair, 0 ), 0.0 );
    EXPECT_EQ( host( pair, 3 ), 0.0 );
    for ( IndexT coordinate = 0; coordinate < Function::num_coords; ++coordinate ) {
      EXPECT_EQ( jacobians[coordinate * pairs.size() + pair], 0.0 );
    }
  }
  EXPECT_THROW( dispatchExecutionMode( GetParam(),
                                       [&]<ExecutionMode Exec>() {
                                         return checkPairQuadratureResults<Exec>( buffer.quadratureResults() );
                                       } ),
                std::invalid_argument );

  // Read derivatives after destroying the input pair list. Reversed element IDs and unequal weights check pair slots.
  auto weighted = dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    const auto weighted_pairs = copyToExecutionSpace<InterfacePair>( { { 0, 2 }, { 0, 0 } } );
    return evaluatePairWeightedJacobiansAndHessians<Exec>(
        weighted_pairs, mesh1.getView(), mesh2.getView(), function,
        [] TRIBOL_HOST_DEVICE( IndexT pair ) { return MdArray<RealT, Function::num_values>{ pair + 2.0 }; } );
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
    for ( IndexT coordinate = 0; coordinate < Function::num_coords; ++coordinate ) {
      EXPECT_NEAR( weighted_jacobians[coordinate * num_weighted_pairs + pair],
                   weight * jacobians[coordinate * pairs.size() + original_pair], tolerance );
    }
    RealT vertical_hessian = 0.0;
    for ( IndexT row : { 1, 3 } ) {
      for ( IndexT column : { 1, 3 } ) {
        vertical_hessian += weighted_hessians[( row * Function::num_coords + column ) * num_weighted_pairs + pair];
      }
    }
    EXPECT_NEAR( vertical_hessian, weight * host( original_pair, 2 ), tolerance );
  }
}

/** @brief Small fixed-size pair function used to compile and execute both Enzyme operations on every backend. */
struct PolynomialPairFunction {
  static constexpr IndexT num_values = 2;  ///< Number of outputs.
  static constexpr IndexT num_coords = 2;  ///< Number of active coordinates.
  IndexT* num_evaluations = nullptr;       ///< Optional counter for complete function evaluations.

  /** @brief Evaluate two coupled polynomial outputs. @param coordinates Active coordinates.
   * @return Function values with successful, active quadrature state. */
  TRIBOL_HOST_DEVICE PairResidualValue<num_values> evaluate( MdSpan<const RealT, num_coords> coordinates ) const
  {
    if ( num_evaluations != nullptr ) {
      ++*num_evaluations;
    }
    return {
        MdArray<RealT, num_values>{ coordinates[0] * coordinates[0] + coordinates[1], coordinates[0] * coordinates[1] },
        { QuadratureStatus::Success, 2 },
        true };
  }
};

/** @brief Device-storable output from both pair derivative operations. */
struct PolynomialDerivativeResult {
  PairValueJacobian<PolynomialPairFunction::num_values, PolynomialPairFunction::num_coords> value_jacobian;
  PairWeightedJacobianHessian<PolynomialPairFunction::num_coords> weighted_hessian;
  IndexT num_evaluations = 0;           ///< Complete evaluations used for values and Jacobian.
  IndexT num_weighted_evaluations = 0;  ///< Complete evaluations used for the weighted Jacobian and Hessian.
};

// Jacobian and Hessian AD calls must also supply their primal outputs without separate evaluations.
TEST_P( EnergyMortarDefinitionTest, enzyme_pair_derivatives_execute_on_backend )
{
  Array1D<PolynomialDerivativeResult> results( 1, 1, allocatorId() );
  auto output = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const MdArray<RealT, PolynomialPairFunction::num_coords> coordinates{ 3.0, 5.0 };
    const MdArray<RealT, PolynomialPairFunction::num_values> weights{ 2.0, -1.0 };
    IndexT num_evaluations = 0;
    output[0].value_jacobian = EnzymePairDerivative<PolynomialPairFunction>::valuesAndJacobian(
        coordinates.view(), PolynomialPairFunction{ &num_evaluations } );
    output[0].num_evaluations = num_evaluations;
    IndexT num_weighted_evaluations = 0;
    output[0].weighted_hessian = EnzymePairDerivative<PolynomialPairFunction>::weightedJacobianAndHessian(
        coordinates.view(), PolynomialPairFunction{ &num_weighted_evaluations }, weights.view() );
    output[0].num_weighted_evaluations = num_weighted_evaluations;
  } );
  const Array1D<PolynomialDerivativeResult, MemorySpace::Host> host_results( results );
  const auto& value_jacobian = host_results[0].value_jacobian;
  EXPECT_EQ( host_results[0].num_evaluations, PolynomialPairFunction::num_coords );
  EXPECT_EQ( value_jacobian.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_EQ( value_jacobian.quadrature_result.num_required_points, 2 );
  EXPECT_TRUE( value_jacobian.has_active_points );
  EXPECT_DOUBLE_EQ( value_jacobian.values[0], 14.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.values[1], 15.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.jacobian( 0, 0 ), 6.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.jacobian( 0, 1 ), 1.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.jacobian( 1, 0 ), 5.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.jacobian( 1, 1 ), 3.0 );
  EXPECT_EQ( host_results[0].num_weighted_evaluations, PolynomialPairFunction::num_coords );
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
  static constexpr IndexT num_values = 1;  ///< One scalar output.
  static constexpr IndexT num_coords = 2;  ///< Two coordinate entries.
  IndexT* num_evaluations;                 ///< Number of evaluate() calls, in backend-accessible memory.
  bool active;                             ///< Activation reported with the polynomial value.
  QuadratureStatus status;                 ///< Quadrature status reported with the polynomial value.

  /** @brief Evaluate x0^2*x1 with the configured status. @param coordinates Polynomial inputs.
   * @return Scalar value, activation, and status; numerical outputs are retained even for inactive or failed inputs
   * so the test can check that differentiation masks them. */
  TRIBOL_HOST_DEVICE PairEnergyValue evaluate( MdSpan<const RealT, num_coords> coordinates ) const
  {
    ++*num_evaluations;
    return { coordinates[0] * coordinates[0] * coordinates[1],
             active,
             { status, status == QuadratureStatus::InvalidGeometry ? 0 : 2 } };
  }
};

// Scalar AD must retain primal values and quadrature state without extra calls. Active zero values keep their Hessian;
// inactive and failed calculations return zeros and stop after the first call.
TEST_P( EnergyMortarDefinitionTest, scalar_derivatives_reuse_ad_values_and_state )
{
  using Function = ScalarPolynomialPairFunction;
  /** @brief Polynomial inputs and reported quadrature state for one backend test case. */
  struct Input {
    MdArray<RealT, Function::num_coords> coordinates;  ///< Polynomial inputs.
    bool active;                                       ///< Reported activation.
    QuadratureStatus status;                           ///< Reported quadrature status.
  };
  const auto cases = copyToExecutionSpace<Input>( {
      { MdArray<RealT, Function::num_coords>{ 3.0, 5.0 }, true, QuadratureStatus::Success },
      { MdArray<RealT, Function::num_coords>{ 0.0, 5.0 }, true, QuadratureStatus::Success },
      { MdArray<RealT, Function::num_coords>{ 3.0, 5.0 }, false, QuadratureStatus::Success },
      { MdArray<RealT, Function::num_coords>{ 3.0, 5.0 }, false, QuadratureStatus::InvalidGeometry },
      { MdArray<RealT, Function::num_coords>{ 3.0, 5.0 }, false, QuadratureStatus::InsufficientCapacity },
  } );
  ArrayT<PairValueJacobianHessian<Function::num_coords>> results( cases.size(), cases.size(), allocatorId() );
  ArrayT<IndexT> num_evaluations( cases.size(), cases.size(), allocatorId() );
  const auto input = cases.view();
  const auto output = results.view();
  const auto num_evaluations_view = num_evaluations.view();
  forAllExec( GetParam(), cases.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
    num_evaluations_view[i] = 0;
    const Function function{ &num_evaluations_view[i], input[i].active, input[i].status };
    output[i] = EnzymePairDerivative<Function>::valueJacobianAndHessian( input[i].coordinates.view(), function );
  } );
  const ArrayT<Input, 1, MemorySpace::Host> host_input( cases );
  const ArrayT<PairValueJacobianHessian<Function::num_coords>, 1, MemorySpace::Host> host( results );
  const ArrayT<IndexT, 1, MemorySpace::Host> host_num_evaluations( num_evaluations );
  for ( IndexT i = 0; i < cases.size(); ++i ) {
    SCOPED_TRACE( i );
    EXPECT_EQ( host[i].quadrature_result.status, host_input[i].status );
    EXPECT_EQ( host[i].quadrature_result.num_required_points, i == 3 ? 0 : 2 );
    EXPECT_EQ( host[i].has_active_points, host_input[i].active );
    EXPECT_EQ( host_num_evaluations[i], i < 2 ? Function::num_coords : 1 );
    EXPECT_DOUBLE_EQ( host[i].value, i == 0 ? 45.0 : 0.0 );
    EXPECT_DOUBLE_EQ( host[i].jacobian[0], i == 0 ? 30.0 : 0.0 );
    EXPECT_DOUBLE_EQ( host[i].jacobian[1], i == 0 ? 9.0 : 0.0 );
    EXPECT_DOUBLE_EQ( host[i].hessian( 0, 0 ), i < 2 ? 10.0 : 0.0 );
    EXPECT_DOUBLE_EQ( host[i].hessian( 0, 1 ), i == 0 ? 6.0 : 0.0 );
    EXPECT_DOUBLE_EQ( host[i].hessian( 1, 0 ), i == 0 ? 6.0 : 0.0 );
    EXPECT_DOUBLE_EQ( host[i].hessian( 1, 1 ), 0.0 );
  }
}

/**
 * @brief Compare a weighted pair function's derivatives with directional coordinate differences.
 * @tparam Function Vector pair function with fixed output and coordinate counts.
 * @param function Fixed configuration, possibly holding a frozen reference rule.
 * @param coordinates Current oblique geometry, away from activation and overlap switches.
 * @param weights Scalar contraction used for the gradient and Hessian check.
 */
template <typename Function>
void checkPairDerivatives( const Function& function, const MdArray<RealT, Function::num_coords>& coordinates,
                           const MdArray<RealT, Function::num_values>& weights )
{
  using CoordinateFunction = PairCoordinateFunction<Function>;
  const CoordinateFunction coordinate_function{ function };
  constexpr RealT step = 1.0e-6;
  const auto base =
      EnzymePairDerivative<CoordinateFunction>::valuesAndJacobian( coordinates.view(), coordinate_function );
  const auto weighted = EnzymePairDerivative<CoordinateFunction>::weightedJacobianAndHessian(
      coordinates.view(), coordinate_function, weights.view() );
  for ( IndexT column = 0; column < Function::num_coords; ++column ) {
    auto plus = coordinates, minus = coordinates;
    plus[column] += step;
    minus[column] -= step;
    const auto positive =
        EnzymePairDerivative<CoordinateFunction>::valuesAndJacobian( plus.view(), coordinate_function );
    const auto negative =
        EnzymePairDerivative<CoordinateFunction>::valuesAndJacobian( minus.view(), coordinate_function );
    RealT positive_scalar = 0.0;
    RealT negative_scalar = 0.0;
    for ( IndexT value = 0; value < Function::num_values; ++value ) {
      positive_scalar += weights[value] * positive.values[value];
      negative_scalar += weights[value] * negative.values[value];
      EXPECT_NEAR( base.jacobian( value, column ), ( positive.values[value] - negative.values[value] ) / ( 2.0 * step ),
                   1.0e-6 );
    }
    EXPECT_NEAR( weighted.jacobian[column], ( positive_scalar - negative_scalar ) / ( 2.0 * step ), 1.0e-6 );
    for ( IndexT row = 0; row < Function::num_coords; ++row ) {
      RealT positive_gradient = 0.0;
      RealT negative_gradient = 0.0;
      for ( IndexT value = 0; value < Function::num_values; ++value ) {
        positive_gradient += weights[value] * positive.jacobian( value, row );
        negative_gradient += weights[value] * negative.jacobian( value, row );
      }
      EXPECT_NEAR( weighted.hessian( row, column ), ( positive_gradient - negative_gradient ) / ( 2.0 * step ),
                   2.0e-5 );
    }
  }
}

/** @brief Check both pair definitions for varying and frozen quadrature. @tparam NumPoints Gauss point count. */
template <int NumPoints>
void checkRuleDerivatives()
{
  using RuleGenerator =
      PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>;
  using Gap = OffsetGap<OpposingNormalGap>;
  const RuleGenerator rule_generator;
  const Gap gap{ {}, 0.05 };
  const EnergyMortarEnergy<RuleGenerator, Gap> energy{ rule_generator, 7.0, gap };
  const EnergyMortarNodalEnergy<RuleGenerator, Gap> nodal{ rule_generator, gap };
  const MdArray<RealT, decltype( energy )::num_coords> coordinates{ 0.0, 0.0, 2.0, 0.15, 1.7, 0.35, 0.3, 0.15 };
  const auto check = [&]( const auto& selected_energy, const auto& selected_nodal ) {
    using SelectedEnergy = std::remove_cvref_t<decltype( selected_energy )>;
    using SelectedNodal = std::remove_cvref_t<decltype( selected_nodal )>;
    ASSERT_EQ( PairCoordinateFunction{ selected_nodal }.evaluate( coordinates.view() ).quadrature_result.status,
               QuadratureStatus::Success );
    checkPairDerivatives( selected_energy, coordinates, MdArray<RealT, SelectedEnergy::num_values>{ 1.0 } );
    checkPairDerivatives( selected_nodal, coordinates,
                          MdArray<RealT, SelectedNodal::num_values>{ 0.3, -0.2, 0.4, 0.1 } );
  };
  SCOPED_TRACE( testing::Message() << "points=" << NumPoints );
  {
    SCOPED_TRACE( "varying quadrature" );
    check( energy, nodal );
  }
  {
    SCOPED_TRACE( "frozen quadrature" );
    using FrozenRuleGenerator = detail::FrozenPairQuadratureRuleGenerator<RuleGenerator>;
    const FrozenRuleGenerator frozen_rule_generator{ rule_generator, coordinates.view() };
    check( EnergyMortarEnergy<FrozenRuleGenerator, Gap>{ frozen_rule_generator, 7.0, gap },
           EnergyMortarNodalEnergy<FrozenRuleGenerator, Gap>{ frozen_rule_generator, gap } );
  }
}

// Perturbing every coordinate changes lengths and normals as well as overlap bounds for all supported rules.
TEST( EnergyMortarDerivatives, differentiates_direct_definitions )
{
  checkRuleDerivatives<1>();
  checkRuleDerivatives<2>();
  checkRuleDerivatives<3>();
}
#endif
}  // namespace

/** @brief Run primal and optional Enzyme tests. @param argc Argument count. @param argv Arguments. @return Status. */
int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
