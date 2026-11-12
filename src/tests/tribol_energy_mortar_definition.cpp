// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check direct Energy Mortar energies, nodal integrals, and explicit field partials on enabled backends.
 * Host Enzyme checks compare the new pair definitions with finite differences for each Gauss count and derivative
 * mode. The energy kernel checks active, open, zero-gap, and invalid pairs on every enabled backend, including the
 * compressive zero-gap Hessian. Forward AD checks values and quadrature state without an extra value evaluation.
 * Compile-time checks accept the supported pair interfaces and reject invalid sizes, results, and coordinate splits.
 * Returned buffers retain the pair allocator and remain valid after the input pair list is destroyed. Weighted
 * Jacobian/Hessian kernels check reordered pairs with unequal weights. MFEM tests separately check assembly and
 * shared-node normalization.
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
struct PairFunctionCandidate {
  static constexpr IndexT value_count = 2;  ///< Two output values.
  static constexpr IndexT dof_count = 4;    ///< Four coordinate entries.

  /** @brief Declare a const evaluation. @param coordinates Pair inputs. @return Values and quadrature state. */
  PairResidualValue<value_count> evaluate( MdSpan<const RealT, dof_count> coordinates ) const;
};

/** @brief Valid mesh-kernel input splits four coordinates between two geometries. */
struct PairKernelFunctionCandidate : PairFunctionCandidate {
  static constexpr IndexT first_dofs = 2;  ///< Two coordinates for each geometry.
};

/** @brief Zero outputs must be rejected before instantiating fixed-size numerical arrays. */
struct ZeroValuePairFunction : PairFunctionCandidate {
  static constexpr IndexT value_count = 0;  ///< Invalid output count.
};

/** @brief Zero coordinates must be rejected before instantiating a coordinate view. */
struct ZeroDofPairFunction : PairFunctionCandidate {
  static constexpr IndexT dof_count = 0;  ///< Invalid coordinate count.
};

/** @brief Output counts must be compile-time constants. */
struct RuntimeSizePairFunction : PairFunctionCandidate {
  IndexT value_count = 2;  ///< Invalid runtime output count.
};

/** @brief The returned value array must agree with the declared output count. */
struct WrongValueCountPairFunction : PairFunctionCandidate {
  static constexpr IndexT value_count = 1;  ///< Disagrees with the inherited two-entry result.
};

/** @brief Pair evaluation must be callable on a const function object. */
struct MutablePairFunction : PairFunctionCandidate {
  /** @brief Declare a mutable-only evaluation. @param coordinates Pair inputs. @return Values and quadrature state. */
  PairResidualValue<value_count> evaluate( MdSpan<const RealT, dof_count> coordinates );
};

/** @brief An array alone supplies no quadrature status or activation. */
struct MissingStatePairFunction : PairFunctionCandidate {
  /** @brief Declare a result without quadrature state. @param coordinates Pair inputs. @return Numerical values only.
   */
  MdArray<RealT, value_count> evaluate( MdSpan<const RealT, dof_count> coordinates ) const;
};

/** @brief The numerical values must be readable from a const evaluation result. */
struct MutableValuesPairFunction : PairFunctionCandidate {
  /** @brief Result that hides the const values() accessor with a mutable-only accessor. */
  struct Result : PairResidualValue<value_count> {
    /** @brief Declare mutable-only value access. @return Pair values. */
    MdArray<RealT, value_count> values();
  };
  /** @brief Declare an incompatible result. @param coordinates Pair inputs. @return Mutable-only numerical values. */
  Result evaluate( MdSpan<const RealT, dof_count> coordinates ) const;
};

/** @brief Activation must have the expected bool type. */
struct WrongActivationPairFunction : PairFunctionCandidate {
  /** @brief Result with an integer flag that hides the required bool member. */
  struct Result : PairResidualValue<value_count> {
    IndexT has_active_points = 0;  ///< Incompatible activation type.
  };
  /** @brief Declare an incompatible result. @param coordinates Pair inputs. @return Values with an integer flag. */
  Result evaluate( MdSpan<const RealT, dof_count> coordinates ) const;
};

/** @brief Quadrature state must have the expected QuadratureResult type. */
struct WrongQuadraturePairFunction : PairFunctionCandidate {
  /** @brief Result whose status member omits the required point count. */
  struct Result : PairResidualValue<value_count> {
    QuadratureStatus quadrature_result{};  ///< Incompatible quadrature result type.
  };
  /** @brief Declare an incompatible result. @param coordinates Pair inputs. @return Values without a point count. */
  Result evaluate( MdSpan<const RealT, dof_count> coordinates ) const;
};

/** @brief A coordinate split must leave entries for geometry1. */
struct EmptyFirstGeometryPairFunction : PairKernelFunctionCandidate {
  static constexpr IndexT first_dofs = 0;  ///< Invalid empty geometry1.
};

/** @brief A coordinate split must leave entries for geometry2. */
struct EmptySecondGeometryPairFunction : PairKernelFunctionCandidate {
  static constexpr IndexT first_dofs = dof_count;  ///< Invalid empty geometry2.
};

// Differentiation needs a const evaluation and fixed numerical sizes; it does not require a mesh coordinate split.
static_assert( PairFunctionType<PairFunctionCandidate> );
static_assert( !PairKernelFunctionType<PairFunctionCandidate> );
static_assert( PairKernelFunctionType<PairKernelFunctionCandidate> );
static_assert( !PairFunctionType<ZeroValuePairFunction> );
static_assert( !PairFunctionType<ZeroDofPairFunction> );
static_assert( !PairFunctionType<RuntimeSizePairFunction> );
static_assert( !PairFunctionType<WrongValueCountPairFunction> );
static_assert( !PairFunctionType<MutablePairFunction> );
static_assert( !PairFunctionType<MissingStatePairFunction> );
static_assert( !PairFunctionType<MutableValuesPairFunction> );
static_assert( !PairFunctionType<WrongActivationPairFunction> );
static_assert( !PairFunctionType<WrongQuadraturePairFunction> );
static_assert( !PairKernelFunctionType<EmptyFirstGeometryPairFunction> );
static_assert( !PairKernelFunctionType<EmptySecondGeometryPairFunction> );

// Both energy and nodal-integral functions implement the same evaluation and coordinate-packing interface.
static_assert( PairKernelFunctionType<EnergyMortarEnergy<
                   PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>>> );
static_assert( PairKernelFunctionType<EnergyMortarNodalEnergy<
                   PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>>> );

/** @brief Pair energy and nodal integrals written by one backend thread. @tparam Count Gauss point count. */
template <int Count>
struct ParallelIntegralResult {
  using RuleGenerator = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<Count>>;
  PairEnergyValue energy;                                               ///< Quadrature-point energy and status.
  typename EnergyMortarNodalEnergy<RuleGenerator>::PrimalResult nodal;  ///< Nodal integrals and status.
};

/** @brief Nodal field inputs and exact penalty partials for one backend test case. */
struct NodalPartialsCase {
  RealT gap;                           ///< Weighted nodal gap.
  RealT area;                          ///< Tributary area, including unsupported values on both sides of zero.
  EnergyMortarNodalPartials expected;  ///< Exact energy and field partials for stiffness four.
};

/** @brief Check constant-gap integrals on an unsmoothed interval. @tparam Count Gauss point count.
 * @param mode Execution backend. @param allocator Backend-accessible allocation resource. */
template <int Count>
void checkParallelIntegrals( ExecutionMode mode, int allocator )
{
  using RuleGenerator = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<Count>>;
  using Energy = EnergyMortarEnergy<RuleGenerator>;
  using Nodal = EnergyMortarNodalEnergy<RuleGenerator>;
  ArrayT<ParallelIntegralResult<Count>> results( 1, 1, allocator );
  const auto output = results.view();
  forAllExec( mode, 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const MdArray<RealT, Energy::dof_count> coordinates{ 0.0, 0.0, 2.0, 0.0, 3.0, 0.1, -1.0, 0.1 };
    const RuleGenerator rule_generator( SmoothedProjectedEdgeMapping{ 0.0 } );
    output[0] = { Energy{ rule_generator, 4.0 }.evaluate( coordinates.view() ),
                  Nodal{ rule_generator }.evaluate( coordinates.view() ) };
  } );
  const ArrayT<ParallelIntegralResult<Count>, 1, MemorySpace::Host> host( results );
  ASSERT_EQ( host[0].energy.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_EQ( host[0].energy.quadrature_result.required_point_count, Count );
  // A length-two edge with gap -0.1 has energy 0.5*4*0.1^2*2, and equal nodal areas of one.
  EXPECT_NEAR( host[0].energy.energy, 0.04, tolerance );
  EXPECT_TRUE( host[0].energy.has_active_points );
  ASSERT_EQ( host[0].nodal.quadrature_result.status, QuadratureStatus::Success );
  for ( IndexT node = 0; node < Nodal::node_count; ++node ) {
    EXPECT_NEAR( host[0].nodal.value[node], -0.1, tolerance );
    EXPECT_NEAR( host[0].nodal.value[Nodal::node_count + node], 1.0, tolerance );
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
  constexpr IndexT pair_count = 3;
  PairValueJacobianBuffer<2, 2> values( pair_count, allocatorId() );
  ElementBlockBuffer<2, 2> blocks( pair_count, allocatorId() );
  const auto value_jacobian_view = values.view();
  const auto block_view = blocks.view();
  forAllExec( GetParam(), pair_count, [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
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
  for ( IndexT pair = 0; pair < pair_count; ++pair ) {
    EXPECT_DOUBLE_EQ( host_values[pair], 10.0 + pair );
    EXPECT_DOUBLE_EQ( host_values[pair_count + pair], 20.0 + pair );
    EXPECT_DOUBLE_EQ( host_jacobians[pair], 30.0 + pair );
    EXPECT_DOUBLE_EQ( host_jacobians[pair_count + pair], 40.0 + pair );
    EXPECT_DOUBLE_EQ( host_jacobians[2 * pair_count + pair], 50.0 + pair );
    EXPECT_DOUBLE_EQ( host_jacobians[3 * pair_count + pair], 60.0 + pair );
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
  EXPECT_EQ( contributions_view.derivatives.pair_count, pairs.size() );
  EXPECT_EQ( buffer.quadratureResults().getAllocatorID(), pairs.getAllocatorID() );
  EXPECT_EQ( value_jacobian.pairCount(), pairs.size() );
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
    for ( IndexT row = 0; row < Function::dof_count; ++row ) {
      summary( pair, 3 ) += std::abs( contributions_view.derivatives.jacobian( pair, row ) );
      for ( IndexT column = 0; column < Function::dof_count; ++column ) {
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
    EXPECT_EQ( jacobian_status[pair].required_point_count, pair == 3 ? 0 : 2 );
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
    for ( IndexT coordinate = 0; coordinate < Function::dof_count; ++coordinate ) {
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
        [] TRIBOL_HOST_DEVICE( IndexT pair ) { return MdArray<RealT, Function::value_count>{ pair + 2.0 }; } );
  } );
  constexpr IndexT weighted_pair_count = 2;
  EXPECT_EQ( weighted.pairCount(), weighted_pair_count );
  EXPECT_EQ( weighted.jacobians().getAllocatorID(), pairs.getAllocatorID() );
  EXPECT_EQ( weighted.hessians().getAllocatorID(), pairs.getAllocatorID() );
  const Array1D<RealT, MemorySpace::Host> weighted_jacobians( weighted.jacobians() );
  const Array1D<RealT, MemorySpace::Host> weighted_hessians( weighted.hessians() );
  for ( IndexT pair = 0; pair < weighted_pair_count; ++pair ) {
    const IndexT original_pair = pair == 0 ? 2 : 0;
    const RealT weight = pair + 2.0;
    for ( IndexT coordinate = 0; coordinate < Function::dof_count; ++coordinate ) {
      EXPECT_NEAR( weighted_jacobians[coordinate * weighted_pair_count + pair],
                   weight * jacobians[coordinate * pairs.size() + original_pair], tolerance );
    }
    RealT vertical_hessian = 0.0;
    for ( IndexT row : { 1, 3 } ) {
      for ( IndexT column : { 1, 3 } ) {
        vertical_hessian += weighted_hessians[( row * Function::dof_count + column ) * weighted_pair_count + pair];
      }
    }
    EXPECT_NEAR( vertical_hessian, weight * host( original_pair, 2 ), tolerance );
  }
}

/** @brief Small fixed-size pair function used to compile and execute both Enzyme operations on every backend. */
struct PolynomialPairFunction {
  static constexpr IndexT value_count = 2;              ///< Number of outputs.
  static constexpr IndexT dof_count = 2;                ///< Number of active coordinates.
  using PrimalResult = PairResidualValue<value_count>;  ///< Polynomial values with quadrature state.
  IndexT* evaluation_count = nullptr;                   ///< Optional counter for complete function evaluations.

  /** @brief Evaluate two coupled polynomial outputs. @param coordinates Active coordinates.
   * @return Function values with successful, active quadrature state. */
  TRIBOL_HOST_DEVICE PrimalResult evaluate( MdSpan<const RealT, dof_count> coordinates ) const
  {
    if ( evaluation_count != nullptr ) {
      ++*evaluation_count;
    }
    return { MdArray<RealT, value_count>{ coordinates[0] * coordinates[0] + coordinates[1],
                                          coordinates[0] * coordinates[1] },
             { QuadratureStatus::Success, 2 },
             true };
  }
};

/** @brief Device-storable output from both pair derivative operations. */
struct PolynomialDerivativeResult {
  PairValueJacobian<PolynomialPairFunction::value_count, PolynomialPairFunction::dof_count> value_jacobian;
  PairWeightedJacobianHessian<PolynomialPairFunction::dof_count> weighted_hessian;
  IndexT evaluation_count = 0;           ///< Complete evaluations used for values and Jacobian.
  IndexT weighted_evaluation_count = 0;  ///< Complete evaluations used for the weighted Jacobian and Hessian.
};

// Jacobian and Hessian AD calls must also supply their primal outputs without separate evaluations.
TEST_P( EnergyMortarDefinitionTest, enzyme_pair_derivatives_execute_on_backend )
{
  Array1D<PolynomialDerivativeResult> results( 1, 1, allocatorId() );
  auto output = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const MdArray<RealT, PolynomialPairFunction::dof_count> coordinates{ 3.0, 5.0 };
    const MdArray<RealT, PolynomialPairFunction::value_count> weights{ 2.0, -1.0 };
    IndexT evaluation_count = 0;
    output[0].value_jacobian = EnzymePairDerivative<PolynomialPairFunction>::valuesAndJacobian(
        coordinates.view(), PolynomialPairFunction{ &evaluation_count } );
    output[0].evaluation_count = evaluation_count;
    IndexT weighted_evaluation_count = 0;
    output[0].weighted_hessian = EnzymePairDerivative<PolynomialPairFunction>::weightedJacobianAndHessian(
        coordinates.view(), PolynomialPairFunction{ &weighted_evaluation_count }, weights.view() );
    output[0].weighted_evaluation_count = weighted_evaluation_count;
  } );
  const Array1D<PolynomialDerivativeResult, MemorySpace::Host> host_results( results );
  const auto& value_jacobian = host_results[0].value_jacobian;
  EXPECT_EQ( host_results[0].evaluation_count, PolynomialPairFunction::dof_count );
  EXPECT_EQ( value_jacobian.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_EQ( value_jacobian.quadrature_result.required_point_count, 2 );
  EXPECT_TRUE( value_jacobian.has_active_points );
  EXPECT_DOUBLE_EQ( value_jacobian.values[0], 14.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.values[1], 15.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.jacobian( 0, 0 ), 6.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.jacobian( 0, 1 ), 1.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.jacobian( 1, 0 ), 5.0 );
  EXPECT_DOUBLE_EQ( value_jacobian.jacobian( 1, 1 ), 3.0 );
  EXPECT_EQ( host_results[0].weighted_evaluation_count, PolynomialPairFunction::dof_count );
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
  static constexpr IndexT value_count = 1;  ///< One scalar output.
  static constexpr IndexT dof_count = 2;    ///< Two coordinate entries.
  IndexT* evaluation_count;                 ///< Number of evaluate() calls, in backend-accessible memory.
  bool active;                              ///< Activation reported with the polynomial value.
  QuadratureStatus status;                  ///< Quadrature status reported with the polynomial value.

  /** @brief Evaluate x0^2*x1 with the configured status. @param coordinates Polynomial inputs.
   * @return Scalar value, activation, and status; numerical outputs are retained even for inactive or failed inputs
   * so the test can check that differentiation masks them. */
  TRIBOL_HOST_DEVICE PairEnergyValue evaluate( MdSpan<const RealT, dof_count> coordinates ) const
  {
    ++*evaluation_count;
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
    MdArray<RealT, Function::dof_count> coordinates;  ///< Polynomial inputs.
    bool active;                                      ///< Reported activation.
    QuadratureStatus status;                          ///< Reported quadrature status.
  };
  const auto cases = copyToExecutionSpace<Input>( {
      { MdArray<RealT, Function::dof_count>{ 3.0, 5.0 }, true, QuadratureStatus::Success },
      { MdArray<RealT, Function::dof_count>{ 0.0, 5.0 }, true, QuadratureStatus::Success },
      { MdArray<RealT, Function::dof_count>{ 3.0, 5.0 }, false, QuadratureStatus::Success },
      { MdArray<RealT, Function::dof_count>{ 3.0, 5.0 }, false, QuadratureStatus::InvalidGeometry },
      { MdArray<RealT, Function::dof_count>{ 3.0, 5.0 }, false, QuadratureStatus::InsufficientCapacity },
  } );
  ArrayT<PairValueJacobianHessian<Function::dof_count>> results( cases.size(), cases.size(), allocatorId() );
  ArrayT<IndexT> counts( cases.size(), cases.size(), allocatorId() );
  const auto input = cases.view();
  const auto output = results.view();
  const auto count = counts.view();
  forAllExec( GetParam(), cases.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
    count[i] = 0;
    const Function function{ &count[i], input[i].active, input[i].status };
    output[i] = EnzymePairDerivative<Function>::valueJacobianAndHessian( input[i].coordinates.view(), function );
  } );
  const ArrayT<Input, 1, MemorySpace::Host> host_input( cases );
  const ArrayT<PairValueJacobianHessian<Function::dof_count>, 1, MemorySpace::Host> host( results );
  const ArrayT<IndexT, 1, MemorySpace::Host> host_counts( counts );
  for ( IndexT i = 0; i < cases.size(); ++i ) {
    SCOPED_TRACE( i );
    EXPECT_EQ( host[i].quadrature_result.status, host_input[i].status );
    EXPECT_EQ( host[i].quadrature_result.required_point_count, i == 3 ? 0 : 2 );
    EXPECT_EQ( host[i].has_active_points, host_input[i].active );
    EXPECT_EQ( host_counts[i], i < 2 ? Function::dof_count : 1 );
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
void checkPairDerivatives( const Function& function, const MdArray<RealT, Function::dof_count>& coordinates,
                           const MdArray<RealT, Function::value_count>& weights )
{
  constexpr RealT step = 1.0e-6;
  const auto base = EnzymePairDerivative<Function>::valuesAndJacobian( coordinates.view(), function );
  const auto weighted =
      EnzymePairDerivative<Function>::weightedJacobianAndHessian( coordinates.view(), function, weights.view() );
  for ( IndexT column = 0; column < Function::dof_count; ++column ) {
    auto plus = coordinates, minus = coordinates;
    plus[column] += step;
    minus[column] -= step;
    const auto positive = EnzymePairDerivative<Function>::valuesAndJacobian( plus.view(), function );
    const auto negative = EnzymePairDerivative<Function>::valuesAndJacobian( minus.view(), function );
    RealT positive_scalar = 0.0;
    RealT negative_scalar = 0.0;
    for ( IndexT value = 0; value < Function::value_count; ++value ) {
      positive_scalar += weights[value] * positive.values[value];
      negative_scalar += weights[value] * negative.values[value];
      EXPECT_NEAR( base.jacobian( value, column ), ( positive.values[value] - negative.values[value] ) / ( 2.0 * step ),
                   1.0e-6 );
    }
    EXPECT_NEAR( weighted.jacobian[column], ( positive_scalar - negative_scalar ) / ( 2.0 * step ), 1.0e-6 );
    for ( IndexT row = 0; row < Function::dof_count; ++row ) {
      RealT positive_gradient = 0.0;
      RealT negative_gradient = 0.0;
      for ( IndexT value = 0; value < Function::value_count; ++value ) {
        positive_gradient += weights[value] * positive.jacobian( value, row );
        negative_gradient += weights[value] * negative.jacobian( value, row );
      }
      EXPECT_NEAR( weighted.hessian( row, column ), ( positive_gradient - negative_gradient ) / ( 2.0 * step ),
                   2.0e-5 );
    }
  }
}

/** @brief Check both pair definitions for varying and frozen quadrature. @tparam Count Gauss point count. */
template <int Count>
void checkRuleDerivatives()
{
  using RuleGenerator = PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<Count>>;
  using Gap = OffsetGap<OpposingNormalGap>;
  const RuleGenerator rule_generator;
  const Gap gap{ {}, 0.05 };
  const EnergyMortarEnergy<RuleGenerator, Gap> energy{ rule_generator, 7.0, gap };
  const EnergyMortarNodalEnergy<RuleGenerator, Gap> nodal{ rule_generator, gap };
  const MdArray<RealT, decltype( energy )::dof_count> coordinates{ 0.0, 0.0, 2.0, 0.15, 1.7, 0.35, 0.3, 0.15 };
  const auto check = [&]( const auto& selected_energy, const auto& selected_nodal ) {
    using SelectedEnergy = std::remove_cvref_t<decltype( selected_energy )>;
    using SelectedNodal = std::remove_cvref_t<decltype( selected_nodal )>;
    ASSERT_EQ( selected_nodal.evaluate( coordinates.view() ).quadrature_result.status, QuadratureStatus::Success );
    checkPairDerivatives( selected_energy, coordinates, MdArray<RealT, SelectedEnergy::value_count>{ 1.0 } );
    checkPairDerivatives( selected_nodal, coordinates,
                          MdArray<RealT, SelectedNodal::value_count>{ 0.3, -0.2, 0.4, 0.1 } );
  };
  SCOPED_TRACE( testing::Message() << "points=" << Count );
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
