// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check local energy/residual definitions, pair integration, and derivative interfaces independently of MFEM
 * assembly. Portable tests check pair integration, geometric and offset gaps, and Energy Mortar energy/activation on
 * every enabled backend. Named
 * nodal outputs are also checked with reordered and repeated integral calls. Host Enzyme tests use analytical
 * gradients, Hessians, and nonsymmetric Jacobians. A specialized evaluator is also exercised without Enzyme.
 */
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "ExecutionTest.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/physics/PairEvaluation.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"
#include "tribol/physics/CommonPlane.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/integ/PairIntegral.hpp"
#include "tribol/integ/PairQuadratureRule.hpp"

namespace pair_evaluation_test {
/** @brief Residual whose evaluator supplies its own derivatives in any build. */
struct SpecializedResidual {
  static constexpr tribol::IndexT dof_count = 2;  ///< Two active variables.
  tribol::RealT scale;                            ///< Fixed coefficient.
  /**
   * @brief Calculate a two-variable linear residual.
   * @param x Active variables.
   * @return Residual entries in variable order.
   */
  tribol::MdArray<tribol::RealT, dof_count> residual( tribol::MdSpan<const tribol::RealT, dof_count> x ) const
  {
    return tribol::MdArray<tribol::RealT, dof_count>{ scale * x[0] + x[1], x[0] - scale * x[1] };
  }
};
inline int specialized_calls = 0;  ///< Number of calls to the specialized derivative calculation.
}  // namespace pair_evaluation_test

namespace tribol {
/** @brief Example of replacing default derivatives for a specific residual definition. */
template <>
class ForcePairEvaluator<pair_evaluation_test::SpecializedResidual> {
 public:
  using Definition = pair_evaluation_test::SpecializedResidual;  ///< Definition using analytical derivatives.
  using Result = PairResidualJacobian<Definition::dof_count>;    ///< Owned residual and Jacobian.
  /**
   * @brief Calculate primal values and use the analytical derivative implementation.
   * @param x Active variables.
   * @param definition Fixed coefficient and primal calculation.
   * @return Residual and its analytical Jacobian.
   */
  static Result evaluate( MdSpan<const RealT, Definition::dof_count> x, const Definition& definition )
  {
    const auto residual = definition.residual( x );
    return evaluate( x, definition, residual.view() );
  }
  /**
   * @brief Reuse supplied primal values while calculating the analytical Jacobian.
   * @param x Unused active variables; this derivative is constant.
   * @param definition Fixed coefficient.
   * @param residual Already calculated residual.
   * @return Owned residual and analytical Jacobian.
   */
  static Result evaluate( [[maybe_unused]] MdSpan<const RealT, Definition::dof_count> x, const Definition& definition,
                          MdSpan<const RealT, Definition::dof_count> residual )
  {
    ++pair_evaluation_test::specialized_calls;
    Result result{};
    for ( IndexT row = 0; row < Definition::dof_count; ++row ) result.residual[row] = residual[row];
    result.jacobian = MdArray<RealT, Definition::dof_count, Definition::dof_count>{
        MdArray<RealT, Definition::dof_count>{ definition.scale, 1.0 },
        MdArray<RealT, Definition::dof_count>{ 1.0, -definition.scale } };
    return result;
  }
};
}  // namespace tribol

namespace {
using namespace tribol;
constexpr RealT tolerance = 1.0e-12;  ///< Roundoff allowance for analytical comparisons.

/** @brief Run portable primal definition checks using backend-accessible storage. */
class PairDefinitionTest : public tribol::detail::ExecutionTest {
 protected:
  /** @brief Check exact parallel-edge energy, activation, and geometry failure. @tparam Count Gauss point count. */
  template <int Count>
  void checkEnergy()
  {
    using Rule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<Count>>;
    struct Case {
      RealT height;           ///< Second-edge height; positive produces penetration for these orientations.
      RealT residual_gap;     ///< Fixed separation offset.
      RealT expected_energy;  ///< Integral of 0.5 * 4.0 * gap^2 over an edge of length 2.0.
      bool active;            ///< Whether the compressive branch is selected, including at zero gap.
    };
    const auto cases = copyToExecutionSpace<Case>(
        { { 0.1, 0.0, 0.04, true }, { -0.1, 0.0, 0.0, false }, { 0.0, 0.0, 0.0, true }, { -0.05, 0.1, 0.01, true } } );
    ArrayT<EnergyMortarEnergyValue> results( cases.size(), cases.size(), allocatorId() );
    const auto input = cases.view();
    const auto output = results.view();
    forAllExec( GetParam(), cases.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
      const MdArray<RealT, EnergyMortarEnergy<Rule, OffsetGap<OpposingNormalGap>>::dof_count> coordinates{
          0.0, 0.0, 2.0, 0.0, 2.0, input[i].height, 0.0, input[i].height };
      // Disable endpoint smoothing so the expected integration length is exactly 2.0.
      const EnergyMortarEnergy<Rule, OffsetGap<OpposingNormalGap>> definition{
          Rule{ SmoothedProjectedEdgeMapping{ 0.0 } }, 4.0,
          EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, input[i].residual_gap } } };
      output[i] = definition.evaluate( coordinates.view() );
    } );
    const ArrayT<Case, 1, MemorySpace::Host> host_cases( cases );
    const ArrayT<EnergyMortarEnergyValue, 1, MemorySpace::Host> host( results );
    for ( IndexT i = 0; i < host.size(); ++i ) {
      SCOPED_TRACE( i );
      EXPECT_EQ( host[i].quadrature_result.status, QuadratureStatus::Success );
      EXPECT_EQ( host[i].quadrature_result.required_point_count, Count );
      EXPECT_EQ( host[i].has_active_points, host_cases[i].active );
      EXPECT_NEAR( host[i].energy, host_cases[i].expected_energy, tolerance );
    }
    ArrayT<EnergyMortarEnergyValue> failed( 1, 1, allocatorId() );
    const auto failed_output = failed.view();
    forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
      // The first edge has coincident endpoints; a failure must be returned before calculating its unit normal.
      const MdArray<RealT, EnergyMortarEnergy<Rule, OffsetGap<OpposingNormalGap>>::dof_count> coordinates{
          0.0, 0.0, 0.0, 0.0, 2.0, 0.1, 0.0, 0.1 };
      failed_output[0] =
          EnergyMortarEnergy<Rule, OffsetGap<OpposingNormalGap>>{
              Rule{}, 4.0, EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, 0.0 } } }
              .evaluate( coordinates.view() );
    } );
    const ArrayT<EnergyMortarEnergyValue, 1, MemorySpace::Host> invalid( failed );
    EXPECT_EQ( invalid[0].quadrature_result.status, QuadratureStatus::InvalidGeometry );
    EXPECT_EQ( invalid[0].quadrature_result.required_point_count, 0 );
    EXPECT_EQ( invalid[0].energy, 0.0 );
    EXPECT_FALSE( invalid[0].has_active_points );
  }
};

/** @brief Calculate area before gap to check that named outputs do not depend on integral call order. */
struct AreaFirstNodalIntegrals {
  static constexpr IndexT node_count = BasisFunction<EDGE, LINEAR_LAGRANGE>::node_count;  ///< Dual element nodes.
  /**
   * @brief Integrate area first, gap second, and area again using the same local quadrature.
   * @tparam IntegrationRule Bound pair quadrature context.
   * @param rule Current geometry and quadrature.
   * @return Named gap and area arrays, independent of evaluation order or repeated integral calls.
   */
  template <typename IntegrationRule>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE NodalPairIntegrals<node_count> integrals( const IntegrationRule& rule ) const
  {
    const auto area = integral( nodal_basis, rule );
    const auto gap = integral( nodal_basis * OpposingNormalGap{}, rule );
    (void)integral( nodal_basis, rule );
    return { gap, area };
  }
};

// Named gap/area outputs must survive reordered and repeated integrals; each node has gap -0.1 and area 1.0.
TEST_P( PairDefinitionTest, keeps_named_integrals_when_evaluation_order_changes )
{
  using Rule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>;
  using Definition = NodalPairDefinition<Rule, AreaFirstNodalIntegrals>;
  ArrayT<typename Definition::PrimalResult> results( 1, 1, allocatorId() );
  const auto output = results.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const Definition definition{ Rule{ SmoothedProjectedEdgeMapping{ 0.0 } } };
    const MdArray<RealT, Definition::dof_count> coordinates{ 0.0, 0.0, 2.0, 0.0, 2.0, 0.1, 0.0, 0.1 };
    output[0] = definition.evaluate( coordinates.view() );
  } );
  const ArrayT<typename Definition::PrimalResult, 1, MemorySpace::Host> host( results );
  EXPECT_EQ( host[0].quadrature_result.status, QuadratureStatus::Success );
  EXPECT_TRUE( host[0].has_active_points );
  for ( IndexT node = 0; node < Definition::node_count; ++node ) {
    EXPECT_NEAR( host[0].value[node], -0.1, tolerance );
    EXPECT_NEAR( host[0].value[Definition::node_count + node], 1.0, tolerance );
  }
}

// The base gaps use current normals; wrapping either gap subtracts exactly the configured offset.
TEST_P( PairDefinitionTest, separates_geometric_gaps_from_their_offsets )
{
  struct Case {
    MdArray<RealT, 4> coordinates1;  ///< First edge endpoints.
    MdArray<RealT, 4> coordinates2;  ///< Second edge endpoints, with opposing orientation.
    RealT opposing_gap;              ///< Expected midpoint gap using normal2 and the alignment correction.
    RealT common_gap;                ///< Expected midpoint separation along the common normal.
    RealT offset;                    ///< Fixed offset applied to both gap definitions.
  };
  struct Result {
    RealT opposing;          ///< Unshifted opposing-normal gap.
    RealT common;            ///< Unshifted common-plane gap.
    RealT prepared_common;   ///< Common-plane gap reusing a prepared frame.
    RealT shifted_opposing;  ///< Opposing-normal gap after subtraction of the offset.
    RealT shifted_common;    ///< Common-plane gap after subtraction of the offset.
    RealT prepared_shifted;  ///< Shifted common-plane gap reusing the frame.
  };
  // The first two cases are parallel edges separated by 0.1, then translated and rotated together.
  // In the oblique case the midpoint separation is 0.4 and normal2 = (1,5)/sqrt(26).
  const auto cases = copyToExecutionSpace<Case>(
      { { MdArray<RealT, 4>{ 0.0, 0.0, 2.0, 0.0 }, MdArray<RealT, 4>{ 2.0, 0.1, 0.0, 0.1 }, -0.1, -0.1, 0.05 },
        { MdArray<RealT, 4>{ 4.0, -3.0, 4.0, -1.0 }, MdArray<RealT, 4>{ 3.9, -1.0, 3.9, -3.0 }, -0.1, -0.1, 0.0 },
        { MdArray<RealT, 4>{ 0.0, 0.0, 2.0, 0.0 }, MdArray<RealT, 4>{ 2.0, 0.2, 0.0, 0.6 }, -0.4 * 25.0 / 26.0,
          -0.4 * std::sqrt( 0.5 * ( 1.0 + 5.0 / std::sqrt( 26.0 ) ) ), 0.05 } } );
  ArrayT<Result> results( cases.size(), cases.size(), allocatorId() );
  const auto input = cases.view();
  const auto output = results.view();
  forAllExec( GetParam(), cases.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
    const LinearEdgeGeometry geometry1( { input[i].coordinates1.data(), LinearEdgeGeometry::Coords::size() } );
    const LinearEdgeGeometry geometry2( { input[i].coordinates2.data(), LinearEdgeGeometry::Coords::size() } );
    const MdArray<RealT, 1> midpoint{ 0.0 };
    const PairQuadraturePoint<LinearEdgeGeometry, LinearEdgeGeometry> point{ geometry1, geometry2, midpoint.view(),
                                                                             midpoint.view() };
    const auto frame = CommonPlaneEdgeMapping::makeFrame( geometry1, geometry2 );
    const OffsetGap shifted_opposing{ OpposingNormalGap{}, input[i].offset };
    const OffsetGap shifted_common{ CommonPlaneGap{}, input[i].offset };
    output[i] = { OpposingNormalGap{}( point ), CommonPlaneGap{}( point ), CommonPlaneGap{}( point, frame ),
                  shifted_opposing( point ),    shifted_common( point ),   shifted_common( point, frame ) };
  } );
  const ArrayT<Case, 1, MemorySpace::Host> host_cases( cases );
  const ArrayT<Result, 1, MemorySpace::Host> host( results );
  for ( IndexT i = 0; i < host.size(); ++i ) {
    SCOPED_TRACE( i );
    EXPECT_NEAR( host[i].opposing, host_cases[i].opposing_gap, tolerance );
    EXPECT_NEAR( host[i].common, host_cases[i].common_gap, tolerance );
    EXPECT_NEAR( host[i].prepared_common, host_cases[i].common_gap, tolerance );
    EXPECT_NEAR( host[i].shifted_opposing, host_cases[i].opposing_gap - host_cases[i].offset, tolerance );
    EXPECT_NEAR( host[i].shifted_common, host_cases[i].common_gap - host_cases[i].offset, tolerance );
    EXPECT_NEAR( host[i].prepared_shifted, host_cases[i].common_gap - host_cases[i].offset, tolerance );
  }
}

#ifdef TRIBOL_USE_ENZYME
// Removing the zero-offset wrapper must preserve both methods' coordinate derivatives, including their current normals.
TEST( GapDefinitions, unshifted_and_zero_offset_definitions_have_equal_derivatives )
{
  using BaseRule = EdgeQuadratureRuleGenerator<3>;
  using EnergyRule = PairQuadratureRule<SmoothedProjectedEdgeMapping, BaseRule>;
  const MdArray<RealT, PairIntegrationRule<EnergyRule>::dof_count> coordinates{ 0.0, 0.0,  2.0, 0.1,
                                                                                1.8, 0.35, 0.2, 0.1 };
  const EnergyMortarEnergy<EnergyRule> unshifted_energy{ EnergyRule{}, 4.0 };
  const EnergyMortarEnergy<EnergyRule, OffsetGap<OpposingNormalGap>> shifted_energy{
      EnergyRule{}, 4.0, EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, 0.0 } } };
  const auto energy1 =
      EnergyPairEvaluator<decltype( unshifted_energy )>::evaluate( coordinates.view(), unshifted_energy );
  const auto energy2 = EnergyPairEvaluator<decltype( shifted_energy )>::evaluate( coordinates.view(), shifted_energy );
  const CommonPlaneForce<BaseRule> unshifted_force{ CommonPlaneQuadratureRule<BaseRule>{}, 4.0 };
  const CommonPlaneForce<BaseRule, OffsetGap<CommonPlaneGap>> shifted_force{
      CommonPlaneQuadratureRule<BaseRule>{}, 4.0, CommonPlaneResidual{ OffsetGap{ CommonPlaneGap{}, 0.0 } } };
  const auto force1 = ForcePairEvaluator<decltype( unshifted_force )>::evaluate( coordinates.view(), unshifted_force );
  const auto force2 = ForcePairEvaluator<decltype( shifted_force )>::evaluate( coordinates.view(), shifted_force );
  EXPECT_GT( energy1.energy, 0.0 );
  EXPECT_LT( force1.residual[1], 0.0 );
  EXPECT_NEAR( energy1.energy, energy2.energy, tolerance );
  for ( IndexT row = 0; row < coordinates.size(); ++row ) {
    EXPECT_NEAR( energy1.residual[row], energy2.residual[row], tolerance );
    EXPECT_NEAR( force1.residual[row], force2.residual[row], tolerance );
    for ( IndexT col = 0; col < coordinates.size(); ++col ) {
      EXPECT_NEAR( energy1.jacobian( row, col ), energy2.jacobian( row, col ), tolerance );
      EXPECT_NEAR( force1.jacobian( row, col ), force2.jacobian( row, col ), tolerance );
    }
  }
}
#endif

// All rule counts must preserve penalty energy, residual-gap shifts, opening, and zero-gap activation.
TEST_P( PairDefinitionTest, computes_energy_from_current_geometry )
{
  checkEnergy<1>();
  checkEnergy<2>();
  checkEnergy<3>();
}

// Pair integration must weight active scalar/array contributions and skip the integrand on empty or invalid rules.
TEST_P( PairDefinitionTest, integrates_scalar_and_vector_point_callables )
{
  using Rule = PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>;
  struct Summary {
    PairIntegralValue<MdArray<RealT, 4>> vector_integral{};
    PairIntegralValue<RealT> filtered_scalar{};
    PairIntegralValue<RealT> invalid_scalar{};
    IndexT invalid_calls{ 0 };
  };
  ArrayT<Summary> device_out( 1, 1, allocatorId() );
  const auto out = device_out.view();
  forAllExec( GetParam(), 1, [=] TRIBOL_HOST_DEVICE( IndexT ) {
    const Rule rule{ SmoothedProjectedEdgeMapping{ 0.0 } };
    const MdArray<RealT, 8> coordinates{ 0.0, 0.0, 2.0, 0.0, 2.0, 0.5, 0.0, 0.5 };
    out[0].vector_integral = integratePair( coordinates.view(), rule, []( const auto& point ) {
      const auto basis1 = point.template evalBasis1<EDGE, LINEAR_LAGRANGE>();
      const auto basis2 = point.template evalBasis2<EDGE, LINEAR_LAGRANGE>();
      const RealT x1 = point.position1()[0];
      return MdArray<RealT, 4>{ basis1[0], basis1[1], basis2[0] + basis2[1], x1 };
    } );
    // Only keep the right half (x1 >= 1.0) of the [0, 2] interval; with 2 Gauss points, one point is active.
    out[0].filtered_scalar =
        integratePair( coordinates.view(), rule, []( const auto& point ) -> QuadraturePointContribution<RealT> {
          if ( point.position1()[0] < 1.0 ) return {};
          return { 1.0, true };
        } );
    const MdArray<RealT, 8> degenerate{ 1.0, 0.0, 1.0, 0.0, 2.0, 0.5, 0.0, 0.5 };
    IndexT calls = 0;
    out[0].invalid_scalar = integratePair( degenerate.view(), rule, [&]( const auto& ) {
      ++calls;
      return 1.0;
    } );
    out[0].invalid_calls = calls;
  } );
  const ArrayT<Summary, 1, MemorySpace::Host> host( device_out );
  EXPECT_EQ( host[0].vector_integral.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_TRUE( host[0].vector_integral.has_active_points );
  EXPECT_NEAR( host[0].vector_integral.value[0], 1.0, tolerance );
  EXPECT_NEAR( host[0].vector_integral.value[1], 1.0, tolerance );
  EXPECT_NEAR( host[0].vector_integral.value[2], 2.0, tolerance );
  EXPECT_NEAR( host[0].vector_integral.value[3], 2.0, tolerance );
  EXPECT_EQ( host[0].filtered_scalar.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_TRUE( host[0].filtered_scalar.has_active_points );
  EXPECT_NEAR( host[0].filtered_scalar.value, 1.0, tolerance );
  EXPECT_EQ( host[0].invalid_scalar.quadrature_result.status, QuadratureStatus::InvalidGeometry );
  EXPECT_FALSE( host[0].invalid_scalar.has_active_points );
  EXPECT_EQ( host[0].invalid_scalar.value, 0.0 );
  EXPECT_EQ( host[0].invalid_calls, 0 );
}
INSTANTIATE_TEST_SUITE_P( execution, PairDefinitionTest, tribol::detail::enabledExecutionModes() );

// Calling the public evaluator must select a method's analytical specialization, even without Enzyme.
TEST( PairEvaluation, uses_specialized_derivatives )
{
  using Definition = pair_evaluation_test::SpecializedResidual;
  const MdArray<RealT, Definition::dof_count> x{ 2.0, -1.0 };
  const Definition definition{ 3.0 };
  pair_evaluation_test::specialized_calls = 0;
  const auto computed = ForcePairEvaluator<Definition>::evaluate( x.view(), definition );
  const auto primal = definition.residual( x.view() );
  const auto reused = ForcePairEvaluator<Definition>::evaluate( x.view(), definition, primal.view() );
  EXPECT_EQ( pair_evaluation_test::specialized_calls, 2 );
  const MdArray<RealT, Definition::dof_count> expected_residual{ 5.0, 5.0 };
  const MdArray<RealT, Definition::dof_count, Definition::dof_count> expected_jacobian{
      MdArray<RealT, Definition::dof_count>{ 3.0, 1.0 }, MdArray<RealT, Definition::dof_count>{ 1.0, -3.0 } };
  for ( IndexT row = 0; row < Definition::dof_count; ++row ) {
    EXPECT_EQ( computed.residual[row], expected_residual[row] );
    EXPECT_EQ( reused.residual[row], expected_residual[row] );
    for ( IndexT column = 0; column < Definition::dof_count; ++column ) {
      EXPECT_EQ( computed.jacobian( row, column ), expected_jacobian( row, column ) );
      EXPECT_EQ( reused.jacobian( row, column ), expected_jacobian( row, column ) );
    }
  }
}

#ifdef TRIBOL_USE_ENZYME
/** @brief Nonlinear energy with a mixed Hessian entry and an independent linear term. */
struct AnalyticalEnergy {
  static constexpr IndexT dof_count = 3;  ///< Active variables, independent of edge geometry sizes.
  RealT scale;                            ///< Fixed nonlinear coefficient.
  RealT load;                             ///< Fixed linear coefficient.
  /** @brief Calculate energy. @param x Active variables. @return Scalar energy. */
  TRIBOL_ENZYME_INLINE RealT energy( MdSpan<const RealT, dof_count> x ) const
  {
    return scale * ( x[0] * x[0] * x[1] + 0.5 * x[1] * x[1] ) + load * x[2];
  }
};

/** @brief Nonlinear residual with distinct off-diagonal Jacobian entries. */
struct AnalyticalResidual {
  static constexpr IndexT dof_count = 3;  ///< Active variables.
  RealT scale;                            ///< Fixed coefficient in the first residual entry.
  RealT load;                             ///< Fixed coupling to the third variable.
  /** @brief Calculate residual. @param x Active variables. @return Residual in variable order. */
  TRIBOL_ENZYME_INLINE MdArray<RealT, dof_count> residual( MdSpan<const RealT, dof_count> x ) const
  {
    return MdArray<RealT, dof_count>{ scale * x[0] * x[1] + load * x[2], x[0] + x[1] * x[1], x[2] * x[2] - x[0] };
  }
};

/** @brief Vector of scalar outputs whose count differs from the active coordinate count. */
struct AnalyticalVector {
  static constexpr IndexT value_count = 2;  ///< Two scalar outputs.
  static constexpr IndexT dof_count = 3;    ///< Three active variables.
  RealT scale;                              ///< Fixed nonlinear coefficient for the first output.
  RealT load;                               ///< Fixed coefficient for the second output.
  /** @brief Calculate scalar outputs. @param x Active variables. @return Scalar outputs in definition order. */
  TRIBOL_ENZYME_INLINE MdArray<RealT, value_count> values( MdSpan<const RealT, dof_count> x ) const
  {
    return MdArray<RealT, value_count>{ scale * x[0] * x[0] * x[1] + x[2], load * x[1] * x[2] - x[0] };
  }
};

// Reverse-mode gradients and forward-over-reverse Hessians must include mixed terms and zero second derivatives.
TEST( PairEvaluation, differentiates_scalar_energy )
{
  const MdArray<RealT, AnalyticalEnergy::dof_count> x{ 1.5, -0.5, 2.0 };
  const AnalyticalEnergy definition{ 4.0, 3.0 };
  const auto computed = EnergyPairEvaluator<AnalyticalEnergy>::evaluate( x.view(), definition );
  const auto reused =
      EnergyPairEvaluator<AnalyticalEnergy>::evaluate( x.view(), definition, definition.energy( x.view() ) );
  const MdArray<RealT, AnalyticalEnergy::dof_count> gradient{ -6.0, 7.0, 3.0 };
  const MdArray<RealT, AnalyticalEnergy::dof_count, AnalyticalEnergy::dof_count> hessian{
      MdArray<RealT, AnalyticalEnergy::dof_count>{ -4.0, 12.0, 0.0 },
      MdArray<RealT, AnalyticalEnergy::dof_count>{ 12.0, 4.0, 0.0 },
      MdArray<RealT, AnalyticalEnergy::dof_count>{ 0.0, 0.0, 0.0 } };
  EXPECT_EQ( computed.energy, 2.0 );
  EXPECT_EQ( reused.energy, computed.energy );
  for ( IndexT row = 0; row < AnalyticalEnergy::dof_count; ++row ) {
    EXPECT_NEAR( computed.residual[row], gradient[row], tolerance );
    EXPECT_EQ( reused.residual[row], computed.residual[row] );
    for ( IndexT column = 0; column < AnalyticalEnergy::dof_count; ++column ) {
      EXPECT_NEAR( computed.jacobian( row, column ), hessian( row, column ), tolerance );
      EXPECT_EQ( reused.jacobian( row, column ), computed.jacobian( row, column ) );
    }
  }
}

// Forward-mode columns must preserve residual-row/coordinate-column order without assuming symmetry.
TEST( PairEvaluation, differentiates_nonsymmetric_residual )
{
  const MdArray<RealT, AnalyticalResidual::dof_count> x{ 1.5, -0.5, 2.0 };
  const AnalyticalResidual definition{ 4.0, 3.0 };
  const auto computed = ForcePairEvaluator<AnalyticalResidual>::evaluate( x.view(), definition );
  const auto primal = definition.residual( x.view() );
  const auto reused = ForcePairEvaluator<AnalyticalResidual>::evaluate( x.view(), definition, primal.view() );
  const MdArray<RealT, AnalyticalResidual::dof_count> residual{ 3.0, 1.75, 2.5 };
  const MdArray<RealT, AnalyticalResidual::dof_count, AnalyticalResidual::dof_count> jacobian{
      MdArray<RealT, AnalyticalResidual::dof_count>{ -2.0, 6.0, 3.0 },
      MdArray<RealT, AnalyticalResidual::dof_count>{ 1.0, -1.0, 0.0 },
      MdArray<RealT, AnalyticalResidual::dof_count>{ -1.0, 0.0, 4.0 } };
  for ( IndexT row = 0; row < AnalyticalResidual::dof_count; ++row ) {
    EXPECT_NEAR( computed.residual[row], residual[row], tolerance );
    EXPECT_EQ( reused.residual[row], computed.residual[row] );
    for ( IndexT column = 0; column < AnalyticalResidual::dof_count; ++column ) {
      EXPECT_NEAR( computed.jacobian( row, column ), jacobian( row, column ), tolerance );
      EXPECT_EQ( reused.jacobian( row, column ), computed.jacobian( row, column ) );
    }
  }
}

// Vector outputs must support reused values, reverse gradients, full Hessians, and leading-output Hessians.
TEST( PairEvaluation, differentiates_vector_outputs_and_leading_hessians )
{
  const MdArray<RealT, AnalyticalVector::dof_count> x{ 1.5, -0.5, 2.0 };
  const AnalyticalVector definition{ 4.0, 3.0 };
  const auto computed = VectorPairEvaluator<AnalyticalVector>::evaluateGradients( x.view(), definition );
  const auto primal = definition.values( x.view() );
  const auto reused = VectorPairEvaluator<AnalyticalVector>::evaluateGradients( x.view(), definition, primal.view() );
  const auto hessians = VectorPairEvaluator<AnalyticalVector>::evaluateHessians( x.view(), definition );
  const auto leading = VectorPairEvaluator<AnalyticalVector>::evaluateHessians<1>( x.view(), definition );
  const MdArray<RealT, AnalyticalVector::value_count> expected_values{ -2.5, -4.5 };
  const MdArray<RealT, AnalyticalVector::value_count, AnalyticalVector::dof_count> expected_gradients{
      MdArray<RealT, AnalyticalVector::dof_count>{ -6.0, 9.0, 1.0 },
      MdArray<RealT, AnalyticalVector::dof_count>{ -1.0, 6.0, -1.5 } };
  const MdArray<RealT, AnalyticalVector::dof_count, AnalyticalVector::dof_count> first_hessian{
      MdArray<RealT, AnalyticalVector::dof_count>{ -4.0, 12.0, 0.0 },
      MdArray<RealT, AnalyticalVector::dof_count>{ 12.0, 0.0, 0.0 },
      MdArray<RealT, AnalyticalVector::dof_count>{ 0.0, 0.0, 0.0 } };
  const MdArray<RealT, AnalyticalVector::dof_count, AnalyticalVector::dof_count> second_hessian{
      MdArray<RealT, AnalyticalVector::dof_count>{ 0.0, 0.0, 0.0 },
      MdArray<RealT, AnalyticalVector::dof_count>{ 0.0, 0.0, 3.0 },
      MdArray<RealT, AnalyticalVector::dof_count>{ 0.0, 3.0, 0.0 } };
  for ( IndexT output = 0; output < AnalyticalVector::value_count; ++output ) {
    EXPECT_NEAR( computed.values[output], expected_values[output], tolerance );
    EXPECT_EQ( reused.values[output], computed.values[output] );
    for ( IndexT dof = 0; dof < AnalyticalVector::dof_count; ++dof ) {
      EXPECT_NEAR( computed.gradients( output, dof ), expected_gradients( output, dof ), tolerance );
      EXPECT_EQ( reused.gradients( output, dof ), computed.gradients( output, dof ) );
    }
  }
  for ( IndexT row = 0; row < AnalyticalVector::dof_count; ++row ) {
    for ( IndexT column = 0; column < AnalyticalVector::dof_count; ++column ) {
      EXPECT_NEAR( hessians( 0, row, column ), first_hessian( row, column ), tolerance );
      EXPECT_NEAR( hessians( 1, row, column ), second_hessian( row, column ), tolerance );
      EXPECT_DOUBLE_EQ( leading( 0, row, column ), hessians( 0, row, column ) );
      EXPECT_EQ( leading( 1, row, column ), 0.0 );
    }
  }
}
#endif  // TRIBOL_USE_ENZYME
}  // namespace

/** @brief Run portable primal and host derivative tests. @param argc Argument count. @param argv Arguments.
 * @return Test status. */
int main( int argc, char** argv )
{
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
