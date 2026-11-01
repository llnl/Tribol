// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check local energy/residual definitions and their derivative interface independently of MFEM assembly.
 * Portable tests check Energy Mortar energy and activation on every enabled backend. Host Enzyme tests use analytical
 * gradients, Hessians, and nonsymmetric Jacobians. A specialized evaluator is also exercised without Enzyme.
 */
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "ExecutionTest.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/physics/PairEvaluation.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
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

/** @brief Run portable primal energy checks using backend-accessible storage. */
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
      const MdArray<RealT, EnergyMortarEnergy<Rule>::dof_count> coordinates{
          0.0, 0.0, 2.0, 0.0, 2.0, input[i].height, 0.0, input[i].height };
      // Disable endpoint smoothing so the expected integration length is exactly 2.0.
      const EnergyMortarEnergy<Rule> definition{ Rule{ SmoothedProjectedEdgeMapping{ 0.0 } }, 4.0,
                                                 input[i].residual_gap };
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
      const MdArray<RealT, EnergyMortarEnergy<Rule>::dof_count> coordinates{ 0.0, 0.0, 0.0, 0.0, 2.0, 0.1, 0.0, 0.1 };
      failed_output[0] = EnergyMortarEnergy<Rule>{ Rule{}, 4.0, 0.0 }.evaluate( coordinates.view() );
    } );
    const ArrayT<EnergyMortarEnergyValue, 1, MemorySpace::Host> invalid( failed );
    EXPECT_EQ( invalid[0].quadrature_result.status, QuadratureStatus::InvalidGeometry );
    EXPECT_EQ( invalid[0].quadrature_result.required_point_count, 0 );
    EXPECT_EQ( invalid[0].energy, 0.0 );
    EXPECT_FALSE( invalid[0].has_active_points );
  }
};

// All rule counts must preserve penalty energy, residual-gap shifts, opening, and zero-gap activation.
TEST_P( PairDefinitionTest, computes_energy_from_current_geometry )
{
  checkEnergy<1>();
  checkEnergy<2>();
  checkEnergy<3>();
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
