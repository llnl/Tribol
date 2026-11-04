// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Local residuals, gradients, Jacobians, and Hessians from pair definitions, with default Enzyme derivatives.
 */
#ifndef SRC_TRIBOL_PHYSICS_PAIREVALUATION_HPP_
#define SRC_TRIBOL_PHYSICS_PAIREVALUATION_HPP_

#include <utility>
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/** @brief Pair energy, quadrature status, and compressive-point activation. */
struct PairEnergyValue {
  RealT energy = 0.0;                    ///< Integrated energy, zero for empty or inactive rules.
  bool has_active_points = false;        ///< Whether a compressive point exists, including zero gap.
  QuadratureResult quadrature_result{};  ///< Failed quadrature output must not be consumed.
};
/** @brief Pair residual, quadrature status, and activation. @tparam DofCount Pair coordinate count. */
template <IndexT DofCount>
struct PairResidualValue {
  static constexpr IndexT coordinate_count = DofCount;  ///< Pair coordinate and residual count.
  MdArray<RealT, DofCount> residual{};                  ///< Residual in element/node/component order.
  QuadratureResult quadrature_result{};                 ///< Failed quadrature output must not be consumed.
  bool has_active_points = false;                       ///< Whether a compressive point exists, including zero gap.
};

/**
 * @brief Own local residual entries and their coordinate derivative.
 * @tparam DofCount Number of coordinate and residual entries, ordered by element, node, and spatial component.
 */
template <IndexT DofCount>
struct PairResidualJacobian {
  MdArray<RealT, DofCount> residual{};            ///< Residual in pair coordinate order.
  MdArray<RealT, DofCount, DofCount> jacobian{};  ///< Row-major matrix: residual rows, coordinate columns.
};

/**
 * @brief Own energy, its coordinate gradient, and its Hessian.
 * @tparam DofCount Number of pair coordinate entries.
 */
template <IndexT DofCount>
struct PairEnergyDerivatives : PairResidualJacobian<DofCount> {
  RealT energy{ 0.0 };  ///< Energy whose gradient is residual and whose Hessian is jacobian.
};

/**
 * @brief Own scalar values and their coordinate gradients.
 * @tparam ValueCount Number of scalar outputs.
 * @tparam DofCount Number of active coordinate entries.
 */
template <IndexT ValueCount, IndexT DofCount>
struct PairVectorGradients {
  MdArray<RealT, ValueCount> values{};               ///< Scalar outputs in definition order.
  MdArray<RealT, ValueCount, DofCount> gradients{};  ///< Row-major matrix: output rows, coordinate columns.
};

/**
 * @brief Evaluate an energy definition and its derivatives on the host.
 *
 * Definition supplies a positive static dof_count and energy(MdSpan<const RealT, dof_count>) returning RealT.
 * It holds only configuration and inactive field inputs; geometry-dependent calculations must use the coordinate
 * argument. Specialize this class for a definition to supply other derivative calculations, including without Enzyme.
 * Both evaluate overloads must return PairEnergyDerivatives<Definition::dof_count>.
 * @tparam Definition Energy calculation and fixed inputs. No object or coordinate view is retained.
 */
template <typename Definition>
class EnergyPairEvaluator;

/**
 * @brief Evaluate a force/residual definition and its Jacobian on the host.
 *
 * Definition supplies a positive static dof_count and residual(MdSpan<const RealT, dof_count>) returning
 * MdArray<RealT, dof_count>. Fixed inputs are inactive; current geometry must come from the coordinate argument.
 * Specialize this class for a definition to supply other derivatives, including without Enzyme. Both evaluate
 * overloads must return PairResidualJacobian<Definition::dof_count>. No symmetry is assumed.
 * @tparam Definition Residual calculation and fixed inputs. No object or coordinate view is retained.
 */
template <typename Definition>
class ForcePairEvaluator;

/**
 * @brief Evaluate a vector of scalar integrals and their first coordinate derivatives on the host.
 *
 * Definition supplies positive static value_count and dof_count and values(MdSpan<const RealT, dof_count>) returning
 * MdArray<RealT, value_count>. Fixed inputs are inactive; current geometry must come from the coordinate argument.
 * Specialize this class for a definition to supply other derivatives, including without Enzyme.
 * @tparam Definition Vector calculation and fixed inputs. No object or coordinate view is retained.
 */
template <typename Definition>
class VectorPairEvaluator;

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Default energy evaluation using reverse-mode gradients and forward-over-reverse Hessians.
 * @tparam Definition Energy calculation with fixed inputs and a static dof_count.
 */
template <typename Definition>
class EnergyPairEvaluator {
 public:
  static constexpr IndexT dof_count = Definition::dof_count;  ///< Active pair coordinate count.
  using Result = PairEnergyDerivatives<dof_count>;            ///< Owned energy, residual, and Jacobian.

  /**
   * @brief Calculate energy and both coordinate derivatives.
   * @param coordinates Current coordinates in pair DOF order.
   * @param definition Energy calculation with inputs held constant during differentiation.
   * @return Energy, residual, and Jacobian; all entries are initialized, including zero derivatives.
   */
  static Result evaluate( MdSpan<const RealT, dof_count> coordinates, const Definition& definition )
  {
    return evaluate( coordinates, definition, definition.energy( coordinates ) );
  }

  /**
   * @brief Reuse a calculated energy while computing its derivatives.
   * @param coordinates Current coordinates in pair DOF order.
   * @param definition Energy calculation with fixed inputs.
   * @param energy Energy already calculated with these same coordinates and inputs.
   * @return Owned energy, residual, and Jacobian. Geometry and quadrature remain active in derivative calculations.
   */
  static Result evaluate( MdSpan<const RealT, dof_count> coordinates, const Definition& definition, RealT energy )
  {
    Result result{};
    result.energy = energy;
    gradientKernel( coordinates.data(), &definition, result.residual.data() );
    for ( IndexT column = 0; column < dof_count; ++column ) {
      MdArray<RealT, dof_count> seed{}, gradient{}, derivative{};
      seed[column] = 1.0;
      __enzyme_fwddiff<void>( (void*)gradientKernel, TRIBOL_ENZYME_DUP, coordinates.data(), seed.data(),
                              TRIBOL_ENZYME_CONST, (const void*)&definition, TRIBOL_ENZYME_DUP, gradient.data(),
                              derivative.data() );
      for ( IndexT row = 0; row < dof_count; ++row ) result.jacobian( row, column ) = derivative[row];
    }
    return result;
  }

 private:
  /**
   * @brief Expose scalar energy to Enzyme while holding the definition constant.
   * @param coordinates Active coordinate storage.
   * @param definition_void Inactive definition, erased to avoid internal-linkage types in Enzyme arguments.
   * @param[out] energy Calculated energy.
   */
  static void energyKernel( const RealT* coordinates, const void* definition_void, RealT* energy )
  {
    const auto& definition = *static_cast<const Definition*>( definition_void );
    *energy = definition.energy( { coordinates, dof_count } );
  }

  /**
   * @brief Expose the energy gradient for forward differentiation of each Hessian column.
   * @param coordinates Active coordinate storage.
   * @param definition_void Inactive definition.
   * @param[out] gradient Initialized coordinate gradient.
   */
  static void gradientKernel( const RealT* coordinates, const void* definition_void, RealT* gradient )
  {
    MdArray<RealT, dof_count> derivative{};
    RealT energy = 0.0;
    RealT seed = 1.0;
    __enzyme_autodiff<void>( (void*)energyKernel, TRIBOL_ENZYME_DUP, coordinates, derivative.data(),
                             TRIBOL_ENZYME_CONST, definition_void, TRIBOL_ENZYME_DUP, &energy, &seed );
    for ( IndexT row = 0; row < dof_count; ++row ) gradient[row] = derivative[row];
  }
};

/**
 * @brief Default residual evaluation using forward-mode Jacobian columns.
 * @tparam Definition Residual calculation with fixed inputs and a static dof_count.
 */
template <typename Definition>
class ForcePairEvaluator {
 public:
  static constexpr IndexT dof_count = Definition::dof_count;  ///< Active pair coordinate count.
  using Result = PairResidualJacobian<dof_count>;             ///< Owned residual and Jacobian.

  /**
   * @brief Calculate a residual and its coordinate derivative.
   * @param coordinates Current coordinates in pair DOF order.
   * @param definition Residual calculation with inputs held constant during differentiation.
   * @return Owned residual and Jacobian, including initialized zeros on inactive branches.
   */
  static Result evaluate( MdSpan<const RealT, dof_count> coordinates, const Definition& definition )
  {
    const auto residual = definition.residual( coordinates );
    return evaluate( coordinates, definition, residual.view() );
  }

  /**
   * @brief Reuse a calculated residual while computing its Jacobian.
   * @param coordinates Current coordinates in pair DOF order.
   * @param definition Residual calculation with fixed inputs.
   * @param residual Residual already calculated with these same coordinates and inputs.
   * @return Owned residual and generally nonsymmetric Jacobian. Geometry and quadrature remain active.
   */
  static Result evaluate( MdSpan<const RealT, dof_count> coordinates, const Definition& definition,
                          MdSpan<const RealT, dof_count> residual )
  {
    Result result{};
    for ( IndexT row = 0; row < dof_count; ++row ) result.residual[row] = residual[row];
    for ( IndexT column = 0; column < dof_count; ++column ) {
      MdArray<RealT, dof_count> seed{}, value{}, derivative{};
      seed[column] = 1.0;
      __enzyme_fwddiff<void>( (void*)residualKernel, TRIBOL_ENZYME_DUP, coordinates.data(), seed.data(),
                              TRIBOL_ENZYME_CONST, (const void*)&definition, TRIBOL_ENZYME_DUP, value.data(),
                              derivative.data() );
      for ( IndexT row = 0; row < dof_count; ++row ) result.jacobian( row, column ) = derivative[row];
    }
    return result;
  }

 private:
  /**
   * @brief Expose residual entries to Enzyme with the definition held constant.
   * @param coordinates Active coordinate storage.
   * @param definition_void Inactive definition, erased to avoid internal-linkage types in Enzyme arguments.
   * @param[out] residual Initialized residual entries.
   */
  static void residualKernel( const RealT* coordinates, const void* definition_void, RealT* residual )
  {
    const auto& definition = *static_cast<const Definition*>( definition_void );
    const auto value = definition.residual( { coordinates, dof_count } );
    for ( IndexT row = 0; row < dof_count; ++row ) residual[row] = value[row];
  }
};

/**
 * @brief Default evaluation of vector-valued scalar integrals using reverse-mode gradients.
 * @tparam Definition Vector calculation with fixed inputs and static value_count and dof_count.
 */
template <typename Definition>
class VectorPairEvaluator {
 public:
  static constexpr IndexT value_count = Definition::value_count;       ///< Number of scalar outputs.
  static constexpr IndexT dof_count = Definition::dof_count;           ///< Active pair coordinate count.
  using GradientResult = PairVectorGradients<value_count, dof_count>;  ///< Owned values and gradients.

  /**
   * @brief Calculate all scalar outputs and their coordinate gradients.
   * @param coordinates Current coordinates in pair DOF order.
   * @param definition Vector calculation with inputs held constant during differentiation.
   * @return Owned values and gradients; all entries are initialized, including zero derivatives.
   */
  static GradientResult evaluateGradients( MdSpan<const RealT, dof_count> coordinates, const Definition& definition )
  {
    const auto values = definition.values( coordinates );
    return evaluateGradients( coordinates, definition, values.view() );
  }

  /**
   * @brief Reuse calculated scalar outputs while computing their coordinate gradients.
   * @param coordinates Current coordinates in pair DOF order.
   * @param definition Vector calculation with fixed inputs.
   * @param values Scalar outputs already calculated with these same coordinates and inputs.
   * @return Owned values and gradients. Geometry and quadrature remain active in derivative calculations.
   */
  static GradientResult evaluateGradients( MdSpan<const RealT, dof_count> coordinates, const Definition& definition,
                                           MdSpan<const RealT, value_count> values )
  {
    GradientResult result{};
    for ( IndexT output = 0; output < value_count; ++output ) result.values[output] = values[output];
    [&]<IndexT... Outputs>( std::integer_sequence<IndexT, Outputs...> ) {
      ( gradientKernel<Outputs>( coordinates.data(), &definition, result.gradients.data() + Outputs * dof_count ),
        ... );
    }( std::make_integer_sequence<IndexT, value_count>{} );
    return result;
  }

 private:
  /**
   * @brief Select one scalar output so reverse differentiation uses a fixed unit seed.
   * @tparam Output Selected output index.
   * @param coordinates Active coordinate storage.
   * @param definition_void Inactive definition, erased to avoid internal-linkage types in Enzyme arguments.
   * @param[out] value Calculated scalar output.
   */
  template <IndexT Output>
  static void scalarKernel( const RealT* coordinates, const void* definition_void, RealT* value )
  {
    const auto& definition = *static_cast<const Definition*>( definition_void );
    const auto values = definition.values( { coordinates, dof_count } );
    *value = values[Output];
  }

  /**
   * @brief Expose one scalar output's coordinate gradient.
   * @tparam Output Selected output index.
   * @param coordinates Active coordinate storage.
   * @param definition_void Inactive definition.
   * @param[out] gradient Initialized coordinate gradient.
   */
  template <IndexT Output>
  static void gradientKernel( const RealT* coordinates, const void* definition_void, RealT* gradient )
  {
    MdArray<RealT, dof_count> derivative{};
    RealT value = 0.0;
    RealT seed = 1.0;
    __enzyme_autodiff<void>( (void*)scalarKernel<Output>, TRIBOL_ENZYME_DUP, coordinates, derivative.data(),
                             TRIBOL_ENZYME_CONST, definition_void, TRIBOL_ENZYME_DUP, &value, &seed );
    for ( IndexT row = 0; row < dof_count; ++row ) gradient[row] = derivative[row];
  }
};
#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_PAIREVALUATION_HPP_
