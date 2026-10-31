// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Local residuals and Jacobians from energy or residual definitions, with default Enzyme derivatives.
 */
#ifndef SRC_TRIBOL_PHYSICS_PAIREVALUATION_HPP_
#define SRC_TRIBOL_PHYSICS_PAIREVALUATION_HPP_

#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"

namespace tribol {

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
#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_PAIREVALUATION_HPP_
