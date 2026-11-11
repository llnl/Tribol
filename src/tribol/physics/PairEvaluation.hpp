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
 * @brief Own a weighted pair-function gradient and its coordinate Hessian.
 * @tparam DofCount Number of active coordinate entries.
 */
template <IndexT DofCount>
struct PairWeightedHessian {
  MdArray<RealT, DofCount> gradient{};           ///< Gradient of the weighted scalar pair function.
  MdArray<RealT, DofCount, DofCount> hessian{};  ///< Row-major coordinate Hessian.
};

/**
 * @brief Differentiate a fixed-size vector pair function on the host or device.
 *
 * Function supplies positive static `value_count` and `dof_count` members and a device-callable
 * `values(MdSpan<const RealT, dof_count>)` function. Function configuration and weights remain inactive while pair
 * coordinates are active. The class owns no state, memory, or MFEM objects and can run inside a CUDA or HIP kernel.
 *
 * @tparam Function Fixed-size pair-local vector function.
 */
#ifdef TRIBOL_USE_ENZYME
template <typename Function>
class EnzymePairDerivative {
 public:
  static constexpr IndexT value_count = Function::value_count;        ///< Number of pair-function outputs.
  static constexpr IndexT dof_count = Function::dof_count;            ///< Number of active coordinates.
  using ValueJacobian = PairVectorGradients<value_count, dof_count>;  ///< Values and first derivatives.
  using WeightedHessian = PairWeightedHessian<dof_count>;             ///< Weighted first and second derivatives.

  /**
   * @brief Calculate all function values and their coordinate Jacobian.
   * @param coordinates Current pair coordinates.
   * @param function Fixed pair-function configuration.
   * @return Values and row-major value-by-coordinate Jacobian.
   */
  TRIBOL_HOST_DEVICE static ValueJacobian valuesAndJacobian( MdSpan<const RealT, dof_count> coordinates,
                                                             const Function& function )
  {
    ValueJacobian result{};
    result.values = function.values( coordinates );
    for ( IndexT coordinate = 0; coordinate < dof_count; ++coordinate ) {
      MdArray<RealT, dof_count> coordinate_seed{};
      MdArray<RealT, value_count> values{}, derivatives{};
      coordinate_seed[coordinate] = 1.0;
      __enzyme_fwddiff<void>( (void*)valuesKernel, TRIBOL_ENZYME_DUP, coordinates.data(), coordinate_seed.data(),
                              TRIBOL_ENZYME_CONST, (const void*)&function, TRIBOL_ENZYME_DUP, values.data(),
                              derivatives.data() );
      for ( IndexT value = 0; value < value_count; ++value ) {
        result.gradients( value, coordinate ) = derivatives[value];
      }
    }
    return result;
  }

  /**
   * @brief Calculate derivatives of a weighted scalar contraction of the pair function.
   * @param coordinates Current pair coordinates.
   * @param function Fixed pair-function configuration.
   * @param weights Inactive weights defining `weights^T * function.values(coordinates)`.
   * @return Weighted coordinate gradient and Hessian.
   */
  TRIBOL_HOST_DEVICE static WeightedHessian weightedHessian( MdSpan<const RealT, dof_count> coordinates,
                                                             const Function& function,
                                                             MdSpan<const RealT, value_count> weights )
  {
    WeightedHessian result{};
    weightedGradientKernel( coordinates.data(), &function, weights.data(), result.gradient.data() );
    for ( IndexT column = 0; column < dof_count; ++column ) {
      MdArray<RealT, dof_count> coordinate_seed{}, gradient{}, derivative{};
      coordinate_seed[column] = 1.0;
      __enzyme_fwddiff<void>( (void*)weightedGradientKernel, TRIBOL_ENZYME_DUP, coordinates.data(),
                              coordinate_seed.data(), TRIBOL_ENZYME_CONST, (const void*)&function, TRIBOL_ENZYME_CONST,
                              weights.data(), TRIBOL_ENZYME_DUP, gradient.data(), derivative.data() );
      for ( IndexT row = 0; row < dof_count; ++row ) {
        result.hessian( row, column ) = derivative[row];
      }
    }
    return result;
  }

 private:
  /**
   * @brief Expose all pair-function values for forward coordinate differentiation.
   * @param coordinates Active coordinate storage.
   * @param function_void Inactive function configuration.
   * @param[out] values Initialized pair-function values.
   */
  TRIBOL_HOST_DEVICE static void valuesKernel( const RealT* coordinates, const void* function_void, RealT* values )
  {
    const auto& function = *static_cast<const Function*>( function_void );
    const auto result = function.values( { coordinates, dof_count } );
    for ( IndexT value = 0; value < value_count; ++value ) {
      values[value] = result[value];
    }
  }

  /**
   * @brief Expose the weighted scalar pair function to Enzyme.
   * @param coordinates Active coordinate storage.
   * @param function_void Inactive function configuration.
   * @param weights Inactive scalar weights.
   * @param[out] scalar Weighted scalar value.
   */
  TRIBOL_HOST_DEVICE static void weightedValueKernel( const RealT* coordinates, const void* function_void,
                                                      const RealT* weights, RealT* scalar )
  {
    const auto& function = *static_cast<const Function*>( function_void );
    const auto values = function.values( { coordinates, dof_count } );
    *scalar = 0.0;
    for ( IndexT value = 0; value < value_count; ++value ) {
      *scalar += weights[value] * values[value];
    }
  }

  /**
   * @brief Expose the weighted coordinate gradient for forward-over-reverse Hessians.
   * @param coordinates Active coordinate storage.
   * @param function_void Inactive function configuration.
   * @param weights Inactive scalar weights.
   * @param[out] gradient Initialized coordinate gradient.
   */
  TRIBOL_HOST_DEVICE static void weightedGradientKernel( const RealT* coordinates, const void* function_void,
                                                         const RealT* weights, RealT* gradient )
  {
    MdArray<RealT, dof_count> derivative{};
    RealT scalar = 0.0;
    RealT scalar_seed = 1.0;
    __enzyme_autodiff<void>( (void*)weightedValueKernel, TRIBOL_ENZYME_DUP, coordinates, derivative.data(),
                             TRIBOL_ENZYME_CONST, function_void, TRIBOL_ENZYME_CONST, weights, TRIBOL_ENZYME_DUP,
                             &scalar, &scalar_seed );
    for ( IndexT coordinate = 0; coordinate < dof_count; ++coordinate ) {
      gradient[coordinate] = derivative[coordinate];
    }
  }
};
#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_PAIREVALUATION_HPP_
