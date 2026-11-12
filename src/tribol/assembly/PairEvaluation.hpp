// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Pair-function values, Jacobians, and Hessians, with fixed-size Enzyme differentiation.
 */
#ifndef SRC_TRIBOL_ASSEMBLY_PAIREVALUATION_HPP_
#define SRC_TRIBOL_ASSEMBLY_PAIREVALUATION_HPP_

#include <concepts>
#include <type_traits>
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/** @brief Pair energy, quadrature status, and compressive-point activation. */
struct PairEnergyValue {
  RealT energy = 0.0;                    ///< Integrated energy, zero for empty or inactive rules.
  bool has_active_points = false;        ///< Whether a compressive point exists, including zero gap.
  QuadratureResult quadrature_result{};  ///< Failed quadrature output must not be consumed.

  /** @brief Return the scalar energy as a one-entry function value vector. @return Integrated pair energy. */
  TRIBOL_HOST_DEVICE MdArray<RealT, 1> values() const { return MdArray<RealT, 1>{ energy }; }
};
/** @brief Pair residual, quadrature status, and activation. @tparam DofCt Pair coordinate count. */
template <IndexT DofCt>
struct PairResidualValue {
  MdArray<RealT, DofCt> residual{};      ///< Residual in element/node/component order.
  QuadratureResult quadrature_result{};  ///< Failed quadrature output must not be consumed.
  bool has_active_points = false;        ///< Whether a compressive point exists, including zero gap.

  /** @brief Return the residual as the function value vector. @return Residual in element/node/component order. */
  TRIBOL_HOST_DEVICE MdArray<RealT, DofCt> values() const { return residual; }
};

/**
 * @brief Own one pair function's values f_i(x), Jacobian J_ij = df_i/dx_j, quadrature status, and activation.
 * @tparam ValueCt Number of scalar outputs.
 * @tparam DofCt Number of active coordinate entries.
 */
template <IndexT ValueCt, IndexT DofCt>
struct PairValueJacobian {
  MdArray<RealT, ValueCt> values{};           ///< Function values f_i in function-defined order.
  MdArray<RealT, ValueCt, DofCt> jacobian{};  ///< J_ij, with value rows and coordinate columns.
  QuadratureResult quadrature_result{};       ///< Quadrature status from the differentiated evaluation.
  bool has_active_points = false;             ///< Activation from the differentiated evaluation.
};

/**
 * @brief Own the weighted Jacobian and Hessian of one pair function.
 *
 * For fixed output weights w_i, these are sum_i w_i J_ij and sum_i w_i H_ijk, where J_ij = df_i/dx_j and
 * H_ijk = d^2 f_i/(dx_j dx_k). Function values and the full unweighted Jacobian and Hessians are not stored.
 *
 * @tparam DofCt Number of active coordinate entries.
 */
template <IndexT DofCt>
struct PairWeightedJacobianHessian {
  MdArray<RealT, DofCt> jacobian{};        ///< Weighted Jacobian sum_i w_i J_ij, one entry per coordinate.
  MdArray<RealT, DofCt, DofCt> hessian{};  ///< Weighted Hessian sum_i w_i H_ijk, in coordinate row/column order.
};

/**
 * @brief Own a scalar pair function's value, coordinate Jacobian and Hessian, quadrature status, and activation.
 * @tparam DofCt Number of active coordinate entries.
 */
template <IndexT DofCt>
struct PairValueJacobianHessian {
  RealT value = 0.0;                       ///< Scalar function value f(x).
  MdArray<RealT, DofCt> jacobian{};        ///< Coordinate Jacobian J_j = df/dx_j.
  MdArray<RealT, DofCt, DofCt> hessian{};  ///< Coordinate Hessian H_jk = d^2 f/(dx_j dx_k).
  QuadratureResult quadrature_result{};    ///< Quadrature status from the differentiated evaluation.
  bool has_active_points = false;          ///< Activation from the differentiated evaluation.
};

/**
 * @brief Require positive fixed sizes and a const coordinate evaluation exposing values, quadrature status, and
 * activation.
 *
 * The evaluation result's const values() method returns an owning fixed-size array. Its quadrature_result and
 * has_active_points members have types QuadratureResult and bool. These requirements check expressions and types;
 * numerical correctness, host/device execution, and Enzyme compatibility must be verified separately.
 * @tparam Function Type evaluating a fixed-size coordinate vector.
 */
template <typename Function>
concept CoordinateFunctionType =
    requires {
      typename std::integral_constant<IndexT, Function::value_ct>;
      typename std::integral_constant<IndexT, Function::coord_ct>;
      requires( Function::value_ct > 0 && Function::coord_ct > 0 );
    } && requires( const Function& function, MdSpan<const RealT, Function::coord_ct> coordinates,
                   const std::remove_cvref_t<decltype( function.evaluate( coordinates ) )>& result ) {
      { result.values() } -> std::same_as<MdArray<RealT, Function::value_ct>>;
      requires std::same_as<std::remove_cvref_t<decltype( result.quadrature_result )>, QuadratureResult>;
      requires std::same_as<std::remove_cvref_t<decltype( result.has_active_points )>, bool>;
    };

/**
 * @brief Require two geometry types and a const pair evaluation with values, quadrature status, and activation.
 *
 * Each geometry is constructed from its Coords view. Coordinate counts define Jacobian columns in geometry1 order,
 * followed by geometry2 order, and must agree with those views. These requirements check types and expressions;
 * numerical correctness, host/device execution, and Enzyme compatibility must be verified separately.
 * @tparam Function Type evaluating a pair of current element geometries.
 */
template <typename Function>
concept PairFunctionType =
    requires {
      typename Function::FirstGeometry;
      typename Function::SecondGeometry;
      typename std::integral_constant<IndexT, Function::value_ct>;
      typename std::integral_constant<IndexT, Function::geom1_coord_ct>;
      typename std::integral_constant<IndexT, Function::geom2_coord_ct>;
      typename std::integral_constant<IndexT, Function::coord_ct>;
      requires( Function::value_ct > 0 && Function::geom1_coord_ct > 0 && Function::geom2_coord_ct > 0 );
      requires( Function::FirstGeometry::Coords::size() == Function::geom1_coord_ct );
      requires( Function::SecondGeometry::Coords::size() == Function::geom2_coord_ct );
      requires( Function::coord_ct == Function::geom1_coord_ct + Function::geom2_coord_ct );
    } &&
    requires( const Function& function, typename Function::FirstGeometry::Coords coordinates1,
              typename Function::SecondGeometry::Coords coordinates2, const typename Function::FirstGeometry& geometry1,
              const typename Function::SecondGeometry& geometry2,
              const std::remove_cvref_t<decltype( function.evaluate( geometry1, geometry2 ) )>& result ) {
      { typename Function::FirstGeometry( coordinates1 ) } -> std::same_as<typename Function::FirstGeometry>;
      { typename Function::SecondGeometry( coordinates2 ) } -> std::same_as<typename Function::SecondGeometry>;
      { result.values() } -> std::same_as<MdArray<RealT, Function::value_ct>>;
      requires std::same_as<std::remove_cvref_t<decltype( result.quadrature_result )>, QuadratureResult>;
      requires std::same_as<std::remove_cvref_t<decltype( result.has_active_points )>, bool>;
    };

/**
 * @brief Adapt a geometry pair function to coordinate differentiation.
 *
 * Geometry1 coordinates precede geometry2 coordinates in element/node/component order. Geometry objects are
 * constructed inside evaluate(), so differentiation includes their cached edge vectors, lengths, and other derived
 * quantities. Passing geometry constructed before differentiation as fixed settings would omit those dependencies.
 * @tparam PairFunction Geometry pair function with fixed coordinate counts.
 */
template <PairFunctionType PairFunction>
class PairCoordinateFunction {
 public:
  static constexpr IndexT value_ct = PairFunction::value_ct;  ///< Number of function outputs.
  static constexpr IndexT coord_ct = PairFunction::coord_ct;  ///< Combined coordinate count.

  /** @brief Copy the fixed physics configuration. @param pair_function Geometry pair function copied into this object.
   */
  TRIBOL_HOST_DEVICE explicit PairCoordinateFunction( PairFunction pair_function ) : pair_function_( pair_function ) {}

  /**
   * @brief Construct both current geometries and evaluate the pair function.
   * @param coordinates Geometry1 coordinates followed by geometry2 coordinates.
   * @return Function values, quadrature status, and activation, following the pair function's evaluation contract.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto evaluate( MdSpan<const RealT, coord_ct> coordinates ) const
  {
    const typename PairFunction::FirstGeometry geometry1( { coordinates.data(), PairFunction::geom1_coord_ct } );
    const typename PairFunction::SecondGeometry geometry2(
        { coordinates.data() + PairFunction::geom1_coord_ct, PairFunction::geom2_coord_ct } );
    return pair_function_.evaluate( geometry1, geometry2 );
  }

 private:
  PairFunction pair_function_;  ///< Fixed pair-function configuration, independent of any geometry instance.
};

/**
 * @brief Differentiate a fixed-size coordinate function on the host or device.
 *
 * For f(x), valuesAndJacobian() returns f and J_ij = df_i/dx_j. weightedJacobianAndHessian() returns the sums
 * sum_i w_i J_ij and sum_i w_i H_ijk, where H_ijk = d^2 f_i/(dx_j dx_k).
 * For scalar f, valueJacobianAndHessian() returns f, its coordinate Jacobian and Hessian, and quadrature state.
 *
 * Function supplies positive static `value_ct` and `coord_ct` members and a host/device-callable evaluate()
 * taking MdSpan<const RealT, coord_ct>. Wrap a geometry pair function in PairCoordinateFunction so geometry
 * construction remains inside the differentiated calculation. The result's values() method returns an
 * MdArray<RealT, value_ct>; quadrature_result and has_active_points describe the calculation. All operations use
 * this evaluate() interface. Only coordinates are differentiated; settings and output weights are held fixed. Output
 * weights multiply components of f independently of any quadrature weights already included in f. Scalar energies and
 * vector residuals use the same Jacobian and Hessian terminology, always referring to derivatives of the input function
 * f.
 *
 * valuesAndJacobian() obtains values, quadrature status, and activation from the forward AD evaluations, without a
 * separate evaluate() call. The caller must check the returned quadrature status before consuming the numerical
 * outputs. weightedJacobianAndHessian() differentiates the returned values without returning status or activation;
 * the caller must check quadrature separately. valueJacobianAndHessian() obtains the scalar value and state from the
 * same AD calls that compute its derivatives. The class allocates no dynamic memory, retains no state, and can run
 * inside a CUDA or HIP kernel.
 *
 * @tparam Function Fixed-size pair-local scalar or vector function.
 */
#ifdef TRIBOL_USE_ENZYME
template <CoordinateFunctionType Function>
class EnzymePairDerivative {
 public:
  static constexpr IndexT value_ct = Function::value_ct;  ///< Number of pair-function outputs.
  static constexpr IndexT coord_ct = Function::coord_ct;  ///< Number of active coordinates.

  /**
   * @brief Calculate all function values and their coordinate Jacobian.
   *
   * Return f_i and J_ij = df_i/dx_j, with one row per value and one column per coordinate, together with quadrature
   * status and activation. Each forward AD call computes values and one Jacobian column; the returned values and
   * status come from these calls. No separate value evaluation, output weighting, or Hessian calculation is performed.
   * A failed quadrature result must be reported before consuming the numerical outputs. Empty and inactive pairs
   * follow Function's evaluate() behavior; a zero value alone does not imply inactivity or a zero Jacobian.
   *
   * For example, suppose f is a PairCoordinateFunction wrapping a configured CommonPlanePairFunction, Function is its
   * type, and coordinates holds geometry1 coordinates followed by geometry2 coordinates. A pair kernel could use:
   * @code
   * const auto result = EnzymePairDerivative<Function>::valuesAndJacobian( coordinates.view(), f );
   * // Record result.quadrature_result and result.has_active_points for checking before assembly.
   * const RealT force_entry = result.values[0];
   * const RealT force_jacobian_entry = result.jacobian( 0, 0 );
   * @endcode
   * Here f is the pair force vector. The caller could store its values and Jacobian for later assembly. The same
   * operation on an EnergyMortarNodalEnergy would return gap/area values and their coordinate Jacobian.
   *
   * @param coordinates Current pair coordinates.
   * @param function Fixed pair-function configuration.
   * @return Function values, their value-by-coordinate Jacobian, quadrature status, and activation. Status is not
   * checked.
   */
  TRIBOL_HOST_DEVICE static PairValueJacobian<value_ct, coord_ct> valuesAndJacobian(
      MdSpan<const RealT, coord_ct> coordinates, const Function& function )
  {
    PairValueJacobian<value_ct, coord_ct> result{};
    for ( IndexT coordinate = 0; coordinate < coord_ct; ++coordinate ) {
      MdArray<RealT, coord_ct> coordinate_seed{};
      MdArray<RealT, value_ct> derivatives{};
      coordinate_seed[coordinate] = 1.0;
      __enzyme_fwddiff<void>( (void*)evaluateKernel, TRIBOL_ENZYME_DUP, coordinates.data(), coordinate_seed.data(),
                              TRIBOL_ENZYME_CONST, (const void*)&function, TRIBOL_ENZYME_DUP, result.values.data(),
                              derivatives.data(), TRIBOL_ENZYME_CONST, &result.quadrature_result, TRIBOL_ENZYME_CONST,
                              &result.has_active_points );
      for ( IndexT value = 0; value < value_ct; ++value ) {
        result.jacobian( value, coordinate ) = derivatives[value];
      }
    }
    return result;
  }

  /**
   * @brief Calculate the weighted coordinate Jacobian and Hessian of the pair function.
   *
   * For fixed weights w_i, return sum_i w_i J_ij and sum_i w_i H_ijk. These are the Jacobian and Hessian of the scalar
   * sum_i w_i f_i(x). Function values, the full unweighted Jacobian, and individual output Hessians are not returned.
   * Changing weights with coordinates in the caller does not cause their derivatives to be included in this call.
   *
   * For example, suppose f is a PairCoordinateFunction wrapping EnergyMortarNodalEnergy for two-node edges, Function is
   * its type, and the outputs are
   * (G0, G1, A0, A1), and its quadrature has been checked. A nodal LM calculation could use fixed multipliers:
   * @code
   * const MdArray<RealT, Function::value_ct> weights{ 2.0, 3.0, 0.0, 0.0 };
   * const auto result = EnzymePairDerivative<Function>::weightedJacobianAndHessian(
   *     coordinates.view(), f, weights.view() );
   * @endcode
   * result.jacobian is 2.0*J_G0 + 3.0*J_G1 and result.hessian is 2.0*H_G0 + 3.0*H_G1. The Hessian could contribute
   * to the coordinate Jacobian of the LM force. For a coordinate-adapted scalar EnergyMortarEnergy E, a single weight
   * of 1.0 returns J_E and H_E, which could supply the force and its coordinate Jacobian.
   *
   * @param coordinates Current pair coordinates.
   * @param function Fixed pair-function configuration.
   * @param weights Fixed weights multiplying function outputs, distinct from quadrature weights.
   * @return Weighted Jacobian and Hessian; function values and quadrature status are not returned or checked.
   */
  TRIBOL_HOST_DEVICE static PairWeightedJacobianHessian<coord_ct> weightedJacobianAndHessian(
      MdSpan<const RealT, coord_ct> coordinates, const Function& function, MdSpan<const RealT, value_ct> weights )
  {
    PairWeightedJacobianHessian<coord_ct> result{};
    for ( IndexT column = 0; column < coord_ct; ++column ) {
      MdArray<RealT, coord_ct> coordinate_seed{}, hessian_column{};
      coordinate_seed[column] = 1.0;
      __enzyme_fwddiff<void>( (void*)weightedJacobianKernel, TRIBOL_ENZYME_DUP, coordinates.data(),
                              coordinate_seed.data(), TRIBOL_ENZYME_CONST, (const void*)&function, TRIBOL_ENZYME_CONST,
                              weights.data(), TRIBOL_ENZYME_DUP, result.jacobian.data(), hessian_column.data() );
      for ( IndexT row = 0; row < coord_ct; ++row ) {
        result.hessian( row, column ) = hessian_column[row];
      }
    }
    return result;
  }

  /**
   * @brief Calculate a scalar pair function's value, coordinate Jacobian and Hessian, and quadrature state together.
   *
   * Forward differentiation of a reverse AD calculation produces one Hessian column per coordinate. Those calls
   * also produce f(x), J_j = df/dx_j, quadrature status, and activation; no separate value or Jacobian calculation is
   * performed. Only coordinates are differentiated, with function settings held fixed.
   *
   * Failed or inactive evaluations return zero value, Jacobian, and Hessian, with the reported status and activation.
   * They stop after the first AD call. Successful active evaluations retain derivatives even when f(x) is zero.
   * The caller must check quadrature status before consuming the numerical outputs.
   *
   * For example, suppose f is a PairCoordinateFunction wrapping EnergyMortarEnergy and Function is its type. A pair
   * kernel could evaluate its energy and derivatives together:
   * @code
   * const auto pair_values = EnzymePairDerivative<Function>::valueJacobianAndHessian( coordinates.view(), f );
   * // Record pair_values.quadrature_result for checking before assembly.
   * const RealT energy = pair_values.value;
   * const RealT force_entry = pair_values.jacobian[0];
   * const RealT force_jacobian_entry = pair_values.hessian( 0, 0 );
   * @endcode
   * After checking quadrature, the caller could sum the energy and assemble the Jacobian as force and the Hessian
   * as the force's coordinate Jacobian.
   *
   * @param coordinates Current pair coordinates.
   * @param function Fixed scalar pair-function configuration, with value_ct equal to one.
   * @return Scalar value, Jacobian, Hessian, quadrature status, and activation; failed or inactive outputs are zero.
   */
  TRIBOL_HOST_DEVICE static PairValueJacobianHessian<coord_ct> valueJacobianAndHessian(
      MdSpan<const RealT, coord_ct> coordinates, const Function& function )
    requires( value_ct == 1 )
  {
    PairValueJacobianHessian<coord_ct> result{};
    for ( IndexT column = 0; column < coord_ct; ++column ) {
      MdArray<RealT, coord_ct> coordinate_seed{}, hessian_column{};
      coordinate_seed[column] = 1.0;
      __enzyme_fwddiff<void>( (void*)scalarJacobianKernel, TRIBOL_ENZYME_DUP, coordinates.data(),
                              coordinate_seed.data(), TRIBOL_ENZYME_CONST, (const void*)&function, TRIBOL_ENZYME_DUP,
                              result.jacobian.data(), hessian_column.data(), TRIBOL_ENZYME_CONST, &result.value,
                              TRIBOL_ENZYME_CONST, &result.quadrature_result, TRIBOL_ENZYME_CONST,
                              &result.has_active_points );
      if ( result.quadrature_result.status != QuadratureStatus::Success || !result.has_active_points ) {
        result.value = 0.0;
        result.jacobian = {};
        return result;
      }
      for ( IndexT row = 0; row < coord_ct; ++row ) {
        result.hessian( row, column ) = hessian_column[row];
      }
    }
    return result;
  }

 private:
  /**
   * @brief Compute a scalar function's Jacobian by reverse AD while retaining its value and quadrature state.
   * @param coordinates Active coordinate storage.
   * @param function_void Pointer to fixed Function settings, passed without differentiation.
   * @param[out] jacobian Coordinate Jacobian, differentiated by forward AD to obtain the Hessian.
   * @param[out] value Scalar function value, retained without forward differentiation.
   * @param[out] quadrature_result Quadrature status, passed without differentiation.
   * @param[out] has_active_points Activation, passed without differentiation.
   */
  TRIBOL_HOST_DEVICE static void scalarJacobianKernel( const RealT* coordinates, const void* function_void,
                                                       RealT* jacobian, RealT* value,
                                                       QuadratureResult* quadrature_result, bool* has_active_points )
  {
    MdArray<RealT, coord_ct> derivative{};
    RealT value_seed = 1.0;
    __enzyme_autodiff<void>( (void*)evaluateKernel, TRIBOL_ENZYME_DUP, coordinates, derivative.data(),
                             TRIBOL_ENZYME_CONST, function_void, TRIBOL_ENZYME_DUP, value, &value_seed,
                             TRIBOL_ENZYME_CONST, quadrature_result, TRIBOL_ENZYME_CONST, has_active_points );
    for ( IndexT coordinate = 0; coordinate < coord_ct; ++coordinate ) {
      jacobian[coordinate] = derivative[coordinate];
    }
  }

  /**
   * @brief Write the pair evaluation so forward AD returns values and status along with numerical derivatives.
   *
   * Keep status and activation in separate inactive outputs so Enzyme differentiates only the numerical values.
   * @param coordinates Active coordinate storage.
   * @param function_void Pointer to the fixed Function configuration, passed without differentiation.
   * @param[out] values Function values, with value_ct entries written and differentiated.
   * @param[out] quadrature_result Quadrature status, passed without differentiation.
   * @param[out] has_active_points Activation, passed without differentiation.
   */
  TRIBOL_HOST_DEVICE static void evaluateKernel( const RealT* coordinates, const void* function_void, RealT* values,
                                                 QuadratureResult* quadrature_result, bool* has_active_points )
  {
    const auto& function = *static_cast<const Function*>( function_void );
    const auto result = function.evaluate( { coordinates, coord_ct } );
    const auto function_values = result.values();
    for ( IndexT value = 0; value < value_ct; ++value ) {
      values[value] = function_values[value];
    }
    // Copy fields individually to keep structure padding out of Enzyme's type analysis.
    quadrature_result->status = result.quadrature_result.status;
    quadrature_result->required_point_count = result.quadrature_result.required_point_count;
    *has_active_points = result.has_active_points;
  }

  /**
   * @brief Compute sum_i weights_i * f_i(x) for Enzyme's reverse differentiation.
   * @param coordinates Active coordinate storage.
   * @param function_void Pointer to the fixed Function configuration, passed without differentiation.
   * @param weights Fixed output weights, with value_ct entries.
   * @param[out] scalar Weighted scalar value.
   */
  TRIBOL_HOST_DEVICE static void weightedValueKernel( const RealT* coordinates, const void* function_void,
                                                      const RealT* weights, RealT* scalar )
  {
    const auto& function = *static_cast<const Function*>( function_void );
    const auto values = function.evaluate( { coordinates, coord_ct } ).values();
    *scalar = 0.0;
    for ( IndexT value = 0; value < value_ct; ++value ) {
      *scalar += weights[value] * values[value];
    }
  }

  /**
   * @brief Compute the weighted Jacobian by reverse differentiation so Enzyme can differentiate it for the Hessian.
   * @param coordinates Active coordinate storage.
   * @param function_void Pointer to the fixed Function configuration, passed without differentiation.
   * @param weights Fixed output weights, with value_ct entries.
   * @param[out] jacobian Weighted Jacobian sum_i weights_i * df_i/dx_j, with one entry per coordinate.
   */
  TRIBOL_HOST_DEVICE static void weightedJacobianKernel( const RealT* coordinates, const void* function_void,
                                                         const RealT* weights, RealT* jacobian )
  {
    MdArray<RealT, coord_ct> derivative{};
    RealT scalar = 0.0;
    RealT scalar_seed = 1.0;
    __enzyme_autodiff<void>( (void*)weightedValueKernel, TRIBOL_ENZYME_DUP, coordinates, derivative.data(),
                             TRIBOL_ENZYME_CONST, function_void, TRIBOL_ENZYME_CONST, weights, TRIBOL_ENZYME_DUP,
                             &scalar, &scalar_seed );
    for ( IndexT coordinate = 0; coordinate < coord_ct; ++coordinate ) {
      jacobian[coordinate] = derivative[coordinate];
    }
  }
};
#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
#endif  // SRC_TRIBOL_ASSEMBLY_PAIREVALUATION_HPP_
