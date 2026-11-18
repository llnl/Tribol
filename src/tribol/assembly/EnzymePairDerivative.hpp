// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Fixed-size Enzyme differentiation of one pair function.
 */
#ifndef SRC_TRIBOL_ASSEMBLY_ENZYMEPAIRDERIVATIVE_HPP_
#define SRC_TRIBOL_ASSEMBLY_ENZYMEPAIRDERIVATIVE_HPP_

#include "tribol/assembly/PairDerivativeData.hpp"
#include "tribol/assembly/PairFunctionData.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/**
 * @brief Differentiate a fixed-size coordinate function on the host or device.
 *
 * For f(x), valuesAndJacobian() returns f and J_ij = df_i/dx_j. weightedJacobianAndHessian() returns the sums
 * sum_i w_i J_ij and sum_i w_i H_ijk, where H_ijk = d^2 f_i/(dx_j dx_k).
 * For scalar f, valueJacobianAndHessian() returns f, its coordinate Jacobian and Hessian, and quadrature state.
 *
 * PairFunction supplies Coords1 and Coords2 views and evaluate(coords1, coords2). Both coordinate arrays are active
 * inputs; Jacobian columns and Hessian rows/columns follow geometry1 coordinates, then geometry2 coordinates. A
 * function that needs geometry constructs it inside evaluate(), so derivatives include cached lengths, normals,
 * and other coordinate-dependent quantities. The result's values() returns MdArray<RealT, size_values>;
 * quadrature_metadata and has_active_points describe the calculation. Settings and output weights are held fixed.
 * Output weights multiply outputs of f independently of any quadrature weights already included in f. Scalar energies
 * and vector residuals use the same Jacobian and Hessian terminology, referring to derivatives of the input
 * function f. Results retain the domain value returned by PairFunction::evaluate() in pair_value and keep numerical
 * derivatives in a separate derivatives payload.
 *
 * valuesAndJacobian() obtains values, quadrature status, and activation from the forward AD evaluations, without a
 * separate evaluate() call. The caller must check the returned quadrature status before consuming the numerical
 * outputs. weightedJacobianAndHessian() differentiates the returned values without returning status or activation;
 * the caller must check quadrature separately. valueJacobianAndHessian() obtains the scalar value and state from the
 * same AD calls that compute its derivatives. The class allocates no dynamic memory, retains no state, and can run
 * inside a CUDA or HIP kernel.
 *
 * @tparam PairFunction Fixed-size pair-local scalar or vector pair_function.
 */
#ifdef TRIBOL_USE_ENZYME
template <DifferentiablePairFunctionType PairFunction>
class EnzymePairDerivative {
 public:
  using PairValue = PairFunctionValue<PairFunction>;                            ///< Domain pair-function result.
  static constexpr IndexT size_values = PairFunction::size_values;              ///< Number of pair-function outputs.
  static constexpr IndexT size_geom1_coords = PairFunction::size_geom1_coords;  ///< Active geometry1 coordinates.
  static constexpr IndexT size_geom2_coords = PairFunction::size_geom2_coords;  ///< Active geometry2 coordinates.
  static constexpr IndexT size_coords = PairFunction::size_coords;              ///< Total number of active coordinates.

  /**
   * @brief Calculate all function values and their coordinate Jacobian.
   *
   * Return f_i and J_ij = df_i/dx_j, with one row per value and one column per coordinate, together with quadrature
   * status and activation. Each forward AD call computes values and one Jacobian column; the returned values and
   * status come from these calls. No separate value evaluation, output weighting, or Hessian calculation is performed.
   * Failed or inactive evaluations return zero values and Jacobian after the first AD call, retaining quadrature
   * status and activation. A failed quadrature result must be reported before consuming the numerical outputs.
   * Successful active evaluations retain derivatives even when their values are zero. Nodal gap and area integrals
   * remain active whenever quadrature is nonempty, before nodal enforcement determines whether contact is compressive.
   *
   * For example, suppose f is a configured PairTractionIntegral of type PairFunction, and coords1 and
   * coords2 hold the two edges' coordinates. A pair kernel could use:
   * @code
   * const auto pair_values_struct =
   *     EnzymePairDerivative<PairFunction>::valuesAndJacobian( coords1.view(), coords2.view(), f );
   * // Record pair_values_struct.pair_value.quadrature_metadata before assembly.
   * const RealT force_entry = pair_values_struct.pair_value.values()[0];
   * const RealT force_jacobian_entry = pair_values_struct.derivatives.jacobian( 0, 0 );
   * @endcode
   * Here f is the pair force vector. The caller could store its values and Jacobian for later assembly. The same
   * operation on an PairNodalGapIntegral would return gap/area values and their coordinate Jacobian.
   *
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param pair_function Fixed pair-function configuration.
   * @return Function values, their value-by-coordinate Jacobian, quadrature status, and activation. Failed or
   * inactive numerical outputs are zero; status is not checked.
   */
  TRIBOL_HOST_DEVICE static PairValueJacobian<PairValue, size_coords> valuesAndJacobian(
      typename PairFunction::Coords1 coords1, typename PairFunction::Coords2 coords2,
      const PairFunction& pair_function )
  {
    PairValueJacobian<PairValue, size_coords> pair_values_struct{};
    for ( IndexT coordinate = 0; coordinate < size_coords; ++coordinate ) {
      MdArray<RealT, size_coords> coordinate_seed{};
      MdArray<RealT, size_values> jacobian_col{};
      coordinate_seed[coordinate] = 1.0;
      __enzyme_fwddiff<void>(
          (void*)evaluatePairFunctionEnzyme, TRIBOL_ENZYME_DUP, coords1.data(), coordinate_seed.data(),
          TRIBOL_ENZYME_DUP, coords2.data(), coordinate_seed.data() + size_geom1_coords, TRIBOL_ENZYME_CONST,
          (const void*)&pair_function, TRIBOL_ENZYME_DUP, pair_values_struct.pair_value.data(), jacobian_col.data(),
          TRIBOL_ENZYME_CONST, &pair_values_struct.pair_value.quadrature_metadata, TRIBOL_ENZYME_CONST,
          &pair_values_struct.pair_value.has_active_points );
      if ( pair_values_struct.pair_value.quadrature_metadata.status != QuadratureStatus::Success ||
           !pair_values_struct.pair_value.has_active_points ) {
        zeroValues( pair_values_struct.pair_value );
        pair_values_struct.derivatives = {};
        return pair_values_struct;
      }
      for ( IndexT value = 0; value < size_values; ++value ) {
        pair_values_struct.derivatives.jacobian( value, coordinate ) = jacobian_col[value];
      }
    }
    return pair_values_struct;
  }

  /**
   * @brief Calculate the weighted coordinate Jacobian and Hessian of the pair function.
   *
   * For fixed weights w_i, return sum_i w_i J_ij and sum_i w_i H_ijk. These are the Jacobian and Hessian of the scalar
   * sum_i w_i f_i(x). Function values, the full unweighted Jacobian, and individual output Hessians are not
   * returned. Changing weights with coordinates in the caller does not cause their derivatives to be included in this
   * call. All-zero weights return zero Jacobian and Hessian without evaluating or differentiating the pair_function.
   * Quadrature validity must still be checked separately.
   *
   * For example, suppose f is an PairNodalGapIntegral of type PairFunction for two-node edges, with
   * outputs (G0, G1, A0, A1), and its quadrature has been checked. A nodal LM calculation could use fixed multipliers:
   * @code
   * const MdArray<RealT, PairFunction::size_values> weights{ 2.0, 3.0, 0.0, 0.0 };
   * const auto pair_derivatives_struct = EnzymePairDerivative<PairFunction>::weightedJacobianAndHessian(
   *     coords1.view(), coords2.view(), f, weights.view() );
   * @endcode
   * pair_derivatives_struct.jacobian is 2.0*J_G0 + 3.0*J_G1 and pair_derivatives_struct.hessian is
   * 2.0*H_G0 + 3.0*H_G1. The Hessian could contribute to the coordinate Jacobian of the LM force. For a scalar
   * PairEnergyIntegral E, a single weight of 1.0 returns J_E and H_E, which could supply the force and its
   * coordinate Jacobian.
   *
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param pair_function Fixed pair-function configuration.
   * @param weights Fixed weights multiplying function outputs, distinct from quadrature weights.
   * @return Weighted Jacobian and Hessian; function values and quadrature status are not returned or checked.
   */
  TRIBOL_HOST_DEVICE static PairJacobianHessian<size_coords> weightedJacobianAndHessian(
      typename PairFunction::Coords1 coords1, typename PairFunction::Coords2 coords2, const PairFunction& pair_function,
      MdSpan<const RealT, size_values> weights )
  {
    PairJacobianHessian<size_coords> pair_derivatives_struct{};
    bool has_nonzero_weights = false;
    for ( IndexT value = 0; value < size_values; ++value ) {
      has_nonzero_weights = has_nonzero_weights || weights[value] != 0.0;
    }
    if ( !has_nonzero_weights ) {
      return pair_derivatives_struct;
    }
    for ( IndexT column = 0; column < size_coords; ++column ) {
      MdArray<RealT, size_coords> coordinate_seed{}, hessian_col{};
      coordinate_seed[column] = 1.0;
      __enzyme_fwddiff<void>( (void*)reverseWeightedJacobianEnzyme, TRIBOL_ENZYME_DUP, coords1.data(),
                              coordinate_seed.data(), TRIBOL_ENZYME_DUP, coords2.data(),
                              coordinate_seed.data() + size_geom1_coords, TRIBOL_ENZYME_CONST,
                              (const void*)&pair_function, TRIBOL_ENZYME_CONST, weights.data(), TRIBOL_ENZYME_DUP,
                              pair_derivatives_struct.jacobian.data(), hessian_col.data() );
      for ( IndexT row = 0; row < size_coords; ++row ) {
        pair_derivatives_struct.hessian( row, column ) = hessian_col[row];
      }
    }
    return pair_derivatives_struct;
  }

  /**
   * @brief Calculate a scalar pair function's value, coordinate Jacobian and Hessian, and quadrature state
   * together.
   *
   * Forward differentiation of a reverse AD calculation produces one Hessian column per coordinate. Those calls
   * also produce f(x), J_j = df/dx_j, quadrature status, and activation; no separate value or Jacobian calculation is
   * performed. Only coordinates are differentiated, with function settings held fixed.
   *
   * Failed or inactive evaluations return zero value, Jacobian, and Hessian, with the reported status and activation.
   * They stop after the first AD call. Successful active evaluations retain derivatives even when f(x) is zero.
   * The caller must check quadrature status before consuming the numerical outputs.
   *
   * For example, suppose f is a configured PairEnergyIntegral of type PairFunction. A pair kernel could
   * evaluate its energy and derivatives together:
   * @code
   * const auto pair_values = EnzymePairDerivative<PairFunction>::valueJacobianAndHessian( coords1.view(),
   * coords2.view(), f
   * );
   * // Record pair_values.pair_value.quadrature_metadata for checking before assembly.
   * const RealT energy = pair_values.pair_value.energy;
   * const RealT force_entry = pair_values.derivatives.jacobian[0];
   * const RealT force_jacobian_entry = pair_values.derivatives.hessian( 0, 0 );
   * @endcode
   * After checking quadrature, the caller could sum the energy and assemble the Jacobian as force and the Hessian
   * as the force's coordinate Jacobian.
   *
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @param pair_function Fixed scalar pair-function configuration, with size_values equal to one.
   * @return Scalar value, Jacobian, Hessian, quadrature status, and activation; failed or inactive outputs are zero.
   */
  TRIBOL_HOST_DEVICE static PairValueJacobianHessian<PairValue, size_coords> valueJacobianAndHessian(
      typename PairFunction::Coords1 coords1, typename PairFunction::Coords2 coords2,
      const PairFunction& pair_function )
    requires( size_values == 1 )
  {
    PairValueJacobianHessian<PairValue, size_coords> pair_values_struct{};
    for ( IndexT column = 0; column < size_coords; ++column ) {
      MdArray<RealT, size_coords> coordinate_seed{}, hessian_col{};
      coordinate_seed[column] = 1.0;
      __enzyme_fwddiff<void>(
          (void*)reverseScalarJacobianEnzyme, TRIBOL_ENZYME_DUP, coords1.data(), coordinate_seed.data(),
          TRIBOL_ENZYME_DUP, coords2.data(), coordinate_seed.data() + size_geom1_coords, TRIBOL_ENZYME_CONST,
          (const void*)&pair_function, TRIBOL_ENZYME_DUP, pair_values_struct.derivatives.jacobian.data(),
          hessian_col.data(), TRIBOL_ENZYME_CONST, pair_values_struct.pair_value.data(), TRIBOL_ENZYME_CONST,
          &pair_values_struct.pair_value.quadrature_metadata, TRIBOL_ENZYME_CONST,
          &pair_values_struct.pair_value.has_active_points );
      if ( pair_values_struct.pair_value.quadrature_metadata.status != QuadratureStatus::Success ||
           !pair_values_struct.pair_value.has_active_points ) {
        zeroValues( pair_values_struct.pair_value );
        pair_values_struct.derivatives = {};
        return pair_values_struct;
      }
      for ( IndexT row = 0; row < size_coords; ++row ) {
        pair_values_struct.derivatives.hessian( row, column ) = hessian_col[row];
      }
    }
    return pair_values_struct;
  }

 private:
  /** @brief Zero numerical values while retaining quadrature metadata and activation. @param pair_value Value to clear.
   */
  TRIBOL_HOST_DEVICE static void zeroValues( PairValue& pair_value )
  {
    for ( IndexT value = 0; value < size_values; ++value ) {
      pair_value.data()[value] = 0.0;
    }
  }

  /**
   * @brief Compute a scalar function's Jacobian by reverse AD while retaining its value and quadrature state.
   * @param coords1 Active geometry1 coordinate storage.
   * @param coords2 Active geometry2 coordinate storage.
   * @param pair_function_void Pointer to fixed PairFunction settings, passed without differentiation.
   * @param[out] scalar_jacobian Coordinate Jacobian, differentiated by forward AD to obtain the Hessian.
   * @param[out] value Scalar function value, retained without forward differentiation.
   * @param[out] quadrature_metadata Quadrature metadata, passed without differentiation.
   * @param[out] has_active_points Activation, passed without differentiation.
   */
  TRIBOL_HOST_DEVICE static void reverseScalarJacobianEnzyme( const RealT* coords1, const RealT* coords2,
                                                              const void* pair_function_void, RealT* scalar_jacobian,
                                                              RealT* value, QuadratureMetadata* quadrature_metadata,
                                                              bool* has_active_points )
  {
    MdArray<RealT, size_coords> jacobian_values{};
    RealT value_seed = 1.0;
    __enzyme_autodiff<void>( (void*)evaluatePairFunctionEnzyme, TRIBOL_ENZYME_DUP, coords1, jacobian_values.data(),
                             TRIBOL_ENZYME_DUP, coords2, jacobian_values.data() + size_geom1_coords,
                             TRIBOL_ENZYME_CONST, pair_function_void, TRIBOL_ENZYME_DUP, value, &value_seed,
                             TRIBOL_ENZYME_CONST, quadrature_metadata, TRIBOL_ENZYME_CONST, has_active_points );
    for ( IndexT coordinate = 0; coordinate < size_coords; ++coordinate ) {
      scalar_jacobian[coordinate] = jacobian_values[coordinate];
    }
  }

  /**
   * @brief Write the pair evaluation so forward AD returns values and status along with numerical derivatives.
   *
   * Keep status and activation in separate inactive outputs so Enzyme differentiates only the numerical values.
   * @param coords1 Active geometry1 coordinate storage.
   * @param coords2 Active geometry2 coordinate storage.
   * @param pair_function_void Pointer to the fixed PairFunction configuration, passed without differentiation.
   * @param[out] values Function values, with size_values entries written and differentiated.
   * @param[out] quadrature_metadata Quadrature metadata, passed without differentiation.
   * @param[out] has_active_points Activation, passed without differentiation.
   */
  TRIBOL_HOST_DEVICE static void evaluatePairFunctionEnzyme( const RealT* coords1, const RealT* coords2,
                                                             const void* pair_function_void, RealT* values,
                                                             QuadratureMetadata* quadrature_metadata,
                                                             bool* has_active_points )
  {
    const auto& pair_function = *static_cast<const PairFunction*>( pair_function_void );
    const auto pair_values_struct =
        pair_function.evaluate( { coords1, size_geom1_coords }, { coords2, size_geom2_coords } );
    const auto function_values = pair_values_struct.values();
    for ( IndexT value = 0; value < size_values; ++value ) {
      values[value] = function_values[value];
    }
    // Copy fields individually to keep structure padding out of Enzyme's type analysis.
    quadrature_metadata->status = pair_values_struct.quadrature_metadata.status;
    quadrature_metadata->num_required_points = pair_values_struct.quadrature_metadata.num_required_points;
    *has_active_points = pair_values_struct.has_active_points;
  }

  /**
   * @brief Compute sum_i weights_i * f_i(x) for Enzyme's reverse differentiation.
   * @param coords1 Active geometry1 coordinate storage.
   * @param coords2 Active geometry2 coordinate storage.
   * @param pair_function_void Pointer to the fixed PairFunction configuration, passed without differentiation.
   * @param weights Fixed output weights, with size_values entries.
   * @param[out] weighted_value Weighted scalar value.
   */
  TRIBOL_HOST_DEVICE static void evaluateWeightedValueEnzyme( const RealT* coords1, const RealT* coords2,
                                                              const void* pair_function_void, const RealT* weights,
                                                              RealT* weighted_value )
  {
    const auto& pair_function = *static_cast<const PairFunction*>( pair_function_void );
    const auto pair_values_struct =
        pair_function.evaluate( { coords1, size_geom1_coords }, { coords2, size_geom2_coords } );
    const auto function_values = pair_values_struct.values();
    *weighted_value = 0.0;
    for ( IndexT value = 0; value < size_values; ++value ) {
      *weighted_value += weights[value] * function_values[value];
    }
  }

  /**
   * @brief Compute the weighted Jacobian by reverse differentiation so Enzyme can differentiate it for the Hessian.
   * @param coords1 Active geometry1 coordinate storage.
   * @param coords2 Active geometry2 coordinate storage.
   * @param pair_function_void Pointer to the fixed PairFunction configuration, passed without differentiation.
   * @param weights Fixed output weights, with size_values entries.
   * @param[out] weighted_jacobian Weighted Jacobian sum_i weights_i * df_i/dx_j, with one entry per coordinate.
   */
  TRIBOL_HOST_DEVICE static void reverseWeightedJacobianEnzyme( const RealT* coords1, const RealT* coords2,
                                                                const void* pair_function_void, const RealT* weights,
                                                                RealT* weighted_jacobian )
  {
    MdArray<RealT, size_coords> jacobian_values{};
    RealT weighted_value = 0.0;
    RealT weighted_value_seed = 1.0;
    __enzyme_autodiff<void>( (void*)evaluateWeightedValueEnzyme, TRIBOL_ENZYME_DUP, coords1, jacobian_values.data(),
                             TRIBOL_ENZYME_DUP, coords2, jacobian_values.data() + size_geom1_coords,
                             TRIBOL_ENZYME_CONST, pair_function_void, TRIBOL_ENZYME_CONST, weights, TRIBOL_ENZYME_DUP,
                             &weighted_value, &weighted_value_seed );
    for ( IndexT coordinate = 0; coordinate < size_coords; ++coordinate ) {
      weighted_jacobian[coordinate] = jacobian_values[coordinate];
    }
  }
};
#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
#endif  // SRC_TRIBOL_ASSEMBLY_ENZYMEPAIRDERIVATIVE_HPP_
