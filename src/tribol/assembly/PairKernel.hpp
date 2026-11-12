// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Evaluate pair functions f and return buffers of their values, coordinate Jacobians, and Hessians.
 *
 * Pair functions accept two geometry objects. Value-only evaluation constructs them from separate mesh coordinate
 * arrays. Derivative kernels use PairCoordinateFunction to construct both geometries inside AD; the combined
 * coordinate array defines Jacobian columns in geometry1 node/component order, followed by geometry2 order.
 */
#ifndef SRC_TRIBOL_ASSEMBLY_PAIRKERNEL_HPP_
#define SRC_TRIBOL_ASSEMBLY_PAIRKERNEL_HPP_

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/QuadratureCommon.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/assembly/PairContribution.hpp"
#include "tribol/assembly/PairEvaluation.hpp"

namespace tribol {
/**
 * @brief Evaluate each pair's values and quadrature state without calculating derivatives.
 *
 * Each pair is evaluated once. This kernel requires no derivative provider or Enzyme support. Failed, empty, and
 * inactive values follow the pair function's evaluate() behavior; callers must check quadrature before assembly.
 * For example, Common Plane explicit contact could obtain a PairValueBuffer from evaluatePairValues(), check its
 * quadratureResults(), and assemble its active residual values as nodal forces.
 * @tparam PairFunction Function evaluating two current element geometries.
 * @tparam Exec Concrete execution mode for pair evaluation.
 * @param pairs Supplied element pairs in execution-accessible memory.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param pair_function Fixed pair configuration.
 * @return Owned values, quadrature results, and activation flags in pairs.getAllocatorID(), with one slot per pair.
 * Empty input returns zero slots. Evaluation finishes before returning; the arrays remain valid after pairs and
 * the input meshes are destroyed. Quadrature failures are recorded for the caller to check.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
PairValueBuffer<PairFunction::value_ct> evaluatePairValues( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                            MeshData::Viewer mesh2, PairFunction pair_function )
{
  PairValueBuffer<PairFunction::value_ct> result( pairs.size(), pairs.getAllocatorID() );
  const auto contributions_view = result.view();
  const auto pairs_view = pairs.view();
  forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs_view[pair_index];
    MdArray<RealT, PairFunction::geom1_coord_ct> coordinates1{};
    MdArray<RealT, PairFunction::geom2_coord_ct> coordinates2{};
    mesh1.getFaceCoords( pair.element_id1_, coordinates1.data() );
    mesh2.getFaceCoords( pair.element_id2_, coordinates2.data() );
    const typename PairFunction::FirstGeometry geometry1( { coordinates1.data(), coordinates1.size() } );
    const typename PairFunction::SecondGeometry geometry2( { coordinates2.data(), coordinates2.size() } );
    const auto pair_values = pair_function.evaluate( geometry1, geometry2 );
    const auto values = pair_values.values();
    contributions_view.quadrature[pair_index] = pair_values.quadrature_result;
    contributions_view.has_active_points[pair_index] = pair_values.has_active_points;
    for ( IndexT value = 0; value < PairFunction::value_ct; ++value ) {
      contributions_view.setValue( pair_index, value, values[value] );
    }
  } );
  return result;
}

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Evaluate each pair's values and coordinate Jacobian and return their buffers.
 *
 * For the input function f(x), write every value f_i and every Jacobian entry J_ij = df_i/dx_j. This applies equally
 * to scalar and vector functions. The values already include any quadrature weights used by f. The caller must check
 * the recorded quadrature results before using these outputs.
 *
 * For example, suppose pair_function is a configured CommonPlanePairFunction of type PairFunction, pairs is an accepted
 * pair list, and mesh1 and mesh2 are current mesh views. A caller could evaluate its local nodal forces and their
 * Jacobians:
 * @code
 * auto results = evaluatePairValuesAndJacobians<Exec>( pairs, mesh1, mesh2, pair_function );
 * checkPairQuadratureResults<Exec>( results.quadratureResults() );
 * @endcode
 * After the check succeeds, the caller could assemble the values into a force vector and the Jacobians into a matrix.
 * This kernel stores each pair's contribution separately, including when pairs share nodes.
 *
 * @tparam PairFunction Pair function whose evaluate() result supplies the fixed-size vector f(x) and quadrature state.
 * @tparam Exec Concrete execution mode for pair evaluation.
 * @param pairs Current pair list in execution-accessible memory.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param pair_function Fixed geometry-pair function configuration.
 * @return Owned values, Jacobians, quadrature results, and activation flags in pairs.getAllocatorID(). Empty input
 * returns zero slots. Evaluation finishes before returning; the arrays remain valid after pairs and the input meshes
 * are destroyed. Quadrature failures are recorded for the caller to check.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
PairValueJacobianBuffer<PairFunction::value_ct, PairFunction::coord_ct> evaluatePairValuesAndJacobians(
    const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1, MeshData::Viewer mesh2, PairFunction pair_function )
{
  PairValueJacobianBuffer<PairFunction::value_ct, PairFunction::coord_ct> result( pairs.size(),
                                                                                  pairs.getAllocatorID() );
  const auto contributions_view = result.view();
  const auto pairs_view = pairs.view();
  const PairCoordinateFunction coordinate_function{ pair_function };
  constexpr IndexT geom1_coord_ct = PairFunction::geom1_coord_ct;
  constexpr IndexT coord_ct = PairFunction::coord_ct;
  forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs_view[pair_index];
    MdArray<RealT, coord_ct> coordinates{};
    mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
    mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + geom1_coord_ct );
    const auto pair_values = EnzymePairDerivative<PairCoordinateFunction<PairFunction>>::valuesAndJacobian(
        coordinates.view(), coordinate_function );
    contributions_view.quadrature[pair_index] = pair_values.quadrature_result;
    contributions_view.has_active_points[pair_index] = pair_values.has_active_points;
    for ( IndexT value = 0; value < PairFunction::value_ct; ++value ) {
      contributions_view.setValue( pair_index, value, pair_values.values[value] );
      for ( IndexT coordinate = 0; coordinate < coord_ct; ++coordinate ) {
        contributions_view.setJacobian( pair_index, value, coordinate, pair_values.jacobian( value, coordinate ) );
      }
    }
  } );
  return result;
}

/**
 * @brief Evaluate weighted Jacobians and Hessians of each pair function f and return their buffers.
 *
 * Let J_ij = df_i/dx_j and H_ijk = d^2 f_i/(dx_j dx_k). For fixed output weights w_i, write sum_i w_i J_ij and
 * sum_i w_i H_ijk. The weighted Jacobian has one entry per coordinate, and the weighted Hessian is a square coordinate
 * matrix. Function values and the full unweighted Jacobian are not returned. The weights multiply components of f;
 * they are distinct from quadrature weights, which are already included in f.
 *
 * For example, suppose pair_function is an EnergyMortarNodalEnergy of type PairFunction for two-node edges and its
 * quadrature has already been checked. Its four outputs are weighted gaps G0, G1 followed by areas A0, A1. A
 * hypothetical nodal LM calculation could use multipliers 2.0 and 3.0, and zero area weights, to differentiate 2.0*G0
 * + 3.0*G1:
 * @code
 * auto derivatives = evaluatePairWeightedJacobiansAndHessians<Exec>(
 *     pairs, mesh1, mesh2, pair_function,
 *     [] TRIBOL_HOST_DEVICE( IndexT ) {
 *       return MdArray<RealT, PairFunction::value_ct>{ 2.0, 3.0, 0.0, 0.0 };
 *     } );
 * @endcode
 * Each pair receives 2.0*J_G0 + 3.0*J_G1 and 2.0*H_G0 + 3.0*H_G1. An LM formulation could assemble the latter into
 * its coordinate force Jacobian. This example uses the same multipliers for every pair; a production reader can
 * obtain different multipliers for each pair from the nodal field.
 *
 * @tparam PairFunction Pair function whose evaluate() result supplies the fixed-size vector f(x) and quadrature state.
 * @tparam WeightReader Device-callable object returning one pair's fixed value weights.
 * @tparam Exec Concrete execution mode for pair evaluation.
 * @param pairs Current pair list in execution-accessible memory.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param pair_function Fixed geometry-pair function configuration.
 * @param read_weights Callable with signature `MdArray<RealT, PairFunction::value_ct>(IndexT)`.
 * @return Owned weighted Jacobians and Hessians in pairs.getAllocatorID(), with one slot per pair. Empty input
 * returns zero slots. Evaluation finishes before returning; the arrays remain valid after pairs and the input meshes
 * are destroyed. Quadrature status is not returned or checked.
 * @pre The caller has checked pair quadrature for errors at these coordinates.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction, typename WeightReader>
PairJacobianHessianBuffer<PairFunction::coord_ct> evaluatePairWeightedJacobiansAndHessians(
    const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1, MeshData::Viewer mesh2, PairFunction pair_function,
    WeightReader read_weights )
{
  PairJacobianHessianBuffer<PairFunction::coord_ct> result( pairs.size(), pairs.getAllocatorID() );
  const auto contributions_view = result.view();
  const auto pairs_view = pairs.view();
  const PairCoordinateFunction coordinate_function{ pair_function };
  constexpr IndexT geom1_coord_ct = PairFunction::geom1_coord_ct;
  constexpr IndexT coord_ct = PairFunction::coord_ct;
  forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs_view[pair_index];
    MdArray<RealT, coord_ct> coordinates{};
    mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
    mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + geom1_coord_ct );
    const auto weights = read_weights( pair_index );
    const auto pair_values = EnzymePairDerivative<PairCoordinateFunction<PairFunction>>::weightedJacobianAndHessian(
        coordinates.view(), coordinate_function, weights.view() );
    for ( IndexT row = 0; row < coord_ct; ++row ) {
      contributions_view.setJacobian( pair_index, row, pair_values.jacobian[row] );
      for ( IndexT column = 0; column < coord_ct; ++column ) {
        contributions_view.setHessian( pair_index, row, column, pair_values.hessian( row, column ) );
      }
    }
  } );
  return result;
}

/**
 * @brief Evaluate each scalar pair function's value, Jacobian, and Hessian and return their buffers.
 *
 * For scalar f(x), write f, Jacobian entries J_j = df/dx_j, and Hessian entries H_jk = d^2 f/(dx_j dx_k). The Jacobian
 * has one entry per coordinate. AD supplies the value, derivatives, and quadrature state together. Failed or inactive
 * pairs receive zero values and derivatives; the caller must check quadrature results before using the outputs.
 * Active inputs retain their derivatives even when f(x) is zero.
 *
 * For example, suppose pair_function is a configured EnergyMortarEnergy of type PairFunction. A quadrature-point
 * formulation could request its scalar energy E, Jacobian J_E, and Hessian H_E together:
 * @code
 * auto results = evaluatePairValuesJacobiansAndHessians<Exec>( pairs, mesh1, mesh2, pair_function );
 * checkPairQuadratureResults<Exec>( results.quadratureResults() );
 * @endcode
 * After the check succeeds, the caller could sum E across pairs and MPI ranks, assemble J_E as the force vector,
 * and assemble H_E as its coordinate Jacobian. The first derivatives are those of the input energy E, not of the
 * resulting force vector.
 *
 * @tparam PairFunction Scalar pair function whose evaluate() result supplies values(), activation, and quadrature
 * status.
 * @tparam Exec Concrete execution mode for pair evaluation.
 * @param pairs Current pair list in execution-accessible memory.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param pair_function Fixed scalar pair function configuration.
 * @return Owned scalar values, quadrature results, Jacobians, and Hessians in pairs.getAllocatorID(), with one slot
 * per pair. Empty input returns zero slots. Evaluation finishes before returning; the arrays remain valid after pairs
 * and the input meshes are destroyed. Quadrature failures are recorded for the caller to check.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
  requires( PairFunction::value_ct == 1 )
PairValueJacobianHessianBuffer<PairFunction::coord_ct> evaluatePairValuesJacobiansAndHessians(
    const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1, MeshData::Viewer mesh2, PairFunction pair_function )
{
  PairValueJacobianHessianBuffer<PairFunction::coord_ct> result( pairs.size(), pairs.getAllocatorID() );
  const auto contributions_view = result.view();
  const auto pairs_view = pairs.view();
  const PairCoordinateFunction coordinate_function{ pair_function };
  constexpr IndexT geom1_coord_ct = PairFunction::geom1_coord_ct;
  constexpr IndexT coord_ct = PairFunction::coord_ct;
  forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs_view[pair_index];
    MdArray<RealT, coord_ct> coordinates{};
    mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
    mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + geom1_coord_ct );
    const auto pair_values = EnzymePairDerivative<PairCoordinateFunction<PairFunction>>::valueJacobianAndHessian(
        coordinates.view(), coordinate_function );
    contributions_view.values[pair_index] = pair_values.value;
    contributions_view.quadrature[pair_index] = pair_values.quadrature_result;
    for ( IndexT row = 0; row < coord_ct; ++row ) {
      contributions_view.derivatives.setJacobian( pair_index, row, pair_values.jacobian[row] );
      for ( IndexT column = 0; column < coord_ct; ++column ) {
        contributions_view.derivatives.setHessian( pair_index, row, column, pair_values.hessian( row, column ) );
      }
    }
  } );
  return result;
}
#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol

#endif  // SRC_TRIBOL_ASSEMBLY_PAIRKERNEL_HPP_
