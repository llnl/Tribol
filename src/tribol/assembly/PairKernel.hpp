// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Evaluate pair functions f and return buffers of their values, coordinate Jacobians, and Hessians.
 *
 * Pair functions accept two coordinate views and construct any needed geometry inside evaluate(). Both coordinate
 * arrays are active inputs for differentiation. Jacobian columns and Hessian rows/columns follow geometry1
 * node, then value-index order, followed by geometry2 order.
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
 * @tparam PairFunction Function evaluating two rank-two node-by-spatial-dimension coordinate views.
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
  requires( PairFunction::Coords1::rank() == 2 && PairFunction::Coords2::rank() == 2 )
PairValueBuffer<PairFunction::size_values> evaluatePairValues( const ArrayT<InterfacePair>& pairs,
                                                               MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                               PairFunction pair_function )
{
  PairValueBuffer<PairFunction::size_values> all_pair_values( pairs.size(), pairs.getAllocatorID() );
  const auto all_pair_values_view = all_pair_values.view();
  const auto pairs_view = pairs.view();
  forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs_view[pair_index];
    // Call function to evaluate pair quantity
    const auto coords1 = mesh1.getFaceCoords<typename PairFunction::Coords1>( pair.element_id1_ );
    const auto coords2 = mesh2.getFaceCoords<typename PairFunction::Coords2>( pair.element_id2_ );
    const auto pair_values_struct = pair_function.evaluate( coords1.view(), coords2.view() );
    // Fill the buffer with quantities evaluated at the pair
    all_pair_values_view.quadrature[pair_index] = pair_values_struct.quadrature_result;
    all_pair_values_view.has_contributing_points[pair_index] = pair_values_struct.has_contributing_points;
    const auto pair_values = pair_values_struct.values();
    for ( IndexT i{ 0 }; i < PairFunction::size_values; ++i ) {
      all_pair_values_view.setValue( pair_index, i, pair_values[i] );
    }
  } );
  return all_pair_values;
}

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Evaluate each pair's values and coordinate Jacobian and return their buffers.
 *
 * For the input function f(x), write every value f_i and every Jacobian entry J_ij = df_i/dx_j. This applies equally
 * to scalar and vector functions. The values already include any quadrature weights used by f. The caller must check
 * the recorded quadrature results before using these outputs.
 *
 * For example, suppose function is a configured CommonPlaneResidualPairFunction of type PairFunction, pairs is an
 * accepted pair list, and mesh1 and mesh2 are current mesh views. A caller could evaluate its local nodal forces and
 * their Jacobians:
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
 * @param pair_function Fixed pair-function configuration.
 * @return Owned values, Jacobians, quadrature results, and activation flags in pairs.getAllocatorID(). Empty input
 * returns zero slots. Evaluation finishes before returning; the arrays remain valid after pairs and the input meshes
 * are destroyed. Quadrature failures are recorded for the caller to check.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
  requires( PairFunction::Coords1::rank() == 2 && PairFunction::Coords2::rank() == 2 )
PairValueJacobianBuffer<PairFunction::size_values, PairFunction::size_coords> evaluatePairValuesAndJacobians(
    const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1, MeshData::Viewer mesh2, PairFunction pair_function )
{
  PairValueJacobianBuffer<PairFunction::size_values, PairFunction::size_coords> all_pair_values(
      pairs.size(), pairs.getAllocatorID() );
  const auto all_pair_values_view = all_pair_values.view();
  const auto pairs_view = pairs.view();
  forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs_view[pair_index];
    // Evaluate pair values and their coordinate Jacobian
    const auto coords1 = mesh1.getFaceCoords<typename PairFunction::Coords1>( pair.element_id1_ );
    const auto coords2 = mesh2.getFaceCoords<typename PairFunction::Coords2>( pair.element_id2_ );
    const auto pair_values_struct =
        EnzymePairDerivative<PairFunction>::valuesAndJacobian( coords1.view(), coords2.view(), pair_function );
    // Fill the buffer with quantities evaluated at the pair
    all_pair_values_view.quadrature[pair_index] = pair_values_struct.quadrature_result;
    all_pair_values_view.has_contributing_points[pair_index] = pair_values_struct.has_contributing_points;
    const auto& pair_values = pair_values_struct.values;
    for ( IndexT i{ 0 }; i < PairFunction::size_values; ++i ) {
      all_pair_values_view.setValue( pair_index, i, pair_values[i] );
      for ( IndexT j{ 0 }; j < PairFunction::size_coords; ++j ) {
        all_pair_values_view.setJacobian( pair_index, i, j, pair_values_struct.jacobian( i, j ) );
      }
    }
  } );
  return all_pair_values;
}

/**
 * @brief Evaluate weighted Jacobians and Hessians of each pair function f and return their buffers.
 *
 * Let J_ij = df_i/dx_j and H_ijk = d^2 f_i/(dx_j dx_k). For fixed output weights w_i, write sum_i w_i J_ij and
 * sum_i w_i H_ijk. The weighted Jacobian has one entry per coordinate, and the weighted Hessian is a square coordinate
 * matrix. Function values and the full unweighted Jacobian are not returned. The weights multiply outputs of f;
 * they are distinct from quadrature weights, which are already included in f.
 *
 * For example, suppose function is an EnergyMortarNodalIntegralPairFunction of type PairFunction for two-node
 * edges and its quadrature has already been checked. Its four outputs are weighted gaps G0, G1 followed by areas A0,
 * A1. A hypothetical nodal LM calculation could use multipliers 2.0 and 3.0, and zero area weights, to
 * differentiate 2.0*G0
 * + 3.0*G1:
 * @code
 * auto derivatives = evaluatePairWeightedJacobiansAndHessians<Exec>(
 *     pairs, mesh1, mesh2, pair_function,
 *     [] TRIBOL_HOST_DEVICE( IndexT ) {
 *       return MdArray<RealT, PairFunction::size_values>{ 2.0, 3.0, 0.0, 0.0 };
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
 * @param pair_function Fixed pair-function configuration.
 * @param read_weights Callable with signature `MdArray<RealT, PairFunction::size_values>(IndexT)`.
 * @return Owned weighted Jacobians and Hessians in pairs.getAllocatorID(), with one slot per pair. Empty input
 * returns zero slots. Evaluation finishes before returning; the arrays remain valid after pairs and the input meshes
 * are destroyed. Quadrature status is not returned or checked.
 * @pre The caller has checked pair quadrature for errors at these coordinates.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction, typename WeightReader>
  requires( PairFunction::Coords1::rank() == 2 && PairFunction::Coords2::rank() == 2 )
PairJacobianHessianBuffer<PairFunction::size_coords> evaluatePairWeightedJacobiansAndHessians(
    const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1, MeshData::Viewer mesh2, PairFunction pair_function,
    WeightReader read_weights )
{
  PairJacobianHessianBuffer<PairFunction::size_coords> all_pair_values( pairs.size(), pairs.getAllocatorID() );
  const auto all_pair_values_view = all_pair_values.view();
  const auto pairs_view = pairs.view();
  forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs_view[pair_index];
    // Evaluate the weighted coordinate Jacobian and Hessian
    const auto coords1 = mesh1.getFaceCoords<typename PairFunction::Coords1>( pair.element_id1_ );
    const auto coords2 = mesh2.getFaceCoords<typename PairFunction::Coords2>( pair.element_id2_ );
    const auto weights = read_weights( pair_index );
    const auto pair_values_struct = EnzymePairDerivative<PairFunction>::weightedJacobianAndHessian(
        coords1.view(), coords2.view(), pair_function, weights.view() );
    // Fill the buffer with quantities evaluated at the pair
    for ( IndexT i{ 0 }; i < PairFunction::size_coords; ++i ) {
      all_pair_values_view.setJacobian( pair_index, i, pair_values_struct.jacobian[i] );
      for ( IndexT j{ 0 }; j < PairFunction::size_coords; ++j ) {
        all_pair_values_view.setHessian( pair_index, i, j, pair_values_struct.hessian( i, j ) );
      }
    }
  } );
  return all_pair_values;
}

/**
 * @brief Evaluate each scalar pair function's value, Jacobian, and Hessian and return their buffers.
 *
 * For scalar f(x), write f, Jacobian entries J_j = df/dx_j, and Hessian entries H_jk = d^2 f/(dx_j dx_k). The Jacobian
 * has one entry per coordinate. AD supplies the value, derivatives, and quadrature state together. Failed or inactive
 * pairs receive zero values and derivatives; the caller must check quadrature results before using the outputs.
 * Active inputs retain their derivatives even when f(x) is zero.
 *
 * For example, suppose function is a configured EnergyMortarEnergyPairFunction of type PairFunction. A
 * quadrature-point formulation could request its scalar energy E, Jacobian J_E, and Hessian H_E together:
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
  requires( PairFunction::size_values == 1 && PairFunction::Coords1::rank() == 2 && PairFunction::Coords2::rank() == 2 )
PairValueJacobianHessianBuffer<PairFunction::size_coords> evaluatePairValuesJacobiansAndHessians(
    const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1, MeshData::Viewer mesh2, PairFunction pair_function )
{
  PairValueJacobianHessianBuffer<PairFunction::size_coords> all_pair_values( pairs.size(), pairs.getAllocatorID() );
  const auto all_pair_values_view = all_pair_values.view();
  const auto pairs_view = pairs.view();
  forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs_view[pair_index];
    // Evaluate the scalar pair value, coordinate Jacobian, and Hessian
    const auto coords1 = mesh1.getFaceCoords<typename PairFunction::Coords1>( pair.element_id1_ );
    const auto coords2 = mesh2.getFaceCoords<typename PairFunction::Coords2>( pair.element_id2_ );
    const auto pair_values_struct =
        EnzymePairDerivative<PairFunction>::valueJacobianAndHessian( coords1.view(), coords2.view(), pair_function );
    // Fill the buffer with quantities evaluated at the pair
    all_pair_values_view.values[pair_index] = pair_values_struct.value;
    all_pair_values_view.quadrature[pair_index] = pair_values_struct.quadrature_result;
    for ( IndexT i{ 0 }; i < PairFunction::size_coords; ++i ) {
      all_pair_values_view.derivatives.setJacobian( pair_index, i, pair_values_struct.jacobian[i] );
      for ( IndexT j{ 0 }; j < PairFunction::size_coords; ++j ) {
        all_pair_values_view.derivatives.setHessian( pair_index, i, j, pair_values_struct.hessian( i, j ) );
      }
    }
  } );
  return all_pair_values;
}
#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol

#endif  // SRC_TRIBOL_ASSEMBLY_PAIRKERNEL_HPP_
