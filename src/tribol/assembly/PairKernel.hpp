// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Evaluate pair functions f and their coordinate Jacobians and Hessians into preallocated buffers. */
#ifndef SRC_TRIBOL_ASSEMBLY_PAIRKERNEL_HPP_
#define SRC_TRIBOL_ASSEMBLY_PAIRKERNEL_HPP_

#include <stdexcept>

#include "axom/core/execution/reductions.hpp"
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/assembly/PairContribution.hpp"
#include "tribol/assembly/PairEvaluation.hpp"

namespace tribol {
namespace detail {

/**
 * @brief Find the most severe quadrature error across all pairs using the specified execution policy.
 * @tparam Exec Concrete execution policy.
 * @param results One status per pair in execution-accessible memory.
 * @return Highest-severity status present in the array.
 */
template <ExecutionMode Exec>
QuadratureStatus reduceQuadratureStatus( ArrayViewT<const QuadratureResult> results )
{
  using ExecSpace = typename toAxomExecSpace<Exec>::type;
  axom::ReduceMax<ExecSpace, int> status( static_cast<int>( QuadratureStatus::Success ) );
  forAllExec<Exec>( results.size(),
                    [=] TRIBOL_HOST_DEVICE( IndexT pair ) { status.max( static_cast<int>( results[pair].status ) ); } );
  return static_cast<QuadratureStatus>( status.get() );
}

}  // namespace detail

/**
 * @brief Check the quadrature rules produced for all pairs and throw if any rule reports an error.
 * @param execution_mode Execution policy used to populate the results.
 * @param results One preallocated result per pair.
 */
inline void checkPairQuadratureResults( ExecutionMode execution_mode, const Array1D<QuadratureResult>& results )
{
  QuadratureStatus status = QuadratureStatus::Success;
  switch ( execution_mode ) {
    case ExecutionMode::Sequential:
      status = detail::reduceQuadratureStatus<ExecutionMode::Sequential>( results.view() );
      break;
#ifdef TRIBOL_USE_OPENMP
    case ExecutionMode::OpenMP:
      status = detail::reduceQuadratureStatus<ExecutionMode::OpenMP>( results.view() );
      break;
#endif
#ifdef TRIBOL_USE_CUDA
    case ExecutionMode::Cuda:
      status = detail::reduceQuadratureStatus<ExecutionMode::Cuda>( results.view() );
      break;
#endif
#ifdef TRIBOL_USE_HIP
    case ExecutionMode::Hip:
      status = detail::reduceQuadratureStatus<ExecutionMode::Hip>( results.view() );
      break;
#endif
    default:
      throw std::invalid_argument( "Pair status reduction requires a concrete execution mode." );
  }
  if ( status != QuadratureStatus::Success ) {
    reportQuadratureFailure( status );
  }
}

/**
 * @brief Require a pair evaluation and a compile-time coordinate split with entries for both geometries.
 *
 * The kernel reads geometry1 coordinates first, followed by geometry2 coordinates starting at first_dofs.
 * Derivative calculations on an already packed coordinate array require only PairFunctionType.
 * @tparam PairFunction Type evaluating a pair gathered from two meshes.
 */
template <typename PairFunction>
concept PairKernelFunctionType = PairFunctionType<PairFunction> && requires {
  typename std::integral_constant<IndexT, PairFunction::first_dofs>;
  requires( PairFunction::first_dofs > 0 && PairFunction::first_dofs < PairFunction::dof_count );
};

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Evaluate each pair's values and coordinate Jacobian into preallocated buffers.
 *
 * For the input function f(x), write every value f_i and every Jacobian entry J_ij = df_i/dx_j. This applies equally
 * to scalar and vector functions. The values already include any quadrature weights used by f. The caller must check
 * the recorded quadrature results before using these outputs.
 *
 * For example, suppose pair_function is a configured CommonPlanePairFunction of type PairFunction, pairs is an accepted
 * pair list, and mesh1 and mesh2 are current mesh views. A caller could evaluate its local nodal forces and their
 * Jacobians:
 * @code
 * PairValueJacobianBuffer<PairFunction::value_count, PairFunction::dof_count> results(
 *     pairs.size(), pairs.getAllocatorID() );
 * evaluatePairValuesAndJacobians( execution_mode, pairs.view(), mesh1, mesh2, pair_function, results.view() );
 * checkPairQuadratureResults( execution_mode, results.quadratureResults() );
 * @endcode
 * After the check succeeds, the caller could assemble the values into a force vector and the Jacobians into a matrix.
 * This kernel stores each pair's contribution separately, including when pairs share nodes.
 *
 * @tparam PairFunction Pair function whose evaluate() result supplies the fixed-size vector f(x) and quadrature state.
 * @param execution_mode Host or device execution policy.
 * @param pairs Current pair list in execution-accessible memory.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param pair_function Fixed pair function configuration.
 * @param output Preallocated values, Jacobians, quadrature results, and activation flags, with one slot per pair.
 * @throws std::invalid_argument Buffer size does not match the pair list; no outputs are written.
 */
template <PairKernelFunctionType PairFunction>
void evaluatePairValuesAndJacobians( ExecutionMode execution_mode, ArrayViewT<const InterfacePair> pairs,
                                     MeshData::Viewer mesh1, MeshData::Viewer mesh2, PairFunction pair_function,
                                     PairValueJacobianView<PairFunction::value_count, PairFunction::dof_count> output )
{
  constexpr IndexT first_dofs = PairFunction::first_dofs;
  constexpr IndexT dof_count = PairFunction::dof_count;
  if ( output.pair_count != pairs.size() ) {
    throw std::invalid_argument( "Pair value buffer size must match the pair list." );
  }
  forAllExec( execution_mode, pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs[pair_index];
    MdArray<RealT, dof_count> coordinates{};
    mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
    mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + first_dofs );
    const auto pair_values = EnzymePairDerivative<PairFunction>::valuesAndJacobian( coordinates.view(), pair_function );
    output.quadrature[pair_index] = pair_values.quadrature_result;
    output.has_active_points[pair_index] = pair_values.has_active_points;
    for ( IndexT value = 0; value < PairFunction::value_count; ++value ) {
      output.setValue( pair_index, value, pair_values.values[value] );
      for ( IndexT coordinate = 0; coordinate < dof_count; ++coordinate ) {
        output.setJacobian( pair_index, value, coordinate, pair_values.jacobian( value, coordinate ) );
      }
    }
  } );
}

/**
 * @brief Evaluate weighted Jacobians and Hessians of each pair function f into preallocated buffers.
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
 * PairHessianBuffer<PairFunction::dof_count> derivatives( pairs.size(), pairs.getAllocatorID() );
 * evaluatePairWeightedJacobiansAndHessians(
 *     execution_mode, pairs.view(), mesh1, mesh2, pair_function,
 *     [] TRIBOL_HOST_DEVICE( IndexT ) {
 *       return MdArray<RealT, PairFunction::value_count>{ 2.0, 3.0, 0.0, 0.0 };
 *     },
 *     derivatives.view() );
 * @endcode
 * Each pair receives 2.0*J_G0 + 3.0*J_G1 and 2.0*H_G0 + 3.0*H_G1. An LM formulation could assemble the latter into
 * its coordinate force Jacobian. This example uses the same multipliers for every pair; a production reader can
 * obtain different multipliers for each pair from the nodal field.
 *
 * @tparam PairFunction Pair function whose evaluate() result supplies the fixed-size vector f(x) and quadrature state.
 * @tparam WeightReader Device-callable object returning one pair's fixed value weights.
 * @param execution_mode Host or device execution policy.
 * @param pairs Current pair list in execution-accessible memory.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param pair_function Fixed pair function configuration.
 * @param read_weights Callable with signature `MdArray<RealT, PairFunction::value_count>(IndexT)`.
 * @param output Preallocated weighted Jacobians and Hessians, with one slot per pair.
 * @pre The caller has checked pair quadrature for errors at these coordinates.
 * @throws std::invalid_argument Buffer size does not match the pair list; no outputs are written.
 */
template <PairKernelFunctionType PairFunction, typename WeightReader>
void evaluatePairWeightedJacobiansAndHessians( ExecutionMode execution_mode, ArrayViewT<const InterfacePair> pairs,
                                               MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                               PairFunction pair_function, WeightReader read_weights,
                                               PairHessianView<PairFunction::dof_count> output )
{
  constexpr IndexT first_dofs = PairFunction::first_dofs;
  constexpr IndexT dof_count = PairFunction::dof_count;
  if ( output.pair_count != pairs.size() ) {
    throw std::invalid_argument( "Pair Hessian buffer size must match the pair list." );
  }
  forAllExec( execution_mode, pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs[pair_index];
    MdArray<RealT, dof_count> coordinates{};
    mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
    mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + first_dofs );
    const auto weights = read_weights( pair_index );
    const auto pair_values = EnzymePairDerivative<PairFunction>::weightedJacobianAndHessian(
        coordinates.view(), pair_function, weights.view() );
    for ( IndexT row = 0; row < dof_count; ++row ) {
      output.setGradient( pair_index, row, pair_values.jacobian[row] );
      for ( IndexT column = 0; column < dof_count; ++column ) {
        output.setHessian( pair_index, row, column, pair_values.hessian( row, column ) );
      }
    }
  } );
}

/**
 * @brief Evaluate each scalar pair function's value, Jacobian, and Hessian into preallocated buffers.
 *
 * For scalar f(x), write f, Jacobian entries J_j = df/dx_j, and Hessian entries H_jk = d^2 f/(dx_j dx_k). The Jacobian
 * has one entry per coordinate. AD supplies the value, derivatives, and quadrature state together. Failed or inactive
 * pairs receive zero values and derivatives; the caller must check quadrature results before using the outputs.
 * Active inputs retain their derivatives even when f(x) is zero.
 *
 * For example, suppose pair_function is a configured EnergyMortarEnergy of type PairFunction. A quadrature-point
 * formulation could request its scalar energy E, Jacobian J_E, and Hessian H_E together:
 * @code
 * PairEnergyHessianBuffer<PairFunction::dof_count> results( pairs.size(), pairs.getAllocatorID() );
 * evaluatePairValuesJacobiansAndHessians( execution_mode, pairs.view(), mesh1, mesh2, pair_function, results.view() );
 * checkPairQuadratureResults( execution_mode, results.quadratureResults() );
 * @endcode
 * After the check succeeds, the caller could sum E across pairs and MPI ranks, assemble J_E as the force vector,
 * and assemble H_E as its coordinate Jacobian. The first derivatives are those of the input energy E, not of the
 * resulting force vector.
 *
 * @tparam PairFunction Scalar pair function whose evaluate() result supplies values(), activation, and quadrature
 * status.
 * @param execution_mode Host or device execution policy.
 * @param pairs Current pair list in execution-accessible memory.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param pair_function Fixed scalar pair function configuration.
 * @param output Preallocated scalar values, quadrature results, Jacobians, and Hessians, with one slot per pair.
 * @throws std::invalid_argument Buffer size does not match the pair list; no outputs are written.
 */
template <PairKernelFunctionType PairFunction>
  requires( PairFunction::value_count == 1 )
void evaluatePairValuesJacobiansAndHessians( ExecutionMode execution_mode, ArrayViewT<const InterfacePair> pairs,
                                             MeshData::Viewer mesh1, MeshData::Viewer mesh2, PairFunction pair_function,
                                             PairEnergyHessianView<PairFunction::dof_count> output )
{
  constexpr IndexT first_dofs = PairFunction::first_dofs;
  constexpr IndexT dof_count = PairFunction::dof_count;
  if ( output.derivatives.pair_count != pairs.size() ) {
    throw std::invalid_argument( "Pair energy buffer size must match the pair list." );
  }
  forAllExec( execution_mode, pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs[pair_index];
    MdArray<RealT, dof_count> coordinates{};
    mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
    mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + first_dofs );
    const auto pair_values =
        EnzymePairDerivative<PairFunction>::valueJacobianAndHessian( coordinates.view(), pair_function );
    output.energies[pair_index] = pair_values.value;
    output.quadrature[pair_index] = pair_values.quadrature_result;
    for ( IndexT row = 0; row < dof_count; ++row ) {
      output.derivatives.setGradient( pair_index, row, pair_values.jacobian[row] );
      for ( IndexT column = 0; column < dof_count; ++column ) {
        output.derivatives.setHessian( pair_index, row, column, pair_values.hessian( row, column ) );
      }
    }
  } );
}
#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol

#endif  // SRC_TRIBOL_ASSEMBLY_PAIRKERNEL_HPP_
