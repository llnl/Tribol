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

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Evaluate each pair's values and coordinate Jacobian into preallocated buffers.
 *
 * For the input function f(x), write every value f_i and every Jacobian entry J_ij = df_i/dx_j. This applies equally
 * to scalar and vector functions. The values already include any quadrature weights used by f. The caller must check
 * the recorded quadrature results before using these outputs.
 *
 * For example, suppose f is a configured CommonPlanePairFunction of type Function, pairs is an accepted pair list,
 * and mesh1 and mesh2 are current mesh views. A caller could evaluate its local nodal forces and their Jacobians:
 * @code
 * PairValueJacobianBuffer<Function::value_count, Function::dof_count> results(
 *     pairs.size(), pairs.getAllocatorID() );
 * evaluatePairValuesAndJacobians( execution_mode, pairs.view(), mesh1, mesh2, f, results.view() );
 * checkPairQuadratureResults( execution_mode, results.quadratureResults() );
 * @endcode
 * After the check succeeds, the caller could assemble the values into a force vector and the Jacobians into a matrix.
 * This kernel stores each pair's contribution separately, including when pairs share nodes.
 *
 * @tparam Function Pair function whose values() returns the fixed-size vector f(x).
 * @param execution_mode Host or device execution policy.
 * @param pairs Current pair list in execution-accessible memory.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param function Fixed pair-function configuration.
 * @param output Preallocated values, Jacobians, quadrature results, and activation flags, with one slot per pair.
 * @throws std::invalid_argument Buffer size does not match the pair list; no outputs are written.
 */
template <typename Function>
void evaluatePairValuesAndJacobians( ExecutionMode execution_mode, ArrayViewT<const InterfacePair> pairs,
                                     MeshData::Viewer mesh1, MeshData::Viewer mesh2, Function function,
                                     PairValueJacobianView<Function::value_count, Function::dof_count> output )
{
  constexpr IndexT first_dofs = Function::first_dofs;
  constexpr IndexT dof_count = Function::dof_count;
  if ( output.pair_count != pairs.size() ) {
    throw std::invalid_argument( "Pair value buffer size must match the pair list." );
  }
  forAllExec( execution_mode, pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs[pair_index];
    MdArray<RealT, dof_count> coordinates{};
    mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
    mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + first_dofs );
    const auto primal = function.evaluate( coordinates.view() );
    output.quadrature[pair_index] = primal.quadrature_result;
    output.has_active_points[pair_index] = primal.has_active_points;
    const auto derivatives = EnzymePairDerivative<Function>::valuesAndJacobian( coordinates.view(), function );
    for ( IndexT value = 0; value < Function::value_count; ++value ) {
      output.setValue( pair_index, value, derivatives.values[value] );
      for ( IndexT coordinate = 0; coordinate < dof_count; ++coordinate ) {
        output.setJacobian( pair_index, value, coordinate, derivatives.gradients( value, coordinate ) );
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
 * For example, suppose f is an EnergyMortarNodalEnergy of type Function for two-node edges and its quadrature has
 * already been checked. Its four outputs are weighted gaps G0, G1 followed by areas A0, A1. A hypothetical nodal LM
 * calculation could use multipliers 2.0 and 3.0, and zero area weights, to differentiate 2.0*G0 + 3.0*G1:
 * @code
 * PairHessianBuffer<Function::dof_count> derivatives( pairs.size(), pairs.getAllocatorID() );
 * evaluatePairWeightedJacobiansAndHessians(
 *     execution_mode, pairs.view(), mesh1, mesh2, f,
 *     [] TRIBOL_HOST_DEVICE( IndexT ) {
 *       return MdArray<RealT, Function::value_count>{ 2.0, 3.0, 0.0, 0.0 };
 *     },
 *     derivatives.view() );
 * @endcode
 * Each pair receives 2.0*J_G0 + 3.0*J_G1 and 2.0*H_G0 + 3.0*H_G1. An LM formulation could assemble the latter into
 * its coordinate force Jacobian. This example uses the same multipliers for every pair; a production reader can
 * obtain different multipliers for each pair from the nodal field.
 *
 * @tparam Function Pair function whose values() returns the fixed-size vector f(x).
 * @tparam WeightReader Device-callable object returning one pair's fixed value weights.
 * @param execution_mode Host or device execution policy.
 * @param pairs Current pair list in execution-accessible memory.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param function Fixed pair-function configuration.
 * @param read_weights Callable with signature `MdArray<RealT, Function::value_count>(IndexT)`.
 * @param output Preallocated weighted Jacobians and Hessians, with one slot per pair.
 * @pre The caller has checked pair quadrature for errors at these coordinates.
 * @throws std::invalid_argument Buffer size does not match the pair list; no outputs are written.
 */
template <typename Function, typename WeightReader>
void evaluatePairWeightedJacobiansAndHessians( ExecutionMode execution_mode, ArrayViewT<const InterfacePair> pairs,
                                               MeshData::Viewer mesh1, MeshData::Viewer mesh2, Function function,
                                               WeightReader read_weights, PairHessianView<Function::dof_count> output )
{
  constexpr IndexT first_dofs = Function::first_dofs;
  constexpr IndexT dof_count = Function::dof_count;
  if ( output.pair_count != pairs.size() ) {
    throw std::invalid_argument( "Pair Hessian buffer size must match the pair list." );
  }
  forAllExec( execution_mode, pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs[pair_index];
    MdArray<RealT, dof_count> coordinates{};
    mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
    mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + first_dofs );
    const auto weights = read_weights( pair_index );
    const auto derivatives =
        EnzymePairDerivative<Function>::weightedHessian( coordinates.view(), function, weights.view() );
    for ( IndexT row = 0; row < dof_count; ++row ) {
      output.setGradient( pair_index, row, derivatives.gradient[row] );
      for ( IndexT column = 0; column < dof_count; ++column ) {
        output.setHessian( pair_index, row, column, derivatives.hessian( row, column ) );
      }
    }
  } );
}

/**
 * @brief Evaluate each scalar pair function's value, Jacobian, and Hessian into preallocated buffers.
 *
 * For scalar f(x), write f, Jacobian entries J_j = df/dx_j, and Hessian entries H_jk = d^2 f/(dx_j dx_k). The Jacobian
 * has one entry per coordinate. Internally, a unit output weight selects this scalar's derivatives. Failed or inactive
 * pairs receive zero values and derivatives; the caller must check quadrature results before using the outputs.
 * Active inputs retain their derivatives even when f(x) is zero.
 *
 * For example, suppose f is a configured EnergyMortarEnergy of type Function. A quadrature-point formulation could
 * request its scalar energy E, Jacobian J_E, and Hessian H_E together:
 * @code
 * PairEnergyHessianBuffer<Function::dof_count> results( pairs.size(), pairs.getAllocatorID() );
 * evaluatePairValuesJacobiansAndHessians( execution_mode, pairs.view(), mesh1, mesh2, f, results.view() );
 * checkPairQuadratureResults( execution_mode, results.quadratureResults() );
 * @endcode
 * After the check succeeds, the caller could sum E across pairs and MPI ranks, assemble J_E as the force vector,
 * and assemble H_E as its coordinate Jacobian. The first derivatives are those of the input energy E, not of the
 * resulting force vector.
 *
 * @tparam Function Scalar pair function returning f in evaluate().energy, with activation and quadrature status.
 * @param execution_mode Host or device execution policy.
 * @param pairs Current pair list in execution-accessible memory.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param function Fixed scalar pair-function configuration.
 * @param output Preallocated scalar values, quadrature results, Jacobians, and Hessians, with one slot per pair.
 * @throws std::invalid_argument Buffer size does not match the pair list; no outputs are written.
 */
template <typename Function>
void evaluatePairValuesJacobiansAndHessians( ExecutionMode execution_mode, ArrayViewT<const InterfacePair> pairs,
                                             MeshData::Viewer mesh1, MeshData::Viewer mesh2, Function function,
                                             PairEnergyHessianView<Function::dof_count> output )
{
  static_assert( Function::value_count == 1, "Pair energy evaluation requires one scalar output." );
  constexpr IndexT first_dofs = Function::first_dofs;
  constexpr IndexT dof_count = Function::dof_count;
  if ( output.derivatives.pair_count != pairs.size() ) {
    throw std::invalid_argument( "Pair energy buffer size must match the pair list." );
  }
  forAllExec( execution_mode, pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs[pair_index];
    MdArray<RealT, dof_count> coordinates{};
    mesh1.getFaceCoords( pair.element_id1_, coordinates.data() );
    mesh2.getFaceCoords( pair.element_id2_, coordinates.data() + first_dofs );
    const auto primal = function.evaluate( coordinates.view() );
    const bool active = primal.quadrature_result.status == QuadratureStatus::Success && primal.has_active_points;
    output.energies[pair_index] = active ? primal.energy : 0.0;
    output.quadrature[pair_index] = primal.quadrature_result;
    PairWeightedHessian<dof_count> derivatives{};
    if ( active ) {
      const MdArray<RealT, 1> weights{ 1.0 };
      derivatives = EnzymePairDerivative<Function>::weightedHessian( coordinates.view(), function, weights.view() );
    }
    for ( IndexT row = 0; row < dof_count; ++row ) {
      output.derivatives.setGradient( pair_index, row, derivatives.gradient[row] );
      for ( IndexT column = 0; column < dof_count; ++column ) {
        output.derivatives.setHessian( pair_index, row, column, derivatives.hessian( row, column ) );
      }
    }
  } );
}
#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol

#endif  // SRC_TRIBOL_ASSEMBLY_PAIRKERNEL_HPP_
