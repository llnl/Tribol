// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Bulk pair evaluation into preallocated execution-space buffers. */
#ifndef SRC_TRIBOL_PHYSICS_PAIRKERNEL_HPP_
#define SRC_TRIBOL_PHYSICS_PAIRKERNEL_HPP_

#include <stdexcept>

#include "axom/core/execution/reductions.hpp"
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/physics/PairContribution.hpp"
#include "tribol/physics/PairEvaluation.hpp"

namespace tribol {
namespace detail {

/**
 * @brief Reduce quadrature status with a compile-time execution policy.
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
 * @brief Report pair quadrature failures after reducing the device status array.
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
 * @brief Evaluate pair values and first derivatives without assembly side effects.
 * @tparam Function Fixed-size vector pair function.
 * @param execution_mode Host or device execution policy.
 * @param pairs Current pair list in execution-accessible memory.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param function Fixed pair-function configuration.
 * @param output Preallocated output with one slot per pair.
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
 * @brief Evaluate weighted pair gradients and Hessians without assembly side effects.
 * @tparam Function Fixed-size vector pair function.
 * @tparam WeightReader Device-callable object returning one pair's fixed value weights.
 * @param execution_mode Host or device execution policy.
 * @param pairs Current pair list in execution-accessible memory.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param function Fixed pair-function configuration.
 * @param read_weights Callable with signature `MdArray<RealT, Function::value_count>(IndexT)`.
 * @param output Preallocated output with one slot per pair.
 */
template <typename Function, typename WeightReader>
void evaluatePairWeightedHessians( ExecutionMode execution_mode, ArrayViewT<const InterfacePair> pairs,
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

}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_PAIRKERNEL_HPP_
