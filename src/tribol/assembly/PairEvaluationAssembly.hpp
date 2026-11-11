// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Evaluate pair functions and assemble their forces and coordinate Jacobians on MFEM parent DOFs. */
#ifndef SRC_TRIBOL_ASSEMBLY_PAIREVALUATIONASSEMBLY_HPP_
#define SRC_TRIBOL_ASSEMBLY_PAIREVALUATIONASSEMBLY_HPP_

#include "tribol/config.hpp"

#ifdef BUILD_REDECOMP
#include "tribol/assembly/MfemPairAssembler.hpp"
#include "tribol/common/GlobalScalarAssembler.hpp"
#include "tribol/assembly/PairContribution.hpp"
#include "tribol/assembly/PairKernel.hpp"

namespace tribol {
namespace detail {

/**
 * @brief Evaluate pair residual values and assemble forces without derivatives or matrix storage.
 * @tparam Function Function with one residual entry per coordinate in element/node/component order.
 * @param execution_mode Host or device execution policy.
 * @param pairs Accepted element pairs in execution-accessible memory.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param function Fixed pair residual configuration.
 * @param mesh_data Borrowed MFEM displacement transfer data.
 * @return Owned force on parent true DOFs; df_dx and other outputs are absent. Empty or inactive pairs contribute zero.
 * Scratch arrays remain alive until evaluation and assembly finish.
 * @throws std::invalid_argument Geometry, execution mode, or assembly dimensions are invalid; no result is returned.
 * @throws std::length_error Quadrature capacity or the supported index range is exceeded; no result is returned.
 */
template <PairKernelFunctionType Function>
ContactFormulationResult evaluateAndAssembleResidualValues( ExecutionMode execution_mode,
                                                            const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                            MeshData::Viewer mesh2, const Function& function,
                                                            const MfemMeshData& mesh_data )
{
  static_assert( Function::value_count == Function::dof_count, "Pair residuals must match coordinate DOFs." );
  PairValueBuffer<Function::value_count> contributions( pairs.size(), pairs.getAllocatorID() );
  const auto view = contributions.view();
  evaluatePairValues( execution_mode, pairs.view(), mesh1, mesh2, function, view );
  checkPairQuadratureResults( execution_mode, contributions.quadratureResults() );
  MfemPairAssembler assembler( mesh_data, nullptr, mesh1, mesh2, pairs.size() );
  assembler.addPairResiduals<Function::dof_count>(
      execution_mode, pairs, [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) {
        return view.has_active_points[pair] ? view.value( pair, row ) : 0.0;
      } );
  return assembler.assemble();
}

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Evaluate pair residuals and assemble their values as forces and their Jacobians as df_dx.
 *
 * Evaluation and assembly use separate kernels. Scratch arrays are allocated for the supplied pair list and remain
 * alive until all operations finish. Quadrature failures are reported before any contributions are assembled.
 *
 * @tparam Function Pair function with one residual entry per coordinate in element/node/component order.
 * @param execution_mode Concrete host or device execution mode for evaluation and numerical assembly.
 * @param pairs Accepted element pairs in execution-accessible memory.
 * @param mesh1 Current first mesh with execution-accessible coordinates and connectivity.
 * @param mesh2 Current second mesh with execution-accessible coordinates and connectivity.
 * @param function Pair residual configuration; borrowed for this call.
 * @param mesh_data Current MFEM displacement transfer data; borrowed for this call.
 * @param jacobian_data Current MFEM Jacobian transfer data; borrowed for this call.
 * @return Owned parent true-DOF force and df_dx; other outputs are absent. Empty or inactive pairs contribute zero.
 * An empty list returns allocated zero force and df_dx.
 * @throws std::invalid_argument Pair geometry, execution mode, or assembly dimensions are invalid; no result is
 * returned.
 * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded; no result is
 * returned.
 */
template <typename Function>
ContactFormulationResult evaluateAndAssembleResidualPairs( ExecutionMode execution_mode,
                                                           const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                           MeshData::Viewer mesh2, const Function& function,
                                                           const MfemMeshData& mesh_data,
                                                           const MfemJacobianData& jacobian_data )
{
  static_assert( Function::value_count == Function::dof_count, "Pair residuals must match coordinate DOFs." );
  PairValueJacobianBuffer<Function::value_count, Function::dof_count> contributions( pairs.size(),
                                                                                     pairs.getAllocatorID() );
  const auto view = contributions.view();
  evaluatePairValuesAndJacobians( execution_mode, pairs.view(), mesh1, mesh2, function, view );
  checkPairQuadratureResults( execution_mode, contributions.quadratureResults() );

  MfemPairAssembler assembler( mesh_data, jacobian_data, mesh1, mesh2, pairs.size() );
  assembler.addPairs<Function::dof_count>(
      execution_mode, pairs,
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) {
        return view.has_active_points[pair] ? view.value( pair, row ) : 0.0;
      },
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
        return view.has_active_points[pair] ? view.jacobian( pair, row, column ) : 0.0;
      } );
  return assembler.assemble();
}

/**
 * @brief Evaluate scalar pair energies and assemble their Jacobians as forces and their Hessians as df_dx.
 *
 * The energy kernel reuses the Jacobian computed with each Hessian as the force contribution. Evaluation and assembly
 * use separate kernels; scratch arrays remain alive until assembly and the global energy sum finish. Quadrature
 * failures are reported before any contributions are assembled.
 *
 * @tparam Function Scalar energy function with coordinates in element/node/component order.
 * @param execution_mode Concrete host or device execution mode for evaluation and numerical assembly.
 * @param pairs Accepted element pairs in execution-accessible memory.
 * @param mesh1 Current first mesh with execution-accessible coordinates and connectivity.
 * @param mesh2 Current second mesh with execution-accessible coordinates and connectivity.
 * @param function Pair energy configuration; borrowed for this call.
 * @param mesh_data Current MFEM displacement transfer data; borrowed for this call.
 * @param jacobian_data Current MFEM Jacobian transfer data; borrowed for this call.
 * @return Owned parent true-DOF force and df_dx and energy summed over the parent communicator; other outputs are
 * absent. Empty or inactive pairs contribute zero. An empty list returns allocated zero force and df_dx and zero
 * energy.
 * @throws std::invalid_argument Pair geometry, execution mode, or assembly dimensions are invalid; no result is
 * returned.
 * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded; no result is
 * returned.
 */
template <typename Function>
ContactFormulationResult evaluateAndAssembleEnergyPairs( ExecutionMode execution_mode,
                                                         const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                         MeshData::Viewer mesh2, const Function& function,
                                                         const MfemMeshData& mesh_data,
                                                         const MfemJacobianData& jacobian_data )
{
  static_assert( Function::value_count == 1, "Pair energies must have one scalar output." );
  PairEnergyHessianBuffer<Function::dof_count> contributions( pairs.size(), pairs.getAllocatorID() );
  const auto view = contributions.view();
  evaluatePairValuesJacobiansAndHessians( execution_mode, pairs.view(), mesh1, mesh2, function, view );
  checkPairQuadratureResults( execution_mode, contributions.quadratureResults() );

  MfemPairAssembler assembler( mesh_data, jacobian_data, mesh1, mesh2, pairs.size() );
  assembler.addPairs<Function::dof_count>(
      execution_mode, pairs,
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) { return view.derivatives.gradient( pair, row ); },
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
        return view.derivatives.hessian( pair, row, column );
      } );
  auto result = assembler.assemble();
  result.energy = GlobalScalarAssembler::sum(
      execution_mode, pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) { return view.energies[pair]; },
      result.force->GetComm() );
  return result;
}

#endif  // TRIBOL_USE_ENZYME
}  // namespace detail
}  // namespace tribol
#endif  // BUILD_REDECOMP

#endif  // SRC_TRIBOL_ASSEMBLY_PAIREVALUATIONASSEMBLY_HPP_
