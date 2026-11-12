// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Evaluate pair functions and assemble their forces and coordinate Jacobians on MFEM parent DOFs. */
#ifndef SRC_TRIBOL_ASSEMBLY_PAIREVALUATIONASSEMBLY_HPP_
#define SRC_TRIBOL_ASSEMBLY_PAIREVALUATIONASSEMBLY_HPP_

#include "tribol/config.hpp"

#ifdef BUILD_REDECOMP
#include "tribol/assembly/MfemVectorAssembler.hpp"
#include "tribol/assembly/MfemMatrixAssembler.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/common/GlobalScalarAssembler.hpp"
#include "tribol/assembly/PairContribution.hpp"
#include "tribol/assembly/PairKernel.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {
/** @brief Own scalar nodal values and their coordinate Jacobian after composing vector and matrix assembly. */
struct AssembledField {
  shared::ParVector values;         ///< Values on scalar submesh true DOFs, without normalization or activation.
  shared::ParSparseMat dvalues_dx;  ///< Scalar submesh rows and parent coordinate columns.
};
/** @brief Configure residual-force calculation without derivatives or Jacobian transfer data. */
struct ResidualForceEvaluation {
  const MfemMeshData& mesh_data;  ///< Displacement transfer data referenced by the evaluation.
};

#ifdef TRIBOL_USE_ENZYME
/** @brief Configure residual-force and coordinate-Jacobian calculation with Enzyme. */
struct ResidualForceJacobianEvaluation {
  const MfemMeshData& mesh_data;          ///< Displacement transfer data referenced by the evaluation.
  const MfemJacobianData& jacobian_data;  ///< Coordinate-matrix transfer data referenced by the evaluation.
};
#endif

namespace detail {

/**
 * @brief Compose scalar nodal vectors and coordinate Jacobians in one pair kernel.
 * @tparam NumFields Independent scalar fields. @tparam NumNodes Scalar rows per pair.
 * @tparam NumCoords Pair coordinates; both geometries contribute half.
 * @tparam ValueReader Device-callable `(pair, field, node) -> RealT` reader.
 * @tparam JacobianReader Device-callable `(pair, field, node, coordinate) -> RealT` reader.
 * @tparam Exec Concrete execution mode for numerical assembly.
 * @param pairs Supplied execution-accessible pairs.
 * @param vectors Scalar-vector components, each configured for geometry1.
 * @param matrices Matrix components with scalar rows and both geometries' coordinate columns.
 * @param read_values Scalar values. @param read_jacobian Coordinate Jacobian entries.
 * @throws std::invalid_argument Component counts or layouts disagree; no contributions are added.
 * @throws std::length_error Required storage exceeds the supported index range.
 * Scratch remains alive until all kernel operations finish and their matrix blocks have been copied.
 */
template <ExecutionMode Exec, IndexT NumFields, IndexT NumNodes, IndexT NumCoords, typename ValueReader,
          typename JacobianReader>
void addNodalValuesAndJacobians( const ArrayT<InterfacePair>& pairs, std::vector<MfemVectorAssembler>& vectors,
                                 std::vector<MfemMatrixAssembler>& matrices, ValueReader read_values,
                                 JacobianReader read_jacobian )
{
  static_assert( NumFields > 0 && NumCoords % 2 == 0 );
  constexpr IndexT element_dofs = NumCoords / 2;
  if ( vectors.size() != static_cast<std::size_t>( NumFields ) || matrices.size() != vectors.size() ) {
    throw std::invalid_argument( "Nodal vector and matrix component counts must agree." );
  }
  std::vector<MfemMatrixBatch<NumNodes, element_dofs, 2>> batches;
  batches.reserve( NumFields );
  MdArray<MfemVectorAssemblyView<1>, NumFields> value_views{};
  MdArray<MfemMatrixBatchView<NumNodes, element_dofs, 2>, NumFields> matrix_views{};
  for ( IndexT field = 0; field < NumFields; ++field ) {
    if ( vectors[field].numberOfValues() != NumNodes ) {
      throw std::invalid_argument( "Nodal vector input extent does not match the scalar row count." );
    }
    value_views[field] = vectors[field].view<Exec, 1>();
    batches.push_back( matrices[field].makeBatch<NumNodes, element_dofs, 2>( pairs.size(), pairs.getAllocatorID() ) );
    matrix_views[field] = batches.back().view();
  }
  const auto pairs_view = pairs.view();
  forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
    for ( IndexT field = 0; field < NumFields; ++field ) {
      value_views[field].addPair( pair, pairs_view[pair], read_values, field );
      matrix_views[field].packPair( pair, pairs_view[pair], read_jacobian, field );
    }
  } );
  for ( IndexT field = 0; field < NumFields; ++field ) {
    matrices[field].addBatch( batches[field] );
  }
}

/**
 * @brief Compose independent force and matrix assemblers in one pair kernel.
 * @tparam NumCoords Pair DOF count; both elements must contribute half.
 * @tparam ForceReader Device-callable `(pair, row) -> RealT` force reader.
 * @tparam JacobianReader Device-callable `(pair, row, column) -> RealT` matrix reader.
 * @tparam Exec Concrete execution mode for numerical assembly.
 * @param pairs Supplied execution-accessible pairs.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param mesh_data Displacement transfer data; must remain valid until the call completes.
 * @param jacobian_data Initialized coordinate-matrix transfer data; must remain valid until the call completes.
 * @param read_force Pair forces in element/node/component order.
 * @param read_jacobian Pair Jacobians in that same row and column order.
 * @return Owned force and df_dx; empty pairs produce allocated zero outputs. Other outputs are absent.
 * Scratch remains alive until the combined kernel and matrix packing finish.
 * @throws std::invalid_argument Layouts are invalid; no result is returned.
 * @throws std::length_error Required storage exceeds the supported index range; no result is returned.
 */
template <ExecutionMode Exec, IndexT NumCoords, typename ForceReader, typename JacobianReader>
ContactFormulationResult assemblePairForcesAndJacobian( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                        MeshData::Viewer mesh2, const MfemMeshData& mesh_data,
                                                        const MfemJacobianData& jacobian_data, ForceReader read_force,
                                                        JacobianReader read_jacobian )
{
  static_assert( NumCoords % 2 == 0 );
  constexpr IndexT element_dofs = NumCoords / 2;
  auto forces = makeDisplacementVectorAssembler( mesh_data, mesh1, mesh2 );
  auto jacobian = makeCoordinateMatrixAssembler( mesh_data, jacobian_data, mesh1, mesh2, pairs.size() );
  auto batch = jacobian.makeBatch<element_dofs, element_dofs, 4>( pairs.size(), pairs.getAllocatorID() );
  const auto force_view = forces.view<Exec, 2>();
  const auto matrix_view = batch.view();
  const auto pairs_view = pairs.view();
  forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs_view[pair_index];
    force_view.addPair( pair_index, pair, read_force );
    matrix_view.packPair( pair_index, pair, read_jacobian );
  } );
  jacobian.addBatch( batch );
  ContactFormulationResult result;
  result.force.reset( forces.assemble().release() );
  result.df_dx.reset( jacobian.assemble().release() );
  return result;
}

/**
 * @brief Evaluate pair residual values and assemble forces without derivatives or matrix storage.
 * @tparam Function Function with one residual entry per coordinate in element/node/component order.
 * @tparam Exec Concrete execution mode for numerical assembly.
 * @param pairs Accepted element pairs in execution-accessible memory.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param function Fixed pair residual configuration.
 * @param evaluation Displacement transfer references for force-only calculation; referenced data must remain valid
 * until the call completes.
 * @return Owned force on parent true DOFs; df_dx and other outputs are absent. Empty or inactive pairs contribute zero.
 * Scratch arrays remain alive until evaluation and assembly finish.
 * @throws std::invalid_argument Geometry or assembly dimensions are invalid; no result is returned.
 * @throws std::length_error Quadrature capacity or the supported index range is exceeded; no result is returned.
 */
template <ExecutionMode Exec, PairFunctionType Function>
ContactFormulationResult evaluateAndAssembleResidualPairs( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                           MeshData::Viewer mesh2, const Function& function,
                                                           ResidualForceEvaluation evaluation )
{
  static_assert( Function::num_values == Function::num_coords, "Pair residuals must match coordinate DOFs." );
  auto contributions = evaluatePairValues<Exec>( pairs, mesh1, mesh2, function );
  const auto contributions_view = contributions.view();
  checkPairQuadratureResults<Exec>( contributions.quadratureResults() );
  auto assembler = makeDisplacementVectorAssembler( evaluation.mesh_data, mesh1, mesh2 );
  assembler.addPairs<Exec, Function::num_coords, 2>( pairs, [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) {
    return contributions_view.has_active_points[pair] ? contributions_view.value( pair, row ) : 0.0;
  } );
  ContactFormulationResult result;
  result.force.reset( assembler.assemble().release() );
  return result;
}

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Evaluate pair residuals and assemble their values as forces and their Jacobians as df_dx.
 *
 * Evaluation and assembly use separate kernels. Scratch arrays are allocated for the supplied pair list and remain
 * alive until all operations finish. Quadrature failures are reported before any contributions are assembled.
 *
 * @tparam Function Pair function with one residual entry per coordinate in element/node/component order.
 * @tparam Exec Concrete execution mode for evaluation and numerical assembly.
 * @param pairs Accepted element pairs in execution-accessible memory.
 * @param mesh1 Current first mesh with execution-accessible coordinates and connectivity.
 * @param mesh2 Current second mesh with execution-accessible coordinates and connectivity.
 * @param function Pair residual configuration; not retained after this call.
 * @param evaluation Displacement and Jacobian transfer references; referenced data must remain valid until the call
 * completes.
 * @return Owned parent true-DOF force and df_dx; other outputs are absent. Empty or inactive pairs contribute zero.
 * An empty list returns allocated zero force and df_dx.
 * @throws std::invalid_argument Pair geometry or assembly dimensions are invalid; no result is
 * returned.
 * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded; no result is
 * returned.
 */
template <ExecutionMode Exec, typename Function>
ContactFormulationResult evaluateAndAssembleResidualPairs( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                           MeshData::Viewer mesh2, const Function& function,
                                                           ResidualForceJacobianEvaluation evaluation )
{
  static_assert( Function::num_values == Function::num_coords, "Pair residuals must match coordinate DOFs." );
  auto contributions = evaluatePairValuesAndJacobians<Exec>( pairs, mesh1, mesh2, function );
  const auto contributions_view = contributions.view();
  checkPairQuadratureResults<Exec>( contributions.quadratureResults() );

  return assemblePairForcesAndJacobian<Exec, Function::num_coords>(
      pairs, mesh1, mesh2, evaluation.mesh_data, evaluation.jacobian_data,
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) {
        return contributions_view.has_active_points[pair] ? contributions_view.value( pair, row ) : 0.0;
      },
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
        return contributions_view.has_active_points[pair] ? contributions_view.jacobian( pair, row, column ) : 0.0;
      } );
}

/**
 * @brief Evaluate scalar pair energies and assemble their Jacobians as forces and their Hessians as df_dx.
 *
 * The energy kernel reuses the Jacobian computed with each Hessian as the force contribution. Evaluation and assembly
 * use separate kernels; scratch arrays remain alive until assembly and the global energy sum finish. Quadrature
 * failures are reported before any contributions are assembled.
 *
 * @tparam Function Scalar energy function with coordinates in element/node/component order.
 * @tparam Exec Concrete execution mode for evaluation and numerical assembly.
 * @param pairs Accepted element pairs in execution-accessible memory.
 * @param mesh1 Current first mesh with execution-accessible coordinates and connectivity.
 * @param mesh2 Current second mesh with execution-accessible coordinates and connectivity.
 * @param function Pair energy configuration; not retained after this call.
 * @param mesh_data Current MFEM displacement transfer data; must remain valid until the call completes.
 * @param jacobian_data Current MFEM Jacobian transfer data; must remain valid until the call completes.
 * @return Owned parent true-DOF force and df_dx and energy summed over the parent communicator; other outputs are
 * absent. Empty or inactive pairs contribute zero. An empty list returns allocated zero force and df_dx and zero
 * energy.
 * @throws std::invalid_argument Pair geometry or assembly dimensions are invalid; no result is
 * returned.
 * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded; no result is
 * returned.
 */
template <ExecutionMode Exec, typename Function>
ContactFormulationResult evaluateAndAssembleEnergyPairs( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                         MeshData::Viewer mesh2, const Function& function,
                                                         const MfemMeshData& mesh_data,
                                                         const MfemJacobianData& jacobian_data )
{
  static_assert( Function::num_values == 1, "Pair energies must have one scalar output." );
  auto contributions = evaluatePairValuesJacobiansAndHessians<Exec>( pairs, mesh1, mesh2, function );
  const auto contributions_view = contributions.view();
  checkPairQuadratureResults<Exec>( contributions.quadratureResults() );

  auto result = assemblePairForcesAndJacobian<Exec, Function::num_coords>(
      pairs, mesh1, mesh2, mesh_data, jacobian_data,
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) {
        return contributions_view.derivatives.jacobian( pair, row );
      },
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
        return contributions_view.derivatives.hessian( pair, row, column );
      } );
  result.energy = GlobalScalarAssembler::sum<Exec>(
      pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) { return contributions_view.values[pair]; },
      result.force->GetComm() );
  return result;
}

#endif  // TRIBOL_USE_ENZYME
}  // namespace detail
}  // namespace tribol
#endif  // BUILD_REDECOMP

#endif  // SRC_TRIBOL_ASSEMBLY_PAIREVALUATIONASSEMBLY_HPP_
