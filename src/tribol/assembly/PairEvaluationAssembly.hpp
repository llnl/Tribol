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
  shared::ParVector values;       ///< Values on scalar submesh true DOFs, without normalization or activation.
  shared::ParSparseMat jacobian;  ///< Scalar submesh rows and parent coordinate columns.
};
namespace detail {

/**
 * @brief Compose scalar nodal vectors and coordinate Jacobians in one pair kernel.
 * @tparam NumFields Independent scalar fields. @tparam NumNodes Scalar rows per pair.
 * @tparam SizeCoords Pair coordinates; both geometries contribute half.
 * @tparam ValueReader Device-callable `(pair, field, node) -> RealT` reader.
 * @tparam JacobianReader Device-callable `(pair, field, node, coordinate) -> RealT` reader.
 * @tparam Exec Concrete execution mode for numerical assembly.
 * @param pairs Supplied execution-accessible pairs.
 * @param vectors Vector assemblers, one per scalar field on geometry1.
 * @param matrices Matrix assemblers, one per field, with scalar rows and both geometries' coordinate columns.
 * @param read_values Scalar values. @param read_jacobian Coordinate Jacobian entries.
 * @throws std::invalid_argument Numbers of field assemblers or their layouts disagree; no contributions are added.
 * @throws std::length_error Required storage exceeds the supported index range.
 * Scratch remains alive until all kernel operations finish and their matrix blocks have been copied.
 */
template <ExecutionMode Exec, IndexT NumFields, IndexT NumNodes, IndexT SizeCoords, typename ValueReader,
          typename JacobianReader>
void addNodalValuesAndJacobians( const ArrayT<InterfacePair>& pairs, std::vector<MfemVectorAssembler>& vectors,
                                 std::vector<MfemMatrixAssembler>& matrices, ValueReader read_values,
                                 JacobianReader read_jacobian )
{
  static_assert( NumFields > 0 && SizeCoords % 2 == 0 );
  constexpr IndexT size_geom_coords = SizeCoords / 2;
  if ( vectors.size() != static_cast<std::size_t>( NumFields ) || matrices.size() != vectors.size() ) {
    throw std::invalid_argument( "Numbers of nodal vector and matrix assemblers must agree." );
  }
  MdArray<MfemVectorAssemblyView<1>, NumFields> value_views{};
  MdArray<MfemMatrixAssemblyView<NumNodes, size_geom_coords, 2>, NumFields> matrix_views{};
  for ( IndexT field = 0; field < NumFields; ++field ) {
    if ( vectors[field].numberOfValues() != NumNodes ) {
      throw std::invalid_argument( "Nodal vector input extent does not match the scalar row count." );
    }
    value_views[field] = vectors[field].view<Exec, 1>();
    matrix_views[field] = matrices[field].view<NumNodes, size_geom_coords, 2>();
  }
  const auto pairs_view = pairs.view();
  forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
    for ( IndexT field = 0; field < NumFields; ++field ) {
      value_views[field].scatterAddPair( pair, pairs_view[pair], read_values, field );
      matrix_views[field].packPair( pair, pairs_view[pair], read_jacobian, field );
    }
  } );
}

/**
 * @brief Compose independent force and matrix assemblers in one pair kernel.
 * @tparam SizeCoords Pair DOF count; both elements must contribute half.
 * @tparam ForceReader Device-callable `(pair, row) -> RealT` force reader.
 * @tparam JacobianReader Device-callable `(pair, row, column) -> RealT` matrix reader.
 * @tparam Exec Concrete execution mode for numerical assembly.
 * @param pairs Supplied execution-accessible pairs.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param mesh_data Displacement transfer data; must remain valid until the call completes.
 * @param jacobian_data Initialized coordinate-matrix transfer data; must remain valid until the call completes.
 * @param read_force Pair forces in element/node, then value-index order.
 * @param read_jacobian Pair Jacobians in that same row and column order.
 * @return Owned force and df_dx; empty pairs produce allocated zero outputs. Other outputs are absent.
 * Scratch remains alive until the combined kernel and matrix packing finish.
 * @throws std::invalid_argument Layouts are invalid; no result is returned.
 * @throws std::length_error Required storage exceeds the supported index range; no result is returned.
 */
template <ExecutionMode Exec, IndexT SizeCoords, typename ForceReader, typename JacobianReader>
ContactFormulationResult assemblePairForcesAndJacobian( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                        MeshData::Viewer mesh2, const MfemMeshData& mesh_data,
                                                        const MfemJacobianData& jacobian_data, ForceReader read_force,
                                                        JacobianReader read_jacobian )
{
  static_assert( SizeCoords % 2 == 0 );
  constexpr IndexT size_geom_coords = SizeCoords / 2;
  auto forces = makeDisplacementVectorAssembler( mesh_data, mesh1, mesh2 );
  auto jacobian =
      makeCoordinateMatrixAssembler( mesh_data, jacobian_data, mesh1, mesh2, pairs.size(), pairs.getAllocatorID() );
  const auto force_view = forces.view<Exec, 2>();
  const auto matrix_view = jacobian.view<size_geom_coords, size_geom_coords, 4>();
  const auto pairs_view = pairs.view();
  forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs_view[pair_index];
    force_view.scatterAddPair( pair_index, pair, read_force );
    matrix_view.packPair( pair_index, pair, read_jacobian );
  } );
  ContactFormulationResult result;
  result.force.reset( forces.assemble().release() );
  result.df_dx.reset( jacobian.assemble().release() );
  return result;
}

}  // namespace detail

/**
 * @brief Evaluate pair residuals and assemble forces without calculating derivatives or allocating matrix storage.
 * @tparam Exec Concrete execution mode for evaluation and assembly kernels.
 * @tparam PairFunction Fixed pair function with one residual entry per coordinate.
 * @param pairs Accepted element pairs; not retained.
 * @param mesh1 Current first mesh; not retained.
 * @param mesh2 Current second mesh; not retained.
 * @param mesh_data Displacement transfer data used for assembly.
 * @param pair_function Configured residual function.
 * @return Owned force; df_dx and other outputs are absent. Empty or inactive pairs produce zero force.
 * @throws std::invalid_argument Pair geometry or assembly dimensions are invalid.
 * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
ContactFormulationResult evaluateAndAssemblePairResidual( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                          MeshData::Viewer mesh2, const MfemMeshData& mesh_data,
                                                          const PairFunction& pair_function )
{
  static_assert( PairFunction::size_values == PairFunction::size_coords, "Pair residuals must match coordinate DOFs." );
  auto all_pair_values = evaluatePairValues<Exec>( pairs, mesh1, mesh2, pair_function );
  checkPairQuadratureResults<Exec>( all_pair_values.quadratureResults() );

  auto assembler = detail::makeDisplacementVectorAssembler( mesh_data, mesh1, mesh2 );
  const auto all_pair_values_view = all_pair_values.view();
  assembler.addPairs<Exec, PairFunction::size_coords, 2>( pairs, [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) {
    return all_pair_values_view.has_contributing_points[pair] ? all_pair_values_view.value( pair, row ) : 0.0;
  } );
  ContactFormulationResult result;
  result.force.reset( assembler.assemble().release() );
  return result;
}

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Evaluate pair residuals and coordinate Jacobians with Enzyme and assemble both outputs.
 * @tparam Exec Concrete execution mode for evaluation and assembly kernels.
 * @tparam PairFunction Fixed pair function with one residual entry per coordinate.
 * @param pairs Accepted element pairs; not retained.
 * @param mesh1 Current first mesh; not retained.
 * @param mesh2 Current second mesh; not retained.
 * @param mesh_data Displacement transfer data used for vector and matrix assembly.
 * @param jacobian_data Coordinate-matrix transfer data used for matrix assembly.
 * @param pair_function Configured residual function.
 * @return Owned force and df_dx; other outputs are absent. Empty or inactive pairs produce allocated zero outputs.
 * @throws std::invalid_argument Pair geometry or assembly dimensions are invalid.
 * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
ContactFormulationResult evaluateAndAssemblePairResidualJacobian( const ArrayT<InterfacePair>& pairs,
                                                                  MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                                  const MfemMeshData& mesh_data,
                                                                  const MfemJacobianData& jacobian_data,
                                                                  const PairFunction& pair_function )
{
  static_assert( PairFunction::size_values == PairFunction::size_coords, "Pair residuals must match coordinate DOFs." );
  auto all_pair_values = evaluatePairValuesAndJacobians<Exec>( pairs, mesh1, mesh2, pair_function );
  checkPairQuadratureResults<Exec>( all_pair_values.quadratureResults() );

  const auto all_pair_values_view = all_pair_values.view();
  return detail::assemblePairForcesAndJacobian<Exec, PairFunction::size_coords>(
      pairs, mesh1, mesh2, mesh_data, jacobian_data,
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) {
        return all_pair_values_view.has_contributing_points[pair] ? all_pair_values_view.value( pair, row ) : 0.0;
      },
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
        return all_pair_values_view.has_contributing_points[pair] ? all_pair_values_view.jacobian( pair, row, column )
                                                                  : 0.0;
      } );
}
#endif  // TRIBOL_USE_ENZYME

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Evaluate scalar energies and assemble their coordinate Jacobians as forces, without Hessians.
 * @tparam Exec Concrete execution mode for evaluation and assembly kernels.
 * @tparam PairFunction Fixed scalar pair-energy function.
 * @param pairs Accepted element pairs; not retained.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param mesh_data Displacement transfer data used for assembly.
 * @param pair_function Configured scalar-energy function.
 * @return Owned force and globally summed energy; df_dx and other outputs are absent. Empty or inactive pairs
 * contribute zero.
 * @throws std::invalid_argument Pair geometry or assembly dimensions are invalid.
 * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
ContactFormulationResult evaluateAndAssemblePairEnergy( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                        MeshData::Viewer mesh2, const MfemMeshData& mesh_data,
                                                        const PairFunction& pair_function )
{
  static_assert( PairFunction::size_values == 1, "Pair energies must have one scalar output." );
  auto all_pair_values = evaluatePairValuesAndJacobians<Exec>( pairs, mesh1, mesh2, pair_function );
  checkPairQuadratureResults<Exec>( all_pair_values.quadratureResults() );

  auto assembler = detail::makeDisplacementVectorAssembler( mesh_data, mesh1, mesh2 );
  const auto all_pair_values_view = all_pair_values.view();
  assembler.addPairs<Exec, PairFunction::size_coords, 2>( pairs, [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) {
    return all_pair_values_view.has_contributing_points[pair] ? all_pair_values_view.jacobian( pair, 0, row ) : 0.0;
  } );
  ContactFormulationResult result;
  result.force.reset( assembler.assemble().release() );
  result.energy = GlobalScalarAssembler::sum<Exec>(
      pairs.size(),
      [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
        return all_pair_values_view.has_contributing_points[pair] ? all_pair_values_view.value( pair, 0 ) : 0.0;
      },
      result.force->GetComm() );
  return result;
}

/**
 * @brief Evaluate scalar energies, Jacobians, and Hessians and assemble energy, force, and force Jacobians.
 * @tparam Exec Concrete execution mode for evaluation and assembly kernels.
 * @tparam PairFunction Fixed scalar pair-energy function.
 * @param pairs Accepted element pairs; not retained.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param mesh_data Displacement transfer data used for vector and matrix assembly.
 * @param jacobian_data Coordinate-matrix transfer data used for matrix assembly.
 * @param pair_function Configured scalar-energy function.
 * @return Owned energy, force, and df_dx; other outputs are absent. Empty or inactive pairs contribute zero.
 * @throws std::invalid_argument Pair geometry or assembly dimensions are invalid.
 * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
ContactFormulationResult evaluateAndAssemblePairEnergyJacobian( const ArrayT<InterfacePair>& pairs,
                                                                MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                                const MfemMeshData& mesh_data,
                                                                const MfemJacobianData& jacobian_data,
                                                                const PairFunction& pair_function )
{
  static_assert( PairFunction::size_values == 1, "Pair energies must have one scalar output." );
  auto all_pair_values = evaluatePairValuesJacobiansAndHessians<Exec>( pairs, mesh1, mesh2, pair_function );
  checkPairQuadratureResults<Exec>( all_pair_values.quadratureResults() );

  const auto all_pair_values_view = all_pair_values.view();
  auto result = detail::assemblePairForcesAndJacobian<Exec, PairFunction::size_coords>(
      pairs, mesh1, mesh2, mesh_data, jacobian_data,
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) {
        return all_pair_values_view.derivatives.jacobian( pair, row );
      },
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
        return all_pair_values_view.derivatives.hessian( pair, row, column );
      } );
  result.energy = GlobalScalarAssembler::sum<Exec>(
      pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) { return all_pair_values_view.values[pair]; },
      result.force->GetComm() );
  return result;
}
#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
#endif  // BUILD_REDECOMP

#endif  // SRC_TRIBOL_ASSEMBLY_PAIREVALUATIONASSEMBLY_HPP_
