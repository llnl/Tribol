// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Evaluate pair functions and assemble residuals and residual Jacobians on MFEM parent DOFs. */
#ifndef SRC_TRIBOL_ASSEMBLY_ASSEMBLEPAIRS_HPP_
#define SRC_TRIBOL_ASSEMBLY_ASSEMBLEPAIRS_HPP_

#include "tribol/config.hpp"
#include "axom/slic/interface/slic.hpp"
#include "axom/slic/interface/slic_macros.hpp"

#ifdef BUILD_REDECOMP
#include <array>
#include <utility>
#include "tribol/assembly/MfemVectorAssembler.hpp"
#include "tribol/assembly/MfemMatrixAssembler.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/common/GlobalScalarAssembler.hpp"
#include "tribol/assembly/EvaluatePairs.hpp"
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
 * @param mesh1 Current first mesh carrying the scalar fields.
 * @param mesh2 Current second coordinate mesh.
 * @param mesh_data Displacement transfer data.
 * @param submesh_data Scalar-field transfer data.
 * @param jacobian_data Scalar-to-coordinate matrix transfer data.
 * @param read_values Scalar values. @param read_jacobian Coordinate Jacobian entries.
 * @return Independent scalar fields and their coordinate Jacobians. Empty pairs produce allocated zero outputs.
 * @note Reports a SLIC error if assembly layouts disagree.
 * @note Reports a SLIC error if required storage exceeds the supported index range.
 * Scratch remains alive until all kernel operations finish and their matrix blocks have been copied.
 */
template <ExecutionMode Exec, IndexT NumFields, IndexT NumNodes, IndexT SizeCoords, typename ValueReader,
          typename JacobianReader>
std::array<AssembledField, NumFields> assembleNodalValuesAndJacobians(
    const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1, MeshData::Viewer mesh2, const MfemMeshData& mesh_data,
    const MfemSubmeshData& submesh_data, const MfemJacobianData& jacobian_data, ValueReader read_values,
    JacobianReader read_jacobian )
{
  static_assert( NumFields > 0 && SizeCoords % 2 == 0 );
  constexpr IndexT size_geom_coords = SizeCoords / 2;
  std::vector<MfemVectorAssembler> vectors;
  std::vector<MfemMatrixAssembler> matrices;
  vectors.reserve( NumFields );
  matrices.reserve( NumFields );
  for ( IndexT field = 0; field < NumFields; ++field ) {
    vectors.push_back( makeScalarFieldVectorAssembler( submesh_data, mesh1 ) );
    matrices.push_back( makeScalarFieldJacobianAssembler( mesh_data, submesh_data, jacobian_data, mesh1, mesh2,
                                                          pairs.size(), pairs.getAllocatorID() ) );
  }
  MdArray<MfemVectorAssemblyView<1>, NumFields> value_views{};
  MdArray<MfemMatrixAssemblyView<NumNodes, size_geom_coords, 2>, NumFields> matrix_views{};
  for ( IndexT field = 0; field < NumFields; ++field ) {
    if ( vectors[field].numberOfValues() != NumNodes ) {
      SLIC_ERROR( "Nodal vector input extent does not match the scalar row count." );
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
  return [&]<std::size_t... Field>( std::index_sequence<Field...> ) {
    return std::array<AssembledField, NumFields>{
        AssembledField{ vectors[Field].assemble( Exec ), matrices[Field].assemble() }... };
  }( std::make_index_sequence<NumFields>{} );
}

/**
 * @brief Compose residual and residual-Jacobian assemblers in one pair kernel.
 * @tparam SizeCoords Pair DOF count; both elements must contribute half.
 * @tparam ResidualReader Device-callable `(pair, row) -> RealT` residual reader.
 * @tparam JacobianReader Device-callable `(pair, row, column) -> RealT` matrix reader.
 * @tparam Exec Concrete execution mode for numerical assembly.
 * @param pairs Supplied execution-accessible pairs.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param mesh_data Displacement transfer data; must remain valid until the call completes.
 * @param jacobian_data Initialized coordinate-matrix transfer data; must remain valid until the call completes.
 * @param read_residual Pair residuals in element/node, then value-index order.
 * @param read_jacobian Pair Jacobians in that same row and column order.
 * @return Owned residual and df_dx; empty pairs produce allocated zero outputs. Other outputs are absent.
 * Scratch remains alive until the combined kernel and matrix packing finish.
 * @note Reports a SLIC error if layouts are invalid.
 * @note Reports a SLIC error if required storage exceeds the supported index range.
 */
template <ExecutionMode Exec, IndexT SizeCoords, typename ResidualReader, typename JacobianReader>
ContactFormulationResult assembleResidualAndJacobian( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                      MeshData::Viewer mesh2, const MfemMeshData& mesh_data,
                                                      const MfemJacobianData& jacobian_data,
                                                      ResidualReader read_residual, JacobianReader read_jacobian )
{
  static_assert( SizeCoords % 2 == 0 );
  constexpr IndexT size_geom_coords = SizeCoords / 2;
  auto residual = makeResidualVectorAssembler( mesh_data, mesh1, mesh2 );
  auto jacobian =
      makeResidualJacobianAssembler( mesh_data, jacobian_data, mesh1, mesh2, pairs.size(), pairs.getAllocatorID() );
  const auto residual_view = residual.view<Exec, 2>();
  const auto matrix_view = jacobian.view<size_geom_coords, size_geom_coords, 4>();
  const auto pairs_view = pairs.view();
  forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    const auto pair = pairs_view[pair_index];
    residual_view.scatterAddPair( pair_index, pair, read_residual );
    matrix_view.packPair( pair_index, pair, read_jacobian );
  } );
  ContactFormulationResult result;
  result.residual.reset( residual.assemble( Exec ).release() );
  result.df_dx.reset( jacobian.assemble().release() );
  return result;
}

}  // namespace detail

/** @brief Evaluate and assemble a contact residual without calculating derivatives. */
struct ResidualOnly {
  const MfemMeshData& mesh_data;  ///< Displacement transfer data; must outlive this policy.

  /**
   * @brief Evaluate and assemble pair residuals without allocating matrix storage.
   * @tparam Exec Concrete execution mode for evaluation and assembly kernels.
   * @tparam PairFunction Fixed pair function with one residual entry per coordinate.
   * @param pairs Accepted element pairs; not retained.
   * @param mesh1 Current first mesh; not retained.
   * @param mesh2 Current second mesh; not retained.
   * @param pair_function Configured residual function.
   * @return Owned residual; df_dx and other outputs are absent. Empty or inactive pairs produce a zero residual.
   * @note Reports a SLIC error if pair geometry or assembly dimensions are invalid.
   * @note Reports a SLIC error if quadrature capacity or the supported assembly index range is exceeded.
   */
  template <ExecutionMode Exec, PairFunctionType PairFunction>
  ContactFormulationResult computeResidual( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                            MeshData::Viewer mesh2, const PairFunction& pair_function ) const
  {
    static_assert( PairFunction::size_values == PairFunction::size_coords,
                   "Pair residuals must match coordinate DOFs." );
    auto all_pair_values = evaluatePairValues<Exec>( pairs, mesh1, mesh2, pair_function );
    checkPairQuadratureMetadata<Exec>( all_pair_values.quadratureMetadata() );

    auto assembler = detail::makeResidualVectorAssembler( mesh_data, mesh1, mesh2 );
    const auto all_pair_values_view = all_pair_values.view();
    assembler.addPairs<Exec, PairFunction::size_coords, 2>(
        pairs, [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) { return all_pair_values_view.value( pair, row ); } );
    ContactFormulationResult result;
    result.residual.reset( assembler.assemble( Exec ).release() );
    return result;
  }
};

#ifdef TRIBOL_USE_ENZYME
/** @brief Evaluate and assemble a contact residual and its coordinate Jacobian. */
struct ResidualAndJacobian {
  const MfemMeshData& mesh_data;          ///< Displacement transfer data; must outlive this policy.
  const MfemJacobianData& jacobian_data;  ///< Matrix transfer data; must outlive this policy.

  /**
   * @brief Evaluate pair residuals and their Jacobians with Enzyme and assemble both outputs.
   * @tparam Exec Concrete execution mode for evaluation and assembly kernels.
   * @tparam PairFunction Fixed pair function with one residual entry per coordinate.
   * @param pairs Accepted element pairs; not retained.
   * @param mesh1 Current first mesh; not retained.
   * @param mesh2 Current second mesh; not retained.
   * @param pair_function Configured residual function.
   * @return Owned residual and df_dx; other outputs are absent. Empty or inactive pairs produce allocated zero outputs.
   * @note Reports a SLIC error if pair geometry or assembly dimensions are invalid.
   * @note Reports a SLIC error if quadrature capacity or the supported assembly index range is exceeded.
   */
  template <ExecutionMode Exec, PairFunctionType PairFunction>
  ContactFormulationResult computeResidual( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                            MeshData::Viewer mesh2, const PairFunction& pair_function ) const
  {
    static_assert( PairFunction::size_values == PairFunction::size_coords,
                   "Pair residuals must match coordinate DOFs." );
    auto all_pair_values = evaluatePairValuesAndJacobians<Exec>( pairs, mesh1, mesh2, pair_function );
    checkPairQuadratureMetadata<Exec>( all_pair_values.quadratureMetadata() );

    const auto all_pair_values_view = all_pair_values.view();
    return detail::assembleResidualAndJacobian<Exec, PairFunction::size_coords>(
        pairs, mesh1, mesh2, mesh_data, jacobian_data,
        [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) { return all_pair_values_view.value( pair, row ); },
        [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
          return all_pair_values_view.jacobian( pair, row, column );
        } );
  }
};

/**
 * @brief Evaluate scalar energies and assemble their energy, residual, and residual-Jacobian contributions.
 * @tparam Exec Concrete execution mode for evaluation and assembly kernels.
 * @tparam PairFunction Fixed scalar pair-energy function.
 * @param pairs Accepted element pairs; not retained.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param mesh_data Displacement transfer data used for vector and matrix assembly.
 * @param jacobian_data Coordinate-matrix transfer data used for matrix assembly.
 * @param pair_function Configured scalar-energy function.
 * @return Owned energy, residual, and df_dx; other outputs are absent. Empty or inactive pairs contribute zero.
 * @note Reports a SLIC error if pair geometry or assembly dimensions are invalid.
 * @note Reports a SLIC error if quadrature capacity or the supported assembly index range is exceeded.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
ContactFormulationResult assembleEnergyResidualAndJacobian( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                            MeshData::Viewer mesh2, const MfemMeshData& mesh_data,
                                                            const MfemJacobianData& jacobian_data,
                                                            const PairFunction& pair_function )
{
  static_assert( PairFunction::size_values == 1, "Pair energies must have one scalar output." );
  auto all_pair_values = evaluatePairValuesJacobiansAndHessians<Exec>( pairs, mesh1, mesh2, pair_function );
  checkPairQuadratureMetadata<Exec>( all_pair_values.quadratureMetadata() );

  const auto all_pair_values_view = all_pair_values.view();
  auto result = detail::assembleResidualAndJacobian<Exec, PairFunction::size_coords>(
      pairs, mesh1, mesh2, mesh_data, jacobian_data,
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) { return all_pair_values_view.jacobian( pair, row ); },
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
        return all_pair_values_view.hessian( pair, row, column );
      } );
  result.energy = GlobalScalarAssembler::sum<Exec>(
      pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) { return all_pair_values_view.value( pair ); },
      result.residual->GetComm() );
  return result;
}
#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
#endif  // BUILD_REDECOMP

#endif  // SRC_TRIBOL_ASSEMBLY_ASSEMBLEPAIRS_HPP_
