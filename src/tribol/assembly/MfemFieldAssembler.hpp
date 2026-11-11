// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Assemble scalar nodal fields and their coordinate Jacobians on MFEM spaces. */
#ifndef SRC_TRIBOL_ASSEMBLY_MFEMFIELDASSEMBLER_HPP_
#define SRC_TRIBOL_ASSEMBLY_MFEMFIELDASSEMBLER_HPP_
#include "tribol/config.hpp"
#ifdef BUILD_REDECOMP
#include <array>
#include <vector>
#include "tribol/common/Atomics.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/assembly/MfemElementAssembly.hpp"

namespace tribol {

/** @brief Own a scalar field on submesh true DOFs and its Jacobian with parent coordinate columns. */
struct AssembledField {
  shared::ParVector values;         ///< Accumulated scalar values, without normalization or activation.
  shared::ParSparseMat dvalues_dx;  ///< Submesh true-DOF rows and parent displacement true-DOF columns.
};

/**
 * @brief Accumulate any number of scalar fields sharing one nodal submesh finite-element space.
 *
 * Pair values are attached to mesh1's element nodes. Coordinate derivatives include both elements in each pair.
 * Mesh1 connectivity must index the scalar Redecomp DOFs; mesh2 connectivity indexes coordinate nodes. All fields
 * share the scalar space supplied by MfemSubmeshData. Vector fields and mixed field spaces are unsupported.
 * This class performs no quadrature, normalization, contact activation, or coordinate-Hessian assembly.
 * Mesh and transfer data are borrowed and must remain valid until assembly finishes; recreate after re-registration.
 */
class MfemFieldAssembler {
 public:
  /**
   * @brief Allocate scalar fields and reserve their two coordinate-gradient blocks per pair.
   * @param mesh_data Parent displacement space and transfers.
   * @param submesh_data Shared scalar field space and transfers, already updated for the current decomposition.
   * @param jacobian_data Submesh-to-parent matrix transfers.
   * @param mesh1 First mesh carrying field rows.
   * @param mesh2 Second mesh contributing coordinate columns.
   * @param field_count Positive number of independent scalar fields.
   * @param pair_capacity Expected pair contributions; allocation is independent of all possible mesh pairs.
   * @throws std::invalid_argument Invalid field count, capacity, dimensions, node counts, or nonscalar field space.
   * @throws std::length_error Required storage exceeds the supported index range; no assembler is returned.
   */
  MfemFieldAssembler( const MfemMeshData& mesh_data, const MfemSubmeshData& submesh_data,
                      const MfemJacobianData& jacobian_data, MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                      IndexT field_count, IndexT pair_capacity );

  /**
   * @brief Add all scalar fields and coordinate gradients for a single pair from host memory.
   * @param pair Valid element IDs in the two meshes.
   * @param values Values in [field][mesh1 node] order.
   * @param gradients Derivatives in [field][mesh1 node][pair coordinate] order; coordinates use element/node/component.
   * @throws std::invalid_argument Input shapes do not match the configured fields and nodes; nothing is added.
   */
  void addPair( InterfacePair pair, ArrayViewT<const RealT, 2> values, ArrayViewT<const RealT, 3> gradients );

  /**
   * @brief Scatter all scalar fields and pack all gradient blocks in one execution-space kernel.
   * Contributions accumulate across host and batch calls; empty batches preserve earlier contributions.
   * @tparam FieldCount Number of independent scalar fields.
   * @tparam NodeCount Field nodes on mesh1's element.
   * @tparam DofCount Total pair coordinates; both elements must contribute half.
   * @tparam ValueReader Device-callable `(pair, field, node) -> RealT` reader.
   * @tparam GradientReader Device-callable `(pair, field, node, coordinate) -> RealT` reader.
   * @param execution_mode Host or device execution policy.
   * @param pairs Pair list in execution-accessible memory.
   * @param value_reader Scalar value reader.
   * @param gradient_reader Coordinate-gradient reader.
   * @throws std::invalid_argument Template counts do not match the configured layouts; nothing is added.
   */
  template <IndexT FieldCount, IndexT NodeCount, IndexT DofCount, typename ValueReader, typename GradientReader>
  void addPairs( ExecutionMode execution_mode, const ArrayT<InterfacePair>& pairs, ValueReader value_reader,
                 GradientReader gradient_reader )
  {
    static_assert( FieldCount > 0 && DofCount % 2 == 0 );
    constexpr IndexT element_dofs = DofCount / 2;
    if ( fields_.size() != static_cast<std::size_t>( FieldCount ) || nodes1_ != NodeCount ||
         dof_offsets_[1] != element_dofs || dof_offsets_[2] != DofCount ) {
      throw std::invalid_argument( "Scalar field batch layout must match the configured fields and element DOFs." );
    }
    const IndexT pair_count = pairs.size();
    std::vector<ElementBlockBuffer<NodeCount, element_dofs>> blocks;
    blocks.reserve( 2 * FieldCount );
    MdArray<ElementBlockView<NodeCount, element_dofs>, FieldCount, 2> block_views{};
    MdArray<RealT*, FieldCount> values{};
    const bool use_device = isOnDevice( execution_mode );
    for ( IndexT field = 0; field < FieldCount; ++field ) {
      fields_[field].UseDevice( use_device );
      values[field] = fields_[field].ReadWrite( use_device );
      for ( IndexT side = 0; side < 2; ++side ) {
        blocks.emplace_back( pair_count, pairs.getAllocatorID() );
        block_views( field, side ) = blocks.back().view();
      }
    }
    const auto pairs_view = pairs.view();
    const auto connectivity0 = connectivity_[0];
    const IndexT nodes2 = connectivity_[1].shape()[1];
    const int dimension = dimension_;
    forAllExec( execution_mode, pair_count, [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
      const auto pair = pairs_view[pair_index];
      for ( IndexT field = 0; field < FieldCount; ++field ) {
        const auto block0 = block_views( field, 0 ), block1 = block_views( field, 1 );
        block0.setElements( pair_index, pair.element_id1_, pair.element_id1_ );
        block1.setElements( pair_index, pair.element_id1_, pair.element_id2_ );
        for ( IndexT node = 0; node < NodeCount; ++node ) {
          atomicAdd( values[field] + connectivity0( pair.element_id1_, node ),
                     value_reader( pair_index, field, node ) );
        }
        for ( IndexT column = 0; column < element_dofs; ++column ) {
          const IndexT coordinate0 = ( column % NodeCount ) * dimension + column / NodeCount;
          const IndexT coordinate1 = element_dofs + ( column % nodes2 ) * dimension + column / nodes2;
          for ( IndexT node = 0; node < NodeCount; ++node ) {
            block0.setValue( pair_index, node, column, gradient_reader( pair_index, field, node, coordinate0 ) );
            block1.setValue( pair_index, node, column, gradient_reader( pair_index, field, node, coordinate1 ) );
          }
        }
      }
    } );
    for ( IndexT field = 0; field < FieldCount; ++field ) {
      for ( IndexT side = 0; side < 2; ++side ) {
        detail::appendElementBlocks( gradients_[field][side], blocks[2 * field + side] );
      }
    }
  }

  /**
   * @brief Transfer all accumulated scalar fields and coordinate gradients to MFEM true DOFs.
   * @return Owned results in field order; no pairs produce allocated zero vectors and matrices.
   * Results remain valid after this assembler is destroyed while the associated FE spaces and communicator live.
   */
  std::vector<AssembledField> assemble() const;

 private:
  const MfemMeshData& mesh_data_;                          ///< Borrowed displacement spaces and transfers.
  const MfemSubmeshData& submesh_data_;                    ///< Borrowed scalar field spaces and transfers.
  const MfemJacobianData& jacobian_data_;                  ///< Borrowed matrix transfers.
  std::array<Array2DView<const IndexT>, 2> connectivity_;  ///< Borrowed pair connectivity.
  int dimension_;                                          ///< Coordinate components per node.
  IndexT nodes1_;                                          ///< Field nodes per mesh1 element.
  MdArray<IndexT, 3> dof_offsets_{};        ///< Pair coordinate offsets before, between, and after elements.
  std::vector<mfem::GridFunction> fields_;  ///< Accumulated scalar fields on Redecomp DOFs.
  std::vector<std::vector<PackedPairJacobianContribs>> gradients_;  ///< Two coordinate blocks per field.
  Array1D<RealT, MemorySpace::Host> gradient_block_;  ///< Reused host storage for one element gradient block.
};
}  // namespace tribol
#endif  // BUILD_REDECOMP
#endif  // SRC_TRIBOL_ASSEMBLY_MFEMFIELDASSEMBLER_HPP_
