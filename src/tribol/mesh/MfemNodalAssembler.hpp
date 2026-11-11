// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Assemble nodal mortar integrals, dual-to-displacement Jacobians, and weighted pair Hessians on MFEM spaces.
 */
#ifndef SRC_TRIBOL_MESH_MFEMNODALASSEMBLER_HPP_
#define SRC_TRIBOL_MESH_MFEMNODALASSEMBLER_HPP_

#include "tribol/config.hpp"
#ifdef BUILD_REDECOMP
#include <array>
#include <vector>
#include "tribol/common/MdArray.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/mesh/MfemPairAssembler.hpp"

namespace tribol {

/** @brief Assembled nodal integrals and their first coordinate derivatives on MFEM true degrees of freedom. */
struct MfemNodalQuantityData {
  shared::ParVector weighted_gap;              ///< Weighted nodal gap (g_tilde) on dual submesh true DOFs.
  shared::ParVector area;                      ///< Nodal tributary area (A) on dual submesh true DOFs.
  shared::ParVector normalized_gap;            ///< Area-normalized nodal gap (g_tilde / A) on dual submesh true DOFs.
  shared::ParSparseMat weighted_gap_jacobian;  ///< Derivative d(g_tilde)/dx on dual rows and parent columns.
  shared::ParSparseMat area_jacobian;          ///< Derivative dA/dx on dual rows and parent columns.
};

/**
 * @brief Accumulate nodal gap/area integrals, their gradients, and weighted pair Hessians across MFEM transfer spaces.
 *
 * Mesh1 is the non-mortar integration side whose nodes carry scalar dual degrees of freedom; mesh2 is the mortar side.
 * Batch operations scatter nodal fields and pack gradient blocks in the selected execution space. Host overloads
 * remain for legacy callers. Mesh connectivity and MFEM transfer data must remain valid until assembly finishes;
 * create a new assembler after mesh re-registration.
 */
class MfemNodalAssembler {
 public:
  /**
   * @brief Prepare Redecomp nodal fields and reserve gradient and Hessian block storage for the supplied pairs.
   * @param mesh_data Current MFEM displacement mesh and transfer data; not owned.
   * @param submesh_data Current MFEM dual submesh and transfer data; not owned.
   * @param jacobian_data Current Jacobian transfer data; not owned.
   * @param mesh1 First (non-mortar) mesh with host-accessible connectivity; coordinates are not retained.
   * @param mesh2 Second (mortar) mesh with host-accessible connectivity; coordinates are not retained.
   * @param pair_capacity Number of pairs expected to contribute, independent of all possible mesh pairs.
   * @throws std::invalid_argument Dimensions, node counts, dual vector dimension, or pair capacity are invalid.
   * @throws std::length_error Required array sizes exceed the supported index range.
   */
  MfemNodalAssembler( const MfemMeshData& mesh_data, const MfemSubmeshData& submesh_data,
                      const MfemJacobianData& jacobian_data, MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                      IndexT pair_capacity );

  /**
   * @brief Add one pair's nodal weighted gaps, tributary areas, and their coordinate gradients.
   * @param pair Valid element IDs in the supplied meshes.
   * @param weighted_gaps Host-accessible nodal gap integrals for mesh1's element nodes.
   * @param areas Host-accessible nodal tributary areas for mesh1's element nodes.
   * @param gap_gradients Row-major matrix [mesh1_node][pair_dof] in element/node/component pair DOF order.
   * @param area_gradients Row-major matrix [mesh1_node][pair_dof] in element/node/component pair DOF order.
   * @throws std::invalid_argument Array dimensions do not match mesh1's node count or the pair DOF count.
   */
  void addPairQuantities( InterfacePair pair, ArrayViewT<const RealT> weighted_gaps, ArrayViewT<const RealT> areas,
                          ArrayViewT<const RealT, 2> gap_gradients, ArrayViewT<const RealT, 2> area_gradients );

  /**
   * @brief Scatter nodal values and pack their coordinate gradients for a complete pair batch.
   * Values and gradients accumulate across batch and host calls; an empty batch preserves earlier contributions.
   * @tparam NodeCount Number of integration-side nodes per pair.
   * @tparam DofCount Pair coordinate count.
   * @tparam ValueReader Device-callable `(pair, value) -> RealT` reader; gaps precede areas.
   * @tparam GradientReader Device-callable `(pair, value, coordinate) -> RealT` reader.
   * @param execution_mode Host or device execution policy.
   * @param pairs Current pair list.
   * @param value_reader Pair nodal-value reader.
   * @param gradient_reader Pair nodal-gradient reader.
   */
  template <IndexT NodeCount, IndexT DofCount, typename ValueReader, typename GradientReader>
  void addPairQuantities( ExecutionMode execution_mode, const ArrayT<InterfacePair>& pairs, ValueReader value_reader,
                          GradientReader gradient_reader )
  {
    static_assert( DofCount % 2 == 0, "Nodal pair assembly requires equal element displacement DOF counts." );
    constexpr IndexT element_dofs = DofCount / 2;
    if ( nodes1_ != NodeCount || dof_offsets_[1] != element_dofs || dof_offsets_[2] != DofCount ) {
      throw std::invalid_argument( "Nodal pair buffer layout must match the MFEM element layouts." );
    }
    const IndexT pair_count = pairs.size();
    const int allocator_id = pairs.getAllocatorID();
    ElementBlockBuffer<NodeCount, element_dofs> gap0( pair_count, allocator_id );
    ElementBlockBuffer<NodeCount, element_dofs> gap1( pair_count, allocator_id );
    ElementBlockBuffer<NodeCount, element_dofs> area0( pair_count, allocator_id );
    ElementBlockBuffer<NodeCount, element_dofs> area1( pair_count, allocator_id );
    const bool use_device = isOnDevice( execution_mode );
    redecomp_gap_.UseDevice( use_device );
    redecomp_area_.UseDevice( use_device );
    auto* gaps = redecomp_gap_.ReadWrite( use_device );
    auto* areas = redecomp_area_.ReadWrite( use_device );
    const auto pairs_view = pairs.view();
    const auto connectivity0 = connectivity_[0];
    const auto connectivity1 = connectivity_[1];
    const IndexT nodes0 = connectivity0.shape()[1];
    const IndexT nodes1 = connectivity1.shape()[1];
    const int dimension = dimension_;
    const auto gap_output0 = gap0.view();
    const auto gap_output1 = gap1.view();
    const auto area_output0 = area0.view();
    const auto area_output1 = area1.view();
    forAllExec( execution_mode, pair_count, [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
      const auto pair = pairs_view[pair_index];
      gap_output0.setElements( pair_index, pair.element_id1_, pair.element_id1_ );
      gap_output1.setElements( pair_index, pair.element_id1_, pair.element_id2_ );
      area_output0.setElements( pair_index, pair.element_id1_, pair.element_id1_ );
      area_output1.setElements( pair_index, pair.element_id1_, pair.element_id2_ );
      for ( IndexT node = 0; node < NodeCount; ++node ) {
        const IndexT redecomp_dof = connectivity0( pair.element_id1_, node );
        atomicAdd( gaps + redecomp_dof, value_reader( pair_index, node ) );
        atomicAdd( areas + redecomp_dof, value_reader( pair_index, NodeCount + node ) );
      }
      for ( IndexT column = 0; column < element_dofs; ++column ) {
        const IndexT coordinate0 = ( column % nodes0 ) * dimension + column / nodes0;
        const IndexT coordinate1 = element_dofs + ( column % nodes1 ) * dimension + column / nodes1;
        for ( IndexT node = 0; node < NodeCount; ++node ) {
          gap_output0.setValue( pair_index, node, column, gradient_reader( pair_index, node, coordinate0 ) );
          gap_output1.setValue( pair_index, node, column, gradient_reader( pair_index, node, coordinate1 ) );
          area_output0.setValue( pair_index, node, column,
                                 gradient_reader( pair_index, NodeCount + node, coordinate0 ) );
          area_output1.setValue( pair_index, node, column,
                                 gradient_reader( pair_index, NodeCount + node, coordinate1 ) );
        }
      }
    } );
    detail::appendElementBlocks( gap_gradient_contributions_[0], gap0 );
    detail::appendElementBlocks( gap_gradient_contributions_[1], gap1 );
    detail::appendElementBlocks( area_gradient_contributions_[0], area0 );
    detail::appendElementBlocks( area_gradient_contributions_[1], area1 );
  }

  /**
   * @brief Transfer accumulated nodal integrals and gradients to dual submesh and parent displacement true DOFs.
   * @param clamp_positive_gaps If true, zero positive entries of weighted_gap and eliminate their Jacobian rows.
   * @param area_tolerance Minimum denominator magnitude used when normalizing weighted_gap by area.
   * @return Owned dual true-DOF vectors and (dual, parent) true-DOF Jacobian matrices.
   */
  MfemNodalQuantityData assembleQuantities( bool clamp_positive_gaps, RealT area_tolerance = 1.0e-14 ) const;

  /**
   * @brief Read the caller-supplied dual field (Lagrange multipliers) on dual submesh true DOFs.
   * @return Owned true-DOF vector extracted from the submesh pressure grid function.
   */
  shared::ParVector getSubmeshPressureTrueDofs() const;

  /**
   * @brief Transfer a dual submesh true-DOF vector to the Redecomp scalar finite-element space.
   * @param submesh_true_dofs Dual vector on submesh true DOFs.
   * @return Redecomp grid function with values populated on every local contact element node.
   * @throws std::invalid_argument Vector length does not match the submesh true-DOF count.
   */
  mfem::GridFunction transferToRedecomp( const mfem::Vector& submesh_true_dofs ) const;

  /**
   * @brief Construct a distributed diagonal matrix on the dual submesh true-DOF space.
   * @param diagonal_true_dofs Diagonal entries on submesh true DOFs.
   * @return Owned diagonal sparse matrix matching the dual submesh row/column partitioning.
   * @throws std::invalid_argument Vector length does not match the submesh true-DOF count.
   */
  shared::ParSparseMat makeSubmeshDiagonalMatrix( const mfem::Vector& diagonal_true_dofs ) const;

  /** @brief Add a pair's already weighted coordinate Hessian. @param pair Element IDs.
   * @param jacobian Pair Hessian in row-major coordinate order; no symmetry is assumed. */
  void addPairJacobian( InterfacePair pair, ArrayViewT<const RealT, 2> jacobian )
  {
    pair_assembler_.addPairJacobian( pair, jacobian );
  }

  /**
   * @brief Pack a complete batch of weighted pair Hessians in the selected execution space.
   * @tparam DofCount Pair coordinate count.
   * @tparam JacobianReader Device-callable `(pair, row, column) -> RealT` reader.
   * @param execution_mode Host or device execution policy.
   * @param pairs Current pair list.
   * @param jacobian_reader Weighted Hessian reader.
   */
  template <IndexT DofCount, typename JacobianReader>
  void addPairJacobians( ExecutionMode execution_mode, const ArrayT<InterfacePair>& pairs,
                         JacobianReader jacobian_reader )
  {
    pair_assembler_.addPairs<DofCount>(
        execution_mode, pairs, [] TRIBOL_HOST_DEVICE( IndexT, IndexT ) { return 0.0; }, jacobian_reader );
  }

  /**
   * @brief Return the assembled weighted-Hessian contribution on parent displacement true DOFs.
   * @return Owned parent true-DOF sparse matrix; zero pairs produce an allocated zero matrix.
   */
  shared::ParSparseMat assembleHessians() const;

 private:
  /**
   * @brief Pack one nodal gradient matrix into column-major element blocks and append to the two pair contributions.
   * @param pair Element IDs in mesh1 and mesh2.
   * @param gradients Row-major matrix [mesh1_node][pair_dof].
   * @param[in,out] contributions Two-element vector holding (mesh1, mesh1) and (mesh1, mesh2) blocks.
   */
  void appendGradientBlocks( InterfacePair pair, ArrayViewT<const RealT, 2> gradients,
                             std::vector<PackedPairJacobianContribs>& contributions );

  /**
   * @brief Transfer a Redecomp scalar field to dual submesh true DOFs.
   * @param redecomp_field Scalar field accumulated on the Redecomp space.
   * @return Owned submesh true-DOF vector.
   */
  shared::ParVector assembleSubmeshVector( const mfem::GridFunction& redecomp_field ) const;

  const MfemMeshData& mesh_data_;                          ///< Borrowed displacement mesh and field transfers.
  const MfemSubmeshData& submesh_data_;                    ///< Borrowed dual submesh and field transfers.
  const MfemJacobianData& jacobian_data_;                  ///< Borrowed Jacobian transfers.
  std::array<Array2DView<const IndexT>, 2> connectivity_;  ///< Borrowed connectivity for each mesh.
  int dimension_;                                          ///< Spatial components per displacement node.
  IndexT nodes1_;                                          ///< Nodes per element on mesh1.
  MdArray<IndexT, 3> dof_offsets_{};                       ///< Pair displacement DOF offsets.
  mfem::GridFunction redecomp_gap_;                        ///< Accumulated weighted gap on the Redecomp dual space.
  mfem::GridFunction redecomp_area_;                       ///< Accumulated tributary area on the Redecomp dual space.
  std::vector<PackedPairJacobianContribs> gap_gradient_contributions_;   ///< Packed (mesh1, mesh1/2) gap gradients.
  std::vector<PackedPairJacobianContribs> area_gradient_contributions_;  ///< Packed (mesh1, mesh1/2) area gradients.
  Array1D<RealT, MemorySpace::Host> gradient_block_;                     ///< Reused column-major gradient block.
  MfemPairAssembler pair_assembler_;  ///< Assembles weighted pair Hessians on parent true DOFs.
};

}  // namespace tribol
#endif  // BUILD_REDECOMP
#endif  // SRC_TRIBOL_MESH_MFEMNODALASSEMBLER_HPP_
