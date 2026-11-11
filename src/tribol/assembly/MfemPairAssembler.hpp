// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Assemble element-pair residuals and Jacobians on MFEM parent true degrees of freedom.
 */
#ifndef SRC_TRIBOL_ASSEMBLY_MFEMPAIRASSEMBLER_HPP_
#define SRC_TRIBOL_ASSEMBLY_MFEMPAIRASSEMBLER_HPP_

#include "tribol/config.hpp"
#ifdef BUILD_REDECOMP
#include <array>
#include <stdexcept>
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Atomics.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/assembly/MfemMatrixAssembler.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/assembly/PairContribution.hpp"

namespace tribol {
/**
 * @brief Accumulate local pair residuals and Jacobians and transfer them to the parent displacement space.
 *
 * Batch operations scatter residuals and pack element blocks in the selected execution space independently of the
 * contact definition or derivative method. Host overloads remain for legacy callers. Mesh connectivity and MFEM
 * transfer data must remain valid until assembly finishes; create a new assembler after mesh re-registration.
 */
class MfemPairAssembler {
 public:
  /**
   * @brief Prepare residual storage and reserve Jacobian storage for the supplied pairs.
   * @param mesh_data Current MFEM displacement mesh and transfer data; not owned.
   * @param jacobian_data Current Jacobian transfer data; not owned.
   * @param mesh1 First mesh with host-accessible connectivity; coordinates are not retained.
   * @param mesh2 Second mesh with host-accessible connectivity; coordinates are not retained.
   * @param pair_capacity Number of pairs expected to contribute, independent of all possible mesh pairs.
   * @param assemble_residual If false, allocate only matrix storage; addPair() and addPairs() are unavailable.
   * @throws std::invalid_argument Dimensions, node counts, or the requested pair capacity are invalid.
   * @throws std::length_error Required array sizes exceed the supported index range.
   */
  MfemPairAssembler( const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data, MeshData::Viewer mesh1,
                     MeshData::Viewer mesh2, IndexT pair_capacity, bool assemble_residual = true )
      : MfemPairAssembler( mesh_data, &jacobian_data, mesh1, mesh2, pair_capacity, assemble_residual )
  {
  }

  /**
   * @brief Prepare pair assembly, optionally omitting all Jacobian storage and transfers.
   * @param mesh_data Borrowed displacement transfer data.
   * @param jacobian_data Borrowed Jacobian transfer data, or nullptr for force-only assembly.
   * @param mesh1 First mesh with host-accessible connectivity; coordinates are not retained.
   * @param mesh2 Second mesh with host-accessible connectivity; coordinates are not retained.
   * @param pair_capacity Expected contributing pairs, used only for Jacobian storage.
   * @param assemble_residual Whether to allocate residual storage.
   * @throws std::invalid_argument Dimensions, node counts, or pair capacity are invalid.
   * @throws std::length_error Required array sizes exceed the supported index range.
   */
  MfemPairAssembler( const MfemMeshData& mesh_data, const MfemJacobianData* jacobian_data, MeshData::Viewer mesh1,
                     MeshData::Viewer mesh2, IndexT pair_capacity, bool assemble_residual = true );

  /**
   * @brief Add one pair's residual and all four Jacobian blocks without retaining the input arrays.
   * @param pair Valid element IDs in the supplied meshes.
   * @param residual Host-accessible entries ordered by element, then node, then spatial component.
   * @param jacobian Host-accessible matrix with residual rows and coordinate columns in that same DOF order.
   * The view uses row-major indexing; no symmetry is assumed.
   * @throws std::invalid_argument Array dimensions do not match the pair DOF count; no contributions are added.
   * @throws std::logic_error Residual or Jacobian assembly was disabled; no contributions are added.
   */
  void addPair( InterfacePair pair, ArrayViewT<const RealT> residual, ArrayViewT<const RealT, 2> jacobian );

  /**
   * @brief Add one pair's four Jacobian blocks without accumulating a residual.
   * @param pair Valid element IDs in the supplied meshes.
   * @param jacobian Host-accessible matrix with residual rows and coordinate columns in pair DOF order.
   * @throws std::invalid_argument Matrix dimensions do not match the pair DOF count; no contributions are added.
   * @throws std::logic_error Jacobian assembly was disabled; no contributions are added.
   */
  void addPairJacobian( InterfacePair pair, ArrayViewT<const RealT, 2> jacobian );

  /**
   * @brief Scatter residual values without allocating or packing matrix blocks.
   * @tparam DofCount Pair residual count; each linear edge contributes half.
   * @tparam ResidualReader Device-callable `(pair, row) -> RealT` reader.
   * @param execution_mode Host or device execution policy.
   * @param pairs Pair list in execution-accessible memory.
   * @param residual_reader Pair residual reader in element/node/component order.
   * @throws std::invalid_argument Pair DOFs do not match the configured layout; no contributions are added.
   * @throws std::logic_error Residual assembly was disabled; no contributions are added.
   */
  template <IndexT DofCount, typename ResidualReader>
  void addPairResiduals( ExecutionMode execution_mode, const ArrayT<InterfacePair>& pairs,
                         ResidualReader residual_reader )
  {
    static_assert( DofCount % 2 == 0, "Pair residual assembly requires equal element DOF counts." );
    constexpr IndexT element_dofs = DofCount / 2;
    if ( dof_offsets_[1] != element_dofs || dof_offsets_[2] != DofCount ) {
      throw std::invalid_argument( "Pair residual DOFs must match the MFEM element layouts." );
    }
    if ( !redecomp_residual_.FESpace() ) {
      throw std::logic_error( "Residual assembly was not enabled for this pair assembler." );
    }
    const bool use_device = isOnDevice( execution_mode );
    redecomp_residual_.UseDevice( use_device );
    auto* residual = redecomp_residual_.ReadWrite( use_device );
    const int scalar_size = redecomp_residual_.FESpace()->GetNDofs();
    const auto connectivity0 = connectivity_[0];
    const auto connectivity1 = connectivity_[1];
    const int dimension = dimension_;
    const auto input = pairs.view();
    forAllExec( execution_mode, pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
      const auto pair = input[pair_index];
      for ( IndexT node = 0; node < connectivity0.shape()[1]; ++node ) {
        for ( int component = 0; component < dimension; ++component ) {
          atomicAdd( residual + connectivity0( pair.element_id1_, node ) + component * scalar_size,
                     residual_reader( pair_index, node * dimension + component ) );
        }
      }
      for ( IndexT node = 0; node < connectivity1.shape()[1]; ++node ) {
        for ( int component = 0; component < dimension; ++component ) {
          atomicAdd( residual + connectivity1( pair.element_id2_, node ) + component * scalar_size,
                     residual_reader( pair_index, element_dofs + node * dimension + component ) );
        }
      }
    } );
  }

  /**
   * @brief Scatter and pack a complete pair batch in the selected execution space.
   *
   * The numerical stage performs no allocation and no MFEM sparse-matrix calls inside the pair kernel. Four dense
   * element blocks per pair are packed into preallocated execution-space buffers, then copied in bulk into the
   * Redecomp transfer representation. Residuals and Jacobians accumulate across batch and host calls; an empty batch
   * leaves all earlier contributions unchanged.
   *
   * @tparam DofCount Pair coordinate and residual count; both linear edges must contribute half.
   * @tparam ResidualReader Device-callable `(pair, row) -> RealT` reader.
   * @tparam JacobianReader Device-callable `(pair, row, column) -> RealT` reader.
   * @param execution_mode Host or device execution policy.
   * @param pairs Pair list in execution-accessible memory.
   * @param residual_reader Pair residual reader.
   * @param jacobian_reader Pair Jacobian reader.
   * @throws std::invalid_argument Pair DOFs do not match the configured layout; no contributions are added.
   * @throws std::logic_error Residual or Jacobian assembly was disabled; no contributions are added.
   */
  template <IndexT DofCount, typename ResidualReader, typename JacobianReader>
  void addPairs( ExecutionMode execution_mode, const ArrayT<InterfacePair>& pairs, ResidualReader residual_reader,
                 JacobianReader jacobian_reader )
  {
    if ( !redecomp_residual_.FESpace() ) {
      throw std::logic_error( "Residual assembly was not enabled for this pair assembler." );
    }
    addPairsImpl<DofCount, true>( execution_mode, pairs, residual_reader, jacobian_reader );
  }

  /**
   * @brief Append all four coordinate Jacobian blocks without allocating or scattering a residual.
   * @tparam DofCount Pair coordinate count; each element contributes half.
   * @tparam JacobianReader Device-callable `(pair, row, column) -> RealT` reader.
   * @param execution_mode Host or device execution policy.
   * @param pairs Pair list in execution-accessible memory.
   * @param jacobian_reader Pair Jacobian reader.
   * @throws std::invalid_argument Pair DOFs do not match the configured layout; no contributions are added.
   */
  template <IndexT DofCount, typename JacobianReader>
  void addPairJacobians( ExecutionMode execution_mode, const ArrayT<InterfacePair>& pairs,
                         JacobianReader jacobian_reader )
  {
    addPairsImpl<DofCount, false>(
        execution_mode, pairs, [] TRIBOL_HOST_DEVICE( IndexT, IndexT ) { return 0.0; }, jacobian_reader );
  }

  /**
   * @brief Return assembled residual and Jacobian on parent true DOFs.
   * @return Owned force and, when enabled, df_dx; other outputs are absent. No pairs produce allocated zero outputs.
   * Without Jacobian transfer data df_dx is absent.
   * Results remain valid after this assembler is destroyed, while their MFEM spaces and communicator remain alive.
   * @throws std::logic_error Residual assembly was disabled; use assembleJacobian() instead.
   */
  ContactFormulationResult assemble() const;

  /**
   * @brief Return the assembled Jacobian on parent displacement true DOFs.
   * @return Owned parent true-DOF sparse matrix; zero pairs produce an allocated zero matrix.
   * @throws std::logic_error Jacobian assembly was disabled.
   */
  shared::ParSparseMat assembleJacobian() const;

  /**
   * @brief Pack matrix blocks and optionally scatter residual entries in one kernel.
   * NVCC requires the enclosing function of an extended device lambda to be public, so this implementation helper
   * is visible even though callers use addPairs() or addPairJacobians().
   * @tparam DofCount Pair coordinate count.
   * @tparam WithResidual Whether to scatter residuals.
   * @tparam ResidualReader Pair residual reader.
   * @tparam JacobianReader Pair Jacobian reader.
   * @param execution_mode Host or device execution policy.
   * @param pairs Pair list in execution-accessible memory.
   * @param residual_reader Pair residual reader, unused when WithResidual is false.
   * @param jacobian_reader Pair Jacobian reader.
   */
  template <IndexT DofCount, bool WithResidual, typename ResidualReader, typename JacobianReader>
  void addPairsImpl( ExecutionMode execution_mode, const ArrayT<InterfacePair>& pairs, ResidualReader residual_reader,
                     JacobianReader jacobian_reader )
  {
    if ( !matrix_ ) {
      throw std::logic_error( "Jacobian assembly was not enabled for this pair assembler." );
    }
    static_assert( DofCount % 2 == 0, "Pair block assembly requires equal element DOF counts." );
    constexpr IndexT element_dofs = DofCount / 2;
    if ( dof_offsets_[1] != element_dofs || dof_offsets_[2] != DofCount ) {
      throw std::invalid_argument( "Pair buffer DOFs must match the MFEM element layouts." );
    }
    const IndexT pair_count = pairs.size();
    auto batch = matrix_->makeBatch<element_dofs, element_dofs, 4>( pair_count, pairs.getAllocatorID() );
    const auto matrix = batch.view();
    const auto pairs_view = pairs.view();
    const auto connectivity0 = connectivity_[0];
    const auto connectivity1 = connectivity_[1];
    const IndexT nodes0 = connectivity0.shape()[1];
    const IndexT nodes1 = connectivity1.shape()[1];
    const int dimension = dimension_;
    int scalar_size = 0;
    RealT* residual = nullptr;
    if constexpr ( WithResidual ) {
      scalar_size = redecomp_residual_.FESpace()->GetNDofs();
      const bool use_device = isOnDevice( execution_mode );
      redecomp_residual_.UseDevice( use_device );
      residual = redecomp_residual_.ReadWrite( use_device );
    }
    forAllExec( execution_mode, pair_count, [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
      const auto pair = pairs_view[pair_index];
      matrix.packPair( pair_index, pair, jacobian_reader );
      if constexpr ( WithResidual ) {
        for ( IndexT node = 0; node < nodes0; ++node ) {
          for ( int component = 0; component < dimension; ++component ) {
            const IndexT row = node * dimension + component;
            atomicAdd( residual + connectivity0( pair.element_id1_, node ) + component * scalar_size,
                       residual_reader( pair_index, row ) );
          }
        }
        for ( IndexT node = 0; node < nodes1; ++node ) {
          for ( int component = 0; component < dimension; ++component ) {
            const IndexT row = element_dofs + node * dimension + component;
            atomicAdd( residual + connectivity1( pair.element_id2_, node ) + component * scalar_size,
                       residual_reader( pair_index, row ) );
          }
        }
      }
    } );
    matrix_->addBatch( batch );
  }

 private:
  const MfemMeshData& mesh_data_;                          ///< Borrowed displacement mesh and field transfers.
  std::array<Array2DView<const IndexT>, 2> connectivity_;  ///< Borrowed connectivity for each mesh.
  int dimension_;                                          ///< Spatial components per node.
  MdArray<IndexT, 3> dof_offsets_{};             ///< Pair DOF offsets before, between, and after the two elements.
  mfem::GridFunction redecomp_residual_;         ///< Accumulated residual in the Redecomp displacement space.
  std::unique_ptr<MfemMatrixAssembler> matrix_;  ///< Matrix component, absent for force-only assembly.
};

}  // namespace tribol
#endif  // BUILD_REDECOMP
#endif  // SRC_TRIBOL_ASSEMBLY_MFEMPAIRASSEMBLER_HPP_
