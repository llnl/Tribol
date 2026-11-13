// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Accumulate nodal pair contributions into an MFEM vector with an explicit transfer destination. */
#ifndef SRC_TRIBOL_ASSEMBLY_MFEMVECTORASSEMBLER_HPP_
#define SRC_TRIBOL_ASSEMBLY_MFEMVECTORASSEMBLER_HPP_
#include "tribol/config.hpp"
#ifdef BUILD_REDECOMP
#include <functional>
#include <stdexcept>
#include <vector>
#include "tribol/common/Atomics.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MfemData.hpp"

namespace tribol {

/** @brief Configure FE-space references and the operation transferring Redecomp entries to final local DOFs. */
struct MfemVectorSpace {
  const mfem::ParFiniteElementSpace& final_fes;  ///< Destination FE space, before true-DOF assembly.
  const mfem::FiniteElementSpace& redecomp_fes;  ///< Space containing accumulated pair entries.
  std::function<void( const mfem::GridFunction&, mfem::Vector& )>
      transfer;  ///< Host transfer, called once per assemble().
};

/** @brief Configure entries contributed by one geometry in each pair. */
struct MfemVectorBlock {
  IndexT geometry;                         ///< 0 for geometry1, 1 for geometry2.
  Array2DView<const IndexT> connectivity;  ///< View of scalar nodal DOF connectivity for that geometry.
  IndexT offset;                           ///< First input entry, in element/node, then value-index order.
};

/**
 * @brief Device-copyable view for scalar or vector nodal accumulation.
 * @tparam NumBlocks Configured geometry blocks per pair.
 */
template <IndexT NumBlocks>
struct MfemVectorAssemblyView {
  MdArray<MfemVectorBlock, NumBlocks> blocks;  ///< Geometry selection and nodal connectivity views.
  RealT* values;                               ///< Writable Redecomp vector.
  int num_values_per_node;                     ///< Entries per node in the vector's FE space.
  IndexT node_stride;                          ///< Redecomp-vector stride between scalar nodal DOFs.
  IndexT value_stride;                         ///< Redecomp-vector stride between values of one node.

  /**
   * @brief Scatter-add one pair's entries at shared nodal DOFs.
   * @tparam ValueReader Device-callable `(pair, indices..., row) -> RealT` reader.
   * @tparam Indices Optional field indices preceding the input row.
   * @param pair_index Pair slot passed to the reader.
   * @param pair Geometry1 and geometry2 element IDs.
   * @param read_value Input reader in node, then value-index order.
   * @param indices Optional field indices for composing multiple vector assemblers in one kernel.
   */
  template <typename ValueReader, typename... Indices>
  TRIBOL_HOST_DEVICE void scatterAddPair( IndexT pair_index, InterfacePair pair, ValueReader read_value,
                                          Indices... indices ) const
  {
    const MdArray<IndexT, 2> elements{ pair.element_id1_, pair.element_id2_ };
    for ( IndexT block = 0; block < NumBlocks; ++block ) {
      const auto layout = blocks[block];
      for ( IndexT node = 0; node < layout.connectivity.shape()[1]; ++node ) {
        const IndexT dof = layout.connectivity( elements[layout.geometry], node );
        for ( int value_index = 0; value_index < num_values_per_node; ++value_index ) {
          atomicAdd( values + dof * node_stride + value_index * value_stride,
                     read_value( pair_index, indices..., layout.offset + node * num_values_per_node + value_index ) );
        }
      }
    }
  }
};

/**
 * @brief Accumulate one scalar or vector nodal field without matrix or derivative data.
 *
 * Geometry blocks specify which pair elements contribute and how their input entries are ordered. The configured
 * FE space supplies the number of values per node and the Redecomp vector ordering. The host transfer callback selects
 * the
 * destination space; it is never invoked inside a pair kernel. Use multiple assemblers for independent fields.
 * FE spaces, transfer prerequisites, and connectivity must remain valid for the assembler's lifetime. Recreate after
 * decomposition or mesh registration.
 */
class MfemVectorAssembler {
 public:
  /**
   * @brief Allocate an initially zero vector in the configured Redecomp FE space.
   * @param space FE-space references and a nonempty host transfer callback, retained by value.
   * @param blocks Nonempty geometry blocks with nonnegative input offsets and positive nodes per element.
   * @throws std::invalid_argument Spaces, callback, geometry selectors, offsets, or node counts are invalid.
   * @throws std::length_error Input extents exceed the supported index range.
   */
  MfemVectorAssembler( MfemVectorSpace space, std::vector<MfemVectorBlock> blocks );

  /** @brief Return the required input extent. @return Scalar entry count, including configured block offsets. */
  IndexT numberOfValues() const { return size_values_; }

  /**
   * @brief Return a writable view for composing pair kernels with other vector or matrix views.
   * @tparam NumBlocks Configured geometry blocks per pair.
   * @tparam Exec Concrete execution mode using the view.
   * @return View into this vector and the supplied connectivity. Keep both alive until the kernel finishes;
   * do not request another memory space while a kernel is using the view.
   * @throws std::invalid_argument NumBlocks does not match the configuration.
   */
  template <ExecutionMode Exec, IndexT NumBlocks>
  MfemVectorAssemblyView<NumBlocks> view()
  {
    static_assert( NumBlocks > 0 );
    if ( blocks_.size() != static_cast<std::size_t>( NumBlocks ) ) {
      throw std::invalid_argument( "Vector view block count does not match its configuration." );
    }
    const bool use_device = isOnDevice( Exec );
    redecomp_values_.UseDevice( use_device );
    const int num_values_per_node = space_.redecomp_fes.GetVDim();
    const bool by_nodes = space_.redecomp_fes.GetOrdering() == mfem::Ordering::byNODES;
    MfemVectorAssemblyView<NumBlocks> result{ {},
                                              redecomp_values_.ReadWrite( use_device ),
                                              num_values_per_node,
                                              by_nodes ? 1 : num_values_per_node,
                                              by_nodes ? space_.redecomp_fes.GetNDofs() : 1 };
    for ( IndexT block = 0; block < NumBlocks; ++block ) {
      result.blocks[block] = blocks_[block];
    }
    return result;
  }

  /**
   * @brief Accumulate a batch of pair entries in the selected execution space.
   * @tparam SizeValues Input scalar entry count. @tparam NumBlocks Geometry blocks per pair.
   * @tparam ValueReader Device-callable `(pair, row) -> RealT` reader.
   * @tparam Exec Concrete execution mode.
   * @param pairs Supplied execution-accessible pairs.
   * @param read_value Input reader in node, then value-index order.
   * @throws std::invalid_argument Template counts do not match the configuration; nothing is added.
   */
  template <ExecutionMode Exec, IndexT SizeValues, IndexT NumBlocks, typename ValueReader>
  void addPairs( const ArrayT<InterfacePair>& pairs, ValueReader read_value )
  {
    if ( SizeValues != size_values_ ) {
      throw std::invalid_argument( "Vector input count does not match its configured blocks." );
    }
    const auto vector_view = view<Exec, NumBlocks>();
    const auto pairs_view = pairs.view();
    forAllExec<Exec>( pairs.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
      vector_view.scatterAddPair( pair, pairs_view[pair], read_value );
    } );
  }

  /**
   * @brief Transfer accumulated entries to final true DOFs.
   * @return Owned vector; no contributions produce an allocated zero vector. Results remain valid after this
   * assembler is destroyed while the associated FE spaces and communicator remain alive.
   */
  shared::ParVector assemble() const;

 private:
  MfemVectorSpace space_;                ///< FE-space references and the host transfer operation.
  std::vector<MfemVectorBlock> blocks_;  ///< Geometry layout with nodal connectivity views.
  IndexT size_values_ = 0;               ///< Input extent including block offsets.
  mfem::GridFunction redecomp_values_;   ///< Owned accumulated dual entries.
};

namespace detail {
/** @brief Configure displacement entries from both geometries.
 * @param data Displacement data for the current decomposition; must outlive the returned assembler.
 * @param mesh1 First geometry. @param mesh2 Second geometry.
 * @return Vector assembler referencing the two connectivities and parent transfer data.
 * @throws std::invalid_argument Geometry dimensions do not match the displacement space.
 * @throws std::length_error Input extents exceed the supported index range. */
MfemVectorAssembler makeDisplacementVectorAssembler( const MfemMeshData& data, MeshData::Viewer mesh1,
                                                     MeshData::Viewer mesh2 );

/** @brief Configure scalar nodal entries from geometry1.
 * @param data Scalar field data for the current decomposition; must outlive the returned assembler.
 * @param mesh1 Geometry1 connectivity indexing the scalar Redecomp DOFs.
 * @return Vector assembler referencing scalar submesh transfers and geometry1 connectivity.
 * @throws std::invalid_argument The configured FE spaces are not scalar, or the node count is invalid.
 * @throws std::length_error Input extent exceeds the supported index range. */
MfemVectorAssembler makeNodalVectorAssembler( const MfemSubmeshData& data, MeshData::Viewer mesh1 );
}  // namespace detail
}  // namespace tribol
#endif  // BUILD_REDECOMP
#endif  // SRC_TRIBOL_ASSEMBLY_MFEMVECTORASSEMBLER_HPP_
