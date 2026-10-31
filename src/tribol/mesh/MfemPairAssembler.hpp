// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Assemble element-pair residuals and Jacobians on MFEM parent true degrees of freedom.
 */
#ifndef SRC_TRIBOL_MESH_MFEMPAIRASSEMBLER_HPP_
#define SRC_TRIBOL_MESH_MFEMPAIRASSEMBLER_HPP_

#include "tribol/config.hpp"
#ifdef BUILD_REDECOMP
#include <array>
#include "tribol/common/MdArray.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/physics/ContactFormulation.hpp"

namespace tribol {

/**
 * @brief Accumulate local pair residuals and Jacobians and transfer them to the parent displacement space.
 *
 * This host-only assembler handles numerical arrays independently of the contact definition or derivative method.
 * It owns scratch arrays for one calculation and copies each pair's contributions. Mesh connectivity and MFEM
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
   * @throws std::invalid_argument Dimensions, node counts, or the requested pair capacity are invalid.
   * @throws std::length_error Required array sizes exceed the supported index range.
   */
  MfemPairAssembler( const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data, MeshData::Viewer mesh1,
                     MeshData::Viewer mesh2, IndexT pair_capacity );

  /**
   * @brief Add one pair's residual and all four Jacobian blocks without retaining the input arrays.
   * @param pair Valid element IDs in the supplied meshes.
   * @param residual Host-accessible entries ordered by element, then node, then spatial component.
   * @param jacobian Host-accessible matrix with residual rows and coordinate columns in that same DOF order.
   * The view uses row-major indexing; no symmetry is assumed.
   * @throws std::invalid_argument Array dimensions do not match the pair DOF count; no contributions are added.
   */
  void addPair( InterfacePair pair, ArrayViewT<const RealT> residual, ArrayViewT<const RealT, 2> jacobian );

  /**
   * @brief Return assembled residual and Jacobian on parent true DOFs.
   * @return Owned force and df_dx; other outputs are absent. No pairs produce allocated zero outputs.
   * Results remain valid after this assembler is destroyed, while their MFEM spaces and communicator remain alive.
   */
  ContactFormulationResult assemble() const;

 private:
  const MfemMeshData& mesh_data_;                          ///< Borrowed displacement mesh and field transfers.
  const MfemJacobianData& jacobian_data_;                  ///< Borrowed displacement Jacobian transfers.
  std::array<Array2DView<const IndexT>, 2> connectivity_;  ///< Borrowed connectivity for each mesh.
  int dimension_;                                          ///< Spatial components per node.
  MdArray<IndexT, 3> dof_offsets_{};      ///< Pair DOF offsets before, between, and after the two elements.
  mfem::GridFunction redecomp_residual_;  ///< Accumulated residual in the Redecomp displacement space.
  std::vector<PackedPairJacobianContribs> jacobian_contributions_;  ///< Four packed element Jacobian blocks.
  Array1D<RealT, MemorySpace::Host> block_;  ///< Reused column-major storage for one element Jacobian block.
};

}  // namespace tribol
#endif  // BUILD_REDECOMP
#endif  // SRC_TRIBOL_MESH_MFEMPAIRASSEMBLER_HPP_
