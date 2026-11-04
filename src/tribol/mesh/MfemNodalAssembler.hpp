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
 * This host-only assembler owns scratch storage for one calculation and copies each pair's contributions. Mesh
 * connectivity and MFEM transfer data must remain valid until assembly finishes; create a new assembler after mesh
 * re-registration.
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

  /**
   * @brief Add one pair's node-weighted gap and area coordinate Hessians to the parent displacement Jacobian.
   * @param pair Valid element IDs in the supplied meshes.
   * @param gap_weights Scalar weights for each node of mesh1's element.
   * @param gap_hessians Row-major array [mesh1_node][pair_row][pair_col] in pair DOF order.
   * @param area_weights Optional scalar weights for each node of mesh1's element; pass empty view to skip area terms.
   * @param area_hessians Optional row-major array [mesh1_node][pair_row][pair_col] in pair DOF order.
   * @throws std::invalid_argument Array dimensions do not match mesh1's node count or the pair DOF count.
   */
  void addPairHessians( InterfacePair pair, ArrayViewT<const RealT> gap_weights,
                        ArrayViewT<const RealT, 3> gap_hessians, ArrayViewT<const RealT> area_weights = {},
                        ArrayViewT<const RealT, 3> area_hessians = {} );

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
  ArrayT<RealT, 2, MemorySpace::Host> pair_jacobian_;  ///< Reused pair matrix for weighted nodal Hessians.
  MfemPairAssembler pair_assembler_;                   ///< Assembles weighted pair Hessians on parent true DOFs.
};

}  // namespace tribol
#endif  // BUILD_REDECOMP
#endif  // SRC_TRIBOL_MESH_MFEMNODALASSEMBLER_HPP_
